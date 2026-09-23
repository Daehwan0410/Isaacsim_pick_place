#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import tf2_ros
try:
    from tf2_geometry_msgs import do_transform_point
except ImportError:
    from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

import cv2
from ultralytics import YOLO


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # ---- 파라미터 (실제 토픽/프레임에 맞게 조정) ----
        self.declare_parameter('color_topic', '/rsd455/rgb')
        self.declare_parameter('depth_topic', '/rsd455/depth')
        self.declare_parameter('camera_info_topic', '/rsd455/camera_info')
        self.declare_parameter('base_frame', 'panda_link0')
        self.declare_parameter('model_path', 'yolov8n-seg.pt')  # seg 가중치 필요(회전각 추정용), 커스텀 학습 권장
        self.declare_parameter('target_class', -1)           # -1: 전체, 특정 class id 지정 가능
        self.declare_parameter('confidence', 0.4)
        self.declare_parameter('use_color_fallback', False)  # True면 HSV 색 검출 사용
        # 색 검출용 HSV 범위 (예: 빨간 큐브). use_color_fallback=True일 때만 사용
        self.declare_parameter('hsv_low', [0, 120, 70])
        self.declare_parameter('hsv_high', [10, 255, 255])
        # 뎁스 카메라는 물체 윗면까지의 거리만 측정하므로,
        # 큐브 중심으로 보정하기 위한 높이(한 변 길이, m)
        self.declare_parameter('cube_height', 0.04)
        # 이미지 평면 회전각(minAreaRect) -> base_frame yaw 변환용 보정값.
        # 카메라 마운트 방향에 따라 부호가 반대이거나 오프셋이 필요할 수 있어
        # 실측 후 조정한다 (기본은 부호 반전 없음, 오프셋 0).
        self.declare_parameter('camera_yaw_sign', 1.0)
        self.declare_parameter('camera_yaw_offset_deg', 0.0)

        gp = self.get_parameter
        self.base_frame = gp('base_frame').value
        self.target_class = gp('target_class').value
        self.conf = gp('confidence').value
        self.use_color = gp('use_color_fallback').value
        self.cube_height = gp('cube_height').value
        self.camera_yaw_sign = gp('camera_yaw_sign').value
        self.camera_yaw_offset_deg = gp('camera_yaw_offset_deg').value

        self.bridge = CvBridge()
        self.K = None
        if not self.use_color:
            self.model = YOLO(gp('model_path').value)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 카메라 내부파라미터
        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self.info_cb, 10)

        # 컬러 + 뎁스 동기화
        color_sub = message_filters.Subscriber(self, Image, gp('color_topic').value,
                                               qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, gp('depth_topic').value,
                                               qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_cb)

        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        self.get_logger().info(
            f'object_detector 시작 (use_color_fallback={self.use_color}, '
            f'cube_height={self.cube_height})')

    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'intrinsics fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    # ---- 인식: YOLO 또는 색 기반 ----
    def detect(self, bgr):
        """반환: ((u, v), angle_deg) 또는 (None, None)
        angle_deg: 이미지 평면에서 물체가 회전된 각도(정육면체 등 사각형 물체용).
        seg 마스크가 없는 모델(순수 detection)일 경우 angle_deg는 None."""
        if self.use_color:
            return self.detect_color(bgr)

        results = self.model(bgr, conf=self.conf, verbose=False)[0]
        has_masks = getattr(results, 'masks', None) is not None
        best_area, best_center, best_angle = 0.0, None, None

        for i, box in enumerate(results.boxes):
            cls = int(box.cls[0])
            if self.target_class >= 0 and cls != self.target_class:
                continue

            if has_masks:
                # seg 모델: 픽셀 마스크로 중심/회전각을 정확히 계산
                mask = results.masks.data[i].cpu().numpy().astype(np.uint8)
                if mask.shape[:2] != bgr.shape[:2]:
                    mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area <= best_area:
                    continue
                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                best_area = area
                best_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                best_angle = self._angle_from_contour(c)
            else:
                # 순수 detection 모델: bbox 중심만 사용, 회전각 정보 없음
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area <= best_area:
                    continue
                best_area = area
                best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                best_angle = None

        return best_center, best_angle

    @staticmethod
    def _angle_from_contour(c):
        """윤곽선(contour)으로부터 정사각형 대칭(-45~45도)으로 정규화된 회전각 계산.
        cv2.minAreaRect()의 angle 필드는 OpenCV 버전마다 규약(범위)이 달라
        신뢰할 수 없으므로, 사각형 꼭짓점(box points)에서 변 벡터를 직접 구해
        atan2로 계산한다 (버전 독립적)."""
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)  # 4개 꼭짓점, (4,2)
        edge = box[1] - box[0]
        angle = math.degrees(math.atan2(edge[1], edge[0]))
        # 정사각형은 90도마다 같은 모양이므로 -45~45도 범위로 정규화
        return ((angle + 45.0) % 90.0) - 45.0

    def detect_color(self, bgr):
        """단순 HSV 색 세그멘테이션 (단색 큐브용 대체 방법)
        반환: ((u, v), angle_deg) 또는 (None, None)"""
        low = np.array(self.get_parameter('hsv_low').value)
        high = np.array(self.get_parameter('hsv_high').value)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return None, None
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        angle = self._angle_from_contour(c)
        return center, angle

    def get_depth(self, depth, u, v, win=5):
        h, w = depth.shape[:2]
        u = min(max(u, 0), w - 1); v = min(max(v, 0), h - 1)
        patch = depth[max(0, v-win):min(h, v+win+1),
                      max(0, u-win):min(w, u+win+1)].astype(np.float32)
        if depth.dtype == np.uint16:      # 16UC1(mm) → m
            patch /= 1000.0
        valid = patch[(patch > 0) & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size else None

    def image_cb(self, color_msg, depth_msg):
        if self.K is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        center, angle_deg = self.detect(bgr)
        if center is None:
            return
        u, v = center
        z = self.get_depth(depth, u, v)
        if z is None or z <= 0.0:
            return

        # 픽셀 → 카메라 광학 좌표계 3D 역투영
        pt = PointStamped()
        pt.header = depth_msg.header      # optical frame
        pt.point.x = (u - self.cx) * z / self.fx
        pt.point.y = (v - self.cy) * z / self.fy
        pt.point.z = z
        self.get_logger().info(
            f'  [debug] frame_id={pt.header.frame_id}, u={u}, v={v}, depth_z={z:.3f}, '
            f'optical_xyz=({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f})')

        # optical → base 변환
        # (eye-in-hand 카메라라도 TF가 매 프레임 실시간으로 팔의 현재 자세를
        #  반영해 변환해주므로, 이 부분은 카메라가 고정이든 그리퍼에 붙어
        #  움직이든 코드 변경 없이 동일하게 동작한다.)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, pt.header.frame_id, rclpy.time.Time())
            t = tf.transform.translation
            self.get_logger().info(
                f'  [debug] TF {self.base_frame}<-{pt.header.frame_id} '
                f'translation=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})')
            pb = do_transform_point(pt, tf)
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            return

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = pb.point
        # 뎁스로 측정된 값은 큐브 "윗면" 중심이므로, base_frame Z(+)가 위쪽 방향일 때
        # 큐브 높이의 절반만큼 내려서 실제 중심 위치로 보정한다.
        pose.pose.position.z -= self.cube_height / 2.0

        if angle_deg is None:
            pose.pose.orientation.w = 1.0
            yaw_log = 0.0
        else:
            # 이미지 평면(카메라 기준) 회전각을 base_frame yaw로 변환.
            # 카메라가 정확히 수직 하향(top-down)이고 image u축이 base_frame
            # X축과 평행하다는 가정 하의 근사치이며, 실제 마운트 각도에 따라
            # camera_yaw_sign/offset 파라미터로 보정이 필요할 수 있다.
            yaw_deg = self.camera_yaw_sign * angle_deg + self.camera_yaw_offset_deg
            yaw = math.radians(yaw_deg)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            yaw_log = yaw_deg

        self.pose_pub.publish(pose)
        self.get_logger().info(
            f'물체 위치(base): ({pb.point.x:.3f}, {pb.point.y:.3f}, {pb.point.z:.3f}), '
            f'yaw={yaw_log:.1f}deg')


def main():
    rclpy.init()
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
