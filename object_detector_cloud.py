#!/usr/bin/env python3
"""Depth 이미지 대신 PointCloud2(organized)에서 직접 3D 위치를 읽는 버전.
RGB에서 (u,v)를 찾고, 같은 (u,v)의 점군 포인트를 사용한다.
점군이 organized(width>1, height>1)여야 (u,v) 인덱싱이 가능하다."""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
try:
    from tf2_geometry_msgs import do_transform_point
except ImportError:
    from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

import cv2
from ultralytics import YOLO


class ObjectDetectorCloud(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('color_topic', '/rsd455/color/image_raw')
        self.declare_parameter('cloud_topic', '/rsd455/depth/points')
        self.declare_parameter('base_frame', 'panda_link0')
        self.declare_parameter('model_path', 'yolov8n-seg.pt')
        self.declare_parameter('target_class', -1)
        self.declare_parameter('confidence', 0.4)
        self.declare_parameter('use_color_fallback', False)
        self.declare_parameter('hsv_low', [0, 120, 70])
        self.declare_parameter('hsv_high', [10, 255, 255])
        # 점군은 실제 3D 좌표를 직접 주므로 원칙적으로 큐브 윗면 보정이
        # depth 이미지 버전만큼 필요하지 않지만, 점군에서도 마스크 상단면의
        # median을 사용하므로 동일하게 절반 높이를 보정해준다.
        self.declare_parameter('cube_height', 0.04)
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
        if not self.use_color:
            self.model = YOLO(gp('model_path').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        color_sub = message_filters.Subscriber(self, Image, gp('color_topic').value,
                                               qos_profile=qos_profile_sensor_data)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, gp('cloud_topic').value,
                                               qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, cloud_sub], queue_size=10, slop=0.15)
        self.sync.registerCallback(self.cb)

        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        self.get_logger().info('object_detector (cloud) 시작')

    @staticmethod
    def _angle_from_contour(c):
        """object_detector.py와 동일한 버전 독립적 각도 계산 (boxPoints 기반)."""
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        edge = box[1] - box[0]
        angle = math.degrees(math.atan2(edge[1], edge[0]))
        return ((angle + 45.0) % 90.0) - 45.0

    def detect(self, bgr):
        """반환: ((u, v), angle_deg) 또는 (None, None)"""
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
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area <= best_area:
                    continue
                best_area = area
                best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                best_angle = None
        return best_center, best_angle

    def detect_color(self, bgr):
        """반환: ((u, v), angle_deg) 또는 (None, None)"""
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

    def point_at(self, cloud, u, v, win=4):
        """organized 점군에서 (u,v) 주변의 유효 포인트 median 반환."""
        w, h = cloud.width, cloud.height
        if h <= 1:
            self.get_logger().warn('점군이 organized 아님(height<=1). depth 이미지 방식이 필요.')
            return None
        u = min(max(u, 0), w - 1); v = min(max(v, 0), h - 1)
        us = range(max(0, u-win), min(w, u+win+1))
        vs = range(max(0, v-win), min(h, v+win+1))
        uvs = [(uu, vv) for vv in vs for uu in us]
        pts = list(pc2.read_points(cloud, field_names=('x', 'y', 'z'),
                                   skip_nans=True, uvs=uvs))
        if not pts:
            return None
        arr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        arr = arr[np.isfinite(arr).all(axis=1)]
        return np.median(arr, axis=0) if arr.size else None

    def cb(self, color_msg, cloud_msg):
        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        center, angle_deg = self.detect(bgr)
        if center is None:
            return
        u, v = center
        xyz = self.point_at(cloud_msg, u, v)
        if xyz is None:
            return

        pt = PointStamped()
        pt.header = cloud_msg.header           # 점군의 프레임(대개 optical)
        pt.point.x, pt.point.y, pt.point.z = map(float, xyz)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, pt.header.frame_id, rclpy.time.Time())
            pb = do_transform_point(pt, tf)
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            return

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = pb.point
        pose.pose.position.z -= self.cube_height / 2.0

        if angle_deg is None:
            pose.pose.orientation.w = 1.0
            yaw_log = 0.0
        else:
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
    node = ObjectDetectorCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
