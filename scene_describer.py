#!/usr/bin/env python3
"""카메라 영상을 Ollama의 Gemma 3 비전 모델(gemma3:4b)로 보내
장면/물체 설명을 받아 /scene_description 토픽으로 발행하는 ROS2 노드.

- 기본: describe_period(초)마다 최신 프레임 1장을 Gemma 3에 전송
- /describe_trigger (std_msgs/Empty) 를 받으면 즉시 1회 설명
Ollama 서버는 별도로 실행돼 있어야 함:  ollama serve  +  ollama pull gemma3:4b
"""
import json
import base64
import threading

import cv2
import requests

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge


class SceneDescriber(Node):
    def __init__(self):
        super().__init__('scene_describer')

        # ---- 파라미터 ----
        self.declare_parameter('image_topic', '/rsd455/color/image_raw')
        self.declare_parameter('ollama_host', 'http://localhost:11434')
        self.declare_parameter('model', 'gemma4:latest')
        self.declare_parameter('describe_period', 5.0)   # 0 이하면 주기 실행 끔(트리거만)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('max_width', 640)         # 전송 전 리사이즈(속도)
        self.declare_parameter('timeout', 120.0)
        self.declare_parameter(
            'prompt',
            '이 이미지에 보이는 물체들을 나열하고 각각을 한 문장으로 설명해줘. '
            '색깔, 대략적인 위치(왼쪽/가운데/오른쪽), 형태를 포함해줘. 한국어로 답해줘.')

        gp = self.get_parameter
        self.host = gp('ollama_host').value.rstrip('/')
        self.model = gp('model').value
        self.quality = gp('jpeg_quality').value
        self.max_width = gp('max_width').value
        self.timeout = gp('timeout').value
        self.prompt = gp('prompt').value

        self.bridge = CvBridge()
        self.latest = None            # 최신 BGR 프레임
        self.busy = False             # 요청 진행 중 플래그(중복 방지)
        self.lock = threading.Lock()

        self.create_subscription(Image, gp('image_topic').value,
                                 self.image_cb, qos_profile_sensor_data)
        self.create_subscription(Empty, '/describe_trigger',
                                 self.trigger_cb, 10)
        self.pub = self.create_publisher(String, '/scene_description', 10)

        period = gp('describe_period').value
        if period > 0:
            self.create_timer(period, self.timer_cb)
            self.get_logger().info(f'{period}초마다 장면 설명 요청')
        self.get_logger().info(
            f'scene_describer 시작 (model={self.model}, host={self.host})')

    def image_cb(self, msg):
        self.latest = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def trigger_cb(self, _msg):
        self.request_description()

    def timer_cb(self):
        self.request_description()

    def request_description(self):
        if self.latest is None:
            self.get_logger().warn('아직 수신된 이미지 없음')
            return
        with self.lock:
            if self.busy:
                self.get_logger().info('이전 요청 처리 중 — 건너뜀')
                return
            self.busy = True
        frame = self.latest.copy()
        threading.Thread(target=self._worker, args=(frame,), daemon=True).start()

    def _encode(self, bgr):
        h, w = bgr.shape[:2]
        if self.max_width and w > self.max_width:
            s = self.max_width / w
            bgr = cv2.resize(bgr, (self.max_width, int(h * s)))
        ok, buf = cv2.imencode('.jpg', bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def _worker(self, frame):
        try:
            b64 = self._encode(frame)
            if b64 is None:
                self.get_logger().error('JPEG 인코딩 실패')
                return
            payload = {
                'model': self.model,
                'prompt': self.prompt,
                'images': [b64],
                'stream': False,
            }
            r = requests.post(f'{self.host}/api/generate',
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json().get('response', '').strip()
            msg = String(); msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f'[장면 설명]\n{text}')
        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f'Ollama 연결 실패 ({self.host}). '
                '`ollama serve` 실행 여부와 host 파라미터를 확인하세요.')
        except Exception as e:
            self.get_logger().error(f'요청 실패: {e}')
        finally:
            with self.lock:
                self.busy = False


def main():
    rclpy.init()
    node = SceneDescriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
