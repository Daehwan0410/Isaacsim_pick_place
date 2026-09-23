#!/usr/bin/env python3
import time, math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint

ARM = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4',
       'panda_joint5','panda_joint6','panda_joint7']

# 정사각형 물체는 90도 대칭이므로 이 각도들이 모두 유효한 파지 방향 후보다.
WORLD_YAWS = (0, 90, -90, 180)


class PickPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')
        self.declare_parameter('place_dx', 0.2)
        self.declare_parameter('approach_height', 0.15)
        self.declare_parameter('grasp_z_offset', 0.10)
        # panda_link8 기준 yaw=0일 때, 실제 그리퍼 손가락이 열리는 방향이
        # 월드 X/Y축과 정렬되지 않는 경우가 많다 (Franka Panda 손목 구조상
        # 보통 45도 정도 오프셋 필요). 실측해서 맞춰 넣는 값.
        self.declare_parameter('gripper_yaw_offset', 0.0)

        self.place_dx = self.get_parameter('place_dx').value
        self.approach = self.get_parameter('approach_height').value
        self.grasp_off = self.get_parameter('grasp_z_offset').value
        self.gripper_yaw_offset = self.get_parameter('gripper_yaw_offset').value
        self.get_logger().info(
            f'[설정값] place_dx={self.place_dx}, approach_height={self.approach}, '
            f'grasp_z_offset={self.grasp_off}, gripper_yaw_offset={self.gripper_yaw_offset}')

        self.latest = None
        self.js = None
        self.create_subscription(PoseStamped, '/detected_object_pose', self._p, 10)
        self.create_subscription(JointState, '/joint_states', self._j, 10)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.arm = ActionClient(self, FollowJointTrajectory, '/panda_arm_controller/follow_joint_trajectory')
        self.grip = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')

    def _p(self, m): self.latest = m
    def _j(self, m): self.js = m

    def _quat(self, yaw):
        t = math.radians(yaw)/2.0
        return (math.cos(t), math.sin(t), 0.0, 0.0)

    def _compute_ik(self, pos, q):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = 'panda_arm'
        r.ik_link_name = 'panda_link8'
        r.pose_stamped.header.frame_id = 'panda_link0'
        r.pose_stamped.pose.position.x = float(pos[0])
        r.pose_stamped.pose.position.y = float(pos[1])
        r.pose_stamped.pose.position.z = float(pos[2])
        r.pose_stamped.pose.orientation.x = q[0]
        r.pose_stamped.pose.orientation.y = q[1]
        r.pose_stamped.pose.orientation.z = q[2]
        r.pose_stamped.pose.orientation.w = q[3]
        if self.js is not None:
            r.robot_state.joint_state = self.js
        r.avoid_collisions = True
        r.timeout.sec = 1
        fut = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return None
        d = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        try:
            return [float(d[j]) for j in ARM]
        except KeyError:
            return None

    def move_to(self, pos, yaws=WORLD_YAWS):
        """pos로 이동. yaws에 주어진 각도들을 순서대로 시도해 IK가 풀리는
        첫 각도를 사용한다. 각 후보 각도에 gripper_yaw_offset을 더해 실제
        그리퍼 손가락 방향과 panda_link8 좌표계 사이의 고정 오프셋을 보정한다.
        반환: (성공 여부, 사용된 후보 yaw(오프셋 적용 전 원래 값))"""
        for yaw in yaws:
            eff_yaw = yaw + self.gripper_yaw_offset
            j = self._compute_ik(pos, self._quat(eff_yaw))
            if j is not None:
                self._send_arm(j)
                self.get_logger().info(
                    f'  -> yaw={yaw} (+offset {self.gripper_yaw_offset} = {eff_yaw}) 이동')
                return True, yaw
        self.get_logger().warn(f'  x {pos} IK 실패')
        return False, None

    def _send_arm(self, joints, dur=3.0):
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = ARM
        p = JointTrajectoryPoint()
        p.positions = joints
        p.time_from_start.sec = int(dur)
        g.trajectory.points = [p]
        self.arm.wait_for_server()
        f = self.arm.send_goal_async(g); rclpy.spin_until_future_complete(self, f)
        rf = f.result().get_result_async(); rclpy.spin_until_future_complete(self, rf)
        time.sleep(0.3)

    def gripper(self, pos, settle_timeout=5.0, tol=0.005):
        """그리퍼를 pos(0=닫힘, 0.04=열림 등)로 이동시킨다.
        액션 결과 상태를 확인하고, 그 결과와 무관하게 실제 조인트 값이
        목표 근처에서 안정될 때까지 폴링해서 Isaac Sim 물리 스텝 지연으로
        인한 '늦게 닫히는' 문제를 방지한다."""
        g = GripperCommand.Goal()
        g.command.position = float(pos); g.command.max_effort = 20.0
        self.grip.wait_for_server()
        f = self.grip.send_goal_async(g); rclpy.spin_until_future_complete(self, f)
        gh = f.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('  그리퍼 목표가 accept 되지 않음')
            return False
        rf = gh.get_result_async(); rclpy.spin_until_future_complete(self, rf)
        result = rf.result()
        status = result.status if result is not None else None
        # status 4 == SUCCEEDED (rclpy.action.GoalStatus.STATUS_SUCCEEDED)
        if status != 4:
            self.get_logger().warn(f'  그리퍼 액션 실패/중단 (status={status})')

        # 액션 결과와 무관하게, 실제 조인트가 목표 위치 근처에 안정적으로
        # 도달했는지 /joint_states를 직접 폴링해서 확인한다.
        # (물체가 두꺼워 pos=0.0까지 물리적으로 못 가는 경우 등은 타임아웃으로
        #  넘어가며, 이 경우 5초 정도 추가 대기 후 계속 진행한다.)
        t0 = time.time()
        settled_count = 0
        while rclpy.ok() and (time.time() - t0) < settle_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.js is None or 'panda_finger_joint1' not in self.js.name:
                continue
            idx = self.js.name.index('panda_finger_joint1')
            cur = self.js.position[idx]
            if abs(cur - pos) < tol:
                settled_count += 1
                if settled_count >= 3:   # 연속 3회(약 0.3초) 안정적으로 도달
                    break
            else:
                settled_count = 0
        time.sleep(0.3)
        return True

    def run(self):
        while rclpy.ok() and self.js is None:
            rclpy.spin_once(self, timeout_sec=0.2)

        # 시작 시점의 관절 각도를 저장해둔다 (나중에 이 자리로 복귀하기 위함).
        # IK를 다시 풀 필요 없이 저장된 관절값으로 바로 돌아갈 수 있어 안전하다.
        try:
            home_joints = [self.js.position[self.js.name.index(j)] for j in ARM]
        except ValueError:
            home_joints = None
            self.get_logger().warn('초기 관절 상태를 읽지 못해 홈 복귀를 못 할 수 있음')

        self.get_logger().info('물체 인식 대기...')
        while rclpy.ok() and self.latest is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        try:
            self._run_sequence()
        finally:
            if home_joints is not None:
                self.get_logger().info('홈 위치로 복귀')
                self._send_arm(home_joints)
            self.get_logger().info('완료')

    def _run_sequence(self):
        p = self.latest.pose.position
        obj = [p.x, p.y, p.z]
        self.get_logger().info(f'집을 물체: {obj}')

        # object_detector가 추정한 큐브의 실제 회전각(yaw)을 읽어온다.
        # 정사각형 대칭이므로 90도씩 더한 각도들도 모두 유효한 파지 방향이다.
        qz = self.latest.pose.orientation.z
        qw = self.latest.pose.orientation.w
        obj_yaw = math.degrees(2.0 * math.atan2(qz, qw))
        cand = [obj_yaw, obj_yaw + 90.0, obj_yaw - 90.0, obj_yaw + 180.0]
        # world 기준 각도는 최후 폴백으로만 남겨둔다.
        cand += list(WORLD_YAWS)
        self.get_logger().info(f'  물체 추정 yaw={obj_yaw:.1f}deg, 후보={[round(c,1) for c in cand]}')

        gz = obj[2] + self.grasp_off
        pre = [obj[0], obj[1], gz+self.approach]
        grasp = [obj[0], obj[1], gz]
        place = [obj[0]+self.place_dx, obj[1], gz]
        pre_pl = [place[0], place[1], gz+self.approach]
        self.get_logger().info('그리퍼 열기'); self.gripper(0.04)

        # grasp 자세에서 먼저 유효한 yaw를 하나 결정한 뒤,
        # 접근/하강/들기 전체에 동일한 yaw를 고정 사용한다.
        # (매 단계마다 다른 yaw를 골라 그리퍼가 모서리로 틀어지는 것을 방지)
        ok, grasp_yaw = self.move_to(pre, yaws=cand)
        if not ok:
            self.get_logger().warn('접근 자세 IK 실패, 종료'); return
        fixed = (grasp_yaw,) + tuple(y for y in cand if y != grasp_yaw)

        self.get_logger().info('하강')
        ok, grasp_yaw2 = self.move_to(grasp, yaws=fixed)
        if not ok:
            self.get_logger().warn('하강 IK 실패, 종료'); return
        fixed = (grasp_yaw2,) + tuple(y for y in cand if y != grasp_yaw2)

        self.get_logger().info('그리퍼 닫기'); self.gripper(0.0)
        self.get_logger().info('들기');       self.move_to(pre, yaws=fixed)
        self.get_logger().info('배치접근');   self.move_to(pre_pl, yaws=fixed)
        self.get_logger().info('배치');       self.move_to(place, yaws=fixed)
        self.get_logger().info('그리퍼 열기'); self.gripper(0.04)
        self.get_logger().info('후퇴');       self.move_to(pre_pl, yaws=fixed)


def main():
    rclpy.init()
    n = PickPlace()
    n.run()
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
