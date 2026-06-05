#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Int32
from copy import deepcopy

from controller.estop import EStop


class SimpleMuxNode(Node):

    def __init__(self):
        super().__init__('simple_mux')

        self.declare_parameter('out_topic',                      'low_level/ackermann_cmd_mux/output')
        self.declare_parameter('in_topic',                       'high_level/ackermann_cmd')
        # Reactive fallback (FTG) — MPCC 가 죽거나 stale 하면 자동 사용.
        # 빈 문자열이면 fallback 비활성.
        self.declare_parameter('fallback_topic',                 '/vesc/ftg_fallback')
        # MPCC stale 으로 인식하는 시간 (초). 이 시간 이상 cmd 안 오면 fallback.
        self.declare_parameter('autodrive_stale_sec',            0.2)
        self.declare_parameter('joy_topic',                      '/joy')
        self.declare_parameter('scan_topic',                     '/scan')
        self.declare_parameter('odom_topic',                     '/vesc/odom')
        self.declare_parameter('rate_hz',                        50.0)
        self.declare_parameter('joy_max_speed',                  4.0)
        self.declare_parameter('joy_max_steer',                  0.4)
        self.declare_parameter('joy_freshness_threshold',        1.0)
        self.declare_parameter('servo_min',                      0.15)
        self.declare_parameter('servo_max',                      0.85)
        self.declare_parameter('steering_angle_to_servo_offset', 0.5)
        self.declare_parameter('steering_angle_to_servo_gain',  -1.2135)
        self.declare_parameter('use_estop',  False)
        self.declare_parameter('sim',  False)
        p = lambda name: self.get_parameter(name).value

        out_topic  = p('out_topic')
        in_topic   = p('in_topic')
        joy_topic  = p('joy_topic')
        scan_topic = p('scan_topic')
        odom_topic = p('odom_topic')

        self.use_estop = p('use_estop')
        self.max_speed               = p('joy_max_speed')
        self.max_steer               = p('joy_max_steer')
        self.joy_freshness_threshold = p('joy_freshness_threshold')

        servo_offset = p('steering_angle_to_servo_offset')
        servo_gain   = p('steering_angle_to_servo_gain')
        self.servo_max_abs = min(
            abs((p('servo_max') - servo_offset) / servo_gain),
            abs((p('servo_min') - servo_offset) / servo_gain),
        )
        

        self.current_host = 'autodrive' if p('sim') else None
        self.human_drive  = None
        self.autodrive    = None
        self.fallback     = None    # latest FTG cmd
        self.scan         = None
        self.odom         = None
        self.autodrive_stale_sec = float(p('autodrive_stale_sec'))
        self._using_fallback = False  # state log: prev cycle fallback 썼나

        self.create_subscription(AckermannDriveStamped, in_topic,  self._drive_cb, 10)
        # Reactive fallback subscribe — 빈 문자열이면 skip
        fallback_topic = str(p('fallback_topic')).strip()
        if fallback_topic:
            self.create_subscription(AckermannDriveStamped, fallback_topic,
                                     self._fallback_cb, 10)
            self.get_logger().info(
                f'[mux] reactive fallback active — listen on {fallback_topic} '
                f'(autodrive stale → {self.autodrive_stale_sec:.2f}s)')
        self.create_subscription(Joy,                   joy_topic, self._joy_cb,   10)
        if self.use_estop:
            self.estop = EStop(self)

            self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
            self.create_subscription(Odometry,  odom_topic, self._odom_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, out_topic, 10)
        # MPCC alive / switch tracking — mpc_debug_logger 가 구독해서 CSV 에 기록.
        # mpcc_active: 매 loop cycle 의 source (True=MPCC, False=fallback/zero).
        # switch_count: 누적 source 전환 횟수 (MPCC↔fallback). 0 부터 시작.
        self.mpcc_active_pub = self.create_publisher(Bool,  '/mux/mpcc_active',  10)
        self.switch_count_pub = self.create_publisher(Int32, '/mux/switch_count', 10)
        self._switch_count = 0
        self._prev_source = None  # 'mpcc' | 'fallback' | 'zero'
        self.create_timer(1.0 / p('rate_hz'), self._loop)

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _drive_cb(self, msg): self.autodrive = msg
    def _fallback_cb(self, msg): self.fallback = msg

    def _is_fresh(self, msg, thresh=None):
        if msg is None:
            return False
        if thresh is None:
            thresh = self.joy_freshness_threshold
        dt = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        return abs(dt) < thresh

    def _clip(self, msg):
        out = deepcopy(msg)
        out.drive.steering_angle = max(-self.servo_max_abs, min(self.servo_max_abs, out.drive.steering_angle))
        return out

    def _loop(self):
        zero = AckermannDriveStamped()
        zero.header.stamp = self.get_clock().now().to_msg()
        source = 'zero'  # 기본값 — 아래에서 갱신

        if self.current_host == 'autodrive':
            # autodrive (MPCC) fresh 여부를 tight threshold (autodrive_stale_sec) 로
            # 판단. stale 이면 fallback (PP/FTG) 으로 자동 switch.
            mpcc_fresh = self._is_fresh(self.autodrive, thresh=self.autodrive_stale_sec)
            if mpcc_fresh:
                out = deepcopy(self.autodrive)
                source = 'mpcc'
                if self._using_fallback:
                    self.get_logger().info('[mux] MPCC 복귀 — autodrive 로 다시 switch')
                    self._using_fallback = False
            elif self._is_fresh(self.fallback, thresh=self.autodrive_stale_sec):
                out = deepcopy(self.fallback)
                source = 'fallback'
                if not self._using_fallback:
                    self.get_logger().warn(
                        f'[mux] MPCC stale (>{self.autodrive_stale_sec:.2f}s) — '
                        f'fallback 사용 중')
                    self._using_fallback = True
            else:
                out = zero  # 둘 다 stale → 안전 정지
                source = 'zero'
        elif self.current_host == 'humandrive' and self._is_fresh(self.human_drive):
            out = deepcopy(self.human_drive)
            source = 'human'
        else:
            out = zero
            source = 'zero'

        if self.use_estop:
            out = self.estop.should_stop(self.scan, self.odom, out)

        self.drive_pub.publish(out)

        # ── MPCC alive / switch tracking ──
        # mpcc_active: 이번 cycle 에서 MPCC 사용했는지 boolean (CSV 평균 → mpcc_alive_frac).
        # switch_count: source 변경 시마다 +1 누적 (학습 데이터에서 안정성 평가용).
        is_mpcc = (source == 'mpcc')
        self.mpcc_active_pub.publish(Bool(data=is_mpcc))
        if self._prev_source is not None and source != self._prev_source:
            # mpcc ↔ fallback ↔ zero ↔ human 모든 전환 카운트.
            self._switch_count += 1
        self._prev_source = source
        self.switch_count_pub.publish(Int32(data=self._switch_count))

    def _joy_cb(self, msg):
        use_human = msg.buttons[4] if len(msg.buttons) > 4 else False
        use_auto  = msg.buttons[5] if len(msg.buttons) > 5 else False

        if use_human:
            drive = AckermannDriveStamped()
            drive.header.stamp = self.get_clock().now().to_msg()
            drive.drive.steering_angle = msg.axes[3] * self.max_steer if len(msg.axes) > 3 else 0.0
            drive.drive.speed          = msg.axes[1] * self.max_speed  if len(msg.axes) > 1 else 0.0
            self.human_drive   = drive
            self.current_host  = 'humandrive'
        elif use_auto:
            self.current_host = 'autodrive'


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
