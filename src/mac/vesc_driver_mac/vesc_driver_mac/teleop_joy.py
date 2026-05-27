#!/usr/bin/env python3
"""Joystick teleop for VESC Ackermann vehicle.

Subscribes to /joy and publishes AckermannDriveStamped to /ackermann_cmd
at a fixed rate (publish_rate Hz). /joy 는 joy_mac 의 Swift bridge 가
값 변화 시점에만 발행하므로, 스틱 고정 시 명령 끊김 방지 위해
마지막 입력 상태를 캐시 + 타이머 기반 지속 발행.

Controls (Xbox layout):
  Left stick Y  → speed (forward/backward)
  Right stick X → steering angle (left/right)
  LB button     → enable driving (deadman switch)
  B button      → emergency stop (press to toggle)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped


class TeleopJoy(Node):
    def __init__(self):
        super().__init__('teleop_joy_node')

        # Parameters
        self.declare_parameter('max_speed', 2.0)           # m/s
        self.declare_parameter('max_steering_angle', 0.34)  # radians (~19.5 deg)
        self.declare_parameter('axis_speed', 1)             # left stick Y
        self.declare_parameter('axis_steering', 3)          # right stick X
        self.declare_parameter('button_enable', 4)          # LB (deadman)
        self.declare_parameter('button_estop', 1)           # B button
        self.declare_parameter('button_tuning', 3)          # Y button (tuning toggle)
        self.declare_parameter('tuning_speed', 1.0)         # m/s fixed in tuning mode
        self.declare_parameter('publish_rate', 50.0)        # Hz

        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering = self.get_parameter('max_steering_angle').value
        self.axis_speed = self.get_parameter('axis_speed').value
        self.axis_steering = self.get_parameter('axis_steering').value
        self.btn_enable = self.get_parameter('button_enable').value
        self.btn_estop = self.get_parameter('button_estop').value
        self.btn_tuning = self.get_parameter('button_tuning').value
        self.tuning_speed = self.get_parameter('tuning_speed').value
        rate = self.get_parameter('publish_rate').value

        self.pub = self.create_publisher(AckermannDriveStamped, 'ackermann_cmd', 10)
        self.sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)

        # 캐시된 입력 상태 — 50Hz timer 가 이걸 보고 ack 발행
        self.estopped = False
        self.tuning = False
        self._enabled = False
        self._target_speed = 0.0
        self._target_steering = 0.0

        # 50Hz 지속 발행 — joy 가 idle 이어도 마지막 상태를 계속 보냄
        self.create_timer(1.0 / rate, self._publish_tick)

        self.get_logger().info(
            f'Teleop ready: LB=enable, B=e-stop | '
            f'max_speed={self.max_speed}, max_steer={self.max_steering:.2f}, '
            f'rate={rate:.0f}Hz'
        )

    def joy_callback(self, msg: Joy):
        """입력 상태만 갱신. 발행은 timer 에서."""
        # E-stop toggle — Swift bridge 가 value-change 시점에만 fire 하므로
        # btn_estop==1 메시지는 누를 때 1번, 뗄 때 0 으로 1번 와서 toggle 안 중복.
        if len(msg.buttons) > self.btn_estop and msg.buttons[self.btn_estop]:
            self.estopped = not self.estopped
            state = 'ENGAGED' if self.estopped else 'RELEASED'
            self.get_logger().info(f'E-stop {state}')

        if len(msg.buttons) > self.btn_tuning and msg.buttons[self.btn_tuning]:
            self.tuning = not self.tuning
            state = 'ON' if self.tuning else 'OFF'
            self.get_logger().info(
                f'Tuning mode {state} (fixed speed {self.tuning_speed} m/s)'
            )

        self._enabled = (
            len(msg.buttons) > self.btn_enable
            and msg.buttons[self.btn_enable] == 1
        )

        if len(msg.axes) > max(self.axis_speed, self.axis_steering):
            self._target_speed = msg.axes[self.axis_speed] * self.max_speed
            # Stick X axis is sign-flipped so the ack.steering_angle follows
            # ROS REP-103 convention (positive = LEFT turn). Combined with a
            # positive steering_angle_to_servo_gain in vesc_config.yaml this
            # keeps stick→wheel direction identical to before.
            self._target_steering = -msg.axes[self.axis_steering] * self.max_steering
        else:
            self._target_speed = 0.0
            self._target_steering = 0.0

    def _publish_tick(self):
        ack = AckermannDriveStamped()
        ack.header.stamp = self.get_clock().now().to_msg()
        ack.header.frame_id = 'base_link'

        if self.estopped or not self._enabled:
            ack.drive.speed = 0.0
            ack.drive.steering_angle = 0.0
        elif self.tuning:
            ack.drive.speed = float(self.tuning_speed)
            ack.drive.steering_angle = float(self._target_steering)
        else:
            ack.drive.speed = float(self._target_speed)
            ack.drive.steering_angle = float(self._target_steering)

        self.pub.publish(ack)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
