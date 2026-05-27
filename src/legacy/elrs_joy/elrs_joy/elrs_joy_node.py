#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELRS Joy Node (ROS1)
- Reads CRSF packets from ELRS receiver via USB-TTL serial
- Publishes sensor_msgs/Joy topic (Xbox-compatible layout)
- No CRC check - uses channel value range validation instead
  (for USB-TTL chips that can't hit exact 420000 baud)
"""

from sensor_msgs.msg import Joy
import serial
import time

import rclpy
from rclpy.node import Node


class ELRSJoyNode(Node):
    CRSF_SYNC = 0xC8
    CRSF_FRAMETYPE_RC_CHANNELS = 0x16
    CRSF_NUM_CHANNELS = 16

    # Valid channel range (with margin)
    CH_MIN = 100
    CH_MAX = 1900
    CH_MID = 992

    # For normalization
    NORM_MIN = 172
    NORM_MAX = 1811


    def _get_param_or_default(self, name, default=None):
        candidates = [name]
        if "/" in name:
            candidates.append(name.replace("/", "."))
            candidates.append(name.lstrip("/"))
        for n in candidates:
            try:
                v = self.get_parameter(n).value
                if v is not None:
                    return v
            except Exception:
                continue
        if default is None:
            return None
        try:
            self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return v if v is not None else default
        except Exception:
            return default

    def __init__(self):
        Node.__init__(
            self, 'elrs_joy_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.port = self._get_param_or_default('port', '/dev/ttyUSB0')
        self.baud_rate = self._get_param_or_default('baud_rate', 416666)
        self.frame_id = self._get_param_or_default('frame_id', 'elrs_joy')
        self.publish_rate = self._get_param_or_default('publish_rate', 100)

        # Xbox-compatible Joy message layout
        # axes_map: {joy_index: crsf_channel}  buttons_map: {joy_index: crsf_channel}
        self.num_axes = self._get_param_or_default('num_axes', 8)
        self.num_buttons = self._get_param_or_default('num_buttons', 11)
        self.axes_joy_indices = self._get_param_or_default('axes_joy_indices', [1, 3])
        self.axes_crsf_channels = self._get_param_or_default('axes_crsf_channels', [0, 2])
        self.button_joy_indices = self._get_param_or_default('button_joy_indices', [4, 5])
        self.button_crsf_channels = self._get_param_or_default('button_crsf_channels', [5, 6])
        self.button_threshold = self._get_param_or_default('button_threshold', 992)
        self.axes_invert = self._get_param_or_default('axes_invert', [1.0, 1.0])
        self.deadzone = self._get_param_or_default('deadzone', 0.05)
        self.failsafe_timeout = self._get_param_or_default('failsafe_timeout', 2.0)

        self.joy_pub = self.create_publisher(Joy, 'joy', 10)

        self.channels = [self.CH_MID] * self.CRSF_NUM_CHANNELS
        self.last_valid_time = time.time()
        self.connected = False
        self.serial_port = None
        self.buffer = bytearray()

        # Stats (print once then stop)
        self.accept_count = 0
        self.reject_count = 0
        self.last_stats_time = time.time()
        self.stats_printed = False

    def normalize_axis(self, value):
        normalized = (value - self.CH_MID) / (self.NORM_MAX - self.CH_MID)
        normalized = max(-1.0, min(1.0, normalized))
        if abs(normalized) < self.deadzone:
            normalized = 0.0
        return normalized

    def channel_to_button(self, value):
        return 1 if value < self.button_threshold else 0

    def validate_channels(self, channels):
        """Validate by checking channel values are in sane range"""
        # At least first 4 channels must be in valid range
        for i in range(min(4, len(channels))):
            if channels[i] < self.CH_MIN or channels[i] > self.CH_MAX:
                return False

        # Must have some variation (not all identical)
        if len(set(channels[:4])) < 2:
            # Exception: all centered is ok (sticks neutral)
            if not all(abs(ch - self.CH_MID) < 50 for ch in channels[:4]):
                return False

        return True

    def parse_rc_channels(self, payload):
        if len(payload) < 22:
            return False

        channels = []
        bit_offset = 0
        for i in range(self.CRSF_NUM_CHANNELS):
            byte_offset = bit_offset // 8
            bit_shift = bit_offset % 8

            if byte_offset + 1 < len(payload):
                value = payload[byte_offset] >> bit_shift
                bits_from_first = 8 - bit_shift
                if bits_from_first < 11 and byte_offset + 1 < len(payload):
                    value |= payload[byte_offset + 1] << bits_from_first
                    bits_from_second = 11 - bits_from_first
                    if bits_from_second > 8 and byte_offset + 2 < len(payload):
                        value |= payload[byte_offset + 2] << (bits_from_first + 8)
                value &= 0x7FF
                channels.append(value)
            else:
                channels.append(self.CH_MID)

            bit_offset += 11

        if self.validate_channels(channels):
            self.channels = channels
            self.last_valid_time = time.time()
            self.accept_count += 1
            return True
        else:
            self.reject_count += 1
            return False

    def parse_crsf_frame(self):
        while len(self.buffer) > 2:
            # Find sync byte
            sync_idx = -1
            for i in range(len(self.buffer)):
                if self.buffer[i] == self.CRSF_SYNC:
                    sync_idx = i
                    break

            if sync_idx == -1:
                self.buffer.clear()
                return

            if sync_idx > 0:
                self.buffer = self.buffer[sync_idx:]

            if len(self.buffer) < 3:
                return

            frame_length = self.buffer[1]

            # RC channels frame: length should be 24 (type + 22 payload + crc)
            if frame_length < 2 or frame_length > 64:
                self.buffer = self.buffer[1:]
                continue

            total_size = 2 + frame_length
            if len(self.buffer) < total_size:
                return

            frame_type = self.buffer[2]

            if frame_type == self.CRSF_FRAMETYPE_RC_CHANNELS:
                payload = self.buffer[3:total_size - 1]
                if self.parse_rc_channels(payload):
                    if not self.connected:
                        self.connected = True
                        self.get_logger().info("CRSF receiver connected!")

            self.buffer = self.buffer[total_size:]

        # Prevent unbounded growth
        if len(self.buffer) > 512:
            self.buffer = self.buffer[-256:]

    def publish_joy(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.axes = [0.0] * self.num_axes
        msg.buttons = [0] * self.num_buttons
        for i, (joy_idx, crsf_ch) in enumerate(zip(self.axes_joy_indices, self.axes_crsf_channels)):
            sign = self.axes_invert[i] if i < len(self.axes_invert) else 1.0
            msg.axes[joy_idx] = sign * self.normalize_axis(self.channels[crsf_ch])
        for joy_idx, crsf_ch in zip(self.button_joy_indices, self.button_crsf_channels):
            msg.buttons[joy_idx] = self.channel_to_button(self.channels[crsf_ch])
        self.joy_pub.publish(msg)

    def check_failsafe(self):
        elapsed = time.time() - self.last_valid_time
        if elapsed > self.failsafe_timeout:
            if self.connected:
                self.get_logger().warning("CRSF signal lost! (no valid packet for %.1fs)", elapsed)
                self.connected = False
                msg = Joy()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.frame_id
                msg.axes = [0.0] * self.num_axes
                msg.buttons = [0] * self.num_buttons
                self.joy_pub.publish(msg)

    def print_stats(self):
        now = time.time()
        if now - self.last_stats_time > 10.0:
            total = self.accept_count + self.reject_count
            if total > 0:
                rate = 100.0 * self.accept_count / total
                self.get_logger().info("CRSF frames: %d accepted, %d rejected (%.1f%% accept rate)",
                              self.accept_count, self.reject_count, rate)
            self.accept_count = 0
            self.reject_count = 0
            self.last_stats_time = now
            self.stats_printed = True

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01
            )
            self.get_logger().info("Serial port %s opened at %d baud", self.port, self.baud_rate)
            return True
        except serial.SerialException as e:
            self.get_logger().error("Failed to open serial port: %s", str(e))
            return False

    def run(self):
        if not self.connect_serial():
            return
        # ROS2: rospy.Rate → 별도 thread Rate 또는 sleep loop
        from rclpy.rate import Rate as _Rate
        # 단순화: time.sleep loop (CRSF 시리얼 polling 패턴)
        period = 1.0 / float(self.publish_rate)
        self.get_logger().info("ELRS Joy Node started (no-CRC mode)")
        self.get_logger().info("  Port: %s @ %d baud", self.port, self.baud_rate)
        self.get_logger().info("  Failsafe timeout: %.1fs", self.failsafe_timeout)

        while not False:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.buffer.extend(data)
                    self.parse_crsf_frame()

                if self.connected:
                    self.publish_joy()

                self.check_failsafe()
                if not self.stats_printed:
                    self.print_stats()
                time.sleep(period)

            except serial.SerialException as e:
                self.get_logger().error("Serial error: %s. Reconnecting...", str(e))
                self.connected = False
                time.sleep(1.0)
                self.connect_serial()
            except KeyboardInterrupt:
                break

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ELRSJoyNode()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()