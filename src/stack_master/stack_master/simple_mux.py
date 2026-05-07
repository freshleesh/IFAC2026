#!/usr/bin/env python3

import json
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float64, String
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from copy import deepcopy
### HJ : dynamic_reconfigure for live launch tuning via rqt_reconfigure
class SimpleMuxNode:

    def __init__(self):
        """
        Initialize the node, subscribe to topics, create publishers and set up member variables.
        """

        # Initialize the node
        self.name = "simple_mux"
        rospy.init_node(self.name, anonymous=True)

        self.out_topic  = self._get_param_or_default("/vesc/out_topic", "low_level/ackermann_cmd_mux/output")
        self.in_topic  = self._get_param_or_default("/vesc/in_topic", "high_level/ackermann_cmd_mux/input/nav_1")
        self.joy_topic  = self._get_param_or_default("/vesc/joy_topic", "/joy")
        self.rate_hz = self._get_param_or_default("/vesc/rate_hz", 50.0)
        self.max_speed = self._get_param_or_default("/vesc/joy_max_speed", 4.0)
        self.max_steer = self._get_param_or_default("/vesc/joy_max_steer", 0.4)
        self.joy_freshness_threshold = self._get_param_or_default("/vesc/joy_freshness_threshold", 1.0)


        vesc_servo_min = self._get_param_or_default("/vesc/vesc_driver/servo_min", 1.0)
        vesc_servo_max = self._get_param_or_default("/vesc/vesc_driver/servo_max", 1.0)
        steering_angle_to_servo_offset = self._get_param_or_default("/vesc/steering_angle_to_servo_offset", 1.0)
        steering_angle_to_servo_gain = self._get_param_or_default("/vesc/steering_angle_to_servo_gain", 1.0)

        servo_max_rad = (vesc_servo_max - steering_angle_to_servo_offset) / steering_angle_to_servo_gain
        servo_min_rad = (vesc_servo_min - steering_angle_to_servo_offset) / steering_angle_to_servo_gain

        self.servo_max_abs = min(abs(servo_max_rad), abs(servo_min_rad))

        self.current_host = None
        self.release_time = None  ### HJ : timestamp when buttons released
        self.autodrive_latched = False  ### HJ : RB edge-triggered latch for autodrive
        self.rb_prev = 0  ### HJ : previous RB state for edge detection

        self.human_drive = None
        self.autodrive = None
        self.zero_msg = AckermannDriveStamped()
        self.zero_msg.header.stamp = self.get_clock().now().to_msg()
        self.zero_msg.drive.steering_angle = 0
        self.zero_msg.drive.speed = 0
        self.cur_v = 0
        self.prev_del_v = 0
        self.vel_planner = 0

        ### HJ : launch control parameters (overridable via rosparam)
        # Single-knob constant-current launch: a flat target_I held for launch_t_total seconds.
        # BEMF + duty saturation provide natural ramp. Tune launch_current_A in field calibration.
        self.launch_arm_button = int(self._get_param_or_default("~launch_arm_button", 0))   # A button on most pads
        self.launch_t_total = float(self._get_param_or_default("~launch_t_total", 0.75))    # launch active duration [s]
        self.launch_current_A = float(self._get_param_or_default("~launch_current_A", 15.0))  # SAFE START: equals measured rotation-start; sweep up in calib
        self.launch_safety_floor_margin = float(self._get_param_or_default("~launch_safety_floor_margin", 0.1))  # m/s; speed >= measured + this after launch hand-off
        # cpp jerk==512 path constants (custom_ackermann_to_vesc.cpp:130-131). Mirror them here for inverse mapping.
        self.a2c_gain = float(self._get_param_or_default("/vesc/acceleration_to_current_gain", 10.0))
        self.a2c_baseline = 10.0
        self.v_ff_coeff = 0.9
        self.v_ff_deadzone = 0.3

        ### HJ : launch state
        self.launch_armed = False
        self.launch_active = False
        self.launch_t0 = None
        self.launch_post_done = False  # one-shot: first tick after DONE applies safety floor
        self.a_prev = 0  # arm-button edge tracking
        self.lb_prev = 0  # LB edge tracking (for explicit launch abort logging)

        # Subscribe to the topics
        self.create_subscription(AckermannDriveStamped, self.in_topic, self.drive_callback, 10)
        self.create_subscription(Odometry, "/vesc/odom", self.odom_callback, 10)
        self.create_subscription(Float64, "/vel_planner", self.planner_callback, 10)

        # Do not use filtered velocity
        # self.create_subscription(Odometry, "/car_state/odom", self.odom_callback, 10)

        self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        ### HJ : external abort trigger (e.g. test_control_publisher duration cutoff)
        # Same effect as LB rising: drop autodrive latch, reset launch state, force release_time so timer
        # broadcasts 1s of zero ackermann then idles.
        self.create_subscription(Empty, "/launch_controller/abort", self._abort_cb, queue_size=1, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.out_topic, 10)
        self.current_pub = self.create_publisher(Float64, "/vesc/commands/motor/current", 10)
        self.launch_debug_pub = self.create_publisher(String, "/launch_controller/debug", 10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.timer_callback)

        ### HJ : dynamic_reconfigure server for live launch tuning (rqt_reconfigure → /simple_mux)
        self.dyn_srv = DynRecServer(SimpleMuxConfig, self._dyn_cb)

        self.get_logger().info("[simple_mux] launch ctrl ready: arm_btn=%d, t_total=%.2fs, current=%.1fA, a2c_gain=%.2f",
                      self.launch_arm_button, self.launch_t_total, self.launch_current_A, self.a2c_gain)

    ### HJ : dynamic_reconfigure callback. Refuses to update mid-launch to avoid mid-flight value jumps.
    def _dyn_cb(self, config, level):
        if self.launch_active:
            self.get_logger().warning("[simple_mux] dynamic update ignored: launch active. Wait for DONE.")
            # Echo current values back so rqt shows actual state
            config.launch_current_A = self.launch_current_A
            config.launch_t_total = self.launch_t_total
            config.launch_safety_floor_margin = self.launch_safety_floor_margin
            return config
        self.launch_current_A = float(config.launch_current_A)
        self.launch_t_total = float(config.launch_t_total)
        self.launch_safety_floor_margin = float(config.launch_safety_floor_margin)
        self.get_logger().info("[simple_mux] dyn updated: current=%.1fA, t_total=%.2fs, floor=%.2fm/s",
                      self.launch_current_A, self.launch_t_total, self.launch_safety_floor_margin)
        return config

    def check_uptodate(self, drive_msg):
        # return True
        if drive_msg is None:
            return False

        if abs(drive_msg.header.stamp.to_sec() - self.get_clock().now().to_msg().to_sec()) < self.joy_freshness_threshold:
            return True
        else:
            return False

    def clip_servo(self, in_drive_msg):
        drive_msg = AckermannDriveStamped()
        drive_msg = deepcopy(in_drive_msg)

        if drive_msg.drive.steering_angle > 0 and drive_msg.drive.steering_angle > self.servo_max_abs:
            drive_msg.drive.steering_angle = self.servo_max_abs
        elif drive_msg.drive.steering_angle < 0 and drive_msg.drive.steering_angle < - self.servo_max_abs:
            drive_msg.drive.steering_angle = -self.servo_max_abs

        return drive_msg

    ### HJ : invert cpp's i_cmd = a2c_gain*accel + 0.9*v + 10 to make cpp publish exactly target_I.
    def _accel_for_target_current(self, target_I):
        v_ff = self.v_ff_coeff * self.cur_v if self.cur_v > self.v_ff_deadzone else 0.0
        return (target_I - self.a2c_baseline - v_ff) / max(self.a2c_gain, 1e-6)

    ### HJ : apply launch override on top of an autodrive ackermann.
    # Steering is NEVER overridden. Only speed/jerk/acceleration get touched.
    # During launch active: jerk=512 with inverse-mapped accel so cpp emits exactly launch_current_A.
    # First tick after DONE: pass-through but enforce safety floor (speed >= measured + small margin)
    # to defend against the edge case where controller's first speed cmd happens to be below measured.
    def _apply_launch_override(self, drive_msg):
        if self.launch_active and self.launch_t0 is not None:
            t = (self.get_clock().now().to_msg() - self.launch_t0).to_sec()
            if t >= self.launch_t_total:
                # DONE: hand off to controller, ENABLE one-shot safety floor for next tick(s)
                self.launch_active = False
                self.launch_t0 = None
                self.launch_post_done = True
                self._launch_done_until = self.get_clock().now().to_msg() + rospy.Duration(0.10)  # 100ms safety window
                self._publish_launch_debug(t, "DONE", drive_msg, target_I=0.0)
                # fall through to safety-floor block below
            else:
                target_I = self.launch_current_A
                accel = self._accel_for_target_current(target_I)
                drive_msg.drive.acceleration = accel
                drive_msg.drive.jerk = 512
                drive_msg.drive.speed = 0  # cpp ignores speed when jerk==512
                self._publish_launch_debug(t, "ACTIVE", drive_msg, target_I=target_I)
                return drive_msg

        # Post-launch safety window: enforce speed floor so VESC speed PID gets positive error
        if self.launch_post_done:
            if self.get_clock().now().to_msg() < self._launch_done_until:
                drive_msg.drive.speed = max(drive_msg.drive.speed, self.cur_v + self.launch_safety_floor_margin)
                self._publish_launch_debug(0.0, "POST", drive_msg, target_I=0.0)
            else:
                self.launch_post_done = False
        return drive_msg

    def _publish_launch_debug(self, t, mode, drive_msg, target_I=0.0):
        d = {
            "t": round(float(t), 3),
            "armed": bool(self.launch_armed),
            "active": bool(self.launch_active),
            "post_done": bool(self.launch_post_done),
            "mode": mode,
            "jerk": int(drive_msg.drive.jerk),
            "speed_cmd": round(float(drive_msg.drive.speed), 3),
            "accel_cmd": round(float(drive_msg.drive.acceleration), 4),
            "target_I_A": round(float(target_I), 2),
            "measured_v": round(float(self.cur_v), 3),
        }
        self.launch_debug_pub.publish(String(data=json.dumps(d)))

    ### HJ : always-on heartbeat for armed/active state, even when no autodrive ackermann is flowing.
    # Without this, /launch_controller/debug only ticks during launch active, leaving any subscriber's
    # cached `armed` value stale until the first launch fire.
    def _publish_idle_state(self):
        d = {
            "t": 0.0,
            "armed": bool(self.launch_armed),
            "active": bool(self.launch_active),
            "post_done": bool(self.launch_post_done),
            "mode": "IDLE",
            "jerk": 0,
            "speed_cmd": 0.0,
            "accel_cmd": 0.0,
            "target_I_A": 0.0,
            "measured_v": round(float(self.cur_v), 3),
        }
        self.launch_debug_pub.publish(String(data=json.dumps(d)))

    def timer_callback(self, event):
        ### HJ : always publish armed/active heartbeat at 50Hz so test_publisher / debug subs never go stale.
        # _apply_launch_override path will publish a richer message during launch ACTIVE/POST.
        if not self.launch_active and not self.launch_post_done:
            self._publish_idle_state()

        ### HJ : when no button is held, publish zero for 1s then stop
        if self.current_host is None:
            if self.release_time is not None and (self.get_clock().now().to_msg() - self.release_time).to_sec() < 1.0:
                self.zero_msg.header.stamp = self.get_clock().now().to_msg()
                self.drive_pub.publish(self.zero_msg)
            return
        if self.current_host == "autodrive" and self.check_uptodate(self.autodrive):
            drive_msg = self.clip_servo(self.autodrive)
            ### HJ : was *= 1.1 (hidden understeer band-aid); set to 1.0 to expose true linkage,
            ### HJ : nonlinear servo cal (steering_servo_poly_coeffs) will absorb the linkage shape.
            drive_msg.drive.steering_angle *= 1.0
            ### HJ : launch override (only when armed+launched). pass-through otherwise.
            drive_msg = self._apply_launch_override(drive_msg)
            self.drive_pub.publish(drive_msg)
        elif self.current_host == "humandrive" and self.check_uptodate(self.human_drive):
            drive_msg = self.clip_servo(self.human_drive)
            # if drive_msg.drive.speed > 0 and self.cur_v < 3.0:
            #     # current_msg = Float64()
            #     # current_msg.data = 50.0
            #     # self.current_pub.publish(current_msg)
            #     # self.get_logger().warning(f"joy_command : {drive_msg.drive.speed}" )

            #     drive_msg.drive.speed = 6.0
            #     # self.drive_pub.publish(drive_msg)
            # else:
            self.drive_pub.publish(drive_msg)
            # self.human_drive = None
        # else:
        #     self.drive_pub.publish(self.zero_msg)
    ### HJ : equivalent of LB press from external client.
    def _abort_cb(self, _msg):
        self.get_logger().warning("[launch] ABORT(external) armed=%s active=%s -> humandrive idle",
                      self.launch_armed, self.launch_active)
        # Reset all launch state
        self.launch_armed = False
        self.launch_active = False
        self.launch_t0 = None
        self.launch_post_done = False
        # Drop autodrive latch and switch to humandrive zero so motor coasts immediately
        self.autodrive_latched = False
        self.human_drive = self.zero_msg
        if self.current_host is not None:
            self.release_time = self.get_clock().now().to_msg()
        self.current_host = None  # timer_callback's release_time path will broadcast 1s of zero ack

    def planner_callback(self, msg):
        self.vel_planner = msg.data

    def odom_callback(self, msg):
        self.cur_v = msg.twist.twist.linear.x

    def joy_callback(self, msg):
        # prev_host = deepcopy(self.current_host)
        use_human_drive = msg.buttons[4]
        rb_pressed = msg.buttons[5]

        ### HJ : RB rising edge → latch autodrive
        rb_edge_rising = rb_pressed and not self.rb_prev
        self.rb_prev = rb_pressed

        ### HJ : LB rising edge → explicitly abort any in-flight launch (defensive; humandrive already overrides via mux)
        lb_rising = use_human_drive and not self.lb_prev
        self.lb_prev = use_human_drive
        if lb_rising and (self.launch_active or self.launch_armed):
            self.get_logger().warning("[launch] ABORT(LB) armed=%s active=%s -> reset both", self.launch_armed, self.launch_active)
            self.launch_armed = False
            self.launch_active = False
            self.launch_t0 = None
            self.launch_post_done = False

        ### HJ : arm button (default A=button[0]) — toggle armed only when not currently launching
        if len(msg.buttons) > self.launch_arm_button:
            a_pressed = msg.buttons[self.launch_arm_button]
            a_rising = a_pressed and not self.a_prev
            self.a_prev = a_pressed
            if a_rising and not self.launch_active:
                old = self.launch_armed
                self.launch_armed = not self.launch_armed
                ### HJ : full button snapshot to identify cross-channel triggers
                btn_snap = ",".join(str(b) for b in msg.buttons[:8])
                self.get_logger().info("[launch] A_TOGGLE %s->%s (a_pressed=%d a_prev_was=%d, btns0-7=[%s])",
                              old, self.launch_armed, int(a_pressed), int(not a_pressed) ^ 0, btn_snap)

        if use_human_drive:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = msg.axes[3] * self.max_steer
            drive_msg.drive.speed = msg.axes[1] * self.max_speed


            # drive_msg.drive.jerk = 512.0
            # del_v = drive_msg.drive.speed -self.cur_v
            # drive_msg.drive.acceleration = del_v * 4.0 + (del_v - self.prev_del_v) * 0.1

            self.human_drive = drive_msg
            self.current_host = "humandrive"
            ### HJ : LB takeover clears autodrive latch; need RB re-press to resume
            self.autodrive_latched = False
        elif rb_edge_rising:
            ### HJ : RB pressed → latch autodrive ON
            self.autodrive_latched = True
            self.current_host = "autodrive"
            ### HJ : trace whether RB rising sees armed=True (fire) or False (no-fire)
            btn_snap = ",".join(str(b) for b in msg.buttons[:8])
            self.get_logger().info("[launch] RB_RISING armed=%s active=%s (btns0-7=[%s])",
                          self.launch_armed, self.launch_active, btn_snap)
            ### HJ : if armed, fire launch one-shot. armed -> False, active -> True at this exact tick.
            if self.launch_armed and not self.launch_active:
                self.launch_armed = False
                self.launch_active = True
                self.launch_t0 = self.get_clock().now().to_msg()
                self.get_logger().warning("[launch] FIRED at t=%.3f, current=%.1fA for %.2fs (one-shot armed->False)",
                              self.launch_t0.to_sec(), self.launch_current_A, self.launch_t_total)
        elif self.autodrive_latched:
            ### HJ : latch held → stay in autodrive even when RB released
            self.current_host = "autodrive"
        else:
            ### HJ : no active control → zero out for 1s then idle
            if self.current_host is not None:
                self.release_time = self.get_clock().now().to_msg()
            self.human_drive = self.zero_msg
            self.current_host = None

    # def drive_callback(self, msg):
    #     self.autodrive = msg


    def drive_callback(self, msg):
        drive_msg = AckermannDriveStamped()
        drive_msg = msg
        # drive_msg.drive.speed = self.vel_planner
        # drive_msg.drive.jerk = 512.0
        # del_v = drive_msg.drive.speed -self.cur_v
        # drive_msg.drive.acceleration = del_v * 4.0 + (del_v - self.prev_del_v) * 0.1
        # if drive_msg.drive.acceleration >0:
        #     drive_msg.drive.steering_angle *= (1+drive_msg.drive.acceleration*0.4)
        # else:
        #     drive_msg.drive.steering_angle *= (1+drive_msg.drive.acceleration*0.2)

        # self.prev_del_v = del_v

        self.autodrive = drive_msg



if __name__ == '__main__':
    simple_mux = SimpleMuxNode()
    rospy.spin()
