#!/usr/bin/env python3

import numpy as np
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from ackermann_msgs.msg import AckermannDriveStamped


class VelAnalyser:
    def __init__(self) -> None:
        # attributes
        self.gb_wpnt_sc = None
        self.frenet_odom = None
        self.pub_cur_vel_flag = True
        self.wheelbase = 0.321 

        # subscribers
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_wpnt_sc_cb, 10)
        rospy.Subscriber(
            "/dyn_sector_speed/parameter_updates", Config, self.dyn_par_cb
        )
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_odom_cb, 10)
        rospy.Subscriber(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
            AckermannDriveStamped,
            self.ctrl_cb,
        )

        # publishers
        self.vel_traj_pub = rospy.Publisher(
            "velocity/trajectory", Point, queue_size=1000
        )
        self.steer_traj_pub = rospy.Publisher(
            "steering/trajectory", Point, queue_size=1000
        )
        self.vel_cur_pub = self.create_publisher(Point, "velocity/current", 10)
        self.steer_cur_pub = self.create_publisher(Point, "steering_input", 10)

        self.dyn_par_cb(None)

    def frenet_odom_cb(self, data: Odometry):
        self.frenet_odom = data

        cur_s = self.frenet_odom.pose.pose.position.x
        cur_v = self.frenet_odom.twist.twist.linear.x
        if np.round(cur_s / 0.05, 1) % 1 == 0.0:  # take points at 0.05 m distance
            if self.pub_cur_vel_flag:
                self.vel_cur_pub.publish(Point(x=cur_s, y=cur_v))
                self.steer_cur_pub.publish(Point(x=cur_s, y=self.steer_input))
                self.pub_cur_vel_flag = False
        else:
            if not self.pub_cur_vel_flag:
                self.pub_cur_vel_flag = True

    def gb_wpnt_sc_cb(self, data: WpntArray):
        self.gb_wpnt_sc = data.wpnts

    def dyn_par_cb(self, data: object):
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)

        rate = rospy.Rate(2000)
        for wpnt in self.gb_wpnt_sc:
            self.vel_traj_pub.publish(Point(x=wpnt.s_m, y=wpnt.vx_mps))
            self.steer_traj_pub.publish(
                Point(x=wpnt.s_m, y=np.arctan(self.wheelbase*wpnt.kappa_radpm))
            )
            rate.sleep()  # needed for publishing correctly all the points

    def ctrl_cb(self, data: AckermannDriveStamped):
        self.steer_input = data.drive.steering_angle

    def loop(self):
        # this is not the loop of the node
        # the sleep is only needed to not avoid misses in publishing the messages
        rate = rospy.Rate(2000)
        for wpnt in self.gb_wpnt_sc:
            self.vel_traj_pub.publish(Point(x=wpnt.s_m, y=wpnt.vx_mps))
            self.steer_traj_pub.publish(
                Point(x=wpnt.s_m, y=np.arctan(self.wheelbase*wpnt.kappa_radpm))
            )
            rate.sleep()
