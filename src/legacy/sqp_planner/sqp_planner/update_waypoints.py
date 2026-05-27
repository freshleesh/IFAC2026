#!/usr/bin/env python3

import numpy as np
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray, Marker
from frenet_conversion_msgs.srv import Frenet2GlobArr
from std_msgs.msg import String
import time


class UpdateWaypoints:
    def __init__(self):
        # Initialize the node
        rospy.init_node('waypoint_updater', anonymous=True)

        # Waypoint publisher
        self.wpnts_updated_pub = self.create_publisher(WpntArray, "/global_waypoints_updated", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/updated_waypoints_marker", 10)

        # Init and params
        self.frenet2glob = rospy.ServiceProxy("convert_frenet2globarr_service", Frenet2GlobArr)

        # Adaptive rate would be nice
        self.loop_rate = 1 #Hz

        # Parameters
        self.state = "GB_TRACK"
        self.hysteresis_time = 2.0 # s
        self.gb_track_start_time = None
        self.max_speed_scaled = 10.0 # m/s
        self.speed_offset = 0 # m/s

        # Callback data
        self.wpnts_scaled_msg = WpntArray()
        self.wpnts_updated_msg = WpntArray()
        self.s_points_array = np.array([])
        self.update_waypoints = True

        # Subscriber
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.odom_cb, 10)
        self.create_subscription(String, "/state_machine", self.state_machine_cb, 10)
        self.create_subscription(Config, "/dynamic_prediction_tuner_node/parameter_updates", self.dyn_param_cb, 10)
        
### Callbacks ###
    def state_machine_cb(self, data: String):
        self.state = data.data
        if self.state == "GB_TRACK" and self.gb_track_start_time is None:
            self.gb_track_start_time = time.time()
        elif self.state != "GB_TRACK":
            self.gb_track_start_time = None

    def odom_cb(self, data: Odometry):
        car_odom = data
        if self.update_waypoints == True:
            if self.s_points_array.any():
                ego_position = car_odom.pose.pose.position.x
                ego_speed = car_odom.twist.twist.linear.x
                ego_approx_indx = np.abs(self.s_points_array - ego_position).argmin()
                # Hysteresis added to prevent the waypoints from being updated to soon afer switching to GB_TRACK
                if self.state == "GB_TRACK" and self.gb_track_start_time is not None and time.time() - self.gb_track_start_time >= self.hysteresis_time:   
                    self.wpnts_updated_msg.wpnts[ego_approx_indx].vx_mps = ego_speed + self.speed_offset
                    if ego_approx_indx == 0 or ego_approx_indx == (len(self.wpnts_updated_msg.wpnts) - 1): # First and last waypoint are the same
                        self.wpnts_updated_msg.wpnts[0].vx_mps
                        self.wpnts_updated_msg.wpnts[-1].vx_mps
        else:
            pass

    def dyn_param_cb(self, params: object):
        self.update_waypoints = self._get_param_or_default("dynamic_prediction_tuner_node/update_waypoints", True)
        self.speed_offset = self._get_param_or_default("dynamic_prediction_tuner_node/speed_offset")

        print(
            f"[Opp. Pred.] Toggled update waypoints"
            f"[Opp. Pred.] Speed offset: {self.speed_offset}"
        )

    ### Helper functions ###
    def visualize_waypoints(self):
        marker_array = MarkerArray()
        for i in range(len(self.wpnts_updated_msg.wpnts)):
            marker = Marker(header = rospy.Header(frame_id="map"), id = i, type = Marker.CYLINDER)
            marker.pose.position.x = self.wpnts_updated_msg.wpnts[i].x_m
            marker.pose.position.y = self.wpnts_updated_msg.wpnts[i].y_m
            
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = self.wpnts_updated_msg.wpnts[i].vx_mps/self.max_speed_scaled
            marker.pose.position.z = marker.scale.z/2
               
            marker.color.a = 1
            marker.color.g = 1.0
                
            marker_array.markers.append(marker)
    
        self.marker_pub.publish(marker_array)
    
    ### Main loop ###
    def loop(self):
        rate = rospy.Rate(self.loop_rate)
        self.get_logger().info("[Update Wpnts] Update Wpnts wating...")
        self.wpnts_scaled_msg = rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        self.wpnts_updated_msg = self.wpnts_scaled_msg
        self.global_wpnts_msg = rospy.wait_for_message("/global_waypoints", WpntArray)
        self.max_speed_scaled = max([self.global_wpnts_msg.wpnts[i].vx_mps for i in range(len(self.global_wpnts_msg.wpnts))])
        self.s_points_array = np.array([wpnt.s_m for wpnt in self.wpnts_updated_msg.wpnts])
        self.get_logger().info("[Update Wptns] Update Wpnts ready!")

        while not rospy.is_shutdown():
            self.wpnts_updated_msg.header.stamp = self.get_clock().now().to_msg()
            self.wpnts_updated_pub.publish(self.wpnts_updated_msg)
            self.visualize_waypoints()

            rate.sleep()

if __name__ == '__main__':
    node = UpdateWaypoints()
    node.loop()