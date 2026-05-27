#!/usr/bin/env python3
import yaml
import numpy as np
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker, MarkerArray

class StaticObstacleSectorPublisher:
    def __init__(self):
        self.srv = Server(static_obs_dyn_sect_tunerConfig, self.callback)
        # Get corresponding yaml file path
        self.sectors = None
        pkg_path = rospkg.get_package_share_directory("stack_master")
        map_name = self._get_param_or_default('/map')
        self.yaml_file_path = pkg_path + "/maps/" + map_name + "/static_obs_sectors.yaml" 
        self.yaml_data = self.get_yaml_values(self.yaml_file_path)
        self.default_config = self.decode_yaml(self.yaml_data)
        self.srv.update_configuration(self.default_config)
        
        self.glb_waypoints = None
        self.sector_pub = self.create_publisher(MarkerArray, '/static_obs_sector_markers', 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.glb_wpnts_cb, 10)
        
    def callback(self, config, level):
        if config.save_params:
            self.save_yaml(config)
            config.save_params = False
        
        return config
    
    def save_yaml(self, config):
        try:
            for key, item in self.sectors.items():
                self.yaml_data[key]['static_obs_section'] = bool(getattr(config, key, None)) 
                
            with open(self.yaml_file_path, "w") as yaml_file:
                # Add sort_keys=False to preserve the original order
                yaml.dump(self.yaml_data, yaml_file, default_flow_style=False, sort_keys=False)
            self.get_logger().info("Configuration saved to YAML file: %s", self.yaml_file_path)

        except Exception as e:
            self.get_logger().error("Failed to save configuration to YAML: %s", str(e))
            
    def get_yaml_values(self, yaml_file_path):
        # Get and return data
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        return data

    def decode_yaml(self, yaml_data):
        default_config = {}
        self.sectors = {k: v for k, v in yaml_data.items() if k.startswith('Static_Obs_sector')}

        for key, item in self.sectors.items():
            # Use .get() for safety in case the key is missing
            default_flag = item.get('static_obs_section', False)
            default_config[key] = bool(default_flag)          
        return default_config
    
    def glb_wpnts_cb(self, data):
        self.glb_waypoints = []
        for waypoint in data.wpnts:
            self.glb_waypoints.append([waypoint.x_m, waypoint.y_m, waypoint.s_m])

    def pub_sector_markers(self):
        rate = rospy.Rate(1)
        while (not rospy.is_shutdown()):
            if self.glb_waypoints is None:
                rate.sleep()
                continue

            n_sectors = self.yaml_data['n_sectors']
            sec_markers = MarkerArray()

            for i in range(n_sectors):
                s = self.yaml_data[f"Static_Obs_sector{i}"]['start']
                if s == (len(self.glb_waypoints) - 1):
                    theta = np.arctan2((self.glb_waypoints[0][1] - self.glb_waypoints[s][1]),(self.glb_waypoints[0][0] - self.glb_waypoints[s][0]))
                else:
                    theta = np.arctan2((self.glb_waypoints[s+1][1] - self.glb_waypoints[s][1]),(self.glb_waypoints[s+1][0] - self.glb_waypoints[s][0]))
                quaternions = quaternion_from_euler(0, 0, theta)
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = marker.ARROW
                marker.scale.x = 0.5
                marker.scale.y = 0.05
                marker.scale.z = 0.15
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.pose.position.x = self.glb_waypoints[s][0]
                marker.pose.position.y = self.glb_waypoints[s][1]
                marker.pose.position.z = 0
                marker.pose.orientation.x = quaternions[0]
                marker.pose.orientation.y = quaternions[1]
                marker.pose.orientation.z = quaternions[2]
                marker.pose.orientation.w = quaternions[3]
                marker.id = i
                sec_markers.markers.append(marker)

                marker_text = Marker()
                marker_text.header.frame_id = "map"
                marker_text.header.stamp = self.get_clock().now().to_msg()
                marker_text.type = marker_text.TEXT_VIEW_FACING
                marker_text.text = f"Start Static Obs Sector {i}"
                marker_text.scale.z = 0.4
                marker_text.color.r = 0.1
                marker_text.color.g = 0.1
                marker_text.color.b = 1.2
                marker_text.color.a = 1.0
                marker_text.pose.position.x = self.glb_waypoints[s][0]
                marker_text.pose.position.y = self.glb_waypoints[s][1]
                marker_text.pose.position.z = 1.5
                marker_text.pose.orientation.x = 0.0
                marker_text.pose.orientation.y = 0.0
                marker_text.pose.orientation.z = 0.0436194
                marker_text.pose.orientation.w = 0.9990482
                marker_text.id = i + n_sectors
                sec_markers.markers.append(marker_text)
            self.sector_pub.publish(sec_markers)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("dynamic_static_obs_sector_tuner", anonymous=False)
    print('Dynamic Static Obs Sector Server Launched...')
    sec_pub = StaticObstacleSectorPublisher()
    sec_pub.pub_sector_markers()