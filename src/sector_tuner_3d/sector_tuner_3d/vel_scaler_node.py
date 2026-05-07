#! /usr/bin/env python3

import numpy as np
import yaml
import matplotlib.pyplot as plt
from f110_msgs.msg import WpntArray
from std_msgs.msg import Bool


class VelocityScaler:
    """
    Sector scaler for the velocity of the global waypoints
    """

    def __init__(self, debug_plot: bool = False) -> None:
        self.debug_plot = self._get_param_or_default("/velocity_scaler/debug_plot", False)

        # sectors params
        self.glb_wpnts_og = None
        self.glb_wpnts_scaled = None
        self.glb_wpnts_sp_og = None
        self.glb_wpnts_sp_scaled = None
        self.update_map = False

        # get initial scaling
        pkg_path = rospkg.get_package_share_directory("stack_master")
        map_name = self._get_param_or_default('/map')
        yaml_file_path = pkg_path + "/maps/" + map_name + "/speed_scaling.yaml" 
        with open(yaml_file_path, "r") as file:
            self.sectors_params = yaml.safe_load(file)
        
        self.n_sectors = self.sectors_params['n_sectors']

        # dyn params sub
        self.glb_wpnts_name = "/global_waypoints"
        self.create_subscription(Config, "/dyn_sector_tuner/speed/parameter_updates", self.dyn_param_cb, 10)
        self.create_subscription(WpntArray, self.glb_wpnts_name, self.glb_wpnts_cb, 10)
        self.create_subscription(WpntArray, self.glb_wpnts_name+"/shortest_path", self.glb_wpnts_sp_cb, 10)
        self.create_subscription(Bool, "update_map", self.update_map_cb, 10)

        # new glb_waypoints pub
        self.scaled_points_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled", 10)
        self.scaled_points_sp_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled/shortest_path", 10)

    def update_map_cb(self, data:Bool):
        self.update_map = True
        
    def glb_wpnts_cb(self, data:WpntArray):
        """
        Saves the global waypoints
        """
        self.glb_wpnts_og = data

    def glb_wpnts_sp_cb(self, data:WpntArray):
        """
        Saves the global waypoints
        """
        self.glb_wpnts_sp_og = data

    def dyn_param_cb(self, params:Config):
        """
        Notices the change in the parameters and scales the global waypoints
        """
        # get global limit
        self.sectors_params['global_limit'] = params.doubles[0].value

        # update params 
        for i in range(self.n_sectors):
            self.sectors_params[f"Sector{i}"]['scaling'] = np.clip(
                params.doubles[i+1].value, 0, self.sectors_params['global_limit']
            )

        self.get_logger().info(self.sectors_params)

    def get_vel_scaling(self, s):
        """
        Gets the dynamically reconfigured velocity scaling for the points.
        Linearly interpolates for points between two sectors
        
        Parameters
        ----------
        s
            s parameter whose sector we want to find
        """
        hl_change = 10

        if self.n_sectors > 1:
            for i in range(self.n_sectors):
                if i == 0 :
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{self.n_sectors-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                    )
                elif i != self.n_sectors-1:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                    )
                else:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i}']['end']-hl_change, self.sectors_params[f'Sector{i}']['end']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{0}']['scaling']]
                    )
        elif self.n_sectors == 1:
            scaler = self.sectors_params["Sector0"]['scaling']

        return scaler

    def scale_points(self):
        """
        Scales the global waypoints' velocities
        """
        scaling = []

        if self.glb_wpnts_scaled is None or self.update_map:
            self.glb_wpnts_scaled = self.glb_wpnts_og
            self.glb_wpnts_sp_scaled = self.glb_wpnts_sp_og
            self.update_map = False

        for i, wpnt  in enumerate(self.glb_wpnts_og.wpnts):
            vel_scaling = self.get_vel_scaling(i)
            new_vel = wpnt.vx_mps*vel_scaling
            self.glb_wpnts_scaled.wpnts[i].vx_mps = new_vel
            scaling.append(self.get_vel_scaling(i))

        if self.debug_plot:
            plt.clf()
            plt.plot(scaling)
            plt.legend(['og', 'scaled'])
            plt.ylim(0,1)
            plt.show()


    def loop(self):
        self.get_logger().info("Waiting for global waypoints...")
        rospy.wait_for_message(self.glb_wpnts_name, WpntArray)
        self.get_logger().info("Global waypoints received!")

        # initialise scaled points
        self.scale_points()

        run_rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.scale_points()
            self.scaled_points_pub.publish(self.glb_wpnts_scaled)
            run_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("vel_scaler")
    vel_scaler = VelocityScaler()
    vel_scaler.loop()
