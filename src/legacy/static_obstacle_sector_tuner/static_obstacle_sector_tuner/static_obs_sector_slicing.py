#!/usr/bin/env python3

import yaml, os, subprocess, time
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arrow

class StaticObstacleSectorSlicer:
    """
    Node for listening to global waypoints and running a GUI to tune the static obstacle sectors,
    exporting them to a yaml file in the original key-based format.
    """
    def __init__(self):
        rospy.init_node('static_obs_sector_slicer_node', anonymous=True)

        self.glb_wpnts = None
        self.track_bounds = None

        self.glob_slider_idx = 0
        self.sector_pnts_indices = [0] 

        self.create_subscription(WpntArray, '/global_waypoints', self.glb_wpnts_cb, 10)
        self.create_subscription(MarkerArray, '/trackbounds/markers', self.bounds_cb, 10)

        self.yaml_dir = self._get_param_or_default('~save_dir')

    def glb_wpnts_cb(self, data):
        self.glb_wpnts = data
    
    def bounds_cb(self, data):
        self.track_bounds = data

    def slice_loop(self):
        self.get_logger().info('Waiting for global waypoints...')
        rospy.wait_for_message('/global_waypoints', WpntArray)
        self.get_logger().info('Waypoints received. Starting GUI.')

        self.sector_gui()
        self.get_logger().info(f'Selected Static Obstacle Sector Indices: {self.sector_pnts_indices}')

        self.sectors_to_yaml()

    def sector_gui(self):
        s_m_list = [wpnt.s_m for wpnt in self.glb_wpnts.wpnts]
        x_m_list = [wpnt.x_m for wpnt in self.glb_wpnts.wpnts]
        y_m_list = [wpnt.y_m for wpnt in self.glb_wpnts.wpnts]
        
        s_m = np.array(s_m_list)
        x_m = np.array(x_m_list)
        y_m = np.array(y_m_list)

        fig, (ax1, axslider, axselect, axfinish) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 1, 1, 1]})
        
        def update_plot_elements(ax):
            ax.cla()
            ax.plot(x_m, y_m, "m-", linewidth=0.7) 
            if self.track_bounds:
                ax.plot([mrk.pose.position.x for mrk in self.track_bounds.markers], [mrk.pose.position.y for mrk in self.track_bounds.markers], 'g-', linewidth=0.4)
            
            ax.grid()
            ax.set_aspect("equal", "datalim")
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            arr_par = {'x': x_m[0], 'dx': 10 * (x_m[1] - x_m[0]), 'y': y_m[0], 'dy': 10 * (y_m[1] - y_m[0]), 'color': 'gray', 'width': 0.5}
            ax.add_artist(Arrow(**arr_par))

        def update_s(val):
            self.glob_slider_idx = int(slider.val)
            update_map(cur_idx=self.glob_slider_idx)
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts_indices.append(self.glob_slider_idx)
            update_map(cur_idx=self.glob_slider_idx)
        
        def finish(event):
            plt.close()
            self.sector_pnts_indices.append(len(s_m) - 1)
            self.sector_pnts_indices = sorted(list(set(self.sector_pnts_indices)))
            return

        def update_map(cur_idx):
            update_plot_elements(ax1)
            ax1.scatter(x_m[cur_idx], y_m[cur_idx], c='red', zorder=10)
            if len(self.sector_pnts_indices) > 0:
                ax1.scatter(x_m[self.sector_pnts_indices], y_m[self.sector_pnts_indices], c='green', zorder=10)

        update_plot_elements(ax1)

        slider = Slider(axslider, 'Waypoint Index', 0, len(s_m)-1, valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select Static Obstacle S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)
        
        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts_indices) <= 1:
            self.get_logger().warning("No sectors selected. Creating a single sector for the whole track.")
            self.sector_pnts_indices = [0, len(self.glb_wpnts.wpnts)-1]
            
        n_sectors = len(self.sector_pnts_indices) - 1
        
        dict_file = {'n_sectors': n_sectors}
        
        for i in range(n_sectors):
            start_idx = self.sector_pnts_indices[i]
            end_idx = self.sector_pnts_indices[i+1]
            
            s_start = self.glb_wpnts.wpnts[start_idx].s_m
            s_end = self.glb_wpnts.wpnts[end_idx].s_m

            sector_key = f"Static_Obs_sector{i}"
            dict_file[sector_key] = {
                'start': int(start_idx),
                'end': int(end_idx),     
                's_start': float(s_start),
                's_end': float(s_end),
                'name': f"sector_{i+1}",
                'static_obs_section': False
            }
        
        yaml_path = os.path.join(self.yaml_dir, 'static_obs_sectors.yaml')
        with open(yaml_path, 'w') as file:
            self.get_logger().info(f'Dumping to {yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        ros_path = rospkg.get_package_share_directory('static_obstacle_sector_tuner')
        cfg_yaml_path = os.path.join(ros_path, 'cfg/static_obs_sectors.yaml')
        with open(cfg_yaml_path, 'w') as file:
            self.get_logger().info(f'Dumping to {cfg_yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        time.sleep(1)
        self.get_logger().info('Copying yaml and building catkin workspace...')
        shell_dir = os.path.join(ros_path, 'scripts/finish_sector.sh')
        subprocess.Popen(shell_dir, shell=True)

if __name__ == "__main__":
    slicer = StaticObstacleSectorSlicer()
    slicer.slice_loop()