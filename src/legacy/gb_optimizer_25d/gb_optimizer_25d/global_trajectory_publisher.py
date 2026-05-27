#!/usr/bin/env python3

from rospy import loginfo
from std_msgs.msg import String, Float32
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray

from readwrite_global_waypoints import read_global_waypoints

class GlobalRepublisher:
    """
    Node for publishing the global waypoints/markers and track bounds markers frequently after they have been calculated
    """
    def __init__(self):

        rospy.init_node('global_republisher_node', anonymous=True)

        self.glb_markers = None
        self.glb_wpnts = None
        self.track_bounds = None
        ### HJ : velocity markers for 3D tracks
        self.glb_vel_markers = None
        ### HJ : end

        self.create_subscription(WpntArray, '/global_waypoints', self.glb_wpnts_cb, 10)
        self.create_subscription(MarkerArray, '/global_waypoints/markers', self.glb_markers_cb, 10)
        self.create_subscription(MarkerArray, '/trackbounds/markers', self.bounds_cb, 10)

        self.glb_wpnts_pub = self.create_publisher(WpntArray, 'global_waypoints', 10)
        self.glb_markers_pub = self.create_publisher(MarkerArray, 'global_waypoints/markers', 10)
        self.vis_track_bnds = self.create_publisher(MarkerArray, 'trackbounds/markers', 10)
        ### HJ : velocity markers publisher
        self.glb_vel_markers_pub = self.create_publisher(MarkerArray, 'global_waypoints/vel_markers', 10)
        ### HJ : end

        # shortest_path
        self.glb_sp_markers = None
        self.glb_sp_wpnts = None
        self.create_subscription(WpntArray, '/global_waypoints/shortest_path', self.glb_sp_wpnts_cb, 10)
        self.create_subscription(MarkerArray, '/global_waypoints/shortest_path/markers', self.glb_sp_markers_cb, 10)
        self.glb_sp_wpnts_pub = self.create_publisher(WpntArray, 'global_waypoints/shortest_path', 10)
        self.glb_sp_markers_pub = self.create_publisher(MarkerArray, 'global_waypoints/shortest_path/markers', 10)

        # centerline
        self.centerline_wpnts = None
        self.centerline_markers = None
        self.create_subscription(WpntArray, '/centerline_waypoints', self.centerline_wpnt_cb, 10)
        self.create_subscription(MarkerArray, '/centerline_waypoints/markers', self.centerline_markers_cb, 10)
        self.centerline_wpnts_pub = self.create_publisher(WpntArray, '/centerline_waypoints', 10)
        self.centerline_markers_pub = self.create_publisher(MarkerArray, '/centerline_waypoints/markers', 10)

        # map infos
        self.map_infos = None
        self.create_subscription(String, '/map_infos', self.map_info_cb, 10)
        self.map_info_pub = self.create_publisher(String, 'map_infos', 10)
        
        self.est_lap_time = None
        self.create_subscription(Float32, 'estimated_lap_time', self.est_lap_time_cb, 10)
        self.est_lap_time_pub = self.create_publisher(Float32, 'estimated_lap_time', 10)


        # graph lattice 
        self.graph_lattice = None
        self.create_subscription(MarkerArray, '/lattice_viz', self.lattice_cb, 10)
        self.lattice_pub = self.create_publisher(MarkerArray, '/lattice_viz', 10)
        # Read info from json file if it is provided, so everything is always published
        if rospy.has_param('/global_republisher/map'):
            map_name = self._get_param_or_default('/global_republisher/map')
            loginfo(f"Reading parameters from {map_name}")

            try:
                self.map_infos, self.est_lap_time, self.centerline_markers, self.centerline_wpnts,\
                self.glb_markers, self.glb_wpnts,\
                self.glb_sp_markers, self.glb_sp_wpnts, \
                self.track_bounds, self.glb_vel_markers = read_global_waypoints(map_name)
            except FileNotFoundError:
                self.get_logger().warning(f"{map_name} param not found. Not publishing")
        else:
            loginfo(f"global_trajectory_publisher did not find any map_name param")

    def glb_wpnts_cb(self, data):
        self.glb_wpnts = data
        track_length = data.wpnts[-1].s_m
        rospy.set_param('global_republisher/track_length', track_length)

    def glb_markers_cb(self, data):
        self.glb_markers = data

    def glb_sp_wpnts_cb(self, data):
        self.glb_sp_wpnts = data

    def glb_sp_markers_cb(self, data):
        self.glb_sp_markers = data

    def centerline_wpnt_cb(self, data: WpntArray):
        self.centerline_wpnts = data

    def centerline_markers_cb(self, data: MarkerArray):
        self.centerline_markers = data

    def bounds_cb(self, data):
        self.track_bounds = data

    def map_info_cb(self, data):
        self.map_infos = data    
    
    def est_lap_time_cb(self, data):
        self.est_lap_time = data

    def lattice_cb(self, data):
        self.graph_lattice = data

    def global_republisher(self):
        rate = rospy.Rate(0.5)  # in Hertz
        while not rospy.is_shutdown():

            if self.glb_wpnts is not None and self.glb_markers is not None:
                self.glb_wpnts_pub.publish(self.glb_wpnts)
                self.glb_markers_pub.publish(self.glb_markers)
            if self.glb_sp_wpnts is not None and self.glb_sp_markers is not None:
                self.glb_sp_wpnts_pub.publish(self.glb_sp_wpnts)
                self.glb_sp_markers_pub.publish(self.glb_sp_markers)
            if self.centerline_wpnts is not None and self.centerline_markers is not None:
                self.centerline_wpnts_pub.publish(self.centerline_wpnts)
                self.centerline_markers_pub.publish(self.centerline_markers)
            if self.track_bounds is not None:
                self.vis_track_bnds.publish(self.track_bounds)
            if self.map_infos is not None:
                self.map_info_pub.publish(self.map_infos)
            if self.est_lap_time is not None:
                self.est_lap_time_pub.publish(self.est_lap_time)
            if self.graph_lattice is not None:
                self.lattice_pub.publish(self.graph_lattice)
            ### HJ : publish velocity markers
            if self.glb_vel_markers is not None:
                self.glb_vel_markers_pub.publish(self.glb_vel_markers)
            ### HJ : end

            rate.sleep()


if __name__ == "__main__":
    republisher = GlobalRepublisher()
    republisher.global_republisher()
