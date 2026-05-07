import math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class EStop:
    def __init__(self, node):
        p = lambda name: node.get_parameter(name).value
        self._logger     = node.get_logger()

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()

        stop_flag = False

        if scan is None or odom is None:
            self._logger.warn(f'Missing scan or odometry data, skipping EStop check')
            return cmd            

        if cmd.drive.speed <= 0.0:
            return cmd # No need to check if we're already stopped or reversing

        car_width = 0.3  # Adjust based on your car's width
        threshold_ttc = 1.0  # Time-to-Collision threshold in seconds
        threshold_dis = 0.2  # Distance threshold in meters

        v_x = odom.twist.twist.linear.x
        start_angle = scan.angle_min
        for i, distance in enumerate(scan.ranges):
            angle = start_angle + i * scan.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            if abs(y) > car_width / 2:
                continue  # Ignore points outside the car's width

            point = np.array([x, y])


            if x < threshold_dis:   # If an obstacle is very close (e.g., within 20 cm), trigger E-Stop immediately
                stop_flag = True
                self._logger.warn(f'EStop triggered: point={point}, too close to the car')
                break

            unit_vector = point / np.linalg.norm(point) if np.linalg.norm(point) > 0 else np.zeros_like(point)
            relative_vel = np.array([v_x, 0]) @ unit_vector
            ttc = distance / relative_vel if relative_vel > 0 else float('inf')

            if ttc < threshold_ttc:
                self._logger.warn(f'EStop triggered: point={point}, relative_vel={relative_vel:.2f}, ttc={ttc:.2f}')
                stop_flag = True
                break
            
        # TODO: Implement an emergency stop (E-Stop) using:
        #   - 2D LiDAR scan data
        #   - Wheel odometry data from the VESC
        #   - TTC (Time-to-Collision) based logic
        #
        # You may modify `cmd` (the original control command) in this function.
        #
        # Useful information:
        #   - scan.ranges                  : distance array [m] for each LiDAR beam
        #   - scan.angle_min               : angle of the first beam [rad]
        #   - scan.angle_max               : angle of the last beam [rad]
        #   - scan.angle_increment         : angular step between beams [rad]
        #   - odom.twist.twist.linear.x    : vehicle forward speed [m/s]
        #   - odom.twist.twist.angular.z   : vehicle yaw rate [rad/s]

        # if ttc < 1.0:  # Adjust threshold as needed
        #     self._logger.warn(f'EStop triggered')
        #     cmd.drive.speed = 0.0
        if stop_flag:
            cmd.drive.speed = 0.0
        return cmd
