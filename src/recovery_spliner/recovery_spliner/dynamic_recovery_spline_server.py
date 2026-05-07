#!/usr/bin/env python3

def callback(config, level):
    # Ensuring nice rounding by either 0.05 or 0.5

    return config

if __name__ == "__main__":
    rospy.init_node("dynamic_recovery_spline_tuner_node", anonymous=False)
    print('[Planner] Dynamic Recovery Spline Server Launched...')
    srv = Server(dyn_recovery_spliner_tunerConfig, callback)
    rospy.spin()

