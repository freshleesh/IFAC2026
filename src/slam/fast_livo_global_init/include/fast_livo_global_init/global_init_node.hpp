#pragma once

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <memory>
#include <mutex>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <vector>

namespace fast_livo_global_init {

enum class State {
  IDLE,           // 매칭 대기 (auto_start 또는 service trigger 대기)
  ACCUMULATING,   // /livox/lidar 누적 중
  MATCHING,       // KISS-Matcher 호출 중
};

class GlobalInitNode : public rclcpp::Node {
 public:
  GlobalInitNode();

 private:
  // params
  std::string pcd_path_;
  std::string lidar_topic_;
  std::string initial_pose_topic_;
  std::string world_frame_;
  double target_voxel_;
  double source_voxel_;
  int num_accumulate_frames_;
  double max_accumulate_seconds_;
  double kiss_resolution_;
  int min_final_inliers_;
  double min_inlier_ratio_;
  double max_translation_norm_;
  int max_retries_;
  std::vector<double> pose_cov_diag_;
  bool publish_debug_clouds_;
  bool shutdown_on_success_;
  bool auto_start_;
  double trigger_timeout_seconds_;

  // state
  std::atomic<State> state_;
  int retries_;
  int frame_count_;
  rclcpp::Time accumulate_start_;

  // data
  std::vector<Eigen::Vector3f> target_vec_;        // prior map (voxelized)
  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulator_;
  std::mutex accum_mtx_;

  // 매칭 완료 신호 (service callback에서 동기 wait)
  std::mutex match_mtx_;
  std::condition_variable match_cv_;
  bool match_done_;
  bool match_success_;
  std::string match_diagnostic_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debug_target_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debug_aligned_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_srv_;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;

  // logic
  bool loadPriorMap();
  void onLidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void runMatching();
  void publishInitialPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
  void publishDebugClouds(const std::vector<Eigen::Vector3f> &src_aligned);

  // 새 매칭 시도 시작 (auto_start + service callback 공통 진입점)
  void beginAttempt();
  // 매칭 결과를 condition_variable로 신호
  void signalDone(bool success, const std::string &diag);

  void onTrigger(const std_srvs::srv::Trigger::Request::SharedPtr req,
                 std_srvs::srv::Trigger::Response::SharedPtr res);
};

}  // namespace fast_livo_global_init
