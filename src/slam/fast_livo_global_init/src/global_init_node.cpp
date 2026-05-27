#include "fast_livo_global_init/global_init_node.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <chrono>
#include <sstream>

namespace fast_livo_global_init {

namespace {

std::vector<Eigen::Vector3f> cloudToVec(
    const pcl::PointCloud<pcl::PointXYZ> &cloud) {
  std::vector<Eigen::Vector3f> out;
  out.reserve(cloud.size());
  for (const auto &p : cloud) out.emplace_back(p.x, p.y, p.z);
  return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxelize(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &in, float leaf) {
  auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.filter(*out);
  return out;
}

}  // namespace

GlobalInitNode::GlobalInitNode()
    : rclcpp::Node("fast_livo_global_init"),
      state_(State::IDLE),
      retries_(0),
      frame_count_(0),
      accumulator_(new pcl::PointCloud<pcl::PointXYZ>()),
      match_done_(false),
      match_success_(false) {
  // params
  pcd_path_ = declare_parameter<std::string>("prior_map.pcd_path", "");
  target_voxel_ = declare_parameter<double>("prior_map.voxel_size", 0.3);
  world_frame_ =
      declare_parameter<std::string>("prior_map.world_frame", "camera_init");
  lidar_topic_ =
      declare_parameter<std::string>("input.lidar_topic", "/livox/lidar");
  num_accumulate_frames_ =
      declare_parameter<int>("input.num_accumulate_frames", 10);
  max_accumulate_seconds_ =
      declare_parameter<double>("input.max_accumulate_seconds", 2.0);
  source_voxel_ = declare_parameter<double>("input.source_voxel_size", 0.2);
  kiss_resolution_ = declare_parameter<double>("kiss_matcher.resolution", 0.3);
  min_final_inliers_ = declare_parameter<int>("accept.min_final_inliers", 30);
  min_inlier_ratio_ = declare_parameter<double>("accept.min_inlier_ratio", 0.5);
  max_translation_norm_ =
      declare_parameter<double>("accept.max_translation_norm", 30.0);
  max_retries_ = declare_parameter<int>("accept.max_retries", 5);
  pose_cov_diag_ = declare_parameter<std::vector<double>>(
      "output.pose_covariance_diag",
      {0.25, 0.25, 0.25, 0.0685, 0.0685, 0.0685});
  publish_debug_clouds_ =
      declare_parameter<bool>("output.publish_debug_clouds", true);
  shutdown_on_success_ =
      declare_parameter<bool>("output.shutdown_on_success", false);
  initial_pose_topic_ =
      declare_parameter<std::string>("output.initial_pose_topic",
                                     "/initialpose");
  auto_start_ = declare_parameter<bool>("control.auto_start", true);
  trigger_timeout_seconds_ =
      declare_parameter<double>("control.trigger_timeout_seconds", 20.0);

  RCLCPP_INFO(get_logger(), "fast_livo_global_init starting");
  RCLCPP_INFO(get_logger(), "  prior_map.pcd_path = %s", pcd_path_.c_str());
  RCLCPP_INFO(get_logger(), "  lidar_topic = %s", lidar_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  initial_pose_topic = %s",
              initial_pose_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  world_frame = %s", world_frame_.c_str());
  RCLCPP_INFO(get_logger(), "  auto_start = %s",
              auto_start_ ? "true" : "false");

  if (!loadPriorMap()) {
    RCLCPP_ERROR(get_logger(), "failed to load prior map; shutting down");
    rclcpp::shutdown();
    return;
  }

  rclcpp::QoS pose_qos(1);
  pose_qos.transient_local().reliable();
  pose_pub_ =
      create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
          initial_pose_topic_, pose_qos);

  if (publish_debug_clouds_) {
    rclcpp::QoS latched(1);
    latched.transient_local().reliable();
    debug_target_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/debug/target_map", latched);
    debug_aligned_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/debug/source_aligned", 1);

    pcl::PointCloud<pcl::PointXYZ> tgt_pcl;
    tgt_pcl.reserve(target_vec_.size());
    for (const auto &p : target_vec_)
      tgt_pcl.emplace_back(p.x(), p.y(), p.z());
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(tgt_pcl, msg);
    msg.header.frame_id = world_frame_;
    msg.header.stamp = now();
    debug_target_pub_->publish(msg);
  }

  // service callback이 cv.wait_for 로 동기 block 하므로 lidar 콜백과
  // 반드시 다른 callback group에 두어 MultiThreadedExecutor 가 병렬 실행하도록.
  lidar_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  service_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::QoS sub_qos(10);
  sub_qos.reliable();
  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = lidar_cb_group_;
  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, sub_qos,
      std::bind(&GlobalInitNode::onLidar, this, std::placeholders::_1),
      sub_opts);

  trigger_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/trigger",
      std::bind(&GlobalInitNode::onTrigger, this,
                std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_cb_group_);
  RCLCPP_INFO(get_logger(),
              "service ready: ~/trigger (std_srvs/srv/Trigger). "
              "Call it to (re)run global init.");

  if (auto_start_) {
    beginAttempt();
    RCLCPP_INFO(get_logger(),
                "auto_start=true: accumulating %d frames or %.1f s on %s",
                num_accumulate_frames_, max_accumulate_seconds_,
                lidar_topic_.c_str());
  } else {
    RCLCPP_INFO(get_logger(),
                "auto_start=false: idle, waiting for ~/trigger call");
  }
}

bool GlobalInitNode::loadPriorMap() {
  if (pcd_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "prior_map.pcd_path is empty");
    return false;
  }
  auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path_, *cloud) != 0) {
    RCLCPP_ERROR(get_logger(), "loadPCDFile failed: %s", pcd_path_.c_str());
    return false;
  }
  RCLCPP_INFO(get_logger(), "loaded prior map: %zu pts", cloud->size());
  auto down = voxelize(cloud, static_cast<float>(target_voxel_));
  target_vec_ = cloudToVec(*down);
  RCLCPP_INFO(get_logger(), "target voxel(%.2f) -> %zu pts", target_voxel_,
              target_vec_.size());
  return true;
}

void GlobalInitNode::beginAttempt() {
  {
    std::lock_guard<std::mutex> lk(accum_mtx_);
    accumulator_->clear();
    frame_count_ = 0;
  }
  {
    std::lock_guard<std::mutex> lk(match_mtx_);
    match_done_ = false;
    match_success_ = false;
    match_diagnostic_.clear();
  }
  retries_ = 0;
  accumulate_start_ = now();
  state_ = State::ACCUMULATING;
}

void GlobalInitNode::signalDone(bool success, const std::string &diag) {
  {
    std::lock_guard<std::mutex> lk(match_mtx_);
    match_done_ = true;
    match_success_ = success;
    match_diagnostic_ = diag;
  }
  match_cv_.notify_all();
  state_ = State::IDLE;
}

void GlobalInitNode::onLidar(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  if (state_.load() != State::ACCUMULATING) return;

  pcl::PointCloud<pcl::PointXYZ> frame;
  pcl::fromROSMsg(*msg, frame);

  size_t total_pts = 0;
  int n_frames = 0;
  bool time_exceeded = false;
  {
    std::lock_guard<std::mutex> lk(accum_mtx_);
    *accumulator_ += frame;
    total_pts = accumulator_->size();
    frame_count_++;
    n_frames = frame_count_;
    time_exceeded =
        (now() - accumulate_start_).seconds() >= max_accumulate_seconds_;
  }

  if (n_frames < num_accumulate_frames_ && !time_exceeded) return;

  RCLCPP_INFO(get_logger(),
              "accumulated %d frames, %zu pts (elapsed %.2f s) -> matching",
              n_frames, total_pts, (now() - accumulate_start_).seconds());
  state_ = State::MATCHING;
  runMatching();
}

void GlobalInitNode::runMatching() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_raw;
  {
    std::lock_guard<std::mutex> lk(accum_mtx_);
    src_raw = accumulator_;
    accumulator_ =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  }

  auto src_filt = pcl::PointCloud<pcl::PointXYZ>::Ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  src_filt->reserve(src_raw->size());
  for (const auto &p : *src_raw) {
    const float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (r < 0.5f || r > 60.0f) continue;
    src_filt->emplace_back(p);
  }
  auto src_down = voxelize(src_filt, static_cast<float>(source_voxel_));
  const auto src_vec = cloudToVec(*src_down);
  RCLCPP_INFO(get_logger(), "source: %zu raw -> %zu filt -> %zu voxel",
              src_raw->size(), src_filt->size(), src_vec.size());

  kiss_matcher::KISSMatcherConfig cfg(
      static_cast<float>(kiss_resolution_));
  kiss_matcher::KISSMatcher matcher(cfg);

  const auto t0 = std::chrono::steady_clock::now();
  const auto sol = matcher.estimate(src_vec, target_vec_);
  const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

  const int rot_inl = matcher.getNumRotationInliers();
  const int fin_inl = matcher.getNumFinalInliers();
  const double inlier_ratio =
      rot_inl > 0 ? static_cast<double>(fin_inl) / rot_inl : 0.0;
  const double t_norm = sol.translation.norm();

  std::ostringstream diag;
  diag << "valid=" << sol.valid << " t_norm=" << t_norm
       << " rot_inl=" << rot_inl << " final_inl=" << fin_inl
       << " ratio=" << inlier_ratio << " elapsed=" << dt << "s";

  RCLCPP_INFO(get_logger(), "match: %s", diag.str().c_str());

  const bool pass = sol.valid && fin_inl >= min_final_inliers_ &&
                    inlier_ratio >= min_inlier_ratio_ &&
                    t_norm <= max_translation_norm_;

  if (!pass) {
    retries_++;
    RCLCPP_WARN(get_logger(),
                "match rejected (retry %d/%d) — accumulating again",
                retries_, max_retries_);
    if (retries_ >= max_retries_) {
      RCLCPP_ERROR(get_logger(),
                   "max retries reached; returning to IDLE");
      signalDone(false, "max retries: " + diag.str());
      return;
    }
    {
      std::lock_guard<std::mutex> lk(accum_mtx_);
      frame_count_ = 0;
    }
    accumulate_start_ = now();
    state_ = State::ACCUMULATING;
    return;
  }

  publishInitialPose(sol.rotation, sol.translation);
  if (publish_debug_clouds_) {
    std::vector<Eigen::Vector3f> aligned;
    aligned.reserve(src_vec.size());
    const Eigen::Matrix3f Rf = sol.rotation.cast<float>();
    const Eigen::Vector3f tf = sol.translation.cast<float>();
    for (const auto &p : src_vec) aligned.emplace_back(Rf * p + tf);
    publishDebugClouds(aligned);
  }

  RCLCPP_INFO(get_logger(), "SUCCESS: published /initialpose");
  signalDone(true, diag.str());
  if (shutdown_on_success_) rclcpp::shutdown();
}

void GlobalInitNode::publishInitialPose(const Eigen::Matrix3d &R,
                                        const Eigen::Vector3d &t) {
  geometry_msgs::msg::PoseWithCovarianceStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = world_frame_;
  msg.pose.pose.position.x = t.x();
  msg.pose.pose.position.y = t.y();
  msg.pose.pose.position.z = t.z();
  const Eigen::Quaterniond q(R);
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.pose.orientation.w = q.w();

  for (size_t i = 0; i < 6 && i < pose_cov_diag_.size(); ++i) {
    msg.pose.covariance[i * 7] = pose_cov_diag_[i];
  }

  pose_pub_->publish(msg);
}

void GlobalInitNode::publishDebugClouds(
    const std::vector<Eigen::Vector3f> &src_aligned) {
  if (!debug_aligned_pub_) return;
  pcl::PointCloud<pcl::PointXYZ> pc;
  pc.reserve(src_aligned.size());
  for (const auto &p : src_aligned) pc.emplace_back(p.x(), p.y(), p.z());
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(pc, msg);
  msg.header.frame_id = world_frame_;
  msg.header.stamp = now();
  debug_aligned_pub_->publish(msg);
}

void GlobalInitNode::onTrigger(
    const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
    std_srvs::srv::Trigger::Response::SharedPtr res) {
  const State cur = state_.load();
  if (cur == State::ACCUMULATING || cur == State::MATCHING) {
    res->success = false;
    res->message = "busy: already running a match attempt";
    return;
  }

  RCLCPP_INFO(get_logger(), "~/trigger called: starting new match attempt");
  beginAttempt();

  // 동기 대기 (timeout)
  std::unique_lock<std::mutex> lk(match_mtx_);
  const auto timeout = std::chrono::milliseconds(
      static_cast<int>(trigger_timeout_seconds_ * 1000));
  const bool got = match_cv_.wait_for(
      lk, timeout, [this] { return match_done_; });

  if (!got) {
    res->success = false;
    res->message = "timeout waiting for match (no /livox/lidar?)";
    state_ = State::IDLE;
    return;
  }

  res->success = match_success_;
  res->message = match_diagnostic_;
}

}  // namespace fast_livo_global_init
