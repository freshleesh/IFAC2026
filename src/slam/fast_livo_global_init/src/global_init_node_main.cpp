#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "fast_livo_global_init/global_init_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fast_livo_global_init::GlobalInitNode>();
  // service callback이 condition_variable.wait_for 로 동기 block 하므로
  // single-threaded executor면 lidar 콜백도 같이 block 됨. multi-threaded 필수.
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
