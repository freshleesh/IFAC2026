cmake_minimum_required(VERSION 3.8)
project(f110_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 Jazzy 포팅 — ament_cmake + rosidl_default_generators 사용.
# 모든 메시지는 원본 ROS1 정의를 그대로 사용 (OTWpntArray.msg 의 `time` 만
# `builtin_interfaces/Time` 으로 변경).
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/CarState.msg
  msg/CarStateStamped.msg
  msg/LapData.msg
  msg/Wpnt.msg
  msg/WpntArray.msg
  msg/Obstacle.msg
  msg/ObstacleArray.msg
  msg/OTWpntArray.msg
  msg/GapData.msg
  msg/PidData.msg
  msg/Prediction.msg
  msg/PredictionArray.msg
  msg/OpponentTrajectory.msg
  msg/OpponentTrajectories.msg
  msg/OppWpnt.msg
  msg/ProjOppPoint.msg
  msg/ProjOppTraj.msg
  msg/BehaviorStrategy.msg
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
