cmake_minimum_required(VERSION 3.16)
project(fast_livo_global_init CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(std_srvs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)

add_executable(global_init_node
  src/global_init_node_main.cpp
  src/global_init_node.cpp
)

target_include_directories(global_init_node PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(global_init_node
  kiss_matcher::kiss_matcher_core
  Eigen3::Eigen
  TBB::tbb
  ${PCL_LIBRARIES}
)

ament_target_dependencies(global_init_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  std_srvs
  tf2
  tf2_eigen
  pcl_conversions
)

add_executable(test_self_match
  src/test_self_match_main.cpp
)
target_include_directories(test_self_match PRIVATE
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(test_self_match
  kiss_matcher::kiss_matcher_core
  Eigen3::Eigen
  TBB::tbb
  ${PCL_LIBRARIES}
)

install(TARGETS global_init_node test_self_match
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
