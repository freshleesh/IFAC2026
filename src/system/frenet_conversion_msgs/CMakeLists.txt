cmake_minimum_required(VERSION 3.8)
project(frenet_conversion_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# srv 만 정의하는 인터페이스 전용 패키지.
# 외부 메시지 의존 없음 (모두 float64/int32 기본 타입).
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  srv/Frenet2Glob.srv
  srv/Frenet2GlobArr.srv
  srv/Glob2Frenet.srv
  srv/Glob2FrenetArr.srv
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
