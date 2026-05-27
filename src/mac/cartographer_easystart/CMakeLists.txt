cmake_minimum_required(VERSION 3.14)
project(cartographer_easystart)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
