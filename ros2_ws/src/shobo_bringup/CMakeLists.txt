cmake_minimum_required(VERSION 3.16)
project(shobo_bringup)

find_package(ament_cmake REQUIRED)

# Install launch and optional config directories if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
