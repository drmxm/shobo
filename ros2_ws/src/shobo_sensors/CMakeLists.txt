cmake_minimum_required(VERSION 3.10)
project(shobo_sensors)

if(DEFINED CATKIN_INSTALL_INTO_PREFIX_ROOT)
  set(_catkin_install_into_prefix_root ${CATKIN_INSTALL_INTO_PREFIX_ROOT})
endif()
if(DEFINED CATKIN_SYMLINK_INSTALL)
  set(_catkin_symlink_install ${CATKIN_SYMLINK_INSTALL})
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(uvc_cam_node src/uvc_cam_node.cpp)
ament_target_dependencies(uvc_cam_node
  rclcpp image_transport sensor_msgs cv_bridge camera_info_manager)
target_link_libraries(uvc_cam_node ${OpenCV_LIBS})

add_executable(csi_cam_node src/csi_cam_node.cpp)
ament_target_dependencies(csi_cam_node
  rclcpp image_transport sensor_msgs cv_bridge camera_info_manager)
target_link_libraries(csi_cam_node ${OpenCV_LIBS})

install(TARGETS uvc_cam_node csi_cam_node DESTINATION lib/${PROJECT_NAME})
ament_package()
