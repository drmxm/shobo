cmake_minimum_required(VERSION 3.10)
project(shobo_detectors)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

enable_language(CUDA)
find_library(TENSORRT_LIB nvinfer)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin)
find_library(CUDART_LIB cudart)

add_executable(detector_node src/trt_detector_node.cpp)
ament_target_dependencies(detector_node
  rclcpp sensor_msgs vision_msgs image_transport cv_bridge)
target_link_libraries(detector_node
  ${OpenCV_LIBS} ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB} ${CUDART_LIB})
set_target_properties(detector_node PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)

install(TARGETS detector_node DESTINATION lib/${PROJECT_NAME})
ament_package()
