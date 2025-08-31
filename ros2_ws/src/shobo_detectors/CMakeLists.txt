cmake_minimum_required(VERSION 3.18)
project(shobo_detectors LANGUAGES CXX CUDA)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Our headers (for kernels.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA include/lib (JetPack default install)
include_directories(/usr/local/cuda/include)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 REQUIRED)

# Build kernels as object lib then link into the node
add_library(cuda_kernels OBJECT src/kernels.cu)
set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(detector_node
  src/trt_detector_node.cpp
  $<TARGET_OBJECTS:cuda_kernels>
)
ament_target_dependencies(detector_node
  rclcpp sensor_msgs vision_msgs image_transport cv_bridge)
target_include_directories(detector_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detector_node
  ${OpenCV_LIBS}
  nvinfer
  nvinfer_plugin
  ${CUDART_LIB}
)

install(TARGETS detector_node DESTINATION lib/${PROJECT_NAME})
ament_package()
