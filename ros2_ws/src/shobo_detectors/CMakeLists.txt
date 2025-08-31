cmake_minimum_required(VERSION 3.18)
project(shobo_detectors LANGUAGES CXX CUDA)

# Build config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
# Orin is SM 87; keep PTX minimal for faster builds
set(CMAKE_CUDA_ARCHITECTURES 87)

# ROS deps
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA/TRT (fail fast)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS /usr/local/cuda/include
  PATHS /usr/include /usr/include/aarch64-linux-gnu /usr/local/include)
find_library(CUDART_LIB NAMES cudart libcudart
  HINTS /usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/lib64
  PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu
  PATHS /usr/include /usr/local/include)
find_library(TENSORRT_LIB NAMES nvinfer
  HINTS /usr/lib/aarch64-linux-gnu
  PATHS /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIB)
  message(FATAL_ERROR "CUDA headers/libs not found (cuda_runtime_api.h / libcudart). Build on Jetson L4T.")
endif()
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT headers/libs not found (NvInfer.h / libnvinfer). Install libnvinfer-dev.")
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIR}
  ${TENSORRT_INCLUDE_DIR}
)

# ---- CUDA kernels as a STATIC lib (avoids nvcc 'single input file' fatal) ----
add_library(shobo_kernels STATIC src/kernels.cu)
target_include_directories(shobo_kernels PUBLIC include ${CUDA_INCLUDE_DIR})
target_link_libraries(shobo_kernels PUBLIC ${CUDART_LIB})
# Quiet CUDA header pedantry warnings
target_compile_options(shobo_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic,-Wno-cpp>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# ---- Detector node ----
add_executable(detector_node src/trt_detector_node.cpp)
target_link_libraries(detector_node
  shobo_kernels
  ${TENSORRT_LIB}
  ${CUDART_LIB}
  ${OpenCV_LIBS})
ament_target_dependencies(detector_node
  rclcpp sensor_msgs image_transport cv_bridge vision_msgs geometry_msgs OpenCV)

# Helpful RPATHs on Jetson
set_target_properties(detector_node PROPERTIES
  BUILD_RPATH "/usr/lib/aarch64-linux-gnu;/usr/local/cuda/lib64;/usr/local/cuda/targets/aarch64-linux/lib"
  INSTALL_RPATH "/usr/lib/aarch64-linux-gnu;/usr/local/cuda/lib64;/usr/local/cuda/targets/aarch64-linux/lib")

install(TARGETS detector_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
