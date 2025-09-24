cmake_minimum_required(VERSION 3.18)
project(shobo_detectors LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)
# DO NOT pull 'dnn' here (it can drag libcuda at link time on Jetson)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# TensorRT optional; DLA is enabled at RUNTIME by TRT (dlopen), not by linker
option(SHB_USE_TRT "Build with TensorRT acceleration" ON)
option(SHB_USE_DLA "Enable DLA path via TRT runtime" ON)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include /usr/include/aarch64-linux-gnu /usr/include/aarch64-linux-gnu/tensorrt)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu)

set(USE_TRT OFF)
if (SHB_USE_TRT AND TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
  set(USE_TRT ON)
  message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
else()
  message(WARNING "TensorRT not found — building without TRT")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(detector_node
  src/trt_detector_node.cpp
  src/kernels.cu
)

set_target_properties(detector_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "87"  # Orin (Ampere)
)

ament_target_dependencies(detector_node
  rclcpp image_transport cv_bridge sensor_msgs vision_msgs
)

target_include_directories(detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link ONLY cudart + OpenCV + (optionally) TensorRT. No driver libs.
target_link_libraries(detector_node CUDA::cudart ${OpenCV_LIBRARIES})
if (TARGET CUDA::cudadevrt)
  target_link_libraries(detector_node CUDA::cudadevrt)
endif()

if (USE_TRT)
  target_include_directories(detector_node PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(detector_node ${TENSORRT_LIB})
  if (TENSORRT_PLUGIN_LIB)
    target_link_libraries(detector_node ${TENSORRT_PLUGIN_LIB})
  endif()
  target_compile_definitions(detector_node PRIVATE SHB_USE_TRT=1)
  if (SHB_USE_DLA)
    target_compile_definitions(detector_node PRIVATE SHB_USE_DLA=1)
  endif()
endif()

# RPATHs so runtime-mounted Jetson libs are found at run time (NOT at link time)
set(NV_RPATHS
  "/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu/tegra"
  "/usr/lib/aarch64-linux-gnu/nvidia"
  "/usr/local/cuda/compat"
  "/usr/local/cuda/lib64"
)
list(REMOVE_DUPLICATES NV_RPATHS)
string(JOIN ":" NV_RPATH_STR ${NV_RPATHS})
set_target_properties(detector_node PROPERTIES
  BUILD_RPATH "${NV_RPATH_STR}"
  INSTALL_RPATH "${NV_RPATH_STR}"
)

install(TARGETS detector_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_package()
