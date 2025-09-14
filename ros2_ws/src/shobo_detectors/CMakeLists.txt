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

# --- Optional TensorRT detection (no hard-fail) ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include /usr/include/aarch64-linux-gnu /usr/include/aarch64-linux-gnu/tensorrt /usr/local/include)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
  message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
  set(USE_TRT ON)
else()
  message(WARNING "TensorRT NOT found - building without TRT acceleration")
  set(USE_TRT OFF)
endif()

# --- Public headers ---
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Node (include kernels.cu directly so CMake runs device link) ---
add_executable(detector_node
  src/trt_detector_node.cpp
  src/kernels.cu
)

# Ensure separable compilation + device symbol resolution on the exe
set_target_properties(detector_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "53;87"   # Nano + Orin; use "87" if only Orin
)

ament_target_dependencies(detector_node
  rclcpp
  image_transport
  cv_bridge
  sensor_msgs
  vision_msgs
)

# Make CUDA headers visible (fix for NvInfer -> cuda_runtime_api.h)
target_include_directories(detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}    # usually /usr/local/cuda/include
)

# Link CUDA imported targets
target_link_libraries(detector_node
  CUDA::cudart
  CUDA::cuda_driver
)

# cudadevrt is not present in all CUDA toolkits or FindCUDAToolkit versions.
# Link it only if the imported target exists; otherwise rely on device linking.
if (TARGET CUDA::cudadevrt)
  target_link_libraries(detector_node CUDA::cudadevrt)
else()
  message(WARNING "CUDA::cudadevrt not found; proceeding without explicit device runtime link")
endif()

# Add TensorRT only if found
if (USE_TRT)
  target_include_directories(detector_node PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(detector_node ${TENSORRT_LIB})
  if (TENSORRT_PLUGIN_LIB)
    target_link_libraries(detector_node ${TENSORRT_PLUGIN_LIB})
  endif()
  target_compile_definitions(detector_node PRIVATE SHB_USE_TRT=1)
endif()

# --- Install & export ---
install(TARGETS detector_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_package()
