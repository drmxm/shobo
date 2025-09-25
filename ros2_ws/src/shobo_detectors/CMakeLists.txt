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
# Custom NMS keeps us on the lighter OpenCV components
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

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

if (USE_TRT)
  # TensorRT pulls in driver/DLA libs; disable if they are unavailable during the build.
  set(_nv_driver_lib_specs
    nv:os
    nv:dla_compiler
    nv:dla_runtime
    nv:rm_gpu
    nv:rm_mem
    nv:cud:la
    c:uda)
  set(_missing_nv_libs)
  foreach(_spec IN LISTS _nv_driver_lib_specs)
    string(REPLACE ":" "" _lib "${_spec}")
    set(_found_var "_nv_lib_${_lib}")
    find_library(${_found_var} ${_lib}
      HINTS
        /usr/lib/aarch64-linux-gnu
        /usr/lib/aarch64-linux-gnu/tegra
        /usr/lib/aarch64-linux-gnu/nvidia
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/aarch64-linux/lib
        /usr/local/cuda/compat)
    if (NOT ${_found_var})
      list(APPEND _missing_nv_libs ${_lib})
    endif()
  endforeach()
  if (_missing_nv_libs)
    list(JOIN _missing_nv_libs ", " _missing_nv_libs_str)
    message(STATUS "TensorRT driver dependencies missing: ${_missing_nv_libs_str}. Building without TRT.")
    set(USE_TRT OFF)
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(detector_node
  src/trt_detector_node.cpp
  src/kernels.cu
)

set_target_properties(detector_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "87"  # Orin
)

ament_target_dependencies(detector_node
  rclcpp image_transport cv_bridge sensor_msgs vision_msgs
)

target_include_directories(detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link ONLY cudart + OpenCV (+ TRT). No driver libs, no cuda stub.
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
  target_link_options(detector_node PRIVATE "-Wl,--no-as-needed")

  set(_nv_link_dirs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/local/cuda/compat)
  foreach(link_dir IN LISTS _nv_link_dirs)
    if (EXISTS ${link_dir})
      target_link_directories(detector_node PRIVATE ${link_dir})
      target_link_options(detector_node PRIVATE "-Wl,-rpath-link,${link_dir}")
    endif()
  endforeach()
  target_compile_definitions(detector_node PRIVATE SHB_USE_TRT=1)
  if (SHB_USE_DLA)
    target_compile_definitions(detector_node PRIVATE SHB_USE_DLA=1)
  endif()
endif()

# RPATH only (so runtime-mounted libs are found when the container RUNS)
set(NV_RPATHS
  "/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu/tegra"
  "/usr/lib/aarch64-linux-gnu/nvidia"
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
