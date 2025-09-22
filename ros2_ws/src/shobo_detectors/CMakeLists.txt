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
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)

# ---- TensorRT detection
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include /usr/include/aarch64-linux-gnu /usr/include/aarch64-linux-gnu/tensorrt /usr/local/include)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
set(USE_TRT OFF)
if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
  message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
  set(USE_TRT ON)
else()
  message(WARNING "TensorRT NOT found - building without TRT acceleration")
endif()

# ---- Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Target
add_executable(detector_node
  src/trt_detector_node.cpp
  src/kernels.cu
)

set_target_properties(detector_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  # Orin (Ampere) is 87; keep 53 only if you truly support the old Nano
  CUDA_ARCHITECTURES "87"
)

ament_target_dependencies(detector_node
  rclcpp
  image_transport
  cv_bridge
  sensor_msgs
  vision_msgs
)

target_include_directories(detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
if (USE_TRT)
  target_include_directories(detector_node PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()

# Prefer not to over-link: allow the linker to drop unused deps
add_link_options(-Wl,--as-needed)

# RPATHs that exist on JetPack
set(NV_EXTRA_LIB_DIRS
  "/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu"
  "/usr/lib/aarch64-linux-gnu/tegra"
  "/usr/lib/aarch64-linux-gnu/nvidia"
  "/usr/local/cuda/lib64"
  "/usr/local/cuda/targets/aarch64-linux/lib"
  "/usr/local/cuda-12.6/targets/aarch64-linux/lib"
)
foreach(dir IN LISTS NV_EXTRA_LIB_DIRS)
  if (EXISTS ${dir})
    target_link_directories(detector_node BEFORE PRIVATE ${dir})
    list(APPEND NV_RPATHS ${dir})
  endif()
endforeach()
if (NV_RPATHS)
  list(REMOVE_DUPLICATES NV_RPATHS)
  string(JOIN ":" NV_RPATH_STR ${NV_RPATHS})
  set_target_properties(detector_node PROPERTIES
    BUILD_RPATH "${NV_RPATH_STR}"
    INSTALL_RPATH "${NV_RPATH_STR}"
  )
  foreach(rpath_dir IN LISTS NV_RPATHS)
    target_link_options(detector_node PRIVATE "-Wl,-rpath-link,${rpath_dir}")
  endforeach()
endif()

# ---- Minimal, correct linkage
target_link_libraries(detector_node
  CUDA::cudart
  ${OpenCV_LIBRARIES}
)

if (TARGET CUDA::cudadevrt)
  target_link_libraries(detector_node CUDA::cudadevrt)
else()
  message(WARNING "CUDA::cudadevrt not found; proceeding without explicit device runtime link")
endif()

if (USE_TRT)
  target_link_libraries(detector_node ${TENSORRT_LIB})
  if (TENSORRT_PLUGIN_LIB)
    target_link_libraries(detector_node ${TENSORRT_PLUGIN_LIB})
  endif()
  target_compile_definitions(detector_node PRIVATE SHB_USE_TRT=1)
  # TensorRT on Jetson depends on cuDLA/NvOS/NVDLA helper libs; link them explicitly
  set(JETSON_DEPS_EXPLAIN "Install 'libcudla-12-6' (or matching cuDLA runtime) so TensorRT's helper libraries are available inside the container.")

  function(find_jetson_lib out_var lib_name)
    set(_candidates)
    foreach(dir ${NV_EXTRA_LIB_DIRS})
      list(APPEND _candidates
        "${dir}/lib${lib_name}.so"
        "${dir}/lib${lib_name}.so.1"
        "${dir}/lib${lib_name}.so.1.0.0"
      )
    endforeach()
    foreach(path_candidate ${_candidates})
      if (EXISTS ${path_candidate})
        set(${out_var} ${path_candidate} PARENT_SCOPE)
        return()
      endif()
    endforeach()

    find_library(_lib_handle NAMES ${lib_name} HINTS ${NV_EXTRA_LIB_DIRS})
    if (_lib_handle)
      set(${out_var} ${_lib_handle} PARENT_SCOPE)
      return()
    endif()

    message(FATAL_ERROR "Required Jetson library '${lib_name}' not found. ${JETSON_DEPS_EXPLAIN}")
  endfunction()

  find_jetson_lib(LIB_CUDLA          cudla)
  find_jetson_lib(LIB_NVCUDLA        nvcudla)
  find_jetson_lib(LIB_NVOS           nvos)
  find_jetson_lib(LIB_NVDLA_COMPILER nvdla_compiler)
  find_jetson_lib(LIB_NVDLA_RUNTIME  nvdla_runtime)

  target_link_libraries(detector_node
    ${LIB_CUDLA}
    ${LIB_NVCUDLA}
    ${LIB_NVOS}
    ${LIB_NVDLA_COMPILER}
    ${LIB_NVDLA_RUNTIME}
  )
endif()

install(TARGETS detector_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_package()
