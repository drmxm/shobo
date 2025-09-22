# ------------------------------------------------------------
# JetPack 6.2.x (L4T r36.4.x)
# ------------------------------------------------------------
ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

# Keep ROS_DISTRO early because we use it in env later
ENV ROS_DISTRO=humble

SHELL ["/bin/bash","-lc"]
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC LANG=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ------------------------------------------------------------
# OS basics & toolchain
# ------------------------------------------------------------
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release locales ca-certificates software-properties-common \
      build-essential cmake git wget unzip pkg-config \
      python3 python3-pip python3-setuptools python3-wheel python3-numpy \
      v4l-utils \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly gstreamer1.0-libav \
      libopencv-dev \
 && add-apt-repository -y universe \
 && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# NVIDIA Jetson package repository (for cuDLA runtime, etc.)
# ------------------------------------------------------------
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson-archive-keyring.gpg \
 && printf '%s\n%s\n' \
      "deb [signed-by=/usr/share/keyrings/nvidia-jetson-archive-keyring.gpg] https://repo.download.nvidia.com/jetson/common r36.4 main" \
      "deb [signed-by=/usr/share/keyrings/nvidia-jetson-archive-keyring.gpg] https://repo.download.nvidia.com/jetson/t234 r36.4 main" \
      > /etc/apt/sources.list.d/nvidia-jetson.list \
 && apt-get update \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# ROS 2 Humble
# ------------------------------------------------------------
RUN set -e \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-desktop \
      ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-image-transport-plugins \
      ros-${ROS_DISTRO}-camera-info-manager ros-${ROS_DISTRO}-vision-msgs \
      ros-${ROS_DISTRO}-sensor-msgs \
      python3-rosdep python3-vcstool python3-colcon-common-extensions \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# TensorRT dev + cuDLA runtime (required for libnvinfer DT_NEEDED deps)
# ------------------------------------------------------------
# libnvdla_* libraries ship with JetPack r36.4 under /usr/lib/aarch64-linux-gnu/nvidia
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libnvinfer-dev \
      libnvinfer-plugin-dev \
      libcudla-12-6 \
      libcudla-dev-12-6 \
 && echo "/usr/lib/aarch64-linux-gnu/tegra"   >  /etc/ld.so.conf.d/tegra.conf \
 && echo "/usr/lib/aarch64-linux-gnu/nvidia"  >  /etc/ld.so.conf.d/nvidia-tegra.conf \
 && echo "/usr/local/cuda/lib64"              >  /etc/ld.so.conf.d/cuda.conf \
 && ldconfig \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Safety checks (headers present, no stray /usr/local OpenCV)
# ------------------------------------------------------------
RUN set -e \
 && test -f /usr/local/cuda/include/cuda_runtime_api.h || { echo "FATAL: Missing CUDA headers"; exit 1; } \
 && { test -f /usr/include/aarch64-linux-gnu/NvInfer.h \
   || test -f /usr/include/aarch64-linux-gnu/tensorrt/NvInfer.h; } \
   || { echo "FATAL: Missing TensorRT headers (NvInfer.h)"; exit 1; } \
 && if compgen -G "/usr/local/lib/libopencv_core.so*" >/dev/null; then \
        echo "FATAL: Found OpenCV in /usr/local (conflicts with ROS/cv_bridge)"; exit 1; \
    else echo "[OK] No stray /usr/local OpenCV"; fi

# Now that OpenCV is installed, set CMake hints to silence warnings
ENV OpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
    CMAKE_PREFIX_PATH=/opt/ros/${ROS_DISTRO}:/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
    PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig

# ------------------------------------------------------------
# Make ROS available to all shells
# ------------------------------------------------------------
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# ------------------------------------------------------------
# Workspace + cv_bridge from source (to match JP OpenCV)
# ------------------------------------------------------------
WORKDIR /work/ros2_ws
COPY ros2_ws/src ./src
RUN set -e \
 && git clone --depth=1 -b humble https://github.com/ros-perception/vision_opencv /tmp/vision_opencv \
 && mv /tmp/vision_opencv/cv_bridge ./src/cv_bridge \
 && rm -rf /tmp/vision_opencv

# Resolve rosdeps (ok as root in Docker)
RUN set -e \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rosdep init || true \
 && rosdep update \
 && apt-get update \
 && rosdep install --from-paths src --ignore-src -r -y \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# NVCC host compiler: force GCC
# ------------------------------------------------------------
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc

# ------------------------------------------------------------
# Build (clean state each time)
# ------------------------------------------------------------
RUN set -e \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rm -rf build install log \
 && colcon build --symlink-install \
      --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

# ------------------------------------------------------------
# Default entry
# ------------------------------------------------------------
ENV ROS_LOG_DIR=/work/ros2_ws/logs
RUN mkdir -p ${ROS_LOG_DIR}
CMD bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && source install/setup.bash && ros2 launch shobo_bringup perception.launch.py"
