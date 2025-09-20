# ------------------------------------------------------------
# JetPack 6.2.x (L4T r36.4.x) + CUDA/TRT base
# ------------------------------------------------------------
ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    ROS_DISTRO=humble \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

SHELL ["/bin/bash","-lc"]

# ------------------------------------------------------------
# OS basics & build toolchain
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
# ROS 2 Humble (Jammy)
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
      ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-camera-info-manager ros-${ROS_DISTRO}-vision-msgs \
      ros-${ROS_DISTRO}-sensor-msgs \
      python3-rosdep python3-vcstool python3-colcon-common-extensions \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# TensorRT dev headers/libs (JetPack 6.x)
# - libnvparsers-dev was removed; do NOT install it
# - libnvonnxparsers-dev may or may not exist; install if available
# ------------------------------------------------------------
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libnvinfer-dev \
      libnvinfer-plugin-dev || true \
 && if apt-cache show libnvonnxparsers-dev >/dev/null 2>&1; then \
        apt-get install -y --no-install-recommends libnvonnxparsers-dev; \
      else \
        echo "[WARN] libnvonnxparsers-dev not available on this image – OK if you don't parse ONNX at runtime."; \
      fi \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Optional NVDLA / cuDLA bits (present on some JP images)
# ------------------------------------------------------------
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends libnvdla-compiler || true \
 && apt-get install -y --no-install-recommends libnvdla-runtime  || true \
 && apt-get install -y --no-install-recommends cuda-cudla-12-5   || true \
 && echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/tegra.conf \
 && ldconfig \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Safety checks — fail fast if CUDA/TRT headers are missing
# and guard against stray OpenCV copies in /usr/local
# ------------------------------------------------------------
RUN set -e \
 && test -f /usr/local/cuda/include/cuda_runtime_api.h || { echo "FATAL: Missing CUDA headers (cuda_runtime_api.h)"; exit 1; } \
 && { test -f /usr/include/aarch64-linux-gnu/NvInfer.h \
   || test -f /usr/include/aarch64-linux-gnu/tensorrt/NvInfer.h; } \
   || { echo "FATAL: Missing TensorRT headers (NvInfer.h). Install libnvinfer-dev"; exit 1; } \
 && if compgen -G "/usr/local/lib/libopencv_core.so*" >/dev/null; then \
        echo "FATAL: Detected OpenCV in /usr/local (likely custom/pip build) — this conflicts with ROS cv_bridge. Remove it."; \
        ls -l /usr/local/lib/libopencv_core.so*; exit 1; \
    else echo "[OK] No stray /usr/local OpenCV found."; fi

# ------------------------------------------------------------
# Make ROS available to all shells
# ------------------------------------------------------------
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------
WORKDIR /work/ros2_ws
COPY ros2_ws/src ./src

# Resolve rosdeps (ok as root in Docker)
RUN set -e \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rosdep init || true \
 && rosdep update \
 && apt-get update \
 && rosdep install --from-paths src --ignore-src -r -y \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# NVCC host compiler: force GCC to avoid Clang line-directive spam
# ------------------------------------------------------------
ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc

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
# Default entry: bringup
# ------------------------------------------------------------
ENV ROS_LOG_DIR=/work/ros2_ws/logs
RUN mkdir -p ${ROS_LOG_DIR}

CMD bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && source install/setup.bash && ros2 launch shobo_bringup perception.launch.py"
