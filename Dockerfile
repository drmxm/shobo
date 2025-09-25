# JetPack 6.2.x (L4T r36.4.x)
ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV ROS_DISTRO=humble
SHELL ["/bin/bash","-lc"]
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC LANG=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# --- OS + toolchain + GStreamer + OpenCV (system) ---
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

# --- Jetson repos (to fetch DEBs without installing meta packages) ---
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson-archive-keyring.gpg \
 && printf '%s\n%s\n' \
      "deb [signed-by=/usr/share/keyrings/nvidia-jetson-archive-keyring.gpg] https://repo.download.nvidia.com/jetson/common r36.4 main" \
      "deb [signed-by=/usr/share/keyrings/nvidia-jetson-archive-keyring.gpg] https://repo.download.nvidia.com/jetson/t234   r36.4 main" \
      > /etc/apt/sources.list.d/nvidia-jetson.list

# --- ROS 2 Humble ---
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

# --- TensorRT headers (SAFE to install) ---
RUN set -e \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libnvinfer-dev libnvinfer-plugin-dev dpkg-dev \
 && echo "/usr/lib/aarch64-linux-gnu/tegra"  > /etc/ld.so.conf.d/tegra.conf \
 && echo "/usr/lib/aarch64-linux-gnu/nvidia" > /etc/ld.so.conf.d/nvidia-tegra.conf \
 && echo "/usr/local/cuda/lib64"             > /etc/ld.so.conf.d/cuda.conf \
 && ldconfig

# --- DLA compiler: copy ONLY the binary (no .so, no preinst scripts) ---
RUN set -e \
 && apt-get update \
 && apt-get download nvidia-l4t-dla-compiler \
 && mkdir -p /tmp/nvdla_extract /usr/local/bin \
 && dpkg-deb -x nvidia-l4t-dla-compiler_*.deb /tmp/nvdla_extract \
 && cp -a /tmp/nvdla_extract/usr/bin/nvdla_compiler /usr/local/bin/ 2>/dev/null || true \
 && rm -rf /tmp/nvdla_extract nvidia-l4t-dla-compiler_*.deb

# --- Env (silence OpenCV warning; keep ROS first) ---
ENV OpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
    CMAKE_PREFIX_PATH=/opt/ros/${ROS_DISTRO} \
    PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# --- Workspace (+ pin cv_bridge to JP OpenCV) ---
WORKDIR /work/ros2_ws
COPY ros2_ws/src ./src

# Safety: fail build if CMake tries to link driver/DLA libs
RUN set -e \
 && if grep -R -nE 'nvos|nvdla|nvrm|cuda[[:space:]]*\)|-lcuda|-lnvcudla|-lnvos|-lnvdla' src; then \
      echo 'FATAL: Your source/CMake still references driver/DLA libs. Remove them.' >&2; exit 1; \
    fi

RUN set -e \
 && git clone --depth=1 -b humble https://github.com/ros-perception/vision_opencv /tmp/vision_opencv \
 && mv /tmp/vision_opencv/cv_bridge ./src/cv_bridge \
 && rm -rf /tmp/vision_opencv

# --- rosdep (do update as non-root to avoid throttling) ---
RUN groupadd -f rosdep && useradd --create-home --shell /bin/bash -g rosdep rosbuild
RUN set -e \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rosdep init || true \
 && rosdep fix-permissions \
 && su - rosbuild -c 'rosdep update' \
 && mkdir -p /root/.ros && cp -a /home/rosbuild/.ros/rosdep /root/.ros/

# --- resolve deps + build (print full link line) ---
RUN set -e \
 && apt-get update \
 && rosdep install --from-paths src --ignore-src -r -y \
 && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc

RUN set -e \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rm -rf build install log \
 && colcon build --symlink-install --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON

ENV ROS_LOG_DIR=/work/ros2_ws/logs
RUN mkdir -p ${ROS_LOG_DIR}

CMD bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && source install/setup.bash && ros2 launch shobo_bringup perception.launch.py"
