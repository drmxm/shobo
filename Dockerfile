# JetPack 6.2.x (R36.4.x) + CUDA/TRT base
ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    ROS_DISTRO=humble

# OS basics & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release locales ca-certificates software-properties-common \
      build-essential cmake git wget unzip pkg-config \
      python3-pip \
      v4l-utils \
      libopencv-dev \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly gstreamer1.0-libav && \
    add-apt-repository -y universe && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Colcon + vcstool via pip
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir vcstool colcon-common-extensions

# Add ROS 2 Humble apt repo and install ROS packages
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-desktop \
      ros-humble-image-transport ros-humble-image-transport-plugins \
      ros-humble-cv-bridge ros-humble-camera-info-manager ros-humble-vision-msgs \
      python3-rosdep && \
    rm -rf /var/lib/apt/lists/*

# Ensure ROS is sourced by default shells (nice to have)
SHELL ["/bin/bash","-lc"]
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# Workspace
WORKDIR /work/ros2_ws
COPY ros2_ws/src ./src

# Resolve deps and build (explicitly source ROS before colcon)
RUN bash -lc 'set -e \
  && source /opt/ros/${ROS_DISTRO}/setup.bash \
  && rosdep init || true \
  && rosdep update \
  && apt-get update \
  && rosdep install --from-paths src --ignore-src -r -y \
  && colcon build --symlink-install \
  && rm -rf /var/lib/apt/lists/*'

CMD bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && source install/setup.bash && ros2 launch shobo_bringup perception.launch.py"
