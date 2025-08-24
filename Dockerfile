FROM nvcr.io/nvidia/l4t-ros:humble-desktop   # JetPack 6.x + ROS2 Humble + CUDA/TRT

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip pkg-config \
    python3-pip python3-colcon-common-extensions \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    ros-humble-image-transport ros-humble-cv-bridge ros-humble-camera-info-manager ros-humble-vision-msgs \
    libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 libnvinfer-dev libnvinfer-plugin-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/ros2_ws
COPY ros2_ws/src ./src
RUN bash -lc "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

ENV ROS_DISTRO=humble
CMD bash -lc "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch shobo_bringup perception.launch.py"
