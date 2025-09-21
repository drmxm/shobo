# Jetson JP 6.x / L4T r36.x (requires: docker login nvcr.io)
FROM nvcr.io/nvidia/l4t-base:r36.2.0

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 lsb-release dirmngr locales tzdata \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
         > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-foxglove-bridge \
      ros-humble-rosbridge-server \
      netcat-traditional \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Source ROS automatically for interactive shells
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc

WORKDIR /work
COPY start.sh /start.sh
COPY params/ /work/params/
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
