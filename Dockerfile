FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install -y --no-install-recommends \
    xserver-xorg

# Cache installtion
RUN apt-get update && apt install -y --no-install-recommends \
    ros-humble-moveit

ENV ROS2_DEPENDENCIES_DIR=/tmp/ros2_dependencies

# Install dependencies
COPY ros2/src ${ROS2_DEPENDENCIES_DIR}/src
RUN rosdep install -r -y -i --from-paths ${ROS2_DEPENDENCIES_DIR} && rm -rf ${ROS2_DEPENDENCIES_DIR}

WORKDIR /workspace

FROM base AS dev

RUN apt-get update && apt install -y --no-install-recommends \
    vim \
    tmux \
    iproute2 \
    x11-apps

CMD ["/bin/bash"]

