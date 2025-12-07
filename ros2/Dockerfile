# syntax=docker/dockerfile:1
# ROS Humble base image (Ubuntu 22.04)
FROM ros:humble-ros-base AS base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add Intel RealSense repository and install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-opencv \
    sudo \
    v4l-utils \
    espeak-ng \
    alsa-utils \
    pulseaudio-utils \
    iproute2 \
    x11-apps \
    ros-humble-moveit \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    librealsense2 \
    librealsense2-utils \
    librealsense2-dev

# Install PyTorch (CUDA 12.4 compatible)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY ros2/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy OpenVLA/prismatic source code for model loading
COPY vla/src/openvla /workspace/vla/src/openvla
ENV PYTHONPATH="/workspace/vla/src/openvla:${PYTHONPATH}"

# Create ROS workspace
WORKDIR /workspace/ros2/src

# Copy ROS source packages
COPY ros2/src /workspace/ros2/src

# Install ROS dependencies
WORKDIR /workspace/ros2
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    rosdep install -r -y --from-paths src --ignore-src

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Setup entrypoint
COPY ros2/scripts/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create user matching host (UID/GID passed at build time)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=ros2

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R ${USER_ID}:${GROUP_ID} /workspace

USER ${USERNAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
