# syntax=docker/dockerfile:1
# ROS Humble base image (Ubuntu 22.04)
FROM ros:humble-ros-base AS base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy rosdep packages list first for better caching
# Regenerate with: ./scripts/generate_rosdep_packages.sh
COPY ros2/rosdep_packages.txt /tmp/rosdep_packages.txt

# Add Intel RealSense repository and install all apt dependencies in one layer
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
    librealsense2 \
    librealsense2-utils \
    librealsense2-dev \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosbridge-server \
    ros-humble-image-transport-plugins \
    gettext-base \
    $(grep -v '^#' /tmp/rosdep_packages.txt | grep -v '^$' | tr '\n' ' ') \
    && rm /tmp/rosdep_packages.txt

# Install PyTorch (CUDA 12.4 compatible)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
# Note: numpy<2.0 constraint is specified first in requirements.txt to prevent
# other packages from installing numpy>=2.0 (cv_bridge requires NumPy 1.x)
COPY ros2/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy OpenVLA/prismatic source code for model loading
COPY vla/src/openvla /workspace/vla/src/openvla

# Copy ManiSkill CRANE-X7 environment
COPY sim/src /workspace/sim/src
ENV PYTHONPATH="/workspace/vla/src/openvla:/workspace/sim/src:${PYTHONPATH}"
# Cache directories for Genesis, Taichi, and Numba (avoid permission issues)
ENV NUMBA_CACHE_DIR="/tmp/numba_cache"
ENV XDG_CACHE_HOME="/tmp/cache"
ENV GS_CACHE_FILE_PATH="/tmp/cache/genesis"
ENV TI_OFFLINE_CACHE_FILE_PATH="/tmp/cache/taichi"

# Create ROS workspace and copy source packages
WORKDIR /workspace/ros2/src
COPY ros2/src /workspace/ros2/src

WORKDIR /workspace/ros2

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Setup entrypoint
COPY ros2/docker/entrypoint.sh /entrypoint.sh
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
