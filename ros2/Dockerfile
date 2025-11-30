# syntax=docker/dockerfile:1.20
# ============================================================================
# ROS2 Stage with VLA Inference Support
# ============================================================================
FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

# Add Intel RealSense repository and install all system dependencies
# Using BuildKit cache mounts for apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=ros2-apt-cache \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=ros2-apt-lists \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    ros-humble-moveit \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    librealsense2 \
    librealsense2-utils \
    librealsense2-dev \
    v4l-utils \
    python3-pip \
    python3-opencv \
    sudo \
    espeak-ng \
    alsa-utils \
    pulseaudio-utils \
    iproute2 \
    x11-apps

# Install PyTorch (CUDA 12.4 compatible) - must match vla/Dockerfile.openvla
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies (unified requirements file)
COPY ros2/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy OpenVLA/prismatic source code for model loading (before ROS source for caching)
COPY vla/src/openvla /workspace/vla/src/openvla

# Set PYTHONPATH to include VLA code
ENV PYTHONPATH="/workspace/vla/src/openvla:${PYTHONPATH}"

# Create workspace directory structure
WORKDIR /workspace/ros2

# Install ROS dependencies with caching optimization
# Copy only package.xml files first for better layer caching
COPY ros2/src /tmp/ros2_dependencies/src
RUN find /tmp/ros2_dependencies/src -type f ! -name "package.xml" -delete && \
    find /tmp/ros2_dependencies/src -type d -empty -delete

# Install dependencies based on package.xml files (optimized: single apt call)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=ros2-apt-cache \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=ros2-apt-lists \
    apt-get update && \
    deps=$(rosdep install -r -y --simulate --from-paths /tmp/ros2_dependencies 2>/dev/null \
        | grep -oP '(?<=apt-get install -y ).*' \
        | tr ' ' '\n' | sort -u | tr '\n' ' ') && \
    if [ -n "$deps" ]; then apt-get install -y $deps; fi && \
    rm -rf /tmp/ros2_dependencies

# Copy source code to workspace
COPY ros2/src ./src

# Build ROS 2 packages
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Copy entrypoint script
COPY ros2/scripts/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create a user that matches host user (UID/GID will be passed at build time)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=ros2

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true \
    && useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chown -R ${USER_ID}:${GROUP_ID} /workspace

USER ${USERNAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
