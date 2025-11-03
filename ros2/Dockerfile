# ============================================================================
# ROS2 Stage
# ============================================================================
FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

# Add Intel RealSense repository and install all system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with fixed versions
RUN pip3 install --no-cache-dir \
    tensorflow==2.15.1 \
    "numpy<2"

# Create workspace directory structure
WORKDIR /workspace/ros2

# Install ROS dependencies with caching optimization
# Copy only package.xml files first for better layer caching
COPY ros2/src /tmp/ros2_dependencies/src
RUN find /tmp/ros2_dependencies/src -type f ! -name "package.xml" -delete && \
    find /tmp/ros2_dependencies/src -type d -empty -delete

# Install dependencies based on package.xml files
RUN apt-get update && \
    rosdep install -r -y -i --from-paths /tmp/ros2_dependencies && \
    rm -rf /tmp/ros2_dependencies && \
    rm -rf /var/lib/apt/lists/*

# Copy source code to workspace
# Note: In development mode, this will be overridden by volume mount
COPY ros2/src ./src

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

CMD ["/bin/bash"]

FROM base AS dev

# Need to re-declare ARGs from base stage for use in dev stage
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=ros2

# Switch back to root to install additional packages
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    iproute2 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user
USER ${USERNAME}

CMD ["/bin/bash"]
