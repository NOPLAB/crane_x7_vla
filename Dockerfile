# ============================================================================
# ROS2 Stage 
# ============================================================================
FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    xserver-xorg \
    ros-humble-moveit \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    tensorflow \
    "numpy<2" \
    opencv-python

# Install ROS dependencies
ENV ROS2_DEPENDENCIES_DIR=/tmp/ros2_dependencies
COPY ros2/src ${ROS2_DEPENDENCIES_DIR}/src
RUN apt-get update && rosdep install -r -y -i --from-paths ${ROS2_DEPENDENCIES_DIR} && rm -rf ${ROS2_DEPENDENCIES_DIR} && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create a user that matches host user (UID/GID will be passed at build time)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=ros2

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set ownership of workspace
RUN mkdir -p /workspace && chown -R ${USER_ID}:${GROUP_ID} /workspace

USER ${USERNAME}

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
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user
USER ${USERNAME}

CMD ["/bin/bash"]

# ============================================================================
# VLA Fine-tuning Stage
# ============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS vla

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install VLA fine-tuning requirements
COPY vla/requirements.txt /tmp/vla_requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/vla_requirements.txt

# Install Flash Attention 2 (optional, for faster training)
# This can take several minutes to compile
RUN pip3 install --no-cache-dir packaging ninja && \
    pip3 install --no-cache-dir flash-attn==2.5.5 --no-build-isolation || \
    echo "Flash Attention installation failed - continuing without it"

# Set up workspace
WORKDIR /workspace

# Copy VLA fine-tuning scripts
COPY vla/ /workspace/vla/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["/bin/bash"]

