FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install -y --no-install-recommends \
    xserver-xorg

# Cache installtion
RUN apt-get update && apt install -y --no-install-recommends \
    ros-humble-moveit

ENV ROS2_DEPENDENCIES_DIR=/tmp/ros2_dependencies

# Install Python dependencies
RUN apt-get update && apt install -y --no-install-recommends \
    python3-pip && \
    pip3 install --no-cache-dir \
    tensorflow \
    numpy \
    opencv-python

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

# ============================================================================
# VLA Fine-tuning Stage (separate from ROS2)
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

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

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

