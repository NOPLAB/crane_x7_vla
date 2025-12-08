# syntax=docker/dockerfile:1
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# VLA Inference Dockerfile for Remote GPU Services (Runpod/Vast.ai)
#
# This image contains:
# - ROS 2 Humble with CycloneDDS (for unicast VPN communication)
# - Tailscale VPN client (userspace networking mode)
# - PyTorch with CUDA support
# - OpenVLA inference dependencies
# - crane_x7_vla package (inference node only)
#
# Usage:
#   docker build -f Dockerfile.vla-inference -t crane_x7_vla-inference ../..
#
#   docker run --gpus all \
#     -e TS_AUTHKEY=tskey-auth-xxxxx \
#     -e LOCAL_PEER_IP=100.x.x.x \
#     -e VLA_MODEL_PATH=/workspace/models/checkpoint \
#     -e VLA_TASK_INSTRUCTION="pick up the object" \
#     -v /path/to/models:/workspace/models:ro \
#     crane_x7_vla-inference

ARG CUDA_VERSION=12.4.1

# ============================================================================
# Base Stage: CUDA + ROS 2 Humble + Tailscale
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# ----------------------------------------------------------------------------
# Install ROS 2 Humble
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       ros-humble-ros-base \
       ros-humble-rmw-cyclonedds-cpp \
       python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install Tailscale
# ----------------------------------------------------------------------------
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
    | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null \
    && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
    | tee /etc/apt/sources.list.d/tailscale.list \
    && apt-get update && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install Python and system dependencies
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    iproute2 \
    gettext-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install compatible setuptools version
# Note: setuptools>=78 has compatibility issues with packaging library
RUN python3 -m pip install --upgrade pip "setuptools<78" wheel "packaging<24"

WORKDIR /workspace

# ----------------------------------------------------------------------------
# Install PyTorch with CUDA support
# ----------------------------------------------------------------------------
RUN pip3 install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ----------------------------------------------------------------------------
# Install VLA inference dependencies
# ----------------------------------------------------------------------------
COPY ros2/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ----------------------------------------------------------------------------
# Copy OpenVLA/prismatic source for model loading
# ----------------------------------------------------------------------------
COPY vla/src/openvla /workspace/vla/src/openvla

# ----------------------------------------------------------------------------
# Create ROS 2 workspace with crane_x7_vla only
# ----------------------------------------------------------------------------
RUN mkdir -p /workspace/ros2/src

# Copy only the crane_x7_vla package (inference node)
COPY ros2/src/crane_x7_vla /workspace/ros2/src/crane_x7_vla

# Build ROS 2 workspace
# Note: --symlink-install is not used due to setuptools compatibility issues
RUN . /opt/ros/humble/setup.sh && \
    cd /workspace/ros2 && \
    colcon build --packages-select crane_x7_vla

# ----------------------------------------------------------------------------
# Copy configuration and entrypoint
# ----------------------------------------------------------------------------
COPY ros2/docker-remote/cyclonedds.xml /etc/cyclonedds.xml
COPY ros2/docker-remote/entrypoint-vla-inference.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories for model checkpoints and Tailscale state
RUN mkdir -p /workspace/models /var/lib/tailscale

# ----------------------------------------------------------------------------
# Environment variables
# ----------------------------------------------------------------------------
# ROS 2 / DDS configuration
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=/etc/cyclonedds.xml \
    ROS_DOMAIN_ID=42

# Python path for OpenVLA
ENV PYTHONPATH="/workspace/vla/src/openvla"

# CUDA/PyTorch optimization
ENV CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Tailscale configuration (override at runtime)
ENV TS_AUTHKEY="" \
    TS_HOSTNAME="crane-x7-inference" \
    TS_STATE_DIR="/var/lib/tailscale" \
    TS_USERSPACE="true"

# VLA configuration (override at runtime)
ENV VLA_MODEL_PATH="" \
    VLA_TASK_INSTRUCTION="pick up the object" \
    VLA_DEVICE="cuda" \
    HF_TOKEN="" \
    HF_HOME="/root/.cache/huggingface"

# Peer IP for CycloneDDS (override at runtime)
ENV LOCAL_PEER_IP=""

# ----------------------------------------------------------------------------
# Entrypoint and default command
# ----------------------------------------------------------------------------
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "crane_x7_vla", "vla_inference_only.launch.py"]
