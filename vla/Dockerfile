# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Multi-stage Dockerfile for VLA fine-tuning with CRANE-X7 dataset
# Supports GPU-accelerated training with PyTorch and TensorFlow
# Includes OpenVLA, OpenPI, and crane_x7_vla packages

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

# ============================================================================
# Base Stage: CUDA + Python + VLA Dependencies
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip and install basic tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace directory structure
WORKDIR /workspace/vla

# Install Python dependencies with caching optimization
# Copy requirements first for better layer caching
COPY vla/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy source code to workspace
# Note: In development mode, this will be overridden by volume mount
COPY vla/src ./src
COPY vla/configs ./configs
COPY vla/examples ./examples

# Install OpenVLA package
RUN if [ -d ./src/openvla ]; then \
        cd ./src/openvla && pip3 install -e . ; \
    fi

# Install OpenPI package
RUN if [ -d ./src/openpi ]; then \
        cd ./src/openpi && pip3 install -e . ; \
    fi

# Install crane_x7_vla package
RUN if [ -d ./src/crane_x7_vla ]; then \
        cd ./src/crane_x7_vla && pip3 install -e . ; \
    fi

# Create directories for data, checkpoints, and logs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Set environment variables for optimal GPU usage
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Create a user that matches host user (UID/GID will be passed at build time)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=vla

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true \
    && useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chown -R ${USER_ID}:${GROUP_ID} /workspace

USER ${USERNAME}

CMD ["/bin/bash"]


# ============================================================================
# Development Stage: Add development tools
# ============================================================================
FROM base AS dev

# Need to re-declare ARGs from base stage for use in dev stage
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=vla

# Switch back to root to install additional packages
USER root

# Install development dependencies
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    tensorboard \
    black \
    ruff \
    pre-commit

# Install additional debugging and system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    nvtop \
    tmux \
    screen \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Switch back to the non-root user
USER ${USERNAME}

# Default to bash for interactive development
CMD ["/bin/bash"]
