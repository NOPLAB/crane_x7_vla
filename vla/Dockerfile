# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Multi-stage Dockerfile for VLA fine-tuning with CRANE-X7 dataset
# Includes OpenVLA, OpenPI, and crane_x7_vla packages with GPU support

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.11

# ============================================================================
# Base Stage: CUDA + Python + VLA Dependencies
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim \
    build-essential software-properties-common \
    ca-certificates gnupg sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    && apt-get purge -y python3-blinker || true \
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip and install basic tools
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/vla

# Install base Python dependencies (layer caching optimization)
COPY vla/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy source code (overridden by volume mount in dev mode)
COPY vla/src ./src
COPY vla/configs ./configs
COPY vla/examples ./examples

# Install packages in development mode
# 1. OpenVLA
RUN if [ -d ./src/openvla ]; then \
        pip3 install -e ./src/openvla ; \
    fi

# 2. OpenPI (monorepo: install client first, then remove workspace refs)
RUN if [ -d ./src/openpi/packages/openpi-client ]; then \
        pip3 install -e ./src/openpi/packages/openpi-client ; \
    fi && \
    if [ -d ./src/openpi ]; then \
        sed -i -e '/openpi-client/d' \
               -e '/\[tool.uv.workspace\]/,/members = /d' \
               ./src/openpi/pyproject.toml && \
        pip3 install --ignore-installed blinker -e ./src/openpi ; \
    fi

# 3. CRANE-X7 VLA
RUN if [ -d ./src/crane_x7_vla ]; then \
        pip3 install -e ./src/crane_x7_vla ; \
    fi

# Create directories for data, checkpoints, and logs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# GPU optimization settings
ENV CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Create non-root user matching host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=vla

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R ${USER_ID}:${GROUP_ID} /workspace

USER ${USERNAME}
CMD ["/bin/bash"]


# ============================================================================
# Development Stage: Additional tools for interactive development
# ============================================================================
FROM base AS dev

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=vla

USER root

# Python development tools
RUN pip3 install --no-cache-dir \
    ipython jupyter jupyterlab \
    matplotlib seaborn pandas scikit-learn \
    tensorboard black ruff pre-commit

# System monitoring and debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop nvtop tmux screen iproute2 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8888 6006

USER ${USERNAME}
CMD ["/bin/bash"]
