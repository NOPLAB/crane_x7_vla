# syntax=docker/dockerfile:1
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Unified VLA Dockerfile for CRANE-X7
# All backends included: openvla, minivla, openpi, openpi-pytorch
#
# Usage:
#   docker build -t crane_x7_vla .

ARG CUDA_VERSION=13.0.0
ARG PYTHON_VERSION=3.11

# ============================================================================
# Builder Stage: Install build dependencies and Python packages
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential \
    ca-certificates \
    software-properties-common \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /build

# Copy requirements files
COPY requirements-*.txt /build/

# Install PyTorch 2.9.1 with CUDA 13.0
RUN python3 -m pip install \
    torch==2.9.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Install JAX with CUDA 12 support
RUN python3 -m pip install "jax[cuda12]==0.5.3" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Flash Attention (optional but recommended for MiniVLA)
RUN python3 -m pip install flash-attn --no-build-isolation || true

# Install all backend requirements
RUN python3 -m pip install -r /build/requirements-openvla.txt && \
    python3 -m pip install -r /build/requirements-minivla.txt && \
    python3 -m pip install -r /build/requirements-openpi.txt && \
    python3 -m pip install -r /build/requirements-openpi-pytorch.txt && \
    # Install dlimp without dependencies (avoids tensorflow version conflict)
    python3 -m pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla" && \
    # Clean up (preserve TensorFlow structure)
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "tests" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "test" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -name "*.pyc" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION} -name "*.pyo" -delete && \
    rm -rf /tmp/* /var/tmp/*

# ============================================================================
# Runtime Stage: Minimal runtime environment
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim \
    ca-certificates sudo \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && apt-get purge -y python3-blinker || true \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip (use python3 -m pip to ensure correct Python version)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /root/.cache

WORKDIR /workspace/vla

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Aggressive cleanup (preserve TensorFlow structure completely)
RUN find /usr/local/lib/python${PYTHON_VERSION} -type d -name "tests" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "test" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "docs" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "examples" ! -path "*/tensorflow*/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION} -name "*.pyc" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION} -name "*.pyo" -delete

# Create directories for data, checkpoints, and logs
RUN mkdir -p /workspace/vla /workspace/data /workspace/checkpoints /workspace/logs

# Copy application source code
COPY --chown=root:root src/crane_x7_vla/ /workspace/vla/src/crane_x7_vla/
COPY --chown=root:root configs/ /workspace/vla/configs/

# GPU optimization settings and PYTHONPATH
ENV CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/workspace/vla/src/crane_x7_vla:/workspace/vla/src

# JAX-specific environment variables (only relevant for openpi backend)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    JAX_PLATFORMS="cuda" \
    JAX_TRACEBACK_FILTERING=off

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
# Development Stage: Add development tools
# ============================================================================
FROM base AS dev

USER root

# Install development tools (apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux htop nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN python3 -m pip install tensorboard

USER ${USERNAME}

EXPOSE 6006

CMD ["/bin/bash"]
