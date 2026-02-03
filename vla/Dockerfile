# syntax=docker/dockerfile:1
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Unified VLA Dockerfile for CRANE-X7
# All backends included: openvla, minivla, pi0, pi0.5

ARG CUDA_VERSION=12.8.0
ARG CUDA_SHORT=cu128
ARG PYTORCH_VERSION=2.9.1
ARG PYTHON_VERSION=3.12
ARG PYTHON_SHORT=312

# ============================================================================
# Builder Stage: Install build dependencies and Python packages
# ============================================================================
# Use runtime image for both stages (base lacks CUDA runtime libraries)
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG PYTHON_SHORT
ARG CUDA_SHORT
ARG PYTORCH_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

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
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock /build/

# Install dependencies using uv (frozen from lock file)
# --no-dev excludes development dependencies
# --system installs to system Python instead of venv
RUN uv sync --frozen --no-dev --no-install-project --python ${PYTHON_VERSION}

# Install Flash Attention from prebuilt wheel
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_RELEASE=v0.6.8
RUN PYTORCH_SHORT=$(echo ${PYTORCH_VERSION} | cut -d. -f1,2) && \
    WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}+${CUDA_SHORT}torch${PYTORCH_SHORT}-cp${PYTHON_SHORT}-cp${PYTHON_SHORT}-linux_x86_64.whl" && \
    wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FLASH_ATTN_RELEASE}/${WHEEL_NAME}" && \
    uv pip install --system "${WHEEL_NAME}" && \
    rm -f "${WHEEL_NAME}"

# Install dlimp without dependencies (avoids tensorflow version conflict)
RUN uv pip install --system --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# Install LeRobot without dependencies (avoids PyTorch downgrade)
RUN uv pip install --system --no-deps "lerobot==0.4.2"

# Force numpy 1.x at the end (tensorflow/other packages install 2.x which breaks on WSL2)
RUN uv pip install --system "numpy>=1.26.0,<2.0.0" --reinstall

# Basic cleanup
RUN rm -rf /tmp/* /var/tmp/* /root/.cache ~/.cache

# ============================================================================
# Runtime Stage: Minimal runtime environment
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    UV_LINK_MODE=copy

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim \
    ca-certificates sudo \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
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

# Install uv for runtime package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/vla

# Copy Python packages from builder
COPY --from=builder /build/.venv /workspace/vla/.venv

# Final cleanup in runtime stage
RUN find /workspace/vla/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/vla/.venv -name "*.pyc" -delete && \
    find /workspace/vla/.venv -name "*.pyo" -delete && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Create directories for data, checkpoints, and logs
RUN mkdir -p /workspace/vla /workspace/data /workspace/checkpoints /workspace/logs

# Copy application source code
COPY --chown=root:root src/crane_x7_vla/ /workspace/vla/src/crane_x7_vla/
COPY --chown=root:root configs/ /workspace/vla/configs/

# Install transformers_replace for Pi0/Pi0.5 (OpenPI compatibility)
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp -r /workspace/vla/src/crane_x7_vla/backends/pi0/models_pytorch/transformers_replace/* \
    ${SITE_PACKAGES}/transformers/ || \
    cp -r /workspace/vla/src/crane_x7_vla/backends/pi0/models_pytorch/transformers_replace/* \
    /workspace/vla/.venv/lib/python${PYTHON_VERSION}/site-packages/transformers/

# GPU optimization settings and PYTHONPATH
ENV CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/workspace/vla/src/crane_x7_vla:/workspace/vla/src \
    VIRTUAL_ENV=/workspace/vla/.venv \
    PATH="/workspace/vla/.venv/bin:$PATH"

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

USER ${USERNAME}

CMD ["/bin/bash"]
