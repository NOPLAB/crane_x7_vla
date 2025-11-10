# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Ultra-optimized multi-stage Dockerfile for VLA fine-tuning with CRANE-X7 dataset
# Size-optimized version targeting <10GB (reduced from 20GB+)

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.11

# ============================================================================
# Builder Stage: Install build dependencies and Python packages
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

# Copy and install dependencies with optimizations
COPY vla/requirements.txt /build/requirements.txt

# Install optimized dependencies
# Key optimizations:
# - tensorflow -> tensorflow-cpu (1-2GB reduction)
# - opencv-python -> opencv-python-headless (no GUI deps)
# - Reduced PyTorch CUDA arch list
RUN pip3 install --no-cache-dir torch>=2.2.0 torchvision>=0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    transformers==4.57.1 \
    tokenizers==0.22.1 \
    timm==1.0.21 \
    Pillow>=10.0.0 \
    peft==0.17.1 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.43.0 \
    tensorflow-cpu>=2.15.0 \
    tensorflow-datasets==4.9.9 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    wandb>=0.16.0 \
    tqdm>=4.66.0 && \
    # Aggressively clean pip and Python caches
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -name "*.pyo" -delete && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Copy source code
COPY vla/src /build/src
COPY vla/configs /build/configs
COPY vla/examples /build/examples

# Install packages in development mode (builds .egg-info and caches)
RUN if [ -d /build/src/openvla ]; then \
    pip3 install --no-cache-dir -e /build/src/openvla ; \
    fi

RUN if [ -d /build/src/openpi/packages/openpi-client ]; then \
    pip3 install --no-cache-dir -e /build/src/openpi/packages/openpi-client ; \
    fi && \
    if [ -d /build/src/openpi ]; then \
    sed -i -e '/openpi-client/d' \
    -e '/\[tool.uv.workspace\]/,/members = /d' \
    /build/src/openpi/pyproject.toml && \
    pip3 install --no-cache-dir --ignore-installed blinker -e /build/src/openpi ; \
    fi

RUN if [ -d /build/src/crane_x7_vla ]; then \
    pip3 install --no-cache-dir -e /build/src/crane_x7_vla ; \
    fi && \
    # Final cleanup of build artifacts
    find /build -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /build -name "*.pyc" -delete && \
    find /build -name "*.pyo" -delete

# ============================================================================
# Runtime Stage: Minimal runtime environment
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS base

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
    python${PYTHON_VERSION}-distutils \
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

# Upgrade pip (minimal version)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /root/.cache

WORKDIR /workspace/vla

# Copy Python packages from builder (with aggressive filtering)
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Aggressive cleanup of copied packages
RUN find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -name "*.pyo" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION}/dist-packages -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    # Remove TensorFlow examples and large unnecessary files
    rm -rf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow/examples \
           /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow/include \
           /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorboard 2>/dev/null || true && \
    # Clean CUDA libraries (keep only essential runtime)
    rm -rf /usr/local/cuda/samples \
           /usr/local/cuda/doc \
           /usr/local/cuda/nsight* 2>/dev/null || true

# Copy source code (symlinked to installed packages)
COPY --from=builder /build/src ./src
COPY --from=builder /build/configs ./configs
COPY --from=builder /build/examples ./examples

# Create directories for data, checkpoints, and logs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# GPU optimization settings (reduced CUDA arch list for smaller binaries)
ENV CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" \
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
# Development Stage: Add development tools (optional)
# ============================================================================
FROM base AS dev

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux htop nano \
    && pip3 install --no-cache-dir \
    jupyter \
    tensorboard \
    ipywidgets \
    matplotlib \
    seaborn \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER ${USERNAME}

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

CMD ["/bin/bash"]
