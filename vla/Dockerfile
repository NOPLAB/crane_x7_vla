# Multi-stage Dockerfile for OpenVLA fine-tuning with CRANE-X7 dataset
# Supports GPU-accelerated training with PyTorch and TensorFlow

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

# Base stage: CUDA + Python environment
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
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

# Create workspace
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install OpenVLA package
COPY src/openvla /workspace/openvla
RUN cd /workspace/openvla && \
    pip3 install -e .

# Copy fine-tuning scripts
COPY src/finetune.py src/finetune_config.py src/crane_x7_dataset.py /workspace/

# Create directories for data and checkpoints
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Set environment variables for optimal GPU usage
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Default command
CMD ["/bin/bash"]


# Development stage: Add development tools
FROM base AS dev

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

# Install additional debugging tools
RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    tmux \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default to bash for interactive development
CMD ["/bin/bash"]
