#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Entrypoint for VLA inference container with Tailscale VPN
#
# This script:
# 1. Starts Tailscale in userspace networking mode
# 2. Configures CycloneDDS for unicast communication
# 3. Sets up ROS 2 workspace
# 4. Validates VLA model configuration
# 5. Executes the launch command

set -e

echo "=========================================="
echo "  VLA Inference Container Startup"
echo "=========================================="
echo ""

# ----------------------------------------------------------------------------
# Tailscale Setup
# ----------------------------------------------------------------------------
echo "=== Tailscale Setup ==="

if [ -z "$TS_AUTHKEY" ]; then
    echo "WARNING: TS_AUTHKEY not set. Tailscale will not be started."
    echo "Set TS_AUTHKEY environment variable to enable VPN connectivity."
    echo "Get an auth key from: https://login.tailscale.com/admin/settings/keys"
    echo ""
else
    echo "Starting Tailscale in userspace networking mode..."

    # Create state directory
    mkdir -p "$TS_STATE_DIR"

    # Start tailscaled in background with userspace networking
    # userspace-networking is required for containers without /dev/net/tun access
    tailscaled \
        --state="$TS_STATE_DIR/tailscaled.state" \
        --socket="$TS_STATE_DIR/tailscaled.sock" \
        --tun=userspace-networking &

    # Wait for tailscaled to be ready
    echo "Waiting for tailscaled to start..."
    sleep 3

    # Connect to Tailscale network
    echo "Connecting to Tailscale network..."
    tailscale --socket="$TS_STATE_DIR/tailscaled.sock" up \
        --authkey="$TS_AUTHKEY" \
        --hostname="${TS_HOSTNAME:-crane-x7-inference}" \
        --accept-routes \
        --reset

    echo ""
    echo "Tailscale connected!"
    tailscale --socket="$TS_STATE_DIR/tailscaled.sock" status

    # Get our Tailscale IP
    TS_IP=$(tailscale --socket="$TS_STATE_DIR/tailscaled.sock" ip -4 2>/dev/null || echo "unknown")
    echo ""
    echo "This machine's Tailscale IP: $TS_IP"
    echo "Share this IP with the local robot for CycloneDDS configuration."
fi
echo ""

# ----------------------------------------------------------------------------
# CycloneDDS Configuration
# ----------------------------------------------------------------------------
echo "=== CycloneDDS Configuration ==="

CYCLONEDDS_CONFIG="/etc/cyclonedds.xml"

if [ -n "$LOCAL_PEER_IP" ]; then
    echo "Configuring CycloneDDS unicast with peer: $LOCAL_PEER_IP"

    # Generate CycloneDDS config with the peer IP
    cat > "$CYCLONEDDS_CONFIG" << EOF
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="false" />
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="$LOCAL_PEER_IP" />
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
    </Discovery>
    <Internal>
      <SocketReceiveBufferSize min="1MB" max="32MB"/>
      <SocketSendBufferSize min="1MB" max="32MB"/>
    </Internal>
  </Domain>
</CycloneDDS>
EOF

    echo "CycloneDDS config generated at: $CYCLONEDDS_CONFIG"
else
    echo "WARNING: LOCAL_PEER_IP not set."
    echo "Set LOCAL_PEER_IP to the Tailscale IP of the local robot machine."
    echo "Using default CycloneDDS config (may not work for VPN communication)."
fi
echo ""

# ----------------------------------------------------------------------------
# ROS 2 Workspace Setup
# ----------------------------------------------------------------------------
echo "=== ROS 2 Workspace Setup ==="

ROS2_WORKSPACE=/workspace/ros2

if [ -f "$ROS2_WORKSPACE/install/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    source "$ROS2_WORKSPACE/install/setup.bash"
    echo "ROS 2 workspace sourced successfully."
else
    echo "ERROR: ROS 2 workspace not found at $ROS2_WORKSPACE"
    echo "Make sure the workspace was built during Docker image creation."
    exit 1
fi

echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "CYCLONEDDS_URI: ${CYCLONEDDS_URI:-not set}"
echo ""

# ----------------------------------------------------------------------------
# GPU Check
# ----------------------------------------------------------------------------
echo "=== GPU Check ==="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
else
    echo "nvidia-smi not found (GPU may not be available)"
fi
echo ""

# ----------------------------------------------------------------------------
# VLA Model Check
# ----------------------------------------------------------------------------
echo "=== VLA Model Check ==="

if [ -z "$VLA_MODEL_PATH" ]; then
    echo "ERROR: VLA_MODEL_PATH not set!"
    echo "Set VLA_MODEL_PATH to the path of your fine-tuned model."
    echo ""
    echo "Examples:"
    echo "  Local path:        VLA_MODEL_PATH=/workspace/models/checkpoint-1500"
    echo "  HuggingFace Hub:   VLA_MODEL_PATH=your-username/crane_x7_openvla"
    echo ""
    echo "Available models in /workspace/models (if mounted):"
    ls -la /workspace/models 2>/dev/null || echo "  (no models found)"
    echo ""
    echo "Available models in /workspace/vla/outputs (if mounted):"
    ls -la /workspace/vla/outputs 2>/dev/null || echo "  (no models found)"
    exit 1
fi

# Check if VLA_MODEL_PATH is a HuggingFace Hub ID (format: user/model)
is_hf_hub=false
if [[ "$VLA_MODEL_PATH" =~ ^[^/]+/[^/]+$ ]] && [[ ! "$VLA_MODEL_PATH" =~ ^/ ]]; then
    is_hf_hub=true
    echo "HuggingFace Hub model: $VLA_MODEL_PATH"
    echo "Model will be downloaded on first use."
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN is set (for private repositories)"
    else
        echo "HF_TOKEN not set (public repositories only)"
    fi
else
    # Local path - check if it exists
    if [ ! -d "$VLA_MODEL_PATH" ]; then
        echo "ERROR: Model path does not exist: $VLA_MODEL_PATH"
        echo "Make sure to mount the model directory to the container."
        echo ""
        echo "Example docker run:"
        echo "  docker run -v /path/to/models:/workspace/models:ro ..."
        exit 1
    fi
    echo "Local model path: $VLA_MODEL_PATH"

    # Check for LoRA adapter
    if [ -f "$VLA_MODEL_PATH/lora_adapters/adapter_config.json" ]; then
        echo "LoRA adapter detected: $VLA_MODEL_PATH/lora_adapters/"
    fi

    # Check for dataset statistics
    if [ -f "$VLA_MODEL_PATH/dataset_statistics.json" ]; then
        echo "Dataset statistics found: $VLA_MODEL_PATH/dataset_statistics.json"
    fi
fi

echo "Task instruction: ${VLA_TASK_INSTRUCTION:-pick up the object}"
echo "Device: ${VLA_DEVICE:-cuda}"
echo ""

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
echo "  Tailscale: ${TS_IP:-not connected}"
echo "  Peer IP:   ${LOCAL_PEER_IP:-not set}"
echo "  Domain ID: ${ROS_DOMAIN_ID:-0}"
echo "  Model:     ${VLA_MODEL_PATH}"
echo "  Task:      ${VLA_TASK_INSTRUCTION:-pick up the object}"
echo "  Device:    ${VLA_DEVICE:-cuda}"
echo "=========================================="
echo ""

# ----------------------------------------------------------------------------
# Execute Command
# ----------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    exec "$@"
else
    echo "No command specified. Starting default launch..."
    exec ros2 launch crane_x7_vla vla_inference_only.launch.py
fi
