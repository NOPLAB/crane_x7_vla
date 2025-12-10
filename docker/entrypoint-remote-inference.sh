#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Entrypoint for VLA inference container using rosbridge (WebSocket)
#
# This script:
# 1. Starts Tailscale in userspace networking mode
# 2. Waits for local peer (rosbridge server) to become reachable
# 3. Validates VLA model configuration
# 4. Executes the VLA inference rosbridge client

set -e

echo "=========================================="
echo "  VLA Inference Container (rosbridge)"
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
    # If no Tailscale, use ROSBRIDGE_HOST as-is (might be an IP or hostname)
    RESOLVED_HOST="$ROSBRIDGE_HOST"
else
    echo "Starting Tailscale in userspace networking mode..."

    # Create state directory
    mkdir -p "$TS_STATE_DIR"

    # Start tailscaled in background with userspace networking
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

    # Get Tailscale IP
    ACTUAL_TS_IP=$(tailscale --socket="$TS_STATE_DIR/tailscaled.sock" ip -4 2>/dev/null || echo "")

    echo ""
    echo "=========================================="
    echo "  Tailscale Connection Info"
    echo "=========================================="
    echo "  Hostname: ${TS_HOSTNAME:-crane-x7-inference}"
    echo "  Tailscale IP: $ACTUAL_TS_IP"
    echo "=========================================="
    echo ""

    # Wait for local peer to become reachable
    echo "=== Waiting for Local Peer ==="

    LOCAL_PEER_HOSTNAME="${ROSBRIDGE_HOST:-crane-x7-local}"
    PEER_WAIT_TIMEOUT="${PEER_WAIT_TIMEOUT:-300}"

    RESOLVED_HOST=$(/usr/local/bin/wait-for-peer.sh "$LOCAL_PEER_HOSTNAME" \
        --timeout "$PEER_WAIT_TIMEOUT" --no-ping \
        --socket "$TS_STATE_DIR/tailscaled.sock")

    if [ -z "$RESOLVED_HOST" ]; then
        echo "ERROR: Failed to resolve local peer '$LOCAL_PEER_HOSTNAME'"
        echo ""
        echo "Please ensure the local robot is running with Tailscale connected."
        exit 1
    fi

    echo "Local peer resolved: $LOCAL_PEER_HOSTNAME -> $RESOLVED_HOST"

    # Save the peer IP for display
    PEER_IP="$RESOLVED_HOST"

    # In userspace networking mode, we need to use tailscale nc for TCP connections
    # Set up a local port forward using socat + tailscale nc
    echo ""
    echo "=== Setting up TCP Port Forward ==="
    LOCAL_FORWARD_PORT="${ROSBRIDGE_PORT:-9090}"

    # Kill any existing socat processes
    pkill -f "socat.*LISTEN:$LOCAL_FORWARD_PORT" 2>/dev/null || true

    # Start socat to forward local port to remote via tailscale nc
    # This creates: localhost:9090 -> tailscale nc -> crane-x7-local:9090
    echo "Starting TCP forward: localhost:$LOCAL_FORWARD_PORT -> $PEER_IP:$LOCAL_FORWARD_PORT"
    socat TCP-LISTEN:$LOCAL_FORWARD_PORT,reuseaddr,fork \
        EXEC:"tailscale --socket=$TS_STATE_DIR/tailscaled.sock nc $PEER_IP $LOCAL_FORWARD_PORT" &
    SOCAT_PID=$!
    echo "Port forward started (PID: $SOCAT_PID)"

    # Give socat a moment to start
    sleep 1

    # Verify socat is running
    if kill -0 $SOCAT_PID 2>/dev/null; then
        echo "Port forward is active"
    else
        echo "WARNING: Port forward may not have started correctly"
    fi

    # Use localhost for the rosbridge connection (goes through socat -> tailscale nc)
    RESOLVED_HOST="127.0.0.1"
fi

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
        exit 1
    fi
    echo "Local model path: $VLA_MODEL_PATH"
fi

echo "Task instruction: ${VLA_TASK_INSTRUCTION:-pick up the object}"
echo "Device: ${VLA_DEVICE:-cuda}"
echo "Inference rate: ${VLA_INFERENCE_RATE:-10.0} Hz"
echo ""

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
echo "  Rosbridge:      ${RESOLVED_HOST}:${ROSBRIDGE_PORT:-9090}"
if [ -n "$PEER_IP" ]; then
echo "  (via socat -> tailscale nc -> ${PEER_IP}:${ROSBRIDGE_PORT:-9090})"
fi
echo "  Model:          ${VLA_MODEL_PATH}"
echo "  Task:           ${VLA_TASK_INSTRUCTION:-pick up the object}"
echo "  Device:         ${VLA_DEVICE:-cuda}"
echo "  Rate:           ${VLA_INFERENCE_RATE:-10.0} Hz"
echo "=========================================="
echo ""

# ----------------------------------------------------------------------------
# Execute Command
# ----------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    # Pass rosbridge host as environment variable
    export ROSBRIDGE_HOST="$RESOLVED_HOST"
    exec "$@"
else
    echo "Starting VLA inference rosbridge client..."
    exec python3 /workspace/scripts/vla_inference_rosbridge.py \
        --rosbridge-host "$RESOLVED_HOST" \
        --rosbridge-port "${ROSBRIDGE_PORT:-9090}" \
        --model-path "$VLA_MODEL_PATH" \
        --task-instruction "${VLA_TASK_INSTRUCTION:-pick up the object}" \
        --device "${VLA_DEVICE:-cuda}" \
        --unnorm-key "${VLA_UNNORM_KEY:-crane_x7}" \
        --inference-rate "${VLA_INFERENCE_RATE:-10.0}"
fi
