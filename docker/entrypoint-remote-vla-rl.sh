#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Entrypoint for Remote VLA-RL container with SSH X11 forwarding
#
# This script:
# 1. Starts Tailscale in userspace networking mode (optional)
# 2. Configures SSH with provided public key
# 3. Starts SSH server for X11 forwarding
# 4. Optionally runs a command or drops to shell

# Note: We don't use 'set -e' here to prevent container restart loops
# when non-critical commands fail (e.g., GPU checks, Python imports)

echo "=========================================="
echo "  Remote VLA-RL Container"
echo "  SSH with X11 Forwarding"
echo "=========================================="
echo ""

# ----------------------------------------------------------------------------
# Tailscale Setup (Optional)
# ----------------------------------------------------------------------------
echo "=== Tailscale Setup ==="

if [ -z "$TS_AUTHKEY" ]; then
    echo "INFO: TS_AUTHKEY not set. Tailscale will not be started."
    echo "Container is accessible via local network or port mapping."
    echo ""
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
    if tailscale --socket="$TS_STATE_DIR/tailscaled.sock" up \
        --authkey="$TS_AUTHKEY" \
        --hostname="${TS_HOSTNAME:-crane-x7-vla-rl}" \
        --accept-routes \
        --reset; then
        echo ""
        echo "Tailscale connected!"
        tailscale --socket="$TS_STATE_DIR/tailscaled.sock" status || true
    else
        echo "WARNING: Tailscale connection failed. Continuing without Tailscale."
        echo "Container is still accessible via port mapping."
    fi

    # Get Tailscale IP
    TS_IP=$(tailscale --socket="$TS_STATE_DIR/tailscaled.sock" ip -4 2>/dev/null || echo "")

    if [ -n "$TS_IP" ]; then
        echo ""
        echo "=========================================="
        echo "  Tailscale Connection Info"
        echo "=========================================="
        echo "  Hostname: ${TS_HOSTNAME:-crane-x7-vla-rl}"
        echo "  Tailscale IP: $TS_IP"
        echo "=========================================="
        echo ""
    fi
fi

# ----------------------------------------------------------------------------
# SSH Key Setup
# ----------------------------------------------------------------------------
echo "=== SSH Key Setup ==="

if [ -z "$SSH_PUBLIC_KEY" ]; then
    echo "WARNING: SSH_PUBLIC_KEY not set!"
    echo "You won't be able to log in via SSH."
    echo ""
    echo "Set the SSH_PUBLIC_KEY environment variable to your public key."
    echo "Example: SSH_PUBLIC_KEY=\"ssh-ed25519 AAAA... user@host\""
    echo ""
else
    # Write public key to authorized_keys
    echo "$SSH_PUBLIC_KEY" > /home/vla/.ssh/authorized_keys
    chmod 600 /home/vla/.ssh/authorized_keys
    chown vla:vla /home/vla/.ssh/authorized_keys
    echo "SSH public key configured for user 'vla'"
fi

# Ensure proper permissions
chmod 700 /home/vla/.ssh
chown -R vla:vla /home/vla/.ssh

# ----------------------------------------------------------------------------
# GPU Check
# ----------------------------------------------------------------------------
echo ""
echo "=== GPU Check ==="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
else
    echo "nvidia-smi not found (GPU may not be available)"
fi
echo ""

# ----------------------------------------------------------------------------
# Python Environment Check
# ----------------------------------------------------------------------------
echo "=== Python Environment ==="
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"

# Check CUDA availability safely
CUDA_AVAILABLE=$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'False')
echo "CUDA available: $CUDA_AVAILABLE"
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "CUDA device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
fi
echo ""

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
if [ -n "$TS_IP" ]; then
echo "  Tailscale IP: $TS_IP"
echo "  Tailscale hostname: ${TS_HOSTNAME:-crane-x7-vla-rl}"
fi
echo "  SSH Port: ${SSH_PORT:-22}"
echo "  SSH User: vla"
echo "  Workspace: /workspace"
echo "  VLA-RL outputs: /workspace/vla-rl/outputs"
echo ""
echo "  Connect with X11 forwarding:"
if [ -n "$TS_IP" ]; then
echo "    ssh -X vla@${TS_HOSTNAME:-crane-x7-vla-rl}"
echo "    ssh -X vla@$TS_IP"
else
echo "    ssh -X -p <mapped-port> vla@<host>"
fi
echo "=========================================="
echo ""

# ----------------------------------------------------------------------------
# Execute Command
# ----------------------------------------------------------------------------
if [ $# -eq 0 ] || [ "$1" = "sshd" ]; then
    echo "Starting SSH server..."
    # Start SSH server in foreground
    exec /usr/sbin/sshd -D -e
elif [ "$1" = "bash" ] || [ "$1" = "/bin/bash" ]; then
    echo "Starting interactive shell..."
    exec /bin/bash
else
    echo "Executing: $@"
    exec "$@"
fi
