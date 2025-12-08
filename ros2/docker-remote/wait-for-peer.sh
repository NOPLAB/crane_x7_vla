#!/bin/sh
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Wait for a Tailscale peer to become reachable via DNS or Tailscale status.
# Works in containers sharing Tailscale's network namespace or with userspace networking.
#
# Usage:
#   wait-for-peer.sh <hostname> [--timeout <seconds>] [--no-ping] [--socket <path>]
#
# Example:
#   wait-for-peer.sh crane-x7-inference --timeout 300
#   wait-for-peer.sh crane-x7-inference --no-ping
#   wait-for-peer.sh crane-x7-local --socket /var/lib/tailscale/tailscaled.sock

# Don't use set -e as we expect some commands to fail during retry loop

# Parse arguments
HOSTNAME=""
TIMEOUT=300
INTERVAL=5
PING_TIMEOUT=3
SKIP_PING=false
TS_SOCKET=""

while [ $# -gt 0 ]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --no-ping)
            SKIP_PING=true
            shift
            ;;
        --socket)
            TS_SOCKET="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [ -z "$HOSTNAME" ]; then
                HOSTNAME="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$HOSTNAME" ]; then
    echo "Usage: $0 <hostname> [--timeout <seconds>] [--no-ping] [--socket <path>]" >&2
    exit 1
fi

# Function to resolve via Tailscale status command
resolve_via_tailscale() {
    _rvt_target="$1"
    _rvt_ip=""

    if [ -n "$TS_SOCKET" ] && [ -S "$TS_SOCKET" ]; then
        # Use tailscale status with custom socket
        _rvt_ip=$(tailscale --socket="$TS_SOCKET" status 2>/dev/null | grep -i "$_rvt_target" | awk '{print $1}' | head -1) || true
    elif command -v tailscale >/dev/null 2>&1; then
        # Try default tailscale command
        _rvt_ip=$(tailscale status 2>/dev/null | grep -i "$_rvt_target" | awk '{print $1}' | head -1) || true
    fi

    echo "$_rvt_ip"
}

# Function to resolve hostname to IP (returns empty string on failure)
resolve_hostname() {
    _rh_target="$1"
    _rh_ip=""

    # Method 1: Try Tailscale status first (works in userspace mode)
    _rh_ip=$(resolve_via_tailscale "$_rh_target")
    if [ -n "$_rh_ip" ]; then
        echo "$_rh_ip"
        return 0
    fi

    # Method 2: Try getent (works when sharing Tailscale network namespace)
    _rh_ip=$(getent hosts "$_rh_target" 2>/dev/null | awk '{print $1}' | head -1) || true
    if [ -n "$_rh_ip" ]; then
        echo "$_rh_ip"
        return 0
    fi

    # Method 3: Fallback to ping-based resolution
    _rh_ip=$(ping -c 1 -W 1 "$_rh_target" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true
    if [ -n "$_rh_ip" ]; then
        echo "$_rh_ip"
        return 0
    fi

    # Return empty string (not an error, just not found yet)
    echo ""
    return 0
}

# Function to check connectivity with ping
ping_check() {
    _pc_ip="$1"
    ping -c 1 -W "$PING_TIMEOUT" "$_pc_ip" > /dev/null 2>&1
}

# Main wait loop
echo "Waiting for peer '$HOSTNAME' to become reachable..." >&2
echo "Timeout: ${TIMEOUT}s, Interval: ${INTERVAL}s, Skip ping: $SKIP_PING" >&2
if [ -n "$TS_SOCKET" ]; then
    echo "Tailscale socket: $TS_SOCKET" >&2
fi

elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    # Try to resolve hostname
    ip=$(resolve_hostname "$HOSTNAME")

    if [ -n "$ip" ]; then
        if [ "$SKIP_PING" = "true" ]; then
            # DNS resolution is enough
            echo "Resolved '$HOSTNAME' -> $ip (ping verification skipped)" >&2
            echo "$ip"
            exit 0
        else
            echo "  Resolved $HOSTNAME -> $ip, verifying connectivity..." >&2
            if ping_check "$ip"; then
                echo "Peer '$HOSTNAME' is reachable at $ip" >&2
                echo "$ip"
                exit 0
            else
                echo "  Ping to $ip failed, retrying..." >&2
            fi
        fi
    else
        echo "  [$elapsed/${TIMEOUT}s] Cannot resolve '$HOSTNAME', waiting..." >&2
    fi

    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done

echo "ERROR: Timeout waiting for peer '$HOSTNAME'" >&2
exit 1
