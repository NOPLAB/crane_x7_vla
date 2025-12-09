#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

# Generate rosdep packages list for Docker caching
# This script should be run inside a ROS 2 container or on a system with ROS 2 installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="${ROS2_DIR}/rosdep_packages.txt"

# Source ROS 2 environment
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Update rosdep database
rosdep update --rosdistro=humble 2>/dev/null || true

# Generate package list
cd "${ROS2_DIR}"

echo "# Auto-generated rosdep packages list" > "$OUTPUT_FILE"
echo "# Generated on: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_FILE"
echo "# Regenerate with: ./scripts/generate_rosdep_packages.sh" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Get rosdep keys and resolve to apt packages
rosdep keys --from-paths src --ignore-src 2>/dev/null | \
    xargs -I {} sh -c 'rosdep resolve {} 2>/dev/null || true' | \
    grep -v '^#' | \
    sort -u >> "$OUTPUT_FILE"

echo "Generated: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
