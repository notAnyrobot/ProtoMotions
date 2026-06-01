#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget a single SMPL .motion file to a robot.
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters or environment directories.
#
# Usage: ./scripts/retarget_single_motion_to_robot.sh <proto_python_or_env> <pyroki_python_or_env> <motion_file> <output_dir> <robot_type>
#
# Example:
#   ./scripts/retarget_single_motion_to_robot.sh \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       /path/to/motion.motion /path/to/output g1
#
# Arguments:
#   proto_python:  Path to Python interpreter or env dir with ProtoMotions installed
#   pyroki_python: Path to Python interpreter or env dir with PyRoki installed
#   motion_file:   Path to input .motion file (SMPL format)
#   output_dir:    Directory where all intermediate and final outputs will be saved
#   robot_type:    Target robot: 'g1', 'h1_2', or 'astro'

set -e  # Exit on error

SUPPORTED_ROBOT_TYPES_DISPLAY="'g1', 'h1_2', or 'astro'"

is_supported_robot_type() {
    case "$1" in
        g1|h1_2|astro) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_python_path() {
    local input_path="$1"
    local label="$2"
    local python_path="$input_path"

    if [ -d "$input_path" ]; then
        python_path="${input_path}/bin/python"
    fi

    if [ ! -f "$python_path" ]; then
        echo "Error: ${label} Python not found: $input_path"
        exit 1
    fi

    echo "$python_path"
}

# Parse arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <proto_python_or_env> <pyroki_python_or_env> <motion_file> <output_dir> <robot_type>"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Path to Python interpreter or env dir with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter or env dir with PyRoki installed"
    echo "  motion_file    Path to input .motion file (SMPL format)"
    echo "  output_dir     Directory where all outputs will be saved"
    echo "  robot_type     Target robot: $SUPPORTED_ROBOT_TYPES_DISPLAY"
    echo ""
    echo "Example:"
    echo "  $0 ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python /data/walk.motion /data/retargeted g1"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
MOTION_FILE="$3"
OUTPUT_DIR="$4"
ROBOT_TYPE="$5"

# Validate robot type
if ! is_supported_robot_type "$ROBOT_TYPE"; then
    echo "Error: robot_type must be $SUPPORTED_ROBOT_TYPES_DISPLAY"
    exit 1
fi

# Validate Python interpreters or env directories exist
PROTO_PYTHON="$(resolve_python_path "$PROTO_PYTHON" "ProtoMotions")"
PYROKI_PYTHON="$(resolve_python_path "$PYROKI_PYTHON" "PyRoki")"

# Validate input file exists and is a .motion file
if [ ! -f "$MOTION_FILE" ]; then
    echo "Error: Motion file not found: $MOTION_FILE"
    exit 1
fi

if [[ "$MOTION_FILE" != *.motion ]]; then
    echo "Error: Input file must be a .motion file: $MOTION_FILE"
    exit 1
fi

# Get the motion filename without extension for naming outputs
MOTION_BASENAME=$(basename "$MOTION_FILE" .motion)

# Create output directories
KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints"
RETARGETED_DIR="${OUTPUT_DIR}/retargeted_${ROBOT_TYPE}"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
PROTO_DIR="${OUTPUT_DIR}/retargeted_${ROBOT_TYPE}_proto"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Retargeting Single Motion to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input:               $MOTION_FILE"
echo "Output dir:          $OUTPUT_DIR"
echo "=============================================="

# Step 1: Extract keypoints from single motion (uses ProtoMotions)
echo ""
echo "[Step 1/5] Extracting keypoints from SMPL motion..."
"$PROTO_PYTHON" data/scripts/extract_keypoints_from_single_motion.py \
    "$MOTION_FILE" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format smpl \
    --force-remake

# Step 2: Run PyRoki retargeting (uses PyRoki)
echo ""
echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
"$PYROKI_PYTHON" pyroki/batch_retarget_from_keypoints.py \
    --robot-type "$ROBOT_TYPE" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --output-dir "$RETARGETED_DIR" \
    --no-visualize

# Step 3: Extract contact labels from source motion (uses PyRoki)
echo ""
echo "[Step 3/5] Extracting foot contact labels from source SMPL motion..."
"$PYROKI_PYTHON" pyroki/batch_retarget_from_keypoints.py \
    --robot-type "$ROBOT_TYPE" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --save-contacts-only \
    --contacts-dir "$CONTACTS_DIR"

# Step 4: Convert to ProtoMotions format with contact labels (uses ProtoMotions)
echo ""
echo "[Step 4/5] Converting to ProtoMotions format..."
"$PROTO_PYTHON" data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir "$RETARGETED_DIR" \
    --output-dir "$PROTO_DIR" \
    --robot-type "$ROBOT_TYPE" \
    --contact-labels-dir "$CONTACTS_DIR" \
    --apply-motion-filter \
    --force-remake

# Step 5: Find and report the output file
echo ""
echo "[Step 5/5] Locating output file..."
OUTPUT_MOTION=$(find "$PROTO_DIR" -name "*.motion" -type f | head -1)

if [ -z "$OUTPUT_MOTION" ]; then
    echo "Error: No output .motion file found in $PROTO_DIR"
    echo "The motion may have been filtered out. Check the logs above."
    exit 1
fi

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output motion: $OUTPUT_MOTION"
echo ""
echo "To visualize the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $OUTPUT_MOTION --robot $ROBOT_TYPE --simulator isaacgym"
echo ""
