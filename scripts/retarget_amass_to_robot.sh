#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget AMASS SMPL motions to a robot.
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters or environment directories.
#
# Usage: ./scripts/retarget_amass_to_robot.sh <proto_python_or_env> <pyroki_python_or_env> <amass_pt_file> <robot_type> [skip_freq]
#
# Example:
#   ./scripts/retarget_amass_to_robot.sh \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       /path/to/amass.pt g1 15
#
# Arguments:
#   proto_python:  Path to Python interpreter or env dir with ProtoMotions installed
#   pyroki_python: Path to Python interpreter or env dir with PyRoki installed
#   amass_pt_file: Path to packaged AMASS MotionLib .pt file under smpl/<split>/
#   robot_type:    Target robot: 'g1', 'h1_2', or 'astro'
#   skip_freq:     (Optional) Skip every N motions for subset processing (default: 1 = all motions)
#   PROTO_PYROKI_REPO: Optional PyRoki checkout root (default: sibling ../pyroki)
#   PROTO_RETARGET_ROOT: Optional retarget data root (default: inferred from smpl/<split>/)
#   PROTO_RETARGET_SPLIT: Optional split name override (default: inferred from path/filename)

set -e  # Exit on error

SUPPORTED_ROBOT_TYPES_DISPLAY="'g1', 'h1_2', or 'astro'"
SUPPORTED_SPLITS_DISPLAY="'train', 'test', or 'validation'"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PYROKI_REPO="$(cd "${REPO_ROOT}/.." && pwd)/pyroki"
PYROKI_REPO="${PROTO_PYROKI_REPO:-$DEFAULT_PYROKI_REPO}"

is_supported_robot_type() {
    case "$1" in
        g1|h1_2|astro) return 0 ;;
        *) return 1 ;;
    esac
}

is_supported_split_name() {
    case "$1" in
        train|test|validation) return 0 ;;
        *) return 1 ;;
    esac
}

infer_split_name() {
    local file_path="$1"
    local parent_name
    local file_stem

    parent_name="$(basename "$(dirname "$file_path")")"
    if is_supported_split_name "$parent_name"; then
        echo "$parent_name"
        return 0
    fi

    file_stem="$(basename "$file_path" .pt)"
    case "$file_stem" in
        amass_smpl_train|train) echo "train" ;;
        amass_smpl_test|test) echo "test" ;;
        amass_smpl_validation|validation) echo "validation" ;;
        *) return 1 ;;
    esac
}

infer_retarget_root() {
    local file_path="$1"
    local file_dir
    local parent_dir
    local parent_name

    file_dir="$(dirname "$file_path")"
    parent_dir="$(dirname "$file_dir")"
    parent_name="$(basename "$parent_dir")"

    if [ "$(basename "$file_dir")" = "smpl" ]; then
        dirname "$file_dir"
    elif is_supported_split_name "$(basename "$file_dir")" && [ "$parent_name" = "smpl" ]; then
        dirname "$parent_dir"
    else
        dirname "$file_dir"
    fi
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

validate_pyroki_repo() {
    if [ ! -d "${PYROKI_REPO}/src/pyroki" ]; then
        echo "Error: PyRoki repo not found or invalid: $PYROKI_REPO"
        echo "Set PROTO_PYROKI_REPO to a checkout containing src/pyroki."
        exit 1
    fi
}

run_pyroki_python() {
    PYTHONPATH="${PYROKI_REPO}/src" "$PYROKI_PYTHON" "$@"
}

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <proto_python_or_env> <pyroki_python_or_env> <amass_pt_file> <robot_type> [skip_freq]"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Path to Python interpreter or env dir with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter or env dir with PyRoki installed"
    echo "  amass_pt_file  Path to packaged AMASS MotionLib .pt file under smpl/<split>/"
    echo "  robot_type     Target robot: $SUPPORTED_ROBOT_TYPES_DISPLAY"
    echo "  skip_freq      (Optional) Skip every N motions (default: 1 = all motions)"
    echo "  PROTO_PYROKI_REPO Optional PyRoki checkout root"
    echo "                 Default: $DEFAULT_PYROKI_REPO"
    echo "  PROTO_RETARGET_ROOT Optional retarget data root"
    echo "                 Default: inferred from smpl/<split>/"
    echo "  PROTO_RETARGET_SPLIT Optional split override: $SUPPORTED_SPLITS_DISPLAY"
    echo ""
    echo "Example:"
    echo "  $0 ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python /data/protomotions/smpl/train/amass_smpl_train.pt g1 1"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
AMASS_PT_FILE="$3"
ROBOT_TYPE="$4"
SKIP_FREQ="${5:-1}"

# Validate robot type
if ! is_supported_robot_type "$ROBOT_TYPE"; then
    echo "Error: robot_type must be $SUPPORTED_ROBOT_TYPES_DISPLAY"
    exit 1
fi

# Validate Python interpreters or env directories exist
PROTO_PYTHON="$(resolve_python_path "$PROTO_PYTHON" "ProtoMotions")"
PYROKI_PYTHON="$(resolve_python_path "$PYROKI_PYTHON" "PyRoki")"

# Validate input file exists
if [ ! -f "$AMASS_PT_FILE" ]; then
    echo "Error: AMASS .pt file not found: $AMASS_PT_FILE"
    exit 1
fi

SPLIT_NAME="${PROTO_RETARGET_SPLIT:-$(infer_split_name "$AMASS_PT_FILE" || true)}"

if ! is_supported_split_name "$SPLIT_NAME"; then
    echo "Error: Could not infer AMASS split for: $AMASS_PT_FILE"
    echo "Place the file under smpl/<split>/ or set PROTO_RETARGET_SPLIT to $SUPPORTED_SPLITS_DISPLAY."
    exit 1
fi

DATA_ROOT="${PROTO_RETARGET_ROOT:-$(infer_retarget_root "$AMASS_PT_FILE")}"
SMPL_SPLIT_DIR="${DATA_ROOT}/smpl/${SPLIT_NAME}"
ROBOT_SPLIT_DIR="${DATA_ROOT}/${ROBOT_TYPE}/${SPLIT_NAME}"
KEYPOINTS_DIR="${SMPL_SPLIT_DIR}/keypoints-for-retarget"
RETARGETED_DIR="${ROBOT_SPLIT_DIR}/pyroki-retargeted-${ROBOT_TYPE}"
CONTACTS_DIR="${SMPL_SPLIT_DIR}/contacts"
PROTO_DIR="${ROBOT_SPLIT_DIR}/proto-${ROBOT_TYPE}"
FINAL_PT="${ROBOT_SPLIT_DIR}/proto-${ROBOT_TYPE}.pt"

echo "=============================================="
echo "Retargeting AMASS to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "PyRoki repo:         $PYROKI_REPO"
echo "Input:               $AMASS_PT_FILE"
echo "Retarget root:       $DATA_ROOT"
echo "Split:               $SPLIT_NAME"
echo "SMPL split dir:      $SMPL_SPLIT_DIR"
echo "Robot split dir:     $ROBOT_SPLIT_DIR"
echo "Skip freq:           $SKIP_FREQ (1 = all motions)"
echo "=============================================="

# Step 1: Extract keypoints from packaged MotionLib (uses ProtoMotions)
echo ""
echo "[Step 1/5] Extracting keypoints from SMPL motions..."
"$PROTO_PYTHON" data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    "$AMASS_PT_FILE" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq "$SKIP_FREQ"

# Step 2: Run PyRoki retargeting (uses PyRoki)
echo ""
echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
validate_pyroki_repo
run_pyroki_python pyroki/batch_retarget_from_keypoints.py \
    --robot-type "$ROBOT_TYPE" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --output-dir "$RETARGETED_DIR" \
    --no-visualize \
    --skip-existing

# Step 3: Extract contact labels from source motions (uses PyRoki)
echo ""
echo "[Step 3/5] Extracting foot contact labels from source SMPL motions..."
run_pyroki_python pyroki/batch_retarget_from_keypoints.py \
    --robot-type "$ROBOT_TYPE" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --save-contacts-only \
    --contacts-dir "$CONTACTS_DIR" \
    --skip-existing

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

# Step 5: Package into MotionLib (uses ProtoMotions)
echo ""
echo "[Step 5/5] Packaging into MotionLib..."
"$PROTO_PYTHON" protomotions/components/motion_lib.py \
    --motion-path "$PROTO_DIR" \
    --output-file "$FINAL_PT"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output MotionLib: $FINAL_PT"
echo ""
echo "To verify the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $FINAL_PT --robot $ROBOT_TYPE --simulator isaacgym"
echo ""
