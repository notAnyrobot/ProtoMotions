#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Retarget a packaged AMASS SMPL MotionLib split to a robot from inside a
# ProtoMotions HPC container.
#
# Usage:
#   ./scripts/docker/retarget_amass_hpc.sh [split] [robot] [skip_freq]
#
# Defaults:
#   split=test, robot=astro, skip_freq=1

set -euo pipefail

SUPPORTED_ROBOTS_DISPLAY="'g1', 'h1_2', or 'astro'"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

fail() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $0 [split] [robot] [skip_freq]

Run the AMASS -> keypoints -> PyRoki -> ProtoMotions packaging flow from inside
a ProtoMotions HPC container.

Arguments:
  split      Optional AMASS split folder under smpl/ (default: test).
             Examples: test, validation, train, sfu, cmu, accad.
  robot      Optional target robot: $SUPPORTED_ROBOTS_DISPLAY (default: astro)
  skip_freq  Optional motion stride for keypoint extraction (default: 1)

Environment overrides:
  MOTION_DATASETS          Dataset root mounted in the container.
                           Default: /data/share/motion_datasets
  PROTO_RETARGET_ROOT      ProtoMotions dataset root.
                           Default: \$MOTION_DATASETS/protomotions
  PROTO_AMASS_PT           Explicit packaged AMASS .pt input.
                           Default: smpl/<split>/amass_smpl_<split>.pt if it
                           exists, otherwise smpl/amass_smpl_<split>.pt
  PROTO_PYTHON             Direct ProtoMotions Python interpreter or venv dir.
                           If unset, uses python from PATH.
  PROTO_ISAACLAB_LAUNCHER  Optional IsaacLab launcher path. If set and
                           PROTO_PYTHON is unset, it is used as
                           "\$PROTO_ISAACLAB_LAUNCHER -p".
  PYROKI_PYTHON            PyRoki Python interpreter or venv dir.
                           Default: /workspace/pyroki-venv/bin/python if it
                           exists, otherwise python from PATH
  PROTO_PYROKI_REPO        PyRoki checkout root.
                           Default: /workspace/pyroki
  PROTO_INPUT_FPS          FPS used by PyRoki velocity costs and converter input.
                           Default: 30
  PROTO_OUTPUT_FPS         FPS written by converter output.
                           Default: \$PROTO_INPUT_FPS
  PROTO_CHUNK_THRESHOLD_FRAMES
                           Long-motion chunking threshold. Default: 900
  PROTO_CHUNK_SIZE_FRAMES  Long-motion chunk window size. Default: 450
  PROTO_CHUNK_OVERLAP_FRAMES
                           Consecutive chunk overlap. Default: 60

Examples:
  $0
  $0 sfu astro 1
  $0 train astro 1
EOF
}

is_supported_robot() {
    case "$1" in
        g1|h1_2|astro) return 0 ;;
        *) return 1 ;;
    esac
}

validate_split_name() {
    case "$1" in
        ''|*/*|*'..'*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

validate_positive_int() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        0)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

resolve_amass_pt() {
    if [ -n "${PROTO_AMASS_PT:-}" ]; then
        echo "$PROTO_AMASS_PT"
        return 0
    fi

    local split_path="${DATA}/smpl/${SPLIT}/amass_smpl_${SPLIT}.pt"
    local flat_path="${DATA}/smpl/amass_smpl_${SPLIT}.pt"

    if [ -f "$split_path" ]; then
        echo "$split_path"
    else
        echo "$flat_path"
    fi
}

resolve_python_command() {
    local python_path="$1"

    if [ -d "$python_path" ]; then
        python_path="${python_path}/bin/python"
    fi

    if [ -f "$python_path" ]; then
        echo "$python_path"
        return 0
    fi

    if command -v "$python_path" >/dev/null 2>&1; then
        echo "$python_path"
        return 0
    fi

    fail "Python interpreter not found: $1"
}

resolve_pyroki_python() {
    if [ -n "${PYROKI_PYTHON:-}" ]; then
        resolve_python_command "$PYROKI_PYTHON"
        return 0
    fi

    if [ -f "/workspace/pyroki-venv/bin/python" ]; then
        echo "/workspace/pyroki-venv/bin/python"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi

    fail "PyRoki Python not found. Set PYROKI_PYTHON."
}

resolve_pyroki_batch_cli() {
    local external_cli="${PYROKI_REPO}/batch_retarget_from_keypoints.py"
    local repo_local_cli="${REPO_ROOT}/pyroki/batch_retarget_from_keypoints.py"

    if [ -n "${PROTO_PYROKI_BATCH_CLI:-}" ]; then
        [ -f "$PROTO_PYROKI_BATCH_CLI" ] || fail "PyRoki batch retarget CLI not found: $PROTO_PYROKI_BATCH_CLI"
        echo "$PROTO_PYROKI_BATCH_CLI"
        return 0
    fi

    if [ -f "$external_cli" ]; then
        echo "$external_cli"
        return 0
    fi

    if [ -f "$repo_local_cli" ]; then
        echo "$repo_local_cli"
        return 0
    fi

    fail "PyRoki batch retarget CLI not found: $external_cli or $repo_local_cli"
}

delete_generated_files() {
    local directory="$1"
    local name_pattern="$2"

    if [ -d "$directory" ]; then
        find "$directory" -maxdepth 1 -type f -name "$name_pattern" -delete
    fi
}

run_proto_python() {
    if [ -n "$PROTO_ISAACLAB_LAUNCHER_RESOLVED" ]; then
        "$PROTO_ISAACLAB_LAUNCHER_RESOLVED" -p "$@"
    else
        "$PROTO_PYTHON_RESOLVED" "$@"
    fi
}

run_pyroki_python() {
    PYTHONPATH="${PYROKI_REPO}/src" "$PYROKI_PYTHON_RESOLVED" "$@"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -gt 3 ]; then
    usage >&2
    exit 1
fi

SPLIT="${1:-${SPLIT:-test}}"
ROBOT="${2:-${ROBOT:-astro}}"
SKIP_FREQ="${3:-${SKIP_FREQ:-1}}"
PROTO_INPUT_FPS="${PROTO_INPUT_FPS:-30}"
PROTO_OUTPUT_FPS="${PROTO_OUTPUT_FPS:-$PROTO_INPUT_FPS}"
PROTO_CHUNK_THRESHOLD_FRAMES="${PROTO_CHUNK_THRESHOLD_FRAMES:-900}"
PROTO_CHUNK_SIZE_FRAMES="${PROTO_CHUNK_SIZE_FRAMES:-450}"
PROTO_CHUNK_OVERLAP_FRAMES="${PROTO_CHUNK_OVERLAP_FRAMES:-60}"

validate_split_name "$SPLIT" || fail "split must be a simple folder name such as test, train, validation, sfu, or cmu"
is_supported_robot "$ROBOT" || fail "robot must be $SUPPORTED_ROBOTS_DISPLAY"
validate_positive_int "$SKIP_FREQ" || fail "skip_freq must be a positive integer"
validate_positive_int "$PROTO_CHUNK_THRESHOLD_FRAMES" || fail "PROTO_CHUNK_THRESHOLD_FRAMES must be a positive integer"
validate_positive_int "$PROTO_CHUNK_SIZE_FRAMES" || fail "PROTO_CHUNK_SIZE_FRAMES must be a positive integer"
case "$PROTO_CHUNK_OVERLAP_FRAMES" in
    ''|*[!0-9]*)
        fail "PROTO_CHUNK_OVERLAP_FRAMES must be a non-negative integer"
        ;;
esac

MOTION_DATASETS="${MOTION_DATASETS:-/data/share/motion_datasets}"
DATA="${PROTO_RETARGET_ROOT:-${DATA:-${MOTION_DATASETS}/protomotions}}"
PYROKI_REPO="${PROTO_PYROKI_REPO:-${PYROKI_REPO:-/workspace/pyroki}}"
PROTO_ISAACLAB_LAUNCHER_RESOLVED=""

if [ -n "${PROTO_PYTHON:-}" ]; then
    PROTO_PYTHON_RESOLVED="$(resolve_python_command "$PROTO_PYTHON")"
elif [ -n "${PROTO_ISAACLAB_LAUNCHER:-}" ]; then
    [ -f "$PROTO_ISAACLAB_LAUNCHER" ] || fail "IsaacLab launcher not found: $PROTO_ISAACLAB_LAUNCHER"
    PROTO_PYTHON_RESOLVED=""
    PROTO_ISAACLAB_LAUNCHER_RESOLVED="$PROTO_ISAACLAB_LAUNCHER"
elif command -v python >/dev/null 2>&1; then
    PROTO_PYTHON_RESOLVED="python"
else
    fail "ProtoMotions Python not found. Set PROTO_PYTHON."
fi

PYROKI_PYTHON_RESOLVED="$(resolve_pyroki_python)"
PYROKI_BATCH_CLI="$(resolve_pyroki_batch_cli)"

AMASS_PT="$(resolve_amass_pt)"
SMPL_SPLIT_DIR="${DATA}/smpl/${SPLIT}"
ROBOT_SPLIT_DIR="${DATA}/${ROBOT}/${SPLIT}"
KEYPOINTS_DIR="${SMPL_SPLIT_DIR}/keypoints-for-retarget"
CONTACTS_DIR="${SMPL_SPLIT_DIR}/contacts"
RETARGETED_DIR="${ROBOT_SPLIT_DIR}/pyroki-retargeted-${ROBOT}"
PROTO_DIR="${ROBOT_SPLIT_DIR}/proto-${ROBOT}"
FINAL_PT="${ROBOT_SPLIT_DIR}/proto-${ROBOT}.pt"

[ -f "$AMASS_PT" ] || fail "AMASS .pt file not found: $AMASS_PT"
[ -d "${PYROKI_REPO}/src/pyroki" ] || fail "PyRoki repo not found or invalid: $PYROKI_REPO"

cd "$REPO_ROOT"
mkdir -p "$KEYPOINTS_DIR" "$CONTACTS_DIR" "$RETARGETED_DIR" "$PROTO_DIR"

export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/protomotions-cache-${SPLIT}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-mpl-cache-${SPLIT}}"

if [ -n "$PROTO_ISAACLAB_LAUNCHER_RESOLVED" ]; then
    PROTO_PYTHON_DISPLAY="$PROTO_ISAACLAB_LAUNCHER_RESOLVED -p"
else
    PROTO_PYTHON_DISPLAY="$PROTO_PYTHON_RESOLVED"
fi

echo "=============================================="
echo "HPC AMASS retargeting"
echo "=============================================="
echo "Split:               $SPLIT"
echo "Robot:               $ROBOT"
echo "Skip freq:           $SKIP_FREQ"
echo "Input FPS:           $PROTO_INPUT_FPS"
echo "Output FPS:          $PROTO_OUTPUT_FPS"
echo "Chunk threshold:     $PROTO_CHUNK_THRESHOLD_FRAMES"
echo "Chunk size:          $PROTO_CHUNK_SIZE_FRAMES"
echo "Chunk overlap:       $PROTO_CHUNK_OVERLAP_FRAMES"
echo "Repo root:           $REPO_ROOT"
echo "Data root:           $DATA"
echo "AMASS input:         $AMASS_PT"
echo "Proto Python:        $PROTO_PYTHON_DISPLAY"
echo "PyRoki Python:       $PYROKI_PYTHON_RESOLVED"
echo "PyRoki repo:         $PYROKI_REPO"
echo "PyRoki batch CLI:    $PYROKI_BATCH_CLI"
echo "Output MotionLib:    $FINAL_PT"
echo "=============================================="

echo ""
echo "[Step 1/5] Extracting SMPL keypoints..."
delete_generated_files "$KEYPOINTS_DIR" "*_keypoints.npy"
run_proto_python data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    "$AMASS_PT" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq "$SKIP_FREQ" \
    --force-remake

echo ""
echo "[Step 2/5] Retargeting keypoints to ${ROBOT}..."
run_pyroki_python "$PYROKI_BATCH_CLI" \
    --robot-type "$ROBOT" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --input-fps "$PROTO_INPUT_FPS" \
    --chunk-long-motions \
    --chunk-threshold-frames "$PROTO_CHUNK_THRESHOLD_FRAMES" \
    --chunk-size-frames "$PROTO_CHUNK_SIZE_FRAMES" \
    --chunk-overlap-frames "$PROTO_CHUNK_OVERLAP_FRAMES" \
    --output-dir "$RETARGETED_DIR" \
    --no-visualize \
    --skip-existing

echo ""
echo "[Step 3/5] Extracting source foot-contact labels..."
delete_generated_files "$CONTACTS_DIR" "*.npy"
run_pyroki_python "$PYROKI_BATCH_CLI" \
    --robot-type "$ROBOT" \
    --subsample-factor 1 \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type smpl \
    --input-fps "$PROTO_INPUT_FPS" \
    --save-contacts-only \
    --contacts-dir "$CONTACTS_DIR" \
    --skip-existing

echo ""
echo "[Step 4/5] Converting PyRoki output to ProtoMotions format..."
run_proto_python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir "$RETARGETED_DIR" \
    --output-dir "$PROTO_DIR" \
    --robot-type "$ROBOT" \
    --input-fps "$PROTO_INPUT_FPS" \
    --output-fps "$PROTO_OUTPUT_FPS" \
    --contact-labels-dir "$CONTACTS_DIR" \
    --apply-motion-filter \
    --force-remake

echo ""
echo "[Step 5/5] Packaging ProtoMotions MotionLib..."
run_proto_python protomotions/components/motion_lib.py \
    --motion-path "$PROTO_DIR" \
    --output-file "$FINAL_PT"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "Output MotionLib: $FINAL_PT"
echo "=============================================="
