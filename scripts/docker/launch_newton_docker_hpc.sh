#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience launcher for ProtoMotions Newton + PyRoki Docker on HPC-1.
#
# HPC-1 uses rootless Docker without the normal NVIDIA runtime. GPU access is
# wired explicitly through /dev/nvidia* devices plus host NVIDIA libraries staged
# under /data/$USER/nvidia-libs. Set PROTO_HPC_GPU_MODE=gpus on hosts where
# docker run --gpus works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROTO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
IMAGE="${PROTO_NEWTON_IMAGE:-protomotions-newton-pyroki:cuda12.4-newton1.0.0-pyroki59aa21f}"
CODE_ROOT="${PROTO_CODE_ROOT:-$(cd "$REPO/.." && pwd)}"
PYROKI_REPO="${PROTO_PYROKI_REPO:-$CODE_ROOT/pyroki}"
CONTAINER_REPO="${PROTO_CONTAINER_REPO:-/workspace/protomotions}"
CONTAINER_PYROKI_REPO="${PROTO_CONTAINER_PYROKI_REPO:-/workspace/pyroki}"

HPC_USER="${PROTO_HPC_USER:-$(id -un)}"
DATA_ROOT="${PROTO_HPC_DATA_ROOT:-/data/$HPC_USER}"
CACHE_DIR="${PROTO_NEWTON_CACHE:-$DATA_ROOT/newton_cache}"

NEWTON_LAUNCHER_PROFILE="hpc"
GPU_MODE_ENV_LABEL="PROTO_HPC_GPU_MODE"
GPU_SELECTION_ENV_LABEL="PROTO_HPC_GPUS"
GPU_MODE="${PROTO_HPC_GPU_MODE:-${PROTO_GPU_MODE:-manual}}"
GPU_SELECTION="${PROTO_HPC_GPUS:-${PROTO_GPUS:-all}}"
CUDA_VISIBLE_OVERRIDE="${PROTO_CUDA_VISIBLE_DEVICES:-}"

NVIDIA_LIBS="${PROTO_HPC_NVIDIA_LIBS:-$DATA_ROOT/nvidia-libs}"
NVIDIA_SMI="${PROTO_HPC_NVIDIA_SMI:-/usr/bin/nvidia-smi}"
DEV_DIR="${PROTO_HPC_DEV_DIR:-/dev}"

NETWORK_MODE="${PROTO_HPC_NETWORK_MODE:-host}"
IPC_MODE="${PROTO_HPC_IPC_MODE:-private}"
SHM_SIZE="${PROTO_HPC_SHM_SIZE:-32g}"
CONTAINER_NAME="${PROTO_CONTAINER_NAME:-}"
DOCKER_RM="${PROTO_DOCKER_RM:-1}"

DATASET_ROOT="${PROTO_DATASET_ROOT:-/data/share/motion_datasets}"
DATASET_READONLY="${PROTO_DATASET_READONLY:-1}"
FIX_OWNERSHIP="${PROTO_FIX_OWNERSHIP:-0}"
CHOWN_PATHS="${PROTO_CHOWN_PATHS:-results output wandb}"
HOST_UID="${PROTO_HOST_UID:-$(id -u)}"
HOST_GID="${PROTO_HOST_GID:-$(id -g)}"

# shellcheck source=scripts/docker/_launch_newton_common.sh
source "$SCRIPT_DIR/_launch_newton_common.sh"
