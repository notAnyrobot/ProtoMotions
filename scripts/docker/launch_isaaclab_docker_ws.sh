#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience launcher for the local Isaac Lab Docker image on a workstation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROTO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
IMAGE="${PROTO_ISAACLAB_IMAGE:-protomotions-isaaclab:2.3.0}"
DATASET_ROOT="${PROTO_DATASET_ROOT:-/media/android/data/motion_lib}"
CACHE_DIR="${PROTO_ISAACLAB_CACHE:-$HOME/.cache/protomotions-isaaclab}"
CONTAINER_REPO="${PROTO_CONTAINER_REPO:-/workspace/protomotions}"
GPU_MODE="${PROTO_GPU_MODE:-gpus}"
DATASET_READONLY="${PROTO_DATASET_READONLY:-1}"
FIX_OWNERSHIP="${PROTO_FIX_OWNERSHIP:-1}"
CHOWN_PATHS="${PROTO_CHOWN_PATHS:-results output wandb}"
HOST_UID="${PROTO_HOST_UID:-$(id -u)}"
HOST_GID="${PROTO_HOST_GID:-$(id -g)}"

usage() {
    cat <<EOF
Usage: $0 [command] [args...]

Commands:
  shell               Launch an interactive shell. This is the default.
  nvidia-smi          Run nvidia-smi inside the image and exit.
  smoke               Check Isaac Lab Python, Torch CUDA, Lightning, TensorDict.
  python [args...]    Run Isaac Lab Python with the given arguments and exit.
  run [args...]       Run an arbitrary command in /workspace/protomotions and exit.
  bash <command>      Run a shell command in /workspace/protomotions and exit.
  print-config        Show resolved image, repo, dataset, and cache paths.

Environment overrides:
  PROTO_ISAACLAB_IMAGE       Docker image tag. Default: $IMAGE
  PROTO_REPO                 Host repo path. Default: $REPO
  PROTO_DATASET_ROOT         Host dataset root mounted into the same path in the container.
                             Default: $DATASET_ROOT
  PROTO_ISAACLAB_CACHE       Host cache dir mounted to /root/.cache.
                             Default: $CACHE_DIR
  PROTO_GPU_MODE             gpus, cdi, legacy, or none. Default: $GPU_MODE
  PROTO_DATASET_READONLY     1 for read-only dataset mount, 0 for writable. Default: $DATASET_READONLY
  PROTO_FIX_OWNERSHIP        1 to chown artifact dirs after container exit, 0 to disable.
                             Default: $FIX_OWNERSHIP
  PROTO_CHOWN_PATHS          Space-separated host paths to chown after container exit.
                             Relative paths are resolved under PROTO_REPO.
                             Default: $CHOWN_PATHS
  PROTO_HOST_UID             Host UID used for chown. Default: $HOST_UID
  PROTO_HOST_GID             Host GID used for chown. Default: $HOST_GID

Examples:
  $0
  $0 shell
  $0 nvidia-smi
  $0 smoke
  $0 python -c "import torch; print(torch.cuda.get_device_name(0))"
  $0 run /workspace/isaaclab/isaaclab.sh -p -u protomotions/train_agent.py --help
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

build_gpu_args() {
    DOCKER_GPU_ARGS=()
    case "$GPU_MODE" in
        gpus)
            DOCKER_GPU_ARGS=(--gpus all)
            ;;
        cdi)
            DOCKER_GPU_ARGS=(--device nvidia.com/gpu=all)
            ;;
        legacy)
            DOCKER_GPU_ARGS=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
            ;;
        none)
            ;;
        *)
            fail "PROTO_GPU_MODE must be one of: gpus, cdi, legacy, none"
            ;;
    esac
}

validate_common_paths() {
    [ -d "$REPO" ] || fail "Repo path does not exist: $REPO"
    [ -d "$DATASET_ROOT" ] || fail "Dataset root does not exist: $DATASET_ROOT"
    mkdir -p "$CACHE_DIR"
}

fix_ownership() {
    if [ "$FIX_OWNERSHIP" = "0" ]; then
        return 0
    fi

    local paths=()
    local path
    local full_path

    for path in $CHOWN_PATHS; do
        case "$path" in
            /*) full_path="$path" ;;
            *) full_path="$REPO/$path" ;;
        esac

        if [ -e "$full_path" ]; then
            paths+=("$full_path")
        fi
    done

    if [ "${#paths[@]}" -eq 0 ]; then
        return 0
    fi

    echo "Fixing host ownership for artifact paths: ${paths[*]}"
    if chown -R "${HOST_UID}:${HOST_GID}" "${paths[@]}" 2>/dev/null; then
        return 0
    fi

    if command -v sudo >/dev/null 2>&1; then
        if sudo chown -R "${HOST_UID}:${HOST_GID}" "${paths[@]}"; then
            return 0
        fi
    fi

    echo "WARNING: Could not fix ownership for artifact paths: ${paths[*]}" >&2
    echo "Run manually: sudo chown -R ${HOST_UID}:${HOST_GID} ${paths[*]}" >&2
}

build_common_args() {
    DATASET_MOUNT="type=bind,src=$DATASET_ROOT,dst=$DATASET_ROOT"
    if [ "$DATASET_READONLY" != "0" ]; then
        DATASET_MOUNT="${DATASET_MOUNT},readonly"
    fi

    DOCKER_COMMON_ARGS=(
        "${DOCKER_GPU_ARGS[@]}"
        --network=host
        --ipc=host
        --shm-size=16g
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e OMNI_KIT_ACCEPT_EULA=YES
        -e ACCEPT_EULA=Y
        --mount "type=bind,src=$REPO,dst=$CONTAINER_REPO"
        --mount "$DATASET_MOUNT"
        --mount "type=bind,src=$CACHE_DIR,dst=/root/.cache"
        -w "$CONTAINER_REPO"
    )
}

run_nvidia_smi() {
    docker run --rm "${DOCKER_GPU_ARGS[@]}" --entrypoint nvidia-smi "$IMAGE"
}

run_isaac_python() {
    local status=0
    docker run --rm \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /workspace/isaaclab/isaaclab.sh \
        "$IMAGE" \
        -p "$@" || status=$?
    fix_ownership
    return "$status"
}

run_shell() {
    local status=0
    docker run --rm -it \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" || status=$?
    fix_ownership
    return "$status"
}

run_bash_command() {
    [ $# -gt 0 ] || fail "bash command is required"
    local status=0
    docker run --rm \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -lc "$*" || status=$?
    fix_ownership
    return "$status"
}

run_command() {
    [ $# -gt 0 ] || fail "command is required"
    local status=0
    docker run --rm \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint "$1" \
        "$IMAGE" \
        "${@:2}" || status=$?
    fix_ownership
    return "$status"
}

print_config() {
    cat <<EOF
Repo:          $REPO
Image:         $IMAGE
Dataset root:  $DATASET_ROOT
Dataset mode:  $([ "$DATASET_READONLY" = "0" ] && echo writable || echo readonly)
Cache dir:     $CACHE_DIR
GPU mode:      $GPU_MODE
Fix ownership: $FIX_OWNERSHIP
Chown paths:   $CHOWN_PATHS
Host UID:GID:  ${HOST_UID}:${HOST_GID}

Container repo path:
  $CONTAINER_REPO

Dataset symlinks under the mounted repo resolve because the host dataset root is
mounted into the same absolute path inside the container:
  $DATASET_ROOT -> $DATASET_ROOT
EOF
}

COMMAND="${1:-shell}"
if [ $# -gt 0 ]; then
    shift
fi

build_gpu_args

case "$COMMAND" in
    help|-h|--help)
        usage
        ;;
    nvidia-smi)
        run_nvidia_smi
        ;;
    print-config)
        print_config
        ;;
    smoke)
        validate_common_paths
        build_common_args
        run_isaac_python -c "import sys, torch, lightning, tensordict; print('python', sys.executable); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('lightning', lightning.__version__); print('tensordict', tensordict.__version__)"
        ;;
    shell)
        validate_common_paths
        build_common_args
        run_shell
        ;;
    python)
        validate_common_paths
        build_common_args
        run_isaac_python "$@"
        ;;
    run)
        validate_common_paths
        build_common_args
        run_command "$@"
        ;;
    bash)
        validate_common_paths
        build_common_args
        run_bash_command "$@"
        ;;
    *)
        usage >&2
        fail "Unknown command: $COMMAND"
        ;;
esac
