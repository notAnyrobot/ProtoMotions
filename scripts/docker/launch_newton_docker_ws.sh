#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience launcher for the local Newton Docker image on a workstation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROTO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
IMAGE="${PROTO_NEWTON_IMAGE:-protomotions-newton:cuda12.4-newton1.0.0}"
DATASET_ROOT="${PROTO_DATASET_ROOT:-/media/android/data/motion_lib}"
CACHE_DIR="${PROTO_NEWTON_CACHE:-$HOME/.cache/protomotions-newton}"
CONTAINER_REPO="${PROTO_CONTAINER_REPO:-/workspace/protomotions}"
GPU_MODE="${PROTO_GPU_MODE:-gpus}"
GPU_SELECTION="${PROTO_GPUS:-all}"
DATASET_READONLY="${PROTO_DATASET_READONLY:-1}"
FIX_OWNERSHIP="${PROTO_FIX_OWNERSHIP:-1}"
CHOWN_PATHS="${PROTO_CHOWN_PATHS:-results output wandb}"
HOST_UID="${PROTO_HOST_UID:-$(id -u)}"
HOST_GID="${PROTO_HOST_GID:-$(id -g)}"

TRAIN_ROBOT="${PROTO_NEWTON_TRAIN_ROBOT:-g1}"
TRAIN_NUM_ENVS="${PROTO_NEWTON_TRAIN_NUM_ENVS:-128}"
TRAIN_BATCH_SIZE="${PROTO_NEWTON_TRAIN_BATCH_SIZE:-256}"
TRAIN_MOTION_FILE="${PROTO_NEWTON_TRAIN_MOTION_FILE:-data/motion_for_trackers/g1_random_subset_tiny.pt}"
TRAIN_EXPERIMENT_PATH="${PROTO_NEWTON_TRAIN_EXPERIMENT_PATH:-examples/experiments/mimic/mlp.py}"
TRAIN_EXPERIMENT_NAME="${PROTO_NEWTON_TRAIN_EXPERIMENT_NAME:-newton-docker-debug}"

usage() {
    cat <<EOF
Usage: $0 [command] [args...]

Commands:
  shell               Launch an interactive shell. This is the default.
  nvidia-smi          Run nvidia-smi inside the image and exit.
  smoke               Check Python, Newton, MuJoCo, Torch CUDA visibility.
  python [args...]    Run Python with the given arguments and exit.
  run [args...]       Run an arbitrary command in $CONTAINER_REPO and exit.
  bash <command>      Run a shell command in $CONTAINER_REPO and exit.
  train-debug [args]  Start a small Newton training run, appending any extra args.
  print-config        Show resolved image, repo, dataset, cache, and training defaults.

Environment overrides:
  PROTO_NEWTON_IMAGE             Docker image tag. Default: $IMAGE
  PROTO_REPO                     Host repo path. Default: $REPO
  PROTO_CONTAINER_REPO           Container repo path. Default: $CONTAINER_REPO
  PROTO_DATASET_ROOT             Host dataset root mounted into the same path in the container.
                                  Default: $DATASET_ROOT
  PROTO_NEWTON_CACHE             Host cache dir mounted to /root/.cache.
                                  Default: $CACHE_DIR
  PROTO_GPU_MODE                 gpus, cdi, legacy, or none. Default: $GPU_MODE
  PROTO_GPUS                     all, none, or comma-separated GPU IDs. Default: $GPU_SELECTION
  PROTO_DATASET_READONLY         1 for read-only dataset mount, 0 for writable. Default: $DATASET_READONLY
  PROTO_FIX_OWNERSHIP            1 to chown artifact dirs after container exit, 0 to disable.
                                  Default: $FIX_OWNERSHIP
  PROTO_CHOWN_PATHS              Space-separated host paths to chown after container exit.
                                  Relative paths are resolved under PROTO_REPO.
                                  Default: $CHOWN_PATHS
  PROTO_HOST_UID                 Host UID used for chown. Default: $HOST_UID
  PROTO_HOST_GID                 Host GID used for chown. Default: $HOST_GID

Training defaults for train-debug:
  PROTO_NEWTON_TRAIN_ROBOT            Default: $TRAIN_ROBOT
  PROTO_NEWTON_TRAIN_NUM_ENVS         Default: $TRAIN_NUM_ENVS
  PROTO_NEWTON_TRAIN_BATCH_SIZE       Default: $TRAIN_BATCH_SIZE
  PROTO_NEWTON_TRAIN_MOTION_FILE      Default: $TRAIN_MOTION_FILE
  PROTO_NEWTON_TRAIN_EXPERIMENT_PATH  Default: $TRAIN_EXPERIMENT_PATH
  PROTO_NEWTON_TRAIN_EXPERIMENT_NAME  Default: $TRAIN_EXPERIMENT_NAME

Examples:
  $0
  $0 shell
  $0 nvidia-smi
  $0 smoke
  $0 python -c "import newton, torch; print(torch.cuda.is_available())"
  PROTO_GPUS=4,5,6,7 $0 shell
  $0 train-debug
  $0 train-debug --training-max-steps 4096
  PROTO_NEWTON_TRAIN_NUM_ENVS=512 PROTO_NEWTON_TRAIN_BATCH_SIZE=1024 $0 train-debug
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

build_gpu_args() {
    DOCKER_GPU_ARGS=()
    local selection="${GPU_SELECTION// /}"

    case "$GPU_SELECTION" in
        none|all)
            ;;
        *)
            case "$selection" in
                ''|*[!0-9,]*)
                    fail "PROTO_GPUS must be all, none, or comma-separated GPU IDs"
                    ;;
            esac
            ;;
    esac

    case "$GPU_MODE" in
        gpus)
            case "$GPU_SELECTION" in
                none)
                    ;;
                all)
                    DOCKER_GPU_ARGS=(--gpus all)
                    ;;
                *)
                    if [[ "$selection" == *,* ]]; then
                        DOCKER_GPU_ARGS=(--gpus "\"device=$selection\"")
                    else
                        DOCKER_GPU_ARGS=(--gpus "device=$selection")
                    fi
                    ;;
            esac
            ;;
        cdi)
            case "$GPU_SELECTION" in
                none)
                    ;;
                all)
                    DOCKER_GPU_ARGS=(--device nvidia.com/gpu=all)
                    ;;
                *)
                    DOCKER_GPU_ARGS=(--device "nvidia.com/gpu=$selection")
                    ;;
            esac
            ;;
        legacy)
            if [ "$GPU_SELECTION" != "none" ]; then
                DOCKER_GPU_ARGS=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=$GPU_SELECTION")
            fi
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
    [ -f "$REPO/protomotions/train_agent.py" ] || fail "PROTO_REPO does not look like the ProtoMotions repo root: $REPO"
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
    local container_paths=()

    for path in $CHOWN_PATHS; do
        case "$path" in
            /*) full_path="$path" ;;
            *) full_path="$REPO/$path" ;;
        esac

        if [ -e "$full_path" ]; then
            paths+=("$full_path")
            case "$full_path" in
                "$REPO"/*)
                    container_paths+=("$CONTAINER_REPO/${full_path#"$REPO"/}")
                    ;;
                "$REPO")
                    container_paths+=("$CONTAINER_REPO")
                    ;;
                *)
                    ;;
            esac
        fi
    done

    if [ "${#container_paths[@]}" -eq 0 ]; then
        return 0
    fi

    echo "Fixing host ownership for artifact paths: ${paths[*]}"
    docker run --rm \
        --mount "type=bind,src=$REPO,dst=$CONTAINER_REPO" \
        --entrypoint /bin/chown \
        "$IMAGE" \
        -R "${HOST_UID}:${HOST_GID}" "${container_paths[@]}" >/dev/null 2>&1 || {
            echo "WARNING: Could not fix ownership for artifact paths: ${paths[*]}" >&2
            echo "Run manually: sudo chown -R ${HOST_UID}:${HOST_GID} ${paths[*]}" >&2
        }
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
        -e MUJOCO_GL=osmesa
        -e "PYTHONPATH=$CONTAINER_REPO:$CONTAINER_REPO/protomotions"
        -e PYTHONUNBUFFERED=1
        --mount "type=bind,src=$REPO,dst=$CONTAINER_REPO"
        --mount "$DATASET_MOUNT"
        --mount "type=bind,src=$CACHE_DIR,dst=/root/.cache"
        -w "$CONTAINER_REPO"
    )
}

prepare_docker_args() {
    validate_common_paths
    build_common_args
}

run_nvidia_smi() {
    docker run --rm "${DOCKER_GPU_ARGS[@]}" --entrypoint nvidia-smi "$IMAGE"
}

run_python() {
    local status=0
    docker run --rm \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint python \
        "$IMAGE" \
        "$@" || status=$?
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

run_train_debug() {
    run_python -u -m protomotions.train_agent \
        --robot-name "$TRAIN_ROBOT" \
        --simulator newton \
        --num-envs "$TRAIN_NUM_ENVS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --motion-file "$TRAIN_MOTION_FILE" \
        --experiment-path "$TRAIN_EXPERIMENT_PATH" \
        --experiment-name "$TRAIN_EXPERIMENT_NAME" \
        "$@"
}

print_config() {
    cat <<EOF
Repo:             $REPO
Image:            $IMAGE
Dataset root:     $DATASET_ROOT
Dataset mode:     $([ "$DATASET_READONLY" = "0" ] && echo writable || echo readonly)
Cache dir:        $CACHE_DIR
GPU mode:         $GPU_MODE
GPU selection:    $GPU_SELECTION
Fix ownership:    $FIX_OWNERSHIP
Chown paths:      $CHOWN_PATHS
Host UID:GID:     ${HOST_UID}:${HOST_GID}

Container repo path:
  $CONTAINER_REPO

train-debug defaults:
  Robot:           $TRAIN_ROBOT
  Simulator:       newton
  Num envs:        $TRAIN_NUM_ENVS
  Batch size:      $TRAIN_BATCH_SIZE
  Motion file:     $TRAIN_MOTION_FILE
  Experiment path: $TRAIN_EXPERIMENT_PATH
  Experiment name: $TRAIN_EXPERIMENT_NAME

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
        prepare_docker_args
        run_python -c "import sys, torch, newton, mujoco, mujoco_warp; print('python', sys.executable); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('newton', getattr(newton, '__version__', 'unknown')); print('mujoco', mujoco.__version__); print('mujoco_warp', getattr(mujoco_warp, '__version__', 'unknown'))"
        ;;
    shell)
        prepare_docker_args
        run_shell
        ;;
    python)
        prepare_docker_args
        run_python "$@"
        ;;
    run)
        prepare_docker_args
        run_command "$@"
        ;;
    bash)
        prepare_docker_args
        run_bash_command "$@"
        ;;
    train-debug)
        prepare_docker_args
        run_train_debug "$@"
        ;;
    *)
        usage >&2
        fail "Unknown command: $COMMAND"
        ;;
esac
