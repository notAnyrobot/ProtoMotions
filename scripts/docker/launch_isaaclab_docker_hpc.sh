#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience launcher for ProtoMotions Isaac Lab Docker on HPC.
#
# The validated HPC host uses rootless Docker without the normal NVIDIA runtime.
# GPU access is wired explicitly through /dev/nvidia* devices plus the host
# NVIDIA libraries staged under /data/$USER/nvidia-libs. Set
# PROTO_HPC_GPU_MODE=gpus on hosts where docker run --gpus works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROTO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
IMAGE="${PROTO_ISAACLAB_IMAGE:-protomotions-isaaclab:2.3.0}"
CONTAINER_REPO="${PROTO_CONTAINER_REPO:-/workspace/protomotions}"

HPC_USER="${PROTO_HPC_USER:-$(id -un)}"
DATA_ROOT="${PROTO_HPC_DATA_ROOT:-/data/$HPC_USER}"
CACHE_ROOT="${PROTO_ISAACLAB_CACHE:-$DATA_ROOT/isaac_cache}"

GPU_MODE="${PROTO_HPC_GPU_MODE:-${PROTO_GPU_MODE:-manual}}"
GPU_SELECTION="${PROTO_HPC_GPUS:-${PROTO_GPUS:-all}}"
CUDA_VISIBLE_OVERRIDE="${PROTO_CUDA_VISIBLE_DEVICES:-}"

NVIDIA_LIBS="${PROTO_HPC_NVIDIA_LIBS:-$DATA_ROOT/nvidia-libs}"
NVIDIA_SMI="${PROTO_HPC_NVIDIA_SMI:-/usr/bin/nvidia-smi}"
DEV_DIR="${PROTO_HPC_DEV_DIR:-/dev}"

SHM_SIZE="${PROTO_HPC_SHM_SIZE:-32g}"
IPC_MODE="${PROTO_HPC_IPC_MODE:-private}"
NETWORK_MODE="${PROTO_HPC_NETWORK_MODE:-host}"

DATASET_ROOT="${PROTO_DATASET_ROOT:-/data/share/motion_datasets/protomotions}"
DATASET_READONLY="${PROTO_DATASET_READONLY:-1}"

FIX_OWNERSHIP="${PROTO_FIX_OWNERSHIP:-0}"
CHOWN_PATHS="${PROTO_CHOWN_PATHS:-results output wandb}"
HOST_UID="${PROTO_HOST_UID:-$(id -u)}"
HOST_GID="${PROTO_HOST_GID:-$(id -g)}"

CONTAINER_NAME="${PROTO_CONTAINER_NAME:-}"
DOCKER_RM="${PROTO_DOCKER_RM:-1}"
RESET_TIMEOUT="${PROTO_RESET_TIMEOUT:-300}"

usage() {
    cat <<USAGE_EOF
Usage: $0 [command] [args...]

Commands:
  shell               Launch an interactive bash shell. Default.
  nvidia-smi          Run nvidia-smi inside the image and exit.
  smoke               Check nvidia-smi, libcuda, Isaac Lab Python, and Torch CUDA.
  reset               Run minimal IsaacLab SimulationContext.reset() test.
  python [args...]    Run Isaac Lab Python with the given args and exit.
  run [args...]       Run an arbitrary command in $CONTAINER_REPO and exit.
  bash <command>      Run a shell command in $CONTAINER_REPO and exit.
  train-debug         Run the small Astro IsaacLab debug training command.
  print-config        Show resolved image, repo, cache, and GPU settings.

Environment overrides:
  PROTO_ISAACLAB_IMAGE        Docker image tag. Default: $IMAGE
  PROTO_REPO                  Host repo path. Default: $REPO
  PROTO_CONTAINER_REPO        Container repo path. Default: $CONTAINER_REPO
  PROTO_HPC_DATA_ROOT         Large host data root. Default: $DATA_ROOT
  PROTO_ISAACLAB_CACHE        Isaac cache root. Default: $CACHE_ROOT

  PROTO_HPC_GPU_MODE          manual, gpus, cdi, legacy, or none. Default: $GPU_MODE
                              manual wires /dev/nvidia* plus staged host libraries.
  PROTO_HPC_GPUS              all, none, or comma-separated GPU IDs. Default: $GPU_SELECTION
                              Examples: all, 6, 0,1
  PROTO_GPUS                  Compatibility alias for PROTO_HPC_GPUS.
  PROTO_CUDA_VISIBLE_DEVICES  Optional CUDA_VISIBLE_DEVICES value inside container.
  PROTO_HPC_NVIDIA_LIBS       Manual-mode host NVIDIA library dir. Default: $NVIDIA_LIBS
  PROTO_HPC_NVIDIA_SMI        Manual-mode host nvidia-smi. Default: $NVIDIA_SMI
  PROTO_HPC_DEV_DIR           Manual-mode device dir. Default: $DEV_DIR
  PROTO_HPC_SHM_SIZE          Docker --shm-size value when IPC is private. Default: $SHM_SIZE
  PROTO_HPC_IPC_MODE          private or host. Default: $IPC_MODE
  PROTO_HPC_NETWORK_MODE      host, bridge, or none. Default: $NETWORK_MODE

  PROTO_DATASET_ROOT          Dataset root mounted to the same absolute path. Default: $DATASET_ROOT
  PROTO_DATASET_READONLY      1 for read-only dataset mount, 0 for writable. Default: $DATASET_READONLY

  PROTO_CONTAINER_NAME        Optional Docker container name.
                              Useful for second terminal: docker exec -it NAME /bin/bash
  PROTO_DOCKER_RM             1 to pass --rm, 0 to keep container after exit. Default: $DOCKER_RM

  PROTO_FIX_OWNERSHIP         1 to chown artifact dirs after container exit. Default: $FIX_OWNERSHIP
  PROTO_CHOWN_PATHS           Space-separated artifact paths. Default: $CHOWN_PATHS
  PROTO_HOST_UID              Host UID for ownership fix. Default: $HOST_UID
  PROTO_HOST_GID              Host GID for ownership fix. Default: $HOST_GID

  PROTO_RESET_TIMEOUT         Timeout for reset command in seconds. Default: $RESET_TIMEOUT

Examples:
  $0 print-config
  $0 nvidia-smi
  PROTO_HPC_GPUS=6 $0 shell
  PROTO_HPC_GPUS=4,5,6,7 $0 shell
  PROTO_HPC_GPU_MODE=gpus PROTO_HPC_GPUS=6 $0 shell
  PROTO_HPC_GPUS=6 PROTO_CONTAINER_NAME=proto_hpc2 $0 shell
  docker exec -it proto_hpc2 /bin/bash
  PROTO_HPC_GPUS=6 $0 reset
  PROTO_HPC_GPUS=6 $0 bash 'nvidia-smi && pwd && ls'
  PROTO_HPC_GPUS=6 $0 train-debug
USAGE_EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is required but was not found in PATH"
}

validate_common_paths() {
    [ -d "$REPO" ] || fail "Repo path does not exist: $REPO"
    [ -f "$REPO/protomotions/train_agent.py" ] || fail "PROTO_REPO does not look like the ProtoMotions repo root: $REPO"

    if [ -n "$DATASET_ROOT" ]; then
        [ -d "$DATASET_ROOT" ] || fail "Dataset root does not exist: $DATASET_ROOT"
    fi

    mkdir -p \
        "$CACHE_ROOT/kit" \
        "$CACHE_ROOT/ov" \
        "$CACHE_ROOT/pip" \
        "$CACHE_ROOT/glcache" \
        "$CACHE_ROOT/computecache" \
        "$CACHE_ROOT/logs" \
        "$CACHE_ROOT/data" \
        "$CACHE_ROOT/documents"
}

validate_gpu_selection() {
    case "$GPU_SELECTION" in
        none|all)
            ;;
        *)
            local selection="${GPU_SELECTION// /}"
            case "$selection" in
                ''|*[!0-9,]*)
                    fail "PROTO_HPC_GPUS must be all, none, or comma-separated GPU IDs"
                    ;;
            esac
            ;;
    esac
}

build_gpu_ids() {
    GPU_IDS=()

    case "$GPU_SELECTION" in
        none)
            return 0
            ;;
        all)
            local path
            local name
            local id
            while IFS= read -r id; do
                GPU_IDS+=("$id")
            done < <(
                for path in "$DEV_DIR"/nvidia[0-9]*; do
                    [ -e "$path" ] || continue
                    name="${path##*/}"
                    id="${name#nvidia}"
                    case "$id" in
                        ''|*[!0-9]*)
                            ;;
                        *)
                            printf "%s\n" "$id"
                            ;;
                    esac
                done | sort -n
            )
            [ "${#GPU_IDS[@]}" -gt 0 ] || fail "No GPU devices found under $DEV_DIR for PROTO_HPC_GPUS=all"
            ;;
        *)
            IFS=',' read -r -a GPU_IDS <<< "${GPU_SELECTION// /}"
            ;;
    esac
}

validate_gpu_file() {
    local path="$1"
    [ -e "$path" ] || fail "Required GPU device does not exist: $path"
}

validate_nvidia_libs() {
    local lib
    [ -d "$NVIDIA_LIBS" ] || fail "NVIDIA library dir does not exist: $NVIDIA_LIBS"
    for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
        [ -e "$NVIDIA_LIBS/$lib" ] || fail "Missing NVIDIA library: $NVIDIA_LIBS/$lib"
    done
    [ -e "$NVIDIA_SMI" ] || fail "nvidia-smi binary does not exist: $NVIDIA_SMI"
}

build_gpu_args() {
    DOCKER_GPU_ARGS=()
    validate_gpu_selection

    case "$GPU_MODE" in
        none)
            ;;
        gpus)
            case "$GPU_SELECTION" in
                none)
                    ;;
                all)
                    DOCKER_GPU_ARGS=(--gpus all)
                    ;;
                *)
                    local selection="${GPU_SELECTION// /}"
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
                    DOCKER_GPU_ARGS=(--device "nvidia.com/gpu=${GPU_SELECTION// /}")
                    ;;
            esac
            ;;
        legacy)
            if [ "$GPU_SELECTION" != "none" ]; then
                DOCKER_GPU_ARGS=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=$GPU_SELECTION")
            fi
            ;;
        manual)
            build_gpu_ids
            if [ "$GPU_SELECTION" != "none" ]; then
                validate_nvidia_libs
                local id
                for id in "${GPU_IDS[@]}"; do
                    validate_gpu_file "$DEV_DIR/nvidia$id"
                    DOCKER_GPU_ARGS+=(--device="$DEV_DIR/nvidia$id")
                done
                validate_gpu_file "$DEV_DIR/nvidiactl"
                validate_gpu_file "$DEV_DIR/nvidia-uvm"
                validate_gpu_file "$DEV_DIR/nvidia-uvm-tools"
                DOCKER_GPU_ARGS+=(
                    --device="$DEV_DIR/nvidiactl"
                    --device="$DEV_DIR/nvidia-uvm"
                    --device="$DEV_DIR/nvidia-uvm-tools"
                    -v "$NVIDIA_LIBS:/host-nvidia-libs:ro"
                    -v "$NVIDIA_SMI:/usr/bin/nvidia-smi:ro"
                    -e LD_LIBRARY_PATH=/host-nvidia-libs
                )
            fi
            ;;
        *)
            fail "PROTO_HPC_GPU_MODE must be one of: manual, gpus, cdi, legacy, none"
            ;;
    esac

    if [ -n "$CUDA_VISIBLE_OVERRIDE" ]; then
        DOCKER_GPU_ARGS+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_OVERRIDE")
    elif [ "$GPU_MODE" = "manual" ] && [ "$GPU_SELECTION" != "all" ] && [ "$GPU_SELECTION" != "none" ]; then
        DOCKER_GPU_ARGS+=(-e "CUDA_VISIBLE_DEVICES=$(container_cuda_ordinals)")
    fi
}

build_runtime_args() {
    DOCKER_RUNTIME_ARGS=()

    case "$DOCKER_RM" in
        1)
            DOCKER_RUNTIME_ARGS+=(--rm)
            ;;
        0)
            ;;
        *)
            fail "PROTO_DOCKER_RM must be 1 or 0"
            ;;
    esac

    if [ -n "$CONTAINER_NAME" ]; then
        DOCKER_RUNTIME_ARGS+=(--name "$CONTAINER_NAME")
    fi

    case "$NETWORK_MODE" in
        host|bridge|none)
            DOCKER_RUNTIME_ARGS+=(--network="$NETWORK_MODE")
            ;;
        *)
            fail "PROTO_HPC_NETWORK_MODE must be host, bridge, or none"
            ;;
    esac

    case "$IPC_MODE" in
        host)
            DOCKER_RUNTIME_ARGS+=(--ipc=host)
            ;;
        private)
            DOCKER_RUNTIME_ARGS+=(--shm-size="$SHM_SIZE")
            ;;
        *)
            fail "PROTO_HPC_IPC_MODE must be private or host"
            ;;
    esac
}

build_common_args() {
    DOCKER_COMMON_ARGS=(
        "${DOCKER_GPU_ARGS[@]}"
        "${DOCKER_RUNTIME_ARGS[@]}"
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e OMNI_KIT_ACCEPT_EULA=YES
        -e ACCEPT_EULA=Y
        -e PRIVACY_CONSENT=N
        -e OMNI_KIT_ALLOW_ROOT=1
        --mount "type=bind,src=$REPO,dst=$CONTAINER_REPO"
        -v "$CACHE_ROOT/kit:/root/.cache/ov/Kit"
        -v "$CACHE_ROOT/ov:/root/.cache/ov"
        -v "$CACHE_ROOT/pip:/root/.cache/pip"
        -v "$CACHE_ROOT/glcache:/root/.cache/nvidia/GLCache"
        -v "$CACHE_ROOT/computecache:/root/.nv/ComputeCache"
        -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs"
        -v "$CACHE_ROOT/data:/root/.local/share/ov/data"
        -v "$CACHE_ROOT/documents:/root/Documents"
        -w "$CONTAINER_REPO"
    )

    if [ -n "$DATASET_ROOT" ]; then
        local dataset_mount="type=bind,src=$DATASET_ROOT,dst=$DATASET_ROOT"
        if [ "$DATASET_READONLY" != "0" ]; then
            dataset_mount="${dataset_mount},readonly"
        fi
        DOCKER_COMMON_ARGS+=(--mount "$dataset_mount")
    fi
}

prepare_docker_args() {
    require_cmd docker
    validate_common_paths
    build_gpu_args
    build_runtime_args
    build_common_args
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
            echo "You may need: sudo chown -R ${HOST_UID}:${HOST_GID} ${paths[*]}" >&2
        }
}

docker_run() {
    docker run "$@"
}

run_nvidia_smi() {
    docker_run \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -lc 'nvidia-smi'
}

run_isaac_python() {
    [ $# -gt 0 ] || set -- -c "import sys; print(sys.executable)"
    local status=0
    docker_run \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /workspace/isaaclab/isaaclab.sh \
        "$IMAGE" \
        -p "$@" || status=$?
    fix_ownership
    return "$status"
}

run_shell() {
    local status=0
    docker_run -it \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" || status=$?
    fix_ownership
    return "$status"
}

run_bash_command() {
    [ $# -gt 0 ] || fail "bash command is required"
    local status=0
    docker_run \
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
    docker_run \
        "${DOCKER_COMMON_ARGS[@]}" \
        --entrypoint "$1" \
        "$IMAGE" \
        "${@:2}" || status=$?
    fix_ownership
    return "$status"
}

run_smoke() {
    run_bash_command '
set -euxo pipefail
echo "===== NVIDIA SMI ====="
nvidia-smi
echo "===== CUDA DRIVER LOAD ====="
/isaac-sim/python.sh -c '"'"'import ctypes; ctypes.CDLL("libcuda.so.1"); print("CUDA driver visible")'"'"'
echo "===== ISAACLAB PYTHON / TORCH ====="
/workspace/isaaclab/isaaclab.sh -p -c '"'"'import os, sys, torch; print("python", sys.executable); print("torch", torch.__version__); print("cuda_visible", os.environ.get("CUDA_VISIBLE_DEVICES")); print("cuda", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'"'"'
'
}

run_reset() {
    run_bash_command '
set -euxo pipefail

echo "===== CHECK PATHS ====="
pwd
ls -l /workspace/isaaclab/isaaclab.sh
ls -ld /workspace/isaaclab
nvidia-smi

cat > /tmp/proto_min_reset.py <<'"'"'PY'"'"'
from isaaclab.app import AppLauncher

print("BEFORE_APP_LAUNCH", flush=True)
app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app
print("AFTER_APP_LAUNCH", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

print("BEFORE_SIM_CONTEXT", flush=True)
sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0"))
print("BEFORE_RESET", flush=True)
sim.reset()
print("RESET_OK", flush=True)
simulation_app.close()
print("CLOSED_OK", flush=True)
PY

echo "===== RUN MIN RESET ====="
timeout '"$RESET_TIMEOUT"' /workspace/isaaclab/isaaclab.sh -p /tmp/proto_min_reset.py
echo "RESET_EXIT_CODE=$?"
'
}

run_train_debug() {
    run_bash_command '
set -euxo pipefail
/workspace/isaaclab/isaaclab.sh -p -u protomotions/train_agent.py \
  --robot-name astro \
  --simulator isaaclab \
  --num-envs 128 \
  --batch-size 256 \
  --motion-file data/motion_for_trackers/astro_amass-test.pt \
  --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --experiment-name astro-motion-tracker-hpc2-debug
'
}

container_cuda_ordinals() {
    local selected="${GPU_SELECTION// /}"
    local ids=()
    local visible=""
    local index

    IFS=',' read -r -a ids <<< "$selected"
    for index in "${!ids[@]}"; do
        if [ -n "$visible" ]; then
            visible="$visible,$index"
        else
            visible="$index"
        fi
    done
    echo "$visible"
}

resolved_cuda_visible() {
    if [ -n "$CUDA_VISIBLE_OVERRIDE" ]; then
        echo "$CUDA_VISIBLE_OVERRIDE"
    elif [ "$GPU_MODE" = "manual" ] && [ "$GPU_SELECTION" != "all" ] && [ "$GPU_SELECTION" != "none" ]; then
        container_cuda_ordinals
    else
        echo auto
    fi
}

print_config() {
    cat <<CONFIG_EOF
Repo:              $REPO
Image:             $IMAGE
Container repo:    $CONTAINER_REPO
Data root:         $DATA_ROOT
Cache root:        $CACHE_ROOT
GPU mode:          $GPU_MODE
GPU selection:     $GPU_SELECTION
CUDA visible:      $(resolved_cuda_visible)
NVIDIA libs:       $NVIDIA_LIBS
nvidia-smi:        $NVIDIA_SMI
Device dir:        $DEV_DIR
Network mode:      $NETWORK_MODE
IPC mode:          $IPC_MODE
Shared memory:     $([ "$IPC_MODE" = "private" ] && echo "$SHM_SIZE" || echo "host IPC")
Dataset root:      ${DATASET_ROOT:-not mounted}
Dataset mode:      $([ "$DATASET_READONLY" = "0" ] && echo writable || echo readonly)
Container name:    ${CONTAINER_NAME:-not set}
Docker --rm:       $DOCKER_RM
Fix ownership:     $FIX_OWNERSHIP
Chown paths:       $CHOWN_PATHS
Host UID:GID:      ${HOST_UID}:${HOST_GID}
Reset timeout:     $RESET_TIMEOUT
CONFIG_EOF
}

COMMAND="${1:-shell}"
if [ $# -gt 0 ]; then
    shift
fi

case "$COMMAND" in
    help|-h|--help)
        usage
        ;;
    print-config)
        print_config
        ;;
    nvidia-smi)
        prepare_docker_args
        run_nvidia_smi
        ;;
    smoke)
        prepare_docker_args
        run_smoke
        ;;
    reset)
        prepare_docker_args
        run_reset
        ;;
    shell)
        prepare_docker_args
        run_shell
        ;;
    python)
        prepare_docker_args
        run_isaac_python "$@"
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
        run_train_debug
        ;;
    *)
        usage >&2
        fail "Unknown command: $COMMAND"
        ;;
esac
