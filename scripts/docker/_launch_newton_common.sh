#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Shared shell-only implementation for ProtoMotions Newton Docker launchers.
# Host-specific wrapper scripts set defaults before sourcing this file.

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is required but was not found in PATH"
}

usage() {
    cat <<EOF
Usage: $0 [shell|help]

Commands:
  shell    Open an interactive container shell. This is the default.
  help     Show this message.

Default image:       $IMAGE
Default dataset:     ${DATASET_ROOT:-not mounted}
Default GPU mode:    $GPU_MODE
Default GPU choice:  $GPU_SELECTION

Common overrides:
  PROTO_NEWTON_IMAGE          Container image tag.
  PROTO_REPO                  Host ProtoMotions checkout.
  PROTO_PYROKI_REPO           Host PyRoki checkout.
  PROTO_DATASET_ROOT          Dataset root mounted at the same absolute path.
  PROTO_DATASET_READONLY      1 for read-only dataset mount, 0 for writable.
  PROTO_NEWTON_CACHE          Host cache directory mounted to /root/.cache.
  PROTO_FIX_OWNERSHIP         1 to repair host artifact ownership after exit.
  PROTO_CHOWN_PATHS           Space-separated artifact paths under PROTO_REPO.
  PROTO_HOST_UID              Host UID used for ownership repair.
  PROTO_HOST_GID              Host GID used for ownership repair.
  $GPU_MODE_ENV_LABEL         gpus, cdi, legacy, manual, or none.
  $GPU_SELECTION_ENV_LABEL    all, none, or comma-separated GPU IDs.
  PROTO_GPUS                  Compatibility alias for GPU selection.

Workflow guide:
  scripts/docker/README.md
EOF
}

validate_common_paths() {
    [ -d "$REPO" ] || fail "Repo path does not exist: $REPO"
    [ -f "$REPO/protomotions/train_agent.py" ] || fail "PROTO_REPO does not look like the ProtoMotions repo root: $REPO"
    [ -d "$PYROKI_REPO" ] || fail "PyRoki repo path does not exist: $PYROKI_REPO"
    [ -d "$PYROKI_REPO/src/pyroki" ] || fail "PROTO_PYROKI_REPO does not look like the PyRoki repo root: $PYROKI_REPO"

    if [ -n "${DATASET_ROOT:-}" ]; then
        [ -d "$DATASET_ROOT" ] || fail "Dataset root does not exist: $DATASET_ROOT"
    fi

    mkdir -p "$CACHE_DIR"
}

validate_gpu_selection() {
    case "$GPU_SELECTION" in
        none|all)
            ;;
        *)
            local selection="${GPU_SELECTION// /}"
            case "$selection" in
                ''|*[!0-9,]*)
                    fail "$GPU_SELECTION_ENV_LABEL must be all, none, or comma-separated GPU IDs"
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
            [ "${#GPU_IDS[@]}" -gt 0 ] || fail "No GPU devices found under $DEV_DIR for $GPU_SELECTION_ENV_LABEL=all"
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
                    -e LD_LIBRARY_PATH=/workspace/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/host-nvidia-libs
                )
            fi
            ;;
        *)
            fail "$GPU_MODE_ENV_LABEL must be one of: gpus, cdi, legacy, manual, none"
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

build_shell_args() {
    DOCKER_SHELL_ARGS=(
        "${DOCKER_GPU_ARGS[@]}"
        "${DOCKER_RUNTIME_ARGS[@]}"
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e MUJOCO_GL=osmesa
        -e "PYTHONPATH=$CONTAINER_PYROKI_REPO/src:$CONTAINER_REPO:$CONTAINER_REPO/protomotions"
        -e PYTHONUNBUFFERED=1
        -e JAX_PLATFORMS=cuda,cpu
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false
        --mount "type=bind,src=$REPO,dst=$CONTAINER_REPO"
        --mount "type=bind,src=$PYROKI_REPO,dst=$CONTAINER_PYROKI_REPO"
        --mount "type=bind,src=$CACHE_DIR,dst=/root/.cache"
        -w "$CONTAINER_REPO"
    )

    if [ -n "${DATASET_ROOT:-}" ]; then
        local dataset_mount="type=bind,src=$DATASET_ROOT,dst=$DATASET_ROOT"
        if [ "$DATASET_READONLY" != "0" ]; then
            dataset_mount="${dataset_mount},readonly"
        fi
        DOCKER_SHELL_ARGS+=(--mount "$dataset_mount")
    fi
}

prepare_shell_args() {
    require_cmd docker
    validate_common_paths
    build_gpu_args
    build_runtime_args
    build_shell_args
}

fix_ownership() {
    if [ "$FIX_OWNERSHIP" = "0" ]; then
        return 0
    fi

    local paths=()
    local container_paths=()
    local path
    local full_path

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

run_shell() {
    local status=0
    docker run -it \
        "${DOCKER_SHELL_ARGS[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" || status=$?
    fix_ownership
    return "$status"
}

COMMAND="${1:-shell}"
if [ $# -gt 0 ]; then
    shift
fi

case "$COMMAND" in
    help|-h|--help)
        usage
        ;;
    shell)
        prepare_shell_args
        run_shell
        ;;
    *)
        usage >&2
        fail "Unknown command: $COMMAND"
        ;;
esac
