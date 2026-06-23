#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Download one retargeted robot motion split from the HPC shared dataset tree to
# the workstation dataset tree.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-atom7@192.168.24.9}"
REMOTE_PROTOMOTIONS_ROOT="${REMOTE_PROTOMOTIONS_ROOT:-/data/share/motion_datasets/protomotions}"
LOCAL_PROTOMOTIONS_ROOT="${LOCAL_PROTOMOTIONS_ROOT:-/media/android/data/motion_datasets/protomotions}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

usage() {
    cat <<EOF
Usage: $0 --split SPLIT [--robot ROBOT] [--dry-run]

Download retargeted motions and packaging artifacts for one robot split from
HPC to the workstation.

Required:
  --split SPLIT     Split folder to pull, such as sfu, test, train, or validation.

Options:
  --robot ROBOT     Robot folder to pull. Default: astro.
  -n, --dry-run     Print rsync actions without copying.
  -h, --help        Show this message.

Environment overrides:
  REMOTE_HOST                Default: ${REMOTE_HOST}
  REMOTE_PROTOMOTIONS_ROOT   Default: ${REMOTE_PROTOMOTIONS_ROOT}
  LOCAL_PROTOMOTIONS_ROOT    Default: ${LOCAL_PROTOMOTIONS_ROOT}
  RSYNC_BIN                  Default: rsync

Examples:
  $0 --split sfu
  $0 --robot astro --split sfu
  REMOTE_HOST=hpc-1 $0 --split validation --dry-run
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        fail "${cmd} is required but not installed"
    fi
}

validate_simple_name() {
    case "$1" in
        ''|*/*|*'..'*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

ROBOT="${ROBOT:-astro}"
SPLIT=""
DRY_RUN=0

while (($#)); do
    case "$1" in
        --robot)
            (($# >= 2)) || fail "--robot requires a value"
            ROBOT="$2"
            shift 2
            ;;
        --robot=*)
            ROBOT="${1#--robot=}"
            shift
            ;;
        --split)
            (($# >= 2)) || fail "--split requires a value"
            SPLIT="$2"
            shift 2
            ;;
        --split=*)
            SPLIT="${1#--split=}"
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

[ -n "$SPLIT" ] || fail "--split is required. Example: $0 --split sfu"
validate_simple_name "$ROBOT" || fail "--robot must be a simple folder name"
validate_simple_name "$SPLIT" || fail "--split must be a simple folder name"

REMOTE_PROTOMOTIONS_ROOT="${REMOTE_PROTOMOTIONS_ROOT%/}"
LOCAL_PROTOMOTIONS_ROOT="${LOCAL_PROTOMOTIONS_ROOT%/}"
REMOTE_SOURCE="${REMOTE_HOST}:${REMOTE_PROTOMOTIONS_ROOT}/${ROBOT}/${SPLIT}/"
LOCAL_DESTINATION="${LOCAL_PROTOMOTIONS_ROOT}/${ROBOT}/${SPLIT}/"

require_cmd "$RSYNC_BIN"
mkdir -p "$LOCAL_DESTINATION"

RSYNC_ARGS=(-az --partial --info=progress2)
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_ARGS+=(--dry-run --stats --itemize-changes)
fi

echo "=============================================="
echo "Download retargeted ProtoMotions split"
echo "=============================================="
echo "Robot:               $ROBOT"
echo "Split:               $SPLIT"
echo "Remote source:       $REMOTE_SOURCE"
echo "Local destination:   $LOCAL_DESTINATION"
echo "Dry run:             $DRY_RUN"
echo "=============================================="

"$RSYNC_BIN" "${RSYNC_ARGS[@]}" "$REMOTE_SOURCE" "$LOCAL_DESTINATION"

echo "==> Download complete."
