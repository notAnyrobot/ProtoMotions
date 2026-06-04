#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-atom7@192.168.24.143}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/atom7/Code/ProtoMotions}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
SSH_BIN="${SSH_BIN:-ssh}"

# Always run from the repository root so rsync paths are stable.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [COMMAND] [OPTIONS]

Commands:
  push                         Sync codebase to remote HPC  [default]
  pull-artifacts [PATH|EXP]    Pull training artifacts to local results/
  pull-results [PATH|EXP]      Alias for pull-artifacts
  help                         Show this message

Options:
  -n, --dry-run                Print rsync actions without copying
      --delete                 Delete files on destination that no longer exist
  -h, --help                   Show this message

Environment overrides:
  REMOTE_HOST                  Default: ${REMOTE_HOST}
  REMOTE_ROOT                  Default: ${REMOTE_ROOT}
  RSYNC_BIN                    Default: rsync
  SSH_BIN                      Default: ssh

Examples:
  $(basename "$0") push
  $(basename "$0") push --dry-run
  $(basename "$0") pull-artifacts
  $(basename "$0") pull-artifacts astro-motion-tracker-amass-test
  REMOTE_ROOT=/scratch/atom7/ProtoMotions $(basename "$0") push
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "${cmd} is required but not installed" >&2
    exit 1
  fi
}

remote_quote() {
  printf "%q" "$1"
}

parse_common_flags() {
  DRY_RUN=0
  DELETE=0
  POSITIONAL=()

  while (($#)); do
    case "$1" in
      -n|--dry-run)
        DRY_RUN=1
        shift
        ;;
      --delete)
        DELETE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        POSITIONAL+=("$@")
        break
        ;;
      -*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
  done
}

rsync_common_args() {
  RSYNC_ARGS=(-az --partial --info=progress2)
  if [[ "${DRY_RUN}" == "1" ]]; then
    RSYNC_ARGS+=(--dry-run)
  fi
  if [[ "${DELETE}" == "1" ]]; then
    RSYNC_ARGS+=(--delete)
  fi
}

cmd_push() {
  parse_common_flags "$@"
  if ((${#POSITIONAL[@]} != 0)); then
    echo "push does not accept positional arguments" >&2
    usage >&2
    exit 1
  fi

  require_cmd "$RSYNC_BIN"
  require_cmd "$SSH_BIN"
  rsync_common_args

  echo "==> Creating remote directory ${REMOTE_HOST}:${REMOTE_ROOT}"
  "$SSH_BIN" "$REMOTE_HOST" "mkdir -p $(remote_quote "$REMOTE_ROOT")"

  echo "==> Syncing codebase to ${REMOTE_HOST}:${REMOTE_ROOT}/"
  "$RSYNC_BIN" "${RSYNC_ARGS[@]}" \
    --filter=':- .gitignore' \
    --exclude='.git/' \
    --exclude='.agents/' \
    --exclude='.codex/' \
    --exclude='.pytest_cache/' \
    --exclude='.venv*/' \
    --exclude='*.tar' \
    --exclude='*.tar.*' \
    --exclude='*.sif' \
    --exclude='*.simg' \
    --exclude='*.sqsh' \
    "$ROOT_DIR/" "${REMOTE_HOST}:${REMOTE_ROOT}/"
  echo "==> Push complete."
}

validate_relative_path() {
  local path="$1"
  if [[ -z "$path" || "$path" == /* ]]; then
    echo "Artifact path must be relative to the remote repo root" >&2
    exit 1
  fi

  local old_ifs="$IFS"
  local part
  IFS='/'
  for part in $path; do
    if [[ "$part" == "." || "$part" == ".." ]]; then
      IFS="$old_ifs"
      echo "Artifact path cannot contain '.' or '..' segments: $path" >&2
      exit 1
    fi
  done
  IFS="$old_ifs"
}

artifact_relative_path() {
  local selector="${1:-results}"
  selector="${selector#./}"
  validate_relative_path "$selector"

  case "$selector" in
    results|results/*|output|output/*|outputs|outputs/*|wandb|wandb/*|exps|exps/*)
      printf "%s\n" "$selector"
      ;;
    *)
      printf "results/%s\n" "$selector"
      ;;
  esac
}

cmd_pull_artifacts() {
  parse_common_flags "$@"
  if ((${#POSITIONAL[@]} > 1)); then
    echo "pull-artifacts accepts at most one PATH or experiment name" >&2
    usage >&2
    exit 1
  fi

  require_cmd "$RSYNC_BIN"
  rsync_common_args

  local rel_path
  rel_path="$(artifact_relative_path "${POSITIONAL[0]:-results}")"
  local local_path="${ROOT_DIR}/${rel_path}"
  local remote_path="${REMOTE_ROOT}/${rel_path}"

  mkdir -p "$local_path"
  echo "==> Pulling ${REMOTE_HOST}:${remote_path}/ to ${local_path}/"
  "$RSYNC_BIN" "${RSYNC_ARGS[@]}" \
    "${REMOTE_HOST}:${remote_path}/" "$local_path/"
  echo "==> Pull complete."
}

COMMAND="${1:-push}"
if (($#)); then
  shift
fi

case "$COMMAND" in
  push)
    cmd_push "$@"
    ;;
  pull-artifacts|pull-results)
    cmd_pull_artifacts "$@"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
