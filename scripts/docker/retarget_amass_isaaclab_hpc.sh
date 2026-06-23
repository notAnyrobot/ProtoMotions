#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Compatibility wrapper for the old IsaacLab-named HPC retargeting entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL_SCRIPT="${SCRIPT_DIR}/retarget_amass_hpc.sh"

if [ ! -f "$CANONICAL_SCRIPT" ]; then
    echo "Error: canonical retarget script not found: $CANONICAL_SCRIPT" >&2
    exit 1
fi

echo "Warning: scripts/docker/retarget_amass_isaaclab_hpc.sh is deprecated; use scripts/docker/retarget_amass_hpc.sh instead." >&2
exec "$CANONICAL_SCRIPT" "$@"
