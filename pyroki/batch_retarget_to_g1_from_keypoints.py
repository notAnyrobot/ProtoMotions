# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Deprecated G1 wrapper for the canonical PyRoki retargeting CLI.

Run this wrapper with the same flags as the canonical CLI; it injects
``--robot-type g1`` before delegation.
"""

from pathlib import Path
import sys
import warnings

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retargeting.cli import main as canonical_main


def main() -> int:
    warnings.warn(
        "pyroki/batch_retarget_to_g1_from_keypoints.py is deprecated; use "
        "pyroki/batch_retarget_from_keypoints.py --robot-type g1 instead.",
        FutureWarning,
        stacklevel=2,
    )
    return canonical_main([*sys.argv[1:], "--robot-type", "g1"])


if __name__ == "__main__":
    raise SystemExit(main())
