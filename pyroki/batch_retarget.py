# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified batch retargeting CLI wrapper.

Accepts ``--robot-config <path>`` or ``--robot <name>`` and forwards all other args to
``retarget_from_keypoints.main()``.

Robot configs are loaded from the robot_configs/ directory when using --robot.
"""

import argparse
import sys
from pathlib import Path

from retarget_from_keypoints import main


def _resolve_robot_config(robot_name: str) -> Path:
    """Resolve robot name to config path in robot_configs/ directory."""
    configs_dir = Path(__file__).resolve().parent / "robot_configs"
    config_path = configs_dir / f"{robot_name}.yaml"
    if not config_path.exists():
        available = [p.stem for p in configs_dir.glob("*.yaml")]
        raise SystemExit(
            f"Error: Robot config '{robot_name}.yaml' not found in {configs_dir}\n"
            f"Available robots: {', '.join(sorted(available))}"
        )
    return config_path


def _parse_robot_config() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot-config", type=str, default=None)
    parser.add_argument("--robot", type=str, default=None)
    known, remaining = parser.parse_known_args(sys.argv[1:])

    # Always pass through the wrapper's remaining argv so the downstream
    # argparse in retarget_from_keypoints.main() owns the rest of the CLI.
    sys.argv = [sys.argv[0]] + remaining

    # Handle --robot (convenience) or --robot-config (explicit path)
    if known.robot is not None:
        return str(_resolve_robot_config(known.robot).resolve())
    if known.robot_config is not None:
        return str(Path(known.robot_config).resolve())

    return None


def _has_help_flag() -> bool:
    return any(flag in sys.argv[1:] for flag in ("-h", "--help"))


if __name__ == "__main__":
    resolved_config = _parse_robot_config()
    if _has_help_flag():
        main(default_robot_config=None)
    else:
        main(default_robot_config=resolved_config)
