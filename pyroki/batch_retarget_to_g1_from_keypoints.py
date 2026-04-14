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

import runpy
import sys
from pathlib import Path


_ROBOT_CONFIG = Path(__file__).resolve().parent / "robot_configs" / "g1.yaml"


def _warn_and_delegate() -> None:
    sys.stderr.write(
        "WARNING: This script is deprecated. Use 'python pyroki/batch_retarget.py --robot-config pyroki/robot_configs/g1.yaml' instead.\n"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.argv = [sys.argv[0], "--robot-config", str(_ROBOT_CONFIG), *sys.argv[1:]]
    runpy.run_path(
        str(Path(__file__).resolve().parent / "batch_retarget.py"), run_name="__main__"
    )


if __name__ == "__main__":
    _warn_and_delegate()
