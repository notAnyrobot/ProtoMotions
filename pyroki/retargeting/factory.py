# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from retargeting.config import PyrokiRetargetConfig
from retargeting.configs import G1_RETARGET_CONFIG, H1_2_RETARGET_CONFIG

_CONFIGS: dict[str, PyrokiRetargetConfig] = {
    G1_RETARGET_CONFIG.robot_type: G1_RETARGET_CONFIG,
    H1_2_RETARGET_CONFIG.robot_type: H1_2_RETARGET_CONFIG,
}


def supported_robot_types() -> tuple[str, ...]:
    return tuple(_CONFIGS)


def get_retarget_config(robot_type: str) -> PyrokiRetargetConfig:
    try:
        return _CONFIGS[robot_type]
    except KeyError as exc:
        supported = ", ".join(supported_robot_types())
        raise ValueError(
            f"Unsupported robot type {robot_type!r}. Supported: {supported}"
        ) from exc
