# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


N_RETARGET = 15
N_AUX = 3
N_KEYPOINTS_WITH_AUX = N_RETARGET + N_AUX

RobotCostHook = Callable[[Any], list[Any]]


@dataclass(frozen=True)
class SourceScale:
    lower_body: tuple[float, float, float]
    upper_body: tuple[float, float, float]


@dataclass(frozen=True)
class LinkMapping:
    source_keypoint: str
    robot_link: str


@dataclass(frozen=True)
class AlignmentPair:
    source_a: str
    source_b: str
    weight: float


@dataclass(frozen=True)
class RetargetingWeights:
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float

    def as_dict(self) -> dict[str, float]:
        return {
            "local_alignment": self.local_alignment,
            "global_alignment": self.global_alignment,
            "root_smoothness": self.root_smoothness,
            "joint_smoothness": self.joint_smoothness,
            "self_collision": self.self_collision,
            "joint_rest_penalty": self.joint_rest_penalty,
            "joint_vel_limit": self.joint_vel_limit,
            "foot_contact": self.foot_contact,
            "foot_tilt": self.foot_tilt,
        }


@dataclass(frozen=True)
class PyrokiRetargetConfig:
    robot_type: str
    display_name: str
    supported_source_types: tuple[str, ...]
    urdf_path: Path
    mesh_dir: Path
    link_mapping: tuple[LinkMapping, ...]
    source_scales: dict[str, SourceScale]
    weights: RetargetingWeights
    hand_aux_offset: tuple[float, float, float]
    torso_link_name: str
    torso_aux_offset: tuple[float, float, float]
    global_alignment_keypoint_weights: dict[str, float] = field(default_factory=dict)
    rest_penalty_joint_names: tuple[str, ...] = ()
    local_alignment_pairs: tuple[AlignmentPair, ...] = ()
    max_joint_velocity: float = 20.0
    max_iterations: int = 800
    robot_specific_cost_hooks: tuple[RobotCostHook, ...] = ()
