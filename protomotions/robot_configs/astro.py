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
#
import math
from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.isaacgym.config import (
    IsaacGymPhysXParams,
    IsaacGymSimParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimParams,
)
from protomotions.simulator.newton.config import NewtonSimParams

ARMATURE_8514_25 = 81.431e-3
ARMATURE_5016_25 = 8.811e-3
ARMATURE_3907_36 = 2.387e-3

DEFAULT_NATURAL_FREQUENCY_HZ = 10.0
DAMPING_RATIO = 2.0

T_POSE_JOINT_POS = {
    "left_shoulder_roll_joint": 1.5,
    "right_shoulder_roll_joint": -1.5,
    ".*_elbow_joint": 1.5,
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
}


def _compute_stiffness(
    armature: float,
    multiplier: float = 1.0,
    natural_frequency_hz: float = DEFAULT_NATURAL_FREQUENCY_HZ,
) -> float:
    omega = math.tau * natural_frequency_hz
    return armature * multiplier * omega**2


def _compute_damping(
    armature: float,
    multiplier: float = 1.0,
    natural_frequency_hz: float = DEFAULT_NATURAL_FREQUENCY_HZ,
    damping_ratio: float = DAMPING_RATIO,
) -> float:
    omega = math.tau * natural_frequency_hz
    return 2.0 * damping_ratio * armature * multiplier * omega


@dataclass
class AstroRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_wrist_yaw_link"],
            "all_right_hand_bodies": ["right_wrist_yaw_link"],
            "head_body_name": ["head_link"],
            "torso_body_name": ["torso_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "head_link",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )

    default_root_height: float = 0.75
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: T_POSE_JOINT_POS)
    anchor_body_name: str = "torso_link"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/astro_v0.1.xml",
            usd_asset_file_name="usd/astro_v1/astro_v1_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "waist_yaw_joint": ControlInfo(
                    stiffness=_compute_stiffness(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    damping=_compute_damping(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                ".*_hip_(pitch|roll|yaw)_joint": ControlInfo(
                    stiffness=_compute_stiffness(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    damping=_compute_damping(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                ".*_knee_joint": ControlInfo(
                    stiffness=_compute_stiffness(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    damping=_compute_damping(
                        ARMATURE_8514_25, natural_frequency_hz=6.0
                    ),
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                "waist_pitch_joint": ControlInfo(
                    stiffness=_compute_stiffness(ARMATURE_5016_25, multiplier=2.0),
                    damping=_compute_damping(ARMATURE_5016_25, multiplier=2.0),
                    effort_limit=60.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25 * 2.0,
                ),
                "waist_roll_joint": ControlInfo(
                    stiffness=_compute_stiffness(ARMATURE_5016_25, multiplier=1.66),
                    damping=_compute_damping(ARMATURE_5016_25, multiplier=1.66),
                    effort_limit=50.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25 * 1.66,
                ),
                ".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint": ControlInfo(
                    stiffness=_compute_stiffness(ARMATURE_5016_25),
                    damping=_compute_damping(ARMATURE_5016_25),
                    effort_limit=30.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25,
                ),
                ".*_ankle_pitch_joint": ControlInfo(
                    stiffness=_compute_stiffness(ARMATURE_5016_25, multiplier=2.0),
                    damping=_compute_damping(ARMATURE_5016_25, multiplier=2.0),
                    effort_limit=60.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25 * 2.0,
                ),
                ".*_ankle_roll_joint": ControlInfo(
                    stiffness=_compute_stiffness(ARMATURE_5016_25, multiplier=1.5),
                    damping=_compute_damping(ARMATURE_5016_25, multiplier=1.5),
                    effort_limit=45.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25 * 1.5,
                ),
                "head_yaw_joint": ControlInfo(
                    stiffness=_compute_stiffness(
                        ARMATURE_3907_36, natural_frequency_hz=12.0
                    ),
                    damping=_compute_damping(
                        ARMATURE_3907_36, natural_frequency_hz=12.0
                    ),
                    effort_limit=10.0,
                    velocity_limit=20.94,
                    armature=ARMATURE_3907_36,
                ),
                ".*_wrist_(pitch|yaw)_joint": ControlInfo(
                    stiffness=_compute_stiffness(
                        ARMATURE_3907_36, natural_frequency_hz=12.0
                    ),
                    damping=_compute_damping(
                        ARMATURE_3907_36, natural_frequency_hz=12.0
                    ),
                    effort_limit=10.0,
                    velocity_limit=20.94,
                    armature=ARMATURE_3907_36,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=100,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
