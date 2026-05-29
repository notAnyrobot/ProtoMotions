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


ARMATURE_8514_25 = 0.08143145775
ARMATURE_5016_25 = 0.008810606409825
ARMATURE_3907_36 = 0.002387232


def _stiffness(
    armature: float,
    natural_frequency_hz: float,
    armature_multiplier: float = 1.0,
) -> float:
    omega = math.tau * natural_frequency_hz
    return armature * armature_multiplier * omega**2


def _damping(
    armature: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    armature_multiplier: float = 1.0,
) -> float:
    omega = math.tau * natural_frequency_hz
    return 2.0 * damping_ratio * armature * armature_multiplier * omega


STIFFNESS_8514_25 = _stiffness(ARMATURE_8514_25, natural_frequency_hz=7.0)
DAMPING_8514_25 = _damping(
    ARMATURE_8514_25, natural_frequency_hz=7.0, damping_ratio=2.5
)
STIFFNESS_5016_25 = _stiffness(ARMATURE_5016_25, natural_frequency_hz=10.0)
DAMPING_5016_25 = _damping(
    ARMATURE_5016_25, natural_frequency_hz=10.0, damping_ratio=2.5
)
DAMPING_5016_25_WAIST = _damping(
    ARMATURE_5016_25, natural_frequency_hz=10.0, damping_ratio=3.0
)
STIFFNESS_3907_36 = _stiffness(ARMATURE_3907_36, natural_frequency_hz=10.0)
DAMPING_3907_36 = _damping(
    ARMATURE_3907_36, natural_frequency_hz=10.0, damping_ratio=3.0
)

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


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
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: str = "torso_link"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="astro/mjcf/astro_v1.xml",
            usd_asset_file_name="usd/astro_29dof/astro_29dof.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            self_collisions=True,
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
                ".*_hip_(pitch|roll|yaw)_joint": ControlInfo(
                    stiffness=STIFFNESS_8514_25,
                    damping=DAMPING_8514_25,
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                ".*_knee_joint": ControlInfo(
                    stiffness=STIFFNESS_8514_25,
                    damping=DAMPING_8514_25,
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                ".*_ankle_pitch_joint": ControlInfo(
                    stiffness=2.0 * STIFFNESS_5016_25,
                    damping=2.0 * DAMPING_5016_25,
                    effort_limit=60.0,
                    velocity_limit=26.18,
                    armature=2.0 * ARMATURE_5016_25,
                ),
                ".*_ankle_roll_joint": ControlInfo(
                    stiffness=1.5 * STIFFNESS_5016_25,
                    damping=1.5 * DAMPING_5016_25,
                    effort_limit=45.0,
                    velocity_limit=26.18,
                    armature=1.5 * ARMATURE_5016_25,
                ),
                "waist_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_8514_25,
                    damping=DAMPING_8514_25,
                    effort_limit=130.0,
                    velocity_limit=18.85,
                    armature=ARMATURE_8514_25,
                ),
                "waist_pitch_joint": ControlInfo(
                    stiffness=2.0 * STIFFNESS_5016_25,
                    damping=2.0 * DAMPING_5016_25_WAIST,
                    effort_limit=60.0,
                    velocity_limit=26.18,
                    armature=2.0 * ARMATURE_5016_25,
                ),
                "waist_roll_joint": ControlInfo(
                    stiffness=1.66 * STIFFNESS_5016_25,
                    damping=1.66 * DAMPING_5016_25_WAIST,
                    effort_limit=50.0,
                    velocity_limit=26.18,
                    armature=1.66 * ARMATURE_5016_25,
                ),
                ".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint": ControlInfo(
                    stiffness=STIFFNESS_5016_25,
                    damping=DAMPING_5016_25,
                    effort_limit=30.0,
                    velocity_limit=26.18,
                    armature=ARMATURE_5016_25,
                ),
                ".*_wrist_(pitch|yaw)_joint": ControlInfo(
                    stiffness=STIFFNESS_3907_36,
                    damping=DAMPING_3907_36,
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
