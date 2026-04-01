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
from protomotions.robot_configs.factory import robot_config


def test_astro_robot_config_loads() -> None:
    config = robot_config("astro")

    assert config.kinematic_info.num_dofs == 30
    assert config.kinematic_info.num_bodies == 31
    assert config.anchor_body_name == "torso_link"
    assert config.default_root_height == 0.75

    for body_name in [
        "pelvis",
        "torso_link",
        "head_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]:
        assert body_name in config.kinematic_info.body_names

    for joint_name in [
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        "head_yaw_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]:
        assert joint_name in config.kinematic_info.dof_names

    assert config.common_naming_to_robot_body_names["head_body_name"] == ["head_link"]
    assert config.common_naming_to_robot_body_names["torso_body_name"] == ["torso_link"]