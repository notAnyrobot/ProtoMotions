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

from pathlib import Path
from xml.etree import ElementTree

import mujoco
import pytest

from protomotions.robot_configs.factory import robot_config


def test_astro_robot_config_loads_from_factory():
    config = robot_config("astro")

    assert config.asset.asset_file_name == "astro/mjcf/astro_v1.xml"
    assert config.kinematic_info.num_dofs == 29
    assert config.number_of_actions == 29
    assert config.kinematic_info.nq == 36
    assert config.kinematic_info.nv == 35
    assert config.anchor_body_name == "torso_link"
    assert config.anchor_body_index == config.kinematic_info.body_names.index(
        "torso_link"
    )
    assert config.trackable_bodies_subset == [
        "torso_link",
        "head_link",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]


def test_astro_robot_config_exposes_29dof_pose_and_control_contract():
    config = robot_config("astro")

    assert config.kinematic_info.dof_names == [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert len(config.default_dof_pos) == config.kinematic_info.num_dofs
    assert config.default_dof_pos[0].item() == pytest.approx(-0.312)
    assert config.default_dof_pos[3].item() == pytest.approx(0.669)
    assert config.default_dof_pos[15].item() == pytest.approx(0.2)
    assert config.default_dof_pos[23].item() == pytest.approx(-0.2)
    assert "head_yaw_joint" not in config.kinematic_info.dof_names
    assert set(config.control.control_info) == set(config.kinematic_info.dof_names)
    assert config.control.control_info["left_hip_pitch_joint"].velocity_limit == 18.85


def test_astro_mjcf_compiles_as_29dof_model():
    config = robot_config("astro")
    mjcf_path = Path(config.asset.asset_root) / config.asset.asset_file_name

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    assert ElementTree.parse(mjcf_path).getroot().attrib["model"] == "astro_29dof"
    assert model.nu == 29
    assert model.nv == 35


def test_astro_robot_config_points_isaaclab_to_generated_usd():
    config = robot_config("astro")

    assert config.asset.usd_asset_file_name == "usd/astro_29dof/astro_29dof.usda"
    assert config.asset.usd_bodies_root_prim_path == "/World/envs/env_.*/Robot/pelvis/"
    assert (Path(config.asset.asset_root) / config.asset.usd_asset_file_name).is_file()
