from pathlib import Path
import sys

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

import pytest

from retargeting.factory import get_retarget_config, supported_robot_types


def test_supported_robot_types_are_stable():
    assert supported_robot_types() == ("g1", "h1_2")


def test_g1_config_copies_current_script_constants():
    config = get_retarget_config("g1")

    assert config.robot_type == "g1"
    assert config.display_name == "G1"
    assert config.supported_source_types == ("smpl", "rigv1")
    assert config.urdf_path.as_posix().endswith(
        "protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    )
    assert config.mesh_dir.as_posix().endswith("protomotions/data/assets/mesh/G1")
    assert config.source_scales["smpl"].lower_body == (0.9, 0.9, 0.85)
    assert config.source_scales["smpl"].upper_body == (0.9, 0.9, 0.8)
    assert config.source_scales["rigv1"].lower_body == (0.8, 0.8, 0.75)
    assert config.source_scales["rigv1"].upper_body == (0.8, 0.8, 0.7)
    assert config.weights.global_alignment == 4.0
    assert config.hand_aux_offset == (0.0, 0.0, 0.14)
    assert config.torso_aux_offset == (0.15, 0.0, -0.1)
    assert config.rest_penalty_joint_names == (
        "waist_roll_joint",
        "right_wrist_pitch_joint",
        "left_wrist_pitch_joint",
    )
    assert config.global_alignment_keypoint_weights["left_elbow"] == 0.25
    assert "left_hip" not in config.global_alignment_keypoint_weights


def test_h1_2_config_copies_current_script_constants():
    config = get_retarget_config("h1_2")

    assert config.robot_type == "h1_2"
    assert config.display_name == "H1_2"
    assert config.urdf_path.as_posix().endswith(
        "protomotions/data/assets/urdf/for_retargeting/h1_2.urdf"
    )
    assert config.mesh_dir.as_posix().endswith("protomotions/data/assets/mesh/H1_2")
    assert config.source_scales["smpl"].lower_body == (1.1, 1.1, 1.1)
    assert config.source_scales["smpl"].upper_body == (1.1, 1.1, 1.0)
    assert config.source_scales["rigv1"].lower_body == (1.0, 1.0, 1.0)
    assert config.source_scales["rigv1"].upper_body == (1.0, 1.0, 0.9)
    assert config.weights.global_alignment == 3.0
    assert config.hand_aux_offset == (0.0, 0.0, 0.2)
    assert config.torso_aux_offset == (0.3, 0.0, 0.0)
    assert config.rest_penalty_joint_names == ()
    assert config.global_alignment_keypoint_weights["left_hip"] == 0.25
    assert config.global_alignment_keypoint_weights["right_hip"] == 0.25


def test_unknown_robot_type_lists_supported_values():
    with pytest.raises(
        ValueError,
        match="Unsupported robot type 'atlas'. Supported: g1, h1_2",
    ):
        get_retarget_config("atlas")
