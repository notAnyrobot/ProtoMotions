from pathlib import Path
import sys

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

import pytest

from retargeting.config import N_RETARGET
from retargeting.factory import get_retarget_config, supported_robot_types


COMMON_LOCAL_ALIGNMENT_PAIRS = (
    ("left_shoulder", "left_elbow", 1.0),
    ("right_shoulder", "right_elbow", 1.0),
    ("left_elbow", "left_wrist", 1.0),
    ("right_elbow", "right_wrist", 1.0),
    ("left_hip", "left_knee", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("right_ankle", "right_foot", 1.0),
)

G1_LINK_MAPPING = (
    ("pelvis", "pelvis_contour_link"),
    ("left_hip", "left_hip_pitch_link"),
    ("right_hip", "right_hip_pitch_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_ankle", "left_ankle_roll_link"),
    ("right_ankle", "right_ankle_roll_link"),
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
    ("left_shoulder", "left_shoulder_pitch_link"),
    ("right_shoulder", "right_shoulder_pitch_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_wrist", "left_wrist_yaw_link"),
    ("right_wrist", "right_wrist_yaw_link"),
)

H1_2_LINK_MAPPING = (
    ("pelvis", "pelvis"),
    ("left_hip", "left_hip_yaw_link"),
    ("right_hip", "right_hip_yaw_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_ankle", "left_ankle_roll_link"),
    ("right_ankle", "right_ankle_roll_link"),
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
    ("left_shoulder", "left_shoulder_roll_link"),
    ("right_shoulder", "right_shoulder_roll_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_wrist", "left_wrist_yaw_link"),
    ("right_wrist", "right_wrist_yaw_link"),
)

COMMON_WEIGHTS = {
    "local_alignment": 1.0,
    "root_smoothness": 1.0,
    "joint_smoothness": 4.0,
    "self_collision": 0.0,
    "joint_rest_penalty": 1.0,
    "joint_vel_limit": 50.0,
    "foot_contact": 30.0,
    "foot_tilt": 1.0,
}


def _link_mapping_as_tuples(config):
    return tuple((item.source_keypoint, item.robot_link) for item in config.link_mapping)


def _alignment_pairs_as_tuples(config):
    return tuple(
        (item.source_a, item.source_b, item.weight)
        for item in config.local_alignment_pairs
    )


def _assert_shared_invariants(config):
    assert config.supported_source_types == tuple(config.source_scales)
    assert len(config.link_mapping) == N_RETARGET
    assert _alignment_pairs_as_tuples(config) == COMMON_LOCAL_ALIGNMENT_PAIRS


def _assert_mapping_update_does_not_leak(config, mapping_name, key, new_value):
    mapping = getattr(config, mapping_name)
    original_value = mapping[key]

    try:
        mapping[key] = new_value
    except TypeError:
        return

    try:
        assert getattr(get_retarget_config(config.robot_type), mapping_name)[
            key
        ] == original_value
    finally:
        mapping[key] = original_value


def test_supported_robot_types_are_stable():
    assert supported_robot_types() == ("g1", "h1_2")


def test_factory_configs_cannot_be_mutated_globally():
    config = get_retarget_config("g1")

    _assert_mapping_update_does_not_leak(
        config,
        "source_scales",
        "smpl",
        config.source_scales["rigv1"],
    )
    _assert_mapping_update_does_not_leak(
        config,
        "global_alignment_keypoint_weights",
        "left_elbow",
        99.0,
    )


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
    assert config.weights.as_dict() == COMMON_WEIGHTS | {"global_alignment": 4.0}
    assert config.hand_aux_offset == (0.0, 0.0, 0.14)
    assert config.torso_link_name == "torso_link"
    assert config.torso_aux_offset == (0.15, 0.0, -0.1)
    assert config.rest_penalty_joint_names == (
        "waist_roll_joint",
        "right_wrist_pitch_joint",
        "left_wrist_pitch_joint",
    )
    assert _link_mapping_as_tuples(config) == G1_LINK_MAPPING
    assert dict(config.global_alignment_keypoint_weights) == {
        "left_elbow": 0.25,
        "right_elbow": 0.25,
        "left_hand_aux": 0.25,
        "right_hand_aux": 0.25,
    }
    assert config.global_alignment_keypoint_weights["left_elbow"] == 0.25
    assert "left_hip" not in config.global_alignment_keypoint_weights
    _assert_shared_invariants(config)


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
    assert config.weights.as_dict() == COMMON_WEIGHTS | {"global_alignment": 3.0}
    assert config.hand_aux_offset == (0.0, 0.0, 0.2)
    assert config.torso_link_name == "torso_link"
    assert config.torso_aux_offset == (0.3, 0.0, 0.0)
    assert config.rest_penalty_joint_names == ()
    assert _link_mapping_as_tuples(config) == H1_2_LINK_MAPPING
    assert dict(config.global_alignment_keypoint_weights) == {
        "left_hip": 0.25,
        "right_hip": 0.25,
        "left_elbow": 0.25,
        "right_elbow": 0.25,
        "left_hand_aux": 0.25,
        "right_hand_aux": 0.25,
    }
    assert config.global_alignment_keypoint_weights["left_hip"] == 0.25
    assert config.global_alignment_keypoint_weights["right_hip"] == 0.25
    _assert_shared_invariants(config)


def test_unknown_robot_type_lists_supported_values():
    with pytest.raises(
        ValueError,
        match="Unsupported robot type 'atlas'. Supported: g1, h1_2",
    ):
        get_retarget_config("atlas")
