# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from retargeting.config import (
    AlignmentPair,
    LinkMapping,
    PyrokiRetargetConfig,
    RetargetingWeights,
    SourceScale,
    freeze_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

LOCAL_ALIGNMENT_PAIRS = (
    AlignmentPair("left_shoulder", "left_elbow", 1.0),
    AlignmentPair("right_shoulder", "right_elbow", 1.0),
    AlignmentPair("left_elbow", "left_wrist", 1.0),
    AlignmentPair("right_elbow", "right_wrist", 1.0),
    AlignmentPair("left_hip", "left_knee", 1.0),
    AlignmentPair("right_hip", "right_knee", 1.0),
    AlignmentPair("left_knee", "left_ankle", 1.0),
    AlignmentPair("right_knee", "right_ankle", 1.0),
    AlignmentPair("left_ankle", "left_foot", 1.0),
    AlignmentPair("right_ankle", "right_foot", 1.0),
)

H1_2_RETARGET_CONFIG = PyrokiRetargetConfig(
    robot_type="h1_2",
    display_name="H1_2",
    supported_source_types=("smpl", "rigv1"),
    urdf_path=REPO_ROOT / "protomotions/data/assets/urdf/for_retargeting/h1_2.urdf",
    mesh_dir=REPO_ROOT / "protomotions/data/assets/mesh/H1_2",
    link_mapping=(
        LinkMapping("pelvis", "pelvis"),
        LinkMapping("left_hip", "left_hip_yaw_link"),
        LinkMapping("right_hip", "right_hip_yaw_link"),
        LinkMapping("left_knee", "left_knee_link"),
        LinkMapping("right_knee", "right_knee_link"),
        LinkMapping("left_ankle", "left_ankle_roll_link"),
        LinkMapping("right_ankle", "right_ankle_roll_link"),
        LinkMapping("left_foot", "left_foot_link"),
        LinkMapping("right_foot", "right_foot_link"),
        LinkMapping("left_shoulder", "left_shoulder_roll_link"),
        LinkMapping("right_shoulder", "right_shoulder_roll_link"),
        LinkMapping("left_elbow", "left_elbow_link"),
        LinkMapping("right_elbow", "right_elbow_link"),
        LinkMapping("left_wrist", "left_wrist_yaw_link"),
        LinkMapping("right_wrist", "right_wrist_yaw_link"),
    ),
    source_scales=freeze_mapping({
        "smpl": SourceScale(lower_body=(1.1, 1.1, 1.1), upper_body=(1.1, 1.1, 1.0)),
        "rigv1": SourceScale(lower_body=(1.0, 1.0, 1.0), upper_body=(1.0, 1.0, 0.9)),
    }),
    weights=RetargetingWeights(
        local_alignment=1.0,
        global_alignment=3.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    ),
    hand_aux_offset=(0.0, 0.0, 0.2),
    torso_link_name="torso_link",
    torso_aux_offset=(0.3, 0.0, 0.0),
    global_alignment_keypoint_weights=freeze_mapping({
        "left_hip": 0.25,
        "right_hip": 0.25,
        "left_elbow": 0.25,
        "right_elbow": 0.25,
        "left_hand_aux": 0.25,
        "right_hand_aux": 0.25,
    }),
    rest_penalty_joint_names=(),
    local_alignment_pairs=LOCAL_ALIGNMENT_PAIRS,
)
