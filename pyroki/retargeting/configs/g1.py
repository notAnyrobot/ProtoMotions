# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from retargeting.config import (
    AlignmentPair,
    LinkMapping,
    PyrokiRetargetConfig,
    RetargetingWeights,
    SourceScale,
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

G1_RETARGET_CONFIG = PyrokiRetargetConfig(
    robot_type="g1",
    display_name="G1",
    supported_source_types=("smpl", "rigv1"),
    urdf_path=REPO_ROOT / "protomotions/data/assets/urdf/for_retargeting/g1.urdf",
    mesh_dir=REPO_ROOT / "protomotions/data/assets/mesh/G1",
    link_mapping=(
        LinkMapping("pelvis", "pelvis_contour_link"),
        LinkMapping("left_hip", "left_hip_pitch_link"),
        LinkMapping("right_hip", "right_hip_pitch_link"),
        LinkMapping("left_knee", "left_knee_link"),
        LinkMapping("right_knee", "right_knee_link"),
        LinkMapping("left_ankle", "left_ankle_roll_link"),
        LinkMapping("right_ankle", "right_ankle_roll_link"),
        LinkMapping("left_foot", "left_foot_link"),
        LinkMapping("right_foot", "right_foot_link"),
        LinkMapping("left_shoulder", "left_shoulder_pitch_link"),
        LinkMapping("right_shoulder", "right_shoulder_pitch_link"),
        LinkMapping("left_elbow", "left_elbow_link"),
        LinkMapping("right_elbow", "right_elbow_link"),
        LinkMapping("left_wrist", "left_wrist_yaw_link"),
        LinkMapping("right_wrist", "right_wrist_yaw_link"),
    ),
    source_scales={
        "smpl": SourceScale(lower_body=(0.9, 0.9, 0.85), upper_body=(0.9, 0.9, 0.8)),
        "rigv1": SourceScale(lower_body=(0.8, 0.8, 0.75), upper_body=(0.8, 0.8, 0.7)),
    },
    weights=RetargetingWeights(
        local_alignment=1.0,
        global_alignment=4.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    ),
    hand_aux_offset=(0.0, 0.0, 0.14),
    torso_link_name="torso_link",
    torso_aux_offset=(0.15, 0.0, -0.1),
    global_alignment_keypoint_weights={
        "left_elbow": 0.25,
        "right_elbow": 0.25,
        "left_hand_aux": 0.25,
        "right_hand_aux": 0.25,
    },
    rest_penalty_joint_names=(
        "waist_roll_joint",
        "right_wrist_pitch_joint",
        "left_wrist_pitch_joint",
    ),
    local_alignment_pairs=LOCAL_ALIGNMENT_PAIRS,
)
