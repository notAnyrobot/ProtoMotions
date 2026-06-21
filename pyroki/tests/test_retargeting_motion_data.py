from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

from retargeting.config import AlignmentPair
from retargeting.factory import get_retarget_config
from retargeting.solver import (
    RetargetWindow,
    build_retarget_mask,
    discover_keypoint_paths,
    generate_retarget_windows,
    get_robot_retarget_indices,
    load_motion_data,
    retargeted_output_path,
    save_contact_labels,
    validate_chunk_parameters,
)


def _write_keypoints(path: Path, frames: int = 3) -> None:
    positions = np.zeros((frames, 18, 3), dtype=np.float32)
    orientations = np.repeat(np.eye(3, dtype=np.float32)[None, None], frames * 18, axis=0)
    orientations = orientations.reshape(frames, 18, 3, 3)
    positions[:, 0, :] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions[:, 1, :] = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    positions[:, 9, :] = np.array([1.0, 4.0, 3.0], dtype=np.float32)
    left_foot_contacts = np.zeros((frames, 2), dtype=bool)
    right_foot_contacts = np.zeros((frames, 2), dtype=bool)
    left_foot_contacts[0] = True
    right_foot_contacts[1:] = True
    np.save(
        path,
        {
            "positions": positions,
            "orientations": orientations,
            "left_foot_contacts": left_foot_contacts,
            "right_foot_contacts": right_foot_contacts,
        },
    )


def test_discover_keypoint_paths_returns_sorted_npy_files(tmp_path):
    (tmp_path / "b.txt").write_text("ignored")
    _write_keypoints(tmp_path / "z.npy")
    _write_keypoints(tmp_path / "a.npy")

    assert [path.name for path in discover_keypoint_paths(tmp_path)] == ["a.npy", "z.npy"]


def test_retargeted_output_path_uses_current_suffix(tmp_path):
    motion_path = tmp_path / "walk.npy"
    assert retargeted_output_path(tmp_path / "out", motion_path) == tmp_path / "out" / "walk_retargeted.npz"


def test_load_motion_data_applies_g1_smpl_scaling_and_padding(tmp_path):
    motion_path = tmp_path / "walk.npy"
    _write_keypoints(motion_path)
    config = get_retarget_config("g1")

    data = load_motion_data(
        motion_path=motion_path,
        config=config,
        source_type="smpl",
        subsample_factor=2,
        target_raw_frames=4,
    )

    assert data.keypoints.shape == (2, 18, 3)
    assert data.orientations.shape == (2, 18, 3, 3)
    assert data.left_foot_contact.shape == (2, 1)
    assert data.right_foot_contact.shape == (2, 1)
    assert data.num_timesteps == 2
    np.testing.assert_allclose(data.keypoints[0, 0], np.array([0.9, 1.8, 2.55]))
    np.testing.assert_allclose(data.keypoints[0, 1], np.array([1.8, 1.8, 2.55]))
    np.testing.assert_allclose(data.keypoints[0, 9], np.array([0.9, 3.6, 2.55]))
    np.testing.assert_allclose(
        data.keypoints[0, 9, 1],
        data.keypoints[0, 0, 1]
        + (4.0 - 2.0) * config.source_scales["smpl"].upper_body[1],
    )


def test_load_motion_data_defaults_target_raw_frames_to_full_motion(tmp_path):
    motion_path = tmp_path / "walk.npy"
    _write_keypoints(motion_path, frames=5)
    config = get_retarget_config("g1")

    data = load_motion_data(
        motion_path=motion_path,
        config=config,
        source_type="smpl",
        subsample_factor=2,
        target_raw_frames=None,
    )

    assert data.keypoints.shape == (3, 18, 3)
    assert data.orientations.shape == (3, 18, 3, 3)
    assert data.left_foot_contact.shape == (3, 1)
    assert data.right_foot_contact.shape == (3, 1)
    assert data.num_timesteps == 3


def test_load_motion_data_repeats_last_frame_when_padding(tmp_path):
    motion_path = tmp_path / "walk.npy"
    _write_keypoints(motion_path)
    motion = np.load(motion_path, allow_pickle=True).item()
    motion["positions"][2, 0, :] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    motion["positions"][2, 1, :] = np.array([5.0, 3.0, 4.0], dtype=np.float32)
    np.save(motion_path, motion)
    config = get_retarget_config("g1")

    data = load_motion_data(
        motion_path=motion_path,
        config=config,
        source_type="smpl",
        subsample_factor=1,
        target_raw_frames=4,
    )

    lower_body_scale = np.array(config.source_scales["smpl"].lower_body)
    expected_root = np.array([2.0, 3.0, 4.0]) * lower_body_scale
    expected_lower = expected_root + np.array([3.0, 0.0, 0.0]) * lower_body_scale
    assert data.keypoints.shape == (4, 18, 3)
    assert data.num_timesteps == 3
    np.testing.assert_allclose(data.keypoints[3, 0], expected_root)
    np.testing.assert_allclose(data.keypoints[3, 1], expected_lower)
    np.testing.assert_allclose(data.keypoints[3], data.keypoints[2])


def test_save_contact_labels_trims_padding(tmp_path):
    output_path = tmp_path / "contacts.npz"
    left = np.array([[0.25], [0.5], [0.75]], dtype=np.float32)
    right = np.array([[1.0], [0.5], [0.0]], dtype=np.float32)

    save_contact_labels(output_path, left, right, num_timesteps=2)

    saved = np.load(output_path)
    np.testing.assert_allclose(
        saved["foot_contacts"],
        np.array([[0.25, 1.0], [0.5, 0.5]], dtype=np.float32),
    )

def test_get_robot_retarget_indices_uses_config_mapping():
    config = get_retarget_config("g1")
    link_names = [
        "right_wrist_yaw_link",
        "left_elbow_link",
        "left_ankle_roll_link",
        "pelvis_contour_link",
        "right_foot_link",
        "left_shoulder_pitch_link",
        "right_knee_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "left_hip_pitch_link",
        "left_foot_link",
        "right_ankle_roll_link",
        "right_elbow_link",
        "left_knee_link",
        "right_hip_pitch_link",
    ]

    source_names, retarget_indices = get_robot_retarget_indices(config, link_names)

    expected_indices = [
        link_names.index(mapping.robot_link) for mapping in config.link_mapping
    ]
    assert source_names == tuple(
        mapping.source_keypoint for mapping in config.link_mapping
    )
    assert retarget_indices.tolist() == expected_indices


def test_build_retarget_mask_is_symmetric_and_uses_pair_weights():
    non_unit_weight = 0.375
    config = replace(
        get_retarget_config("g1"),
        local_alignment_pairs=(
            AlignmentPair("left_shoulder", "left_elbow", non_unit_weight),
        ),
    )
    source_names = tuple(mapping.source_keypoint for mapping in config.link_mapping)

    mask = build_retarget_mask(config, source_names)

    left_shoulder = source_names.index("left_shoulder")
    left_elbow = source_names.index("left_elbow")
    pelvis = source_names.index("pelvis")
    assert float(mask[left_shoulder, left_elbow]) == non_unit_weight
    assert float(mask[left_elbow, left_shoulder]) == non_unit_weight
    assert float(mask[pelvis, left_shoulder]) == 0.0


def test_validate_chunk_parameters_rejects_invalid_values():
    with pytest.raises(ValueError, match="chunk_size_frames must be positive"):
        validate_chunk_parameters(
            chunk_threshold_frames=900,
            chunk_size_frames=0,
            chunk_overlap_frames=60,
        )

    with pytest.raises(ValueError, match="chunk_overlap_frames must be non-negative"):
        validate_chunk_parameters(
            chunk_threshold_frames=900,
            chunk_size_frames=450,
            chunk_overlap_frames=-1,
        )

    with pytest.raises(ValueError, match="chunk_overlap_frames must be smaller"):
        validate_chunk_parameters(
            chunk_threshold_frames=900,
            chunk_size_frames=450,
            chunk_overlap_frames=450,
        )

    with pytest.raises(ValueError, match="chunk_threshold_frames must be at least"):
        validate_chunk_parameters(
            chunk_threshold_frames=449,
            chunk_size_frames=450,
            chunk_overlap_frames=60,
        )


def test_generate_retarget_windows_keeps_short_motion_as_one_window():
    windows = generate_retarget_windows(
        num_frames=900,
        chunk_threshold_frames=900,
        chunk_size_frames=450,
        chunk_overlap_frames=60,
    )

    assert windows == [RetargetWindow(start=0, end=900)]


def test_generate_retarget_windows_chunks_long_motion_with_overlap():
    windows = generate_retarget_windows(
        num_frames=1200,
        chunk_threshold_frames=900,
        chunk_size_frames=450,
        chunk_overlap_frames=60,
    )

    assert windows == [
        RetargetWindow(start=0, end=450),
        RetargetWindow(start=390, end=840),
        RetargetWindow(start=750, end=1200),
    ]
    assert all(window.end > window.start for window in windows)
    assert windows[-1].end == 1200


def test_generate_retarget_windows_rejects_empty_motion():
    with pytest.raises(ValueError, match="num_frames must be positive"):
        generate_retarget_windows(
            num_frames=0,
            chunk_threshold_frames=900,
            chunk_size_frames=450,
            chunk_overlap_frames=60,
        )
