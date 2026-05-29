from pathlib import Path
import sys

import numpy as np

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

from retargeting.factory import get_retarget_config
from retargeting.solver import (
    discover_keypoint_paths,
    load_motion_data,
    retargeted_output_path,
    save_contact_labels,
)


def _write_keypoints(path: Path, frames: int = 3) -> None:
    positions = np.zeros((frames, 18, 3), dtype=np.float32)
    orientations = np.repeat(np.eye(3, dtype=np.float32)[None, None], frames * 18, axis=0)
    orientations = orientations.reshape(frames, 18, 3, 3)
    positions[:, 0, :] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions[:, 1, :] = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    positions[:, 9, :] = np.array([1.0, 4.0, 3.0], dtype=np.float32)
    np.save(
        path,
        {
            "positions": positions,
            "orientations": orientations,
            "left_foot_contacts": np.array([[1, 1], [0, 0], [0, 0]], dtype=bool),
            "right_foot_contacts": np.array([[0, 0], [1, 1], [1, 1]], dtype=bool),
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
