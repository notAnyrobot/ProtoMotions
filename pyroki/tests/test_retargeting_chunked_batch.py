from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

from retargeting.cli import BatchRetargetingOptions
from retargeting.factory import get_retarget_config
import retargeting.solver as solver


def _write_keypoints(path: Path, frames: int) -> None:
    positions = np.zeros((frames, 18, 3), dtype=np.float32)
    positions[:, 0, 0] = np.arange(frames, dtype=np.float32)
    orientations = np.repeat(
        np.eye(3, dtype=np.float32)[None, None],
        frames * 18,
        axis=0,
    )
    orientations = orientations.reshape(frames, 18, 3, 3)
    np.save(
        path,
        {
            "positions": positions,
            "orientations": orientations,
            "left_foot_contacts": np.zeros((frames, 2), dtype=bool),
            "right_foot_contacts": np.zeros((frames, 2), dtype=bool),
        },
    )


def _options(
    keypoints_dir: Path,
    output_dir: Path,
    *,
    chunk_long_motions: bool,
) -> BatchRetargetingOptions:
    return BatchRetargetingOptions(
        keypoints_folder_path=keypoints_dir,
        output_dir=output_dir,
        subsample_factor=1,
        target_raw_frames=None,
        skip_existing=False,
        source_type="smpl",
        save_contacts_only=False,
        contacts_dir=None,
        input_fps=60.0,
        visualize=False,
        chunk_long_motions=chunk_long_motions,
        chunk_threshold_frames=6,
        chunk_size_frames=4,
        chunk_overlap_frames=2,
    )


def test_non_visual_batch_keeps_short_motion_on_full_solve(monkeypatch, tmp_path):
    keypoints_dir = tmp_path / "keypoints"
    output_dir = tmp_path / "out"
    keypoints_dir.mkdir()
    _write_keypoints(keypoints_dir / "short.npy", frames=6)
    calls = []

    monkeypatch.setattr(
        solver,
        "_load_robot",
        lambda config: (
            None,
            object(),
            None,
            (),
            (),
            np.array([], dtype=np.int32),
            np.zeros((0, 0)),
        ),
    )

    def fake_solve_motion_data_retargeting(**kwargs):
        motion_data = kwargs["motion_data"]
        calls.append(motion_data.num_timesteps)
        return solver.RetargetedMotion(
            base_frame_pos=motion_data.keypoints[:, 0, :],
            base_frame_wxyz=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (motion_data.num_timesteps, 1),
            ),
            joint_angles=np.zeros((motion_data.num_timesteps, 2), dtype=np.float32),
        )

    monkeypatch.setattr(
        solver,
        "solve_motion_data_retargeting",
        fake_solve_motion_data_retargeting,
    )

    solver.run_non_visualized_batch(
        get_retarget_config("g1"),
        _options(keypoints_dir, output_dir, chunk_long_motions=True),
        [keypoints_dir / "short.npy"],
    )

    assert calls == [6]
    saved = np.load(output_dir / "short_retargeted.npz")
    assert saved["base_frame_pos"].shape == (6, 3)
    assert saved["joint_angles"].shape == (6, 2)


def test_non_visual_batch_chunks_long_motion_and_writes_one_output(monkeypatch, tmp_path):
    keypoints_dir = tmp_path / "keypoints"
    output_dir = tmp_path / "out"
    keypoints_dir.mkdir()
    _write_keypoints(keypoints_dir / "long.npy", frames=10)
    calls = []

    monkeypatch.setattr(
        solver,
        "_load_robot",
        lambda config: (
            None,
            object(),
            None,
            (),
            (),
            np.array([], dtype=np.int32),
            np.zeros((0, 0)),
        ),
    )

    def fake_solve_motion_data_retargeting(**kwargs):
        motion_data = kwargs["motion_data"]
        calls.append(motion_data.num_timesteps)
        return solver.RetargetedMotion(
            base_frame_pos=motion_data.keypoints[:, 0, :],
            base_frame_wxyz=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (motion_data.num_timesteps, 1),
            ),
            joint_angles=motion_data.keypoints[:, 0, 0:1],
        )

    monkeypatch.setattr(
        solver,
        "solve_motion_data_retargeting",
        fake_solve_motion_data_retargeting,
    )

    solver.run_non_visualized_batch(
        get_retarget_config("g1"),
        _options(keypoints_dir, output_dir, chunk_long_motions=True),
        [keypoints_dir / "long.npy"],
    )

    assert calls == [4, 4, 4, 4]
    saved = np.load(output_dir / "long_retargeted.npz")
    assert set(saved.files) == {"base_frame_pos", "base_frame_wxyz", "joint_angles"}
    assert saved["base_frame_pos"].shape == (10, 3)
    assert saved["joint_angles"].shape == (10, 1)
    np.testing.assert_allclose(saved["base_frame_pos"][:, 0], np.arange(10) * 0.9)


def test_non_visual_batch_skip_existing_still_skips_final_output(monkeypatch, tmp_path):
    keypoints_dir = tmp_path / "keypoints"
    output_dir = tmp_path / "out"
    keypoints_dir.mkdir()
    output_dir.mkdir()
    _write_keypoints(keypoints_dir / "long.npy", frames=10)
    np.savez_compressed(
        output_dir / "long_retargeted.npz",
        base_frame_pos=np.zeros((1, 3)),
        base_frame_wxyz=np.zeros((1, 4)),
        joint_angles=np.zeros((1, 2)),
    )
    calls = []

    monkeypatch.setattr(
        solver,
        "_load_robot",
        lambda config: (
            None,
            object(),
            None,
            (),
            (),
            np.array([], dtype=np.int32),
            np.zeros((0, 0)),
        ),
    )
    monkeypatch.setattr(
        solver,
        "solve_motion_data_retargeting",
        lambda **kwargs: calls.append(kwargs["motion_data"].num_timesteps),
    )

    options = replace(
        _options(keypoints_dir, output_dir, chunk_long_motions=True),
        skip_existing=True,
    )
    solver.run_non_visualized_batch(
        get_retarget_config("g1"),
        options,
        [keypoints_dir / "long.npy"],
    )

    assert calls == []
