from pathlib import Path
import sys

import pytest
import torch
import numpy as np


DATA_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "data" / "scripts"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))

from convert_pyroki_retargeted_robot_motions_to_proto import (
    get_robot_mjcf_path,
    main,
    resample_contact_probabilities,
    resample_motion_components,
)
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.robot_configs.factory import robot_config


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    root_rot = torch.zeros(num_frames, 4, dtype=torch.float32)
    root_rot[:, 0] = 1.0
    return root_rot


def _num_dofs_for_robot(robot_type: str) -> int:
    robot_cfg = robot_config(robot_type)
    mjcf_path = get_robot_mjcf_path(robot_type, robot_cfg)
    kinematic_info = extract_kinematic_info(str(mjcf_path))
    return kinematic_info.nq - 7


def _foot_body_indices_for_robot(robot_type: str) -> tuple[int, int]:
    robot_cfg = robot_config(robot_type)
    mjcf_path = get_robot_mjcf_path(robot_type, robot_cfg)
    kinematic_info = extract_kinematic_info(str(mjcf_path))
    left_foot_name = robot_cfg.common_naming_to_robot_body_names[
        "all_left_foot_bodies"
    ][0]
    right_foot_name = robot_cfg.common_naming_to_robot_body_names[
        "all_right_foot_bodies"
    ][0]
    return (
        kinematic_info.body_names.index(left_foot_name),
        kinematic_info.body_names.index(right_foot_name),
    )


def test_converter_accepts_astro_robot_config_asset(tmp_path):
    main(
        retargeted_motion_dir=tmp_path / "retargeted",
        output_dir=tmp_path / "proto",
        input_fps=30,
        output_fps=30,
        force_remake=False,
        ignore_first_n_frames=0,
        apply_motion_filter=False,
        min_height_threshold=-0.05,
        max_velocity_threshold=15.0,
        max_dof_vel_threshold=40.0,
        duration_height_filter=0.1,
        duration_height_seconds=0.6,
        robot_type="astro",
        contact_labels_dir=None,
    )


def test_resample_motion_components_upsamples_30_to_50_on_time_grid():
    root_pos = torch.arange(4, dtype=torch.float32).reshape(4, 1).repeat(1, 3)
    root_rot = _identity_root_rot(4)
    joint_angles = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    resampled_root_pos, _, resampled_joint_angles = resample_motion_components(
        root_pos, root_rot, joint_angles, input_fps=30, output_fps=50
    )

    assert resampled_root_pos.shape[0] == 6
    assert torch.allclose(
        resampled_root_pos[:, 0],
        torch.tensor([0.0, 0.6, 1.2, 1.8, 2.4, 3.0]),
    )
    assert torch.allclose(
        resampled_joint_angles[:, 0],
        torch.tensor([0.0, 1.2, 2.4, 3.6, 4.8, 6.0]),
    )


def test_resample_motion_components_downsamples_120_to_30_on_same_time_grid():
    root_pos = torch.arange(9, dtype=torch.float32).reshape(9, 1).repeat(1, 3)
    root_rot = _identity_root_rot(9)
    joint_angles = torch.arange(18, dtype=torch.float32).reshape(9, 2)

    resampled_root_pos, _, resampled_joint_angles = resample_motion_components(
        root_pos, root_rot, joint_angles, input_fps=120, output_fps=30
    )

    assert resampled_root_pos.shape[0] == 3
    assert torch.allclose(resampled_root_pos[:, 0], torch.tensor([0.0, 4.0, 8.0]))
    assert torch.allclose(
        resampled_joint_angles[:, 0], torch.tensor([0.0, 8.0, 16.0])
    )


def test_resample_motion_components_preserves_first_and_last_frames():
    root_pos = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    root_rot = _identity_root_rot(5)
    joint_angles = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    resampled_root_pos, resampled_root_rot, resampled_joint_angles = (
        resample_motion_components(
            root_pos, root_rot, joint_angles, input_fps=30, output_fps=50
        )
    )

    assert resampled_root_pos.shape[0] == 8
    assert torch.allclose(resampled_root_pos[0], root_pos[0])
    assert torch.allclose(resampled_root_pos[-1], root_pos[-1])
    assert torch.allclose(resampled_root_rot[0], root_rot[0])
    assert torch.allclose(resampled_root_rot[-1], root_rot[-1])
    assert torch.allclose(resampled_joint_angles[0], joint_angles[0])
    assert torch.allclose(resampled_joint_angles[-1], joint_angles[-1])


def test_resample_motion_components_slerps_wxyz_root_quaternions():
    root_pos = torch.zeros(2, 3, dtype=torch.float32)
    root_rot = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    joint_angles = torch.zeros(2, 1, dtype=torch.float32)

    _, resampled_root_rot, _ = resample_motion_components(
        root_pos, root_rot, joint_angles, input_fps=1, output_fps=2
    )

    assert torch.allclose(
        resampled_root_rot.norm(dim=-1), torch.ones(3), atol=1e-6
    )
    assert torch.allclose(
        resampled_root_rot[1],
        torch.tensor([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=torch.float32),
        atol=1e-5,
    )


def test_resample_motion_components_unwraps_joint_angles_before_interpolation():
    root_pos = torch.zeros(2, 3, dtype=torch.float32)
    root_rot = _identity_root_rot(2)
    joint_angles = torch.tensor([[3.0], [-3.0]], dtype=torch.float32)

    _, _, resampled_joint_angles = resample_motion_components(
        root_pos, root_rot, joint_angles, input_fps=1, output_fps=2
    )

    assert resampled_joint_angles.shape == (3, 1)
    assert resampled_joint_angles[1, 0].item() == pytest.approx(np.pi, abs=1e-5)
    assert resampled_joint_angles[-1, 0].item() == pytest.approx(
        -3.0 + 2 * np.pi, abs=1e-5
    )


def test_resample_contact_probabilities_linearly_resamples_contacts():
    foot_contacts = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    resampled_contacts = resample_contact_probabilities(
        foot_contacts, input_fps=1, output_fps=2
    )

    assert torch.allclose(
        resampled_contacts,
        torch.tensor(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_converter_resamples_non_divisible_30_to_50_and_motionlib_dt(tmp_path):
    retargeted_dir = tmp_path / "retargeted"
    output_dir = tmp_path / "proto"
    retargeted_dir.mkdir()

    num_frames = 4
    np.savez(
        retargeted_dir / "walk_retargeted.npz",
        base_frame_pos=np.stack(
            [
                np.linspace(0.0, 0.03, num_frames, dtype=np.float32),
                np.zeros(num_frames, dtype=np.float32),
                np.full(num_frames, 0.8, dtype=np.float32),
            ],
            axis=-1,
        ),
        base_frame_wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_frames, 1)
        ),
        joint_angles=np.zeros((num_frames, _num_dofs_for_robot("g1")), dtype=np.float32),
    )

    main(
        retargeted_motion_dir=retargeted_dir,
        output_dir=output_dir,
        input_fps=30,
        output_fps=50,
        force_remake=False,
        ignore_first_n_frames=0,
        apply_motion_filter=False,
        min_height_threshold=-0.05,
        max_velocity_threshold=15.0,
        max_dof_vel_threshold=40.0,
        duration_height_filter=0.1,
        duration_height_seconds=0.6,
        robot_type="g1",
        contact_labels_dir=None,
    )

    motion_path = output_dir / "walk_retargeted.motion"
    assert motion_path.exists()

    motion_data = torch.load(motion_path, weights_only=False)
    assert motion_data["fps"] == 50
    assert motion_data["dof_pos"].shape[0] == 6

    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(motion_path)), device="cpu"
    )
    packaged_path = tmp_path / "walk.pt"
    motion_lib.save_to_file(packaged_path)

    packaged_data = torch.load(packaged_path, weights_only=False)
    assert packaged_data["motion_dt"][0].item() == pytest.approx(1.0 / 50.0)


def test_converter_trims_resampled_contacts_in_output_frame_space(tmp_path, capsys):
    retargeted_dir = tmp_path / "retargeted"
    contact_labels_dir = tmp_path / "contacts"
    output_dir = tmp_path / "proto"
    retargeted_dir.mkdir()
    contact_labels_dir.mkdir()

    num_frames = 4
    np.savez(
        retargeted_dir / "walk_retargeted.npz",
        base_frame_pos=np.stack(
            [
                np.linspace(0.0, 0.03, num_frames, dtype=np.float32),
                np.zeros(num_frames, dtype=np.float32),
                np.full(num_frames, 0.8, dtype=np.float32),
            ],
            axis=-1,
        ),
        base_frame_wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_frames, 1)
        ),
        joint_angles=np.zeros((num_frames, _num_dofs_for_robot("g1")), dtype=np.float32),
    )
    np.savez(
        contact_labels_dir / "walk_contacts.npz",
        foot_contacts=np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    main(
        retargeted_motion_dir=retargeted_dir,
        output_dir=output_dir,
        input_fps=30,
        output_fps=50,
        force_remake=False,
        ignore_first_n_frames=2,
        apply_motion_filter=False,
        min_height_threshold=-0.05,
        max_velocity_threshold=15.0,
        max_dof_vel_threshold=40.0,
        duration_height_filter=0.1,
        duration_height_seconds=0.6,
        robot_type="g1",
        contact_labels_dir=contact_labels_dir,
    )

    captured = capsys.readouterr()
    assert "Motion length (4) does not match contact length (6)" not in captured.out

    motion_path = output_dir / "walk_retargeted.motion"
    assert motion_path.exists()

    motion_data = torch.load(motion_path, weights_only=False)
    contacts = motion_data["rigid_body_contacts"]
    left_foot_idx, right_foot_idx = _foot_body_indices_for_robot("g1")

    assert contacts.shape[0] == 4
    assert torch.equal(
        contacts[:, left_foot_idx],
        torch.tensor([False, True, True, True]),
    )
    assert torch.equal(
        contacts[:, right_foot_idx],
        torch.tensor([True, False, False, False]),
    )
