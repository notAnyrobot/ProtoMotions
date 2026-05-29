# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as onp

from retargeting.config import N_AUX, N_RETARGET, PyrokiRetargetConfig


@dataclass(frozen=True)
class MotionData:
    keypoints: onp.ndarray
    orientations: onp.ndarray
    left_foot_contact: onp.ndarray
    right_foot_contact: onp.ndarray
    num_timesteps: int


def discover_keypoint_paths(keypoints_folder_path: str | Path) -> list[Path]:
    return sorted(Path(keypoints_folder_path).glob("*.npy"))


def retargeted_output_path(output_dir: str | Path, motion_path: str | Path) -> Path:
    base_filename = Path(motion_path).stem
    return Path(output_dir) / f"{base_filename}_retargeted.npz"


def contacts_output_path(contacts_dir: str | Path, motion_path: str | Path) -> Path:
    base_filename = Path(motion_path).stem
    return Path(contacts_dir) / f"{base_filename}_contacts.npz"


def _crossfade_contacts(contact_flags: onp.ndarray, window_size: int = 5) -> onp.ndarray:
    smoothed = onp.zeros_like(contact_flags)
    for i in range(len(contact_flags)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(contact_flags), i + window_size // 2 + 1)
        smoothed[i] = onp.mean(contact_flags[start_idx:end_idx])
    return smoothed


def _scale_keypoints(
    keypoints: onp.ndarray,
    config: PyrokiRetargetConfig,
    source_type: str,
) -> onp.ndarray:
    try:
        scale = config.source_scales[source_type]
    except KeyError as exc:
        supported = ", ".join(config.supported_source_types)
        raise ValueError(
            f"Invalid source type {source_type!r} for {config.robot_type}. Supported: {supported}"
        ) from exc

    root = keypoints[:, 0, :]
    local = keypoints - root[:, None, :]
    lower_body_local = local[:, 1:9, :] * onp.array(scale.lower_body)[None, None, :]
    upper_body_local = local[:, 9 : N_RETARGET + N_AUX, :] * onp.array(
        scale.upper_body
    )[None, None, :]
    scaled_local = onp.concatenate([lower_body_local, upper_body_local], axis=1)
    scaled_root = root * onp.array(scale.lower_body)[None, :]
    scaled_keypoints = scaled_root[:, None, :] + scaled_local
    return onp.concatenate([scaled_root[:, None, :], scaled_keypoints], axis=1)


def load_motion_data(
    motion_path: str | Path,
    config: PyrokiRetargetConfig,
    source_type: str,
    subsample_factor: int,
    target_raw_frames: int,
) -> MotionData:
    print(f"Loading motion from: {motion_path}")
    motion_data = onp.load(motion_path, allow_pickle=True).item()
    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    raw_positions = motion_data["positions"]
    raw_orientations = motion_data["orientations"]
    raw_left_foot_contacts = motion_data["left_foot_contacts"]
    raw_right_foot_contacts = motion_data["right_foot_contacts"]
    original_raw_frames = raw_positions.shape[0]

    print(f"Original motion length: {original_raw_frames} frames.")
    assert original_raw_frames > 0
    original_subsampled_display_count = raw_positions[::subsample_factor].shape[0]
    num_timesteps = min(original_subsampled_display_count, target_subsampled_frames)
    print(
        "Motion will be displayed for "
        f"{num_timesteps} subsampled frames "
        f"(original subsampled count: {original_subsampled_display_count})."
    )

    if original_raw_frames >= target_raw_frames:
        processed_positions = raw_positions[:target_raw_frames]
        processed_orientations = raw_orientations[:target_raw_frames]
        processed_left_foot_contacts = raw_left_foot_contacts[:target_raw_frames]
        processed_right_foot_contacts = raw_right_foot_contacts[:target_raw_frames]
    else:
        padding_count = target_raw_frames - original_raw_frames
        processed_positions = onp.concatenate(
            (raw_positions, onp.repeat(raw_positions[-1:], padding_count, axis=0)),
            axis=0,
        )
        processed_orientations = onp.concatenate(
            (
                raw_orientations,
                onp.repeat(raw_orientations[-1:], padding_count, axis=0),
            ),
            axis=0,
        )
        processed_left_foot_contacts = onp.concatenate(
            (
                raw_left_foot_contacts,
                onp.repeat(raw_left_foot_contacts[-1:], padding_count, axis=0),
            ),
            axis=0,
        )
        processed_right_foot_contacts = onp.concatenate(
            (
                raw_right_foot_contacts,
                onp.repeat(raw_right_foot_contacts[-1:], padding_count, axis=0),
            ),
            axis=0,
        )

    left_contacts_avg = onp.mean(processed_left_foot_contacts.astype(float), axis=1)[
        :, None
    ]
    right_contacts_avg = onp.mean(processed_right_foot_contacts.astype(float), axis=1)[
        :, None
    ]
    left_foot_contacts_smoothed = _crossfade_contacts(left_contacts_avg)
    right_foot_contacts_smoothed = _crossfade_contacts(right_contacts_avg)

    simplified_keypoints = _scale_keypoints(
        processed_positions[::subsample_factor], config, source_type
    )
    keypoint_orientations = processed_orientations[::subsample_factor]
    left_foot_contact = left_foot_contacts_smoothed[::subsample_factor]
    right_foot_contact = right_foot_contacts_smoothed[::subsample_factor]

    expected_pos_shape = (target_subsampled_frames, N_RETARGET + N_AUX, 3)
    expected_orient_shape = (target_subsampled_frames, N_RETARGET + N_AUX, 3, 3)
    expected_contact_shape = (target_subsampled_frames, 1)
    assert simplified_keypoints.shape == expected_pos_shape, (
        f"Expected positions shape {expected_pos_shape}, got {simplified_keypoints.shape}"
    )
    assert keypoint_orientations.shape == expected_orient_shape, (
        f"Expected orientations shape {expected_orient_shape}, got {keypoint_orientations.shape}"
    )
    assert left_foot_contact.shape == expected_contact_shape, (
        f"Expected left foot contacts shape {expected_contact_shape}, got {left_foot_contact.shape}"
    )
    assert right_foot_contact.shape == expected_contact_shape, (
        f"Expected right foot contacts shape {expected_contact_shape}, got {right_foot_contact.shape}"
    )

    return MotionData(
        keypoints=simplified_keypoints,
        orientations=keypoint_orientations,
        left_foot_contact=left_foot_contact,
        right_foot_contact=right_foot_contact,
        num_timesteps=num_timesteps,
    )


def save_contact_labels(
    output_path: str | Path,
    left_foot_contact: onp.ndarray,
    right_foot_contact: onp.ndarray,
    num_timesteps: int,
) -> None:
    left_contacts = left_foot_contact[:num_timesteps].squeeze(-1)
    right_contacts = right_foot_contact[:num_timesteps].squeeze(-1)
    foot_contacts = onp.stack([left_contacts, right_contacts], axis=-1)
    onp.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


def get_robot_retarget_indices(
    config: PyrokiRetargetConfig,
    link_names: list[str] | tuple[str, ...],
):
    import jax.numpy as jnp

    source_names: list[str] = []
    robot_indices: list[int] = []
    for mapping in config.link_mapping:
        source_names.append(mapping.source_keypoint)
        robot_indices.append(link_names.index(mapping.robot_link))
    return tuple(source_names), jnp.array(robot_indices)


def build_retarget_mask(config: PyrokiRetargetConfig, source_names: tuple[str, ...]):
    import jax.numpy as jnp

    n_retarget = len(source_names)
    retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for pair in config.local_alignment_pairs:
        retarget_idx_a = source_names.index(pair.source_a)
        retarget_idx_b = source_names.index(pair.source_b)
        retarget_mask = retarget_mask.at[retarget_idx_a, retarget_idx_b].set(
            pair.weight
        )
        retarget_mask = retarget_mask.at[retarget_idx_b, retarget_idx_a].set(
            pair.weight
        )
    return retarget_mask
