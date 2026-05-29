# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import time
from typing import Tuple, TypedDict

import numpy as onp

from retargeting.config import (
    N_AUX,
    N_RETARGET,
    PyrokiRetargetConfig,
    RetargetingWeights,
)

try:
    import jax
    import jax.numpy as jnp
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy
except ModuleNotFoundError as exc:
    jax = None
    jnp = None
    jdc = None
    jaxlie = None
    jaxls = None
    pk = None
    yourdfpy = None
    _SOLVER_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _SOLVER_IMPORT_ERROR = None


class SolverWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float


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


def run_batch_retargeting(config: PyrokiRetargetConfig, options) -> None:
    keypoint_paths = discover_keypoint_paths(options.keypoints_folder_path)
    if not keypoint_paths:
        print(f"No .npy files found in {options.keypoints_folder_path}. Exiting.")
        return

    if options.save_contacts_only:
        run_contacts_only(config, options, keypoint_paths)
        return

    if options.visualize:
        run_visualizer(config, options, keypoint_paths)
        return

    run_non_visualized_batch(config, options, keypoint_paths)


def run_contacts_only(
    config: PyrokiRetargetConfig,
    options,
    keypoint_paths: list[Path],
) -> None:
    print(
        "Running in save-contacts-only mode. "
        "Extracting foot contact labels from source motions."
    )
    contacts_dir = (
        options.contacts_dir
        if options.contacts_dir is not None
        else options.keypoints_folder_path / "contacts"
    )
    contacts_dir.mkdir(parents=True, exist_ok=True)

    for i, motion_path in enumerate(keypoint_paths):
        print(f"Processing motion {i + 1}/{len(keypoint_paths)}: {motion_path.name}")
        output_path = contacts_output_path(contacts_dir, motion_path)
        if options.skip_existing and output_path.exists():
            print(f"Output file {output_path.name} already exists, skipping...")
            continue

        motion_data = load_motion_data(
            motion_path=motion_path,
            config=config,
            source_type=options.source_type,
            subsample_factor=options.subsample_factor,
            target_raw_frames=options.target_raw_frames,
        )
        save_contact_labels(
            output_path,
            motion_data.left_foot_contact,
            motion_data.right_foot_contact,
            motion_data.num_timesteps,
        )


def _require_solver_dependencies() -> None:
    if _SOLVER_IMPORT_ERROR is not None:
        raise ImportError(
            "PyRoki solver execution requires JAX/PyRoki runtime dependencies"
        ) from _SOLVER_IMPORT_ERROR


def _load_robot(config: PyrokiRetargetConfig):
    _require_solver_dependencies()
    urdf = yourdfpy.URDF.load(str(config.urdf_path), mesh_dir=str(config.mesh_dir))
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = None
    link_names = tuple(robot.links.names)
    source_names, joint_retarget_indices = get_robot_retarget_indices(
        config, link_names
    )
    retarget_mask = build_retarget_mask(config, source_names)
    return (
        urdf,
        robot,
        robot_coll,
        link_names,
        source_names,
        joint_retarget_indices,
        retarget_mask,
    )


def run_visualizer(
    config: PyrokiRetargetConfig,
    options,
    keypoint_paths: list[Path],
) -> None:
    _require_solver_dependencies()

    import viser
    from viser.extras import ViserUrdf

    (
        urdf,
        robot,
        robot_coll,
        link_names,
        source_names,
        joint_retarget_indices,
        retarget_mask,
    ) = _load_robot(config)

    current_motion_index = 0
    motion_data = load_motion_data(
        motion_path=keypoint_paths[current_motion_index],
        config=config,
        source_type=options.source_type,
        subsample_factor=options.subsample_factor,
        target_raw_frames=options.target_raw_frames,
    )
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider(
        "timestep",
        0,
        motion_data.num_timesteps - 1 if motion_data.num_timesteps > 0 else 0,
        1,
        0,
    )

    def reset_timeline_callback(_: viser.GuiEvent):
        timestep_slider.value = 0

    reset_timeline_button = server.gui.add_button("Reset Timeline")
    reset_timeline_button.on_click(reset_timeline_callback)

    weights = pk.viewer.WeightTuner(
        server,
        config.weights.as_dict(),  # type: ignore
    )

    ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal ts_world_root, joints
        gen_button.disabled = True
        retarget_next_button.disabled = True
        tuned_config = replace(
            config,
            weights=RetargetingWeights(**weights.get_weights()),  # type: ignore
        )
        ts_world_root, joints = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=motion_data.keypoints,
            target_orientations=motion_data.orientations,
            left_foot_contact=motion_data.left_foot_contact,
            right_foot_contact=motion_data.right_foot_contact,
            joint_retarget_indices=joint_retarget_indices,
            retarget_mask=retarget_mask,
            source_names=source_names,
            link_names=link_names,
            config=tuned_config,
            subsample_factor=options.subsample_factor,
            input_fps=options.input_fps,
        )
        gen_button.disabled = False
        retarget_next_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    def retarget_next_motion(_: viser.GuiEvent):
        nonlocal current_motion_index, ts_world_root, joints, motion_data
        current_motion_index = (current_motion_index + 1) % len(keypoint_paths)
        motion_data = load_motion_data(
            motion_path=keypoint_paths[current_motion_index],
            config=config,
            source_type=options.source_type,
            subsample_factor=options.subsample_factor,
            target_raw_frames=options.target_raw_frames,
        )

        timestep_slider.max = (
            motion_data.num_timesteps - 1 if motion_data.num_timesteps > 0 else 0
        )
        timestep_slider.value = 0

        ts_world_root, joints = None, None
        generate_trajectory()

    retarget_next_button = server.gui.add_button("Retarget Next")
    retarget_next_button.on_click(retarget_next_motion)

    generate_trajectory()
    assert ts_world_root is not None and joints is not None

    while True:
        with server.atomic():
            if playing.value and motion_data.num_timesteps > 0:
                timestep_slider.value = (
                    timestep_slider.value + 1
                ) % motion_data.num_timesteps
            tstep = timestep_slider.value

        try:
            base_frame.wxyz = onp.array(ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(ts_world_root.wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(motion_data.keypoints[tstep]),
                onp.array((0, 0, 255))[None].repeat(
                    motion_data.keypoints.shape[1], axis=0
                ),
                point_size=0.01,
            )
        except Exception:
            pass

        time.sleep(options.subsample_factor / options.input_fps)


def run_non_visualized_batch(
    config: PyrokiRetargetConfig,
    options,
    keypoint_paths: list[Path],
) -> None:
    print("Running in non-visualize mode. Retargeting all motions and saving to disk.")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (
        _,
        robot,
        robot_coll,
        link_names,
        source_names,
        joint_retarget_indices,
        retarget_mask,
    ) = _load_robot(config)

    for i, motion_path in enumerate(keypoint_paths):
        print(f"Processing motion {i + 1}/{len(keypoint_paths)}: {motion_path.name}")
        output_path = retargeted_output_path(options.output_dir, motion_path)
        if options.skip_existing and output_path.exists():
            print(f"Output file {output_path.name} already exists, skipping...")
            continue

        motion_data = load_motion_data(
            motion_path=motion_path,
            config=config,
            source_type=options.source_type,
            subsample_factor=options.subsample_factor,
            target_raw_frames=options.target_raw_frames,
        )
        ts_world_root, joints = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=motion_data.keypoints,
            target_orientations=motion_data.orientations,
            left_foot_contact=motion_data.left_foot_contact,
            right_foot_contact=motion_data.right_foot_contact,
            joint_retarget_indices=joint_retarget_indices,
            retarget_mask=retarget_mask,
            source_names=source_names,
            link_names=link_names,
            config=config,
            subsample_factor=options.subsample_factor,
            input_fps=options.input_fps,
        )
        results_to_save = {
            "base_frame_pos": onp.array(
                ts_world_root.wxyz_xyz[: motion_data.num_timesteps, 4:]
            ),
            "base_frame_wxyz": onp.array(
                ts_world_root.wxyz_xyz[: motion_data.num_timesteps, :4]
            ),
            "joint_angles": onp.array(joints[: motion_data.num_timesteps]),
        }
        onp.savez_compressed(output_path, **results_to_save)
        print(f"Saved retargeted motion to {output_path}")


if _SOLVER_IMPORT_ERROR is None:

    @jaxls.Cost.create_factory
    def joint_vel_limit_cost(
        var_values: jaxls.VarValues,
        var_joints_curr: jaxls.Var[jnp.ndarray],
        var_joints_prev: jaxls.Var[jnp.ndarray],
        max_vel: float,
        dt: float,
        weight: float,
    ) -> jax.Array:
        """Joint velocity limit cost to prevent excessive joint velocities."""
        joints_curr = var_values[var_joints_curr]
        joints_prev = var_values[var_joints_prev]

        joint_vel = (joints_curr - joints_prev) / dt
        excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)

        return excess_vel.flatten() * weight

    @jaxls.Cost.create_factory
    def foot_contact_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root_curr: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
        var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
        var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
        robot: pk.Robot,
        left_foot_contact: jnp.ndarray,
        right_foot_contact: jnp.ndarray,
        joint_retarget_indices: jnp.ndarray,
        foot_indices: jnp.ndarray,
        weight: float,
    ) -> jax.Array:
        """Penalize contacted foot velocity and ankle/toe z-height mismatch."""
        T_world_root_curr = var_values[var_Ts_world_root_curr]
        T_world_root_prev = var_values[var_Ts_world_root_prev]
        robot_cfg_curr = var_values[var_robot_cfg_curr]
        robot_cfg_prev = var_values[var_robot_cfg_prev]

        T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr))
        T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
        T_world_link_curr = T_world_root_curr @ T_root_link_curr
        T_world_link_prev = T_world_root_prev @ T_root_link_prev

        left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices

        left_ankle_robot_idx = joint_retarget_indices[left_ankle_idx]
        right_ankle_robot_idx = joint_retarget_indices[right_ankle_idx]
        left_foot_robot_idx = joint_retarget_indices[left_foot_idx]
        right_foot_robot_idx = joint_retarget_indices[right_foot_idx]

        robot_positions_curr = T_world_link_curr.translation()
        robot_positions_prev = T_world_link_prev.translation()

        left_ankle_curr = robot_positions_curr[left_ankle_robot_idx]
        right_ankle_curr = robot_positions_curr[right_ankle_robot_idx]
        left_foot_curr = robot_positions_curr[left_foot_robot_idx]
        right_foot_curr = robot_positions_curr[right_foot_robot_idx]

        left_ankle_prev = robot_positions_prev[left_ankle_robot_idx]
        right_ankle_prev = robot_positions_prev[right_ankle_robot_idx]
        left_foot_prev = robot_positions_prev[left_foot_robot_idx]
        right_foot_prev = robot_positions_prev[right_foot_robot_idx]

        left_ankle_vel = left_ankle_curr - left_ankle_prev
        right_ankle_vel = right_ankle_curr - right_ankle_prev
        left_foot_vel = left_foot_curr - left_foot_prev
        right_foot_vel = right_foot_curr - right_foot_prev

        left_ankle_toe_z_diff = left_ankle_curr[2] - left_foot_curr[2]
        right_ankle_toe_z_diff = right_ankle_curr[2] - right_foot_curr[2]

        left_contact_weight = left_foot_contact[0]
        right_contact_weight = right_foot_contact[0]

        left_ankle_vel_cost = left_contact_weight * left_ankle_vel
        right_ankle_vel_cost = right_contact_weight * right_ankle_vel
        left_foot_vel_cost = left_contact_weight * left_foot_vel
        right_foot_vel_cost = right_contact_weight * right_foot_vel

        left_z_consistency_cost = left_contact_weight * left_ankle_toe_z_diff
        right_z_consistency_cost = right_contact_weight * right_ankle_toe_z_diff

        return (
            jnp.concatenate(
                [
                    left_ankle_vel_cost.flatten(),
                    right_ankle_vel_cost.flatten(),
                    left_foot_vel_cost.flatten(),
                    right_foot_vel_cost.flatten(),
                    jnp.array([left_z_consistency_cost]),
                    jnp.array([right_z_consistency_cost]),
                ]
            )
            * weight
        )

    @jaxls.Cost.create_factory
    def foot_tilt_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        robot: pk.Robot,
        left_foot_contact: jnp.ndarray,
        right_foot_contact: jnp.ndarray,
        joint_retarget_indices: jnp.ndarray,
        foot_indices: jnp.ndarray,
        weight: float,
    ) -> jax.Array:
        """Cost to penalize foot tilting when in contact - keep z axis up."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        left_ankle_idx, right_ankle_idx, _, _ = foot_indices

        left_ankle_robot_idx = joint_retarget_indices[left_ankle_idx]
        right_ankle_robot_idx = joint_retarget_indices[right_ankle_idx]

        left_foot_ori = T_world_link.rotation().as_matrix()[left_ankle_robot_idx]
        right_foot_ori = T_world_link.rotation().as_matrix()[right_ankle_robot_idx]

        left_contact_weight = left_foot_contact[0]
        right_contact_weight = right_foot_contact[0]

        left_tilt_residual = left_contact_weight * (left_foot_ori[2, 2] - 1.0)
        right_tilt_residual = right_contact_weight * (right_foot_ori[2, 2] - 1.0)

        return (
            jnp.concatenate(
                [
                    jnp.array([left_tilt_residual]),
                    jnp.array([right_tilt_residual]),
                ]
            )
            * weight
        )

    def _solve_retargeting_impl(
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision | None,
        target_keypoints: jnp.ndarray,
        target_orientations: jnp.ndarray,
        left_foot_contact: jnp.ndarray,
        right_foot_contact: jnp.ndarray,
        joint_retarget_indices: jnp.ndarray,
        retarget_mask: jnp.ndarray,
        weights: SolverWeights,
        foot_indices: jnp.ndarray,
        left_wrist_robot_idx: jnp.ndarray,
        right_wrist_robot_idx: jnp.ndarray,
        torso_link_idx: int,
        hand_aux_offset: jnp.ndarray,
        torso_aux_offset: jnp.ndarray,
        keypoint_weight_indices: jnp.ndarray,
        keypoint_weight_multipliers: jnp.ndarray,
        rest_penalty_joint_indices: jnp.ndarray,
        max_joint_velocity: float,
        subsample_factor: int = 1,
        input_fps: float = 30.0,
        *,
        max_iterations: int,
    ) -> Tuple[jaxlie.SE3, jnp.ndarray]:
        """Solve the simplified retargeting problem."""
        n_retarget = len(joint_retarget_indices)
        timesteps = target_keypoints.shape[0]

        class SimplifiedJointsScaleVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.ones((n_retarget, n_retarget)),
        ): ...

        var_joints = robot.joint_var_cls(jnp.arange(timesteps))
        var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
        var_joints_scale = SimplifiedJointsScaleVar(jnp.zeros(timesteps))

        root_init_se3_list = []
        for t in range(timesteps):
            root_pos_t = target_keypoints[t, 0, :]
            root_rot_t = target_orientations[t, 0, :, :]

            root_se3_t = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(root_rot_t), root_pos_t
            )
            root_init_se3_list.append(root_se3_t)

        root_init_values = jaxlie.SE3(
            jnp.stack([se3.wxyz_xyz for se3 in root_init_se3_list])
        )

        costs: list[jaxls.Cost] = []

        @jaxls.Cost.create_factory
        def retargeting_cost(
            var_values: jaxls.VarValues,
            var_Ts_world_root: jaxls.SE3Var,
            var_robot_cfg: jaxls.Var[jnp.ndarray],
            var_joints_scale: SimplifiedJointsScaleVar,
            keypoints: jnp.ndarray,
        ) -> jax.Array:
            """Retargeting factor for relative keypoint vectors and angles."""
            robot_cfg = var_values[var_robot_cfg]
            T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
            T_world_root = var_values[var_Ts_world_root]
            T_world_link = T_world_root @ T_root_link

            target_pos = keypoints[:n_retarget, :]
            robot_pos = T_world_link.translation()[jnp.array(joint_retarget_indices)]

            delta_target = target_pos[:, None] - target_pos[None, :]
            delta_robot = robot_pos[:, None] - robot_pos[None, :]

            position_scale = var_values[var_joints_scale][..., None]
            residual_position_delta = (
                (delta_target - delta_robot * position_scale)
                * (1 - jnp.eye(delta_target.shape[0])[..., None])
                * retarget_mask[..., None]
            )

            delta_target_normalized = delta_target / jnp.linalg.norm(
                delta_target + 1e-6, axis=-1, keepdims=True
            )
            delta_robot_normalized = delta_robot / jnp.linalg.norm(
                delta_robot + 1e-6, axis=-1, keepdims=True
            )
            residual_angle_delta = 1 - (
                delta_target_normalized * delta_robot_normalized
            ).sum(axis=-1)
            residual_angle_delta = (
                residual_angle_delta
                * (1 - jnp.eye(residual_angle_delta.shape[0]))
                * retarget_mask
            )

            residual = (
                jnp.concatenate(
                    [
                        residual_position_delta.flatten(),
                        residual_angle_delta.flatten(),
                    ]
                )
                * weights["local_alignment"]
            )
            return residual

        @jaxls.Cost.create_factory
        def scale_regularization(
            var_values: jaxls.VarValues,
            var_joints_scale: SimplifiedJointsScaleVar,
        ) -> jax.Array:
            """Regularize the scale of the retargeted joints."""
            res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
            res_1 = (
                var_values[var_joints_scale] - var_values[var_joints_scale].T
            ).flatten() * 100.0
            res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
            return jnp.concatenate([res_0, res_1, res_2])

        @jaxls.Cost.create_factory
        def pc_alignment_cost(
            var_values: jaxls.VarValues,
            var_Ts_world_root: jaxls.SE3Var,
            var_robot_cfg: jaxls.Var[jnp.ndarray],
            var_joints_scale: SimplifiedJointsScaleVar,
            keypoints: jnp.ndarray,
        ) -> jax.Array:
            """Soft cost to align the target keypoints to the robot."""
            T_world_root = var_values[var_Ts_world_root]
            robot_cfg = var_values[var_robot_cfg]
            T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
            T_world_link = T_world_root @ T_root_link
            link_pos = T_world_link.translation()[joint_retarget_indices]

            link_pos_left_wrist = T_world_link.translation()[left_wrist_robot_idx]
            link_rot_mat_left_wrist = T_world_link.rotation().as_matrix()[
                left_wrist_robot_idx
            ]
            left_hand_aux_pos = (
                link_pos_left_wrist + link_rot_mat_left_wrist @ hand_aux_offset
            )

            link_pos_right_wrist = T_world_link.translation()[right_wrist_robot_idx]
            link_rot_mat_right_wrist = T_world_link.rotation().as_matrix()[
                right_wrist_robot_idx
            ]
            right_hand_aux_pos = (
                link_pos_right_wrist + link_rot_mat_right_wrist @ hand_aux_offset
            )

            link_pos_torso = T_world_link.translation()[torso_link_idx]
            link_rot_mat_torso = T_world_link.rotation().as_matrix()[torso_link_idx]
            torso_aux_pos = link_pos_torso + link_rot_mat_torso @ torso_aux_offset

            link_pos_with_aux = jnp.concatenate(
                [
                    link_pos,
                    left_hand_aux_pos[None, :],
                    right_hand_aux_pos[None, :],
                    torso_aux_pos[None, :],
                ],
                axis=0,
            )

            keypoint_pos = keypoints
            if keypoint_weight_indices.size:
                keypoint_pos = keypoint_pos.at[keypoint_weight_indices, :].set(
                    keypoint_pos[keypoint_weight_indices, :]
                    * keypoint_weight_multipliers[:, None]
                )
                link_pos_with_aux = link_pos_with_aux.at[
                    keypoint_weight_indices, :
                ].set(
                    link_pos_with_aux[keypoint_weight_indices, :]
                    * keypoint_weight_multipliers[:, None]
                )

            return (link_pos_with_aux - keypoint_pos).flatten() * weights[
                "global_alignment"
            ]

        @jaxls.Cost.create_factory
        def root_smoothness(
            var_values: jaxls.VarValues,
            var_Ts_world_root: jaxls.SE3Var,
            var_Ts_world_root_prev: jaxls.SE3Var,
        ) -> jax.Array:
            """Smoothness cost for the robot root pose."""
            return (
                var_values[var_Ts_world_root].inverse()
                @ var_values[var_Ts_world_root_prev]
            ).log().flatten() * weights["root_smoothness"]

        costs = [
            retargeting_cost(
                var_Ts_world_root,
                var_joints,
                var_joints_scale,
                target_keypoints,
            ),
            scale_regularization(var_joints_scale),
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                var_joints,
                100.0,
            ),
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                weights["joint_smoothness"],
            ),
            root_smoothness(
                jaxls.SE3Var(jnp.arange(1, timesteps)),
                jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
            ),
            pc_alignment_cost(
                var_Ts_world_root,
                var_joints,
                var_joints_scale,
                target_keypoints,
            ),
            joint_vel_limit_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                max_joint_velocity,
                subsample_factor / input_fps,
                weights["joint_vel_limit"],
            ),
        ]

        if rest_penalty_joint_indices.size:
            costs.append(
                pk.costs.rest_cost(
                    var_joints,
                    var_joints.default_factory()[None],
                    jnp.full(var_joints.default_factory().shape, 0.02)
                    .at[rest_penalty_joint_indices]
                    .set(weights["joint_rest_penalty"])[None],
                )
            )

        for t in range(1, timesteps):
            costs.append(
                foot_contact_cost(
                    jaxls.SE3Var(t),
                    jaxls.SE3Var(t - 1),
                    robot.joint_var_cls(t),
                    robot.joint_var_cls(t - 1),
                    robot,
                    left_foot_contact[t],
                    right_foot_contact[t],
                    joint_retarget_indices,
                    foot_indices,
                    weights["foot_contact"],
                )
            )

        for t in range(timesteps):
            costs.append(
                foot_tilt_cost(
                    jaxls.SE3Var(t),
                    robot.joint_var_cls(t),
                    robot,
                    left_foot_contact[t],
                    right_foot_contact[t],
                    joint_retarget_indices,
                    foot_indices,
                    weights["foot_tilt"],
                )
            )

        solution = (
            jaxls.LeastSquaresProblem(
                costs, [var_joints, var_Ts_world_root, var_joints_scale]
            )
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make(
                    [
                        var_joints,
                        var_Ts_world_root.with_value(root_init_values),
                        var_joints_scale,
                    ]
                ),
                termination=jaxls.TerminationConfig(
                    max_iterations=max_iterations
                ),
            )
        )

        return solution[var_Ts_world_root], solution[var_joints]

    def _solve_retargeting_jit(max_iterations: int):
        return jdc.jit(
            partial(_solve_retargeting_impl, max_iterations=max_iterations)
        )

    def solve_retargeting(
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision | None,
        target_keypoints: jnp.ndarray,
        target_orientations: jnp.ndarray,
        left_foot_contact: jnp.ndarray,
        right_foot_contact: jnp.ndarray,
        joint_retarget_indices: jnp.ndarray,
        retarget_mask: jnp.ndarray,
        source_names: tuple[str, ...],
        link_names: tuple[str, ...],
        config: PyrokiRetargetConfig,
        subsample_factor: int = 1,
        input_fps: float = 30.0,
    ) -> Tuple[jaxlie.SE3, jnp.ndarray]:
        weights: SolverWeights = config.weights.as_dict()  # type: ignore[assignment]
        foot_indices = jnp.array(
            [
                source_names.index("left_ankle"),
                source_names.index("right_ankle"),
                source_names.index("left_foot"),
                source_names.index("right_foot"),
            ],
            dtype=jnp.int32,
        )
        left_wrist_robot_idx = joint_retarget_indices[source_names.index("left_wrist")]
        right_wrist_robot_idx = joint_retarget_indices[
            source_names.index("right_wrist")
        ]
        torso_link_idx = link_names.index(config.torso_link_name)
        hand_aux_offset = jnp.array(config.hand_aux_offset)
        torso_aux_offset = jnp.array(config.torso_aux_offset)

        keypoint_labels = source_names + (
            "left_hand_aux",
            "right_hand_aux",
            "torso_aux",
        )
        keypoint_weight_indices = jnp.array(
            [
                keypoint_labels.index(label)
                for label in config.global_alignment_keypoint_weights
            ],
            dtype=jnp.int32,
        )
        keypoint_weight_multipliers = jnp.array(
            list(config.global_alignment_keypoint_weights.values())
        )
        rest_penalty_joint_indices = jnp.array(
            [
                robot.joints.actuated_names.index(name)
                for name in config.rest_penalty_joint_names
            ],
            dtype=jnp.int32,
        )

        return _solve_retargeting_jit(config.max_iterations)(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=target_keypoints,
            target_orientations=target_orientations,
            left_foot_contact=left_foot_contact,
            right_foot_contact=right_foot_contact,
            joint_retarget_indices=joint_retarget_indices,
            retarget_mask=retarget_mask,
            weights=weights,
            foot_indices=foot_indices,
            left_wrist_robot_idx=left_wrist_robot_idx,
            right_wrist_robot_idx=right_wrist_robot_idx,
            torso_link_idx=torso_link_idx,
            hand_aux_offset=hand_aux_offset,
            torso_aux_offset=torso_aux_offset,
            keypoint_weight_indices=keypoint_weight_indices,
            keypoint_weight_multipliers=keypoint_weight_multipliers,
            rest_penalty_joint_indices=rest_penalty_joint_indices,
            max_joint_velocity=config.max_joint_velocity,
            subsample_factor=subsample_factor,
            input_fps=input_fps,
        )

else:

    def joint_vel_limit_cost(*args, **kwargs):
        _require_solver_dependencies()

    def foot_contact_cost(*args, **kwargs):
        _require_solver_dependencies()

    def foot_tilt_cost(*args, **kwargs):
        _require_solver_dependencies()

    def _solve_retargeting_jit(max_iterations: int):
        _require_solver_dependencies()

    def solve_retargeting(
        robot,
        robot_coll,
        target_keypoints,
        target_orientations,
        left_foot_contact,
        right_foot_contact,
        joint_retarget_indices,
        retarget_mask,
        source_names,
        link_names,
        config,
        subsample_factor: int = 1,
        input_fps: float = 30.0,
    ):
        _require_solver_dependencies()
