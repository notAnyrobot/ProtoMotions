# protomotions/

Core Python package for ProtoMotions — GPU-accelerated RL framework for physically simulated humanoids and robots.

## Package Structure

```
protomotions/
├── agents/            # RL algorithms (PPO, AMP, ASE, MaskedMimic)
├── envs/              # MDP environment: observations, rewards, terminations, control
├── simulator/         # Multi-backend physics (IsaacGym, IsaacLab, Newton, Genesis, MuJoCo)
├── components/        # MotionLib, PoseLib, SceneLib, Terrain
├── robot_configs/     # Per-robot config files (G1, H1_2, Astro, SMPL, SOMA23, etc.)
├── utils/             # Shared utilities (rotations, config building, export)
├── data/              # Data loading (assets for checkerboard texture, etc.)
├── tests/             # Test suite
├── train_agent.py     # Main training entry point
├── inference_agent.py # Inference entry point
└── train_slurm.py     # SLURM distributed training
```

---

## Motion Data Format

ProtoMotions uses a two-level motion data hierarchy: individual **`.motion` files** (one per clip) and **packaged `.pt` files** (many clips concatenated for fast loading).

### Individual `.motion` File

A `.motion` file is a `torch.save()`-serialized dict produced by the conversion script (`data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py`). It contains one motion clip as a `RobotState` (see `simulator/base_simulator/simulator_state.py`).

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `rigid_body_pos` | `[F, B, 3]` | float32 | Global 3D positions of all rigid bodies. Body 0 = root. Computed via forward kinematics from joint angles. |
| `rigid_body_rot` | `[F, B, 4]` | float32 | Global rotations as quaternions (**xyzw** ordering). Computed via FK. |
| `rigid_body_vel` | `[F, B, 3]` | float32 | Linear velocities of all rigid bodies. Computed via finite-difference with multi-horizon minimum filtering. |
| `rigid_body_ang_vel` | `[F, B, 3]` | float32 | Angular velocities of all rigid bodies. Same computation as `rigid_body_vel`. |
| `dof_pos` | `[F, D]` | float32 | Joint positions (angle-axis or exp-map depending on joint type). Re-extracted from FK transforms to ensure `[-pi, pi]` range. |
| `dof_vel` | `[F, D]` | float32 | Joint velocities. Computed via `compute_cartesian_velocity()` on joint angles. |
| `rigid_body_contacts` | `[F, B]` | bool | Per-body binary contact labels. Typically only foot bodies have `True` entries. Sourced from the original SMPL motion (Step 3 of the pipeline), not re-computed from retargeted data. |
| `fps` | scalar | int | Frames per second (typically 30). |
| `motion_dt` | scalar | float | `1 / fps` — time step between consecutive frames. |
| `motion_num_frames` | scalar | int | Number of frames `F`. |
| `motion_length` | scalar | float | `(F - 1) * motion_dt` — total clip duration in seconds. |
| `state_conversion` | enum | — | Always `StateConversion.COMMON` (simulator-agnostic ordering). |

> **Dimension key:** `F` = num_frames, `B` = num_bodies, `D` = num_dofs. Exact values depend on the robot (e.g., G1: B=21, D=37; Astro: B=26, D=25).

### Packaged `.pt` MotionLib File

The packaging step (`protomotions/components/motion_lib.py --motion-path <dir> --output-file <out.pt>`) concatenates all `.motion` files in a directory into a single `.pt` file for efficient loading. Frames from all clips are stacked along dimension 0, with indexing metadata to recover per-clip boundaries.

| Field | Shape | Description |
|---|---|---|
| `gts` | `[N, B, 3]` | Concatenated `rigid_body_pos` from all motions |
| `grs` | `[N, B, 4]` | Concatenated `rigid_body_rot` (xyzw quaternions) |
| `gvs` | `[N, B, 3]` | Concatenated `rigid_body_vel` |
| `gavs` | `[N, B, 3]` | Concatenated `rigid_body_ang_vel` |
| `dps` | `[N, D]` | Concatenated `dof_pos` |
| `dvs` | `[N, D]` | Concatenated `dof_vel` |
| `contacts` | `[N, B]` | Concatenated `rigid_body_contacts` (bool or float after smoothing) |
| `length_starts` | `[M]` | Cumulative frame offset for each motion (`length_starts[i]` = first frame index of motion `i`) |
| `motion_lengths` | `[M]` | Duration of each motion in seconds |
| `motion_dt` | `[M]` | Time step for each motion |
| `motion_num_frames` | `[M]` | Frame count for each motion |
| `motion_weights` | `[M]` | Sampling weight for each motion (default 1.0 for all) |
| `motion_files` | tuple of str | Original `.motion` file paths (for provenance) |
| `lrs` *(optional)* | `[N, B, 4]` | Local rigid body rotations, if present — used for SLERP interpolation of `dof_pos` during blending |

> **Dimension key:** `N` = total frames across all motions, `B` = num_bodies, `D` = num_dofs, `M` = num_motions.

**Field name mapping** (packaged `.pt` field → per-frame `RobotState` field):

```
gts  → rigid_body_pos        grs  → rigid_body_rot
gvs  → rigid_body_vel        gavs → rigid_body_ang_vel
dps  → dof_pos                dvs  → dof_vel
```

### Accessing Motion Data at Runtime

The `MotionLib` class loads a `.pt` file and provides O(1) indexed access to any frame of any motion:

```python
# Retrieve interpolated reference state for given motion IDs and times
motion_state: RobotState = motion_lib.get_motion_state(motion_ids, motion_times)
```

Frame lookup uses `length_starts[motion_id] + frame_index` to index into the concatenated tensors. Between-frame blending uses **linear interpolation** for positions/velocities and **SLERP** for quaternion rotations.

---

## How Motion Data Drives Training

In motion tracking / imitation learning (mimic) tasks, the reference motion data serves as the ground-truth target that the RL policy learns to reproduce. The data flows through three stages:

### 1. Motion Sampling & Time Progression

**`MotionManager`** (`envs/motion_manager/`) assigns each parallel environment a motion clip and tracks playback time:

- On environment reset: samples a new `motion_id` (weighted random from the library) and a random start `motion_time` within the clip.
- Each simulation step: advances `motion_time += env_dt`.
- When a clip ends: marks the environment for reset (or resamples, depending on `resample_on_reset` config).

The manager calls `MotionLib.get_motion_state(motion_ids, motion_times)` to produce the **reference `RobotState`** — the pose/velocity the robot *should* be in at this moment.

### 2. Observations — What the Policy Sees

Observation functions in `envs/obs/` compare the **current robot state** (from the simulator) to the **reference motion state** (from MotionLib):

| Observation Function | File | What It Computes |
|---|---|---|
| `build_max_coords_target_poses_future_rel` | `obs/target_poses.py` | Relative position/rotation deltas between current and future reference frames (multi-timestep lookahead) |
| `compute_target_poses_only` | `obs/masked_mimic.py` | Sparse masked reference targets for generative policies (MaskedMimic) |

These observations give the policy a "preview" of where the reference motion is heading, enabling anticipatory control.

### 3. Rewards — How the Policy Is Scored

Reward functions in `envs/rewards/tracking.py` measure how closely the robot matches the reference:

| Reward Function | Reference Field Used | What It Measures |
|---|---|---|
| `compute_gt_rew` | `rigid_body_pos` (gts) | Global position tracking error per body |
| `compute_gr_rew` | `rigid_body_rot` (grs) | Global rotation tracking error per body |
| `compute_gv_rew` | `rigid_body_vel` (gvs) | Linear velocity tracking |
| `compute_gav_rew` | `rigid_body_ang_vel` (gavs) | Angular velocity tracking |
| `compute_rh_rew` | `rigid_body_pos` (gts) | Root height tracking |
| `compute_relative_body_pos_rew` | `rigid_body_pos` (gts) | Relative body positions (root-local frame) |
| `compute_relative_body_ori_rew` | `rigid_body_rot` (grs) | Relative body orientations (root-local frame) |

All tracking rewards use `exp(-k * error)` form, where `k` controls sensitivity. The total reward is a weighted sum configured per experiment.

### 4. Terminations — When to Reset

Termination conditions in `envs/terminations/tracking.py` end an episode when tracking error exceeds tolerance:

| Termination Function | Reference Field Used | Condition |
|---|---|---|
| `mean_body_pos_error` | `rigid_body_pos` | Mean position error across all bodies exceeds threshold |
| `max_body_pos_error` | `rigid_body_pos` | Any single body exceeds position error threshold |
| `mean_body_rot_error` | `rigid_body_rot` | Mean rotation error across all bodies exceeds threshold |
| `motion_clip_done` | `motion_lengths` | Motion playback reached end of clip |

### 5. Contact Labels — Foot Contact Rewards

When `rigid_body_contacts` is present in the motion data, additional reward/observation components can use it to:

- Reward the robot for matching the reference foot contact pattern (feet on ground when the reference says they should be).
- Penalize foot sliding during contact phases.
- Provide contact state as an observation to the policy.

Contact labels are sourced from the original SMPL motion (not the retargeted robot), making them more reliable. They can be optionally smoothed via `MotionLib.smooth_contacts(window_size)` to produce soft contact probabilities in `[0, 1]` instead of hard binary labels.

> **Note:** If all contact labels in a loaded `.pt` file are zero (indicating missing/broken contact data), MotionLib discards them entirely and logs a warning. Any component that reads contacts will then raise an error, preventing silent training on meaningless data.

---

## Data Pipeline Summary

```
AMASS (.npz)
  │
  ▼ data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py
Keypoints (.npy)
  │
  ├──────────────────────────────────────┐
  ▼                                      ▼
Retargeted joints (.npz)           Contact labels (.npz)
  pyroki/batch_retarget.py           pyroki/batch_retarget.py --save-contacts-only
  │                                      │
  └──────────────┬───────────────────────┘
                 ▼
  data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py
                 │
                 ▼
          .motion files  (per-clip RobotState dicts)
                 │
                 ▼  protomotions/components/motion_lib.py
          Packaged .pt  (concatenated MotionLib)
                 │
                 ▼  MotionLib.get_motion_state()
          Reference RobotState at runtime
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
 Observations  Rewards  Terminations
```

---

## Further Reading

- [Official Retargeting Tutorial](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/retargeting_pyroki.html)
- [AMASS Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/amass_preparation.html)
- [Architecture Overview](https://nvlabs.github.io/ProtoMotions/concepts/architecture.html)
- [`tools/README.md`](../tools/README.md) — CLI quick reference for the retargeting pipeline
