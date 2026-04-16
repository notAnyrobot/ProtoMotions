# protomotions/

Core package. 244 source .py files, ~78K lines.

## STRUCTURE

```
protomotions/
├── agents/           # RL algorithm implementations (PPO → AMP → ASE, MaskedMimic)
├── envs/             # Environment: MdpComponents, observations, rewards, terminations, control
├── simulator/        # Multi-backend physics abstraction (5 simulators)
├── components/       # MotionLib, PoseLib, SceneLib, Terrain
├── robot_configs/    # Per-robot config files (G1, H1_2, SMPL, SOMA23, etc.)
├── utils/            # Shared utilities (rotations, config building, export)
├── data/             # Data loading (assets subdir for checkerboard texture)
├── tests/            # Minimal test suite (1 pytest + 3 standalone scripts)
├── train_agent.py    # Main training entry point
├── inference_agent.py # Inference entry point
└── train_slurm.py    # SLURM distributed training
```

## WHERE TO LOOK

| Task | Start here |
|------|-----------|
| Training pipeline | `train_agent.py` — config build → env create → agent.fit() |
| Inference pipeline | `inference_agent.py` — loads resolved_configs_inference.pt |
| Config construction | `utils/config_builder.py` + `utils/hydra_replacement.py` |
| Component factory | `utils/component_builder.py` — builds terrain, scene, motion_lib, simulator |

## MOTION DATA FORMAT

See [`protomotions/README.md`](README.md) for full documentation with shapes, types, and downstream usage.

### Packaged `.pt` (MotionLib) — Tensor Fields

All frames from all clips concatenated along dim 0. `length_starts[i]` gives the first frame index of motion `i`.

| Field | Shape | Maps to RobotState field |
|-------|-------|--------------------------|
| `gts` | `[N, B, 3]` | `rigid_body_pos` — global body positions |
| `grs` | `[N, B, 4]` | `rigid_body_rot` — global body rotations (xyzw quat) |
| `gvs` | `[N, B, 3]` | `rigid_body_vel` — body linear velocities |
| `gavs` | `[N, B, 3]` | `rigid_body_ang_vel` — body angular velocities |
| `dps` | `[N, D]` | `dof_pos` — joint positions |
| `dvs` | `[N, D]` | `dof_vel` — joint velocities |
| `contacts` | `[N, B]` | `rigid_body_contacts` — foot contact labels |

Metadata: `length_starts [M]`, `motion_lengths [M]`, `motion_dt [M]`, `motion_num_frames [M]`, `motion_weights [M]`, `motion_files (tuple)`.

N = total frames, B = num_bodies, D = num_dofs, M = num_motions.

### Data Flow in Training

```
MotionLib.get_motion_state(ids, times) → RobotState
  ├→ obs/target_poses.py     (policy input: future reference deltas)
  ├→ rewards/tracking.py     (compute_gt_rew, compute_gr_rew, etc.)
  └→ terminations/tracking.py (mean_body_pos_error, motion_clip_done, etc.)
```

## CONVENTIONS

- Entry points use `simulator_imports.ensure_simulator_imported()` before any other imports.
- Config pipeline: experiment file → robot_factory() → env_config() → agent_config() → CLI overrides → pickle.
- All models inherit `TensorDictModuleBase`. Forward passes read/write shared `TensorDict`.
