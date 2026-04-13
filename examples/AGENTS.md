# examples/

Experiment configurations, tutorials, and benchmarks. NOT runnable scripts — loaded by train_agent.py.

## STRUCTURE

```
examples/
├── experiments/          # Python config files defining full experiments
│   ├── format.py         # Documents required function signatures
│   ├── mimic/            # Motion tracking task configs (mlp.py, etc.)
│   ├── add/              # ADD algorithm configs
│   ├── amp/              # AMP configs
│   ├── ase/              # ASE configs
│   ├── masked_mimic/     # MaskedMimic distillation configs
│   ├── steering/         # Locomotion steering task configs
│   └── path_follower/    # Path following task configs
├── tutorial/             # Step-by-step learning examples
├── benchmark/            # Performance benchmarking configs
├── data/                 # Tutorial-specific data files
├── motion_libs_visualizer.py  # Standalone: visualize motion libraries
├── random_pose_visualizer.py  # Standalone: visualize random poses
└── env_kinematic_playback.py  # Standalone: kinematic motion playback
```

## EXPERIMENT CONFIG STRUCTURE

Each experiment file exports 3-4 functions (see `format.py` for signatures):

1. `robot_factory()` / `simulator_factory()` — create base configs.
2. `configure_robot_and_simulator()` — customize for this experiment.
3. `env_config()` — build env config with MdpComponents (obs, rewards, terminations).
4. `agent_config()` — build agent config.

## HOW CONFIGS WIRE COMPONENTS

- Common obs/rewards: use factories from `envs/component_factories.py` (e.g., `max_coords_obs_factory`).
- Task-specific: define `MdpComponent(compute_func=..., dynamic_vars={...})` directly.
- Controls: pass control component instances in `env_config()`.
- All context binding via `EnvContext` paths.

## CONVENTIONS

- Experiment files are **config files**, not scripts. They're imported by `train_agent.py`.
- Configs are pickled to `resolved_configs.pt`. Resume loads pickle — experiment file NOT re-executed.
- Overrides via CLI `--overrides` are PERMANENT (saved to pickle).
