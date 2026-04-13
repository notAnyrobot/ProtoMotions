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

## CONVENTIONS

- Entry points use `simulator_imports.ensure_simulator_imported()` before any other imports.
- Config pipeline: experiment file → robot_factory() → env_config() → agent_config() → CLI overrides → pickle.
- All models inherit `TensorDictModuleBase`. Forward passes read/write shared `TensorDict`.
