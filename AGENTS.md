# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-13
**Commit:** 9456f08
**Branch:** develop

## OVERVIEW

GPU-accelerated RL framework for training physically simulated humanoids/robots. Multi-simulator (IsaacGym, IsaacLab, Newton, Genesis, MuJoCo), multi-algorithm (PPO, AMP, ASE, MaskedMimic). Python 3.8+, torch + Lightning Fabric + TensorDict.

## STRUCTURE

```
ProtoMotions/
├── protomotions/          # Core package — agents, envs, simulator, components, utils
├── examples/              # Experiment configs (Python, NOT scripts) + tutorials + benchmarks
├── deployment/            # ONNX export + MuJoCo deployment testing
├── pyroki/                # Embedded PyRoki retargeting library
├── data/                  # Motion files, pretrained models, conversion scripts
├── tools/                 # Standalone GUIs (retargeting, keypoint viz)
├── scripts/               # Shell scripts (retargeting, smoke test, video)
├── usd_convert/           # MJCF → USD conversion for IsaacLab
├── g1_deploy/             # G1 robot deployment patches (RoboJuDo)
├── docs/                  # Sphinx documentation source
├── Dockerfile.*           # Per-simulator Docker images (isaacgym, isaaclab, newton)
└── requirements_*.txt     # Per-simulator dependency files
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new RL algorithm | `protomotions/agents/` | Extend PPO or BaseAgent; see mimic/agent_add.py (~50 lines) |
| Add new observation | `protomotions/envs/obs/` | Pure tensor function + wire via MdpComponent in experiment config |
| Add new reward | `protomotions/envs/rewards/` | Pure function + MdpComponent factory in component_factories.py |
| Add new robot | `protomotions/robot_configs/` | Config file + register in factory.py + MJCF in data/robots/ |
| Add new simulator | `protomotions/simulator/` | Implement ~17 abstract methods from base_simulator/simulator.py |
| Add new task | `protomotions/envs/control/` | Control component + obs/reward functions + experiment config |
| Configure experiment | `examples/experiments/` | Python file with robot_factory, env_config, agent_config functions |
| Export ONNX | `deployment/` | export_bm_tracker_onnx.py; adapts per-config obs keys |
| Retarget motions | `pyroki/` | PyRoki-based, one-command AMASS retargeting |
| Convert data | `data/scripts/` | AMASS/SOMA/BVH/CSV → Proto format converters |
| Understand config flow | `protomotions/train_agent.py` | Top docstring explains full pipeline |
| Understand MdpComponent | `protomotions/envs/mdp_component.py` | Design docs in module docstring |
| Understand context paths | `protomotions/envs/context_views.py` | FieldPath descriptors, EnvContext |

## CONVENTIONS

- **Config system**: Python experiment files (not YAML). Configs pickled to `resolved_configs.pt`. Resume loads pickle directly — experiment file NOT re-executed.
- **`_target_` strings**: All configs use dynamic class instantiation (e.g., `_target_: "protomotions.agents.ppo.agent.PPO"`).
- **Models**: All `TensorDictModuleBase` subclasses. `nn.LazyLinear` for input shape inference.
- **MdpComponents**: Level 1 (pure tensor), Level 2 (aggregated, ONNX-safe), Level 3 (side effects).
- **Quaternions**: Common format = xyzw. IsaacGym/IsaacLab = wxyz (auto-converted).
- **Body/DOF ordering**: Per-simulator. Conversion tensors computed once in `_finalize_setup()`.
- **OOP for models**, functional for data processing pipelines.
- **License header**: Full Apache-2.0 on all new .py files (see CLAUDE.md for template).
- **Commits**: Require sign-off (`git commit -s`) per DCO.

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** run `pre-commit run --all-files` — causes 100+ unrelated changes. Use `--files <file1> ...`
- **DO NOT** fix F822 errors in `component_factories.py` `__all__` — pre-existing, known
- **DO NOT** use `weights_only=True` with `torch.load()` on `resolved_configs.pt` — it's pickle
- **DO NOT** expect CLI `--overrides` to work on resume — configs frozen from pickle
- **DO NOT** pass strings for `common_naming_to_robot_body_names` values — must be **lists**
- **DO NOT** import torch before simulators — IsaacGym/IsaacLab require import-first (see `simulator_imports.py`)

## COMMANDS

```bash
# Train
python protomotions/train_agent.py --robot-name g1 --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py --motion-file <path> --num-envs 4096

# Inference
python protomotions/inference_agent.py --checkpoint <ckpt> --motion-file <path> --simulator isaacgym

# Test
pytest protomotions/tests/

# Lint (targeted only)
pre-commit run --files <file1> <file2>

# ONNX export
python deployment/export_bm_tracker_onnx.py --checkpoint <ckpt>
```

## NOTES

- **Friction**: PhysX uses AVERAGE, Newton uses MAX. `convert_friction_for_simulator()` handles conversion.
- **Two-phase init**: Simulator constructor creates shell; `_initialize_with_markers()` allocates GPU after env provides markers.
- **Control modes**: BUILT_IN_PD (simulator PD), PROPORTIONAL (custom PD + action scaling), TORQUE (direct).
- **Newton**: Pinned to commit `e7a737c`, installed from source.
- **MuJoCo**: CPU-only, `num_envs=1` only. For lightweight testing.
- **Training overrides are PERMANENT** — saved to resolved_configs.pt. Use new experiment name for temporary changes.
- **CI**: Only docs deployment (GitHub Pages). No automated test pipeline. Docker images per simulator.
- **Tests**: Minimal — 1 pytest test + 3 standalone scripts. Most need GPU/simulator.
