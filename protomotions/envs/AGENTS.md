# protomotions/envs/

MDP environment system. Modular component architecture separating compute from state.

## CORE ABSTRACTIONS

1. **MdpComponent** (`mdp_component.py`) — binds a pure tensor function to context paths. Three levels: L1 (pure tensor), L2 (aggregated, ONNX-safe), L3 (side effects).
2. **ComponentManager** (`component_manager.py`) — executes MdpComponents with `torch.compile` caching.
3. **FieldPath/EnvContext** (`context_views.py`, `context_paths.py`) — type-safe descriptors. Class-level = path string, instance-level = tensor. Views: `CurrentStateView`, `HistoricalView`, task-specific views (mimic, steering, etc.).
4. **BaseEnv** (`base_env/env.py`) — orchestrates simulator + ComponentManager + ControlManager.

## STEP FLOW

```
BaseEnv.step(actions):
  1. Action processing (PD control, clamping) → ComponentManager
  2. simulator.step(actions) → physics substeps
  3. post_physics_step() → update robot state, context
  4. Control components step (motion tracking, steering, etc.)
  5. Observations → ComponentManager (MdpComponents)
  6. Rewards → ComponentManager (MdpComponents)
  7. Terminations → ComponentManager (MdpComponents)
  8. Reset done environments
```

## STRUCTURE

```
envs/
├── base_env/           # BaseEnv, config, state management
├── mdp_component.py    # MdpComponent class
├── component_manager.py # Executes MdpComponents
├── context_views.py    # EnvContext, typed views, FieldPath descriptors
├── context_paths.py    # FieldPath/NestedField descriptor implementations
├── component_factories.py # 50+ factory functions creating pre-configured MdpComponents
├── control/            # Task control components (mimic, steering, path, etc.)
├── obs/                # Observation compute kernels (pure functions)
├── rewards/            # Reward functions (tracking, regularization, task)
├── terminations/       # Termination conditions (tracking, base, task)
├── action/             # Action processing (PD control, transforms)
├── motion_manager/     # Motion sampling and time progression
└── utils/              # Path generator, scene utilities
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new observation | `obs/` + `component_factories.py` | Pure function → factory → experiment config |
| Add new reward | `rewards/` + `component_factories.py` | Same pattern as observations |
| Add new task | `control/` + obs/rewards for it | See steering_control.py as template |
| Wire components | `component_factories.py` | Factories bind compute_func + context paths |
| Understand context | `context_views.py` | FieldPath = class→path, instance→tensor |

## ANTI-PATTERNS

- **DO NOT** fix F822 errors in `component_factories.py` `__all__` — pre-existing, known.
- **DO NOT** add side effects in L1/L2 MdpComponents — breaks ONNX export.
