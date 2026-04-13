# protomotions/envs/control/

Task control components. Manage stateful task logic — motion sampling, target generation, state progression.

## COMPONENTS

| File | Task | Role |
|------|------|------|
| `mimic_control.py` | Motion tracking | Samples reference motions, advances time, provides target state |
| `steering_control.py` | Locomotion | Periodically samples heading/speed targets, manages facing direction |
| `path_follower_control.py` | Path following | Generates paths via PathGenerator, provides upcoming waypoints |
| `masked_mimic_control.py` | Generative policy | Extends MimicControl with sparse body conditioning for MaskedMimic |
| `kinematic_replay_control.py` | Visualization | Plays motions without physics — for debugging/rendering |
| `external_kinematic_control.py` | External control | Receives targets from external source |
| `base.py` | Abstract base | `BaseControl` interface: `step()`, `reset()`, `populate_context()` |
| `manager.py` | Orchestrator | `ControlManager` — initializes and steps all active control components |

## HOW THEY CONNECT

1. ControlManager is created by BaseEnv with experiment-specified controls.
2. Each control component populates its own EnvContext view (e.g., `ctx.mimic`, `ctx.steering`).
3. Observation/reward MdpComponents bind to these context paths to read task targets.
4. On `step()`, controls advance their state (next motion frame, new heading target, etc.).
5. On `reset()`, controls re-sample for done environments.

## ADDING A NEW TASK

1. Create `my_task_control.py` extending `BaseControl`.
2. Add a context view class in `context_views.py` (e.g., `MyTaskView`).
3. Write obs/reward functions in `envs/obs/` and `envs/rewards/`.
4. Wire in experiment config: `control_components=[MyTaskControl(...)]`.
