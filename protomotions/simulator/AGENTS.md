# protomotions/simulator/

Multi-backend physics simulator abstraction. 5 backends behind one interface.

## ABSTRACT INTERFACE

`Simulator` (base_simulator/simulator.py) defines ~17 abstract methods:
- Environment lifecycle: `_create_envs()`, `_finalize_setup()`, `_initialize_with_markers()`
- Step: `step()`, `_apply_control()`, `_refresh_sim_tensors()`
- State: `get_robot_state()`, `set_robot_state()`, `reset_envs()`
- Objects: `create_object()`, `set_object_state()`
- Terrain: `create_terrain()`
- Viz: `create_marker()`, `set_marker_state()`

## BACKENDS

| Backend | Quat | Friction | Terrain | GPU | Notes |
|---------|------|----------|---------|-----|-------|
| IsaacGym | wxyz (auto-convert) | AVERAGE | Triangle mesh | Yes | Preview 4, legacy |
| IsaacLab | xyzw | AVERAGE, per-body | Height field | Yes | IsaacSim 5.0+ |
| Newton | xyzw | MAX, single coeff | Ground plane | Yes | Pinned commit e7a737c |
| Genesis | xyzw | Basic | Plane/height field | Yes | Community-contributed |
| MuJoCo | xyzw | N/A | Plane | **CPU only** | `num_envs=1` only |

## STATE CONVERSION

`simulator_state.py` defines: `RobotState` (full FK), `ResetState` (root+DOF), `ObjectState`, `RootOnlyState`.
Each has `.convert(StateConversion.TO_COMMON)` / `.convert(StateConversion.TO_SIMULATOR)`.
Conversion tensors (`body_convert_to_common`, `dof_convert_to_sim`) computed once in `_finalize_setup()`.

## TWO-PHASE INIT

1. Constructor → creates shell (configs, no GPU).
2. `_initialize_with_markers()` → allocates GPU after env provides visualization markers.

## CONTROL MODES

- `BUILT_IN_PD`: Simulator-native PD controller.
- `PROPORTIONAL`: Custom PD with action scaling. Action = target position offset.
- `TORQUE`: Direct torque application.

## WHERE TO LOOK

| Task | File |
|------|------|
| Add new simulator | Implement `Simulator` in new subdir; register in `factory.py` |
| Understand state exchange | `base_simulator/simulator_state.py` |
| Simulator factory | `factory.py` — creates config, handles sim2sim inference switching |
| IsaacLab-specific | `isaaclab/utils/` — scene/USD handling helpers |
