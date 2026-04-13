# protomotions/robot_configs/

Per-robot configuration files. Each defines assets, control params, body mappings.

## FILES

| File | Robot | Notes |
|------|-------|-------|
| `base.py` | `RobotConfig` base | Abstract config with `KinematicInfo` extraction from MJCF |
| `factory.py` | Registry | Maps `--robot-name` CLI arg to config class |
| `g1.py` | Unitree G1 | 23-DOF humanoid, deploy-ready |
| `h1_2.py` | Unitree H1_2 | 19-DOF humanoid |
| `smpl.py` | SMPL | Parametric body model (animation character) |
| `smplx.py` | SMPL-X | Extended SMPL with hands/face |
| `soma23.py` | SOMA23 | SOMA skeleton (23 joints) |
| `rigv1.py` | RigV1 | Custom rig |
| `astro.py` | Astro | Apptronik robot |
| `amp.py` | AMP humanoid | Legacy AMP-style humanoid |

## KEY CONFIG FIELDS

- `common_naming_to_robot_body_names`: Semantic body mapping. Values **MUST be lists**, not strings.
- `control_info`: Per-DOF stiffness/damping/effort limits (extracted from MJCF).
- `simulation_params`: Per-simulator physics params (friction, solver iterations, etc.).
- `KinematicInfo`: Extracted at `__post_init__` via `pose_lib.extract_kinematic_info()`.

## ADDING A NEW ROBOT

1. Add MJCF `.xml` to `protomotions/data/robots/`.
2. Create config file here (copy `g1.py` as template).
3. Register in `factory.py`.
