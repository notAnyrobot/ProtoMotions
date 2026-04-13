# protomotions/envs/obs/

Observation compute kernels. Pure tensor functions — no side effects, no env state mutation.

## FILES

| File | What it computes |
|------|-----------------|
| `humanoid.py` | Core humanoid state: max_coords (global), reduced_coords (local). DOF pos/vel. |
| `humanoid_historical.py` | Historical state buffer observations. Past N frames of humanoid state. |
| `target_poses.py` | Reference motion targets for mimic tasks. Multi-horizon future poses. |
| `masked_mimic.py` | Sparse conditioning observations for MaskedMimic (masked body targets). |
| `steering.py` | Steering task: target direction/speed/facing in robot-local frame → 5D vector. |
| `path.py` | Path follower: upcoming waypoints in local frame. |
| `scene_obs.py` | Scene object observations (positions, orientations relative to robot). |
| `terrain_obs.py` | Terrain height samples around the robot. |
| `prior.py` | Prior/latent observations for ASE-style algorithms. |
| `observation_noise.py` | Noise injection for domain randomization. |
| `state_history_buffer.py` | Ring buffer for historical state tracking. |
| `utils.py` | Shared observation utilities. |

## CONVENTIONS

- Every function is pure: `f(tensor_args..., static_params...) → tensor`.
- Functions are wired to env state via `MdpComponent(compute_func=fn, dynamic_vars={...})`.
- Factory functions in `../component_factories.py` pre-configure common MdpComponents.
- Observation keys in the TensorDict must match what the agent model expects.
