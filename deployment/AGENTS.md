# deployment/

ONNX export and deployment testing for trained policies.

## FILES

| File | Role |
|------|------|
| `export_bm_tracker_onnx.py` | Exports BeyondMimic tracker to ONNX. Auto-detects actor obs keys from checkpoint. |
| `test_tracker_mujoco.py` | Tests exported ONNX policy in MuJoCo (CPU). ~870 lines. |
| `motion_utils.py` | Motion processing utilities for deployment. |
| `state_utils.py` | State handling utilities for deployment. |

## WORKFLOW

1. Train in IsaacGym/Newton → checkpoint saved.
2. `export_bm_tracker_onnx.py --checkpoint <ckpt>` → ONNX model with obs computation baked in.
3. `test_tracker_mujoco.py` → validate ONNX policy in MuJoCo.
4. Deploy ONNX to real robot (e.g., G1 via RoboJuDo — see `g1_deploy/`).

## NOTES

- ONNX export bakes observation computation into the model — deployment only needs raw sensor signals.
- For non-BeyondMimic configs, copy and adapt `export_bm_tracker_onnx.py` to match your obs keys.
- See `protomotions/utils/export_utils.py` for the underlying ONNX tracing utilities.
