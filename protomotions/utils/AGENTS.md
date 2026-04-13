# protomotions/utils/

Shared utilities. 13 modules covering config, math, import order, export.

## FILES

| File | Role |
|------|------|
| `simulator_imports.py` | **CRITICAL**: Ensures IsaacGym/IsaacLab imported before torch. Called by entry points. |
| `component_builder.py` | Factory functions building terrain, scene_lib, motion_lib, simulator from configs. |
| `config_builder.py` | Config construction pipeline (experiment file → robot → env → agent → pickle). |
| `config_utils.py` | Config merging, override application, serialization. |
| `hydra_replacement.py` | Lightweight `_target_` instantiation system (replaces Hydra dependency). |
| `rotations.py` | Quaternion math: conversions, SLERP, matrix ops. ~900 lines. Default: w_last=True (xyzw). |
| `export_utils.py` | ONNX export: model tracing, obs baking, isaac-deploy YAML generation. ~1400 lines. |
| `fabric_config.py` | Lightning Fabric configuration helpers. |
| `inference_utils.py` | Checkpoint loading, inference pipeline setup. |
| `mesh_utils.py` | Mesh processing: bounding boxes, trimesh integration. |
| `motion_interpolation_utils.py` | Position/quaternion interpolation for motion data. |
| `torch_utils.py` | PyTorch utilities. |

## ANTI-PATTERNS

- **DO NOT** import torch before calling `simulator_imports.ensure_simulator_imported()`.
- **DO NOT** use `weights_only=True` when loading `resolved_configs.pt` — it's pickle.
