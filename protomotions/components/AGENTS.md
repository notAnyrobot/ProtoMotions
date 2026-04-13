# protomotions/components/

Core simulation components. Heavy data structures for motions, kinematics, scenes, terrain.

## FILES

| File | Class | Role | Lines |
|------|-------|------|-------|
| `motion_lib.py` | `MotionLib` | Reference motion storage/sampling. Concatenated tensors (gts, grs, gvs, gavs, dps, dvs) with `length_starts` for O(1) indexing. SLERP interpolation. Distributed `.slurmrank.pt` loading. | ~800 |
| `pose_lib.py` | `PoseLib` | Batched FK from MJCF. Multi-horizon velocity estimation. Auto region weight discovery (leaf→root limb tracing). | ~1800 |
| `scene_lib.py` | `SceneLib` | Object management (mesh/box/sphere/cylinder). Pointcloud sampling via trimesh. Static + dynamic object support. | ~2200 |
| `terrains/` | `Terrain` | Procedural height field generation. Curriculum levels. Flat "object playground" region. | ~7 files |

## TERRAIN SUBDIR

| File | Role |
|------|------|
| `terrain.py` | Main Terrain class — height field grid, curriculum |
| `subterrain.py` | Subterrain types: slope, stairs, stepping stones, etc. |
| `subterrain_generator.py` | Generates subterrain patches |
| `config.py` | TerrainConfig |
| `shape_utils.py` | Geometric shape utilities |
| `terrain_utils.py` | Height field manipulation |

## GOTCHAS

- MotionLib requires FPS for motion data. Missing FPS → wrong interpolation.
- PoseLib assumes free root joint in MJCF.
- SceneLib needs consistent object counts per scene.
