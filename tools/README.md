# ProtoMotions Retargeting GUIs

Graphical front-ends for the motion retargeting pipeline. The architecture is split into two specialized applications to match the separate Python environments required for ProtoMotions and PyRoki.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/UI-tkinter-green)

## Quick Start

### 1. ProtoMotions GUI (Steps 1, 4, 5)
Handles keypoint extraction, motion visualization, and final packaging. This application features **6 tabs** covering the full ProtoMotions workflow, from extraction to training-ready packaging.
```bash
python tools/protomotions_retarget_gui.py
```

### 2. PyRoki GUI (Steps 2, 3)
Handles the GPU-accelerated optimization and contact extraction. This application features **3 tabs** focused on the PyRoki retargeting engine and keypoint mapping.
```bash
python tools/pyroki_retarget_gui.py
```

---

## Environment & Configuration Shared State

Both GUIs share the same hidden configuration file: **`tools/.retargeting_gui_config.json`**. 

This file stores:
- Paths to the `env_isaaclab` and `pyroki-cuda` Python executables.
- Recently used motion files and output directories.

Changes made to environment paths in one GUI will automatically reflect in the other, ensuring a seamless transition between pipeline stages.

| GUI | Environment | Purpose |
|---|---|---|
| **ProtoMotions** | `env_isaaclab` | Step 1 (Extract), Step 4 (Convert), Step 5 (Package), Visualization |
| **PyRoki** | `pyroki-cuda` | Step 2 (Retarget), Step 3 (Contacts), Keypoint Tuning |

---

## The 5-Step Pipeline

1. **Extract Keypoints** (Proto GUI) — SMPL/AMASS → Keypoint tensors.
2. **PyRoki Retarget** (PyRoki GUI) — Keypoints → Robot joint angles (IK).
3. **Extract Contacts** (PyRoki GUI) — Physics-based foot contact labeling.
4. **Convert to Proto** (Proto GUI) — Joint angles → `.motion` files.
5. **Package MotionLib** (Proto GUI) — Directory of motions → Training `.pt`.

---

## Features & GUIs

### ProtoMotions Retargeting GUI (`protomotions_retarget_gui.py`)

- **Batch / Single Motion** — Automated pipelines for Steps 1, 4, and 5.
- **Visualize** — Preview retargeted motions in IsaacLab or IsaacGym.
- **USD Conversion** — Convert MuJoCo MJCF to USD for IsaacLab.
- **Joint Order** — Compare URDF vs MJCF joint mapping to prevent corruption.

### PyRoki Retargeting GUI (`pyroki_retarget_gui.py`)

- **Batch Retarget** — High-throughput Steps 2 and 3.
- **Step-by-Step** — Fine-grained control over optimization weights (9 sliders).
- **Keypoint Config** — Canonical location for tuning human-to-robot mappings.
  - **Launch Tuner** — Opens the interactive MuJoCo-based mapping tuner.
  - **Config Editor** — Direct YAML editing with template scaffolding.

---

## Interactive Keypoint Tuning

The canonical tool for adjusting how human joints map to robot links is:
`tools/visualize_keypoint_mapping_gui.py`

Launch it via the **Keypoint Config** tab in the PyRoki GUI to ensure it uses the correct PyRoki/JAX environment.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run from the project root: `python tools/protomotions_retarget_gui.py` |
| GUI freezes | Subprocesses run in background threads; check the log panel for status |
| Wrong Python | Verify paths in the **Environment Panel** at the top of each GUI |

---

## File Structure

```
tools/
├── README.md                          ← this file
├── protomotions_retarget_gui.py       ← Main ProtoMotions interface
├── pyroki_retarget_gui.py             ← Main PyRoki interface
├── retarget_gui_common.py             ← Shared GUI logic and styling
├── visualize_keypoint_mapping_gui.py  ← Interactive mapping tuner
└── .retargeting_gui_config.json       ← Auto-saved environment paths
```

---

## CLI Quick Reference

Below are the individual pipeline commands for the **Astro** robot using AMASS SMPL data. Adapt `--robot-type`, `--source-type`, and paths for other robots (g1, h1_2) or skeleton formats (rigv1).

> **Tip:** For a fully automated one-shot pipeline, use the convenience script instead:
> ```bash
> ./scripts/retarget_amass_to_robot.sh \
>     ~/miniconda3/envs/env_isaaclab/bin/python \
>     ~/miniconda3/envs/pyroki-cuda/bin/python \
>     ~/Data/smpl/amass_motionlib/proto-smpl/amass_smpl_train.pt \
>     astro 1
> ```
> For a single `.motion` file see `./scripts/retarget_single_motion_to_robot.sh`.

### Step 1 — Extract Keypoints from Packaged MotionLib

**Env:** `env_isaaclab`

```bash
python data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    ~/Data/smpl/amass_motionlib/proto-smpl/amass_smpl_train.pt \
    --output-path ~/Data/smpl/amass_motionlib/keypoints-for-retarget/train/ \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq 1
```

| Flag | Default | Description |
|---|---|---|
| `<positional>` | *(required)* | Path to the packaged `.pt` MotionLib |
| `--output-path` | auto-suffix near input | Directory for extracted keypoint `.npy` files |
| `--skeleton-format` | `rigv1` | Source skeleton: `smpl` or `rigv1` |
| `--start-idx` | `1` | First motion index to process |
| `--end-idx` | *(all)* | Last motion index to process |
| `--skip-freq` | `35` | Process every Nth motion (use `1` for all, `50` for quick test) |
| `--force-remake` | `False` | Overwrite existing keypoint files |

### Step 2 — PyRoki Batch Retargeting

**Env:** `pyroki-cuda`

```bash
python pyroki/batch_retarget.py \
    --robot-config pyroki/robot_configs/astro.yaml \
    --keypoints-folder-path ~/Data/smpl/amass_motionlib/keypoints-for-retarget/train/ \
    --output-dir ~/Data/smpl/amass_motionlib/pyroki-retargeted-astro/train/ \
    --source-type smpl \
    --subsample-factor 1 \
    --no-visualize \
    --skip-existing
```

| Flag | Description |
|---|---|
| `--robot-config` | Path to robot YAML config (or use `--robot astro` shorthand) |
| `--keypoints-folder-path` | Directory with `.npy` keypoint files from Step 1 |
| `--output-dir` | Directory for retargeted `.npz` motion files |
| `--source-type` | Source skeleton type: `smpl` or `rigv1` |
| `--subsample-factor` | Temporal subsampling (1 = full rate) |
| `--no-visualize` | Skip visualization (required for batch runs) |
| `--skip-existing` | Resume interrupted runs |

> **Note:** The per-robot shims (`batch_retarget_to_astro_from_keypoints.py`, etc.) are deprecated. Use `batch_retarget.py --robot-config` instead.

### Step 3 — Extract Contact Labels from Source Motion

**Env:** `pyroki-cuda`

Runs the *same* retarget script with `--save-contacts-only` to extract foot contact labels from the **source** SMPL keypoints (more reliable than re-computing from retargeted motions).

```bash
python pyroki/batch_retarget.py \
    --robot-config pyroki/robot_configs/astro.yaml \
    --keypoints-folder-path ~/Data/smpl/amass_motionlib/keypoints-for-retarget/train/ \
    --source-type smpl \
    --subsample-factor 1 \
    --save-contacts-only \
    --contacts-dir ~/Data/smpl/amass_motionlib/contacts/train/
```

### Step 4 — Convert Retargeted Motions to Proto Format

**Env:** `env_isaaclab`

```bash
python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir ~/Data/smpl/amass_motionlib/pyroki-retargeted-astro/train/ \
    --output-dir ~/Data/smpl/amass_motionlib/proto-astro/train \
    --robot-type astro \
    --contact-labels-dir ~/Data/smpl/amass_motionlib/contacts/train/ \
    --apply-motion-filter \
    --force-remake
```

| Flag | Default | Description |
|---|---|---|
| `--retargeted-motion-dir` | *(required)* | Directory with retargeted `.npz` files |
| `--output-dir` | *(required)* | Output directory for `.motion` files |
| `--robot-type` | `g1` | Target robot (`g1`, `h1_2`, `astro`) |
| `--contact-labels-dir` | *(none)* | Directory with `*_contacts.npz` files from Step 3 |
| `--apply-motion-filter` | `False` | Apply smoothing filter to reduce jitter |
| `--force-remake` | `False` | Overwrite existing files |
| `--input-fps` | `30` | Input motion FPS |
| `--output-fps` | `30` | Output motion FPS |
| `--min-height-threshold` | `-0.05` | Min height for motion quality filter |
| `--max-velocity-threshold` | `15.0` | Max velocity for motion quality filter |
| `--max-dof-vel-threshold` | `40.0` | Max DOF velocity for motion quality filter |

### Step 5 — Package into MotionLib `.pt`

**Env:** `env_isaaclab`

```bash
python protomotions/components/motion_lib.py \
    --motion-path ~/Data/smpl/amass_motionlib/proto-astro/train/ \
    --output-file ~/Data/smpl/amass_motionlib/proto-astro/train.pt
```

| Flag | Default | Description |
|---|---|---|
| `--motion-path` | `""` | Path to `.motion` file, `.yaml` manifest, `.pt`, or directory |
| `--output-file` | `motion_lib.pt` | Output `.pt` file path |
| `--device` | `cpu` | Device for processing (`cpu` or `cuda`) |

### Step 6 — Visualize & Verify

**Env:** `env_isaaclab`

```bash
python examples/motion_libs_visualizer.py \
    --motion_files ~/Data/smpl/amass_motionlib/proto-astro/train.pt \
    --robot astro \
    --simulator isaaclab \
    --cpu-only
```

| Flag | Default | Description |
|---|---|---|
| `--motion_files` | *(required)* | One or more `.pt` MotionLib files (space-separated for side-by-side) |
| `--robot` | `g1` | Robot: `g1`, `h1_2`, `astro`, `smpl`, `rigv1`, `soma23` |
| `--simulator` | `isaacgym` | Simulator: `isaacgym`, `isaaclab`, `newton` |
| `--cpu-only` | `False` | CPU-only mode (no GPU required) |
| `--headless` | `False` | Run without display |
| `--playback_speed` | `1.0` | Playback speed multiplier |
| `--metric` | `nj` | Smoothness metric: `nj` (normalized jerk), `oi` (oscillation index), `pj` (purposeful jerk) |
| `--smoothness_threshold` | `6500.0` | Threshold to highlight rough bodies |

**Visualizer controls:** **R** = next motion, **1/2** = speed up/down, **3/4** = adjust smoothness threshold.

---

## Further Reading

- [Official Retargeting Tutorial](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/retargeting_pyroki.html) — full walkthrough with theory
- [AMASS Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/amass_preparation.html) — preparing the source `.pt` file
- [PyRoki README](../pyroki/README.md) — CUDA setup, single-motion retargeting, keypoint tuning

---

## License

Apache-2.0 — see the project root [LICENSE.md](../LICENSE.md).
