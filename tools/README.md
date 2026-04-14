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

## License

Apache-2.0 — see the project root [LICENSE.md](../LICENSE.md).
