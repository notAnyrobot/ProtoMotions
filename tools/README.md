# ProtoMotions Retargeting GUI

A graphical front-end for the entire motion retargeting pipeline — from raw AMASS motion data through keypoint extraction, PyRoki retargeting, and final ProtoMotions-format packaging.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/UI-tkinter-green)

## Quick Start

```bash
cd /path/to/ProtoMotions
python tools/retargeting_gui.py
```

No extra dependencies beyond what ProtoMotions and PyRoki already require (tkinter is included with most Python distributions).

## Prerequisites

The GUI orchestrates two separate Python environments:

| Environment | Purpose | Typical Path |
|---|---|---|
| **ProtoMotions Python** | Keypoint extraction, format conversion, packaging, visualization | `env_isaaclab/bin/python` |
| **PyRoki Python** | Motion retargeting via JAX/PyRoki (GPU-accelerated) | `~/miniforge3/envs/pyroki-cuda/bin/python` |

The GUI auto-detects both interpreters on startup. You can override the paths in the **Python Environments** panel at the top of the window.

### Setup Checklist

1. **ProtoMotions** installed in a venv (`pip install -e .` inside `env_isaaclab`).
2. **PyRoki + JAX** installed in a conda env with CUDA support (see `requirements_mujoco.txt` or the project README for details).
3. **Motion data** — AMASS `.pt` files or individual `.motion` files.
4. **Robot configs** — YAML files in `pyroki/robot_configs/` (e.g., `astro.yaml`, `g1.yaml`, `h1_2.yaml`).

---

## Application Layout

The window is divided into three areas:

```
┌──────────────────────────────────────────────┐
│  Python Environments  [Verify] [Exit]        │  ← top panel
├──────────────────────────────────────────────┤
│  Tabs: Batch | Single | Step-by-Step |       │
│        Visualize | Keypoint Config |         │
│        USD | Joint Order                     │  ← main area (7 tabs)
├──────────────────────────────────────────────┤
│  Output Log  [Clear] [Save Log]              │  ← bottom panel (dark theme)
└──────────────────────────────────────────────┘
```

---

## Features & Tabs

### Environment Panel (Top)

- **ProtoMotions Python** — Path to the interpreter with ProtoMotions installed. Click **Browse** to change.
- **PyRoki Python** — Path to the interpreter with PyRoki + JAX installed.
- **PyRoki Acceleration** — Choose `cuda` (default, GPU) or `cpu`. This sets `JAX_PLATFORMS` for all PyRoki subprocess calls.
- **Verify Environments** — Tests both interpreters and checks JAX device availability.
- **Exit** — Gracefully closes the application (prompts if a process is running).

### Tab 1 — Batch Retarget

One-click pipeline for processing an entire AMASS `.pt` file through all 5 steps:

1. Extract keypoints from SMPL motions
2. Run PyRoki retargeting to the target robot
3. Extract foot contact labels
4. Convert to ProtoMotions format
5. Package into a MotionLib `.pt`

**How to use:**

1. Click **Browse** next to "AMASS .pt File" and select your packaged motion file.
2. Choose a **Robot Type** (default: `astro`).
3. Optionally set **Skip Frequency** > 1 for quick testing (e.g., 35 = process every 35th motion).
4. The output paths are auto-derived from your input file location. Override **Output Directory** if needed.
5. Click **Run All 5 Steps**. Progress is shown in the log and status bar.
6. Click **Cancel** at any time to abort.

### Tab 2 — Single Motion

Same pipeline as Batch, but for a single `.motion` file. Useful for testing one motion before running a full batch.

**How to use:**

1. Select a `.motion` file.
2. Choose **Robot Type** and **Output Directory**.
3. Click **Run All Steps**.

### Tab 3 — Step-by-Step

Full manual control over each pipeline stage with all available parameters exposed.

#### Step 1 — Extract Keypoints

- Choose between batch mode (`.pt` file) or single mode (`.motion` file).
- Set skeleton format, index range, and skip frequency.

#### Step 2 — PyRoki Retarget

- Select a **Robot Config** YAML file (default: `pyroki/robot_configs/astro.yaml`).
- Point to the **Keypoints Folder** from Step 1.
- Set subsample factor, target frame count, input FPS.
- **Retargeting Weights** — Tune all 9 optimization weights directly:

  | Weight | Default | Description |
  |---|---|---|
  | Local Alignment | 1.0 | Local link position alignment |
  | Global Alignment | 4.0 | Root-relative global alignment |
  | Root Smoothness | 1.0 | Root rotation smoothness |
  | Joint Smoothness | 4.0 | Joint angle smoothness |
  | Self Collision | 0.0 | Self-collision avoidance |
  | Joint Rest Penalty | 1.0 | Deviation from rest pose |
  | Joint Vel Limit | 50.0 | Joint velocity limit enforcement |
  | Foot Contact | 30.0 | Foot contact constraint |
  | Foot Tilt | 1.0 | Foot tilt constraint |

- **Load from Config** — reads weights from the selected robot config YAML.
- **Override config weights** — when checked (default), a temporary config is created with your custom weights. Uncheck to use the config file as-is.

#### Step 3 — Extract Contacts

Runs foot contact label extraction using the same robot config and keypoints.

#### Step 4 — Convert to ProtoMotions Format

Converts PyRoki output to `.motion` files. Includes motion quality filter controls:

- Min height, max velocity, max DOF velocity thresholds
- Duration-based height filter

#### Step 5 — Package MotionLib

Packages a directory of `.motion` files into a single `.pt` file for training.

### Tab 4 — Visualize

Dedicated tab for previewing retargeted motions in the simulator:

- Select one or more **Motion Files** (`.pt` or `.motion`, space-separated).
- Choose **Robot** (default: `astro`) and **Simulator** (default: `isaaclab`).
- Set **Playback Speed** (1.0 = real-time).
- Optional: **Headless** mode (for recording) or **CPU Only** mode.
- Click **Launch Visualizer** to open the motion viewer.

### Tab 5 — Keypoint Config

Two sections:

**Launch Keypoint Mapping Tuner** — Opens the interactive MuJoCo-based GUI (`visualize_keypoint_mapping_gui.py`) where you can visually inspect and tune keypoint-to-robot-link mappings.

**Robot Config Editor** — Load, edit, and save robot config YAML files directly in a text editor widget. Includes a "New from Template" button to scaffold a blank config.

### Tab 6 — USD Conversion

Convert MuJoCo MJCF files to USDA format for IsaacLab:

1. **Flatten MJCF** — Resolves MuJoCo defaults into a flat XML.
2. **Convert to USD** — Produces a USDA file (requires IsaacLab).
3. **Run Both** — Executes flatten then convert sequentially.

### Tab 7 — Joint Order

Compare joint/body ordering between a URDF (used by PyRoki) and an MJCF (used for simulation). Mismatches can cause silent data corruption. Results are shown in a color-coded table:

- 🟩 Green = matching
- 🟥 Red = mismatch
- 🟨 Yellow = extra (one file has more joints)

---

## Directory Browse Helpers

Every directory input field has three utility buttons:

| Button | Action |
|---|---|
| **Browse** | Opens a standard directory picker dialog |
| **~/Data** | Quick-navigate to `~/Data` (common motion data location) |
| **Create Dir** | Creates the directory typed in the field if it doesn't exist |

---

## Configuration Persistence

The GUI saves your Python interpreter paths to `tools/.retargeting_gui_config.json` on exit. These are restored automatically on next launch.

---

## Keyboard & Mouse Tips

- **Mouse wheel** scrolls the Step-by-Step and Batch tabs.
- **Hover** over any label to see a tooltip with parameter details.
- **Auto-scroll** in the log panel keeps the latest output visible (toggle via checkbox).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: retargeting_gui_subprocess` | Run from the project root: `python tools/retargeting_gui.py` |
| "Command not found" in log | Check that the Python paths in the Environment panel are correct |
| PyRoki runs on CPU despite `cuda` selected | Run **Verify Environments** — if JAX shows CPU-only, check your CUDA/JAX installation |
| GUI freezes during long runs | The GUI should remain responsive (subprocesses run in threads). If not, check the log for errors |
| Browse dialog opens in wrong directory | The dialog defaults to the current field value or the project root. Use **~/Data** for quick access |
| "Process is still running" on exit | Click **Yes** to force-cancel, or **No** to go back and cancel manually first |

---

## File Structure

```
tools/
├── README.md                          ← this file
├── retargeting_gui.py                 ← main GUI application (~1780 lines)
├── retargeting_gui_subprocess.py      ← thread-safe subprocess runner
├── visualize_keypoint_mapping_gui.py  ← MuJoCo keypoint mapping tuner (launched from Tab 4)
└── .retargeting_gui_config.json       ← auto-saved interpreter paths (gitignored)
```

---

## Supported Robots

| Robot | Config File | Notes |
|---|---|---|
| **Astro** | `pyroki/robot_configs/astro.yaml` | Default in all tabs |
| **G1** | `pyroki/robot_configs/g1.yaml` | Unitree G1 humanoid |
| **H1-2** | `pyroki/robot_configs/h1_2.yaml` | Unitree H1 v2 |

---

## License

Apache-2.0 — see the project root [LICENSE.md](../LICENSE.md).
