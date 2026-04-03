# Implementation Plan: ProtoMotions Retargeting GUI Application

## Overview

A comprehensive tkinter-based GUI application that wraps the entire ProtoMotions retargeting pipeline, making it accessible to users unfamiliar with CLI commands. The application handles dual Python environment management (ProtoMotions + PyRoki), provides configurable parameters with defaults and descriptions, integrates the existing keypoint mapping tuner, and includes utilities for IsaacLab asset generation and URDF/MJCF joint order comparison.

## Requirements

Based on the user's 6 specified features:

1. **Full Pipeline Coverage** — Support all retargeting use cases:
   - Convenience script (batch retarget from .pt → robot motionlib)
   - Single motion retarget (from .motion file)
   - Step-by-step execution (each of the 5 pipeline steps independently + visualization)

2. **All CLI Parameters Configurable** — Every CLI argument from all underlying scripts exposed as GUI fields with defaults and tooltip descriptions.

3. **PyRoki Environment Management** — Special handling for the pyroki conda environment (separate Python interpreter), CUDA device verification, JAX backend detection.

4. **Keypoint Mapping & Config Tuning** — Integrate/launch the existing `visualize_keypoint_mapping_gui.py` functionality for interactive scale factor/aux offset tuning with MuJoCo viewer.

5. **IsaacLab Asset Generation** — Button to run the USD conversion pipeline (`flatten_mjcf.py` → `convert_robot_mjcf_to_usda.py`) with environment verification.

6. **URDF/MJCF Joint Order Comparison** — Button to extract and display joint/body ordering from both a URDF (used in pyroki retargeting) and an MJCF (used in IsaacLab training), showing a side-by-side comparison to verify consistency.

## Architecture

### File Structure

```
tools/
├── retargeting_gui.py              # Main entry point & GUI application
├── retargeting_gui_subprocess.py   # SubprocessRunner engine
└── .retargeting_gui_config.json    # Auto-generated config persistence (gitignored)
```

### Key Design Decisions

- **tkinter** UI framework (consistent with existing `visualize_keypoint_mapping_gui.py`)
- **Tab-based layout** using `ttk.Notebook` for organizing different use cases
- **Subprocess-based execution** to handle different Python environments
- **Thread-safe logging** via `root.after()` for real-time output streaming
- **Configuration persistence** via JSON for remembering user paths/settings

## Implementation Steps

### Step 1: SubprocessRunner Engine (`tools/retargeting_gui_subprocess.py`)

A shared utility class for running commands:

```python
class SubprocessRunner:
    - start(cmd, cwd, env_vars) → launches subprocess in background thread
    - cancel() → sends SIGTERM to running process
    - is_running → bool property
    - on_output(callback) → called per line of output
    - on_complete(callback) → called when process finishes (with return code)
```

Key design:
- All subprocess calls use the appropriate Python interpreter (proto vs pyroki)
- Working directory is always the project root
- Thread-safe communication with tkinter main loop via `root.after()`

### Step 2: Main Application Skeleton (`tools/retargeting_gui.py`)

- Apache-2.0 license header
- `RetargetingGUI` class with:
  - Environment Configuration Panel (top, shared)
  - `ttk.Notebook` with 6 tabs
  - Log Output Panel (bottom, shared)
  - Configuration persistence (save/load JSON on exit/startup)

### Step 3: Environment Configuration Panel (top-level, shared by all tabs)

Persistent top panel visible across all tabs:

- **ProtoMotions Python Path**: Text entry + Browse button. Default: `<project_root>/env_isaaclab/bin/python`
- **PyRoki Python Path**: Text entry + Browse button. Default: auto-detect from `~/miniforge3/envs/pyroki-cuda/bin/python` or `~/miniforge3/envs/pyroki/bin/python`
- **Project Root**: Auto-detected from script location (parent of `tools/`)
- **Verify Environments** button:
  - ProtoMotions: `<proto_python> -c "import protomotions; print('OK')"`
  - PyRoki: `<pyroki_python> -c "import pyroki; print('OK')"`
  - CUDA: `<pyroki_python> -c "import jax; print(jax.devices())"`
  - IsaacLab: `<proto_python> -c "import omni.isaac.lab; print('OK')"` (optional)
- Status indicator: Green/red per environment

### Step 4: Tab 1 — Convenience Script (Batch Retarget)

Replicates `scripts/retarget_amass_to_robot.sh` logic.

**Fields** (with defaults from the bash script):
- **AMASS .pt File**: File browser, filter `*.pt`
- **Robot Type**: Dropdown (`g1`, `h1_2`, `astro`)
- **Skip Frequency**: Spinbox, default `1` (1 = all motions)
- **Output Directory**: Auto-populated from `dirname(amass_pt_file)`, editable
- **Skeleton Format**: Dropdown (`smpl`, `rigv1`), default `smpl`

**Derived paths** (shown read-only):
- Keypoints dir: `{output_dir}/keypoints-for-retarget`
- Retargeted dir: `{output_dir}/pyroki-retargeted-{robot}`
- Contacts dir: `{output_dir}/contacts`
- Proto dir: `{output_dir}/proto-{robot}`
- Final .pt: `{output_dir}/proto-{robot}.pt`

**Actions**: Run All Steps, Progress bar (1/5 through 5/5), Cancel

### Step 5: Tab 2 — Single Motion Retarget

Replicates `scripts/retarget_single_motion_to_robot.sh`.

**Fields**:
- **Motion File**: File browser, filter `*.motion`
- **Robot Type**: Dropdown (`g1`, `h1_2`, `astro`)
- **Output Directory**: Directory browser

**Derived paths**: keypoints, retargeted, contacts, proto subdirectories

### Step 6: Tab 3 — Step-by-Step Execution

Each pipeline step as an expandable section with its own "Run" button:

1. **Extract Keypoints** — From .pt (batch) or .motion (single)
   - All params from `extract_retargeting_input_keypoints_from_packaged_motionlib.py`
   - Python env: ProtoMotions

2. **PyRoki Retarget** — Full retarget_from_keypoints params
   - `--robot-config`, `--keypoints-folder-path`, `--output-dir`, `--subsample-factor`, `--target-raw-frames`, `--skip-existing`, `--source-type`, `--no-visualize`, `--input-fps`
   - Python env: PyRoki

3. **Extract Contact Labels** — Same as step 2 with `--save-contacts-only`
   - Python env: PyRoki

4. **Convert to Proto Format** — Full convert script params including motion filter options
   - `--retargeted-motion-dir`, `--output-dir`, `--robot-type`, `--contact-labels-dir`, `--apply-motion-filter`, `--force-remake`, `--input-fps`, `--output-fps`, `--ignore-first-n-frames`, plus filter thresholds
   - Python env: ProtoMotions

5. **Package MotionLib** — `--motion-path`, `--output-file`
   - Python env: ProtoMotions

6. **Visualization** — Launch `motion_libs_visualizer.py`
   - `--motion_files`, `--robot`, `--simulator`, `--headless`, `--cpu-only`, `--playback_speed`
   - Python env: ProtoMotions

### Step 7: Tab 4 — Keypoint Mapping & Config Tuner

- **Launch GUI** button: Runs `visualize_keypoint_mapping_gui.py` via PyRoki Python
  - Params: `--robot-config`, `--source-type`, `--smpl-mjcf`, `--spacing`
- **Inline Robot Config Editor**: Load/edit/save YAML config
  - name, urdf_path, mesh_dir, keypoint_mapping table, scale_factors, weights, aux offsets, display_pose_preset
  - Save/New Config buttons

### Step 8: Tab 5 — IsaacLab Asset Generation

Wraps the `usd_convert/` pipeline:
- **Input MJCF**: File browser (*.xml)
- **Flatten MJCF** button: `flatten_mjcf.py <input.xml>` with `--no-verify` option
- **Convert to USDA** button: `convert_robot_mjcf_to_usda.py <flattened.xml>` with `--output-dir`
- **Run Both** button
- Validates IsaacLab availability before convert step

### Step 9: Tab 6 — Joint Order Comparison

- **URDF File**: File browser (*.urdf)
- **MJCF File**: File browser (*.xml)
- **Compare** button
- **Output**: Two-column treeview showing:
  - URDF joints vs MJCF joints (side-by-side)
  - URDF links vs MJCF bodies
  - Highlighted mismatches
  - Match Status indicator

### Step 10: Log Output Panel (bottom, shared)

- Scrollable text area with real-time stdout/stderr
- Clear, Save Log buttons
- Auto-scroll toggle

### Step 11: Configuration Persistence

JSON file at `tools/.retargeting_gui_config.json`:
- proto_python, pyroki_python, project_root
- Last-used robot type, directories
- Window geometry
- Saved on exit, loaded on startup

## Environment Paths (Current Setup)

- ProtoMotions Python: `/home/android/Code/notAnyrobot/ProtoMotions/env_isaaclab/bin/python`
- PyRoki CUDA Python: `/home/android/miniforge3/envs/pyroki-cuda/bin/python`
- Project root: `/home/android/Code/notAnyrobot/ProtoMotions`

## Key Files Referenced

| File | Purpose | Python Env |
|------|---------|-----------|
| `data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py` | Step 1 (batch) | ProtoMotions |
| `data/scripts/extract_keypoints_from_single_motion.py` | Step 1 (single) | ProtoMotions |
| `pyroki/batch_retarget_to_{robot}_from_keypoints.py` | Step 2 | PyRoki |
| `pyroki/retarget_from_keypoints.py` | Step 2 (direct) | PyRoki |
| `data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py` | Step 4 | ProtoMotions |
| `protomotions/components/motion_lib.py` | Step 5 | ProtoMotions |
| `examples/motion_libs_visualizer.py` | Visualization | ProtoMotions |
| `pyroki/visualize_keypoint_mapping_gui.py` | Config tuner | PyRoki |
| `usd_convert/flatten_mjcf.py` | USD pipeline | ProtoMotions |
| `usd_convert/convert_robot_mjcf_to_usda.py` | USD pipeline | IsaacLab |
| `protomotions/components/pose_lib.py` | Joint comparison | ProtoMotions |
