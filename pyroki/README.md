# PyRoki — Robot Retargeting in ProtoMotions

This directory contains the **PyRoki-based motion retargeting** pipeline for
ProtoMotions. It converts human motion capture keypoints (SMPL / RigV1) into
physically plausible robot joint trajectories using nonlinear inverse-kinematics
optimization powered by [PyRoki](https://github.com/chungmin99/pyroki) and
[JAX](https://github.com/google/jax).

## Directory Layout

```
pyroki/
├── retarget_from_keypoints.py         # Core retargeting engine
├── batch_retarget.py                  # Unified CLI for robot-agnostic batch processing
├── robot_configs/                     # YAML configs for different robots
│   ├── g1.yaml                        # Unitree G1
│   ├── h1_2.yaml                      # Unitree H1-2
│   └── astro.yaml                     # Apptronik Astro
└── (deprecated shims for g1, h1_2, astro batch retargeting)
```

## Running Retargeting (CLI)

### 1. Unified Batch Retargeting

The preferred way to run batch retargeting is via `pyroki/batch_retarget.py`. This script automatically uses the correct robot configuration.

```bash
conda activate pyroki-cuda
export JAX_PLATFORMS=cuda

python pyroki/batch_retarget.py \
    --robot-config pyroki/robot_configs/g1.yaml \
    --keypoints-folder-path /path/to/keypoints/ \
    --output-dir /path/to/output/ \
    --source-type smpl \
    --no-visualize
```

### 2. Single Motion (with visualization)

Use the core engine for interactive debugging:

```bash
python pyroki/retarget_from_keypoints.py \
    --robot-config pyroki/robot_configs/g1.yaml \
    --keypoints-folder-path /path/to/keypoints/ \
    --source-type smpl
```

### 3. Deprecated Wrappers

The following scripts are **deprecated** and now serve as shims to `batch_retarget.py`:
- `pyroki/batch_retarget_to_g1_from_keypoints.py`
- `pyroki/batch_retarget_to_h1_2_from_keypoints.py`
- `pyroki/batch_retarget_to_astro_from_keypoints.py`

They will emit a deprecation warning and forward all arguments to the unified script.

---

## Retargeting GUI

For a graphical interface, use the dedicated **PyRoki GUI**:

```bash
python tools/pyroki_retarget_gui.py
```

The GUI manages the PyRoki environment and JAX acceleration, and includes the **Keypoint Config** tab for tuning robot mappings.

---

## Keypoint Mapping Tuner

The interactive MuJoCo-based tuner has moved to:
`tools/visualize_keypoint_mapping_gui.py`

Launch it via the **Keypoint Config** tab in the PyRoki GUI.

---

## CUDA-Accelerated Setup (Recommended)

CUDA acceleration speeds up retargeting by **10–50×** compared to CPU.
JAX compiles the optimization solver to GPU kernels automatically.

### Step 1 — Install Miniforge (if not already installed)

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Step 2 — Create the `pyroki-cuda` Conda Environment

```bash
conda create -n pyroki-cuda python=3.10 -y
conda activate pyroki-cuda
```

### Step 3 — Install JAX with CUDA Support

Match the JAX version to your GPU driver. For CUDA 12:

```bash
pip install jax==0.6.0 jaxlib==0.6.0 jax-cuda12-pjrt==0.6.0 jax-cuda12-plugin==0.6.0
```

### Step 4 — Install PyRoki and Dependencies

```bash
# Install PyRoki from source (editable mode)
git clone https://github.com/chungmin99/pyroki.git ~/pyroki-src
pip install -e ~/pyroki-src

# Install remaining dependencies
pip install numpy scipy pyyaml
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| CPU instead of CUDA | Run `python -c "import jax; print(jax.devices())"`. If it shows CPU, check JAX installation |
| Missing `pyroki` module | Ensure `pyroki-cuda` environment is active and PyRoki is installed via `pip install -e` |
| Buffered output | Set `PYTHONUNBUFFERED=1` to see real-time progress per-motion |

---

## License

Apache-2.0 — see the project root [LICENSE.md](../LICENSE.md).
