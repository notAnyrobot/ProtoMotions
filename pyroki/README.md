# PyRoki — Robot Retargeting in ProtoMotions

This directory contains the **PyRoki-based motion retargeting** pipeline for
ProtoMotions. It converts human motion capture keypoints (SMPL / RigV1) into
physically plausible robot joint trajectories using nonlinear inverse-kinematics
optimization powered by [PyRoki](https://github.com/chungmin99/pyroki) and
[JAX](https://github.com/google/jax).

## Directory Layout

```
pyroki/
├── retarget_from_keypoints.py         # Core retargeting script (single or batch)
├── batch_retarget_to_g1_from_keypoints.py
├── batch_retarget_to_h1_2_from_keypoints.py
├── batch_retarget_to_astro_from_keypoints.py
├── visualize_keypoint_mapping.py      # Headless keypoint mapping helper
├── visualize_keypoint_mapping_gui.py  # MuJoCo GUI keypoint tuner
└── robot_configs/
    ├── g1.yaml                        # Unitree G1
    ├── h1_2.yaml                      # Unitree H1-2
    └── astro.yaml                     # Apptronik Astro
```

## Quick Start (CPU)

If you only need CPU retargeting (smaller batches or testing):

```bash
conda create -n pyroki python=3.10 -y
conda activate pyroki

# Clone and install PyRoki from source
git clone https://github.com/chungmin99/pyroki.git ~/pyroki-src
pip install -e ~/pyroki-src

# Install additional dependencies
pip install numpy scipy pyyaml
```

Then run from the ProtoMotions root:

```bash
conda activate pyroki
cd /path/to/ProtoMotions

python pyroki/retarget_from_keypoints.py \
    --robot-config pyroki/robot_configs/g1.yaml \
    --keypoints-folder-path /path/to/keypoints/ \
    --output-dir /path/to/output/ \
    --source-type smpl \
    --no-visualize
```

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

JAX provides prebuilt wheels with CUDA bundled. Match the JAX version to your
GPU driver. For CUDA 12 (covers most modern GPUs and drivers ≥ 525):

```bash
# Option A: Latest stable JAX with CUDA 12 (recommended)
pip install "jax[cuda12]"

# Option B: Pin specific versions (tested configuration)
pip install jax==0.6.0 jaxlib==0.6.0 jax-cuda12-pjrt==0.6.0 jax-cuda12-plugin==0.6.0
```

Verify CUDA is working:

```bash
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
# If you see [CpuDevice(id=0)], CUDA is not being used — see Troubleshooting.
```

### Step 4 — Install PyRoki and Dependencies

```bash
# Install PyRoki from source (editable mode)
git clone https://github.com/chungmin99/pyroki.git ~/pyroki-src
pip install -e ~/pyroki-src

# Install remaining dependencies used by the retargeting scripts
pip install numpy scipy pyyaml
```

### Step 5 — Verify the Full Stack

```bash
conda activate pyroki-cuda
python -c "
import jax
import pyroki
import jaxlie
import jaxls
import yourdfpy
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('All imports OK')
"
```

Expected output:

```
JAX version: 0.6.0
JAX devices: [CudaDevice(id=0)]
All imports OK
```

---

## Running Retargeting

### Single Motion (with visualization)

```bash
conda activate pyroki-cuda
cd /path/to/ProtoMotions

python pyroki/retarget_from_keypoints.py \
    --robot-config pyroki/robot_configs/g1.yaml \
    --keypoints-folder-path /path/to/keypoints/ \
    --source-type smpl
```

This opens an interactive 3D viewer for one motion at a time.

### Batch Retargeting (headless, CUDA recommended)

```bash
conda activate pyroki-cuda
cd /path/to/ProtoMotions

# Force CUDA backend
export JAX_PLATFORMS=cuda

python pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path /path/to/keypoints/ \
    --output-dir /path/to/retargeted/ \
    --source-type smpl \
    --no-visualize \
    --skip-existing
```

Replace `g1` with `h1_2` or `astro` for other robots.

### Extract Contact Labels Only

```bash
python pyroki/retarget_from_keypoints.py \
    --robot-config pyroki/robot_configs/g1.yaml \
    --keypoints-folder-path /path/to/keypoints/ \
    --source-type smpl \
    --save-contacts-only \
    --contacts-dir /path/to/contacts/
```

### Using the Retargeting GUI

The ProtoMotions Retargeting GUI (`tools/retargeting_gui.py`) provides a
graphical interface. Set the **PyRoki Python** field to your `pyroki-cuda`
interpreter:

```
~/miniforge3/envs/pyroki-cuda/bin/python
```

The GUI handles `JAX_PLATFORMS` and `PYTHONUNBUFFERED` automatically.

---

## Robot Configuration Files

Each robot requires a YAML config in `pyroki/robot_configs/`. Key fields:

```yaml
name: g1
urdf_path: ../../protomotions/data/assets/urdf/for_retargeting/g1.urdf
mesh_dir: ../../protomotions/data/assets/mesh/G1

# Human keypoint → robot link name mapping
keypoint_mapping:
  - [pelvis, pelvis_contour_link]
  - [left_hip, left_hip_pitch_link]
  - [left_knee, left_knee_link]
  # ... (see g1.yaml for full list)

# Source skeleton scaling factors
scale_factors:
  smpl: { ... }
  rigv1: { ... }

# Optimization weights (tunable per robot)
weights:
  local_alignment: 1.0
  global_alignment: 4.0
  joint_smoothness: 4.0
  # ...
```

Use the **Keypoint Config** tab in the GUI (or `visualize_keypoint_mapping_gui.py`
directly) to visually tune keypoint mappings in MuJoCo.

---

## Controlling JAX Backend

| Environment Variable | Effect |
|---------------------|--------|
| `JAX_PLATFORMS=cuda` | Force GPU (fails if no GPU) |
| `JAX_PLATFORMS=cpu` | Force CPU only |
| (unset) | Auto-detect (prefers GPU) |

In the GUI, set this via the **PyRoki Acceleration** dropdown.

From the command line:

```bash
# Force CPU even when GPU is available
JAX_PLATFORMS=cpu python pyroki/retarget_from_keypoints.py ...

# Force CUDA
JAX_PLATFORMS=cuda python pyroki/retarget_from_keypoints.py ...
```

---

## Full Pipeline (End-to-End)

The typical workflow from AMASS motion data to ProtoMotions training:

```
1. Extract keypoints     (ProtoMotions Python / env_isaaclab)
   ↓
2. PyRoki retargeting    (pyroki-cuda Python)   ← CUDA accelerated
   ↓
3. Extract contacts      (pyroki-cuda Python)   ← CUDA accelerated
   ↓
4. Convert to proto      (ProtoMotions Python / env_isaaclab)
   ↓
5. Package MotionLib     (ProtoMotions Python / env_isaaclab)
```

Steps 2 and 3 use the `pyroki-cuda` environment. All other steps use the
ProtoMotions `env_isaaclab` environment.

---

## Troubleshooting

### `[CpuDevice(id=0)]` instead of `[CudaDevice(id=0)]`

JAX cannot find CUDA. Check:

```bash
# Is NVIDIA driver loaded?
nvidia-smi

# Are CUDA JAX packages installed?
pip list | grep -i cuda
# Should show: jax-cuda12-pjrt, jax-cuda12-plugin

# Reinstall if missing
pip install "jax[cuda12]"
```

### `ModuleNotFoundError: No module named 'pyroki'`

You are using the wrong Python interpreter. Make sure to activate `pyroki-cuda`:

```bash
conda activate pyroki-cuda
which python  # Should point to miniforge3/envs/pyroki-cuda/bin/python
```

### `RuntimeWarning: JAX plugin version ... is not compatible`

Mismatch between `jax`, `jaxlib`, and cuda plugin versions. Pin them all to the
same version:

```bash
pip install jax==0.6.0 jaxlib==0.6.0 jax-cuda12-pjrt==0.6.0 jax-cuda12-plugin==0.6.0
```

### Retargeting runs but is very slow

- Verify CUDA is active: `python -c "import jax; print(jax.devices())"`
- First run is slow due to JIT compilation (JAX compiles the solver). Subsequent
  motions are much faster.
- Set `JAX_PLATFORMS=cuda` explicitly.

### Output is buffered / no progress visible

The scripts print progress per-motion. If running via the GUI, ensure
`PYTHONUNBUFFERED=1` is set (the GUI does this automatically). For manual runs:

```bash
PYTHONUNBUFFERED=1 python pyroki/retarget_from_keypoints.py ...
```
