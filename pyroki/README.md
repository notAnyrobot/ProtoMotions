# PyRoki Retargeting Workflow

This directory contains ProtoMotions' local PyRoki retargeting entrypoints and
configuration glue. The workflow is:

1. Prepare SMPL MotionLib `.pt` files from AMASS split YAML files.
2. Extract SMPL keypoints and source contact labels.
3. Retarget keypoints to a target robot with PyRoki.
4. Convert PyRoki `.npz` output to ProtoMotions `.motion`.
5. Package `.motion` files into MotionLib `.pt` files for visualization and
   tracker training.

PyRoki dependencies are managed separately from the ProtoMotions Python
environment. On the workstation, use the conda PyRoki environment for PyRoki
steps and the ProtoMotions/IsaacLab environment for conversion, packaging, and
visualization.

```bash
cd /home/android/Code/NVlabs/ProtoMotions

export PROTO_ROOT=/home/android/Code/NVlabs/ProtoMotions
export PYROKI_REPO=/home/android/Code/NVlabs/pyroki
export PYROKI_PY=/home/android/miniforge3/envs/pyroki-cuda/bin/python
export PROTO_PY="$PROTO_ROOT/.venv_isaaclab/bin/python"
```

Use `PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src"` for local retargeting
commands. The first path exposes this repo's `retargeting/*` CLI code, while
the second path exposes the official PyRoki package.

## Data Preparation

AMASS split YAML files select the source motions and slices. Each entry's
`file` path is relative to the AMASS `.motion` root, and `sub_motions.timings`
defines the clip to package from the full converted motion.

The prepared local layout is split-aware:

```text
/media/android/data/motion_datasets/protomotions/
└── smpl/
    ├── sfu/
    │   ├── amass_smpl_sfu.pt
    │   ├── keypoints-for-retarget/
    │   └── contacts/
    ├── train/
    │   ├── amass_smpl_train.pt
    │   ├── keypoints-for-retarget/
    │   └── contacts/
    ├── test/
    │   ├── amass_smpl_test.pt
    │   ├── keypoints-for-retarget/
    │   └── contacts/
    └── validation/
        ├── amass_smpl_validation.pt
        ├── keypoints-for-retarget/
        └── contacts/
```

To prepare one split from a YAML file:

```bash
export AMASS_ROOT=/media/android/data/motion_datasets/smpl/amass_smpl+hg
export DATA=/media/android/data/motion_datasets/protomotions
export SPLIT=sfu
export SPLIT_YAML="$DATA/smpl/yaml_files/amass_smpl_${SPLIT}.yaml"
export SPLIT_DIR="$DATA/smpl/$SPLIT"

mkdir -p "$SPLIT_DIR"

"$PROTO_PY" data/scripts/convert_amass_to_motionlib.py \
  "$AMASS_ROOT" \
  "$SPLIT_DIR" \
  --motion-config "$SPLIT_YAML" \
  --humanoid-type smpl \
  --output-fps 30 \
  --device cpu
```

Expected output:

```text
$SPLIT_DIR/amass_smpl_sfu.pt
```

For multiple split YAMLs, pass `--motion-config` multiple times or run the same
command per split directory. Keep each split's `.pt`, keypoints, and contacts
under its own subfolder so robot retargeting can write split-aware outputs.

## Convenience Scripts

### `scripts/retarget_single_motion_to_robot.sh`


### `scripts/retarget_amass_to_robot.sh`


## Single Motion Retargeting

This section uses the verified long-motion SFU example:

```text
SMPL split: sfu
Motion: 0015_0015_KendoKata001_poses_keypoints.npy
Source frames: 4678
Robot: astro
Chunking: threshold=900, size=450, overlap=60
Expected final chunk: [4290, 4678), length 388
```

Set common paths:

```bash
cd /home/android/Code/NVlabs/ProtoMotions

export PROTO_ROOT=/home/android/Code/NVlabs/ProtoMotions
export PYROKI_REPO=/home/android/Code/NVlabs/pyroki
export PYROKI_PY=/home/android/miniforge3/envs/pyroki-cuda/bin/python
export PROTO_PY="$PROTO_ROOT/.venv_isaaclab/bin/python"

export DATA=/media/android/data/motion_datasets/protomotions
export SPLIT=sfu
export ROBOT=astro
export SMPL_PT="$DATA/smpl/$SPLIT/amass_smpl_$SPLIT.pt"
export KEYPOINTS_DIR="$DATA/smpl/$SPLIT/keypoints-for-retarget"
export CONTACTS_DIR="$DATA/smpl/$SPLIT/contacts"
```

### 1. Extract SMPL Keypoints

Skip this step if `$KEYPOINTS_DIR` already contains the selected motion.

```bash
mkdir -p "$KEYPOINTS_DIR"

"$PROTO_PY" data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
  "$SMPL_PT" \
  --output-path "$KEYPOINTS_DIR" \
  --skeleton-format smpl \
  --start-idx 0 \
  --skip-freq 1
```

The output files are `.npy` dictionaries with:

```text
positions             (T, 18, 3)
orientations          (T, 18, 3, 3)
left_foot_contacts    (T, 2)
right_foot_contacts   (T, 2)
```

### 2. Pick A Long Motion

List the longest extracted keypoint files:

```bash
PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src" "$PYROKI_PY" - <<'PY'
from pathlib import Path
import numpy as np

root = Path("/media/android/data/motion_datasets/protomotions/smpl/sfu/keypoints-for-retarget")
rows = []
for path in sorted(root.glob("*.npy")):
    data = np.load(path, allow_pickle=True).item()
    rows.append((data["positions"].shape[0], path.name))

for frames, name in sorted(rows, reverse=True)[:10]:
    print(frames, name)
PY
```

For the verified long-motion test:

```bash
export MOTION=0015_0015_KendoKata001_poses_keypoints
export WORK="$DATA/$ROBOT/debug-long-motion-sfu-$MOTION-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORK/keypoints" \
         "$WORK/pyroki-retargeted-$ROBOT" \
         "$WORK/contacts" \
         "$WORK/proto-$ROBOT"

cp "$KEYPOINTS_DIR/$MOTION.npy" "$WORK/keypoints/"
```

### 3. Verify Chunk Plan

```bash
PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src" "$PYROKI_PY" - <<PY
from pathlib import Path
import numpy as np
from retargeting.solver import generate_retarget_windows

src = Path("$WORK/keypoints/$MOTION.npy")
data = np.load(src, allow_pickle=True).item()
frames = data["positions"].shape[0]

windows = generate_retarget_windows(
    num_frames=frames,
    chunk_threshold_frames=900,
    chunk_size_frames=450,
    chunk_overlap_frames=60,
)

print("frames", frames)
print("chunks", len(windows))
for i, w in enumerate(windows, 1):
    print(f"{i:02d}: {w.start} {w.end} len={w.length}")
PY
```

Expected tail:

```text
10: 3510 3960 len=450
11: 3900 4350 len=450
12: 4290 4678 len=388
```

### 4. Retarget With PyRoki

Do not pass `--target-raw-frames` for this validation. Leaving it unset verifies
the full source trajectory.

```bash
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XDG_CACHE_HOME=/tmp/protomotions-cache-sfu-long
export MPLCONFIGDIR=/tmp/protomotions-mpl-sfu-long

PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src" "$PYROKI_PY" \
  pyroki/batch_retarget_from_keypoints.py \
  --robot-type "$ROBOT" \
  --subsample-factor 1 \
  --keypoints-folder-path "$WORK/keypoints" \
  --source-type smpl \
  --input-fps 30 \
  --chunk-long-motions \
  --chunk-threshold-frames 900 \
  --chunk-size-frames 450 \
  --chunk-overlap-frames 60 \
  --output-dir "$WORK/pyroki-retargeted-$ROBOT" \
  --no-visualize \
  2>&1 | tee "$WORK/retarget.log"
```

Expected log lines:

```text
Chunking long motion into 12 chunks: size=450, overlap=60
Retargeting chunk 12/12: frames [4290, 4678)
Saved retargeted motion to ..._retargeted.npz
```

The retargeted `.npz` contains generalized robot state only:

```text
base_frame_pos   (T, 3)   root position in meters
base_frame_wxyz  (T, 4)   root orientation quaternion, wxyz
joint_angles     (T, N)   target robot joint angles in radians
```

It does not contain rigid body states. The ProtoMotions conversion step runs FK
and creates rigid-body positions, rotations, velocities, DOF velocities, and
contacts in the `.motion` file.

Inspect the `.npz`:

```bash
OUT="$WORK/pyroki-retargeted-$ROBOT/${MOTION}_retargeted.npz"

PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src" "$PYROKI_PY" - <<PY
import numpy as np
out = "$OUT"
d = np.load(out)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
assert d["base_frame_pos"].shape[0] == 4678
assert d["base_frame_wxyz"].shape[0] == 4678
assert d["joint_angles"].shape[0] == 4678
print("OK:", out)
PY
```

### 5. Generate Contact Labels

```bash
PYTHONPATH="$PROTO_ROOT/pyroki:$PYROKI_REPO/src" "$PYROKI_PY" \
  pyroki/batch_retarget_from_keypoints.py \
  --robot-type "$ROBOT" \
  --subsample-factor 1 \
  --keypoints-folder-path "$WORK/keypoints" \
  --source-type smpl \
  --input-fps 30 \
  --save-contacts-only \
  --contacts-dir "$WORK/contacts"
```

### 6. Convert `.npz` To ProtoMotions `.motion`

```bash
"$PROTO_PY" data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
  --retargeted-motion-dir "$WORK/pyroki-retargeted-$ROBOT" \
  --output-dir "$WORK/proto-$ROBOT" \
  --robot-type "$ROBOT" \
  --input-fps 30 \
  --output-fps 30 \
  --contact-labels-dir "$WORK/contacts" \
  --apply-motion-filter \
  --force-remake
```

Expected output:

```text
$WORK/proto-astro/0015_0015_KendoKata001_poses_keypoints_retargeted.motion
```

### 7. Package `.motion` To MotionLib `.pt`

```bash
"$PROTO_PY" protomotions/components/motion_lib.py \
  --motion-path "$WORK/proto-$ROBOT" \
  --output-file "$WORK/proto-$ROBOT.pt" \
  --device cpu
```

### 8. Visualize Astro Retarget

```bash
"$PROTO_PY" examples/motion_libs_visualizer.py \
  --motion_files "$WORK/proto-$ROBOT.pt" \
  --robot "$ROBOT" \
  --simulator isaaclab \
  --playback_speed 1.0
```

### 9. Visualize The Matching SMPL Reference

Package only the matching source SMPL `.motion`:

```bash
export SMPL_REF_DIR="$WORK/smpl-reference"
export SMPL_REF_MOTION=/media/android/data/motion_datasets/smpl/amass_smpl+hg/SFU/0015/0015_KendoKata001_poses.motion

mkdir -p "$SMPL_REF_DIR"

"$PROTO_PY" protomotions/components/motion_lib.py \
  --motion-path "$SMPL_REF_MOTION" \
  --output-file "$SMPL_REF_DIR/smpl-kendokata.pt" \
  --device cpu
```

Visualize the SMPL reference:

```bash
"$PROTO_PY" examples/motion_libs_visualizer.py \
  --motion_files "$SMPL_REF_DIR/smpl-kendokata.pt" \
  --robot smpl \
  --simulator isaaclab \
  --playback_speed 1.0
```

Use the SMPL reference to compare high-level timing, heading, limb intent, and
contact timing. Use the Astro visualization to check robot feasibility: foot
sliding, joint-limit artifacts, torso stability, arm posture, and chunk-boundary
smoothness.
