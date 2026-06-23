# Newton / IsaacLab + PyRoki Docker Workflow

The Docker launchers are intentionally small. They open an interactive container
with the ProtoMotions checkout, sibling PyRoki checkout, the full
`motion_datasets` root, GPU devices, and cache paths mounted. Validation,
retargeting, conversion, packaging, and policy training are normal commands that
you type inside that shell.

## Contents

- [Quick Start](#quick-start)
  - [Launch A Container](#launch-a-container)
  - [Run The Retargeting Workflow](#run-the-retargeting-workflow)
  - [Train A Tracker Policy](#train-a-tracker-policy)
- [Data Preparation](#data-preparation)
  - [Pull Prepared Data From HPC](#pull-prepared-data-from-hpc)
  - [Pull Retargeted Robot Split From HPC](#pull-retargeted-robot-split-from-hpc)
  - [Push Local Subset Splits To HPC](#push-local-subset-splits-to-hpc)
- [Retargeting](#retargeting)
  - [Runtime Check](#runtime-check)
  - [Canonical HPC PyRoki Retargeting](#canonical-hpc-pyroki-retargeting)
- [Policy Training](#policy-training)
- [FAQs](#faqs)
  - [How Do I Repair Rootless HPC Artifact Ownership?](#how-do-i-repair-rootless-hpc-artifact-ownership)
  - [Delete and recreate disposable results](#delete-and-recreate-disposable-results)

## Quick Start

### Launch A Container

Workstation:

```bash
./scripts/docker/launch_newton_docker_ws.sh
```

HPC:

```bash
PROTO_HPC_GPUS=7 ./scripts/docker/launch_newton_docker_hpc.sh
```

IsaacLab backend on HPC:

```bash
PROTO_DATASET_READONLY=0 PROTO_HPC_GPUS=7 ./scripts/docker/launch_isaaclab_docker_hpc.sh
```

Current HPC operating status: use the Newton backend container for retargeting
and tracker policy training. The IsaacLab backend container can launch and run
some utility commands, but tracker training currently hangs during the
IsaacLab/Isaac Sim launch stage on HPC. There is no active plan to fix the
IsaacLab HPC training path, so do not use it as the primary training backend.

The dataset mount defaults to the parent `motion_datasets` directory so source
AMASS data such as `amass_smpl+hg`, generated ProtoMotions artifacts under
`protomotions`, SMPL assets, and other sibling datasets are visible together:

```bash
# Workstation default
/media/android/data/motion_datasets

# HPC default
/data/share/motion_datasets
```

Use `PROTO_DATASET_ROOT` only when you want a narrower mount. Retargeting writes
to the dataset tree, so launch with a writable dataset mount:

```bash
PROTO_DATASET_READONLY=0 ./scripts/docker/launch_newton_docker_ws.sh
PROTO_DATASET_READONLY=0 PROTO_HPC_GPUS=7 ./scripts/docker/launch_newton_docker_hpc.sh
```

Useful GPU selection examples:

```bash
PROTO_GPUS=0 ./scripts/docker/launch_newton_docker_ws.sh
PROTO_HPC_GPUS=4,5,6,7 ./scripts/docker/launch_newton_docker_hpc.sh
PROTO_HPC_GPU_MODE=gpus PROTO_HPC_GPUS=0 ./scripts/docker/launch_newton_docker_hpc.sh
```

### Run The Retargeting Workflow

Inside the retargeting container, run the canonical convenience script from the
repo root. Newton is the primary HPC container for retargeting and tracker
training, so the script defaults ProtoMotions steps to plain `python` from
`PATH` unless you explicitly set `PROTO_PYTHON` or
`PROTO_ISAACLAB_LAUNCHER`.

```bash
cd /workspace/protomotions

./scripts/docker/retarget_amass_hpc.sh [split] [robot] [skip_freq]
```

Arguments are optional. Defaults are `split=test`, `robot=astro`, and
`skip_freq=1`. The `split` argument can be any simple folder name under
`smpl/`, including dataset subsets such as `sfu`, `cmu`, and `accad`.

`retarget_amass_isaaclab_hpc.sh is a compatibility wrapper` for older notes and
jobs. It delegates to `retarget_amass_hpc.sh` and prints a deprecation warning.

Examples:

```bash
./scripts/docker/retarget_amass_hpc.sh
./scripts/docker/retarget_amass_hpc.sh test astro 50
./scripts/docker/retarget_amass_hpc.sh sfu astro 1
./scripts/docker/retarget_amass_hpc.sh train astro 1
```

### Train A Tracker Policy

Use the packaged MotionLib file as the motion source. Example Astro Newton run:

```bash
DATA=/media/android/data/motion_datasets/protomotions
SPLIT=test

python -u -m protomotions.train_agent \
  --robot-name astro \
  --simulator newton \
  --num-envs 2048 \
  --batch-size 4096 \
  --motion-file "$DATA/astro/$SPLIT/proto-astro.pt" \
  --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --experiment-name astro-motion-tracker-newton-container
```

## Data Preparation

Prepared ProtoMotions datasets live under a shared `protomotions` dataset root.
The expected roots are:

```bash
# HPC
/data/share/motion_datasets/protomotions

# Workstation
/media/android/data/motion_datasets/protomotions
```

The current layout is split-aware. SMPL source MotionLibs, extracted keypoints,
and source contact labels stay under `smpl/<split>/`. Robot-specific retargeted
data is written under `<robot>/<split>/`.

```text
protomotions/
├── smpl/
│   ├── accad/
│   │   └── amass_smpl_accad.pt
│   ├── cmu/
│   │   └── amass_smpl_cmu.pt
│   ├── test/
│   │   ├── amass_smpl_test.pt
│   │   ├── keypoints-for-retarget/
│   │   └── contacts/
│   ├── train/
│   │   ├── amass_smpl_train.pt
│   │   ├── keypoints-for-retarget/
│   │   └── contacts/
│   └── validation/
│       ├── amass_smpl_validation.pt
│       ├── keypoints-for-retarget/
│       └── contacts/
├── astro/
│   ├── test/
│   │   ├── pyroki-retargeted-astro/
│   │   ├── proto-astro/
│   │   └── proto-astro.pt
│   ├── train/
│   └── validation/
└── g1/
│   ├── test/
│   ├── train/
│   └── validation/
```

Older flat files such as `smpl/amass_smpl_train.pt` should be treated as
legacy compatibility artifacts once matching split copies exist at
`smpl/train/amass_smpl_train.pt`.

### Pull Prepared Data From HPC

Use a dry run first. The trailing slash on the source path is important: it
copies the contents into the matching local directory instead of creating a
nested directory.

```bash
REMOTE=atom7@192.168.24.9:/data/share/motion_datasets/protomotions
LOCAL=/media/android/data/motion_datasets/protomotions

mkdir -p "$LOCAL"

rsync -avhn --stats --itemize-changes \
  "$REMOTE/smpl/" \
  "$LOCAL/smpl/"
```

If the workstation has an SSH alias for the HPC, replace the `REMOTE` host with
that alias, for example `hpc-1:/data/share/motion_datasets/protomotions`.

If the dry run looks right, pull the prepared SMPL MotionLib inputs, extracted
keypoints, and contact labels:

```bash
rsync -avh --info=progress2 --partial \
  "$REMOTE/smpl/" \
  "$LOCAL/smpl/"
```

### Pull Retargeted Robot Split From HPC

For per-split retargeted robot outputs, use the convenience wrapper. It pulls
the requested HPC folder:

```text
/data/share/motion_datasets/protomotions/<robot>/<split>/
```

into the matching workstation folder:

```text
/media/android/data/motion_datasets/protomotions/<robot>/<split>/
```

Astro is the default robot, and `--split` is required:

```bash
./scripts/docker/download_retargeted_motion_from_hpc.sh --split sfu
./scripts/docker/download_retargeted_motion_from_hpc.sh --robot astro --split sfu
./scripts/docker/download_retargeted_motion_from_hpc.sh --robot astro --split validation --dry-run
```

Use `REMOTE_HOST` if this workstation reaches the HPC through an SSH alias:

```bash
REMOTE_HOST=hpc-1 ./scripts/docker/download_retargeted_motion_from_hpc.sh --split sfu
```

For a lightweight SMPL pull that only copies packaged `.pt` files and skips
extracted keypoints/contact labels, sync the split folders without descending
into keypoint/contact directories:

```bash
mkdir -p "$LOCAL/smpl"

rsync -avh --info=progress2 --partial \
  --include='*/' \
  --include='amass_smpl_*.pt' \
  --exclude='*' \
  "$REMOTE/smpl/" \
  "$LOCAL/smpl/"
```

Verify a completed transfer by checking the local layout and rerunning the dry
run. A clean dry run should not list new `>f+++++++++` or `cd+++++++++` entries.

```bash
ls -lh "$LOCAL/smpl"
du -sh "$LOCAL/smpl" "$LOCAL/astro" 2>/dev/null || true
find "$LOCAL/smpl" -maxdepth 2 -type f -name "*.pt" -ls

rsync -avhn --stats --itemize-changes \
  "$REMOTE/smpl/" \
  "$LOCAL/smpl/"
```

These commands intentionally avoid `--delete` so local-only validation artifacts
or backups are preserved. Use `--delete` only when you want the local directory
to exactly mirror the HPC source directory.

### Push Local Subset Splits To HPC

Use this when the workstation has prepared dataset-specific SMPL subsets such as
`accad/`, `cmu/`, or `sfu/`, and the HPC already has clean `train/`, `test`, and
`validation` splits.

Do not push workstation `train/`, `test`, or `validation` if those folders
contain stale `contacts/` or `keypoints-for-retarget/` artifacts. Sync only the
new subset folders that contain packaged `.pt` files.

```bash
LOCAL=/media/android/data/motion_datasets/protomotions/smpl
REMOTE_HOST=atom7@192.168.24.9
REMOTE=/data/share/motion_datasets/protomotions/smpl

SUBSETS=(
  accad
  biomotionlab_ntroje
  bmlhandball
  bmlmovi
  cmu
  dancedb
  dfaust_67
  ekut
  eyes_japan_dataset
  kit
  mpi_hdm05
  mpi_limits
  mpi_mosh
  sfu
  ssm_synced
  tcd_handmocap
  totalcapture
  transitions_mocap
)

RSYNC_SOURCES=()
for SPLIT in "${SUBSETS[@]}"; do
  RSYNC_SOURCES+=("$LOCAL/$SPLIT")
done
```

If the remote parent does not exist yet, create it once:

```bash
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE'"
```

Then run one `rsync` command for all subset folders. This avoids opening one SSH
session per subset, which is especially painful on password-authenticated hosts.

Dry run:

```bash
rsync -avhn --stats --itemize-changes \
  "${RSYNC_SOURCES[@]}" \
  "$REMOTE_HOST:$REMOTE/"
```

Real sync:

```bash
rsync -avh --info=progress2 --partial \
  "${RSYNC_SOURCES[@]}" \
  "$REMOTE_HOST:$REMOTE/"
```

Verify on HPC:

```bash
DATA=/data/share/motion_datasets/protomotions/smpl

find "$DATA" -maxdepth 2 -type f -name 'amass_smpl_*.pt' | sort
```

Expected examples:

```text
/data/share/motion_datasets/protomotions/smpl/accad/amass_smpl_accad.pt
/data/share/motion_datasets/protomotions/smpl/cmu/amass_smpl_cmu.pt
/data/share/motion_datasets/protomotions/smpl/test/amass_smpl_test.pt
/data/share/motion_datasets/protomotions/smpl/train/amass_smpl_train.pt
/data/share/motion_datasets/protomotions/smpl/validation/amass_smpl_validation.pt
```

## Retargeting

### Runtime Check

Inside the container:

```bash
nvidia-smi

python - <<'PY'
import jax
import jax.numpy as jnp
import mujoco
import mujoco_warp
import newton
import torch

print("torch cuda", torch.cuda.is_available())
print("torch gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("newton", getattr(newton, "__version__", "unknown"))
print("mujoco", mujoco.__version__)
print("mujoco_warp", getattr(mujoco_warp, "__version__", "unknown"))
print("jax backend", jax.default_backend())
print("jax devices", jax.devices())
print("jax cpu devices", jax.devices("cpu"))

assert torch.cuda.is_available()
assert jax.default_backend() == "gpu"
assert jax.devices("cpu")
print(jnp.zeros(6).block_until_ready())

@jax.jit
def with_callback(x):
    jax.debug.callback(lambda value: None, x)
    return x + 1

print(with_callback(jnp.array(1.0)).block_until_ready())
PY

python pyroki/batch_retarget_from_keypoints.py --help
```

### Canonical HPC PyRoki Retargeting

Use this workflow inside the Newton or IsaacLab backend container when
retargeting AMASS MotionLib `.pt` files to a robot. The canonical script is
backend-neutral: ProtoMotions steps use `python` from `PATH` by default, and
IsaacLab launcher use is explicit through `PROTO_ISAACLAB_LAUNCHER` or
`PROTO_PYTHON`.

PyRoki steps validate `PROTO_PYROKI_REPO` and set
`PYTHONPATH=$PROTO_PYROKI_REPO/src` so the external PyRoki package wins over any
repo-local `pyroki` namespace. The batch CLI is resolved from
`PROTO_PYROKI_BATCH_CLI`, then `$PROTO_PYROKI_REPO/batch_retarget_from_keypoints.py`,
then the repo-local `pyroki/batch_retarget_from_keypoints.py` helper. This
matches the Newton container layout where `/workspace/pyroki` can provide the
package while `/workspace/protomotions/pyroki` provides the ProtoMotions batch
wrapper.

Run the canonical convenience script from inside the container:

```bash
cd /workspace/protomotions

./scripts/docker/retarget_amass_hpc.sh [split] [robot] [skip_freq]
```

Arguments are optional. Defaults are `split=test`, `robot=astro`, and
`skip_freq=1`. The split can be a standard train/test/validation split or an
arbitrary subset folder such as `sfu`, `cmu`, or `accad`.

`retarget_amass_isaaclab_hpc.sh is a compatibility wrapper` kept for old
commands. New commands should call `retarget_amass_hpc.sh` directly.

Examples:

```bash
./scripts/docker/retarget_amass_hpc.sh
./scripts/docker/retarget_amass_hpc.sh test astro 50
./scripts/docker/retarget_amass_hpc.sh sfu astro 1
./scripts/docker/retarget_amass_hpc.sh train astro 1
```

Small-split SFU smoke:

```bash
DATA=/data/share/motion_datasets/protomotions
ls "$DATA/smpl/sfu/amass_smpl_sfu.pt"
./scripts/docker/retarget_amass_hpc.sh sfu astro 1
```

The script looks for packaged AMASS input in `smpl/<split>/` first, then falls
back to the flat layout:

```text
/data/share/motion_datasets/protomotions/smpl/<split>/amass_smpl_<split>.pt
/data/share/motion_datasets/protomotions/smpl/amass_smpl_<split>.pt
```

It writes split-aware outputs:

```text
/data/share/motion_datasets/protomotions/smpl/<split>/keypoints-for-retarget/
/data/share/motion_datasets/protomotions/smpl/<split>/contacts/
/data/share/motion_datasets/protomotions/<robot>/<split>/pyroki-retargeted-<robot>/
/data/share/motion_datasets/protomotions/<robot>/<split>/proto-<robot>/
/data/share/motion_datasets/protomotions/<robot>/<split>/proto-<robot>.pt
```

Useful overrides:

```bash
MOTION_DATASETS=/data/share/motion_datasets
PROTO_RETARGET_ROOT=$MOTION_DATASETS/protomotions
PROTO_AMASS_PT=/path/to/amass_smpl_test.pt
PROTO_PYTHON=/path/to/protomotions/python
PROTO_ISAACLAB_LAUNCHER=/workspace/isaaclab/isaaclab.sh
PYROKI_PYTHON=/workspace/pyroki-venv/bin/python
PROTO_PYROKI_REPO=/workspace/pyroki
PROTO_PYROKI_BATCH_CLI=/workspace/protomotions/pyroki/batch_retarget_from_keypoints.py
```

Long-motion retargeting is enabled by default in the HPC helper. Motions at or
below the threshold use the original full-trajectory PyRoki solve. Motions above
the threshold are solved in overlapping windows, then stitched back into one
`*_retargeted.npz` file. The final `.motion` files and packaged `.pt` MotionLib
preserve one entry per original AMASS motion.

Default chunking values:

```bash
PROTO_CHUNK_THRESHOLD_FRAMES=900
PROTO_CHUNK_SIZE_FRAMES=450
PROTO_CHUNK_OVERLAP_FRAMES=60
```

For AMASS MotionLibs that are effectively 60 FPS, override both pipeline FPS
values explicitly:

```bash
PROTO_INPUT_FPS=60 PROTO_OUTPUT_FPS=60 ./scripts/docker/retarget_amass_hpc.sh train astro 1
```

This keeps PyRoki velocity costs and converter timing aligned with the source
cadence.

Resume behavior follows PyRoki's `--skip-existing` semantics at the final
`*_retargeted.npz` level. If the final file already exists, the helper skips
that source motion. If the final file is absent, the helper recomputes any
required long-motion chunking internally and writes one final retargeted file.

For orientation, the helper wraps the same core PyRoki retarget, ProtoMotions
conversion, and MotionLib packaging commands:

```bash
PYTHONPATH="$PROTO_PYROKI_REPO/src" python "$PROTO_PYROKI_REPO/batch_retarget_from_keypoints.py" \
  --robot-type astro \
  --keypoints-folder-path "$PROTO_RETARGET_ROOT/smpl/<split>/keypoints-for-retarget" \
  --skip-existing

python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
  --retargeted-motion-dir "$PROTO_RETARGET_ROOT/astro/<split>/pyroki-retargeted-astro" \
  --output-dir "$PROTO_RETARGET_ROOT/astro/<split>/proto-astro" \
  --input-fps "$PROTO_INPUT_FPS" \
  --output-fps "$PROTO_OUTPUT_FPS"

python protomotions/components/motion_lib.py \
  --motion-path "$PROTO_RETARGET_ROOT/astro/<split>/proto-astro" \
  --output-file "$PROTO_RETARGET_ROOT/astro/<split>/proto-astro.pt"
```

The script deliberately does not auto-select `/workspace/isaaclab/isaaclab.sh`
just because that path exists. In the Newton container, plain `python` is the
expected ProtoMotions runner. In an IsaacLab container, set
`PROTO_ISAACLAB_LAUNCHER=/workspace/isaaclab/isaaclab.sh` when you want
ProtoMotions steps to run through `isaaclab.sh -p`.

## Policy Training

Use the Newton backend container for HPC tracker policy training. The IsaacLab
backend container currently hangs during the launch stage on HPC and is not part
of the active training workflow.

Use the packaged MotionLib file as the motion source. Example Astro Newton run:

```bash
DATA=/media/android/data/motion_datasets/protomotions
SPLIT=test

python -u -m protomotions.train_agent \
  --robot-name astro \
  --simulator newton \
  --num-envs 2048 \
  --batch-size 4096 \
  --motion-file "$DATA/astro/$SPLIT/proto-astro.pt" \
  --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --experiment-name astro-motion-tracker-newton-container
```

For a short local Newton training smoke, use a full-epoch budget rather than a
single step:

```bash
python -u -m protomotions.train_agent \
  --robot-name g1 \
  --simulator newton \
  --num-envs 128 \
  --batch-size 256 \
  --motion-file data/motion_for_trackers/g1_random_subset_tiny.pt \
  --experiment-path examples/experiments/mimic/mlp.py \
  --experiment-name newton-container-smoke \
  --training-max-steps 512
```

## FAQs

### How Do I Repair Rootless HPC Artifact Ownership?

Rootless HPC artifact cleanup depends on the host UID/GID mapping.

The workstation launcher repairs repo-local artifact ownership for `results`,
`output`, and `wandb` after the shell exits. The HPC launcher leaves this off by
default because true rootless Docker has different UID mapping behavior.

For a failed cleanup like `results/smpl-motion-tracker-amass`, verify the owner
on the HPC host before choosing the fix:

```bash
cd /data/atom7/Code/ProtoMotions

stat -c '%n uid=%u gid=%g owner=%U group=%G mode=%A type=%F' \
  results \
  results/smpl-motion-tracker-amass

namei -l results/smpl-motion-tracker-amass

find results/smpl-motion-tracker-amass -maxdepth 1 \
  -printf '%u:%g %m %p\n' | head -20
```

In true rootless Docker, container root normally maps to the launching host user.
On HPC-1, `chown 1007:1007` inside the container maps to host UID/GID
`559758:559758`, not host `atom7`. Do not use host numeric IDs inside the
container for ownership repair on this host. Use container `0:0` when you want
the result to be owned by host `atom7`.

### Delete and recreate disposable results

If `results/` only contains disposable validation artifacts, reset the whole
artifact root:

```bash
cd /data/atom7/Code/ProtoMotions

docker run --rm \
  --mount type=bind,src=/data/atom7/Code/ProtoMotions,dst=/repo \
  --entrypoint /bin/sh \
  protomotions-newton-pyroki:cuda12.4-newton1.0.0-pyroki59aa21f \
  -lc '
set -eu
if [ -e /repo/results ]; then
  chown -R 0:0 /repo/results
  rm -rf /repo/results
fi
mkdir -m 775 /repo/results
chown 0:0 /repo/results
'

stat -c '%n uid=%u gid=%g owner=%U group=%G mode=%A type=%F' results
```

The expected final owner is `atom7:atom7`.

For a repo-local test run that only needs deletion, remove it from the same
rootless Docker namespace instead of guessing a host-side `chown`:

```bash
docker run --rm \
  --mount type=bind,src=/data/atom7/Code/ProtoMotions/results,dst=/results \
  --entrypoint /bin/rm \
  protomotions-newton-pyroki:cuda12.4-newton1.0.0-pyroki59aa21f \
  -rf /results/smpl-motion-tracker-amass
```

If the directory is host `root:root`, use the host administrator path instead:

```bash
sudo chown -R "$(id -u):$(id -g)" results/smpl-motion-tracker-amass
rm -rf results/smpl-motion-tracker-amass
```

Dataset artifacts created by manual retargeting may still need an ownership fix.
On true rootless HPC, verify the UID mapping first; `chown "$(id -u):$(id -g)"`
inside a rootless container changes to container numeric IDs, not necessarily the
same host IDs.

Without `sudo`, use the container image to repair a dataset path only on a
Docker setup where container numeric UIDs map to the same host numeric UIDs:

```bash
docker run --rm \
  --mount type=bind,src=/media/android/data/motion_datasets/protomotions/astro,dst=/ownership-target \
  --entrypoint /bin/chown \
  protomotions-newton-pyroki:cuda12.4-newton1.0.0-pyroki59aa21f \
  -R "$(id -u):$(id -g)" /ownership-target
```

On HPC, replace the source path with
`/data/share/motion_datasets/protomotions/astro` only after confirming that UID
mapping behavior.
