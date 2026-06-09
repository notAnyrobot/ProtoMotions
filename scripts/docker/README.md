# Newton + PyRoki Docker Workflow

The Newton launchers are intentionally small. They open an interactive container
with the ProtoMotions checkout, sibling PyRoki checkout, the full
`motion_datasets` root, GPU devices, and cache paths mounted. Validation,
retargeting, conversion, packaging, and policy training are normal commands that
you type inside that shell.

## Launch A Container

Workstation:

```bash
./scripts/docker/launch_newton_docker_ws.sh
```

HPC:

```bash
PROTO_HPC_GPUS=7 ./scripts/docker/launch_newton_docker_hpc.sh
```

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

## Runtime Check

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

## PyRoki Retargeting

Choose the mounted dataset root for the host, then point commands at the
specific source and output subtrees:

```bash
# Workstation
MOTION_DATASETS=/media/android/data/motion_datasets

# HPC
MOTION_DATASETS=/data/share/motion_datasets

DATA=$MOTION_DATASETS/protomotions
AMASS=$MOTION_DATASETS/amass_smpl+hg
SPLIT=test
```

The AMASS source YAMLs remain usable at the source path:

```bash
python protomotions/components/motion_lib.py \
  --motion-path "$AMASS/amass_smpl_train.yaml" \
  --output-file "$DATA/smpl/motion_lib/amass_smpl_train.pt"
```

Retarget keypoints to Astro:

```bash
python pyroki/batch_retarget_from_keypoints.py \
  --robot-type astro \
  --subsample-factor 1 \
  --keypoints-folder-path "$DATA/smpl/keypoints-for-retarget/$SPLIT" \
  --source-type smpl \
  --output-dir "$DATA/astro/pyroki-retargeted/$SPLIT" \
  --no-visualize \
  --skip-existing
```

Generate or refresh contact labels:

```bash
python pyroki/batch_retarget_from_keypoints.py \
  --robot-type astro \
  --subsample-factor 1 \
  --keypoints-folder-path "$DATA/smpl/keypoints-for-retarget/$SPLIT" \
  --source-type smpl \
  --save-contacts-only \
  --contacts-dir "$DATA/smpl/contacts/$SPLIT" \
  --skip-existing
```

Convert PyRoki output to ProtoMotions motion files:

```bash
python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
  --retargeted-motion-dir "$DATA/astro/pyroki-retargeted/$SPLIT" \
  --output-dir "$DATA/astro/proto/$SPLIT" \
  --robot-type astro \
  --contact-labels-dir "$DATA/smpl/contacts/$SPLIT" \
  --input-fps 30 \
  --output-fps 30 \
  --apply-motion-filter \
  --force-remake
```

Package the converted motions into a MotionLib `.pt` file:

```bash
python protomotions/components/motion_lib.py \
  --motion-path "$DATA/astro/proto/$SPLIT" \
  --output-file "$DATA/astro/proto/amass-$SPLIT.pt"
```

## Policy Training

Use the packaged MotionLib file as the motion source. Example Astro Newton run:

```bash
DATA=/media/android/data/motion_datasets/protomotions
SPLIT=test

python -u -m protomotions.train_agent \
  --robot-name astro \
  --simulator newton \
  --num-envs 2048 \
  --batch-size 4096 \
  --motion-file "$DATA/astro/proto/amass-$SPLIT.pt" \
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

## Ownership Repair

The workstation launcher repairs repo-local artifact ownership for `results`,
`output`, and `wandb` after the shell exits. The HPC launcher leaves this off by
default because true rootless Docker has different UID mapping behavior.

### Rootless HPC artifact cleanup

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
