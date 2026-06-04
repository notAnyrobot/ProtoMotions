# Isaac Lab Docker Training Guide

This guide explains how to use the local Docker image
`protomotions-isaaclab:2.3.0` to validate and run ProtoMotions Isaac Lab
training on a workstation before converting the image for HPC.

The commands assume this checkout:

```bash
export REPO=/home/android/Code/NVlabs/ProtoMotions
export IMAGE=protomotions-isaaclab:2.3.0
```

Set those variables once in the shell where you run the examples.

Inside the container, always run Isaac Lab Python through:

```bash
/workspace/isaaclab/isaaclab.sh -p
```

Do not use plain `python` unless you have already confirmed it points to the
Isaac Lab Python environment.

## Workstation Launcher

For repeated local use on this workstation, prefer the repo-local launcher:

```bash
cd /home/android/Code/NVlabs/ProtoMotions

./scripts/docker/launch_isaaclab_docker_ws.sh nvidia-smi
./scripts/docker/launch_isaaclab_docker_ws.sh smoke
./scripts/docker/launch_isaaclab_docker_ws.sh shell
```

The launcher mounts the repo to `/workspace/protomotions` and also mounts the
real dataset root `/media/android/data/motion_lib` into the same absolute path
inside the container. That keeps symlinks such as `datasets/dobot/astro` working
inside Docker.

Training artifacts written inside the container land in the mounted host repo:

```text
container: /workspace/protomotions/results/<experiment-name>
host:      /home/android/Code/NVlabs/ProtoMotions/results/<experiment-name>
```

The launcher runs the container as root, then fixes host ownership for artifact
directories after the container exits. By default it fixes `results`, `output`,
and `wandb`.

Useful launcher overrides:

```bash
PROTO_GPU_MODE=cdi ./scripts/docker/launch_isaaclab_docker_ws.sh nvidia-smi
PROTO_DATASET_READONLY=0 ./scripts/docker/launch_isaaclab_docker_ws.sh shell
PROTO_FIX_OWNERSHIP=0 ./scripts/docker/launch_isaaclab_docker_ws.sh shell
PROTO_CHOWN_PATHS="results output wandb" ./scripts/docker/launch_isaaclab_docker_ws.sh shell
```

Use `./scripts/docker/launch_isaaclab_docker_ws.sh print-config` to inspect the resolved image,
repo, dataset, cache, and ownership-fix settings.

Inside the launched shell, run your own training commands directly:

```bash
/workspace/isaaclab/isaaclab.sh -p -u protomotions/train_agent.py \
  --robot-name astro \
  --simulator isaaclab \
  --num-envs 2048 \
  --batch-size 4096 \
  --motion-file datasets/dobot/astro/proto/amass-test.pt \
  --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --experiment-name astro-motion-tracker-amass-test
```

## 1. Prerequisites

Confirm the image exists locally:

```bash
docker image ls protomotions-isaaclab:2.3.0
```

Confirm Docker can see the NVIDIA runtime using the local training image:

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  protomotions-isaaclab:2.3.0
```

This should print the same GPU table as host `nvidia-smi`. If the literal image
tag works but a command using `"$IMAGE"` fails with `invalid reference format`,
set the variable in the current shell:

```bash
export IMAGE=protomotions-isaaclab:2.3.0
```

If the local-image `nvidia-smi` command fails, fix Docker GPU access before
debugging ProtoMotions. See the Docker GPU runtime troubleshooting section at
the end of this guide.

The training image was built from `Dockerfile.isaaclab` and contains Isaac Lab
plus the Python dependencies from `requirements_isaaclab.txt`. The full
ProtoMotions source is not baked into the image; mount the repo into the
container so the container uses the current checkout.

## 2. Common Docker Run Shape

Most commands use this shape:

```bash
cd "$REPO"

docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint <program> \
  "$IMAGE" \
  <args>
```

Flag notes:

- `--gpus all` exposes GPUs to the container.
- `--network=host` avoids networking surprises for logging and package caches.
- `--ipc=host` and `--shm-size=16g` reduce shared-memory issues during training.
- The EULA variables are required for Isaac Sim/Isaac Lab.
- The repo mount makes local code, datasets, assets, and results visible inside
  the container.
- The cache mount keeps Isaac/Python caches outside the container image.
- The image has an Isaac Sim startup entrypoint. Override it with
  `--entrypoint` for utility, shell, Python, and training commands.

Docker normally runs as root inside the container. Files written to mounted
folders can become root-owned on the host. The workstation launcher fixes
artifact ownership automatically after container exit. If you use the raw
`docker run` commands in this guide, fix ownership manually after a run:

```bash
sudo chown -R "$(id -u):$(id -g)" "$REPO/results" "$REPO/output" "$REPO/wandb"
```

## 3. Smoke Test the Image

Run this before training:

```bash
cd "$REPO"
mkdir -p "$HOME/.cache/protomotions-isaaclab"

docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions,readonly \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" \
  -p -c "import torch, lightning, tensordict; print(torch.__version__, torch.cuda.is_available())"
```

Expected result:

- The command exits with status 0.
- It prints a Torch version.
- It prints `True` for CUDA availability.

## 4. Launch an Interactive Container

Use an interactive shell when you want to inspect files or run commands by hand:

```bash
cd "$REPO"
mkdir -p "$HOME/.cache/protomotions-isaaclab"

docker run --rm -it --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /bin/bash \
  "$IMAGE"
```

Inside the container, verify the Python environment:

```bash
/workspace/isaaclab/isaaclab.sh -p -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
```

## 5. Fresh Astro Isaac Lab Training

This starts a fresh run if `results/astro-motion-tracker-amass-test` does not
already contain `last.ckpt`.

```bash
cd "$REPO"
mkdir -p "$HOME/.cache/protomotions-isaaclab"

docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" \
  -p -u protomotions/train_agent.py \
    --robot-name astro \
    --simulator isaaclab \
    --num-envs 2048 \
    --batch-size 4096 \
    --motion-file datasets/dobot/astro/proto/amass-test.pt \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name astro-motion-tracker-amass-test
```

This command matches the saved configuration in
`results/astro-motion-tracker-amass-test/config.yaml`.

## 6. Checkpoint Modes

ProtoMotions has three training modes:

1. Fresh start: no `results/<experiment-name>/last.ckpt` exists, and no
   `--checkpoint` is passed.
2. True resume: `results/<experiment-name>/last.ckpt` exists. The code loads
   `results/<experiment-name>/config.yaml` and `resolved_configs.pt`.
3. Warm start: `--checkpoint /path/to/file.ckpt` is passed with a new
   experiment name that does not already have `last.ckpt`.

Important behavior:

- True resume is selected by the existing result folder and experiment name.
- For true resume, do not pass `--checkpoint`.
- During true resume, CLI overrides are ignored because saved configs are reused.
- Use warm start when you want old weights but new configs.
- Use a new experiment name for warm start so it does not accidentally become a
  true resume of an existing run.

## 7. True Resume from an Existing `last.ckpt`

To resume `results/astro-motion-tracker-amass-test/last.ckpt`, run the same
training command with the same `--experiment-name` and no `--checkpoint`.

```bash
cd "$REPO"
mkdir -p "$HOME/.cache/protomotions-isaaclab"

docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" \
  -p -u protomotions/train_agent.py \
    --robot-name astro \
    --simulator isaaclab \
    --num-envs 2048 \
    --batch-size 4096 \
    --motion-file datasets/dobot/astro/proto/amass-test.pt \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name astro-motion-tracker-amass-test
```

Expected log lines:

```text
RESUME: Found checkpoint in save_dir: results/astro-motion-tracker-amass-test/last.ckpt
Loading saved args from results/astro-motion-tracker-amass-test/config.yaml
Loading configs from results/astro-motion-tracker-amass-test/resolved_configs.pt
RESUME: Using exact configs from first run
```

## 8. Safe Resume Validation with a Scratch Copy

Use this before deploying to HPC if you want to prove the container can resume
without touching the original result directory.

Create the validation copy:

```bash
cd "$REPO"

VAL=/tmp/protomotions-isaaclab-resume-$(date +%Y%m%d-%H%M%S)
mkdir -p "$VAL/results" "$VAL/cache" "$VAL/logs"
rsync -a results/astro-motion-tracker-amass-test/ \
  "$VAL/results/astro-motion-tracker-amass-test/"
touch "$VAL/start.marker"
echo "$VAL"
```

Run resume against the scratch result directory:

```bash
set -o pipefail

timeout 20m docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions,readonly \
  --mount type=bind,src="$VAL/results/astro-motion-tracker-amass-test",dst=/workspace/protomotions/results/astro-motion-tracker-amass-test \
  --mount type=bind,src="$VAL/cache",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" \
  -p -u protomotions/train_agent.py \
    --robot-name astro \
    --simulator isaaclab \
    --num-envs 2048 \
    --batch-size 4096 \
    --motion-file datasets/dobot/astro/proto/amass-test.pt \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name astro-motion-tracker-amass-test \
  2>&1 | tee "$VAL/logs/resume.log"
```

Inspect the log:

```bash
grep -E "RESUME|Loading configs|collecting data|Saved checkpoint" "$VAL/logs/resume.log"
```

Check files written after the validation started:

```bash
find "$VAL/results/astro-motion-tracker-amass-test" \
  -maxdepth 2 \
  -newer "$VAL/start.marker" \
  -type f \
  -print
```

If files are root-owned after the Docker run, fix the scratch directory:

```bash
sudo chown -R "$(id -u):$(id -g)" "$VAL"
```

## 9. Warm Start from an Arbitrary `*.ckpt`

Warm start is for using model weights from a checkpoint while creating a new
run with new configs. Use a new experiment name.

```bash
cd "$REPO"
mkdir -p "$HOME/.cache/protomotions-isaaclab"

docker run --rm --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  --mount type=bind,src="$REPO",dst=/workspace/protomotions \
  --mount type=bind,src="$HOME/.cache/protomotions-isaaclab",dst=/root/.cache \
  -w /workspace/protomotions \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" \
  -p -u protomotions/train_agent.py \
    --robot-name astro \
    --simulator isaaclab \
    --num-envs 2048 \
    --batch-size 4096 \
    --motion-file datasets/dobot/astro/proto/amass-test.pt \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name astro-motion-tracker-warm-start-test \
    --checkpoint results/astro-motion-tracker-amass-test/last.ckpt
```

Expected log line:

```text
WARM START: Using checkpoint for initialization: results/astro-motion-tracker-amass-test/last.ckpt
```

Do not warm-start Astro from an unrelated robot checkpoint unless the robot
state, body layout, action size, and model shapes are known to be compatible.

## 10. Monitor Outputs

Training writes to:

```text
container: /workspace/protomotions/results/<experiment-name>/
host:      /home/android/Code/NVlabs/ProtoMotions/results/<experiment-name>/
```

Important files:

```text
config.yaml                  # Saved CLI args and wandb id
resolved_configs.pt          # Pickled training configs used by true resume
resolved_configs.yaml        # Human-readable config dump
last.ckpt                    # Latest checkpoint
score_based.ckpt             # Best checkpoint by evaluator score
epoch_<N>.ckpt               # Periodic checkpoints
env_<task_id>.ckpt           # Environment state for exact resume
lightning_logs/              # TensorBoard logs
```

Watch logs from a running Docker command with the terminal output. For a
completed or copied validation run, inspect:

```bash
find results/astro-motion-tracker-amass-test -maxdepth 2 -type f -name "*.ckpt" -ls
find results/astro-motion-tracker-amass-test/lightning_logs -type f -ls
```

## 11. Convert the Image for HPC

The local Docker image is an intermediate artifact. Convert it to the format
required by the cluster container runtime.

For Enroot/Pyxis:

```bash
enroot import -o protomotions-isaaclab-2.3.0.sqsh \
  dockerd://protomotions-isaaclab:2.3.0
```

For Apptainer/Singularity:

```bash
apptainer build protomotions-isaaclab-2.3.0.sif \
  docker-daemon://protomotions-isaaclab:2.3.0
```

After copying the resulting image to the cluster, update
`CONTAINER_IMAGES["isaaclab"]` in `protomotions/train_slurm.py` to the cluster
path for the `.sqsh` or `.sif` file.

Also confirm cluster mounts include the paths needed for:

- experiment outputs
- motion files
- datasets
- robot assets
- checkpoint directories

## 12. Troubleshooting

### Docker Cannot Discover the GPU

If host `nvidia-smi` works but Docker fails before the container starts with an
error like:

```text
failed to discover GPU vendor from CDI: no known GPU vendor found
```

the image is not the problem. Docker cannot find NVIDIA GPU devices through the
host NVIDIA Container Toolkit.

First check whether the NVIDIA Container Toolkit CLI is installed:

```bash
command -v nvidia-ctk
nvidia-ctk --version
```

If `nvidia-ctk` is missing, install NVIDIA Container Toolkit on the host. On
Ubuntu or Debian, NVIDIA's installation guide uses this flow:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.1-1
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

Then configure Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If `nvidia-ctk` exists, regenerate the CDI GPU specification:

```bash
sudo systemctl restart nvidia-cdi-refresh.service || true
sudo mkdir -p /var/run/cdi
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
nvidia-ctk cdi list
```

`nvidia-ctk cdi list` should include entries like:

```text
nvidia.com/gpu=all
nvidia.com/gpu=0
```

Then test CDI explicitly:

```bash
docker run --rm \
  --device nvidia.com/gpu=all \
  --entrypoint nvidia-smi \
  protomotions-isaaclab:2.3.0
```

If CDI is still not working, test the legacy NVIDIA runtime path:

```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --entrypoint nvidia-smi \
  protomotions-isaaclab:2.3.0
```

Once either GPU smoke test works, use the same GPU exposure style in the
longer training commands. For CDI, replace `--gpus all` with
`--device nvidia.com/gpu=all`. For the legacy runtime, replace `--gpus all`
with `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`.

If Isaac Lab imports fail, make sure the command uses:

```bash
/workspace/isaaclab/isaaclab.sh -p
```

If a true resume ignores CLI overrides, that is expected. The saved
`resolved_configs.pt` is used for reproducibility.

If training resumes when you expected warm start, remove or rename the existing
`results/<experiment-name>/last.ckpt`, or use a new experiment name.

If files become root-owned on the host, run:

```bash
sudo chown -R "$(id -u):$(id -g)" results/<experiment-name>
```
