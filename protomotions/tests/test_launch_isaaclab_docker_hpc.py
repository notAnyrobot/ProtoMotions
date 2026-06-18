import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "launch_isaaclab_docker_hpc.sh"
HPC_DATASET_ROOT = "/data/share/motion_datasets"
IMAGE = "protomotions-isaaclab-pyroki:2.3.0-pyroki59aa21f"
REMOVED_COMMANDS = [
    "nvidia-smi",
    "smoke",
    "python",
    "run",
    "bash",
    "retarget-astro",
    "print-config",
    "pyroki-shell",
    "pyroki-smoke",
    "train-debug",
    "train-astro-debug",
    "reset",
]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _base_env(
    tmp_path: Path, *, gpu_mode: str | None = "gpus", gpus: str = "all"
) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    _make_executable(
        fake_bin / "docker",
        "#!/usr/bin/env bash\n"
        "if [ -n \"${DOCKER_ARG_LOG:-}\" ]; then\n"
        "  for arg in \"$@\"; do\n"
        "    printf '<arg>%s\\n' \"$arg\" >> \"$DOCKER_ARG_LOG\"\n"
        "  done\n"
        "fi\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\"\n"
        "done\n",
    )
    dataset_root = tmp_path / "motion_datasets"
    dataset_root.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROTO_HPC_GPUS": gpus,
            "PROTO_ISAACLAB_CACHE": str(tmp_path / "isaac_cache"),
            "PROTO_DATASET_ROOT": str(dataset_root),
            "PROTO_FIX_OWNERSHIP": "0",
        }
    )
    if gpu_mode is not None:
        env["PROTO_HPC_GPU_MODE"] = gpu_mode
    return env


def _add_pyroki_repo(env: dict[str, str], tmp_path: Path) -> Path:
    pyroki_repo = tmp_path / "pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    (pyroki_repo / "src" / "pyroki" / "__init__.py").write_text("")
    env["PROTO_PYROKI_REPO"] = str(pyroki_repo)
    return pyroki_repo


def _manual_gpu_env(
    tmp_path: Path, *, gpus: str = "0", gpu_mode: str | None = "manual"
) -> dict[str, str]:
    env = _base_env(tmp_path, gpu_mode=gpu_mode, gpus=gpus)

    dev_dir = tmp_path / "dev"
    for name in [
        "nvidia0",
        "nvidia1",
        "nvidia2",
        "nvidia3",
        "nvidia4",
        "nvidia5",
        "nvidia6",
        "nvidia7",
        "nvidiactl",
        "nvidia-uvm",
        "nvidia-uvm-tools",
    ]:
        _touch(dev_dir / name)

    libs_dir = tmp_path / "nvidia-libs"
    for name in ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]:
        _touch(libs_dir / name)

    nvidia_smi = _make_executable(tmp_path / "nvidia-smi", "#!/usr/bin/env bash\n")

    env.update(
        {
            "PROTO_HPC_DEV_DIR": str(dev_dir),
            "PROTO_HPC_NVIDIA_LIBS": str(libs_dir),
            "PROTO_HPC_NVIDIA_SMI": str(nvidia_smi),
        }
    )
    return env


def _docker_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.removeprefix("<arg>") for line in result.stdout.splitlines()]


def test_isaaclab_hpc_launcher_help_is_shell_only():
    result = subprocess.run(
        ["bash", str(SCRIPT), "help"],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()

    assert "shell" in result.stdout
    assert "help" in result.stdout
    assert "scripts/docker/README.md" in result.stdout
    assert f"Default image:       {IMAGE}" in lines
    assert f"Default dataset:     {HPC_DATASET_ROOT}" in lines
    assert "Default GPU mode:    manual" in lines
    assert "Default GPU choice:  all" in lines

    for command in REMOVED_COMMANDS:
        assert command not in result.stdout


def test_isaaclab_hpc_launcher_rejects_removed_commands(tmp_path):
    env = _base_env(tmp_path)
    _add_pyroki_repo(env, tmp_path)

    for command in REMOVED_COMMANDS:
        result = subprocess.run(
            ["bash", str(SCRIPT), command],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert f"Unknown command: {command}" in result.stderr


def test_isaaclab_hpc_launcher_defaults_to_manual_gpu_passthrough_for_shell(
    tmp_path,
):
    env = _manual_gpu_env(tmp_path, gpu_mode=None)
    pyroki_repo = _add_pyroki_repo(env, tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)
    dataset_root = tmp_path / "motion_datasets"

    assert "--gpus" not in args
    assert "--runtime=nvidia" not in args
    assert f"--device={tmp_path}/dev/nvidia0" in args
    assert f"--device={tmp_path}/dev/nvidiactl" in args
    assert f"{tmp_path}/nvidia-libs:/host-nvidia-libs:ro" in args
    assert f"{tmp_path}/nvidia-smi:/usr/bin/nvidia-smi:ro" in args
    assert (
        "LD_LIBRARY_PATH=/workspace/pyroki-venv/lib/python3.12/site-packages/"
        "nvidia/cudnn/lib:/host-nvidia-libs" in args
    )
    assert "CUDA_VISIBLE_DEVICES=0" in args
    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert f"type=bind,src={pyroki_repo},dst=/workspace/pyroki" in args
    assert f"type=bind,src={dataset_root},dst={dataset_root},readonly" in args
    assert f"{tmp_path}/isaac_cache/kit:/root/.cache/ov/Kit" in args
    assert f"{tmp_path}/isaac_cache/pip:/root/.cache/pip" in args
    assert "PYTHONPATH=/workspace/pyroki/src:/workspace/protomotions:/workspace/protomotions/protomotions" in args
    assert "PYROKI_PYTHON=/workspace/pyroki-venv/bin/python" in args
    assert not any(arg.startswith("PYROKI_JAX_CUDA_LIB=") for arg in args)
    assert "JAX_PLATFORMS=cuda,cpu" in args
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in args
    assert "--entrypoint" in args
    assert "/bin/bash" in args
    assert IMAGE in args


def test_isaaclab_hpc_launcher_uses_normal_docker_gpus_when_requested(tmp_path):
    env = _base_env(tmp_path, gpu_mode="gpus", gpus="6")
    _add_pyroki_repo(env, tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert "device=6" in args


def test_isaaclab_hpc_launcher_supports_manual_multi_gpu_subset(tmp_path):
    env = _manual_gpu_env(tmp_path, gpus="4,5,6,7")
    _add_pyroki_repo(env, tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" not in args
    for selected in ["4", "5", "6", "7"]:
        assert f"--device={tmp_path}/dev/nvidia{selected}" in args
    for busy in ["0", "1", "2", "3"]:
        assert f"--device={tmp_path}/dev/nvidia{busy}" not in args
    assert f"--device={tmp_path}/dev/nvidiactl" in args
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in args


def test_isaaclab_hpc_launcher_accepts_proto_gpus_alias_for_manual_subset(tmp_path):
    env = _manual_gpu_env(tmp_path, gpus="all")
    _add_pyroki_repo(env, tmp_path)
    env.pop("PROTO_HPC_GPUS")
    env["PROTO_GPUS"] = "5,6,7"

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    for selected in ["5", "6", "7"]:
        assert f"--device={tmp_path}/dev/nvidia{selected}" in args
    assert f"--device={tmp_path}/dev/nvidia4" not in args
    assert "CUDA_VISIBLE_DEVICES=0,1,2" in args


def test_isaaclab_hpc_launcher_supports_named_kept_container_for_exec_workflow(
    tmp_path,
):
    env = _manual_gpu_env(tmp_path, gpus="1")
    _add_pyroki_repo(env, tmp_path)
    env.update(
        {
            "PROTO_CONTAINER_NAME": "proto_hpc",
            "PROTO_DOCKER_RM": "0",
            "PROTO_HPC_NETWORK_MODE": "host",
            "PROTO_HPC_IPC_MODE": "host",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--rm" not in args
    assert "--name" in args
    assert "proto_hpc" in args
    assert "--network=host" in args
    assert "--ipc=host" in args
