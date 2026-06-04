import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "launch_newton_docker_hpc.sh"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _base_env(tmp_path: Path, *, gpu_mode: str = "gpus", gpus: str = "all") -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    _make_executable(
        fake_bin / "docker",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\"\n"
        "done\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROTO_HPC_GPU_MODE": gpu_mode,
            "PROTO_HPC_GPUS": gpus,
            "PROTO_NEWTON_CACHE": str(tmp_path / "newton_cache"),
            "PROTO_FIX_OWNERSHIP": "0",
        }
    )
    return env


def _manual_gpu_env(tmp_path: Path, *, gpus: str = "0") -> dict[str, str]:
    env = _base_env(tmp_path, gpu_mode="manual", gpus=gpus)

    dev_dir = tmp_path / "dev"
    for name in ["nvidia0", "nvidia1", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"]:
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


def test_newton_hpc_launcher_uses_normal_docker_gpus_and_repo_mount(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=_base_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert "all" in args
    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert f"{tmp_path}/newton_cache:/root/.cache" in args
    assert "PYTHONPATH=/workspace/protomotions:/workspace/protomotions/protomotions" in args
    assert "--entrypoint" in args
    assert "/bin/bash" in args
    assert "protomotions-newton:cuda12.4-newton1.0.0" in args


def test_newton_hpc_launcher_can_select_specific_normal_docker_gpu(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "nvidia-smi"],
        env=_base_env(tmp_path, gpus="6"),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert "device=6" in args


def test_newton_hpc_launcher_supports_manual_gpu_passthrough(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "smoke"],
        env=_manual_gpu_env(tmp_path, gpus="1"),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" not in args
    assert f"--device={tmp_path}/dev/nvidia1" in args
    assert f"--device={tmp_path}/dev/nvidia0" not in args
    assert f"--device={tmp_path}/dev/nvidiactl" in args
    assert f"{tmp_path}/nvidia-libs:/host-nvidia-libs:ro" in args
    assert f"{tmp_path}/nvidia-smi:/usr/bin/nvidia-smi:ro" in args
    assert "LD_LIBRARY_PATH=/host-nvidia-libs" in args


def test_newton_hpc_launcher_train_astro_debug_uses_finite_newton_command(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "train-astro-debug", "--training-max-steps", "131072"],
        env=_base_env(tmp_path, gpus="0"),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "python" in args
    assert "-u" in args
    assert "-m" in args
    assert "protomotions.train_agent" in args
    assert "--robot-name" in args
    assert "astro" in args
    assert "--simulator" in args
    assert "newton" in args
    assert "--motion-file" in args
    assert "data/motion_for_trackers/astro_amass-test.pt" in args
    assert "--experiment-path" in args
    assert "data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py" in args
    assert "--experiment-name" in args
    assert "astro-motion-tracker-newton-hpc-debug" in args
    assert "--training-max-steps" in args
    assert "65536" in args
    assert "131072" in args
