import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "isaaclab_docker_hpc1.sh"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _base_env(tmp_path: Path, *, gpus: str = "0") -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    _make_executable(
        fake_bin / "docker",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\"\n"
        "done\n",
    )

    dev_dir = tmp_path / "dev"
    for name in ["nvidia0", "nvidia1", "nvidia2", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"]:
        _touch(dev_dir / name)

    libs_dir = tmp_path / "nvidia-libs"
    for name in ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]:
        _touch(libs_dir / name)

    nvidia_smi = _make_executable(tmp_path / "nvidia-smi", "#!/usr/bin/env bash\n")
    cache_dir = tmp_path / "isaac_cache"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROTO_HPC_GPUS": gpus,
            "PROTO_HPC_DEV_DIR": str(dev_dir),
            "PROTO_HPC_NVIDIA_LIBS": str(libs_dir),
            "PROTO_HPC_NVIDIA_SMI": str(nvidia_smi),
            "PROTO_ISAACLAB_CACHE": str(cache_dir),
            "PROTO_FIX_OWNERSHIP": "0",
        }
    )
    return env


def _docker_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.removeprefix("<arg>") for line in result.stdout.splitlines()]


def test_hpc1_launcher_resolves_repo_root_from_scripts_on_hpc_location():
    result = subprocess.run(
        ["bash", str(SCRIPT), "print-config"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"Repo:              {REPO_ROOT}" in result.stdout


def test_hpc1_launcher_uses_manual_gpu_passthrough_without_nvidia_runtime(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "nvidia-smi"],
        env=_base_env(tmp_path, gpus="0,2"),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" not in args
    assert "--runtime=nvidia" not in args
    assert "--device=nvidia.com/gpu=all" not in args
    assert f"--device={tmp_path}/dev/nvidia0" in args
    assert f"--device={tmp_path}/dev/nvidia2" in args
    assert f"--device={tmp_path}/dev/nvidia1" not in args
    assert f"--device={tmp_path}/dev/nvidiactl" in args
    assert f"--device={tmp_path}/dev/nvidia-uvm" in args
    assert f"--device={tmp_path}/dev/nvidia-uvm-tools" in args
    assert f"{tmp_path}/nvidia-libs:/host-nvidia-libs:ro" in args
    assert f"{tmp_path}/nvidia-smi:/usr/bin/nvidia-smi:ro" in args
    assert "LD_LIBRARY_PATH=/host-nvidia-libs" in args


def test_hpc1_launcher_mounts_repo_and_isaac_cache_for_shell(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=_base_env(tmp_path, gpus="all"),
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--entrypoint" in args
    assert "/bin/bash" in args
    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert "-w" in args
    assert "/workspace/protomotions" in args
    assert f"{tmp_path}/isaac_cache/kit:/root/.cache/ov/Kit" in args
    assert f"{tmp_path}/isaac_cache/ov:/root/.cache/ov" in args
    assert f"{tmp_path}/isaac_cache/pip:/root/.cache/pip" in args
    assert f"{tmp_path}/isaac_cache/glcache:/root/.cache/nvidia/GLCache" in args
    assert f"{tmp_path}/isaac_cache/computecache:/root/.nv/ComputeCache" in args
    assert f"{tmp_path}/isaac_cache/logs:/root/.nvidia-omniverse/logs" in args
    assert f"{tmp_path}/isaac_cache/data:/root/.local/share/ov/data" in args
    assert f"{tmp_path}/isaac_cache/documents:/root/Documents" in args
    assert "ACCEPT_EULA=Y" in args
    assert "NVIDIA_DRIVER_CAPABILITIES=all" in args
    assert "PRIVACY_CONSENT=N" in args
    assert "OMNI_KIT_ALLOW_ROOT=1" in args
    assert "protomotions-isaaclab:2.3.0" in args


def test_hpc1_launcher_supports_named_kept_container_for_exec_workflow(tmp_path):
    env = _base_env(tmp_path, gpus="1")
    env.update(
        {
            "PROTO_CONTAINER_NAME": "proto_hpc1",
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
    assert "proto_hpc1" in args
    assert "--network=host" in args
    assert "--ipc=host" in args


def test_hpc1_launcher_has_reset_probe_for_isaaclab_physx_runtime(tmp_path):
    env = _base_env(tmp_path, gpus="0")
    env["PROTO_RESET_TIMEOUT"] = "123"

    result = subprocess.run(
        ["bash", str(SCRIPT), "reset"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)
    joined_args = "\n".join(args)

    assert "/workspace/isaaclab/isaaclab.sh" in joined_args
    assert "timeout 123" in joined_args
    assert "SimulationContext" in joined_args
    assert "RESET_OK" in joined_args
