import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "launch_newton_docker_ws.sh"


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _base_env(tmp_path: Path, *, gpus: str = "all") -> dict[str, str]:
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

    dataset_root = tmp_path / "motion_lib"
    dataset_root.mkdir()
    cache_dir = tmp_path / "cache"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROTO_DATASET_ROOT": str(dataset_root),
            "PROTO_NEWTON_CACHE": str(cache_dir),
            "PROTO_FIX_OWNERSHIP": "0",
            "PROTO_GPUS": gpus,
        }
    )
    return env


def _docker_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.removeprefix("<arg>") for line in result.stdout.splitlines()]


def test_newton_ws_launcher_mounts_repo_dataset_and_cache_for_shell(tmp_path):
    env = _base_env(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert "all" in args
    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert f"type=bind,src={tmp_path}/motion_lib,dst={tmp_path}/motion_lib,readonly" in args
    assert f"type=bind,src={tmp_path}/cache,dst=/root/.cache" in args
    assert "PYTHONPATH=/workspace/protomotions:/workspace/protomotions/protomotions" in args
    assert "-w" in args
    assert "/workspace/protomotions" in args
    assert "--entrypoint" in args
    assert "/bin/bash" in args
    assert "protomotions-newton:cuda12.4-newton1.0.0" in args


def test_newton_ws_launcher_can_select_specific_normal_docker_gpu(tmp_path):
    env = _base_env(tmp_path, gpus="6")

    result = subprocess.run(
        ["bash", str(SCRIPT), "nvidia-smi"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert "device=6" in args


def test_newton_ws_launcher_quotes_normal_docker_multi_gpu_request(tmp_path):
    env = _base_env(tmp_path, gpus="4,5,6,7")

    result = subprocess.run(
        ["bash", str(SCRIPT), "nvidia-smi"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)

    assert "--gpus" in args
    assert '"device=4,5,6,7"' in args


def test_newton_ws_launcher_smoke_runs_direct_python_import_checks(tmp_path):
    env = _base_env(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "smoke"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)
    joined_args = "\n".join(args)

    assert "--entrypoint" in args
    assert "python" in args
    assert "import sys, torch, newton, mujoco, mujoco_warp" in joined_args
    assert "torch.cuda.is_available()" in joined_args


def test_newton_ws_launcher_train_debug_uses_newton_training_defaults(tmp_path):
    env = _base_env(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "train-debug", "--training-max-steps", "4096"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)
    joined_args = "\n".join(args)

    assert "--entrypoint" in args
    assert "python" in args
    assert "-u" in args
    assert "-m" in args
    assert "protomotions.train_agent" in args
    assert "--robot-name" in args
    assert "g1" in args
    assert "--simulator" in args
    assert "newton" in args
    assert "--motion-file" in args
    assert "data/motion_for_trackers/g1_random_subset_tiny.pt" in args
    assert "--experiment-path" in args
    assert "examples/experiments/mimic/mlp.py" in args
    assert "--experiment-name" in args
    assert "newton-docker-debug" in args
    assert "--training-max-steps" in args
    assert "4096" in args
    assert "protomotions-newton:cuda12.4-newton1.0.0" in joined_args


def test_newton_ws_launcher_repairs_artifact_ownership_through_container(tmp_path):
    env = _base_env(tmp_path)
    arg_log = tmp_path / "docker_args.log"
    env.update(
        {
            "DOCKER_ARG_LOG": str(arg_log),
            "PROTO_FIX_OWNERSHIP": "1",
            "PROTO_CHOWN_PATHS": "protomotions/tests",
        }
    )

    subprocess.run(
        ["bash", str(SCRIPT), "shell"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    logged_args = [
        line.removeprefix("<arg>") for line in arg_log.read_text().splitlines()
    ]

    assert "--entrypoint" in logged_args
    assert "/bin/chown" in logged_args
    assert "-R" in logged_args
    assert "1000:1000" in logged_args
    assert "/workspace/protomotions/protomotions/tests" in logged_args
