import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "launch_newton_docker_ws.sh"
IMAGE = "protomotions-newton-pyroki:cuda12.4-newton1.0.0-pyroki59aa21f"
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
]


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

    dataset_root = tmp_path / "motion_datasets"
    dataset_root.mkdir(parents=True)
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


def _add_pyroki_repo(env: dict[str, str], tmp_path: Path) -> Path:
    pyroki_repo = tmp_path / "pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    (pyroki_repo / "src" / "pyroki" / "__init__.py").write_text("")
    env["PROTO_PYROKI_REPO"] = str(pyroki_repo)
    return pyroki_repo


def _docker_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.removeprefix("<arg>") for line in result.stdout.splitlines()]


def test_newton_ws_launcher_help_is_shell_only():
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
    assert "Default dataset:     /media/android/data/motion_datasets" in lines

    for command in REMOVED_COMMANDS:
        assert command not in result.stdout


def test_newton_ws_launcher_rejects_removed_commands(tmp_path):
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


def test_newton_ws_launcher_defaults_to_shell(tmp_path):
    env = _base_env(tmp_path)
    pyroki_repo = _add_pyroki_repo(env, tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _docker_args(result)
    dataset_root = Path(env["PROTO_DATASET_ROOT"])

    assert "--gpus" in args
    assert "all" in args
    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert f"type=bind,src={pyroki_repo},dst=/workspace/pyroki" in args
    assert f"type=bind,src={dataset_root},dst={dataset_root},readonly" in args
    assert f"type=bind,src={tmp_path}/cache,dst=/root/.cache" in args
    assert "PYTHONPATH=/workspace/pyroki/src:/workspace/protomotions:/workspace/protomotions/protomotions" in args
    assert "JAX_PLATFORMS=cuda,cpu" in args
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in args
    assert "-w" in args
    assert "/workspace/protomotions" in args
    assert "--entrypoint" in args
    assert "/bin/bash" in args
    assert IMAGE in args


def test_newton_ws_launcher_can_select_specific_normal_docker_gpu(tmp_path):
    env = _base_env(tmp_path, gpus="6")
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


def test_newton_ws_launcher_quotes_normal_docker_multi_gpu_request(tmp_path):
    env = _base_env(tmp_path, gpus="4,5,6,7")
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
    assert '"device=4,5,6,7"' in args


def test_newton_ws_launcher_repairs_repo_artifact_ownership_through_container(tmp_path):
    env = _base_env(tmp_path)
    _add_pyroki_repo(env, tmp_path)
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
    assert f"{os.getuid()}:{os.getgid()}" in logged_args
    assert "/workspace/protomotions/protomotions/tests" in logged_args
