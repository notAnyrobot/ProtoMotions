import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "launch_isaaclab_docker_ws.sh"


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _docker_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.removeprefix("<arg>") for line in result.stdout.splitlines()]


def test_ws_launcher_mounts_repo_root_from_scripts_docker_location(tmp_path):
    fake_bin = tmp_path / "bin"
    _make_executable(
        fake_bin / "docker",
        "#!/usr/bin/env bash\n"
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
            "PROTO_ISAACLAB_CACHE": str(cache_dir),
            "PROTO_FIX_OWNERSHIP": "0",
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

    assert f"type=bind,src={REPO_ROOT},dst=/workspace/protomotions" in args
    assert "-w" in args
    assert "/workspace/protomotions" in args
