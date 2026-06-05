import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "sync_to_hpc.sh"


def _make_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _logged_args(path: Path) -> list[str]:
    return [line.removeprefix("<arg>") for line in path.read_text().splitlines()]


def test_sync_to_hpc_push_uses_repo_root_from_scripts_docker_location(tmp_path):
    fake_bin = tmp_path / "bin"
    rsync_log = tmp_path / "rsync_args.log"

    _make_executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\n"
        "exit 0\n",
    )
    _make_executable(
        fake_bin / "rsync",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\" >> \"$RSYNC_ARG_LOG\"\n"
        "done\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RSYNC_ARG_LOG": str(rsync_log),
            "REMOTE_HOST": "hpc.example",
            "REMOTE_ROOT": "/remote/ProtoMotions",
        }
    )

    subprocess.run(
        ["bash", str(SCRIPT), "push", "--dry-run"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _logged_args(rsync_log)

    assert f"{REPO_ROOT}/" in args
    assert f"{REPO_ROOT / 'scripts'}/" not in args


def test_sync_to_hpc_push_excludes_artifact_roots(tmp_path):
    fake_bin = tmp_path / "bin"
    rsync_log = tmp_path / "rsync_args.log"

    _make_executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\n"
        "exit 0\n",
    )
    _make_executable(
        fake_bin / "rsync",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\" >> \"$RSYNC_ARG_LOG\"\n"
        "done\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RSYNC_ARG_LOG": str(rsync_log),
            "REMOTE_HOST": "hpc.example",
            "REMOTE_ROOT": "/remote/ProtoMotions",
        }
    )

    subprocess.run(
        ["bash", str(SCRIPT), "push", "--dry-run"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = _logged_args(rsync_log)

    for artifact_root in ("results/", "output/", "outputs/", "wandb/", "exps/"):
        assert f"--exclude={artifact_root}" in args
