import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "download_retargeted_motion_from_hpc.sh"


def _write_executable(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _logged_args(path: Path) -> list[str]:
    return [line.removeprefix("<arg>") for line in path.read_text().splitlines()]


def test_download_retargeted_motion_requires_split():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--robot", "astro"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--split is required" in result.stderr


def test_download_retargeted_motion_defaults_to_astro_and_syncs_requested_split(tmp_path):
    rsync_log = tmp_path / "rsync_args.log"
    fake_rsync = _write_executable(
        tmp_path / "bin" / "rsync",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\" >> \"$RSYNC_ARG_LOG\"\n"
        "done\n",
    )
    local_root = tmp_path / "motion_datasets" / "protomotions"

    env = os.environ.copy()
    env.update(
        {
            "RSYNC_BIN": str(fake_rsync),
            "RSYNC_ARG_LOG": str(rsync_log),
            "REMOTE_HOST": "atom7@oem-WB-R5350-G6",
            "REMOTE_PROTOMOTIONS_ROOT": "/data/share/motion_datasets/protomotions",
            "LOCAL_PROTOMOTIONS_ROOT": str(local_root),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--split", "sfu"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Robot:               astro" in result.stdout
    assert "Split:               sfu" in result.stdout
    assert "Remote source:       atom7@oem-WB-R5350-G6:/data/share/motion_datasets/protomotions/astro/sfu/" in result.stdout
    assert f"Local destination:   {local_root / 'astro' / 'sfu'}/" in result.stdout

    args = _logged_args(rsync_log)
    assert "-az" in args
    assert "--partial" in args
    assert "--info=progress2" in args
    assert "atom7@oem-WB-R5350-G6:/data/share/motion_datasets/protomotions/astro/sfu/" in args
    assert f"{local_root / 'astro' / 'sfu'}/" in args
    assert "--delete" not in args
    assert (local_root / "astro" / "sfu").is_dir()


def test_download_retargeted_motion_accepts_explicit_robot_and_dry_run(tmp_path):
    rsync_log = tmp_path / "rsync_args.log"
    fake_rsync = _write_executable(
        tmp_path / "bin" / "rsync",
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf '<arg>%s\\n' \"$arg\" >> \"$RSYNC_ARG_LOG\"\n"
        "done\n",
    )
    local_root = tmp_path / "motion_datasets" / "protomotions"

    env = os.environ.copy()
    env.update(
        {
            "RSYNC_BIN": str(fake_rsync),
            "RSYNC_ARG_LOG": str(rsync_log),
            "REMOTE_HOST": "hpc-1",
            "REMOTE_PROTOMOTIONS_ROOT": "/remote/protomotions",
            "LOCAL_PROTOMOTIONS_ROOT": str(local_root),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--robot", "g1", "--split", "validation", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Robot:               g1" in result.stdout
    assert "Split:               validation" in result.stdout

    args = _logged_args(rsync_log)
    assert "--dry-run" in args
    assert "hpc-1:/remote/protomotions/g1/validation/" in args
    assert f"{local_root / 'g1' / 'validation'}/" in args
