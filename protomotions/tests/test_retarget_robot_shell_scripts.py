import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _touch_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_single_motion_wrapper_accepts_astro_and_pyroki_env_dir(tmp_path):
    proto_python = _touch_python(tmp_path / "proto" / "bin" / "python")
    pyroki_env = tmp_path / "pyroki"
    _touch_python(pyroki_env / "bin" / "python")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_single_motion_to_robot.sh"),
            str(proto_python),
            str(pyroki_env),
            str(tmp_path / "missing.motion"),
            str(tmp_path / "out"),
            "astro",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: Motion file not found:" in result.stdout
    assert "robot_type must" not in result.stdout
    assert "PyRoki Python not found" not in result.stdout


def test_amass_wrapper_accepts_astro_and_pyroki_env_dir(tmp_path):
    proto_python = _touch_python(tmp_path / "proto" / "bin" / "python")
    pyroki_env = tmp_path / "pyroki"
    _touch_python(pyroki_env / "bin" / "python")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_amass_to_robot.sh"),
            str(proto_python),
            str(pyroki_env),
            str(tmp_path / "missing.pt"),
            "astro",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: AMASS .pt file not found:" in result.stdout
    assert "robot_type must" not in result.stdout
    assert "PyRoki Python not found" not in result.stdout
