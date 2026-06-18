import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _touch_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_executable(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)
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


def test_single_motion_wrapper_uses_default_python_paths(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_single_motion_to_robot.sh"),
            str(tmp_path / "missing.motion"),
            str(tmp_path / "out"),
            "astro",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: Motion file not found:" in result.stdout
    assert f"ProtoMotions Python not found: {REPO_ROOT / '.venv_isaaclab' / 'bin' / 'python'}" not in result.stdout
    assert "PyRoki Python not found:" not in result.stdout


def test_single_motion_wrapper_sets_pyroki_repo_pythonpath(tmp_path):
    proto_python = _write_executable(
        tmp_path / "proto" / "bin" / "python",
        "#!/bin/bash\nexit 0\n",
    )
    env_log = tmp_path / "pyroki_env.log"
    pyroki_python = _write_executable(
        tmp_path / "pyroki-python" / "bin" / "python",
        f"#!/bin/bash\necho \"$PYTHONPATH\" >> {env_log}\nexit 0\n",
    )
    pyroki_repo = tmp_path / "external-pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    motion_file = tmp_path / "input.motion"
    motion_file.touch()

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_single_motion_to_robot.sh"),
            str(proto_python),
            str(pyroki_python),
            str(motion_file),
            str(tmp_path / "out"),
            "g1",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PROTO_PYROKI_REPO": str(pyroki_repo)},
    )

    assert result.returncode == 1
    assert env_log.read_text().splitlines() == [
        str(pyroki_repo / "src"),
        str(pyroki_repo / "src"),
    ]
    assert "Error: No output .motion file found" in result.stdout


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


def test_amass_wrapper_sets_pyroki_repo_pythonpath(tmp_path):
    proto_python = _write_executable(
        tmp_path / "proto" / "bin" / "python",
        "#!/bin/bash\nexit 0\n",
    )
    env_log = tmp_path / "pyroki_env.log"
    pyroki_python = _write_executable(
        tmp_path / "pyroki-python" / "bin" / "python",
        f"#!/bin/bash\necho \"$PYTHONPATH\" >> {env_log}\nexit 0\n",
    )
    pyroki_repo = tmp_path / "external-pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    amass_file = tmp_path / "protomotions" / "smpl" / "test" / "amass_smpl_test.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_amass_to_robot.sh"),
            str(proto_python),
            str(pyroki_python),
            str(amass_file),
            "g1",
            "50",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PROTO_PYROKI_REPO": str(pyroki_repo)},
    )

    assert result.returncode == 0
    assert env_log.read_text().splitlines() == [
        str(pyroki_repo / "src"),
        str(pyroki_repo / "src"),
    ]


def test_amass_wrapper_defaults_to_split_aware_robot_output_layout(tmp_path):
    command_log = tmp_path / "commands.log"
    proto_python = _write_executable(
        tmp_path / "proto" / "bin" / "python",
        f"#!/bin/bash\nprintf 'PROTO %s\\n' \"$*\" >> {command_log}\nexit 0\n",
    )
    pyroki_python = _write_executable(
        tmp_path / "pyroki-python" / "bin" / "python",
        f"#!/bin/bash\nprintf 'PYROKI %s\\n' \"$*\" >> {command_log}\nexit 0\n",
    )
    pyroki_repo = tmp_path / "external-pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    amass_file = (
        tmp_path / "protomotions" / "smpl" / "train" / "amass_smpl_train.pt"
    )
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "retarget_amass_to_robot.sh"),
            str(proto_python),
            str(pyroki_python),
            str(amass_file),
            "astro",
            "1",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PROTO_PYROKI_REPO": str(pyroki_repo)},
    )

    assert result.returncode == 0
    smpl_split_dir = tmp_path / "protomotions" / "smpl" / "train"
    robot_split_dir = tmp_path / "protomotions" / "astro" / "train"
    log = command_log.read_text()

    assert f"--output-path {smpl_split_dir / 'keypoints-for-retarget'}" in log
    assert f"--contacts-dir {smpl_split_dir / 'contacts'}" in log
    assert f"--output-dir {robot_split_dir / 'pyroki-retargeted-astro'}" in log
    assert f"--retargeted-motion-dir {robot_split_dir / 'pyroki-retargeted-astro'}" in log
    assert f"--output-dir {robot_split_dir / 'proto-astro'}" in log
    assert f"--motion-path {robot_split_dir / 'proto-astro'}" in log
    assert f"--output-file {robot_split_dir / 'proto-astro.pt'}" in log
