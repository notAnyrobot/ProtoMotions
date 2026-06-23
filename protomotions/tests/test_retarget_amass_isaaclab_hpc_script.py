import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SCRIPT = REPO_ROOT / "scripts" / "docker" / "retarget_amass_hpc.sh"
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "docker" / "retarget_amass_isaaclab_hpc.sh"


def _write_executable(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)
    return path


def _make_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    command_log = tmp_path / "commands.log"
    proto_python = _write_executable(
        tmp_path / "proto-python" / "bin" / "python",
        f"#!/usr/bin/env bash\nprintf 'PROTO %s\\n' \"$*\" >> {command_log}\n",
    )
    pyroki_python = _write_executable(
        tmp_path / "pyroki-python" / "bin" / "python",
        f"#!/usr/bin/env bash\nprintf 'PYROKI PYTHONPATH=%s ARGS=%s\\n' \"$PYTHONPATH\" \"$*\" >> {command_log}\n",
    )
    pyroki_repo = tmp_path / "pyroki"
    (pyroki_repo / "src" / "pyroki").mkdir(parents=True)
    (pyroki_repo / "batch_retarget_from_keypoints.py").touch()

    motion_datasets = tmp_path / "motion_datasets"
    data_root = motion_datasets / "protomotions"

    env = os.environ.copy()
    env.update(
        {
            "MOTION_DATASETS": str(motion_datasets),
            "PROTO_PYTHON": str(proto_python),
            "PYROKI_PYTHON": str(pyroki_python),
            "PROTO_PYROKI_REPO": str(pyroki_repo),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "MPLCONFIGDIR": str(tmp_path / "mpl-cache"),
        }
    )
    return env, data_root, command_log


def test_isaaclab_hpc_retarget_script_defaults_to_test_astro(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    amass_file = data_root / "smpl" / "amass_smpl_test.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Split:               test" in result.stdout
    assert "Robot:               astro" in result.stdout
    assert f"AMASS input:         {amass_file}" in result.stdout
    assert f"Output MotionLib: {data_root / 'astro' / 'test' / 'proto-astro.pt'}" in result.stdout

    log = command_log.read_text()
    smpl_split_dir = data_root / "smpl" / "test"
    robot_split_dir = data_root / "astro" / "test"

    assert f"--output-path {smpl_split_dir / 'keypoints-for-retarget'}" in log
    assert "--skeleton-format smpl" in log
    assert "--skip-freq 1" in log
    assert "PYROKI PYTHONPATH=" in log
    assert f"{tmp_path / 'pyroki' / 'src'}" in log
    assert f"ARGS={tmp_path / 'pyroki' / 'batch_retarget_from_keypoints.py'}" in log
    assert "ARGS=pyroki/batch_retarget_from_keypoints.py" not in log
    assert "--robot-type astro" in log
    assert "--input-fps 30" in log
    assert "--chunk-long-motions" in log
    assert "--chunk-threshold-frames 900" in log
    assert "--chunk-size-frames 450" in log
    assert "--chunk-overlap-frames 60" in log
    assert f"--output-dir {robot_split_dir / 'pyroki-retargeted-astro'}" in log
    assert f"--contacts-dir {smpl_split_dir / 'contacts'}" in log
    assert f"--retargeted-motion-dir {robot_split_dir / 'pyroki-retargeted-astro'}" in log
    assert f"--output-dir {robot_split_dir / 'proto-astro'}" in log
    assert f"--output-file {robot_split_dir / 'proto-astro.pt'}" in log


def test_isaaclab_hpc_retarget_script_accepts_split_robot_and_skip_freq(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    amass_file = data_root / "smpl" / "validation" / "amass_smpl_validation.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "validation", "g1", "50"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Split:               validation" in result.stdout
    assert "Robot:               g1" in result.stdout
    assert "Skip freq:           50" in result.stdout
    assert f"AMASS input:         {amass_file}" in result.stdout
    assert f"Output MotionLib: {data_root / 'g1' / 'validation' / 'proto-g1.pt'}" in result.stdout

    log = command_log.read_text()
    smpl_split_dir = data_root / "smpl" / "validation"
    robot_split_dir = data_root / "g1" / "validation"

    assert f"--output-path {smpl_split_dir / 'keypoints-for-retarget'}" in log
    assert "--skip-freq 50" in log
    assert "--robot-type g1" in log
    assert f"--contacts-dir {smpl_split_dir / 'contacts'}" in log
    assert f"--motion-path {robot_split_dir / 'proto-g1'}" in log
    assert f"--output-file {robot_split_dir / 'proto-g1.pt'}" in log


def test_isaaclab_hpc_retarget_script_forwards_fps_and_chunk_overrides(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    env.update(
        {
            "PROTO_INPUT_FPS": "60",
            "PROTO_OUTPUT_FPS": "60",
            "PROTO_CHUNK_THRESHOLD_FRAMES": "1200",
            "PROTO_CHUNK_SIZE_FRAMES": "600",
            "PROTO_CHUNK_OVERLAP_FRAMES": "90",
        }
    )
    amass_file = data_root / "smpl" / "train" / "amass_smpl_train.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "train", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Input FPS:           60" in result.stdout
    assert "Output FPS:          60" in result.stdout
    assert "Chunk threshold:     1200" in result.stdout
    assert "Chunk size:          600" in result.stdout
    assert "Chunk overlap:       90" in result.stdout

    log = command_log.read_text()
    assert "--input-fps 60" in log
    assert "--output-fps 60" in log
    assert "--chunk-long-motions" in log
    assert "--chunk-threshold-frames 1200" in log
    assert "--chunk-size-frames 600" in log
    assert "--chunk-overlap-frames 90" in log


def test_retarget_hpc_script_accepts_dataset_subset_split(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Split:               sfu" in result.stdout
    assert "Robot:               astro" in result.stdout
    assert f"AMASS input:         {amass_file}" in result.stdout
    assert f"Output MotionLib: {data_root / 'astro' / 'sfu' / 'proto-astro.pt'}" in result.stdout

    log = command_log.read_text()
    smpl_split_dir = data_root / "smpl" / "sfu"
    robot_split_dir = data_root / "astro" / "sfu"

    assert f"--output-path {smpl_split_dir / 'keypoints-for-retarget'}" in log
    assert f"--keypoints-folder-path {smpl_split_dir / 'keypoints-for-retarget'}" in log
    assert f"--contacts-dir {smpl_split_dir / 'contacts'}" in log
    assert f"--retargeted-motion-dir {robot_split_dir / 'pyroki-retargeted-astro'}" in log
    assert f"--motion-path {robot_split_dir / 'proto-astro'}" in log
    assert f"--output-file {robot_split_dir / 'proto-astro.pt'}" in log


def test_retarget_hpc_script_uses_python_when_proto_python_is_unset(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    env.pop("PROTO_PYTHON")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "python",
        f"#!/usr/bin/env bash\nprintf 'PROTO-FALLBACK %s\\n' \"$*\" >> {command_log}\n",
    )
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Proto Python:        python" in result.stdout

    log = command_log.read_text()
    assert "PROTO-FALLBACK data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py" in log
    assert "PROTO-FALLBACK data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py" in log
    assert "PROTO-FALLBACK protomotions/components/motion_lib.py" in log


def test_retarget_hpc_script_uses_explicit_isaaclab_launcher(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    env.pop("PROTO_PYTHON")
    launcher = _write_executable(
        tmp_path / "isaaclab" / "isaaclab.sh",
        f"#!/usr/bin/env bash\nprintf 'ISAACLAB %s\\n' \"$*\" >> {command_log}\n",
    )
    env["PROTO_ISAACLAB_LAUNCHER"] = str(launcher)

    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"Proto Python:        {launcher} -p" in result.stdout

    log = command_log.read_text()
    assert "ISAACLAB -p data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py" in log
    assert "ISAACLAB -p data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py" in log
    assert "ISAACLAB -p protomotions/components/motion_lib.py" in log


def test_retarget_hpc_script_falls_back_to_repo_local_pyroki_batch_cli(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    (tmp_path / "pyroki" / "batch_retarget_from_keypoints.py").unlink()
    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    repo_local_batch_cli = REPO_ROOT / "pyroki" / "batch_retarget_from_keypoints.py"

    assert result.returncode == 0
    assert f"PyRoki batch CLI:    {repo_local_batch_cli}" in result.stdout

    log = command_log.read_text()
    assert f"ARGS={repo_local_batch_cli}" in log
    assert f"PYROKI PYTHONPATH={tmp_path / 'pyroki' / 'src'}" in log


def test_retarget_hpc_script_rebuilds_keypoints_and_contacts_every_run(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    keypoints_dir = data_root / "smpl" / "sfu" / "keypoints-for-retarget"
    contacts_dir = data_root / "smpl" / "sfu" / "contacts"
    amass_file.parent.mkdir(parents=True)
    keypoints_dir.mkdir()
    contacts_dir.mkdir()
    amass_file.touch()

    stale_keypoints = keypoints_dir / "stale_keypoints.npy"
    stale_readme = keypoints_dir / "README.txt"
    stale_contact = contacts_dir / "stale_contacts.npy"
    stale_contact_note = contacts_dir / "README.txt"
    stale_keypoints.touch()
    stale_readme.touch()
    stale_contact.touch()
    stale_contact_note.touch()

    result = subprocess.run(
        ["bash", str(CANONICAL_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not stale_keypoints.exists()
    assert stale_readme.exists()
    assert not stale_contact.exists()
    assert stale_contact_note.exists()

    log = command_log.read_text()
    assert "--force-remake" in log
    assert f"--contacts-dir {contacts_dir}" in log


def test_legacy_isaaclab_hpc_script_delegates_to_canonical_script(tmp_path):
    env, data_root, command_log = _make_env(tmp_path)
    amass_file = data_root / "smpl" / "sfu" / "amass_smpl_sfu.pt"
    amass_file.parent.mkdir(parents=True)
    amass_file.touch()

    result = subprocess.run(
        ["bash", str(LEGACY_SCRIPT), "sfu", "astro", "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "deprecated" in result.stderr.lower()
    assert "retarget_amass_hpc.sh" in result.stderr
    assert f"Output MotionLib: {data_root / 'astro' / 'sfu' / 'proto-astro.pt'}" in result.stdout

    log = command_log.read_text()
    assert "--robot-type astro" in log
    assert f"--output-file {data_root / 'astro' / 'sfu' / 'proto-astro.pt'}" in log
