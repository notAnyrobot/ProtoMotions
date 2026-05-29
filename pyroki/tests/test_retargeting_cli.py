from dataclasses import replace
from pathlib import Path
import subprocess
import sys

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

from retargeting.cli import BatchRetargetingOptions, build_parser, main
from retargeting.factory import get_retarget_config


def test_parser_accepts_existing_args_plus_robot_type(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--robot-type",
            "g1",
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-type",
            "smpl",
            "--subsample-factor",
            "2",
            "--target-raw-frames",
            "60",
            "--skip-existing",
            "--save-contacts-only",
            "--contacts-dir",
            str(tmp_path / "contacts"),
            "--input-fps",
            "60",
            "--no-visualize",
        ]
    )

    assert args.robot_type == "g1"
    assert args.keypoints_folder_path == str(tmp_path / "keypoints")
    assert args.output_dir == str(tmp_path / "out")
    assert args.source_type == "smpl"
    assert args.subsample_factor == 2
    assert args.target_raw_frames == 60
    assert args.skip_existing is True
    assert args.save_contacts_only is True
    assert args.contacts_dir == str(tmp_path / "contacts")
    assert args.input_fps == 60.0
    assert args.visualize is False


def test_main_builds_config_and_options(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, options):
        captured["config"] = config
        captured["options"] = options
        return 7

    monkeypatch.setattr("retargeting.cli.run_batch_retargeting", fake_run)

    result = main(
        [
            "--robot-type",
            "h1_2",
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
            "--output-dir",
            str(tmp_path / "out"),
            "--urdf-path",
            str(tmp_path / "robot.urdf"),
            "--mesh-dir",
            str(tmp_path / "mesh"),
            "--no-visualize",
        ]
    )

    assert result == 7
    assert captured["config"] == replace(
        get_retarget_config("h1_2"),
        urdf_path=tmp_path / "robot.urdf",
        mesh_dir=tmp_path / "mesh",
    )
    assert captured["options"] == BatchRetargetingOptions(
        keypoints_folder_path=tmp_path / "keypoints",
        output_dir=tmp_path / "out",
        subsample_factor=1,
        target_raw_frames=450,
        skip_existing=False,
        source_type="smpl",
        save_contacts_only=False,
        contacts_dir=None,
        input_fps=30.0,
        visualize=False,
    )


def test_canonical_script_help_works_without_pyroki_runtime_import():
    result = subprocess.run(
        [sys.executable, "pyroki/batch_retarget_from_keypoints.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--robot-type {g1,h1_2}" in result.stdout
    assert "--keypoints-folder-path" in result.stdout
