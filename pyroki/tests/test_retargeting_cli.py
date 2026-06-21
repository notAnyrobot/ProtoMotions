from dataclasses import replace
import importlib
from inspect import signature
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PYROKI_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(PYROKI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_SCRIPT_DIR))

from retargeting.cli import BatchRetargetingOptions, build_parser, main
from retargeting.factory import get_retarget_config
import retargeting.solver as solver


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
            "--chunk-long-motions",
            "--chunk-threshold-frames",
            "1200",
            "--chunk-size-frames",
            "600",
            "--chunk-overlap-frames",
            "90",
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
    assert args.chunk_long_motions is True
    assert args.chunk_threshold_frames == 1200
    assert args.chunk_size_frames == 600
    assert args.chunk_overlap_frames == 90


def test_parser_defaults_robot_type_to_g1(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
        ]
    )

    assert args.robot_type == "g1"


def test_parser_defaults_target_raw_frames_to_motion_length(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
        ]
    )

    assert args.target_raw_frames is None


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
        target_raw_frames=None,
        skip_existing=False,
        source_type="smpl",
        save_contacts_only=False,
        contacts_dir=None,
        input_fps=30.0,
        visualize=False,
        chunk_long_motions=False,
        chunk_threshold_frames=900,
        chunk_size_frames=450,
        chunk_overlap_frames=60,
    )


def test_main_uses_g1_config_when_robot_type_is_omitted(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, options):
        captured["config"] = config
        captured["options"] = options
        return 0

    monkeypatch.setattr("retargeting.cli.run_batch_retargeting", fake_run)

    result = main(
        [
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
            "--no-visualize",
        ]
    )

    assert result == 0
    assert captured["config"] == get_retarget_config("g1")
    assert captured["options"].keypoints_folder_path == tmp_path / "keypoints"


def test_main_rejects_invalid_chunk_options(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "pyroki/batch_retarget_from_keypoints.py",
            "--keypoints-folder-path",
            str(tmp_path / "keypoints"),
            "--chunk-long-motions",
            "--chunk-size-frames",
            "450",
            "--chunk-overlap-frames",
            "450",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert (
        "--chunk-overlap-frames must be smaller than --chunk-size-frames"
        in result.stderr
    )


def test_canonical_script_help_works_without_pyroki_runtime_import():
    result = subprocess.run(
        [sys.executable, "pyroki/batch_retarget_from_keypoints.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--robot-type {g1,h1_2,astro}" in result.stdout
    assert "--keypoints-folder-path" in result.stdout


def test_solve_retargeting_keeps_static_values_out_of_jit_signature():
    public_params = signature(solver.solve_retargeting).parameters
    assert {"config", "source_names", "link_names"}.issubset(public_params)

    helper_params = signature(solver._solve_retargeting_jit).parameters
    assert not {"config", "source_names", "link_names"} & set(helper_params)


def test_solve_retargeting_jit_factory_is_cached():
    assert hasattr(solver._solve_retargeting_jit, "cache_info")

    if solver._SOLVER_IMPORT_ERROR is None:
        assert solver._solve_retargeting_jit(800) is solver._solve_retargeting_jit(800)


def test_missing_solver_dependency_message_names_module():
    if solver._SOLVER_IMPORT_ERROR is None:
        pytest.skip("solver dependencies are installed")

    missing_name = solver._SOLVER_IMPORT_ERROR.name
    with pytest.raises(ImportError, match=missing_name):
        solver._solve_retargeting_jit(800)


def test_canonical_cli_contacts_only_writes_current_schema(tmp_path):
    keypoints_dir = tmp_path / "keypoints"
    contacts_dir = tmp_path / "contacts"
    keypoints_dir.mkdir()
    positions = np.zeros((3, 18, 3), dtype=np.float32)
    orientations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 3 * 18, axis=0)
    orientations = orientations.reshape(3, 18, 3, 3)
    np.save(
        keypoints_dir / "walk.npy",
        {
            "positions": positions,
            "orientations": orientations,
            "left_foot_contacts": np.array([[1, 1], [0, 0], [0, 0]], dtype=bool),
            "right_foot_contacts": np.array([[0, 0], [1, 1], [1, 1]], dtype=bool),
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "pyroki/batch_retarget_from_keypoints.py",
            "--robot-type",
            "g1",
            "--keypoints-folder-path",
            str(keypoints_dir),
            "--save-contacts-only",
            "--contacts-dir",
            str(contacts_dir),
            "--target-raw-frames",
            "3",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Running in save-contacts-only mode" in result.stdout
    saved = np.load(contacts_dir / "walk_contacts.npz")
    assert set(saved.files) == {"foot_contacts"}
    expected = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0],
        ]
    )
    np.testing.assert_allclose(saved["foot_contacts"], expected)


def test_g1_wrapper_help_warns_and_delegates_to_canonical_cli():
    result = subprocess.run(
        [sys.executable, "pyroki/batch_retarget_to_g1_from_keypoints.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "deprecated" in result.stderr.lower()
    assert "--robot-type {g1,h1_2,astro}" in result.stdout


def test_h1_2_wrapper_help_warns_and_delegates_to_canonical_cli():
    result = subprocess.run(
        [sys.executable, "pyroki/batch_retarget_to_h1_2_from_keypoints.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "deprecated" in result.stderr.lower()
    assert "--robot-type {g1,h1_2,astro}" in result.stdout


def test_g1_wrapper_robot_type_wins_over_conflicting_user_arg(monkeypatch):
    wrapper = importlib.import_module("batch_retarget_to_g1_from_keypoints")
    captured = {}

    def fake_canonical_main(argv):
        captured["argv"] = argv
        return 11

    monkeypatch.setattr(wrapper, "canonical_main", fake_canonical_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyroki/batch_retarget_to_g1_from_keypoints.py",
            "--robot-type",
            "h1_2",
            "--keypoints-folder-path",
            "keypoints",
        ],
    )

    with pytest.warns(FutureWarning, match="deprecated"):
        assert wrapper.main() == 11

    assert captured["argv"][-2:] == ["--robot-type", "g1"]


def test_h1_2_wrapper_robot_type_wins_over_conflicting_user_arg(monkeypatch):
    wrapper = importlib.import_module("batch_retarget_to_h1_2_from_keypoints")
    captured = {}

    def fake_canonical_main(argv):
        captured["argv"] = argv
        return 12

    monkeypatch.setattr(wrapper, "canonical_main", fake_canonical_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyroki/batch_retarget_to_h1_2_from_keypoints.py",
            "--robot-type",
            "g1",
            "--keypoints-folder-path",
            "keypoints",
        ],
    )

    with pytest.warns(FutureWarning, match="deprecated"):
        assert wrapper.main() == 12

    assert captured["argv"][-2:] == ["--robot-type", "h1_2"]
