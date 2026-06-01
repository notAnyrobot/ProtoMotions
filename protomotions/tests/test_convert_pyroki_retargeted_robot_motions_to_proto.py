from pathlib import Path
import sys


DATA_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "data" / "scripts"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))

from convert_pyroki_retargeted_robot_motions_to_proto import main


def test_converter_accepts_astro_robot_config_asset(tmp_path):
    main(
        retargeted_motion_dir=tmp_path / "retargeted",
        output_dir=tmp_path / "proto",
        input_fps=30,
        output_fps=30,
        force_remake=False,
        ignore_first_n_frames=0,
        apply_motion_filter=False,
        min_height_threshold=-0.05,
        max_velocity_threshold=15.0,
        max_dof_vel_threshold=40.0,
        duration_height_filter=0.1,
        duration_height_seconds=0.6,
        robot_type="astro",
        contact_labels_dir=None,
    )
