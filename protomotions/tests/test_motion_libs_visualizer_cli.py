import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUALIZER_SCRIPT = REPO_ROOT / "examples" / "motion_libs_visualizer.py"


def test_motion_libs_visualizer_advertises_astro_robot_choice():
    result = subprocess.run(
        [sys.executable, str(VISUALIZER_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--robot {g1,rigv1,h1_2,smpl,soma23,astro}" in result.stdout


def test_motion_libs_visualizer_has_astro_robot_spec():
    module = ast.parse(VISUALIZER_SCRIPT.read_text())
    robot_specs = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ROBOT_SPECS"
            for target in node.targets
        )
    )

    robot_names = {
        key.value
        for key in robot_specs.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert "astro" in robot_names
