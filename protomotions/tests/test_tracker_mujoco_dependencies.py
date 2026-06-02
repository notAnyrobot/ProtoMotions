import builtins
import importlib
import sys

import pytest


MINIMAL_FREE_ROOT_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <freejoint/>
      <geom type="sphere" size="0.05" group="1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" group="3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_tracker_mujoco_imports_without_onnxruntime(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    pytest.importorskip("yaml")

    monkeypatch.delitem(sys.modules, "deployment.test_tracker_mujoco", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)

    real_import = builtins.__import__

    def import_without_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise ImportError("onnxruntime intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_onnxruntime)

    module = importlib.import_module("deployment.test_tracker_mujoco")

    assert hasattr(module, "_parse_args")


def test_reference_ghost_defaults_to_render_only():
    from deployment import test_tracker_mujoco as tracker

    assert tracker._should_show_reference_ghost(render=True, reference_ghost=None)
    assert not tracker._should_show_reference_ghost(
        render=False, reference_ghost=None
    )
    assert not tracker._should_show_reference_ghost(
        render=True, reference_ghost=False
    )
    assert tracker._should_show_reference_ghost(
        render=False, reference_ghost=True
    )


def test_reference_ghost_mode_selects_mjcf_geom_groups():
    from deployment import test_tracker_mujoco as tracker

    assert tracker._reference_ghost_groups_for_mode("collision") == (3,)
    assert tracker._reference_ghost_groups_for_mode("mesh") == (1,)
    assert tracker._reference_ghost_groups_for_mode("all") == (1, 3)


def test_reference_ghost_sets_pose_and_populates_collision_scene(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    from deployment import test_tracker_mujoco as tracker

    xml_path = tmp_path / "minimal.xml"
    xml_path.write_text(MINIMAL_FREE_ROOT_XML)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ghost = tracker.ReferenceGhost(
        model=model,
        data=data,
        rgba=(0.2, 0.8, 1.0, 0.25),
        offset=(0.1, 0.2, 0.0),
    )

    state = {
        "body_pos": np.array(
            [[1.0, 2.0, 0.5], [1.0, 2.0, 0.7]], dtype=np.float32
        ),
        "body_rot": np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "dof_pos": np.array([0.3], dtype=np.float32),
    }

    ghost.set_pose(state)

    np.testing.assert_allclose(data.qpos[:3], [1.1, 2.2, 0.5])
    np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.qpos[7:], [0.3])

    scene = mujoco.MjvScene(model, maxgeom=32)
    ghost.update_scene(
        scene,
        pert=mujoco.MjvPerturb(),
    )

    assert scene.ngeom == 1
    assert scene.geoms[0].objid == 1
    for geom_idx in range(scene.ngeom):
        np.testing.assert_allclose(scene.geoms[geom_idx].rgba, [0.2, 0.8, 1.0, 0.25])
