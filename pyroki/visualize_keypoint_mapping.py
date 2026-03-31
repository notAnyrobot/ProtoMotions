# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from retarget_from_keypoints import load_robot_config

SMPL_KEYPOINT_TO_BODY = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "left_foot": "L_Toe",
    "right_foot": "R_Toe",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}

LOWER_BODY_KEYPOINTS = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
}

PAIR_COLORS = [
    (0.122, 0.467, 0.706, 1.0),
    (1.000, 0.498, 0.055, 1.0),
    (0.173, 0.627, 0.173, 1.0),
    (0.839, 0.153, 0.157, 1.0),
    (0.580, 0.404, 0.741, 1.0),
    (0.549, 0.337, 0.294, 1.0),
    (0.890, 0.467, 0.761, 1.0),
    (0.498, 0.498, 0.498, 1.0),
    (0.737, 0.741, 0.133, 1.0),
    (0.090, 0.745, 0.812, 1.0),
    (0.682, 0.780, 0.909, 1.0),
    (1.000, 0.733, 0.471, 1.0),
    (0.596, 0.875, 0.541, 1.0),
    (1.000, 0.596, 0.588, 1.0),
    (0.773, 0.690, 0.835, 1.0),
]

SMPL_BONE_RGBA = (0.85, 0.78, 0.65, 0.85)  # warm cream — SMPL side
ROBOT_BONE_RGBA = (0.50, 0.58, 0.70, 0.85)  # steel blue — robot side
ROBOT_AUX_COLORS = {
    "left_hand_aux": (0.00, 0.65, 0.95, 1.0),
    "right_hand_aux": (0.00, 0.75, 0.45, 1.0),
    "torso_aux": (0.95, 0.35, 0.15, 1.0),
}


def _default_smpl_mjcf_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
        / "smpl_humanoid.xml"
    )


def _resolve_existing_robot_urdf(robot_config: dict[str, Any]) -> Path:
    urdf_path = Path(robot_config["urdf_path"])
    if urdf_path.exists():
        return urdf_path

    robot_name = str(robot_config.get("name", "")).lower()
    search_dir = urdf_path.parent
    candidates = sorted(search_dir.glob("*.urdf"))
    preferred = [path for path in candidates if path.stem.lower().startswith(robot_name)]

    if len(preferred) == 1:
        print(
            "Configured URDF path does not exist; "
            f"using {preferred[0]} instead of {urdf_path}."
        )
        return preferred[0]

    if len(candidates) == 1:
        print(
            "Configured URDF path does not exist; "
            f"using {candidates[0]} instead of {urdf_path}."
        )
        return candidates[0]

    raise FileNotFoundError(
        f"Robot URDF not found at {urdf_path} and no unique fallback exists in {search_dir}"
    )


def _parse_vec3(raw_value: str | None) -> np.ndarray:
    if raw_value is None:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(value) for value in raw_value.split()], dtype=np.float64)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy

    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _compute_tpose_joint_angles(
    urdf_path: Path,
    display_pose_preset: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return joint angles for a strict T-pose display (arms extended sideways).

    If display_pose_preset is provided, those joint angles are used after
    clamping to URDF limits. Otherwise, a generic shoulder-roll heuristic is
    used as a best-effort fallback.

    All angles are clamped to the joint limits declared in the URDF.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limit_map: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            limit_map[name] = (
                float(limit_elem.attrib.get("lower", "-3.14159")),
                float(limit_elem.attrib.get("upper", "3.14159")),
            )

    if display_pose_preset is not None:
        angles: dict[str, float] = {}
        for joint_name, angle in display_pose_preset.items():
            if joint_name in limit_map:
                lower, upper = limit_map[joint_name]
                angles[joint_name] = float(np.clip(angle, lower, upper))
            else:
                angles[joint_name] = angle
        return angles

    # Generic heuristic: set shoulder_roll joints to ±π/2 (best-effort approximation).
    angles = {}
    for name, (lower, upper) in limit_map.items():
        if "shoulder_roll" not in name.lower():
            continue
        if "right" in name.lower():
            angles[name] = max(-np.pi / 2, lower)
        else:
            angles[name] = min(np.pi / 2, upper)
    return angles


def _load_urdf_link_kinematics(
    urdf_path: Path,
    joint_angles: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[tuple[str, str]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    joints_by_parent: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str, str]]] = {}
    child_links: set[str] = set()

    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            continue

        parent_link = parent_element.attrib["link"]
        child_link = child_element.attrib["link"]
        origin_element = joint.find("origin")
        joint_xyz = _parse_vec3(None if origin_element is None else origin_element.attrib.get("xyz"))
        joint_rpy = _parse_vec3(None if origin_element is None else origin_element.attrib.get("rpy"))
        axis_element = joint.find("axis")
        joint_axis = (
            np.array([float(v) for v in axis_element.attrib["xyz"].split()], dtype=np.float64)
            if axis_element is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        joint_type = joint.attrib.get("type", "fixed")
        joint_name = joint.attrib.get("name", "")

        joints_by_parent.setdefault(parent_link, []).append(
            (child_link, joint_xyz, joint_rpy, joint_axis, joint_type, joint_name)
        )
        child_links.add(child_link)

    root_candidates = [link_name for link_name in links if link_name not in child_links]
    if not root_candidates:
        raise ValueError(f"Could not identify a URDF root link in {urdf_path}")

    preferred_root = "world" if "world" in root_candidates else root_candidates[0]
    positions: dict[str, np.ndarray] = {preferred_root: np.zeros(3, dtype=np.float64)}
    rotations: dict[str, np.ndarray] = {preferred_root: np.eye(3, dtype=np.float64)}
    edges: list[tuple[str, str]] = []
    stack = [preferred_root]

    _joint_angles = joint_angles or {}

    while stack:
        parent_link = stack.pop()
        parent_position = positions[parent_link]
        parent_rotation = rotations[parent_link]

        for child_link, joint_xyz, joint_rpy, joint_axis, joint_type, joint_name in joints_by_parent.get(parent_link, []):
            child_rotation = parent_rotation @ _rotation_from_rpy(joint_rpy)
            child_position = parent_position + parent_rotation @ joint_xyz
            if joint_type == "revolute" and joint_name in _joint_angles:
                angle = _joint_angles[joint_name]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                K = np.array(
                    [
                        [0.0, -joint_axis[2], joint_axis[1]],
                        [joint_axis[2], 0.0, -joint_axis[0]],
                        [-joint_axis[1], joint_axis[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                R_joint = np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * (K @ K)
                child_rotation = child_rotation @ R_joint
            positions[child_link] = child_position
            rotations[child_link] = child_rotation
            if parent_link != "world" and np.linalg.norm(child_position - parent_position) >= 1e-5:
                edges.append((parent_link, child_link))
            stack.append(child_link)

    if preferred_root == "world":
        positions.pop("world", None)
        rotations.pop("world", None)

    return positions, rotations, edges


def _load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nq > 0:
        data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    return model, data


def _body_names(model: mujoco.MjModel) -> dict[int, str]:
    names: dict[int, str] = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is not None:
            names[body_id] = body_name
    return names


def _extract_body_positions(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for body_id, body_name in _body_names(model).items():
        positions[body_name] = np.array(data.xpos[body_id], dtype=np.float64)
    return positions


def _extract_body_rotations(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, np.ndarray]:
    rotations: dict[str, np.ndarray] = {}
    for body_id, body_name in _body_names(model).items():
        rotations[body_name] = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return rotations


def _skeleton_edges(
    model: mujoco.MjModel,
    positions: dict[str, np.ndarray],
) -> list[tuple[str, str]]:
    body_names = _body_names(model)
    edges: list[tuple[str, str]] = []
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id <= 0:
            continue

        child_name = body_names.get(body_id)
        parent_name = body_names.get(parent_id)
        if child_name is None or parent_name is None:
            continue
        if child_name not in positions or parent_name not in positions:
            continue
        if np.linalg.norm(positions[child_name] - positions[parent_name]) < 1e-5:
            continue

        edges.append((parent_name, child_name))

    return edges


def _validate_mapping(robot_config: dict[str, Any]) -> list[tuple[str, str]]:
    keypoint_mapping = robot_config.get("keypoint_mapping")
    if not isinstance(keypoint_mapping, list):
        raise ValueError("robot_config['keypoint_mapping'] must be a list")

    validated_pairs: list[tuple[str, str]] = []
    for pair in keypoint_mapping:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid keypoint mapping entry: {pair!r}")
        human_name, robot_link_name = pair
        if human_name not in SMPL_KEYPOINT_TO_BODY:
            raise ValueError(
                f"Unsupported human keypoint {human_name!r}; expected one of {list(SMPL_KEYPOINT_TO_BODY)}"
            )
        validated_pairs.append((human_name, robot_link_name))

    if len(validated_pairs) != len(SMPL_KEYPOINT_TO_BODY):
        raise ValueError(
            f"Expected {len(SMPL_KEYPOINT_TO_BODY)} keypoint pairs, got {len(validated_pairs)}"
        )

    return validated_pairs


def _display_translation(
    positions: dict[str, np.ndarray], root_body: str, x_anchor: float
) -> np.ndarray:
    root_position = positions[root_body]
    min_z = min(position[2] for position in positions.values())
    return np.array([x_anchor - root_position[0], -root_position[1], -min_z], dtype=np.float64)


def _translated_positions(
    positions: dict[str, np.ndarray], translation: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        body_name: position + translation for body_name, position in positions.items()
    }


def _scaled_source_positions(
    source_positions: dict[str, np.ndarray],
    keypoint_pairs: list[tuple[str, str]],
    source_scale: dict[str, Any],
    robot_root_position: np.ndarray,
) -> dict[str, np.ndarray]:
    root_name = "pelvis"
    root_body = SMPL_KEYPOINT_TO_BODY[root_name]
    root_source_position = source_positions[root_body]

    scaled_positions: dict[str, np.ndarray] = {}
    scaled_root_position = robot_root_position + root_source_position * 0.0
    scaled_positions[root_name] = scaled_root_position

    for human_name, _ in keypoint_pairs:
        if human_name == root_name:
            continue

        body_name = SMPL_KEYPOINT_TO_BODY[human_name]
        local_position = source_positions[body_name] - root_source_position
        scale_key = "lower_body" if human_name in LOWER_BODY_KEYPOINTS else "upper_body"
        scaled_local = local_position * np.array(source_scale[scale_key], dtype=np.float64)
        scaled_positions[human_name] = scaled_root_position + scaled_local

    return scaled_positions


def _robot_aux_positions(
    robot_positions: dict[str, np.ndarray],
    robot_rotations: dict[str, np.ndarray],
    robot_config: dict[str, Any],
    translation: np.ndarray,
) -> dict[str, np.ndarray]:
    left_wrist_body = dict(robot_config["keypoint_mapping"])["left_wrist"]
    right_wrist_body = dict(robot_config["keypoint_mapping"])["right_wrist"]
    torso_body = robot_config["torso_link_name"]

    hand_aux_offset = np.array(robot_config["hand_aux_offset"], dtype=np.float64)
    torso_aux_offset = np.array(robot_config["torso_aux_offset"], dtype=np.float64)

    aux_positions = {
        "left_hand_aux": robot_positions[left_wrist_body]
        + robot_rotations[left_wrist_body] @ hand_aux_offset,
        "right_hand_aux": robot_positions[right_wrist_body]
        + robot_rotations[right_wrist_body] @ hand_aux_offset,
        "torso_aux": robot_positions[torso_body]
        + robot_rotations[torso_body] @ torso_aux_offset,
    }
    return {
        marker_name: position + translation
        for marker_name, position in aux_positions.items()
    }


def _fmt(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _checkerboard_texture_path() -> Path | None:
    candidate = (
        Path(__file__).resolve().parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "checkerboard"
        / "checkerboard_texture.png"
    )
    return candidate if candidate.exists() else None


def _build_scene_xml(
    smpl_positions: dict[str, np.ndarray],
    smpl_edges: list[tuple[str, str]],
    robot_positions: dict[str, np.ndarray],
    robot_edges: list[tuple[str, str]],
    keypoint_pairs: list[tuple[str, str]],
    scaled_source_positions: dict[str, np.ndarray],
    robot_aux_positions: dict[str, np.ndarray],
) -> str:
    checkerboard_tex = _checkerboard_texture_path()

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mujoco model="keypoint_mapping_viewer">',
        '  <option gravity="0 0 0" timestep="0.0166667"/>',
        '  <visual>',
        '    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>',
        '    <map znear="0.01" zfar="30"/>',
        '  </visual>',
    ]

    if checkerboard_tex is not None:
        lines += [
            '  <asset>',
            f'    <texture name="checkerboard_tex" type="2d" file="{checkerboard_tex}"/>',
            '    <material name="checkerboard_mat" texture="checkerboard_tex"'
            ' texrepeat="2 2" texuniform="true"/>',
            '  </asset>',
        ]

    lines += [
        '  <worldbody>',
        '    <light name="scene_light" pos="0 0 4.5" dir="0 0 -1" directional="true" diffuse="0.85 0.85 0.85"/>',
    ]

    if checkerboard_tex is not None:
        lines.append(
            '    <geom name="ground" type="plane" pos="0 0 0" size="8 8 0.1"'
            ' material="checkerboard_mat"/>'
        )
    else:
        lines.append(
            '    <geom name="ground" type="plane" pos="0 0 0" size="8 8 0.1"'
            ' rgba="0.96 0.96 0.96 1"/>'
        )

    for side, positions, edges, bone_rgba in (
        ("smpl", smpl_positions, smpl_edges, SMPL_BONE_RGBA),
        ("robot", robot_positions, robot_edges, ROBOT_BONE_RGBA),
    ):
        for parent_name, child_name in edges:
            lines.append(
                "    "
                + f'<geom name="{side}_bone_{parent_name}_{child_name}" '
                + 'type="capsule" '
                + f'fromto="{_fmt(positions[parent_name])} {_fmt(positions[child_name])}" '
                + 'size="0.012" '
                + f'rgba="{_fmt(bone_rgba)}"/>'
            )

    for index, (human_name, robot_link_name) in enumerate(keypoint_pairs, start=1):
        color = PAIR_COLORS[index - 1]
        smpl_body = SMPL_KEYPOINT_TO_BODY[human_name]
        # SMPL spheres: pastel (50% blend with white) so the two sides are immediately
        # distinguishable; robot spheres keep the vivid pair color.
        smpl_color = tuple((c + 1.0) / 2.0 for c in color[:3]) + (1.0,)
        scaled_color = color[:3] + (0.45,)

        lines.append(
            "    "
            + f'<geom name="smpl_pair_{index:02d}_{human_name}" type="sphere" '
            + f'pos="{_fmt(smpl_positions[smpl_body])}" size="0.030" '
            + f'rgba="{_fmt(smpl_color)}"/>'
        )
        lines.append(
            "    "
            + f'<geom name="robot_pair_{index:02d}_{human_name}_to_{robot_link_name}" type="sphere" '
            + f'pos="{_fmt(robot_positions[robot_link_name])}" size="0.035" '
            + f'rgba="{_fmt(color)}"/>'
        )
        lines.append(
            "    "
            + f'<geom name="scaled_source_{index:02d}_{human_name}" type="box" '
            + f'pos="{_fmt(scaled_source_positions[human_name])}" size="0.020 0.020 0.020" '
            + f'rgba="{_fmt(scaled_color)}"/>'
        )

    for marker_name, color in ROBOT_AUX_COLORS.items():
        if marker_name not in robot_aux_positions:
            continue
        lines.append(
            "    "
            + f'<geom name="{marker_name}" type="sphere" '
            + f'pos="{_fmt(robot_aux_positions[marker_name])}" size="0.028" '
            + f'rgba="{_fmt(color)}"/>'
        )

    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def _print_mapping_summary(
    keypoint_pairs: list[tuple[str, str]],
    robot_config: dict[str, Any],
    source_type: str,
) -> None:
    print("SMPL to robot keypoint mapping")
    print("============================")
    for index, (human_name, robot_link_name) in enumerate(keypoint_pairs, start=1):
        color = PAIR_COLORS[index - 1]
        print(
            f"{index:02d}. {human_name:<16} -> {robot_link_name:<28} "
            f"rgba=({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}, {color[3]:.3f})"
        )

    print()
    print(
        f"Scaled source overlay uses scale_factors['{source_type}'] = "
        f"{robot_config['scale_factors'][source_type]}"
    )
    print(
        f"Aux markers use hand_aux_offset={robot_config['hand_aux_offset']} "
        f"and torso_aux_offset={robot_config['torso_aux_offset']}"
    )
    print()
    print("Viewer legend")
    print("-------------")
    print("- Left pastel spheres:  raw SMPL keypoints  (T-pose, zero joint angles)")
    print("- Right vivid spheres:  mapped robot links  (T-pose, robot-specific preset)")
    print(
        "- Right translucent boxes (same color, 45% alpha): SMPL keypoints after"
        " scale_factors are applied and the pelvis is re-aligned to the robot root."
        " These are the positions the retargeter will try to match — they visualise"
        " the same scaling as _apply_source_scaling() in retarget_from_keypoints.py."
    )
    print("- Cream bones: SMPL skeleton | Steel-blue bones: robot skeleton")
    print("- Cyan/green/orange markers: left hand aux, right hand aux, torso aux")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SMPL-to-robot keypoint mapping in MuJoCo without running retargeting."
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        required=True,
        help="Path to a robot YAML config matching the g1.yaml / h1_2.yaml format.",
    )
    parser.add_argument(
        "--smpl-mjcf",
        type=str,
        default=str(_default_smpl_mjcf_path()),
        help="Path to the SMPL MJCF asset.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="smpl",
        help="Source type key used to select scale_factors for the scaled overlay.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.7,
        help="Half-distance between the SMPL display on the left and the robot display on the right.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_config = load_robot_config(args.robot_config)
    keypoint_pairs = _validate_mapping(robot_config)

    scale_factors = robot_config.get("scale_factors")
    if args.source_type not in scale_factors:
        available = ", ".join(sorted(scale_factors))
        raise ValueError(
            f"source_type={args.source_type!r} is not present in scale_factors. Available: {available}"
        )

    smpl_path = Path(args.smpl_mjcf).resolve()
    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL MJCF not found: {smpl_path}")

    robot_urdf_path = _resolve_existing_robot_urdf(robot_config)

    smpl_model, smpl_data = _load_model(smpl_path)
    smpl_positions_raw = _extract_body_positions(smpl_model, smpl_data)
    tpose_angles = _compute_tpose_joint_angles(
        robot_urdf_path,
        display_pose_preset=robot_config.get("display_pose_preset"),
    )
    robot_positions_raw, robot_rotations_raw, robot_edges = _load_urdf_link_kinematics(
        robot_urdf_path, joint_angles=tpose_angles
    )

    missing_smpl = [
        SMPL_KEYPOINT_TO_BODY[human_name]
        for human_name, _ in keypoint_pairs
        if SMPL_KEYPOINT_TO_BODY[human_name] not in smpl_positions_raw
    ]
    missing_robot = [
        robot_link_name
        for _, robot_link_name in keypoint_pairs
        if robot_link_name not in robot_positions_raw
    ]
    if missing_smpl:
        raise ValueError(f"SMPL model is missing expected bodies: {missing_smpl}")
    if missing_robot:
        raise ValueError(f"Robot URDF is missing expected links: {missing_robot}")
    if robot_config["torso_link_name"] not in robot_positions_raw:
        raise ValueError(
            f"Robot URDF is missing torso_link_name={robot_config['torso_link_name']!r}"
        )

    smpl_root_body = SMPL_KEYPOINT_TO_BODY["pelvis"]
    robot_root_body = dict(keypoint_pairs)["pelvis"]

    smpl_translation = _display_translation(smpl_positions_raw, smpl_root_body, -args.spacing)
    robot_translation = _display_translation(robot_positions_raw, robot_root_body, args.spacing)

    smpl_positions = _translated_positions(smpl_positions_raw, smpl_translation)
    robot_positions = _translated_positions(robot_positions_raw, robot_translation)
    smpl_edges = _skeleton_edges(smpl_model, smpl_positions_raw)

    scaled_source_positions = _scaled_source_positions(
        smpl_positions_raw,
        keypoint_pairs,
        robot_config["scale_factors"][args.source_type],
        robot_positions[robot_root_body],
    )
    robot_aux_positions = _robot_aux_positions(
        robot_positions_raw,
        robot_rotations_raw,
        robot_config,
        robot_translation,
    )

    scene_xml = _build_scene_xml(
        smpl_positions=smpl_positions,
        smpl_edges=smpl_edges,
        robot_positions=robot_positions,
        robot_edges=robot_edges,
        keypoint_pairs=keypoint_pairs,
        scaled_source_positions=scaled_source_positions,
        robot_aux_positions=robot_aux_positions,
    )

    scene_model = mujoco.MjModel.from_xml_string(scene_xml)
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)

    _print_mapping_summary(keypoint_pairs, robot_config, args.source_type)

    with mujoco.viewer.launch_passive(scene_model, scene_data) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.85])
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0

        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()