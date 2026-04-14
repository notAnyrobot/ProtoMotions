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
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

# Heavy dependencies — guarded so this module can be imported without mujoco
try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    import yaml

    # Ensure pyroki/ is on sys.path for imports when running from tools/
    _pyroki_dir = str(Path(__file__).resolve().parent.parent / "pyroki")
    if _pyroki_dir not in sys.path:
        sys.path.insert(0, _pyroki_dir)

    from retarget_from_keypoints import load_robot_config
    from visualize_keypoint_mapping import (
        SMPL_KEYPOINT_TO_BODY,
        _build_scene_xml,
        _compute_tpose_joint_angles,
        _default_smpl_mjcf_path,
        _display_translation,
        _extract_body_positions,
        _load_model,
        _load_urdf_link_kinematics,
        _resolve_existing_robot_urdf,
        _robot_aux_positions,
        _scaled_source_positions,
        _skeleton_edges,
        _translated_positions,
        _validate_mapping,
    )
except ImportError:
    mujoco = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    yaml = None  # type: ignore[assignment]

SOURCE_OPTIONS = ["smpl", "rigv1", "g1 (not available yet)"]
SCALE_GROUPS = ("root", "lower_body", "upper_body")
SCALE_AXES = ("x", "y", "z")
AUX_OFFSET_GROUPS = ("hand_aux_offset", "torso_aux_offset")


def _default_robot_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "pyroki" / "robot_configs" / "g1.yaml"
    )


@dataclass
class SceneBundle:
    robot_config_path: Path
    robot_config: dict[str, Any]
    keypoint_pairs: list[tuple[str, str]]
    smpl_positions_raw: dict[str, np.ndarray]
    robot_positions: dict[str, np.ndarray]
    robot_root_body: str
    scene_model: mujoco.MjModel
    scene_data: mujoco.MjData


class GuiApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Keypoint Mapping Tuner — ProtoMotions")
        self.root.geometry("900x1000")

        # Modern color scheme
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#0066cc"
        self.root.configure(bg=self.bg_color)

        self.robot_config_path = Path(args.robot_config).resolve()
        self.selected_source_type = args.source_type
        self.previous_source_type = args.source_type
        self.viewer_spacing = args.spacing

        self.pending_reload_path: Path | None = None
        self.reload_requested = False
        self.quit_requested = False
        self.scale_dirty = False
        self.spacing_dirty = False
        self._ignore_slider_callback = False
        self.aux_offset_dirty = False

        self.scale_overrides: dict[str, list[float]] = {
            "root": [1.0, 1.0, 1.0],
            "lower_body": [1.0, 1.0, 1.0],
            "upper_body": [1.0, 1.0, 1.0],
        }
        self.aux_offset_overrides: dict[str, list[float]] = {
            "hand_aux_offset": [0.0, 0.0, 0.0],
            "torso_aux_offset": [0.0, 0.0, 0.0],
        }
        self.slider_vars: dict[tuple[str, int], tk.DoubleVar] = {}
        self.aux_slider_vars: dict[tuple[str, int], tk.DoubleVar] = {}

        self.scene: SceneBundle | None = None
        self.scaled_geom_ids: dict[str, int] = {}
        self.aux_geom_ids: dict[str, int] = {}

        self._build_gui()

    def _build_gui(self) -> None:
        # Configure ttk style for modern look
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure(
            "TLabelframe", background=self.bg_color, foreground=self.fg_color
        )
        style.configure("TButton", background=self.bg_color)

        # Main container with scrolling
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ============= Robot Configuration Section =============
        config_frame = ttk.LabelFrame(
            scrollable_frame, text="Robot Configuration", padding=12
        )
        config_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        config_inner = ttk.Frame(config_frame)
        config_inner.pack(fill=tk.X)

        ttk.Label(config_inner, text="Config File:", width=14).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.config_path_var = tk.StringVar(value=str(self.robot_config_path))
        ttk.Entry(config_inner, textvariable=self.config_path_var, width=50).pack(
            side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True
        )
        ttk.Button(config_inner, text="Browse", command=self._on_browse_config).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(config_inner, text="Load", command=self._on_load_config).pack(
            side=tk.LEFT
        )

        # ============= Source Type Section =============
        source_frame = ttk.LabelFrame(
            scrollable_frame, text="Source Type & Viewer", padding=12
        )
        source_frame.pack(fill=tk.X, padx=12, pady=6)

        source_row = ttk.Frame(source_frame)
        source_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(source_row, text="Source:", width=14).pack(side=tk.LEFT, padx=(0, 6))
        self.source_var = tk.StringVar(value=self.selected_source_type)
        source_menu = ttk.OptionMenu(source_row, self.source_var, *SOURCE_OPTIONS)
        source_menu.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            source_row,
            text="Apply Source",
            command=self._on_apply_source,
        ).pack(side=tk.LEFT)
        ttk.Label(source_row, text="(g1 disabled for now)", foreground="#666666").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # Distance slider
        distance_row = ttk.Frame(source_frame)
        distance_row.pack(fill=tk.X)

        ttk.Label(distance_row, text="Viewer Distance:", width=14).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.distance_var = tk.DoubleVar(value=self.viewer_spacing)
        distance_scale = ttk.Scale(
            distance_row,
            from_=0.5,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.distance_var,
            command=self._on_distance_change,
        )
        distance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.distance_label = ttk.Label(
            distance_row, text=f"{self.viewer_spacing:.2f}", width=6
        )
        self.distance_label.pack(side=tk.LEFT)

        # ============= Scale Factors Section =============
        scale_frame = ttk.LabelFrame(
            scrollable_frame, text="Scale Factors — [0.1, 3.0]", padding=12
        )
        scale_frame.pack(fill=tk.X, padx=12, pady=6)

        for row_idx, group_name in enumerate(SCALE_GROUPS):
            group_label = ttk.Label(
                scale_frame,
                text=group_name,
                font=("TkDefaultFont", 9, "bold"),
                width=11,
            )
            group_label.grid(row=row_idx * 4, column=0, sticky="w", padx=(0, 12))

            for axis_idx, axis_name in enumerate(SCALE_AXES):
                var = tk.DoubleVar(value=1.0)
                self.slider_vars[(group_name, axis_idx)] = var

                label = ttk.Label(scale_frame, text=f"├─ {axis_name.upper()}", width=11)
                label.grid(
                    row=row_idx * 4 + axis_idx + 1, column=0, sticky="w", padx=(14, 8)
                )

                scale = ttk.Scale(
                    scale_frame,
                    variable=var,
                    from_=0.1,
                    to=3.0,
                    orient=tk.HORIZONTAL,
                    command=lambda _value, g=group_name, ag=axis_idx: (
                        self._on_slider_change(g),
                        self._update_scale_label(g),
                    ),
                )
                scale.grid(
                    row=row_idx * 4 + axis_idx + 1, column=1, sticky="ew", padx=(0, 8)
                )

                value_label = ttk.Label(scale_frame, text="1.00", width=5)
                value_label.grid(row=row_idx * 4 + axis_idx + 1, column=2)
                # Store reference for updating
                if not hasattr(self, "scale_value_labels"):
                    self.scale_value_labels = {}
                self.scale_value_labels[(group_name, axis_idx)] = (value_label, var)

        scale_frame.columnconfigure(1, weight=1)

        # ============= Aux Offsets Section =============
        aux_frame = ttk.LabelFrame(
            scrollable_frame, text="Aux Offsets (meters)", padding=12
        )
        aux_frame.pack(fill=tk.X, padx=12, pady=6)

        for row_idx, aux_group in enumerate(AUX_OFFSET_GROUPS):
            group_label = ttk.Label(
                aux_frame,
                text=aux_group.replace("_", " ").title(),
                font=("TkDefaultFont", 9, "bold"),
                width=20,
            )
            group_label.grid(row=row_idx * 4, column=0, sticky="w", padx=(0, 12))

            for axis_idx, axis_name in enumerate(SCALE_AXES):
                var = tk.DoubleVar(value=0.0)
                self.aux_slider_vars[(aux_group, axis_idx)] = var

                label = ttk.Label(aux_frame, text=f"├─ {axis_name.upper()}", width=20)
                label.grid(
                    row=row_idx * 4 + axis_idx + 1, column=0, sticky="w", padx=(14, 8)
                )

                scale = ttk.Scale(
                    aux_frame,
                    variable=var,
                    from_=-1.0,
                    to=1.0,
                    orient=tk.HORIZONTAL,
                    command=lambda _value, ag=aux_group: (
                        self._on_aux_slider_change(ag),
                        self._update_aux_label(ag),
                    ),
                )
                scale.grid(
                    row=row_idx * 4 + axis_idx + 1, column=1, sticky="ew", padx=(0, 8)
                )

                value_label = ttk.Label(aux_frame, text="0.00", width=6)
                value_label.grid(row=row_idx * 4 + axis_idx + 1, column=2)
                # Store reference for updating
                if not hasattr(self, "aux_value_labels"):
                    self.aux_value_labels = {}
                self.aux_value_labels[(aux_group, axis_idx)] = (value_label, var)

        aux_frame.columnconfigure(1, weight=1)

        # ============= Action Buttons Section =============
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, padx=12, pady=12)

        button_left = ttk.Frame(action_frame)
        button_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(
            button_left,
            text="Reset Sliders to Config",
            command=self._on_reset_scales,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            button_left,
            text="Reset Aux Offsets",
            command=self._on_reset_aux,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            button_left,
            text="Save Config",
            command=self._on_save_config,
        ).pack(side=tk.LEFT)

        ttk.Button(
            button_left,
            text="Exit",
            command=self._on_close,
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(
            action_frame,
            textvariable=self.status_var,
            foreground="#666666",
            font=("TkDefaultFont", 9),
        )
        status_label.pack(side=tk.RIGHT, padx=(12, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_scale_label(self, scale_group: str) -> None:
        """Update scale value labels when sliders move."""
        for (group_name, axis_idx), (label, var) in self.scale_value_labels.items():
            if group_name == scale_group:
                label.config(text=f"{var.get():.2f}")

    def _update_aux_label(self, aux_group: str) -> None:
        """Update aux value labels when sliders move."""
        for (group, axis_idx), (label, var) in self.aux_value_labels.items():
            if group == aux_group:
                label.config(text=f"{var.get():.3f}")

    def _refresh_all_scale_labels(self) -> None:
        for group_name in SCALE_GROUPS:
            self._update_scale_label(group_name)

    def _refresh_all_aux_labels(self) -> None:
        for aux_group in AUX_OFFSET_GROUPS:
            self._update_aux_label(aux_group)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _resolve_active_source(self) -> str:
        if self.selected_source_type in ("smpl", "rigv1"):
            return self.selected_source_type
        return "smpl"

    def _on_apply_source(self) -> None:
        requested = self.source_var.get()
        if requested.startswith("g1"):
            self.source_var.set(self.previous_source_type)
            self._set_status(
                "Source type 'g1' is not available yet. Using previous selection."
            )
            return

        self.selected_source_type = requested
        self.previous_source_type = requested
        self._sync_sliders_from_active_source()
        self.scale_dirty = True
        self._set_status(f"Source type switched to '{requested}'.")

    def _on_browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select robot config YAML",
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
            initialdir=str(self.robot_config_path.parent),
        )
        if selected:
            self.config_path_var.set(selected)

    def _on_load_config(self) -> None:
        path = Path(self.config_path_var.get()).expanduser().resolve()
        self.pending_reload_path = path
        self.reload_requested = True
        self._set_status(f"Reload requested: {path}")

    def _on_reset_scales(self) -> None:
        self._sync_sliders_from_active_source()
        self.scale_dirty = True
        self._set_status("Scale sliders reset from loaded config.")

    def _on_reset_aux(self) -> None:
        self._ignore_slider_callback = True
        for aux_group in AUX_OFFSET_GROUPS:
            for axis_idx in range(len(SCALE_AXES)):
                self.aux_slider_vars[(aux_group, axis_idx)].set(0.0)
                self.aux_offset_overrides[aux_group][axis_idx] = 0.0
        self._ignore_slider_callback = False
        self._refresh_all_aux_labels()
        self.aux_offset_dirty = True
        self.scale_dirty = True
        self._set_status("Aux offsets reset to zero.")

    def _on_slider_change(self, group_name: str) -> None:
        if self._ignore_slider_callback:
            return

        values = [
            float(self.slider_vars[(group_name, idx)].get())
            for idx in range(len(SCALE_AXES))
        ]
        self.scale_overrides[group_name] = values
        self.scale_dirty = True

    def _on_aux_slider_change(self, aux_group: str) -> None:
        if self._ignore_slider_callback:
            return

        values = [
            float(self.aux_slider_vars[(aux_group, idx)].get())
            for idx in range(len(SCALE_AXES))
        ]
        self.aux_offset_overrides[aux_group] = values
        self.aux_offset_dirty = True
        self.scale_dirty = True

    def _on_distance_change(self, _value: str) -> None:
        self.viewer_spacing = float(self.distance_var.get())
        self.distance_label.config(text=f"{self.viewer_spacing:.2f}")
        self.spacing_dirty = True

    def _on_close(self) -> None:
        self.quit_requested = True
        self.root.destroy()

    def _on_save_config(self) -> None:
        """Save current scale and aux offset values to robot config YAML."""
        try:
            # Load original config
            with open(self.robot_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update scale factors for the active source type only.
            active_source_type = self._resolve_active_source()
            config["scale_factors"][active_source_type] = {
                "root": [round(v, 2) for v in self.scale_overrides["root"]],
                "lower_body": [round(v, 2) for v in self.scale_overrides["lower_body"]],
                "upper_body": [round(v, 2) for v in self.scale_overrides["upper_body"]],
            }

            # Update aux offsets (round to 3 decimals)
            config["hand_aux_offset"] = [
                round(v, 3) for v in self.aux_offset_overrides["hand_aux_offset"]
            ]
            config["torso_aux_offset"] = [
                round(v, 3) for v in self.aux_offset_overrides["torso_aux_offset"]
            ]

            # Custom YAML dump that avoids anchors/aliases and preserves simple format
            class NoAliasDumper(yaml.SafeDumper):
                def ignore_aliases(self, data):
                    return True

            with open(self.robot_config_path, "w") as f:
                yaml.dump(
                    config,
                    f,
                    Dumper=NoAliasDumper,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            self._set_status(f"Config saved: {self.robot_config_path.name}")
            messagebox.showinfo(
                "Success", f"Config saved to {self.robot_config_path.name}"
            )
        except Exception as exc:
            error_msg = f"Failed to save config: {exc}"
            self._set_status(error_msg)
            messagebox.showerror("Error", error_msg)

    def _sync_sliders_from_active_source(self) -> None:
        if self.scene is None:
            return

        source_type = self._resolve_active_source()
        config_scales = self.scene.robot_config["scale_factors"][source_type]

        self._ignore_slider_callback = True
        for group_name in SCALE_GROUPS:
            raw_values = config_scales[group_name]
            clamped = [float(np.clip(value, 0.1, 3.0)) for value in raw_values]
            self.scale_overrides[group_name] = clamped
            for axis_idx, value in enumerate(clamped):
                self.slider_vars[(group_name, axis_idx)].set(value)
        self._ignore_slider_callback = False
        self._refresh_all_scale_labels()

    def _sync_aux_offsets_from_config(self) -> None:
        if self.scene is None:
            return

        self._ignore_slider_callback = True
        for aux_group in AUX_OFFSET_GROUPS:
            config_values = self.scene.robot_config.get(aux_group, [0.0, 0.0, 0.0])
            for axis_idx, value in enumerate(config_values):
                self.aux_slider_vars[(aux_group, axis_idx)].set(value)
            self.aux_offset_overrides[aux_group] = list(config_values)
        self._ignore_slider_callback = False
        self._refresh_all_aux_labels()

    def _build_scene_bundle(self, robot_config_path: Path) -> SceneBundle:
        robot_config = load_robot_config(str(robot_config_path))
        keypoint_pairs = _validate_mapping(robot_config)

        scale_factors = robot_config.get("scale_factors", {})
        for required_source in ("smpl", "rigv1"):
            if required_source not in scale_factors:
                raise ValueError(
                    f"Robot config missing scale_factors['{required_source}']"
                )

        smpl_path = Path(self.args.smpl_mjcf).resolve()
        if not smpl_path.exists():
            raise FileNotFoundError(f"SMPL MJCF not found: {smpl_path}")

        robot_urdf_path = _resolve_existing_robot_urdf(robot_config)

        smpl_model, smpl_data = _load_model(smpl_path)
        smpl_positions_raw = _extract_body_positions(smpl_model, smpl_data)

        tpose_angles = _compute_tpose_joint_angles(
            robot_urdf_path,
            display_pose_preset=robot_config.get("display_pose_preset"),
        )
        robot_positions_raw, robot_rotations_raw, robot_edges = (
            _load_urdf_link_kinematics(
                robot_urdf_path,
                joint_angles=tpose_angles,
            )
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

        smpl_root_body = SMPL_KEYPOINT_TO_BODY["pelvis"]
        robot_root_body = dict(keypoint_pairs)["pelvis"]

        smpl_translation = _display_translation(
            smpl_positions_raw,
            smpl_root_body,
            -self.viewer_spacing,
        )
        robot_translation = _display_translation(
            robot_positions_raw,
            robot_root_body,
            self.viewer_spacing,
        )

        smpl_positions = _translated_positions(smpl_positions_raw, smpl_translation)
        robot_positions = _translated_positions(robot_positions_raw, robot_translation)
        smpl_edges = _skeleton_edges(smpl_model, smpl_positions_raw)

        source_type = self._resolve_active_source()
        scaled_source_positions = _scaled_source_positions(
            smpl_positions_raw,
            keypoint_pairs,
            robot_config["scale_factors"][source_type],
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

        return SceneBundle(
            robot_config_path=robot_config_path,
            robot_config=robot_config,
            keypoint_pairs=keypoint_pairs,
            smpl_positions_raw=smpl_positions_raw,
            robot_positions=robot_positions,
            robot_root_body=robot_root_body,
            scene_model=scene_model,
            scene_data=scene_data,
        )

    def _cache_scaled_geom_ids(self) -> None:
        assert self.scene is not None
        self.scaled_geom_ids = {}
        for index, (human_name, _robot_link_name) in enumerate(
            self.scene.keypoint_pairs,
            start=1,
        ):
            geom_name = f"scaled_source_{index:02d}_{human_name}"
            geom_id = mujoco.mj_name2id(
                self.scene.scene_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id >= 0:
                self.scaled_geom_ids[human_name] = geom_id

    def _cache_aux_geom_ids(self) -> None:
        """Cache geom IDs for aux offset markers (hand and torso)."""
        assert self.scene is not None
        self.aux_geom_ids = {}
        aux_markers = ["left_hand_aux", "right_hand_aux", "torso_aux"]
        for marker_name in aux_markers:
            geom_id = mujoco.mj_name2id(
                self.scene.scene_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                marker_name,
            )
            if geom_id >= 0:
                self.aux_geom_ids[marker_name] = geom_id

    def _apply_scaled_overlay_update(self) -> None:
        assert self.scene is not None

        source_scale = {
            group_name: self.scale_overrides[group_name] for group_name in SCALE_GROUPS
        }
        scaled_positions = _scaled_source_positions(
            self.scene.smpl_positions_raw,
            self.scene.keypoint_pairs,
            source_scale,
            self.scene.robot_positions[self.scene.robot_root_body],
        )

        for human_name, position in scaled_positions.items():
            geom_id = self.scaled_geom_ids.get(human_name)
            if geom_id is not None:
                self.scene.scene_model.geom_pos[geom_id] = position

        mujoco.mj_forward(self.scene.scene_model, self.scene.scene_data)

    def _apply_aux_offset_update(self) -> None:
        """Update aux marker positions based on current offset sliders."""
        assert self.scene is not None

        # Recompute aux positions with current offset overrides
        robot_positions_raw, robot_rotations_raw, _ = _load_urdf_link_kinematics(
            _resolve_existing_robot_urdf(self.scene.robot_config),
            joint_angles=_compute_tpose_joint_angles(
                _resolve_existing_robot_urdf(self.scene.robot_config),
                display_pose_preset=self.scene.robot_config.get("display_pose_preset"),
            ),
        )

        # Build robot_translation
        smpl_root_body = SMPL_KEYPOINT_TO_BODY["pelvis"]
        robot_root_body = self.scene.robot_root_body
        robot_translation = _display_translation(
            robot_positions_raw,
            robot_root_body,
            self.viewer_spacing,
        )

        # Create modified robot config with current aux offsets
        modified_config = self.scene.robot_config.copy()
        modified_config["hand_aux_offset"] = self.aux_offset_overrides[
            "hand_aux_offset"
        ]
        modified_config["torso_aux_offset"] = self.aux_offset_overrides[
            "torso_aux_offset"
        ]

        # Recompute aux positions
        robot_aux_positions = _robot_aux_positions(
            robot_positions_raw,
            robot_rotations_raw,
            modified_config,
            robot_translation,
        )

        # Update geom positions
        for marker_name, new_pos in robot_aux_positions.items():
            if marker_name in self.aux_geom_ids:
                geom_id = self.aux_geom_ids[marker_name]
                self.scene.scene_model.geom_pos[geom_id] = new_pos

        mujoco.mj_forward(self.scene.scene_model, self.scene.scene_data)

    def _reload_if_requested(self) -> bool:
        if not self.reload_requested:
            return False

        self.reload_requested = False
        assert self.pending_reload_path is not None

        try:
            new_scene = self._build_scene_bundle(self.pending_reload_path)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            return False

        self.scene = new_scene
        self.robot_config_path = self.pending_reload_path
        self.config_path_var.set(str(self.robot_config_path))
        self._sync_sliders_from_active_source()
        self._sync_aux_offsets_from_config()
        self.scale_dirty = True
        self._set_status(f"Loaded robot config: {self.robot_config_path.name}")
        return True

    def run(self) -> None:
        self.scene = self._build_scene_bundle(self.robot_config_path)
        self._sync_sliders_from_active_source()
        self._sync_aux_offsets_from_config()
        self.scale_dirty = True

        while not self.quit_requested:
            assert self.scene is not None
            self._cache_scaled_geom_ids()
            self._cache_aux_geom_ids()

            with mujoco.viewer.launch_passive(
                self.scene.scene_model,
                self.scene.scene_data,
            ) as viewer:
                viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.85])
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -12.0

                self._set_status("Viewer running. Adjust sliders for live tuning.")

                while viewer.is_running() and not self.quit_requested:
                    try:
                        self.root.update_idletasks()
                        self.root.update()
                    except tk.TclError:
                        self.quit_requested = True
                        break

                    if self._reload_if_requested():
                        # Break and re-open a fresh viewer with the new model.
                        break

                    if self.spacing_dirty:
                        # Need to rebuild scene with new spacing
                        try:
                            new_scene = self._build_scene_bundle(self.robot_config_path)
                            self.scene = new_scene
                            self.spacing_dirty = False
                            break  # Re-open viewer with new spacing
                        except Exception as exc:
                            self._set_status(f"Spacing update failed: {exc}")
                            self.spacing_dirty = False

                    if self.scale_dirty:
                        self._apply_scaled_overlay_update()

                    if self.aux_offset_dirty:
                        self._apply_aux_offset_update()
                        self.aux_offset_dirty = False
                        self.scale_dirty = False

                    viewer.sync()
                    time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive GUI for SMPL-to-robot keypoint mapping with runtime "
            "robot-config loading and scale-factor tuning."
        )
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default=str(_default_robot_config_path()),
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
        choices=("smpl", "rigv1"),
        help="Initial source type used for scale factors.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Initial viewer spacing between SMPL and robot display.",
    )
    return parser.parse_args()


def main() -> None:
    if mujoco is None:
        sys.exit(
            "Error: 'mujoco' package is required for the keypoint mapping tuner.\n"
            "Install it with:  pip install mujoco"
        )
    args = parse_args()
    app = GuiApp(args)
    app.run()


if __name__ == "__main__":
    main()
