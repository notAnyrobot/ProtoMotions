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
"""PyRoki-only retargeting GUI.

This GUI intentionally contains only the PyRoki-environment portions of the
retargeting workflow: Steps 2 and 3 plus the keypoint config editor/tuner.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from retarget_gui_common import (
    PROJECT_ROOT,
    SCRIPT_DIR,
    SKELETON_FORMATS,
    _ACCENT,
    _LOG_BG,
    _LOG_FG,
    _MONO_FONT,
    _MUTED,
    _labeled_check,
    _labeled_combo,
    _labeled_entry,
    _labeled_spin,
    _readonly_entry,
    _section_header,
    RetargetGUIBase,
    yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyRoki-only retargeting GUI for steps 2 and 3."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    app = PyRokiRetargetGUI()
    app.run()


class PyRokiRetargetGUI(RetargetGUIBase):
    """PyRoki-environment GUI split from the monolithic retargeting app."""

    def __init__(self) -> None:
        super().__init__(
            title="PyRoki — Retargeting Tools (PyRoki Environment)",
            env_mode="pyroki",
        )
        self._build_notebook()

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._build_tab_batch()
        self._build_tab_steps()
        self._build_tab_keypoint()

    def _build_tab_batch(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Batch Retarget ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        _section_header(inner, "Input")

        _, self._batch_config_var = _labeled_entry(
            inner,
            "Robot Config (.yaml):",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Path to the YAML robot retarget config.",
            browse="file",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        _, self._batch_kp_var = _labeled_entry(
            inner,
            "Keypoints Folder:",
            "",
            tooltip="Path to folder containing extracted keypoint .npy files.",
            browse_dir=True,
        )
        _, self._batch_outdir_var = _labeled_entry(
            inner,
            "Output Directory:",
            "",
            tooltip="Base directory for retargeted motions and contact labels.",
            browse_dir=True,
        )

        _section_header(inner, "Output (auto-derived)")

        _, self._batch_ret_var = _readonly_entry(inner, "Retargeted Dir:")
        _, self._batch_contacts_var = _readonly_entry(inner, "Contacts Dir:")

        _section_header(inner, "Options")

        _, self._batch_source_var = _labeled_combo(
            inner,
            "Source Type:",
            SKELETON_FORMATS,
            "smpl",
            tooltip="Source skeleton type.",
        )
        _, self._batch_subsample_var = _labeled_spin(
            inner,
            "Subsample Factor:",
            1,
            100,
            1,
            tooltip="Subsample factor for keypoints.",
        )
        _, self._batch_frames_var = _labeled_spin(
            inner,
            "Target Raw Frames:",
            1,
            10000,
            450,
            tooltip="Target raw frames before subsampling (450 = 15s at 30fps).",
        )
        _, self._batch_fps_var = _labeled_spin(
            inner,
            "Input FPS:",
            1,
            120,
            30,
            tooltip="FPS of the input keypoint data.",
        )
        _, self._batch_skip_var = _labeled_check(
            inner,
            "Skip Existing",
            tooltip="Skip motions that already have output files.",
        )
        _, self._batch_novis_var = _labeled_check(
            inner,
            "No Visualize",
            default=True,
            tooltip="Run without MuJoCo visualization (batch mode).",
        )

        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        self._batch_run_btn = ttk.Button(
            btn_frame,
            text="  Run Steps 2 and 3  ",
            command=self._batch_run,
            style="Accent.TButton",
        )
        self._batch_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._batch_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._batch_cancel_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._batch_progress_var = tk.StringVar(value="")
        ttk.Label(
            btn_frame, textvariable=self._batch_progress_var, foreground=_MUTED
        ).pack(side=tk.LEFT)

        self._batch_config_var.trace_add("write", self._batch_update_derived)
        self._batch_kp_var.trace_add("write", self._batch_update_derived)
        self._batch_outdir_var.trace_add("write", self._batch_update_derived)
        self._batch_update_derived()

    def _batch_robot_slug(self) -> str:
        cfg_path = self._batch_config_var.get().strip()
        if cfg_path and os.path.isfile(cfg_path):
            try:
                cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
                name = cfg.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
            except Exception:
                pass
            return Path(cfg_path).stem
        return "robot"

    def _batch_update_derived(self, *_args: object) -> None:
        out_dir = self._batch_outdir_var.get().strip()
        kp_dir = self._batch_kp_var.get().strip()
        if not out_dir and kp_dir:
            out_dir = str(Path(kp_dir).parent)
            self._batch_outdir_var.set(out_dir)
            return
        if out_dir:
            robot = self._batch_robot_slug()
            self._batch_ret_var.set(os.path.join(out_dir, f"pyroki-retargeted-{robot}"))
            self._batch_contacts_var.set(os.path.join(out_dir, "contacts"))
        else:
            self._batch_ret_var.set("")
            self._batch_contacts_var.set("")

    def _validate_pyroki_python(self) -> str | None:
        pyroki = self._pyroki_python_var.get()
        if not os.path.isfile(pyroki):
            messagebox.showerror("Error", f"PyRoki Python not found: {pyroki}")
            return None
        return pyroki

    def _build_step2_command(
        self,
        *,
        config: str,
        kp_path: str,
        out_dir: str,
        source_type: str,
        subsample_factor: str,
        target_raw_frames: str,
        input_fps: str,
        skip_existing: bool,
        no_visualize: bool,
    ) -> list[str]:
        cmd = [
            self._pyroki_python_var.get(),
            "pyroki/retarget_from_keypoints.py",
            "--robot-config",
            config,
            "--keypoints-folder-path",
            kp_path,
            "--output-dir",
            out_dir,
            "--source-type",
            source_type,
            "--subsample-factor",
            subsample_factor,
            "--target-raw-frames",
            target_raw_frames,
            "--input-fps",
            input_fps,
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        if no_visualize:
            cmd.append("--no-visualize")
        return cmd

    def _build_step3_command(
        self,
        *,
        config: str,
        kp_path: str,
        source_type: str,
        contacts_dir: str,
        skip_existing: bool,
    ) -> list[str]:
        cmd = [
            self._pyroki_python_var.get(),
            "pyroki/retarget_from_keypoints.py",
            "--robot-config",
            config,
            "--keypoints-folder-path",
            kp_path,
            "--source-type",
            source_type,
            "--save-contacts-only",
        ]
        if contacts_dir:
            cmd += ["--contacts-dir", contacts_dir]
        if skip_existing:
            cmd.append("--skip-existing")
        return cmd

    def _batch_run(self) -> None:
        pyroki = self._validate_pyroki_python()
        if pyroki is None:
            return
        config = self._batch_config_var.get().strip()
        if not config or not os.path.isfile(config):
            messagebox.showerror("Error", f"Robot config not found: {config}")
            return
        kp_path = self._batch_kp_var.get().strip()
        if not kp_path or not os.path.isdir(kp_path):
            messagebox.showerror("Error", f"Keypoints folder not found: {kp_path}")
            return
        out_dir = self._batch_outdir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Output directory is required.")
            return

        ret_dir = self._batch_ret_var.get().strip()
        contacts_dir = self._batch_contacts_var.get().strip()
        step2_cmd = self._build_step2_command(
            config=config,
            kp_path=kp_path,
            out_dir=ret_dir,
            source_type=self._batch_source_var.get(),
            subsample_factor=self._batch_subsample_var.get(),
            target_raw_frames=self._batch_frames_var.get(),
            input_fps=self._batch_fps_var.get(),
            skip_existing=self._batch_skip_var.get(),
            no_visualize=self._batch_novis_var.get(),
        )
        step3_cmd = self._build_step3_command(
            config=config,
            kp_path=kp_path,
            source_type=self._batch_source_var.get(),
            contacts_dir=contacts_dir,
            skip_existing=self._batch_skip_var.get(),
        )

        self._log(
            f"\n[GUI] Batch run: pyroki={pyroki}, config={config}, output={out_dir}\n"
        )
        steps = [
            (
                "Running PyRoki retargeting",
                step2_cmd,
                str(PROJECT_ROOT),
                self._pyroki_env(),
            ),
            (
                "Extracting foot contact labels",
                step3_cmd,
                str(PROJECT_ROOT),
                self._pyroki_env(),
            ),
        ]
        self._set_running(True)
        self._multi.start(steps)

    def _build_tab_steps(self) -> None:
        tab = ttk.Frame(self._notebook, padding=0)
        self._notebook.add(tab, text=" Step-by-Step ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        s2 = ttk.LabelFrame(
            inner, text="Step 2 — PyRoki Retarget (PyRoki Python)", padding=8
        )
        s2.pack(fill=tk.X, padx=8, pady=4)

        _, self._s2_config_var = _labeled_entry(
            s2,
            "Robot Config (.yaml):",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Path to the YAML robot retarget config.",
            browse="file",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        _, self._s2_kp_var = _labeled_entry(
            s2,
            "Keypoints Folder:",
            "",
            tooltip="Path to folder containing the extracted keypoint .npy files.",
            browse_dir=True,
        )
        _, self._s2_outdir_var = _labeled_entry(
            s2,
            "Output Directory:",
            "",
            tooltip="Directory to save retargeted motion files.",
            browse_dir=True,
        )
        _, self._s2_source_var = _labeled_combo(
            s2,
            "Source Type:",
            SKELETON_FORMATS,
            "smpl",
            tooltip="Source skeleton type.",
        )
        _, self._s2_subsample_var = _labeled_spin(
            s2,
            "Subsample Factor:",
            1,
            100,
            1,
            tooltip="Subsample factor for keypoints.",
        )
        _, self._s2_frames_var = _labeled_spin(
            s2,
            "Target Raw Frames:",
            1,
            10000,
            450,
            tooltip="Target raw frames before subsampling (450 = 15s at 30fps).",
        )
        _, self._s2_fps_var = _labeled_spin(
            s2,
            "Input FPS:",
            1,
            120,
            30,
            tooltip="FPS of the input keypoint data.",
        )
        _, self._s2_skip_var = _labeled_check(
            s2,
            "Skip Existing",
            tooltip="Skip motions that already have output files.",
        )
        _, self._s2_novis_var = _labeled_check(
            s2,
            "No Visualize",
            default=True,
            tooltip="Run without MuJoCo visualization (batch mode).",
        )

        wf = ttk.LabelFrame(
            s2, text="Retargeting Weights (override config values)", padding=4
        )
        wf.pack(fill=tk.X, padx=4, pady=4)

        _, self._s2_w_local_var = _labeled_spin(
            wf,
            "Local Alignment:",
            0,
            100,
            1.0,
            increment=0.5,
            tooltip="Weight for local link position alignment.",
        )
        _, self._s2_w_global_var = _labeled_spin(
            wf,
            "Global Alignment:",
            0,
            100,
            4.0,
            increment=0.5,
            tooltip="Weight for global (root-relative) alignment.",
        )
        _, self._s2_w_root_smooth_var = _labeled_spin(
            wf,
            "Root Smoothness:",
            0,
            100,
            1.0,
            increment=0.5,
            tooltip="Weight for root rotation smoothness.",
        )
        _, self._s2_w_joint_smooth_var = _labeled_spin(
            wf,
            "Joint Smoothness:",
            0,
            100,
            4.0,
            increment=0.5,
            tooltip="Weight for joint angle smoothness.",
        )
        _, self._s2_w_self_coll_var = _labeled_spin(
            wf,
            "Self Collision:",
            0,
            100,
            0.0,
            increment=0.1,
            tooltip="Weight for self-collision avoidance.",
        )
        _, self._s2_w_rest_var = _labeled_spin(
            wf,
            "Joint Rest Penalty:",
            0,
            100,
            1.0,
            increment=0.5,
            tooltip="Penalty for deviating from rest pose.",
        )
        _, self._s2_w_vel_limit_var = _labeled_spin(
            wf,
            "Joint Vel Limit:",
            0,
            200,
            50.0,
            increment=5.0,
            tooltip="Weight for joint velocity limit enforcement.",
        )
        _, self._s2_w_foot_contact_var = _labeled_spin(
            wf,
            "Foot Contact:",
            0,
            200,
            30.0,
            increment=5.0,
            tooltip="Weight for foot contact constraint.",
        )
        _, self._s2_w_foot_tilt_var = _labeled_spin(
            wf,
            "Foot Tilt:",
            0,
            100,
            1.0,
            increment=0.1,
            tooltip="Weight for foot tilt constraint.",
        )
        _, self._s2_w_override_var = _labeled_check(
            wf,
            "Override config weights with values above",
            default=True,
            tooltip="When checked, the above weights replace those in the YAML config.",
        )
        ttk.Button(wf, text="Load from Config", command=self._s2_load_weights).pack(
            anchor="w", padx=4, pady=2
        )

        self._s2_run_btn = ttk.Button(s2, text="Run Step 2", command=self._run_step2)
        self._s2_run_btn.pack(anchor="w", padx=4, pady=4)

        s3 = ttk.LabelFrame(
            inner, text="Step 3 — Extract Contact Labels (PyRoki Python)", padding=8
        )
        s3.pack(fill=tk.X, padx=8, pady=4)

        _, self._s3_config_var = _labeled_entry(
            s3,
            "Robot Config (.yaml):",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Path to robot retarget config (same as Step 2).",
            browse="file",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        _, self._s3_kp_var = _labeled_entry(
            s3,
            "Keypoints Folder:",
            "",
            tooltip="Path to folder with extracted keypoints.",
            browse_dir=True,
        )
        _, self._s3_source_var = _labeled_combo(
            s3,
            "Source Type:",
            SKELETON_FORMATS,
            "smpl",
        )
        _, self._s3_contacts_var = _labeled_entry(
            s3,
            "Contacts Output Dir:",
            "",
            tooltip="Directory to save contact label files.",
            browse_dir=True,
        )
        _, self._s3_skip_var = _labeled_check(
            s3,
            "Skip Existing",
        )
        self._s3_run_btn = ttk.Button(s3, text="Run Step 3", command=self._run_step3)
        self._s3_run_btn.pack(anchor="w", padx=4, pady=4)

    def _s2_load_weights(self) -> None:
        """Load weight values from the currently selected robot config."""
        cfg_path = self._s2_config_var.get()
        if not cfg_path or not os.path.isfile(cfg_path):
            messagebox.showerror("Error", "Robot config file not found.")
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            w = cfg.get("weights", {})
            self._s2_w_local_var.set(str(w.get("local_alignment", 1.0)))
            self._s2_w_global_var.set(str(w.get("global_alignment", 4.0)))
            self._s2_w_root_smooth_var.set(str(w.get("root_smoothness", 1.0)))
            self._s2_w_joint_smooth_var.set(str(w.get("joint_smoothness", 4.0)))
            self._s2_w_self_coll_var.set(str(w.get("self_collision", 0.0)))
            self._s2_w_rest_var.set(str(w.get("joint_rest_penalty", 1.0)))
            self._s2_w_vel_limit_var.set(str(w.get("joint_vel_limit", 50.0)))
            self._s2_w_foot_contact_var.set(str(w.get("foot_contact", 30.0)))
            self._s2_w_foot_tilt_var.set(str(w.get("foot_tilt", 1.0)))
            self._log(f"[GUI] Loaded weights from: {cfg_path}\n")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load config:\n{exc}")

    def _s2_make_config_with_weights(self) -> str | None:
        """If weight override is enabled, create a temp config with adjusted weights.

        Returns the path to the temp config (or None if no override).
        """
        if not self._s2_w_override_var.get():
            return None
        cfg_path = self._s2_config_var.get()
        if not cfg_path or not os.path.isfile(cfg_path):
            return None
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["weights"] = {
                "local_alignment": float(self._s2_w_local_var.get()),
                "global_alignment": float(self._s2_w_global_var.get()),
                "root_smoothness": float(self._s2_w_root_smooth_var.get()),
                "joint_smoothness": float(self._s2_w_joint_smooth_var.get()),
                "self_collision": float(self._s2_w_self_coll_var.get()),
                "joint_rest_penalty": float(self._s2_w_rest_var.get()),
                "joint_vel_limit": float(self._s2_w_vel_limit_var.get()),
                "foot_contact": float(self._s2_w_foot_contact_var.get()),
                "foot_tilt": float(self._s2_w_foot_tilt_var.get()),
            }
            # Convert relative paths to absolute paths so they work from /tmp/
            cfg_dir = Path(cfg_path).parent
            for key in ("urdf_path", "mesh_dir"):
                if key in cfg:
                    path_val = cfg[key]
                    if path_val and not Path(path_val).is_absolute():
                        cfg[key] = str((cfg_dir / path_val).resolve())
            fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="retarget_cfg_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            self._log(f"[GUI] Created temp config with custom weights: {tmp_path}\n")
            return tmp_path
        except Exception as exc:
            self._log(f"[GUI] Warning: failed to create temp config: {exc}\n")
            return None

    def _run_step2(self) -> None:
        pyroki = self._validate_pyroki_python()
        if pyroki is None:
            return
        config = self._s2_make_config_with_weights() or self._s2_config_var.get()
        if not config or not os.path.isfile(config):
            messagebox.showerror("Error", f"Robot config not found: {config}")
            return
        kp_path = self._s2_kp_var.get()
        if not kp_path or not os.path.isdir(kp_path):
            messagebox.showerror("Error", f"Keypoints folder not found: {kp_path}")
            return
        out_dir = self._s2_outdir_var.get()
        if not out_dir:
            messagebox.showerror("Error", "Output directory is required.")
            return
        cmd = self._build_step2_command(
            config=config,
            kp_path=kp_path,
            out_dir=out_dir,
            source_type=self._s2_source_var.get(),
            subsample_factor=self._s2_subsample_var.get(),
            target_raw_frames=self._s2_frames_var.get(),
            input_fps=self._s2_fps_var.get(),
            skip_existing=self._s2_skip_var.get(),
            no_visualize=self._s2_novis_var.get(),
        )

        self._log(f"\n[GUI] Step 2: Retarget using PyRoki Python: {pyroki}\n")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT), env=self._pyroki_env())

    def _run_step3(self) -> None:
        pyroki = self._validate_pyroki_python()
        if pyroki is None:
            return
        config = self._s3_config_var.get()
        if not config or not os.path.isfile(config):
            messagebox.showerror("Error", f"Robot config not found: {config}")
            return
        kp_path = self._s3_kp_var.get()
        if not kp_path or not os.path.isdir(kp_path):
            messagebox.showerror("Error", f"Keypoints folder not found: {kp_path}")
            return
        cmd = self._build_step3_command(
            config=config,
            kp_path=kp_path,
            source_type=self._s3_source_var.get(),
            contacts_dir=self._s3_contacts_var.get(),
            skip_existing=self._s3_skip_var.get(),
        )

        self._log(f"\n[GUI] Step 3: Extract contacts using PyRoki Python: {pyroki}\n")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT), env=self._pyroki_env())

    def _build_tab_keypoint(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Keypoint Config ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        launch_frame = ttk.LabelFrame(
            inner,
            text="Launch Keypoint Mapping Tuner (MuJoCo Viewer)",
            padding=8,
        )
        launch_frame.pack(fill=tk.X, padx=4, pady=4)

        _, self._kp_config_var = _labeled_entry(
            launch_frame,
            "Robot Config:",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Robot config YAML for the keypoint mapping tuner.",
            browse="file",
            filetypes=[("YAML", "*.yaml *.yml")],
        )
        _, self._kp_source_var = _labeled_combo(
            launch_frame,
            "Source Type:",
            ("smpl", "rigv1"),
            "smpl",
        )
        _, self._kp_spacing_var = _labeled_spin(
            launch_frame,
            "Viewer Spacing:",
            0.1,
            5.0,
            1.0,
            increment=0.1,
        )
        ttk.Button(
            launch_frame,
            text="  Launch Keypoint Tuner GUI  ",
            command=self._launch_keypoint_gui,
            style="Accent.TButton",
        ).pack(anchor="w", padx=4, pady=6)

        editor_frame = ttk.LabelFrame(inner, text="Robot Config Editor", padding=8)
        editor_frame.pack(fill=tk.X, padx=4, pady=4)

        _, self._cfg_path_var = _labeled_entry(
            editor_frame,
            "Config File:",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Robot config YAML to load and edit.",
            browse="file",
            filetypes=[("YAML", "*.yaml *.yml")],
        )

        btn_row = ttk.Frame(editor_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Load Config", command=self._load_robot_config).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row, text="Save Config", command=self._save_robot_config).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            btn_row, text="New from Template", command=self._new_robot_config
        ).pack(side=tk.LEFT)

        self._cfg_text = tk.Text(
            editor_frame,
            height=30,
            width=90,
            font=_MONO_FONT,
            bg=_LOG_BG,
            fg=_LOG_FG,
            insertbackground=_LOG_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=8,
        )
        cfg_scroll = ttk.Scrollbar(
            editor_frame, orient=tk.VERTICAL, command=self._cfg_text.yview
        )
        self._cfg_text.configure(yscrollcommand=cfg_scroll.set)
        self._cfg_text.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4
        )
        cfg_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

    def _launch_keypoint_gui(self) -> None:
        pyroki = self._pyroki_python_var.get()
        cmd = [
            pyroki,
            str(SCRIPT_DIR / "visualize_keypoint_mapping_gui.py"),
            "--robot-config",
            self._kp_config_var.get(),
            "--source-type",
            self._kp_source_var.get(),
            "--spacing",
            self._kp_spacing_var.get(),
        ]
        self._log("\n[GUI] Launching keypoint tuner as separate process...\n")
        self._log(f"[GUI] Using PyRoki Python: {pyroki}\n")
        self._log(f"[GUI] $ {' '.join(cmd)}\n")
        try:
            merged_env = dict(os.environ)
            pyroki_dir = str(PROJECT_ROOT / "pyroki")
            existing = merged_env.get("PYTHONPATH", "")
            merged_env["PYTHONPATH"] = (
                f"{pyroki_dir}:{existing}" if existing else pyroki_dir
            )
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=merged_env)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to launch: {exc}")

    def _load_robot_config(self) -> None:
        path = self._cfg_path_var.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Config file not found.")
            return
        try:
            text = Path(path).read_text()
            self._cfg_text.delete("1.0", tk.END)
            self._cfg_text.insert("1.0", text)
            self._log(f"[GUI] Loaded config: {path}\n")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _save_robot_config(self) -> None:
        path = self._cfg_path_var.get()
        if not path:
            path = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml *.yml")],
            )
        if not path:
            return
        try:
            text = self._cfg_text.get("1.0", tk.END)
            Path(path).write_text(text)
            self._log(f"[GUI] Saved config: {path}\n")
            messagebox.showinfo("Success", f"Config saved to {path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _new_robot_config(self) -> None:
        template = """\
# Robot retargeting config — created from template
name: "my_robot"

# Path to URDF (used by PyRoki for FK)
urdf_path: "path/to/robot.urdf"
mesh_dir: "path/to/meshes/"

# Keypoint mapping: human_keypoint -> robot_link_name
keypoint_mapping:
  - [pelvis, pelvis_link]
  - [left_hip, left_hip_link]
  - [right_hip, right_hip_link]
  - [left_knee, left_knee_link]
  - [right_knee, right_knee_link]
  - [left_ankle, left_ankle_link]
  - [right_ankle, right_ankle_link]
  - [left_foot, left_foot_link]
  - [right_foot, right_foot_link]
  - [left_shoulder, left_shoulder_link]
  - [right_shoulder, right_shoulder_link]
  - [left_elbow, left_elbow_link]
  - [right_elbow, right_elbow_link]
  - [left_wrist, left_wrist_link]
  - [right_wrist, right_wrist_link]

# Scale factors per source type
scale_factors:
  smpl:
    root: [1.0, 1.0, 1.0]
    lower_body: [1.0, 1.0, 1.0]
    upper_body: [1.0, 1.0, 1.0]
  rigv1:
    root: [1.0, 1.0, 1.0]
    lower_body: [1.0, 1.0, 1.0]
    upper_body: [1.0, 1.0, 1.0]

# Optimization weights
weights:
  local_alignment: 3.0
  global_alignment: 0.5
  root_smoothness: 0.5
  joint_smoothness: 0.5
  limit_cost: 0.1
  joint_vel_limit: 0.5
  foot_contact: 1.0
  foot_tilt: 0.3
  root_alignment: 0.5

# Auxiliary offsets
hand_aux_offset: [0.0, -0.05, 0.0]

torso_link_name: "torso_link"
torso_aux_offset: [0.15, 0.0, 0.0]

downweight_hips: false
joints_to_move_less: []
display_pose_preset: null
"""
        self._cfg_text.delete("1.0", tk.END)
        self._cfg_text.insert("1.0", template)
        self._cfg_path_var.set("")
        self._log("[GUI] New config created from template.\n")

    def _get_run_buttons(self) -> list[ttk.Button]:
        return [self._batch_run_btn, self._s2_run_btn, self._s3_run_btn]

    def _get_cancel_buttons(self) -> list[ttk.Button]:
        return [self._batch_cancel_btn]


if __name__ == "__main__":
    main()
