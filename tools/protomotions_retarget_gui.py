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
"""ProtoMotions-only retargeting GUI.

This GUI intentionally contains only the ProtoMotions-environment portions of
the retargeting workflow: Steps 1, 4, and 5 plus visualization and asset tools.
Steps 2 and 3 must be run in the separate PyRoki GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from retarget_gui_common import (
    _ACCENT,
    _LOG_BG,
    _LOG_FG,
    _MONO_FONT,
    _MUTED,
    _TEXT,
    PROJECT_ROOT,
    ROBOT_TYPES,
    SIMULATORS,
    SKELETON_FORMATS,
    VISUALIZER_ROBOTS,
    RetargetGUIBase,
    _labeled_check,
    _labeled_combo,
    _labeled_entry,
    _labeled_spin,
    _readonly_entry,
    _section_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProtoMotions-only retargeting GUI for steps 1, 4, and 5."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    app = ProtoMotionsRetargetGUI()
    app.run()


class ProtoMotionsRetargetGUI(RetargetGUIBase):
    """ProtoMotions-environment GUI split from the monolithic retargeting app."""

    def __init__(self) -> None:
        super().__init__(
            title="ProtoMotions — Retargeting Tools (Proto Environment)",
            env_mode="proto",
        )
        self._build_notebook()

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._build_tab_batch()
        self._build_tab_single()
        self._build_tab_steps()
        self._build_tab_visualize()
        self._build_tab_usd()
        self._build_tab_joints()

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

        _, self._batch_pt_var = _labeled_entry(
            inner,
            "AMASS .pt File:",
            "",
            tooltip="Path to packaged AMASS MotionLib .pt file.",
            browse="file",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        _, self._batch_robot_var = _labeled_combo(
            inner,
            "Robot Type:",
            ROBOT_TYPES,
            "astro",
            tooltip="Target robot morphology.",
        )
        _, self._batch_skip_var = _labeled_spin(
            inner,
            "Skip Frequency:",
            1,
            1000,
            1,
            tooltip="Skip every N motions (1 = process all). Useful for quick testing.",
        )
        _, self._batch_skel_var = _labeled_combo(
            inner,
            "Skeleton Format:",
            SKELETON_FORMATS,
            "smpl",
            tooltip="Source skeleton format of the AMASS data.",
        )

        _section_header(inner, "Output (auto-derived)")

        _, self._batch_outdir_var = _labeled_entry(
            inner,
            "Output Directory:",
            "",
            tooltip="Base directory for all outputs (defaults to directory of input .pt).",
            browse_dir=True,
        )
        self._batch_pt_var.trace_add("write", self._batch_update_derived)
        self._batch_robot_var.trace_add("write", self._batch_update_derived)

        _, self._batch_kp_var = _readonly_entry(inner, "Keypoints Dir:")
        _, self._batch_ret_var = _readonly_entry(inner, "Retargeted Dir:")
        _, self._batch_contacts_var = _readonly_entry(inner, "Contacts Dir:")
        _, self._batch_proto_var = _readonly_entry(inner, "Proto Dir:")
        _, self._batch_final_var = _readonly_entry(inner, "Final .pt:")

        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        self._batch_run_btn = ttk.Button(
            btn_frame,
            text="  Run Steps 1, 4, 5  ",
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

    def _batch_update_derived(self, *_args: Any) -> None:
        pt_path = self._batch_pt_var.get()
        robot = self._batch_robot_var.get()
        if pt_path:
            out_dir = self._batch_outdir_var.get() or os.path.dirname(pt_path)
            if not self._batch_outdir_var.get():
                self._batch_outdir_var.set(out_dir)
        else:
            out_dir = self._batch_outdir_var.get()
        if out_dir:
            self._batch_kp_var.set(os.path.join(out_dir, "keypoints-for-retarget"))
            self._batch_ret_var.set(os.path.join(out_dir, f"pyroki-retargeted-{robot}"))
            self._batch_contacts_var.set(os.path.join(out_dir, "contacts"))
            self._batch_proto_var.set(os.path.join(out_dir, f"proto-{robot}"))
            self._batch_final_var.set(os.path.join(out_dir, f"proto-{robot}.pt"))

    def _validate_step4_prereqs(self, ret_dir: str, contacts_dir: str) -> bool:
        ret_ok = os.path.isdir(ret_dir) and any(
            path.is_file() for path in Path(ret_dir).iterdir()
        )
        contacts_ok = os.path.isdir(contacts_dir) and any(
            path.is_file() for path in Path(contacts_dir).iterdir()
        )
        if ret_ok and contacts_ok:
            return True
        messagebox.showerror(
            "Error",
            "Retargeted motion files not found. Run Steps 2 and 3 in the PyRoki GUI first.",
        )
        return False

    def _batch_run(self) -> None:
        pt_file = self._batch_pt_var.get()
        if not pt_file or not os.path.isfile(pt_file):
            messagebox.showerror("Error", "Please select a valid AMASS .pt file.")
            return

        proto = self._proto_python_var.get()
        if not os.path.isfile(proto):
            messagebox.showerror("Error", f"ProtoMotions Python not found: {proto}")
            return
        robot = self._batch_robot_var.get()

        skip = self._batch_skip_var.get()
        skel = self._batch_skel_var.get()
        out_dir = self._batch_outdir_var.get() or os.path.dirname(pt_file)
        kp_dir = os.path.join(out_dir, "keypoints-for-retarget")
        ret_dir = os.path.join(out_dir, f"pyroki-retargeted-{robot}")
        contacts_dir = os.path.join(out_dir, "contacts")
        proto_dir = os.path.join(out_dir, f"proto-{robot}")
        final_pt = os.path.join(out_dir, f"proto-{robot}.pt")
        cwd = str(PROJECT_ROOT)

        if not self._validate_step4_prereqs(ret_dir, contacts_dir):
            return

        steps = [
            (
                "Extracting keypoints from SMPL motions",
                [
                    proto,
                    "data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py",
                    pt_file,
                    "--output-path",
                    kp_dir,
                    "--skeleton-format",
                    skel,
                    "--start-idx",
                    "0",
                    "--skip-freq",
                    str(skip),
                ],
                cwd,
            ),
            (
                "Converting to ProtoMotions format",
                [
                    proto,
                    "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
                    "--retargeted-motion-dir",
                    ret_dir,
                    "--output-dir",
                    proto_dir,
                    "--robot-type",
                    robot,
                    "--contact-labels-dir",
                    contacts_dir,
                    "--apply-motion-filter",
                    "--force-remake",
                ],
                cwd,
            ),
            (
                "Packaging into MotionLib .pt",
                [
                    proto,
                    "protomotions/components/motion_lib.py",
                    "--motion-path",
                    proto_dir,
                    "--output-file",
                    final_pt,
                ],
                cwd,
            ),
        ]

        self._set_running(True)
        self._multi.start(steps)

    def _build_tab_single(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Single Motion ")

        _section_header(tab, "Input")

        _, self._single_motion_var = _labeled_entry(
            tab,
            "Motion File (.motion):",
            "",
            tooltip="Path to a single .motion file in SMPL format.",
            browse="file",
            filetypes=[("Motion", "*.motion"), ("All", "*.*")],
        )
        _, self._single_robot_var = _labeled_combo(
            tab,
            "Robot Type:",
            ROBOT_TYPES,
            "astro",
            tooltip="Target robot morphology.",
        )
        _, self._single_outdir_var = _labeled_entry(
            tab,
            "Output Directory:",
            "",
            tooltip="Directory where all intermediate and final outputs will be saved.",
            browse_dir=True,
        )

        _section_header(tab, "Output (auto-derived)")

        _, self._single_kp_var = _readonly_entry(tab, "Keypoints Dir:")
        _, self._single_ret_var = _readonly_entry(tab, "Retargeted Dir:")
        _, self._single_contacts_var = _readonly_entry(tab, "Contacts Dir:")
        _, self._single_proto_var = _readonly_entry(tab, "Proto Dir:")
        _, self._single_final_var = _readonly_entry(tab, "Final .pt:")

        self._single_outdir_var.trace_add("write", self._single_update_derived)
        self._single_robot_var.trace_add("write", self._single_update_derived)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        self._single_run_btn = ttk.Button(
            btn_frame,
            text="  Run Steps 1, 4, 5  ",
            command=self._single_run,
            style="Accent.TButton",
        )
        self._single_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._single_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._single_cancel_btn.pack(side=tk.LEFT)

    def _single_update_derived(self, *_args: Any) -> None:
        out = self._single_outdir_var.get()
        robot = self._single_robot_var.get()
        if out:
            self._single_kp_var.set(os.path.join(out, "keypoints"))
            self._single_ret_var.set(os.path.join(out, f"retargeted_{robot}"))
            self._single_contacts_var.set(os.path.join(out, "contacts"))
            self._single_proto_var.set(os.path.join(out, f"retargeted_{robot}_proto"))
            self._single_final_var.set(
                os.path.join(out, f"retargeted_{robot}_proto.pt")
            )

    def _single_run(self) -> None:
        motion = self._single_motion_var.get()
        if not motion or not os.path.isfile(motion):
            messagebox.showerror("Error", "Please select a valid .motion file.")
            return
        out_dir = self._single_outdir_var.get()
        if not out_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        proto = self._proto_python_var.get()
        if not os.path.isfile(proto):
            messagebox.showerror("Error", f"ProtoMotions Python not found: {proto}")
            return
        robot = self._single_robot_var.get()
        cwd = str(PROJECT_ROOT)

        kp_dir = os.path.join(out_dir, "keypoints")
        ret_dir = os.path.join(out_dir, f"retargeted_{robot}")
        contacts_dir = os.path.join(out_dir, "contacts")
        proto_dir = os.path.join(out_dir, f"retargeted_{robot}_proto")
        final_pt = os.path.join(out_dir, f"retargeted_{robot}_proto.pt")

        if not self._validate_step4_prereqs(ret_dir, contacts_dir):
            return

        steps = [
            (
                "Extracting keypoints from single motion",
                [
                    proto,
                    "data/scripts/extract_keypoints_from_single_motion.py",
                    motion,
                    "--output-path",
                    kp_dir,
                    "--skeleton-format",
                    "smpl",
                    "--force-remake",
                ],
                cwd,
            ),
            (
                "Converting to ProtoMotions format",
                [
                    proto,
                    "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
                    "--retargeted-motion-dir",
                    ret_dir,
                    "--output-dir",
                    proto_dir,
                    "--robot-type",
                    robot,
                    "--contact-labels-dir",
                    contacts_dir,
                    "--apply-motion-filter",
                    "--force-remake",
                ],
                cwd,
            ),
            (
                "Packaging into MotionLib .pt",
                [
                    proto,
                    "protomotions/components/motion_lib.py",
                    "--motion-path",
                    proto_dir,
                    "--output-file",
                    final_pt,
                ],
                cwd,
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

        s1 = ttk.LabelFrame(
            inner,
            text="Step 1 — Extract Keypoints (ProtoMotions Python)",
            padding=8,
        )
        s1.pack(fill=tk.X, padx=8, pady=4)

        _, self._s1_mode_var = _labeled_combo(
            s1,
            "Source Mode:",
            ("From .pt (batch)", "From .motion (single)"),
            tooltip="Extract from a packaged .pt or a single .motion file.",
        )
        _, self._s1_input_var = _labeled_entry(
            s1,
            "Input File:",
            "",
            tooltip=".pt file (batch) or .motion file (single).",
            browse="file",
            filetypes=[("Motion data", "*.pt *.motion"), ("All", "*.*")],
        )
        _, self._s1_skel_var = _labeled_combo(
            s1,
            "Skeleton Format:",
            SKELETON_FORMATS,
            "smpl",
            tooltip="smpl or rigv1 skeleton format.",
        )
        _, self._s1_output_var = _labeled_entry(
            s1,
            "Output Path:",
            "",
            tooltip="Directory to save extracted keypoint .npy files.",
            browse_dir=True,
        )
        _, self._s1_force_var = _labeled_check(
            s1,
            "Force Remake",
            tooltip="Overwrite existing keypoint files.",
        )
        _, self._s1_start_var = _labeled_spin(
            s1,
            "Start Index:",
            0,
            100000,
            0,
            tooltip="Start from this motion index (batch mode only).",
        )
        _, self._s1_end_var = _labeled_entry(
            s1,
            "End Index:",
            "",
            tooltip="End at this motion index (empty = process all). Leave blank to process all motions.",
        )
        _, self._s1_skip_var = _labeled_spin(
            s1,
            "Skip Frequency:",
            1,
            1000,
            1,
            tooltip="Skip every N motions (batch mode only).",
        )
        ttk.Button(s1, text="Run Step 1", command=self._run_step1).pack(
            anchor="w", padx=4, pady=4
        )

        s4 = ttk.LabelFrame(
            inner,
            text="Step 4 — Convert to ProtoMotions Format (ProtoMotions Python)",
            padding=8,
        )
        s4.pack(fill=tk.X, padx=8, pady=4)

        _, self._s4_retdir_var = _labeled_entry(
            s4,
            "Retargeted Motion Dir:",
            "",
            tooltip="Directory with retargeted motion files (.npz or .csv).",
            browse_dir=True,
        )
        _, self._s4_outdir_var = _labeled_entry(
            s4,
            "Output Dir:",
            "",
            tooltip="Directory to save ProtoMotions .motion files.",
            browse_dir=True,
        )
        _, self._s4_robot_var = _labeled_combo(s4, "Robot Type:", ROBOT_TYPES, "astro")
        _, self._s4_contacts_var = _labeled_entry(
            s4,
            "Contact Labels Dir:",
            "",
            tooltip="Directory with contact label files (optional).",
            browse_dir=True,
        )
        _, self._s4_filter_var = _labeled_check(
            s4,
            "Apply Motion Filter",
            default=True,
            tooltip="Apply motion quality filter (height, velocity, DOF limits).",
        )
        _, self._s4_force_var = _labeled_check(s4, "Force Remake")
        _, self._s4_ifps_var = _labeled_spin(s4, "Input FPS:", 1, 120, 30)
        _, self._s4_ofps_var = _labeled_spin(s4, "Output FPS:", 1, 120, 30)
        _, self._s4_ignore_var = _labeled_spin(
            s4,
            "Ignore First N Frames:",
            0,
            1000,
            0,
            tooltip="Skip the first N frames of each motion.",
        )

        filter_frame = ttk.LabelFrame(s4, text="Motion Filter Thresholds", padding=4)
        filter_frame.pack(fill=tk.X, padx=4, pady=4)
        _, self._s4_min_h_var = _labeled_spin(
            filter_frame,
            "Min Height:",
            -10,
            10,
            -0.05,
            increment=0.01,
            tooltip="Minimum height threshold for motion filter.",
        )
        _, self._s4_max_v_var = _labeled_spin(
            filter_frame,
            "Max Velocity:",
            0,
            100,
            15.0,
            increment=1.0,
            tooltip="Maximum velocity threshold.",
        )
        _, self._s4_max_dof_var = _labeled_spin(
            filter_frame,
            "Max DOF Velocity:",
            0,
            200,
            40.0,
            increment=1.0,
            tooltip="Maximum DOF velocity threshold.",
        )
        _, self._s4_dur_h_var = _labeled_spin(
            filter_frame,
            "Duration Height Filter:",
            0,
            5,
            0.1,
            increment=0.01,
            tooltip="Height threshold for duration filter.",
        )
        _, self._s4_dur_s_var = _labeled_spin(
            filter_frame,
            "Duration Height Secs:",
            0,
            10,
            0.6,
            increment=0.1,
            tooltip="Duration in seconds for height filter.",
        )
        ttk.Button(s4, text="Run Step 4", command=self._run_step4).pack(
            anchor="w", padx=4, pady=4
        )

        s5 = ttk.LabelFrame(
            inner,
            text="Step 5 — Package MotionLib (ProtoMotions Python)",
            padding=8,
        )
        s5.pack(fill=tk.X, padx=8, pady=4)

        _, self._s5_motdir_var = _labeled_entry(
            s5,
            "Motion Path:",
            "",
            tooltip="Path to directory of .motion files (or a YAML listing motions).",
            browse_dir=True,
        )
        _, self._s5_output_var = _labeled_entry(
            s5,
            "Output .pt Filename:",
            "",
            tooltip="Output .pt file path. Type a name or use Browse to pick a save location.",
            browse="savefile",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        ttk.Button(s5, text="Run Step 5", command=self._run_step5).pack(
            anchor="w", padx=4, pady=4
        )

    def _run_step1(self) -> None:
        proto = self._proto_python_var.get()
        mode = self._s1_mode_var.get()
        inp = self._s1_input_var.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        cmd: list[str]
        if "batch" in mode.lower() or ".pt" in mode.lower():
            end_idx_val = self._s1_end_var.get().strip()
            cmd = [
                proto,
                "data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py",
                inp,
                "--skeleton-format",
                self._s1_skel_var.get(),
                "--start-idx",
                self._s1_start_var.get(),
                "--skip-freq",
                self._s1_skip_var.get(),
            ]
            if end_idx_val:
                cmd += ["--end-idx", end_idx_val]
            if self._s1_output_var.get():
                cmd += ["--output-path", self._s1_output_var.get()]
            if self._s1_force_var.get():
                cmd += ["--force-remake"]
        else:
            cmd = [
                proto,
                "data/scripts/extract_keypoints_from_single_motion.py",
                inp,
                "--skeleton-format",
                self._s1_skel_var.get(),
            ]
            if self._s1_output_var.get():
                cmd += ["--output-path", self._s1_output_var.get()]
            if self._s1_force_var.get():
                cmd += ["--force-remake"]

        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    def _run_step4(self) -> None:
        proto = self._proto_python_var.get()
        if not os.path.isfile(proto):
            messagebox.showerror("Error", f"ProtoMotions Python not found: {proto}")
            return
        retdir = self._s4_retdir_var.get()
        if not retdir or not os.path.isdir(retdir):
            messagebox.showerror(
                "Error", f"Retargeted motion directory not found: {retdir}"
            )
            return
        out_dir = self._s4_outdir_var.get()
        if not out_dir:
            messagebox.showerror("Error", "Output directory is required.")
            return
        robot_type = self._s4_robot_var.get()
        if not robot_type:
            messagebox.showerror("Error", "Robot type is required.")
            return
        contacts_dir = self._s4_contacts_var.get()
        if contacts_dir and not os.path.isdir(contacts_dir):
            messagebox.showerror(
                "Error", f"Contact labels directory not found: {contacts_dir}"
            )
            return
        cmd = [
            proto,
            "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
            "--retargeted-motion-dir",
            retdir,
            "--output-dir",
            out_dir,
            "--robot-type",
            robot_type,
            "--input-fps",
            self._s4_ifps_var.get(),
            "--output-fps",
            self._s4_ofps_var.get(),
            "--ignore-first-n-frames",
            self._s4_ignore_var.get(),
            "--min-height-threshold",
            self._s4_min_h_var.get(),
            "--max-velocity-threshold",
            self._s4_max_v_var.get(),
            "--max-dof-vel-threshold",
            self._s4_max_dof_var.get(),
            "--duration-height-filter",
            self._s4_dur_h_var.get(),
            "--duration-height-seconds",
            self._s4_dur_s_var.get(),
        ]
        if contacts_dir:
            cmd += ["--contact-labels-dir", contacts_dir]
        if self._s4_filter_var.get():
            cmd.append("--apply-motion-filter")
        if self._s4_force_var.get():
            cmd.append("--force-remake")

        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    def _run_step5(self) -> None:
        proto = self._proto_python_var.get()
        mot_path = self._s5_motdir_var.get()
        out_file = self._s5_output_var.get().strip()
        if not mot_path or not os.path.exists(mot_path):
            messagebox.showerror("Error", f"Motion path not found: {mot_path}")
            return
        if not out_file:
            messagebox.showerror("Error", "Please provide an output .pt filename.")
            return
        if not out_file.endswith(".pt"):
            out_file += ".pt"
            self._s5_output_var.set(out_file)
        if os.sep not in out_file and "/" not in out_file:
            parent = os.path.dirname(mot_path) if os.path.isfile(mot_path) else mot_path
            out_file = os.path.join(
                os.path.dirname(parent) if os.path.isfile(mot_path) else parent,
                out_file,
            )
            self._s5_output_var.set(out_file)
            self._log(f"[GUI] Output file resolved to: {out_file}\n")
        cmd = [
            proto,
            "protomotions/components/motion_lib.py",
            "--motion-path",
            mot_path,
            "--output-file",
            out_file,
        ]
        self._log(f"\n[GUI] Step 5: Packaging motions from {mot_path} → {out_file}\n")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    def _run_visualizer(self) -> None:
        proto = self._proto_python_var.get()
        files = self._sv_files_var.get().strip().split()
        if not files:
            messagebox.showerror("Error", "Please specify at least one motion file.")
            return
        cmd = [
            proto,
            "examples/motion_libs_visualizer.py",
            "--robot",
            self._sv_robot_var.get(),
            "--simulator",
            self._sv_sim_var.get(),
            "--playback_speed",
            self._sv_speed_var.get(),
        ]
        cmd += ["--motion_files"] + files
        if self._sv_headless_var.get():
            cmd.append("--headless")
        if self._sv_cpu_var.get():
            cmd.append("--cpu-only")

        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    def _build_tab_visualize(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text=" Visualize ")

        _section_header(tab, "Motion Visualizer")
        ttk.Label(
            tab,
            text=(
                "Preview retargeted motions in the simulator.  Supports .pt (packaged "
                "MotionLib) and individual .motion files.  Multiple files can be "
                "space-separated."
            ),
            wraplength=850,
            foreground=_MUTED,
        ).pack(fill=tk.X, padx=6, pady=(0, 10))

        _, self._sv_files_var = _labeled_entry(
            tab,
            "Motion Files:",
            "",
            tooltip="Path(s) to .pt or .motion files, space-separated.",
            browse="file",
            filetypes=[("Motion", "*.pt *.motion"), ("All", "*.*")],
        )
        _, self._sv_robot_var = _labeled_combo(
            tab,
            "Robot:",
            VISUALIZER_ROBOTS,
            "astro",
            tooltip="Robot to visualize motions on.",
        )
        _, self._sv_sim_var = _labeled_combo(
            tab,
            "Simulator:",
            SIMULATORS,
            "isaaclab",
            tooltip="Physics simulator backend for playback.",
        )
        _, self._sv_speed_var = _labeled_spin(
            tab,
            "Playback Speed:",
            0.1,
            10,
            1.0,
            increment=0.1,
            tooltip="Playback speed multiplier (1.0 = real-time).",
        )
        _, self._sv_headless_var = _labeled_check(
            tab,
            "Headless",
            tooltip="Run without GUI window (useful for recording).",
        )
        _, self._sv_cpu_var = _labeled_check(
            tab,
            "CPU Only",
            tooltip="Force CPU-only mode (no GPU required).",
        )

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=6, pady=(18, 4))
        self._sv_run_btn = ttk.Button(
            btn_frame,
            text="  Launch Visualizer  ",
            command=self._run_visualizer,
            style="Accent.TButton",
        )
        self._sv_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._sv_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._sv_cancel_btn.pack(side=tk.LEFT)

    def _build_tab_usd(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" USD Conversion ")

        _section_header(tab, "IsaacLab Asset Generation (MJCF → USD)")
        ttk.Label(
            tab,
            text=(
                "Convert a MuJoCo MJCF file to USDA format for IsaacLab.\n"
                "Step 1 flattens the MJCF (resolves defaults). "
                "Step 2 converts to USD (requires IsaacLab environment)."
            ),
            wraplength=800,
            foreground=_MUTED,
        ).pack(fill=tk.X, padx=4, pady=(0, 8))

        _, self._usd_input_var = _labeled_entry(
            tab,
            "Input MJCF (.xml):",
            "",
            tooltip="Path to the MJCF XML file to convert.",
            browse="file",
            filetypes=[("XML", "*.xml"), ("All", "*.*")],
        )
        _, self._usd_outdir_var = _labeled_entry(
            tab,
            "Output Directory:",
            "",
            tooltip="Output directory for USD files (optional, defaults to same dir).",
            browse_dir=True,
        )
        _, self._usd_noverify_var = _labeled_check(
            tab,
            "Skip MuJoCo Verification (flatten)",
            tooltip="Skip MuJoCo model verification after flattening.",
        )

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        ttk.Button(btn_frame, text="1. Flatten MJCF", command=self._usd_flatten).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_frame, text="2. Convert to USD", command=self._usd_convert).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            btn_frame,
            text="  Run Both  ",
            command=self._usd_both,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._usd_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._usd_cancel_btn.pack(side=tk.LEFT)

        _, self._usd_flat_var = _readonly_entry(
            tab,
            "Flattened MJCF:",
            "",
            tooltip="Auto-derived path to the flattened MJCF output.",
        )
        self._usd_input_var.trace_add("write", self._usd_update_flat)

    def _usd_update_flat(self, *_args: Any) -> None:
        inp = self._usd_input_var.get()
        if inp:
            stem = Path(inp).stem
            directory = os.path.dirname(inp)
            self._usd_flat_var.set(os.path.join(directory, f"{stem}_flat.xml"))

    def _usd_flatten(self) -> None:
        proto = self._proto_python_var.get()
        inp = self._usd_input_var.get()
        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Error", "Please select a valid MJCF file.")
            return
        cmd = [proto, "usd_convert/flatten_mjcf.py", inp]
        if self._usd_noverify_var.get():
            cmd.append("--no-verify")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    def _usd_convert(self) -> None:
        proto = self._proto_python_var.get()
        flat = self._usd_flat_var.get()
        if not flat or not os.path.isfile(flat):
            messagebox.showerror(
                "Error", "Flattened MJCF not found. Run 'Flatten' first."
            )
            return
        cmd = [proto, "usd_convert/convert_robot_mjcf_to_usda.py", flat]
        if self._usd_outdir_var.get():
            cmd += ["--output-dir", self._usd_outdir_var.get()]
        # Environment variables to fix GLXBadFBConfig on Wayland/NVIDIA
        env = {
            "DISPLAY": ":0",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "MESA_GL_VERSION_OVERRIDE": "4.6",
        }
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT), env=env)

    def _usd_both(self) -> None:
        proto = self._proto_python_var.get()
        inp = self._usd_input_var.get()
        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Error", "Please select a valid MJCF file.")
            return

        flat_path = self._usd_flat_var.get()
        flatten_cmd = [proto, "usd_convert/flatten_mjcf.py", inp]
        if self._usd_noverify_var.get():
            flatten_cmd.append("--no-verify")

        convert_cmd = [proto, "usd_convert/convert_robot_mjcf_to_usda.py", flat_path]
        if self._usd_outdir_var.get():
            convert_cmd += ["--output-dir", self._usd_outdir_var.get()]

        # Environment variables to fix GLXBadFBConfig on Wayland/NVIDIA
        env = {
            "DISPLAY": ":0",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "MESA_GL_VERSION_OVERRIDE": "4.6",
        }
        steps = [
            ("Flattening MJCF", flatten_cmd, str(PROJECT_ROOT), None),
            ("Converting to USD", convert_cmd, str(PROJECT_ROOT), env),
        ]
        self._set_running(True)
        self._multi.start(steps)

    def _build_tab_joints(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Joint Order ")

        _section_header(tab, "URDF vs MJCF Joint/Body Order Comparison")
        ttk.Label(
            tab,
            text=(
                "Compare the joint and body ordering between a URDF (used in PyRoki retargeting) "
                "and an MJCF (used in simulation/training). Mismatches can cause silent data "
                "corruption in motion data."
            ),
            wraplength=800,
            foreground=_MUTED,
        ).pack(fill=tk.X, padx=4, pady=(0, 8))

        _, self._jc_urdf_var = _labeled_entry(
            tab,
            "URDF File:",
            "",
            tooltip="URDF file used for PyRoki retargeting.",
            browse="file",
            filetypes=[("URDF", "*.urdf"), ("All", "*.*")],
        )
        _, self._jc_mjcf_var = _labeled_entry(
            tab,
            "MJCF File:",
            str(
                PROJECT_ROOT
                / "protomotions"
                / "data"
                / "assets"
                / "mjcf"
                / "g1_bm_box_feet.xml"
            ),
            tooltip="MJCF file used for simulation training.",
            browse="file",
            filetypes=[("XML", "*.xml"), ("All", "*.*")],
        )

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(
            btn_row,
            text="  Compare Joint Order  ",
            command=self._compare_joints,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._jc_status_var = tk.StringVar(value="")
        ttk.Label(
            btn_row,
            textvariable=self._jc_status_var,
            font=("Helvetica", 10, "bold"),
        ).pack(side=tk.LEFT)

        columns = ("idx", "urdf_joint", "mjcf_joint", "match")
        self._jc_tree = ttk.Treeview(tab, columns=columns, show="headings", height=20)
        self._jc_tree.heading("idx", text="#")
        self._jc_tree.heading("urdf_joint", text="URDF Joint")
        self._jc_tree.heading("mjcf_joint", text="MJCF Joint (DOF)")
        self._jc_tree.heading("match", text="Match")
        self._jc_tree.column("idx", width=40, anchor="center")
        self._jc_tree.column("urdf_joint", width=280)
        self._jc_tree.column("mjcf_joint", width=280)
        self._jc_tree.column("match", width=60, anchor="center")

        jc_scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=self._jc_tree.yview)
        self._jc_tree.configure(yscrollcommand=jc_scroll.set)
        self._jc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        jc_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        self._jc_tree.tag_configure("mismatch", background="#fecaca", foreground=_TEXT)
        self._jc_tree.tag_configure("match", background="#bbf7d0", foreground=_TEXT)
        self._jc_tree.tag_configure("extra", background="#fef3c7", foreground=_TEXT)

    def _parse_urdf_joints(self, urdf_path: str) -> list[str]:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = []
        for joint_elem in root.iter("joint"):
            jtype = joint_elem.get("type", "fixed")
            if jtype in ("revolute", "prismatic", "continuous"):
                name = joint_elem.get("name", "unnamed")
                joints.append(name)
        return joints

    def _parse_urdf_links(self, urdf_path: str) -> list[str]:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        return [link.get("name", "unnamed") for link in root.iter("link")]

    def _compare_joints(self) -> None:
        urdf_path = self._jc_urdf_var.get()
        mjcf_path = self._jc_mjcf_var.get()

        if not urdf_path or not os.path.isfile(urdf_path):
            messagebox.showerror("Error", "Please select a valid URDF file.")
            return
        if not mjcf_path or not os.path.isfile(mjcf_path):
            messagebox.showerror("Error", "Please select a valid MJCF file.")
            return

        try:
            urdf_joints = self._parse_urdf_joints(urdf_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to parse URDF: {exc}")
            return

        proto = self._proto_python_var.get()
        script = (
            "import json, sys; "
            "sys.path.insert(0, '.'); "
            "from protomotions.components.pose_lib import extract_kinematic_info; "
            f"ki = extract_kinematic_info('{mjcf_path}'); "
            "print(json.dumps({'body_names': ki.body_names, 'dof_names': ki.dof_names}))"
        )
        try:
            result = subprocess.run(
                [proto, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                self._log(f"[GUI] MJCF parse error:\n{result.stderr}\n")
                messagebox.showerror(
                    "Error", f"Failed to parse MJCF:\n{result.stderr[:500]}"
                )
                return
            mjcf_data = json.loads(result.stdout)
            mjcf_joints = mjcf_data["dof_names"]
            mjcf_bodies = mjcf_data["body_names"]
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to parse MJCF: {exc}")
            return

        for item in self._jc_tree.get_children():
            self._jc_tree.delete(item)

        max_len = max(len(urdf_joints), len(mjcf_joints))
        match_count = 0
        for i in range(max_len):
            urdf_j = urdf_joints[i] if i < len(urdf_joints) else "—"
            mjcf_j = mjcf_joints[i] if i < len(mjcf_joints) else "—"

            if urdf_j == "—" or mjcf_j == "—":
                tag = "extra"
                match_str = "EXTRA"
            elif urdf_j == mjcf_j:
                tag = "match"
                match_str = "YES"
                match_count += 1
            else:
                tag = "mismatch"
                match_str = "NO"

            self._jc_tree.insert(
                "", "end", values=(i + 1, urdf_j, mjcf_j, match_str), tags=(tag,)
            )

        if match_count == max_len and len(urdf_joints) == len(mjcf_joints):
            self._jc_status_var.set(f"MATCH — All {match_count} joints match")
            self._log(
                f"[GUI] Joint order comparison: MATCH ({match_count}/{match_count})\n"
            )
        else:
            self._jc_status_var.set(
                f"MISMATCH — {match_count}/{max_len} match "
                f"(URDF: {len(urdf_joints)}, MJCF: {len(mjcf_joints)})"
            )
            self._log(
                f"[GUI] Joint order comparison: MISMATCH "
                f"({match_count}/{max_len}, URDF={len(urdf_joints)}, MJCF={len(mjcf_joints)})\n"
            )

        urdf_links = self._parse_urdf_links(urdf_path)
        self._log(f"[GUI] URDF links ({len(urdf_links)}): {urdf_links}\n")
        self._log(f"[GUI] MJCF bodies ({len(mjcf_bodies)}): {mjcf_bodies}\n")

    def _get_run_buttons(self) -> list[ttk.Button]:
        return [self._batch_run_btn, self._single_run_btn, self._sv_run_btn]

    def _get_cancel_buttons(self) -> list[ttk.Button]:
        return [
            self._batch_cancel_btn,
            self._single_cancel_btn,
            self._sv_cancel_btn,
            self._usd_cancel_btn,
        ]


if __name__ == "__main__":
    main()
