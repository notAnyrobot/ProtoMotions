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
"""ProtoMotions Retargeting GUI — a graphical front-end for the entire
retargeting pipeline (AMASS → keypoints → PyRoki → ProtoMotions format).

Launch
------
    python tools/retargeting_gui.py

The application covers:
  1. Batch retarget (convenience script equivalent)
  2. Single-motion retarget
  3. Step-by-step pipeline execution with full parameter control
  4. Keypoint mapping & robot config tuning (launches the existing MuJoCo GUI)
  5. IsaacLab asset generation (MJCF → USD conversion)
  6. URDF vs MJCF joint-order comparison
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import tkinter as tk
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import yaml
from retargeting_gui_subprocess import MultiStepRunner, SubprocessRunner

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / ".retargeting_gui_config.json"
_DATA_DIR = Path(os.path.expanduser("~/Data"))

ROBOT_TYPES = ("g1", "h1_2", "astro")
SKELETON_FORMATS = ("smpl", "rigv1")
SIMULATORS = ("isaacgym", "isaaclab", "newton", "mujoco")
VISUALIZER_ROBOTS = ("g1", "h1_2", "astro", "rigv1", "smpl", "soma23")

# Default Python interpreter paths — users can override via GUI
_DEFAULT_PROTO_PYTHON = str(PROJECT_ROOT / "env_isaaclab" / "bin" / "python")
_DEFAULT_PYROKI_CANDIDATES = [
    os.path.expanduser("~/miniforge3/envs/pyroki-cuda/bin/python"),
    os.path.expanduser("~/miniforge3/envs/pyroki/bin/python"),
    os.path.expanduser("~/miniconda3/envs/pyroki-cuda/bin/python"),
    os.path.expanduser("~/miniconda3/envs/pyroki/bin/python"),
]

# ---------------------------------------------------------------------------
# UI Theme
# ---------------------------------------------------------------------------
_ACCENT = "#2563eb"           # Blue-600  — primary action color
_ACCENT_HOVER = "#1d4ed8"     # Blue-700
_BG = "#f1f5f9"               # Slate-50  — window / frame background
_SURFACE = "#ffffff"           # White     — input fields & cards
_BORDER = "#cbd5e1"            # Slate-300
_TEXT = "#1e293b"              # Slate-800
_MUTED = "#64748b"             # Slate-500
_SUCCESS = "#16a34a"           # Green-600
_ERROR = "#dc2626"             # Red-600
_LOG_BG = "#1e293b"            # Dark slate — log background
_LOG_FG = "#e2e8f0"            # Slate-200  — log text
_LOG_ACCENT = "#38bdf8"        # Sky-400    — log highlights
# Font families the user can cycle through
_FONT_FAMILIES = ("Helvetica", "DejaVu Sans", "Noto Sans", "Liberation Sans", "Ubuntu", "Segoe UI", "Arial")
_DEFAULT_FONT_FAMILY = "Helvetica"
_DEFAULT_FONT_SIZE = 11

_HEADER_FONT = (_DEFAULT_FONT_FAMILY, 12, "bold")
_MONO_FONT = ("Courier", 11)


def _detect_pyroki_python() -> str:
    for candidate in _DEFAULT_PYROKI_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return _DEFAULT_PYROKI_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
class ToolTip:
    """Simple hover tooltip for tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str):
        self._widget = widget
        self._text = text
        self._tw: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 2
        self._tw = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self._text,
            background=_LOG_BG,
            foreground=_LOG_FG,
            relief="flat",
            borderwidth=0,
            font=(_DEFAULT_FONT_FAMILY, _DEFAULT_FONT_SIZE - 1),
            wraplength=400,
            justify="left",
            padx=10,
            pady=6,
        )
        label.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self._tw:
            self._tw.destroy()
            self._tw = None


# ---------------------------------------------------------------------------
# Helpers to build common widgets
# ---------------------------------------------------------------------------

def _labeled_entry(
    parent: tk.Widget,
    label: str,
    default: str = "",
    tooltip: str = "",
    width: int = 60,
    row: int | None = None,
    browse: str | None = None,
    browse_dir: bool = False,
    filetypes: list[tuple[str, str]] | None = None,
) -> tuple[ttk.Entry, tk.StringVar]:
    """Create a label + entry (+ optional browse button) packed in a row frame."""
    frame = ttk.Frame(parent)
    if row is not None:
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
    else:
        frame.pack(fill=tk.X, padx=4, pady=2)

    lbl = ttk.Label(frame, text=label, width=24, anchor="w")
    lbl.pack(side=tk.LEFT, padx=(0, 4))
    if tooltip:
        ToolTip(lbl, tooltip)

    var = tk.StringVar(value=default)
    entry = ttk.Entry(frame, textvariable=var, width=width)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    if browse or browse_dir:

        def _browse() -> None:
            init = var.get() or str(PROJECT_ROOT)
            if browse_dir:
                path = filedialog.askdirectory(initialdir=init)
            elif browse == "savefile":
                path = filedialog.asksaveasfilename(
                    initialdir=os.path.dirname(init) if init != str(PROJECT_ROOT) else init,
                    filetypes=filetypes or [("All files", "*.*")],
                )
            else:
                path = filedialog.askopenfilename(
                    initialdir=os.path.dirname(init) if init != str(PROJECT_ROOT) else init,
                    filetypes=filetypes or [("All files", "*.*")],
                )
            if path:
                var.set(path)

        ttk.Button(frame, text="Browse", command=_browse, width=7).pack(side=tk.LEFT)

        if browse_dir:

            def _goto_data() -> None:
                if _DATA_DIR.is_dir():
                    var.set(str(_DATA_DIR))
                else:
                    messagebox.showinfo("Info", f"{_DATA_DIR} does not exist.")

            ttk.Button(frame, text="~/Data", command=_goto_data, width=7).pack(
                side=tk.LEFT, padx=(2, 0)
            )

            def _mkdir() -> None:
                cur = var.get().strip()
                if not cur:
                    messagebox.showinfo("Info", "Enter a directory path first.")
                    return
                try:
                    os.makedirs(cur, exist_ok=True)
                    messagebox.showinfo("Created", f"Directory created:\n{cur}")
                except Exception as exc:
                    messagebox.showerror("Error", f"Cannot create directory:\n{exc}")

            ttk.Button(frame, text="Create Dir", command=_mkdir, width=9).pack(
                side=tk.LEFT, padx=(2, 0)
            )

    return entry, var


def _labeled_combo(
    parent: tk.Widget,
    label: str,
    values: tuple[str, ...] | list[str],
    default: str = "",
    tooltip: str = "",
    width: int = 18,
) -> tuple[ttk.Combobox, tk.StringVar]:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=4, pady=2)
    lbl = ttk.Label(frame, text=label, width=24, anchor="w")
    lbl.pack(side=tk.LEFT, padx=(0, 4))
    if tooltip:
        ToolTip(lbl, tooltip)
    var = tk.StringVar(value=default or values[0])
    combo = ttk.Combobox(frame, textvariable=var, values=list(values), width=width, state="readonly")
    combo.pack(side=tk.LEFT)
    return combo, var


def _labeled_spin(
    parent: tk.Widget,
    label: str,
    from_: float,
    to: float,
    default: float = 1,
    tooltip: str = "",
    increment: float = 1,
    width: int = 10,
) -> tuple[ttk.Spinbox, tk.StringVar]:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=4, pady=2)
    lbl = ttk.Label(frame, text=label, width=24, anchor="w")
    lbl.pack(side=tk.LEFT, padx=(0, 4))
    if tooltip:
        ToolTip(lbl, tooltip)
    var = tk.StringVar(value=str(default))
    spin = ttk.Spinbox(frame, from_=from_, to=to, textvariable=var, increment=increment, width=width)
    spin.pack(side=tk.LEFT)
    return spin, var


def _labeled_check(
    parent: tk.Widget,
    label: str,
    default: bool = False,
    tooltip: str = "",
) -> tuple[ttk.Checkbutton, tk.BooleanVar]:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=4, pady=2)
    var = tk.BooleanVar(value=default)
    chk = ttk.Checkbutton(frame, text=label, variable=var)
    chk.pack(side=tk.LEFT, padx=(0, 4))
    if tooltip:
        ToolTip(chk, tooltip)
    return chk, var


def _readonly_entry(
    parent: tk.Widget, label: str, default: str = "", tooltip: str = ""
) -> tuple[ttk.Entry, tk.StringVar]:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=4, pady=2)
    lbl = ttk.Label(frame, text=label, width=24, anchor="w")
    lbl.pack(side=tk.LEFT, padx=(0, 4))
    if tooltip:
        ToolTip(lbl, tooltip)
    var = tk.StringVar(value=default)
    entry = ttk.Entry(frame, textvariable=var, width=60, state="readonly")
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return entry, var


def _section_header(parent: tk.Widget, text: str) -> ttk.Label:
    lbl = ttk.Label(parent, text=text, style="Header.TLabel")
    lbl.pack(fill=tk.X, padx=6, pady=(12, 2))
    sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
    sep.pack(fill=tk.X, padx=6, pady=(0, 6))
    return lbl


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════


class RetargetingGUI:
    """Main GUI application wrapping the ProtoMotions retargeting pipeline."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ProtoMotions — Retargeting Pipeline")
        self.root.geometry("1200x920")
        self.root.minsize(960, 720)

        # Load persisted preferences first (needed for font selection below)
        self._saved_config: dict[str, Any] = {}
        self._load_config()

        # Style — modern theme
        style = ttk.Style()
        style.theme_use("clam")
        self.root.configure(bg=_BG)
        self._style = style
        self._font_family = self._saved_config.get("font_family", _DEFAULT_FONT_FAMILY)
        self._font_size = _DEFAULT_FONT_SIZE
        self._apply_theme()

        # Build UI
        self._build_env_panel()
        self._build_notebook()
        self._build_log_panel()
        self._build_status_bar()

        # Subprocess runners
        self._runner = SubprocessRunner(self.root, self._log, self._on_cmd_done)
        self._multi = MultiStepRunner(
            self.root, self._log, self._on_step_change, self._on_multi_done
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_status()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _apply_theme(self, family: str | None = None) -> None:
        """(Re-)apply the full ttk theme.  Called on init and when the user
        picks a different font."""
        if family is not None:
            self._font_family = family
        fam = self._font_family
        sz = self._font_size
        style = self._style

        self.root.configure(bg=_BG)
        style.configure(".", background=_BG, foreground=_TEXT, font=(fam, sz))
        style.configure("TFrame", background=_BG)
        style.configure("TLabel", background=_BG, foreground=_TEXT, font=(fam, sz))
        style.configure("TLabelframe", background=_BG, foreground=_TEXT)
        style.configure(
            "TLabelframe.Label", background=_BG, foreground=_ACCENT,
            font=(fam, sz + 1, "bold"),
        )
        style.configure("TNotebook", background=_BG, tabmargins=[4, 4, 4, 0])
        style.configure(
            "TNotebook.Tab", padding=[16, 7],
            font=(fam, sz + 1, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", _SURFACE), ("!selected", _BG)],
            foreground=[("selected", _ACCENT), ("!selected", _MUTED)],
        )
        style.configure(
            "TButton", font=(fam, sz),
        )
        style.configure(
            "Accent.TButton", background=_ACCENT, foreground="#ffffff",
            font=(fam, sz + 1, "bold"), padding=[14, 6],
        )
        style.map("Accent.TButton", background=[("active", _ACCENT_HOVER)])
        style.configure("TEntry", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TSpinbox", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TCombobox", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TCheckbutton", font=(fam, sz))
        style.configure("Header.TLabel", font=(fam, sz + 1, "bold"), foreground=_ACCENT)
        style.configure(
            "Treeview", background=_SURFACE, fieldbackground=_SURFACE,
            foreground=_TEXT, rowheight=26, font=(fam, sz),
        )
        style.configure("Treeview.Heading", font=(fam, sz, "bold"))

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                self._saved_config = json.loads(CONFIG_FILE.read_text())
            except Exception:
                self._saved_config = {}

    def _save_config(self) -> None:
        cfg: dict[str, Any] = {
            "proto_python": self._proto_python_var.get(),
            "pyroki_python": self._pyroki_python_var.get(),
            "font_family": self._font_family,
        }
        try:
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Environment Panel (top)
    # ------------------------------------------------------------------

    def _build_env_panel(self) -> None:
        frame = ttk.LabelFrame(self.root, text="  Python Environments  ", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=(10, 4))

        # ProtoMotions Python
        proto_default = self._saved_config.get("proto_python", _DEFAULT_PROTO_PYTHON)
        _, self._proto_python_var = _labeled_entry(
            frame,
            "ProtoMotions Python:",
            proto_default,
            tooltip="Path to the Python interpreter with ProtoMotions installed (env_isaaclab).",
            browse="file",
            filetypes=[("Python", "python*"), ("All", "*")],
        )

        # PyRoki Python
        pyroki_default = self._saved_config.get("pyroki_python", _detect_pyroki_python())
        _, self._pyroki_python_var = _labeled_entry(
            frame,
            "PyRoki Python:",
            pyroki_default,
            tooltip="Path to the Python interpreter with PyRoki + JAX installed (pyroki-cuda env).",
            browse="file",
            filetypes=[("Python", "python*"), ("All", "*")],
        )

        # JAX acceleration mode
        _, self._pyroki_accel_var = _labeled_combo(
            frame, "PyRoki Acceleration:", ("cuda", "cpu"), "cuda",
            tooltip="JAX backend for PyRoki retargeting. 'cuda' uses GPU (default), 'cpu' for CPU-only.",
        )

        # Buttons row
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            btn_row, text="  Verify Environments  ", command=self._verify_envs,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._env_status_var = tk.StringVar(value="Not verified")
        ttk.Label(btn_row, textvariable=self._env_status_var, foreground=_MUTED).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_row, text="Exit", command=self._on_close).pack(
            side=tk.RIGHT, padx=(8, 0)
        )

        # Font selector
        font_row = ttk.Frame(frame)
        font_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(font_row, text="UI Font:", width=24, anchor="w").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._font_var = tk.StringVar(value=self._font_family)
        font_combo = ttk.Combobox(
            font_row, textvariable=self._font_var,
            values=list(_FONT_FAMILIES), width=22, state="readonly",
        )
        font_combo.pack(side=tk.LEFT)
        font_combo.bind("<<ComboboxSelected>>", self._on_font_change)
        ToolTip(font_combo, "Choose the UI font family. Saved across sessions.")

    def _verify_envs(self) -> None:
        self._log("\n[GUI] Verifying environments...\n")
        results: list[str] = []

        proto = self._proto_python_var.get()
        pyroki = self._pyroki_python_var.get()

        # ProtoMotions
        try:
            out = subprocess.run(
                [proto, "-c", "import protomotions; print('OK')"],
                capture_output=True, text=True, timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            if "OK" in out.stdout:
                results.append("Proto: OK")
                self._log("[GUI] ProtoMotions Python: OK\n")
            else:
                results.append("Proto: FAIL")
                self._log(f"[GUI] ProtoMotions Python: FAIL\n{out.stderr}\n")
        except Exception as exc:
            results.append("Proto: FAIL")
            self._log(f"[GUI] ProtoMotions Python: {exc}\n")

        # PyRoki
        try:
            out = subprocess.run(
                [pyroki, "-c", "import pyroki; print('OK')"],
                capture_output=True, text=True, timeout=30,
            )
            if "OK" in out.stdout:
                results.append("PyRoki: OK")
                self._log("[GUI] PyRoki Python: OK\n")
            else:
                results.append("PyRoki: FAIL")
                self._log(f"[GUI] PyRoki Python: FAIL\n{out.stderr}\n")
        except Exception as exc:
            results.append("PyRoki: FAIL")
            self._log(f"[GUI] PyRoki Python: {exc}\n")

        # CUDA
        try:
            out = subprocess.run(
                [pyroki, "-c", "import jax; print(jax.devices())"],
                capture_output=True, text=True, timeout=30,
            )
            self._log(f"[GUI] JAX devices: {out.stdout.strip()}\n")
            if "CudaDevice" in out.stdout or "cuda" in out.stdout.lower():
                results.append("CUDA: OK")
            else:
                results.append("CUDA: CPU-only")
        except Exception:
            results.append("CUDA: N/A")

        self._env_status_var.set(" | ".join(results))

    def _on_font_change(self, _event: tk.Event | None = None) -> None:
        family = self._font_var.get()
        self._apply_theme(family)
        self._log(f"[GUI] Font changed to: {family}\n")

    # ------------------------------------------------------------------
    # Notebook (tabs)
    # ------------------------------------------------------------------

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._build_tab_batch()
        self._build_tab_single()
        self._build_tab_steps()
        self._build_tab_visualize()
        self._build_tab_keypoint()
        self._build_tab_usd()
        self._build_tab_joints()

    # ------------------------------------------------------------------
    # Tab 1 — Batch Retarget
    # ------------------------------------------------------------------

    def _build_tab_batch(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Batch Retarget ")

        # Scrollable
        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        _section_header(inner, "Input")

        _, self._batch_pt_var = _labeled_entry(
            inner, "AMASS .pt File:", "",
            tooltip="Path to packaged AMASS MotionLib .pt file.",
            browse="file", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        _, self._batch_robot_var = _labeled_combo(
            inner, "Robot Type:", ROBOT_TYPES, "astro",
            tooltip="Target robot morphology.",
        )
        _, self._batch_skip_var = _labeled_spin(
            inner, "Skip Frequency:", 1, 1000, 1,
            tooltip="Skip every N motions (1 = process all). Useful for quick testing.",
        )
        _, self._batch_skel_var = _labeled_combo(
            inner, "Skeleton Format:", SKELETON_FORMATS, "smpl",
            tooltip="Source skeleton format of the AMASS data.",
        )

        _section_header(inner, "Output (auto-derived)")

        _, self._batch_outdir_var = _labeled_entry(
            inner, "Output Directory:", "",
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

        # Buttons
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        self._batch_run_btn = ttk.Button(
            btn_frame, text="  Run All 5 Steps  ", command=self._batch_run,
            style="Accent.TButton",
        )
        self._batch_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._batch_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._batch_cancel_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._batch_progress_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._batch_progress_var, foreground=_MUTED).pack(
            side=tk.LEFT
        )

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

    def _batch_run(self) -> None:
        pt_file = self._batch_pt_var.get()
        if not pt_file or not os.path.isfile(pt_file):
            messagebox.showerror("Error", "Please select a valid AMASS .pt file.")
            return

        proto = self._proto_python_var.get()
        pyroki = self._pyroki_python_var.get()
        if not os.path.isfile(proto):
            messagebox.showerror("Error", f"ProtoMotions Python not found: {proto}")
            return
        if not os.path.isfile(pyroki):
            messagebox.showerror("Error", f"PyRoki Python not found: {pyroki}")
            return
        robot = self._batch_robot_var.get()

        retarget_script = f"pyroki/batch_retarget_to_{robot}_from_keypoints.py"
        if not os.path.isfile(os.path.join(str(PROJECT_ROOT), retarget_script)):
            messagebox.showerror(
                "Error",
                f"Batch retarget script not found: {retarget_script}\n"
                f"Available robots: g1, h1_2, astro",
            )
            return

        skip = self._batch_skip_var.get()
        skel = self._batch_skel_var.get()
        out_dir = self._batch_outdir_var.get() or os.path.dirname(pt_file)
        kp_dir = os.path.join(out_dir, "keypoints-for-retarget")
        ret_dir = os.path.join(out_dir, f"pyroki-retargeted-{robot}")
        contacts_dir = os.path.join(out_dir, "contacts")
        proto_dir = os.path.join(out_dir, f"proto-{robot}")
        final_pt = os.path.join(out_dir, f"proto-{robot}.pt")
        cwd = str(PROJECT_ROOT)

        pyroki_env = self._pyroki_env()

        self._log(f"\n[GUI] Batch run: proto={proto}, pyroki={pyroki}, robot={robot}\n")
        self._log(f"[GUI] Retarget script: {retarget_script}\n")

        steps = [
            (
                "Extracting keypoints from SMPL motions",
                [
                    proto,
                    "data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py",
                    pt_file,
                    "--output-path", kp_dir,
                    "--skeleton-format", skel,
                    "--start-idx", "0",
                    "--skip-freq", str(skip),
                ],
                cwd,
            ),
            (
                f"Running PyRoki retargeting to {robot.upper()}",
                [
                    pyroki, retarget_script,
                    "--subsample-factor", "1",
                    "--keypoints-folder-path", kp_dir,
                    "--source-type", skel,
                    "--output-dir", ret_dir,
                    "--no-visualize",
                    "--skip-existing",
                ],
                cwd,
                pyroki_env,
            ),
            (
                "Extracting foot contact labels",
                [
                    pyroki, retarget_script,
                    "--subsample-factor", "1",
                    "--keypoints-folder-path", kp_dir,
                    "--source-type", skel,
                    "--save-contacts-only",
                    "--contacts-dir", contacts_dir,
                    "--skip-existing",
                ],
                cwd,
                pyroki_env,
            ),
            (
                "Converting to ProtoMotions format",
                [
                    proto,
                    "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
                    "--retargeted-motion-dir", ret_dir,
                    "--output-dir", proto_dir,
                    "--robot-type", robot,
                    "--contact-labels-dir", contacts_dir,
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
                    "--motion-path", proto_dir,
                    "--output-file", final_pt,
                ],
                cwd,
            ),
        ]

        self._set_running(True)
        self._multi.start(steps)

    # ------------------------------------------------------------------
    # Tab 2 — Single Motion Retarget
    # ------------------------------------------------------------------

    def _build_tab_single(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Single Motion ")

        _section_header(tab, "Input")

        _, self._single_motion_var = _labeled_entry(
            tab, "Motion File (.motion):", "",
            tooltip="Path to a single .motion file in SMPL format.",
            browse="file", filetypes=[("Motion", "*.motion"), ("All", "*.*")],
        )
        _, self._single_robot_var = _labeled_combo(
            tab, "Robot Type:", ROBOT_TYPES, "astro",
            tooltip="Target robot morphology.",
        )
        _, self._single_outdir_var = _labeled_entry(
            tab, "Output Directory:", "",
            tooltip="Directory where all intermediate and final outputs will be saved.",
            browse_dir=True,
        )

        _section_header(tab, "Output (auto-derived)")

        _, self._single_kp_var = _readonly_entry(tab, "Keypoints Dir:")
        _, self._single_ret_var = _readonly_entry(tab, "Retargeted Dir:")
        _, self._single_contacts_var = _readonly_entry(tab, "Contacts Dir:")
        _, self._single_proto_var = _readonly_entry(tab, "Proto Dir:")

        self._single_outdir_var.trace_add("write", self._single_update_derived)
        self._single_robot_var.trace_add("write", self._single_update_derived)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        self._single_run_btn = ttk.Button(
            btn_frame, text="  Run All Steps  ", command=self._single_run,
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
        pyroki = self._pyroki_python_var.get()
        robot = self._single_robot_var.get()
        cwd = str(PROJECT_ROOT)

        kp_dir = os.path.join(out_dir, "keypoints")
        ret_dir = os.path.join(out_dir, f"retargeted_{robot}")
        contacts_dir = os.path.join(out_dir, "contacts")
        proto_dir = os.path.join(out_dir, f"retargeted_{robot}_proto")
        retarget_script = f"pyroki/batch_retarget_to_{robot}_from_keypoints.py"
        pyroki_env = self._pyroki_env()

        steps = [
            (
                "Extracting keypoints from single motion",
                [
                    proto,
                    "data/scripts/extract_keypoints_from_single_motion.py",
                    motion,
                    "--output-path", kp_dir,
                    "--skeleton-format", "smpl",
                    "--force-remake",
                ],
                cwd,
            ),
            (
                f"Running PyRoki retargeting to {robot.upper()}",
                [
                    pyroki, retarget_script,
                    "--subsample-factor", "1",
                    "--keypoints-folder-path", kp_dir,
                    "--source-type", "smpl",
                    "--output-dir", ret_dir,
                    "--no-visualize",
                ],
                cwd,
                pyroki_env,
            ),
            (
                "Extracting foot contact labels",
                [
                    pyroki, retarget_script,
                    "--subsample-factor", "1",
                    "--keypoints-folder-path", kp_dir,
                    "--source-type", "smpl",
                    "--save-contacts-only",
                    "--contacts-dir", contacts_dir,
                ],
                cwd,
                pyroki_env,
            ),
            (
                "Converting to ProtoMotions format",
                [
                    proto,
                    "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
                    "--retargeted-motion-dir", ret_dir,
                    "--output-dir", proto_dir,
                    "--robot-type", robot,
                    "--contact-labels-dir", contacts_dir,
                    "--apply-motion-filter",
                    "--force-remake",
                ],
                cwd,
            ),
        ]

        self._set_running(True)
        self._multi.start(steps)

    # ------------------------------------------------------------------
    # Tab 3 — Step-by-Step
    # ------------------------------------------------------------------

    def _build_tab_steps(self) -> None:
        tab = ttk.Frame(self._notebook, padding=0)
        self._notebook.add(tab, text=" Step-by-Step ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ---- Step 1: Extract Keypoints ----
        s1 = ttk.LabelFrame(inner, text="Step 1 — Extract Keypoints (ProtoMotions Python)", padding=8)
        s1.pack(fill=tk.X, padx=8, pady=4)

        _, self._s1_mode_var = _labeled_combo(
            s1, "Source Mode:", ("From .pt (batch)", "From .motion (single)"),
            tooltip="Extract from a packaged .pt or a single .motion file.",
        )
        _, self._s1_input_var = _labeled_entry(
            s1, "Input File:", "",
            tooltip=".pt file (batch) or .motion file (single).",
            browse="file", filetypes=[("Motion data", "*.pt *.motion"), ("All", "*.*")],
        )
        _, self._s1_skel_var = _labeled_combo(
            s1, "Skeleton Format:", SKELETON_FORMATS, "smpl",
            tooltip="smpl or rigv1 skeleton format.",
        )
        _, self._s1_output_var = _labeled_entry(
            s1, "Output Path:", "",
            tooltip="Directory to save extracted keypoint .npy files.",
            browse_dir=True,
        )
        _, self._s1_force_var = _labeled_check(
            s1, "Force Remake",
            tooltip="Overwrite existing keypoint files.",
        )
        _, self._s1_start_var = _labeled_spin(
            s1, "Start Index:", 0, 100000, 0,
            tooltip="Start from this motion index (batch mode only).",
        )
        _, self._s1_end_var = _labeled_spin(
            s1, "End Index:", 1, 100000, 3500,
            tooltip="End at this motion index (batch mode only).",
        )
        _, self._s1_skip_var = _labeled_spin(
            s1, "Skip Frequency:", 1, 1000, 35,
            tooltip="Skip every N motions (batch mode only).",
        )
        ttk.Button(s1, text="Run Step 1", command=self._run_step1).pack(anchor="w", padx=4, pady=4)

        # ---- Step 2: PyRoki Retarget ----
        s2 = ttk.LabelFrame(inner, text="Step 2 — PyRoki Retarget (PyRoki Python)", padding=8)
        s2.pack(fill=tk.X, padx=8, pady=4)

        _, self._s2_config_var = _labeled_entry(
            s2, "Robot Config (.yaml):",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Path to the YAML robot retarget config.",
            browse="file", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        _, self._s2_kp_var = _labeled_entry(
            s2, "Keypoints Folder:", "",
            tooltip="Path to folder containing the extracted keypoint .npy files.",
            browse_dir=True,
        )
        _, self._s2_outdir_var = _labeled_entry(
            s2, "Output Directory:", "",
            tooltip="Directory to save retargeted motion files.",
            browse_dir=True,
        )
        _, self._s2_source_var = _labeled_combo(
            s2, "Source Type:", SKELETON_FORMATS, "smpl",
            tooltip="Source skeleton type.",
        )
        _, self._s2_subsample_var = _labeled_spin(
            s2, "Subsample Factor:", 1, 100, 1,
            tooltip="Subsample factor for keypoints.",
        )
        _, self._s2_frames_var = _labeled_spin(
            s2, "Target Raw Frames:", 1, 10000, 450,
            tooltip="Target raw frames before subsampling (450 = 15s at 30fps).",
        )
        _, self._s2_fps_var = _labeled_spin(
            s2, "Input FPS:", 1, 120, 30,
            tooltip="FPS of the input keypoint data.",
        )
        _, self._s2_skip_var = _labeled_check(
            s2, "Skip Existing",
            tooltip="Skip motions that already have output files.",
        )
        _, self._s2_novis_var = _labeled_check(
            s2, "No Visualize", default=True,
            tooltip="Run without MuJoCo visualization (batch mode).",
        )

        # Retargeting weights
        wf = ttk.LabelFrame(s2, text="Retargeting Weights (override config values)", padding=4)
        wf.pack(fill=tk.X, padx=4, pady=4)

        _, self._s2_w_local_var = _labeled_spin(
            wf, "Local Alignment:", 0, 100, 1.0, increment=0.5,
            tooltip="Weight for local link position alignment.",
        )
        _, self._s2_w_global_var = _labeled_spin(
            wf, "Global Alignment:", 0, 100, 4.0, increment=0.5,
            tooltip="Weight for global (root-relative) alignment.",
        )
        _, self._s2_w_root_smooth_var = _labeled_spin(
            wf, "Root Smoothness:", 0, 100, 1.0, increment=0.5,
            tooltip="Weight for root rotation smoothness.",
        )
        _, self._s2_w_joint_smooth_var = _labeled_spin(
            wf, "Joint Smoothness:", 0, 100, 4.0, increment=0.5,
            tooltip="Weight for joint angle smoothness.",
        )
        _, self._s2_w_self_coll_var = _labeled_spin(
            wf, "Self Collision:", 0, 100, 0.0, increment=0.1,
            tooltip="Weight for self-collision avoidance.",
        )
        _, self._s2_w_rest_var = _labeled_spin(
            wf, "Joint Rest Penalty:", 0, 100, 1.0, increment=0.5,
            tooltip="Penalty for deviating from rest pose.",
        )
        _, self._s2_w_vel_limit_var = _labeled_spin(
            wf, "Joint Vel Limit:", 0, 200, 50.0, increment=5.0,
            tooltip="Weight for joint velocity limit enforcement.",
        )
        _, self._s2_w_foot_contact_var = _labeled_spin(
            wf, "Foot Contact:", 0, 200, 30.0, increment=5.0,
            tooltip="Weight for foot contact constraint.",
        )
        _, self._s2_w_foot_tilt_var = _labeled_spin(
            wf, "Foot Tilt:", 0, 100, 1.0, increment=0.1,
            tooltip="Weight for foot tilt constraint.",
        )
        _, self._s2_w_override_var = _labeled_check(
            wf, "Override config weights with values above", default=True,
            tooltip="When checked, the above weights replace those in the YAML config.",
        )
        ttk.Button(wf, text="Load from Config", command=self._s2_load_weights).pack(
            anchor="w", padx=4, pady=2
        )

        ttk.Button(s2, text="Run Step 2", command=self._run_step2).pack(anchor="w", padx=4, pady=4)

        # ---- Step 3: Extract Contacts ----
        s3 = ttk.LabelFrame(inner, text="Step 3 — Extract Contact Labels (PyRoki Python)", padding=8)
        s3.pack(fill=tk.X, padx=8, pady=4)

        _, self._s3_config_var = _labeled_entry(
            s3, "Robot Config (.yaml):",
            str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Path to robot retarget config (same as Step 2).",
            browse="file", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        _, self._s3_kp_var = _labeled_entry(
            s3, "Keypoints Folder:", "",
            tooltip="Path to folder with extracted keypoints.",
            browse_dir=True,
        )
        _, self._s3_source_var = _labeled_combo(
            s3, "Source Type:", SKELETON_FORMATS, "smpl",
        )
        _, self._s3_contacts_var = _labeled_entry(
            s3, "Contacts Output Dir:", "",
            tooltip="Directory to save contact label files.",
            browse_dir=True,
        )
        _, self._s3_skip_var = _labeled_check(
            s3, "Skip Existing",
        )
        ttk.Button(s3, text="Run Step 3", command=self._run_step3).pack(anchor="w", padx=4, pady=4)

        # ---- Step 4: Convert to Proto ----
        s4 = ttk.LabelFrame(inner, text="Step 4 — Convert to ProtoMotions Format (ProtoMotions Python)", padding=8)
        s4.pack(fill=tk.X, padx=8, pady=4)

        _, self._s4_retdir_var = _labeled_entry(
            s4, "Retargeted Motion Dir:", "",
            tooltip="Directory with retargeted motion files (.npz or .csv).",
            browse_dir=True,
        )
        _, self._s4_outdir_var = _labeled_entry(
            s4, "Output Dir:", "",
            tooltip="Directory to save ProtoMotions .motion files.",
            browse_dir=True,
        )
        _, self._s4_robot_var = _labeled_combo(
            s4, "Robot Type:", ROBOT_TYPES, "astro",
        )
        _, self._s4_contacts_var = _labeled_entry(
            s4, "Contact Labels Dir:", "",
            tooltip="Directory with contact label files (optional).",
            browse_dir=True,
        )
        _, self._s4_filter_var = _labeled_check(
            s4, "Apply Motion Filter", default=True,
            tooltip="Apply motion quality filter (height, velocity, DOF limits).",
        )
        _, self._s4_force_var = _labeled_check(s4, "Force Remake")
        _, self._s4_ifps_var = _labeled_spin(s4, "Input FPS:", 1, 120, 30)
        _, self._s4_ofps_var = _labeled_spin(s4, "Output FPS:", 1, 120, 30)
        _, self._s4_ignore_var = _labeled_spin(
            s4, "Ignore First N Frames:", 0, 1000, 0,
            tooltip="Skip the first N frames of each motion.",
        )

        # Motion filter sub-options
        filter_frame = ttk.LabelFrame(s4, text="Motion Filter Thresholds", padding=4)
        filter_frame.pack(fill=tk.X, padx=4, pady=4)
        _, self._s4_min_h_var = _labeled_spin(
            filter_frame, "Min Height:", -10, 10, -0.05, increment=0.01,
            tooltip="Minimum height threshold for motion filter.",
        )
        _, self._s4_max_v_var = _labeled_spin(
            filter_frame, "Max Velocity:", 0, 100, 15.0, increment=1.0,
            tooltip="Maximum velocity threshold.",
        )
        _, self._s4_max_dof_var = _labeled_spin(
            filter_frame, "Max DOF Velocity:", 0, 200, 40.0, increment=1.0,
            tooltip="Maximum DOF velocity threshold.",
        )
        _, self._s4_dur_h_var = _labeled_spin(
            filter_frame, "Duration Height Filter:", 0, 5, 0.1, increment=0.01,
            tooltip="Height threshold for duration filter.",
        )
        _, self._s4_dur_s_var = _labeled_spin(
            filter_frame, "Duration Height Secs:", 0, 10, 0.6, increment=0.1,
            tooltip="Duration in seconds for height filter.",
        )
        ttk.Button(s4, text="Run Step 4", command=self._run_step4).pack(anchor="w", padx=4, pady=4)

        # ---- Step 5: Package MotionLib ----
        s5 = ttk.LabelFrame(inner, text="Step 5 — Package MotionLib (ProtoMotions Python)", padding=8)
        s5.pack(fill=tk.X, padx=8, pady=4)

        _, self._s5_motdir_var = _labeled_entry(
            s5, "Motion Path:", "",
            tooltip="Path to directory of .motion files (or a YAML listing motions).",
            browse_dir=True,
        )
        _, self._s5_output_var = _labeled_entry(
            s5, "Output .pt Filename:", "",
            tooltip="Output .pt file path. Type a name or use Browse to pick a save location.",
            browse="savefile", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        ttk.Button(s5, text="Run Step 5", command=self._run_step5).pack(anchor="w", padx=4, pady=4)

    # Step runners

    def _run_step1(self) -> None:
        proto = self._proto_python_var.get()
        mode = self._s1_mode_var.get()
        inp = self._s1_input_var.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        cmd: list[str]
        if "batch" in mode.lower() or ".pt" in mode.lower():
            cmd = [
                proto,
                "data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py",
                inp,
                "--skeleton-format", self._s1_skel_var.get(),
                "--start-idx", self._s1_start_var.get(),
                "--end-idx", self._s1_end_var.get(),
                "--skip-freq", self._s1_skip_var.get(),
            ]
            if self._s1_output_var.get():
                cmd += ["--output-path", self._s1_output_var.get()]
            if self._s1_force_var.get():
                cmd += ["--force-remake"]
        else:
            cmd = [
                proto,
                "data/scripts/extract_keypoints_from_single_motion.py",
                inp,
                "--skeleton-format", self._s1_skel_var.get(),
            ]
            if self._s1_output_var.get():
                cmd += ["--output-path", self._s1_output_var.get()]
            if self._s1_force_var.get():
                cmd += ["--force-remake"]

        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

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
            fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="retarget_cfg_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            self._log(f"[GUI] Created temp config with custom weights: {tmp_path}\n")
            return tmp_path
        except Exception as exc:
            self._log(f"[GUI] Warning: failed to create temp config: {exc}\n")
            return None

    def _run_step2(self) -> None:
        pyroki = self._pyroki_python_var.get()
        if not os.path.isfile(pyroki):
            messagebox.showerror("Error", f"PyRoki Python not found: {pyroki}")
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
        cmd = [
            pyroki,
            "pyroki/retarget_from_keypoints.py",
            "--robot-config", config,
            "--keypoints-folder-path", kp_path,
            "--output-dir", out_dir,
            "--source-type", self._s2_source_var.get(),
            "--subsample-factor", self._s2_subsample_var.get(),
            "--target-raw-frames", self._s2_frames_var.get(),
            "--input-fps", self._s2_fps_var.get(),
        ]
        if self._s2_skip_var.get():
            cmd.append("--skip-existing")
        if self._s2_novis_var.get():
            cmd.append("--no-visualize")

        self._log(f"\n[GUI] Step 2: Retarget using PyRoki Python: {pyroki}\n")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT), env=self._pyroki_env())

    def _run_step3(self) -> None:
        pyroki = self._pyroki_python_var.get()
        if not os.path.isfile(pyroki):
            messagebox.showerror("Error", f"PyRoki Python not found: {pyroki}")
            return
        config = self._s3_config_var.get()
        if not config or not os.path.isfile(config):
            messagebox.showerror("Error", f"Robot config not found: {config}")
            return
        kp_path = self._s3_kp_var.get()
        if not kp_path or not os.path.isdir(kp_path):
            messagebox.showerror("Error", f"Keypoints folder not found: {kp_path}")
            return
        cmd = [
            pyroki,
            "pyroki/retarget_from_keypoints.py",
            "--robot-config", config,
            "--keypoints-folder-path", kp_path,
            "--source-type", self._s3_source_var.get(),
            "--save-contacts-only",
        ]
        if self._s3_contacts_var.get():
            cmd += ["--contacts-dir", self._s3_contacts_var.get()]
        if self._s3_skip_var.get():
            cmd.append("--skip-existing")

        self._log(f"\n[GUI] Step 3: Extract contacts using PyRoki Python: {pyroki}\n")
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT), env=self._pyroki_env())

    def _run_step4(self) -> None:
        proto = self._proto_python_var.get()
        cmd = [
            proto,
            "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py",
            "--retargeted-motion-dir", self._s4_retdir_var.get(),
            "--output-dir", self._s4_outdir_var.get(),
            "--robot-type", self._s4_robot_var.get(),
            "--input-fps", self._s4_ifps_var.get(),
            "--output-fps", self._s4_ofps_var.get(),
            "--ignore-first-n-frames", self._s4_ignore_var.get(),
            "--min-height-threshold", self._s4_min_h_var.get(),
            "--max-velocity-threshold", self._s4_max_v_var.get(),
            "--max-dof-vel-threshold", self._s4_max_dof_var.get(),
            "--duration-height-filter", self._s4_dur_h_var.get(),
            "--duration-height-seconds", self._s4_dur_s_var.get(),
        ]
        if self._s4_contacts_var.get():
            cmd += ["--contact-labels-dir", self._s4_contacts_var.get()]
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
        # Ensure .pt extension
        if not out_file.endswith(".pt"):
            out_file += ".pt"
            self._s5_output_var.set(out_file)
        # If user typed just a name (no directory), place it next to the motion dir
        if os.sep not in out_file and "/" not in out_file:
            parent = os.path.dirname(mot_path) if os.path.isfile(mot_path) else mot_path
            out_file = os.path.join(os.path.dirname(parent) if os.path.isfile(mot_path) else parent, out_file)
            self._s5_output_var.set(out_file)
            self._log(f"[GUI] Output file resolved to: {out_file}\n")
        cmd = [
            proto,
            "protomotions/components/motion_lib.py",
            "--motion-path", mot_path,
            "--output-file", out_file,
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
            proto, "examples/motion_libs_visualizer.py",
            "--robot", self._sv_robot_var.get(),
            "--simulator", self._sv_sim_var.get(),
            "--playback_speed", self._sv_speed_var.get(),
        ]
        cmd += ["--motion_files"] + files
        if self._sv_headless_var.get():
            cmd.append("--headless")
        if self._sv_cpu_var.get():
            cmd.append("--cpu-only")

        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

    # ------------------------------------------------------------------
    # Tab 4 — Visualize Motions
    # ------------------------------------------------------------------

    def _build_tab_visualize(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text=" Visualize ")

        _section_header(tab, "Motion Visualizer")
        ttk.Label(tab, text=(
            "Preview retargeted motions in the simulator.  Supports .pt (packaged "
            "MotionLib) and individual .motion files.  Multiple files can be "
            "space-separated."
        ), wraplength=850, foreground=_MUTED).pack(fill=tk.X, padx=6, pady=(0, 10))

        _, self._sv_files_var = _labeled_entry(
            tab, "Motion Files:", "",
            tooltip="Path(s) to .pt or .motion files, space-separated.",
            browse="file", filetypes=[("Motion", "*.pt *.motion"), ("All", "*.*")],
        )
        _, self._sv_robot_var = _labeled_combo(
            tab, "Robot:", VISUALIZER_ROBOTS, "astro",
            tooltip="Robot to visualize motions on.",
        )
        _, self._sv_sim_var = _labeled_combo(
            tab, "Simulator:", SIMULATORS, "isaaclab",
            tooltip="Physics simulator backend for playback.",
        )
        _, self._sv_speed_var = _labeled_spin(
            tab, "Playback Speed:", 0.1, 10, 1.0, increment=0.1,
            tooltip="Playback speed multiplier (1.0 = real-time).",
        )
        _, self._sv_headless_var = _labeled_check(
            tab, "Headless",
            tooltip="Run without GUI window (useful for recording).",
        )
        _, self._sv_cpu_var = _labeled_check(
            tab, "CPU Only",
            tooltip="Force CPU-only mode (no GPU required).",
        )

        # Action buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=6, pady=(18, 4))
        self._sv_run_btn = ttk.Button(
            btn_frame, text="  Launch Visualizer  ", command=self._run_visualizer,
            style="Accent.TButton",
        )
        self._sv_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._sv_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled",
        )
        self._sv_cancel_btn.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Tab 5 — Keypoint Mapping & Config
    # ------------------------------------------------------------------

    def _build_tab_keypoint(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Keypoint Config ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Launch existing GUI ---
        launch_frame = ttk.LabelFrame(inner, text="Launch Keypoint Mapping Tuner (MuJoCo Viewer)", padding=8)
        launch_frame.pack(fill=tk.X, padx=4, pady=4)

        _, self._kp_config_var = _labeled_entry(
            launch_frame, "Robot Config:", str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Robot config YAML for the keypoint mapping tuner.",
            browse="file", filetypes=[("YAML", "*.yaml *.yml")],
        )
        _, self._kp_source_var = _labeled_combo(
            launch_frame, "Source Type:", ("smpl", "rigv1"), "smpl",
        )
        _, self._kp_spacing_var = _labeled_spin(
            launch_frame, "Viewer Spacing:", 0.1, 5.0, 1.0, increment=0.1,
        )
        ttk.Button(
            launch_frame, text="  Launch Keypoint Tuner GUI  ",
            command=self._launch_keypoint_gui,
            style="Accent.TButton",
        ).pack(anchor="w", padx=4, pady=6)

        # --- Robot Config Editor ---
        editor_frame = ttk.LabelFrame(inner, text="Robot Config Editor", padding=8)
        editor_frame.pack(fill=tk.X, padx=4, pady=4)

        _, self._cfg_path_var = _labeled_entry(
            editor_frame, "Config File:", str(PROJECT_ROOT / "pyroki" / "robot_configs" / "astro.yaml"),
            tooltip="Robot config YAML to load and edit.",
            browse="file", filetypes=[("YAML", "*.yaml *.yml")],
        )

        btn_row = ttk.Frame(editor_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="Load Config", command=self._load_robot_config).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Save Config", command=self._save_robot_config).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="New from Template", command=self._new_robot_config).pack(side=tk.LEFT)

        # Editable text area for the YAML
        self._cfg_text = tk.Text(
            editor_frame, height=30, width=90,
            font=_MONO_FONT,
            bg=_LOG_BG, fg=_LOG_FG,
            insertbackground=_LOG_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=10, pady=8,
        )
        cfg_scroll = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self._cfg_text.yview)
        self._cfg_text.configure(yscrollcommand=cfg_scroll.set)
        self._cfg_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        cfg_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

    def _launch_keypoint_gui(self) -> None:
        # The keypoint tuner needs mujoco (available in ProtoMotions/env_isaaclab)
        # plus pyroki helper functions (found via PYTHONPATH to pyroki/).
        proto = self._proto_python_var.get()
        cmd = [
            proto,
            str(SCRIPT_DIR / "visualize_keypoint_mapping_gui.py"),
            "--robot-config", self._kp_config_var.get(),
            "--source-type", self._kp_source_var.get(),
            "--spacing", self._kp_spacing_var.get(),
        ]
        self._log("\n[GUI] Launching keypoint tuner as separate process...\n")
        self._log(f"[GUI] Using ProtoMotions Python (has mujoco): {proto}\n")
        self._log(f"[GUI] $ {' '.join(cmd)}\n")
        try:
            merged_env = dict(os.environ)
            # Ensure pyroki/ is on PYTHONPATH so the tuner can import helpers
            pyroki_dir = str(PROJECT_ROOT / "pyroki")
            existing = merged_env.get("PYTHONPATH", "")
            merged_env["PYTHONPATH"] = f"{pyroki_dir}:{existing}" if existing else pyroki_dir
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
  pelvis: "pelvis_link"
  left_hip: "left_hip_link"
  right_hip: "right_hip_link"
  left_knee: "left_knee_link"
  right_knee: "right_knee_link"
  left_ankle: "left_ankle_link"
  right_ankle: "right_ankle_link"
  left_foot: "left_foot_link"
  right_foot: "right_foot_link"
  left_shoulder: "left_shoulder_link"
  right_shoulder: "right_shoulder_link"
  left_elbow: "left_elbow_link"
  right_elbow: "right_elbow_link"
  left_wrist: "left_wrist_link"
  right_wrist: "right_wrist_link"

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

    # ------------------------------------------------------------------
    # Tab 6 — USD Conversion
    # ------------------------------------------------------------------

    def _build_tab_usd(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" USD Conversion ")

        _section_header(tab, "IsaacLab Asset Generation (MJCF → USD)")
        ttk.Label(tab, text=(
            "Convert a MuJoCo MJCF file to USDA format for IsaacLab.\n"
            "Step 1 flattens the MJCF (resolves defaults). "
            "Step 2 converts to USD (requires IsaacLab environment)."
        ), wraplength=800, foreground=_MUTED).pack(fill=tk.X, padx=4, pady=(0, 8))

        _, self._usd_input_var = _labeled_entry(
            tab, "Input MJCF (.xml):", "",
            tooltip="Path to the MJCF XML file to convert.",
            browse="file", filetypes=[("XML", "*.xml"), ("All", "*.*")],
        )
        _, self._usd_outdir_var = _labeled_entry(
            tab, "Output Directory:", "",
            tooltip="Output directory for USD files (optional, defaults to same dir).",
            browse_dir=True,
        )
        _, self._usd_noverify_var = _labeled_check(
            tab, "Skip MuJoCo Verification (flatten)",
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
            btn_frame, text="  Run Both  ", command=self._usd_both,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._usd_cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, state="disabled"
        )
        self._usd_cancel_btn.pack(side=tk.LEFT)

        # Flattened file path (auto-derived)
        _, self._usd_flat_var = _readonly_entry(
            tab, "Flattened MJCF:", "",
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
            messagebox.showerror("Error", "Flattened MJCF not found. Run 'Flatten' first.")
            return
        cmd = [proto, "usd_convert/convert_robot_mjcf_to_usda.py", flat]
        if self._usd_outdir_var.get():
            cmd += ["--output-dir", self._usd_outdir_var.get()]
        self._set_running(True)
        self._runner.start(cmd, cwd=str(PROJECT_ROOT))

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

        steps = [
            ("Flattening MJCF", flatten_cmd, str(PROJECT_ROOT)),
            ("Converting to USD", convert_cmd, str(PROJECT_ROOT)),
        ]
        self._set_running(True)
        self._multi.start(steps)

    # ------------------------------------------------------------------
    # Tab 7 — Joint Order Comparison
    # ------------------------------------------------------------------

    def _build_tab_joints(self) -> None:
        tab = ttk.Frame(self._notebook, padding=8)
        self._notebook.add(tab, text=" Joint Order ")

        _section_header(tab, "URDF vs MJCF Joint/Body Order Comparison")
        ttk.Label(tab, text=(
            "Compare the joint and body ordering between a URDF (used in PyRoki retargeting) "
            "and an MJCF (used in simulation/training). Mismatches can cause silent data "
            "corruption in motion data."
        ), wraplength=800, foreground=_MUTED).pack(fill=tk.X, padx=4, pady=(0, 8))

        _, self._jc_urdf_var = _labeled_entry(
            tab, "URDF File:", "",
            tooltip="URDF file used for PyRoki retargeting.",
            browse="file", filetypes=[("URDF", "*.urdf"), ("All", "*.*")],
        )
        _, self._jc_mjcf_var = _labeled_entry(
            tab, "MJCF File:", str(PROJECT_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "g1_bm_box_feet.xml"),
            tooltip="MJCF file used for simulation training.",
            browse="file", filetypes=[("XML", "*.xml"), ("All", "*.*")],
        )

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(
            btn_row, text="  Compare Joint Order  ", command=self._compare_joints,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._jc_status_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self._jc_status_var, font=("Helvetica", 10, "bold")).pack(
            side=tk.LEFT
        )

        # Treeview for results
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

        # Tag for mismatch highlighting
        self._jc_tree.tag_configure("mismatch", background="#fecaca", foreground=_TEXT)
        self._jc_tree.tag_configure("match", background="#bbf7d0", foreground=_TEXT)
        self._jc_tree.tag_configure("extra", background="#fef3c7", foreground=_TEXT)

    def _parse_urdf_joints(self, urdf_path: str) -> list[str]:
        """Parse joint names from a URDF file (standard XML)."""
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
        """Parse link names from a URDF file."""
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

        # Parse URDF joints
        try:
            urdf_joints = self._parse_urdf_joints(urdf_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to parse URDF: {exc}")
            return

        # Parse MJCF via subprocess (needs ProtoMotions imports)
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
                capture_output=True, text=True, timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                self._log(f"[GUI] MJCF parse error:\n{result.stderr}\n")
                messagebox.showerror("Error", f"Failed to parse MJCF:\n{result.stderr[:500]}")
                return
            mjcf_data = json.loads(result.stdout)
            mjcf_joints = mjcf_data["dof_names"]
            mjcf_bodies = mjcf_data["body_names"]
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to parse MJCF: {exc}")
            return

        # Clear treeview
        for item in self._jc_tree.get_children():
            self._jc_tree.delete(item)

        # Compare joints
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

            self._jc_tree.insert("", "end", values=(i + 1, urdf_j, mjcf_j, match_str), tags=(tag,))

        # Summary
        if match_count == max_len and len(urdf_joints) == len(mjcf_joints):
            self._jc_status_var.set(f"MATCH — All {match_count} joints match")
            self._log(f"[GUI] Joint order comparison: MATCH ({match_count}/{match_count})\n")
        else:
            self._jc_status_var.set(
                f"MISMATCH — {match_count}/{max_len} match "
                f"(URDF: {len(urdf_joints)}, MJCF: {len(mjcf_joints)})"
            )
            self._log(
                f"[GUI] Joint order comparison: MISMATCH "
                f"({match_count}/{max_len}, URDF={len(urdf_joints)}, MJCF={len(mjcf_joints)})\n"
            )

        # Also log body/link comparison
        urdf_links = self._parse_urdf_links(urdf_path)
        self._log(f"[GUI] URDF links ({len(urdf_links)}): {urdf_links}\n")
        self._log(f"[GUI] MJCF bodies ({len(mjcf_bodies)}): {mjcf_bodies}\n")

    # ------------------------------------------------------------------
    # Log Panel (bottom)
    # ------------------------------------------------------------------

    def _build_log_panel(self) -> None:
        log_frame = ttk.LabelFrame(self.root, text="  Output Log  ", padding=6)
        log_frame.pack(fill=tk.X, padx=10, pady=(4, 10))

        btn_row = ttk.Frame(log_frame)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self._clear_log).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Save Log", command=self._save_log).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Copy Last Cmd", command=self._copy_last_cmd).pack(side=tk.LEFT, padx=(0, 6))
        self._autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_row, text="Auto-scroll", variable=self._autoscroll_var).pack(side=tk.LEFT)
        self._timestamps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_row, text="Timestamps", variable=self._timestamps_var).pack(side=tk.LEFT, padx=(8, 0))

        self._log_text = tk.Text(
            log_frame, height=12, width=120,
            font=_MONO_FONT, wrap=tk.WORD,
            bg=_LOG_BG, fg=_LOG_FG,
            insertbackground=_LOG_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=10, pady=8,
        )
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _log(self, text: str) -> None:
        if self._timestamps_var.get() and text.strip():
            ts = datetime.now().strftime("%H:%M:%S")
            text = f"[{ts}] {text}"
        self._log_text.insert(tk.END, text)
        if self._autoscroll_var.get():
            self._log_text.see(tk.END)

    def _clear_log(self) -> None:
        self._log_text.delete("1.0", tk.END)

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Log", "*.log"), ("All", "*.*")],
        )
        if path:
            Path(path).write_text(self._log_text.get("1.0", tk.END))
            self._log(f"[GUI] Log saved to: {path}\n")

    def _copy_last_cmd(self) -> None:
        """Copy the last [GUI] $ command line to the clipboard for terminal debugging."""
        log_text = self._log_text.get("1.0", tk.END)
        # Find all lines matching [GUI] $ <cmd>
        cmds = re.findall(r"\[GUI\] \$ (.+)", log_text)
        if cmds:
            self.root.clipboard_clear()
            self.root.clipboard_append(cmds[-1])
            self._log("[GUI] Last command copied to clipboard.\n")
        else:
            self._log("[GUI] No command found in log.\n")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=10, pady=(0, 6))

        self._status_label_var = tk.StringVar(value="Idle")
        lbl = ttk.Label(bar, textvariable=self._status_label_var, foreground=_MUTED)
        lbl.pack(side=tk.LEFT)

        self._pid_label_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._pid_label_var, foreground=_MUTED).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        self._elapsed_label_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._elapsed_label_var, foreground=_MUTED).pack(
            side=tk.RIGHT
        )

    def _tick_status(self) -> None:
        """Update the status bar every 500ms."""
        runner = self._runner
        multi_runner = self._multi
        is_running = runner.is_running or multi_runner.is_running

        if is_running:
            active = multi_runner._runner if multi_runner.is_running else runner
            pid = active.pid
            elapsed = active.elapsed
            self._status_label_var.set("Running")
            self._pid_label_var.set(f"PID: {pid}" if pid else "PID: starting…")
            mins, secs = divmod(int(elapsed), 60)
            self._elapsed_label_var.set(f"Elapsed: {mins}:{secs:02d}")
        else:
            self._status_label_var.set("Idle")
            self._pid_label_var.set("")
            self._elapsed_label_var.set("")

        self.root.after(500, self._tick_status)

    # ------------------------------------------------------------------
    # Shared callbacks
    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        cancel_state = "normal" if running else "disabled"
        for btn in self._get_run_buttons():
            btn.configure(state=state)
        for btn in self._get_cancel_buttons():
            btn.configure(state=cancel_state)

    def _get_run_buttons(self) -> list[ttk.Button]:
        btns = [self._batch_run_btn, self._single_run_btn, self._sv_run_btn]
        return btns

    def _get_cancel_buttons(self) -> list[ttk.Button]:
        btns = [self._batch_cancel_btn, self._single_cancel_btn, self._sv_cancel_btn, self._usd_cancel_btn]
        return btns

    def _cancel(self) -> None:
        if self._multi.is_running:
            self._multi.cancel()
        elif self._runner.is_running:
            self._runner.cancel()

    def _on_cmd_done(self, rc: int) -> None:
        self._set_running(False)
        if rc == 0:
            self._log("[GUI] \u2714 Command completed successfully.\n")
            self._status_label_var.set("Done (success)")
        else:
            self._log(f"[GUI] \u2718 Command exited with code {rc}.\n")
            self._status_label_var.set(f"Done (exit code {rc})")

    def _on_step_change(self, step: int, total: int, desc: str) -> None:
        self._batch_progress_var.set(f"Step {step}/{total}: {desc}")
        self._status_label_var.set(f"Running step {step}/{total}")

    def _on_multi_done(self, success: bool) -> None:
        self._set_running(False)
        self._batch_progress_var.set("")
        if success:
            self._log("\n[GUI] \u2714 All steps completed successfully!\n")
            self._status_label_var.set("Done (all steps passed)")
            messagebox.showinfo("Done", "All pipeline steps completed successfully!")
        else:
            self._log("\n[GUI] \u2718 Pipeline stopped (error or cancellation).\n")
            self._status_label_var.set("Stopped (error/cancelled)")

    def _on_close(self) -> None:
        if self._runner.is_running or self._multi.is_running:
            if not messagebox.askyesno("Confirm", "A process is still running. Exit anyway?"):
                return
            self._cancel()
        self._save_config()
        self.root.destroy()

    def _pyroki_env(self) -> dict[str, str]:
        """Return extra env vars for PyRoki/JAX subprocess based on acceleration setting."""
        accel = self._pyroki_accel_var.get()
        return {"JAX_PLATFORMS": accel}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._log("[GUI] ProtoMotions Retargeting GUI started.\n")
        self._log(f"[GUI] Project root: {PROJECT_ROOT}\n")
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = RetargetingGUI()
    app.run()
