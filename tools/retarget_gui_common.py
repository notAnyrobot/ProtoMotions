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
"""Shared GUI utilities for the ProtoMotions retargeting tool suite.

This module provides:
- Path constants and default interpreter paths
- UI theme constants
- ToolTip helper class
- Widget builder functions (``_labeled_entry``, ``_labeled_combo``, etc.)
- ``RetargetGUIBase`` — an abstract base that wires up config persistence,
  theme, environment panel, log panel, status bar, and subprocess management.

Both ``protomotions_retarget_gui.py`` and ``pyroki_retarget_gui.py`` subclass
``RetargetGUIBase`` and share the same ``CONFIG_FILE``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import yaml  # noqa: F401 — available for subclass convenience
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
_ACCENT = "#2563eb"  # Blue-600  — primary action color
_ACCENT_HOVER = "#1d4ed8"  # Blue-700
_BG = "#f1f5f9"  # Slate-50  — window / frame background
_SURFACE = "#ffffff"  # White     — input fields & cards
_BORDER = "#cbd5e1"  # Slate-300
_TEXT = "#1e293b"  # Slate-800
_MUTED = "#64748b"  # Slate-500
_SUCCESS = "#16a34a"  # Green-600
_ERROR = "#dc2626"  # Red-600
_LOG_BG = "#1e293b"  # Dark slate — log background
_LOG_FG = "#e2e8f0"  # Slate-200  — log text
_LOG_ACCENT = "#38bdf8"  # Sky-400    — log highlights
# Font families the user can cycle through
_FONT_FAMILIES = (
    "Helvetica",
    "DejaVu Sans",
    "Noto Sans",
    "Liberation Sans",
    "Ubuntu",
    "Segoe UI",
    "Arial",
)
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
                    initialdir=os.path.dirname(init)
                    if init != str(PROJECT_ROOT)
                    else init,
                    filetypes=filetypes or [("All files", "*.*")],
                )
            else:
                path = filedialog.askopenfilename(
                    initialdir=os.path.dirname(init)
                    if init != str(PROJECT_ROOT)
                    else init,
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
    combo = ttk.Combobox(
        frame, textvariable=var, values=list(values), width=width, state="readonly"
    )
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
    spin = ttk.Spinbox(
        frame, from_=from_, to=to, textvariable=var, increment=increment, width=width
    )
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
# Shared GUI Base Class
# ═══════════════════════════════════════════════════════════════════════════


class RetargetGUIBase:
    """Abstract base class shared by both the ProtoMotions and PyRoki GUIs.

    Provides:
    - ``tk.Tk`` root window creation with config persistence
    - Theme application
    - Environment panel (parameterized by *env_mode*)
    - Log panel + status bar
    - Subprocess runner management (``SubprocessRunner`` / ``MultiStepRunner``)
    - Shared callbacks: cancel, on_cmd_done, on_close, etc.

    Subclasses MUST call ``super().__init__()`` then build their own notebook
    via ``self._build_notebook()``.

    Parameters
    ----------
    title : str
        Window title.
    env_mode : ``"proto"`` | ``"pyroki"`` | ``"both"``
        Which interpreter fields to show in the environment panel.
    """

    def __init__(self, title: str, env_mode: str = "both") -> None:
        self._env_mode = env_mode

        self.root = tk.Tk()
        self.root.title(title)
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

        self._build_env_panel()
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
    # Theme
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
            "TLabelframe.Label",
            background=_BG,
            foreground=_ACCENT,
            font=(fam, sz + 1, "bold"),
        )
        style.configure("TNotebook", background=_BG, tabmargins=[4, 4, 4, 0])
        style.configure(
            "TNotebook.Tab",
            padding=[16, 7],
            font=(fam, sz + 1, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", _SURFACE), ("!selected", _BG)],
            foreground=[("selected", _ACCENT), ("!selected", _MUTED)],
        )
        style.configure(
            "TButton",
            font=(fam, sz),
        )
        style.configure(
            "Accent.TButton",
            background=_ACCENT,
            foreground="#ffffff",
            font=(fam, sz + 1, "bold"),
            padding=[14, 6],
        )
        style.map("Accent.TButton", background=[("active", _ACCENT_HOVER)])
        style.configure("TEntry", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TSpinbox", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TCombobox", fieldbackground=_SURFACE, font=(fam, sz))
        style.configure("TCheckbutton", font=(fam, sz))
        style.configure("Header.TLabel", font=(fam, sz + 1, "bold"), foreground=_ACCENT)
        style.configure(
            "Treeview",
            background=_SURFACE,
            fieldbackground=_SURFACE,
            foreground=_TEXT,
            rowheight=26,
            font=(fam, sz),
        )
        style.configure("Treeview.Heading", font=(fam, sz, "bold"))

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                self._saved_config = json.loads(CONFIG_FILE.read_text())
            except Exception:
                self._saved_config = {}

    def _save_config(self) -> None:
        cfg: dict[str, Any] = {
            "font_family": self._font_family,
        }
        if hasattr(self, "_proto_python_var"):
            cfg["proto_python"] = self._proto_python_var.get()
        if hasattr(self, "_pyroki_python_var"):
            cfg["pyroki_python"] = self._pyroki_python_var.get()
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

        mode = self._env_mode

        # ProtoMotions Python — shown for "proto" or "both"
        if mode in ("proto", "both"):
            proto_default = self._saved_config.get(
                "proto_python", _DEFAULT_PROTO_PYTHON
            )
            _, self._proto_python_var = _labeled_entry(
                frame,
                "ProtoMotions Python:",
                proto_default,
                tooltip="Path to the Python interpreter with ProtoMotions installed (env_isaaclab).",
                browse="file",
                filetypes=[("Python", "python*"), ("All", "*")],
            )

        # PyRoki Python — shown for "pyroki" or "both"
        if mode in ("pyroki", "both"):
            pyroki_default = self._saved_config.get(
                "pyroki_python", _detect_pyroki_python()
            )
            _, self._pyroki_python_var = _labeled_entry(
                frame,
                "PyRoki Python:",
                pyroki_default,
                tooltip="Path to the Python interpreter with PyRoki + JAX installed (pyroki-cuda env).",
                browse="file",
                filetypes=[("Python", "python*"), ("All", "*")],
            )

            # JAX acceleration mode — only relevant when PyRoki is shown
            _, self._pyroki_accel_var = _labeled_combo(
                frame,
                "PyRoki Acceleration:",
                ("cuda", "cpu"),
                "cuda",
                tooltip="JAX backend for PyRoki retargeting. 'cuda' uses GPU (default), 'cpu' for CPU-only.",
            )

        # Buttons row
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            btn_row,
            text="  Verify Environments  ",
            command=self._verify_envs,
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
            font_row,
            textvariable=self._font_var,
            values=list(_FONT_FAMILIES),
            width=22,
            state="readonly",
        )
        font_combo.pack(side=tk.LEFT)
        font_combo.bind("<<ComboboxSelected>>", self._on_font_change)
        ToolTip(font_combo, "Choose the UI font family. Saved across sessions.")

    def _verify_envs(self) -> None:
        self._log("\n[GUI] Verifying environments...\n")
        results: list[str] = []
        mode = self._env_mode

        # ProtoMotions
        if mode in ("proto", "both"):
            proto = self._proto_python_var.get()
            try:
                out = subprocess.run(
                    [proto, "-c", "import protomotions; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
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
        if mode in ("pyroki", "both"):
            pyroki = self._pyroki_python_var.get()
            try:
                out = subprocess.run(
                    [pyroki, "-c", "import pyroki; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
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
                    capture_output=True,
                    text=True,
                    timeout=30,
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
    # Log Panel (bottom)
    # ------------------------------------------------------------------

    def _build_log_panel(self) -> None:
        log_frame = ttk.LabelFrame(self.root, text="  Output Log  ", padding=6)
        log_frame.pack(fill=tk.X, padx=10, pady=(4, 10))

        btn_row = ttk.Frame(log_frame)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self._clear_log).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row, text="Save Log", command=self._save_log).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row, text="Copy Last Cmd", command=self._copy_last_cmd).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btn_row, text="Auto-scroll", variable=self._autoscroll_var
        ).pack(side=tk.LEFT)
        self._timestamps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_row, text="Timestamps", variable=self._timestamps_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self._log_text = tk.Text(
            log_frame,
            height=12,
            width=120,
            font=_MONO_FONT,
            wrap=tk.WORD,
            bg=_LOG_BG,
            fg=_LOG_FG,
            insertbackground=_LOG_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=8,
        )
        log_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self._log_text.yview
        )
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
        """Return all 'Run' buttons.  Subclasses MUST override."""
        return []

    def _get_cancel_buttons(self) -> list[ttk.Button]:
        """Return all 'Cancel' buttons.  Subclasses MUST override."""
        return []

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
            if not messagebox.askyesno(
                "Confirm", "A process is still running. Exit anyway?"
            ):
                return
            self._cancel()
        self._save_config()
        self.root.destroy()

    def _pyroki_env(self) -> dict[str, str]:
        """Return extra env vars for PyRoki/JAX subprocess based on acceleration setting."""
        if hasattr(self, "_pyroki_accel_var"):
            accel = self._pyroki_accel_var.get()
            return {"JAX_PLATFORMS": accel}
        return {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._log(f"[GUI] {self.root.title()} started.\n")
        self._log(f"[GUI] Project root: {PROJECT_ROOT}\n")
        self.root.mainloop()
