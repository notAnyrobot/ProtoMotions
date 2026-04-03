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
"""Thread-safe subprocess runner for the retargeting GUI.

Launches external commands in background threads and delivers stdout/stderr
line-by-line to a tkinter callback via ``root.after()``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable, Optional


class SubprocessRunner:
    """Run a command asynchronously and stream its output."""

    def __init__(
        self,
        tk_root,
        on_output: Callable[[str], None],
        on_complete: Callable[[int], None],
    ):
        self._root = tk_root
        self._on_output = on_output
        self._on_complete = on_complete
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cancelled = False
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def pid(self) -> int | None:
        proc = self._process
        return proc.pid if proc and proc.poll() is None else None

    @property
    def elapsed(self) -> float:
        if self._start_time and self.is_running:
            return time.monotonic() - self._start_time
        return 0.0

    def start(
        self,
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if self.is_running:
            self._emit("[GUI] A process is already running. Cancel it first.\n")
            return

        self._cancelled = False
        merged_env = dict(os.environ)
        # Force unbuffered stdout so output streams to the log in real time
        merged_env["PYTHONUNBUFFERED"] = "1"
        if env:
            merged_env.update(env)

        self._emit(f"[GUI] $ {' '.join(cmd)}\n")
        self._emit(f"[GUI] cwd={cwd or os.getcwd()}\n")
        if env:
            self._emit(f"[GUI] env: {env}\n")

        self._start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            args=(cmd, cwd, merged_env),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        if self._process and self._process.poll() is None:
            self._cancelled = True
            self._emit("[GUI] Sending SIGTERM...\n")
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

    def _run(
        self,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str],
    ) -> None:
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
            self._emit(f"[GUI] Process started (PID {self._process.pid})\n")
            assert self._process.stdout is not None
            # Use readline() loop instead of iterator — the iterator
            # internally buffers even with bufsize=1, which delays output.
            while True:
                line = self._process.stdout.readline()
                if not line:
                    break
                if self._cancelled:
                    break
                self._emit(line)
            self._process.wait()
            rc = self._process.returncode
        except FileNotFoundError as exc:
            self._emit(f"[GUI] Command not found: {exc}\n")
            self._emit(f"[GUI] Check that the Python interpreter path is correct.\n")
            rc = -1
        except Exception as exc:
            self._emit(f"[GUI] Error: {exc}\n")
            rc = -1
        finally:
            elapsed = time.monotonic() - self._start_time if self._start_time else 0
            self._process = None
            if self._cancelled:
                self._emit("[GUI] Process cancelled.\n")
            self._emit(f"[GUI] Elapsed: {elapsed:.1f}s\n")
            self._root.after(0, self._on_complete, rc)

    def _emit(self, text: str) -> None:
        self._root.after(0, self._on_output, text)


class MultiStepRunner:
    """Run multiple subprocess commands sequentially with progress tracking."""

    def __init__(
        self,
        tk_root,
        on_output: Callable[[str], None],
        on_step_change: Callable[[int, int, str], None],
        on_complete: Callable[[bool], None],
    ):
        self._root = tk_root
        self._on_output = on_output
        self._on_step_change = on_step_change
        self._on_all_complete = on_complete
        self._runner = SubprocessRunner(tk_root, on_output, self._step_done)
        self._steps: list[tuple[str, list[str], str | None]] = []
        self._current_step = 0
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self._runner.is_running

    def start(self, steps: list[tuple[str, list[str], str | None] | tuple[str, list[str], str | None, dict[str, str] | None]]) -> None:
        """Start running a list of (description, cmd, cwd[, env]) tuples sequentially."""
        if self._runner.is_running:
            self._on_output("[GUI] A process is already running.\n")
            return

        self._steps = steps
        self._current_step = 0
        self._cancelled = False
        self._run_next()

    def cancel(self) -> None:
        self._cancelled = True
        self._runner.cancel()

    def _run_next(self) -> None:
        if self._cancelled or self._current_step >= len(self._steps):
            success = not self._cancelled and self._current_step >= len(self._steps)
            self._root.after(0, self._on_all_complete, success)
            return

        step = self._steps[self._current_step]
        desc, cmd, cwd = step[0], step[1], step[2]
        env = step[3] if len(step) > 3 else None
        total = len(self._steps)
        step_num = self._current_step + 1
        self._root.after(0, self._on_step_change, step_num, total, desc)
        self._on_output(f"\n{'='*60}\n[Step {step_num}/{total}] {desc}\n{'='*60}\n")
        self._runner.start(cmd, cwd=cwd, env=env)

    def _step_done(self, rc: int) -> None:
        if rc != 0 and not self._cancelled:
            self._on_output(
                f"\n[GUI] Step failed with return code {rc}. Stopping.\n"
            )
            self._root.after(0, self._on_all_complete, False)
            return

        self._current_step += 1
        self._root.after(10, self._run_next)
