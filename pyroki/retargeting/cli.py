# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from retargeting.config import PyrokiRetargetConfig
from retargeting.factory import get_retarget_config, supported_robot_types


@dataclass(frozen=True)
class BatchRetargetingOptions:
    keypoints_folder_path: Path
    output_dir: Path
    subsample_factor: int
    target_raw_frames: int
    skip_existing: bool
    source_type: str
    save_contacts_only: bool
    contacts_dir: Path | None
    input_fps: float
    visualize: bool


def run_batch_retargeting(
    config: PyrokiRetargetConfig, options: BatchRetargetingOptions
) -> int:
    from retargeting.solver import run_batch_retargeting as solver_run_batch_retargeting

    solver_run_batch_retargeting(config, options)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        default="g1",
        choices=supported_robot_types(),
        help="Target robot type (default: g1).",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
    )
    parser.add_argument("--keypoints-folder-path", required=True)
    parser.add_argument("--output-dir", default="./retargeted_output_motions")
    parser.add_argument("--urdf-path")
    parser.add_argument("--mesh-dir")
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--target-raw-frames", type=int, default=450)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--source-type", default="smpl")
    parser.add_argument("--save-contacts-only", action="store_true")
    parser.add_argument("--contacts-dir")
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.set_defaults(visualize=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = get_retarget_config(args.robot_type)
    if args.urdf_path is not None:
        config = replace(config, urdf_path=Path(args.urdf_path))
    if args.mesh_dir is not None:
        config = replace(config, mesh_dir=Path(args.mesh_dir))

    options = BatchRetargetingOptions(
        keypoints_folder_path=Path(args.keypoints_folder_path),
        output_dir=Path(args.output_dir),
        subsample_factor=args.subsample_factor,
        target_raw_frames=args.target_raw_frames,
        skip_existing=args.skip_existing,
        source_type=args.source_type,
        save_contacts_only=args.save_contacts_only,
        contacts_dir=Path(args.contacts_dir) if args.contacts_dir is not None else None,
        input_fps=args.input_fps,
        visualize=args.visualize,
    )
    return run_batch_retargeting(config, options)
