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
    target_raw_frames: int | None
    skip_existing: bool
    source_type: str
    save_contacts_only: bool
    contacts_dir: Path | None
    input_fps: float
    visualize: bool
    chunk_long_motions: bool = False
    chunk_threshold_frames: int = 900
    chunk_size_frames: int = 450
    chunk_overlap_frames: int = 60


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
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=None,
        help="Raw source frames to optimize. Defaults to each motion's full length.",
    )
    parser.add_argument(
        "--chunk-long-motions",
        action="store_true",
        help="Retarget long motions as overlapping chunks and stitch one output.",
    )
    parser.add_argument("--chunk-threshold-frames", type=int, default=900)
    parser.add_argument("--chunk-size-frames", type=int, default=450)
    parser.add_argument("--chunk-overlap-frames", type=int, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--source-type", default="smpl")
    parser.add_argument("--save-contacts-only", action="store_true")
    parser.add_argument("--contacts-dir")
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.set_defaults(visualize=True)
    return parser


def validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.chunk_size_frames <= 0:
        parser.error("--chunk-size-frames must be positive")
    if args.chunk_overlap_frames < 0:
        parser.error("--chunk-overlap-frames must be non-negative")
    if args.chunk_overlap_frames >= args.chunk_size_frames:
        parser.error("--chunk-overlap-frames must be smaller than --chunk-size-frames")
    if args.chunk_threshold_frames < args.chunk_size_frames:
        parser.error("--chunk-threshold-frames must be at least --chunk-size-frames")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_cli_args(parser, args)

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
        chunk_long_motions=args.chunk_long_motions,
        chunk_threshold_frames=args.chunk_threshold_frames,
        chunk_size_frames=args.chunk_size_frames,
        chunk_overlap_frames=args.chunk_overlap_frames,
    )
    return run_batch_retargeting(config, options)
