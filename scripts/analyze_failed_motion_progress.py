#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Analyze distributed failed-motion logs over training epochs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FAILED_MOTION_PATTERN = re.compile(
    r"^failed_motions_epoch_(?P<epoch>\d+)_rank_(?P<rank>\d+)\.txt$"
)


@dataclass(frozen=True)
class FailedMotionFile:
    epoch: int
    rank: int
    motion_ids: tuple[int, ...]
    path: Path


@dataclass
class EpochSummary:
    epoch: int
    ranks_present: tuple[int, ...]
    missing_ranks: tuple[int, ...]
    rank_counts: dict[int, int]
    unique_failed_ids: tuple[int, ...]
    sum_rank_failures: int
    unique_failed_count: int
    duplicate_failures: int
    failed_percent: Optional[float]
    is_complete: bool
    primary_epoch: bool


@dataclass(frozen=True)
class MotionPersistence:
    motion_id: int
    failed_epoch_count: int
    failed_epoch_percent: Optional[float]
    first_failed_epoch: int
    last_failed_epoch: int
    failed_in_final_primary_epoch: bool


@dataclass
class AnalysisResult:
    run_dir: Path
    failed_dir: Path
    expected_ranks: int
    num_motions: Optional[int]
    num_motions_source: str
    epoch_summaries: list[EpochSummary]
    primary_epoch_summaries: list[EpochSummary]
    persistence_summaries: list[MotionPersistence]
    incomplete_epochs: list[int]
    warnings: list[str]


def resolve_input_path(run_path: Path) -> tuple[Path, Path]:
    """Resolve either a run directory or a direct failed_motions directory."""
    path = run_path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {path}")

    nested_failed_dir = path / "failed_motions"
    if nested_failed_dir.is_dir():
        return path, nested_failed_dir

    has_failed_files = any(
        child.is_file() and FAILED_MOTION_PATTERN.match(child.name)
        for child in path.iterdir()
    )
    if path.name == "failed_motions" or has_failed_files:
        return path.parent, path

    raise FileNotFoundError(
        f"Could not find failed_motions directory under run path: {path}"
    )


def _read_config_ngpu(run_dir: Path) -> Optional[int]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    text = config_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and "ngpu" in data:
        try:
            ngpu = int(data["ngpu"])
        except (TypeError, ValueError):
            return None
        return ngpu if ngpu > 0 else None

    match = re.search(r"['\"]?ngpu['\"]?\s*[:=]\s*['\"]?(\d+)", text)
    if not match:
        return None
    ngpu = int(match.group(1))
    return ngpu if ngpu > 0 else None


def parse_failed_motion_files(
    failed_dir: Path,
) -> tuple[list[FailedMotionFile], list[str]]:
    records: list[FailedMotionFile] = []
    warnings: list[str] = []

    for path in sorted(failed_dir.iterdir()):
        if not path.is_file():
            continue

        match = FAILED_MOTION_PATTERN.match(path.name)
        if not match:
            if path.name.startswith("failed_motions_epoch_") and path.suffix == ".txt":
                warnings.append(
                    f"Ignoring unrecognized failed-motion file: {path.name}"
                )
            continue

        motion_ids: list[int] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                motion_id = int(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid motion id in {path.name} line {line_number}: "
                    f"{stripped!r}"
                ) from exc
            if motion_id < 0:
                raise ValueError(
                    f"Invalid negative motion id in {path.name} line {line_number}: "
                    f"{motion_id}"
                )
            motion_ids.append(motion_id)

        records.append(
            FailedMotionFile(
                epoch=int(match.group("epoch")),
                rank=int(match.group("rank")),
                motion_ids=tuple(motion_ids),
                path=path,
            )
        )

    if not records:
        raise ValueError(f"No failed-motion files found in: {failed_dir}")

    records.sort(key=lambda record: (record.epoch, record.rank))
    return records, warnings


def _expected_ranks_from_records(records: list[FailedMotionFile]) -> int:
    return max(record.rank for record in records) + 1


def _resolve_expected_ranks(
    run_dir: Path,
    records: list[FailedMotionFile],
    expected_ranks: Optional[int],
) -> int:
    if expected_ranks is not None:
        if expected_ranks <= 0:
            raise ValueError("--expected-ranks must be greater than zero")
        return expected_ranks

    config_ngpu = _read_config_ngpu(run_dir)
    if config_ngpu is not None:
        return config_ngpu

    return _expected_ranks_from_records(records)


def _resolve_num_motions(
    records: list[FailedMotionFile],
    num_motions: Optional[int],
) -> tuple[Optional[int], str]:
    if num_motions is not None:
        if num_motions <= 0:
            raise ValueError("--num-motions must be greater than zero")
        return num_motions, "provided"

    all_motion_ids = [
        motion_id for record in records for motion_id in record.motion_ids
    ]
    if not all_motion_ids:
        return None, "unavailable"
    return max(all_motion_ids) + 1, "inferred"


def _summarize_epochs(
    records: list[FailedMotionFile],
    expected_ranks: int,
    num_motions: Optional[int],
    include_incomplete_primary: bool,
) -> list[EpochSummary]:
    records_by_epoch: dict[int, list[FailedMotionFile]] = defaultdict(list)
    for record in records:
        records_by_epoch[record.epoch].append(record)

    expected_rank_set = set(range(expected_ranks))
    summaries: list[EpochSummary] = []

    for epoch in sorted(records_by_epoch):
        epoch_records = sorted(records_by_epoch[epoch], key=lambda record: record.rank)
        ranks_present = tuple(record.rank for record in epoch_records)
        ranks_present_set = set(ranks_present)
        missing_ranks = tuple(
            rank for rank in range(expected_ranks) if rank not in ranks_present_set
        )
        is_complete = ranks_present_set == expected_rank_set

        rank_counts = {
            record.rank: len(record.motion_ids) for record in epoch_records
        }
        all_ids = [
            motion_id
            for record in epoch_records
            for motion_id in record.motion_ids
        ]
        unique_failed_ids = tuple(sorted(set(all_ids)))
        unique_failed_count = len(unique_failed_ids)
        sum_rank_failures = len(all_ids)
        failed_percent = (
            (unique_failed_count / num_motions) * 100.0
            if num_motions is not None
            else None
        )

        summaries.append(
            EpochSummary(
                epoch=epoch,
                ranks_present=ranks_present,
                missing_ranks=missing_ranks,
                rank_counts=rank_counts,
                unique_failed_ids=unique_failed_ids,
                sum_rank_failures=sum_rank_failures,
                unique_failed_count=unique_failed_count,
                duplicate_failures=sum_rank_failures - unique_failed_count,
                failed_percent=failed_percent,
                is_complete=is_complete,
                primary_epoch=is_complete or include_incomplete_primary,
            )
        )

    return summaries


def _build_persistence(
    primary_summaries: list[EpochSummary],
) -> list[MotionPersistence]:
    if not primary_summaries:
        return []

    epochs_by_motion: dict[int, list[int]] = defaultdict(list)
    for summary in primary_summaries:
        for motion_id in summary.unique_failed_ids:
            epochs_by_motion[motion_id].append(summary.epoch)

    final_primary_epoch = primary_summaries[-1].epoch
    primary_epoch_count = len(primary_summaries)
    persistence: list[MotionPersistence] = []

    for motion_id, epochs in epochs_by_motion.items():
        sorted_epochs = sorted(epochs)
        failed_epoch_count = len(sorted_epochs)
        persistence.append(
            MotionPersistence(
                motion_id=motion_id,
                failed_epoch_count=failed_epoch_count,
                failed_epoch_percent=(failed_epoch_count / primary_epoch_count)
                * 100.0,
                first_failed_epoch=sorted_epochs[0],
                last_failed_epoch=sorted_epochs[-1],
                failed_in_final_primary_epoch=final_primary_epoch in sorted_epochs,
            )
        )

    persistence.sort(
        key=lambda item: (
            -item.failed_epoch_count,
            item.first_failed_epoch,
            item.motion_id,
        )
    )
    return persistence


def build_analysis(
    run_path: Path | str,
    *,
    expected_ranks: Optional[int] = None,
    num_motions: Optional[int] = None,
    include_incomplete_primary: bool = False,
) -> AnalysisResult:
    run_dir, failed_dir = resolve_input_path(Path(run_path))
    records, warnings = parse_failed_motion_files(failed_dir)
    resolved_expected_ranks = _resolve_expected_ranks(
        run_dir, records, expected_ranks
    )
    resolved_num_motions, num_motions_source = _resolve_num_motions(
        records, num_motions
    )
    epoch_summaries = _summarize_epochs(
        records,
        resolved_expected_ranks,
        resolved_num_motions,
        include_incomplete_primary,
    )
    primary_epoch_summaries = [
        summary for summary in epoch_summaries if summary.primary_epoch
    ]
    incomplete_epochs = [
        summary.epoch for summary in epoch_summaries if not summary.is_complete
    ]
    persistence_summaries = _build_persistence(primary_epoch_summaries)

    if incomplete_epochs and not include_incomplete_primary:
        warnings.append(
            f"{len(incomplete_epochs)} epoch(s) have missing rank files and are "
            "excluded from the primary trend by default"
        )

    return AnalysisResult(
        run_dir=run_dir,
        failed_dir=failed_dir,
        expected_ranks=resolved_expected_ranks,
        num_motions=resolved_num_motions,
        num_motions_source=num_motions_source,
        epoch_summaries=epoch_summaries,
        primary_epoch_summaries=primary_epoch_summaries,
        persistence_summaries=persistence_summaries,
        incomplete_epochs=incomplete_epochs,
        warnings=warnings,
    )


def _bool_field(value: bool) -> str:
    return "true" if value else "false"


def _list_field(values: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(value) for value in values)


def _percent_field(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_epoch_csv(analysis: AnalysisResult, output_path: Path) -> None:
    fieldnames = [
        "epoch",
        "is_complete",
        "primary_epoch",
        "ranks_present",
        "missing_ranks",
        "unique_failed_count",
        "failed_percent",
        "sum_rank_failures",
        "duplicate_failures",
        "rank_counts",
        "unique_failed_ids",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in analysis.epoch_summaries:
            writer.writerow(
                {
                    "epoch": summary.epoch,
                    "is_complete": _bool_field(summary.is_complete),
                    "primary_epoch": _bool_field(summary.primary_epoch),
                    "ranks_present": _list_field(summary.ranks_present),
                    "missing_ranks": _list_field(summary.missing_ranks),
                    "unique_failed_count": summary.unique_failed_count,
                    "failed_percent": _percent_field(summary.failed_percent),
                    "sum_rank_failures": summary.sum_rank_failures,
                    "duplicate_failures": summary.duplicate_failures,
                    "rank_counts": json.dumps(
                        summary.rank_counts, sort_keys=True, separators=(",", ":")
                    ),
                    "unique_failed_ids": _list_field(summary.unique_failed_ids),
                }
            )


def write_persistence_csv(analysis: AnalysisResult, output_path: Path) -> None:
    fieldnames = [
        "motion_id",
        "failed_epoch_count",
        "failed_epoch_percent",
        "first_failed_epoch",
        "last_failed_epoch",
        "failed_in_final_primary_epoch",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in analysis.persistence_summaries:
            writer.writerow(
                {
                    "motion_id": item.motion_id,
                    "failed_epoch_count": item.failed_epoch_count,
                    "failed_epoch_percent": _percent_field(
                        item.failed_epoch_percent
                    ),
                    "first_failed_epoch": item.first_failed_epoch,
                    "last_failed_epoch": item.last_failed_epoch,
                    "failed_in_final_primary_epoch": _bool_field(
                        item.failed_in_final_primary_epoch
                    ),
                }
            )


def _tick_indices(count: int, max_ticks: int = 10) -> list[int]:
    if count <= max_ticks:
        return list(range(count))
    step = max(1, count // max_ticks)
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def render_report(
    analysis: AnalysisResult,
    output_path: Path,
    *,
    top_k: int = 20,
) -> None:
    if not analysis.primary_epoch_summaries:
        raise ValueError(
            "No primary epochs available for plotting. Complete rank coverage is "
            "missing for every epoch; use --include-incomplete-primary to plot "
            "incomplete epochs anyway."
        )

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "protomotions_matplotlib_cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = analysis.primary_epoch_summaries
    epochs = [summary.epoch for summary in primary]
    unique_counts = [summary.unique_failed_count for summary in primary]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    title_bits = [
        analysis.run_dir.name,
        f"expected ranks: {analysis.expected_ranks}",
    ]
    if analysis.num_motions is not None:
        title_bits.append(
            f"denominator: {analysis.num_motions} ({analysis.num_motions_source})"
        )
    else:
        title_bits.append("denominator: unavailable")
    fig.suptitle("Failed motion progress - " + " | ".join(title_bits), fontsize=14)

    ax = axes[0]
    ax.plot(
        epochs,
        unique_counts,
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="primary epochs",
    )
    incomplete = [
        summary for summary in analysis.epoch_summaries if not summary.is_complete
    ]
    if incomplete:
        ax.scatter(
            [summary.epoch for summary in incomplete],
            [summary.unique_failed_count for summary in incomplete],
            marker="x",
            color="tab:gray",
            label="incomplete rank coverage",
            zorder=3,
        )
    ax.set_ylabel("Unique failed motions")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    all_epochs = [summary.epoch for summary in analysis.epoch_summaries]
    for rank in range(analysis.expected_ranks):
        counts = [
            summary.rank_counts.get(rank, float("nan"))
            for summary in analysis.epoch_summaries
        ]
        ax.plot(all_epochs, counts, marker=".", linewidth=1.0, label=f"rank {rank}")
    ax.set_ylabel("Per-rank failures")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=min(analysis.expected_ranks, 4))

    ax = axes[2]
    top_persistence = analysis.persistence_summaries[:top_k]
    if top_persistence:
        motion_ids = [item.motion_id for item in top_persistence]
        primary_sets = [
            set(summary.unique_failed_ids) for summary in analysis.primary_epoch_summaries
        ]
        matrix = [
            [1 if motion_id in failed_ids else 0 for failed_ids in primary_sets]
            for motion_id in motion_ids
        ]
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys")
        ax.set_yticks(range(len(motion_ids)))
        ax.set_yticklabels([str(motion_id) for motion_id in motion_ids])
        tick_indices = _tick_indices(len(epochs))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(epochs[index]) for index in tick_indices], rotation=45)
        ax.set_ylabel("Motion ID")
        ax.set_xlabel("Primary epoch")
        ax.set_title(f"Top {len(motion_ids)} persistent failed motions")
    else:
        ax.text(
            0.5,
            0.5,
            "No failed motions in primary epochs",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    analysis: AnalysisResult,
    output_dir: Path,
    *,
    top_k: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "failed_motion_progress.png"
    epoch_csv_path = output_dir / "failed_motion_progress_by_epoch.csv"
    persistence_csv_path = output_dir / "failed_motion_persistence.csv"

    write_epoch_csv(analysis, epoch_csv_path)
    write_persistence_csv(analysis, persistence_csv_path)
    render_report(analysis, png_path, top_k=top_k)
    return png_path, epoch_csv_path, persistence_csv_path


def print_summary(
    analysis: AnalysisResult,
    *,
    png_path: Path,
    epoch_csv_path: Path,
    persistence_csv_path: Path,
) -> None:
    complete_summaries = [
        summary for summary in analysis.epoch_summaries if summary.is_complete
    ]
    if complete_summaries:
        first_complete = complete_summaries[0]
        final_complete = complete_summaries[-1]
        print(
            f"First complete epoch: {first_complete.epoch} "
            f"({first_complete.unique_failed_count} unique failures)"
        )
        print(
            f"Final complete epoch: {final_complete.epoch} "
            f"({final_complete.unique_failed_count} unique failures)"
        )
    else:
        print("First complete epoch: none")
        print("Final complete epoch: none")

    if analysis.primary_epoch_summaries:
        min_summary = min(
            analysis.primary_epoch_summaries,
            key=lambda summary: summary.unique_failed_count,
        )
        print(
            f"Minimum primary failures: {min_summary.unique_failed_count} "
            f"at epoch {min_summary.epoch}"
        )
    print(f"Incomplete epochs: {len(analysis.incomplete_epochs)}")
    if analysis.num_motions is not None:
        print(
            f"Motion denominator: {analysis.num_motions} "
            f"({analysis.num_motions_source})"
        )
    else:
        print("Motion denominator: unavailable")

    for warning in analysis.warnings:
        print(f"Warning: {warning}")

    print(f"PNG: {png_path}")
    print(f"Epoch CSV: {epoch_csv_path}")
    print(f"Persistence CSV: {persistence_csv_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze distributed failed-motion files from a ProtoMotions run. "
            "Canonical usage: python scripts/analyze_failed_motion_progress.py "
            "results/<experiment-name>"
        )
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="Experiment run directory, or a direct failed_motions directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for failed_motion_progress.png and CSV outputs. Defaults "
            "to <run_dir>/failed_motions/progress_report."
        ),
    )
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=None,
        help="Override expected distributed rank count.",
    )
    parser.add_argument(
        "--num-motions",
        type=int,
        default=None,
        help="Exact motion denominator for percentages.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of persistent failed motion IDs to show in the report.",
    )
    parser.add_argument(
        "--include-incomplete-primary",
        action="store_true",
        help="Include epochs with missing rank files in the primary trend.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.top_k <= 0:
        print("Error: --top-k must be greater than zero", file=sys.stderr)
        return 2

    try:
        analysis = build_analysis(
            args.run_path,
            expected_ranks=args.expected_ranks,
            num_motions=args.num_motions,
            include_incomplete_primary=args.include_incomplete_primary,
        )
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else analysis.failed_dir / "progress_report"
        )
        png_path, epoch_csv_path, persistence_csv_path = write_outputs(
            analysis, out_dir, top_k=args.top_k
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        analysis,
        png_path=png_path,
        epoch_csv_path=epoch_csv_path,
        persistence_csv_path=persistence_csv_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
