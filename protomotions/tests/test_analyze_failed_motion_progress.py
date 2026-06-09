import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_failed_motion_progress.py"


def load_analyzer_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "analyze_failed_motion_progress", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_failed_file(run_dir: Path, epoch: int, rank: int, values: list[int]) -> None:
    failed_dir = run_dir / "failed_motions"
    failed_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(value) for value in values)
    if body:
        body += "\n"
    (failed_dir / f"failed_motions_epoch_{epoch}_rank_{rank}.txt").write_text(body)


def make_run_dir(tmp_path: Path, ngpu: int = 3) -> Path:
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(json.dumps({"ngpu": ngpu}))
    return run_dir


def test_build_analysis_aggregates_unique_failures_and_rank_coverage(tmp_path):
    module = load_analyzer_module()
    run_dir = make_run_dir(tmp_path, ngpu=3)

    write_failed_file(run_dir, epoch=100, rank=0, values=[1, 2])
    write_failed_file(run_dir, epoch=100, rank=1, values=[2, 3])
    write_failed_file(run_dir, epoch=100, rank=2, values=[])
    write_failed_file(run_dir, epoch=200, rank=0, values=[3])
    write_failed_file(run_dir, epoch=200, rank=1, values=[4])
    write_failed_file(run_dir, epoch=300, rank=0, values=[])
    write_failed_file(run_dir, epoch=300, rank=1, values=[4])
    write_failed_file(run_dir, epoch=300, rank=2, values=[4, 5])

    analysis = module.build_analysis(run_dir)
    summaries = {summary.epoch: summary for summary in analysis.epoch_summaries}

    assert analysis.failed_dir == run_dir / "failed_motions"
    assert analysis.expected_ranks == 3
    assert analysis.num_motions == 6
    assert analysis.num_motions_source == "inferred"

    assert summaries[100].is_complete is True
    assert summaries[100].ranks_present == (0, 1, 2)
    assert summaries[100].missing_ranks == ()
    assert summaries[100].rank_counts == {0: 2, 1: 2, 2: 0}
    assert summaries[100].sum_rank_failures == 4
    assert summaries[100].unique_failed_count == 3
    assert summaries[100].duplicate_failures == 1
    assert summaries[100].failed_percent == 50.0

    assert summaries[200].is_complete is False
    assert summaries[200].missing_ranks == (2,)
    assert summaries[200].unique_failed_count == 2

    assert [summary.epoch for summary in analysis.primary_epoch_summaries] == [100, 300]
    assert analysis.incomplete_epochs == [200]


def test_build_analysis_accepts_failed_motions_directory_and_num_motions_override(
    tmp_path,
):
    module = load_analyzer_module()
    run_dir = make_run_dir(tmp_path, ngpu=2)
    write_failed_file(run_dir, epoch=100, rank=0, values=[1, 2])
    write_failed_file(run_dir, epoch=100, rank=1, values=[2, 3])

    analysis = module.build_analysis(run_dir / "failed_motions", num_motions=10)
    summary = analysis.epoch_summaries[0]

    assert analysis.run_dir == run_dir
    assert analysis.failed_dir == run_dir / "failed_motions"
    assert analysis.num_motions == 10
    assert analysis.num_motions_source == "provided"
    assert summary.unique_failed_count == 3
    assert summary.failed_percent == 30.0


def test_persistence_uses_primary_complete_epochs_by_default(tmp_path):
    module = load_analyzer_module()
    run_dir = make_run_dir(tmp_path, ngpu=2)
    write_failed_file(run_dir, epoch=100, rank=0, values=[1, 2])
    write_failed_file(run_dir, epoch=100, rank=1, values=[2])
    write_failed_file(run_dir, epoch=200, rank=0, values=[3])
    write_failed_file(run_dir, epoch=300, rank=0, values=[2, 3])
    write_failed_file(run_dir, epoch=300, rank=1, values=[3])

    analysis = module.build_analysis(run_dir)
    persistence = {
        item.motion_id: item for item in analysis.persistence_summaries
    }

    assert set(persistence) == {1, 2, 3}
    assert persistence[2].failed_epoch_count == 2
    assert persistence[2].first_failed_epoch == 100
    assert persistence[2].last_failed_epoch == 300
    assert persistence[2].failed_in_final_primary_epoch is True
    assert persistence[1].failed_in_final_primary_epoch is False


def test_invalid_motion_id_reports_file_and_line(tmp_path):
    module = load_analyzer_module()
    run_dir = make_run_dir(tmp_path, ngpu=1)
    failed_dir = run_dir / "failed_motions"
    failed_dir.mkdir()
    bad_file = failed_dir / "failed_motions_epoch_100_rank_0.txt"
    bad_file.write_text("1\nnot-an-int\n")

    try:
        module.build_analysis(run_dir)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected invalid motion id to raise ValueError")

    assert bad_file.name in message
    assert "line 2" in message
    assert "not-an-int" in message


def test_cli_writes_png_and_csv_outputs(tmp_path):
    run_dir = make_run_dir(tmp_path, ngpu=2)
    write_failed_file(run_dir, epoch=100, rank=0, values=[1, 2])
    write_failed_file(run_dir, epoch=100, rank=1, values=[2])
    write_failed_file(run_dir, epoch=200, rank=0, values=[])
    write_failed_file(run_dir, epoch=200, rank=1, values=[3])
    out_dir = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "failed_motion_progress.png").is_file()
    epoch_csv = out_dir / "failed_motion_progress_by_epoch.csv"
    persistence_csv = out_dir / "failed_motion_persistence.csv"
    assert epoch_csv.is_file()
    assert persistence_csv.is_file()
    assert "First complete epoch: 100" in result.stdout
    assert "Final complete epoch: 200" in result.stdout

    with epoch_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["epoch"] == "100"
    assert rows[0]["is_complete"] == "true"
    assert rows[0]["unique_failed_count"] == "2"
    assert rows[0]["primary_epoch"] == "true"
