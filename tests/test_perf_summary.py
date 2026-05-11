"""Tests for ``scripts/perf_summary.py``.

Covers (a) the aggregator runs end-to-end when ``bench_results/`` has
at least one CSV per known primitive, (b) the printed output includes
the expected primitive names, (c) the 1000-game smoke wall-time
estimate is non-zero and in a sensible magnitude, (d) missing
primitives are surfaced as a note rather than crashing the script,
and (e) the ``--csv`` output mode writes a well-formed CSV.

These tests don't depend on ``mtg_bot_engine`` — the aggregator only
parses CSV files on disk, so the tests synthesize tiny CSV fixtures
in a tmp dir and point ``perf_summary`` at them via ``--bench-dir``.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import pathlib
import sys
from contextlib import redirect_stdout


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "perf_summary.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "perf_summary", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("perf_summary", module)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _populate_full_bench_dir(bench_dir: pathlib.Path) -> None:
    """Write one CSV per known primitive into ``bench_dir`` with a
    realistic shape (state-size grid rows, the primitive-appropriate
    us-column populated)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    # clone — uses ``us_per_call``
    _write_csv(
        bench_dir / "clone_20260101T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "us_per_call", "iters"],
        [
            ["(1, 1, 5)", 1, 1, 5, 0.21, 10000],
            ["(3, 3, 15)", 3, 3, 15, 0.22, 10000],
            ["(5, 7, 30)", 5, 7, 30, 0.23, 10000],
        ],
    )
    # apply_action — uses ``apply_us``
    _write_csv(
        bench_dir / "apply_action_20260101T000000Z.csv",
        [
            "label", "bf_size", "hand_size", "lib_size",
            "combined_us", "clone_us", "apply_us", "iters",
        ],
        [
            ["(1, 1, 5)", 1, 1, 5, 0.45, 0.21, 0.24, 10000],
            ["(3, 3, 15)", 3, 3, 15, 0.47, 0.22, 0.25, 10000],
            ["(5, 7, 30)", 5, 7, 30, 0.49, 0.23, 0.26, 10000],
        ],
    )
    # fingerprint — uses ``fp_us`` (forward-looking schema)
    _write_csv(
        bench_dir / "fingerprint_20260101T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "fp_us", "iters"],
        [
            ["(1, 1, 5)", 1, 1, 5, 1.5, 10000],
            ["(5, 7, 30)", 5, 7, 30, 3.2, 10000],
        ],
    )
    # legal_decisions — uses ``us_per_call``
    _write_csv(
        bench_dir / "legal_decisions_20260101T000000Z.csv",
        ["label", "us_per_call", "n_actions", "us_per_action"],
        [
            ["(1, 1, 5)", 0.8, 5, 0.16],
            ["(5, 7, 30)", 1.4, 12, 0.12],
        ],
    )


def test_runs_end_to_end_with_full_bench_dir(tmp_path):
    """All four primitives present -> script runs, prints all
    primitives, computes a non-zero wall-time estimate."""
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    _populate_full_bench_dir(bench_dir)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(["--bench-dir", str(bench_dir)])
    out = buf.getvalue()
    assert rc == 0

    # Expected primitive names appear in the printed table.
    for primitive in ("clone", "apply_action", "fingerprint", "legal_decisions"):
        assert primitive in out, (
            f"expected primitive {primitive!r} in output:\n{out}"
        )

    # Wall-time estimate header is present, total line is present
    # and parses to a positive number of seconds.
    assert "1000-game parity-smoke wall-time estimate" in out
    total_line = [
        line for line in out.splitlines() if line.strip().startswith("total:")
    ]
    assert total_line, f"missing total line in:\n{out}"
    # "total:    0.37 s  (~0.01 min)"
    seconds_str = total_line[0].split("total:")[1].strip().split()[0]
    total_seconds = float(seconds_str)
    assert total_seconds > 0.0
    # Sanity magnitude: the synthetic us numbers (~us/call) * the
    # CALLS_PER_GAME ratios * 1000 games should land in seconds-to-
    # minutes range, not microseconds or hours. With us values
    # 0.21--3.2 and total calls/game ~2100, that's roughly
    # 2100 * 1000 * ~1us = 2.1 s upper bound.
    assert total_seconds < 60.0, (
        f"wall-time estimate {total_seconds}s is implausibly large "
        f"for the test fixture"
    )


def test_missing_primitives_handled_gracefully(tmp_path):
    """When only some primitives have CSVs, the script doesn't crash
    and surfaces a 'missing primitives' note for the rest."""
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    bench_dir.mkdir()
    # Only clone present.
    _write_csv(
        bench_dir / "clone_20260101T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "us_per_call", "iters"],
        [["(5, 7, 30)", 5, 7, 30, 0.23, 10000]],
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(["--bench-dir", str(bench_dir)])
    out = buf.getvalue()
    assert rc == 0
    assert "clone" in out
    assert "missing primitives" in out
    for missing in ("apply_action", "fingerprint", "legal_decisions"):
        assert missing in out


def test_latest_csv_per_primitive_is_picked(tmp_path):
    """Two clone CSVs with different timestamps -> aggregator selects
    the latest by filename stamp."""
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    bench_dir.mkdir()
    _write_csv(
        bench_dir / "clone_20260101T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "us_per_call", "iters"],
        [["(5, 7, 30)", 5, 7, 30, 9.99, 10000]],  # "old, slow" sentinel
    )
    _write_csv(
        bench_dir / "clone_20260601T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "us_per_call", "iters"],
        [["(5, 7, 30)", 5, 7, 30, 0.21, 10000]],  # "new, fast"
    )

    rows, _missing = mod.build_summary(bench_dir)
    clone_rows = [r for r in rows if r.primitive == "clone"]
    assert len(clone_rows) == 1
    assert clone_rows[0].csv_filename == "clone_20260601T000000Z.csv"
    assert clone_rows[0].latest_us_per_call == 0.21


def test_state_size_filter(tmp_path):
    """``--state-size`` filters the table to the matching label."""
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    _populate_full_bench_dir(bench_dir)

    rows, _missing = mod.build_summary(
        bench_dir, state_size_filter="(5, 7, 30)"
    )
    assert rows  # at least one row survives
    for r in rows:
        assert r.state_size == "(5, 7, 30)", (
            f"filter leaked: {r.state_size!r}"
        )


def test_csv_output_mode(tmp_path):
    """``--csv PATH`` writes a well-formed summary CSV."""
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    _populate_full_bench_dir(bench_dir)
    out_csv = tmp_path / "summary.csv"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([
            "--bench-dir", str(bench_dir),
            "--csv", str(out_csv),
        ])
    assert rc == 0
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    assert out_rows
    expected_columns = {
        "primitive", "state_size", "latest_us_per_call", "csv_filename"
    }
    assert set(out_rows[0].keys()) >= expected_columns
    # At least one row per primitive in the unified summary.
    primitives_seen = {r["primitive"] for r in out_rows}
    assert primitives_seen >= {
        "clone", "apply_action", "fingerprint", "legal_decisions"
    }


def test_estimate_uses_representative_state_size(tmp_path):
    """The wall-time estimate prefers the representative state-size
    row when present — verify by constructing a clone CSV where the
    representative row has a clearly different us value than the
    other rows, and checking the math lines up.
    """
    mod = _load_module()
    bench_dir = tmp_path / "bench_results"
    bench_dir.mkdir()
    _write_csv(
        bench_dir / "clone_20260101T000000Z.csv",
        ["label", "bf_size", "hand_size", "lib_size", "us_per_call", "iters"],
        [
            ["(1, 1, 5)", 1, 1, 5, 99.0, 10000],
            ["(5, 7, 30)", 5, 7, 30, 1.0, 10000],
            ["(20, 30, 90)", 20, 30, 90, 99.0, 10000],
        ],
    )

    rows, _missing = mod.build_summary(bench_dir)
    total, per_primitive = mod.estimate_1000_game_wall_seconds(rows)
    # clone @ 1.0 us * 1500 calls/game * 1000 games = 1.5 s
    assert abs(per_primitive["clone"] - 1.5) < 1e-6, (
        f"expected clone contribution ~1.5s (representative row), "
        f"got {per_primitive['clone']}"
    )
    # Total equals clone here (other primitives missing).
    assert abs(total - 1.5) < 1e-6
