"""Tests for ``scripts/bench_rust_fingerprint_decomposition.py``.

The bench itself is a perf measurement, not a correctness check —
these tests cover that the script runs end-to-end and the CSV
schema matches what downstream tooling (and the next perf-summary
slice) will read.

Skipped when ``mtg_bot_engine`` isn't installed.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("mtg_bot_engine")


_BENCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_rust_fingerprint_decomposition.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_rust_fingerprint_decomposition", _BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(
        "bench_rust_fingerprint_decomposition", module
    )
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_bench()


def test_run_bench_writes_csv_with_expected_columns(bench, tmp_path):
    """End-to-end smoke: ``run_bench`` + ``write_csv`` produces a CSV
    under the given dir with one row per benchmarked state size and
    the full schema downstream tooling will read.

    Uses a tiny iteration count so the test stays fast — perf-quality
    numbers aren't the point here, well-formedness is. The
    ``fp_us`` and ``fp_minus_noop_us`` columns are the headline
    decomposition outputs; the others are the per-phase proxies
    that let a reader recover the breakdown.
    """
    rows = bench.run_bench(n=50)
    csv_path = bench.write_csv(rows, tmp_path)
    assert csv_path.exists()
    assert csv_path.parent == tmp_path
    assert csv_path.name.startswith("fingerprint_decomposition_")
    assert csv_path.name.endswith(".csv")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    assert len(out_rows) == len(bench.SIZES)

    expected_columns = {
        "label",
        "bf_size",
        "hand_size",
        "lib_size",
        "noop_us",
        "walk_us",
        "json_py_us",
        "fp_us",
        "fp_minus_noop_us",
        "iters",
    }
    for row in out_rows:
        assert set(row.keys()) == expected_columns, (
            f"unexpected columns: got {set(row.keys())}, "
            f"expected {expected_columns}"
        )
        # Every per-phase column should parse as a positive float
        # (timings are wall-clock, must be > 0).
        for col in ("noop_us", "walk_us", "json_py_us", "fp_us"):
            assert float(row[col]) > 0.0, (
                f"expected positive {col}, got {row[col]}"
            )
        # fp_minus_noop_us should be very close to fp_us - noop_us
        # (small float-precision drift OK).
        delta = float(row["fp_us"]) - float(row["noop_us"])
        assert abs(float(row["fp_minus_noop_us"]) - delta) < 1e-6
        assert int(row["iters"]) == 50


def test_python_fingerprint_helper_is_stable(bench):
    """The Python-side analog helper is the basis of ``json_py_us`` —
    it must return a stable 12-char hex digest so the timing loop
    isn't measuring a randomly-failing path.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    fp1 = bench._python_fingerprint(state)
    fp2 = bench._python_fingerprint(state)
    assert isinstance(fp1, str)
    assert len(fp1) == 12
    assert fp1 == fp2
