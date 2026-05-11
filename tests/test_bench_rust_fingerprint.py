"""Tests for ``scripts/bench_rust_fingerprint.py``.

The bench itself is a perf measurement, not a correctness check —
these tests cover (a) the bench script runs end-to-end without
exploding, (b) the perf numbers are sane enough to catch a
catastrophic regression, and (c) ``fingerprint()`` continues to
return a stable 12-char hex digest.

Skipped when ``mtg_bot_engine`` isn't installed.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import re
import sys

import pytest

pytest.importorskip("mtg_bot_engine")


# Load the bench module by path so the tests don't depend on
# ``scripts/`` being importable as a package.
_BENCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_rust_fingerprint.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_rust_fingerprint", _BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_rust_fingerprint", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_bench()


def test_bench_runs_on_smallest_size(bench):
    """Mini stress test: the bench's inner loop runs end-to-end on the
    smallest (empty) state without raising. Uses a tiny iteration count
    so the test stays fast.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(0, 0, 0)
    fp_ns = bench.time_fingerprint_ns(state, n=50)
    clone_ns = bench.time_clone_ns(state, n=50)
    assert fp_ns > 0
    assert clone_ns > 0


def test_bench_fingerprint_well_under_one_second_on_largest(bench):
    """Catastrophic perf regression guard: a single ``fingerprint()``
    call on the largest benchmarked size must be far under 1 second.
    The current measurement is ~180 us; a 5000x slowdown would still
    pass this assertion. The only thing it can't is a complete hang.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(50, 7, 50)
    # 100 iters keeps the test under a second even if perf cratered
    # by 50x; the threshold below is the real regression guard.
    fp_ns = bench.time_fingerprint_ns(state, n=100)
    per_call_seconds = fp_ns / 100 / 1e9
    assert per_call_seconds < 1.0, (
        f"fingerprint() is taking {per_call_seconds:.3f}s per call "
        f"on the (50, 7, 50) state — catastrophic regression"
    )


def test_fingerprint_is_at_least_2x_slower_than_clone_on_moderate(bench):
    """Sanity: at a moderate state size, ``fingerprint()`` is at
    least 2x slower than ``clone_once()``. Clone is a near-free Arc
    refcount bump; fingerprint walks every object + SHA1s the result.
    If this assertion ever flips, either (a) clone got dramatically
    more expensive (regression) or (b) fingerprint got dramatically
    cheaper (which would be surprising — flag for investigation).
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(5, 7, 30)
    # Generous iteration count so the ratio is stable.
    fp_ns = bench.time_fingerprint_ns(state, n=500)
    clone_ns = bench.time_clone_ns(state, n=500)
    assert clone_ns > 0
    ratio = fp_ns / clone_ns
    assert ratio >= 2.0, (
        f"expected fingerprint at least 2x slower than clone on a "
        f"(5, 7, 30) state, got ratio={ratio:.2f} "
        f"(fp_ns={fp_ns}, clone_ns={clone_ns})"
    )


def test_run_bench_writes_csv_with_distinct_sizes(bench, tmp_path):
    """End-to-end smoke: ``run_bench`` + ``write_csv`` writes a CSV
    under the given dir with a row per benchmarked state size, and
    the rows cover at least 3 distinct ``(bf_size, hand_size,
    lib_size)`` triples. Use a tiny iteration count so the test
    stays fast — perf-quality numbers aren't the point here, well-
    formedness is. The ``fp_us`` column is the one ``perf_summary.py``
    reads as the fingerprint per-call cost, so verify it parses as a
    positive float.
    """
    rows = bench.run_bench(n=50)
    csv_path = bench.write_csv(rows, tmp_path)
    assert csv_path.exists()
    assert csv_path.parent == tmp_path
    assert csv_path.name.startswith(
        "fingerprint_"
    ) and csv_path.name.endswith(".csv")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    # One row per benchmarked size.
    assert len(out_rows) == len(bench.SIZES)
    # Required columns are present and populated. ``fp_us`` is the
    # column ``perf_summary.py`` reads; the others mirror the sibling
    # bench CSVs for cross-reference.
    for row in out_rows:
        assert set(row.keys()) >= {
            "label",
            "bf_size",
            "hand_size",
            "lib_size",
            "fp_us",
            "clone_us",
            "ratio",
            "iters",
        }
        assert float(row["fp_us"]) > 0.0
        assert float(row["clone_us"]) > 0.0
        assert int(row["iters"]) == 50

    distinct_triples = {
        (row["bf_size"], row["hand_size"], row["lib_size"])
        for row in out_rows
    }
    assert len(distinct_triples) >= 3, (
        f"expected at least 3 distinct state sizes, got {distinct_triples}"
    )


def test_fingerprint_returns_stable_12_char_hex():
    """The bench's whole premise is that ``fingerprint()`` returns a
    stable 12-char hex digest. Mirror the format check from
    ``tests_rust/test_fingerprint.py`` so this test file is
    self-contained as a regression guard for the bench's input.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    fp1 = state.fingerprint()
    fp2 = state.fingerprint()
    assert isinstance(fp1, str)
    assert len(fp1) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", fp1), fp1
    assert fp1 == fp2
