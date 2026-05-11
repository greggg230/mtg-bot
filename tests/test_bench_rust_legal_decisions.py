"""Tests for ``scripts/bench_rust_legal_decisions.py``.

The bench itself is a perf measurement, not a correctness check —
these tests cover (a) the bench script's inner loop runs end-to-end
without exploding on the smallest state, (b) the perf numbers are
sane enough to catch a catastrophic regression on the largest state,
(c) ``legal_decisions()`` always emits at least one action
(PassPriority is always available), and (d) the emitted list is
deterministic for a fixed state.

Skipped when ``mtg_bot_engine`` isn't installed.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("mtg_bot_engine")


# Load the bench module by path so the tests don't depend on
# ``scripts/`` being importable as a package.
_BENCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_rust_legal_decisions.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_rust_legal_decisions", _BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_rust_legal_decisions", module)
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
    total_ns = bench.time_legal_decisions_ns(state, n=50)
    assert total_ns > 0


def test_bench_well_under_one_second_per_call_on_largest(bench):
    """Catastrophic perf regression guard: a single
    ``legal_decisions()`` call on the largest benchmarked size must
    be far under 1 second. A 5000x slowdown from the current
    measurement would still pass this assertion. The only thing it
    can't is a complete hang.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(50, 7, 50)
    # 100 iters keeps the test under a second even if perf cratered
    # by 50x; the threshold below is the real regression guard.
    total_ns = bench.time_legal_decisions_ns(state, n=100)
    per_call_seconds = total_ns / 100 / 1e9
    assert per_call_seconds < 1.0, (
        f"legal_decisions() is taking {per_call_seconds:.3f}s per call "
        f"on the (50, 7, 50) state — catastrophic regression"
    )


def test_legal_decisions_emits_at_least_one_action_at_every_size():
    """``PassPriority`` is always a legal decision when the queried
    player has priority, so every size in ``SIZES`` must emit at
    least one entry. A zero-length list at any size would mean
    either the synthetic builder broke or the enumerator regressed
    below the always-on baseline.
    """
    import mtg_bot_engine

    bench = _load_bench()
    for bf, hand, lib, label in bench.SIZES:
        state = mtg_bot_engine.make_synthetic_state(bf, hand, lib)
        actions = state.legal_decisions(bench.PLAYER_ID)
        assert len(actions) >= 1, (
            f"legal_decisions returned empty list at size {label}"
        )


def test_run_bench_writes_csv_with_distinct_sizes(bench, tmp_path):
    """End-to-end smoke: ``run_bench`` + ``write_csv`` writes a CSV
    under the given dir with a row per benchmarked state size, and
    the rows cover at least 3 distinct ``(bf_size, hand_size,
    lib_size)`` triples. Use a tiny iteration count so the test
    stays fast — perf-quality numbers aren't the point here, well-
    formedness is. The ``us_per_call`` column is the one
    ``perf_summary.py`` reads as the legal_decisions per-call cost,
    so verify it parses as a positive float.
    """
    rows = bench.run_bench(n=50)
    csv_path = bench.write_csv(rows, tmp_path)
    assert csv_path.exists()
    assert csv_path.parent == tmp_path
    assert csv_path.name.startswith(
        "legal_decisions_"
    ) and csv_path.name.endswith(".csv")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    # One row per benchmarked size.
    assert len(out_rows) == len(bench.SIZES)
    # Required columns are present and populated. ``us_per_call`` is
    # the column ``perf_summary.py`` reads; the others mirror the
    # sibling bench CSVs for cross-reference.
    for row in out_rows:
        assert set(row.keys()) >= {
            "label",
            "bf_size",
            "hand_size",
            "lib_size",
            "us_per_call",
            "n_actions",
            "us_per_action",
            "iters",
        }
        assert float(row["us_per_call"]) > 0.0
        assert int(row["n_actions"]) >= 1
        assert int(row["iters"]) == 50

    distinct_triples = {
        (row["bf_size"], row["hand_size"], row["lib_size"])
        for row in out_rows
    }
    assert len(distinct_triples) >= 3, (
        f"expected at least 3 distinct state sizes, got {distinct_triples}"
    )


def test_legal_decisions_is_deterministic():
    """Same state → same list across repeated calls. The Rust
    enumerator builds the list fresh each call (no shared state),
    so two back-to-back invocations should produce structurally
    identical output.
    """
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    first = state.legal_decisions(0)
    second = state.legal_decisions(0)
    assert first == second
