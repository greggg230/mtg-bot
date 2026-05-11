"""Tests for ``scripts/bench_rust_fingerprint_during_game.py``.

The bench itself is a perf measurement, not a correctness check —
these tests cover (a) ``run_bench`` walks the trajectory end-to-end
without raising and emits one checkpoint per decision (plus the
initial-state baseline), and (b) ``run_bench`` + ``write_csv``
produce a well-formed CSV with the expected columns + at least
``N + 1`` rows.

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
    / "bench_rust_fingerprint_during_game.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_rust_fingerprint_during_game", _BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_rust_fingerprint_during_game", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_bench()


def test_run_bench_walks_trajectory_end_to_end(bench):
    """Smoke: ``run_bench`` walks a tiny trajectory without raising
    and emits one row per decision step plus the initial-state row.

    Uses small ``num_decisions`` and ``calls_per_checkpoint`` so the
    test stays fast — perf-quality numbers aren't the point here,
    end-to-end well-formedness is.
    """
    n_decisions = 5
    rows = bench.run_bench(
        size=(2, 3, 5),
        num_decisions=n_decisions,
        calls_per_checkpoint=20,
    )
    # Initial-state baseline + one row per decision.
    assert len(rows) == n_decisions + 1
    # Index -1 is the baseline; 0..N-1 are post-decision rows.
    assert rows[0]["decision_index"] == -1
    assert rows[0]["action_kind"] == "initial"
    for i, r in enumerate(rows[1:]):
        assert r["decision_index"] == i
        # action_kind comes from the actual applied action.
        assert isinstance(r["action_kind"], str) and r["action_kind"]
    # Every checkpoint should have a positive fingerprint timing and
    # a non-negative state size.
    for r in rows:
        assert r["fingerprint_us"] > 0.0
        assert r["state_size_after"] >= 0


def test_run_bench_writes_csv_with_expected_columns(bench, tmp_path):
    """End-to-end: ``run_bench`` + ``write_csv`` writes a CSV with the
    documented columns and at least ``N + 1`` rows (one per
    checkpoint).
    """
    n_decisions = 4
    rows = bench.run_bench(
        size=(2, 3, 5),
        num_decisions=n_decisions,
        calls_per_checkpoint=20,
    )
    csv_path = bench.write_csv(rows, tmp_path)
    assert csv_path.exists()
    assert csv_path.parent == tmp_path
    assert csv_path.name.startswith(
        "fingerprint_during_game_"
    ) and csv_path.name.endswith(".csv")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    assert len(out_rows) >= n_decisions + 1
    for row in out_rows:
        assert set(row.keys()) >= {
            "decision_index",
            "action_kind",
            "fingerprint_us",
            "state_size_after",
        }
        # decision_index round-trips through CSV as a string; the
        # baseline row is ``"-1"`` and post-decision rows are
        # non-negative integers.
        assert int(row["decision_index"]) >= -1
        # fingerprint_us must parse as a positive float.
        assert float(row["fingerprint_us"]) > 0.0
        # state_size_after must parse as a non-negative int.
        assert int(row["state_size_after"]) >= 0
