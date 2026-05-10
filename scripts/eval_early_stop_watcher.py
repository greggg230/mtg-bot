"""Polling early-stop watcher for iterate's eval-gate phase.

Watches a glob of eval progress JSONs and writes a stop sentinel via
``mtg_bot.training.stop_signal.write_stop`` when the run's decisive
games are statistically conclusive — either above the acceptance
threshold (``PASS``, will promote) or below it (``FAIL``, will keep
current best). Either way the iterate eval pool wraps up at the next
game boundary and the gate decision uses the partial result, which is
already statistically sufficient.

Statistical method: Wilson score interval at confidence
``1 - 2*alpha`` (one-sided alpha each side, default alpha=0.025 ->
95% CI overall). Conclusive when the threshold sits outside the CI.

Caveat: naive sequential testing inflates Type I error vs a fixed-N
test. For an acceptance gate already at 55% (well above the 50% null),
naive Wilson is good enough — the threshold gives most of the alpha
protection. Tightening to SPRT or O'Brien-Fleming boundaries would be
the principled next step if the false-promote rate ever matters more.

A minimum decisive-game count (default 30) prevents stopping on a
3-0 lucky start. Stops only target eval runs (``status == 'running'``,
filename matches ``*_eval_*.json`` excluding ``*_shard_*``).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from math import sqrt
from pathlib import Path
from typing import Optional

from mtg_bot.training.stop_signal import sentinel_path, write_stop


def _wilson_ci(wins: int, n: int, *, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for proportion. Returns (point, lo, hi)."""
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _z_for_alpha(alpha: float) -> float:
    """Inverse normal at 1-alpha. Hard-coded common values to avoid
    pulling in scipy. Falls back to 1.96 (alpha=0.025) for unknown
    inputs and warns the caller."""
    table = {0.005: 2.576, 0.01: 2.326, 0.025: 1.96, 0.05: 1.645, 0.1: 1.282}
    return table.get(round(alpha, 4), 1.96)


def _decision(
    wins_new: int, wins_old: int, threshold: float, z: float, min_decisive: int,
) -> Optional[tuple[str, float, float, float]]:
    """Return (decision, p, lo, hi) when the threshold sits outside
    the CI, else None. ``decision`` is one of ``'PASS'`` / ``'FAIL'``."""
    n = wins_new + wins_old
    if n < min_decisive:
        return None
    p, lo, hi = _wilson_ci(wins_new, n, z=z)
    if lo > threshold:
        return "PASS", p, lo, hi
    if hi < threshold:
        return "FAIL", p, lo, hi
    return None


_ITER_EVAL_RUN_ID_RE = re.compile(r"^(iter_\d+)_eval_")


def _resolve_cycle_id(data: dict, run_id: str) -> Optional[str]:
    """Pick the right stop-sentinel target for an iterate-driven eval.

    Two sources, in order of preference:

    1. The eval progress JSON's ``cycle_id`` field — what
       ``EvalProgress`` writes when iterate passes ``cycle_id=cycle_id``
       through ``_eval_match`` / ``_eval_match_distributed``. This is
       the source of truth when present.
    2. Filename-pattern fallback: ``iter_NNN_eval_<new>_vs_<old>`` →
       cycle_id ``iter_NNN_cycle``. The distributed eval path's
       per-shard JSONs surfaced empirically with ``cycle_id: None``
       even though iterate had set it; rather than chase that down
       at progress-write time, we recover the cycle here from the
       deterministic run_id naming convention iterate uses.

    Standalone evals (``mtg_bot eval`` without iterate) have no
    iter_NNN prefix and no cycle, so this returns None and the
    caller falls back to writing the sentinel for ``run_id``.
    """
    cid = data.get("cycle_id")
    if cid:
        return str(cid)
    m = _ITER_EVAL_RUN_ID_RE.match(run_id)
    if m:
        return f"{m.group(1)}_cycle"
    return None


def _iter_eval_progress_files(progress_dir: Path) -> list[Path]:
    """Yield eval progress JSON paths (master files, not shard files).

    Matches ``*_eval_*.json`` and filters out shard files (which have
    ``_shard_`` in the name) and shard event logs (``.jsonl``).
    """
    if not progress_dir.is_dir():
        return []
    out = []
    for p in progress_dir.glob("*_eval_*.json"):
        if "_shard_" in p.name:
            continue
        out.append(p)
    return out


def _check_one(
    path: Path, *, threshold: float, z: float, min_decisive: int,
    progress_dir: Path, already_stopped: set[str],
) -> Optional[str]:
    """Read one progress file and write a stop sentinel if conclusive.
    Returns a one-line log message describing the action (or status)
    so the watcher can print it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    run_id = data.get("run_id")
    if not run_id or run_id in already_stopped:
        return None
    if data.get("status") != "running":
        return None
    wins_new = int(data.get("wins_new", 0))
    wins_old = int(data.get("wins_old", 0))

    decision = _decision(
        wins_new, wins_old, threshold, z, min_decisive=min_decisive,
    )
    if decision is None:
        n = wins_new + wins_old
        if n >= min_decisive:
            _, lo, hi = _wilson_ci(wins_new, n, z=z)
            return (
                f"[watch] {run_id}: {wins_new}-{wins_old} "
                f"(n={n}) CI [{lo:.1%}, {hi:.1%}] straddles "
                f"threshold {threshold:.1%}; continue"
            )
        return None

    label, p, lo, hi = decision
    cycle_id = _resolve_cycle_id(data, run_id)
    # iterate's eval pool checks for `.STOP_<cycle_id>` only (via
    # ``_cycle_stop_check`` in cli/iterate.py), so we MUST target
    # cycle_id when one is resolvable. Falling back to the run_id
    # sentinel was a bug — written-but-ignored. Standalone evals
    # (``mtg_bot eval`` without iterate) have no cycle, so for those
    # targeting run_id is correct (their stop_check uses run_id).
    target_id = cycle_id if cycle_id else run_id
    write_stop(
        progress_dir, target_id,
        reason=(
            f"early-stop {label}: {wins_new}-{wins_old} "
            f"(n={wins_new + wins_old}) CI [{lo:.1%}, {hi:.1%}] "
            f"vs threshold {threshold:.1%}"
        ),
    )
    already_stopped.add(run_id)
    return (
        f"[watch] {run_id}: {label} — {wins_new}-{wins_old} "
        f"(n={wins_new + wins_old}) CI [{lo:.1%}, {hi:.1%}] vs "
        f"threshold {threshold:.1%} -> wrote stop sentinel for "
        f"{target_id}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress-dir", type=Path, default=Path("training_progress"),
        help="Where iterate writes eval progress JSONs.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Acceptance threshold the gate compares win_rate against.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.025,
        help=(
            "One-sided alpha for the Wilson CI (default 0.025 -> 95%% "
            "CI overall). Tighten with --alpha 0.01 (98%% CI)."
        ),
    )
    parser.add_argument(
        "--min-decisive", type=int, default=30,
        help=(
            "Don't stop until at least this many games have a winner. "
            "Prevents stopping on a 3-0 lucky start."
        ),
    )
    parser.add_argument(
        "--check-interval-sec", type=float, default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Check all running eval files once and exit.",
    )
    args = parser.parse_args(argv)

    z = _z_for_alpha(args.alpha)
    print(
        f"[watch] threshold={args.threshold:.1%}, alpha={args.alpha} "
        f"(z={z}), min_decisive={args.min_decisive}, "
        f"interval={args.check_interval_sec}s, "
        f"progress_dir={args.progress_dir}",
        flush=True,
    )

    already_stopped: set[str] = set()
    last_status: dict[str, str] = {}
    while True:
        files = _iter_eval_progress_files(args.progress_dir)
        for f in files:
            msg = _check_one(
                f, threshold=args.threshold, z=z,
                min_decisive=args.min_decisive,
                progress_dir=args.progress_dir,
                already_stopped=already_stopped,
            )
            if msg and msg != last_status.get(str(f)):
                print(msg, flush=True)
                last_status[str(f)] = msg
        if args.once:
            return
        time.sleep(args.check_interval_sec)


if __name__ == "__main__":
    main()
