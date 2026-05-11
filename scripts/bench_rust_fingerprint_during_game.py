"""Per-decision-step perf bench for the Rust ``GameState.fingerprint()``.

Complements ``scripts/bench_rust_fingerprint.py`` (per-state-size, static
states) by measuring how ``fingerprint()`` cost evolves as state mutates
through a TRAJECTORY of decisions. The static-size bench answers
"how expensive is fingerprint at size X" — this bench answers
"how does fingerprint cost change as a single game plays out".

This is useful for the eventual 1000-game smoke wall-time estimate:
the static grid samples three points; a trajectory walk samples every
intermediate state and surfaces growth patterns (e.g. fingerprint cost
creeping upward as the graveyard or stack accumulates entries during
a long game).

We measure cost at each decision index by:

1. Build a Rust synthetic state at a representative size (5, 7, 30).
2. At each step:
   a. Time the median of K ``fingerprint()`` calls on the current state.
   b. Read ``legal_decisions(priority_player)``, pick the first non-
      PassPriority action if any, else PassPriority.
   c. Apply the chosen action.
3. After N decisions, also measure the post-trajectory state.

Run from a worktree:
    PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe \\
        scripts/bench_rust_fingerprint_during_game.py

Writes a CSV to ``bench_results/fingerprint_during_game_<UTC-timestamp>.csv``
with one row per checkpoint (initial state + after each decision);
columns: ``decision_index`` (-1 for pre-first-decision baseline, 0..N-1
for after-step-i), ``action_kind`` (kind of action applied to reach this
state, ``"initial"`` for the baseline row), ``fingerprint_us``,
``state_size_after`` (sum of all zone sizes across both players).

Skips with a hint if ``mtg_bot_engine`` isn't installed yet (run
``maturin develop --release -m rust/Cargo.toml``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import pathlib
import statistics
import time

try:
    import mtg_bot_engine
except ImportError:
    mtg_bot_engine = None


# Representative state size — matches the middle row of
# ``bench_rust_fingerprint.py`` SIZES (the "typical mid-game" point).
DEFAULT_SIZE = (5, 7, 30)

DEFAULT_NUM_DECISIONS = 50
# Number of fingerprint calls timed at each checkpoint. We take the
# median to ride over OS-scheduling jitter; K=200 is well over the
# noise floor for a sub-microsecond primitive on modern hardware.
DEFAULT_CALLS_PER_CHECKPOINT = 200
WARMUP_ITERS = 50


def _state_size(state) -> int:
    """Sum of all zone sizes across both players.

    Approximate ``state_size_after`` proxy — bigger zones generally
    mean more bytes to walk in fingerprint. Includes battlefield,
    hand, library, graveyard, exile. (Stack isn't a per-player list,
    so it's omitted; under PassPriority-only trajectories the stack
    stays empty anyway.)
    """
    total = 0
    for zone_sizes in (
        state.battlefield_sizes,
        state.hand_sizes,
        state.library_sizes,
        state.graveyard_sizes,
        state.exile_sizes,
    ):
        total += sum(zone_sizes)
    return total


def _median_fingerprint_us(state, k: int) -> float:
    """Median per-call ``fingerprint()`` cost in microseconds across
    ``k`` timed calls.

    Median (not mean) so a single descheduling blip on one call
    doesn't poison the checkpoint reading. Warmup runs first to
    exclude first-touch costs.
    """
    for _ in range(WARMUP_ITERS):
        state.fingerprint()
    samples_ns: list[int] = []
    for _ in range(k):
        start = time.perf_counter_ns()
        state.fingerprint()
        samples_ns.append(time.perf_counter_ns() - start)
    return statistics.median(samples_ns) / 1_000.0


def _pick_action(legal: list[dict]) -> dict | None:
    """Pick the first non-``pass_priority`` action if any, else the
    first action (which is conventionally ``pass_priority`` per the
    enumerator's emission order).

    Returns ``None`` if ``legal`` is empty — caller treats that as
    a hard stop on the trajectory walk.
    """
    if not legal:
        return None
    for d in legal:
        if d.get("kind") != "pass_priority":
            return d
    return legal[0]


def run_bench(
    size: tuple[int, int, int] = DEFAULT_SIZE,
    num_decisions: int = DEFAULT_NUM_DECISIONS,
    calls_per_checkpoint: int = DEFAULT_CALLS_PER_CHECKPOINT,
) -> list[dict]:
    """Walk a synthetic state through ``num_decisions`` and return
    per-checkpoint timing rows.

    Returns ``num_decisions + 1`` rows: index ``-1`` is the pre-walk
    baseline (initial synthetic state), indices ``0..num_decisions - 1``
    are post-decision-step measurements.
    """
    if mtg_bot_engine is None:
        return []

    bf, hand, lib = size
    state = mtg_bot_engine.make_synthetic_state(bf, hand, lib)

    rows: list[dict] = []
    rows.append(
        {
            "decision_index": -1,
            "action_kind": "initial",
            "fingerprint_us": _median_fingerprint_us(state, calls_per_checkpoint),
            "state_size_after": _state_size(state),
        }
    )

    for i in range(num_decisions):
        legal = state.legal_decisions(state.get_priority_player())
        action = _pick_action(legal)
        if action is None:
            # Empty legal_decisions — contract violation in normal
            # operation, but defensively stop walking rather than
            # raise (bench is informational, not a gate).
            break
        try:
            state.apply_action(action)
        except ValueError:
            # Stubbed dispatch (e.g. ActivateAbility per
            # parity_smoke's notes) — pass_priority is always legal
            # and applies cleanly, so fall back to it.
            state.apply_action({"kind": "pass_priority"})
            kind_applied = "pass_priority"
        else:
            kind_applied = action.get("kind", "?")

        rows.append(
            {
                "decision_index": i,
                "action_kind": kind_applied,
                "fingerprint_us": _median_fingerprint_us(state, calls_per_checkpoint),
                "state_size_after": _state_size(state),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Compute summary stats across the trajectory (median, min, max,
    std of per-checkpoint ``fingerprint_us``).

    Drops the index ``-1`` row from the summary so the numbers reflect
    cost across the actual walked decisions — the initial-state
    measurement is reported separately as a baseline.
    """
    if not rows:
        return {
            "n_checkpoints": 0,
            "median_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
            "std_us": 0.0,
        }
    samples = [r["fingerprint_us"] for r in rows if r["decision_index"] >= 0]
    if not samples:
        # Only the baseline row exists — degenerate, but emit
        # consistent fields so callers don't branch.
        baseline = rows[0]["fingerprint_us"]
        return {
            "n_checkpoints": 1,
            "median_us": baseline,
            "min_us": baseline,
            "max_us": baseline,
            "std_us": 0.0,
        }
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "n_checkpoints": len(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "std_us": std,
    }


def print_table(rows: list[dict], summary: dict, size: tuple[int, int, int]) -> None:
    """Print a per-checkpoint table + a summary footer."""
    print(
        f"fingerprint during game — state_size={size}, "
        f"n_decisions={len(rows) - 1}, K calls/checkpoint median"
    )
    print()
    print("  step | action_kind          | fingerprint us | state_size_after")
    print("  -----|----------------------|----------------|-----------------")
    for r in rows:
        step_label = "init" if r["decision_index"] == -1 else f"{r['decision_index']:>4d}"
        print(
            f"  {step_label} | {r['action_kind']:<20} |"
            f"     {r['fingerprint_us']:7.3f} us  |"
            f"      {r['state_size_after']:>5d}"
        )
    print()
    print(
        f"summary (post-decision checkpoints, n={summary['n_checkpoints']}): "
        f"median={summary['median_us']:.3f}us  "
        f"min={summary['min_us']:.3f}us  "
        f"max={summary['max_us']:.3f}us  "
        f"std={summary['std_us']:.3f}us"
    )


def write_csv(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    """Write rows as a CSV under ``out_dir``; return the path written.

    Filename: ``fingerprint_during_game_<UTC-timestamp>.csv``. The dir
    is created if it doesn't exist. Columns: ``decision_index``,
    ``action_kind``, ``fingerprint_us``, ``state_size_after`` — one row
    per checkpoint (initial + after each decision step).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"fingerprint_during_game_{stamp}.csv"
    fieldnames = [
        "decision_index",
        "action_kind",
        "fingerprint_us",
        "state_size_after",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path


def main() -> None:
    if mtg_bot_engine is None:
        print(
            "rust engine: not installed — run "
            "`maturin develop --release -m rust/Cargo.toml`"
        )
        return

    print(f"Rust engine version: {mtg_bot_engine.__version__}")
    print()
    rows = run_bench(
        size=DEFAULT_SIZE,
        num_decisions=DEFAULT_NUM_DECISIONS,
        calls_per_checkpoint=DEFAULT_CALLS_PER_CHECKPOINT,
    )
    summary = summarize(rows)
    print_table(rows, summary, DEFAULT_SIZE)
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "bench_results"
    csv_path = write_csv(rows, out_dir)
    print()
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
