"""Per-state-size perf bench for the Rust ``GameState.legal_decisions()``.

The parity_smoke trajectory walker (``scripts/parity_smoke.py --mode
trajectory``) calls ``state.legal_decisions(player_id)`` on every step
to pick the next move. The enumerator emits PassPriority + PlayLand +
CastSpell + ActivateAbility (under sorcery timing) + DeclareAttackers
/ DeclareBlockers (during combat phases); its cost scales with hand
size (for PlayLand / CastSpell scans) and battlefield size (for
ActivateAbility / DeclareAttackers scans). Knowing the per-call cost
across representative state shapes gives us a concrete number for
trajectory throughput.

Run from a worktree:
    PYTHONPATH=src .venv-py313/Scripts/python.exe \\
        scripts/bench_rust_legal_decisions.py

Writes a CSV to ``bench_results/legal_decisions_<UTC-timestamp>.csv``
with one row per benchmarked state size; columns: ``label``,
``bf_size``, ``hand_size``, ``lib_size``, ``us_per_call``,
``n_actions``, ``us_per_action``, ``iters``. ``perf_summary.py``
reads the ``us_per_call`` column as the legal_decisions per-call cost.
Skips with a hint if ``mtg_bot_engine`` isn't installed yet (run
``maturin develop --release -m rust/Cargo.toml``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import pathlib
import time

try:
    import mtg_bot_engine
except ImportError:
    mtg_bot_engine = None


# (bf_per_side, hand_per_side, lib_per_side, label)
SIZES = [
    (0, 0, 0, "(0, 0, 0)"),
    (2, 3, 5, "(2, 3, 5)"),
    (5, 7, 30, "(5, 7, 30)"),
    (20, 7, 50, "(20, 7, 50)"),
    (50, 7, 50, "(50, 7, 50)"),
]

DEFAULT_ITERS = 10_000
WARMUP_ITERS = 200
PLAYER_ID = 0


def time_legal_decisions_ns(state, n: int) -> float:
    """Total wall-clock nanoseconds for ``n`` ``legal_decisions()`` calls."""
    # Warmup — exclude first-touch costs (any lazy caches, code-path
    # warmup). We're benchmarking the steady-state cost.
    for _ in range(WARMUP_ITERS):
        state.legal_decisions(PLAYER_ID)
    start = time.perf_counter_ns()
    for _ in range(n):
        state.legal_decisions(PLAYER_ID)
    return time.perf_counter_ns() - start


def run_bench(n: int = DEFAULT_ITERS) -> list[dict]:
    """Run the bench and return a list of per-size result rows."""
    if mtg_bot_engine is None:
        return []

    rows: list[dict] = []
    for bf, hand, lib, label in SIZES:
        state = mtg_bot_engine.make_synthetic_state(bf, hand, lib)
        n_actions = len(state.legal_decisions(PLAYER_ID))
        total_ns = time_legal_decisions_ns(state, n)
        us_per_call = total_ns / n / 1_000.0
        us_per_action = (
            us_per_call / n_actions if n_actions > 0 else float("inf")
        )
        rows.append(
            {
                "label": label,
                "bf_size": bf,
                "hand_size": hand,
                "lib_size": lib,
                "us_per_call": us_per_call,
                "n_actions": n_actions,
                "us_per_action": us_per_action,
                "iters": n,
            }
        )
    return rows


def print_table(rows: list[dict], n: int) -> None:
    """Print a small results table to stdout."""
    print(
        f"legal_decisions perf — N={n} iters per size, "
        f"player_id={PLAYER_ID}"
    )
    print()
    print(
        "  size            | legal_dec us/call | n_actions | us per action"
    )
    print(
        "  ----------------|-------------------|-----------|---------------"
    )
    for r in rows:
        print(
            f"  {r['label']:<15} |       {r['us_per_call']:8.2f} us  |"
            f"   {r['n_actions']:5d}   |    {r['us_per_action']:7.2f} us"
        )


def write_csv(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    """Write rows as a CSV under ``out_dir``; return the path written.

    Filename: ``legal_decisions_<UTC-timestamp>.csv``. The dir is
    created if it doesn't exist. Columns: ``label``, ``bf_size``,
    ``hand_size``, ``lib_size``, ``us_per_call``, ``n_actions``,
    ``us_per_action``, ``iters`` — one row per benchmarked state
    size. ``perf_summary.py`` reads the ``us_per_call`` column as the
    legal_decisions per-call cost; the other columns mirror the
    sibling bench CSVs for cross-reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"legal_decisions_{stamp}.csv"
    fieldnames = [
        "label",
        "bf_size",
        "hand_size",
        "lib_size",
        "us_per_call",
        "n_actions",
        "us_per_action",
        "iters",
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
    rows = run_bench(DEFAULT_ITERS)
    print_table(rows, DEFAULT_ITERS)
    # Default output dir is repo-relative ``bench_results/`` so the
    # bench writes alongside the other bench artifacts in this dir.
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "bench_results"
    csv_path = write_csv(rows, out_dir)
    print()
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
