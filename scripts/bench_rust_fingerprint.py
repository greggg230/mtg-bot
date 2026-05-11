"""Per-state-size perf bench for the Rust ``GameState.fingerprint()``.

Complements ``scripts/bench_rust_clone.py`` (the sibling per-state-
size clone bench). ``fingerprint()`` runs on every step in
``scripts/parity_smoke.py --mode trajectory``, so its per-call cost
as state size grows is load-bearing: clone is an Arc refcount bump
(essentially free), but fingerprint walks every object in every
zone + computes a SHA1 digest. This bench measures the magnitude of
that difference across a handful of representative state shapes.

Run from a worktree:
    PYTHONPATH=src .venv-py313/Scripts/python.exe \\
        scripts/bench_rust_fingerprint.py

Writes a CSV to ``bench_results/fingerprint_<UTC-timestamp>.csv`` with
one row per benchmarked state size; columns: ``label``, ``bf_size``,
``hand_size``, ``lib_size``, ``fp_us``, ``clone_us``, ``ratio``,
``iters``. ``perf_summary.py`` reads the ``fp_us`` column as the
fingerprint per-call cost. Skips with a hint if ``mtg_bot_engine``
isn't installed yet (run ``maturin develop --release -m
rust/Cargo.toml``).
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


def time_fingerprint_ns(state, n: int) -> float:
    """Total wall-clock nanoseconds for ``n`` ``fingerprint()`` calls."""
    # Warmup — exclude first-touch costs (any lazy caches, code-path
    # warmup). We're benchmarking the steady-state cost.
    for _ in range(WARMUP_ITERS):
        state.fingerprint()
    start = time.perf_counter_ns()
    for _ in range(n):
        state.fingerprint()
    return time.perf_counter_ns() - start


def time_clone_ns(state, n: int) -> float:
    """Total wall-clock nanoseconds for ``n`` ``clone_once()`` calls.

    Note: ``clone_n_times_micros`` runs the loop in Rust without PyO3
    round-trip costs, which makes it a tighter measurement of pure
    Rust clone cost. Here we use ``clone_once()`` so both clone and
    fingerprint pay the same Python→Rust call overhead — the ratio
    reflects what a Python caller (parity_smoke) actually pays per
    step. ``bench_rust_clone.py`` makes the same choice.
    """
    for _ in range(WARMUP_ITERS):
        state.clone_once()
    start = time.perf_counter_ns()
    for _ in range(n):
        state.clone_once()
    return time.perf_counter_ns() - start


def run_bench(n: int = DEFAULT_ITERS) -> list[dict]:
    """Run the bench and return a list of per-size result rows."""
    if mtg_bot_engine is None:
        return []

    rows: list[dict] = []
    for bf, hand, lib, label in SIZES:
        state = mtg_bot_engine.make_synthetic_state(bf, hand, lib)
        fp_total_ns = time_fingerprint_ns(state, n)
        clone_total_ns = time_clone_ns(state, n)
        fp_us = fp_total_ns / n / 1_000.0
        clone_us = clone_total_ns / n / 1_000.0
        ratio = fp_us / clone_us if clone_us > 0 else float("inf")
        rows.append(
            {
                "label": label,
                "bf_size": bf,
                "hand_size": hand,
                "lib_size": lib,
                "fp_us": fp_us,
                "clone_us": clone_us,
                "ratio": ratio,
                "iters": n,
            }
        )
    return rows


def print_table(rows: list[dict], n: int) -> None:
    """Print a small results table to stdout."""
    print(f"fingerprint perf — N={n} iters per size")
    print()
    print("  size            | fingerprint us/call | clone us/call | ratio")
    print("  ----------------|---------------------|---------------|------")
    for r in rows:
        print(
            f"  {r['label']:<15} |        {r['fp_us']:7.2f} us   |"
            f"  {r['clone_us']:6.2f} us    | {r['ratio']:6.1f}x"
        )


def write_csv(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    """Write rows as a CSV under ``out_dir``; return the path written.

    Filename: ``fingerprint_<UTC-timestamp>.csv``. The dir is created
    if it doesn't exist. Columns: ``label``, ``bf_size``, ``hand_size``,
    ``lib_size``, ``fp_us``, ``clone_us``, ``ratio``, ``iters`` — one
    row per benchmarked state size. ``perf_summary.py`` reads the
    ``fp_us`` column as the fingerprint per-call cost; the other
    columns mirror the sibling bench CSVs for cross-reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"fingerprint_{stamp}.csv"
    fieldnames = [
        "label",
        "bf_size",
        "hand_size",
        "lib_size",
        "fp_us",
        "clone_us",
        "ratio",
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
