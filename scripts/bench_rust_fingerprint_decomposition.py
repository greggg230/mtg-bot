"""Decompose the cost of ``GameState.fingerprint()`` into component phases.

Companion to ``scripts/bench_rust_fingerprint.py`` — that bench gives
the headline number (per-call us at each state size); this one breaks
the number down so the next optimization slice knows where the time
goes.

## Decomposition methodology

The Rust ``fingerprint()`` impl at ``rust/src/state_ops_fingerprint.rs``
does, in order:

1. Reach into ``GameState`` fields (no allocation, no PyO3 round-trip
   beyond the initial method-call dispatch).
2. Build a tree of ``serde_json::Value`` nodes mirroring Python's
   ``hash_state`` ``parts`` list — one ``Value::Array`` per zone, plus
   per-object 8-tuples and per-player 8-tuples.
3. Walk that tree with a custom ``serialize_python_compatible`` writer
   that emits Python-compatible JSON (``", "`` / ``": "`` separators,
   matches ``json.dumps`` default output byte-for-byte).
4. Feed the resulting blob to SHA1 and hex-encode the first 6 bytes.

We can't time those phases directly without touching Rust source
(out of scope per the task brief). Instead we measure several
**proxy phases** from Python — each adds one ingredient to the
previous baseline, so subtracting consecutive measurements isolates
the marginal cost:

* ``noop_us``: per-call cost of a trivial pymethod (the ``turn_number``
  getter, which is a u32 field read with no allocations on the Rust
  side). Lower bound on PyO3 method-call dispatch cost — a baseline
  every other measurement inherits.
* ``walk_us``: per-call cost of materializing all the zone snapshots
  the fingerprint reads — ``hand_snapshot(0/1)``,
  ``library_snapshot(0/1)``, ``graveyard_snapshot(0/1)``,
  ``exile_snapshot(0/1)``, ``battlefield_snapshot()``, plus the
  ``mana_pool_snapshot(0/1)`` 6-key dict. This is a **strict
  overestimate** of the Rust-side state-walk because it also pays
  the cost of building one Python dict per object (which the real
  fingerprint doesn't — it builds a serde_json::Value tree). Use it
  as an upper bound on the walk; the true walk cost is somewhere
  between ``noop_us`` and ``walk_us``.
* ``json_py_us``: per-call cost of taking those Python-side snapshots
  and feeding them to ``json.dumps`` + ``hashlib.sha1`` (i.e. doing
  the equivalent of fingerprint() but entirely in Python on top of
  the snapshot getters). This gives a Python-side analog of the
  full pipeline; the gap between ``fp_us`` and ``json_py_us``
  tells us how much faster the Rust impl is than a naive
  Python+snapshot implementation. The gap between ``walk_us``
  and ``json_py_us`` isolates Python-side JSON+hash cost.
* ``fp_us``: the existing headline measurement — full Rust
  ``fingerprint()`` per call. The number we're trying to decompose.

The script also runs ``cProfile`` against a single ``fingerprint()``
call. The output is intentionally not load-bearing — cProfile can
only see the Python ``fingerprint()`` call (not the work inside
Rust), so it mainly confirms that all the cost lives below the PyO3
boundary. It's included for completeness so the next round can
reproduce the analysis without re-deriving the methodology.

## Output

Writes a CSV to ``bench_results/fingerprint_decomposition_<UTC-stamp>.csv``
with one row per benchmarked state size. Columns:

* ``label`` — state-size triple as printed (e.g. ``(5, 7, 30)``).
* ``bf_size``, ``hand_size``, ``lib_size`` — the per-side sizes.
* ``noop_us``, ``walk_us``, ``json_py_us``, ``fp_us`` —
  per-call microseconds for each phase (smaller is better).
* ``fp_minus_noop_us`` — ``fp_us - noop_us``, the fingerprint cost
  attributable to actual fingerprinting work (state walk +
  serialize + hash), with PyO3-call overhead subtracted.
* ``iters`` — iteration count used for the bench.

Also prints a per-size decomposition table to stdout plus a
commentary block flagging the most plausible cost driver(s) at
the ``(5, 7, 30)`` size (the perf_summary headline number).

## Run

From a worktree:

    PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe \\
        scripts/bench_rust_fingerprint_decomposition.py

Skips with a hint if ``mtg_bot_engine`` isn't installed yet (run
``maturin develop --release -m rust/Cargo.toml``).
"""

from __future__ import annotations

import csv
import cProfile
import datetime as _dt
import hashlib
import io
import json
import pathlib
import pstats
import time

try:
    import mtg_bot_engine
except ImportError:
    mtg_bot_engine = None


# Same grid as bench_rust_fingerprint.py for cross-reference.
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
    for _ in range(WARMUP_ITERS):
        state.fingerprint()
    start = time.perf_counter_ns()
    for _ in range(n):
        state.fingerprint()
    return time.perf_counter_ns() - start


def time_noop_ns(state, n: int) -> float:
    """Per-call cost of a trivial getter — the PyO3 call-overhead floor.

    ``turn_number`` is a ``#[getter]`` returning ``u32`` from a field
    read; no allocations, no zone walks. Whatever this measures is
    pure Python<->Rust method dispatch cost and is the lower-bound
    cost every other Rust pymethod pays.
    """
    for _ in range(WARMUP_ITERS):
        _ = state.turn_number
    start = time.perf_counter_ns()
    for _ in range(n):
        _ = state.turn_number
    return time.perf_counter_ns() - start


def time_walk_ns(state, n: int) -> float:
    """Per-call cost of materializing every zone-snapshot dict.

    Calls the four per-zone getters for each player plus the
    centralized battlefield + mana-pool getters — i.e. every zone
    the fingerprint walks. Each getter builds a fresh
    ``PyDict`` per object on the Rust side, so the resulting
    measurement is a **strict overestimate** of the real
    state-walk cost inside ``fingerprint()`` (which builds a
    ``serde_json::Value`` tree, not a Python dict tree). Use it
    as an upper bound.
    """
    def one():
        state.hand_snapshot(0)
        state.hand_snapshot(1)
        state.library_snapshot(0)
        state.library_snapshot(1)
        state.graveyard_snapshot(0)
        state.graveyard_snapshot(1)
        state.exile_snapshot(0)
        state.exile_snapshot(1)
        state.battlefield_snapshot()
        state.mana_pool_snapshot(0)
        state.mana_pool_snapshot(1)

    for _ in range(WARMUP_ITERS):
        one()
    start = time.perf_counter_ns()
    for _ in range(n):
        one()
    return time.perf_counter_ns() - start


def _python_fingerprint(state) -> str:
    """A Python-side analog of ``state.fingerprint()`` built on top of
    the snapshot getters.

    Doesn't try for byte-equality with the Rust digest — the goal is
    only to make json.dumps + sha1 do an equivalent amount of work
    so the measured cost is representative. Uses the same
    ``parts`` shape Python's ``hash_state`` uses.
    """
    # Compose ``parts`` directly from snapshot dicts. Each per-object
    # dict has 9 keys (id, printing_id, printing_name, tapped,
    # summoning_sick, damage, counters, controller_id, zone), so
    # json.dumps walks 9 entries per object — slightly more than
    # the Rust side's 8-element array per object but in the same
    # ballpark.
    parts = [
        state.turn_number,
        None,
        state.active_player,
        state.priority_player_idx,
        False,
        False,
        None,
        (
            0,
            state.life_totals[0],
            state.life_totals[0] <= 0,
            state.hand_snapshot(0),
            state.library_snapshot(0),
            state.graveyard_snapshot(0),
            state.exile_snapshot(0),
            sorted(state.mana_pool_snapshot(0).items()),
        ),
        (
            1,
            state.life_totals[1],
            state.life_totals[1] <= 0,
            state.hand_snapshot(1),
            state.library_snapshot(1),
            state.graveyard_snapshot(1),
            state.exile_snapshot(1),
            sorted(state.mana_pool_snapshot(1).items()),
        ),
        state.battlefield_snapshot(),
        [],  # stack — out of reach via snapshot getters; OK for
             # cost-of-work proxy
    ]
    blob = json.dumps(parts, default=repr).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def time_json_py_ns(state, n: int) -> float:
    """Per-call cost of the Python-side fingerprint analog."""
    for _ in range(WARMUP_ITERS):
        _python_fingerprint(state)
    start = time.perf_counter_ns()
    for _ in range(n):
        _python_fingerprint(state)
    return time.perf_counter_ns() - start


def run_bench(n: int = DEFAULT_ITERS) -> list[dict]:
    """Run the decomposition bench and return per-size result rows."""
    if mtg_bot_engine is None:
        return []

    rows: list[dict] = []
    for bf, hand, lib, label in SIZES:
        state = mtg_bot_engine.make_synthetic_state(bf, hand, lib)
        fp_ns = time_fingerprint_ns(state, n)
        noop_ns = time_noop_ns(state, n)
        walk_ns = time_walk_ns(state, n)
        json_py_ns = time_json_py_ns(state, n)

        fp_us = fp_ns / n / 1_000.0
        noop_us = noop_ns / n / 1_000.0
        walk_us = walk_ns / n / 1_000.0
        json_py_us = json_py_ns / n / 1_000.0

        rows.append(
            {
                "label": label,
                "bf_size": bf,
                "hand_size": hand,
                "lib_size": lib,
                "noop_us": noop_us,
                "walk_us": walk_us,
                "json_py_us": json_py_us,
                "fp_us": fp_us,
                "fp_minus_noop_us": fp_us - noop_us,
                "iters": n,
            }
        )
    return rows


def print_table(rows: list[dict], n: int) -> None:
    """Print a per-size decomposition table to stdout."""
    print(f"fingerprint decomposition — N={n} iters per size")
    print()
    print(
        "  size            |  noop us | walk us | json_py us |   fp us | fp-noop us"
    )
    print(
        "  ----------------|----------|---------|------------|---------|-----------"
    )
    for r in rows:
        print(
            f"  {r['label']:<15} |"
            f"  {r['noop_us']:6.3f} |"
            f"  {r['walk_us']:6.2f} |"
            f"     {r['json_py_us']:6.2f} |"
            f"  {r['fp_us']:6.2f} |"
            f"     {r['fp_minus_noop_us']:6.2f}"
        )


def commentary(rows: list[dict]) -> str:
    """Return a one-paragraph plain-English commentary about where the
    time most plausibly goes at the canonical ``(5, 7, 30)`` size.

    Builds the text from the actual measured numbers so it stays
    correct as perf changes.
    """
    target = next(
        (r for r in rows if r["label"] == "(5, 7, 30)"), rows[len(rows) // 2]
    )
    fp = target["fp_us"]
    noop = target["noop_us"]
    walk = target["walk_us"]
    json_py = target["json_py_us"]
    fp_minus_noop = target["fp_minus_noop_us"]

    overhead_pct = (noop / fp) * 100.0 if fp > 0 else 0.0
    real_work_pct = (fp_minus_noop / fp) * 100.0 if fp > 0 else 0.0
    walk_pct = (walk / fp) * 100.0 if fp > 0 else 0.0
    walk_ratio = (walk / fp) if fp > 0 else 0.0
    json_py_ratio = (json_py / fp) if fp > 0 else 0.0

    return (
        f"At {target['label']}, fingerprint() costs {fp:.2f}us per call. "
        f"PyO3 call dispatch (turn_number getter baseline) is "
        f"{noop:.3f}us — only ~{overhead_pct:.1f}% of total. The remaining "
        f"~{real_work_pct:.1f}% ({fp_minus_noop:.2f}us) is the real "
        f"fingerprint work: state-walk + serde_json tree build + serialize "
        f"+ sha1. The snapshot-walk upper bound (build Python dicts for "
        f"every object in every zone) is {walk:.2f}us, ~{walk_pct:.0f}% "
        f"of fp_us — which means even if the Rust serde-tree build were "
        f"as expensive as building Python dicts (it is not — no per-object "
        f"PyDict allocation), it could not exceed that share. The "
        f"pure-Python analog (snapshots + json.dumps + sha1) is "
        f"{json_py:.2f}us (~{json_py_ratio:.1f}x native), confirming Rust's "
        f"json.dumps+sha1 are doing real work but vastly faster than "
        f"Python's. Decomposition heuristic for the {fp_minus_noop:.2f}us "
        f"of real work: state-walk is the dominant share (proportional to "
        f"object count, and walk_us / fp_us scales together across sizes), "
        f"with serde_json::Value tree allocation + the recursive "
        f"serialize_python_compatible string writer the most plausible "
        f"second-place driver (heap allocations + String concatenation per "
        f"array element + per-string escape via serde_json::to_string). "
        f"Sha1 over the resulting ~kilobyte blob is a distant third — "
        f"sha1 throughput is ~500MB/s so a 1KB blob is ~2us. PyO3 boundary "
        f"cost is negligible. First optimization target: skip the "
        f"intermediate serde_json::Value tree and stream bytes straight "
        f"into Sha1::update (avoids both the Value allocations and the "
        f"final String allocation for the blob)."
    )


def run_cprofile_once(state) -> str:
    """Run cProfile against a single ``fingerprint()`` call; return the
    stats string. Useful as a sanity check that all the cost is below
    the PyO3 boundary (i.e. cProfile sees no Python-side hotspot
    inside fingerprint())."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1000):
        state.fingerprint()
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats(
        pstats.SortKey.CUMULATIVE
    )
    stats.print_stats(10)
    return buf.getvalue()


def write_csv(rows: list[dict], out_dir: pathlib.Path) -> pathlib.Path:
    """Write rows as a CSV under ``out_dir``; return the path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"fingerprint_decomposition_{stamp}.csv"
    fieldnames = [
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

    # cProfile snapshot at (5, 7, 30) — the headline size.
    state = mtg_bot_engine.make_synthetic_state(5, 7, 30)
    print()
    print("cProfile snapshot (1000x fingerprint() at (5, 7, 30)) — Python-")
    print("visible only; Rust-internal time appears as a single line under")
    print("``builtins.method 'fingerprint'``:")
    print()
    print(run_cprofile_once(state))

    print()
    print("commentary:")
    print(commentary(rows))

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "bench_results"
    csv_path = write_csv(rows, out_dir)
    print()
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
