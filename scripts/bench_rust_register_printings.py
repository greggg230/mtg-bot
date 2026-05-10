"""Bench: how long does it take to register a card printing through
the Python <-> Rust dict bridge?

The number matters because deck-load happens once per game and the
real EoE pool is ~5000 archetypes * ~60 cards = 300k card instances
(though far fewer unique printings — ~321 unique). If parsing cost
is ~us-per-card, deck-load is negligible. If it's ms-per-card, it
shows up at game start.

Run from a worktree:

    PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe \\
        scripts/bench_rust_register_printings.py
"""

from __future__ import annotations

import time

try:
    import mtg_bot_engine
except ImportError:
    raise SystemExit(
        "rust extension not built; run "
        "`maturin develop --release -m rust/Cargo.toml`"
    )


SHOCK = {
    "name": "Shock",
    "mana_cost": {"R": 1},
    "effects": [
        {"kind": "deal_damage", "amount": 2, "target": {"kind": "any_creature"}}
    ],
}


def main() -> None:
    print(f"Rust engine version: {mtg_bot_engine.__version__}")

    reg = mtg_bot_engine.Registry()
    # In-Rust microbench (skips the per-call PyO3 dispatch overhead).
    reg.register_n_times_micros(SHOCK, 100)
    in_rust_us = reg.register_n_times_micros(SHOCK, 10_000)
    print(
        f"in-Rust parse loop:        {in_rust_us:7.3f}us per call "
        f"(10000 iterations)"
    )

    # Round-trip from Python: includes per-call PyO3 dispatch +
    # interior mutability check on the Registry pyclass.
    n = 10_000
    fresh = mtg_bot_engine.Registry()
    for _ in range(20):
        fresh.register_printing(SHOCK)
    fresh = mtg_bot_engine.Registry()
    start = time.perf_counter()
    for _ in range(n):
        fresh.register_printing(SHOCK)
    elapsed = time.perf_counter() - start
    py_call_us = elapsed * 1_000_000 / n
    print(
        f"Python -> register_printing: {py_call_us:7.3f}us per call "
        f"({n} iterations, registry now {len(fresh)} entries)"
    )

    # Round-trip cost via get_printing_effects (Rust enum -> Python dict).
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(SHOCK)
    for _ in range(20):
        reg.get_printing_effects(pid)
    start = time.perf_counter()
    for _ in range(n):
        reg.get_printing_effects(pid)
    elapsed = time.perf_counter() - start
    print(
        f"Rust -> get_printing_effects: {elapsed * 1_000_000 / n:7.3f}us per call "
        f"({n} iterations)"
    )

    # Scaling spot-check: 5000 unique-ish card prints in one fresh
    # registry. (The EoE main set is 266 unique; this is a generous
    # upper bound.)
    fresh = mtg_bot_engine.Registry()
    n_cards = 5000
    start = time.perf_counter()
    for i in range(n_cards):
        card = dict(SHOCK)
        card["name"] = f"Shock #{i}"
        fresh.register_printing(card)
    bulk_elapsed = time.perf_counter() - start
    print(
        f"\nbulk register {n_cards} cards: {bulk_elapsed * 1000:7.1f}ms "
        f"({bulk_elapsed * 1_000_000 / n_cards:.2f}us per card; "
        f"registry size = {len(fresh)})"
    )


if __name__ == "__main__":
    main()
