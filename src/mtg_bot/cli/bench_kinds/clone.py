"""Micro-benchmark: GameState.clone() vs copy.deepcopy(state).

Run:
    .venv/Scripts/python.exe scripts/bench_clone.py

Not part of the test suite — this script times a hot-path that search
will exercise thousands of times per decision. If someone regresses
the clone path, the ratio here should surface it.
"""

from __future__ import annotations

import copy
import random
import time

from mtg_bot.agents import RandomAgent
from mtg_bot.simulate import simulate_game


def build_mid_game_state(turns: int = 5, seed: int = 17):
    agents = [
        RandomAgent(name="a0", rng=random.Random(1001)),
        RandomAgent(name="a1", rng=random.Random(2002)),
    ]
    return simulate_game(
        agents=agents,
        colors=["G", "R"],
        seed=seed,
        max_turns=turns,
    )


def time_calls(fn, n: int) -> float:
    """Return total wall-clock seconds for ``n`` calls to ``fn``."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return time.perf_counter() - start


def main(argv=None) -> None:  # pragma: no cover
    # Fixed-config bench; argv accepted for CLI dispatch uniformity.
    del argv
    state = build_mid_game_state()
    print(
        f"Mid-game state: turn={state.turn_number}, "
        f"battlefield={len(state.battlefield)}, "
        f"stack={len(state.stack)}, "
        f"logs={len(state.log)}, "
        f"lib_sizes={[len(p.library) for p in state.players]}, "
        f"hand_sizes={[len(p.hand) for p in state.players]}"
    )

    n = 1000

    # Warmup — JITs don't matter in CPython, but page-in / cache is real.
    for _ in range(20):
        state.clone()
    for _ in range(20):
        copy.deepcopy(state)

    t_clone = time_calls(state.clone, n)
    t_deep = time_calls(lambda: copy.deepcopy(state), n)

    def fmt(seconds: float) -> str:
        return f"{seconds * 1000:.1f}ms for {n} calls " \
               f"({seconds * 1_000_000 / n:.1f}us each)"

    ratio = t_deep / t_clone if t_clone > 0 else float("inf")
    print(f"clone():   {fmt(t_clone)}")
    print(f"deepcopy:  {fmt(t_deep)}")
    print(f"ratio:     {ratio:.2f}x faster than deepcopy")


if __name__ == "__main__":
    main()
