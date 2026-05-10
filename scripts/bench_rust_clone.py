"""Head-to-head: Python ``GameState.clone()`` vs the Rust engine's clone.

Runs both on roughly the same shape (10 battlefield permanents, 7 in
hand, 50 in library, per player) so the numbers are comparable. The
Python side simulates 5 turns to get a real mid-game state; the Rust
side builds a synthetic state with matching dimensions.

Run from a worktree:
    PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe \\
        scripts/bench_rust_clone.py

Skips the Rust comparison with a hint if ``mtg_bot_engine`` isn't
installed yet (run ``maturin develop --release -m rust/pyproject.toml``).
"""

from __future__ import annotations

import copy
import random
import time

from mtg_bot.agents import RandomAgent
from mtg_bot.simulate import simulate_game

try:
    import mtg_bot_engine
except ImportError:
    mtg_bot_engine = None


def build_python_state():
    agents = [
        RandomAgent(name="a0", rng=random.Random(1001)),
        RandomAgent(name="a1", rng=random.Random(2002)),
    ]
    return simulate_game(agents=agents, colors=["G", "R"], seed=17, max_turns=5)


def time_python_clone(state, n: int) -> float:
    for _ in range(20):
        state.clone()
    start = time.perf_counter()
    for _ in range(n):
        state.clone()
    return time.perf_counter() - start


def time_python_deepcopy(state, n: int) -> float:
    for _ in range(20):
        copy.deepcopy(state)
    start = time.perf_counter()
    for _ in range(n):
        copy.deepcopy(state)
    return time.perf_counter() - start


def main() -> None:
    state = build_python_state()
    py_dims = (
        len(state.battlefield),
        sum(len(p.hand) for p in state.players) // 2,
        sum(len(p.library) for p in state.players) // 2,
    )
    print(
        f"Python mid-game state: turn={state.turn_number}, "
        f"battlefield={len(state.battlefield)}, "
        f"hand_sizes={[len(p.hand) for p in state.players]}, "
        f"library_sizes={[len(p.library) for p in state.players]}"
    )

    n = 1000
    t_clone = time_python_clone(state, n)
    t_deep = time_python_deepcopy(state, n)
    print(
        f"py state.clone():   {t_clone * 1_000_000 / n:7.2f}us "
        f"(total {t_clone * 1000:.1f}ms / {n})"
    )
    print(
        f"py copy.deepcopy:   {t_deep * 1_000_000 / n:7.2f}us "
        f"(total {t_deep * 1000:.1f}ms / {n})"
    )

    if mtg_bot_engine is None:
        print(
            "\nrust engine: not installed — run "
            "`maturin develop --release -m rust/pyproject.toml`"
        )
        return

    print(f"\nRust engine version: {mtg_bot_engine.__version__}")
    bf_per_side = max(2, py_dims[0] // 2)
    hand_per_side = max(1, py_dims[1])
    lib_per_side = max(1, py_dims[2])
    rust_state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=bf_per_side,
        num_hand=hand_per_side,
        num_library=lib_per_side,
    )
    print(
        f"rust state dims: battlefield={list(rust_state.battlefield_sizes)}, "
        f"hand={list(rust_state.hand_sizes)}, "
        f"library={list(rust_state.library_sizes)}"
    )
    if hasattr(rust_state, "registry_size"):
        print(f"rust registry size: {rust_state.registry_size} printings")
    if hasattr(mtg_bot_engine, "gameobject_byte_size"):
        print(f"rust GameObject size: {mtg_bot_engine.gameobject_byte_size()} bytes")
    rust_state.clone_n_times_micros(100)
    rust_us = rust_state.clone_n_times_micros(n)
    print(f"rust clone():       {rust_us:7.2f}us (total {rust_us * n / 1000:.1f}ms / {n})")

    py_us = t_clone * 1_000_000 / n
    if rust_us > 0:
        print(f"speedup vs py clone:    {py_us / rust_us:.2f}x")
    if rust_us > 0:
        print(f"speedup vs deepcopy:    {(t_deep * 1_000_000 / n) / rust_us:.2f}x")

    # Reference: 8e20f6b (Arc<Vec<GameObject>> CoW, name+p/t inline on
    # GameObject) reported 0.07us per clone / 269x vs Python clone in
    # its commit message. The current branch shrinks GameObject to
    # POD (printing_id + per-instance state — no inline String/p/t).
    # The bench measures pure refcount-bump CoW cost; with smaller
    # GameObjects the inner Vec contents are smaller bytes, but only
    # the Arc refcount is touched per clone, so the bench isn't
    # expected to swing dramatically. The win lands later, when
    # search workloads first-touch a zone and Arc::make_mut clones
    # the inner Vec — that clone now copies fixed-size POD bytes
    # instead of `String`-bearing structs.
    print("\nReference (pre-registry, commit 8e20f6b): 0.07us / 269x vs py clone")


if __name__ == "__main__":
    main()
