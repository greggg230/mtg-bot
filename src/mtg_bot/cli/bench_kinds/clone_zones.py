"""Per-zone breakdown of GameState.clone() cost.

Diagnostic for the zone-level perf sprint after the CoW commit
(library / graveyard / exile became copy-on-write; hand / battlefield /
mana_pool did not). This script builds a representative mid-game and
late-game state, times ``clone()`` baseline, then monkey-patches the
non-CoW zones one at a time to skip their work and attributes the
delta.

Run::

    PYTHONPATH=src .venv-py313/Scripts/python.exe scripts/bench_clone_zones.py

Not part of the test suite. Numbers are noisy under heavy concurrent
load; use a high iteration count and/or repeat after the host quiets.
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

from mtg_bot import game as game_mod
from mtg_bot.agents import RandomAgent
from mtg_bot.game import GameState
from mtg_bot.simulate import simulate_game


def build_state(target_turn: int, seed: int = 17) -> GameState:
    agents = [
        RandomAgent(name="a0", rng=random.Random(1001)),
        RandomAgent(name="a1", rng=random.Random(2002)),
    ]
    return simulate_game(
        agents=agents,
        colors=["G", "R"],
        seed=seed,
        max_turns=target_turn,
    )


def state_summary(state: GameState) -> str:
    lib = [len(p.library) for p in state.players]
    hand = [len(p.hand) for p in state.players]
    gy = [len(p.graveyard) for p in state.players]
    ex = [len(p.exile) for p in state.players]
    mp = [len(p.mana_pool) for p in state.players]
    return (
        f"turn={state.turn_number} bf={len(state.battlefield)} "
        f"stack={len(state.stack)} lib={lib} hand={hand} "
        f"gy={gy} exile={ex} mana_pool={mp}"
    )


def time_calls(fn, n: int, repeats: int = 5) -> tuple[float, float]:
    """Return (best_us_per_call, median_us_per_call) over ``repeats``
    timing trials, each running ``n`` calls."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - start
        samples.append(elapsed * 1_000_000 / n)
    return min(samples), statistics.median(samples)


# ---------------------------------------------------------------- monkey-patch
#
# We clone the original method, then install replacements that skip the
# work for one zone at a time. ``np.X = []`` (or share-by-ref where
# safe) yields a state that doesn't satisfy clone's full semantics, but
# the bench only times — it doesn't run a game on the result. The
# delta vs baseline attributes wall-clock cost to the skipped zone.

_ORIG_CLONE = GameState.clone


def _patched_clone(skip: str):
    """Return a clone implementation that skips work for ``skip``.

    ``skip`` is one of: ``"hand"``, ``"battlefield"``, ``"mana_pool"``,
    or ``"none"`` (baseline).
    """
    if skip == "none":
        return _ORIG_CLONE

    # Re-derive the original's body inline. We can't simply call the
    # original and then null out a zone — we want to remove the WORK,
    # not add it back. So we copy the body verbatim and substitute the
    # one zone with a cheap version (share-by-reference). This is the
    # ugly-but-honest path; keeping the bench faithful matters more
    # than DRY here.
    _CowList = game_mod._CowList
    _clone_game_object = game_mod._clone_game_object
    _clone_mana_coin = game_mod._clone_mana_coin
    _clone_player = game_mod._clone_player
    _clone_stack_item = game_mod._clone_stack_item
    LogBuffer = game_mod.LogBuffer
    import random as _random

    def clone(self):
        id_map: dict[int, object] = {}

        def clone_obj(o):
            existing = id_map.get(o.id)
            if existing is not None:
                return existing
            new = _clone_game_object(o)
            id_map[o.id] = new
            return new

        new_players = []
        for p in self.players:
            np = _clone_player(p)
            src_lib = p.library
            if not isinstance(src_lib, _CowList):
                src_lib = _CowList(src_lib)
                p.library = src_lib
            src_gy = p.graveyard
            if not isinstance(src_gy, _CowList):
                src_gy = _CowList(src_gy)
                p.graveyard = src_gy
            src_ex = p.exile
            if not isinstance(src_ex, _CowList):
                src_ex = _CowList(src_ex)
                p.exile = src_ex

            new_lib = _CowList(src_lib)
            new_gy = _CowList(src_gy)
            new_ex = _CowList(src_ex)
            for clone_side, src_side in (
                (new_lib, src_lib),
                (new_gy, src_gy),
                (new_ex, src_ex),
            ):
                if src_side._owned:
                    group = [src_side, clone_side]
                    src_side._owned = False
                    src_side._clone_each = _clone_game_object
                    src_side._share_group = group
                else:
                    group = src_side._share_group
                    group.append(clone_side)
                clone_side._owned = False
                clone_side._clone_each = _clone_game_object
                clone_side._share_group = group
            np.library = new_lib
            np.graveyard = new_gy
            np.exile = new_ex

            if skip == "hand":
                np.hand = p.hand  # share by reference (unsafe; bench only)
            else:
                np.hand = [clone_obj(o) for o in p.hand]

            if skip == "mana_pool":
                np.mana_pool = p.mana_pool  # share by reference
            else:
                np.mana_pool = [_clone_mana_coin(c) for c in p.mana_pool]

            new_players.append(np)

        if skip == "battlefield":
            new_battlefield = self.battlefield  # share by reference
        else:
            new_battlefield = [clone_obj(o) for o in self.battlefield]

        new_stack = []
        for item in self.stack:
            ni = _clone_stack_item(item)
            ni.source = clone_obj(item.source)
            if item.cost_paid_objects:
                ni.cost_paid_objects = tuple(
                    clone_obj(o) for o in item.cost_paid_objects
                )
            new_stack.append(ni)

        new_log = LogBuffer()
        new_log.enabled = self.log.enabled

        new_rng = _random.Random()
        new_rng.setstate(self.rng.getstate())

        new_resolving_source = (
            clone_obj(self.resolving_source)
            if self.resolving_source is not None else None
        )
        new_resolving_cost_paid_objects = (
            tuple(clone_obj(o) for o in self.resolving_cost_paid_objects)
            if self.resolving_cost_paid_objects else ()
        )

        new_state = GameState(
            players=new_players,
            battlefield=new_battlefield,
            stack=new_stack,
            active_player_idx=self.active_player_idx,
            priority_player_idx=self.priority_player_idx,
            turn_number=self.turn_number,
            step=self.step,
            next_object_id=self.next_object_id,
            continuous_effects=list(self.continuous_effects),
            next_timestamp=self.next_timestamp,
            log=new_log,
            rng=new_rng,
            winner_id=self.winner_id,
            is_draw=self.is_draw,
            permanent_left_battlefield_this_turn=(
                self.permanent_left_battlefield_this_turn
            ),
            spell_warped_this_turn=self.spell_warped_this_turn,
            once_per_turn_fired=set(self.once_per_turn_fired),
            delayed_triggers=[dict(d) for d in self.delayed_triggers],
            resolving_source=new_resolving_source,
            resolving_controller_id=self.resolving_controller_id,
            resolving_x_value=self.resolving_x_value,
            resolving_kicked=self.resolving_kicked,
            resolving_modes=self.resolving_modes,
            resolving_additional_cost_choice_id=(
                self.resolving_additional_cost_choice_id
            ),
            resolving_combat_damage_amount=(
                self.resolving_combat_damage_amount
            ),
            resolving_combat_damage_source_controller_id=(
                self.resolving_combat_damage_source_controller_id
            ),
            resolving_counter_amount=self.resolving_counter_amount,
            resolving_cost_paid_objects=new_resolving_cost_paid_objects,
            pending_end_of_combat_sacrifices=list(
                self.pending_end_of_combat_sacrifices
            ),
            players_who_created_tokens_this_turn=set(
                self.players_who_created_tokens_this_turn
            ),
            ability_resolutions_this_turn=dict(
                self.ability_resolutions_this_turn
            ),
            prevent_combat_damage_this_turn=(
                self.prevent_combat_damage_this_turn
            ),
            extra_turn_queue=list(self.extra_turn_queue),
            creatures_died_this_turn=self.creatures_died_this_turn,
        )
        return new_state

    return clone


def bench_state(label: str, state: GameState, n: int, repeats: int) -> None:
    print(f"\n=== {label} ===")
    print(f"  state: {state_summary(state)}")

    # Warmup
    for _ in range(50):
        state.clone()

    variants = [
        ("baseline", "none"),
        ("skip_hand", "hand"),
        ("skip_battlefield", "battlefield"),
        ("skip_mana_pool", "mana_pool"),
    ]
    results: dict[str, tuple[float, float]] = {}
    for name, skip in variants:
        GameState.clone = _patched_clone(skip)
        # Re-warmup (CoW share groups already exist from baseline; this
        # is fine — the bench captures steady-state cost).
        for _ in range(20):
            state.clone()
        best, med = time_calls(state.clone, n, repeats=repeats)
        results[name] = (best, med)
        print(
            f"  {name:18s}  best={best:7.2f} us/clone  "
            f"median={med:7.2f} us/clone"
        )

    GameState.clone = _ORIG_CLONE

    base_best = results["baseline"][0]
    print(f"\n  per-zone delta (best-of {repeats}, baseline minus skip):")
    for name, skip in variants[1:]:
        skip_best = results[name][0]
        delta = base_best - skip_best
        pct = 100.0 * delta / base_best if base_best > 0 else 0.0
        zone = skip
        print(
            f"    {zone:14s}  delta={delta:7.2f} us  ({pct:5.1f}% of baseline)"
        )


def main(argv=None) -> None:  # pragma: no cover
    # Fixed-config diagnostic; argv is accepted for CLI dispatch
    # uniformity but not parsed (this bench has no flags).
    del argv
    n = 1500
    repeats = 7

    print(f"# bench_clone_zones.py — n={n} per trial, {repeats} trials each")

    mid = build_state(target_turn=8, seed=17)
    bench_state("MID-GAME (target turn 8)", mid, n=n, repeats=repeats)

    late = build_state(target_turn=18, seed=17)
    bench_state("LATE-GAME (target turn 18)", late, n=n, repeats=repeats)

    # A deeper late-game with a different seed — hands often get
    # bigger, exile/graveyard fill out further. Helpful for the
    # hand-cost sanity check (the turn-18 sample had only 1 card in
    # hand, which understates hand cost).
    fat = build_state(target_turn=14, seed=42)
    bench_state("FAT MID-GAME (target turn 14, seed 42)", fat, n=n, repeats=repeats)


if __name__ == "__main__":
    main()
