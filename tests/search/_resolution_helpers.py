"""Shared fixtures for the resolution-decision tests: a hand-built
target pause (Bombard into two creatures) and "chaos" decks drawn from
the whole card pool so random play reaches many pause kinds."""

from __future__ import annotations

import random

from mtg_bot.actions import CastSpell, PassPriority
from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.cards.source import get_active_source
from mtg_bot.game import Zone
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.simulate import setup_game
from mtg_bot.turn import Step
from tests._test_decks import simple_test_decklists


def bombard_state(seed: int = 42):
    """P0 holds Bombard ("4 damage to target creature") with five
    Mountains; P1 has two creatures."""
    printings = get_active_source().get_printings()
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(),
        seed=seed,
    )
    for _ in range(5):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD,
        )
        state.battlefield.append(m)
    card = state.new_object(printings["Bombard"], owner_id=0, zone=Zone.HAND)
    state.players[0].hand.append(card)
    for name in ("Wild Bear", "Sapling"):
        c = state.new_object(
            SYNTHETIC_CREATURES[name], owner_id=1, zone=Zone.BATTLEFIELD,
        )
        c.summoning_sick = False
        state.battlefield.append(c)
    return state


def drive_to_precombat_main(adapter) -> None:
    for _ in range(20):
        if adapter.engine_state.step == Step.PRECOMBAT_MAIN:
            return
        adapter.apply(PassPriority())


def paused_bombard_adapter(**kwargs) -> MtgGameStateAdapter:
    """An adapter paused on Bombard's target choice (P0 decides)."""
    kwargs.setdefault("expose_resolution_decisions", True)
    adapter = MtgGameStateAdapter(bombard_state(), **kwargs)
    drive_to_precombat_main(adapter)
    cast = next(
        a for a in adapter.legal_decisions() if isinstance(a, CastSpell)
    )
    adapter.apply(cast)
    assert adapter.is_resolution_paused
    return adapter


def chaos_decklists(seed: int) -> list[list[str]]:
    """Two 60-card lists: 5 x 5 basics + 35 random nonland cards from
    the whole active pool (deterministic per seed)."""
    rng = random.Random(seed)
    printings = get_active_source().get_printings()
    names = sorted(
        n for n, p in printings.items()
        if not p.is_land and not getattr(p, "is_token", False)
    )
    decks = []
    for _ in range(2):
        deck = [b for b in ("Plains", "Island", "Swamp", "Mountain", "Forest")
                for _ in range(5)]
        deck += rng.sample(names, 35)
        decks.append(deck)
    return decks


def chaos_adapter(seed: int, **kwargs) -> MtgGameStateAdapter:
    state = setup_game(
        [RandomAgent(rng=random.Random(seed)),
         RandomAgent(rng=random.Random(seed + 1))],
        decks=chaos_decklists(seed), seed=seed,
    )
    kwargs.setdefault("expose_resolution_decisions", True)
    return MtgGameStateAdapter(state, **kwargs)
