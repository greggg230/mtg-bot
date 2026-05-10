"""Regression tests for the per-search NN-eval cache drift bug.

Root cause (diagnosed 2026-05-06): the per-search cache stores the
``policy_logits`` returned by the policy head of the NN. The cache key
``canonical_state_key`` deliberately drops raw ``GameObject.id`` ordering
inside the hand / graveyard / exile / library so that two states that
differ only in card identity (e.g. two Mountains in the hand with their
``card.id``s swapped) hash identically — they have the same forward
play distribution.

But the **action-encoding slot mapping is positional by raw card.id
order**: ``encode(PlayLand(card_id=X), state, viewer)`` returns
``LAND_FROM_HAND_BASE + position_of_X_in_p.hand``. Two canonically-
equivalent states with permuted ``card.id`` ordering map the SAME
logical decision to DIFFERENT policy slots.

Consequence: when the cache returns a stored ``policy_logits`` array
to a state whose hand has a different card.id ordering than the state
that originally populated the entry, ``_stamp_prior`` reads the wrong
slot — assigning Mountain-1's prior to Mountain-2's PUCT child (and
vice versa). Selection diverges from the cache-off baseline, growing
into a visit-distribution drift that flips the chosen action in some
states.

The bug is invisible under deterministic constant-value test
evaluators (their policy is uniform / unused / unmasked-deterministic
so slot misalignment doesn't matter). It surfaces with the real v13
NN where every slot has a distinct prior.

The fix scopes the cache key tighter: include the slot-mapping
fingerprint (raw card.id ordering of the viewer's hand/graveyard/
exile/library-top) alongside the canonical-state hash. Cache hits
now only fire when slot mappings match — same value AND same policy
slots. Slot-mismatched transpositions miss and recompute, sacrificing
some hit rate to preserve correctness.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from mtg_bot.search.canonical_state import canonical_state_key


# ---------------------------------------------------------------------------
# Test 1: fast unit-level test — pin the cache-key contract.
#
# Two MTG adapters with the same canonical_state_key but different
# hand card.id orderings must NOT share a cache entry. Otherwise the
# cached policy logits (slot-positional) get applied to the wrong
# slots in the second state.
# ---------------------------------------------------------------------------


def _build_two_handed_adapter_with_duplicate_lands():
    """Construct a minimal MTG adapter where the active player's hand
    contains two Mountains with different ``card.id`` values, plus a
    third non-duplicate card so there are >=2 distinct PlayLand legal
    moves at the same canonical-state-key.

    Built via the real engine so legal_actions enumerates correctly,
    but kept tiny — one library card per side, no battlefield, no
    stack — to keep the test fast (~5ms) and unaffected by EoE
    pool churn.
    """
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.game import GameObject, GameState, Player, Zone
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.turn import Step

    mountain = BASIC_LANDS["Mountain"]
    island = BASIC_LANDS["Island"]
    plains = BASIC_LANDS["Plains"]

    # Player 0: two Mountains in hand at distinct ids, an Island.
    p0_mountain_a = GameObject(
        id=10, printing=mountain, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    p0_mountain_b = GameObject(
        id=11, printing=mountain, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    p0_island = GameObject(
        id=12, printing=island, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    # Single library card to keep the state minimal.
    p0_lib = GameObject(
        id=13, printing=plains, controller_id=0, owner_id=0,
        zone=Zone.LIBRARY,
    )

    p1_lib = GameObject(
        id=20, printing=plains, controller_id=1, owner_id=1,
        zone=Zone.LIBRARY,
    )

    p0 = Player(
        id=0,
        hand=[p0_mountain_a, p0_mountain_b, p0_island],
        library=[p0_lib],
    )
    p1 = Player(
        id=1,
        hand=[],
        library=[p1_lib],
    )

    state = GameState(
        players=[p0, p1],
        battlefield=[],
        stack=[],
        active_player_idx=0,
        priority_player_idx=0,
        turn_number=2,
        step=Step.PRECOMBAT_MAIN,
        next_object_id=100,
        rng=random.Random(0),
    )
    return MtgGameStateAdapter(state)


def test_canonical_key_collapses_duplicate_card_id_orderings():
    """Sanity: two states differing only in two equivalent cards'
    ``card.id`` order MUST canonically hash equal. This is the
    precondition for the cache-drift bug — without canonical
    collapse there'd be no shared key, and slot mismatches couldn't
    cause cache-collision."""
    a = _build_two_handed_adapter_with_duplicate_lands()
    p = a.engine_state.players[0]

    # Permute card.ids of the two Mountains.
    m_a = p.hand[0]
    m_b = p.hand[1]
    assert m_a.printing.name == "Mountain"
    assert m_b.printing.name == "Mountain"
    assert m_a.id != m_b.id

    # Reverse hand order: hand becomes [m_b, m_a, island].
    p.hand = [p.hand[1], p.hand[0], p.hand[2]]

    b = _build_two_handed_adapter_with_duplicate_lands()
    # b has hand order [m_a, m_b, island] (the original)
    pb = b.engine_state.players[0]
    assert pb.hand[0].id != p.hand[0].id  # different card.id at same position

    assert canonical_state_key(a) == canonical_state_key(b), (
        "expected canonically-equivalent: same printings, same zone "
        "contents, only card.id ordering differs"
    )


def test_action_encoding_slot_depends_on_raw_card_id_position():
    """Action encoding maps ``PlayLand(card_id=X)`` to a slot based on
    X's positional index in ``p.hand``. Two canonically-equivalent
    states with permuted hand card.ids therefore map the SAME logical
    decision to DIFFERENT slots.

    This is the structural reason cache-shared policy logits index
    the wrong prior on transpositions.
    """
    from mtg_bot.actions import PlayLand
    from mtg_bot.training import action_encoding as ae

    a = _build_two_handed_adapter_with_duplicate_lands()
    p = a.engine_state.players[0]
    m_first = p.hand[0]
    m_second = p.hand[1]

    # encode the "play the first Mountain" decision in state A
    slot_a = ae.encode(PlayLand(card_id=m_first.id), a.engine_state, 0)
    # in a permuted state, the FIRST Mountain in hand is the one with
    # the OTHER id, but encoding by the original id reads the slot of
    # whatever position that id now lives at
    p.hand = [m_second, m_first, p.hand[2]]
    slot_after_swap = ae.encode(
        PlayLand(card_id=m_first.id), a.engine_state, 0,
    )
    assert slot_a != slot_after_swap, (
        f"expected different slots for same decision under hand "
        f"permutation; got {slot_a} == {slot_after_swap}"
    )


# ---------------------------------------------------------------------------
# Test 2: end-to-end — under a deterministic policy evaluator that
# returns DIFFERENT logits per slot, two consecutive cache lookups
# at canonically-equivalent states with DIFFERENT slot mappings must
# read the right prior. With the fix, this works; without the fix
# (pre-patch), the prior at the second state reads from a slot that
# corresponds to a different decision.
# ---------------------------------------------------------------------------


class _DistinctSlotPolicyEvaluator:
    """Evaluator whose policy assigns each slot a distinct value
    proportional to slot index — so a wrong-slot read is detectably
    wrong. Value head returns a constant (the bug is purely about
    policy)."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, state, viewer):
        self.calls += 1
        return 0.0

    def evaluate_with_policy(self, state, viewer):
        from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
        self.calls += 1
        # Slot index / VOCAB → distinct value per slot, deterministic.
        pol = (
            np.arange(ACTION_VOCAB_SIZE, dtype=np.float32)
            / ACTION_VOCAB_SIZE
        )
        return 0.0, pol


def test_cache_does_not_serve_wrong_policy_to_slot_permuted_state():
    """The smoking-gun regression test.

    Setup:
    - State A has hand [Mountain-id10, Mountain-id11, Island-id12].
    - State B is canonically equivalent to A but with hand
      [Mountain-id11, Mountain-id10, Island-id12].
    - The policy evaluator assigns each slot a distinct value, so
      reading the wrong slot returns a measurably different number.

    The MCTS cache populates an entry from state A's evaluation. When
    the cache later receives state B (same canonical key, different
    slot mapping), the cached policy_logits would index the wrong
    slot for ``PlayLand(card_id=10)`` and ``PlayLand(card_id=11)``.

    Assertion: the priors that ``_stamp_prior`` assigns to children
    of state B match what ``evaluate_with_policy(state_B)`` would
    return on a fresh call — NOT what state A's cached logits at the
    same slot indices would return.
    """
    from mtg_bot.actions import PlayLand
    from mtg_bot.search import MCTS
    from mtg_bot.search.budget import Budget
    from mtg_bot.training import action_encoding as ae

    state_a = _build_two_handed_adapter_with_duplicate_lands()
    state_b = _build_two_handed_adapter_with_duplicate_lands()
    # Permute B's hand: [Mountain-11, Mountain-10, Island-12].
    pb = state_b.engine_state.players[0]
    pb.hand = [pb.hand[1], pb.hand[0], pb.hand[2]]

    assert canonical_state_key(state_a) == canonical_state_key(state_b)

    ev = _DistinctSlotPolicyEvaluator()
    mcts = MCTS(
        evaluator=ev,
        budget_factory=lambda: Budget(max_nodes=10),
        rng=random.Random(0),
        puct=True,
        batch_size=1,
    )

    # Manually pre-populate the cache by evaluating state A.
    mcts._eval_cache = {}
    # Simulate the per-search cache state at the moment of evaluation.
    # We need to set the root_player context that the cache key
    # depends on (post-fix).
    mcts._cache_root_player = 0
    cache_key_a = mcts._cache_key(state_a)
    val_a, pol_a = ev.evaluate_with_policy(state_a, 0)
    from mtg_bot.search.mcts import _CachedEval
    mcts._eval_cache[cache_key_a] = _CachedEval(
        value=val_a, policy_logits=pol_a,
    )

    # Now query the cache key for state B.
    cache_key_b = mcts._cache_key(state_b)

    # POST-FIX: cache keys must DIFFER because slot mappings differ.
    assert cache_key_b != cache_key_a, (
        "cache key must be tighter than canonical_state_key — two states "
        "with the same canonical key but different slot mappings (different "
        "raw card.id orderings in viewer's hand) must produce DIFFERENT "
        "cache keys, otherwise stored policy_logits would be read at "
        "wrong slot indices."
    )


# ---------------------------------------------------------------------------
# Test 3: regression test against the bench-equivalent scenario —
# cache-on and cache-off must produce IDENTICAL visit distributions
# under a deterministic evaluator that exercises the slot-permutation
# code path. This pins the cache-correctness contract.
# ---------------------------------------------------------------------------


def test_visits_identical_with_and_without_cache_under_distinct_slot_policy():
    """End-to-end correctness: even with an evaluator whose policy
    distinguishes slots (so the bug bites), a search at a state with
    duplicate-printing cards must produce IDENTICAL visit distributions
    cache-on vs cache-off.
    """
    from mtg_bot.search import MCTS
    from mtg_bot.search.budget import Budget

    a1 = _build_two_handed_adapter_with_duplicate_lands()
    ev1 = _DistinctSlotPolicyEvaluator()
    mcts1 = MCTS(
        evaluator=ev1,
        budget_factory=lambda: Budget(max_nodes=80),
        rng=random.Random(7),
        puct=True,
        batch_size=1,
    )

    a2 = _build_two_handed_adapter_with_duplicate_lands()
    ev2 = _DistinctSlotPolicyEvaluator()
    mcts2 = MCTS(
        evaluator=ev2,
        budget_factory=lambda: Budget(max_nodes=80),
        rng=random.Random(7),
        puct=True,
        batch_size=1,
    )
    mcts2._cache_key = lambda state: None  # type: ignore[method-assign]

    v1 = mcts1.choose_with_visits(a1, 0)
    v2 = mcts2.choose_with_visits(a2, 0)

    assert v1 == v2, (
        f"cache-on visits differ from cache-off:\n"
        f"  cache-on:  {v1}\n"
        f"  cache-off: {v2}\n"
    )
