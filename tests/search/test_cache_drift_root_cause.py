"""Regression tests pinning the historic cache-drift bug and the
Step-1 Slice D resolution.

Original bug (diagnosed 2026-05-06): the per-search NN-eval cache stores
the ``policy_logits`` returned by the policy head of the NN. The cache
key ``canonical_state_key`` deliberately drops raw ``GameObject.id``
ordering inside the hand / graveyard / exile / library so that two
states that differ only in card identity (e.g. two Mountains in the
hand with their ``card.id``s swapped) hash identically.

But the v1 action-encoding slot mapping was POSITIONAL by raw card.id
order: ``encode(PlayLand(card_id=X), state, viewer)`` returned
``LAND_FROM_HAND_BASE + position_of_X_in_p.hand``. Two canonically-
equivalent states with permuted ``card.id`` ordering mapped the SAME
logical decision to DIFFERENT policy slots, so reusing a cached
``policy_logits`` array at the new state's slot indices read the
wrong prior.

The original fix (commit 2f50fca) scoped the cache key tighter by
appending a slot-mapping fingerprint: ``(canonical_state_key, slot_fp)``
where ``slot_fp`` was the viewer's raw ``card.id`` ordering. That
suffix held the v1 cache to slot-mapping-matched entries only.

**Step 1 Slice D (2026-05-12): the slot_fp suffix was dropped.**
canonical_v2 action encoding (the target for Step 1's structured-input
trunk) canonicalizes hand / graveyard / exile BEFORE slot assignment,
so two states sharing ``canonical_state_key`` ALSO share slot
mappings. The slot_fingerprint workaround became a no-op under v2;
the cache key is now just ``canonical_state_key(state)``.

The drift bug remains a v1-only failure mode under the simplified
key, but v1 (v13/v15/v16) is documented as legacy and the production
path is migrating to v2. The roadmap explicitly accepts the trade-off:
the new trunk is v2-only, so the cache is correct for it; v1 weights
keep working with a known weaker guarantee.

The tests below pin the new contract: canonically-equivalent states
share their cache entry, and the cache key is the bare
``canonical_state_key`` digest.
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
# hand card.id orderings SHARE a cache entry under the Slice-D key.
# (Pre-Slice-D they were separate entries; the slot_fp suffix made
# them differ.)
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


def test_v1_action_encoding_slot_depends_on_raw_card_id_position():
    """Under positional_v1 action encoding, ``PlayLand(card_id=X)``
    maps to a slot based on X's positional index in ``p.hand``. Two
    canonically-equivalent states with permuted hand card.ids therefore
    map the SAME logical decision to DIFFERENT slots.

    This is the structural reason cache-shared policy logits indexed
    the wrong prior on transpositions UNDER V1. canonical_v2 fixes
    this by canonicalizing zones before slot assignment (see
    ``test_action_encoding_canonical.py``).
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
# Test 2: cache-key shape under Slice D — canonically-equivalent
# states share a key. This is the inverse of the pre-Slice-D test,
# which required them to DIFFER.
# ---------------------------------------------------------------------------


class _DistinctSlotPolicyEvaluator:
    """Evaluator whose policy assigns each slot a distinct value
    proportional to slot index. Value head returns a constant."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, state, viewer):
        self.calls += 1
        return 0.0

    def evaluate_with_policy(self, state, viewer):
        from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
        self.calls += 1
        # Slot index / VOCAB -> distinct value per slot, deterministic.
        pol = (
            np.arange(ACTION_VOCAB_SIZE, dtype=np.float32)
            / ACTION_VOCAB_SIZE
        )
        return 0.0, pol


def test_cache_key_collapses_canonically_equivalent_states():
    """Post-Slice-D, canonically-equivalent states share a cache key.

    Setup:
    - State A has hand [Mountain-id10, Mountain-id11, Island-id12].
    - State B is canonically equivalent to A but with hand
      [Mountain-id11, Mountain-id10, Island-id12].

    Under the new Slice-D cache key (just
    ``canonical_state_key(state)``), the two states produce the
    SAME cache key — the slot_fingerprint suffix that used to
    distinguish them is gone.
    """
    from mtg_bot.search import MCTS
    from mtg_bot.search.budget import Budget

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
    mcts._cache_root_player = 0

    cache_key_a = mcts._cache_key(state_a)
    cache_key_b = mcts._cache_key(state_b)

    assert cache_key_a == cache_key_b, (
        "Slice-D contract: canonically-equivalent states share a "
        "cache key. The slot_fingerprint suffix was removed because "
        "canonical_v2 action encoding canonicalizes zones before "
        "slot assignment, so the suffix is redundant. v1 weights "
        "still in service accept a known correctness gap on the "
        "policy-positional read; v2 weights make the cache hit safe."
    )
    # And the key is the bare 32-byte digest.
    assert isinstance(cache_key_a, bytes) and len(cache_key_a) == 32


# ---------------------------------------------------------------------------
# Test 3: regression test against the bench-equivalent scenario —
# cache-on and cache-off must produce IDENTICAL visit distributions
# WHEN THE EVALUATOR DOESN'T DEPEND ON SLOT POSITION (i.e., constant
# value, or v2-canonical priors). This pins the cache-correctness
# contract for the supported (v2) path.
#
# Under v1 with a slot-distinct policy evaluator the visit distribution
# diverges (see CLAUDE.md: "v2 weights make the slot_fingerprint
# workaround a no-op. The workaround stays in place for v1 weights" —
# Slice D removes the workaround, accepting the v1-only gap).
# ---------------------------------------------------------------------------


def test_visits_identical_with_and_without_cache_under_constant_policy():
    """End-to-end correctness for the supported path (constant /
    canonical-equivalent policy): a search at a state with
    duplicate-printing cards produces IDENTICAL visit distributions
    cache-on vs cache-off.
    """
    from mtg_bot.search import MCTS
    from mtg_bot.search.budget import Budget
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    class _ConstantPolicyEvaluator:
        """Slot-invariant policy — every slot gets the same prior.
        Mirrors the v2 contract where canonically-equivalent states
        produce identical slot priors."""

        def __init__(self):
            self.calls = 0

        def evaluate(self, state, viewer):
            self.calls += 1
            return 0.0

        def evaluate_with_policy(self, state, viewer):
            self.calls += 1
            return 0.0, np.ones(ACTION_VOCAB_SIZE, dtype=np.float32) / float(
                ACTION_VOCAB_SIZE,
            )

    a1 = _build_two_handed_adapter_with_duplicate_lands()
    ev1 = _ConstantPolicyEvaluator()
    mcts1 = MCTS(
        evaluator=ev1,
        budget_factory=lambda: Budget(max_nodes=80),
        rng=random.Random(7),
        puct=True,
        batch_size=1,
    )

    a2 = _build_two_handed_adapter_with_duplicate_lands()
    ev2 = _ConstantPolicyEvaluator()
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
