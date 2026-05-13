"""Directional pin tests for the Step-1 Slice-D MCTS NN-eval cache key.

Slice D rebuilds the cache key around ``canonical_state_key`` alone
(dropping the v1-era ``slot_fingerprint`` suffix). Per the
NN-representation roadmap section 5 resolved decision, this is the
right key for canonical_v2 because v2 makes the slot fingerprint
redundant — canonically-equivalent states share their slot mappings.

The tests below pin the two-directional contract that defines the key:

1. **Canonical equivalence collapses to one entry.** Two states that
   differ only in card-ordering inside zones (the canonical_v2 zone
   permutations) must produce the same cache key.
2. **Genuine differences are distinguished.** Two states that differ
   in forward-relevant state (life totals, battlefield contents,
   active player, etc.) must produce different cache keys.
3. **Cache lookups are correct.** Under a slot-invariant evaluator
   (the v2 contract), cache hits never serve a wrong (value, policy)
   pair for any state — i.e., a stored entry retrieved by canonical
   equivalence is exactly what the evaluator would have returned
   on the equivalent state.

Hammer test (3): construct many canonical-equivalence pairs and many
non-equivalent pairs and assert the partition matches the cache-key
partition.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mtg_bot.search import MCTS
from mtg_bot.search.budget import Budget
from mtg_bot.search.canonical_state import canonical_state_key
from mtg_bot.search.mcts import _CachedEval


def _build_minimal_adapter(
    *,
    p0_hand_ids: tuple[int, ...] = (10, 11, 12),
    p0_life: int = 20,
    p1_life: int = 20,
    p0_battlefield_creatures: int = 0,
    seed: int = 0,
):
    """Construct a tiny adapter with controlled state for cache-key tests.

    Player 0's hand contains two Mountains (with the first two ids in
    ``p0_hand_ids``) and an Island (with the third id). Library has
    one Plains. Player 1 has an empty hand and one Plains in library.
    Life totals and battlefield population are tunable.
    """
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.game import GameObject, GameState, Player, Zone
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.turn import Step

    mountain = BASIC_LANDS["Mountain"]
    island = BASIC_LANDS["Island"]
    plains = BASIC_LANDS["Plains"]

    m_a = GameObject(
        id=p0_hand_ids[0], printing=mountain, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    m_b = GameObject(
        id=p0_hand_ids[1], printing=mountain, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    isl = GameObject(
        id=p0_hand_ids[2], printing=island, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    p0_lib = GameObject(
        id=50, printing=plains, controller_id=0, owner_id=0,
        zone=Zone.LIBRARY,
    )
    p1_lib = GameObject(
        id=60, printing=plains, controller_id=1, owner_id=1,
        zone=Zone.LIBRARY,
    )

    next_id = 100
    battlefield: list = []
    for _ in range(p0_battlefield_creatures):
        # Mountain-named placeholder creatures on the battlefield —
        # just to vary state. Use Mountain printing (basic land) so
        # we don't depend on more elaborate card data.
        obj = GameObject(
            id=next_id, printing=mountain, controller_id=0, owner_id=0,
            zone=Zone.BATTLEFIELD,
        )
        battlefield.append(obj)
        next_id += 1

    p0 = Player(id=0, hand=[m_a, m_b, isl], library=[p0_lib], life=p0_life)
    p1 = Player(id=1, hand=[], library=[p1_lib], life=p1_life)

    state = GameState(
        players=[p0, p1],
        battlefield=battlefield,
        stack=[],
        active_player_idx=0,
        priority_player_idx=0,
        turn_number=2,
        step=Step.PRECOMBAT_MAIN,
        next_object_id=next_id,
        rng=random.Random(seed),
    )
    return MtgGameStateAdapter(state)


def _make_mcts(seed: int = 0) -> MCTS:
    """Build an MCTS instance with placeholder evaluator and a viewer
    pinned so ``_cache_key`` returns a real digest."""

    class _DummyPolicyEvaluator:
        def evaluate(self, state, viewer):
            return 0.0

        def evaluate_with_policy(self, state, viewer):
            from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
            return 0.0, np.zeros(ACTION_VOCAB_SIZE, dtype=np.float32)

    mcts = MCTS(
        evaluator=_DummyPolicyEvaluator(),
        budget_factory=lambda: Budget(max_nodes=4),
        rng=random.Random(seed),
        puct=True,
        batch_size=1,
    )
    mcts._cache_root_player = 0
    return mcts


# ---------------------------------------------------------------------------
# Direction 1: canonically-equivalent states share a cache key.
# ---------------------------------------------------------------------------


def test_cache_key_collapses_zone_reorder_in_hand():
    """Two states differing only in hand-card ordering (the kind of
    permutation canonical_v2 normalizes away) share a cache key."""
    a = _build_minimal_adapter(p0_hand_ids=(10, 11, 12))
    b = _build_minimal_adapter(p0_hand_ids=(11, 10, 12))
    # Sanity: canonical hashes are equal (precondition).
    assert canonical_state_key(a) == canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) == mcts._cache_key(b)


def test_cache_key_collapses_reverse_hand_order():
    """Reversing the entire hand list still produces a canonically-
    equivalent state — same cache key."""
    a = _build_minimal_adapter()
    b = _build_minimal_adapter()
    pb = b.engine_state.players[0]
    pb.hand = list(reversed(pb.hand))
    assert canonical_state_key(a) == canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) == mcts._cache_key(b)


def test_cache_key_collapses_graveyard_reorder():
    """Two states with the same graveyard contents but different
    insertion order share a cache key. canonical_state_key sorts the
    graveyard by ``(printing.name,)`` (see canonical_state.py
    module docstring)."""
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.game import GameObject, Zone

    a = _build_minimal_adapter()
    b = _build_minimal_adapter()
    mountain = BASIC_LANDS["Mountain"]
    island = BASIC_LANDS["Island"]

    for adapter, order in [
        (a, ["Mountain", "Island", "Mountain"]),
        (b, ["Island", "Mountain", "Mountain"]),
    ]:
        p = adapter.engine_state.players[0]
        p.graveyard = [
            GameObject(
                id=200 + i,
                printing=mountain if name == "Mountain" else island,
                controller_id=0,
                owner_id=0,
                zone=Zone.GRAVEYARD,
            )
            for i, name in enumerate(order)
        ]
    assert canonical_state_key(a) == canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) == mcts._cache_key(b)


# ---------------------------------------------------------------------------
# Direction 2: genuinely different states get different cache keys.
# ---------------------------------------------------------------------------


def test_cache_key_distinguishes_different_life_totals():
    """Different life totals are forward-relevant; cache keys differ."""
    a = _build_minimal_adapter(p0_life=20)
    b = _build_minimal_adapter(p0_life=10)
    assert canonical_state_key(a) != canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) != mcts._cache_key(b)


def test_cache_key_distinguishes_different_opponent_life():
    """Opponent life delta also produces distinct cache keys."""
    a = _build_minimal_adapter(p1_life=20)
    b = _build_minimal_adapter(p1_life=5)
    assert canonical_state_key(a) != canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) != mcts._cache_key(b)


def test_cache_key_distinguishes_different_battlefield_population():
    """Adding a permanent to the battlefield changes the cache key."""
    a = _build_minimal_adapter(p0_battlefield_creatures=0)
    b = _build_minimal_adapter(p0_battlefield_creatures=1)
    assert canonical_state_key(a) != canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) != mcts._cache_key(b)


def test_cache_key_distinguishes_different_hand_printings():
    """A genuinely different hand (different card printings) produces
    a different cache key — even when zone size and life totals
    match."""
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.game import GameObject, Zone

    a = _build_minimal_adapter()
    b = _build_minimal_adapter()
    # Replace one Mountain in b with a Plains so the multiset of
    # printings differs.
    plains = BASIC_LANDS["Plains"]
    pb = b.engine_state.players[0]
    pb.hand[0] = GameObject(
        id=pb.hand[0].id, printing=plains, controller_id=0, owner_id=0,
        zone=Zone.HAND,
    )
    assert canonical_state_key(a) != canonical_state_key(b)
    mcts = _make_mcts()
    assert mcts._cache_key(a) != mcts._cache_key(b)


# ---------------------------------------------------------------------------
# Cache-key shape: bytes digest, dict-key suitable.
# ---------------------------------------------------------------------------


def test_cache_key_is_32_byte_digest():
    """The Slice-D cache key is the bare ``canonical_state_key``
    digest — 32-byte ``bytes``, suitable as a ``dict`` key."""
    a = _build_minimal_adapter()
    mcts = _make_mcts()
    k = mcts._cache_key(a)
    assert isinstance(k, bytes)
    assert len(k) == 32
    # Confirm it works as a dict key.
    d: dict[bytes, int] = {k: 1}
    assert d[k] == 1


def test_cache_key_independent_of_cache_root_player_under_slice_d():
    """The pre-Slice-D key embedded a viewer-keyed slot fingerprint.
    The Slice-D key is just ``canonical_state_key`` and doesn't read
    ``_cache_root_player`` — both viewers see the same digest.

    This pins the simplification: the cache key is now purely a
    function of the state."""
    a = _build_minimal_adapter()
    mcts0 = _make_mcts()
    mcts0._cache_root_player = 0
    mcts1 = _make_mcts()
    mcts1._cache_root_player = 1
    assert mcts0._cache_key(a) == mcts1._cache_key(a)


def test_cache_key_returns_none_for_toy_state():
    """Toy states without ``engine_state`` bypass the cache (toys'
    evaluators are cheap — caching is pointless and the canonical
    hash is MTG-specific)."""

    class _ToyState:
        pass

    mcts = _make_mcts()
    assert mcts._cache_key(_ToyState()) is None


# ---------------------------------------------------------------------------
# Hammer test: cache lookups under a slot-invariant evaluator never
# return a wrong-state value.
# ---------------------------------------------------------------------------


class _SlotInvariantEvaluator:
    """Value = function of canonical_state_key (deterministic).
    Policy = uniform (slot-invariant). Mirrors the canonical_v2
    contract: equivalent states must have equivalent priors at
    matching slots, which a uniform policy trivially satisfies."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, state, viewer):
        from mtg_bot.search.canonical_state import canonical_state_key
        self.calls += 1
        k = canonical_state_key(state)
        return (k[0] / 255.0) * 2.0 - 1.0

    def evaluate_with_policy(self, state, viewer):
        from mtg_bot.search.canonical_state import canonical_state_key
        from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
        self.calls += 1
        k = canonical_state_key(state)
        val = (k[0] / 255.0) * 2.0 - 1.0
        pol = np.ones(ACTION_VOCAB_SIZE, dtype=np.float32) / float(
            ACTION_VOCAB_SIZE,
        )
        return val, pol


def test_cache_lookup_returns_value_consistent_with_canonical_key_partition():
    """Hammer test. Construct many state pairs:

    - **Equivalent pairs** (zone reorder): cache lookup at one state
      must return what evaluating the other would return.
    - **Inequivalent pairs**: cache populated from one must NOT be
      returned when looking up the other.

    Drives home the directional invariant under a real (state-keyed)
    evaluator: cache correctness reduces to the
    ``canonical_state_key`` equivalence partition matching the cache
    partition.
    """
    rng = random.Random(0)
    ev = _SlotInvariantEvaluator()
    mcts = _make_mcts()

    # Build 16 distinct base states by varying life and battlefield.
    bases = []
    for i in range(16):
        bases.append(
            _build_minimal_adapter(
                p0_life=20 - i,
                p1_life=20 - (i // 2),
                p0_battlefield_creatures=i % 4,
                seed=i,
            )
        )

    # For each base, also build a canonically-equivalent twin by
    # reversing the hand list.
    twins = []
    for base in bases:
        twin = _build_minimal_adapter(
            p0_life=base.engine_state.players[0].life,
            p1_life=base.engine_state.players[1].life,
            p0_battlefield_creatures=len(base.engine_state.battlefield),
            seed=42,
        )
        twin.engine_state.players[0].hand = list(reversed(
            twin.engine_state.players[0].hand,
        ))
        twins.append(twin)

    # Sanity-check the construction: base[i] and twin[i] are
    # canonically equivalent; base[i] and base[j] (i != j) are not.
    for i, (base, twin) in enumerate(zip(bases, twins)):
        assert canonical_state_key(base) == canonical_state_key(twin), (
            f"base/twin pair {i} not canonically equivalent"
        )
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            assert canonical_state_key(bases[i]) != canonical_state_key(
                bases[j],
            ), f"distinct bases {i}/{j} unexpectedly canonical-equal"

    # Populate the cache by evaluating every base.
    for base in bases:
        val, pol = ev.evaluate_with_policy(base, 0)
        k = mcts._cache_key(base)
        mcts._eval_cache[k] = _CachedEval(value=val, policy_logits=pol)

    # Direction 1 (equivalence shares entry): each twin lookup must
    # produce the same cached value as its sibling base.
    for i, (base, twin) in enumerate(zip(bases, twins)):
        base_entry = mcts._eval_cache[mcts._cache_key(base)]
        twin_entry = mcts._eval_cache[mcts._cache_key(twin)]
        assert base_entry is twin_entry, (
            f"pair {i}: twin lookup returned a DIFFERENT cache entry "
            f"than base — canonical equivalence not respected at the "
            f"cache layer"
        )

    # Direction 2 (inequivalence keeps distinct entries): for every
    # distinct pair (i, j), the cached entries must be different
    # objects with different values.
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            ei = mcts._eval_cache[mcts._cache_key(bases[i])]
            ej = mcts._eval_cache[mcts._cache_key(bases[j])]
            assert ei is not ej
            assert ei.value != ej.value, (
                f"distinct states {i}/{j} resolved to the same cached "
                f"value {ei.value}; canonical-key collision suspected"
            )
