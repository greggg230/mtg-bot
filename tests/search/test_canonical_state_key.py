"""Tests for ``canonical_state_key``: the content-addressed hash used
by the per-search NN-eval cache in :class:`MCTS`.

The bar is the equivalence rules from the module docstring:

- Equivalence (must hash equal): two states differing only in
  ``card.id`` ordering of equivalent same-printing copies; mana-pool
  list permutations; continuous-effects registry list-order;
  hand-card list order.
- Distinguishment (must hash differently): differing ``_pass_count``,
  ``_combat_phase``, mana-pool color, stack target, life total, life
  totals, counters, turn number, step.
- Stability: hashing the same state twice produces the same key;
  hashing a clone produces the same key as the original.
"""

from __future__ import annotations

import random

import pytest

from mtg_bot.actions import PassPriority
from mtg_bot.agents import RandomAgent
from mtg_bot.game import (
    ContinuousEffect, GameObject, ManaCoin, StackItem, Zone,
)
from mtg_bot.search.canonical_state import (
    _canonical_state_tuple, canonical_state_key,
)
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.simulate import setup_game
from tests._test_decks import simple_test_decklists


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _adapter(seed: int = 42) -> MtgGameStateAdapter:
    """Build a real MTG adapter from the simple two-seat deck pair."""
    state = setup_game(
        agents=[RandomAgent(rng=random.Random(seed)),
                RandomAgent(rng=random.Random(seed + 1))],
        decks=simple_test_decklists(),
        seed=seed,
    )
    return MtgGameStateAdapter(state)


def _drive_a_few_turns(adapter: MtgGameStateAdapter, n_steps: int = 30):
    """Walk the adapter forward for up to ``n_steps`` random decisions
    (or until terminal) so we end up at a non-trivial mid-game state."""
    rng = random.Random(0)
    for _ in range(n_steps):
        if adapter.is_terminal():
            return
        legal = list(adapter.legal_decisions())
        if not legal:
            return
        adapter.apply(rng.choice(legal))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_stable_repeats_for_same_adapter():
    a = _adapter(seed=1)
    _drive_a_few_turns(a)
    assert canonical_state_key(a) == canonical_state_key(a)
    # And invariant under a no-op identity check across re-reads.
    keys = [canonical_state_key(a) for _ in range(5)]
    assert len(set(keys)) == 1


def test_clone_preserves_key():
    """Cloning the adapter produces a structurally identical state, so
    the canonical key must match. This is the foundational invariant
    the per-search cache relies on: when MCTS clones a state at a
    transposition, the cache lookup must hit."""
    a = _adapter(seed=2)
    _drive_a_few_turns(a)
    b = a.clone()
    assert canonical_state_key(a) == canonical_state_key(b)


def test_clone_then_diverge_produces_different_keys():
    """A second sanity check: clone, then drive one further random
    decision on the clone — the keys must now differ."""
    a = _adapter(seed=3)
    _drive_a_few_turns(a)
    b = a.clone()
    legal = list(b.legal_decisions())
    if len(legal) < 2:
        pytest.skip("forced-move-only state; cannot diverge")
    rng = random.Random(0)
    # Pick a non-pass when available — passes can be no-ops at certain
    # priority moments (auto-pass heuristics).
    nonpass = [d for d in legal if not isinstance(d, PassPriority)]
    pick = rng.choice(nonpass if nonpass else legal)
    b.apply(pick)
    assert canonical_state_key(a) != canonical_state_key(b)


# ---------------------------------------------------------------------------
# Equivalence (must hash equal)
# ---------------------------------------------------------------------------


def test_card_id_permutation_in_hand_is_equivalent():
    """Two equivalent same-printing copies in the hand: swapping their
    raw ids (and order in the hand list) must NOT change the key."""
    a = _adapter(seed=4)
    state = a.engine_state
    p = state.players[0]
    if len(p.hand) < 2:
        pytest.skip("need at least 2 cards in hand")
    # Pick the first two same-printing pair, if any.
    by_name: dict[str, list[GameObject]] = {}
    for c in p.hand:
        by_name.setdefault(c.printing.name, []).append(c)
    pair = next((v for v in by_name.values() if len(v) >= 2), None)
    if pair is None:
        pytest.skip("hand has no two same-printing copies")
    c1, c2 = pair[0], pair[1]
    key_before = canonical_state_key(a)
    # Swap their ids AND their positions in the hand list. Both swaps
    # are pure-history under the canonical rules; hash must be stable.
    c1.id, c2.id = c2.id, c1.id
    i1, i2 = p.hand.index(c1), p.hand.index(c2)
    p.hand[i1], p.hand[i2] = p.hand[i2], p.hand[i1]
    key_after = canonical_state_key(a)
    assert key_before == key_after


def test_mana_pool_list_order_is_irrelevant():
    """Mana pool list order doesn't change the canonical key: only the
    multiset of (color, restriction) coins matters."""
    a = _adapter(seed=5)
    p = a.engine_state.players[0]
    p.mana_pool = [ManaCoin(color="W"), ManaCoin(color="U"), ManaCoin(color="W")]
    key_before = canonical_state_key(a)
    p.mana_pool = [ManaCoin(color="W"), ManaCoin(color="W"), ManaCoin(color="U")]
    key_after = canonical_state_key(a)
    assert key_before == key_after
    # And reversing.
    p.mana_pool = list(reversed(p.mana_pool))
    assert canonical_state_key(a) == key_before


def test_continuous_effects_list_order_is_irrelevant():
    """Two registries with the same effect-content in different list
    order must hash equal — the engine sorts by (layer, sublayer,
    timestamp) at apply-time, so order in the list itself doesn't
    affect downstream play."""
    a = _adapter(seed=6)
    state = a.engine_state
    e1 = ContinuousEffect(
        source_id=999, layer=6, timestamp=1, kind="grant_keyword",
        params=("flying",),
    )
    e2 = ContinuousEffect(
        source_id=999, layer=7, timestamp=2, kind="pump_pt",
        params=(1, 1),
    )
    state.continuous_effects = [e1, e2]
    key_a = canonical_state_key(a)
    state.continuous_effects = [e2, e1]
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_continuous_effects_with_mixed_none_optional_fields_sort_cleanly():
    """Regression: prior bug where a registry holding two effects with
    different None/int patterns on optional fields (target_id,
    expires_at, expires_at_player_id) crashed canonical_state_key with
    ``TypeError: '<' not supported between NoneType and int`` during
    the per-effect ``tuple(sorted(...))``. Reproduced live during the
    100-game distributed self-play bench on 2026-05-07. Fix: replace
    None with -1 sentinel inside _canonical_continuous_effect so the
    per-effect tuples stay homogeneous and sortable."""
    a = _adapter(seed=11)
    state = a.engine_state
    # One effect targets a permanent (target_id=int); the other is
    # global (target_id=None). Same layer so the sort has to compare
    # past layer into the target_id slot and trigger the bug.
    targeted = ContinuousEffect(
        source_id=42, layer=6, timestamp=1, kind="grant_keyword",
        params=("flying",), target_id=999,
    )
    untargeted = ContinuousEffect(
        source_id=42, layer=6, timestamp=2, kind="grant_keyword",
        params=("haste",), target_id=None,
    )
    state.continuous_effects = [targeted, untargeted]
    # Must not raise. Was: TypeError: '<' not supported between
    # instances of 'NoneType' and 'int'.
    key_a = canonical_state_key(a)
    # And reordering the registry must still hash equal.
    state.continuous_effects = [untargeted, targeted]
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_hand_list_order_is_irrelevant():
    """Hand canonicalization sorts by card-key tuple, so reordering
    the hand list must NOT change the key."""
    a = _adapter(seed=7)
    p = a.engine_state.players[0]
    if len(p.hand) < 2:
        pytest.skip("need at least 2 cards in hand")
    key_before = canonical_state_key(a)
    p.hand = list(reversed(p.hand))
    assert canonical_state_key(a) == key_before


def test_attached_to_canonicalization_for_equivalent_targets():
    """An aura/equipment whose ``attached_to`` points at one of two
    structurally-equivalent permanents must hash identically regardless
    of which raw-id copy it picked. Construct a minimal scenario:
    two identical permanents on the battlefield, plus an "attached"
    object whose attached_to is each of them in turn — keys must match.
    """
    a = _adapter(seed=8)
    state = a.engine_state
    # Take any two cards from p0's hand and clone them onto the
    # battlefield as creatures we control. Bypass the engine's apply
    # paths since we only need structural equivalence to test the
    # canonicalization logic.
    p0 = state.players[0]
    if len(p0.hand) < 2:
        pytest.skip("not enough cards to construct equivalent permanents")
    src = p0.hand[0]
    # Two structurally identical copies on the battlefield with
    # different raw ids.
    o1 = GameObject(
        id=10001, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    o2 = GameObject(
        id=10002, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    aura = GameObject(
        id=10003, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.extend([o1, o2, aura])
    # First key: aura attached to o1.
    aura.attached_to = o1.id
    key_attached_to_o1 = canonical_state_key(a)
    # Now flip the attachment to o2 (the structurally equivalent
    # copy). Keys must be equal — the canonical-index remap
    # collapses both attachment targets to the same canonical
    # battlefield slot.
    aura.attached_to = o2.id
    key_attached_to_o2 = canonical_state_key(a)
    assert key_attached_to_o1 == key_attached_to_o2


# ---------------------------------------------------------------------------
# Distinguishment (must hash differently)
# ---------------------------------------------------------------------------


def test_pass_count_distinguishes():
    a = _adapter(seed=9)
    key_before = canonical_state_key(a)
    a._pass_count += 1
    key_after = canonical_state_key(a)
    assert key_before != key_after


def test_combat_phase_distinguishes():
    a = _adapter(seed=10)
    key_before = canonical_state_key(a)
    a._combat_phase = "attackers"
    key_after = canonical_state_key(a)
    assert key_before != key_after


def test_pending_attackers_distinguishes():
    a = _adapter(seed=11)
    key_before = canonical_state_key(a)
    a._pending_attackers = [123, 456]
    key_after = canonical_state_key(a)
    assert key_before != key_after


def test_chosen_attackers_distinguishes():
    a = _adapter(seed=12)
    key_before = canonical_state_key(a)
    a._chosen_attackers = [7]
    assert canonical_state_key(a) != key_before


def test_pending_blockers_distinguishes():
    a = _adapter(seed=13)
    key_before = canonical_state_key(a)
    a._pending_blockers = [42]
    assert canonical_state_key(a) != key_before


def test_chosen_blocks_distinguishes():
    a = _adapter(seed=14)
    key_before = canonical_state_key(a)
    a._chosen_blocks = {1: 2}
    assert canonical_state_key(a) != key_before


def test_resolution_pause_distinguishes():
    a = _adapter(seed=15)
    key_before = canonical_state_key(a)
    a._resolution_paused = True
    a._resolution_kind = "target"
    a._resolution_player = 0
    a._resolution_options = (1, 2, 3)
    assert canonical_state_key(a) != key_before


def test_life_total_distinguishes():
    a = _adapter(seed=16)
    key_before = canonical_state_key(a)
    a.engine_state.players[0].life -= 1
    assert canonical_state_key(a) != key_before


def test_turn_number_distinguishes():
    a = _adapter(seed=17)
    key_before = canonical_state_key(a)
    a.engine_state.turn_number += 1
    assert canonical_state_key(a) != key_before


def test_mana_pool_color_distinguishes():
    a = _adapter(seed=18)
    p = a.engine_state.players[0]
    p.mana_pool = [ManaCoin(color="W")]
    key_w = canonical_state_key(a)
    p.mana_pool = [ManaCoin(color="U")]
    key_u = canonical_state_key(a)
    assert key_w != key_u


def test_battlefield_counters_distinguishes():
    """Two states with identical printings on the battlefield but
    different counter counts must hash differently."""
    a = _adapter(seed=19)
    state = a.engine_state
    p0 = state.players[0]
    if not p0.hand:
        pytest.skip("need a card to add to battlefield")
    src = p0.hand[0]
    obj = GameObject(
        id=99001, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(obj)
    obj.plus_one_counters = 0
    key_zero = canonical_state_key(a)
    obj.plus_one_counters = 2
    key_two = canonical_state_key(a)
    assert key_zero != key_two


def test_continuous_effect_content_distinguishes():
    """Two registries with the SAME LENGTH but different effect content
    must hash differently. (The scout's hash collapsed to ``len(...)``
    — production hash must walk content.)"""
    a = _adapter(seed=20)
    state = a.engine_state
    state.continuous_effects = [
        ContinuousEffect(
            source_id=1, layer=6, timestamp=1, kind="grant_keyword",
            params=("flying",),
        )
    ]
    key_flying = canonical_state_key(a)
    state.continuous_effects = [
        ContinuousEffect(
            source_id=1, layer=6, timestamp=1, kind="grant_keyword",
            params=("haste",),
        )
    ]
    key_haste = canonical_state_key(a)
    assert key_flying != key_haste


def test_stack_target_distinguishes():
    """Two stack items differing only in their target permanent must
    hash differently — even after canonicalization, distinct-printing
    targets remap to distinct canonical indices."""
    a = _adapter(seed=21)
    state = a.engine_state
    p0 = state.players[0]
    if len(p0.hand) < 2:
        pytest.skip("need at least 2 distinct printings")
    # Find two distinct-printing cards.
    c1 = p0.hand[0]
    c2 = next(
        (c for c in p0.hand[1:] if c.printing.name != c1.printing.name),
        None,
    )
    if c2 is None:
        pytest.skip("need two distinct printings")
    o1 = GameObject(
        id=2001, printing=c1.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    o2 = GameObject(
        id=2002, printing=c2.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.extend([o1, o2])
    spell_src = c1
    state.stack = [
        StackItem(
            source=spell_src, controller_id=p0.id,
            targets=(("permanent", o1.id),),
        )
    ]
    key_target_o1 = canonical_state_key(a)
    state.stack = [
        StackItem(
            source=spell_src, controller_id=p0.id,
            targets=(("permanent", o2.id),),
        )
    ]
    key_target_o2 = canonical_state_key(a)
    assert key_target_o1 != key_target_o2


def test_step_distinguishes():
    """Same state but different ``state.step`` must hash differently."""
    from mtg_bot.turn import Step
    a = _adapter(seed=22)
    state = a.engine_state
    state.step = Step.PRECOMBAT_MAIN
    key_pre = canonical_state_key(a)
    state.step = Step.POSTCOMBAT_MAIN
    key_post = canonical_state_key(a)
    assert key_pre != key_post


def test_priority_player_distinguishes():
    a = _adapter(seed=23)
    state = a.engine_state
    state.priority_player_idx = 0
    key_p0 = canonical_state_key(a)
    state.priority_player_idx = 1
    key_p1 = canonical_state_key(a)
    assert key_p0 != key_p1


def test_zone_distinguishes_same_printing():
    """Same printing in different zones must hash differently."""
    a = _adapter(seed=24)
    state = a.engine_state
    p = state.players[0]
    if not p.hand:
        pytest.skip("need a card in hand")
    src = p.hand[0]
    # Move it from hand to graveyard.
    p.hand.remove(src)
    src.zone = Zone.GRAVEYARD
    p.graveyard.append(src)
    key_in_grave = canonical_state_key(a)
    # Move back.
    p.graveyard.remove(src)
    src.zone = Zone.HAND
    p.hand.append(src)
    key_in_hand = canonical_state_key(a)
    assert key_in_grave != key_in_hand


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_returns_32_byte_digest():
    a = _adapter(seed=25)
    key = canonical_state_key(a)
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_accepts_raw_engine_state():
    """The hash function takes either an adapter or a raw engine state.
    For a raw engine state, the adapter-block tail is empty; the hash
    is well-defined but DIFFERENT from the same state wrapped in an
    adapter (since the empty adapter-block != the populated one)."""
    a = _adapter(seed=26)
    raw_state = a.engine_state
    # Both should produce 32-byte digests.
    key_adapter = canonical_state_key(a)
    key_raw = canonical_state_key(raw_state)
    assert isinstance(key_raw, bytes) and len(key_raw) == 32
    # They differ because the adapter has _pass_count etc. which the
    # raw state lacks.
    assert key_adapter != key_raw


def test_marshal_path_handles_engine_objects_in_delayed_triggers():
    """``state.delayed_triggers`` can hold dicts whose values are
    arbitrary engine effect objects (e.g. ``WarpExileSelf``). These
    are not directly marshallable, so the hash function must coerce
    them — a regression test for the bug where switching the
    serializer from ``repr`` to ``marshal`` initially crashed the
    ``test_batched_puct_cache_hits_skip_batch`` integration test
    with ``ValueError: unmarshallable object`` because the
    delayed-trigger walk left the effect instance as-is in the
    canonical tuple."""
    from mtg_bot.effects.generic.exile import WarpExileSelf

    a = _adapter(seed=99)
    # Inject a delayed trigger holding a non-primitive effect value.
    a.engine_state.delayed_triggers.append({
        "kind": "leaves_battlefield_then",
        "source_id": 12345,
        "effect": WarpExileSelf(),
    })
    # The hash must not raise.
    k = canonical_state_key(a)
    assert isinstance(k, bytes) and len(k) == 32
    # And it must be stable (deterministic across repeated calls
    # despite the non-primitive value).
    assert canonical_state_key(a) == k


def test_canonical_state_tuple_is_inspectable():
    """The internal tuple is exposed for tests that want to peer at
    structural equivalence rather than just the digest."""
    a = _adapter(seed=27)
    tup = _canonical_state_tuple(a)
    assert isinstance(tup, tuple)
    # Match the digest's deterministic shape: feed the inspectable
    # tuple through ``marshal.dumps`` (the production serializer) and
    # blake2b, mirroring ``canonical_state_key``. Marshal replaced an
    # earlier ``repr(tup).encode()`` round-trip; the public API
    # contract documented in the module docstring is "build the tuple,
    # serialize it, hash it" — this test asserts the chosen
    # serializer is what the module exposes.
    import hashlib
    import marshal
    expected = hashlib.blake2b(marshal.dumps(tup), digest_size=32).digest()
    assert canonical_state_key(a) == expected


# ---------------------------------------------------------------------------
# Optional-field None-vs-value sort safety (audit class)
# ---------------------------------------------------------------------------
#
# These tests guard the bug class identified during the 2026-05-07 audit:
# fields typed ``Optional[<scalar>]`` that end up as ``None`` for some
# elements and a ``str`` / ``int`` for others within the same sortable
# iterable. Python tuple comparison walks elements positionally, and
# raises ``TypeError: '<' not supported between instances of 'NoneType'
# and 'str'/'int'`` the first time the comparison hits a mismatched slot.
#
# The fix in each case is to coerce ``None`` to a sentinel (``-1`` for
# Optional[int]; ``""`` for Optional[str]) BEFORE the value enters the
# sort, so the per-element tuples stay homogeneous. Real values cannot
# collide with the sentinels — ids/counters/turn numbers are always
# >= 0, and real string tags (color codes, restriction names, printing
# names) are always non-empty strings.
#
# Each test below constructs a state where two elements differ at the
# Optional slot, drops them into the relevant zone / registry, and
# asserts that ``canonical_state_key`` doesn't raise. We also assert
# the equivalence rule that motivated the original sort still holds
# (reordering the inputs produces the same digest).


def test_mana_pool_with_mixed_none_spend_restriction_sorts_cleanly():
    """Regression: ``ManaCoin.spend_restriction`` is ``Optional[str]``;
    a pool holding one unrestricted coin (``spend_restriction=None``)
    plus one restricted coin (``spend_restriction="artifact_spell"``)
    used to crash ``canonical_state_key`` inside
    ``_canonical_mana_pool``'s ``tuple(sorted(Counter(...).items()))``
    when the sort compared the two ``((color, restriction), count)``
    pairs and hit the None/str slot. Surfaced live in iter_005
    self-play (1/1000 games crashed). Fix: coerce None to ``""``
    before counting."""
    a = _adapter(seed=21)
    p = a.engine_state.players[0]
    # Same color so the sort has to break ties on the restriction slot.
    unrestricted = ManaCoin(color="W", spend_restriction=None)
    restricted = ManaCoin(color="W", spend_restriction="artifact_spell")
    p.mana_pool = [unrestricted, restricted]
    # Must not raise. Was: TypeError: '<' not supported between
    # instances of 'NoneType' and 'str'.
    key_a = canonical_state_key(a)
    # Reordering the pool list must still hash equal.
    p.mana_pool = [restricted, unrestricted]
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_mana_pool_with_three_distinct_restrictions_sorts_cleanly():
    """Heavier variant: a pool with three coins covering None, plus
    two distinct restriction tags. Exercises a deeper sort that has
    to compare every pair of items, not just two."""
    a = _adapter(seed=22)
    p = a.engine_state.players[0]
    p.mana_pool = [
        ManaCoin(color="U", spend_restriction=None),
        ManaCoin(color="U", spend_restriction="legendary_spell"),
        ManaCoin(color="U", spend_restriction="artifact_spell"),
    ]
    # Must not raise.
    key_a = canonical_state_key(a)
    # And reversing.
    p.mana_pool = list(reversed(p.mana_pool))
    assert canonical_state_key(a) == key_a


def test_battlefield_index_with_mixed_none_painter_chosen_color_sorts_cleanly():
    """Regression: ``GameObject.painter_chosen_color`` is
    ``Optional[str]`` — None on every non-Painter permanent, "W"/"U"/
    etc. on a Painter source after its ETB color choice resolves.
    ``_shape_key`` feeds this slot into the canonical-battlefield-index
    sort, and a board with one Painter (color set) plus one
    structurally-similar non-Painter (color=None) used to crash with
    ``TypeError: '<' not supported between 'NoneType' and 'str'`` when
    the two shapes tied through the field's prefix. Fix: coerce None
    to ``""`` inside ``_shape_key``."""
    a = _adapter(seed=23)
    state = a.engine_state
    p0 = state.players[0]
    if not p0.hand:
        pytest.skip("need a card to clone onto the battlefield")
    src = p0.hand[0]
    # Two near-identical permanents — same printing, same controller,
    # same scalar fields — but one carries painter_chosen_color="R"
    # and the other carries None. The shape-key sort has to compare
    # past the prefix into the painter slot.
    with_color = GameObject(
        id=20001, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        painter_chosen_color="R",
    )
    without_color = GameObject(
        id=20002, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        painter_chosen_color=None,
    )
    state.battlefield.extend([with_color, without_color])
    # Must not raise.
    key_a = canonical_state_key(a)
    # Reordering on the battlefield must still hash equal — the
    # canonical-index sort is structural.
    state.battlefield[-2], state.battlefield[-1] = (
        state.battlefield[-1], state.battlefield[-2],
    )
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_battlefield_index_with_mixed_none_controller_id_before_aura_swap_sorts_cleanly():
    """Regression: ``GameObject.controller_id_before_any_aura_swap`` is
    ``Optional[int]`` — None until a layer-2 control-change captures
    the baseline. Two near-identical permanents where one has the
    baseline captured (an int) and the other doesn't (None) used to
    crash the shape-key sort with ``TypeError: '<' not supported
    between 'NoneType' and 'int'``. Fix: coerce None to ``-1`` inside
    ``_shape_key`` (real player ids are always >= 0)."""
    a = _adapter(seed=24)
    state = a.engine_state
    p0 = state.players[0]
    if not p0.hand:
        pytest.skip("need a card to clone onto the battlefield")
    src = p0.hand[0]
    captured = GameObject(
        id=20011, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        controller_id_before_any_aura_swap=p0.id,
    )
    not_captured = GameObject(
        id=20012, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        controller_id_before_any_aura_swap=None,
    )
    state.battlefield.extend([captured, not_captured])
    # Must not raise.
    key_a = canonical_state_key(a)
    state.battlefield[-2], state.battlefield[-1] = (
        state.battlefield[-1], state.battlefield[-2],
    )
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_battlefield_index_with_mixed_none_original_printing_sorts_cleanly():
    """Regression: ``GameObject.original_printing`` is
    ``Optional[CardPrinting]`` — None unless a "becomes a copy" effect
    has previously mutated ``printing`` and snapshotted the original.
    ``_shape_key`` projects this to ``original_printing.name`` (or
    None when absent). Two near-identical permanents — one with a
    snapshotted name string, one without — used to crash the
    shape-key sort with ``TypeError: '<' not supported between
    'NoneType' and 'str'``. Fix: coerce None to ``""`` inside
    ``_shape_key``."""
    a = _adapter(seed=25)
    state = a.engine_state
    p0 = state.players[0]
    if not p0.hand:
        pytest.skip("need a card to clone onto the battlefield")
    src = p0.hand[0]
    # Use a different printing for the snapshot, sourced from the
    # opponent's hand if available, else the same printing (the test
    # only needs the snapshot to be non-None — its content doesn't
    # otherwise affect the bug).
    p1 = state.players[1]
    snapshot_printing = (
        p1.hand[0].printing if p1.hand else src.printing
    )
    with_snapshot = GameObject(
        id=20021, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        original_printing=snapshot_printing,
    )
    without_snapshot = GameObject(
        id=20022, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        original_printing=None,
    )
    state.battlefield.extend([with_snapshot, without_snapshot])
    # Must not raise.
    key_a = canonical_state_key(a)
    state.battlefield[-2], state.battlefield[-1] = (
        state.battlefield[-1], state.battlefield[-2],
    )
    key_b = canonical_state_key(a)
    assert key_a == key_b


def test_shape_key_sentinels_distinguish_from_real_values():
    """The sentinel choices (``-1`` for Optional[int], ``""`` for
    Optional[str]) must NOT collide with any real value, otherwise a
    state with the sentinel would hash equal to a state with a real
    value matching the sentinel. Sanity check: two boards differing
    only in painter_chosen_color (None vs "R") must hash differently —
    the sentinel coerces None to "" not to "R", so the keys diverge."""
    a = _adapter(seed=26)
    state = a.engine_state
    p0 = state.players[0]
    if not p0.hand:
        pytest.skip("need a card to clone onto the battlefield")
    src = p0.hand[0]
    obj = GameObject(
        id=20031, printing=src.printing, owner_id=p0.id,
        controller_id=p0.id, zone=Zone.BATTLEFIELD,
        painter_chosen_color=None,
    )
    state.battlefield.append(obj)
    key_none = canonical_state_key(a)
    obj.painter_chosen_color = "R"
    key_red = canonical_state_key(a)
    # None-coerced-to-"" != "R", so the keys must diverge.
    assert key_none != key_red


def test_mana_pool_sentinel_distinguishes_from_real_restriction():
    """Companion of the shape-key sentinel test for the mana-pool fix:
    two pools differing only in spend_restriction (None vs
    "artifact_spell") must hash differently."""
    a = _adapter(seed=27)
    p = a.engine_state.players[0]
    p.mana_pool = [ManaCoin(color="W", spend_restriction=None)]
    key_none = canonical_state_key(a)
    p.mana_pool = [ManaCoin(color="W", spend_restriction="artifact_spell")]
    key_restricted = canonical_state_key(a)
    assert key_none != key_restricted
