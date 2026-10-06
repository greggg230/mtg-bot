"""Tests for the v2 (``"canonical_v2"``) action-encoding slot mapping.

The contract:

- v1 (``"positional_v1"``, default) is bit-exact with the
  pre-canonical behavior: positions in hand / graveyard / exile come
  from the engine's raw zone list.
- v2 (``"canonical_v2"``) sorts hand / graveyard / exile by
  ``(printing.name, modifier_tuple..., card.id)`` before slot
  assignment. Two states differing only in card.id permutation produce
  IDENTICAL slot mappings under v2 (the bug v1 has).
- ``encode`` / ``decode`` round-trip under both versions on the same
  underlying state.

These tests are version-axis-of-the-codec. The pre-existing
``test_action_encoding.py`` covers the v1-default behavior on real
games at end-to-end scale; this file zooms in on the v1-vs-v2
difference and the version-validation surface.
"""

from __future__ import annotations

import random

import pytest

from mtg_bot.actions import CastSpell, PassPriority, PlayLand
from mtg_bot.agents import RandomAgent
from mtg_bot.search import HeuristicsConfig, MtgGameStateAdapter
from mtg_bot.search.protocol import CHANCE_PLAYER
from mtg_bot.simulate import setup_game
from mtg_bot.training.action_encoding import (
    ACTION_ENCODING_VERSION_V1,
    ACTION_ENCODING_VERSION_V2,
    ACTION_VOCAB_SIZE,
    SLOT_LAND_FROM_HAND_BASE,
    SLOT_PASS,
    SUPPORTED_ACTION_ENCODING_VERSIONS,
    decode,
    encode,
    legal_action_mask,
)
from tests._test_decks import simple_test_decklists


def _build_adapter(seed: int) -> MtgGameStateAdapter:
    state = setup_game(
        [RandomAgent(rng=random.Random(seed + 1)),
         RandomAgent(rng=random.Random(seed + 2))],
        decks=simple_test_decklists(), seed=seed,
    )
    return MtgGameStateAdapter(
        state, expose_combat_decisions=True,
        heuristics=HeuristicsConfig(
            phase_prune_enabled=False,
            force_landdrop_enabled=False,
        ),
    )


def test_version_constants_exposed() -> None:
    """The supported versions are public module-level constants and
    the supported tuple lists exactly those three (v3 = v2 + the
    resolution block, ``docs/RESOLUTION_DECISIONS.md``)."""
    from mtg_bot.training.action_encoding import ACTION_ENCODING_VERSION_V3
    assert ACTION_ENCODING_VERSION_V1 == "positional_v1"
    assert ACTION_ENCODING_VERSION_V2 == "canonical_v2"
    assert ACTION_ENCODING_VERSION_V3 == "canonical_v3"
    assert set(SUPPORTED_ACTION_ENCODING_VERSIONS) == {
        ACTION_ENCODING_VERSION_V1, ACTION_ENCODING_VERSION_V2,
        ACTION_ENCODING_VERSION_V3,
    }


def test_unknown_version_raises() -> None:
    """encode / decode / legal_action_mask each reject an unknown
    version string up front (early loud failure beats silent
    fall-through to v1)."""
    adapter = _build_adapter(seed=0)
    viewer = adapter.active_player()
    with pytest.raises(ValueError, match="action_encoding_version"):
        encode(PassPriority(), adapter, viewer, version="bogus")
    with pytest.raises(ValueError, match="action_encoding_version"):
        decode(SLOT_PASS, adapter, viewer, version="bogus")
    with pytest.raises(ValueError, match="action_encoding_version"):
        legal_action_mask(adapter, viewer, version="bogus")


def test_v1_default_matches_explicit_v1() -> None:
    """When no version is passed, the codec uses
    ``ACTION_ENCODING_VERSION_V1`` — back-compat with v13/v15/v16
    which were trained against this slot mapping. No-version-arg call
    must equal explicit ``version="positional_v1"`` call bit-for-bit
    on every legal decision in a real game's first many priority
    moments."""
    adapter = _build_adapter(seed=42)
    rng = random.Random(123)
    for _ in range(80):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        for d in legal:
            slot_default = encode(d, adapter, active)
            slot_v1 = encode(
                d, adapter, active,
                version=ACTION_ENCODING_VERSION_V1,
            )
            assert slot_default == slot_v1, (
                f"default-version slot diverges from explicit v1 on {d!r}"
            )
        # And the mask too.
        mask_default = legal_action_mask(adapter, active)
        mask_v1 = legal_action_mask(
            adapter, active, version=ACTION_ENCODING_VERSION_V1,
        )
        assert (mask_default == mask_v1).all()
        adapter.apply(rng.choice(legal))


@pytest.mark.parametrize("version", [
    ACTION_ENCODING_VERSION_V1, ACTION_ENCODING_VERSION_V2,
])
def test_round_trip_under_each_version(version: str) -> None:
    """``decode(encode(d, s, v), s, v) == d`` for every legal decision
    at every priority moment — under both versions."""
    adapter = _build_adapter(seed=7)
    rng = random.Random(7000)
    for _ in range(100):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        for d in legal:
            slot = encode(d, adapter, active, version=version)
            assert 0 <= slot < ACTION_VOCAB_SIZE
            d2 = decode(slot, adapter, active, version=version)
            assert d2 == d, (
                f"v={version} roundtrip fail: {d!r} -> {slot} -> {d2!r}"
            )
        adapter.apply(rng.choice(legal))


def _sorted_card_ids(cards: list) -> list[int]:
    return sorted(c.id for c in cards)


def test_v2_invariant_under_card_id_permutation() -> None:
    """The load-bearing v2 contract: two states differing ONLY by a
    permutation of card.ids in the viewer's hand produce IDENTICAL
    v2-encoded slots for the same logical decision (PlayLand on an
    arbitrary hand card). v1, in contrast, returns different slots —
    that's the latent bug the v2 mode was designed to fix."""
    adapter = _build_adapter(seed=11)
    # Drive a few decisions deep so the hand has some structure.
    rng = random.Random(11000)
    for _ in range(8):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        adapter.apply(rng.choice(legal))
    if adapter.is_terminal():
        pytest.skip("game ended before any non-chance decision")

    viewer = adapter.active_player()
    if viewer == CHANCE_PLAYER:
        viewer = 0
    eng = adapter.engine_state
    hand = list(eng.player_by_id(viewer).hand)
    if len(hand) < 2:
        pytest.skip("hand too small to permute")

    # Decide on a target card to PlayLand (the actual decision content
    # is irrelevant — we just need a Decision whose slot depends on the
    # card's hand position).
    target = hand[0]
    decision = PlayLand(card_id=target.id)

    # Snapshot v1 + v2 slots in the original ordering.
    v1_orig = encode(
        decision, adapter, viewer,
        version=ACTION_ENCODING_VERSION_V1,
    )
    v2_orig = encode(
        decision, adapter, viewer,
        version=ACTION_ENCODING_VERSION_V2,
    )

    # Permute card.ids: clone the adapter and reverse hand order in
    # the cloned engine. The cards themselves are the same objects;
    # only their position in p.hand changes. v1 encoding is
    # position-by-list-index, so the slot must shift. v2 sorts by
    # canonical key (printing.name + modifier flags + card.id), so a
    # mere list reversal that doesn't change the canonical sort order
    # leaves the v2 slot unchanged.
    clone = adapter.clone()
    clone_eng = clone.engine_state
    clone_p = clone_eng.player_by_id(viewer)
    # Rebuild the hand in reversed engine-order. The objects in the
    # cloned engine have the same .id values as ``hand`` (clone
    # preserves ids); we just shuffle them.
    clone_p.hand[:] = list(reversed(clone_p.hand))

    v1_perm = encode(
        decision, clone, viewer,
        version=ACTION_ENCODING_VERSION_V1,
    )
    v2_perm = encode(
        decision, clone, viewer,
        version=ACTION_ENCODING_VERSION_V2,
    )

    # v2: same canonical sort order in both adapters (same set of
    # printings and modifier tuples) → same slot.
    assert v2_perm == v2_orig, (
        f"v2 slot drifted under hand permutation: {v2_orig} -> {v2_perm}"
    )
    # v1: positional, so the reversed list moves the target card to a
    # different slot. (If the hand only had one card we'd skip; the
    # earlier guard ensures len(hand) >= 2.)
    assert v1_perm != v1_orig, (
        "v1 slot did NOT shift under hand permutation — that means "
        "the bug this fix was written to address has gone away "
        "underneath us; rerun the test with a more meaningful "
        "permutation."
    )


def test_v2_canonical_sort_uses_printing_name_then_id() -> None:
    """v2 sort key is ``(printing.name, modifier_flags..., card.id)``.
    Same printing copies break the tie via card.id (smaller id ==
    smaller slot index) — load-bearing for deterministic decoder
    behavior. Builds a synthetic hand of two same-printing cards via
    a real game state and verifies the slot order matches the id
    order."""
    adapter = _build_adapter(seed=3)
    viewer = adapter.active_player()
    if viewer == CHANCE_PLAYER:
        viewer = 0
    eng = adapter.engine_state
    hand = list(eng.player_by_id(viewer).hand)

    # Find two same-printing cards in the opening hand. A simple deck
    # with a 60-card mono-color list has plenty of duplicates.
    name_to_cards: dict[str, list] = {}
    for c in hand:
        name_to_cards.setdefault(c.printing.name, []).append(c)
    duplicates = next(
        (lst for lst in name_to_cards.values() if len(lst) >= 2), None,
    )
    if duplicates is None:
        pytest.skip("opening hand has no duplicate-printing cards")

    a, b = duplicates[0], duplicates[1]
    # Whichever card has the smaller id is canonically the smaller-
    # slot card under v2.
    smaller, larger = (a, b) if a.id < b.id else (b, a)
    s_smaller = encode(
        PlayLand(card_id=smaller.id), adapter, viewer,
        version=ACTION_ENCODING_VERSION_V2,
    )
    s_larger = encode(
        PlayLand(card_id=larger.id), adapter, viewer,
        version=ACTION_ENCODING_VERSION_V2,
    )
    assert s_smaller < s_larger, (
        f"v2 tie-break (id ascending) violated: smaller-id slot "
        f"{s_smaller} not less than larger-id slot {s_larger}"
    )
    # Both must lie in the LAND_FROM_HAND range.
    assert SLOT_LAND_FROM_HAND_BASE <= s_smaller < (
        SLOT_LAND_FROM_HAND_BASE + 16
    )
    assert SLOT_LAND_FROM_HAND_BASE <= s_larger < (
        SLOT_LAND_FROM_HAND_BASE + 16
    )


def test_legal_action_mask_v2_matches_legal_decisions() -> None:
    """Mask consistency under v2: every True slot decodes to a legal
    decision, and every legal decision encodes to a True slot.
    Pre-existing v1 mask test covers v1; this is the v2 mirror."""
    import numpy as np
    adapter = _build_adapter(seed=13)
    rng = random.Random(13_000)
    for _ in range(80):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        expected = {
            encode(d, adapter, active, version=ACTION_ENCODING_VERSION_V2)
            for d in legal
        }
        mask = legal_action_mask(
            adapter, active, version=ACTION_ENCODING_VERSION_V2,
        )
        actual = set(np.flatnonzero(mask).tolist())
        assert expected == actual, (
            f"v2 mask mismatch: expected {sorted(expected)}, got "
            f"{sorted(actual)}"
        )
        legal_set = set(legal)
        for slot in actual:
            d = decode(
                slot, adapter, active,
                version=ACTION_ENCODING_VERSION_V2,
            )
            assert d in legal_set, (
                f"v2 slot {slot} flagged legal but decodes to {d!r} "
                f"not in legal set"
            )
        adapter.apply(rng.choice(legal))


def test_v1_v2_share_pass_slot() -> None:
    """``SLOT_PASS`` is version-agnostic — it's slot 0, no zone
    indexing involved. Round-trip and slot equality both hold."""
    adapter = _build_adapter(seed=99)
    viewer = adapter.active_player()
    for version in SUPPORTED_ACTION_ENCODING_VERSIONS:
        slot = encode(PassPriority(), adapter, viewer, version=version)
        assert slot == SLOT_PASS
        assert decode(slot, adapter, viewer, version=version) == (
            PassPriority()
        )


def test_v2_legal_decisions_unaffected_by_engine_hand_reorder() -> None:
    """Functional consequence of canonicality: the SET of legal slots
    under v2 must be invariant under a hand-list reorder that doesn't
    change the canonical zone tuple. (v1 mask shifts because the slot
    indices shift; v2 mask stays put.) This is what makes v2 weights
    immune to the slot_fingerprint workaround being a no-op."""
    import numpy as np
    adapter = _build_adapter(seed=5)
    # Step a few moves so the hand has interesting content.
    rng = random.Random(5_000)
    for _ in range(6):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        adapter.apply(rng.choice(legal))
    viewer = adapter.active_player()
    if viewer == CHANCE_PLAYER:
        viewer = 0
    if adapter.is_terminal() or len(
        adapter.engine_state.player_by_id(viewer).hand,
    ) < 2:
        pytest.skip("setup didn't produce a multi-card hand")

    mask_orig_v2 = legal_action_mask(
        adapter, viewer, version=ACTION_ENCODING_VERSION_V2,
    )
    clone = adapter.clone()
    clone.engine_state.player_by_id(viewer).hand[:] = list(reversed(
        clone.engine_state.player_by_id(viewer).hand,
    ))
    mask_perm_v2 = legal_action_mask(
        clone, viewer, version=ACTION_ENCODING_VERSION_V2,
    )
    assert (mask_orig_v2 == mask_perm_v2).all(), (
        "v2 mask drifted under engine-hand reorder — canonicality "
        "broken"
    )


def test_decode_round_trips_after_permutation_under_v2() -> None:
    """Drive home: under v2, encoding a decision in state S and
    decoding the resulting slot in state S' (where S' is a card.id-
    permutation of S) recovers the same decision IFF the same card is
    still present (canonical-equivalent) in S'. v1 returns the wrong
    card; v2 returns the right one."""
    adapter = _build_adapter(seed=21)
    rng = random.Random(21_000)
    # Drive the game until viewer 0 has multiple cards in hand. Only
    # one driving step needed in practice (opening hand has 7).
    for _ in range(2):
        if adapter.is_terminal():
            break
        active = adapter.active_player()
        if active == CHANCE_PLAYER:
            outs = adapter.chance_outcomes()
            adapter.apply(rng.choices(
                [o for o, _ in outs], [w for _, w in outs],
            )[0])
            continue
        legal = adapter.legal_decisions()
        adapter.apply(rng.choice(legal))
    viewer = 0
    eng = adapter.engine_state
    hand = list(eng.player_by_id(viewer).hand)
    if len(hand) < 2:
        pytest.skip("hand too small")

    target = hand[1]  # not the first, so list-position differs from sort
    decision = PlayLand(card_id=target.id)
    v2_slot = encode(
        decision, adapter, viewer,
        version=ACTION_ENCODING_VERSION_V2,
    )

    clone = adapter.clone()
    clone.engine_state.player_by_id(viewer).hand[:] = list(reversed(
        clone.engine_state.player_by_id(viewer).hand,
    ))
    decoded_v2 = decode(
        v2_slot, clone, viewer, version=ACTION_ENCODING_VERSION_V2,
    )
    assert decoded_v2 == decision, (
        f"v2 decode after permutation mismatch: {decoded_v2!r} != "
        f"{decision!r}"
    )

    # And the v1 baseline confirms the bug exists (decoder returns the
    # wrong card under permutation). If this assertion ever flips, the
    # v1 bug has gone away on its own and v2 mode is no longer needed
    # — the suite should fail loud and ask the maintainer to verify.
    v1_slot = encode(
        decision, adapter, viewer,
        version=ACTION_ENCODING_VERSION_V1,
    )
    decoded_v1_in_clone = decode(
        v1_slot, clone, viewer, version=ACTION_ENCODING_VERSION_V1,
    )
    assert decoded_v1_in_clone != decision, (
        "v1 decoder no longer returns the wrong card under "
        "permutation — surprising; reconfirm the v2 motivation"
    )
