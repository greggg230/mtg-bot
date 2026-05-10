"""Tests for the hand-zone action dedup in ``legal.py``.

Two cards in hand with the same printing and no per-object divergent
state are search-equivalent: the action shape (``PlayLand(card_id=N)`` or
``CastSpell(card_id=N, ...)``) and the apply-time effect are identical
modulo the surviving ``card_id``. ``legal.py`` collapses such duplicates
to a single canonical action keyed on
``(printing.name, *modifier_tuple)``, keeping the first ``card_id`` in
hand order (deterministic).

Modifier-distinguished casts (e.g., a printed-cost cast and an Impending
cast on the same Overlord) stay separate — they're different actions.

See ``docs/SEARCH_LIMITATIONS.md`` for the full failure-mode and
fix-scope writeup.
"""

from mtg_bot import legal_moves
from mtg_bot.actions import CastSpell, PlayLand
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.game import GameState, Player, Zone
from mtg_bot.turn import Step

from mtg_bot.cards.source import get_active_source

_PRINTINGS = get_active_source().get_printings()


def _new_state():
    state = GameState()
    state.players.append(Player(id=0))
    state.players.append(Player(id=1))
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = Step.PRECOMBAT_MAIN
    return state


def _put_in_hand(state, printing, owner_id=0):
    obj = state.new_object(printing, owner_id, Zone.HAND)
    state.players[owner_id].hand.append(obj)
    return obj


def _put_in_battlefield(state, printing, owner_id=0):
    obj = state.new_object(printing, owner_id, Zone.BATTLEFIELD)
    state.battlefield.append(obj)
    return obj


# -----------------------------------------------------------------------
# PlayLand: identical-printing copies in hand collapse to one action
# -----------------------------------------------------------------------


def test_three_same_basic_lands_emit_one_play_land():
    """Three identical Forests in hand → exactly 1 ``PlayLand`` action."""
    state = _new_state()
    f1 = _put_in_hand(state, BASIC_LANDS["Forest"])
    _put_in_hand(state, BASIC_LANDS["Forest"])
    _put_in_hand(state, BASIC_LANDS["Forest"])
    play_lands = [a for a in legal_moves(state) if isinstance(a, PlayLand)]
    assert len(play_lands) == 1
    # First-encountered card_id wins (hand-order preserved → deterministic).
    assert play_lands[0].card_id == f1.id


def test_play_land_dedup_keeps_first_in_hand_order():
    """Re-ordering the hand changes which copy survives — confirms the
    'first in hand order' contract isn't an accident of object id."""
    state = _new_state()
    f1 = _put_in_hand(state, BASIC_LANDS["Forest"])
    f2 = _put_in_hand(state, BASIC_LANDS["Forest"])
    # Swap hand order: f2 first, f1 second.
    state.players[0].hand[0], state.players[0].hand[1] = (
        state.players[0].hand[1], state.players[0].hand[0],
    )
    play_lands = [a for a in legal_moves(state) if isinstance(a, PlayLand)]
    assert len(play_lands) == 1
    assert play_lands[0].card_id == f2.id
    # Sanity: f1 has a smaller id but isn't picked because it's no
    # longer first in hand order.
    assert f1.id < f2.id


def test_mixed_printings_in_hand_yield_one_play_land_per_printing():
    """2 Forests + 1 Plains → 2 ``PlayLand`` actions total (one per
    printing, not one per copy)."""
    state = _new_state()
    f1 = _put_in_hand(state, BASIC_LANDS["Forest"])
    _put_in_hand(state, BASIC_LANDS["Forest"])
    plains = _put_in_hand(state, BASIC_LANDS["Plains"])
    play_lands = [a for a in legal_moves(state) if isinstance(a, PlayLand)]
    assert len(play_lands) == 2
    assert {a.card_id for a in play_lands} == {f1.id, plains.id}


# -----------------------------------------------------------------------
# CastSpell: identical-printing creatures collapse to one (sorcery speed)
# -----------------------------------------------------------------------


def test_two_same_creatures_emit_one_cast_spell():
    """Two identical Wild Bears in hand at sorcery speed with the cost
    payable → exactly one ``CastSpell`` action for that printing."""
    state = _new_state()
    # Mana: 2 Forests covers {1}{G}.
    _put_in_battlefield(state, BASIC_LANDS["Forest"])
    _put_in_battlefield(state, BASIC_LANDS["Forest"])
    bear1 = _put_in_hand(state, SYNTHETIC_CREATURES["Wild Bear"])
    _put_in_hand(state, SYNTHETIC_CREATURES["Wild Bear"])
    casts = [
        a for a in legal_moves(state)
        if isinstance(a, CastSpell)
        and not a.use_warp and not a.kicked
        and not a.cast_for_free and not a.use_impending
        and a.cast_room_half is None
    ]
    assert len(casts) == 1
    assert casts[0].card_id == bear1.id


def test_two_different_creature_printings_stay_split():
    """Two different creature printings don't dedup — the equivalence
    key includes the printing name."""
    state = _new_state()
    # Plenty of mana for both.
    for _ in range(4):
        _put_in_battlefield(state, BASIC_LANDS["Forest"])
    bear = _put_in_hand(state, SYNTHETIC_CREATURES["Wild Bear"])
    sapling = _put_in_hand(state, SYNTHETIC_CREATURES["Sapling"])
    casts = [
        a for a in legal_moves(state)
        if isinstance(a, CastSpell)
    ]
    cast_ids = {a.card_id for a in casts}
    assert bear.id in cast_ids
    assert sapling.id in cast_ids


# -----------------------------------------------------------------------
# CastSpell: modifier-distinguished casts of the same printing stay split
# -----------------------------------------------------------------------


def test_overlord_emits_both_printed_and_impending_after_dedup():
    """An Overlord card has both a printed-cost cast and an Impending
    cast available at sorcery speed. Both must survive the dedup —
    they have different modifier tuples (``use_impending=False`` vs
    ``use_impending=True``) and are distinct actions in the search
    tree."""
    state = _new_state()
    # Plenty of Plains for both costs.
    for _ in range(8):
        _put_in_battlefield(state, BASIC_LANDS["Plains"])
    overlord = _put_in_hand(state, _PRINTINGS["Overlord of the Mistmoors"])
    casts = [
        a for a in legal_moves(state)
        if isinstance(a, CastSpell) and a.card_id == overlord.id
    ]
    # Printed cost (no flags set) AND impending variant.
    has_printed = any(
        not a.use_impending and not a.use_warp and not a.kicked
        and not a.cast_for_free
        for a in casts
    )
    has_impending = any(a.use_impending for a in casts)
    assert has_printed
    assert has_impending


def test_two_overlord_copies_dedup_each_modifier_arm():
    """Two copies of the same Overlord in hand → one printed-cost cast
    AND one impending cast, NOT four (2 copies x 2 arms). Each
    modifier-tuple is independently deduped."""
    state = _new_state()
    for _ in range(8):
        _put_in_battlefield(state, BASIC_LANDS["Plains"])
    overlord1 = _put_in_hand(state, _PRINTINGS["Overlord of the Mistmoors"])
    _put_in_hand(state, _PRINTINGS["Overlord of the Mistmoors"])
    casts = [
        a for a in legal_moves(state)
        if isinstance(a, CastSpell)
        and a.card_id in {c.id for c in state.players[0].hand}
    ]
    printed = [
        a for a in casts
        if not a.use_impending and not a.use_warp and not a.kicked
        and not a.cast_for_free
    ]
    impending = [a for a in casts if a.use_impending]
    assert len(printed) == 1
    assert len(impending) == 1
    # Both arms point at the first-encountered copy.
    assert printed[0].card_id == overlord1.id
    assert impending[0].card_id == overlord1.id
