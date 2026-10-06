"""Unit tests for ``mtg_bot.search.resolution_slots`` -- the
``canonical_v3`` resolution-block layout and the per-pause option ->
slot assignment the channel agent records at pause time."""

from __future__ import annotations

import logging

import pytest

from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.game import StackItem, Zone
from mtg_bot.search import resolution_slots as rs
from mtg_bot.simulate import setup_game
from mtg_bot.training import action_encoding as ae
from tests._test_decks import simple_test_decklists


@pytest.fixture(autouse=True)
def _fresh_counters():
    rs.reset_counts()
    yield
    rs.reset_counts()


def _state():
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(),
        seed=3,
    )
    state.battlefield.clear()
    return state


def _creature(state, name, owner, **flags):
    obj = state.new_object(SYNTHETIC_CREATURES[name], owner, Zone.BATTLEFIELD)
    for k, v in flags.items():
        setattr(obj, k, v)
    state.battlefield.append(obj)
    return obj


def _card(state, name, owner, zone):
    printing = BASIC_LANDS.get(name) or SYNTHETIC_CREATURES[name]
    obj = state.new_object(printing, owner, zone)
    getattr(state.players[owner], {
        Zone.HAND: "hand", Zone.GRAVEYARD: "graveyard",
        Zone.EXILE: "exile", Zone.LIBRARY: "library",
    }[zone]).append(obj)
    return obj


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------


def test_block_size_is_pinned():
    assert rs.RESOLUTION_BLOCK_SIZE == 288
    assert ae.ACTION_VOCAB_SIZE_V3 == ae.ACTION_VOCAB_SIZE + 288 == 805


def test_families_tile_the_block_without_gaps_or_overlap():
    spans = sorted(rs.FAMILIES.values())
    cursor = 0
    for offset, size in spans:
        assert offset == cursor, (offset, cursor)
        assert size > 0
        cursor = offset + size
    assert cursor == rs.RESOLUTION_BLOCK_SIZE
    assert rs.CARD_FROM_LIBRARY_DECLINE == (
        rs.FAMILIES["card_from_library"][0] + rs.LIBRARY_OPTION_CAP
    )


def test_caps_match_the_canonical_v2_constants():
    assert rs.BF_CAP == ae.BATTLEFIELD_OWN_CAP
    assert rs.GY_CAP == ae.GRAVEYARD_CAP
    assert rs.EXILE_CAP == ae.EXILE_CAP
    assert rs.HAND_CAP == ae.HAND_SIZE_CAP


# ---------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------


def test_target_players_are_relative_to_the_decider():
    state = _state()
    offs = rs.compute_offsets(
        "target", state, 1, (("player", 1), ("player", 0)),
    )
    assert offs == (rs.TGT_PLAYER, rs.TGT_PLAYER + 1)


def test_target_battlefield_uses_canonical_per_side_positions():
    state = _state()
    # Entry order deliberately differs from the canonical (name) order.
    b = _creature(state, "Wild Bear", 0)
    a = _creature(state, "Sapling", 0)
    o = _creature(state, "Raging Goblin", 1)
    offs = rs.compute_offsets(
        "target", state, 0,
        (("creature", a.id), ("creature", b.id), ("creature", o.id)),
    )
    # Sapling < Wild Bear by name -> own positions 0, 1; opponent 0.
    assert offs == (rs.TGT_BF_OWN, rs.TGT_BF_OWN + 1, rs.TGT_BF_OPP)
    # Same objects from the other seat: sides swap.
    offs1 = rs.compute_offsets(
        "target", state, 1, (("creature", o.id), ("creature", a.id)),
    )
    assert offs1 == (rs.TGT_BF_OWN, rs.TGT_BF_OPP)


def test_target_battlefield_order_is_invariant_to_entry_order():
    s1 = _state()
    x1 = _creature(s1, "Wild Bear", 1)
    y1 = _creature(s1, "Sapling", 1)
    s2 = _state()
    y2 = _creature(s2, "Sapling", 1)
    x2 = _creature(s2, "Wild Bear", 1)
    o1 = rs.compute_offsets("target", s1, 0, (("creature", x1.id), ("creature", y1.id)))
    o2 = rs.compute_offsets("target", s2, 0, (("creature", x2.id), ("creature", y2.id)))
    assert o1 == o2


def test_target_stack_graveyard_exile_and_decline():
    state = _state()
    spell_src = _card(state, "Wild Bear", 1, Zone.HAND)
    state.players[1].hand.remove(spell_src)
    spell_src.zone = Zone.STACK
    state.stack.append(StackItem(source=spell_src, controller_id=1))
    g_own = _card(state, "Sapling", 0, Zone.GRAVEYARD)
    g_opp = _card(state, "Wild Bear", 1, Zone.GRAVEYARD)
    x_own = _card(state, "Raging Goblin", 0, Zone.EXILE)
    x_opp = _card(state, "Sapling", 1, Zone.EXILE)
    offs = rs.compute_offsets(
        "target", state, 0,
        (
            ("stack", 0), ("graveyard_creature", g_own.id),
            ("card_in_graveyard", g_opp.id), ("exile_card", x_own.id),
            ("exile_creature", x_opp.id), None,
        ),
    )
    assert offs == (
        rs.TGT_STACK, rs.TGT_GY_OWN, rs.TGT_GY_OPP, rs.TGT_EXILE_OWN,
        rs.TGT_EXILE_OPP, rs.TGT_DECLINE,
    )


def test_any_damage_target_player_id_falls_back_to_player_slot():
    state = _state()
    # No object with id 0/1 on the battlefield or in a public zone
    # after the clear above (ids start at the deck cards).
    offs = rs.compute_offsets(
        "target", state, 0,
        (("any_damage_target", 1), ("any_damage_target", 0)),
    )
    ids_in_zones = {
        c.id for p in state.players
        for z in (p.graveyard, p.exile) for c in z
    }
    assert 0 not in ids_in_zones and 1 not in ids_in_zones
    assert offs == (rs.TGT_PLAYER + 1, rs.TGT_PLAYER)


def test_target_overflow_is_counted_and_warned_once(caplog):
    state = _state()
    hand_card = _card(state, "Sapling", 0, Zone.HAND)
    with caplog.at_level(logging.WARNING, logger=rs.__name__):
        offs = rs.compute_offsets(
            "target", state, 0,
            (("creature_card", hand_card.id), ("player", 99), ("stack", 5)),
        )
        rs.compute_offsets("target", state, 0, (("creature_card", hand_card.id),))
    assert offs == (None, None, None)
    assert rs.overflow_counts() == {"target": 4}
    assert sum("overflow (target)" in r.message for r in caplog.records) == 1


def test_battlefield_position_past_the_cap_overflows():
    state = _state()
    objs = [_creature(state, "Sapling", 1) for _ in range(rs.BF_CAP + 1)]
    offs = rs.compute_offsets(
        "target", state, 0, tuple(("creature", o.id) for o in objs),
    )
    assert offs[: rs.BF_CAP] == tuple(
        rs.TGT_BF_OPP + i for i in range(rs.BF_CAP)
    )
    assert offs[rs.BF_CAP] is None


def test_stack_position_past_the_cap_overflows():
    state = _state()
    for _ in range(rs.STACK_CAP + 1):
        src = _card(state, "Sapling", 0, Zone.HAND)
        state.players[0].hand.remove(src)
        state.stack.append(StackItem(source=src, controller_id=0))
    # stack index 0 is the bottom: STACK_CAP items above it.
    offs = rs.compute_offsets("target", state, 0, (("stack", 0),))
    assert offs == (None,)


def test_collisions_are_counted():
    state = _state()
    c = _creature(state, "Sapling", 1)
    offs = rs.compute_offsets(
        "target", state, 0, (("creature", c.id), ("permanent", c.id)),
    )
    assert offs[0] == offs[1] == rs.TGT_BF_OPP
    assert rs.collision_counts() == {"target": 1}


# ---------------------------------------------------------------------
# Other kinds
# ---------------------------------------------------------------------


def test_sacrifice_battlefield_and_card_options():
    state = _state()
    own = _creature(state, "Sapling", 0)
    opp = _creature(state, "Wild Bear", 1)
    in_exile = _card(state, "Raging Goblin", 0, Zone.EXILE)
    in_hand = _card(state, "Forest", 0, Zone.HAND)
    offs = rs.compute_offsets(
        "sacrifice", state, 0, (own.id, in_exile.id, in_hand.id, opp.id),
    )
    # Non-battlefield options rank by canonical key: Forest < Raging Goblin.
    assert offs == (
        rs.SACRIFICE, rs.SACRIFICE_CARD + 1, rs.SACRIFICE_CARD, None,
    )
    assert rs.overflow_counts() == {"sacrifice": 1}


def test_sacrifice_battlefield_position_cap():
    state = _state()
    objs = [_creature(state, "Sapling", 0) for _ in range(rs.BF_CAP + 1)]
    offs = rs.compute_offsets("sacrifice", state, 0, tuple(o.id for o in objs))
    assert offs[-1] is None and offs[0] == rs.SACRIFICE


def test_card_choices_rank_by_canonical_key():
    state = _state()
    f = _card(state, "Forest", 0, Zone.HAND)
    s = _card(state, "Sapling", 0, Zone.HAND)
    b = _card(state, "Wild Bear", 0, Zone.HAND)
    assert rs.compute_offsets("discard", state, 0, (b.id, f.id, s.id)) == (
        rs.DISCARD + 2, rs.DISCARD, rs.DISCARD + 1,
    )
    assert rs.compute_offsets(
        "card_from_opponent_hand", state, 1, (s.id, f.id),
    ) == (rs.CARD_FROM_OPP_HAND + 1, rs.CARD_FROM_OPP_HAND)
    lib = _card(state, "Wild Bear", 0, Zone.LIBRARY)
    assert rs.compute_offsets(
        "card_from_library", state, 0, (lib.id, f.id, None),
    ) == (
        rs.CARD_FROM_LIBRARY + 1, rs.CARD_FROM_LIBRARY,
        rs.CARD_FROM_LIBRARY_DECLINE,
    )


def test_card_rank_past_the_cap_and_unknown_ids():
    state = _state()
    cards = [_card(state, "Sapling", 0, Zone.HAND) for _ in range(rs.HAND_CAP + 1)]
    offs = rs.compute_offsets("discard", state, 0, tuple(c.id for c in cards))
    assert offs[-1] is None
    # An id no zone holds ranks last (by id) instead of crashing.
    offs2 = rs.compute_offsets("discard", state, 0, (987654, cards[0].id))
    assert offs2 == (rs.DISCARD + 1, rs.DISCARD)


def test_scalar_kinds():
    state = _state()
    assert rs.compute_offsets("mode", state, 0, (0, 1, 2)) == (
        rs.MODE, rs.MODE + 1, rs.MODE + 2,
    )
    assert rs.compute_offsets("mode", state, 0, (rs.MODE_CAP,)) == (None,)
    assert rs.compute_offsets("modes", state, 0, ((0,), (1,), (0, 1))) == (
        rs.MODES, rs.MODES + 1, rs.MODES + 2,
    )
    assert rs.compute_offsets("modes", state, 0, ((4,), ())) == (None, None)
    assert rs.compute_offsets("x", state, 0, (0, 20, 21)) == (
        rs.X, rs.X + 20, None,
    )
    assert rs.compute_offsets("surveil", state, 0, (True, False)) == (
        rs.SURVEIL, rs.SURVEIL + 1,
    )
    assert rs.compute_offsets("shock_life", state, 0, (True, False)) == (
        rs.SHOCK_LIFE, rs.SHOCK_LIFE + 1,
    )
    assert rs.compute_offsets("yes_no", state, 0, (True, False)) == (
        rs.YES_NO, rs.YES_NO + 1,
    )
    assert rs.compute_offsets("devour_lands", state, 0, ((), (4, 5))) == (
        rs.DEVOUR, rs.DEVOUR + 1,
    )


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown resolution kind"):
        rs.compute_offsets("nope", _state(), 0, ())


# ---------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------


def test_dedupe_cards_keeps_one_per_canonical_key():
    state = _state()
    f1 = _card(state, "Forest", 0, Zone.LIBRARY)
    f2 = _card(state, "Forest", 0, Zone.LIBRARY)
    s = _card(state, "Sapling", 0, Zone.LIBRARY)
    assert rs.dedupe_cards(state, (f1.id, f2.id, s.id, None, None)) == (
        f1.id, s.id, None,
    )
    # Unknown ids are kept (one per id).
    assert rs.dedupe_cards(state, (424242, 424242, 777)) == (424242, 777)


def test_dedupe_exact_and_find_card():
    assert rs.dedupe_exact((("a", 1), ("a", 1), ("b", 2))) == (("a", 1), ("b", 2))
    state = _state()
    c = _creature(state, "Sapling", 0)
    assert rs.find_card(state, c.id) is c
    assert rs.card_key(state, -5) is None
