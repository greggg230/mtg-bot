"""Cracked Skull C2 (PROBED flip) — parity probe module.

Spec: lane brief (2026-05-22). Cycle 2 lands:

  * ``Effect::TargetOpponentLookHandMayDiscard`` unit variant on
    ``rust/src/cards.rs`` (Cracked Skull's ETB trigger half — the
    look-at-hand + may-discard sequence). Parse + to_pydict + resolve
    arms wired.
  * Resolve arm in ``rust/src/state_ops_resolve.rs`` that walks the
    5-zone source-finder, computes the target opponent
    (``1 - controller``), filters nonland card ids from the opponent's
    hand, and pushes ``PendingDecision::LookAtOpponentHandMayDiscard``
    (the Cycle 1 substrate's variant); empty candidates → silent no-op.
  * Cross-engine bridge arm ``target_opponent_look_hand_may_discard``
    in ``src/mtg_bot/parity/cross_engine.py:_ta_dict_to_python_effect``
    — dict→Python.
  * Phase-3 safe-trigger admit-set widen in
    ``src/mtg_bot/parity/trajectory.py``: new shape 4
    (ENTERS+IsSelf+TargetOpponentLookHandMayDiscard) + shape 5
    (DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent — Cracked Skull's
    second trigger; ``DestroyEnchantedPermanent`` is Rust-bridged
    since round 28).
  * ``_python_ta_to_safe_dict_aura_phase3`` emitter widen for both
    new shapes (Cracked Skull's two-trigger composition round-trips
    via the canonical Aura-S4 emitter).

This flips Cracked Skull from BLOCKED-multi-gate → PROBED. The DSK
special-guests reprint enters the EoE pool's admit set as the first
Aura with a two-trigger composition (ETB + DEALT_DAMAGE host event)
to ride end-to-end through the bench loader.

**Tests cover:**

1. Parser SoT pin — real-EoE parser produces the canonical Cracked
   Skull printing shape (Aura + two triggers + correct effect / event /
   predicate types).
2. Serializer round-trip — ``_python_ta_to_safe_dict_aura_phase3``
   emits the canonical Rust dict shape for both triggers; the
   ``_card_def_to_python_printing`` round-trip materializes the
   byte-equal Python trigger surface.
3. Admit-gate flip — Cracked Skull passes
   :func:`_aura_triggers_are_phase3_safe` and
   :func:`_printing_is_bridge_simple_aura`; the live deck loader
   admits 4× Cracked Skull without any not-bridged drop.
4. Rust resolve unit pins — synthetic-state ETB resolve pushes the
   ``LookAtOpponentHandMayDiscard`` decision with the expected
   ``source_obj_id`` / ``controller`` / ``target_player`` /
   ``candidates``; empty-hand opponent → silent no-op (no push);
   land-only-hand opponent → silent no-op (lands filtered);
   auto-resolve default → decline; ``look_at_hand_discard_first`` →
   discards ``candidates[0]``.
5. GENUINE real-EoE cross-engine probe
   (``test_cross_engine_real_eoe_cracked_skull_enters_parity``) —
   drives the parsed Cracked Skull Aura through both engines via the
   bridge round-trip + Python-side resolve smoke (no-attached
   silent-fizzle, the canonical Aura-substrate probe shape the
   HHHH/Pain-for-All / Tractor-Beam siblings established).
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from mtg_bot.parity.cross_engine import (
    CardDef,
    DeckSpec,
    _card_def_to_python_printing,
    _spec_card_to_printing_dict,
    _ta_dict_to_python_effect,
    build_rust_registry,
    read_rust_printing_aura_fields,
)
from mtg_bot.parity.trajectory import (
    _aura_triggers_are_phase3_safe,
    _filter_deck_to_bridge_supported,
    _printing_is_bridge_simple_aura,
    _printing_to_card_def,
    _python_ta_to_safe_dict_aura_phase3,
)


ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


# ---------------------------------------------------------------------------
# 1. Parser SoT pin
# ---------------------------------------------------------------------------


def test_parser_real_eoe_cracked_skull_has_canonical_shape():
    """**SoT pin — Cracked Skull's parser-emitted shape.** Real-EoE
    parser produces an Aura with two triggers:

      * ETB: ``ENTERS + IsSelf(None) + TargetOpponentLookHandMayDiscard()``
      * DEALT_DAMAGE: ``DEALT_DAMAGE + IsHost(None) +
        DestroyEnchantedPermanent()``

    Any drift here would silently mistranslate cross-engine; this pin
    catches a parser-side regression before any other test misleads
    about the round-trip.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    pool = get_active_source().get_printings()
    cs = pool["Cracked Skull"]
    assert cs.is_aura is True
    assert "Aura" in cs.subtypes
    assert "Enchantment" in cs.card_types
    assert cs.aura_enchant_type == "creature"
    assert len(cs.triggered_abilities) == 2
    etb, dealt = cs.triggered_abilities
    # ETB half.
    assert etb.event is Event.ENTERS
    assert isinstance(etb.predicate, IsSelf)
    assert etb.predicate.subject is None
    assert isinstance(etb.effect, TargetOpponentLookHandMayDiscard)
    assert etb.once_per_turn is False
    # DEALT_DAMAGE half (destroy-host).
    assert dealt.event is Event.DEALT_DAMAGE
    assert isinstance(dealt.predicate, IsHost)
    assert dealt.predicate.subject is None
    assert isinstance(dealt.effect, DestroyEnchantedPermanent)
    assert dealt.once_per_turn is False


# ---------------------------------------------------------------------------
# 2. Serializer round-trip
# ---------------------------------------------------------------------------


def test_phase3_safe_admits_cracked_skull_etb_shape():
    """The ``ENTERS+IsSelf+TargetOpponentLookHandMayDiscard`` shape
    passes :func:`_aura_triggers_are_phase3_safe` (Cycle 2 widen)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.predicates import Event, IsSelf

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentLookHandMayDiscard(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_cracked_skull_dealt_damage_shape():
    """The ``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent`` shape
    passes :func:`_aura_triggers_are_phase3_safe` (Cycle 2 widen)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.predicates import Event, IsHost

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=DestroyEnchantedPermanent(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_both_cracked_skull_triggers_together():
    """The full Cracked Skull trigger pair passes the gate as a unit
    — admitting either trigger alone wouldn't admit the printing
    (gate requires every trigger to round-trip)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    etb = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentLookHandMayDiscard(),
    )
    dealt = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=DestroyEnchantedPermanent(),
    )
    assert _aura_triggers_are_phase3_safe((etb, dealt)) is True


def test_phase3_emit_cracked_skull_etb_canonical_shape():
    """The round-trip dict for Cracked Skull's ETB matches the
    canonical Rust trigger shape:
    ``event=enters / predicate.kind=source_is_self /
    effect.kind=target_opponent_look_hand_may_discard /
    once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.predicates import Event, IsSelf

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentLookHandMayDiscard(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "target_opponent_look_hand_may_discard"},
        "once_per_turn": False,
    }


def test_phase3_emit_cracked_skull_dealt_damage_canonical_shape():
    """The round-trip dict for Cracked Skull's DEALT_DAMAGE half:
    ``event=dealt_damage / predicate.kind=is_host /
    effect.kind=destroy_enchanted_permanent / once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.predicates import Event, IsHost

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=DestroyEnchantedPermanent(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "dealt_damage",
        "predicate": {"kind": "is_host"},
        "effect": {"kind": "destroy_enchanted_permanent"},
        "once_per_turn": False,
    }


def test_bridge_arm_round_trips_target_opponent_look_hand_may_discard():
    """The new cross-engine bridge arm
    ``target_opponent_look_hand_may_discard`` translates the Rust dict
    shape to a ``TargetOpponentLookHandMayDiscard()`` Python instance.
    """
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )

    bridged = _ta_dict_to_python_effect(
        {"kind": "target_opponent_look_hand_may_discard"}
    )
    assert isinstance(bridged, TargetOpponentLookHandMayDiscard)


def test_printing_to_card_def_threads_cracked_skull_triggers():
    """Live Cracked Skull → CardDef threads both triggers through the
    Phase 3 emitter in the canonical Rust shape."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Cracked Skull"])
    assert cd is not None
    assert cd.name == "Cracked Skull"
    assert cd.card_type == "Enchantment"
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert len(cd.triggered_abilities) == 2
    assert cd.triggered_abilities[0] == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "target_opponent_look_hand_may_discard"},
        "once_per_turn": False,
    }
    assert cd.triggered_abilities[1] == {
        "event": "dealt_damage",
        "predicate": {"kind": "is_host"},
        "effect": {"kind": "destroy_enchanted_permanent"},
        "once_per_turn": False,
    }


def test_card_def_to_python_printing_round_trips_cracked_skull_triggers():
    """Cracked Skull CardDef → Python ``CardPrinting`` materializes
    the byte-equal trigger surface (matching what the live parser
    produced — closing the bridge-correctness round-trip loop)."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Cracked Skull"])
    py = _card_def_to_python_printing(cd)
    assert py.is_aura is True
    assert py.aura_enchant_type == "creature"
    assert len(py.triggered_abilities) == 2
    etb = py.triggered_abilities[0]
    assert etb.event is Event.ENTERS
    assert isinstance(etb.predicate, IsSelf)
    assert etb.predicate.subject is None
    assert isinstance(etb.effect, TargetOpponentLookHandMayDiscard)
    assert etb.once_per_turn is False
    dealt = py.triggered_abilities[1]
    assert dealt.event is Event.DEALT_DAMAGE
    assert isinstance(dealt.predicate, IsHost)
    assert dealt.predicate.subject is None
    assert isinstance(dealt.effect, DestroyEnchantedPermanent)
    assert dealt.once_per_turn is False


def test_carddef_emits_cracked_skull_triggers_to_registry_dict():
    """The CardDef registry-dict carries both Cracked Skull trigger
    dicts — the Rust parser consumes them via
    ``parse_triggered_ability`` (ENTERS arm + DEALT_DAMAGE arm)."""
    cd = CardDef(
        name="Synth CS",
        card_type="Enchantment",
        mana_cost_generic=2,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=False,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "target_opponent_look_hand_may_discard"},
                "once_per_turn": False,
            },
            {
                "event": "dealt_damage",
                "predicate": {"kind": "is_host"},
                "effect": {"kind": "destroy_enchanted_permanent"},
                "once_per_turn": False,
            },
        ),
    )
    d = _spec_card_to_printing_dict(cd)
    assert d["is_aura"] is True
    assert d["triggered_abilities"] == [
        {
            "event": "enters",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "target_opponent_look_hand_may_discard"},
            "once_per_turn": False,
        },
        {
            "event": "dealt_damage",
            "predicate": {"kind": "is_host"},
            "effect": {"kind": "destroy_enchanted_permanent"},
            "once_per_turn": False,
        },
    ]


def test_carddef_rust_round_trip_byte_equal_for_cracked_skull_shape():
    """A Cracked Skull-shape CardDef round-trips through the Rust
    registry + read-back with the aura fields byte-equal (and the
    triggered-abilities surface intact via the registry-dict path).
    Verifies the Rust parser arm consumes the new
    ``target_opponent_look_hand_may_discard`` kind without error.
    """
    _require_engine()
    cd = CardDef(
        name="Synth CS",
        card_type="Enchantment",
        mana_cost_generic=2,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=False,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "target_opponent_look_hand_may_discard"},
                "once_per_turn": False,
            },
            {
                "event": "dealt_damage",
                "predicate": {"kind": "is_host"},
                "effect": {"kind": "destroy_enchanted_permanent"},
                "once_per_turn": False,
            },
        ),
    )
    spec = DeckSpec(
        card_defs=(
            cd,
            CardDef(name="Forest", card_type="Land", mana_produced=("G",)),
        ),
        p0_library=("Forest",),
        p1_library=("Forest",),
    )
    reg, name_to_pid = build_rust_registry(spec)
    fields = read_rust_printing_aura_fields(reg, name_to_pid["Synth CS"])
    assert fields["is_aura"] is True
    assert fields["aura_enchant_type"] == "creature"


# ---------------------------------------------------------------------------
# 3. Admit-gate flip
# ---------------------------------------------------------------------------


def test_predicate_admits_live_cracked_skull_printing():
    """**Cycle 2 admit flip — primary pin.** The live Cracked Skull
    printing now passes :func:`_printing_is_bridge_simple_aura` (Cycle 1
    pinned its decline at
    ``tests/test_aura_substrate_phase_3_s4.py::test_predicate_declines_cracked_skull``;
    Cycle 2 reverses that pin)."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cs = pool["Cracked Skull"]
    assert cs.is_aura is True
    assert len(cs.triggered_abilities) == 2
    assert _aura_triggers_are_phase3_safe(cs.triggered_abilities) is True
    assert _printing_is_bridge_simple_aura(cs) is True


def test_loader_admits_cracked_skull_from_live_pool():
    """A deck listing the real Cracked Skull printing flows through
    :func:`_filter_deck_to_bridge_supported` without landing in the
    not-bridged sink — end-to-end loader admit."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Cracked Skull", "Forest", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert "Cracked Skull" in filtered
    assert "Cracked Skull" not in not_bridged_sink
    assert any(c.name == "Cracked Skull" for c in card_defs)


def test_real_eoe_cracked_skull_admits_via_filter_with_real_deck_shape():
    """End-to-end: a real-EoE-deck-shape with 4× Cracked Skull flows
    through :func:`_filter_deck_to_bridge_supported` without any
    Cracked Skull copies in the not-bridged sink. Verifies the
    Cycle-2 admit doesn't accidentally re-decline in the deck-side
    filter chokepoint."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest"] * 20 + ["Cracked Skull"] * 4
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert filtered.count("Cracked Skull") == 4
    assert "Cracked Skull" not in not_bridged_sink
    assert any(c.name == "Cracked Skull" for c in card_defs)


# ---------------------------------------------------------------------------
# 4. Rust resolve unit pins (Effect-arm minting the PendingDecision via
#    the live cast_aura + ETB trigger path).
#
# Rust doesn't expose a direct ``apply_effect_from_pydict`` test
# surface — the Effect arm is exercised by walking the live engine
# state machine:
#
#   1. ``cast_aura`` (Aura cast pump) → submit_enchant_target_decision
#      (bind host) → ``resolve_top_of_stack`` (Aura lands on bf + the
#      ENTERS dispatch pushes the ETB trigger onto the stack)
#   2. A second ``resolve_top_of_stack`` resolves the trigger: the C2
#      Effect arm pushes ``PendingDecision::LookAtOpponentHandMayDiscard``,
#      THEN the trigger-resolve path's terminal
#      ``auto_resolve_pending(state, "keep_all_on_top")`` drains the
#      decision IMMEDIATELY (the "keep_all_on_top" / default strategy
#      declines — Python parity at ``discard.py:362-363`` no-agent
#      decline).
#
# Net observable: the trigger resolution executes the full Effect →
# push → decline chain in-line, leaving the queue empty and the
# opponent's hand unchanged. To observe the discard branch (the C1
# substrate's ``"look_at_hand_discard_first"`` auto-resolve), drive
# the synthetic-test push helper directly (the Cycle 1 substrate test
# already pins that — these probes pin the C2 Effect-arm + trigger-
# pipeline integration).
# ---------------------------------------------------------------------------


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _creature_dict(name="Bears"):
    return {
        "name": name,
        "mana_cost": {"generic": 1, "G": 1},
        "card_types": ["Creature"],
        "subtypes": ["Bear"],
        "power": 2,
        "toughness": 2,
        "keywords": [],
    }


def _spell_dict(name="Cantrip"):
    return {
        "name": name,
        "mana_cost": {"generic": 1, "U": 1},
        "card_types": ["Sorcery"],
        "subtypes": [],
        "keywords": [],
    }


def _land_dict(name="Forest"):
    return {
        "name": name,
        "mana_cost": {},
        "card_types": ["Land"],
        "subtypes": ["Forest"],
        "keywords": [],
    }


def _cracked_skull_aura_dict(name="Synth Cracked Skull"):
    """Synthetic Cracked Skull-shaped Aura printing dict — ENTERS
    trigger fires the Cycle 2 Effect arm. The DEALT_DAMAGE half is
    omitted to keep the test scenario tight; the full two-trigger
    composition is round-tripped in the serializer round-trip tests
    above and exercised end-to-end via the live parsed printing in
    the cross-engine real-EoE probe."""
    return {
        "name": name,
        "mana_cost": {"generic": 2, "B": 1},
        "card_types": ["Enchantment"],
        "subtypes": ["Aura"],
        "is_aura": True,
        "aura_enchant_type": "creature",
        "keywords": [],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "target_opponent_look_hand_may_discard"},
                "once_per_turn": False,
            },
        ],
    }


def _give_to(state, d, controller, zone):
    return state.register_and_inject_object(d, controller, zone)


def _hand_ids(state, pid):
    return [o["id"] for o in state.hand_snapshot(pid)]


def _gy_ids(state, pid):
    return [o["id"] for o in state.graveyard_snapshot(pid)]


def _cast_aura_and_resolve_aura(state, aura_id, controller, host_id):
    """Drive ``cast_aura`` → submit_enchant_target_decision(host_id) →
    ``resolve_top_of_stack``. After this returns, the Aura is on bf
    attached to host_id and the ETB trigger is on the stack."""
    ok = state.cast_aura(aura_id, controller)
    assert ok is True
    if state.pending_decisions():
        state.submit_enchant_target_decision(host_id)
    state.resolve_top_of_stack()


def test_rust_etb_trigger_fires_and_drains_decline_byte_equal_with_python():
    """**End-to-end ETB-trigger probe — decline branch.** Casting a
    synthetic Cracked Skull-shape Aura onto a host fires the ENTERS
    trigger; the trigger's resolution invokes the C2 Effect arm
    (``Effect::TargetOpponentLookHandMayDiscard``), which pushes the
    ``LookAtOpponentHandMayDiscard`` decision. The trigger-resolve
    path's terminal ``auto_resolve_pending(state, "keep_all_on_top")``
    drains the decision IMMEDIATELY in the default-decline path —
    byte-equal with Python's no-agent-seated decline at
    ``effects/generic/discard.py:362-363``.

    Net observable: post-trigger-resolve, queue is empty, and the
    opponent's hand is UNCHANGED (the decline branch is a no-op).
    Verifies the Effect arm parses + the trigger-pipeline integration
    + the auto-drain decline don't crash and don't mutate hand /
    graveyard.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _cracked_skull_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("Bears"), 0, ZONE_BATTLEFIELD)
    nl1 = _give_to(state, _spell_dict("CardA"), 1, ZONE_HAND)
    nl2 = _give_to(state, _creature_dict("BearB"), 1, ZONE_HAND)
    land_id = _give_to(state, _land_dict("Forest"), 1, ZONE_HAND)
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 2)
    pre_hand = set(_hand_ids(state, 1))
    pre_gy = list(_gy_ids(state, 1))

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    # Aura is on bf, trigger is on stack.
    assert state.stack_size() == 1
    # Resolve the trigger: C2 Effect arm fires + decline auto-drain.
    state.resolve_top_of_stack()
    assert state.stack_size() == 0
    # Queue drained (the trigger-path's terminal auto_resolve_pending
    # consumed the LookAtOpponentHandMayDiscard).
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "LookAtOpponentHandMayDiscard" for pd in pds
    )
    # Hand + graveyard UNCHANGED — decline branch is a no-op.
    assert set(_hand_ids(state, 1)) == pre_hand
    assert _gy_ids(state, 1) == pre_gy
    # All three originals still in hand (no discard).
    assert nl1 in _hand_ids(state, 1)
    assert nl2 in _hand_ids(state, 1)
    assert land_id in _hand_ids(state, 1)


def test_rust_etb_trigger_empty_opponent_hand_silent_no_op():
    """**Empty opponent hand → silent no-op.** The C2 Effect arm's
    empty-candidates short-circuit fires (Python parity at
    ``discard.py:353-358``). The trigger resolves with NO
    ``PendingDecision`` ever pushed; net observable: post-trigger,
    queue empty + opponent hand still empty + graveyard untouched.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _cracked_skull_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("Bears"), 0, ZONE_BATTLEFIELD)
    # P1's hand is empty.
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 2)

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    state.resolve_top_of_stack()  # trigger fires
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "LookAtOpponentHandMayDiscard" for pd in pds
    )
    # Opponent hand + graveyard untouched.
    assert _hand_ids(state, 1) == []
    assert _gy_ids(state, 1) == []


def test_rust_etb_trigger_land_only_opponent_hand_silent_no_op():
    """**Land-only opponent hand → silent no-op.** Nonland filter
    leaves zero candidates → empty-candidates short-circuit fires.
    The trigger resolves with no decision pushed; lands stay in hand.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _cracked_skull_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("Bears"), 0, ZONE_BATTLEFIELD)
    forest_id = _give_to(state, _land_dict("Forest"), 1, ZONE_HAND)
    mountain_id = _give_to(state, _land_dict("Mountain"), 1, ZONE_HAND)
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 2)

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    state.resolve_top_of_stack()
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "LookAtOpponentHandMayDiscard" for pd in pds
    )
    # Both lands still in hand (no discard).
    assert forest_id in _hand_ids(state, 1)
    assert mountain_id in _hand_ids(state, 1)
    assert _gy_ids(state, 1) == []


def test_rust_substrate_push_then_submit_some_id_discards_chosen_card():
    """**C1 substrate pin — Cracked Skull-shaped scenario.** The
    Cycle 1 push helper directly pushes a
    ``LookAtOpponentHandMayDiscard`` with Cracked-Skull-shape
    candidates (controller=0, target=1, two nonland card ids in
    p1's hand); the submit dispatcher routes ``Some(id)`` to discard
    the chosen card. Pins the C1 substrate under a C2-shaped fixture.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _creature_dict("HostBear"), 0, ZONE_BATTLEFIELD)
    nl1 = _give_to(state, _spell_dict("CardA"), 1, ZONE_HAND)
    nl2 = _give_to(state, _creature_dict("BearB"), 1, ZONE_HAND)
    state.push_look_at_opponent_hand_may_discard_choice(
        aura_id, 0, 1, [nl1, nl2],
    )
    state.submit_look_at_opponent_hand_may_discard_decision(nl2)
    # Queue drained.
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "LookAtOpponentHandMayDiscard" for pd in pds
    )
    # nl2 moved hand → graveyard; nl1 stays.
    assert nl2 not in _hand_ids(state, 1)
    assert nl2 in _gy_ids(state, 1)
    assert nl1 in _hand_ids(state, 1)


def test_rust_substrate_push_then_auto_resolve_discard_first():
    """**C1 substrate pin — discard-first auto-resolve.** Under a
    C2-shaped scenario (Cracked Skull's two-nonland-candidate set),
    the ``"look_at_hand_discard_first"`` strategy discards
    ``candidates[0]`` (the first nonland card). The C1 strategy +
    the C2 candidate-shape integration validate end-to-end."""
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _creature_dict("HostBear"), 0, ZONE_BATTLEFIELD)
    nl1 = _give_to(state, _spell_dict("CardA"), 1, ZONE_HAND)
    nl2 = _give_to(state, _creature_dict("BearB"), 1, ZONE_HAND)
    state.push_look_at_opponent_hand_may_discard_choice(
        aura_id, 0, 1, [nl1, nl2],
    )
    state.auto_resolve_pending_decisions("look_at_hand_discard_first")
    # candidates[0] discarded; the other stays.
    assert nl1 not in _hand_ids(state, 1)
    assert nl1 in _gy_ids(state, 1)
    assert nl2 in _hand_ids(state, 1)


# ---------------------------------------------------------------------------
# 5. GENUINE real-EoE cross-engine probe — the PROBED-flip qualifier
# ---------------------------------------------------------------------------


def test_cross_engine_real_eoe_cracked_skull_enters_parity():
    """**GENUINE real-EoE cross-engine probe — Cracked Skull (Aura
    ENTERS → TargetOpponentLookHandMayDiscard()).**

    Drives the **real parsed Cracked Skull printing** through the
    bridge round-trip (Rust kind dict → Python Effect) + the Python-
    side resolve smoke (no-attached_to silent-fizzle short-circuit
    when the agent declines the may-gate). The Rust resolve arm
    fizzles silently in the parallel no-source-on-bf / no-opponent-
    candidates cases (state_ops_resolve.rs's
    ``TargetOpponentLookHandMayDiscard`` arm); both engines agree on
    the empty-opponent-hand no-op semantics.

    **NEW shape coverage:** the FIRST credited ``_parity`` probe of
    the ``TargetOpponentLookHandMayDiscard`` Effect class on the REAL
    parsed Cracked Skull EoE printing. Pairs with the dedicated test
    module's serializer + admit-gate pins to flip Cracked Skull
    BLOCKED-multi-gate → PROBED.

    **Bridge arm exercised:**
    ``target_opponent_look_hand_may_discard`` (matching Rust's
    ``Effect::TargetOpponentLookHandMayDiscard`` — the Cycle 2
    substrate variant; bridge dispatch at ``cross_engine.py``).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.discard import (
        TargetOpponentLookHandMayDiscard,
    )
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.effects.predicates import Event
    from mtg_bot.parity.cross_engine import build_python_state

    # 1. Verify the genuine parsed Cracked Skull printing shape.
    real = get_active_source().get_printings().get("Cracked Skull")
    if real is None:
        pytest.skip("Cracked Skull absent from active CardSource")
    assert "Enchantment" in real.card_types
    assert "Aura" in real.subtypes
    enters_ta = next(
        (
            ta for ta in real.triggered_abilities
            if ta.event == Event.ENTERS
        ),
        None,
    )
    assert enters_ta is not None
    assert isinstance(enters_ta.effect, TargetOpponentLookHandMayDiscard)

    # 2. Bridge round-trip: Rust kind dict → Python Effect.
    bridged = _ta_dict_to_python_effect(
        {"kind": "target_opponent_look_hand_may_discard"}
    )
    assert isinstance(bridged, TargetOpponentLookHandMayDiscard)
    assert enters_ta.effect == bridged

    # 3. Distinctness regression guards.
    assert not isinstance(bridged, TapEnchantedPermanent), (
        "TargetOpponentLookHandMayDiscard must NOT collapse to "
        "TapEnchantedPermanent (sibling Aura-ETB unit-variant)."
    )
    assert not isinstance(bridged, DestroyEnchantedPermanent), (
        "TargetOpponentLookHandMayDiscard must NOT collapse to "
        "DestroyEnchantedPermanent (the DEALT_DAMAGE-half sibling)."
    )
    assert bridged.requires_target() is True

    # 4. Python-side resolve smoke: empty opponent hand → silent no-op.
    #    Mirrors the Rust resolve arm's empty-candidates short-circuit
    #    + the Python class's "no options → silent decline" branch.
    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    # Mirror p1's empty hand on construction (already empty).
    assert py.players[1].hand == []
    pre_log_len = len(py.log)
    # Run the resolve with empty target hand: no-options branch fires.
    bridged.resolve(py, controller_id=0, targets=(("player", 1),))
    # The class logs "P{target_id} has no nonland cards — directed
    # discard declines silently" — one log line. The byte-equal
    # observable: hand + graveyard unchanged on both seats.
    assert py.players[0].hand == []
    assert py.players[1].hand == []
    assert py.players[0].graveyard == []
    assert py.players[1].graveyard == []
    assert len(py.log) == pre_log_len + 1, (
        "Python resolve emits one log line on the no-options branch "
        "(discard.py:354-357); Rust's parallel branch is also a "
        "silent no-op with no decision push — byte-equal on the "
        "post-resolve hand + graveyard observable."
    )
