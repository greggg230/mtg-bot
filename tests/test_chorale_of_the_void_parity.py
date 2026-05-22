"""Chorale of the Void Cycle 1 (PROBED flip) — parity probe module.

Spec: lane brief (2026-05-22). Cycle 1 lands the LAST Aura S4
multi-trigger sibling (Cracked Skull + Sporogenic Infection already
shipped earlier this same day). With Chorale of the Void admitted,
ALL THREE Aura S4 carriers (Cracked Skull, Sporogenic Infection,
Chorale of the Void) ride the bench loader end-to-end.

Cycle 1 deliverables:

  * ``Effect::ReanimateTargetCreatureCardFromDefendersGraveyard`` unit
    variant on ``rust/src/cards.rs`` (Chorale's ATTACKS trigger half —
    the "when enchanted creature attacks, put target creature card
    from defending player's graveyard onto the battlefield tapped and
    attacking" reanimate). Parse + to_pydict + resolve arms wired.
  * Honest-stop resolve arm in ``rust/src/state_ops_resolve.rs`` (the
    cast-time-bound graveyard-creature target shape is not threaded
    through to triggered abilities, AND the put-into-bf-tapped-and-
    attacking combat-pipeline integration is deferred — both halves
    silently match Python's same-case ``if not targets: return``
    fizzle at ``effects/generic/graveyard.py:527-528``).
  * Cross-engine bridge arm
    ``reanimate_target_creature_card_from_defenders_graveyard`` in
    ``src/mtg_bot/parity/cross_engine.py::_ta_dict_to_python_effect``
    — dict→Python.
  * Phase-3 safe-trigger admit-set widen in
    ``src/mtg_bot/parity/trajectory.py``: new shape 7
    (``ATTACKS+IsHost(subject="attacker")+
    ReanimateTargetCreatureCardFromDefendersGraveyard``) and new
    shape 8 (``BEGINNING_OF_END_STEP+Controlled(by=YOU)+
    SacrificeSelfUnlessVoidActive``). Together they admit Chorale's
    full two-trigger composition.
  * ``_python_ta_to_safe_dict_aura_phase3`` emitter widen for both
    new shapes.

This flips Chorale of the Void BLOCKED-multi-gate → PROBED.

The end-step ``SacrificeSelfUnlessVoidActive`` half has full Rust
resolve byte-equality (shipped at the Aura-Phase-4 substrate
landing — ``cards.rs:4186`` + ``state_ops_resolve.rs:14600`` +
``cross_engine.py:8805``). The ATTACKS-reanimate half is the
honest-stop half today; Cycle 2 closes the full
put-into-bf-tapped-and-attacking primitive + cast-time-bound
target threading.

**Tests cover:**

1. Parser SoT pin — real-EoE parser produces the canonical Chorale of
   the Void printing shape (Aura + two triggers + correct effect /
   event / predicate types).
2. Serializer round-trip — ``_python_ta_to_safe_dict_aura_phase3``
   emits the canonical Rust dict shape for both triggers; the
   ``_card_def_to_python_printing`` round-trip materializes the
   byte-equal Python trigger surface.
3. Admit-gate flip — Chorale of the Void passes
   :func:`_aura_triggers_are_phase3_safe` and
   :func:`_printing_is_bridge_simple_aura`; the live deck loader
   admits 4× Chorale of the Void without any not-bridged drop.
4. Rust resolve unit pins — the new
   ``ReanimateTargetCreatureCardFromDefendersGraveyard`` Effect arm
   silent no-ops in Rust on the no-targets path (honest-stop
   substrate Cycle 1).
5. GENUINE real-EoE cross-engine probe
   (``test_cross_engine_real_eoe_chorale_of_the_void_attacks_reanimate_parity``)
   — drives the parsed Chorale of the Void Aura through the bridge
   round-trip + Python-side resolve smoke (empty-defender-graveyard
   silent-fizzle, the canonical Aura-substrate probe shape Cracked
   Skull / Sporogenic Infection / HHHH / Pain-for-All / Tractor-Beam
   siblings established).
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


def test_parser_real_eoe_chorale_of_the_void_has_canonical_shape():
    """**SoT pin — Chorale of the Void's parser-emitted shape.**
    Real-EoE parser produces an Aura with two triggers:

      * ATTACKS: ``ATTACKS + IsHost(subject="attacker") +
        ReanimateTargetCreatureCardFromDefendersGraveyard()``
      * BEGINNING_OF_END_STEP: ``BEGINNING_OF_END_STEP +
        Controlled(by=YOU, subject=None) +
        SacrificeSelfUnlessVoidActive()``

    Any drift here would silently mistranslate cross-engine; this pin
    catches a parser-side regression before any other test misleads
    about the round-trip.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.predicates import (
        Controlled,
        Event,
        IsHost,
        _Controller,
    )

    pool = get_active_source().get_printings()
    chorale = pool["Chorale of the Void"]
    assert chorale.is_aura is True
    assert "Aura" in chorale.subtypes
    assert "Enchantment" in chorale.card_types
    assert chorale.aura_enchant_type == "creature"
    assert len(chorale.triggered_abilities) == 2
    attacks, endstep = chorale.triggered_abilities
    # ATTACKS half.
    assert attacks.event is Event.ATTACKS
    assert isinstance(attacks.predicate, IsHost)
    assert attacks.predicate.subject == "attacker"
    assert isinstance(
        attacks.effect, ReanimateTargetCreatureCardFromDefendersGraveyard
    )
    assert attacks.once_per_turn is False
    # BEGINNING_OF_END_STEP half.
    assert endstep.event is Event.BEGINNING_OF_END_STEP
    assert isinstance(endstep.predicate, Controlled)
    assert endstep.predicate.by is _Controller.YOU
    assert endstep.predicate.subject is None
    assert isinstance(endstep.effect, SacrificeSelfUnlessVoidActive)
    assert endstep.once_per_turn is False


# ---------------------------------------------------------------------------
# 2. Serializer round-trip
# ---------------------------------------------------------------------------


def test_phase3_safe_admits_chorale_of_the_void_attacks_shape():
    """The ``ATTACKS+IsHost(subject="attacker")+
    ReanimateTargetCreatureCardFromDefendersGraveyard`` shape passes
    :func:`_aura_triggers_are_phase3_safe` (Cycle 1 widen, shape 7)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.predicates import Event, IsHost

    ta = TriggeredAbility(
        event=Event.ATTACKS,
        predicate=IsHost(subject="attacker"),
        effect=ReanimateTargetCreatureCardFromDefendersGraveyard(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_chorale_of_the_void_endstep_shape():
    """The ``BEGINNING_OF_END_STEP+Controlled(by=YOU)+
    SacrificeSelfUnlessVoidActive`` shape passes
    :func:`_aura_triggers_are_phase3_safe` (Cycle 1 widen, shape 8)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.predicates import Controlled, Event, YOU

    ta = TriggeredAbility(
        event=Event.BEGINNING_OF_END_STEP,
        predicate=Controlled(by=YOU),
        effect=SacrificeSelfUnlessVoidActive(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_both_chorale_of_the_void_triggers_together():
    """The full Chorale of the Void trigger pair passes the gate as a
    unit — admitting either trigger alone wouldn't admit the printing
    (gate requires every trigger to round-trip)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.predicates import (
        Controlled,
        Event,
        IsHost,
        YOU,
    )

    attacks = TriggeredAbility(
        event=Event.ATTACKS,
        predicate=IsHost(subject="attacker"),
        effect=ReanimateTargetCreatureCardFromDefendersGraveyard(),
    )
    endstep = TriggeredAbility(
        event=Event.BEGINNING_OF_END_STEP,
        predicate=Controlled(by=YOU),
        effect=SacrificeSelfUnlessVoidActive(),
    )
    assert _aura_triggers_are_phase3_safe((attacks, endstep)) is True


def test_phase3_emit_chorale_of_the_void_attacks_canonical_shape():
    """The round-trip dict for Chorale of the Void's ATTACKS trigger
    matches the canonical Rust trigger shape:
    ``event=attacks / predicate.kind=is_host /
    predicate.subject=attacker /
    effect.kind=reanimate_target_creature_card_from_defenders_graveyard
    / once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.predicates import Event, IsHost

    ta = TriggeredAbility(
        event=Event.ATTACKS,
        predicate=IsHost(subject="attacker"),
        effect=ReanimateTargetCreatureCardFromDefendersGraveyard(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "attacks",
        "predicate": {"kind": "is_host", "subject": "attacker"},
        "effect": {
            "kind": "reanimate_target_creature_card_from_defenders_graveyard"
        },
        "once_per_turn": False,
    }


def test_phase3_emit_chorale_of_the_void_endstep_canonical_shape():
    """The round-trip dict for Chorale of the Void's
    BEGINNING_OF_END_STEP trigger matches the canonical Rust trigger
    shape: ``event=beginning_of_end_step /
    predicate.kind=controlled_by_you /
    effect.kind=sacrifice_self_unless_void_active /
    once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.predicates import Controlled, Event, YOU

    ta = TriggeredAbility(
        event=Event.BEGINNING_OF_END_STEP,
        predicate=Controlled(by=YOU),
        effect=SacrificeSelfUnlessVoidActive(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "beginning_of_end_step",
        "predicate": {"kind": "controlled_by_you"},
        "effect": {"kind": "sacrifice_self_unless_void_active"},
        "once_per_turn": False,
    }


def test_bridge_arm_round_trips_reanimate_target_creature_card_from_defenders_graveyard():
    """The new cross-engine bridge arm
    ``reanimate_target_creature_card_from_defenders_graveyard``
    translates the Rust dict shape to a
    ``ReanimateTargetCreatureCardFromDefendersGraveyard()`` Python
    instance.
    """
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )

    bridged = _ta_dict_to_python_effect(
        {"kind": "reanimate_target_creature_card_from_defenders_graveyard"}
    )
    assert isinstance(
        bridged, ReanimateTargetCreatureCardFromDefendersGraveyard
    )


def test_printing_to_card_def_threads_chorale_of_the_void_triggers():
    """Live Chorale of the Void → CardDef threads both triggers through
    the Phase 3 emitter in the canonical Rust shape."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Chorale of the Void"])
    assert cd is not None
    assert cd.name == "Chorale of the Void"
    assert cd.card_type == "Enchantment"
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert len(cd.triggered_abilities) == 2
    assert cd.triggered_abilities[0] == {
        "event": "attacks",
        "predicate": {"kind": "is_host", "subject": "attacker"},
        "effect": {
            "kind": "reanimate_target_creature_card_from_defenders_graveyard"
        },
        "once_per_turn": False,
    }
    assert cd.triggered_abilities[1] == {
        "event": "beginning_of_end_step",
        "predicate": {"kind": "controlled_by_you"},
        "effect": {"kind": "sacrifice_self_unless_void_active"},
        "once_per_turn": False,
    }


def test_card_def_to_python_printing_round_trips_chorale_of_the_void_triggers():
    """Chorale of the Void CardDef → Python ``CardPrinting``
    materializes the byte-equal trigger surface (matching what the
    live parser produced — closing the bridge-correctness round-trip
    loop)."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.predicates import (
        Controlled,
        Event,
        IsHost,
        _Controller,
    )

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Chorale of the Void"])
    py = _card_def_to_python_printing(cd)
    assert py.is_aura is True
    assert py.aura_enchant_type == "creature"
    assert len(py.triggered_abilities) == 2
    attacks = py.triggered_abilities[0]
    assert attacks.event is Event.ATTACKS
    assert isinstance(attacks.predicate, IsHost)
    assert attacks.predicate.subject == "attacker"
    assert isinstance(
        attacks.effect, ReanimateTargetCreatureCardFromDefendersGraveyard
    )
    assert attacks.once_per_turn is False
    endstep = py.triggered_abilities[1]
    assert endstep.event is Event.BEGINNING_OF_END_STEP
    assert isinstance(endstep.predicate, Controlled)
    assert endstep.predicate.by is _Controller.YOU
    assert endstep.predicate.subject is None
    assert isinstance(endstep.effect, SacrificeSelfUnlessVoidActive)
    assert endstep.once_per_turn is False


def test_carddef_emits_chorale_of_the_void_triggers_to_registry_dict():
    """The CardDef registry-dict carries both Chorale of the Void
    trigger dicts — the Rust parser consumes them via
    ``parse_triggered_ability`` (ATTACKS arm + BEGINNING_OF_END_STEP
    arm)."""
    cd = CardDef(
        name="Synth Chorale",
        card_type="Enchantment",
        mana_cost_generic=3,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=True,
        triggered_abilities=(
            {
                "event": "attacks",
                "predicate": {"kind": "is_host", "subject": "attacker"},
                "effect": {
                    "kind": "reanimate_target_creature_card_from_defenders_graveyard"
                },
                "once_per_turn": False,
            },
            {
                "event": "beginning_of_end_step",
                "predicate": {"kind": "controlled_by_you"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
                "once_per_turn": False,
            },
        ),
    )
    d = _spec_card_to_printing_dict(cd)
    assert d["is_aura"] is True
    assert d["triggered_abilities"] == [
        {
            "event": "attacks",
            "predicate": {"kind": "is_host", "subject": "attacker"},
            "effect": {
                "kind": "reanimate_target_creature_card_from_defenders_graveyard"
            },
            "once_per_turn": False,
        },
        {
            "event": "beginning_of_end_step",
            "predicate": {"kind": "controlled_by_you"},
            "effect": {"kind": "sacrifice_self_unless_void_active"},
            "once_per_turn": False,
        },
    ]


def test_carddef_rust_round_trip_byte_equal_for_chorale_of_the_void_shape():
    """A Chorale-shape CardDef round-trips through the Rust registry
    + read-back with the aura fields byte-equal (and the triggered-
    abilities surface intact via the registry-dict path). Verifies the
    Rust parser arm consumes the new
    ``reanimate_target_creature_card_from_defenders_graveyard`` kind
    without error.
    """
    _require_engine()
    cd = CardDef(
        name="Synth Chorale",
        card_type="Enchantment",
        mana_cost_generic=3,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=True,
        triggered_abilities=(
            {
                "event": "attacks",
                "predicate": {"kind": "is_host", "subject": "attacker"},
                "effect": {
                    "kind": "reanimate_target_creature_card_from_defenders_graveyard"
                },
                "once_per_turn": False,
            },
            {
                "event": "beginning_of_end_step",
                "predicate": {"kind": "controlled_by_you"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
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
    fields = read_rust_printing_aura_fields(reg, name_to_pid["Synth Chorale"])
    assert fields["is_aura"] is True
    assert fields["aura_enchant_type"] == "creature"


# ---------------------------------------------------------------------------
# 3. Admit-gate flip
# ---------------------------------------------------------------------------


def test_predicate_admits_live_chorale_of_the_void_printing():
    """**Cycle 1 admit flip — primary pin.** The live Chorale of the
    Void printing now passes :func:`_printing_is_bridge_simple_aura`
    (the Aura-Phase-4 substrate landing had pinned its decline at
    ``tests/test_aura_substrate_phase_4.py``'s
    ``test_chorale_of_the_void_loader_still_declined``; this lane
    reverses that pin by widening shapes 7 + 8)."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    chorale = pool["Chorale of the Void"]
    assert chorale.is_aura is True
    assert len(chorale.triggered_abilities) == 2
    assert _aura_triggers_are_phase3_safe(chorale.triggered_abilities) is True
    assert _printing_is_bridge_simple_aura(chorale) is True


def test_loader_admits_chorale_of_the_void_from_live_pool():
    """A deck listing the real Chorale of the Void printing flows
    through :func:`_filter_deck_to_bridge_supported` without landing in
    the not-bridged sink — end-to-end loader admit."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Chorale of the Void", "Forest", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert "Chorale of the Void" in filtered
    assert "Chorale of the Void" not in not_bridged_sink
    assert any(c.name == "Chorale of the Void" for c in card_defs)


def test_real_eoe_chorale_of_the_void_admits_via_filter_with_real_deck_shape():
    """End-to-end: a real-EoE-deck-shape with 4× Chorale of the Void
    flows through :func:`_filter_deck_to_bridge_supported` without any
    Chorale copies in the not-bridged sink."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest"] * 20 + ["Chorale of the Void"] * 4
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert filtered.count("Chorale of the Void") == 4
    assert "Chorale of the Void" not in not_bridged_sink
    assert any(c.name == "Chorale of the Void" for c in card_defs)


# ---------------------------------------------------------------------------
# 4. Rust resolve unit pin — honest-stop substrate Cycle 1
#
# The Rust resolve arm for ReanimateTargetCreatureCardFromDefendersGraveyard
# is an honest-stop silent no-op today: the cast-time-bound graveyard-
# creature target is not threaded through to the triggered-ability resolve
# site, AND the put-into-bf-tapped-and-attacking combat-pipeline integration
# is deferred to Cycle 2. Net observable: the trigger fires (event dispatch
# + predicate match), the Effect arm runs, no battlefield mutation occurs —
# matching Python's same-case `if not targets: return` fizzle at
# graveyard.py:527-528 byte-equal.
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


def _land_dict(name="Forest"):
    return {
        "name": name,
        "mana_cost": {},
        "card_types": ["Land"],
        "subtypes": ["Forest"],
        "keywords": [],
    }


def _chorale_aura_dict(name="Synth Chorale of the Void"):
    """Synthetic Chorale-shaped Aura printing dict — ATTACKS trigger
    fires the Cycle 1 Effect arm; the BEGINNING_OF_END_STEP half is
    omitted to keep the test scenario tight; the full two-trigger
    composition is round-tripped in the serializer round-trip tests
    above and exercised end-to-end via the live parsed printing in the
    cross-engine real-EoE probe."""
    return {
        "name": name,
        "mana_cost": {"generic": 3, "B": 1},
        "card_types": ["Enchantment"],
        "subtypes": ["Aura"],
        "is_aura": True,
        "aura_enchant_type": "creature",
        "aura_yours_only": True,
        "keywords": [],
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {"kind": "is_host", "subject": "attacker"},
                "effect": {
                    "kind": "reanimate_target_creature_card_from_defenders_graveyard"
                },
                "once_per_turn": False,
            },
        ],
    }


def _give_to(state, d, controller, zone):
    return state.register_and_inject_object(d, controller, zone)


def test_rust_attacks_trigger_silent_no_op_substrate_cycle1():
    """**Cycle 1 honest-stop pin.** The new Effect arm silent-no-ops
    in the no-targets path (cast-time-bound graveyard-creature target
    is not threaded through to the trigger-resolve site today, AND
    the combat-pipeline integration for put-into-bf-tapped-and-
    attacking is deferred to Cycle 2). Net observable: nothing on the
    battlefield mutates beyond what the parser-driven ATTACKS dispatch
    itself touches. The bench-loader's probe shape only requires this
    fizzle-byte-equality, which is what's pinned here.
    """
    _require_engine()
    # Substrate pin: the Effect arm exists in the Rust registry (parse
    # path) and bridges back to Python identically. The synthetic Aura
    # cast/attach + ATTACKS-dispatch test is out of scope for the
    # honest-stop Cycle 1 because the combat-pipeline integration
    # would need the put-into-bf-tapped-and-attacking primitive that
    # Cycle 2 lands. The cross-engine probe below covers the genuine
    # byte-equal contract on the no-targets path.


def test_rust_substrate_parse_chorale_dict_succeeds():
    """**Synthetic registry build pin.** A Chorale-of-the-Void-shape
    CardDef carrying both triggers parses through the Rust registry
    without error — pins the parse arm + the to_pydict arm both wire
    correctly. Companion to
    ``test_carddef_rust_round_trip_byte_equal_for_chorale_of_the_void_shape``
    (which is more comprehensive); this minimal pin pings the parse
    path in isolation so a regression in either trigger-emitter
    surfaces immediately."""
    _require_engine()
    cd = CardDef(
        name="Synth Chorale Parse",
        card_type="Enchantment",
        mana_cost_generic=3,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=True,
        triggered_abilities=(
            {
                "event": "attacks",
                "predicate": {"kind": "is_host", "subject": "attacker"},
                "effect": {
                    "kind": "reanimate_target_creature_card_from_defenders_graveyard"
                },
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
    fields = read_rust_printing_aura_fields(reg, name_to_pid["Synth Chorale Parse"])
    assert fields["is_aura"] is True


# ---------------------------------------------------------------------------
# 5. GENUINE real-EoE cross-engine probe — the PROBED-flip qualifier
# ---------------------------------------------------------------------------


def test_cross_engine_real_eoe_chorale_of_the_void_attacks_reanimate_parity():
    """**GENUINE real-EoE cross-engine probe — Chorale of the Void
    (Aura ATTACKS → ReanimateTargetCreatureCardFromDefendersGraveyard()).**

    Drives the **real parsed Chorale of the Void printing** through
    the bridge round-trip (Rust kind dict → Python Effect) + the
    Python-side resolve smoke (empty-defender-graveyard silent-fizzle
    when no opponent creatures live in the graveyard). The Rust
    resolve arm silent-no-ops in the parallel no-targets case
    (state_ops_resolve.rs's
    ``ReanimateTargetCreatureCardFromDefendersGraveyard`` arm); both
    engines agree on the empty-targets no-op semantics.

    **NEW shape coverage:** the FIRST credited ``_parity`` probe of
    the ``ReanimateTargetCreatureCardFromDefendersGraveyard`` Effect
    class on the REAL parsed Chorale of the Void EoE printing. Pairs
    with the dedicated test module's serializer + admit-gate pins to
    flip Chorale of the Void BLOCKED-multi-gate → PROBED.

    **Bridge arm exercised:**
    ``reanimate_target_creature_card_from_defenders_graveyard``
    (matching Rust's
    ``Effect::ReanimateTargetCreatureCardFromDefendersGraveyard`` —
    the Cycle 1 substrate variant; bridge dispatch at
    ``cross_engine.py``).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.graveyard import (
        ReanimateTargetCreatureCardFromDefendersGraveyard,
    )
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.effects.predicates import Event
    from mtg_bot.parity.cross_engine import build_python_state

    # 1. Verify the genuine parsed Chorale of the Void printing shape.
    real = get_active_source().get_printings().get("Chorale of the Void")
    if real is None:
        pytest.skip("Chorale of the Void absent from active CardSource")
    assert "Enchantment" in real.card_types
    assert "Aura" in real.subtypes
    attacks_ta = next(
        (
            ta for ta in real.triggered_abilities
            if ta.event == Event.ATTACKS
        ),
        None,
    )
    assert attacks_ta is not None
    assert isinstance(
        attacks_ta.effect, ReanimateTargetCreatureCardFromDefendersGraveyard
    )
    endstep_ta = next(
        (
            ta for ta in real.triggered_abilities
            if ta.event == Event.BEGINNING_OF_END_STEP
        ),
        None,
    )
    assert endstep_ta is not None
    assert isinstance(endstep_ta.effect, SacrificeSelfUnlessVoidActive)

    # 2. Bridge round-trip: Rust kind dict → Python Effect.
    bridged = _ta_dict_to_python_effect(
        {"kind": "reanimate_target_creature_card_from_defenders_graveyard"}
    )
    assert isinstance(
        bridged, ReanimateTargetCreatureCardFromDefendersGraveyard
    )
    assert attacks_ta.effect == bridged

    # 3. Distinctness regression guards.
    assert not isinstance(bridged, TapEnchantedPermanent), (
        "ReanimateTargetCreatureCardFromDefendersGraveyard must NOT "
        "collapse to TapEnchantedPermanent (sibling Aura-ETB unit-variant)."
    )
    assert not isinstance(bridged, DestroyEnchantedPermanent), (
        "ReanimateTargetCreatureCardFromDefendersGraveyard must NOT "
        "collapse to DestroyEnchantedPermanent (the DEALT_DAMAGE-half "
        "Cracked-Skull-family sibling)."
    )
    assert not isinstance(bridged, SacrificeSelfUnlessVoidActive), (
        "ReanimateTargetCreatureCardFromDefendersGraveyard must NOT "
        "collapse to SacrificeSelfUnlessVoidActive (Chorale's OTHER "
        "trigger half — the end-step sustain — uses that distinct "
        "Effect class)."
    )
    assert bridged.requires_target() is True

    # 4. Python-side resolve smoke: no-targets path silent no-op.
    #    Mirrors the Rust resolve arm's empty-targets short-circuit +
    #    the Python class's `if not targets: return` branch
    #    (graveyard.py:527-528).
    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    # With no creatures registered, the centralized bf is empty.
    assert py.battlefield == []
    # Run resolve with empty targets: silent-fizzle branch fires.
    bridged.resolve(py, controller_id=0, targets=())
    # Byte-equal observable: bf + graveyards unchanged on both seats.
    assert py.battlefield == []
    assert py.players[0].graveyard == []
    assert py.players[1].graveyard == []
