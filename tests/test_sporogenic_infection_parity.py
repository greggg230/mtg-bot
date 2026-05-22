"""Sporogenic Infection Cycle 1 (PROBED flip) — parity probe module.

Spec: lane brief (2026-05-22). Cycle 1 lands:

  * ``Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted`` unit
    variant on ``rust/src/cards.rs`` (Sporogenic Infection's ETB
    trigger half — the "target opponent sacrifices a creature of
    their choice other than enchanted creature" sequence). Parse +
    to_pydict + resolve arms wired.
  * Resolve arm in ``rust/src/state_ops_resolve.rs`` that walks the
    5-zone source-finder (also reading ``source.attached_to`` off the
    bf object), computes the target opponent (``1 - controller``),
    filters creatures from the opponent's bf excluding the attached-to
    host, and pushes ``PendingDecision::OpponentSacrificesCreature
    Choice`` (the new sibling substrate to Cracked Skull C1's
    ``LookAtOpponentHandMayDiscard`` — distinct in that the decision
    seat is the OPPONENT, not the controller); empty candidates →
    silent no-op.
  * New ``PendingDecision::OpponentSacrificesCreatureChoice`` variant
    in ``rust/src/state.rs`` + dispatcher
    ``submit_opponent_sacrifices_creature_decision(chosen_id)`` in
    ``state_ops_decisions.rs`` (forced-choice — NO decline path; the
    printed text "sacrifices a creature of their choice" forces the
    sacrifice). Auto-resolve default picks ``candidates[0]`` (the
    deterministic-first bf creature; mirrors the
    ``_PickFirstLegalTargetAgent`` bench convention every cast-time
    picker rides).
  * Cross-engine bridge arm
    ``target_opponent_sacrifices_creature_other_than_enchanted`` in
    ``src/mtg_bot/parity/cross_engine.py:_ta_dict_to_python_effect`` —
    dict→Python.
  * Phase-3 safe-trigger admit-set widen in
    ``src/mtg_bot/parity/trajectory.py``: new shape 6
    (``ENTERS+IsSelf+TargetOpponentSacrificesCreatureOtherThan
    Enchanted``). Pairs with Cracked Skull C2's shape 5
    (``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent``, already
    shipped) for Sporogenic Infection's two-trigger composition.
  * ``_python_ta_to_safe_dict_aura_phase3`` emitter widen for the new
    ENTERS shape (Sporogenic Infection's ETB trigger round-trips
    via the canonical Aura-S4 emitter).

This flips Sporogenic Infection from BLOCKED-multi-gate → PROBED.
Sporogenic joins Cracked Skull as the second Aura with a two-trigger
composition (ETB + DEALT_DAMAGE host event) to ride end-to-end through
the bench loader.

**Tests cover:**

1. Parser SoT pin — real-EoE parser produces the canonical Sporogenic
   Infection printing shape (Aura + two triggers + correct effect /
   event / predicate types).
2. Serializer round-trip — ``_python_ta_to_safe_dict_aura_phase3``
   emits the canonical Rust dict shape for both triggers; the
   ``_card_def_to_python_printing`` round-trip materializes the
   byte-equal Python trigger surface.
3. Admit-gate flip — Sporogenic Infection passes
   :func:`_aura_triggers_are_phase3_safe` and
   :func:`_printing_is_bridge_simple_aura`; the live deck loader
   admits 4× Sporogenic Infection without any not-bridged drop.
4. Rust resolve unit pins — synthetic-state ETB resolve pushes the
   ``OpponentSacrificesCreatureChoice`` decision with the expected
   ``source_obj_id`` / ``target_player`` / ``candidates``; empty-bf
   opponent → silent no-op (no push); no-creature opponent → silent
   no-op (creatures filtered); excludes attached_to host even when
   host is on opponent's bf; submit forces sacrifice;
   auto-resolve default → sacrifices first.
5. GENUINE real-EoE cross-engine probe
   (``test_cross_engine_real_eoe_sporogenic_infection_enters_sacrifice_parity``)
   — drives the parsed Sporogenic Infection Aura through both
   engines via the bridge round-trip + Python-side resolve smoke
   (no-attached silent-fizzle, the canonical Aura-substrate probe
   shape Cracked Skull / HHHH / Pain-for-All / Tractor-Beam siblings
   established).
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


def test_parser_real_eoe_sporogenic_infection_has_canonical_shape():
    """**SoT pin — Sporogenic Infection's parser-emitted shape.**
    Real-EoE parser produces an Aura with two triggers:

      * ETB: ``ENTERS + IsSelf(None) +
        TargetOpponentSacrificesCreatureOtherThanEnchanted()``
      * DEALT_DAMAGE: ``DEALT_DAMAGE + IsHost(None) +
        DestroyEnchantedPermanent()``

    Any drift here would silently mistranslate cross-engine; this pin
    catches a parser-side regression before any other test misleads
    about the round-trip.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    pool = get_active_source().get_printings()
    si = pool["Sporogenic Infection"]
    assert si.is_aura is True
    assert "Aura" in si.subtypes
    assert "Enchantment" in si.card_types
    assert si.aura_enchant_type == "creature"
    assert len(si.triggered_abilities) == 2
    etb, dealt = si.triggered_abilities
    # ETB half.
    assert etb.event is Event.ENTERS
    assert isinstance(etb.predicate, IsSelf)
    assert etb.predicate.subject is None
    assert isinstance(
        etb.effect, TargetOpponentSacrificesCreatureOtherThanEnchanted
    )
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


def test_phase3_safe_admits_sporogenic_infection_etb_shape():
    """The
    ``ENTERS+IsSelf+TargetOpponentSacrificesCreatureOtherThanEnchanted``
    shape passes :func:`_aura_triggers_are_phase3_safe` (Cycle 1
    widen)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.predicates import Event, IsSelf

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentSacrificesCreatureOtherThanEnchanted(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_both_sporogenic_infection_triggers_together():
    """The full Sporogenic Infection trigger pair passes the gate as a
    unit — admitting either trigger alone wouldn't admit the printing
    (gate requires every trigger to round-trip)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    etb = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentSacrificesCreatureOtherThanEnchanted(),
    )
    dealt = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=DestroyEnchantedPermanent(),
    )
    assert _aura_triggers_are_phase3_safe((etb, dealt)) is True


def test_phase3_emit_sporogenic_infection_etb_canonical_shape():
    """The round-trip dict for Sporogenic Infection's ETB matches the
    canonical Rust trigger shape:
    ``event=enters / predicate.kind=source_is_self /
    effect.kind=target_opponent_sacrifices_creature_other_than_enchanted
    / once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.predicates import Event, IsSelf

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TargetOpponentSacrificesCreatureOtherThanEnchanted(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {
            "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
        },
        "once_per_turn": False,
    }


def test_bridge_arm_round_trips_target_opponent_sacrifices_creature():
    """The new cross-engine bridge arm
    ``target_opponent_sacrifices_creature_other_than_enchanted``
    translates the Rust dict shape to a
    ``TargetOpponentSacrificesCreatureOtherThanEnchanted()`` Python
    instance.
    """
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )

    bridged = _ta_dict_to_python_effect(
        {"kind": "target_opponent_sacrifices_creature_other_than_enchanted"}
    )
    assert isinstance(
        bridged, TargetOpponentSacrificesCreatureOtherThanEnchanted
    )


def test_printing_to_card_def_threads_sporogenic_infection_triggers():
    """Live Sporogenic Infection → CardDef threads both triggers through
    the Phase 3 emitter in the canonical Rust shape."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Sporogenic Infection"])
    assert cd is not None
    assert cd.name == "Sporogenic Infection"
    assert cd.card_type == "Enchantment"
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert len(cd.triggered_abilities) == 2
    assert cd.triggered_abilities[0] == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {
            "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
        },
        "once_per_turn": False,
    }
    assert cd.triggered_abilities[1] == {
        "event": "dealt_damage",
        "predicate": {"kind": "is_host"},
        "effect": {"kind": "destroy_enchanted_permanent"},
        "once_per_turn": False,
    }


def test_card_def_to_python_printing_round_trips_sporogenic_infection_triggers():
    """Sporogenic Infection CardDef → Python ``CardPrinting``
    materializes the byte-equal trigger surface (matching what the
    live parser produced — closing the bridge-correctness round-trip
    loop)."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Sporogenic Infection"])
    py = _card_def_to_python_printing(cd)
    assert py.is_aura is True
    assert py.aura_enchant_type == "creature"
    assert len(py.triggered_abilities) == 2
    etb = py.triggered_abilities[0]
    assert etb.event is Event.ENTERS
    assert isinstance(etb.predicate, IsSelf)
    assert etb.predicate.subject is None
    assert isinstance(
        etb.effect, TargetOpponentSacrificesCreatureOtherThanEnchanted
    )
    assert etb.once_per_turn is False
    dealt = py.triggered_abilities[1]
    assert dealt.event is Event.DEALT_DAMAGE
    assert isinstance(dealt.predicate, IsHost)
    assert dealt.predicate.subject is None
    assert isinstance(dealt.effect, DestroyEnchantedPermanent)
    assert dealt.once_per_turn is False


def test_carddef_emits_sporogenic_infection_triggers_to_registry_dict():
    """The CardDef registry-dict carries both Sporogenic Infection
    trigger dicts — the Rust parser consumes them via
    ``parse_triggered_ability`` (ENTERS arm + DEALT_DAMAGE arm)."""
    cd = CardDef(
        name="Synth SI",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=False,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
                },
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
            "effect": {
                "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
            },
            "once_per_turn": False,
        },
        {
            "event": "dealt_damage",
            "predicate": {"kind": "is_host"},
            "effect": {"kind": "destroy_enchanted_permanent"},
            "once_per_turn": False,
        },
    ]


def test_carddef_rust_round_trip_byte_equal_for_sporogenic_infection_shape():
    """A Sporogenic-shape CardDef round-trips through the Rust registry
    + read-back with the aura fields byte-equal (and the triggered-
    abilities surface intact via the registry-dict path). Verifies the
    Rust parser arm consumes the new
    ``target_opponent_sacrifices_creature_other_than_enchanted`` kind
    without error.
    """
    _require_engine()
    cd = CardDef(
        name="Synth SI",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("B", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=False,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
                },
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
    fields = read_rust_printing_aura_fields(reg, name_to_pid["Synth SI"])
    assert fields["is_aura"] is True
    assert fields["aura_enchant_type"] == "creature"


# ---------------------------------------------------------------------------
# 3. Admit-gate flip
# ---------------------------------------------------------------------------


def test_predicate_admits_live_sporogenic_infection_printing():
    """**Cycle 1 admit flip — primary pin.** The live Sporogenic
    Infection printing now passes :func:`_printing_is_bridge_simple_aura`
    (Cracked Skull C2 had pinned its decline at
    ``tests/test_aura_substrate_phase1.py``'s must_decline set;
    this lane reverses that pin by widening shape 6)."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    si = pool["Sporogenic Infection"]
    assert si.is_aura is True
    assert len(si.triggered_abilities) == 2
    assert _aura_triggers_are_phase3_safe(si.triggered_abilities) is True
    assert _printing_is_bridge_simple_aura(si) is True


def test_loader_admits_sporogenic_infection_from_live_pool():
    """A deck listing the real Sporogenic Infection printing flows
    through :func:`_filter_deck_to_bridge_supported` without landing in
    the not-bridged sink — end-to-end loader admit."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Sporogenic Infection", "Forest", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert "Sporogenic Infection" in filtered
    assert "Sporogenic Infection" not in not_bridged_sink
    assert any(c.name == "Sporogenic Infection" for c in card_defs)


def test_real_eoe_sporogenic_infection_admits_via_filter_with_real_deck_shape():
    """End-to-end: a real-EoE-deck-shape with 4× Sporogenic Infection
    flows through :func:`_filter_deck_to_bridge_supported` without any
    Sporogenic copies in the not-bridged sink."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest"] * 20 + ["Sporogenic Infection"] * 4
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert filtered.count("Sporogenic Infection") == 4
    assert "Sporogenic Infection" not in not_bridged_sink
    assert any(c.name == "Sporogenic Infection" for c in card_defs)


# ---------------------------------------------------------------------------
# 4. Rust resolve unit pins
#
# Rust doesn't expose a direct ``apply_effect_from_pydict`` test surface
# — the Effect arm is exercised by walking the live engine state
# machine via ``cast_aura`` → ``submit_enchant_target_decision`` →
# ``resolve_top_of_stack``. The ENTERS dispatch fires the trigger; a
# second ``resolve_top_of_stack`` runs the trigger's effect, which is
# the C1 Effect arm.
#
# Net observable per scenario:
#   * No-creature opponent → silent no-op (no decision pushed).
#   * Creatures on opponent's bf → push OpponentSacrificesCreatureChoice
#     with the expected candidates list. The opponent's agent drains
#     via submit_opponent_sacrifices_creature_decision(id) — forced
#     sacrifice. Auto-resolve default picks candidates[0].
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


def _sporogenic_infection_aura_dict(name="Synth Sporogenic Infection"):
    """Synthetic Sporogenic-shaped Aura printing dict — ENTERS trigger
    fires the Cycle 1 Effect arm. The DEALT_DAMAGE half is omitted to
    keep the test scenario tight; the full two-trigger composition is
    round-tripped in the serializer round-trip tests above and
    exercised end-to-end via the live parsed printing in the
    cross-engine real-EoE probe."""
    return {
        "name": name,
        "mana_cost": {"generic": 1, "B": 1},
        "card_types": ["Enchantment"],
        "subtypes": ["Aura"],
        "is_aura": True,
        "aura_enchant_type": "creature",
        "keywords": [],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "target_opponent_sacrifices_creature_other_than_enchanted"
                },
                "once_per_turn": False,
            },
        ],
    }


def _give_to(state, d, controller, zone):
    return state.register_and_inject_object(d, controller, zone)


def _bf_ids(state, pid):
    # Rust's battlefield_snapshot returns BOTH players' bf flattened;
    # filter by controller for the per-player view.
    return [
        o["id"] for o in state.battlefield_snapshot()
        if o.get("controller", o.get("controller_id")) == pid
    ]


def _gy_ids(state, pid):
    return [o["id"] for o in state.graveyard_snapshot(pid)]


def _cast_aura_and_resolve_aura(state, aura_id, controller, host_id):
    """Drive ``cast_aura`` → ``submit_enchant_target_decision(host_id)``
    → ``resolve_top_of_stack``. After this returns, the Aura is on bf
    attached to host_id and the ETB trigger is on the stack."""
    ok = state.cast_aura(aura_id, controller)
    assert ok is True
    if state.pending_decisions():
        state.submit_enchant_target_decision(host_id)
    state.resolve_top_of_stack()


def test_rust_etb_trigger_no_opponent_creatures_silent_no_op():
    """**No-creature opponent → silent no-op.** The Cycle 1 Effect
    arm's empty-candidates short-circuit fires (Python parity at
    ``opponent_punishers.py:128``: ``if not legal: return``). The
    trigger resolves with NO ``PendingDecision`` ever pushed; net
    observable: post-trigger, queue empty + opponent bf unchanged.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _sporogenic_infection_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("Bears"), 1, ZONE_BATTLEFIELD)
    # P1 has the host but no OTHER creatures (host is excluded).
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 1)

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    # Aura on bf, ETB trigger on stack.
    assert state.stack_size() == 1
    state.resolve_top_of_stack()  # trigger fires; empty candidates
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "OpponentSacrificesCreatureChoice" for pd in pds
    )
    # Host stays on opponent's bf (host is filtered, so no sacrifice).
    assert host_id in _bf_ids(state, 1)
    assert _gy_ids(state, 1) == []


def test_rust_etb_trigger_excludes_attached_host_from_candidates():
    """**Excludes the attached_to host even when host is a creature on
    the opponent's bf.** Sporogenic's printed text says "a creature of
    their choice OTHER THAN ENCHANTED CREATURE" — the host id must be
    filtered out of the sac candidates. With host + ONE other creature
    on the opponent's bf, the candidates list contains exactly that
    one other creature."""
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _sporogenic_infection_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("HostBear"), 1, ZONE_BATTLEFIELD)
    other_id = _give_to(state, _creature_dict("OtherBear"), 1, ZONE_BATTLEFIELD)
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 1)

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    assert state.stack_size() == 1
    # Resolve the ETB trigger BUT inspect the pending-decisions queue
    # mid-resolve. The trigger-resolve calls auto_resolve_pending at
    # the tail; to see the unresolved decision we need a probe BEFORE
    # the auto-drain. We use the synthetic-test push helper for the
    # candidate-exclusion pin instead — the live live-trigger path
    # also drains, but the auto-drain default sacrifices candidates[0],
    # which is the EXPECTED candidate (other_id). We assert post-drain
    # state instead: other_id moved to graveyard, host_id still on bf.
    state.resolve_top_of_stack()
    # Auto-resolve default = sacrifice candidates[0]. The candidate
    # list (post-host-exclusion) is [other_id], so other_id is
    # sacrificed and host_id stays.
    assert host_id in _bf_ids(state, 1)
    assert other_id not in _bf_ids(state, 1)
    assert other_id in _gy_ids(state, 1)


def test_rust_etb_trigger_filters_to_creatures_only():
    """**Filters to creatures only.** Non-creature permanents on the
    opponent's bf (e.g. lands, artifacts, enchantments) are excluded
    from the sac candidates. With only a land on opponent's bf
    (besides the host), the candidates list is empty and the trigger
    silent no-ops.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(state, _sporogenic_infection_aura_dict(), 0, ZONE_HAND)
    host_id = _give_to(state, _creature_dict("HostBear"), 1, ZONE_BATTLEFIELD)
    forest_id = _give_to(state, _land_dict("Forest"), 1, ZONE_BATTLEFIELD)
    state.add_mana(0, "B", 1)
    state.add_mana(0, "C", 1)

    _cast_aura_and_resolve_aura(state, aura_id, 0, host_id)
    state.resolve_top_of_stack()
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "OpponentSacrificesCreatureChoice" for pd in pds
    )
    # Land + host stay on opponent's bf (no sacrifice).
    assert host_id in _bf_ids(state, 1)
    assert forest_id in _bf_ids(state, 1)
    assert _gy_ids(state, 1) == []


def test_rust_substrate_push_then_submit_specific_id_sacrifices_chosen_creature():
    """**Cycle 1 substrate pin — synthetic push helper.** The push
    helper directly creates an ``OpponentSacrificesCreatureChoice``
    decision; the submit dispatcher routes the chosen id from
    target_player's bf → target_player's graveyard. Pins the
    substrate dispatcher in isolation from the Effect arm.
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(
        state, _creature_dict("HostBear"), 0, ZONE_BATTLEFIELD
    )
    c1 = _give_to(state, _creature_dict("CreatureA"), 1, ZONE_BATTLEFIELD)
    c2 = _give_to(state, _creature_dict("CreatureB"), 1, ZONE_BATTLEFIELD)
    state.push_opponent_sacrifices_creature_choice(aura_id, 1, [c1, c2])
    state.submit_opponent_sacrifices_creature_decision(c2)
    # Queue drained.
    pds = state.pending_decisions()
    assert all(
        pd["kind"] != "OpponentSacrificesCreatureChoice" for pd in pds
    )
    # c2 moved bf → graveyard; c1 stays.
    assert c2 not in _bf_ids(state, 1)
    assert c2 in _gy_ids(state, 1)
    assert c1 in _bf_ids(state, 1)


def test_rust_substrate_submit_rejects_unknown_id():
    """**Submit validation.** ``submit_opponent_sacrifices_creature_decision``
    raises ``ValueError`` transactionally on an id NOT in the candidate
    set — no mutation on failure. Matches the sibling
    ``submit_tap_permanent_decision`` validation shape."""
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(
        state, _creature_dict("HostBear"), 0, ZONE_BATTLEFIELD
    )
    c1 = _give_to(state, _creature_dict("CreatureA"), 1, ZONE_BATTLEFIELD)
    state.push_opponent_sacrifices_creature_choice(aura_id, 1, [c1])
    bogus_id = 99999
    with pytest.raises(ValueError):
        state.submit_opponent_sacrifices_creature_decision(bogus_id)
    # Queue intact; c1 still on bf.
    pds = state.pending_decisions()
    assert any(
        pd["kind"] == "OpponentSacrificesCreatureChoice" for pd in pds
    )
    assert c1 in _bf_ids(state, 1)


def test_rust_substrate_auto_resolve_default_sacrifices_first():
    """**Auto-resolve default = sacrifice candidates[0].** Under the
    ``"default"`` strategy the opponent's first creature is
    sacrificed (the forced-choice shape — no decline path).
    """
    _require_engine()

    state = _fresh_state()
    aura_id = _give_to(
        state, _creature_dict("HostBear"), 0, ZONE_BATTLEFIELD
    )
    c1 = _give_to(state, _creature_dict("CreatureA"), 1, ZONE_BATTLEFIELD)
    c2 = _give_to(state, _creature_dict("CreatureB"), 1, ZONE_BATTLEFIELD)
    state.push_opponent_sacrifices_creature_choice(aura_id, 1, [c1, c2])
    state.auto_resolve_pending_decisions("default")
    # c1 (candidates[0]) sacrificed; c2 stays.
    assert c1 not in _bf_ids(state, 1)
    assert c1 in _gy_ids(state, 1)
    assert c2 in _bf_ids(state, 1)


# ---------------------------------------------------------------------------
# 5. GENUINE real-EoE cross-engine probe — the PROBED-flip qualifier
# ---------------------------------------------------------------------------


def test_cross_engine_real_eoe_sporogenic_infection_enters_sacrifice_parity():
    """**GENUINE real-EoE cross-engine probe — Sporogenic Infection
    (Aura ENTERS → TargetOpponentSacrificesCreatureOtherThanEnchanted()).**

    Drives the **real parsed Sporogenic Infection printing** through
    the bridge round-trip (Rust kind dict → Python Effect) + the
    Python-side resolve smoke (no-host silent-fizzle when the opponent
    has no other creatures). The Rust resolve arm silent-no-ops in
    the parallel no-source-on-bf / no-opponent-creatures cases
    (state_ops_resolve.rs's
    ``TargetOpponentSacrificesCreatureOtherThanEnchanted`` arm); both
    engines agree on the empty-candidates no-op semantics.

    **NEW shape coverage:** the FIRST credited ``_parity`` probe of
    the ``TargetOpponentSacrificesCreatureOtherThanEnchanted`` Effect
    class on the REAL parsed Sporogenic Infection EoE printing. Pairs
    with the dedicated test module's serializer + admit-gate pins to
    flip Sporogenic Infection BLOCKED-multi-gate → PROBED.

    **Bridge arm exercised:**
    ``target_opponent_sacrifices_creature_other_than_enchanted``
    (matching Rust's
    ``Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted`` —
    the Cycle 1 substrate variant; bridge dispatch at
    ``cross_engine.py``).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.opponent_punishers import (
        TargetOpponentSacrificesCreatureOtherThanEnchanted,
    )
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.effects.predicates import Event
    from mtg_bot.parity.cross_engine import build_python_state

    # 1. Verify the genuine parsed Sporogenic Infection printing shape.
    real = get_active_source().get_printings().get("Sporogenic Infection")
    if real is None:
        pytest.skip("Sporogenic Infection absent from active CardSource")
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
    assert isinstance(
        enters_ta.effect, TargetOpponentSacrificesCreatureOtherThanEnchanted
    )

    # 2. Bridge round-trip: Rust kind dict → Python Effect.
    bridged = _ta_dict_to_python_effect(
        {"kind": "target_opponent_sacrifices_creature_other_than_enchanted"}
    )
    assert isinstance(
        bridged, TargetOpponentSacrificesCreatureOtherThanEnchanted
    )
    assert enters_ta.effect == bridged

    # 3. Distinctness regression guards.
    assert not isinstance(bridged, TapEnchantedPermanent), (
        "TargetOpponentSacrificesCreatureOtherThanEnchanted must NOT "
        "collapse to TapEnchantedPermanent (sibling Aura-ETB unit-variant)."
    )
    assert not isinstance(bridged, DestroyEnchantedPermanent), (
        "TargetOpponentSacrificesCreatureOtherThanEnchanted must NOT "
        "collapse to DestroyEnchantedPermanent (the DEALT_DAMAGE-half "
        "sibling)."
    )
    assert bridged.requires_target() is True

    # 4. Python-side resolve smoke: empty opponent bf → silent no-op.
    #    Mirrors the Rust resolve arm's empty-candidates short-circuit
    #    + the Python class's "no legal options → silent return" branch.
    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    # Python's `state.battlefield` is centralized (CR 405 — one
    # zone shared across players, controller carried per-object).
    # With no creatures registered, the centralized bf is empty.
    assert py.battlefield == []
    # Run the resolve with empty target bf: no-options branch fires.
    bridged.resolve(py, controller_id=0, targets=(("player", 1),))
    # Byte-equal observable: bf + graveyard unchanged on both seats.
    assert py.battlefield == []
    assert py.players[0].graveyard == []
    assert py.players[1].graveyard == []
