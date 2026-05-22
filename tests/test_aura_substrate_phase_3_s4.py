"""Aura-attachment substrate — Phase 3 / S4 (triggered Aura admit;
Pain for All closure).

Spec: ``docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`` §3 Phase 3. Closes
the long-standing Pain for All honest-stop by widening the bench-
loader admit predicate :func:`_printing_is_bridge_simple_aura` to
include the Pain-for-All triggered-Aura shape:

* ``ENTERS+IsSelf+PainForAllETBEffect`` (ETB damage to any other
  target).
* ``DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect`` (damage echo).

All three pieces of the substrate were already on main:

* Rust ``Effect::PainForAllETBEffect`` + ``Effect::
  PainForAllDamageEchoEffect`` + ``Predicate::IsHost`` parity
  landed in the Pain for All closure slice (commit ``60fea01b``).
* Python ``apply_aura_static_grants_on_attach`` materializes
  ``card.attached_to`` (``actions.py:2746``).
* Rust ``move_aura_post_resolve`` stamps ``attached_to =
  Some(host_id)`` on Aura resolution (commit ``643163c6``, Phase 2 S2).

The bench-loader admit gate was the only remaining gap — this slice
flips it. Pain for All is the FIRST Phase-3 / S4 triggered-Aura
admitted end-to-end.

**Cards unblocked (live EoE pool).** Pain for All — 4,367 17lands
decks per ``docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md`` §4 (the
``BLOCKED-genuinely_missing`` tier shrink target from the Phase-2-S2-
extension slice).

**Cards that stay declined (S4 / non-Phase-3-safe shapes).**

* Chorale of the Void — ATTACKS + ``ReanimateTargetCreatureCardFromDefendersGraveyard`` + BEGINNING_OF_END_STEP + ``SacrificeSelfUnlessVoidActive`` (each effect outside the Phase-3 safe set).
* Cracked Skull — ENTERS + ``TargetOpponentLookHandMayDiscard`` + DEALT_DAMAGE + ``DestroyEnchantedPermanent`` (each effect outside the Phase-3 safe set).
* Sporogenic Infection — ENTERS + ``TargetOpponentSacrificesCreatureOtherThanEnchanted`` + DEALT_DAMAGE + ``DestroyEnchantedPermanent`` (each effect outside the Phase-3 safe set).
* Meltstrider's Resolve — ENTERS + ``EnchantedCreatureFightsTargetEffect`` (the residue trigger needs its own follow-up slice).
* Tractor Beam — declined separately on the lone-control +
  prevents-untap rider (Phase-2-S2-extension honest-stop, unchanged
  by Phase 3).
* Moonlit Meditation — S3 replacement-rider decline (Phase 4 will
  widen).

This is by design — Phase 3's safe-trigger gate is narrow to keep
the §2 teeth honest (a wider gate would silently mistranslate to an
effectless permanent on Rust for any uncovered Effect arm). Each
remaining card is its own follow-up Rust slice.
"""

from __future__ import annotations

import importlib

import pytest

from mtg_bot.parity.cross_engine import (
    CardDef,
    DeckSpec,
    _card_def_to_python_printing,
    _spec_card_to_printing_dict,
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


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


# ---------------------------------------------------------------------------
# Phase 3 safe-trigger gate
# ---------------------------------------------------------------------------


def test_phase3_safe_admits_pain_for_all_etb_shape():
    """The ``ENTERS+IsSelf+PainForAllETBEffect`` shape passes
    :func:`_aura_triggers_are_phase3_safe`."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllETBEffect

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=PainForAllETBEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_pain_for_all_damage_echo_shape():
    """The ``DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect`` shape
    passes :func:`_aura_triggers_are_phase3_safe`."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsHost
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=PainForAllDamageEchoEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_admits_both_pain_for_all_triggers_together():
    """The full Pain for All trigger pair passes the gate as a unit
    — admitting either trigger alone wouldn't admit the printing
    (gate requires every trigger to round-trip)."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf
    from mtg_bot.effects.bespoke import (
        PainForAllDamageEchoEffect,
        PainForAllETBEffect,
    )

    etb = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=PainForAllETBEffect(),
    )
    echo = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=PainForAllDamageEchoEffect(),
    )
    assert _aura_triggers_are_phase3_safe((etb, echo)) is True


def test_phase3_safe_admits_tractor_beam_etb_shape_too():
    """Phase 3 is a strict superset of the Phase-2-S2-extension gate
    — the Tractor-Beam ``ENTERS+IsSelf+TapEnchantedPermanent`` shape
    also passes (so the Phase-3 gate is the single canonical chokepoint).
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TapEnchantedPermanent(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is True


def test_phase3_safe_rejects_empty_tuple():
    """Empty tuple is NOT admitted — callers should gate on
    ``if printing.triggered_abilities:`` first."""
    assert _aura_triggers_are_phase3_safe(()) is False


def test_phase3_safe_rejects_once_per_turn():
    """Aura triggers fire per-occurrence, never once-per-turn — the
    gate rejects any once-per-turn trigger to keep the substrate
    honest."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllETBEffect

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=PainForAllETBEffect(),
        once_per_turn=True,
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


def test_phase3_safe_rejects_unsafe_effect():
    """An Aura trigger with an effect outside the Phase-3 safe set
    is rejected — the gate is the §2 positive allowlist.

    **Updated 2026-05-22 — Cracked Skull C2 closure landed.**
    ``DestroyEnchantedPermanent`` is no longer the canonical
    rejection example — the C2 widen admits the
    ``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent`` shape (Cracked
    Skull's second trigger). Use a genuinely-uncovered shape here:
    ``Event.ENTERS + IsSelf + DestroyTargetCreature`` — DestroyTarget
    Creature is a spell-effect primitive that has no Aura-ETB
    carrier today and is positively declined by the gate.
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.destroy import DestroyTargetCreature

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=DestroyTargetCreature(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


def test_phase3_safe_rejects_wrong_event_for_pain_for_all_etb():
    """A trigger with the Pain for All ETB effect but on the wrong
    event (DIES) is rejected — the shape is fully specified, no
    cross-shape pollution."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllETBEffect

    ta = TriggeredAbility(
        event=Event.DIES,
        predicate=IsSelf(),
        effect=PainForAllETBEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


def test_phase3_safe_rejects_wrong_predicate_for_damage_echo():
    """The damage-echo arm requires ``IsHost`` — a damage-echo with
    ``IsSelf`` predicate (a different relational shape) is rejected
    so a future ``IsSelf+DEALT_DAMAGE`` non-Aura shape doesn't
    silently piggyback."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsSelf(),
        effect=PainForAllDamageEchoEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


def test_phase3_safe_rejects_non_triggered_ability_input():
    """Defensive: a non-``TriggeredAbility`` element in the input
    tuple is rejected. Callers should pass parser-emitted
    ``triggered_abilities`` tuples, but the gate stays honest
    against accidental misuse."""
    assert _aura_triggers_are_phase3_safe(("not_a_ta",)) is False


def test_phase3_safe_rejects_wrong_predicate_subject_pain_for_all_etb():
    """The ETB arm requires ``IsSelf(subject=None)`` — a non-None
    subject slot (a different relational shape) is rejected so an
    accidental subject-bound ETB doesn't get silently misrouted."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllETBEffect

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(subject="attacker"),
        effect=PainForAllETBEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


def test_phase3_safe_rejects_subject_bound_is_host_for_damage_echo():
    """The damage-echo arm requires ``IsHost(subject=None)`` — the
    multi-subject form (e.g. ``subject="attacker"``) is rejected
    so a future equipment-shape DEALT_DAMAGE trigger doesn't
    accidentally hit the Pain for All emitter arm."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsHost
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(subject="attacker"),
        effect=PainForAllDamageEchoEffect(),
    )
    assert _aura_triggers_are_phase3_safe((ta,)) is False


# ---------------------------------------------------------------------------
# Phase 3 emitter
# ---------------------------------------------------------------------------


def test_phase3_emit_pain_for_all_etb_canonical_shape():
    """The round-trip dict matches the canonical Rust trigger shape
    for the Pain for All ETB:
    ``event=enters / predicate.kind=source_is_self /
    effect.kind=pain_for_all_etb / once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.bespoke import PainForAllETBEffect

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=PainForAllETBEffect(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "pain_for_all_etb"},
        "once_per_turn": False,
    }


def test_phase3_emit_pain_for_all_damage_echo_canonical_shape():
    """The round-trip dict for the Pain for All damage echo:
    ``event=dealt_damage / predicate.kind=is_host /
    effect.kind=pain_for_all_damage_echo / once_per_turn=False``."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsHost
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    ta = TriggeredAbility(
        event=Event.DEALT_DAMAGE,
        predicate=IsHost(),
        effect=PainForAllDamageEchoEffect(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "dealt_damage",
        "predicate": {"kind": "is_host"},
        "effect": {"kind": "pain_for_all_damage_echo"},
        "once_per_turn": False,
    }


def test_phase3_emit_tractor_beam_shape_back_compat():
    """The Phase 3 emitter preserves the Phase-2-S2-extension
    Tractor-Beam shape verbatim — the substrate continuity is
    bit-equal so the existing extension probes don't need to drift."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=TapEnchantedPermanent(),
    )
    d = _python_ta_to_safe_dict_aura_phase3(ta)
    assert d == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "tap_enchanted_permanent"},
        "once_per_turn": False,
    }


def test_phase3_emit_raises_on_unsupported_effect():
    """The emitter raises ``ValueError`` for an unsupported effect
    — a future widen of the safe-trigger gate without a matching
    emitter widen surfaces immediately rather than silently
    mistranslating."""
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards

    ta = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=DrawCards(1),
    )
    with pytest.raises(ValueError, match="_python_ta_to_safe_dict_aura_phase3"):
        _python_ta_to_safe_dict_aura_phase3(ta)


# ---------------------------------------------------------------------------
# Loader admit predicate — Pain for All
# ---------------------------------------------------------------------------


def test_predicate_admits_live_pain_for_all_printing():
    """**Phase 3 / S4 first deliverable.** The live Pain for All
    printing now passes :func:`_printing_is_bridge_simple_aura` —
    the first triggered-Aura admitted end-to-end through the bench
    loader."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    pfa = pool["Pain for All"]
    assert pfa.is_aura is True
    assert len(pfa.triggered_abilities) == 2
    assert _aura_triggers_are_phase3_safe(pfa.triggered_abilities) is True
    assert _printing_is_bridge_simple_aura(pfa) is True


def test_loader_admits_pain_for_all_from_live_pool():
    """A deck listing the real Pain for All printing flows through
    :func:`_filter_deck_to_bridge_supported` without landing in the
    not-bridged sink — end-to-end loader admit."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Pain for All", "Forest", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert "Pain for All" in filtered
    assert "Pain for All" not in not_bridged_sink
    assert any(c.name == "Pain for All" for c in card_defs)


# ---------------------------------------------------------------------------
# CardDef / Python round-trip
# ---------------------------------------------------------------------------


def test_printing_to_card_def_threads_pain_for_all_triggers():
    """Live Pain for All → CardDef threads both triggers through the
    Phase 3 emitter in the canonical Rust shape."""
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Pain for All"])
    assert cd is not None
    assert cd.name == "Pain for All"
    assert cd.card_type == "Enchantment"
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_yours_only is True
    assert len(cd.triggered_abilities) == 2
    assert cd.triggered_abilities[0] == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "pain_for_all_etb"},
        "once_per_turn": False,
    }
    assert cd.triggered_abilities[1] == {
        "event": "dealt_damage",
        "predicate": {"kind": "is_host"},
        "effect": {"kind": "pain_for_all_damage_echo"},
        "once_per_turn": False,
    }


def test_card_def_to_python_printing_round_trips_pain_for_all_triggers():
    """Pain for All CardDef → Python ``CardPrinting`` materializes
    the byte-equal trigger surface (matching what the live parser
    produced — closing the bridge-correctness round-trip loop)."""
    from mtg_bot.effects.bespoke import (
        PainForAllDamageEchoEffect,
        PainForAllETBEffect,
    )
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf
    from mtg_bot.cards.source import get_active_source

    pool = get_active_source().get_printings()
    cd = _printing_to_card_def(pool["Pain for All"])
    py = _card_def_to_python_printing(cd)
    assert py.is_aura is True
    assert py.aura_enchant_type == "creature"
    assert py.aura_yours_only is True
    assert len(py.triggered_abilities) == 2
    etb_ta = py.triggered_abilities[0]
    assert etb_ta.event is Event.ENTERS
    assert isinstance(etb_ta.predicate, IsSelf)
    assert etb_ta.predicate.subject is None
    assert isinstance(etb_ta.effect, PainForAllETBEffect)
    assert etb_ta.once_per_turn is False
    echo_ta = py.triggered_abilities[1]
    assert echo_ta.event is Event.DEALT_DAMAGE
    assert isinstance(echo_ta.predicate, IsHost)
    assert echo_ta.predicate.subject is None
    assert isinstance(echo_ta.effect, PainForAllDamageEchoEffect)
    assert echo_ta.once_per_turn is False


# ---------------------------------------------------------------------------
# Rust registry round-trip
# ---------------------------------------------------------------------------


def test_carddef_rust_round_trip_byte_equal_for_pain_for_all_shape():
    """A Pain for All CardDef round-trips through the Rust registry
    + read-back with the aura fields byte-equal (and the triggered-
    abilities surface intact via the registry-dict path)."""
    _require_engine()
    cd = CardDef(
        name="Synth PFA",
        card_type="Enchantment",
        mana_cost_generic=2,
        mana_cost_colored=(("R", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=True,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "pain_for_all_etb"},
                "once_per_turn": False,
            },
            {
                "event": "dealt_damage",
                "predicate": {"kind": "is_host"},
                "effect": {"kind": "pain_for_all_damage_echo"},
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
    fields = read_rust_printing_aura_fields(reg, name_to_pid["Synth PFA"])
    assert fields["is_aura"] is True
    assert fields["aura_enchant_type"] == "creature"
    assert fields["aura_yours_only"] is True


def test_carddef_emits_pain_for_all_triggers_to_registry_dict():
    """The CardDef registry-dict carries both Pain for All trigger
    dicts — the Rust parser consumes them via
    ``parse_triggered_ability`` (ENTERS arm + DEALT_DAMAGE arm)."""
    cd = CardDef(
        name="Synth PFA",
        card_type="Enchantment",
        mana_cost_generic=2,
        mana_cost_colored=(("R", 1),),
        is_aura=True,
        aura_yours_only=True,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "pain_for_all_etb"},
                "once_per_turn": False,
            },
            {
                "event": "dealt_damage",
                "predicate": {"kind": "is_host"},
                "effect": {"kind": "pain_for_all_damage_echo"},
                "once_per_turn": False,
            },
        ),
    )
    d = _spec_card_to_printing_dict(cd)
    assert d["is_aura"] is True
    assert d["aura_yours_only"] is True
    assert d["triggered_abilities"] == [
        {
            "event": "enters",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "pain_for_all_etb"},
            "once_per_turn": False,
        },
        {
            "event": "dealt_damage",
            "predicate": {"kind": "is_host"},
            "effect": {"kind": "pain_for_all_damage_echo"},
            "once_per_turn": False,
        },
    ]


# ---------------------------------------------------------------------------
# Negative pins — non-Phase-3-safe S4 Auras stay declined
# ---------------------------------------------------------------------------


def test_predicate_declines_chorale_of_the_void():
    """Chorale of the Void has ATTACKS + Reanimate +
    BEGINNING_OF_END_STEP + SacrificeSelfUnlessVoidActive triggers
    — none in the Phase-3 safe set. Stays declined post-Phase-3."""
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Chorale of the Void"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is False


def test_predicate_admits_cracked_skull_post_cycle2_closure():
    """**Updated 2026-05-22 — Cracked Skull C2 closure landed.**
    Cracked Skull is now bench-loader-ADMITTED: the two-trigger
    composition (ENTERS + TargetOpponentLookHandMayDiscard +
    DEALT_DAMAGE + DestroyEnchantedPermanent) is now Phase-3-safe
    via the C2 widen of :func:`_aura_triggers_are_phase3_safe`
    (the C2 Effect arm + cross-engine bridge arm + admit-set widen
    landed in ``feature/cracked-skull-cycle-2-p1``). Cycle 1's
    pin (Cracked Skull declined) is reversed here; the
    canonical PROBED-flip test lives at
    ``tests/test_cracked_skull_parity.py::test_predicate_admits_live_cracked_skull_printing``.
    """
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Cracked Skull"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is True


def test_predicate_admits_sporogenic_infection_post_cycle1_closure():
    """**Updated 2026-05-22 — Sporogenic Infection Cycle 1 closure
    landed.** Sporogenic Infection is now bench-loader-ADMITTED: the
    two-trigger composition
    (``ENTERS+IsSelf+TargetOpponentSacrificesCreatureOtherThan
    Enchanted`` + ``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent``)
    is now Phase-3-safe via this lane's widen of
    :func:`_aura_triggers_are_phase3_safe` (the Cycle 1 Effect arm +
    cross-engine bridge arm + admit-set widen landed in
    ``feature/sporogenic-infection-cycle-1-p1``). Pre-Cycle 1's pin
    (Sporogenic Infection declined) is reversed here; the canonical
    PROBED-flip test lives at
    ``tests/test_sporogenic_infection_parity.py::
    test_predicate_admits_live_sporogenic_infection_printing``.
    """
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Sporogenic Infection"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is True


def test_predicate_declines_meltstriders_resolve():
    """Meltstrider's Resolve carries an ENTERS trigger with
    ``EnchantedCreatureFightsTargetEffect`` — outside the Phase-3
    safe set (its own follow-up Rust slice)."""
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Meltstrider's Resolve"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is False


def test_predicate_admits_tractor_beam_post_cast_time_seat_alignment():
    """**Updated 2026-05-20 — cast-time auto-resolve seat alignment
    follow-up slice landed.** Tractor Beam is now bench-loader-
    ADMITTED: the cross-engine play-parity gap (cast-time enchant-
    target seat alignment + non-creature summoning_sick clear +
    aura-host-control revert on non-SBA destroy paths +
    BiosynthicBurst legal/resolve walks both bfs by controller) is
    closed. The Phase-3 safe-trigger gate
    (``_aura_triggers_are_phase3_safe``) admits the
    ``ENTERS+IsSelf+TapEnchantedPermanent`` ETB shape; the §2
    teeth admit the joint ``attached_grants_control +
    attached_prevents_untap`` rider combo."""
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Tractor Beam"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is True


def test_predicate_admits_moonlit_meditation_post_phase_4_s3_closure():
    """Moonlit Meditation's ``replaces_first_tokens_with_copies``
    rider (S3) admits at the loader post-Phase-4-S3 closure
    (2026-05-21). Pre-closure pin (kept for change-tracking): the
    predicate declined the rider regardless of Phase 3 widening
    because the Rust CreateToken arm hadn't yet consulted the
    controller's attached Moonlit Aura. Post-closure: ``moonlit_
    token_substitution`` (rust/src/state_ops_resolve.rs) wires the
    consumer end-to-end, the loader admits, and the rider round-
    trips byte-equal cross-engine."""
    from mtg_bot.cards.source import get_active_source

    p = get_active_source().get_printings()["Moonlit Meditation"]
    assert p.is_aura is True
    assert _printing_is_bridge_simple_aura(p) is True


# ---------------------------------------------------------------------------
# End-to-end: Pain for All triggers fire on host when attached
# ---------------------------------------------------------------------------


def _new_two_player_state():
    """Build a minimal two-player state with a deterministic target
    agent (PickFirst). Mirrors ``test_coverage_catchup._new_state``."""
    from mtg_bot.agents import Agent, RandomAgent
    from mtg_bot.game import GameState, Player
    from mtg_bot.turn import Step

    class _PickFirstAgent(Agent):
        def choose_target(self, state, player_id, legal_targets):
            return legal_targets[0] if legal_targets else None

    state = GameState()
    state.players.append(Player(id=0, agent=_PickFirstAgent()))
    state.players.append(Player(id=1, agent=RandomAgent()))
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = Step.PRECOMBAT_MAIN
    return state


def test_pain_for_all_etb_fizzle_byte_equal_with_no_attached_host():
    """Pain for All's ETB resolve fizzles silently when
    ``aura.attached_to is None``. Mirrors Python's bespoke.py:389-391
    short-circuit + Rust's same fizzle path — cross-engine byte-equal
    in the no-host case."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.bespoke import PainForAllETBEffect
    from mtg_bot.game import Zone

    state = _new_two_player_state()
    pool = get_active_source().get_printings()
    aura = state.new_object(pool["Pain for All"], 0, Zone.BATTLEFIELD)
    state.battlefield.append(aura)
    aura.attached_to = None
    state.resolving_source = aura
    pre = [p.life for p in state.players]
    try:
        PainForAllETBEffect().resolve(state, 0, (("player", 1),))
    finally:
        state.resolving_source = None
    post = [p.life for p in state.players]
    assert pre == post, (
        "Pain for All ETB on a host-less Aura must fizzle — "
        "byte-equal with Rust's no-attached_to fizzle path."
    )


def test_pain_for_all_etb_damages_first_legal_target_when_attached_to_host():
    """When attached to a host with positive power, Pain for All's
    ETB delivers ``effective_power(host)`` damage to the first legal
    target. Mirrors the legal_targets ordering Python uses
    (every-non-lost-player then every-non-host-creature, in id order).
    """
    from mtg_bot.cards import SYNTHETIC_CREATURES
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.bespoke import PainForAllETBEffect
    from mtg_bot.game import Zone

    state = _new_two_player_state()
    pool = get_active_source().get_printings()
    bear = state.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.BATTLEFIELD)
    state.battlefield.append(bear)
    aura = state.new_object(pool["Pain for All"], 0, Zone.BATTLEFIELD)
    state.battlefield.append(aura)
    aura.attached_to = bear.id
    state.resolving_source = aura
    try:
        # Pick player 1 explicitly — mirrors PickFirst on the (player,1)
        # target ordering legal_targets would surface.
        PainForAllETBEffect().resolve(state, 0, (("player", 1),))
    finally:
        state.resolving_source = None
    # Wild Bear is a 2/2; ETB drives 2 life loss to P1.
    assert state.players[1].life == 18
    assert state.players[0].life == 20


def test_pain_for_all_damage_echo_fires_to_each_opponent():
    """Pain for All's damage-echo resolves to LIFE LOSS on each
    non-controller opponent. The amount comes from
    ``state.resolving_combat_damage_amount`` — the in-flight
    DEALT_DAMAGE payload. Mirrors bespoke.py:443-457 exactly."""
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    state = _new_two_player_state()
    state.resolving_combat_damage_amount = 4
    p0_pre = state.players[0].life
    p1_pre = state.players[1].life
    PainForAllDamageEchoEffect().resolve(state, 0, ())
    assert state.players[0].life == p0_pre  # controller spared
    assert state.players[1].life == p1_pre - 4


def test_pain_for_all_zero_damage_echo_noop():
    """Damage echo with amount=0 is a no-op — guards the inherent
    damage-feedback loop (a non-positive amount must not even touch
    life), byte-equal with the bespoke.py:448-449 guard."""
    from mtg_bot.effects.bespoke import PainForAllDamageEchoEffect

    state = _new_two_player_state()
    state.resolving_combat_damage_amount = 0
    pre = [p.life for p in state.players]
    PainForAllDamageEchoEffect().resolve(state, 0, ())
    post = [p.life for p in state.players]
    assert pre == post


# ---------------------------------------------------------------------------
# Decline non-Aura via Phase 3 (defensive)
# ---------------------------------------------------------------------------


def test_predicate_still_declines_non_aura():
    """A non-Aura printing (Forest) is positively declined — the
    Phase 3 widening doesn't relax the is_aura gate."""
    from mtg_bot.cards.source import get_active_source

    forest = get_active_source().get_printings()["Forest"]
    assert _printing_is_bridge_simple_aura(forest) is False


# ---------------------------------------------------------------------------
# Real-EoE trajectory probe — Pain for All deck setup is byte-equal
# ---------------------------------------------------------------------------


def test_real_eoe_pain_for_all_deck_setup_byte_equal_cross_engine():
    """A real-EoE-deck-shape with Pain for All in both players' decks
    sets up identical Python + Rust states at game start. Verifies
    the loader admit + bridge round-trip + Rust registry parse all
    succeed end-to-end on the canonical Pain for All printing — the
    pre-condition for byte-equal trajectory replay through the bench.
    """
    _require_engine()
    from mtg_bot.parity import trajectory

    deck = ["Forest"] * 20 + ["Pain for All"] * 4
    py_state, rust_state = trajectory.setup_matched_game_from_real_decks(
        deck, deck, shared_rng_seed=0, mulligans=False,
    )
    # Pain for All landed in the bridge-filtered deck on BOTH engines —
    # the loader admit + bridge round-trip both succeeded. The two
    # libraries reach the same total count (after pad-to-size 60) on
    # both engines; if the loader silently dropped Pain for All the
    # Python deck would still have 4 copies but the Rust registry
    # build would have raised a missing-printing error.
    py_libs = [len(p.library) for p in py_state.players]
    assert py_libs[0] == py_libs[1]

    def _count_total(p):
        return sum(
            1 for o in (*p.library, *p.hand)
            if o.printing.name == "Pain for All"
        )

    # 4 Pain for All copies in library + hand across both players —
    # the loader admit landed cleanly without any silent drop.
    assert _count_total(py_state.players[0]) == 4
    assert _count_total(py_state.players[1]) == 4


def test_real_eoe_pain_for_all_admits_via_filter_with_real_deck_shape():
    """End-to-end: a real-EoE-deck-shape with 4× Pain for All flows
    through :func:`_filter_deck_to_bridge_supported` without any
    Pain for All copies in the not-bridged sink. Verifies the
    Phase 3 admit doesn't accidentally re-decline in the deck-side
    filter chokepoint."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    deck = ["Forest"] * 20 + ["Pain for All"] * 4
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck, printings, not_bridged_sink=not_bridged_sink,
    )
    assert filtered.count("Pain for All") == 4
    assert "Pain for All" not in not_bridged_sink
    assert any(c.name == "Pain for All" for c in card_defs)
