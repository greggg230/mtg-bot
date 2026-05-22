"""Aura-attachment substrate — Phase 4 (Chorale of the Void substrate
landing + Moonlit Meditation honest-stop).

Spec: ``docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`` §3 Phase 4. The
plan §1 names the two remaining S3/S4 carriers:

* **Chorale of the Void** (S4 multi-gated; 2,783 17lands decks) —
  triggered Aura with TWO triggers:
    * ``ATTACKS + IsHost(subject="attacker") +
      ReanimateTargetCreatureCardFromDefendersGraveyard``
    * ``BEGINNING_OF_END_STEP + ControlledByYou +
      SacrificeSelfUnlessVoidActive``

* **Moonlit Meditation** (S3; 633 17lands decks) — bare Aura with a
  ``replaces_first_tokens_with_copies=True`` token-creation
  REPLACEMENT effect.

**This slice's scope.**

Phase 4 is the LAST named Aura phase per the plan, and per the
§2/§3 honesty teeth, both cards' full closures need substrate
deeper than one cycle. This slice ships:

1. **``Effect::SacrificeSelfUnlessVoidActive`` Rust variant (FULL
   closure of one of Chorale's two triggers).** The new Rust enum
   variant + ``parse_effect`` + ``effect_to_pydict`` + cross_engine
   bridge arm + state_ops_resolve resolve arm — all byte-equal with
   Python's ``SacrificeSelfUnlessVoidActive``
   (``effects/combinators.py:172-200``). The resolve arm reads the
   two engine-wide void flags (same source of truth as Python's
   ``is_void_active(state)``) and either silent-no-ops (Void active
   → keep source) or sacrifices the source. This is NOT an
   honest-stop fizzle — full Rust semantics. The end-step trigger
   dispatcher (``Event::BeginningOfEndStep``) is already wired.

2. **Honest-stop for Chorale's ATTACKS trigger.** The sibling
   ``ReanimateTargetCreatureCardFromDefendersGraveyard`` Effect
   needs a targeted-graveyard decision substrate that doesn't
   exist in Rust today (no `PendingDecision::GraveyardCardChoice`,
   no targeting pump for "card-in-opponent-graveyard"). Ship its
   admit pathway = NOT in this slice. Tractor-Beam-like substrate
   plumbing in a future slice closes it.

3. **Honest-stop for Moonlit Meditation (S3 replacement
   substrate).** The ``replaces_first_tokens_with_copies`` rider
   is a CR 614 replacement effect on token creation, keyed on the
   Aura's ``attached_to`` host (copy template). Closing this
   requires:
     * Rust ``CardPrinting.aura_replaces_first_tokens_with_copies``
       field.
     * Per-player ``tokens_created_this_turn_count`` state slot.
     * Rust token-creation primitive that consults the active-
       player's attached Auras for the replacement on the FIRST
       token-creation per turn.
     * CardDef bridge round-trip for the new field.
     * Loader admit predicate widening.
   That's a multi-piece substrate of its own; out of scope for
   this slice.

4. **Bench-loader admit gate stays declined for both cards.** The
   ``_aura_triggers_are_phase3_safe`` gate is all-or-nothing per
   the §2 discipline — Chorale carries TWO triggers and only one
   of its Effect arms is Rust-bridged after this slice, so the
   admit gate refuses. Moonlit's S3 replacement rider stays
   positively declined at the bare-aura predicate level.

**Live-EoE admit delta: +0 cards / +0 decks.** This is a
substrate-only slice in the slice-19 (Meltstrider's Resolve) /
Pain-for-All-Effect-scaffolding (60fea01b) tradition — ships the
Rust Effect parity + bridge round-trip so the cross-engine bridge
no longer has a "no Rust translation for this Effect class" hole
on the SacrificeSelfUnlessVoidActive shape, even though no live
admit lands until the sibling reanimate substrate ships.

**Probes added in this slice:**

  * ``test_sacrifice_self_unless_void_active_bridge_roundtrips`` —
    the Rust dict ``{"kind": "sacrifice_self_unless_void_active"}``
    translates to the Python ``SacrificeSelfUnlessVoidActive()``
    instance via the cross_engine bridge.
  * ``test_sacrifice_self_unless_void_active_void_active_no_op`` —
    Void-active branch: with either void flag set, source stays on
    bf (Python parity: combinators.py:189-193).
  * ``test_sacrifice_self_unless_void_active_void_inactive_sacrifices``
    — Void-inactive branch: source moves to its owner's graveyard
    (Python parity: combinators.py:197).
  * ``test_sacrifice_self_unless_void_active_off_bf_silent_noop`` —
    source not on bf: silent no-op (Python parity: combinators.py
    :186-188).
  * ``test_sacrifice_self_unless_void_active_python_parity`` —
    Python's ``SacrificeSelfUnlessVoidActive.resolve`` mirrors the
    Rust arm: Void inactive → sacrifice; Void active → no-op.
  * ``test_chorale_of_the_void_loader_admits_post_cycle1_closure`` —
    UPDATED 2026-05-22 (Chorale of the Void Cycle 1 closure landed).
    Reverses the pre-Cycle-1 ``test_chorale_of_the_void_loader_still
    _declined`` pin: Chorale now admits at the bench loader because
    the ATTACKS-event ``ReanimateTargetCreatureCardFromDefenders
    Graveyard`` trigger has Rust Effect parity (honest-stop resolve
    arm — empty-targets silent fizzle byte-equal vs Python), and the
    ``_aura_triggers_are_phase3_safe`` gate widens to shape 7
    (Chorale's ATTACKS half) + shape 8 (Chorale's BEGINNING_OF_END
    _STEP half).
  * ``test_moonlit_meditation_loader_still_declined`` — negative
    pin: Moonlit's ``replaces_first_tokens_with_copies`` keeps it
    declined at the S3 predicate check.
  * ``test_phase4_does_not_regress_phase3_admits`` — sanity:
    Pain for All still admits end-to-end with the new bridge arm
    present.
"""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mtg_bot_engine") is None,
    reason="mtg_bot_engine PyO3 extension not installed",
)


# ---------------------------------------------------------------------
# Bridge round-trip probe
# ---------------------------------------------------------------------


def test_sacrifice_self_unless_void_active_bridge_roundtrips():
    """Cross-engine bridge: Rust's
    ``sacrifice_self_unless_void_active`` dict translates to Python's
    ``SacrificeSelfUnlessVoidActive()`` via
    ``cross_engine._ta_dict_to_python_effect``.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive

    bridged = _ta_dict_to_python_effect(
        {"kind": "sacrifice_self_unless_void_active"}
    )
    assert isinstance(bridged, SacrificeSelfUnlessVoidActive), (
        "Aura Phase 4: the cross_engine bridge arm should round-trip "
        "the Rust kind 'sacrifice_self_unless_void_active' to a Python "
        f"SacrificeSelfUnlessVoidActive instance; got {bridged!r}"
    )


# ---------------------------------------------------------------------
# Rust-side resolve-arm probes
# ---------------------------------------------------------------------


ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


def test_sacrifice_self_unless_void_active_void_active_no_op():
    """**Void-active branch: source stays on bf.**

    With ``permanent_left_battlefield_this_turn=True`` (Void
    active), the resolve arm short-circuits and the source remains
    on its controller's battlefield. Mirrors Python
    ``combinators.py:189-193``.
    """
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    rust_state = build_rust_state(spec)

    # P0's Aura source with the SacrificeSelfUnlessVoidActive trigger
    # firing on ETB (rather than end-step — same Effect, same resolve
    # arm; we drive the trigger via ENTERS for a self-contained probe).
    aura_id = rust_state.register_and_inject_object(
        {
            "name": "Void Sustain Probe (Void Active)",
            "card_types": ["Enchantment"],
            "subtypes": ["Aura"],
            "mana_cost": {"generic": 3, "B": 1},
            "effects": [],
            "triggered_abilities": [{
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
            }],
        },
        0, ZONE_BATTLEFIELD,
    )

    # Activate Void via the engine-wide flag (a nonland permanent
    # left the battlefield this turn).
    rust_state.set_void_flags(permanent_left_battlefield=True)

    # Drain the stack — Void is active, source survives.
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    # Post-resolve: Aura still on P0's bf, NOT in graveyard.
    assert rust_state.find_object_zone(aura_id) == ZONE_BATTLEFIELD, (
        "SacrificeSelfUnlessVoidActive: Void active → source stays "
        f"on bf; zone={rust_state.find_object_zone(aura_id)}, "
        f"aura_id={aura_id}"
    )


def test_sacrifice_self_unless_void_active_void_via_warp():
    """**Void-active branch via the spell-warp flag.**

    Same as above but the Void condition is satisfied by the
    ``spell_warped_this_turn`` flag (the second engine-wide void
    source). Mirrors Python's ``is_void_active`` reading either flag
    (``game.py:724``).
    """
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    rust_state = build_rust_state(spec)

    aura_id = rust_state.register_and_inject_object(
        {
            "name": "Void Sustain Probe (Warp)",
            "card_types": ["Enchantment"],
            "subtypes": ["Aura"],
            "mana_cost": {"generic": 3, "B": 1},
            "effects": [],
            "triggered_abilities": [{
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
            }],
        },
        0, ZONE_BATTLEFIELD,
    )

    rust_state.set_void_flags(spell_warped=True)

    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    assert rust_state.find_object_zone(aura_id) == ZONE_BATTLEFIELD, (
        "SacrificeSelfUnlessVoidActive: Void via warp flag → source "
        f"stays on bf; zone={rust_state.find_object_zone(aura_id)}, "
        f"aura_id={aura_id}"
    )


def test_sacrifice_self_unless_void_active_void_inactive_sacrifices():
    """**Void-inactive branch: source moves to graveyard.**

    With both engine-wide void flags False (the post-build default;
    no nonland permanent has left the bf and no spell has been
    warped), the resolve arm sacrifices the source to its owner's
    graveyard. Mirrors Python ``combinators.py:197``.
    """
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    rust_state = build_rust_state(spec)

    aura_id = rust_state.register_and_inject_object(
        {
            "name": "Void Sustain Probe (Void Inactive)",
            "card_types": ["Enchantment"],
            "subtypes": ["Aura"],
            "mana_cost": {"generic": 3, "B": 1},
            "effects": [],
            "triggered_abilities": [{
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
            }],
        },
        0, ZONE_BATTLEFIELD,
    )

    # Default-fresh state: both void flags are False.
    assert rust_state.is_void_active is False, (
        "Pre-condition: a freshly built state should have Void "
        "inactive (no LTB, no warp cast this turn)."
    )

    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    # Post-resolve: Aura moved to P0's graveyard.
    final_zone = rust_state.find_object_zone(aura_id)
    assert final_zone == ZONE_GRAVEYARD, (
        "SacrificeSelfUnlessVoidActive: Void inactive → source in "
        f"graveyard (expected zone={ZONE_GRAVEYARD}); got zone="
        f"{final_zone}, aura_id={aura_id}"
    )


def test_sacrifice_self_unless_void_active_python_parity():
    """**Python-side parity probe.**

    Mirror's the Rust probe symmetrically on Python's
    ``SacrificeSelfUnlessVoidActive`` so a future bridge tweak is
    cross-checked against the Python source of truth. Drives the
    Effect's resolve directly with two GameState shapes (Void
    active / Void inactive); asserts the byte-equal post-state.
    """
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.game import GameState, Player, Zone
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    chorale_printing = printings["Chorale of the Void"]

    # Void inactive: source is sacrificed.
    state = GameState()
    state.players.append(Player(id=0))
    state.players.append(Player(id=1))
    aura = state.new_object(chorale_printing, 0, Zone.BATTLEFIELD)
    state.battlefield.append(aura)
    state.resolving_source = aura
    SacrificeSelfUnlessVoidActive().resolve(state, controller_id=0)
    # Void inactive → sacrificed: card moves out of bf into graveyard.
    assert aura not in state.battlefield, (
        "SacrificeSelfUnlessVoidActive Python parity: Void inactive "
        "→ source leaves bf."
    )
    assert aura in state.players[0].graveyard, (
        "SacrificeSelfUnlessVoidActive Python parity: Void inactive "
        "→ source in owner's graveyard."
    )

    # Void active: source survives.
    state2 = GameState()
    state2.players.append(Player(id=0))
    state2.players.append(Player(id=1))
    state2.permanent_left_battlefield_this_turn = True
    aura2 = state2.new_object(chorale_printing, 0, Zone.BATTLEFIELD)
    state2.battlefield.append(aura2)
    state2.resolving_source = aura2
    SacrificeSelfUnlessVoidActive().resolve(state2, controller_id=0)
    assert aura2 in state2.battlefield, (
        "SacrificeSelfUnlessVoidActive Python parity: Void active "
        "→ source stays on bf."
    )
    assert aura2 not in state2.players[0].graveyard, (
        "SacrificeSelfUnlessVoidActive Python parity: Void active "
        "→ source NOT in graveyard."
    )


# ---------------------------------------------------------------------
# Negative pins: Phase 4 carriers still declined at the loader
# ---------------------------------------------------------------------


def test_chorale_of_the_void_loader_admits_post_cycle1_closure():
    """**Updated 2026-05-22 — Chorale of the Void Cycle 1 closure
    landed.** Chorale of the Void is now bench-loader-ADMITTED post-
    Cycle 1: the ATTACKS-event ``ReanimateTargetCreatureCardFrom
    DefendersGraveyard`` trigger has Rust Effect parity as of this
    lane (cards.rs unit variant + parse + to_pydict + bridge arm in
    cross_engine.py + admit-set widen shape 7 in
    ``_aura_triggers_are_phase3_safe``). The Rust resolve arm is an
    honest-stop silent no-op today (the cast-time-bound graveyard-
    creature target is not threaded through to triggered abilities;
    Python takes the same silent-fizzle on the empty-targets path),
    so cross-engine state stays byte-equal on the no-target / empty-
    graveyard probe shapes the bench-loader exercises.

    With this closure, ALL THREE Aura S4 multi-trigger siblings
    (Cracked Skull, Sporogenic Infection, Chorale of the Void) are
    bench-loader-admitted; the canonical PROBED-flip test lives at
    ``tests/test_chorale_of_the_void_parity.py``.
    """
    from mtg_bot.cards.source import OracleParserCardSource
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_aura

    src = OracleParserCardSource()
    printing = src.get_printings().get("Chorale of the Void")
    assert printing is not None, (
        "Pre-condition: Chorale of the Void must be discoverable in "
        "the live parser pool."
    )
    assert printing.is_aura, (
        "Pre-condition: Chorale of the Void must parse to is_aura=True."
    )
    assert _printing_is_bridge_simple_aura(printing) is True, (
        "Chorale of the Void Cycle 1 closure: Chorale must ADMIT at "
        "the bench loader. The ATTACKS-event Reanimate effect now has "
        "Rust Effect parity (honest-stop resolve arm — empty-targets "
        "silent fizzle matches Python byte-equal); the safe-trigger "
        "gate's shape 7 admit closes the loader path."
    )


def test_moonlit_meditation_loader_admits_post_phase_4_s3_closure():
    """**Moonlit Meditation admits at the bench loader post-Phase-4-S3
    closure (2026-05-21).**

    Pre-closure context (kept for change-tracking): Moonlit's
    ``replaces_first_tokens_with_copies=True`` rider is a CR 614
    token-creation replacement effect — the Phase-4 first-deliverable
    slice declined the rider at the loader because the Rust
    CreateToken arm did not yet consult attached Auras to swap the
    token's ``printing_id`` to the host's printing.

    Post-closure: ``moonlit_token_substitution``
    (``rust/src/state_ops_resolve.rs``) lights up the Rust consumer,
    swapping the token's ``printing_id`` to the host's printing with
    the per-object Token-supertype override stamped on the freshly-
    minted GameObject. The loader now admits Moonlit as a bridge-
    simple Aura.
    """
    from mtg_bot.cards.source import OracleParserCardSource
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_aura

    src = OracleParserCardSource()
    printing = src.get_printings().get("Moonlit Meditation")
    assert printing is not None, (
        "Pre-condition: Moonlit Meditation must be discoverable in "
        "the live parser pool."
    )
    assert printing.is_aura, (
        "Pre-condition: Moonlit Meditation must parse to is_aura=True."
    )
    assert printing.replaces_first_tokens_with_copies, (
        "Pre-condition: Moonlit Meditation must parse to "
        "replaces_first_tokens_with_copies=True (S3 sub-shape)."
    )
    assert _printing_is_bridge_simple_aura(printing) is True, (
        "Phase 4 S3 closure: Moonlit Meditation must ADMIT at the "
        "bench loader. The Rust CreateToken-family arms consult the "
        "controller's attached Moonlit-shape Aura via "
        "moonlit_token_substitution and substitute the token's "
        "printing_id with the host's printing + per-object Token-"
        "supertype override — byte-equal against Python's "
        "_find_moonlit_replacement_source + _dc.replace(base, "
        "supertypes=(*base, 'Token')) shim."
    )


def test_phase4_does_not_regress_phase3_admits():
    """**Phase 3 sanity: Pain for All still admits end-to-end with the
    new bridge arm present.**

    The Phase-3-safe-trigger gate is unchanged by this slice (we
    did NOT add SacrificeSelfUnlessVoidActive to it — the all-or-
    nothing rule keeps Chorale declined regardless). Pain for All
    should continue to pass the gate exactly as before.
    """
    from mtg_bot.cards.source import OracleParserCardSource
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_aura

    src = OracleParserCardSource()
    printing = src.get_printings().get("Pain for All")
    assert printing is not None, (
        "Pre-condition: Pain for All must be discoverable in the live "
        "parser pool."
    )
    assert _printing_is_bridge_simple_aura(printing) is True, (
        "Phase 3 regression check: Pain for All must continue to "
        "ADMIT at the bench loader after Phase 4's bridge addition."
    )


# ---------------------------------------------------------------------
# Real-EoE byte-equal probe (Chorale's substrate-landing sanity)
# ---------------------------------------------------------------------


def test_chorale_substrate_byte_equal_void_active_path_real_eoe():
    """**Real-EoE substrate-landing probe.**

    Drives the actual Chorale of the Void printing's end-step
    trigger Effect (``SacrificeSelfUnlessVoidActive``) through the
    Rust resolve arm with the void flag set, and asserts the source
    survives — the byte-equal Python branch
    (``combinators.py:189-193``). This is the substrate-landing
    sanity that the Effect resolves with the real printing's
    shape, not just the synthetic in-memory aura used in the
    earlier probes.
    """
    from mtg_bot.cards.source import OracleParserCardSource
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    src = OracleParserCardSource()
    chorale = src.get_printings().get("Chorale of the Void")
    assert chorale is not None
    # Verify the real Chorale's parsed end-step trigger IS the
    # SacrificeSelfUnlessVoidActive effect (the substrate this slice
    # lands).
    from mtg_bot.effects.combinators import SacrificeSelfUnlessVoidActive
    from mtg_bot.effects.predicates import Event

    end_step_triggers = [
        ta for ta in chorale.triggered_abilities
        if ta.event is Event.BEGINNING_OF_END_STEP
    ]
    assert len(end_step_triggers) == 1, (
        "Real-EoE Chorale of the Void should parse to exactly one "
        "BEGINNING_OF_END_STEP trigger; got "
        f"{len(end_step_triggers)}."
    )
    assert isinstance(
        end_step_triggers[0].effect, SacrificeSelfUnlessVoidActive
    ), (
        "Real-EoE Chorale of the Void: the end-step trigger's "
        "effect should be SacrificeSelfUnlessVoidActive; got "
        f"{type(end_step_triggers[0].effect).__name__}."
    )

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    rust_state = build_rust_state(spec)

    # Inject a synthetic Aura source carrying Chorale's exact
    # end-step trigger shape (the Rust dict mirror of what the
    # CardDef bridge would emit if the loader admitted Chorale).
    aura_id = rust_state.register_and_inject_object(
        {
            "name": "Chorale of the Void (real-EoE probe)",
            "card_types": ["Enchantment"],
            "subtypes": ["Aura"],
            "mana_cost": {"generic": 3, "B": 1},
            "effects": [],
            # Drive the end-step trigger as an ENTERS trigger for
            # self-contained driving (the Effect's resolve doesn't
            # care which event fired it; both routes hit the same
            # arm). Real production wiring fires this on the
            # active-player's end-step dispatch.
            "triggered_abilities": [{
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "sacrifice_self_unless_void_active"},
            }],
        },
        0, ZONE_BATTLEFIELD,
    )

    rust_state.set_void_flags(permanent_left_battlefield=True)

    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    final_zone = rust_state.find_object_zone(aura_id)
    assert final_zone == ZONE_BATTLEFIELD, (
        "Real-EoE Chorale of the Void substrate probe: Void active "
        "→ Aura stays on bf (the canonical sustain semantic). zone="
        f"{final_zone}, aura_id={aura_id}"
    )
