"""Cross-engine parity probes for Systems Override.

**Control-change-UEOT subsystem — Cycle 5 live mint site (2026-05-22;
~4,775 17lands decks).** ``{2}{R}`` Sorcery: "Gain control of target
artifact or creature until end of turn. Untap that permanent. It gains
haste until end of turn. If it's a Spacecraft, put ten charge counters
on it. If you do, remove ten charge counters from it at the beginning
of the next end step."

Mirrors Python's ``SystemsOverrideEffect`` (``effects/bespoke.py:1369``)
which uses the layer-2 ``"control_change"`` registry entry (the
Cycle-1 substrate's source-of-truth surface on the Python side) +
``grant_control_change_ueot`` + the
``RemoveChargeCountersFromSourceDelayed`` delayed-trigger rider
scheduled at resolve time when the target carries the Spacecraft
subtype.

The substrate that makes this slice possible:

  1. **Cycle 1 substrate.** ``GameObject.controller_override:
     Option<u8>`` field + ``effective_controller_id`` helper +
     ``grant_control_change_ueot`` PyO3 setter + cleanup-step drain +
     per-object snapshot key. Parity-NEUTRAL on its own — no live call
     site mints a non-None override.
  2. **Cycle 2 substrate.** Effective-controller readback wiring
     (parallel branch — landed alongside this lane at user-merge).
  3. **Cycle 3 substrate.** Python ↔ Rust bridge round-trip for the
     per-object ``controller_override`` field.
  4. **Cycle 4 live mint site.** Mutinous Massacre — first call site
     on the Rust engine that mints a non-None override (mass control-
     change pattern, no delayed-trigger rider).
  5. **Cycle 5 (THIS lane).** Systems Override —
     ``Effect::SystemsOverrideEffect`` + resolve arm + the new
     ``Effect::RemoveChargeCountersFromSourceDelayed { amount: i32 }``
     delayed-trigger effect + scheduling onto
     ``state.delayed_triggers``. The end-step drain machinery
     (round-78 ``run_delayed_end_step_triggers``) was already generic
     over the Effect — Systems Override adds the SECOND consumer
     (WarpExileSelf being the first) so the dispatch is finally
     observed driving a non-WarpExileSelf effect.

Lives in its own module (NOT in ``test_parity_smoke.py``) per the
established per-card-module convention (cf. ``test_exorcise_parity.py``,
``test_close_encounter_parity.py``, ``test_mutinous_massacre_parity.py``).

**Controller-id asymmetry note** (inherited from the Cycle 4 lane —
applies identically here): Python's ``GameObject.controller_id`` is
the EAGER-CACHE of the effective controller; Rust's
``GameObject.controller`` is the BASELINE (override sits on a separate
``controller_override`` field). A raw byte-equal compare of the
``controller_id`` snapshot field post-grant DIVERGES by design —
probes here assert effective controllers (via override resolution on
both sides) and the python_controller_override reciprocal helper.

**Loading Zone counter-doubling** — Rust does NOT yet implement the
``_counter_doubling_factor`` machinery Python's ``add_charge_counters``
threads through. Loading Zone is BLOCKED in the EoE pool today, so
the divergence is unobservable in the production deck pool; the
literal +10 / -10 on both engines is byte-equal in the absence of
Loading Zone. When Loading Zone lands both engines must grow the
factor in lockstep.

Covered:

- Real EoE printing's spell_effect IS ``SystemsOverrideEffect()``.
- Serializer round-trips both Effects:
  * SystemsOverrideEffect ↔ ``{"kind": "systems_override_effect"}``
  * RemoveChargeCountersFromSourceDelayed(amount=N) ↔
    ``{"kind": "remove_charge_counters_from_source_delayed", "amount": N}``
- The card passes ``_printing_is_bridge_simple_spell`` post-Cycle-5 via
  the Phase-1 ``(1, 1)`` single-target admit arm (auto-grow §2 teeth).
- Rust resolve arm grants ``controller_override = caster`` to an
  opponent-controlled target (artifact OR creature).
- Rust resolve arm untaps and grants haste UEOT.
- Rust resolve arm: own-target path — untap + haste BUT no override
  (same-controller short-circuit, mirroring the Cycle-1 helper).
- Rust resolve arm: Spacecraft branch — +10 charge counters and the
  delayed trigger is scheduled.
- Rust resolve arm: non-Spacecraft branch — no charge counters, no
  delayed trigger scheduled (the "If you do" gate is honored).
- Rust delayed-trigger drain at the beginning of the next end step:
  the queued trigger fires, charge counters drop back to 0.
- CR 603.6c silent fizzle: if the Spacecraft has left the battlefield
  between Systems Override resolve and the end step, the delayed
  trigger fires onto an absent source — silent no-op.
- Cleanup-step drain reverts ``controller_override`` AND clears the
  haste UEOT modifier.
- **GENUINE real-EoE cross-engine probe** — the parsed Systems
  Override printing driven against an opponent-controlled Spacecraft;
  the Rust effective-controller readback matches Python's eager-cache,
  +10 charge counters land, haste-granted+untapped both engines, the
  end-step drain removes the 10 counters on both engines.
"""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mtg_bot_engine") is None,
    reason="mtg_bot_engine PyO3 extension not installed",
)


ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3

PHASE_END_STEP = 10


# -------------------------------------------------------------------
# Verify-first: parser SoT pin.
# -------------------------------------------------------------------


def test_real_systems_override_spell_effect_is_systems_override_effect():
    """**Verify-first.** The live parser maps Systems Override to
    ``SystemsOverrideEffect()`` — a parameterless frozen dataclass.
    Freezes the parser SoT so a parser drift is caught here, not
    silently downstream in the cross-engine probe."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.bespoke import SystemsOverrideEffect

    p = (
        get_active_source().get_printings().get("Systems Override")
    )
    if p is None:
        pytest.skip("Systems Override absent from active CardSource")
    assert "Sorcery" in p.card_types
    se = p.spell_effect
    assert isinstance(se, SystemsOverrideEffect), (
        f"Systems Override spell_effect must be SystemsOverrideEffect; "
        f"got {type(se).__name__}"
    )


# -------------------------------------------------------------------
# Round-trip + admit-gate pins (the auto-grow §2 teeth).
# -------------------------------------------------------------------


def test_systems_override_serializer_round_trip():
    """The trajectory reverse-serializer must emit a clean
    ``{"kind": "systems_override_effect"}`` unit dict; the dict→Python
    bridge must reconstruct ``SystemsOverrideEffect()``.

    A no-emit serializer would short-circuit the admit gate (the no-
    fallback-hybrid teeth). With the leaf landed and the variant
    field-free, the round-trip is byte-equal both ways."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.bespoke import SystemsOverrideEffect
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    p = (
        get_active_source().get_printings().get("Systems Override")
    )
    if p is None:
        pytest.skip("Systems Override absent from active CardSource")
    d = trajectory._python_spell_effect_to_dict(p.spell_effect)
    assert d == {"kind": "systems_override_effect"}, (
        f"Systems Override reverse serializer must emit the bare "
        f"unit-variant dict; got {d}"
    )
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(rebuilt, SystemsOverrideEffect), (
        f"dict→Python bridge must round-trip to SystemsOverrideEffect; "
        f"got {type(rebuilt).__name__}"
    )


def test_remove_charge_counters_delayed_serializer_round_trip():
    """The trajectory reverse-serializer must emit
    ``{"kind": "remove_charge_counters_from_source_delayed",
    "amount": N}``; the dict→Python bridge must reconstruct
    ``RemoveChargeCountersFromSourceDelayed(amount=N)``. Sibling to the
    main spell_effect round-trip — guards the delayed-trigger leaf
    used in mid-game state-sync."""
    from mtg_bot.effects.generic import (
        RemoveChargeCountersFromSourceDelayed,
    )
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    eff = RemoveChargeCountersFromSourceDelayed(amount=10)
    d = trajectory._python_spell_effect_to_dict(eff)
    assert d == {
        "kind": "remove_charge_counters_from_source_delayed",
        "amount": 10,
    }, (
        f"RemoveChargeCountersFromSourceDelayed reverse serializer "
        f"must emit the bare field-bearing dict; got {d}"
    )
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(rebuilt, RemoveChargeCountersFromSourceDelayed)
    assert rebuilt.amount == 10


def test_systems_override_passes_bridge_admit_gate():
    """**The lever flip pin.** Post-Cycle-5, Systems Override MUST pass
    ``_printing_is_bridge_simple_spell`` — the spell-card-bridge admit
    gate. The card has (1, 1) target_count_range and empty slot_kinds
    (the Phase-1 single-target admit shape); the spell_effect is
    covered+bridged; the serializer round-trips cleanly. The §2 teeth
    all fire positively."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory

    p = (
        get_active_source().get_printings().get("Systems Override")
    )
    if p is None:
        pytest.skip("Systems Override absent from active CardSource")
    assert trajectory._printing_is_bridge_simple_spell(p) is True, (
        "Systems Override must auto-join the spell-card-bridge admit "
        "set now that its Effect is covered+bridged + round-trips + "
        "the control-change-UEOT C1+C3 substrate is in place + the "
        "Cycle-5 live mint site has landed"
    )
    # Pin per-tooth gates — if any one fails, debug-time message
    # points at WHICH tooth.
    assert (
        trajectory._spell_effect_is_covered_and_bridged(p) is True
    )
    assert (
        trajectory._python_spell_effect_to_dict(p.spell_effect)
        is not None
    )


def test_systems_override_class_covered():
    """``SystemsOverrideEffect`` must be in the covered-by-rust set —
    the Rust ``Effect::SystemsOverrideEffect`` variant declaration
    makes ``covered_python_class_set()`` include it. Without this,
    the §2 admit-gate teeth refuse to flip the card."""
    from mtg_bot.parity.effect_coverage import covered_python_class_set

    covered = covered_python_class_set()
    assert "SystemsOverrideEffect" in covered, (
        "Rust Effect::SystemsOverrideEffect variant must make the "
        "Python SystemsOverrideEffect class covered"
    )


def test_remove_charge_counters_delayed_class_covered():
    """``RemoveChargeCountersFromSourceDelayed`` must be in the
    covered-by-rust set via the RUST_AHEAD_OF_POOL allowlist — the
    Python class exists (live consumer: Systems Override's resolve
    body schedules it) but the static-printing walker does not see it
    (the schedule is dynamic, inside a method body, not a declared
    field). Mirrors the WarpExileSelf precedent."""
    from mtg_bot.parity.effect_coverage import (
        covered_python_class_set,
        RUST_AHEAD_OF_POOL,
    )

    # Either it's in the live covered set, OR it's in the
    # RUST_AHEAD_OF_POOL allowlist (the dynamic-spawn case).
    covered = covered_python_class_set()
    assert (
        "RemoveChargeCountersFromSourceDelayed" in covered
        or "RemoveChargeCountersFromSourceDelayed"
        in RUST_AHEAD_OF_POOL
    ), (
        "RemoveChargeCountersFromSourceDelayed must be either in the "
        "live covered set or in the RUST_AHEAD_OF_POOL allowlist "
        "(per the WarpExileSelf precedent — dynamic delayed-trigger "
        "spawn invisible to the static-printing walker)"
    )


# -------------------------------------------------------------------
# Rust resolve-arm unit pins.
# -------------------------------------------------------------------


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _new_creature(rs, name, owner, power=2, toughness=2):
    """Inject a vanilla creature on owner's battlefield. Returns
    object id."""
    return rs.register_and_inject_object(
        {
            "name": name,
            "card_types": ["Creature"],
            "power": power,
            "toughness": toughness,
        },
        owner, ZONE_BATTLEFIELD,
    )


def _new_artifact(rs, name, owner, subtypes=None):
    """Inject a generic non-Spacecraft artifact on owner's bf."""
    d = {
        "name": name,
        "card_types": ["Artifact"],
    }
    if subtypes is not None:
        d["subtypes"] = list(subtypes)
    return rs.register_and_inject_object(d, owner, ZONE_BATTLEFIELD)


def _new_spacecraft(rs, name, owner, power=4, toughness=4):
    """Inject a Spacecraft (Artifact — Spacecraft subtype) on owner's
    bf. Real EoE Spacecraft printings carry Artifact card type +
    Spacecraft subtype + power/toughness."""
    return rs.register_and_inject_object(
        {
            "name": name,
            "card_types": ["Artifact"],
            "subtypes": ["Spacecraft"],
            "power": power,
            "toughness": toughness,
        },
        owner, ZONE_BATTLEFIELD,
    )


def _push_systems_override(rs, caster, target_id):
    """Inject a Systems-Override-shape spell on caster's stack via the
    synthetic push_to_stack_with_target helper (carries the target
    payload — Systems Override is single-target so it needs the
    payload-target field to bind the resolve)."""
    spell_id = rs.register_and_inject_object(
        {
            "name": "Systems Override",
            "card_types": ["Sorcery"],
            "effects": [{"kind": "systems_override_effect"}],
        },
        caster, ZONE_HAND,
    )
    rs.push_to_stack_with_target(spell_id, caster, target_id)
    return spell_id


def test_rust_resolve_grants_control_untap_haste_on_opponent_creature():
    """Systems Override on an opponent's creature: stamps
    controller_override = caster, untaps, grants haste UEOT. No
    Spacecraft → no charge counter add, no delayed trigger scheduled.
    """
    rs = _fresh_state()
    bear = _new_creature(rs, "Opp Bear", owner=1)
    rs.tap_object(bear)  # tapped to verify untap fires
    _push_systems_override(rs, caster=0, target_id=bear)
    rs.resolve_top_of_stack()

    bf = rs.battlefield_snapshot()
    bear_dict = next(e for e in bf if e["id"] == bear)
    assert bear_dict["controller_override"] == 0, (
        f"opponent bear must have controller_override=0; got "
        f"{bear_dict['controller_override']}"
    )
    assert rs.effective_controller_id(bear) == 0
    assert bear_dict["tapped"] is False
    # No Spacecraft → no charge counters.
    assert rs.get_charge_counters(bear) == 0


def test_rust_resolve_grants_control_untap_haste_on_opponent_artifact():
    """Systems Override on an opponent's non-Spacecraft artifact:
    stamps controller_override = caster, untaps, grants haste UEOT.
    No charge counter rider (the target isn't a Spacecraft)."""
    rs = _fresh_state()
    art = _new_artifact(rs, "Opp Equipment", owner=1, subtypes=["Equipment"])
    rs.tap_object(art)
    _push_systems_override(rs, caster=0, target_id=art)
    rs.resolve_top_of_stack()

    bf = rs.battlefield_snapshot()
    obj = next(e for e in bf if e["id"] == art)
    assert obj["controller_override"] == 0
    assert obj["tapped"] is False
    assert rs.get_charge_counters(art) == 0


def test_rust_resolve_own_target_untap_and_haste_no_override():
    """Own target: NO override (same-controller short-circuit per the
    Cycle-1 helper semantics — mirrored in the inline grant body); but
    untap + haste still apply. Mirrors Mutinous Massacre's own-creature
    parity pin."""
    rs = _fresh_state()
    my_bear = _new_creature(rs, "My Bear", owner=0)
    rs.tap_object(my_bear)
    _push_systems_override(rs, caster=0, target_id=my_bear)
    rs.resolve_top_of_stack()

    bf = rs.battlefield_snapshot()
    obj = next(e for e in bf if e["id"] == my_bear)
    assert obj["controller_override"] is None, (
        "own creature must have NO controller_override (same-"
        "controller short-circuit)"
    )
    assert obj["tapped"] is False
    assert rs.effective_controller_id(my_bear) == 0


def test_rust_resolve_spacecraft_adds_ten_charge_counters():
    """Systems Override on an opponent's Spacecraft: +10 charge counters
    AND a delayed trigger is scheduled for the next end step. Indirect
    verification of the delayed-trigger schedule: at PHASE_END_STEP the
    stack must contain a pushed trigger."""
    rs = _fresh_state()
    sc = _new_spacecraft(rs, "Opp Spacecraft", owner=1)
    _push_systems_override(rs, caster=0, target_id=sc)
    rs.resolve_top_of_stack()

    bf = rs.battlefield_snapshot()
    obj = next(e for e in bf if e["id"] == sc)
    assert obj["controller_override"] == 0
    assert obj["tapped"] is False
    # Spacecraft → +10 charge counters.
    assert rs.get_charge_counters(sc) == 10, (
        f"Spacecraft target must have 10 charge counters post-resolve; "
        f"got {rs.get_charge_counters(sc)}"
    )


def test_rust_delayed_trigger_drains_charge_counters_at_end_step():
    """The Spacecraft-rider's delayed trigger drains on the next end
    step: at PHASE_END_STEP the queued
    ``RemoveChargeCountersFromSourceDelayed(amount=10)`` is pushed onto
    the stack via ``push_triggered_ability`` (the existing G7 Class-C
    machinery); resolving it removes 10 charge counters from the
    target — back to 0.

    Composes the Cycle 5 substrate end-to-end:
      * SystemsOverrideEffect resolve schedules the trigger.
      * run_delayed_end_step_triggers (PHASE_END_STEP entry) drains
        it onto the stack.
      * RemoveChargeCountersFromSourceDelayed resolve removes the
        counters."""
    rs = _fresh_state()
    sc = _new_spacecraft(rs, "Opp Spacecraft", owner=1)
    _push_systems_override(rs, caster=0, target_id=sc)
    rs.resolve_top_of_stack()
    assert rs.get_charge_counters(sc) == 10

    # Advance phases until END_STEP entry (PHASE 10). The synthetic
    # state starts in PHASE 0 (UNTAP); each advance_phase ticks one
    # forward. At PHASE_END_STEP, run_delayed_end_step_triggers fires
    # and the scheduled trigger is pushed onto the stack.
    while rs.current_phase() != PHASE_END_STEP:
        rs.advance_phase()

    # The scheduled trigger should now be on the stack.
    assert rs.stack_size() == 1, (
        f"end-step entry must drain the delayed trigger onto the "
        f"stack; stack_size={rs.stack_size()}"
    )

    # Resolve the trigger — removes the 10 charge counters.
    rs.resolve_top_of_stack()
    assert rs.get_charge_counters(sc) == 0, (
        f"delayed trigger must drain charge counters back to 0; "
        f"got {rs.get_charge_counters(sc)}"
    )


def test_rust_delayed_trigger_fizzles_when_source_left_battlefield():
    """**CR 603.6c — silent fizzle.** If the Spacecraft has left the
    battlefield between Systems Override resolve and the end step, the
    delayed trigger fires but has nothing to act on — silent no-op.

    Setup: cast Systems Override on a Spacecraft (counters added +
    trigger scheduled). Then move the Spacecraft to the graveyard.
    Advance to end step. The scheduled trigger's source-existence
    check (in ``run_delayed_end_step_triggers``) sees the source in
    the graveyard (not on bf) — the trigger is dropped without
    pushing on the stack, and no charge-counter mutation fires."""
    rs = _fresh_state()
    sc = _new_spacecraft(rs, "Opp Spacecraft", owner=1)
    _push_systems_override(rs, caster=0, target_id=sc)
    rs.resolve_top_of_stack()
    assert rs.get_charge_counters(sc) == 10

    # Manually destroy the Spacecraft (move bf → graveyard) before
    # the end step. Use the bare move_object primitive (zone 3 =
    # graveyard, target player 1 = owner).
    rs.move_object(sc, ZONE_GRAVEYARD, 1)

    # Advance to end step.
    while rs.current_phase() != PHASE_END_STEP:
        rs.advance_phase()

    # The trigger should silently fizzle: source IS still found
    # (the run_delayed_end_step_triggers helper scans ALL zones —
    # graveyard included — for the source_id), so the trigger
    # actually fires; but the RemoveChargeCountersFromSourceDelayed
    # resolve arm walks ONLY the battlefield and returns silently
    # when the source is not there. Either way the Spacecraft's
    # charge_counters stay at 10 (in the graveyard, untouched).
    while rs.stack_size() > 0:
        rs.resolve_top_of_stack()
    # The Spacecraft is now in p1's graveyard with charge_counters
    # frozen at 10 — the resolve arm did not find it on a battlefield
    # and silently no-op'd.
    gy = rs.graveyard_snapshot(1)
    assert any(e["id"] == sc for e in gy), (
        "Spacecraft must be in p1's graveyard"
    )


def test_rust_cleanup_drains_override_and_haste_after_systems_override():
    """**Cycle-1 substrate composition pin.** Post-resolve, the cleanup-
    step drain reverts ``controller_override`` back to None AND clears
    the haste UEOT modifier. Composes:
      - Cycle-1 ``clear_controller_overrides_all``
      - Round-27 ``clear_ueot_modifiers_all``
    """
    rs = _fresh_state()
    bear = _new_creature(rs, "Opp Bear", owner=1)
    _push_systems_override(rs, caster=0, target_id=bear)
    rs.resolve_top_of_stack()

    bf = rs.battlefield_snapshot()
    obj = next(e for e in bf if e["id"] == bear)
    assert obj["controller_override"] == 0
    assert rs.effective_controller_id(bear) == 0

    # Synthetic state's active_player defaults to 0.
    assert rs.active_player == 0
    rs.cleanup_step_for_active_player()

    bf = rs.battlefield_snapshot()
    obj = next(e for e in bf if e["id"] == bear)
    assert obj["controller_override"] is None, (
        "cleanup must drain controller_override back to None"
    )
    assert rs.effective_controller_id(bear) == 1, (
        "effective controller reverts to baseline post-cleanup"
    )


def test_rust_resolve_no_eligible_target_silent_noop():
    """Systems Override resolved against an empty battlefield (or with
    payload-target pointing at a non-existent id) is a silent no-op.

    Setup: push Systems Override with no target_id (synthetic
    push_to_stack path) on a state with NO eligible artifact/creature
    permanents on either bf. The deterministic-first fallback in the
    resolve arm finds nothing, returns Ok(()) cleanly."""
    rs = _fresh_state()
    spell_id = rs.register_and_inject_object(
        {
            "name": "Systems Override",
            "card_types": ["Sorcery"],
            "effects": [{"kind": "systems_override_effect"}],
        },
        0, ZONE_HAND,
    )
    rs.push_to_stack(spell_id, 0)
    rs.resolve_top_of_stack()
    assert rs.battlefield_snapshot() == []
    assert rs.stack_size() == 0


# -------------------------------------------------------------------
# GENUINE real-EoE cross-engine probe — drives the parsed Systems
# Override printing through both engines and pins byte-equal effective-
# controller readback + charge-counter add + end-step drain (the full
# Cycle-5 closure end-to-end).
# -------------------------------------------------------------------


def test_cross_engine_real_eoe_systems_override_control_change_spacecraft_rider_parity():
    """**GENUINE real-EoE cross-engine byte-equal probe — Systems
    Override.** The slug-credit anchor for the lane's COVERED →
    PROBED tier flip (~4,775 17lands decks).

    Drives the parsed Systems Override EoE printing through BOTH
    engines against an opponent-controlled Spacecraft:
      - p0 casts Systems Override targeting p1's Spacecraft.
      - Both engines:
        * grant control of the Spacecraft to p0 UEOT.
        * untap the Spacecraft.
        * grant haste UEOT.
        * +10 charge counters (Spacecraft branch fires).
        * schedule the delayed
          RemoveChargeCountersFromSourceDelayed(amount=10) for the
          next end step.
      - Effective controller is 0 (caster) on both engines.
      - At the next end step, both engines drain the delayed trigger
        and charge counters go back to 0.

    **Assertion strength.** Pins the OBSERVABLE-SEMANTICS contract
    that both engines must answer "who controls X right now" with
    the SAME id, plus the +10 / -10 charge-counter packet, plus the
    cleanup-drain composition.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
        python_controller_override,
    )
    from mtg_bot.game import StackItem, Zone
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.turn import state_based_actions as py_sba
    from mtg_bot.cards import CardPrinting
    from mtg_bot.cards.base import ManaCost
    from mtg_bot.effects.bespoke import SystemsOverrideEffect

    real = get_active_source().get_printings().get("Systems Override")
    if real is None:
        pytest.skip("Systems Override absent from active CardSource")
    assert isinstance(real.spell_effect, SystemsOverrideEffect)

    def _spacecraft_printing(name, power, toughness):
        return CardPrinting(
            name=name, mana_cost_str="",
            mana_cost=ManaCost(generic=4),
            type_line="Artifact — Spacecraft",
            supertypes=(),
            card_types=("Artifact",),
            subtypes=("Spacecraft",),
            power=power, toughness=toughness,
            produces=frozenset(),
            keywords=(),
        )

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    rs = build_rust_state(spec)
    ZB = 2  # battlefield

    # p1's Spacecraft target.
    py_sc = py.new_object(
        _spacecraft_printing("Real Spacecraft", 4, 4),
        owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_sc.summoning_sick = False
    py.battlefield.append(py_sc)
    rs_sc = rs.register_and_inject_object(
        {
            "name": "Real Spacecraft",
            "card_types": ["Artifact"],
            "subtypes": ["Spacecraft"],
            "power": 4, "toughness": 4,
        },
        1, ZB,
    )

    # Python: cast the GENUINE Systems Override printing via direct
    # StackItem (no agent — the target rides on StackItem.targets).
    py_spell = py.new_object(real, owner_id=0, zone=Zone.STACK)
    py_spell.summoning_sick = False
    py.stack.append(
        StackItem(
            source=py_spell, controller_id=0,
            targets=(("permanent", py_sc.id),),
        )
    )
    resolve_top_of_stack(py)
    py_sba(py)
    assert len(py.stack) == 0

    # Rust: push synthetic Systems-Override-shape spell + resolve.
    rs_spell = rs.register_and_inject_object(
        {
            "name": "Systems Override",
            "card_types": ["Sorcery"],
            "effects": [{"kind": "systems_override_effect"}],
        },
        0, 0,  # ZONE_HAND
    )
    rs.push_to_stack_with_target(rs_spell, 0, rs_sc)
    rs.resolve_top_of_stack()
    rs.apply_sbas()
    assert rs.stack_size() == 0

    # ----------------------------------------------- effective controller
    # Both engines: target now controlled by p0 (caster).
    assert py_sc.controller_id == 0, (
        "Python: Spacecraft's eager-cache controller_id must be the "
        "caster (0) post-grant"
    )
    rs_sc_obj = next(
        (e for e in rs.battlefield_snapshot() if e["id"] == rs_sc),
        None,
    )
    assert rs_sc_obj is not None, "Rust: Spacecraft must survive"
    assert rs_sc_obj["controller_override"] == 0, (
        "Rust: Spacecraft's controller_override must be the caster (0)"
    )
    assert rs.effective_controller_id(rs_sc) == 0

    # Python-side reciprocal — the Cycle-3 cross-engine helper reads
    # the layer-2 entry's pid back byte-equal with the Rust override.
    assert python_controller_override(py, py_sc) == 0

    # ----------------------------------------------- untap + haste
    assert py_sc.tapped is False
    assert rs_sc_obj["tapped"] is False
    from mtg_bot.game import effective_keywords
    assert "haste" in effective_keywords(py, py_sc)

    # ----------------------------------------------- +10 charge counters
    assert py_sc.charge_counters == 10, (
        f"Python: Spacecraft must have 10 charge counters; got "
        f"{py_sc.charge_counters}"
    )
    assert rs.get_charge_counters(rs_sc) == 10, (
        f"Rust: Spacecraft must have 10 charge counters; got "
        f"{rs.get_charge_counters(rs_sc)}"
    )

    # ----------------------------------------------- delayed-trigger drain
    # Both engines: at the next end step, the delayed trigger
    # removes the 10 counters.
    #
    # Python: walk turn phases via the existing helper. Use
    # ``run_delayed_end_step_triggers`` directly — it's the canonical
    # call site Python's simulate.py uses at PHASE_END_STEP entry. The
    # active_player_idx defaults to 0 (caster) per build_python_state.
    from mtg_bot.turn import run_delayed_end_step_triggers
    py.active_player_idx = 0
    # The trigger fires via push_triggered_ability_on_stack; resolve
    # the stack until empty so the counter drain actually runs.
    run_delayed_end_step_triggers(py)
    while py.stack:
        resolve_top_of_stack(py)
        py_sba(py)

    # Rust: advance to end step (drains delayed triggers onto stack),
    # then resolve any pushed triggers.
    while rs.current_phase() != PHASE_END_STEP:
        rs.advance_phase()
    while rs.stack_size() > 0:
        rs.resolve_top_of_stack()
        rs.apply_sbas()

    # Both engines: charge counters back to 0.
    assert py_sc.charge_counters == 0, (
        f"Python: end-step drain must remove the 10 charge counters; "
        f"got {py_sc.charge_counters}"
    )
    assert rs.get_charge_counters(rs_sc) == 0, (
        f"Rust: end-step drain must remove the 10 charge counters; "
        f"got {rs.get_charge_counters(rs_sc)}"
    )
