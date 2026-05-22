"""Darkness-shape combat-damage prevention — Cycle 2 (routing).

Spec: lane brief (2026-05-21). The Rust side now routes
``deal_combat_damage`` through the
``prevent_all_combat_damage_this_turn`` flag added in Cycle 1.
When the flag is set, ``deal_combat_damage`` short-circuits to a
no-op return BEFORE the plan is built — no per-creature damage
markers updated, no triggers fired, no lifelink life-gain, no
``resolving_combat_damage_amount`` slot write, no stack push.

CR 615.4 — "If a damage event would be prevented entirely, it
doesn't happen at all." Cycle 2 implements the routing-point
guard; Cycle 1 already provided the substrate (flag + setter /
getter / standalone drain + cleanup-step drain + summary key).
Cycle 3 will mint the flag from a live Effect (Darkness's
``Effect::PreventAllCombatDamageThisTurn`` resolve arm + bridge
round-trip + admit-shape widen + PROBED flip).

**Substrate-only invariant.** No production trajectory mints the
flag today; the guard is a no-op on every flag-unset path, so
the Cycle 2 landing is byte-equivalent on every pre-Cycle-3
cross-engine trajectory (back-compat baseline).

Tests cover (13 pins — > the 12+ floor in the lane brief):

1. **Flag-false routing path is back-compat byte-equivalent.**
   A vanilla 2/2 attacker hits the defender for 2 under the
   default (flag-unset) state. Mirrors a known pre-existing
   combat-damage test scenario; pins that the Cycle 2 guard is
   a no-op on the default path.
2. **Flag-true zeroes out unblocked player damage.** A 2/2
   attacker against an undefended opponent: with the flag set,
   the defender's life is untouched.
3. **Flag-true zeroes out blocker damage.** A 2/2 attacker
   blocked by a 2/2: with the flag set, neither creature is
   marked with damage. (No SBA destruction either.)
4. **Flag-true blocks ALL combat damage from multiple attackers.**
   Two 2/2 attackers, one undefended, one blocked: with the
   flag set, no damage applied anywhere. Single guard at the
   routing site catches both.
5. **Flag-true with lifelink: no life gain.** A 3/3 lifelink
   attacker against an undefended opponent: with the flag set,
   the attacker's controller does NOT gain life (lifelink
   triggers on damage dealt; zero damage means no trigger).
6. **Flag-true with "when this deals damage to a player"
   trigger: trigger does NOT fire.** An attacker carrying a
   ``deals_damage`` SourceIsSelf draw-card trigger; with the
   flag set, the stack remains empty post-``deal_combat_damage``.
7. **Flag-true with "when this is dealt damage" trigger:
   trigger does NOT fire.** A blocker carrying a ``dealt_damage``
   SourceIsSelf trigger; with the flag set, the trigger doesn't
   land on the stack (the blocker received zero damage, so no
   DealtDamage event was dispatched).
8. **Flag-true with trample: no spillover to defender.** A 5/5
   trample attacker blocked by a 2/2: with the flag set, neither
   the blocker nor the defending player takes damage.
9. **Flag-true does NOT affect non-combat damage.** The flag
   guards ``deal_combat_damage`` only; ``deal_damage`` (the
   non-combat primitive in ``state_ops_damage.rs``) is
   unaffected.
10. **Cleanup-step drain restores combat damage on next turn.**
    Grant the flag, call ``deal_combat_damage`` (no-op), call
    ``cleanup_step_for_active_player`` (drains the flag), then
    declare attackers + ``deal_combat_damage`` on the next turn
    — damage applies normally. Pins the Cycle 1 drain
    composition with the Cycle 2 guard.
11. **Flag-true + ``resolving_combat_damage_amount`` slot stays
    untouched.** The I32 write site is inside ``apply_damage_
    pass``; the Cycle 2 short-circuit returns BEFORE
    ``apply_damage_pass`` is called, so the slot is not written.
12. **Flag-true and no attackers: still a no-op.** A combat
    damage step with no attackers declared and the flag set
    returns cleanly without side effects.
13. **CoW invariant on the flag-true path.** Granting the flag
    on a cloned state and running combat damage on the clone
    doesn't affect the source's combat damage behavior on a
    parallel trajectory.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


# Zone constants — mirror ``rust/src/state_ops_zones.rs``.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3
ZONE_EXILE = 5

# Synthetic state id layout (defaults: 10 battlefield, 7 hand, 50 library
# per side). Same constants as ``tests_rust/test_combat.py``.
P0_FIRST_BEAR_ID = 5  # battlefield slots 5..9 are Grizzly Bears (2/2)
P1_FIRST_BEAR_ID = 72  # p1 battlefield slots 72..76 are Grizzly Bears (2/2)


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _fresh_state():
    """Default-sized synthetic state (10 bf / 7 hand / 50 library per
    side). Pre-built creature slots P0_FIRST_BEAR_ID + 0..4 +
    P1_FIRST_BEAR_ID + 0..4 are 2/2 Grizzly Bears available as
    attackers / blockers."""
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state()


def _keyword_creature_dict(power, toughness, keywords):
    """A creature printing with the given P/T and keyword list.
    Mirrors ``tests_rust/test_combat.py::_keyword_creature_dict``;
    duplicated here so the Cycle 2 tests don't take a hard dep on
    that file's private helper.
    """
    return {
        "name": "Cycle2 " + "+".join(keywords) + f" {power}/{toughness}",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "power": power,
        "toughness": toughness,
        "keywords": list(keywords),
        "effects": [],
    }


def _deals_damage_self_draw_dict():
    """A 2/2 creature with a ``source_is_self`` DealsDamage trigger
    that draws 1 card. When this attacker hits a player for >0
    combat damage, the trigger lands on the stack. With the
    Cycle 2 prevention flag set, the underlying
    ``dispatch_combat_damage_to_player`` is never reached — so the
    trigger does NOT fire.
    """
    return {
        "name": "Cycle2 DealsDamage SelfDraw 2/2",
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "effects": [],
        "triggered_abilities": [
            {
                "event": "deals_damage",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _dealt_damage_self_draw_dict():
    """A 4/4 creature with a ``source_is_self`` DealtDamage trigger
    (RR Pin 3 shape) that draws 1 card. When this creature
    RECEIVES nonzero combat damage, the trigger lands on the stack.
    With the Cycle 2 prevention flag set, ``apply_damage_pass``'s
    per-target damage marker is never written — so no DealtDamage
    event is dispatched, and the trigger does NOT fire.

    Toughness 4 so the blocker survives a 2/2 attacker's hit on
    the flag-FALSE path (so the trigger fires there for the
    parity baseline).
    """
    return {
        "name": "Cycle2 DealtDamage SelfDraw 4/4",
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 3},
        "power": 1,
        "toughness": 4,
        "effects": [],
        "triggered_abilities": [
            {
                "event": "dealt_damage",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Pin 1 — flag-false routing path is back-compat byte-equivalent
# ---------------------------------------------------------------------------


def test_flag_false_combat_damage_applies_normally():
    """Default (flag unset) state: a 2/2 attacker hits the
    defender for 2. Pins that the Cycle 2 guard is a no-op on the
    flag-unset path — every pre-Cycle-3 cross-engine trajectory
    is byte-equivalent through the Cycle 2 landing because no
    production caller mints the flag yet.

    Mirrors the same exchange shape exercised by the pre-Cycle-1
    test ``tests_rust/test_combat.py::test_unblocked_attacker_
    deals_damage_to_defender`` — choose the simplest unblocked
    scenario so the back-compat claim is unambiguous.
    """
    _require_engine()
    state = _fresh_state()
    # Flag MUST be False at default — pin the Cycle 1 substrate
    # invariant alongside the Cycle 2 routing pin.
    assert state.prevent_all_combat_damage_this_turn is False
    assert state.get_life(1) == 20
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    state.deal_combat_damage()
    # Flag-unset → routing applied normally → defender took 2.
    assert state.get_life(1) == 18, (
        "flag-unset routing path must apply combat damage normally — "
        "byte-equivalent with pre-Cycle-1 trajectories"
    )


# ---------------------------------------------------------------------------
# Pin 2 — flag-true zeroes out unblocked player damage
# ---------------------------------------------------------------------------


def test_flag_true_unblocked_attacker_deals_zero_to_defender():
    """Flag set: the unblocked 2/2 attacker's hit on the defender
    is ZEROED. The defender's life is unchanged. CR 615.4 — "if a
    damage event would be prevented entirely, it doesn't happen
    at all" — and the routing point is ``deal_combat_damage``
    itself, not a per-entry filter.
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    assert state.get_life(1) == 20
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    state.deal_combat_damage()
    # Defender's life is untouched.
    assert state.get_life(1) == 20, (
        "flag-set routing path must zero out combat damage to player "
        "(CR 615.4 — prevented event doesn't happen at all)"
    )


# ---------------------------------------------------------------------------
# Pin 3 — flag-true zeroes out blocker / attacker creature damage
# ---------------------------------------------------------------------------


def test_flag_true_blocked_combat_zero_damage_to_creatures():
    """Flag set: a 2/2 attacker meets a 2/2 blocker. Neither
    creature is marked with damage; no SBA-relevant updates fire.
    On the flag-unset path this same exchange would mutually
    destroy both with 2 marked damage each.
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    assert state.declare_blocker(P1_FIRST_BEAR_ID, P0_FIRST_BEAR_ID) is True
    state.deal_combat_damage()
    # Neither creature took ANY damage.
    assert state.get_damage(P0_FIRST_BEAR_ID) == 0, (
        "attacker should NOT have any damage marker after a prevented "
        "combat damage step"
    )
    assert state.get_damage(P1_FIRST_BEAR_ID) == 0, (
        "blocker should NOT have any damage marker after a prevented "
        "combat damage step"
    )


# ---------------------------------------------------------------------------
# Pin 4 — flag-true blocks ALL combat damage from MULTIPLE attackers
# ---------------------------------------------------------------------------


def test_flag_true_zeroes_all_attackers_in_multi_creature_combat():
    """Flag set: TWO attackers — one undefended, one blocked. With
    a single guard at the routing site, both attackers' combat
    damage is zeroed. Pins that the guard is at the
    ``deal_combat_damage`` bottleneck, not per-source / per-entry
    (which would leave a gap if a later attacker iterator forgot
    to check the flag).
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    # Attacker 1: undefended. P0_FIRST_BEAR_ID = id 5.
    attacker_1 = P0_FIRST_BEAR_ID
    # Attacker 2: blocked. P0_FIRST_BEAR_ID + 1 = id 6 (next Bear).
    attacker_2 = P0_FIRST_BEAR_ID + 1
    assert state.declare_attacker(attacker_1) is True
    assert state.declare_attacker(attacker_2) is True
    assert state.declare_blocker(P1_FIRST_BEAR_ID, attacker_2) is True
    assert state.get_life(1) == 20
    state.deal_combat_damage()
    # Defender life untouched — attacker_1's unblocked hit was zeroed.
    assert state.get_life(1) == 20
    # Attacker_2 untouched — blocker's back-damage was zeroed.
    assert state.get_damage(attacker_2) == 0
    # Blocker untouched — attacker_2's forward-damage was zeroed.
    assert state.get_damage(P1_FIRST_BEAR_ID) == 0


# ---------------------------------------------------------------------------
# Pin 5 — flag-true + lifelink: no life gain
# ---------------------------------------------------------------------------


def test_flag_true_with_lifelink_attacker_does_not_gain_life():
    """Flag set: a 3/3 lifelink attacker against an undefended
    opponent. Lifelink fires off ``damage dealt`` (CR 702.15);
    with combat damage zeroed, the lifelink trigger doesn't fire,
    so the attacker's controller does NOT gain life. Mirrors
    ``tests_rust/test_combat.py::test_lifelink_attacker_unblocked_
    gains_life_for_controller`` but flipped to the flag-set path.
    """
    _require_engine()
    state = _fresh_state()
    attacker_id = state.register_and_inject_object(
        _keyword_creature_dict(3, 3, ["lifelink"]), 0, ZONE_BATTLEFIELD
    )
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.get_life(0) == 20
    assert state.get_life(1) == 20
    assert state.declare_attacker(attacker_id) is True
    state.deal_combat_damage()
    # Defender life untouched.
    assert state.get_life(1) == 20
    # No lifelink fired — attacker controller life untouched.
    assert state.get_life(0) == 20, (
        "lifelink must NOT fire when combat damage is fully prevented — "
        "zero damage dealt = no lifelink trigger"
    )


# ---------------------------------------------------------------------------
# Pin 6 — flag-true + "when this deals damage to a player" trigger: no fire
# ---------------------------------------------------------------------------


def test_flag_true_suppresses_deals_damage_to_player_trigger():
    """Flag set: an attacker with a ``deals_damage`` SourceIsSelf
    trigger gets sent in undefended. With combat damage prevented,
    ``dispatch_combat_damage_to_player`` is never reached, so the
    trigger does NOT land on the stack. Pin the
    "no-damage = no-trigger" invariant at the stack-size surface
    (the most direct observable for trigger firing).
    """
    _require_engine()
    state = _fresh_state()
    attacker_id = state.register_and_inject_object(
        _deals_damage_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.declare_attacker(attacker_id) is True
    # Pre: stack is empty (no triggers staged yet — declare_attacker
    # has no ATTACKS trigger on this printing).
    assert state.stack_size() == 0
    state.deal_combat_damage()
    # Post: stack still empty — no DealsDamage trigger fired
    # because no damage was dealt.
    assert state.stack_size() == 0, (
        "'deals damage to player' trigger must NOT fire when combat damage "
        "is fully prevented — zero damage = no DealsDamage event dispatch"
    )


# ---------------------------------------------------------------------------
# Pin 7 — flag-true + "when this is dealt damage" trigger: no fire
# ---------------------------------------------------------------------------


def test_flag_true_suppresses_dealt_damage_to_creature_trigger():
    """Flag set: a blocker with a ``dealt_damage`` SourceIsSelf
    trigger (RR Pin 3 shape) blocks a 2/2 attacker. With the
    flag set, ``apply_damage_pass`` is never reached, so the
    blocker's ``damage`` marker stays at 0 and no DealtDamage
    event is dispatched — the trigger does NOT fire.

    Parity baseline (flag-FALSE) — to verify the trigger DOES
    fire normally on the unguarded path — is implicit: the
    Rust trigger substrate's ``test_dealt_damage_*`` tests in
    ``tests_rust/test_triggers.py`` already pin that surface.
    Here we pin the *negative* direction (the guard's
    suppression).
    """
    _require_engine()
    state = _fresh_state()
    blocker_id = state.register_and_inject_object(
        _dealt_damage_self_draw_dict(), 1, ZONE_BATTLEFIELD
    )
    state.grant_prevent_all_combat_damage_this_turn()
    attacker_id = P0_FIRST_BEAR_ID  # 2/2
    assert state.declare_attacker(attacker_id) is True
    assert state.declare_blocker(blocker_id, attacker_id) is True
    assert state.stack_size() == 0
    state.deal_combat_damage()
    # Blocker took 0 damage; no DealtDamage dispatched; stack stays empty.
    assert state.get_damage(blocker_id) == 0
    assert state.stack_size() == 0, (
        "'when this is dealt damage' trigger must NOT fire when combat "
        "damage is fully prevented — zero damage marker = no DealtDamage "
        "dispatch"
    )


# ---------------------------------------------------------------------------
# Pin 8 — flag-true + trample: no spillover to defender
# ---------------------------------------------------------------------------


def test_flag_true_with_trample_no_spillover_to_defender():
    """Flag set: a 5/5 trample attacker is blocked by a 2/2.
    Normally trample would assign 2 to the blocker (lethal) and
    spill 3 to the defender. With the flag set, neither the
    blocker nor the defender takes any damage. Pins that the
    single-point guard correctly intercepts EVERY damage
    assignment, including trample's player-spillover entries
    (which are emitted in plan-building, AFTER the guard
    short-circuits the function).
    """
    _require_engine()
    state = _fresh_state()
    attacker_id = state.register_and_inject_object(
        _keyword_creature_dict(5, 5, ["trample"]), 0, ZONE_BATTLEFIELD
    )
    blocker_id = P1_FIRST_BEAR_ID  # 2/2 vanilla
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.get_life(1) == 20
    assert state.declare_attacker(attacker_id) is True
    assert state.declare_blocker(blocker_id, attacker_id) is True
    state.deal_combat_damage()
    # Blocker took 0; defender life untouched.
    assert state.get_damage(blocker_id) == 0
    assert state.get_life(1) == 20, (
        "trample spillover must be zeroed when combat damage is fully "
        "prevented — the guard runs BEFORE plan construction, so the "
        "trample-excess emission never gets to apply"
    )


# ---------------------------------------------------------------------------
# Pin 9 — flag-true does NOT affect non-combat damage
# ---------------------------------------------------------------------------


def test_flag_true_does_not_affect_non_combat_damage():
    """The flag guards ``deal_combat_damage`` only; non-combat
    damage (``deal_damage`` in ``state_ops_damage.rs``) is
    unaffected. Mirrors Python's "combat damage" specificity in
    ``state.prevent_combat_damage_this_turn`` — a Lightning Bolt
    resolving while the flag is set still deals 3 damage to its
    target, because the Bolt's resolve path goes through
    ``deal_damage``, not the combat damage step.
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    # Direct-damage primitive: target a creature with 1 damage.
    # Returns True on success (mirroring tap_object / set_counters).
    ok = state.deal_damage(P0_FIRST_BEAR_ID, 1)
    assert ok is True
    # Damage marker IS written — non-combat damage is NOT prevented.
    assert state.get_damage(P0_FIRST_BEAR_ID) == 1, (
        "prevent_all_combat_damage_this_turn must NOT affect non-combat "
        "damage (deal_damage primitive) — the flag is combat-specific"
    )


# ---------------------------------------------------------------------------
# Pin 10 — cleanup drain restores combat damage next turn
# ---------------------------------------------------------------------------


def test_cleanup_drain_restores_combat_damage_next_turn():
    """End-to-end Cycle 1 + Cycle 2 composition: grant the flag,
    run combat (no-op due to prevention), call
    ``cleanup_step_for_active_player`` (drains the flag), declare
    fresh attackers, run combat again — damage applies normally.

    Pins that the Cycle 1 drain composes cleanly with the Cycle 2
    guard: no cross-turn leakage, no stuck-prevention state.
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    # Turn N: prevented combat.
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    state.deal_combat_damage()
    assert state.get_life(1) == 20  # No damage.
    # Cleanup drains the flag (CR 514.3) AND clears combat (CR 514.x
    # plus end-of-combat scaffolding — we clear it explicitly below
    # to model the turn boundary so the next turn's deal_combat_damage
    # only sees freshly-declared attackers).
    summary = state.cleanup_step_for_active_player()
    assert summary["prevent_all_combat_damage_drained"] is True
    assert state.prevent_all_combat_damage_this_turn is False
    state.clear_combat()  # Drop turn N's attackers / blockers.
    # Turn N+1: declare a fresh untapped attacker (next Bear slot,
    # id 6 — id 5 is still tapped from turn N's declare).
    next_attacker = P0_FIRST_BEAR_ID + 1
    assert state.declare_attacker(next_attacker) is True
    state.deal_combat_damage()
    # Damage applies normally — flag drained, guard is no-op.
    assert state.get_life(1) == 18, (
        "after cleanup drains the prevention flag, combat damage must "
        "apply normally on the next turn — no cross-turn leakage of "
        "the flag-true short-circuit"
    )


# ---------------------------------------------------------------------------
# Pin 11 — flag-true leaves resolving_combat_damage_amount untouched
# ---------------------------------------------------------------------------


def test_flag_true_does_not_write_resolving_combat_damage_amount_slot():
    """Flag set: the I32 write site (``resolving_combat_damage_
    amount``) lives inside ``apply_damage_pass``, which the Cycle
    2 short-circuit skips. So a pre-set sentinel in the slot
    SURVIVES the prevented ``deal_combat_damage`` call (mirroring
    the existing ``test_resolving_combat_damage_amount_no_attackers_
    leaves_slot_untouched`` shape, but with the "no attackers"
    short-circuit replaced by the new "prevention flag" short-
    circuit).

    Pins that the routing-point guard is BEFORE the apply pass,
    not inside it — so no I32 slot write happens on the prevented
    path either.
    """
    _require_engine()
    state = _fresh_state()
    state.set_resolving_combat_damage_amount(7)
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    state.deal_combat_damage()
    # Slot was NOT written by the apply loop (it never ran).
    assert state.resolving_combat_damage_amount == 7, (
        "the I32 slot's write site is inside apply_damage_pass; when "
        "Cycle 2's prevention short-circuit fires, apply_damage_pass "
        "is never called and the slot is untouched"
    )


# ---------------------------------------------------------------------------
# Pin 12 — flag-true + no attackers: still a clean no-op
# ---------------------------------------------------------------------------


def test_flag_true_no_attackers_is_clean_noop():
    """A degenerate combat damage step with no attackers declared
    AND the flag set returns cleanly (no panic, no exception, no
    side effect). Both short-circuits (the new Cycle 2 prevention
    guard and the no-plan-entries empty-pass) are safe to compose.
    """
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.get_attackers() == []
    # No raise, no panic — clean return.
    state.deal_combat_damage()
    # No side effects.
    assert state.get_life(0) == 20
    assert state.get_life(1) == 20
    assert state.prevent_all_combat_damage_this_turn is True


# ---------------------------------------------------------------------------
# Pin 13 — CoW invariant on the flag-true path
# ---------------------------------------------------------------------------


def test_flag_true_on_clone_does_not_affect_source_combat():
    """Granting the prevention flag on a CLONE and running combat
    on the clone (no-op due to prevention) does NOT change the
    source state's combat damage behavior. Pins that the flag's
    routing guard reads from the clone's own
    ``prevent_all_combat_damage_this_turn`` slot — not via shared
    reference back to the source.

    Same CoW canary shape as Cycle 1's
    ``test_grant_on_clone_does_not_affect_source``, but exercising
    the Cycle 2 routing read site through ``deal_combat_damage``.
    """
    _require_engine()
    state = _fresh_state()
    # Declare an attacker on the SOURCE.
    assert state.declare_attacker(P0_FIRST_BEAR_ID) is True
    # Clone — the clone carries the attacker list forward (and the
    # default flag-False).
    cloned = state.clone_once()
    assert cloned.prevent_all_combat_damage_this_turn is False
    # Grant on the CLONE only.
    cloned.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is False
    assert cloned.prevent_all_combat_damage_this_turn is True
    # Run combat on the CLONE — prevented.
    cloned.deal_combat_damage()
    assert cloned.get_life(1) == 20  # Prevented.
    # Run combat on the SOURCE — damage applies normally.
    state.deal_combat_damage()
    assert state.get_life(1) == 18, (
        "the source's combat damage must NOT be affected by the clone's "
        "flag grant — CoW isolation invariant"
    )
