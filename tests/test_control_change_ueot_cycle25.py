"""Control-change-UEOT subsystem — Cycle 2.5 (resolve-side reroutes).

**Cycle 2.5 closes Cycle 2's deferred resolve-side scan reroutes.**
Cycle 2 explicitly kept ~30+ source-controller multi-zone scans in
``rust/src/state_ops_resolve.rs`` raw on the grounds that they were
substrate-equivalent (no live mint site yet). Cycles 4 + 5 minted
live ``controller_override`` values on bf via Mutinous Massacre and
Systems Override — which means resolve-side reads of ``obj.controller``
on bf permanents MAY now observe stale values for objects with active
overrides.

This lane walks the resolve-side scans and reroutes only the explicit
**gameplay-decision** sites through ``effective_controller_id``:

  - **Aura yours_only resolve-time legality** (the
    ``resolve_aura_post_resolve`` enchant-target check)
  - **find_bf_controller_and_printing** (drives the per-player
    "entered under your control this turn" ETB counter)
  - **BiosynthicBurstEffect controller-restricted target filter**
    (the +1/+1 counter + 3-keyword grant target picker)
  - **InvasiveManeuversEffect-shape slot-0 friendly + slot-1 opponent**
    CR 608.2b re-legality checks
  - **TaintedHonor-shape "became own creature" fizzle** (CR 608.2b)
  - **TapTargetOpponentPermanentAndStunIt opponent-only filter**
  - **CompletePlanetaryAssault-shape "controlled by source" filter**
    (the source-controller-controls-Spacecraft-or-Planet predicate)

Kept-raw under documented classification:

  - **Source-controller for the resolving spell** (~28 sites that
    scan the 5 per-player zones to locate the spell's source's
    controller — caster identity, not affected by UEOT control
    change; the spell's *source* on the stack/in-hand is never on a
    bf with an override stamp).
  - **StackItem.controller** (~26 fallback paths reading the
    authoritative stack-item field — that's caster identity).
  - **Constructor-writes** (~12 sites that *write* ``obj.controller``
    when constructing a new GameObject / re-stamping on zone
    change — the override is None on fresh objects per the
    Cycle-1 default).
  - **Zone-residency lookups via ``obj.controller as usize``** (~30
    sites that use ``obj.controller`` as the bf-Vec key — works
    correctly post-override because ``grant_control_change_ueot``
    never mutates ``obj.controller``; only the parallel
    ``controller_override`` slot is touched. The bf-storage key
    remains the original owner-pidx).
  - **Trigger-observer reads** (~3 sites in ``dispatch_event``-
    style scans collecting bf + non-bf observers; deferred — the
    trigger-firing path is a separate lane and the override-on-
    leaving-bf semantics need a dedicated test surface).

Spec: lane brief (2026-05-22). Lane base = Cycle 5
(``feature/control-change-ueot-cycle-5-p1``). Substrate-only — every
reroute is byte-equivalent when no override is set.

**Test surface**: drive a live resolve-arm reroute (BiosynthicBurst's
controller-restricted target filter) with a UEOT control stamp
minted via ``grant_control_change_ueot``. Verify:

  1. The pre-Cycle-2.5 baseline behavior is preserved (no override
     active → the rerouted filter picks targets identically to the
     raw read).
  2. The override flip is observed by the rerouted filter (an
     opp-owned creature with ``controller_override = caster``
     becomes a legal BiosynthicBurst target, where pre-Cycle-2.5
     it would have been silently skipped).
  3. The cleanup-step drain reverts: post-cleanup, the override is
     gone and the rerouted filter again sees the raw controller.

Mirrors the unit-test pattern in
``tests/test_control_change_ueot_cycle1.py`` (Cycle-1 substrate) +
``tests/test_mutinous_massacre_parity.py`` (Cycle-4 live mint).
"""

from __future__ import annotations

import importlib
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


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _new_creature(rs, name, owner, power=2, toughness=2):
    return rs.register_and_inject_object(
        {
            "name": name,
            "card_types": ["Creature"],
            "power": power,
            "toughness": toughness,
        },
        owner,
        ZONE_BATTLEFIELD,
    )


def _bb_carrier(rs, owner):
    """Inject a Biosynthic Burst ETB carrier on the given player's bf.
    Returns the carrier's id. The carrier's enters-the-battlefield
    triggered ability pushes a BiosynthicBurstEffect onto the stack
    on the next dispatch, which is the cleanest synthetic-test driver
    for the controller-restricted target filter."""
    return rs.register_and_inject_object(
        {
            "name": "BB Carrier",
            "card_types": ["Creature"],
            "mana_cost": {"G": 1, "generic": 1},
            "power": 1,
            "toughness": 1,
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {"kind": "biosynthic_burst_effect"},
                }
            ],
        },
        owner,
        ZONE_BATTLEFIELD,
    )


# ---------------------------------------------------------------------------
# Baseline pin — pre-override, the rerouted filter is byte-equal to raw.
# ---------------------------------------------------------------------------


def test_baseline_no_override_rerouted_filter_picks_own_creature():
    """**Substrate-only invariant pin.** With no UEOT control stamp
    minted, the rerouted BiosynthicBurst controller-restricted target
    filter MUST be byte-equal to the pre-Cycle-2.5 raw read. p0 has
    own creature + opp has its own creature; BB picks p0's own."""
    rs = _fresh_state()
    own_bear = _new_creature(rs, "Own Bear", owner=0)
    opp_bear = _new_creature(rs, "Opp Bear", owner=1)
    assert rs.effective_controller_id(own_bear) == 0
    assert rs.effective_controller_id(opp_bear) == 1
    _bb_carrier(rs, owner=0)
    # The ETB carrier's etb trigger fires + pushes the BB effect on
    # the stack. Resolve it.
    assert rs.stack_size() == 1
    rs.resolve_top_of_stack()
    # Own bear got the counter + keywords; opp bear untouched.
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[own_bear]["counters"] == 1, (
        "own creature must get the +1/+1 counter (controller-restricted "
        "filter picks own); raw and rerouted reads agree pre-override"
    )
    assert bf[opp_bear]["counters"] == 0, (
        "opp creature must NOT get the counter — opp is not "
        "controller-equal-to-caster pre-override"
    )
    for kw in ("reach", "trample", "indestructible"):
        assert rs.has_keyword(own_bear, kw) is True
        assert rs.has_keyword(opp_bear, kw) is False


# ---------------------------------------------------------------------------
# Reroute observes override — opp creature with override flips into the
# controller-restricted filter as a legal pick.
# ---------------------------------------------------------------------------


def test_override_flips_opp_creature_into_controller_restricted_filter():
    """**The Cycle 2.5 reroute pin.** Mint a UEOT override on an
    opponent creature flipping its effective controller to p0
    (caster). The rerouted BiosynthicBurst filter now sees the
    overridden creature as a legal pick (``effective_controller_id ==
    caster``). Pre-Cycle-2.5 the filter would have read the raw
    ``obj.controller == 1`` and silently rejected the creature —
    which is the gap Cycle 2.5 closes."""
    rs = _fresh_state()
    # p1's creature, which we'll override to p0 (caster).
    opp_bear = _new_creature(rs, "Opp Bear (override-flipped)", owner=1)
    # Verify raw + effective pre-grant.
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[opp_bear]["controller_id"] == 1
    assert bf[opp_bear]["controller_override"] is None
    assert rs.effective_controller_id(opp_bear) == 1
    # Mint the override (UEOT control stamp).
    assert rs.grant_control_change_ueot(opp_bear, 0) is True
    # Effective controller now == p0; raw controller still p1.
    assert rs.effective_controller_id(opp_bear) == 0
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[opp_bear]["controller_override"] == 0
    assert bf[opp_bear]["controller_id"] == 1, (
        "raw controller field MUST be untouched — grant only writes "
        "controller_override; Cycle-1 invariant"
    )
    # Cast BiosynthicBurst by p0 (the ETB carrier on p0's bf, which
    # pushes the BB Effect onto the stack with source_controller=0).
    _bb_carrier(rs, owner=0)
    assert rs.stack_size() == 1
    rs.resolve_top_of_stack()
    # POST-REROUTE: the BB filter (line ~8425, rerouted through
    # effective_controller_id) sees the opp_bear as a legal pick
    # because its effective controller is 0 (the caster). The
    # +1/+1 counter + 3-keyword grant lands on it.
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[opp_bear]["counters"] == 1, (
        "**THE REROUTE PIN.** The overridden opp creature MUST receive "
        "the +1/+1 counter — its effective controller is now 0 "
        "(caster), so the controller-restricted target filter must "
        "pick it. Pre-Cycle-2.5 this assertion would FAIL (raw "
        "controller=1 != caster=0, filter skip)."
    )
    for kw in ("reach", "trample", "indestructible"):
        assert rs.has_keyword(opp_bear, kw) is True, (
            f"overridden opp creature MUST gain {kw} UEOT — "
            "controller-restricted filter sees it as own"
        )


# ---------------------------------------------------------------------------
# Cleanup drain reverts — post-cleanup, the rerouted filter again sees
# the raw controller (no longer a legal pick).
# ---------------------------------------------------------------------------


def test_cleanup_drain_reverts_then_rerouted_filter_skips_again():
    """**The cleanup drain composes uniformly.** Mint the override,
    run the cleanup step. Post-cleanup ``controller_override`` is
    None on every bf permanent (the Cycle-1
    ``clear_controller_overrides_all`` pass). A fresh resolve of the
    BB filter then again skips the (now-not-overridden) opp creature
    — the rerouted helper falls back to the raw controller exactly
    like pre-Cycle-2.5."""
    rs = _fresh_state()
    opp_bear = _new_creature(rs, "Opp Bear", owner=1)
    assert rs.grant_control_change_ueot(opp_bear, 0) is True
    assert rs.effective_controller_id(opp_bear) == 0
    # Run the cleanup step (CR 514.3 — the UEOT drain).
    summary = rs.cleanup_step_for_active_player()
    # The Cycle-1 drain count is in the summary dict.
    assert summary.get("controller_overrides_cleared", 0) == 1, (
        f"cleanup must drain the override; summary={summary}"
    )
    # Effective controller reverted to raw baseline (1).
    assert rs.effective_controller_id(opp_bear) == 1
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[opp_bear]["controller_override"] is None
    # Now cast BB by p0. The opp_bear is again seen as NOT
    # controller-equal-to-caster (post-drain raw == effective == 1).
    _bb_carrier(rs, owner=0)
    assert rs.stack_size() == 1
    rs.resolve_top_of_stack()
    bf = {e["id"]: e for e in rs.battlefield_snapshot()}
    assert bf[opp_bear]["counters"] == 0, (
        "post-drain, the opp creature MUST be skipped by the "
        "controller-restricted filter — same as pre-Cycle-2.5 raw "
        "behavior"
    )
    for kw in ("reach", "trample", "indestructible"):
        assert rs.has_keyword(opp_bear, kw) is False, (
            "post-drain, no keywords grant — controller-restricted "
            "skip"
        )


# ---------------------------------------------------------------------------
# Helper-equivalence pin — effective_controller_id == raw controller when
# no override is set on EVERY bf permanent.
# ---------------------------------------------------------------------------


def test_effective_equals_raw_when_no_override():
    """**The substrate byte-equality pin.** On a state with no
    overrides minted, every bf object's ``effective_controller_id``
    matches its raw ``controller`` field. This is the substrate-only
    invariant that every Cycle 2.5 reroute relies on for back-compat:
    when the helper degenerates to the raw read, the rerouted code
    path is byte-equivalent to the pre-Cycle-2.5 raw read."""
    rs = _fresh_state()
    for owner in (0, 1):
        for name in ("A", "B", "C"):
            _new_creature(rs, f"{name}-{owner}", owner=owner)
    for entry in rs.battlefield_snapshot():
        obj_id = entry["id"]
        assert entry["controller_override"] is None
        assert (
            rs.effective_controller_id(obj_id)
            == entry["controller_id"]
        ), (
            f"effective_controller_id MUST == raw controller when no "
            f"override is set; obj_id={obj_id}, raw={entry['controller_id']}, "
            f"effective={rs.effective_controller_id(obj_id)}"
        )
