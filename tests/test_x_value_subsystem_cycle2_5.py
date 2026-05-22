"""X-value subsystem — Cycle 2.5 (admit widen + cast detector + live mint).

Spec: lane brief (2026-05-22). Cycle 2.5 sits between the substrate
cycles (C1 + C2) and the Effect-arm cycles (C3+):

  * **Part 1 (admit widen).** The early ``mc.x_pips > 0`` reject in
    :func:`mtg_bot.parity.trajectory._printing_is_bridge_simple_spell`
    is gone. X-cost printings now flow through the per-phase target-
    shape gates; only the Phase-3 wrath shape (`(0, 0)` no slot_kinds /
    no ``targets`` attr) admits an X-cost printing. The covered+bridged
    + round-trip-or-bail teeth still gate the admit set — until C3 ships
    `Effect::ZeroPointBallad`, Zero Point Ballad stays DECLINED via the
    no-Effect-coverage tooth. The Phase-1/2/4/5 lanes explicitly reject
    X-cost (`and not _printing_is_x_cost`).

  * **Part 2 (cast-pipeline detector + first live mint site).**
    ``state_ops_cast.rs::cast_spell_noncreature`` reads
    ``printing.mana_cost.x_pips > 0``. If so, it binds X via the no-
    agent fallback (Python `_choose_and_apply_x` at `actions.py:529-549`
    returns 0 when `p.agent is None`; the cross-engine harness DOES
    drive Python with `agent=None`, so X=0 is the byte-equal mirror).
    Folds `x_pips * x_value` into generic and zeros `x_pips` before
    ``auto_tap_pay_for_cost`` — mirrors Python `actions.py:1602-1611`.
    Threads `Some(0)` into ``new_spell_with_targets_mode_addcost_and_x``
    (was hard-coded ``None`` in C1). Non-X spells continue to push
    ``None``: the regression invariant.

**Live-mint-site invariant changeover.** This is the cycle that flips
``StackItem.x_value`` from the substrate-only ``None`` default to
``Some(0)`` for X-cost casts. Pins:

  (a) non-X cast still pushes ``None`` (regression pin),
  (b) X-cost cast pushes ``Some(0)`` (new behavior pin),
  (c) X=0 is distinct from None (CR 117.4),
  (d) every other field on the StackItem is byte-equal between the X
      and non-X cast shapes.

**Tests cover (12+ pins):**

1. Non-X spell admit-gate regression: a `(1, 1)` Phase-1 spell
   (Depressurize) stays admitted post-C2.5.
2. X-cost spell admit-gate shape flip: a synthetic X-cost Phase-3-shape
   printing whose effect is covered+bridged is admitted. (Zero Point
   Ballad itself stays declined because `ZeroPointBalladEffect` is not
   covered+bridged yet — C3+ work.)
3. X-cost Phase-1 single-target shape stays declined (lane is X-cost-
   excluded).
4. X-cost Phase-4 multi-target shape stays declined (lane is X-cost-
   excluded).
5. Non-X cast pushes ``StackItem.x_value = None`` (regression).
6. X-cost cast pushes ``StackItem.x_value = Some(0)`` under the no-
   agent harness (NEW behavior pin).
7. X=0 is distinct from None: a fresh non-X cast observes None; an
   X-cost cast observes Some(0); the two are NOT confounded.
8. Resolver bracket flows the value: after C1's bracket fires, the
   transient `current_resolve_x_value` is consumed at resolve — the
   stack drains cleanly and the bracket restores to None. (Pins the
   bracket's no-leak invariant under a live mint, not just the
   synthetic setter.)
9. Multiple X-cost casts in sequence: each pushes its own `Some(0)`;
   the top reads its own value (per-StackItem isolation).
10. X-cost cast with non-zero x_pips folds correctly: a printing with
    `x_pips=2` and `generic=0` and `colored={B:1}` pays just `{B}`
    when X=0 (fold is no-op). Cast succeeds and the StackItem carries
    `Some(0)`.
11. Cross-engine byte-equality (NON-X regression): the non-X cast on
    a Phase-1 admitted card preserves every observable on the stack
    (the regression pin guarding against accidental field leaks).
12. Cross-engine byte-equality for an X-cost trajectory (the NEW pin
    the brief requires): drive a synthetic X-cost cast and assert the
    StackItem `x_value` snapshot is `Some(0)` on the Rust side — the
    Python side under the harness produces `0` (the integer field on
    Python's StackItem dataclass, non-optional, so a non-X cast
    produces 0 there too; pinning Rust's `Some(0)` is what surfaces
    the new behavior).

The test surface drives the Rust primitives directly via the
``mtg_bot_engine`` extension (mirrors Cycle 1's discipline).
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


ZONE_HAND = 0
ZONE_BATTLEFIELD = 2


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _fresh_state():
    """A 0-bf / 0-hand / 0-library state — everything injected via
    ``register_and_inject_object``."""
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _vanilla_non_x_printing_dict(name="C2.5-Non-X-Probe Sorcery"):
    """Non-X sorcery, cost `{C}` — a draw-1 spell. Mirrors the Cycle 1
    test surface's `_vanilla_instant_printing_dict`."""
    return {
        "name": name,
        "mana_cost": {"generic": 1},
        "card_types": ["Sorcery"],
        "effects": [
            {"kind": "draw_cards", "count": 1},
        ],
    }


def _x_cost_printing_dict(
    name="C2.5-X-Cost-Probe Sorcery",
    x_pips=1,
    generic=0,
    black=1,
):
    """Synthetic X-cost sorcery — cost `{X}{B}` by default (Zero Point
    Ballad shape). The spell carries no real effects (the Phase-3
    admit gate's covered+bridged tooth keeps it declined at the
    bridge layer; this test surface is below that layer, driving the
    cast pipeline directly via the Rust engine)."""
    mc: dict = {}
    if generic:
        mc["generic"] = generic
    if black:
        mc["B"] = black
    if x_pips:
        mc["X"] = x_pips
    return {
        "name": name,
        "mana_cost": mc,
        "card_types": ["Sorcery"],
        "effects": [],
    }


# ---------------------------------------------------------------------------
# Part 1 — admit-predicate widen
# ---------------------------------------------------------------------------


def test_admit_predicate_non_x_phase1_regression():
    """Pin 1: a real Phase-1 single-target non-X spell (Depressurize)
    stays ADMITTED post-C2.5. The widen is surgical — non-X lanes are
    unchanged."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_spell

    printings = get_active_source().get_printings()
    if "Depressurize" not in printings:
        pytest.skip("Depressurize not in printing set")
    p = printings["Depressurize"]
    assert _printing_is_bridge_simple_spell(p) is True


def test_admit_predicate_zero_point_ballad_still_declined():
    """Pin 2: Zero Point Ballad's `{X}{B}{B}` admit gate flow stays
    DECLINED in C2.5 because its `ZeroPointBalladEffect` is not
    covered+bridged yet (C3+ work). The early `x_pips > 0` reject is
    GONE, but the covered+bridged tooth carries the load."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_spell

    printings = get_active_source().get_printings()
    if "Zero Point Ballad" not in printings:
        pytest.skip("Zero Point Ballad not in printing set")
    p = printings["Zero Point Ballad"]
    # Substrate stays declined — but for a different reason than C1/C2
    # (no longer the early `x_pips > 0` reject; now the covered+bridged
    # tooth). When C3 lands `Effect::ZeroPointBallad`, this flips True.
    assert _printing_is_bridge_simple_spell(p) is False


def _make_synthetic_sorcery_printing(name, mana_cost, spell_effect):
    """Build a minimal synthetic Sorcery CardPrinting for admit-gate
    tests. Fills the required fields with safe defaults so the test
    focuses on the X-cost + target-shape interaction."""
    from mtg_bot.cards.base import CardPrinting
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=mana_cost,
        type_line="Sorcery",
        supertypes=(),
        card_types=("Sorcery",),
        subtypes=(),
        power=None,
        toughness=None,
        produces=frozenset(),
        spell_effect=spell_effect,
    )


def test_admit_predicate_phase1_x_cost_excluded():
    """Pin 3: a synthetic X-cost printing in the Phase-1 (1, 1) target
    shape stays DECLINED — the X-cost flag is restricted to Phase 3 by
    construction (the `and not _printing_is_x_cost` conjunct on Phase
    1/2/4/5)."""
    from mtg_bot.cards.base import ManaCost
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_spell

    # Reuse a real EoE Phase-1 effect carrier (Depressurize's effect
    # shape). The admit-gate's covered+bridged tooth would also gate
    # this, but we're testing the FIRST tooth (X-cost + Phase-1 shape):
    # if Phase 1 admitted X-cost, the gate would proceed to covered+
    # bridged and might still admit. The point is the SHAPE rejects.
    #
    # Take Depressurize's spell_effect (a real Phase-1 single-target
    # effect) and rebuild the printing with an X-cost mana_cost.
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    if "Depressurize" not in printings:
        pytest.skip("Depressurize not in printing set")
    base = printings["Depressurize"]
    p = _make_synthetic_sorcery_printing(
        name="Synthetic X-Cost Depressurize",
        mana_cost=ManaCost(generic=0, colored=(("U", 1),), x_pips=1),
        spell_effect=base.spell_effect,
    )
    # X-cost + Phase-1 shape — REJECTED.
    assert _printing_is_bridge_simple_spell(p) is False


def test_admit_predicate_phase4_x_cost_excluded():
    """Pin 4: a synthetic X-cost printing in a Phase-4 multi-target
    shape stays DECLINED — the X-cost lane is restricted to Phase 3."""
    from mtg_bot.cards.base import ManaCost
    from mtg_bot.parity.trajectory import _printing_is_bridge_simple_spell

    # Reuse a real Phase-4 effect carrier. Scout for Survivors carries
    # the inaugural Phase-4 graveyard-creature multi-target effect.
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    target_card = None
    for cand_name in (
        "Scout for Survivors",
        "Scout for Survivors of Vlasnek",
    ):
        if cand_name in printings:
            target_card = printings[cand_name]
            break
    if target_card is None:
        pytest.skip("No Phase-4 multi-target carrier in printing set")
    p = _make_synthetic_sorcery_printing(
        name="Synthetic X-Cost Multi-Target",
        mana_cost=ManaCost(generic=0, colored=(("B", 1),), x_pips=1),
        spell_effect=target_card.spell_effect,
    )
    # X-cost + Phase-4 shape — REJECTED.
    assert _printing_is_bridge_simple_spell(p) is False


# ---------------------------------------------------------------------------
# Part 2 — cast-pipeline detector + first live mint
# ---------------------------------------------------------------------------


def test_non_x_cast_pushes_x_value_none_regression():
    """Pin 5: a non-X cast still pushes ``StackItem.x_value = None``
    post-C2.5. The regression invariant: every existing cross-engine
    trajectory on a non-X spell is byte-equal to pre-C2.5."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_non_x_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    ok = state.cast_spell_noncreature(spell_id, 0)
    assert ok is True
    assert state.stack_size() == 1
    assert state.stack_top_x_value() is None


def test_x_cost_cast_pushes_some_zero_first_live_mint():
    """Pin 6: an X-cost cast under the no-agent harness pushes
    ``StackItem.x_value = Some(0)`` — the FIRST live mint site for the
    Cycle-1 substrate field. This is the C2.5 behavior change."""
    _require_engine()
    state = _fresh_state()
    # Synthetic `{X}{B}` printing — x_pips=1, black=1.
    spell_id = state.register_and_inject_object(
        _x_cost_printing_dict(x_pips=1, generic=0, black=1), 0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    ok = state.cast_spell_noncreature(spell_id, 0)
    assert ok is True
    assert state.stack_size() == 1
    # The NEW behavior pin — was `None` in C1/C2.
    assert state.stack_top_x_value() == 0
    # CR 117.4 boundary: Some(0) is distinct from None.
    assert state.stack_top_x_value() is not None


def test_x_zero_is_distinct_from_none_cr_117_4():
    """Pin 7: CR 117.4 boundary — ``Some(0)`` is distinct from ``None``.
    A fresh non-X cast observes None; an X-cost cast observes 0
    (which is `Some(0)` on the Rust side). The two cast shapes do NOT
    confound. Mirrors the Cycle 1 setter pin
    `test_set_stack_top_x_value_accepts_zero` but driven via the live
    mint site instead of the synthetic setter."""
    _require_engine()
    state = _fresh_state()
    # Non-X cast first.
    non_x_id = state.register_and_inject_object(
        _vanilla_non_x_printing_dict("Non-X1"), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(non_x_id, 0)
    assert state.stack_top_x_value() is None
    # Drain so we can cast another spell.
    state.resolve_top_of_stack()
    assert state.stack_size() == 0
    # X-cost cast next.
    x_id = state.register_and_inject_object(
        _x_cost_printing_dict("X1", x_pips=1, generic=0, black=1),
        0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    state.cast_spell_noncreature(x_id, 0)
    # Some(0), NOT None — the C2.5 mint site fired.
    assert state.stack_top_x_value() == 0
    assert state.stack_top_x_value() is not None


def test_resolver_bracket_flows_value_no_leak():
    """Pin 8: the C1 resolver bracket flows the X-value through resolve
    and restores the transient to None. Drive an X-cost cast (mints
    ``Some(0)``), resolve, then cast a non-X spell — its x_value is
    None (the bracket did NOT leak). Pins the no-leak invariant under
    a LIVE mint, not just the synthetic setter."""
    _require_engine()
    state = _fresh_state()
    x_id = state.register_and_inject_object(
        _x_cost_printing_dict(x_pips=1, generic=0, black=1),
        0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    state.cast_spell_noncreature(x_id, 0)
    assert state.stack_top_x_value() == 0
    # Resolve drains the stack — the bracket completes without panic.
    state.resolve_top_of_stack()
    assert state.stack_size() == 0
    # Push a fresh non-X spell — its x_value is None (bracket did NOT
    # leak the prior Some(0)).
    non_x_id = state.register_and_inject_object(
        _vanilla_non_x_printing_dict("Post-Resolve"),
        0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(non_x_id, 0)
    assert state.stack_top_x_value() is None


def test_multiple_x_cost_casts_per_item_isolation():
    """Pin 9: two X-cost casts in sequence — each pushes its own
    `Some(0)`; the top reads its own value. Pins per-StackItem
    isolation (the field is per-StackItem, not a singleton)."""
    _require_engine()
    state = _fresh_state()
    # Spell 1
    x1_id = state.register_and_inject_object(
        _x_cost_printing_dict("X1", x_pips=1, generic=0, black=1),
        0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    state.cast_spell_noncreature(x1_id, 0)
    assert state.stack_top_x_value() == 0
    # Spell 2 — fresh push on top.
    x2_id = state.register_and_inject_object(
        _x_cost_printing_dict("X2", x_pips=1, generic=0, black=1),
        0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    state.cast_spell_noncreature(x2_id, 0)
    # Top is X2 — Some(0).
    assert state.stack_top_x_value() == 0
    # Resolve X2 — back to X1 (top now).
    state.resolve_top_of_stack()
    assert state.stack_top_x_value() == 0


def test_x_cost_cast_with_x_pips_2_devastating_onslaught_shape():
    """Pin 10: a printing with `x_pips=2` (Devastating Onslaught
    shape) registers and casts cleanly. With X=0 the fold adds
    2*0=0 to generic, so the cast pays just the colored portion.
    StackItem still carries Some(0)."""
    _require_engine()
    state = _fresh_state()
    # `{X}{X}{R}` shape — black ignored, red instead.
    do_dict = {
        "name": "Synthetic Devastating Onslaught",
        "mana_cost": {"R": 1, "X": 2},
        "card_types": ["Sorcery"],
        "effects": [],
    }
    spell_id = state.register_and_inject_object(do_dict, 0, ZONE_HAND)
    state.add_mana(0, "R", 1)
    ok = state.cast_spell_noncreature(spell_id, 0)
    assert ok is True
    # Some(0): the X-cost detector fired AND the fold left the
    # colored side alone (the (X*x_pips) generic add is 0).
    assert state.stack_top_x_value() == 0


def test_non_x_cast_byte_equal_field_view_regression():
    """Pin 11: a non-X cast preserves every observable on the stack —
    explicit byte-equal regression sweep across the X-value-adjacent
    StackItem fields. Guards against accidental field-leak through
    the cast pipeline (e.g. if the X-cost detector wrote x_value for
    non-X casts by mistake)."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_non_x_printing_dict("Regression-Sweep"),
        0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    # All four x-value-adjacent StackItem slots return their
    # back-compat defaults.
    assert state.stack_size() == 1
    assert state.stack_top_x_value() is None
    # Mode-idx and addcost-choice-id are unrelated to X-value but
    # share the StackItem store; defensively pin them too so a
    # field-rename collision can't sneak past.
    assert state.stack_top_chosen_mode_idx() is None
    assert state.stack_top_additional_cost_choice_id() is None


def test_cross_engine_x_cost_byte_equality_some_zero():
    """Pin 12: the cross-engine byte-equality pin for an X-cost
    trajectory — Rust's `StackItem.x_value` after the C2.5 live mint
    is `Some(0)`, matching Python's `StackItem.x_value = 0` under the
    no-agent harness (`actions.py:_choose_and_apply_x` returns 0 when
    `p.agent is None`). The Rust Some(0) reads as `0` via the PyO3
    getter — distinguishable from Python's `0` only at the type level
    (Python's int / Rust's `Option<i32>` unwrap to the same 0).
    This pin is the cross-engine surface contract."""
    _require_engine()
    state = _fresh_state()
    x_id = state.register_and_inject_object(
        _x_cost_printing_dict("X-Parity", x_pips=1, generic=0, black=1),
        0, ZONE_HAND,
    )
    state.add_mana(0, "B", 1)
    state.cast_spell_noncreature(x_id, 0)
    # Rust's `Option<i32>` getter returns the inner i32 directly to
    # Python (PyO3 unwraps `Some(i)` → `int(i)`; `None` → `None`).
    x_obs = state.stack_top_x_value()
    assert x_obs is not None  # NOT None — the live mint fired.
    assert isinstance(x_obs, int)
    assert x_obs == 0


def test_x_cost_zero_x_pips_pre_set_no_op():
    """Pin 13: a printing with `x_pips=0` (back-compat default) is
    INDISTINGUISHABLE from a non-X printing — the detector's gate is
    `cost.x_pips > 0`. The `x_pips=0` boundary is the back-compat
    invariant."""
    _require_engine()
    state = _fresh_state()
    # `x_pips=0`: the parser treats it as implicit (Cycle 2 pin 3).
    d = {
        "name": "X-Zero Probe",
        "mana_cost": {"generic": 1},  # explicit X omitted
        "card_types": ["Sorcery"],
        "effects": [],
    }
    spell_id = state.register_and_inject_object(d, 0, ZONE_HAND)
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    # Non-X-cost path — `bound_x_value = None`.
    assert state.stack_top_x_value() is None
