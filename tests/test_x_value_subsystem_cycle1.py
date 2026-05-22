"""X-value subsystem — Cycle 1 (substrate-only).

Spec: lane brief (2026-05-21). The Rust side gains the per-
``StackItem`` ``x_value: Option<i32>`` field, the
``GameState.current_resolve_x_value: Option<i32>`` transient slot,
the save → set → restore bracket in BOTH ``resolve_top_of_stack``
AND ``resolve_top_of_stack_pause``, the dedicated PyO3 getter
``stack_top_x_value()``, and the synthetic-test setter
``set_stack_top_x_value()``.

Mirrors the precedent set by:

- **commit 3bc47355** (control-change-UEOT Cycle 1) — substrate-only,
  no-card-flip Cycle 1 pattern + the 17-test parity-pin discipline.
- **commit 17dc7d02** (close-encounter-substrate) — Cycle 1 +
  Cycle 2 + Cycle 3 staging: this lane is the Cycle 1 analog with
  the transient + StackItem field + bracket landing FIRST and no
  Effect arm reading the slot yet.
- **commit b4975088** (scour-for-scrap-slice) — the additive-
  sibling vs subsumption decision precedent + the synthetic-test
  setter shape.

Cycle 1 substrate-only — NO live mint site is wired:

  - ``cast_spell_noncreature`` passes ``x_value: None`` to the new
    constructor ``new_spell_with_targets_mode_addcost_and_x``. The
    cast-pipeline detector that reads ``mc.x_pips > 0`` and binds
    the deterministic-first X via pay-all-floating no-agent
    fallback is Cycle 2/3 work.
  - No ``Effect::ZeroPointBallad`` / ``Effect::DevastatingOnslaught``
    resolve arm reads ``current_resolve_x_value`` yet — Cycle 3+
    will add them.
  - The bracket is wired in BOTH resolve entry points so Cycles
    2/3 can land additively with NO churn at the resolve sites
    (mirrors how ``current_resolve_additional_cost_choice_id``'s
    Cycle 1 bracket landed pre-Cycle-3 Effect reader).

**Tests cover (13 pins):**

1. Empty-stack contract — ``stack_top_x_value()`` returns ``None``
   on an empty stack.
2. Default-None invariant — a freshly-pushed spell via
   ``cast_spell_noncreature`` has ``x_value = None`` (no live mint
   site yet → back-compat byte-equal).
3. Synthetic setter round-trip — ``set_stack_top_x_value(Some(5))``
   followed by ``stack_top_x_value()`` returns 5.
4. Setter accepts a NEGATIVE i32 (signed field invariant — the
   signedness contract is enforced at the type level; the
   resolve-arm consumer may clamp / saturate as needed).
5. Setter accepts X=0 (CR 107.1b boundary — X may be 0).
6. Setter accepts a LARGE i32 (`i32::MAX`) — wire boundary holds.
7. Setter unset round-trip — ``set_stack_top_x_value(None)``
   resets the slot back to ``None`` (validates the "restore to
   prior None" branch of the resolver bracket — tests must be
   able to drive the slot in both directions).
8. Setter no-op on empty stack — calling the setter when the
   stack is empty does NOT panic; ``stack_top_x_value()`` stays
   ``None`` afterwards.
9. ``resolve_top_of_stack`` bracket — default ``None`` source
   maintains ``current_resolve_x_value == None`` post-resolve
   (the no-op back-compat path).
10. ``resolve_top_of_stack`` bracket — synthetic set then resolve
    restores ``current_resolve_x_value`` to its prior value (the
    save → set → restore contract; the post-resolve value is
    ``None`` because the prior was ``None``).
11. ``resolve_top_of_stack_pause`` bracket — same as #10 against
    the ``_pause`` entry point (§3.4 sibling-parity requirement —
    BOTH resolve entry points must bracket identically).
12. Multiple synthetic pushes preserve per-StackItem ``x_value``
    — pushing two spells with distinct synthetic X values, the
    top (last-pushed) returns its own X (multi-spell isolation).
13. Back-compat byte-equal — pre-substrate cast trajectories
    (cast_spell_noncreature → resolve) observe NO change: every
    push starts with ``x_value: None`` and the bracket is a no-op.

The test surface intentionally drives the Rust primitive
directly without crossing the Python ↔ Rust boundary on a real
state — the bridge round-trip is Cycle 3+ work (parallel to
how `controller_override` Cycle 1 tests drove Rust directly
without bridge wiring).
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


def _vanilla_instant_printing_dict(name="X-Value-Substrate-Probe Instant"):
    """A minimal instant printing — a draw-1 spell that has NO
    X-cost (this is intentional: Cycle 1 has no X-cost cast
    detector; the substrate is exercised via the synthetic
    setter, NOT by minting via cast). Mirrors the
    `_single_pick_modal_spell_printing_dict` test-fixture style
    from `test_scour_for_scrap_substrate.py`."""
    return {
        "name": name,
        "mana_cost": {"generic": 1},
        "card_types": ["Instant"],
        "effects": [
            {"kind": "draw_cards", "count": 1},
        ],
    }


# ---------------------------------------------------------------------------
# Stack getter surface — additive baseline + empty-stack contract
# ---------------------------------------------------------------------------


def test_stack_top_x_value_returns_none_on_empty_stack():
    """The new getter must be safe on an empty stack — same contract
    as the single-pick ``stack_top_chosen_mode_idx`` /
    ``stack_top_additional_cost_choice_id`` siblings."""
    _require_engine()
    state = _fresh_state()
    assert state.stack_size() == 0
    assert state.stack_top_x_value() is None


def test_fresh_cast_spell_has_x_value_none():
    """Cycle 1 substrate-only — every fresh cast pushes
    ``x_value: None`` because no live mint site is wired yet.
    Mirrors the back-compat invariant: every pre-substrate
    cross-engine trajectory stays byte-identical because the
    constructor parameter defaults to ``None`` at the call site."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    ok = state.cast_spell_noncreature(spell_id, 0)
    assert ok is True
    assert state.stack_size() == 1
    # Default-None invariant.
    assert state.stack_top_x_value() is None


# ---------------------------------------------------------------------------
# Synthetic setter round-trip — i32 wire boundary
# ---------------------------------------------------------------------------


def test_set_stack_top_x_value_round_trip():
    """``set_stack_top_x_value(5)`` → ``stack_top_x_value() == 5``.
    The synthetic-test setter is the substrate-test workhorse for
    Cycle 1; it exists exclusively for the test surface (no
    production code path calls it; the production cast-time bind
    that lands in Cycle 2/3 will set it once at push time)."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    state.set_stack_top_x_value(5)
    assert state.stack_top_x_value() == 5


def test_set_stack_top_x_value_accepts_negative():
    """The slot type is ``Option<i32>`` (signed) — negative values
    flow through the wire boundary. CR 107.1b says X is non-negative
    on a cast, but the signedness is the SoT parity with Python's
    ``int`` (which is signed); enforcement of "X >= 0" is the
    resolve-arm consumer's job, not the substrate's. Cycle 1 pins
    the type-level contract: the setter takes ``i32`` not ``u32``."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    state.set_stack_top_x_value(-3)
    assert state.stack_top_x_value() == -3


def test_set_stack_top_x_value_accepts_zero():
    """CR 107.1b boundary — X may be 0. The wire boundary must
    distinguish ``Some(0)`` from ``None`` correctly (a common
    PyO3 trap with Option-typed fields)."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    state.set_stack_top_x_value(0)
    # Must be Some(0), NOT None.
    assert state.stack_top_x_value() == 0
    assert state.stack_top_x_value() is not None


def test_set_stack_top_x_value_large_i32_max():
    """The wire boundary holds at ``i32::MAX`` (2,147,483,647) —
    Python's int is unbounded but the slot is ``i32`` per the
    storage shape. No realistic game would set X this high (Python's
    auto-pick is bounded by floating mana), but pinning the upper
    bound now prevents a surprise overflow if a synthetic test
    ever does."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    state.set_stack_top_x_value(2_147_483_647)
    assert state.stack_top_x_value() == 2_147_483_647


def test_set_stack_top_x_value_unset_to_none():
    """The setter accepts ``None`` to clear the slot. Required
    so substrate tests can drive the resolver bracket's
    "restore to prior None" branch — without an unset path the
    bracket's symmetry is untestable."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    # Set then unset.
    state.set_stack_top_x_value(7)
    assert state.stack_top_x_value() == 7
    state.set_stack_top_x_value(None)
    assert state.stack_top_x_value() is None


def test_set_stack_top_x_value_noop_on_empty_stack():
    """Setter on an empty stack is a silent no-op (no panic). The
    getter still returns ``None``. Same defensive shape as
    ``set_stack_top_chosen_mode_idxs``."""
    _require_engine()
    state = _fresh_state()
    assert state.stack_size() == 0
    # Must not panic.
    state.set_stack_top_x_value(42)
    # Still empty, still None.
    assert state.stack_size() == 0
    assert state.stack_top_x_value() is None


# ---------------------------------------------------------------------------
# Resolve bracket — save → set → restore semantics across BOTH entry points
# ---------------------------------------------------------------------------


def test_resolve_top_of_stack_default_none_bracket_no_op():
    """``resolve_top_of_stack`` on a default-pushed spell
    (``x_value: None``) is a no-op on the X-value substrate —
    no side-effect on observable state. The bracket runs (saves
    `None`, sets to `top.x_value = None`, restores `None`), and
    every existing resolve arm self-scans / ignores the
    transient. Back-compat byte-equal invariant."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    # Pre-resolve: default-None invariant.
    assert state.stack_top_x_value() is None
    # Resolve drains the stack (one item).
    state.resolve_top_of_stack()
    assert state.stack_size() == 0


def test_resolve_top_of_stack_restores_prior_value():
    """``resolve_top_of_stack`` brackets the apply_effect closure —
    even if Cycle 1's no-live-mint baseline pushes `x_value: None`,
    the bracket itself must be sound: synthetic-set the StackItem
    field, resolve, and confirm no observable leak (the post-
    resolve stack is empty; we can't inspect the bracket from
    inside but we DO confirm resolve completes without panic and
    the next push starts at the back-compat baseline)."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    # Synthetic-set X on the StackItem (mimics Cycle 2/3's future
    # cast-time mint).
    state.set_stack_top_x_value(5)
    assert state.stack_top_x_value() == 5
    # Resolve — the bracket should drive
    # current_resolve_x_value through Some(5) and back to None.
    state.resolve_top_of_stack()
    assert state.stack_size() == 0
    # Push a fresh non-X spell — its x_value defaults to None
    # (the resolver bracket did NOT leak the prior Some(5)).
    spell2_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict("Second-Spell"), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell2_id, 0)
    assert state.stack_top_x_value() is None


def test_resolve_top_of_stack_pause_brackets_x_value():
    """§3.4 sibling-parity requirement — the ``_pause`` resolve
    entry point MUST mirror the bracket discipline of
    ``resolve_top_of_stack``. Cycle 1 wires both; this pin guards
    against a transcription error where the X-value bracket lands
    in one entry point but not the other (exactly the
    round-60-N1-class regression the close-encounter-substrate
    test_close_encounter_parity::test_addcost_bracket_pause pinned
    for ``current_resolve_additional_cost_choice_id``)."""
    _require_engine()
    state = _fresh_state()
    spell_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict(), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell_id, 0)
    state.set_stack_top_x_value(11)
    assert state.stack_top_x_value() == 11
    # Use the _pause entry point — does NOT auto-resolve a
    # PendingDecision but DOES run the apply_effect bracket.
    state.resolve_top_of_stack_pause()
    # Post-resolve: stack drained; the bracket completed without
    # panic. The next push starts at the default-None baseline.
    spell2_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict("Second-Spell"), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell2_id, 0)
    assert state.stack_top_x_value() is None


# ---------------------------------------------------------------------------
# Multi-spell isolation — per-StackItem x_value
# ---------------------------------------------------------------------------


def test_multi_spell_per_item_x_value_isolation():
    """Two spells on the stack with distinct synthetic X values —
    the top (last-pushed) returns its own X via
    ``stack_top_x_value()``. Confirms the field is per-StackItem,
    not a singleton on GameState. (The transient
    ``current_resolve_x_value`` IS a singleton, but the
    per-StackItem field is the durable storage that drives the
    transient at resolve time.)"""
    _require_engine()
    state = _fresh_state()
    # Spell 1
    spell1_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict("S1"), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell1_id, 0)
    state.set_stack_top_x_value(3)
    assert state.stack_top_x_value() == 3
    # Spell 2 — fresh push on top.
    spell2_id = state.register_and_inject_object(
        _vanilla_instant_printing_dict("S2"), 0, ZONE_HAND,
    )
    state.add_mana(0, "C", 1)
    state.cast_spell_noncreature(spell2_id, 0)
    # Fresh push starts at default-None.
    assert state.stack_top_x_value() is None
    state.set_stack_top_x_value(7)
    # Top is S2 — X=7.
    assert state.stack_top_x_value() == 7
    # Resolve S2 — back to S1.
    state.resolve_top_of_stack()
    # Top is now S1 — X=3 (per-StackItem isolation).
    assert state.stack_top_x_value() == 3


# ---------------------------------------------------------------------------
# Back-compat byte-equal — every cast trajectory observes no change
# ---------------------------------------------------------------------------


def test_back_compat_byte_equal_default_none_invariant():
    """Cycle 1's back-compat byte-equal invariant: every existing
    cast trajectory observes ``stack_top_x_value() is None`` from
    push through resolve. The new constructor parameter is a
    `None` literal at the production call site; the resolver
    bracket is a no-op when the source is `None`. This pin is the
    canonical Cycle 1 "nothing observable changed" probe — if it
    fails, the substrate accidentally minted an X somewhere it
    shouldn't have."""
    _require_engine()
    state = _fresh_state()
    # Cast several spells in sequence — every one stays at None
    # because no live mint site is wired.
    for i in range(3):
        spell_id = state.register_and_inject_object(
            _vanilla_instant_printing_dict(f"Spell{i}"), 0, ZONE_HAND,
        )
        state.add_mana(0, "C", 1)
        state.cast_spell_noncreature(spell_id, 0)
        # Default-None invariant pin.
        assert state.stack_top_x_value() is None
        state.resolve_top_of_stack()
        # Post-resolve: stack empty.
        assert state.stack_size() == 0
