"""Darkness-shape combat-damage prevention substrate — Cycle 1 (substrate-only).

Spec: lane brief (2026-05-21). The Rust side gains the state-level
``prevent_all_combat_damage_this_turn: bool`` field, the
``grant_prevent_all_combat_damage_this_turn`` setter, the
``prevent_all_combat_damage_this_turn`` getter, the standalone
``clear_prevent_all_combat_damage_this_turn`` drain primitive, and
the ``cleanup_step_for_active_player`` integration (drain pass +
``prevent_all_combat_damage_drained: bool`` summary dict key).

Cycle 1 is intentionally substrate-only — no combat-damage routing
site consumes the flag yet (that's Cycle 2); no
``Effect::PreventAllCombatDamageThisTurn`` arm mints it (that's
Cycle 3 with the bridge round-trip + admit-gate widen + PROBED flip).
The lane's tests use the Rust-side primitive directly.

Mirrors Python's ``state.prevent_combat_damage_this_turn`` field at
``src/mtg_bot/game.py:2732`` (set by
``PreventAllCombatDamageThisTurn.resolve`` at
``src/mtg_bot/effects/generic/stack_control.py:266``, consumed by
``combat._resolve_damage_step`` at ``src/mtg_bot/combat.py:479``,
cleared at end-of-turn cleanup at ``src/mtg_bot/turn.py:1102``).

**Tests cover (16 cases — > the 12+ floor in the lane brief):**

1. Default — every freshly-allocated state has
   ``prevent_all_combat_damage_this_turn == False`` on both the
   synthetic factory + ``make_state`` paths.
2. Snapshot — the getter reads back the back-compat default; calling
   from Python via attribute access returns ``False`` pre-grant.
3. Grant — ``grant_prevent_all_combat_damage_this_turn()`` flips the
   flag to ``True``; the getter reads back ``True``.
4. Idempotent — granting twice in succession is observable as the
   same ``True`` state; no double-write artifact.
5. Same-state short-circuit — granting an already-set flag is a
   no-op (the flag stays ``True``).
6. Cleanup drain (composition) —
   ``cleanup_step_for_active_player`` resets the flag back to
   ``False`` and reports ``prevent_all_combat_damage_drained:
   True`` in the summary dict.
7. Cleanup drain (warm path) — cleanup on a state with no prevention
   set reports ``prevent_all_combat_damage_drained: False`` and
   leaves the flag at ``False``.
8. Standalone drain — ``clear_prevent_all_combat_damage_this_turn``
   resets the flag and returns the pre-drain value as a bool.
9. Standalone drain (warm path) — calling clear on an unset flag
   returns ``False`` and leaves the flag at ``False``.
10. CoW invariant (grant) — granting on a cloned state does NOT
    mutate the source.
11. CoW invariant (drain) — draining on a cloned state does NOT
    mutate the source's flag (source-set, then clone, then drain
    on clone — source must still observe the flag set).
12. Summary dict — the cleanup-step summary carries
    ``prevent_all_combat_damage_drained`` alongside ``discarded``
    / ``damage_cleared`` / ``ueot_cleared`` /
    ``controller_overrides_cleared``. No pre-existing key was
    renamed.
13. Cleanup-summary back-compat — the new
    ``prevent_all_combat_damage_drained`` key composes additively
    with the four pre-existing drain keys; on a state with no
    prevention set, every count stays zero and the new key reads
    ``False``.
14. Clone before grant — a clone taken before granting on the
    source observes a fresh `False` flag independent of the
    source's later grant.
15. Multi-turn drain — granting on turn N, draining at cleanup,
    re-granting on turn N+1, draining at cleanup: each round is
    independent (the substrate has no cross-turn leakage).
16. Make-state path — the production-style ``make_state``
    construction site (the empty-pool case) also defaults the
    flag to ``False``.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


# ---------------------------------------------------------------------------
# Default + snapshot
# ---------------------------------------------------------------------------


def test_default_prevent_all_combat_damage_is_false():
    """Every freshly-allocated state (synthetic factory) has
    ``prevent_all_combat_damage_this_turn == False``. Pre-substrate
    pinned baselines stay byte-equivalent."""
    _require_engine()
    state = _fresh_state()
    assert state.prevent_all_combat_damage_this_turn is False


def test_make_state_defaults_prevent_all_combat_damage_to_false():
    """The production-style ``make_state`` construction site also
    defaults the flag to ``False``. Mirrors Python's
    ``GameState(prevent_combat_damage_this_turn=False)`` default at
    ``game.py:2732``."""
    _require_engine()
    import mtg_bot_engine
    # Empty pool — registry construction + empty libraries. Mirrors
    # how other cycle-1 substrate slices exercise this path
    # (registry-less synthetic-state + production-state both go
    # through the same field-defaults code path).
    registry = mtg_bot_engine.Registry()
    state = mtg_bot_engine.make_state(registry, [], [], None, None, None, None)
    assert state.prevent_all_combat_damage_this_turn is False


def test_getter_reads_back_default():
    """The PyO3 getter (`#[getter]` on
    ``prevent_all_combat_damage_this_turn``) reads back the
    back-compat ``False`` default. Attribute-access shape matches
    Python's ``state.prevent_combat_damage_this_turn`` read pattern
    at ``combat.py:479``."""
    _require_engine()
    state = _fresh_state()
    # Attribute access, not method call — matches Python's read site.
    assert state.prevent_all_combat_damage_this_turn is False


# ---------------------------------------------------------------------------
# Grant + readback
# ---------------------------------------------------------------------------


def test_grant_sets_flag_to_true():
    """``grant_prevent_all_combat_damage_this_turn()`` flips the
    flag to ``True``; the getter reads back ``True``. Mirrors
    Python's ``state.prevent_combat_damage_this_turn = True`` write
    at ``effects/generic/stack_control.py:266``."""
    _require_engine()
    state = _fresh_state()
    assert state.prevent_all_combat_damage_this_turn is False
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True


def test_grant_idempotent_same_state_twice():
    """Granting twice in succession is observable as the same
    ``True`` state. No double-write artifact (the underlying field
    is a single ``bool``; a second ``= true`` is a no-op)."""
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True


def test_grant_to_already_set_flag_short_circuits():
    """Granting an already-set flag is a no-op — the flag stays
    ``True``. (The substrate has no "was this already set" telemetry
    surface; the no-op is observable only via the stable ``True``
    readback.)"""
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    # Second grant — same state.
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True


# ---------------------------------------------------------------------------
# Cleanup-step drain (composed in cleanup_step_for_active_player)
# ---------------------------------------------------------------------------


def test_cleanup_step_drains_prevent_flag():
    """``cleanup_step_for_active_player`` includes a CR 514.3 drain
    pass that resets the prevention flag back to ``False``. The
    returned summary dict reports
    ``prevent_all_combat_damage_drained: True`` when the flag was
    set before the call. Mirrors Python's
    ``state.prevent_combat_damage_this_turn = False`` reset at
    ``turn.py:1102``."""
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    summary = state.cleanup_step_for_active_player()
    assert summary["prevent_all_combat_damage_drained"] is True
    assert state.prevent_all_combat_damage_this_turn is False


def test_cleanup_step_warm_path_no_prevention():
    """A cleanup step on a state with no prevention set reports
    ``prevent_all_combat_damage_drained: False`` and leaves the flag
    at ``False``. Pre-substrate baselines (which never set the
    flag) hit this path; the cycle-1 landing is back-compat for
    every such trajectory."""
    _require_engine()
    state = _fresh_state()
    assert state.prevent_all_combat_damage_this_turn is False
    summary = state.cleanup_step_for_active_player()
    assert summary["prevent_all_combat_damage_drained"] is False
    assert state.prevent_all_combat_damage_this_turn is False


def test_cleanup_step_summary_back_compat_with_existing_drain_keys():
    """The new ``prevent_all_combat_damage_drained`` key composes
    additively with the four pre-existing drain keys (``discarded``,
    ``damage_cleared``, ``ueot_cleared``,
    ``controller_overrides_cleared``). Pre-substrate callers that
    read only the old keys keep working — the new key is additive
    but the old keys' values are invariant on a state with no
    prevention set. Pins the cleanup-summary back-compat
    invariant that's the heart of the Cycle 1 byte-equal claim."""
    _require_engine()
    state = _fresh_state()
    # Pre-substrate baseline shape — no prevention set; cleanup
    # reports `False` for the new key and `0` for the old counts.
    summary = state.cleanup_step_for_active_player()
    assert summary["discarded"] == 0
    assert summary["damage_cleared"] == 0
    assert summary["ueot_cleared"] == 0
    assert summary["controller_overrides_cleared"] == 0
    assert summary["prevent_all_combat_damage_drained"] is False


# ---------------------------------------------------------------------------
# Standalone drain primitive
# ---------------------------------------------------------------------------


def test_clear_prevent_all_combat_damage_standalone():
    """The drain primitive is exposed independently of the full
    cleanup-step composition. Useful for tests that want to assert
    the drain in isolation from the hand-size + damage-clear +
    ueot-modifier + controller-override passes. Returns the
    pre-drain value as a bool (``True`` when the flag was set)."""
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    was_set = state.clear_prevent_all_combat_damage_this_turn()
    assert was_set is True
    assert state.prevent_all_combat_damage_this_turn is False


def test_clear_prevent_all_combat_damage_warm_path():
    """``clear_prevent_all_combat_damage_this_turn`` on a state
    with no prevention set returns ``False`` and leaves the flag
    at ``False``. Mirrors ``clear_ueot_modifiers_all``'s
    warm-path shape."""
    _require_engine()
    state = _fresh_state()
    assert state.prevent_all_combat_damage_this_turn is False
    was_set = state.clear_prevent_all_combat_damage_this_turn()
    assert was_set is False
    assert state.prevent_all_combat_damage_this_turn is False


# ---------------------------------------------------------------------------
# CoW invariant — grant and drain are clone-safe
# ---------------------------------------------------------------------------


def test_grant_on_clone_does_not_affect_source():
    """Granting the prevention flag on a cloned state must not
    mutate the source. Same CoW canary shape as
    ``test_clone_then_discard_does_not_affect_original`` and the
    sister UEOT-modifier slice. The flag is a single bool on the
    state — clone copies the value, mutation on the clone touches
    only the clone."""
    _require_engine()
    state = _fresh_state()
    cloned = state.clone_once()
    cloned.grant_prevent_all_combat_damage_this_turn()
    # Cloned: flag set.
    assert cloned.prevent_all_combat_damage_this_turn is True
    # Source: untouched.
    assert state.prevent_all_combat_damage_this_turn is False


def test_clear_on_clone_does_not_affect_source():
    """Draining the cloned state's flag must not touch the source's
    flag. Set the flag on the SOURCE first, then clone, then drain
    on the clone — source must still observe the flag set. Sister
    CoW canary for the cleanup side."""
    _require_engine()
    state = _fresh_state()
    state.grant_prevent_all_combat_damage_this_turn()
    cloned = state.clone_once()
    assert cloned.prevent_all_combat_damage_this_turn is True
    cloned.clear_prevent_all_combat_damage_this_turn()
    # Clone: drained.
    assert cloned.prevent_all_combat_damage_this_turn is False
    # Source: still has the flag set.
    assert state.prevent_all_combat_damage_this_turn is True


def test_clone_before_grant_observes_independent_default():
    """A clone taken BEFORE granting on the source observes a fresh
    ``False`` flag independent of the source's later grant. Pins
    the clone-as-snapshot semantic — the clone is a point-in-time
    copy, not a live view of the source."""
    _require_engine()
    state = _fresh_state()
    cloned = state.clone_once()
    state.grant_prevent_all_combat_damage_this_turn()
    # Source: granted.
    assert state.prevent_all_combat_damage_this_turn is True
    # Clone (from pre-grant moment): still False.
    assert cloned.prevent_all_combat_damage_this_turn is False


# ---------------------------------------------------------------------------
# Cleanup composition — prevent_all_combat_damage_drained coexists with
# other drain counts
# ---------------------------------------------------------------------------


def test_cleanup_summary_carries_all_five_keys():
    """The cleanup summary dict carries
    ``discarded`` + ``damage_cleared`` + ``ueot_cleared`` +
    ``controller_overrides_cleared`` + ``prevent_all_combat_damage_
    drained`` — Cycle 1 of this lane added the fifth. No
    pre-existing key was renamed; readers using the old four keys
    keep working."""
    _require_engine()
    state = _fresh_state()
    summary = state.cleanup_step_for_active_player()
    assert "discarded" in summary
    assert "damage_cleared" in summary
    assert "ueot_cleared" in summary
    assert "controller_overrides_cleared" in summary
    assert "prevent_all_combat_damage_drained" in summary


# ---------------------------------------------------------------------------
# Multi-turn — no cross-turn leakage of the prevention flag
# ---------------------------------------------------------------------------


def test_multi_turn_drain_no_cross_turn_leakage():
    """Granting on turn N, draining at cleanup, re-granting on
    turn N+1, draining at cleanup: each round is independent.
    Pins the substrate's per-turn lifecycle — no cross-turn
    leakage even if the drain is skipped on one turn (the drain is
    a single bool reset, not a counter)."""
    _require_engine()
    state = _fresh_state()
    # Turn N: grant + cleanup drain.
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    summary_a = state.cleanup_step_for_active_player()
    assert summary_a["prevent_all_combat_damage_drained"] is True
    assert state.prevent_all_combat_damage_this_turn is False
    # Turn N+1: re-grant + cleanup drain.
    state.grant_prevent_all_combat_damage_this_turn()
    assert state.prevent_all_combat_damage_this_turn is True
    summary_b = state.cleanup_step_for_active_player()
    assert summary_b["prevent_all_combat_damage_drained"] is True
    assert state.prevent_all_combat_damage_this_turn is False
