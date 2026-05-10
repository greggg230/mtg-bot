"""Priority management primitives on the Rust ``GameState``.

Ported from Python's ``src/mtg_bot/game.py`` (the
``GameState.priority_player_idx`` field) and ``src/mtg_bot/turn.py``
(the priority-round mechanism). The Rust side keeps
``priority_player`` as a ``u8`` field on ``GameState`` (already
present from the foundation commit) and exposes Python-callable
getters / setters and pass-priority semantics in
``rust/src/state_ops_priority.rs``.

**Slice scope.**
- Pass-priority is a 2-player toggle: ``priority_player`` flips to
  the other player. Multi-player priority order (CR 800.4 / APNAP) is
  out of scope until/unless the engine grows past 2 players.
- Composition with phase advancement (the active player gets
  priority at the start of each step in CR 117.1a) is NOT
  implemented in this slice. ``advance_phase`` lives in
  ``state_ops_steps.rs`` and intentionally does NOT touch
  ``priority_player`` here. A future slice will integrate
  state_ops_priority + state_ops_steps so the active-player-gets-
  priority rule fires automatically. The
  ``test_priority_unaffected_by_phase_advance_in_this_slice`` test
  locks that boundary in.
- A priority "round" (both players pass with empty stack -> step
  ends per CR 117.4) is NOT modeled in this slice — it depends on a
  stack object which the Rust engine does not have yet.
"""

from __future__ import annotations

import pytest


def test_initial_priority_is_active_player(synthetic_state):
    """Fresh synthetic state: priority_player matches active_player
    (both start at 0)."""
    assert synthetic_state.active_player == 0
    assert synthetic_state.get_priority_player() == 0


def test_get_priority_player_returns_field(synthetic_state):
    """The getter just exposes the underlying u8 field."""
    assert synthetic_state.get_priority_player() == 0


def test_pass_priority_flips_to_other_player(synthetic_state):
    """`pass_priority` toggles priority from 0 to 1 in the 2-player
    model and returns the new value."""
    new_priority = synthetic_state.pass_priority()
    assert new_priority == 1
    assert synthetic_state.get_priority_player() == 1


def test_pass_priority_twice_returns_to_original(synthetic_state):
    """The pass-priority toggle is involutive: two passes return to
    the starting priority player."""
    synthetic_state.pass_priority()
    synthetic_state.pass_priority()
    assert synthetic_state.get_priority_player() == 0


def test_give_priority_to_explicit(synthetic_state):
    """`give_priority_to(player_id)` directly sets priority to the
    named player (skips the toggle semantics)."""
    synthetic_state.give_priority_to(1)
    assert synthetic_state.get_priority_player() == 1
    synthetic_state.give_priority_to(0)
    assert synthetic_state.get_priority_player() == 0


def test_give_priority_to_unknown_raises_value_error(synthetic_state):
    """Out-of-range player ids are rejected loudly (CR 800.4 doesn't
    apply here — we only have player ids 0 and 1)."""
    with pytest.raises(ValueError):
        synthetic_state.give_priority_to(2)
    with pytest.raises(ValueError):
        synthetic_state.give_priority_to(255)
    # State unchanged after the rejected calls.
    assert synthetic_state.get_priority_player() == 0


def test_pass_priority_does_not_affect_active_player(synthetic_state):
    """Active player and priority player are independent fields.
    Passing priority moves only `priority_player`; `active_player`
    stays put (only turn rollover via advance_phase from CLEANUP
    swaps active_player)."""
    starting_active = synthetic_state.active_player
    synthetic_state.pass_priority()
    assert synthetic_state.active_player == starting_active
    synthetic_state.pass_priority()
    assert synthetic_state.active_player == starting_active


def test_clone_preserves_priority(synthetic_state):
    """Clone copies the priority_player field (it's a u8 — by-value
    clone, no Arc/CoW involved)."""
    synthetic_state.give_priority_to(1)
    cloned = synthetic_state.clone_once()
    assert cloned.get_priority_player() == 1


def test_pass_priority_on_clone_does_not_affect_original(synthetic_state):
    """CoW canary: priority_player is a u8 on GameState, so a clone is
    a by-value copy of the field. Mutating priority on the clone must
    NOT bleed back into the original. (For Arc-backed zones this is
    where CoW would be load-bearing; for a plain u8 it's free, but
    the test still locks in that mutating-the-clone semantics is what
    callers expect.)"""
    cloned = synthetic_state.clone_once()
    cloned.pass_priority()
    assert cloned.get_priority_player() == 1
    assert synthetic_state.get_priority_player() == 0


def test_priority_unaffected_by_phase_advance_in_this_slice(synthetic_state):
    """SLICE BOUNDARY LOCK-IN.

    CR 117.1a: at the start of each step (and after each spell or
    ability resolves), the active player gets priority. The Rust
    engine does NOT implement that composition yet — `advance_phase`
    is in `state_ops_steps.rs` and intentionally does not touch
    `priority_player`. A future slice will integrate the two
    primitives.

    Until then, this test pins the current behavior: advancing the
    phase leaves priority alone. If a future slice changes that, this
    test must be updated (or removed) deliberately so the integration
    is a conscious choice rather than a silent semantics drift.
    """
    # Hand priority to the non-active player so we can detect any
    # accidental "active player gets priority" reset.
    synthetic_state.give_priority_to(1)
    assert synthetic_state.get_priority_player() == 1
    synthetic_state.advance_phase()
    # Priority did NOT reset to the active player (0). This proves the
    # composition is NOT wired up yet.
    assert synthetic_state.get_priority_player() == 1


# ---------------------------------------------------------------------------
# Round-9 composition slice: priority + stack.
#
# When BOTH players consecutively pass priority while the stack is
# non-empty, the top stack item resolves (CR 117.4 / 117.5). This
# slice exposes ``complete_priority_round()`` as the priority-round-
# completion primitive: caller decides both players have passed, then
# invokes this method.
#
# **Scope (pop-only stepping-stone).** This method does NOT dispatch
# the resolved item's effects. It pops the top stack item, returns
# its dict, and resets priority to ``active_player`` (CR 117.1c — the
# AP gets priority after a spell or ability resolves). Real effect
# dispatch is the job of ``resolve_top_of_stack`` in
# ``state_ops_resolve.rs`` — a future composition slice can chain
# them together once the cross-file file-isolation rule relaxes.
#
# **Cross-file private-method friction.** This slice WANTS to call
# ``resolve_top_of_stack`` from ``state_ops_resolve.rs``, but that
# method is private to its module's pyclass impl block and the slice
# rules forbid touching that file. Per the recurring inlining
# pattern (5+ instances across slices: round-7 counterspell inlined
# move_object, round-8 resolve-effects inlined draw_card, etc.), the
# pragmatic call here is to inline a *minimal* version: pop + reset
# priority, no effect dispatch. This keeps the slice surface small
# and lets the real-effect path stay solely in ``state_ops_resolve``.
# ---------------------------------------------------------------------------


def test_complete_priority_round_with_empty_stack_returns_none(synthetic_state):
    """Empty stack -> ``complete_priority_round()`` returns ``None``
    and is a no-op on stack/priority. Mirrors the empty-stack
    convention of ``pop_top_of_stack`` / ``resolve_top_of_stack``."""
    assert synthetic_state.is_stack_empty() is True
    result = synthetic_state.complete_priority_round()
    assert result is None
    assert synthetic_state.is_stack_empty() is True


def test_complete_priority_round_with_nonempty_stack_returns_popped_item_dict(
    synthetic_state,
):
    """Stack has one item -> ``complete_priority_round()`` returns the
    popped item as a dict ({source_obj_id, controller}) and the stack
    shrinks by 1."""
    synthetic_state.push_to_stack(42, 0)
    assert synthetic_state.stack_size() == 1
    result = synthetic_state.complete_priority_round()
    assert isinstance(result, dict)
    assert result == {"source_obj_id": 42, "controller": 0}
    assert synthetic_state.stack_size() == 0
    assert synthetic_state.is_stack_empty() is True


def test_complete_priority_round_passes_priority_to_active_player(synthetic_state):
    """After a stack item resolves (or in this slice's stepping-stone
    shape, after the top is popped), priority returns to the active
    player (CR 117.1c). Even if priority was the non-active player at
    the time of the round-completion, it resets to AP."""
    synthetic_state.push_to_stack(9, 1)
    # Hand priority to the non-active player (player 1) before
    # completing the round, so we can detect the post-resolution reset.
    synthetic_state.give_priority_to(1)
    assert synthetic_state.get_priority_player() == 1
    synthetic_state.complete_priority_round()
    assert synthetic_state.get_priority_player() == synthetic_state.active_player
    assert synthetic_state.get_priority_player() == 0  # active_player == 0 in synth.


def test_complete_priority_round_does_not_advance_phase(synthetic_state):
    """Resolving a stack item does NOT advance the phase. Phase /
    step is the job of ``advance_phase`` from
    ``state_ops_steps.rs``; the priority-round-completion primitive
    just unwinds one stack item."""
    starting_phase = synthetic_state.current_phase()
    synthetic_state.push_to_stack(11, 0)
    synthetic_state.complete_priority_round()
    assert synthetic_state.current_phase() == starting_phase


def test_complete_priority_round_with_multi_item_stack_pops_only_top(synthetic_state):
    """Push A then B (B on top). ``complete_priority_round()`` pops
    only the TOP (B); A is still on the stack. Returns B's dict."""
    synthetic_state.push_to_stack(100, 0)  # A (bottom)
    synthetic_state.push_to_stack(200, 1)  # B (top)
    assert synthetic_state.stack_size() == 2
    result = synthetic_state.complete_priority_round()
    assert result == {"source_obj_id": 200, "controller": 1}
    assert synthetic_state.stack_size() == 1
    # A is still on top.
    top = synthetic_state.peek_top_of_stack()
    assert top == {"source_obj_id": 100, "controller": 0}


def test_clone_then_complete_priority_round_does_not_affect_original(synthetic_state):
    """CoW canary: completing a priority round on a clone unwinds the
    clone's stack and resets the clone's priority. The original's
    stack and priority must be untouched."""
    synthetic_state.push_to_stack(55, 0)
    synthetic_state.give_priority_to(1)
    cloned = synthetic_state.clone_once()
    cloned.complete_priority_round()
    # Clone: stack drained, priority reset to active player (0).
    assert cloned.stack_size() == 0
    assert cloned.get_priority_player() == 0
    # Original: untouched.
    assert synthetic_state.stack_size() == 1
    assert synthetic_state.get_priority_player() == 1
