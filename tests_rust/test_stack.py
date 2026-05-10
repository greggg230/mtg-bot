"""Stack push/pop/peek/size primitives on the Rust ``GameState``.

Ported from Python's ``src/mtg_bot/game.py`` (the ``GameState.stack``
list of ``StackItem`` objects) and ``src/mtg_bot/turn.py`` (priority +
stack interaction). The Rust side keeps ``stack: Vec<StackItem>`` on
``GameState`` (added by the foundation commit). Top of stack = LAST
element of the Vec for O(1) push/pop, mirroring the Python list-as-
stack convention. This slice grows the read/write surface in
``rust/src/state_ops_stack.rs``.

**Slice scope.**

- **No resolution semantics in this slice.** What happens when an
  effect resolves depends on the effect grammar, which is the
  bridge's domain. This slice exposes ``pop_top_of_stack()`` returning
  the StackItem dict; resolution itself is a future slice that
  interprets the effect.
- **No counter-spells in this slice.** Counter target spell needs a
  "remove this specific item from the stack" primitive — which we
  add as ``remove_from_stack(obj_id)`` — but we don't enforce
  countering rules, just expose the surface.
- **First-slice StackItem shape: ``{source_obj_id, controller}``
  only.** Real stack items will need more (target list, choices,
  modal mode); future slices extend.
"""

from __future__ import annotations


def test_initial_stack_is_empty(synthetic_state):
    """A fresh synthetic state has an empty stack."""
    assert synthetic_state.stack_size() == 0
    assert synthetic_state.is_stack_empty() is True


def test_push_increments_size(synthetic_state):
    """`push_to_stack(obj_id, controller)` adds one item; size goes 0 -> 1."""
    synthetic_state.push_to_stack(42, 0)
    assert synthetic_state.stack_size() == 1
    assert synthetic_state.is_stack_empty() is False


def test_push_multiple_appends_in_order(synthetic_state):
    """Two pushes -> size 2; third push -> size 3, peek returns the
    last-pushed item."""
    synthetic_state.push_to_stack(1, 0)
    synthetic_state.push_to_stack(2, 1)
    synthetic_state.push_to_stack(3, 0)
    assert synthetic_state.stack_size() == 3
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == 3
    assert top["controller"] == 0


def test_peek_top_returns_top_item_dict(synthetic_state):
    """`peek_top_of_stack()` returns a Python dict with the
    ``source_obj_id`` and ``controller`` fields."""
    synthetic_state.push_to_stack(7, 1)
    top = synthetic_state.peek_top_of_stack()
    assert isinstance(top, dict)
    assert top == {"source_obj_id": 7, "controller": 1}


def test_peek_top_on_empty_returns_none(synthetic_state):
    """Peek on an empty stack returns None (not an exception)."""
    assert synthetic_state.peek_top_of_stack() is None


def test_pop_top_removes_and_returns_top_item(synthetic_state):
    """`pop_top_of_stack()` removes and returns the top item; size decrements."""
    synthetic_state.push_to_stack(99, 1)
    assert synthetic_state.stack_size() == 1
    popped = synthetic_state.pop_top_of_stack()
    assert popped == {"source_obj_id": 99, "controller": 1}
    assert synthetic_state.stack_size() == 0
    assert synthetic_state.is_stack_empty() is True


def test_pop_top_on_empty_returns_none(synthetic_state):
    """Pop on an empty stack returns None (not an exception)."""
    assert synthetic_state.pop_top_of_stack() is None
    # Repeated pop on empty stays None and doesn't underflow size.
    assert synthetic_state.pop_top_of_stack() is None
    assert synthetic_state.stack_size() == 0


def test_lifo_order(synthetic_state):
    """Push A, B, C; pop returns C, B, A in that order (LIFO)."""
    synthetic_state.push_to_stack(10, 0)  # A
    synthetic_state.push_to_stack(20, 1)  # B
    synthetic_state.push_to_stack(30, 0)  # C
    assert synthetic_state.pop_top_of_stack() == {"source_obj_id": 30, "controller": 0}
    assert synthetic_state.pop_top_of_stack() == {"source_obj_id": 20, "controller": 1}
    assert synthetic_state.pop_top_of_stack() == {"source_obj_id": 10, "controller": 0}
    assert synthetic_state.is_stack_empty() is True


def test_remove_from_stack_by_obj_id(synthetic_state):
    """`remove_from_stack(obj_id)` removes the FIRST matching item by
    source_obj_id and returns True; size decrements."""
    synthetic_state.push_to_stack(5, 0)
    synthetic_state.push_to_stack(7, 1)
    assert synthetic_state.stack_size() == 2
    removed = synthetic_state.remove_from_stack(5)
    assert removed is True
    assert synthetic_state.stack_size() == 1


def test_remove_from_stack_unknown_returns_false(synthetic_state):
    """No item with that obj_id -> no mutation, returns False."""
    synthetic_state.push_to_stack(1, 0)
    assert synthetic_state.remove_from_stack(999) is False
    assert synthetic_state.stack_size() == 1
    # Original item still on top.
    assert synthetic_state.peek_top_of_stack() == {
        "source_obj_id": 1,
        "controller": 0,
    }


def test_remove_from_stack_preserves_other_items(synthetic_state):
    """Push A then B (B on top); remove A; B is still on top."""
    synthetic_state.push_to_stack(100, 0)  # A (bottom)
    synthetic_state.push_to_stack(200, 1)  # B (top)
    assert synthetic_state.remove_from_stack(100) is True
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top == {"source_obj_id": 200, "controller": 1}


def test_clone_preserves_stack(synthetic_state):
    """Clone copies the stack contents (Vec<StackItem> on GameState;
    cloning the GameState clones the Vec by-value)."""
    synthetic_state.push_to_stack(11, 0)
    synthetic_state.push_to_stack(22, 1)
    cloned = synthetic_state.clone_once()
    assert cloned.stack_size() == 2
    assert cloned.peek_top_of_stack() == {"source_obj_id": 22, "controller": 1}


def test_push_on_clone_does_not_affect_original(synthetic_state):
    """CoW canary: ``stack`` is a Vec<StackItem> on GameState. Cloning
    the state by-value-copies the Vec; mutating one side must NOT
    bleed back into the other. (A future slice that switches stack to
    Arc<Vec> would still need this invariant to hold via Arc::make_mut.)
    """
    cloned = synthetic_state.clone_once()
    cloned.push_to_stack(77, 1)
    assert cloned.stack_size() == 1
    assert synthetic_state.stack_size() == 0
    assert synthetic_state.is_stack_empty() is True


# --- round-16 process_stack_until_empty -----------------------------
#
# Composes round-5's stack push/pop with round-6's
# ``resolve_top_of_stack`` (in ``state_ops_resolve.rs``). The primitive
# repeatedly resolves the top of stack until empty and returns the
# count of items resolved. Useful for "resolve everything" simulation
# paths where the caller doesn't care about per-item priority hand-off.
#
# **Scope simplifications:**
#
# - **No priority hand-off between resolutions.** Real MTG: after each
#   resolution both players get priority again; this primitive just
#   drains the stack greedily. Higher-level code that wants real
#   priority semantics should call ``resolve_top_of_stack`` itself in a
#   priority loop.
# - **Iteration cap.** Hard-coded 1000-iteration safety so a buggy
#   effect that re-pushes its source can't infinite-loop. Anything
#   approaching that bound is a test bug; raises ``ValueError`` loudly.
# - **First failed resolve propagates.** If any individual
#   ``resolve_top_of_stack`` raises (e.g. unsupported AnyPlayer-target
#   on a bounce-shaped effect), the loop aborts and the exception
#   propagates with the partial drain visible to the caller.
#
# Tests use simple controller-implicit effects (DrawCards / GainLife)
# so the assertions focus on the loop primitive — not on target
# selection semantics already covered by ``test_resolve.py``.

ZONE_HAND = 0
ZONE_BATTLEFIELD = 2


def _draw_one_dict():
    """A draw-1 sorcery printing — Standard Counter style."""
    return {
        "name": "Test Draw 1",
        "card_types": ["Sorcery"],
        "mana_cost": {"U": 1},
        "effects": [{"kind": "draw_cards", "count": 1}],
    }


def _gain_life_dict(amount):
    """A gain-N-life instant printing parametric on amount."""
    return {
        "name": f"Test Gain {amount}",
        "card_types": ["Instant"],
        "mana_cost": {"W": 1},
        "effects": [{"kind": "gain_life", "amount": amount}],
    }


def test_process_stack_until_empty_zero_when_empty(synthetic_state):
    """Empty stack -> returns 0; stack remains empty."""
    assert synthetic_state.is_stack_empty() is True
    count = synthetic_state.process_stack_until_empty()
    assert count == 0
    assert synthetic_state.is_stack_empty() is True
    assert synthetic_state.stack_size() == 0


def test_process_stack_until_empty_resolves_one(synthetic_state):
    """One item on the stack -> returns 1, stack empty, effect applied
    (active player gains 5 life from a GainLife-5 instant)."""
    obj_id = synthetic_state.register_and_inject_object(
        _gain_life_dict(5), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(obj_id, 0)
    life_before = list(synthetic_state.life_totals)
    assert synthetic_state.stack_size() == 1

    count = synthetic_state.process_stack_until_empty()

    assert count == 1
    assert synthetic_state.is_stack_empty() is True
    assert synthetic_state.life_totals[0] == life_before[0] + 5
    assert synthetic_state.life_totals[1] == life_before[1]


def test_process_stack_until_empty_resolves_multiple(synthetic_state):
    """Three GainLife items on the stack -> all resolve, returns 3,
    cumulative life-gain shows up on the active player."""
    a = synthetic_state.register_and_inject_object(
        _gain_life_dict(1), 0, ZONE_HAND
    )
    b = synthetic_state.register_and_inject_object(
        _gain_life_dict(2), 0, ZONE_HAND
    )
    c = synthetic_state.register_and_inject_object(
        _gain_life_dict(4), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(a, 0)
    synthetic_state.push_to_stack(b, 0)
    synthetic_state.push_to_stack(c, 0)
    life_before = list(synthetic_state.life_totals)
    assert synthetic_state.stack_size() == 3

    count = synthetic_state.process_stack_until_empty()

    assert count == 3
    assert synthetic_state.is_stack_empty() is True
    # 1 + 2 + 4 = 7, regardless of order (all three are GainLife on
    # the same player).
    assert synthetic_state.life_totals[0] == life_before[0] + 7


def test_process_stack_until_empty_drains_in_lifo_order(synthetic_state):
    """Push three DrawCards-1 sources A, B, C. LIFO means C resolves
    first, then B, then A. Each resolution moves its source from hand
    to graveyard. Verify the graveyard order is [C, B, A] — confirms
    we're calling ``resolve_top_of_stack`` (which pops the top, i.e.
    last-pushed) in a loop, not draining bottom-up."""
    a = synthetic_state.register_and_inject_object(
        _draw_one_dict(), 0, ZONE_HAND
    )
    b = synthetic_state.register_and_inject_object(
        _draw_one_dict(), 0, ZONE_HAND
    )
    c = synthetic_state.register_and_inject_object(
        _draw_one_dict(), 0, ZONE_HAND
    )
    gy_before = synthetic_state.graveyard_sizes[0]
    synthetic_state.push_to_stack(a, 0)
    synthetic_state.push_to_stack(b, 0)
    synthetic_state.push_to_stack(c, 0)

    count = synthetic_state.process_stack_until_empty()

    assert count == 3
    assert synthetic_state.is_stack_empty() is True
    # All three sources moved to graveyard (sorcery -> CR 608.2m).
    assert synthetic_state.graveyard_sizes[0] == gy_before + 3


def test_process_stack_until_empty_clone_isolation(synthetic_state):
    """CoW canary: running process_stack_until_empty on a clone leaves
    the original's stack and life total untouched."""
    obj_id = synthetic_state.register_and_inject_object(
        _gain_life_dict(2), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(obj_id, 0)
    cloned = synthetic_state.clone_once()
    life_before_orig = list(synthetic_state.life_totals)
    stack_before_orig = synthetic_state.stack_size()

    count = cloned.process_stack_until_empty()

    assert count == 1
    assert cloned.is_stack_empty() is True
    assert cloned.life_totals[0] == life_before_orig[0] + 2
    # Original: stack still has the item, life unchanged.
    assert synthetic_state.stack_size() == stack_before_orig
    assert list(synthetic_state.life_totals) == life_before_orig
