"""Counter target spell on the Rust ``GameState``.

Composes round-5's stack slice (``push_to_stack`` / ``stack_size`` /
``is_stack_empty``) and round-1's zone slice (``move_object``) to
implement counter-spell semantics: remove a stack item by source
obj_id and move the source GameObject to its controller's graveyard
(CR 701.6 — countered spells are placed in their owner's graveyard).

Ported from ``tests/test_counter_cancellation.py`` is a misnomer —
that file is about +1/+1 / -1/-1 counter cancellation SBA, not
about the counter-target-spell action. The semantic source of truth
for the *spell-counter* operation is
``src/mtg_bot/effects/generic/stack_control.py:CounterTargetSpell``
(it sets ``item.countered = True`` and the resolution layer later
short-circuits the spell, sending the source to the graveyard).

**Slice scope simplifications** (mirrored in
``rust/src/state_ops_counterspell.rs``):

- **No "can't be countered" enforcement.** A future slice will add
  ``cant_be_countered`` to ``CardPrinting`` and reject here. For
  now we trust the caller picked a legal target.
- **Caller picks the target obj_id.** No verification that the
  source is itself a "counter spell" effect — that's a parser
  decision, not an engine primitive.
- **Source goes to controller's graveyard.** The stack item carries
  the controller; we look up the source GameObject in any zone and
  hand it to ``move_object`` with ``ZONE_GRAVEYARD`` and
  ``stack_item.controller`` as the destination. Real MTG would
  send it to the *owner's* graveyard (which equals controller for
  spells the player cast themselves) — owner tracking is a future
  slice.
- **Duplicate-source semantics: FIFO.** If two stack items share a
  source obj_id (rare but possible — e.g. a copy-this-spell
  effect that re-uses the source), ``counter_spell`` removes the
  bottom-most one (smallest stack index). Matches the existing
  ``remove_from_stack`` primitive's bottom-to-top scan.
- **No replacement effects, no triggers.** "When a spell is
  countered..." triggers (Wickerbough Elder etc.) are deferred.
"""

from __future__ import annotations


# Zone constants — mirror ``rust/src/state_ops_zones.rs``.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


def _vanilla_dict():
    """A no-effects printing — we don't need an effect for the
    counter-spell tests because resolve isn't being called. The source
    just needs to exist so the move-to-graveyard step can find it."""
    return {
        "name": "Test Spell",
        "card_types": ["Instant"],
        "mana_cost": {"generic": 1, "U": 1},
        "effects": [],
    }


# --- core remove-from-stack semantics --------------------------------


def test_counter_target_removes_stack_item(synthetic_state):
    """A stack item whose source is obj_id 42 vanishes on counter."""
    synthetic_state.push_to_stack(42, 0)
    assert synthetic_state.stack_size() == 1
    synthetic_state.counter_spell(42)
    assert synthetic_state.stack_size() == 0
    assert synthetic_state.is_stack_empty() is True


def test_counter_target_returns_true_on_success(synthetic_state):
    """``counter_spell`` returns True iff a matching stack item was
    found and removed."""
    synthetic_state.push_to_stack(42, 0)
    assert synthetic_state.counter_spell(42) is True


def test_counter_target_unknown_obj_id_returns_false(synthetic_state):
    """No stack item with the given source -> False, no mutation."""
    synthetic_state.push_to_stack(11, 0)
    assert synthetic_state.counter_spell(99) is False
    # Stack untouched.
    assert synthetic_state.stack_size() == 1
    assert synthetic_state.peek_top_of_stack() == {
        "source_obj_id": 11,
        "controller": 0,
    }


def test_counter_target_only_removes_one_item_when_duplicates(synthetic_state):
    """Two stack items share source_obj_id 42. ``counter_spell(42)``
    removes ONE — specifically the bottom-most (FIFO, smallest stack
    index), per the underlying ``remove_from_stack`` primitive's
    bottom-to-top scan. The other stays on the stack."""
    synthetic_state.push_to_stack(42, 0)  # bottom
    synthetic_state.push_to_stack(42, 1)  # top
    assert synthetic_state.stack_size() == 2

    assert synthetic_state.counter_spell(42) is True
    assert synthetic_state.stack_size() == 1
    # The remaining item is the one that was pushed second (controller=1)
    # because the bottom one (controller=0) was removed.
    remaining = synthetic_state.peek_top_of_stack()
    assert remaining == {"source_obj_id": 42, "controller": 1}


def test_counter_target_preserves_other_stack_items(synthetic_state):
    """Push A, B, C; counter B; verify A, C remain in original order."""
    synthetic_state.push_to_stack(10, 0)  # A: bottom
    synthetic_state.push_to_stack(20, 1)  # B: middle
    synthetic_state.push_to_stack(30, 0)  # C: top
    assert synthetic_state.stack_size() == 3

    assert synthetic_state.counter_spell(20) is True
    assert synthetic_state.stack_size() == 2
    # Top is still C (30).
    assert synthetic_state.peek_top_of_stack() == {
        "source_obj_id": 30,
        "controller": 0,
    }
    # Pop C and confirm A is now the top.
    synthetic_state.pop_top_of_stack()
    assert synthetic_state.peek_top_of_stack() == {
        "source_obj_id": 10,
        "controller": 0,
    }


# --- source-to-graveyard movement ------------------------------------


def test_counter_target_moves_source_to_graveyard(synthetic_state):
    """End-to-end: register a printing for player 0, inject a
    GameObject onto the battlefield (the spell's "source" — a
    stand-in for whatever zone the spell originated from — for first
    slice we just route to graveyard regardless), push it on the
    stack, counter it. Stack should empty AND the source should
    land in player 0's graveyard."""
    src_obj_id = synthetic_state.register_and_inject_object(
        _vanilla_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.push_to_stack(src_obj_id, 0)
    p0_graveyard_before = synthetic_state.graveyard_sizes[0]

    assert synthetic_state.counter_spell(src_obj_id) is True

    # (a) stack drained.
    assert synthetic_state.is_stack_empty() is True
    # (b) source object in p0's graveyard now.
    assert synthetic_state.find_object_zone(src_obj_id) == ZONE_GRAVEYARD
    assert synthetic_state.graveyard_sizes[0] == p0_graveyard_before + 1


def test_counter_does_not_affect_other_players_graveyard(synthetic_state):
    """Controller of countered spell is player 0 -> only p0's
    graveyard size grows; p1's stays put."""
    src_obj_id = synthetic_state.register_and_inject_object(
        _vanilla_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.push_to_stack(src_obj_id, 0)
    p1_graveyard_before = synthetic_state.graveyard_sizes[1]

    synthetic_state.counter_spell(src_obj_id)

    assert synthetic_state.graveyard_sizes[1] == p1_graveyard_before


# --- CoW canary ------------------------------------------------------


def test_clone_then_counter_does_not_affect_original(synthetic_state):
    """Counter on a clone must not bleed into the original — neither
    the stack nor the source's zone on the original side should
    change. Load-bearing CoW test."""
    src_obj_id = synthetic_state.register_and_inject_object(
        _vanilla_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.push_to_stack(src_obj_id, 0)
    cloned = synthetic_state.clone_once()

    assert cloned.counter_spell(src_obj_id) is True
    # Clone: stack drained, source in graveyard.
    assert cloned.is_stack_empty() is True
    assert cloned.find_object_zone(src_obj_id) == ZONE_GRAVEYARD

    # Original: stack still has the item, source still on battlefield.
    assert synthetic_state.stack_size() == 1
    assert synthetic_state.find_object_zone(src_obj_id) == ZONE_BATTLEFIELD
