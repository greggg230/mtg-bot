"""Zone-movement primitives on ``GameState``.

Ported from the zone-mutation patterns in ``src/mtg_bot/turn.py``
(``draw_card``) and ``src/mtg_bot/actions.py`` / ``effects/*`` (the
``card.zone = Zone.X`` + per-zone list mutation pattern). On the Python
side the operation is two coupled mutations — ``p.library.pop(0);
top.zone = Zone.HAND; p.hand.append(top)`` — so the Rust port collapses
both into a single primitive that owns the membership move + the
``GameObject.zone`` tag update.

**Zone identifier choice (option b in the slice brief):** raw ``u8``
constants rather than a Rust ``ZoneKind`` enum. Adding an enum to
``state.rs`` would have forced an edit on a file two sibling slices
also need to grow this commit, so we keep the encoding inline. The
constants are duplicated test-side for readability — this is a
first-slice expedient and a follow-up can either expose them as
module-level Python attrs on ``mtg_bot_engine`` or upgrade to an enum
once the parallel-slice constraint relaxes.
"""

from __future__ import annotations

import pytest

# Zone constants — must match the values in
# ``rust/src/state_ops_zones.rs``. Hard-coded here rather than imported
# so the test file is self-contained and the encoding contract is
# visible at the call site.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3
ZONE_STACK = 4
ZONE_EXILE = 5


# `make_synthetic_state(num_battlefield=10, num_hand=7, num_library=50)`
# lays player 0 out as:
#   battlefield: ids 0..9
#   hand:        ids 10..16
#   library:     ids 17..66
# and player 1's range starts at 67. We rely on those id ranges below.
P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17


def test_find_object_zone_returns_battlefield_for_battlefield_object(synthetic_state):
    """An object known to live on the battlefield resolves to
    ``ZONE_BATTLEFIELD`` regardless of the placeholder value its
    ``GameObject.zone`` field carried at construction time."""
    assert synthetic_state.find_object_zone(P0_FIRST_BATTLEFIELD_ID) == ZONE_BATTLEFIELD


def test_find_object_zone_returns_hand_for_hand_object(synthetic_state):
    assert synthetic_state.find_object_zone(P0_FIRST_HAND_ID) == ZONE_HAND


def test_find_object_zone_returns_library_for_library_object(synthetic_state):
    assert synthetic_state.find_object_zone(P0_FIRST_LIBRARY_ID) == ZONE_LIBRARY


def test_find_object_zone_unknown_id_returns_none(synthetic_state):
    """Far-out-of-range id returns None (no exception)."""
    assert synthetic_state.find_object_zone(999_999) is None


def test_move_object_to_graveyard_removes_from_source_zone(synthetic_state):
    """Moving a battlefield object into the graveyard removes it from
    the battlefield and pushes it onto the graveyard. The object's
    ``zone`` field updates to track the new home."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    bf_before, gy_before = (
        synthetic_state.battlefield_sizes,
        synthetic_state.graveyard_sizes,
    )
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD

    moved = synthetic_state.move_object(obj_id, ZONE_GRAVEYARD, 0)
    assert moved is True

    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert synthetic_state.battlefield_sizes[0] == bf_before[0] - 1
    assert synthetic_state.graveyard_sizes[0] == gy_before[0] + 1
    # Other player untouched.
    assert synthetic_state.battlefield_sizes[1] == bf_before[1]
    assert synthetic_state.graveyard_sizes[1] == gy_before[1]


def test_move_object_unknown_id_returns_false(synthetic_state):
    """Asked to move an id we don't know about: no-op, return False."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes
    moved = synthetic_state.move_object(999_999, ZONE_GRAVEYARD, 0)
    assert moved is False
    # All zones unchanged.
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before


def test_move_object_preserves_id_and_printing(synthetic_state):
    """The moved object keeps its identity — same id, same printing
    (so the registry lookup chain still resolves to the same name)."""
    obj_id = P0_FIRST_HAND_ID  # an elf
    name_before = synthetic_state.printing_name_of(obj_id)
    assert name_before == "Llanowar Elves"

    assert synthetic_state.move_object(obj_id, ZONE_GRAVEYARD, 0) is True

    # Same id, still resolves to the same printing.
    assert synthetic_state.printing_name_of(obj_id) == "Llanowar Elves"
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_clone_then_move_does_not_affect_original(synthetic_state):
    """Load-bearing CoW canary: mutating a zone on a clone must not
    leak back into the source state. Each ``Player`` zone is an
    ``Arc<Vec<GameObject>>``; ``Arc::make_mut`` clones the inner Vec
    on first write, isolating the clone from its peers."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    cloned = synthetic_state.clone_once()
    bf_source_before = synthetic_state.battlefield_sizes
    gy_source_before = synthetic_state.graveyard_sizes

    assert cloned.move_object(obj_id, ZONE_GRAVEYARD, 0) is True

    # Clone diverged.
    assert cloned.find_object_zone(obj_id) == ZONE_GRAVEYARD
    # Source untouched: object is still on the battlefield with the
    # same zone-size totals.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes == bf_source_before
    assert synthetic_state.graveyard_sizes == gy_source_before


def test_draw_card_moves_top_of_library_to_hand(synthetic_state):
    """``draw_card`` returns the drawn object's id, shrinks the
    library by one, and grows the hand by one. The Rust convention is
    that the LAST element of the library Vec is the "top of library"
    (Python's ``library.pop(0)`` treats the front as the top — the
    Rust port flips that for ``Vec::pop`` cheapness; documented in
    the Rust docstring)."""
    lib_before = synthetic_state.library_sizes
    hand_before = synthetic_state.hand_sizes

    drawn = synthetic_state.draw_card(0)

    assert drawn is not None
    assert isinstance(drawn, int)
    assert synthetic_state.library_sizes[0] == lib_before[0] - 1
    assert synthetic_state.hand_sizes[0] == hand_before[0] + 1
    # Other player untouched.
    assert synthetic_state.library_sizes[1] == lib_before[1]
    assert synthetic_state.hand_sizes[1] == hand_before[1]
    # Drawn card now lives in hand.
    assert synthetic_state.find_object_zone(drawn) == ZONE_HAND


def test_draw_card_from_empty_library_returns_none(engine):
    """Draw from an empty library returns None — the Python engine
    flips ``has_lost = True`` here, but the Rust slice keeps the
    primitive narrow: pop returns ``None``, callers layer policy on
    top later."""
    state = engine.make_synthetic_state(
        num_battlefield=2, num_hand=2, num_library=0
    )
    assert state.library_sizes[0] == 0
    assert state.draw_card(0) is None
    assert state.hand_sizes[0] == 2  # unchanged


# --- round-14 exile-aware zone primitives ----------------------------
#
# Round-1 ``find_object_zone`` and ``move_object`` only iterated four
# zones (HAND / LIBRARY / BATTLEFIELD / GRAVEYARD) — round-11 added
# ``Player.exile`` as a *write-only* zone reachable through the
# ``ExileTarget`` effect, but the read primitives didn't grow to cover
# it. This slice closes that gap so callers can:
#
# - resolve an object id that lives in exile to ``ZONE_EXILE``;
# - move an object INTO exile via ``move_object(..., ZONE_EXILE, p)``;
# - move an object OUT of exile back to a tracked zone
#   (``move_object(exile_obj, ZONE_BATTLEFIELD, p)``, the foundation
#   for "return-from-exile" effects landing in a later slice).
#
# The tests below use the existing ``ExileTarget`` resolve dispatch
# (round-11) to get an object into exile cleanly — then poke the
# round-14 read/write primitives. The CoW canary mirrors the
# round-1 ``test_clone_then_move_does_not_affect_original`` shape.


def _exile_creature_dict():
    """A tiny instant 'exile target creature' printing — mirrors the
    fixture in ``test_resolve.py`` but local to this file so the
    slice's tests don't share state with the resolve slice."""
    return {
        "name": "Test Exile Instant (zones-r14)",
        "mana_cost": {"W": 1, "generic": 1},
        "card_types": ["Instant"],
        "effects": [
            {
                "kind": "exile_target",
                "target": {"kind": "any_creature"},
            }
        ],
    }


def _put_object_into_p0_exile(state):
    """Helper: resolves an ExileTarget effect from p0 onto the first
    creature on p1's battlefield. The caveat: round-11 lands the
    exiled card in the TARGET's exile (== p1's exile under the
    controller==owner convention), so we then ``move_object`` it to
    p0's exile to give us a p0-exile case for the find tests.

    Returns the obj_id of the now-in-p0-exile object.
    """
    # Pick p1's first creature on battlefield as the target. In a
    # fresh synthetic state, p1's battlefield IDs are 67..76 (10
    # objects, half forest / half bears — bears are creatures, so
    # the FIRST creature target is the first bear at id 72 with
    # the default 5/5 split. ``ExileTarget`` walks p1's battlefield
    # and picks the first creature.)
    inj = state.register_and_inject_object(_exile_creature_dict(), 0, ZONE_HAND)
    state.push_to_stack(inj, 0)
    state.resolve_top_of_stack()
    # Now there's exactly one extra entry in p1's exile. Find it by
    # scanning ids: anything in p1's exile with damage==0, counters==0
    # is fine. We need its id. Use exile_contains across the known
    # P1 battlefield id range to discover which one landed.
    for candidate in range(67, 77):
        if state.exile_contains(candidate):
            # Move it from p1 exile to p0 exile so the test exercises
            # find_object_zone / move_object on p0's exile zone.
            assert state.move_object(candidate, ZONE_EXILE, 0) is True
            return candidate
    raise AssertionError("ExileTarget didn't move any p1 creature")


def test_find_object_zone_finds_exiled_object(synthetic_state):
    """An object that lives in a player's exile resolves to
    ``ZONE_EXILE``. Pre-r14 ``find_object_zone`` returned None for
    exile residents (it didn't iterate the exile zone)."""
    obj_id = _put_object_into_p0_exile(synthetic_state)
    assert synthetic_state.exile_contains(obj_id) is True
    # The behavior under test:
    assert synthetic_state.find_object_zone(obj_id) == ZONE_EXILE


def test_move_object_to_exile_zone(synthetic_state):
    """``move_object(obj_id, ZONE_EXILE, target_player)`` moves an
    object from its current tracked zone into the target player's
    exile. Pre-r14 ``move_object`` silently routed unknown
    destinations (incl. exile) into the graveyard — this test pins
    the new dispatch arm."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes
    p0_exile_before = synthetic_state.exile_sizes[0]
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD

    moved = synthetic_state.move_object(obj_id, ZONE_EXILE, 0)
    assert moved is True

    # Now in p0's exile, gone from battlefield, NOT in graveyard.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_EXILE
    assert synthetic_state.exile_contains(obj_id) is True
    assert synthetic_state.battlefield_sizes[0] == bf_before[0] - 1
    assert synthetic_state.exile_sizes[0] == p0_exile_before + 1
    # Graveyard NOT bumped — this distinguishes ZONE_EXILE from the
    # pre-r14 "unknown -> graveyard" fallback.
    assert synthetic_state.graveyard_sizes == gy_before
    # Other player untouched.
    assert synthetic_state.battlefield_sizes[1] == bf_before[1]
    assert synthetic_state.exile_sizes[1] == 0


def test_move_object_from_exile_back_to_battlefield(synthetic_state):
    """An object in exile can be moved back to a tracked zone. This
    is the foundational primitive a future "return from exile"
    effect (e.g. flicker) will sit on top of."""
    obj_id = _put_object_into_p0_exile(synthetic_state)
    assert synthetic_state.find_object_zone(obj_id) == ZONE_EXILE
    p0_exile_before = synthetic_state.exile_sizes[0]
    p0_bf_before = synthetic_state.battlefield_sizes[0]

    moved = synthetic_state.move_object(obj_id, ZONE_BATTLEFIELD, 0)
    assert moved is True

    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.exile_contains(obj_id) is False
    assert synthetic_state.exile_sizes[0] == p0_exile_before - 1
    assert synthetic_state.battlefield_sizes[0] == p0_bf_before + 1


def test_clone_then_move_to_exile_does_not_affect_original(synthetic_state):
    """CoW canary for the new exile-aware ``move_object`` arm. Mutating
    a clone (move battlefield obj -> p0 exile) must not bleed back
    into the source state."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    cloned = synthetic_state.clone_once()
    bf_source_before = synthetic_state.battlefield_sizes
    exile_source_before = synthetic_state.exile_sizes

    assert cloned.move_object(obj_id, ZONE_EXILE, 0) is True

    # Clone diverged: object is in clone's p0 exile.
    assert cloned.find_object_zone(obj_id) == ZONE_EXILE
    assert cloned.exile_contains(obj_id) is True
    assert cloned.exile_sizes[0] == exile_source_before[0] + 1
    # Source untouched: object is still on the battlefield with the
    # same zone-size totals; exile is empty as before.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes == bf_source_before
    assert synthetic_state.exile_sizes == exile_source_before
    assert synthetic_state.exile_contains(obj_id) is False
