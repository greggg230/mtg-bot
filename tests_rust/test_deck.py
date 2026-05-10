"""Deck-loading primitives on ``GameState``.

Ports the deck-replacement pattern from ``tests/_test_decks.py`` +
``src/mtg_bot/game.py`` (which builds a player's library by
constructing a fresh GameObject per card name in a decklist and
assigning them to ``player.library``). The Rust slice collapses that
pattern into a single primitive that owns:

- printing-id validation (each id must point at a real registry slot),
- player-id validation,
- fresh GameObject id allocation (must not collide with any existing
  GameObject in any zone on either player),
- whole-library replacement via ``Arc::make_mut`` (so cloning a
  GameState and then loading a deck on the clone leaves the source
  state's library untouched — CoW canary).

**Top-of-library convention.** The round-1 zone slice settled
``library[Vec::last()] = top of library``. ``load_deck`` honors that:
the LAST element of the input ``printing_ids`` list ends up at the
top of the library, so the next ``draw_card`` call returns it.

**No automatic shuffle.** Callers that want a randomized library
follow ``load_deck`` with ``shuffle_library_with_seed`` (round-4
library slice).
"""

from __future__ import annotations

import pytest


# Synthetic state's registry holds 3 printings:
#   pid 0 = Forest   (mana_produced=["G"], no P/T)
#   pid 1 = Grizzly Bears (creature, 2/2)
#   pid 2 = Llanowar Elves (creature 1/1, also produces G)
FOREST_PID = 0
BEARS_PID = 1
ELVES_PID = 2


def _all_obj_ids(state):
    """Collect every GameObject id across both players, all four
    zones. Used to verify uniqueness invariants."""
    ids = []
    for player_id in (0, 1):
        # We can't directly read zones from Python, but every object
        # has a unique id in the synthetic-state initializer, and
        # find_object_zone searches all four zones — we can probe
        # the size getters and walk via printing_name_of indirectly.
        # Easier path: use scry to dump the library, and rely on
        # synthetic-state ids being a contiguous range.
        pass
    return ids


def _library_obj_ids(state, player_id, n):
    """Top-first dump of player's library obj_ids (from scry)."""
    return state.scry(player_id, n)


# --- Happy path -------------------------------------------------------------


def test_load_deck_replaces_library(synthetic_state):
    """``load_deck(0, [Forest, Forest, Bears, Elves])`` swaps player
    0's library for 4 fresh GameObjects, one per printing_id in input
    order. Top of library = LAST element of the input list."""
    synthetic_state.load_deck(0, [FOREST_PID, FOREST_PID, BEARS_PID, ELVES_PID])

    assert synthetic_state.library_sizes[0] == 4
    # Top-first dump should resolve to: Elves (last), then Bears, then
    # Forest, then Forest.
    top4 = synthetic_state.scry(0, 4)
    assert len(top4) == 4
    names = [synthetic_state.printing_name_of(oid) for oid in top4]
    assert names == ["Llanowar Elves", "Grizzly Bears", "Forest", "Forest"]


def test_load_deck_replaces_existing_library(synthetic_state):
    """The OLD library is replaced wholesale, not appended to. Old
    obj_ids should no longer be present in the new library."""
    # Snapshot a few of the OLD library obj_ids — synthetic state
    # builds player 0 with library ids 17..66.
    old_top_id = synthetic_state.scry(0, 1)[0]
    assert synthetic_state.library_sizes[0] == 50

    synthetic_state.load_deck(0, [FOREST_PID, BEARS_PID])

    assert synthetic_state.library_sizes[0] == 2
    new_ids = synthetic_state.scry(0, 2)
    # Old top obj_id must NOT be in the new library — it was a fresh
    # build, not an append.
    assert old_top_id not in new_ids


def test_load_deck_top_of_library_is_last_element(synthetic_state):
    """First call to ``draw_card`` returns the LAST printing_id in the
    input list (top of library = Vec.last())."""
    # Distinguishable printing_ids so we can verify which one drew.
    synthetic_state.load_deck(0, [FOREST_PID, BEARS_PID, ELVES_PID])
    drawn_id = synthetic_state.draw_card(0)
    assert drawn_id is not None
    # The drawn card was at the top of the library; per the
    # convention that's the LAST input — Llanowar Elves.
    assert synthetic_state.printing_name_of(drawn_id) == "Llanowar Elves"


def test_load_deck_assigns_unique_ids(synthetic_state):
    """The 4 fresh GameObjects must have ids that don't collide with
    any existing GameObject (battlefield, hand, or other player's
    zones)."""
    # Pre-existing objects span ids 0..133 in the default synthetic
    # state (P0: 0..66 across battlefield/hand/library, P1: 67..133).
    # After load_deck on P0, the new library ids must avoid colliding
    # with P0's hand+battlefield (0..16) and P1's zones (67..133).
    synthetic_state.load_deck(0, [FOREST_PID, FOREST_PID, BEARS_PID, ELVES_PID])

    new_lib_ids = synthetic_state.scry(0, 4)
    # All unique among themselves.
    assert len(set(new_lib_ids)) == 4
    # No collision with P0's surviving hand or battlefield ids
    # (0..16). The synthetic state lays out P0 hand at 10..16 and
    # battlefield at 0..9; new ids must be outside that range.
    p0_existing = set(range(0, 17))  # battlefield + hand
    p1_existing = set(range(67, 134))  # all of P1
    for nid in new_lib_ids:
        assert nid not in p0_existing
        assert nid not in p1_existing


def test_load_deck_with_empty_list_clears_library(synthetic_state):
    """``load_deck(0, [])`` empties the library (size 0)."""
    assert synthetic_state.library_sizes[0] == 50
    synthetic_state.load_deck(0, [])
    assert synthetic_state.library_sizes[0] == 0
    assert synthetic_state.scry(0, 5) == []


# --- Validation -------------------------------------------------------------


def test_load_deck_invalid_printing_id_raises(synthetic_state):
    """An out-of-range printing_id raises ValueError. State must not
    be partially mutated — library sizes unchanged after the raise."""
    libs_before = synthetic_state.library_sizes
    with pytest.raises(ValueError):
        synthetic_state.load_deck(0, [FOREST_PID, 99])
    assert synthetic_state.library_sizes == libs_before


def test_load_deck_invalid_player_raises(synthetic_state):
    """An out-of-range player_id raises ValueError."""
    libs_before = synthetic_state.library_sizes
    with pytest.raises(ValueError):
        synthetic_state.load_deck(99, [FOREST_PID])
    assert synthetic_state.library_sizes == libs_before


# --- Isolation --------------------------------------------------------------


def test_load_deck_does_not_affect_other_player_library(synthetic_state):
    """Loading a deck on player 0 leaves player 1's library
    untouched."""
    p1_before = _library_obj_ids(synthetic_state, 1, 50)
    assert len(p1_before) == 50

    synthetic_state.load_deck(0, [FOREST_PID, BEARS_PID])

    p1_after = _library_obj_ids(synthetic_state, 1, 50)
    assert p1_after == p1_before
    assert synthetic_state.library_sizes[1] == 50


def test_clone_then_load_deck_does_not_affect_original(synthetic_state):
    """CoW canary: load_deck on a cloned state must not leak back into
    the source."""
    cloned = synthetic_state.clone_once()
    source_lib_before = _library_obj_ids(synthetic_state, 0, 50)

    cloned.load_deck(0, [FOREST_PID, BEARS_PID, ELVES_PID])

    # Clone diverged.
    assert cloned.library_sizes[0] == 3
    # Source untouched.
    assert synthetic_state.library_sizes[0] == 50
    source_lib_after = _library_obj_ids(synthetic_state, 0, 50)
    assert source_lib_after == source_lib_before
