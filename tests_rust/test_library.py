"""Library / discard / scry primitives on ``GameState``.

Ported from the library-mutation patterns in ``src/mtg_bot/game.py``
(``shuffle_library``) and the discard pattern in
``src/mtg_bot/actions.py`` (ward-discard: ``p.hand.remove(card);
card.zone = Zone.GRAVEYARD; p.graveyard.append(card)``). On the Python
side these are open-coded at each call site; the Rust port collapses
them into single primitives that own membership + zone-tag semantics.

**RNG choice (inline xorshift64):** to avoid a Cargo.toml race with
sibling slices, the shuffle uses a tiny inline xorshift64 generator
rather than pulling in the ``rand`` crate. Determinism guarantee:
``shuffle_library_with_seed(player_id, seed)`` is a pure function of
``(library, seed)`` — same library + same seed produces the same
permutation, byte for byte.

**"Top of library" convention** (inherited from
``state_ops_zones.rs``): the LAST element of the library ``Vec`` is
the top of the library. ``scry`` therefore reads the tail and returns
ids in top-first order (last element first).
"""

from __future__ import annotations

import pytest


# `make_synthetic_state(num_battlefield=10, num_hand=7, num_library=50)`
# lays player 0 out as:
#   battlefield: ids 0..9
#   hand:        ids 10..16
#   library:     ids 17..66
# Player 1's range starts at 67.
P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17
P0_LIBRARY_SIZE = 50


# --- Discard ----------------------------------------------------------------


def test_discard_moves_card_to_graveyard(synthetic_state):
    """Discarding a card from hand removes it from hand and pushes it
    onto the graveyard. The object's identity is preserved (registry
    name still resolves)."""
    obj_id = P0_FIRST_HAND_ID  # an elf
    hand_before = synthetic_state.hand_sizes
    gy_before = synthetic_state.graveyard_sizes
    name_before = synthetic_state.printing_name_of(obj_id)
    assert name_before == "Llanowar Elves"

    discarded = synthetic_state.discard(obj_id, 0)

    assert discarded is True
    assert synthetic_state.hand_sizes[0] == hand_before[0] - 1
    assert synthetic_state.graveyard_sizes[0] == gy_before[0] + 1
    # Other player untouched.
    assert synthetic_state.hand_sizes[1] == hand_before[1]
    assert synthetic_state.graveyard_sizes[1] == gy_before[1]
    # Same id, same printing — found in the graveyard now.
    assert synthetic_state.printing_name_of(obj_id) == "Llanowar Elves"


def test_discard_unknown_id_returns_false(synthetic_state):
    """Far-out-of-range id returns False (no exception, no zone
    mutation)."""
    hand_before = synthetic_state.hand_sizes
    gy_before = synthetic_state.graveyard_sizes
    assert synthetic_state.discard(999_999, 0) is False
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.graveyard_sizes == gy_before


def test_discard_card_not_in_hand_returns_false(synthetic_state):
    """Trying to discard a battlefield object: precondition fails,
    return False, no zone mutation."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    bf_before = synthetic_state.battlefield_sizes
    hand_before = synthetic_state.hand_sizes
    gy_before = synthetic_state.graveyard_sizes

    assert synthetic_state.discard(obj_id, 0) is False

    # Object stayed on the battlefield.
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.graveyard_sizes == gy_before


def test_discard_clone_does_not_affect_original(synthetic_state):
    """Load-bearing CoW canary: discarding on a clone must not leak
    back into the source state."""
    obj_id = P0_FIRST_HAND_ID
    cloned = synthetic_state.clone_once()
    hand_source_before = synthetic_state.hand_sizes
    gy_source_before = synthetic_state.graveyard_sizes

    assert cloned.discard(obj_id, 0) is True

    # Clone diverged.
    assert cloned.hand_sizes[0] == hand_source_before[0] - 1
    assert cloned.graveyard_sizes[0] == gy_source_before[0] + 1
    # Source untouched.
    assert synthetic_state.hand_sizes == hand_source_before
    assert synthetic_state.graveyard_sizes == gy_source_before


# --- Shuffle ---------------------------------------------------------------


def _library_obj_ids(state, player_id, scry_n):
    """Snapshot the player's library obj_ids by scrying the whole
    thing (top-first). The library size is at most P0_LIBRARY_SIZE
    here, so scry(player, P0_LIBRARY_SIZE) returns the entire library
    in top-first order without mutating it."""
    return state.scry(player_id, scry_n)


def test_shuffle_with_seed_is_deterministic(synthetic_state):
    """Same source state cloned twice + shuffled with the same seed
    produces identical library orders. Pure function of (library,
    seed)."""
    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()

    a.shuffle_library_with_seed(0, 42)
    b.shuffle_library_with_seed(0, 42)

    a_order = _library_obj_ids(a, 0, P0_LIBRARY_SIZE)
    b_order = _library_obj_ids(b, 0, P0_LIBRARY_SIZE)

    assert a_order == b_order
    assert len(a_order) == P0_LIBRARY_SIZE


def test_shuffle_with_different_seeds_likely_different(synthetic_state):
    """Different seeds produce different orders. Probabilistic, but
    for a 50-card library two random shuffles colliding is ~1/50!,
    i.e. effectively never."""
    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()

    a.shuffle_library_with_seed(0, 42)
    b.shuffle_library_with_seed(0, 99)

    a_order = _library_obj_ids(a, 0, P0_LIBRARY_SIZE)
    b_order = _library_obj_ids(b, 0, P0_LIBRARY_SIZE)

    assert a_order != b_order


def test_shuffle_preserves_card_set(synthetic_state):
    """Shuffle is a permutation: same set of obj_ids before and after,
    no duplicates, no losses."""
    before = sorted(_library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE))
    synthetic_state.shuffle_library_with_seed(0, 12345)
    after = sorted(_library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE))
    assert before == after
    assert len(set(after)) == P0_LIBRARY_SIZE


def test_shuffle_clone_does_not_affect_original(synthetic_state):
    """Shuffling a clone's library must not leak back into the
    source. CoW canary."""
    cloned = synthetic_state.clone_once()
    source_before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)

    cloned.shuffle_library_with_seed(0, 7)
    cloned_after = _library_obj_ids(cloned, 0, P0_LIBRARY_SIZE)

    # Source still in original order.
    source_after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert source_after == source_before
    # Clone actually got shuffled (it shouldn't match the source for
    # a 50-card library with any non-degenerate seed).
    assert cloned_after != source_before


# --- Scry ------------------------------------------------------------------


def test_scry_top_n_returns_n_obj_ids_in_top_order(synthetic_state):
    """``scry(0, 3)`` returns the top 3 obj_ids of player 0's library
    in top-first order. Top of library = LAST element of the Vec, so
    the returned list reads the tail in reverse."""
    top3 = synthetic_state.scry(0, 3)
    assert len(top3) == 3
    # Library was built as ids 17..66 in push order, so the LAST 3
    # pushed are 64, 65, 66; top-first iteration yields 66, 65, 64.
    assert top3 == [66, 65, 64]


def test_scry_more_than_library_size_returns_what_exists(synthetic_state):
    """Scry N where N > library size: clamp to library size, return
    everything that exists (top-first)."""
    all_ids = synthetic_state.scry(0, 100)
    assert len(all_ids) == P0_LIBRARY_SIZE


def test_scry_zero_returns_empty(synthetic_state):
    assert synthetic_state.scry(0, 0) == []


def test_scry_does_not_mutate_library(synthetic_state):
    """Scry is a peek — library size is unchanged after the call."""
    lib_before = synthetic_state.library_sizes
    _ = synthetic_state.scry(0, 5)
    assert synthetic_state.library_sizes == lib_before


def test_apply_scry_keep_on_top_preserves_order(synthetic_state):
    """Scry the top 3, put all 3 back on top in the same order:
    library is unchanged."""
    before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    top3 = synthetic_state.scry(0, 3)

    ok = synthetic_state.apply_scry(0, top3, [])
    assert ok is True

    after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert after == before
    assert synthetic_state.library_sizes[0] == P0_LIBRARY_SIZE


def test_apply_scry_send_some_to_bottom_reorders_library(synthetic_state):
    """Scry top 3 (ids 66, 65, 64), keep 66 + 65 on top, send 64 to
    the bottom. Library size unchanged; new top-first reads 66, 65,
    then-the-old-fourth-card (id 63); the bottom card (formerly id
    17) is now id 64."""
    top3 = synthetic_state.scry(0, 3)
    assert top3 == [66, 65, 64]

    ok = synthetic_state.apply_scry(0, [66, 65], [64])
    assert ok is True

    assert synthetic_state.library_sizes[0] == P0_LIBRARY_SIZE
    new_top3 = synthetic_state.scry(0, 3)
    assert new_top3 == [66, 65, 63]

    # Old bottom card (id 17) is now second-from-bottom; new bottom
    # is id 64. Read whole library and check the very bottom (= first
    # element of the Vec, = LAST element of a top-first scry).
    all_ids = synthetic_state.scry(0, P0_LIBRARY_SIZE)
    assert all_ids[-1] == 64
    assert all_ids[-2] == 17


def test_apply_scry_validation_failure_returns_false(synthetic_state):
    """If the union of top_ids + bottom_ids does not match the top N
    we peeked at, validation fails and the library is unchanged."""
    before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    # Pretend we scryed 2, but pass an id that isn't in the top 2.
    ok = synthetic_state.apply_scry(0, [66, 999_999], [])
    assert ok is False
    after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert after == before


# --- Search library --------------------------------------------------------

# In synthetic_state, the library is built entirely of Forests (pid 0)
# at ids 17..66 for player 0. So a search for FOREST_PID must hit, and
# a search for an unknown printing_id must miss.
FOREST_PID = 0
ZONE_HAND = 0


def test_search_library_for_first_with_printing_returns_match(synthetic_state):
    """Synthetic state seeds Forest in library; search returns the
    obj_id of the FIRST GameObject in the library Vec matching the
    printing_id. The library Vec was filled push-order (ids 17..66),
    so lib[0] = id 17 = the first match."""
    found = synthetic_state.search_library_for_first_with_printing(0, FOREST_PID)
    assert found == P0_FIRST_LIBRARY_ID  # 17


def test_search_library_for_first_with_printing_returns_none_when_not_found(
    synthetic_state,
):
    """Search for a printing_id that isn't in the library returns
    None (no exception, no mutation)."""
    found = synthetic_state.search_library_for_first_with_printing(0, 99)
    assert found is None


def test_search_library_for_first_does_not_move_or_reorder(synthetic_state):
    """Search is a pure read — library order and size unchanged after
    the call."""
    before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    size_before = synthetic_state.library_sizes
    hand_before = synthetic_state.hand_sizes

    _ = synthetic_state.search_library_for_first_with_printing(0, FOREST_PID)

    after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert after == before
    assert synthetic_state.library_sizes == size_before
    assert synthetic_state.hand_sizes == hand_before


def test_search_library_and_move_to_hand_moves_card(synthetic_state):
    """Search + move: hand grows by 1, library shrinks by 1."""
    hand_before = synthetic_state.hand_sizes
    lib_before = synthetic_state.library_sizes

    found = synthetic_state.search_library_and_move_to_hand(0, FOREST_PID, 42)

    assert found is not None
    assert synthetic_state.hand_sizes[0] == hand_before[0] + 1
    assert synthetic_state.library_sizes[0] == lib_before[0] - 1
    # Other player untouched.
    assert synthetic_state.hand_sizes[1] == hand_before[1]
    assert synthetic_state.library_sizes[1] == lib_before[1]


def test_search_library_and_move_to_hand_zone_field_updated(synthetic_state):
    """Moved card's zone field is set to ZONE_HAND, and the card lives
    in hand now (find_object_zone reports ZONE_HAND)."""
    found = synthetic_state.search_library_and_move_to_hand(0, FOREST_PID, 42)
    assert found is not None
    assert synthetic_state.find_object_zone(found) == ZONE_HAND


def test_search_library_and_move_to_hand_failed_search_still_shuffles(
    synthetic_state,
):
    """Real MTG: even a failed search reveals to opponents and shuffles
    the library. Two clones, search for missing pid with different
    seeds, libraries should differ."""
    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()
    hand_before = synthetic_state.hand_sizes

    found_a = a.search_library_and_move_to_hand(0, 99, 1)
    found_b = b.search_library_and_move_to_hand(0, 99, 99)

    assert found_a is None
    assert found_b is None
    # Hand sizes unchanged on both — nothing moved.
    assert a.hand_sizes == hand_before
    assert b.hand_sizes == hand_before
    # Library sizes unchanged on both — nothing moved.
    assert a.library_sizes == synthetic_state.library_sizes
    assert b.library_sizes == synthetic_state.library_sizes
    # But the libraries WERE shuffled — different seeds produce
    # different orders.
    a_order = _library_obj_ids(a, 0, P0_LIBRARY_SIZE)
    b_order = _library_obj_ids(b, 0, P0_LIBRARY_SIZE)
    assert a_order != b_order


def test_search_library_and_move_to_hand_returns_none_on_miss(synthetic_state):
    """Missing printing_id returns None (and shuffles, per the rule
    above — but the return value is None)."""
    found = synthetic_state.search_library_and_move_to_hand(0, 99, 7)
    assert found is None


def test_clone_then_search_does_not_affect_original(synthetic_state):
    """CoW canary: search+move on a clone must not leak back into the
    source. Both library order AND hand membership are checked."""
    cloned = synthetic_state.clone_once()
    source_lib_before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    source_hand_before = synthetic_state.hand_sizes
    source_lib_size_before = synthetic_state.library_sizes

    found = cloned.search_library_and_move_to_hand(0, FOREST_PID, 42)
    assert found is not None

    # Source untouched.
    assert synthetic_state.hand_sizes == source_hand_before
    assert synthetic_state.library_sizes == source_lib_size_before
    source_lib_after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert source_lib_after == source_lib_before


# --- Search library by type (round-15) -------------------------------------

# Synthetic registry pids:
#   pid 0 = Forest         (card_types = ["Land"])
#   pid 1 = Grizzly Bears  (card_types = ["Creature"])
#   pid 2 = Llanowar Elves (card_types = ["Creature"])
BEARS_PID = 1
ELVES_PID = 2


def test_search_library_by_type_finds_creature(synthetic_state):
    """Library laid out (bottom -> top) as [Forest, Forest, Bears, Forest]:
    top-to-bottom search for "Creature" returns the Bears' obj_id (the
    Forest at the top is skipped, Bears is the next card down)."""
    synthetic_state.load_deck(0, [FOREST_PID, FOREST_PID, BEARS_PID, FOREST_PID])
    # After load_deck, top-of-library = the LAST input = Forest. Below
    # that = Bears. Top-to-bottom iteration skips the Forest at the very
    # top and returns Bears' obj_id.
    top_first_ids = synthetic_state.scry(0, 4)
    bears_id = top_first_ids[1]  # second from top
    assert synthetic_state.printing_name_of(bears_id) == "Grizzly Bears"

    found = synthetic_state.search_library_for_first_of_type(0, "Creature")
    assert found == bears_id


def test_search_library_by_type_finds_land(synthetic_state):
    """Default synthetic state has 50 Forests in the library. A
    top-to-bottom search for "Land" returns the obj_id of the TOP of
    the library (id 66 — the last element of the Vec)."""
    found = synthetic_state.search_library_for_first_of_type(0, "Land")
    # Top of library = last element of Vec; synthetic seeded ids 17..66
    # so id 66 is the top.
    assert found == 66


def test_search_library_by_type_unknown_type_returns_none(synthetic_state):
    """Searching for a type string with no matching card in the library
    returns None (no exception, no mutation). Default library is all
    Forests, so "Planeswalker" finds nothing."""
    found = synthetic_state.search_library_for_first_of_type(0, "Planeswalker")
    assert found is None


def test_search_library_by_type_empty_library_returns_none(synthetic_state):
    """Empty library + any search type returns None."""
    synthetic_state.load_deck(0, [])
    assert synthetic_state.library_sizes[0] == 0
    found = synthetic_state.search_library_for_first_of_type(0, "Land")
    assert found is None


def test_search_library_by_type_unknown_player_raises(synthetic_state):
    """An out-of-range player_id raises ValueError rather than silently
    returning None."""
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.search_library_for_first_of_type(7, "Creature")


def test_search_library_by_type_does_not_mutate_library(synthetic_state):
    """Search alone is a pure read — library order and size unchanged
    after the call."""
    before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    size_before = synthetic_state.library_sizes
    hand_before = synthetic_state.hand_sizes

    _ = synthetic_state.search_library_for_first_of_type(0, "Land")

    after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert after == before
    assert synthetic_state.library_sizes == size_before
    assert synthetic_state.hand_sizes == hand_before


def test_search_and_move_to_hand_moves_card(synthetic_state):
    """Composition: search creature + move to hand. Library shrinks by
    1, hand grows by 1, moved card has zone = ZONE_HAND."""
    synthetic_state.load_deck(0, [FOREST_PID, FOREST_PID, BEARS_PID, FOREST_PID])
    hand_before = synthetic_state.hand_sizes
    lib_before = synthetic_state.library_sizes

    found = synthetic_state.search_library_for_first_of_type_and_move_to_hand(
        0, "Creature"
    )

    assert found is not None
    assert synthetic_state.printing_name_of(found) == "Grizzly Bears"
    assert synthetic_state.hand_sizes[0] == hand_before[0] + 1
    assert synthetic_state.library_sizes[0] == lib_before[0] - 1
    # Moved card now lives in hand.
    assert synthetic_state.find_object_zone(found) == ZONE_HAND
    # Other player untouched.
    assert synthetic_state.hand_sizes[1] == hand_before[1]
    assert synthetic_state.library_sizes[1] == lib_before[1]


def test_search_and_move_to_hand_no_match_returns_none_no_mutation(synthetic_state):
    """No match: returns None, library unchanged, hand unchanged.
    Unlike round-14's printing-search variant (which shuffles on miss),
    this composition is a pure no-op when nothing matches — the
    library/hand sizes and ordering are bit-for-bit identical."""
    lib_before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    hand_before = synthetic_state.hand_sizes
    lib_size_before = synthetic_state.library_sizes

    found = synthetic_state.search_library_for_first_of_type_and_move_to_hand(
        0, "Planeswalker"
    )

    assert found is None
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.library_sizes == lib_size_before
    lib_after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert lib_after == lib_before


# --- Search library by type + ALWAYS shuffle (round-16, CR 701.19) --------

# Per CR 701.19, a tutor-style "search library, then shuffle" effect
# ALWAYS shuffles afterward — whether or not a card was found. This is
# distinct from round-15's `search_library_for_first_of_type_and_move_to_hand`
# (no shuffle) and from round-14's printing-id variant
# `search_library_and_move_to_hand` (shuffles unconditionally too).


def test_search_and_shuffle_finds_creature_and_shuffles(synthetic_state):
    """Library with a creature: search + always-shuffle returns the
    moved card's obj_id, library size shrinks by 1, and the library
    order is changed by the post-search shuffle. Default synthetic
    library is 50 Forests; we load a mixed deck so there's a creature
    to find AND enough cards to make the shuffle observable."""
    # 50 cards: 49 Forests + 1 Bears so after removal there are still
    # 49 cards (>= 2) for Fisher-Yates to do something meaningful.
    deck = [FOREST_PID] * 49 + [BEARS_PID]
    synthetic_state.load_deck(0, deck)
    lib_before = _library_obj_ids(synthetic_state, 0, len(deck))
    hand_before = synthetic_state.hand_sizes
    lib_size_before = synthetic_state.library_sizes

    found = synthetic_state.search_library_for_first_of_type_and_shuffle(
        0, "Creature", 42
    )

    assert found is not None
    assert synthetic_state.printing_name_of(found) == "Grizzly Bears"
    # Library shrank by 1, hand grew by 1.
    assert synthetic_state.library_sizes[0] == lib_size_before[0] - 1
    assert synthetic_state.hand_sizes[0] == hand_before[0] + 1
    # The library was shuffled — its top-first order is no longer the
    # original lib_before minus the moved card. (With 49 cards left
    # after removal, the chance of a non-trivial Fisher-Yates landing
    # on the identity permutation is ~1/49!.)
    lib_after = _library_obj_ids(synthetic_state, 0, len(deck) - 1)
    # The set of remaining ids is (lib_before's ids) - {found}.
    expected_remaining = [i for i in lib_before if i != found]
    assert sorted(lib_after) == sorted(expected_remaining)
    assert lib_after != expected_remaining


def test_search_and_shuffle_no_match_still_shuffles(synthetic_state):
    """No match: returns None, library size UNCHANGED, but the library
    is STILL shuffled (CR 701.19). Default synthetic library is 50
    Forests, so "Planeswalker" finds nothing."""
    lib_before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    hand_before = synthetic_state.hand_sizes
    lib_size_before = synthetic_state.library_sizes

    found = synthetic_state.search_library_for_first_of_type_and_shuffle(
        0, "Planeswalker", 42
    )

    assert found is None
    # Hand and library SIZES unchanged.
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.library_sizes == lib_size_before
    # But library ORDER changed — the shuffle ran even on a miss.
    lib_after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert sorted(lib_after) == sorted(lib_before)
    assert lib_after != lib_before


def test_search_and_shuffle_empty_library_returns_none(synthetic_state):
    """Empty library: returns None, no error, no mutation. Shuffle on
    a size-0 library is a no-op."""
    synthetic_state.load_deck(0, [])
    assert synthetic_state.library_sizes[0] == 0

    found = synthetic_state.search_library_for_first_of_type_and_shuffle(
        0, "Creature", 42
    )

    assert found is None
    assert synthetic_state.library_sizes[0] == 0


def test_search_and_shuffle_unknown_player_raises(synthetic_state):
    """Out-of-range player_id raises (matches the search-only
    sibling's loud-failure convention)."""
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.search_library_for_first_of_type_and_shuffle(
            7, "Creature", 42
        )


def test_search_and_shuffle_deterministic_with_same_seed(synthetic_state):
    """Same source state cloned twice + same seed produces identical
    results: same moved card AND identical resulting library order."""
    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()

    found_a = a.search_library_for_first_of_type_and_shuffle(0, "Land", 42)
    found_b = b.search_library_for_first_of_type_and_shuffle(0, "Land", 42)

    assert found_a == found_b
    a_order = _library_obj_ids(a, 0, P0_LIBRARY_SIZE - 1)
    b_order = _library_obj_ids(b, 0, P0_LIBRARY_SIZE - 1)
    assert a_order == b_order


def test_search_and_shuffle_different_seed_different_order(synthetic_state):
    """Different seeds applied to identical clones produce different
    library orders. With a 50-card library, false equality is ~1/50!,
    effectively never."""
    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()

    a.search_library_for_first_of_type_and_shuffle(0, "Land", 1)
    b.search_library_for_first_of_type_and_shuffle(0, "Land", 99)

    a_order = _library_obj_ids(a, 0, P0_LIBRARY_SIZE - 1)
    b_order = _library_obj_ids(b, 0, P0_LIBRARY_SIZE - 1)
    assert a_order != b_order


def test_search_and_shuffle_clone_isolation(synthetic_state):
    """CoW canary: search+shuffle on a clone must not leak back into
    the source. Both library order AND hand membership are checked."""
    cloned = synthetic_state.clone_once()
    source_lib_before = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    source_hand_before = synthetic_state.hand_sizes
    source_lib_size_before = synthetic_state.library_sizes

    found = cloned.search_library_for_first_of_type_and_shuffle(
        0, "Land", 42
    )
    assert found is not None

    # Source untouched.
    assert synthetic_state.hand_sizes == source_hand_before
    assert synthetic_state.library_sizes == source_lib_size_before
    source_lib_after = _library_obj_ids(synthetic_state, 0, P0_LIBRARY_SIZE)
    assert source_lib_after == source_lib_before
