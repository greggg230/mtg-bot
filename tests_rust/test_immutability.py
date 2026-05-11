"""Comprehensive CoW (copy-on-write) isolation canaries for the Rust
``GameState``.

Each existing edge-case slice in ``tests_rust/`` already carries one or
two CoW canaries beside the primitive it owns
(``test_tap.py::test_untap_all_for_player_clone_isolation``,
``test_damage.py::test_deal_damage_on_clone_does_not_affect_original``,
``test_zones.py::test_clone_then_move_does_not_affect_original``, ...).
This module consolidates and extends them so the Arc-backed zone-
storage invariant is exercised through ALL state-mutating primitives
in a single place — when someone adds a new mutator and forgets to
run it through ``Arc::make_mut`` on the first write, ONE of the
canaries here will fire instead of leaking silently into
``parity_smoke``'s per-decision state hashes.

**Pattern.** Every test in this module follows the same shape:

1. Build / mutate the source state into a known shape.
2. Snapshot the source's fingerprint AND the field(s) the primitive
   under test will touch (``get_damage`` / ``is_tapped`` /
   ``find_object_zone`` / ``hand_sizes`` / ``get_life`` /
   ``get_mana_pool`` / ``get_priority_player`` / ``current_phase``).
3. ``clone = state.clone_once()`` — the same clone MCTS / data_gen use
   on every node expansion.
4. Mutate ONLY the clone via the primitive under test.
5. Assert the clone observed the mutation.
6. Assert the source's fingerprint AND each tracked field is bit-
   identical to the pre-clone snapshot — the Arc on the source's zone
   was not retroactively patched.

For the per-Vec-in-Arc primitives (zones, battlefield-object fields),
the load-bearing assertion is that ``Arc::make_mut`` on the clone's
zone allocated a fresh Vec rather than mutating the shared inner
storage. For the by-value primitives (priority, phase, life, mana
pool), the equivalent assertion is that the clone receives a
``Copy``-style by-value snapshot of the field.

**Two-clone shape.** A subset of the tests also exercise the two-
clone case: ``cloneA = state.clone_once()``, ``cloneB =
state.clone_once()``. Mutating ``cloneA`` must leave both
``cloneB`` AND the original untouched. This catches a particular
class of bug where ``Arc::make_mut`` on cloneA accidentally
patches the storage that cloneB still points at via the same Arc.

**Nested-clone shape.** ``state.clone_once().clone_once()`` — chain
depth doesn't break the CoW. Two bonus tests at the bottom exercise
nested clones to pin that deep-cloning doesn't share a state buffer
all the way through the chain.

The primitives canaried here, mirrored against the slice that
introduced each one:

1. ``deal_damage``                — ``test_damage.py``
2. ``clear_damage``               — ``test_damage.py``
3. ``tap_object`` / ``untap_object``    — ``test_tap.py``
4. ``untap_all_for_player``       — ``test_tap.py``
5. ``move_object``                — ``test_zones.py``
6. ``draw_card``                  — ``test_zones.py`` / ``test_library.py``
7. ``mill_target_player``         — ``test_library.py``
8. ``add_mana``                   — ``test_mana.py``
9. ``pass_priority``              — ``test_priority.py``
10. ``advance_phase`` (full lap)  — ``test_steps.py``
11. ``apply_sbas_until_quiescent`` — ``test_sba.py``
12. ``apply_action({"kind": "pass_priority"})`` / ``play_land``
                                  — ``test_actions.py``
"""

from __future__ import annotations


# Zone constants — must match the values in
# ``rust/src/state_ops_zones.rs``. Hard-coded for self-containment;
# mirror of the same block in ``test_zones.py``.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3
ZONE_STACK = 4
ZONE_EXILE = 5


# Synthetic-state id layout: ``make_synthetic_state(10, 7, 50)`` lays
# player 0 out as battlefield 0..9, hand 10..16, library 17..66 and
# player 1's range starts at 67. Mirror of the same block in
# ``test_damage.py`` and ``test_zones.py``.
P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17
P0_FIRST_BEAR_ID = 5  # 5..9 are 2/2 Grizzly Bears on p0's battlefield
P1_FIRST_BATTLEFIELD_ID = 67


# -----------------------------------------------------------------
# 1. deal_damage — creature damage stamp
# -----------------------------------------------------------------


def test_cow_deal_damage_isolates_clone_from_original(synthetic_state):
    """``deal_damage`` mutates the per-object ``damage`` field on a
    GameObject living in ``Arc<Vec<GameObject>>``. The clone must
    observe its mutation while the original's matching object is
    completely untouched — same fingerprint, same per-object damage."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    fp_before = synthetic_state.fingerprint()
    assert synthetic_state.get_damage(obj_id) == 0

    clone = synthetic_state.clone_once()
    assert clone.deal_damage(obj_id, 4) is True

    # Clone observed the mutation.
    assert clone.get_damage(obj_id) == 4
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.get_damage(obj_id) == 0
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: deal_damage on clone leaked into original "
        "(fingerprint diverged)"
    )

    # Mutating both clone-then-original separately produces different
    # fingerprints (clone diverges from original after independent
    # mutations).
    synthetic_state.deal_damage(obj_id, 1)
    assert synthetic_state.fingerprint() != clone.fingerprint()


def test_cow_deal_damage_two_clone_isolation(synthetic_state):
    """Two-clone canary: cloneA and cloneB share the original's Arc'd
    battlefield Vec at construction time. A ``deal_damage`` on cloneA
    must allocate a fresh Vec (via ``Arc::make_mut``) — cloneB still
    points at the original Arc, so its view must be unchanged."""
    obj_id = P0_FIRST_BEAR_ID
    clone_a = synthetic_state.clone_once()
    clone_b = synthetic_state.clone_once()
    fp_b_before = clone_b.fingerprint()
    fp_orig_before = synthetic_state.fingerprint()

    clone_a.deal_damage(obj_id, 7)

    assert clone_a.get_damage(obj_id) == 7
    # cloneB unchanged.
    assert clone_b.get_damage(obj_id) == 0
    assert clone_b.fingerprint() == fp_b_before
    # Original unchanged.
    assert synthetic_state.get_damage(obj_id) == 0
    assert synthetic_state.fingerprint() == fp_orig_before


# -----------------------------------------------------------------
# 2. clear_damage — cleanup-style wipe
# -----------------------------------------------------------------


def test_cow_clear_damage_isolates_clone_from_original(synthetic_state):
    """``clear_damage`` is the end-of-turn cleanup-step primitive
    (CR 514.2). Setting up damage on the source, cloning, then
    ``clear_damage`` on the clone must leave the source's accumulated
    damage intact."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(obj_id, 3)
    assert synthetic_state.get_damage(obj_id) == 3
    fp_before = synthetic_state.fingerprint()

    clone = synthetic_state.clone_once()
    assert clone.clear_damage(obj_id) is True

    # Clone wiped.
    assert clone.get_damage(obj_id) == 0
    # Original keeps the damage.
    assert synthetic_state.get_damage(obj_id) == 3
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: clear_damage on clone leaked into original"
    )
    # Independent mutation paths produce divergent fingerprints.
    synthetic_state.clear_damage(obj_id)
    assert synthetic_state.get_damage(obj_id) == 0
    # Now both are 0; fingerprints converge again.
    assert synthetic_state.fingerprint() == clone.fingerprint()


# -----------------------------------------------------------------
# 3. tap_object / untap_object — per-permanent flag
# -----------------------------------------------------------------


def test_cow_tap_and_untap_isolates_clone_from_original(synthetic_state):
    """``tap_object`` and ``untap_object`` flip the ``tapped`` bit on a
    GameObject inside ``Arc<Vec<GameObject>>``. Each direction must
    isolate via ``Arc::make_mut``. Round-trip the clone: tap it, then
    untap it — the original's tap state must stay False across both
    legs."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    fp_before = synthetic_state.fingerprint()
    assert synthetic_state.is_tapped(obj_id) is False

    clone = synthetic_state.clone_once()
    assert clone.tap_object(obj_id) is True
    assert clone.is_tapped(obj_id) is True
    assert synthetic_state.is_tapped(obj_id) is False, (
        "CoW broken: tap_object on clone leaked into original"
    )
    assert synthetic_state.fingerprint() == fp_before

    # Untap the clone — original still untouched (and still untapped).
    assert clone.untap_object(obj_id) is True
    assert clone.is_tapped(obj_id) is False
    assert synthetic_state.is_tapped(obj_id) is False
    assert synthetic_state.fingerprint() == fp_before

    # Mutating both clone-then-original independently keeps fingerprints
    # divergent: tap clone, untap nothing on original — they differ.
    clone.tap_object(obj_id)
    assert synthetic_state.fingerprint() != clone.fingerprint()


def test_cow_tap_two_clone_isolation(synthetic_state):
    """Two-clone canary for tap_object. Tap on cloneA must leave
    cloneB unchanged — same Arc-make_mut allocation must not patch
    the shared inner Vec."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    clone_a = synthetic_state.clone_once()
    clone_b = synthetic_state.clone_once()
    fp_b_before = clone_b.fingerprint()

    clone_a.tap_object(obj_id)

    assert clone_a.is_tapped(obj_id) is True
    assert clone_b.is_tapped(obj_id) is False
    assert clone_b.fingerprint() == fp_b_before
    assert synthetic_state.is_tapped(obj_id) is False


# -----------------------------------------------------------------
# 4. untap_all_for_player — bulk untap (untap-step primitive)
# -----------------------------------------------------------------


def test_cow_untap_all_for_player_isolates_clone_from_original(synthetic_state):
    """The bulk-untap path used by the untap step. Tap every p0
    permanent on the source, clone, then ``untap_all_for_player(0)``
    on the clone. The clone must come up with everything untapped
    while the original keeps every flag tapped."""
    p0_ids = list(range(0, 10))
    for obj_id in p0_ids:
        synthetic_state.tap_object(obj_id)
    fp_before = synthetic_state.fingerprint()
    # Sanity: every p0 permanent is tapped.
    for obj_id in p0_ids:
        assert synthetic_state.is_tapped(obj_id) is True

    clone = synthetic_state.clone_once()
    clone.untap_all_for_player(0)

    # Clone got the bulk untap.
    for obj_id in p0_ids:
        assert clone.is_tapped(obj_id) is False
    # Original unchanged.
    for obj_id in p0_ids:
        assert synthetic_state.is_tapped(obj_id) is True
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: untap_all_for_player on clone leaked into original"
    )


# -----------------------------------------------------------------
# 5. move_object — zone change (the heaviest CoW write surface)
# -----------------------------------------------------------------


def test_cow_move_object_isolates_clone_from_original(synthetic_state):
    """``move_object`` is the heaviest CoW write surface — it mutates
    TWO ``Arc<Vec<GameObject>>``s (source zone + destination zone) AND
    the moved object's ``zone`` tag. Moving on the clone must leave
    BOTH source and dest zones on the original untouched.

    This is the parent canary for every effect that resolves into a
    zone change (destroy, exile, bounce, mill, reanimate, ...). If
    this one breaks, the entire effect pipeline silently corrupts
    the parent state during MCTS rollouts."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    fp_before = synthetic_state.fingerprint()
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    clone = synthetic_state.clone_once()
    assert clone.move_object(obj_id, ZONE_GRAVEYARD, 0) is True

    # Clone diverged.
    assert clone.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert clone.battlefield_sizes[0] == bf_before[0] - 1
    assert clone.graveyard_sizes[0] == gy_before[0] + 1
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: move_object on clone leaked into original"
    )


def test_cow_move_object_two_clone_isolation(synthetic_state):
    """Two-clone canary for move_object — the heaviest write surface
    (two zones touched). cloneA destroys a permanent; cloneB still
    sees it on the battlefield."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    clone_a = synthetic_state.clone_once()
    clone_b = synthetic_state.clone_once()
    fp_b_before = clone_b.fingerprint()
    fp_orig_before = synthetic_state.fingerprint()

    assert clone_a.move_object(obj_id, ZONE_GRAVEYARD, 0) is True

    assert clone_a.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert clone_b.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert clone_b.fingerprint() == fp_b_before
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.fingerprint() == fp_orig_before


# -----------------------------------------------------------------
# 6. draw_card — library -> hand (turn-based action)
# -----------------------------------------------------------------


def test_cow_draw_card_isolates_clone_from_original(synthetic_state):
    """``draw_card`` mutates BOTH the library Arc'd Vec (pop from tail)
    AND the hand Arc'd Vec (push), and updates the drawn object's
    ``zone`` tag. Drawing on the clone must leave the original's
    library size + hand size unchanged."""
    fp_before = synthetic_state.fingerprint()
    lib_before = synthetic_state.library_sizes
    hand_before = synthetic_state.hand_sizes

    clone = synthetic_state.clone_once()
    drawn = clone.draw_card(0)
    assert drawn is not None

    # Clone shrank library by 1, grew hand by 1.
    assert clone.library_sizes[0] == lib_before[0] - 1
    assert clone.hand_sizes[0] == hand_before[0] + 1
    assert clone.find_object_zone(drawn) == ZONE_HAND
    # Original sees the card still in the library, hand unchanged.
    assert synthetic_state.find_object_zone(drawn) == ZONE_LIBRARY
    assert synthetic_state.library_sizes == lib_before
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: draw_card on clone leaked into original"
    )


# -----------------------------------------------------------------
# 7. mill_target_player — top-of-library bulk move to graveyard
# -----------------------------------------------------------------


def test_cow_mill_target_player_isolates_clone_from_original(synthetic_state):
    """``mill_target_player`` is a bulk variant of move_object — it
    pops N cards from the library tail and pushes them onto the
    graveyard, mutating two Arc'd Vecs and N ``zone`` tags. Milling
    on the clone must leave the original's library + graveyard
    untouched."""
    fp_before = synthetic_state.fingerprint()
    lib_before = synthetic_state.library_sizes
    gy_before = synthetic_state.graveyard_sizes

    clone = synthetic_state.clone_once()
    milled = clone.mill_target_player(0, 5)
    assert milled == 5

    # Clone got the mill.
    assert clone.library_sizes[0] == lib_before[0] - 5
    assert clone.graveyard_sizes[0] == gy_before[0] + 5
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.library_sizes == lib_before
    assert synthetic_state.graveyard_sizes == gy_before
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: mill_target_player on clone leaked into original"
    )


# -----------------------------------------------------------------
# 8. add_mana — mana pool primitive (by-value Player struct)
# -----------------------------------------------------------------


def test_cow_add_mana_isolates_clone_from_original(synthetic_state):
    """``add_mana`` mutates ``Player.mana_pool``. ``Player`` is stored
    by value in ``GameState.players: [Player; 2]`` (NOT behind an Arc),
    so a clone deep-copies each Player struct — there's no Arc CoW
    involved here, but the test still pins the by-value isolation
    semantics so a future slice that moves Player into an Arc-backed
    container doesn't silently leak mutations back."""
    fp_before = synthetic_state.fingerprint()
    pool_before_p0 = synthetic_state.get_mana_pool(0)
    pool_before_p1 = synthetic_state.get_mana_pool(1)

    clone = synthetic_state.clone_once()
    clone.add_mana(0, "U", 3)
    clone.add_mana(1, "B", 2)

    # Clone got the mana.
    assert clone.get_mana_pool(0)["U"] == 3
    assert clone.get_mana_pool(1)["B"] == 2
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.get_mana_pool(0) == pool_before_p0
    assert synthetic_state.get_mana_pool(1) == pool_before_p1
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: add_mana on clone leaked into original"
    )


# -----------------------------------------------------------------
# 9. pass_priority — u8 field toggle
# -----------------------------------------------------------------


def test_cow_pass_priority_isolates_clone_from_original(synthetic_state):
    """``pass_priority`` flips the ``priority_player`` u8 on
    GameState. By-value scalar — no Arc involved — but the
    isolation contract must still hold. Mutating priority on a
    clone must not retroactively touch the original's priority
    field."""
    fp_before = synthetic_state.fingerprint()
    assert synthetic_state.get_priority_player() == 0

    clone = synthetic_state.clone_once()
    new_priority = clone.pass_priority()
    assert new_priority == 1

    assert clone.get_priority_player() == 1
    assert synthetic_state.get_priority_player() == 0
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: pass_priority on clone leaked into original"
    )

    # Independent mutations diverge fingerprints.
    synthetic_state.pass_priority()
    # Now both have priority=1 — fingerprints converge again.
    assert synthetic_state.get_priority_player() == 1
    assert synthetic_state.fingerprint() == clone.fingerprint()


# -----------------------------------------------------------------
# 10. advance_phase — full lap through all 12 steps
# -----------------------------------------------------------------


def test_cow_advance_phase_full_lap_isolates_clone_from_original(synthetic_state):
    """``advance_phase`` ticks the phase/step counter AND (on
    CLEANUP -> UNTAP) swaps active_player AND fires the per-controller
    untap-step bulk untap on the new active player. A full lap (12
    ticks) thus touches phase + active_player + battlefield Arc on
    BOTH sides (via the wrap untap).

    The combined side effect makes this a maximal canary: if ANY of
    the four touched fields (phase, active_player, turn_number, the
    untapped permanents' tapped flag) leaks across the clone boundary,
    the assertion bundle below fires."""
    # Tap a p1 permanent so the untap-step composition on the wrap is
    # observable. (synthetic_state seeds active_player=0; after 12
    # advances we wrap CLEANUP -> UNTAP, swap to active_player=1, and
    # the untap-step fires on p1's permanents.)
    p1_obj = P1_FIRST_BATTLEFIELD_ID
    synthetic_state.tap_object(p1_obj)
    assert synthetic_state.is_tapped(p1_obj) is True

    fp_before = synthetic_state.fingerprint()
    phase_before = synthetic_state.current_phase()
    active_before = synthetic_state.active_player
    turn_before = synthetic_state.turn_number

    clone = synthetic_state.clone_once()
    # Full lap: 12 advances brings us back to UNTAP on the next turn.
    for _ in range(12):
        clone.advance_phase()

    # Clone walked a full lap: same phase as start (UNTAP), but
    # active_player swapped and turn_number incremented; the p1
    # permanent on the clone got auto-untapped during the wrap.
    assert clone.current_phase() == phase_before
    assert clone.active_player == 1 - active_before
    assert clone.turn_number == turn_before + 1
    assert clone.is_tapped(p1_obj) is False

    # Original is bit-identical to the pre-clone snapshot — phase
    # didn't tick, active_player didn't swap, turn didn't bump, and
    # the p1 permanent is still tapped.
    assert synthetic_state.current_phase() == phase_before
    assert synthetic_state.active_player == active_before
    assert synthetic_state.turn_number == turn_before
    assert synthetic_state.is_tapped(p1_obj) is True
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: advance_phase full-lap on clone leaked into original"
    )


# -----------------------------------------------------------------
# 11. apply_sbas_until_quiescent — state-based actions loop
# -----------------------------------------------------------------


def test_cow_apply_sbas_isolates_clone_from_original(synthetic_state):
    """``apply_sbas_until_quiescent`` loops the lethal-damage /
    zero-toughness SBAs until no further mutation. On lethal damage,
    the dying creature moves battlefield -> graveyard (a move_object
    chain inside the loop). The clone gets the destruction; the
    original keeps the creature on the battlefield."""
    obj_id = P0_FIRST_BEAR_ID  # 2/2 Grizzly Bears
    # Stamp lethal damage on the source so the clone enters the loop.
    synthetic_state.deal_damage(obj_id, 2)
    fp_before = synthetic_state.fingerprint()
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD

    clone = synthetic_state.clone_once()
    n = clone.apply_sbas_until_quiescent()
    assert n == 1  # one SBA fired (lethal damage destruction)

    # Clone moved the Bear into the graveyard.
    assert clone.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert clone.battlefield_sizes[0] == bf_before[0] - 1
    assert clone.graveyard_sizes[0] == gy_before[0] + 1
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: apply_sbas_until_quiescent on clone leaked into original"
    )


# -----------------------------------------------------------------
# 12. apply_action — Action bridge
#
# Two coverage points: PassPriority (trivial scalar flip via the
# bridge) and PlayLand (zone move via the bridge — hand -> battlefield).
# -----------------------------------------------------------------


def test_cow_apply_action_pass_priority_isolates_clone_from_original(synthetic_state):
    """``apply_action({"kind": "pass_priority"})`` routes through the
    Action enum dispatcher to ``pass_priority``. Mutating the clone via
    the bridge must NOT bleed into the original's priority — same
    CoW contract as the raw primitive, but exercised through the
    action-encoding boundary."""
    fp_before = synthetic_state.fingerprint()
    assert synthetic_state.get_priority_player() == 0

    clone = synthetic_state.clone_once()
    clone.apply_action({"kind": "pass_priority"})

    assert clone.get_priority_player() == 1
    assert synthetic_state.get_priority_player() == 0
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: apply_action(pass_priority) on clone leaked into original"
    )


def test_cow_apply_action_play_land_isolates_clone_from_original(synthetic_state):
    """``apply_action({"kind": "play_land", ...})`` routes through the
    bridge to ``play_land(card_id, priority_player)`` which is a
    zone move (hand -> battlefield) under the hood. The clone's
    battlefield grows by 1 and hand shrinks by 1; the original is
    bit-identical to the pre-clone snapshot.

    Synthetic seed has only Llanowar Elves in p0's hand; the slice
    helper pulls a Forest in from the library first, then plays it."""
    # Pull a Forest from p0's library into p0's hand on the SOURCE so
    # the precondition is encoded in the original — then clone and
    # play. We must do this BEFORE cloning so the Forest exists in
    # both the original AND the clone, and the play_land on the clone
    # exercises a hand->battlefield move that the original can compare
    # against.
    forest_id = P0_FIRST_LIBRARY_ID
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    assert synthetic_state.move_object(forest_id, ZONE_HAND, 0) is True
    fp_before = synthetic_state.fingerprint()
    hand_before = synthetic_state.hand_sizes
    bf_before = synthetic_state.battlefield_sizes
    assert synthetic_state.find_object_zone(forest_id) == ZONE_HAND

    clone = synthetic_state.clone_once()
    clone.apply_action({"kind": "play_land", "card_id": forest_id})

    # Clone played the land.
    assert clone.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert clone.hand_sizes[0] == hand_before[0] - 1
    assert clone.battlefield_sizes[0] == bf_before[0] + 1
    # Original is bit-identical to the pre-clone snapshot.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_HAND
    assert synthetic_state.hand_sizes == hand_before
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.fingerprint() == fp_before, (
        "CoW broken: apply_action(play_land) on clone leaked into original"
    )


# -----------------------------------------------------------------
# Bonus: nested-clone shape — chain depth doesn't break CoW.
#
# ``state.clone_once().clone_once()`` is a real shape in the MCTS
# code path — a child node may itself spawn children. The CoW
# invariant must compose: a mutation on the inner-most clone must
# not bleed into the intermediate clone OR the original.
# -----------------------------------------------------------------


def test_cow_nested_clone_zone_mutation_isolation(synthetic_state):
    """``state.clone_once().clone_once()`` chain depth doesn't break
    the CoW. Mutating a zone on the inner-most clone (clone_of_clone)
    must leave BOTH the intermediate clone AND the original untouched.

    Exercises a worst-case zone-move on the chain depth: the inner
    clone destroys a permanent (battlefield -> graveyard, two Arc'd
    Vecs touched)."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    fp_orig_before = synthetic_state.fingerprint()
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    middle = synthetic_state.clone_once()
    fp_middle_before = middle.fingerprint()
    inner = middle.clone_once()

    # Mutate ONLY the inner-most clone.
    assert inner.move_object(obj_id, ZONE_GRAVEYARD, 0) is True

    # Inner observed the mutation.
    assert inner.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert inner.battlefield_sizes[0] == bf_before[0] - 1
    assert inner.graveyard_sizes[0] == gy_before[0] + 1
    # Middle clone is bit-identical to its pre-mutation snapshot.
    assert middle.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert middle.battlefield_sizes == bf_before
    assert middle.graveyard_sizes == gy_before
    assert middle.fingerprint() == fp_middle_before, (
        "CoW broken: inner-clone mutation leaked into intermediate clone"
    )
    # Original is bit-identical to its pre-clone snapshot.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before
    assert synthetic_state.fingerprint() == fp_orig_before, (
        "CoW broken: inner-clone mutation leaked into original"
    )


def test_cow_nested_clone_mixed_primitive_isolation(synthetic_state):
    """Same nested-clone shape, but each level mutates a DIFFERENT
    primitive. Pins that the CoW boundary holds across heterogeneous
    mutations stacked along the clone chain — not just the easy case
    of "every level touches the same field".

    Layout:
      - inner: deal_damage on a creature (battlefield Arc<Vec> CoW)
      - middle: tap_object on a different permanent (battlefield Arc<Vec> CoW)
      - original: add_mana (by-value Player struct mutation)

    Each layer must see ONLY its own mutation; nothing else."""
    bear_id = P0_FIRST_BEAR_ID
    forest_id = P0_FIRST_BATTLEFIELD_ID

    middle = synthetic_state.clone_once()
    inner = middle.clone_once()

    # Apply distinct mutations at each level.
    inner.deal_damage(bear_id, 5)
    middle.tap_object(forest_id)
    synthetic_state.add_mana(0, "R", 2)

    # Inner: damage stamped, no tap on Forest, no R mana.
    assert inner.get_damage(bear_id) == 5
    assert inner.is_tapped(forest_id) is False
    assert inner.get_mana_pool(0)["R"] == 0

    # Middle: Forest tapped, no damage on Bear, no R mana.
    assert middle.get_damage(bear_id) == 0
    assert middle.is_tapped(forest_id) is True
    assert middle.get_mana_pool(0)["R"] == 0

    # Original: 2 R in pool, Bear unmarked, Forest untapped.
    assert synthetic_state.get_damage(bear_id) == 0
    assert synthetic_state.is_tapped(forest_id) is False
    assert synthetic_state.get_mana_pool(0)["R"] == 2

    # All three fingerprints are pairwise distinct (every level made a
    # genuine state-affecting mutation along a different primitive).
    fps = {
        synthetic_state.fingerprint(),
        middle.fingerprint(),
        inner.fingerprint(),
    }
    assert len(fps) == 3, (
        "CoW broken: nested clones with distinct mutations should "
        "produce three distinct fingerprints, got "
        f"{len(fps)} unique"
    )
