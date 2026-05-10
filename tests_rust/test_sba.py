"""State-based action (SBA) loop primitives on the Rust ``GameState``.

Ported from ``src/mtg_bot/turn.py::state_based_actions`` /
``_state_based_actions_pass`` and ``CR 704.5`` audit numbering.

**Slice scope: lethal-damage (CR 704.5g) AND 0-toughness (CR 704.5f) SBAs.**

- CR 704.5g — a creature whose marked damage is >= its toughness is
  destroyed (moved to its controller's graveyard).
- CR 704.5f — a creature with toughness 0 (or less) is put into its
  owner's graveyard, regardless of marked damage.

This slice exposes ``apply_sbas() -> int`` (count of destructions
this pass) plus the underlying ``is_lethal(obj_id) -> Option<bool>``
helper. ``apply_sbas`` applies 704.5f BEFORE 704.5g per CR rule order;
in practice the same set of victims gets destroyed but the ordering
is preserved for niche edge cases.

Effective toughness for the 0-toughness check is computed as
``printing.toughness + obj.counters`` (round-3's net-i32 +1/+1 vs
-1/-1 model). A creature with base toughness 1 and a -1/-1 counter
(counters = -1) has effective toughness 0 and dies.

**Out of scope, deferred to future slices:**

- **SBA loop driver / quiescence.** Real MTG checks SBAs simultaneously
  and re-checks until stable. This slice does ONE pass; iteration to
  fixed point is a future slice that integrates apply_sbas into the
  combat / spell-resolution call sites.
- **Order independence / simultaneous destruction.** This slice
  collects all victims first, then moves them — so two creatures both
  past lethal both die in the same call (test below). Real MTG
  semantics around layered toughness modifications + cancellation
  (704.5r) are deferred.
- **Marked damage clearing.** Damage clears at end of turn (CR 514.2),
  NOT on every SBA pass. apply_sbas does NOT zero damage on surviving
  creatures (verified below). For destroyed creatures the damage on
  the now-graveyarded GameObject is left as-is (zone change makes it
  irrelevant; CR 614.12 says marked damage is removed on zone change
  but for this slice we don't bother explicitly zeroing it).
- **Indestructibility / regeneration / planeswalker loyalty / legend
  rule / +1/+1 vs -1/-1 cancellation / saga age / aura on illegal
  target / etc.** All other SBAs from ``_state_based_actions_pass``
  belong to subsequent slices.
- **Battlefield-only.** apply_sbas only inspects battlefield zones;
  damage on non-battlefield objects is ignored (it shouldn't exist
  there in normal play, and the lethal-damage SBA is intrinsically
  battlefield-scoped).

**Composes with prior slices:**
- ``state_ops_damage`` (deal_damage to set up lethal scenarios).
- ``CardPrinting.toughness: Option<i16>`` from the bridge-ext slice.
- ``state_ops_zones::move_object`` for graveyard movement.
"""

from __future__ import annotations

import pytest

# `make_synthetic_state(num_battlefield=10, num_hand=7, num_library=50)`
# lays player 0 out as:
#   battlefield: ids 0..9 (split: ids 0..4 are Forests, 5..9 are
#                          Grizzly Bears — see make_player in state.rs)
#   hand:        ids 10..16 (Llanowar Elves)
#   library:     ids 17..66 (Forests)
# Player 1's range starts at 67 with the same layout.
P0_FIRST_BEAR_ID = 5            # First Grizzly Bears (2/2) on p0 battlefield
P0_SECOND_BEAR_ID = 6
P0_FIRST_FOREST_ID = 0          # First Forest on p0 battlefield (no toughness)
P1_FIRST_BEAR_ID = 67 + 5       # First Bear on p1 battlefield


def test_apply_sbas_no_op_on_clean_state(synthetic_state):
    """Fresh state, no creatures damaged. apply_sbas returns 0
    (no destructions) and battlefield sizes are unchanged."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes
    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before


def test_apply_sbas_returns_count_of_destructions(synthetic_state):
    """Damage Grizzly Bears (toughness 2) for 2; apply_sbas returns 1
    indicating one creature was destroyed by the lethal-damage SBA."""
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    assert synthetic_state.apply_sbas() == 1


def test_apply_sbas_moves_lethal_creature_to_graveyard(synthetic_state):
    """Damage a 2/2 for 2. apply_sbas moves the dead creature off the
    battlefield and into its controller's graveyard."""
    obj_id = P0_FIRST_BEAR_ID
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    synthetic_state.deal_damage(obj_id, 2)
    assert synthetic_state.apply_sbas() == 1

    # Battlefield shrinks by one, graveyard grows by one — for player 0.
    assert synthetic_state.battlefield_sizes[0] == bf_before[0] - 1
    assert synthetic_state.graveyard_sizes[0] == gy_before[0] + 1
    # Player 1 untouched.
    assert synthetic_state.battlefield_sizes[1] == bf_before[1]
    assert synthetic_state.graveyard_sizes[1] == gy_before[1]
    # The find_object_zone helper reports the object as graveyarded now.
    # ZONE_GRAVEYARD = 3 (see state_ops_zones.rs).
    assert synthetic_state.find_object_zone(obj_id) == 3


def test_apply_sbas_does_not_clear_non_lethal_damage(synthetic_state):
    """Damage a 2/2 for 1 (non-lethal). apply_sbas leaves the creature
    on the battlefield AND does NOT zero its marked damage. Damage
    clears at end of turn (CR 514.2), not on every SBA pass."""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.deal_damage(obj_id, 1)
    assert synthetic_state.apply_sbas() == 0
    # Still on battlefield.
    assert synthetic_state.find_object_zone(obj_id) == 2
    # Damage marker preserved (CR 514.2 — clears at end-of-turn cleanup).
    assert synthetic_state.get_damage(obj_id) == 1


def test_apply_sbas_lethal_creature_damage_post_destruction(synthetic_state):
    """When a lethally-damaged creature is moved to graveyard the
    damage on the destroyed GameObject can either be reset to 0 or
    left as-is — this slice picks "leave as-is" for simplicity (the
    object has left the battlefield so its damage marker is not
    consulted by anything). Document via assertion: the object exists
    in the graveyard after destruction; we don't assert the damage
    value. (CR 614.12 says marked damage is removed on zone change,
    but for SBA scope we treat the marker as ignored once off-
    battlefield.)"""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.deal_damage(obj_id, 5)  # overkill
    assert synthetic_state.apply_sbas() == 1
    # Object is in graveyard (zone 3).
    assert synthetic_state.find_object_zone(obj_id) == 3


def test_apply_sbas_handles_multiple_lethal_creatures(synthetic_state):
    """Damage two creatures lethally; apply_sbas returns 2. Both move
    to graveyard in the same call (collect-then-mutate pattern)."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    synthetic_state.deal_damage(P0_SECOND_BEAR_ID, 2)

    assert synthetic_state.apply_sbas() == 2
    # Both moved.
    assert synthetic_state.find_object_zone(P0_FIRST_BEAR_ID) == 3
    assert synthetic_state.find_object_zone(P0_SECOND_BEAR_ID) == 3
    # Battlefield -2, graveyard +2.
    assert synthetic_state.battlefield_sizes[0] == bf_before[0] - 2
    assert synthetic_state.graveyard_sizes[0] == gy_before[0] + 2


def test_apply_sbas_only_affects_creatures_not_lands(synthetic_state):
    """A Forest (no toughness — non-creature land) damaged for 99 is
    NOT destroyed by lethal-damage SBA. Forest stays on battlefield;
    apply_sbas returns 0. The check is gated on
    ``CardPrinting.toughness.is_some()``."""
    forest_id = P0_FIRST_FOREST_ID
    bf_before = synthetic_state.battlefield_sizes

    # Forest has no toughness; deal_damage still works (it's a raw
    # marker on GameObject), but the SBA must not destroy it.
    synthetic_state.deal_damage(forest_id, 99)

    assert synthetic_state.apply_sbas() == 0
    # Forest still on battlefield (zone 2).
    assert synthetic_state.find_object_zone(forest_id) == 2
    assert synthetic_state.battlefield_sizes == bf_before


def test_apply_sbas_creature_with_more_damage_than_toughness_still_destroyed(
    synthetic_state,
):
    """Overkill is fine — a 2/2 with 99 damage marked is still
    destroyed (the predicate is damage >= toughness, not damage ==
    toughness)."""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.deal_damage(obj_id, 99)
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == 3


def test_apply_sbas_lethal_creature_on_p1_goes_to_p1_graveyard(
    synthetic_state,
):
    """Symmetric: lethal damage on a player-1 creature moves it to
    player 1's graveyard, not player 0's. The destination follows the
    creature's controller."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    synthetic_state.deal_damage(P1_FIRST_BEAR_ID, 2)
    assert synthetic_state.apply_sbas() == 1

    # Player 1 lost a creature, gained a graveyard entry.
    assert synthetic_state.battlefield_sizes[1] == bf_before[1] - 1
    assert synthetic_state.graveyard_sizes[1] == gy_before[1] + 1
    # Player 0 untouched.
    assert synthetic_state.battlefield_sizes[0] == bf_before[0]
    assert synthetic_state.graveyard_sizes[0] == gy_before[0]


def test_is_lethal_helper_creature_at_threshold(synthetic_state):
    """``is_lethal(obj_id)`` returns Some(True) when damage >= toughness,
    Some(False) when below, and None for non-creatures or unknown ids."""
    bear_id = P0_FIRST_BEAR_ID
    forest_id = P0_FIRST_FOREST_ID

    # Bear at 0 damage: not lethal.
    assert synthetic_state.is_lethal(bear_id) is False

    # Bear at 2 damage (= toughness): lethal.
    synthetic_state.deal_damage(bear_id, 2)
    assert synthetic_state.is_lethal(bear_id) is True

    # Forest (no toughness): always None.
    assert synthetic_state.is_lethal(forest_id) is None

    # Unknown id: None.
    assert synthetic_state.is_lethal(999_999) is None


def test_apply_sbas_no_op_returns_zero_repeatedly(synthetic_state):
    """Calling apply_sbas twice on a clean state returns 0 both times.
    Idempotent on a state with no pending destructions."""
    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.apply_sbas() == 0


def test_apply_sbas_after_destruction_returns_zero(synthetic_state):
    """After one apply_sbas destroys a creature, calling it again
    returns 0 — there's nothing left to destroy. (No SBA cascade in
    this slice; second call is a true no-op.)"""
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.apply_sbas() == 0


def test_clone_preserves_sba_pending(synthetic_state):
    """**CoW canary for the SBA path.** Damage a creature on a clone,
    apply_sbas on the clone destroys it, original's battlefield is
    unchanged. Same load-bearing invariant as
    ``test_deal_damage_on_clone_does_not_affect_original`` from the
    damage slice but exercising the destruction path through
    move_object."""
    obj_id = P0_FIRST_BEAR_ID
    bf_before_p0 = synthetic_state.battlefield_sizes[0]
    gy_before_p0 = synthetic_state.graveyard_sizes[0]

    other = synthetic_state.clone_once()
    # Mutate ONLY the clone.
    other.deal_damage(obj_id, 2)
    assert other.apply_sbas() == 1

    # Clone reflects destruction.
    assert other.find_object_zone(obj_id) == 3  # graveyard
    assert other.battlefield_sizes[0] == bf_before_p0 - 1
    assert other.graveyard_sizes[0] == gy_before_p0 + 1

    # Original must NOT see the destruction.
    assert synthetic_state.find_object_zone(obj_id) == 2  # battlefield
    assert synthetic_state.battlefield_sizes[0] == bf_before_p0
    assert synthetic_state.graveyard_sizes[0] == gy_before_p0
    assert synthetic_state.get_damage(obj_id) == 0


# ----------------------------- 0-toughness SBA (CR 704.5f) -----------------
#
# A creature with toughness 0 (or less) is destroyed regardless of marked
# damage. Effective toughness = printed toughness + counters (net-i32
# +1/+1 vs -1/-1 model). A 1/1 with a -1/-1 counter (counters=-1) has
# effective toughness 0 and dies. A 0/0 creature dies on entering a state
# where SBAs are checked.
#
# These tests use ``register_and_inject_object`` from
# ``state_ops_resolve.rs`` to put printings with explicit power/toughness
# onto the battlefield; combined with the ``set_counters`` primitive from
# round-3 that's enough to construct every shape we want to exercise.

ZONE_BATTLEFIELD = 2  # mirrors rust/src/state_ops_zones.rs
ZONE_GRAVEYARD = 3


def _zero_zero_creature_dict():
    """A printing for a 0/0 creature — dies the moment SBAs check it."""
    return {
        "name": "Test 0/0",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 0,
        "toughness": 0,
        "keywords": [],
        "effects": [],
    }


def test_zero_toughness_creature_dies(synthetic_state):
    """A 1/1 creature with a -1/-1 counter (effective toughness 0) is
    destroyed by ``apply_sbas`` — that's CR 704.5f. The damage marker
    is irrelevant; what matters is effective toughness <= 0."""
    obj_id = P0_FIRST_BEAR_ID  # Grizzly Bears, 2/2
    # Drop net counters to -2 so effective toughness = 2 + (-2) = 0.
    synthetic_state.set_counters(obj_id, -2)
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_negative_toughness_via_counters_dies(synthetic_state):
    """Effective toughness can go strictly negative (more -1/-1 than
    base toughness). 704.5f says toughness 0 OR LESS triggers destruction;
    a 2/2 with -3 net counters (effective toughness -1) is destroyed."""
    obj_id = P0_FIRST_BEAR_ID  # 2/2
    synthetic_state.set_counters(obj_id, -3)  # effective toughness -1
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_creature_with_zero_base_toughness_immediately_dies(synthetic_state):
    """Register a 0/0 creature directly to battlefield (no counters,
    no damage). apply_sbas destroys it via 704.5f — base toughness 0 is
    enough on its own."""
    obj_id = synthetic_state.register_and_inject_object(
        _zero_zero_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_apply_sbas_handles_zero_toughness_alongside_lethal_damage(synthetic_state):
    """One creature dies via 704.5f (0-toughness via counters), another
    via 704.5g (lethal damage). Both go to graveyard in the same
    apply_sbas pass; the count is 2."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes

    # First bear: lethal damage (704.5g).
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    # Second bear: 0 effective toughness via counters (704.5f).
    synthetic_state.set_counters(P0_SECOND_BEAR_ID, -2)

    assert synthetic_state.apply_sbas() == 2

    assert synthetic_state.find_object_zone(P0_FIRST_BEAR_ID) == ZONE_GRAVEYARD
    assert synthetic_state.find_object_zone(P0_SECOND_BEAR_ID) == ZONE_GRAVEYARD
    # Both moved off p0's battlefield, both into p0's graveyard.
    assert synthetic_state.battlefield_sizes[0] == bf_before[0] - 2
    assert synthetic_state.graveyard_sizes[0] == gy_before[0] + 2


def test_apply_sbas_returns_correct_count_with_zero_toughness_destructions(
    synthetic_state,
):
    """A pass that destroys only via 704.5f still returns the destruction
    count — the count tallies 0-toughness victims the same as lethal-damage
    victims. Three creatures, all driven to effective toughness 0 by
    counters → return value 3."""
    for obj_id in (P0_FIRST_BEAR_ID, P0_SECOND_BEAR_ID, P1_FIRST_BEAR_ID):
        synthetic_state.set_counters(obj_id, -2)  # 2/2 -> effective 0
    assert synthetic_state.apply_sbas() == 3
    for obj_id in (P0_FIRST_BEAR_ID, P0_SECOND_BEAR_ID, P1_FIRST_BEAR_ID):
        assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_apply_sbas_positive_counters_save_creature_from_lethal(synthetic_state):
    """Anti-canary: the 0-toughness check must NOT fire on creatures
    with positive net effective toughness. A 2/2 with +1 counters has
    effective toughness 3 — undamaged, survives apply_sbas."""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.set_counters(obj_id, 1)
    assert synthetic_state.apply_sbas() == 0
    # Still on battlefield.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


def test_apply_sbas_zero_toughness_does_not_destroy_non_creatures(synthetic_state):
    """A Forest (no toughness) cannot satisfy the 0-toughness check —
    the predicate is gated on ``printing.toughness.is_some()``. Even
    setting a -10 counter on a Forest does not destroy it."""
    forest_id = P0_FIRST_FOREST_ID
    synthetic_state.set_counters(forest_id, -10)
    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD


# ----------------------------- Indestructible (CR 702.12) ------------------
#
# Indestructible creatures don't die from either of the destruction SBAs
# this slice owns:
# - 704.5g (lethal damage): "destroy" effects (and thus the lethal-damage
#   SBA, which destroys) skip indestructible permanents.
# - 704.5f (0-or-negative toughness): a creature whose toughness is 0 or
#   less is put into its owner's graveyard — but per CR 702.12b, that
#   wording reads "is destroyed", so indestructible creatures are
#   exempt here too.
#
# Indestructible creatures CAN still take damage (the damage marker
# updates), they just don't die from it. Damage clears at end of turn
# via the cleanup-step path. They CAN still be exiled (exile is not
# "destroy"); this slice doesn't change ExileTarget behavior.
#
# Effective-toughness tracking still applies — counters update, the
# damage field updates — only the destruction step is skipped.


def _indestructible_2_2_creature_dict():
    """A 2/2 creature printing with the indestructible keyword."""
    return {
        "name": "Test Indestructible 2/2",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2},
        "power": 2,
        "toughness": 2,
        "keywords": ["indestructible"],
        "effects": [],
    }


def _indestructible_1_1_creature_dict():
    """A 1/1 creature printing with the indestructible keyword."""
    return {
        "name": "Test Indestructible 1/1",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "keywords": ["indestructible"],
        "effects": [],
    }


def test_indestructible_creature_with_lethal_damage_survives(synthetic_state):
    """A 2/2 indestructible creature dealt 5 damage is NOT destroyed by
    apply_sbas (CR 702.12 exempts it from 704.5g). The creature stays on
    the battlefield and the damage marker is preserved (damage clears at
    end of turn, not on SBA pass)."""
    obj_id = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.deal_damage(obj_id, 5)  # overkill on a 2/2

    assert synthetic_state.apply_sbas() == 0
    # Still on battlefield.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    # Damage marker preserved.
    assert synthetic_state.get_damage(obj_id) == 5


def test_indestructible_creature_with_zero_toughness_survives(synthetic_state):
    """A 1/1 indestructible creature with a -1/-1 counter (effective
    toughness 0) is NOT destroyed by apply_sbas — CR 702.12b exempts
    indestructible creatures from 704.5f. The creature stays on the
    battlefield with the counter intact."""
    obj_id = synthetic_state.register_and_inject_object(
        _indestructible_1_1_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.set_counters(obj_id, -1)  # effective toughness 0

    assert synthetic_state.apply_sbas() == 0
    # Still on battlefield.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


def test_indestructible_creature_with_negative_toughness_survives(synthetic_state):
    """An indestructible 2/2 with -3 net counters (effective toughness
    -1) still survives. 702.12b exempts indestructible creatures even
    from strictly negative effective toughness."""
    obj_id = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.set_counters(obj_id, -3)  # effective toughness -1

    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


def test_indestructible_count_returns_zero_for_indestructible_creatures(
    synthetic_state,
):
    """apply_sbas returns 0 destructions when only indestructible
    creatures would otherwise die. Set up two indestructibles past
    lethal, both via different rules; count should still be 0."""
    a = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    b = synthetic_state.register_and_inject_object(
        _indestructible_1_1_creature_dict(), 1, ZONE_BATTLEFIELD
    )
    # a: lethal damage path
    synthetic_state.deal_damage(a, 9)
    # b: 0-toughness path
    synthetic_state.set_counters(b, -1)

    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.find_object_zone(a) == ZONE_BATTLEFIELD
    assert synthetic_state.find_object_zone(b) == ZONE_BATTLEFIELD


def test_indestructible_does_not_protect_non_indestructible_neighbor(
    synthetic_state,
):
    """Sanity / mixed-pass: an indestructible creature past lethal AND a
    non-indestructible creature past lethal in the same apply_sbas pass.
    Only the non-indestructible one dies; count is 1."""
    indestructible_id = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.deal_damage(indestructible_id, 5)
    # Vanilla Grizzly Bears (2/2, no keywords) on the synthetic
    # state — kill via lethal damage.
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)

    assert synthetic_state.apply_sbas() == 1
    # Indestructible survives.
    assert synthetic_state.find_object_zone(indestructible_id) == ZONE_BATTLEFIELD
    # Non-indestructible Grizzly Bear is destroyed.
    assert synthetic_state.find_object_zone(P0_FIRST_BEAR_ID) == ZONE_GRAVEYARD


def test_non_indestructible_creature_with_lethal_still_dies(synthetic_state):
    """Anti-canary: the indestructible carve-out must NOT mistakenly
    spare creatures that happen to share the battlefield with an
    indestructible. A vanilla 2/2 still dies from lethal damage when an
    indestructible 2/2 also exists on the board."""
    # Add an indestructible to the board (no damage, healthy).
    indestructible_id = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    # Damage a vanilla Grizzly Bear lethally.
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)

    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(P0_FIRST_BEAR_ID) == ZONE_GRAVEYARD
    # Indestructible was untouched and remains.
    assert synthetic_state.find_object_zone(indestructible_id) == ZONE_BATTLEFIELD


# ----------------------------- SBA loop driver (CR 704.3) ------------------
#
# CR 704.3 — "Whenever a player would get priority [...] the game first
# performs all applicable state-based actions as a single event. [...]
# After SBAs are performed, [...] if any SBAs were performed as a result
# of the check, all applicable SBAs are checked again."
#
# Round-5's apply_sbas is a SINGLE pass. apply_sbas_until_quiescent loops
# until no SBA fires, returning the number of passes that fired. With
# only the destruction SBAs available today (no cascading effects like
# +1/+1 counter on death yet), every realistic scenario completes in
# exactly one pass — but pinning the API now means later slices that
# add cascading effects (sagas advancing on resolve, death triggers,
# etc.) plug into the loop without re-shaping callers.


def test_apply_sbas_until_quiescent_zero_passes_on_clean_state(synthetic_state):
    """No SBAs fire on a clean state -> returns 0 passes. Battlefields
    and graveyards unchanged."""
    bf_before = synthetic_state.battlefield_sizes
    gy_before = synthetic_state.graveyard_sizes
    assert synthetic_state.apply_sbas_until_quiescent() == 0
    assert synthetic_state.battlefield_sizes == bf_before
    assert synthetic_state.graveyard_sizes == gy_before


def test_apply_sbas_until_quiescent_one_pass_kills_lethal(synthetic_state):
    """One lethal-damage creature -> first pass destroys it, second pass
    is a no-op so loop exits. Returns 1 (one pass fired SBAs)."""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.deal_damage(obj_id, 2)
    assert synthetic_state.apply_sbas_until_quiescent() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_apply_sbas_until_quiescent_one_pass_kills_multiple(synthetic_state):
    """Three lethal-damage creatures -> all die in pass 1 (CR 704.3
    says SBAs check simultaneously). Returns 1 (one pass, three
    deaths)."""
    for obj_id in (P0_FIRST_BEAR_ID, P0_SECOND_BEAR_ID, P1_FIRST_BEAR_ID):
        synthetic_state.deal_damage(obj_id, 2)
    assert synthetic_state.apply_sbas_until_quiescent() == 1
    for obj_id in (P0_FIRST_BEAR_ID, P0_SECOND_BEAR_ID, P1_FIRST_BEAR_ID):
        assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_apply_sbas_until_quiescent_handles_zero_toughness(synthetic_state):
    """A 0-toughness creature (round-12 SBA) is destroyed in one pass.
    Returns 1."""
    obj_id = P0_FIRST_BEAR_ID  # 2/2
    synthetic_state.set_counters(obj_id, -2)  # effective toughness 0
    assert synthetic_state.apply_sbas_until_quiescent() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_apply_sbas_until_quiescent_combined_lethal_and_zero_toughness(
    synthetic_state,
):
    """One lethal-damage creature AND one 0-toughness creature -> both
    die in pass 1 (single apply_sbas pass kills both). Returns 1."""
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    synthetic_state.set_counters(P0_SECOND_BEAR_ID, -2)
    assert synthetic_state.apply_sbas_until_quiescent() == 1
    assert synthetic_state.find_object_zone(P0_FIRST_BEAR_ID) == ZONE_GRAVEYARD
    assert synthetic_state.find_object_zone(P0_SECOND_BEAR_ID) == ZONE_GRAVEYARD


def test_apply_sbas_until_quiescent_clone_isolation(synthetic_state):
    """CoW canary for the loop primitive — running it on a clone does
    not affect the original."""
    obj_id = P0_FIRST_BEAR_ID
    bf_before_p0 = synthetic_state.battlefield_sizes[0]
    gy_before_p0 = synthetic_state.graveyard_sizes[0]

    other = synthetic_state.clone_once()
    other.deal_damage(obj_id, 2)
    assert other.apply_sbas_until_quiescent() == 1

    # Clone reflects destruction.
    assert other.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert other.battlefield_sizes[0] == bf_before_p0 - 1
    assert other.graveyard_sizes[0] == gy_before_p0 + 1

    # Original is untouched.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before_p0
    assert synthetic_state.graveyard_sizes[0] == gy_before_p0


def test_apply_sbas_until_quiescent_indestructible_skipped(synthetic_state):
    """A round-13 indestructible 2/2 with lethal damage stays alive;
    no SBA fires so the loop returns 0."""
    obj_id = synthetic_state.register_and_inject_object(
        _indestructible_2_2_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.deal_damage(obj_id, 5)  # overkill on a 2/2 indestructible
    assert synthetic_state.apply_sbas_until_quiescent() == 0
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


# ----------------------------- Counter-modified toughness for 704.5g --------
#
# The lethal-damage SBA (CR 704.5g) compares marked damage against EFFECTIVE
# toughness, not printed toughness. Effective toughness folds in the +1/+1
# vs -1/-1 counter contribution: ``printing.toughness + obj.counters`` for
# the round-2 net-i32 model.
#
# Round-5's apply_sbas originally used the raw printed value, which drifts
# from real MTG once counters land on the creature: a 2/2 with two +1/+1
# counters takes 3 damage and survives (effective toughness 4); a 4/4 with
# two -1/-1 counters takes 3 damage and dies (effective toughness 2). The
# round-12 0-toughness SBA (704.5f) has always used effective toughness
# since it landed; this group of tests pins the same semantics into 704.5g.
#
# Pre-existing tests (no-counter cases) still pass under the upgraded
# predicate — when ``obj.counters == 0`` effective and printed toughness
# coincide.


def _two_two_creature_dict():
    """Vanilla 2/2 creature printing — no keywords, no special effects."""
    return {
        "name": "Test 2/2",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2},
        "power": 2,
        "toughness": 2,
        "keywords": [],
        "effects": [],
    }


def _four_four_creature_dict():
    """Vanilla 4/4 creature printing — no keywords, no special effects."""
    return {
        "name": "Test 4/4",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 4},
        "power": 4,
        "toughness": 4,
        "keywords": [],
        "effects": [],
    }


def _six_six_creature_dict():
    """Vanilla 6/6 creature printing — no keywords, no special effects."""
    return {
        "name": "Test 6/6",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 6},
        "power": 6,
        "toughness": 6,
        "keywords": [],
        "effects": [],
    }


def test_sba_lethal_damage_uses_counter_modified_toughness_increased(
    synthetic_state,
):
    """A 2/2 with +2 net counters (effective toughness 4) takes 3 damage —
    NOT lethal under counter-modified semantics. Creature survives. Pins
    the upgraded 704.5g predicate against the raw-printed-toughness
    regression."""
    obj_id = P0_FIRST_BEAR_ID  # Grizzly Bears, 2/2
    synthetic_state.set_counters(obj_id, 2)  # effective toughness 2 + 2 = 4
    synthetic_state.deal_damage(obj_id, 3)
    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


def test_sba_lethal_damage_uses_counter_modified_toughness_decreased(
    synthetic_state,
):
    """A 4/4 with -2 net counters (effective toughness 2) takes 3 damage —
    IS lethal under counter-modified semantics. Without the upgrade,
    raw toughness 4 would mean 3 < 4 and the creature would survive
    (the raw-toughness regression). Effective toughness 2 means 3 >= 2,
    and the creature dies."""
    obj_id = synthetic_state.register_and_inject_object(
        _four_four_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.set_counters(obj_id, -2)  # effective toughness 4 - 2 = 2
    synthetic_state.deal_damage(obj_id, 3)
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_sba_lethal_damage_pre_existing_passes(synthetic_state):
    """Pin the no-counter case: a 2/2 with 0 counters and 2 damage still
    dies. The upgraded predicate must reduce to the printed-toughness
    behavior when counters == 0."""
    obj_id = P0_FIRST_BEAR_ID
    synthetic_state.deal_damage(obj_id, 2)
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_sba_zero_toughness_pre_existing_passes(synthetic_state):
    """Pin the printed-zero-toughness case: a 0/0 creature dies on the
    first SBA pass via 704.5f, even with no counters and no damage.
    Independent of the 704.5g upgrade (the slice doesn't touch this
    path) but documented here as a baseline canary."""
    obj_id = synthetic_state.register_and_inject_object(
        _zero_zero_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    assert synthetic_state.apply_sbas() == 1
    assert synthetic_state.find_object_zone(obj_id) == ZONE_GRAVEYARD


def test_sba_negative_counters_dont_kill_high_toughness(synthetic_state):
    """Anti-canary: a 6/6 with -2 net counters (effective toughness 4) and
    no damage doesn't trigger ANY destruction SBA. The 704.5f check sees
    effective 4 > 0; the 704.5g check sees damage 0 < 4; both pass.
    Apply_sbas returns 0."""
    obj_id = synthetic_state.register_and_inject_object(
        _six_six_creature_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.set_counters(obj_id, -2)  # effective toughness 4
    assert synthetic_state.apply_sbas() == 0
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD


def test_sba_clone_isolation_with_counters(synthetic_state):
    """CoW canary for the counter-modified 704.5g path. Damage and
    counter-set on a CLONE drives the clone's creature past effective
    toughness; apply_sbas on the clone destroys it; the original is
    untouched (battlefield, graveyard, damage, counters all unchanged)."""
    obj_id = P0_FIRST_BEAR_ID  # Grizzly Bears, 2/2
    bf_before_p0 = synthetic_state.battlefield_sizes[0]
    gy_before_p0 = synthetic_state.graveyard_sizes[0]

    other = synthetic_state.clone_once()
    # On the clone: drop to effective toughness 1 with -1 net counter,
    # deal 1 damage. Lethal under counter-modified 704.5g (1 >= 1).
    other.set_counters(obj_id, -1)
    other.deal_damage(obj_id, 1)
    assert other.apply_sbas() == 1

    # Clone reflects destruction.
    assert other.find_object_zone(obj_id) == ZONE_GRAVEYARD
    assert other.battlefield_sizes[0] == bf_before_p0 - 1
    assert other.graveyard_sizes[0] == gy_before_p0 + 1

    # Original is untouched: still on battlefield, no damage, no counters.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before_p0
    assert synthetic_state.graveyard_sizes[0] == gy_before_p0
    assert synthetic_state.get_damage(obj_id) == 0
    assert synthetic_state.get_counters(obj_id) == 0
