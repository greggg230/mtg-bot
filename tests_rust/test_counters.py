"""Counter primitives on the Rust ``GameState``.

Ported from ``tests/test_counters.py`` and the per-counter-type fields
in ``src/mtg_bot/game.py`` (``plus_one_counters``, ``minus_one_counters``,
``charge_counters``, ``stun_counters`` etc.) plus the +1/+1 / -1/-1
cancellation rule at ``state_based_actions`` (CR 704.5p).

Scope: a single per-instance ``i32`` net counter on ``GameObject``
modeling +1/+1 (positive) and -1/-1 (negative) — they cancel pairwise
at SBAs, so the net value captures the post-cancellation amount that
contributes to effective P/T. Multi-kind counter maps (charge / lore /
loyalty / brick / stun / time) are a future slice.

**Out of scope of this slice:**

- SBA-side cancellation. The Python engine cancels +1/+1 vs -1/-1 at
  ``state_based_actions``; the Rust side stores the net value
  pre-cancellation today (caller can subtract). Lifting this means
  adding an SBA pass — separate slice.
- Real effective-P/T composition (base + counters + auras + ContinuousModifiers).
  The synthetic Grizzly Bears printing has base P/T 0/0 today; the
  card-bridge slice that wires P/T into ``CardPrinting`` is the
  prerequisite. For this slice ``power_bonus_from_counters`` returns
  the raw counter contribution.

The CoW canary lives in ``test_counter_change_on_clone_does_not_affect_original``
— like ``test_tap.py``'s ``test_tap_one_clone_does_not_tap_other``,
this is the load-bearing assertion that ``Arc::make_mut`` was used
correctly on the per-zone Vec.
"""

from __future__ import annotations


# `make_synthetic_state(num_battlefield=10, num_hand=7, num_library=50)`
# lays player 0 out as:
#   battlefield: ids 0..9
#   hand:        ids 10..16
#   library:     ids 17..66
# and player 1's range starts at 67. We rely on these id ranges below.
P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17
P1_FIRST_BATTLEFIELD_ID = 67


def test_object_starts_with_zero_counters(synthetic_state):
    """Fresh synthetic objects come up with `counters == 0` (mirrors
    `GameObject.counters: i32 = 0` default in `rust/src/state.rs`)."""
    for obj_id in range(P0_FIRST_BATTLEFIELD_ID, P0_FIRST_BATTLEFIELD_ID + 10):
        assert synthetic_state.get_counters(obj_id) == 0
    # And in hand / library too.
    assert synthetic_state.get_counters(P0_FIRST_HAND_ID) == 0
    assert synthetic_state.get_counters(P0_FIRST_LIBRARY_ID) == 0


def test_get_counters_unknown_id_returns_none(synthetic_state):
    """Reading counters off an id that doesn't exist anywhere returns
    `None`. Matches the read-side `is_tapped` / `find_object_zone`
    convention of distinguishing 'not found' from a real value."""
    assert synthetic_state.get_counters(999_999) is None


def test_set_counters_changes_value(synthetic_state):
    """`set_counters(obj_id, n)` writes the absolute value and reports
    success. Subsequent `get_counters` reads back `n`."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    assert synthetic_state.set_counters(obj_id, 3) is True
    assert synthetic_state.get_counters(obj_id) == 3


def test_set_counters_unknown_id_returns_false(synthetic_state):
    """Setting counters on an id that isn't in any zone returns False
    rather than raising — mirrors the silent-no-op pattern that
    `tap_object` uses for the same scenario (object already left
    the relevant zone)."""
    assert synthetic_state.set_counters(999_999, 5) is False


def test_set_counters_works_in_any_zone(synthetic_state):
    """+1/+1 counters most often live on battlefield permanents but
    nothing in MTG forbids counters in other zones (e.g. Athreos
    counters on cards in graveyard, time counters on suspended cards
    in exile). The lookup must search all zones, not just battlefield."""
    # Hand and library too.
    assert synthetic_state.set_counters(P0_FIRST_HAND_ID, 2) is True
    assert synthetic_state.get_counters(P0_FIRST_HAND_ID) == 2
    assert synthetic_state.set_counters(P0_FIRST_LIBRARY_ID, 4) is True
    assert synthetic_state.get_counters(P0_FIRST_LIBRARY_ID) == 4


def test_add_counters_increments(synthetic_state):
    """`add_counters(obj_id, delta)` is the cumulative variant — two
    additions of (2, 1) leave the object at 3."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    assert synthetic_state.add_counters(obj_id, 2) is True
    assert synthetic_state.add_counters(obj_id, 1) is True
    assert synthetic_state.get_counters(obj_id) == 3


def test_add_negative_counters_decrements(synthetic_state):
    """Adding a negative delta models -1/-1 counters arriving on a
    creature that already had +1/+1 counters. The pairwise SBA
    cancellation (CR 704.5p) is implicit in the net-i32 representation
    — adding -2 to a 3 leaves 1 (net contribution to effective P/T
    after cancellation)."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.set_counters(obj_id, 3)
    assert synthetic_state.add_counters(obj_id, -2) is True
    assert synthetic_state.get_counters(obj_id) == 1


def test_add_counters_can_go_negative(synthetic_state):
    """The engine reports the raw net value — negative values mean
    'more -1/-1 than +1/+1 counters'. Whether the creature dies at
    0 toughness is a higher-layer SBA decision that this primitive
    does not enforce."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    assert synthetic_state.add_counters(obj_id, -2) is True
    assert synthetic_state.get_counters(obj_id) == -2


def test_add_counters_unknown_id_returns_false(synthetic_state):
    """Symmetric with `set_counters` — additive write to an id that
    isn't in any zone is a silent no-op returning False."""
    assert synthetic_state.add_counters(999_999, 1) is False


def test_power_bonus_from_counters_matches_get_counters(synthetic_state):
    """Per the +1/+1 model, a creature with N net counters contributes
    +N to its power. Real composition with base P/T + auras + layers
    happens elsewhere; this is just the counter contribution."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    # Default 0.
    assert synthetic_state.power_bonus_from_counters(obj_id) == 0
    synthetic_state.set_counters(obj_id, 4)
    assert synthetic_state.power_bonus_from_counters(obj_id) == 4
    synthetic_state.set_counters(obj_id, -1)
    assert synthetic_state.power_bonus_from_counters(obj_id) == -1


def test_power_bonus_unknown_id_returns_none(synthetic_state):
    """Same not-found convention as `get_counters`."""
    assert synthetic_state.power_bonus_from_counters(999_999) is None


def test_toughness_bonus_matches_power_bonus(synthetic_state):
    """+1/+1 affects power and toughness equally (and -1/-1 the same).
    The two methods read the same underlying field; they exist as
    distinct names so callers that care about the asymmetry case
    (e.g. *just* boosting power via a future +1/+0 counter type)
    have a clean upgrade path."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    for n in (0, 3, -2, 1):
        synthetic_state.set_counters(obj_id, n)
        assert (
            synthetic_state.power_bonus_from_counters(obj_id)
            == synthetic_state.toughness_bonus_from_counters(obj_id)
            == n
        )


def test_toughness_bonus_unknown_id_returns_none(synthetic_state):
    assert synthetic_state.toughness_bonus_from_counters(999_999) is None


def test_clone_preserves_counters(synthetic_state):
    """A clone of the state — the same `Clone` impl MCTS exercises on
    every node expansion — carries counter values forward. They live
    inside `Arc<Vec<GameObject>>` and ride the refcount bump."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.set_counters(obj_id, 3)
    snapshot = synthetic_state.clone_once()
    assert snapshot.get_counters(obj_id) == 3


def test_counter_change_on_clone_does_not_affect_original(synthetic_state):
    """**CoW canary.** The whole point of `Arc<Vec<GameObject>>` zones
    is that two live clones of a GameState observe independent
    mutations. Setting counters on one clone must NOT visibly mutate
    the same object on the other — `Arc::make_mut` should clone the
    inner Vec on the first mutation while another reference is live.

    Same load-bearing invariant as `test_tap_one_clone_does_not_tap_other`,
    but for the counter field. If this fails every CoW assumption
    downstream is broken."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    # Original starts at 0; baseline assertion before any mutation.
    assert synthetic_state.get_counters(obj_id) == 0
    other = synthetic_state.clone_once()
    assert other.get_counters(obj_id) == 0
    # Mutate ONLY the clone.
    assert other.set_counters(obj_id, 5) is True
    assert other.get_counters(obj_id) == 5
    # Original must NOT see the change.
    assert synthetic_state.get_counters(obj_id) == 0, (
        "CoW broken: setting counters on clone leaked into original"
    )
    # Symmetric direction: mutating the original after the clone
    # diverges should not retroactively touch the clone.
    assert synthetic_state.add_counters(obj_id, 2) is True
    assert synthetic_state.get_counters(obj_id) == 2
    assert other.get_counters(obj_id) == 5
