"""Damage primitives on the Rust ``GameState``.

Ported from ``tests/test_combat.py`` and the per-creature ``damage``
field in ``src/mtg_bot/game.py`` (``GameObject.damage: int = 0``)
plus the end-of-turn cleanup in ``src/mtg_bot/turn.py`` that zeroes
``obj.damage`` on every battlefield permanent (CR 514.2 — "Damage
wears off").

Scope: a single per-instance ``i16`` accumulating damage marker on
``GameObject``. The primitives in this slice are the read / write /
clear / batch-clear operations every higher layer (combat, direct
damage spells, pingers, end-of-turn cleanup) needs from the engine.

**Lethal-damage SBA enforcement is out of scope (CR 704.5h).** This
slice reports raw marked damage; "damage >= toughness => destroy"
is a state-based action that needs an SBA loop on the Rust side
(future slice). Until then, callers can ask
``is_damage_at_least(obj_id, threshold)`` if they want the boolean
form against an externally-supplied threshold.

**Stepping-stone for is_lethal: ``is_damage_at_least``.** ``GameObject``
does not yet carry a base toughness — a sibling agent is wiring
``base_toughness`` onto ``CardPrinting`` in the bridge-ext slice this
round. So the lethal-damage helper exposed here is the threshold
form ``is_damage_at_least(obj_id, threshold)`` which a future slice
trivially specializes into ``is_lethal(obj_id) == is_damage_at_least(
obj_id, base_toughness)`` once the bridge lands. Keeping this on
the schedule today (vs deferring to the bridge slice) gets the read
shape right and surfaces the obvious follow-up by name.

The CoW canary lives in
``test_deal_damage_on_clone_does_not_affect_original`` —
load-bearing assertion that ``Arc::make_mut`` was used correctly
on the per-zone Vec when applying damage. Symmetric with the
canaries in ``test_tap.py`` and ``test_counters.py``.
"""

from __future__ import annotations

import pytest


# `make_synthetic_state(num_battlefield=10, num_hand=7, num_library=50)`
# lays player 0 out as:
#   battlefield: ids 0..9
#   hand:        ids 10..16
#   library:     ids 17..66
# and player 1's range starts at 67. We rely on these id ranges below.
P0_FIRST_BATTLEFIELD_ID = 0
P0_BATTLEFIELD_LAST_ID = 9
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17
P1_FIRST_BATTLEFIELD_ID = 67


def test_object_starts_with_zero_damage(synthetic_state):
    """Fresh synthetic objects come up with `damage == 0` (mirrors
    `GameObject.damage: i16 = 0` default in `rust/src/state.rs`)."""
    for obj_id in range(P0_FIRST_BATTLEFIELD_ID, P0_FIRST_BATTLEFIELD_ID + 10):
        assert synthetic_state.get_damage(obj_id) == 0


def test_get_damage_returns_value(synthetic_state):
    """After dealing damage, `get_damage` reads back the accumulated
    marker. Smallest end-to-end check that read and write touch the
    same field."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(obj_id, 3)
    assert synthetic_state.get_damage(obj_id) == 3


def test_get_damage_unknown_id_returns_none(synthetic_state):
    """Reading damage off an id that doesn't exist anywhere returns
    `None`. Matches the read-side `is_tapped` / `get_counters`
    convention of distinguishing 'not found' from a real value."""
    assert synthetic_state.get_damage(999_999) is None


def test_deal_damage_increments(synthetic_state):
    """`deal_damage(obj_id, n)` adds n to the marked damage. Two
    successive deals of (2, 3) accumulate to 5 — damage is additive,
    not absolute (CR 120.3, multiple sources mark separately)."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    assert synthetic_state.deal_damage(obj_id, 2) is True
    assert synthetic_state.deal_damage(obj_id, 3) is True
    assert synthetic_state.get_damage(obj_id) == 5


def test_deal_damage_unknown_id_returns_false(synthetic_state):
    """Dealing damage to an id that isn't in any zone returns False
    rather than raising — mirrors the silent-no-op pattern that
    `tap_object` and `set_counters` use for the same scenario
    (object already left the relevant zone)."""
    assert synthetic_state.deal_damage(999_999, 3) is False


def test_deal_damage_negative_amount_raises_value_error(synthetic_state):
    """Per the project's "fail loud" convention (mirrors `gain_life`
    and `add_mana` in earlier slices), a negative `amount` to
    `deal_damage` is a caller bug rather than a request to heal —
    raises `ValueError`. `amount = 0` is allowed and is a true no-op."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    with pytest.raises(ValueError):
        synthetic_state.deal_damage(obj_id, -1)
    # 0 is allowed as a no-op.
    assert synthetic_state.deal_damage(obj_id, 0) is True
    assert synthetic_state.get_damage(obj_id) == 0


def test_clear_damage_zeroes(synthetic_state):
    """`clear_damage(obj_id)` zeroes the per-object damage marker.
    The end-of-turn cleanup step (CR 514.2) uses this to wear off
    damage on every battlefield permanent. Returns `True` on hit,
    `False` on unknown id."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(obj_id, 4)
    assert synthetic_state.get_damage(obj_id) == 4
    assert synthetic_state.clear_damage(obj_id) is True
    assert synthetic_state.get_damage(obj_id) == 0
    # Unknown id is a silent False, symmetric with the rest of the
    # not-found convention.
    assert synthetic_state.clear_damage(999_999) is False


def test_clear_damage_for_player_zeroes_only_their_objects(synthetic_state):
    """`clear_damage_for_player(player_id)` walks one player's
    battlefield and zeroes each permanent's damage marker. The other
    player's permanents must not be touched — needed for any future
    "remove all damage from creatures you control" effect (Heroic
    Intervention / Make a Stand etc.) and as a building block for
    the per-player end-of-turn pass."""
    p0_obj = P0_FIRST_BATTLEFIELD_ID
    p1_obj = P1_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(p0_obj, 3)
    synthetic_state.deal_damage(p1_obj, 4)
    assert synthetic_state.get_damage(p0_obj) == 3
    assert synthetic_state.get_damage(p1_obj) == 4
    synthetic_state.clear_damage_for_player(0)
    assert synthetic_state.get_damage(p0_obj) == 0
    # Player 1's marked damage must be untouched.
    assert synthetic_state.get_damage(p1_obj) == 4


def test_clear_damage_for_all_objects(synthetic_state):
    """`clear_all_damage()` is the wholesale end-of-turn cleanup form
    — mirrors the `for obj in state.battlefield: obj.damage = 0` loop
    in `src/mtg_bot/turn.py` cleanup. Walks both players' battlefields
    and zeroes every damage marker in one call."""
    p0_obj = P0_FIRST_BATTLEFIELD_ID
    p0_obj2 = P0_FIRST_BATTLEFIELD_ID + 1
    p1_obj = P1_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(p0_obj, 1)
    synthetic_state.deal_damage(p0_obj2, 2)
    synthetic_state.deal_damage(p1_obj, 3)
    synthetic_state.clear_all_damage()
    assert synthetic_state.get_damage(p0_obj) == 0
    assert synthetic_state.get_damage(p0_obj2) == 0
    assert synthetic_state.get_damage(p1_obj) == 0


def test_is_damage_at_least_with_toughness_helper(synthetic_state):
    """`is_damage_at_least(obj_id, threshold)` is the threshold form
    of the lethal-damage check. Returns `Some(true)` when the
    object's marked damage is >= threshold, `Some(false)` when it
    is below, and `None` if no object exists with that id.

    Threshold form (vs full `is_lethal`) is the stepping-stone until
    `CardPrinting.base_toughness` lands in a parallel sibling slice
    — see this file's docstring."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    # No damage yet: threshold of 1 says "not at least 1".
    assert synthetic_state.is_damage_at_least(obj_id, 1) is False
    # threshold of 0 is trivially true (0 >= 0).
    assert synthetic_state.is_damage_at_least(obj_id, 0) is True
    synthetic_state.deal_damage(obj_id, 2)
    assert synthetic_state.is_damage_at_least(obj_id, 2) is True
    assert synthetic_state.is_damage_at_least(obj_id, 3) is False
    # Unknown id returns None (not found is distinct from a boolean).
    assert synthetic_state.is_damage_at_least(999_999, 1) is None


def test_clone_preserves_damage(synthetic_state):
    """A clone of the state — the same `Clone` impl MCTS exercises on
    every node expansion — carries damage values forward. They live
    inside `Arc<Vec<GameObject>>` and ride the refcount bump."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.deal_damage(obj_id, 3)
    snapshot = synthetic_state.clone_once()
    assert snapshot.get_damage(obj_id) == 3


def test_deal_damage_on_clone_does_not_affect_original(synthetic_state):
    """**CoW canary.** The whole point of `Arc<Vec<GameObject>>` zones
    is that two live clones of a GameState observe independent
    mutations. Dealing damage on one clone must NOT visibly mutate
    the same object on the other — `Arc::make_mut` should clone the
    inner Vec on the first mutation while another reference is live.

    Same load-bearing invariant as
    `test_counter_change_on_clone_does_not_affect_original` but for
    the damage field. If this fails every CoW assumption downstream
    is broken."""
    obj_id = P0_FIRST_BATTLEFIELD_ID
    # Original starts at 0; baseline assertion before any mutation.
    assert synthetic_state.get_damage(obj_id) == 0
    other = synthetic_state.clone_once()
    assert other.get_damage(obj_id) == 0
    # Mutate ONLY the clone.
    assert other.deal_damage(obj_id, 5) is True
    assert other.get_damage(obj_id) == 5
    # Original must NOT see the change.
    assert synthetic_state.get_damage(obj_id) == 0, (
        "CoW broken: dealing damage on clone leaked into original"
    )
    # Symmetric direction: mutating the original after the clone
    # diverges should not retroactively touch the clone.
    assert synthetic_state.deal_damage(obj_id, 2) is True
    assert synthetic_state.get_damage(obj_id) == 2
    assert other.get_damage(obj_id) == 5
    # And clear_damage on the clone must not retroactively zero the
    # original.
    assert other.clear_damage(obj_id) is True
    assert other.get_damage(obj_id) == 0
    assert synthetic_state.get_damage(obj_id) == 2


# -----------------------------------------------------------------
# `deal_damage_to_player` — direct life mutation primitive.
#
# Distinct from `deal_damage` (above), which targets a `GameObject`
# and accumulates onto its per-instance damage marker. This variant
# subtracts directly from `Player.life` and is the building block
# for: combat damage to a defending player, burn-spell damage to a
# player, "deals damage to each opponent" effects, etc.
#
# Lives in the damage slice (rather than life slice) because it's a
# damage-source primitive — the call site is "this thing deals N
# damage to player P", not "player P loses N life arbitrarily".
# The current implementation collapses the two operations (Rust
# rules-text fidelity for damage→life-loss arrives in a later slice
# once replacement effects + lifelink-like routing land), but the
# named entry point for "damage to player" exists today so callers
# don't have to know the simplification.
# -----------------------------------------------------------------


def test_deal_damage_to_player_subtracts_life(synthetic_state):
    """`deal_damage_to_player(player_id, amount)` subtracts `amount`
    from that player's life total. Fresh synthetic state starts both
    players at life=20; dealing 5 to player 0 leaves them at 15."""
    assert synthetic_state.get_life(0) == 20
    synthetic_state.deal_damage_to_player(0, 5)
    assert synthetic_state.get_life(0) == 15


def test_deal_damage_to_player_zero_amount_no_op(synthetic_state):
    """`amount = 0` is a true no-op (life unchanged). Symmetric with
    the `deal_damage` zero-amount semantics on the GameObject form
    and the `gain_life` / `lose_life` zero-amount semantics in the
    life slice."""
    assert synthetic_state.get_life(0) == 20
    synthetic_state.deal_damage_to_player(0, 0)
    assert synthetic_state.get_life(0) == 20


def test_deal_damage_to_player_negative_raises(synthetic_state):
    """A negative `amount` is a caller bug — raises ValueError per
    the project's "fail loud" convention (mirrors `gain_life`,
    `lose_life`, and `deal_damage`)."""
    with pytest.raises(ValueError):
        synthetic_state.deal_damage_to_player(0, -1)
    # Life is unchanged after the failed call.
    assert synthetic_state.get_life(0) == 20


def test_deal_damage_to_player_unknown_player_raises(synthetic_state):
    """`player_id` outside [0, 1] raises — matches the life-slice
    `check_player_id` contract."""
    with pytest.raises((ValueError, IndexError)):
        synthetic_state.deal_damage_to_player(99, 1)


def test_deal_damage_to_player_can_take_life_below_zero(synthetic_state):
    """No clamping at zero — the engine reports raw life totals;
    "loss at <=0 is a game-loss" is a state-based action handled at
    a higher layer (CR 704.5a). Same rule as `lose_life` in the life
    slice."""
    assert synthetic_state.get_life(0) == 20
    synthetic_state.deal_damage_to_player(0, 25)
    assert synthetic_state.get_life(0) == -5


def test_deal_damage_to_player_does_not_affect_other_player(synthetic_state):
    """Dealing damage to player 0 must not touch player 1's life
    total, and vice versa."""
    assert synthetic_state.get_life(0) == 20
    assert synthetic_state.get_life(1) == 20
    synthetic_state.deal_damage_to_player(0, 7)
    assert synthetic_state.get_life(0) == 13
    assert synthetic_state.get_life(1) == 20
    synthetic_state.deal_damage_to_player(1, 3)
    assert synthetic_state.get_life(0) == 13
    assert synthetic_state.get_life(1) == 17


def test_clone_then_deal_damage_to_player_does_not_affect_original(
    synthetic_state,
):
    """CoW canary for the player-life path. `Player` lives by-value
    inside `GameState.players: [Player; 2]` (not behind an Arc), so
    a `GameState.clone()` deep-copies each Player struct and `life`
    is a Copy field — mutating life on one clone must not bleed into
    the other. Symmetric with the analogous canary in
    `test_life.py::test_modify_one_clone_does_not_affect_other`."""
    assert synthetic_state.get_life(0) == 20
    other = synthetic_state.clone_once()
    assert other.get_life(0) == 20
    # Mutate ONLY the clone.
    other.deal_damage_to_player(0, 4)
    assert other.get_life(0) == 16
    # Original must NOT see the change.
    assert synthetic_state.get_life(0) == 20, (
        "CoW broken: deal_damage_to_player on clone leaked into original"
    )
    # Symmetric direction: mutating original after divergence does
    # not retroactively touch the clone.
    synthetic_state.deal_damage_to_player(0, 2)
    assert synthetic_state.get_life(0) == 18
    assert other.get_life(0) == 16
