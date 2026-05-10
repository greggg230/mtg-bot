"""Mana production / mana pool primitives on the Rust ``GameState``.

Ported from ``src/mtg_bot/effects/generic/mana.py`` (the ``AddManaToPool``
and friends sites that mutate ``Player.mana_pool``) plus the
"tap-a-Forest-for-{G}" idiom that the Python side spreads across an
``ActivatedAbility(is_mana_ability=True)`` plus an ``AddManaToPool``
effect. The Rust foundation commit (``df71d56``) added a typed
``Player.mana_pool: ManaPool { white, blue, black, red, green, colorless: u8 }``;
this slice grows the methods that read and mutate it.

**Negative-amount semantics decision** (mirrors the life slice's
choice — see ``test_life.py``): ``add_mana`` and ``remove_mana``
take ``i32`` amounts and raise ``ValueError`` on a negative value
rather than silently flooring at zero. ``amount = 0`` is allowed and
is a true no-op. ``remove_mana`` underflow (e.g. removing 1 green
from an empty pool) raises rather than clamping at zero — silent
underflow would mask paid-but-not-collected bugs.

**Mana-production lookup is provisional** (option (a) in the slice
plan): ``tap_for_mana`` recognizes Forests by printing-name string
match against the synthetic registry's "Forest" entry. This keeps
the slice scoped to ``state_ops_mana.rs`` — adding a typed
``mana_produced`` field to ``CardPrinting`` would require touching
``cards.rs``, which a sibling agent is also editing this round.
A future bridge slice will replace the name-keyed lookup with the
proper field.
"""

from __future__ import annotations

import pytest


def _bf_ids(state, player_id):
    """Return the GameObject ids on ``player_id``'s battlefield.

    Mirrors ``tests_rust/test_tap.py``'s helper. Synthetic-state
    defaults: 10 battlefield / 7 hand / 50 library per side.
    Player 0's battlefield ids are 0..9; player 1's start at 67.
    The first half (``num_battlefield // 2 = 5``) of each
    battlefield is Forests, the second half is Grizzly Bears
    (see ``make_player`` in ``rust/src/state.rs``).
    """
    if player_id == 0:
        return list(range(0, 10))
    return list(range(67, 77))


def _forest_id(state, player_id):
    """First Forest id on ``player_id``'s battlefield."""
    return _bf_ids(state, player_id)[0]


def _bears_id(state, player_id):
    """First Grizzly Bears id on ``player_id``'s battlefield (slot 5,
    since 5 forests come first)."""
    return _bf_ids(state, player_id)[5]


# --- mana pool reads ------------------------------------------------


def test_initial_mana_pool_is_empty(synthetic_state):
    """Both players' pools start at all-zero (the foundation commit
    defaults ``Player.mana_pool: ManaPool`` in ``make_player``)."""
    expected = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    assert synthetic_state.get_mana_pool(0) == expected
    assert synthetic_state.get_mana_pool(1) == expected


def test_get_mana_pool_returns_dict_view(synthetic_state):
    """The 6-key dict has all six expected keys, all int-typed."""
    pool = synthetic_state.get_mana_pool(0)
    assert set(pool.keys()) == {"W", "U", "B", "R", "G", "C"}
    for v in pool.values():
        assert isinstance(v, int)


def test_get_mana_pool_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.get_mana_pool(99)


# --- add_mana / remove_mana / empty_mana_pool -----------------------


def test_add_mana_increments_pool(synthetic_state):
    synthetic_state.add_mana(0, "G", 3)
    assert synthetic_state.get_mana_pool(0)["G"] == 3
    # Other colors / opponent pool untouched.
    pool = synthetic_state.get_mana_pool(0)
    for color in ("W", "U", "B", "R", "C"):
        assert pool[color] == 0
    assert synthetic_state.get_mana_pool(1)["G"] == 0


def test_add_mana_all_six_colors(synthetic_state):
    """Every supported color (W/U/B/R/G/C) is acceptable."""
    for color in ("W", "U", "B", "R", "G", "C"):
        synthetic_state.add_mana(0, color, 1)
    pool = synthetic_state.get_mana_pool(0)
    assert pool == {"W": 1, "U": 1, "B": 1, "R": 1, "G": 1, "C": 1}


def test_add_mana_unknown_color_raises_value_error(synthetic_state):
    with pytest.raises(ValueError):
        synthetic_state.add_mana(0, "Q", 1)
    # Lowercase is also rejected — case-sensitive by convention.
    with pytest.raises(ValueError):
        synthetic_state.add_mana(0, "g", 1)


def test_add_mana_negative_amount_raises(synthetic_state):
    """Mirrors the life-slice convention: fail loudly on negative
    ``amount`` rather than silently clamping or wrapping."""
    with pytest.raises(ValueError):
        synthetic_state.add_mana(0, "G", -1)
    # State must be untouched after the rejected call.
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_add_mana_zero_amount_is_noop(synthetic_state):
    synthetic_state.add_mana(0, "G", 0)
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_add_mana_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.add_mana(99, "G", 1)


def test_remove_mana_decrements(synthetic_state):
    """Round-trip: add then remove leaves the pool unchanged."""
    synthetic_state.add_mana(0, "G", 5)
    synthetic_state.remove_mana(0, "G", 3)
    assert synthetic_state.get_mana_pool(0)["G"] == 2


def test_remove_mana_underflow_raises(synthetic_state):
    """Removing more than the pool holds raises rather than silently
    flooring at 0 — silent underflow would mask paid-but-not-
    collected bugs in cost-payment code."""
    with pytest.raises(ValueError):
        synthetic_state.remove_mana(0, "G", 1)
    # State must be untouched after the rejected call.
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_remove_mana_negative_amount_raises(synthetic_state):
    with pytest.raises(ValueError):
        synthetic_state.remove_mana(0, "G", -1)


def test_remove_mana_unknown_color_raises(synthetic_state):
    with pytest.raises(ValueError):
        synthetic_state.remove_mana(0, "Q", 1)


def test_empty_mana_pool_resets_player_to_zero(synthetic_state):
    """``empty_mana_pool(p)`` zeroes only ``p``'s pool — leaves the
    opponent alone. Mirrors the cleanup-step behavior on the Python
    side."""
    synthetic_state.add_mana(0, "G", 4)
    synthetic_state.add_mana(0, "R", 2)
    synthetic_state.add_mana(1, "U", 3)
    synthetic_state.empty_mana_pool(0)
    expected_zero = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    assert synthetic_state.get_mana_pool(0) == expected_zero
    # Opponent's pool untouched.
    assert synthetic_state.get_mana_pool(1)["U"] == 3


def test_empty_mana_pool_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.empty_mana_pool(99)


# --- tap_for_mana ---------------------------------------------------


def test_tap_for_mana_on_forest_produces_green(synthetic_state):
    """Find a Forest on player 0's battlefield, tap it for mana,
    expect ``Some("G")``, the Forest is now tapped, and player 0's
    green count went up by 1."""
    forest_id = _forest_id(synthetic_state, 0)
    # Sanity: the synthetic state seeds a Forest at this slot.
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    assert synthetic_state.is_tapped(forest_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    result = synthetic_state.tap_for_mana(forest_id)
    assert result == "G"
    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


def test_tap_for_mana_credits_correct_controller(synthetic_state):
    """A Forest on player 1's battlefield credits player 1's pool,
    not player 0's."""
    forest_id = _forest_id(synthetic_state, 1)
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    result = synthetic_state.tap_for_mana(forest_id)
    assert result == "G"
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    assert synthetic_state.get_mana_pool(1)["G"] == 1


def test_tap_for_mana_on_already_tapped_returns_none(synthetic_state):
    """Tapping an already-tapped Forest returns None and adds no
    mana (matches the Python convention that a mana ability fails
    its tap-cost on an already-tapped permanent)."""
    forest_id = _forest_id(synthetic_state, 0)
    # Tap the Forest first.
    assert synthetic_state.tap_for_mana(forest_id) == "G"
    assert synthetic_state.get_mana_pool(0)["G"] == 1
    # Second tap is a no-op.
    assert synthetic_state.tap_for_mana(forest_id) is None
    assert synthetic_state.get_mana_pool(0)["G"] == 1


def test_tap_for_mana_on_non_land_returns_none(synthetic_state):
    """Grizzly Bears is a creature, not a land — tap_for_mana
    declines (returns None, no mana, no tap)."""
    bears_id = _bears_id(synthetic_state, 0)
    assert synthetic_state.printing_name_of(bears_id) == "Grizzly Bears"
    assert synthetic_state.tap_for_mana(bears_id) is None
    # The Bears must NOT have been tapped as a side effect.
    assert synthetic_state.is_tapped(bears_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_tap_for_mana_unknown_id_returns_none(synthetic_state):
    """An obj_id that doesn't exist on any battlefield is a quiet
    no-op — matches ``tap_object``'s behavior on missing ids."""
    assert synthetic_state.tap_for_mana(999_999) is None


# --- CoW canary -----------------------------------------------------


def test_clone_then_add_mana_does_not_affect_original(synthetic_state):
    """Load-bearing CoW canary. ``Player`` lives by-value in
    ``GameState.players: [Player; 2]``, so cloning the GameState
    deep-copies the Player struct (including its ``ManaPool`` —
    which is itself ``Copy``). A mutation on the clone must not
    bleed back into the original.

    The mana_pool case is structurally identical to life (POD
    embedded by value in Player), so this should "just work" without
    any Arc juggling."""
    cloned = synthetic_state.clone_once()
    cloned.add_mana(0, "G", 4)
    assert cloned.get_mana_pool(0)["G"] == 4
    # Original untouched.
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_clone_then_tap_for_mana_does_not_affect_original(synthetic_state):
    """Same canary, but exercising the full tap-for-mana path —
    which mutates BOTH a battlefield ``Arc<Vec<GameObject>>`` (via
    ``Arc::make_mut``) AND the by-value ``ManaPool``. Either side
    leaking into the original would be a CoW bug."""
    forest_id = _forest_id(synthetic_state, 0)
    cloned = synthetic_state.clone_once()
    assert cloned.tap_for_mana(forest_id) == "G"
    # Clone shows the tap + the mana.
    assert cloned.is_tapped(forest_id) is True
    assert cloned.get_mana_pool(0)["G"] == 1
    # Original observed neither.
    assert synthetic_state.is_tapped(forest_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0


# --- pay_mana_cost (CR 202.2) ----------------------------------------


def test_pay_mana_cost_simple_colored(synthetic_state):
    """pool={G:1}, cost={G:1} → True; pool ends empty."""
    synthetic_state.add_mana(0, "G", 1)
    assert synthetic_state.pay_mana_cost(0, {"G": 1}) is True
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0
    }


def test_pay_mana_cost_insufficient_color_returns_false(synthetic_state):
    """pool={W:1}, cost={G:1} → False; pool unchanged.

    A colored pip can only be paid with the matching color — the
    one white in the pool can't satisfy the green requirement.
    No mutation on the rejection path."""
    synthetic_state.add_mana(0, "W", 1)
    assert synthetic_state.pay_mana_cost(0, {"G": 1}) is False
    assert synthetic_state.get_mana_pool(0) == {
        "W": 1, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0
    }


def test_pay_mana_cost_generic_drawn_from_any_color(synthetic_state):
    """pool={G:2}, cost={generic:1, G:1} → True; pool ends empty.

    Generic mana is paid with any color (CR 107.4). After the green
    pip is reserved, the second green satisfies the generic 1."""
    synthetic_state.add_mana(0, "G", 2)
    assert synthetic_state.pay_mana_cost(0, {"generic": 1, "G": 1}) is True
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0
    }


def test_pay_mana_cost_insufficient_total_for_generic_returns_false(synthetic_state):
    """pool={G:1}, cost={generic:2} → False; pool unchanged."""
    synthetic_state.add_mana(0, "G", 1)
    assert synthetic_state.pay_mana_cost(0, {"generic": 2}) is False
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "C": 0
    }


def test_pay_mana_cost_zero_cost_returns_true(synthetic_state):
    """pool={G:1}, cost={} → True; pool unchanged.

    A zero-mana cost is trivially payable from any pool, including
    an empty one. Mirrors pay_life_cost(0) → True."""
    synthetic_state.add_mana(0, "G", 1)
    assert synthetic_state.pay_mana_cost(0, {}) is True
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "C": 0
    }


def test_pay_mana_cost_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.pay_mana_cost(99, {"G": 1})


def test_pay_mana_cost_clone_isolation(synthetic_state):
    """CoW canary — paying on a clone leaves the original pool
    untouched. ManaPool is by-value-Copy in Player, so this should
    "just work" the same way the add_mana clone canary does."""
    synthetic_state.add_mana(0, "G", 2)
    cloned = synthetic_state.clone_once()
    assert cloned.pay_mana_cost(0, {"G": 1}) is True
    assert cloned.get_mana_pool(0)["G"] == 1
    # Original untouched.
    assert synthetic_state.get_mana_pool(0)["G"] == 2


def test_pay_mana_cost_drains_colorless_first_for_generic(synthetic_state):
    """pool={C:1, G:2}, cost={generic:1} → True; pool ends {G:2}.

    Deterministic generic-deduction order is colorless first, then
    WUBRG. Colorless is the most-restricted source (it can ONLY pay
    generic), so draining it first preserves the most flexibility
    for any later cost-payment from the same pool."""
    synthetic_state.add_mana(0, "C", 1)
    synthetic_state.add_mana(0, "G", 2)
    assert synthetic_state.pay_mana_cost(0, {"generic": 1}) is True
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 2, "C": 0
    }


def test_pay_mana_cost_complex_multi_color(synthetic_state):
    """pool={W:1, U:1, B:1, R:1, G:1, C:1}, cost={generic:3, W:1, R:1}
    → True; pool ends {U:0, B:0, G:1} (rest zero).

    Step-by-step:
    1. Reserve colored pips: W:1 → W=0, R:1 → R=0.
       Intermediate pool: {W:0, U:1, B:1, R:0, G:1, C:1}.
    2. Pay generic=3 from order C, W, U, B, R, G (W and R already 0):
       drain C (1) → drain U (1) → drain B (1) → 3 paid.
    3. Final: {W:0, U:0, B:0, R:0, G:1, C:0}."""
    for color in ("W", "U", "B", "R", "G", "C"):
        synthetic_state.add_mana(0, color, 1)
    cost = {"generic": 3, "W": 1, "R": 1}
    assert synthetic_state.pay_mana_cost(0, cost) is True
    assert synthetic_state.get_mana_pool(0) == {
        "W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "C": 0
    }
