"""Life modification primitives on the Rust ``GameState``.

Ported from Python's ``src/mtg_bot/game.py`` (the ``gain_life`` helper
and the many ``p.life -= amount`` direct mutations scattered through
``effects/``, ``combat.py``, and ``actions.py``). The Python side
exposes only ``gain_life`` as a named helper — there is no
``lose_life`` or ``set_life`` function — so the Rust API surface here
is a small generalization rather than a one-to-one port.

**Negative-amount semantics decision** (documented divergence from
Python): the Python ``gain_life`` silently no-ops on
``amount <= 0`` (returns 0, makes no mutation). The Rust port raises
``ValueError`` on a negative ``amount`` for ``gain_life`` /
``lose_life`` instead. Rationale: silent no-ops on bad inputs are a
known anti-pattern in this codebase (see the user's
"silent no-op pipeline runs are pipeline bugs" guidance) — the engine
should fail loudly when a caller passes nonsense. ``amount = 0`` is
allowed and is a true no-op (it's a legitimate value, not an error).

**No clamping at zero**: ``lose_life`` allows the result to go
negative. The engine reports raw life totals; "loss at 0" is a
state-based action concern handled at a higher layer (not yet
implemented in Rust).
"""

from __future__ import annotations

import pytest


def test_initial_life_is_twenty(synthetic_state):
    assert synthetic_state.life_totals == [20, 20]


def test_get_life_returns_player_life(synthetic_state):
    assert synthetic_state.get_life(0) == 20
    assert synthetic_state.get_life(1) == 20


def test_get_life_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.get_life(99)


def test_set_life_changes_only_target_player(synthetic_state):
    synthetic_state.set_life(0, 15)
    assert synthetic_state.get_life(0) == 15
    assert synthetic_state.get_life(1) == 20
    assert synthetic_state.life_totals == [15, 20]


def test_set_life_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.set_life(99, 10)


def test_lose_life_subtracts(synthetic_state):
    synthetic_state.lose_life(0, 5)
    assert synthetic_state.get_life(0) == 15
    assert synthetic_state.get_life(1) == 20


def test_gain_life_adds(synthetic_state):
    synthetic_state.gain_life(0, 5)
    assert synthetic_state.get_life(0) == 25
    assert synthetic_state.get_life(1) == 20


def test_lose_life_can_go_negative(synthetic_state):
    # Engine reports raw life; "loss at 0" is a state-based action
    # at a higher layer, not implemented here.
    synthetic_state.lose_life(0, 25)
    assert synthetic_state.get_life(0) == -5


def test_lose_life_negative_amount_rejects(synthetic_state):
    # Documented divergence from Python's silent no-op:
    # the Rust engine raises on negative `amount`.
    with pytest.raises(ValueError):
        synthetic_state.lose_life(0, -3)
    # State must be untouched after the rejected call.
    assert synthetic_state.get_life(0) == 20


def test_gain_life_negative_amount_rejects(synthetic_state):
    with pytest.raises(ValueError):
        synthetic_state.gain_life(0, -3)
    assert synthetic_state.get_life(0) == 20


def test_lose_life_zero_amount_is_noop(synthetic_state):
    synthetic_state.lose_life(0, 0)
    assert synthetic_state.get_life(0) == 20


def test_gain_life_zero_amount_is_noop(synthetic_state):
    synthetic_state.gain_life(0, 0)
    assert synthetic_state.get_life(0) == 20


def test_clone_preserves_life_totals(synthetic_state):
    synthetic_state.set_life(0, 12)
    synthetic_state.set_life(1, 17)
    cloned = synthetic_state.clone_once()
    assert cloned.life_totals == [12, 17]


def test_modify_one_clone_does_not_affect_other(synthetic_state):
    # Load-bearing CoW test: Player is held by-value in
    # GameState.players, so cloning the GameState copies the Player,
    # and a life mutation on the clone must not bleed back into the
    # original.
    cloned = synthetic_state.clone_once()
    cloned.set_life(0, 1)
    cloned.lose_life(1, 7)
    assert cloned.life_totals == [1, 13]
    assert synthetic_state.life_totals == [20, 20]


def test_set_life_to_negative_value_is_allowed(synthetic_state):
    # `set_life` is a raw setter — no clamping, no sign restriction.
    synthetic_state.set_life(0, -5)
    assert synthetic_state.get_life(0) == -5


# ---------------------------------------------------------------------------
# pay_life_cost — CR 118.4 cost-payment counterpart to lose_life.
#
# Costs are checked-then-paid: a player can pay X life as a cost only if
# their current life total is strictly greater than X (paying X would
# drop them to current - X, which must be >= 1; CR 118.4: "A player
# can't pay more life than they have"). Concretely, the no-pay-to-die
# rule means pay_life_cost(life=5, amount=5) returns False — paying it
# would zero them out, which isn't allowed for a voluntary cost.
#
# Returns True on successful payment (life decremented), False on
# inability to pay (life unchanged). amount=0 is a no-op success.
# Negative amount and unknown player_id raise (matches gain/lose_life).
# ---------------------------------------------------------------------------


def test_pay_life_cost_decrements_life(synthetic_state):
    assert synthetic_state.pay_life_cost(0, 5) is True
    assert synthetic_state.get_life(0) == 15
    assert synthetic_state.get_life(1) == 20


def test_pay_life_cost_zero_amount_is_noop_success(synthetic_state):
    assert synthetic_state.pay_life_cost(0, 0) is True
    assert synthetic_state.get_life(0) == 20


def test_pay_life_cost_negative_amount_raises(synthetic_state):
    with pytest.raises(ValueError):
        synthetic_state.pay_life_cost(0, -1)
    assert synthetic_state.get_life(0) == 20


def test_pay_life_cost_unknown_player_raises(synthetic_state):
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.pay_life_cost(99, 1)


def test_pay_life_cost_exact_life_returns_false(synthetic_state):
    # Paying 5 life when at 5 would zero the player; not allowed for a
    # voluntary cost (CR 118.4 — can't pay more life than you have, and
    # going to 0 from a cost is the no-pay-to-die boundary).
    synthetic_state.set_life(0, 5)
    assert synthetic_state.pay_life_cost(0, 5) is False
    assert synthetic_state.get_life(0) == 5


def test_pay_life_cost_more_than_life_returns_false(synthetic_state):
    synthetic_state.set_life(0, 3)
    assert synthetic_state.pay_life_cost(0, 7) is False
    assert synthetic_state.get_life(0) == 3


def test_pay_life_cost_one_less_than_life_returns_true(synthetic_state):
    # Paying life - 1 leaves the player at 1; that's the legal boundary.
    synthetic_state.set_life(0, 5)
    assert synthetic_state.pay_life_cost(0, 4) is True
    assert synthetic_state.get_life(0) == 1


def test_pay_life_cost_clone_preserves(synthetic_state):
    # CoW canary: paying life on a clone must not bleed back into the
    # original.
    cloned = synthetic_state.clone_once()
    assert cloned.pay_life_cost(0, 7) is True
    assert cloned.get_life(0) == 13
    assert synthetic_state.get_life(0) == 20
