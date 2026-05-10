"""Win / loss condition primitives on the Rust ``GameState``.

Ported from Python's ``src/mtg_bot/game.py`` (the ``Player.has_lost``
flag, ``GameState.winner_id`` / ``GameState.is_draw``, and
``GameState.is_over`` helper).

**Composes with the round-1 life slice.** A player at life <= 0 is a
loser (CR 104.3a, simplified).

**Implementation choice — pure derivation (option c).** ``losers()``
is computed purely from current life totals (life <= 0). There is
NO ``Player.has_lost`` field on the Rust ``Player`` struct in this
slice — adding such a field would require editing
``rust/src/state.rs``, which is shared with sibling TDD-slice agents
(damage, cards-bridge) running in parallel against the same base
commit. Doing so would cause merge conflicts.

Consequently, "you lose the game" effects (Phage the Untouchable,
Bargain, decking-out, etc.) are deferred to a future slice that can
add the explicit-loss field at a moment when ``state.rs`` is not
being shared with siblings. The
``test_explicit_lose_game_marks_player`` test stays in the file as a
``@pytest.mark.skip(reason=...)`` to lock in the API surface for
whoever picks up that follow-up.

**Winner semantics.** With pure derivation in a 2-player game:
- 0 losers -> winner = None (game ongoing).
- exactly 1 loser, the OTHER player not also a loser -> winner =
  Some(other_player_id).
- 2 losers (both at <= 0 simultaneously, e.g. mutual lethal damage)
  -> winner = None (this is the "draw" case for now; a future slice
  may distinguish CR 104.4 simultaneous-loss draws from
  pre-game-end concessions).
"""

from __future__ import annotations

import pytest


def test_initial_state_no_loser(synthetic_state):
    """Fresh synthetic state: both players at 20 life, no losers."""
    assert list(synthetic_state.losers()) == []


def test_initial_state_no_winner(synthetic_state):
    """Fresh synthetic state: no winner yet (game ongoing)."""
    assert synthetic_state.winner() is None


def test_initial_state_not_game_over(synthetic_state):
    """Fresh synthetic state: game is not over."""
    assert synthetic_state.is_game_over() is False


def test_player_at_zero_life_is_loser(synthetic_state):
    """Setting player 0 to 0 life flags them as a loser."""
    synthetic_state.set_life(0, 0)
    assert list(synthetic_state.losers()) == [0]


def test_player_at_negative_life_is_loser(synthetic_state):
    """Setting player 0 to negative life flags them as a loser
    (CR 104.3a: 0 or less life loses the game)."""
    synthetic_state.set_life(0, -5)
    assert list(synthetic_state.losers()) == [0]


def test_winner_when_one_player_loses(synthetic_state):
    """Player 0 at 0 life, player 1 still alive: winner is player 1."""
    synthetic_state.set_life(0, 0)
    assert synthetic_state.winner() == 1


def test_winner_player_zero_when_player_one_loses(synthetic_state):
    """Symmetric: player 1 at 0 life, player 0 still alive: winner is 0."""
    synthetic_state.set_life(1, 0)
    assert synthetic_state.winner() == 0


def test_no_winner_when_both_lose(synthetic_state):
    """Both players at 0 life simultaneously (e.g. mutual lethal damage):
    no winner — this is the draw case under pure-derivation semantics."""
    synthetic_state.set_life(0, 0)
    synthetic_state.set_life(1, 0)
    assert synthetic_state.winner() is None
    # Both should be in the losers list.
    assert sorted(synthetic_state.losers()) == [0, 1]


def test_is_game_over_true_when_any_loser(synthetic_state):
    """Any loser at all -> game is over."""
    synthetic_state.set_life(0, 0)
    assert synthetic_state.is_game_over() is True


def test_is_game_over_true_when_both_lose(synthetic_state):
    """Both lose -> still game over (draw is over)."""
    synthetic_state.set_life(0, 0)
    synthetic_state.set_life(1, 0)
    assert synthetic_state.is_game_over() is True


def test_is_player_alive_true_for_positive_life(synthetic_state):
    """is_player_alive returns True for life > 0."""
    assert synthetic_state.is_player_alive(0) is True
    assert synthetic_state.is_player_alive(1) is True


def test_is_player_alive_false_at_zero_life(synthetic_state):
    """is_player_alive returns False at exactly 0 life
    (CR 104.3a: 0 or less life loses)."""
    synthetic_state.set_life(0, 0)
    assert synthetic_state.is_player_alive(0) is False
    assert synthetic_state.is_player_alive(1) is True


def test_is_player_alive_false_for_negative_life(synthetic_state):
    """is_player_alive returns False for negative life."""
    synthetic_state.set_life(0, -3)
    assert synthetic_state.is_player_alive(0) is False


def test_is_player_alive_unknown_player_raises(synthetic_state):
    """is_player_alive on an out-of-range player id raises."""
    with pytest.raises((ValueError, IndexError, OverflowError)):
        synthetic_state.is_player_alive(99)


def test_lose_life_to_zero_makes_player_loser(synthetic_state):
    """Composes with life slice: lose_life that brings life to 0
    flips the player into the losers list."""
    synthetic_state.lose_life(0, 20)
    assert list(synthetic_state.losers()) == [0]
    assert synthetic_state.winner() == 1


def test_lose_life_past_zero_makes_player_loser(synthetic_state):
    """Composes with life slice: lose_life that brings life negative
    also flips the player into the losers list."""
    synthetic_state.lose_life(0, 25)
    assert synthetic_state.get_life(0) == -5
    assert list(synthetic_state.losers()) == [0]


def test_gain_life_from_zero_revives_player(synthetic_state):
    """Pure-derivation: bringing a 0-life player back above 0 with
    gain_life takes them OUT of the losers list. (Real MTG would have
    the SBA fire first making this irrelevant; in this pure-derivation
    slice we accept the simpler model — the SBA enforcement is a
    future slice.)"""
    synthetic_state.set_life(0, 0)
    assert list(synthetic_state.losers()) == [0]
    synthetic_state.gain_life(0, 5)
    assert list(synthetic_state.losers()) == []
    assert synthetic_state.is_game_over() is False


def test_clone_preserves_loss_state(synthetic_state):
    """Cloning a state where a player is at 0 life preserves the
    derived loser status on the clone."""
    synthetic_state.set_life(0, 0)
    cloned = synthetic_state.clone_once()
    assert list(cloned.losers()) == [0]
    assert cloned.winner() == 1
    assert cloned.is_game_over() is True


def test_clone_loss_does_not_bleed_back(synthetic_state):
    """CoW canary: knocking the clone's player to 0 must not flag the
    original as a loser."""
    cloned = synthetic_state.clone_once()
    cloned.set_life(0, 0)
    # Clone shows the loss.
    assert list(cloned.losers()) == [0]
    assert cloned.winner() == 1
    assert cloned.is_game_over() is True
    # Original is untouched.
    assert list(synthetic_state.losers()) == []
    assert synthetic_state.winner() is None
    assert synthetic_state.is_game_over() is False


def test_winner_unaffected_by_clone_then_lose(synthetic_state):
    """Stronger CoW canary: even after the clone reports a winner,
    the original keeps None — the per-Player life mutation on the
    clone does not bleed back through the [Player; 2] copy."""
    cloned = synthetic_state.clone_once()
    cloned.lose_life(1, 20)
    assert cloned.winner() == 0
    assert synthetic_state.winner() is None


@pytest.mark.skip(
    reason=(
        "explicit-loss field deferred to future slice (pure-derivation "
        "option c chosen here to avoid editing state.rs while sibling "
        "agents are also editing it). When that slice lands, this test "
        "should pass: state.lose_game(0) flags player 0 as a loser "
        "regardless of life total — for 'you lose the game' effects "
        "like Phage / Bargain / decking out."
    )
)
def test_explicit_lose_game_marks_player(synthetic_state):
    """Locked-in API surface for the future explicit-loss slice."""
    synthetic_state.lose_game(0)
    assert list(synthetic_state.losers()) == [0]
    assert synthetic_state.winner() == 1
    assert synthetic_state.is_game_over() is True
