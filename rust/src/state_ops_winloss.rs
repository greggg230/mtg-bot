//! Win / loss condition primitives on `GameState`.
//!
//! Composes with the round-1 life slice — a player with life <= 0
//! loses (CR 104.3a, simplified). Future slices add the other loss
//! conditions (deck-out 104.3b, poison counters 104.3c, etc.).
//!
//! **Implementation choice — pure derivation (option c).** ``losers``
//! is computed purely from current life totals (life <= 0). There is
//! NO ``Player.has_lost`` field on the Rust ``Player`` struct in this
//! slice — adding such a field would require editing
//! ``rust/src/state.rs``, which is shared with sibling TDD-slice
//! agents (damage, cards-bridge) running in parallel against the same
//! base commit. Doing so would cause merge conflicts.
//!
//! Consequently, "you lose the game" effects (Phage the Untouchable,
//! Bargain, decking-out, etc.) are deferred to a future slice that
//! can add the explicit-loss field at a moment when ``state.rs`` is
//! not being shared with siblings. The test
//! ``test_explicit_lose_game_marks_player`` in
//! ``tests_rust/test_winloss.py`` stays in place as
//! ``@pytest.mark.skip`` to lock in the API surface.
//!
//! **Winner semantics in a 2-player game.**
//! - 0 losers -> winner = None (game ongoing).
//! - exactly 1 loser, the OTHER player not also a loser -> winner =
//!   Some(other_player_id).
//! - 2 losers (e.g. mutual lethal damage) -> winner = None
//!   (the "draw" case under pure-derivation semantics).
//!
//! **Pure-derivation caveat for life-back-above-zero.** Real MTG would
//! fire the loss SBA the moment life drops to 0, which would lock in
//! the loss before any subsequent gain_life can revive the player. In
//! this slice there is no SBA loop, so calling ``gain_life`` on a
//! 0-life player does take them out of ``losers()``. The SBA-driven
//! "loss is permanent" model lands with the explicit-loss field in a
//! future slice.

use crate::state::GameState;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

fn check_player_id(player_id: u8) -> PyResult<usize> {
    let idx = player_id as usize;
    if idx >= 2 {
        return Err(PyIndexError::new_err(format!(
            "player_id {player_id} out of range (must be 0 or 1)"
        )));
    }
    Ok(idx)
}

#[pymethods]
impl GameState {
    /// Return the list of player ids currently in a losing state.
    /// Pure-derivation: a player is a loser iff ``life <= 0``.
    /// Sorted by player id (0 before 1) for deterministic output.
    fn losers(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for p in &self.players {
            if p.life <= 0 {
                out.push(p.id);
            }
        }
        out
    }

    /// Return ``Some(player_id)`` of the unique winner, else ``None``.
    /// Winner is the unique non-loser when exactly one player has
    /// lost. ``None`` when no losers (game ongoing) or when both
    /// players have lost (draw).
    fn winner(&self) -> Option<u8> {
        let p0_lost = self.players[0].life <= 0;
        let p1_lost = self.players[1].life <= 0;
        match (p0_lost, p1_lost) {
            (false, true) => Some(self.players[0].id),
            (true, false) => Some(self.players[1].id),
            // (false, false): no loss yet.
            // (true, true):  simultaneous loss -> no winner.
            _ => None,
        }
    }

    /// True iff at least one player has lost. Note: in the
    /// pure-derivation slice, "game over" includes the simultaneous
    /// double-loss case (a draw is still a finished game).
    fn is_game_over(&self) -> bool {
        self.players[0].life <= 0 || self.players[1].life <= 0
    }

    /// True iff ``player_id`` has life > 0. Raises ``IndexError`` if
    /// ``player_id`` is not 0 or 1.
    fn is_player_alive(&self, player_id: u8) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        Ok(self.players[idx].life > 0)
    }
}
