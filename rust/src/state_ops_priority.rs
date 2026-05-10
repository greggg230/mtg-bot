//! Priority management primitives on `GameState`.
//!
//! Mirrors `src/mtg_bot/game.py:GameState.priority_player_idx` and
//! the priority-round mechanism in `src/mtg_bot/turn.py`. The Rust
//! side keeps `priority_player` as a `u8` field on `GameState`
//! (already present from the foundation commit) and exposes
//! Python-callable getters / setters and pass-priority semantics
//! here.
//!
//! ## Slice scope
//!
//! Pass-priority is a 2-player toggle: `priority_player` flips to
//! the other player. Multi-player priority (CR 800.4 / APNAP) is
//! out of scope until/unless the engine grows past 2 players.
//!
//! ## Out of scope (deferred to future slices)
//!
//! - **Composition with phase advancement.** CR 117.1a says the
//!   active player gets priority at the start of each step.
//!   `advance_phase` lives in `state_ops_steps.rs` and intentionally
//!   does NOT touch `priority_player` here — a future slice will
//!   integrate state_ops_priority + state_ops_steps so the
//!   active-player-gets-priority rule fires automatically. The
//!   test `test_priority_unaffected_by_phase_advance_in_this_slice`
//!   in `tests_rust/test_priority.py` pins this slice boundary.
//!
//! ## Round-9 extension: priority + stack composition
//!
//! `complete_priority_round` is the priority-round-completion
//! primitive. CR 117.4 / 117.5: when both players consecutively pass
//! priority while the stack is non-empty, the top stack item
//! resolves; CR 117.1c then says the active player gets priority
//! again. This file's caller is responsible for tracking whether
//! both players have passed (round-9 doesn't extend `GameState` with
//! a consecutive-pass counter — that would need a `state.rs` edit
//! which the parallel-slice file-isolation rule forbids). The caller
//! invokes `complete_priority_round()` once it's decided the round
//! is complete.
//!
//! ### Pop-only stepping-stone (NOT a full resolver)
//!
//! `complete_priority_round` does NOT dispatch the resolved item's
//! effects. It pops the top stack item, returns its dict, and resets
//! `priority_player` to `active_player`. Real effect dispatch lives
//! in `state_ops_resolve.rs:resolve_top_of_stack`.
//!
//! **Cross-file private-method friction.** This slice WANTS to call
//! `resolve_top_of_stack` directly so a single primitive does both
//! the bookkeeping AND the effect dispatch — but that method is
//! private to its module's `pyclass` impl block, and the parallel-
//! slice file-isolation rule forbids touching `state_ops_resolve.rs`
//! from this slice. Per the recurring inlining pattern (5+
//! instances: round-7 counterspell inlined `move_object`, round-8
//! resolve-effects inlined `draw_card`, etc.), the pragmatic call
//! here is to **inline a minimal version**: pop + reset priority, no
//! effect dispatch. That keeps the slice surface small AND
//! preserves a clean separation of concerns between
//! priority-bookkeeping (this file) and effect-dispatch
//! (`state_ops_resolve.rs`). A future composition slice can chain
//! the two — e.g. add a `resolve_and_complete_round()` umbrella —
//! once the file-isolation rule relaxes.

use crate::state::{GameState, StackItem};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build a 2-key dict from a StackItem. Mirrors the helper in
/// `state_ops_stack.rs` and `state_ops_resolve.rs` — duplicated here
/// per the per-slice file-isolation rule (private helpers can't be
/// shared across slice files until the consolidation slice).
fn stack_item_to_dict<'py>(
    py: Python<'py>,
    item: &StackItem,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source_obj_id", item.source_obj_id)?;
    d.set_item("controller", item.controller)?;
    Ok(d)
}

#[pymethods]
impl GameState {
    /// Read the player who currently has priority. Returns the raw
    /// `u8` player id (0 or 1 in the 2-player model).
    fn get_priority_player(&self) -> u8 {
        self.priority_player
    }

    /// Pass priority to the other player. In the 2-player model
    /// this is a simple toggle (`1 - priority_player`). Returns the
    /// new priority-holder's id so callers can chain off the result
    /// without a follow-up `get_priority_player()` call.
    ///
    /// Note: this only moves `priority_player`; `active_player`
    /// stays put. Active-player rotation is the job of the
    /// turn-rollover path in `state_ops_steps.rs:advance_phase`.
    fn pass_priority(&mut self) -> u8 {
        self.priority_player = 1 - self.priority_player;
        self.priority_player
    }

    /// Direct priority setter for tests / fast-forward scenarios.
    /// Validates the player_id is in {0, 1}; raises `ValueError`
    /// otherwise rather than silently corrupting state.
    fn give_priority_to(&mut self, player_id: u8) -> PyResult<()> {
        if player_id > 1 {
            return Err(PyValueError::new_err(format!(
                "player_id out of range: {player_id} (must be 0 or 1)"
            )));
        }
        self.priority_player = player_id;
        Ok(())
    }

    /// Round-9 priority + stack composition primitive.
    ///
    /// Caller signals "both players have consecutively passed
    /// priority" by invoking this method. Behavior:
    /// - **Empty stack** -> return `None`. No mutation. (CR 117.4
    ///   would end the current step here, but step-ending is the job
    ///   of `state_ops_steps.rs:advance_phase` — a future composition
    ///   slice wires these together.)
    /// - **Non-empty stack** -> pop the top item, reset
    ///   `priority_player` to `active_player` (CR 117.1c), and return
    ///   the popped item's dict (`{source_obj_id, controller}`).
    ///
    /// **This method does NOT dispatch effects.** Real MTG resolves
    /// the popped spell/ability's effects (deal damage, draw cards,
    /// etc.); this is a pop-only stepping-stone for the priority-
    /// round bookkeeping. Effect dispatch lives in
    /// `state_ops_resolve.rs:resolve_top_of_stack` — see the module
    /// docstring above for the cross-file-friction rationale and
    /// future-consolidation note.
    fn complete_priority_round<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Empty stack -> None, no mutation.
        let Some(popped) = self.stack.pop() else {
            return Ok(None);
        };
        // CR 117.1c: AP gets priority after a spell or ability
        // resolves. (We're a stepping-stone: the "after resolves"
        // here is "after the top is popped"; a future slice that
        // adds effect dispatch keeps this same priority reset.)
        self.priority_player = self.active_player;
        Ok(Some(stack_item_to_dict(py, &popped)?))
    }
}
