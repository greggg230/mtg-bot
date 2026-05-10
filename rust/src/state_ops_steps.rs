//! Phase / step advancement primitives on `GameState`.
//!
//! Mirrors `src/mtg_bot/turn.py:STEP_ORDER` and the simulation
//! driver that walks through it. The Python side carries a `Step`
//! enum and a global `STEP_ORDER` tuple; the Rust side keeps `phase`
//! as a `u8` field on `GameState` (already present from the
//! foundation commit) and constants live here at module scope.
//!
//! ## Phase encoding (option a — 12 raw u8 constants)
//!
//! Like the zones-slice's zone constants: a `Phase` enum on
//! `state.rs` would have forced an edit on a file two sibling TDD
//! slices were also growing in parallel, so the encoding stays
//! inline as `pub const u8` constants. Duplicated test-side in
//! `tests_rust/test_steps.py` for readability; a follow-up slice
//! can promote either side once parallel-slice constraints relax.
//!
//! Order matches Python's `STEP_ORDER`:
//!
//! UNTAP / UPKEEP / DRAW / PRECOMBAT_MAIN / BEGINNING_OF_COMBAT /
//! DECLARE_ATTACKERS / DECLARE_BLOCKERS / COMBAT_DAMAGE /
//! END_OF_COMBAT / POSTCOMBAT_MAIN / END / CLEANUP.
//!
//! ## Cross-slice composition (inlined workaround)
//!
//! `advance_phase` should fire the tap-slice's `untap_all_for_player`
//! sweep whenever the new phase is UNTAP — that's the load-bearing
//! integration test for the parallel-TDD harness composing.
//!
//! `multiple-pymethods` does let two sibling `#[pymethods] impl
//! GameState` blocks coexist, but methods defined in each block keep
//! their default Rust visibility (private to the module that owns
//! them). The tap-slice's `untap_all_for_player` is `fn`-private to
//! `state_ops_tap.rs`, so a direct call from this module hits E0624.
//! The brief's contingency for exactly this case is to INLINE the
//! equivalent body here. The inlined version mirrors the tap-slice
//! logic (skip the `Arc::make_mut` clone if every permanent is
//! already untapped; one CoW clone covers all the in-place flips).
//! When the parallel-slice constraint relaxes a follow-up can either
//! promote `untap_all_for_player` to `pub(crate)` or factor the sweep
//! into a free function — until then the duplication is the right
//! call (no edits to sibling files mid-parallel-development).

use crate::state::{GameObject, GameState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

// `#[allow(dead_code)]` on the constants only the test side reads —
// they're part of the published phase vocabulary so we don't want to
// drop them, but Rust-side code only references the boundaries
// (UNTAP / CLEANUP) and the predicate-test fences (MAIN1 / MAIN2 /
// BEGIN_COMBAT / END_COMBAT). Mirror of `STEP_ORDER` in Python's
// `src/mtg_bot/turn.py`.
pub const PHASE_UNTAP: u8 = 0;
#[allow(dead_code)]
pub const PHASE_UPKEEP: u8 = 1;
#[allow(dead_code)]
pub const PHASE_DRAW: u8 = 2;
pub const PHASE_MAIN1: u8 = 3;
pub const PHASE_BEGIN_COMBAT: u8 = 4;
#[allow(dead_code)]
pub const PHASE_DECLARE_ATTACKERS: u8 = 5;
#[allow(dead_code)]
pub const PHASE_DECLARE_BLOCKERS: u8 = 6;
#[allow(dead_code)]
pub const PHASE_COMBAT_DAMAGE: u8 = 7;
pub const PHASE_END_COMBAT: u8 = 8;
pub const PHASE_MAIN2: u8 = 9;
#[allow(dead_code)]
pub const PHASE_END_STEP: u8 = 10;
pub const PHASE_CLEANUP: u8 = 11;

const PHASE_MAX: u8 = PHASE_CLEANUP;

/// Free-function mirror of `state_ops_tap.rs:untap_all_for_player`.
/// See the module docstring for why this duplication exists. Same
/// CoW shape — short-circuits if every permanent on the target
/// battlefield is already untapped, otherwise one `Arc::make_mut`
/// covers the in-place sweep.
fn untap_all_for_player_inline(state: &mut GameState, player_id: u8) {
    let player = match state.players.iter_mut().find(|p| p.id == player_id) {
        Some(p) => p,
        None => return,
    };
    if player.battlefield.iter().all(|o| !o.tapped) {
        return;
    }
    let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
    for obj in zone.iter_mut() {
        obj.tapped = false;
    }
}

/// Clear `summoning_sick` on every permanent the given player
/// controls (CR 302.1 — summoning sickness wears off when the
/// creature has been continuously controlled by its controller
/// since the start of that player's most recent turn). Fires from
/// `advance_phase` whenever the new phase is UNTAP, on the new
/// active player. Same CoW shape as `untap_all_for_player_inline`:
/// short-circuit when nothing is sick so the synthetic-state's
/// "no sick permanents on battlefield" warm path doesn't pay an
/// `Arc::make_mut` clone, otherwise one make_mut covers the sweep.
fn clear_summoning_sick_for_player_inline(state: &mut GameState, player_id: u8) {
    let player = match state.players.iter_mut().find(|p| p.id == player_id) {
        Some(p) => p,
        None => return,
    };
    if player.battlefield.iter().all(|o| !o.summoning_sick) {
        return;
    }
    let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
    for obj in zone.iter_mut() {
        obj.summoning_sick = false;
    }
}

#[pymethods]
impl GameState {
    /// Expose `active_player` to Python. The field already exists on
    /// `GameState` (set by `make_synthetic_state` to 0) but had no
    /// `#[getter]` attached; this slice's tests need to read it to
    /// verify the cleanup-step turn rollover. `state.rs` is owned by
    /// other slices this commit, so the getter lands here instead.
    #[getter]
    fn active_player(&self) -> u8 {
        self.active_player
    }

    /// Read-only phase query. Matches the Python convention of
    /// returning the active phase as a value rather than a method
    /// dispatch on the enum.
    fn current_phase(&self) -> u8 {
        self.phase
    }

    /// Advance one step in the canonical phase order, returning the
    /// new phase. When the previous phase was CLEANUP (11), wrap
    /// back to UNTAP (0), bump `turn_number`, and swap
    /// `active_player`. When the new phase is UNTAP, automatically
    /// fire the untap-step sweep on the new active player by calling
    /// the tap-slice's `untap_all_for_player`. This auto-fire happens
    /// on every UNTAP entry, including the initial wrap that flips
    /// the active player — the new active player's permanents
    /// untap at the top of their turn.
    fn advance_phase(&mut self) -> u8 {
        if self.phase == PHASE_CLEANUP {
            self.phase = PHASE_UNTAP;
            self.turn_number += 1;
            // Two-player swap. If the engine ever grows past two
            // players, replace with `(active_player + 1) % n`.
            self.active_player = 1 - self.active_player;
        } else {
            self.phase += 1;
        }
        if self.phase == PHASE_UNTAP {
            // Inlined sibling-slice logic — see the workaround note
            // in the module docstring. Mirrors
            // `state_ops_tap.rs:untap_all_for_player`: skip the
            // `Arc::make_mut` clone when nothing is tapped, then do
            // a single CoW clone covering every in-place flip.
            untap_all_for_player_inline(self, self.active_player);
            // CR 302.1 — at the start of the active player's untap
            // step, every permanent that player has controlled
            // continuously since the start of their previous turn
            // loses summoning sickness. The synthetic-state seed
            // doesn't track "continuous control" yet (every newly
            // ETB'd permanent gets sickness via the cast slice and
            // resolve slice's `summoning_sick = true` stamps); the
            // operational rule reduces to "clear sickness on the
            // new active player's permanents at their UNTAP entry".
            // Same CoW shape as the untap sweep — see the helper.
            clear_summoning_sick_for_player_inline(self, self.active_player);
        }
        self.phase
    }

    /// True when the current phase is one of the two main phases
    /// (PRECOMBAT_MAIN / POSTCOMBAT_MAIN). Mirrors Python's
    /// `MAIN_STEPS` set in `turn.py`.
    fn is_main_phase(&self) -> bool {
        self.phase == PHASE_MAIN1 || self.phase == PHASE_MAIN2
    }

    /// True when the current phase is one of the five combat steps
    /// (BEGINNING_OF_COMBAT through END_OF_COMBAT inclusive).
    /// Mirrors Python's `COMBAT_STEPS` set in `turn.py`.
    fn is_combat_phase(&self) -> bool {
        (PHASE_BEGIN_COMBAT..=PHASE_END_COMBAT).contains(&self.phase)
    }

    /// Direct phase setter for tests / fast-forward scenarios.
    /// Validates the phase is in `[0, PHASE_CLEANUP]`. Production
    /// code should prefer `advance_phase` so the cross-slice
    /// composition fires; this setter intentionally does NOT trigger
    /// the untap step (that's `advance_phase`'s job — `set_phase` is
    /// raw state assignment).
    fn set_phase(&mut self, phase: u8) -> PyResult<()> {
        if phase > PHASE_MAX {
            return Err(PyValueError::new_err(format!(
                "phase out of range: {phase} (must be in [0, {PHASE_MAX}])"
            )));
        }
        self.phase = phase;
        Ok(())
    }
}
