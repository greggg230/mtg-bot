//! Cleanup-step primitives on `GameState`.
//!
//! Filled in by a TDD-slice branch (see `tests_rust/README.md`).
//! Stub block exists so other slices can grow `GameState` in
//! parallel without conflicting on a single `#[pymethods]` block.
//!
//! First-slice scope: hand-size limit enforcement at cleanup
//! (CR 514.1 — discard down to maximum hand size, default 7).
//! Composes with state_ops_library's `discard` and the round-2
//! `state_ops_steps`'s phase encoding.
//!
//! Round-8 extension: a composition primitive
//! `cleanup_step_for_active_player()` that runs the two halves of
//! CR 514 we model — hand-size discard (514.1) for the active
//! player followed by clearing all marked damage (514.2) on both
//! players' battlefields. Returns a summary dict
//! `{"discarded": N, "damage_cleared": M}` so callers can log /
//! assert the per-step counts without re-walking zones.
//!
//! Ordering: hand-size discard FIRST, then damage clearance — the
//! numbered sub-rules of CR 514 execute in order (514.1 before
//! 514.2). Mirrors the source-of-truth `run_cleanup_step` in
//! `src/mtg_bot/turn.py` (hand-size while-loop precedes the
//! damage-clear loop). Matters once "discard triggers" arrive: a
//! hand-size discard could trigger an ability that pings a
//! creature, and that ping's damage is then cleared by 514.2 in
//! the same cleanup step.
//!
//! Out of scope (CR 514.3 / 514.4):
//!  - 514.3 "until end of turn" effects expiration — composes with
//!    continuous-effect machinery the engine doesn't have yet.
//!  - 514.4 "no abilities triggered? skip the second cleanup step"
//!    — composes with triggered-ability machinery.
//!
//! ## Discard target choice (deterministic — stepping-stone)
//!
//! ## Discard target choice (deterministic — stepping-stone)
//!
//! Real MTG: when forced to discard at cleanup, the active player
//! chooses which cards to discard. This slice's primitive deterministically
//! pops from the END of the hand Vec (most recently drawn). Once the
//! search side has a discard-decision node, a future slice can take
//! a vector of obj_ids to discard rather than picking blindly. The
//! deterministic choice is good enough for state-evolution parity
//! tests today and lets the cleanup step make forward progress in
//! search rollouts. See the same shortcut in
//! `src/mtg_bot/turn.py:run_cleanup_step` (`ap.hand.pop()` with no
//! decision node), which is what we're matching.
//!
//! ## CoW
//!
//! Mutates two zones (hand + graveyard) on the same player. Both
//! routed through `Arc::make_mut`, preserving the CoW invariant.
//! `test_clone_then_discard_does_not_affect_original` is the canary.
//! The round-8 composition adds a third mutation — battlefield
//! damage markers — so its CoW canary
//! (`test_clone_then_cleanup_does_not_affect_original`) extends the
//! check across all three.
//!
//! ## Cross-slice composition (inlined logic)
//!
//! `cleanup_step_for_active_player` ought to delegate to the
//! existing `enforce_hand_size_limit` (this file) and
//! `clear_all_damage` (state_ops_damage.rs). PyO3's
//! `multiple-pymethods` lets the two `#[pymethods]` blocks coexist
//! but the methods stay Rust-private to their owning modules.
//! `enforce_hand_size_limit` is callable from this file (same
//! module) but `clear_all_damage` is not — so the damage half is
//! INLINED here, mirroring the two-loop shape of
//! `state_ops_damage.rs::clear_damage_for_player` (skip
//! Arc::make_mut when no permanent has damage; one CoW clone
//! covers the per-player sweep). Same friction the round-7
//! counterspell agent and the round-2 untap-step composition
//! described — until parallel-slice constraints relax, the
//! duplication is deliberate. A follow-up consolidation can promote
//! `clear_all_damage` to `pub(crate)` (or factor into a free
//! function) and wire this primitive through it.

use crate::state::{GameObject, GameState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pymethods]
impl GameState {
    /// Enforce the hand-size limit for `player_id`: while the hand
    /// is larger than `max_hand_size`, pop a card from the end of
    /// hand and push it onto that player's graveyard. Returns the
    /// number of cards discarded.
    ///
    /// Validates `player_id` is in `[0, 1]` and `max_hand_size >= 0`;
    /// raises `ValueError` otherwise. (No silent no-op — invalid
    /// inputs at this layer indicate a caller bug.)
    ///
    /// CoW-safe: each zone mutation routes through `Arc::make_mut`,
    /// so cloning a `GameState` and then enforcing on the clone
    /// leaves the source's hand and graveyard untouched.
    fn enforce_hand_size_limit(
        &mut self,
        player_id: u8,
        max_hand_size: i32,
    ) -> PyResult<usize> {
        if player_id >= 2 {
            return Err(PyValueError::new_err(format!(
                "player_id must be 0 or 1, got {player_id}"
            )));
        }
        if max_hand_size < 0 {
            return Err(PyValueError::new_err(format!(
                "max_hand_size must be >= 0, got {max_hand_size}"
            )));
        }
        let max = max_hand_size as usize;
        let idx = player_id as usize;
        let player = &mut self.players[idx];
        let current = player.hand.len();
        if current <= max {
            return Ok(0);
        }
        let to_discard = current - max;
        // Mutating both zones: route each through Arc::make_mut so
        // the CoW invariant holds.
        let hand = Arc::make_mut(&mut player.hand);
        let mut popped: Vec<crate::state::GameObject> = Vec::with_capacity(to_discard);
        for _ in 0..to_discard {
            // pop() from the END of the Vec — see module docstring
            // on the deterministic-choice stepping-stone.
            let mut obj = hand
                .pop()
                .expect("hand.len() > max guaranteed by current > max check above");
            obj.zone = crate::state_ops_zones::ZONE_GRAVEYARD;
            popped.push(obj);
        }
        let graveyard = Arc::make_mut(&mut player.graveyard);
        for obj in popped {
            graveyard.push(obj);
        }
        Ok(to_discard)
    }

    /// Run the cleanup step for the current `active_player` —
    /// composition of CR 514.1 (hand-size discard, max=7 hard-coded)
    /// and CR 514.2 (clear all marked damage on both battlefields).
    ///
    /// Order: hand-size discard FIRST, damage clearance second
    /// (matches CR 514's numbered sub-rules and the source-of-truth
    /// loop in `src/mtg_bot/turn.py:run_cleanup_step`). See module
    /// docstring on why ordering matters.
    ///
    /// Returns a Python dict `{"discarded": N, "damage_cleared": M}`:
    /// - `discarded`: number of cards moved from active player's
    ///   hand to graveyard (0 if hand was already <= 7).
    /// - `damage_cleared`: count of permanents (across both
    ///   battlefields) whose marked damage was non-zero before the
    ///   call. A no-op cleanup (no damage anywhere) reports 0.
    ///
    /// Hard-codes max-hand-size = 7 because every active-player
    /// cleanup in canonical MTG uses 7. The lower-level
    /// `enforce_hand_size_limit(player_id, max)` stays available for
    /// callers that need a different cap (Reliquary Tower sentinel
    /// once we add a no-max representation).
    ///
    /// Does NOT advance the phase or check whether the engine is
    /// actually IN the cleanup phase — the caller picks when to
    /// fire this. Same shape as the round-2 untap composition: a
    /// primitive that the phase machinery layers on top of, not
    /// the phase-transition itself.
    fn cleanup_step_for_active_player<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // (1) CR 514.1 — hand-size discard for the active player.
        let active = self.active_player;
        let discarded = self.enforce_hand_size_limit(active, 7)?;

        // (2) CR 514.2 — clear all marked damage on both battlefields.
        // Inlined per the cross-slice composition note in the module
        // docstring. Two-pass shape per player: first count the
        // permanents with non-zero damage (so the summary is
        // accurate), then if any, route through `Arc::make_mut`
        // exactly once and zero them in place. Skipping the CoW
        // clone when nothing has damage matches the optimization
        // in `state_ops_damage.rs::clear_damage_for_player`.
        let mut damage_cleared: usize = 0;
        for player in self.players.iter_mut() {
            let count = player.battlefield.iter().filter(|o| o.damage != 0).count();
            if count == 0 {
                continue;
            }
            damage_cleared += count;
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                obj.damage = 0;
            }
        }

        let d = PyDict::new(py);
        d.set_item("discarded", discarded)?;
        d.set_item("damage_cleared", damage_cleared)?;
        Ok(d)
    }
}
