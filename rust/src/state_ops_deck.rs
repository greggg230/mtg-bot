//! Deck-loading primitives on `GameState`.
//!
//! Filled in by a TDD-slice branch (see `tests_rust/README.md`).
//! Stub block exists so other slices can grow `GameState` in
//! parallel without conflicting on a single `#[pymethods]` block.
//!
//! First-slice scope: `load_deck(player_id, [printing_id, ...])`
//! replaces the player's library with fresh GameObjects (one per
//! printing_id in the input list, in order); top-of-library = LAST
//! element of the Vec to mirror the round-1 zones convention.
//! Composes with the round-4 library shuffle and round-1 zones.
//!
//! ## Fresh-id allocation
//!
//! Each new GameObject must have an id that doesn't collide with any
//! existing GameObject in any zone on either player. The simplest
//! correct allocator scans every zone for the maximum id in use, then
//! allocates sequential ids starting at `max + 1`. The cost is O(N)
//! over total objects; deck loads happen at game setup, not in the
//! hot loop, so the linear scan is fine.
//!
//! ## Validation
//!
//! Both `player_id` (must be 0 or 1) and every entry of `printing_ids`
//! (must point at a registry slot, i.e. `< registry.len()`) are
//! checked up-front; on any validation failure we raise `PyValueError`
//! WITHOUT mutating the library. The whole-or-nothing contract is the
//! `*_invalid_*_raises` tests' load-bearing invariant.
//!
//! ## CoW
//!
//! The new library Vec replaces the old one via `Arc::make_mut` +
//! `*lib = new_vec`. Cloning a `GameState` and then loading a deck on
//! the clone leaves the source's library untouched — verified by
//! `test_clone_then_load_deck_does_not_affect_original`.
//!
//! ## No automatic shuffle
//!
//! `load_deck` builds the library in input order (last input = top
//! of library). Callers wanting randomization invoke
//! `shuffle_library_with_seed` from the round-4 library slice.

use crate::state::{GameObject, GameState};
use crate::state_ops_zones::ZONE_LIBRARY;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl GameState {
    /// Replace `player_id`'s library with fresh GameObjects, one per
    /// entry of `printing_ids` (in input order, last input = top of
    /// library). Each new object gets a unique id allocated above the
    /// current max id across all zones on both players.
    ///
    /// Raises `ValueError` if `player_id` is not 0 or 1, or if any
    /// entry of `printing_ids` is out of registry range. On validation
    /// failure no state mutation occurs (whole-or-nothing).
    fn load_deck(
        &mut self,
        player_id: u8,
        printing_ids: Vec<u32>,
    ) -> PyResult<()> {
        // Validation 1: player_id in range.
        let idx = player_id as usize;
        if idx >= 2 {
            return Err(PyValueError::new_err(format!(
                "player_id {player_id} out of range (must be 0 or 1)"
            )));
        }
        // Validation 2: every printing_id points at a real slot.
        let reg_len = self.registry.len() as u32;
        for &pid in &printing_ids {
            if pid >= reg_len {
                return Err(PyValueError::new_err(format!(
                    "printing_id {pid} out of range \
                    (registry has {reg_len} printings)"
                )));
            }
        }

        // Allocate fresh ids: scan every zone on both players for
        // the current max id, allocate starting at max + 1.
        let mut max_id: u32 = 0;
        let mut found_any = false;
        for player in &self.players {
            for zone in [
                &player.hand,
                &player.library,
                &player.battlefield,
                &player.graveyard,
            ] {
                for obj in zone.iter() {
                    if !found_any || obj.id > max_id {
                        max_id = obj.id;
                        found_any = true;
                    }
                }
            }
        }
        let mut next_id: u32 = if found_any { max_id + 1 } else { 0 };

        // Build the new library Vec.
        let mut new_lib: Vec<GameObject> = Vec::with_capacity(printing_ids.len());
        for pid in printing_ids {
            new_lib.push(GameObject {
                id: next_id,
                printing_id: pid,
                controller: player_id,
                zone: ZONE_LIBRARY,
                tapped: false,
                summoning_sick: false,
                damage: 0,
                counters: 0,
            });
            next_id += 1;
        }

        // Replace the library through Arc::make_mut so cloning a
        // GameState and then loading a deck on the clone doesn't
        // mutate the source's Vec.
        let player = &mut self.players[idx];
        let lib = Arc::make_mut(&mut player.library);
        *lib = new_lib;
        Ok(())
    }
}
