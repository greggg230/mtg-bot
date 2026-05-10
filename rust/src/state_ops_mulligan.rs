//! Mulligan / opening-hand primitives on `GameState`.
//!
//! Filled in by a TDD-slice branch (see `tests_rust/README.md`).
//! Stub block exists so other slices can grow `GameState` in
//! parallel without conflicting on a single `#[pymethods]` block.
//!
//! ## Slice scope
//!
//! `mulligan(player_id, seed)` moves all cards in the player's hand
//! BACK to the library, shuffles the library with `seed`, then draws
//! 7 cards (or until the library is empty). The caller separately
//! tracks the mulligan count and (per CR 103.4 / London mulligan)
//! chooses which N cards to put on the bottom afterwards — that
//! "put N to bottom" step is a separate primitive that this slice
//! does not yet provide (deferred to a future slice).
//!
//! There is no separate `opening_hand` convenience: the synthetic
//! state seeds 7 cards in hand already, so an opening-hand call is
//! logically a hand-replace, which is what `mulligan` does. One
//! primitive covers both the at-game-start and iterative-mulligan
//! call sites.
//!
//! ## Cross-file friction (inlined logic)
//!
//! `shuffle_library_with_seed` (round-4 library) and `draw_card`
//! (round-1 zones) are pymethods, exposed to Python but Rust-private.
//! Calling them from this module would require either making them
//! `pub` (touches sibling files in this round) or duplicating their
//! logic here. Per the recurring round pattern, this slice inlines
//! both:
//!
//!   - The xorshift64 RNG + Fisher-Yates shuffle (mirroring
//!     `state_ops_library.rs::shuffle_library_with_seed`).
//!   - The pop-from-library-tail + push-to-hand draw step
//!     (mirroring `state_ops_zones.rs::draw_card`).
//!
//! When a later slice consolidates the public surface, the obvious
//! refactor is to lift the helpers into a private module both files
//! can depend on without growing `pub` API. Documented here so the
//! duplication is visible to the next round.
//!
//! ## CoW
//!
//! All mutation flows through `Arc::make_mut` on the touched zones
//! (the player's hand and library). Cloning a `GameState` and then
//! calling `mulligan` on the clone leaves the source state's hand
//! and library untouched — verified by
//! `test_clone_then_mulligan_does_not_affect_original`.

use crate::state::GameState;
use crate::state_ops_zones::{ZONE_HAND, ZONE_LIBRARY};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Inline xorshift64 — duplicated from `state_ops_library.rs`. See
/// the module docstring for the cross-file-friction rationale.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0xdead_beef_dead_beef } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize_below(&mut self, n: usize) -> usize {
        let n_u64 = n as u64;
        let bound = u64::MAX - (u64::MAX % n_u64);
        loop {
            let r = self.next_u64();
            if r < bound {
                return (r % n_u64) as usize;
            }
        }
    }
}

#[pymethods]
impl GameState {
    /// Move every card in `player_id`'s hand back to that player's
    /// library, shuffle the library with `seed`, then draw 7 cards
    /// (or until the library is empty). Each touched object's `zone`
    /// field is updated to match its destination.
    ///
    /// Raises `ValueError` if `player_id` is not 0 or 1.
    ///
    /// CoW-safe: hand + library mutations both route through
    /// `Arc::make_mut`. Same `(state, seed)` produces the same
    /// post-mulligan state, byte for byte (determinism guarantee).
    fn mulligan(&mut self, player_id: u8, seed: u64) -> PyResult<()> {
        let idx = player_id as usize;
        if idx >= 2 {
            return Err(PyValueError::new_err(format!(
                "player_id {player_id} out of range (must be 0 or 1)"
            )));
        }
        let player = &mut self.players[idx];

        // Step 1: drain the hand into the library. Each returned
        // object's zone field is restamped to ZONE_LIBRARY so a
        // post-mulligan find_object_zone read of the field agrees
        // with where the object actually lives. Scoped so the &mut
        // borrows of player.hand / player.library don't outlast the
        // step (we need a fresh &mut player.hand in step 3).
        {
            let hand = Arc::make_mut(&mut player.hand);
            let returning: Vec<_> = hand.drain(..).collect();
            let lib = Arc::make_mut(&mut player.library);
            for mut obj in returning {
                obj.zone = ZONE_LIBRARY;
                lib.push(obj);
            }

            // Step 2: Fisher-Yates shuffle of the library (no-op when
            // n < 2). Mirrors state_ops_library.rs's shuffle.
            let n = lib.len();
            if n >= 2 {
                let mut rng = XorShift64::new(seed);
                for i in (1..n).rev() {
                    let j = rng.next_usize_below(i + 1);
                    lib.swap(i, j);
                }
            }
        }

        // Step 3: draw 7 (or until empty). Top of library = LAST
        // element of the Vec, matching state_ops_zones.rs's draw_card
        // convention.
        let hand = Arc::make_mut(&mut player.hand);
        let lib = Arc::make_mut(&mut player.library);
        for _ in 0..7 {
            let Some(mut obj) = lib.pop() else {
                break;
            };
            obj.zone = ZONE_HAND;
            hand.push(obj);
        }
        Ok(())
    }
}
