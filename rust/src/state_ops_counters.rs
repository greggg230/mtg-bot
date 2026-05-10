//! Counter primitives on `GameState`.
//!
//! Filled in by a TDD-slice branch (see `tests_rust/README.md`).
//! Stub block exists so other slices can grow `GameState` in
//! parallel without conflicting on a single `#[pymethods]` block.
//!
//! `GameObject.counters: i32` already exists (per-instance scalar);
//! the first slice models +1/+1 (positive) and -1/-1 (negative)
//! as a single net count, since they cancel at SBAs. Future slices
//! extend to a proper multi-kind map once a card needs charge / lore
//! / loyalty / brick / etc.
//!
//! ## Net-i32 model and the SBA-cancellation TODO
//!
//! The Python engine stores `plus_one_counters` and `minus_one_counters`
//! as separate fields and cancels them pairwise inside
//! `state_based_actions` (CR 704.5p). The Rust port collapses both
//! into a single signed integer: `+N` ≡ N net +1/+1 counters,
//! `-N` ≡ N net -1/-1 counters. That captures the post-cancellation
//! amount that contributes to effective P/T, which is the only thing
//! every downstream caller actually reads. The price is that "how
//! many distinct +1/+1 counters does this creature have right now"
//! becomes inexpressible — fine until a card asks (e.g. an effect
//! that doubles +1/+1 counters specifically), which is a future slice.
//!
//! TODO: when SBA enforcement lands on the Rust side, callers that
//! mutate counters can stop pre-cancelling via the i32 representation
//! and the SBA pass can do its own bookkeeping. Today this primitive
//! is the entire counter system; the i32 IS the post-SBA net value.
//!
//! ## Read locations
//!
//! +1/+1 counters can technically live in any zone (Athreos returns
//! cards from the graveyard with counter triggers, time counters tick
//! on suspended cards in exile, etc.) — so the lookup walks every
//! tracked zone, not just battlefield. This differs from
//! `state_ops_tap.rs::is_tapped`, which restricts to battlefield
//! because only battlefield permanents have a defined "tapped" state.

use crate::state::{GameObject, GameState};
use pyo3::prelude::*;
use std::sync::Arc;

/// Walk every player's four tracked zones (hand, library, battlefield,
/// graveyard) and apply `f` to the matching object via `Arc::make_mut`
/// on the containing zone Vec. Returns `true` if an object with
/// `obj_id` was found (regardless of whether `f` actually mutated it),
/// `false` if no matching object exists.
///
/// Mirrors the shape of `state_ops_tap.rs::mutate_battlefield_obj`,
/// scoped to all-zones. Lives in this file (not shared with `_zones`)
/// per the slice-isolation rule — refactoring into a shared helper is
/// a follow-up once the parallel-slice constraint relaxes.
fn mutate_object_anywhere<F>(state: &mut GameState, obj_id: u32, mut f: F) -> bool
where
    F: FnMut(&mut GameObject),
{
    for player in state.players.iter_mut() {
        // Hand.
        if player.hand.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.hand);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
        // Library.
        if player.library.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.library);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
        // Battlefield.
        if player.battlefield.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
        // Graveyard.
        if player.graveyard.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.graveyard);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
    }
    false
}

/// Read-only sibling of `mutate_object_anywhere`. Walks the four
/// tracked zones on both players and returns the matching object's
/// `counters` field, or `None` if no such object exists.
fn read_counters(state: &GameState, obj_id: u32) -> Option<i32> {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    return Some(obj.counters);
                }
            }
        }
    }
    None
}

#[pymethods]
impl GameState {
    /// Read the net counter value on the object with `obj_id`.
    ///
    /// Returns `Some(n)` where `n` is the signed net count
    /// (+ for +1/+1, - for -1/-1) if the object exists in any
    /// tracked zone; `None` otherwise. Distinguishes "object not
    /// found" from "object found and at zero counters" — callers
    /// validating a target before applying an effect rely on that.
    fn get_counters(&self, obj_id: u32) -> Option<i32> {
        read_counters(self, obj_id)
    }

    /// Write the absolute counter value on the object with `obj_id`.
    ///
    /// Returns `true` if an object with that id was found and
    /// updated, `false` if no matching object exists in any tracked
    /// zone. Like `tap_object`, this is a silent no-op rather than
    /// an exception — matches the Python convention for effects
    /// whose targeted permanent has already left the relevant zone.
    fn set_counters(&mut self, obj_id: u32, value: i32) -> bool {
        mutate_object_anywhere(self, obj_id, |obj| {
            obj.counters = value;
        })
    }

    /// Increment (or decrement, for negative `delta`) the counter
    /// value on the object with `obj_id`.
    ///
    /// Returns `true` if an object with that id was found and
    /// updated, `false` if no matching object exists. The value can
    /// freely cross zero in either direction — see the module
    /// docstring on the net-i32 model.
    fn add_counters(&mut self, obj_id: u32, delta: i32) -> bool {
        mutate_object_anywhere(self, obj_id, |obj| {
            obj.counters = obj.counters.saturating_add(delta);
        })
    }

    /// The +1/+1 contribution to effective power.
    ///
    /// For the net-i32 model this is just the counter value. Real
    /// effective-power composition (base + counters + auras +
    /// ContinuousModifiers) is a future slice; this method returns
    /// the counter contribution alone. Returns `None` if no object
    /// with `obj_id` exists.
    fn power_bonus_from_counters(&self, obj_id: u32) -> Option<i32> {
        read_counters(self, obj_id)
    }

    /// The +1/+1 contribution to effective toughness.
    ///
    /// Symmetric with `power_bonus_from_counters` because +1/+1
    /// (and -1/-1) counters affect both stats equally. The two
    /// methods exist as distinct names so a future slice that
    /// introduces asymmetric counter types (+1/+0 etc.) has a
    /// clean upgrade path.
    fn toughness_bonus_from_counters(&self, obj_id: u32) -> Option<i32> {
        read_counters(self, obj_id)
    }
}
