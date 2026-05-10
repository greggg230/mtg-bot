//! Zone-movement primitives on `GameState`.
//!
//! First TDD slice on the engine port. Owns the per-object membership
//! moves (battlefield -> graveyard, library -> hand on draw, etc.) and
//! a couple of read helpers (`find_object_zone`, `graveyard_sizes`)
//! that the move tests need to assert against.
//!
//! ## Zone identifier encoding (option b)
//!
//! Zones are represented by raw `u8` constants exposed as `pub const`
//! at module scope. The alternative — a `ZoneKind` enum on
//! `state.rs` — would have forced an edit on a file two sibling TDD
//! slices also need to grow this commit, so we keep the encoding
//! inline. The constants are duplicated in `tests_rust/test_zones.py`
//! for readability; a follow-up slice can either expose them as
//! module-level Python attrs on `mtg_bot_engine` or upgrade to an
//! enum once the parallel-slice constraint relaxes.
//!
//! ## "Top of library" convention
//!
//! `draw_card` pops the LAST element of the library `Vec` and pushes
//! it onto the hand. This matches `Vec::pop`'s O(1) semantics. The
//! Python engine uses `library.pop(0)` (front = top) — that's an O(n)
//! shift on each draw and an artifact of `list` having no fast
//! pop-from-end-as-top idiom in idiomatic Python. The Rust port flips
//! the orientation; once a Rust-side initializer for libraries lands
//! the seed order can be reversed to keep "top of library" intuitive
//! in production code. For the first slice this is a documented
//! convention rather than a bug.

use crate::state::{GameObject, GameState, Zone};
use pyo3::prelude::*;
use std::sync::Arc;

pub const ZONE_HAND: u8 = 0;
pub const ZONE_LIBRARY: u8 = 1;
pub const ZONE_BATTLEFIELD: u8 = 2;
pub const ZONE_GRAVEYARD: u8 = 3;
#[allow(dead_code)]
pub const ZONE_STACK: u8 = 4;
#[allow(dead_code)]
pub const ZONE_EXILE: u8 = 5;

/// Find an object in `zone` by id; if present, remove it (preserving
/// order) and return the owned `GameObject`. CoW-safe: takes
/// `&mut Zone` (= `&mut Arc<Vec<GameObject>>`) and goes through
/// `Arc::make_mut`, so the inner `Vec` is cloned on first write only
/// when other clones are observing it.
fn take_from_zone(zone: &mut Zone, obj_id: u32) -> Option<GameObject> {
    let pos = zone.iter().position(|o| o.id == obj_id)?;
    let v = Arc::make_mut(zone);
    Some(v.remove(pos))
}

#[pymethods]
impl GameState {
    /// Mirror of `battlefield_sizes` / `hand_sizes` for the graveyard
    /// zone. Lives here rather than `state.rs` because the
    /// graveyard-side assertions land with the first move-into-
    /// graveyard test in this slice.
    #[getter]
    fn graveyard_sizes(&self) -> [usize; 2] {
        [
            self.players[0].graveyard.len(),
            self.players[1].graveyard.len(),
        ]
    }

    /// Resolve an object id to the zone constant naming the zone it
    /// currently lives in. Returns `None` if no object with that id
    /// is in any of the five tracked zones (hand / library /
    /// battlefield / graveyard / exile) on either player. The stack
    /// is still not searched (a stack item is keyed by source id,
    /// not by full GameObject membership) — once it is, the search
    /// extends here.
    ///
    /// **Round-14 update:** exile is now searched. Round-1 ignored
    /// exile because nothing read FROM exile yet; round-11 added
    /// `Effect::ExileTarget` (write into exile) and this slice is
    /// the matching "exile is now a first-class read source" half.
    ///
    /// The lookup is membership-based, NOT a read of the object's
    /// `zone` field. That field carries placeholder data on
    /// `make_synthetic_state`-built objects (see `make_object` in
    /// `state.rs` — every fresh object is stamped `zone: 1`
    /// regardless of where it actually lands). `move_object` keeps
    /// the field in sync going forward; this method is the source of
    /// truth either way.
    fn find_object_zone(&self, obj_id: u32) -> Option<u8> {
        for player in &self.players {
            if player.hand.iter().any(|o| o.id == obj_id) {
                return Some(ZONE_HAND);
            }
            if player.library.iter().any(|o| o.id == obj_id) {
                return Some(ZONE_LIBRARY);
            }
            if player.battlefield.iter().any(|o| o.id == obj_id) {
                return Some(ZONE_BATTLEFIELD);
            }
            if player.graveyard.iter().any(|o| o.id == obj_id) {
                return Some(ZONE_GRAVEYARD);
            }
            if player.exile.iter().any(|o| o.id == obj_id) {
                return Some(ZONE_EXILE);
            }
        }
        None
    }

    /// Move the object with id `obj_id` out of its current zone and
    /// into `target_zone` on player `target_player`. Updates the
    /// object's `zone` field to `target_zone` so a later
    /// `find_object_zone` agrees with whatever inspects the field
    /// directly. Returns `True` on success, `False` if no object with
    /// that id exists in any tracked zone.
    ///
    /// CoW-safe: each zone mutation goes through `Arc::make_mut`, so
    /// cloning a `GameState` and then calling `move_object` on the
    /// clone leaves the source state's zone Vecs untouched.
    ///
    /// **Round-14 update:** the source-zone search now also iterates
    /// exile, and `ZONE_EXILE` is a valid destination — so callers
    /// can move an object INTO exile (sidestepping the round-11
    /// ExileTarget effect dispatch when the bare primitive is what's
    /// needed) and OUT of exile back to a tracked zone. Pre-r14 the
    /// dispatch silently routed unknown destinations (incl. exile)
    /// into the graveyard; that fallback remains for any other
    /// non-modeled value (including the stack, which still has no
    /// per-object zone Vec). The module docstring covers the
    /// stack limitation.
    fn move_object(
        &mut self,
        obj_id: u32,
        target_zone: u8,
        target_player: u8,
    ) -> bool {
        // Find + remove from source zone (search both players, all five
        // tracked zones — round-14 added exile to the iteration).
        let mut taken: Option<GameObject> = None;
        for player in self.players.iter_mut() {
            if let Some(obj) = take_from_zone(&mut player.hand, obj_id) {
                taken = Some(obj);
                break;
            }
            if let Some(obj) = take_from_zone(&mut player.library, obj_id) {
                taken = Some(obj);
                break;
            }
            if let Some(obj) = take_from_zone(&mut player.battlefield, obj_id) {
                taken = Some(obj);
                break;
            }
            if let Some(obj) = take_from_zone(&mut player.graveyard, obj_id) {
                taken = Some(obj);
                break;
            }
            if let Some(obj) = take_from_zone(&mut player.exile, obj_id) {
                taken = Some(obj);
                break;
            }
        }
        let Some(mut obj) = taken else {
            return false;
        };
        obj.zone = target_zone;
        obj.controller = target_player;

        let target = &mut self.players[target_player as usize];
        let dest: &mut Zone = match target_zone {
            ZONE_HAND => &mut target.hand,
            ZONE_LIBRARY => &mut target.library,
            ZONE_BATTLEFIELD => &mut target.battlefield,
            ZONE_EXILE => &mut target.exile,
            // ZONE_GRAVEYARD and any unknown value (incl. stack
            // until that lands) fall into the graveyard for now. The
            // module docstring covers the limitation.
            _ => &mut target.graveyard,
        };
        Arc::make_mut(dest).push(obj);
        true
    }

    /// Pop the top of `player_id`'s library and push it onto that
    /// player's hand. Returns the drawn object's id, or `None` if
    /// the library was empty.
    ///
    /// "Top of library" = LAST element of the library `Vec`. See the
    /// module docstring for why the Python convention (front = top)
    /// is flipped here.
    fn draw_card(&mut self, player_id: u8) -> Option<u32> {
        let player = &mut self.players[player_id as usize];
        let lib = Arc::make_mut(&mut player.library);
        let mut obj = lib.pop()?;
        obj.zone = ZONE_HAND;
        let drawn_id = obj.id;
        Arc::make_mut(&mut player.hand).push(obj);
        Some(drawn_id)
    }
}
