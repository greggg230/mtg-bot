//! Damage primitives on `GameState`.
//!
//! Ports the per-creature ``damage`` marker semantics from the Python
//! engine — `GameObject.damage: int = 0` in `src/mtg_bot/game.py`,
//! mutated additively by combat damage in `src/mtg_bot/combat.py`
//! (`obj.damage += amount`) and zeroed in bulk by the end-of-turn
//! cleanup loop in `src/mtg_bot/turn.py` (CR 514.2 — "Damage wears
//! off"). The Rust side keeps that exact shape: per-instance i16
//! marker on `GameObject.damage`, monotonically increasing across a
//! turn, cleared in bulk at cleanup.
//!
//! API on `GameState`:
//! - `get_damage(obj_id) -> Option<i16>`
//! - `deal_damage(obj_id, amount) -> bool` — additive; ValueError on
//!   negative amount; returns False on unknown id (silent no-op
//!   convention, same as `tap_object` / `set_counters`).
//! - `clear_damage(obj_id) -> bool` — zero one object's marker.
//! - `clear_damage_for_player(player_id) -> ()` — walk one player's
//!   battlefield, zero each permanent's marker.
//! - `clear_all_damage() -> ()` — wholesale end-of-turn cleanup form.
//! - `is_damage_at_least(obj_id, threshold) -> Option<bool>` — the
//!   threshold form of the lethal-damage check (see stepping-stone
//!   note below).
//!
//! ## Stepping-stone for `is_lethal`
//!
//! `GameObject` does not yet carry a base toughness. A sibling agent
//! is wiring `base_toughness` onto `CardPrinting` in the
//! bridge-extension slice this round. Until that lands, the
//! lethal-damage helper exposed here is the threshold form
//! `is_damage_at_least(obj_id, threshold)`, which a future slice
//! trivially specializes to `is_lethal(obj_id)` once the printing
//! carries its toughness:
//!
//! ```ignore
//! fn is_lethal(&self, obj_id: u32) -> Option<bool> {
//!     let obj = find(self, obj_id)?;
//!     let toughness = self.registry.lookup(obj.printing_id)?.base_toughness;
//!     Some(obj.damage as i32 >= toughness as i32)
//! }
//! ```
//!
//! Keeping the threshold form on the schedule today (vs deferring
//! whole the lethal-damage helper to the bridge slice) gets the read
//! shape right and surfaces the obvious follow-up by name.
//!
//! ## SBA enforcement is out of scope
//!
//! This primitive reports raw marked damage. "Damage >= toughness =>
//! destroy" is a state-based action (CR 704.5h) that needs an SBA
//! loop on the Rust side; lethal-damage SBA enforcement is a
//! future slice once that loop lands. Same shape as the
//! "+1/+1 vs -1/-1 cancellation runs in SBAs" note in
//! `state_ops_counters.rs`.
//!
//! ## CoW
//!
//! Each per-object mutator routes through a private `mutate_object`
//! helper that scopes `Arc::make_mut` to the one zone Vec containing
//! the target object — preserves CoW so cloning a `GameState` and
//! mutating one clone does not leak into the other. The
//! `test_deal_damage_on_clone_does_not_affect_original` test is the
//! load-bearing canary, mirroring the analogous CoW canaries in
//! `test_tap.py` and `test_counters.py`. Per the round-2/3
//! per-slice isolation rule the helper lives in this file rather
//! than being shared with sibling slices.

use crate::state::{GameObject, GameState};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

fn check_player_id(player_id: u8) -> PyResult<usize> {
    let idx = player_id as usize;
    if idx >= 2 {
        return Err(PyIndexError::new_err(format!(
            "player_id {player_id} out of range (must be 0 or 1)"
        )));
    }
    Ok(idx)
}

fn check_non_negative(name: &str, amount: i32) -> PyResult<()> {
    if amount < 0 {
        return Err(PyValueError::new_err(format!(
            "{name}: amount must be non-negative, got {amount}"
        )));
    }
    Ok(())
}

/// Walk every player's four tracked zones (hand, library, battlefield,
/// graveyard) and apply `f` to the matching object via `Arc::make_mut`
/// on the containing zone Vec. Returns `true` if an object with
/// `obj_id` was found, `false` otherwise.
///
/// Mirrors the shape of `state_ops_counters.rs::mutate_object_anywhere`.
/// Lives in this file (not shared) per the per-slice isolation rule —
/// a follow-up consolidation slice can pull these helpers up once the
/// parallel-slice constraint relaxes.
fn mutate_object<F>(state: &mut GameState, obj_id: u32, mut f: F) -> bool
where
    F: FnMut(&mut GameObject),
{
    for player in state.players.iter_mut() {
        if player.hand.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.hand);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
        if player.library.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.library);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
        if player.battlefield.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
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

/// Read-only sibling of `mutate_object`. Walks the four tracked zones
/// on both players and returns the matching object's `damage` field,
/// or `None` if no such object exists.
fn read_damage(state: &GameState, obj_id: u32) -> Option<i16> {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    return Some(obj.damage);
                }
            }
        }
    }
    None
}

#[pymethods]
impl GameState {
    /// Read the marked damage on the object with `obj_id`.
    ///
    /// Returns `Some(n)` where `n` is the accumulated damage marker
    /// if the object exists in any tracked zone; `None` otherwise.
    /// Distinguishes "object not found" from "object found and at
    /// zero damage" — callers validating a target before applying
    /// an effect rely on that.
    fn get_damage(&self, obj_id: u32) -> Option<i16> {
        read_damage(self, obj_id)
    }

    /// Deal `amount` damage to the object with `obj_id`. Additive —
    /// successive deals accumulate (CR 120.3, multiple sources mark
    /// damage separately, the marker is the running sum).
    ///
    /// Returns `true` if an object with that id was found and the
    /// damage applied, `false` if no matching object exists in any
    /// tracked zone (silent no-op, same convention as `tap_object`
    /// and `set_counters`). Raises `ValueError` on a negative
    /// `amount` per the project's "fail loud on bad inputs"
    /// guidance — see `state_ops_life.rs` and `state_ops_mana.rs`
    /// for the same shape. `amount = 0` is allowed and is a true
    /// no-op (still returns True on a found object).
    ///
    /// Saturating add to the i16 marker — 32_767 damage is more
    /// than any real game produces, and silent wrap-around would be
    /// a worse failure mode than capping.
    fn deal_damage(&mut self, obj_id: u32, amount: i32) -> PyResult<bool> {
        check_non_negative("deal_damage", amount)?;
        let amount_i16 = amount.min(i16::MAX as i32) as i16;
        Ok(mutate_object(self, obj_id, |obj| {
            obj.damage = obj.damage.saturating_add(amount_i16);
        }))
    }

    /// Zero the marked damage on the object with `obj_id`. Returns
    /// `true` on hit, `false` if no matching object exists. The
    /// per-object form of the end-of-turn cleanup (CR 514.2 — see
    /// `clear_all_damage` for the wholesale form).
    fn clear_damage(&mut self, obj_id: u32) -> bool {
        mutate_object(self, obj_id, |obj| {
            obj.damage = 0;
        })
    }

    /// Walk `player_id`'s battlefield and zero each permanent's
    /// damage marker. Touches only that player's permanents — the
    /// opponent's marked damage is preserved.
    ///
    /// Building block for the per-player end-of-turn pass and a
    /// direct fit for any future "remove all damage from creatures
    /// you control" effect (Heroic Intervention / Make a Stand).
    /// Routes through `Arc::make_mut` on the battlefield Vec so the
    /// CoW invariant holds for clones observing the original.
    ///
    /// Battlefield-only by design: marked damage on a creature that
    /// has left the battlefield isn't a thing the rules track (CR
    /// 614.12 — damage is removed on zone change), so the cleanup
    /// helpers don't need to walk hand / library / graveyard.
    fn clear_damage_for_player(&mut self, player_id: u8) -> PyResult<()> {
        let idx = check_player_id(player_id)?;
        let player = &mut self.players[idx];
        // Optimization: skip the Arc::make_mut clone-on-write if no
        // permanent on this battlefield actually has damage. Keeps
        // a no-op call free of per-zone CoW costs in the common
        // "nothing took damage this turn" case.
        if !player.battlefield.iter().any(|o| o.damage != 0) {
            return Ok(());
        }
        let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
        for obj in zone.iter_mut() {
            obj.damage = 0;
        }
        Ok(())
    }

    /// Wholesale end-of-turn cleanup form — walk both players'
    /// battlefields and zero every damage marker in one call.
    /// Mirrors the `for obj in state.battlefield: obj.damage = 0`
    /// loop in `src/mtg_bot/turn.py` cleanup (CR 514.2).
    ///
    /// Implemented as two `clear_damage_for_player` calls so the
    /// per-zone CoW + skip-if-empty optimization applies symmetrically
    /// to each side.
    fn clear_all_damage(&mut self) -> PyResult<()> {
        self.clear_damage_for_player(0)?;
        self.clear_damage_for_player(1)?;
        Ok(())
    }

    /// Threshold form of the lethal-damage check. Returns
    /// `Some(true)` when the object's marked damage is `>= threshold`,
    /// `Some(false)` when it is below, and `None` if no object exists
    /// with that id.
    ///
    /// Stepping-stone for the eventual `is_lethal(obj_id)` —
    /// `GameObject` doesn't carry a base toughness yet (the bridge-ext
    /// slice this round is wiring `CardPrinting.base_toughness`),
    /// so callers pass the threshold explicitly today. Once the
    /// bridge lands, `is_lethal` becomes a one-line specialization
    /// — see this file's docstring.
    fn is_damage_at_least(&self, obj_id: u32, threshold: i32) -> Option<bool> {
        let damage = read_damage(self, obj_id)?;
        Some(damage as i32 >= threshold)
    }

    /// Deal `amount` damage directly to `player_id`'s life total.
    ///
    /// Distinct from `deal_damage`, which marks an object's per-instance
    /// damage marker. This variant is the building block for combat
    /// damage to a defending player, burn-spell damage to a player,
    /// "deals damage to each opponent" effects, etc. — anywhere a
    /// damage source resolves directly against a player rather than
    /// a creature/planeswalker.
    ///
    /// Raises `ValueError` on negative `amount` per the project's
    /// "fail loud" convention (mirrors `gain_life` / `lose_life` /
    /// `deal_damage`); `amount = 0` is allowed and is a true no-op.
    /// Raises `IndexError` if `player_id` is not 0 or 1.
    ///
    /// **No clamping at zero.** Result life can go negative — "loss
    /// at <=0 is a game-loss" is a state-based action handled at a
    /// higher layer (CR 704.5a). Same rule as `lose_life` in the
    /// life slice.
    ///
    /// **Today this collapses damage→life-loss into a single field
    /// write.** Real MTG rules-text fidelity (replacement effects on
    /// damage, lifelink-style routing, "damage to player can't be
    /// prevented") arrives in a later slice once those layers land
    /// on the Rust side. The named entry point exists today so
    /// callers don't have to know about the simplification.
    fn deal_damage_to_player(&mut self, player_id: u8, amount: i32) -> PyResult<()> {
        let idx = check_player_id(player_id)?;
        check_non_negative("deal_damage_to_player", amount)?;
        self.players[idx].life -= amount;
        Ok(())
    }
}
