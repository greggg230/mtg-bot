//! Mana production / mana pool primitives on `GameState`.
//!
//! Ports the Python side's per-player mana-pool mutations
//! (`Player.mana_pool` in `src/mtg_bot/game.py`, populated by
//! `AddManaToPool` and friends in `src/mtg_bot/effects/generic/mana.py`)
//! into a small, named API on the Rust `GameState`:
//! `get_mana_pool`, `add_mana`, `remove_mana`, `empty_mana_pool`,
//! `tap_for_mana`.
//!
//! **Negative-amount semantics** (mirrors the life slice — see
//! `state_ops_life.rs`): `add_mana` and `remove_mana` take `i32`
//! amounts and raise `ValueError` on a negative value rather than
//! silently flooring or wrapping. Rationale: the Python side's
//! "silent no-op on bad inputs" is a known anti-pattern (see the
//! user's "silent no-op pipeline runs are pipeline bugs" guidance).
//! `amount = 0` is allowed and is a true no-op. `remove_mana`
//! underflow (e.g. removing 1 green from an empty pool) raises
//! rather than clamping at zero — silent underflow would mask
//! paid-but-not-collected bugs in cost-payment code.
//!
//! **Mana-production lookup reads `CardPrinting.mana_produced`.**
//! Round-3 retired the round-2 placeholder (string-match on
//! `"Forest"`); `tap_for_mana` now reads the printing's typed
//! `mana_produced: Option<Vec<String>>` field. First-slice scope
//! only handles single-color, single-pip producers (Vec length 1):
//! a `Some(vec!["G"])` Forest produces one G into the controller's
//! pool. Multi-pip producers (a future "tap for {G}{G}" bicycling
//! land) and multi-color producers ("tap for one of any color")
//! return `None` for now — the trip-wire is `produced.len() != 1`
//! in `tap_for_mana`, which will fail loudly on a registry seed
//! that produces more than one pip. Add a richer schema (and a
//! follow-up slice) before adding such a printing.
//!
//! **CoW for the mana pool.** `Player` lives by-value in
//! `GameState.players: [Player; 2]`, and `ManaPool` is itself
//! `Copy + Clone + Default`. Cloning a `GameState` therefore
//! deep-copies the mana pool by value — no `Arc::make_mut`
//! juggling, same shape as life. The `tap_for_mana` path also
//! mutates a battlefield zone (`Arc<Vec<GameObject>>`) and routes
//! that through `Arc::make_mut` so the per-zone CoW invariant
//! still holds for the tap side.

use crate::state::{GameObject, GameState};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

/// Project a (color: &str) onto a mut ref to the matching `u8` slot
/// of a `ManaPool`. Returns `Err(ValueError)` if the color string
/// isn't one of W/U/B/R/G/C (case-sensitive — matches the closed
/// vocabulary used elsewhere in the engine).
fn slot_mut<'a>(
    pool: &'a mut crate::state::ManaPool,
    color: &str,
) -> PyResult<&'a mut u8> {
    match color {
        "W" => Ok(&mut pool.white),
        "U" => Ok(&mut pool.blue),
        "B" => Ok(&mut pool.black),
        "R" => Ok(&mut pool.red),
        "G" => Ok(&mut pool.green),
        "C" => Ok(&mut pool.colorless),
        other => Err(PyValueError::new_err(format!(
            "unknown mana color {other:?}; valid: W, U, B, R, G, C"
        ))),
    }
}

#[pymethods]
impl GameState {
    /// Return `player_id`'s mana pool as a 6-key dict
    /// `{"W", "U", "B", "R", "G", "C"}`. Always returns all six
    /// keys, even when the corresponding count is 0 — Python
    /// callers can rely on the shape and skip a `.get(color, 0)`
    /// guard.
    fn get_mana_pool<'py>(
        &self,
        py: Python<'py>,
        player_id: u8,
    ) -> PyResult<Bound<'py, PyDict>> {
        let idx = check_player_id(player_id)?;
        let pool = &self.players[idx].mana_pool;
        let d = PyDict::new(py);
        d.set_item("W", pool.white)?;
        d.set_item("U", pool.blue)?;
        d.set_item("B", pool.black)?;
        d.set_item("R", pool.red)?;
        d.set_item("G", pool.green)?;
        d.set_item("C", pool.colorless)?;
        Ok(d)
    }

    /// Add `amount` mana of `color` to `player_id`'s pool. Raises
    /// `ValueError` on a negative `amount` or an unknown color;
    /// `IndexError` on an out-of-range `player_id`. `amount = 0`
    /// is a no-op.
    fn add_mana(&mut self, player_id: u8, color: &str, amount: i32) -> PyResult<()> {
        let idx = check_player_id(player_id)?;
        check_non_negative("add_mana", amount)?;
        let pool = &mut self.players[idx].mana_pool;
        let slot = slot_mut(pool, color)?;
        // Saturating to u8::MAX is intentional — 255 of any one
        // color is more than any real game produces, and silently
        // wrapping would be a worse failure mode than capping.
        *slot = slot.saturating_add(amount as u8);
        Ok(())
    }

    /// Remove `amount` mana of `color` from `player_id`'s pool.
    /// Raises `ValueError` on a negative `amount`, an unknown
    /// color, or an underflow (removing more than the pool holds).
    /// `amount = 0` is a no-op.
    fn remove_mana(
        &mut self,
        player_id: u8,
        color: &str,
        amount: i32,
    ) -> PyResult<()> {
        let idx = check_player_id(player_id)?;
        check_non_negative("remove_mana", amount)?;
        let pool = &mut self.players[idx].mana_pool;
        let slot = slot_mut(pool, color)?;
        let current = *slot as i32;
        if amount > current {
            return Err(PyValueError::new_err(format!(
                "remove_mana: underflow — pool has {current} {color:?}, \
                 can't remove {amount}"
            )));
        }
        *slot = (current - amount) as u8;
        Ok(())
    }

    /// Zero `player_id`'s entire mana pool. Mirrors the cleanup-step
    /// "empty the mana pool" rule. Leaves the opponent's pool
    /// untouched.
    fn empty_mana_pool(&mut self, player_id: u8) -> PyResult<()> {
        let idx = check_player_id(player_id)?;
        self.players[idx].mana_pool = crate::state::ManaPool::default();
        Ok(())
    }

    /// Attempt to pay `cost` from `player_id`'s mana pool (CR 202.2).
    /// Returns `true` and deducts the mana on success; returns
    /// `false` and leaves the pool unchanged if the cost can't be
    /// paid. Mirrors the `pay_life_cost` shape (round-15) — a
    /// pure cost-payment primitive that higher-level cast/activation
    /// code calls before committing to the action.
    ///
    /// `cost` is a Python dict shaped like the `mana_cost` field on
    /// card printings: keys are `"generic"` / `"W"` / `"U"` / `"B"`
    /// / `"R"` / `"G"`, values are non-negative integers. `"C"` is
    /// accepted as an alias for `"generic"` (matches `parse_mana_cost`
    /// in cards.rs — colorless costs are paid identically to generic
    /// costs from the pool side; the distinction would only matter
    /// for "can't be paid with colored mana" costs, which aren't in
    /// the closed vocabulary yet). Unknown keys raise `ValueError`.
    /// An empty dict is a zero cost — trivially payable, returns
    /// `true` with no mutation.
    ///
    /// **Algorithm:**
    /// 1. For each colored requirement (W/U/B/R/G), check
    ///    `pool[color] >= cost[color]`. If any colored pip can't be
    ///    matched by same-color mana → return `false`, no mutation.
    /// 2. Compute the post-colored remaining pool. If
    ///    `remaining_total < cost.generic` → return `false`, no
    ///    mutation.
    /// 3. Commit: deduct colored requirements, then deduct generic
    ///    from a deterministic order — **colorless first, then
    ///    W/U/B/R/G**. Rationale: colorless is the most-restricted
    ///    mana source (it can ONLY pay generic), so draining it
    ///    first preserves the most flexibility for any later cost
    ///    payment from the same pool. The WUBRG tail order is
    ///    deterministic for reproducibility but otherwise arbitrary —
    ///    a future slice that adds "can't be paid with X" will
    ///    revisit the order.
    ///
    /// Raises `ValueError` on a malformed cost dict (unknown key,
    /// non-integer value, negative value) and `IndexError` on an
    /// out-of-range `player_id`.
    fn pay_mana_cost(
        &mut self,
        player_id: u8,
        cost: &Bound<'_, PyDict>,
    ) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Local parse — `cards::parse_mana_cost` is private and we
        // can't widen its visibility in this slice (sibling agents
        // are also editing cards.rs is off-limits this round).
        // Same key vocabulary, same "C is generic alias" convention.
        let mut req_generic: u32 = 0;
        let mut req_white: u32 = 0;
        let mut req_blue: u32 = 0;
        let mut req_black: u32 = 0;
        let mut req_red: u32 = 0;
        let mut req_green: u32 = 0;
        for (key_obj, value_obj) in cost.iter() {
            let key: String = key_obj.extract()?;
            let value: u32 = value_obj.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "pay_mana_cost: cost[{key:?}] must be a non-negative integer (got: {e})"
                ))
            })?;
            match key.as_str() {
                "generic" | "C" => req_generic = req_generic.saturating_add(value),
                "W" => req_white = value,
                "U" => req_blue = value,
                "B" => req_black = value,
                "R" => req_red = value,
                "G" => req_green = value,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "pay_mana_cost: unknown cost key {other:?}; valid: generic, W, U, B, R, G, C"
                    )))
                }
            }
        }
        let pool = &self.players[idx].mana_pool;
        // Step 1: each colored requirement must be matchable by
        // same-color mana already in the pool.
        if (pool.white as u32) < req_white
            || (pool.blue as u32) < req_blue
            || (pool.black as u32) < req_black
            || (pool.red as u32) < req_red
            || (pool.green as u32) < req_green
        {
            return Ok(false);
        }
        // Step 2: post-colored remaining pool must cover generic.
        let rem_white = pool.white as u32 - req_white;
        let rem_blue = pool.blue as u32 - req_blue;
        let rem_black = pool.black as u32 - req_black;
        let rem_red = pool.red as u32 - req_red;
        let rem_green = pool.green as u32 - req_green;
        let rem_colorless = pool.colorless as u32;
        let rem_total = rem_white + rem_blue + rem_black + rem_red + rem_green + rem_colorless;
        if rem_total < req_generic {
            return Ok(false);
        }
        // Step 3: commit. Deduct colored, then drain generic from a
        // deterministic order: C → W → U → B → R → G.
        let pool = &mut self.players[idx].mana_pool;
        pool.white -= req_white as u8;
        pool.blue -= req_blue as u8;
        pool.black -= req_black as u8;
        pool.red -= req_red as u8;
        pool.green -= req_green as u8;
        let mut remaining_generic = req_generic;
        for slot in [
            &mut pool.colorless,
            &mut pool.white,
            &mut pool.blue,
            &mut pool.black,
            &mut pool.red,
            &mut pool.green,
        ] {
            if remaining_generic == 0 {
                break;
            }
            let avail = *slot as u32;
            let take = avail.min(remaining_generic);
            *slot -= take as u8;
            remaining_generic -= take;
        }
        // Sanity: step-2's `rem_total >= req_generic` check guarantees
        // we always drain to zero. Belt-and-suspenders only.
        debug_assert_eq!(remaining_generic, 0);
        Ok(true)
    }

    /// Tap a permanent for mana. Reads the printing's
    /// `mana_produced: Option<Vec<String>>` (round-3 retired the
    /// previous name-keyed lookup). Returns `Some(color)` for a
    /// single-pip single-color producer, `None` on every failure
    /// mode (not on a battlefield, already tapped, printing has no
    /// `mana_produced`, or `mana_produced` has length != 1 —
    /// stepping-stone, see module docstring). The failure paths
    /// are silent (no error raised) to match the Python convention
    /// where a mana ability that can't activate is a no-op at the
    /// action layer.
    fn tap_for_mana(&mut self, obj_id: u32) -> PyResult<Option<String>> {
        // Phase 1: locate the object, peek at its printing's
        // `mana_produced`, and decide whether the tap will produce.
        // Done as an immutable scan so we don't open `Arc::make_mut`
        // unless we know the tap is going to commit.
        let mut found: Option<(usize, String)> = None; // (controller_idx, color)
        for (player_idx, player) in self.players.iter().enumerate() {
            for obj in player.battlefield.iter() {
                if obj.id == obj_id {
                    if obj.tapped {
                        return Ok(None);
                    }
                    let printing = match self.registry.lookup(obj.printing_id) {
                        Some(p) => p,
                        None => return Ok(None),
                    };
                    let produced = match &printing.mana_produced {
                        Some(v) => v,
                        None => return Ok(None),
                    };
                    // Stepping-stone: only single-color, single-pip
                    // producers handled this slice. A future slice
                    // adds multi-pip / multi-color support and may
                    // change the return type.
                    if produced.len() != 1 {
                        return Ok(None);
                    }
                    found = Some((player_idx, produced[0].clone()));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let (controller_idx, produced_color) = match found {
            Some(t) => t,
            None => return Ok(None),
        };
        // Phase 2: commit the mutation. Tap the permanent (via
        // `Arc::make_mut` on the zone) and credit the controller's
        // pool. `slot_mut` validates `produced_color` against the
        // closed vocabulary; an invalid symbol made it into the
        // registry only if `parse_printing`'s validator was
        // bypassed, which would be a registry-construction bug, not
        // a runtime tap-for-mana bug — so we surface it loudly
        // rather than silently dropping the production.
        let player = &mut self.players[controller_idx];
        let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
        for obj in zone.iter_mut() {
            if obj.id == obj_id {
                obj.tapped = true;
                break;
            }
        }
        let slot = slot_mut(&mut player.mana_pool, &produced_color)?;
        *slot = slot.saturating_add(1);
        Ok(Some(produced_color))
    }
}
