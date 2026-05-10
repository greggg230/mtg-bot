//! Cast / play primitives on `GameState`.
//!
//! First Rust port of the "play" surface — see Python's
//! ``_play_land`` and ``_cast_spell`` in ``src/mtg_bot/actions.py``
//! for the semantics being trimmed down here. The Python pipeline
//! threads alt-costs, room halves, warp / impulse / top-of-library
//! origins, ETB triggers, stack push, and priority pass into one
//! call; this slice carves off the irreducible primitive: take a
//! card from hand, pay its mana, deposit it on the battlefield.
//!
//! ## Scope simplifications
//!
//! - **No stack yet.** Casting a creature spell moves it directly
//!   hand -> battlefield after paying mana. Sorceries / instants /
//!   triggered-via-stack interactions are deferred until a Rust-side
//!   stack lands.
//! - **No one-land-per-turn enforcement.** ``play_land`` does not
//!   consult or update ``lands_played_this_turn``. The Python side
//!   gates this in ``legal_actions``, not inside the play primitive
//!   itself; the primitive's responsibility is the membership move
//!   and the printing-type check. **TODO**: revisit once a per-turn
//!   flag lives on ``GameState`` (today's ``GameState`` has no such
//!   field — adding one would force an edit on ``state.rs`` that
//!   sibling slices need to keep clean this round).
//! - **No sorcery-speed restrictions.** Same rationale.
//!
//! ## Type-based detectors: card_types from foundation r6 (cf1fc6c)
//!
//! Foundation r6 added `CardPrinting.card_types: Vec<String>` and
//! seeded the synthetic registry with proper types — Forest carries
//! `["Land"]`, Grizzly Bears and Llanowar Elves both carry
//! `["Creature"]`. This slice rewires both detectors to consult
//! `card_types` directly:
//!
//! - `play_land` accepts iff `card_types` contains `"Land"`.
//! - `cast_creature` accepts iff `card_types` contains `"Creature"`.
//!
//! The round-4 stepping stone — `mana_produced.is_some()` for
//! "this is a land" — has been retired. That predicate produced a
//! documented false-positive on Llanowar Elves (which has
//! `mana_produced = Some(["G"])` in the synthetic seed for the
//! mana-slice's tap-for-mana coverage); the new card_types-based
//! detector correctly rejects elves-as-lands. Symmetrically,
//! `cast_creature` no longer relies on `mana_cost` / `power`
//! heuristics — a Land printing with a mana cost (none exist in
//! the synthetic seed, but the field is structurally allowed)
//! would not slip through.
//!
//! ## Mana payment
//!
//! ``cast_creature`` reads the printing's ``mana_cost``, deducts
//! colored pips from the matching pool slot, then fills generic from
//! whatever's left in W/U/B/R/G/C order. Returns ``false`` (no
//! mutation) if the pool can't cover the colored requirement or the
//! generic fill. Hybrid / X / phyrexian / snow costs are not
//! modeled. The "best-effort spend" heuristic mirrors Python's
//! ``pay_mana_cost`` for first-pass scope; a future slice can
//! introduce a proper search when costs admit multiple legal payment
//! plans.
//!
//! ## CoW invariants
//!
//! Both methods mutate two zones (hand + battlefield) on the same
//! player and the player's mana pool. Each zone goes through
//! ``Arc::make_mut`` so cloning a ``GameState`` and casting on the
//! clone leaves the source state's zones untouched. The mana pool
//! lives by-value in ``Player`` (``ManaPool`` is ``Copy``), so
//! cloning the GameState already deep-copies the pool — no Arc
//! juggling needed there. ``test_clone_then_cast_does_not_affect_original``
//! and ``test_play_land_clone_then_play_does_not_affect_original``
//! are the load-bearing canaries.

use crate::state::{GameObject, GameState, ManaPool, Player};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::sync::Arc;

const ZONE_BATTLEFIELD: u8 = 2;

fn check_player_id(player_id: u8) -> PyResult<usize> {
    let idx = player_id as usize;
    if idx >= 2 {
        return Err(PyIndexError::new_err(format!(
            "player_id {player_id} out of range (must be 0 or 1)"
        )));
    }
    Ok(idx)
}

/// Total pip count of a `ManaCost`. Used by `pay_mana_cost`'s
/// insufficient-total fast-fail before the per-color spend loop
/// touches anything.
fn cost_total(cost: &crate::cards::ManaCost) -> u32 {
    cost.generic as u32
        + cost.white as u32
        + cost.blue as u32
        + cost.black as u32
        + cost.red as u32
        + cost.green as u32
}

fn pool_total(pool: &ManaPool) -> u32 {
    pool.white as u32
        + pool.blue as u32
        + pool.black as u32
        + pool.red as u32
        + pool.green as u32
        + pool.colorless as u32
}

/// Try to spend `cost` out of `pool`. On success the pool is
/// debited in place and `true` is returned; on failure the pool
/// is left untouched and `false` is returned.
///
/// Algorithm: pay every colored requirement first (a colored pip
/// can only be paid by its own color), then fill generic in
/// **C first, then W/U/B/R/G** order. The colorless-first ordering
/// matches Magic's "spend colorless before colored" intuition: a
/// player who has both colorless and colored mana will normally
/// prefer to keep their colored mana free for future colored
/// requirements. The test
/// ``test_cast_creature_with_generic_cost_uses_any_color`` pins
/// that ordering against a stray red pip in the pool. A future
/// slice with hybrid / X / phyrexian costs replaces this best-
/// effort spend with a proper plan-search.
fn pay_mana_cost(pool: &mut ManaPool, cost: &crate::cards::ManaCost) -> bool {
    // Fast-fail on totals. Saves a partial spend + rollback in the
    // common "you don't have enough mana, period" case.
    if pool_total(pool) < cost_total(cost) {
        return false;
    }
    // Snapshot so we can roll back on a partial-pay failure (a
    // colored pip the pool can't cover).
    let snapshot = *pool;
    if pool.white < cost.white
        || pool.blue < cost.blue
        || pool.black < cost.black
        || pool.red < cost.red
        || pool.green < cost.green
    {
        return false;
    }
    pool.white -= cost.white;
    pool.blue -= cost.blue;
    pool.black -= cost.black;
    pool.red -= cost.red;
    pool.green -= cost.green;
    let mut generic_remaining = cost.generic as u32;
    for slot in [
        &mut pool.colorless,
        &mut pool.white,
        &mut pool.blue,
        &mut pool.black,
        &mut pool.red,
        &mut pool.green,
    ] {
        if generic_remaining == 0 {
            break;
        }
        let take = (*slot as u32).min(generic_remaining) as u8;
        *slot -= take;
        generic_remaining -= take as u32;
    }
    if generic_remaining > 0 {
        // Shouldn't happen — pool_total covered the case above —
        // but defend anyway with a rollback.
        *pool = snapshot;
        return false;
    }
    true
}

/// Move the GameObject with id `obj_id` out of `player.hand` and
/// onto `player.battlefield`, stamping its `zone` field to
/// battlefield. Returns the moved object on success, or `None` if
/// no object with that id is in hand. Both zone mutations route
/// through `Arc::make_mut` so the CoW invariant holds.
fn move_from_hand_to_battlefield(
    player: &mut Player,
    obj_id: u32,
) -> Option<GameObject> {
    let pos = player.hand.iter().position(|o| o.id == obj_id)?;
    let hand = Arc::make_mut(&mut player.hand);
    let mut obj = hand.remove(pos);
    obj.zone = ZONE_BATTLEFIELD;
    let bf = Arc::make_mut(&mut player.battlefield);
    bf.push(obj);
    // Return a copy of the freshly-pushed object for the caller's
    // book-keeping (e.g. summoning-sick stamping is done by the
    // caller before this push, see `cast_creature`).
    Some(obj)
}

#[pymethods]
impl GameState {
    /// Read the `summoning_sick` flag for the battlefield permanent
    /// with id `obj_id`. Returns `Some(flag)` when found, `None`
    /// when no such object lives on either player's battlefield.
    /// Mirrors the read shape of `is_tapped` in `state_ops_tap.rs`.
    /// Lives in this module because the cast slice is the first to
    /// stamp `summoning_sick = true` on a fresh permanent and the
    /// test ``test_cast_creature_summoning_sick_after_cast`` reads
    /// it back. Earlier slices used `summoning_sick: false` as the
    /// `make_object` default but never observed it.
    fn is_summoning_sick(&self, obj_id: u32) -> Option<bool> {
        for player in &self.players {
            for obj in player.battlefield.iter() {
                if obj.id == obj_id {
                    return Some(obj.summoning_sick);
                }
            }
        }
        None
    }

    /// Play a land from `player_id`'s hand onto their battlefield.
    /// Returns `true` on success, `false` on every precondition
    /// failure: no object with `obj_id` in that player's hand, no
    /// such printing in the registry, or the printing's `card_types`
    /// does not contain `"Land"` (clean type-based detector — see
    /// module docstring).
    ///
    /// **Does not enforce one-land-per-turn** — the Python side
    /// gates that in `legal_actions`. **Does not consult sorcery
    /// speed.** Caller's responsibility for both.
    fn play_land(&mut self, obj_id: u32, player_id: u8) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Locate + read the printing immutably to decide the call's
        // outcome before any Arc::make_mut clone fires.
        let printing_id = match self.players[idx]
            .hand
            .iter()
            .find(|o| o.id == obj_id)
            .map(|o| o.printing_id)
        {
            Some(pid) => pid,
            None => return Ok(false),
        };
        let printing = match self.registry.lookup(printing_id) {
            Some(p) => p,
            None => return Ok(false),
        };
        if !printing.card_types.iter().any(|t| t == "Land") {
            return Ok(false);
        }
        // Commit the move. The hand search re-runs inside
        // `move_from_hand_to_battlefield`; that's fine — the cost is
        // a single second linear scan we already know will hit, and
        // it keeps the helper self-contained.
        let player = &mut self.players[idx];
        let moved = move_from_hand_to_battlefield(player, obj_id);
        Ok(moved.is_some())
    }

    /// Cast a creature spell from `player_id`'s hand. Pays the
    /// printing's `mana_cost` out of the player's mana pool, moves
    /// the card to the battlefield, and stamps the new permanent
    /// with `summoning_sick = true`. Returns `true` on success,
    /// `false` on any precondition failure: object not in hand, no
    /// such printing, the printing's `card_types` does not contain
    /// `"Creature"`, or insufficient mana.
    ///
    /// **No-stack semantics.** The Python pipeline pushes a
    /// `StackItem`, settles priority, then resolves; this primitive
    /// short-circuits to the resolved state. ETB triggers, "when X
    /// is cast" hooks, and any choice points the spell normally
    /// presents are deferred until a Rust-side stack lands.
    fn cast_creature(&mut self, obj_id: u32, player_id: u8) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Phase 1: read-only lookup. Find the object in hand, fetch
        // the printing's cost + types. Bail before any mutation if
        // any step fails. The type check happens here (pre-mutation)
        // so a non-creature spell never has its mana speculatively
        // debited.
        let printing_id = match self.players[idx]
            .hand
            .iter()
            .find(|o| o.id == obj_id)
            .map(|o| o.printing_id)
        {
            Some(pid) => pid,
            None => return Ok(false),
        };
        let cost = match self.registry.lookup(printing_id) {
            Some(p) => {
                if !p.card_types.iter().any(|t| t == "Creature") {
                    return Ok(false);
                }
                p.mana_cost.clone()
            }
            None => return Ok(false),
        };
        // Phase 2: try the mana spend on a copy of the pool. If it
        // succeeds, commit; otherwise return false with no mutation.
        let player = &mut self.players[idx];
        let mut pool_copy = player.mana_pool;
        if !pay_mana_cost(&mut pool_copy, &cost) {
            return Ok(false);
        }
        player.mana_pool = pool_copy;
        // Phase 3: move the card. Stamp summoning sickness AFTER the
        // push (stamp on the in-place battlefield slot, since
        // `move_from_hand_to_battlefield` returns the snapshot at
        // push time).
        let _ = move_from_hand_to_battlefield(player, obj_id);
        let bf = Arc::make_mut(&mut player.battlefield);
        if let Some(obj) = bf.iter_mut().find(|o| o.id == obj_id) {
            obj.summoning_sick = true;
        }
        Ok(true)
    }
}
