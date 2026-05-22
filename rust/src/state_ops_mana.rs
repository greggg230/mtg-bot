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
        // Validate the color via `slot_mut` (it raises on unknown
        // colors); use the byte form for the ordered-pool push so the
        // `coins` vec and the per-color count stay in lockstep.
        let _slot = slot_mut(pool, color)?;
        let color_byte = match color {
            "W" => b'W',
            "U" => b'U',
            "B" => b'B',
            "R" => b'R',
            "G" => b'G',
            "C" => b'C',
            // Unreachable: `slot_mut` above already validated the
            // closed vocabulary. Belt-and-suspenders pattern.
            _ => unreachable!("slot_mut validated color vocabulary"),
        };
        // G7 S6-Class-E ordered-pool: push N coins to the insertion-
        // ordered `coins` vec AND bump the per-color count cache in
        // lockstep. `push_coins` does both with saturating-add on the
        // count (255-cap mirrors the pre-ordered-pool shape — 255 of
        // any one color is more than any real game produces).
        pool.push_coins(color_byte, amount as u8);
        pool.assert_count_sync();
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
        // G7 S6-Class-E ordered-pool: drain from `coins` in insertion
        // order (FIFO over coins matching `color`). The
        // `drain_first_n_color` helper decrements the per-color count
        // cache as it removes each coin, keeping the two sides
        // synced. We don't need to touch the slot ref above — the
        // helper owns the cache update.
        let color_byte = match color {
            "W" => b'W',
            "U" => b'U',
            "B" => b'B',
            "R" => b'R',
            "G" => b'G',
            "C" => b'C',
            _ => unreachable!("slot_mut validated color vocabulary"),
        };
        let drained = pool.drain_first_n_color(color_byte, amount as u32);
        debug_assert_eq!(
            drained, amount as u32,
            "remove_mana: drain mismatch — coin/count sync drift"
        );
        pool.assert_count_sync();
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

    /// Zero BOTH players' mana pools. Implements CR 106.4: "When a
    /// phase or step ends, each player's mana pool empties." This is
    /// the multi-player wrapper around `empty_mana_pool` that the
    /// phase/step-end machinery calls. `ManaPool` is `Copy + Default`,
    /// so each assignment is a cheap by-value reset.
    fn empty_all_mana_pools(&mut self) -> PyResult<()> {
        self.players[0].mana_pool = crate::state::ManaPool::default();
        self.players[1].mana_pool = crate::state::ManaPool::default();
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
        //
        // **G7 S6-Class-E ordered-pool parity-neutral migration**:
        // this `pay_mana_cost` PyMethod is NOT the Python SoT for
        // CastSpell / ActivateAbility — those route through
        // `auto_tap_pay_for_cost` in `state_ops_cast.rs` (the lands-
        // auto-tapping byte-mirror of `auto_tap_for_cost`). This
        // method is the count-only legacy pay primitive; preserving
        // its existing C→W→U→B→R→G drain order keeps Slice-1 byte-
        // neutral (no test churn). What we ADD here is keeping the
        // ordered `coins` vec in sync: for each per-color slot the
        // method decrements, drain that many coins of that color from
        // the head of `coins`. The two sides remain consistent so
        // future readers (or future drain consumers) see the same
        // pool.
        let pool = &mut self.players[idx].mana_pool;
        pool.drain_first_n_color(b'W', req_white);
        pool.drain_first_n_color(b'U', req_blue);
        pool.drain_first_n_color(b'B', req_black);
        pool.drain_first_n_color(b'R', req_red);
        pool.drain_first_n_color(b'G', req_green);
        let mut remaining_generic = req_generic;
        // Walk the legacy C → W → U → B → R → G priority for the
        // generic drain; for each color, drain from `coins` (FIFO of
        // matching color) to keep `coins`/count cache synced. The
        // legacy order is preserved verbatim (parity-neutral); the
        // CLASS-E FIX in `auto_tap_pay_for_cost` is a SEPARATE drain
        // path that uses `drain_first_n_fifo` (any-color FIFO).
        for color in [b'C', b'W', b'U', b'B', b'R', b'G'] {
            if remaining_generic == 0 {
                break;
            }
            let avail = match color {
                b'W' => pool.white as u32,
                b'U' => pool.blue as u32,
                b'B' => pool.black as u32,
                b'R' => pool.red as u32,
                b'G' => pool.green as u32,
                b'C' => pool.colorless as u32,
                _ => 0,
            };
            let take = avail.min(remaining_generic);
            pool.drain_first_n_color(color, take);
            remaining_generic -= take;
        }
        // Sanity: step-2's `rem_total >= req_generic` check guarantees
        // we always drain to zero. Belt-and-suspenders only.
        debug_assert_eq!(remaining_generic, 0);
        pool.assert_count_sync();
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
        // G7 S6-Class-E ordered-pool: route the produced-color credit
        // through `push_coins` so the insertion-ordered `coins` vec
        // and the per-color count cache are kept in sync. The `&str`
        // produced_color is one of W/U/B/R/G/C (gated by the registry
        // validator + the `slot_mut` lookup below as a belt-and-
        // suspenders no-op-on-invalid).
        // Validate the color through `slot_mut` first to preserve the
        // existing error-on-bad-symbol contract (a registry-construct-
        // ion bug raises here, not silently drops).
        let _slot = slot_mut(&mut player.mana_pool, &produced_color)?;
        let color_byte = match produced_color.as_str() {
            "W" => b'W',
            "U" => b'U',
            "B" => b'B',
            "R" => b'R',
            "G" => b'G',
            "C" => b'C',
            _ => unreachable!("slot_mut validated color vocabulary"),
        };
        player.mana_pool.push_coins(color_byte, 1);
        player.mana_pool.assert_count_sync();
        // Round-58 BECOMES_TAPPED event bridge — tap-for-mana site.
        // Python's `auto_tap_for_cost` (`src/mtg_bot/actions.py:
        // 981-985`) routes every land it taps through
        // `turn.tap_permanent`, which fires `Event.BECOMES_TAPPED`
        // (the docstring there explicitly cites Chrome Companion /
        // Grove of the Burnwillows becomes-tapped mana triggers).
        // The flip is already committed above (the pool credit needs
        // the `player` mut-borrow held); dispatch the event now that
        // the borrow is released. We pass `dispatch_only=true` to the
        // chokepoint via the inline form below: the flip happened
        // here (mana-credit ordering), so re-flipping through
        // `tap_permanent_and_dispatch` would see an already-tapped
        // permanent and no-op the EVENT too. Instead emit the
        // dispatch directly — the no-op-if-already-tapped gate is
        // moot here because we only reach this line when the
        // immutable Phase-1 scan proved the permanent was UNTAPPED
        // (it `return Ok(None)`s on `obj.tapped`), so this is exactly
        // the "state actually changed" branch Python's
        // `tap_permanent` dispatches on.
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::BecomesTapped,
            obj_id,
        )?;
        Ok(Some(produced_color))
    }
}

/// **legal_decisions mana-awareness — Slice 1 (the SoT-flip-blocking
/// non-destructive `can_pay_cost` predicate).**
///
/// Non-destructive mana-affordability predicate. Returns `true` iff
/// `player_idx` could pay `cost` right now by consuming floating pool
/// mana then auto-tapping every untapped, non-summoning-sick mana
/// source they control. Reads ZERO mutation (it never mutates, so it
/// snapshots nothing). The named-public counterpart to the
/// destructive `pay_mana_cost` twin above — the SL note's exact fix
/// shape: "permissive in `legal_decisions`, strict in apply", the two
/// co-located so the pair is one unit.
///
/// Mirrors Python `mtg_bot.actions.can_pay_cost`
/// (`src/mtg_bot/actions.py:336-403`) — the source of truth —
/// algorithm-for-algorithm for the live CastSpell surface's
/// `generic + 5-colored` `ManaCost` shape. **X-value subsystem
/// Cycle 2 (2026-05-22) added an `x_pips: u8` slot** to
/// `cards::ManaCost` so X-cost printings (Zero Point Ballad,
/// Devastating Onslaught) can round-trip the bridge — but the
/// admit gate still rejects them and `can_pay_cost` still IGNORES
/// `x_pips` (every existing call site passes a cost with
/// `x_pips == 0`; Cycle 2.5+ will fold `x_pips * x_value` into
/// generic at the cast-pipeline detector before invoking this
/// predicate, so the predicate signature stays unchanged).
/// `hybrid` and Phyrexian fields still don't exist; Python's
/// hybrid handling stays unreachable here. The two halves of the
/// algorithm already exist
/// and are byte-equal-pinned by the ActivateAbility gate (pins I50):
///
///   * `auto_tap_headroom_pool(state, idx)`
///     (`state_ops_legal.rs:559`) — the non-destructive "current
///     floating pool PLUS every untapped non-sick mana source"
///     hypothetical pool. Exactly Python's pool + `_untapped_lands`
///     auto-tap accounting (`actions.py:364-392`).
///   * `pool_can_cover(pool, cost)` (`state_ops_legal.rs:514`) —
///     exact colored-then-generic affordability over a count pool,
///     mirroring `pay_mana_cost`'s pre-mutation gate and Python's
///     "pay colored first, then generic with the remainder"
///     (`actions.py:371-403`).
///
/// So for the live surface `can_pay_cost` IS
/// `pool_can_cover(&auto_tap_headroom_pool(state, idx), cost)`. This
/// wrapper gives the predicate one canonical name (the SL note can
/// point at it), one place for the alt-cost / restricted-coin /
/// convoke / X-hybrid-Phyrexian extensions to grow (see the design
/// doc §2.4/§2.5 — designed extension points, NOT silent gaps), and
/// co-locates it with `pay_mana_cost`.
///
/// `spell_kind` is Python's per-coin spend-restriction context
/// (`actions.py:338`). The Rust `ManaPool` is a bare count struct
/// with NO per-coin restriction model (verified: `pool_can_cover` /
/// `auto_tap_headroom_pool` operate on raw counts); no live
/// bridge-admitted CastSpell card exercises a restricted-coin
/// context (those are Treasure/Eldrazi/artifact-spend idioms off the
/// cast surface). Carried in the signature for forward-composition
/// (a stable signature is the parity contract for the pins) but
/// IGNORED by the predicate — equivalent to Python's behavior when no
/// coin carries a restriction (the dominant real case), and it can
/// only keep the predicate a SUPERSET of Python in the restricted
/// corner, preserving the `Rust ⊇ Python` direction the
/// single-script-replay contract needs (design doc §4 proves
/// parity-neutrality). The restricted-coin port is bundled with the
/// alt-cost slice.
///
/// `convoke` / `sac_for_mana` are Python's upper-bound wideners
/// ("only ever make a spell cheaper"); no live admitted axis card has
/// convoke and sac-for-mana is the Treasure idiom — both off the live
/// cast surface. Omitted from Slice 1 (matching the live admitted
/// surface); adding them later is a pure widening (never narrows the
/// emitted set) so it lands without re-pinning. Same surgical-scope
/// discipline as WARP_COST_PHASE2 §1.
pub(crate) fn can_pay_cost(
    state: &crate::state::GameState,
    player_idx: usize,
    cost: &crate::cards::ManaCost,
    spell_kind: &SpellKind,
) -> bool {
    // Slice 1: `spell_kind` is the forward-composition stub (the
    // Rust ManaPool has no per-coin restriction model; no live
    // admitted card exercises a restricted-coin context). Ignored
    // here exactly as Python's `_coin_satisfies_context` is a no-op
    // when no coin carries a restriction (the dominant real case).
    let _ = spell_kind;
    //
    // **G7 S6 Class-D 1/23 Thawbringer fix (cycle-37 Phase 2).**
    //
    // The prior implementation —
    // `pool_can_cover(&auto_tap_headroom_pool(state, idx), cost)` —
    // over-credits dual-producing lands. `auto_tap_headroom_pool`
    // sums each land's `mana_produced` symbols into per-color counts,
    // so a Stomping Ground (`mana_produced = ["R", "G"]`) credits
    // both `pool.red += 1` AND `pool.green += 1` — counted as TWO
    // mana of capacity when in reality the land taps for ONE
    // (CR 605.1 — choose ONE produced color on activation).
    //
    // Python's `actions.can_pay_cost` (`actions.py:336-403` — the
    // SoT) tracks lands as a SET: each land is consumed once, either
    // for a colored requirement (any color it can produce) OR for
    // generic. arch30/s1 step 111 Thawbringer `{2}{G}` with p1
    // controlling untapped Stomping Ground + untapped Forest:
    // Python pays {G} using one of them (used_lands.add) → only 1
    // remaining land → generic 2 unpayable → False (Python prunes
    // the cast). Rust's old gate counted 1R + 2G headroom = 3 total
    // → pool_can_cover returns true → Rust over-emitted the cast.
    //
    // Fix: byte-port Python's set-based algorithm. Pool coins are
    // consumed by index; lands by id. Each land's produced-color
    // set is its `mana_produced` field (Forest = {G}; Stomping
    // Ground = {R,G}). A land matches a colored requirement iff
    // that color is in its set. Each land may be used at most once.
    // Mirrors apply-time `auto_tap_pay_for_cost` Phase 1 PLAN —
    // legal-emit and apply now agree on what's payable.
    //
    // **Scope of the change vs the prior comment block.** The prior
    // header described `can_pay_cost` as a wrapper over
    // `pool_can_cover(auto_tap_headroom_pool(...))`. After this fix
    // those two primitives still exist (and `pool_can_cover` is
    // still used by other call sites — `state_ops_legal.rs::
    // additional_costs_payable` and similar — that operate on a
    // POOL only, NO land-tap headroom). Only the live affordability
    // wrapper here switches to the set-based port. The mana-creature
    // breadth axis (Python uses lands ONLY; Rust's prior `auto_tap_
    // headroom_pool` also credits non-sick mana creatures) is a
    // DISTINCT, separately-tracked C1 axis (see the comment in
    // state_ops_cast.rs::auto_tap_pay_for_cost) — out of scope here.
    // We restrict to lands too (the apply path is lands-only, so
    // the legal path matching it is the parity-neutral move; mana
    // creatures auto-tap is the apply-side widening, not relevant
    // for this gate).
    can_pay_cost_set_based(state, player_idx, cost)
}

/// Set-based port of Python `actions.can_pay_cost` for the live
/// CastSpell surface (generic + 5-colored ManaCost; no hybrid/X/
/// Phyrexian on the Rust ManaCost struct). Each pool coin is
/// consumed by INDEX, each land by ID — exactly Python's
/// `used_coins` / `used_lands` sets. Lands are filtered to the
/// controller's untapped, non-summoning-sick lands (the
/// `_untapped_lands` shape — lands only, NOT mana creatures, to
/// byte-match `auto_tap_pay_for_cost`'s lands-only apply path).
///
/// Returns true iff the cost is payable. Pure read-only (no mutation
/// of state).
fn can_pay_cost_set_based(
    state: &crate::state::GameState,
    player_idx: usize,
    cost: &crate::cards::ManaCost,
) -> bool {
    // Snapshot untapped controller lands in battlefield order,
    // with their produced-color set. Mirrors Python's
    // `lands = _untapped_lands(state, player_id)` (`actions.py:196`).
    // Lands-only — matches the apply-side `auto_tap_pay_for_cost`
    // discipline (Python's `auto_tap_for_cost` is also lands-only).
    let mut lands: Vec<(u32, Vec<String>)> = Vec::new();
    for obj in state.players[player_idx].battlefield.iter() {
        if obj.tapped {
            continue;
        }
        let Some(printing) = state.registry.lookup(obj.printing_id) else {
            continue;
        };
        if !printing.card_types.iter().any(|t| t == "Land") {
            continue;
        }
        let symbols: Vec<String> = match &printing.mana_produced {
            Some(v) => v.clone(),
            None => Vec::new(),
        };
        if symbols.is_empty() {
            continue;
        }
        lands.push((obj.id, symbols));
    }
    // Snapshot the controller's pool coins (already the Class-E
    // insertion-ordered Vec<ManaCoin>).
    let pool: Vec<crate::state::ManaCoin> =
        state.players[player_idx].mana_pool.coins.clone();
    let mut used_coins: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut used_lands: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    // Pay colored requirements first — pool coins of the matching
    // color first, then any matching untapped land. Iteration order
    // is Python's `ManaCost.colored` order (B, G, R, U, W) per
    // `cards.parse_mana_cost`.
    let colored: [(u8, &str, u8); 5] = [
        (cost.black, "B", b'B'),
        (cost.green, "G", b'G'),
        (cost.red, "R", b'R'),
        (cost.blue, "U", b'U'),
        (cost.white, "W", b'W'),
    ];
    for (need_count, sym_str, sym_byte) in colored {
        let mut need = need_count as u32;
        if need == 0 {
            continue;
        }
        // Pool coins of the matching color first (insertion order;
        // mirrors `for idx, coin in enumerate(usable_pool)`).
        for (i, coin) in pool.iter().enumerate() {
            if need == 0 {
                break;
            }
            if used_coins.contains(&i) {
                continue;
            }
            if coin.color == sym_byte {
                used_coins.insert(i);
                need -= 1;
            }
        }
        if need == 0 {
            continue;
        }
        // Match-color lands (Python: `[l for l in lands if color in
        // effective_land_produces(l) and l.id not in used_lands]`).
        let matching: Vec<u32> = lands
            .iter()
            .filter(|(lid, syms)| {
                !used_lands.contains(lid)
                    && syms.iter().any(|s| s == sym_str)
            })
            .map(|(lid, _)| *lid)
            .collect();
        if (matching.len() as u32) < need {
            return false;
        }
        for lid in matching.into_iter().take(need as usize) {
            used_lands.insert(lid);
        }
    }
    // Generic (and hybrid pip count, which is 0 on the live Rust
    // ManaCost surface — the field exists in Python but Rust's
    // `cards::ManaCost` has no `hybrid` field, so we add 0 here).
    let remaining_pool = pool.len() - used_coins.len();
    let remaining_lands = lands
        .iter()
        .filter(|(lid, _)| !used_lands.contains(lid))
        .count();
    let needed = cost.generic as usize;
    remaining_pool + remaining_lands >= needed
}

/// The Rust analog of Python's `spell_kind: str | frozenset[str]`
/// (`src/mtg_bot/actions.py:338`) — the per-coin spend-restriction
/// context. **Slice 1: a forward-composition stub** — an opaque
/// carried value the predicate ignores (§2.4 of the design doc). The
/// shape is fixed NOW so the restricted-coin port (bundled with the
/// alt-cost slice) does not change `can_pay_cost`'s signature again —
/// a stable signature is the parity contract for the pin tests. The
/// unit variant is the only one Slice 1 constructs (`SpellKind::Any`,
/// the "no restriction" default mirroring Python's plain `"spell"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpellKind {
    /// No spend restriction (Python's default `"spell"` context —
    /// every coin/source is usable). The only variant Slice 1 uses.
    Any,
}
