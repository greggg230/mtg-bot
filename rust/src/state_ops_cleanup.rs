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
//! Round-27 extension (UEOT framework): CR 514.3 "until end of turn"
//! effects expiration now fires inside `cleanup_step_for_active_player`
//! AFTER CR 514.1 (hand-size discard) and CR 514.2 (damage clear). The
//! shape:
//!  - walk both players' battlefield zones
//!  - for each object whose `ueot_modifiers` is non-empty, route through
//!    `Arc::make_mut` (skipping the clone when every object on the
//!    battlefield is UEOT-clean) and drain its modifiers Vec.
//! See `clear_ueot_modifiers_all` for the primitive and the round-27
//! brief for the full architectural framing. This closes parity gap G7.
//!
//! Out of scope (CR 514.4):
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
use crate::state_ops_steps::PHASE_CLEANUP;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Read-side helper for the control-change-UEOT subsystem (Cycle 1).
/// Returns the effective controller id of `obj`: the per-object
/// `controller_override` if set, else `obj.controller` (the printed
/// baseline / Aura-swapped controller cache).
///
/// Cycle 1 (this lane) provides the helper only — read sites in
/// `state_ops_legal` / `state_ops_combat` / `state_ops_decisions` /
/// `state_ops_actions` still read raw `obj.controller`. Cycle 2
/// reroutes those call sites through this helper; Cycle 3 wires
/// the bridge round-trip; Cycle 4 mints the override from
/// Mutinous-Massacre / Systems-Override resolution arms. The
/// helper is `pub(crate)` so future cycles' module-internal call
/// sites can adopt it without exporting an unstable surface.
///
/// Mirrors Python's `effective_controller(state, obj)` in
/// `src/mtg_bot/game.py:999` — the layer-2 control-change registry
/// walk that backs Python's source-of-truth control-change-UEOT
/// implementation. The Python side keeps `obj.controller_id` as a
/// write-through cache of `effective_controller`'s value (eager
/// invalidation at every grant + cleanup site); Rust's
/// `controller_override` field keeps the override as the
/// source-of-truth and the helper is the read path. The
/// observable surface is identical from a snapshot perspective —
/// both engines answer "who controls X right now" with the same
/// id — but the storage shape differs (Python: registry list +
/// cache; Rust: per-object override field).
pub(crate) fn effective_controller_id(obj: &GameObject) -> u8 {
    obj.controller_override.unwrap_or(obj.controller)
}

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

    /// Round-27 UEOT framework primitive — clear every UEOT modifier
    /// on every battlefield permanent (CR 514.3). Returns the number
    /// of objects whose `ueot_modifiers` Vec was non-empty before
    /// draining. Skips the `Arc::make_mut` clone on a battlefield
    /// where no permanent has any UEOT modifier — the warm path
    /// (no UEOT effect active at end-of-turn) pays zero allocation.
    ///
    /// Callable independently for tests that want to assert the
    /// UEOT-clear in isolation from the hand-size + damage-clear
    /// passes; `cleanup_step_for_active_player` also fires this
    /// AFTER CR 514.1 + CR 514.2 (the CR 514 numbered-rule order).
    ///
    /// **Closeable parity gap**: this is the round-27 closure of G7.
    fn clear_ueot_modifiers_all(&mut self) -> usize {
        let mut cleared: usize = 0;
        for player in self.players.iter_mut() {
            // Warm-path skip: nothing to clear on this battlefield.
            if player
                .battlefield
                .iter()
                .all(|o| o.ueot_modifiers.is_empty())
            {
                continue;
            }
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if !obj.ueot_modifiers.is_empty() {
                    cleared += 1;
                    obj.ueot_modifiers.clear();
                }
            }
        }
        cleared
    }

    /// **Control-change-UEOT subsystem — Cycle 1 (2026-05-21).**
    /// Drain every per-object `controller_override` back to `None`
    /// across both players' battlefields. CR 514.3 / CR 613 layer-2
    /// "until end of turn" expiration for control-changing effects
    /// (Mutinous Massacre, Systems Override, Threaten / Act of
    /// Treason shape).
    ///
    /// Returns the number of objects whose `controller_override`
    /// was `Some(_)` before draining. Skips the `Arc::make_mut`
    /// clone on a battlefield where no permanent has a UEOT
    /// controller override — the warm path (no UEOT control-grant
    /// active at end-of-turn) pays zero allocation. Same shape as
    /// `clear_ueot_modifiers_all`.
    ///
    /// Callable independently for tests that want to assert the
    /// drain in isolation; `cleanup_step_for_active_player` also
    /// fires this after CR 514.3 (UEOT modifier expiration).
    ///
    /// Mirrors Python's controller-refresh loop in
    /// `src/mtg_bot/turn.py:run_cleanup_step` (the
    /// `for obj in state.battlefield: obj.controller_id =
    /// effective_controller(state, obj)` pass that runs after
    /// `expire_continuous_effects("cleanup")`). The Rust shape is
    /// lighter because the override IS the source-of-truth — no
    /// registry walk + cache refresh; just clear the override.
    fn clear_controller_overrides_all(&mut self) -> usize {
        let mut cleared: usize = 0;
        for player in self.players.iter_mut() {
            // Warm-path skip: nothing to clear on this battlefield.
            if player.battlefield.iter().all(|o| o.controller_override.is_none()) {
                continue;
            }
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.controller_override.is_some() {
                    cleared += 1;
                    obj.controller_override = None;
                }
            }
        }
        cleared
    }

    /// **Control-change-UEOT subsystem — Cycle 1 (2026-05-21).**
    /// Read-side accessor — returns the effective controller of the
    /// object with id `obj_id`, walking BOTH players' battlefields.
    /// Resolves the per-object `controller_override` (Some) or the
    /// printed/Aura-cached `controller` (None). Raises `IndexError`
    /// when no battlefield permanent matches `obj_id` — the caller
    /// is asking "who controls this object on the battlefield" and
    /// a missing-permanent answer is genuinely ambiguous (off-bf
    /// objects have no controller in the CR sense; they revert to
    /// owner on zone change).
    ///
    /// Exposed for Python so the cross-engine harness + future
    /// Cycle 3 bridge round-trip + Cycle 4 minting tests can
    /// observe the override surface symmetrically with Python's
    /// `effective_controller(state, obj)`. Cycle 1 does NOT
    /// reroute Rust's internal `.controller` read sites through
    /// this method — that's Cycle 2's invasive cross-module
    /// rerouting.
    ///
    /// Mirrors Python's `effective_controller(state, obj)` in
    /// `src/mtg_bot/game.py:999`.
    fn effective_controller_id(&self, obj_id: u32) -> PyResult<u8> {
        for player in &self.players {
            for obj in player.battlefield.iter() {
                if obj.id == obj_id {
                    return Ok(effective_controller_id(obj));
                }
            }
        }
        Err(PyIndexError::new_err(format!(
            "no battlefield permanent with id {obj_id}"
        )))
    }

    /// **Control-change-UEOT subsystem — Cycle 1 (2026-05-21).**
    /// Stamp a UEOT controller override on the battlefield permanent
    /// with id `target_id`. The override sets that object's
    /// "effective controller" to `new_controller_id` until the next
    /// cleanup-step drain (CR 514.3 / CR 613 layer-2 expiration).
    ///
    /// Substrate-only in Cycle 1: this method writes the override
    /// field directly, with no companion registry entry, no
    /// re-routing of read sites, and no Effect-variant integration.
    /// Cycle 2 reroutes call sites; Cycle 4 wires the Mutinous
    /// Massacre / Systems Override `Effect` arms that call this.
    ///
    /// **Arguments:**
    ///
    /// * `target_id` — GameObject id of the battlefield permanent
    ///   to override. Looked up across both players' battlefields.
    /// * `new_controller_id` — Player id `0` or `1` who will be
    ///   the effective controller until cleanup.
    ///
    /// **Returns** `True` when the override was applied,
    /// `False` when the target permanent isn't on the battlefield
    /// (mirroring Python's `if target is None: return` defensive
    /// silent no-op at the head of `grant_control_change_ueot` —
    /// the Mutinous-Massacre `for o in state.battlefield` loop
    /// constructs the call from a battlefield iteration, so the
    /// not-found branch is paranoia, not a hot path).
    ///
    /// **Raises** `IndexError` when `new_controller_id` is out of
    /// range (must be 0 or 1).
    ///
    /// **Idempotent same-controller short-circuit.** If the
    /// effective controller is ALREADY `new_controller_id` (either
    /// because no override is set and `obj.controller ==
    /// new_controller_id`, or because the override is already set
    /// to `new_controller_id`), returns `True` WITHOUT mutating —
    /// matches Python's `if effective_controller(...) ==
    /// new_controller_id: return` short-circuit.
    ///
    /// **CoW-safe.** Routes the battlefield mutation through
    /// `Arc::make_mut`, preserving the CoW invariant. A cloned
    /// state's battlefield is unaffected by this call on the
    /// source.
    ///
    /// Mirrors Python's `GameState.grant_control_change_ueot` in
    /// `src/mtg_bot/game.py:2949`.
    fn grant_control_change_ueot(
        &mut self,
        target_id: u32,
        new_controller_id: u8,
    ) -> PyResult<bool> {
        if new_controller_id >= 2 {
            return Err(PyIndexError::new_err(format!(
                "new_controller_id must be 0 or 1, got {new_controller_id}"
            )));
        }
        // Find which player's battlefield carries the target. Read
        // the current effective controller first so we can no-op
        // before paying the Arc::make_mut cost.
        let mut found_player: Option<usize> = None;
        let mut current_effective: u8 = 0;
        for (pidx, player) in self.players.iter().enumerate() {
            for obj in player.battlefield.iter() {
                if obj.id == target_id {
                    found_player = Some(pidx);
                    current_effective = effective_controller_id(obj);
                    break;
                }
            }
            if found_player.is_some() {
                break;
            }
        }
        let Some(pidx) = found_player else {
            // Defensive silent no-op — Python returns without raising.
            return Ok(false);
        };
        // Idempotent same-controller short-circuit. Skip the CoW
        // clone when the override would be a no-op.
        if current_effective == new_controller_id {
            return Ok(true);
        }
        let zone: &mut Vec<GameObject> = Arc::make_mut(
            &mut self.players[pidx].battlefield,
        );
        for obj in zone.iter_mut() {
            if obj.id == target_id {
                obj.controller_override = Some(new_controller_id);
                return Ok(true);
            }
        }
        // Unreachable: the read-pass above found the object on this
        // player's battlefield, and Arc::make_mut returns a mutable
        // ref to the same Vec. A no-op fallthrough is still safer
        // than panicking under PyO3.
        Ok(false)
    }

    /// Run the cleanup step for the current `active_player` —
    /// composition of CR 514.1 (hand-size discard, max=7 hard-coded),
    /// CR 514.2 (clear all marked damage on both battlefields), and
    /// CR 514.3 (until-end-of-turn modifier expiration).
    ///
    /// Order: hand-size discard FIRST, damage clearance second,
    /// UEOT-modifier expiration third (matches CR 514's numbered
    /// sub-rules and the source-of-truth loop in
    /// `src/mtg_bot/turn.py:run_cleanup_step`). See module docstring
    /// on why ordering matters.
    ///
    /// Returns a Python dict `{"discarded": N, "damage_cleared": M,
    /// "ueot_cleared": K}`:
    /// - `discarded`: number of cards moved from active player's
    ///   hand to graveyard (0 if hand was already <= 7).
    /// - `damage_cleared`: count of permanents (across both
    ///   battlefields) whose marked damage was non-zero before the
    ///   call. A no-op cleanup (no damage anywhere) reports 0.
    /// - `ueot_cleared`: count of permanents (across both
    ///   battlefields) whose `ueot_modifiers` Vec was non-empty
    ///   before draining. Round-27 addition.
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

        // (3) CR 514.3 — drain every UEOT modifier on every
        // battlefield permanent (round-27 closure of parity gap G7).
        // Same CoW shape as the damage-clear pass — skip the
        // `Arc::make_mut` clone when no object has any UEOT modifier.
        let mut ueot_cleared: usize = 0;
        for player in self.players.iter_mut() {
            if player
                .battlefield
                .iter()
                .all(|o| o.ueot_modifiers.is_empty())
            {
                continue;
            }
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if !obj.ueot_modifiers.is_empty() {
                    ueot_cleared += 1;
                    obj.ueot_modifiers.clear();
                }
            }
        }

        // (3b) Control-change-UEOT subsystem — Cycle 1 (2026-05-21):
        // drain every per-object `controller_override` back to `None`.
        // CR 514.3 / CR 613 layer-2 "until end of turn" expiration
        // for control-changing effects (Mutinous Massacre, Systems
        // Override, Threaten / Act of Treason shape). Substrate-only
        // in Cycle 1 — no live mint site has landed yet, but the
        // drain is part of the substrate contract: if a future
        // cycle mints an override and forgets to clear it, the
        // override would silently leak across turns.
        //
        // Same CoW shape as the UEOT-modifier drain above — skip
        // the `Arc::make_mut` clone when no object has an override.
        let mut controller_overrides_cleared: usize = 0;
        for player in self.players.iter_mut() {
            if player.battlefield.iter().all(|o| o.controller_override.is_none()) {
                continue;
            }
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.controller_override.is_some() {
                    controller_overrides_cleared += 1;
                    obj.controller_override = None;
                }
            }
        }

        // (4) ETB-trigger Slice 1 — reset the once-per-turn ledger
        // so any `once_per_turn = true` triggered ability becomes
        // re-fireable next turn. Mirrors Python's
        // `state.once_per_turn_fired.clear()` at cleanup.
        // Substrate-only in Slice 1: no Slice-1 trigger fixture
        // uses the gate, but the reset is part of the substrate
        // contract.
        self.triggered_this_turn.clear();

        // (5) CE20 Slice 5 (Walker loyalty Slice C) — reset the
        // once-per-turn-per-walker ledger so every walker becomes
        // re-activatable next turn. CR 606.3 is "this turn"-scoped;
        // the cleanup step is the natural reset point (same site as
        // the ETB once-per-turn ledger above). A walker that died
        // mid-turn has a stale ledger entry that survives until
        // here; the clear is unconditional (cheaper than walking
        // bf to filter-stamp).
        self.loyalty_activated_this_turn.clear();

        // (3c) Darkness-shape combat-damage prevention substrate —
        // Cycle 1 (2026-05-21): drain the state-level
        // `prevent_all_combat_damage_this_turn` flag back to `false`.
        // CR 514.3 — "all damage prevention effects with duration
        // 'this turn' end at the cleanup step". Mirrors Python's
        // `state.prevent_combat_damage_this_turn = False` at
        // `src/mtg_bot/turn.py:1102`. Substrate-only in Cycle 1 —
        // no live mint site has landed yet (no Effect arm sets the
        // flag), but the drain is part of the substrate contract:
        // if a future cycle stamps the flag and forgets to clear
        // it, the prevention would silently leak across turns.
        //
        // Cheaper than the per-permanent UEOT-modifier drain above
        // — this is a single bool flip, no CoW / Arc::make_mut
        // bookkeeping. Returns true iff the flag was set before
        // the drain (so the summary dict reports a drain-happened
        // signal that future Cycle 2/3/4 assertions can pin).
        let prevent_all_combat_damage_drained: bool =
            self.prevent_all_combat_damage_this_turn;
        self.prevent_all_combat_damage_this_turn = false;

        let d = PyDict::new(py);
        d.set_item("discarded", discarded)?;
        d.set_item("damage_cleared", damage_cleared)?;
        d.set_item("ueot_cleared", ueot_cleared)?;
        // Control-change-UEOT subsystem — Cycle 1 (2026-05-21):
        // surface the drain count for tests + future Cycle 2/3/4
        // assertions. Pre-Cycle-1 cleanup summaries never carried
        // this key; callers reading the dict with a default fallback
        // (`.get("controller_overrides_cleared", 0)`) stay back-
        // compat. Substrate-only — every value will be 0 until
        // Cycle 4 mints overrides.
        d.set_item("controller_overrides_cleared", controller_overrides_cleared)?;
        // Darkness-shape combat-damage prevention substrate —
        // Cycle 1 (2026-05-21): surface the drain signal alongside
        // the other CR-514.3 drain counts. `true` iff the flag was
        // set before the drain (Cycle 1's substrate-only landing
        // means this is `false` on every pre-Cycle-3 trajectory —
        // no mint site exists yet). Callers reading the dict with
        // a default fallback (`.get("prevent_all_combat_damage_
        // drained", False)`) stay back-compat.
        d.set_item(
            "prevent_all_combat_damage_drained",
            prevent_all_combat_damage_drained,
        )?;
        Ok(d)
    }

    /// **Round-17: end-of-turn wrap composition primitive.** Run the
    /// cleanup step for the current active player, then advance phase
    /// to the NEXT turn's UNTAP step. After this returns:
    ///
    /// - `turn_number` is incremented by 1
    /// - `phase` is `PHASE_UNTAP` (0)
    /// - `active_player` is swapped (CR 500.7 — turn rolls to the
    ///   non-active player at the cleanup -> untap wrap)
    /// - every permanent on the battlefield has its marked damage
    ///   zeroed (CR 514.2)
    /// - the previously-active player's hand is at most 7 cards (CR 514.1)
    /// - the new active player's permanents have lost summoning
    ///   sickness (CR 302.1, fired automatically by `advance_phase`'s
    ///   round-14 untap-step hook)
    ///
    /// Composes:
    /// 1. `cleanup_step_for_active_player()` (same file — handles
    ///    CR 514.1 hand-size + CR 514.2 damage clear)
    /// 2. `set_phase(PHASE_CLEANUP)` to land on cleanup
    /// 3. `advance_phase()` to wrap CLEANUP -> UNTAP, which fires:
    ///    - turn-counter bump
    ///    - active-player swap
    ///    - the round-2 untap sweep on the new active player
    ///    - the round-14 summoning-sick clear on the new active player
    ///
    /// Order matters: `cleanup_step_for_active_player` is keyed on
    /// `self.active_player` which is the OLD active player (the one
    /// whose turn is ending), so the cleanup must run BEFORE the
    /// CLEANUP -> UNTAP wrap flips active. Setting phase to CLEANUP
    /// first ensures `advance_phase`'s `phase == PHASE_CLEANUP` branch
    /// fires the wrap regardless of the engine's prior phase — callers
    /// can invoke `end_turn_and_start_next` from any phase to skip
    /// straight to the next turn's UNTAP.
    ///
    /// **Cross-file dispatch (round-16 `Bound<'_, Self>` pattern).**
    /// `advance_phase` is defined in `state_ops_steps.rs` and is
    /// Rust-private to that module under `multiple-pymethods` (E0624
    /// if called as `self.advance_phase()` from this file). The fix is
    /// to take `Bound<'_, Self>` rather than `&mut self` and dispatch
    /// through the Python attribute layer with `call_method0`. PyO3's
    /// `Bound<T>` is a borrow-checked handle to the Python object;
    /// `borrow_mut()`'s cost is identical to the borrow the
    /// `#[pymethods]` macro would have generated. See
    /// `state_ops_stack.rs::process_stack_until_empty` for the
    /// reference implementation.
    ///
    /// `cleanup_step_for_active_player` lives in this same file so a
    /// direct `borrow_mut().cleanup_step_for_active_player(py)?` call
    /// is allowed; routing it through `call_method0` keeps the dispatch
    /// shape uniform with the cross-file `advance_phase` call. We use
    /// `call_method0` for both for consistency.
    fn end_turn_and_start_next(slf: Bound<'_, Self>) -> PyResult<()> {
        // (1) Run cleanup step for the OLD active player. This must
        // happen before the active-player swap; CR 514 fires during
        // the cleanup step of the player whose turn is ending.
        slf.call_method0("cleanup_step_for_active_player")?;

        // (2) Land on the CLEANUP phase so the next advance_phase tick
        // wraps CLEANUP -> UNTAP. Going through set_phase rather than
        // walking advance_phase forward is intentional: a single
        // CLEANUP -> UNTAP wrap is the semantically right "end of turn"
        // step, regardless of how many phases the engine has already
        // advanced through this turn.
        slf.borrow_mut().phase = PHASE_CLEANUP;

        // (3) Advance once to wrap CLEANUP -> UNTAP. This fires:
        //     - turn_number += 1
        //     - active_player swap
        //     - round-2 untap sweep on the new active player
        //     - round-14 summoning-sick clear on the new active player
        slf.call_method0("advance_phase")?;

        Ok(())
    }
}

/// **Darkness-shape combat-damage prevention substrate — Cycle 1
/// (2026-05-21).** Standalone-impl-block append for the per-state
/// `prevent_all_combat_damage_this_turn` flag's PyO3 surface +
/// standalone cleanup-step drain primitive. New impl block (not
/// touching the surrounding control-change-UEOT impl block above)
/// to keep this lane's diff append-only against concurrent agents'
/// edits to the existing block — the crate is configured with
/// `multiple-pymethods` so multiple `#[pymethods] impl GameState`
/// blocks coexist by design (see the per-`state_ops_*.rs` module
/// pattern).
///
/// The substrate provides:
/// - `grant_prevent_all_combat_damage_this_turn` — idempotent
///   set-to-true setter (no args; mirrors Python's
///   `state.prevent_combat_damage_this_turn = True` write at
///   `effects/generic/stack_control.py:266`).
/// - `prevent_all_combat_damage_this_turn` — `#[getter]` over the
///   flag; lets the future Cycle 2 combat-damage routing site +
///   the Cycle 3 bridge round-trip + the cross-engine harness
///   read the state symmetrically with Python's
///   `state.prevent_combat_damage_this_turn` attribute access.
/// - `clear_prevent_all_combat_damage_this_turn` — standalone
///   drain primitive returning the pre-drain value as a bool so
///   tests can assert the drain in isolation without composing
///   the full `cleanup_step_for_active_player` summary.
#[pymethods]
impl GameState {
    /// **Darkness-shape combat-damage prevention substrate — Cycle 1
    /// (2026-05-21).** Stamp the state-level "all combat damage is
    /// prevented this turn" flag.
    ///
    /// Substrate-only in Cycle 1: this method writes the flag
    /// directly. No live caller routes through it yet — Cycle 3's
    /// `Effect::PreventAllCombatDamageThisTurn` resolve arm
    /// (Darkness, `{B}` Instant) will call this from Rust;
    /// cross-engine tests / future Cycle 3 bridge round-trip can
    /// call it directly to mirror Python's
    /// `PreventAllCombatDamageThisTurn.resolve` write at
    /// `src/mtg_bot/effects/generic/stack_control.py:266`.
    ///
    /// **Idempotent set-to-true.** Calling this on a state that
    /// already has the flag set is a no-op (the flag stays `true`,
    /// no observable change). Mirrors the Python write site (a
    /// bare `state.prevent_combat_damage_this_turn = True`) which
    /// is also idempotent.
    ///
    /// **No CoW concern.** The flag is a `bool` on `GameState`,
    /// not a per-zone field, so the assignment is a single byte
    /// write — no `Arc::make_mut` is needed. A cloned state's
    /// flag is independent of the source's (the `Clone` impl
    /// already carries the value forward at clone time).
    ///
    /// **No args.** Darkness's effect is unconditional: prevent
    /// ALL combat damage this turn. Future variants (e.g. "prevent
    /// combat damage this turn dealt to you / your creatures")
    /// would need a NEW per-player or per-target flag, not a
    /// re-parameterization of this one.
    fn grant_prevent_all_combat_damage_this_turn(&mut self) {
        self.prevent_all_combat_damage_this_turn = true;
    }

    /// **Darkness-shape combat-damage prevention substrate — Cycle 1
    /// (2026-05-21).** Read-only getter for the state-level "all
    /// combat damage is prevented this turn" flag.
    ///
    /// Substrate-only in Cycle 1: no Rust call site consumes the
    /// flag yet — Cycle 2 will reroute combat-damage routing
    /// through the preventer; Cycle 3 wires the bridge round-trip.
    /// Cross-engine tests + the harness can read the flag via this
    /// getter to mirror Python's attribute-access read pattern at
    /// `combat.py:479` (`if state.prevent_combat_damage_this_turn:
    /// ...`).
    ///
    /// The PyO3 `#[getter]` annotation makes the field reachable
    /// from Python as `state.prevent_all_combat_damage_this_turn`
    /// (no parens — attribute access matching Python's read site
    /// shape), keeping the Rust API ergonomic for cross-engine
    /// parity assertions.
    #[getter]
    fn prevent_all_combat_damage_this_turn(&self) -> bool {
        self.prevent_all_combat_damage_this_turn
    }

    /// **Darkness-shape combat-damage prevention substrate — Cycle 1
    /// (2026-05-21).** Standalone drain primitive — reset the
    /// state-level "all combat damage is prevented this turn" flag
    /// back to `false` and return the pre-drain value as a bool.
    /// CR 514.3 — "all damage prevention effects with duration
    /// 'this turn' end at the cleanup step".
    ///
    /// Returns `true` iff the flag was set before the drain.
    /// Callable independently for tests that want to assert the
    /// drain in isolation from the full `cleanup_step_for_active_
    /// player` composition; `cleanup_step_for_active_player` ALSO
    /// drains the flag inline (single bool flip, no allocation
    /// concern — both call sites share the substrate contract).
    ///
    /// Mirrors Python's `state.prevent_combat_damage_this_turn =
    /// False` at `src/mtg_bot/turn.py:1102`.
    fn clear_prevent_all_combat_damage_this_turn(&mut self) -> bool {
        let was_set = self.prevent_all_combat_damage_this_turn;
        self.prevent_all_combat_damage_this_turn = false;
        was_set
    }
}
