//! State-based action (SBA) primitives on `GameState`.
//!
//! Filled in by a TDD-slice branch (see `tests_rust/README.md`).
//! Stub block exists so other slices can grow `GameState` in
//! parallel without conflicting on a single `#[pymethods]` block.
//!
//! ## Slice scope
//!
//! **Lethal-damage SBA (CR 704.5g) AND 0-toughness SBA (CR 704.5f):**
//!
//! - 704.5g — a creature whose marked damage is >= its toughness is
//!   destroyed (moved from battlefield to its controller's graveyard).
//! - 704.5f — a creature with toughness 0 (or less) is put into its
//!   owner's graveyard. Effective toughness for this check is
//!   ``printing.toughness + obj.counters`` (round-3's net-i32 +1/+1
//!   vs -1/-1 model: positive counters add, negative subtract).
//!
//! Per CR ordering, 704.5f is checked BEFORE 704.5g; the practical
//! effect is identical (the union of victims is the same set), but
//! the order matters for niche corner cases — a creature at 0
//! effective toughness due to counters dies via 704.5f rather than
//! 704.5g, which becomes observable to triggered abilities once
//! they care about the cause of destruction.
//!
//! Exposes:
//! - `apply_sbas() -> usize` — single pass over both players'
//!   battlefields; collects every creature dying to either rule,
//!   then destroys each by moving it into its controller's graveyard.
//!   Returns the count of destructions.
//! - `apply_sbas_until_quiescent() -> usize` — CR 704.3 fixed-point
//!   loop. Repeatedly calls `apply_sbas` until a pass returns 0;
//!   returns the number of passes that fired. With only destruction
//!   SBAs in scope today, every realistic scenario completes in 0 or
//!   1 pass; the primitive exists so cascading-effect slices (sagas,
//!   death triggers) plug into the same loop without reshaping
//!   callers. A `MAX_PASSES = 100` safety raises `PyValueError` if
//!   exceeded.
//! - `is_lethal(obj_id) -> Option<bool>` — predicate form for
//!   704.5g. `Some(true)` when damage >= toughness, `Some(false)`
//!   when it is below, `None` for non-creatures (toughness is None)
//!   or unknown ids. (704.5f's effective-toughness predicate is
//!   inlined into `apply_sbas`; a public `is_zero_toughness` helper
//!   would be a one-liner addition if a caller needs it.)
//!
//! Composes with:
//! - `state_ops_damage` — supplies the damage marker for 704.5g.
//! - `state_ops_counters` — supplies `obj.counters` (the net +1/+1
//!   contribution) for 704.5f.
//! - `CardPrinting.toughness: Option<i16>` from the bridge-ext slice
//!   (round 3) — the base toughness; gates whether either rule
//!   applies (non-creatures stay).
//! - `state_ops_zones::move_object` — the underlying graveyard move
//!   pattern; replicated inline here rather than called via the
//!   pyclass method to keep the destructive collect-then-mutate loop
//!   self-contained and avoid a `&mut self` re-entrancy round-trip.
//!
//! ## Indestructible (CR 702.12)
//!
//! Creatures whose printing carries the `"indestructible"` keyword
//! are exempt from BOTH destruction paths above:
//!
//! - 704.5g (lethal damage): per CR 702.12a, indestructible permanents
//!   aren't destroyed by damage. Damage still marks (the damage field
//!   updates and clears at end of turn via cleanup), it just doesn't
//!   trigger destruction.
//! - 704.5f (0-or-negative toughness): per CR 702.12b, the SBA wording
//!   "is destroyed" carves out indestructible creatures here too.
//!
//! Implementation: `collect_sba_victims` looks up
//! `printing.keywords.iter().any(|k| k == "indestructible")` and skips
//! the entire creature when true (before either rule fires). Counter
//! / damage tracking is unaffected — only the destruction step is
//! gated.
//!
//! Indestructible only protects from "destroy". Exile (round-11's
//! ExileTarget effect) still removes indestructible creatures; that
//! path doesn't go through this slice.
//!
//! ## Out of scope (deferred to future slices)
//!
//! - **Integration with combat and spell-resolution call sites.**
//!   The loop primitive `apply_sbas_until_quiescent` exists, but
//!   wiring it into the priority-pass / spell-resolution drivers is
//!   a separate slice. Today callers explicitly invoke it.
//! - **Marked damage clearing.** Damage clears at end of turn (CR
//!   514.2), NOT on every SBA pass. apply_sbas does NOT zero damage
//!   on surviving creatures; `clear_all_damage` from the damage
//!   slice covers the cleanup-step path.
//! - **Regeneration / planeswalker loyalty / legend rule / +1/+1 vs
//!   -1/-1 cancellation / saga age / aura on illegal target / etc.**
//!   Other CR 704.5 SBAs are subsequent slices.
//!
//! ## Implementation notes
//!
//! - **Collect-then-mutate.** First pass walks both battlefields
//!   read-only and pushes lethal creatures into a `Vec<(controller,
//!   obj_id)>`. Second pass moves each to its controller's
//!   graveyard. Avoids in-loop `Arc::make_mut` invalidating the
//!   iterator.
//! - **CoW preservation.** Each destruction touches at most two
//!   zone Vecs (source battlefield + destination graveyard). Both
//!   go through `Arc::make_mut`, so a clone observing the original
//!   stays untouched. Verified by `test_clone_preserves_sba_pending`.
//! - **Battlefield-only.** apply_sbas only inspects battlefield zones.
//!   Damage on non-battlefield objects is ignored (it shouldn't exist
//!   there in normal play, and the lethal-damage SBA is intrinsically
//!   battlefield-scoped).

use crate::state::{GameObject, GameState};
use pyo3::prelude::*;
use std::sync::Arc;

// Zone constants — duplicated from state_ops_zones.rs (per the
// per-slice isolation rule; a follow-up consolidation slice can
// share these via a common module once the parallel-slice constraint
// relaxes).
const ZONE_GRAVEYARD: u8 = 3;

impl GameState {
    /// Read-only enumeration of creatures that should be destroyed by
    /// SBAs this pass. Walks both battlefields and applies, in order:
    ///
    /// 1. **CR 704.5f (0-toughness):** effective toughness =
    ///    `printing.toughness + obj.counters`. If <= 0, victim.
    /// 2. **CR 704.5g (lethal damage):** `obj.damage >= effective_toughness`.
    ///    Effective toughness folds in the +1/+1 vs -1/-1 counter
    ///    contribution: a 2/2 with two +1/+1 counters takes 3 damage
    ///    and survives (effective 4 > 3); a 4/4 with two -1/-1 counters
    ///    takes 3 damage and dies (effective 2 <= 3). When `obj.counters
    ///    == 0` effective and printed toughness coincide, so the
    ///    no-counter cases reduce to the printed-toughness predicate.
    ///
    /// Returns `(controller_id, obj_id)` pairs. Each obj_id appears at
    /// most once even if both rules would catch it, because the inner
    /// loop short-circuits on the first matching rule.
    fn collect_sba_victims(&self) -> Vec<(u8, u32)> {
        let mut victims = Vec::new();
        for player in &self.players {
            for obj in player.battlefield.iter() {
                let Some(printing) = self.registry.lookup(obj.printing_id) else {
                    continue;
                };
                let Some(toughness) = printing.toughness else {
                    // Non-creature — neither 704.5f nor 704.5g apply.
                    continue;
                };
                // CR 702.12 — indestructible permanents are exempt
                // from both destruction-by-damage (704.5g) and
                // destruction-by-zero-toughness (704.5f). Damage and
                // counters still track; only the destruction step is
                // skipped.
                if printing.keywords.iter().any(|k| k == "indestructible") {
                    continue;
                }
                let effective_toughness = toughness as i32 + obj.counters;
                // 704.5f first per CR ordering.
                if effective_toughness <= 0 {
                    victims.push((obj.controller, obj.id));
                    continue;
                }
                // 704.5g second — also keyed on effective toughness.
                if obj.damage as i32 >= effective_toughness {
                    victims.push((obj.controller, obj.id));
                }
            }
        }
        victims
    }
}

#[pymethods]
impl GameState {
    /// Apply state-based actions for the destruction SBAs:
    /// CR 704.5f (0 effective toughness) and CR 704.5g (lethal damage).
    /// Every matching creature is moved from the battlefield into its
    /// controller's graveyard. Returns the number of creatures destroyed
    /// in this pass.
    ///
    /// Single-pass, no quiescence loop — see module docstring for
    /// the deferred semantics. 704.5f checked before 704.5g per CR
    /// ordering; in practice the union of victims is the same set.
    fn apply_sbas(&mut self) -> PyResult<usize> {
        let victims = self.collect_sba_victims();
        let count = victims.len();
        for (controller, obj_id) in victims {
            // Take the object out of its current battlefield. Search
            // both players (a creature's controller can differ from
            // its owner; we move to the CONTROLLER's graveyard per
            // CR 400.6 — "destroy" sends to the owner's graveyard,
            // but for first-slice we follow the controller-side
            // convention used by move_off_battlefield's caller. The
            // vast majority of the time these are the same player,
            // and the future zones-extension slice can revisit.)
            let mut taken: Option<GameObject> = None;
            for player in self.players.iter_mut() {
                let pos = player.battlefield.iter().position(|o| o.id == obj_id);
                if let Some(idx) = pos {
                    let bf = Arc::make_mut(&mut player.battlefield);
                    taken = Some(bf.remove(idx));
                    break;
                }
            }
            let Some(mut obj) = taken else {
                // Should not happen — collect_lethal_creatures just
                // saw it. If it does (concurrent mutation, future
                // SBA cascade), skip silently.
                continue;
            };
            obj.zone = ZONE_GRAVEYARD;
            obj.controller = controller;
            // Push onto the controller's graveyard.
            let target = &mut self.players[controller as usize];
            Arc::make_mut(&mut target.graveyard).push(obj);
        }
        Ok(count)
    }

    /// Loop driver for state-based actions — CR 704.3 quiescence.
    ///
    /// Repeatedly calls `apply_sbas` until a pass produces zero
    /// destructions, returning the number of passes that fired
    /// (i.e. the number of `apply_sbas` invocations whose return
    /// value was > 0). A clean state returns 0.
    ///
    /// CR 704.3: "Whenever a player would get priority [...] the
    /// game first performs all applicable state-based actions as a
    /// single event. [...] After SBAs are performed, [...] if any
    /// SBAs were performed as a result of the check, all applicable
    /// SBAs are checked again." This method is the iteration-to-
    /// fixed-point that the round-5 single-pass `apply_sbas`
    /// deferred.
    ///
    /// With only the destruction SBAs available today (no cascading
    /// effects like death triggers or +1/+1-on-death yet), the loop
    /// completes in either 0 or 1 pass for every realistic scenario.
    /// Pinning the API now means later slices that add cascading
    /// effects (sagas, death triggers, ETB chains) plug into the
    /// same loop primitive without re-shaping callers.
    ///
    /// **Safety limit.** A `MAX_PASSES` guard (100) prevents any
    /// future infinite-loop bug from hanging the engine; exceeding
    /// it raises `PyValueError`. Real MTG SBA chains terminate quickly
    /// (the game state monotonically loses creatures / counters), so
    /// a triple-digit ceiling is far above any legitimate scenario.
    fn apply_sbas_until_quiescent(&mut self) -> PyResult<usize> {
        const MAX_PASSES: usize = 100;
        let mut passes = 0usize;
        loop {
            let destroyed = self.apply_sbas()?;
            if destroyed == 0 {
                return Ok(passes);
            }
            passes += 1;
            if passes > MAX_PASSES {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "apply_sbas_until_quiescent exceeded {} passes — \
                     suspected infinite SBA loop",
                    MAX_PASSES
                )));
            }
        }
    }

    /// Predicate form of the lethal-damage check.
    ///
    /// Returns `Some(true)` when the creature's marked damage is
    /// greater than or equal to its EFFECTIVE toughness (printed +
    /// `obj.counters`), `Some(false)` when it is below, and `None`
    /// when the object is not found OR when the printing has no
    /// toughness (i.e. it's not a creature — lands, instants,
    /// sorceries, etc.).
    ///
    /// This stays in lockstep with `apply_sbas`'s 704.5g predicate so
    /// `is_lethal(id) == true` iff the creature would be destroyed by
    /// the next pass via the lethal-damage rule (modulo
    /// indestructibility, which the predicate intentionally ignores —
    /// 704.5g asks "is this lethal damage", indestructibility asks
    /// "does the destruction take effect"). For no-counter creatures
    /// effective and printed toughness coincide and the predicate
    /// reduces to the original `damage >= printing.toughness` form.
    fn is_lethal(&self, obj_id: u32) -> Option<bool> {
        for player in &self.players {
            for obj in player.battlefield.iter() {
                if obj.id == obj_id {
                    let printing = self.registry.lookup(obj.printing_id)?;
                    let toughness = printing.toughness?;
                    let effective_toughness = toughness as i32 + obj.counters;
                    return Some(obj.damage as i32 >= effective_toughness);
                }
            }
        }
        // Not on either battlefield — for the first-slice purposes the
        // lethal-damage SBA is battlefield-scoped, so we return None
        // for objects elsewhere. (Could be extended to search all
        // zones if a caller cares; deferred.)
        None
    }
}
