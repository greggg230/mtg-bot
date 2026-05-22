//! Stack primitives on `GameState`.
//!
//! Mirrors `src/mtg_bot/game.py:GameState.stack` (a list of
//! `StackItem` objects) and the stack interactions in
//! `src/mtg_bot/turn.py`. The Rust side keeps `stack: Vec<StackItem>`
//! on `GameState` (added by the foundation commit). Top of stack =
//! LAST element of the Vec for O(1) push/pop, mirroring the Python
//! list-as-stack convention.
//!
//! ## Slice scope
//!
//! - **No resolution semantics in this slice.** What happens when
//!   an effect resolves depends on the effect grammar, which is the
//!   bridge's domain. This slice exposes `pop_top_of_stack()`
//!   returning the StackItem dict; resolution itself is a future
//!   slice that interprets the effect.
//! - **No counter-spells in this slice.** Counter target spell needs
//!   a "remove this specific item from the stack" primitive — which
//!   we add as `remove_from_stack(obj_id)` — but we don't enforce
//!   countering rules, just expose the surface.
//! - **First-slice StackItem shape: `{source_obj_id, controller}`
//!   only.** Real stack items will need more (target list, choices,
//!   modal mode); future slices extend.
//!
//! The Python-facing dict shape is intentionally minimal so callers
//! that read it today (the bridge layer) can be extended additively
//! when more fields land — they'll just see new keys on the same
//! dict, no API break.

use crate::state::{GameState, StackItem};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymethods]
impl GameState {
    /// Number of items currently on the stack. O(1).
    fn stack_size(&self) -> usize {
        self.stack.len()
    }

    /// True iff the stack has zero items. Cheaper-named alias for
    /// `stack_size() == 0`; matches Python's idiomatic
    /// `not state.stack` usage.
    fn is_stack_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Push a new stack item. Top of stack = LAST element of the
    /// Vec, so this is `Vec::push` — O(1) amortized.
    ///
    /// `source_obj_id` is the GameObject id of the spell or
    /// activated-ability source; `controller` is the player who
    /// owns the stack item (for priority / counter-target logic in
    /// future slices).
    fn push_to_stack(&mut self, source_obj_id: u32, controller: u8) {
        // Slice 1 of ETB-trigger closure: every Python-side
        // `push_to_stack` call is a spell push (or a spell-shaped
        // activated-ability push). Triggered-ability pushes go
        // through `state_ops_resolve::push_triggered_ability` instead,
        // which constructs a `StackItem::new_trigger`.
        self.stack.push(StackItem::new_spell(source_obj_id, controller));
    }

    /// **Aura-attachment substrate — Phase 1 STEP 3 (round-78,
    /// 2026-05-20).** Read the `target_obj_id` slot of the top
    /// stack item. Returns `Some(id)` if the top item carries a
    /// bound target (the Aura's chosen enchant target after
    /// `submit_enchant_target_decision(Some(id))`, OR the cast-time
    /// bound target for a single-target spell like Lightning Bolt),
    /// `None` if the top item has no bound target OR the stack is
    /// empty.
    ///
    /// **Why a dedicated getter instead of extending
    /// `stack_snapshot()` / `peek_top_of_stack()`.** Those dict
    /// surfaces are pinned by the existing `tests_rust/test_stack.py`
    /// suite (exact `{"source_obj_id", "controller"}` dict shape) —
    /// adding a key would breaking-change every pin. A dedicated
    /// getter is additive (no existing test reads it; new STEP-3
    /// tests opt in by calling it). Same pattern the
    /// `pending_decisions()` dict / `submit_*_decision` substrate
    /// uses: separate read-only and submit surfaces from the
    /// existing snapshot surfaces.
    fn stack_top_target_id(&self) -> Option<u32> {
        self.stack.last().and_then(|item| item.target_obj_id)
    }

    /// **ModalChooseOne envelope — Cycle 1 (cast-time mode binding,
    /// 2026-05-21).** Read the cast-time-bound mode index of the
    /// top stack item. Returns `Some(idx)` if the top item carries a
    /// bound mode pick (a modal spell whose first effect is
    /// `Effect::ChooseOneMode { modes }` cast through
    /// `cast_spell_noncreature`), `None` if the top item has no
    /// mode bind OR the stack is empty.
    ///
    /// **Why a dedicated getter** — parallel to
    /// [`Self::stack_top_target_id`]: the existing
    /// `peek_top_of_stack()` / `stack_snapshot()` dict surfaces are
    /// pinned by `tests_rust/test_stack.py` (the
    /// `{"source_obj_id", "controller"}` 2-key shape); adding a key
    /// would break every pin. A dedicated getter is additive —
    /// existing readers ignore it; the Cycle 1 tests + the
    /// cross-engine `cross_engine.py` modal-spell bridge opt in
    /// explicitly. Mirrors Python's
    /// `state.stack[-1].chosen_modes` access pattern (the bridge
    /// translates Python's tuple to/from this scalar via the
    /// first-element-or-None convention — single-pick Cycle 1).
    fn stack_top_chosen_mode_idx(&self) -> Option<u8> {
        self.stack.last().and_then(|item| item.chosen_mode_idx)
    }

    /// **Close Encounter additional-cost-choice envelope — Cycle 2
    /// (cast-pipeline binding getter, 2026-05-21).** Read-only getter
    /// for the cast-time-bound `additional_cost_choice_id` field of
    /// the top stack item. Returns `Some(id)` if the top item carries
    /// a bound additional-cost choice (a spell whose printing's
    /// `additional_cost_choose_your_creature == true` cast through
    /// `cast_spell_noncreature`), `None` if the top item has no
    /// choice bind OR the stack is empty.
    ///
    /// **Why a dedicated getter** — parallel to
    /// [`Self::stack_top_chosen_mode_idx`] /
    /// [`Self::stack_top_target_id`]: the existing
    /// `peek_top_of_stack()` / `stack_snapshot()` dict surfaces are
    /// pinned by pre-existing tests (the `{"source_obj_id",
    /// "controller"}` 2-key shape); adding a key would break every
    /// pin. A dedicated getter is additive — existing readers ignore
    /// it; the Cycle 2 unit tests opt in explicitly. Mirrors Python's
    /// `state.stack[-1].additional_cost_choice_id` access pattern.
    fn stack_top_additional_cost_choice_id(&self) -> Option<u32> {
        self.stack.last().and_then(|item| item.additional_cost_choice_id)
    }

    /// **ModalChooseOne envelope — scour-for-scrap-slice (Cycle 1
    /// widen, 2026-05-21).** Read the cast-time-bound multi-pick
    /// mode-index subset of the top stack item. Returns `Some(vec)`
    /// if the top item carries a multi-pick bind (an or_both=true
    /// modal spell cast through `cast_spell_noncreature`), `None` if
    /// the top item has no multi-pick bind (every pre-widen modal +
    /// non-modal spell — the single-pick `stack_top_chosen_mode_idx`
    /// sibling reads those) OR the stack is empty.
    ///
    /// Mirrors the `stack_top_chosen_mode_idx` getter's contract /
    /// rationale (additive surface — `peek_top_of_stack` /
    /// `stack_snapshot` dict surfaces are pinned to their existing
    /// shape; a dedicated getter avoids breaking those pins).
    /// Returns a CLONED `Vec<u8>` (typically 0-2 entries) so the
    /// caller owns its read.
    fn stack_top_chosen_mode_idxs(&self) -> Option<Vec<u32>> {
        // Convert Vec<u8> -> Vec<u32> at the wire boundary because
        // PyO3 maps `Vec<u8>` to Python `bytes`, not `list[int]`.
        // The internal storage stays Vec<u8> (matches the StackItem
        // schema); the cross-language read returns a list[int].
        self.stack
            .last()
            .and_then(|item| item.chosen_mode_idxs.as_ref())
            .map(|v| v.iter().map(|x| *x as u32).collect())
    }

    /// **scour-for-scrap-slice Cycle 1 widen (2026-05-21).** Synthetic-
    /// test setter for the top stack item's multi-pick mode slot.
    /// Mirrors the synthetic `push_to_stack_with_target` shape: the
    /// production cast-time bind always picks all modes (the no-agent
    /// default), but synthetic tests need to override the subset to
    /// validate sub-subset / out-of-range / empty-subset resolve
    /// semantics. No-op when the stack is empty.
    ///
    /// **Production safety.** Never called from production code paths
    /// (the cast-time bind sets it once at push time; the resolver
    /// reads it during apply_effect; no re-binding mid-resolve). The
    /// setter exists solely for the substrate test surface.
    fn set_stack_top_chosen_mode_idxs(&mut self, idxs: Vec<u32>) {
        // Accept Vec<u32> at the wire boundary (mirror of the
        // getter's `Vec<u32>`) and downcast each entry to u8 for
        // the storage shape. Out-of-range u8 entries silently
        // saturate at 255 — the resolve arm clamps to
        // modes.len()-1 anyway.
        if let Some(item) = self.stack.last_mut() {
            let internal: Vec<u8> = idxs.iter().map(|x| *x as u8).collect();
            item.chosen_mode_idxs = Some(internal);
        }
    }

    /// **X-value subsystem — Cycle 1 (cast-time X bind getter,
    /// 2026-05-21).** Read-only getter for the cast-time-bound
    /// `x_value` field of the top stack item. Returns `Some(x)` if
    /// the top item carries a bound X (an X-cost spell cast through
    /// `cast_spell_noncreature`'s Cycle 2/3 X-cost detector — NOT
    /// wired yet in Cycle 1), `None` if the top item has no X bind
    /// OR the stack is empty.
    ///
    /// **Why a dedicated getter** — parallel to
    /// [`Self::stack_top_chosen_mode_idx`] /
    /// [`Self::stack_top_additional_cost_choice_id`] /
    /// [`Self::stack_top_target_id`]: the existing
    /// `peek_top_of_stack()` / `stack_snapshot()` dict surfaces are
    /// pinned by pre-existing tests (the `{"source_obj_id",
    /// "controller"}` 2-key shape); adding a key would break every
    /// pin. A dedicated getter is additive — existing readers ignore
    /// it; Cycle 1 substrate tests opt in explicitly. Mirrors
    /// Python's `state.stack[-1].x_value` access pattern.
    ///
    /// **Cycle 1 baseline behavior.** Every pre-substrate push sets
    /// `x_value: None`, so this getter returns `None` for every
    /// current cross-engine trajectory. The first non-None
    /// observation will come from the Cycle 2/3 cast-pipeline mint
    /// site OR from the synthetic-test setter
    /// `set_stack_top_x_value` below.
    fn stack_top_x_value(&self) -> Option<i32> {
        self.stack.last().and_then(|item| item.x_value)
    }

    /// **X-value subsystem — Cycle 1 (synthetic-test setter,
    /// 2026-05-21).** Synthetic-test setter for the top stack item's
    /// `x_value` slot. Mirrors `set_stack_top_chosen_mode_idxs`'s
    /// shape: the production cast-time bind (Cycle 2/3) sets the
    /// value once at push time; this setter exists solely so unit
    /// tests can validate the resolver's save → set → restore
    /// bracket semantics against a known synthetic X. No-op when
    /// the stack is empty.
    ///
    /// **Production safety.** Never called from production code paths
    /// — the cast-time bind sets it once at push time; the resolver
    /// reads it during apply_effect; no re-binding mid-resolve. The
    /// setter exists solely for the substrate test surface.
    ///
    /// Accepts `Option<i32>` directly so tests can also unset the
    /// slot back to `None` (validate the "restore to prior None"
    /// branch of the resolver bracket).
    #[pyo3(signature = (x=None))]
    fn set_stack_top_x_value(&mut self, x: Option<i32>) {
        if let Some(item) = self.stack.last_mut() {
            item.x_value = x;
        }
    }

    /// G7 lander-controller-mismatch closure — synthetic-test stack
    /// push that carries the cast-time-bound target. The production
    /// cast-spell path (`cast_spell_noncreature`) sets target via the
    /// cast-time binding (read at resolve via the
    /// `current_resolve_payload_target_id` swap); the production
    /// triggered-ability path (`push_triggered_ability`) sets target
    /// via `first_legal_trigger_target`. Synthetic tests that bypass
    /// both (push directly + `resolve_top_of_stack` for a Tier-A
    /// targeted spell like Seedship Impact) previously relied on the
    /// arm self-scanning the bf at resolve time — now removed (the
    /// self-scan was the SoT-divergence root cause: when the bound
    /// target ceases between cast and resolve, the arm picked a
    /// stale-pool replacement). This sibling exposes the existing
    /// `new_spell_with_targets` constructor so the synthetic push-
    /// and-resolve idiom can seed the StackItem's target field
    /// directly. `target_obj_id_2` is `None`-defaulted (Phase-1
    /// single-target — Diplomatic Relations is the only 2-slot card
    /// and is cast through `cast_spell_noncreature`).
    fn push_to_stack_with_target(
        &mut self,
        source_obj_id: u32,
        controller: u8,
        target_obj_id: u32,
    ) {
        self.stack.push(StackItem::new_spell_with_targets(
            source_obj_id,
            controller,
            Some(target_obj_id),
            None,
        ));
    }

    /// Return the top stack item as a Python dict
    /// `{"source_obj_id", "controller"}`, or `None` if the stack is
    /// empty. Read-only — does not mutate the stack.
    fn peek_top_of_stack<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.stack.last() {
            None => Ok(None),
            Some(item) => Ok(Some(stack_item_to_dict(py, item)?)),
        }
    }

    /// Pop and return the top stack item as a Python dict, or `None`
    /// if the stack is empty. The dict shape matches
    /// `peek_top_of_stack`.
    ///
    /// This slice does NOT interpret the popped item — resolution
    /// (what the spell/ability does) is a future slice's concern.
    /// Callers that just want to discard the top use this and ignore
    /// the return value; callers that want to resolve will hand the
    /// dict to the resolution layer once it exists.
    fn pop_top_of_stack<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.stack.pop() {
            None => Ok(None),
            Some(item) => Ok(Some(stack_item_to_dict(py, &item)?)),
        }
    }

    /// **Round-16: process the stack greedily until empty.** Repeatedly
    /// calls `resolve_top_of_stack()` (round-6 slice) until
    /// `self.stack` is empty, returning the number of items resolved.
    ///
    /// Returns:
    /// - `0` if the stack is already empty on entry (no-op).
    /// - `>= 1` if at least one item was drained.
    ///
    /// **Composition note**: this composes round-5 push/pop with
    /// round-6 resolve. It does NOT model real MTG priority hand-off
    /// (real MTG: after every resolution both players re-receive
    /// priority); this primitive just drains the stack greedily.
    /// Higher-level code that wants real priority semantics should
    /// call `resolve_top_of_stack` itself in a priority loop.
    ///
    /// **Iteration cap**: 1000-iteration safety bound so a buggy
    /// effect that re-pushes its source can't infinite-loop the
    /// caller. Hitting the bound raises `ValueError` loudly.
    ///
    /// **Cross-file friction (E0624 workaround).**
    /// `resolve_top_of_stack` is a `#[pymethods]` method defined in
    /// `state_ops_resolve.rs`. PyO3's `multiple-pymethods` feature
    /// lets the methods coexist on the same class, but the underlying
    /// Rust fns stay module-private — we can't call
    /// `self.resolve_top_of_stack(py)?` directly from this slice
    /// without touching `state_ops_resolve.rs` (forbidden by the
    /// per-slice file-isolation rule). The fix is to take
    /// `Bound<'_, Self>` rather than `&mut self` and dispatch through
    /// the Python attribute layer with `call_method0`. PyO3's
    /// `Bound<T>` is a borrow-checked handle to a Python object; the
    /// `.borrow()` / `.borrow_mut()` cost is identical to the borrow
    /// the `#[pymethods]` macro would have generated for `&mut self`,
    /// so the pattern adds at most one extra Python-attr-lookup per
    /// loop iteration vs an inlined call. Trade-off: avoid copy-
    /// pasting ~500 lines of `apply_effect` + post-resolve movement +
    /// ETB dispatch into this file.
    fn process_stack_until_empty(slf: Bound<'_, Self>) -> PyResult<usize> {
        let mut count: usize = 0;
        let max_iterations: usize = 1000;
        loop {
            // Read stack length without holding a long-lived borrow:
            // `borrow()` returns a `PyRef` that drops at the end of
            // the statement.
            let is_empty = slf.borrow().stack.is_empty();
            if is_empty {
                return Ok(count);
            }
            if count >= max_iterations {
                return Err(PyValueError::new_err(format!(
                    "process_stack_until_empty exceeded {max_iterations} iterations \
                     (likely a buggy effect re-pushing its source)"
                )));
            }
            // Dispatch through the Python attribute layer — equivalent
            // to `self.resolve_top_of_stack()` but crosses the file
            // boundary cleanly (no E0624). Discards the returned dict;
            // we only care about the side-effects on `self.stack`.
            slf.call_method0("resolve_top_of_stack")?;
            count += 1;
        }
    }

    /// Read-only snapshot of the entire stack as a list of dicts, one
    /// per `StackItem`, in bottom-to-top order (Vec iteration order =
    /// stack[0] is bottom, stack[-1] is top — matches how
    /// `peek_top_of_stack` returns `stack.last()`).
    ///
    /// Each dict carries the same two keys as `peek_top_of_stack` /
    /// `pop_top_of_stack` — `source_obj_id` and `controller`. The
    /// returned dicts are freshly allocated; mutating them does NOT
    /// affect the underlying Rust `StackItem` storage (this is the
    /// round-21 bridge pattern: `state.stack_snapshot()` parallels
    /// `state.hand_snapshot()` et al., and the parity harness's
    /// `_zone_signature` walks the returned list dict-by-dict).
    ///
    /// Not a `#[getter]` — pymethods returning `Vec<Bound<PyDict>>`
    /// compose cleanly here and `state.stack_snapshot()` matches the
    /// established round-21 bridge naming.
    fn stack_snapshot<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.stack
            .iter()
            .map(|item| stack_item_to_dict(py, item))
            .collect()
    }

    /// Remove the FIRST stack item with the given `source_obj_id`.
    /// Returns `true` if an item was removed, `false` otherwise (no
    /// matching item -> no mutation).
    ///
    /// "First" walks bottom-to-top (the Vec's natural iteration
    /// order). In practice each `source_obj_id` appears at most once
    /// per spell-cast event, so the bottom-vs-top distinction
    /// rarely matters; if it ever does (a triggered ability's
    /// source object also has a copy of itself on the stack via a
    /// later effect), a future slice can add a top-down variant.
    ///
    /// Used by counter-spell-shaped effects: removing the targeted
    /// stack item is the surface, *enforcing* counter rules (legality,
    /// "can't be countered", replacement effects) is left to the
    /// caller.
    fn remove_from_stack(&mut self, source_obj_id: u32) -> bool {
        if let Some(pos) = self
            .stack
            .iter()
            .position(|item| item.source_obj_id == source_obj_id)
        {
            self.stack.remove(pos);
            true
        } else {
            false
        }
    }
}

/// Build a 2-key dict from a StackItem. Helper kept private to this
/// module — callers outside should go through the pymethods above
/// rather than rebuilding the dict shape ad hoc.
fn stack_item_to_dict<'py>(
    py: Python<'py>,
    item: &StackItem,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source_obj_id", item.source_obj_id)?;
    d.set_item("controller", item.controller)?;
    Ok(d)
}
