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
        self.stack.push(StackItem {
            source_obj_id,
            controller,
        });
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
