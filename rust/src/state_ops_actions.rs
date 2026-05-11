//! `apply_action(action_dict)` pymethod on `GameState`.
//!
//! Phase 3 dispatcher — `docs/RUST_PARITY_PLAN.md`. Parses an `Action`
//! dict via `actions::parse_action`, matches on the closed enum, and
//! routes to the right primitive (e.g. `Action::PassPriority` →
//! existing `pass_priority()` from the round-4 priority slice;
//! `Action::PlayLand` → `play_land()` from round 4's cast slice).
//!
//! ## Slice scope
//!
//! Round 22 wired `PassPriority`; round 23 added `PlayLand` (10 of 420
//! decisions / ~2.4% in round-20's sample game — the 4th-most-frequent
//! variant), `DeclareAttackers` (16 of 420 / ~3.8% — the
//! 3rd-most-frequent variant), and `DeclareBlockers` (5 of 420 /
//! ~1.2% — the 4th-most-frequent non-pass variant). Future slices
//! grow the match arm (CastSpell, Activate, ...) plus a
//! `legal_decisions` enumerator. The post-round-25
//! `pass_priority-step-advance` slice closes G3 at this layer by
//! adding a `state.consecutive_passes: u8` counter on `GameState`
//! and wiring CR 117.4 in this dispatcher — empty-stack/both-passed
//! fires `advance_phase()` from the PassPriority arm. The bare
//! `pass_priority()` primitive in `state_ops_priority.rs` stays a
//! pure toggle; G3's stack-resolution leg is still deferred (callers
//! reach for `complete_priority_round()` explicitly).
//!
//! ## Actor convention for hand-targeting actions
//!
//! The Python `PlayLand(card_id: int)` dataclass carries only the
//! card id — the actor is implicit. Python's `apply_action` takes an
//! explicit `player_id` parameter; the Rust dispatcher derives the
//! same value from `state.priority_player` (the player currently
//! making decisions). The `play_land` primitive itself takes
//! `(obj_id, player_id)` and only succeeds if `obj_id` is in that
//! player's hand, so misrouted dispatch fails cleanly rather than
//! silently mutating the wrong side.
//!
//! ## Actor convention for combat actions
//!
//! `DeclareAttackers` is a combat-step action — it fires on the
//! active player's turn, not at an arbitrary priority moment. The
//! `declare_attacker` primitive (round 5) is `(&mut self, obj_id: u32)`:
//! it only takes the creature id and reads the owning player from the
//! battlefield index, so the dispatcher doesn't need to pass a
//! player_id parameter. The active-vs-priority distinction is moot
//! here for declaring attackers (only the active player can declare
//! attackers; priority is at the start of the declare-attackers step
//! when this fires), but the dispatch loop still iterates with the
//! Vec's stable order so callers can pin a deterministic application
//! order via sorting the set before passing.
//!
//! `DeclareBlockers` is symmetric: the round-5
//! `declare_blocker(blocker_id, attacker_id)` primitive takes only the
//! two object ids and reads the owning player from the battlefield
//! index, so the dispatcher doesn't pass a player_id either. The
//! dispatch loop iterates the `Vec<(u32, u32)>` of pairs in input
//! order — matters for damage assignment when multiple blockers
//! block the same attacker (the first declared blocker eats the
//! attacker's full damage under the round-5 single-blocker damage
//! model).
//!
//! ## Cross-file dispatch pattern (round-16)
//!
//! `pass_priority()` lives in `state_ops_priority.rs` as a
//! `#[pymethods]` method on `GameState`. PyO3's `multiple-pymethods`
//! feature lets the methods coexist on the same class, but the
//! underlying Rust fn stays module-private — we can't call
//! `self.pass_priority()` directly from this slice without touching
//! `state_ops_priority.rs` (forbidden by the per-slice file-isolation
//! rule). The fix is to take `Bound<'_, Self>` rather than `&mut self`
//! and dispatch through the Python attribute layer with
//! `call_method0`. PyO3's `Bound<T>` is a borrow-checked handle to a
//! Python object; the `.borrow()` / `.borrow_mut()` cost is identical
//! to the borrow the `#[pymethods]` macro would have generated for
//! `&mut self`, so the pattern adds at most one extra Python-attr
//! lookup per call. See `state_ops_stack.rs:process_stack_until_empty`
//! for the original use of this pattern.

use crate::actions::{parse_action, Action};
use crate::state::GameState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pymethods]
impl GameState {
    /// Apply a Python-side Action dict to this state. The dict's
    /// `kind` selects the arm; each arm routes to the matching
    /// primitive on `GameState`.
    ///
    /// Returns `None` — this is a side-effecting mutator (Python
    /// convention for void methods).
    ///
    /// Arms:
    /// - `{"kind": "pass_priority"}` → calls `pass_priority()`
    ///   (round 22) AND composes CR 117.4 step-advance
    ///   (pass-priority-step-advance slice). When both players
    ///   pass in succession with an empty stack, fires
    ///   `advance_phase()` after the toggle. See the arm's inline
    ///   comment for the `consecutive_passes` counter mechanics
    ///   and why the bare primitive stays a pure toggle.
    /// - `{"kind": "play_land", "card_id": <u32>}` → calls
    ///   `play_land(card_id, priority_player)` (round 23).
    /// - `{"kind": "declare_attackers", "attacker_ids": [<u32>, ...]}`
    ///   → iterates the Vec and calls
    ///   `declare_attacker(attacker_id)` once per id (round 23).
    ///   Ineligible ids are silently rejected by the primitive's
    ///   `bool` return — mirrors Python's `_declare_attackers`
    ///   filtering behavior.
    /// - `{"kind": "declare_blockers", "assignments":
    ///   [(<u32>, <u32>), ...]}` → iterates the Vec of pairs and calls
    ///   `declare_blocker(blocker_id, attacker_id)` once per pair in
    ///   input order (round 23). Ineligible pairs are silently
    ///   rejected by the primitive's `bool` return.
    /// - `{"kind": "activate_ability", "source_id": <u32>,
    ///   "ability_index": <u32>}` → mana-ability dispatch (round 25).
    ///   For `ability_index = 0` on an untapped battlefield permanent
    ///   whose printing has `mana_produced.is_some()`, taps the
    ///   source and credits the controller's mana pool with the
    ///   pips listed in `mana_produced` (one per element, saturating
    ///   at u8::MAX per color). Mirrors the legal_decisions emission
    ///   gate (round 24, `state_ops_legal.rs`) which only emits this
    ///   shape for the same predicate. Raises `ValueError` if
    ///   `ability_index != 0`, the source isn't on any battlefield,
    ///   the source is already tapped, or the printing has no
    ///   `mana_produced` (non-mana activated abilities are a
    ///   future-slice scope — they need richer cost-pay + effect-
    ///   resolution primitives).
    /// - `{"kind": "cast_spell", "card_id": <u32>, "use_warp":
    ///   <bool>?, "kicked": <bool>?, "cast_for_free": <bool>?,
    ///   "use_impending": <bool>?, "cast_room_half": <str|None>?}`
    ///   → calls `cast_creature(card_id, priority_player)`. The 5
    ///   alt-cost / mode flags are parsed as data and ROUND-TRIP
    ///   through the bridge but are IGNORED by this dispatch arm
    ///   (first-slice scope, mirrors the ActivateAbility "scaffold
    ///   the variant, defer the dispatch semantics" pattern — see
    ///   the `Action::CastSpell` arm's inline comment for the
    ///   future-slice plumbing of warp / kicker / impending /
    ///   cast_for_free / cast_room_half).
    ///
    /// Unknown `kind` raises `ValueError`; missing `kind` or other
    /// required keys (e.g. `card_id` for `play_land`,
    /// `attacker_ids` for `declare_attackers`, `assignments` for
    /// `declare_blockers`, `source_id` / `ability_index` for
    /// `activate_ability`) raise `KeyError`. Both bubble up from
    /// `actions::parse_action`.
    fn apply_action(slf: Bound<'_, Self>, action_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let action = parse_action(action_dict)?;
        // CR 117.4 composition (RUST_PARITY_GAPS.md G3): non-pass
        // actions break any in-flight "consecutive passes" chain.
        // Reset the counter up-front so e.g. PassPriority -> PlayLand
        // -> PassPriority no longer counts the first pass toward
        // round-completion. The PassPriority arm restores the
        // increment itself. This keeps the bare `pass_priority()`
        // primitive in `state_ops_priority.rs` unchanged — the round
        // tracking lives entirely at the apply_action layer.
        if !matches!(action, Action::PassPriority) {
            slf.borrow_mut().consecutive_passes = 0;
        }
        match action {
            Action::PassPriority => {
                // Cross-file pymethod dispatch (round-16 pattern):
                // `pass_priority` lives in state_ops_priority.rs and
                // is private to that module's `impl GameState` block.
                // Go through the Python attribute layer to call it.
                //
                // ## CR 117.4 composition (RUST_PARITY_GAPS.md G3)
                //
                // The bare primitive is a pure toggle on
                // `priority_player`. At the action layer we wrap it
                // with the CR 117.4 step-advance rule: when both
                // players pass priority in succession with an empty
                // stack, the game advances to the next step. Python's
                // `apply_action(PassPriority())` is conceptually a
                // no-op (its outer search loop drives the step
                // machine separately) — for cross-engine parity to
                // make progress on real decks we need ONE side to
                // model the rule, and the cleanest choice is here.
                //
                // Implementation:
                // 1. Call the bare toggle (sibling-file private
                //    method, so go through the Python attribute
                //    layer — same pattern as the other arms below).
                // 2. Increment `consecutive_passes` (saturating; in
                //    practice it never exceeds 2 because step 3
                //    resets it the moment it hits 2).
                // 3. If the counter reached 2 AND the stack is
                //    empty, fire `advance_phase()` and zero the
                //    counter. (Non-empty stack: G3 will close the
                //    stack-resolution branch in a future slice via
                //    the existing `complete_priority_round()`
                //    primitive; this slice keeps the scope to
                //    empty-stack step-advance, which is what
                //    unblocks the parity_smoke real-deck stress
                //    measurement getting past UNTAP.)
                //
                // The borrow split (drop the `&mut` borrow before
                // re-entering Python via `call_method0`) mirrors the
                // PlayLand arm's pattern below — `call_method0` may
                // re-borrow under the hood and a held `RefMut` would
                // collide.
                slf.call_method0("pass_priority")?;
                let should_advance = {
                    let mut state = slf.borrow_mut();
                    state.consecutive_passes = state.consecutive_passes.saturating_add(1);
                    let go = state.consecutive_passes >= 2 && state.stack.is_empty();
                    if go {
                        state.consecutive_passes = 0;
                    }
                    go
                };
                if should_advance {
                    // `advance_phase` is the cross-slice composition
                    // hook: it also fires `untap_all_for_player` on
                    // turn rollover (state_ops_steps.rs) so a chain
                    // of pass_priority calls walks the full phase
                    // ladder including the UNTAP entry sweep — which
                    // is the parity-smoke unblocker: real decks now
                    // reach MAIN1 via legal_decisions emitting
                    // PassPriority pairs at each pre-main step.
                    slf.call_method0("advance_phase")?;
                }
            }
            Action::PlayLand { card_id } => {
                // Same cross-file pattern as `pass_priority`, but
                // `play_land(obj_id, player_id)` takes args, so
                // use `call_method1`. The actor is the priority
                // player — Python's `PlayLand` dataclass carries
                // only `card_id` and the dispatcher resolves the
                // implicit actor here. We borrow `slf` for the
                // single u8 field read, then drop the borrow before
                // re-entering Python via `call_method1` (which
                // would re-borrow under the hood).
                let player_id = slf.borrow().priority_player;
                slf.call_method1("play_land", (card_id, player_id))?;
            }
            Action::DeclareAttackers { attacker_ids } => {
                // Same cross-file pattern as `play_land`: iterate the
                // Vec and dispatch one `declare_attacker(obj_id)` call
                // per id. The primitive takes only the creature id
                // (not a player_id) — it reads the owning player from
                // the battlefield index, so the dispatcher doesn't
                // need to pre-resolve an actor here. Ineligible ids
                // (tapped, non-creature, summoning-sick without haste,
                // defender, etc.) are silently rejected by the
                // primitive's `bool` return; we discard the bool to
                // match Python's `_declare_attackers` filtering
                // behavior (the eligible-set filter on the Python side
                // is implicit here via the primitive's per-id gate).
                //
                // Iteration order is the Vec's order — stable from
                // parse_action's `extract::<Vec<u32>>()`, which
                // preserves the Python iterable's iteration order.
                // Callers that want a canonical order sort the set
                // before passing (the parity harness does this).
                for attacker_id in attacker_ids {
                    slf.call_method1("declare_attacker", (attacker_id,))?;
                }
            }
            Action::DeclareBlockers { assignments } => {
                // Same cross-file pattern: iterate the `Vec<(u32, u32)>`
                // of `(blocker_id, attacker_id)` pairs and dispatch one
                // `declare_blocker(blocker_id, attacker_id)` per pair.
                // The primitive takes both ids (not a player_id) — it
                // reads the owning player from the battlefield index,
                // so the dispatcher doesn't pre-resolve an actor here.
                // Ineligible pairs (tapped blocker, non-creature
                // blocker, attacker not in `state.attackers`, flying/
                // skulk gate failure, unknown ids) are silently
                // rejected by the primitive's `bool` return; we
                // discard the bool to match the silent-skip behavior
                // of the `DeclareAttackers` arm above.
                //
                // Iteration order is the Vec's order — stable from
                // `parse_action`'s `try_iter` + per-element extract,
                // which preserves the Python iterable's iteration
                // order. Order matters: when multiple blockers block
                // the same attacker, the round-5 combat-damage model
                // assigns the attacker's power to the first declared
                // blocker for that attacker.
                for (blocker_id, attacker_id) in assignments {
                    slf.call_method1("declare_blocker", (blocker_id, attacker_id))?;
                }
            }
            Action::CastSpell {
                card_id,
                use_warp: _,
                kicked: _,
                cast_for_free: _,
                use_impending: _,
                cast_room_half: _,
            } => {
                // Round-23(+) slice: ship the `CastSpell` variant +
                // bridge with all 5 alt-cost / mode flags as data;
                // dispatch IGNORES the flags and routes to the
                // existing round-4 `cast_creature(obj_id, player_id)`
                // primitive. Same pattern as `play_land` for the
                // actor resolution (the Python `CastSpell` dataclass
                // carries only `card_id`; the implicit actor is the
                // priority player). Borrow `slf` for the u8 read,
                // then drop the borrow before re-entering Python via
                // `call_method1` (which would re-borrow under the
                // hood).
                //
                // **Flags are silently ignored for first-slice.**
                // Mirrors the round-23 ActivateAbility "scaffold the
                // variant, defer the dispatch semantics" pattern —
                // search-loop dispatch can already drive Rust state
                // through the production `CastSpell` dict shape
                // (warp / kicker / impending / Room-half / free-cast
                // payloads round-trip losslessly through the
                // bridge), and a future slice wires each flag
                // through to a richer primitive:
                //   - use_warp -> alt-cost from exile (needs
                //     Effect::Warp + the cost-from-zone primitive).
                //   - kicked -> additional mana cost stacked on the
                //     base (needs cost-stacking infrastructure).
                //   - cast_for_free -> bypass the mana payment in
                //     `cast_creature`'s phase 2.
                //   - use_impending -> defer creature-ness via time
                //     counters + the new face-down-ish state field.
                //   - cast_room_half -> route to the picked Room
                //     half's mana cost + unlock effect.
                // No runtime warning when a non-default flag is set
                // — the test suite covers the "flags pass through
                // silently" contract (search-loop callers shouldn't
                // see ergonomic noise from the deferred-semantics
                // gap; the gap is documented here + in the variant
                // docstring + in the porting-status table).
                let player_id = slf.borrow().priority_player;
                slf.call_method1("cast_creature", (card_id, player_id))?;
            }
            Action::TurnFaceUp { card_id: _ } => {
                // First-slice scope (mirrors round-23's ActivateAbility
                // pattern): ship the Action variant + bridge round-trip
                // scaffold; the dispatch arm is a stub because no Rust
                // face-down GameObject state exists yet. The
                // `GameObject` struct has no `face_down` flag today, and
                // there's no Spacecraft charge-counter resolution
                // primitive — there's nothing for a `turn_face_up`
                // dispatch to flip. A future slice adds the per-object
                // face-down state field + flip primitive (analogous to
                // how `tap` / `untap` toggle the `tapped` flag) and then
                // flips this arm to a real `call_method1(
                // "turn_face_up", (card_id, priority_player))` dispatch
                // — at which point the variant + bridge already shipped
                // here keep working unchanged. The variant ships now so
                // callers building Action dicts can already use the
                // `turn_face_up` shape and hit a clear "not implemented
                // yet" error path instead of `unknown action kind`,
                // and the parity harness surfaces the gap loudly when
                // Python emits a TurnFaceUp decision (CR 701.34 /
                // 707.4 — EoE Spacecraft station+launch is the
                // production driver).
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "TurnFaceUp dispatch not implemented yet (Rust face-down \
                     GameObject state + Spacecraft primitive missing; future \
                     slice flips the dispatch). This slice ships only the \
                     Action variant + bridge round-trip — apply_action raises \
                     to surface the gap loudly to the parity harness."
                ));
            }
            Action::UnlockRoomHalf {
                card_id: _,
                half_name: _,
            } => {
                // First-slice scope (mirrors round-23's ActivateAbility
                // and round-24's TurnFaceUp pattern): ship the Action
                // variant + bridge round-trip scaffold; the dispatch arm
                // is a stub because no Rust Room-half state exists yet.
                // Python's `GameObject` carries a per-Room
                // `unlocked_halves: set[str]` field that tracks which
                // halves of a Room enchantment have been unlocked — the
                // Rust `GameObject` struct has no such field today, and
                // there's no half-specific effect activation primitive.
                // A future slice adds the per-object Room state +
                // `unlock_room_half(card_id, half_name)` primitive
                // (analogous to how Python's `_unlock_room_half`
                // mutates the set, pays the half's mana cost, and fires
                // the half-specific ENTERS triggers) and then flips
                // this arm to a real `call_method1("unlock_room_half",
                // (card_id, half_name, priority_player))` dispatch —
                // at which point the variant + bridge already shipped
                // here keep working unchanged. The variant ships now so
                // callers building Action dicts can already use the
                // `unlock_room_half` shape and hit a clear "not
                // implemented yet" error path instead of `unknown
                // action kind`, and the parity harness surfaces the
                // gap loudly when Python emits an UnlockRoomHalf
                // decision (CR 715.3-4 — DSK Rooms mechanic is the
                // production driver).
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "UnlockRoomHalf dispatch not implemented yet (Rust Room \
                     enchantment state + half-name resolution missing; future \
                     slice flips the dispatch). This slice ships only the \
                     Action variant + bridge round-trip."
                ));
            }
            Action::ActivateStation {
                spacecraft_id: _,
                crew_id: _,
            } => {
                // First-slice scope (mirrors round-23's ActivateAbility
                // and round-24's TurnFaceUp / UnlockRoomHalf pattern):
                // ship the Action variant + bridge round-trip scaffold;
                // the dispatch arm is a stub because no Rust Spacecraft
                // charge-counter state / crew-tap resolution primitive
                // exists yet. Python's `_activate_station` taps the crew
                // creature (cost), reads its power, and adds that many
                // charge counters to the Spacecraft (effect) — Rust has
                // neither the Spacecraft state field for charge counters
                // nor a primitive that links tap-as-cost to counter-add-
                // as-effect. A future slice adds the per-object
                // Spacecraft charge state + the activate-station
                // primitive (analogous to how Python's
                // `_activate_station` mutates both sides atomically) and
                // then flips this arm to a real `call_method1(
                // "activate_station", (spacecraft_id, crew_id,
                // priority_player))` dispatch — at which point the
                // variant + bridge already shipped here keep working
                // unchanged. The variant ships now so callers building
                // Action dicts can already use the `activate_station`
                // shape and hit a clear "not implemented yet" error
                // path instead of `unknown action kind`, and the parity
                // harness surfaces the gap loudly when Python emits an
                // ActivateStation decision (EoE Spacecraft station
                // mechanic — the EoE Spacecraft cycle is the production
                // driver). **This completes the Action enum scaffold:
                // all 8 Python Action subclasses now have Rust
                // variants.**
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ActivateStation dispatch not implemented yet (Rust \
                     Spacecraft charge-counter state + crew-tap resolution \
                     missing; future slice flips the dispatch). This slice \
                     ships only the Action variant + bridge round-trip."
                ));
            }
            Action::ActivateAbility {
                source_id,
                ability_index,
            } => {
                // Round-25 slice: flip the round-23 stub to a real
                // dispatch for **mana abilities only**. The matching
                // emission in `legal_decisions` (round-24,
                // `state_ops_legal.rs`) only emits ActivateAbility
                // for untapped permanents whose printing's
                // `mana_produced.is_some()`, with `ability_index = 0`.
                // We mirror that filter exactly: only `ability_index
                // = 0` + `mana_produced.is_some()` is supported in
                // this slice; anything else raises (loud failure on
                // out-of-scope ActivateAbility caller input —
                // honors the "silent no-op pipeline runs are
                // pipeline bugs" guidance: a caller that built an
                // ActivateAbility dict for a non-mana ability should
                // know immediately that the dispatcher can't honor
                // it yet rather than getting a misleading-success
                // no-mutation return).
                //
                // **Why inline rather than `call_method1` cross-file
                // dispatch.** The mana-ability flow is small —
                // locate the source on a battlefield, validate the
                // printing has `mana_produced`, validate the source
                // is untapped, then tap it + credit the controller's
                // mana pool. There's no existing
                // `activate_ability(source_id, ability_index)`
                // primitive on `state_ops_mana.rs` to call (the
                // existing `tap_for_mana(obj_id)` doesn't take an
                // ability_index, doesn't validate the input source
                // belongs to the priority player vs the opponent —
                // legal_decisions' emission gate handles that
                // upstream — and returns `Option<String>` silently
                // on every failure mode, which would mask the
                // ability_index!=0 / not-on-battlefield / tapped /
                // non-mana-source error paths this dispatch needs
                // to surface). Inlining the logic here keeps every
                // error path's message specific and the slice
                // self-contained — no sibling-file edits required.
                //
                // **Future slice.** Non-mana activated abilities
                // (e.g. tap a creature for an effect, mana-cost-pay
                // for a pump ability, loyalty abilities on
                // planeswalkers) need richer primitives — cost-pay
                // composition + effect-resolution composition. That
                // belongs in a follow-up slice when the production
                // workload (parity_smoke trajectory mode emitting
                // legal non-mana ActivateAbility decisions) demands
                // it; until then the `ability_index != 0` raise
                // here surfaces the gap loudly to the harness.
                let mut binding = slf.borrow_mut();
                let state: &mut GameState = &mut *binding;
                if ability_index != 0 {
                    return Err(PyValueError::new_err(format!(
                        "ActivateAbility dispatch: only ability_index=0 \
                         (mana ability) is supported in this slice; got \
                         {ability_index}. Non-mana activated abilities \
                         require richer primitives (future slice)."
                    )));
                }
                // Phase 1: locate the source on a battlefield (either
                // player's — legal_decisions only emits for the
                // priority player, but the dispatcher is permissive
                // here so a direct caller building the dict by hand
                // gets the same error message regardless of which
                // side controls the source; the unique invariant we
                // enforce is "on SOME battlefield, untapped,
                // mana_produced"). Read the printing's
                // `mana_produced` Vec up-front so the mutation phase
                // doesn't need to re-borrow the registry.
                let mut found: Option<(usize, usize, Vec<String>)> = None;
                // (controller_idx, bf_pos, mana_symbols)
                for pidx in 0..state.players.len() {
                    if let Some(pos) = state.players[pidx]
                        .battlefield
                        .iter()
                        .position(|o| o.id == source_id)
                    {
                        let obj = &state.players[pidx].battlefield[pos];
                        let printing =
                            state.registry.lookup(obj.printing_id).ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "ActivateAbility: source {source_id} \
                                     printing not in registry"
                                ))
                            })?;
                        let mana = printing.mana_produced.clone().ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "ActivateAbility: source {source_id} has no \
                                 mana_produced (only mana abilities supported \
                                 in first slice)"
                            ))
                        })?;
                        if obj.tapped {
                            return Err(PyValueError::new_err(format!(
                                "ActivateAbility: source {source_id} is \
                                 already tapped"
                            )));
                        }
                        found = Some((pidx, pos, mana));
                        break;
                    }
                }
                let (controller_idx, bf_pos, mana_symbols) = match found {
                    Some(t) => t,
                    None => {
                        return Err(PyValueError::new_err(format!(
                            "ActivateAbility: source {source_id} not on any \
                             battlefield"
                        )))
                    }
                };
                // Phase 2: commit. Tap the source via Arc::make_mut on
                // the controller's battlefield zone (preserves the
                // per-zone CoW invariant — clones of the state that
                // observed the zone before the tap remain
                // untouched), then credit the controller's mana pool.
                // Saturating u8 add per the existing AddManaToPool /
                // tap_for_mana convention in state_ops_mana.rs (255
                // pips of any one color is more than any real game
                // produces; silent wrap would be a worse failure
                // mode than capping).
                let bf_mut = Arc::make_mut(&mut state.players[controller_idx].battlefield);
                bf_mut[bf_pos].tapped = true;
                let pool = &mut state.players[controller_idx].mana_pool;
                for sym in &mana_symbols {
                    match sym.as_str() {
                        "W" => pool.white = pool.white.saturating_add(1),
                        "U" => pool.blue = pool.blue.saturating_add(1),
                        "B" => pool.black = pool.black.saturating_add(1),
                        "R" => pool.red = pool.red.saturating_add(1),
                        "G" => pool.green = pool.green.saturating_add(1),
                        "C" => pool.colorless = pool.colorless.saturating_add(1),
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "ActivateAbility: unknown mana symbol \
                                 {other:?} in mana_produced (valid: W, U, B, \
                                 R, G, C)"
                            )))
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
