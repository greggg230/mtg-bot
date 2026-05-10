//! Stack-resolution primitives on `GameState`.
//!
//! First-cut composition slice: pops the top stack item, finds its
//! source GameObject (search both players' four tracked zones),
//! reads the source printing's effects via `self.registry`, and
//! dispatches each effect against the closed `Effect` enum the
//! round-3 bridge slice landed.
//!
//! Composes:
//! - round-5 stack slice (`push/pop/peek_top_of_stack`,
//!   `state.stack: Vec<StackItem>`).
//! - round-3 bridge-ext (`CardPrinting.effects: Vec<Effect>` with
//!   the `DealDamage` variant).
//! - round-3 damage slice (`deal_damage` shape — though resolve
//!   here goes straight to `players[idx].life` for AnyPlayer and
//!   to `obj.damage += amount` via `Arc::make_mut` for AnyCreature
//!   to avoid coupling slice files; once the parallel-slice
//!   constraint relaxes a follow-up consolidates onto
//!   `state_ops_damage::deal_damage`).
//!
//! ## Slice scope
//!
//! - **Supported effects: `DealDamage`, `DrawCards`, `GainLife`.**
//!   Round-3 landed `DealDamage`; round-8 foundation extended the
//!   closed enum with `DrawCards { count }` and `GainLife { amount }`,
//!   and this slice wires both to dispatch. The dispatcher is a
//!   `match` on the closed enum so adding a new variant in the
//!   bridge is a build-time error here until this file dispatches
//!   it — the desired "fail loud on unknown" property comes for
//!   free from Rust exhaustiveness.
//!   - **DrawCards: controller = active_player.** Same first-slice
//!     convention as DealDamage. The `StackItem.controller` field
//!     exists already but isn't read here — future slice plumbs it
//!     through. Empty library short-circuits silently (deck-out
//!     integration via SBA is deferred).
//!   - **GainLife: controller = active_player.** Direct field write
//!     so a negative compile-time amount silently subtracts rather
//!     than firing the round-1 `gain_life` raise-on-negative guard.
//! - **Targeting is deterministic + simplified.**
//!   - `TargetFilter::AnyPlayer` -> the defending player
//!     (`1 - active_player`). No "choose a target" surface yet.
//!   - `TargetFilter::AnyCreature` -> the FIRST object on the
//!     defending player's battlefield (smallest battlefield-Vec
//!     index). No creature-type filtering, no protection /
//!     hexproof / shroud, no per-spell target-selection. If the
//!     defender's battlefield is empty, the spell fizzles —
//!     stack item still pops, but no damage is applied.
//!   Real targeting is a future slice once the cast-time target-
//!   choice surface lands.
//! - **Post-resolve spell-source movement** (round-7 slice).
//!   Real MTG CR 608.2m: when a spell finishes resolving, the
//!   spell-as-game-object moves out of the stack zone. An instant
//!   or sorcery goes to its owner's graveyard; a permanent spell
//!   becomes a permanent on the battlefield. After applying
//!   effects, `resolve_top_of_stack` looks up the source's
//!   `card_types` and dispatches:
//!   - "Instant" / "Sorcery" -> controller's graveyard
//!   - "Creature" / "Artifact" / "Enchantment" / "Planeswalker" /
//!     "Battle" -> controller's battlefield
//!   - empty `card_types` (synthetic-test escape hatch) -> leave
//!     the source where it is. Real MTG cards always have at
//!     least one card type; this branch only fires for
//!     first-slice synthetic cards.
//!   Creatures specifically: when entering the battlefield via
//!   this path, `summoning_sick` is stamped `true` (CR 302.1 —
//!   the new permanent has summoning sickness).
//! - **ETB-triggered effects** (round-9 slice). After post-resolve
//!   movement places a permanent on the controller's battlefield,
//!   the source printing's `etb_effects: Vec<Effect>` are dispatched
//!   one-by-one through the same `apply_effect` path the spell-
//!   effects use. First-slice simplifications (mirrored in
//!   `tests_rust/test_resolve.py`):
//!   - **Apply immediately, no real stack triggering.** Real MTG:
//!     ETB triggers go onto the stack and resolve in priority
//!     order, can be responded to with instant-speed effects, and
//!     (for "deal damage" ETBs) carry their own targets chosen
//!     on-trigger. First slice applies them inline at resolution
//!     time, same convention as the spell-effects path.
//!   - **Gating: only fire when the source landed on battlefield.**
//!     If post-resolve sent the source to graveyard (instant /
//!     sorcery) or left it in place (empty card_types escape
//!     hatch), `etb_effects` are skipped — by definition the
//!     trigger condition "this enters the battlefield" wasn't met.
//!     A schema-allowed but semantically-nonsense Instant carrying
//!     an etb_effects entry will NOT fire it; only its `effects`
//!     do (the spell's effect).
//!   - **Controller of ETB effects = `state.active_player`.** Same
//!     first-slice convention as DealDamage / DrawCards / GainLife.
//!     Real MTG plumbs the trigger's controller through the trigger
//!     ability; deferred along with the rest of the
//!     `StackItem.controller` plumbing.
//!   - **No replacement effects** ("ETB tapped", "ETB with N
//!     counters") — those are CR 614.1c replacement effects, a
//!     different schema. Not in scope.
//!   - **No "leaves the battlefield" triggers (LtB)** — different
//!     schema, different timing. Future slice.
//! - **Empty stack returns None gracefully** (no exception). Same
//!   convention as `pop_top_of_stack`.
//! - **Round-14 Sacrifice** (this slice). CR 701.16 — sacrifice a
//!   target permanent (it goes to its owner's graveyard).
//!   - **AnyCreature target -> first object on the ACTIVE player's
//!     battlefield (smallest index).** Asymmetric vs the bounce/exile/
//!     tap convention: in real MTG you sacrifice your OWN permanents,
//!     not the defender's. First-slice simplification picks
//!     ``state.active_player``'s battlefield[0] (mirroring the source-
//!     spell controller convention). No creature-type filter. Empty
//!     active-battlefield fizzles silently.
//!   - **AnyPlayer target** -> raises ValueError (cannot sacrifice a
//!     player), matching the bounce/exile/tap/untap AnyPlayer shape.
//!   - **Per-object state reset on zone change** (CR 400.7) — tapped
//!     / summoning_sick / damage / counters reset on the way out, same
//!     as ReturnToHand / ExileTarget.
//!   - Inlines ``Arc::make_mut`` pop-from-battlefield + push-to-
//!     graveyard per the recurring private-helper-cross-file pattern.
//! - **Round-13 UntapTarget**. Symmetric counterpart
//!   to round-12's TapTarget — same target selection convention,
//!   opposite tap state mutation. AnyCreature target -> first object
//!   on the defender's battlefield gets ``tapped = false``. Empty
//!   defender battlefield fizzles silently. AnyPlayer raises
//!   ValueError per the bounce/exile/tap convention. Already-
//!   untapped target is a silent no-op (idempotent). Inlines
//!   ``Arc::make_mut`` + ``tapped = false`` per the recurring
//!   private-helper-cross-file pattern.
//! - **Round-11 ExileTarget**. When a permanent is
//!   exiled, the dispatch pops it from its current zone (battlefield
//!   under the AnyCreature targeting convention) and pushes onto its
//!   owner's exile zone (CR 406). Per-object state — `tapped`,
//!   `summoning_sick`, `damage`, `counters` — resets on the way out
//!   per CR 400.7 (a card that changes zones is treated as a brand-
//!   new object). Mirrors the ReturnToHand reset convention.
//!   - **AnyCreature target** -> first object on the defender's
//!     battlefield (smallest index). Empty defender battlefield
//!     fizzles silently.
//!   - **AnyPlayer target** -> raises ValueError loudly (cannot
//!     exile a player), same shape as ReturnToHand-with-AnyPlayer.
//!   - **ZONE_EXILE = 5**: the constant was already declared in
//!     `state_ops_zones.rs` (alongside `ZONE_STACK = 4`) before this
//!     slice landed; we use it rather than picking a new integer.
//!   - **Read helpers in this slice** (`exile_sizes`,
//!     `exile_contains`, `exile_object_*`): pre-r11 read methods
//!     (`find_object_zone`, `get_damage`, etc.) only walk the four
//!     tracked zones, so the slice adds an exile-only read surface
//!     locally to make the dispatch positively testable. A future
//!     consolidation slice can extend the four-zone walkers once
//!     exile has more readers.
//!
//! ## Test-fixture surface
//!
//! `register_and_inject_object(printing_dict, controller, zone) ->
//! u32` is a test-only helper that (a) parses + appends the
//! printing dict into the in-place registry via `Arc::make_mut`
//! and (b) constructs a fresh GameObject pointing at it on the
//! requested zone. Lives here (not in `state.rs`) because the
//! parallel-slice file-isolation rule prevents touching
//! `state.rs`; a follow-up consolidation slice can move this
//! into a more central location once that constraint relaxes.
//! The id assigned is `1 + max(existing object id)` so it never
//! collides with synthetic-state ids.

use crate::cards::{parse_printing, Effect, RegistryData, TargetFilter};
use crate::state::{GameObject, GameState, StackItem};
use crate::state_ops_zones::{
    ZONE_BATTLEFIELD, ZONE_EXILE, ZONE_GRAVEYARD, ZONE_HAND, ZONE_LIBRARY,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Build a 2-key dict from a StackItem. Mirrors the helper in
/// `state_ops_stack.rs` — duplicated here per the per-slice
/// file-isolation rule (private helpers can't be shared across
/// slice files until the consolidation slice).
fn stack_item_to_dict<'py>(
    py: Python<'py>,
    item: &StackItem,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source_obj_id", item.source_obj_id)?;
    d.set_item("controller", item.controller)?;
    Ok(d)
}

/// Find the source GameObject by id across both players' four
/// tracked zones. Returns a copy (GameObject is `Copy`) along with
/// its printing_id, so the caller can drop the borrow on `state`
/// before mutating it (e.g. to apply damage).
fn find_source_printing_id(state: &GameState, obj_id: u32) -> Option<u32> {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    return Some(obj.printing_id);
                }
            }
        }
    }
    None
}

/// Take the GameObject with id `obj_id` out of whichever of the
/// four tracked zones currently holds it. Returns the owned
/// `GameObject` (and the player_idx + source_zone constant for
/// observability) so the caller can update its `zone` field and
/// push it elsewhere. CoW-safe via `Arc::make_mut`. Returns `None`
/// if no object with that id exists in any tracked zone (the
/// "lost track of source" case).
fn take_source(state: &mut GameState, obj_id: u32) -> Option<(GameObject, u8, u8)> {
    for (pidx, player) in state.players.iter_mut().enumerate() {
        for (zone_const, zone) in [
            (ZONE_HAND, &mut player.hand),
            (ZONE_LIBRARY, &mut player.library),
            (ZONE_BATTLEFIELD, &mut player.battlefield),
            (ZONE_GRAVEYARD, &mut player.graveyard),
        ] {
            if let Some(pos) = zone.iter().position(|o| o.id == obj_id) {
                let v = Arc::make_mut(zone);
                let obj = v.remove(pos);
                return Some((obj, pidx as u8, zone_const));
            }
        }
    }
    None
}

/// Decision returned by the post-resolve dispatch: what zone (if
/// any) does this card_types tuple route to? `None` = leave source
/// in place (empty card_types = synthetic-test escape hatch).
fn post_resolve_destination(card_types: &[String]) -> Option<u8> {
    let is_spell = card_types
        .iter()
        .any(|t| t == "Instant" || t == "Sorcery");
    if is_spell {
        return Some(ZONE_GRAVEYARD);
    }
    let is_permanent = card_types.iter().any(|t| {
        t == "Creature"
            || t == "Artifact"
            || t == "Enchantment"
            || t == "Planeswalker"
            || t == "Battle"
    });
    if is_permanent {
        return Some(ZONE_BATTLEFIELD);
    }
    None
}

/// True iff `card_types` includes "Creature" — gates the
/// summoning-sick stamp on battlefield entry.
fn is_creature(card_types: &[String]) -> bool {
    card_types.iter().any(|t| t == "Creature")
}

/// Apply post-resolve movement (CR 608.2m). Looks up the source
/// printing's `card_types`, dispatches to graveyard / battlefield
/// per the rules above, no-op for empty card_types. Stamps
/// `summoning_sick = true` for creatures landing on the
/// battlefield. Returns the destination zone constant (so the
/// caller can tell whether to fire ETB triggers — round-9 slice
/// gates ETB on `Some(ZONE_BATTLEFIELD)`), or `None` if the source
/// was left in place (empty card_types) or no movement happened.
fn move_source_post_resolve(state: &mut GameState, source_obj_id: u32) -> Option<u8> {
    // Pull the printing's card_types into an owned Vec so we drop
    // the registry borrow before mutating zones.
    let printing_id = find_source_printing_id(state, source_obj_id)?;
    let (destination, creature) = match state.registry.lookup(printing_id) {
        Some(printing) => (
            post_resolve_destination(&printing.card_types),
            is_creature(&printing.card_types),
        ),
        None => return None,
    };
    let dest_zone = destination?; // empty card_types — leave source in place

    // Take from current zone; if not found, treat as silent no-op.
    let (mut obj, controller, _src_zone) = take_source(state, source_obj_id)?;
    obj.zone = dest_zone;
    if dest_zone == ZONE_BATTLEFIELD && creature {
        obj.summoning_sick = true;
    }
    let player = &mut state.players[controller as usize];
    let dest = match dest_zone {
        ZONE_HAND => &mut player.hand,
        ZONE_LIBRARY => &mut player.library,
        ZONE_BATTLEFIELD => &mut player.battlefield,
        ZONE_GRAVEYARD => &mut player.graveyard,
        // Unreachable per `post_resolve_destination`'s closed
        // returns (BATTLEFIELD or GRAVEYARD only); fall back to
        // graveyard rather than panic.
        _ => &mut player.graveyard,
    };
    Arc::make_mut(dest).push(obj);
    Some(dest_zone)
}

/// Apply a single Effect to `state`. Closed-enum match: adding a
/// new variant in the bridge will turn this into a build-time
/// error (exhaustiveness) until handled here — the desired "fail
/// loud" property for free.
fn apply_effect(state: &mut GameState, effect: &Effect) -> PyResult<()> {
    match effect {
        Effect::DealDamage { amount, target } => {
            let defender: u8 = 1 - state.active_player;
            let amount_i32 = *amount;
            match target {
                TargetFilter::AnyPlayer => {
                    state.players[defender as usize].life -= amount_i32;
                }
                TargetFilter::AnyCreature => {
                    // First object in the defender's battlefield Vec
                    // (smallest index). No creature-type filter — see
                    // module docstring. If the battlefield is empty,
                    // the spell fizzles silently.
                    let defender_idx = defender as usize;
                    let bf = &state.players[defender_idx].battlefield;
                    if bf.is_empty() {
                        return Ok(());
                    }
                    let target_id = bf[0].id;
                    let amount_i16 = amount_i32.min(i16::MAX as i32) as i16;
                    let zone: &mut Vec<GameObject> =
                        Arc::make_mut(&mut state.players[defender_idx].battlefield);
                    for obj in zone.iter_mut() {
                        if obj.id == target_id {
                            obj.damage = obj.damage.saturating_add(amount_i16);
                            break;
                        }
                    }
                }
            }
        }
        // Round-8: DrawCards. Caller's controller draws ``count``
        // cards. Convention (mirrors the DealDamage path above): the
        // controller for a resolving spell is `state.active_player`
        // for first slice. Real MTG plumbs the controller through
        // `StackItem.controller`; deferred to a future slice.
        //
        // **Inlining note (cross-file friction).** `draw_card` is a
        // `#[pymethods]` method on `GameState` defined in
        // `state_ops_zones.rs`. Per-slice file-isolation rule
        // forbids touching that file from this slice, so we inline
        // the pop-from-library + push-to-hand logic here, mirroring
        // how round-7's counterspell agent inlined `move_object`. A
        // follow-up consolidation slice can collapse the two
        // implementations once that constraint relaxes.
        //
        // Empty-library handling: if the library is exhausted before
        // we've drawn `count` cards, we stop silently. Real MTG: the
        // player loses on the next state-based-action check (CR
        // 704.5b). Deck-out integration is deferred — for first slice
        // this is a silent short-circuit.
        Effect::DrawCards { count } => {
            let controller = state.active_player as usize;
            let n = (*count).max(0) as usize;
            for _ in 0..n {
                let player = &mut state.players[controller];
                let lib = Arc::make_mut(&mut player.library);
                let Some(mut obj) = lib.pop() else {
                    break; // empty library — silent short-circuit
                };
                obj.zone = ZONE_HAND;
                Arc::make_mut(&mut player.hand).push(obj);
            }
        }
        // Round-8: GainLife. Caller's controller gains `amount` life.
        // Direct field write (no `gain_life` call) so a negative
        // compile-time amount silently subtracts rather than
        // tripping the round-1 raise-on-negative guard. No real MTG
        // card today wants this, but the lowest-friction shape for
        // first slice is "don't validate, just write".
        Effect::GainLife { amount } => {
            let controller = state.active_player as usize;
            state.players[controller].life += *amount;
        }
        // Round-10: ReturnToHand. Bounce a target permanent back to
        // its current controller's hand (= owner's hand in the
        // first-slice convention where controller == owner).
        //
        // Targeting (mirrors DealDamage):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for a bounce. Raise
        //     ValueError loudly, matching the unknown-effect-kind
        //     shape — schema-allowed but doesn't make sense.
        //
        // **Inlining note (cross-file friction).** `move_object` is a
        // private helper on `state_ops_zones.rs`; per per-slice
        // file-isolation, we inline the minimal Arc::make_mut
        // pop-from-current-zone + push-to-target-hand here, mirroring
        // how DrawCards inlines the pop-library + push-hand logic.
        Effect::ReturnToHand { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "ReturnToHand with AnyPlayer target is semantically nonsense (cannot bounce a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let defender = (1 - state.active_player) as usize;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Pop from defender's battlefield (target's current
                // zone). Use Arc::make_mut for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was bf[0] but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                // First-slice: target's controller == owner. The
                // bounce sends to the target's controller's (= owner's)
                // hand — which is `defender` here, since that's
                // where the target lives.
                obj.zone = ZONE_HAND;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                Arc::make_mut(&mut state.players[defender].hand).push(obj);
            }
        },
        // Round-10: Mill. Caller's controller (= active_player, first-
        // slice convention) puts the top `count` cards of their
        // library into their graveyard. "Top" is the LAST element of
        // the library Vec (mirrors round-1 zones convention used by
        // DrawCards above).
        //
        // Empty library mid-mill: stop silently. Real MTG: mill on
        // empty library is NOT a deck-out by itself (unlike a draw
        // attempt — CR 704.5b applies to drawing, not milling). So a
        // silent stop is closer to real rules than the DrawCards
        // short-circuit.
        //
        // Negative `count` is silently clamped to 0 (same shape as
        // DrawCards), which is also the count-0 no-op path.
        Effect::Mill { count } => {
            let controller = state.active_player as usize;
            let n = (*count).max(0) as usize;
            for _ in 0..n {
                let player = &mut state.players[controller];
                let lib = Arc::make_mut(&mut player.library);
                let Some(mut obj) = lib.pop() else {
                    break; // empty library — silent stop
                };
                obj.zone = ZONE_GRAVEYARD;
                Arc::make_mut(&mut player.graveyard).push(obj);
            }
        }
        // Round-11: ExileTarget. Move a target permanent from its
        // current zone to its owner's exile zone (CR 406). First-
        // slice convention: owner == controller, so "owner's exile"
        // is the target's current controller's exile.
        //
        // Targeting (mirrors DealDamage / ReturnToHand):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for exile (you can't
        //     exile a player). Raise ValueError loudly, matching the
        //     ReturnToHand-with-AnyPlayer shape.
        //
        // **Inlining note (cross-file friction).** Zone movement is
        // internal here, same as the ReturnToHand / Mill / DrawCards
        // paths above — `move_object` is a private helper on
        // `state_ops_zones.rs` and per per-slice file-isolation we
        // inline the minimal Arc::make_mut pop-from-current-zone +
        // push-to-target-exile here.
        //
        // **Object-state reset on zone change.** CR 400.7: a card
        // that changes zones is treated as a brand new object — its
        // tapped / summoning_sick / damage / counters fields all
        // reset on the way to exile. Mirrors the ReturnToHand reset
        // convention.
        Effect::ExileTarget { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "ExileTarget with AnyPlayer target is semantically nonsense (cannot exile a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let defender = (1 - state.active_player) as usize;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Pop from defender's battlefield. Use Arc::make_mut
                // for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was bf[0] but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                // Stamp exile zone + reset per-object state. First-
                // slice: target's controller == owner, so push onto
                // `defender`'s exile (where the target lives).
                obj.zone = ZONE_EXILE;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                Arc::make_mut(&mut state.players[defender].exile).push(obj);
            }
        },
        // Round-12: TapTarget. Twiddle-shaped: tap a target permanent.
        //
        // Targeting (mirrors DealDamage / ReturnToHand / ExileTarget):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for tap (you can't
        //     tap a player). Raise ValueError loudly, matching the
        //     ReturnToHand / ExileTarget AnyPlayer shape.
        //
        // **Idempotent on already-tapped target.** Tapping a creature
        // that's already tapped is a silent no-op (no error, no other
        // state change). Mirrors the round-1 ``tap_object`` shape
        // (which returns ``False`` rather than raising).
        //
        // **Inlining note (cross-file friction).** ``tap_object`` is a
        // private ``#[pymethods]`` method in ``state_ops_tap.rs``;
        // per per-slice file-isolation we inline the minimal
        // ``Arc::make_mut`` + ``tapped = true`` here, mirroring how
        // DrawCards / Mill / ReturnToHand / ExileTarget all inline
        // their primitives for the same reason.
        Effect::TapTarget { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "TapTarget with AnyPlayer target is semantically nonsense (cannot tap a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let defender = (1 - state.active_player) as usize;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Flip ``tapped = true`` on the target. Idempotent:
                // already-tapped targets are silently re-stamped
                // True (no-op). Use Arc::make_mut for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.tapped = true;
                        break;
                    }
                }
            }
        },
        // Round-13: UntapTarget. Symmetric counterpart to
        // round-12's TapTarget — same target selection convention,
        // opposite tap state mutation.
        //
        // Targeting (mirrors DealDamage / ReturnToHand / ExileTarget /
        // TapTarget):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for untap (you can't
        //     untap a player). Raise ValueError loudly, matching the
        //     bounce/exile/tap AnyPlayer convention.
        //
        // **Idempotent on already-untapped target.** Untapping a
        // creature that's already untapped is a silent no-op (no
        // error, no other state change). Mirrors the round-12
        // TapTarget already-tapped idempotency shape.
        //
        // **Inlining note (cross-file friction).** ``untap_object`` is
        // a private ``#[pymethods]`` method in ``state_ops_tap.rs``;
        // per per-slice file-isolation we inline the minimal
        // ``Arc::make_mut`` + ``tapped = false`` here, mirroring how
        // round-12 TapTarget inlined its primitive for the same E0624
        // cross-file friction.
        Effect::UntapTarget { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "UntapTarget with AnyPlayer target is semantically nonsense (cannot untap a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let defender = (1 - state.active_player) as usize;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Flip ``tapped = false`` on the target. Idempotent:
                // already-untapped targets are silently re-stamped
                // False (no-op). Use Arc::make_mut for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.tapped = false;
                        break;
                    }
                }
            }
        },
        // Round-14: Sacrifice. CR 701.16 — a sacrificed permanent is
        // put into its owner's graveyard.
        //
        // **Asymmetric vs the bounce/exile/tap convention.** Real MTG:
        // you sacrifice your OWN permanents (not the defender's). The
        // first-slice simplification picks ``state.active_player``'s
        // battlefield[0] (mirroring the source-spell controller
        // convention: a Sacrifice cast by the active player sacrifices
        // one of the active player's own permanents). No creature-type
        // filter, no per-spell target choice. Empty active-battlefield
        // -> fizzle silently (stack still pops).
        //
        // **AnyPlayer target raises** (semantically nonsense — you
        // can't sacrifice a player). Matches the
        // ReturnToHand/ExileTarget/TapTarget/UntapTarget AnyPlayer shape.
        //
        // **Object-state reset on zone change.** CR 400.7: a card that
        // changes zones is treated as a brand new object — its tapped
        // / summoning_sick / damage / counters fields all reset on the
        // way to graveyard. Mirrors the ReturnToHand / ExileTarget
        // reset convention.
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation, we inline the minimal Arc::make_mut pop-from-
        // battlefield + push-to-graveyard here, mirroring how
        // ReturnToHand / ExileTarget / Mill / DrawCards inline their
        // primitives for the same E0624 cross-file reason.
        Effect::Sacrifice { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "Sacrifice with AnyPlayer target is semantically nonsense (cannot sacrifice a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let active = state.active_player as usize;
                let bf = &state.players[active].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Pop from active player's battlefield. Use
                // Arc::make_mut for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[active].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was bf[0] but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                // Stamp graveyard zone + reset per-object state. First-
                // slice: target's controller == owner, so push onto
                // active player's graveyard (where the sacrificed
                // permanent's owner lives).
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                Arc::make_mut(&mut state.players[active].graveyard).push(obj);
            }
        },
        // Round-15: AddCounters. Routes through round-2's signed net-
        // i32 counter model — positive `count` adds +1/+1 counters,
        // negative `count` adds -1/-1 counters; both deltas land on
        // the same i32 field via `saturating_add`. Net values can
        // freely cross zero in either direction.
        //
        // Targeting (mirrors DealDamage / TapTarget / UntapTarget /
        // ReturnToHand / ExileTarget):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for counters (you
        //     can't put counters on a player). Raise ValueError
        //     loudly, matching the bounce/exile/tap/untap AnyPlayer
        //     shape.
        //
        // **Inlining note (cross-file friction).** `add_counters` is a
        // private `#[pymethods]` method on `state_ops_counters.rs`; per
        // per-slice file-isolation we inline the minimal
        // `Arc::make_mut` + `saturating_add` here, mirroring how
        // TapTarget / UntapTarget / Sacrifice inlined their primitives
        // for the same E0624 cross-file reason.
        Effect::AddCounters { target, count } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "AddCounters with AnyPlayer target is semantically nonsense (cannot put counters on a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                let defender = (1 - state.active_player) as usize;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                let delta = *count;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.counters = obj.counters.saturating_add(delta);
                        break;
                    }
                }
            }
        },
        // Round-15: DiscardCard. CR 701.8 — a discarded card is put
        // into its owner's graveyard. First-slice: owner == controller,
        // so the discard lands on the target player's own graveyard.
        //
        // Targeting:
        //   - AnyPlayer -> the DEFENDING player. First-slice: the
        //     active player's spell forces the opponent to discard
        //     (Mind Rot / Wrench Mind shape). Real MTG plumbs target-
        //     player choice through the cast surface; deferred.
        //   - AnyCreature -> semantically nonsense for discard (you
        //     can't discard a creature). Raise ValueError loudly,
        //     mirrored from the bounce/exile/tap/untap AnyPlayer shape
        //     for the symmetric AnyCreature-on-DiscardCard error.
        //
        // **Deterministic choice = hand[0].** First slice picks the
        // first card in the target's hand by index (no per-spell
        // discard choice yet). Empty hand fizzles silently (stack
        // still pops, source still moves post-resolve).
        //
        // **Per-object state reset on zone change** (CR 400.7) —
        // tapped / summoning_sick / damage / counters reset on the way
        // to graveyard, mirroring the ReturnToHand / ExileTarget /
        // Sacrifice reset convention. (In practice a card in hand has
        // those fields at default already, but keep the reset for
        // schema-uniformity with the other zone-change paths.)
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation we inline the minimal `Arc::make_mut`
        // `Vec::remove(0)` + push-to-graveyard here, mirroring
        // ReturnToHand / ExileTarget / Sacrifice / Mill / DrawCards.
        Effect::DiscardCard { target } => match target {
            TargetFilter::AnyCreature => {
                return Err(PyValueError::new_err(
                    "DiscardCard with AnyCreature target is semantically nonsense (cannot discard a creature)",
                ));
            }
            TargetFilter::AnyPlayer => {
                let defender = (1 - state.active_player) as usize;
                let hand = &state.players[defender].hand;
                if hand.is_empty() {
                    return Ok(()); // fizzle silently
                }
                // Pop hand[0] (deterministic first-slice choice). Use
                // Arc::make_mut for CoW safety.
                let hand_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].hand);
                let mut obj = hand_mut.remove(0);
                // Stamp graveyard zone + reset per-object state
                // (CR 400.7).
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                Arc::make_mut(&mut state.players[defender].graveyard).push(obj);
            }
        },
        // Round-16: ScryN. CR 701.18 — controller looks at the top
        // ``count`` cards of their library, may put any number on the
        // bottom, the rest stay on top in any order.
        //
        // **First-slice deterministic shortcut: keep all on top.**
        // This satisfies the "scry happened" semantics without
        // requiring a decision callback to choose which cards go to
        // the bottom (deferred to a future slice that plumbs a real
        // "which to bottom" decision surface, mirroring how
        // ``StackItem.controller`` is plumbed deferred-style). Keep-
        // on-top is mechanically a no-op on the library — no zone
        // moves, no order change — so there is literally nothing to
        // mutate here.
        //
        // **Controller convention: ``state.active_player``** — same
        // first-slice shape as DrawCards / GainLife (both also
        // implicit-controller). Real MTG plumbs the spell controller
        // through ``StackItem.controller``; deferred.
        //
        // **Negative ``count`` raises** loudly (CR 701.18 doesn't
        // define Scry -1; surface it as a schema error). This differs
        // from DrawCards / Mill which silently clamp negatives to 0.
        //
        // **count > library size** clamps to library length per
        // CR 701.18d ("if there aren't enough cards in your library,
        // look at all the cards"). With keep-on-top, the clamp is also
        // a no-op — we just don't error.
        //
        // **Inlining note (cross-file friction).** Round-4 landed
        // ``apply_scry`` as a private ``#[pymethods]`` method on
        // ``state_ops_library.rs``; per per-slice file-isolation +
        // E0624 cross-file friction we'd have to inline anyway, and
        // the keep-on-top first-slice has no library mutation, so the
        // dispatch reduces to a sign check. A future slice that
        // plumbs the real "which to bottom" decision can either
        // collapse onto ``apply_scry`` (after a consolidation slice)
        // or grow the inline path here.
        Effect::ScryN { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "ScryN count must be non-negative, got {count}"
                )));
            }
            // First-slice deterministic keep-on-top: no library
            // mutation needed. Empty / clamped library is a silent
            // no-op (nothing to look at, nothing to move). The
            // ``state.active_player`` reference here is the controller
            // we'd resolve against once a real decision callback
            // lands; binding a local makes the controller convention
            // explicit for future readers even though we don't
            // dispatch on it today.
            let _controller = state.active_player as usize;
        }
    }
    Ok(())
}

#[pymethods]
impl GameState {
    /// Resolve the top stack item. Returns the resolved item as a
    /// dict (same shape as `pop_top_of_stack`), or `None` if the
    /// stack was empty.
    ///
    /// Steps:
    /// 1. Peek top; if empty, return None.
    /// 2. Look up the source GameObject by id across all four
    ///    zones on both players.
    /// 3. Read the source printing's effects from the registry.
    /// 4. Dispatch each effect (currently only `DealDamage`).
    /// 5. Move the source per CR 608.2m: instant/sorcery to
    ///    controller's graveyard, permanents to controller's
    ///    battlefield (summoning-sick if creature). Empty
    ///    card_types leaves source in place.
    /// 6. Pop the stack item and return its dict.
    ///
    /// If the source GameObject can't be found in any zone, the
    /// item still pops — no effects fire (treated like a fizzle).
    /// This matches the "lost track of source" failure mode: the
    /// stack drains, the player observes no effect.
    fn resolve_top_of_stack<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Peek without popping so we can read effects before
        // committing to drain.
        let top = match self.stack.last() {
            None => return Ok(None),
            Some(item) => item.clone(),
        };

        // Find the source's printing_id, then collect a snapshot of
        // its effects so we can drop the registry borrow before
        // mutating self.
        let effects: Vec<Effect> = match find_source_printing_id(self, top.source_obj_id)
        {
            Some(printing_id) => match self.registry.lookup(printing_id) {
                Some(printing) => printing.effects.clone(),
                None => Vec::new(),
            },
            None => Vec::new(),
        };

        for effect in &effects {
            apply_effect(self, effect)?;
        }

        // CR 608.2m: post-resolve, move the spell source. Instants /
        // sorceries -> controller's graveyard; permanents (creature,
        // artifact, enchantment, planeswalker, battle) ->
        // controller's battlefield with summoning sickness for
        // creatures. Empty card_types is the synthetic-test escape
        // hatch — leave source in place. Done BEFORE the stack pop
        // so a panic in movement doesn't leave the stack drained
        // with no movement applied (transactional shape).
        let post_resolve_zone = move_source_post_resolve(self, top.source_obj_id);

        // Round-9: fire ETB-triggered effects when (and only when)
        // the source landed on the battlefield. Real MTG: ETB
        // triggers go onto the stack and resolve in priority order.
        // First slice applies them inline, same convention as the
        // spell-effects path. Snapshot etb_effects via the same
        // borrow-then-drop pattern we use for spell effects so we
        // can mutate state while iterating.
        if post_resolve_zone == Some(ZONE_BATTLEFIELD) {
            let etb_effects: Vec<Effect> =
                match find_source_printing_id(self, top.source_obj_id) {
                    Some(printing_id) => match self.registry.lookup(printing_id) {
                        Some(printing) => printing.etb_effects.clone(),
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                };
            for effect in &etb_effects {
                apply_effect(self, effect)?;
            }
        }

        // Now drain the popped item and return its dict.
        let popped = self
            .stack
            .pop()
            .expect("peeked top but pop returned None — stack invariant broken");
        Ok(Some(stack_item_to_dict(py, &popped)?))
    }

    /// **Test-fixture surface.** Register a printing dict into the
    /// in-place registry, construct a fresh GameObject pointing at
    /// it on `target_zone` for player `controller`, and return the
    /// new object's id.
    ///
    /// Used by `tests_rust/test_resolve.py` to set up cross-slice
    /// composition tests (a Bombard-shaped DealDamage source on the
    /// battlefield) without needing a separate `Registry` pyclass
    /// or hand-rolling synthetic states for every test. The id is
    /// `1 + max(existing obj.id across all zones)` so it never
    /// collides with synthetic-state ids.
    ///
    /// Lives in this slice (not `state.rs`) per the per-slice
    /// file-isolation rule. A follow-up consolidation slice can
    /// move it once the parallel-slice constraint relaxes.
    fn register_and_inject_object(
        &mut self,
        printing_dict: &Bound<'_, PyDict>,
        controller: u8,
        target_zone: u8,
    ) -> PyResult<u32> {
        if (controller as usize) >= 2 {
            return Err(PyIndexError::new_err(format!(
                "controller {controller} out of range (must be 0 or 1)"
            )));
        }

        // Append to the registry. Arc::make_mut clones RegistryData
        // on first write only when other clones observe it — same
        // CoW shape as zones.
        let printing = parse_printing(printing_dict)?;
        let reg: &mut RegistryData = Arc::make_mut(&mut self.registry);
        let printing_id = reg.printings.len() as u32;
        reg.printings.push(printing);

        // Pick a fresh obj_id: 1 + max across all zones on both
        // players. Synthetic-state ids start at 0 and pack densely;
        // this guarantees no collision regardless of seed shape.
        let mut next_id: u32 = 0;
        for player in &self.players {
            for zone in [
                &player.hand,
                &player.library,
                &player.battlefield,
                &player.graveyard,
            ] {
                for obj in zone.iter() {
                    if obj.id >= next_id {
                        next_id = obj.id + 1;
                    }
                }
            }
        }

        let obj = GameObject {
            id: next_id,
            printing_id,
            controller,
            zone: target_zone,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: 0,
        };

        let player = &mut self.players[controller as usize];
        let dest = match target_zone {
            ZONE_HAND => &mut player.hand,
            ZONE_LIBRARY => &mut player.library,
            ZONE_BATTLEFIELD => &mut player.battlefield,
            ZONE_GRAVEYARD => &mut player.graveyard,
            other => {
                return Err(PyValueError::new_err(format!(
                    "register_and_inject_object: unsupported target_zone {other} (valid: 0-3)"
                )));
            }
        };
        Arc::make_mut(dest).push(obj);
        Ok(next_id)
    }

    // -------------------------------------------------------------
    // Round-11 exile read helpers.
    //
    // ``find_object_zone`` / ``get_damage`` / ``get_counters`` /
    // ``is_summoning_sick`` only iterate the four pre-r11 zones (hand
    // / library / battlefield / graveyard) — they don't search exile.
    // The exile-dispatch tests need a way to positively assert on an
    // exiled object's location and reset state, so we add the minimal
    // read surface here. Lives in this slice's file per per-slice
    // file-isolation; a follow-up consolidation slice can extend
    // ``find_object_zone`` (and friends) once exile has more readers.
    // -------------------------------------------------------------

    /// Per-player exile zone size, mirror of ``graveyard_sizes`` /
    /// ``battlefield_sizes``.
    #[getter]
    fn exile_sizes(&self) -> [usize; 2] {
        [self.players[0].exile.len(), self.players[1].exile.len()]
    }

    /// True iff some object with id `obj_id` is in either player's
    /// exile zone.
    fn exile_contains(&self, obj_id: u32) -> bool {
        for player in &self.players {
            if player.exile.iter().any(|o| o.id == obj_id) {
                return true;
            }
        }
        false
    }

    /// Read the `tapped` flag for an object in either player's exile.
    /// Returns `None` if no exile object with that id exists.
    fn exile_object_tapped(&self, obj_id: u32) -> Option<bool> {
        for player in &self.players {
            for obj in player.exile.iter() {
                if obj.id == obj_id {
                    return Some(obj.tapped);
                }
            }
        }
        None
    }

    /// Read the `summoning_sick` flag for an object in either
    /// player's exile.
    fn exile_object_summoning_sick(&self, obj_id: u32) -> Option<bool> {
        for player in &self.players {
            for obj in player.exile.iter() {
                if obj.id == obj_id {
                    return Some(obj.summoning_sick);
                }
            }
        }
        None
    }

    /// Read the `damage` marker for an object in either player's
    /// exile.
    fn exile_object_damage(&self, obj_id: u32) -> Option<i16> {
        for player in &self.players {
            for obj in player.exile.iter() {
                if obj.id == obj_id {
                    return Some(obj.damage);
                }
            }
        }
        None
    }

    /// Read the `counters` net value for an object in either player's
    /// exile.
    fn exile_object_counters(&self, obj_id: u32) -> Option<i32> {
        for player in &self.players {
            for obj in player.exile.iter() {
                if obj.id == obj_id {
                    return Some(obj.counters);
                }
            }
        }
        None
    }

    /// Read the `zone` field stamped on an exiled object (should be
    /// `ZONE_EXILE` after a successful ExileTarget dispatch).
    fn exile_object_zone_field(&self, obj_id: u32) -> Option<u8> {
        for player in &self.players {
            for obj in player.exile.iter() {
                if obj.id == obj_id {
                    return Some(obj.zone);
                }
            }
        }
        None
    }
}
