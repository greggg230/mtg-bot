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
//! - **ETB-triggered effects** (round-9 slice; Slice 3 of the
//!   ETB-trigger closure retired the inline path). After
//!   post-resolve movement places a permanent on the controller's
//!   battlefield, every ENTERS-shaped triggered ability on the
//!   source printing fires through the predicate-path
//!   `dispatch_event(Event::Enters, source_id)` walker. Legacy
//!   `etb_effects: Vec<Effect>` entries are bridged to the same
//!   substrate at parse time (`cards.rs::parse_printing` synthesizes
//!   a `TriggeredAbility { event: Enters, predicate: SourceIsSelf,
//!   effect: ... }` per `etb_effects` entry), so a printing carrying
//!   `etb_effects=[DrawCards(1)]` and a printing carrying
//!   `triggered_abilities=[{event: enters, predicate: source_is_self,
//!   effect: draw_cards(1)}]` produce identical observable state.
//!   First-slice simplifications (mirrored in
//!   `tests_rust/test_resolve.py`):
//!   - **Real on-stack triggering** post-Slice-3. ENTERS triggers
//!     get pushed onto the stack via `push_triggered_ability` and
//!     resolve LIFO on the next priority round. Pre-Slice-3 they
//!     applied inline at resolution time; the on-stack shape is
//!     CR-faithful and supports response windows.
//!   - **Gating: only fire when the source landed on battlefield.**
//!     If post-resolve sent the source to graveyard (instant /
//!     sorcery) or left it in place (empty card_types escape
//!     hatch), ENTERS triggers are skipped — by definition the
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

use crate::bf_iter::bf_iter_etb_sorted;
use crate::cards::{parse_printing, Effect, Event, RegistryData, TargetFilter};
use crate::state::{
    GameObject, GameState, PendingDecision, StackItem, StackItemKind, UeotModifier,
};
use crate::state_ops_decisions::auto_resolve_pending;
use crate::state_ops_zones::{
    mark_left_battlefield_if_nonland, ZONE_BATTLEFIELD, ZONE_EXILE,
    ZONE_GRAVEYARD, ZONE_HAND, ZONE_LIBRARY, ZONE_STACK,
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
/// tracked zones. Returns its `printing_id` (a `u32`) so the
/// caller can drop the borrow on `state` before mutating it
/// (e.g. to apply damage). Round-27 note: pre-r27 this returned a
/// full Copy of the GameObject; the UEOT-modifiers Vec made
/// GameObject `Clone` rather than `Copy`, but every caller only
/// needs the `printing_id`, so the return shape contracted to a
/// scalar.
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
    // G7 S5a (Option β): the cast spell's owned GameObject lives in
    // `state.stack_objects` (zone=ZONE_STACK) while it is on the
    // stack — NOT in any per-player Vec. Search it as the 5th source
    // zone so `move_source_post_resolve` can discover the resolving
    // spell's printing/effects exactly as before S5a (when the spell
    // was hand-resident). Python's analog reads the source directly
    // off `item.source` (`actions.py:2708`) — same object, no zone
    // walk; the id walk here is Rust's id-keyed equivalent.
    for obj in state.stack_objects.iter() {
        if obj.id == obj_id {
            return Some(obj.printing_id);
        }
    }
    None
}

/// **Aura-attachment Phase 4 S3 closure (2026-05-21) — Moonlit
/// Meditation token-creation replacement.**
///
/// Returns the `(token_pid, supertype_override)` pair that a
/// `CreateToken`-family arm should use to mint a token batch for
/// `controller_idx`. The default return is `(requested_pid, false)`
/// — no replacement, no override — and is hit on every state where
/// no Moonlit-shape Aura is in play OR the controller has already
/// created tokens this turn.
///
/// **Replacement teeth.** Substitution fires only when all of the
/// following hold (mirrors Python's
/// `_find_moonlit_replacement_source` + the surrounding membership
/// check at `effects/generic/tokens.py:217-247`):
///
/// 1. `controller_idx` is within the per-player flag array bounds
///    AND `state.created_tokens_this_turn[controller_idx] == false`
///    (this is the FIRST token batch this turn for that player).
/// 2. Some battlefield permanent under `controller_idx`'s control
///    carries a printing with
///    `aura_replaces_first_tokens_with_copies = true`.
/// 3. That Aura's `attached_to` is `Some(host_id)` AND the host
///    object is alive on either player's battlefield. Mirrors
///    Python's `next(o for o in state.battlefield if o.id ==
///    moonlit.attached_to)` — a host that has left the battlefield
///    leaves the Aura with a stale `attached_to`, in which case the
///    replacement is a no-op (the Aura would also be sacrificed via
///    CR 704.5n SBA, but the per-call gate here is defensive).
///
/// **Override flag semantics.** When substitution fires, the
/// returned `supertype_override` is `true` only when the host's
/// printing does NOT already carry `"Token"` in its `supertypes`
/// Vec. Mirrors Python's conditional `["Token"] if "Token" not in
/// base.supertypes else []` at `tokens.py:240-242` — a Moonlit-on-
/// a-Token-host case (e.g. a Robot token enchanted by Moonlit, a
/// degenerate-but-CR-legal state) is a no-op for supertype
/// injection because the host already has the supertype.
///
/// **Why a single helper.** All four token-mint arms
/// (`Effect::CreateToken`, `CreateTokenTapped`,
/// `CreateTokenEachPlayer`, `CreateTokenCopyOfTargetCreatureYouControl`)
/// must consult the same replacement check to mirror Python's
/// `CreateToken.resolve` semantics. The Moonlit replacement is a
/// **per-recipient** check (`each_player` iterates the recipient
/// set, calling the same gate per recipient), so the helper takes
/// `controller_idx` rather than reading `active_player`.
pub(crate) fn moonlit_token_substitution(
    state: &GameState,
    controller_idx: u8,
    requested_pid: u32,
) -> (u32, bool) {
    let cidx = controller_idx as usize;
    if cidx >= state.created_tokens_this_turn.len() {
        return (requested_pid, false);
    }
    if state.created_tokens_this_turn[cidx] {
        return (requested_pid, false);
    }
    // Locate a Moonlit-shape Aura under controller_idx's control
    // whose host is alive on the battlefield. Iterates owner zones in
    // p0-then-p1 order, then within each zone in insertion order —
    // matching Python's `for o in state.battlefield` walk order (the
    // centralized state.battlefield list is built p0 first per the
    // existing CE4 ordering).
    let mut host_pid: Option<u32> = None;
    'outer: for owner in &state.players {
        for aura in owner.battlefield.iter() {
            if aura.controller != controller_idx {
                continue;
            }
            let aura_printing = match state.registry.lookup(aura.printing_id) {
                Some(p) => p,
                None => continue,
            };
            if !aura_printing.aura_replaces_first_tokens_with_copies {
                continue;
            }
            let host_id = match aura.attached_to {
                Some(h) => h,
                None => continue,
            };
            // Resolve the host's printing_id by scanning both bf
            // zones for the host obj.
            for host_owner in &state.players {
                for host_obj in host_owner.battlefield.iter() {
                    if host_obj.id == host_id {
                        host_pid = Some(host_obj.printing_id);
                        break 'outer;
                    }
                }
            }
        }
    }
    let Some(pid) = host_pid else {
        return (requested_pid, false);
    };
    // Conditional supertype injection — skip when the host's
    // printing already carries "Token" (mirrors Python's list-append
    // guard at tokens.py:240-242).
    let override_flag = state
        .registry
        .lookup(pid)
        .map(|p| !p.supertypes.iter().any(|s| s == "Token"))
        .unwrap_or(false);
    (pid, override_flag)
}

/// Take the GameObject with id `obj_id` out of whichever of the
/// four tracked zones currently holds it. Returns the owned
/// `GameObject` (and the player_idx + source_zone constant for
/// observability) so the caller can update its `zone` field and
/// push it elsewhere. CoW-safe via `Arc::make_mut`. Returns `None`
/// if no object with that id exists in any tracked zone (the
/// "lost track of source" case).
fn take_source(state: &mut GameState, obj_id: u32) -> Option<(GameObject, u8, u8)> {
    // G7 S5a (Option β): check the spell-stack-source-zone store
    // FIRST. The cast moved the spell hand → `stack_objects` with
    // `obj.zone = ZONE_STACK` and `obj.controller = caster` (the
    // direct mirror of Python's `p.hand.remove(card)` +
    // `card.zone = Zone.STACK` + `card.controller_id = player_id`,
    // `actions.py:1403`/`:1751`/`:1752`). The returned pidx is the
    // object's OWN `controller` — for a hand/gy/exile-origin cast
    // that is the caster, byte-identical to the player-index the
    // pre-S5a hand walk returned (you cast from your own hand). This
    // is the moment Python's `card = item.source` reads the owned
    // object off the StackItem (`actions.py:2708`); Rust takes the
    // owned object out of the store here so
    // `move_source_post_resolve` can push it to its post-resolve
    // destination, byte-equal to Python's `put_into_battlefield`.
    if let Some(pos) =
        state.stack_objects.iter().position(|o| o.id == obj_id)
    {
        let obj = state.stack_objects.remove(pos);
        let controller = obj.controller;
        return Some((obj, controller, ZONE_STACK));
    }
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
    if dest_zone == ZONE_BATTLEFIELD {
        // CR 302.1: only creatures have summoning sickness. A
        // permanent's pre-bf `summoning_sick` is `true` by default
        // (the deck-loader sets every library card to `true` for
        // parity with Python's universal default). On battlefield
        // entry, Python's `put_into_battlefield`
        // (`actions.py:3003-3006`) EXPLICITLY clears the flag for
        // non-creatures and stamps it `true` for creatures — neither
        // branch relies on the default. Mirror that here so an
        // Enchantment / Artifact / Planeswalker / Battle (or an Aura
        // routed through this mover) doesn't carry a stale `true`
        // summoning-sick bit onto the battlefield. Without the else
        // branch, non-creature permanent casts diverge from Python on
        // the per-object summoning_sick slot in the field-view /
        // fingerprint (surfaced by the Tractor Beam production-deck
        // shadow bake-in at archetype 2 / seed 1 / step 874).
        obj.summoning_sick = creature;
    }
    // CE4 closure: when this resolution lands the source on the
    // battlefield (ResolveCreature / ResolveArtifact / Enchantment /
    // Planeswalker — every permanent-typed source), stamp a fresh
    // ETB timestamp so it sorts after every previously-ETBed object.
    // Non-bf destinations (graveyard for instants / sorceries) leave
    // the field at its prior (likely zero) value — the snapshot /
    // fingerprint only sort-reads `etb_timestamp` for bf entries.
    if dest_zone == ZONE_BATTLEFIELD {
        obj.etb_timestamp = crate::state::reserve_etb_timestamp(state);
        // CE20 Slice 5 (Walker loyalty — `starting_loyalty` ETB
        // seeding). CR 606.5b: a planeswalker enters with loyalty
        // counters equal to its printed starting loyalty. No-op for
        // non-walker printings.
        crate::state::seed_starting_loyalty_for_etb(state, &mut obj);
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

/// **Aura-attachment substrate — Phase 1 STEP 4 (round-78,
/// 2026-05-20).** Post-resolve mover for Aura spells. Mirrors
/// Python's Aura resolve path at `actions.py:2720-2758`:
///
/// 1. Read the cast-time bound target via `target_id_opt` (the
///    caller passes `top.target_obj_id` — STEP 3 binds this on
///    `submit_enchant_target_decision`). NB: we do *not* read
///    `state.current_resolve_payload_target_id` here — the resolver
///    restores that swap BEFORE this mover runs (the `saved_target`
///    bracket completes before `move_*_post_resolve` is called).
///    The target slot is threaded explicitly via the function arg.
/// 2. Validate at resolve time per CR 608.2b:
///    - Target must be on a battlefield.
///    - Target's card_types must match the printing's
///      `aura_enchant_type` (`"artifact"` → Artifact; default
///      `"creature"` → Creature).
///    - If `aura_yours_only=true`, target's controller must equal
///      the Aura's controller (the resolving spell's controller).
/// 3. **Illegal target → fizzle (CR 608.2b).** Route the Aura to its
///    owner's graveyard with `attached_to = None`. Returns
///    `Some(ZONE_GRAVEYARD)` so the caller skips the ETB dispatch
///    (the Aura never entered the battlefield).
/// 4. **Legal target → attach.** Route to controller's battlefield,
///    stamp `obj.attached_to = Some(host_id)`, reserve fresh ETB
///    timestamp. Returns `Some(ZONE_BATTLEFIELD)` so the caller
///    fires ETB triggers (none today for the S1 P/T/keyword shape,
///    but the substrate is now in place for STEP-5+ triggered-on-
///    attach auras).
///
/// **Layer-7 P/T + layer-6 keyword grants** are NOT applied here —
/// they are *continuous* effects sourced by the bf-resident Aura and
/// keyed on `attached_to`. The dynamic reader analogue in
/// `state_ops_combat.rs::battlefield_creature_power` /
/// `_toughness` / `_has_keyword` scans the bf for any Aura whose
/// `attached_to` matches the queried object and folds the bonuses in
/// at read time. The Python source-of-truth scan is in
/// `game.py:969-996` (P/T) and `game.py:1120-1145` (keywords).
///
/// `controller_id` is the resolving spell's controller (read from
/// the StackItem). The fizzle path needs it to route the Aura to the
/// *owner's* graveyard — Python's `_send_to_graveyard_or_library`
/// walks owner not controller, and the Aura's stack-source
/// `controller` field is the caster (the owner of a hand-cast
/// spell). For cast-from-exile (STEP-6 territory) this would
/// diverge, but Phase 1's only cast origin is hand → the resolving
/// controller IS the owner.
fn move_aura_post_resolve(
    state: &mut GameState,
    source_obj_id: u32,
    controller_id: u8,
    target_id_opt: Option<u32>,
) -> Option<u8> {
    // Look up printing fields we need (aura_enchant_type, yours_only).
    let printing_id = find_source_printing_id(state, source_obj_id)?;
    let (enchant_type, yours_only) = match state.registry.lookup(printing_id) {
        Some(p) => {
            if !p.is_aura {
                // Defensive: caller should only route Aura spells
                // here. Fall through to the regular mover by returning
                // None — the caller treats that as "no aura semantics
                // apply" and re-routes through move_source_post_resolve.
                return None;
            }
            (p.aura_enchant_type.clone(), p.aura_yours_only)
        }
        None => return None,
    };
    // Resolve-time target legality: target must (a) exist on a bf,
    // (b) match the enchant type, (c) satisfy yours_only.
    let legal_target_id: Option<u32> = match target_id_opt {
        Some(tid) => {
            let mut matched: Option<u32> = None;
            'outer: for player in state.players.iter() {
                for obj in player.battlefield.iter() {
                    if obj.id != tid {
                        continue;
                    }
                    // Found candidate; check type + yours_only.
                    let Some(p) = state.registry.lookup(obj.printing_id) else {
                        break 'outer;
                    };
                    let type_ok = if enchant_type == "artifact" {
                        p.card_types.iter().any(|t| t == "Artifact")
                    } else {
                        // Default "creature" — matches the cast-time
                        // filter in cast_aura.
                        p.card_types.iter().any(|t| t == "Creature")
                    };
                    if !type_ok {
                        break 'outer;
                    }
                    if yours_only && obj.controller != controller_id {
                        break 'outer;
                    }
                    matched = Some(obj.id);
                    break 'outer;
                }
            }
            matched
        }
        None => None,
    };

    // Take the spell-source object out of the stack store.
    let (mut obj, owner, _src_zone) = take_source(state, source_obj_id)?;

    if let Some(host_id) = legal_target_id {
        // Legal target → battlefield with attached_to stamped.
        // CR 303.4: an Aura entering the battlefield must be attached
        // to its enchant target.
        obj.zone = ZONE_BATTLEFIELD;
        obj.attached_to = Some(host_id);
        obj.etb_timestamp = crate::state::reserve_etb_timestamp(state);
        // CR 302.1: only creatures have summoning sickness. Auras are
        // Enchantments — clear the summoning-sick flag the deck
        // loader stamped on cards-in-library to mirror Python's
        // `put_into_battlefield` `actions.py:3003-3006` non-creature
        // branch. Without this, the resolved Aura carries a stale
        // `summoning_sick=true` onto the bf (surfaced by the Tractor
        // Beam production-deck shadow bake-in at archetype 2 / seed
        // 1 / step 874).
        obj.summoning_sick = false;
        // Sticky-controller convention: the Aura's controller is the
        // resolving spell's controller (the caster). Already set by
        // cast_aura's rehome path, but assert it via the owner walk
        // for the fizzle-vs-attach symmetry below.
        let _ = owner;
        // Aura-attachment substrate — Phase 2 S2 (round-79+,
        // 2026-05-20). Layer-2 control-change static grant: if the
        // Aura's printing carries `aura_grants_control`, flip the
        // host's `controller` to the Aura's controller. Stash the
        // host's prior controller in `controller_baseline` on FIRST
        // grant (the captured-baseline so subsequent grants revert
        // through the original owner). Mirrors Python's
        // `apply_aura_static_grants_on_attach` control-change branch
        // (`actions.py:2828-2854`).
        //
        // **Honest-stop:** timestamp-based LIFO ordering of multiple
        // control-changing Auras on the same host is not modeled in
        // this slice — Phase-2 S2 carriers (Duskmourn's Domination)
        // are single-host single-grant in the live EoE pool. A
        // second Aura would observe the first's host swap and grant
        // its OWN controller (correctly by latest-wins); only the
        // revert-via-baseline assumes one outstanding grant. Phase 3+
        // can add a continuous-effect registry for the multi-grant
        // CR 613 layer-2 timestamp walk.
        let grants_control = state
            .registry
            .lookup(printing_id)
            .map(|p| p.aura_grants_control)
            .unwrap_or(false);
        if grants_control {
            // Find the host in whatever bf it currently lives on,
            // capture-baseline (only if no baseline yet), then flip
            // the host's controller to the resolving spell's
            // controller. We do this BEFORE pushing the Aura so the
            // host walk doesn't find the freshly-pushed Aura. Walk
            // both players' bfs.
            for player_idx in 0..state.players.len() {
                let pos = state.players[player_idx]
                    .battlefield
                    .iter()
                    .position(|o| o.id == host_id);
                if let Some(idx) = pos {
                    let bf = Arc::make_mut(
                        &mut state.players[player_idx].battlefield,
                    );
                    let h = &mut bf[idx];
                    if h.controller_baseline.is_none() {
                        h.controller_baseline = Some(h.controller);
                    }
                    h.controller = controller_id;
                    break;
                }
            }
        }
        let bf =
            &mut state.players[controller_id as usize].battlefield;
        Arc::make_mut(bf).push(obj);
        Some(ZONE_BATTLEFIELD)
    } else {
        // No legal target → fizzle (CR 608.2b). Aura goes to owner's
        // graveyard. Phase 1 cast origin is hand only, so owner ==
        // controller; the comment above explains why we route via
        // controller_id for parity with Python's
        // `_send_to_graveyard_or_library`.
        obj.zone = ZONE_GRAVEYARD;
        obj.attached_to = None;
        let gy =
            &mut state.players[controller_id as usize].graveyard;
        Arc::make_mut(gy).push(obj);
        Some(ZONE_GRAVEYARD)
    }
}

/// Apply a single Effect to `state`. Closed-enum match: adding a
/// new variant in the bridge will turn this into a build-time
/// error (exhaustiveness) until handled here — the desired "fail
/// loud" property for free.
///
/// `source_obj_id` is the resolving stack item's source GameObject
/// id. Most arms ignore it (they target the defending player /
/// active_player / a Vec-index pick); the round-23
/// `PutPlusOneCounterOnSelf` arm needs it to locate the "self" /
/// "this creature" recipient — the source object on the
/// battlefield. Threaded through here rather than re-finding from
/// the stack so the existing borrow-then-drop pattern in
/// `resolve_top_of_stack` (snapshot effects, drop registry borrow,
/// iterate) stays intact.
/// Cross-module wrapper for `apply_effect`. Round-28
/// `state_ops_decisions.rs::auto_resolve_pending_decisions` needs to
/// recursively apply the MayDo inner effect when the strategy resolves
/// to "yes"; the private `apply_effect` isn't reachable across module
/// boundaries so we expose this thin pub wrapper.
// --- ETB-trigger substrate (Slice 1) -----------------------------
//
// Implements the dispatch walker + push helper that fan
// `Event::Enters` (and future events) out to every triggered
// ability on every printing on the battlefield. Mirrors Python's
// `dispatch_event` (`src/mtg_bot/actions.py:3283-3399`) +
// `push_triggered_ability` (`src/mtg_bot/actions.py:3172-3280`).
//
// Slice 1 scope (per `docs/RUST_ETB_TRIGGER_AUDIT.md` §6.1):
//
// 1. Walk `bf_iter_etb_sorted(state)` (CE5-introduced helper at
//    `rust/src/bf_iter.rs:61`) — order matches Python's
//    `list(state.battlefield)`.
// 2. Snapshot the bf list AT DISPATCH ENTRY so triggers that ETB
//    new permanents during dispatch don't observe their own event
//    in the same pass (mirrors Python's
//    `actions.py:3343-3345` snapshot comment).
// 3. For each (source, ability) pair, check (a) the event matches
//    and (b) the predicate evaluates true.
// 4. Once-per-turn gate: skip if `(source_id, ability_index) in
//    state.triggered_this_turn`.
// 5. On match + non-gated: call `push_triggered_ability` to put a
//    `StackItem::new_trigger` on the stack.
//
// **Defers to Slice 2+:**
// - APNAP ordering (Python deliberately doesn't either — see
//   audit §2.5).
// - Panharmonicon-shape `etb_trigger_multiplier`.
// - Target selection on triggers that need it (DealDamage trigger
//   with a chosen target). Slice 1 only proves DrawCards-shape /
//   GainLife-shape (no target) for the substrate smoke probe.
// - Fan-out from the other 11 ETB-stamping sites (audit §4).
// - Combat-damage aggregator dedup.

/// **Slice 1 — dispatch a single event to every triggered ability
/// on the battlefield.** Walks every permanent in
/// ETB-timestamp order, matches each printing's
/// `triggered_abilities` against `(event, subject_id, source)`,
/// and pushes matching triggers onto the stack via
/// `push_triggered_ability`.
///
/// Slice 1 surface accepts `Event::Enters`. Slice 5 adds
/// `Event::Dies`. Other variants raise `PyValueError` loudly so the
/// parser surface and the dispatcher's coverage stay consistent.
///
/// Slice 5 (DIES): the dying object has already been moved to the
/// graveyard by the caller. The walker still snapshots the
/// **battlefield** as the observer-set, but DIES dispatch
/// additionally includes the dying object as an "extra observer" so
/// `Predicate::SourceIsSelf` triggers (the canonical "when this
/// creature dies" shape) can match. Mirrors Python's
/// `dispatch_event(..., extra_observers=[obj])` pattern at
/// `src/mtg_bot/turn.py:281` for the `move_off_battlefield` site.
pub(crate) fn dispatch_event(
    state: &mut GameState,
    event: Event,
    subject_id: u32,
) -> PyResult<()> {
    // Defensive: Slice 9 wires ENTERS + DIES + CAST + ATTACKS +
    // BLOCKS + LEAVES_BATTLEFIELD; round-58 adds BECOMES_TAPPED;
    // round-13 adds DEALS_DAMAGE (fired by
    // ``dispatch_combat_damage_to_player`` for player-targeted
    // combat-damage entries); round-14 adds the three PHASE_STEP
    // events (BEGINNING_OF_END_STEP, BEGINNING_OF_UPKEEP,
    // BEGINNING_OF_COMBAT) fired by ``advance_phase`` in
    // ``state_ops_steps.rs``; round-15 adds YOU_ATTACK fired by the
    // ``DeclareAttackers`` arm in ``state_ops_actions.rs``; round-16
    // (RR Pins 1+2+3) adds COUNTERS_ADDED (fired after counter
    // placement in the AddCounters / PutPlusOneCounterOnSelf arms),
    // IS_SACRIFICED (fired from the Sacrifice effect arm), and
    // DEALT_DAMAGE (fired when a permanent receives combat damage).
    // Other events declared in the enum stay raise-on-dispatch so a
    // future caller who skipped the fan-out plan surfaces it loudly
    // rather than silently no-op.
    if event != Event::Enters
        && event != Event::Dies
        && event != Event::Cast
        && event != Event::Attacks
        && event != Event::Blocks
        && event != Event::LeavesBattlefield
        && event != Event::BecomesTapped
        && event != Event::DealsDamage
        && event != Event::BeginningOfEndStep
        && event != Event::BeginningOfUpkeep
        && event != Event::BeginningOfCombat
        && event != Event::YouAttack
        && event != Event::CountersAdded
        && event != Event::IsSacrificed
        && event != Event::DealtDamage
        && event != Event::BeginningOfPostcombatMain
        && event != Event::DiscardsCard
    {
        return Err(PyValueError::new_err(format!(
            "dispatch_event: event {event:?} is not yet wired; \
             valid: Event::Enters, Event::Dies, Event::Cast, \
             Event::Attacks, Event::Blocks, Event::LeavesBattlefield, \
             Event::BecomesTapped, Event::DealsDamage, \
             Event::BeginningOfEndStep, Event::BeginningOfUpkeep, \
             Event::BeginningOfCombat, Event::YouAttack, \
             Event::CountersAdded, Event::IsSacrificed, Event::DealtDamage, \
             Event::BeginningOfPostcombatMain, Event::DiscardsCard \
             (Slice 9 + r58 + r13 + r14 + r15 + r16-RR + PPP + MMM)"
        )));
    }

    // Snapshot the bf source-id + ability-payload list AT ENTRY so
    // triggers that ETB new permanents during dispatch don't
    // observe their own event in the same pass. Materializes
    // (source_id, source_controller, ability_index, effect-clone)
    // tuples — once the bf-snapshot is done we no longer hold an
    // immutable borrow on `state`, so `push_triggered_ability`
    // (which mutates `state.stack`) is free to fire.
    //
    // For DIES events, the dying object has been moved to the
    // graveyard already; we synthesize an EXTRA observer entry for
    // it (mirroring Python's `extra_observers=[obj]`) so an
    // `IsSelf`-shape "when this creature dies" trigger can still
    // see itself as a candidate observer.
    //
    // For CAST events, the spell-being-cast is on the STACK at
    // dispatch time (G7 S5a: the cast moves the spell hand →
    // `state.stack_objects` with zone=ZONE_STACK, then dispatches
    // CAST, then pushes the StackItem — mirroring Python's
    // `card.zone = Zone.STACK` BEFORE the `StackItem` append,
    // `actions.py:1751`/`:1759`). The subject is the object in the
    // stack store; SourceIsSelf-shape "when you cast this" triggers
    // need the casting card as an extra observer so the predicate's
    // source==subject test can match. Mirrors the DIES extra-observer
    // pattern. (Pre-S5a the spell was hand-resident at CAST dispatch;
    // the 6th-zone `stack_objects` walk below replaces the hand
    // lookup for that case while keeping the 5-per-player-zone walk
    // for DIES/LEAVES, whose subjects are NOT stack-resident.)
    let firings: Vec<(u32, u8, u32, Effect, bool)> = {
        let bf = bf_iter_etb_sorted(state);
        // Build the observer list: every bf permanent, plus
        // event-specific extra observers (the dying creature for
        // DIES, the cast-spell for CAST). The extra observer is
        // keyed by id so the ability-walk lookup uses the same
        // registry lookup the bf permanents go through.
        let mut observer_entries: Vec<(u32, u8, u32)> = bf
            .iter()
            .map(|o| (o.id, o.controller, o.printing_id))
            .collect();
        if event == Event::Dies
            || event == Event::Cast
            || event == Event::LeavesBattlefield
            || event == Event::IsSacrificed
        {
            // DIES: look up the dying object (now in graveyard).
            // CAST: look up the cast object (still in hand).
            // LEAVES_BATTLEFIELD: look up the leaving object (now
            // in graveyard / exile / hand / library depending on
            // the bf-exit primitive that fired the dispatch). The
            // four-zone walk + library covers each destination.
            // IS_SACRIFICED: look up the sacrificed object (now in
            // graveyard). Same extra-observer pattern as DIES —
            // the sacrificed permanent has already left the
            // battlefield so ``IsSelf``-shape triggers on the
            // sacrificed object need the graveyard walk to find it.
            //
            // All three share the multi-zone walk shape. Skip if
            // not found — the dispatch caller is loud-but-safe.
            //
            // Dedup guard: if the subject somehow also appears in
            // `bf` (shouldn't happen for these events' callers),
            // don't double-register it as an observer.
            if !observer_entries.iter().any(|(id, _, _)| *id == subject_id) {
                let mut found = false;
                'outer: for player in &state.players {
                    for zone in [
                        &player.hand,
                        &player.library,
                        &player.battlefield,
                        &player.graveyard,
                        &player.exile,
                    ] {
                        for obj in zone.iter() {
                            if obj.id == subject_id {
                                observer_entries.push((
                                    obj.id,
                                    obj.controller,
                                    obj.printing_id,
                                ));
                                found = true;
                                break 'outer;
                            }
                        }
                    }
                }
                // G7 S5a: 6th zone — the spell-stack-source-zone
                // store. For a CAST event the subject is here (the
                // cast moved it hand → `stack_objects`); the 5
                // per-player zones above no longer hold it. DIES /
                // LEAVES subjects are never stack-resident so this is
                // a correct no-op for them. Keeps SourceIsSelf "when
                // you cast this" CAST triggers + respond-to-cast
                // firing byte-equal to pre-S5a / Python (without it
                // they would SILENTLY stop matching —
                // `feedback_silent_noop_pipelines`).
                if !found {
                    for obj in state.stack_objects.iter() {
                        if obj.id == subject_id {
                            observer_entries.push((
                                obj.id,
                                obj.controller,
                                obj.printing_id,
                            ));
                            break;
                        }
                    }
                }
            }
        }
        let mut firings = Vec::new();
        for (source_id, source_controller, printing_id) in &observer_entries {
            // Resolve the source's printing -> its triggered_abilities.
            let printing = match state.registry.lookup(*printing_id) {
                Some(p) => p,
                None => continue,
            };
            for (ability_index, ability) in
                printing.triggered_abilities.iter().enumerate()
            {
                if ability.event != event {
                    continue;
                }
                if !ability
                    .predicate
                    .matches(state, *source_id, subject_id)
                {
                    continue;
                }
                // Once-per-turn gate: skip if already fired this turn.
                if ability.once_per_turn
                    && state
                        .triggered_this_turn
                        .contains(&(*source_id, ability_index as u32))
                {
                    continue;
                }
                firings.push((
                    *source_id,
                    *source_controller,
                    ability_index as u32,
                    ability.effect.clone(),
                    ability.once_per_turn,
                ));
            }
        }
        firings
    };

    // Push each matching firing onto the stack. Order matches the
    // bf walk (Python's "no APNAP simplification" — audit §2.5).
    for (source_id, controller, ability_index, effect, once_per_turn) in firings {
        push_triggered_ability(
            state,
            source_id,
            controller,
            ability_index,
            effect,
            once_per_turn,
        )?;
    }
    Ok(())
}

/// **Round-58 BECOMES_TAPPED event bridge — the centralized "tap a
/// permanent and fire its becomes-tapped triggers" chokepoint.**
///
/// This is the Rust analogue of Python's `turn.tap_permanent`
/// (`src/mtg_bot/turn.py:542-571`): flip `tapped` on the battlefield
/// permanent `obj_id`, and — iff the flip actually changed state
/// (the object was UNTAPPED and is now tapped) — dispatch
/// `Event::BecomesTapped` for it.
///
/// **No-op-if-already-tapped parity.** Python's `tap_permanent`
/// early-returns when `obj.tapped` is already True (no re-trigger).
/// This helper mirrors that exactly: a tap of an already-tapped
/// permanent is a silent no-op (no event, no `Arc::make_mut` clone).
/// The "did it actually change" gate is what makes the event-firing
/// set byte-equal to Python's.
///
/// **Subject stays on the bf.** Tapping does not move the permanent,
/// so the subject is still on the battlefield at dispatch time —
/// `dispatch_event`'s bf-snapshot observer set already contains it
/// and `Predicate::SourceIsSelf` ("whenever this creature becomes
/// tapped") matches naturally via `source_id == subject_id`. No
/// DIES/CAST-style extra-observer entry is needed (this is exactly
/// the ATTACKS shape — see the `Event::Attacks` dispatch at
/// `state_ops_combat.rs`).
///
/// **Why a free function, not a `tap_object` rewrite.** The existing
/// `tap_object` pymethod in `state_ops_tap.rs` is a test/cost
/// primitive that deliberately does NOT dispatch (its callers
/// pre-date the event). Routing it through dispatch would change
/// every existing tap-cost test's stack invariants. Per the
/// per-file isolation rule, the production tap sites
/// (`state_ops_mana`, `state_ops_actions`, the TapTarget-family
/// effect arms in this file, and the attack-tap in
/// `state_ops_combat`) each inline a flip + a call to this single
/// helper so the dispatch logic lives in exactly one place and the
/// no-op gate can't drift between sites.
///
/// Returns `Ok(true)` if the permanent became tapped (event fired),
/// `Ok(false)` if it was already tapped / not on any bf (no-op), or
/// `Err` if the dispatched trigger's effect raised.
pub(crate) fn tap_permanent_and_dispatch(
    state: &mut GameState,
    obj_id: u32,
) -> PyResult<bool> {
    // Phase 1: flip, scoped so the Arc::make_mut borrow is released
    // before dispatch_event re-borrows state mutably. Mirrors the
    // `state_ops_tap::tap_object` CoW shape: skip the make_mut clone
    // entirely on the already-tapped no-op path.
    let mut changed = false;
    'find: for player in state.players.iter_mut() {
        if player.battlefield.iter().any(|o| o.id == obj_id) {
            // Pre-check tapped before paying the make_mut clone.
            let already_tapped = player
                .battlefield
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.tapped)
                .unwrap_or(true);
            if already_tapped {
                break 'find;
            }
            let zone: &mut Vec<GameObject> =
                std::sync::Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    obj.tapped = true;
                    changed = true;
                    break 'find;
                }
            }
        }
    }
    if !changed {
        // Already tapped, or no bf permanent with that id — Python's
        // `tap_permanent` early-return path. No event.
        return Ok(false);
    }
    // Phase 2: fire becomes-tapped triggers. Same dispatcher every
    // other event flows through.
    dispatch_event(state, Event::BecomesTapped, obj_id)?;
    Ok(true)
}

/// **Round-13 DEALS_DAMAGE dispatch — fires DEALS_DAMAGE triggers for
/// one (attacker, defending_player) combat-damage interaction.**
///
/// Called from ``state_ops_combat::GameState::apply_damage_pass`` for
/// each plan entry whose ``target`` is ``DamageTarget::Player``. The
/// caller already filters to player-targeted entries only, so the
/// ``IsPlayer`` half of Python's two-subject predicate
/// ``IsSelf(subject="source") & Subject("target", IsPlayer())`` is
/// implicit here: every call is player-targeted by definition.
///
/// **Single-subject simplification (round-13 scope).** Python's
/// ``Event.DEALS_COMBAT_DAMAGE`` carries TWO named subjects
/// ``{source: <attacker>, target: <player>}``; Rust's
/// ``dispatch_event`` API carries a SINGLE ``subject_id``. This
/// helper passes ``source_id`` (the attacker) as the subject so that
/// ``Predicate::SourceIsSelf`` (``source_id == subject_id``) fires
/// naturally for the attacker itself. The ``target`` subject (the
/// defending player id ``target_player``) is NOT passed to
/// ``dispatch_event``; it is written into
/// ``state.resolving_combat_damage_target_player_id`` so that effects
/// like ``DrawCards(1)`` that don't need a target can still have the
/// target player available if needed by future effects, and so that
/// the ``fixed_targets`` channel remains available for effects like
/// Virulent Silencer's ``PutPoisonCountersOnTargetPlayer`` (deferred
/// — Virulent Silencer's predicate requires ``Not(SourceIsSelf)``
/// filtering on artifact-creature sources which is out of scope for
/// round-13).
///
/// **Bridge parity.** Python's
/// ``_fire_combat_damage_to_player_triggers`` calls
/// ``dispatch_event(state, Event.DEALS_COMBAT_DAMAGE, source=source,
/// target=target_player_obj)`` for each (source, target_player,
/// amount) triple where amount > 0 and source is still on the
/// battlefield. This function mirrors that shape: guard on
/// ``source_id`` still being on the battlefield (source may have died
/// from simultaneous damage via deathtouch), then call
/// ``dispatch_event(state, Event::DealsDamage, source_id)``.
pub(crate) fn dispatch_combat_damage_to_player(
    state: &mut GameState,
    source_id: u32,
    target_player: u8,
    amount: i32,
) -> PyResult<()> {
    // Guard: the source must still be on the battlefield. A creature
    // that took lethal simultaneous damage may have already been
    // removed by SBAs — don't fire its trigger posthumously (mirrors
    // Python's ``if source is None: continue`` guard in
    // ``_fire_combat_damage_to_player_triggers``).
    let source_on_bf = state.players.iter().any(|p| {
        p.battlefield.iter().any(|o| o.id == source_id)
    });
    if !source_on_bf {
        return Ok(());
    }

    // Write the resolving-combat-damage-amount slot so effects like
    // ``DrawCardsEqualToResolvingCombatDamage`` (Starwinder family)
    // can read the amount at resolve time. Mirrors Python's
    // ``combat_damage_amount=amount`` kwarg threaded through
    // ``dispatch_event`` into each ``StackItem``.
    //
    // Also record the target player id on the state so future effects
    // that need it (Virulent Silencer's poison-counter apply) can
    // read it without an extra argument. Cleared to 255 (sentinel =
    // "no active combat damage target") after the dispatch returns.
    state.resolving_combat_damage_amount = amount;
    // **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
    // Write the target player id before dispatch so that triggered
    // effects like ``PutPoisonCountersOnTargetPlayer`` can read it at
    // resolve time (mirrors Python's target=target_player_obj threading
    // through _fire_combat_damage_to_player_triggers -> StackItem.targets).
    let prev_target_pid = state.resolving_combat_damage_target_player_id;
    state.resolving_combat_damage_target_player_id = target_player;

    dispatch_event(state, crate::cards::Event::DealsDamage, source_id)?;

    // Reset both slots after dispatch so they don't leak into
    // unrelated resolves (mirrors the ``apply_damage_pass`` reset).
    state.resolving_combat_damage_amount = 0;
    state.resolving_combat_damage_target_player_id = prev_target_pid;

    Ok(())
}

/// **Slice 1 — push one triggered ability onto the stack.** Adds a
/// `StackItem::new_trigger` carrying the source object's id, the
/// trigger's controller, and the payload effect. Bumps the
/// once-per-turn ledger when the ability is gated.
///
/// Slice 1 simplifications (deferred to later slices):
/// - **No target selection.** Effects that need a target
///   (DealDamage with AnyPlayer / AnyCreature) fall back on the
///   existing `apply_effect` resolution-time targeting (defending
///   player / first defender bf object). Real MTG picks targets at
///   push time; deferred to Slice 4+ along with the rest of the
///   target-on-cast plumbing.
/// - **No fizzle check.** Targets are picked at resolve time, so
///   "no legal targets at push" is moot under Slice 1's deferred-
///   targeting shape. Slice 4+ adds the fizzle gate.
/// **Tier-C target-seating shared sub-substrate.** The deterministic
/// no-agent push-time target seat for a SAFE *targeted* triggered
/// ability — the single byte-equal-able first-legal rule shared by
/// every per-card lever-a consumer (Sunset Saboteur today; Exosuit
/// Savior next), mirrored exactly by the Python seat at
/// `actions.py:_put_triggered_ability_on_stack` (the
/// `controller.agent is None` branch).
///
/// **The rule (identical on both engines, by construction).** Scan
/// `state.players` in ascending seat index; within each player's
/// `battlefield` Vec in insertion order; return the id of the first
/// object that passes the effect's slot filter. This is exactly
/// Python's `Effect._enumerate_for_filter` order
/// (`base.py` — battlefield in list order then players in order);
/// `legal_targets[0]` IS this pick, which is what the campaign's
/// `_PickFirstLegalTargetAgent` shim family encodes. The
/// cross-engine harness keeps object ids + bf insertion order in
/// lockstep (`build_rust_state` registers in Python bf order), so
/// "smallest bf index passing the filter" names the SAME object on
/// both engines.
///
/// **Gated to lever-a effects ONLY.** Returns `None` for every
/// effect that is NOT a Tier-C target-seating lever variant —
/// including every one of the 20+ live UNTARGETED safe triggers
/// (`DrawCards` / `GainLife` / `Surveil` / `CreateToken` / …). For
/// those `push_triggered_ability` takes the unchanged
/// `new_trigger` path → byte-identical behavior, zero regression.
/// The per-card filter is the lever variant's own (matched here):
/// `PutPlusOneCounterOnOpponentsCreature` → first creature an
/// opponent of `controller` controls. (`new_trigger_with_target`
/// is the EXISTING constructor — CE20 Slice 4 / Equipment-equip;
/// no new `StackItem`/`TriggerPayload` field, see `state.rs`.)
///
/// **None semantics at the call site.** `count_min>=1` + no eligible
/// target ⇒ the trigger must NOT be pushed (Python fizzles,
/// `return False` — CR 603.3c). `count_min==0` + no eligible ⇒ push
/// with no target (the optional effect resolves as a no-op,
/// byte-equal with Python's `targets=()`). The call site encodes
/// this; the helper only answers "which object, if any".
fn first_legal_trigger_target(
    state: &GameState,
    source_id: u32,
    controller: u8,
    effect: &Effect,
) -> Option<u32> {
    match effect {
        // Sunset Saboteur (`ATTACKS`, source_is_self,
        // `PutPlusOneCounterOnOpponentsCreature`): slot filter
        // `IsCreature() & ControlledByOpponent()`. Opponent = any
        // player whose seat != the source-controller's seat (2-seat
        // model). Scan ascending seat then bf-Vec order; first
        // creature an opponent controls.
        Effect::PutPlusOneCounterOnOpponentsCreature { .. } => {
            for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue; // own creatures excluded by the filter
                }
                for obj in player.battlefield.iter() {
                    let printing =
                        state.registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        return Some(obj.id);
                    }
                }
            }
            None
        }
        // Exosuit Savior (`ENTERS`, source_is_self,
        // `ReturnTargetPermanentYouControlToOwnersHand`): slot filter
        // `ControlledByYou() & ~IsPlayer() & ~IsSource()`. Scan the
        // source-controller's OWN battlefield in Vec order; first
        // permanent that is not the source (players are never bf
        // objects, so `~IsPlayer()` is satisfied by construction).
        // `count_min=0` (optional) — `None` here means "no eligible
        // own permanent": `push_triggered_ability` then pushes via
        // the unchanged `new_trigger` path and the resolve arm
        // no-ops (byte-equal with Python's `targets=()` decline). It
        // is NOT in `lever_trigger_requires_target` so the trigger is
        // NEVER fizzle-skipped (contrast Sunset Saboteur).
        Effect::ReturnTargetPermanentYouControlToOwnersHand => {
            let cidx = controller as usize;
            for obj in state.players[cidx].battlefield.iter() {
                if obj.id == source_id {
                    continue; // ~IsSource()
                }
                return Some(obj.id);
            }
            None
        }
        // **G7 follow-up f8e3a2b1.** DepressurizeEffect as a TRIGGER
        // payload — the synthetic test pin shape at
        // `tests/test_parity_smoke.py::_build_rust_depressurize_state`
        // (no real EoE card emits DepressurizeEffect from a trigger
        // today; the lone EoE consumer Depressurize is an Instant
        // cast through `cast_spell_noncreature`, which seats
        // `bound_target` at Delta-5). When DepressurizeEffect IS
        // pushed as a triggered-ability payload (the
        // `_build_rust_depressurize_state` ETB shape), there is no
        // cast-time binder — without a seat here the resolve arm's
        // `current_resolve_payload_target_id`-driven CR 608.2b gate
        // would fizzle, regressing the 4 pre-existing synthetic
        // pins. Seat the SAME deterministic-first ANY-creature pick
        // the prior `Effect::DepressurizeEffect` resolve scan used
        // (`bf_iter_etb_sorted` find first Creature on EITHER bf —
        // the verified-non-controller-restricted shape pinned by
        // `test_rust_depressurize_targets_opponents_creature_too`).
        // Byte-equal to the pre-fix observable for every existing
        // test, and forward-correct for the (currently-non-existent)
        // future card that hypothetically triggers DepressurizeEffect.
        Effect::DepressurizeEffect { .. } => {
            crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing =
                        state.registry.lookup(obj.printing_id)?;
                    if printing
                        .card_types
                        .iter()
                        .any(|t| t == "Creature")
                    {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
        }
        // G7 lander-controller-mismatch closure (token-creation parity
        // 3x). Seed the SeedshipImpactEffect target at push time when
        // fired as an ETB trigger so the resolve arm reads the same
        // `current_resolve_payload_target_id` regardless of cast-spell
        // or ETB-trigger entry. Slot filter `IsArtifact() |
        // IsEnchantment()`, non-indestructible — mirrors
        // `state_ops_legal.rs::Effect::SeedshipImpactEffect.legal_targets`
        // exactly + Python's `legal_targets` at
        // `effects/bespoke.py:1227-1235`. Scan in
        // `bf_iter_etb_sorted` order across both bfs (the same
        // first-legal convention the cast-time bind uses + the
        // `_PickFirstLegalTargetAgent` bake-in convention). `count_min
        // >=1` (Seedship Impact requires a target) — so an empty
        // legal set fizzles the ETB trigger entirely (the matching
        // `lever_trigger_requires_target` entry below). Same
        // deterministic-first idiom the cast-time
        // `cast_spell_noncreature` Seedship Impact bind uses, so the
        // two paths agree byte-equal.
        Effect::SeedshipImpactEffect { .. } => {
            crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state
                        .registry
                        .printings
                        .get(obj.printing_id as usize)?;
                    let is_artifact = printing
                        .card_types
                        .iter()
                        .any(|t| t == "Artifact");
                    let is_enchantment = printing
                        .card_types
                        .iter()
                        .any(|t| t == "Enchantment");
                    if !(is_artifact || is_enchantment) {
                        return None;
                    }
                    if printing
                        .keywords
                        .iter()
                        .any(|k| k == "indestructible")
                    {
                        return None;
                    }
                    Some(obj.id)
                })
        }
        // **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
        // PutPoisonCountersOnTargetPlayer: the "target" is the player
        // who received combat damage. At push time (called from within
        // `dispatch_event` which is called from `dispatch_combat_damage_to_player`),
        // `state.resolving_combat_damage_target_player_id` holds the
        // target player's id (0 or 1). Seat it as the `target_id` in the
        // `TriggerPayload` so it survives until `resolve_top_of_stack`
        // swaps it into `current_resolve_payload_target_id`.
        //
        // Sentinel `u8::MAX` means "no active combat-damage target"
        // (synthetic-test push path or any push outside combat damage).
        // In that case return `None` (untargeted path → resolve arm
        // no-ops gracefully per the `u8::MAX` guard).
        //
        // Player ids (0, 1) are far below any object id (which starts
        // at 0 for the synthetic state but realistically at ≥4 for real
        // game states); using player id as the target_id is safe in the
        // 2-player EoE model. The resolve arm treats values ≤1 as player
        // ids and looks them up via `state.players.iter_mut().find(|p| p.id == pid)`.
        Effect::PutPoisonCountersOnTargetPlayer { .. } => {
            let pid = state.resolving_combat_damage_target_player_id;
            if pid == u8::MAX {
                None  // no active combat-damage target; resolve arm no-ops
            } else {
                Some(pid as u32)
            }
        }
        // **PPP Pin 2 (10th Rust slice).** PutPlusOneCounterOnTargetCreature:
        // bare "target creature" — no controller restriction. Scan
        // `bf_iter_etb_sorted` (smallest-bf-index order across both
        // players) for the first Creature (and when
        // `attacking_only=true`, the first attacking Creature). The
        // shared seat ensures both engines deterministically pick the
        // SAME object. `count_min=1` semantics (the matching
        // `lever_trigger_requires_target` entry below fizzles the
        // trigger when no legal target exists, CR 603.3c).
        Effect::PutPlusOneCounterOnTargetCreature {
            attacking_only,
            ..
        } => {
            crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state
                        .registry
                        .printings
                        .get(obj.printing_id as usize)?;
                    if !printing.card_types.iter().any(|t| t == "Creature") {
                        return None;
                    }
                    if *attacking_only && !state.attackers.contains(&obj.id) {
                        return None;
                    }
                    Some(obj.id)
                })
        }
        // Every other effect (all existing untargeted safe triggers,
        // and any non-lever targeted trigger the engine handles
        // elsewhere): NOT a target-seating consumer — the helper is
        // a no-op so `push_triggered_ability` takes the unchanged
        // `new_trigger` path.
        _ => None,
    }
}

/// True iff `effect` is a Tier-C target-seating lever-a variant whose
/// trigger MUST fizzle (NOT be pushed) when no legal target exists —
/// the `count_min>=1` cards. Sunset Saboteur's
/// `PutPlusOneCounterOnOpponentsCreature` is `count_min=1` (CR
/// 603.3c: a targeted ability with no legal target is removed from
/// the stack — Python's `_put_triggered_ability_on_stack` logs
/// "no legal targets; fizzles" and `return False`). The
/// `count_min==0` optional cards — Exosuit Savior's
/// `ReturnTargetPermanentYouControlToOwnersHand` — are
/// DELIBERATELY NOT in this set: with no eligible target they push
/// via the unchanged `new_trigger` path and the resolve arm no-ops
/// (byte-equal with Python's `targets=()` `Bounce.resolve`
/// early-return — the optional was simply declined, not fizzled).
fn lever_trigger_requires_target(effect: &Effect) -> bool {
    // SeedshipImpactEffect's Python `requires_target()` returns True;
    // an ETB trigger with no legal artifact/enchantment target must
    // fizzle (CR 603.3c) — byte-equal with Python's _put_triggered_
    // ability_on_stack's "no legal targets" return-False path. Added
    // alongside the cast-time bind so the ETB-trigger path and the
    // cast-spell path agree.
    matches!(
        effect,
        Effect::PutPlusOneCounterOnOpponentsCreature { .. }
            | Effect::SeedshipImpactEffect { .. }
            // PPP Pin 2: PutPlusOneCounterOnTargetCreature is a
            // count_min=1 targeted ability — with no legal target the
            // trigger fizzles per CR 603.3c. Byte-equal with Python's
            // `_put_triggered_ability_on_stack` "no legal targets"
            // return-False path.
            | Effect::PutPlusOneCounterOnTargetCreature { .. }
    )
}

pub(crate) fn push_triggered_ability(
    state: &mut GameState,
    source_id: u32,
    controller: u8,
    ability_index: u32,
    effect: Effect,
    once_per_turn: bool,
) -> PyResult<()> {
    // Tier-C target-seating: compute the deterministic first-legal
    // seat. `None` for every non-lever effect (every existing safe
    // trigger — preserves the byte-identical pre-substrate path).
    let tgt = first_legal_trigger_target(state, source_id, controller, &effect);
    if tgt.is_none() && lever_trigger_requires_target(&effect) {
        // `count_min>=1` + no eligible target: the trigger fizzles
        // and is NEVER pushed (CR 603.3c), byte-equal with Python's
        // `_put_triggered_ability_on_stack` "no legal targets;
        // fizzles" `return False`. The once-per-turn ledger is NOT
        // bumped (the ability didn't go on the stack — matches
        // Python, where `should_fire_trigger` already passed but the
        // fizzle path returns before the StackItem append; the ledger
        // bump is downstream of the push in both engines).
        return Ok(());
    }
    match tgt {
        Some(t) => state.stack.push(StackItem::new_trigger_with_target(
            source_id,
            controller,
            effect,
            ability_index,
            t,
        )),
        // Unchanged path: untargeted safe triggers AND the
        // `count_min==0` optional lever cards with no eligible target
        // (push with no seat → the optional effect resolves as a
        // no-op, byte-equal with Python's `targets=()`).
        None => state
            .stack
            .push(StackItem::new_trigger(source_id, controller, effect, ability_index)),
    }
    if once_per_turn {
        state
            .triggered_this_turn
            .insert((source_id, ability_index));
    }
    Ok(())
}

pub(crate) fn apply_effect_pub(
    state: &mut GameState,
    effect: &Effect,
    source_obj_id: u32,
) -> PyResult<()> {
    apply_effect(state, effect, source_obj_id)
}

/// **Controller-aware variant of `apply_effect_pub`.** Swaps
/// `state.current_resolve_source_controller` to `Some(controller)`
/// for the duration of the inner `apply_effect` call, restoring the
/// prior value (typically `None`) afterward.
///
/// Effect arms that need the source's controller (today
/// `Effect::DrawCards` per I59, `Effect::GainLife` per I60,
/// `Effect::DamageEachOpponent` per I63, and `Effect::DealDamage`
/// per I64 — see CR 119.5 / the "Controller of effect" convention
/// notes in those arms' comments) read the swapped field. Arms that
/// don't care continue to read `state.active_player` unchanged —
/// back-compat for every existing call site that hasn't been
/// migrated.
///
/// **Why this exists.** The pre-fix DrawCards arm read
/// `state.active_player` as the controller, which is wrong when the
/// resolving spell / triggered ability has a controller != the
/// active player (e.g. an opponent's instant resolves during your
/// turn and draws cards — pre-fix the spell would have drawn cards
/// for YOU, not its actual controller). Mirrors Python's
/// `DrawCards.resolve(state, controller_id, targets)` which takes
/// the source's controller explicitly.
///
/// **Pattern mirrored:** the CE20 Slice 1 bridge's `active_player`
/// swap (`state_ops_actions.rs`) and Slice 4's
/// `current_resolve_payload_target_id` swap. Cleaner than mutating
/// `active_player` because it doesn't touch any of the OTHER
/// `active_player`-keyed arms (PumpCreaturesYouControl, search-zone
/// resolvers, etc.) — those still mean exactly what they meant
/// before. Only arms that explicitly opt into reading
/// `current_resolve_source_controller` change behavior.
///
/// **Callers (post-I61):** every pending-decision dispatcher in
/// `state_ops_decisions.rs` (MayDo / IfYouDo / ChooseMode /
/// OptionalManaCost auto-resolve + submit). The in-tree stack
/// resolver branches still set the slot manually around their
/// `apply_effect` loops (avoids an extra function-call layer for the
/// hot stack-resolve path); the helper is the cleaner route for the
/// one-shot pending-decision path where readability beats nano-perf.
pub(crate) fn apply_effect_pub_with_controller(
    state: &mut GameState,
    effect: &Effect,
    source_obj_id: u32,
    controller: u8,
) -> PyResult<()> {
    let saved = state.current_resolve_source_controller;
    state.current_resolve_source_controller = Some(controller);
    let result = apply_effect(state, effect, source_obj_id);
    state.current_resolve_source_controller = saved;
    result
}

/// **Slice 2 — drain the deferred ETB dispatch queue.** Effect arms
/// that move permanents onto the battlefield (RS2-RS12 in
/// `docs/RUST_ETB_TRIGGER_AUDIT.md`) push the entered permanent's id
/// onto `state.pending_etb_dispatches` instead of dispatching directly.
/// The surrounding caller (every site that owns a "resolving slot" on
/// the stack — `resolve_top_of_stack`, `_pause`, plus the standalone
/// bf-entry primitives `play_land` / `cast_creature` / `move_object` /
/// `move_object_from_graveyard_to_battlefield` /
/// `register_and_inject_object` / `return_paired_exiles_for` /
/// `resolve_manifest_dread_to_battlefield`) calls this drainer AFTER
/// the resolving slot is removed from the stack so the trigger pushes
/// land on top of a clean (or at least non-resolving) stack.
///
/// **Drain order:** FIFO. The arm pushes ids in the order they
/// landed on the battlefield; the drainer dispatches in the same
/// order, mirroring Python's per-permanent
/// `put_into_battlefield -> _trigger_etb_abilities` loop.
///
/// **Re-entrancy:** drain takes the queue with `std::mem::take` to
/// avoid borrow conflicts; if a dispatch's effect re-fills the queue
/// (e.g. a chain of ETB triggers that themselves create permanents),
/// the new ids land in the queue and a subsequent drain picks them
/// up. Today the only way to re-fill mid-drain is via a triggered
/// ability that creates a token / reanimates — substrate-wise that's
/// supported and the drain loops until empty.
pub(crate) fn drain_pending_etb_dispatches(state: &mut GameState) -> PyResult<()> {
    // Loop until the queue stays empty across a drain (re-entrant
    // dispatches can re-fill mid-loop). Bounded by the number of
    // permanents that can enter in a single resolution — finite,
    // not unbounded.
    loop {
        let drained: Vec<u32> = std::mem::take(&mut state.pending_etb_dispatches);
        if drained.is_empty() {
            return Ok(());
        }
        for entered_id in drained {
            // **Round-82 (2026-05-20, Mechan Shieldmate closure).**
            // Bump the per-controller artifact-bf-entry counter
            // BEFORE the ENTERS dispatch, mirroring Python's
            // `actions.put_into_battlefield` inline increment
            // (`actions.py:3140`) which runs before
            // `_trigger_etb_abilities`. The drainer is the
            // centralized chokepoint every bf-entry route hits, so
            // one bump here catches play_land / cast_creature /
            // token-creation arms / resolve_top_of_stack / etc.
            // without per-site duplication. Counts artifact
            // creatures and artifact tokens too (the printing's
            // `card_types` Vec already carries both `"Artifact"`
            // and `"Creature"` for Mechan-shape printings).
            if let Some((controller_idx, printing_id)) =
                find_bf_controller_and_printing(state, entered_id)
            {
                if let Some(printing) = state.registry.lookup(printing_id) {
                    if printing.card_types.iter().any(|t| t == "Artifact") {
                        state.players[controller_idx]
                            .artifacts_entered_under_your_control_this_turn += 1;
                    }
                }
            }
            dispatch_event(state, Event::Enters, entered_id)?;
        }
    }
}

/// **Round-82 (2026-05-20, Mechan Shieldmate closure).** Locate a
/// just-entered battlefield permanent by id and return its
/// `(controller_player_idx, printing_id)`. Mirrors
/// `state_ops_combat::locate_battlefield_obj` (a sibling
/// `(player_idx, printing_id)` reader) but reads `obj.controller`
/// instead of the player slot index — when control of a permanent
/// differs from its owner (Aura under enchanted creature, control-
/// magic), the controller drives the per-player "entered under your
/// control this turn" counter. Returns `None` if the id isn't on a
/// battlefield (defensive — the drainer's queue only carries
/// just-pushed bf ids, so the lookup hits in the live path).
fn find_bf_controller_and_printing(
    state: &GameState,
    obj_id: u32,
) -> Option<(usize, u32)> {
    for player in state.players.iter() {
        for obj in player.battlefield.iter() {
            if obj.id == obj_id {
                return Some((obj.controller as usize, obj.printing_id));
            }
        }
    }
    None
}

/// **Slice 5 — DIES sibling of `drain_pending_etb_dispatches`.**
/// Mirror-shape drainer for `state.pending_dies_dispatches`. Called
/// by `apply_sbas` after the SBA destroy loop completes; iterates
/// the queued ids in FIFO order, dispatching `Event::Dies` for each.
/// Re-entrant fills (a DIES trigger that pushes a permanent onto bf
/// — Reaper-King-shape — re-enters into the ENTERS queue; a DIES
/// trigger that schedules ANOTHER death is rarer but the loop
/// handles it cleanly).
pub(crate) fn drain_pending_dies_dispatches(state: &mut GameState) -> PyResult<()> {
    loop {
        let drained: Vec<u32> = std::mem::take(&mut state.pending_dies_dispatches);
        if drained.is_empty() {
            return Ok(());
        }
        for dead_id in drained {
            dispatch_event(state, Event::Dies, dead_id)?;
        }
    }
}

/// **Slice 8 — LEAVES_BATTLEFIELD sibling of
/// `drain_pending_etb_dispatches` / `drain_pending_dies_dispatches`.**
/// Drains `state.pending_leaves_dispatches` in FIFO order,
/// dispatching `Event::LeavesBattlefield` for each queued id.
/// Called by every bf-exit primitive AFTER the primitive's mutation
/// completes (and AFTER any sibling DIES drain, so death-then-leaves
/// stack-push order matches Python's
/// `move_off_battlefield` sequence at `turn.py:280-289`).
///
/// Re-entrant fills: a LEAVES trigger whose resolution causes
/// another bf-exit (e.g. exile-on-LTB) re-enters into the queue;
/// the outer loop drains FIFO until the queue is empty.
pub(crate) fn drain_pending_leaves_dispatches(state: &mut GameState) -> PyResult<()> {
    loop {
        let drained: Vec<u32> = std::mem::take(&mut state.pending_leaves_dispatches);
        if drained.is_empty() {
            break;
        }
        for leaving_id in drained {
            dispatch_event(state, Event::LeavesBattlefield, leaving_id)?;
        }
    }
    // Round-55 token-cease-to-exist (CR 111.7 / 704.5d). The LEAVES
    // (and any preceding DIES) dispatch has now fired with the token
    // still in its destination zone (so its own death/leave triggers
    // saw it as an extra observer — Python's by-reference semantics
    // in `move_off_battlefield`). Now remove it so the final state
    // has the token in NO zone, byte-equal to Python (whose token
    // never lands in graveyard/exile/hand at all). A LEAVES trigger
    // that itself caused a token bf-exit re-filled the queue during
    // the loop above; this single drain after the loop catches it
    // (the drain is itself idempotent / re-entrant-safe).
    crate::state_ops_zones::drain_pending_token_cease(state);
    Ok(())
}

/// **Slice 14 — IS_SACRIFICED sibling of
/// `drain_pending_leaves_dispatches`.**
/// Drains `state.pending_is_sacrificed_dispatches` in FIFO order,
/// dispatching `Event::IsSacrificed` for each queued id. Called by
/// the surrounding resolver AFTER the resolving sacrifice spell's
/// stack pop, so the IS_SACRIFICED trigger lands on the now-empty
/// top of stack rather than above the still-present resolving spell
/// (which would cause the spell's epilogue `stack.pop()` to consume
/// the trigger).
///
/// Mirrors Python's `turn.sacrifice` semantics where
/// `dispatch_event(state, Event.IS_SACRIFICED, ...)` fires AFTER the
/// spell's resolve has returned (trigger lands at the next SBA +
/// priority cycle, original sacrifice spell no longer on the stack).
///
/// Re-entrant fills: an IS_SACRIFICED trigger whose resolution causes
/// another sacrifice re-enters into the queue; the outer loop drains
/// FIFO until the queue is empty.
pub(crate) fn drain_pending_is_sacrificed_dispatches(
    state: &mut GameState,
) -> PyResult<()> {
    loop {
        let drained: Vec<u32> =
            std::mem::take(&mut state.pending_is_sacrificed_dispatches);
        if drained.is_empty() {
            break;
        }
        for sac_id in drained {
            dispatch_event(state, Event::IsSacrificed, sac_id)?;
        }
    }
    Ok(())
}

/// Read effective power of a battlefield creature (printed +
/// UEOT power-bonus modifiers, sized off the SetBasePT / become-
/// land-creature overrides per CR 613.4 layer 7b/7c). Returns 0
/// for non-creatures or unknown ids.
///
/// Inlines `state_ops_combat::battlefield_creature_power` because
/// that helper is module-private to the combat slice (per the
/// per-slice file-isolation rule). Matches its semantics exactly:
/// SetBasePT > BecomeLandCreature > printed_base, plus
/// UEOT power_bonus addends. Used by the FightTargetCreatureWith
/// TargetCreature resolve arm to size each side's contribution to
/// the simultaneous-damage exchange.
///
/// **Inlining note (cross-file friction).** Same shape as the
/// `draw_card` and `dispatch_event` inlinings already in this
/// file — the per-slice rule forbids touching `state_ops_combat`
/// from this slice. A follow-up consolidation slice can collapse
/// these once that constraint relaxes; the public
/// `state.effective_power(obj_id)` PyMethod returns the same
/// value but is a `&self` getter not callable from `apply_effect`'s
/// `&mut state` borrow path without splitting the borrow.
fn read_battlefield_creature_power(state: &GameState, obj_id: u32) -> i16 {
    // Locate the object on either player's bf.
    let mut found: Option<(usize, &GameObject)> = None;
    for (idx, player) in state.players.iter().enumerate() {
        if let Some(obj) = player.battlefield.iter().find(|o| o.id == obj_id) {
            found = Some((idx, obj));
            break;
        }
    }
    let Some((player_idx, obj)) = found else {
        return 0;
    };
    let Some(printing) = state.registry.printings.get(obj.printing_id as usize) else {
        return 0;
    };
    let printed_base = printing.power.unwrap_or(0) as i32;
    let _ = player_idx; // silence unused (kept for future log/debug hooks)
    let base = obj
        .ueot_base_pt_override()
        .map(|(p, _)| p)
        .or_else(|| obj.ueot_becomes_land_creature_pt().map(|(p, _)| p))
        .unwrap_or(printed_base);
    let bonus = obj.ueot_power_bonus();
    let effective = base + bonus;
    effective.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// **Round-53 §2 fix — effective-power snapshot for the round-31
/// `Effect::DoubleTargetCreaturesPowerUEOT` (Mightform Harmonizer
/// landfall) resolve arm.**
///
/// Returns the target creature's CURRENT effective power as the
/// number to double, computed to MATCH Python's source-of-truth
/// `DoubleTargetCreaturesPowerUEOT.resolve` (`pump.py:234`), which
/// snapshots `mtg_bot.game.effective_power(state, obj)` =
/// `_compute_effective_pt` (game.py:2138).
///
/// **Why this exists / what it fixes.** The round-31 resolve arm
/// snapshotted via `read_battlefield_creature_power`, which
/// DELIBERATELY omits `obj.counters` (a pinned combat-side
/// asymmetry — see the comment at `state_ops_combat.rs:551-556`).
/// Python's `_compute_effective_pt` FOLDS IN the signed +1/+1 net at
/// game.py:2251 (`p += obj.plus_one_counters - obj.minus_one_counters`).
/// So on a target carrying +1/+1 counters the round-31 Rust arm
/// doubled BASE power while Python doubled EFFECTIVE power — a real
/// §2 silent mistranslation, latent only because the round-31/36
/// genuine Mightform probe staged a counter-free creature
/// (`docs/SEARCH_LIMITATIONS.md`, round-53 entry). On a 3/3 with
/// +2/+2 counters: pre-fix Rust doubled 3, Python doubled 5.
///
/// **The fix.** This helper composes
/// `base (UEOT-SetBasePT override / manland / printed)
///  + ueot_power_bonus()  + obj.counters`
/// where `obj.counters` (signed i32) is Rust's analogue of Python's
/// `plus_one_counters - minus_one_counters` (the SBA path at
/// `state_ops_sba.rs:200` uses exactly `base + obj.counters +
/// ueot_*_bonus()` — same composition, confirming the mapping). The
/// round-31 arm now reads through here, so BOTH engines double the
/// SAME effective-power snapshot.
///
/// **Documented residual (SEARCH_LIMITATIONS).** Layer-7c static
/// anthems and layer-7e P/T switches are NOT folded here (nor by the
/// combat-side reader, nor reachable by any bridge-correct Mightform
/// Harmonizer scenario — the card has no anthem/switch rider and the
/// genuine probe stages none). Same established combat-side
/// asymmetry; documented alongside the round-53 SEARCH_LIMITATIONS
/// entry.
fn read_doubling_power(state: &GameState, obj_id: u32) -> i32 {
    let mut found: Option<&GameObject> = None;
    for player in state.players.iter() {
        if let Some(obj) =
            player.battlefield.iter().find(|o| o.id == obj_id)
        {
            found = Some(obj);
            break;
        }
    }
    let Some(obj) = found else {
        return 0;
    };
    let Some(printing) =
        state.registry.printings.get(obj.printing_id as usize)
    else {
        return 0;
    };
    let printed_base = printing.power.unwrap_or(0) as i32;
    let base = obj
        .ueot_base_pt_override()
        .map(|(p, _)| p)
        .or_else(|| {
            obj.ueot_becomes_land_creature_pt().map(|(p, _)| p)
        })
        .unwrap_or(printed_base);
    base + obj.ueot_power_bonus() + obj.counters
}

/// **BLOCKED-RUST-EFFECT slice (round 54) — Famished Worldsire ETB
/// source-power read.**
///
/// Returns the source creature's CURRENT effective power = `X` in
/// "look at the top X cards of your library, where X is this
/// creature's power". Computed to MATCH Python's source-of-truth
/// `FamishedWorldsireETBEffect.resolve` (`effects/bespoke.py:116`),
/// which reads `mtg_bot.game.effective_power(state, source)` =
/// `_compute_effective_pt` (`game.py:2138`). Famished Worldsire's
/// Devour rider seeds +1/+1 counters as it enters, so the read MUST
/// fold in `obj.counters` — a base-power read would be a real §2
/// silent mistranslation (same class the round-53 Mightform fix
/// closed).
///
/// **Composition (per-slice file-isolation: re-derived here rather
/// than cross-calling `read_doubling_power`).**
/// `base (UEOT-SetBasePT override / manland / printed)
///  + ueot_power_bonus()  + obj.counters`
/// — `obj.counters` (signed i32) is Rust's analogue of Python's
/// `plus_one_counters - minus_one_counters` (`game.py:2251`); the SBA
/// path at `state_ops_sba.rs:200` uses the SAME composition,
/// confirming the mapping. Source not on a battlefield → 0 (matches
/// Python's `if source is None: return` short-circuit chained with
/// `if x <= 0: return`).
///
/// **Documented residual (SEARCH_LIMITATIONS, round-54 entry).**
/// Layer-7c static anthems and layer-7e P/T switches are NOT folded
/// here (same established combat-side / round-53 asymmetry); no
/// bridge-correct Famished Worldsire scenario reaches them (the card
/// has no anthem/switch rider and the genuine probe stages none).
fn read_famished_source_power(state: &GameState, obj_id: u32) -> i32 {
    let mut found: Option<&GameObject> = None;
    for player in state.players.iter() {
        if let Some(obj) =
            player.battlefield.iter().find(|o| o.id == obj_id)
        {
            found = Some(obj);
            break;
        }
    }
    let Some(obj) = found else {
        return 0;
    };
    let Some(printing) =
        state.registry.printings.get(obj.printing_id as usize)
    else {
        return 0;
    };
    let printed_base = printing.power.unwrap_or(0) as i32;
    let base = obj
        .ueot_base_pt_override()
        .map(|(p, _)| p)
        .or_else(|| {
            obj.ueot_becomes_land_creature_pt().map(|(p, _)| p)
        })
        .unwrap_or(printed_base);
    base + obj.ueot_power_bonus() + obj.counters
}

/// Push `amount` damage onto the battlefield creature's per-instance
/// `damage` marker (saturating add into i16). Routes through
/// `Arc::make_mut` on the containing battlefield Vec to preserve CoW.
/// Walks both players' battlefields; returns silently on unknown ids
/// (silent no-op convention, mirrors the public `state.deal_damage`
/// behavior).
///
/// Inlines `state_ops_damage::deal_damage` because that helper is
/// module-private (per the per-slice file-isolation rule). The two
/// implementations match: monotonic accumulation on the marker,
/// saturating into i16, no SBA-firing inline. Used by the
/// FightTargetCreatureWithTargetCreature resolve arm to apply both
/// sides of the simultaneous-damage exchange (CR 701.12).
fn apply_damage_marker(state: &mut GameState, obj_id: u32, amount: i16) {
    if amount <= 0 {
        return;
    }
    for player in state.players.iter_mut() {
        if player.battlefield.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    obj.damage = obj.damage.saturating_add(amount);
                    return;
                }
            }
        }
    }
}

/// **BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB.**
/// Destroy the source-controller's opponent's first battlefield
/// creature (the `defender = 1 - controller` bf[0] first-slice
/// deterministic pick, identical to the `DestroyTarget { AnyCreature }`
/// arm), then — iff that creature carried NO marked damage at the
/// moment of resolution (`obj.damage == 0`, the byte-equal Rust read
/// of Python's `not obj.damaged_this_turn` for the realistic
/// no-prior-damage ETB) — the victim's controller draws `draw_amount`
/// cards (CR 121.1 — the printed text gives the draw to the destroyed
/// creature's controller).
///
/// Mirrors the `DestroyTarget { AnyCreature }` resolve body verbatim
/// (controller resolution, indestructible-suppress CR 702.12, zone-
/// change reset CR 400.7, Void LTB nonland flag, LTB fan-out via
/// `pending_leaves_dispatches` + `queue_token_cease`) so the destroy
/// half stays byte-equal with the existing destroy substrate; the
/// only additions are the pre-destroy `was_undamaged` snapshot, the
/// victim-controller capture, and the post-destroy conditional draw
/// (inlined draw loop — same shape as `Effect::DrawCards`,
/// re-derived here per the per-slice file-isolation rule rather than
/// cross-calling the DrawCards arm). Empty defender bf / indestructible
/// target = silent fizzle, and the draw rider does NOT run in either
/// fizzle case (CR 608.2c — the "up to one" optional shape resolves
/// as a no-op).
fn destroy_first_defender_creature_then_draw_if_undamaged(
    state: &mut GameState,
    draw_amount: i32,
) {
    let controller = state
        .current_resolve_source_controller
        .map(|c| c as usize)
        .unwrap_or(state.active_player as usize);
    let defender = 1 - controller;
    let bf = &state.players[defender].battlefield;
    if bf.is_empty() {
        return; // fizzle silently — "up to one" no-op, no draw
    }
    let target_id = bf[0].id;
    // Snapshot the undamaged condition BEFORE any mutation: leaving
    // the battlefield resets `obj.damage`, so the read must happen
    // now (mirrors Python's `was_undamaged = not obj.damaged_this_turn`
    // captured before `destroy()` in DestroyTargetCreature.resolve).
    let was_undamaged = bf[0].damage == 0;
    // The victim's controller (defender in the 2-player first slice;
    // == owner). The draw goes here per the printed text, not to the
    // spell's controller. Mirrors Python's
    // `victim_controller_id = obj.controller_id`.
    let victim_controller = defender;
    // Indestructible check (CR 702.12): peek the target's printing
    // keywords before mutating. Suppress destroy AND draw silently if
    // present (Python's turn.destroy carries the same gate; the rider
    // only runs alongside an actual destruction).
    let target_printing_id = bf[0].printing_id;
    let is_indestructible = match state.registry.lookup(target_printing_id) {
        Some(printing) => printing
            .keywords
            .iter()
            .any(|k| k == "indestructible"),
        None => false,
    };
    if is_indestructible {
        return; // CR 702.12 — suppress silently, no draw
    }
    // Pop from defender's battlefield (CoW-safe).
    let bf_mut: &mut Vec<GameObject> =
        Arc::make_mut(&mut state.players[defender].battlefield);
    let pos = bf_mut
        .iter()
        .position(|o| o.id == target_id)
        .expect("target_id was bf[0] but not found on re-scan");
    let mut obj = bf_mut.remove(pos);
    // Zone-change reset (CR 400.7) — identical to the DestroyTarget
    // arm. summoning_sick deliberately NOT reset (Python's
    // turn.move_off_battlefield omits the clear — CR 302.1, see I35).
    obj.zone = ZONE_GRAVEYARD;
    obj.tapped = false;
    obj.damage = 0;
    // R77-CFI1: counters are NOT cleared on the dead object. Python's
    // `turn.move_off_battlefield` (turn.py:231-240) clears combat
    // marks (damage / tapped / attached_to) but deliberately leaves
    // `+1/+1` counters intact; the SBA destroy path
    // (`state_ops_sba.rs` ~:417) and the closed `DepressurizeEffect`
    // arm both follow that observable ("counters retained on the dead
    // object" — the established byte-equal contract). The prior stray
    // `obj.counters = 0;` here diverged from BOTH Python (SoT) and
    // Rust's own SBA destroy path; removed so a counter'd creature
    // destroyed via this hand-rolled arm carries its counters into the
    // graveyard byte-equal (R77-CFI1; same class as the closed
    // CFI-1 — see the FINDINGS NOTE in
    // tests/test_probe_cross_feature_interaction.py).
    obj.attached_to = None;
    obj.exiled_with_source = None;
    let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
    Arc::make_mut(&mut state.players[defender].graveyard).push(obj);
    // Round-29 Void LTB tracking: destroy is a battlefield exit.
    mark_left_battlefield_if_nonland(state, printing_id);
    // LTB fan-out — queue the leaving id; the drain runs in
    // `resolve_top_of_stack` AFTER the resolving slot pops (same
    // deferral shape as the DestroyTarget arm).
    state.pending_leaves_dispatches.push(leaving_obj_id);
    crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
    // Faller's Faithful draw rider: iff the target carried no marked
    // damage at resolve time, the victim's controller draws
    // `draw_amount` cards. Negative clamps to 0 (X-from-count zero
    // short-circuit convention). Inlined draw loop — same pattern as
    // `Effect::DrawCards`.
    let n = draw_amount.max(0) as usize;
    if was_undamaged && n > 0 {
        for _ in 0..n {
            let player = &mut state.players[victim_controller];
            let lib = Arc::make_mut(&mut player.library);
            if lib.is_empty() {
                // CR 704.5b — empty-library draw flags the player.
                player.attempted_draw_on_empty = true;
                break;
            }
            let mut drawn = lib.remove(0);
            drawn.zone = ZONE_HAND;
            Arc::make_mut(&mut player.hand).push(drawn);
        }
    }
}

/// **BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner target
/// pick.** Resolve the source's controller by scanning every tracked
/// zone for `source_obj_id` (identical to the bare
/// `Effect::PumpTargetCreatureUntilEOT` arm's source-controller
/// search — Perigee is its `granted_triggered_ability`-carrying
/// sibling, so the controller resolution must match exactly), then
/// return `(controller_idx, target_id)` for the FIRST creature on
/// that controller's battlefield whose id is NOT `source_obj_id`
/// (Perigee's "ANOTHER target creature you control" — CR 115.4
/// "another" excludes the source itself; mirrors Python's
/// `PumpTargetCreatureUntilEOT(exclude_self=True)` dropping
/// `state.resolving_source` from `legal_targets`).
///
/// Returns `None` when the source is untrackable (silent fizzle) or
/// the controller controls no OTHER creature (Perigee's ETB is a
/// non-`optional` "another target creature you control" — with no
/// legal target the trigger still resolves but does nothing, CR
/// 608.2b; observationally a silent no-op, matching Python's
/// `if not legal_targets: return`).
///
/// Creature-ness gates on the printing's `card_types` containing
/// "Creature" — the canonical creature-detection pattern shared with
/// the bare pump arm (state_ops_cast.rs:304 / state_ops_legal.rs:571).
fn perigee_pick_another_own_creature(
    state: &GameState,
    source_obj_id: u32,
) -> Option<(usize, u32)> {
    let mut src_controller: Option<u8> = None;
    'outer: for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
            &player.exile,
        ] {
            if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                src_controller = Some(obj.controller);
                break 'outer;
            }
        }
    }
    let controller = src_controller?;
    let cidx = controller as usize;
    let bf = &state.players[cidx].battlefield;
    let registry = &state.registry;
    let target_id = bf.iter().find_map(|obj| {
        if obj.id == source_obj_id {
            return None; // CR 115.4 "another" — exclude the source
        }
        let printing = registry.printings.get(obj.printing_id as usize)?;
        if printing.card_types.iter().any(|t| t == "Creature") {
            Some(obj.id)
        } else {
            None
        }
    })?;
    Some((cidx, target_id))
}

/// 20th Rust slice (VVV Pin 2 closure) — public re-export of the
/// inline sacrifice helper for the `state_ops_decisions` module's
/// auto-resolve decline branch. The decline path needs the same
/// "sacrifice source" routine the `Effect::SacrificeSelfUnlessTap
/// Permanent` arm uses for its empty-candidates fast path; exposing
/// the helper here avoids duplicating the LTB / IS_SACRIFICED /
/// token-cease triple-push across modules.
pub(crate) fn sacrifice_source_for_tap_decline(
    state: &mut GameState,
    controller_idx: usize,
    source_obj_id: u32,
) -> PyResult<()> {
    sacrifice_source_inline(state, controller_idx, source_obj_id)
}

/// 20th Rust slice (VVV Pin 2 closure) — shared sacrifice helper for
/// the `SacrificeSelfUnlessTapPermanent` Effect arm and the
/// `submit_tap_permanent_decision` decline branch.
///
/// Sacrifice the source permanent at `source_obj_id` on
/// `controller_idx`'s battlefield. Mirrors the inline sacrifice shape
/// of `Effect::Sacrifice { TargetFilter::AnyCreature }` at
/// `state_ops_resolve.rs:2532` — the same `Arc::make_mut` pop + GY
/// push + LTB / IS_SACRIFICED / token-cease queuing pattern, but for
/// a SPECIFIC source id rather than the first-bf-creature pick. The
/// `Effect::Sacrifice` arm uses `bf[0]` as the target; this helper
/// finds the source by id (the Aura-ETB-shape "sac this" target
/// binding, not the "sac a creature" arm).
///
/// **Caller contract.** The caller MUST have already verified the
/// source is on `state.players[controller_idx].battlefield` (the
/// `Effect::SacrificeSelfUnlessTapPermanent` arm's
/// `source_on_bf` precondition check); this helper unconditionally
/// drops the source via `Arc::make_mut` + `bf.remove(pos)`. If the
/// source is somehow missing at remove time (race with another
/// Effect arm), the helper falls through silently — defensive.
fn sacrifice_source_inline(
    state: &mut GameState,
    controller_idx: usize,
    source_obj_id: u32,
) -> PyResult<()> {
    let bf_mut: &mut Vec<GameObject> =
        Arc::make_mut(&mut state.players[controller_idx].battlefield);
    let Some(pos) = bf_mut.iter().position(|o| o.id == source_obj_id) else {
        // Defensive — caller's precondition would catch this, but a
        // missed precondition shouldn't panic. Silent partial: drop
        // the sacrifice request and continue (mirrors the "source LTB
        // during decision window" fallback shape in
        // `PendingDecision::ManifestDreadChoice`'s submit dispatcher).
        return Ok(());
    };
    let mut obj = bf_mut.remove(pos);
    // Stamp graveyard zone + reset per-object state. Mirrors the
    // `Effect::Sacrifice { AnyCreature }` arm's zone-stamp shape
    // verbatim (state_ops_resolve.rs:2585-2606).
    obj.zone = ZONE_GRAVEYARD;
    obj.tapped = false;
    // summoning_sick: NOT reset (CR 302.1 — bf-only state; see I35).
    obj.damage = 0;
    // counters: NOT cleared (R77-CFI1 byte-equal contract; see the
    // Effect::Sacrifice arm's note).
    obj.attached_to = None;
    obj.exiled_with_source = None;
    let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
    Arc::make_mut(&mut state.players[controller_idx].graveyard).push(obj);
    // Round-29 Void LTB tracking: sacrifice is a battlefield-exit.
    mark_left_battlefield_if_nonland(state, printing_id);
    // Slice 10 LTB fan-out + Round-55 token-cease + Slice 14
    // IS_SACRIFICED dispatch — same triple-push as the sibling
    // `Effect::Sacrifice` arm at state_ops_resolve.rs:2622-2642.
    state.pending_leaves_dispatches.push(leaving_obj_id);
    crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
    state
        .pending_is_sacrificed_dispatches
        .push(leaving_obj_id);
    Ok(())
}

fn apply_effect(
    state: &mut GameState,
    effect: &Effect,
    source_obj_id: u32,
) -> PyResult<()> {
    match effect {
        Effect::DealDamage { amount, target } => {
            // **Controller-of-effect fix (DealDamage defender).** The
            // defender is the OPPONENT of the SOURCE's controller, not
            // the opponent of `active_player`. CR 119.5 / 608.2: an
            // effect resolves under its source's controller; the
            // first-slice "target = the defending player" deterministic
            // pick must be relative to the resolving spell's
            // controller, not whichever player happens to have priority
            // at resolve time. Pre-fix Rust hardcoded
            // `defender = 1 - state.active_player`, which is wrong when
            // the resolving spell / triggered ability has a controller
            // != active_player. Example: on p0's turn, p1 casts a
            // Lightning Bolt — pre-fix the damage went to p1
            // (1 - active_player = 1 - 0 = 1), not p0. Same fix shape
            // as I59 (DrawCards) and I60 (GainLife); reads the same
            // `current_resolve_source_controller` transient slot, then
            // derives `defender = 1 - source_controller`.
            //
            // Mirrors Python's `DealDamageTargetPlayer` /
            // `DealDamageAnyTarget` deterministic-defender pick at
            // `effects/generic/damage.py`, which uses the source's
            // controller as the choosing player.
            let source_controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let defender: u8 = 1 - source_controller;
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
        // cards. Controller resolution: prefer the transient
        // `state.current_resolve_source_controller` (set by
        // `apply_effect_pub_with_controller` / by the stack resolver
        // around `apply_effect` calls — see the DrawCards
        // controller-of-effect fix), falling back to
        // `state.active_player` for back-compat with every direct
        // `apply_effect_pub` caller that hasn't migrated.
        //
        // **Why the fix.** Pre-fix this arm hardcoded
        // `state.active_player`, which is wrong when the resolving
        // spell / triggered ability has a controller != the active
        // player. Example: on player 0's turn, player 1 casts an
        // instant DrawCards-1 in response — pre-fix the card went
        // into PLAYER 0's hand (active_player), not player 1's
        // hand. Mirrors Python's `DrawCards.resolve(state,
        // controller_id, targets)` (effects/generic/draw_mill.py:213)
        // which receives the source's controller explicitly.
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
        // Empty-library handling (CR 704.5b — round-26 G10 closure).
        // If the library is exhausted before we've drawn `count` cards
        // we stop AND flip the player's `attempted_draw_on_empty`
        // cause-signal. The next `apply_sbas` pass transitions that
        // into the sticky `has_lost` flag; `losers()` ALSO reads the
        // cause-signal directly so the loss is observable BEFORE the
        // SBA pass runs. Mirrors Python's eager `p.has_lost = True`
        // in `turn.draw_card`.
        Effect::DrawCards { count } => {
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let n = (*count).max(0) as usize;
            for _ in 0..n {
                let player = &mut state.players[controller];
                let lib = Arc::make_mut(&mut player.library);
                if lib.is_empty() {
                    // CR 704.5b — empty-library draw flags the player.
                    player.attempted_draw_on_empty = true;
                    break;
                }
                // Post-CE3: top of library = lib[0].
                let mut obj = lib.remove(0);
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
        //
        // **Controller resolution (post-I59 follow-up — I60)**:
        // prefer the transient
        // `state.current_resolve_source_controller` (set by the
        // stack resolver around each `apply_effect` call), falling
        // back to `state.active_player` for back-compat with every
        // direct `apply_effect_pub` caller that hasn't migrated.
        // Same pattern as `Effect::DrawCards` above — GainLife
        // resolves to the SOURCE's controller per CR 119.5 /
        // 608.2, NOT necessarily the active player. Mirrors
        // Python's `GainLife.resolve(state, controller_id, ...)`
        // at `effects/generic/life.py:33`. The example failure
        // mode pre-fix: on p0's turn, p1's Germinating Wurm
        // (ETB → you gain 2 life) ETBs (perhaps because p1 cast
        // it on its own end-of-turn step that got skipped, or
        // because of a more-exotic timing), trigger fires with
        // controller=p1 — pre-fix p0 gained the 2 life, post-fix
        // p1 does.
        Effect::GainLife { amount } => {
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            state.players[controller].life += *amount;
            // Track life-gained-this-turn for ConditionalIfYouGainedLifeAtLeast.
            // Only positive amounts count — mirrors Python's gain_life() guard
            // (game.py:1360: "if amount <= 0: return") which skips the
            // `p.life_gained_this_turn += amount` increment when amount <= 0.
            if *amount > 0 {
                state.players[controller].life_gained_this_turn += *amount;
            }
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
                // **Controller resolution (I76 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 / I71 / I72 / I73 / I74 / I75 controller-
                // of-effect substrate).** CR 119.5 / 608.2: an effect
                // resolves under its source's controller.
                // ReturnToHand's first-slice AnyCreature shortcut
                // picks the SOURCE'S controller's opponent's bf[0],
                // where defender = 1 - controller. Sibling closure
                // to I70 ExileTarget / I75 DestroyTarget — identical
                // target selection path (bf[0] of defender),
                // identical LTB fan-out via
                // pending_leaves_dispatches. Differs from I70 / I75
                // only in destination zone (defender's HAND vs
                // exile / graveyard).
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts an Unsummon-shape instant — pre-
                // fix it self-bounced p1's own bf[0] back to p1's
                // hand (defender = 1 - active_player = 1 - 0 = 1).
                // Post-fix it correctly bounces p0's bf[0] back to
                // p0's hand (the source's controller's opponent —
                // standard Unsummon / Boomerang / Vapor Snag shape).
                //
                // **First-slice owner-vs-controller convention
                // preserved.** Target's controller == owner, so the
                // bounce destination is the defender's hand (where
                // the target's owner's hand lives). The controller
                // fix only changes which side that defender refers
                // to; the destination-is-defender's-hand semantics
                // are unchanged.
                //
                // **LTB fan-out preserved** — leaving id still
                // pushes onto `pending_leaves_dispatches` so the
                // drain runs in `resolve_top_of_stack` AFTER the
                // resolving slot pops.
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
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
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state; preserving the flag in non-battlefield
                // zones has no gameplay impact, and the re-ETB path
                // restamps `sick=true`). See I35.
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[defender].hand).push(obj);
                // Round-29 Void LTB tracking: bounce is a battlefield-
                // exit. Flag flips iff the bounced object is nonland.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        },
        // Round-10: Mill. Caller's controller puts the top `count`
        // cards of their library into their graveyard. "Top" is
        // lib[0] post-CE3 (mirrors zones convention used by
        // DrawCards above).
        //
        // **Controller resolution (I65 follow-up to the I59 / I60 /
        // I63 / I64 controller-of-effect substrate).** Prefer the
        // transient `state.current_resolve_source_controller` (set
        // by the stack resolver around each `apply_effect` call),
        // falling back to `state.active_player` for back-compat
        // with direct `apply_effect_pub` callers. Same pattern as
        // `Effect::DrawCards` / `Effect::GainLife` /
        // `Effect::DamageEachOpponent` / `Effect::DealDamage` —
        // CR 119.5 / 608.2: an effect resolves under its source's
        // controller. For Mill (self-mill), the milling player is
        // the SOURCE'S controller, not the active player.
        //
        // **Why the fix.** Pre-fix this arm hardcoded
        // `state.active_player`. Wrong when the resolving spell /
        // triggered ability has a controller != active_player.
        // Example: on p0's turn, p1 casts an instant Mill-2 self-
        // mill in response (e.g. a graveyard-fueling reactive
        // spell) — pre-fix the cards left p0's library (active_-
        // player), post-fix they leave p1's library (the spell's
        // controller). Mirrors Python's analogous self-mill
        // handling (drawn off `MillTargetPlayer` with target =
        // controller; the Rust `Effect::Mill` arm is the
        // controller-target shortcut).
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
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let n = (*count).max(0) as usize;
            for _ in 0..n {
                let player = &mut state.players[controller];
                let lib = Arc::make_mut(&mut player.library);
                if lib.is_empty() {
                    break; // empty library — silent stop
                }
                // Always remove index 0 — top after each prior remove.
                let mut obj = lib.remove(0);
                obj.zone = ZONE_GRAVEYARD;
                Arc::make_mut(&mut player.graveyard).push(obj);
            }
        }
        // MillTargetPlayer — Scrabbling Skullcrab pattern (CR 701.13b).
        // 2 cards in the EoE pool per round-20 measurement.
        //
        // **Distinct from Effect::Mill.** Mill (above) mills the
        // CONTROLLER's library — no target. MillTargetPlayer mills a
        // chosen target PLAYER's library. First-slice deterministic
        // choice: defender (= 1 - active_player), mirroring the
        // DealDamage / TapTarget / TargetOpponentLosesLife "the one
        // opponent" convention. Future multiplayer / archenemy work
        // needs an explicit choice callback.
        //
        // **Negative count raises** loudly (defense-in-depth; the
        // parse_effect bridge already checks at register time, but
        // synthetic / injected printings could bypass the parser).
        // **count=0 is a legal no-op** (mirrors GainLife / DrawCards /
        // Mill / EachOpponentLosesLife / LoseLife zero-count
        // conventions). **Fewer cards in library than `count`**: mill
        // all remaining cards (Python parity with
        // `if not target.library: break`).
        //
        // **Top-of-library = LAST element of the Vec** (round-1 zones
        // convention, mirrored by the existing Mill arm and library
        // primitives in `state_ops_library.rs`). Per-object state reset
        // on graveyard move (CR 400.7), mirroring DestroyAllCreatures /
        // DestroyTarget / ExileTarget reset shape.
        //
        // **Inlining note.** A `mill_target_player` primitive exists in
        // `state_ops_library.rs` (round-19); per the recurring per-slice
        // file-isolation pattern (E0624 cross-file `#[pymethods]`
        // friction) this arm inlines the equivalent pop-N + push-to-
        // graveyard logic. A follow-up consolidation slice can collapse
        // onto the shared primitive once cross-file private-fn calls
        // re-open.
        Effect::MillTargetPlayer { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "MillTargetPlayer count must be non-negative, got {count}"
                )));
            }
            // **Controller resolution (I77 follow-up to the
            // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
            // I70 / I71 / I72 / I73 / I74 / I75 / I76 controller-of-
            // effect substrate).** CR 119.5 / 608.2: an effect
            // resolves under its source's controller. MillTargetPlayer's
            // first-slice deterministic target picker chooses the
            // SOURCE'S controller's opponent (= the source's other
            // player), where target_idx = 1 - controller.
            //
            // **Why the fix.** Pre-fix this arm computed
            // `target_idx = 1 - state.active_player`. Wrong when the
            // resolving spell / triggered ability has a controller
            // != active_player. Example: on p0's turn, p1 casts a
            // Scrabbling-Skullcrab-shape "target player mills 2"
            // instant — pre-fix it self-milled p1's own library
            // (target_idx = 1 - active_player = 1 - 0 = 1).
            // Post-fix it correctly mills p0's library (the source's
            // controller's opponent — standard Mind Funeral / Tome
            // Scour shape).
            //
            // Sibling closure to I65 Mill (self-mill, where controller
            // mills their own library) — I65 routes to controller,
            // I77 routes to (1 - controller). Both close the
            // active_player-vs-source's-controller divergence; they
            // differ only in WHICH side mills (self vs target-player).
            //
            // (Real target-player choice plumbing remains deferred
            // — mirrors existing single-target first-slice
            // conventions.)
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let target_idx = 1 - controller;
            let library_mut = Arc::make_mut(&mut state.players[target_idx].library);
            let n_to_mill = (*count as usize).min(library_mut.len());
            if n_to_mill == 0 {
                // count=0 OR empty target library — silent no-op.
            } else {
                // Post-CE3: top of library = lib[0]. Remove from head
                // n_to_mill times.
                let mut milled: Vec<GameObject> = Vec::with_capacity(n_to_mill);
                for _ in 0..n_to_mill {
                    milled.push(library_mut.remove(0));
                }
                let gy_mut = Arc::make_mut(&mut state.players[target_idx].graveyard);
                for mut obj in milled {
                    obj.zone = ZONE_GRAVEYARD;
                    obj.tapped = false;
                    obj.summoning_sick = false;
                    obj.damage = 0;
                    obj.counters = 0;
                    obj.attached_to = None;
                    obj.exiled_with_source = None;
                    gy_mut.push(obj);
                }
            }
        }
        // BLOCKED-RUST-EFFECT slice (round 42) -- mill an opponent
        // a number of cards equal to the source-controller's current
        // life total. Sister of round-32 / 35 / 36 / 37 (damage
        // substrate), round-33 (counters substrate), and round-34
        // (surveil substrate) -- same dynamic-X-from-count plumbing,
        // NEW apply substrate (mill-target-player) AND NEW denominator
        // axis (source-controller's life total). Mirrors Python's
        // `MillTargetPlayer(use_controller_life=True)`
        // (`draw_mill.py:151` -- Space-Time Anomaly's instant
        // resolution).
        //
        // **Source-controller resolution.** Uses
        // `current_resolve_source_controller` (mirrors the existing
        // `Effect::MillTargetPlayer` arm's I77 post-fix pattern at
        // state_ops_resolve.rs:1199 — both mill arms route through
        // the same controller-of-effect substrate). Fallback to
        // `state.active_player` if unset (matches the existing
        // MillTargetPlayer arm's defensive fallback for trigger
        // shapes where the source-controller wasn't seeded).
        //
        // **Life-total denominator.** `state.players[controller].life`
        // is i32; `max(0, life)` clamps negative life totals
        // (arbitrary lose-life-past-zero / infect-style scenarios)
        // to zero — silent no-op, matching Python's
        // `n = max(0, state.player_by_id(controller_id).life)`
        // clamp at draw_mill.py:187.
        //
        // **Apply / fizzle.** Bit-equal to the `MillTargetPlayer`
        // arm's pop-N + push-to-graveyard logic: take from lib[0]
        // (post-CE3 top-of-library convention), reset per-object
        // state on graveyard move (CR 400.7), stop early on empty
        // library mid-mill (matches Python's `if not target.library:
        // break`). Target player = `1 - controller` (first-slice
        // deterministic single-opponent pick; mirrors the existing
        // MillTargetPlayer arm's defender-pick).
        //
        // **Inlining note.** Same E0624 cross-file `#[pymethods]`
        // friction as the existing MillTargetPlayer arm — re-inlines
        // the pop-N + push-to-graveyard logic instead of routing
        // through state_ops_library.rs's mill_target_player primitive.
        Effect::MillTargetPlayerEqualToYourLifeTotal => {
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            // Life-total denominator: clamp negative life to 0
            // (matches Python's `max(0, life)` clamp).
            let life = state.players[controller].life;
            let count = life.max(0) as usize;
            if count == 0 {
                // life <= 0 -> silent no-op (count = 0).
                return Ok(());
            }
            let target_idx = 1 - controller;
            let library_mut = Arc::make_mut(&mut state.players[target_idx].library);
            let n_to_mill = count.min(library_mut.len());
            if n_to_mill == 0 {
                // empty target library — silent no-op (mid-mill empty
                // is the same shape via the loop break below, but
                // pre-check saves an Arc::make_mut churn).
                return Ok(());
            }
            // Post-CE3: top of library = lib[0]. Remove from head
            // n_to_mill times. Mirrors the round-19 MillTargetPlayer
            // arm's pop-N shape bit-for-bit (only the count source
            // differs — dynamic from life vs static from variant
            // field).
            let mut milled: Vec<GameObject> = Vec::with_capacity(n_to_mill);
            for _ in 0..n_to_mill {
                milled.push(library_mut.remove(0));
            }
            let gy_mut = Arc::make_mut(&mut state.players[target_idx].graveyard);
            for mut obj in milled {
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                gy_mut.push(obj);
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
                // **Controller resolution (I70 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 controller-of-effect substrate).** Prefer
                // the transient `state.current_resolve_source_-
                // controller`, falling back to `state.active_-
                // player`. Same pattern as `Effect::DealDamage`
                // (I64) / `Effect::DealDamageAnyTarget` (I67) —
                // CR 119.5 / 608.2: an effect resolves under its
                // source's controller. The "AnyCreature" first-
                // slice target shortcut picks the FIRST creature
                // on the DEFENDING side — where defender = the
                // SOURCE'S controller's opponent.
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts an Exile-target-creature instant
                // — pre-fix it exiled p1's own first creature
                // (defender = 1 - active_player = 1 - 0 = 1),
                // self-exiling. Post-fix it correctly exiles p0's
                // first creature (the source's controller's
                // opponent).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
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
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[defender].exile).push(obj);
                // Round-29 Void LTB tracking: exile is a battlefield-
                // exit. Flag flips iff the exiled object is nonland.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
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
                // **Controller resolution (I71 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 controller-of-effect substrate).**
                // Same pattern as I70 ExileTarget — CR 119.5 /
                // 608.2: an effect resolves under its source's
                // controller. "AnyCreature" first-slice target
                // shortcut picks the FIRST object on the
                // DEFENDING side, where defender = the SOURCE'S
                // controller's opponent.
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts a Twiddle-shape tap instant —
                // pre-fix it tapped p1's own bf[0] (self-tap).
                // Post-fix it correctly taps p0's bf[0] (the
                // source's controller's opponent).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Round-58 BECOMES_TAPPED event bridge — tapper-
                // effect site. Python's TapTarget-family effects
                // (`effects/generic/tap.py`'s TapTarget /
                // TapAnotherTargetCreature etc., plus
                // `effects/combinators.py:163` and
                // `effects/bespoke.py:496`) all route the actual tap
                // through `turn.tap_permanent`, which fires
                // `Event.BECOMES_TAPPED` for the tapped target — and
                // ONLY when the target was untapped (Python's
                // `tap_permanent` no-ops on an already-tapped
                // permanent: no re-stamp, no event). Route through
                // the centralized chokepoint so the no-op-if-already-
                // tapped gate matches Python exactly (an already-
                // tapped target is a silent no-op with NO
                // BECOMES_TAPPED dispatch — byte-equal to Python's
                // early return).
                crate::state_ops_resolve::tap_permanent_and_dispatch(
                    state, target_id,
                )?;
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
                // **Controller resolution (I72 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 / I71 controller-of-effect substrate).**
                // Same pattern as I71 TapTarget — CR 119.5 /
                // 608.2: an effect resolves under its source's
                // controller. "AnyCreature" first-slice target
                // shortcut picks the FIRST object on the
                // DEFENDING side, where defender = the SOURCE'S
                // controller's opponent.
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts a Twiddle-shape untap instant —
                // pre-fix it untapped p1's own bf[0] (self-untap).
                // Post-fix it correctly untaps p0's bf[0] (the
                // source's controller's opponent).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
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
                // **Controller resolution (I66 follow-up to the I59 /
                // I60 / I63 / I64 / I65 controller-of-effect
                // substrate).** Prefer the transient
                // `state.current_resolve_source_controller` (set by
                // the stack resolver around each `apply_effect` call),
                // falling back to `state.active_player` for back-
                // compat with direct `apply_effect_pub` callers. Same
                // pattern as `Effect::DrawCards` / `Effect::GainLife`
                // / `Effect::DamageEachOpponent` / `Effect::DealDamage`
                // / `Effect::Mill` — CR 119.5 / 608.2: an effect
                // resolves under its source's controller. For
                // Sacrifice (the "you sacrifice" arm), the
                // sacrificing player is the SOURCE'S controller, not
                // the active player.
                //
                // **Why the fix.** Pre-fix this arm hardcoded
                // `state.active_player`. Wrong when the resolving
                // spell / triggered ability has a controller !=
                // active_player. Example: on p0's turn, p1 casts an
                // instant Altar's-Reap-shape spell ("sacrifice a
                // creature, draw two cards") — pre-fix the spell
                // sacrificed p0's first battlefield permanent
                // (active_player), post-fix it correctly sacrifices
                // p1's (the spell's controller).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let bf = &state.players[controller].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Pop from sacrificing player's battlefield. Use
                // Arc::make_mut for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[controller].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was bf[0] but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                // Stamp graveyard zone + reset per-object state. First-
                // slice: target's controller == owner, so push onto
                // sacrificing player's graveyard (where the sacrificed
                // permanent's owner lives).
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object. A sacrifice is a death that routes through
                // Python's `turn.move_off_battlefield` (turn.py:538
                // docstring: "sacrifice ... calls
                // move_off_battlefield directly"), which clears combat
                // marks but deliberately leaves `+1/+1` counters
                // intact — the SAME byte-equal contract the SBA
                // destroy path / the closed `DepressurizeEffect` arm
                // follow. Prior stray `obj.counters = 0;` here
                // diverged from the Python SoT for a counter'd
                // sacrificed permanent; removed (R77-CFI1; audited
                // sibling of the destroy arms, same class as the
                // closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[controller].graveyard).push(obj);
                // Round-29 Void LTB tracking: sacrifice is a
                // battlefield-exit. Flag flips iff sac'd object is nonland.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            // Slice 14 (TTT Pin 1): defer IS_SACRIFICED dispatch via
            // `pending_is_sacrificed_dispatches`. Mirrors the LEAVES
            // sibling push above. The direct `dispatch_event` call
            // this replaced fired DURING the sacrifice spell's
            // resolve while the spell was still top-of-stack; the
            // dispatcher pushed the IS_SACRIFICED trigger above the
            // resolving spell and the spell's epilogue `stack.pop()`
            // consumed the trigger before it could resolve. The
            // queued id is drained later by
            // `drain_pending_is_sacrificed_dispatches` AFTER the
            // resolving slot is popped — mirrors Python's
            // `turn.sacrifice` where the IS_SACRIFICED dispatch fires
            // AFTER the spell's resolve has returned.
            state
                .pending_is_sacrificed_dispatches
                .push(leaving_obj_id);
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
                // **Controller resolution (I73 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 / I71 / I72 controller-of-effect
                // substrate).** Same pattern as I71 TapTarget /
                // I72 UntapTarget — CR 119.5 / 608.2: an effect
                // resolves under its source's controller.
                // "AnyCreature" first-slice target shortcut picks
                // the FIRST object on the DEFENDING side, where
                // defender = the SOURCE'S controller's opponent.
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts a +1/+1-counter instant — pre-fix
                // it self-buffed p1's own bf[0]; post-fix it
                // correctly places the counter on p0's bf[0]
                // (the source's controller's opponent, mirroring
                // the bounce / exile / tap / untap defender
                // convention).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
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
                // Round-16 RR Pin 1: dispatch COUNTERS_ADDED after the
                // counter mutation. Subject = the object that received
                // the counters (target_id). Mirrors Python's
                // ``fire_counters_added_to_your_creature_triggers`` call
                // in ``game.add_plus_one_counters``. Only fire when delta
                // is positive (adding counters, not removing).
                if delta > 0 {
                    dispatch_event(
                        state,
                        crate::cards::Event::CountersAdded,
                        target_id,
                    )?;
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
                // **Controller resolution (I74 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 / I71 / I72 / I73 controller-of-effect
                // substrate).** CR 119.5 / 608.2: an effect
                // resolves under its source's controller. Mind-Rot-
                // shape "target player discards a card" first-slice
                // target shortcut picks the SOURCE'S controller's
                // opponent (defender), where defender = 1 -
                // controller.
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts a Mind-Rot-shape instant — pre-
                // fix it self-milled p1's own hand[0]; post-fix
                // it correctly discards p0's hand[0] (the source's
                // controller's opponent — matching the convention
                // that AnyPlayer-on-discard means "target opponent
                // discards", a first-slice approximation of the
                // real "target player choice" surface).
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
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
                obj.attached_to = None;
                obj.exiled_with_source = None;
                Arc::make_mut(&mut state.players[defender].graveyard).push(obj);
                // **MMM Pin 2 (11th Rust slice).** Dispatch
                // `Event::DiscardsCard` for the discarding player AFTER
                // the card has been moved to graveyard. Mirrors
                // Python's IS_DISCARD dispatch shape: subject is the
                // discarding player (`PLAYER_SUBJECT_OFFSET +
                // defender` sentinel — same convention as
                // YOU_ATTACK / BEGINNING_OF_END_STEP). Entropic
                // Battlecruiser's "Whenever an opponent discards a
                // card, they lose 3 life" trigger fires here.
                let discarding_player_subject =
                    crate::cards::PLAYER_SUBJECT_OFFSET + defender as u32;
                dispatch_event(
                    state,
                    crate::cards::Event::DiscardsCard,
                    discarding_player_subject,
                )?;
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
        // Round-17: DestroyTarget. CR 701.7 — a destroyed permanent is
        // put into its owner's graveyard.
        //
        // Targeting (mirrors DealDamage / ReturnToHand / ExileTarget /
        // TapTarget):
        //   - AnyCreature -> first object on the DEFENDING player's
        //     battlefield (smallest index). No creature-type filter.
        //     If the defender's battlefield is empty, the spell
        //     fizzles silently.
        //   - AnyPlayer -> semantically nonsense for destroy (you can't
        //     destroy a player). Raise ValueError loudly, matching the
        //     ReturnToHand / ExileTarget / TapTarget / UntapTarget /
        //     Sacrifice / AddCounters AnyPlayer shape.
        //
        // **Asymmetric vs Sacrifice.** Sacrifice picks the ACTIVE
        // player's own battlefield; DestroyTarget picks the DEFENDING
        // player's battlefield. Diverges from the "you sacrifice your
        // own permanent" semantics — destroy is the standard removal
        // shape (Murder / Doom Blade / Naturalize) targeting an
        // opponent's permanent.
        //
        // **Indestructible is honored** (CR 702.12): if the target's
        // printing has the ``indestructible`` keyword, the destroy is
        // suppressed silently — Ok(()) with no permanent movement.
        // Mirrors the round-13 SBA indestructible logic. Effect still
        // ran (no error); the stack still pops, the source still
        // moves post-resolve.
        //
        // **Object-state reset on zone change** (CR 400.7) — tapped /
        // summoning_sick / damage / counters reset on the way to
        // graveyard, mirroring the ReturnToHand / ExileTarget /
        // Sacrifice reset convention.
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation, we inline the minimal Arc::make_mut pop-from-
        // battlefield + push-to-graveyard here, mirroring how
        // Sacrifice / ExileTarget / ReturnToHand inline their
        // primitives for the same E0624 cross-file reason.
        Effect::DestroyTarget { target } => match target {
            TargetFilter::AnyPlayer => {
                return Err(PyValueError::new_err(
                    "DestroyTarget with AnyPlayer target is semantically nonsense (cannot destroy a player)",
                ));
            }
            TargetFilter::AnyCreature => {
                // **Controller resolution (I75 follow-up to the
                // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 /
                // I69 / I70 / I71 / I72 / I73 / I74 controller-of-
                // effect substrate).** CR 119.5 / 608.2: an effect
                // resolves under its source's controller.
                // DestroyTarget's first-slice AnyCreature shortcut
                // picks the SOURCE'S controller's opponent's bf[0],
                // where defender = 1 - controller. Sibling closure
                // to I70 ExileTarget — identical target selection
                // path (bf[0] of defender), identical indestructible-
                // suppress gate, identical LTB fan-out via
                // pending_leaves_dispatches. Differs from I70 only
                // in destination zone (graveyard vs exile).
                //
                // **Why the fix.** Pre-fix this arm computed
                // `defender = 1 - state.active_player`. Wrong when
                // the resolving spell / triggered ability has a
                // controller != active_player. Example: on p0's
                // turn, p1 casts a Doom-Blade-shape instant —
                // pre-fix it self-destroyed p1's own bf[0]
                // (defender = 1 - active_player = 1 - 0 = 1).
                // Post-fix it correctly destroys p0's bf[0] (the
                // source's controller's opponent — the standard
                // Murder / Doom Blade / Naturalize removal shape).
                //
                // **Indestructible gate preserved** — the keyword
                // check runs on the resolved defender's bf[0] and
                // still short-circuits silently per CR 702.12.
                //
                // **LTB fan-out preserved** — the leaving id still
                // pushes onto `pending_leaves_dispatches` so the
                // drain runs in `resolve_top_of_stack` AFTER the
                // resolving slot pops, mirroring the
                // pending_etb_dispatches deferral shape.
                let controller = state
                    .current_resolve_source_controller
                    .map(|c| c as usize)
                    .unwrap_or(state.active_player as usize);
                let defender = 1 - controller;
                let bf = &state.players[defender].battlefield;
                if bf.is_empty() {
                    return Ok(()); // fizzle silently
                }
                let target_id = bf[0].id;
                // Indestructible check: peek the target's printing's
                // keywords before mutating anything. Suppress the
                // destroy silently if "indestructible" is present.
                let target_printing_id = bf[0].printing_id;
                let is_indestructible = match state.registry.lookup(target_printing_id) {
                    Some(printing) => printing
                        .keywords
                        .iter()
                        .any(|k| k == "indestructible"),
                    None => false,
                };
                if is_indestructible {
                    return Ok(()); // CR 702.12 — suppress silently
                }
                // Pop from defender's battlefield. Use Arc::make_mut
                // for CoW safety.
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was bf[0] but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                // Stamp graveyard zone + reset per-object state
                // (CR 400.7). First-slice: target's controller ==
                // owner, so push onto defender's graveyard (where
                // the destroyed permanent's owner lives).
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object. Python's `turn.move_off_battlefield`
                // (turn.py:231-240) clears combat marks but
                // deliberately leaves `+1/+1` counters intact; the SBA
                // destroy path (`state_ops_sba.rs` ~:417) and the
                // closed `DepressurizeEffect` arm both follow that
                // observable ("counters retained on the dead object" —
                // the established byte-equal contract). The prior stray
                // `obj.counters = 0;` here diverged from BOTH Python
                // (SoT) and Rust's own SBA destroy path; removed
                // (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[defender].graveyard).push(obj);
                // Round-29 Void LTB tracking: destroy is a battlefield-
                // exit. Flag flips iff destroyed object is nonland.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        },
        // BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB
        // (~14,108 17lands decks). "Destroy up to one OTHER target
        // creature. If that creature wasn't dealt damage this turn,
        // its controller draws `draw_amount` cards." Structural
        // sibling of `DestroyTarget { AnyCreature }` — same defender
        // bf[0] first-slice pick + indestructible gate + LTB fan-out
        // — plus the pre-destroy undamaged snapshot + post-destroy
        // conditional victim-controller draw. Logic lives in the
        // module-private `destroy_first_defender_creature_then_draw_
        // if_undamaged` helper (per the per-slice file-isolation
        // rule). Mirrors Python's
        // `DestroyTargetCreature(draw_if_undamaged_amount=draw_amount,
        // optional=True, exclude_self=True)`
        // (`effects/generic/destroy.py:94`).
        Effect::DestroyTargetCreatureThenDrawIfUndamaged { draw_amount } => {
            destroy_first_defender_creature_then_draw_if_undamaged(
                state, *draw_amount,
            );
        }
        // Round-17: DamageEachOpponent. CR 701.4 + AOE pattern — the
        // controller deals ``amount`` damage to each of their
        // opponents.
        //
        // **Two-player engine: damages the source-controller's
        // opponent.** No explicit target — the controller is the
        // SOURCE's controller (per CR 119.5 / 608.2; mirrors
        // Python's `DealDamageEachOpponent.resolve(state,
        // controller_id, targets)` at
        // `effects/generic/damage.py:152` which iterates
        // `state.players` and skips `p.id == controller_id`). The
        // one opponent is `1 - controller`. Mirrors the I59/I60
        // controller-of-effect pattern: prefer the transient
        // `state.current_resolve_source_controller` (set by the
        // stack resolver around each `apply_effect` call), falling
        // back to `state.active_player` for back-compat with direct
        // `apply_effect_pub` callers.
        //
        // **Why the fix (I62).** Pre-fix this arm hardcoded
        // `1 - state.active_player` as the defender. Wrong when the
        // resolving spell / triggered ability has a controller !=
        // active_player. Example: on p0's turn, p1's creature has
        // an "ETB → deal 2 damage to each opponent" trigger (e.g.
        // a flicker-ETB scenario where the creature blinks back
        // during p0's turn). Pre-fix the trigger damaged p1
        // (active_player=0's opponent = p1), self-damaging the
        // trigger's controller. Post-fix the trigger correctly
        // damages p0 (the SOURCE's opponent).
        //
        // **Negative ``amount`` raises** loudly (mirrors round-3
        // DealDamage's raise-on-negative shape; "deal -2 damage" is
        // schema nonsense). amount=0 is a legitimate no-op success
        // (mirrors GainLife/DrawCards/Mill zero-count conventions).
        Effect::DamageEachOpponent { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DamageEachOpponent amount must be non-negative, got {amount}"
                )));
            }
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender = 1 - controller;
            state.players[defender].life -= *amount;
        }
        // Round-18: DamageEachCreature. CR 701.4 — Pyroclasm pattern:
        // mark ``damage += amount`` on every creature on every player's
        // battlefield.
        //
        // **Symmetric AOE: applies to BOTH players' battlefields.**
        // Unlike DealDamage / DestroyTarget / TapTarget (which target
        // the defender) and Sacrifice (which targets the active player),
        // DamageEachCreature is implicit-target and hits every creature
        // regardless of controller. No target filter — the variant
        // doesn't carry a TargetFilter for that reason.
        //
        // **Creature-only filter via printing's card_types.** Lands /
        // artifacts / enchantments / planeswalkers on the battlefield
        // are not touched — only objects whose printing has "Creature"
        // in card_types take the damage mark. The check goes through
        // ``state.registry.lookup(obj.printing_id)``; no fallback if
        // lookup fails (treat as non-creature, skip silently — matches
        // the "lost track of source" failure mode elsewhere).
        //
        // **Indestructible is honored at SBA time, NOT here.** This
        // slice only marks damage; SBAs decide who dies. So we do NOT
        // filter on indestructible — write damage to every creature,
        // round-13 SBA will suppress destruction for indestructible
        // ones with lethal damage. Mirrors the round-3 DealDamage shape
        // where lethal damage is also written, then SBAs reap.
        //
        // **Negative ``amount`` raises** loudly (mirrors round-3
        // DealDamage and round-17 DamageEachOpponent raise-on-negative
        // shapes). amount=0 is a legitimate no-op success (mirrors
        // GainLife / DrawCards / Mill / DamageEachOpponent zero-count
        // conventions).
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation we inline the minimal ``Arc::make_mut`` +
        // ``saturating_add`` here on each player's battlefield,
        // mirroring how round-3 DealDamage / round-15 AddCounters
        // inline their primitives for the same E0624 cross-file reason.
        //
        // **bf-ETB iteration (I33 closure).** Visit order matches
        // Python's centralized `state.battlefield` walk via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper. Outcome is
        // order-invariant for damage marking (each eligible creature
        // gets `+amount` exactly once via the saturating_add), so the
        // observable post-state is identical pre- and post-closure on
        // every existing fixture. The change is pre-emptive: visit
        // order becomes load-bearing when the future trigger emitter
        // (CE8 substrate) walks the same victims to push
        // damage-trigger / death-trigger stack items — at that point
        // the relative emission order must mirror Python's centralized
        // walk for cross-engine smoke parity. See I33 in
        // `docs/RUST_PARITY_GAPS.md`.
        Effect::DamageEachCreature { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DamageEachCreature amount must be non-negative, got {amount}"
                )));
            }
            let amount_i16 = (*amount).min(i16::MAX as i32) as i16;
            // Collect (controller_idx, obj_id) for every Creature on
            // either bf in bf-ETB-timestamp order, so the per-player
            // mutating pass below applies damage in the same order
            // Python's centralized `state.battlefield` walk emits.
            let victims: Vec<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .filter_map(|obj| {
                    let printing = state.registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some((obj.controller as usize, obj.id))
                    } else {
                        None
                    }
                })
                .collect();
            if victims.is_empty() {
                return Ok(());
            }
            for (pidx, target_id) in victims {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
        }
        // Round-19: CreateToken. CR 701.6 — Servo/Soldier maker
        // pattern. The active player (= controller of the resolving
        // spell, first-slice convention mirrored from DealDamage /
        // DrawCards / GainLife) gets ``count`` fresh GameObjects on
        // their battlefield, all referencing ``token_printing_id``.
        //
        // **Each token enters with summoning sickness** (CR 302.1 —
        // fresh permanents have summoning sickness), untapped, no
        // damage, no counters, ``zone = ZONE_BATTLEFIELD``. Mirrors
        // the round-7 post-resolve creature-on-bf stamp shape (also
        // ``summoning_sick = true`` there) and the round-15
        // ``move_object_from_graveyard_to_battlefield`` reanimation
        // primitive.
        //
        // **count=0 is a legal no-op** — mirrors GainLife / DrawCards
        // / Mill / DamageEach* zero-count conventions. **count<0
        // raises** loudly, mirrors ScryN / DealDamage /
        // DamageEachOpponent / DamageEachCreature raise-on-negative.
        //
        // **Unknown ``token_printing_id``** (not present in the
        // registry) raises loudly — schema validation seam, the
        // printing must exist before you can make tokens of it.
        // Mirrors the registry-bounds check in ``load_deck``.
        //
        // **Fresh-id allocation (R76-1)**: draws from the monotonic
        // ``GameState.next_object_id`` counter via
        // ``crate::state::reserve_object_id`` — CR 111.7 (each object
        // gets a NEW unique id; a ceased object's id is never
        // reused). This mirrors Python's per-object
        // ``self.next_object_id += 1`` (game.py:2787-2798) byte-for-
        // byte. (Pre-R76-1 this arm re-scanned every live zone for
        // ``max_id + 1`` — non-monotonic: a token that had ceased to
        // exist no longer contributed to ``max_id``, so a later mint
        // recycled the freed id and diverged from the Python SoT.)
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation we build the GameObjects + push to the active
        // player's battlefield via ``Arc::make_mut`` directly, same
        // shape as ``load_deck`` (which lives in a sibling file we
        // can't touch from this slice).
        Effect::CreateToken {
            count,
            token_printing_id,
        } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "CreateToken count must be non-negative, got {count}"
                )));
            }
            // Registry-bounds check up front so an unknown printing_id
            // surfaces before any state mutation (mirrors load_deck's
            // whole-or-nothing validation contract).
            let token_pid = *token_printing_id;
            if state.registry.lookup(token_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "token_printing_id {token_pid} not in registry"
                )));
            }
            let n = *count as usize;
            if n == 0 {
                return Ok(()); // legal no-op
            }
            // **Controller resolution (I81 follow-up to the
            // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
            // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77 / I78 /
            // I79 / I80 controller-of-effect substrate).**
            // CR 119.5 / 608.2 / 111.4: a token is created under the
            // controller of the spell or ability that creates it —
            // which is the SOURCE's controller, not necessarily the
            // active player.
            //
            // **Why the fix.** Pre-fix this arm computed
            // `active = state.active_player`, used both for the
            // token's `controller` field and for which player's
            // battlefield receives the push. Wrong when the resolving
            // spell / triggered ability has a controller !=
            // active_player. Example: on p0's turn, p1 casts a
            // Servo-Schematic-shape instant — pre-fix the tokens
            // entered under p0's control on p0's battlefield
            // (active_player=0); post-fix they correctly enter under
            // p1's control on p1's battlefield. Mirrors Python's
            // `CreateToken.resolve(state, controller_id, targets)`
            // at `effects/generic/tokens.py:202`, which builds the
            // `recipients` list from `[controller_id]` (or every
            // living player when `each_player=True`).
            //
            // Same transient-slot lookup preamble as I59-I80; only
            // the post-resolve consumer differs (the controller
            // value is used for the token's `controller` byte AND
            // for picking the recipient battlefield Vec).
            //
            // Each-player and other token shapes (CreateTokenAndAttachSelf,
            // CreateTokenPerMulticoloredPermanent,
            // CreateTokenCopyOfTargetCreatureYouControl, the Moonlit
            // Meditation replacement, CR 614 token-creation
            // replacements like Exalted Sunborn / Anointed Procession,
            // the end-of-combat-sacrifice flag, the `tapped`/`attacking`
            // flag bundle) are not yet plumbed in this Rust arm — the
            // first-slice surface here is the basic "create N tokens
            // of `token_printing_id` under the controller" shape only.
            // Those richer surfaces still resolve through the Python
            // path; this fix only addresses the active_player-vs-
            // source's-controller divergence for the basic shape.
            let active = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);

            // **Aura-attachment Phase 4 S3 closure (2026-05-21) —
            // Moonlit Meditation replacement.** Per CR 614, if the
            // controller has an attached Aura with the
            // `aura_replaces_first_tokens_with_copies` rider AND this
            // is the FIRST token-creation batch of this turn for that
            // player, the tokens are created as copies of the Aura's
            // `attached_to` host instead of the requested printing,
            // with the Token supertype added. Mirrors Python's
            // `_find_moonlit_replacement_source` + `_dc.replace(base,
            // supertypes=(*base, "Token"))` shim at
            // `effects/generic/tokens.py:217-247`.
            //
            // **Substitution semantics.**
            //   * `token_pid` is swapped to the host's `printing_id`.
            //     The freshly-minted GameObjects then carry the host's
            //     printing — same name, P/T, keywords, abilities — but
            //     with the per-object Token-supertype override flag
            //     set so every `is_token` read site (token-cease
            //     scrub, "is a token" predicates) treats them as
            //     tokens. This avoids minting a fresh runtime
            //     `printing_id` for the host-with-Token-supertype copy
            //     and matches Python's in-memory `_dc.replace`
            //     supertype-injection shim in spirit (Python returns a
            //     fresh CardPrinting instance; Rust keeps the host's
            //     pid + per-object override).
            //   * `moonlit_override` records whether the Token
            //     supertype must be injected on the substituted token.
            //     `false` when no Moonlit-shape Aura is attached, or
            //     when this player has already created tokens this
            //     turn (the per-player `created_tokens_this_turn`
            //     flag), or when the host's printing already carries
            //     "Token" in its supertypes (a Token Aura host — the
            //     supertype is already present so the override is a
            //     no-op; the bridge-truthy shape mirrors Python's
            //     `_dc.replace` which skips injection in the same
            //     case).
            //   * `summoning_sick` is recomputed off the (possibly
            //     swapped) `token_pid` — a Moonlit substitution may
            //     change a non-creature placeholder into a creature
            //     token whose printing carries `Creature`, or vice
            //     versa.
            //
            // **Pre-resolve flag-flip ordering.** The per-player
            // `created_tokens_this_turn` flag is read here as a
            // GATE on whether substitution applies, BEFORE the
            // post-resolve flag flip at the tail of this arm. Python's
            // ordering matches (the membership check at `tokens.py:225`
            // precedes the `players_who_created_tokens_this_turn.add`
            // at `:294`). This guarantees the substitution fires once
            // per turn per controller as CR 614 prescribes.
            //
            // **Honest-stop teeth.** Substitution requires:
            //   1. The active controller's slot in
            //      `created_tokens_this_turn` is `false`.
            //   2. Some bf object under the controller's control whose
            //      printing carries `aura_replaces_first_tokens_with
            //      _copies = true` AND whose `attached_to` is `Some`
            //      and points at a live bf object on either player.
            // Either failure → `replacement_pid = None` → no swap.
            // See the `moonlit_token_substitution` helper for the
            // full per-condition tooth list. Mirrors Python's
            // `_find_moonlit_replacement_source` return-None
            // semantics at `effects/base.py:316-324`.
            let (token_pid, moonlit_override) =
                moonlit_token_substitution(state, active, token_pid);
            let token_is_creature = state
                .registry
                .lookup(token_pid)
                .map(|p| p.card_types.iter().any(|t| t == "Creature"))
                .unwrap_or(false);

            // R76-1: draw fresh ids from the monotonic
            // `next_object_id` counter (CR 111.7 — never reuse a
            // freed object's id). This REPLACES the old non-monotonic
            // `max_id + 1` live-zone re-scan, which recycled the id
            // of a token that had ceased to exist. `reserve_object_id`
            // mirrors Python's per-object `self.next_object_id += 1`
            // (game.py:2787-2798) exactly, so the produced ids are
            // byte-equal cross-engine. Reserve all `n` ids up front —
            // before the `Arc::make_mut` battlefield borrow below —
            // since `reserve_object_id` needs `&mut state`.
            let n_u32 = n as u32; // n is the *count parameter (i32 → usize → u32 safe; count ≥ 0 was validated above)
            // ETB-trigger Slice 2 (RS4): collect each fresh token's id
            // so we can dispatch_event(Event::Enters, ...) for each
            // AFTER the bf-push pass releases the player-mut borrow.
            // Strictly-increasing per the per-object reserve, same as
            // Python's per-token loop.
            let mut token_ids: Vec<u32> = Vec::with_capacity(n);
            for _ in 0..n {
                token_ids.push(crate::state::reserve_object_id(state));
            }

            // Push n tokens onto the active player's battlefield via
            // Arc::make_mut so a clone of the state pre-resolve keeps
            // its old battlefield Vec untouched.
            //
            // CE4 closure: stamp each token with a fresh
            // etb_timestamp. Multi-token effects produce strictly-
            // monotonic timestamps so the snapshot / fingerprint walk
            // preserves the within-effect creation order while
            // intermixing correctly with timestamps from prior /
            // subsequent ETBs.
            let bf_etb_base = state.next_etb_timestamp;
            // Slice 14 (TTT Pin 2 alignment with the Python fix at
            // actions.py:2993): summoning sickness is a creature-only
            // property per CR 302.1 + 612.6. `token_is_creature` was
            // computed above in the Moonlit-substitution tuple
            // destructure off the (possibly swapped) `token_pid`, so
            // a Moonlit-substituted token whose host printing is a
            // Creature gets `summoning_sick=true` and a non-creature
            // host gets `false`, matching Python's read off the
            // post-`_dc.replace` printing.
            state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(n_u32);
            let bf: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[active as usize].battlefield);
            for (i, &tok_id) in token_ids.iter().enumerate() {
                bf.push(GameObject {
                    id: tok_id,
                    printing_id: token_pid,
                    controller: active,
                    zone: ZONE_BATTLEFIELD,
                    tapped: false,
                    summoning_sick: token_is_creature,
                    damage: 0,
                    counters: 0,
                    charge_counters: 0,
                    loyalty_counters: 0,
                    attached_to: None,
                    exiled_with_source: None,
                    // Round-30 stun-counter substrate: tokens enter
                    // with no stun counters per make_object precedent.
                    stun_counters: 0,
                    etb_timestamp: bf_etb_base.saturating_add(i as u32),
                    ueot_modifiers: Vec::new(),
                    // warp_cost substrate (Phase 1, round-63): tokens
                    // are never warp-cast (no warp_cost on a token
                    // printing). Back-compat default.
                    warp_exile_pending: false,
                    warped: false,
                    cast_from_exile_with_lightstall_surcharge: false,
                    controller_baseline: None,
                    // **Aura-attachment Phase 4 S3 closure
                    // (2026-05-21).** Per-object Token supertype
                    // override — set `true` only when this token was
                    // minted as a Moonlit-Meditation replacement
                    // copy (`token_printing_id` was swapped to the
                    // host's printing AND the host's printing does
                    // NOT already carry the Token supertype). The
                    // `moonlit_override` local below is computed BEFORE
                    // the loop from the pre-mint replacement-source
                    // lookup and threads through to every pushed token.
                    token_supertype_override: moonlit_override,
                    // Control-change-UEOT substrate — Cycle 1
                    // (2026-05-21): freshly-minted tokens never start
                    // with a UEOT controller override.
                    controller_override: None,
                });
            }
            // ETB-trigger Slice 2 (RS4): queue ENTERS per token in
            // stamp order. Drained by the surrounding resolver after
            // the resolving spell's stack item pops.
            state.pending_etb_dispatches.extend(token_ids);
            // **Aura-attachment substrate — Phase 4 S3 (Moonlit
            // Meditation substrate landing, 2026-05-20).** Flip the
            // per-player created-tokens-this-turn flag once at least
            // one token was created on the controller's bf. Mirrors
            // Python's `state.players_who_created_tokens_this_turn.
            // add(recipient)` at `tokens.py:294`, gated on
            // `effective_count > 0`. Here `n > 0` (we early-returned
            // on `n == 0` above) and every iteration of the bf push
            // produces a token, so the flip is guaranteed-non-zero.
            // The full Moonlit replacement consumer (swap the token's
            // printing_id to the host's printing on the controller's
            // first batch this turn) is NOT yet wired — substrate-
            // only landing per the field docstring.
            if (active as usize) < state.created_tokens_this_turn.len() {
                state.created_tokens_this_turn[active as usize] = true;
            }
        }
        // BLOCKED-RUST-EFFECT slice (round 43) -- each living player
        // creates `count` tokens of the given printing under their
        // own control. Sister of the `Effect::CreateToken` arm
        // above; differs only in the recipient set (every living
        // player vs source-controller). Mirrors Python's
        // `CreateToken(each_player=True).resolve` at
        // `effects/generic/tokens.py:202-294` (the `each_player`
        // branch builds `recipients = [p.id for p in state.players
        // if not p.has_lost]` and runs the per-token push loop once
        // per recipient).
        //
        // EoE consumer: Edge Rover (Artifact Creature -- Robot
        // Scout, "When this creature dies, each player creates a
        // Lander token."). Each-player-creates-token is the FIRST
        // each-player token shape in the audit's BLOCKED-RUST-EFFECT
        // residue and the precondition for any future "each player
        // creates X" sister card.
        //
        // **Mechanism -- fanout.** Two-pass: (1) up-front validation
        // (count >= 0, registry hit on the printing id) so an
        // invalid input surfaces BEFORE any state mutation -- a
        // mid-fanout failure would half-give some recipients tokens
        // and corrupt state. (2) For each living player (skip
        // `has_lost == true` per CR 102.5; mirrors Python's
        // `if not p.has_lost` guard at tokens.py:212), allocate a
        // fresh id range and push `count` tokens onto THEIR
        // battlefield Vec (NOT the source-controller's), tagged
        // `controller = recipient_idx`. Recipients are iterated in
        // `state.players` index order (= controller id order),
        // mirroring Python's `for p in state.players` natural order.
        //
        // **ETB timestamps.** Strict per-token monotonic stamps
        // across all recipients combined (recipient0 first, then
        // recipient1, ...). The CE4 ordering invariant -- a single
        // effect's tokens always carry strictly-increasing stamps,
        // regardless of which player ends up owning them -- mirrors
        // the sister `CreateToken` arm's `bf_etb_base + i` loop and
        // matches Python's natural fanout order.
        //
        // **Pending ENTERS dispatch.** Every fresh token (across
        // every recipient) gets queued in
        // `pending_etb_dispatches`. The surrounding resolver fans
        // out ENTERS event dispatch in queue order, which equals
        // (recipient0_tokens..., recipient1_tokens..., ...).
        //
        // **Out-of-scope first-slice riders.** CR 614 token-creation
        // replacements (Anointed Procession, Moonlit Meditation,
        // Exalted Sunborn), the `tapped`/`attacking`/end-of-combat
        // sacrifice flag bundle -- NONE are plumbed here for the
        // same reason the sister `CreateToken` arm defers them: the
        // EoE consumer (Edge Rover -> Lander) doesn't engage any of
        // them, and the Python path still handles those cases for
        // future cards.
        Effect::CreateTokenEachPlayer {
            count,
            token_printing_id,
        } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "CreateTokenEachPlayer count must be non-negative, got {count}"
                )));
            }
            let token_pid = *token_printing_id;
            if state.registry.lookup(token_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "token_printing_id {token_pid} not in registry"
                )));
            }
            let n = *count as usize;
            if n == 0 {
                return Ok(()); // legal no-op (mirrors sister CreateToken arm)
            }
            // Pre-pass 1: collect living-recipient indices in
            // `state.players` order. Mirrors Python's
            // `[p.id for p in state.players if not p.has_lost]`.
            let n_players = state.players.len();
            let mut living: Vec<usize> = Vec::with_capacity(n_players);
            for idx in 0..n_players {
                if !state.players[idx].has_lost {
                    living.push(idx);
                }
            }
            if living.is_empty() {
                // No living players (degenerate state) -- silent
                // no-op, mirrors Python's empty-recipients-Vec
                // natural fallthrough.
                return Ok(());
            }
            // R76-1: draw fresh ids from the monotonic
            // `next_object_id` counter (CR 111.7), REPLACING the old
            // non-monotonic `max_id + 1` re-scan. Reserve all
            // `total_tokens` ids up front — before any per-recipient
            // `Arc::make_mut` battlefield borrow — in the SAME order
            // they are consumed (recipient0's n, then recipient1's n,
            // ...), so the produced ids stay strictly increasing and
            // byte-equal with Python's per-token
            // `self.next_object_id += 1` over the same fanout order.
            let total_tokens = (n * living.len()) as u32;
            let mut fresh_ids: Vec<u32> =
                Vec::with_capacity(total_tokens as usize);
            for _ in 0..total_tokens {
                fresh_ids.push(crate::state::reserve_object_id(state));
            }
            // **Aura-attachment Phase 4 S3 closure (2026-05-21) —
            // Moonlit Meditation replacement, each-player variant.**
            // Each recipient's batch is independently checked against
            // its OWN per-player `created_tokens_this_turn` slot AND
            // its OWN attached Moonlit-shape Aura set — mirrors
            // Python's `for recipient in recipients: ... moonlit =
            // _find_moonlit_replacement_source(state, recipient)`
            // loop at `effects/generic/tokens.py:216-247`. A recipient
            // who already created tokens this turn or who controls no
            // Moonlit-shape Aura gets the original `token_pid` and
            // `override=false`; a Moonlit-eligible recipient gets the
            // host's printing + `override=true` (modulo the supertype-
            // injection guard inside the helper).
            //
            // Pre-pass computes per-recipient `(pid, override,
            // is_creature)` BEFORE any battlefield mutation begins,
            // so the per-recipient `Arc::make_mut` borrow below
            // doesn't conflict with the immutable state walk inside
            // `moonlit_token_substitution`.
            let per_recipient: Vec<(u32, bool, bool)> = living
                .iter()
                .map(|&recipient_idx| {
                    let (pid, override_flag) =
                        moonlit_token_substitution(
                            state,
                            recipient_idx as u8,
                            token_pid,
                        );
                    let is_creature = state
                        .registry
                        .lookup(pid)
                        .map(|p| {
                            p.card_types.iter().any(|t| t == "Creature")
                        })
                        .unwrap_or(false);
                    (pid, override_flag, is_creature)
                })
                .collect();
            // CE4 closure: stamp each token (across all recipients
            // combined) with a fresh strictly-monotonic etb_timestamp.
            let bf_etb_base = state.next_etb_timestamp;
            state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(total_tokens);
            // Per-token global index so timestamps stay strictly
            // monotonic across the whole fanout (also the index into
            // the pre-reserved `fresh_ids`).
            let mut global_token_idx: u32 = 0;
            let mut token_ids: Vec<u32> = Vec::with_capacity(total_tokens as usize);
            // **Aura-attachment substrate — Phase 4 S3 (Moonlit
            // Meditation substrate landing, 2026-05-20).** Snapshot
            // the recipient indices BEFORE the per-recipient consume
            // so we can flip the per-player created-tokens-this-turn
            // flag after the fanout. Mirrors the sister
            // `Effect::CreateToken` arm's flag flip, fanned out across
            // every living recipient (Python's
            // `recipients = [p.id for p in state.players if not
            // p.has_lost]` / `state.players_who_created_tokens_this_
            // turn.add(recipient)` at `tokens.py:212+294`).
            let recipients_for_flag_flip: Vec<usize> = living.clone();
            for (recipient_pos, recipient_idx) in living.iter().enumerate() {
                let recipient_idx = *recipient_idx;
                let recipient_u8 = recipient_idx as u8;
                let (recipient_pid, recipient_override, recipient_is_creature) =
                    per_recipient[recipient_pos];
                let bf: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[recipient_idx].battlefield);
                for _ in 0..n {
                    let tok_id = fresh_ids[global_token_idx as usize];
                    bf.push(GameObject {
                        id: tok_id,
                        printing_id: recipient_pid,
                        controller: recipient_u8,
                        zone: ZONE_BATTLEFIELD,
                        tapped: false,
                        summoning_sick: recipient_is_creature,
                        damage: 0,
                        counters: 0,
                        charge_counters: 0,
                        loyalty_counters: 0,
                        attached_to: None,
                        exiled_with_source: None,
                        // Tokens enter with no stun counters per
                        // make_object precedent (round 30).
                        stun_counters: 0,
                        etb_timestamp: bf_etb_base.saturating_add(global_token_idx),
                        ueot_modifiers: Vec::new(),
                        // warp_cost substrate (Phase 1, round-63):
                        // tokens are never warp-cast. Back-compat
                        // default.
                        warp_exile_pending: false,
                        warped: false,
                        cast_from_exile_with_lightstall_surcharge: false,
                        controller_baseline: None,
                        // Aura-attachment Phase 4 S3 closure
                        // (2026-05-21): each-player variant Moonlit
                        // substitution. Per-recipient override flag
                        // was computed once above in the `per_recipient`
                        // pre-pass.
                        token_supertype_override: recipient_override,
                        // Control-change-UEOT substrate — Cycle 1
                        // (2026-05-21): freshly-minted tokens never
                        // start with a UEOT controller override.
                        controller_override: None,
                    });
                    token_ids.push(tok_id);
                    global_token_idx += 1;
                }
            }
            // Queue ENTERS for every fresh token in fanout order
            // (recipient0 first, then recipient1, ...). Mirrors the
            // sister `CreateToken` arm's pending dispatch shape.
            state.pending_etb_dispatches.extend(token_ids);
            // **Aura-attachment substrate — Phase 4 S3 (Moonlit
            // Meditation substrate landing, 2026-05-20).** Flip the
            // per-player created-tokens-this-turn flag for every
            // living recipient that just received tokens. Mirrors the
            // sister `Effect::CreateToken` arm's flag flip — Python's
            // each-player branch at `tokens.py:212+294` iterates the
            // same recipient set and adds each to the
            // `players_who_created_tokens_this_turn` set per recipient.
            for recipient_idx in recipients_for_flag_flip {
                if recipient_idx < state.created_tokens_this_turn.len() {
                    state.created_tokens_this_turn[recipient_idx] = true;
                }
            }
        }
        // Phase 6 token-printing substrate -- Phase 4 (round 50): the
        // `tapped` create-token rider. Sister of the `Effect::
        // CreateToken` arm above; the ONLY behavioral difference is
        // that every fresh token enters the battlefield TAPPED
        // (`tapped: true` instead of `tapped: false`). Mirrors
        // Python's `CreateToken(tapped=True).resolve` at
        // `effects/generic/tokens.py:286-287` (the `if self.tapped:
        // obj.tapped = True` post-ETB branch -- the token still enters
        // through `put_into_battlefield` so its ETB fires, then is
        // flipped tapped).
        //
        // EoE consumer: Gravpack Monoist (Creature -- Robot, flying;
        // "When this creature dies, create a tapped 2/2 Robot
        // token." -- 25,252 17lands decks). The Robot token is vanilla
        // (no abilities/keywords, 2/2 Artifact Creature), so this
        // variant + the Phase-1 vanilla-token recognizer + the
        // recognizer widen flip Gravpack Monoist drop->admit with NO
        // Phase-3 Lander-activated-ability dependency.
        //
        // **Mechanism (parity with the sister `CreateToken` arm).**
        // Identical to the sister arm -- count>=0 validation, registry
        // bounds check, controller-of-effect resolution
        // (`current_resolve_source_controller`, I81 fix), fresh-id
        // allocator, CE4 monotonic etb_timestamp, summoning_sick=true,
        // pending ENTERS dispatch -- with the SINGLE byte difference
        // that each pushed `GameObject` carries `tapped: true`. The
        // each-player / attacking / schedule-eoc-sac riders are NOT
        // plumbed here for the same reason the sister arm defers them
        // (no EoE consumer of the tapped shape engages them; the
        // Python path still handles those cases for future cards).
        Effect::CreateTokenTapped {
            count,
            token_printing_id,
        } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "CreateTokenTapped count must be non-negative, got {count}"
                )));
            }
            // Registry-bounds check up front so an unknown printing_id
            // surfaces before any state mutation (mirrors the sister
            // `CreateToken` arm's whole-or-nothing validation).
            let token_pid = *token_printing_id;
            if state.registry.lookup(token_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "token_printing_id {token_pid} not in registry"
                )));
            }
            let n = *count as usize;
            if n == 0 {
                return Ok(()); // legal no-op (mirrors sister CreateToken arm)
            }
            // Controller resolution: same I81 controller-of-effect
            // substrate as the sister `CreateToken` arm -- both the
            // recipient battlefield Vec pick AND the token's
            // `controller` byte derive from
            // `current_resolve_source_controller` (active_player as the
            // back-compat fallback). Mirrors Python's
            // `CreateToken.resolve(state, controller_id, targets)`
            // building `recipients = [controller_id]`.
            let active = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);

            // **Aura-attachment Phase 4 S3 closure (2026-05-21) —
            // Moonlit Meditation replacement, tapped variant.**
            // Mirrors the sister `Effect::CreateToken` arm's
            // substitution: a Moonlit-shape Aura controlled by the
            // recipient with a live `attached_to` host AND a clear
            // `created_tokens_this_turn` slot causes the requested
            // token printing to be swapped for the host's printing
            // with the per-object Token-supertype override stamped.
            // The tapped-on-entry rider survives the substitution —
            // a Moonlit-replaced token still enters tapped per the
            // requesting Effect variant's semantics (Python's
            // `_dc.replace` only swaps the printing; the surrounding
            // `obj.tapped = True` post-ETB stamp at
            // `tokens.py:286-287` is unchanged).
            let (token_pid, moonlit_override) =
                moonlit_token_substitution(state, active, token_pid);

            // R76-1: draw fresh ids from the monotonic
            // `next_object_id` counter (CR 111.7), REPLACING the old
            // non-monotonic `max_id + 1` re-scan. Reserve all `n`
            // ids up front (before the `Arc::make_mut` battlefield
            // borrow) so the per-token ids are strictly increasing
            // and byte-equal with Python's per-token
            // `self.next_object_id += 1`.
            let n_u32 = n as u32;
            let mut token_ids: Vec<u32> = Vec::with_capacity(n);
            for _ in 0..n {
                token_ids.push(crate::state::reserve_object_id(state));
            }

            // Slice 14 (TTT Pin 2 alignment) -- mirror sister
            // `CreateToken` arm: non-creature tokens enter with
            // summoning_sick=false per CR 302.1 + 612.6. Reads off
            // the (possibly Moonlit-swapped) `token_pid` so a
            // substituted creature host correctly stamps
            // summoning_sick=true.
            let token_is_creature = state
                .registry
                .lookup(token_pid)
                .map(|printing| {
                    printing.card_types.iter().any(|t| t == "Creature")
                })
                .unwrap_or(false);
            // CE4 closure: stamp each token with a fresh strictly-
            // monotonic etb_timestamp (mirrors the sister arm).
            let bf_etb_base = state.next_etb_timestamp;
            state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(n_u32);
            let bf: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[active as usize].battlefield);
            for (i, &tok_id) in token_ids.iter().enumerate() {
                bf.push(GameObject {
                    id: tok_id,
                    printing_id: token_pid,
                    controller: active,
                    zone: ZONE_BATTLEFIELD,
                    // The Phase-4 byte difference vs the sister
                    // `CreateToken` arm: the token enters TAPPED.
                    // (warp_cost substrate marker — see the
                    // warp_exile_pending/warped defaults at this
                    // literal's tail; tokens are never warp-cast.)
                    tapped: true,
                    summoning_sick: token_is_creature,
                    damage: 0,
                    counters: 0,
                    charge_counters: 0,
                    loyalty_counters: 0,
                    attached_to: None,
                    exiled_with_source: None,
                    stun_counters: 0,
                    etb_timestamp: bf_etb_base.saturating_add(i as u32),
                    ueot_modifiers: Vec::new(),
                    // warp_cost substrate (Phase 1, round-63): tapped
                    // token enters; never warp-cast. Back-compat
                    // default.
                    warp_exile_pending: false,
                    warped: false,
                    cast_from_exile_with_lightstall_surcharge: false,
                    controller_baseline: None,
                    // Aura-attachment Phase 4 S3 closure (2026-05-21):
                    // tapped-token-variant Moonlit substitution.
                    // `moonlit_override` was computed once above off
                    // the pre-mint replacement-source lookup.
                    token_supertype_override: moonlit_override,
                    // Control-change-UEOT substrate — Cycle 1
                    // (2026-05-21): freshly-minted tokens never start
                    // with a UEOT controller override.
                    controller_override: None,
                });
            }
            // Queue ENTERS per token in stamp order -- mirrors the
            // sister `CreateToken` arm's pending dispatch shape. The
            // token enters tapped but its ETB still fires (CR 603.6e
            // -- entering tapped does not suppress the enters trigger).
            state.pending_etb_dispatches.extend(token_ids);
            // **Aura-attachment substrate — Phase 4 S3 (Moonlit
            // Meditation substrate landing, 2026-05-20).** Same
            // per-player created-tokens-this-turn flag flip as the
            // sister `Effect::CreateToken` arm — the tapped variant
            // is still a "you created one or more tokens" event for
            // Python's `players_who_created_tokens_this_turn` set
            // (`tokens.py:294`).
            if (active as usize) < state.created_tokens_this_turn.len() {
                state.created_tokens_this_turn[active as usize] = true;
            }
        }
        // Round-23: AddManaToPool. CR 106.1 — the resolution of a mana
        // ability. The active player (= controller of the resolving
        // mana ability or stack item, first-slice convention mirrored
        // from DealDamage / DrawCards / GainLife / CreateToken) gets
        // ``amount`` mana of ``color`` credited to their pool.
        //
        // **Controller convention: ``state.active_player``** — same
        // first-slice shape as every other implicit-controller arm.
        // Real MTG plumbs the spell/ability controller through
        // ``StackItem.controller``; deferred to a future slice along
        // with the rest of the controller plumbing.
        //
        // **Negative ``amount`` raises** loudly (mirrors ScryN /
        // DealDamage / DamageEach* / CreateToken raise-on-negative
        // shape). amount=0 is a legitimate no-op success.
        //
        // **Inlining note (cross-file friction).** The Rust `add_mana`
        // primitive on `GameState` lives in `state_ops_mana.rs`
        // (round 2) — it does exactly what we need (validate
        // non-negative, slot-mut by color, saturating_add). Routing
        // through it would require a cross-`#[pymethods]` call which
        // PyO3 lets us model only via `Bound::call_method1`, an awkward
        // shape inside `apply_effect` where we already hold
        // `state: &mut GameState`. Inlining the slot match here is
        // simpler and keeps `apply_effect` as a pure-Rust dispatch.
        // A follow-up consolidation slice can collapse this onto the
        // shared primitive once cross-file private-fn calls are
        // re-enabled.
        Effect::AddManaToPool { color, amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "AddManaToPool amount must be non-negative, got {amount}"
                )));
            }
            let controller_idx = state.active_player as usize;
            let pool = &mut state.players[controller_idx].mana_pool;
            let amount_u8 = (*amount).min(u8::MAX as i32) as u8;
            // G7 S6-Class-E ordered-pool: route through `push_coins`
            // so the insertion-ordered `coins` vec and the per-color
            // count cache stay synced.
            let color_byte = match color.as_str() {
                "W" => b'W',
                "U" => b'U',
                "B" => b'B',
                "R" => b'R',
                "G" => b'G',
                "C" => b'C',
                other => {
                    return Err(PyValueError::new_err(format!(
                        "AddManaToPool: unknown color {other:?}; valid: W, U, B, R, G, C"
                    )))
                }
            };
            pool.push_coins(color_byte, amount_u8);
            pool.assert_count_sync();
        }
        // Round-23: SearchLibraryForBasicLandTapped. CR 701.19 search +
        // 701.22 shuffle. The active player (= controller of resolving
        // stack item, first-slice convention mirrored from every other
        // implicit-controller arm) searches their library for up to
        // `count` basic land cards, puts each onto the battlefield
        // tapped. Powers Rampant Growth / Beamsaw Prospector / Kavaron
        // Harrier — 15 cards in the EoE pool per round-20 measurement.
        //
        // **Deterministic first-match pick (top-to-bottom).** Library
        // is bottom-first in the Vec (lib[len-1] = top of library); scan
        // via iter().rev() to grab the topmost matching object. Real
        // MTG lets the player choose any matching card; deferred until
        // decision callbacks exist.
        //
        // **Basic-land filter: `card_types contains "Land"` AND
        // `supertypes contains "Basic"`.** Mirrors Python's
        // `is_basic_land` = `is_land and "Basic" in supertypes` exactly
        // (CR 205.4a — a basic land is a land with the Basic supertype).
        // Round-30 supertypes substrate (`feature/rust-supertypes-
        // registry` merge `ef8fa004`) added `CardPrinting.supertypes:
        // Vec<String>` with JSON load + getter; the round-22 sibling
        // `Effect::SearchLibraryForCard` arm flipped its
        // `legendary_creature` predicate to consume it at the same
        // time, but this arm was left on the card_types-only proxy
        // (PARTIAL-RESOLVED in SEARCH_LIMITATIONS entry #15). This
        // slice closes that residue: predicate now reads
        // `card_types Land AND supertypes Basic`, matching Python's
        // `is_basic_land` bit-for-bit. Pre-migration the proxy was
        // faithful for the EoE pool (every basic land carried
        // card_types=["Land"] with no non-basic land sharing this
        // dispatch path), so the round-21 regression pins still pass —
        // this is a code-cleanup migration that hardens the predicate
        // against future non-basic-land entries on the bridge.
        //
        // **Post-search shuffle (CR 701.19 / 701.22 — G19 closure).**
        // After all `count` basic-land picks (or partial-fill on
        // early exhaust), the library is Fisher-Yates shuffled in
        // place. Mirrors the post-CE8 `fisher_yates_shuffle_optional`
        // primitive in `state_ops_library.rs` (and its 7 sibling
        // shuffle sites) — same xorshift64 control flow, just
        // inlined here per the per-slice file-isolation convention
        // every other arm in this file follows.
        //
        // **Seed source — `state.rng_ref()` external rng if set,
        // xorshift64 fallback otherwise (Effect-rng-threading closure,
        // G19 follow-up).** CE8 Slice 2 wired all 8 pymethod-surface
        // shuffle sites to consume an optional `rng: &Bound<PyAny>`
        // kwarg, falling through to xorshift64 when `None`. The
        // Effect-rng-threading closure extends that substrate to the
        // single Effect arm that consumes randomness today
        // (`Effect::SearchLibraryForBasicLandTapped`) by stashing the
        // optional `random.Random` on `GameState.rng` (caller-set via
        // `set_external_rng`) rather than threading a parameter
        // through every intervening call site
        // (`apply_effect` / `resolve_top_of_stack` / etc — Option A in
        // `docs/RUST_EFFECT_RNG_AUDIT.md`). When `state.rng_ref()` is
        // `Some(py_rng)`, the post-search Fisher-Yates reads draws via
        // `external_rng::next_below(py_rng, n)` → byte-identical to
        // Python's `random.Random.shuffle(list)`. When `None`, the
        // legacy xorshift64 path runs — seed derived deterministically
        // from `(turn_number, active_player, source_obj_id,
        // library_len)`, stable across reruns of the same starting
        // state.
        //
        // **count<0 raises** loudly (defense-in-depth; parse_effect
        // catches this at register time). **count=0 is a legal no-op**
        // (mirrors GainLife / DrawCards / Mill / CreateToken zero-count
        // conventions).
        //
        // **Partial fill on early-exhaust.** If fewer than `count`
        // basic lands exist in the library, stop early silently —
        // matches Python's `next(..., None)` short-circuit on miss.
        //
        // **Per-object state on battlefield entry.** Lands enter
        // tapped (`tapped = true`) and NOT summoning-sick
        // (`summoning_sick = false` — CR 302.1 only applies to
        // creatures; a land bypassing creature-spell resolution via
        // a search effect has no summoning sickness regardless). The
        // `controller` field is stamped to the SEARCHER (the
        // source's controller, per I69) — searching for someone
        // else's basic land into their own battlefield isn't a
        // thing in real MTG, so the searched land enters under the
        // searcher's control.
        //
        // **Inlining note (cross-file friction).** Per per-slice file-
        // isolation we inline the search + move logic here rather
        // than calling `search_library_for_first_of_type_and_shuffle`
        // (which is a `#[pymethods]` method in
        // `state_ops_library.rs` requiring a `Bound::call_method1`
        // awkward shape inside `apply_effect`). A future
        // consolidation slice can collapse onto the shared primitive.
        Effect::SearchLibraryForBasicLandTapped { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "SearchLibraryForBasicLandTapped count must be non-negative, got {count}"
                )));
            }
            let n = *count as usize;
            if n == 0 {
                return Ok(());
            }
            // **Controller resolution (I69 follow-up to the I59 /
            // I60 / I63 / I64 / I65 / I66 / I67 / I68 controller-
            // of-effect substrate).** Prefer the transient
            // `state.current_resolve_source_controller` (set by
            // the stack resolver around each `apply_effect` call),
            // falling back to `state.active_player` for back-compat
            // with direct `apply_effect_pub` callers. Same pattern
            // as `Effect::DrawCards` / `Effect::GainLife` /
            // `Effect::Mill` / `Effect::Sacrifice` — CR 119.5 /
            // 608.2: an effect resolves under its source's
            // controller. For SearchLibraryForBasicLandTapped
            // ("your library"), the searcher is the SOURCE'S
            // controller, not the active player. Mirrors the
            // Python `SearchLibraryForBasicLandTapped.resolve`
            // shape, which takes `controller_id` from the
            // resolver (the StackItem's controller).
            //
            // **Why the fix.** Pre-fix this arm hardcoded
            // `controller_idx = state.active_player`. Wrong when
            // the resolving spell / triggered ability has a
            // controller != active_player. Example: on p0's turn,
            // p1's Cultivate-shape spell ("search your library
            // for up to 2 basic land cards") resolves — pre-fix
            // it pulled basics from P0'S library onto P0's
            // battlefield (the active player), not p1's; post-fix
            // it correctly pulls from p1's library onto p1's
            // battlefield.
            //
            // **xorshift seed unchanged.** The fallback seed at
            // the post-search shuffle still mixes `state.active_-
            // player` (alongside turn, source_obj_id, library_-
            // len). Changing it would break the existing
            // `test_resolve_search_library_for_basic_land_tapped_-
            // shuffle_is_deterministic` pin, and the seed is a
            // pure-state determinism input, not a behavior switch
            // — per-run uniqueness comes from source_obj_id +
            // library_len mixing.
            let controller_idx = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let controller_u8 = controller_idx as u8;
            // ETB-trigger Slice 2 (RS2): collect the obj_id of every
            // land we move onto the battlefield so we can fan out
            // `dispatch_event(Event::Enters, ...)` for each AFTER the
            // post-search shuffle drops its registry/zone borrows.
            let mut entered_ids: Vec<u32> = Vec::with_capacity(n);
            for _ in 0..n {
                // Locate the topmost basic-land match (forward iter
                // = top-to-bottom post-CE3, lib[0] = top). Inlined
                // here (vs. delegating to the search-only sibling)
                // so we have the Vec INDEX for removal, not just the
                // obj_id.
                let pos: Option<usize> = {
                    let lib = &state.players[controller_idx].library;
                    let mut found: Option<usize> = None;
                    for (i, obj) in lib.iter().enumerate() {
                        let printing = match state.registry.lookup(obj.printing_id) {
                            Some(p) => p,
                            None => continue,
                        };
                        // Round-37 supertypes-substrate migration:
                        // basic-land predicate is `card_types Land AND
                        // supertypes Basic` (CR 205.4a), bit-equivalent
                        // to Python's `is_basic_land`. Sibling round-22
                        // `Effect::SearchLibraryForCard` arm consumes
                        // the same `printing.supertypes.iter().any(|s|
                        // s == "Basic")` substrate for its
                        // `legendary_creature` predicate.
                        if printing.card_types.iter().any(|t| t == "Land")
                            && printing.supertypes.iter().any(|s| s == "Basic")
                        {
                            found = Some(i);
                            break;
                        }
                    }
                    found
                };
                let p = match pos {
                    Some(p) => p,
                    None => break, // no more basic lands — stop early silently
                };
                // CE4 closure: reserve before the player-mut borrow.
                let etb_ts = crate::state::reserve_etb_timestamp(state);
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                let mut obj = lib.remove(p);
                // Stamp battlefield entry: tapped, not summoning-sick
                // (CR 302.1 applies to creatures; lands don't have
                // summoning sickness regardless). Controller = the
                // SEARCHER (source's controller, per I69).
                obj.zone = ZONE_BATTLEFIELD;
                obj.tapped = true;
                obj.summoning_sick = false;
                obj.controller = controller_u8;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                obj.etb_timestamp = etb_ts;
                entered_ids.push(obj.id);
                Arc::make_mut(&mut player.battlefield).push(obj);
            }
            // G19 closure: CR 701.19 / 701.22 — always shuffle the
            // library after a search, whether or not a card was
            // found (the early-`break` above on no-match falls
            // through to here too). Effect-rng-threading closure:
            // route through `state.rng_ref()` when set (external
            // Python `random.Random`), fall back to inlined
            // xorshift64 otherwise; seed mixes turn / active_player /
            // source_obj_id / library_len for determinism within a
            // run without giving identical permutations across
            // separate searches in the same game.
            let lib_len = state.players[controller_idx].library.len();
            if lib_len >= 2 {
                // Effect-rng path: consume from the stashed Python
                // `random.Random` if the caller installed one via
                // `set_external_rng`. We extract the `Py<PyAny>` into
                // an owned reference BEFORE taking the `&mut
                // self.players[..]` borrow so the rng borrow doesn't
                // conflict with the mutable-zone borrow that
                // `Arc::make_mut` produces below.
                let py_rng_owned: Option<Py<PyAny>> = state
                    .rng_ref()
                    .map(|r| Python::with_gil(|py| r.clone_ref(py)));
                if let Some(py_rng) = py_rng_owned {
                    let player = &mut state.players[controller_idx];
                    let lib = Arc::make_mut(&mut player.library);
                    let n = lib.len();
                    Python::with_gil(|py| -> PyResult<()> {
                        let bound = py_rng.bind(py);
                        for i in (1..n).rev() {
                            let j = crate::external_rng::next_below(bound, i + 1)?;
                            lib.swap(i, j);
                        }
                        Ok(())
                    })?;
                } else {
                    // xorshift64 fallback: distinct seed per (turn,
                    // ap, source, lib_len) tuple. The zero-collapse
                    // defense matches every other xorshift site in
                    // the engine.
                    let mut seed = (state.turn_number as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ ((state.active_player as u64) << 56)
                        ^ ((source_obj_id as u64) << 16)
                        ^ (lib_len as u64);
                    if seed == 0 {
                        seed = 0xdead_beef_dead_beef;
                    }
                    let player = &mut state.players[controller_idx];
                    let lib = Arc::make_mut(&mut player.library);
                    let n = lib.len();
                    let mut x = seed;
                    for i in (1..n).rev() {
                        // xorshift64 step.
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        // Rejection-sample to an unbiased index in
                        // [0, i+1). i+1 <= library size (~100), so the
                        // rejection-rate cost is negligible.
                        let n_u64 = (i + 1) as u64;
                        let bound = u64::MAX - (u64::MAX % n_u64);
                        let mut r = x;
                        while r >= bound {
                            r ^= r << 13;
                            r ^= r >> 7;
                            r ^= r << 17;
                            x = r;
                        }
                        let j = (r % n_u64) as usize;
                        lib.swap(i, j);
                    }
                }
            }
            // ETB-trigger Slice 2 (RS2): queue ENTERS dispatches per
            // searched-up land, in stamp order (= the order they were
            // moved onto the battlefield above). The caller
            // (`resolve_top_of_stack` / `_pause`) drains the queue
            // after popping the resolving spell's stack item.
            state.pending_etb_dispatches.extend(entered_ids);
        }
        // Round-23: PutPlusOneCounterOnSelf. CR 122.6 + 701.13 — put
        // `amount` +1/+1 counters on the resolving spell's source
        // permanent ("this creature" / "self" target). Powers Blade
        // of the Swarm, Atomic Microsizer — 16 cards in the EoE pool
        // per round-20 measurement.
        //
        // **"Self" vs AddCounters' "defender's first creature".**
        // AddCounters deterministically picks the defending player's
        // battlefield[0] (an opponent-target shape). This arm
        // recipient = the source GameObject of the resolving stack
        // item, located via `source_obj_id` threaded through
        // `apply_effect`. Mirrors Python's `state.resolving_source`
        // lookup in `PutPlusOneCounterOnSelf.resolve`.
        //
        // **Source-not-on-battlefield = silent no-op.** Real MTG: if
        // the source has left the battlefield by the time the
        // counter resolves (e.g. it was destroyed in response to
        // the trigger), the counter "lands on nothing" / fizzles.
        // First slice honors that — we walk both players' battlefield
        // Vecs looking for an object with the matching id; if not
        // found, return Ok(()) silently. Mirrors Python's
        // `source is None` short-circuit. NOTE: this means a
        // schema-allowed Instant/Sorcery printing carrying this
        // effect (the source still sits on the stack at resolve
        // time, NOT on the battlefield) fizzles — semantically
        // nonsense ("the instant's source isn't a creature"), but
        // doesn't error.
        //
        // **amount<0 raises** loudly (defense-in-depth; parse_effect
        // catches at register time). **amount=0 is a legal no-op**
        // (mirrors GainLife / DrawCards / Mill / SearchLibrary /
        // CreateToken zero-amount conventions).
        //
        // **Round-2 net-counter model.** The positive `amount` lands
        // as a positive delta on the i32 `counters` field via
        // `saturating_add` — same shape as AddCounters' positive
        // branch.
        //
        // **Inlining note (cross-file friction).** Per per-slice
        // file-isolation we inline the battlefield scan +
        // `Arc::make_mut` + `saturating_add` here, mirroring how
        // AddCounters / DealDamage AnyCreature / TapTarget /
        // UntapTarget inline their counter-mutations.
        Effect::PutPlusOneCounterOnSelf { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutPlusOneCounterOnSelf amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            let delta = *amount;
            // Locate the source on a battlefield zone. If it lives
            // somewhere else (hand / library / graveyard / stack /
            // exile) the counter add fizzles silently per the docstring
            // — schema-allowed but semantically a no-op for the spell
            // / ETB / trigger that referenced "self".
            let mut found_player: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    found_player = Some(pidx);
                    break;
                }
            }
            let pidx = match found_player {
                Some(p) => p,
                None => return Ok(()), // fizzle silently
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == source_obj_id {
                    obj.counters = obj.counters.saturating_add(delta);
                    break;
                }
            }
            // Round-16 RR Pin 1: dispatch COUNTERS_ADDED after the
            // self-counter mutation. Subject = source_obj_id (the
            // object that received counters on itself). Mirrors Python's
            // ``fire_counters_added_to_your_creature_triggers`` call
            // in ``game.add_plus_one_counters``.
            dispatch_event(
                state,
                crate::cards::Event::CountersAdded,
                source_obj_id,
            )?;
        }
        // Round-27 G20 closure: Surveil. CR 701.43 — controller looks
        // at the top `count` cards of their library and may put any
        // number of them into their graveyard; the rest stay on top
        // in any order. Powers Appendage Amalgam, etc.
        //
        // **Sibling of ScryN (round 16).** Same look-at-top-N shape,
        // different destination zone: Scry routes chosen cards to the
        // bottom of the library; Surveil routes them to the graveyard.
        //
        // **Decision-callback substrate (Option B: state-machine).**
        // Surveil pops `count` cards (clamped to library size) off the
        // library top and packages them into a
        // `PendingDecision::SurveilChoice` on `state.pending_decisions`.
        // The Effect arm short-circuits — the cards are NOT yet in the
        // graveyard and NOT yet back on the library. The caller drains
        // via `submit_surveil_decision(top, graveyard)` (the dispatcher
        // moves each card to its chosen zone) or
        // `auto_resolve_pending_decisions("keep_all_on_top" |
        // "send_all_to_graveyard")` (the deterministic fast-path that
        // preserves r23 baseline behavior).
        //
        // **Why state-machine over Pyo3 callback (Option A).** Pending
        // decisions participate in fingerprint / clone / CoW just like
        // any other gameplay state. A `Bound<PyAny>` callback baked into
        // the apply_effect signature would not survive fingerprinting,
        // would break clone semantics (callbacks aren't cloneable across
        // a Python boundary safely), and would couple every Effect-
        // resolution call site to a Python callable — incompatible with
        // pure-Rust MCTS rollouts. Mirrors Python's pause-and-resume
        // `ResolutionSurveilDecision` shape in
        // `mtg_bot.search.resolution_channel`.
        //
        // **Controller convention: ``state.active_player``** — first-
        // slice shape consistent with ScryN / DrawCards / GainLife (all
        // implicit-controller). Real MTG plumbs the spell controller
        // through `StackItem.controller`; deferred. `source_obj_id` is
        // recorded on the SurveilChoice for log-correlation and future
        // controller-resolution work.
        //
        // **Negative ``count`` raises** loudly (CR 701.43 doesn't
        // define Surveil -1; surface it as a schema error). Mirrors
        // ScryN's negative-count rejection.
        //
        // **count > library size**: CR 701.43 clamps to library size
        // ("look at all cards in the library"). We pop `min(count,
        // library.len())` cards; the rest of the dispatch is unchanged.
        //
        // **count = 0**: legitimate no-op. We still push an empty
        // SurveilChoice so the caller's invariant (every Surveil emits
        // exactly one PendingDecision) holds; auto-resolve discards it
        // in O(1). Pre-r27 r23-baseline tests use auto-resolve via the
        // post-resolve hook, so count=0 plus auto-resolve is bit-equal
        // to the pre-r27 behavior.
        Effect::Surveil { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "Surveil count must be non-negative, got {count}"
                )));
            }
            // I61: prefer the transient source-controller swap (same
            // pattern as I59 / I60 reader-side fix).
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let pidx = controller as usize;
            // Clamp count to library size (CR 701.43 fold).
            let lib_len = state.players[pidx].library.len();
            let n = (*count as usize).min(lib_len);
            // Remove n cards off library TOP. Post-CE3 zones
            // convention: top of library = lib[0]. So we remove from
            // the head n times. Order in `cards`: cards[0] is the
            // topmost card (first removed), cards[1] is second-from-
            // top, etc. The submit dispatcher preserves this ordering
            // when routing back to the library top.
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(n);
            for _ in 0..n {
                // Always remove index 0 — the new top after each
                // prior remove shifts down.
                let removed = lib_mut.remove(0);
                cards.push(removed);
                // The card currently lives in NO zone — its `zone`
                // field is stale until the submit dispatcher writes
                // the new zone. The submit path rewrites zone before
                // pushing back to library or graveyard.
            }
            state.pending_decisions.push(PendingDecision::SurveilChoice {
                source_obj_id,
                controller,
                cards,
            });
        }
        // Round-23: CounterTargetSpell. CR 701.5 — counter target
        // spell on the stack; the countered spell moves to its
        // controller's graveyard (CR 701.6 — first slice routes to
        // controller's graveyard since owner-vs-controller tracking
        // isn't yet split out). Powers Annul, Cancel, etc. — 7 cards
        // in the EoE pool per round-20 measurement.
        //
        // **Deterministic target choice: topmost-non-self stack item.**
        // Scan the stack from top (Vec last) down, pick the first
        // `source_obj_id` that isn't this resolving effect's own
        // source — a counterspell sitting on the stack shouldn't
        // counter itself before it resolves. The `source_obj_id`
        // parameter (round-23 refactor on `apply_effect`) carries the
        // resolving stack item's source id; we exclude it here. Real
        // MTG plumbs target choice through the cast surface; first
        // slice deterministic-top-non-self mirrors the
        // "DealDamage picks defender's battlefield[0]" convention.
        //
        // **Empty stack / only-self stack: silent fizzle.** Mirrors
        // Python's `CounterTargetSpell.resolve` short-circuit on
        // `if not targets`. The Effect doesn't error; the
        // post-resolve mover still routes the spell source to the
        // graveyard, the stack still pops at the outer
        // `resolve_top_of_stack`.
        //
        // **No filter_types / no `cant_be_countered` enforcement.**
        // Python's `filter_types` (Annul's Artifact/Enchantment-only
        // restriction) and `_is_uncounterable` checks
        // (`spell_is_uncounterable`, `grants_creature_spells_uncounterable`)
        // are deferred — the Rust bridge has neither field today.
        // First slice trusts the caller picked a legal target shape,
        // same convention as the round-7 counter_spell primitive.
        //
        // **Inlining note (cross-file friction).** Round-7's
        // `counter_spell(target_obj_id) -> bool` primitive on
        // `state_ops_counterspell.rs` does exactly what we need
        // (locate stack item, remove, move source to controller's
        // graveyard). Routing through it would require
        // `Bound::call_method1` cross-`#[pymethods]` plumbing inside
        // `apply_effect` where we already hold `state: &mut GameState`
        // — the same E0624 friction every other Effect arm hit. Per
        // the recurring per-slice file-isolation pattern (every other
        // arm — DealDamage / ReturnToHand / Mill / DrawCards /
        // ExileTarget / Sacrifice / DestroyTarget / TapTarget /
        // UntapTarget / AddCounters / DiscardCard / CreateToken /
        // AddManaToPool / SearchLibraryForBasicLandTapped /
        // PutPlusOneCounterOnSelf — inlines its primitive for the
        // same reason), we inline the equivalent logic here. A
        // follow-up consolidation slice can collapse onto the shared
        // `counter_spell` primitive once cross-file private-fn calls
        // re-open.
        Effect::CounterTargetSpell => {
            // Find the topmost stack item whose source isn't this
            // effect's own source. iter().rev() walks top-to-bottom;
            // capture both the Vec index (for stack removal) and the
            // controller (for graveyard routing).
            //
            // **Static / layered abilities substrate — Phase 2 (round-80,
            // 2026-05-20).** Skip stack items that are uncounterable
            // per CR 101.2. Two sources of uncounterability mirror
            // Python's `CounterTargetSpell._is_uncounterable` walk
            // (`effects/generic/stack_control.py:150`):
            //
            //   1. The stack item's source printing carries
            //      `spell_is_uncounterable=true` (e.g. Frenzied Baloth's
            //      "This spell can't be countered.").
            //   2. The stack item is a creature spell AND the
            //      controller has any battlefield permanent carrying
            //      `grants_creature_spells_uncounterable=true` (e.g.
            //      Frenzied Baloth on the battlefield granting the
            //      same protection to your other creature spells).
            //
            // Both checks scan from top-of-stack down, skipping any
            // item that hits either gate. If no admissible target
            // remains, silent fizzle (matches Python's `if not legal_
            // targets: return` short-circuit at resolve time).
            let target: Option<(usize, u32, u8)> = (|| {
                for (i, item) in state.stack.iter().enumerate().rev() {
                    if item.source_obj_id == source_obj_id {
                        continue;
                    }
                    // Look up the source's printing via the source
                    // GameObject. Sweep stack_objects + 4 per-player
                    // zones; identical search shape to the move-to-
                    // graveyard step below (the source typically lives
                    // in stack_objects for a spell on the stack).
                    let src_printing_id: Option<u32> = state
                        .stack_objects
                        .iter()
                        .find(|o| o.id == item.source_obj_id)
                        .map(|o| o.printing_id)
                        .or_else(|| {
                            for player in state.players.iter() {
                                for zone in [
                                    &player.hand,
                                    &player.library,
                                    &player.battlefield,
                                    &player.graveyard,
                                ] {
                                    if let Some(obj) =
                                        zone.iter().find(|o| o.id == item.source_obj_id)
                                    {
                                        return Some(obj.printing_id);
                                    }
                                }
                            }
                            None
                        });
                    let Some(pid) = src_printing_id else {
                        // Lost-track-of-source: skip this item per
                        // the same "best-effort" convention the
                        // graveyard-move step below uses.
                        continue;
                    };
                    let Some(printing) = state.registry.lookup(pid) else {
                        continue;
                    };
                    // Gate 1: spell_is_uncounterable on the source's
                    // printing.
                    if printing.spell_is_uncounterable {
                        continue;
                    }
                    // Gate 2: creature spell + any permanent under
                    // the controller carries
                    // grants_creature_spells_uncounterable.
                    let is_creature = printing
                        .card_types
                        .iter()
                        .any(|t| t == "Creature");
                    if is_creature {
                        let controller_idx = item.controller as usize;
                        if let Some(player) = state.players.get(controller_idx) {
                            let mut granted = false;
                            for obj in player.battlefield.iter() {
                                if let Some(p) =
                                    state.registry.lookup(obj.printing_id)
                                {
                                    if p.grants_creature_spells_uncounterable {
                                        granted = true;
                                        break;
                                    }
                                }
                            }
                            if granted {
                                continue;
                            }
                        }
                    }
                    return Some((i, item.source_obj_id, item.controller));
                }
                None
            })();
            let (stack_pos, target_obj_id, target_controller) = match target {
                Some(t) => t,
                None => return Ok(()), // empty / only-self / all-uncounterable: silent fizzle
            };
            // Remove the matched stack item (inlines round-7's
            // `counter_spell` stack-removal step).
            state.stack.remove(stack_pos);
            // Best-effort: move the source GameObject to its
            // controller's graveyard (inlines round-7's
            // `move_to_graveyard` step). Sweep both players' four
            // tracked zones to find the source — mirrors how
            // `state_ops_counterspell::move_to_graveyard` itself walks
            // hand/library/battlefield/graveyard. CR 400.7 zone-change
            // reset (tapped / summoning_sick / damage / counters) is
            // applied on the way out, mirroring every other zone-move
            // arm in this file (ReturnToHand / ExileTarget / Sacrifice
            // / DestroyTarget / DiscardCard). If the source is missing
            // (lost-track-of-source failure mode), the counter still
            // "lands" — stack item removed — but the graveyard step
            // is a no-op, same tolerance the round-7 primitive shows.
            let mut taken: Option<GameObject> = None;
            // G7 S5a (Option β): a spell on the stack has its owned
            // GameObject in `state.stack_objects` (NOT a per-player
            // zone). Check the store FIRST — the countered spell IS
            // the target here, so for the common counter-a-spell case
            // the source lives in the store. (Pre-S5a it was hand-
            // resident and the 4-zone sweep found it; the store check
            // is the byte-equivalent for the re-homed model. Without
            // this the countered spell's GameObject would orphan in
            // the store and never reach the graveyard — an off-target
            // leak the reconciliation must close,
            // `feedback_silent_noop_pipelines`.)
            if let Some(pos) = state
                .stack_objects
                .iter()
                .position(|o| o.id == target_obj_id)
            {
                taken = Some(state.stack_objects.remove(pos));
            }
            if taken.is_none() {
                'outer: for player in state.players.iter_mut() {
                    for zone in [
                        &mut player.hand,
                        &mut player.library,
                        &mut player.battlefield,
                        &mut player.graveyard,
                    ] {
                        if let Some(pos) = zone.iter().position(|o| o.id == target_obj_id) {
                            let v = Arc::make_mut(zone);
                            taken = Some(v.remove(pos));
                            break 'outer;
                        }
                    }
                }
            }
            if let Some(mut obj) = taken {
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                obj.controller = target_controller;
                Arc::make_mut(
                    &mut state.players[target_controller as usize].graveyard,
                )
                .push(obj);
            }
        }
        // Round-24: ReturnCreatureFromYourGraveyardToBattlefield.
        // CR 400.7 (zone-change reset) + CR 302.1 (new permanent has
        // summoning sickness) — reanimate a creature card from the
        // controller's graveyard onto their battlefield. Powers Emerge
        // from the Cocoon, Reanimate-style sugar — 8 cards in the EoE
        // pool per round-20 measurement.
        //
        // **Controller = `state.active_player`** (first-slice
        // convention, same as DrawCards / GainLife / Mill /
        // SearchLibraryForBasicLandTapped / CounterTargetSpell).
        // `source_obj_id` is unused here — the target zone is "your"
        // graveyard (= controller's), not source-relative.
        //
        // **Deterministic first-creature pick.** Scan the controller's
        // graveyard in Vec iteration order (front = smallest index =
        // oldest by `move_object`-time push order); pick the first
        // object whose printing's `card_types` contains "Creature".
        // No filtering past that (no mana-value cap, no
        // creature-or-enchantment widening — see the variant's
        // docstring for the deferred Python schema). Empty graveyard
        // or no-creature-found: silent fizzle (Python parity with
        // `ReturnFromGraveyard.resolve`'s `if not targets: return`
        // short-circuit, and matches the round-17 primitive's
        // `Ok(false)` return on absent obj_id).
        //
        // **Inlining note (cross-file friction).** The round-17
        // `move_object_from_graveyard_to_battlefield(player_id,
        // obj_id)` primitive on `state_ops_zones.rs` does exactly the
        // recipient-side work we need (remove from graveyard, push
        // onto battlefield, stamp `summoning_sick = true`, zero
        // tapped / damage / counters). Per the recurring per-slice
        // file-isolation pattern (CounterTargetSpell / Sacrifice /
        // ExileTarget / ReturnToHand / DrawCards / Mill / ... all
        // inline their primitives), this arm inlines the equivalent
        // logic. A follow-up consolidation slice can collapse onto
        // the shared primitive once cross-file private-fn calls
        // re-open.
        //
        // **Registry lookup pattern.** We hold `state: &mut GameState`,
        // and `state.registry.lookup(printing_id)` returns a borrow on
        // the registry. The `find_map` closure releases each borrow on
        // each iteration; once the loop completes we hold an
        // `Option<u32>` and the registry borrow is dropped, leaving
        // the mutable state borrow free for the subsequent
        // `Arc::make_mut` calls. Mirrors the borrow-then-drop snapshot
        // pattern `resolve_top_of_stack` uses for `printing.effects`.
        // Round-25: DiscardSelf. CR 701.8 — "you discard a card."
        // The SOURCE's controller discards `amount` cards from their
        // hand. Punisher / drawback pattern (Irreverent Gremlin
        // shape) — 8 cards in the EoE pool per round-20 measurement.
        //
        // **Distinct from round-15's DiscardCard.** DiscardCard's
        // AnyPlayer branch targets the DEFENDING player (active
        // player's spell forces the opponent to discard, Mind Rot
        // shape). DiscardSelf targets the SOURCE's controller —
        // the punisher fires on whoever cast or controls the
        // source, regardless of active_player. Cross-controller-
        // active-player tests below verify the distinction.
        //
        // **Source-finder dispatch (round-23 pattern).** Threads
        // `source_obj_id` through `apply_effect` and walks both
        // players' five tracked zones (hand / library /
        // battlefield / graveyard / exile) to recover the source's
        // `controller`. Differs from
        // `PutPlusOneCounterOnSelf`'s battlefield-only scan because
        // the punisher pattern fires on whoever controls the source
        // regardless of where the source currently sits — the
        // controller field doesn't change when a permanent leaves
        // the battlefield. Additionally scans the stack: the source
        // may still be resolving (e.g. an instant/sorcery still on
        // the stack at its own resolution time), in which case the
        // `StackItem.controller` is the authoritative controller.
        //
        // **First-slice deterministic choice: discard hand[0].**
        // Mirrors round-15's `DiscardCard` arm — pop the front of
        // the controller's hand Vec by index, no per-spell discard
        // choice yet. Up to `amount` discards in a row (stop early
        // on empty hand, matching Python's `for _ in
        // range(self.amount): if not p.hand: break` shape).
        //
        // **Empty hand = silent no-op** (Python parity:
        // `DiscardSelf.resolve` short-circuits when `p.hand` is
        // empty). **Source not found anywhere = silent no-op**
        // (lost-track-of-source tolerance, mirrors
        // `PutPlusOneCounterOnSelf` / `CounterTargetSpell` fizzle
        // shape).
        //
        // **Per-object state reset on zone change** (CR 400.7) —
        // tapped / summoning_sick / damage / counters reset on the
        // way to graveyard. In practice a card in hand already has
        // defaults; the reset is kept for schema-uniformity with
        // every other zone-move arm in this file (ReturnToHand /
        // ExileTarget / Sacrifice / DiscardCard / CounterTargetSpell
        // / DestroyTarget all apply the same reset).
        //
        // **amount<0 raises** loudly (defense-in-depth;
        // parse_effect catches at register time). **amount=0 is a
        // legal no-op** (mirrors GainLife / DrawCards / Mill /
        // ScryN / CreateToken / SearchLibrary / PutPlusOneCounterOnSelf
        // zero-amount conventions).
        //
        // **Inlining note (cross-file friction).** Per the recurring
        // per-slice file-isolation pattern (DealDamage / ReturnToHand
        // / Mill / DrawCards / ExileTarget / Sacrifice / DestroyTarget
        // / TapTarget / UntapTarget / AddCounters / DiscardCard /
        // CreateToken / AddManaToPool /
        // SearchLibraryForBasicLandTapped / PutPlusOneCounterOnSelf
        // / CounterTargetSpell /
        // ReturnCreatureFromYourGraveyardToBattlefield all inline
        // their primitives), this arm inlines the minimal
        // `Arc::make_mut` + `Vec::remove(0)` + push-to-graveyard
        // here.
        Effect::DiscardSelf { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DiscardSelf amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            // Locate the source's controller. Walk both players'
            // five tracked zones (hand / library / battlefield /
            // graveyard / exile). The source may have already moved
            // away from where it was when the effect was queued —
            // the controller is constant across zone changes so any
            // matching obj.id resolves the question.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            // Stack scan: the source may still be resolving (its
            // own stack item triggered this effect). `StackItem`
            // carries an authoritative `controller` field — use
            // it when the zone scan didn't find the source.
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found anywhere: silent fizzle
            };
            let controller_idx = controller as usize;
            // Up to `amount` discards in a row. Stop early if the
            // hand goes empty mid-loop (Python parity with the
            // `if not p.hand: break` short-circuit inside the
            // amount-range loop in `DiscardSelf.resolve`).
            let target_amount = *amount as usize;
            for _ in 0..target_amount {
                if state.players[controller_idx].hand.is_empty() {
                    break;
                }
                // Pop hand[0] (deterministic first-slice choice).
                let hand_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[controller_idx].hand);
                let mut obj = hand_mut.remove(0);
                // CR 400.7 zone-change reset.
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                Arc::make_mut(&mut state.players[controller_idx].graveyard)
                    .push(obj);
                // **MMM Pin 2 (11th Rust slice).** Dispatch
                // `Event::DiscardsCard` ONCE PER DISCARDED CARD on
                // the controlling player (the discarding player for
                // DiscardSelf). Per-card firing matches CR 701.8a
                // semantics ("each card discarded") and mirrors
                // Python's IS_DISCARD dispatch shape. Subject =
                // PLAYER_SUBJECT_OFFSET + controller.
                let discarding_player_subject =
                    crate::cards::PLAYER_SUBJECT_OFFSET + controller as u32;
                dispatch_event(
                    state,
                    crate::cards::Event::DiscardsCard,
                    discarding_player_subject,
                )?;
            }
        }
        // Round-28 coverage. Mirror of
        // ReturnCreatureFromYourGraveyardToBattlefield but destination
        // is hand. Same first-creature-in-graveyard target selection;
        // same CR 400.7 zone-change reset (tapped/damage/counters
        // zeroed). summoning_sick set to true per the r28 CE1
        // universal-true default (gameplay-irrelevant in hand zone).
        //
        // **Controller resolution (I82 follow-up to the
        // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
        // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77 / I78 /
        // I79 / I80 / I81 controller-of-effect substrate).**
        // "Your graveyard" in oracle text means the source-
        // controller's graveyard, NOT the active player's
        // graveyard. CR 119.5 / 608.2 — an effect resolves under
        // its source's controller; "your" pronouns in oracle text
        // are bound to that controller, not to whichever player
        // happens to have priority at resolve time.
        //
        // **Why the fix.** Pre-fix this arm computed
        // `controller_idx = state.active_player as usize`, used
        // for both the source graveyard scan AND the destination
        // hand push. Wrong when the resolving spell / triggered
        // ability has a controller != active_player. Example:
        // on p0's turn, p1 casts a Bone-Plate-Skeleton-shape
        // instant — pre-fix it scanned p0's graveyard for a
        // creature and returned that creature to p0's hand
        // (active_player=0); post-fix it correctly scans p1's
        // graveyard and returns the creature to p1's hand.
        // Mirrors Python's
        // `ReturnFromGraveyard.resolve(state, controller_id, targets)`
        // at `effects/generic/graveyard.py:195` and its
        // `_move_one(state, controller_id, card)` helper, which
        // both pin the "your graveyard"/"your hand" zones to
        // `state.player_by_id(controller_id)`.
        //
        // Sibling closure to I81 (CreateToken — positive-
        // controller arm). Same transient-slot lookup preamble.
        Effect::ReturnCreatureFromYourGraveyardToHand => {
            let _ = source_obj_id;
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(());
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[controller_idx].hand).push(obj);
        }
        // **Phase 5 rider-widened return-to-hand: artifact-or-
        // creature variant.** Mirrors the bare unit-variant arm
        // above exactly, with one filter delta: the disjunction
        // `Creature OR Artifact` (vs the unit variant's
        // `Creature` only).
        //
        // **Controller-of-effect.** Inherits the I82 contract:
        // `controller_idx` derives from
        // `current_resolve_source_controller`, falling back to
        // `active_player` for direct apply_effect_pub callers. The
        // graveyard scan and hand push both anchor to the source's
        // controller — cross-controller routing (p1's ETB resolving
        // on p0's turn) Just Works.
        //
        // **Disjunction shape.** The legal-target filter widens
        // from `card_types.contains("Creature")` to
        // `card_types.contains("Creature") ||
        //  card_types.contains("Artifact")`. Artifact-creatures
        // (printings carrying BOTH types) match at the first
        // branch and are pulled.
        //
        // **Real EoE consumer.** Interceptor Mechan (EoE):
        // "When this creature enters, return target artifact or
        // creature card from your graveyard to your hand."
        Effect::ReturnArtifactOrCreatureFromYourGraveyardToHand => {
            let _ = source_obj_id;
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature")
                            || printing
                                .card_types
                                .iter()
                                .any(|t| t == "Artifact") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(());
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[controller_idx].hand).push(obj);
        }
        // **Phase 5 rider-widened return-to-hand: creature-or-
        // Spacecraft variant.** Sibling to the artifact-or-
        // creature arm. Filter delta: the disjunction is
        // `Creature OR subtype Spacecraft` — Spacecraft is a
        // SUBTYPE (artifact subtype), not a card type, so the
        // second branch reads `printing.subtypes`, not
        // `printing.card_types`. Mirrors Python's
        // `printing.is_spacecraft` which reads
        // `"Spacecraft" in subtypes` off the Subtypes substrate.
        //
        // **Effective-subtypes scope.** Same caveat as the other
        // subtype-reading sites in the Rust codebase (see
        // `subject_has_subtype` at cards.rs:2141): reads the
        // static `printing.subtypes` directly. No effective_-
        // subtypes layer (CR 613 layer 4) — type-modifying
        // continuous effects that would grant "Spacecraft" to
        // a non-Spacecraft printing aren't observable here. In
        // the EoE pool, the cards a player would want to recur
        // are PRINTED Spacecraft, so the static read is correct
        // for the consumer set.
        //
        // **Real EoE consumer.** Fell Gravship (EoE) suffix:
        // "...then return a creature or Spacecraft card from
        // your graveyard to your hand." (The leading mill-3 is
        // composed elsewhere in the trigger's effect sequence;
        // this arm is the leaf return-effect substrate.)
        Effect::ReturnCreatureOrSpacecraftFromYourGraveyardToHand => {
            let _ = source_obj_id;
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature")
                            || printing
                                .subtypes
                                .iter()
                                .any(|s| s == "Spacecraft") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(());
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[controller_idx].hand).push(obj);
        }
        // **Phase 5 rider-widened return-to-hand: enchantment-or-
        // creature variant.** Sibling to the artifact-or-creature
        // and creature-or-spacecraft arms. Filter delta: the
        // disjunction is `Creature OR Enchantment` — both branches
        // read off `printing.card_types` (Enchantment is a CARD
        // TYPE, not a subtype, so this differs from the Spacecraft
        // arm which reads `printing.subtypes` for the second
        // branch).
        //
        // **Controller-of-effect.** Inherits the I82 contract:
        // `controller_idx` derives from
        // `current_resolve_source_controller`, falling back to
        // `active_player` for direct apply_effect_pub callers. The
        // graveyard scan and hand push both anchor to the source's
        // controller — cross-controller routing (p1's ETB resolving
        // on p0's turn) Just Works.
        //
        // **Disjunction shape.** The legal-target filter widens
        // from `card_types.contains("Creature")` to
        // `card_types.contains("Creature") ||
        //  card_types.contains("Enchantment")`. Enchantment-
        // creatures (printings carrying BOTH types — the EoE
        // pool's "Enchantment Creature — ..." printings) match at
        // the first branch and are pulled.
        //
        // **EoE consumer (audit).** Rescue Skiff (EoE) is the
        // unique EoE source carrying the "creature or enchantment
        // card from your graveyard" rider, but its destination is
        // the BATTLEFIELD, not the hand — Rescue Skiff itself
        // unblocks via the to-battlefield sibling that lands as a
        // separate slice. This to-hand variant is the direct
        // sibling of the existing `or_artifact` / `or_spacecraft`
        // to-hand arms and lands here for substrate symmetry: as
        // soon as a real-EoE consumer of "creature or enchantment
        // card from your graveyard to your hand" appears in a
        // future set on the audit, the bridge is already in place.
        Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToHand => {
            let _ = source_obj_id;
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature")
                            || printing
                                .card_types
                                .iter()
                                .any(|t| t == "Enchantment") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(());
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[controller_idx].hand).push(obj);
        }
        // **scour-for-scrap-cycle-2 (2026-05-21) — Scour for Scrap mode
        // 1 leaf.** Return target artifact card from the source-
        // controller's graveyard to that player's hand. Mirrors
        // Python's `ReturnTargetArtifactCardFromYourGraveyardToHand`
        // (`effects/generic/bounce.py:338`, a `ReturnFromGraveyard`
        // subclass pinning `artifact=True`, `destination="hand"`,
        // `target_kind="graveyard_artifact"`).
        //
        // **Cast-time-bound target via
        // `current_resolve_payload_target_id`.** Per the established
        // cast-time-bound-target pattern (SeedshipImpactEffect at
        // `state_ops_resolve.rs:7271`, etc.), this arm reads the
        // pre-bound target id and re-validates at resolve time per
        // CR 608.2b. The validation matches the cast-time per-mode
        // detector's filter exactly: the target must still live in
        // the controller's graveyard AND still be an artifact card
        // (defense-in-depth — the printing is immutable, but a
        // future continuous type-changing effect could in theory
        // strip Artifact). If the target left the graveyard (e.g.
        // another effect moved it) or no longer matches the filter,
        // silent fizzle — byte-equal with Python's "target became
        // illegal → mode silently does nothing" semantics from
        // `ModalChooseOne._resolve_or_both`
        // (`combinators.py:1290-1302` — picks the first remaining
        // target whose value appears in mode.legal_targets;
        // `None` => `modal fizzle: mode {idx} requires a legal
        // target`).
        //
        // **Cycle-1 substrate fallback.** When `current_resolve_
        // payload_target_id` is `None` (e.g. a direct
        // apply_effect_pub call from a synthetic test that didn't
        // route through the cast-time binder), fall back to the
        // deterministic-first-legal scan from the controller's
        // graveyard. Mirrors the sibling
        // `ReturnArtifactOrCreatureFromYourGraveyardToHand`'s
        // self-scan behavior (`state_ops_resolve.rs:5357`) — safe
        // back-compat for the substrate-only paths.
        //
        // **Controller / zone-change reset.** Identical contract to
        // the sibling return-to-hand variants: controller_idx from
        // `current_resolve_source_controller`; on move to hand,
        // reset `tapped`, `summoning_sick=true`, `damage=0`,
        // `counters=0`, `attached_to=None`, `exiled_with_source=
        // None`. The `summoning_sick=true` matches the round-28
        // CE1 universal-true default for hand-zone objects.
        Effect::ReturnTargetArtifactCardFromYourGraveyardToHand => {
            let _ = source_obj_id;
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            // Re-validation predicate shared by both code paths:
            // graveyard object's printing must contain "Artifact" in
            // its card_types.
            let is_legal_artifact_in_gy = |state: &GameState, id: u32| -> bool {
                state.players[controller_idx]
                    .graveyard
                    .iter()
                    .any(|o| {
                        if o.id != id {
                            return false;
                        }
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types.iter().any(|t| t == "Artifact")
                            })
                            .unwrap_or(false)
                    })
            };
            let target_obj_id: Option<u32> =
                match state.current_resolve_payload_target_id {
                    Some(payload_id) => {
                        if is_legal_artifact_in_gy(state, payload_id) {
                            Some(payload_id)
                        } else {
                            // CR 608.2b: silent fizzle when the
                            // bound target left the graveyard or is
                            // no longer an artifact card. Mirrors
                            // Python's modal-fizzle path for an
                            // illegal-at-resolve mode target.
                            None
                        }
                    }
                    None => {
                        // Substrate fallback — deterministic-first
                        // legal scan from the controller's
                        // graveyard. Used by direct apply_effect_pub
                        // callers (synthetic tests) that bypass the
                        // cast-time per-mode binder.
                        state.players[controller_idx]
                            .graveyard
                            .iter()
                            .find_map(|obj| {
                                match state.registry.lookup(obj.printing_id) {
                                    Some(printing)
                                        if printing
                                            .card_types
                                            .iter()
                                            .any(|t| t == "Artifact") =>
                                    {
                                        Some(obj.id)
                                    }
                                    _ => None,
                                }
                            })
                    }
                };
            let Some(target_id) = target_obj_id else {
                return Ok(());
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[controller_idx].hand).push(obj);
        }
        // **Controller resolution (I83 follow-up to the
        // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
        // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77 / I78 /
        // I79 / I80 / I81 / I82 controller-of-effect substrate).**
        // Sibling closure to I82 (ReturnCreatureFromYourGraveyard-
        // ToHand) — same "your graveyard" reanimation shape;
        // differs only in destination zone (battlefield vs hand).
        // "Your" pronouns bind to the source's controller, not
        // the active player. CR 119.5 / 608.2.
        //
        // **Why the fix.** Pre-fix this arm computed
        // `controller_idx = state.active_player as usize`, used
        // for the source graveyard scan, the post-find Vec
        // position re-scan, the planeswalker starting-loyalty
        // pre-read, the graveyard-pop, AND the destination
        // battlefield push. Wrong when the resolving spell /
        // triggered ability has a controller != active_player.
        // Example: on p0's turn, p1 casts a Zombify-shape
        // sorcery — pre-fix it scanned p0's graveyard for a
        // creature and reanimated it on p0's battlefield under
        // p0's control (active_player=0); post-fix it correctly
        // scans p1's graveyard and reanimates on p1's battlefield
        // under p1's control. Mirrors Python's
        // `ReturnFromGraveyard.resolve(state, controller_id,
        // targets)` at `effects/generic/graveyard.py:195` and
        // its `_move_one(state, controller_id, card)` helper,
        // which both pin "your graveyard"/"your battlefield"
        // zones to `state.player_by_id(controller_id)`. The
        // reanimated permanent's GameObject.controller field
        // (preserved as the existing obj.controller from the
        // pre-pop state) is left unchanged by this fix — the
        // CR-correct semantics here are that the reanimated
        // permanent enters under the reanimation effect's
        // controller, which is what the post-fix
        // controller_idx now correctly resolves to.
        //
        // Same transient-slot lookup preamble as I59-I82; the
        // controller_idx value cascades into all five readers
        // (scan, position re-scan, walker pre-read, graveyard-
        // pop, bf-push). The ETB-trigger dispatch
        // (`pending_etb_dispatches`) sees the reanimated id
        // regardless of recipient bf, so the trigger fan-out
        // shape is unaffected.
        Effect::ReturnCreatureFromYourGraveyardToBattlefield => {
            let _ = source_obj_id; // controller-targeted, not source-relative
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            // Find first creature in controller's graveyard (front =
            // smallest Vec index = top of the iteration order). Drop
            // registry borrow before mutating.
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(()); // empty / no-creature: silent fizzle
            };
            // Inline the move (mirrors CounterTargetSpell / Sacrifice /
            // ExileTarget inlining — apply_effect's free-fn signature
            // makes cross-file primitive call awkward). Move from
            // graveyard to battlefield; per round-17 conv:
            // summoning_sick=true, tapped=false, damage=0, counters=0,
            // zone=ZONE_BATTLEFIELD.
            // CE4 closure: reserve ETB timestamp before player-mut borrow.
            let etb_ts = crate::state::reserve_etb_timestamp(state);
            // CE20 Slice 5 (Walker loyalty — starting_loyalty ETB
            // seeding). Pre-read the printing's walker shape on the
            // immutable registry BEFORE the player-mut borrow.
            let starting_loyalty: i32 = {
                let pos_gy = state.players[controller_idx]
                    .graveyard
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was just found in graveyard");
                let pid = state.players[controller_idx].graveyard[pos_gy].printing_id;
                match state.registry.lookup(pid) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Planeswalker") =>
                    {
                        printing.starting_loyalty
                    }
                    _ => 0,
                }
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_BATTLEFIELD;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            // CE20 Slice 5: walker reanimation — reset loyalty to
            // printed starting value per CR 613.x (the new object is
            // distinct from the previous one and starts fresh).
            obj.loyalty_counters = starting_loyalty;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            obj.etb_timestamp = etb_ts;
            // I83 controller-of-effect fix: stamp the reanimated
            // permanent's `controller` byte with the
            // reanimation-effect's controller. CR 110.2 / 614.x —
            // when an object enters the battlefield from another
            // zone, its controller becomes the player whose effect
            // moved it (or the player who put it there). For the
            // typical "return target creature card from YOUR
            // graveyard" reanimation, that's the source-controller
            // we just resolved. Pre-fix this byte was left as the
            // pre-move value (owner-keyed in graveyard, which
            // happened to match `state.active_player` only when
            // controller == active_player and the original owner
            // was the active player). Post-fix, the new
            // battlefield permanent is correctly controlled by the
            // reanimating effect's controller regardless of where
            // the card came from.
            obj.controller = controller_idx as u8;
            let entered_id = obj.id;
            Arc::make_mut(&mut state.players[controller_idx].battlefield).push(obj);
            // ETB-trigger Slice 2 (RS3): queue ENTERS dispatch for
            // the reanimated creature. Drained by the surrounding
            // resolver after the resolving spell's stack item pops.
            state.pending_etb_dispatches.push(entered_id);
        }
        // **Phase 5 rider-widened return-to-battlefield: enchantment-
        // or-creature variant.** Sibling to the bare to-bf unit
        // variant immediately above. Filter delta: the disjunction
        // is `Creature OR Enchantment` — both branches read off
        // `printing.card_types` (Enchantment is a CARD TYPE, not a
        // subtype). Same first-creature-in-graveyard scan order
        // (front = oldest by push), same I83 controller-of-effect
        // contract (`controller_idx` from
        // `current_resolve_source_controller`, fallback to
        // `active_player`), same CR 400.7 zone-change reset, same
        // ETB-trigger fan-out via `pending_etb_dispatches`.
        //
        // **EoE consumer.** Rescue Skiff (EoE): "When this Spacecraft
        // enters, return target creature or enchantment card from
        // your graveyard to the battlefield." Unlike the to-hand
        // sibling variant which has no real-EoE consumer (Rescue
        // Skiff is to-bf, not to-hand), THIS arm is the direct
        // unblocker for Rescue Skiff itself.
        //
        // **Disjunction shape.** The legal-target filter widens
        // from `card_types.contains("Creature")` to
        // `card_types.contains("Creature") ||
        //  card_types.contains("Enchantment")`. Enchantment-
        // creatures (printings carrying BOTH types — the EoE
        // pool's "Enchantment Creature — ..." printings) match at
        // the first branch and are pulled.
        //
        // **Walker reseed (CE20 Slice 5).** Same pre-read of
        // `starting_loyalty` on the registry BEFORE the player-mut
        // borrow as the bare to-bf arm; reseeds `loyalty_counters`
        // on the reanimated permanent. Real practice: an
        // Enchantment-Planeswalker would be admitted by both
        // branches of the disjunction (Creature primary or
        // Enchantment widening); the walker reseed correctly fires.
        //
        // **I83 controller stamp.** The reanimated permanent's
        // `controller` byte stamps to `controller_idx` (the
        // reanimation-effect's controller per CR 110.2 / 614.x),
        // mirroring the bare to-bf arm. On p0's turn, p1's Rescue-
        // Skiff ETB scans p1's graveyard, reanimates onto p1's
        // battlefield, and the new permanent is correctly p1-
        // controlled.
        Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield => {
            let _ = source_obj_id; // controller-targeted, not source-relative
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            // Find first creature OR enchantment in controller's
            // graveyard (front = smallest Vec index = top of the
            // iteration order). Drop registry borrow before mutating.
            let target_obj_id: Option<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .find_map(|obj| match state.registry.lookup(obj.printing_id) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Creature")
                            || printing
                                .card_types
                                .iter()
                                .any(|t| t == "Enchantment") =>
                    {
                        Some(obj.id)
                    }
                    _ => None,
                });
            let Some(target_id) = target_obj_id else {
                return Ok(()); // empty / no-match: silent fizzle
            };
            // CE4 closure: reserve ETB timestamp before player-mut borrow.
            let etb_ts = crate::state::reserve_etb_timestamp(state);
            // CE20 Slice 5 (Walker loyalty — starting_loyalty ETB
            // seeding). Pre-read the printing's walker shape on the
            // immutable registry BEFORE the player-mut borrow.
            let starting_loyalty: i32 = {
                let pos_gy = state.players[controller_idx]
                    .graveyard
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id was just found in graveyard");
                let pid = state.players[controller_idx].graveyard[pos_gy].printing_id;
                match state.registry.lookup(pid) {
                    Some(printing)
                        if printing.card_types.iter().any(|t| t == "Planeswalker") =>
                    {
                        printing.starting_loyalty
                    }
                    _ => 0,
                }
            };
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let pos = graveyard_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in graveyard but not on re-scan");
            let mut obj = graveyard_mut.remove(pos);
            obj.zone = ZONE_BATTLEFIELD;
            obj.tapped = false;
            obj.summoning_sick = true;
            obj.damage = 0;
            obj.counters = 0;
            obj.loyalty_counters = starting_loyalty;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            obj.etb_timestamp = etb_ts;
            // I83 controller-of-effect stamp: reanimated permanent's
            // `controller` byte = the reanimation-effect's controller
            // (CR 110.2 / 614.x). Mirrors the bare to-bf arm.
            obj.controller = controller_idx as u8;
            let entered_id = obj.id;
            Arc::make_mut(&mut state.players[controller_idx].battlefield).push(obj);
            // ETB-trigger fan-out: queue ENTERS dispatch for the
            // reanimated permanent. Drained by the surrounding
            // resolver after the resolving spell's stack item pops.
            state.pending_etb_dispatches.push(entered_id);
        }
        // Round-26: EachOpponentLosesLife. CR 118.4 — "life loss"
        // (not damage). Each opponent of the SOURCE's controller
        // loses `amount` life. Punisher / drawback pattern
        // (Balemurk Leech shape) — 6 cards in the EoE pool per
        // round-20 measurement.
        //
        // **Distinct from round-17 DamageEachOpponent.** DamageEach-
        // Opponent deals DAMAGE (CR 701.4) — routes through damage-
        // prevention / Worship-style replacement effects (CR 614)
        // when those land. EachOpponentLosesLife is direct life
        // subtraction (CR 118.4) and is NOT reducible. Two-player
        // engine first slice: both arms have identical end-state for
        // the defender's life total because neither variant routes
        // through damage-prevention yet, but the variants stay
        // distinct to mirror Python's closed `Effect` vocabulary
        // (round-15 pattern). Future slices that wire damage-
        // prevention through DealDamage / DamageEachOpponent keep
        // this arm a direct subtraction.
        //
        // **Source-finder dispatch (round-24 pattern).** Walks both
        // players' five tracked zones (hand / library / battlefield
        // / graveyard / exile) plus the stack to recover the
        // source's `controller`. Mirrors DiscardSelf — punisher
        // effects fire on whoever controls the source regardless of
        // where the source currently sits, since the controller
        // field doesn't change on zone moves.
        //
        // **Two-player "each opponent" reduces to "the one opponent".**
        // opponent_id = 1 - controller. Direct life subtraction on
        // `players[opponent_idx].life`; negative life handled by
        // the SBA loop (round-5+), no clamp here.
        //
        // **amount<0 raises** loudly (defense-in-depth; parse_effect
        // catches at register time). **amount=0 is a legal no-op**
        // (mirrors GainLife / DrawCards / Mill / DamageEachOpponent /
        // DiscardSelf zero-amount conventions). **Source not found
        // anywhere = silent fizzle** (lost-track-of-source tolerance,
        // mirrors DiscardSelf shape).
        Effect::EachOpponentLosesLife { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "EachOpponentLosesLife amount must be non-negative, got {amount}"
                )));
            }
            // Source-controller resolution mirrors round-24
            // DiscardSelf's 5-zone walk + stack scan.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Each opponent: the OTHER player in 2-player land.
            let opponent_id = 1 - controller;
            let opponent_idx = opponent_id as usize;
            // Direct life subtraction — not damage. Negative life is
            // handled by SBA loop.
            state.players[opponent_idx].life =
                state.players[opponent_idx].life.saturating_sub(*amount);
        }
        // BLOCKED-RUST-EFFECT slice (round 44) -- each opponent of the
        // source-controller loses life equal to the source-controller's
        // artifact count. Sister of round-26's
        // `Effect::EachOpponentLosesLife { amount }` (same recipient set
        // + apply step) and round-32's
        // `DealDamageToOpponentCreatureByArtifactCount` / round-34's
        // `SurveilXEqualToArtifactCount` (same artifact-count
        // denominator). FIFTH apply substrate to pick up the X-from-
        // count plumbing established by round-32 (damage), round-33
        // (counters), round-34 (surveil), round-42 (mill).
        //
        // **Direct life subtraction (CR 118.4) -- NOT damage.** Mirrors
        // round-26 EachOpponentLosesLife exactly: routes around damage-
        // prevention / Worship-style replacement effects (CR 614).
        // Two-player engine: 2-player "each opponent" reduces to "the
        // one opponent" at `1 - controller`.
        //
        // **Artifact denominator.** Walk the source-controller's bf
        // once; for each obj count +1 if `printing.card_types` contains
        // "Artifact". BIT-FOR-BIT mirror of the round-32 / round-34
        // artifact-count walking pattern -- same scan, same predicate,
        // same `saturating_add` overflow guard. Lands that are also
        // artifacts count, tokens like Lander count. Dual-typed
        // artifact-creatures count for BOTH the artifact-count and the
        // creatures-controlled denominators independently. The source
        // itself counts toward the denominator if it is an Artifact
        // still on the bf at resolve time (mirrors Python's vanilla
        // walk over `state.battlefield` -- no source-exclusion clause).
        //
        // **Source-finder dispatch (round-26 / round-32 sister
        // pattern).** 5-zone walk + stack scan. Source not found
        // anywhere = silent fizzle (lost-track-of-source tolerance,
        // mirrors EachOpponentLosesLife / DiscardSelf shape).
        //
        // **Zero-count = silent no-op** (matches Python's `if
        // artifact_count <= 0: return` short-circuit before the
        // per-opponent loop; also mirrors GainLife / DrawCards / Mill /
        // EachOpponentLosesLife / DiscardSelf zero-amount conventions).
        Effect::EachOpponentLosesLifeEqualToArtifactCount => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-26 EachOpponentLosesLife sister shape
            // exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_eollac: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_eollac;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Count source-controller's artifacts on bf. Mirrors
            // Python's `sum(1 for o in state.battlefield if
            // o.controller_id == controller_id and o.printing.is_artifact)`
            // shape. Walks the round-32 artifact-count loop
            // BIT-FOR-BIT.
            let cidx = controller as usize;
            let mut amount: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Artifact") {
                    amount = amount.saturating_add(1);
                }
            }
            if amount <= 0 {
                return Ok(()); // no artifacts → silent no-op (matches Python's `if artifact_count <= 0: return`)
            }
            // Each opponent: the OTHER player in 2-player land.
            let opponent_id = 1 - controller;
            let opponent_idx = opponent_id as usize;
            // Direct life subtraction -- not damage. Negative life is
            // handled by SBA loop. Mirrors round-26 sister BIT-FOR-BIT.
            state.players[opponent_idx].life =
                state.players[opponent_idx].life.saturating_sub(amount);
        }
        // BLOCKED-RUST-EFFECT slice (round 48) -- each opponent of the
        // source-controller loses life equal to the source-controller's
        // CURRENT LIFE TOTAL. Sister of round-44's
        // `Effect::EachOpponentLosesLifeEqualToArtifactCount` (SAME
        // recipient set + apply step + apply substrate) and round-42's
        // `MillTargetPlayerEqualToYourLifeTotal` / round-47's
        // `DealDamageToOpponentCreatureByLifeTotal` (SAME life-total
        // denominator on OTHER substrates). SECOND denominator on the
        // life-loss-each-opponent apply substrate; walks the
        // X-from-count matrix to a third (substrate, life-total) cell.
        //
        // **Direct life subtraction (CR 118.4) -- NOT damage.** Mirrors
        // round-44 EachOpponentLosesLifeEqualToArtifactCount exactly:
        // routes around damage-prevention / Worship-style replacement
        // effects (CR 614). Two-player engine: "each opponent" reduces
        // to "the one opponent" at `1 - controller`.
        //
        // **Life-total denominator.** Read the source-controller's
        // current life total and clamp negative life to 0
        // (`life.max(0)`) -- BIT-FOR-BIT mirror of the round-42
        // `MillTargetPlayerEqualToYourLifeTotal` life clamp and Python's
        // `max(0, life)` shape. The round-44 sister walks the bf for an
        // artifact count; this arm reads the life total instead -- the
        // ONLY mechanical difference between the two arms (same source-
        // finder, same apply, same zero-no-op).
        //
        // **Source-finder dispatch (round-44 sister pattern).** 5-zone
        // walk + stack scan. Source not found anywhere = silent fizzle
        // (lost-track-of-source tolerance, mirrors
        // EachOpponentLosesLife / EachOpponentLosesLifeEqualToArtifactCount
        // / DiscardSelf shape). Uses the 5-zone walk rather than
        // `current_resolve_source_controller` for cross-arm consistency
        // with the round-44 sister and so the inject-and-fire-ENTERS
        // synthetic harness resolves correctly.
        //
        // **Zero-life = silent no-op** (matches Python's `if amount <=
        // 0: return` short-circuit before the per-opponent loop; also
        // mirrors GainLife / DrawCards / Mill / EachOpponentLosesLife /
        // EachOpponentLosesLifeEqualToArtifactCount / DiscardSelf
        // zero-amount conventions).
        Effect::EachOpponentLosesLifeEqualToYourLifeTotal => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-44 EachOpponentLosesLifeEqualToArtifactCount
            // sister shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_eollyl: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_eollyl;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Life-total denominator: clamp negative life to 0
            // (matches Python's `max(0, life)` clamp and the round-42
            // MillTargetPlayerEqualToYourLifeTotal precedent).
            let cidx = controller as usize;
            let amount: i32 = state.players[cidx].life.max(0);
            if amount <= 0 {
                return Ok(()); // life <= 0 -> silent no-op
            }
            // Each opponent: the OTHER player in 2-player land.
            let opponent_id = 1 - controller;
            let opponent_idx = opponent_id as usize;
            // Direct life subtraction -- not damage. Negative life is
            // handled by SBA loop. Mirrors round-44 sister BIT-FOR-BIT.
            state.players[opponent_idx].life =
                state.players[opponent_idx].life.saturating_sub(amount);
        }
        // BLOCKED-RUST-EFFECT slice (round 45) -- the source's
        // controller GAINS life equal to the source-controller's
        // artifact count. Sister of round-26's `Effect::GainLife
        // { amount }` (same recipient = controller-self + same apply
        // step) and round-32's
        // `DealDamageToOpponentCreatureByArtifactCount` / round-34's
        // `SurveilXEqualToArtifactCount` / round-44's
        // `EachOpponentLosesLifeEqualToArtifactCount` (same
        // artifact-count denominator). SIXTH apply substrate to pick
        // up the X-from-count plumbing established by round-32
        // (damage), round-33 (counters), round-34 (surveil), round-42
        // (mill), round-44 (life-loss-each-opponent). Controller-self
        // mirror of round-44: same denominator, inverted recipient
        // (controller-self vs opponents-of-controller) AND inverted
        // sign (gain vs loss).
        //
        // **Direct life addition (CR 119.3) -- NOT damage healing.**
        // Mirrors round-26 GainLife exactly: simple `+=` (not
        // `saturating_add`, matching the round-26 sister bit-for-bit;
        // life total is i32 so realistic gain values won't overflow).
        // Routes around damage-prevention hooks (CR 119 distinguishes
        // life gain from damage).
        //
        // **Artifact denominator.** Walk the source-controller's bf
        // once; for each obj count +1 if `printing.card_types` contains
        // "Artifact". BIT-FOR-BIT mirror of the round-32 / round-34 /
        // round-44 artifact-count walking pattern -- same scan, same
        // predicate, same `saturating_add` overflow guard. Lands that
        // are also artifacts count, tokens like Lander count. Dual-
        // typed artifact-creatures count for BOTH the artifact-count
        // and the creatures-controlled denominators independently.
        // The source itself counts toward the denominator if it is an
        // Artifact still on the bf at resolve time (mirrors Python's
        // vanilla walk over `state.battlefield` -- no source-exclusion
        // clause).
        //
        // **Source-finder dispatch (round-26 / round-32 / round-44
        // sister pattern).** 5-zone walk + stack scan. Source not
        // found anywhere = silent fizzle (lost-track-of-source
        // tolerance, mirrors EachOpponentLosesLife / DiscardSelf
        // shape).
        //
        // **Zero-count = silent no-op** (matches Python's `if
        // artifact_count <= 0: return` short-circuit before the
        // life-add; also mirrors GainLife / DrawCards / Mill /
        // EachOpponentLosesLife / DiscardSelf zero-amount conventions).
        //
        // **Why not lean on `current_resolve_source_controller` like
        // bare GainLife.** Bare `Effect::GainLife { amount }` uses
        // the resolve-harness-set `current_resolve_source_controller`
        // shortcut because its denominator is a static field; the
        // dynamic-X arm needs the 5-zone walk because the count must
        // be derived from the source's ACTUAL controller's bf even
        // when the resolve harness's shortcut isn't set (e.g. inject-
        // and-fire-ENTERS synthetic harnesses, plus consistency with
        // the round-44 sister source-finder).
        Effect::GainLifeEqualToArtifactCount => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-26 EachOpponentLosesLife sister and
            // round-44 EachOpponentLosesLifeEqualToArtifactCount sister
            // shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_glac: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_glac;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Count source-controller's artifacts on bf. Mirrors
            // Python's `sum(1 for o in state.battlefield if
            // o.controller_id == controller_id and o.printing.is_artifact)`
            // shape. Walks the round-32 / round-44 artifact-count loop
            // BIT-FOR-BIT.
            let cidx = controller as usize;
            let mut amount: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Artifact") {
                    amount = amount.saturating_add(1);
                }
            }
            if amount <= 0 {
                return Ok(()); // no artifacts → silent no-op (matches Python's `if artifact_count <= 0: return`)
            }
            // Direct life addition -- not damage. Mirrors round-26
            // GainLife sister BIT-FOR-BIT (simple `+=`, not
            // saturating_add).
            state.players[cidx].life += amount;
        }
        // BLOCKED-RUST-EFFECT slice (round 49) -- the source's
        // controller GAINS life equal to the source-controller's
        // CURRENT LIFE TOTAL. Sister of round-45's
        // `Effect::GainLifeEqualToArtifactCount` (SAME recipient =
        // controller-self + SAME apply step: direct life addition via
        // `+=`, mirrors round-26 GainLife bit-for-bit, AND the SAME
        // apply substrate: gain-life-controller-self), only the
        // DENOMINATOR read differs (source-controller's current life
        // total instead of their artifact-count walk). SECOND
        // denominator on the gain-life-controller-self apply substrate
        // (artifact-count round-45 -> life-total round-49). Life-total
        // denominator established by round-42's
        // `MillTargetPlayerEqualToYourLifeTotal` (mill), round-47's
        // `DealDamageToOpponentCreatureByLifeTotal` (damage), round-48's
        // `EachOpponentLosesLifeEqualToYourLifeTotal` (life-loss-each-
        // opponent) on OTHER substrates -- this walks the X-from-count
        // matrix to a FOURTH (substrate, life-total) cell.
        //
        // **Life-total clamp.** Clamp negative life to 0 (`life.max(0)`)
        // before use as the denominator -- matches Python's
        // `max(0, life)` and the round-42 / round-47 / round-48
        // life-total clamping precedent (a negative life total can
        // transiently arise from a prior loss/damage in the same
        // resolution window before the SBA loop runs; a negative
        // denominator would be nonsensical). Zero clamped life ->
        // silent no-op (matches Python's `if amount <= 0: return`
        // short-circuit before the life-add; also mirrors GainLife /
        // DrawCards / Mill / EachOpponentLosesLife /
        // GainLifeEqualToArtifactCount / DiscardSelf zero-amount
        // conventions).
        //
        // **Source-finder dispatch (round-45 sister pattern).** 5-zone
        // walk + stack scan. Source not found anywhere = silent fizzle
        // (lost-track-of-source tolerance, mirrors
        // GainLifeEqualToArtifactCount / EachOpponentLosesLife /
        // DiscardSelf shape). The dynamic-X arm needs the 5-zone walk
        // (not the `current_resolve_source_controller` shortcut bare
        // GainLife uses) because the life-total denominator must read
        // from the source's ACTUAL controller even when the resolve
        // harness's shortcut isn't set (e.g. inject-and-fire-ENTERS
        // synthetic harnesses, plus consistency with the round-45/48
        // sister source-finders).
        Effect::GainLifeEqualToYourLifeTotal => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-45 GainLifeEqualToArtifactCount sister and
            // round-48 EachOpponentLosesLifeEqualToYourLifeTotal sister
            // shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_glyl: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_glyl;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Life-total denominator: clamp negative life to 0
            // (matches Python's `max(0, life)` clamp and the round-42
            // mill / round-47 damage / round-48 each-opponent-loses
            // life-total clamping precedent).
            let cidx = controller as usize;
            let amount: i32 = state.players[cidx].life.max(0);
            if amount <= 0 {
                return Ok(()); // life <= 0 -> silent no-op
            }
            // Direct life addition -- not damage. Mirrors round-45
            // GainLifeEqualToArtifactCount sister BIT-FOR-BIT (simple
            // `+=`, not saturating_add), only the amount source differs.
            state.players[cidx].life += amount;
        }
        // LoseLife: CR 118.4 — "life loss" (not damage). The SOURCE's
        // controller loses `amount` life. Punisher / drawback pattern
        // (Decode Transmissions shape) — 7 cards in the EoE pool per
        // round-20 measurement.
        //
        // **Distinct from round-26 EachOpponentLosesLife.** Same source-
        // finder pattern (5-zone walk + stack scan), but the recipient
        // is the source's CONTROLLER (typically the caster paying a
        // self-inflicted life cost) — NOT the controller's opponent.
        // Inverted target on an otherwise identical dispatch shape.
        //
        // **Source-finder dispatch (round-24 pattern).** Walks both
        // players' five tracked zones (hand / library / battlefield /
        // graveyard / exile) plus the stack to recover the source's
        // `controller`. Mirrors EachOpponentLosesLife / DiscardSelf —
        // punisher effects fire on whoever controls the source
        // regardless of where the source currently sits, since the
        // `controller` field doesn't change on zone moves.
        //
        // **Direct life subtraction.** Negative life is handled by the
        // SBA loop (round-5+); no clamp here beyond the saturating_sub
        // that prevents underflow on the i32.
        //
        // **amount<0 raises** loudly (defense-in-depth; parse_effect
        // catches at register time). **amount=0 is a legal no-op**
        // (mirrors GainLife / DrawCards / Mill / EachOpponentLosesLife
        // / DiscardSelf zero-amount conventions). **Source not found
        // anywhere = silent fizzle** (lost-track-of-source tolerance,
        // mirrors EachOpponentLosesLife / DiscardSelf shape).
        Effect::LoseLife { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "LoseLife amount must be non-negative, got {amount}"
                )));
            }
            // Source-controller resolution mirrors round-24
            // EachOpponentLosesLife / DiscardSelf 5-zone walk + stack
            // scan.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            let controller_idx = controller as usize;
            // CONTROLLER (not opponent) loses life — distinguishes from
            // round-26 EachOpponentLosesLife.
            state.players[controller_idx].life =
                state.players[controller_idx].life.saturating_sub(*amount);
        }
        // TargetOpponentLosesLife: CR 118.4 — "life loss" (not damage).
        // Target opponent loses `amount` life. 2 cards in EoE pool per
        // round-20 measurement (Dubious Delicacy etc.).
        //
        // **Distinct from EachOpponentLosesLife.** That arm hits ALL
        // opponents (in 2-player land = the one opponent).
        // TargetOpponentLosesLife is targeted — the caster CHOOSES which
        // opponent loses life. In a 2-player engine these collapse to
        // the same runtime effect, but the variants stay distinct to
        // mirror Python's closed Effect vocabulary (targeted vs
        // each-opponent semantics — round-15 pattern).
        //
        // **First-slice deterministic choice**: pick the one opponent
        // (= 1 - controller). Multiplayer / archenemy work needs an
        // explicit choice callback; deferred.
        //
        // **Source-finder dispatch (round-24 pattern).** Walks both
        // players' five tracked zones (hand / library / battlefield /
        // graveyard / exile) plus the stack to recover the source's
        // `controller`. Mirrors EachOpponentLosesLife / LoseLife /
        // DiscardSelf — punisher / targeted effects fire on whoever
        // controls the source regardless of where the source currently
        // sits, since the `controller` field doesn't change on zone
        // moves.
        //
        // **amount<0 raises** loudly (defense-in-depth; parse_effect
        // catches at register time). **amount=0 is a legal no-op**
        // (mirrors GainLife / DrawCards / Mill / EachOpponentLosesLife
        // / LoseLife zero-amount conventions). **Source not found
        // anywhere = silent fizzle** (lost-track-of-source tolerance,
        // mirrors EachOpponentLosesLife / LoseLife shape).
        Effect::TargetOpponentLosesLife { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "TargetOpponentLosesLife amount must be non-negative, got {amount}"
                )));
            }
            // Source-controller resolution (mirrors round-24 /
            // round-25): 5-zone walk + stack scan.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // 2-player: "target opponent" = the one opponent of the
            // controller. Identical runtime behavior to
            // EachOpponentLosesLife in 2-player land.
            let opponent_id = 1 - controller;
            let opponent_idx = opponent_id as usize;
            state.players[opponent_idx].life =
                state.players[opponent_idx].life.saturating_sub(*amount);
        }
        // **PPP Pin 1 (10th Rust slice).** TargetPlayerLosesLife: the
        // PRE-FIXED target player loses `amount` life. Distinct from
        // TargetOpponentLosesLife — that arm auto-picks the defending
        // opponent at resolve time; this arm reads the pre-wired
        // target player id from the payload-target seat.
        //
        // **First-slice resolve path.** Read
        // `current_resolve_payload_target_id` — the dispatch site
        // (e.g. the DISCARDS_CARD `fixed_targets` plumbing for
        // Entropic Battlecruiser's 1+ tier) sets this to the player
        // id BEFORE the trigger resolves. Convention is the same
        // small-id-namespace used by PutPoisonCountersOnTargetPlayer
        // (8th Rust slice, JJJ Pin 2): values 0 or 1 are player ids
        // in the 2-player engine.
        //
        // **No-target observable.** Unset payload-target (None) →
        // silent fizzle. Mirrors Python's TargetPlayerLosesLife.resolve
        // (life.py:90) which early-returns when `targets` is empty.
        // This is byte-equal with the canonical "no target supplied"
        // no-op; production end-to-end use depends on the
        // DISCARDS_CARD `fixed_targets` dispatch path being wired,
        // which lands in a follow-on slice.
        //
        // **amount<0 raises**; **amount=0 is a legal no-op**.
        Effect::TargetPlayerLosesLife { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "TargetPlayerLosesLife amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            // Read the pre-fixed target player id from the payload
            // seat. Unset → no target supplied → silent fizzle.
            let payload_target = match state.current_resolve_payload_target_id {
                Some(t) => t,
                None => return Ok(()),
            };
            // Player ids are 0 or 1 in the 2-player engine; the seat
            // is set to `pid as u32` by the dispatch site (mirrors
            // PutPoisonCountersOnTargetPlayer's seat convention).
            let pid = payload_target as u8;
            let player_idx_opt = state
                .players
                .iter()
                .position(|p| p.id == pid);
            let Some(pidx) = player_idx_opt else {
                return Ok(()); // unknown pid → silent fizzle
            };
            state.players[pidx].life =
                state.players[pidx].life.saturating_sub(*amount);
        }
        // EachOpponentDiscards: CR 701.8 — "discard a card." Each
        // opponent of the SOURCE's controller discards `count` cards.
        // Let's Play a Game pattern — 2 cards in the EoE pool per
        // round-20 measurement. Mirrors Python's
        // `EachOpponentDiscards(count: int = 1)` dataclass.
        //
        // **Distinct from round-15 DiscardCard { target: AnyPlayer }.**
        // DiscardCard's AnyPlayer arm targets the DEFENDING player
        // (active player's spell forces the opponent to discard, Mind
        // Rot / caller-targeted shape). EachOpponentDiscards fires on
        // the SOURCE's controller's opponents — source-relative,
        // independent of active_player at apply-effect time. In a
        // 2-player engine both collapse to "the one opponent discards
        // hand[0]" but the variants stay distinct to mirror Python's
        // closed Effect vocabulary (round-15 pattern).
        //
        // **Distinct from round-24 DiscardSelf.** Same source-finder +
        // hand[0]-pop primitive, but DiscardSelf fires on the
        // CONTROLLER while this arm fires on the OPPONENT of the
        // controller (= 1 - controller in 2-player land). Inverted
        // target on otherwise identical dispatch shape — composes
        // round-24 EachOpponentLosesLife's opponent-resolution with
        // round-24 DiscardSelf's discard primitive.
        //
        // **Source-finder dispatch (round-24 pattern).** Walks both
        // players' five tracked zones (hand / library / battlefield /
        // graveyard / exile) plus the stack to recover the source's
        // `controller`. Mirrors DiscardSelf / EachOpponentLosesLife —
        // punisher / source-relative effects fire on whoever controls
        // the source regardless of where the source currently sits,
        // since the `controller` field doesn't change on zone moves.
        //
        // **First-slice deterministic choice**: opponent discards
        // hand[0]. Mirrors round-15 DiscardCard + round-24 DiscardSelf
        // hand[0] convention. Up to `count` discards in a row; stop
        // early on empty hand (Python parity with `for _ in
        // range(self.count): if not p.hand: break` inside
        // `EachOpponentDiscards.resolve`).
        //
        // **Per-object state reset on zone change** (CR 400.7) —
        // tapped / summoning_sick / damage / counters reset on the way
        // to graveyard. Schema-uniform with every other zone-move arm.
        //
        // **count<0 raises** loudly (defense-in-depth; parse_effect
        // catches at register time). **count=0 is a legal no-op**
        // (mirrors DiscardSelf / GainLife / DrawCards / Mill zero-
        // count conventions). **Empty opponent hand = silent no-op**
        // (Python parity with `if not p.hand: break`). **Source not
        // found anywhere = silent fizzle** (lost-track-of-source
        // tolerance, mirrors round-24 / round-25 / round-26 shape).
        Effect::EachOpponentDiscards { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "EachOpponentDiscards count must be non-negative, got {count}"
                )));
            }
            if *count == 0 {
                return Ok(());
            }
            // Source-controller resolution (mirrors round-24 /
            // round-25 / round-26 shape): 5-zone walk + stack scan.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // 2-player "each opponent" reduces to "the one opponent".
            let opponent_idx = (1 - controller) as usize;
            let target_count = *count as usize;
            for _ in 0..target_count {
                if state.players[opponent_idx].hand.is_empty() {
                    break; // empty hand: silent no-op (Python parity)
                }
                // Pop hand[0] of the opponent (deterministic first-
                // slice choice; mirrors round-15 DiscardCard +
                // round-24 DiscardSelf hand[0] convention).
                let hand_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[opponent_idx].hand);
                let mut obj = hand_mut.remove(0);
                // CR 400.7 zone-change reset (mirrors DiscardSelf).
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                Arc::make_mut(&mut state.players[opponent_idx].graveyard)
                    .push(obj);
                // **MMM Pin 2 (11th Rust slice).** Dispatch
                // `Event::DiscardsCard` ONCE PER DISCARDED CARD on
                // the discarding opponent. Same per-card firing
                // convention as `Effect::DiscardSelf`. Subject =
                // PLAYER_SUBJECT_OFFSET + opponent_idx.
                let discarding_player_subject =
                    crate::cards::PLAYER_SUBJECT_OFFSET + opponent_idx as u32;
                dispatch_event(
                    state,
                    crate::cards::Event::DiscardsCard,
                    discarding_player_subject,
                )?;
            }
        }
        // Round-24: DestroyAllCreatures. CR 701.7 + 701.4 AOE — Wrath
        // of God pattern. Destroys every creature on the battlefield
        // (both players), honoring indestructible (CR 702.12) inline.
        // **bf-ETB iteration (I33 closure).** The victim-collection
        // walk uses the shared `crate::bf_iter::bf_iter_etb_sorted`
        // helper so destroyable creatures are enumerated in Python's
        // centralized `state.battlefield` (ETB-monotone) order rather
        // than p0-then-p1 per-player order. Outcome is order-invariant:
        // each destroyable creature ends up in its OWN controller's
        // graveyard regardless of visit order, and per-controller
        // graveyard insertion order is preserved (Python's centralized
        // walk filtered to a single controller produces the same
        // per-controller sequence as the per-player walk did). The
        // change is pre-emptive: visit order becomes load-bearing for
        // future death-trigger emission (CE8 substrate). See I33 in
        // `docs/RUST_PARITY_GAPS.md`.
        Effect::DestroyAllCreatures => {
            let _ = source_obj_id; // implicit-target AOE; source not relevant
            // Collect (controller_idx, obj_id) for every destroyable
            // Creature on either bf in bf-ETB-timestamp order, then
            // pop from each controller's bf and push into their
            // graveyard.
            let destroyable: Vec<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .filter_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if !printing.card_types.iter().any(|t| t == "Creature") {
                        return None;
                    }
                    if printing.keywords.iter().any(|k| k == "indestructible") {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                })
                .collect();
            for (pidx, target_id) in destroyable {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = match bf_mut.iter().position(|o| o.id == target_id) {
                    Some(p) => p,
                    None => continue, // defensively skip — should never miss
                };
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object. Python's `turn.move_off_battlefield`
                // (turn.py:231-240) clears combat marks but
                // deliberately leaves `+1/+1` counters intact; the SBA
                // destroy path (`state_ops_sba.rs` ~:417) and the
                // closed `DepressurizeEffect` arm both follow that
                // observable ("counters retained on the dead object" —
                // the established byte-equal contract). The prior stray
                // `obj.counters = 0;` here diverged from BOTH Python
                // (SoT) and Rust's own SBA destroy path; removed
                // (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let leaving_printing_id = obj.printing_id;
                let leaving_obj_id = obj.id;
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Slice 11 (DestroyAllCreatures LTB residual). Mirror
                // the 13 single-target Effect-arm bf-exit sites wired
                // by ETB Slice 10: route the void-flag stamping
                // through `mark_left_battlefield_if_nonland` (so the
                // CE13 / I29 invariant lands the same way on the
                // mass-destroy path as on every single-target destroy)
                // AND push the leaving id onto pending_leaves_dispatches
                // so LTB triggers fire on a Wrath. Pre-Slice-11 this
                // arm directly set `permanent_left_battlefield_this_turn`
                // and queued NO LTB triggers — silent on a mass
                // destroy, the 14th LTB-silent site documented in
                // ETB Slice 10's report.
                //
                // **Ordering**: the per-victim drain is deferred to
                // `resolve_top_of_stack`'s
                // `drain_pending_leaves_dispatches` call AFTER the
                // resolving spell's stack pop. Mirrors the Slice 10
                // defer-past-pop pattern — draining inline here would
                // push LTB triggers ABOVE the still-present resolving
                // spell and the subsequent stack.pop() would pop the
                // wrong slot. The queue's natural FIFO + LIFO stack-
                // resolve order land LTB triggers in
                // bf-ETB-timestamp-collection order (oldest victim's
                // LTB resolves LAST since LIFO stack pops most-recent-
                // pushed first).
                mark_left_battlefield_if_nonland(state, leaving_printing_id);
                state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // **Beyond-the-Quiet slice (2026-05-21).** Exile every creature
        // AND every Spacecraft on either bf (the multi-type predicate;
        // Lane T Spacecraft-subtype finding). Mirrors Python's
        // `ExileAllCreaturesAndSpacecraft.resolve` at
        // `effects/generic/exile.py:697`: walks every bf permanent,
        // filters by `card_types ∋ "Creature" || subtypes ∋ "Spacecraft"`,
        // routes each victim through `move_off_battlefield(destination=
        // "exile")`. NO indestructible skip (exile bypasses
        // indestructible per CR 702.12); NO DIES dispatch (CR 700.4 —
        // dies = put into graveyard from bf, not exile). The bf-ETB
        // iteration uses `crate::bf_iter::bf_iter_etb_sorted` to match
        // Python's centralized ETB-monotone walk. Per-victim LEAVES +
        // token-cease are queued; the drain runs in
        // `resolve_top_of_stack` AFTER the resolving spell's stack pop
        // (Slice-11 defer-past-pop pattern, mirrored from
        // `DestroyAllCreatures`).
        Effect::ExileAllCreaturesAndSpacecraft => {
            let _ = source_obj_id; // implicit-target AOE; source not relevant
            // Collect (controller_idx, obj_id) for every bf permanent
            // that is a Creature OR a Spacecraft, in bf-ETB-timestamp
            // order. No indestructible skip — exile bypasses
            // indestructible (CR 702.12).
            let victims: Vec<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .filter_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    let is_creature =
                        printing.card_types.iter().any(|t| t == "Creature");
                    // Spacecraft is a SUBTYPE per Python's
                    // `is_spacecraft = "Spacecraft" in subtypes`
                    // (cards/base.py:1056-1057). NOT a card type —
                    // Spacecraft printings carry "Artifact" today, but
                    // the filter mirrors Python's pure-subtype check
                    // (so a hypothetical non-Artifact Spacecraft still
                    // matches). Lane T finding.
                    let is_spacecraft =
                        printing.subtypes.iter().any(|s| s == "Spacecraft");
                    if !(is_creature || is_spacecraft) {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                })
                .collect();
            for (pidx, target_id) in victims {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = match bf_mut.iter().position(|o| o.id == target_id) {
                    Some(p) => p,
                    None => continue, // defensively skip — should never miss
                };
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_EXILE;
                obj.tapped = false;
                // CE13: `summoning_sick` is NOT reset (Python's
                // `move_off_battlefield` deliberately omits the clear;
                // sickness is battlefield-only state per CR 302.1).
                obj.damage = 0;
                // Mirrors the `ExileTarget` per-object reset (counters
                // cleared on the way to exile — distinct from the
                // `DestroyAllCreatures` graveyard path where counters
                // are retained for the SBA-destroy observable). Exile
                // is a zone change; the receiving zone is a NEW object
                // per CR 400.7.
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let leaving_printing_id = obj.printing_id;
                let leaving_obj_id = obj.id;
                // Push onto owner's exile zone. First-slice convention
                // (mirrors `DestroyAllCreatures` / `ExileTarget`):
                // owner == controller, so "owner's exile" is the
                // current controller's exile.
                Arc::make_mut(&mut state.players[pidx].exile).push(obj);
                // Mirror the `DestroyAllCreatures` Slice-11 LTB +
                // token-cease deferred-drain pattern. Each victim
                // queues LEAVES + token-cease unconditionally; the
                // per-victim drain is deferred to
                // `resolve_top_of_stack`'s
                // `drain_pending_leaves_dispatches` /
                // `drain_pending_token_cease` calls AFTER the
                // resolving spell's stack pop (draining inline here
                // would push LTB triggers ABOVE the still-present
                // resolving spell and the subsequent stack.pop() would
                // pop the wrong slot).
                mark_left_battlefield_if_nonland(state, leaving_printing_id);
                state.pending_leaves_dispatches.push(leaving_obj_id);
                crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
                // NO DIES dispatch — exile is a zone change to exile,
                // NOT to graveyard (CR 700.4). Python's
                // `move_off_battlefield` gates dies triggers on
                // `destination == "graveyard"` (`turn.py:253`); this
                // arm matches that gate by omitting the dies push.
            }
        }
        // DestroyTargetNonlandPermanent. CR 701.7 with non-land filter
        // — Drag to the Roots / Drill Too Deep pattern. Walks BOTH
        // players' battlefield in ETB-timestamp order via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper, picking the
        // first non-Land, non-indestructible permanent. Mirrors
        // Python's `effects/generic/destroy.py:DestroyTargetNonlandPermanent`
        // `TargetSpec(~IsLand() & ~IsPlayer())` filter against the
        // centralized `state.battlefield` walk (no controller
        // restriction). See invariant I39.
        Effect::DestroyTargetNonlandPermanent => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Land") {
                        return None;
                    }
                    if printing.keywords.iter().any(|k| k == "indestructible") {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                });
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Round-29 Void LTB: nonland target by filter; helper
                // re-checks via printing for defense-in-depth.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // DestroyTargetNoncreaturePermanent (round 52). CR 701.7 with a
        // NONCREATURE filter — Extinguisher Battleship ETB head. Walks
        // BOTH players' battlefield in ETB-timestamp order via the
        // shared `crate::bf_iter::bf_iter_etb_sorted` helper, picking
        // the first non-Creature, non-indestructible permanent. Mirrors
        // Python's
        // `effects/generic/destroy.py:DestroyTargetNoncreaturePermanent`
        // `TargetSpec(~IsCreature() & ~IsPlayer())` filter against the
        // centralized `state.battlefield` walk (no controller
        // restriction). Identical dispatch + LTB-fanout shape to the
        // round-24 `DestroyTargetNonlandPermanent` arm above with the
        // filter predicate swapped Land→Creature. Players are
        // structurally excluded (the bf walk only iterates
        // GameObjects). See invariant I39.
        Effect::DestroyTargetNoncreaturePermanent => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        return None;
                    }
                    if printing.keywords.iter().any(|k| k == "indestructible") {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                });
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Round-29 Void LTB: helper re-checks via printing for
                // defense-in-depth. (A noncreature destroy never moves
                // a creature off bf, but the helper short-circuits on
                // creature printings anyway — same call as the nonland
                // arm for fan-out symmetry.)
                mark_left_battlefield_if_nonland(state, printing_id);
                // Slice 10: LTB fan-out at the Effect arm — same
                // deferred-drain rationale as the nonland arm above
                // (bf-exit bypasses move_object; queue the leaving id,
                // the drain runs in `resolve_top_of_stack` after the
                // resolving slot pops).
                state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // DestroyTargetArtifact. CR 701.7 with Artifact filter — Drill
        // Too Deep / Scour for Scrap pattern. Walks BOTH players'
        // battlefield in ETB-timestamp order via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper, picking the
        // first Artifact that is not indestructible. Mirrors
        // Python's `effects/generic/destroy.py:DestroyTargetArtifact`
        // `TargetSpec(IsArtifact())` filter against the centralized
        // `state.battlefield` walk (no controller restriction). See
        // invariant I39.
        Effect::DestroyTargetArtifact => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if !printing.card_types.iter().any(|t| t == "Artifact") {
                        return None; // not an artifact
                    }
                    if printing.keywords.iter().any(|k| k == "indestructible") {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                });
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Round-29 Void LTB: artifact target may also be a
                // land (treasure tokens / artifact lands). Helper gates
                // on the printing.
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // DestroyTargetArtifactOrTappedCreature (Effect-uncovered
        // Tier-A — Radiant Strike inner half). CR 701.7 with a typed-OR
        // filter: (Artifact) OR (Creature AND tapped). Walks BOTH
        // players' battlefield in ETB-timestamp order via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper, picking the
        // first permanent that is either an Artifact OR a tapped
        // Creature and is not indestructible. Mirrors Python's
        // `effects/generic/destroy.py:475
        // DestroyTargetArtifactOrTappedCreature` whose `TargetSpec`
        // filter is `IsArtifact() | (IsCreature() & IsTapped())`,
        // re-verified at resolution per CR 608.2b against the
        // centralized `state.battlefield` walk (no controller
        // restriction). `obj.tapped` is read off the scanned live
        // GameObject — the round-53 effective-state lesson (read where
        // the real card would; the engine has no continuous-effect
        // tap-state layer, so `obj.tapped` IS the effective tapped
        // state both engines agree on). Identical dispatch/LTB-fanout
        // shape to the round-25 `DestroyTargetArtifact` arm above with
        // the predicate widened Artifact → Artifact-OR-tapped-Creature.
        // The `Sequence`+`GainLife` outer shell of Radiant Strike's
        // full effect runs through the already-live `Effect::Sequence`
        // / `Effect::GainLife` arms. See invariant I39.
        Effect::DestroyTargetArtifactOrTappedCreature => {
            // **Battlefield STATE_EQ axis closure (todo `e5b7c9d8`).**
            // Migrated to the `current_resolve_payload_target_id`
            // pattern (SeedshipImpactEffect / DepressurizeEffect
            // precedent). Pre-fix this arm self-scanned for the first
            // Artifact-OR-tapped-Creature on either bf — diverging
            // from Python whenever the cast-time-bound target ceased
            // between cast and resolve (CR 608.2b — Python re-checks
            // the ORIGINAL target id and fizzles; Rust pre-fix
            // re-targeted the next matching permanent).
            //
            // Repro (arch=253 seed=2 step 1147 / `docs/BATTLEFIELD_
            // AXIS_INVESTIGATION.md` §2): P0 attacks with Beamsaw
            // Prospector id=0 (tapped). P1 casts Radiant Strike on
            // Beamsaw, then Focus Fire on Beamsaw. Focus Fire
            // resolves first (LIFO top), kills Beamsaw, dies-trigger
            // creates Lander id=120 (Artifact). Now Radiant Strike
            // resolves: Python re-checks bound target id=0
            // (Beamsaw), finds it in graveyard → fizzle. Rust pre-fix
            // scanned bf, picked Lander (first Artifact) and
            // destroyed it. Post-fix: Rust reads
            // `current_resolve_payload_target_id == Some(0)`, scans
            // for id=0 on bf, finds None → fizzle. Byte-equal.
            //
            // **None-payload synthetic-test fallback.** The cross-
            // engine real-EoE probe + the spell-card-bridge use
            // `cast_spell_noncreature` which binds the target at
            // cast time. The low-level synthetic-test `push_to_stack`
            // API does NOT seat a target. For that path — and ONLY
            // that path — fall back to the pre-fix deterministic-
            // first scan: identical observable to prior behavior on
            // every synthetic pin, no risk of re-introducing the bug
            // because the bug requires a NON-None bound target whose
            // target ceased to be legal. Documented in
            // `docs/SEARCH_LIMITATIONS.md` (battlefield STATE_EQ
            // axis closure entry).
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                let is_artifact = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Artifact");
                                let is_creature = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature");
                                // Typed-OR filter re-checked at
                                // resolve (defense-in-depth + the
                                // CR 608.2b "still matches the
                                // typed filter" leg).
                                if !(is_artifact
                                    || (is_creature && o.tapped))
                                {
                                    return None;
                                }
                                if printing
                                    .keywords
                                    .iter()
                                    .any(|k| k == "indestructible")
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        let printing =
                            state.registry.lookup(obj.printing_id)?;
                        let is_artifact = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_creature = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        if !(is_artifact || (is_creature && obj.tapped)) {
                            return None;
                        }
                        if printing
                            .keywords
                            .iter()
                            .any(|k| k == "indestructible")
                        {
                            return None;
                        }
                        Some((obj.controller as usize, obj.id))
                    }),
            };
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Round-29 Void LTB: target may be an artifact land or
                // a creature; helper gates on the printing.
                mark_left_battlefield_if_nonland(state, printing_id);
                // R76-2: a "destroy" of a CREATURE is a death (CR
                // 700.4) — its `Event.DIES` triggered abilities MUST
                // fire, exactly as the generic SBA destroy path does
                // (`state_ops_sba.rs`) and as Python's
                // `DestroyTargetArtifactOrTappedCreature.resolve` ->
                // `turn.destroy` -> `move_off_battlefield` does
                // (`_fire_dies_triggers` + the predicate-path
                // `dispatch_event(DIES)` BEFORE
                // `_fire_leaves_battlefield_triggers`). This
                // hand-rolled destroy arm (Radiant Strike's destroy
                // half) previously pushed ONLY
                // `pending_leaves_dispatches`, so a destroyed
                // dies-trigger creature (e.g. Gravpack Monoist ->
                // Robot token) silently never fired its trigger — the
                // round-76 bench-resweep R76-2 divergence (same class
                // as R74-2 `f38df833`, a DIFFERENT Effect arm; that
                // fix added the DIES drain at the 4 resolution sites
                // but did NOT touch this arm). Python (SoT) correct;
                // Rust was wrong. Push order mirrors the SBA path +
                // the `Event::LeavesBattlefield` cards.rs doc-comment
                // contract (DIES first via `pending_dies_dispatches`,
                // LEAVES second via `pending_leaves_dispatches`); the
                // matching resolution-path drain order (DIES drained
                // then LEAVES; LIFO -> LEAVES resolves first)
                // preserves Python's `move_off_battlefield` sequence.
                // Creature-gated like the SBA path / the R74-2
                // DepressurizeEffect fix so a destroyed ARTIFACT (the
                // OR-filter's other leg) does NOT fire a spurious DIES
                // (CR 700.4 — only a creature death fires DIES).
                let victim_is_creature = state
                    .registry
                    .lookup(printing_id)
                    .map(|p| {
                        p.card_types
                            .iter()
                            .any(|t| t == "Creature")
                    })
                    .unwrap_or(false);
                if victim_is_creature {
                    state
                        .pending_dies_dispatches
                        .push(leaving_obj_id);
                }
                // Slice 10: LTB fan-out at the Effect arm — same
                // deferred-drain rationale as the artifact arm above
                // (bf-exit bypasses move_object; queue the leaving id,
                // the drain runs in `resolve_top_of_stack` after the
                // resolving slot pops).
                state.pending_leaves_dispatches.push(leaving_obj_id);
                // Round-55: token-cease sibling of the LEAVES push. The
                // id is queued unconditionally; the is_token gate lives
                // in drain_pending_token_cease (CR 111.7 / 704.5d).
                crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // SeedshipImpactEffect (Effect-uncovered Tier-A — Seedship
        // Impact, Lander-rider). CR 701.7 destroy with an
        // Artifact-OR-Enchantment filter + a conditional Lander rider.
        // Mirrors Python's `SeedshipImpactEffect.resolve`
        // (`effects/bespoke.py:1237`): pick the first
        // (Artifact OR "Enchantment" in card_types) permanent; read
        // its PRINTED mana value BEFORE destroy; destroy it (skip
        // indestructible — the scan filter); if it was actually
        // destroyed AND its mv ≤ 2, create one Lander token under the
        // source's controller. Destroy-half clones the round-25/65
        // `DestroyTargetArtifact` / `DestroyTargetArtifactOrTapped
        // Creature` body verbatim with the filter swapped to
        // Artifact-OR-Enchantment. The Lander-create block is
        // re-derived privately (file-iso) from the `Effect::Create
        // Token` arm — registry-bounds check, fresh-id allocator,
        // source-controller seat, fresh etb_timestamp, ENTERS
        // dispatch queue. `token_printing_id` is the Lander's Rust
        // registry index (carried on the variant; Python's class is
        // field-free and uses TOKEN_PRINTINGS["Lander"] intrinsically
        // — the pid is dropped at the bridge). See invariant I39 /
        // I81 (source-controller token seat).
        //
        // **§2 indestructible parity**: the scan SKIPS indestructible
        // permanents (the shipped DestroyTargetArtifact precedent).
        // Python's `legal_targets` includes indestructible artifacts/
        // enchantments but `destroy()` returns False for them →
        // early-return, NO Lander consolation. An indestructible-ONLY
        // board reaches the SAME observable on both engines (no
        // destroy, no Lander). The probe pins single-legal-target
        // boards so the deterministic-first pick is unambiguous; a
        // mixed indestructible+destroyable board is not exercised
        // (different first-pick conventions — out of probe scope, as
        // with the Radiant Strike precedent).
        Effect::SeedshipImpactEffect { token_printing_id } => {
            let lander_pid: u32 = *token_printing_id;
            // Registry-bounds check up front (mirrors the CreateToken
            // arm's whole-or-nothing contract) — surface an unknown
            // pid before any state mutation.
            if state.registry.lookup(lander_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "SeedshipImpactEffect token_printing_id \
                     {lander_pid} not in registry"
                )));
            }
            // G7 lander-controller-mismatch closure (token-creation
            // parity 3x — arch=10/11/12 seed=2). Read the cast-time-
            // bound target via `current_resolve_payload_target_id`
            // (set by the stack resolver around the apply_effect loop
            // at `state_ops_resolve.rs:12416-12417`; the spell-card-
            // bridge lever from round-70). The pre-fix self-scan
            // picked the first artifact-or-enchantment on the bf at
            // RESOLVE time, which diverges from Python whenever the
            // bound target ceases between cast and resolve (Python
            // fizzles per the `obj is None` early-return at
            // `effects/bespoke.py:1251-1259`). The S6 verify-first
            // repro: P0 casts Orbital Plunge (Lander rider creates
            // Lander0 under P0), P0 casts Seedship Impact targeting
            // Lander0, P1 casts Seedship Impact targeting Lander0.
            // LIFO resolves P1's first: destroys Lander0, mv≤2 → P1
            // mints Lander1 under P1 (correct). Then P0's resolves:
            // Python fizzles (bound Lander0 is gone); Rust pre-fix
            // self-scanned, picked Lander1, destroyed it and minted
            // Lander2 under P0 — diverges from Python by both
            // controller AND id. CR 608.2b: a targeted spell whose
            // only target has become illegal is removed from the
            // stack without effect; cast-time bind is the SoT.
            // `cast_spell_noncreature` now binds the
            // artifact-or-enchantment target at cast time so the
            // payload reaches this arm filled.
            let payload_target = match state.current_resolve_payload_target_id {
                Some(t) => t,
                None => return Ok(()),
            };
            // Resolve-time re-legality: find the target on EITHER bf
            // and re-check the (Artifact OR Enchantment) +
            // non-indestructible filter. Same find_map shape as the
            // pre-fix self-scan; only the candidate pool narrows from
            // "all bf" to "the bound id". Mirrors Python's
            // `obj = next((o for o in state.battlefield if o.id ==
            // target_id), None)` + the post-find filter check at
            // `effects/bespoke.py:1251-1259`. If the target left the
            // bf or no longer matches the filter (e.g. became
            // indestructible via a UEOT modifier — defense-in-depth
            // since the printing is immutable), silent fizzle —
            // byte-equal with Python's `obj is None` / filter-fail
            // early-returns. Capture the PRINTED mana value at scan
            // time (read BEFORE destroy — Python reads
            // `obj.printing.mana_cost.total` before `destroy(...)`).
            let target: Option<(usize, u32, u32)> =
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        if obj.id != payload_target {
                            return None;
                        }
                        let printing =
                            state.registry.lookup(obj.printing_id)?;
                        let is_artifact = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_enchantment = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Enchantment");
                        if !(is_artifact || is_enchantment) {
                            return None;
                        }
                        if printing
                            .keywords
                            .iter()
                            .any(|k| k == "indestructible")
                        {
                            return None;
                        }
                        // Printed mana value = generic + all colored
                        // pips (Rust ManaCost has no hybrid/x field;
                        // the registry flattens hybrid at parse time —
                        // matches Python `mana_cost.total` for
                        // non-hybrid costs; the registry-level
                        // hybrid-flatten is the existing documented
                        // simplification, not new here).
                        let mc = &printing.mana_cost;
                        let mv: u32 = mc.generic as u32
                            + mc.white as u32
                            + mc.blue as u32
                            + mc.black as u32
                            + mc.red as u32
                            + mc.green as u32;
                        Some((obj.controller as usize, obj.id, mv))
                    });
            if let Some((pidx, target_id, mv)) = target {
                // ---- Destroy half (verbatim DestroyTargetArtifact
                // remove→graveyard + LTB + token-cease tail).
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here
                // (CR 302.1 — battlefield-only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                //
                // **Aura-attachment substrate — cast-time auto-resolve
                // seat alignment follow-up (Tractor Beam closure,
                // 2026-05-20).** Snapshot `attached_to` BEFORE we
                // clear it so the aura-host-control revert helper can
                // see the prior host. Mirrors Python's
                // `_recompute_obj_controller` cache refresh after
                // `move_off_battlefield` drops the control_change
                // entry — without this, a Seedship-Impact-destroyed
                // Tractor Beam leaves its host stuck under the
                // caster's control (surfaced by arch14/s4/step2256
                // STATE_EQ asymmetry).
                let prior_attached_to = obj.attached_to;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) =
                    (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard)
                    .push(obj);
                crate::state_ops_sba::revert_aura_host_control_on_leave(
                    state,
                    leaving_obj_id,
                    printing_id,
                    prior_attached_to,
                );
                // Round-29 Void LTB: target is artifact/enchantment;
                // helper gates on the printing.
                mark_left_battlefield_if_nonland(state, printing_id);
                // R76-2 (audit of the verbatim-clone sibling — the
                // bench-resweep explicitly flagged this destroy-half
                // for the same omission). Python's `SeedshipImpact
                // Effect.resolve` calls the generic `turn.destroy`,
                // which fires `Event.DIES` for a CREATURE death (CR
                // 603 / 700.4). Seedship's filter is Artifact OR
                // Enchantment, but a dual-typed **Artifact Creature**
                // is admissible AND a creature — destroying it IS a
                // creature death and MUST fire its DIES trigger
                // byte-equal with Python. The verbatim destroy-clone
                // previously pushed ONLY LEAVES (same R76-2 class as
                // the Radiant Strike `DestroyTargetArtifactOrTapped
                // Creature` arm above). Creature-gated so a pure
                // (non-creature) artifact/enchantment does NOT fire a
                // spurious DIES. DIES pushed BEFORE LEAVES (the SBA /
                // R74-2 push order; LIFO -> LEAVES resolves first),
                // drained by the existing R74-2
                // `drain_pending_dies_dispatches` at the 4 resolution
                // sites.
                let victim_is_creature = state
                    .registry
                    .lookup(printing_id)
                    .map(|p| {
                        p.card_types
                            .iter()
                            .any(|t| t == "Creature")
                    })
                    .unwrap_or(false);
                if victim_is_creature {
                    state
                        .pending_dies_dispatches
                        .push(leaving_obj_id);
                }
                // Slice 10: LTB fan-out at the Effect arm.
                state.pending_leaves_dispatches.push(leaving_obj_id);
                // Round-55: token-cease sibling of the LEAVES push.
                crate::state_ops_zones::queue_token_cease(
                    state,
                    leaving_obj_id,
                );

                // ---- Conditional Lander rider: only when the target
                // was actually destroyed (it was — the scan already
                // excluded indestructible, matching Python's
                // `if not destroy(...): return` no-Lander-consolation)
                // AND its printed mv ≤ 2.
                if mv <= 2 {
                    // Source-controller seat (CR 111.4 — token under
                    // the spell's controller, not active_player). The
                    // in-tree stack resolver sets
                    // `current_resolve_source_controller` around its
                    // apply_effect loop (the I81 substrate); fall back
                    // to active_player for direct apply_effect_pub
                    // callers (back-compat).
                    let recipient: u8 = state
                        .current_resolve_source_controller
                        .unwrap_or(state.active_player);
                    // R76-1: draw a fresh id from the monotonic
                    // `next_object_id` counter (CR 111.7), REPLACING
                    // the old non-monotonic `max_id + 1` re-scan that
                    // recycled a ceased token's id. Byte-equal with
                    // Python's `self.next_object_id += 1`.
                    let new_id: u32 =
                        crate::state::reserve_object_id(state);
                    // Fresh etb_timestamp (CE4 closure — strictly
                    // monotonic so the snapshot/fingerprint walk
                    // orders this token after prior ETBs).
                    let tok_etb = state.next_etb_timestamp;
                    state.next_etb_timestamp =
                        state.next_etb_timestamp.saturating_add(1);
                    let bf: &mut Vec<GameObject> = Arc::make_mut(
                        &mut state.players[recipient as usize]
                            .battlefield,
                    );
                    bf.push(GameObject {
                        id: new_id,
                        printing_id: lander_pid,
                        controller: recipient,
                        zone: ZONE_BATTLEFIELD,
                        tapped: false,
                        // Slice 14 (TTT Pin 2 alignment): Lander is a
                        // non-creature (Token Artifact); per CR 302.1
                        // + 612.6 it enters with summoning_sick=false.
                        summoning_sick: false,
                        damage: 0,
                        counters: 0,
                        charge_counters: 0,
                        loyalty_counters: 0,
                        attached_to: None,
                        exiled_with_source: None,
                        stun_counters: 0,
                        etb_timestamp: tok_etb,
                        ueot_modifiers: Vec::new(),
                        warp_exile_pending: false,
                        warped: false,
                        cast_from_exile_with_lightstall_surcharge: false,
                        controller_baseline: None,
                        // Aura-attachment Phase 4 S3 closure
                        // (2026-05-21): Seedship-Impact-shape bespoke
                        // token-mints route through `state.new_object`
                        // on Python, NOT through `CreateToken.resolve`
                        // — so Python skips the Moonlit replacement
                        // check entirely. Mirror that: the override
                        // stays `false`.
                        token_supertype_override: false,
                        // Control-change-UEOT substrate — Cycle 1
                        // (2026-05-21): freshly-minted tokens never
                        // start with a UEOT controller override.
                        controller_override: None,
                    });
                    // ETB-trigger Slice 2 (RS4): queue ENTERS for the
                    // fresh token; drained by the surrounding resolver
                    // after the resolving spell's stack item pops.
                    state.pending_etb_dispatches.push(new_id);
                }
            }
        }
        // EmergencyEjectEffect (Effect-uncovered single-gated tier —
        // Emergency Eject, Lander-rider). CR 701.7 destroy with a
        // non-land permanent filter + an UNCONDITIONAL Lander rider
        // going to the **former controller** of the destroyed
        // permanent. Mirrors Python's
        // `DestroyTargetNonlandPermanentWithLanderForController
        // .resolve` (`effects/generic/destroy.py:611`).
        //
        // **Three deviations from `SeedshipImpactEffect` (same
        // Lander-pid carrier shape):**
        // 1. Filter is non-land permanent (CR 205.2a) — every
        //    battlefield object whose printing.card_types does NOT
        //    contain `"Land"`. Indestructible permanents still
        //    skipped at the scan (same Seedship precedent: Python's
        //    `destroy()` returns False on indestructible → no Lander
        //    consolation).
        // 2. No printed-MV gate — the Lander rider fires whenever
        //    the destroy succeeds (Python: `destroy(state, obj)`
        //    returns True iff actually destroyed; no `if mv <= 2`
        //    branch). MV read entirely omitted.
        // 3. Lander recipient = the **former controller** of the
        //    destroyed permanent (the target's `obj.controller`
        //    field at scan time — captured BEFORE the destroy moves
        //    the object to graveyard). Python:
        //    `former_controller = obj.controller_id` BEFORE
        //    `destroy(state, obj)` → `state.new_object(...,
        //    owner_id=former_controller, ...)` →
        //    `put_into_battlefield(..., former_controller)`. This is
        //    the load-bearing distinguishing feature vs Seedship
        //    Impact (which seats Lander under the SOURCE's
        //    controller). Emergency Eject typically targets an
        //    opponent's permanent, so the opponent gets the Lander
        //    as consolation, NOT the caster.
        //
        // **§2 indestructible parity** (same Seedship precedent):
        // the scan SKIPS indestructible permanents → no destroy →
        // no Lander, byte-equal to Python's `destroy()`-returns-
        // False → early-return.
        Effect::EmergencyEjectEffect { token_printing_id } => {
            let lander_pid: u32 = *token_printing_id;
            // Registry-bounds check up front (mirrors Seedship arm).
            if state.registry.lookup(lander_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "EmergencyEjectEffect token_printing_id \
                     {lander_pid} not in registry"
                )));
            }
            // Read cast-time-bound target via
            // `current_resolve_payload_target_id` (the proven
            // SeedshipImpactEffect / DestroyTargetArtifactOrTapped
            // Creature payload-target swap pattern). When None
            // (synthetic-test `push_to_stack` path without
            // `push_to_stack_with_target`), fall back to the
            // deterministic-first non-land permanent in ETB-timestamp
            // order across BOTH battlefields, mirroring the
            // pre-payload-target Seedship body's self-scan shape so
            // the no-callback synthetic fallback works (same
            // FocusFire / Depressurize / CutPropulsion convention).
            let payload_target_opt =
                state.current_resolve_payload_target_id;
            // Resolve-time re-legality: find target + capture
            // (controller, target_id) BEFORE any mutation. Mirrors
            // Python's `obj = next((o for o in state.battlefield if
            // o.id == target_id), None)` + the post-find filter
            // check at `effects/generic/destroy.py:625-640`. The
            // captured `controller` is the FORMER controller for
            // the Lander recipient — read BEFORE the destroy moves
            // the object to graveyard (where Python's owner-stamp
            // semantics on leave would otherwise overwrite it).
            let target: Option<(usize, u32, u8)> =
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        // If we have a bound target, only the bound
                        // id qualifies; otherwise the deterministic-
                        // first non-land permanent wins.
                        if let Some(pt) = payload_target_opt {
                            if obj.id != pt {
                                return None;
                            }
                        }
                        let printing =
                            state.registry.lookup(obj.printing_id)?;
                        let is_land = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Land");
                        if is_land {
                            return None;
                        }
                        if printing
                            .keywords
                            .iter()
                            .any(|k| k == "indestructible")
                        {
                            return None;
                        }
                        Some((
                            obj.controller as usize,
                            obj.id,
                            obj.controller,
                        ))
                    });
            if let Some((pidx, target_id, former_controller)) = target {
                // ---- Destroy half (clones the proven Seedship
                // remove→graveyard + LTB + token-cease tail).
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                obj.damage = 0;
                // Aura-attachment substrate (Tractor Beam closure):
                // snapshot attached_to BEFORE clearing so the
                // aura-host-control revert helper sees the prior
                // host. Mirrors the Seedship arm's prior_attached_to
                // pattern.
                let prior_attached_to = obj.attached_to;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) =
                    (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard)
                    .push(obj);
                crate::state_ops_sba::revert_aura_host_control_on_leave(
                    state,
                    leaving_obj_id,
                    printing_id,
                    prior_attached_to,
                );
                // Round-29 Void LTB: target is non-land permanent;
                // helper gates on the printing.
                mark_left_battlefield_if_nonland(state, printing_id);
                // R76-2: DIES dispatch for a Creature victim
                // (Python's generic `turn.destroy` fires DIES for a
                // creature death — CR 603 / 700.4). Emergency
                // Eject's filter is non-land permanent, which
                // INCLUDES creatures (the most common target — most
                // EoE creatures are non-land permanents); destroying
                // them IS a creature death and MUST fire its DIES
                // trigger byte-equal with Python. Creature-gated so
                // a destroyed pure (non-creature) artifact /
                // enchantment / planeswalker does NOT fire a
                // spurious DIES. DIES pushed BEFORE LEAVES (the
                // R74-2 push order; LIFO -> LEAVES resolves first).
                let victim_is_creature = state
                    .registry
                    .lookup(printing_id)
                    .map(|p| {
                        p.card_types
                            .iter()
                            .any(|t| t == "Creature")
                    })
                    .unwrap_or(false);
                if victim_is_creature {
                    state
                        .pending_dies_dispatches
                        .push(leaving_obj_id);
                }
                state.pending_leaves_dispatches.push(leaving_obj_id);
                crate::state_ops_zones::queue_token_cease(
                    state,
                    leaving_obj_id,
                );

                // ---- Unconditional Lander rider — goes to FORMER
                // controller of the destroyed permanent (NOT
                // source-controller). The captured `former_controller`
                // is the target's `obj.controller` field at scan
                // time (CR 111.4 — token under the specified
                // controller). Python:
                // `former_controller = obj.controller_id` BEFORE
                // destroy → `state.new_object(...,
                // owner_id=former_controller, ...)` →
                // `put_into_battlefield(state, lander,
                // former_controller)`.
                let recipient: u8 = former_controller;
                let new_id: u32 =
                    crate::state::reserve_object_id(state);
                let tok_etb = state.next_etb_timestamp;
                state.next_etb_timestamp =
                    state.next_etb_timestamp.saturating_add(1);
                let bf: &mut Vec<GameObject> = Arc::make_mut(
                    &mut state.players[recipient as usize]
                        .battlefield,
                );
                bf.push(GameObject {
                    id: new_id,
                    printing_id: lander_pid,
                    controller: recipient,
                    zone: ZONE_BATTLEFIELD,
                    tapped: false,
                    // Lander is a non-creature (Token Artifact);
                    // per CR 302.1 + 612.6 it enters with
                    // summoning_sick=false.
                    summoning_sick: false,
                    damage: 0,
                    counters: 0,
                    charge_counters: 0,
                    loyalty_counters: 0,
                    attached_to: None,
                    exiled_with_source: None,
                    stun_counters: 0,
                    etb_timestamp: tok_etb,
                    ueot_modifiers: Vec::new(),
                    warp_exile_pending: false,
                    warped: false,
                    cast_from_exile_with_lightstall_surcharge: false,
                    controller_baseline: None,
                    // Bespoke token-mint shape (same as Seedship /
                    // Orbital Plunge): Python's bespoke token-mints
                    // route through `state.new_object`, NOT through
                    // `CreateToken.resolve` — so Python skips the
                    // Moonlit replacement check entirely. Mirror
                    // that: the override stays `false`.
                    token_supertype_override: false,
                    // Control-change-UEOT substrate — Cycle 1
                    // (2026-05-21): freshly-minted tokens never start
                    // with a UEOT controller override.
                    controller_override: None,
                });
                // Queue ENTERS for the fresh token.
                state.pending_etb_dispatches.push(new_id);
            }
        }
        // DealDamageToTargetCreatureWithExcessLanderRider
        // (Effect-uncovered Tier-A — Orbital Plunge, Lander-rider).
        // Mirrors Python's
        // `DealDamageToTargetCreatureWithExcessLanderRider.resolve`
        // (`effects/generic/damage.py:357`): pick the target creature
        // (IsCreature slot; deterministic-first scan over BOTH
        // battlefields in ETB-timestamp order — Python's centralized
        // `state.battlefield` walk + `_slot_target_still_legal`
        // re-check, the established no-callback first-creature
        // campaign convention); compute `remaining = max(0,
        // effective_toughness − obj.damage)` and `excess = max(0,
        // amount − remaining)` BEFORE applying damage; apply `amount`
        // via `obj.damage.saturating_add`; if `excess > 0` create one
        // Lander under the source's controller. Lifelink/deathtouch
        // NOT modeled (Python parity — the documented simplification).
        //
        // **§2 counters-asymmetry (load-bearing, the mini-spec risk).**
        // Python's `effective_toughness` folds `+1/+1` counters (CR
        // 613 layer 7d). The combat-side Rust getter deliberately does
        // NOT, but the SBA lethal check DOES via `base + obj.counters
        // + ueot_toughness_bonus()`. To stay byte-equal with the
        // Python SoT, the remaining-lethal denominator here is
        // computed with the counters-INCLUSIVE formula re-derived
        // privately (file-iso): `base = ueot_base_pt_override().t OR
        // printed_toughness`, then `+ obj.counters +
        // obj.ueot_toughness_bonus()` — the exact
        // `state_ops_sba.rs::collect_sba_victims` shape. Pinned by a
        // +1/+1-counter'd target so a wrong accessor can't silently
        // diverge the excess.
        Effect::DealDamageToTargetCreatureWithExcessLanderRider {
            amount,
            token_printing_id,
        } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DealDamageToTargetCreatureWithExcessLanderRider \
                     amount must be non-negative, got {amount}"
                )));
            }
            let lander_pid: u32 = *token_printing_id;
            if state.registry.lookup(lander_pid).is_none() {
                return Err(PyValueError::new_err(format!(
                    "DealDamageToTargetCreatureWithExcessLanderRider \
                     token_printing_id {lander_pid} not in registry"
                )));
            }
            let dmg: i32 = *amount;
            // Target = first creature in ETB-timestamp order across
            // BOTH bfs (the IsCreature TargetSpec; Python's
            // `state.battlefield` centralized walk + the
            // deterministic-first no-callback convention).
            let target: Option<(usize, u32)> =
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        let printing =
                            state.registry.lookup(obj.printing_id)?;
                        if printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature")
                        {
                            Some((obj.controller as usize, obj.id))
                        } else {
                            None
                        }
                    });
            if let Some((pidx, target_id)) = target {
                // Compute excess BEFORE mutating damage. Counters-
                // inclusive effective toughness (the Python-SoT view —
                // collect_sba_victims shape; NOT the combat-side
                // getter which omits counters).
                let (eff_tough, prior_damage): (i32, i32) = {
                    let obj = state.players[pidx]
                        .battlefield
                        .iter()
                        .find(|o| o.id == target_id)
                        .expect("target_id from scan but not found");
                    let printing = state
                        .registry
                        .lookup(obj.printing_id)
                        .expect("scanned printing must resolve");
                    let printed_t = printing.toughness.unwrap_or(0) as i32;
                    let base_t = obj
                        .ueot_base_pt_override()
                        .map(|(_, t)| t)
                        .unwrap_or(printed_t);
                    let eff = base_t
                        + obj.counters
                        + obj.ueot_toughness_bonus();
                    (eff, obj.damage as i32)
                };
                let remaining: i32 = (eff_tough - prior_damage).max(0);
                let excess: i32 = (dmg - remaining).max(0);
                // Apply `amount` damage (DealDamageAnyTarget shape:
                // saturating_add into the i16 damage field).
                let dmg_i16 = dmg.min(i16::MAX as i32) as i16;
                {
                    let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                        &mut state.players[pidx].battlefield,
                    );
                    for o in bf_mut.iter_mut() {
                        if o.id == target_id {
                            o.damage = o.damage.saturating_add(dmg_i16);
                            break;
                        }
                    }
                }
                // Conditional Lander rider: only when excess damage
                // was dealt (CR 120.10 — at-resolve excess, NO
                // damaged-this-turn ledger; the Faller's-r57 lesson).
                if excess > 0 {
                    // Source-controller seat (CR 111.4). The in-tree
                    // stack resolver sets
                    // `current_resolve_source_controller`; fall back
                    // to active_player (back-compat for direct
                    // apply_effect_pub callers).
                    let recipient: u8 = state
                        .current_resolve_source_controller
                        .unwrap_or(state.active_player);
                    // R76-1: draw a fresh id from the monotonic
                    // `next_object_id` counter (CR 111.7), REPLACING
                    // the old non-monotonic `max_id + 1` re-scan that
                    // recycled a ceased token's id. Byte-equal with
                    // Python's `self.next_object_id += 1`.
                    let new_id: u32 =
                        crate::state::reserve_object_id(state);
                    let tok_etb = state.next_etb_timestamp;
                    state.next_etb_timestamp =
                        state.next_etb_timestamp.saturating_add(1);
                    let bf: &mut Vec<GameObject> = Arc::make_mut(
                        &mut state.players[recipient as usize]
                            .battlefield,
                    );
                    bf.push(GameObject {
                        id: new_id,
                        printing_id: lander_pid,
                        controller: recipient,
                        zone: ZONE_BATTLEFIELD,
                        tapped: false,
                        // Slice 14 (TTT Pin 2 alignment): Lander is a
                        // non-creature (Token Artifact); per CR 302.1
                        // + 612.6 it enters with summoning_sick=false.
                        summoning_sick: false,
                        damage: 0,
                        counters: 0,
                        charge_counters: 0,
                        loyalty_counters: 0,
                        attached_to: None,
                        exiled_with_source: None,
                        stun_counters: 0,
                        etb_timestamp: tok_etb,
                        ueot_modifiers: Vec::new(),
                        warp_exile_pending: false,
                        warped: false,
                        cast_from_exile_with_lightstall_surcharge: false,
                        controller_baseline: None,
                        // Aura-attachment Phase 4 S3 closure
                        // (2026-05-21): Orbital-Plunge-Lander-shape
                        // bespoke mints don't route through
                        // `CreateToken.resolve` on Python (same
                        // rationale as the sister Seedship Impact
                        // arm above).
                        token_supertype_override: false,
                        // Control-change-UEOT substrate — Cycle 1
                        // (2026-05-21): freshly-minted tokens never
                        // start with a UEOT controller override.
                        controller_override: None,
                    });
                    state.pending_etb_dispatches.push(new_id);
                }
            }
        }
        // DepressurizeEffect (Effect-uncovered Tier-A — Depressurize).
        // Mirrors Python's `DepressurizeEffect.resolve`
        // (`effects/bespoke.py:1554`): target = the CAST-TIME-BOUND
        // creature target (`targets[0]`); push a `−power_penalty/−0`
        // UEOT `PowerToughnessBonus`; then read POST-UEOT EFFECTIVE
        // power and `if eff_power <= 0` destroy it (respecting
        // indestructible — CR 702.12; Python's `turn.destroy` gate).
        //
        // **G7 follow-up f8e3a2b1 (arch=9/seed=4 Robot residual).**
        // Pre-fix this arm SCANNED for the first creature in ETB
        // order at RESOLVE time — a deterministic-first re-pick
        // (`bf_iter_etb_sorted` find_map). Under the
        // `s6_scale_verify_v3.json` arch=9 seed=4 step 1724 repro:
        // both Depressurize copies were cast targeting Gravpack
        // Monoist (id 63). The first Depressurize resolved and
        // killed GM; the GM dies-trigger then resolved and created
        // the Robot token (id 120). Now Depressurize #2 resolves:
        // Python's `resolve` (`effects/bespoke.py:1554`) reads
        // `targets[0]` (the cast-time-bound `(creature, 63)`), looks
        // up id 63 on `state.battlefield`, finds None (GM is in
        // graveyard), and FIZZLES (CR 608.2b — illegal target →
        // spell does nothing). Pre-fix Rust re-scanned bf for "first
        // creature", silently re-routed the destroy to the Robot,
        // and destroyed it. The STATE_EQ tripwire surfaced this as a
        // Python-has-Robot vs Rust-bf-empty divergence one step
        // after Depressurize #2 resolved.
        //
        // **Fix mirror.** Read the CAST-TIME-BOUND target id off
        // `state.current_resolve_payload_target_id` (the same value
        // `resolve_top_of_stack` swaps in from
        // `StackItem.target_obj_id` at every resolve entry — the
        // already-established payload-target convention used by the
        // Tier-C target-seated arms at e.g. line 8714 / 10052; see
        // the per-arm comment block at `:6649-6653`). Locate it on
        // EITHER battlefield (Depressurize's `legal_targets` is NOT
        // controller-restricted — Python's `next(o for o in
        // state.battlefield if o.id == target_id)` walks the
        // centralized list across both players). Fizzle on every
        // illegal-target shape: `None` payload, target left bf,
        // target type-changed off Creature (the three CR 608.2b
        // cases Python's `obj is None` early-return naturally
        // covers since the SBA destroy of a creature already moved
        // it off bf before this resolve fires). The bound target
        // itself is computed at cast time by
        // `cast_spell_noncreature` Delta-5
        // (`state_ops_cast.rs:~1439-1530`, `first-creature on either
        // bf in bf_iter_etb_sorted order`) — byte-equal to Python's
        // `_select_spell_targets` Phase-1 single-target pick — so on
        // the happy path (target still on bf at resolve, no
        // intervening SBA destroy) we land on the SAME creature the
        // pre-fix scan would have picked, byte-equal.
        //
        // **§2 counters-asymmetry (load-bearing, the mini-spec
        // risk).** Python's `effective_power`
        // (`game._compute_effective_pt`) folds the just-pushed UEOT
        // bonus AND `+1/+1` counters (CR 613 layer 7c+7d). Rust's
        // combat-side `effective_power` getter folds the UEOT bonus
        // but NOT counters. To stay byte-equal with the Python SoT,
        // the conditional reads the counters-INCLUSIVE effective
        // power re-derived privately (file-iso): `base =
        // ueot_base_pt_override().p OR printed_power`, plus
        // `obj.counters`, plus `obj.ueot_power_bonus()` (which now
        // includes the −pp just pushed). A combat-getter read would
        // silently diverge on a counter'd target — pinned.
        Effect::DepressurizeEffect { power_penalty } => {
            let pp: i32 = *power_penalty;
            // Cast-time-bound target id, swapped in by
            // `resolve_top_of_stack` from `StackItem.target_obj_id`
            // (set by `cast_spell_noncreature` Delta-5 at cast time,
            // or by `push_triggered_ability` via the trigger-push
            // seat in `first_legal_trigger_target`).
            //
            // **f8e3a2b1 bug scenario:** Some → a cast-time bind
            // succeeded but the target may have since left bf.
            // Locate the bound target on EITHER bf; not-found ⇒ CR
            // 608.2b fizzle, byte-equal to Python's `obj is None`
            // early-return at `effects/bespoke.py:1567`. NOT a
            // fallback-to-scan — that re-pick is exactly the pre-fix
            // bug that destroyed the Robot at arch=9/seed=4.
            //
            // **None payload:** no target ever bound. The only
            // production path that pushes DepressurizeEffect is
            // either `cast_spell_noncreature` (always binds) or a
            // future trigger via `push_triggered_ability` (binds via
            // `first_legal_trigger_target`'s DepressurizeEffect arm,
            // also added in this slice). The remaining caller is the
            // low-level synthetic-test `push_to_stack(obj, ctrl)`
            // API at `state_ops_stack.rs:56`, documented to seat NO
            // targets ("First-slice StackItem shape:
            // `{source_obj_id, controller}` only"). For that path —
            // and ONLY that path — fall back to the pre-fix
            // deterministic-first ANY-creature scan: identical
            // observable to the prior behavior on every synthetic
            // pin, with no risk of re-introducing the bug because the
            // bug requires a NON-None bound target whose target left
            // bf. Documented in `docs/SEARCH_LIMITATIONS.md` under
            // the G7 follow-up f8e3a2b1 closure.
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                if printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                                {
                                    Some((pidx, o.id))
                                } else {
                                    None
                                }
                            })
                    }),
                None => crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        let printing =
                            state.registry.lookup(obj.printing_id)?;
                        if printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature")
                        {
                            Some((obj.controller as usize, obj.id))
                        } else {
                            None
                        }
                    }),
            };
            if let Some((pidx, target_id)) = target {
                // Push the −pp/−0 UEOT modifier (verbatim from the
                // PumpTargetCreatureUntilEOT push shape).
                {
                    let zone: &mut Vec<GameObject> = Arc::make_mut(
                        &mut state.players[pidx].battlefield,
                    );
                    for o in zone.iter_mut() {
                        if o.id == target_id {
                            o.ueot_modifiers.push(
                                UeotModifier::PowerToughnessBonus {
                                    power: -pp,
                                    toughness: 0,
                                },
                            );
                            break;
                        }
                    }
                }
                // Read POST-UEOT effective power, counters-inclusive
                // (the Python-SoT view; NOT the combat-side getter
                // which omits counters). base = ueot base override
                // (.p) OR printed power; + obj.counters +
                // obj.ueot_power_bonus() (now includes the −pp).
                let (eff_power, is_indestructible): (i32, bool) = {
                    let o = state.players[pidx]
                        .battlefield
                        .iter()
                        .find(|x| x.id == target_id)
                        .expect("target_id from scan but not found");
                    let printing = state
                        .registry
                        .lookup(o.printing_id)
                        .expect("scanned printing must resolve");
                    let printed_p = printing.power.unwrap_or(0) as i32;
                    let base_p = o
                        .ueot_base_pt_override()
                        .map(|(p, _)| p)
                        .unwrap_or(printed_p);
                    let eff =
                        base_p + o.counters + o.ueot_power_bonus();
                    let indest = printing
                        .keywords
                        .iter()
                        .any(|k| k == "indestructible")
                        || o.ueot_has_keyword("indestructible");
                    (eff, indest)
                };
                // "Then if that creature's power is 0 or less,
                // destroy it." Python's `turn.destroy` respects
                // indestructible (CR 702.12 — the pump still applied,
                // only the destroy is suppressed).
                if eff_power <= 0 && !is_indestructible {
                    let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                        &mut state.players[pidx].battlefield,
                    );
                    let pos = bf_mut
                        .iter()
                        .position(|o| o.id == target_id)
                        .expect("target_id present but not found");
                    let mut obj = bf_mut.remove(pos);
                    obj.zone = ZONE_GRAVEYARD;
                    obj.tapped = false;
                    // CE13: summoning_sick NOT reset (CR 302.1, I35).
                    obj.damage = 0;
                    // CFI-1: counters are NOT cleared on the dead
                    // object. Python's `turn.move_off_battlefield`
                    // (turn.py) clears combat marks (damage / tapped /
                    // attached_to) but deliberately leaves `+1/+1`
                    // counters intact; the SBA destroy path
                    // (`state_ops_sba.rs` ~:417) follows that observable
                    // ("counters retained on the dead object" — the
                    // established byte-equal contract, pinned by the
                    // Orbital Plunge probe). A prior stray
                    // `obj.counters = 0;` here diverged from BOTH Python
                    // (SoT) and Rust's own SBA destroy path; removed so
                    // a counter'd creature destroyed by Depressurize
                    // carries its counters into the graveyard byte-equal
                    // (CFI-1 closed; see the FINDINGS NOTE in
                    // tests/test_probe_cross_feature_interaction.py).
                    obj.attached_to = None;
                    obj.exiled_with_source = None;
                    let (printing_id, leaving_obj_id) =
                        (obj.printing_id, obj.id);
                    Arc::make_mut(
                        &mut state.players[pidx].graveyard,
                    )
                    .push(obj);
                    mark_left_battlefield_if_nonland(
                        state, printing_id,
                    );
                    // R74-2: a "destroy" of a creature is a death (CR
                    // 700.4) — its `Event.DIES` triggered abilities
                    // MUST fire, exactly as the generic SBA destroy
                    // path does (`state_ops_sba.rs` ~:467-482) and as
                    // Python's `turn.destroy` -> `move_off_battlefield`
                    // does (`turn.py:253-283`: `_fire_dies_triggers` +
                    // the predicate-path `dispatch_event(DIES)` BEFORE
                    // `_fire_leaves_battlefield_triggers`). The prior
                    // hand-rolled arm pushed ONLY
                    // `pending_leaves_dispatches`, so a destroyed
                    // dies-trigger creature (e.g. Gravpack Monoist ->
                    // Robot token) silently never fired its trigger —
                    // a Rust-side divergence from the Python SoT
                    // surfaced by the round-74 post-spell-admit bench
                    // sweep. Push order mirrors the SBA path + the
                    // `Event::LeavesBattlefield` cards.rs doc-comment
                    // contract (DIES first via
                    // `pending_dies_dispatches`, LEAVES second via
                    // `pending_leaves_dispatches`); the matching
                    // resolution-path drain order (DIES drained then
                    // LEAVES; LIFO -> LEAVES resolves first) preserves
                    // Python's `move_off_battlefield` sequence.
                    // Creature-gated like the SBA path so a
                    // type-changed non-creature does not fire a
                    // spurious DIES (CR 700.4 — Depressurize's scan
                    // already filters to creatures, but the gate keeps
                    // the SBA-path invariant byte-equal).
                    let victim_is_creature = state
                        .registry
                        .lookup(printing_id)
                        .map(|p| {
                            p.card_types
                                .iter()
                                .any(|t| t == "Creature")
                        })
                        .unwrap_or(false);
                    if victim_is_creature {
                        state
                            .pending_dies_dispatches
                            .push(leaving_obj_id);
                    }
                    state
                        .pending_leaves_dispatches
                        .push(leaving_obj_id);
                    crate::state_ops_zones::queue_token_cease(
                        state,
                        leaving_obj_id,
                    );
                }
            }
        }
        // BiosynthicBurstEffect (Effect-uncovered Tier-A — Biosynthic
        // Burst). Mirrors Python's `BiosynthicBurstEffect.resolve`
        // (`effects/bespoke.py:1487`): own-creature target;
        // `add_plus_one_counters(state, obj, 1)`;
        // `ApplyEffect(GrantKeywords({reach,trample,indestructible}),
        // EndOfTurn()).resolve(...)`; `obj.tapped = False`.
        //
        // **Controller-restricted scan (verified deviation, pinned).**
        // Python's `legal_targets` IS controller-restricted
        // (`o.controller_id == controller_id`, bespoke.py:1484) —
        // UNLIKE `DepressurizeEffect` whose legal_targets is ANY
        // creature. So this scan walks ONLY the source-controller's
        // battlefield (the same controller-only first-creature scan
        // `PutPlusOneCounterOnCreatureYouControl` `:7721` and
        // `GrantKeywordsToTargetCreatureUntilEOT` `:10509` use), NOT
        // both bfs via `bf_iter_etb_sorted`. `resolve` also re-guards
        // `obj.controller_id == controller_id` after target
        // selection — the scan already enforces it.
        //
        // **Counter shape (the established byte-equal mapping).** The
        // +1/+1 counter is `obj.counters.saturating_add(1)` — the
        // EXACT shape `PutPlusOneCounterOnCreatureYouControl` `:7780`
        // uses. Python's `add_plus_one_counters` writes
        // `obj.plus_one_counters`; the cross-engine bridge maps that
        // to Rust's signed scalar `obj.counters` (cross_engine.py
        // ~:1276 "mirrors Rust's scalar"). Python's
        // counter-doubling-replacement / Terrasymbiosis observer
        // fan-out is NOT replicated here — the proven sibling arm
        // makes the identical simplification (no doubler / no
        // counters-added trigger in the EoE pool); documented in
        // `docs/SEARCH_LIMITATIONS.md`.
        //
        // **3-keyword UEOT grant (the proven push shape).** Re-derive
        // the `GrantKeywordsToTargetCreatureUntilEOT` `:10559` push
        // privately: for each of {reach, trample, indestructible}
        // push `UeotModifier::GrantKeyword { keyword }` (lowercase,
        // exactly as that arm does). That arm's continuous_effects ↔
        // ueot_modifiers equivalence is already cross-engine-proven
        // byte-equal (Python's `GrantKeywords` calls
        // `state.add_continuous_effect(layer=6, kind="grant_keyword")`
        // wrapped in `ApplyEffect(..., EndOfTurn())`; the harness
        // proves the mapping). The UEOT keywords drain at the
        // cleanup step on BOTH engines (the round-39 SBA-on-0/0
        // class — indestructible protects THROUGH the end-of-turn
        // SBA, then the keyword expires at cleanup; pinned).
        Effect::BiosynthicBurstEffect => {
            // Locate source controller (5-zone walk; identical to the
            // `PutPlusOneCounterOnCreatureYouControl` /
            // `GrantKeywordsToTargetCreatureUntilEOT` siblings).
            let mut src_controller: Option<u8> = None;
            'bb_outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) =
                        zone.iter().find(|o| o.id == source_obj_id)
                    {
                        src_controller = Some(obj.controller);
                        break 'bb_outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            if let Some(controller) = src_controller {
                // Pick first creature CONTROLLED by `controller`
                // across BOTH battlefields (the controller-restricted
                // deviation — Python's legal_targets is controller-
                // only, walking `state.battlefield` and filtering on
                // `o.controller_id`). The OWNER-only walk pre-fix
                // missed creatures whose controller had been flipped
                // by a control-changing effect (Tractor Beam's
                // `attached_grants_control`); surfaced by the
                // bench-loader admit follow-up — arch14/s4/step2249.
                let target_id: Option<u32> =
                    crate::bf_iter::bf_iter_etb_sorted(state)
                        .into_iter()
                        .find_map(|obj| {
                            if obj.controller != controller {
                                return None;
                            }
                            let printing = state
                                .registry
                                .printings
                                .get(obj.printing_id as usize)?;
                            if printing
                                .card_types
                                .iter()
                                .any(|t| t == "Creature")
                            {
                                Some(obj.id)
                            } else {
                                None
                            }
                        });
                if let Some(target_id) = target_id {
                    // Locate the target's OWNER's battlefield (the
                    // physical zone is keyed on owner; controller is
                    // a separate per-object slot the bf-walker reads
                    // and the apply-loop also reads).
                    let target_owner_idx: Option<usize> = (0..state
                        .players
                        .len())
                        .find(|&pidx| {
                            state.players[pidx]
                                .battlefield
                                .iter()
                                .any(|o| o.id == target_id)
                        });
                    if let Some(oidx) = target_owner_idx {
                        let zone: &mut Vec<GameObject> = Arc::make_mut(
                            &mut state.players[oidx].battlefield,
                        );
                        for obj in zone.iter_mut() {
                            if obj.id == target_id {
                                // +1/+1 counter (the established
                                // saturating_add scalar shape).
                                obj.counters =
                                    obj.counters.saturating_add(1);
                                // 3-keyword UEOT grant (the proven
                                // GrantKeyword push shape, lowercase).
                                for kw in
                                    ["reach", "trample", "indestructible"]
                                {
                                    obj.ueot_modifiers.push(
                                        UeotModifier::GrantKeyword {
                                            keyword: kw.to_string(),
                                        },
                                    );
                                }
                                // Untap it.
                                obj.tapped = false;
                                break;
                            }
                        }
                    }
                }
                // No own creature → silent no-op (Python returns
                // early on empty legal_targets / non-creature).
            }
        }
        // FocusFireEffect (Effect-uncovered Tier-A — Focus Fire). The
        // FIRST effect-resolve arm to READ combat-declaration state.
        // Mirrors Python's `FocusFireEffect.resolve` /
        // `FocusFireEffect._amount` (`effects/bespoke.py:1784/1792`):
        // deal `(base + |source-controller's creatures∪Spacecraft|)`
        // damage to the first attacking-or-blocking creature.
        //
        // **The set-union denominator (the §2 subtle-divergence
        // pin).** Walk the source-controller's bf ONCE; +1 for each
        // obj whose `printing.card_types` contains "Creature" OR
        // "Spacecraft". A dual-typed Artifact Creature Spacecraft is
        // ONE `Vec` element → counts ONCE. This is NOT a sum of a
        // creature-count and a spacecraft-count (that double-counts
        // dual-typed permanents). BIT-FOR-BIT mirrors Python's
        // `base + sum(1 for o in state.battlefield if o.controller_id
        // == controller_id and (o.printing.is_creature or
        // o.printing.is_spacecraft))` — the `or` is on the object.
        //
        // **The combat predicate (verify-first-confirmed byte-equal
        // Python↔Rust).** A creature is attacking iff its id is in
        // `state.attackers` (`Vec<u32>`); blocking iff its id appears
        // as a `blocker_id` in `state.blockers` (`Vec<(attacker_id,
        // blocker_id)>`). This is the EXACT pair `declare_attacker`
        // (`state_ops_combat.rs:681`) / `declare_blocker` (`:772`)
        // push and the multi-blocker cross-engine probes have proven
        // byte-equal vs Python's `obj.attacking` /
        // `obj.blocking is not None`. Python re-checks
        // `_slot_target_still_legal` at resolve (CR 608.2b); if no
        // creature is attacking-or-blocking the spell still resolves
        // and the damage fizzles (silent no-op).
        //
        // **Target scan = deterministic-first across BOTH bfs in
        // ETB-timestamp order** (Python's filter is NOT controller-
        // restricted — `IsCreature ∧ (IsAttacking ∨ IsBlocking)`,
        // ANY creature; the same `bf_iter_etb_sorted` +
        // deterministic-first convention `DepressurizeEffect` `:5889`
        // uses, the cross-engine spell-card-bridge `_PickFirstLegal
        // TargetAgent` mirrors it). Damage applied via
        // `obj.damage.saturating_add` (the `DealDamageAnyTarget`
        // shape).
        Effect::FocusFireEffect { base } => {
            // Source-controller resolution (5-zone walk + stack scan;
            // the established sister shape — `DealDamageToOpponent
            // CreatureByCreaturesYouControl` `:7065`,
            // `BiosynthicBurstEffect` `:6040`).
            let mut src_controller: Option<u8> = None;
            'ff_outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) =
                        zone.iter().find(|o| o.id == source_obj_id)
                    {
                        src_controller = Some(obj.controller);
                        break 'ff_outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Set-union denominator: ONE pass over the source-
            // controller's bf; +1 if Creature OR Spacecraft (the `||`
            // is on the object — a dual-typed obj is one Vec element,
            // counts ONCE; NOT a two-scan sum).
            let cidx = controller as usize;
            let mut x: i32 = *base;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state
                    .registry
                    .printings
                    .get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing
                    .card_types
                    .iter()
                    .any(|t| t == "Creature" || t == "Spacecraft")
                {
                    x = x.saturating_add(1);
                }
            }
            // **Battlefield STATE_EQ axis closure (todo `e5b7c9d8`).**
            // Migrated to the `current_resolve_payload_target_id`
            // pattern (SeedshipImpactEffect / DepressurizeEffect
            // precedent). Pre-fix this arm self-scanned for the
            // first attacking-or-blocking creature in ETB-timestamp
            // order — diverging from Python whenever the cast-time-
            // bound target was NOT the oldest attacker/blocker.
            //
            // Repro (arch=281 seed=4 step 2358 / `docs/BATTLEFIELD_
            // AXIS_INVESTIGATION.md` §3): P0 declares 3 attackers
            // (Knight Luminary id=10, Human Soldiers id=120 and
            // id=121). P1 casts 2 Focus Fires each targeting id=120.
            // Python resolves each Focus Fire against bound target
            // id=120 (damage to Human Soldier 120, which dies). Rust
            // pre-fix scanned for first attacking-or-blocking
            // creature in ETB order → picked Knight id=10 (oldest
            // attacker) twice → Knight dies, id=120 survives.
            // Diverges from Python by which creature dies. Post-fix:
            // Rust reads `current_resolve_payload_target_id ==
            // Some(120)`, scans for id=120 on bf, finds it (still
            // attacking) → deals damage to id=120. Byte-equal.
            //
            // **None-payload synthetic-test fallback.** Mirrors the
            // DepressurizeEffect synthetic-fallback precedent
            // (`:6808`). The low-level `push_to_stack` API used by
            // synthetic tests (e.g.
            // `test_rust_focus_fire_denominator_is_source_
            // controllers_board`, `test_cross_engine_real_eoe_focus_
            // fire_drives_combat_then_casts_parity`) does NOT seat a
            // target. For that path — and ONLY that path — fall
            // back to the pre-fix deterministic-first scan. Combat
            // predicate (`id ∈ state.attackers ∨ id ∈ blockers`)
            // preserved in both branches: the bound-target leg re-
            // checks it at resolve (CR 608.2b "still matches the
            // typed filter" leg — if a bound attacker died and got
            // removed from `state.attackers`, the spell fizzles
            // silently as Python does).
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                if !printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                                {
                                    return None;
                                }
                                let is_attacking =
                                    state.attackers.contains(&o.id);
                                let is_blocking = state
                                    .blockers
                                    .iter()
                                    .any(|(_att, blk)| *blk == o.id);
                                if is_attacking || is_blocking {
                                    Some((pidx, o.id))
                                } else {
                                    None
                                }
                            })
                    }),
                None => {
                    let attackers = &state.attackers;
                    let blockers = &state.blockers;
                    let registry = &state.registry;
                    crate::bf_iter::bf_iter_etb_sorted(state)
                        .into_iter()
                        .find_map(|obj| {
                            let printing =
                                registry.lookup(obj.printing_id)?;
                            if !printing
                                .card_types
                                .iter()
                                .any(|t| t == "Creature")
                            {
                                return None; // not a creature
                            }
                            let is_attacking =
                                attackers.contains(&obj.id);
                            let is_blocking = blockers
                                .iter()
                                .any(|(_att, blk)| *blk == obj.id);
                            if is_attacking || is_blocking {
                                Some((obj.controller as usize, obj.id))
                            } else {
                                None
                            }
                        })
                }
            };
            if let Some((pidx, target_id)) = target {
                let amount_i16 = x.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                    &mut state.players[pidx].battlefield,
                );
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage =
                            obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no attacking/blocking creature → silent fizzle
            // (CR 608.2b — the spell resolves, damage is not dealt;
            // Python's `_slot_target_still_legal` returns early).
        }
        // DiplomaticRelationsEffect (Effect-uncovered Tier-A #7 —
        // Diplomatic Relations; spell-card-bridge Phase 2). The
        // FIRST 2-slot heterogeneous-target spell arm. Mirrors
        // Python's `DiplomaticRelationsEffect.resolve`
        // (`effects/bespoke.py:1940-2008`) BIT-FOR-BIT against the
        // Python source of truth:
        //  - slot 0 (friendly pumped creature) rides on
        //    `current_resolve_payload_target_id` (swapped in by both
        //    resolve entry points from `StackItem.target_obj_id`);
        //  - slot 1 (opponent's damaged creature) rides on
        //    `current_resolve_payload_target_id_2` (from
        //    `StackItem.target_obj_id_2`).
        // Python: pump slot-0 +1/+0 + vigilance UEOT (target
        // re-validated: still on a bf AND still caster-controlled,
        // else the WHOLE effect fizzles — `my_obj=None` short-
        // circuits before damage too); then if slot-1 exists +
        // is opponent-controlled, deal slot-0's POST-PUMP effective
        // power (counters-inclusive) as damage.
        Effect::DiplomaticRelationsEffect => {
            // Source-controller resolution (the established sister
            // 5-zone-walk + stack-scan shape — `BiosynthicBurst
            // Effect` `:6091`, `FocusFireEffect` `:6215`). This is
            // the caster (= `controller_id` in Python's `resolve`).
            let mut src_controller: Option<u8> = None;
            'dr_outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) =
                        zone.iter().find(|o| o.id == source_obj_id)
                    {
                        src_controller = Some(obj.controller);
                        break 'dr_outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Slot 0: the friendly pumped creature. Read the bound
            // target, re-validate (CR 608.2b — Python re-checks the
            // creature is on a bf AND still controlled by the
            // caster, else `my_obj = None` and the WHOLE effect is a
            // no-op — NO damage either).
            let slot0_id = state.current_resolve_payload_target_id;
            let mut my_pidx: Option<usize> = None;
            if let Some(t0) = slot0_id {
                'find0: for (pidx, player) in
                    state.players.iter().enumerate()
                {
                    if let Some(obj) = player
                        .battlefield
                        .iter()
                        .find(|o| o.id == t0)
                    {
                        // Still controlled by the caster?
                        if obj.controller == controller {
                            my_pidx = Some(pidx);
                        }
                        break 'find0;
                    }
                }
            }
            let (Some(my_pidx), Some(my_id)) = (my_pidx, slot0_id)
            else {
                // slot-0 illegal at resolve → whole effect fizzles
                // (Python: `my_obj = None` → early return before
                // damage). Silent no-op.
                return Ok(());
            };
            // Pump slot-0 +1/+0 UEOT + grant vigilance UEOT — the
            // EXACT push shapes the cross-engine-proven
            // `DepressurizeEffect` (PowerToughnessBonus) /
            // `BiosynthicBurstEffect` (GrantKeyword, lowercase) arms
            // use. Both drain at cleanup byte-equal on both engines.
            {
                let zone: &mut Vec<GameObject> = Arc::make_mut(
                    &mut state.players[my_pidx].battlefield,
                );
                for obj in zone.iter_mut() {
                    if obj.id == my_id {
                        obj.ueot_modifiers.push(
                            UeotModifier::PowerToughnessBonus {
                                power: 1,
                                toughness: 0,
                            },
                        );
                        obj.ueot_modifiers.push(
                            UeotModifier::GrantKeyword {
                                keyword: "vigilance".to_string(),
                            },
                        );
                        break;
                    }
                }
            }
            // Slot 1: the opponent's damaged creature. Read the
            // second bound target, re-validate (must be on a bf AND
            // controlled by a NON-caster — Python checks
            // `opp_obj.controller_id != controller_id`). If
            // illegal/absent, the pump already applied and the
            // damage component is skipped (Python returns after the
            // pump in that branch).
            let slot1_id = state.current_resolve_payload_target_id_2;
            let mut opp_pidx: Option<usize> = None;
            if let Some(t1) = slot1_id {
                'find1: for (pidx, player) in
                    state.players.iter().enumerate()
                {
                    if let Some(obj) = player
                        .battlefield
                        .iter()
                        .find(|o| o.id == t1)
                    {
                        if obj.controller != controller {
                            opp_pidx = Some(pidx);
                        }
                        break 'find1;
                    }
                }
            }
            let (Some(opp_pidx), Some(opp_id)) = (opp_pidx, slot1_id)
            else {
                return Ok(()); // pump applied, damage component
                               // fizzles — Python early return.
            };
            // Damage = slot-0's POST-PUMP EFFECTIVE power, counters-
            // INCLUSIVE (Python's `effective_power` view — the EXACT
            // counters-inclusive formula re-derived privately per
            // the file-isolation rule, identical to the
            // `DepressurizeEffect` conditional read `:5928-5951`:
            // base = ueot_base_pt_override().p OR printed_power,
            // plus obj.counters, plus obj.ueot_power_bonus() — which
            // now includes the just-pushed +1).
            let amount: i32 = {
                let o = state.players[my_pidx]
                    .battlefield
                    .iter()
                    .find(|x| x.id == my_id)
                    .expect("slot-0 id validated but not found");
                let printing = state
                    .registry
                    .lookup(o.printing_id)
                    .expect("slot-0 printing must resolve");
                let printed_p = printing.power.unwrap_or(0) as i32;
                let base_p = o
                    .ueot_base_pt_override()
                    .map(|(p, _)| p)
                    .unwrap_or(printed_p);
                base_p + o.counters + o.ueot_power_bonus()
            };
            // Apply damage to slot-1 via the proven
            // `obj.damage.saturating_add` primitive shape
            // (`DealDamageAnyTarget` / `FocusFireEffect`). SBA
            // handles lethal post-resolve byte-equal. Applied
            // UNCONDITIONALLY (no `amount > 0` clamp): Python's
            // `_apply_damage` does `obj.damage += amount`
            // unconditionally — `effective_power` is NOT clamped to
            // 0 (`game.py:_compute_effective_pt`), so a byte-equal
            // mirror must add the (rare) non-positive amount too.
            // With the just-pushed +1/+0 the realistic amount is
            // >= 1 for any non-negative-power creature; the
            // unconditional add only matters for a theoretically
            // negative effective power, where matching Python's
            // unclamped `+=` is the byte-equal contract. `saturating
            // _add` on the i16 mirrors Python's unbounded int add
            // for the realistic damage range (same shape as the
            // proven FocusFire/DealDamageAnyTarget arms).
            let amount_i16 = amount
                .clamp(i16::MIN as i32, i16::MAX as i32)
                as i16;
            let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                &mut state.players[opp_pidx].battlefield,
            );
            for obj in bf_mut.iter_mut() {
                if obj.id == opp_id {
                    obj.damage =
                        obj.damage.saturating_add(amount_i16);
                    break;
                }
            }
        }
        // **Effect-uncovered Tier-A — Cut Propulsion.** "Target
        // creature deals damage to itself equal to its power. If that
        // creature has flying, it deals twice that much damage to
        // itself instead." Mirrors Python's `CutPropulsionEffect`
        // (`effects/bespoke.py:1843-1873`) BIT-FOR-BIT against the
        // Python source of truth.
        //
        // **Target binding.** Single target, ANY creature on either
        // battlefield (Python's `legal_targets` is the centralized
        // `state.battlefield` walk filtered by `printing.is_creature`
        // — NOT controller-restricted). Read the cast-time-bound
        // target id from `state.current_resolve_payload_target_id`
        // (the established `FocusFireEffect` / `DepressurizeEffect`
        // payload-target swap pattern). When `None` (synthetic-test
        // `push_to_stack` path), fall back to the deterministic-
        // first creature in ETB-timestamp order across BOTH bfs.
        //
        // **Resolve-time validation (CR 608.2b).** The target must
        // still be on a battlefield AND its printing must satisfy
        // `card_types ∋ "Creature"`. Illegal at resolve → silent
        // fizzle (Python's `obj is None` short-circuit at
        // `bespoke.py:1854-1858`).
        //
        // **Damage amount.** Read the target's effective power via
        // the proven `DepressurizeEffect`/`DiplomaticRelationsEffect`
        // counters-inclusive formula (`base = ueot_base_pt_override()
        // .p OR printed_power, plus obj.counters, plus
        // obj.ueot_power_bonus()`). If `power <= 0`, silent no-op
        // (Python's `if power <= 0: return` at
        // `bespoke.py:1860-1865` short-circuits BEFORE `_apply_damage`
        // — so the receiver-side DEALT_DAMAGE dispatch does NOT fire
        // either, byte-equal). Otherwise read effective keyword set:
        // if "flying" is granted (`battlefield_has_keyword` folds
        // printed + UEOT-granted + aura-granted + anthem-granted
        // keywords; Python's `effective_keywords` walk), amount =
        // `power * 2`; else amount = `power`. Apply via the proven
        // `obj.damage.saturating_add(amount_i16)` shape. After
        // applying, dispatch the DEALT_DAMAGE receiver-side event
        // (Python's `_fire_enchanted_dealt_damage_triggers` call at
        // the tail of `_apply_damage`, `effects/base.py:303-305`)
        // so aura-attached "when enchanted creature is dealt damage"
        // triggers (Cryoshatter, Pain for All) fire byte-equal.
        Effect::CutPropulsionEffect => {
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                if !printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => {
                    // Synthetic-test fallback (the established
                    // `FocusFireEffect` / `DepressurizeEffect` None-
                    // payload fallback): deterministic-first creature
                    // in ETB-timestamp order across BOTH bfs (mirrors
                    // `bf_iter_etb_sorted` — Python's centralized
                    // `state.battlefield` walk = the SoT).
                    let registry = &state.registry;
                    crate::bf_iter::bf_iter_etb_sorted(state)
                        .into_iter()
                        .find_map(|obj| {
                            let printing =
                                registry.lookup(obj.printing_id)?;
                            if printing
                                .card_types
                                .iter()
                                .any(|t| t == "Creature")
                            {
                                Some((obj.controller as usize, obj.id))
                            } else {
                                None
                            }
                        })
                }
            };
            let Some((pidx, target_id)) = target else {
                return Ok(()); // no eligible target → silent fizzle
            };
            // Effective power (counters-inclusive, UEOT-pump-
            // inclusive). Re-derived privately per the file-isolation
            // convention — identical to the `DiplomaticRelations
            // Effect` `:8355-8371` / `DepressurizeEffect` body.
            let power: i32 = {
                let o = state.players[pidx]
                    .battlefield
                    .iter()
                    .find(|x| x.id == target_id)
                    .expect("target id validated but not found");
                let printing = state
                    .registry
                    .lookup(o.printing_id)
                    .expect("target printing must resolve");
                let printed_p = printing.power.unwrap_or(0) as i32;
                let base_p = o
                    .ueot_base_pt_override()
                    .map(|(p, _)| p)
                    .unwrap_or(printed_p);
                base_p + o.counters + o.ueot_power_bonus()
            };
            if power <= 0 {
                // Python `bespoke.py:1860-1865`: silent log-only no-
                // op; the receiver-side DEALT_DAMAGE dispatch does
                // NOT fire (no `_apply_damage` call). Byte-equal:
                // skip the damage path entirely.
                return Ok(());
            }
            // Flying doubler — read effective keyword set on the
            // target. `battlefield_has_keyword` folds printed + UEOT
            // grant + aura grant + anthem grant + self-grant (the
            // established public reader). Lowercase match (the bridge
            // already lowercases printed keywords at register time).
            let amount: i32 = if crate::state_ops_combat::battlefield_has_keyword(
                state, target_id, "flying",
            ) {
                power.saturating_mul(2)
            } else {
                power
            };
            let amount_i16 = amount.min(i16::MAX as i32) as i16;
            // Apply damage marker. Mirrors the proven
            // `apply_damage_marker` shape (no SBA inline, just the
            // saturating_add on `obj.damage`).
            let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                &mut state.players[pidx].battlefield,
            );
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.damage =
                        obj.damage.saturating_add(amount_i16);
                    break;
                }
            }
            // Round-16 RR Pin 3 sibling: dispatch DEALT_DAMAGE for
            // the creature that RECEIVED damage (receiver-side event).
            // Mirrors Python's `_fire_enchanted_dealt_damage_triggers`
            // at `effects/base.py:303-305`. Fires after the damage
            // marker is applied — auras like Cryoshatter / Pain for
            // All attached to the receiver see the post-damage state,
            // byte-equal with Python.
            dispatch_event(
                state,
                crate::cards::Event::DealtDamage,
                target_id,
            )?;
        }
        // **Effect-uncovered Tier-A — Invasive Maneuvers.** "Invasive
        // Maneuvers deals 3 damage to target creature. It deals 5
        // damage instead if you control a Spacecraft." Mirrors
        // Python's `DealDamageToTargetCreatureSpacecraftBoosted`
        // (`effects/generic/damage.py:850-905`) BIT-FOR-BIT against
        // the Python source of truth.
        //
        // **Target binding.** Single target, ANY creature on either
        // battlefield (Python's `IsCreature()` slot filter walks
        // `state.battlefield` — NOT controller-restricted). Read the
        // cast-time-bound target id from
        // `state.current_resolve_payload_target_id` (the established
        // `FocusFireEffect` / `CutPropulsionEffect` payload-target
        // swap pattern). When `None` (synthetic-test `push_to_stack`
        // path), fall back to the deterministic-first creature in
        // ETB-timestamp order across BOTH bfs.
        //
        // **Resolve-time validation (CR 608.2b).** The target must
        // still be on a battlefield AND its printing must satisfy
        // `card_types ∋ "Creature"`. Illegal at resolve → silent
        // fizzle (Python's `_slot_target_still_legal` at
        // `damage.py:891-893` returns early).
        //
        // **Source-controller resolution (5-zone walk + stack scan).**
        // Mirrors the `FocusFireEffect` sister shape. The Python
        // `_amount(state, controller_id)` receives `controller_id`
        // passed in from the resolve caller; on the Rust side we
        // derive it from `source_obj_id`.
        //
        // **Spacecraft predicate.** Walk the source-controller's bf
        // ONCE; +Spacecraft check via `printing.subtypes` (NOT
        // `card_types` — Spacecraft is a SUBTYPE per
        // `cards/base.py:1056-1057`: `is_spacecraft = "Spacecraft" in
        // subtypes`). Mirrors Python's `o.printing.is_spacecraft`
        // predicate at `damage.py:876-879`. Amount = `boosted` if any
        // controlled Spacecraft present (5 on the printed card), else
        // `base` (3).
        //
        // **Damage application.** Apply via the proven
        // `obj.damage.saturating_add(amount_i16)` shape. After
        // applying, dispatch the DEALT_DAMAGE receiver-side event
        // (Python's `_fire_enchanted_dealt_damage_triggers` call at
        // the tail of `_apply_damage`, `effects/base.py:303-305`)
        // so aura-attached "when enchanted creature is dealt damage"
        // triggers (Cryoshatter, Pain for All) fire byte-equal.
        Effect::InvasiveManeuversEffect { base, boosted } => {
            // Source-controller resolution (5-zone walk + stack scan;
            // the established `FocusFireEffect` / `BiosynthicBurst
            // Effect` sister shape).
            let mut src_controller: Option<u8> = None;
            'im_outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) =
                        zone.iter().find(|o| o.id == source_obj_id)
                    {
                        src_controller = Some(obj.controller);
                        break 'im_outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Target resolution (the established
            // `CutPropulsionEffect` / `FocusFireEffect` payload-
            // target swap pattern; None-payload synthetic-test
            // fallback to deterministic-first creature in ETB-
            // timestamp order across BOTH bfs).
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                if !printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => {
                    // Synthetic-test fallback: deterministic-first
                    // creature in ETB-timestamp order across BOTH bfs
                    // (mirrors `bf_iter_etb_sorted` — Python's
                    // centralized `state.battlefield` walk = the SoT).
                    let registry = &state.registry;
                    crate::bf_iter::bf_iter_etb_sorted(state)
                        .into_iter()
                        .find_map(|obj| {
                            let printing =
                                registry.lookup(obj.printing_id)?;
                            if printing
                                .card_types
                                .iter()
                                .any(|t| t == "Creature")
                            {
                                Some((obj.controller as usize, obj.id))
                            } else {
                                None
                            }
                        })
                }
            };
            let Some((pidx, target_id)) = target else {
                return Ok(()); // no eligible target → silent fizzle
            };
            // Spacecraft predicate on the source-controller's bf.
            // Reads `printing.subtypes` (NOT `card_types`) — mirrors
            // Python's `is_spacecraft = "Spacecraft" in subtypes`
            // (`cards/base.py:1056-1057`). One bf pass; the closure
            // short-circuits on the first match.
            let cidx = controller as usize;
            let has_spacecraft: bool = state.players[cidx]
                .battlefield
                .iter()
                .any(|obj| {
                    state
                        .registry
                        .lookup(obj.printing_id)
                        .map(|p| {
                            p.subtypes
                                .iter()
                                .any(|s| s == "Spacecraft")
                        })
                        .unwrap_or(false)
                });
            let amount: i32 = if has_spacecraft { *boosted } else { *base };
            let amount_i16 = amount.min(i16::MAX as i32) as i16;
            // Apply damage marker. Mirrors the proven
            // `apply_damage_marker` shape (no SBA inline, just the
            // saturating_add on `obj.damage`; SBA fires post-resolve
            // at the outer `apply_sbas` boundary).
            let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                &mut state.players[pidx].battlefield,
            );
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.damage =
                        obj.damage.saturating_add(amount_i16);
                    break;
                }
            }
            // DEALT_DAMAGE receiver-side dispatch (mirrors Python's
            // `_fire_enchanted_dealt_damage_triggers` at
            // `effects/base.py:303-305`). Fires after the damage
            // marker is applied — auras like Cryoshatter / Pain for
            // All attached to the receiver see the post-damage state,
            // byte-equal with Python.
            dispatch_event(
                state,
                crate::cards::Event::DealtDamage,
                target_id,
            )?;
        }
        // DestroyTargetArtifactEnchantmentOrFlyingCreature
        // (Effect-uncovered single-gated tier — Shattered Wings,
        // ~7,793 17lands decks). CR 701.7 with a 3-way OR target
        // filter: (Artifact) OR (Enchantment) OR (Creature AND
        // has-flying). Mirrors Python's
        // `effects/generic/destroy.py:501 DestroyTargetArtifact
        // EnchantmentOrFlyingCreature` whose `TargetSpec` filter is
        // `IsArtifact() | IsEnchantment() | (IsCreature() &
        // HasKeyword("flying"))`, re-verified at resolution per CR
        // 608.2b against the centralized `state.battlefield` walk
        // (no controller restriction).
        //
        // **Cast-time-bound target via
        // `current_resolve_payload_target_id`** (the
        // `DestroyTargetArtifactOrTappedCreature` /
        // `SeedshipImpactEffect` payload-target swap pattern; CR
        // 608.2b). Pre-fix would have self-scanned for the first
        // matching permanent on either bf — diverging from Python
        // whenever the cast-time-bound target ceased between cast
        // and resolve (Python re-checks the ORIGINAL target id and
        // fizzles).
        //
        // **None-payload synthetic-test fallback.** The cross-engine
        // real-EoE probe + the spell-card-bridge use
        // `cast_spell_noncreature` which binds the target at cast
        // time. The low-level synthetic-test `push_to_stack` API
        // does NOT seat a target. For that path — and ONLY that
        // path — fall back to the deterministic-first scan
        // (identical observable to the OrTapped sibling).
        //
        // **Flying-leg reads EFFECTIVE keywords.**
        // `battlefield_has_keyword` folds printed + UEOT-granted +
        // aura-granted + anthem-granted keywords. Python's
        // `HasKeyword.matches` reads `effective_keywords` which is
        // the same fold — so a creature granted flying by an aura
        // / anthem / UEOT pump is legally targetable on BOTH
        // engines byte-equal.
        //
        // **DIES dispatch is creature-gated** (CR 700.4 — only a
        // creature death fires DIES). The R76-2 / R74-2 fix shape:
        // the artifact + enchantment legs do NOT fire DIES; the
        // flying-creature leg does. Push order: DIES via
        // `pending_dies_dispatches` (creature only), LEAVES via
        // `pending_leaves_dispatches` (always), token-cease
        // (always); the drain order in `resolve_top_of_stack`
        // matches the SBA path (DIES first then LEAVES; LIFO ⇒
        // LEAVES resolves first), byte-equal with Python's
        // `move_off_battlefield` sequence.
        Effect::DestroyTargetArtifactEnchantmentOrFlyingCreature => {
            // Re-derive the 3-way OR predicate privately (file-iso
            // — the `state_ops_legal.rs` `spell_targets_satisfiable`
            // arm has its own copy; same convention as the OrTapped
            // sibling). The "flying" leg reads effective keywords
            // via the established `battlefield_has_keyword` reader
            // (folds printed + UEOT + aura + anthem). The
            // indestructible-skip leg matches the shipped
            // `DestroyTargetArtifact` /
            // `DestroyTargetArtifactOrTappedCreature` precedent.
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                let is_artifact = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Artifact");
                                let is_enchantment = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Enchantment");
                                let is_creature = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature");
                                let has_flying = is_creature
                                    && crate::state_ops_combat::battlefield_has_keyword(
                                        state, o.id, "flying",
                                    );
                                // 3-way OR filter re-checked at
                                // resolve (defense-in-depth + the
                                // CR 608.2b "still matches the
                                // typed filter" leg).
                                if !(is_artifact
                                    || is_enchantment
                                    || has_flying)
                                {
                                    return None;
                                }
                                if printing
                                    .keywords
                                    .iter()
                                    .any(|k| k == "indestructible")
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => {
                    // Synthetic-test fallback — deterministic-first
                    // permanent matching the 3-way OR filter across
                    // BOTH bfs in ETB-timestamp order. Collect the
                    // candidate ids in a first pass (immutable
                    // borrow via the helper iter + a private
                    // `has_flying` check on each), then walk again
                    // to find the controller index by id. This
                    // two-pass shape is required because
                    // `battlefield_has_keyword` takes `&GameState`
                    // and the bf walk gives us `&GameObject` —
                    // both immutable borrows live happily side by
                    // side.
                    let mut picked: Option<(usize, u32)> = None;
                    for obj in
                        crate::bf_iter::bf_iter_etb_sorted(state).into_iter()
                    {
                        let Some(printing) =
                            state.registry.lookup(obj.printing_id)
                        else {
                            continue;
                        };
                        let is_artifact = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_enchantment = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Enchantment");
                        let is_creature = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        let has_flying = is_creature
                            && crate::state_ops_combat::battlefield_has_keyword(
                                state, obj.id, "flying",
                            );
                        if !(is_artifact || is_enchantment || has_flying) {
                            continue;
                        }
                        if printing
                            .keywords
                            .iter()
                            .any(|k| k == "indestructible")
                        {
                            continue;
                        }
                        picked = Some((obj.controller as usize, obj.id));
                        break;
                    }
                    picked
                }
            };
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here
                // (CR 302.1 — sickness is battlefield-only state).
                // See I35.
                obj.damage = 0;
                // R77-CFI1: counters NOT cleared on the dead object
                // (the established byte-equal contract; same as the
                // shipped `DestroyTargetArtifactOrTappedCreature`
                // arm).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) =
                    (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // LTB fan-out (the `mark_left_battlefield_if_nonland`
                // helper gates on the printing — artifact / enchantment
                // / creature all reach the LTB mark; lands cannot
                // satisfy the 3-way OR filter anyway).
                mark_left_battlefield_if_nonland(state, printing_id);
                // R76-2 / R74-2 DIES dispatch — creature-gated (CR
                // 700.4). The flying-creature leg fires DIES; the
                // artifact + enchantment legs do not (no spurious
                // DIES). Mirrors the shipped OrTapped sibling's
                // creature-gate.
                let victim_is_creature = state
                    .registry
                    .lookup(printing_id)
                    .map(|p| {
                        p.card_types
                            .iter()
                            .any(|t| t == "Creature")
                    })
                    .unwrap_or(false);
                if victim_is_creature {
                    state
                        .pending_dies_dispatches
                        .push(leaving_obj_id);
                }
                // LEAVES fan-out (always — every bf-exit fires
                // LEAVES per CR 603.6c / `Event::Leaves
                // Battlefield`'s contract).
                state.pending_leaves_dispatches.push(leaving_obj_id);
                // Round-55: token-cease sibling of the LEAVES push.
                // The id is queued unconditionally; the is_token
                // gate lives in drain_pending_token_cease (CR
                // 111.7 / 704.5d).
                crate::state_ops_zones::queue_token_cease(
                    state,
                    leaving_obj_id,
                );
            }
        }
        // ExorciseEffect (Effect-uncovered single-gated tier — Exorcise,
        // Duskmourn). CR 406 (exile) with a 3-way OR target filter:
        // (Artifact) OR (Enchantment) OR (Creature AND effective_power
        // >= 4). Mirrors Python's `ExorciseEffect`
        // (`effects/bespoke.py:2268`) whose `_matches` helper is
        // `obj.printing.is_artifact OR "Enchantment" in
        // obj.printing.card_types OR (obj.printing.is_creature AND
        // effective_power(state, obj) >= 4)`, re-verified at resolution
        // per CR 608.2b against the centralized `state.battlefield`
        // walk (no controller restriction).
        //
        // **Cast-time-bound target via
        // `current_resolve_payload_target_id`** (the
        // `DestroyTargetArtifactEnchantmentOrFlyingCreature` /
        // `SeedshipImpactEffect` payload-target swap pattern; CR
        // 608.2b). None-payload synthetic-test fallback: deterministic-
        // first permanent matching the 3-way OR filter in
        // ETB-timestamp order across BOTH bfs (mirrors the Shattered
        // Wings sibling's fallback).
        //
        // **Power-4 leg reads EFFECTIVE power** via the file-local
        // `read_doubling_power` helper (base after
        // `ueot_base_pt_override` / `ueot_becomes_land_creature_pt` /
        // printed, plus `ueot_power_bonus`, plus signed `obj.counters`
        // — matches Python's `effective_power` =
        // `_compute_effective_pt` at game.py:2138-2251). A 3/3 creature
        // pumped to effective power 4+ via counters / UEOT modifier IS
        // legally targetable on BOTH engines byte-equal.
        //
        // **Indestructible is NOT skipped** here — exile bypasses
        // indestructible per CR 702.12 (indestructible only prevents
        // destroy + lethal-damage, not exile). Matches the shipped
        // `Effect::ExileTarget { AnyCreature }` arm which has NO
        // indestructible-skip. Distinguishes Exorcise from the destroy
        // siblings (Shattered Wings, etc.) which DO skip indestructible.
        //
        // **DIES dispatch is creature-gated** (CR 700.4 — only a
        // creature death fires DIES). The R76-2 / R74-2 fix shape: the
        // artifact + enchantment legs do NOT fire DIES; the 4+ power
        // creature leg does. Wait — exile doesn't trigger DIES
        // (creatures going to exile go to exile, NOT to graveyard, so
        // they don't "die" per CR 700.4 — "die" specifically means
        // bf→graveyard). Skip the DIES push entirely. Only LEAVES +
        // token-cease are queued. Mirrors the shipped
        // `Effect::ExileTarget { AnyCreature }` arm which does NOT
        // push pending_dies_dispatches.
        Effect::ExorciseEffect => {
            // Re-derive the 3-way OR predicate privately (file-iso —
            // the `state_ops_legal.rs` `spell_targets_satisfiable` arm
            // has its own copy; same convention as the Shattered Wings
            // sibling). The "power 4+" leg reads effective power via
            // the file-local `read_doubling_power` helper (folds
            // base + ueot_power_bonus + obj.counters — matches Python's
            // `effective_power`).
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                let is_artifact = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Artifact");
                                let is_enchantment = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Enchantment");
                                let is_creature = printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature");
                                let power_4_plus = is_creature
                                    && read_doubling_power(
                                        state, o.id,
                                    ) >= 4;
                                // 3-way OR filter re-checked at
                                // resolve (defense-in-depth + the
                                // CR 608.2b "still matches the typed
                                // filter" leg). Indestructible is
                                // NOT a disqualifier — exile bypasses
                                // indestructible per CR 702.12.
                                if !(is_artifact
                                    || is_enchantment
                                    || power_4_plus)
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => {
                    // Synthetic-test fallback — deterministic-first
                    // permanent matching the 3-way OR filter across
                    // BOTH bfs in ETB-timestamp order. Mirrors the
                    // Shattered Wings sibling's fallback shape.
                    let mut picked: Option<(usize, u32)> = None;
                    for obj in
                        crate::bf_iter::bf_iter_etb_sorted(state)
                            .into_iter()
                    {
                        let Some(printing) =
                            state.registry.lookup(obj.printing_id)
                        else {
                            continue;
                        };
                        let is_artifact = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_enchantment = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Enchantment");
                        let is_creature = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        let power_4_plus = is_creature
                            && read_doubling_power(state, obj.id)
                                >= 4;
                        if !(is_artifact
                            || is_enchantment
                            || power_4_plus)
                        {
                            continue;
                        }
                        picked = Some((obj.controller as usize, obj.id));
                        break;
                    }
                    picked
                }
            };
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect(
                        "target_id from scan but not found on re-scan",
                    );
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_EXILE;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here
                // (CR 302.1 — sickness is battlefield-only state).
                // See I35.
                obj.damage = 0;
                // The shipped `ExileTarget { AnyCreature }` arm
                // RESETS `obj.counters = 0` on zone change (a card
                // entering a new zone is a new object per CR 400.7;
                // counters specifically reset on bf→exile per the
                // round-1 exile reset convention this arm mirrors).
                // Diverges intentionally from the destroy siblings,
                // which preserve counters on the dead-in-graveyard
                // object per the R77-CFI1 byte-equal contract — that
                // contract is specific to the bf→graveyard transition
                // and doesn't extend to bf→exile.
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) =
                    (obj.printing_id, obj.id);
                // First-slice: target's controller == owner, so push
                // onto `pidx`'s exile (where the target lives).
                Arc::make_mut(&mut state.players[pidx].exile).push(obj);
                // LTB fan-out (the `mark_left_battlefield_if_nonland`
                // helper gates on the printing — artifact / enchantment
                // / creature all reach the LTB mark; lands cannot
                // satisfy the 3-way OR filter anyway).
                mark_left_battlefield_if_nonland(state, printing_id);
                // DIES is NOT pushed — exile is bf→exile, not
                // bf→graveyard. Only the bf→graveyard transition
                // satisfies CR 700.4's "die" definition. Mirrors the
                // shipped `Effect::ExileTarget { AnyCreature }` arm
                // which also omits the DIES push.
                //
                // LEAVES fan-out (always — every bf-exit fires LEAVES
                // per CR 603.6c / `Event::LeavesBattlefield`'s
                // contract).
                state.pending_leaves_dispatches.push(leaving_obj_id);
                // Round-55: token-cease sibling of the LEAVES push.
                // The id is queued unconditionally; the is_token gate
                // lives in drain_pending_token_cease (CR 111.7 /
                // 704.5d).
                crate::state_ops_zones::queue_token_cease(
                    state,
                    leaving_obj_id,
                );
            }
        }
        // **Close Encounter — additional-cost-choice closure (Cycle
        // 3, 2026-05-21; ~16,098 17lands decks — highest-deck-count
        // BLOCKED card in the substrate-gated tier).** Mirrors
        // Python's `CloseEncounterEffect.resolve` at
        // `effects/bespoke.py:1094-1128` byte-equal:
        // 1. Read bound target (single creature) from
        //    `current_resolve_payload_target_id`. Synthetic-test
        //    fallback: first creature in `bf_iter_etb_sorted` order.
        // 2. CR 608.2b — re-validate at resolve: target must be a
        //    battlefield Creature. Illegal → silent fizzle.
        // 3. Read bound additional-cost choice from
        //    `current_resolve_additional_cost_choice_id` (Cycle 1
        //    transient). `None` → silent no-op (no choice was bound,
        //    e.g. the choose-your-creature flag wasn't set).
        // 4. Look up chosen object:
        //    - Battlefield: EFFECTIVE power (`read_doubling_power`).
        //    - Caster's exile + `warped`: PRINTED power
        //      (`printing.power.unwrap_or(0)`).
        //    - Gone: amount = 0.
        // 5. Apply damage marker, dispatch DEALT_DAMAGE.
        Effect::CloseEncounterEffect => {
            let target: Option<(usize, u32)> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => state
                    .players
                    .iter()
                    .enumerate()
                    .find_map(|(pidx, player)| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == payload_target)
                            .and_then(|o| {
                                let printing = state
                                    .registry
                                    .lookup(o.printing_id)?;
                                if !printing
                                    .card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                                {
                                    return None;
                                }
                                Some((pidx, o.id))
                            })
                    }),
                None => {
                    // Synthetic-test fallback (the established
                    // `CutPropulsionEffect` / `ExorciseEffect` None-
                    // payload fallback): deterministic-first creature
                    // in ETB-timestamp order across BOTH bfs.
                    let registry = &state.registry;
                    crate::bf_iter::bf_iter_etb_sorted(state)
                        .into_iter()
                        .find_map(|obj| {
                            let printing =
                                registry.lookup(obj.printing_id)?;
                            if printing
                                .card_types
                                .iter()
                                .any(|t| t == "Creature")
                            {
                                Some((obj.controller as usize, obj.id))
                            } else {
                                None
                            }
                        })
                }
            };
            let Some((pidx, target_id)) = target else {
                return Ok(()); // no eligible target → silent fizzle
            };
            // Read the cast-time-bound additional-cost choice id.
            // `None` means the choose-your-creature flag wasn't set
            // on the printing (back-compat fallback); resolve is a
            // no-op. Mirrors Python `if chosen_id is None: return`
            // at `bespoke.py:1106-1107`.
            let Some(chosen_id) =
                state.current_resolve_additional_cost_choice_id
            else {
                return Ok(());
            };
            // Look up the chosen object: battlefield first
            // (effective power), then caster's exile (printed power
            // for a warped creature card). Gone → amount 0.
            //
            // **Source-controller resolution.** We need the
            // caster's exile, not the active player's. Prefer the
            // transient `current_resolve_source_controller` (set by
            // the stack resolver around each `apply_effect` call),
            // falling back to `state.active_player` for back-compat
            // with direct `apply_effect_pub` callers. Same pattern
            // as `DealDamageAnyTarget` / `DealDamage`.
            let caster_idx = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let amount: i32 = {
                // Battlefield branch: scan BOTH bfs for the chosen
                // id. Mirrors Python's `next((o for o in
                // state.battlefield if o.id == chosen_id), None)`
                // at `bespoke.py:1108-1110` — `state.battlefield` is
                // the centralized cross-bf view.
                let bf_pick: Option<u32> = state
                    .players
                    .iter()
                    .find_map(|player| {
                        player
                            .battlefield
                            .iter()
                            .find(|o| o.id == chosen_id)
                            .map(|o| o.id)
                    });
                if let Some(bf_id) = bf_pick {
                    // EFFECTIVE power (printed + ueot_base_pt
                    // override + counters + ueot_power_bonus). Same
                    // reader the `ExorciseEffect` / `CutPropulsion
                    // Effect` arms use.
                    read_doubling_power(state, bf_id)
                } else {
                    // Caster's exile branch: scan for the chosen id
                    // among caster's exile cards. Match must also
                    // be a Creature AND warped — Python `:1117-1121`.
                    let exile_pick: Option<&GameObject> = state
                        .players[caster_idx]
                        .exile
                        .iter()
                        .find(|o| o.id == chosen_id);
                    match exile_pick {
                        Some(obj) if obj.warped => {
                            // PRINTED power (the exile pool is not
                            // on the battlefield; continuous
                            // effects don't apply per CR 110.1).
                            // Mirrors Python `chosen.printing.power
                            // or 0` at `bespoke.py:1121`.
                            state
                                .registry
                                .lookup(obj.printing_id)
                                .and_then(|p| p.power)
                                .unwrap_or(0)
                                as i32
                        }
                        // Gone (left bf in response between cast
                        // and resolve, or not warped, or not in
                        // caster's exile). Amount = 0 — Python
                        // `:1125-1127`.
                        _ => 0,
                    }
                }
            };
            if amount <= 0 {
                // No damage to apply. The Python arm at `:1128`
                // still calls `_apply_damage(state, targets[0],
                // amount)` which short-circuits inside on
                // `amount <= 0` (per `_apply_damage` at
                // `effects/base.py:286-288`). Byte-equal: skip the
                // damage path entirely. The receiver-side
                // DEALT_DAMAGE dispatch does NOT fire (no damage
                // was dealt — CR 119.2).
                return Ok(());
            }
            let amount_i16 = amount.min(i16::MAX as i32) as i16;
            // Apply damage marker via the proven shape
            // (`apply_damage_marker` body inlined). Same `Arc::make
            // _mut` + saturating_add as `CutPropulsionEffect` /
            // `InvasiveManeuversEffect`.
            let bf_mut: &mut Vec<GameObject> = Arc::make_mut(
                &mut state.players[pidx].battlefield,
            );
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.damage =
                        obj.damage.saturating_add(amount_i16);
                    break;
                }
            }
            // DEALT_DAMAGE receiver-side event (Python's
            // `_fire_enchanted_dealt_damage_triggers` call at the
            // tail of `_apply_damage`, `effects/base.py:303-305`).
            dispatch_event(
                state,
                crate::cards::Event::DealtDamage,
                target_id,
            )?;
        }
        // DestroyTargetLand. CR 701.7 with Land filter — Dust Bowl /
        // Petrified Field pattern. Walks BOTH players' battlefield
        // in ETB-timestamp order via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper, picking the
        // first Land that is not indestructible. Mirrors Python's
        // `effects/generic/destroy.py:DestroyTargetLand`
        // `TargetSpec(IsLand())` filter against the centralized
        // `state.battlefield` walk (no controller restriction). See
        // invariant I39.
        Effect::DestroyTargetLand => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if !printing.card_types.iter().any(|t| t == "Land") {
                        return None; // not a land
                    }
                    if printing.keywords.iter().any(|k| k == "indestructible") {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                });
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = bf_mut
                    .iter()
                    .position(|o| o.id == target_id)
                    .expect("target_id from scan but not found on re-scan");
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_GRAVEYARD;
                obj.tapped = false;
                // CE13 closure: `summoning_sick` is NOT reset here.
                // Python's `turn.move_off_battlefield` deliberately
                // omits the clear (CR 302.1 — sickness is battlefield-
                // only state). See I35.
                obj.damage = 0;
                // R77-CFI1: counters are NOT cleared on the dead
                // object — the established byte-equal contract
                // (Python `turn.move_off_battlefield` turn.py:231-240
                // / SBA destroy `state_ops_sba.rs` ~:417 / the closed
                // `DepressurizeEffect` arm all retain `+1/+1` counters
                // on the dead object). Prior stray `obj.counters = 0;`
                // removed (R77-CFI1; same class as the closed CFI-1).
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
                Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
                // Round-29 Void LTB: target is a Land by filter, so the
                // helper short-circuits to no-op — but call it anyway
                // for uniformity (the gate lives in one place, every
                // bf-exit site goes through the same code path).
                mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // DealDamageAnyTarget — Lightning Bolt's "any target" shape
        // (CR 701.4). The "any target" filter is intrinsic to the
        // variant; distinct from the existing DealDamage variant
        // which dispatches on TargetFilter::{AnyPlayer, AnyCreature}.
        //
        // **First-slice deterministic choice**: scan the defender's
        // battlefield top-to-bottom and pick the FIRST object whose
        // printing's `card_types` contains "Creature". If none, fall
        // back to the defender-as-player life-subtraction shape. This
        // composes the AnyCreature arm's damage-write shape with the
        // AnyPlayer arm's life-subtraction shape from the existing
        // DealDamage match. Future slice plumbs an explicit choice
        // callback.
        //
        // **Negative amount raises** at register time AND here as
        // defense-in-depth (mirrors the round-3 raise-on-negative
        // shape for DealDamage's amount; registered effects already
        // validated `amount >= 0`, but synthetic / injected printings
        // could bypass the parser).
        Effect::DealDamageAnyTarget { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DealDamageAnyTarget amount must be non-negative, got {amount}"
                )));
            }
            // **Controller resolution (I67 follow-up to the I59 /
            // I60 / I63 / I64 / I65 / I66 controller-of-effect
            // substrate).** Prefer the transient
            // `state.current_resolve_source_controller` (set by the
            // stack resolver around each `apply_effect` call),
            // falling back to `state.active_player` for back-compat
            // with direct `apply_effect_pub` callers. Same pattern
            // as `Effect::DealDamage` / `Effect::DamageEachOpponent`
            // — CR 119.5 / 608.2: an effect resolves under its
            // source's controller. For DealDamageAnyTarget, "any
            // target" defaulting to "the defender" means "the
            // SOURCE'S controller's opponent", not "the active
            // player's opponent".
            //
            // **Why the fix.** Pre-fix this arm computed
            // `defender_idx = 1 - state.active_player`. Wrong when
            // the resolving spell / triggered ability has a
            // controller != active_player. Worst-case example: p1's
            // creature has an ETB "deal 2 damage to any target"
            // trigger that fires on p0's turn (a p0 flicker
            // re-ETBs p1's creature). Pre-fix the trigger self-
            // damaged p1's first creature / life (defender_idx =
            // 1 - 0 = 1). Post-fix it correctly targets p0's first
            // creature / life (the trigger's controller's
            // opponent).
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender_idx = 1 - controller;
            let amount_i16 = (*amount).min(i16::MAX as i32) as i16;
            // First-slice deterministic choice: prefer defender's
            // first creature (mirrors the AnyCreature arm's
            // type-filtered scan); fall back to defender-as-player.
            let creature_target: Option<u32> = {
                let bf = &state.players[defender_idx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            if let Some(target_id) = creature_target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender_idx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            } else {
                // No creature on defender's battlefield — damage the
                // player directly. Mirror the AnyPlayer arm's
                // life-subtraction shape (raw `-=`, no saturation —
                // life is i32 and amounts are bounded by deck pool).
                state.players[defender_idx].life -= *amount;
            }
        }
        // Round-27 UEOT-framework proof. Source gets +P/+T UEOT.
        //
        // Mechanism: locate the source GameObject on whichever
        // battlefield holds it; push a
        // `UeotModifier::PowerToughnessBonus { power, toughness }`
        // entry onto the object's `ueot_modifiers` Vec. The effective
        // P/T readers in `state_ops_combat.rs` /
        // `state_ops_sba.rs::is_lethal` walk the Vec on top of
        // printed/counter values. The entry auto-expires at the
        // next `cleanup_step_for_active_player` /
        // `advance_phase`-through-PHASE_CLEANUP via the round-27
        // CR 514.3 UEOT-clear pass landed in `state_ops_cleanup.rs`.
        //
        // **Source-not-on-battlefield handling.** Mirrors the round-23
        // `PutPlusOneCounterOnSelf` arm: walk both players'
        // battlefields, no-op if the source isn't there (the source
        // could be an in-flight stack item that hasn't yet moved,
        // a graveyard-resident ability source for a future variant,
        // etc.). Python parity: `if source is None: return`.
        //
        // **0/0 bonus is a legal no-op** — registration accepts
        // it for parity edge cases. The Vec push still happens
        // (so the next cleanup-step UEOT-clear pass touches the
        // object), but downstream `ueot_power_bonus()` /
        // `ueot_toughness_bonus()` readers see zero — the only
        // observable effect is that the object goes through one
        // CoW clone on the next cleanup. Skip the Vec push when
        // both bonuses are zero, to keep the warm path allocation-
        // free.
        Effect::PumpSelfUntilEOT { power_bonus, toughness_bonus } => {
            if *power_bonus == 0 && *toughness_bonus == 0 {
                return Ok(());
            }
            // Walk both battlefields to find the source object.
            for pidx in 0..state.players.len() {
                if state.players[pidx]
                    .battlefield
                    .iter()
                    .any(|o| o.id == source_obj_id)
                {
                    let zone: &mut Vec<GameObject> =
                        Arc::make_mut(&mut state.players[pidx].battlefield);
                    for obj in zone.iter_mut() {
                        if obj.id == source_obj_id {
                            obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                                power: *power_bonus,
                                toughness: *toughness_bonus,
                            });
                            break;
                        }
                    }
                    break;
                }
            }
        }
        // Round-28 UEOT-coverage extension. Same mechanism as
        // PumpSelfUntilEOT but on a TARGET creature: the first
        // creature on the source's controller's battlefield (mirrors
        // Giant Growth-style oracle text "target creature gets +X/+Y
        // UEOT" — typically cast to buff your own creature). The
        // source's controller is looked up by finding the source
        // object's `controller` field across any zone; if the source
        // can't be located (rare; resolve_top fizzle path) the spell
        // is a silent no-op.
        //
        // **Eligibility = is_creature_now on the source-controller's
        // battlefield.** First creature in zone-iteration order wins;
        // empty battlefield (or no creatures on it) fizzles silently.
        // No creature-type filter beyond `is_creature_now` (per the
        // round-23 PutPlusOneCounterOnSelf precedent that gates on
        // `card_types contains "Creature"`).
        //
        // `(0, 0)` short-circuits before doing the lookup — pure
        // no-op without touching state.
        Effect::PumpTargetCreatureUntilEOT { power_bonus, toughness_bonus } => {
            if *power_bonus == 0 && *toughness_bonus == 0 {
                return Ok(());
            }
            // Locate source controller by scanning every zone.
            // Mirrors the same "source could be anywhere" search the
            // PumpSelfUntilEOT arm uses, extended to non-battlefield
            // zones because the source spell typically resolves from
            // the stack (technically zone=stack, but the source
            // GameObject lives in whatever zone the spell came from
            // before being put on the stack — first-slice: scan all
            // tracked zones).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untrackable → silent fizzle
            };
            // Pick the first creature on the controller's battlefield.
            // Creature-ness gates on the printing's `card_types`
            // (mirrors state_ops_cast.rs:304 and state_ops_legal.rs:571
            // — the canonical creature-detection pattern).
            let cidx = controller as usize;
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: *power_bonus,
                        toughness: *toughness_bonus,
                    });
                    break;
                }
            }
        }
        Effect::PumpTargetCreatureUntilEotGrantDiesReturnTapped {
            power_bonus,
            toughness_bonus,
        } => {
            // BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner
            // ETB. Same source-controller scan + first-creature-on-
            // own-bf target pick as `PumpTargetCreatureUntilEOT`, with
            // the "ANOTHER target creature you control" exclusion
            // (the chosen target must NOT be the source object — CR
            // 115.4 "another"). Push BOTH the +P/+T pump AND the
            // `GrantedDiesReturnTapped` marker the SBA destroy path
            // honors. (0/0) is still a legal target-and-mark op (the
            // rider applies even at +0/+0 — Perigee is +2/+0 but the
            // marker, not the pump, is the load-bearing observable),
            // so unlike the bare pump arm there is NO (0,0) short-
            // circuit. Empty / only-self own-bf → silent fizzle
            // (Python's non-optional `if not legal_targets: return`).
            let Some((cidx, target_id)) =
                perigee_pick_another_own_creature(state, source_obj_id)
            else {
                return Ok(()); // untrackable / no eligible OTHER creature
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: *power_bonus,
                        toughness: *toughness_bonus,
                    });
                    obj.ueot_modifiers
                        .push(UeotModifier::GrantedDiesReturnTapped);
                    break;
                }
            }
        }
        // BLOCKED-RUST-EFFECT slice (round 31). Double target
        // creature's power UEOT — snapshot the chosen target's
        // effective power AT RESOLVE TIME and push that as a
        // PowerToughnessBonus { power: snapshot, toughness: 0 }
        // UEOT modifier. Subsequent counter gains / pumps within the
        // turn do NOT re-double (mirrors Python's snapshot semantic
        // at pump.py:234-236).
        //
        // Source-controller resolution mirrors the
        // PumpTargetCreatureUntilEOT shape: prefer the I79
        // current_resolve_source_controller transient slot, fall
        // back to scanning every zone for the source object, and
        // ultimately default to active_player. The target picker
        // walks the source-controller's bf and picks the FIRST
        // creature (smallest Vec index — same shape as
        // PumpTargetCreatureUntilEOT). Snapshot is read via the
        // private `read_battlefield_creature_power` helper already in
        // place from the round-17 Fight slice (per the per-slice
        // file-isolation rule).
        //
        // Fizzle cases (silent — no modifier pushed):
        // - source-controller untrackable → silent fizzle (defaults
        //   to active_player after the I79 fallback chain so this is
        //   rare in practice).
        // - source-controller has no creatures on bf → silent fizzle.
        // - chosen target's effective power ≤ 0 at resolve time →
        //   silent fizzle (matches Python's `if snapshot <= 0:
        //   return` guard at pump.py:235).
        Effect::DoubleTargetCreaturesPowerUEOT => {
            // Source-controller resolution (I79 substrate).
            let src_controller: u8 = state
                .current_resolve_source_controller
                .or_else(|| {
                    for player in &state.players {
                        for zone in [
                            &player.hand,
                            &player.library,
                            &player.battlefield,
                            &player.graveyard,
                            &player.exile,
                        ] {
                            if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                                return Some(obj.controller);
                            }
                        }
                    }
                    None
                })
                .unwrap_or(state.active_player);
            let cidx = src_controller as usize;
            // Pick the first creature on the source-controller's bf.
            // Creature-ness gates on the printing's `card_types`
            // (mirrors PumpTargetCreatureUntilEOT's filter).
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            // Snapshot effective power at resolve time.
            //
            // **Round-53 §2 fix.** Was
            // `read_battlefield_creature_power` (the round-17 Fight
            // helper), which DELIBERATELY omits `obj.counters` (a
            // pinned combat-side asymmetry). Python's source-of-truth
            // `DoubleTargetCreaturesPowerUEOT.resolve` (`pump.py:234`)
            // snapshots `effective_power` = `_compute_effective_pt`,
            // which FOLDS IN +1/+1 counters (game.py:2251). On a
            // target with +1/+1 counters the two engines doubled
            // DIFFERENT values (Rust base, Python effective) — a real
            // §2 silent mistranslation, latent only because the
            // round-31/36 genuine Mightform probe staged a
            // counter-free creature. Now reads `read_doubling_power`
            // (base + ueot_power_bonus + obj.counters) so BOTH engines
            // double the SAME effective-power snapshot. Pinned by the
            // round-53 +1/+1-counter regression + cross-engine pins +
            // the extended genuine Mightform probe. Documented in
            // `docs/SEARCH_LIMITATIONS.md`.
            let snapshot = read_doubling_power(state, target_id);
            if snapshot <= 0 {
                return Ok(()); // ≤ 0 power → silent fizzle (no inversion)
            }
            // Push the snapshot as a UEOT power-only modifier.
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: snapshot,
                        toughness: 0,
                    });
                    break;
                }
            }
        }
        // BLOCKED-RUST-EFFECT slice (round 32). Deal damage to
        // target creature an opponent controls equal to the number
        // of artifacts the source's controller controls (Warmaker
        // Gunship's ETB; 1 EoE consumer). First dynamic-X-from-
        // permanent-count primitive in Rust.
        //
        // **Mechanism — count + apply.** Compute `amount` as the
        // number of permanents on the source-controller's bf whose
        // printing's `card_types` contains "Artifact" (matches
        // Python: lands that are also artifacts count, tokens like
        // Lander count). If amount ≤ 0 → silent fizzle (no targets
        // chosen, no damage). Otherwise route through the same
        // opponent-creature target shape as
        // `Effect::DealDamageToOpponentCreature`: scan opponents in
        // ascending player order, pick the first creature on each
        // opponent's bf, apply `amount` damage to it. No opponent
        // creature → silent fizzle.
        //
        // **Source-controller resolution.** Mirrors
        // `Effect::DealDamageToOpponentCreature` shape — 5-zone scan
        // + stack scan, falling through to silent fizzle if the
        // source isn't trackable (e.g. counterspelled). Distinct
        // from the I79 transient-slot lookup
        // (current_resolve_source_controller) used by
        // PumpTargetCreatureUntilEOT / DoubleTargetCreaturesPowerUEOT
        // because those arms target the source's OWN side and the
        // I79 slot is the canonical source-controller cache there.
        // For the `damage opponent` shape we already commit to the
        // 5-zone scan precedent set by DealDamageToOpponentCreature
        // for cross-arm consistency on this target shape.
        Effect::DealDamageToOpponentCreatureByArtifactCount => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors DealDamageToOpponentCreature shape).
            let mut source_controller: Option<u8> = None;
            'outer_sc: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Count source-controller's artifacts on bf. Mirrors
            // Python's `sum(1 for o in state.battlefield if
            // o.controller_id == controller_id and
            // o.printing.is_artifact)` at damage.py:445-448.
            // Scryfall's card_types convention has "Artifact" in
            // the card_types list for any artifact (including
            // artifact-creatures, artifact-lands, tokens with
            // Artifact type).
            let cidx = controller as usize;
            let mut amount: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Artifact") {
                    amount = amount.saturating_add(1);
                }
            }
            if amount <= 0 {
                return Ok(()); // no artifacts → silent fizzle (matches Python damage.py:449-450)
            }
            // Pick first eligible target: scan opponents in
            // ascending order, pick the first Creature on each
            // opponent's bf. Mirrors DealDamageToOpponentCreature
            // target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 51) — Monoist
        // Circuit-Feeder ETB (real EoE card; 14,727 17lands decks).
        // Until EOT: the source-controller's FIRST creature gets
        // +X/+0; an opponent's FIRST creature gets -0/-X; X = the
        // source-controller's artifact count (clamped >= 0; X<=0 ->
        // silent no-op). Sister of round-32
        // DealDamageToOpponentCreatureByArtifactCount: SAME
        // source-controller resolution (5-zone + stack scan) and
        // SAME artifact-count denominator, NEW two-subject until-EOT
        // pump+debuff apply substrate (reuses the
        // PumpTargetCreatureUntilEOT UeotModifier::PowerToughnessBonus
        // push shape). Mirrors Python's deterministic-pick bridge
        // sibling PumpYoursAndDebuffOpponentByArtifactCountUntilEOT.
        //
        // **Per-side independence (CR 608.2c).** The oracle uses TWO
        // independent "target" clauses (NOT "up to one"); if one
        // side has no legal creature the ability still resolves the
        // half that does. So each side is applied independently:
        // missing source-side creature -> source-side no-op (opponent
        // side still applies) and vice-versa. Matches Python's
        // per-slot independent apply in
        // PumpYoursAndDebuffOpponentByArtifactCountUntilEOT.
        Effect::TwoSubjectPumpAndDebuffByArtifactCount => {
            // Source-controller resolution (5-zone walk + stack
            // scan; mirrors round-32 sister shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_tspd: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_tspd;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Artifact-count denominator on the source-controller's
            // bf. Mirrors round-32
            // DealDamageToOpponentCreatureByArtifactCount BIT-FOR-BIT
            // (Scryfall card_types has "Artifact" for any artifact,
            // including artifact-creatures / artifact-lands / tokens
            // with the Artifact type) and Python's
            // `CountControlledMatching(IsArtifact())` formula.
            let cidx = controller as usize;
            let mut x: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) =
                    state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Artifact") {
                    x = x.saturating_add(1);
                }
            }
            if x <= 0 {
                return Ok(()); // X <= 0 -> silent no-op
            }
            // Source-side target: FIRST creature on the
            // source-controller's bf (smallest Vec index). Same
            // deterministic scan FightTargetCreatureWithTargetCreature
            // / PumpTargetCreatureUntilEOT use.
            let own_target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing =
                        registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            // Opponent-side target: FIRST creature on the opponent's
            // bf. 2-player engine = exactly one opponent at
            // 1 - controller.
            let opponent: u8 = 1 - controller;
            let oidx = opponent as usize;
            let opp_target_id: Option<u32> = {
                let bf = &state.players[oidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing =
                        registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            // Apply the source-side +X/+0 UEOT modifier (independent
            // of whether the opponent side has a target — CR 608.2c).
            if let Some(target_id) = own_target_id {
                let zone: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[cidx].battlefield);
                for obj in zone.iter_mut() {
                    if obj.id == target_id {
                        obj.ueot_modifiers.push(
                            UeotModifier::PowerToughnessBonus {
                                power: x,
                                toughness: 0,
                            },
                        );
                        break;
                    }
                }
            }
            // Apply the opponent-side +0/-X UEOT modifier
            // (independent of the source side).
            if let Some(target_id) = opp_target_id {
                let zone: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[oidx].battlefield);
                for obj in zone.iter_mut() {
                    if obj.id == target_id {
                        obj.ueot_modifiers.push(
                            UeotModifier::PowerToughnessBonus {
                                power: 0,
                                toughness: -x,
                            },
                        );
                        break;
                    }
                }
            }
            // else (either side): no eligible creature → that side
            // is a silent no-op; the other side still applied.
        }
        // BLOCKED-RUST-EFFECT slice (round 35) — deal damage to an
        // opponent's creature equal to the source-controller's hand
        // size. Sister of round-32's
        // DealDamageToOpponentCreatureByArtifactCount: same target
        // shape (first eligible Creature on an opponent's bf, scanned
        // in ascending player order), same fizzle semantics
        // (silent no-op when count <= 0 or no opponent creature),
        // same source-controller resolution (5-zone scan + stack
        // scan), NEW denominator (hand size in place of artifact
        // count). Mirrors Python's
        // `DealDamageToOpponentCreatureByCardsInHand`
        // (`damage.py:464` — Roaring Furnace's door trigger,
        // 1 EoE consumer).
        //
        // **Hand-size denominator.** `state.players[cidx].hand.len()`
        // BIT-FOR-BIT mirrors Python's
        // `len(state.player_by_id(controller_id).hand)` at
        // damage.py:511. The source itself is already on the
        // battlefield by the time the door / ETB trigger resolves
        // (it left the controller's hand on cast resolution), so the
        // hand-of-N + cast-source case produces N damage, not N+1 —
        // identical to Python.
        Effect::DealDamageToOpponentCreatureByCardsInHand => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-32 sister shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_chnd: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_chnd;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Read source-controller's hand size. Mirrors Python's
            // `hand_size = len(state.player_by_id(controller_id).hand)`
            // at damage.py:511. usize → i32 saturating cast (any
            // realistic hand fits in i32; defensive cap follows the
            // round-32 saturating-add denominator convention).
            let cidx = controller as usize;
            let hand_len = state.players[cidx].hand.len();
            let amount: i32 = if hand_len > i32::MAX as usize {
                i32::MAX
            } else {
                hand_len as i32
            };
            if amount <= 0 {
                return Ok(()); // empty hand → silent fizzle (matches Python damage.py:512-513)
            }
            // Pick first eligible target: scan opponents in ascending
            // order, pick the first Creature on each opponent's bf.
            // BIT-FOR-BIT mirror of the round-32 sister target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg_chnd: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg_chnd;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 36) — deal damage to an
        // opponent's creature equal to the source-controller's lands
        // count. Sister of round-32's
        // DealDamageToOpponentCreatureByArtifactCount and round-35's
        // DealDamageToOpponentCreatureByCardsInHand: same target shape
        // (first eligible Creature on an opponent's bf, scanned in
        // ascending player order), same fizzle semantics (silent
        // no-op when count <= 0 or no opponent creature), same
        // source-controller resolution (5-zone scan + stack scan),
        // NEW denominator (lands-controlled in place of artifact-count
        // / hand-size). Mirrors Python's
        // `DealDamageToOpponentCreatureByLandsControlled`
        // (shipped in this slice; 0 EoE consumers today, slice ships
        // per round-27/31 round-trip-surface-coverage rule and to
        // complete the THIRD orthogonal denominator on the round-32
        // damage-to-opponent-creature substrate).
        //
        // **Lands denominator.** Walk the source-controller's bf
        // once; for each obj, count +1 if `printing.card_types`
        // contains "Land". Mirrors the round-32 artifact-count
        // walking pattern BIT-FOR-BIT with "Artifact" replaced by
        // "Land" -- and matches Python's
        // `sum(1 for o in state.battlefield if o.controller_id ==
        // controller_id and o.printing.is_land)` denominator shape.
        // Dual-typed artifact-lands carry BOTH "Land" and "Artifact"
        // in card_types -- they count for both the artifact-count
        // and the lands-controlled denominators independently
        // (mirrors Scryfall typeline conventions and Python's
        // `is_land` / `is_artifact` disjunction at the printing
        // level).
        Effect::DealDamageToOpponentCreatureByLandsControlled => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-32 / round-35 sister shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_clnd: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_clnd;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Count source-controller's lands on bf. Mirrors Python's
            // `sum(1 for o in state.battlefield if o.controller_id ==
            // controller_id and o.printing.is_land)` shape. Walks the
            // round-32 artifact-count loop BIT-FOR-BIT with the type
            // string replaced.
            let cidx = controller as usize;
            let mut amount: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Land") {
                    amount = amount.saturating_add(1);
                }
            }
            if amount <= 0 {
                return Ok(()); // no lands → silent fizzle (matches Python's `if lands_count <= 0: return`)
            }
            // Pick first eligible target: scan opponents in
            // ascending order, pick the first Creature on each
            // opponent's bf. BIT-FOR-BIT mirror of the round-32 /
            // round-35 sister target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg_clnd: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg_clnd;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 37) -- deal damage to an
        // opponent's creature equal to the source-controller's
        // creatures count. Sister of round-32's
        // DealDamageToOpponentCreatureByArtifactCount, round-35's
        // DealDamageToOpponentCreatureByCardsInHand, and round-36's
        // DealDamageToOpponentCreatureByLandsControlled: same target
        // shape (first eligible Creature on an opponent's bf, scanned
        // in ascending player order), same fizzle semantics (silent
        // no-op when count <= 0 or no opponent creature), same
        // source-controller resolution (5-zone scan + stack scan),
        // NEW denominator (creatures-controlled in place of artifact
        // count / hand-size / lands-controlled). Mirrors Python's
        // `DealDamageToOpponentCreatureByCreaturesYouControl`
        // (shipped in this slice; 0 EoE consumers today, slice ships
        // per round-27/31 round-trip-surface-coverage rule and to
        // complete the FOURTH orthogonal denominator on the round-32
        // damage-to-opponent-creature substrate).
        //
        // **Creatures denominator.** Walk the source-controller's bf
        // once; for each obj, count +1 if `printing.card_types`
        // contains "Creature". Mirrors the round-32 artifact-count
        // walking pattern BIT-FOR-BIT with "Artifact" replaced by
        // "Creature" -- and matches Python's
        // `sum(1 for o in state.battlefield if o.controller_id ==
        // controller_id and o.printing.is_creature)` denominator
        // shape. The source itself counts toward the denominator if it
        // is a Creature still on the source-controller's bf at resolve
        // time (mirrors Python's vanilla walk over state.battlefield
        // -- no source-exclusion clause). Dual-typed artifact-
        // creatures carry BOTH "Creature" and "Artifact" in card_types
        // -- they count for both the artifact-count and the
        // creatures-controlled denominators independently (mirrors
        // Scryfall typeline conventions and Python's `is_artifact` /
        // `is_creature` disjunction at the printing level).
        Effect::DealDamageToOpponentCreatureByCreaturesYouControl => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-32 / round-35 / round-36 sister shape
            // exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_cyc: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_cyc;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Count source-controller's creatures on bf. Mirrors
            // Python's `sum(1 for o in state.battlefield if
            // o.controller_id == controller_id and
            // o.printing.is_creature)` shape. Walks the round-32
            // artifact-count loop BIT-FOR-BIT with the type string
            // replaced.
            let cidx = controller as usize;
            let mut amount: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Creature") {
                    amount = amount.saturating_add(1);
                }
            }
            if amount <= 0 {
                return Ok(()); // no creatures → silent fizzle (matches Python's `if creatures_count <= 0: return`)
            }
            // Pick first eligible target: scan opponents in ascending
            // order, pick the first Creature on each opponent's bf.
            // BIT-FOR-BIT mirror of the round-32 / round-35 / round-36
            // sister target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg_cyc: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg_cyc;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 46) -- deal damage to an
        // opponent's creature equal to the source-controller's
        // graveyard size. Sister of round-32's
        // DealDamageToOpponentCreatureByArtifactCount, round-35's
        // DealDamageToOpponentCreatureByCardsInHand, round-36's
        // DealDamageToOpponentCreatureByLandsControlled, and round-37's
        // DealDamageToOpponentCreatureByCreaturesYouControl: same
        // target shape (first eligible Creature on an opponent's bf,
        // scanned in ascending player order), same fizzle semantics
        // (silent no-op when count <= 0 or no opponent creature), same
        // source-controller resolution (5-zone scan + stack scan), NEW
        // denominator (graveyard size in place of artifact-count /
        // hand-size / lands-controlled / creatures-controlled).
        // Mirrors Python's
        // `DealDamageToOpponentCreatureByCardsInGraveyard`
        // (`damage.py` -- shipped in this slice; no pre-existing
        // class). FIFTH denominator on the damage substrate; SIXTH
        // overall X-from-count denominator across the substrate matrix
        // (prior five: artifact-count / hand-size / lands-controlled /
        // creatures-controlled / life-total). FIRST zone-size
        // denominator pulled from a non-battlefield, non-hand source.
        //
        // **Graveyard-size denominator.**
        // `state.players[cidx].graveyard.len()` BIT-FOR-BIT mirrors
        // Python's `len(state.player_by_id(controller_id).graveyard)`
        // shape. Mirrors the round-35 hand-size denominator's
        // `state.players[cidx].hand.len()` shape exactly with `hand`
        // swapped for `graveyard`. The source itself is on the
        // battlefield at resolve time for an ETB-style trigger -- it
        // does NOT contribute to the denominator (not in graveyard
        // yet). For an LTB-style or graveyard-resolve trigger where
        // the source has moved to the graveyard before this arm runs,
        // the source WOULD count by virtue of the move (no source-
        // exclusion clause -- mirrors round-32 / round-35 / round-36 /
        // round-37 sister convention).
        Effect::DealDamageToOpponentCreatureByCardsInGraveyard => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-32 / round-35 / round-36 / round-37 sister
            // shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_cgv: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_cgv;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Read source-controller's graveyard size. Mirrors
            // Python's
            // `gy_size = len(state.player_by_id(controller_id).graveyard)`
            // shape. usize -> i32 saturating cast (any realistic
            // graveyard fits in i32; defensive cap follows the
            // round-32 saturating-add denominator convention).
            let cidx = controller as usize;
            let gy_len = state.players[cidx].graveyard.len();
            let amount: i32 = if gy_len > i32::MAX as usize {
                i32::MAX
            } else {
                gy_len as i32
            };
            if amount <= 0 {
                return Ok(()); // empty graveyard → silent fizzle (matches Python's `if gy_size <= 0: return`)
            }
            // Pick first eligible target: scan opponents in ascending
            // order, pick the first Creature on each opponent's bf.
            // BIT-FOR-BIT mirror of the round-32 / round-35 / round-36
            // / round-37 sister target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg_cgv: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg_cgv;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 47) -- deal damage to an
        // opponent's creature equal to the source-controller's LIFE
        // TOTAL. Sister of round-32's
        // DealDamageToOpponentCreatureByArtifactCount, round-35's
        // DealDamageToOpponentCreatureByCardsInHand, round-36's
        // DealDamageToOpponentCreatureByLandsControlled, round-37's
        // DealDamageToOpponentCreatureByCreaturesYouControl, and
        // round-46's DealDamageToOpponentCreatureByCardsInGraveyard:
        // same target shape (first eligible Creature on an opponent's
        // bf, scanned in ascending player order), same fizzle
        // semantics (silent no-op when count <= 0 or no opponent
        // creature), same source-controller resolution (5-zone scan +
        // stack scan), NEW denominator (clamped life total in place of
        // artifact-count / hand-size / lands-controlled /
        // creatures-controlled / graveyard-size). Mirrors Python's
        // `DealDamageToOpponentCreatureByLifeTotal` (`damage.py` --
        // shipped in this slice; no pre-existing class). SIXTH (final)
        // denominator on the damage substrate -- the damage substrate
        // row of the X-from-count matrix is now fully saturated across
        // the artifact / hand / lands / creatures / graveyard /
        // life-total denominator axis.
        //
        // **Life-total denominator.** `state.players[cidx].life` is
        // i32; `life.max(0)` clamps negative life totals (arbitrary
        // lose-life-past-zero / infect-style scenarios; an SBA would
        // normally have ended the game) to zero -- silent no-op,
        // BIT-FOR-BIT matching Python's
        // `max(0, state.player_by_id(controller_id).life)` clamp and
        // the round-42 `MillTargetPlayerEqualToYourLifeTotal`
        // `life.max(0)` precedent at state_ops_resolve.rs:1276. The
        // life total is read from the source-controller (the player
        // whose creature/spell is the effect source), NOT the target
        // -- a "deals damage equal to your life total" effect reads
        // *your* (controller's) life.
        Effect::DealDamageToOpponentCreatureByLifeTotal => {
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors round-32 / round-35 / round-36 / round-37 /
            // round-46 sister shape exactly).
            let mut source_controller: Option<u8> = None;
            'outer_sc_lft: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_sc_lft;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // Read source-controller's life total and clamp negative
            // life to 0. Mirrors Python's
            // `n = max(0, state.player_by_id(controller_id).life)`
            // shape (round-42 mill `life.max(0)` precedent).
            let cidx = controller as usize;
            let life = state.players[cidx].life;
            let amount: i32 = life.max(0);
            if amount <= 0 {
                return Ok(()); // life <= 0 → silent fizzle (matches Python's `if life_total <= 0: return`)
            }
            // Pick first eligible target: scan opponents in ascending
            // order, pick the first Creature on each opponent's bf.
            // BIT-FOR-BIT mirror of the round-32 / round-35 / round-36
            // / round-37 / round-46 sister target shape.
            let mut found: Option<(usize, u32)> = None;
            'outer_tg_lft: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_tg_lft;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = amount.min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // BLOCKED-RUST-EFFECT slice (round 33) — put a +1/+1 counter
        // on the source for each OTHER creature and/or artifact the
        // source's controller controls. Mirrors Python's
        // `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact`
        // (`counters.py:394` — Luxknight Breacher's ETB).
        //
        // **Source-controller resolution.** Battlefield-only scan
        // (mirrors round-23 `PutPlusOneCounterOnSelf` shape). A
        // "+1/+1 counter on this creature" effect requires the source
        // to be on a battlefield zone; if it's gone (counterspelled,
        // bounced, etc.), silent fizzle.
        //
        // **Other-creature-or-artifact count.** Walk the source-
        // controller's battlefield once; for each obj other than the
        // source itself, count +1 if `printing.card_types` contains
        // "Creature" OR "Artifact". Artifact-creatures count once
        // (the disjunction short-circuits — same semantics as
        // Python's `is_creature or is_artifact`).
        //
        // **Apply.** If count > 0, add `count` to the source's
        // `counters` field via `saturating_add` on the
        // `Arc::make_mut`-borrowed bf vec. Mirrors the round-23
        // `PutPlusOneCounterOnSelf` mutation pattern bit-for-bit.
        //
        // **Fizzle paths.** Source not on bf → silent fizzle. Count
        // ≤ 0 (only the source on the controller's bf) → silent
        // fizzle (matches Python's `if count <= 0: return`).
        Effect::PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact => {
            // Locate the source on a battlefield zone (round-23
            // PutPlusOneCounterOnSelf shape).
            let mut found_player: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    found_player = Some(pidx);
                    break;
                }
            }
            let pidx = match found_player {
                Some(p) => p,
                None => return Ok(()), // source not on bf → silent fizzle
            };
            // Count source-controller's other creatures+artifacts.
            // Mirrors Python's `for o in state.battlefield: if
            // o.id == source.id: continue; if o.controller_id !=
            // controller_id: continue; if o.printing.is_creature or
            // o.printing.is_artifact: count += 1` at counters.py:411-418.
            let mut count: i32 = 0;
            for obj in state.players[pidx].battlefield.iter() {
                if obj.id == source_obj_id {
                    continue;
                }
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                let is_creature_or_artifact = printing
                    .card_types
                    .iter()
                    .any(|t| t == "Creature" || t == "Artifact");
                if is_creature_or_artifact {
                    count = count.saturating_add(1);
                }
            }
            if count <= 0 {
                return Ok(()); // no other creatures/artifacts → silent fizzle
            }
            // Apply: add `count` +1/+1 counters to source. Mirrors
            // round-23 PutPlusOneCounterOnSelf mutation pattern.
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == source_obj_id {
                    obj.counters = obj.counters.saturating_add(count);
                    break;
                }
            }
        }
        // BLOCKED-RUST-EFFECT slice (round 34) — surveil X, where X is
        // the source-controller's artifact count. Sister of round-32
        // DealDamageToOpponentCreatureByArtifactCount (same X-from-
        // artifact-count denominator) and round-15 Effect::Surveil
        // (same SurveilChoice PendingDecision shape). Resolves to a
        // SurveilChoice carrying min(artifact_count, library_len)
        // cards off the controller's library top, with the controller
        // resolved via I61 transient slot first then 5-zone+stack
        // scan fallback.
        Effect::SurveilXEqualToArtifactCount => {
            // Source-controller resolution — try I61 transient slot
            // first (mirrors round-15 Effect::Surveil), then fall back
            // to 5-zone scan + stack scan (mirrors round-32
            // DealDamageToOpponentCreatureByArtifactCount robustness).
            let mut source_controller: Option<u8> =
                state.current_resolve_source_controller;
            if source_controller.is_none() {
                'outer_sc: for player in state.players.iter() {
                    for zone in [
                        &player.hand,
                        &player.library,
                        &player.battlefield,
                        &player.graveyard,
                        &player.exile,
                    ] {
                        for obj in zone.iter() {
                            if obj.id == source_obj_id {
                                source_controller = Some(obj.controller);
                                break 'outer_sc;
                            }
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            // Final fallback: active_player. Preserves the "always
            // emit exactly one PendingDecision" contract that auto-
            // resolve depends on (mirrors round-15 Effect::Surveil's
            // unconditional push at state_ops_resolve.rs:2726-2730).
            let controller = source_controller.unwrap_or(state.active_player);
            let cidx = controller as usize;
            // Count source-controller's artifacts on bf. BIT-FOR-BIT
            // mirror of round-32's DealDamageToOpponentCreatureByArtifactCount
            // counting denominator at state_ops_resolve.rs:4550-4560.
            let mut count: i32 = 0;
            for obj in state.players[cidx].battlefield.iter() {
                let Some(printing) = state.registry.printings.get(obj.printing_id as usize)
                else {
                    continue;
                };
                if printing.card_types.iter().any(|t| t == "Artifact") {
                    count = count.saturating_add(1);
                }
            }
            // Pop min(count, library_len) cards off library TOP into
            // a SurveilChoice PendingDecision. IDENTICAL post-pop
            // state shape to round-15 Effect::Surveil resolve arm at
            // state_ops_resolve.rs:2705-2730. count == 0 (controller
            // has zero artifacts) is a legitimate no-op surveil — we
            // still push an empty SurveilChoice so the auto-resolve
            // invariant (every Surveil emits exactly one
            // PendingDecision) holds; auto-resolve discards it in
            // O(1) (matches round-15 count = 0 comment).
            let lib_len = state.players[cidx].library.len();
            let n = (count.max(0) as usize).min(lib_len);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(n);
            for _ in 0..n {
                // Always remove index 0 — the new top after each prior
                // remove shifts down. Mirrors round-15 Surveil's
                // identical pop loop.
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(PendingDecision::SurveilChoice {
                source_obj_id,
                controller,
                cards,
            });
        }
        // BLOCKED-RUST-EFFECT slice — Fight effect arm. "Target
        // creature you control fights target creature an opponent
        // controls" (CR 701.12). Both fighters deal damage equal to
        // their power to the other simultaneously; per CR 701.12c, if
        // either fighter has left the battlefield by resolve time, no
        // damage is dealt at all.
        //
        // Mirrors the Pump-Target-Creature target-pick shape: scan
        // the source controller's bf for the FIRST creature (caster-
        // side fighter), and the OPPONENT's bf for the FIRST creature
        // (opponent-side fighter). Both creature-eligibility checks
        // gate on `card_types contains "Creature"`.
        //
        // Cross-controller routing: source-controller is resolved by
        // scanning every zone for `source_obj_id` (mirrors
        // PumpTargetCreatureUntilEOT's source-controller lookup). If
        // the source is untrackable (already gone from every zone —
        // e.g. counterspelled?) it's a silent fizzle.
        //
        // Damage-application: compute BOTH effective-power values
        // BEFORE applying any damage so the simultaneous-exchange
        // semantics hold (CR 701.12 — a fighter that takes lethal
        // still deals its damage). Skip the per-side write when the
        // attacker's power is 0 or negative (negative power doesn't
        // heal in MTG; the printed-power 0 case fires no damage
        // marker change either way, but `deal_damage` would have
        // raised on a negative argument so we explicitly clamp to
        // skip the call). SBAs are NOT triggered inline by this
        // arm; they fire on the next SBA pass after the resolve, by
        // which time both damage markers are in place.
        Effect::FightTargetCreatureWithTargetCreature => {
            // Source-controller resolution — scan every zone for the
            // source object. Same shape as PumpTargetCreatureUntilEOT.
            // Untrackable source = silent fizzle (mirrors the pump
            // arm's behavior).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untrackable → silent fizzle
            };
            let opponent: u8 = 1 - controller;

            // Pick the first creature on the source-controller's bf
            // (the caster-side fighter). Mirrors
            // PumpTargetCreatureUntilEOT's same-side scan.
            let cidx = controller as usize;
            let oidx = opponent as usize;
            let registry = &state.registry;
            let caster_fighter_id: Option<u32> = state.players[cidx]
                .battlefield
                .iter()
                .find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                });
            // Pick the first creature on the OPPONENT's bf (the
            // opponent-side fighter). Mirrors DealDamageAnyTarget's
            // defender-side scan but creature-only (no fall-back to
            // damaging the player — fight is creature-vs-creature).
            let opponent_fighter_id: Option<u32> = state.players[oidx]
                .battlefield
                .iter()
                .find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                });
            // CR 701.12c: if either fighter is missing at resolve
            // time, the fight does NOT deal damage on either side.
            // Silent fizzle (mirrors the pump arm's no-target
            // behavior).
            let (Some(a_id), Some(b_id)) = (caster_fighter_id, opponent_fighter_id) else {
                return Ok(());
            };

            // Compute BOTH effective-power values BEFORE applying
            // any damage (CR 701.12 simultaneous-damage event).
            // Use the public `effective_power` getter shape
            // (state_ops_combat.rs:516) which returns the
            // printed-power + UEOT modifier sum, saturating into
            // i16. None for non-creatures or unknown ids — treat
            // as 0 defensively (the creature-filter scan above
            // should make this unreachable).
            let a_power = read_battlefield_creature_power(state, a_id);
            let b_power = read_battlefield_creature_power(state, b_id);

            // Apply both damages. `deal_damage` is the per-instance
            // damage-marker primitive (state_ops_damage.rs:196). It
            // raises on negative amount per the project's "fail
            // loud" convention, so explicitly clamp to >=0 (negative
            // power doesn't heal in MTG — it just deals 0 damage).
            // The two damages are independent: a fighter that
            // takes lethal in this exchange still dealt its damage
            // first because we read both powers BEFORE either
            // deal_damage call.
            if a_power > 0 {
                apply_damage_marker(state, b_id, a_power);
                // Round-16 RR Pin 3: dispatch DEALT_DAMAGE for the
                // creature that received damage in the fight (b_id).
                dispatch_event(
                    state,
                    crate::cards::Event::DealtDamage,
                    b_id,
                )?;
            }
            if b_power > 0 {
                apply_damage_marker(state, a_id, b_power);
                // Round-16 RR Pin 3: dispatch DEALT_DAMAGE for the
                // other fighter that received damage (a_id).
                dispatch_event(
                    state,
                    crate::cards::Event::DealtDamage,
                    a_id,
                )?;
            }
        }
        // Round-28 highest-frequency-class coverage. Sequence walks
        // its child Vec in order, calling apply_effect recursively on
        // each child with the same source_obj_id. Children share
        // resolve-time state — a Sequence of
        // [DrawCards { count: 1 }, DiscardSelf] resolves to "draw,
        // then discard" (a looting effect). Empty Vec is a legal
        // no-op. Recursion is safe: Effect is heap-allocated via
        // Vec<Effect>, and apply_effect is a regular function.
        //
        // **Round-36 fix (Sequence(Surveil, DrawCards) ordering)**:
        // After each leaf effect resolves, drain any PendingDecisions
        // it pushed via the default "keep_all_on_top" auto-resolve
        // strategy BEFORE running the next leaf effect. This matches
        // Python's no-agent flow where Surveil's resolve never mutates
        // the library, so a following DrawCards-N draws from the
        // pre-Surveil top. Without this drain, Rust's
        // SurveilXEqualToArtifactCount (and any sister arm that pushes
        // a PendingDecision) leaves cards popped off the library top;
        // the next leaf (e.g. DrawCards-3 in Cerebral Download) then
        // reads the post-pop top and draws DIFFERENT cards than Python.
        // The terminal `auto_resolve_pending` call at
        // `resolve_top_of_stack` (line ~8533) re-inserts the surveiled
        // cards AFTER the entire spell loop, which is too late for the
        // intra-Sequence ordering. Driving the drain inside the loop
        // restores cross-engine parity for Cerebral Download
        // (Sequence(Surveil(use_artifact_count=True), DrawCards(3)))
        // and any future Sequence-of-pending-then-zone-reading shape.
        // See `docs/SEARCH_LIMITATIONS.md` "Sequence(Surveil,
        // DrawCards) PendingDecision-vs-inline ordering".
        Effect::Sequence { effects } => {
            for child in effects {
                apply_effect(state, child, source_obj_id)?;
                auto_resolve_pending(state, "keep_all_on_top")?;
            }
        }
        // Round-28 anthem-shape UEOT effect. Iterates every creature
        // on the source's controller's battlefield and pushes a
        // UeotModifier::PowerToughnessBonus onto each. Creature
        // eligibility gates on `card_types contains "Creature"`
        // (mirrors PumpTargetCreatureUntilEOT's predicate).
        //
        // `(0, 0)` short-circuits before doing the scan. Empty
        // battlefield or no eligible creatures is a silent no-op.
        Effect::PumpCreaturesYouControlUntilEOT { power_bonus, toughness_bonus } => {
            if *power_bonus == 0 && *toughness_bonus == 0 {
                return Ok(());
            }
            // Locate source controller (mirrors DiscardSelf /
            // PumpTargetCreatureUntilEOT pattern).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(());
            };
            let cidx = controller as usize;
            // Collect creature ids first (immutable borrow on
            // registry + battlefield), then apply mutations.
            let creature_ids: Vec<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter()
                    .filter_map(|obj| {
                        let printing = registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            if creature_ids.is_empty() {
                return Ok(());
            }
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if creature_ids.contains(&obj.id) {
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: *power_bonus,
                        toughness: *toughness_bonus,
                    });
                }
            }
        }
        // Round-28 coverage extension. Atmospheric-Greenhouse-shape:
        // put `amount` +1/+1 counter(s) on each creature the source's
        // controller controls. Per-object scalar mutation; non-creature
        // permanents skipped. amount=0 short-circuits; negative raises
        // (defense-in-depth — parse_effect catches at register time).
        Effect::PutPlusOneCounterOnEachCreatureYouControl { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutPlusOneCounterOnEachCreatureYouControl amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            // Locate source controller (mirrors PumpCreaturesYouControl).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(());
            };
            let cidx = controller as usize;
            let creature_ids: Vec<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter()
                    .filter_map(|obj| {
                        let printing = registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            if creature_ids.is_empty() {
                return Ok(());
            }
            let delta = *amount;
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if creature_ids.contains(&obj.id) {
                    obj.counters = obj.counters.saturating_add(delta);
                }
            }
        }
        // Round-28 single-target counter add. Rayblade-Trooper-shape:
        // pick the FIRST creature on source-controller's bf and add
        // `amount` to its `counters` scalar. Empty bf or no eligible
        // creatures fizzles silently. amount=0 short-circuits;
        // negative raises.
        Effect::PutPlusOneCounterOnCreatureYouControl { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutPlusOneCounterOnCreatureYouControl amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(());
            };
            let cidx = controller as usize;
            // Pick first creature on controller's bf.
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(());
            };
            let delta = *amount;
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.counters = obj.counters.saturating_add(delta);
                    break;
                }
            }
        }
        // **PPP Pin 2 (10th Rust slice).** PutPlusOneCounterOnTargetCreature:
        // bare "target creature" (no controller restriction). Reads
        // the payload-target seat (set at push time by
        // `first_legal_trigger_target`). Optional `attacking_only`
        // rider restricts to attacking creatures only (Hospital Room
        // half). amount<0 raises; amount=0 short-circuits.
        //
        // **CR 608.2b resolve-time re-legality.** The seated target
        // must still be on SOME bf and still a Creature (and still
        // attacking when `attacking_only=true`). Failure → silent
        // fizzle, byte-equal with Python's
        // `PutPlusOneCounters.resolve`/`_slot_target_still_legal`.
        //
        // **Unseeded payload.** None → silent fizzle (mirrors the
        // `PutPlusOneCounterOnOpponentsCreature` defense-in-depth).
        // The production push path always seats via
        // `lever_trigger_requires_target` + `first_legal_trigger_target`.
        Effect::PutPlusOneCounterOnTargetCreature {
            amount,
            attacking_only,
        } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutPlusOneCounterOnTargetCreature amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            let payload_target = match state.current_resolve_payload_target_id {
                Some(t) => t,
                None => return Ok(()),
            };
            // Scan both bfs for the target. The target may be on
            // EITHER player's bf (no controller restriction here).
            let mut tgt_loc: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if let Some(obj) =
                    player.battlefield.iter().find(|o| o.id == payload_target)
                {
                    let printing = match state
                        .registry
                        .printings
                        .get(obj.printing_id as usize)
                    {
                        Some(p) => p,
                        None => return Ok(()),
                    };
                    if !printing.card_types.iter().any(|t| t == "Creature") {
                        return Ok(()); // no longer a creature → fizzle
                    }
                    if *attacking_only {
                        // Check the engine's attacker registry — a
                        // Vec<u32> of attacker object ids.
                        if !state.attackers.contains(&obj.id) {
                            return Ok(()); // no longer attacking → fizzle
                        }
                    }
                    tgt_loc = Some(pidx);
                    break;
                }
            }
            let Some(tidx) = tgt_loc else {
                return Ok(()); // target left the bf → silent fizzle
            };
            let delta = *amount;
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[tidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == payload_target {
                    obj.counters = obj.counters.saturating_add(delta);
                    break;
                }
            }
        }
        // **Tier-C target-seating sub-substrate (lever-a, Sunset
        // Saboteur).** Put `amount` +1/+1 counter(s) on a target
        // creature an OPPONENT of the source's controller controls.
        // Mirrors Python's `PutPlusOneCounterOnOpponentsCreature`
        // (a `PutPlusOneCounters` subclass; resolve walks its single
        // `TargetSpec(IsCreature() & ControlledByOpponent())` slot,
        // re-checks legality per CR 608.2b, and adds `amount` via
        // `add_plus_one_counters`).
        //
        // **Payload-target convention (NOT a self-scan).** Unlike the
        // you-control sibling arm above (which deterministically scans
        // the source-controller's own bf at apply time), this arm
        // reads the push-time-seated target from
        // `state.current_resolve_payload_target_id` exactly like
        // `Effect::EquipToTargetCreatureYouControl`. The seat is
        // computed by `first_legal_trigger_target` at
        // `push_triggered_ability` time (the Tier-C shared-seat
        // micro-substrate) so BOTH engines pick the SAME object: the
        // deterministic smallest-bf-index creature an opponent of the
        // source's controller controls. The CR 608.2b resolve-time
        // re-legality check (target still on a bf, still a creature,
        // still opponent-controlled) silently fizzles a target that
        // left the bf / changed control between push and resolve,
        // byte-equal with Python's
        // `PutPlusOneCounters.resolve`/`_slot_target_still_legal`
        // re-check. `amount=0` short-circuits; negative raises
        // (defense-in-depth — parse catches it at register time).
        Effect::PutPlusOneCounterOnOpponentsCreature { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutPlusOneCounterOnOpponentsCreature amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            // Locate the source's controller (the same five-zone walk
            // + stack fallback the you-control sibling arm uses).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untracked → silent fizzle
            };
            // CR 608.2b resolve-time re-legality: the seated target
            // must still be on SOME player's bf, be a Creature, and be
            // controlled by an OPPONENT of `controller`. An unseeded
            // payload (no `first_legal_trigger_target` ran, e.g. a
            // direct `apply_effect_pub` caller) → silent fizzle, the
            // same observable as Python's no-target no-op (the seat is
            // the production path; this is defense-in-depth, mirrors
            // the EquipToTargetCreatureYouControl `None` fall-through
            // but WITHOUT a first-eligible self-scan — a self-scan
            // here is exactly the cross-engine divergence the shared
            // seat eliminates).
            let payload_target = match state.current_resolve_payload_target_id {
                Some(t) => t,
                None => return Ok(()),
            };
            let mut tgt_loc: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if let Some(obj) =
                    player.battlefield.iter().find(|o| o.id == payload_target)
                {
                    // Opponent-controlled (controller of the obj is not
                    // the source's controller) AND a creature.
                    if obj.controller == controller {
                        return Ok(()); // became own creature → fizzle
                    }
                    let printing = match state
                        .registry
                        .printings
                        .get(obj.printing_id as usize)
                    {
                        Some(p) => p,
                        None => return Ok(()),
                    };
                    if !printing.card_types.iter().any(|t| t == "Creature") {
                        return Ok(()); // no longer a creature → fizzle
                    }
                    tgt_loc = Some(pidx);
                    break;
                }
            }
            let Some(tidx) = tgt_loc else {
                return Ok(()); // target left the bf → silent fizzle
            };
            let delta = *amount;
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[tidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == payload_target {
                    obj.counters = obj.counters.saturating_add(delta);
                    break;
                }
            }
        }
        // **Tier-C target-seating sub-substrate (lever-a, Exosuit
        // Savior — the count_min=0 optional-bounce edge).** Return
        // the push-time-SEATED target permanent (the source's
        // controller's own non-source non-player permanent) to its
        // owner's hand. Mirrors Python's
        // `ReturnTargetPermanentYouControlToOwnersHand` (a `Bounce`
        // subclass; `Bounce.resolve` no-ops on empty `targets` — the
        // optional decline).
        //
        // **The count_min=0 optional edge.** No seat
        // (`current_resolve_payload_target_id` is `None` — the
        // shared seat returned `None` because no eligible own
        // permanent existed, so `push_triggered_ability` pushed via
        // the unchanged `new_trigger` path) ⇒ silent no-op, byte
        // equal with Python's `targets=()` `Bounce.resolve`
        // early-`return`. NOT a fizzle-skip-push (that is Sunset
        // Saboteur's count_min>=1 semantics).
        //
        // **Payload-target convention + CR 608.2b.** When seated,
        // re-validate at resolve: the target must still be on the
        // source's controller's battlefield, not be the source, and
        // be a permanent (it always is — players are excluded by the
        // seat filter; the re-check is defense-in-depth + the
        // left-bf / changed-control fizzle). Bounce lifecycle mirrors
        // the `Effect::ReturnToHand` AnyCreature arm: zone→HAND,
        // reset damage/counters/attach, mark left-bf-if-nonland, LTB
        // fan-out via `pending_leaves_dispatches` + the round-55
        // token-cease sibling.
        Effect::ReturnTargetPermanentYouControlToOwnersHand => {
            // Locate the source's controller (the same five-zone walk
            // + stack fallback the counter arms use).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untracked → silent fizzle
            };
            // count_min=0 optional: no seat → no-op (the optional was
            // declined; byte-equal with Python's targets=() no-op).
            let payload_target = match state.current_resolve_payload_target_id {
                Some(t) => t,
                None => return Ok(()),
            };
            // CR 608.2b resolve-time re-legality: still on the
            // source-controller's bf, not the source itself. (Players
            // are never seated — the filter excludes them — but the
            // not-source guard mirrors Python's ~IsSource() slot.)
            let cidx = controller as usize;
            let pos = match state.players[cidx]
                .battlefield
                .iter()
                .position(|o| o.id == payload_target && o.id != source_obj_id)
            {
                Some(p) => p,
                None => return Ok(()), // left bf / became illegal → fizzle
            };
            // Bounce lifecycle — mirrors the Effect::ReturnToHand
            // AnyCreature arm (the established bounce-to-hand shape).
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            // CE13: summoning_sick NOT reset (Python's
            // move_off_battlefield omits it; re-ETB restamps it).
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            // Owner == the source's controller (Exosuit Savior bounces
            // a permanent ITS controller controls; controller == owner
            // for the EoE consumer — same owner/controller convention
            // as the ReturnToHand first-slice arm).
            Arc::make_mut(&mut state.players[cidx].hand).push(obj);
            // Round-29 Void LTB tracking: bounce is a battlefield-exit;
            // flag flips iff the bounced object is nonland.
            mark_left_battlefield_if_nonland(state, printing_id);
            // LTB fan-out + round-55 token-cease sibling — queued for
            // the post-pop drain in resolve_top_of_stack (identical to
            // the ReturnToHand arm).
            state.pending_leaves_dispatches.push(leaving_obj_id);
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-28 Lotus-Field-shape mana minting. First-slice
        // deterministic color choice: pick options[0]. Mirrors
        // Python's "no-agent-seated" fallback. amount=0 is a no-op;
        // negative raises (defense-in-depth, parse catches at register
        // time). Empty options Vec would have failed at parse — treat
        // as defensive invariant here.
        Effect::AddNManaOfChosenColor { amount, options } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "AddNManaOfChosenColor amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(());
            }
            let Some(chosen) = options.first() else {
                return Err(PyValueError::new_err(
                    "AddNManaOfChosenColor: options is empty",
                ));
            };
            // Source controller (mirrors AddManaToPool's active-player
            // shape — first-slice convention).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let controller_idx = match src_controller {
                Some(c) => c as usize,
                None => state.active_player as usize,
            };
            let amount_u8 = (*amount).min(u8::MAX as i32) as u8;
            let pool = &mut state.players[controller_idx].mana_pool;
            // G7 S6-Class-E ordered-pool: route through `push_coins`
            // so the insertion-ordered `coins` vec and the per-color
            // count cache stay synced.
            let color_byte = match chosen.as_str() {
                "W" => b'W',
                "U" => b'U',
                "B" => b'B',
                "R" => b'R',
                "G" => b'G',
                "C" => b'C',
                other => {
                    return Err(PyValueError::new_err(format!(
                        "AddNManaOfChosenColor: unknown color {other:?}; valid: W, U, B, R, G, C"
                    )));
                }
            };
            pool.push_coins(color_byte, amount_u8);
            pool.assert_count_sync();
        }
        // Round-28 generalized library tutor. Scans the source-
        // controller's library top-to-bottom for the FIRST card
        // matching filter_name; moves match to controller's hand
        // (revealing implicit). Unknown filter_name silently fizzles
        // (matches Python's _matches default-False). Deferred: shuffle
        // (G19), may_fail interactive choice (no decision callback yet
        // for "decline fetch").
        //
        // **Round-30 supertypes/subtypes substrate.** The
        // `legendary_creature` / `spacecraft` / `sliver` / `demon`
        // filters that Round-21 documented as silent-fizzling are now
        // dispatched off the new `supertypes` field
        // (cards.rs:`CardPrinting.supertypes`) and the existing
        // `subtypes` field. Mirrors Python's `SearchLibraryForCard.
        // _matches` at `src/mtg_bot/effects/generic/search.py:48-81`
        // arm-for-arm. Unknown filter still silent-fizzles via the
        // `_ => false` fallthrough (matches Python's terminal `return
        // False`).
        Effect::SearchLibraryForCard {
            filter_name,
            max_mv,
            may_fail: _,
        } => {
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let cidx = match src_controller {
                Some(c) => c as usize,
                None => state.active_player as usize,
            };
            // Top-to-bottom scan via forward iter (post-CE3: lib[0]
            // is top).
            let max_mv_local = *max_mv;
            let pos: Option<usize> = {
                let lib = &state.players[cidx].library;
                let registry = &state.registry;
                let mut found: Option<usize> = None;
                for (i, obj) in lib.iter().enumerate() {
                    let printing = match registry.lookup(obj.printing_id) {
                        Some(p) => p,
                        None => continue,
                    };
                    let matches = match filter_name.as_str() {
                        "basic_land" => printing.card_types.iter().any(|t| t == "Land"),
                        "basic_plains" => &*printing.name == "Plains",
                        "basic_island" => &*printing.name == "Island",
                        "basic_swamp" => &*printing.name == "Swamp",
                        "basic_mountain" => &*printing.name == "Mountain",
                        "basic_forest" => &*printing.name == "Forest",
                        "creature_small" => {
                            if !printing.card_types.iter().any(|t| t == "Creature") {
                                false
                            } else {
                                let mc = &printing.mana_cost;
                                let total = mc.generic as i32
                                    + mc.white as i32
                                    + mc.blue as i32
                                    + mc.black as i32
                                    + mc.red as i32
                                    + mc.green as i32;
                                total <= max_mv_local
                            }
                        }
                        "artifact" => printing.card_types.iter().any(|t| t == "Artifact"),
                        // Round-30 supertypes-substrate dispatch. Legendary
                        // creature card: type line carries both "Legendary"
                        // (supertype) and "Creature" (card type). Mirrors
                        // Python's `printing.is_creature and "Legendary" in
                        // printing.supertypes` at search.py:50-53.
                        "legendary_creature" => {
                            printing.card_types.iter().any(|t| t == "Creature")
                                && printing.supertypes.iter().any(|s| s == "Legendary")
                        }
                        // Round-30 subtypes-keyed dispatch. Spacecraft is a
                        // subtype (CR 205.3p). Mirrors Python's
                        // `printing.is_spacecraft` at search.py:70-71 which
                        // reads `"Spacecraft" in printing.subtypes`. Note
                        // we DON'T also gate on `card_types contains
                        // "Artifact"` — Spacecraft printings carry the
                        // Artifact card type today, but the filter mirrors
                        // Python's pure-subtype check so a hypothetical
                        // non-Artifact Spacecraft (off-set fixture, future
                        // type-changing effect) still matches.
                        "spacecraft" => {
                            printing.subtypes.iter().any(|s| s == "Spacecraft")
                        }
                        // Round-30 subtypes-keyed dispatch. Sliver is a
                        // creature subtype. Mirrors Python's
                        // `"Sliver" in printing.subtypes` at search.py:72-73
                        // (no creature-card-type gate on the Python side
                        // either — the subtype implies the creature type
                        // for any well-formed printing, and the bridge
                        // mirrors that contract bit-for-bit).
                        "sliver" => {
                            printing.subtypes.iter().any(|s| s == "Sliver")
                        }
                        // Round-30 subtypes-keyed dispatch. Demon is a
                        // creature subtype (Demonic Counsel from Duskmourn).
                        // Mirrors Python's `"Demon" in printing.subtypes`
                        // at search.py:79-80.
                        "demon" => {
                            printing.subtypes.iter().any(|s| s == "Demon")
                        }
                        _ => false, // unknown filter → fizzle
                    };
                    if matches {
                        found = Some(i);
                        break;
                    }
                }
                found
            };
            let Some(p) = pos else {
                return Ok(()); // no match → silent fizzle
            };
            let player = &mut state.players[cidx];
            let lib = Arc::make_mut(&mut player.library);
            let mut obj = lib.remove(p);
            obj.zone = ZONE_HAND;
            obj.tapped = false;
            obj.summoning_sick = true; // r28 CE1 universal-true default
            obj.controller = cidx as u8;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut player.hand).push(obj);
            // No shuffle (G19 deferral).
        }
        // Round-28 debuff anthem. Mirror of
        // PumpCreaturesYouControlUntilEOT but on the source-
        // controller's OPPONENTS' battlefields. Two-player engine
        // simplification: opponents = the one player whose id is
        // 1 - controller. (0, 0) short-circuits; empty opponent bf or
        // no creatures fizzles silently.
        Effect::PumpCreaturesOpponentsControlUntilEOT {
            power_bonus,
            toughness_bonus,
        } => {
            if *power_bonus == 0 && *toughness_bonus == 0 {
                return Ok(());
            }
            // Source controller lookup.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(());
            };
            let opponent_idx = (1 - controller) as usize;
            let creature_ids: Vec<u32> = {
                let bf = &state.players[opponent_idx].battlefield;
                let registry = &state.registry;
                bf.iter()
                    .filter_map(|obj| {
                        let printing = registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            if creature_ids.is_empty() {
                return Ok(());
            }
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[opponent_idx].battlefield);
            for obj in zone.iter_mut() {
                if creature_ids.contains(&obj.id) {
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: *power_bonus,
                        toughness: *toughness_bonus,
                    });
                }
            }
        }
        // Mouth-of-the-Storm sibling — extended-duration debuff
        // anthem ("until your next turn"). First-slice resolve
        // semantics are IDENTICAL to the bare-EOT sibling above:
        // same source-controller lookup (5-zone walk + stack
        // fallback); same opponent-bf creature scan; same
        // `UeotModifier::PowerToughnessBonus` push. The cleanup-
        // gate divergence (Python keeps the modifier into the
        // opponent's turn, Rust drops it at the casting-turn
        // cleanup) is a deferred follow-up Rust slice — see
        // SEARCH_LIMITATIONS.md and the doc-comment on
        // `Effect::PumpCreaturesOpponentsControlUntilCasterNextTurn`
        // in cards.rs. The discriminator carries the parametric
        // distinction so the bridge can route Mouth of the Storm
        // cleanly TODAY and the cleanup-gate slice can swap the
        // modifier-push line below WITHOUT any bridge / call-site
        // changes.
        Effect::PumpCreaturesOpponentsControlUntilCasterNextTurn {
            power_bonus,
            toughness_bonus,
        } => {
            if *power_bonus == 0 && *toughness_bonus == 0 {
                return Ok(());
            }
            let mut src_controller: Option<u8> = None;
            'outer2: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer2;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(());
            };
            let opponent_idx = (1 - controller) as usize;
            let creature_ids: Vec<u32> = {
                let bf = &state.players[opponent_idx].battlefield;
                let registry = &state.registry;
                bf.iter()
                    .filter_map(|obj| {
                        let printing = registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            if creature_ids.is_empty() {
                return Ok(());
            }
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[opponent_idx].battlefield);
            for obj in zone.iter_mut() {
                if creature_ids.contains(&obj.id) {
                    // FIRST-SLICE: same UEOT modifier as the bare
                    // sibling. The cleanup-gate slice will swap
                    // this for a tagged-lifetime payload that
                    // expires at the source-controller's untap
                    // step instead of the next cleanup.
                    obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                        power: *power_bonus,
                        toughness: *toughness_bonus,
                    });
                }
            }
        }
        // Round-28 graveyard-to-library-bottom move. Scans
        // graveyards in player order; takes graveyard[0] (first card)
        // from whichever player has a non-empty graveyard; moves to
        // that player's library bottom (lib[0] per Rust's convention).
        // Empty-graveyards-both-sides = silent fizzle.
        Effect::PutCardFromGraveyardOnBottomOfLibrary => {
            let _ = source_obj_id;
            let mut found: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if !player.graveyard.is_empty() {
                    found = Some(pidx);
                    break;
                }
            }
            let Some(pidx) = found else {
                return Ok(()); // no graveyards have any cards
            };
            let gy_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].graveyard);
            let mut obj = gy_mut.remove(0);
            obj.zone = ZONE_LIBRARY;
            obj.tapped = false;
            obj.summoning_sick = true; // r28 CE1 universal-true default
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            lib_mut.push(obj); // bottom of library (post-CE3: tail)
        }
        // Round-28 Dockworker-Drone-shape counter move. Transfer the
        // source's counters scalar to the first creature on the
        // source-controller's battlefield; clear source's counters
        // on transfer (true "move" semantics, see variant docstring).
        // No-op if source has 0 counters, isn't found, has no
        // controller, or no creature exists on controller's bf.
        Effect::MoveSelfCountersToTargetCreatureYouControl => {
            // Find source: pull counter count + controller. Search all
            // zones (typical use is dies-trigger so source may be in
            // graveyard).
            let mut src_info: Option<(u8, usize, i32)> = None; // (controller, player_idx, count)
            'outer: for (pidx, player) in state.players.iter().enumerate() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_info = Some((obj.controller, pidx, obj.counters));
                        break 'outer;
                    }
                }
            }
            let (controller, _src_player_idx, src_counters) = match src_info {
                Some(info) => info,
                None => return Ok(()), // source not found → fizzle
            };
            if src_counters <= 0 {
                return Ok(()); // nothing to move
            }
            let cidx = controller as usize;
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible target → fizzle
            };
            // Clear source's counters (true move). Source may be on
            // battlefield, graveyard, or anywhere — walk every zone
            // to flip it. Skip if source IS the target (same obj id),
            // which is a degenerate self-target case.
            if source_obj_id != target_id {
                for player in state.players.iter_mut() {
                    for zone in [
                        &mut player.hand,
                        &mut player.library,
                        &mut player.battlefield,
                        &mut player.graveyard,
                        &mut player.exile,
                    ] {
                        if zone.iter().any(|o| o.id == source_obj_id) {
                            let z = Arc::make_mut(zone);
                            for obj in z.iter_mut() {
                                if obj.id == source_obj_id {
                                    obj.counters = 0;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Add to target on controller's battlefield.
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.counters = obj.counters.saturating_add(src_counters);
                    break;
                }
            }
        }
        // Round-28 power-scaling damage spell (Mishra's Bauble shape).
        // Compute source's effective power inline, then route through
        // the same any-target selection as DealDamageAnyTarget.
        Effect::DealDamageEqualToSourcePowerAnyTarget => {
            // Look up source on either battlefield to get printed power
            // and UEOT bonus. If source isn't on battlefield, effective
            // power is 0 → no damage.
            let mut amount: i32 = 0;
            for player in &state.players {
                if let Some(obj) = player.battlefield.iter().find(|o| o.id == source_obj_id)
                {
                    if let Some(printing) =
                        state.registry.printings.get(obj.printing_id as usize)
                    {
                        let base = printing.power.unwrap_or(0) as i32;
                        let bonus = obj.ueot_power_bonus();
                        amount = (base + bonus).max(0);
                    }
                    break;
                }
            }
            if amount == 0 {
                return Ok(());
            }
            // **Controller resolution (I68 follow-up to the I59 /
            // I60 / I63 / I64 / I65 / I66 / I67 controller-of-effect
            // substrate).** Prefer the transient
            // `state.current_resolve_source_controller`, falling
            // back to `state.active_player`. Same pattern as
            // `Effect::DealDamageAnyTarget` (I67) — CR 119.5 /
            // 608.2: an effect resolves under its source's
            // controller. "Any target" defaulting to "the
            // defender" means "the SOURCE'S controller's opponent".
            //
            // **Why the fix.** Pre-fix this arm computed
            // `defender_idx = 1 - state.active_player`. Wrong when
            // the resolving spell / triggered ability has a
            // controller != active_player. Example: p1's
            // Mishra's-Bauble-shape spell ("this creature deals
            // damage equal to its power to any target") fires on
            // p0's turn. Pre-fix the trigger self-damaged p1's
            // first creature. Post-fix it correctly damages p0's
            // first creature (the source's controller's opponent).
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            // Reuse DealDamageAnyTarget's selection shape.
            let defender_idx = 1 - controller;
            let creature_target: Option<u32> = {
                let bf = &state.players[defender_idx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let amount_i16 = amount.min(i16::MAX as i32) as i16;
            if let Some(target_id) = creature_target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[defender_idx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            } else {
                state.players[defender_idx].life -= amount;
            }
        }
        // Round-28 broader-filter tap. Walks BOTH players' battlefield
        // in ETB-timestamp order via the shared
        // `crate::bf_iter::bf_iter_etb_sorted` helper, picking the
        // first permanent whose printing's `card_types` contains
        // "Artifact" OR "Creature". Idempotent on already-tapped
        // target. No eligible target → silent fizzle. Mirrors
        // Python's `effects/generic/tap.py:TapTargetArtifactOrCreature`
        // filter against the centralized `state.battlefield` walk
        // (no controller restriction). See invariant I39.
        Effect::TapTargetArtifactOrCreature => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing
                        .card_types
                        .iter()
                        .any(|t| t == "Artifact" || t == "Creature")
                    {
                        Some((obj.controller as usize, obj.id))
                    } else {
                        None
                    }
                });
            let Some((_pidx, target_id)) = target else {
                return Ok(()); // fizzle
            };
            // Round-58 BECOMES_TAPPED event bridge — tapper-effect
            // site (Python `effects/generic/tap.py:
            // TapTargetArtifactOrCreature`, routes the tap through
            // `turn.tap_permanent`). Centralized chokepoint so the
            // no-op-if-already-tapped gate matches Python (an
            // already-tapped target: silent no-op, NO event).
            crate::state_ops_resolve::tap_permanent_and_dispatch(
                state, target_id,
            )?;
        }
        // Round-28 intervening-if conditional. Count source-
        // controller's tapped creatures; if >= min_count, recursively
        // apply nested effect. Otherwise silent no-op.
        Effect::ConditionalOnTappedCreatures { effect, min_count } => {
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source not found → silent fizzle
            };
            let cidx = controller as usize;
            // Count tapped creatures on controller's battlefield.
            let tapped_count: i32 = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter()
                    .filter(|obj| {
                        if !obj.tapped {
                            return false;
                        }
                        let Some(printing) = registry.printings.get(obj.printing_id as usize)
                        else {
                            return false;
                        };
                        printing.card_types.iter().any(|t| t == "Creature")
                    })
                    .count() as i32
            };
            if tapped_count >= *min_count {
                apply_effect(state, effect.as_ref(), source_obj_id)?;
            }
            // else: silent no-op
        }
        // Round-28 multi-target counter add (Weftblade-Enhancer-shape).
        // Pick first min(max_n, eligible) creatures across both
        // battlefields in player-order (source controller first, then
        // opponent), add a +1/+1 counter to each. max_n=0 short-
        // circuits.
        Effect::PutPlusOneCounterOnEachOfUpToNTargetCreatures { max_n } => {
            if *max_n <= 0 {
                return Ok(());
            }
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let cidx = match src_controller {
                Some(c) => c as usize,
                None => state.active_player as usize,
            };
            let oidx = 1 - cidx;
            let limit = *max_n as usize;
            // Collect first min(limit, total) creature ids: controller
            // first, then opponent.
            let mut picked_ids: Vec<(usize, u32)> = Vec::with_capacity(limit);
            for &pidx in &[cidx, oidx] {
                if picked_ids.len() >= limit {
                    break;
                }
                let bf = &state.players[pidx].battlefield;
                let registry = &state.registry;
                for obj in bf.iter() {
                    if picked_ids.len() >= limit {
                        break;
                    }
                    let Some(printing) = registry.printings.get(obj.printing_id as usize)
                    else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        picked_ids.push((pidx, obj.id));
                    }
                }
            }
            if picked_ids.is_empty() {
                return Ok(());
            }
            // Apply +1 counter to each. Use Arc::make_mut once per
            // affected battlefield.
            // Group by player.
            for &pidx in &[cidx, oidx] {
                let any = picked_ids.iter().any(|(p, _)| *p == pidx);
                if !any {
                    continue;
                }
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if picked_ids.iter().any(|(p, id)| *p == pidx && *id == obj.id) {
                        obj.counters = obj.counters.saturating_add(1);
                    }
                }
            }
        }
        // Round-28 charge-counters-on-self (Blast-Zone-shape).
        // Single-counter-slot Rust: charge counters route to the same
        // scalar as +1/+1 (G8 multi-kind counter map is future work).
        // amount_from_x=true defers to a future X-value tracking
        // slice and currently resolves to 0 (silent no-op).
        Effect::PutChargeCountersOnSelf { amount, amount_from_x } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutChargeCountersOnSelf amount must be non-negative, got {amount}"
                )));
            }
            let effective_amount = if *amount_from_x { 0 } else { *amount };
            if effective_amount == 0 {
                return Ok(());
            }
            // Locate source on a battlefield (Python's PutPlus also
            // restricts to battlefield, same shape).
            let mut found_player: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    found_player = Some(pidx);
                    break;
                }
            }
            let Some(pidx) = found_player else {
                return Ok(()); // source not on battlefield → silent no-op
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == source_obj_id {
                    obj.counters = obj.counters.saturating_add(effective_amount);
                    break;
                }
            }
        }
        // Round-28 token-copy. Pick first creature on source-
        // controller's bf (deterministic first-slice target); allocate
        // a new obj_id from the monotonic `next_object_id` counter
        // (R76-1 — CR 111.7, byte-equal with Python's
        // `self.next_object_id += 1`); push a new GameObject with the
        // same printing_id onto controller's bf. Token ETBs
        // summoning-sick.
        Effect::CreateTokenCopyOfTargetCreatureYouControl => {
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let cidx = match src_controller {
                Some(c) => c as usize,
                None => state.active_player as usize,
            };
            // Pick first creature on controller's bf.
            let target_printing_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.printing_id)
                    } else {
                        None
                    }
                })
            };
            let Some(printing_id) = target_printing_id else {
                return Ok(()); // no eligible target
            };
            // **Aura-attachment Phase 4 S3 closure (2026-05-21) —
            // Moonlit Meditation replacement, copy-of-target variant.**
            // Python's `CreateTokenCopyOfTargetCreatureYouControl`
            // resolves by building a printing-with-Token-supertype
            // from the target and calling
            // `CreateToken(token_printing=..., count=1).resolve(...)`,
            // which then consults `_find_moonlit_replacement_source`
            // and substitutes the host's printing if Moonlit is in
            // play (`effects/generic/tokens.py:350-389` → 216-247).
            // The Rust arm mirrors that by applying the same
            // substitution here BEFORE the GameObject is constructed —
            // a Moonlit-eligible controller mints a copy of the
            // attached Aura's host instead of the originally chosen
            // creature target, with the per-object Token-supertype
            // override stamped.
            let (printing_id, moonlit_override) =
                moonlit_token_substitution(state, cidx as u8, printing_id);
            // R76-1: draw a fresh id from the monotonic
            // `next_object_id` counter (CR 111.7), REPLACING the old
            // non-monotonic `max_id + 1` re-scan that recycled a
            // ceased token's id. Byte-equal with Python's
            // `self.next_object_id += 1`.
            let new_id = crate::state::reserve_object_id(state);
            // CE4 closure: stamp the fresh-token ETB timestamp.
            let etb_ts = state.next_etb_timestamp;
            state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(1);
            let new_obj = GameObject {
                id: new_id,
                printing_id,
                controller: cidx as u8,
                zone: ZONE_BATTLEFIELD,
                tapped: false,
                summoning_sick: true,
                damage: 0,
                counters: 0,
                charge_counters: 0,
                loyalty_counters: 0,
                attached_to: None,
                exiled_with_source: None,
                // Round-30 stun-counter substrate: cloned tokens
                // enter with no stun counters (per the bare-token
                // ETB precedent above).
                stun_counters: 0,
                etb_timestamp: etb_ts,
                ueot_modifiers: Vec::new(),
                // warp_cost substrate (Phase 1, round-63): cloned
                // tokens are never warp-cast. Back-compat default.
                warp_exile_pending: false,
                warped: false,
                cast_from_exile_with_lightstall_surcharge: false,
                controller_baseline: None,
                // Aura-attachment Phase 4 S3 closure (2026-05-21):
                // copy-of-target variant Moonlit substitution. The
                // copy mechanism's own "Token supertype injection"
                // (via Python's `_dc.replace(base, supertypes=...)`
                // for non-Moonlit copies) lands at the same per-
                // object override site here. Pre-substrate Rust
                // didn't carry the Token supertype on copied tokens
                // because the registered printing didn't (the copy
                // was implicit via `printing_id` reuse) — this
                // closure makes the bridge round-trip honest for the
                // Moonlit-eligible case. The non-Moonlit copy case
                // stays at `false` here for back-compat; a future
                // slice can widen this to also stamp the override
                // when a copy is made unconditionally if that
                // semantic gap surfaces.
                token_supertype_override: moonlit_override,
                // Control-change-UEOT substrate — Cycle 1
                // (2026-05-21): cloned tokens never start with a
                // UEOT controller override.
                controller_override: None,
            };
            Arc::make_mut(&mut state.players[cidx].battlefield).push(new_obj);
            // ETB-trigger Slice 2 (RS5): queue ENTERS dispatch for
            // the cloned token. Same emission contract as RS4 —
            // drained by the surrounding resolver after the
            // resolving spell's stack item pops.
            state.pending_etb_dispatches.push(new_id);
            // **Aura-attachment substrate — Phase 4 S3 (Moonlit
            // Meditation substrate landing, 2026-05-20).** The
            // copy-token primitive IS a token-creation event for
            // the controller — flip the per-player created-tokens-
            // this-turn flag, same shape as the sister `CreateToken`
            // arm (Python's `tokens.py:294` records ANY non-zero
            // batch, including the copy-token branch driven via
            // `CreateToken`).
            if cidx < state.created_tokens_this_turn.len() {
                state.created_tokens_this_turn[cidx] = true;
            }
        }
        // Round-28 "may sac for inner effect". First-slice
        // deterministic-yes: if controller controls an artifact,
        // sacrifice the first one + recursively apply inner. No
        // artifact = full no-op (no inner either).
        Effect::OptionalSacArtifactForEffect { inner_effect } => {
            // Locate source controller.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let cidx = match src_controller {
                Some(c) => c as usize,
                None => state.active_player as usize,
            };
            // Pick first artifact on controller's bf.
            let artifact_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Artifact") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(artifact_id) = artifact_id else {
                return Ok(()); // no artifact → no-op
            };
            // Sacrifice: remove from bf, push to graveyard with CR 400.7 reset.
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == artifact_id)
                .expect("artifact_id was just found in bf but not on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_GRAVEYARD;
            obj.tapped = false;
            obj.summoning_sick = true; // r28 CE1 universal default
            obj.damage = 0;
            // R77-CFI1: counters are NOT cleared on the dead object.
            // A sacrifice routes through Python's
            // `turn.move_off_battlefield` (turn.py:538 docstring:
            // "sacrifice ... calls move_off_battlefield directly"),
            // which leaves `+1/+1` counters intact — the SAME
            // byte-equal contract the SBA destroy path / the closed
            // `DepressurizeEffect` arm follow. Prior stray
            // `obj.counters = 0;` removed (R77-CFI1; audited sibling
            // of the destroy arms, same class as the closed CFI-1).
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[cidx].graveyard).push(obj);
            // Round-29 Void LTB: optional artifact sac. Artifact target
            // may also be a land — helper gates on printing.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            // Apply inner effect.
            apply_effect(state, inner_effect.as_ref(), source_obj_id)?;
        }
        // Round-28 "owner puts target artifact-or-creature on top or
        // bottom of their library." Post-13th-slice deterministic-
        // TOP: Python prompts the owner's agent for top vs bottom;
        // Rust has no decision callback substrate yet so we always
        // pick TOP (matching Python's no-agent default at
        // `effects/generic/zone_moves.py:92-93` which sets
        // `choice = 0` and head-inserts into the library). The
        // post-CE3 convention is library[0] = TOP (next drawn) and
        // library[-1] = BOTTOM.
        //
        // Target selection mirrors TapTargetArtifactOrCreature: first
        // defender (= 1 - active_player) permanent whose
        // card_types contains "Artifact" OR "Creature". No eligible
        // target = silent fizzle.
        //
        // Owner = target's controller (Rust doesn't track an
        // owner-vs-controller distinction; first-slice synthetic
        // states never split the two, so controller == owner).
        //
        // CR 400.7 zone-change reset: tapped, damage, counters,
        // attached_to all reset. summoning_sick is PRESERVED on the
        // bf->library transition (13th Rust slice / RRR Pin 2
        // closure) — Python's resolve in `effects/generic/zone_moves
        // .py` likewise never touches summoning_sick, matching the
        // post-CE13 LTB preserve-semantics standardised by slice 12
        // on the sibling exile arm.
        Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary => {
            let _ = source_obj_id;
            // **Controller resolution (I78 follow-up to the
            // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
            // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77
            // controller-of-effect substrate).** CR 119.5 / 608.2:
            // an effect resolves under its source's controller.
            // OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOf-
            // Library's first-slice target picker chooses the
            // SOURCE'S controller's opponent's first
            // Artifact-or-Creature, where defender = 1 - controller.
            // Sibling closure to I70 / I75 / I76 — identical
            // defender-swap shape; differs only in destination
            // zone (owner's library bottom vs exile / graveyard /
            // defender's hand).
            //
            // **Why the fix.** Pre-fix this arm computed
            // `defender = 1 - state.active_player`. Wrong when the
            // resolving spell / triggered ability has a controller
            // != active_player. Example: on p0's turn, p1 casts a
            // Lost-in-Space-shape sorcery. Pre-fix it self-targeted
            // p1's own Artifact-or-Creature
            // (defender = 1 - active_player = 1 - 0 = 1). Post-fix
            // it correctly targets p0's Artifact-or-Creature (the
            // source's controller's opponent — standard
            // Lost-in-Space removal shape).
            //
            // **Owner-as-controller convention preserved.** Owner
            // is captured from the target object's `controller`
            // field on re-scan after the target is identified;
            // first-slice synthetic states never split owner vs
            // controller, so owner_idx == defender == 1 - controller.
            // The library destination remains the OWNER's library
            // (CR 401.1 "into its owner's library"); the controller
            // fix only changes which side the target is picked from.
            //
            // **LTB fan-out preserved** — the leaving id still
            // pushes onto `pending_leaves_dispatches` so the drain
            // runs in `resolve_top_of_stack` AFTER the resolving
            // slot pops, mirroring the pending_etb_dispatches
            // deferral shape.
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender = 1 - controller;
            let target_id: Option<u32> = {
                let bf = &state.players[defender].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    if printing
                        .card_types
                        .iter()
                        .any(|t| t == "Artifact" || t == "Creature")
                    {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // fizzle
            };
            // Capture owner (= controller) before mutating zones.
            let owner_idx: usize = {
                let bf = &state.players[defender].battlefield;
                let obj = bf
                    .iter()
                    .find(|o| o.id == target_id)
                    .expect("target_id was just found in bf but missing on re-scan");
                obj.controller as usize
            };
            // Remove from battlefield, reset per CR 400.7, insert at
            // owner's lib[0] (= TOP of library; matches Python's
            // no-agent default `choice=0` -> `library.insert(0, obj)`).
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[defender].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in bf but missing on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_LIBRARY;
            obj.tapped = false;
            // RRR Pin 2 closure (13th Rust slice): the explicit
            // unconditional summoning-sick stamp was dropped here to
            // match Python's resolve in `effects/generic/zone_moves.py:
            // 96-128`, which strips combat / counter state but never
            // touches the summoning-sick field. The pre-bounce value
            // now persists naturally through the bf->library
            // transition, mirroring the CE13 LTB preserve-semantics
            // that slice 12 applied to the sibling
            // `ExileOpponentsNonlandPermanentUntilLeaves` arm. 1 EoE
            // card aligned: Lost in Space.
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[owner_idx].library);
            // RRR Pin 3 closure (13th Rust slice): the no-agent
            // top-vs-bottom-of-library default now matches Python's
            // `effects/generic/zone_moves.py:86-94` no-agent branch
            // (`choice = 0` selects the head-insert leg). Prior code
            // here used a tail-append (BOTTOM-default); the new
            // head-insert below converges on Python's TOP-default.
            // CE13's Rust-converges-on-Python precedent applies. Full
            // agent-driven choice plumbing is a deeper modal-effect
            // gap deferred from this slice — the no-agent default
            // alignment is the minimal honest fix on the byte-equal
            // axis (post-CE3: library[0] = TOP, library[-1] = BOTTOM).
            lib_mut.insert(0, obj); // TOP of library = head of Vec
            // Round-29 Void LTB: target is Artifact OR Creature —
            // either way the helper's nonland gate filters correctly.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-28 EoE Equipment-ETB attach. Locate the source on
        // its controller's battlefield (Equipment ETB resolution
        // happens after the Equipment has entered the bf — CR
        // 603.6a), pick the first OTHER creature on the same bf
        // (mirrors PumpTargetCreatureUntilEOT's deterministic
        // first-slice target), and stamp
        // `source.attached_to = Some(target_id)`. Silent fizzle when
        // the source isn't on a bf (e.g. dispatched through a wrong
        // resolution path) or no eligible target exists.
        Effect::AttachToTargetCreatureYouControl => {
            // Locate source on a battlefield + capture its
            // controller. The source must be on a battlefield for
            // attach to make sense — CR 303.4 / 301.5: attach
            // relationship is a battlefield-only property.
            let mut src_loc: Option<(usize, u32)> = None; // (player_idx, source_id)
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    src_loc = Some((pidx, source_obj_id));
                    break;
                }
            }
            let Some((cidx, src_id)) = src_loc else {
                return Ok(()); // source not on bf → silent fizzle
            };
            // Pick the first creature on source-controller's bf
            // that ISN'T the source itself (an Equipment can't
            // attach to itself). Creature-ness gates on
            // `card_types contains "Creature"` — the canonical
            // creature-detection pattern mirroring
            // PumpTargetCreatureUntilEOT / OptionalSacArtifactForEffect.
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    if obj.id == src_id {
                        return None; // can't attach to self
                    }
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == src_id {
                    obj.attached_to = Some(target_id);
                    break;
                }
            }
        }
        // Round-29 Equip activated-ability resolution (CR 702.6).
        // Re-stamps `source.attached_to` to the chosen target
        // creature. CR 702.6c: equip can re-target an Equipment
        // that's already attached — stamping over an existing
        // `attached_to` is the expected behavior.
        //
        // **CE20 follow-up Slice 4 (Equipment-equip) target-pick:**
        // Reads the activator's chosen target from
        // `state.current_resolve_payload_target_id` (swapped in by
        // the stack resolver from the StackItem's
        // `trigger_payload.target_id` for activated-ability call
        // sites). When the field is `Some(id)`, stamp that as the
        // target. Pre-Slice-4 callers (any direct
        // `apply_effect(EquipToTargetCreatureYouControl)` driven
        // through a path that doesn't seed the slot) fall back to
        // the first-eligible-creature walk for back-compat —
        // matches the prior single-slice behavior. The fall-back is
        // load-bearing for the ETB-shape attach probe, where the
        // ETB trigger fires before any activator picks a target.
        Effect::EquipToTargetCreatureYouControl => {
            // Locate source on a battlefield + capture its controller.
            let mut src_loc: Option<(usize, u32)> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    src_loc = Some((pidx, source_obj_id));
                    break;
                }
            }
            let Some((cidx, src_id)) = src_loc else {
                return Ok(()); // source not on bf → silent fizzle
            };
            // CE20 follow-up Slice 4: prefer the payload target_id
            // when the resolver has seeded it (targeted-activation
            // path). Validate target is still legal at resolve time
            // (CR 608.2b — fizzle if the target has become illegal
            // since cost-pay): target must be on the activator's bf
            // AND a creature AND not the source itself. Illegal
            // target at resolve → silent fizzle (CR 608.2b).
            let target_id: Option<u32> = match state
                .current_resolve_payload_target_id
            {
                Some(payload_target) => {
                    // Resolve-time legality check.
                    let bf = &state.players[cidx].battlefield;
                    let registry = &state.registry;
                    bf.iter().find_map(|obj| {
                        if obj.id != payload_target {
                            return None;
                        }
                        if obj.id == src_id {
                            return None; // can't attach to self
                        }
                        let printing =
                            registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                }
                None => {
                    // Pre-Slice-4 first-eligible walk. Kept for
                    // back-compat with any caller path that drives
                    // the variant without seeding the target slot
                    // (no current production caller, but the
                    // fall-back is load-bearing for safety + parity
                    // with the AttachToTargetCreatureYouControl
                    // sibling variant).
                    let bf = &state.players[cidx].battlefield;
                    let registry = &state.registry;
                    bf.iter().find_map(|obj| {
                        if obj.id == src_id {
                            return None; // can't attach to self
                        }
                        let printing =
                            registry.printings.get(obj.printing_id as usize)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
                }
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == src_id {
                    obj.attached_to = Some(target_id);
                    break;
                }
            }
        }
        // Round-28 enchanted-permanent destroy (Cryoshatter-shape).
        // Read source's `attached_to`; if Some(target_id) and target
        // exists on a battlefield, move it to its controller's
        // graveyard with CR 400.7 zone-change reset. Source not on
        // bf / attached_to is None / target untracked → silent fizzle.
        Effect::DestroyEnchantedPermanent => {
            // Locate source on a battlefield, capture its attached_to.
            let mut target_id_opt: Option<u32> = None;
            for player in &state.players {
                if let Some(src) = player.battlefield.iter().find(|o| o.id == source_obj_id) {
                    target_id_opt = src.attached_to;
                    break;
                }
            }
            let Some(target_id) = target_id_opt else {
                return Ok(()); // source not on bf or no host → fizzle
            };
            // Locate target on either battlefield + capture owner-as-controller.
            let mut target_loc: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == target_id) {
                    target_loc = Some(pidx);
                    break;
                }
            }
            let Some(tidx) = target_loc else {
                return Ok(()); // target untracked → fizzle
            };
            // Remove from battlefield, reset per CR 400.7, push to
            // owner's graveyard.
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[tidx].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in bf but missing on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_GRAVEYARD;
            obj.tapped = false;
            obj.summoning_sick = true; // r28 CE1 universal-true default
            obj.damage = 0;
            // R77-CFI1: counters are NOT cleared on the dead object —
            // the established byte-equal contract (Python
            // `turn.move_off_battlefield` turn.py:231-240 / SBA
            // destroy `state_ops_sba.rs` ~:417 / the closed
            // `DepressurizeEffect` arm all retain `+1/+1` counters on
            // the dead object). Prior stray `obj.counters = 0;`
            // removed (R77-CFI1; same class as the closed CFI-1).
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[tidx].graveyard).push(obj);
            // Round-29 Void LTB: destroying an enchanted host. Host
            // is any permanent (creature / artifact / enchantment /
            // planeswalker / land); helper gates on the printing.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-28 enchanted-permanent tap. Same `attached_to` lookup
        // as DestroyEnchantedPermanent but mutates the target's
        // `tapped` flag instead of moving zones. Idempotent on
        // already-tapped.
        Effect::TapEnchantedPermanent => {
            let mut target_id_opt: Option<u32> = None;
            for player in &state.players {
                if let Some(src) = player.battlefield.iter().find(|o| o.id == source_obj_id) {
                    target_id_opt = src.attached_to;
                    break;
                }
            }
            let Some(target_id) = target_id_opt else {
                return Ok(());
            };
            let mut target_loc: Option<usize> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == target_id) {
                    target_loc = Some(pidx);
                    break;
                }
            }
            if target_loc.is_none() {
                return Ok(());
            }
            // Round-58 BECOMES_TAPPED event bridge — enchanted-
            // permanent tapper-effect site. Python's
            // `effects/generic/tap.py:TapEnchantedPermanent.resolve`
            // (Stay Hidden Stay Silent's Aura-ETB tap) is a pure
            // tap with an explicit `if host.tapped: return` guard
            // followed by `turn.tap_permanent(state, host)` — exactly
            // the centralized chokepoint's no-op-if-already-tapped +
            // dispatch shape. Route through it so an already-tapped
            // host is a silent no-op with NO BECOMES_TAPPED event.
            crate::state_ops_resolve::tap_permanent_and_dispatch(
                state, target_id,
            )?;
        }
        // Round-30 stun-counter substrate. Walks both players'
        // battlefield in ETB-timestamp order via the shared
        // `bf_iter_etb_sorted` helper, picking the FIRST already-
        // tapped creature it finds (no controller restriction; the
        // Python class places none either — "up to one target tapped
        // creature"). No eligible target → silent fizzle. Increments
        // `stun_counters` on the matched object via Arc::make_mut.
        // Sister of TapTargetArtifactOrCreature (no-controller
        // filter); differs in (1) Creature-only filter, (2) tapped-
        // only gate, (3) stun_counters increment instead of tap-flip.
        Effect::PutStunCounterOnTappedCreature => {
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    if !obj.tapped {
                        return None;
                    }
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some((obj.controller as usize, obj.id))
                    } else {
                        None
                    }
                });
            let Some((pidx, target_id)) = target else {
                return Ok(()); // fizzle
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.stun_counters = obj.stun_counters.saturating_add(1);
                    break;
                }
            }
        }
        // Round-30 stun-counter substrate. Both effects (tap + stun
        // counter) apply to the SAME chosen target. Walks both
        // players' battlefield in ETB-timestamp order, picking the
        // FIRST (Artifact OR Creature) permanent NOT controlled by
        // the source's controller. No eligible target → silent fizzle.
        // Tap is idempotent on already-tapped target (CR 701.20a).
        Effect::TapTargetOpponentPermanentAndStunIt => {
            // Locate source controller — the "an opponent controls"
            // filter needs the source's controller to compare against.
            // Mirrors the controller-resolution shape of the Round-28
            // I79 follow-up: prefer current_resolve_source_controller,
            // fall back to source-on-bf scan, fall back to active_player.
            let src_controller: u8 = state
                .current_resolve_source_controller
                .or_else(|| {
                    state
                        .players
                        .iter()
                        .flat_map(|p| p.battlefield.iter())
                        .find(|o| o.id == source_obj_id)
                        .map(|o| o.controller)
                })
                .unwrap_or(state.active_player);
            let target: Option<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .find_map(|obj| {
                    if obj.controller == src_controller {
                        return None; // opponent-only filter
                    }
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing
                        .card_types
                        .iter()
                        .any(|t| t == "Artifact" || t == "Creature")
                    {
                        Some((obj.controller as usize, obj.id))
                    } else {
                        None
                    }
                });
            let Some((pidx, target_id)) = target else {
                return Ok(()); // fizzle
            };
            // Round-58 BECOMES_TAPPED event bridge — tap+stun
            // combined site. Python's
            // `effects/generic/tap.py:TapTargetOpponentPermanentAndStunIt.resolve`
            // (tap.py:237-256) calls `turn.tap_permanent(state, obj)`
            // (fires BECOMES_TAPPED ONLY if the target was untapped —
            // no-op early-return otherwise) and THEN
            // `add_stun_counters(state, obj, 1)` UNCONDITIONALLY (the
            // stun counter is added even when the creature was
            // already tapped). So the no-op-if-already-tapped gate
            // applies to the EVENT only, not the stun rider — the
            // centralized chokepoint can't be reused wholesale here.
            // Snapshot `was_tapped` before the flip, apply tap+stun
            // exactly as before, then dispatch BECOMES_TAPPED iff the
            // creature actually transitioned untapped->tapped.
            let was_tapped = state.players[pidx]
                .battlefield
                .iter()
                .find(|o| o.id == target_id)
                .map(|o| o.tapped)
                .unwrap_or(true);
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].battlefield);
            for obj in bf_mut.iter_mut() {
                if obj.id == target_id {
                    obj.tapped = true;
                    obj.stun_counters = obj.stun_counters.saturating_add(1);
                    break;
                }
            }
            if !was_tapped {
                dispatch_event(
                    state, Event::BecomesTapped, target_id,
                )?;
            }
        }
        // Round-28 "until source leaves" exile. Pick defender's first
        // creature, stamp `exiled_with_source = Some(source_id)`,
        // move to exile. CR 400.7 zone-change reset for all other
        // per-object state. The link enables manual return via
        // `return_paired_exiles_for(source_id)`.
        Effect::ExileTargetCreatureUntilSourceLeaves => {
            // **Controller resolution (I79 follow-up to the
            // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
            // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77 / I78
            // controller-of-effect substrate).** CR 119.5 / 608.2:
            // an effect resolves under its source's controller.
            // ExileTargetCreatureUntilSourceLeaves's first-slice
            // target picker chooses the SOURCE'S controller's
            // opponent's first Creature, where defender =
            // 1 - controller. Sibling closure to I70 / I75 / I76 /
            // I78 — identical defender-swap shape; differs in
            // destination zone (exile-with-source-link vs exile /
            // graveyard / hand / library bottom) and in the
            // round-28 `exiled_with_source` bookkeeping that
            // enables manual return via
            // `return_paired_exiles_for(source_id)`.
            //
            // **Why the fix.** Pre-fix this arm computed
            // `defender = 1 - state.active_player`. Wrong when the
            // resolving spell / triggered ability has a controller
            // != active_player. Example: on p0's turn, p1's
            // All-Fates-Stalker-shape Creature has an ETB that
            // exiles a target creature until the source leaves.
            // Pre-fix it self-exiled p1's own first Creature
            // (defender = 1 - active_player = 1 - 0 = 1). Post-fix
            // it correctly exiles p0's first Creature (the
            // source's controller's opponent — the standard
            // Banisher-Priest / Fiend-Hunter ETB-exile shape).
            //
            // **Bookkeeping invariants preserved.** The
            // `exiled_with_source = Some(source_obj_id)` stamp
            // (which enables `return_paired_exiles_for`) is
            // unchanged — the controller fix only changes the
            // defender from which the target is picked, never the
            // pairing semantics. The owner-on-re-scan (target's
            // controller == owner in first-slice synthetic states)
            // continues to route the exiled object onto its
            // owner's exile zone. CR 400.7 zone-change reset
            // (tapped, summoning_sick=true, damage, counters,
            // attached_to) preserved. LTB fan-out via
            // `pending_leaves_dispatches` preserved.
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender = 1 - controller;
            let target_id: Option<u32> = {
                let bf = &state.players[defender].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // fizzle — no eligible creature
            };
            // Capture target's owner (= controller) before mutation.
            let owner_idx: usize = {
                let obj = state.players[defender]
                    .battlefield
                    .iter()
                    .find(|o| o.id == target_id)
                    .expect("target_id was just found in bf");
                obj.controller as usize
            };
            // Remove from defender's bf, stamp link, move to owner's
            // exile zone with CR 400.7 reset (preserving the link).
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[defender].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in bf but missing on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            obj.summoning_sick = true; // r28 CE1 universal-true default
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = Some(source_obj_id);
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[owner_idx].exile).push(obj);
            // Round-29 Void LTB: until-leaves exile. Target is a
            // Creature by filter → nonland → flag flips.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-28 broader-filter "until leaves" exile. Same shape
        // as ExileTargetCreatureUntilSourceLeaves but filters on
        // "not a Land" rather than "is a Creature" — so artifacts,
        // enchantments, planeswalkers, and creatures are all
        // eligible. Defender = 1 - active_player.
        Effect::ExileOpponentsNonlandPermanentUntilLeaves => {
            // **Controller resolution (I80 follow-up to the
            // I59 / I60 / I63 / I64 / I65 / I66 / I67 / I68 / I69 /
            // I70 / I71 / I72 / I73 / I74 / I75 / I76 / I77 / I78 /
            // I79 controller-of-effect substrate).** CR 119.5 /
            // 608.2: an effect resolves under its source's
            // controller. ExileOpponentsNonlandPermanentUntilLeaves's
            // first-slice target picker chooses the SOURCE'S
            // controller's opponent's first non-land permanent,
            // where defender = 1 - controller. Sibling closure to
            // I79 — same broader-shape "until source leaves" exile;
            // differs from I79 only in the target filter ("not a
            // Land" instead of "is a Creature").
            //
            // **Why the fix.** Pre-fix this arm computed
            // `defender = 1 - state.active_player`. Wrong when the
            // resolving spell / triggered ability has a controller
            // != active_player. Example: on p0's turn, p1's
            // Hardlight-Containment-shape Enchantment has an ETB
            // that exiles an opponent's non-land permanent until
            // the source leaves. Pre-fix it self-exiled p1's own
            // first non-land permanent (defender =
            // 1 - active_player = 1 - 0 = 1); post-fix it correctly
            // exiles p0's first non-land permanent (the source's
            // controller's opponent — the standard
            // Hardlight-Containment shape, where "opponent's" in
            // the card text actually means the source-controller's
            // opponent, not the active-player's opponent).
            //
            // **Bookkeeping invariants preserved (sibling to I79
            // gotcha-list #7).** The
            // `exiled_with_source = Some(source_obj_id)` stamp
            // (which enables `return_paired_exiles_for`) is
            // unchanged — the controller fix only changes the
            // defender from which the target is picked, never the
            // pairing semantics. The owner-on-re-scan (target's
            // controller == owner in first-slice synthetic states)
            // continues to route the exiled object onto its
            // owner's exile zone. CR 400.7 zone-change reset
            // preserved. LTB fan-out via
            // `pending_leaves_dispatches` preserved.
            //
            // **Closes the post-I77 forward inventory of
            // defender-shape `(1 - state.active_player)` sites in
            // state_ops_resolve.rs.** The remaining single-target
            // first-slice defender-swap inventory is empty after
            // I80. Remaining candidates (CreateToken,
            // ReturnCreatureFromYourGraveyard family, Site at
            // ~5626) need audit — they don't fit the
            // defender-swap shape and require per-arm
            // investigation. See `docs/RUST_PARITY_GAPS.md` I80
            // forward notes.
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender = 1 - controller;
            let target_id: Option<u32> = {
                let bf = &state.players[defender].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    let is_land = printing.card_types.iter().any(|t| t == "Land");
                    if !is_land {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // fizzle — no eligible permanent
            };
            let owner_idx: usize = {
                let obj = state.players[defender]
                    .battlefield
                    .iter()
                    .find(|o| o.id == target_id)
                    .expect("target_id was just found in bf");
                obj.controller as usize
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[defender].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in bf but missing on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            // **RRR Pin 1 closure (12th Rust slice).** Dropped the
            // explicit summoning_sick stamp here. The CE13 closure
            // (2026-05-12) standardised Rust LTB arms to PRESERVE
            // summoning_sick across bf-exit (matching Python's
            // ``move_off_battlefield`` which never touches the field
            // — ``turn.py:141-315``). This Effect arm was missed by
            // CE13; the bare ``remove`` -> ``push`` sequence
            // preserves the pre-exile value naturally. 3 EoE cards
            // parity-aligned: Banishing Light, Hardlight Containment,
            // Seam Rip (the last via the ``WithMaxMvUntilLeaves``
            // sibling arm below — sibling closure same shape).
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = Some(source_obj_id);
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[owner_idx].exile).push(obj);
            // Round-29 Void LTB: exile-opps-nonland — target is nonland
            // by filter. Helper short-circuits on lands defensively.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-14 rider-widened "until leaves" exile (Option B —
        // additive sibling variant). Same shape as the bare
        // `ExileOpponentsNonlandPermanentUntilLeaves` arm above;
        // adds the `max_mv` mana-value cap to the legal-target
        // pool. EoE consumer: Seam Rip (`{W}` Enchantment, ETB
        // exile target nonland permanent an opponent controls
        // with mana value 2 or less until this enchantment leaves
        // the battlefield).
        //
        // **Controller-of-effect.** Mirrors the bare sibling's
        // I80 fix: controller derives from
        // `current_resolve_source_controller` (with `active_player`
        // fallback); defender = 1 - controller.
        //
        // **MV-cap evaluation.** Reads off the printing's static
        // mana cost (CR 117.x): the sum of generic + W + U + B + R
        // + G. Uses the same per-color-channel sum pattern as
        // `Effect::SearchLibraryForCard` "creature_small" arm at
        // ~line 4658. X-cost cards have X stored as 0 in the
        // printing's `mana_cost.generic` field (per the registry's
        // parse_mana_cost contract), matching CR 202.3b's "X is
        // treated as 0 in zones other than the stack".
        //
        // **Bookkeeping invariants.** Same as the bare sibling:
        // stamps `exiled_with_source = Some(source_obj_id)`,
        // pushes onto owner's exile zone, queues
        // `pending_leaves_dispatches` for LTB fan-out. CR 400.7
        // zone-change reset (tapped, summoning_sick=true per r28
        // CE1, damage, counters, attached_to) preserved.
        Effect::ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves { max_mv } => {
            let max_mv_local = *max_mv;
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let defender = 1 - controller;
            let target_id: Option<u32> = {
                let bf = &state.players[defender].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.lookup(obj.printing_id)?;
                    let is_land = printing.card_types.iter().any(|t| t == "Land");
                    if is_land {
                        return None;
                    }
                    let mc = &printing.mana_cost;
                    let total = mc.generic as i32
                        + mc.white as i32
                        + mc.blue as i32
                        + mc.black as i32
                        + mc.red as i32
                        + mc.green as i32;
                    if total > max_mv_local {
                        return None;
                    }
                    Some(obj.id)
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // fizzle — no eligible permanent
            };
            let owner_idx: usize = {
                let obj = state.players[defender]
                    .battlefield
                    .iter()
                    .find(|o| o.id == target_id)
                    .expect("target_id was just found in bf");
                obj.controller as usize
            };
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[defender].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id was just found in bf but missing on re-scan");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            // **RRR Pin 1 sibling closure (12th Rust slice).** Same
            // drop as the bare ``ExileOpponentsNonlandPermanentUntilLeaves``
            // arm above — preserve pre-exile ``summoning_sick`` via
            // the bare ``remove`` -> ``push`` sequence. Seam Rip uses
            // this ``max_mv``-rider variant; the closure aligns it
            // with Banishing Light's bare-shape sibling.
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = Some(source_obj_id);
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[owner_idx].exile).push(obj);
            mark_left_battlefield_if_nonland(state, printing_id);
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // Round-28 may-do decision wrapper. "You may [inner_effect]."
        // Pushes a PendingDecision::MayDoChoice onto the queue and
        // short-circuits. The controller drains via
        // `submit_may_do_decision(yes)` (manual) or
        // `auto_resolve_pending_decisions("default" | "may_do_yes" |
        // "may_do_no")` (deterministic). On "yes" the inner effect is
        // applied with the same `source_obj_id` so identity-coupled
        // riders ("if you do, draw a card") retain the resolving
        // spell's identity. The `_prompt` field rides along on the
        // PendingDecision dict snapshot for agent introspection but
        // isn't used here.
        //
        // Controller convention (I61 follow-up): prefer the transient
        // `state.current_resolve_source_controller` (set by the stack
        // resolver around each `apply_effect` call to the
        // StackItem.controller), falling back to `state.active_player`
        // for back-compat with direct `apply_effect_pub` callers. This
        // mirrors the I59 / I60 reader-side fix and closes the latent
        // cross-controller MayDo divergence: a p1-controlled spell on
        // p0's turn must push a MayDoChoice with controller=1, not 0.
        Effect::MayDo { inner_effect, prompt: _prompt } => {
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            state.pending_decisions.push(PendingDecision::MayDoChoice {
                source_obj_id,
                controller,
                inner_effect: inner_effect.clone(),
            });
        }
        // Round-28 "look at top N, put one in hand, rest to graveyard"
        // — Commune-with-Evil family. Same pop-and-package shape as
        // Surveil but routes to hand+graveyard instead of library-top
        // +graveyard, and asks the controller to pick ONE rather than
        // partitioning the whole set.
        //
        // Controller convention: `state.active_player` — same first-
        // slice shape as Surveil / DrawCards.
        //
        // Empty / clamp behavior: `n=0` or library_len<n emits a
        // PendingDecision with a (possibly empty) cards Vec. The
        // submit dispatcher and auto-resolve handle an empty Vec as
        // an O(1) no-op. Mirrors Python's `if not examined: return`
        // short-circuit while preserving the substrate invariant
        // (one PendingDecision per Effect arm).
        Effect::LookTopNPutOneInHandRestToGraveyard { n } => {
            if *n < 0 {
                return Err(PyValueError::new_err(format!(
                    "LookTopNPutOneInHandRestToGraveyard: n must be non-negative, got {n}"
                )));
            }
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let pidx = controller as usize;
            let lib_len = state.players[pidx].library.len();
            let count = (*n as usize).min(lib_len);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(count);
            for _ in 0..count {
                // Post-CE3: top = lib[0]. Always remove(0).
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(
                PendingDecision::PickCardFromExaminedChoice {
                    source_obj_id,
                    controller,
                    cards,
                },
            );
        }
        // **BLOCKED-RUST-EFFECT slice (round 54) — Famished Worldsire
        // ETB.** "Look at the top X cards of your library, where X is
        // this creature's power. Put any number of land cards from
        // among them onto the battlefield tapped, then shuffle."
        // Mirrors Python's `FamishedWorldsireETBEffect.resolve`
        // (`effects/bespoke.py:106`) STEP-FOR-STEP so the post-resolve
        // state is byte-equal:
        //   source = state.resolving_source            (Rust: source_obj_id)
        //   if source is None: return                  (read returns 0 → x<=0 no-op)
        //   x = effective_power(state, source)         (read_famished_source_power)
        //   if x <= 0: return
        //   examined = p.library[:x]
        //   lands = [c for c in examined if c.printing.is_land]
        //   for land in lands: remove → bf tapped; fire_landfall
        //   shuffle_library(state, controller_id)
        //
        // **No PendingDecision.** The oracle's "put any number" is a
        // deterministic greedy-ALL on the Python source-of-truth side
        // (`bespoke.py:121`, documented "any number" → greedy-land
        // simplification, strictly ramp-positive). Both engines mirror
        // it; no decision is plumbed, so there is no §2
        // silent-mistranslation surface from the choice.
        Effect::FamishedWorldsireLookTopXByPowerPutLandsTapped => {
            // Source-controller resolution — same shape as
            // `SearchLibraryForBasicLandTapped` / `LookTopN...`
            // (I69 / I61): "your library" = the SOURCE's controller's
            // library, NOT the active player's. CR 119.5 / 608.2.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller as usize;
            let controller_u8 = controller;
            // X = source's CURRENT effective power (post-Devour +1/+1
            // counters). Source off-battlefield → read returns 0.
            // x <= 0 → silent no-op (mirrors Python's chained
            // `if source is None: return` / `if x <= 0: return`).
            let x = read_famished_source_power(state, source_obj_id);
            if x <= 0 {
                return Ok(());
            }
            let lib_len = state.players[controller_idx].library.len();
            // `p.library[:x]` — examine the top min(x, lib_len) cards.
            // Post-CE3 the top of library is lib[0]; the examined
            // window is the first `count` entries.
            let count = (x as usize).min(lib_len);
            // Collect the Vec INDICES of land cards in the examined
            // window, top-to-bottom (Python builds `lands` via a
            // top-to-bottom list comprehension over `examined`, then
            // `.remove(land)` each in that order — the survivors are
            // unaffected since removal is by identity). We snapshot
            // the indices first, then remove from the BACK so earlier
            // indices stay valid (equivalent to Python's
            // identity-`.remove()` loop; non-land "rest" cards stay in
            // the library exactly where they were — `bespoke.py:122`).
            let land_indices: Vec<usize> = {
                let lib = &state.players[controller_idx].library;
                let mut idxs: Vec<usize> = Vec::new();
                for (i, obj) in lib.iter().enumerate().take(count) {
                    let printing = match state.registry.lookup(obj.printing_id) {
                        Some(p) => p,
                        None => continue,
                    };
                    // Python's `printing.is_land` = `"Land" in
                    // card_types` (`cards/base.py:1021`). Famished
                    // Worldsire accepts ANY land (no Basic supertype
                    // gate — unlike `SearchLibraryForBasicLandTapped`).
                    if printing.card_types.iter().any(|t| t == "Land") {
                        idxs.push(i);
                    }
                }
                idxs
            };
            // ETB-trigger fan-out: collect entered land obj_ids so we
            // can queue `Event::Enters` (landfall) AFTER the post-move
            // shuffle drops its borrows — same shape as
            // `SearchLibraryForBasicLandTapped`. Python calls
            // `fire_landfall_triggers(state, land)` per land in
            // top-to-bottom order; we preserve that order by draining
            // `land_indices` low→high and pushing in that order.
            let mut entered_ids: Vec<u32> =
                Vec::with_capacity(land_indices.len());
            // Remove from the BACK so earlier indices stay valid.
            for &i in land_indices.iter().rev() {
                // CE4 closure: reserve before the player-mut borrow.
                let etb_ts = crate::state::reserve_etb_timestamp(state);
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                let mut obj = lib.remove(i);
                // Stamp battlefield entry: tapped, not summoning-sick
                // (CR 302.1 is creature-only; a land entering via this
                // effect has no summoning sickness regardless).
                // Controller = the source's controller. Same per-object
                // reset shape as `SearchLibraryForBasicLandTapped`.
                obj.zone = ZONE_BATTLEFIELD;
                obj.tapped = true;
                obj.summoning_sick = false;
                obj.controller = controller_u8;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                obj.etb_timestamp = etb_ts;
                entered_ids.push(obj.id);
                Arc::make_mut(&mut player.battlefield).push(obj);
            }
            // Restore top-to-bottom landfall order (we removed
            // back-to-front above, so `entered_ids` is currently
            // bottom-to-top; Python fires landfall top-to-bottom).
            entered_ids.reverse();
            // Shuffle the WHOLE remaining library — CR 701.19 / 701.22,
            // mirrors Python's `shuffle_library(state, controller_id)`
            // = `state.rng.shuffle(p.library)`. Identical dual-path
            // block to `SearchLibraryForBasicLandTapped`: external
            // Python `random.Random` via `external_rng::next_below`
            // (byte-identical to `random.Random.shuffle`) when
            // installed, xorshift64 fallback otherwise.
            let lib_len_post = state.players[controller_idx].library.len();
            if lib_len_post >= 2 {
                let py_rng_owned: Option<Py<PyAny>> = state
                    .rng_ref()
                    .map(|r| Python::with_gil(|py| r.clone_ref(py)));
                if let Some(py_rng) = py_rng_owned {
                    let player = &mut state.players[controller_idx];
                    let lib = Arc::make_mut(&mut player.library);
                    let n = lib.len();
                    Python::with_gil(|py| -> PyResult<()> {
                        let bound = py_rng.bind(py);
                        for i in (1..n).rev() {
                            let j = crate::external_rng::next_below(bound, i + 1)?;
                            lib.swap(i, j);
                        }
                        Ok(())
                    })?;
                } else {
                    // xorshift64 fallback: distinct seed per (turn,
                    // ap, source, lib_len) tuple — same control flow as
                    // every other xorshift site in the engine.
                    let mut seed = (state.turn_number as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ ((state.active_player as u64) << 56)
                        ^ ((source_obj_id as u64) << 16)
                        ^ (lib_len_post as u64);
                    if seed == 0 {
                        seed = 0xdead_beef_dead_beef;
                    }
                    let player = &mut state.players[controller_idx];
                    let lib = Arc::make_mut(&mut player.library);
                    let n = lib.len();
                    let mut x_rng = seed;
                    for i in (1..n).rev() {
                        x_rng ^= x_rng << 13;
                        x_rng ^= x_rng >> 7;
                        x_rng ^= x_rng << 17;
                        let n_u64 = (i + 1) as u64;
                        let bound = u64::MAX - (u64::MAX % n_u64);
                        let mut r = x_rng;
                        while r >= bound {
                            r ^= r << 13;
                            r ^= r >> 7;
                            r ^= r << 17;
                            x_rng = r;
                        }
                        let j = (r % n_u64) as usize;
                        lib.swap(i, j);
                    }
                }
            }
            // Queue ENTERS (landfall) dispatches per entered land, in
            // top-to-bottom order — the caller (`resolve_top_of_stack`
            // / `_pause`) drains after popping the resolving stack
            // item. Mirrors Python's per-land
            // `fire_landfall_triggers`.
            state.pending_etb_dispatches.extend(entered_ids);
        }
        // **16th Rust slice (TTT Pin 3 closure) — Astelli Reclaimer
        // ETB.** "Return target noncreature, nonland permanent card
        // with mana value X or less from your graveyard to the
        // battlefield, where X is the amount of mana spent to cast
        // this creature." Mirrors Python's
        // `AstelliReclaimerETBEffect.resolve`
        // (`effects/bespoke.py:62-82`) STEP-FOR-STEP so the
        // post-resolve state is byte-equal:
        //   source = state.resolving_source             (Rust: source_obj_id on bf)
        //   if source is None: return                   (off-bf → silent no-op)
        //   x = warp_cost.total if cast_via_warp else mana_cost.total
        //   pick the first GY card with is_permanent && !is_creature
        //        && !is_land && mana_cost.total <= x
        //   if no card matches: return                  (Python parity)
        //   remove from GY; put_into_battlefield(card, controller)
        //   queue Event::Enters dispatch (post-pop, via the
        //   pending_etb_dispatches queue)
        //
        // **No PendingDecision.** Python's `requires_target()` is
        // True, but the target binding is a `legal_targets`-driven
        // deterministic first-legal pick at resolve time on both
        // sides (no choice plumbed — the bespoke effect is
        // graveyard-order deterministic).
        Effect::AstelliReclaimerETBEffect => {
            // Source-controller resolution — same shape as
            // `FamishedWorldsireLookTopXByPowerPutLandsTapped`
            // (round 54): "your graveyard" = the SOURCE's
            // controller's graveyard, CR 119.5 / 608.2.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller as usize;
            let controller_u8 = controller;
            // Locate the source on the controller's battlefield (the
            // ETB trigger fires AFTER move_source_post_resolve has
            // stamped the entered permanent onto bf). If absent
            // (counterspelled pre-resolve / blinked away mid-stack /
            // any other off-bf path), silent no-op — mirrors Python's
            // `if source is None: return` chained guards.
            let (cast_via_warp, printing_id) = {
                let bf = &state.players[controller_idx].battlefield;
                match bf.iter().find(|o| o.id == source_obj_id) {
                    Some(obj) => (obj.warp_exile_pending, obj.printing_id),
                    None => return Ok(()),
                }
            };
            // X = mana spent to cast self. `warp_exile_pending` is
            // Rust's `cast_via_warp` flag (set at `cast_creature_warp`
            // time on the stack-resident source, rides through
            // `move_source_post_resolve` onto the entered bf object —
            // confirmed by `state_ops_cast.rs:1242-1265` doc comment).
            // If the warp-cast flag is set AND the printing carries a
            // `warp_cost` payload, X = warp_cost total. Otherwise
            // X = printing.mana_cost total. Mirrors Python's
            // `_x_from_source` (`bespoke.py:40-46`):
            //   if source.cast_via_warp and warp_cost is not None:
            //       return warp_cost.total
            //   return mana_cost.total
            let printing = match state.registry.lookup(printing_id) {
                Some(p) => p,
                None => return Ok(()),
            };
            let x: u32 = if cast_via_warp {
                match printing.warp_cost.as_ref() {
                    Some(wc) => {
                        wc.generic as u32
                            + wc.white as u32
                            + wc.blue as u32
                            + wc.black as u32
                            + wc.red as u32
                            + wc.green as u32
                    }
                    None => {
                        let mc = &printing.mana_cost;
                        mc.generic as u32
                            + mc.white as u32
                            + mc.blue as u32
                            + mc.black as u32
                            + mc.red as u32
                            + mc.green as u32
                    }
                }
            } else {
                let mc = &printing.mana_cost;
                mc.generic as u32
                    + mc.white as u32
                    + mc.blue as u32
                    + mc.black as u32
                    + mc.red as u32
                    + mc.green as u32
            };
            // First-legal graveyard scan — pick the FIRST card in
            // insertion order whose printing is a noncreature,
            // nonland permanent with mana_cost.total <= X. Mirrors
            // Python's `legal_targets` shape (a list comprehension
            // over `p.graveyard` in insertion order) plus the
            // first-legal pick the requires_target target binding
            // makes deterministic. `is_permanent` =
            // `card_types intersects {Creature, Artifact, Land,
            // Enchantment, Planeswalker}`; the `!is_creature` and
            // `!is_land` filters reject Creature and Land, so the
            // survivors are Artifact / Enchantment / Planeswalker
            // permanent cards (CR 110.4 + Python's
            // `cards/base.py:1068-1076` `is_permanent` property).
            let target_idx: Option<usize> = {
                let gy = &state.players[controller_idx].graveyard;
                let mut found: Option<usize> = None;
                for (i, obj) in gy.iter().enumerate() {
                    let pr = match state.registry.lookup(obj.printing_id) {
                        Some(p) => p,
                        None => continue,
                    };
                    let is_creature =
                        pr.card_types.iter().any(|t| t == "Creature");
                    let is_land =
                        pr.card_types.iter().any(|t| t == "Land");
                    let is_artifact =
                        pr.card_types.iter().any(|t| t == "Artifact");
                    let is_enchantment =
                        pr.card_types.iter().any(|t| t == "Enchantment");
                    let is_planeswalker =
                        pr.card_types.iter().any(|t| t == "Planeswalker");
                    let is_permanent = is_creature
                        || is_artifact
                        || is_land
                        || is_enchantment
                        || is_planeswalker;
                    if !is_permanent || is_creature || is_land {
                        continue;
                    }
                    let mc = &pr.mana_cost;
                    let mv: u32 = mc.generic as u32
                        + mc.white as u32
                        + mc.blue as u32
                        + mc.black as u32
                        + mc.red as u32
                        + mc.green as u32;
                    if mv <= x {
                        found = Some(i);
                        break;
                    }
                }
                found
            };
            let pick_idx = match target_idx {
                Some(i) => i,
                None => return Ok(()),
            };
            // Reserve a fresh ETB timestamp BEFORE the player-mut
            // borrow (same pattern as Famished Worldsire +
            // SearchLibraryForBasicLandTapped).
            let etb_ts = crate::state::reserve_etb_timestamp(state);
            // Remove from graveyard; stamp battlefield-entry fields
            // and push to bf. The picked card is a noncreature
            // permanent — per CR 302.1 + 612.6 (slice 14 TTT Pin 2
            // closure) `summoning_sick = false` for non-creature
            // entries. Tapped = false (Astelli Reclaimer's ETB does
            // NOT specify "tapped" — Python's
            // `put_into_battlefield` uses the printing's
            // `enters_tapped` flag which is False for noncreature
            // non-land permanents in EoE; we mirror the un-tapped
            // default).
            let entered_id: u32 = {
                let player = &mut state.players[controller_idx];
                let gy = Arc::make_mut(&mut player.graveyard);
                let mut obj = gy.remove(pick_idx);
                obj.zone = ZONE_BATTLEFIELD;
                obj.tapped = false;
                obj.summoning_sick = false;
                obj.controller = controller_u8;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                obj.etb_timestamp = etb_ts;
                let id = obj.id;
                Arc::make_mut(&mut player.battlefield).push(obj);
                id
            };
            // Queue ENTERS dispatch — drained AFTER the resolving
            // Astelli Reclaimer ETB trigger frame pops. Mirrors
            // Python's `put_into_battlefield`'s
            // `dispatch_event(Event.Enters, ...)` (which lands as a
            // triggers-this-turn entry and resolves at the next
            // priority cycle).
            state.pending_etb_dispatches.push(entered_id);
        }
        // **19th Rust slice (honest-stop substrate landing) —
        // Meltstrider's Resolve Aura ETB "fight".** "When this Aura
        // enters, enchanted creature fights up to one target creature
        // an opponent controls." Mirrors Python's
        // `EnchantedCreatureFightsTargetEffect.resolve`
        // (`effects/bespoke.py:538-575`).
        //
        // **HONEST-STOP semantics.** The Aura's source-side fighter is
        // its `attached_to` host. Today Rust does NOT set `attached_to`
        // when an Aura spell resolves (see docs/AURA_ATTACHMENT_
        // SUBSTRATE_PLAN.md §2 — the largest §2-declined Rust
        // multi-round substrate). Every Rust Aura has
        // `attached_to = None` at this resolve-arm's entry today, so
        // this arm fizzles silently — which is BYTE-EQUAL with Python's
        // behavior when `attached_to is None` (bespoke.py:546-548:
        // `if aura is None or aura.attached_to is None: return`). Once
        // the Aura attach substrate lands (Phases 1-2 of the substrate
        // plan), the hot-path branch below (both-fighters-present)
        // becomes live and the full simultaneous-damage logic exercises.
        //
        // **Resolve flow (mirrors bespoke.py:541-575 step-for-step).**
        //   1. Locate source object across all zones (Aura may still be
        //      on bf as its ETB trigger fires).
        //   2. Read source.attached_to. None → silent fizzle (honest-
        //      stop; byte-equal with Python).
        //   3. Resolve the enchanted creature on its controller's bf by
        //      id. Not found (already left bf) → silent fizzle (CR
        //      701.12c).
        //   4. Pick the opponent-side target — FIRST creature on the
        //      OPPONENT's bf (deterministic; mirrors
        //      `FightTargetCreatureWithTargetCreature`'s opponent scan).
        //      No opponent creature → silent fizzle ("up to one" =
        //      legal 0-pick).
        //   5. Compute BOTH effective powers BEFORE applying damage
        //      (CR 701.12 simultaneous damage event). Apply via
        //      `apply_damage_marker` + dispatch DEALT_DAMAGE for each
        //      receiver — mirrors the Fight arm's machinery.
        Effect::EnchantedCreatureFightsTarget => {
            // 1. Locate source — scan all zones (mirrors Fight arm).
            let mut src_zone_controller: Option<u8> = None;
            let mut src_attached_to: Option<u32> = None;
            'find_src: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_zone_controller = Some(obj.controller);
                        src_attached_to = obj.attached_to;
                        break 'find_src;
                    }
                }
            }
            // Source untrackable → silent fizzle (defensive — the
            // ETB-trigger resolve path should keep the source findable
            // on bf at this point).
            let Some(controller) = src_zone_controller else {
                return Ok(());
            };
            // 2. HONEST-STOP branch — Aura `attached_to` is None today
            // because Rust has no Aura attach substrate. Python takes
            // the same fizzle in this case (bespoke.py:546-548), so
            // cross-engine state stays byte-equal. When the attach
            // substrate lands (docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md
            // Phases 1-2), `attached_to` becomes `Some(host_id)` and
            // the code below the early-return runs.
            //
            // Helper: source-substitution for Aura `attached_to`. This
            // is the substrate primitive future Aura-shape Effect arms
            // (Pain for All ETB + damage echo — the substrate-sharing
            // pattern in docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md §)
            // reuse to route "source = the attached creature" without
            // duplicating the lookup. Inlined here for the first
            // consumer; lift to a helper once the second consumer
            // lands (the Pain for All ETB slice).
            let Some(enchanted_id) = src_attached_to else {
                return Ok(()); // honest-stop fizzle
            };
            // 3. Resolve the enchanted creature on its controller's
            // bf by id. The Aura's host is on the source-controller's
            // bf (an Aura attaches to a permanent its controller can
            // target; multi-player control changes are out of scope
            // for this honest-stop slice — when the attach substrate
            // lands the host's controller is read at resolve via the
            // same all-zone scan).
            let cidx = controller as usize;
            let enchanted_present = state.players[cidx]
                .battlefield
                .iter()
                .any(|o| o.id == enchanted_id);
            if !enchanted_present {
                return Ok(()); // host left bf → CR 701.12c fizzle
            }
            // 4. Pick the opponent-side target — FIRST creature on
            // the OPPONENT's bf. Mirrors FightTargetCreatureWithTarget
            // Creature's opponent-side scan.
            let opponent: u8 = 1 - controller;
            let oidx = opponent as usize;
            let registry = &state.registry;
            let opponent_fighter_id: Option<u32> = state.players[oidx]
                .battlefield
                .iter()
                .find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                });
            // "Up to one" — no opponent creature → legal 0-pick + the
            // spell resolves with no fight applied (Python parity:
            // bespoke.py:544 `if not targets: return`).
            let Some(opp_id) = opponent_fighter_id else {
                return Ok(());
            };
            // 5. Compute BOTH effective powers BEFORE applying any
            // damage (CR 701.12 simultaneous-damage event). Mirrors
            // Fight arm's same-shape read.
            let enchanted_power = read_battlefield_creature_power(state, enchanted_id);
            let opp_power = read_battlefield_creature_power(state, opp_id);
            // Apply both damages. A fighter that takes lethal in this
            // exchange still dealt its damage first because we read
            // both powers BEFORE either deal_damage call (CR 701.12).
            if enchanted_power > 0 {
                apply_damage_marker(state, opp_id, enchanted_power);
                dispatch_event(
                    state,
                    crate::cards::Event::DealtDamage,
                    opp_id,
                )?;
            }
            if opp_power > 0 {
                apply_damage_marker(state, enchanted_id, opp_power);
                dispatch_event(
                    state,
                    crate::cards::Event::DealtDamage,
                    enchanted_id,
                )?;
            }
        }
        // **Pain for All closure (Aura Phase 3 / S4 triggered shape).**
        // "When this Aura enters, enchanted creature deals damage equal
        // to its power to any other target." Mirrors Python's
        // `PainForAllETBEffect.resolve` (`effects/bespoke.py:383-422`)
        // STEP-FOR-STEP so the post-resolve state is byte-equal with
        // Python under matching inputs.
        //
        // **HONEST-STOP semantics.** Same shape as slice-19's
        // `EnchantedCreatureFightsTarget` arm. The source-side damage
        // dealer is the Aura's `attached_to` host; today Rust does NOT
        // set `attached_to` on Aura resolve (see
        // docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md §2 — the largest
        // §2-declined Rust multi-round substrate). Every Rust Aura has
        // `attached_to = None` at this resolve-arm's entry today, so
        // this arm fizzles silently — which is BYTE-EQUAL with Python's
        // behavior when `attached_to is None` (bespoke.py:389-391:
        // `if aura is None or aura.attached_to is None: return`). Once
        // the Aura attach substrate lands (Phases 1-2 of the substrate
        // plan), the hot-path branch below (enchanted-present)
        // becomes live and the full damage-deal logic exercises.
        //
        // **Target picking (deterministic first-legal).** Python's
        // `PainForAllETBEffect.legal_targets` returns
        // [(player, p.id) for each non-lost player] + [(creature, o.id)
        // for each opposing creature]. The PickFirst agent picks the
        // FIRST legal target. To stay byte-equal under the engine's
        // first-legal convention here, mirror the same target-pool
        // construction: enumerate players first (in player-id order),
        // then opponent creatures in bf order, and pick the first.
        // With two players, the first legal is the host's controller
        // (P0) — but Pain for All's pool EXCLUDES the enchanted
        // creature only; players are NOT excluded, so the resolving
        // controller's OWN player slot is a legal "any other target"
        // (the rules text "any other target" excludes the enchanted
        // creature, not the controller). The PickFirst agent for
        // first-legal sims selects the lowest-id player. Python's
        // `_PickFirstAgent.choose_target` selects `legal_targets[0]`
        // → P0 (the controller). The damage is `effective_power(bear)`
        // = 2, so P0 loses 2 life. THIS IS THE BYTE-EQUAL PATH.
        Effect::PainForAllETBEffect => {
            // 1. Locate source — scan all zones.
            let mut src_zone_controller: Option<u8> = None;
            let mut src_attached_to: Option<u32> = None;
            'find_src_pfa: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_zone_controller = Some(obj.controller);
                        src_attached_to = obj.attached_to;
                        break 'find_src_pfa;
                    }
                }
            }
            // Source untrackable → silent no-op (defensive).
            let Some(controller) = src_zone_controller else {
                return Ok(());
            };
            // 2. HONEST-STOP branch — Aura `attached_to` is None today
            // because Rust has no Aura attach substrate. Python takes
            // the same fizzle in this case (bespoke.py:389-391), so
            // cross-engine state stays byte-equal in the no-host case.
            // When the attach substrate lands, the code below the
            // early-return runs.
            let Some(enchanted_id) = src_attached_to else {
                return Ok(()); // honest-stop fizzle
            };
            // 3. Resolve the enchanted creature on its controller's bf.
            let cidx = controller as usize;
            let enchanted_present = state.players[cidx]
                .battlefield
                .iter()
                .any(|o| o.id == enchanted_id);
            if !enchanted_present {
                return Ok(()); // host left bf → silent fizzle
            }
            // 4. Compute effective power of the enchanted creature.
            //    Python parity: `amount = effective_power(state, enchanted)`.
            let amount = read_battlefield_creature_power(state, enchanted_id);
            if amount <= 0 {
                return Ok(()); // Python parity: `if amount <= 0: return`
            }
            // 5. Pick the deterministic first-legal target. Python's
            //    `legal_targets` returns players (in id order) followed
            //    by opposing creatures; the PickFirst agent selects
            //    [0] = lowest-id player = the controller (P0).
            //    Equivalent shape here: target the controller as the
            //    first-legal player. The damage is dealt as life-loss
            //    (Python parity: `victim.life -= amount` —
            //    bespoke.py:404-410).
            let target_player_idx = 0u8; // first-legal player
            if (target_player_idx as usize) < state.players.len() {
                state.players[target_player_idx as usize]
                    .life -= amount as i32;
            }
        }
        // **Pain for All closure (Aura Phase 3 / S4 triggered shape) —
        // PainForAllDamageEchoEffect resolve arm.** "Whenever enchanted
        // creature is dealt damage, it deals that much damage to each
        // opponent." Mirrors Python's
        // `PainForAllDamageEchoEffect.resolve` (`bespoke.py:443-457`)
        // STEP-FOR-STEP so the post-resolve state is byte-equal under
        // matching inputs.
        //
        // **Loop-guard discipline.** The echo applies as LIFE-LOSS
        // (`p.life -= amount`), NOT via the damage pipeline. This is
        // explicit in Python's docstring (`bespoke.py:438-441`:
        // "modeling 'deals damage' as life loss for simplicity in the
        // non-creature-target case") and inherently breaks the
        // potential recursion: the life-loss does not dispatch
        // DEALT_DAMAGE again, so the trigger doesn't re-fire on damage
        // it itself caused. The inventory called this out as a
        // "damage-feedback loop-guard" requirement; this is the
        // implementation.
        //
        // **Honest-stop continuity with the ETB arm.** Like the ETB
        // arm above, this resolve runs only when the Aura's
        // DEALT_DAMAGE+IsHost trigger fired — which requires
        // `source.attached_to == subject_id`. Today
        // `find_object_attached_to` returns `None` on every Aura, so
        // `Predicate::IsHost` returns false, so this arm is unreached
        // through the dispatch pipeline. It IS reachable through a
        // direct effect-resolve test path (probe) — that test path
        // exercises the byte-equal life-loss math without depending
        // on the attach substrate.
        Effect::PainForAllDamageEchoEffect => {
            // 1. Read the in-context damage amount. Python parity:
            //    `amount = state.resolving_combat_damage_amount`.
            let amount = state.resolving_combat_damage_amount;
            if amount <= 0 {
                return Ok(());
            }
            // 2. Resolve the effect's controller. Same pattern as the
            //    `SacrificeSelfUnlessTapPermanent` arm: prefer the
            //    transient set by the resolver around apply_effect,
            //    fall back to active_player.
            let controller_u8 = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            // 3. For each non-controller, non-lost player, apply the
            //    echo as life-loss. Python parity: bespoke.py:450-457.
            for idx in 0..state.players.len() {
                let player = &mut state.players[idx];
                if player.id == controller_u8 {
                    continue;
                }
                if player.has_lost {
                    continue;
                }
                player.life -= amount as i32;
            }
        }
        // **Aura Phase 4 (Chorale of the Void substrate landing) —
        // SacrificeSelfUnlessVoidActive.** End-step "sacrifice this
        // Aura unless Void is active". Mirrors Python's
        // `SacrificeSelfUnlessVoidActive.resolve`
        // (`effects/combinators.py:179-197`) STEP-FOR-STEP so the
        // post-resolve state is byte-equal:
        //
        //   source = state.resolving_source  (Rust: source_obj_id on bf)
        //   if source is None: return        (off-bf -> silent no-op)
        //   if is_void_active(state): return (Void active -> log + no-op)
        //   _sacrifice(state, source)        (Void NOT active -> sac)
        //
        // **Void state.** `state.is_void_active()` reads the engine-
        // wide `permanent_left_battlefield_this_turn ||
        // spell_warped_this_turn` flags (state.rs:2449-2455), set by
        // the LeavesBattlefield event dispatch and the warp-cost
        // cast path respectively. Same source of truth as Python's
        // `is_void_active(state)` (game.py:724 — reads the
        // `game.state` flags maintained by the same two dispatch
        // sites).
        //
        // **Controller resolution.** Same pattern as the
        // `SacrificeSelfUnlessTapPermanent` arm: prefer the transient
        // `current_resolve_source_controller` set by the resolver
        // around apply_effect, fall back to active_player. The end-
        // step trigger fires for the active player (the player whose
        // end step it is), so this transient is reliably populated
        // for the Chorale shape.
        Effect::SacrificeSelfUnlessVoidActive => {
            // 1. Locate the source on its controller's bf. Mirrors
            //    `SacrificeSelfUnlessTapPermanent` source-finder + the
            //    Python `if source is None: return` fizzle at
            //    combinators.py:186-188.
            let controller_u8 = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller_u8 as usize;
            let source_on_bf = state.players[controller_idx]
                .battlefield
                .iter()
                .any(|o| o.id == source_obj_id);
            if !source_on_bf {
                // Off-bf -> silent no-op (Python parity:
                // `if source is None: return` at combinators.py:186-188).
                return Ok(());
            }
            // 2. If Void is active -> silent no-op (Python parity:
            //    `if is_void_active(state): ...; return` at
            //    combinators.py:189-193 — the Void-active branch just
            //    logs and returns). Reads the same two engine-wide
            //    flags as `Effect::ConditionalIfVoid` (line 14535):
            //    `permanent_left_battlefield_this_turn ||
            //    spell_warped_this_turn`. This is the byte-equal
            //    Rust analogue of the pyo3 `is_void_active` getter
            //    at state.rs:2449-2455.
            let void_active = state.permanent_left_battlefield_this_turn
                || state.spell_warped_this_turn;
            if void_active {
                return Ok(());
            }
            // 3. Otherwise -> sacrifice the source directly (Python
            //    parity: `_sacrifice(state, source)` at
            //    combinators.py:197).
            sacrifice_source_inline(state, controller_idx, source_obj_id)?;
        }
        // **Chorale of the Void Cycle 1 (Aura S4 last sibling) —
        // ReanimateTargetCreatureCardFromDefendersGraveyard.** ATTACKS-
        // triggered "put target creature card from defending player's
        // graveyard onto the battlefield under your control tapped and
        // attacking". Mirrors Python's
        // `ReanimateTargetCreatureCardFromDefendersGraveyard.resolve`
        // (`effects/generic/graveyard.py:521-547`):
        //
        //   if not targets: return
        //   kind, target_id = targets[0]
        //   if kind != "graveyard_creature": return
        //   for p in state.players:
        //       card = next((c for c in p.graveyard if c.id == target_id), None)
        //       if card is None: continue
        //       p.graveyard.remove(card)
        //       put_into_battlefield(state, card, controller_id)
        //       card.tapped = True
        //       card.attacking = True
        //       card.summoning_sick = False
        //       return
        //
        // **HONEST-STOP RESOLVE ARM — substrate landed, full byte-equality
        // deferred (slice-19 Meltstrider precedent).** Today's
        // closure surfaces this resolve arm without a cast-time-bound
        // graveyard-creature target read (no `state.resolving_targets`
        // slot threads `("graveyard_creature", id)` shape through to
        // apply_effect_pub_with_controller — the cast-time bound-target
        // pump only fires for the spell path, not the trigger path).
        // The triggered-ability path drops the target read silently —
        // the Python class's `if not targets: return` fizzle at
        // graveyard.py:527-528 matches the Rust silent-no-op here. The
        // genuine resolve (move target's card → controller's bf tapped +
        // attacking) is reached when (a) the trigger-target pump threads
        // the cast-time target into the resolve site, AND (b) the
        // put-into-bf-tapped-and-attacking combat-pipeline integration
        // (state.attackers.push at mid-combat resolve) lands. See the
        // Effect variant docstring in `cards.rs` for the full Cycle 2
        // scope.
        //
        // **Cross-engine parity contract today.** The empty-defender-
        // graveyard / no-resolve-target case is byte-equal: both engines
        // observe the trigger fire, fizzle silently, and leave both
        // graveyards + battlefields unchanged. The PROBED-flip rides
        // this fizzle path (the only path the bench-loader's synthetic
        // probe needs to admit). The Aura admit gate widens to shape 7
        // (ATTACKS+IsHost(subject="attacker")+ReanimateTargetCreature
        // CardFromDefendersGraveyard) in this same lane.
        Effect::ReanimateTargetCreatureCardFromDefendersGraveyard => {
            // No-op: honest stop. The cast-time-bound graveyard-creature
            // target is not threaded through to this resolve site for
            // triggered abilities, so the candidate set is effectively
            // empty (mirrors Python's `if not targets: return` fizzle).
            // The substrate gate flips when the triggered-ability target
            // pump + combat-pipeline integration ship together.
            let _ = source_obj_id;
        }
        // **20th Rust slice (VVV Pin 2 closure) — Command Bridge ETB
        // "sacrifice this unless you tap an untapped permanent you
        // control".** Mirrors Python's
        // `SacrificeSelfUnlessTapPermanent.resolve`
        // (`effects/combinators.py:115-163`) STEP-FOR-STEP so the
        // post-resolve state is byte-equal:
        //
        //   source = state.resolving_source  (Rust: source_obj_id on bf)
        //   if source is None: return        (off-bf -> silent no-op)
        //   candidates = [o for o in state.battlefield
        //                 if o.controller_id == controller
        //                 and not o.tapped
        //                 and o.id != source.id]
        //   if not candidates:               (no choice -> sac source)
        //       _sacrifice(state, source); return
        //   --- here Python dispatches choose_target(kind=
        //       "untapped_permanent") and either taps or sacs ---
        //
        // **Cost-decision pump.** When the candidate set is non-empty,
        // we push a `PendingDecision::TapPermanentChoice` on the queue
        // (mirroring the `SurveilChoice` / `MayDoChoice` push-and-
        // short-circuit pattern). The controller drains via
        // `submit_tap_permanent_decision(Some(id))` (tap that
        // permanent, keep source) or
        // `submit_tap_permanent_decision(None)` (decline -> sacrifice
        // source). Auto-resolve `"default"` declines (matches Python's
        // no-agent fallback at `combinators.py:139-141`).
        //
        // **Source-on-bf scan.** Command Bridge typically enters
        // tapped before this ETB resolves (the land's own
        // enters-tapped status is set by `move_from_hand_to_battlefield`
        // before the ETB drain), so the candidate filter EXCLUDES
        // the source itself by id regardless of its tapped state
        // (Python: `o.id != source.id` at combinators.py:128).
        Effect::SacrificeSelfUnlessTapPermanent => {
            // 1. Locate the source on its controller's bf. Mirrors the
            //    `Effect::Sacrifice` controller-resolution shape (CR
            //    119.5 / 608.2 — effect resolves under the source's
            //    controller). The source-controller transient (set by
            //    the resolver around each apply_effect call) takes
            //    precedence over active_player.
            //
            //    The source MUST be on its controller's bf for this ETB
            //    to fire at all (an ENTERS-triggered ability fires from
            //    bf), so we scan only that one bf rather than every
            //    zone — mirrors the controller-pinned shape of the
            //    other Sacrifice / Tap / etc. arms above.
            let controller_u8 = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller_u8 as usize;
            let source_on_bf = state.players[controller_idx]
                .battlefield
                .iter()
                .any(|o| o.id == source_obj_id);
            if !source_on_bf {
                // Off-bf -> silent no-op (Python parity:
                // `if source is None: return` at combinators.py:122-123).
                // Defensive — the ETB trigger resolve path should keep
                // the source findable on bf at this point.
                return Ok(());
            }
            // 2. Build candidate set: untapped permanents controlled by
            //    the source's controller, EXCLUDING the source itself.
            //    Mirrors `combinators.py:124-129` exactly.
            let candidates: Vec<u32> = state.players[controller_idx]
                .battlefield
                .iter()
                .filter(|o| !o.tapped && o.id != source_obj_id)
                .map(|o| o.id)
                .collect();
            // 3. Empty candidate set -> sacrifice source directly (no
            //    decision needed). Mirrors `combinators.py:130-136`:
            //    `if not candidates: _sacrifice(state, source); return`.
            if candidates.is_empty() {
                sacrifice_source_inline(state, controller_idx, source_obj_id)?;
                return Ok(());
            }
            // 4. Non-empty -> push the cost-decision. The submit
            //    dispatcher routes Some(id)->tap, None->sacrifice.
            state.pending_decisions.push(
                crate::state::PendingDecision::TapPermanentChoice {
                    source_obj_id,
                    controller: controller_u8,
                    candidates,
                },
            );
        }
        // **Cracked Skull look-at-hand + may-discard substrate — Cycle 2
        // (2026-05-22).** Mirrors Python's
        // `TargetOpponentLookHandMayDiscard.resolve`
        // (`effects/generic/discard.py:330-385`) — Cracked Skull's
        // ETB trigger half.
        //
        // **Resolve steps (mirror Python).**
        // 1. Source-controller resolution: 5-zone walk + stack scan
        //    (mirrors the round-24 EachOpponentDiscards source-finder
        //    pattern). Source-not-found = silent fizzle.
        // 2. 2-player "target opponent" reduces to the one opponent:
        //    `opponent_idx = (1 - controller) as usize`.
        // 3. Compute candidates: every nonland card id in the
        //    opponent's hand, in hand-iteration order (matches
        //    Python's list-comprehension at `discard.py:352`).
        // 4. Empty candidates: silent no-op (Python parity at
        //    `discard.py:353-358`).
        // 5. Non-empty candidates: push
        //    `PendingDecision::LookAtOpponentHandMayDiscard
        //    {source_obj_id, controller, target_player, candidates}`
        //    onto the queue. The Cycle 1 substrate's
        //    `submit_look_at_opponent_hand_may_discard_decision`
        //    dispatcher consumes it; auto-resolve `"default"`
        //    declines.
        //
        // **No `revealed_to` mutation.** Rust's `GameObject` does NOT
        // carry a per-object `revealed_to` slot today; Python's
        // `reveal_to(card, controller)` step is a Python-side helper
        // that has no Rust-side analog. The cross-engine observable
        // that matters for parity (the discard/decline outcome on
        // `target_player.hand` / `target_player.graveyard`) is byte-
        // equal because neither engine reads `revealed_to` for any
        // subsequent state mutation. Documented honest-stop in
        // `docs/SEARCH_LIMITATIONS.md`.
        // **Sporogenic Infection — Cycle 1 (2026-05-22).** Mirrors
        // Python's `TargetOpponentSacrificesCreatureOtherThanEnchanted
        // .resolve` (`effects/generic/opponent_punishers.py:100-138`)
        // — Sporogenic Infection's Aura ETB trigger half.
        //
        // **Resolve steps (mirror Python).**
        // 1. Source-finder: walk all players' battlefields for the
        //    Aura source (the trigger only fires for an Aura on bf;
        //    if the source has LTB'd between trigger queueing and
        //    resolve, silent fizzle — defensive).
        // 2. Read `excluded_host_id = source.attached_to` (the Aura's
        //    enchanted creature; may be None on a stale Aura).
        // 3. 2-player "target opponent" reduces to the one opponent:
        //    `opponent_idx = (1 - controller) as usize`.
        // 4. Build candidates: every creature on the target opponent's
        //    battlefield EXCLUDING the attached-to host
        //    (`o.controller_id == target_idx && o.printing.is_creature
        //    && o.id != excluded_id`), bf-iteration order. Mirrors
        //    Python's list comprehension at
        //    `opponent_punishers.py:122-127`.
        // 5. Empty candidates: silent no-op (Python parity at
        //    `opponent_punishers.py:128`: `if not legal: return`).
        // 6. Non-empty candidates: push
        //    `PendingDecision::OpponentSacrificesCreatureChoice
        //    {source_obj_id, target_player, candidates}` onto the
        //    queue. The opponent's agent drains via
        //    `submit_opponent_sacrifices_creature_decision(id)`.
        //
        // **Forced-choice shape (no decline path).** Unlike Cracked
        // Skull's may-gate, the printed text "sacrifices a creature
        // of their choice" forces the sacrifice — the submit
        // dispatcher takes a required `chosen_id: u32` (not
        // `Option<u32>`).
        //
        // **`token` filter NOT applied.** Python's
        // `EachOpponentSacrificesCreature` ships a nontoken filter on
        // a different shape; Sporogenic's printed text says "a
        // creature" (not "a nontoken creature"), so tokens are
        // eligible candidates here. Python parity at
        // `opponent_punishers.py:84-86`.
        Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted => {
            // 1. Source-controller + attached_to resolution: 5-zone
            //    walk + stack scan (mirrors the sibling
            //    Effect::TargetOpponentLookHandMayDiscard arm verbatim
            //    on the source-finder shape; while the Aura should be
            //    on bf when the ETB fires, the source-finder defends
            //    against the source-LTB'd-between-trigger-queue-and-
            //    resolve edge case and reads `attached_to` from the
            //    bf object when present).
            let mut source_controller: Option<u8> = None;
            let mut excluded_host_id: Option<u32> = None;
            'outer: for player in state.players.iter() {
                for (zone_idx, zone) in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ]
                .iter()
                .enumerate()
                {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            // `attached_to` only meaningful on bf; we
                            // could read it off any zone, but the field
                            // is only non-None for an Aura on bf.
                            if zone_idx == 2 {
                                // 2 == battlefield in the array above
                                excluded_host_id = obj.attached_to;
                            }
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(src_controller) = source_controller else {
                return Ok(()); // source not found → silent fizzle
            };
            // 2. 2-player "target opponent" reduces to the one
            //    opponent.
            let target_idx = (1 - src_controller) as usize;
            // 3. Build candidate set: every creature on target
            //    opponent's bf EXCLUDING the attached-to host
            //    (filtered out even when the host is a creature on
            //    the same opponent's bf). Token creatures are NOT
            //    filtered (printed text "a creature", not "a nontoken
            //    creature").
            let candidates: Vec<u32> = state.players[target_idx]
                .battlefield
                .iter()
                .filter_map(|o| {
                    let printing = state.registry.lookup(o.printing_id)?;
                    if !printing.card_types.iter().any(|t| t == "Creature") {
                        return None;
                    }
                    if Some(o.id) == excluded_host_id {
                        return None;
                    }
                    Some(o.id)
                })
                .collect();
            // 4. Empty candidates: silent no-op (Python parity).
            if candidates.is_empty() {
                return Ok(());
            }
            // 5. Non-empty: push the decision. The opponent (target
            //    player) is the decision-maker; the variant carries
            //    no `controller` field by design.
            state.pending_decisions.push(
                crate::state::PendingDecision::OpponentSacrificesCreatureChoice {
                    source_obj_id,
                    target_player: target_idx as u8,
                    candidates,
                },
            );
        }
        Effect::TargetOpponentLookHandMayDiscard => {
            // 1. Source-controller resolution: 5-zone walk + stack
            //    scan. Mirrors round-24 EachOpponentDiscards /
            //    round-26 EachOpponentLosesLife shape.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found → silent fizzle
            };
            // 2. 2-player "target opponent" reduces to the one
            //    opponent.
            let opponent_idx = (1 - controller) as usize;
            // 3. Build candidate set: every nonland card id in the
            //    opponent's hand, in hand-iteration order.
            let candidates: Vec<u32> = state.players[opponent_idx]
                .hand
                .iter()
                .filter_map(|o| {
                    let printing = state.registry.lookup(o.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Land") {
                        None
                    } else {
                        Some(o.id)
                    }
                })
                .collect();
            // 4. Empty candidates: silent no-op (Python parity).
            if candidates.is_empty() {
                return Ok(());
            }
            // 5. Non-empty: push the decision.
            state.pending_decisions.push(
                crate::state::PendingDecision::LookAtOpponentHandMayDiscard {
                    source_obj_id,
                    controller,
                    target_player: opponent_idx as u8,
                    candidates,
                },
            );
        }
        // **20th Rust slice (XXX Pin 3 honest-stop closure) --
        // Lightstall Inquisitor each-opponent-exiles-card-from-hand-
        // with-cast-permission.** Mirrors Python's
        // `EachOpponentExilesCardFromHandWithCastPermission.resolve`
        // (`effects/generic/discard.py:187-208`) on the exile portion;
        // the persistent cast-from-exile permission flag substrate is
        // HONEST-STOPPED (Rust's GameObject has no
        // `cast_from_exile_with_lightstall_surcharge` field today; the
        // legal-actions emission + cast-machinery + tapped-land-entry
        // riders are not plumbed). See the Effect variant docstring
        // for the substrate-scope breakdown.
        //
        // **Cross-engine byte-equal up to the exile step.** Both
        // engines move a card from each opponent's hand to exile under
        // identical determinism (Python's `agent.choose_discard`
        // returning the first hand id == Rust's hand[0] pop). Post-
        // resolve zone counts agree. The cast-back-from-exile branch
        // is the partial-FIXED boundary -- Python surfaces the cast
        // in legal_actions; Rust does not.
        //
        // **Resolve steps (mirror Python).**
        // 1. Source-controller resolution -- 5-zone walk + stack scan
        //    (mirrors round-24 EachOpponentDiscards / EachOpponentLoses
        //    Life source-finder pattern). Source-not-found = silent
        //    fizzle.
        // 2. 2-player "each opponent" reduces to "the one opponent":
        //    `opponent_idx = (1 - controller) as usize`.
        // 3. Empty hand on the opponent = silent no-op (Python parity
        //    -- `if not p.hand: continue` short-circuits the loop body).
        // 4. Pop hand[0] of the opponent (deterministic first-legal
        //    pick; Python's `agent.choose_discard` returning the first
        //    hand id matches this pick byte-equal).
        // 5. CR 400.7 zone-change reset on the moved object: tapped /
        //    summoning_sick / damage / counters / attached_to /
        //    exiled_with_source all cleared; zone = ZONE_EXILE.
        // 6. Push the object to the opponent's exile vec.
        //
        // **No event dispatch.** This effect exiles (not discards) --
        // no `Event::DiscardsCard` fires (Python parity:
        // `discard.py:172-214` has no `fire_opponent_discards_triggers`
        // call; only `EachOpponentDiscards` does, at `discard.py:163`).
        // Honest-attribution boundary respected.
        Effect::EachOpponentExilesCardFromHandWithCastPermission => {
            // 1. Source-controller resolution: 5-zone walk + stack scan.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // 2. 2-player "each opponent" reduces to "the one
            // opponent" — opponent_idx = 1 - controller.
            let opponent_idx = (1 - controller) as usize;
            // 3. Empty hand: silent no-op (Python parity).
            if state.players[opponent_idx].hand.is_empty() {
                return Ok(());
            }
            // 4. Pop hand[0] of the opponent (deterministic first-
            // legal pick).
            let hand_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[opponent_idx].hand);
            let mut obj = hand_mut.remove(0);
            // 5. CR 400.7 zone-change reset.
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            obj.summoning_sick = false;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            // **Cast-from-exile-with-cost-hike substrate (Lightstall
            // Inquisitor residue closure).** Stamp the persistent
            // permission flag on the exiled card. Mirrors Python's
            // `effects/generic/discard.py:203` --
            // `card.cast_from_exile_with_lightstall_surcharge =
            // True`. The flag is honored by the legal emitter
            // (`state_ops_legal.rs::legal_decisions` exile-cast
            // surface) and the cast pay path
            // (`state_ops_cast.rs::cast_creature` /
            // `cast_spell_noncreature` / `play_land` for the land
            // contract); the cast pay path clears the flag on
            // successful cast/play. Persists until the card leaves
            // exile (not cleared at cleanup).
            obj.cast_from_exile_with_lightstall_surcharge = true;
            // 6. Push to opponent's exile.
            Arc::make_mut(&mut state.players[opponent_idx].exile).push(obj);
        }
        // **21st Rust slice (XXX Pin 1 closure) -- House Cartographer
        // reveal-until-land library walk.** "Reveal cards from the top
        // of your library until you reveal a land card. Put that card
        // into your hand and the rest on the bottom of your library in
        // a random order." Mirrors Python's
        // `HouseCartographerSurvivalEffect.resolve`
        // (`effects/bespoke.py:2690-2726`) STEP-FOR-STEP so the
        // post-resolve state is byte-equal:
        //   1. resolve source controller (CR 119.5 / 608.2)
        //   2. walk the controller's library from the top; collect
        //      non-lands into `revealed` until a land is popped or
        //      the library empties
        //   3. if a land was popped: push it to the controller's hand
        //      (CR 400.7 zone-change reset on the moved object)
        //   4. shuffle `revealed` via the byte-equal external_rng path
        //      (when a Python random.Random is installed via
        //      `set_external_rng`) / xorshift64 fallback otherwise
        //   5. append each shuffled card onto the BOTTOM of the
        //      controller's library
        //
        // **No PendingDecision.** Both engines: deterministic library
        // walk + state.rng-owned "random order" tail.
        Effect::HouseCartographerSurvivalEffect => {
            // 1. Source-controller resolution. Same shape as the
            // slice-18 Roving Actuator / slice-20 Lightstall arms:
            // prefer the transient current_resolve_source_controller
            // (set by the stack resolver around each apply_effect
            // call), fall back to active_player. "Your library" =
            // the source's controller's library.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller as usize;
            // 2. Library walk: pop from the top (index 0) of the
            // controller's library until either the library is empty
            // OR a land is popped. The popped land becomes
            // `land_card`; non-lands accumulate in `revealed` (which
            // becomes the "rest" the bottom-shuffle operates on).
            //
            // Mirrors Python's `while p.library: top = p.library.pop(0)`
            // loop at `bespoke.py:2699-2704`. The two-phase split
            // (drain into local vecs first, then apply the moves to
            // the player's zones) avoids a borrow conflict between
            // the `state.registry.lookup` immutable borrow and the
            // `Arc::make_mut(&mut player.library)` mutable borrow.
            let (revealed, land_card): (Vec<GameObject>, Option<GameObject>) = {
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                let mut revealed: Vec<GameObject> = Vec::new();
                let mut land_card: Option<GameObject> = None;
                while !lib.is_empty() {
                    let top = lib.remove(0);
                    let is_land = match state.registry.lookup(top.printing_id) {
                        Some(pr) => pr.card_types.iter().any(|t| t == "Land"),
                        None => false,
                    };
                    if is_land {
                        land_card = Some(top);
                        break;
                    }
                    revealed.push(top);
                }
                (revealed, land_card)
            };
            // 3. Route the revealed land (if any) into the controller's
            // hand. Apply the CR 400.7 zone-change reset so the moved
            // object is byte-equal with Python's library->hand
            // transition. Mirrors `bespoke.py:2705-2711`.
            if let Some(mut land_card) = land_card {
                land_card.zone = ZONE_HAND;
                land_card.tapped = false;
                land_card.summoning_sick = false;
                land_card.damage = 0;
                land_card.counters = 0;
                land_card.attached_to = None;
                land_card.exiled_with_source = None;
                let player = &mut state.players[controller_idx];
                Arc::make_mut(&mut player.hand).push(land_card);
            }
            // 4. Shuffle the local `revealed` Vec via the same dual-
            // path RNG primitive every other in-Effect shuffle site
            // uses (external `random.Random.shuffle` via
            // `external_rng::next_below` when an external rng is
            // installed; inline xorshift64 fallback otherwise).
            // Mirrors `bespoke.py:2718` (`state.rng.shuffle(revealed)`).
            //
            // The shuffle happens BEFORE the bottom-append so the
            // append order = the shuffled order (Python parity:
            // `state.rng.shuffle(revealed); for card in revealed:
            // p.library.append(card)`).
            let mut revealed = revealed;
            let n = revealed.len();
            if n >= 2 {
                let py_rng_owned: Option<Py<PyAny>> = state
                    .rng_ref()
                    .map(|r| Python::with_gil(|py| r.clone_ref(py)));
                if let Some(py_rng) = py_rng_owned {
                    Python::with_gil(|py| -> PyResult<()> {
                        let bound = py_rng.bind(py);
                        for i in (1..n).rev() {
                            let j = crate::external_rng::next_below(bound, i + 1)?;
                            revealed.swap(i, j);
                        }
                        Ok(())
                    })?;
                } else {
                    // xorshift64 fallback: distinct seed per (turn,
                    // ap, source, n) tuple — same control flow as
                    // every other xorshift site in the engine
                    // (SearchLibraryForBasicLandTapped /
                    // FamishedWorldsireLookTopXByPowerPutLandsTapped
                    // precedent).
                    let mut seed = (state.turn_number as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ ((state.active_player as u64) << 56)
                        ^ ((source_obj_id as u64) << 16)
                        ^ (n as u64);
                    if seed == 0 {
                        seed = 0xdead_beef_dead_beef;
                    }
                    let mut x = seed;
                    for i in (1..n).rev() {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let n_u64 = (i + 1) as u64;
                        let bound = u64::MAX - (u64::MAX % n_u64);
                        let mut r = x;
                        while r >= bound {
                            r ^= r << 13;
                            r ^= r >> 7;
                            r ^= r << 17;
                            x = r;
                        }
                        let j = (r % n_u64) as usize;
                        revealed.swap(i, j);
                    }
                }
            }
            // 5. Append each shuffled card onto the BOTTOM of the
            // controller's library (= `lib.push(card)` -- index 0 is
            // the top; append IS the bottom). Stamp zone=LIBRARY on
            // the way in (mirrors Python's `card.zone = Zone.LIBRARY;
            // p.library.append(card)` at `bespoke.py:2719-2721`).
            if !revealed.is_empty() {
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                for mut card in revealed.into_iter() {
                    card.zone = ZONE_LIBRARY;
                    lib.push(card);
                }
            }
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Sothera, the Supervoid's DIES-trigger body: each opponent
        // chooses one creature they control and exiles it. With
        // `link_to_source = true`, each exiled object would be tagged
        // with `exiled_by_source = state.resolving_source.id` so a
        // sibling end-step ability on the same source can later
        // recall it.
        //
        // **HONEST-STOP RESOLVE ARM.** The exile-portion of the body
        // (each opponent exiles their first creature) is shipped
        // verbatim — byte-equal with Python's resolve when no agent
        // hook is plumbed (Python uses `agent.choose_sacrifice`
        // returning the first candidate by deterministic test-agent
        // contract; Rust here picks the first creature without an
        // agent callback, which matches the no-agent-supplied
        // deterministic fallback at `opponent_punishers.py:200-201`:
        // `chosen = next((o for o in candidates if o.id == pick_id),
        // candidates[0])`).
        //
        // **Source-link bookkeeping deferred.** When
        // `link_to_source = true`, Python stamps
        // `chosen.exiled_by_source = source.id`; Rust here can't
        // record the link because `GameObject` has no
        // `exiled_by_source` field (large multi-cycle substrate per
        // `BLOCKED_GENUINELY_MISSING_INVENTORY.md` Sothera entry).
        // The exile move itself still ships; only the cross-card
        // source-link bookkeeping is the deferred piece. Without the
        // link, the sister `ReanimateOneExiledWithSourceWithCounters`
        // arm cannot find the exiled creature at end-of-turn — but
        // that arm ALSO honest-stops to a silent fizzle in this
        // slice, so the cross-engine state remains byte-equal on the
        // FULL DIES + end-step sequence (both engines: P_opp loses
        // one creature; neither engine recalls it; Sothera is
        // sacrificed normally without the exile-link suppression).
        Effect::EachOpponentExilesOwnCreature { link_to_source: _ } => {
            // 1. Source-controller resolution: 5-zone walk + stack scan
            // (mirrors slice-20 Lightstall source-controller resolution).
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // 2. Per-opponent loop. 2-player engine: the one opponent
            // is `1 - controller`. Mirrors Python's
            // `for opp in state.players: if opp.id == controller_id or
            // opp.has_lost: continue` at `opponent_punishers.py:187-188`.
            let opponent_idx = (1 - controller) as usize;
            if opponent_idx >= state.players.len() {
                return Ok(());
            }
            // 3. First-legal creature candidate scan on the opponent's
            // battlefield. Mirrors `candidates = [o for o in
            // state.battlefield if o.controller_id == opp.id and
            // o.printing.is_creature]`; deterministic first pick (no-
            // agent fallback `candidates[0]`).
            let pick_id: Option<u32> = {
                let bf = &state.players[opponent_idx].battlefield;
                let mut found: Option<u32> = None;
                for obj in bf.iter() {
                    let is_creature = match state.registry.lookup(obj.printing_id) {
                        Some(pr) => pr.card_types.iter().any(|t| t == "Creature"),
                        None => false,
                    };
                    if is_creature {
                        found = Some(obj.id);
                        break;
                    }
                }
                found
            };
            let Some(pick_id) = pick_id else {
                // No legal creature: silent skip (Python parity at
                // `opponent_punishers.py:194-195`).
                return Ok(());
            };
            // 4. Move the chosen object from the opponent's bf to the
            // opponent's exile zone. Mirrors `move_off_battlefield(
            // state, chosen, destination="exile")` with CR 400.7
            // zone-change reset on the moved object.
            //
            // **Source-link bookkeeping is the deferred piece.**
            // Python stamps `chosen.exiled_by_source = source.id` here
            // when `link_to_source = true`; Rust does not have such a
            // field. Honest-stop: the exile move ships; the link does
            // not. See the Effect variant docstring.
            let bf_mut = Arc::make_mut(&mut state.players[opponent_idx].battlefield);
            let Some(pos) = bf_mut.iter().position(|o| o.id == pick_id) else {
                return Ok(()); // raced; defensive
            };
            let mut obj = bf_mut.remove(pos);
            let printing_id = obj.printing_id;
            let leaving_obj_id = obj.id;
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            obj.damage = 0;
            obj.counters = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            Arc::make_mut(&mut state.players[opponent_idx].exile).push(obj);
            // Battlefield-exit bookkeeping (mirrors the `Effect::Exile`
            // / `Effect::Sacrifice` arm conventions).
            mark_left_battlefield_if_nonland(state, printing_id);
            state.pending_leaves_dispatches.push(leaving_obj_id);
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Sothera, the Supervoid's end-step "put one creature card
        // exiled with this permanent onto the battlefield under your
        // control with N additional +1/+1 counters" body.
        //
        // **HONEST-STOP RESOLVE ARM — silent fizzle.** The candidate
        // set is built from cards tagged with `exiled_by_source ==
        // source.id`, but Rust's `GameObject` has no `exiled_by_source`
        // field today (large multi-cycle substrate per
        // `BLOCKED_GENUINELY_MISSING_INVENTORY.md` Sothera entry). The
        // candidate set is always empty — silent fizzle. Python takes
        // the same fizzle when `candidates` is empty
        // (`exile.py:1178-1179`: `if not candidates: return`), so the
        // cross-engine state stays byte-equal in the no-candidate case.
        // The hot-path branch (`put_into_battlefield` + add counters)
        // becomes ready-to-flip when the source-link substrate ships.
        Effect::ReanimateOneExiledWithSourceWithCounters { counter_amount: _ } => {
            // No-op: honest stop. The substrate gate
            // (`obj.exiled_by_source == source.id` field check) is
            // never satisfied because the field doesn't exist. This is
            // the slice-19 Meltstrider precedent: silently match
            // Python's same-case fizzle until the deferred substrate
            // lands.
            let _ = source_obj_id;
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Sothera, the Supervoid's "sacrifice this, preserving the
        // source-link" body. Companion to
        // `ReanimateOneExiledWithSourceWithCounters` — together they
        // form Sothera's end-step Sequence (recall first, sac second).
        //
        // **HONEST-STOP RESOLVE ARM.** The sacrifice itself ships via
        // the existing `sacrifice_source_inline` helper. The one-shot
        // `suppress_exiled_by_source_on_leave_for` arming is a no-op
        // in Rust today because the on-leave auto-return hook itself
        // isn't wired — both halves of the suppression machinery (the
        // auto-return AND the suppression) are deferred together, so
        // cross-engine state stays byte-equal on the sac event
        // (Python's pre-suppression state: source is on bf; Python's
        // post-resolve state: source is in graveyard with no auto-
        // return having fired because no creature was tagged with the
        // missing source-link in the first place — identical to
        // Rust's post-resolve state).
        Effect::SacrificeSelfPreservingExileLink => {
            // Locate the source's controller. Mirrors the Sothera
            // pattern: the source is presumed on bf (CR 701.16: "The
            // controller of the ability sacrifices the source").
            let mut source_controller: Option<u8> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if player.battlefield.iter().any(|o| o.id == source_obj_id) {
                    source_controller = Some(pidx as u8);
                    break;
                }
            }
            let Some(controller) = source_controller else {
                // Source already off the bf (e.g. destroyed by an
                // earlier sequence step) — silent no-op, mirroring
                // Python's `if source not in state.battlefield: return`
                // at `exile.py:1262-1263`.
                return Ok(());
            };
            sacrifice_source_inline(state, controller as usize, source_obj_id)?;
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Requiem Monolith's UEOT grant of a quoted dealt-damage
        // trigger.
        //
        // **HONEST-STOP RESOLVE ARM — silent no-op.** Rust has no
        // general layer-6 grant-triggered-ability registry (only the
        // narrow Perigee `GrantedDiesReturnTapped` marker — see
        // `state.rs:299`). The Effect-level closure ships the variant
        // + parse + to_pydict + bridge round-trip; the resolve arm
        // records nothing today. When the broader grant-triggered-
        // ability registry lands, this arm flips to writing a UEOT-
        // bounded layer-6 entry on the chosen target.
        Effect::GrantTriggeredAbilityUEOTRequiem => {
            // No-op: honest stop. The chosen target is left untouched
            // because there's nowhere to write the granted ability.
            let _ = source_obj_id;
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Requiem Monolith's optional 1-damage ping: "That creature's
        // controller may have <source> deal N damage to it."
        //
        // **HONEST-STOP RESOLVE ARM — silent no-op.** The choice is
        // owned by the TARGET's controller (NOT the activator), which
        // requires a `choose_yes_no` agent callback Rust has not
        // plumbed at this resolve site. Python's same-case fallback
        // (`target_controller.agent is None: return` at
        // `damage.py:1161`) is identical — both engines fizzle when
        // no agent is wired.
        //
        // The cross-engine round-trip closes for this Effect class via
        // the bridge arm; the hot-path branch (`tgt.damage += amount`
        // + fire enchanted-creature dealt-damage triggers) becomes
        // ready-to-flip when the `choose_yes_no` callback substrate
        // is plumbed.
        Effect::MayHaveSourceDealDamageToTarget { amount: _ } => {
            // No-op: honest stop. Cross-engine state stays byte-equal
            // because Python ALSO returns early in the no-agent case.
            let _ = source_obj_id;
        }
        // **22nd Rust slice (Sothera + Requiem closures honest-stop).**
        // Requiem Monolith's granted-trigger BODY: "You draw that many
        // cards and lose that much life." Mirrors Python's
        // `DrawCardsAndLoseLifeEqualToResolvingDamage.resolve`
        // (`effects/generic/damage.py:1089-1108`).
        //
        // **Direct-resolve byte-equal (not honest-stop).** This arm is
        // exercised only when a caller drives it directly via
        // `apply_effect` with `state.resolving_combat_damage_amount`
        // pre-set (e.g. a synthetic probe). It reads the amount slot,
        // fizzles silently if 0, locates the source's controller, and
        // applies `draw_card × amount` + `life -= amount` to that
        // controller. The Python parity-checks: `if amount <= 0:
        // return`, `if source is None: return`. The arm is wired even
        // though no production code-path reaches it today, because
        // the cross-engine round-trip needs the resolve to be present
        // for the dict-shape-consistency probe to pass.
        Effect::DrawCardsAndLoseLifeEqualToResolvingDamage => {
            let amount = state.resolving_combat_damage_amount;
            if amount <= 0 {
                return Ok(());
            }
            // Source-controller resolution: 5-zone walk + stack scan.
            // Mirrors the slice-20 Lightstall source-controller-
            // resolution shape.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            let Some(controller) = source_controller else {
                return Ok(());
            };
            let controller_idx = controller as usize;
            if controller_idx >= state.players.len() {
                return Ok(());
            }
            // Draw `amount` cards. Mirrors Python's `for _ in
            // range(amount): draw_card(state, owner.id)`. Inlined per
            // the file-isolation convention (see the `Effect::DrawCards`
            // arm's inlining note at `state_ops_resolve.rs:1892`).
            for _ in 0..amount.max(0) {
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                if lib.is_empty() {
                    // CR 704.5b — empty-library draw flags the player
                    // (same shape as the `Effect::DrawCards` arm).
                    player.attempted_draw_on_empty = true;
                    break;
                }
                let mut obj = lib.remove(0);
                obj.zone = ZONE_HAND;
                Arc::make_mut(&mut player.hand).push(obj);
            }
            // Lose `amount` life. Mirrors `owner.life -= amount`.
            let player_mut = &mut state.players[controller_idx];
            player_mut.life = player_mut.life.saturating_sub(amount);
        }
        // **18th Rust slice (XXX Pin 2 closure) — Roving Actuator
        // Void ETB.** "Exile up to one target instant or sorcery card
        // with mana value 2 or less from your graveyard. Copy it. You
        // may cast the copy without paying its mana cost." Mirrors
        // Python's `RovingActuatorETBEffect.resolve`
        // (`effects/bespoke.py:661-711`) STEP-FOR-STEP so the
        // post-resolve state is byte-equal:
        //   1. resolve source controller (CR 119.5 / 608.2)
        //   2. first-legal gy scan: instant/sorcery && MV <= max_mv
        //   3. if no card matches: silent no-op ("up to one")
        //   4. remove from gy; stamp zone=EXILE; push to controller's exile
        //   5. apply the exiled card's spell_effect (printing.effects[0])
        //      recursively via apply_effect with the exiled card's id
        //      as the new source — the copy resolves on the engine
        //      directly without going through the stack (intentional
        //      first-slice simplification; the copy CAN'T be countered).
        //
        // **No PendingDecision.** Python's `requires_target()` is True,
        // but the target binding is a `legal_targets`-driven first-
        // legal pick at resolve time on both sides (mirrors the slice
        // 16 Astelli Reclaimer / round-54 Famished Worldsire pattern).
        // The COPY's own targets (e.g. Plasma Bolt → 2 damage to a
        // target) bind via the recursive `apply_effect` arm's standard
        // first-legal picker. The source-controller transient stays
        // pinned through the recursive call so any source-controller-
        // aware arm (DealDamage / DrawCards / GainLife) reads the
        // Roving Actuator's controller as the casting source for the
        // copy.
        Effect::RovingActuatorETBEffect { max_mv } => {
            // 1. Source-controller resolution. Mirrors the Astelli
            // Reclaimer (slice 16) shape: prefer the transient
            // current_resolve_source_controller (set by the stack
            // resolver around each apply_effect call), fall back to
            // active_player.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let controller_idx = controller as usize;
            let max_mv_u32 = *max_mv as u32;
            // 2. First-legal graveyard scan. Walk in insertion order;
            // pick the FIRST card whose printing satisfies
            // (is_instant || is_sorcery) && mana_cost.total() <= max_mv.
            // The instant/sorcery filter rejects creatures / lands /
            // permanents (Python's `legal_targets` builds a
            // `graveyard_spell` tuple-kind only for those two
            // card_types, see `bespoke.py:650-659`).
            let pick_idx: Option<usize> = {
                let gy = &state.players[controller_idx].graveyard;
                let mut found: Option<usize> = None;
                for (i, obj) in gy.iter().enumerate() {
                    let pr = match state.registry.lookup(obj.printing_id) {
                        Some(p) => p,
                        None => continue,
                    };
                    let is_instant =
                        pr.card_types.iter().any(|t| t == "Instant");
                    let is_sorcery =
                        pr.card_types.iter().any(|t| t == "Sorcery");
                    if !is_instant && !is_sorcery {
                        continue;
                    }
                    let mc = &pr.mana_cost;
                    let mv: u32 = mc.generic as u32
                        + mc.white as u32
                        + mc.blue as u32
                        + mc.black as u32
                        + mc.red as u32
                        + mc.green as u32;
                    if mv <= max_mv_u32 {
                        found = Some(i);
                        break;
                    }
                }
                found
            };
            // 3. If no card matches: silent no-op (Python parity —
            // "up to one" is fine with 0 picks).
            let pick_idx = match pick_idx {
                Some(i) => i,
                None => return Ok(()),
            };
            // 4. Remove the picked card from the graveyard; stamp
            // zone=EXILE; reset transient state; push to controller's
            // exile. Mirrors Python's `bespoke.py:677-679`
            // (`p.graveyard.remove(card); card.zone = Zone.EXILE;
            // p.exile.append(card)`). Clone printing_id + obj id
            // BEFORE the move so the recursive apply_effect call below
            // can look the exiled card up on the exile by id.
            let exiled_id: u32 = {
                let player = &mut state.players[controller_idx];
                let gy = Arc::make_mut(&mut player.graveyard);
                let mut obj = gy.remove(pick_idx);
                obj.zone = ZONE_EXILE;
                // Clear per-zone transient fields (mirrors the
                // gravetomb-to-exile Effect::Counterspell-shape arm
                // at state_ops_resolve.rs:2319-2331). The copy's
                // resolution reads only printing_id, so most of these
                // are belt-and-suspenders, but we mirror the
                // convention exactly to avoid byte-equal drift on
                // future probes.
                obj.tapped = false;
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let id = obj.id;
                Arc::make_mut(&mut player.exile).push(obj);
                id
            };
            // 5. Apply the exiled card's spell_effect. Python:
            // `spell_effect.resolve(state, controller_id, copy_targets)`.
            // Rust: the printing's `effects: Vec<Effect>` is the
            // single-element list the bridge builds from
            // `card.spell_effect` (`cross_engine.py:10207-10208`).
            // Iterate and call apply_effect recursively with the
            // EXILED card's id as the new source — that id lives on
            // exile (we just pushed it) so source-controller-aware
            // arms find it via the standard zone-scan + correctly
            // attribute the copy's effect to the Roving Actuator's
            // controller.
            //
            // **Empty effects = silent no-op.** If the exiled card has
            // no spell_effect (e.g. a creature card — but the
            // instant/sorcery filter above rejects those; this is a
            // belt-and-suspenders guard for synthetic test cards that
            // tag a printing as Instant without giving it a
            // spell_effect, which the Python code path also handles
            // via `if spell_effect is None: return`).
            //
            // Clone the effects Vec into a local before the recursive
            // call so we drop the borrow on state.registry across the
            // recursive mutation. The clone is cheap (a handful of
            // Effect enum copies, typically one) and mirrors the
            // Sequence arm's per-iteration `effects` consumption.
            let effects_to_apply: Vec<Effect> = match state
                .registry
                .lookup(
                    state.players[controller_idx]
                        .exile
                        .iter()
                        .find(|o| o.id == exiled_id)
                        .map(|o| o.printing_id)
                        .unwrap_or(0),
                ) {
                Some(pr) => pr.effects.clone(),
                None => Vec::new(),
            };
            // Pin the source-controller for the duration of the
            // copy's apply_effect calls so any source-controller-
            // aware arm (DealDamage / DrawCards / GainLife) reads
            // the Roving Actuator's controller as the casting source.
            // Mirrors `apply_effect_pub_with_controller`'s
            // save-swap-restore pattern; inline here because we're
            // already inside `apply_effect` (which is `fn`-private
            // and can't recurse through the `pub_with_controller`
            // helper without an extra hop).
            let saved_controller = state.current_resolve_source_controller;
            state.current_resolve_source_controller = Some(controller);
            let result: PyResult<()> = (|| {
                for child in &effects_to_apply {
                    apply_effect(state, child, exiled_id)?;
                }
                Ok(())
            })();
            state.current_resolve_source_controller = saved_controller;
            result?;
        }
        // Round-28 ModalChooseOne first-slice + Cycle-2 cast-time-
        // bind widening.
        //
        // **Cycle 2 (cast-time mode dispatch, 2026-05-21).** When the
        // resolving spell carries a cast-time-bound mode pick via
        // `state.current_resolve_chosen_mode_idx` (set by the stack
        // resolver from the StackItem's `chosen_mode_idx`), dispatch
        // directly to `modes[idx]` via `apply_effect_pub_with_
        // controller` — bypassing the resolve-time decision callback
        // entirely. Mirrors Python's `ModalChooseOne.resolve` at
        // `combinators.py:1213-1215`:
        //     cast_modes = state.resolving_modes
        //     if cast_modes:
        //         idx = cast_modes[0]
        //         ...
        //         chosen = self.modes[idx]
        //         chosen.resolve(...)
        // The cast-time-bound case is the spell-cast hot path (every
        // EoE modal spell cast through `cast_spell_noncreature` with
        // a deterministic-first agent — the bench's `_PickFirstLegal
        // TargetAgent` convention or the no-agent Python fallback at
        // `actions.py:1222-1223`).
        //
        // **Fallback to resolve-time PendingDecision.** When
        // `current_resolve_chosen_mode_idx` is `None` (no cast-time
        // bind — every nested-as-trigger ChooseOneMode + every
        // ChooseOneMode applied via `apply_effect` directly outside
        // the spell-cast pipeline, e.g. in synthetic tests), the
        // pre-cycle resolve-time PendingDecision path stays intact.
        // The honest no-fallback-hybrid discipline: the resolve arm
        // routes by the EXPLICIT presence/absence of a cast-time
        // bind, not by a printing scan.
        //
        // **Out-of-range clamp.** `idx >= modes.len()` is reachable
        // only via a deliberately malformed StackItem (the cast-time
        // bind sets `Some(0)` and modes is parse-rejected at empty).
        // Mirror Python's `idx = max(0, min(idx, len(modes) - 1))`
        // clamp at `combinators.py:1215` defensively.
        //
        // Controller convention (both paths): prefer the transient
        // source-controller swap (I61). Modes are cloned for the
        // PendingDecision shape (owns its copy); the cast-time-bind
        // dispatch passes the registry-clone `mode` ref into
        // apply_effect_pub_with_controller (already-cloned at the
        // outer effect-snapshot in `resolve_top_of_stack`).
        Effect::ChooseOneMode { modes, or_both: _ } => {
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            // **scour-for-scrap-slice Cycle 1 widen (2026-05-21).**
            // Prefer the multi-pick swap when set; fall through to
            // the single-pick swap for back-compat; final fallback is
            // the resolve-time PendingDecision (no cast-time bind).
            //
            // Multi-pick dispatch iterates the chosen subset in the
            // SUPPLIED ORDER (mirrors Python's CR 700.2d "modes
            // resolve in the order chosen by the controller"). Each
            // chosen mode is dispatched via
            // `apply_effect_pub_with_controller` exactly like the
            // single-pick arm — same source / same controller / same
            // semantics, repeated.
            //
            // **Out-of-range clamp.** Indices >= modes.len() are
            // silently dropped (mirrors Python's sanitisation at
            // `combinators.py::ModalChooseOne.resolve` which drops
            // out-of-range and dedupes via `dict.fromkeys`).
            if let Some(idxs) = &state.current_resolve_chosen_mode_idxs {
                let chosen_subset: Vec<crate::cards::Effect> = idxs
                    .iter()
                    .filter_map(|i| modes.get(*i as usize).cloned())
                    .collect();
                for chosen in &chosen_subset {
                    apply_effect_pub_with_controller(
                        state,
                        chosen,
                        source_obj_id,
                        controller,
                    )?;
                }
                // If every supplied index was out-of-range AND the
                // chosen subset is empty, fall through silently — the
                // resolver did its job (the caller asked for nothing
                // legal). This mirrors Python's "no valid indices →
                // fall back to mode 0" sanitiser ONLY at the spec
                // boundary; at the Rust cast-time bind we always
                // emit at least one valid index, so the empty path
                // is unreachable through the production carrier.
            } else {
                match state.current_resolve_chosen_mode_idx {
                    Some(idx_u8) => {
                        let idx = (idx_u8 as usize).min(modes.len().saturating_sub(1));
                        let chosen = modes[idx].clone();
                        apply_effect_pub_with_controller(
                            state,
                            &chosen,
                            source_obj_id,
                            controller,
                        )?;
                    }
                    None => {
                        state.pending_decisions.push(
                            PendingDecision::ChooseModeChoice {
                                source_obj_id,
                                controller,
                                modes: modes.clone(),
                            },
                        );
                    }
                }
            }
        }
        // Round-28 IfYouDo first-slice: push an IfYouDoChoice
        // carrying both nested Effects on the heap. Controller drains
        // via submit_if_you_do_decision(yes) (manual) or auto-resolve
        // "default" / "if_you_do_yes" (apply both) / "if_you_do_no"
        // (skip both).
        //
        // Controller convention: state.active_player. Both nested
        // effects cloned so the registry's printing keeps its
        // originals and the PendingDecision owns its copies.
        Effect::IfYouDo { do_action, then_action } => {
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            state.pending_decisions.push(PendingDecision::IfYouDoChoice {
                source_obj_id,
                controller,
                do_action: do_action.clone(),
                then_action: then_action.clone(),
            });
        }
        // Round-28 EachOpponentMillsCards (3+ EoE cards). Mirrors the
        // round-24 EachOpponentDiscards / EachOpponentLosesLife
        // source-finder shape: 5-zone walk + stack scan to recover
        // source's controller, then mill the (single 2-player)
        // opponent's library `count` cards. Empty library stops the
        // inner loop silently — CR 704.5b applies to drawing, not
        // milling.
        // Round-28 broader-filter pump (artifact OR creature). Same
        // shape as PumpTargetCreatureUntilEOT but the eligibility
        // filter is "card_types contains Artifact OR Creature"
        // rather than just "Creature". UEOT-modifier mechanism is
        // identical — modifier auto-expires at the next cleanup step
        // via the r27 CR 514.3 UEOT-clear pass.
        Effect::PumpTargetArtifactOrCreatureUntilEOT {
            power_bonus,
            toughness_bonus,
            granted_keywords,
        } => {
            // Reroute Systems closure (2026-05-21): a bare-pump (no
            // granted keywords) with (0, 0) bonus short-circuits as
            // before; once granted_keywords is non-empty, the
            // keyword-grant alone justifies running the target scan
            // even if power+toughness are both 0 (Reroute Systems
            // mode 0 IS the canonical 0/0/{"indestructible"} shape).
            if *power_bonus == 0 && *toughness_bonus == 0 && granted_keywords.is_empty() {
                return Ok(());
            }
            // Source-controller resolution (mirrors PumpTargetCreatureUntilEOT).
            // 5-zone walk + stack fallback.
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            if src_controller.is_none() {
                return Ok(()); // source untrackable → silent fizzle
            }
            // Reroute Systems closure (2026-05-21): widen the target
            // scan from controller-only to BOTH battlefields. Python
            // SoT (`PumpTargetArtifactOrCreatureUntilEOT` →
            // unified `Pump.resolve` walking ANY-permanent filter)
            // does NOT controller-restrict — the printed oracle
            // "target artifact or creature" carries no controller
            // filter. The pre-closure controller-only scan was a
            // first-slice narrowing that was load-bearing-WRONG for
            // Reroute Systems mode 0 ("gains indestructible UEOT"
            // on an opponent's permanent is a legal target). Mirrors
            // `DepressurizeEffect` / `CutPropulsionEffect` /
            // `DealDamageToTargetCreatureSpacecraftBoosted` ANY-
            // creature scan precedent.
            //
            // Prefer the at-cast bound target (via the per-mode
            // detector table extended in this closure); fall back to
            // the deterministic-first scan over both bfs if no
            // bound target was threaded (defensive — the at-cast
            // bind is the live path for the modal admit, the
            // resolve-time scan stays the safety net for synthetic
            // direct-effect tests).
            let bound = state
                .stack
                .iter()
                .find(|i| i.source_obj_id == source_obj_id)
                .and_then(|i| i.target_obj_id);
            let target_id: Option<u32> = if let Some(bid) = bound {
                Some(bid)
            } else {
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        let printing = state.registry.lookup(obj.printing_id)?;
                        let is_eligible = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact" || t == "Creature");
                        if is_eligible {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible artifact/creature → silent fizzle
            };
            // Locate the target's home player (target may be on
            // either bf).
            let target_pidx_opt: Option<usize> = state
                .players
                .iter()
                .position(|p| p.battlefield.iter().any(|o| o.id == target_id));
            let Some(target_pidx) = target_pidx_opt else {
                return Ok(()); // target left bf between bind + resolve
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[target_pidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    if *power_bonus != 0 || *toughness_bonus != 0 {
                        obj.ueot_modifiers.push(UeotModifier::PowerToughnessBonus {
                            power: *power_bonus,
                            toughness: *toughness_bonus,
                        });
                    }
                    // Reroute Systems closure (2026-05-21): grant
                    // each keyword as its own `GrantKeyword` modifier
                    // (mirrors the `GrantKeywordsToTargetCreature
                    // UntilEOT` per-keyword fanout at
                    // `state_ops_resolve.rs:15695-15700`). All
                    // entries auto-expire at the next cleanup step
                    // via the r27 CR 514.3 UEOT-clear pass.
                    for kw in granted_keywords {
                        obj.ueot_modifiers.push(UeotModifier::GrantKeyword {
                            keyword: kw.clone(),
                        });
                    }
                    break;
                }
            }
        }
        Effect::DealDamageToTargetTappedCreature { amount } => {
            // Reroute Systems closure (2026-05-21). Mirrors Python
            // `DealDamageToTargetTappedCreature(amount=N)` (sugar on
            // `DealDamage` with filter `IsCreature() & IsTapped()`;
            // `effects/generic/damage.py:932-950`). The cast-time
            // bind on `StackItem.target_id` locks a specific target;
            // this arm re-verifies the target is STILL a tapped
            // creature on the battlefield per CR 608.2b and applies
            // the damage. A target that untapped or left the bf
            // between cast and resolve is silently skipped (mirrors
            // the unified Python `DealDamage.resolve` filter fizzle
            // at `damage.py:78-79`).
            if *amount <= 0 {
                return Ok(()); // CR 119.5: 0-or-less damage is a no-op
            }
            // Source-controller resolution (5-zone walk + stack
            // fallback). Mirrors the sibling pump arm above.
            let mut src_controller: Option<u8> = None;
            'outer_ddtt: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer_ddtt;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            if src_controller.is_none() {
                return Ok(()); // source untrackable → silent fizzle
            }
            // Prefer the at-cast bound target; fall back to the
            // deterministic-first tapped-creature scan over both
            // bfs (defensive — synthetic direct-effect tests skip
            // the cast path).
            let bound = state
                .stack
                .iter()
                .find(|i| i.source_obj_id == source_obj_id)
                .and_then(|i| i.target_obj_id);
            let target_id: Option<u32> = if let Some(bid) = bound {
                Some(bid)
            } else {
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        if !obj.tapped {
                            return None;
                        }
                        let printing = state.registry.lookup(obj.printing_id)?;
                        if printing.card_types.iter().any(|t| t == "Creature") {
                            Some(obj.id)
                        } else {
                            None
                        }
                    })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible tapped creature → silent fizzle
            };
            // At-resolve filter recheck per CR 608.2b: the target
            // must STILL be a tapped creature on the battlefield.
            let target_pidx_opt: Option<usize> = state
                .players
                .iter()
                .position(|p| p.battlefield.iter().any(|o| o.id == target_id));
            let Some(target_pidx) = target_pidx_opt else {
                return Ok(()); // target left bf between bind + resolve
            };
            {
                let bf = &state.players[target_pidx].battlefield;
                let Some(obj) = bf.iter().find(|o| o.id == target_id) else {
                    return Ok(());
                };
                if !obj.tapped {
                    return Ok(()); // untapped between cast + resolve
                }
                let is_creature = state
                    .registry
                    .lookup(obj.printing_id)
                    .map(|p| p.card_types.iter().any(|t| t == "Creature"))
                    .unwrap_or(false);
                if !is_creature {
                    return Ok(()); // no longer a creature
                }
            }
            // Apply damage to the target (CR 119.3: damage marked
            // on creature; SBAs handle the lethal check later). Use
            // saturating add into i16 mirror to match the
            // `deal_damage` PyMethod convention.
            let amount_i16 = (*amount).min(i16::MAX as i32) as i16;
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[target_pidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.damage = obj.damage.saturating_add(amount_i16);
                    break;
                }
            }
        }
        // Round-28 EachOpponentSacrificesCreature (3+ EoE cards).
        // Mirrors the round-24 EachOpponentDiscards source-finder
        // shape; on the resolution side, picks the (single 2-player)
        // opponent's first creature on battlefield and sends it to
        // their graveyard with CR 400.7 zone-change reset. No
        // eligible creature → silent fizzle.
        // Round-28 UntapTargetOtherPermanent (Nanoform-Sentinel
        // trigger family, 3+ EoE cards). Scans both players'
        // battlefields in zone-iteration order; picks the FIRST
        // tapped permanent whose id != source_obj_id and untaps it.
        // Mirrors Python's `[o for o in state.battlefield if
        // o.id != source_id and o.tapped]` filter + pick-first.
        //
        // No eligible permanent (everything untapped, or only the
        // source itself is tapped) → silent fizzle.
        // Round-28 SetBasePTUntilEOT (Genemorph-Imago family, 2+ EoE
        // cards). CR 613.4 layer 7b base-P/T override. Same target-
        // pick shape as PumpTargetCreatureUntilEOT but pushes
        // UeotModifier::SetBasePT instead of ::PowerToughnessBonus.
        // Composer reads the LAST SetBasePT modifier (last-applied
        // wins). Layer-7c additive bonuses stack on top.
        Effect::SetBasePTUntilEOT { power, toughness } => {
            // Source-controller resolution (5-zone walk).
            let mut src_controller: Option<u8> = None;
            'outer_sbp: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer_sbp;
                    }
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untrackable → silent fizzle
            };
            let cidx = controller as usize;
            // Pick first creature on controller's bf.
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.ueot_modifiers.push(UeotModifier::SetBasePT {
                        power: *power,
                        toughness: *toughness,
                    });
                    break;
                }
            }
        }
        // Round-28 ManifestDread first-slice. CR 701.55 — pop top
        // min(2, library_len) cards into a ManifestDreadChoice and
        // short-circuit. Substrate-aligned with LookTopN: same pop-
        // and-package shape, but the submit dispatcher routes the
        // picked card to bf (with ETB-reset) instead of hand.
        //
        // Controller convention: state.active_player (first-slice
        // shape consistent with Surveil / LookTopN / MayDo /
        // ChooseOneMode / IfYouDo).
        //
        // First-slice: count = 1 ONLY. count > 1 raises with a clear
        // "deferred" message so a typo doesn't silently no-op. Python's
        // multi-iter form (Slimy Aquarium, Weight Room riders) and X-
        // times form (Valgavoth's Onslaught) are documented in
        // SEARCH_LIMITATIONS.md as deferred.
        //
        // Empty / 1-card library partials: emit a PendingDecision
        // with a (possibly empty / 1-card) cards Vec. Auto-resolve
        // and submit handle both — empty drains as O(1) no-op,
        // 1-card manifests the lone card.
        Effect::ManifestDread { count } => {
            if *count != 1 {
                return Err(PyValueError::new_err(format!(
                    "ManifestDread: first-slice supports count = 1 only \
                     (got count = {count}); count > 1 (multi-iter) and \
                     count = 0 (no-op) deferred — see \
                     docs/SEARCH_LIMITATIONS.md"
                )));
            }
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let pidx = controller as usize;
            let lib_len = state.players[pidx].library.len();
            let pop_count = lib_len.min(2);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(pop_count);
            for _ in 0..pop_count {
                // Post-CE3: top = lib[0]. Remove from head.
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(
                PendingDecision::ManifestDreadChoice {
                    source_obj_id,
                    controller,
                    cards,
                    then_attach_to_source: false,
                    then_plus_one_counter_on_manifested: false,
                    counter_amount: 0,
                },
            );
        }
        // Round-29 ManifestDreadThenAttachSelf composition. Mirrors
        // Python's `ManifestDreadThenAttachSelf.resolve` (Conductive-
        // Machete-family Equipment ETB, freq 3). Same pop-and-package
        // shape as `Effect::ManifestDread`, but the pending carries
        // `then_attach_to_source: true` so the submit dispatcher runs
        // the attach as a follow-up.
        //
        // Controller convention DIFFERS from plain ManifestDread: for
        // ETB-resolution on an Equipment, the source is on the bf at
        // resolve time. Read the controller off the source rather than
        // off `state.active_player` so an Equipment-spell cast (or
        // ETB-triggered) by p1 during p0's turn manifests dread from
        // p1's library. Mirrors Python's `state.resolving_source`
        // access (resolving source's `.controller_id` carries the
        // source's controller).
        //
        // Silent fizzle when source isn't on a bf — mirrors Python's
        // `source is None: return` + `"Equipment" not in subtypes:
        // return` early-exit gates (same observable behavior as
        // "source unreachable").
        //
        // First-slice: count = 1 ONLY (same gate as ManifestDread).
        Effect::ManifestDreadThenAttachSelf { count } => {
            if *count != 1 {
                return Err(PyValueError::new_err(format!(
                    "ManifestDreadThenAttachSelf: first-slice supports count = 1 only \
                     (got count = {count}); count > 1 (multi-iter) and \
                     count = 0 (no-op) deferred — see \
                     docs/SEARCH_LIMITATIONS.md"
                )));
            }
            // Locate source on a battlefield → silent fizzle if absent.
            let mut src_loc: Option<(usize, u8)> = None; // (player_idx, controller)
            for (pidx, player) in state.players.iter().enumerate() {
                if let Some(obj) = player.battlefield.iter().find(|o| o.id == source_obj_id) {
                    src_loc = Some((pidx, obj.controller));
                    break;
                }
            }
            let Some((_pidx, controller)) = src_loc else {
                return Ok(()); // source not on bf → silent fizzle
            };
            let cidx = controller as usize;
            let lib_len = state.players[cidx].library.len();
            let pop_count = lib_len.min(2);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(pop_count);
            for _ in 0..pop_count {
                // Post-CE3: top = lib[0]. Remove from head.
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(
                PendingDecision::ManifestDreadChoice {
                    source_obj_id,
                    controller,
                    cards,
                    then_attach_to_source: true,
                    then_plus_one_counter_on_manifested: false,
                    counter_amount: 0,
                },
            );
        }
        // Round-29 ManifestDreadThenPutPlusOneCountersOnIt composition
        // (Moldering-Gym / Weight-Room Room-door family). Mirrors
        // Python's `ManifestDreadThenPutPlusOneCountersOnIt.resolve`
        // (`effects/generic/library_lookat.py:451` — Slimy Aquarium
        // counter_amount=1, Weight Room counter_amount=3). Same pop-
        // and-package shape as `Effect::ManifestDread`; the pending
        // carries `then_plus_one_counter_on_manifested: true` and the
        // chosen `counter_amount` so the submit dispatcher stamps the
        // counter on the manifested permanent as a follow-up.
        //
        // **Controller convention.** `state.active_player` — matches
        // plain `Effect::ManifestDread`. These are Room-door triggers
        // that resolve during the controller's turn (the unlocked-half
        // ETB fires on the active player's turn-cycle); active_player
        // IS the source's controller in every printed surface in the
        // pool today. Python reads `controller_id` from the trigger
        // call site, which is also active_player at resolution.
        //
        // **First-slice scope.** `count = 1` ONLY at the manifest
        // iteration level (same gate as `Effect::ManifestDread` — no
        // Python printing of this class emits count != 1). The
        // `counter_amount` rider accepts any non-negative integer
        // (Slimy Aquarium=1, Weight Room=3).
        //
        // **Empty / 1-card library partials.** Same drain shape as
        // plain ManifestDread: empty cards Vec → submit no-ops both
        // the manifest and the counter add; 1-card Vec → manifest the
        // lone card and stamp the counter on it.
        Effect::ManifestDreadThenPutPlusOneCountersOnIt {
            count,
            counter_amount,
        } => {
            if *count != 1 {
                return Err(PyValueError::new_err(format!(
                    "ManifestDreadThenPutPlusOneCountersOnIt: first-slice supports \
                     count = 1 only (got count = {count}); count > 1 (multi-iter) \
                     and count = 0 (no-op) deferred — see \
                     docs/SEARCH_LIMITATIONS.md"
                )));
            }
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let pidx = controller as usize;
            let lib_len = state.players[pidx].library.len();
            let pop_count = lib_len.min(2);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(pop_count);
            for _ in 0..pop_count {
                // Post-CE3: top = lib[0]. Remove from head.
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(
                PendingDecision::ManifestDreadChoice {
                    source_obj_id,
                    controller,
                    cards,
                    then_attach_to_source: false,
                    then_plus_one_counter_on_manifested: true,
                    counter_amount: *counter_amount,
                },
            );
        }
        // Round-28 LookAtTopNRevealOrCounter — Living-Phone / Pulsar-
        // Squadron-Ace family (freq 2 in EoE pool). Pop top
        // min(n, library_len) cards into a LookAtTopNRevealOrCounterChoice
        // and short-circuit. Submit dispatcher routes reveal-yes →
        // (chosen to hand, rest to library bottom); reveal-no → (all
        // to library bottom + optional counter-on-miss rider).
        //
        // Controller convention: state.active_player (first-slice
        // shape consistent with Surveil / LookTopN / MayDo / ChooseOneMode
        // / IfYouDo / ManifestDread).
        //
        // Empty / partial library: emit a PendingDecision with a
        // (possibly empty / partial) cards Vec. Auto-resolve and submit
        // handle both — empty drains as "no reveal possible" so the
        // counter-on-miss rider fires (matches Python: empty examined
        // → no reveal happened → counter rider fires).
        Effect::LookAtTopNRevealOrCounter {
            n,
            filter_name,
            max_power,
            counter_on_miss,
        } => {
            if *n < 0 {
                return Err(PyValueError::new_err(format!(
                    "LookAtTopNRevealOrCounter: n must be non-negative, got {n}"
                )));
            }
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            let pidx = controller as usize;
            let lib_len = state.players[pidx].library.len();
            let pop_count = (*n as usize).min(lib_len);
            let lib_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[pidx].library);
            let mut cards: Vec<GameObject> = Vec::with_capacity(pop_count);
            for _ in 0..pop_count {
                // Post-CE3: top = lib[0]. Remove from head.
                let removed = lib_mut.remove(0);
                cards.push(removed);
            }
            state.pending_decisions.push(
                PendingDecision::LookAtTopNRevealOrCounterChoice {
                    source_obj_id,
                    controller,
                    cards,
                    filter_name: filter_name.clone(),
                    max_power: *max_power,
                    counter_on_miss: *counter_on_miss,
                },
            );
        }
        // Round-28 BecomeLandCreatureUEOT (Blinkmoth Nexus / Lavaclaw
        // Reaches / Wandering Fumarole family, 13 EoE cards). CR 305.3
        // — animated land is BOTH a land and a creature UEOT. Same
        // "self" semantics as PumpSelfUntilEOT: the recipient is the
        // source GameObject (the activated-ability source on the
        // battlefield). If the source has left the battlefield between
        // activation and resolve, silent no-op. Pushes a
        // `UeotModifier::BecomesLandCreature` onto the source's
        // `ueot_modifiers`; effective-P/T readers honor the modifier
        // when present (base P/T from the modifier, layer 7c additive
        // bonuses stack on top). The modifier auto-expires at the next
        // cleanup step via the r27 CR 514.3 UEOT-clear pass, reverting
        // the land to its non-creature state.
        Effect::BecomeLandCreatureUEOT {
            power,
            toughness,
            granted_subtype,
        } => {
            // Walk both battlefields to find the source object.
            for pidx in 0..state.players.len() {
                if state.players[pidx]
                    .battlefield
                    .iter()
                    .any(|o| o.id == source_obj_id)
                {
                    let zone: &mut Vec<GameObject> =
                        Arc::make_mut(&mut state.players[pidx].battlefield);
                    for obj in zone.iter_mut() {
                        if obj.id == source_obj_id {
                            obj.ueot_modifiers.push(UeotModifier::BecomesLandCreature {
                                power: *power,
                                toughness: *toughness,
                                color_or_subtype: granted_subtype
                                    .clone()
                                    .unwrap_or_default(),
                            });
                            break;
                        }
                    }
                    break;
                }
            }
        }
        // Round-28 mass reanimation. Mirrors Python's
        // ReturnAllCreaturesFromYourGraveyardToBattlefield (Awakening-
        // Hall family, 2+ EoE cards). Iterate source-controller's
        // graveyard in stored order (oldest first — Vec front); move
        // every creature to the controller's bf with the standard ETB
        // reset (summoning_sick=true, tapped=false, etc.).
        Effect::ReturnAllCreaturesFromYourGraveyardToBattlefield => {
            let _ = source_obj_id; // controller-targeted, not source-relative
            let controller_idx = state.active_player as usize;
            // Collect creature ids first to avoid registry-vs-graveyard
            // borrow conflicts.
            let creature_ids: Vec<u32> = state.players[controller_idx]
                .graveyard
                .iter()
                .filter_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
                .collect();
            if creature_ids.is_empty() {
                return Ok(()); // empty / no-creature graveyard: silent no-op
            }
            // Drain matching creatures from graveyard; collect for bf.
            let graveyard_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].graveyard);
            let mut moved: Vec<GameObject> = Vec::with_capacity(creature_ids.len());
            graveyard_mut.retain(|o| {
                if creature_ids.contains(&o.id) {
                    let mut obj = o.clone();
                    obj.zone = ZONE_BATTLEFIELD;
                    obj.tapped = false;
                    obj.summoning_sick = true;
                    obj.damage = 0;
                    obj.counters = 0;
                    obj.attached_to = None;
                    obj.exiled_with_source = None;
                    moved.push(obj);
                    false
                } else {
                    true
                }
            });
            // CE4 closure: reserve a contiguous block of timestamps so
            // each retrieved creature gets a strictly-monotonic ETB
            // timestamp matching the in-effect resolution order. The
            // counter is bumped once per moved creature.
            let bf_etb_base = state.next_etb_timestamp;
            state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(moved.len() as u32);
            // ETB-trigger Slice 2 (RS6): collect each reanimated
            // creature's id for the post-push dispatch fan-out.
            let mut entered_ids: Vec<u32> = Vec::with_capacity(moved.len());
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[controller_idx].battlefield);
            for (i, mut obj) in moved.into_iter().enumerate() {
                obj.etb_timestamp = bf_etb_base.saturating_add(i as u32);
                entered_ids.push(obj.id);
                bf_mut.push(obj);
            }
            // ETB-trigger Slice 2 (RS6): queue ENTERS per
            // mass-reanimated creature in stamp order. Drained by
            // the surrounding resolver after the resolving spell's
            // stack item pops. Order matters — death/ETB chains can
            // ripple, but mirror Python's centralized list-walk by
            // emitting in the ETB-stamp order we just stamped above.
            state.pending_etb_dispatches.extend(entered_ids);
        }
        // Round-28 TapAnotherTargetCreature (Starport-Security family,
        // 2+ EoE cards). Same scan shape as UntapTargetOtherPermanent
        // but filters on Creature and taps. Already-tapped target is
        // a silent no-op (matches the existing TapTarget contract).
        //
        // CE7 closure: cross-player find-first picks the ETB-timestamp-
        // first eligible target via the shared `bf_iter_etb_sorted`
        // helper, matching Python's centralized `state.battlefield`
        // walk in `effects/generic/tap.py:277-295` (TapAnotherTargetCreature
        // resolves via the base Tap class's `legal_targets`, which walks
        // `state.battlefield` filtered by `IsCreature() & ~IsSource()`).
        // Pre-CE7 the Rust side walked p0-then-p1, which diverged from
        // Python whenever a p1 permanent had a lower ETB stamp than
        // any p0 eligible target.
        Effect::TapAnotherTargetCreature => {
            let mut found: Option<(usize, u32)> = None;
            for obj in crate::bf_iter::bf_iter_etb_sorted(state) {
                if obj.id == source_obj_id {
                    continue;
                }
                let printing_idx = obj.printing_id as usize;
                let Some(printing) = state.registry.printings.get(printing_idx) else {
                    continue;
                };
                if !printing.card_types.iter().any(|t| t == "Creature") {
                    continue;
                }
                found = Some((obj.controller as usize, obj.id));
                break;
            }
            if let Some((_pidx, target_id)) = found {
                // Round-58 BECOMES_TAPPED event bridge — "tap another
                // target creature" tapper-effect site. Python's
                // `effects/generic/tap.py:TapAnotherTargetCreature`
                // extends the base `Tap` class, whose `resolve`
                // routes the tap through `turn.tap_permanent` (the
                // docstring at tap.py:36 spells out "no-op if already
                // tapped"). Centralized chokepoint so the
                // no-op-if-already-tapped gate matches Python (an
                // already-tapped target is a silent no-op with NO
                // BECOMES_TAPPED event — byte-equal to Python's
                // early return).
                crate::state_ops_resolve::tap_permanent_and_dispatch(
                    state, target_id,
                )?;
            }
            // else: no eligible creature → silent fizzle.
        }
        Effect::UntapTargetOtherPermanent => {
            // Find first eligible (tapped, not source) across both
            // players' battlefields.
            //
            // CE7 closure: cross-player find-first picks the ETB-
            // timestamp-first eligible target via the shared
            // `bf_iter_etb_sorted` helper, matching Python's centralized
            // `state.battlefield` walk in
            // `effects/generic/tap.py:346-386` (`UntapTargetOtherPermanent.legal_targets`
            // returns `[("permanent", o.id) for o in state.battlefield
            // if o.id != source_id and o.tapped]`). Pre-CE7 the Rust
            // side walked p0-then-p1, which diverged from Python
            // whenever a p1 tapped permanent had a lower ETB stamp
            // than any p0 eligible target.
            let mut found: Option<(usize, u32)> = None;
            for obj in crate::bf_iter::bf_iter_etb_sorted(state) {
                if obj.id != source_obj_id && obj.tapped {
                    found = Some((obj.controller as usize, obj.id));
                    break;
                }
            }
            if let Some((pidx, target_id)) = found {
                let zone: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in zone.iter_mut() {
                    if obj.id == target_id {
                        obj.tapped = false;
                        break;
                    }
                }
            }
            // else: no eligible permanent → silent fizzle.
        }
        // Round-28 GrantKeywordToTargetCreatureUntilEOT
        // (GrantUnblockable family, 3+ EoE cards; generalized to any
        // keyword). Same shape as PumpTargetCreatureUntilEOT but
        // pushes UeotModifier::GrantKeyword rather than
        // ::PowerToughnessBonus. Modifier auto-expires at the next
        // cleanup step via the r27 CR 514.3 UEOT-clear pass.
        Effect::GrantKeywordToTargetCreatureUntilEOT { keyword } => {
            // Source-controller resolution (5-zone walk).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untrackable → silent fizzle
            };
            let cidx = controller as usize;
            // Pick first creature on controller's bf.
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    obj.ueot_modifiers.push(UeotModifier::GrantKeyword {
                        keyword: keyword.clone(),
                    });
                    break;
                }
            }
        }
        // Round-15 (post-r14) rider-widened sibling of
        // GrantKeywordToTargetCreatureUntilEOT. Same controller-of-effect
        // resolution + first-creature pick, but: (1) admits a LIST of
        // keywords (one UeotModifier::GrantKeyword push per entry); (2)
        // honors `exclude_source` to skip the resolving source itself in
        // the target-pick scan ("another target creature"). Empty
        // keywords list short-circuits to a silent no-op (parser also
        // rejects empty at register time so this is a defensive guard,
        // not a reachable runtime path from registered cards).
        //
        // **EoE consumer.** Blooming Stinger (1G Plant Scorpion) is the
        // unique EoE source for the `exclude_source=true` shape:
        // "When this creature enters, another target creature you
        // control gains deathtouch until end of turn." Forward-
        // compatible with future multi-keyword grants via the
        // `Vec<String>` field.
        Effect::GrantKeywordsToTargetCreatureUntilEOT { keywords, exclude_source } => {
            if keywords.is_empty() {
                return Ok(());
            }
            // Source-controller resolution (5-zone walk; identical to
            // the single-keyword sibling above).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            let Some(controller) = src_controller else {
                return Ok(()); // source untrackable → silent fizzle
            };
            let cidx = controller as usize;
            // Pick first creature on controller's bf, optionally
            // excluding the resolving source itself ("another target
            // creature" — Blooming Stinger's exclude_source=true).
            let target_id: Option<u32> = {
                let bf = &state.players[cidx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    if *exclude_source && obj.id == source_obj_id {
                        return None;
                    }
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            let zone: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[cidx].battlefield);
            for obj in zone.iter_mut() {
                if obj.id == target_id {
                    for kw in keywords {
                        obj.ueot_modifiers.push(UeotModifier::GrantKeyword {
                            keyword: kw.clone(),
                        });
                    }
                    break;
                }
            }
        }
        Effect::EachOpponentSacrificesCreature => {
            // Source-controller resolution (mirrors EachOpponentDiscards).
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            let opponent_idx = (1 - controller) as usize;
            // Pick first NONTOKEN creature on opponent's bf.
            //
            // **Round-55 shared-root closure.** The printed oracle
            // text on Susurian Dirgecraft (+ the other 3+ EoE
            // consumers of this arm) is "each opponent sacrifices a
            // NONTOKEN creature of their choice" — the nontoken
            // filter is part of the rules text. Python's
            // `EachOpponentSacrificesCreature.resolve`
            // (effects/generic/opponent_punishers.py:47-52) builds
            // its `legal` list filtering `not o.printing.is_token`
            // BEFORE the agent picks. Pre-round-55 Rust had no
            // `is_token` notion so this `find_map` matched the FIRST
            // creature regardless of token status — diverging from
            // the rules text (and Python) whenever a token sat at
            // bf-iter index 0 ahead of a nontoken. Round-55's
            // `RegistryData::is_token` (derived from the bridge-
            // round-tripped `supertypes` Vec) closes the SAME shared
            // root as the token-cease-to-exist SBA: gate the
            // predicate on `!is_token` so both engines pick the
            // first NONTOKEN creature identically. This is the
            // SEARCH_LIMITATIONS "EachOpponentSacrificesCreature
            // nontoken-filter shortcut" entry's fix-scope step 1.
            let target_id: Option<u32> = {
                let bf = &state.players[opponent_idx].battlefield;
                let registry = &state.registry;
                bf.iter().find_map(|obj| {
                    let printing = registry.printings.get(obj.printing_id as usize)?;
                    // Aura-attachment Phase 4 S3 closure (2026-05-21):
                    // consult the per-object override so a Moonlit-
                    // Meditation-replacement token is excluded from
                    // the "non-token creature" pool even though its
                    // printing_id points at the host's (non-Token)
                    // printing. Mirrors Python's
                    // `not obj.printing.is_token` read against the
                    // injected supertype tuple.
                    if printing.card_types.iter().any(|t| t == "Creature")
                        && !obj.effective_is_token(registry)
                    {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
            };
            let Some(target_id) = target_id else {
                return Ok(()); // no eligible creature → silent fizzle
            };
            // Pop from opponent's bf, route to opponent's graveyard
            // with CR 400.7 zone-change reset.
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut state.players[opponent_idx].battlefield);
            let pos = bf_mut
                .iter()
                .position(|o| o.id == target_id)
                .expect("target_id found above");
            let mut obj = bf_mut.remove(pos);
            obj.zone = ZONE_GRAVEYARD;
            obj.tapped = false;
            // CE13 closure: `summoning_sick` is NOT reset here.
            // Python's `turn.move_off_battlefield` deliberately
            // omits the clear (CR 302.1 — sickness is battlefield-
            // only state). See I35.
            obj.damage = 0;
            // R77-CFI1: counters are NOT cleared on the dead object.
            // A sacrifice routes through Python's
            // `turn.move_off_battlefield` (turn.py:538 docstring:
            // "sacrifice ... calls move_off_battlefield directly"),
            // which leaves `+1/+1` counters intact — the SAME
            // byte-equal contract the SBA destroy path / the closed
            // `DepressurizeEffect` arm follow. Prior stray
            // `obj.counters = 0;` removed (R77-CFI1; audited sibling
            // of the destroy arms, same class as the closed CFI-1).
            obj.attached_to = None;
            obj.exiled_with_source = None;
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[opponent_idx].graveyard).push(obj);
            // Round-29 Void LTB: each-opponent-sacs-creature — target
            // is a Creature by filter → nonland → flag flips.
            mark_left_battlefield_if_nonland(state, printing_id);
            // Slice 10: LTB fan-out at the Effect arm. The bf-exit
            // bypasses move_object, so the LEAVES dispatch needs to
            // be wired here (Slice 8 only covered move_object +
            // SBA destroy). Queue the leaving id; the drain runs in
            // `resolve_top_of_stack` AFTER the resolving slot is
            // popped — mirrors how pending_etb_dispatches defers
            // ENTERS dispatch past the spell's stack pop. Draining
            // here would push the LTB trigger ABOVE the still-
            // present resolving spell and the subsequent
            // `stack.pop()` would pop the wrong slot.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            // Round-55: token-cease sibling of the LEAVES push. The
            // id is queued unconditionally; the is_token gate lives
            // in drain_pending_token_cease (CR 111.7 / 704.5d).
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        Effect::EachOpponentMillsCards { count } => {
            if *count < 0 {
                return Err(PyValueError::new_err(format!(
                    "EachOpponentMillsCards count must be non-negative, got {count}"
                )));
            }
            if *count == 0 {
                return Ok(());
            }
            // Source-controller resolution (mirrors EachOpponentDiscards).
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // 2-player "each opponent" → "the one opponent". Mill
            // `count` cards off the opponent's library top (post-
            // CE3: top = lib[0]).
            let opponent_idx = (1 - controller) as usize;
            let n = *count as usize;
            for _ in 0..n {
                let lib_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[opponent_idx].library);
                if lib_mut.is_empty() {
                    break; // empty library: silent stop (CR parity with Effect::Mill)
                }
                let mut obj = lib_mut.remove(0);
                obj.zone = ZONE_GRAVEYARD;
                Arc::make_mut(&mut state.players[opponent_idx].graveyard).push(obj);
            }
        }
        Effect::ExileEachOpponentsGraveyard => {
            // Structural twin of EachOpponentMillsCards: same
            // source-controller finder (battlefield/hand/library/
            // graveyard/exile sweep, then stack fallback), then a
            // per-opponent loop. Mirrors Python's
            // `ExileEachOpponentsGraveyard.resolve` (exile.py:260):
            // every card in each opponent's graveyard moves to that
            // player's OWN exile zone (CR 406 — exile is a public
            // zone; the card stays owned by its owner; 400.7
            // zone-change reset). 2-player simplification:
            // opponent_idx = 1 - controller.
            let mut source_controller: Option<u8> = None;
            'outer_eeg: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer_eeg;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            let opponent_idx = (1 - controller) as usize;
            // Drain the opponent's whole graveyard into their own
            // exile zone. `mem::take` mirrors Python's
            // `list(p.graveyard)` snapshot-then-clear (the per-card
            // remove/append loop in Python is observationally
            // identical to a bulk move — both empty the graveyard and
            // append in graveyard order to exile).
            let drained: Vec<GameObject> = std::mem::take(
                Arc::make_mut(&mut state.players[opponent_idx].graveyard),
            );
            if !drained.is_empty() {
                let exile_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[opponent_idx].exile);
                for mut obj in drained {
                    obj.zone = ZONE_EXILE;
                    exile_mut.push(obj);
                }
            }
        }
        // Round-29 intervening-if keyed on EoE "Void" condition.
        // Mirrors Python's `ConditionalIfVoid.resolve` (combinators.py:
        // 1446): read `is_void_active(state)`; if true, apply
        // `then_effect`; else if `else_effect` is Some, apply it; else
        // no-op. Recurses via `apply_effect` with the same
        // source_obj_id so nested target-resolution / source-relative
        // lookups stay anchored to the outer spell.
        Effect::ConditionalIfVoid { then_effect, else_effect } => {
            let void_active = state.permanent_left_battlefield_this_turn
                || state.spell_warped_this_turn;
            let branch: Option<&Effect> = if void_active {
                Some(then_effect.as_ref())
            } else {
                else_effect.as_deref()
            };
            if let Some(inner) = branch {
                apply_effect(state, inner, source_obj_id)?;
            }
            // else: silent no-op (matches Python's None-branch fall-
            // through).
        }
        // DealDamageToOpponentCreature — Glassworks // Shattered Yard
        // family (2 EoE cards). Mirrors Python's
        // `DealDamageToOpponentCreature(DealDamage)` sugar with filter
        // `IsCreature() & ControlledByOpponent()`. Distinct from
        // `DealDamageAnyTarget` (which can hit a player) AND from
        // `DealDamage { target: AnyCreature }` (which picks bf[0] of
        // the active player's defender without a card-type filter).
        //
        // Source-controller direction: scan source GameObject across
        // the 5 zones + stack to recover `source.controller`. Opponent
        // pool = every player whose id != source_controller (in 2-player
        // engine = the one opponent).
        //
        // First-slice deterministic choice: iterate opponents in
        // ascending player order; pick the FIRST `Creature` printing on
        // each opponent's battlefield (smallest bf index). Return on
        // first match — silent fizzle if no opponent has a creature.
        // Source-controller's own creatures are excluded by construction
        // (we only scan the opponents).
        //
        // Negative `amount` raises at register time AND here as defense-
        // in-depth (mirrors the DealDamage / DealDamageAnyTarget
        // raise-on-negative shape).
        Effect::DealDamageToOpponentCreature { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "DealDamageToOpponentCreature amount must be non-negative, got {amount}"
                )));
            }
            // Source-controller resolution (5-zone walk + stack scan;
            // mirrors LoseLife / TargetOpponentLosesLife shape).
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found: silent fizzle
            };
            // First eligible target: scan opponents (players whose id
            // != controller) in ascending order; pick the first
            // Creature on each opponent's bf (smallest index). 2-player
            // engine: exactly one opponent at `1 - controller`, but
            // generalize via the iterator to keep the shape n-player-
            // ready.
            let mut found: Option<(usize, u32)> = None;
            'outer_dd: for (pidx, player) in state.players.iter().enumerate() {
                if pidx as u8 == controller {
                    continue;
                }
                for obj in player.battlefield.iter() {
                    let printing_idx = obj.printing_id as usize;
                    let Some(printing) = state.registry.printings.get(printing_idx) else {
                        continue;
                    };
                    if printing.card_types.iter().any(|t| t == "Creature") {
                        found = Some((pidx, obj.id));
                        break 'outer_dd;
                    }
                }
            }
            if let Some((pidx, target_id)) = found {
                let amount_i16 = (*amount).min(i16::MAX as i32) as i16;
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.damage = obj.damage.saturating_add(amount_i16);
                        break;
                    }
                }
            }
            // else: no opponent creature → silent fizzle.
        }
        // Round-29 OptionalManaCostThenEffect — Kavaron-Harrier /
        // Oreplate-Pangolin family. "You may pay {cost}. If you do,
        // [inner_effect]." Pushes a PendingDecision::OptionalManaCostChoice
        // carrying the mana cost and inner Effect on the heap, and
        // short-circuits. Controller drains via
        // submit_optional_mana_cost_decision(pay) (manual) or
        // auto_resolve_pending_decisions("default" | "optional_pay_yes"
        // | "optional_pay_no") (deterministic).
        //
        // Controller convention: state.active_player — first-slice
        // shape consistent with Surveil / MayDo / IfYouDo / etc.
        //
        // No cost / inner-effect evaluation happens here: the decision
        // is just packaged. Affordability is checked at submit time so
        // the controller's pool state is the one in effect WHEN the
        // decision is resolved (matching Python's
        // OptionalManaCostThenEffect.resolve, which inspects pool
        // state at resolve time).
        Effect::OptionalManaCostThenEffect { cost, inner_effect } => {
            // I61: prefer the transient source-controller swap.
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            state.pending_decisions.push(
                PendingDecision::OptionalManaCostChoice {
                    source_obj_id,
                    controller,
                    cost: cost.clone(),
                    inner_effect: inner_effect.clone(),
                },
            );
        }
        // Round-28 freq-2 closure — Starwinder / Fear-of-Failed-Tests.
        // "When a creature you control deals combat damage to a player,
        // you may draw that many cards." Reads
        // state.resolving_combat_damage_amount and draws that many
        // cards for the SOURCE's controller (zone-scan + stack-scan
        // fallback, same shape as DiscardSelf et al.).
        //
        // **Write site (I32).** The slot is written per-event inside
        // `state_ops_combat.rs::apply_damage_pass` — set to
        // `entry.amount` before per-entry mutations, reset to 0 after.
        // Mirrors Python's per-event set/reset boundary
        // (combat.py:618,623 and actions.py:2684,2699). The Effect
        // reads the slot at resolve time; the apply-loop write keeps
        // the value live for any trigger that fires during the
        // event's downstream effects. See RUST_PARITY_GAPS.md I32.
        //
        // amount <= 0 → silent no-op (matches Python).
        Effect::DrawCardsEqualToResolvingCombatDamage => {
            let amount = state.resolving_combat_damage_amount;
            if amount <= 0 {
                return Ok(());
            }
            // Locate source controller (zone-scan first, stack scan
            // fallback). Same pattern as DiscardSelf / EachOpponent*.
            let mut source_controller: Option<u8> = None;
            'outer: for player in state.players.iter() {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    for obj in zone.iter() {
                        if obj.id == source_obj_id {
                            source_controller = Some(obj.controller);
                            break 'outer;
                        }
                    }
                }
            }
            if source_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    source_controller = Some(item.controller);
                }
            }
            let Some(controller) = source_controller else {
                return Ok(()); // source not found anywhere: silent fizzle
            };
            let controller_idx = controller as usize;
            // Inline draw loop — same pattern as Effect::DrawCards.
            // CR 704.5b empty-library handling flips
            // attempted_draw_on_empty so the SBA pass converts it to
            // has_lost.
            let n = amount as usize;
            for _ in 0..n {
                let player = &mut state.players[controller_idx];
                let lib = Arc::make_mut(&mut player.library);
                if lib.is_empty() {
                    player.attempted_draw_on_empty = true;
                    break;
                }
                let mut obj = lib.remove(0);
                obj.zone = ZONE_HAND;
                Arc::make_mut(&mut player.hand).push(obj);
            }
        }
        // **G7 Class-C delayed-trigger port (round-78).** The
        // `WarpExileSelf` delayed triggered ability resolves: re-look
        // up the warp creature on the battlefield by `source_obj_id`
        // and move it to its owner's exile with the CR-400.7 reset.
        // If the creature is no longer on the battlefield (countered
        // / died / bounced / exiled between cast and end-step
        // resolve), this is a silent no-op per CR 603.6c — Python's
        // `WarpExileSelf.resolve` (`effects/generic/exile.py:150-172`)
        // does the same `if obj is None: return`.
        //
        // **Scan both players' battlefields** by id. The warp
        // creature's controller is the source's owner in the
        // first-slice; scanning both seats keeps the resolve robust
        // against future control-change effects between cast and
        // end-step.
        //
        // **Reset shape** mirrors Python's `WarpExileSelf.resolve`
        // body, which routes through `turn.move_off_battlefield`:
        // damage = 0, tapped = false, attached_to = None,
        // exiled_with_source = None. **Counters are NOT cleared**
        // (R77-CFI1 — Python's `move_off_battlefield` deliberately
        // clears combat marks but leaves `+1/+1` counters intact;
        // the SBA destroy / sacrifice / DepressurizeEffect arms all
        // follow this same byte-equal contract). The prior stray
        // `obj.counters = 0;` here diverged from the Python SoT for
        // a counter'd warp-cast creature (surfaced by S6 scale-verify
        // v2 arch=14/seed=4: Germinating Wurm carrying a +1/+1
        // counter into warp-exile — Python net=1, Rust net=0).
        // `summoning_sick` is NOT cleared (CE13). `warp_exile_pending`
        // is cleared (the cast-time marker has discharged its
        // obligation). `warped = true` if non-token — same gate as
        // the deterministic-sweep arm and Python's
        // `if not obj.printing.is_token: obj.warped = True`.
        Effect::WarpExileSelf => {
            let mut found_at: Option<(usize, usize)> = None;
            for (pidx, player) in state.players.iter().enumerate() {
                if let Some(idx) = player
                    .battlefield
                    .iter()
                    .position(|o| o.id == source_obj_id)
                {
                    found_at = Some((pidx, idx));
                    break;
                }
            }
            let Some((pidx, idx)) = found_at else {
                // CR 603.6c — the trigger fired but has no permanent
                // to act on; silent no-op.
                return Ok(());
            };
            let player = &mut state.players[pidx];
            let bf: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            let mut obj = bf.remove(idx);
            // CR 400.7 reset — mirrors Python's
            // `turn.move_off_battlefield` body (R77-CFI1: counters
            // NOT cleared).
            obj.zone = ZONE_EXILE;
            obj.tapped = false;
            obj.damage = 0;
            obj.attached_to = None;
            obj.exiled_with_source = None;
            obj.warp_exile_pending = false;
            // Python's `WarpExileSelf.resolve` sets `obj.warped = True`
            // only for non-tokens; Phase-1 warp cards are never tokens
            // but the gate keeps this forward-correct.
            // Aura-attachment Phase 4 S3 closure (2026-05-21): consult
            // the per-object override so a Moonlit-Meditation-
            // replacement token reaching a warp-exile path (a
            // pathological case — warp-cast creatures don't sit
            // attached to Moonlit Auras in EoE, but the gate stays
            // forward-correct) is still treated as a token here.
            let is_token = obj.effective_is_token(&state.registry);
            if !is_token {
                obj.warped = true;
            }
            let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);
            Arc::make_mut(&mut state.players[pidx].exile).push(obj);
            // Round-29 Void LTB tracking: exile is a battlefield-exit.
            mark_left_battlefield_if_nonland(state, printing_id);
            // LEAVES + token-cease dispatch, same pattern as
            // `Effect::ExileTarget` arm: queue, drain at the tail of
            // `resolve_top_of_stack`.
            state.pending_leaves_dispatches.push(leaving_obj_id);
            crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
        }
        // **BBB downstream complement — 7th Rust slice (GGG Pin 2).**
        // DefendingPlayerLosesHalfLifeRoundedUp: each opponent of the
        // source-controller loses ceil(their_life / 2) life.  Mirrors
        // Python's `DefendingPlayerLosesHalfLifeRoundedUp.resolve`
        // (`effects/generic/life.py:576-588`).
        //
        // Controller resolution: prefer
        // `state.current_resolve_source_controller` (set by the stack
        // resolver around each `apply_effect` call), falling back to
        // `state.active_player`. Same pattern as `Effect::GainLife`
        // and `Effect::DrawCards`.
        //
        // Formula: `loss = (p.life + 1) / 2` — integer division of
        // `(p.life + 1)` by 2, which is ceiling division for positive
        // life totals and correct for zero / negative (e.g. life=0 →
        // loss=0; life=-1 → loss=0). Matches Python's `(p.life + 1) // 2`.
        Effect::DefendingPlayerLosesHalfLifeRoundedUp => {
            let controller = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player);
            for p in state.players.iter_mut() {
                if p.id == controller || p.has_lost {
                    continue;
                }
                let loss = (p.life + 1) / 2;
                p.life -= loss;
                // No Rust log field; Python-side logging is done by
                // DefendingPlayerLosesHalfLifeRoundedUp.resolve.
            }
        }
        // **BBB downstream complement — 7th Rust slice (XX Pin 1).**
        // ConditionalIfYouGainedLifeAtLeast: check
        // `controller.life_gained_this_turn >= threshold`; if met,
        // apply `inner`. Mirrors Python's
        // `ConditionalIfYouGainedLifeAtLeast.resolve`
        // (`effects/combinators.py:985-996`).
        //
        // Controller resolution: prefer
        // `state.current_resolve_source_controller` (set by the stack
        // resolver); fall back to `state.active_player`. Same pattern
        // as `Effect::GainLife`.
        Effect::ConditionalIfYouGainedLifeAtLeast { threshold, inner } => {
            let controller = state
                .current_resolve_source_controller
                .map(|c| c as usize)
                .unwrap_or(state.active_player as usize);
            let gained = state.players[controller].life_gained_this_turn;
            if gained >= *threshold {
                apply_effect(state, inner, source_obj_id)?;
            }
            // else: silent no-op (mirrors Python's early return with log).
        }
        // **BBB downstream complement — 7th Rust slice (XX Pin 2).**
        // ConditionalIfSourcePowerAtLeast: read `source_obj_id`'s
        // effective power; if `>= threshold`, apply `effect`. Mirrors
        // Python's `ConditionalIfSourcePowerAtLeast.resolve`
        // (`effects/combinators.py:1084-1101`).
        //
        // Source-power read: use
        // `predicate_subject_effective_power(state, source_obj_id)`
        // which is already pub(crate) and handles UEOT + counter
        // folding. `None` → source absent from battlefield → no-op
        // (mirrors Python's `if source is None: return`). Power below
        // threshold → silent no-op.
        Effect::ConditionalIfSourcePowerAtLeast { threshold, effect } => {
            match crate::state_ops_combat::predicate_subject_effective_power(
                state,
                source_obj_id,
            ) {
                Some(power) if power >= *threshold => {
                    apply_effect(state, effect, source_obj_id)?;
                }
                _ => {
                    // Source absent or power below threshold: silent no-op.
                }
            }
        }
        // **EEE downstream complement — 8th Rust slice (JJJ Pin 1).**
        // DyadrineAttackEffect: Dyadrine, Synthesis Amalgam's attack trigger.
        //
        // Honest-stop sentinel — parse-only arm.  The resolve logic is
        // card-specific and complex (agent-interactive counter-removal from
        // two controller-owned creatures, conditional draw, Robot token
        // creation).  This arm is a deliberate no-op sentinel that keeps the
        // Rust parse+build path working (Dyadrine can now be loaded into a
        // registry and its triggered ability round-trips through
        // parse_triggered_ability) without attempting to replicate the full
        // Python resolve.  Full resolve parity is deferred to a follow-up
        // slice; see `Effect::DyadrineAttackEffect` doc in cards.rs.
        //
        // Mirrors Python's `DyadrineAttackEffect.resolve` (bespoke.py:997).
        // Python resolves: find 2 creatures with counters, optionally remove
        // a counter from each (agent-selected), draw a card, create a Robot
        // token.  The Rust arm silently no-ops — observable divergence in
        // games where Dyadrine attacks while two creatures with counters
        // exist.  Acceptable for the parse-unlock goal of this slice.
        Effect::DyadrineAttackEffect => {
            // Honest-stop sentinel: no-op for now.
            // Full resolve (counter-removal + draw + Robot token) is
            // deferred to a follow-up slice.  Resolve is a silent no-op
            // so the parse-layer unblock is verifiable without behavioral
            // parity noise.
            let _ = state; // suppress unused-state warning in this arm.
        }
        // **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
        // PutPoisonCountersOnTargetPlayer: give `amount` poison counters
        // to the target player. Mirrors Python's
        // `PutPoisonCountersOnTargetPlayer.resolve`
        // (`effects/generic/counters.py:111-128`).
        //
        // Target resolution: reads `state.current_resolve_payload_target_id`
        // — the player-id payload bound at trigger-fire time by the
        // DEALS_COMBAT_DAMAGE event firing path.  Falls back to the
        // stack-item's built-in target if the payload slot is absent, then
        // silently no-ops if no target is identifiable.
        //
        // Python target shape: `targets[0] = ("player", pid)`.  The
        // DEALS_COMBAT_DAMAGE event dispatcher seats the damaged player's id
        // as the trigger subject via the standard (`"obj"`, `"target"`)
        // subject-slot convention; the bridge passes it through as the
        // `target_player_id` payload.
        //
        // SBA: after incrementing `poison_counters`, the game's
        // `apply_sbas` is responsible for checking the 10-poison-counter
        // loss condition (CR 704.5c) — the resolve arm does NOT call
        // `apply_sbas` itself (matches Python's resolve which only
        // increments the counter; SBAs fire at the next check point).
        Effect::PutPoisonCountersOnTargetPlayer { amount } => {
            // The target player id was seated into TriggerPayload.target_id
            // at push time (via `first_legal_trigger_target` returning
            // `Some(player_id as u32)` when
            // `state.resolving_combat_damage_target_player_id != u8::MAX`).
            // At resolve time, `resolve_top_of_stack` swaps that value into
            // `state.current_resolve_payload_target_id` before calling
            // `apply_effect`, so we read it here.
            //
            // Guard: `None` → no target was seated (synthetic-test push
            // outside combat damage; `first_legal_trigger_target` returned
            // `None`) → silent no-op (mirrors Python's `if not targets: return`
            // at counters.py:115).
            //
            // Player ids (0 or 1) are seated directly as `u32` values;
            // interpret as player id by looking up `p.id == target_u8`.
            if let Some(target_u32) = state.current_resolve_payload_target_id {
                let target_u8 = target_u32 as u8;
                // Find the player by id and increment poison_counters.
                if let Some(p) = state.players.iter_mut().find(|p| p.id == target_u8) {
                    p.poison_counters += *amount;
                    // Python logs "P{pid} gets {amount} poison counter(s)
                    // (now {p.poison_counters})"; Rust resolve has no log slot.
                }
            }
            // Else: no target seated (synthetic path) — silent no-op.
        }
        // **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for
        // all-untargeted modal cards, 2026-05-21).** Mirrors Python's
        // `ExileAllArtifactsWithManaValueAtMost.resolve` at
        // `effects/generic/exile.py:482-494`. Snapshot every
        // battlefield object whose printing has Artifact in
        // `card_types` AND whose total mana cost is `<= max_mv`, then
        // exile each via `move_off_battlefield_to_zone(... "exile")`
        // (the same off-battlefield helper sister `DestroyAllCreatures`
        // uses). Tokens cease to exist on exile per CR 704.5d
        // (handled by the move-off helper). Snapshot semantics avoid
        // mid-pass mutation (Python's `victims = [...]` then iterate).
        //
        // **Negative `max_mv` raises** (defense-in-depth — parse arm
        // already rejects but resolve also enforces; mirrors the
        // sibling `DamageEachOpponent` / `DamageEachCreature`
        // raise-on-negative shape). `max_mv=0` is a legitimate
        // restrictive value (only MV-0 artifacts hit — Lander tokens
        // hit; mana-cost-bearing artifacts safe).
        //
        // **Mana-cost computation.** The total mana cost is sum of
        // generic + colored + hybrid (the same metric Python's
        // `mana_cost.total` exposes — `costs.py:ManaCost.total`).
        // Walks each player's bf to collect candidates in seat-order
        // then bf-Vec order (the canonical iteration order; aligns
        // with Python's `state.battlefield` order via the cross-engine
        // harness's lockstep registry).
        Effect::ExileAllArtifactsWithManaValueAtMost { max_mv } => {
            if *max_mv < 0 {
                return Err(PyValueError::new_err(format!(
                    "ExileAllArtifactsWithManaValueAtMost max_mv must be non-negative, got {max_mv}"
                )));
            }
            let max_mv_local = *max_mv;
            let _ = source_obj_id; // implicit-target AOE; source not relevant
            // Snapshot (controller_idx, obj_id) for every artifact on
            // either battlefield in bf-ETB-timestamp order whose total
            // mana cost is `<= max_mv`. Mirrors the
            // `ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves` MV
            // computation at line 13149 (sum of generic + W + U + B
            // + R + G — the static printing cost; X is 0 in zones
            // outside the stack per CR 202.3b).
            let victims: Vec<(usize, u32)> = crate::bf_iter::bf_iter_etb_sorted(state)
                .into_iter()
                .filter_map(|obj| {
                    let printing = state.registry.lookup(obj.printing_id)?;
                    let is_artifact = printing
                        .card_types
                        .iter()
                        .any(|t| t == "Artifact");
                    if !is_artifact {
                        return None;
                    }
                    let mc = &printing.mana_cost;
                    let total = mc.generic as i32
                        + mc.white as i32
                        + mc.blue as i32
                        + mc.black as i32
                        + mc.red as i32
                        + mc.green as i32;
                    if total > max_mv_local {
                        return None;
                    }
                    Some((obj.controller as usize, obj.id))
                })
                .collect();
            // Move each victim off-battlefield to its owner's exile
            // zone. Mirrors the inline exile pattern at
            // `Effect::ExileTarget { AnyCreature }` (line 2755-2776):
            // pop from owner's bf, stamp ZONE_EXILE + reset
            // per-object state (tapped, damage, counters,
            // attached_to, exiled_with_source), push onto owner's
            // exile, mark left-bf if nonland, queue
            // pending_leaves_dispatches for LTB fan-out, queue
            // token-cease for CR 704.5d. The mass-shape mirrors
            // `Effect::DestroyAllCreatures` (line 6594) but routes
            // to ZONE_EXILE instead of ZONE_GRAVEYARD.
            for (pidx, victim_id) in victims {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                let pos = match bf_mut.iter().position(|o| o.id == victim_id) {
                    Some(p) => p,
                    None => continue, // defensively skip — should never miss
                };
                let mut obj = bf_mut.remove(pos);
                obj.zone = ZONE_EXILE;
                obj.tapped = false;
                // Per the `ExileTarget` arm: summoning_sick is NOT
                // reset (CR 302.1 — sickness is battlefield-only
                // state; see I35).
                obj.damage = 0;
                obj.counters = 0;
                obj.attached_to = None;
                obj.exiled_with_source = None;
                let leaving_printing_id = obj.printing_id;
                let leaving_obj_id = obj.id;
                Arc::make_mut(&mut state.players[pidx].exile).push(obj);
                // Void / LTB fan-out (CE13 / I29 / Slice-10).
                mark_left_battlefield_if_nonland(state, leaving_printing_id);
                state.pending_leaves_dispatches.push(leaving_obj_id);
                // Token-cease queue (CR 704.5d / 111.7).
                crate::state_ops_zones::queue_token_cease(state, leaving_obj_id);
            }
        }
        // **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
        // 2026-05-21).** Put `amount` charge counters on the
        // cast-time-bound target IF the target is still a Spacecraft
        // OR Planet permanent AND is still controlled by the source-
        // controller at resolve time (CR 608.2b re-check). Mirrors
        // Python's
        // `PutChargeCountersOnTargetSpacecraftOrPlanetYouControl
        // .resolve` at `effects/generic/counters.py:518-538`.
        //
        // **Source-controller lookup** (the "you control" filter
        // target). Look up by `source_obj_id` across all zones; fall
        // back to the active player on a synthetic-test path with no
        // source. Mirrors the `CreateTokenCopyOfTargetCreatureYouControl`
        // arm's source-controller scan (state_ops_resolve.rs:12188-
        // 12217).
        //
        // **Re-targeting at resolve (CR 608.2b).** Read the bound
        // target via `state.current_resolve_payload_target_id` (the
        // proven `SeedshipImpactEffect` / `DestroyTargetArtifactOr
        // TappedCreature` payload-target swap pattern). If the bound
        // id is no longer on either battlefield (target left the bf
        // between cast and resolve), silently fizzle — Python's
        // `obj is None` early return at `counters.py:526-531`. If
        // the target is on a battlefield but no longer qualifies
        // (subtype changed / control changed), also fizzle.
        //
        // **None-payload synthetic-test fallback.** The cross-engine
        // real-EoE probe + the spell-card-bridge use
        // `cast_spell_noncreature` which binds the target at cast
        // time (the per-mode detector table extension below). The
        // low-level synthetic `push_to_stack` API does NOT seat a
        // target. For that path — and ONLY that path — fall back to
        // the deterministic-first Spacecraft-or-Planet permanent the
        // source-controller controls (mirrors the
        // DestroyTargetArtifactOrTappedCreature precedent's
        // None-payload synthetic fallback).
        Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { amount } => {
            if *amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "PutChargeCountersOnTargetSpacecraftOrPlanetYouControl amount must be non-negative, got {amount}"
                )));
            }
            if *amount == 0 {
                return Ok(()); // legitimate no-op (mirrors PutChargeCountersOnSelf)
            }
            // Locate source controller across all zones (mirrors
            // CreateTokenCopyOfTargetCreatureYouControl source scan).
            let mut src_controller: Option<u8> = None;
            'outer: for player in &state.players {
                for zone in [
                    &player.hand,
                    &player.library,
                    &player.battlefield,
                    &player.graveyard,
                    &player.exile,
                ] {
                    if let Some(obj) = zone.iter().find(|o| o.id == source_obj_id) {
                        src_controller = Some(obj.controller);
                        break 'outer;
                    }
                }
            }
            if src_controller.is_none() {
                if let Some(item) = state
                    .stack
                    .iter()
                    .find(|i| i.source_obj_id == source_obj_id)
                {
                    src_controller = Some(item.controller);
                }
            }
            let controller_id: u8 = match src_controller {
                Some(c) => c,
                None => state.active_player,
            };
            let payload_target_opt = state.current_resolve_payload_target_id;
            // Helper: object is a qualifying Spacecraft-or-Planet
            // permanent the source-controller controls.
            let qualifies = |obj: &GameObject| -> bool {
                if obj.controller != controller_id {
                    return false;
                }
                let Some(printing) = state.registry.lookup(obj.printing_id) else {
                    return false;
                };
                let is_spacecraft = printing
                    .subtypes
                    .iter()
                    .any(|s| s == "Spacecraft");
                let is_planet = printing
                    .subtypes
                    .iter()
                    .any(|s| s == "Planet");
                is_spacecraft || is_planet
            };
            // Resolve-time re-legality: locate target. With a bound
            // payload, require that exact id on a battlefield and that
            // it still qualifies; otherwise fizzle (CR 608.2b). Without
            // a payload, fall back to deterministic-first
            // (synthetic-test path).
            let target: Option<(usize, u32)> = if let Some(pt) = payload_target_opt {
                let mut found: Option<(usize, u32)> = None;
                for (pidx, player) in state.players.iter().enumerate() {
                    if let Some(obj) = player.battlefield.iter().find(|o| o.id == pt) {
                        if qualifies(obj) {
                            found = Some((pidx, obj.id));
                        }
                        break;
                    }
                }
                found
            } else {
                crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .find_map(|obj| {
                        if !qualifies(obj) {
                            return None;
                        }
                        // Locate the obj's player index for the mut path.
                        for (pidx, player) in state.players.iter().enumerate() {
                            if player.battlefield.iter().any(|o| o.id == obj.id) {
                                return Some((pidx, obj.id));
                            }
                        }
                        None
                    })
            };
            if let Some((pidx, target_id)) = target {
                let bf_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[pidx].battlefield);
                for obj in bf_mut.iter_mut() {
                    if obj.id == target_id {
                        obj.charge_counters =
                            obj.charge_counters.saturating_add(*amount);
                        break;
                    }
                }
            }
        }
        // **Multi-target admit-gate Phase 4 widen (2026-05-21).** Scout
        // for Survivors: return up to `max_n` creature cards from the
        // source-controller's graveyard whose summed mana value is
        // <= `max_total_mv`, each entering the battlefield with a
        // +1/+1 counter. Substrate-only resolve-time self-scan: front-
        // to-back graveyard order, per-card MV gate + running sum-MV
        // budget. Mirrors Python's `ReturnFromGraveyard.resolve` at
        // `effects/generic/graveyard.py:195-242` for `max_n > 1`. See
        // `docs/SEARCH_LIMITATIONS.md` entry "Multi-target Phase 4
        // resolve-time self-scan" for the deliberate divergence vs
        // cast-time binding (cross-engine harness uses
        // `_PickFirstLegalTargetAgent` which coincides with self-scan).
        Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters {
            max_n,
            max_total_mv,
        } => {
            if *max_n < 0 {
                return Err(PyValueError::new_err(format!(
                    "ReturnUpToNSmallCreaturesFromGraveyardWithCounters max_n must be non-negative, got {max_n}"
                )));
            }
            if *max_total_mv < 0 {
                return Err(PyValueError::new_err(format!(
                    "ReturnUpToNSmallCreaturesFromGraveyardWithCounters max_total_mv must be non-negative, got {max_total_mv}"
                )));
            }
            if *max_n == 0 {
                return Ok(()); // legitimate no-op
            }
            let controller_idx = state
                .current_resolve_source_controller
                .unwrap_or(state.active_player) as usize;
            let max_n_u = *max_n as usize;
            let max_total_mv_local = *max_total_mv;
            // Compute total mana value of a printing (generic +
            // colored). Mirrors the
            // `ExileAllArtifactsWithManaValueAtMost` MV computation
            // pattern at line 16639-16644 above.
            let mv_of = |obj: &GameObject| -> i32 {
                match state.registry.lookup(obj.printing_id) {
                    Some(printing) => {
                        let mc = &printing.mana_cost;
                        mc.generic as i32
                            + mc.white as i32
                            + mc.blue as i32
                            + mc.black as i32
                            + mc.red as i32
                            + mc.green as i32
                    }
                    None => 0,
                }
            };
            // Collect winners by walking the graveyard front-to-back,
            // accepting creature cards whose individual MV fits AND
            // whose addition keeps the running sum within budget.
            // Front = oldest (smallest Vec index) — the same iteration
            // order `ReturnCreatureFromYourGraveyardToBattlefield` uses
            // for its single-target scan. Python's
            // `_PickFirstLegalTargetAgent` picks pool[0] then
            // pool[0]-of-remaining etc., which collapses to "first-N
            // in legal_targets order" — and `ReturnFromGraveyard.
            // legal_targets` walks `p.graveyard` in Vec order — so the
            // self-scan matches Python byte-equal under the
            // cross-engine harness.
            let mut winners: Vec<u32> = Vec::new();
            let mut running_mv: i32 = 0;
            for obj in state.players[controller_idx].graveyard.iter() {
                if winners.len() >= max_n_u {
                    break;
                }
                let Some(printing) = state.registry.lookup(obj.printing_id) else {
                    continue;
                };
                if !printing.card_types.iter().any(|t| t == "Creature") {
                    continue;
                }
                let card_mv = mv_of(obj);
                if card_mv > max_total_mv_local {
                    continue; // per-card cap (Python's `max_mv` rider)
                }
                if running_mv.saturating_add(card_mv) > max_total_mv_local {
                    continue; // would bust the running budget
                }
                running_mv = running_mv.saturating_add(card_mv);
                winners.push(obj.id);
            }
            if winners.is_empty() {
                return Ok(()); // silent fizzle
            }
            // Move each winner from graveyard → battlefield with
            // summoning_sick=true, fresh etb_timestamp, +1/+1 counter.
            // Mirrors the
            // `ReturnCreatureFromYourGraveyardToBattlefield` arm at
            // line 5567-5658 for per-card move semantics, repeated per
            // winner. ETB triggers fan out via pending_etb_dispatches.
            for target_id in winners {
                let etb_ts = crate::state::reserve_etb_timestamp(state);
                // No walker reseed scan — this variant filters on
                // Creature, not Planeswalker; loyalty_counters stays 0.
                let graveyard_mut: &mut Vec<GameObject> =
                    Arc::make_mut(&mut state.players[controller_idx].graveyard);
                let pos = match graveyard_mut.iter().position(|o| o.id == target_id) {
                    Some(p) => p,
                    None => continue, // defensively skip if vanished
                };
                let mut obj = graveyard_mut.remove(pos);
                obj.zone = ZONE_BATTLEFIELD;
                obj.tapped = false;
                obj.summoning_sick = true;
                obj.damage = 0;
                // Apply the +1/+1 counter rider. Mirrors Python's
                // `add_plus_one_counters(state, card, 1)` post-move
                // call at `effects/generic/graveyard.py:283`.
                obj.counters = obj.counters.saturating_add(1);
                obj.attached_to = None;
                obj.exiled_with_source = None;
                obj.etb_timestamp = etb_ts;
                // I83 controller-of-effect: stamp to the reanimation
                // effect's controller (same as the single-target
                // sibling arm above).
                obj.controller = controller_idx as u8;
                let entered_id = obj.id;
                Arc::make_mut(&mut state.players[controller_idx].battlefield).push(obj);
                state.pending_etb_dispatches.push(entered_id);
            }
        }
    }
    Ok(())
}

#[pymethods]
impl GameState {
    /// Round-28 "until source leaves" — return every exiled object
    /// whose `exiled_with_source` matches `source_id` to its owner's
    /// battlefield. CR 400.7 zone-change reset on the way back
    /// (tapped, summoning_sick=true per r28 CE1, damage, counters,
    /// attached_to, exiled_with_source all cleared).
    ///
    /// Manual hook: callers (harness / future leaves-bf primitive
    /// extension) must invoke this when the source leaves the
    /// battlefield. Auto-firing on every bf-leaving zone change is a
    /// follow-up slice. Returns the count of objects returned.
    fn return_paired_exiles_for(&mut self, source_id: u32) -> PyResult<i32> {
        let mut returned = 0i32;
        // ETB-trigger Slice 2 (RS7): collect each recalled
        // permanent's id across both players so we can drain ENTERS
        // dispatches after all bf-pushes complete.
        let mut entered_ids: Vec<u32> = Vec::new();
        for pidx in 0..self.players.len() {
            // Collect ids first to avoid borrow conflicts.
            let to_return: Vec<u32> = self.players[pidx]
                .exile
                .iter()
                .filter_map(|o| {
                    if o.exiled_with_source == Some(source_id) {
                        Some(o.id)
                    } else {
                        None
                    }
                })
                .collect();
            if to_return.is_empty() {
                continue;
            }
            let exile_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut self.players[pidx].exile);
            let mut popped: Vec<GameObject> = Vec::with_capacity(to_return.len());
            exile_mut.retain(|o| {
                if to_return.contains(&o.id) {
                    let mut obj = o.clone();
                    obj.zone = ZONE_BATTLEFIELD;
                    obj.tapped = false;
                    obj.summoning_sick = true; // r28 CE1 default; also CR 302.1 for ETB
                    obj.damage = 0;
                    obj.counters = 0;
                    obj.attached_to = None;
                    obj.exiled_with_source = None;
                    popped.push(obj);
                    false
                } else {
                    true
                }
            });
            // CE4 closure: reserve a contiguous block of timestamps
            // so each returning permanent gets a strictly-monotonic
            // ETB timestamp matching the in-loop drain order.
            let bf_etb_base = self.next_etb_timestamp;
            self.next_etb_timestamp = self.next_etb_timestamp.saturating_add(popped.len() as u32);
            let bf_mut: &mut Vec<GameObject> =
                Arc::make_mut(&mut self.players[pidx].battlefield);
            for (i, mut obj) in popped.into_iter().enumerate() {
                obj.etb_timestamp = bf_etb_base.saturating_add(i as u32);
                entered_ids.push(obj.id);
                bf_mut.push(obj);
                returned += 1;
            }
        }
        // ETB-trigger Slice 2 (RS7): queue ENTERS per recalled
        // permanent + drain. Mirrors Python's `put_into_battlefield`
        // per recalled object inside `return_paired_exiles_for`. Uses
        // the deferred-dispatch pattern (push to queue then drain) so
        // the call site is consistent with RS2-RS6's resolver-driven
        // drain shape and re-entrant ETB chains land correctly.
        self.pending_etb_dispatches.extend(entered_ids);
        drain_pending_etb_dispatches(self)?;
        Ok(returned)
    }

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

        // ETB-trigger Slice 1: triggered-ability resolution path.
        // A `StackItem::TriggeredAbility` carries its effect inline
        // in `trigger_payload.effect`; the resolver fires that effect
        // via `apply_effect` and skips `move_source_post_resolve`
        // (triggers aren't permanents — their source already lives
        // wherever the trigger fired from).
        if top.kind == StackItemKind::TriggeredAbility {
            let payload = top
                .trigger_payload
                .as_ref()
                .expect("StackItemKind::TriggeredAbility must carry trigger_payload");
            let trigger_effect = payload.effect.clone();
            // CE20 follow-up Slice 4: swap in payload target_id so
            // targeted-activation effect arms (today only
            // `Effect::EquipToTargetCreatureYouControl`) can read
            // the activator's chosen target. Restored after the
            // apply so no observable state leaks across resolution
            // boundaries. Mirrors the `state.active_player` swap
            // pattern at state_ops_actions.rs:725 (Slice 1 bridge
            // follow-up).
            let saved_target = self.current_resolve_payload_target_id;
            self.current_resolve_payload_target_id = payload.target_id;
            // DrawCards controller-of-effect fix: also swap in the
            // trigger's controller so `Effect::DrawCards` (and any
            // future arm that opts into the source-controller
            // convention) draws into the trigger's controller's
            // hand, NOT `state.active_player`'s hand. Restored
            // after apply.
            let saved_ctrl = self.current_resolve_source_controller;
            self.current_resolve_source_controller = Some(top.controller);
            let apply_result = apply_effect(self, &trigger_effect, top.source_obj_id);
            self.current_resolve_source_controller = saved_ctrl;
            self.current_resolve_payload_target_id = saved_target;
            apply_result?;
            let popped = self
                .stack
                .pop()
                .expect("peeked top but pop returned None — stack invariant broken");
            // Slice 2 — drain any ETB dispatches the trigger's
            // effect deferred. A trigger that lands a permanent on
            // bf (e.g. a token-creating ETB trigger) populates the
            // queue; we fan out the events now that the resolving
            // trigger slot has been popped.
            drain_pending_etb_dispatches(self)?;
            // R74-2 — drain any DIES dispatches a trigger's effect
            // may have queued (a trigger whose Effect destroys a
            // creature via the DepressurizeEffect hand-rolled arm).
            // Drained BEFORE LEAVES so the stack-push order is DIES
            // then LEAVES (LIFO -> LEAVES resolves first), byte-equal
            // to the SBA destroy path (`state_ops_sba.rs` :497/:503)
            // and Python's `move_off_battlefield` (turn.py:253-289).
            // Outside `apply_sbas` the only producer of this queue is
            // the DepressurizeEffect arm, so this drain is a no-op
            // (empty queue) for every pre-R74-2 trigger resolution.
            drain_pending_dies_dispatches(self)?;
            // Slice 10 — drain any LEAVES dispatches a trigger's
            // effect may have queued (e.g. a trigger that destroys
            // / bounces / exiles another bf permanent via an Effect
            // arm in state_ops_resolve.rs). Drain happens AFTER the
            // trigger slot is popped so the LEAVES trigger goes
            // onto the now-empty top, mirroring the spell-path's
            // pop-then-fan-out ordering.
            drain_pending_leaves_dispatches(self)?;
            // Slice 14 (TTT Pin 1) — drain any IS_SACRIFICED
            // dispatches a trigger's effect deferred via the
            // `Effect::Sacrifice` arm. Same defer-after-pop
            // discipline as the LEAVES sibling. No-op (empty
            // queue) for every pre-Slice-14 trigger resolution.
            drain_pending_is_sacrificed_dispatches(self)?;
            // Auto-resolve any pending decisions the trigger effect
            // may have queued (Surveil on ETB, etc.). Matches the
            // spell-path's terminal call below.
            auto_resolve_pending(self, "keep_all_on_top")?;
            return Ok(Some(stack_item_to_dict(py, &popped)?));
        }

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

        // DrawCards controller-of-effect fix: swap in the resolving
        // spell's controller for the duration of the effect-fan-out
        // loop so `Effect::DrawCards` (and any future arm reading
        // `current_resolve_source_controller`) routes to the
        // spell's controller, not `state.active_player`. Mirrors the
        // triggered-ability branch above. Restored after the loop
        // (also restored on early-return via the `?` operator by
        // dropping out of the function — but since we don't have a
        // RAII guard here, the swap is bracketed inline; any
        // `apply_effect` error propagates as `?` and the inline
        // restore line is skipped, but since the field default is
        // `None` and the resolver is the outermost frame, the leak
        // is bounded to "next call to apply_effect sees Some(this
        // controller) instead of None" — semantically a no-op for
        // every back-compat caller because DrawCards is the only
        // reader and any DrawCards in the error path would have
        // produced wrong output ANYWAY pre-fix. Bracketed restore
        // below covers the success path.
        let saved_ctrl = self.current_resolve_source_controller;
        self.current_resolve_source_controller = Some(top.controller);
        // Spell-card-bridge lever (round-70): swap in the spell's
        // deterministic-first bound target so any per-card bespoke
        // Effect resolve arm that reads the target via the SAME
        // `current_resolve_payload_target_id` swap the
        // `TriggeredAbility` path uses (above; today only
        // `EquipToTargetCreatureYouControl`) resolves byte-equal
        // whether driven by a trigger OR a spell cast. `None` for
        // every pre-lever spell push (back-compat — the field
        // defaults `None`; the closed Tier-A spell arms self-scan the
        // battlefield and ignore this, so the swap is a correct no-op
        // for them). Save → set → restore bracketing the apply
        // closure exactly as `saved_ctrl` does, so no observable
        // state leaks across resolution boundaries (pin #4 / #6).
        let saved_target = self.current_resolve_payload_target_id;
        self.current_resolve_payload_target_id = top.target_obj_id;
        // Spell-card-bridge lever Phase 2 (round-76): swap in the
        // SECOND bound target for a 2-slot spell (Diplomatic
        // Relations). Identical save → set → restore bracketing as
        // slot 0; `None` for every single-target / pre-Phase-2 push
        // (back-compat — the closed Phase-1 single-target arms
        // self-scan and ignore BOTH payload slots, so the swap is a
        // correct no-op for them). Mirrored into BOTH resolve entry
        // points (the §3.4 sibling-parity requirement; pin #4 / #6).
        let saved_target_2 = self.current_resolve_payload_target_id_2;
        self.current_resolve_payload_target_id_2 = top.target_obj_id_2;
        // ModalChooseOne envelope — Cycle 2 (cast-time-bound mode
        // dispatch). Swap in the StackItem's cast-time-bound mode
        // index so the `Effect::ChooseOneMode` arm reads the
        // pre-picked choice instead of pushing a resolve-time
        // `PendingDecision::ChooseModeChoice`. Mirrors Python's
        // `state.resolving_modes = item.chosen_modes` swap at
        // `actions.py:2712`. Save → set → restore bracketed exactly
        // like its payload-target sibling. `None` for every non-
        // modal push (back-compat — every closed Phase-1 / Phase-2
        // single-target / 2-slot arm + every trigger / activation
        // never sets it, so the swap is a correct no-op for them).
        let saved_mode_idx = self.current_resolve_chosen_mode_idx;
        self.current_resolve_chosen_mode_idx = top.chosen_mode_idx;
        // Close Encounter additional-cost-choice envelope — Cycle 1
        // (resolve-time swap). Save → set → restore bracketed
        // exactly like the mode-index sibling above. `None` for
        // every non-additional-cost push (back-compat — every
        // closed Phase-1 / Phase-2 / modal arm + every trigger /
        // activation never sets it, so the swap is a correct no-op
        // for them). Mirrors Python's
        // `state.resolving_additional_cost_choice_id =
        // item.additional_cost_choice_id` swap at
        // `actions.py:2711`, with the `None` restore mirroring
        // the `finally` clause at `:2795`.
        let saved_addcost_choice_id = self.current_resolve_additional_cost_choice_id;
        self.current_resolve_additional_cost_choice_id =
            top.additional_cost_choice_id;
        // **scour-for-scrap-slice Cycle 1 widen (2026-05-21).**
        // Multi-pick mirror — save → set → restore-bracketed exactly
        // like the single-pick sibling. The resolve arm of
        // `Effect::ChooseOneMode` reads `current_resolve_chosen_mode_idxs`
        // first; if `Some(vec)`, it dispatches every chosen mode in
        // order. If `None`, it falls back to the single-pick path,
        // which keeps every pre-widen modal cast byte-identical.
        // The `clone()` is a Vec<u8> (typically 0-2 entries) — cheap.
        let saved_mode_idxs = self.current_resolve_chosen_mode_idxs.clone();
        self.current_resolve_chosen_mode_idxs = top.chosen_mode_idxs.clone();
        let effects_result: PyResult<()> = (|| {
            for effect in &effects {
                apply_effect(self, effect, top.source_obj_id)?;
            }
            Ok(())
        })();
        self.current_resolve_chosen_mode_idxs = saved_mode_idxs;
        self.current_resolve_additional_cost_choice_id =
            saved_addcost_choice_id;
        self.current_resolve_chosen_mode_idx = saved_mode_idx;
        self.current_resolve_payload_target_id_2 = saved_target_2;
        self.current_resolve_payload_target_id = saved_target;
        self.current_resolve_source_controller = saved_ctrl;
        effects_result?;

        // CR 608.2m: post-resolve, move the spell source. Instants /
        // sorceries -> controller's graveyard; permanents (creature,
        // artifact, enchantment, planeswalker, battle) ->
        // controller's battlefield with summoning sickness for
        // creatures. Empty card_types is the synthetic-test escape
        // hatch — leave source in place. Done BEFORE the stack pop
        // so a panic in movement doesn't leave the stack drained
        // with no movement applied (transactional shape).
        //
        // **Aura-attachment substrate STEP 4 (round-78).** If the
        // resolving spell is an Aura, route to `move_aura_post_resolve`
        // first: it consults the cast-time bound target (already
        // swapped into `current_resolve_payload_target_id` above),
        // performs the resolve-time legality check per CR 608.2b,
        // fizzles to graveyard if illegal, OR attaches + lands on bf
        // if legal. Mirrors Python's branch at `actions.py:2720-2758`.
        // Non-Aura permanents / instants / sorceries continue through
        // the regular `move_source_post_resolve`.
        let is_aura_source = find_source_printing_id(self, top.source_obj_id)
            .and_then(|pid| self.registry.lookup(pid))
            .map(|p| p.is_aura)
            .unwrap_or(false);
        let post_resolve_zone = if is_aura_source {
            move_aura_post_resolve(
                self,
                top.source_obj_id,
                top.controller,
                top.target_obj_id,
            )
        } else {
            move_source_post_resolve(self, top.source_obj_id)
        };

        // ETB-trigger Slice 3: legacy `etb_effects` inline loop
        // retired. The parse-time bridge in `cards.rs::parse_printing`
        // synthesizes a `TriggeredAbility { event: Enters,
        // predicate: SourceIsSelf, effect: ... }` per `etb_effects`
        // entry at registry-load time, so the predicate-path
        // `dispatch_event(Event::Enters, top.source_obj_id)` call
        // below handles every self-ETB effect that used to fire here.
        // The previous pop-then-dispatch ordering is preserved
        // (drain the resolving slot from the stack first, then
        // dispatch ENTERS) so the trigger goes on the now-empty
        // stack and resolves LIFO on the next priority round.

        // Drain the popped item BEFORE firing dispatch_event below —
        // ETB triggers go onto the stack ABOVE where the resolving
        // spell sat, but the resolving spell itself has now finished
        // and its stack-item must come off first. Mirrors Python's
        // flow: the spell pops first (the spell completed); the
        // trigger then pushes onto the now-empty stack, resolving
        // LIFO on the next priority round.
        let popped = self
            .stack
            .pop()
            .expect("peeked top but pop returned None — stack invariant broken");

        // ETB-trigger predicate-path dispatch fan-out (Slice 1
        // wiring; Slice 3 absorbed the legacy `etb_effects` inline
        // hatch by synthesizing
        // `TriggeredAbility { event=Enters, predicate=SourceIsSelf,
        // effect=... }` per `etb_effects` entry at parse time —
        // every self-ETB effect now fires through the same walker
        // as the Panharmonicon-shape observer triggers).
        if post_resolve_zone == Some(ZONE_BATTLEFIELD) {
            dispatch_event(self, Event::Enters, top.source_obj_id)?;
        }
        // Slice 2 — drain any ETB dispatches that effect arms
        // deferred via `pending_etb_dispatches` (RS2-RS7's spell-
        // effect resolutions: search-and-bf, reanimation, token
        // creation, mass reanim, clone-target, exile-recall). The
        // queue mechanism keeps the stack untouched during
        // `apply_effect` so the resolver's `pop()` always drains
        // the resolving slot, regardless of how many permanents
        // entered during the effect pass.
        drain_pending_etb_dispatches(self)?;
        // R74-2 — drain any DIES dispatches the spell's Effect arms
        // deferred via `pending_dies_dispatches`. Today the only
        // producer outside `apply_sbas` (which self-drains) is the
        // DepressurizeEffect hand-rolled destroy arm — its destroyed
        // creature's `Event.DIES` triggered ability (e.g. Gravpack
        // Monoist -> Robot token) MUST fire to match Python's
        // `turn.destroy` -> `move_off_battlefield`. Drained BEFORE
        // LEAVES so the stack-push order is DIES then LEAVES (LIFO
        // -> LEAVES resolves first), byte-equal to the SBA destroy
        // path (`state_ops_sba.rs` :497/:503) and Python's
        // `move_off_battlefield` (turn.py:253-289). No-op (empty
        // queue) for every pre-R74-2 spell resolution.
        drain_pending_dies_dispatches(self)?;
        // Slice 10 — drain any LEAVES dispatches the spell's Effect
        // arms deferred via `pending_leaves_dispatches` (bounce /
        // destroy / exile / sacrifice-as-effect at the 13 bf-exit
        // sites in `state_ops_resolve.rs` that bypass `move_object`
        // and call `mark_left_battlefield_if_nonland` directly). The
        // queue mechanism is the LEAVES counterpart of the Slice 2
        // ETB queue and follows the same defer-past-pop ordering:
        // the resolving spell pops first, then LEAVES triggers
        // pushed during `apply_effect` land on the now-empty top.
        drain_pending_leaves_dispatches(self)?;
        // Slice 14 (TTT Pin 1) — drain any IS_SACRIFICED dispatches
        // the spell's Effect arms deferred via the
        // `Effect::Sacrifice` arm's
        // `pending_is_sacrificed_dispatches.push`. Mirrors the LEAVES
        // sibling — defer-after-pop discipline so the IS_SACRIFICED
        // trigger lands on the now-empty top of stack, not above the
        // still-present resolving sacrifice spell.
        drain_pending_is_sacrificed_dispatches(self)?;
        // Auto-resolve any pending decisions (Round-27 G20 closure).
        // Pre-r27 behavior: Surveil etc. resolved with deterministic
        // "keep all on top". The decision-callback substrate now
        // packages those into PendingDecisions during apply_effect;
        // stack resolution defaults to the same "keep all on top"
        // strategy so existing baseline tests (test_library.py r23
        // surveil, etc.) are bit-equal to the pre-r27 path.
        // Callers that want to observe / control the decision use
        // `resolve_top_of_stack_pause` (no auto-resolve) instead.
        auto_resolve_pending(self, "keep_all_on_top")?;
        Ok(Some(stack_item_to_dict(py, &popped)?))
    }

    /// Same as `resolve_top_of_stack` but does NOT auto-resolve any
    /// `PendingDecision`s the effect dispatch leaves on the queue.
    /// The caller is responsible for either submitting decisions
    /// (`submit_surveil_decision`) or calling
    /// `auto_resolve_pending_decisions(strategy)` afterward.
    ///
    /// For tests / harnesses that need to introspect mid-resolution
    /// state (a SurveilChoice carrying its observed cards, before
    /// the controller routes them). The CR 608.2m post-resolve
    /// source mover still fires (the Sorcery still lands in the
    /// graveyard); the only difference is the pending decision
    /// queue remains populated.
    fn resolve_top_of_stack_pause<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let top = match self.stack.last() {
            None => return Ok(None),
            Some(item) => item.clone(),
        };
        // ETB-trigger Slice 1: trigger resolution path (same as
        // `resolve_top_of_stack` above, minus the auto-resolve of
        // pending decisions — `_pause` leaves them on the queue).
        if top.kind == StackItemKind::TriggeredAbility {
            let payload = top
                .trigger_payload
                .as_ref()
                .expect("StackItemKind::TriggeredAbility must carry trigger_payload");
            let trigger_effect = payload.effect.clone();
            // CE20 follow-up Slice 4: same payload-target swap as
            // `resolve_top_of_stack` above. See the longer comment
            // there for the rationale.
            let saved_target = self.current_resolve_payload_target_id;
            self.current_resolve_payload_target_id = payload.target_id;
            // DrawCards controller-of-effect fix: same controller
            // swap as `resolve_top_of_stack` above.
            let saved_ctrl = self.current_resolve_source_controller;
            self.current_resolve_source_controller = Some(top.controller);
            let apply_result = apply_effect(self, &trigger_effect, top.source_obj_id);
            self.current_resolve_source_controller = saved_ctrl;
            self.current_resolve_payload_target_id = saved_target;
            apply_result?;
            let popped = self
                .stack
                .pop()
                .expect("peeked top but pop returned None — stack invariant broken");
            // Slice 2: drain any RS-site deferred dispatches.
            drain_pending_etb_dispatches(self)?;
            // R74-2: drain DIES before LEAVES — see the matching
            // comment in `resolve_top_of_stack`'s trigger branch.
            drain_pending_dies_dispatches(self)?;
            // Slice 10: drain LEAVES sibling — see the matching
            // comment in `resolve_top_of_stack`'s trigger branch.
            drain_pending_leaves_dispatches(self)?;
            // Slice 14 (TTT Pin 1): drain IS_SACRIFICED sibling —
            // see matching comment in `resolve_top_of_stack`.
            drain_pending_is_sacrificed_dispatches(self)?;
            return Ok(Some(stack_item_to_dict(py, &popped)?));
        }
        let effects: Vec<Effect> = match find_source_printing_id(self, top.source_obj_id)
        {
            Some(printing_id) => match self.registry.lookup(printing_id) {
                Some(printing) => printing.effects.clone(),
                None => Vec::new(),
            },
            None => Vec::new(),
        };
        // DrawCards controller-of-effect fix: same controller swap
        // as `resolve_top_of_stack` above.
        let saved_ctrl = self.current_resolve_source_controller;
        self.current_resolve_source_controller = Some(top.controller);
        // Spell-card-bridge lever (round-70): the SAME spell-target
        // swap as `resolve_top_of_stack`'s spell arm. The two resolve
        // entry points MUST stay in lockstep — a missed sibling here
        // is exactly the round-60-N1-class transcription bug the
        // verify-first discipline guards against (pin #6 is the
        // sibling-parity guard). See the longer comment in
        // `resolve_top_of_stack` for the rationale.
        let saved_target = self.current_resolve_payload_target_id;
        self.current_resolve_payload_target_id = top.target_obj_id;
        // Spell-card-bridge lever Phase 2 (round-76): swap in the
        // SECOND bound target for a 2-slot spell (Diplomatic
        // Relations). Identical save → set → restore bracketing as
        // slot 0; `None` for every single-target / pre-Phase-2 push
        // (back-compat — the closed Phase-1 single-target arms
        // self-scan and ignore BOTH payload slots, so the swap is a
        // correct no-op for them). Mirrored into BOTH resolve entry
        // points (the §3.4 sibling-parity requirement; pin #4 / #6).
        let saved_target_2 = self.current_resolve_payload_target_id_2;
        self.current_resolve_payload_target_id_2 = top.target_obj_id_2;
        // ModalChooseOne envelope — Cycle 2 (cast-time-bound mode
        // dispatch). Swap in the StackItem's cast-time-bound mode
        // index so the `Effect::ChooseOneMode` arm reads the
        // pre-picked choice instead of pushing a resolve-time
        // `PendingDecision::ChooseModeChoice`. Mirrors Python's
        // `state.resolving_modes = item.chosen_modes` swap at
        // `actions.py:2712`. Save → set → restore bracketed exactly
        // like its payload-target sibling. `None` for every non-
        // modal push (back-compat — every closed Phase-1 / Phase-2
        // single-target / 2-slot arm + every trigger / activation
        // never sets it, so the swap is a correct no-op for them).
        let saved_mode_idx = self.current_resolve_chosen_mode_idx;
        self.current_resolve_chosen_mode_idx = top.chosen_mode_idx;
        // Close Encounter additional-cost-choice envelope — Cycle 1
        // (resolve-time swap, `_pause` sibling). Save → set →
        // restore bracketed identically to the `resolve_top_of_stack`
        // entry point above. Mirrored across BOTH resolve entry
        // points per the §3.4 sibling-parity requirement (the same
        // pattern Phase-2 / mode-index siblings follow).
        let saved_addcost_choice_id = self.current_resolve_additional_cost_choice_id;
        self.current_resolve_additional_cost_choice_id =
            top.additional_cost_choice_id;
        // **scour-for-scrap-slice Cycle 1 widen (2026-05-21).**
        // Multi-pick mirror — save → set → restore-bracketed exactly
        // like the single-pick sibling. The resolve arm of
        // `Effect::ChooseOneMode` reads `current_resolve_chosen_mode_idxs`
        // first; if `Some(vec)`, it dispatches every chosen mode in
        // order. If `None`, it falls back to the single-pick path,
        // which keeps every pre-widen modal cast byte-identical.
        // The `clone()` is a Vec<u8> (typically 0-2 entries) — cheap.
        let saved_mode_idxs = self.current_resolve_chosen_mode_idxs.clone();
        self.current_resolve_chosen_mode_idxs = top.chosen_mode_idxs.clone();
        let effects_result: PyResult<()> = (|| {
            for effect in &effects {
                apply_effect(self, effect, top.source_obj_id)?;
            }
            Ok(())
        })();
        self.current_resolve_chosen_mode_idxs = saved_mode_idxs;
        self.current_resolve_additional_cost_choice_id =
            saved_addcost_choice_id;
        self.current_resolve_chosen_mode_idx = saved_mode_idx;
        self.current_resolve_payload_target_id_2 = saved_target_2;
        self.current_resolve_payload_target_id = saved_target;
        self.current_resolve_source_controller = saved_ctrl;
        effects_result?;
        let post_resolve_zone = move_source_post_resolve(self, top.source_obj_id);
        // ETB-trigger Slice 3: legacy `etb_effects` inline loop
        // retired in `_pause` for the same reason as in
        // `resolve_top_of_stack` (the parse-time bridge synthesizes
        // `TriggeredAbility { event: Enters, predicate: SourceIsSelf,
        // effect: ... }` per `etb_effects` entry; the dispatch_event
        // call below handles them).
        // Pop the resolving spell's stack item BEFORE firing
        // dispatch_event — see the same-shaped comment in
        // `resolve_top_of_stack` above for the ordering rationale.
        let popped = self
            .stack
            .pop()
            .expect("peeked top but pop returned None — stack invariant broken");
        if post_resolve_zone == Some(ZONE_BATTLEFIELD) {
            // ETB-trigger Slice 1: same predicate-path fan-out as
            // `resolve_top_of_stack` above.
            dispatch_event(self, Event::Enters, top.source_obj_id)?;
        }
        // Slice 2: drain any RS-site deferred dispatches from the
        // effect pass above. Mirrors the `resolve_top_of_stack`
        // drain — the `_pause` variant just skips the
        // auto-resolve-pending step, not the ETB fan-out.
        drain_pending_etb_dispatches(self)?;
        // R74-2: drain DIES before LEAVES — see the matching
        // comment in `resolve_top_of_stack`'s spell branch.
        drain_pending_dies_dispatches(self)?;
        // Slice 10: drain LEAVES sibling — see the matching
        // comment in `resolve_top_of_stack`'s spell branch.
        drain_pending_leaves_dispatches(self)?;
        // Slice 14 (TTT Pin 1): drain IS_SACRIFICED sibling — see
        // matching comment in `resolve_top_of_stack`'s spell branch.
        drain_pending_is_sacrificed_dispatches(self)?;
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

        // CE4 closure: register_and_inject_object stamps a fresh
        // etb_timestamp only when injecting into ZONE_BATTLEFIELD;
        // injections into hand / library / graveyard / exile use the
        // default 0 (the object isn't on the battlefield yet, so its
        // ordering relative to bf objects is undefined). The stamp
        // bump is unconditional to keep the call site branch-free —
        // an unused timestamp leaks no observable state because only
        // the bf entries are sort-keyed by `etb_timestamp`.
        let etb_ts = if target_zone == ZONE_BATTLEFIELD {
            let ts = self.next_etb_timestamp;
            self.next_etb_timestamp = self.next_etb_timestamp.saturating_add(1);
            ts
        } else {
            0
        };
        let obj = GameObject {
            id: next_id,
            printing_id,
            controller,
            zone: target_zone,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: 0,
            charge_counters: 0,
            loyalty_counters: 0,
            attached_to: None,
            exiled_with_source: None,
            // Round-30 stun-counter substrate: caller-injected
            // objects start unstunned per make_object precedent.
            stun_counters: 0,
            etb_timestamp: etb_ts,
            ueot_modifiers: Vec::new(),
            // warp_cost substrate (Phase 1, round-63): caller-
            // injected objects default to not-warp. Back-compat.
            warp_exile_pending: false,
            warped: false,
            cast_from_exile_with_lightstall_surcharge: false,
            controller_baseline: None,
            // Aura-attachment Phase 4 S3 closure (2026-05-21):
            // caller-injected objects never start as Moonlit-
            // substituted tokens. Back-compat default.
            token_supertype_override: false,
            // Control-change-UEOT substrate — Cycle 1 (2026-05-21):
            // caller-injected objects start with no UEOT controller
            // override. Back-compat default; not in the per-object
            // fingerprint tuple.
            controller_override: None,
        };

        // R76-1: advance the monotonic id counter past this injected
        // object. `register_and_inject_object` is a test/setup
        // injection primitive (it runs before mid-game mints), so the
        // `max + 1` scan above is correct *here* for the injected
        // object's own id; but a subsequent mid-game token mint draws
        // from `next_object_id`, which must never re-issue `next_id`.
        // Take the max so a prior setup primitive that already pushed
        // the counter higher is never walked backwards (CR 111.7 —
        // the counter is monotonic).
        self.next_object_id =
            self.next_object_id.max(next_id.saturating_add(1));

        let player = &mut self.players[controller as usize];
        // Test-injection convention: for LIBRARY zone, insert at the
        // TOP of library (lib[0] post-CE3) so injected cards are
        // immediately reachable by top-N readers (scry, surveil,
        // search-from-top). Pre-CE3 the push-to-tail landed cards at
        // the top (which was the tail then); the post-CE3 equivalent
        // is insert(0, ...). For other zones (HAND, BATTLEFIELD,
        // GRAVEYARD) append-to-tail is the established pattern and
        // stays.
        // ETB-trigger Slice 2 (RS8): capture entered_id when the test
        // injection lands directly on the battlefield, so we can
        // dispatch `Event::Enters` AFTER releasing the player-mut
        // borrow below. Non-bf injections are no-ops on the trigger
        // path (the object isn't on the battlefield, so no permanent
        // observes its arrival).
        let entered_id: Option<u32> = if target_zone == ZONE_BATTLEFIELD {
            Some(next_id)
        } else {
            None
        };
        match target_zone {
            ZONE_HAND => {
                Arc::make_mut(&mut player.hand).push(obj);
            }
            ZONE_LIBRARY => {
                Arc::make_mut(&mut player.library).insert(0, obj);
            }
            ZONE_BATTLEFIELD => {
                Arc::make_mut(&mut player.battlefield).push(obj);
            }
            ZONE_GRAVEYARD => {
                Arc::make_mut(&mut player.graveyard).push(obj);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "register_and_inject_object: unsupported target_zone {other} (valid: 0-3)"
                )));
            }
        }
        // ETB-trigger Slice 2 (RS8): test-fixture battlefield
        // injections fan out ENTERS via the deferred-dispatch queue
        // so per-test substrate fixtures can exercise observer-style
        // triggers. Non-bf injections are a no-op on the trigger path.
        if let Some(id) = entered_id {
            self.pending_etb_dispatches.push(id);
            drain_pending_etb_dispatches(self)?;
        }
        Ok(next_id)
    }

    /// **Static / layered abilities substrate — Phase 2 (round-80,
    /// 2026-05-20).** Test-injection primitive: register a printing
    /// AND inject a fresh GameObject directly into `stack_objects`
    /// AND push a matching `StackItem` onto the stack. Used by
    /// counter-spell-gate tests to exercise the CR 101.2 uncounterable
    /// fizzle without driving a full `cast_creature` path (mana pool /
    /// land setup / etc.). Mirrors `register_and_inject_object`'s
    /// id-allocation + registry-extend pattern; the GameObject is
    /// placed in `state.stack_objects` (the G7 S5a home of spells
    /// on the stack) and the StackItem carries the same `source_obj_id`.
    /// Returns the new object id (= stack source id).
    ///
    /// **Phase-2 scope.** Spell-shape only — `kind == StackItemKind::
    /// Spell`, `target_obj_id == None`, `trigger_payload == None`.
    /// Sufficient for the counter-spell uncounterable-gate tests; a
    /// future slice can widen to triggered abilities + targeted
    /// spells.
    fn register_and_inject_stack_spell(
        &mut self,
        printing_dict: &Bound<'_, PyDict>,
        controller: u8,
    ) -> PyResult<u32> {
        if (controller as usize) >= 2 {
            return Err(PyIndexError::new_err(format!(
                "controller {controller} out of range (must be 0 or 1)"
            )));
        }
        let printing = parse_printing(printing_dict)?;
        let reg: &mut RegistryData = Arc::make_mut(&mut self.registry);
        let printing_id = reg.printings.len() as u32;
        reg.printings.push(printing);

        // Pick a fresh obj_id across all zones AND stack_objects (the
        // counter-spell gate needs to look the source up in
        // stack_objects).
        let mut next_id: u32 = self.next_object_id;
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
        for obj in self.stack_objects.iter() {
            if obj.id >= next_id {
                next_id = obj.id + 1;
            }
        }
        self.next_object_id = next_id.saturating_add(1);

        let obj = GameObject {
            id: next_id,
            printing_id,
            controller,
            zone: ZONE_STACK,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: 0,
            charge_counters: 0,
            loyalty_counters: 0,
            attached_to: None,
            exiled_with_source: None,
            stun_counters: 0,
            etb_timestamp: 0,
            ueot_modifiers: Vec::new(),
            warp_exile_pending: false,
            warped: false,
            cast_from_exile_with_lightstall_surcharge: false,
            controller_baseline: None,
            // Aura-attachment Phase 4 S3 closure (2026-05-21):
            // freshly-pushed stack objects (spell-as-source) are
            // never Moonlit-substituted tokens. Back-compat default.
            token_supertype_override: false,
            // Control-change-UEOT substrate — Cycle 1 (2026-05-21):
            // freshly-pushed stack objects start with no UEOT
            // controller override.
            controller_override: None,
        };
        self.stack_objects.push(obj);
        self.stack.push(StackItem {
            source_obj_id: next_id,
            controller,
            kind: StackItemKind::Spell,
            trigger_payload: None,
            target_obj_id: None,
            target_obj_id_2: None,
            // ModalChooseOne envelope — Cycle 1: this is the
            // bespoke synthetic-spell push helper (no printing
            // lookup); modal mode binding rides on the production
            // `cast_spell_noncreature` path. Default `None`.
            chosen_mode_idx: None,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1: synthetic-spell push has no printing →
            // additional-cost flag never set → default `None`.
            additional_cost_choice_id: None,
            // scour-for-scrap-slice Cycle 1 widen: same rationale.
            chosen_mode_idxs: None,
        });
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

    /// **Static / layered abilities substrate — Phase 3 (round-81,
    /// 2026-05-20) — test helper.** Directly set
    /// `Player.spells_cast_this_turn` for a player index. Used by
    /// the Phase-3 test harness to drive the
    /// ``at_least_n_spells_cast_this_turn`` condition tag through
    /// without going through the full cast pipeline (which would
    /// require a stack-and-priority simulation for a stub spell).
    /// Mirrors Python tests setting
    /// `state.players[0].spells_cast_this_turn = n` directly.
    fn set_spells_cast_this_turn(
        &mut self,
        player_idx: usize,
        value: i32,
    ) -> PyResult<()> {
        if player_idx >= 2 {
            return Err(PyIndexError::new_err(format!(
                "player_idx {player_idx} out of range (must be 0 or 1)"
            )));
        }
        self.players[player_idx].spells_cast_this_turn = value;
        Ok(())
    }

    /// **Round-82 (2026-05-20, Mechan Shieldmate closure) — test
    /// helper.** Directly set
    /// `Player.artifacts_entered_under_your_control_this_turn` for
    /// a player index. Used by the Mechan-shape test harness to
    /// drive the ``controlled_artifact_entered_this_turn``
    /// condition tag without going through a full bf-entry
    /// pipeline (which would require crafting a real artifact ETB
    /// for every probe). Mirrors `set_spells_cast_this_turn`'s
    /// shape one-for-one.
    fn set_artifacts_entered_under_your_control_this_turn(
        &mut self,
        player_idx: usize,
        value: i32,
    ) -> PyResult<()> {
        if player_idx >= 2 {
            return Err(PyIndexError::new_err(format!(
                "player_idx {player_idx} out of range (must be 0 or 1)"
            )));
        }
        self.players[player_idx]
            .artifacts_entered_under_your_control_this_turn = value;
        Ok(())
    }

    /// **Round-82 (2026-05-20, Mechan Shieldmate closure) — test
    /// helper.** Read
    /// `Player.artifacts_entered_under_your_control_this_turn`.
    fn get_artifacts_entered_under_your_control_this_turn(
        &self,
        player_idx: usize,
    ) -> PyResult<i32> {
        if player_idx >= 2 {
            return Err(PyIndexError::new_err(format!(
                "player_idx {player_idx} out of range (must be 0 or 1)"
            )));
        }
        Ok(self.players[player_idx]
            .artifacts_entered_under_your_control_this_turn)
    }
}
