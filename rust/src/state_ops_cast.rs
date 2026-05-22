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
//! - **No sorcery-speed restriction (CR 307.1) yet — so flash
//!   (CR 702.8) is currently a no-op.** ``cast_creature`` does not
//!   gate on ``state.phase`` at all today: a creature spell can be
//!   cast in any phase, including UNTAP, the opponent's combat,
//!   etc. The Python side gates sorcery-speed in ``legal_actions``,
//!   not in the play primitive. When a future slice adds the phase
//!   gate (rejecting non-main casts), the same slice will read
//!   ``printing.keywords`` for ``"flash"`` and skip the gate when
//!   the keyword is present. Until that gate lands, flash creatures
//!   already cast at any phase — vacuously satisfying the rule —
//!   and the ``test_cast.py`` flash tests pin that behavior so the
//!   future slice can extend the guard without regressing flash.
//!   One-land-per-turn is still caller-side, same rationale.
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

use crate::state::{GameObject, GameState, ManaPool};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::sync::Arc;

const ZONE_BATTLEFIELD: u8 = 2;
/// Local mirror of `crate::state_ops_zones::ZONE_HAND` (= 0),
/// matching the existing `ZONE_BATTLEFIELD` local-const pattern in
/// this module. Pre-S5a this was used by the exile-recast zone
/// normalization (exile -> hand); S5a re-homes that to the stack
/// store, but the const stays (kept for symmetry with the local
/// zone-const pattern; the re-home helper reads `ZONE_STACK`).
#[allow(dead_code)]
const ZONE_HAND: u8 = 0;
/// Local mirror of `crate::state_ops_zones::ZONE_STACK` (= 4).
/// **G7 S5a (Option β):** the cast moves the spell's owned
/// `GameObject` hand/gy/exile → `GameState.stack_objects` and stamps
/// `obj.zone = ZONE_STACK` (byte-mirroring Python's
/// `card.zone = Zone.STACK`, `actions.py:1751`). Same local-const
/// pattern as `ZONE_BATTLEFIELD`/`ZONE_HAND` above.
const ZONE_STACK: u8 = 4;

/// **G7 S5a (Option β) — re-home the cast spell into the
/// spell-stack-source-zone store.**
///
/// Removes the object with id `obj_id` from `player[idx]`'s `hand`
/// (the common case), else `exile` (warp exile-recast), else
/// `graveyard` (warp-from-graveyard), and pushes the owned
/// `GameObject` into `state.stack_objects` with `obj.zone =
/// ZONE_STACK` and `obj.controller = player_id`. This is the exact
/// byte-mirror of Python's cast-time
/// `p.hand.remove(card)` / `p.exile.remove(card)` /
/// `p.graveyard.remove(card)` (`actions.py:1403`/`:1501`/`:1450`)
/// then `card.zone = Zone.STACK` + `card.controller_id = player_id`
/// (`:1751`/`:1752`) — the card leaves every per-player zone and is
/// owned by the stack until resolution.
///
/// Idempotent-safe: if the id is not in any of the three source
/// zones (already moved, or a synthetic-state edge), it is a no-op
/// returning `false` (the caller's `Ok(false)` bail / honest
/// no-mutation contract still holds — `feedback_no_fallback_hybrids`).
/// Returns `true` iff an object was moved.
fn rehome_spell_to_stack_store(
    state: &mut GameState,
    idx: usize,
    obj_id: u32,
    player_id: u8,
) -> bool {
    // hand → stack store (the common cast-from-hand path).
    if let Some(pos) =
        state.players[idx].hand.iter().position(|o| o.id == obj_id)
    {
        let hand = Arc::make_mut(&mut state.players[idx].hand);
        let mut obj = hand.remove(pos);
        obj.zone = ZONE_STACK;
        obj.controller = player_id;
        state.stack_objects.push(obj);
        return true;
    }
    // exile → stack store (warp exile-recast — Timeline Culler etc.).
    if let Some(pos) =
        state.players[idx].exile.iter().position(|o| o.id == obj_id)
    {
        let exile = Arc::make_mut(&mut state.players[idx].exile);
        let mut obj = exile.remove(pos);
        obj.zone = ZONE_STACK;
        obj.controller = player_id;
        state.stack_objects.push(obj);
        return true;
    }
    // graveyard → stack store (warp-from-graveyard).
    if let Some(pos) =
        state.players[idx].graveyard.iter().position(|o| o.id == obj_id)
    {
        let gy = Arc::make_mut(&mut state.players[idx].graveyard);
        let mut obj = gy.remove(pos);
        obj.zone = ZONE_STACK;
        obj.controller = player_id;
        state.stack_objects.push(obj);
        return true;
    }
    false
}

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
/// `pub(crate)` re-export wrapper for the local `pay_mana_cost`.
///
/// **G7 frontier (a): NO live callers.** This wrapper's sole caller
/// was the `ActivateAbility` apply mana-pay site in
/// `state_ops_actions.rs`, which the frontier-(a) fix rerouted through
/// `auto_tap_pay_for_cost` (the §2-4 byte-mirror of Python's
/// `auto_tap_for_cost`) so activated-ability apply auto-taps lands
/// exactly as Python's `_activate_ability` does. The strict pool-only
/// primitive is kept (NOT deleted — minimal scope; the helper docs and
/// the `SEARCH_LIMITATIONS.md` G7 note reference it as the "old
/// pool-only `pay_mana_cost`" honest-failure baseline the
/// plan-then-commit helper degrades to) and explicitly NOT a runtime
/// fallback: it has zero live callers, so nothing routes to it — there
/// is no fallback hybrid (`feedback_no_fallback_hybrids`).
#[allow(dead_code)]
pub(crate) fn pay_mana_cost_pub(pool: &mut ManaPool, cost: &crate::cards::ManaCost) -> bool {
    pay_mana_cost(pool, cost)
}

fn pay_mana_cost(pool: &mut ManaPool, cost: &crate::cards::ManaCost) -> bool {
    // Fast-fail on totals. Saves a partial spend + rollback in the
    // common "you don't have enough mana, period" case.
    if pool_total(pool) < cost_total(cost) {
        return false;
    }
    // Snapshot so we can roll back on a partial-pay failure (a
    // colored pip the pool can't cover). G7 S6-Class-E ordered-pool:
    // `ManaPool` carries `Vec<ManaCoin>` and isn't `Copy`; clone
    // explicitly so the rollback path can restore both the per-color
    // counts AND the insertion-ordered `coins` vec.
    let snapshot = pool.clone();
    if pool.white < cost.white
        || pool.blue < cost.blue
        || pool.black < cost.black
        || pool.red < cost.red
        || pool.green < cost.green
    {
        return false;
    }
    // G7 S6-Class-E ordered-pool: drain colored from `coins` in FIFO
    // (per-color) to keep the insertion-ordered vec in sync with the
    // per-color count cache. Same effective per-color delta as the
    // legacy `pool.X -= cost.X` shape, but `coins` no longer drifts.
    pool.drain_first_n_color(b'W', cost.white as u32);
    pool.drain_first_n_color(b'U', cost.blue as u32);
    pool.drain_first_n_color(b'B', cost.black as u32);
    pool.drain_first_n_color(b'R', cost.red as u32);
    pool.drain_first_n_color(b'G', cost.green as u32);
    let mut generic_remaining = cost.generic as u32;
    for color in [b'C', b'W', b'U', b'B', b'R', b'G'] {
        if generic_remaining == 0 {
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
        let take = avail.min(generic_remaining);
        pool.drain_first_n_color(color, take);
        generic_remaining -= take;
    }
    if generic_remaining > 0 {
        // Shouldn't happen — pool_total covered the case above —
        // but defend anyway with a rollback.
        *pool = snapshot;
        return false;
    }
    pool.assert_count_sync();
    true
}

/// **G7 mid-game STATE_EQ closure — apply-side auto-tap-from-lands
/// payment, byte-mirroring Python's `auto_tap_for_cost`.**
///
/// The three stack-push cast primitives (`cast_creature`,
/// `cast_creature_warp`, `cast_spell_noncreature`) previously paid
/// `cost` STRICTLY from the controller's floating `mana_pool`
/// (`pay_mana_cost`), bailing `Ok(false)` (a silent no-op) when the
/// pool alone could not cover it. But the Rust legal emitter is
/// PERMISSIVE: `auto_tap_headroom_pool` (`state_ops_legal.rs:622`) /
/// `can_pay_cost` (`state_ops_mana.rs:460`) offer a `CastSpell` whenever
/// floating pool **plus auto-tappable lands** could pay — exactly
/// Python's `can_pay_cost` (`actions.py:336-403`). Python's
/// `_cast_spell` then ACTUALLY pays by `auto_tap_for_cost`
/// (`actions.py:919-986`): consume floating pool first, then tap lands
/// for the remainder. Rust's apply path had no such auto-tap, so a
/// production-deck cast whose cost needs land-tapping (pool alone
/// insufficient) was emitted-legal-then-silently-dropped — the
/// post-S1/S4 mid-game STATE_EQ frontier (arch2/s1 step 273
/// Galactic Wayfarer `{2}{G}`; arch5/s0 step 322 Icecave Crasher).
///
/// This helper closes that gap: it pays `cost` for `player_id` by
/// **byte-mirroring `auto_tap_for_cost`'s algorithm exactly** —
///
///   1. Each colored requirement, in Python's `ManaCost.colored`
///      iteration order (alphabetical by symbol: **B, G, R, U, W** —
///      `cards.parse_mana_cost` sorts the pairs), is paid from the
///      matching floating-pool slot FIRST, then by tapping untapped
///      controller lands that produce that color (in `battlefield`
///      iteration order, `[:need]` — the same selection
///      `auto_tap_for_cost` makes via `_untapped_lands` +
///      `effective_land_produces`).
///   2. Generic (`cost.generic`) is paid from the remaining floating
///      pool (deterministic C→W→U→B→R→G drain, matching
///      `pay_mana_cost`'s post-colored generic order), then by tapping
///      the remaining untapped lands `[:generic_need]`.
///   3. Each tapped land has `tapped = true` set and fires
///      `Event::BecomesTapped` via the SAME post-borrow inline
///      dispatch `tap_for_mana` uses (`state_ops_mana.rs:369-393`) —
///      Python routes every tapped land through `turn.tap_permanent`
///      which fires `Event.BECOMES_TAPPED` (`actions.py:981-985`).
///
/// **Lands only — NOT mana creatures.** Python's `auto_tap_for_cost`
/// AND its `can_pay_cost` predicate both source from `_untapped_lands`
/// (`actions.py:196-200`, `o.printing.is_land` only), NOT mana
/// creatures. The Rust legal-side `auto_tap_headroom_pool` is broader
/// (it also counts non-sick mana creatures) — that emission-side
/// over-permissiveness vs Python's lands-only is a DISTINCT C1 axis
/// (`RUST_LEGAL_DECISIONS_MANA_AWARENESS_DESIGN.md`), out of scope
/// here and not exercised by these repros. This apply-side helper is
/// deliberately lands-only so it byte-matches the apply SoT
/// (`auto_tap_for_cost`).
///
/// **Land detection / produced colors.** A land is
/// `printing.card_types` containing `"Land"` (Python's
/// `printing.is_land`). Its produced colors come from
/// `printing.mana_produced` — the SAME field the legal-side
/// `auto_tap_headroom_pool` (`state_ops_legal.rs:631`) and
/// `tap_for_mana` (`state_ops_mana.rs:328`) read; the registry stores
/// a single-color land like Forest as `Some(vec!["G"])`
/// (`cards.rs:4566-4574`). On the live bridge-admitted surface there
/// are no layer-3/4 land-type rewrites, so `mana_produced` is exactly
/// Python's `effective_land_produces` produced-color set for the
/// lands these casts auto-tap.
///
/// Returns `true` and commits (pool debited, lands tapped, events
/// dispatched) iff payable; `false` with NO mutation otherwise (the
/// caller bails exactly as the old pool-only `pay_mana_cost` did, so
/// the unpayable path is unchanged).
///
/// **G7 frontier (a) — `ActivateAbility` apply auto-tap (the §2-4
/// analog).** Originally module-private (the three cast pay sites were
/// the only §2-4 callers). Promoted to `pub(crate)` so the
/// `ActivateAbility` apply arm in `state_ops_actions.rs` pays the
/// activated-ability mana cost through the SAME byte-mirror of Python's
/// `auto_tap_for_cost`. Python's `_activate_ability`
/// (`actions.py:1992-1995`) pays the ability mana via the IDENTICAL
/// `auto_tap_for_cost` call `_cast_spell` makes (`actions.py:1662`), so
/// the activated-ability apply path MUST auto-tap lands exactly as the
/// cast path does. The legal emitter already gated `ActivateAbility` on
/// pool + auto-tap headroom (`state_ops_legal.rs:2105`,
/// `pool_can_cover(&auto_tap_headroom_pool(...))` = the `can_pay_cost`
/// shape, byte-equal to Python's `legal.py:653` activated-ability
/// predicate); routing apply through this helper makes apply honor what
/// legal offered — closing the Rust-internal legal-emit-vs-apply
/// self-inconsistency the bake-in surfaced at arch2/s1 step 318
/// (`ActivateAbility(121,0)` `{2}` Lander, pool `{U:1}` + 2 untapped
/// Forests: emitted-legal then pool-only-rejected). Lands-only (matches
/// Python's `auto_tap_for_cost`); the mana-creature-breadth C1 axis is
/// separate and unchanged.
pub(crate) fn auto_tap_pay_for_cost(
    state: &mut GameState,
    idx: usize,
    cost: &crate::cards::ManaCost,
) -> PyResult<bool> {
    // Colored requirements in Python's `ManaCost.colored` order
    // (alphabetical by symbol). Each entry: (count, the produced
    // symbol a land must list to pay it). B, G, R, U, W.
    let colored: [(u8, &str); 5] = [
        (cost.black, "B"),
        (cost.green, "G"),
        (cost.red, "R"),
        (cost.blue, "U"),
        (cost.white, "W"),
    ];

    // Snapshot the untapped controller lands in battlefield order,
    // recording each land's produced color symbols. Mirrors Python's
    // `lands = _untapped_lands(state, player_id)` (battlefield order,
    // controller-owned, `is_land`, not tapped). Produced colors come
    // from `printing.mana_produced` (the same field the legal-side
    // `auto_tap_headroom_pool` / `tap_for_mana` read; a single-color
    // land like Forest is `Some(vec!["G"])`).
    let mut lands: Vec<(u32, Vec<String>)> = Vec::new();
    for obj in state.players[idx].battlefield.iter() {
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
        lands.push((obj.id, symbols));
    }

    // ---- Phase 1: PLAN (no mutation). Decide which pool COIN INDICES
    // are consumed and which land ids are tapped. Bail with no mutation
    // if the plan cannot be completed — byte-equivalent to
    // `auto_tap_for_cost` being reachable only after `can_pay_cost`
    // (Python) / the permissive legal emit (Rust) said yes, but
    // strictly self-checked so an over-permissive emit degrades to the
    // same honest `false`-bail the old `pay_mana_cost` produced (NOT a
    // partial spend). ----
    //
    // **G7 S6-Class-E ordered-pool fix (Python SoT parity).** Python's
    // `auto_tap_for_cost` (`actions.py:919-986`) drains the pool by
    // tracking COIN INDICES (`used_coin_ids`) and iterating
    // `enumerate(p.mana_pool)` for both the colored and generic
    // phases. The colored phase scans coins IN INSERTION ORDER picking
    // matching-color coins; the generic phase scans IN INSERTION ORDER
    // picking any un-used coin. The interleaving means a pool like
    // `[G, G, U, U]` paying cost `{2}` (pure generic) consumes the
    // first two list entries (G, G) → residual `U:2`. The legacy
    // Rust drain order (C→W→U→B→R→G fixed priority) would have
    // drained `U` before `G` → residual `G:2`. THAT is the Class-E
    // signature.
    //
    // The plan here mirrors Python exactly: it walks the controller's
    // `coins` vec to pick INDICES into that vec, never per-color
    // counts. The commit phase removes those indices (high-to-low to
    // keep lower indices stable, exactly Python's
    // `sorted(used_coin_ids, reverse=True)`).
    let pool_coins: Vec<crate::state::ManaCoin> =
        state.players[idx].mana_pool.coins.clone();
    let mut used_coin_indices: Vec<usize> = Vec::new();
    let mut used_land: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    let mut tapped_plan: Vec<u32> = Vec::new();

    for (need_count, sym) in colored {
        let mut need = need_count as u32;
        if need == 0 {
            continue;
        }
        // Pool coins of the matching color first — iterate the
        // insertion-ordered `coins` vec, skipping already-used
        // indices, picking matching-color coins. Byte-mirror of
        // Python `for idx, coin in enumerate(p.mana_pool):` +
        // `if _coin_matches_color(coin, color)` (`actions.py:940-949`).
        // Rust's `ManaCoin` is single-color (W/U/B/R/G/C — no "any"
        // wildcard in the live registry), so the match is byte-equal.
        let want = match sym {
            "W" => b'W',
            "U" => b'U',
            "B" => b'B',
            "R" => b'R',
            "G" => b'G',
            _ => unreachable!("colored symbol vocabulary is W/U/B/R/G"),
        };
        for (i, coin) in pool_coins.iter().enumerate() {
            if need == 0 {
                break;
            }
            if used_coin_indices.contains(&i) {
                continue;
            }
            if coin.color == want {
                used_coin_indices.push(i);
                need -= 1;
            }
        }
        if need == 0 {
            continue;
        }
        // Then tap matching untapped lands, in battlefield order,
        // `[:need]` (Python's `matching[:need]`).
        for (lid, syms) in lands.iter() {
            if need == 0 {
                break;
            }
            if used_land.contains(lid) {
                continue;
            }
            if syms.iter().any(|s| s == sym) {
                used_land.insert(*lid);
                tapped_plan.push(*lid);
                need -= 1;
            }
        }
        if need > 0 {
            // Unpayable (an over-permissive emit). No mutation.
            return Ok(false);
        }
    }

    // Generic: walk `coins` IN INSERTION ORDER picking any un-used
    // coin until `generic_need` is satisfied — byte-mirror of Python
    // `for idx, coin in enumerate(p.mana_pool): ... used_coin_ids.
    // append(idx); generic_need -= 1` (`actions.py:965-973`). The
    // remainder is paid by tapping lands `[:generic_need]`. THIS is
    // the Class-E fix: the order now follows pool LIST ORDER, not a
    // fixed C→W→U→B→R→G color priority.
    let mut generic_need = cost.generic as u32;
    for (i, _coin) in pool_coins.iter().enumerate() {
        if generic_need == 0 {
            break;
        }
        if used_coin_indices.contains(&i) {
            continue;
        }
        used_coin_indices.push(i);
        generic_need -= 1;
    }
    if generic_need > 0 {
        for (lid, _syms) in lands.iter() {
            if generic_need == 0 {
                break;
            }
            if used_land.contains(lid) {
                continue;
            }
            used_land.insert(*lid);
            tapped_plan.push(*lid);
            generic_need -= 1;
        }
    }
    if generic_need > 0 {
        // Unpayable. No mutation.
        return Ok(false);
    }

    // ---- Phase 2: COMMIT. The plan is complete; remove the planned
    // coin indices from the pool (high-index-first so lower indices
    // stay stable — same as Python's `sorted(used_coin_ids,
    // reverse=True)` then `del p.mana_pool[idx]`,
    // `actions.py:978-980`) and tap the planned lands. The drain
    // helper keeps the per-color count cache in sync as each coin is
    // removed. ----
    {
        let mp = &mut state.players[idx].mana_pool;
        // Walk indices high-to-low so each `remove(idx)` doesn't shift
        // later indices we still want to remove.
        let mut sorted_indices = used_coin_indices.clone();
        sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted_indices {
            // Defensive: indices come from a Vec we just snapshotted;
            // they MUST still be valid because nothing else mutated
            // the pool between snapshot and now (PyMethod, single-
            // threaded).
            debug_assert!(i < mp.coins.len(), "stale coin index after plan");
            let removed = mp.coins.remove(i);
            // Mirror in the per-color count cache so the existing
            // count-reader call sites (`pool_can_cover`, fingerprint,
            // snapshot, etc.) stay consistent.
            match removed.color {
                b'W' => mp.white = mp.white.saturating_sub(1),
                b'U' => mp.blue = mp.blue.saturating_sub(1),
                b'B' => mp.black = mp.black.saturating_sub(1),
                b'R' => mp.red = mp.red.saturating_sub(1),
                b'G' => mp.green = mp.green.saturating_sub(1),
                b'C' => mp.colorless = mp.colorless.saturating_sub(1),
                _ => {} // Defensive: vocabulary already validated.
            }
        }
        mp.assert_count_sync();
    }
    for lid in tapped_plan {
        {
            let bf = Arc::make_mut(&mut state.players[idx].battlefield);
            for obj in bf.iter_mut() {
                if obj.id == lid {
                    obj.tapped = true;
                    break;
                }
            }
        }
        // Same post-borrow inline BECOMES_TAPPED dispatch
        // `tap_for_mana` uses (`state_ops_mana.rs:369-393`): the flip
        // is committed above (the bf mut-borrow is released), and the
        // Phase-1 plan only ever scheduled UNTAPPED lands, so this is
        // exactly the "state actually changed" branch Python's
        // `tap_permanent` dispatches on.
        crate::state_ops_resolve::dispatch_event(
            state,
            crate::cards::Event::BecomesTapped,
            lid,
        )?;
    }
    Ok(true)
}

/// Move the GameObject with id `obj_id` out of `state.players[idx].hand`
/// and onto `state.players[idx].battlefield`, stamping its `zone`
/// field to battlefield. Returns the moved object on success, or
/// `None` if no object with that id is in hand. Both zone mutations
/// route through `Arc::make_mut` so the CoW invariant holds.
///
/// **CE4 closure**: stamps the freshly-entering permanent's
/// `etb_timestamp` from `state.next_etb_timestamp` (then bumps the
/// counter by 1, saturating). This is THE canonical bf-entry helper
/// for play_land + cast_creature; future helpers that fold spell
/// resolution into the same primitive should route through here.
fn move_from_hand_to_battlefield(
    state: &mut GameState,
    idx: usize,
    obj_id: u32,
) -> PyResult<Option<GameObject>> {
    let Some(pos) = state.players[idx].hand.iter().position(|o| o.id == obj_id) else {
        return Ok(None);
    };
    // CE4: reserve the timestamp BEFORE the borrow-conflict-prone
    // mutable zone slices so we never need a second borrow of `state`.
    let etb_ts = state.next_etb_timestamp;
    state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(1);
    // CE20 Slice 5 (Walker loyalty — starting_loyalty ETB seeding).
    // Pre-read the printing's walker-and-starting-loyalty shape on
    // the immutable registry BEFORE taking the player-mutable borrow
    // below — Rust borrow checker doesn't do field splitting through
    // method calls so we can't read `state.registry` while holding
    // `&mut state.players[idx]`. Look up by hand.position's pid.
    let starting_loyalty: i32 = {
        let pid = state.players[idx].hand[pos].printing_id;
        match state.registry.lookup(pid) {
            Some(printing)
                if printing.card_types.iter().any(|t| t == "Planeswalker") =>
            {
                printing.starting_loyalty
            }
            _ => 0,
        }
    };
    let player = &mut state.players[idx];
    let hand = Arc::make_mut(&mut player.hand);
    let mut obj = hand.remove(pos);
    obj.zone = ZONE_BATTLEFIELD;
    obj.etb_timestamp = etb_ts;
    // CR 606.5b ETB seed (continued). For non-walkers
    // starting_loyalty is 0, leaving loyalty_counters at its current
    // value (also 0 — the default from make_object).
    if starting_loyalty != 0 {
        obj.loyalty_counters = starting_loyalty;
    }
    let snapshot = obj.clone();
    let entered_id = obj.id;
    let bf = Arc::make_mut(&mut player.battlefield);
    bf.push(obj);
    // ETB-trigger Slice 2 (RS11): queue + drain ENTERS for the
    // freshly-arrived permanent. Routes both `play_land` (CR 305.5
    // — playing a land is not casting a spell, so the land bypasses
    // the stack) and `cast_creature`'s no-stack primitive through
    // the predicate path. The normal stack-cast flow lands via RS1
    // in `state_ops_resolve.rs::resolve_top_of_stack` instead, so
    // this call site is the one place where hand->bf moves outside
    // of a stack resolution emit ENTERS.
    state.pending_etb_dispatches.push(entered_id);
    crate::state_ops_resolve::drain_pending_etb_dispatches(state)?;
    // Return a snapshot clone of the freshly-pushed object for the
    // caller's book-keeping (e.g. summoning-sick stamping is done by
    // the caller AFTER this push via `bf.iter_mut().find()`, see
    // `cast_creature`). Round-27 note: `GameObject` lost its `Copy`
    // derive when `ueot_modifiers: Vec<UeotModifier>` landed, so the
    // pre-r27 "return the moved value by copy" shape now requires an
    // explicit clone — the empty-modifiers warm path makes that
    // clone cheap (a null heap ptr + zero len/cap).
    Ok(Some(snapshot))
}

/// **Phase-1 warp exile-recast (round-63).** Sibling of
/// `move_from_hand_to_battlefield` for the warp exile-recast path: a
/// `warped` object in `player.exile` moves to the battlefield (CR
/// 400.7 — new object; the caller clears `warped`/
/// `warp_exile_pending`). Mirrors the hand helper's CE4 timestamp
/// reservation + walker-loyalty seed + ENTERS drain so the recast
/// permanent is byte-identical to a hand-cast one (modulo the
/// origin zone). Returns the pushed-object snapshot, or `None` if
/// the id isn't a live exile object for `idx`.
///
/// **G7 Slice S1:** `cast_creature` no longer fuses cast+resolve, so
/// the exile-recast no longer moves exile -> battlefield in the cast
/// primitive (it normalizes exile -> hand at cast time and lets
/// `move_source_post_resolve` land it on the battlefield at
/// resolution, byte-mirroring Python). This helper is retained for
/// G7 Slice S4 (`cast_creature_warp` + riders) which still has the
/// exile-recast move pattern to re-home; `#[allow(dead_code)]` keeps
/// the warning quiet in the S1-only interim without deleting the
/// verified helper.
#[allow(dead_code)]
fn move_from_exile_to_battlefield(
    state: &mut GameState,
    idx: usize,
    obj_id: u32,
) -> PyResult<Option<GameObject>> {
    let Some(pos) = state.players[idx]
        .exile
        .iter()
        .position(|o| o.id == obj_id)
    else {
        return Ok(None);
    };
    let etb_ts = state.next_etb_timestamp;
    state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(1);
    let starting_loyalty: i32 = {
        let pid = state.players[idx].exile[pos].printing_id;
        match state.registry.lookup(pid) {
            Some(printing)
                if printing.card_types.iter().any(|t| t == "Planeswalker") =>
            {
                printing.starting_loyalty
            }
            _ => 0,
        }
    };
    let player = &mut state.players[idx];
    let exile = Arc::make_mut(&mut player.exile);
    let mut obj = exile.remove(pos);
    obj.zone = ZONE_BATTLEFIELD;
    obj.etb_timestamp = etb_ts;
    if starting_loyalty != 0 {
        obj.loyalty_counters = starting_loyalty;
    }
    let snapshot = obj.clone();
    let entered_id = obj.id;
    let bf = Arc::make_mut(&mut player.battlefield);
    bf.push(obj);
    state.pending_etb_dispatches.push(entered_id);
    crate::state_ops_resolve::drain_pending_etb_dispatches(state)?;
    Ok(Some(snapshot))
}

/// **Phase-2 warp-cast-from-graveyard (Timeline Culler).** Sibling of
/// `move_from_hand_to_battlefield` / `move_from_exile_to_battlefield`
/// for the graveyard→battlefield warp-cast path: a
/// `warp_castable_from_graveyard` object in `player.graveyard` moves
/// to the battlefield (mirrors Python's `actions.py:1437-1464`
/// `p.graveyard.remove(card)` then `put_into_battlefield`). Identical
/// CE4 timestamp reservation + walker-loyalty seed + ENTERS drain as
/// the hand / exile helpers so the graveyard-warp-cast permanent is
/// byte-identical to a hand-warp-cast one (modulo the origin zone —
/// Timeline Culler has `triggered_abilities=()` so the ETB drain is
/// inert for THIS card but keeps the path forward-correct and
/// identical to the hand arm). Returns the pushed-object snapshot, or
/// `None` if the id isn't a live graveyard object for `idx`.
///
/// **G7 Slice S4:** `cast_creature_warp` no longer fuses
/// cast+resolve, so the graveyard-warp source no longer moves
/// graveyard -> battlefield in the cast primitive (it normalizes
/// graveyard -> hand at cast time — the S1 exile-recast pattern — and
/// lets `move_source_post_resolve` land it on the battlefield at
/// resolution, byte-mirroring Python). This helper is now S4-dead;
/// `#[allow(dead_code)]` keeps the warning quiet without deleting the
/// verified helper (exact treatment S1 gave
/// `move_from_exile_to_battlefield`).
#[allow(dead_code)]
fn move_from_graveyard_to_battlefield(
    state: &mut GameState,
    idx: usize,
    obj_id: u32,
) -> PyResult<Option<GameObject>> {
    let Some(pos) = state.players[idx]
        .graveyard
        .iter()
        .position(|o| o.id == obj_id)
    else {
        return Ok(None);
    };
    let etb_ts = state.next_etb_timestamp;
    state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(1);
    let starting_loyalty: i32 = {
        let pid = state.players[idx].graveyard[pos].printing_id;
        match state.registry.lookup(pid) {
            Some(printing)
                if printing.card_types.iter().any(|t| t == "Planeswalker") =>
            {
                printing.starting_loyalty
            }
            _ => 0,
        }
    };
    let player = &mut state.players[idx];
    let graveyard = Arc::make_mut(&mut player.graveyard);
    let mut obj = graveyard.remove(pos);
    obj.zone = ZONE_BATTLEFIELD;
    obj.etb_timestamp = etb_ts;
    if starting_loyalty != 0 {
        obj.loyalty_counters = starting_loyalty;
    }
    let snapshot = obj.clone();
    let entered_id = obj.id;
    let bf = Arc::make_mut(&mut player.battlefield);
    bf.push(obj);
    state.pending_etb_dispatches.push(entered_id);
    crate::state_ops_resolve::drain_pending_etb_dispatches(state)?;
    Ok(Some(snapshot))
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
        // outcome before any Arc::make_mut clone fires. Hand is the
        // common path; exile is the Lightstall-permission fallback
        // (cast-from-exile-with-cost-hike substrate).
        let in_hand = self.players[idx]
            .hand
            .iter()
            .any(|o| o.id == obj_id);
        let from_exile_lightstall = !in_hand
            && self.players[idx]
                .exile
                .iter()
                .any(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                });
        let printing_id = if in_hand {
            match self.players[idx]
                .hand
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else if from_exile_lightstall {
            match self.players[idx]
                .exile
                .iter()
                .find(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                })
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else {
            return Ok(false);
        };
        let printing = match self.registry.lookup(printing_id) {
            Some(p) => p,
            None => return Ok(false),
        };
        if !printing.card_types.iter().any(|t| t == "Land") {
            return Ok(false);
        }
        if from_exile_lightstall {
            // **Cast-from-exile-with-cost-hike substrate -- the
            // tapped-land-entry rider.** A land played via the
            // Lightstall permission ETBs tapped regardless of the
            // printing's static `enters_tapped` flag. Mirrors Python
            // `actions.py:1080-1094` (`lightstall_land =
            // card.cast_from_exile_with_lightstall_surcharge;
            // card.cast_from_exile_with_lightstall_surcharge = False;
            // ... card.tapped = bool(card.printing.enters_tapped);
            // if lightstall_land: card.tapped = True`). The flag is
            // cleared at play time on the exile-resident object
            // (before the move) so the moved object lands on bf
            // with the flag already False (Python parity --
            // actions.py:1083 clears BEFORE the
            // `card.zone = BATTLEFIELD` assignment).
            let exile = Arc::make_mut(&mut self.players[idx].exile);
            let Some(pos) = exile.iter().position(|o| o.id == obj_id)
            else {
                return Ok(false);
            };
            let mut obj = exile.remove(pos);
            obj.cast_from_exile_with_lightstall_surcharge = false;
            obj.zone = ZONE_BATTLEFIELD;
            obj.controller = player_id;
            obj.tapped = true;
            // CE4: reserve and stamp a fresh ETB timestamp (mirrors
            // `move_from_hand_to_battlefield`).
            let etb_ts = self.next_etb_timestamp;
            self.next_etb_timestamp =
                self.next_etb_timestamp.saturating_add(1);
            obj.etb_timestamp = etb_ts;
            let entered_id = obj.id;
            let bf = Arc::make_mut(&mut self.players[idx].battlefield);
            bf.push(obj);
            // ETB-trigger Slice 2 (RS11): queue + drain ENTERS for
            // the freshly-arrived permanent. Mirrors
            // `move_from_hand_to_battlefield`'s drain.
            self.pending_etb_dispatches.push(entered_id);
            crate::state_ops_resolve::drain_pending_etb_dispatches(self)?;
            return Ok(true);
        }
        // Commit the move. The hand search re-runs inside
        // `move_from_hand_to_battlefield`; that's fine — the cost is
        // a single second linear scan we already know will hit, and
        // it keeps the helper self-contained.
        let moved = move_from_hand_to_battlefield(self, idx, obj_id)?;
        Ok(moved.is_some())
    }

    /// **G7 Slice S1 (stack-and-priority creature cast).** Cast a
    /// creature spell from `player_id`'s hand. Pays the printing's
    /// `mana_cost` out of the player's mana pool, fires `Event::Cast`,
    /// and **pushes a spell `StackItem`** — the card does NOT enter
    /// the battlefield here. Returns `true` on success, `false` on
    /// any precondition failure: object not in hand (nor a `warped`
    /// exile-recast), no such printing, the printing's `card_types`
    /// does not contain `"Creature"`, or insufficient mana.
    ///
    /// **Stack-and-priority semantics (G7 Slice S1 — was no-stack).**
    /// This primitive now mirrors `cast_spell_noncreature` exactly:
    /// it pays mana, dispatches the CAST event, pushes a
    /// `StackItem::new_spell` (the spell GameObject stays in the hand
    /// zone — Rust's pre-existing spell-stack model;
    /// `find_source_printing_id` / `take_source` search the hand at
    /// resolve), resets priority/passes, and **returns WITHOUT
    /// resolving**. The harness drives `resolve_top_of_stack`
    /// separately (CE5 cast-vs-resolve mirror today; the real
    /// adapter priority loop in production). At resolution the
    /// ALREADY-PRESENT `move_source_post_resolve`
    /// (`state_ops_resolve.rs:285-332`) lands the creature on the
    /// battlefield with `summoning_sick = true` (`:301-303`) + a
    /// fresh ETB timestamp (`:311-318`), then `resolve_top_of_stack`
    /// dispatches `Event::Enters` — byte-mirroring Python's
    /// resolve-time `put_into_battlefield` (`actions.py:2778` ->
    /// `:2993` sets `summoning_sick = True`). NO `summoning_sick`
    /// stamp is applied here at cast time — Python doesn't either
    /// (`_cast_spell` `actions.py:1403/1751-1767` only removes the
    /// card from its source zone, sets `zone=STACK`, and appends the
    /// `StackItem`; the bf entry + summoning sickness happen at
    /// resolve).
    ///
    /// Creature spells carry NO cast-time target (a vanilla permanent
    /// spell has no `_select_spell_targets` pick), so the bare
    /// untargeted `StackItem::new_spell` constructor is used (vs
    /// `cast_spell_noncreature`'s `new_spell_with_targets` for
    /// single-/two-target instants/sorceries). `cast_creature_warp`
    /// + the warp/impending/room riders are deferred to G7 Slice S4
    /// (this S1 lane is creature-only, non-warp; the
    /// `cast_creature_warp` primitive is untouched here).
    fn cast_creature(&mut self, obj_id: u32, player_id: u8) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Phase 1: read-only lookup. Find the object in hand, fetch
        // the printing's cost + types. Bail before any mutation if
        // any step fails. The type check happens here (pre-mutation)
        // so a non-creature spell never has its mana speculatively
        // debited.
        //
        // **Phase-1 warp exile-recast (round-63):** if the object is
        // NOT in hand but IS a `warped` object in this player's
        // exile, this is a warp exile-recast — pay the NORMAL
        // `mana_cost` (legal.py:478-489 / actions.py:1476-1510). The
        // hand path is tried first (the common case); the exile
        // fallback only fires for a `warped` exile object. Both pay
        // `mana_cost` (the recast is at full cost — `use_warp=false`).
        let in_hand = self.players[idx]
            .hand
            .iter()
            .any(|o| o.id == obj_id);
        let from_exile_warped = !in_hand
            && self.players[idx]
                .exile
                .iter()
                .any(|o| o.id == obj_id && o.warped);
        // **Cast-from-exile-with-cost-hike substrate (Lightstall
        // Inquisitor residue closure).** If the object is in this
        // player's exile with `cast_from_exile_with_lightstall_
        // surcharge = true` (and is not the warp-recast case
        // already), this is a Lightstall-permission cast -- pay
        // `printing.mana_cost + {1}` generic surcharge and clear
        // the flag at cast time. Mirrors Python
        // `actions.py:1467-1510`.
        let from_exile_lightstall = !in_hand
            && !from_exile_warped
            && self.players[idx]
                .exile
                .iter()
                .any(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                });
        let printing_id = if in_hand {
            match self.players[idx]
                .hand
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else if from_exile_warped {
            match self.players[idx]
                .exile
                .iter()
                .find(|o| o.id == obj_id && o.warped)
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else if from_exile_lightstall {
            match self.players[idx]
                .exile
                .iter()
                .find(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                })
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else {
            return Ok(false);
        };
        let cost = match self.registry.lookup(printing_id) {
            Some(p) => {
                if !p.card_types.iter().any(|t| t == "Creature") {
                    return Ok(false);
                }
                if from_exile_lightstall {
                    // Apply +{1} generic surcharge. Mirrors Python
                    // `actions.py:1494-1499`. Colored pips unchanged.
                    crate::cards::ManaCost {
                        generic: p.mana_cost.generic.saturating_add(1),
                        white: p.mana_cost.white,
                        blue: p.mana_cost.blue,
                        black: p.mana_cost.black,
                        red: p.mana_cost.red,
                        green: p.mana_cost.green,
                    }
                } else {
                    p.mana_cost.clone()
                }
            }
            None => return Ok(false),
        };
        // Phase 2: pay the cost. Python's `_cast_spell` pays via
        // `auto_tap_for_cost` (`actions.py:1662`) — floating pool
        // FIRST, then tap lands for the remainder. `auto_tap_pay_for_
        // cost` byte-mirrors that (lands-only, colored-then-generic,
        // BECOMES_TAPPED per land); it commits iff payable, else
        // `false` with NO mutation — the exact bail the prior
        // pool-only `pay_mana_cost` produced, just no longer
        // false-negative when the pool alone is short but the
        // controller's lands cover it (the G7 mid-game STATE_EQ
        // frontier: legal emitted the cast counting auto-tap headroom,
        // apply must actually auto-tap).
        if !auto_tap_pay_for_cost(self, idx, &cost)? {
            return Ok(false);
        }
        // **G7 S5a (Option β): re-home the spell hand/exile → the
        // spell-stack-source-zone store.** Python's `_cast_spell`
        // does `p.hand.remove(card)` (`:1403`) / `p.exile.remove(
        // card)` (`:1501`) then `card.zone = Zone.STACK` (`:1751`) +
        // `card.controller_id = player_id` (`:1752`) — the card
        // leaves every per-player zone and is owned by the stack
        // until resolution. `rehome_spell_to_stack_store` is that
        // exact byte-mirror; `move_source_post_resolve` takes it from
        // the store at resolve (creature → bf + summoning_sick),
        // byte-equal to Python's `put_into_battlefield`. The
        // exile-recast is a fresh permanent (CR 400.7) — clear the
        // warp markers AT CAST TIME on the now-stack-resident object
        // (Python's cast-time `card.warped = False`), found by id in
        // the store (it is no longer in any per-player zone).
        if !rehome_spell_to_stack_store(self, idx, obj_id, player_id) {
            return Ok(false);
        }
        if from_exile_warped {
            if let Some(obj) = self
                .stack_objects
                .iter_mut()
                .find(|o| o.id == obj_id)
            {
                obj.warped = false;
                obj.warp_exile_pending = false;
            }
        }
        if from_exile_lightstall {
            // **Cast-from-exile-with-cost-hike substrate.** Clear the
            // Lightstall flag on the stack-resident source object.
            // Mirrors Python `actions.py:1505` --
            // `card.cast_from_exile_with_lightstall_surcharge =
            // False` at cast time. The card is a fresh object on
            // resolution (CR 400.7) so the flag must not survive.
            if let Some(obj) = self
                .stack_objects
                .iter_mut()
                .find(|o| o.id == obj_id)
            {
                obj.cast_from_exile_with_lightstall_surcharge = false;
            }
        }
        // **MMM Pin 1 (11th Rust slice).** Bump the casting player's
        // `spells_cast_this_turn` counter BEFORE the IS_CAST
        // dispatch. Mirrors Python's `actions._cast_spell` which
        // increments `p.spells_cast_this_turn += 1` at
        // `actions.py:1777` (pre-dispatch). The post-increment value
        // is read by `StatePredicate::SpellsCastThisTurn` ("your
        // second spell each turn" latches when the count reaches 2).
        self.players[player_id as usize].spells_cast_this_turn += 1;
        // Dispatch Event::Cast NOW — after the spell is stack-
        // resident (S5a) but before the StackItem push. CR 601.2 — a
        // spell is "cast" the moment its cost is paid. The subject is
        // the card-being-cast, now in `stack_objects`; the
        // `dispatch_event` observer walk's 6th-zone `stack_objects`
        // sweep finds it (mirrors Python firing IS_CAST with
        // `card.zone == Zone.STACK`, `actions.py:1751`/`:1795`).
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Cast,
            obj_id,
        )?;
        // **G7 S1/S5a: push the spell StackItem, do NOT move to
        // battlefield, do NOT auto-resolve.** The spell's owned
        // GameObject is already in `stack_objects` (S5a); the
        // StackItem stays id-only (Option β — no `state.rs` StackItem
        // shape / fingerprint-slot churn). A vanilla creature spell
        // carries NO cast-time target (Python `_select_spell_targets`
        // returns `()` for a bare permanent spell), so the bare
        // untargeted `new_spell` constructor is used.
        // `resolve_top_of_stack` later applies any `printing.effects`
        // (none for a vanilla creature) and `move_source_post_resolve`
        // sends the creature stack-store → battlefield (CR 608.2m via
        // the ALREADY-PRESENT `post_resolve_destination` permanent ->
        // ZONE_BATTLEFIELD route, stamping `summoning_sick` + the ETB
        // timestamp), then `dispatch_event(Event::Enters)` fires the
        // ETB triggers — all at resolution time, byte-mirroring
        // Python's `put_into_battlefield`.
        self.stack
            .push(crate::state::StackItem::new_spell(obj_id, player_id));
        // CR 117.1c — the active player gets priority after a spell
        // is cast. Verbatim from `cast_spell_noncreature` Delta 7
        // (`:958-962`) / `cast_creature_warp`. The cast push (not the
        // resolution pop) is the reset moment, matching Python: the
        // priority loop resets priority via the post-non-pass-action
        // path (`simulate.py:274-282`) when the cast is applied, then
        // resets again after the eventual resolution
        // (`simulate.py:266-267`). The post-cast reset here is the
        // first of those two; the post-resolution reset is the loop's
        // job (S3 verifies the priority-round composition end-to-end).
        // Also reset `consecutive_passes` — a successful cast is a
        // non-pass mutation that breaks any pass streak.
        self.priority_player = self.active_player;
        self.consecutive_passes = 0;
        Ok(true)
    }

    /// **G7 Slice S4 (stack-and-priority warp-creature cast).**
    /// Warp-cast a creature from `player_id`'s hand OR (if absent
    /// there and the printing is `warp_castable_from_graveyard`) the
    /// GRAVEYARD. Mirrors `cast_creature`'s S1 stack-push shape with
    /// the warp deltas. Verified against the live `cast_creature` +
    /// Python `actions.py:1393-1560` / `put_into_battlefield`.
    ///
    /// **Stack-and-priority semantics (G7 Slice S4 — was atomic).**
    /// Pays `warp_cost`, debits the warp life rider, sets
    /// `state.spell_warped_this_turn = true` (the genuinely cast-time
    /// riders — Python sets all three in `_cast_spell` BEFORE the
    /// StackItem push), stamps `warp_exile_pending = true` on the
    /// stack-resident SOURCE object (NOT on a bf object — there is no
    /// bf object yet), fires `Event::Cast`, pushes a bare
    /// `StackItem::new_spell` (the spell GameObject stays in its
    /// origin zone — Rust's spell-stack model), resets
    /// priority/passes, and **returns WITHOUT resolving**. At
    /// resolution the ALREADY-PRESENT `move_source_post_resolve`
    /// (`state_ops_resolve.rs:285-332`) lands the creature on the
    /// battlefield with `summoning_sick = true` (`:301-303`) + a
    /// fresh ETB timestamp, then `resolve_top_of_stack` dispatches
    /// `Event::Enters` — byte-mirroring Python's resolve-time
    /// `put_into_battlefield` (`actions.py:2993` sets
    /// `summoning_sick`). `take_source` moves the WHOLE owned object
    /// (fields intact), so the cast-time `warp_exile_pending` rides
    /// onto the post-resolution bf object — exactly where the
    /// end-step sweep (`state_ops_steps.rs::
    /// warp_exile_sweep_on_end_step`) keys off it. NO `summoning_sick`
    /// / `warp_exile_pending`-on-a-bf-object stamp here at cast time
    /// (the design's key risk — resolved structurally by riding the
    /// flag on the source object, the direct mirror of S1's
    /// exile-recast zone normalization).
    ///
    /// **Deltas vs `cast_creature` (S1 shape):**
    ///
    /// 0. **Source-zone cascade:** hand first, then graveyard gated
    ///    on `warp_castable_from_graveyard` — mirrors Python's single
    ///    `_cast_spell` `if hand: ... elif gy-warp:` cascade so the
    ///    `Action::CastSpell` dispatch site stays byte-identical.
    /// 1. **Cost source:** pays `printing.warp_cost` (unwrapped — the
    ///    legal emitter only offered `use_warp=true` when this is
    ///    `Some`) instead of `printing.mana_cost`. Mirrors Python's
    ///    `effective_warp_cost(...)[0]` (`actions.py:1399-1400`) / the
    ///    graveyard branch's `card.printing.warp_cost` (`:1448`).
    /// 1b. **Life rider:** after the mana spend commits, debit
    ///    `printing.warp_additional_life_cost` UNCONDITIONALLY (CR
    ///    117.9b — part of the warp alternative cost, no decision
    ///    point). Genuinely cast-time in Python (`:1432`/`:1460`).
    /// 2. **`spell_warped_this_turn`:** set `true` after the mana
    ///    spend commits, mirroring Python `:1427` (hand) / `:1457`
    ///    (graveyard). A `GameState` bool; genuinely cast-time.
    /// 3. **`warp_exile_pending`:** stamp `true` on the STACK-RESIDENT
    ///    SOURCE object (hand arm) or normalize the graveyard-warp
    ///    source graveyard→hand and stamp it there (graveyard arm),
    ///    so it rides through `move_source_post_resolve` onto the
    ///    post-resolution bf object. Set IDENTICALLY for both arms —
    ///    Python schedules the SAME `WarpExileSelf` for both
    ///    (`actions.py:1550-1559`). Rust's model: a deterministic
    ///    phase-entry sweep keyed off the bf object's flag, NOT
    ///    Python's stack-pushed delayed trigger (documented in
    ///    `SEARCH_LIMITATIONS.md`).
    /// 4. **Graveyard-source normalization:** like S1's exile-recast
    ///    (`cast_creature` exile→hand at cast time), the
    ///    graveyard-warp source is normalized graveyard→hand at cast
    ///    time so it byte-matches Python's `p.graveyard.remove(card)`
    ///    + `card.zone = STACK` (`:1450`/`:1751`) and the spell-stack
    ///    source model stays uniform with the hand arm. The
    ///    `move_from_graveyard_to_battlefield` /
    ///    `move_from_exile_to_battlefield` helpers are S4-dead for
    ///    this path (retained `#[allow(dead_code)]`, same treatment
    ///    S1 gave the exile helper).
    /// 5. **Priority tail:** verbatim from `cast_creature` (S1).
    ///
    /// Returns `true` on success, `false` on the same precondition
    /// failures as `cast_creature` plus "not in hand AND (not in
    /// graveyard OR not `warp_castable_from_graveyard`)" / "the
    /// printing has no `warp_cost`" (defensive — the legal emitter
    /// never offers `use_warp=true` without these).
    fn cast_creature_warp(
        &mut self,
        obj_id: u32,
        player_id: u8,
    ) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // **Phase-2 source-zone resolution (mirrors Python's
        // `_cast_spell` `if card in hand: ... elif graveyard-warp:`
        // cascade in ONE function — actions.py:1393/1437).** Look up
        // the object in the HAND first; if absent, fall through to
        // the GRAVEYARD gated on `warp_castable_from_graveyard`. This
        // keeps the `Action::CastSpell` dispatch site byte-identical
        // (no new arm) — exactly as Python keeps the cascade inside
        // `_cast_spell`. `from_graveyard` records which arm fired so
        // the move + provenance match Python's two branches.
        let (printing_id, from_graveyard) = if let Some(pid) = self
            .players[idx]
            .hand
            .iter()
            .find(|o| o.id == obj_id)
            .map(|o| o.printing_id)
        {
            (pid, false)
        } else if let Some(pid) = self.players[idx]
            .graveyard
            .iter()
            .find(|o| o.id == obj_id)
            .map(|o| o.printing_id)
        {
            // Only the graveyard-cast rider authorizes a
            // graveyard-origin warp cast (Python actions.py:1444
            // `grave_card.printing.warp_castable_from_graveyard`).
            let gy_ok = self
                .registry
                .lookup(pid)
                .map(|p| p.warp_castable_from_graveyard)
                .unwrap_or(false);
            if !gy_ok {
                return Ok(false);
            }
            (pid, true)
        } else {
            return Ok(false);
        };
        // Delta 1: pay `warp_cost` (not `mana_cost`). Bail if the
        // printing has no warp (defensive) or isn't a creature.
        let (cost, life_cost) = match self.registry.lookup(printing_id) {
            Some(p) => {
                if !p.card_types.iter().any(|t| t == "Creature") {
                    return Ok(false);
                }
                let wc = match &p.warp_cost {
                    Some(wc) => wc.clone(),
                    None => return Ok(false),
                };
                (wc, p.warp_additional_life_cost)
            }
            None => return Ok(false),
        };
        // Pay the (warp) cost via the auto-tap-from-lands path —
        // Python's `_cast_spell` pays the warp cost through the SAME
        // `auto_tap_for_cost` (`actions.py:1662`, the cost var holds
        // the selected warp cost). Floating pool first, then tap
        // lands; commit iff payable, else `false` no-mutation.
        if !auto_tap_pay_for_cost(self, idx, &cost)? {
            return Ok(false);
        }
        // Delta 2: production write of `spell_warped_this_turn`
        // (mirrors Python actions.py:1427 hand / :1457 graveyard —
        // BOTH branches set it). Plain bool on GameState, no CoW.
        // Set after the mana spend commits.
        self.spell_warped_this_turn = true;
        // **Phase-2: pay the warp ability's "Pay N life" rider**
        // (Timeline Culler). The legal emitter has already gated
        // `life > warp_additional_life_cost`, so this never drops
        // the caster below 1. `life_cost == 0` (every Phase-1/1.5
        // card) is a no-op. Mirrors Python actions.py:1430-1436
        // (hand) / :1458-1464 (graveyard) — debited unconditionally
        // inside the cast (CR 117.9b: the Pay-2-life is part of the
        // warp alternative cost, no decision point). Player.life is
        // i32; widen the u8 rider to match.
        if life_cost > 0 {
            self.players[idx].life -= life_cost as i32;
        }
        // **G7 S5a (Option β): re-home the warp spell hand/graveyard
        // → the spell-stack-source-zone store.** Python's warp arms
        // do `p.hand.remove(card)` (`:1403`) / `p.graveyard.remove(
        // card)` (`:1450`) then `card.zone = Zone.STACK` (`:1751`) +
        // `card.controller_id = player_id` (`:1752`) — the source
        // leaves every per-player zone and is owned by the stack
        // until resolution (byte-identical hand and graveyard arms;
        // nothing on the battlefield yet). `rehome_spell_to_stack_
        // store` is that exact byte-mirror for BOTH arms (it tries
        // hand then exile then graveyard), so the from-graveyard warp
        // (Timeline Culler) and from-hand warp converge on one
        // stack-resident shape. The `move_from_graveyard_to_
        // battlefield` helper stays S4-dead (`#[allow(dead_code)]`).
        if !rehome_spell_to_stack_store(self, idx, obj_id, player_id) {
            return Ok(false);
        }
        let _ = from_graveyard;
        // **G7 S4/S5a Delta 3: stamp `warp_exile_pending` on the
        // STACK-RESIDENT SOURCE object (now in `stack_objects`, NOT a
        // bf object — there is none yet).** `take_source` moves the
        // WHOLE owned GameObject from the store (fields intact,
        // nothing reset) and `move_source_post_resolve` pushes it to
        // the battlefield, so this cast-time flag rides through
        // resolution onto the post-resolution bf object — exactly
        // where the end-step sweep
        // (`state_ops_steps.rs::warp_exile_sweep_on_end_step`) keys
        // off it (the sweep reads bf only, so WHERE the spell sits
        // pre-resolve is irrelevant — the flag rides the owned
        // object). Set IDENTICALLY for the hand AND graveyard arm —
        // Python schedules the SAME `WarpExileSelf` for both
        // (`actions.py:1550-1559` runs for any `use_warp`). The
        // design's "key risk" stays resolved structurally: no new
        // field, no StackItem rider slot — S5a only changes the Vec
        // the source lives in pre-resolve.
        if let Some(obj) = self
            .stack_objects
            .iter_mut()
            .find(|o| o.id == obj_id)
        {
            obj.warp_exile_pending = true;
        }
        // **G7 Class-C delayed-trigger port (round-78).** Schedule the
        // `WarpExileSelf` delayed triggered ability for the warp
        // creature's controller's next END step. Mirrors Python's
        // `actions.py:1550-1559` cast-time schedule. Replaces the
        // round-63 deterministic-sweep path: the `warp_exile_pending`
        // flag above is now the cast-time MARKER (mirroring Python's
        // `cast_via_warp`) that the resolver (`Effect::WarpExileSelf`)
        // clears on resolve; the actual end-step bf→exile move
        // happens through the stack now (response window, CR 603.7).
        //
        // `offset = 0` iff `player_id == active_player`: a warp cast
        // on the caster's own turn (sorcery-speed) fires the trigger
        // at this turn's end step. `offset = 1` for an instant-speed
        // warp on the opponent's turn (no Phase-1 cards admit this,
        // but the offset shape is generalized for forward-
        // correctness). Same shape as Python's `offset = 0 if
        // state.active_player().id == player_id else 1`.
        let offset: u32 = if player_id == self.active_player { 0 } else { 1 };
        self.delayed_triggers.push(crate::cards::DelayedTrigger {
            fire_at_turn: self.turn_number.saturating_add(offset),
            controller_id: player_id,
            source_id: obj_id,
            effect: crate::cards::Effect::WarpExileSelf,
        });
        // **MMM Pin 1 (11th Rust slice).** Bump
        // `spells_cast_this_turn` before the IS_CAST dispatch.
        // Same pre-dispatch position as `cast_creature` /
        // `cast_spell_noncreature` — mirrors Python's
        // `actions._cast_spell` increment at `actions.py:1777`.
        self.players[player_id as usize].spells_cast_this_turn += 1;
        // Dispatch Event::Cast NOW — after the spell is stack-
        // resident (S5a) + the cast-time riders commit, before the
        // StackItem push. The subject is in `stack_objects`; the
        // `dispatch_event` observer walk's 6th-zone sweep finds it
        // (mirrors Python firing IS_CAST with `card.zone == STACK`).
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Cast,
            obj_id,
        )?;
        // **G7 S4: push the spell StackItem, do NOT move to
        // battlefield, do NOT auto-resolve.** Verbatim from
        // `cast_creature` (S1, `:619-620`). A vanilla creature spell
        // (warp cards: Timeline Culler etc. have
        // `triggered_abilities=()`) carries NO cast-time target, so
        // the bare untargeted `new_spell` constructor is used.
        // `resolve_top_of_stack` later runs `move_source_post_resolve`
        // (creature hand -> battlefield, stamping `summoning_sick` +
        // the ETB timestamp; the `warp_exile_pending` flag rides
        // along on the owned object) then `dispatch_event(Enters)` —
        // all at resolution, byte-mirroring Python's
        // `put_into_battlefield`.
        self.stack
            .push(crate::state::StackItem::new_spell(obj_id, player_id));
        // Delta 5: priority/passes reset, verbatim from `cast_creature`
        // (S1, `:633-634`). The cast push (not the resolution pop) is
        // the reset moment, matching Python's post-non-pass-action
        // priority reset (`simulate.py:274-282`); the post-resolution
        // reset is the priority loop's job.
        self.priority_player = self.active_player;
        self.consecutive_passes = 0;
        Ok(true)
    }

    /// **Spell-card-bridge lever, Phase 1 (round-70).** Cast a
    /// non-creature Instant/Sorcery from `player_id`'s hand. Sibling
    /// of `cast_creature` / `cast_creature_warp` with these deltas
    /// (verified against both live bodies):
    ///
    /// 1. **Hand lookup only** (no exile-recast branch — that is
    ///    warp-specific). Mirrors `cast_creature_warp`'s simple
    ///    `hand.iter().find(...).map(|o| o.printing_id)` shape.
    /// 2. **Type gate inverted:** require `"Instant"` OR `"Sorcery"`
    ///    in `card_types` AND NOT `"Creature"` (the dual of
    ///    `cast_creature`'s `:444` Creature gate). Same pre-mutation
    ///    `Ok(false)` bail shape so a non-spell never has its mana
    ///    speculatively debited. Returning `Ok(false)` for a creature
    ///    lets the `Action::CastSpell` dispatch try this primitive
    ///    first-after-warp then fall through to `cast_creature`.
    /// 3. **Pay `printing.mana_cost`** (the guardrail rejected
    ///    X / kicker / warp / additional-cost-choice, so `mana_cost`
    ///    is the whole cost). Exact `cast_creature:453-457` shape.
    /// 4. **Push a spell `StackItem`, do NOT move to battlefield.**
    ///    The spell GameObject stays in the hand zone (Rust's
    ///    pre-existing spell-stack model — `find_source_printing_id` /
    ///    `take_source` search hand; the existing
    ///    `mass_removal_side_channel` / Doom-Blade resolve path relies
    ///    on this). `resolve_top_of_stack` then applies
    ///    `printing.effects` and `move_source_post_resolve` takes the
    ///    card from hand → graveyard via the ALREADY-PRESENT
    ///    Instant/Sorcery → ZONE_GRAVEYARD route
    ///    (`post_resolve_destination:251-256`, UNCHANGED — the
    ///    spell→graveyard move is FREE; CR 608.2m already handled).
    /// 5. **Bind the single target at cast time** via the same
    ///    deterministic-first scan the per-card bespoke Effect resolve
    ///    arms apply (the no-callback first-legal-target campaign
    ///    convention; mirrors Python's `_select_spell_targets`
    ///    binding ONE target onto the StackItem at cast time). Stored
    ///    in `StackItem.target_obj_id` (Gate B'', §3.4). The closed
    ///    Tier-A spell arms self-scan the battlefield and ignore the
    ///    payload target, so the swap is a correct no-op for them
    ///    (byte-equality is delivered by self-scan == Python
    ///    first-legal, proven by the Tier-A real-EoE probes); the
    ///    field is the forward substrate for targeted arms that opt
    ///    into the payload-target convention.
    /// 6. **Fire the `Event::Cast` dispatch** before the stack push,
    ///    same ordering as `cast_creature:470-474` /
    ///    `cast_creature_warp:585-589`.
    /// 7. **Resolve is NOT fused.** The primitive pushes the
    ///    StackItem and returns; the harness drives
    ///    `resolve_top_of_stack` separately (CE5 cast-vs-resolve
    ///    mirror — `step_both` drains both stacks before the
    ///    field-view compare, so the mid-stack card-in-hand asymmetry
    ///    vs Python is never compared; the compare is post-resolve).
    ///
    /// Returns `true` on success, `false` on the same precondition
    /// failures as `cast_creature` (object not in hand, no such
    /// printing, insufficient mana) PLUS "the printing is not a
    /// non-creature Instant/Sorcery".
    fn cast_spell_noncreature(
        &mut self,
        obj_id: u32,
        player_id: u8,
    ) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Delta 1: hand lookup, with a fallback to the Lightstall
        // exile zone for noncreature spells cast via the
        // cast-from-exile-with-cost-hike substrate (Lightstall
        // Inquisitor residue closure). Mirrors `cast_creature`'s
        // hand-or-exile cascade; the warp exile-recast path is
        // creature-only on the live surface so it does NOT
        // duplicate here.
        let in_hand = self.players[idx]
            .hand
            .iter()
            .any(|o| o.id == obj_id);
        let from_exile_lightstall = !in_hand
            && self.players[idx]
                .exile
                .iter()
                .any(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                });
        let printing_id = if in_hand {
            match self.players[idx]
                .hand
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else if from_exile_lightstall {
            match self.players[idx]
                .exile
                .iter()
                .find(|o| {
                    o.id == obj_id
                        && o.cast_from_exile_with_lightstall_surcharge
                })
                .map(|o| o.printing_id)
            {
                Some(pid) => pid,
                None => return Ok(false),
            }
        } else {
            return Ok(false);
        };
        // Delta 2: inverted type gate — Instant/Sorcery AND NOT
        // Creature. Bail (no mutation) on anything else so the
        // dispatch can fall through to `cast_creature`.
        let cost = match self.registry.lookup(printing_id) {
            Some(p) => {
                let is_spell = p
                    .card_types
                    .iter()
                    .any(|t| t == "Instant" || t == "Sorcery");
                let is_creature = p
                    .card_types
                    .iter()
                    .any(|t| t == "Creature");
                if !is_spell || is_creature {
                    return Ok(false);
                }
                if from_exile_lightstall {
                    // Apply +{1} generic surcharge (Lightstall path).
                    // Mirrors Python `actions.py:1494-1499`.
                    crate::cards::ManaCost {
                        generic: p.mana_cost.generic.saturating_add(1),
                        white: p.mana_cost.white,
                        blue: p.mana_cost.blue,
                        black: p.mana_cost.black,
                        red: p.mana_cost.red,
                        green: p.mana_cost.green,
                    }
                } else {
                    p.mana_cost.clone()
                }
            }
            None => return Ok(false),
        };
        // Delta 3: pay `mana_cost` via the auto-tap-from-lands path
        // (Python's instant/sorcery branch of `_cast_spell` pays
        // through the SAME `auto_tap_for_cost`, `actions.py:1662`).
        // Floating pool first, then tap lands; commit iff payable,
        // else `false` no-mutation.
        if !auto_tap_pay_for_cost(self, idx, &cost)? {
            return Ok(false);
        }
        // **G7 S5a (Option β): re-home the spell hand → the
        // spell-stack-source-zone store.** A noncreature spell is
        // cast from hand only (the Delta-1 lookup gated on hand);
        // Python does `p.hand.remove(card)` (`:1403`) + `card.zone =
        // Zone.STACK` (`:1751`) + `card.controller_id = player_id`
        // (`:1752`). `rehome_spell_to_stack_store` is that exact
        // byte-mirror. At resolve `move_source_post_resolve` sends
        // the instant/sorcery stack-store → graveyard (CR 608.2m via
        // `post_resolve_destination`'s Instant/Sorcery → GY route),
        // byte-equal to Python's resolve cleanup.
        if !rehome_spell_to_stack_store(self, idx, obj_id, player_id) {
            return Ok(false);
        }
        if from_exile_lightstall {
            // **Cast-from-exile-with-cost-hike substrate.** Clear the
            // Lightstall flag on the stack-resident source object.
            // Mirrors Python `actions.py:1505`. CR 400.7 -- the card
            // is a fresh object on resolution; the flag must not
            // survive.
            if let Some(obj) = self
                .stack_objects
                .iter_mut()
                .find(|o| o.id == obj_id)
            {
                obj.cast_from_exile_with_lightstall_surcharge = false;
            }
        }
        // **MMM Pin 1 (11th Rust slice).** Bump
        // `spells_cast_this_turn` before the IS_CAST dispatch.
        // Same pre-dispatch position as `cast_creature` /
        // `cast_creature_warp` — mirrors Python's
        // `actions._cast_spell` increment at `actions.py:1777`.
        self.players[player_id as usize].spells_cast_this_turn += 1;
        // Delta 6: CAST event after the spell is stack-resident (S5a)
        // and before the stack push, same ordering as `cast_creature`
        // / `cast_creature_warp`. The subject is in `stack_objects`;
        // the `dispatch_event` observer walk's 6th-zone sweep finds
        // it (mirrors Python firing IS_CAST with `card.zone ==
        // Zone.STACK`).
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Cast,
            obj_id,
        )?;
        // Delta 5: bind the deterministic-first legal target(s).
        //
        // **Phase 1 (single-target, `target_slot_kinds()==()`):** the
        // first creature on EITHER battlefield in `bf_iter_etb_sorted`
        // order (Python's centralized `state.battlefield` walk + the
        // deterministic-first convention). The closed Tier-A spell
        // arms self-scan and ignore it; it is the forward substrate
        // for payload-target-reading arms.
        //
        // **Phase 2 (2-slot heterogeneous grammar — Diplomatic
        // Relations):** the printing's `effects[0]` is the 2-slot
        // effect. Python's `_select_spell_targets` iterates
        // `target_slot_kinds()` and, per slot, filters the flat
        // `legal_targets` to the slot's kind and picks the
        // first-legal (the `_PickFirstLegalTargetAgent` bench/probe
        // convention). For Diplomatic Relations the kinds are
        // `("my_creature","creature")` and `legal_targets` is
        // `mine ++ theirs` in `state.battlefield` order, so:
        //   slot 0 = first creature the caster controls,
        //   slot 1 = first creature a NON-caster controls.
        // We re-derive that EXACT pair here off `bf_iter_etb_sorted`
        // (the cross-engine-proven equivalent of Python's
        // `state.battlefield` ordering — the same equivalence the
        // Phase-1 Depressurize/Focus-Fire real-EoE probes pinned),
        // partitioned by `o.controller == player_id`. Bound at cast
        // time so the picks are fixed on the StackItem (matching
        // Python binding two targets onto the StackItem at cast
        // time). The bare-single-target detection stays the default:
        // a 2-slot effect is detected structurally via
        // `Effect::DiplomaticRelationsEffect` (the ONLY 2-slot
        // grammar admitted today; the guardrail
        // `_printing_is_bridge_simple_spell` Phase-2 widening only
        // lets this exact shape through).
        let is_two_slot = self
            .registry
            .lookup(printing_id)
            .map(|p| {
                matches!(
                    p.effects.first(),
                    Some(crate::cards::Effect::DiplomaticRelationsEffect)
                )
            })
            .unwrap_or(false);
        // G7 lander-controller-mismatch (token-creation parity): the
        // first-effect SHAPE drives which target-kind to bind at cast
        // time. Most Phase-1 single-target spells take a creature, so
        // the default scans for a Creature. But `SeedshipImpactEffect`
        // takes an artifact-OR-enchantment (Python
        // `SeedshipImpactEffect.legal_targets` filters on
        // `is_artifact OR "Enchantment" in card_types`, non-
        // indestructible — `effects/bespoke.py:1227-1235`). Without
        // this dedicated arm, `bound_target` is `None` and the
        // resolve arm's self-scan picks a STALE first-artifact when
        // the bound target ceases between cast and resolve — exactly
        // the G7 S6 arch=10/11/12 seed=2 divergence shape. The
        // verify-first repro: P0 casts Orbital Plunge (Lander-rider
        // creates Lander0 under P0), P0 casts Seedship Impact
        // targeting Lander0, P1 casts Seedship Impact targeting
        // Lander0. P1's resolves first (LIFO), destroys Lander0, mv≤2
        // → P1 mints Lander1 (correct). P0's resolves: Python
        // fizzles (bound target Lander0 is gone); Rust pre-fix
        // self-scans, picks Lander1, destroys it and mints Lander2
        // under P0 — diverges from Python by both controller AND id.
        // The fix binds the artifact-or-enchantment target at cast
        // time on the StackItem; the resolve arm reads
        // `current_resolve_payload_target_id` and fizzles on stale.
        let is_seedship_impact = self
            .registry
            .lookup(printing_id)
            .map(|p| {
                matches!(
                    p.effects.first(),
                    Some(crate::cards::Effect::SeedshipImpactEffect { .. })
                )
            })
            .unwrap_or(false);
        // Emergency Eject: takes any non-land permanent (Python's
        // `DestroyTargetNonlandPermanentWithLanderForController
        // .legal_targets` `effects/generic/destroy.py:602-609`).
        // Same cast-time-bind shape as Seedship — the resolve arm
        // reads `current_resolve_payload_target_id` and fizzles on
        // stale (CR 608.2b). Without this dedicated arm, the
        // default first-creature scan would mis-pick when the spell
        // targets a non-creature permanent (artifact / enchantment
        // / planeswalker).
        let is_emergency_eject = self
            .registry
            .lookup(printing_id)
            .map(|p| {
                matches!(
                    p.effects.first(),
                    Some(crate::cards::Effect::EmergencyEjectEffect { .. })
                )
            })
            .unwrap_or(false);
        // **Battlefield STATE_EQ axis closure (todo `e5b7c9d8`).**
        // Per-effect cast-time binds for the two Effect arms migrated
        // to the `current_resolve_payload_target_id` pattern in
        // `state_ops_resolve.rs` (Radiant Strike's destroy half +
        // Focus Fire's damage half). The resolve arms now read the
        // bound target id and fizzle on stale/illegal targets per CR
        // 608.2b — so the cast-time bind MUST pick a target matching
        // the effect's `legal_targets` filter (Python parity), not
        // the default "first creature".
        //
        // Radiant Strike: spell_effect is
        // `Sequence([DestroyTargetArtifactOrTappedCreature, GainLife(3)])`,
        // so the head effect lives inside a Sequence. Peek through.
        // Filter = `IsArtifact() | (IsCreature() & IsTapped())`, non-
        // indestructible. Mirrors
        // `state_ops_resolve.rs::Effect::DestroyTargetArtifactOrTapped
        // Creature` + Python's `effects/generic/destroy.py:487-493`
        // legal_targets.
        let is_radiant_strike_destroy = self
            .registry
            .lookup(printing_id)
            .map(|p| {
                let head = match p.effects.first() {
                    Some(crate::cards::Effect::Sequence { effects }) => {
                        effects.first()
                    }
                    other => other,
                };
                matches!(
                    head,
                    Some(crate::cards::Effect::DestroyTargetArtifactOrTappedCreature)
                )
            })
            .unwrap_or(false);
        // Focus Fire: top-level `FocusFireEffect { base }` (no
        // Sequence wrapper). Filter = `IsCreature() & (IsAttacking()
        // | IsBlocking())`. Mirrors
        // `state_ops_resolve.rs::Effect::FocusFireEffect` + Python's
        // `effects/bespoke.py:1771-1782` legal_targets.
        let is_focus_fire = self
            .registry
            .lookup(printing_id)
            .map(|p| {
                matches!(
                    p.effects.first(),
                    Some(crate::cards::Effect::FocusFireEffect { .. })
                )
            })
            .unwrap_or(false);
        let (bound_target, bound_target_2): (Option<u32>, Option<u32>) =
            if is_two_slot {
                // 2-slot: slot 0 = first caster-controlled creature,
                // slot 1 = first non-caster-controlled creature, both
                // in `bf_iter_etb_sorted` order.
                let mut slot0: Option<u32> = None;
                let mut slot1: Option<u32> = None;
                for o in crate::bf_iter::bf_iter_etb_sorted(self) {
                    let is_creature = self
                        .registry
                        .lookup(o.printing_id)
                        .map(|p| {
                            p.card_types.iter().any(|t| t == "Creature")
                        })
                        .unwrap_or(false);
                    if !is_creature {
                        continue;
                    }
                    if o.controller == player_id {
                        if slot0.is_none() {
                            slot0 = Some(o.id);
                        }
                    } else if slot1.is_none() {
                        slot1 = Some(o.id);
                    }
                    if slot0.is_some() && slot1.is_some() {
                        break;
                    }
                }
                (slot0, slot1)
            } else if is_radiant_strike_destroy {
                // Radiant Strike's destroy half: first artifact-OR-
                // tapped-creature on EITHER bf in
                // `bf_iter_etb_sorted` order, non-indestructible
                // (matches the resolve arm's filter + Python
                // `legal_targets` filter exactly). The deterministic-
                // first pick mirrors the
                // `_PickFirstLegalTargetAgent` bake-in convention the
                // Python SoT uses for `choose_target`.
                let t = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        let printing =
                            self.registry.lookup(o.printing_id)?;
                        let is_artifact = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_creature = printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        if !(is_artifact || (is_creature && o.tapped)) {
                            return None;
                        }
                        if printing
                            .keywords
                            .iter()
                            .any(|k| k == "indestructible")
                        {
                            return None;
                        }
                        Some(o.id)
                    });
                (t, None)
            } else if is_focus_fire {
                // Focus Fire: first attacking-or-blocking creature on
                // EITHER bf in `bf_iter_etb_sorted` order (matches
                // the resolve arm's filter + Python's `legal_targets`
                // exactly). Combat predicate: id in
                // `state.attackers` OR id is a `blocker_id` in
                // `state.blockers`.
                let t = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        let printing =
                            self.registry.lookup(o.printing_id)?;
                        if !printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature")
                        {
                            return None;
                        }
                        let is_attacking =
                            self.attackers.contains(&o.id);
                        let is_blocking = self
                            .blockers
                            .iter()
                            .any(|(_att, blk)| *blk == o.id);
                        if is_attacking || is_blocking {
                            Some(o.id)
                        } else {
                            None
                        }
                    });
                (t, None)
            } else if is_seedship_impact {
                // Seedship Impact: first artifact-OR-enchantment on
                // EITHER bf in `bf_iter_etb_sorted` order, non-
                // indestructible (matches the resolve arm's scan
                // filter + Python `legal_targets` filter exactly).
                // The deterministic-first pick mirrors the
                // `_PickFirstLegalTargetAgent` bake-in convention the
                // Python SoT uses for `choose_target`.
                let t = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        let printing =
                            self.registry.lookup(o.printing_id)?;
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
                        Some(o.id)
                    });
                (t, None)
            } else if is_emergency_eject {
                // Emergency Eject: first non-land permanent on
                // EITHER bf in `bf_iter_etb_sorted` order, non-
                // indestructible (matches the resolve arm's scan
                // filter + Python `legal_targets` exactly). The
                // _PickFirstLegalTargetAgent bake-in convention.
                let t = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        let printing =
                            self.registry.lookup(o.printing_id)?;
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
                        Some(o.id)
                    });
                (t, None)
            } else {
                // Phase-1 single-target: first creature on EITHER bf.
                let t = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        let printing =
                            self.registry.lookup(o.printing_id)?;
                        if printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature")
                        {
                            Some(o.id)
                        } else {
                            None
                        }
                    });
                (t, None)
            };
        // **ModalChooseOne envelope — Cycle 1 (cast-time mode
        // binding, 2026-05-21).** Detect whether the spell's first
        // effect is an `Effect::ChooseOneMode { modes }` and, if so,
        // bind the deterministic-first mode pick (`Some(0)`) onto
        // the StackItem. Mirrors Python's
        // `actions._choose_modes_at_cast` no-agent fallback at
        // `actions.py:1222-1223` (returns `(0,)` when
        // `p.agent is None`) — the harness convention used by every
        // cross-engine probe and the bench's
        // `_PickFirstLegalTargetAgent`. The bind is read at resolve
        // time by the Cycle-2 widening of the
        // `Effect::ChooseOneMode` arm; Cycle 1 ships only the
        // cast-time pick + the round-trip-on-StackItem.
        //
        // **First-slice scope.** Single-pick only (Python's no-
        // `or_both` branch). The detection looks at the top-level
        // first effect — NO Sequence-peek today. Modal cards in
        // EoE (per `cards.rs:4296` doc: 13 EoE cards) print their
        // ChooseOneMode at the spell-effect head; a wrapped form
        // (Sequence([ChooseOneMode(...), Other])) would not be
        // detected and would fall through to the resolve-time
        // PendingDecision path — fine, since no EoE card today has
        // that wrapped shape.
        //
        // **Why `Some(0)` and not via the agent.** Rust's bench and
        // cross-engine harness drive deterministic-first agents
        // (the `_PickFirstLegalTargetAgent` campaign-wide
        // convention); a real agent-driven mode pick would require
        // a Decision-style callback (parallel to
        // `submit_choose_mode_decision` for the resolve-time path)
        // — explicitly out of Cycle-1 scope. The deterministic-
        // first pick matches Python's no-agent fallback byte-equal.
        let bound_mode_idx: Option<u8> = self
            .registry
            .lookup(printing_id)
            .and_then(|p| match p.effects.first() {
                Some(crate::cards::Effect::ChooseOneMode { .. }) => Some(0u8),
                _ => None,
            });
        // **scour-for-scrap-slice Cycle 1 widen (2026-05-21).** When
        // the cast spell's `Effect::ChooseOneMode` carries
        // `or_both: true`, bind the FULL mode-subset (one entry per
        // mode index, in printed order) onto the StackItem's
        // `chosen_mode_idxs` slot — the multi-pick sibling of
        // `chosen_mode_idx`. Mirrors Python's no-agent fallback for
        // the or_both branch: pick ALL modes (Scour for Scrap ALWAYS
        // lets you do both, so picking both is the deterministic-
        // first answer; a real agent-driven sub-subset selection is
        // a Cycle-2 follow-up requiring a Decision callback).
        //
        // **Mutual-exclusion contract.** When `bound_mode_idxs` is
        // `Some`, the multi-pick path is active and the single-pick
        // `bound_mode_idx` is implicitly redundant (the resolve arm
        // reads multi first); we still keep `bound_mode_idx`
        // populated so the existing per-mode cast-time target binder
        // (the `chosen_mode` peek below) operates on the FIRST mode
        // for its target-slot scan — the only target slot bound
        // today, since the per-leaf `ReturnTargetArtifactCard...`
        // detector + admit-set flip ship in a follow-up cycle.
        let bound_mode_idxs: Option<Vec<u8>> = self
            .registry
            .lookup(printing_id)
            .and_then(|p| match p.effects.first() {
                Some(crate::cards::Effect::ChooseOneMode {
                    modes,
                    or_both: true,
                }) => {
                    // Pick all modes by default (no-agent deterministic
                    // fallback). The vec is `[0, 1, ..., modes.len()-1]`
                    // — Python's `[i for i in range(len(modes))]`
                    // shape under the no-agent path.
                    let idxs: Vec<u8> = (0..modes.len() as u8).collect();
                    Some(idxs)
                }
                _ => None,
            });
        // **ModalChooseOne envelope — Cycle 5 (per-mode target
        // binding, 2026-05-21).** When the cast spell is modal AND
        // the chosen mode is a recognized single-target leaf, bind
        // the deterministic-first legal target for THAT mode's
        // filter onto `bound_target`. Mirrors Python's
        // `_select_spell_targets` which, after `_choose_modes_at_cast`,
        // walks the chosen mode's `target_slot_kinds()` /
        // `legal_targets` and picks first-legal per slot
        // (`actions.py:1337-1357`).
        //
        // **Why this lives AFTER the non-modal `bound_target`
        // computation.** The non-modal cast-time-bind block above
        // (the `is_two_slot` / `is_radiant_strike_destroy` /
        // `is_focus_fire` / `is_seedship_impact` / `is_emergency_eject`
        // / default-first-creature ladder) detects whether the
        // TOP-LEVEL `effects[0]` matches a known targeted shape. For
        // a modal spell, `effects[0]` is `Effect::ChooseOneMode { .. }`
        // — NONE of those is_* arms match, so the ladder falls
        // through to the default-first-creature scan, which binds
        // the FIRST creature regardless of whether the chosen
        // mode actually targets a creature. That binding is
        // load-bearing-WRONG for modal arms whose chosen leaf has a
        // non-creature filter (e.g. mode=DestroyTargetArtifact would
        // need a non-indestructible artifact, not the first
        // creature). The Cycle 5 widen OVERRIDES `bound_target` to
        // the per-mode filter's first-legal scan when modal AND the
        // mode's leaf is in the recognized set.
        //
        // **First-slice detector table (per-leaf shape).** Mirrors
        // the existing `is_*` cast-time-bind detectors at the top of
        // this function: pattern-match the chosen mode's Effect
        // variant, run the matching scan, return its first-legal id.
        // The table starts narrow — one entry for
        // `DestroyTargetArtifact` (Drill Too Deep mode 1) — and
        // grows per-mode-leaf as the per-leaf Effect arms ship in
        // Cycle 6+. An unrecognized chosen mode falls through to
        // the default-first-creature bound from the ladder above;
        // that's safe for the substrate-only landing because the
        // admit predicate `_printing_is_bridge_simple_spell` STILL
        // REJECTS every some-targeted modal at line 4713 (the
        // target-grammar gate accepts only PHASE1/PHASE2/PHASE3
        // shapes, not modal `(1, 1)` / `(2, 2)` / `(0, 1)` shapes).
        // Cycle 6 widens the admit predicate AS the matching
        // per-leaf entries get added to this table — keeping the
        // "byte-equal-or-don't-admit" no-fallback-hybrid discipline.
        let bound_target: Option<u32> = if let Some(idx_u8) = bound_mode_idx {
            // Modal: peek into the chosen mode (or, when `or_both`,
            // the FIRST chosen mode whose leaf has a recognized
            // per-mode binder entry) and try to bind via per-mode
            // filter. Falls back to whatever the non-modal ladder
            // bound if the chosen mode is not in the recognized leaf
            // table (back-compat: non-overriding default).
            //
            // **scour-for-scrap-cycle-2 (2026-05-21) — or_both walk.**
            // When `bound_mode_idxs` is `Some` (the or_both=True
            // multi-pick path), walk every chosen mode index in the
            // supplied order. The peeked mode for the detector match
            // is the FIRST chosen mode that IS a recognized targeted
            // leaf (uses a small allowlist of known targeted Effect
            // variants). If no chosen mode matches, fall through to
            // the single-pick `bound_mode_idx` peek behavior.
            //
            // This admits Scour for Scrap's shape: mode 0
            // (`SearchLibraryForCard`) is untargeted and skipped by
            // the walk; mode 1
            // (`ReturnTargetArtifactCardFromYourGraveyardToHand`)
            // IS a recognized targeted leaf and its scan emits the
            // cast-time-bound graveyard-artifact id. For the single-
            // pick path (`bound_mode_idxs` is `None`) the existing
            // single-mode peek behavior is preserved byte-identical —
            // `bound_mode_idx` points at the chosen mode and the
            // detector match is run once for that mode.
            //
            // The allowlist of "recognized targeted leaves" mirrors
            // the detector arms in the match below — we skip a
            // chosen mode whose Effect variant is NOT in that list
            // so an untargeted mode (e.g. `SearchLibraryForCard`,
            // `DrawCards`) is skipped, NOT picked as the peek.
            fn is_recognized_targeted_mode_leaf(e: &crate::cards::Effect) -> bool {
                use crate::cards::Effect;
                matches!(
                    e,
                    Effect::DestroyTargetArtifact
                        | Effect::PumpTargetArtifactOrCreatureUntilEOT { .. }
                        | Effect::DealDamageToTargetTappedCreature { .. }
                        | Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { .. }
                        | Effect::ReturnTargetArtifactCardFromYourGraveyardToHand
                )
            }
            let chosen_mode: Option<crate::cards::Effect> = self
                .registry
                .lookup(printing_id)
                .and_then(|p| match p.effects.first() {
                    Some(crate::cards::Effect::ChooseOneMode { modes, or_both: _ }) => {
                        // Multi-pick (`or_both`) path: walk
                        // `bound_mode_idxs` in supplied order, peek
                        // the FIRST chosen mode whose Effect variant
                        // is a recognized targeted leaf. Skips
                        // untargeted modes (e.g. Scour mode 0
                        // `SearchLibraryForCard`).
                        if let Some(idxs) = &bound_mode_idxs {
                            for i in idxs {
                                if let Some(m) = modes.get(*i as usize) {
                                    if is_recognized_targeted_mode_leaf(m) {
                                        return Some(m.clone());
                                    }
                                }
                            }
                            // No chosen mode is a recognized
                            // targeted leaf — fall through to the
                            // single-pick peek behavior below.
                        }
                        // Single-pick fallback (back-compat byte-
                        // identical for every non-or_both modal).
                        let idx = (idx_u8 as usize).min(
                            modes.len().saturating_sub(1),
                        );
                        modes.get(idx).cloned()
                    }
                    _ => None,
                });
            match chosen_mode.as_ref() {
                // `DestroyTargetArtifact`: first non-indestructible
                // artifact on EITHER bf in `bf_iter_etb_sorted` order.
                // Filter mirrors the resolve arm at
                // `state_ops_resolve.rs:6919-6931` exactly
                // (Artifact + not Indestructible).
                Some(crate::cards::Effect::DestroyTargetArtifact) => {
                    crate::bf_iter::bf_iter_etb_sorted(self)
                        .into_iter()
                        .find_map(|o| {
                            let p = self.registry.lookup(o.printing_id)?;
                            if !p.card_types.iter().any(|t| t == "Artifact") {
                                return None;
                            }
                            if p.keywords.iter().any(|k| k == "indestructible") {
                                return None;
                            }
                            Some(o.id)
                        })
                }
                // **Reroute Systems closure (2026-05-21) — mode 0:
                // `PumpTargetArtifactOrCreatureUntilEOT`.** First
                // artifact-or-creature on EITHER bf in
                // `bf_iter_etb_sorted` order. NO controller filter
                // — the printed oracle "target artifact or creature"
                // is ANY-permanent (Python SoT `Pump.resolve` /
                // `PumpTargetArtifactOrCreatureUntilEOT.__post_init__`
                // walks any permanent, no controller narrowing). The
                // filter is BROADER than `DestroyTargetArtifact` (no
                // indestructible reject — pump targets DO include
                // indestructible permanents; the granted keyword is
                // additive, not destructive).
                Some(crate::cards::Effect::PumpTargetArtifactOrCreatureUntilEOT { .. }) => {
                    crate::bf_iter::bf_iter_etb_sorted(self)
                        .into_iter()
                        .find_map(|o| {
                            let p = self.registry.lookup(o.printing_id)?;
                            let is_artifact = p.card_types.iter().any(|t| t == "Artifact");
                            let is_creature = p.card_types.iter().any(|t| t == "Creature");
                            if is_artifact || is_creature {
                                Some(o.id)
                            } else {
                                None
                            }
                        })
                }
                // **Reroute Systems closure (2026-05-21) — mode 1:
                // `DealDamageToTargetTappedCreature`.** First TAPPED
                // creature on EITHER bf in `bf_iter_etb_sorted`
                // order. NO controller filter — the printed oracle
                // "target tapped creature" is ANY-side. Filter
                // mirrors Python's
                // `DealDamageToTargetTappedCreature.__post_init__`
                // TargetSpec(`IsCreature() & IsTapped()`) exactly.
                // The at-resolve recheck (per CR 608.2b) is enforced
                // in the resolve arm (`state_ops_resolve.rs`); the
                // cast-time bind picks a target that IS tapped right
                // now, the resolve recheck handles "untapped between
                // cast and resolve".
                Some(crate::cards::Effect::DealDamageToTargetTappedCreature { .. }) => {
                    crate::bf_iter::bf_iter_etb_sorted(self)
                        .into_iter()
                        .find_map(|o| {
                            if !o.tapped {
                                return None;
                            }
                            let p = self.registry.lookup(o.printing_id)?;
                            if p.card_types.iter().any(|t| t == "Creature") {
                                Some(o.id)
                            } else {
                                None
                            }
                        })
                }
                // **Cycle 6 — Drill Too Deep mode 0
                // (2026-05-21).** `PutChargeCountersOnTarget
                // SpacecraftOrPlanetYouControl`: first permanent on
                // EITHER bf in `bf_iter_etb_sorted` order whose
                // printing has `subtypes ∋ "Spacecraft"` OR
                // `subtypes ∋ "Planet"` AND whose controller is the
                // caster. Filter mirrors the resolve arm at
                // `state_ops_resolve.rs` `PutChargeCountersOnTarget
                // SpacecraftOrPlanetYouControl` and Python's
                // `effects/generic/counters.py:508-516`
                // `legal_targets` exactly. The
                // source-controller-restricted scan matches Python's
                // `_select_spell_targets` per-mode first-legal
                // convention (`actions.py:1337-1357`).
                Some(crate::cards::Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { .. }) => {
                    let caster = player_id as u8;
                    crate::bf_iter::bf_iter_etb_sorted(self)
                        .into_iter()
                        .find_map(|o| {
                            if o.controller != caster {
                                return None;
                            }
                            let p = self.registry.lookup(o.printing_id)?;
                            let is_spacecraft = p
                                .subtypes
                                .iter()
                                .any(|s| s == "Spacecraft");
                            let is_planet = p
                                .subtypes
                                .iter()
                                .any(|s| s == "Planet");
                            if !(is_spacecraft || is_planet) {
                                return None;
                            }
                            Some(o.id)
                        })
                }
                // **scour-for-scrap-cycle-2 (2026-05-21) — Scour for
                // Scrap mode 1: `ReturnTargetArtifactCardFromYour
                // GraveyardToHand`.** First artifact card in the
                // CASTER's graveyard (insertion order). Filter
                // mirrors the resolve arm at
                // `state_ops_resolve.rs::ReturnTargetArtifactCardFrom
                // YourGraveyardToHand` exactly:
                // `printing.card_types.iter().any(|t| t ==
                // "Artifact")` against the caster's graveyard. No
                // indestructible reject (return-to-hand bypasses the
                // indestructible filter — moving to hand is a zone
                // change, not destruction). Source-controller scope
                // mirrors Python's `ReturnFromGraveyard.legal_targets`
                // walking `state.player_by_id(controller_id).graveyard`.
                Some(crate::cards::Effect::ReturnTargetArtifactCardFromYourGraveyardToHand) => {
                    let caster = player_id as usize;
                    self.players[caster]
                        .graveyard
                        .iter()
                        .find_map(|obj| {
                            let p = self.registry.lookup(obj.printing_id)?;
                            if p.card_types.iter().any(|t| t == "Artifact") {
                                Some(obj.id)
                            } else {
                                None
                            }
                        })
                }
                // Any other chosen mode (or no recognized leaf): keep
                // the default `bound_target` from the ladder above.
                // The admit predicate keeps modal spells declined
                // until Cycle 6 adds the matching per-leaf entry +
                // the admit-set widen.
                _ => bound_target,
            }
        } else {
            // Non-modal spell: keep the non-modal ladder's binding
            // (back-compat byte-identical — this branch is the
            // pre-Cycle-5 behavior).
            bound_target
        };
        // **Close Encounter additional-cost-choice envelope — Cycle 2
        // (cast-pipeline binding, 2026-05-21).** When the cast spell's
        // printing carries `additional_cost_choose_your_creature =
        // true` (today: Close Encounter; the `printing.additional_
        // cost_choose_your_creature` flag is the gate), scan the
        // caster's eligible objects for the deterministic-first
        // legal choice and bind onto the StackItem. Mirrors Python's
        // no-agent fallback at `actions.py:1737-1745` exactly:
        //   - Battlefield-creatures-you-control come first (in
        //     `bf_iter_etb_sorted` order, restricted to
        //     `o.controller == player_id` AND printing-is-creature).
        //   - Then warped creature CARDS in the caster's exile (warp-
        //     exile cards have `o.warped == true` set by the warp
        //     ETB/cleanup path; the printing must still be a Creature).
        // The first eligible id wins (Python's `options[0].id` no-
        // agent fallback). `None` for every non-additional-cost
        // spell (the flag short-circuits — back-compat byte-equal).
        //
        // **Cycle 2 scope.** This wires the cast-time bind onto the
        // StackItem; the resolve-arm consumer is Cycle 3 (the
        // `Effect::CloseEncounterEffect` arm reads
        // `state.current_resolve_additional_cost_choice_id` to
        // compute the damage amount). Today no Effect arm reads the
        // field, so Cycle 2 is parity-NEUTRAL — every existing
        // Tier-A / Phase-1 / Phase-2 / modal arm self-scans and
        // ignores the additional-cost slot.
        //
        // **Why deterministic-first and not via the agent.** Rust's
        // bench and cross-engine harness drive deterministic-first
        // agents (the `_PickFirstLegalTargetAgent` campaign-wide
        // convention); a real agent-driven choice would require a
        // Decision-style callback (parallel to
        // `submit_choose_mode_decision` for the resolve-time path).
        // The deterministic-first pick matches Python's no-agent
        // fallback byte-equal.
        let bound_additional_cost_choice_id: Option<u32> = self
            .registry
            .lookup(printing_id)
            .filter(|p| p.additional_cost_choose_your_creature)
            .and_then(|_| {
                // Battlefield creatures the caster controls first
                // (Python `:1727-1730` — `options` is initialised
                // with this filter).
                let bf_pick = crate::bf_iter::bf_iter_etb_sorted(self)
                    .into_iter()
                    .find_map(|o| {
                        if o.controller != player_id {
                            return None;
                        }
                        let p = self.registry.lookup(o.printing_id)?;
                        if p.card_types.iter().any(|t| t == "Creature") {
                            Some(o.id)
                        } else {
                            None
                        }
                    });
                if bf_pick.is_some() {
                    return bf_pick;
                }
                // Then warped creature cards in the caster's exile
                // (Python `:1731-1736` — `options.extend(o for o in
                // p.exile if o.owner_id == player_id and o.warped
                // and o.printing.is_creature)`).
                let player_idx = idx;
                self.players[player_idx]
                    .exile
                    .iter()
                    .find_map(|o| {
                        if !o.warped {
                            return None;
                        }
                        // `o.owner == player_id` is implied by the
                        // exile zone belonging to player_idx (Rust's
                        // exile is a per-player zone — every object
                        // in `players[idx].exile` is owned by player
                        // idx, mirroring Python's `p.exile` per-
                        // player list).
                        let p = self.registry.lookup(o.printing_id)?;
                        if p.card_types.iter().any(|t| t == "Creature") {
                            Some(o.id)
                        } else {
                            None
                        }
                    })
            });
        // Delta 4: push the spell StackItem (card stays in hand —
        // Rust's pre-existing spell-stack model). The existing
        // `resolve_top_of_stack` drains it: applies `printing.effects`
        // with the target(s) swapped in, then
        // `move_source_post_resolve` sends the spell hand → graveyard
        // (FREE — CR 608.2m).
        //
        // **Composite Lane KK + Lane LL routing.** Two non-overlapping
        // cast-time channels feed the StackItem push:
        //   - Lane LL (or_both=True multi-pick, `bound_mode_idxs`
        //     `Some`): route to `new_spell_with_targets_and_modes`.
        //     `or_both` modal spells are not on the
        //     additional-cost-choose-your-creature pathway, so the
        //     additional-cost slot is implicitly `None`. Mutual-
        //     exclusion contract: this constructor leaves
        //     `chosen_mode_idx` `None`.
        //   - Default (single-pick modal OR non-modal OR additional-
        //     cost): route to `new_spell_with_targets_mode_addcost_and_x`
        //     which carries the cast-time-bound target(s) PLUS the
        //     single-pick modal idx PLUS the additional-cost choice
        //     id PLUS the cast-time-bound X value (Cycle 1 substrate
        //     — passes `None` here; Cycle 2/3 will mint X-cost
        //     bindings for Zero Point Ballad / Devastating Onslaught).
        //     The multi-pick sibling is `None` in this constructor.
        //
        // **X-value subsystem Cycle 1 (2026-05-21).** The new
        // `x_value` constructor parameter is hard-coded to `None`
        // at this site — Cycle 1 is substrate-only, no live mint
        // site is wired yet (the cast-pipeline detector that reads
        // `mc.x_pips > 0` and binds the deterministic-first X via
        // pay-all-floating no-agent fallback is Cycle 2/3 work).
        // Back-compat invariant: every existing cross-engine
        // trajectory stays byte-identical because every push here
        // sets `x_value: None`, the resolver swap is a no-op for
        // `None`, and no Effect arm reads the transient yet.
        if let Some(idxs) = bound_mode_idxs {
            self.stack.push(crate::state::StackItem::new_spell_with_targets_and_modes(
                obj_id,
                player_id,
                bound_target,
                bound_target_2,
                Some(idxs),
            ));
        } else {
            self.stack.push(crate::state::StackItem::new_spell_with_targets_mode_addcost_and_x(
                obj_id,
                player_id,
                bound_target,
                bound_target_2,
                bound_mode_idx,
                bound_additional_cost_choice_id,
                // X-value subsystem Cycle 1: no live mint site.
                None,
            ));
        }
        // Delta 7: priority/passes reset, verbatim from
        // `cast_creature` / `cast_creature_warp`.
        self.priority_player = self.active_player;
        self.consecutive_passes = 0;
        Ok(true)
    }

    /// **Aura-attachment substrate — Phase 1 STEP 3 (round-78,
    /// 2026-05-20).** Cast an Aura spell from `player_id`'s hand.
    /// Mirrors `cast_spell_noncreature`'s S5a shape (mana payment +
    /// rehome to stack store + IS_CAST + StackItem push + priority
    /// reset) WITH the cast-time enchant-target decision pushed onto
    /// the pending-decisions queue (CR 303.4 / 601.2c — Auras pick
    /// their enchant target at cast time). Mirrors Python's Aura
    /// branch in `_select_spell_targets` at `actions.py:1337-1357`.
    ///
    /// Returns `true` on success (cost paid, spell on stack, decision
    /// queued or skipped per the empty-candidate branch), `false` on
    /// any precondition failure (object not in hand, no such
    /// printing, not an Aura, insufficient mana) with NO mutation
    /// (transactional shape — the same `Ok(false)` semantics
    /// `cast_creature` / `cast_spell_noncreature` use to let the
    /// caller's dispatch fall through to the next primitive).
    ///
    /// **STEP-3 scope.** Stops at "decision queued + target chosen +
    /// bound onto StackItem." The resolution arm (STEP 4 — stamping
    /// `aura.attached_to = Some(host_id)` + applying the layer-7 P/T
    /// + keyword grant) is honest-stopped per the plan; today the
    /// resolved Aura goes to bf as a host-less floating enchantment
    /// (the pre-STEP-3 Rust behavior — `Aura.attached_to = None`
    /// because no zone-change resets it to anything else). When STEP
    /// 4 lands it will read `current_resolve_payload_target_id` (the
    /// resolve-time swap of `StackItem.target_obj_id`) and do the
    /// attach. Fizzle-on-no-legal-target (Python parity at
    /// `actions.py:2740-2745` — Aura with `target_obj is None` goes
    /// to graveyard without entering bf) is also STEP 4's concern.
    ///
    /// **No-legal-target cast is ALLOWED at cast time** (Python
    /// parity: `actions.py:1354 if legal:` short-circuits the agent
    /// pick when zero candidates exist, but the cast itself still
    /// completes — fizzle happens at resolve). The empty-candidate
    /// branch skips the decision-push entirely (no choice to make);
    /// the StackItem rides with `target_obj_id = None` and STEP 4
    /// will fizzle on resolve. This matches the Python cast pump's
    /// behavior — there is no cast-time gate that says "an Aura with
    /// zero legal hosts cannot be cast at all."
    fn cast_aura(&mut self, obj_id: u32, player_id: u8) -> PyResult<bool> {
        let idx = check_player_id(player_id)?;
        // Phase 1: hand lookup. Bail before any mutation if absent.
        let printing_id = match self.players[idx]
            .hand
            .iter()
            .find(|o| o.id == obj_id)
            .map(|o| o.printing_id)
        {
            Some(pid) => pid,
            None => return Ok(false),
        };
        // Phase 2: aura-shape gate. The printing MUST be an Aura
        // (`is_aura=True` set by STEP 2's parse arm reading the
        // bridge's `is_aura` registry-dict key). Non-Aura -> bail
        // (`Ok(false)` no-mutation) so the dispatcher falls through
        // to `cast_spell_noncreature` / `cast_creature`.
        let (cost, enchant_type, yours_only) = match self.registry.lookup(printing_id) {
            Some(p) => {
                if !p.is_aura {
                    return Ok(false);
                }
                (
                    p.mana_cost.clone(),
                    p.aura_enchant_type.clone(),
                    p.aura_yours_only,
                )
            }
            None => return Ok(false),
        };
        // Phase 3: build the candidate set BEFORE paying cost so we
        // can push the decision with a stable id list. Pre-pay scan
        // is read-only — no state mutation until phase 4 commits the
        // cost. Mirrors Python `_select_spell_targets`'s aura branch
        // at `actions.py:1337-1357`:
        //   - `enchant_type == "artifact"` -> legal hosts are
        //     `o.printing.is_artifact` permanents (both bfs).
        //   - else ("creature" — the Phase-1 dominant case) -> legal
        //     hosts are `o.printing.is_creature` permanents.
        //   - `yours_only` further restricts to `o.controller_id ==
        //     player_id` (the printed "Enchant <type> you control"
        //     wording, Shardmage's Rescue).
        // Order: `bf_iter_etb_sorted` (cross-engine-proven equivalent
        // of Python's `state.battlefield` walk — same ordering every
        // other cast-time target picker uses). `candidates[0]` is
        // the deterministic-first legal host (the auto-resolve
        // default).
        let candidates: Vec<u32> = crate::bf_iter::bf_iter_etb_sorted(self)
            .into_iter()
            .filter_map(|o| {
                let p = self.registry.lookup(o.printing_id)?;
                let type_ok = if enchant_type == "artifact" {
                    p.card_types.iter().any(|t| t == "Artifact")
                } else {
                    // Default "creature" enchant-type. Mirrors
                    // Python's branch at actions.py:1347 (the `else`
                    // arm — any aura_enchant_type that isn't
                    // explicitly "artifact" reads as "creature").
                    p.card_types.iter().any(|t| t == "Creature")
                };
                if !type_ok {
                    return None;
                }
                if yours_only && o.controller != player_id {
                    return None;
                }
                Some(o.id)
            })
            .collect();
        // Phase 4: pay the mana cost. Floating pool first, then
        // auto-tap lands — byte-mirrors `cast_creature` /
        // `cast_spell_noncreature`. `false` on failure (no mutation).
        if !auto_tap_pay_for_cost(self, idx, &cost)? {
            return Ok(false);
        }
        // Phase 5: rehome the spell hand -> stack store (Option β
        // mirror of Python's `p.hand.remove(card)` + `card.zone =
        // Zone.STACK`). Same primitive `cast_spell_noncreature` uses.
        if !rehome_spell_to_stack_store(self, idx, obj_id, player_id) {
            // Defensive: hand-lookup succeeded in phase 1 but the
            // rehome failed — should not happen because phase 1
            // confirmed presence. Returning `false` preserves the
            // transactional contract.
            return Ok(false);
        }
        // Phase 6: bump spells-cast counter (MMM Pin 1 — Python parity
        // at `actions.py:1777`, BEFORE IS_CAST dispatch).
        self.players[player_id as usize].spells_cast_this_turn += 1;
        // Phase 7: dispatch Event::Cast — spell is stack-resident,
        // before the StackItem push. Same ordering as
        // `cast_creature` / `cast_spell_noncreature`.
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Cast,
            obj_id,
        )?;
        // Phase 8: push the StackItem. Target is initially `None`;
        // the controller binds it via `submit_enchant_target_decision`
        // (or the auto-resolve default picks `candidates[0]`). When
        // STEP 4 lands, the resolve arm reads the bound target via
        // `current_resolve_payload_target_id` (the standard
        // resolve-time swap of `StackItem.target_obj_id`). Use the
        // `new_spell_with_targets` constructor with both slots
        // `None` for back-compat with the existing Spell-shape
        // resolve path; `submit_enchant_target_decision` mutates the
        // StackItem's `target_obj_id` in place.
        self.stack.push(crate::state::StackItem::new_spell_with_targets(
            obj_id,
            player_id,
            None,
            None,
        ));
        // Phase 9: queue the enchant-target decision IFF there's a
        // non-empty candidate set. Python parity: `actions.py:1354
        // if legal:` short-circuits the agent pick when zero
        // candidates exist (the cast still completes, no decision is
        // ever made; the Aura fizzles on resolve at
        // `actions.py:2740-2745`). An empty candidate set MUST NOT
        // push the decision — otherwise the caster faces an
        // unanswerable choice (no legal id to submit). The empty-set
        // StackItem rides with `target_obj_id = None`; STEP 4's
        // resolve arm will fizzle it to graveyard.
        if !candidates.is_empty() {
            self.pending_decisions.push(
                crate::state::PendingDecision::EnchantTargetChoice {
                    source_obj_id: obj_id,
                    controller: player_id,
                    candidates,
                },
            );
        }
        // Phase 10: priority + passes reset (CR 117.1c — active
        // player gets priority after a spell is cast). Verbatim from
        // `cast_creature` / `cast_spell_noncreature`.
        self.priority_player = self.active_player;
        self.consecutive_passes = 0;
        Ok(true)
    }
}
