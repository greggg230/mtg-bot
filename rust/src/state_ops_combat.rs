//! Combat primitives on `GameState`.
//!
//! Ports the irreducible end-to-end "attacker hits a blocker, both
//! might die" combat scenario from `src/mtg_bot/combat.py`. The
//! Python side spreads combat across hundreds of lines (predicate-
//! style event dispatch, first-strike / double-strike timing, trample,
//! deathtouch, lifelink, infect, prevention, exalted / vigilance,
//! "whenever attacks/blocks/deals combat damage" triggers). The first
//! Rust slice carves off the four primitives the higher layers will
//! call into: `declare_attacker`, `declare_blocker`,
//! `deal_combat_damage`, `clear_combat`, plus read accessors
//! `get_attackers` / `get_blockers`.
//!
//! The state shape (`attackers: Vec<u32>`, `blockers:
//! Vec<(u32, u32)>`) was added in this round's foundation commit on
//! `state.rs`; this slice fills in the operations.
//!
//! ## Scope simplifications (mirrored on `tests_rust/test_combat.py`)
//!
//! - **No infect / protection / unblockable.** Double-strike +
//!   first-strike + trample land here; the remaining keywords are
//!   deferred. Future combat slices peel off one keyword at a time.
//! - **Vigilance (CR 702.20) IS modeled.** A creature with
//!   `"vigilance"` in `printing.keywords` skips the tap step on
//!   declare-attacker — it gets pushed onto `self.attackers` but
//!   `obj.tapped` stays False. Vigilance does NOT bypass the
//!   "must be untapped to attack" precondition (a pre-tapped
//!   vigilance creature is still rejected) and does NOT grant haste
//!   (a summoning-sick vigilance creature is still rejected).
//! - **Deathtouch (CR 702.2) IS modeled.** When a creature with
//!   `"deathtouch"` in `printing.keywords` deals NONZERO combat
//!   damage to another creature, the recipient's marked `damage` is
//!   upgraded to AT LEAST its printed `toughness` so the round-5
//!   lethal-damage SBA pass (`apply_sbas`) destroys it. The literal
//!   damage dealt is still the source's power — deathtouch only
//!   changes the lethality treatment, not the amount any other rule
//!   reads (CR 702.2c). Zero-power deathtouch creatures do NOT
//!   trigger the upgrade because deathtouch requires nonzero damage
//!   (CR 702.2b).
//! - **Lifelink (CR 702.15) IS modeled.** When a creature with
//!   `"lifelink"` in `printing.keywords` deals combat damage, its
//!   controller gains that much life — composed with the round-1
//!   life slice's mutation (`players[controller].life += amount`).
//!   Fires on damage dealt to creatures (blockers / attackers) AND
//!   on damage dealt to the defending player. This first slice only
//!   covers combat damage because `deal_combat_damage` is the only
//!   damage source today; non-combat damage sources will gain the
//!   same lifelink hook when they grow up.
//! - **Trample (CR 702.19) IS modeled — single-blocker case only.**
//!   When an attacker with `"trample"` in `printing.keywords` is
//!   blocked, any excess of `attacker.power` over the (single)
//!   blocker's printed `toughness` is dealt to the defending player.
//!   `excess = max(attacker.power - blocker.toughness, 0)`. Trample
//!   does NOT add a second helping on the unblocked path — an
//!   unblocked attacker already deals all its power to the defender
//!   through the existing path; trample only widens the blocked path.
//!   Multi-blocker damage distribution (CR 510.1c) stays deferred —
//!   the round-5 combat slice itself only applies attacker.power to
//!   the FIRST declared blocker.
//! - **Trample + deathtouch (CR 702.19c) IS modeled.** When an
//!   attacker has BOTH `"trample"` and `"deathtouch"`, only 1 damage
//!   needs to be assigned to each blocker before trampling the rest
//!   through — because deathtouch makes 1 damage count as lethal
//!   (CR 702.2b). The per-blocker trample cap becomes 1 instead of
//!   `blocker.toughness`, so `excess = max(attacker.power - 1, 0)`.
//!   Deathtouch's lethal-mark upgrade still fires on the recipient
//!   (the blocker's marked damage gets bumped to its toughness for
//!   the SBA pass). Trample-only (no deathtouch) keeps the cap at
//!   `blocker.toughness`; deathtouch-only (no trample) keeps the
//!   round-5 mutual exchange with deathtouch's lethal upgrade and no
//!   trample-through.
//! - **First-strike (CR 702.7) IS modeled.** When ANY participant in
//!   combat has `"first_strike"` OR `"double_strike"` in
//!   `printing.keywords`, `deal_combat_damage` runs in two passes: a
//!   "first-strike damage step" applying first-strike-tagged entries,
//!   then a "regular damage step" applying regular-tagged entries —
//!   only from sources still alive (i.e. their marked `damage` is <
//!   their printed `toughness` after the first step). Trample /
//!   deathtouch / lifelink fire inside whichever pass actually
//!   applies the damage. SBAs are still NOT fired between the two
//!   passes — the within-function "alive" check uses
//!   `damage < toughness` rather than running `apply_sbas`,
//!   mirroring the round-5 simplification that the SBA slice owns
//!   lethal-damage cleanup.
//! - **Double-strike (CR 702.4) IS modeled.** A creature with
//!   `"double_strike"` in its printing's keywords participates in
//!   BOTH the first-strike substep AND the regular substep — it
//!   emits a damage entry per substep at plan-construction time
//!   (`passes_for` returns `[FirstStrike, Regular]` for DS sources).
//!   The regular-substep entry is gated by the same alive-after-step-1
//!   predicate that gates non-first-strike sources, so a DS creature
//!   that took lethal damage in step 1 doesn't strike a second time.
//!   First-strike + double-strike compose to "double-strike" for
//!   damage timing (FS + DS both emit in step 1; DS additionally
//!   emits in step 2). **No DS + trample interaction** in this slice
//!   — when a single blocker dies in step 1, a DS attacker's
//!   step-2 strike has no creature target and (since trample on the
//!   blocked-side-of-step-2 isn't wired) fizzles. Real MTG would
//!   trample-spill the second strike to the defender; deferred.
//! - **Haste (CR 702.10) IS modeled.** A creature with `"haste"` in
//!   `printing.keywords` bypasses the summoning-sickness rejection
//!   in `declare_attacker` — a freshly-cast haste creature can
//!   attack the turn it enters. Haste does NOT bypass the "must be
//!   untapped to attack" precondition (a pre-tapped haste creature
//!   is still rejected) and does NOT skip the tap-on-attack step
//!   (haste != vigilance — a haste creature without vigilance still
//!   taps when declared as attacker). Haste + vigilance composes:
//!   the haste branch handles the summoning-sick gate, the vigilance
//!   branch handles the tap step. Haste also affects activated
//!   tap-abilities in real MTG (CR 302.1 covers both attacking and
//!   activating tap-cost abilities); this slice only models the
//!   attacking half because `declare_attacker` is the only summoning-
//!   sick gate today.
//! - **Defender (CR 702.3) IS modeled.** A creature whose printing
//!   carries the `"defender"` keyword is rejected at `declare_attacker`.
//!   Defender wins over haste — a creature with both keywords still
//!   cannot attack (haste only bypasses the summoning-sickness gate;
//!   it does not grant the ability to attack a creature otherwise
//!   prohibited from attacking). Defender does NOT affect blocking,
//!   damage, or any other phase. Other "can't attack" effects (Propaganda,
//!   summoning-sick-without-haste was already covered) and attacked-alone
//!   restrictions are still deferred.
//! - **Menace (CR 702.110) IS modeled — predicate, not declare-time gate.**
//!   A creature with `"menace"` can't be blocked except by two or more
//!   creatures. Enforcement uses an `is_legal_block_assignment(attacker_id)`
//!   predicate rather than rejecting at `declare_blocker`: the first
//!   blocker assigned to a menace attacker is illegal in isolation but
//!   legal once a second is added — declare-time rejection would force
//!   callers into an artificial "all-at-once" submission API. The
//!   predicate counts entries in `self.blockers` matching `attacker_id`
//!   and returns false iff the attacker has menace AND that count is
//!   exactly 1. Zero blockers (unblocked) and 2+ blockers are legal;
//!   non-menace attackers always pass. Other multi-blocker constraints
//!   (e.g. "can't be blocked by more than two") stay deferred.
//! - **No other attack-eligibility legality checks** beyond
//!   "creature on attacker's battlefield, untapped, not summoning-sick
//!   (modulo haste), not defender". "Can't attack" continuous effects
//!   beyond defender, attacked-alone restrictions, etc. remain deferred.
//! - **Flying (CR 702.9) + reach (CR 702.17) IS modeled.** A flying
//!   attacker (printing carries `"flying"`) can only be blocked by
//!   creatures whose printing carries `"flying"` or `"reach"`; other
//!   blockers are refused at `declare_blocker`. Reach on the BLOCKER
//!   is the gate-opener; reach on the ATTACKER is a no-op for blocking
//!   purposes (a reach attacker is treated as a ground creature —
//!   blockable by anything). Defender-only, color protection,
//!   "can't be blocked by ..." gates are still deferred.
//! - **Skulk (CR 702.118) IS modeled.** A creature with `"skulk"` in
//!   its printing keywords can't be blocked by creatures with greater
//!   power — STRICTLY greater (equal-power blockers are still legal).
//!   Enforced at `declare_blocker`, after the flying gate; composes
//!   with flying so a skulk+flying attacker requires the blocker to
//!   pass BOTH gates (flying-or-reach AND `blocker.power <=
//!   attacker.power`).
//! - **Single-blocker damage assignment.** When multiple blockers are
//!   declared on the same attacker, the attacker assigns its full
//!   power to the FIRST declared blocker. All declared blockers
//!   still deal their power back to the attacker. Trample / multi-
//!   blocker damage ordering (CR 510.1c) is the SBA-and-trample
//!   slice's job.
//! - **No "whenever attacks/blocks/deals combat damage" triggers.**
//!   No event dispatcher in Rust yet.
//! - **No SBAs after damage.** Lethal damage staying past
//!   `deal_combat_damage` is intentional — the SBA sibling slice
//!   owns "damage >= toughness => destroy" (CR 704.5h).
//! - **`clear_combat` does not untap attackers.** Untapping is the
//!   untap-step's job (`state_ops_steps.rs:advance_phase` /
//!   `state_ops_tap.rs:untap_all_for_player`). `clear_combat` only
//!   zeroes the `attackers` / `blockers` Vecs.
//!
//! ## Cross-slice composition
//!
//! - **Tap on declare-attacker:** `declare_attacker` flips
//!   `tapped = true` on the declared creature in-place via
//!   `Arc::make_mut`. Mirrors the tap-slice's `tap_object` shape but
//!   inlined here so the parallel-TDD constraint (sibling slices own
//!   their own files) doesn't force a `pub(crate)` promotion on
//!   `state_ops_tap.rs`.
//! - **Damage on combat damage:** `deal_combat_damage` mutates each
//!   damaged creature's `damage` field in-place via `Arc::make_mut` on
//!   the containing battlefield Vec. Mirrors the damage-slice's
//!   `deal_damage` shape but inlined here for the same parallel-TDD
//!   reason.
//! - **Life on unblocked:** unblocked attackers subtract their power
//!   from the defending player's `life` field directly — same path
//!   `lose_life` uses. The two-player model picks the defender as
//!   `1 - active_player`.
//!
//! ## CoW invariant
//!
//! Both `declare_attacker` (taps) and `deal_combat_damage` (marks
//! damage) mutate per-zone Vecs through `Arc::make_mut`, so cloning
//! a `GameState` and mutating the clone leaves the original
//! untouched. The two `test_clone_then_..._does_not_affect_original`
//! canaries pin this — same load-bearing shape as the per-slice CoW
//! canaries in `test_tap.py` and `test_damage.py`.

use crate::state::{GameObject, GameState};
use pyo3::prelude::*;
use std::sync::Arc;

/// Locate a battlefield permanent by id and return `(player_idx,
/// printing_id)` so callers can read printing data (power, etc.) and
/// know which player's battlefield to mutate. Returns `None` if no
/// battlefield permanent has that id.
fn locate_battlefield_obj(state: &GameState, obj_id: u32) -> Option<(usize, u32)> {
    for (idx, player) in state.players.iter().enumerate() {
        for obj in player.battlefield.iter() {
            if obj.id == obj_id {
                return Some((idx, obj.printing_id));
            }
        }
    }
    None
}

/// Apply `f` to the battlefield permanent with `obj_id` via
/// `Arc::make_mut` on the containing battlefield Vec. Returns whether
/// the object was found. Inlined per-slice because sibling tap/damage
/// slices own their own copies of this helper (parallel-TDD
/// constraint — see module docstring).
fn mutate_battlefield_obj<F>(state: &mut GameState, obj_id: u32, mut f: F) -> bool
where
    F: FnMut(&mut GameObject),
{
    for player in state.players.iter_mut() {
        if player.battlefield.iter().any(|o| o.id == obj_id) {
            let zone: &mut Vec<GameObject> = Arc::make_mut(&mut player.battlefield);
            for obj in zone.iter_mut() {
                if obj.id == obj_id {
                    f(obj);
                    return true;
                }
            }
        }
    }
    false
}

/// Read-only check: is this object on a battlefield AND a creature
/// (printing.power.is_some())? Returns `Some((tapped, summoning_sick))`
/// when the object exists on a battlefield AND is a creature; `None`
/// otherwise (object missing OR non-creature).
fn read_creature_status(state: &GameState, obj_id: u32) -> Option<(bool, bool)> {
    let (idx, printing_id) = locate_battlefield_obj(state, obj_id)?;
    let printing = state.registry.lookup(printing_id)?;
    if printing.power.is_none() {
        return None;
    }
    // Re-find the object to read its tapped / summoning_sick flags.
    for obj in state.players[idx].battlefield.iter() {
        if obj.id == obj_id {
            return Some((obj.tapped, obj.summoning_sick));
        }
    }
    None
}

/// Read effective power for a battlefield creature, returning 0 for
/// non-creatures or unknown ids. "Effective" =
/// `printed_power + ueot_power_bonus` per CR 613 layer 7c additive
/// stacking (round-27 UEOT framework). Counters do not yet feed into
/// the power side (round-2's net-counter model is toughness-only —
/// `state_ops_sba.rs` adds `obj.counters` to effective toughness,
/// not to effective power; that asymmetry is intentional, mirroring
/// Python's `effective_power` shape). Used by `deal_combat_damage`
/// to size each side's contribution to the symmetric exchange.
fn battlefield_creature_power(state: &GameState, obj_id: u32) -> i16 {
    let Some((player_idx, printing_id)) = locate_battlefield_obj(state, obj_id) else {
        return 0;
    };
    let Some(printing) = state.registry.lookup(printing_id) else {
        return 0;
    };
    let printed_base = printing.power.unwrap_or(0) as i32;
    let obj = state.players[player_idx]
        .battlefield
        .iter()
        .find(|o| o.id == obj_id);
    // Round-28 CR 613.4 layer 7b: a UEOT SetBasePT override replaces
    // the printed base. Last-applied wins. Layer 7c additive bonuses
    // (PowerToughnessBonus) still stack on top.
    //
    // Round-28 manland animation: a UEOT BecomesLandCreature ALSO
    // overrides the base P/T (the land's "creature half" has base
    // power and toughness as specified by the activated ability —
    // e.g. Blinkmoth Nexus's 1/1, Lavaclaw Reaches's 2/2). Same
    // last-applied wins shape as SetBasePT. SetBasePT takes
    // precedence when both are present (e.g. a Genemorph-Imago-style
    // SetBasePT cast on an already-animated land); this mirrors the
    // intuition that an explicit "has base P/T N/N" override is more
    // specific than the manland's intrinsic creature-half P/T.
    let base = obj
        .and_then(|o| o.ueot_base_pt_override())
        .map(|(p, _)| p)
        .or_else(|| obj.and_then(|o| o.ueot_becomes_land_creature_pt()).map(|(p, _)| p))
        .unwrap_or(printed_base);
    let bonus = obj.map(|o| o.ueot_power_bonus()).unwrap_or(0);
    let effective = base + bonus;
    // Saturate into i16 to match the existing return-shape contract;
    // a 32k+ effective power is unreachable on any real card.
    effective.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Does this battlefield permanent's printing carry the
/// `"vigilance"` keyword? Returns false for unknown ids and for
/// printings whose `keywords` Vec doesn't contain the lowercase
/// `"vigilance"` string. Used by `declare_attacker` to skip the
/// tap-on-attack step (CR 702.20).
fn battlefield_has_vigilance(state: &GameState, obj_id: u32) -> bool {
    battlefield_has_keyword(state, obj_id, "vigilance")
}

/// Generic "does this battlefield permanent carry `keyword`?" check.
/// Returns false for unknown ids, non-battlefield objects, and
/// printings whose `keywords` Vec doesn't contain the lowercase
/// keyword string. Shared by the vigilance / deathtouch / lifelink
/// branches.
///
/// **Round-27 UEOT framework.** The keyword set this reader walks
/// is the UNION of the printing's printed keywords AND any
/// `UeotModifier::GrantKeyword` currently on the object's
/// `ueot_modifiers` stack (CR 613 layer 6 keyword grants). The
/// printed-side stays the strict baseline; UEOT grants stack
/// additively on top and clear on entry to PHASE_CLEANUP (see
/// `state_ops_cleanup.rs::clear_ueot_modifiers_all`). This is what
/// gates the "and gains haste/flying/lifelink UEOT" pump-rider
/// branches at combat time.
pub(crate) fn battlefield_has_keyword(state: &GameState, obj_id: u32, keyword: &str) -> bool {
    let Some((player_idx, printing_id)) = locate_battlefield_obj(state, obj_id) else {
        return false;
    };
    let Some(printing) = state.registry.lookup(printing_id) else {
        return false;
    };
    if printing.keywords.iter().any(|k| k == keyword) {
        return true;
    }
    // UEOT-granted keywords (CR 613 layer 6).
    state.players[player_idx]
        .battlefield
        .iter()
        .find(|o| o.id == obj_id)
        .map(|o| o.ueot_has_keyword(keyword))
        .unwrap_or(false)
}

/// Read effective toughness for a battlefield creature, returning 0
/// for non-creatures or unknown ids. "Effective" =
/// `printed_toughness + ueot_toughness_bonus` per CR 613 layer 7c
/// additive stacking (round-27 UEOT framework). Used by
/// `deal_combat_damage` to size the deathtouch upgrade (the
/// recipient's damage is bumped to at least this value).
///
/// **Asymmetry note (intentional, pinned by test_combat.py).** This
/// reader does NOT fold `obj.counters` into effective toughness, even
/// though `state_ops_sba.rs::collect_sba_victims` DOES. Combat math
/// in the round-5 + round-13 slices was sized off printed toughness;
/// the counters-affect-combat path is deferred until a parity test
/// explicitly demands it. UEOT bonuses fold in here because they
/// otherwise have no consumer at combat time, defeating the
/// PumpTargetCreatureUntilEOT proof (a +2/+2 attacker NEEDS the
/// toughness math to size the deathtouch upgrade off the bumped
/// value).
fn battlefield_creature_toughness(state: &GameState, obj_id: u32) -> i16 {
    let Some((player_idx, printing_id)) = locate_battlefield_obj(state, obj_id) else {
        return 0;
    };
    let Some(printing) = state.registry.lookup(printing_id) else {
        return 0;
    };
    let printed_base = printing.toughness.unwrap_or(0) as i32;
    let obj = state.players[player_idx]
        .battlefield
        .iter()
        .find(|o| o.id == obj_id);
    // Round-28 CR 613.4 layer 7b: SetBasePT overrides the printed
    // base. Last-applied wins. Layer 7c additive bonuses stack on
    // top. Round-28 manland animation also overrides the printed
    // base; SetBasePT takes precedence when both are present (same
    // semantics as battlefield_creature_power above).
    let base = obj
        .and_then(|o| o.ueot_base_pt_override())
        .map(|(_, t)| t)
        .or_else(|| obj.and_then(|o| o.ueot_becomes_land_creature_pt()).map(|(_, t)| t))
        .unwrap_or(printed_base);
    let ueot_bonus = obj.map(|o| o.ueot_toughness_bonus()).unwrap_or(0);
    let effective = base + ueot_bonus;
    effective.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Read controller (player index) of a battlefield creature, or
/// `None` if not present. Used by `deal_combat_damage` to credit
/// lifelink life-gain to the right player.
fn battlefield_creature_controller(state: &GameState, obj_id: u32) -> Option<u8> {
    for player in state.players.iter() {
        for obj in player.battlefield.iter() {
            if obj.id == obj_id {
                return Some(obj.controller);
            }
        }
    }
    None
}

/// Read currently-marked `damage` on a battlefield creature, or 0 if
/// not present. Used by the first-strike "is the source still alive
/// for the regular damage step?" check in `deal_combat_damage`.
fn battlefield_creature_damage(state: &GameState, obj_id: u32) -> i16 {
    for player in state.players.iter() {
        for obj in player.battlefield.iter() {
            if obj.id == obj_id {
                return obj.damage;
            }
        }
    }
    0
}

/// Plan target for one entry in `deal_combat_damage`'s damage plan.
/// Hoisted out of the function body so the per-pass apply helper
/// (`apply_damage_pass`) can take the entry list as a parameter.
enum DamageTarget {
    Creature(u32),
    Player(u8),
}

/// Which substep of `deal_combat_damage`'s two-substep timing this
/// entry fires in. CR 702.7 (first-strike) introduces two damage
/// substeps when ANY participant has first-strike; CR 702.4
/// (double-strike) lets a creature deal damage in BOTH substeps.
///
/// Per-source rules:
/// - first-strike only → emits entries with `Pass::FirstStrike`
/// - double-strike only → emits TWO copies of each entry, one per pass
/// - first-strike + double-strike → same as double-strike (both passes;
///   real MTG treats these as identical for damage timing)
/// - neither (vanilla) → emits entries with `Pass::Regular`
///
/// `Pass` replaces the round-9 `source_has_first_strike: bool` flag —
/// the partition into pass-1 vs pass-2 reads `Pass::FirstStrike` vs
/// `Pass::Regular` and the keyword check is folded into plan
/// construction (which now also handles entry duplication for
/// double-strike).
#[derive(Copy, Clone, PartialEq, Eq)]
enum Pass {
    FirstStrike,
    Regular,
}

/// One entry in `deal_combat_damage`'s damage plan: a source dealing
/// `amount` damage to `target` in a specific substep `pass` (CR 702.7
/// first-strike timing). A double-strike source contributes TWO
/// entries per damage assignment (one per pass) — see `Pass`.
struct PlanEntry {
    source_id: u32,
    target: DamageTarget,
    amount: i16,
    pass: Pass,
}

/// Which substeps does a source with these keywords fire in?
/// - Vanilla (neither FS nor DS): `[Regular]` — strikes once in step 2.
/// - First-strike only: `[FirstStrike]` — strikes once in step 1.
/// - Double-strike (with or without first-strike — they coincide for
///   damage timing): `[FirstStrike, Regular]` — strikes once per step.
///
/// Returns a fixed-size array slice via `&[Pass]` so the caller can
/// iterate without allocating. We return a small `Vec<Pass>` instead
/// because the two-element variant has size 2 and the one-element
/// variants have size 1 — a stack-allocated `Vec` of capacity 2 is
/// cheap and avoids the lifetime gymnastics of returning a slice into
/// a `static` array.
fn passes_for(has_first_strike: bool, has_double_strike: bool) -> Vec<Pass> {
    if has_double_strike {
        vec![Pass::FirstStrike, Pass::Regular]
    } else if has_first_strike {
        vec![Pass::FirstStrike]
    } else {
        vec![Pass::Regular]
    }
}

/// "Is this creature still alive after the first-strike damage step?"
/// Returns true unless the creature's current marked `damage` is
/// >= its printed `toughness` (CR 704.5g lethal-damage condition,
/// inlined here without firing SBAs — the SBA slice owns the
/// destroy-creature mutation; we only need the predicate to gate
/// pass-2 damage from a creature that took lethal in pass 1).
/// Returns true for creatures with toughness <= 0 (treated as
/// already-dead via the > check) — but that's a degenerate seed and
/// the test surface doesn't exercise it.
fn battlefield_creature_alive(state: &GameState, obj_id: u32) -> bool {
    let toughness = battlefield_creature_toughness(state, obj_id);
    if toughness <= 0 {
        return false;
    }
    let damage = battlefield_creature_damage(state, obj_id);
    (damage as i32) < (toughness as i32)
}

#[pymethods]
impl GameState {
    /// **Round-27 UEOT framework: effective-power getter.** Returns
    /// the current effective power of a battlefield creature
    /// (printing.power.unwrap_or(0) + sum of
    /// `UeotModifier::PowerToughnessBonus.power` on the object's
    /// `ueot_modifiers` Vec). Saturates into i16 to match the
    /// combat-side return shape; `None` if the object isn't on a
    /// battlefield or its printing has no `power` (non-creature).
    ///
    /// Mirrors Python's `mtg_bot.game.effective_power(state, obj)`
    /// for tests + the parity-smoke harness. Combat readers
    /// (`declare_attacker`, `deal_combat_damage`) use the
    /// module-private `battlefield_creature_power` of the same
    /// shape; this public getter wraps it.
    fn effective_power(&self, obj_id: u32) -> Option<i16> {
        if locate_battlefield_obj(self, obj_id).is_none() {
            return None;
        }
        let (player_idx, printing_id) = locate_battlefield_obj(self, obj_id)?;
        let printing = self.registry.lookup(printing_id)?;
        // Creature predicate is satisfied either by a printed-power
        // value OR by a UEOT BecomesLandCreature modifier on the
        // object (round-28 manland animation — the land has no
        // printed power, but its activated ability has flipped it to
        // a creature with base P/T from the modifier).
        if printing.power.is_none() {
            let is_animated = self.players[player_idx]
                .battlefield
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.ueot_is_creature_via_become())
                .unwrap_or(false);
            if !is_animated {
                return None;
            }
        }
        Some(battlefield_creature_power(self, obj_id))
    }

    /// **Round-27 UEOT framework: effective-toughness getter.**
    /// Returns the current effective toughness of a battlefield
    /// creature (printing.toughness + sum of
    /// `UeotModifier::PowerToughnessBonus.toughness`). Saturates
    /// into i16; `None` if the object isn't on a battlefield or
    /// its printing has no `toughness` (non-creature).
    ///
    /// Mirrors Python's `mtg_bot.game.effective_toughness(state,
    /// obj)`. The combat-side `battlefield_creature_toughness`
    /// returns the same number; this public getter wraps it.
    /// NOTE: this getter does NOT fold in `obj.counters` (the
    /// signed +1/+1 / -1/-1 net-counter contribution) — the
    /// asymmetry matches `battlefield_creature_toughness`'s
    /// combat-side shape and is pinned by `test_combat.py`. The
    /// SBA-side `is_lethal` predicate DOES fold counters in; use
    /// that for the lethal-damage view.
    fn effective_toughness(&self, obj_id: u32) -> Option<i16> {
        let (player_idx, printing_id) = locate_battlefield_obj(self, obj_id)?;
        let printing = self.registry.lookup(printing_id)?;
        // Same creature-predicate widening as `effective_power`:
        // a UEOT BecomesLandCreature modifier flips a printing-side
        // non-creature into an effective creature for the duration.
        if printing.toughness.is_none() {
            let is_animated = self.players[player_idx]
                .battlefield
                .iter()
                .find(|o| o.id == obj_id)
                .map(|o| o.ueot_is_creature_via_become())
                .unwrap_or(false);
            if !is_animated {
                return None;
            }
        }
        Some(battlefield_creature_toughness(self, obj_id))
    }

    /// **Round-27 UEOT framework: effective-keyword getter.**
    /// Returns `true` iff the battlefield permanent's effective
    /// keyword set contains `keyword` — the UNION of the printing's
    /// printed keywords AND any `UeotModifier::GrantKeyword` on the
    /// object's `ueot_modifiers` Vec. Lowercase match (the bridge
    /// already lowercases printed keywords at register time;
    /// callers pass the keyword in lowercase). Returns `false` for
    /// unknown obj_ids or non-battlefield objects, mirroring the
    /// existing `battlefield_has_keyword` private helper's shape.
    fn has_keyword(&self, obj_id: u32, keyword: &str) -> bool {
        battlefield_has_keyword(self, obj_id, keyword)
    }

    /// Declare the battlefield creature with `obj_id` as an attacker.
    ///
    /// On success: pushes `obj_id` onto `self.attackers`, taps the
    /// creature (UNLESS the printing carries the `"vigilance"`
    /// keyword — CR 702.20), returns `true`. On any precondition
    /// failure: returns `false` with no mutation.
    ///
    /// Preconditions:
    /// - object exists on some battlefield
    /// - object's printing has `power.is_some()` (creature predicate)
    /// - object is NOT tapped
    /// - object is NOT summoning-sick — UNLESS the printing carries
    ///   the `"haste"` keyword (CR 702.10), in which case the
    ///   summoning-sickness restriction is bypassed.
    ///
    /// Vigilance only suppresses the tap-on-attack step; it does NOT
    /// bypass the "must be untapped to attack" precondition (a
    /// pre-tapped vigilance creature still fails) and does NOT grant
    /// haste.
    ///
    /// Haste only suppresses the summoning-sickness rejection; it
    /// does NOT bypass the "must be untapped to attack" precondition
    /// (a pre-tapped haste creature still fails) and does NOT grant
    /// vigilance — a haste creature without vigilance still taps when
    /// it attacks. Haste + vigilance composes (attack on the same turn
    /// it entered AND skip the tap).
    ///
    /// Defender (CR 702.3) is the fifth gate, added in this slice: a
    /// creature with `"defender"` in its printing keywords is rejected
    /// regardless of taps / summoning sickness / haste. Defender wins
    /// over haste — a creature with both keywords still cannot attack.
    ///
    /// **No legality checks beyond the five above.** "Can't attack"
    /// continuous effects beyond defender, attacked-alone restrictions,
    /// and "doesn't untap during untap step" interactions are deferred —
    /// see module docstring.
    fn declare_attacker(&mut self, obj_id: u32) -> PyResult<bool> {
        // Read-only validation pass — bail before any mutation if the
        // creature isn't eligible.
        let Some((tapped, summoning_sick)) = read_creature_status(self, obj_id) else {
            return Ok(false);
        };
        if tapped {
            return Ok(false);
        }
        // Defender (CR 702.3): a creature with defender can't attack.
        // This is independent of haste / summoning sickness — defender
        // wins over haste even when haste would otherwise bypass the
        // summoning-sickness gate.
        if battlefield_has_keyword(self, obj_id, "defender") {
            return Ok(false);
        }
        // Haste (CR 702.10) bypasses summoning sickness ONLY. Read the
        // keyword once here so the eligibility gate and the (later)
        // tap step can both see it — though only the eligibility gate
        // currently consults it; haste does NOT skip the tap step.
        // Pre-read so we don't pay the Arc::make_mut clone cost when
        // the creature is going to be rejected for some other reason.
        let has_haste = battlefield_has_keyword(self, obj_id, "haste");
        if summoning_sick && !has_haste {
            return Ok(false);
        }
        // Vigilance check: read before mutating so we don't pay the
        // Arc::make_mut clone cost when we're going to skip the tap.
        let has_vigilance = battlefield_has_vigilance(self, obj_id);
        // Eligible. Tap the creature (CR 508.1c — attacking taps the
        // attacker, modulo vigilance) and record it on the attackers
        // Vec. The tap goes through Arc::make_mut so the CoW invariant
        // survives. Vigilance creatures skip the tap step entirely.
        if !has_vigilance {
            let tapped_ok = mutate_battlefield_obj(self, obj_id, |obj| {
                obj.tapped = true;
            });
            if !tapped_ok {
                // Should be unreachable — read_creature_status already
                // located it. Defensive: if a clone race or other
                // instability removed it, undo nothing.
                return Ok(false);
            }
        }
        self.attackers.push(obj_id);
        // Slice 7: dispatch Event::Attacks for the just-declared
        // attacker. The attacker is on the battlefield (it has to be,
        // per the read_creature_status gate above), so the bf
        // observer set already contains it — `Predicate::SourceIsSelf`
        // ("whenever this creature attacks") matches naturally
        // without needing the DIES/CAST extra-observer pattern.
        //
        // Mirrors Python's `combat._fire_attacks_triggers(state, obj)`
        // call at `src/mtg_bot/combat.py:115` — fired per-attacker
        // immediately after the attacking flag is set, so triggers
        // observe each attacker in declaration order.
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Attacks,
            obj_id,
        )?;
        Ok(true)
    }

    /// Declare the battlefield creature with `blocker_id` as a
    /// blocker against the previously-declared attacker `attacker_id`.
    ///
    /// On success: pushes `(attacker_id, blocker_id)` onto
    /// `self.blockers`, returns `true`. On any precondition failure:
    /// returns `false` with no mutation.
    ///
    /// Preconditions:
    /// - `attacker_id` is in `self.attackers`
    /// - `blocker_id` resolves to a battlefield creature
    /// - blocker is NOT tapped
    /// - if the attacker has the `"flying"` keyword (CR 702.9), the
    ///   blocker must have either `"flying"` or `"reach"` (CR 702.17)
    ///   — otherwise the block is refused.
    /// - if the attacker has the `"skulk"` keyword (CR 702.118), the
    ///   blocker's power must NOT be greater than the attacker's
    ///   power — equal-power blockers pass; strictly-greater-power
    ///   blockers are refused.
    ///
    /// **No blocking restrictions beyond the five above.** Defender-only,
    /// color protection, "can't be blocked by ..." gates are deferred —
    /// see module docstring.
    ///
    /// Blocking does NOT tap the blocker (CR 509.1).
    ///
    /// **Slice 9 signature change**: returns `PyResult<bool>` (was
    /// `bool`). The `Ok(bool)` payload preserves the prior "did the
    /// block stick" contract; the `Err` arm only fires when the
    /// post-push `Event::Blocks` dispatch raises (a Python-side
    /// error during effect resolution). The only caller — the
    /// `Action::DeclareBlockers` arm in `state_ops_actions.rs` —
    /// already invokes via `call_method1(...)?` so the Python-facing
    /// behavior is unchanged.
    fn declare_blocker(&mut self, blocker_id: u32, attacker_id: u32) -> PyResult<bool> {
        if !self.attackers.contains(&attacker_id) {
            return Ok(false);
        }
        let Some((tapped, _summoning_sick)) = read_creature_status(self, blocker_id) else {
            // Object missing OR non-creature.
            return Ok(false);
        };
        if tapped {
            return Ok(false);
        }
        // Flying gate (CR 702.9): a flying attacker can only be blocked
        // by creatures with flying or reach (CR 702.17). Reach on the
        // attacker is irrelevant for the blocking direction — only
        // flying on the attacker triggers this restriction.
        if battlefield_has_keyword(self, attacker_id, "flying") {
            let blocker_has_flying = battlefield_has_keyword(self, blocker_id, "flying");
            let blocker_has_reach = battlefield_has_keyword(self, blocker_id, "reach");
            if !blocker_has_flying && !blocker_has_reach {
                return Ok(false);
            }
        }
        // Skulk gate (CR 702.118): a creature with skulk can't be
        // blocked by creatures with greater power. Strictly greater —
        // equal-power blockers are still legal. Composes with flying:
        // a skulk+flying attacker requires the blocker to pass BOTH
        // gates (flying-or-reach AND power <= attacker.power).
        if battlefield_has_keyword(self, attacker_id, "skulk") {
            let attacker_power = battlefield_creature_power(self, attacker_id);
            let blocker_power = battlefield_creature_power(self, blocker_id);
            if blocker_power > attacker_power {
                return Ok(false);
            }
        }
        // Note: we do NOT check the blocker's summoning_sick flag.
        // Summoning sickness restricts attacking and tap-abilities,
        // not blocking (CR 302.1) — a freshly cast creature can block
        // the turn it comes down.
        self.blockers.push((attacker_id, blocker_id));
        // Slice 9: dispatch Event::Blocks for the just-declared
        // blocker. The blocker is on the battlefield (it has to be,
        // per the read_creature_status gate above), so the bf
        // observer set already contains it — `Predicate::SourceIsSelf`
        // ("whenever this creature blocks") matches naturally
        // without needing the DIES/CAST/LEAVES extra-observer
        // pattern.
        //
        // Mirrors Python's per-pair `dispatch_event(state,
        // Event.BLOCKS, ...)` call in `src/mtg_bot/combat.py:274` —
        // fired per blocker→attacker pair immediately after the
        // blocker.blocking / attacker.blocked_by mutations, so
        // triggers observe each blocker in declaration order.
        //
        // **Subject mapping divergence from Python**: Python's
        // BLOCKS event carries two subjects (blocker + attacker)
        // and exposes both to predicates via the multi-subject
        // schema. Rust's single-subject `dispatch_event` API only
        // threads the blocker today; multi-subject predicates
        // against the attacked creature (e.g. Skystinger's
        // "whenever ~ blocks a creature with flying") need the
        // `Predicate::Subject(slot, inner)` substrate that a
        // future slice will introduce alongside
        // `Event::DealsCombatDamage`. Until then, BLOCKS triggers
        // whose predicates only key on the blocker (SourceIsSelf,
        // SubjectIsCreature, ControlledByYou, etc.) work as
        // expected.
        crate::state_ops_resolve::dispatch_event(
            self,
            crate::cards::Event::Blocks,
            blocker_id,
        )?;
        Ok(true)
    }

    /// Read the declared attackers as a list of object ids in
    /// declaration order. Returns a fresh `Vec<u32>` (a Python list
    /// after the PyO3 conversion).
    fn get_attackers(&self) -> Vec<u32> {
        self.attackers.clone()
    }

    /// Read the declared blocker pairs as a list of `(attacker_id,
    /// blocker_id)` tuples in declaration order.
    fn get_blockers(&self) -> Vec<(u32, u32)> {
        self.blockers.clone()
    }

    /// Validation predicate: is the current set of blockers against
    /// `attacker_id` a legal assignment under the keyword constraints
    /// this slice models?
    ///
    /// Today's only check: **menace (CR 702.110)** — a creature with
    /// `"menace"` can't be blocked except by two or more creatures.
    /// The predicate returns `false` iff the attacker has menace AND
    /// has exactly one blocker assigned in `self.blockers`. Returns
    /// `true` for:
    /// - non-menace attackers (any block count is legal)
    /// - menace attackers with zero blockers (unblocked is legal —
    ///   menace constrains HOW the attacker is blocked, not whether
    ///   it must be)
    /// - menace attackers with two or more blockers (multi-block
    ///   requirement satisfied)
    ///
    /// Used as a post-declare validator: callers run this after all
    /// blocks have been declared and unwind / reject the assignment
    /// when it returns false. We don't enforce at `declare_blocker`
    /// because the first blocker assigned to a menace attacker is
    /// illegal in isolation but legal once a second is added —
    /// declare-time rejection would force an "all-at-once"
    /// submission API.
    ///
    /// Read-only: never mutates state. Returns `true` for unknown
    /// `attacker_id` (no blockers can match → 0 blockers → legal).
    pub fn is_legal_block_assignment(&self, attacker_id: u32) -> bool {
        let blocker_count = self
            .blockers
            .iter()
            .filter(|(a, _)| *a == attacker_id)
            .count();
        if battlefield_has_keyword(self, attacker_id, "menace") && blocker_count == 1 {
            return false;
        }
        true
    }

    /// Resolve one round of combat damage.
    ///
    /// For each declared attacker:
    /// - If the attacker has at least one blocker (the FIRST entry
    ///   in `self.blockers` matching `attacker_id`): assign attacker
    ///   power as damage to that blocker. If the attacker has
    ///   `"trample"`, any excess of `attacker.power` over the
    ///   blocker's printed toughness is also dealt to the defending
    ///   player (CR 702.19).
    /// - If the attacker has no blocker: subtract attacker power from
    ///   the defending player's life. Defender = `1 - active_player`
    ///   in the two-player model. Trample does NOT add a second
    ///   helping on the unblocked path.
    ///
    /// For each declared blocker: assign blocker power as damage to
    /// the attacker. This is the symmetric back-damage half of the
    /// exchange.
    ///
    /// **First-strike (CR 702.7) and double-strike (CR 702.4) ARE
    /// modeled.** When ANY plan entry fires in the first-strike
    /// substep (i.e. any participant has first-strike or
    /// double-strike), damage runs in two passes: pass 1 applies
    /// the first-strike-tagged entries; pass 2 applies the
    /// regular-tagged entries, excluding sources whose marked
    /// `damage` >= printed `toughness` after pass 1 (i.e. sources
    /// that took lethal in pass 1 don't strike back, and DS sources
    /// that took lethal in pass 1 don't strike a second time). A DS
    /// source contributes one entry to each pass per damage
    /// assignment; vanilla sources contribute one entry to pass 2;
    /// FS-only sources contribute one entry to pass 1. When NO
    /// participant has first-strike or double-strike, every entry is
    /// regular-tagged and the path degenerates to one pass —
    /// bit-identical to the pre-first-strike behavior.
    ///
    /// **No SBA enforcement between passes.** The "alive" check in
    /// pass 2 is a within-function `damage < toughness` predicate;
    /// dead creatures are NOT moved to the graveyard. Damage marks
    /// stay; lethal-damage-destroys-creature is the SBA slice's job.
    fn deal_combat_damage(&mut self) -> PyResult<()> {
        // Snapshot attacker/blocker plans first. Mutating the
        // battlefield (damage marks, life) is held off until after the
        // plan is built so simultaneous-damage semantics aren't
        // sequence-corrupted (e.g. attacker dies mid-loop and stops
        // dealing). Each plan entry carries:
        // - the source's id (so we can re-read deathtouch / lifelink
        //   keywords without re-walking the per-attacker logic)
        // - the target (creature / player)
        // - the literal `amount` of damage
        // - which substep (`Pass::FirstStrike` or `Pass::Regular`) the
        //   entry fires in. First-strike (CR 702.7) sources emit
        //   `FirstStrike`-tagged entries; double-strike (CR 702.4)
        //   sources emit BOTH a `FirstStrike`-tagged and a
        //   `Regular`-tagged entry per damage assignment; vanilla
        //   sources emit `Regular`-tagged entries.
        let mut plan: Vec<PlanEntry> = Vec::new();

        let defending_player: u8 = 1 - self.active_player;

        for &attacker_id in &self.attackers {
            let attacker_power = battlefield_creature_power(self, attacker_id);
            let attacker_has_first_strike =
                battlefield_has_keyword(self, attacker_id, "first_strike");
            let attacker_has_double_strike =
                battlefield_has_keyword(self, attacker_id, "double_strike");
            let attacker_has_trample = battlefield_has_keyword(self, attacker_id, "trample");
            // Per-pass emission for this source. DS sources emit in
            // both substeps; FS-only sources emit only in step 1;
            // vanilla sources emit only in step 2.
            let passes_for_source = passes_for(
                attacker_has_first_strike,
                attacker_has_double_strike,
            );
            // Find ALL blockers matching this attacker, ordered by
            // bf-ETB timestamp via the shared `bf_iter_etb_sorted`
            // helper. Mirrors Python's `combat.py:389,405` walk:
            // ``[o for o in state.battlefield if o.id in attacker.blocked_by]``
            // — bf-ETB-sorted eligible blockers. CE9 closure
            // (see `docs/RUST_PARITY_GAPS.md` and the playbook §3
            // Option A): the previous `self.blockers.iter().find_map(...)`
            // walked declare-blockers push order, which diverged from
            // Python whenever blockers were declared in non-bf order.
            // (CR 510.1d says the ACTIVE player chooses damage
            // assignment order — declare order — so Python's walk is
            // technically a bug vs CR, but parity with observable
            // Python is the reference here; long-term correctness is
            // tracked as a follow-up Python-side fix.)
            //
            // CE14 closure: collect the FULL bf-ETB-sorted blocker list
            // (not just first_blocker). The no-trample branch walks every
            // blocker with `need_i = 1 if deathtouch else
            // effective_toughness - blocker.damage`, mirroring Python's
            // combat.py:419-435 per-blocker assignment loop.
            //
            // CE15 closure: the trample branch now also walks every
            // blocker with the same `need_i` formula, mirroring Python's
            // loop end-to-end. The last blocker DOES NOT absorb the
            // remainder (Python's `is_last and not has_trample` tail
            // intentionally excludes the trample case); whatever damage
            // remains after each blocker has absorbed its `need_i` spills
            // to the defender via the existing trample-spill emission.
            // Pre-CE15, the trample branch only emitted to first_blocker
            // (round-9 simplification) — multi-blocker trample under-
            // killed by N-1 blockers per engagement.
            let attacker_blocker_ids: std::collections::HashSet<u32> = self
                .blockers
                .iter()
                .filter_map(|(a, b)| if *a == attacker_id { Some(*b) } else { None })
                .collect();
            let bf_ordered_blockers: Vec<u32> = crate::bf_iter::bf_iter_etb_sorted(self)
                .into_iter()
                .filter_map(|obj| {
                    if attacker_blocker_ids.contains(&obj.id) {
                        Some(obj.id)
                    } else {
                        None
                    }
                })
                .collect();
            let first_blocker: Option<u32> = bf_ordered_blockers.first().copied();
            let attacker_has_deathtouch =
                battlefield_has_keyword(self, attacker_id, "deathtouch");
            // The `first_blocker.is_some()` check gates "blocked vs
            // unblocked": when at least one blocker exists, we enter
            // the multi-block damage assignment walk (CE14/CE15);
            // otherwise we fall through to the `else if attacker_power
            // > 0` unblocked branch. The actual loop iterates
            // `bf_ordered_blockers` directly — `first_blocker` itself
            // is no longer used inside either branch (pre-CE15 both
            // branches assigned damage to `first_blocker` as a
            // shortcut; post-CE15 both walk every blocker).
            if let Some(_first_blocker_id) = first_blocker {
                if attacker_power > 0 {
                    // Trample (CR 702.19): the attacker is required
                    // to assign at least lethal (= blocker.toughness)
                    // to the blocker, but may assign at most that
                    // amount to the blocker, sending the excess to
                    // the defender. We use the strict "lethal to
                    // blocker, excess to defender" assignment so
                    // tests can see the blocker's marked damage equal
                    // its toughness (not the attacker's full power).
                    //
                    // Without trample, the attacker assigns damage
                    // to each blocker in turn per CR 510.1c, with
                    // `need = 1 if deathtouch else blocker_toughness`
                    // per blocker (CR 702.2b's deathtouch lethal
                    // redefinition); the last blocker absorbs the
                    // remainder (mirrors Python's
                    // combat.py:419-435 "is_last gets remaining" tail).
                    if attacker_has_trample {
                        // CE15 closure: walk ALL blockers (not just
                        // first_blocker), mirroring Python's
                        // combat.py:419-435 loop shape under
                        // has_trample=True. Per CR 702.19b, the attacker
                        // must assign at least `need_i = blocker_i.
                        // toughness - blocker_i.damage` to each blocker
                        // before tramping over to the defender; CR
                        // 702.19c carves out the deathtouch case (1
                        // damage suffices per blocker because CR 702.2b
                        // makes that lethal). Whatever power remains
                        // after every blocker has absorbed its `need_i`
                        // spills to the defender via the trample-excess
                        // emission below.
                        //
                        // Python's loop tail `is_last and not has_trample:
                        // assigned = remaining` is intentionally NOT
                        // applied here — under trample the last blocker
                        // also takes only `min(remaining, need)`, and the
                        // remainder becomes the trample-through.
                        let mut remaining: i16 = attacker_power;
                        for &blocker_id_i in bf_ordered_blockers.iter() {
                            if remaining <= 0 {
                                break;
                            }
                            // CR 702.19c: deathtouch redefines lethal
                            // to 1 per blocker, freeing up the rest as
                            // trample-through.
                            let need: i16 = if attacker_has_deathtouch {
                                1
                            } else {
                                let blocker_toughness =
                                    battlefield_creature_toughness(self, blocker_id_i);
                                let blocker_damage =
                                    battlefield_creature_damage(self, blocker_id_i);
                                let raw = (blocker_toughness as i32)
                                    - (blocker_damage as i32);
                                if raw < 0 { 0 } else { raw as i16 }
                            };
                            // Trample case: no is_last-takes-remainder
                            // tail (the trample-spill code path below
                            // handles the remainder). Each blocker
                            // absorbs only its lethal floor.
                            let assigned: i16 = std::cmp::min(remaining, need);
                            if assigned > 0 {
                                for &p in &passes_for_source {
                                    plan.push(PlanEntry {
                                        source_id: attacker_id,
                                        target: DamageTarget::Creature(blocker_id_i),
                                        amount: assigned,
                                        pass: p,
                                    });
                                }
                            }
                            remaining -= assigned;
                        }
                        // Trample-through: whatever power remains after
                        // every blocker has absorbed its `need_i` spills
                        // to the defender (Python's `if has_trample and
                        // remaining > 0` block in combat.py:436-440).
                        if remaining > 0 {
                            for &p in &passes_for_source {
                                plan.push(PlanEntry {
                                    source_id: attacker_id,
                                    target: DamageTarget::Player(defending_player),
                                    amount: remaining,
                                    pass: p,
                                });
                            }
                        }
                    } else {
                        // CE14 closure: walk ALL blockers (not just
                        // first_blocker), mirroring Python's
                        // combat.py:419-435 loop shape. Per CR 702.2b
                        // + CR 510.1c, an attacker with deathtouch
                        // treats 1 damage to each blocker as lethal
                        // for the assignment-order rule, so
                        // `need = 1 if deathtouch else
                        // effective_toughness - blocker.damage`. The
                        // last blocker (no trample) absorbs whatever
                        // power remains — Python's `is_last and not
                        // has_trample: assigned = remaining` tail —
                        // which preserves the round-5 single-blocker
                        // "assign full power to first blocker"
                        // observable shape on the 1-blocker case
                        // (one blocker = is_last from i=0, takes all
                        // attacker_power).
                        let n_blockers = bf_ordered_blockers.len();
                        let mut remaining: i16 = attacker_power;
                        for (i, &blocker_id) in bf_ordered_blockers.iter().enumerate() {
                            if remaining <= 0 {
                                break;
                            }
                            let need: i16 = if attacker_has_deathtouch {
                                1
                            } else {
                                let blocker_toughness =
                                    battlefield_creature_toughness(self, blocker_id);
                                let blocker_damage =
                                    battlefield_creature_damage(self, blocker_id);
                                let raw = (blocker_toughness as i32)
                                    - (blocker_damage as i32);
                                if raw < 0 { 0 } else { raw as i16 }
                            };
                            let is_last = i == n_blockers - 1;
                            // is_last and not has_trample: absorb the
                            // remainder (Python's
                            // combat.py:427-428 tail).
                            let assigned: i16 = if is_last {
                                remaining
                            } else {
                                std::cmp::min(remaining, need)
                            };
                            if assigned > 0 {
                                for &p in &passes_for_source {
                                    plan.push(PlanEntry {
                                        source_id: attacker_id,
                                        target: DamageTarget::Creature(blocker_id),
                                        amount: assigned,
                                        pass: p,
                                    });
                                }
                            }
                            remaining -= assigned;
                        }
                    }
                }
            } else if attacker_power > 0 {
                // Unblocked: the full attacker power goes to the
                // defender. Trample does NOT double-count here.
                for &p in &passes_for_source {
                    plan.push(PlanEntry {
                        source_id: attacker_id,
                        target: DamageTarget::Player(defending_player),
                        amount: attacker_power,
                        pass: p,
                    });
                }
            }
        }
        // Back-damage from EVERY declared blocker, regardless of
        // assignment order. (Real MTG: each blocker also deals damage
        // simultaneously; multi-blocker complexity is deferred but
        // the back-damage lane is symmetric.)
        //
        // CE9 closure: iterate the blockers in bf-ETB timestamp order
        // via `bf_iter_etb_sorted`, filtered to ids that appear in
        // `self.blockers`. This matches Python's `combat.py` back-
        // damage walk, which is also bf-ordered (the per-blocker
        // damage value is unchanged — only the relative order of
        // accumulation differs, and end-state fields agree because
        // every blocker still takes its assigned damage). The
        // attacker_id per blocker is recovered via a lookup into
        // `self.blockers`.
        let blocker_to_attacker: std::collections::HashMap<u32, u32> = self
            .blockers
            .iter()
            .map(|(a, b)| (*b, *a))
            .collect();
        for obj in crate::bf_iter::bf_iter_etb_sorted(self) {
            let Some(&attacker_id) = blocker_to_attacker.get(&obj.id) else {
                continue;
            };
            let blocker_id = obj.id;
            let blocker_power = battlefield_creature_power(self, blocker_id);
            let blocker_has_first_strike =
                battlefield_has_keyword(self, blocker_id, "first_strike");
            let blocker_has_double_strike =
                battlefield_has_keyword(self, blocker_id, "double_strike");
            let passes_for_blocker = passes_for(
                blocker_has_first_strike,
                blocker_has_double_strike,
            );
            if blocker_power > 0 {
                for &p in &passes_for_blocker {
                    plan.push(PlanEntry {
                        source_id: blocker_id,
                        target: DamageTarget::Creature(attacker_id),
                        amount: blocker_power,
                        pass: p,
                    });
                }
            }
        }

        // First-strike / double-strike timing (CR 702.7 / 702.4): if
        // ANY plan entry fires in the first-strike substep, run two
        // passes. Pass 1 applies only the `Pass::FirstStrike` entries;
        // then pass 2 applies the `Pass::Regular` entries — but only
        // from sources still alive after pass 1 (a creature that took
        // lethal first-strike damage doesn't strike back, AND a
        // double-strike creature that took lethal first-strike damage
        // doesn't strike a second time).
        //
        // When no plan entry has first-strike OR double-strike, every
        // entry is `Pass::Regular`, the partition's pass_1 is empty,
        // and the two-pass code degenerates to one pass over the
        // original list — bit-identical to the pre-first-strike
        // behavior.
        let any_first_strike = plan.iter().any(|e| e.pass == Pass::FirstStrike);

        if any_first_strike {
            // Partition into (first-strike pass, regular pass).
            let (pass_1, pass_2_all): (Vec<PlanEntry>, Vec<PlanEntry>) =
                plan.into_iter().partition(|e| e.pass == Pass::FirstStrike);
            // Apply pass 1.
            self.apply_damage_pass(pass_1)?;
            // Filter pass 2:
            // (a) Source must be still alive after pass 1 — a creature
            //     that took lethal damage in step 1 doesn't strike
            //     back (gates out non-FS sources hit by a FS source,
            //     and gates out DS sources that took lethal in step
            //     1 from striking a second time).
            // (b) Creature targets — if the original creature target
            //     died in pass 1, REDIRECT the entry to
            //     `Player(defending_player)` IF the attacker (= the
            //     source) has no remaining alive blockers. This
            //     mirrors Python's "rebuild blockers list per-substep
            //     yields empty → route to defender" shape
            //     (combat.py:_resolve_damage_step at line 405/413).
            //     If at least one other blocker is still alive (e.g.
            //     a 2-blocker setup where only one died), the entry
            //     is dropped — the remaining blocker(s) absorbed
            //     their own assignments via separate plan entries,
            //     and there's no spillover.
            //     Player targets always pass — `Player` entries are
            //     never filtered here.
            //
            // SBAs are NOT fired between passes — the alive check is
            // a within-function `damage < toughness` predicate, same
            // shape as the round-9 first-strike slice.
            //
            // CE18 closure (I45). Before the closure, Rust simply
            // dropped pass-2 creature-target entries when the target
            // died in FS. That filter was CR-correct (CR 510.1h: a
            // creature stays blocked even if all blockers leave; the
            // attacker's remaining damage is wasted without trample),
            // but Python's behavior is over-permissive — it rebuilds
            // the blockers list per-substep, finds it empty, and
            // routes the damage to the defender. Per Phase 5
            // convention "converge Rust toward Python regardless of
            // CR-correctness", we redirect here.
            //
            // Pre-build alive-blocker-by-attacker map from the
            // declared `self.blockers` list: for each attacker, count
            // how many of its declared blockers are still alive
            // post-FS-pass. The map is keyed by attacker_id so the
            // pass-2 filter can decide redirect-vs-drop per attacker
            // source.
            let alive_blockers_per_attacker: std::collections::HashMap<u32, u32> = {
                let mut m: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for (att_id, blk_id) in self.blockers.iter() {
                    if battlefield_creature_alive(self, *blk_id) {
                        *m.entry(*att_id).or_insert(0) += 1;
                    }
                }
                m
            };
            // Set of declared attacker ids — used to gate the I45
            // redirect: ONLY entries whose source is an attacker
            // (`self.attackers`) qualify for the spillover-to-defender
            // redirect. A blocker's back-damage entry that targets a
            // dead attacker still gets DROPPED (matches Python's
            // single-attacker-per-blocker model — a blocker has one
            // creature target, the attacker; if the attacker is dead,
            // the back-damage is wasted, not redirected to the
            // blocker's own controller or anywhere else).
            let attacker_id_set: std::collections::HashSet<u32> =
                self.attackers.iter().copied().collect();
            let pass_2: Vec<PlanEntry> = pass_2_all
                .into_iter()
                .filter(|e| battlefield_creature_alive(self, e.source_id))
                .filter_map(|e| match e.target {
                    DamageTarget::Creature(target_id) => {
                        if battlefield_creature_alive(self, target_id) {
                            Some(e)
                        } else if attacker_id_set.contains(&e.source_id) {
                            // Source is an attacker AND its creature
                            // target is dead in FS pass. CE18 / I45
                            // redirect: if the attacker has no
                            // remaining alive blockers, redirect to
                            // the defending player. Otherwise drop
                            // (the surviving blocker(s) have their
                            // own entries and there's no spillover).
                            let n_alive = alive_blockers_per_attacker
                                .get(&e.source_id)
                                .copied()
                                .unwrap_or(0);
                            if n_alive == 0 {
                                Some(PlanEntry {
                                    source_id: e.source_id,
                                    target: DamageTarget::Player(defending_player),
                                    amount: e.amount,
                                    pass: e.pass,
                                })
                            } else {
                                None
                            }
                        } else {
                            // Source is a blocker (back-damage entry)
                            // whose attacker target died in FS pass.
                            // No redirect — back-damage is wasted.
                            None
                        }
                    }
                    DamageTarget::Player(_) => Some(e),
                })
                .collect();
            self.apply_damage_pass(pass_2)?;
        } else {
            self.apply_damage_pass(plan)?;
        }
        Ok(())
    }

    /// End-of-combat scaffolding cleanup: empty `self.attackers` and
    /// `self.blockers`.
    ///
    /// Does NOT untap declared attackers — that's the untap-step's
    /// job (CR 502.1). Does NOT clear marked damage on creatures —
    /// damage wears off in the cleanup step (CR 514.2), which lives
    /// in `state_ops_damage.rs:clear_all_damage`.
    fn clear_combat(&mut self) {
        self.attackers.clear();
        self.blockers.clear();
    }
}

// Private (non-pymethods) impl for damage-pass application. Lives in
// a sibling `impl` so `#[pymethods]` doesn't try to expose it to Python
// (taking a `Vec<PlanEntry>` of a Rust-only struct, it can't be
// marshalled anyway).
impl GameState {
    /// Apply one damage pass: iterate the entries, mutate damage
    /// marks / player life, fire deathtouch upgrades and lifelink
    /// gains. Shared between the single-pass (no first-strike) and
    /// two-pass (first-strike present) branches of
    /// `deal_combat_damage` so the deathtouch / lifelink composition
    /// stays in one place.
    ///
    /// **`state.resolving_combat_damage_amount` write site (I32).**
    /// For each per-event damage entry, the slot is set to the literal
    /// amount BEFORE the damage is applied (so any downstream effect
    /// that reads it sees this event's amount), and reset to 0 AFTER
    /// the per-entry mutations + lifelink fire. Mirrors Python's
    /// per-event set/reset boundary: `combat.py:618,623` sets/resets
    /// around `ability.effect.resolve` for granted dealt-damage
    /// triggers; `actions.py:2684,2699` does the same around
    /// `resolve_top_of_stack` for stack-pushed triggers. Rust today
    /// has no combat-damage trigger dispatch in the apply loop, but
    /// the slot is written here so:
    ///   1. The `Effect::DrawCardsEqualToResolvingCombatDamage`
    ///      arm — which reads the slot at resolve-arm time — sees a
    ///      non-zero value when a future stack-pushed trigger from
    ///      this damage event resolves while the slot is still set.
    ///   2. The `set_resolving_combat_damage_amount` test setter is
    ///      no longer the only production writer, closing the
    ///      `SEARCH_LIMITATIONS.md` deferral.
    ///   3. Between events the slot is 0, so a Late-resolving trigger
    ///      doesn't see a stale value from a prior event (the reset
    ///      is what keeps event-N's amount from leaking into event-
    ///      N+1's resolution).
    fn apply_damage_pass(&mut self, entries: Vec<PlanEntry>) -> PyResult<()> {
        for entry in entries {
            // Read the source's keywords + controller BEFORE applying
            // mutations. Deathtouch upgrades the recipient's mark to
            // at least its toughness (CR 702.2b — only fires on
            // nonzero damage, which the plan already filters via
            // `> 0` guards). Lifelink credits the source's controller
            // with `amount` life regardless of target kind.
            let source_has_deathtouch =
                battlefield_has_keyword(self, entry.source_id, "deathtouch");
            let source_has_lifelink =
                battlefield_has_keyword(self, entry.source_id, "lifelink");
            let source_controller = battlefield_creature_controller(self, entry.source_id);

            // I32: set the resolving-combat-damage-amount slot for the
            // duration of this event so any downstream effect reading
            // `state.resolving_combat_damage_amount` (notably
            // `Effect::DrawCardsEqualToResolvingCombatDamage`, the
            // Starwinder / Fear-of-Failed-Tests family) sees this
            // event's amount. Reset to 0 after the per-entry mutations
            // and lifelink fire so the slot is 0 between events.
            self.resolving_combat_damage_amount = entry.amount as i32;

            // Capture the creature-target id (if any) BEFORE applying
            // mutations so the post-damage DealsCombatDamage dispatch
            // can pass it as the Target slot id even if the target
            // would later die from this damage. Player targets are not
            // wired through the predicate-style dispatch yet — see
            // `dispatch_event_multi_subject` doc for the gap.
            let creature_target_id: Option<u32> = match entry.target {
                DamageTarget::Creature(id) => Some(id),
                DamageTarget::Player(_) => None,
            };

            match entry.target {
                DamageTarget::Creature(obj_id) => {
                    let recipient_toughness = battlefield_creature_toughness(self, obj_id);
                    let amount = entry.amount;
                    mutate_battlefield_obj(self, obj_id, |obj| {
                        obj.damage = obj.damage.saturating_add(amount);
                        // Deathtouch: upgrade the recipient's marked
                        // damage to at least its toughness so the SBA
                        // slice destroys it. The plan only contains
                        // entries with `amount > 0`, satisfying CR
                        // 702.2b's "nonzero damage" precondition.
                        if source_has_deathtouch && recipient_toughness > 0 {
                            if (obj.damage as i32) < (recipient_toughness as i32) {
                                obj.damage = recipient_toughness;
                            }
                        }
                    });
                }
                DamageTarget::Player(player_id) => {
                    let idx = player_id as usize;
                    if idx < self.players.len() {
                        self.players[idx].life -= entry.amount as i32;
                    }
                }
            }

            // Lifelink: gain life on the source's controller equal to
            // the literal damage amount. Fires once per damage entry
            // (so a lifelink attacker hitting a player gains its
            // power; a lifelink blocker hitting an attacker gains
            // its power; trample-excess entries also fire lifelink
            // for their excess amount).
            if source_has_lifelink {
                if let Some(controller) = source_controller {
                    let cidx = controller as usize;
                    if cidx < self.players.len() {
                        self.players[cidx].life += entry.amount as i32;
                    }
                }
            }

            // ETB-Subject substrate consumer slice: dispatch
            // `Event::DealsCombatDamage` for every CREATURE-target
            // damage entry. Player-target entries skip this dispatch
            // (see `dispatch_event_multi_subject` doc for the gap).
            // Mirrors Python's per-event invocation at
            // `src/mtg_bot/combat.py:729-742`.
            //
            // Subject map: Source = dealing creature; Target =
            // creature receiving damage. The default subject
            // (passed as `subject_id` to existing implicit-subject
            // predicates and SourceIsSelf) is the dealing creature —
            // matches the canonical "whenever ~ deals combat damage"
            // self-trigger shape.
            //
            // Fired AFTER damage mutations so a trigger that reads
            // post-damage state (e.g. an SBA-survivor predicate) sees
            // the updated values. Fired BEFORE the resolving-amount
            // slot reset so any DrawCardsEqualToResolvingCombatDamage
            // payload on this trigger reads the correct amount when
            // the trigger eventually resolves (caveat: the trigger
            // resolves from the stack, by which time the slot has
            // been zeroed — Python uses the same shape and accepts
            // that resolving-time-read divergence; see I32 notes for
            // the slot lifecycle).
            if let Some(target_id) = creature_target_id {
                let subjects_by_slot = [
                    (crate::cards::SubjectSlot::Source, entry.source_id),
                    (crate::cards::SubjectSlot::Target, target_id),
                ];
                crate::state_ops_resolve::dispatch_event_multi_subject(
                    self,
                    crate::cards::Event::DealsCombatDamage,
                    entry.source_id,
                    &subjects_by_slot,
                )?;
            }

            // I32 reset: clear the slot between events so a future
            // Late-resolving trigger doesn't read a stale value from
            // a prior event. Matches Python's `finally: state.
            // resolving_combat_damage_amount = 0` reset shape.
            self.resolving_combat_damage_amount = 0;
        }
        Ok(())
    }
}
