//! `legal_decisions(player_id) -> list[Action]` enumerator.
//!
//! The Phase 3 keystone: emits the closed set of legal Actions the
//! priority-holder can take right now. Python's MCTS / ISMCTS drives
//! off this enumeration; Rust must match Python's emission order
//! exactly (Action types + per-type inner ordering) for trajectory
//! parity in `scripts/parity_smoke.py`.
//!
//! ## Emission-order audit vs Python's `legal_actions`
//!
//! Python's `src/mtg_bot/legal.py:legal_actions(state, player_id)` is
//! the source of truth. The relevant line ranges this slice mirrors:
//!
//! - **Line 178**: `actions: list[Action] = [PassPriority()]` —
//!   PassPriority is the FIRST element, prepended before any other
//!   variant. Future slices that grow the enumerator must keep
//!   PassPriority at index 0; the task brief's "PassPriority is last"
//!   wording is incorrect — the Python source has it first, and
//!   parity is bit-for-bit relative to Python.
//! - **Line 180-181**: `if not _has_priority(state, player_id):
//!   return actions` — the priority gate. Only the priority holder
//!   gets a non-trivial list; off-priority callers see
//!   `[PassPriority()]` and bail.
//! - **Lines 21-22**: `_has_priority` compares
//!   `state.players[state.priority_player_idx].id == player_id`. We
//!   use `state.priority_player == player_id` directly (Rust stores
//!   player ids equal to their indices in the 2-player model).
//! - **Lines 25-32**: `_is_sorcery_timing` — active player == priority
//!   player, current step is in `MAIN_STEPS`, stack is empty, AND
//!   priority is held. PlayLand only emits inside sorcery timing.
//! - **Line 303-304**: `if not _is_sorcery_timing(...): return actions`
//!   — after the (deferred) instant-speed cast block, Python bails
//!   if it isn't sorcery timing. First-slice's no-instant-casts world
//!   means we can short-circuit on this check at the top.
//! - **Lines 319-329**: PlayLand emission. `if p.lands_played_this_turn
//!   < max_lands: ... for card in p.hand: if card.printing.is_land:
//!   ... actions.append(PlayLand(card.id))`. Hand iteration order
//!   matches Python's, with a dedup by `(printing.name,)` key.
//!
//! ## Deliberate divergence: one-land-per-turn gate
//!
//! Python's line 319 reads `if p.lands_played_this_turn < max_lands`.
//! Rust's `GameState` has no `lands_played_this_turn` field today —
//! the parallel-slice file-isolation rule forbids adding it to
//! `state.rs` from this slice. First-slice behavior: emit a PlayLand
//! action for every Land in hand regardless of the per-turn limit.
//! Documented in `docs/SEARCH_LIMITATIONS.md`; caller / future slice
//! enforcement (either by adding the field + checking here, or by
//! re-checking via the underlying `play_land` primitive's mutation-
//! tracker) is the Phase 5 fix.
//!
//! ## Hand-card dedup by printing name (line 322-329)
//!
//! Two copies of the same printing in hand reach identical end-states
//! modulo the GameObject id (Python's invariant — see the long comment
//! at lines 220-229). The dedup key in Python is `(printing.name,)`;
//! mirroring that here makes the emission list collapse "two Forests"
//! to a single PlayLand action with the first hand-order Forest's id.
//!
//! ## CastSpell extension (rust(legal-decisions-cast-spell))
//!
//! Mirrors Python's sorcery-speed CastSpell emission at
//! `src/mtg_bot/legal.py:335-445` (the `_is_sorcery_speed_spell` loop
//! that comes AFTER PlayLand). The Python source also has an
//! instant-speed CastSpell block at lines 230-301 — that block runs
//! BEFORE the sorcery-timing gate (line 303), so on a state that fails
//! sorcery timing Python may still emit instant-speed casts. **First
//! slice ignores the instant-speed block**: every cast is gated by the
//! same sorcery-timing short-circuit at line 148 of this file. Future
//! slices add the instant-speed window by moving the CastSpell loop
//! ABOVE the sorcery-timing return and threading the
//! `card.printing.is_instant || "flash" in keywords` predicate; the
//! per-card emission code stays the same.
//!
//! **Deferred mana-affordability check.** Python's `_can_cast` (lines
//! 252-254 / 383-385) calls `state.can_pay_cost(...)` to refuse spells
//! whose mana cost the player can't pay. Rust's `state_ops_mana.rs`
//! has only the destructive `pay_mana_cost(player_id, cost_dict)`
//! primitive — no non-destructive `can_pay_cost` predicate exists
//! yet. First-slice CastSpell emits for EVERY non-Land in hand
//! regardless of mana availability; a future slice adds the
//! affordability check (either by extracting a `can_pay_cost` /
//! `try_pay_mana_cost`-with-rollback predicate or by snapshotting
//! the mana pool before each `pay_mana_cost` attempt). Documented
//! in `docs/SEARCH_LIMITATIONS.md` alongside the one-land-per-turn
//! gate.
//!
//! **Deferred alt-cost flags.** Python's CastSpell loop emits up to
//! four extra dicts per card for `kicked=True`, `use_warp=True`,
//! `use_impending=True`, and `cast_for_free=True` (the kicker /
//! warp / impending / Weftwalking variants, each gated by a
//! per-printing predicate on `kicker_cost` / `warp_cost` /
//! `impending_cost` / Weftwalking grant). First-slice emits ONLY the
//! base "no-flags" CastSpell with `use_warp=False, kicked=False,
//! cast_for_free=False, use_impending=False, cast_room_half=None` —
//! the same six-key shape `action_to_pydict` produces and
//! `parse_action` accepts. The alt-cost / Room-half emissions land
//! in follow-up slices once the alt-cost detector primitive ports
//! over.
//!
//! **Sorcery-speed-spell filter.** Python's `_is_sorcery_speed_spell`
//! (lines 339-345) accepts a card if it is a Creature, a Sorcery, or
//! a non-Land non-Instant Permanent (artifact, enchantment,
//! battle, ...). First-slice approximates this with the inverse: any
//! non-Land card in hand is a CastSpell candidate. The synthetic-
//! state registry only has Forest (Land) and Llanowar Elves /
//! Grizzly Bears (Creature) — no Instants exist to mis-classify, so
//! the approximation is correct for the entire test surface. When
//! the registry grows Instants, the filter must add the
//! `is_instant` exclusion to match Python (Instants get cast in the
//! instant-speed block, not the sorcery-speed one, so they would
//! emit twice).
//!
//! ## Dedup key shape
//!
//! Python's CastSpell dedup key at lines 233-240 / 352-359 is the
//! 6-tuple `(printing.name, use_warp, kicked, cast_for_free,
//! use_impending, cast_room_half)`. With all flags False/None in the
//! first slice, the tuple collapses to effectively `(printing.name,)`
//! — same dedup behavior as PlayLand. The Rust HashSet is keyed on
//! the printing name string slice; when the alt-cost flags land, the
//! future slice extends the key to a 6-tuple (string + 4 bools +
//! `Option<String>`).
//!
//! ## DeclareAttackers extension (rust(legal-decisions-declare-attackers))
//!
//! **Deliberate divergence from Python.** Python's `legal_actions`
//! does NOT emit DeclareAttackers. The Python engine docstring
//! (`src/mtg_bot/engine.py:10-19`) is explicit: "the combat-declaration
//! actions (`DeclareAttackers`, `DeclareBlockers`) that aren't
//! enumerable as legal moves because their decision space is
//! combinatorial." Python instead uses a factored search-side approach
//! in `src/mtg_bot/search/combat_decisions.py` — `AttackerDecision`
//! (one per eligible attacker, 2 branches each) feeds MCTS at search
//! time, and the adapter accumulates them into a single engine-level
//! `DeclareAttackers(frozenset[int])` once all per-creature decisions
//! are collected.
//!
//! For the Rust port's first-slice we emit a powerset enumeration
//! directly from `legal_decisions` so a uniform "one Python dict per
//! legal Action" iterator covers the DECLARE_ATTACKERS phase too.
//! This is a Rust-only emission (no Python counterpart) — the
//! parity-smoke harness skips comparing emission sets during the
//! DECLARE_ATTACKERS phase. When the factored-search shape lands in
//! Rust (one `AttackerDecision` action per eligible attacker, mirroring
//! `search/combat_decisions.py:44-55`), this powerset block goes away
//! and the per-attacker enumeration replaces it.
//!
//! **Gating.** Emit DeclareAttackers ONLY when:
//! - `phase == PHASE_DECLARE_ATTACKERS` (round-2 step constant; see
//!   `state_ops_steps.rs:63`).
//! - `active_player == player_id` (only the active player declares
//!   attackers — CR 508.1).
//! - `priority_player == player_id` (already verified by the early
//!   priority gate at line 217).
//! - `stack.is_empty()` (declare-attackers happens before any spells
//!   resolve in the combat step).
//!
//! **Eligibility** (mirrors `src/mtg_bot/combat.py:eligible_attackers`
//! at lines 15-23):
//! - On the active player's battlefield.
//! - `card_types` contains "Creature" (mirrors Python's
//!   `is_creature_now`).
//! - `!tapped` (CR 508.1a — a tapped creature can't attack).
//! - `!summoning_sick || has_haste` keyword (CR 302.1 + 702.10).
//!
//! **Deferred** (silently included as candidates; the
//! `declare_attacker` primitive's bool-return filters them at apply
//! time):
//! - `"defender"` keyword (CR 702.3) — Python's
//!   `eligible_attackers` filters this out at enumeration time;
//!   first slice defers to the primitive's per-id gate at apply.
//! - Continuous "can't attack" restrictions (Propaganda, Ghostly
//!   Prison, etc.) — no continuous-effects layer in the Rust engine
//!   yet.
//! - Defending player's "must block" / intimidate / fear and other
//!   defender-side keywords that change the *blocker* legality
//!   (these don't change attacker eligibility but interact in the
//!   declare-blockers step which is out of this slice's scope).
//!
//! **Emission shape**: one Python dict per subset, matching
//! `actions.rs:action_to_pydict` for `Action::DeclareAttackers`:
//! `{"kind": "declare_attackers", "attacker_ids": [id1, id2, ...]}`.
//! The `attacker_ids` list is in battlefield iteration order (stable,
//! matches `Vec` ordering); the empty set emits `attacker_ids = []`.
//!
//! **Powerset cap.** With `n` eligible attackers and
//! `n <= DECLARE_ATTACKERS_POWERSET_CAP` (4), emit all `2^n` subsets.
//! With `n > 4`, emit only two dicts: the empty subset ("don't
//! attack") and the full subset ("attack with everyone"). The cap
//! exists so a 10-creature board doesn't produce 1024 dicts per
//! `legal_decisions` call; a future factored-search slice replaces
//! this with `n` separate `AttackerDecision`-style dicts.
//!
//! ## ActivateAbility extension (rust(legal-decisions-activate-ability))
//!
//! Mirrors Python's ActivateAbility emission block at
//! `src/mtg_bot/legal.py:612-829` (the loop opening
//! `# Activated abilities on permanents this player controls.`).
//! Python's `ActivateAbility(source_id, ability_index)` constructor
//! call is at lines 827-829.
//!
//! **Emission order vs Python.** Python's ActivateAbility loop lives
//! AFTER the sorcery-timing early-return at lines 303-304. Although
//! activated abilities are nominally instant-speed (per the CR),
//! Python's `legal_actions` only emits them when sorcery timing
//! holds. The individual `ability.sorcery_speed` check at line 620 is
//! then a no-op (function already returned before this point if it
//! wasn't sorcery timing). First-slice mirrors that: ActivateAbility
//! lives INSIDE the sorcery-timing block, after PlayLand + CastSpell
//! emission, before any post-block work. The instant-speed window for
//! mana abilities is a Python-side limitation we inherit unchanged
//! for trajectory parity.
//!
//! **First-slice scope: mana-producer permanents only.** Python's
//! `CardPrinting` has an `activated_abilities: tuple[ActivatedAbility,
//! ...]` field (see `src/mtg_bot/cards/base.py:490`) listing every
//! activated ability the printing grants. Rust's `CardPrinting` (see
//! `cards.rs:361-396`) does NOT have an `activated_abilities` field —
//! bridge slices through round-23 added `mana_produced`, `keywords`,
//! `card_types`, `power`/`toughness`, but `activated_abilities` is a
//! deferred bridge extension because it requires porting the
//! `ActivatedAbility` + `ActivationCost` + per-effect dataclass tree.
//!
//! Without `activated_abilities`, we can't iterate abilities directly.
//! **First-slice approximation**: treat every untapped permanent on
//! the priority player's battlefield whose printing has
//! `mana_produced.is_some()` as having a single implicit "tap for
//! mana" ability at `ability_index=0`. This covers Forests + every
//! other basic-land equivalent in the Rust card pool today; once the
//! `activated_abilities` bridge ports over (a bigger slice that wires
//! the full ability/cost/effect dataclass tree through PyO3), this
//! shortcut goes away and the enumerator iterates
//! `effective_activated_abilities(state, obj)` per Python's line
//! 617-618. Documented in `docs/SEARCH_LIMITATIONS.md`.
//!
//! **Gating** (mirror Python):
//! - Source on the priority player's battlefield (Python's outer loop
//!   filters `obj.controller_id != player_id` at line 614 — we walk
//!   `self.players[idx].battlefield` directly, which has the same
//!   semantic in the per-player-zone Rust model).
//! - Source untapped: tap-for-mana abilities pay `{T}` as cost (CR
//!   605.1a — mana abilities tap the source). Python's line 658-660
//!   gates `if ability.cost.tap_self and obj.tapped: continue`.
//! - **Summoning sickness gate** (parity gap G6 closure): a
//!   permanent whose `summoning_sick` flag is true does NOT emit a
//!   tap-for-mana ActivateAbility. Mirrors Python's
//!   `legal.py:661-663` short-circuit inside the `tap_self`
//!   cost-payability block. The general gate is
//!   `summoning_sick && cost.tap_self`; mana abilities always tap
//!   (CR 605.1a), so for this first-slice's mana-only filter
//!   `summoning_sick` alone is sufficient. A future slice landing
//!   non-mana activated abilities will narrow the gate to skip
//!   ONLY tap-cost abilities. The haste-keyword exception (CR
//!   702.10 — haste creatures shed sickness on entry) lands with
//!   the full `activated_abilities` bridge.
//!
//! **Dispatch is a known stub.** The Rust `Action::ActivateAbility`
//! variant's `apply_action` arm at `state_ops_actions.rs:248-277`
//! raises `ValueError("ActivateAbility dispatch not implemented
//! yet ...")` because no Rust `activate_ability` primitive exists.
//! The trajectory-mode `parity-smoke` harness's retry loop catches
//! the ValueError and skips. This slice ships emission only; dispatch
//! is the next slice's scope.
//!
//! **Dict shape**: `{"kind": "activate_ability", "source_id": <id>,
//! "ability_index": 0}`. Round-trips through `parse_action` (round-23
//! `Action::ActivateAbility` variant) and `action_to_pydict`
//! unchanged.
//!
//! ## DeclareBlockers extension (rust(legal-decisions-declare-blockers))
//!
//! **Deliberate divergence from Python** — same shape as the
//! DeclareAttackers extension above. Python's `legal_actions` does
//! NOT emit DeclareBlockers (see `src/mtg_bot/engine.py:10-19` for
//! the explicit "combat-declaration actions ... aren't enumerable as
//! legal moves because their decision space is combinatorial"
//! statement). Python uses factored `BlockerDecision`s in
//! `src/mtg_bot/search/combat_decisions.py` — one decision per
//! eligible blocker, with one branch per attacker plus a "don't
//! block" branch — fed to MCTS at search time. The adapter
//! accumulates per-blocker decisions into a single engine-level
//! `DeclareBlockers(assignments=tuple[tuple[int,int], ...])` at
//! apply time.
//!
//! For the Rust port we emit a direct enumeration of assignments so
//! `legal_decisions` covers the DECLARE_BLOCKERS phase uniformly.
//! Combinatorial complexity is HARDER than DeclareAttackers: with N
//! eligible blockers and M attackers, each blocker independently
//! picks one of `M + 1` options (block attacker_j, or don't block),
//! yielding `(M + 1)^N` total assignments. This explodes faster
//! than DeclareAttackers's `2^N` — a 4-blocker / 4-attacker board
//! has `5^4 = 625` assignments. The cap is critical.
//!
//! **Gating**: emit DeclareBlockers ONLY when:
//! - `phase == PHASE_DECLARE_BLOCKERS` (see `state_ops_steps.rs:65`).
//! - `priority_player == player_id` (already verified by the
//!   priority gate above).
//! - `priority_player != active_player` — defender declares blockers,
//!   not the active player (CR 509.1). This is the OPPOSITE of
//!   DeclareAttackers's `active == player_id` check.
//! - `stack.is_empty()` (declare-blockers happens before any spells
//!   resolve in the combat step).
//!
//! **Eligibility** (mirrors `src/mtg_bot/combat.py:eligible_blockers`
//! at lines 26-32):
//! - On the defender's battlefield.
//! - `card_types` contains "Creature".
//! - `!tapped` (CR 509.1a — a tapped creature can't block).
//! - **Summoning sickness does NOT exclude** (CR 302.1 only restricts
//!   attacking and tap-cost activated abilities; freshly-summoned
//!   creatures CAN block). This is the key sickness-vs-attacking
//!   asymmetry.
//!
//! **Deferred**: keyword-driven block-legality filters (flying / reach
//! / can_only_block_creatures_with_flying, color protection,
//! is_cant_be_blocked, skulk) are NOT applied at enumeration. The
//! `declare_blocker` primitive's per-pair bool-return filters these
//! at apply time — same silent-skip pattern as DeclareAttackers's
//! defender-keyword deferral. Wide-emission + apply-time filter is
//! the first-slice cost; future slices can add the keyword filter at
//! enumeration once `effective_keywords` ports through PyO3.
//!
//! **Emission shape**: one Python dict per assignment, matching
//! `actions.rs:action_to_pydict` for `Action::DeclareBlockers`:
//! `{"kind": "declare_blockers", "assignments": [[blocker_id,
//! attacker_id], ...]}`. The `assignments` list contains only the
//! blockers that ARE blocking — "doesn't block" is encoded by
//! omission, not by a `(blocker_id, None)` pair. This matches
//! Python's `DeclareBlockers(assignments=tuple[tuple[int, int],
//! ...])` dataclass shape.
//!
//! **Combinatorial product cap.** With `n_blockers <=
//! DECLARE_BLOCKERS_PRODUCT_CAP` (3) AND `m_attackers <=
//! DECLARE_BLOCKERS_MAX_ATTACKERS` (3), emit all `(M+1)^N`
//! assignments — each blocker independently chooses among M
//! attackers + "don't block." With either dimension over its cap,
//! emit ONLY the "no blocks" assignment (an empty Vec). The dropped
//! middle assignments are still reachable at apply time via direct
//! `Action::DeclareBlockers` construction; they're just not
//! enumerated by `legal_decisions`. Same tunable-knob story as
//! DeclareAttackers's `POWERSET_CAP`: the factored-search shape from
//! `search/combat_decisions.py:120-178` is the eventual replacement.
//! See "DeclareBlockers Rust-only enumeration" in
//! `docs/SEARCH_LIMITATIONS.md`.

use crate::state::GameState;
use crate::state_ops_steps::{
    PHASE_DECLARE_ATTACKERS, PHASE_DECLARE_BLOCKERS, PHASE_MAIN1, PHASE_MAIN2,
};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

/// First-slice cap on the powerset enumeration in
/// `DeclareAttackers` emission. With `n <= 4` eligible attackers the
/// enumerator emits the full `2^n` subsets (so a small board produces
/// exhaustive choices, mirroring the search-side combinatorial pattern
/// in Python's `search/combat_decisions.py`). With `n > 4`, the
/// enumerator emits only the canonical "attack with nothing" + "attack
/// with everyone" pair so the dict-count stays bounded at 2 instead
/// of exploding to 32+. Tunable knob: future slices can lift this once
/// the factored-search shape (one AttackerDecision per eligible
/// attacker, per `search/combat_decisions.py:44-55`) lands and replaces
/// the powerset emission. See "DeclareAttackers Rust-only enumeration"
/// in `docs/SEARCH_LIMITATIONS.md`.
const DECLARE_ATTACKERS_POWERSET_CAP: usize = 4;

/// First-slice cap on the product enumeration in `DeclareBlockers`
/// emission. With `n_blockers <= DECLARE_BLOCKERS_PRODUCT_CAP` (3)
/// AND `m_attackers <= DECLARE_BLOCKERS_MAX_ATTACKERS` (3), the
/// enumerator emits the full `(M+1)^N` assignments — every blocker
/// independently picks one of `M` attackers OR "don't block." Worst
/// case at the cap: `(3+1)^3 = 64` dicts. Above either threshold the
/// enumerator emits only the canonical "no blocks" assignment (an
/// empty Vec) so the dict-count stays bounded at 1 instead of
/// exploding to thousands. Combinatorial growth is harder than
/// DeclareAttackers's `2^N` — a 4×4 board would be `5^4 = 625`, a
/// 5×5 board `6^5 = 7776`. Tunable knob: future slices can lift
/// either cap once the factored-search shape (one BlockerDecision
/// per eligible blocker, per `search/combat_decisions.py:120-178`)
/// lands and replaces the product emission. See "DeclareBlockers
/// Rust-only enumeration" in `docs/SEARCH_LIMITATIONS.md`.
const DECLARE_BLOCKERS_PRODUCT_CAP: usize = 3;

/// Companion cap on the attacker dimension of the `DeclareBlockers`
/// product. See `DECLARE_BLOCKERS_PRODUCT_CAP` for the rationale —
/// both dimensions must be at-or-below their cap for the full
/// enumeration to emit; either dimension over its cap falls through
/// to the bounded "no-blocks-only" branch.
const DECLARE_BLOCKERS_MAX_ATTACKERS: usize = 3;

fn check_player_id(player_id: u8) -> PyResult<usize> {
    let idx = player_id as usize;
    if idx >= 2 {
        return Err(PyIndexError::new_err(format!(
            "player_id {player_id} out of range (must be 0 or 1)"
        )));
    }
    Ok(idx)
}

/// Build a `{"kind": "pass_priority"}` dict.
fn pass_priority_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "pass_priority")?;
    Ok(d)
}

/// Build a `{"kind": "play_land", "card_id": <id>}` dict.
fn play_land_dict(py: Python<'_>, card_id: u32) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "play_land")?;
    d.set_item("card_id", card_id)?;
    Ok(d)
}

/// Build a `{"kind": "cast_spell", "card_id": <id>, ...}` dict with all
/// six fields populated. Mirrors `action_to_pydict`'s symmetric-emit
/// shape (`actions.rs:385-410`): every CastSpell dict — emitted from
/// `legal_decisions` or from `action_to_pydict` round-trip — carries
/// the full 7 keys (kind + card_id + 5 flag fields) so consumers can
/// rely on key-presence without `dict.get(...)`-with-default.
///
/// First-slice defaults: `use_warp=False, kicked=False,
/// cast_for_free=False, use_impending=False, cast_room_half=None`.
/// Future alt-cost slices add new constructor calls that flip individual
/// flags; the helper's signature grows to take them as parameters.
fn cast_spell_dict(py: Python<'_>, card_id: u32) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "cast_spell")?;
    d.set_item("card_id", card_id)?;
    d.set_item("use_warp", false)?;
    d.set_item("kicked", false)?;
    d.set_item("cast_for_free", false)?;
    d.set_item("use_impending", false)?;
    // None as cast_room_half — Python None at the dict layer.
    d.set_item("cast_room_half", py.None())?;
    Ok(d)
}

/// Build a `{"kind": "activate_ability", "source_id": <id>,
/// "ability_index": <u32>}` dict. Mirrors
/// `actions.rs:action_to_pydict`'s `Action::ActivateAbility` arm so a
/// dict produced here round-trips through `parse_action` and back
/// without key drift. First-slice always uses `ability_index = 0`
/// because Rust's `CardPrinting` doesn't carry an
/// `activated_abilities` list yet — see module docstring's
/// "ActivateAbility extension" section.
fn activate_ability_dict(
    py: Python<'_>,
    source_id: u32,
    ability_index: u32,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "activate_ability")?;
    d.set_item("source_id", source_id)?;
    d.set_item("ability_index", ability_index)?;
    Ok(d)
}

/// Build a `{"kind": "declare_attackers", "attacker_ids": [...]}` dict.
/// Mirrors `actions.rs:action_to_pydict`'s `Action::DeclareAttackers`
/// arm so a dict produced here round-trips through `parse_action` and
/// back without key drift. The `attacker_ids` Vec is cloned into a
/// fresh Python list — PyO3's `Vec<u32>` -> Python list conversion
/// preserves Vec iteration order, so the subset's order on the wire
/// equals the order the enumerator pushed ids in.
fn declare_attackers_dict<'py>(
    py: Python<'py>,
    attacker_ids: &[u32],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "declare_attackers")?;
    d.set_item("attacker_ids", attacker_ids.to_vec())?;
    Ok(d)
}

/// Build a `{"kind": "declare_blockers", "assignments":
/// [[blocker_id, attacker_id], ...]}` dict. Mirrors
/// `actions.rs:action_to_pydict`'s `Action::DeclareBlockers` arm so
/// a dict produced here round-trips through `parse_action` and back
/// without key drift. The `assignments` Vec is cloned into a fresh
/// Python list of 2-tuples — PyO3's `Vec<(u32, u32)>` -> Python list
/// conversion preserves Vec iteration order. Blockers that don't
/// block are omitted from the list (not encoded as a sentinel pair),
/// matching Python's `DeclareBlockers(assignments=tuple[tuple[int,
/// int], ...])` dataclass shape.
fn declare_blockers_dict<'py>(
    py: Python<'py>,
    assignments: &[(u32, u32)],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "declare_blockers")?;
    d.set_item("assignments", assignments.to_vec())?;
    Ok(d)
}

#[pymethods]
impl GameState {
    /// Return the list of legal Actions for `player_id` at this
    /// priority moment. Mirrors Python's
    /// `src/mtg_bot/legal.py:legal_actions` emission order (lines
    /// 168-895) — same Action types in the same sequence with the
    /// same per-type tie-breaking, so Python search drivers (MCTS /
    /// ISMCTS) can be drop-in.
    ///
    /// **First-slice scope (rust(legal-decisions-skeleton)):**
    /// - `PassPriority` — always emitted at index 0 (Python's
    ///   `legal.py:178`).
    /// - `PlayLand` — one per Land in the priority player's hand,
    ///   under sorcery-timing gates (Python's `legal.py:319-329`),
    ///   deduped by printing name.
    ///
    /// `CastSpell`, `ActivateAbility`, `DeclareAttackers`, and
    /// `DeclareBlockers` are now also emitted in their respective
    /// phases (the latter two as Rust-only divergences vs Python; see
    /// module docstring).
    ///
    /// Returns a list of Action dicts (round-trip-able with
    /// `apply_action`). Raises `IndexError` on an out-of-range
    /// `player_id`.
    fn legal_decisions<'py>(
        &self,
        py: Python<'py>,
        player_id: u8,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let idx = check_player_id(player_id)?;

        // Python line 178: actions: list[Action] = [PassPriority()]
        let mut actions: Vec<Bound<'py, PyDict>> = Vec::with_capacity(1);
        actions.push(pass_priority_dict(py)?);

        // Python lines 180-181: priority gate. Off-priority callers
        // bail with just the PassPriority action.
        if self.priority_player != player_id {
            return Ok(actions);
        }

        // DeclareAttackers emission (Rust-only — Python's
        // `legal_actions` does NOT emit this variant; see the module
        // docstring's "DeclareAttackers extension" section for the
        // divergence rationale + the `search/combat_decisions.py`
        // factored-search alternative Python uses).
        //
        // Gate: `phase == PHASE_DECLARE_ATTACKERS` AND
        // `active_player == player_id` AND `stack.is_empty()`.
        // (Priority is already confirmed above.)
        //
        // Eligibility per `src/mtg_bot/combat.py:eligible_attackers`
        // (lines 15-23): on the active player's battlefield, is a
        // Creature, not tapped, not summoning-sick OR has haste.
        // Defender + continuous-restriction filters are deferred —
        // see module docstring.
        if self.phase == PHASE_DECLARE_ATTACKERS
            && self.active_player == player_id
            && self.stack.is_empty()
        {
            // Collect eligible attackers in battlefield iteration
            // order (stable across calls; first-slice doesn't
            // canonicalize ids in any other way).
            let mut eligible: Vec<u32> = Vec::new();
            for obj in self.players[idx].battlefield.iter() {
                if obj.tapped {
                    continue;
                }
                let Some(printing) = self.registry.lookup(obj.printing_id) else {
                    continue;
                };
                if !printing.card_types.iter().any(|t| t == "Creature") {
                    continue;
                }
                let has_haste = printing.keywords.iter().any(|k| k == "haste");
                if obj.summoning_sick && !has_haste {
                    continue;
                }
                eligible.push(obj.id);
            }

            let n = eligible.len();
            if n <= DECLARE_ATTACKERS_POWERSET_CAP {
                // Full powerset: bitmask 0 (= empty set, "attack with
                // nothing") through 2^n - 1 (= full set, "attack with
                // everyone"). Each subset's ids are pushed in
                // `eligible`-iteration order (smallest bit index
                // first) so the wire shape is deterministic.
                let total = 1u32 << n;
                for bitmask in 0..total {
                    let mut subset: Vec<u32> = Vec::with_capacity(n);
                    for i in 0..n {
                        if (bitmask >> i) & 1 == 1 {
                            subset.push(eligible[i]);
                        }
                    }
                    actions.push(declare_attackers_dict(py, &subset)?);
                }
            } else {
                // Past the cap: emit only the two canonical extremes
                // ("don't attack" + "attack with everyone") so the
                // dict-count stays bounded. The dropped middle
                // subsets are still reachable at apply time via
                // direct `Action::DeclareAttackers` construction —
                // they're just not enumerated by `legal_decisions`.
                actions.push(declare_attackers_dict(py, &[])?);
                actions.push(declare_attackers_dict(py, &eligible)?);
            }
        }

        // DeclareBlockers emission (Rust-only — Python's
        // `legal_actions` does NOT emit this variant either; see the
        // module docstring's "DeclareBlockers extension" section for
        // the divergence rationale + the `search/combat_decisions.py`
        // factored-search alternative Python uses).
        //
        // Gate: `phase == PHASE_DECLARE_BLOCKERS` AND
        // `priority_player != active_player` (defender declares, NOT
        // the attacker — CR 509.1) AND `stack.is_empty()`. Priority
        // is already confirmed above. The
        // `priority_player != active_player` check is the inversion
        // of the DeclareAttackers gate — only the defender (off-turn
        // player) declares blockers, so we want to emit for the
        // OFF-active priority holder.
        //
        // Eligibility per `src/mtg_bot/combat.py:eligible_blockers`
        // (lines 26-32): on the defender's battlefield, is a
        // Creature, not tapped. Summoning sickness does NOT exclude
        // (CR 302.1 only restricts attacking + tap-cost activated
        // abilities, not blocking). Flying / reach / protection /
        // skulk / can_only_block-with-flying filters are deferred to
        // the `declare_blocker` primitive's per-pair gate at apply
        // time.
        if self.phase == PHASE_DECLARE_BLOCKERS
            && self.priority_player != self.active_player
            && self.stack.is_empty()
        {
            // Collect eligible blockers in battlefield iteration
            // order (stable across calls). Per CR 302.1, summoning-
            // sick creatures CAN block, so we do NOT filter by
            // `summoning_sick` — the sickness-vs-attacking asymmetry
            // is the headline rule difference vs the DeclareAttackers
            // loop above.
            let mut eligible_blockers: Vec<u32> = Vec::new();
            for obj in self.players[idx].battlefield.iter() {
                if obj.tapped {
                    continue;
                }
                let Some(printing) = self.registry.lookup(obj.printing_id) else {
                    continue;
                };
                if !printing.card_types.iter().any(|t| t == "Creature") {
                    continue;
                }
                eligible_blockers.push(obj.id);
            }

            // Attackers come from `self.attackers` — the set the
            // active player committed to during the DECLARE_ATTACKERS
            // phase (see `state_ops_combat.rs:declare_attacker`,
            // which pushes onto this Vec). We mirror its iteration
            // order so emission ordering is deterministic.
            let attackers: &[u32] = &self.attackers;
            let n_blockers = eligible_blockers.len();
            let m_attackers = attackers.len();

            if n_blockers <= DECLARE_BLOCKERS_PRODUCT_CAP
                && m_attackers <= DECLARE_BLOCKERS_MAX_ATTACKERS
            {
                // Full product space: each blocker independently
                // picks one of (M+1) choices — block attacker_j for
                // j in 0..M, or "don't block" (the M-th choice, by
                // convention). With N blockers and M attackers,
                // total assignments = (M+1)^N. Iterate combo from
                // 0..(M+1)^N; for each, decompose into per-blocker
                // choices via repeated `combo % (M+1)` / `combo /=
                // (M+1)`. Choice index `< M` -> block attackers[j];
                // choice index `== M` -> blocker omitted from the
                // assignment list (the "don't block" sentinel).
                //
                // Edge cases this handles naturally:
                // - N=0 (no blockers): (M+1)^0 = 1 iteration emits a
                //   single empty assignment list. Matches Python's
                //   "must emit something" + "empty assignments means
                //   no blocks" semantics.
                // - M=0 (no attackers in `state.attackers`):
                //   (0+1)^N = 1 iteration; each blocker's choice
                //   index is 0, which is `== M` (the "don't block"
                //   sentinel), so the assignment list is empty.
                //   Matches "no attackers means no blocks possible."
                let m_plus_1 = (m_attackers + 1) as u32;
                let total = (m_plus_1 as u64).pow(n_blockers as u32);
                for combo_raw in 0..total {
                    let mut combo = combo_raw;
                    let mut assignments: Vec<(u32, u32)> =
                        Vec::with_capacity(n_blockers);
                    for &blocker_id in eligible_blockers.iter() {
                        let choice = (combo % (m_plus_1 as u64)) as usize;
                        combo /= m_plus_1 as u64;
                        if choice < m_attackers {
                            assignments.push((blocker_id, attackers[choice]));
                        }
                        // else: blocker doesn't block — omitted.
                    }
                    actions.push(declare_blockers_dict(py, &assignments)?);
                }
            } else {
                // Past the cap: emit only the canonical "no blocks"
                // assignment (an empty Vec). The dropped middle
                // assignments are still reachable at apply time via
                // direct `Action::DeclareBlockers` construction —
                // they're just not enumerated by `legal_decisions`.
                // Bounded at 1 dict regardless of N or M.
                actions.push(declare_blockers_dict(py, &[])?);
            }
        }

        // Python's `_is_sorcery_timing` (lines 25-32):
        //   active == priority AND step in MAIN_STEPS AND stack
        //   empty AND _has_priority. Priority is already confirmed
        //   above; the remaining conjuncts gate the sorcery-speed
        //   block (PlayLand here, plus CastSpell / DeclareAttackers /
        //   ActivateAbility in future slices).
        //
        // First-slice short-circuit: there are no instant-speed
        // actions to emit yet, so if it isn't sorcery timing we
        // return early at this point — Python's instant-speed block
        // (lines 246-301) is a no-op in our world. When CastSpell
        // lands, this early-return shrinks to a `continue past
        // sorcery-only block`.
        let is_sorcery_timing = self.active_player == player_id
            && (self.phase == PHASE_MAIN1 || self.phase == PHASE_MAIN2)
            && self.stack.is_empty();
        if !is_sorcery_timing {
            return Ok(actions);
        }

        // Python lines 319-329: PlayLand emission.
        //
        // Python gates on `p.lands_played_this_turn < max_lands` —
        // Rust has no such field today (see module docstring's
        // "Deliberate divergence" section); skip the gate, emit for
        // every Land in hand.
        //
        // Hand iteration order: zone-insertion order (matches
        // Python's `for card in p.hand`). Dedup key:
        // `(printing.name,)` — two copies of the same printing
        // collapse to one action.
        let mut seen_playland: HashSet<&str> = HashSet::new();
        for obj in self.players[idx].hand.iter() {
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            // Land detection: card_types contains "Land". Mirrors
            // Python's `card.printing.is_land` property, which is a
            // thin wrapper over the same containment check.
            let is_land = printing
                .card_types
                .iter()
                .any(|t| t == "Land");
            if !is_land {
                continue;
            }
            // Dedup by printing name; emit the first hand-order copy.
            let name: &str = printing.name.as_ref();
            if seen_playland.contains(name) {
                continue;
            }
            seen_playland.insert(name);
            actions.push(play_land_dict(py, obj.id)?);
        }

        // PlayLand goes BEFORE PassPriority in Python's natural
        // append order — but Python prepends PassPriority at line
        // 178, so Python's final list shape is
        // `[PassPriority, PlayLand_1, PlayLand_2, ...]`. We match
        // that by pushing PassPriority first then PlayLands after.

        // Python lines 335-445: sorcery-speed CastSpell emission.
        //
        // Walks hand for non-Land cards and emits a base CastSpell
        // (no alt-cost flags). The Python source has additional
        // alt-cost emissions (kicker / warp / impending /
        // cast_for_free / Room-half) interleaved inside the same
        // loop body; first-slice ships the no-flags emission only —
        // see the "Deferred alt-cost flags" section of the module
        // docstring.
        //
        // **Filter** (Python lines 366-345, narrowed): non-Land
        // cards. Python additionally excludes Instants (they get
        // cast in the instant-speed block above, lines 246-301)
        // and gates `_is_sorcery_speed_spell` on
        // is_creature || is_sorcery || (is_permanent && !is_instant
        // && !is_land). First-slice approximation: "not Land"
        // captures every printing in the synthetic-state registry
        // (Forest = Land filtered out; Creatures emit; no other
        // types registered yet).
        //
        // **Dedup** (Python lines 349-363): keyed on
        // `(printing.name, ...flags)` 6-tuple. With all flags
        // defaulted, this collapses to `(printing.name,)` — same
        // shape as the PlayLand dedup above. Two Llanowar Elves
        // in hand emit a single CastSpell carrying the first
        // hand-order Elves' id.
        let mut seen_castspell: HashSet<&str> = HashSet::new();
        for obj in self.players[idx].hand.iter() {
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            // Skip Lands — they emit PlayLand actions, not
            // CastSpell. Lands play via the special action, not
            // cast via the stack (CR 305.1).
            let is_land = printing.card_types.iter().any(|t| t == "Land");
            if is_land {
                continue;
            }
            // Dedup by printing name (first-slice key — see
            // module docstring for the future 6-tuple shape).
            let name: &str = printing.name.as_ref();
            if seen_castspell.contains(name) {
                continue;
            }
            seen_castspell.insert(name);
            actions.push(cast_spell_dict(py, obj.id)?);
        }

        // Python lines 612-829: ActivateAbility emission for activated
        // abilities on permanents the priority player controls. We
        // are INSIDE the sorcery-timing block at this point (Python's
        // line 303-304 early return guarantees the same gating); the
        // per-ability `ability.sorcery_speed` filter at line 620
        // would be a no-op in this position.
        //
        // **First-slice approximation**: Rust's `CardPrinting` has no
        // `activated_abilities` field yet (see `cards.rs:361-396` and
        // the module docstring's "ActivateAbility extension"
        // section). We approximate by emitting one ActivateAbility
        // (ability_index=0) per untapped, non-summoning-sick
        // mana-producer permanent — i.e. every untapped object on
        // the priority player's battlefield whose printing has
        // `mana_produced.is_some()`. This covers Forest's
        // tap-for-{G} ability and every basic-land equivalent, plus
        // creature mana sources (Llanowar Elves, Birds of Paradise)
        // once they've shed summoning sickness. A just-cast
        // creature mana source is excluded by the `summoning_sick`
        // gate below — mirrors Python's `legal.py:661-663`
        // (`is_creature_now(obj) && obj.summoning_sick` short-
        // circuit inside the tap-self cost-payability block).
        //
        // **The gate is `summoning_sick && requires_tap_cost`.** All
        // mana abilities have an implicit `{T}` cost (CR 605.1a), so
        // for the first-slice mana-only filter every emitted ability
        // is a tap-cost ability and `summoning_sick` alone is
        // sufficient to reject. A future slice that lands non-mana
        // activated abilities (loyalty abilities, sacrifice abilities
        // without tap, etc.) will narrow this to "skip the source
        // ONLY for abilities whose `cost.tap_self` is true" so a
        // summoning-sick creature with a non-tap-cost ability still
        // emits. Pins G6 closure; see `docs/RUST_PARITY_GAPS.md`.
        //
        // **No dedup by printing name.** Unlike PlayLand / CastSpell,
        // each ActivateAbility carries a per-object source_id and
        // refers to a battlefield slot whose state (tapped /
        // counters / attached_to) is unique to that object. Two
        // Forests on the battlefield are NOT interchangeable —
        // activating Forest A taps Forest A, not Forest B. Python's
        // ActivateAbility loop at lines 612-829 has no dedup HashSet
        // for the same reason.
        for obj in self.players[idx].battlefield.iter() {
            if obj.tapped {
                continue;
            }
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            if printing.mana_produced.is_none() {
                continue;
            }
            // CR 302.1 — summoning sickness applies ONLY to creatures
            // tapping for tap-cost abilities. Lands (Forests,
            // Mountains, etc.) can tap for mana the turn they enter
            // the battlefield, even though every new permanent
            // carries `summoning_sick = true` (the CE1 universal
            // default). The gate is
            // `is_creature && summoning_sick && requires_tap_cost`;
            // for this mana-only slice every emitted ability has an
            // implicit `{T}` cost (CR 605.1a), so
            // `is_creature && summoning_sick` is sufficient.
            // Mirrors Python's `legal.py:661-663` short-circuit
            // (`is_creature_now(obj) && obj.summoning_sick` inside
            // the `tap_self` cost-payability block). A future slice
            // that lands non-mana activated abilities will narrow
            // this to "skip the source ONLY for abilities whose
            // `cost.tap_self` is true". See I17 in
            // `docs/RUST_PARITY_GAPS.md`.
            let is_creature = printing.card_types.iter().any(|t| t == "Creature");
            if is_creature && obj.summoning_sick {
                continue;
            }
            actions.push(activate_ability_dict(py, obj.id, 0)?);
        }

        // Final emission order: [PassPriority, (DeclareAttackers...),
        // PlayLand..., CastSpell..., ActivateAbility...]. Matches
        // Python's `legal.py` natural append order under the
        // sorcery-timing gate (DeclareAttackers is the Rust-only
        // exception, see its module-docstring section).

        Ok(actions)
    }
}
