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
//! that comes AFTER PlayLand) AND the instant-speed CastSpell block at
//! `src/mtg_bot/legal.py:217-301` (the `instant_speed = card.printing
//! .is_instant or "flash" in card.printing.keywords` loop that comes
//! BEFORE the sorcery-timing gate at line 303). **Post-CE19 closure**
//! the Rust enumerator places the entire CastSpell loop ABOVE the
//! sorcery-timing short-circuit and gates per-card on
//! `is_instant || keywords.contains("flash") || is_sorcery_timing`,
//! so a flash creature or Instant in the priority holder's hand emits
//! at non-active priority windows the same way Python does. The
//! shared `seen_castspell` dedup HashSet collapses any double-eligible
//! card (theoretically flash-on-instant — degenerate but harmless) to
//! a single emission. See I46 in `docs/RUST_PARITY_GAPS.md`.
//!
//! **Mana-affordability check — CLOSED (legal_decisions
//! mana-awareness Slice 1).** Python's `_can_cast` (lines 252-254 /
//! 383-385) calls `can_pay_cost(...)` to refuse spells whose mana
//! cost the player can't pay. Rust now has the non-destructive
//! `state_ops_mana::can_pay_cost` predicate (composing
//! `pool_can_cover` + `auto_tap_headroom_pool`, co-located with the
//! destructive `pay_mana_cost` twin); every live CastSpell emission
//! site below gates on it (the §3.5.2 `seen_castspell`
//! add-on-emit ordering preserved). Parity-NEUTRAL by the
//! single-script-replay §4 proof.
//!
//! **Target-satisfiability check — CLOSED (legal_decisions
//! mana-awareness Slice 2).** Python additionally gates each
//! CastSpell emission on `_targets_are_satisfiable`
//! (`legal.py:79-114`, CR 601.2c — a spell whose targeted
//! `spell_effect` has 0 legal targets is never enumerated). Rust now
//! has the non-destructive `spell_targets_satisfiable` predicate
//! (per live bridge-admitted targeted `Effect` variant, scanning for
//! ≥1 legal target with the SAME `bf_iter_etb_sorted` filter the
//! matching resolve arm uses — single source of truth); every live
//! CastSpell emission site gates on it (folded into the same
//! `*_payable` boolean so the `seen_castspell` ordering is
//! preserved). Parity-NEUTRAL by the same §4 subset-preserving
//! argument.
//!
//! **Additional-cost check — CLOSED (legal_decisions
//! mana-awareness Slice 3).** Python additionally gates each
//! CastSpell emission on `_additional_costs_payable`
//! (`legal.py:117-165`, CR 601.2f/601.2h — a spell whose cast-time
//! additional non-mana cost is unpayable is never enumerated). The
//! 3 `additional_cost_*` flags are now on the Rust `CardPrinting`
//! (parsed from the registry dict, back-compat default false), and
//! the non-destructive `additional_costs_payable` predicate gates
//! every live CastSpell emission site (folded into the same
//! non-mana precondition var so the `seen_castspell` ordering is
//! preserved). Live incidence is ZERO (all 4 EoE additional-cost
//! cards are bridge-DECLINED — design §5 prediction) ⇒
//! parity-NEUTRAL by construction; the full predicate is the
//! faithful Python mirror + the single named extension point.
//! All three axes' deliberate identity sub-simplifications (the
//! `eff()` cost-reduction seam; the conservative
//! trivially-satisfiable branch for non-admitted targeted shapes;
//! the conservative-true additional-cost branch for the
//! zero-live-incidence flags) are recorded in
//! `docs/SEARCH_LIMITATIONS.md`. **Residual divergence (still
//! deferred):** ONLY the non-live alt-cost axes (kicker / impending
//! / granted-warp / Room-half / X — no Rust emission site at all;
//! below). Every Python prune axis with a live Rust CastSpell
//! emission site (mana / target / additional-cost) is CLOSED — the
//! C1 SoT-flip keystone at the legal-decision-set level.
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
//! ## DeclareAttackers — factored per-attacker emission (G1: the
//! C1-analog for the combat axis; rust(legal-decisions-factored-combat))
//!
//! **G1 CLOSED — byte-compatible with the Python search seam.**
//! Python's `legal_actions` does NOT emit a whole `DeclareAttackers`
//! action — the engine docstring (`src/mtg_bot/engine.py:10-19`) is
//! explicit that the combat-declaration actions "aren't enumerable as
//! legal moves because their decision space is combinatorial." Python
//! uses a FACTORED search-side decomposition in
//! `src/mtg_bot/search/combat_decisions.py` — `AttackerDecision(
//! attacker_id, attack)`, ONE per eligible attacker with exactly two
//! branches (`attack=True` then `attack=False`) — surfaced by
//! `MtgGameStateAdapter.legal_decisions` (`engine_adapter.py:808-814`)
//! for the HEAD pending attacker, accumulated by the adapter into a
//! single engine-level `DeclareAttackers(frozenset[int])` at finalize
//! (`engine_adapter.py:1080-1108`).
//!
//! Pre-G1 the Rust port emitted a Rust-only capped `2^n` POWERSET of
//! whole `DeclareAttackers` actions — neither a superset nor
//! byte-equal to Python's factored set (it lost the per-creature MCTS
//! progressive-widening shape the codec ATTACKER slots are built for,
//! and silently dropped the middle of the combinatorial space above
//! the cap). That was the documented G1 flip-blocker
//! (`NN_RUST_DECISION_SOURCING_WIRING_DESIGN.md` §2.4 G1). G1 replaces
//! the powerset block with the byte-compatible factored emission: for
//! the HEAD pending attacker (smallest-id eligible — matching the
//! adapter's `_pending_attackers = sorted(o.id for o in eligible)`
//! `engine_adapter.py:1315` AND the codec's `_pending_attacker_head`
//! `min(...)` fallback `action_encoding.py:990-995`) emit exactly two
//! `{"kind":"attacker_decision","attacker_id":head,"attack":<bool>}`
//! dicts (`attack=true` then `attack=false`), mapping 1:1 to the
//! codec's implicit-attacker `SLOT_ATTACKER_ATTACK` /
//! `SLOT_ATTACKER_SKIP` slots (`action_encoding.py:708-712`). The B3
//! Python provider (sibling slice — NOT this one) owns the
//! per-creature accumulation + finalize-to-whole-`declare_attackers`
//! (the existing whole-action `apply_action` arm — UNCHANGED). It is
//! parity-NEUTRAL under the Python-sourced single-script-replay
//! harness (the C1 §4 proof — the harness never sources a choice from
//! `rust.legal_decisions`; it replays a Python-derived
//! `declare_attackers` whole-action dict, which the unchanged
//! whole-action arm still applies byte-identically).
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
//! **Enumeration-time defender filter (closed by CE17 / I40).** The
//! eligibility loop short-circuits any battlefield permanent whose
//! effective keyword set contains `"defender"` via the shared
//! `crate::state_ops_combat::battlefield_has_keyword(self, obj.id,
//! "defender")` helper, matching the apply-time gate at
//! `state_ops_combat.rs:639-641` and Python's
//! `combat.py:eligible_attackers` (line 22) `"defender" not in
//! effective_keywords(state, o)`. The helper reads printed keywords
//! AND UEOT-granted keywords (CR 613 layer 6), so a creature that
//! gained defender from a UEOT modifier this turn is filtered too.
//! Pre-CE17 the filter was deferred to the per-id apply-time gate;
//! that left `legal_decisions` emitting illegal subsets (search-side
//! waste, see CE17 entry in `docs/RUST_PARITY_GAPS.md`). Post-CE17
//! enumeration matches Python's eligible set exactly on the defender
//! axis.
//!
//! **Deferred** (silently included as candidates; the
//! `declare_attacker` primitive's bool-return filters them at apply
//! time):
//! - Continuous "can't attack" restrictions (Propaganda, Ghostly
//!   Prison, etc.) — no continuous-effects layer in the Rust engine
//!   yet.
//! - Defending player's "must block" / intimidate / fear and other
//!   defender-side keywords that change the *blocker* legality
//!   (these don't change attacker eligibility but interact in the
//!   declare-blockers step which is out of this slice's scope).
//!
//! **Emission shape (post-G1, the byte-compatible factored shape)**:
//! exactly two Python dicts for the head pending attacker, matching
//! `actions.rs:action_to_pydict` for `Action::AttackerDecision`:
//! `{"kind":"attacker_decision","attacker_id":<head>,"attack":true}`
//! then `{"kind":"attacker_decision","attacker_id":<head>,
//! "attack":false}`. The `attack=true`-first order mirrors
//! `engine_adapter.py:811-814`. Zero eligible attackers ⇒ NO combat
//! dict emitted (the adapter skips the factored phase entirely,
//! `engine_adapter.py:1295-1310`; the priority window opens instead).
//!
//! **No cap, no dropped middle.** The factored shape has a fixed
//! fan-out of 2 per `legal_decisions` call regardless of board width
//! — the `2^n` powerset explosion + its `DECLARE_ATTACKERS_POWERSET
//! _CAP` lossy bound are GONE (the cap dropped the strategically-
//! valuable middle subsets on wide boards; the factored walk reaches
//! every attack pattern through MCTS depth, not a capped breadth).
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
//! **Two-path emission (CE20 Slice 1):** Rust's `CardPrinting`
//! now carries BOTH a legacy `mana_produced: Option<Vec<String>>`
//! field (free-mana-ability hatch) AND an `activated_abilities:
//! Vec<ActivatedAbility>` Vec (cost-paying abilities). The
//! emission loop walks both:
//!
//! - **Legacy path** — every untapped permanent whose printing
//!   has `mana_produced.is_some()` emits a single
//!   ActivateAbility at `ability_index = 0`. Covers Forest, etc.
//!   The legacy path will eventually be bridged into a
//!   synthesized `ActivatedAbility { is_mana_ability: true,
//!   cost: { tap_self: true }, effect: AddManaToPool(...) }`
//!   entry by a follow-up slice.
//! - **New path** — every `printing.activated_abilities[i]` is
//!   considered per-ability; emits when the cost is affordable
//!   from the controller's pool AND (for tap_self abilities) the
//!   source is untapped + non-sick (creature-only, CR 302.1).
//!
//! **Affordability gate is pool-only** (narrower than Python's
//! `can_pay_cost` which accounts for auto-tap-from-lands).
//! `pool_can_cover(pool, cost.mana)` mirrors
//! `state_ops_cast.rs::pay_mana_cost`'s pre-mutation check:
//! colored pips matchable + post-colored generic coverage. A
//! follow-up slice grows an auto-tap composer to converge with
//! Python's headroom accounting.
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
//!   creature whose `summoning_sick` flag is true does NOT emit a
//!   tap-cost ActivateAbility. Mirrors Python's
//!   `legal.py:661-663` short-circuit. The gate is
//!   `is_creature && summoning_sick && cost.tap_self`. The haste-
//!   keyword exception (CR 702.10) lands with a future static-
//!   abilities slice.
//!
//! **Dispatch is wired** post-CE20-Slice-1. See
//! `state_ops_actions.rs::Action::ActivateAbility` arm — it pays
//! the cost via `state_ops_cast::pay_mana_cost_pub`, taps the source
//! for `tap_self`, and either dispatches `apply_effect` directly
//! (`is_mana_ability=true`) or pushes a `StackItem::new_trigger`-
//! shaped slot for the next `resolve_top_of_stack` to fire
//! (`is_mana_ability=false`, mirrors CR 605/113.3b).
//!
//! **Dict shape**: `{"kind": "activate_ability", "source_id": <id>,
//! "ability_index": 0}`. Round-trips through `parse_action` (round-23
//! `Action::ActivateAbility` variant) and `action_to_pydict`
//! unchanged.
//!
//! ## DeclareBlockers — factored per-blocker emission (G1: the
//! C1-analog for the combat axis; rust(legal-decisions-factored-combat))
//!
//! **G1 CLOSED — byte-compatible with the Python search seam** (same
//! shape story as the DeclareAttackers factored section above).
//! Python's `legal_actions` does NOT emit a whole `DeclareBlockers`
//! action (`src/mtg_bot/engine.py:10-19` — the combat-declaration
//! actions "aren't enumerable as legal moves because their decision
//! space is combinatorial"). Python uses FACTORED `BlockerDecision(
//! blocker_id, attacker_id)`s in
//! `src/mtg_bot/search/combat_decisions.py` — ONE per eligible
//! blocker, with one branch per currently-attacking creature plus a
//! final `attacker_id=None` ("don't block") branch — surfaced for
//! the HEAD pending blocker by `MtgGameStateAdapter.legal_decisions`
//! (`engine_adapter.py:815-828`), accumulated by the adapter into a
//! single engine-level `DeclareBlockers(assignments=tuple[tuple[int,
//! int], ...])` at finalize (`engine_adapter.py:1114-1144`).
//!
//! Pre-G1 the Rust port emitted a Rust-only capped `(M+1)^N` PRODUCT
//! of whole `DeclareBlockers` assignments — neither a superset nor
//! byte-equal to Python's factored set. G1 replaces it with the
//! byte-compatible factored emission: for the HEAD pending blocker
//! (smallest-id eligible — matching the adapter's `_pending_blockers
//! = sorted(o.id for o in eligible)` `engine_adapter.py:1360` AND the
//! codec's `_pending_blocker_head` `min(...)` fallback
//! `action_encoding.py:1007-1012`) emit one
//! `{"kind":"blocker_decision","blocker_id":head,"attacker_id":<atk>}`
//! per currently-attacking creature (in `self.attackers` declaration
//! order, mirroring the adapter's battlefield-attacking walk),
//! followed by a final `{"kind":"blocker_decision","blocker_id":
//! head,"attacker_id":None}` ("don't block"). The codec recovers the
//! policy slot `SLOT_BLOCKER_BASE + atk_pos` from the `attacker_id`
//! VALUE via `_attacking_creatures` (sorted-by-id,
//! `action_encoding.py:414-420,714-733`) / `SLOT_BLOCKER_SKIP` for
//! the None branch — so the slot is independent of branch list order
//! (list order is parity-NEUTRAL under the Python-sourced
//! single-script-replay harness; the C1 §4 proof). The B3 Python
//! provider (sibling slice — NOT this one) owns the per-blocker
//! accumulation + finalize-to-whole-`declare_blockers` (the
//! UNCHANGED whole-action `apply_action` arm, which keeps the CE16
//! menace post-validation `.retain` at the finalize boundary).
//!
//! **Gating**: emit factored blocker decisions ONLY when:
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
//! **Emission shape (post-G1, the byte-compatible factored shape)**:
//! `m+1` Python dicts for the head pending blocker, matching
//! `actions.rs:action_to_pydict` for `Action::BlockerDecision`:
//! `{"kind":"blocker_decision","blocker_id":<head>,"attacker_id":
//! <atk>}` per attacking creature, then a final `…,"attacker_id":
//! None` (the "don't block" branch — encoded as an explicit Python
//! `None`, mirroring `BlockerDecision(attacker_id=None)`, NOT by
//! omission). Zero eligible blockers ⇒ NO combat dict emitted (the
//! adapter skips the factored blocker phase entirely,
//! `engine_adapter.py:1339-1356`; the priority window opens instead).
//!
//! **No cap, no dropped middle.** The factored shape has a fixed
//! fan-out of `m+1` per `legal_decisions` call (one branch per live
//! attacker + don't-block) regardless of blocker-board width — the
//! `(M+1)^N` product explosion + its `DECLARE_BLOCKERS_PRODUCT_CAP` /
//! `DECLARE_BLOCKERS_MAX_ATTACKERS` lossy bounds are GONE (the caps
//! collapsed every non-trivial blocker board to "no blocks only";
//! the factored walk reaches every block pattern through MCTS depth,
//! not a capped breadth). The two `docs/SEARCH_LIMITATIONS.md`
//! "Rust-only enumeration" entries (DeclareAttackers powerset +
//! DeclareBlockers product) are replaced at landing by the single
//! closed-G1 note.

use crate::state::GameState;
use crate::state_ops_steps::{
    PHASE_DECLARE_ATTACKERS, PHASE_DECLARE_BLOCKERS, PHASE_MAIN1, PHASE_MAIN2,
};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

/// Effective-power reader for the `Effect::ExorciseEffect`
/// `spell_targets_satisfiable` arm (the "creature with power 4 or
/// greater" leg of Exorcise's 3-way OR filter).
///
/// Mirrors the resolve-side `read_doubling_power` helper at
/// `state_ops_resolve.rs:1798-1825` byte-equal — same composition:
/// `base (UEOT-SetBasePT override / manland / printed) +
///  ueot_power_bonus() + obj.counters` (CR 613 layer 7b/7c/7d).
/// Inlined here per the per-slice file-isolation rule (the combat
/// slice's `battlefield_creature_power` is module-private to that
/// file; the resolve slice's `read_doubling_power` is module-private
/// to that file; this is the third copy of the same recipe for the
/// legal-target slice).
///
/// Matches Python's `effective_power(state, obj)` =
/// `_compute_effective_pt` at `game.py:2138-2251` — the same SoT the
/// Python `ExorciseEffect._matches` helper at `effects/bespoke.py:
/// 2304` reads. Returns 0 for non-creatures or unknown ids.
///
/// **Documented residual.** Layer-7c static anthems and layer-7e P/T
/// switches are NOT folded here (nor by the combat-side reader, nor
/// by the resolve-side `read_doubling_power`). Same established
/// combat-side asymmetry documented in the round-53 SEARCH_LIMITATIONS
/// entry.
fn legal_effective_power(state: &GameState, obj_id: u32) -> i32 {
    let mut found: Option<&crate::state::GameObject> = None;
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
        .or_else(|| obj.ueot_becomes_land_creature_pt().map(|(p, _)| p))
        .unwrap_or(printed_base);
    base + obj.ueot_power_bonus() + obj.counters
}

// **G1 (the C1-analog for the combat axis) — the powerset / product
// caps are GONE.** The former `DECLARE_ATTACKERS_POWERSET_CAP` (4),
// `DECLARE_BLOCKERS_PRODUCT_CAP` (3), and
// `DECLARE_BLOCKERS_MAX_ATTACKERS` (3) constants gated a Rust-only
// `2^n` powerset / `(M+1)^N` product enumeration of WHOLE
// `DeclareAttackers`/`DeclareBlockers` actions — a deliberate
// divergence from Python's factored search seam that was neither a
// superset nor byte-equal to the per-creature
// `AttackerDecision`/`BlockerDecision` set
// (`src/mtg_bot/search/combat_decisions.py` +
// `engine_adapter.py:808-828`). G1 replaces that emission with the
// byte-compatible factored per-creature shape (one
// `attacker_decision` / `blocker_decision` dict per HEAD pending
// creature), so the caps — and the combinatorial-explosion failure
// mode they bounded — no longer exist. The factored shape has a
// fixed tiny fan-out (2 for attackers; `M+1` for blockers) with no
// dropped-middle information loss. The two `SEARCH_LIMITATIONS.md`
// "Rust-only enumeration" entries are replaced by the closed-G1
// note at landing (see the module docstring's factored-emission
// section).

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

/// Build a `{"kind":"cast_spell", ..., "use_warp":true, ...}` dict.
/// **Phase-1 warp substrate (round-63):** the warp-cast variant.
/// Same 7 keys as `cast_spell_dict` but `use_warp=true`. Round-trips
/// through `parse_action` (`actions.rs` reads `use_warp`) and
/// `action_to_pydict` (writes it) byte-equal — the pre-Phase-1 gap
/// was *semantic* (Rust ignored the parsed flag), never a
/// serialization gap, so no bridge change is needed for the
/// round-trip.
fn cast_spell_warp_dict(py: Python<'_>, card_id: u32) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "cast_spell")?;
    d.set_item("card_id", card_id)?;
    d.set_item("use_warp", true)?;
    d.set_item("kicked", false)?;
    d.set_item("cast_for_free", false)?;
    d.set_item("use_impending", false)?;
    d.set_item("cast_room_half", py.None())?;
    Ok(d)
}

/// Build a `{"kind": "activate_ability", "source_id": <id>,
/// "ability_index": <u32>, "target_creature_id": <Option<u32>>}` dict.
/// Mirrors `actions.rs:action_to_pydict`'s `Action::ActivateAbility`
/// arm so a dict produced here round-trips through `parse_action` and
/// back without key drift. Post-CE20-Slice-1: callers can pass any
/// `ability_index` in the printing's `activated_abilities` range
/// (legacy `mana_produced` path always uses `ability_index = 0`).
///
/// CE20 follow-up Slice 4 (Equipment-equip): the dict shape grows an
/// optional `target_creature_id` key. `None` for every non-target
/// activation (the pre-Slice-4 shape — bridge-synthesized mana
/// abilities, Bear-Caster-style draw abilities, sacrifice-self
/// abilities). `Some(id)` for activations whose cost flags
/// `requires_target_creature_you_control = true` (Equipment-equip).
fn activate_ability_dict(
    py: Python<'_>,
    source_id: u32,
    ability_index: u32,
    target_creature_id: Option<u32>,
    sacrifice_target_id: Option<u32>,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "activate_ability")?;
    d.set_item("source_id", source_id)?;
    d.set_item("ability_index", ability_index)?;
    d.set_item("target_creature_id", target_creature_id)?;
    d.set_item("sacrifice_target_id", sacrifice_target_id)?;
    Ok(d)
}

/// **CE20 Slice 1.** Pure-pool affordability check for an activated
/// ability's mana cost. Mirrors `state_ops_cast.rs::pay_mana_cost`'s
/// pre-mutation gate (colored pips must be matchable by same-color
/// pool mana, then `pool_total - colored_total >= generic`) without
/// the commit step. Returns `true` iff `pool` can cover `cost`
/// outright. Composes with `auto_tap_headroom_pool` for the broader
/// emission-side affordability check.
pub(crate) fn pool_can_cover(
    pool: &crate::state::ManaPool,
    cost: &crate::cards::ManaCost,
) -> bool {
    if pool.white < cost.white
        || pool.blue < cost.blue
        || pool.black < cost.black
        || pool.red < cost.red
        || pool.green < cost.green
    {
        return false;
    }
    let rem_total = (pool.white - cost.white) as u32
        + (pool.blue - cost.blue) as u32
        + (pool.black - cost.black) as u32
        + (pool.red - cost.red) as u32
        + (pool.green - cost.green) as u32
        + (pool.colorless) as u32;
    rem_total >= cost.generic as u32
}

/// **CE20 follow-up — auto-tap-from-lands headroom composer.**
/// Returns a hypothetical `ManaPool` the `player_idx` controller
/// COULD achieve by tapping every available mana source on their
/// battlefield (lands + mana creatures), on top of their current
/// floating pool.
///
/// Walks the controller's battlefield for sources that meet ALL of:
///   - `printing.mana_produced.is_some()` (a free `{T}: Add ...`
///     mana ability — Forests, Llanowar Elves, Birds of Paradise),
///   - the object is NOT tapped (a tapped land has no headroom),
///   - if the object is a creature, it is NOT summoning-sick (CR
///     302.1 — newly-entered creatures can't pay tap costs).
///
/// For each qualifying source, credits each color symbol in
/// `mana_produced` saturating-added to the matching pool slot.
///
/// Used as the emission-side affordability composer for
/// `ActivateAbility` enumeration: a player with 0 floating mana but
/// 3 untapped Forests on bf should see `{2}{G}` activated abilities
/// emit (Python's `can_pay_cost` accounts for this auto-tap headroom
/// at `actions.py:336-403`). The APPLY-TIME cost-pay path remains
/// strict: a player still must have the mana physically in their
/// pool to activate (via separate `ActivateAbility(forest, 0)`
/// actions that tap the lands first). Matches Python's "permissive
/// in `legal_actions`, strict in `_activate_ability`" model.
///
/// Pins I50; see `docs/RUST_PARITY_GAPS.md`.
pub(crate) fn auto_tap_headroom_pool(
    state: &crate::state::GameState,
    player_idx: usize,
) -> crate::state::ManaPool {
    // G7 S6-Class-E ordered-pool: `ManaPool` is no longer `Copy`
    // (it carries `Vec<ManaCoin>` — Python parity, list[ManaCoin]).
    // Clone to build the hypothetical pool the controller COULD
    // achieve by tapping their headroom sources. The headroom adds
    // below mutate ONLY the per-color count cache on the local
    // clone — this fn is a read-only AFFORDABILITY composer; the
    // local pool is never written back to state. The legal-side
    // affordability check (`pool_can_cover`) reads only the per-color
    // counts, so leaving the local clone's `coins` vec untouched
    // here is intentional (it's never read by the consumer of this
    // return value). Real `add_mana` mutations that DO write back
    // (e.g. the resolver `AddMana` arms) push to `coins` to keep the
    // sync invariant.
    let mut pool = state.players[player_idx].mana_pool.clone();
    for obj in state.players[player_idx].battlefield.iter() {
        let Some(printing) = state.registry.lookup(obj.printing_id) else {
            continue;
        };
        let Some(symbols) = printing.mana_produced.as_ref() else {
            continue;
        };
        if obj.tapped {
            continue;
        }
        let is_creature = printing.card_types.iter().any(|t| t == "Creature");
        if is_creature && obj.summoning_sick {
            continue;
        }
        for sym in symbols {
            match sym.as_str() {
                "W" => pool.white = pool.white.saturating_add(1),
                "U" => pool.blue = pool.blue.saturating_add(1),
                "B" => pool.black = pool.black.saturating_add(1),
                "R" => pool.red = pool.red.saturating_add(1),
                "G" => pool.green = pool.green.saturating_add(1),
                "C" => pool.colorless = pool.colorless.saturating_add(1),
                _ => {} // validate_mana_symbol gates registration; defensive no-op.
            }
        }
    }
    pool
}

/// **legal_decisions mana-awareness Slice 1 — the cost-reduction
/// seam.** Returns the EFFECTIVE `ManaCost` for affordability, the
/// Rust analog of Python's `_effective_cost` / `apply_cost_reductions`
/// (`src/mtg_bot/actions.py:406-438`, called as `_effective_cost` at
/// `legal.py:187-188` BEFORE every `_can_cast`). Python applies
/// spell-local + battlefield-permanent-granted generic-mana
/// reductions before the payability check; the predicate must see the
/// reduced cost, not the printed one.
///
/// **Live-surface scope (verify-first, evidence-bound):** Rust's
/// `cards.rs` has NO `cost_reductions` / `external_cost_reductions`
/// fields on the live bridge-admitted cast surface — the bridge
/// declines cost-reduction cards (`docs/COST_REDUCTION_PHASE1_DESIGN.md`
/// is "realizable-0"). So `eff(c) = c` (identity) for every live
/// CastSpell axis. This is the single NAMED insertion point so a
/// future cost-reduction slice has exactly one place to wire the
/// reduction in (it does NOT have to re-touch the 6 gate sites).
/// Deliberate live-surface simplification, recorded in
/// `SEARCH_LIMITATIONS.md`; a pin asserts the live admitted set
/// carries no `cost_reductions` (the §3.5.4 regression tripwire).
#[inline]
fn eff(cost: &crate::cards::ManaCost) -> &crate::cards::ManaCost {
    cost
}

/// **legal_decisions mana-awareness Slice 2 — target-satisfiability
/// prune (CR 601.2c).** Returns `true` iff casting the spell printed
/// by `printing_id`, with the source controlled by `controller_idx`,
/// has its targeting requirement satisfiable RIGHT NOW — i.e. there
/// exists at least one legal target for its targeted spell effect.
/// Mirrors Python `mtg_bot.legal._targets_are_satisfiable`
/// (`src/mtg_bot/legal.py:79-114`) — the SoT — for the live
/// bridge-admitted CastSpell surface.
///
/// **Python parity contract (verify-first, file:line precise).**
/// Python's gate (`legal.py:92-114`):
///   - `effect = card.printing.spell_effect`; if `effect is None or
///     not effect.requires_target(): return True` (`:92-94`) — a
///     non-targeting spell / a creature with no `spell_effect`
///     trivially satisfies. **Every bridge-simple creature has
///     `spell_effect is None`** so it takes this trivial branch (this
///     is why Slice 2 is parity-NEUTRAL on the dominant admit axis —
///     it never prunes a creature cast).
///   - `legal = effect.legal_targets(state, player_id)`; `if not
///     legal: return False` (`:95-97`) — the load-bearing CR 601.2c
///     prune: zero legal targets ⇒ the spell can't be cast at all.
///   - `min_t, _ = effect.target_count_range(); if min_t <= 1: return
///     True` (`:98-100`). Every live bridge-admitted targeted spell
///     is an OLD-shape single-target effect (`target_count_range()`
///     defaults to `(1,1)` for `requires_target()==True` old-shape
///     effects — `effects/base.py:163`), so `min_t == 1` and the
///     per-slot-kind branch (`:101-114`) is UNREACHABLE on the live
///     surface EXCEPT Diplomatic Relations (the only 2-slot grammar;
///     `min_t == 2`, `target_slot_kinds() == ("my_creature",
///     "creature")`) — handled explicitly below.
///
/// **Single-source-of-truth discipline (the Slice-1 shape).** Each
/// arm's legal-target scan uses the SAME object filter the matching
/// `state_ops_resolve.rs` resolve arm uses (the deterministic-first
/// `bf_iter_etb_sorted` scan), so the emission-side prune and the
/// resolve-side CR 608.2b re-check can never disagree about what a
/// "legal target" is — exactly why Slice 1 reused
/// `auto_tap_headroom_pool` rather than re-deriving land tapping.
/// This makes the prune **at-least-as-permissive as Python** (it
/// returns `false` ONLY when GENUINELY zero legal targets exist), so
/// it never over-prunes a Python-sourced cast — the parity-neutral
/// subset-preserving property the single-script-replay §4 proof
/// needs (`docs/RUST_LEGAL_DECISIONS_MANA_AWARENESS_DESIGN.md` §4).
///
/// **Scope (verify-first, evidence-bound).** Covers EXACTLY the live
/// bridge-admitted targeted-spell `Effect` variants
/// (`DepressurizeEffect`, `FocusFireEffect`,
/// `DestroyTargetArtifactOrTappedCreature`, `SeedshipImpactEffect`,
/// `DealDamageToTargetCreatureWithExcessLanderRider`,
/// `BiosynthicBurstEffect`, `DiplomaticRelationsEffect`,
/// `CutPropulsionEffect`,
/// `DestroyTargetArtifactEnchantmentOrFlyingCreature`,
/// `EmergencyEjectEffect`). Any other
/// `Effect` (untargeted, or a not-yet-admitted targeted shape) takes
/// the conservative `true` (trivially-satisfiable) branch — NEVER
/// pruned. Conservative = NEVER narrower than Python ⇒ parity-neutral
/// by construction; a future targeted-spell admit slice adds its arm
/// here (the single named extension point).
fn spell_targets_satisfiable(
    state: &crate::state::GameState,
    printing_id: u32,
    controller_idx: usize,
) -> bool {
    use crate::cards::Effect;
    let Some(printing) = state.registry.lookup(printing_id) else {
        // Unknown printing — conservative: don't prune (Python's
        // `card.printing.spell_effect` would be a parser concern, not
        // a legality one; never narrower than Python).
        return true;
    };
    // Python iterates `card.printing.spell_effect` (a SINGLE effect).
    // The Rust bridge models the spell body as `printing.effects`
    // (a Vec). For the bridge-admitted surface the targeted leaf is
    // `effects[0]` UNLESS the spell body is `Sequence`-wrapped: the
    // bridge's `sequence` arm (`cross_engine.py:1494`) sends the whole
    // body as ONE `{"kind":"sequence","effects":[...]}` dict, which
    // `cards.rs` `parse_effect` materializes as a SINGLE nested
    // `Effect::Sequence { effects: vec![...] }` element — NOT a
    // flattened Vec. Radiant Strike's
    // `Sequence(DestroyTargetArtifactOrTappedCreature, GainLife(3))`
    // is therefore `printing.effects == [Effect::Sequence{..}]`; the
    // targeted leaf `DestroyTargetArtifactOrTappedCreature` is a CHILD
    // of that Sequence, not a top-level element. Python's SoT looks
    // THROUGH the Sequence (its `spell_effect` IS the Sequence, whose
    // `requires_target()`/`legal_targets()` delegate to the targeted
    // child — `effects/combinators.py` Sequence protocol). So the gate
    // MUST recurse into `Effect::Sequence` to find the targeted leaf
    // ANYWHERE in the (possibly nested) effect tree — flattening the
    // tree into the targeted-leaf scan below. (Pre-fix the flat loop
    // matched only top-level variants, hit the `_ => {}` catch-all on
    // the Sequence wrapper, and fell through to the conservative
    // `true` — never pruning a zero-legal-target Sequence-wrapped
    // spell ⇒ the G7 S6 Class-D `cast_spell` LEGAL_SET_EQ asymmetry.)
    // Untargeted bodies still fall through to `true`.
    let controller = controller_idx as u8;
    // Depth-first flatten: descend `Effect::Sequence` children so the
    // targeted-leaf match arms below see the leaves Python's Sequence
    // protocol delegates to. Bounded by the registry's finite parsed
    // effect tree (no cycles — `parse_effect` builds a tree).
    fn flatten_effects<'e>(
        effs: &'e [crate::cards::Effect],
        out: &mut Vec<&'e crate::cards::Effect>,
    ) {
        use crate::cards::Effect;
        for e in effs.iter() {
            if let Effect::Sequence { effects } = e {
                flatten_effects(effects, out);
            } else {
                out.push(e);
            }
        }
    }
    let mut flat: Vec<&Effect> = Vec::new();
    flatten_effects(&printing.effects, &mut flat);
    for effect in flat.into_iter() {
        match effect {
            // ANY-creature scan (NOT controller-restricted). Mirrors
            // Python `DepressurizeEffect.legal_targets`
            // (`effects/bespoke.py:1545-1552`: every creature on
            // `state.battlefield`) — and the resolve arm's
            // `bf_iter_etb_sorted` any-creature scan
            // (`state_ops_resolve.rs:6042`).
            Effect::DepressurizeEffect { .. }
            | Effect::DealDamageToTargetCreatureWithExcessLanderRider {
                ..
            } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Focus Fire: `IsCreature() & (IsAttacking() |
            // IsBlocking())` (Python `FocusFireEffect.__post_init__`
            // TargetSpec, `effects/bespoke.py:1771-1782`). A creature
            // is attacking iff its id is in `state.attackers`;
            // blocking iff it appears as a `blocker_id` in
            // `state.blockers` — the EXACT predicate the resolve arm
            // uses (`state_ops_resolve.rs` FocusFire arm).
            Effect::FocusFireEffect { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        let is_creature = state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false);
                        if !is_creature {
                            return false;
                        }
                        let attacking =
                            state.attackers.iter().any(|a| *a == o.id);
                        let blocking = state
                            .blockers
                            .iter()
                            .any(|(_, b)| *b == o.id);
                        attacking || blocking
                    });
                if !any {
                    return false;
                }
            }
            // Radiant Strike's destroy half: `IsArtifact() |
            // (IsCreature() & IsTapped())`, non-indestructible
            // (Python `DestroyTargetArtifactOrTappedCreature`
            // `effects/generic/destroy.py:487-493`; the resolve arm's
            // exact typed-OR filter `state_ops_resolve.rs:5548-5567`).
            Effect::DestroyTargetArtifactOrTappedCreature => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        let Some(p) =
                            state.registry.lookup(o.printing_id)
                        else {
                            return false;
                        };
                        let is_artifact = p
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_creature = p
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        if !(is_artifact || (is_creature && o.tapped)) {
                            return false;
                        }
                        !p.keywords
                            .iter()
                            .any(|k| k == "indestructible")
                    });
                if !any {
                    return false;
                }
            }
            // Seedship Impact: `IsArtifact() | IsEnchantment()`,
            // non-indestructible (Python
            // `SeedshipImpactEffect.legal_targets`
            // `effects/bespoke.py:1227-1235`; the resolve arm's exact
            // filter `state_ops_resolve.rs:5696-5719`).
            Effect::SeedshipImpactEffect { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        let Some(p) =
                            state.registry.lookup(o.printing_id)
                        else {
                            return false;
                        };
                        let is_artifact = p
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_enchantment = p
                            .card_types
                            .iter()
                            .any(|t| t == "Enchantment");
                        if !(is_artifact || is_enchantment) {
                            return false;
                        }
                        !p.keywords
                            .iter()
                            .any(|k| k == "indestructible")
                    });
                if !any {
                    return false;
                }
            }
            // Biosynthic Burst: creature the CONTROLLER controls
            // (Python `BiosynthicBurstEffect.legal_targets`
            // `effects/bespoke.py:1478-1485`: `o.controller_id ==
            // controller_id` — the controller-restricted deviation;
            // the resolve arm's controller-only scan
            // `state_ops_resolve.rs:6285-6302`).
            //
            // **Walk BOTH battlefields and filter on
            // `o.controller`** (NOT `state.players[controller_idx].
            // battlefield`). A creature physically lives on its
            // OWNER's battlefield list but its `controller` may have
            // been flipped by a control-changing effect (Tractor
            // Beam's `attached_grants_control` is the live driver —
            // Aura-attachment Phase 2 S2). Python walks the combined
            // `state.battlefield` and reads `o.controller_id`, so the
            // owner-only walk pre-fix mis-narrowed the legal-target
            // scan and missed control-flipped creatures the spell
            // CAN legally target. Surfaced by the Tractor Beam
            // bench-loader admit follow-up — arch14/s4/step2249
            // LEGAL_SET_EQ asymmetry.
            Effect::BiosynthicBurstEffect => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        if o.controller != controller {
                            return false;
                        }
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Cut Propulsion: ANY-creature scan (NOT controller-
            // restricted). Mirrors Python `CutPropulsionEffect.legal_
            // targets` (`effects/bespoke.py:1834-1841`: every creature
            // on `state.battlefield`) — the established
            // `DepressurizeEffect` ANY-creature shape. The flying-
            // doubler is intrinsic to the resolve amount, NOT a
            // legality predicate.
            Effect::CutPropulsionEffect => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Invasive Maneuvers: ANY-creature scan (NOT controller-
            // restricted). Mirrors Python
            // `DealDamageToTargetCreatureSpacecraftBoosted`'s
            // `IsCreature()` slot filter (`damage.py:867-873`) — the
            // established `CutPropulsionEffect` ANY-creature shape.
            // The Spacecraft-boosted amount is intrinsic to the
            // resolve arm (read off the source-controller's bf), NOT
            // a legality predicate.
            Effect::InvasiveManeuversEffect { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Shattered Wings's destroy half: `IsArtifact() |
            // IsEnchantment() | (IsCreature() & HasKeyword("flying"))`,
            // non-indestructible (Python
            // `DestroyTargetArtifactEnchantmentOrFlyingCreature`
            // `effects/generic/destroy.py:501-521`; the resolve arm's
            // exact 3-way OR filter). The "flying" leg reads
            // EFFECTIVE keywords via `battlefield_has_keyword`
            // (Python's `HasKeyword.matches` reads `effective_
            // keywords` — same fold). NOT controller-restricted.
            Effect::DestroyTargetArtifactEnchantmentOrFlyingCreature => {
                let mut found = false;
                for o in crate::bf_iter::bf_iter_etb_sorted(state) {
                    let Some(p) = state.registry.lookup(o.printing_id)
                    else {
                        continue;
                    };
                    let is_artifact =
                        p.card_types.iter().any(|t| t == "Artifact");
                    let is_enchantment =
                        p.card_types.iter().any(|t| t == "Enchantment");
                    let is_creature =
                        p.card_types.iter().any(|t| t == "Creature");
                    let has_flying = is_creature
                        && crate::state_ops_combat::battlefield_has_keyword(
                            state, o.id, "flying",
                        );
                    if !(is_artifact || is_enchantment || has_flying) {
                        continue;
                    }
                    if p.keywords.iter().any(|k| k == "indestructible") {
                        continue;
                    }
                    found = true;
                    break;
                }
                if !found {
                    return false;
                }
            }
            // Exorcise's exile half (Effect-uncovered single-gated
            // tier, Duskmourn): `IsArtifact() | IsEnchantment() |
            // (IsCreature() & effective_power >= 4)`. Mirrors Python
            // `ExorciseEffect.legal_targets` (`effects/bespoke.py:
            // 2308-2315`) — its `_matches` helper returns True iff
            // `is_artifact OR "Enchantment" in card_types OR
            // (is_creature AND effective_power(state, obj) >= 4)`. The
            // power-4 leg reads EFFECTIVE power via the file-local
            // `legal_effective_power` helper (folds base +
            // ueot_power_bonus + obj.counters — matches Python's
            // `effective_power` = `_compute_effective_pt` at
            // game.py:2138-2251 and the resolve arm's
            // `read_doubling_power` reader). **Indestructible is NOT
            // a disqualifier** — exile bypasses indestructible per
            // CR 702.12 (the resolve arm has no indestructible-skip;
            // mirror that here). NOT controller-restricted.
            Effect::ExorciseEffect => {
                let mut found = false;
                for o in crate::bf_iter::bf_iter_etb_sorted(state) {
                    let Some(p) = state.registry.lookup(o.printing_id)
                    else {
                        continue;
                    };
                    let is_artifact =
                        p.card_types.iter().any(|t| t == "Artifact");
                    let is_enchantment =
                        p.card_types.iter().any(|t| t == "Enchantment");
                    let is_creature =
                        p.card_types.iter().any(|t| t == "Creature");
                    let power_4_plus = is_creature
                        && legal_effective_power(state, o.id) >= 4;
                    if !(is_artifact || is_enchantment || power_4_plus)
                    {
                        continue;
                    }
                    found = true;
                    break;
                }
                if !found {
                    return false;
                }
            }
            // Close Encounter — additional-cost-choice closure
            // (Cycle 3, 2026-05-21; ~16,098 17lands decks). ANY-
            // creature scan (NOT controller-restricted). Mirrors
            // Python `CloseEncounterEffect.legal_targets`
            // (`effects/bespoke.py:1085-1092`: every Creature on
            // `state.battlefield`) — the established
            // `CutPropulsionEffect` / `DepressurizeEffect` ANY-
            // creature shape. The damage amount is intrinsic to
            // the resolve arm (read off the additional-cost-choice
            // object's power), NOT a legality predicate. The
            // additional-cost predicate (`additional_cost_choose_
            // your_creature` flag) is checked separately by
            // `additional_costs_payable` (the CR 601.2f gate).
            Effect::CloseEncounterEffect => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Emergency Eject: any non-land permanent on either
            // battlefield, non-indestructible (Python
            // `DestroyTargetNonlandPermanentWithLanderForController
            // .legal_targets` `effects/generic/destroy.py:602-609`:
            // `[("permanent", o.id) for o in state.battlefield if
            // not o.printing.is_land]`). The resolve arm SKIPS
            // indestructible at scan time (same Seedship precedent);
            // mirror that here so the legality gate agrees with the
            // resolve filter. NOT controller-restricted (it's an
            // opponent-targeting removal spell).
            Effect::EmergencyEjectEffect { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        let Some(p) =
                            state.registry.lookup(o.printing_id)
                        else {
                            return false;
                        };
                        let is_land = p
                            .card_types
                            .iter()
                            .any(|t| t == "Land");
                        if is_land {
                            return false;
                        }
                        !p.keywords
                            .iter()
                            .any(|k| k == "indestructible")
                    });
                if !any {
                    return false;
                }
            }
            // **ModalChooseOne envelope — Cycle 6 (Drill Too Deep
            // closure, 2026-05-21).**
            // `PutChargeCountersOnTargetSpacecraftOrPlanetYouControl`:
            // source-controller-restricted (Python's `legal_targets`
            // at `effects/generic/counters.py:508-516` filters
            // `o.controller_id == controller_id`). Walks BOTH bfs
            // and filters on `o.controller == controller` (the same
            // control-flip-aware shape `BiosynthicBurstEffect` uses,
            // per the Tractor Beam follow-up). Satisfiable iff the
            // source-controller controls AT LEAST one Spacecraft or
            // Planet permanent.
            Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        if o.controller != controller {
                            return false;
                        }
                        let Some(p) =
                            state.registry.lookup(o.printing_id)
                        else {
                            return false;
                        };
                        let is_spacecraft = p
                            .subtypes
                            .iter()
                            .any(|s| s == "Spacecraft");
                        let is_planet = p
                            .subtypes
                            .iter()
                            .any(|s| s == "Planet");
                        is_spacecraft || is_planet
                    });
                if !any {
                    return false;
                }
            }
            // Diplomatic Relations: the only 2-slot grammar
            // (`target_slot_kinds() == ("my_creature", "creature")`,
            // `min_t == 2`). Python `_targets_are_satisfiable`
            // `:101-114` per-slot enforcement: each slot-kind needs ≥
            // its slot-count of legal entries. Mirror the cast-time
            // 2-slot bind (`state_ops_cast.rs:894-923`): slot 0 = a
            // creature the caster controls, slot 1 = a creature a
            // NON-caster controls. Satisfiable iff BOTH exist.
            Effect::DiplomaticRelationsEffect => {
                let mut have_mine = false;
                let mut have_theirs = false;
                for o in crate::bf_iter::bf_iter_etb_sorted(state) {
                    let is_creature = state
                        .registry
                        .lookup(o.printing_id)
                        .map(|p| {
                            p.card_types
                                .iter()
                                .any(|t| t == "Creature")
                        })
                        .unwrap_or(false);
                    if !is_creature {
                        continue;
                    }
                    if o.controller == controller {
                        have_mine = true;
                    } else {
                        have_theirs = true;
                    }
                    if have_mine && have_theirs {
                        break;
                    }
                }
                if !(have_mine && have_theirs) {
                    return false;
                }
            }
            // **Reroute Systems closure (2026-05-21) — mode 0 leaf
            // `PumpTargetArtifactOrCreatureUntilEOT`.** Satisfiable
            // iff SOME artifact-or-creature on EITHER bf. Mirrors
            // the cast-time per-mode detector's filter at
            // `state_ops_cast.rs::cast_spell_noncreature` exactly
            // (ANY-permanent, no controller filter, no indestructible
            // reject — pump grants additive keywords, not
            // destructive). Python SoT: `Pump.legal_targets` walks
            // any permanent matching the filter
            // (`effects/generic/pump.py` unified `Pump`).
            Effect::PumpTargetArtifactOrCreatureUntilEOT { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        let Some(p) = state.registry.lookup(o.printing_id)
                        else {
                            return false;
                        };
                        let is_artifact =
                            p.card_types.iter().any(|t| t == "Artifact");
                        let is_creature =
                            p.card_types.iter().any(|t| t == "Creature");
                        is_artifact || is_creature
                    });
                if !any {
                    return false;
                }
            }
            // **scour-for-scrap-cycle-2 (2026-05-21) — Scour for
            // Scrap mode 1 leaf
            // `ReturnTargetArtifactCardFromYourGraveyardToHand`.**
            // Satisfiable iff the caster controls AT LEAST one
            // artifact card in their graveyard. Mirrors the cast-
            // time per-mode detector's filter exactly (caster's
            // graveyard, artifact in card_types). Note that for
            // Scour for Scrap specifically the printed effect is a
            // `ModalChooseOne(or_both=True)` whose mode 0 is
            // untargeted (`SearchLibraryForCard`); requires_target()
            // on the modal returns False because ANY mode is
            // playable, so this arm never short-circuits the
            // satisfiability gate to False for the modal as a whole
            // — but the per-leaf arm is honest about the targeted-
            // mode predicate for any future card that prints this
            // leaf at the top level (or in a non-or_both modal).
            // Python SoT: `ReturnFromGraveyard.legal_targets`
            // (effects/generic/graveyard.py) walks the controller's
            // graveyard and emits `("graveyard_artifact", card.id)`
            // entries per artifact card.
            Effect::ReturnTargetArtifactCardFromYourGraveyardToHand => {
                let any = state.players[controller_idx]
                    .graveyard
                    .iter()
                    .any(|o| {
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Artifact")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // **Reroute Systems closure (2026-05-21) — mode 1 leaf
            // `DealDamageToTargetTappedCreature`.** Satisfiable iff
            // SOME tapped creature on EITHER bf. Mirrors the cast-
            // time per-mode detector's filter exactly (ANY-side,
            // creature + tapped). Python SoT: the unified
            // `DealDamage.legal_targets` walks any creature
            // matching the TargetSpec filter `IsCreature() &
            // IsTapped()`.
            Effect::DealDamageToTargetTappedCreature { .. } => {
                let any = crate::bf_iter::bf_iter_etb_sorted(state)
                    .into_iter()
                    .any(|o| {
                        if !o.tapped {
                            return false;
                        }
                        state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                    });
                if !any {
                    return false;
                }
            }
            // Every other Effect (untargeted bodies — the dominant
            // bridge-simple-creature axis whose `spell_effect is
            // None`; or a not-yet-admitted targeted shape): the
            // conservative trivially-satisfiable branch. NEVER
            // pruned ⇒ never narrower than Python ⇒ parity-neutral
            // by construction (the single-script-replay §4 proof's
            // subset-preserving property).
            _ => {}
        }
    }
    true
}

/// **legal_decisions mana-awareness Slice 3 — additional-cost prune
/// (CR 601.2f/601.2h).** Returns `true` iff casting the spell printed
/// by `printing_id`, with the source controlled by `controller_idx`,
/// has its cast-time additional NON-mana costs payable RIGHT NOW.
/// Mirrors Python `mtg_bot.legal._additional_costs_payable`
/// (`src/mtg_bot/legal.py:117-165`) — the SoT — for the live
/// bridge-admitted CastSpell surface.
///
/// **Python parity contract (verify-first, file:line precise).**
/// Python's gate (`legal.py:124-165`) returns `False` iff ANY of:
///   - `additional_cost_sacrifice_creature_or_artifact` AND the
///     caster controls NO creature-or-artifact (`:124-131`).
///   - `additional_cost_sacrifice_creature_or_enchantment` AND the
///     caster controls NO creature-or-Enchantment (`:132-147`; an
///     Aura is an Enchantment subtype — `"Enchantment" in
///     card_types` catches it).
///   - `additional_cost_choose_your_creature` AND the caster
///     controls NO battlefield creature AND owns NO warped creature
///     card in exile (`:148-164`).
/// Otherwise `True` (no additional cost, or it is payable).
///
/// **Live-surface incidence is ZERO (verify-first, evidence-bound —
/// the design doc §5 prediction realized).** All 4 EoE printings
/// carrying any additional-cost flag (Embrace Oblivion / Scrounge
/// for Eternity — sac creature-or-artifact; Final Vengeance — sac
/// creature-or-enchantment; Close Encounter — choose-your-creature)
/// are bridge-DECLINED by BOTH `_printing_is_bridge_simple_spell`
/// and `_printing_is_bridge_simple_creature`, so NO card on any of
/// the 6 live Rust CastSpell emission axes carries one. The three
/// `printing.additional_cost_*` fields are therefore `false` for
/// every live-surface card ⇒ this predicate is `true` for every
/// live-surface card ⇒ it NEVER prunes a live emission ⇒ it is
/// **parity-NEUTRAL by construction** (never narrower than Python —
/// the single-script-replay §4 subset-preserving property, same as
/// Slices 1–2's conservative-branch arguments). The full predicate
/// is implemented anyway (NOT a stub) so it is the faithful Python
/// mirror AND the single named extension point that fires correctly
/// the instant an additional-cost card becomes bridge-admitted —
/// honest-frontier discipline, not a diverging stub. Documented in
/// `docs/SEARCH_LIMITATIONS.md`.
fn additional_costs_payable(
    state: &crate::state::GameState,
    printing_id: u32,
    controller_idx: usize,
) -> bool {
    let Some(printing) = state.registry.lookup(printing_id) else {
        // Unknown printing — conservative: don't prune (never
        // narrower than Python).
        return true;
    };
    let none_flag_set = !printing
        .additional_cost_sacrifice_creature_or_artifact
        && !printing.additional_cost_sacrifice_creature_or_enchantment
        && !printing.additional_cost_choose_your_creature;
    if none_flag_set {
        // The dominant (in fact, on the live surface, the ONLY) case:
        // no additional cost ⇒ trivially payable (Python's implicit
        // `return True` fall-through). Fast-path before any bf scan.
        return true;
    }
    // "creature or artifact you control" (Python `:124-131`).
    if printing.additional_cost_sacrifice_creature_or_artifact {
        let have = state.players[controller_idx]
            .battlefield
            .iter()
            .any(|o| {
                state
                    .registry
                    .lookup(o.printing_id)
                    .map(|p| {
                        p.card_types
                            .iter()
                            .any(|t| t == "Creature" || t == "Artifact")
                    })
                    .unwrap_or(false)
            });
        if !have {
            return false;
        }
    }
    // "creature or enchantment you control" (Python `:132-147`; Auras
    // count — `"Enchantment" in card_types`).
    if printing.additional_cost_sacrifice_creature_or_enchantment {
        let have = state.players[controller_idx]
            .battlefield
            .iter()
            .any(|o| {
                state
                    .registry
                    .lookup(o.printing_id)
                    .map(|p| {
                        p.card_types.iter().any(|t| {
                            t == "Creature" || t == "Enchantment"
                        })
                    })
                    .unwrap_or(false)
            });
        if !have {
            return false;
        }
    }
    // "a creature you control OR a warped creature card you own in
    // exile" — either source satisfies (Python `:148-164`).
    if printing.additional_cost_choose_your_creature {
        let have_bf = state.players[controller_idx]
            .battlefield
            .iter()
            .any(|o| {
                state
                    .registry
                    .lookup(o.printing_id)
                    .map(|p| {
                        p.card_types.iter().any(|t| t == "Creature")
                    })
                    .unwrap_or(false)
            });
        if !have_bf {
            // Python checks the OWNER's exile for a warped creature
            // card the player owns (`legal.py:155-163`:
            // `o.owner_id == player_id and o.warped and
            // o.printing.is_creature`). Rust models exile per-player
            // (`state.players[idx].exile`), so iterating
            // `players[controller_idx].exile` IS the
            // `owner_id == player_id` scoping (the same structural
            // owner-via-zone shape the warp exile-recast loop at the
            // bottom of `legal_decisions` uses — no separate `owner`
            // field on `GameObject`). Mirror `warped` +
            // `is_creature`.
            let have_exiled = state.players[controller_idx]
                .exile
                .iter()
                .any(|o| {
                    o.warped
                        && state
                            .registry
                            .lookup(o.printing_id)
                            .map(|p| {
                                p.card_types
                                    .iter()
                                    .any(|t| t == "Creature")
                            })
                            .unwrap_or(false)
                });
            if !have_exiled {
                return false;
            }
        }
    }
    true
}

/// Build a `{"kind": "attacker_decision", "attacker_id": <id>,
/// "attack": <bool>}` dict — the G1 factored per-attacker
/// sub-decision (the C1-analog for the combat axis). Byte-compatible
/// mirror of Python's `AttackerDecision(attacker_id: int, attack:
/// bool)` dataclass (`src/mtg_bot/search/combat_decisions.py:44-55`)
/// serialized to a dict, and round-trips through
/// `actions.rs:parse_action`'s `"attacker_decision"` arm /
/// `action_to_pydict`'s `Action::AttackerDecision` arm without key
/// drift. Replaced the former whole-action `declare_attackers_dict`
/// powerset-subset emitter — the powerset emission is gone (the
/// factored shape is the byte-compatible search-seam shape; see the
/// module docstring's "DeclareAttackers extension" section, now the
/// factored-emission section).
fn attacker_decision_dict<'py>(
    py: Python<'py>,
    attacker_id: u32,
    attack: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "attacker_decision")?;
    d.set_item("attacker_id", attacker_id)?;
    d.set_item("attack", attack)?;
    Ok(d)
}

/// Build a `{"kind": "blocker_decision", "blocker_id": <id>,
/// "attacker_id": <id|None>}` dict — the G1 factored per-blocker
/// sub-decision. Byte-compatible mirror of Python's
/// `BlockerDecision(blocker_id: int, attacker_id: Optional[int])`
/// dataclass (`src/mtg_bot/search/combat_decisions.py:57-73`)
/// serialized to a dict. `attacker_id = None` ⇒ the "don't block"
/// branch (PyO3 sets a Python `None`; `parse_action`'s
/// `"blocker_decision"` arm maps absent-or-None → `None`).
/// Round-trips through `action_to_pydict`'s
/// `Action::BlockerDecision` arm. Replaced the former whole-action
/// `declare_blockers_dict` product-assignment emitter — the
/// `(M+1)^N` product emission is gone.
fn blocker_decision_dict<'py>(
    py: Python<'py>,
    blocker_id: u32,
    attacker_id: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "blocker_decision")?;
    d.set_item("blocker_id", blocker_id)?;
    d.set_item("attacker_id", attacker_id)?;
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
        //
        // **G7 S6 Class-B option-β (coordinated engine+adapter slice;
        // new-code-is-right toward the `simulate.py` SoT).** The
        // former early `return` here gated ALL emission — including
        // the factored-combat DECLARE_ATTACKERS / DECLARE_BLOCKERS
        // blocks — on `priority_player == player_id`. That only held
        // because the Rust-driver adapter eagerly forced
        // `priority_player = active` after every step end
        // (`rust_engine_adapter.py:473`), so during the
        // factored-combat decision window Rust's `priority_player`
        // happened to equal the structural decision-maker. The
        // `simulate.py` SoT does NOT reset priority at the
        // BEGINNING_OF_COMBAT pass-round end — it CARRIES
        // `priority_player_idx` through the factored-combat window
        // (only the DECLARE_ATTACKERS `priority_loop`'s first pass
        // resets it, AFTER the attacker decision —
        // `simulate.py:222-247`); `MtgGameStateAdapter` faithfully
        // mirrors that (its `_enter_declare_attackers` /
        // `_enter_declare_blockers` RETURN WITHOUT
        // `_enter_priority_for_current_step`,
        // `engine_adapter.py:1312-1316/:1358-1361`, leaving
        // `priority_player_idx` carried; `active_player()` overrides
        // the decision-maker to active/defender,
        // `engine_adapter.py:762-766`). The coordinated
        // `rust_engine_adapter.py` change defers that eager reset for
        // factored-combat steps so Rust's `priority_player` stays
        // carried like the SoT — at which point the old
        // `priority_player == player_id` gate would WRONGLY suppress
        // the factored-combat emission (chicken-and-egg: detecting
        // the window needs the emission, the old gate needs
        // priority == active). So: relax this to a flag, let the
        // factored-combat blocks below emit keyed on the STRUCTURAL
        // decision-maker (attackers: `active_player == player_id`;
        // blockers: `player_id == 1 - active_player`, Rust's
        // canonical defender), and re-apply the byte-identical
        // priority-window early return immediately BEFORE the genuine
        // priority-window emission (the instant-speed CastSpell /
        // PlayLand / sorcery / ActivateAbility block) — fencing ONLY
        // that, exactly as Python's `legal.py:180-181` fences its
        // priority-window emission. See
        // `docs/G7_MIDGAME_STATEEQ_VERIFY_FINDING.md` §9/§9.1.
        let is_priority_holder = self.priority_player == player_id;

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
        // Creature, not tapped, not summoning-sick OR has haste,
        // does NOT have the defender keyword. The defender short-
        // circuit was added by CE17 closure (I40) — mirrors the
        // apply-time gate at `state_ops_combat.rs:639-641`.
        // Continuous "can't attack" restrictions remain deferred —
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
                // Defender (CR 702.3): a creature with defender can't
                // attack. Mirrors Python's `combat.py:eligible_attackers`
                // (line 22) "defender" not in effective_keywords(...)
                // and the apply-time gate at
                // `state_ops_combat.rs:639-641`. Uses the shared
                // `battlefield_has_keyword` helper so the UEOT-granted
                // case (CR 613 layer 6 keyword grants) is honored
                // alongside printed defender — same shape the
                // apply-time gate enforces. See I40 for the post-CE17
                // invariant pin.
                if crate::state_ops_combat::battlefield_has_keyword(self, obj.id, "defender") {
                    continue;
                }
                eligible.push(obj.id);
            }

            // **G1 — factored per-attacker emission (the C1-analog
            // for the combat axis).** Replaces the former capped
            // powerset block. Byte-compatible mirror of the Python
            // search seam: `MtgGameStateAdapter.legal_decisions`
            // (`engine_adapter.py:808-814`) issues, for the HEAD
            // pending attacker, exactly two `AttackerDecision`s —
            // `attack=True` then `attack=False`. The adapter's
            // pending list is `sorted(o.id for o in eligible)`
            // (`engine_adapter.py:1315`) so the head is the
            // smallest-id eligible attacker; the codec's no-adapter
            // fallback uses the same `min(o.id for o in elig)`
            // (`action_encoding.py:990-995`). We emit for that same
            // head so the dict maps 1:1 to the codec's implicit-
            // attacker `SLOT_ATTACKER_ATTACK` / `SLOT_ATTACKER_SKIP`
            // slots (`action_encoding.py:708-712`).
            //
            // The B3 Python provider (a sibling slice — NOT this
            // one) owns the per-creature accumulation + the
            // finalize-to-whole-`declare_attackers` step
            // (`engine_adapter.py:1080-1108` ported verbatim per
            // `NN_RUST_DECISION_SOURCING_WIRING_DESIGN.md` §3.3); it
            // applies the chosen factored decision (which `tap`s a
            // declared attacker via the round-5 primitive, removing
            // it from `eligible` on the next call so the head
            // advances) OR, for `attack=false`, the provider pops
            // the head from its own pending list and re-queries —
            // at which point the next-smallest eligible id becomes
            // the head here. This is the stateless-Rust /
            // stateful-provider split the design specifies; Rust
            // emits the head's factored option pair, the provider
            // drives the walk. Vigilance creatures stay untapped,
            // so the provider's pending-pop is what advances the
            // head past a non-attacking or vigilant declared
            // creature — exactly the Python adapter's
            // `_pending_attackers.pop(0)` semantics
            // (`engine_adapter.py:1097`).
            //
            // Empty `eligible` ⇒ no combat decision is emitted (the
            // adapter's `_enter_declare_attackers` skips the
            // factored phase entirely when `not eligible`,
            // `engine_adapter.py:1295-1310`; the step's priority
            // window opens instead — handled by the priority
            // emission below). The former powerset block emitted an
            // empty-subset `declare_attackers` dict even with zero
            // eligible; the factored shape correctly emits NOTHING,
            // matching Python (which has no DeclareAttackers /
            // AttackerDecision in `legal_actions` at all and whose
            // adapter skips the phase).
            if let Some(&head) = eligible.iter().min() {
                actions.push(attacker_decision_dict(py, head, true)?);
                actions.push(attacker_decision_dict(py, head, false)?);
            }
        }

        // DeclareBlockers emission (Rust-only — Python's
        // `legal_actions` does NOT emit this variant either; see the
        // module docstring's "DeclareBlockers extension" section for
        // the divergence rationale + the `search/combat_decisions.py`
        // factored-search alternative Python uses).
        //
        // Gate: `phase == PHASE_DECLARE_BLOCKERS` AND
        // `player_id == 1 - active_player` (the STRUCTURAL defender
        // declares, NOT the attacker — CR 509.1) AND
        // `stack.is_empty()` AND `!self.attackers.is_empty()` (there
        // is at least one attacking creature — see the no-attackers
        // gate below).
        //
        // **G7 S6 Class-B option-β.** The gate was formerly
        // `self.priority_player != self.active_player`. That only
        // distinguished the defender because the Rust-driver adapter
        // forced `priority_player == active` during the
        // factored-combat window; with the coordinated adapter change
        // carrying `priority_player` like the `simulate.py` SoT, that
        // proxy no longer holds. Key on the STRUCTURAL defender
        // instead: `player_id == 1 - self.active_player` — Rust's own
        // canonical defender id (the SAME `1 - self.active_player`
        // the combat-damage plan uses, `state_ops_combat.rs:941`, and
        // `choose_first_creature_for_defender`,
        // `state_ops_choices.rs:104`), byte-equal to Python
        // `MtgGameStateAdapter`'s `_combat_defender_id =
        // _next_alive(state, (active+1) % n)`
        // (`engine_adapter.py:1331-1336`) in the loader-admitted
        // no-loss 2-player model and to Python's
        // `active_player()` returning `_combat_defender_id` during
        // `_COMBAT_BLOCKERS` (`engine_adapter.py:764-766`). This
        // emits the factored blocker decision for the structural
        // defender independent of who holds `priority_player` —
        // exactly what the DECLARE_ATTACKERS block already does for
        // the active player (`self.active_player == player_id`).
        //
        // **No-attackers gate (G7 combat-blocker LEGAL_SET fix,
        // new-code-is-right toward the Python SoT).** Python's
        // `MtgGameStateAdapter._enter_declare_blockers`
        // (`engine_adapter.py:1322-1329`) computes
        // `attackers = [o for o in battlefield if o.attacking]` and,
        // if it is EMPTY, takes the `if not attackers:
        // self._enter_priority_for_current_step(); return` branch —
        // it NEVER sets `_combat_phase = _COMBAT_BLOCKERS`, so the
        // adapter's `legal_decisions()` (the rust_shadow combat-window
        // reference, `rust_shadow.py:281-286`) emits NO
        // `BlockerDecision` at all (it falls through to the raw
        // priority-window `legal_actions`). Pre-fix the Rust emitter
        // gated only on phase/priority/empty-stack + an eligible
        // (untapped) defender creature, so with zero attackers it
        // STILL emitted `('blocker', head, None)` — over-emitting a
        // blocker decision Python prunes (the
        // `rust\\py=[('blocker', N, None)] py\\rust=[]` G7
        // divergence, `docs/G7_MIDGAME_STATEEQ_VERIFY_FINDING.md` §5;
        // arch2/s1 step 312 N=9, arch5/s0 step 344 N=12). The Rust
        // canonical "is attacking" predicate is `state.attackers`
        // membership (see the `FocusFireEffect` arm above,
        // `state_ops_legal.rs:789-790`: *"A creature is attacking iff
        // its id is in `state.attackers`"*); Python's `o.attacking`
        // is the byte-equal analog. `self.attackers` is the
        // active-player-committed attacker set
        // (`state_ops_combat.rs:declare_attacker` pushes onto it).
        // So `!self.attackers.is_empty()` byte-mirrors Python's
        // `if not attackers` no-blocking-phase guard.
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
            && player_id == 1 - self.active_player
            && self.stack.is_empty()
            && !self.attackers.is_empty()
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

            // **G1 — factored per-blocker emission (the C1-analog
            // for the combat axis).** Replaces the former capped
            // `(M+1)^N` product block. Byte-compatible mirror of the
            // Python search seam: `MtgGameStateAdapter.legal_decisions`
            // (`engine_adapter.py:815-828`) issues, for the HEAD
            // pending blocker, one `BlockerDecision(blocker_id=head,
            // attacker_id=atk)` per currently-attacking creature (in
            // `attacking`-iteration order) followed by a final
            // `BlockerDecision(blocker_id=head, attacker_id=None)`
            // ("don't block"). The adapter's pending list is
            // `sorted(o.id for o in eligible)`
            // (`engine_adapter.py:1360`) so the head is the
            // smallest-id eligible blocker; the codec's no-adapter
            // fallback uses the same `min(o.id for o in elig)`
            // (`action_encoding.py:1007-1012`). We emit for that
            // same head.
            //
            // **Attacker branch ids + ordering.** Branches come from
            // `self.attackers` — the set the active player committed
            // during DECLARE_ATTACKERS (`state_ops_combat.rs:
            // declare_attacker` pushes onto this Vec). The Python
            // adapter walks `[o.id for o in battlefield if
            // o.attacking]`; the codec recovers the policy slot via
            // `_attacking_creatures` = `sorted(... by id)`
            // (`action_encoding.py:414-420,714-733`). The codec
            // derives `atk_pos` from the `attacker_id` VALUE (its
            // index in the sorted attacking list), so the slot is
            // independent of the order we list the branch dicts in —
            // we mirror the adapter's literal seam order
            // (`self.attackers` declaration order, "don't block"
            // last) for faithfulness; list order is parity-NEUTRAL
            // under the Python-sourced single-script-replay harness
            // (the C1 §4 proof — the harness never sources a choice
            // from `rust.legal_decisions`).
            //
            // The B3 Python provider (sibling slice — NOT this one)
            // owns the per-blocker accumulation + finalize-to-whole-
            // `declare_blockers` (`engine_adapter.py:1114-1144`
            // ported verbatim per the wiring design §3.3, which
            // keeps the CE16 menace post-validation `.retain` at the
            // finalize boundary). It pops its pending head and
            // re-queries; the next-smallest eligible blocker id
            // becomes the head here — the stateless-Rust /
            // stateful-provider split the design specifies.
            //
            // Empty `eligible_blockers` ⇒ no combat decision is
            // emitted (the adapter skips the factored blocker phase
            // entirely when `not eligible`,
            // `engine_adapter.py:1339-1356`, opening the priority
            // window instead — handled by the priority emission
            // below). The former product block emitted an empty-
            // assignment `declare_blockers` dict even with zero
            // eligible blockers; the factored shape correctly emits
            // NOTHING, matching Python (no DeclareBlockers /
            // BlockerDecision in `legal_actions` at all, adapter
            // skips the phase).
            let attackers: &[u32] = &self.attackers;
            if let Some(&head) = eligible_blockers.iter().min() {
                for &atk in attackers.iter() {
                    actions.push(blocker_decision_dict(
                        py,
                        head,
                        Some(atk),
                    )?);
                }
                actions.push(blocker_decision_dict(py, head, None)?);
            }
        }

        // **G7 S6 Class-B option-β — relocated priority-window
        // gate.** This is the byte-identical relocation of the
        // former line-1199 `if self.priority_player != player_id {
        // return Ok(actions); }` early return. Everything BELOW this
        // point is the genuine priority-window emission (instant-
        // speed CastSpell, PlayLand, sorcery-speed CastSpell,
        // ActivateAbility) — exactly what Python's `legal.py:180-181`
        // `if not _has_priority(...): return actions` fences (that
        // Python gate fences ALL of `legal_actions`, including the
        // instant-speed CastSpell block at `legal.py:217-301`). The
        // factored-combat DECLARE_ATTACKERS / DECLARE_BLOCKERS
        // blocks ABOVE now run independent of `priority_player`
        // (keyed on the structural decision-maker), mirroring the
        // `simulate.py` SoT which carries `priority_player_idx`
        // through the factored-combat window. Semantics for the
        // priority-window emission are UNCHANGED: an off-priority
        // caller still bails here with just `[PassPriority]`, exactly
        // as before — only the factored-combat emission moved above
        // the gate. See `docs/G7_MIDGAME_STATEEQ_VERIFY_FINDING.md`
        // §9/§9.1.
        if !is_priority_holder {
            return Ok(actions);
        }

        // Python's `_is_sorcery_timing` (lines 25-32):
        //   active == priority AND step in MAIN_STEPS AND stack
        //   empty AND _has_priority. Priority is already confirmed
        //   above; the remaining conjuncts gate the sorcery-speed
        //   block (PlayLand, sorcery-speed CastSpell, ActivateAbility).
        //
        // **CE19 closure**: pre-closure this short-circuit returned
        // early after the DeclareAttackers / DeclareBlockers
        // branches, dropping the CastSpell loop entirely when
        // sorcery timing failed. That mis-modeled the instant-speed
        // CastSpell window: a flash creature or printed Instant in
        // the priority holder's hand should emit CastSpell at any
        // priority window the holder has — active or not (CR 702.13a
        // for flash; CR 117.1 + 307.1 for instants). Post-closure
        // the CastSpell loop lives ABOVE the sorcery-timing gate and
        // gates per-card on
        // `is_instant || has("flash") || is_sorcery_timing` — same
        // shape as Python's `legal.py:217-301` instant-speed block
        // + `legal.py:335-445` sorcery-speed block sharing the
        // `seen_instant` HashSet. See I46 (post-closure invariant
        // pin in `docs/RUST_PARITY_GAPS.md`).
        let is_sorcery_timing = self.active_player == player_id
            && (self.phase == PHASE_MAIN1 || self.phase == PHASE_MAIN2)
            && self.stack.is_empty();

        // **CastSpell emission — instant-speed window first**
        // (Python `legal.py:217-301`). Walk the hand and emit a
        // CastSpell for any printed Instant or any card with the
        // "flash" keyword, regardless of sorcery timing. The dedup
        // HashSet is shared with the sorcery-speed pass below — a
        // card that could emit at both windows (degenerate but
        // harmless) is emitted ONCE at the instant-speed slot, same
        // as Python's `seen_instant` set living in scope across
        // both blocks.
        //
        // **Filter**:
        //   * Land cards never CastSpell (CR 305.1).
        //   * Per-card timing gate: `is_instant` (CR 307.1) OR
        //     `"flash" in keywords` (CR 702.13a). Other cards skip
        //     this pass and may be picked up by the sorcery-speed
        //     pass below.
        //
        // **Dedup** (Python lines 233-244 / 349-363): keyed on
        // `(printing.name, ...flags)` 6-tuple. With all flags
        // defaulted, this collapses to `(printing.name,)` — same
        // shape as the PlayLand dedup below.
        //
        // **Emission order** (matches Python's `legal.py`):
        // `[PassPriority, (DeclareAttackers...), CastSpell
        // (instant/flash)..., PlayLand..., CastSpell (sorcery)...,
        // ActivateAbility...]`. The instant-speed CastSpell pass
        // runs BEFORE the sorcery-timing gate (so a non-active
        // priority holder still emits flashes / instants); the
        // sorcery-speed pass runs AFTER PlayLand. This is the CE19
        // closure shape (see I46 in `docs/RUST_PARITY_GAPS.md`) —
        // pre-closure the entire CastSpell block was gated by the
        // single sorcery-timing short-circuit, dropping flash + any
        // future Instants at off-active priority windows.
        let mut seen_castspell: HashSet<&str> = HashSet::new();
        for obj in self.players[idx].hand.iter() {
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            let is_land = printing.card_types.iter().any(|t| t == "Land");
            if is_land {
                continue;
            }
            let is_instant = printing
                .card_types
                .iter()
                .any(|t| t == "Instant");
            let has_flash = printing.keywords.iter().any(|k| k == "flash");
            if !(is_instant || has_flash) {
                continue;
            }
            let name: &str = printing.name.as_ref();
            if seen_castspell.contains(name) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 1.** Mirror
            // Python's `_emit_hand_cast`-only-inside-`if _can_cast`
            // ordering (`legal.py:252-271`): compute base + warp
            // affordability FIRST, then dedup-insert + push ONLY the
            // affordable subset. `eff(c) = c` is the cost-reduction
            // seam — Rust's `cards.rs` carries NO `cost_reductions`
            // on the live bridge-admitted surface (the bridge
            // declines cost-reduction cards; `COST_REDUCTION_PHASE1`
            // is realizable-0), so the effective cost IS the printed
            // cost; the identity wrapper is the single named
            // insertion point for a future cost-reduction slice.
            let sk = crate::state_ops_mana::SpellKind::Any;
            // **legal_decisions mana-awareness Slice 2 —
            // target-satisfiability prune (CR 601.2c).** Python gates
            // every CastSpell emission on `_can_cast` AND
            // `_targets_are_satisfiable` (`legal.py:252-254`); a spell
            // whose targeted effect has ZERO legal targets is never
            // enumerated. Computed ONCE and `&&`-ed into BOTH the base
            // and warp affordability booleans below: warp is an
            // alternative COST, not a different spell body, so the
            // SAME spell effect (and thus the SAME target requirement)
            // applies to the base and warp casts of one card — both
            // must be target-satisfiable to emit. A non-targeting
            // spell / bridge-simple creature trivially returns `true`
            // (Python `:92-94` `effect is None or not
            // requires_target()`), so this never prunes the dominant
            // creature-cast axis (parity-neutral on it by
            // construction). Folding it into `*_payable` (rather than
            // a separate early `continue`) preserves the Slice-1
            // `seen_castspell` add-on-emit ordering VERBATIM (the
            // §3.5.2 load-bearing rule — the insert still happens only
            // on the path that actually pushes).
            //
            // **legal_decisions mana-awareness Slice 3 —
            // additional-cost prune (CR 601.2f/601.2h).** Python
            // gates each CastSpell on `_can_cast` AND
            // `_targets_are_satisfiable` AND `_additional_costs_-
            // payable` (`legal.py:252-254` instant / `:383-385`
            // sorcery). The non-mana-precondition is the conjunction
            // of the target + additional-cost gates (same spell body
            // for base & warp). On the live surface
            // `additional_costs_payable` is `true` for every card (the
            // 4 EoE additional-cost cards are bridge-DECLINED — design
            // §5 verified-zero-incidence), so this is parity-NEUTRAL
            // by construction; folding it into the SAME precondition
            // var preserves the Slice-1 `seen_castspell` ordering.
            let nonmana_ok =
                spell_targets_satisfiable(self, obj.printing_id, idx)
                    && additional_costs_payable(
                        self,
                        obj.printing_id,
                        idx,
                    );
            let base_payable = nonmana_ok
                && crate::state_ops_mana::can_pay_cost(
                    self,
                    idx,
                    eff(&printing.mana_cost),
                    &sk,
                );
            // Warp variant is gated on its OWN cost (`warp_cost`),
            // INDEPENDENT of the base `mana_cost` gate (mirrors
            // Python's separate `_can_cast(warp_cost)` at
            // `legal.py:264-266` — do NOT share one affordability
            // boolean across base+warp). The warp life gate
            // (`life > warp_additional_life_cost`) is preserved
            // verbatim (warp Phase 2 landed it; mirrors
            // `legal.py:269`). The Slice-2+3 `nonmana_ok`
            // precondition applies identically (same spell body) —
            // Python's warp arm also `&& _targets_are_satisfiable &&
            // _additional_costs_payable` (`legal.py:267-268`).
            let warp_payable = match printing.warp_cost.as_ref() {
                Some(wc) => {
                    nonmana_ok
                        && (self.players[idx].life as i64)
                            > (printing.warp_additional_life_cost
                                as i64)
                        && crate::state_ops_mana::can_pay_cost(
                            self, idx, eff(wc), &sk,
                        )
                }
                None => false,
            };
            if !base_payable && !warp_payable {
                // Nothing affordable/targetable for this card — DO NOT
                // insert into `seen_castspell` (Python only
                // `seen_instant.add`s inside `_emit_hand_cast`, which
                // is only called after `_can_cast` AND
                // `_targets_are_satisfiable` pass —
                // `legal.py:241-244`,`:252-255`). Skipping the insert
                // keeps a later same-named copy in a payable+targetable
                // context still emittable (the §3.5.2 load-bearing
                // ordering rule). Continue WITHOUT marking the name
                // seen.
                continue;
            }
            // At least one of base/warp is affordable → the card
            // emits, so claim the name-dedup slot now (mirrors
            // Python's `_emit_hand_cast` add-on-emit).
            seen_castspell.insert(name);
            if base_payable {
                actions.push(cast_spell_dict(py, obj.id)?);
            }
            if warp_payable {
                // Phase-1 warp: the use_warp=true variant, mirroring
                // Python legal.py:262-271 (instant-speed). The warp
                // dict is intentionally NOT added to `seen_castspell`
                // (name-only set) — it is a distinct action (use_warp
                // flag differs); Python's dedup key is the 6-tuple
                // (name, use_warp, ...). For 2x the same warp printing
                // Python yields {base, warp} once and Rust's name-only
                // set yields {base?} once + this warp push once = the
                // same pair. The warp life gate + the new
                // `can_pay_cost(warp_cost)` gate are folded into
                // `warp_payable` above (mirrors Python's combined
                // `_can_cast(warp_cost) && p.life > rider`).
                actions.push(cast_spell_warp_dict(py, obj.id)?);
            }
        }

        // Post-instant-speed sorcery-timing gate. Everything after
        // this point (PlayLand, sorcery-speed CastSpell,
        // ActivateAbility) is sorcery-speed only, so if it isn't
        // sorcery timing we return here with whatever
        // PassPriority + DeclareAttackers / DeclareBlockers +
        // instant-speed CastSpell emissions we've already pushed.
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

        // **CastSpell emission — sorcery-speed pass** (Python
        // `legal.py:335-445`). Walk the hand and emit a CastSpell
        // for every non-Land card that isn't already in the
        // `seen_castspell` HashSet from the instant-speed pass
        // above. The shared HashSet means an instant or flash card
        // emitted instant-speed is NOT re-emitted here (matches
        // Python's sorcery-speed loop short-circuit at line 366:
        // `if "flash" in card.printing.keywords: continue` plus the
        // `_is_sorcery_speed_spell` filter at lines 339-345 that
        // excludes Instants).
        //
        // **Filter** (Python lines 339-345, narrowed): non-Land
        // cards that aren't Instant and don't have flash. Python
        // gates `_is_sorcery_speed_spell` on
        // `is_creature || is_sorcery || (is_permanent && !is_instant
        // && !is_land)`. The complement of (Instant ∪ Flash ∪ Land)
        // covers Creatures, Sorceries, and other non-Land non-
        // Instant permanents.
        //
        // **Dedup** uses the SHARED `seen_castspell` HashSet —
        // mirrors Python's `seen_instant` set living in scope
        // across both blocks. Two Llanowar Elves in hand emit a
        // single CastSpell carrying the first hand-order Elves' id.
        for obj in self.players[idx].hand.iter() {
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            let is_land = printing.card_types.iter().any(|t| t == "Land");
            if is_land {
                continue;
            }
            // Instants + flash were already eligible at the instant-
            // speed pass; the shared HashSet would dedupe them, but
            // we still skip explicitly so the per-card filter
            // mirrors Python's `_is_sorcery_speed_spell` (which
            // excludes Instants and flash-keyword cards from this
            // loop's filter pre-dedup) one-to-one.
            let is_instant = printing
                .card_types
                .iter()
                .any(|t| t == "Instant");
            let has_flash = printing.keywords.iter().any(|k| k == "flash");
            if is_instant || has_flash {
                continue;
            }
            let name: &str = printing.name.as_ref();
            if seen_castspell.contains(name) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 1** — same
            // affordability-first + dedup-only-on-emit ordering as
            // the instant-speed pass above, mirroring Python's
            // sorcery-speed `_emit_hand_cast`-inside-`if _can_cast`
            // (`legal.py:382-401`). `eff(c)=c` cost-reduction seam;
            // warp gated INDEPENDENTLY on `warp_cost` + the preserved
            // life gate.
            let sk = crate::state_ops_mana::SpellKind::Any;
            // **legal_decisions mana-awareness Slice 2 —
            // target-satisfiability prune (CR 601.2c).** Same
            // computed-once + folded-into-both-`*_payable` shape as
            // the instant-speed pass above (mirrors Python's
            // sorcery-speed `_can_cast(...) and
            // _targets_are_satisfiable(...)` at `legal.py:382-385`
            // base / `:394-397` warp). Slice 3 adds the
            // `_additional_costs_payable` conjunct (`legal.py:385`
            // base / `:398` warp) into the SAME `nonmana_ok`
            // precondition — parity-NEUTRAL on the live surface (the
            // 4 EoE additional-cost cards are bridge-DECLINED). The
            // Slice-1 `seen_castspell` add-on-emit ordering preserved.
            let nonmana_ok =
                spell_targets_satisfiable(self, obj.printing_id, idx)
                    && additional_costs_payable(
                        self,
                        obj.printing_id,
                        idx,
                    );
            let base_payable = nonmana_ok
                && crate::state_ops_mana::can_pay_cost(
                    self,
                    idx,
                    eff(&printing.mana_cost),
                    &sk,
                );
            let warp_payable = match printing.warp_cost.as_ref() {
                Some(wc) => {
                    nonmana_ok
                        && (self.players[idx].life as i64)
                            > (printing.warp_additional_life_cost
                                as i64)
                        && crate::state_ops_mana::can_pay_cost(
                            self, idx, eff(wc), &sk,
                        )
                }
                None => false,
            };
            if !base_payable && !warp_payable {
                // Nothing affordable/targetable — skip WITHOUT
                // claiming the name-dedup slot (the §3.5.2 ordering
                // rule; mirrors Python's add-on-emit `seen_instant`).
                continue;
            }
            seen_castspell.insert(name);
            if base_payable {
                actions.push(cast_spell_dict(py, obj.id)?);
            }
            if warp_payable {
                // Phase-1 warp (sorcery-speed): mirror Python
                // legal.py:392-401. The LIVE path for the Phase-1
                // Artifact-Creature warp cards (Red Tiger Mechan,
                // Bygone Colossus) + Timeline Culler. Emission order:
                // base then warp immediately after, matching Python's
                // `:386` then `:401` positionally. The life gate + the
                // new `can_pay_cost(warp_cost)` gate are folded into
                // `warp_payable` (mirrors Python's combined predicate).
                actions.push(cast_spell_warp_dict(py, obj.id)?);
            }
        }

        // **Phase-2 warp-cast-from-graveyard (Timeline Culler).**
        // Mirrors Python's SEPARATE graveyard loop
        // (`legal.py:447-476`) — placed AFTER the hand sorcery loop
        // and BEFORE the exile-recast loop so the emission order
        // matches Python positionally (hand casts, then graveyard
        // warp-casts, then exile recasts). Sorcery-speed only (all
        // such cards in the pool are creatures; Python guards the
        // loop with `if card.printing.is_instant: continue` + the
        // same "currently only sorcery-speed" comment). Emits exactly
        // one `use_warp=true` CastSpell per graveyard object that is
        // `warp_castable_from_graveyard` && `warp_cost.is_some()` &&
        // !Instant && `life > warp_additional_life_cost`. NO base /
        // free-cast variant from the graveyard (CR 117.9b — the
        // warp-from-graveyard permission is itself the alternative
        // cost; see the `legal.py:464-476` comment block + the
        // test_weftwalking_and_possibility regression). The graveyard
        // loop is independent of `seen_castspell` — Python's
        // graveyard loop has no dedup either (a graveyard object is a
        // distinct castable from any hand copy).
        for obj in self.players[idx].graveyard.iter() {
            let Some(printing) = self.registry.lookup(obj.printing_id)
            else {
                continue;
            };
            if !printing.warp_castable_from_graveyard {
                continue;
            }
            if printing.warp_cost.is_none() {
                continue;
            }
            if printing.card_types.iter().any(|t| t == "Instant") {
                continue;
            }
            if (self.players[idx].life as i64)
                <= (printing.warp_additional_life_cost as i64)
            {
                continue;
            }
            // **legal_decisions mana-awareness Slice 1.** Mirror
            // Python's graveyard-warp `_can_cast(warp_cost)` gate
            // (`legal.py:457-462`). The `warp_cost.is_none()` skip
            // above guarantees `warp_cost` is Some here.
            let sk = crate::state_ops_mana::SpellKind::Any;
            let wc = printing
                .warp_cost
                .as_ref()
                .expect("warp_cost checked Some by the is_none skip");
            if !crate::state_ops_mana::can_pay_cost(
                self, idx, eff(wc), &sk,
            ) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 2 —
            // target-satisfiability prune (CR 601.2c).** Python's
            // graveyard-warp loop also gates on
            // `_targets_are_satisfiable` (`legal.py:459`). Every
            // graveyard-warp-castable card in the live pool is a
            // creature (Timeline Culler — no `spell_effect`), so this
            // returns `true` trivially; adding it is the faithful
            // Python mirror + the single named extension point if a
            // targeted graveyard-warp card is ever admitted.
            if !spell_targets_satisfiable(self, obj.printing_id, idx) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 3 —
            // additional-cost prune (CR 601.2f/601.2h).** Python's
            // graveyard-warp loop also gates on
            // `_additional_costs_payable` (`legal.py:460`). No
            // graveyard-warp-castable card carries an additional cost
            // (the 4 EoE additional-cost cards are bridge-DECLINED) so
            // this returns `true` trivially; the faithful Python
            // mirror + the single named extension point.
            if !additional_costs_payable(self, obj.printing_id, idx) {
                continue;
            }
            actions.push(cast_spell_warp_dict(py, obj.id)?);
        }

        // Phase-1 warp exile-recast (round-63): a warp-exiled
        // permanent (warped=true) in exile is re-castable at its
        // NORMAL mana_cost on a later turn. Mirrors Python
        // legal.py:478-489. Sorcery-speed only (both Phase-1 cards
        // are creatures; an instant in warp-exile would recast at
        // instant speed but none exist in the pool). Emits the BASE
        // cast_spell_dict (use_warp=false) — the recast pays
        // mana_cost, not warp_cost. The `cast_creature` primitive's
        // exile branch (state_ops_cast.rs) does the actual pay +
        // clears `warped`.
        for obj in self.players[idx].exile.iter() {
            if !obj.warped {
                continue;
            }
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            if printing.card_types.iter().any(|t| t == "Instant") {
                continue;
            }
            // **legal_decisions mana-awareness Slice 1.** The warp
            // exile-recast pays the NORMAL `mana_cost` (not
            // `warp_cost`). Mirror Python's `_can_cast(mana_cost)`
            // gate on the exile-recast (`legal.py:486-489`).
            let sk = crate::state_ops_mana::SpellKind::Any;
            if !crate::state_ops_mana::can_pay_cost(
                self,
                idx,
                eff(&printing.mana_cost),
                &sk,
            ) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 2 —
            // target-satisfiability prune (CR 601.2c).** Python's
            // exile-recast loop also gates on
            // `_targets_are_satisfiable` (`legal.py:488`). Both
            // Phase-1 warp-exile cards are creatures (no
            // `spell_effect`) so this returns `true` trivially; the
            // faithful Python mirror + the single named extension
            // point.
            if !spell_targets_satisfiable(self, obj.printing_id, idx) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 3 —
            // additional-cost prune (CR 601.2f/601.2h).** Python's
            // exile-recast loop also gates on
            // `_additional_costs_payable` (`legal.py:488`). Both
            // Phase-1 warp-exile cards carry no additional cost (the
            // 4 EoE additional-cost cards are bridge-DECLINED) so this
            // returns `true` trivially; the faithful Python mirror +
            // the single named extension point.
            if !additional_costs_payable(self, obj.printing_id, idx) {
                continue;
            }
            actions.push(cast_spell_dict(py, obj.id)?);
        }

        // **Cast-from-exile-with-cost-hike substrate (Lightstall
        // Inquisitor residue closure).** A card sitting in exile with
        // `cast_from_exile_with_lightstall_surcharge = true` is
        // castable by the card's controller (per CR 400.7 the card's
        // controller was reset to its owner at exile time, and the
        // permission "they may play that card" -- "they" =
        // owner-of-exiled-card -- means the priority player can cast
        // it iff the card lives in their own exile zone). Pays the
        // printed mana cost + {1} generic surcharge. Sorcery timing
        // only on the LIVE surface (Lightstall exiles a chosen-by-
        // the-opponent hand card and there are no instant printings
        // bridge-admitted with this shape -- mirrors Python's oracle
        // text reading at `legal.py:536-543`: "could cast it" gated
        // at sorcery timing for the model). Lands played this way
        // ETB tapped -- enforced by the cast pay path
        // (`state_ops_cast.rs::play_land`), legal emitter just
        // emits `play_land_dict`. Mirrors Python
        // `legal.py:544-574`.
        //
        // Dedup is NOT performed across the exile-cast loop -- two
        // distinct exiled cards of the same printing are independent
        // castables (each emits its own action; the `obj.id` keys it).
        // This mirrors Python's `legal.py:544-574` per-card loop
        // (no `seen_castspell` insertion).
        for obj in self.players[idx].exile.iter() {
            if !obj.cast_from_exile_with_lightstall_surcharge {
                continue;
            }
            // The warp + Lightstall flags are mutually exclusive on
            // the live surface (warp-exile path doesn't stamp the
            // Lightstall flag; the Lightstall resolve arm doesn't
            // stamp `warped`), but guard against double-emission
            // anyway since the warp exile-recast loop above already
            // emits the base cast_spell_dict for `warped` objects.
            // Mirrors Python `legal.py:547-548` (`if card.warped:
            // continue`).
            if obj.warped {
                continue;
            }
            let Some(printing) = self.registry.lookup(obj.printing_id)
            else {
                continue;
            };
            let is_land = printing
                .card_types
                .iter()
                .any(|t| t == "Land");
            if is_land {
                // Land path: a Land played this way ETBs tapped (the
                // tapped-entry rider). PlayLand has no mana cost --
                // the only gate at the legal emitter is sorcery
                // timing, which the outer `is_sorcery_timing` block
                // already enforces. No per-turn-land-count gate
                // (mirrors Rust's existing PlayLand divergence -- see
                // module docstring; Python gates here, Rust does
                // not). Emit `play_land_dict` keyed on `obj.id`; the
                // cast pay path (`state_ops_cast.rs::play_land`)
                // honors the exile origin via the flag (see
                // `play_land` exile branch).
                actions.push(play_land_dict(py, obj.id)?);
                continue;
            }
            // Non-land: pay `printing.mana_cost + {1}` generic
            // surcharge. Sorcery timing only -- the outer block
            // already enforces this. Mirror Python's `legal.py:561-
            // 574` (`base_with_surcharge = ManaCost(generic + 1,
            // colored, x_pips)` then the `_can_cast / _targets_are_
            // satisfiable / _additional_costs_payable` triad). The
            // surcharge increments ONLY the `generic` slot (the {1}
            // is colorless); colored pip slots are unchanged.
            let sk = crate::state_ops_mana::SpellKind::Any;
            let base_cost = eff(&printing.mana_cost);
            // X-value subsystem Cycle 2: `x_pips` template is
            // preserved across the +{1} Lightstall surcharge (CR
            // 601.2f). Today's `can_pay_cost` ignores `x_pips`
            // (X-cost cards stay admit-declined); when Cycle 2.5+
            // wires the cast-pipeline X detector the predicate will
            // fold `x_pips * x_value` into generic before checking.
            let surcharged_cost = crate::cards::ManaCost {
                generic: base_cost.generic.saturating_add(1),
                white: base_cost.white,
                blue: base_cost.blue,
                black: base_cost.black,
                red: base_cost.red,
                green: base_cost.green,
                x_pips: base_cost.x_pips,
            };
            if !crate::state_ops_mana::can_pay_cost(
                self, idx, &surcharged_cost, &sk,
            ) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 2 -- target-
            // satisfiability prune (CR 601.2c).** Mirror Python's
            // `_targets_are_satisfiable` gate at `legal.py:571`. The
            // live surface for Lightstall-exiled cards is sourced
            // from each opponent's hand -- the bridge-admitted hand
            // is the same `_printing_is_bridge_simple_*` surface, so
            // every castable target-bearing spell goes through the
            // same satisfiability check as a normal hand-cast.
            if !spell_targets_satisfiable(self, obj.printing_id, idx) {
                continue;
            }
            // **legal_decisions mana-awareness Slice 3 -- additional-
            // cost prune (CR 601.2f/601.2h).** Mirror Python's
            // `_additional_costs_payable` gate at `legal.py:572`. No
            // bridge-admitted card carries both a Lightstall-eligible
            // exile origin AND an additional cost (the 4 EoE
            // additional-cost cards are bridge-DECLINED), so this
            // returns `true` trivially on the live surface; the
            // faithful Python mirror + the single named extension
            // point.
            if !additional_costs_payable(self, obj.printing_id, idx) {
                continue;
            }
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
            let Some(printing) = self.registry.lookup(obj.printing_id) else {
                continue;
            };
            let is_creature = printing.card_types.iter().any(|t| t == "Creature");
            // CE20 Slice 5 (Walker loyalty Slice C): CR 606.3 —
            // "a player may activate a loyalty ability of a
            // planeswalker they control only if no player has
            // activated a loyalty ability of that planeswalker this
            // turn." Per-walker (not per-ability): if the walker has
            // already had ANY loyalty ability fire this turn, skip
            // ALL of its activated abilities for the remainder of
            // the turn. Gate fires BEFORE the per-ability loop so
            // none of the walker's abilities emit.
            //
            // Today every walker's `activated_abilities` are loyalty
            // abilities by construction (the only walker substrate
            // landed is Tezzeret-shape, where each ability carries a
            // `cost.loyalty` value). A future walker printing with
            // both loyalty + non-loyalty activated abilities (e.g.
            // The Wanderer's "0: prevent damage" reads as a loyalty
            // ability, not a non-loyalty activated) would still want
            // the gate at this granularity per CR 606.3's wording.
            let is_walker = printing.card_types.iter().any(|t| t == "Planeswalker");
            if is_walker
                && self
                    .loyalty_activated_this_turn
                    .contains(&(idx as u8, obj.id))
            {
                continue;
            }
            // **Bridge follow-up (post-CE20-Slice-1).** The legacy
            // `mana_produced.is_some()` emission block that USED to
            // live here is gone — `parse_printing` now synthesizes a
            // single ActivatedAbility at `ability_index=0` for every
            // printing with `mana_produced.is_some()` (see
            // `cards.rs` bridge synthesis), so the walk below covers
            // both the new-substrate cost-paying abilities AND the
            // bridged mana-only ability uniformly. No more dual-
            // path complexity at emission.
            //
            // CR 302.1 (creature summoning sickness for tap-cost
            // abilities) is still enforced per-ability on the
            // `cost.tap_self` branch.
            // CE20 Slice 1 + auto-tap follow-up + bridge follow-up:
            // walk the activated_abilities Vec (which now includes
            // the synthesized mana-ability entry for any printing
            // that had legacy `mana_produced`). Per-ability:
            //   - tap_self abilities skip a tapped source AND a
            //     summoning-sick creature source (CR 302.1 narrows
            //     the sickness gate to abilities whose cost requires
            //     tapping the source — non-tap-cost abilities on a
            //     fresh creature still emit).
            //   - mana affordability: emission-side checks the
            //     controller's current pool PLUS auto-tap headroom
            //     from untapped non-sick mana sources on bf (via
            //     `auto_tap_headroom_pool`). Matches Python's
            //     `can_pay_cost` (`actions.py:336-403`), which
            //     composes pool + tappable-land coverage. The
            //     apply-time cost-pay path stays strict (the
            //     controller must have the mana already in pool to
            //     activate); only emission widens. Pins I50.
            let headroom_pool = auto_tap_headroom_pool(self, idx);
            for (ability_index, ability) in
                printing.activated_abilities.iter().enumerate()
            {
                // CE20 follow-up Slice 3 (Spacecraft station-charge):
                // tier_threshold gates usability. The source must
                // carry at least `tier_threshold` charge counters
                // for the ability to be activatable. Defaults to 0
                // (no gate) for every non-Spacecraft printing —
                // back-compat for every pre-Slice-3 caller.
                //
                // Ordering: tier gate fires FIRST (BEFORE tap_self /
                // mana affordability) — a source below tier is
                // outright inert; checking its mana / tap state is
                // wasted work and could leak observable signal.
                if ability.tier_threshold > 0
                    && obj.charge_counters < ability.tier_threshold
                {
                    continue;
                }
                // CE20 follow-up Slice 5 (Walker loyalty Slice A):
                // positive `cost.loyalty` is a pay-loyalty cost
                // (CR 606). The walker must have at least
                // `cost.loyalty` counters to activate. Negative
                // values are add-loyalty (always affordable). Zero
                // is no gate. Mirrors the tier-gate slot: fires
                // BEFORE tap_self / mana affordability — a walker
                // that can't afford its -N ability is outright
                // unable, regardless of mana state.
                if ability.cost.loyalty > 0
                    && obj.loyalty_counters < ability.cost.loyalty
                {
                    continue;
                }
                if ability.cost.tap_self {
                    if obj.tapped {
                        continue;
                    }
                    if is_creature && obj.summoning_sick {
                        continue;
                    }
                }
                if !pool_can_cover(&headroom_pool, &ability.cost.mana) {
                    continue;
                }
                // CE20 follow-up Slice 4 (Equipment-equip): when the
                // cost requires a target creature you control, fan
                // out one ActivateAbility per legal target. The
                // target is any creature on the activator's
                // battlefield that ISN'T the source itself (CR
                // 702.6c — an Equipment can't equip itself). Target's
                // summoning sickness is NOT a barrier — equip is a
                // mana-only cost so the target's tap-state is
                // irrelevant.
                //
                // **No legal targets → no emission.** A bare
                // Equipment on bf with no other creatures emits
                // zero ActivateAbility dicts (the activation has
                // nothing to target; CR 601.2c says a player can't
                // begin to cast a spell or activate an ability
                // unless they choose legal targets for it).
                //
                // **CE20 Slice 6 (targeted-sacrifice variants).** When
                // the cost flags a `sacrifice_*` target shape, fan out
                // one ActivateAbility per legal sac target. The two
                // target axes (target_creature_id for equip,
                // sacrifice_target_id for sac) are independent in the
                // action-dict shape but the EoE pool has no printing
                // that combines both, so emission walks each shape
                // separately and the cross-product fan-out is deferred.
                if ability.cost.requires_target_creature_you_control {
                    for target in self.players[idx].battlefield.iter() {
                        if target.id == obj.id {
                            // Can't target self.
                            continue;
                        }
                        let Some(t_printing) =
                            self.registry.lookup(target.printing_id)
                        else {
                            continue;
                        };
                        if !t_printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature")
                        {
                            continue;
                        }
                        actions.push(activate_ability_dict(
                            py,
                            obj.id,
                            ability_index as u32,
                            Some(target.id),
                            None,
                        )?);
                    }
                } else if ability.cost.sacrifice_another_creature_or_artifact
                    || ability.cost.sacrifice_a_land
                    || ability.cost.sacrifice_another_artifact
                    || ability.cost.sacrifice_food
                    || ability.cost.sacrifice_a_desert
                {
                    // CE20 Slice 6 + Slice 7 sacrifice target fan-out.
                    // Eligibility varies per variant:
                    //
                    // - sacrifice_another_creature_or_artifact: target
                    //   has card_type Creature OR Artifact, NOT self.
                    // - sacrifice_a_land: target has card_type Land,
                    //   SELF IS eligible (printed text reads "a land",
                    //   no "another" qualifier).
                    // - sacrifice_another_artifact: target has
                    //   card_type Artifact, NOT self.
                    // - sacrifice_food: target's subtypes contains
                    //   "Food", SELF IS eligible (no "another"
                    //   qualifier; matches `sacrifice_a_land` reading).
                    //   Rides on the CardPrinting.subtypes substrate
                    //   landed in commits `03aebefb` / `dd15d7ed`.
                    // - sacrifice_a_desert: target's subtypes contains
                    //   "Desert", SELF IS eligible (same lenient
                    //   reading as `sacrifice_food`).
                    //
                    // Walk the activator's bf once and emit one dict
                    // per eligible target. Empty result → no emission
                    // (CR 601.2c: can't begin to activate without a
                    // legal target). The variants are semantically
                    // mutually exclusive on a single ability per the
                    // Python dataclass shape; if a future printing
                    // carries multiple flags Rust's else-if cascade
                    // honors the first set flag (priority order:
                    // creature_or_artifact > land > artifact > food >
                    // desert) — no real EoE printing combines flags.
                    for target in self.players[idx].battlefield.iter() {
                        let is_self = target.id == obj.id;
                        let Some(t_printing) =
                            self.registry.lookup(target.printing_id)
                        else {
                            continue;
                        };
                        let is_creature = t_printing
                            .card_types
                            .iter()
                            .any(|t| t == "Creature");
                        let is_artifact = t_printing
                            .card_types
                            .iter()
                            .any(|t| t == "Artifact");
                        let is_land = t_printing
                            .card_types
                            .iter()
                            .any(|t| t == "Land");
                        // Subtype-filtered checks (Slice 7).
                        // Case-sensitive Title Case match per
                        // CR/Scryfall convention.
                        let is_food = t_printing
                            .subtypes
                            .iter()
                            .any(|s| s == "Food");
                        let is_desert = t_printing
                            .subtypes
                            .iter()
                            .any(|s| s == "Desert");
                        let eligible = if ability
                            .cost
                            .sacrifice_another_creature_or_artifact
                        {
                            !is_self && (is_creature || is_artifact)
                        } else if ability.cost.sacrifice_a_land {
                            // "a land" includes self when self is a
                            // land (no "another" qualifier).
                            is_land
                        } else if ability.cost.sacrifice_another_artifact {
                            !is_self && is_artifact
                        } else if ability.cost.sacrifice_food {
                            // "a Food" — no "another"; self IS
                            // eligible iff its own subtypes contain
                            // "Food".
                            is_food
                        } else if ability.cost.sacrifice_a_desert {
                            // "a Desert" — same lenient reading.
                            is_desert
                        } else {
                            false
                        };
                        if !eligible {
                            continue;
                        }
                        actions.push(activate_ability_dict(
                            py,
                            obj.id,
                            ability_index as u32,
                            None,
                            Some(target.id),
                        )?);
                    }
                } else {
                    actions.push(activate_ability_dict(
                        py,
                        obj.id,
                        ability_index as u32,
                        None,
                        None,
                    )?);
                }
            }
        }

        // Final emission order: [PassPriority, (DeclareAttackers...),
        // (DeclareBlockers...), CastSpell(instant/flash)...,
        // PlayLand..., CastSpell(sorcery)..., ActivateAbility...].
        // Matches Python's `legal.py` natural append order:
        // instant-speed CastSpell block (lines 217-301), sorcery-
        // timing gate (line 303), PlayLand (lines 319-329), sorcery-
        // speed CastSpell block (lines 335-445), ActivateAbility
        // (lines 612-829). DeclareAttackers / DeclareBlockers are
        // Rust-only emissions outside Python's loop (see their
        // module-docstring sections). CE19 closure shape; see I46.

        Ok(actions)
    }
}
