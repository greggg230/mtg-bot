//! `fingerprint() -> str` pymethod on `GameState`.
//!
//! Round-22 phase-2b slice — implementation behind the stub the
//! foundation laid down. The pymethod produces a 12-char lowercase
//! hex digest computed identically to what
//! `scripts/parity_smoke.py:hash_state` produces for an equivalent
//! Python `GameState`. Two states (one Python, one Rust) that share
//! the same gameplay-relevant invariants produce the same digest.
//!
//! ## Hash recipe (mirror of `scripts/parity_smoke.py:hash_state`)
//!
//! ```text
//! parts = [
//!     turn_number,                         # int
//!     step,                                # str (Step.<NAME>) or None
//!     active_player_idx,                   # int
//!     priority_player_idx,                 # int or None
//!     is_over,                             # bool
//!     is_draw,                             # bool
//!     winner_id,                           # int or None
//!     (p0_id, life, has_lost, hand_sig, library_sig, graveyard_sig,
//!      exile_sig, mana_pool_sorted_items),
//!     (p1_id, ...),                        # same shape
//!     battlefield_signature,
//!     stack_signature,
//! ]
//! blob = json.dumps(parts, default=repr, sort_keys=False).encode("utf-8")
//! return hashlib.sha1(blob).hexdigest()[:12]
//! ```
//!
//! Each zone signature is a tuple of per-object 8-tuples
//! `(printing_name | None, int(tapped), int(summoning_sick),
//!   damage, counters, attached_to, controller_id, id)`.
//!
//! `mana_pool_sorted_items` mirrors the Python expression
//! `tuple(sorted({"W": w, "U": u, ..., "C": c}.items()))` — a
//! list of `[key, count]` 2-arrays sorted lexicographically by key.
//!
//! ## JSON-format gotchas
//!
//! Python `json.dumps(..., default=repr, sort_keys=False)` has a
//! specific output shape we have to match exactly to land the same
//! sha1 digest:
//!
//! * Python tuples and lists both serialize as JSON arrays — there
//!   are no JSON tuples. Rust mirrors by emitting `serde_json::Value::Array`
//!   for both.
//! * Booleans: Python emits `true` / `false`; `serde_json` matches.
//! * `None`: Python emits `null`; `serde_json::Value::Null` matches.
//! * Integers: bare digits both sides.
//! * Strings: JSON-quoted with backslash-escapes; `serde_json`'s
//!   default escaping is bytewise identical to Python's `json` for
//!   ASCII strings (which is all we emit — printing names come from
//!   the registry, which we control).
//! * Whitespace: Python's default (no spaces around `:` and `,`)
//!   matches `serde_json::to_string` (compact). Python by default
//!   uses `", "` / `": "` separators — BUT `json.dumps` without
//!   `indent=` uses `(", ", ": ")` separators which DOES emit a
//!   single space after both. `serde_json::to_string` emits no
//!   spaces. We override below by calling a compact serializer
//!   that matches Python's default-separator output, OR equivalently
//!   we manually emit a string. We chose: build the structure and
//!   serialize with a small custom writer that inserts the same
//!   spaces Python does. See `serialize_python_compatible`.
//! * `sort_keys=False`: we never emit JSON objects (only arrays),
//!   so this is moot for our output.
//!
//! ## Battlefield ordering (CE4 closure)
//!
//! The battlefield slot is emitted in `etb_timestamp` order via a
//! stable sort over the union of both players' battlefield Vecs (see
//! `battlefield_snapshot` in `state_ops_snapshot.rs` for the matching
//! per-object dict variant). The per-object 8-tuple shape does NOT
//! include `etb_timestamp` — that field is bookkeeping only, surfaced
//! through the sort key. Synthetic-state baselines (every bf object
//! at timestamp 0) keep the pre-CE4 p0-then-p1 walk order via stable
//! sort ties, so pinned digests remain bit-equal. Real ETB-driven
//! flows produce strictly-monotonic timestamps and emit in
//! Python-equivalent append-order.
//!
//! ## Cross-engine parity scope
//!
//! Full byte-equality between the Rust fingerprint and Python's
//! `hash_state` on an equivalent state requires constructing the
//! Python state with matching shape:
//!
//! * Python's `state.mana_pool` is a `list[ManaCoin]` in
//!   `src/mtg_bot/game.py`, NOT a dict — so Python's hash_state
//!   `isinstance(..., dict)` check fails and the list flows through
//!   `default=repr`, producing a string like
//!   `"[ManaCoin(color='W', ...)]"`. Rust treats mana_pool as a
//!   dict (5 colors + colorless) and emits sorted items. To get
//!   cross-engine parity, the Python side would need a shimmed
//!   GameState with `mana_pool` as the equivalent dict.
//! * Python objects in zones have a `printing` attribute with `.name`;
//!   we look the same up through the registry. For object id alone
//!   (no printing), Python emits `None`; we match.
//! * Python's `attached_to` defaults to `None` (`getattr(..., None)`);
//!   Rust matches: `obj.attached_to` is `Option<u32>`, emitted as
//!   `null` when None / the u32 id when Some. Default-None preserves
//!   byte-equality with every pre-r28 pinned baseline.
//!
//! Round-22 doesn't ship a Python-side shim; the cross-engine parity
//! test in `tests_rust/test_fingerprint.py` asserts equality against
//! a hard-coded baseline string computed from the same algorithm,
//! locking the format in a regression-safe way. A future round can
//! add the Python shim and assert byte-equality across the engines.

use crate::state::{GameObject, GameState, PendingDecision};
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha1::{Digest, Sha1};

/// Phase u8 -> Python `Step.<NAME>` string. Order mirrors the
/// `STEP_ORDER` tuple in `src/mtg_bot/turn.py`. Returns `None` for
/// phase values outside `[0, 11]` (defensive — `set_phase` already
/// rejects out-of-range writes; the `None` here lets a corrupted
/// state still produce *some* fingerprint instead of panicking).
fn phase_to_step_name(phase: u8) -> Option<&'static str> {
    match phase {
        0 => Some("UNTAP"),
        1 => Some("UPKEEP"),
        2 => Some("DRAW"),
        3 => Some("PRECOMBAT_MAIN"),
        4 => Some("BEGINNING_OF_COMBAT"),
        5 => Some("DECLARE_ATTACKERS"),
        6 => Some("DECLARE_BLOCKERS"),
        7 => Some("COMBAT_DAMAGE"),
        8 => Some("END_OF_COMBAT"),
        9 => Some("POSTCOMBAT_MAIN"),
        10 => Some("END"),
        11 => Some("CLEANUP"),
        _ => None,
    }
}

/// Project one GameObject to the 8-element JSON array mirror of
/// `_zone_signature`. Field order is the Python tuple order:
/// `(printing_name | None, int(tapped), int(summoning_sick),
///   damage, counters, attached_to, controller_id, id)`.
///
/// `int(tapped)` / `int(summoning_sick)` — Python casts to int (0/1).
/// We follow.
fn object_signature(state: &GameState, obj: &GameObject) -> Value {
    let name: Value = match state.registry.lookup(obj.printing_id) {
        Some(p) => Value::String(p.name.to_string()),
        None => Value::Null,
    };
    // Tapped + summoning_sick get `int(...)` cast on the Python side.
    // Round-28 attach foundation: emit obj.attached_to as either Null
    // (None — the universal default for every freshly-allocated /
    // never-attached object) or the target id (Some(id)). The slot
    // previously hardcoded Null; default-None preserves byte-equality
    // with every pre-r28 pinned fingerprint baseline.
    let attached_to: Value = match obj.attached_to {
        Some(id) => Value::from(id),
        None => Value::Null,
    };
    Value::Array(vec![
        name,
        json!(obj.tapped as i32),
        json!(obj.summoning_sick as i32),
        json!(obj.damage),
        json!(obj.counters),
        attached_to,
        json!(obj.controller),
        json!(obj.id),
    ])
}

/// Project a zone (slice of GameObjects) to the JSON-array form of
/// `_zone_signature`'s `tuple(per-object 8-tuples)`. Order-sensitive.
fn zone_signature(state: &GameState, zone: &[GameObject]) -> Value {
    Value::Array(zone.iter().map(|o| object_signature(state, o)).collect())
}

/// Player tuple mirror — 8 fields, all populated even when
/// the corresponding Rust field doesn't exist yet (e.g. `has_lost`
/// is derived from `life <= 0` until the explicit-loss field lands).
///
/// `mana_pool_sorted_items` mirrors Python's
/// `tuple(sorted({"W": w, "U": u, ..., "C": c}.items()))` — a JSON
/// array of `[key, count]` 2-arrays, keys lex-sorted (B, C, G, R, U, W).
fn player_signature(state: &GameState, idx: usize) -> Value {
    let p = &state.players[idx];
    let pool = &p.mana_pool;
    // Python `tuple(sorted({"W":w, "U":u, ...}.items()))` keys lex-sorted.
    // Lex order: B, C, G, R, U, W.
    let mana_items: Vec<Value> = vec![
        Value::Array(vec![json!("B"), json!(pool.black)]),
        Value::Array(vec![json!("C"), json!(pool.colorless)]),
        Value::Array(vec![json!("G"), json!(pool.green)]),
        Value::Array(vec![json!("R"), json!(pool.red)]),
        Value::Array(vec![json!("U"), json!(pool.blue)]),
        Value::Array(vec![json!("W"), json!(pool.white)]),
    ];
    Value::Array(vec![
        json!(p.id),
        json!(p.life),
        // has_lost: round-26 G11 closure — the explicit sticky loss
        // flag now lives on `Player`. Pinned baselines were computed
        // when the slot was the derived `life <= 0`; for every state
        // used in those baselines the player is alive (life=20), so
        // `has_lost == false == (life <= 0 is false)` and the digest
        // is identical. New flows that drop `has_lost = true` without
        // dropping life shift the digest deliberately (mirrors Python).
        json!(p.has_lost),
        zone_signature(state, &p.hand),
        zone_signature(state, &p.library),
        zone_signature(state, &p.graveyard),
        zone_signature(state, &p.exile),
        Value::Array(mana_items),
    ])
}

/// Serialize a `serde_json::Value` with the same separator convention
/// as Python's default `json.dumps(...)` — `", "` and `": "` (a space
/// after every comma and colon). `serde_json::to_string` is compact
/// (no spaces); we want the spaces so the resulting blob matches the
/// Python side byte-for-byte before we hash.
///
/// We only ever emit arrays and primitives (no objects), so the colon
/// rule never fires. Implementation: walk the Value and emit each
/// piece.
fn serialize_python_compatible(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            // Use serde_json's string escaping for correctness — it
            // matches Python's json default ASCII-escape behavior for
            // the printable ASCII names we emit.
            let escaped = serde_json::to_string(s).unwrap();
            out.push_str(&escaped);
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                serialize_python_compatible(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Defensive — we never emit objects, but if we ever do,
            // mirror Python's `", "` / `": "` separators with
            // insertion-order (sort_keys=False).
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let key_escaped = serde_json::to_string(k).unwrap();
                out.push_str(&key_escaped);
                out.push_str(": ");
                serialize_python_compatible(v, out);
            }
            out.push('}');
        }
    }
}

#[pymethods]
impl GameState {
    /// 12-char lowercase hex SHA1 digest of the gameplay-relevant
    /// state. Mirrors `scripts/parity_smoke.py:hash_state` — same
    /// `parts` shape, same JSON serialization, same SHA1 truncation.
    ///
    /// Determinism: same state -> same fingerprint, always. Clones
    /// share a fingerprint with their source until the source mutates.
    /// Mutating a clone after cloning produces a different fingerprint
    /// without affecting the source's.
    ///
    /// See the module docstring for the full hash recipe + the
    /// cross-engine parity scope (the Python side's mana_pool is a
    /// list-of-ManaCoin, not a dict, so a Python-side shim is
    /// needed for cross-engine byte-equality — round 22 ships only
    /// the Rust side; the regression-safe baseline test asserts
    /// against a hard-coded digest computed from a known synthetic
    /// state).
    fn fingerprint(&self) -> String {
        // Build `parts` mirroring Python `hash_state`.
        let step: Value = phase_to_step_name(self.phase)
            .map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);
        let mut parts_vec: Vec<Value> = vec![
            json!(self.turn_number),
            step,
            json!(self.active_player),
            json!(self.priority_player),
            // Inlined `is_over` — sibling module's `is_game_over` /
            // `is_over` pymethods are private to their module (E0624),
            // so we duplicate the trivial life<=0 check here.
            json!(self.players[0].life <= 0 || self.players[1].life <= 0),
            json!(self.players[0].life <= 0 && self.players[1].life <= 0),
            match (self.players[0].life <= 0, self.players[1].life <= 0) {
                (false, true) => json!(self.players[0].id),
                (true, false) => json!(self.players[1].id),
                _ => Value::Null,
            },
            player_signature(self, 0),
            player_signature(self, 1),
            // Battlefield is centralized on Python's GameState; on
            // the Rust side it's split across players. CE4 closure:
            // collect refs from both Vecs, stable-sort by
            // `etb_timestamp` so the emitted order matches Python's
            // `state.battlefield` append-order. Synthetic states
            // (where every bf object has timestamp 0 because
            // `make_player` doesn't call `reserve_etb_timestamp`)
            // keep the pre-CE4 p0-then-p1 ordering via stable-sort
            // ties — pinned baselines stay bit-equal.
            {
                let mut refs: Vec<&GameObject> = Vec::with_capacity(
                    self.players[0].battlefield.len() + self.players[1].battlefield.len(),
                );
                for player in &self.players {
                    for obj in player.battlefield.iter() {
                        refs.push(obj);
                    }
                }
                refs.sort_by_key(|o| o.etb_timestamp);
                Value::Array(refs.iter().map(|o| object_signature(self, o)).collect())
            },
            // Stack: today's StackItem has just (source_obj_id,
            // controller). For fingerprint purposes we emit it in
            // the same 8-tuple shape as a zone object — most fields
            // are Null / 0 since a stack item isn't a permanent.
            // Python's stack is a list of cards-on-stack with the
            // same `_zone_signature` projection.
            Value::Array(
                self.stack
                    .iter()
                    .map(|si| {
                        Value::Array(vec![
                            Value::Null,           // printing name unknown for now
                            json!(0),              // tapped
                            json!(0),              // summoning_sick
                            json!(0),              // damage
                            json!(0),              // counters
                            Value::Null,           // attached_to
                            json!(si.controller),  // controller_id
                            json!(si.source_obj_id), // id
                        ])
                    })
                    .collect::<Vec<_>>(),
            ),
        ];
        // Round-27 G20 closure: pending_decisions slot. Appended
        // ONLY when non-empty so the pre-r27 pinned fingerprint
        // baselines (every state in test_fingerprint.py has empty
        // pending_decisions) remain bit-equal. A state mid-Surveil
        // — pending SurveilChoice waiting on the controller's pick
        // — produces a digest that differs from the pre-decision
        // baseline by the appended JSON array. Cross-engine parity:
        // Python's hash_state has no equivalent slot today; the
        // r27 invariant is "Rust emits this iff a decision is
        // pending", and Python states (which never carry pending
        // decisions through the parity_smoke harness) match the
        // empty-pending Rust path.
        if !self.pending_decisions.is_empty() {
            let pds: Vec<Value> = self
                .pending_decisions
                .iter()
                .map(|pd| match pd {
                    PendingDecision::SurveilChoice {
                        source_obj_id,
                        controller,
                        cards,
                    } => Value::Array(vec![
                        Value::String("SurveilChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            cards
                                .iter()
                                .map(|o| object_signature(self, o))
                                .collect(),
                        ),
                    ]),
                    // Round-28 MayDo decision: stable serialized form
                    // by variant-tag + source/controller. The inner
                    // effect is serialized by Debug-format because
                    // round-tripping serde_json for the Effect enum
                    // (closed but with Box<Effect> recursion) would
                    // require derived Serialize/Deserialize on every
                    // Effect arm — much wider scope than this slice.
                    // Debug formatting is deterministic across Rust
                    // releases for our enum shape and gives a stable
                    // fingerprint for parity-smoke runs that may
                    // pause mid-MayDo.
                    PendingDecision::MayDoChoice {
                        source_obj_id,
                        controller,
                        inner_effect,
                    } => Value::Array(vec![
                        Value::String("MayDoChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::String(format!("{:?}", inner_effect)),
                    ]),
                    // Round-28 pick-card-from-examined decision: stable
                    // serialized form by variant-tag + source/controller
                    // + per-examined-card object signature (same shape
                    // as SurveilChoice — the examined cards live in
                    // NEITHER library nor graveyard while pending, so
                    // they need to participate in the fingerprint to
                    // make a pending-card state distinguishable from
                    // a state where the cards already moved).
                    PendingDecision::PickCardFromExaminedChoice {
                        source_obj_id,
                        controller,
                        cards,
                    } => Value::Array(vec![
                        Value::String("PickCardFromExaminedChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            cards
                                .iter()
                                .map(|o| object_signature(self, o))
                                .collect(),
                        ),
                    ]),
                    // Round-28 choose-one-mode decision: stable
                    // serialized form by variant-tag + source/controller
                    // + each mode's Debug format. Same Debug-over-serde
                    // choice as MayDoChoice (avoids deriving Serialize
                    // on every Effect arm in this slice).
                    PendingDecision::ChooseModeChoice {
                        source_obj_id,
                        controller,
                        modes,
                    } => Value::Array(vec![
                        Value::String("ChooseModeChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            modes
                                .iter()
                                .map(|m| Value::String(format!("{:?}", m)))
                                .collect(),
                        ),
                    ]),
                    // Round-28 if-you-do decision: variant-tag +
                    // source/controller + each Effect's Debug format.
                    // Same Debug-over-Serialize choice as MayDoChoice
                    // and ChooseModeChoice.
                    PendingDecision::IfYouDoChoice {
                        source_obj_id,
                        controller,
                        do_action,
                        then_action,
                    } => Value::Array(vec![
                        Value::String("IfYouDoChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::String(format!("{:?}", do_action)),
                        Value::String(format!("{:?}", then_action)),
                    ]),
                    // Round-28 manifest-dread decision: same shape as
                    // PickCardFromExaminedChoice (variant-tag +
                    // source/controller + per-examined-card object
                    // signature). The examined cards live in NEITHER
                    // library nor bf/graveyard while pending, so they
                    // must participate in the fingerprint to make a
                    // pending-card state distinguishable from a state
                    // where the cards already routed.
                    PendingDecision::ManifestDreadChoice {
                        source_obj_id,
                        controller,
                        cards,
                        then_attach_to_source,
                        then_plus_one_counter_on_manifested,
                        counter_amount,
                    } => Value::Array(vec![
                        Value::String("ManifestDreadChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            cards
                                .iter()
                                .map(|o| object_signature(self, o))
                                .collect(),
                        ),
                        json!(*then_attach_to_source),
                        json!(*then_plus_one_counter_on_manifested),
                        json!(*counter_amount),
                    ]),
                    // Round-28 look-at-top-N-reveal-or-counter: same
                    // examined-cards-as-object-signatures shape as
                    // PickCardFromExaminedChoice / ManifestDreadChoice,
                    // plus the filter context (filter_name + max_power
                    // + counter_on_miss) so two pending decisions that
                    // share examined ids but differ in filter scope
                    // produce different digests.
                    PendingDecision::LookAtTopNRevealOrCounterChoice {
                        source_obj_id,
                        controller,
                        cards,
                        filter_name,
                        max_power,
                        counter_on_miss,
                    } => Value::Array(vec![
                        Value::String("LookAtTopNRevealOrCounterChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            cards
                                .iter()
                                .map(|o| object_signature(self, o))
                                .collect(),
                        ),
                        Value::String(filter_name.clone()),
                        json!(*max_power),
                        Value::Bool(*counter_on_miss),
                    ]),
                    // Round-29 optional-mana-cost decision: variant-tag +
                    // source/controller + Debug-format of both cost and
                    // inner_effect. Same Debug-over-Serialize choice as
                    // MayDoChoice / ChooseModeChoice / IfYouDoChoice
                    // (avoids deriving Serialize on the Effect enum
                    // recursion).
                    PendingDecision::OptionalManaCostChoice {
                        source_obj_id,
                        controller,
                        cost,
                        inner_effect,
                    } => Value::Array(vec![
                        Value::String("OptionalManaCostChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::String(format!("{:?}", cost)),
                        Value::String(format!("{:?}", inner_effect)),
                    ]),
                    // Round-30 G24 closure — Paris scry-on-keep. Same
                    // examined-cards-as-object-signatures shape as
                    // SurveilChoice / PickCardFromExaminedChoice — the
                    // pre-popped library top lives only in this variant
                    // until the controller picks keep_on_top or
                    // send_to_bottom, so the examined card must
                    // participate in the fingerprint to make a pending-
                    // scry state distinguishable from one where the
                    // scry already routed.
                    PendingDecision::ScryChoice {
                        source_obj_id,
                        controller,
                        cards,
                    } => Value::Array(vec![
                        Value::String("ScryChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            cards
                                .iter()
                                .map(|o| object_signature(self, o))
                                .collect(),
                        ),
                    ]),
                    // 20th Rust slice (VVV Pin 2 closure) — Command
                    // Bridge cost-decision fingerprint. The candidate
                    // ids fully determine the pending-state shape:
                    // re-resolving the decision with a different
                    // candidate set would expose a different choice
                    // surface, so the candidates must participate in
                    // the fingerprint. Flat list of u32 ids (mirrors
                    // the dict surface shape).
                    PendingDecision::TapPermanentChoice {
                        source_obj_id,
                        controller,
                        candidates,
                    } => Value::Array(vec![
                        Value::String("TapPermanentChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            candidates.iter().map(|id| json!(*id)).collect(),
                        ),
                    ]),
                    // **Aura-attachment substrate — Phase 1 STEP 3
                    // (round-78, 2026-05-20).** Cast-time enchant-
                    // target choice. Candidate ids fully determine
                    // the pending-state shape: re-resolving with a
                    // different candidate set would surface a
                    // different choice. Flat u32 list mirrors
                    // `TapPermanentChoice`'s shape.
                    PendingDecision::EnchantTargetChoice {
                        source_obj_id,
                        controller,
                        candidates,
                    } => Value::Array(vec![
                        Value::String("EnchantTargetChoice".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        Value::Array(
                            candidates.iter().map(|id| json!(*id)).collect(),
                        ),
                    ]),
                    // **Cracked Skull look-at-hand + may-discard
                    // substrate — Cycle 1 (2026-05-22).** The candidate
                    // ids + the target_player slot fully determine the
                    // pending-state shape: re-resolving with a different
                    // candidate set OR a different target seat would
                    // expose a different choice surface, so both
                    // participate in the fingerprint. Shape mirrors
                    // `TapPermanentChoice`'s flat u32 list, extended
                    // with the target_player slot (distinct from the
                    // controller — the discard target seat).
                    PendingDecision::LookAtOpponentHandMayDiscard {
                        source_obj_id,
                        controller,
                        target_player,
                        candidates,
                    } => Value::Array(vec![
                        Value::String("LookAtOpponentHandMayDiscard".to_string()),
                        json!(*source_obj_id),
                        json!(*controller),
                        json!(*target_player),
                        Value::Array(
                            candidates.iter().map(|id| json!(*id)).collect(),
                        ),
                    ]),
                    // **Sporogenic Infection — Cycle 1 (2026-05-22).**
                    // Source-id + target_player + candidate ids
                    // determine the pending-state shape. The variant
                    // has no `controller` slot — the controller of
                    // the trigger doesn't make any choice in this
                    // shape, only the target opponent does. Shape
                    // mirrors `TapPermanentChoice`'s flat u32 list,
                    // with the `target_player` slot replacing
                    // `controller`.
                    PendingDecision::OpponentSacrificesCreatureChoice {
                        source_obj_id,
                        target_player,
                        candidates,
                    } => Value::Array(vec![
                        Value::String(
                            "OpponentSacrificesCreatureChoice".to_string(),
                        ),
                        json!(*source_obj_id),
                        json!(*target_player),
                        Value::Array(
                            candidates.iter().map(|id| json!(*id)).collect(),
                        ),
                    ]),
                })
                .collect();
            parts_vec.push(Value::Array(pds));
        }
        let parts = Value::Array(parts_vec);

        let mut blob = String::new();
        serialize_python_compatible(&parts, &mut blob);

        let mut hasher = Sha1::new();
        hasher.update(blob.as_bytes());
        let digest = hasher.finalize();
        // 12-char lowercase hex (first 6 bytes of the 20-byte SHA1).
        let mut out = String::with_capacity(12);
        for byte in &digest[..6] {
            out.push_str(&format!("{:02x}", byte));
        }
        out
    }
}
