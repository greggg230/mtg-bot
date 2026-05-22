//! State construction from explicit zone contents.
//!
//! Bridge surface for the parity harness: lets Python build a Rust
//! GameState with caller-supplied registry + per-player library /
//! hand contents, so the same starting state can be constructed on
//! both engines from the same deck list.
//!
//! Round-23 slice (`rust-state-from-decklists`) fills in the
//! `make_state` pyfunction below. Companion to
//! `state::make_synthetic_state` — instead of hardcoding 3 printings
//! in an in-process registry, the caller hands us a `Registry`
//! they've already populated (typically with the full EoE printing
//! set parsed from Scryfall JSON) along with per-player library /
//! hand printing-id lists. We snapshot the registry, allocate fresh
//! GameObject ids sequentially from 0, and stamp each object with
//! the correct zone constant.
//!
//! What this function deliberately does NOT do:
//!
//! * Shuffle libraries by default — the caller decides whether to
//!   pre-shuffle (production setup) or hand us a fixed deterministic
//!   order (parity-smoke fixture). The optional `shuffle_seed` param
//!   added in phase-2-polish opts into an in-Rust deterministic
//!   shuffle for callers who want the convenience.
//! * Run the mulligan ritual — that's a caller responsibility
//!   layered on top of this primitive.
//! * Populate battlefield / graveyard / exile / stack — those zones
//!   are empty in a freshly-constructed state. Phase-2's parity smoke
//!   diffs the same fingerprint on both sides starting from an
//!   identical empty-board / fresh-hand state.
//!
//! ## Optional in-Rust shuffle (`shuffle_seed` + `rng`)
//!
//! When `shuffle_seed` is `Some(s)` (and `rng` is `None`), each
//! player's library is shuffled in place with an inline xorshift64
//! + Fisher-Yates (same RNG as
//! `state_ops_library.rs::shuffle_library_with_seed`). Hands are NOT
//! shuffled — hand order is caller-controlled (matches Python's
//! `simulate.setup_game`, where the hand is drawn AFTER the shuffle in
//! library order).
//!
//! When using xorshift fallback, p0 and p1 get independent permutations:
//! p1's seed is derived as `s.wrapping_mul(0x9E3779B97F4A7C15)` (the
//! golden-ratio xor constant — same pattern round-16 search-and-shuffle
//! uses to decorrelate two streams from a single user-facing seed). Two
//! make_state calls with the same `shuffle_seed` always produce the
//! same permutation, byte for byte.
//!
//! ### CE8 Slice 1 — external-RNG path
//!
//! When `rng` is `Some(py_random)` (a Python `random.Random` instance),
//! both player libraries consume Fisher-Yates draws from the SAME
//! shared external stream via `external_rng::next_below`. This matches
//! exactly what Python's `random.Random.shuffle(list)` produces, so
//! the cross-engine harness can pass the same `random.Random` into
//! both engines and get bit-identical library_p0 / library_p1 post-
//! shuffle. The `P1_SEED_MULTIPLIER` derivation is bypassed because
//! the rng's own state machine owns draw order — p0 consumes its
//! draws first, then p1 consumes the next draws from the same stream
//! (matches `random.Random.shuffle(p0); random.Random.shuffle(p1)`
//! on the Python side).
//!
//! When `rng` is `None` and `shuffle_seed` is `None` (default),
//! library order is preserved exactly as the caller passed it —
//! backward-compatible with the round-23 phase-2 signature.
//!
//! Precedence: if `rng` is `Some(_)`, the rng owns the shuffle and
//! `shuffle_seed` is ignored. This is the CE8-substrate's contract:
//! once a caller opts into externalized RNG, the integer seed is
//! obsolete because the rng's internal state replaces it.

#![allow(dead_code)]

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::cards;
use crate::state::{GameObject, GameState, ManaPool, Player};
use crate::state_ops_zones::{ZONE_HAND, ZONE_LIBRARY};

/// Initial life total per Python `simulate.setup_game` convention.
/// Mirrors the hardcoded `life: 20` in `state::make_player`.
const STARTING_LIFE: i32 = 20;

/// PHASE_UNTAP — first phase of turn 1. Mirrors Python's
/// `Phase.UNTAP = 0`. Used as the initial phase byte.
const PHASE_UNTAP: u8 = 0;

/// Golden-ratio xor constant used to derive p1's shuffle seed from p0's.
/// Same constant round-16 search-and-shuffle uses to decorrelate two
/// xorshift64 streams from a single user-facing seed.
const P1_SEED_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// Inline xorshift64 — same RNG used by `state_ops_library.rs::shuffle_library_with_seed`.
/// Kept in-module (rather than reused via a shared helper) so this file
/// has no cross-module dependency on the library primitive's internals
/// and the determinism contract is self-contained.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 collapses to zero if seeded with zero; map 0 to
        // a non-zero constant. Mirrors `state_ops_library.rs`.
        let state = if seed == 0 { 0xdead_beef_dead_beef } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Unbiased index in `[0, n)` for small `n` (n << u64::MAX).
    fn next_usize_below(&mut self, n: usize) -> usize {
        let n_u64 = n as u64;
        let bound = u64::MAX - (u64::MAX % n_u64);
        loop {
            let r = self.next_u64();
            if r < bound {
                return (r % n_u64) as usize;
            }
        }
    }
}

/// Fisher-Yates (Knuth) shuffle of `v` in place using xorshift64 seeded
/// with `seed`. No-op for len < 2.
fn fisher_yates_shuffle<T>(v: &mut [T], seed: u64) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let mut rng = XorShift64::new(seed);
    for i in (1..n).rev() {
        let j = rng.next_usize_below(i + 1);
        v.swap(i, j);
    }
}

/// Fisher-Yates (Knuth) shuffle of `v` in place consuming draws from a
/// Python `random.Random` instance via `external_rng::next_below`.
/// No-op for len < 2. Returns PyResult because each `_randbelow`
/// callback can raise (e.g. if the passed object lacks `_randbelow`).
///
/// CE8 Slice 1 substrate: gives the cross-engine harness a way to pass
/// the SAME `random.Random` into both engines so draw-order divergence
/// surfaces loudly. See `external_rng` module docstring for the
/// rationale.
fn fisher_yates_shuffle_with_external_rng<T>(
    v: &mut [T],
    rng: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let n = v.len();
    if n < 2 {
        return Ok(());
    }
    for i in (1..n).rev() {
        let j = crate::external_rng::next_below(rng, i + 1)?;
        v.swap(i, j);
    }
    Ok(())
}

/// Validate every entry in one printing-id Vec is `< registry_len`;
/// raise PyValueError with the offending id on first miss.
fn validate_printing_ids(ids: &[u32], registry_len: u32) -> PyResult<()> {
    for &pid in ids {
        if pid >= registry_len {
            return Err(PyValueError::new_err(format!(
                "printing_id {pid} not in registry (len {registry_len})"
            )));
        }
    }
    Ok(())
}

/// Push one fresh GameObject (zone-stamped, per-object state zeroed)
/// into `out` and advance `next_id`. Caller owns id allocation.
fn push_object(
    out: &mut Vec<GameObject>,
    next_id: &mut u32,
    controller: u8,
    zone: u8,
    printing_id: u32,
) {
    out.push(GameObject {
        id: *next_id,
        printing_id,
        controller,
        zone,
        tapped: false,
        // CR 302.1 + parity with Python: every freshly-allocated
        // GameObject defaults to summoning_sick=true. The field is
        // gameplay-relevant only on the battlefield (cleared at the
        // controller's untap step); for library / hand / graveyard
        // / exile objects the value is benign. Closes CE1 (round-27
        // cross-engine divergence — see RUST_PARITY_GAPS.md).
        summoning_sick: true,
        damage: 0,
        counters: 0,
        charge_counters: 0,
        loyalty_counters: 0,
        attached_to: None,
        exiled_with_source: None,
        // Round-30 stun-counter substrate: freshly-allocated objects
        // never carry stun counters; default 0 mirrors make_object.
        stun_counters: 0,
        // CE4 closure: make_state never places objects directly on
        // battlefield (bf is built up by ETB primitives), so the
        // default 0 is fine for hand / library / starting zones.
        etb_timestamp: 0,
        ueot_modifiers: Vec::new(),
        // warp_cost substrate (Phase 1, round-63): make_state seeds
        // hand/library/starting zones; nothing is warp-cast at setup.
        warp_exile_pending: false,
        warped: false,
        cast_from_exile_with_lightstall_surcharge: false,
        controller_baseline: None,
        // Aura-attachment Phase 4 S3 closure (2026-05-21): setup-
        // seeded objects are never Moonlit-substituted tokens.
        token_supertype_override: false,
        // Control-change-UEOT substrate — Cycle 1 (2026-05-21):
        // setup-seeded objects start with no UEOT controller override.
        controller_override: None,
    });
    *next_id += 1;
}

/// Build a GameState with caller-supplied registry + per-player zone
/// contents. The complement to `make_synthetic_state` — instead of
/// hardcoding 3 printings, it takes whatever the caller has already
/// registered and uses that.
///
/// **Inputs:**
///
/// * `registry` — a Python `Registry` handle, already populated with
///   the printings the caller cares about. We `snapshot()` it (clones
///   the printings Vec once into an `Arc<RegistryData>`); subsequent
///   mutations to the Python-side builder won't be visible to the
///   returned state.
/// * `p0_library_printing_ids` / `p1_library_printing_ids` — printing
///   ids to seed each player's library in the given order. Empty Vecs
///   are allowed. When `shuffle_seed` is `None`, the caller's order is
///   preserved exactly.
/// * `p0_hand_printing_ids` / `p1_hand_printing_ids` — optional
///   printing ids to seed each player's hand. `None` and an empty
///   Vec are equivalent: both produce an empty hand. No mulligan is
///   run; the caller decides what (if anything) starts in hand.
///   Hand order is never shuffled, even when `shuffle_seed` is set
///   (hand is conceptually drawn AFTER the shuffle, in library order).
/// * `shuffle_seed` — optional `u64`. When `Some(s)` and `rng` is
///   `None`, each player's library is shuffled in place via inline
///   xorshift64 + Fisher-Yates before the state is constructed. p0
///   uses seed `s` directly; p1 uses
///   `s.wrapping_mul(0x9E3779B97F4A7C15)` (golden-ratio xor constant —
///   decorrelates the two streams from a single user-facing seed).
///   When `None` (and `rng` is also `None`), no shuffle is performed
///   — library order is the caller's input order. IGNORED when `rng`
///   is `Some(_)`.
/// * `rng` — optional Python `random.Random` instance (CE8 Slice 1
///   substrate). When `Some(_)`, both library shuffles consume draws
///   from this shared external stream via
///   `external_rng::next_below`. Matches what
///   `random.Random.shuffle(p0); random.Random.shuffle(p1)` produces
///   on the Python side, enabling cross-engine bit-parity on shuffled
///   libraries. Defaults to `None`; production callers (cli, bench)
///   stay on the xorshift64 path.
///
/// **Object ids:** allocated sequentially starting from 0 in this
/// order — p0 library, p0 hand, p1 library, p1 hand. Within each
/// segment, ids match the caller's index order. The shuffle (if any)
/// runs AFTER id allocation, so it reorders GameObjects (id + printing
/// pair together) — the same printing_id keeps the same id regardless
/// of where it lands in the shuffled library.
///
/// **Initial state:** life=20, active_player=0, priority_player=0,
/// turn_number=1, phase=PHASE_UNTAP, empty stack / attackers /
/// blockers / battlefield / graveyard / exile, empty mana pools.
///
/// **Validation:** every printing_id in all four Vec params must be
/// `< registry.len()`. A single out-of-range id raises PyValueError
/// with the offending id and the registry length. Validation runs
/// before any allocation, so a bad input doesn't leak a partially-
/// built state.
#[pyfunction]
#[pyo3(signature = (
    registry,
    p0_library_printing_ids,
    p1_library_printing_ids,
    p0_hand_printing_ids = None,
    p1_hand_printing_ids = None,
    shuffle_seed = None,
    rng = None,
))]
pub fn make_state(
    registry: &Bound<'_, cards::Registry>,
    p0_library_printing_ids: Vec<u32>,
    p1_library_printing_ids: Vec<u32>,
    p0_hand_printing_ids: Option<Vec<u32>>,
    p1_hand_printing_ids: Option<Vec<u32>>,
    shuffle_seed: Option<u64>,
    rng: Option<&Bound<'_, PyAny>>,
) -> PyResult<GameState> {
    let snapshot = registry.borrow().snapshot();
    let registry_len = snapshot.len() as u32;

    let p0_hand_ids = p0_hand_printing_ids.unwrap_or_default();
    let p1_hand_ids = p1_hand_printing_ids.unwrap_or_default();

    // Validate everything up front so a bad input never leaks a
    // half-built state. The check is O(n) over the union of all four
    // lists, dominated by registry lookups (just a length compare).
    validate_printing_ids(&p0_library_printing_ids, registry_len)?;
    validate_printing_ids(&p1_library_printing_ids, registry_len)?;
    validate_printing_ids(&p0_hand_ids, registry_len)?;
    validate_printing_ids(&p1_hand_ids, registry_len)?;

    let mut next_id: u32 = 0;

    let mut p0_library_vec = Vec::with_capacity(p0_library_printing_ids.len());
    for pid in &p0_library_printing_ids {
        push_object(&mut p0_library_vec, &mut next_id, 0, ZONE_LIBRARY, *pid);
    }
    let mut p0_hand_vec = Vec::with_capacity(p0_hand_ids.len());
    for pid in &p0_hand_ids {
        push_object(&mut p0_hand_vec, &mut next_id, 0, ZONE_HAND, *pid);
    }
    let mut p1_library_vec = Vec::with_capacity(p1_library_printing_ids.len());
    for pid in &p1_library_printing_ids {
        push_object(&mut p1_library_vec, &mut next_id, 1, ZONE_LIBRARY, *pid);
    }
    let mut p1_hand_vec = Vec::with_capacity(p1_hand_ids.len());
    for pid in &p1_hand_ids {
        push_object(&mut p1_hand_vec, &mut next_id, 1, ZONE_HAND, *pid);
    }

    // Optional in-Rust shuffle. Runs AFTER id allocation so the
    // shuffle reorders whole GameObjects (id + printing_id together).
    // Hands are intentionally NOT shuffled — see module docstring.
    //
    // Precedence (CE8 Slice 1):
    //   * `rng = Some(py_random)` → both libraries consume from the
    //     SHARED external stream, in the order p0 then p1, matching
    //     `random.Random.shuffle(p0); random.Random.shuffle(p1)`.
    //     `shuffle_seed` is ignored on this path.
    //   * `rng = None, shuffle_seed = Some(s)` → legacy xorshift64
    //     path: p0 uses `s`, p1 uses `s.wrapping_mul(P1_SEED_MULTIPLIER)`.
    //     The derived p1 seed keeps the two streams decorrelated so
    //     p0 and p1 don't share a permutation even when their
    //     starting libraries happen to be identical.
    //   * `rng = None, shuffle_seed = None` → no shuffle (caller-
    //     order preserved). Backward-compat.
    if let Some(py_rng) = rng {
        fisher_yates_shuffle_with_external_rng(&mut p0_library_vec, py_rng)?;
        fisher_yates_shuffle_with_external_rng(&mut p1_library_vec, py_rng)?;
    } else if let Some(seed) = shuffle_seed {
        fisher_yates_shuffle(&mut p0_library_vec, seed);
        let p1_seed = seed.wrapping_mul(P1_SEED_MULTIPLIER);
        fisher_yates_shuffle(&mut p1_library_vec, p1_seed);
    }

    let p0 = Player {
        id: 0,
        life: STARTING_LIFE,
        mana_pool: ManaPool::default(),
        hand: Arc::new(p0_hand_vec),
        library: Arc::new(p0_library_vec),
        battlefield: Arc::new(Vec::new()),
        graveyard: Arc::new(Vec::new()),
        exile: Arc::new(Vec::new()),
        has_lost: false,
        attempted_draw_on_empty: false,
        mulligan_count: 0,
        life_gained_this_turn: 0,
        poison_counters: 0,
        spells_cast_this_turn: 0,
        artifacts_entered_under_your_control_this_turn: 0,
    };
    let p1 = Player {
        id: 1,
        life: STARTING_LIFE,
        mana_pool: ManaPool::default(),
        hand: Arc::new(p1_hand_vec),
        library: Arc::new(p1_library_vec),
        battlefield: Arc::new(Vec::new()),
        graveyard: Arc::new(Vec::new()),
        exile: Arc::new(Vec::new()),
        has_lost: false,
        attempted_draw_on_empty: false,
        mulligan_count: 0,
        life_gained_this_turn: 0,
        poison_counters: 0,
        spells_cast_this_turn: 0,
        artifacts_entered_under_your_control_this_turn: 0,
    };

    Ok(GameState {
        players: [p0, p1],
        registry: snapshot,
        active_player: 0,
        priority_player: 0,
        turn_number: 1,
        phase: PHASE_UNTAP,
        attackers: Vec::new(),
        blockers: Vec::new(),
        stack: Vec::new(),
        // G7 S5a: empty spell-stack-source-zone store at game start
        // (Option β). Same lifecycle as `stack`: filled by a cast,
        // drained by `move_source_post_resolve` at resolution.
        stack_objects: Vec::new(),
        // G7 Class-C delayed-trigger port: empty at game start; warp
        // casts schedule WarpExileSelf entries; end-step entry drains.
        delayed_triggers: Vec::new(),
        consecutive_passes: 0,
        permanent_left_battlefield_this_turn: false,
        spell_warped_this_turn: false,
        // Phase 4 S3 (Moonlit Meditation substrate landing,
        // 2026-05-20): per-player created-tokens-this-turn flags
        // start cleared at game start, same shape as the void flags.
        created_tokens_this_turn: [false; 2],
        pending_decisions: Vec::new(),
        next_etb_timestamp: 0,
        // R76-1: seed the monotonic object-id counter past every
        // object the setup build created. `next_id` was bumped once
        // per pushed object across p0_library / p0_hand / p1_library
        // / p1_hand, so it now equals the object count = 1 + the max
        // initial id (the optional in-Rust shuffle above only
        // reorders whole GameObjects, never mints, so `next_id` is
        // still correct post-shuffle). This is byte-equal to Python's
        // `next_object_id` after `build_python_state` builds the
        // equivalent population via `state.new_object` in the same
        // p0-then-p1 order (see cross_engine.build_python_state).
        next_object_id: next_id,
        resolving_combat_damage_amount: 0,
        resolving_combat_damage_target_player_id: u8::MAX,
        winner_id_sticky: None,
        // Effect-rng-threading closure: default `None`; cross-engine
        // smoke callers opt in via `set_external_rng(Some(rng))` after
        // `make_state` returns.
        rng: None,
        // ETB-trigger Slice 1: empty once-per-turn ledger at construction.
        triggered_this_turn: std::collections::HashSet::new(),
        // ETB-trigger Slice 2: empty deferred-dispatch queue at
        // construction.
        pending_etb_dispatches: Vec::new(),
        // ETB-trigger Slice 5: empty deferred-DIES queue at
        // construction.
        pending_dies_dispatches: Vec::new(),
        // ETB-trigger Slice 8: empty deferred-LEAVES queue at
        // construction.
        pending_leaves_dispatches: Vec::new(),
        // Slice 14: empty deferred IS_SACRIFICED queue at
        // construction.
        pending_is_sacrificed_dispatches: Vec::new(),
        // Round-55: empty token-cease queue at construction.
        pending_token_cease: Vec::new(),
        // CE20 follow-up Slice 4: transient payload-target slot;
        // None by default. Set by the resolver during targeted
        // activation dispatch, cleared after each apply.
        current_resolve_payload_target_id: None,
        // Spell-card-bridge lever Phase 2: second resolve-time
        // payload-target slot (2-slot spells). None by default.
        current_resolve_payload_target_id_2: None,
        // CE20 Slice 5 (Walker loyalty Slice C): empty
        // once-per-turn-per-walker ledger at construction.
        loyalty_activated_this_turn: std::collections::HashSet::new(),
        // DrawCards controller-of-effect fix: transient slot, None
        // at construction; set by the resolver during apply_effect.
        current_resolve_source_controller: None,
        // ModalChooseOne envelope — Cycle 2: transient mode-index
        // slot; None at construction (no spell on the stack yet).
        // Set by the resolver during apply_effect for a modal
        // spell-cast resolution.
        current_resolve_chosen_mode_idx: None,
        // Close Encounter additional-cost-choice envelope —
        // Cycle 1: transient choice-id slot; None at construction
        // (no spell on the stack yet). Set by the resolver during
        // apply_effect for an additional-cost-choose-your-creature
        // spell-cast resolution.
        current_resolve_additional_cost_choice_id: None,
        // scour-for-scrap-slice Cycle 1 widen: transient multi-pick
        // slot; None at construction; set during a modal-or-both
        // spell-cast resolution.
        current_resolve_chosen_mode_idxs: None,
        // X-value subsystem Cycle 1: transient X-value slot; None at
        // construction (no X-cost spell on the stack yet — pre-
        // Cycle-1 every spell pushed `x_value: None`, so the
        // transient stays None until Cycle 2/3 mints a real X bind).
        current_resolve_x_value: None,
    })
}
