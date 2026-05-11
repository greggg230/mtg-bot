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
//! ## Optional in-Rust shuffle (`shuffle_seed`)
//!
//! When `shuffle_seed` is `Some(s)`, each player's library is shuffled
//! in place with an inline xorshift64 + Fisher-Yates (same RNG as
//! `state_ops_library.rs::shuffle_library_with_seed`). Hands are NOT
//! shuffled — hand order is caller-controlled (matches Python's
//! `simulate.setup_game`, where the hand is drawn AFTER the shuffle in
//! library order).
//!
//! p0 and p1 get independent permutations: p1's seed is derived as
//! `s.wrapping_mul(0x9E3779B97F4A7C15)` (the golden-ratio xor
//! constant — same pattern round-16 search-and-shuffle uses to
//! decorrelate two streams from a single user-facing seed). Two
//! make_state calls with the same `shuffle_seed` always produce the
//! same permutation, byte for byte.
//!
//! When `shuffle_seed` is `None` (default), library order is preserved
//! exactly as the caller passed it — backward-compatible with the
//! round-23 phase-2 signature.
//!
//! The primary parity path remains Python-side shuffling via
//! `random.Random(seed).shuffle(deck_ids)` before calling make_state;
//! the in-Rust shuffle is for Rust-only parity-smoke modes and future
//! RNG-externalization work.

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
        summoning_sick: false,
        damage: 0,
        counters: 0,
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
/// * `shuffle_seed` — optional `u64`. When `Some(s)`, each player's
///   library is shuffled in place via inline xorshift64 + Fisher-Yates
///   before the state is constructed. p0 uses seed `s` directly; p1
///   uses `s.wrapping_mul(0x9E3779B97F4A7C15)` (golden-ratio xor
///   constant — decorrelates the two streams from a single
///   user-facing seed). When `None`, no shuffle is performed —
///   library order is the caller's input order.
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
))]
pub fn make_state(
    registry: &Bound<'_, cards::Registry>,
    p0_library_printing_ids: Vec<u32>,
    p1_library_printing_ids: Vec<u32>,
    p0_hand_printing_ids: Option<Vec<u32>>,
    p1_hand_printing_ids: Option<Vec<u32>>,
    shuffle_seed: Option<u64>,
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
    // p1 uses a derived seed so p0 and p1 don't share a permutation
    // even when their starting libraries happen to be identical.
    // Hands are intentionally NOT shuffled — see module docstring.
    if let Some(seed) = shuffle_seed {
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
        consecutive_passes: 0,
    })
}
