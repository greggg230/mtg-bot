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
//! ## Streaming-bytes optimization (this slice)
//!
//! The original Round-22 impl built a `serde_json::Value` tree, ran a
//! recursive `serialize_python_compatible` walker that pushed onto a
//! `String`, then fed that String to SHA1. Decomposition bench
//! (`scripts/bench_rust_fingerprint_decomposition.py`) showed ~60% of
//! the per-call cost lived in the Value-tree allocation + the
//! intermediate `String` churn. This slice replaces both with a
//! streaming serializer that emits JSON bytes directly into the SHA1
//! state via a small `Sha1Writer` (a single 4KiB stack buffer that
//! flushes to `Sha1::update` when full). No `Value` tree, no
//! intermediate `String`, no per-array `Vec<Value>` allocations.
//!
//! All emit helpers are written against the `ByteSink` trait so the
//! same code drives both the SHA1 stream and a test-only
//! `Vec<u8>`-backed sink — the `_fingerprint_blob` pymethod exposed
//! to tests collects the streamed bytes through the same emitters
//! the real fingerprint uses, so the streaming-vs-reference parity
//! test in `tests_rust/test_fingerprint.py` is a true cross-check
//! of the byte stream rather than a parallel reimplementation.
//!
//! Byte-equality with the previous impl is preserved EXACTLY — every
//! pinned baseline (`93ab1bd03f94`, `e1d70afe6742`, `a1193453edec`,
//! the +9 in `test_fingerprint.py` and `test_setup.py`) continues to
//! match.
//!
//! ## JSON-format gotchas
//!
//! Python `json.dumps(..., default=repr, sort_keys=False)` has a
//! specific output shape we have to match exactly to land the same
//! sha1 digest:
//!
//! * Python tuples and lists both serialize as JSON arrays — there
//!   are no JSON tuples. We mirror by emitting `[` / `]` for both.
//! * Booleans: Python emits `true` / `false`; we match.
//! * `None`: Python emits `null`; we match.
//! * Integers: bare digits both sides.
//! * Strings: JSON-quoted with backslash-escapes; we replicate
//!   serde_json's default escape set byte-for-byte (the previous
//!   impl's escape was performed by `serde_json::to_string`, and
//!   every baseline was pinned against that output — so this slice
//!   matches serde_json, NOT Python's `ensure_ascii=True`. The card
//!   names actually emitted from the registry are ASCII, so the
//!   difference is invisible today; if a non-ASCII name ever lands,
//!   the baseline tests will catch the divergence and force the
//!   recipe update on both sides).
//! * Whitespace: Python's default uses `", "` / `": "` separators
//!   (a single space after every comma and colon). We emit those
//!   spaces explicitly between array items.
//! * `sort_keys=False`: we never emit JSON objects (only arrays),
//!   so this is moot for our output.
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
//!   Rust has no `attached_to` field yet — we emit `null`.
//!
//! Round-22 doesn't ship a Python-side shim; the cross-engine parity
//! test in `tests_rust/test_fingerprint.py` asserts equality against
//! a hard-coded baseline string computed from the same algorithm,
//! locking the format in a regression-safe way.

use crate::state::{GameObject, GameState, StackItem};
use pyo3::prelude::*;
use sha1::{Digest, Sha1};

/// Sink the emit helpers write JSON bytes into. The real fingerprint
/// path implements this against a 4KiB-buffered SHA1 writer; the
/// test-only `_fingerprint_blob` implements it against a `Vec<u8>`
/// so the same emit code produces the same bytes for both paths.
trait ByteSink {
    fn write_byte(&mut self, b: u8);
    fn write_bytes(&mut self, bytes: &[u8]);
}

/// Streaming SHA1 writer with a small stack-allocated buffer.
///
/// JSON bytes are pushed one piece at a time (one bracket, one
/// digit run, one short string); the buffer absorbs the per-byte /
/// per-small-write overhead so `Sha1::update` is called in
/// cache-friendly bursts.
///
/// Buffer size choice: 4096 bytes is one page on x86_64 and large
/// enough that even the biggest synthetic-state blob (~6KiB at the
/// (50, 7, 50) bench size) flushes only one or two times per
/// fingerprint call. Smaller (256, 512) would force flushes
/// mid-blob and erase the gain; larger (8KiB+) doesn't help
/// because the blobs aren't long enough.
struct Sha1Writer {
    hasher: Sha1,
    buf: [u8; 4096],
    pos: usize,
}

impl Sha1Writer {
    fn new() -> Self {
        Self {
            hasher: Sha1::new(),
            buf: [0u8; 4096],
            pos: 0,
        }
    }

    #[inline(always)]
    fn flush(&mut self) {
        if self.pos > 0 {
            self.hasher.update(&self.buf[..self.pos]);
            self.pos = 0;
        }
    }

    fn finalize(mut self) -> [u8; 20] {
        self.flush();
        self.hasher.finalize().into()
    }
}

impl ByteSink for Sha1Writer {
    #[inline(always)]
    fn write_byte(&mut self, b: u8) {
        if self.pos == self.buf.len() {
            self.flush();
        }
        self.buf[self.pos] = b;
        self.pos += 1;
    }

    #[inline(always)]
    fn write_bytes(&mut self, bytes: &[u8]) {
        let remaining = self.buf.len() - self.pos;
        if bytes.len() <= remaining {
            self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
            self.pos += bytes.len();
            return;
        }
        // Slow path: split-flush. For payloads larger than the buffer
        // flush what's pending and feed the long slice straight to the
        // hasher (avoids a memcpy-then-flush round-trip in the unusual
        // long-string case).
        self.flush();
        if bytes.len() >= self.buf.len() {
            self.hasher.update(bytes);
        } else {
            self.buf[..bytes.len()].copy_from_slice(bytes);
            self.pos = bytes.len();
        }
    }
}

impl ByteSink for Vec<u8> {
    #[inline(always)]
    fn write_byte(&mut self, b: u8) {
        self.push(b);
    }
    #[inline(always)]
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// Emit a `u64` as decimal digits with no heap allocation.
#[inline(always)]
fn emit_u64<W: ByteSink>(w: &mut W, n: u64) {
    let mut buf = [0u8; 24];
    let s = u64_to_str(n, &mut buf);
    w.write_bytes(s);
}

/// Emit an `i64` as decimal digits with no heap allocation.
#[inline(always)]
fn emit_i64<W: ByteSink>(w: &mut W, n: i64) {
    let mut buf = [0u8; 24];
    let s = i64_to_str(n, &mut buf);
    w.write_bytes(s);
}

/// Format an `i64` into the back of `buf` and return the slice that
/// holds the digits (plus the optional leading `-`).
#[inline(always)]
fn i64_to_str(n: i64, buf: &mut [u8; 24]) -> &[u8] {
    if n < 0 {
        // i64::MIN-safe magnitude via i128.
        let mag = (n as i128).unsigned_abs() as u64;
        let mut end = buf.len();
        u64_write(mag, &mut end, buf);
        end -= 1;
        buf[end] = b'-';
        return &buf[end..];
    }
    let mut end = buf.len();
    u64_write(n as u64, &mut end, buf);
    &buf[end..]
}

#[inline(always)]
fn u64_to_str(n: u64, buf: &mut [u8; 24]) -> &[u8] {
    let mut end = buf.len();
    u64_write(n, &mut end, buf);
    &buf[end..]
}

/// Write `n`'s decimal digits into `buf` starting from `*end` and
/// growing backwards. Updates `*end` to point at the first written
/// digit.
#[inline(always)]
fn u64_write(mut n: u64, end: &mut usize, buf: &mut [u8; 24]) {
    if n == 0 {
        *end -= 1;
        buf[*end] = b'0';
        return;
    }
    while n > 0 {
        *end -= 1;
        buf[*end] = b'0' + (n % 10) as u8;
        n /= 10;
    }
}

/// Emit a JSON-encoded string with serde_json's default escape rules:
/// `"` -> `\"`, `\` -> `\\`, control chars 0x00..=0x1F use short forms
/// (`\b`, `\t`, `\n`, `\f`, `\r`) or `\u00XX`, every other byte passes
/// through. Non-ASCII UTF-8 is emitted verbatim (matching
/// `serde_json::to_string`, the impl the previous fingerprint
/// baselines were pinned against — see module doc).
fn emit_string<W: ByteSink>(w: &mut W, s: &str) {
    w.write_byte(b'"');
    let bytes = s.as_bytes();
    let mut run_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let escape: Option<&[u8]> = match b {
            b'"' => Some(b"\\\""),
            b'\\' => Some(b"\\\\"),
            0x08 => Some(b"\\b"),
            0x09 => Some(b"\\t"),
            0x0A => Some(b"\\n"),
            0x0C => Some(b"\\f"),
            0x0D => Some(b"\\r"),
            0x00..=0x1F => None, // generic \u00XX path below
            _ => {
                i += 1;
                continue;
            }
        };
        if i > run_start {
            w.write_bytes(&bytes[run_start..i]);
        }
        match escape {
            Some(esc) => w.write_bytes(esc),
            None => {
                w.write_bytes(b"\\u00");
                w.write_byte(hex_digit(b >> 4));
                w.write_byte(hex_digit(b & 0x0F));
            }
        }
        i += 1;
        run_start = i;
    }
    if run_start < bytes.len() {
        w.write_bytes(&bytes[run_start..]);
    }
    w.write_byte(b'"');
}

#[inline(always)]
fn hex_digit(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        10..=15 => b'a' + (nibble - 10),
        _ => unreachable!(),
    }
}

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

/// Emit the per-object 8-tuple as a JSON array, item-separator
/// `, ` (matches Python's default `json.dumps` output). Field
/// order: `(printing_name | None, int(tapped), int(summoning_sick),
/// damage, counters, attached_to, controller_id, id)`.
#[inline]
fn emit_object_signature<W: ByteSink>(w: &mut W, state: &GameState, obj: &GameObject) {
    w.write_byte(b'[');
    // 0: printing name | None
    match state.registry.lookup(obj.printing_id) {
        Some(p) => emit_string(w, &p.name),
        None => w.write_bytes(b"null"),
    }
    w.write_bytes(b", ");
    // 1: int(tapped) -> 0 / 1
    w.write_byte(if obj.tapped { b'1' } else { b'0' });
    w.write_bytes(b", ");
    // 2: int(summoning_sick) -> 0 / 1
    w.write_byte(if obj.summoning_sick { b'1' } else { b'0' });
    w.write_bytes(b", ");
    // 3: damage (i16 — Python int)
    emit_i64(w, obj.damage as i64);
    w.write_bytes(b", ");
    // 4: counters (i32 — Python int)
    emit_i64(w, obj.counters as i64);
    w.write_bytes(b", ");
    // 5: attached_to — no Rust field yet, always null
    w.write_bytes(b"null");
    w.write_bytes(b", ");
    // 6: controller_id (u8)
    emit_u64(w, obj.controller as u64);
    w.write_bytes(b", ");
    // 7: id (u32)
    emit_u64(w, obj.id as u64);
    w.write_byte(b']');
}

/// Emit a zone (slice of GameObjects) as a JSON array of 8-tuples.
#[inline]
fn emit_zone<W: ByteSink>(w: &mut W, state: &GameState, zone: &[GameObject]) {
    w.write_byte(b'[');
    for (i, obj) in zone.iter().enumerate() {
        if i > 0 {
            w.write_bytes(b", ");
        }
        emit_object_signature(w, state, obj);
    }
    w.write_byte(b']');
}

/// Emit the per-player 8-tuple — see module doc for the field order.
fn emit_player_signature<W: ByteSink>(w: &mut W, state: &GameState, idx: usize) {
    let p = &state.players[idx];
    let pool = &p.mana_pool;
    w.write_byte(b'[');
    // 0: id (u8)
    emit_u64(w, p.id as u64);
    w.write_bytes(b", ");
    // 1: life (i32)
    emit_i64(w, p.life as i64);
    w.write_bytes(b", ");
    // 2: has_lost — life <= 0 (bool)
    w.write_bytes(if p.life <= 0 { &b"true"[..] } else { &b"false"[..] });
    w.write_bytes(b", ");
    // 3: hand_sig
    emit_zone(w, state, &p.hand);
    w.write_bytes(b", ");
    // 4: library_sig
    emit_zone(w, state, &p.library);
    w.write_bytes(b", ");
    // 5: graveyard_sig
    emit_zone(w, state, &p.graveyard);
    w.write_bytes(b", ");
    // 6: exile_sig
    emit_zone(w, state, &p.exile);
    w.write_bytes(b", ");
    // 7: mana_pool_sorted_items — fixed lex-sorted order
    //    [["B", b], ["C", c], ["G", g], ["R", r], ["U", u], ["W", w]]
    w.write_byte(b'[');
    w.write_bytes(b"[\"B\", ");
    emit_u64(w, pool.black as u64);
    w.write_bytes(b"], [\"C\", ");
    emit_u64(w, pool.colorless as u64);
    w.write_bytes(b"], [\"G\", ");
    emit_u64(w, pool.green as u64);
    w.write_bytes(b"], [\"R\", ");
    emit_u64(w, pool.red as u64);
    w.write_bytes(b"], [\"U\", ");
    emit_u64(w, pool.blue as u64);
    w.write_bytes(b"], [\"W\", ");
    emit_u64(w, pool.white as u64);
    w.write_byte(b']');
    w.write_byte(b']');
    w.write_byte(b']');
}

/// Emit one StackItem as the same 8-tuple shape a zone object uses.
/// Most slots are null / 0 since a stack item isn't a permanent.
#[inline]
fn emit_stack_item<W: ByteSink>(w: &mut W, si: &StackItem) {
    w.write_byte(b'[');
    w.write_bytes(b"null, "); // 0: printing name unknown for now
    w.write_bytes(b"0, "); // 1: tapped
    w.write_bytes(b"0, "); // 2: summoning_sick
    w.write_bytes(b"0, "); // 3: damage
    w.write_bytes(b"0, "); // 4: counters
    w.write_bytes(b"null, "); // 5: attached_to
    emit_u64(w, si.controller as u64);
    w.write_bytes(b", ");
    emit_u64(w, si.source_obj_id as u64);
    w.write_byte(b']');
}

/// Stream the full `parts` array bytes into `w`. Same shape as the
/// previous serde_json::Value tree, emitted directly into the sink.
fn emit_parts<W: ByteSink>(w: &mut W, state: &GameState) {
    w.write_byte(b'[');
    // 0: turn_number
    emit_u64(w, state.turn_number as u64);
    w.write_bytes(b", ");
    // 1: step name | None
    match phase_to_step_name(state.phase) {
        Some(name) => emit_string(w, name),
        None => w.write_bytes(b"null"),
    }
    w.write_bytes(b", ");
    // 2: active_player (u8)
    emit_u64(w, state.active_player as u64);
    w.write_bytes(b", ");
    // 3: priority_player (u8) — original impl emitted it as a bare
    //    int (json!(self.priority_player)); we match.
    emit_u64(w, state.priority_player as u64);
    w.write_bytes(b", ");
    // 4: is_over — either player's life <= 0
    let p0_lost = state.players[0].life <= 0;
    let p1_lost = state.players[1].life <= 0;
    w.write_bytes(if p0_lost || p1_lost { &b"true"[..] } else { &b"false"[..] });
    w.write_bytes(b", ");
    // 5: is_draw — both lost
    w.write_bytes(if p0_lost && p1_lost { &b"true"[..] } else { &b"false"[..] });
    w.write_bytes(b", ");
    // 6: winner_id (or null)
    match (p0_lost, p1_lost) {
        (false, true) => emit_u64(w, state.players[0].id as u64),
        (true, false) => emit_u64(w, state.players[1].id as u64),
        _ => w.write_bytes(b"null"),
    }
    w.write_bytes(b", ");
    // 7: player 0 signature
    emit_player_signature(w, state, 0);
    w.write_bytes(b", ");
    // 8: player 1 signature
    emit_player_signature(w, state, 1);
    w.write_bytes(b", ");
    // 9: battlefield — p0's then p1's, same shape as zone sig.
    w.write_byte(b'[');
    let bf0 = &state.players[0].battlefield;
    let bf1 = &state.players[1].battlefield;
    let mut first = true;
    for obj in bf0.iter().chain(bf1.iter()) {
        if !first {
            w.write_bytes(b", ");
        }
        first = false;
        emit_object_signature(w, state, obj);
    }
    w.write_byte(b']');
    w.write_bytes(b", ");
    // 10: stack — list of 8-tuples in the same shape.
    w.write_byte(b'[');
    for (i, si) in state.stack.iter().enumerate() {
        if i > 0 {
            w.write_bytes(b", ");
        }
        emit_stack_item(w, si);
    }
    w.write_byte(b']');
    w.write_byte(b']');
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
    /// Implementation: streams JSON bytes directly into a SHA1 hasher
    /// via a small stack-allocated buffer — no intermediate
    /// `serde_json::Value` tree and no intermediate `String` blob.
    /// See the module doc for the byte-equality guarantee and the
    /// pinned baseline set.
    fn fingerprint(&self) -> String {
        let mut w = Sha1Writer::new();
        emit_parts(&mut w, self);
        let digest = w.finalize();
        // 12-char lowercase hex (first 6 bytes of the 20-byte SHA1).
        let mut out = String::with_capacity(12);
        for byte in &digest[..6] {
            out.push(hex_digit(byte >> 4) as char);
            out.push(hex_digit(byte & 0x0F) as char);
        }
        out
    }

    /// Internal helper exposed for the streaming-vs-reference parity
    /// sanity tests in `tests_rust/test_fingerprint.py`: returns the
    /// EXACT JSON-encoded blob `fingerprint()` streams into SHA1.
    ///
    /// Implemented by running the same `emit_parts` walk against a
    /// `Vec<u8>` sink (via the `ByteSink` trait), so any divergence
    /// from the SHA1-streaming path would require divergence in the
    /// `ByteSink` impl — which the tests themselves would also catch.
    ///
    /// Not part of the public API surface; callers outside the test
    /// suite should always use `fingerprint()`.
    #[pyo3(name = "_fingerprint_blob")]
    fn fingerprint_blob(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        emit_parts(&mut buf, self);
        buf
    }
}
