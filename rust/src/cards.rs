// Card-data bridge: closed-vocabulary Rust representations of card
// printings + Python <-> Rust dict parsing / serialization.
//
// Design (see rust/README.md and worklog 2026-05-09):
//
// - Python builds nested dicts describing each card; the dicts are
//   plain (serializable, diffable, easy to fixture in tests).
// - Rust owns the closed enum vocabulary: `Effect`, `TargetFilter`,
//   `ManaCost`, `CardPrinting`. Every variant Python emits has a
//   Rust counterpart.
// - The dict parser is the single point of validation. Unknown
//   `kind` strings, missing required fields, type mismatches all
//   raise loudly at `register_printing` time, not at game-resolution
//   time.
// - `Registry` is a `#[pyclass]` Python holds explicitly. No global
//   mutable state. CardPrintings are immutable once registered;
//   GameObjects reference printings by id (see state.rs).
//
// First-slice scope: one effect (`DealDamage`), two target filters
// (`AnyPlayer`, `AnyCreature`), 5-color mana cost. Future commits
// add variants one at a time, each with parity tests against the
// Python equivalent on a real EoE card.
//
// **Storage shape:** `RegistryData` is the plain-Rust struct that
// owns the printings. `Registry` is the pyclass wrapper that holds
// a builder Vec; `snapshot()` freezes a clone of that Vec into an
// `Arc<RegistryData>` that `GameState` shares at refcount-bump
// cost. The first-slice register-then-snapshot workflow (build the
// registry up front, then snapshot once) means we don't yet need
// interior mutability — append-then-freeze is enough.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ManaCost {
    pub generic: u8,
    pub white: u8,
    pub blue: u8,
    pub black: u8,
    pub red: u8,
    pub green: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetFilter {
    AnyPlayer,
    AnyCreature,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    DealDamage { amount: i32, target: TargetFilter },
    /// Controller draws `count` cards. No target — caller's
    /// controller is implicit. Real "target player draws N" effects
    /// (e.g. Wheel of Fortune) need a player-target variant.
    DrawCards { count: i32 },
    /// Controller gains `amount` life. No target — caller's
    /// controller is implicit. Real "target player gains N life"
    /// effects need a player-target variant.
    GainLife { amount: i32 },
    /// Bounce: target permanent's owner (= controller in first slice)
    /// puts it back into their hand. Composes round-1 zones move.
    ReturnToHand { target: TargetFilter },
    /// Mill: caller's controller puts the top `count` cards of their
    /// library into their graveyard. No target — controller implicit.
    Mill { count: i32 },
    /// Target player puts the top `count` cards of their library into
    /// their graveyard (CR 701.13b "mill"). 2 cards in EoE pool per
    /// round-20 measurement (Scrabbling Skullcrab etc.).
    ///
    /// **Distinct from `Effect::Mill`** — that arm mills the
    /// CONTROLLER's library. `MillTargetPlayer` mills a chosen target
    /// player (in 2-player engine = defender). First-slice
    /// deterministic choice: defender (= 1 - controller). Future
    /// multiplayer / archenemy work needs an explicit choice callback.
    ///
    /// Negative `count` raises. count=0 is a legal no-op. If the
    /// library has fewer than `count` cards, all remaining cards mill
    /// (Python parity).
    MillTargetPlayer { count: i32 },
    /// **BLOCKED-RUST-EFFECT slice (round 42) -- mill an opponent
    /// player a number of cards equal to the source-controller's
    /// current life total.** Mirrors Python's
    /// `MillTargetPlayer(use_controller_life=True)` (Space-Time
    /// Anomaly, EoE; `effects/generic/draw_mill.py:151`). Sister of
    /// rounds 32 / 35 / 36 / 37 (damage substrate, four denominators:
    /// artifact-count / hand-size / lands-controlled / creatures-you-
    /// control), round 33 (counters-on-self substrate, one denominator
    /// so far), and round 34 (surveil substrate, one denominator so
    /// far) — same dynamic-X-from-count plumbing, NEW apply substrate
    /// (mill-target-player in place of damage / counters / surveil)
    /// AND NEW denominator axis (source-controller life total, vs the
    /// four bf-permanent-count denominators and hand-size denominator
    /// already shipped). Establishes the FIFTH denominator and the
    /// FOURTH apply substrate on the X-from-count plumbing,
    /// confirming bidirectional extensibility across substrate AND
    /// denominator axes.
    ///
    /// **Mechanism — count + apply.** At resolve time, resolve the
    /// source's controller (5-zone walk + stack scan; mirrors
    /// round-32 / 35 / 36 / 37 sister source-controller resolution
    /// pattern). Read `state.players[controller].life`; clamp to a
    /// non-negative count via `max(0, life)` (a negative life total
    /// — e.g. arbitrary loss-of-life past zero — produces a
    /// zero-count silent no-op, matching Python's
    /// `max(0, state.player_by_id(controller_id).life)` clamp at
    /// `draw_mill.py:187`). Mill that many cards from the opponent's
    /// library top (= `1 - controller` per round-32 / 35 / 36 / 37
    /// sister target shape; first-slice deterministic single-opponent
    /// pick). If the opponent's library is empty mid-mill, stop early
    /// (matches Python's `if not target.library: break` shape and
    /// mirrors the round-19 MillTargetPlayer arm's
    /// `n_to_mill = min(count, lib.len())` clamp).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 / 35 / 36 / 37 sister shape — needs to
    /// resolve even if source has left bf, e.g. counterspelled
    /// trigger that still drives the count from the original ETB
    /// snapshot). If untrackable -> silent fizzle.
    ///
    /// **Target semantics — first-slice deterministic.** Pick
    /// `1 - controller` on a 2-player engine (mirrors the round-19
    /// MillTargetPlayer arm's deterministic `target_idx = 1 -
    /// controller` shape post-I77 controller-of-effect fix). Future
    /// multiplayer fanout needs an explicit target-player choice
    /// callback (mirrors round-19's deferred-multiplayer note).
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `MillTargetPlayer` carries an
    /// `amount: int = 1` AND a `use_controller_life: bool = False`
    /// flag; this Rust arm is the dedicated dynamic-X sister to the
    /// existing `MillTargetPlayer { count: i32 }` arm, matching the
    /// sister-variant precedent established by rounds 32 / 35 / 36 /
    /// 37 (damage substrate splits the static-amount sister
    /// `DealDamageToOpponentCreature { amount }` from the
    /// dynamic-X-from-count sisters). The Python bridge translates
    /// this Rust unit variant to
    /// `MillTargetPlayer(use_controller_life=True)` (the existing
    /// Python class's flag lane) — no new Python class needed
    /// because the flag-on-existing-class shape already exists and is
    /// tested; only the Rust substrate and the bridge arm are new.
    /// Mirrors round-30 / 31 / 32 / 33 / 34 / 35 / 36 / 37
    /// unit-variant schema convention.
    MillTargetPlayerEqualToYourLifeTotal,
    /// Exile target permanent (CR 406). Removes from its current
    /// zone to its owner's exile zone. First-slice owner =
    /// controller.
    ExileTarget { target: TargetFilter },
    /// Tap target permanent (CR 701.20a). Idempotent on already-
    /// tapped targets. AnyPlayer target raises (you can't tap a
    /// player).
    TapTarget { target: TargetFilter },
    /// Untap target permanent (CR 701.20b). Idempotent on already-
    /// untapped targets. AnyPlayer target raises.
    UntapTarget { target: TargetFilter },
    /// Sacrifice target permanent (CR 701.16). First-slice: target
    /// is the active player's first creature on their own
    /// battlefield (you sacrifice your own permanents). AnyPlayer
    /// raises.
    Sacrifice { target: TargetFilter },
    /// Add `count` +1/+1 (positive) or -1/-1 (negative) counters to
    /// target creature on the defending player's battlefield. Routes
    /// through round-2's net-counter model — `add_counters` adds the
    /// signed amount to a single i32 net counter on the GameObject.
    /// AnyPlayer raises (you can't counter a player). Round-15 first
    /// slice picks the AnyCreature target deterministically as the
    /// first creature on the defending player's battlefield, mirroring
    /// the DealDamage / TapTarget / UntapTarget convention.
    AddCounters { target: TargetFilter, count: i32 },
    /// Target player discards one card from hand (CR 701.8). First
    /// slice: deterministic choice = first card in target's hand
    /// (defending player for opponent-targeted discards). AnyCreature
    /// raises (you can't discard a creature). Empty hand = silent
    /// no-op (Python parity).
    DiscardCard { target: TargetFilter },
    /// Scry N (CR 701.18): controller looks at the top `count` cards
    /// of their library and chooses any number to put on the bottom;
    /// the rest stay on top in any order. No target — controller
    /// implicit. Routes through round-4's apply_scry primitive.
    /// First-slice deterministic choice: keep all cards on top
    /// (no-op-shaped-as-scry; future slice will plumb a decision
    /// callback for "which to bottom"). Negative `count` raises.
    ScryN { count: i32 },
    /// Destroy target permanent (CR 701.7). Diverges from Sacrifice
    /// in two ways: (a) the target is on the DEFENDING player's
    /// battlefield (not the active player's), and (b) it bypasses
    /// "you sacrifice your own permanent" semantics. First slice
    /// picks the AnyCreature target as the defender's first creature,
    /// mirroring DealDamage / TapTarget convention. AnyPlayer raises
    /// (you can't destroy a player). Empty defender battlefield =
    /// fizzle silently. Indestructible (round-13) is honored — the
    /// destroy is suppressed exactly like in the SBA path.
    DestroyTarget { target: TargetFilter },
    /// **BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB
    /// (~14,108 17lands decks).** "When this creature enters, destroy
    /// up to one OTHER target creature. If that creature wasn't dealt
    /// damage this turn, its controller draws `draw_amount` cards."
    ///
    /// Structural sibling of `DestroyTarget { AnyCreature }`: same
    /// source-controller resolution (`current_resolve_source_controller`
    /// with active-player fallback), same `defender = 1 - controller`
    /// bf[0] first-slice deterministic target pick, same
    /// indestructible-suppress gate (CR 702.12), same LTB fan-out
    /// (`pending_leaves_dispatches.push` + `queue_token_cease`). Adds
    /// the Faller's-Faithful draw rider: BEFORE the destroy mutates
    /// anything, snapshot whether the chosen creature carries marked
    /// damage (`obj.damage == 0` → "wasn't dealt damage"); record the
    /// victim's controller; AFTER the destroy, if the snapshot said
    /// undamaged, the victim's controller draws `draw_amount` cards
    /// (CR 121.1 — the destroyed creature's controller, per the
    /// printed text). The "other target" / `exclude_self` shape is
    /// already enforced by the defender-bf scan (the source ETBs on
    /// the controller's bf; the defender scan never sees it).
    ///
    /// **Effect-half / damage-ledger split (see SEARCH_LIMITATIONS.md).**
    /// Python maintains a per-object `damaged_this_turn` boolean
    /// (`effects/base.py:300`, cleared on zone moves) that survives a
    /// heal-to-0; Rust carries only `obj.damage` (marked damage,
    /// cleared at cleanup). For the realistic Faller's Faithful ETB
    /// (instant-speed destroy with no prior damage marked on the
    /// target this turn) the two reads are byte-equal: a creature with
    /// no marked damage `obj.damage == 0` ⇔ Python's
    /// `not obj.damaged_this_turn`. The asymmetric corner —
    /// damaged-then-healed-to-0 within the same turn — is NOT modeled
    /// on the Rust side; the full general damaged-this-turn ledger is
    /// the loader-gated subsystem half (the card is NOT admitted to
    /// the production-deck loader allowlist). Mirrors Python's
    /// `DestroyTargetCreature(draw_if_undamaged_amount=draw_amount,
    /// optional=True, exclude_self=True)`
    /// (`effects/generic/destroy.py:94`).
    ///
    /// **Empty defender bf = silent fizzle** (the "up to one"
    /// optional shape collapses to a no-op when no legal target — CR
    /// 608.2c — and the draw rider does not run).
    DestroyTargetCreatureThenDrawIfUndamaged { draw_amount: i32 },
    /// Deal `amount` damage to each opponent of the controller
    /// (CR 701.4 + AOE pattern). Two-player engine first slice:
    /// damages defender (= the one opponent). No target — implicit.
    /// Negative `amount` raises. Composes with round-14's
    /// deal_damage_to_player primitive.
    DamageEachOpponent { amount: i32 },
    /// Deal `amount` damage to each creature on the battlefield
    /// (CR 701.4 — Pyroclasm/Anger of the Gods pattern). Damages all
    /// creatures regardless of controller; round-13 indestructible
    /// honored at SBA time. No target — implicit. Negative `amount`
    /// raises.
    DamageEachCreature { amount: i32 },
    /// Create `count` token creatures on the controller's battlefield
    /// (CR 701.6). The token's stats come from a pre-registered
    /// printing referenced by `token_printing_id` — caller registers
    /// the token-printing once (e.g. "1/1 white Soldier creature
    /// token") and references its id in the effect. Tokens enter
    /// summoning-sick like other creatures; future "tokens cease to
    /// exist when they leave the battlefield" CR 110.5g handling is
    /// deferred. Negative `count` raises.
    CreateToken { count: i32, token_printing_id: u32 },
    /// **BLOCKED-RUST-EFFECT slice (round 43) -- each living player
    /// creates `count` token creatures under their own control.**
    /// Sister of the static `CreateToken { count, token_printing_id }`
    /// arm above; the only behavioral difference is the recipient set
    /// (every living player, each under their own control, vs the
    /// source-controller alone). Mirrors Python's
    /// `CreateToken(token_printing=..., count=N, each_player=True)`
    /// (`effects/generic/tokens.py:198,210` — when `each_player=True`,
    /// `recipients = [p.id for p in state.players if not p.has_lost]`).
    ///
    /// EoE consumer: Edge Rover (Artifact Creature -- Robot Scout,
    /// "When this creature dies, each player creates a Lander
    /// token."). This is the FIRST each-player token shape in the
    /// audit's BLOCKED-RUST-EFFECT residue, and a precondition for
    /// the broader "each player creates X" family if future EoE
    /// expansions (or other set work) prints sister cards.
    ///
    /// **Mechanism -- recipient fanout.** Iterate over
    /// `0..state.players.len()`, skip any player with `has_lost ==
    /// true` (CR 102.5 living-players-only convention; mirrors
    /// Python's `if not p.has_lost` guard at tokens.py:212). For each
    /// living recipient, push `count` fresh tokens onto THEIR
    /// battlefield (NOT the source-controller's), tagged with
    /// `controller = recipient`. Each recipient gets the SAME static
    /// `count`. Tokens enter summoning-sick like every other ETB
    /// (mirrors the sister `CreateToken` arm's summoning_sick=true).
    ///
    /// **Schema parity with Python.** Mirrors the sister
    /// `CreateToken` arm's two-field schema; adds NO new field
    /// because the `each_player` boolean is encoded by variant name
    /// (closed-vocab convention -- bool fields are second-class in
    /// the bridge, every other binary choice lives in a separate
    /// variant). Python's `each_player=True` is the same shape
    /// (per-instance bool on the existing dataclass), but the bridge
    /// MAPS to a distinct dict `kind` for round-trip clarity.
    ///
    /// **Negative `count` raises** loudly (mirrors the sister
    /// `CreateToken` arm). `count == 0` is a legal no-op.
    /// **Unknown `token_printing_id` raises** at resolve time before
    /// any state mutation (mirrors the sister arm's whole-or-nothing
    /// validation contract -- one invalid registry id should not
    /// half-fanout some recipients before erroring).
    ///
    /// **First-slice scope (parity with the sister arm).**
    /// CR 614 token-creation replacement effects (Anointed Procession
    /// doubling, Moonlit-Meditation copy-replacement, Exalted Sunborn
    /// counter-stamping) and the `tapped` / `attacking` /
    /// `schedule_end_of_combat_sacrifice` flag bundle are NOT
    /// plumbed in this Rust arm -- the EoE consumer (Edge Rover ->
    /// Lander) doesn't engage any of those riders, and the sister
    /// `CreateToken` arm's first-slice docs explicitly defer the
    /// same surface for the same reason. Those richer surfaces
    /// still resolve through the Python path.
    ///
    /// **Each-player ENTERS dispatch.** Mirrors the sister
    /// `CreateToken` arm's `pending_etb_dispatches.extend(token_ids)`
    /// shape: every fresh token id is queued for ENTERS event
    /// dispatch by the surrounding resolver, regardless of which
    /// player it ended up under. The CE4 etb_timestamp stamping
    /// preserves a strict per-token monotonic order across all
    /// recipients combined (recipient iteration order =
    /// `state.players` index order = controller id order, mirroring
    /// Python's `for p in state.players` natural order).
    CreateTokenEachPlayer { count: i32, token_printing_id: u32 },
    /// **Phase 6 token-printing substrate -- Phase 4 (round 50): the
    /// `tapped` create-token rider.** Sister of the static
    /// `CreateToken { count, token_printing_id }` arm; the ONLY
    /// behavioral difference is that every fresh token enters the
    /// battlefield TAPPED (CR 701.6 + the "create a tapped ~" oracle
    /// rider). Mirrors Python's
    /// `CreateToken(token_printing=..., count=N, tapped=True)`
    /// (`effects/generic/tokens.py:286-287` -- the `if self.tapped:
    /// obj.tapped = True` branch).
    ///
    /// **EoE consumer: Gravpack Monoist** (Creature -- Robot, flying;
    /// "When this creature dies, create a tapped 2/2 Robot token.").
    /// 25,252 of the 129,996 17lands EoE decks run it -- the single
    /// highest real-card-coverage lever in the token-printing
    /// substrate after Phase 1 (its Robot token is already vanilla, so
    /// Phase 4 needs no Phase-3 Lander-activated-ability work). Round
    /// 49's `tapped=True` rider was DECLINED by the Phase-1 recognizer
    /// (`docs/TOKEN_PRINTING_SUBSTRATE.md` §2.3) precisely because the
    /// bare `CreateToken` variant has no `tapped` mirror -- this
    /// variant closes that decline.
    ///
    /// **Schema parity with Python -- sister-variant convention.**
    /// Mirrors the sister `CreateToken` / `CreateTokenEachPlayer`
    /// two-field schema (`count: i32`, `token_printing_id: u32`); adds
    /// NO new field because the `tapped` boolean is encoded by variant
    /// name (closed-vocab convention -- bool fields are second-class
    /// in the bridge, every binary choice rides on a separate variant;
    /// round-19/30/31/.. / round-43 `CreateTokenEachPlayer`
    /// precedent). Python's `tapped=True` is a per-instance bool on
    /// the existing dataclass, but the bridge MAPS it to a distinct
    /// dict `kind` (`create_token_tapped`) for round-trip clarity.
    ///
    /// **Negative `count` raises** loudly (mirrors the sister
    /// `CreateToken` arm). `count == 0` is a legal no-op. **Unknown
    /// `token_printing_id` raises** at resolve time before any state
    /// mutation (mirrors the sister arm's whole-or-nothing validation
    /// contract).
    ///
    /// **First-slice scope (parity with the sister arm).** CR 614
    /// token-creation replacement effects (Anointed Procession
    /// doubling, ...) and the `each_player` / `attacking` /
    /// `schedule_end_of_combat_sacrifice` riders are NOT plumbed here
    /// -- the EoE consumer (Gravpack Monoist -> a single tapped Robot
    /// on its own controller's battlefield) engages NONE of them, and
    /// the sister arms defer the same surface for the same reason.
    /// Those richer surfaces still resolve through the Python path.
    ///
    /// **ENTERS dispatch.** Mirrors the sister `CreateToken` arm's
    /// `pending_etb_dispatches.extend(token_ids)` shape: every fresh
    /// token id is queued for ENTERS event dispatch by the
    /// surrounding resolver. The CE4 etb_timestamp stamping preserves
    /// a strict per-token monotonic order. The token enters TAPPED but
    /// is otherwise identical to the sister arm's token (same
    /// summoning-sick=true, controller-of-effect resolution, fresh-id
    /// allocator).
    CreateTokenTapped { count: i32, token_printing_id: u32 },
    /// Controller adds `amount` mana of `color` to their mana pool.
    /// CR 106.1 — the resolution of a mana ability. Color is one of
    /// W/U/B/R/G/C; amount is non-negative. AddManaToPool is the
    /// second-most-frequent Effect across the EoE pool (33 cards per
    /// round-20 measurement), powering land mana abilities and
    /// mana-rocks like Mind Stone.
    ///
    /// **Schema simplification vs Python.** Python's `AddManaToPool`
    /// dataclass carries a `produces: tuple[tuple[str, int], ...]`
    /// and a `spend_restriction: str | None` — a single Python
    /// dataclass can mint multiple colors per resolution and tag
    /// each coin with a payment-filtering restriction (e.g.
    /// "spend only on artifact spells"). The Rust closed-vocabulary
    /// bridge models the same behavior with a flatter shape: one
    /// `AddManaToPool` per (color, amount) pair, no spend
    /// restriction. A printing that produces multi-color mana
    /// stacks multiple `AddManaToPool` entries in its `effects`
    /// Vec; spend-restriction tagging is deferred until the bridge
    /// schema needs it (no EoE card today combines mana production
    /// with the restriction tagging through this dispatch path).
    AddManaToPool { color: String, amount: i32 },
    /// Search controller's library for up to ``count`` basic lands,
    /// put each onto the battlefield tapped, then shuffle the
    /// library (CR 701.19 search + 701.22 shuffle). Powers Rampant
    /// Growth, Beamsaw Prospector, Kavaron Harrier — 15 cards in
    /// the EoE pool per round-20 measurement. The "basic land"
    /// filter and "tapped" arrival flag are part of the variant's
    /// spec (no field). The single `count` field mirrors Python's
    /// `SearchLibraryForBasicLandTapped(count: int = 1)` dataclass
    /// shape — a count > 1 finds up to `count` basic lands one at
    /// a time (Larval Scoutlander / cycling-land bodies).
    ///
    /// **First-slice scope:**
    /// - **Deterministic first-match pick.** Scans the library
    ///   top-to-bottom (Vec iter().rev()) and grabs the topmost
    ///   matching object. Real MTG lets the player choose any
    ///   matching card; that's a future plumbing decision once
    ///   decision callbacks exist.
    /// - **Basic land filter = card_types contains "Land".** The
    ///   Rust bridge has no `supertypes` field today, so the
    ///   "basic" half of "basic land" is approximated by the
    ///   `card_types` Land check. In the EoE pool every basic land
    ///   is exactly card_types=["Land"] with no compound non-basic
    ///   land carrying this dispatch, so the proxy is faithful for
    ///   the current set. See SEARCH_LIMITATIONS entry for the
    ///   deferred supertypes work.
    /// - **No shuffle.** CR 701.19 says searches always shuffle the
    ///   library regardless of search outcome. This slice
    ///   deliberately does NOT shuffle — the deterministic first-
    ///   match pick makes the dispatch reproducible without
    ///   threading an RNG seed through `apply_effect`. The
    ///   `search_library_for_first_of_type_and_shuffle` primitive
    ///   in `state_ops_library.rs` is the right composition for a
    ///   future slice once RNG-externalization lands; see
    ///   SEARCH_LIMITATIONS entry.
    /// - **count<0 raises** loudly (mirrors ScryN / DealDamage /
    ///   DamageEach* / CreateToken raise-on-negative shape).
    /// - **count=0 is a legal no-op** (mirrors GainLife / DrawCards
    ///   / Mill / CreateToken zero-count conventions).
    /// - **Empty library / count > matching lands** stops early
    ///   silently — partial fill is allowed (Python's
    ///   `next(... None)` short-circuit on miss).
    SearchLibraryForBasicLandTapped { count: i32 },
    /// Put `amount` +1/+1 counters on the resolving spell's source
    /// permanent — the "this creature" / "self" target (CR 122.6 +
    /// 701.13). Powers Blade of the Swarm, Atomic Microsizer — 16
    /// cards in the EoE pool per round-20 measurement.
    ///
    /// **"Self" semantics.** The recipient is the source GameObject
    /// at the top of the stack (i.e. the resolving spell's source).
    /// This is distinct from `AddCounters`, which deterministically
    /// picks the DEFENDING player's first creature. Mirrors Python's
    /// `PutPlusOneCounterOnSelf` which reads
    /// `state.resolving_source` during ability resolution.
    ///
    /// **Counter shape.** `amount` is the number of +1/+1 counters
    /// to add; it lands as a positive delta on the round-2 signed
    /// net-counter i32 field via `saturating_add`. Default is 1
    /// (mirrors Python's `amount: int = 1` dataclass default).
    /// Negative `amount` raises at register / resolve time —
    /// "put -N +1/+1 counters on self" isn't a sensible printing.
    ///
    /// **Source-finder dispatch.** Round-23's earlier arms (AddMana
    /// to Pool, SearchLibraryForBasicLandTapped, AddCounters,
    /// DamageEachCreature, etc.) get their target from
    /// `state.active_player` or from the defending player's
    /// battlefield Vec; "self" needs the actual source object. The
    /// resolve loop owns `top.source_obj_id` and threads it into
    /// `apply_effect` so this arm can locate the source across
    /// zones. The other arms ignore it (the param is opaque to
    /// them).
    ///
    /// **Source-not-on-battlefield handling.** Real MTG: if the
    /// source has left the battlefield by the time the counter is
    /// added (e.g. the creature was destroyed in response), the
    /// effect "puts the counter on nothing" / fizzles. First slice
    /// honors that: if the source is not on a battlefield zone, the
    /// counter add is a silent no-op (the source-finder walks both
    /// players' battlefields; sources sitting in hand / library /
    /// graveyard / stack are NOT counter targets). Mirrors Python's
    /// `source is None` short-circuit in the dataclass's `resolve`.
    ///
    /// **amount=0 is a legal no-op** (mirrors GainLife / DrawCards /
    /// Mill / SearchLibraryForBasicLandTapped / CreateToken
    /// zero-amount conventions).
    PutPlusOneCounterOnSelf { amount: i32 },
    /// Surveil N (CR 701.42 / equivalent): controller looks at the top
    /// `count` cards of their library, then chooses any number to put
    /// into their graveyard. The rest stay on top of the library in
    /// any order. Powers Appendage Amalgam, etc. — 16 cards in the EoE
    /// pool per round-20 measurement. Sibling of ScryN (round 16) —
    /// same shape, different destination zone (graveyard not bottom).
    ///
    /// First-slice deterministic choice: put NONE into the graveyard;
    /// keep all `count` cards on top in their current order. This is
    /// the safest deterministic default (mirrors ScryN's "keep all on
    /// top" first-slice choice from round 16). Future slice plumbs a
    /// real "which to graveyard" decision callback through. Negative
    /// count raises.
    Surveil { count: i32 },
    /// Counter target spell on the stack (CR 701.5). Powers Annul,
    /// Cancel, etc. — 7 cards in the EoE pool per round-20 measurement.
    ///
    /// **Python schema simplification.** Python's `CounterTargetSpell`
    /// dataclass carries a `filter_types: frozenset[str] = frozenset()`
    /// field — an empty set means "any spell", non-empty restricts what
    /// card types are legal targets (e.g. `frozenset({"Artifact",
    /// "Enchantment"})` for Annul). The Rust first slice does NOT
    /// carry that field — assumes target is "any spell". Refined-target
    /// variants (counter creature spell only, counter non-creature,
    /// etc.) wait for richer target plumbing on the Rust side, the
    /// same deferral shape as TargetFilter for the rest of the variant
    /// family.
    ///
    /// **First-slice deterministic choice: pick the TOP stack item
    /// that isn't this Effect's own source.** Mirrors the "first-
    /// eligible" convention from existing Effect arms (DealDamage
    /// picks defender's battlefield[0], AddCounters likewise). The
    /// self-exclusion guard uses the round-23 `source_obj_id`
    /// parameter threaded through `apply_effect`: a counter-spell
    /// resolving on the stack shouldn't counter itself before it
    /// resolves. If the stack is empty (or only contains self), the
    /// effect fizzles silently — Python parity with the
    /// `if not targets: return` short-circuit in
    /// `CounterTargetSpell.resolve`.
    ///
    /// **Composition with round-7 counter_spell primitive.** The
    /// round-7 `state_ops_counterspell.rs::counter_spell(target_obj_id)`
    /// primitive removes the target from the stack and moves its
    /// source to the controller's graveyard (CR 701.6). Per the
    /// per-slice file-isolation pattern + the E0624 cross-file
    /// `#[pymethods]` friction, this arm inlines the equivalent
    /// logic rather than calling through to the primitive: find the
    /// stack item by source_obj_id, remove it, find the source
    /// GameObject in any tracked zone, move it to the stack item
    /// controller's graveyard. A follow-up consolidation slice can
    /// collapse onto the shared primitive once cross-file private-fn
    /// calls re-open.
    ///
    /// **No "can't be countered" enforcement.** Python's
    /// `_is_uncounterable` checks `spell_is_uncounterable` on the
    /// printing and `grants_creature_spells_uncounterable` on
    /// battlefield static abilities. The Rust bridge has neither
    /// flag today — deferred along with the rest of the
    /// `cant_be_countered` shape (mirrors the round-7 counter_spell
    /// primitive's same deferral).
    CounterTargetSpell,
    /// Return a creature card from your graveyard to the battlefield
    /// (CR 400.7 + 302.1 — reanimation). Powers Emerge from the Cocoon,
    /// Reanimate-style sugar, etc. — 8 cards in the EoE pool per
    /// round-20 measurement.
    ///
    /// **First-slice deterministic target choice.** The recipient is
    /// the FIRST creature card encountered scanning the controller's
    /// (= `state.active_player`'s, by first-slice convention) graveyard
    /// in zone-iteration order (front = oldest, smallest Vec index).
    /// "Creature card" = the printing's `card_types` contains
    /// "Creature". Real MTG lets the player choose any creature card
    /// in the graveyard as the target; richer target plumbing is a
    /// future slice once decision callbacks exist.
    ///
    /// **Python schema simplification.** Python's
    /// `ReturnCreatureFromYourGraveyardToBattlefield` extends the
    /// unified `ReturnFromGraveyard` dataclass which carries a number
    /// of optional widening / capping fields (`or_enchantment`,
    /// `or_spacecraft`, `or_artifact`, `max_mv`, `max_n`, etc.). The
    /// Rust first slice carries NONE of these — the variant is a unit
    /// shape ("first creature card in your graveyard"). Refined-target
    /// variants ("creature with mana value <= N", "creature or
    /// enchantment", multi-pick) wait for richer target plumbing on
    /// the Rust side, same deferral shape as TargetFilter / filter_types.
    ///
    /// **Empty / no-creature graveyard: silent fizzle.** Mirrors the
    /// `if not targets: return` short-circuit shape every other
    /// graveyard-walk arm uses (and the round-17
    /// `move_object_from_graveyard_to_battlefield` primitive's
    /// `Ok(false)` return on absent obj_id).
    ///
    /// **Composition with round-17 primitive.** The round-17
    /// `state_ops_zones.rs::move_object_from_graveyard_to_battlefield(
    /// player_id, obj_id)` primitive does exactly the recipient-side
    /// work this Effect needs (remove from graveyard, push onto
    /// battlefield with summoning_sick=true + per-object state reset
    /// per CR 400.7). Routing through it would require
    /// cross-`#[pymethods]` plumbing inside `apply_effect` where we
    /// already hold `state: &mut GameState` — the same E0624 friction
    /// every other Effect arm hit. Per the recurring per-slice
    /// file-isolation pattern (DealDamage / ReturnToHand / Mill /
    /// DrawCards / ExileTarget / Sacrifice / DestroyTarget / TapTarget
    /// / UntapTarget / AddCounters / DiscardCard / CreateToken /
    /// AddManaToPool / SearchLibraryForBasicLandTapped /
    /// PutPlusOneCounterOnSelf / CounterTargetSpell all inline their
    /// primitives for the same reason), this arm inlines the
    /// equivalent logic. A follow-up consolidation slice can collapse
    /// onto the shared primitive once cross-file private-fn calls
    /// re-open.
    ReturnCreatureFromYourGraveyardToBattlefield,
    /// **Phase 5 rider-widened return-to-battlefield (Option B —
    /// additive bespoke arm).** Return the first creature OR
    /// enchantment card from the source-controller's graveyard to
    /// that player's battlefield. Mirrors Python's
    /// `ReturnCreatureFromYourGraveyardToBattlefield(or_enchantment=True)`
    /// (`graveyard.py:361`) — the rider widens the legal-target
    /// pool to admit enchantment cards alongside creatures, and the
    /// destination is the BATTLEFIELD (vs the to-hand sibling which
    /// pushes to hand).
    ///
    /// **EoE consumer (audit).** Rescue Skiff (EoE): "When this
    /// Spacecraft enters, return target creature or enchantment
    /// card from your graveyard to the battlefield." Unlike the
    /// to-hand sibling arm (where Rescue Skiff does NOT consume),
    /// THIS arm is the direct unblocker for Rescue Skiff itself.
    ///
    /// **Filter contract.** A graveyard card qualifies iff its
    /// printing's `card_types` contains "Creature" OR "Enchantment".
    /// (CR 400 / 506 — type predicates read off the printing's
    /// static card_types; the Rust registry stores card_types as
    /// the Scryfall type-line components on the LHS of the em-dash.)
    /// Sister enchantment-creatures (printings carrying BOTH
    /// "Enchantment" AND "Creature") match the disjunction at the
    /// first hit and are pulled. Cards with neither type are
    /// skipped. The scan preserves the unit-variant's iterator
    /// order (`state.players[controller_idx].graveyard.iter().find_map`),
    /// so test predictability matches the existing arms exactly.
    ///
    /// **Controller-of-effect.** Inherits the I83 contract (same
    /// pattern as the bare to-bf unit variant): `controller_idx`
    /// derives from `current_resolve_source_controller`, falling
    /// back to `active_player` for direct `apply_effect_pub`
    /// callers. The graveyard scan, the destination battlefield
    /// push, AND the GameObject.controller stamp on the
    /// reanimated permanent all anchor to the source's controller —
    /// cross-controller routing (p1's ETB resolving on p0's turn)
    /// Just Works.
    ///
    /// **Zone-change reset (CR 400.7) + summoning sickness (CR 302.1).**
    /// Same as the bare to-bf unit variant: `tapped`,
    /// `summoning_sick=true`, `damage`, `counters`, `attached_to`,
    /// `exiled_with_source`, `loyalty_counters` all reset on the
    /// move to battlefield. A fresh `etb_timestamp` is reserved
    /// before the player-mut borrow (CE4 closure). The reanimated
    /// permanent fires its ENTERS triggers via the deferred-
    /// dispatch queue (`pending_etb_dispatches.push(entered_id)`)
    /// regardless of recipient bf, mirroring the unit variant.
    /// Walker-shape printings reseed `loyalty_counters` to the
    /// printed `starting_loyalty` (CE20 Slice 5).
    ///
    /// **Schema simplification vs Python.** Drops `max_mv` and the
    /// other rider flags (`or_artifact`, `or_spacecraft`,
    /// untargeted_all, plus_one_counter, log_suffix). Bare
    /// `or_enchantment` only — Rescue Skiff carries no max_mv cap
    /// or +1/+1-counter rider on the printed text, so the bare
    /// rider arm is sufficient for the unique EoE consumer.
    ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield,
    /// "You discard a card" (CR 701.8). The SOURCE's controller
    /// discards `amount` cards from their hand. Punisher / drawback
    /// shape (Irreverent Gremlin pattern) — 8 cards in the EoE pool
    /// per round-20 measurement.
    ///
    /// **Distinct from `DiscardCard { target: AnyPlayer }`.**
    /// `DiscardCard` deterministically targets the DEFENDING player
    /// (Mind Rot / Wrench Mind shape — the active player forces the
    /// opponent to discard). `DiscardSelf` is the SOURCE's own
    /// controller — the punisher fires on whoever cast / controls
    /// the source, regardless of active_player.
    ///
    /// **Source-finder dispatch (round-23 pattern).** Locates the
    /// source by `source_obj_id` threaded through `apply_effect`.
    /// The scan walks both players' hand / library / battlefield /
    /// graveyard / exile zones to find the source's `controller`
    /// (this is how DiscardSelf differs from
    /// `PutPlusOneCounterOnSelf`, which restricts the source to the
    /// battlefield — the punisher pattern fires regardless of where
    /// the source has moved to, since the controller doesn't change
    /// when a permanent leaves the battlefield). Additionally checks
    /// the stack — the source may still be resolving (e.g. an
    /// instant / sorcery / triggered ability that just fired with
    /// `you discard a card` as its effect).
    ///
    /// **First-slice deterministic choice: discard hand[0]** of the
    /// source's controller (zone-iteration order, mirrors the round-15
    /// `DiscardCard` arm's hand[0] convention). Empty hand = silent
    /// no-op (Python parity with `if not p.hand: break` in
    /// `DiscardSelf.resolve`). Source not found anywhere = silent
    /// no-op (lost-track-of-source tolerance, mirrors
    /// `PutPlusOneCounterOnSelf`'s `source is None` short-circuit).
    ///
    /// **`amount` shape mirrors Python.** Python's
    /// `DiscardSelf(amount: int = 1)` carries an `amount: int = 1`
    /// default; this Rust variant mirrors that exactly. The
    /// dispatch resolves up to `amount` discards in a row (stop
    /// early on empty hand, matching the Python `for _ in
    /// range(self.amount): if not p.hand: break` shape). Negative
    /// `amount` raises at register / resolve time.
    DiscardSelf { amount: i32 },
    /// Each opponent of the source's controller loses `amount` life
    /// (CR 118.4 — life loss, not damage). Punisher / drawback shape
    /// (Balemurk Leech pattern) — 6 cards in the EoE pool per
    /// round-20 measurement.
    ///
    /// **Distinct from `DamageEachOpponent` (round 17).** Damage
    /// routes through damage-prevention / Worship-style replacement
    /// effects (CR 614 — damage replacement); life-loss directly
    /// subtracts from `players[idx].life` and is NOT reducible by
    /// damage-prevention. Two-player engine first slice: both arms
    /// look identical in their net effect on the defender's life
    /// total, but the variants stay distinct to mirror Python's
    /// closed `Effect` vocabulary (round-15 pattern). Future slices
    /// that wire damage-prevention through DealDamage /
    /// DamageEachOpponent will keep this arm a direct subtraction.
    ///
    /// **Source-finder dispatch (round-23 pattern).** Threads
    /// `source_obj_id` through `apply_effect` and walks both
    /// players' five tracked zones (hand / library / battlefield /
    /// graveyard / exile) plus the stack to recover the source's
    /// `controller`. Mirrors `DiscardSelf` (round-24) — punisher
    /// effects fire on whoever controls the source regardless of
    /// where the source currently sits, since the `controller` field
    /// doesn't change on zone moves. Source not found anywhere =
    /// silent fizzle (lost-track-of-source tolerance).
    ///
    /// **Two-player "each opponent" reduces to "the one opponent".**
    /// The opponent of player N is `1 - N`. The single opponent's
    /// `life` field is decremented by `amount`. Negative life is
    /// handled by the SBA loop (round-5+); no clamp here.
    ///
    /// **amount=0 is a legal no-op** (mirrors GainLife / DrawCards /
    /// Mill / DamageEachOpponent / DiscardSelf zero-amount
    /// conventions). **amount<0 raises** loudly (mirrors round-3
    /// DealDamage / round-17 DamageEachOpponent / round-24
    /// DiscardSelf raise-on-negative shape; "lose -2 life" is
    /// schema nonsense).
    EachOpponentLosesLife { amount: i32 },
    /// **BLOCKED-RUST-EFFECT slice (round 44) -- each opponent of the
    /// source-controller loses life equal to the number of ARTIFACTS the
    /// source-controller controls.** Mirrors Python's
    /// `EachOpponentLosesLifeEqualToArtifactCount`
    /// (`effects/generic/life.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-26's static
    /// `EachOpponentLosesLife { amount }` -- same recipient set + dispatch
    /// (each opponent of the source-controller, direct life subtraction
    /// CR 118.4 not damage), NEW dynamic-X denominator (artifact-count
    /// instead of static `amount`). FIFTH apply substrate to receive the
    /// X-from-count plumbing established by round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount` (damage), round-33's
    /// `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact` (counters),
    /// round-34's `SurveilXEqualToArtifactCount` (surveil), round-42's
    /// `MillTargetPlayerEqualToYourLifeTotal` (mill -- different
    /// denominator). Walks the apply-substrate axis to a fifth entry
    /// (life-loss-each-opponent), confirming the round-35
    /// bidirectionally-extensible X-from-count plumbing generalizes
    /// across both the substrate axis (damage / counters / surveil / mill
    /// / life-loss) and the denominator axis (artifact-count workhorse
    /// across damage / counters / surveil now also drives life-loss).
    ///
    /// **Mechanism -- count + apply.** At resolve time, count the
    /// source-controller's artifacts on bf
    /// (`printing.card_types.iter().any(|t| t == "Artifact")`) -- matches
    /// Python's `sum(1 for o in state.battlefield if o.controller_id ==
    /// controller_id and o.printing.is_artifact)` shape (mirrors the
    /// round-32 / round-34 artifact-count denominator BIT-FOR-BIT --
    /// same scan, same predicate, same `saturating_add` overflow guard).
    /// Lands that are also artifacts count, tokens like Lander count.
    /// Dual-typed artifact-creatures count for BOTH the artifact-count
    /// denominator and any creatures-controlled denominator
    /// independently (Scryfall lists both "Artifact" and "Creature" in
    /// card_types; checked independently per arm). The
    /// source-controller's OWN artifacts count toward the denominator if
    /// the source is itself an Artifact still on the bf at resolve time
    /// (mirrors Python's vanilla walk over `state.battlefield` -- no
    /// source-exclusion clause, mirrors round-32 / round-34 sister
    /// convention). If the artifact count is <= 0 -> silent no-op
    /// (matches Python's `if artifact_count <= 0: return` short-circuit
    /// before the per-opponent loop, also mirrors GainLife / DrawCards /
    /// Mill / EachOpponentLosesLife / DiscardSelf zero-amount no-op
    /// convention).
    ///
    /// **Apply -- each opponent loses `count` life.** In 2-player land
    /// this is "the one opponent" at `1 - controller`. Direct life
    /// subtraction via `saturating_sub` -- not damage; not reducible by
    /// damage-prevention / Worship-style replacement effects (CR 614);
    /// negative life is handled by the SBA loop on both engines, no
    /// clamp here. Mirrors the round-26 sister
    /// `Effect::EachOpponentLosesLife` apply step BIT-FOR-BIT, only the
    /// amount source differs (dynamic-X count vs static field).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors the round-26 sister `EachOpponentLosesLife` shape +
    /// every other dynamic-X arm exactly -- needs to resolve even if
    /// source has left bf, e.g. counterspelled trigger that still drives
    /// the count from the original ETB snapshot). If untrackable ->
    /// silent fizzle (lost-track-of-source tolerance, mirrors
    /// EachOpponentLosesLife / DiscardSelf shape).
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// no constructor args (the dynamic-X is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 / round-42 unit-variant schema convention.
    EachOpponentLosesLifeEqualToArtifactCount,
    /// **BLOCKED-RUST-EFFECT slice (round 48) -- each opponent of the
    /// source-controller loses life equal to the source-controller's
    /// CURRENT LIFE TOTAL.** Mirrors Python's
    /// `EachOpponentLosesLifeEqualToYourLifeTotal`
    /// (`effects/generic/life.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-44's
    /// `EachOpponentLosesLifeEqualToArtifactCount` -- SAME recipient set
    /// + dispatch (each opponent of the source-controller, direct life
    /// subtraction CR 118.4 not damage) and SAME apply substrate
    /// (life-loss-each-opponent), only the DENOMINATOR read differs
    /// (source-controller's current life total instead of their
    /// artifact count). This is the SECOND denominator on the
    /// life-loss-each-opponent apply substrate (artifact-count round-44
    /// -> life-total round-48); the round-42
    /// `MillTargetPlayerEqualToYourLifeTotal` (mill substrate) and
    /// round-47 `DealDamageToOpponentCreatureByLifeTotal` (damage
    /// substrate) are the life-total-denominator precedents on OTHER
    /// substrates. Walks the X-from-count matrix to a third
    /// (substrate, life-total) cell.
    ///
    /// **Mechanism -- read + apply.** At resolve time, read the
    /// source-controller's current life total and clamp negative life
    /// to 0 (`life.max(0)`) -- matches Python's `max(0, life)` clamp
    /// and the round-42 `MillTargetPlayerEqualToYourLifeTotal` /
    /// round-47 `DealDamageToOpponentCreatureByLifeTotal` life-total-
    /// denominator clamping precedent (negative life can arise from a
    /// prior loss/damage in the same resolution window before the SBA
    /// loop runs; a negative denominator would be nonsensical). If the
    /// clamped life is 0 -> silent no-op (matches Python's `if amount
    /// <= 0: return` short-circuit before the per-opponent loop, also
    /// mirrors GainLife / DrawCards / Mill / EachOpponentLosesLife /
    /// EachOpponentLosesLifeEqualToArtifactCount / DiscardSelf
    /// zero-amount no-op convention).
    ///
    /// **Apply -- each opponent loses `life` life.** In 2-player land
    /// this is "the one opponent" at `1 - controller`. Direct life
    /// subtraction via `saturating_sub` -- not damage; not reducible by
    /// damage-prevention / Worship-style replacement effects (CR 614);
    /// negative life is handled by the SBA loop on both engines, no
    /// clamp on the APPLY side here (the clamp is on the DENOMINATOR
    /// read only). Mirrors the round-44 sister
    /// `Effect::EachOpponentLosesLifeEqualToArtifactCount` apply step
    /// BIT-FOR-BIT, only the amount source differs (life-total read vs
    /// artifact-count walk).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors the round-44 sister
    /// `EachOpponentLosesLifeEqualToArtifactCount` shape exactly --
    /// needs to resolve even if source has left bf, e.g.
    /// counterspelled trigger that still drives the count from the
    /// original ETB snapshot). If untrackable -> silent fizzle
    /// (lost-track-of-source tolerance, mirrors
    /// EachOpponentLosesLife / DiscardSelf shape). NOTE: unlike the
    /// round-42 mill arm (which leans on
    /// `current_resolve_source_controller`), the dynamic-X each-opponent
    /// arms use the 5-zone walk for cross-arm consistency with the
    /// round-44 sister and so the inject-and-fire-ENTERS synthetic
    /// harness (where the resolve-harness shortcut isn't set) resolves
    /// correctly.
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// no constructor args (the dynamic-X is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 / round-42 / round-44 / round-45 / round-47
    /// unit-variant schema convention.
    EachOpponentLosesLifeEqualToYourLifeTotal,
    /// **BLOCKED-RUST-EFFECT slice (round 45) -- the source's
    /// controller GAINS life equal to the number of ARTIFACTS the
    /// source-controller controls.** Mirrors Python's
    /// `GainLifeEqualToArtifactCount` (`effects/generic/life.py` --
    /// shipped in this slice; no pre-existing class). Sister of
    /// round-26's static `GainLife { amount }` -- same recipient
    /// (the source's controller, NOT the opponent) + same apply step
    /// (direct life addition CR 119.3, not damage healing), NEW
    /// dynamic-X denominator (artifact-count instead of static
    /// `amount`). SIXTH apply substrate to receive the X-from-count
    /// plumbing established by round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount` (damage),
    /// round-33's `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact`
    /// (counters), round-34's `SurveilXEqualToArtifactCount` (surveil),
    /// round-42's `MillTargetPlayerEqualToYourLifeTotal` (mill --
    /// different denominator), and round-44's
    /// `EachOpponentLosesLifeEqualToArtifactCount` (life-loss-each-
    /// opponent). Walks the apply-substrate axis to a SIXTH entry
    /// (gain-life-controller-self), and is the controller-self
    /// MIRROR of the round-44 each-opponent-life-loss arm: same
    /// denominator (artifact-count workhorse), same dispatch shape,
    /// inverted recipient (controller-self vs opponents-of-controller)
    /// AND inverted sign (gain vs loss).
    ///
    /// **Mechanism -- count + apply.** At resolve time, count the
    /// source-controller's artifacts on bf
    /// (`printing.card_types.iter().any(|t| t == "Artifact")`) -- matches
    /// Python's `sum(1 for o in state.battlefield if o.controller_id ==
    /// controller_id and o.printing.is_artifact)` shape (mirrors the
    /// round-32 / round-34 / round-44 artifact-count denominator
    /// BIT-FOR-BIT -- same scan, same predicate, same `saturating_add`
    /// overflow guard). Lands that are also artifacts count, tokens
    /// like Lander count. Dual-typed artifact-creatures count for BOTH
    /// the artifact-count denominator and any creatures-controlled
    /// denominator independently (Scryfall lists both "Artifact" and
    /// "Creature" in card_types; checked independently per arm). The
    /// source-controller's OWN artifacts count toward the denominator
    /// if the source is itself an Artifact still on the bf at resolve
    /// time (mirrors Python's vanilla walk over `state.battlefield` --
    /// no source-exclusion clause, mirrors round-32 / round-34 /
    /// round-44 sister convention). If the artifact count is <= 0 ->
    /// silent no-op (matches Python's `if artifact_count <= 0: return`
    /// short-circuit before the per-controller life-add, also mirrors
    /// GainLife / DrawCards / Mill / EachOpponentLosesLife /
    /// DiscardSelf zero-amount no-op convention).
    ///
    /// **Apply -- the source's controller gains `count` life.** Direct
    /// life addition via simple `+=` (matches the round-26 sister
    /// `Effect::GainLife { amount }` apply step BIT-FOR-BIT -- only the
    /// amount source differs, dynamic-X count vs static field). Not
    /// damage; not amplifiable by life-gain triggers' damage-prevention
    /// hooks (CR 119.3 distinguishes life gain from damage prevention).
    /// In the engine's two-player land the recipient is exactly the
    /// source-controller (= the caster on a typical ETB).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors the round-26 sister `EachOpponentLosesLife` /
    /// round-44 `EachOpponentLosesLifeEqualToArtifactCount` shape +
    /// every other dynamic-X arm exactly -- needs to resolve even if
    /// source has left bf, e.g. counterspelled trigger that still
    /// drives the count from the original ETB snapshot). If
    /// untrackable -> silent fizzle (lost-track-of-source tolerance,
    /// mirrors EachOpponentLosesLife / DiscardSelf shape). Bare
    /// `Effect::GainLife { amount }` uses
    /// `state.current_resolve_source_controller` as a shortcut (set by
    /// the standard resolve harness), but the dynamic-X arm needs the
    /// 5-zone walk because the count denominator must read from the
    /// source's ACTUAL controller's bf, not whoever happens to be the
    /// resolve-time active player.
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored on
    /// the variant. Python's `@dataclass(frozen=True)` class has no
    /// constructor args (the dynamic-X is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 / round-42 / round-44 unit-variant schema
    /// convention.
    GainLifeEqualToArtifactCount,
    /// **BLOCKED-RUST-EFFECT slice (round 49) -- the source's
    /// controller GAINS life equal to the source-controller's CURRENT
    /// LIFE TOTAL.** Mirrors Python's `GainLifeEqualToYourLifeTotal`
    /// (`effects/generic/life.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-45's
    /// `GainLifeEqualToArtifactCount` -- SAME recipient (the source's
    /// controller, NOT the opponent) + SAME apply step (direct life
    /// addition via the `gain_life` helper CR 119.3, not damage
    /// healing) and SAME apply substrate (gain-life-controller-self),
    /// only the DENOMINATOR read differs (source-controller's current
    /// life total instead of their artifact count). This is the SECOND
    /// denominator on the gain-life-controller-self apply substrate
    /// (artifact-count round-45 -> life-total round-49); the round-42
    /// `MillTargetPlayerEqualToYourLifeTotal` (mill substrate), round-47
    /// `DealDamageToOpponentCreatureByLifeTotal` (damage substrate) and
    /// round-48 `EachOpponentLosesLifeEqualToYourLifeTotal`
    /// (life-loss-each-opponent substrate) are the life-total-
    /// denominator precedents on OTHER substrates. Walks the
    /// X-from-count matrix to a FOURTH (substrate, life-total) cell.
    ///
    /// **Mechanism -- read + apply.** At resolve time, read the
    /// source-controller's current life total and clamp negative life
    /// to 0 (`life.max(0)`) -- matches Python's `max(0, life)` clamp
    /// and the round-42 `MillTargetPlayerEqualToYourLifeTotal` /
    /// round-47 `DealDamageToOpponentCreatureByLifeTotal` / round-48
    /// `EachOpponentLosesLifeEqualToYourLifeTotal` life-total-
    /// denominator clamping precedent (negative life can arise from a
    /// prior loss/damage in the same resolution window before the SBA
    /// loop runs; a negative denominator would be nonsensical). If the
    /// clamped life is 0 -> silent no-op (matches Python's `if amount
    /// <= 0: return` short-circuit before the life-add, also mirrors
    /// GainLife / DrawCards / Mill / EachOpponentLosesLife /
    /// GainLifeEqualToArtifactCount / DiscardSelf zero-amount no-op
    /// convention).
    ///
    /// **Apply -- the source's controller gains `life` life.** Direct
    /// life addition via the `gain_life` helper (mirrors the round-45
    /// sister `Effect::GainLifeEqualToArtifactCount` apply step
    /// BIT-FOR-BIT -- only the amount source differs, life-total read
    /// vs artifact-count walk). Not damage; not amplifiable by
    /// life-gain triggers' damage-prevention hooks (CR 119.3
    /// distinguishes life gain from damage prevention). In the engine's
    /// two-player land the recipient is exactly the source-controller
    /// (= the caster on a typical ETB).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors the round-45 sister `GainLifeEqualToArtifactCount`
    /// shape exactly -- needs to resolve even if source has left bf,
    /// e.g. counterspelled trigger that still drives the count from the
    /// original ETB snapshot). If untrackable -> silent fizzle
    /// (lost-track-of-source tolerance, mirrors
    /// GainLifeEqualToArtifactCount / EachOpponentLosesLife /
    /// DiscardSelf shape). Bare `Effect::GainLife { amount }` uses
    /// `state.current_resolve_source_controller` as a shortcut (set by
    /// the standard resolve harness), but the dynamic-X arm needs the
    /// 5-zone walk because the life-total denominator must read from
    /// the source's ACTUAL controller, not whoever happens to be the
    /// resolve-time active player.
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored on
    /// the variant. Python's `@dataclass(frozen=True)` class has no
    /// constructor args (the dynamic-X is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 / round-42 / round-44 / round-45 / round-47
    /// / round-48 unit-variant schema convention.
    GainLifeEqualToYourLifeTotal,
    /// The SOURCE's controller loses `amount` life (CR 118.4 — life
    /// loss, not damage). 7 cards in EoE pool per round-20 measurement
    /// (Decode Transmissions etc.). Punisher / drawback pattern — the
    /// caster pays a life cost on resolution.
    ///
    /// **Distinct from `EachOpponentLosesLife` (round 24)** — that arm
    /// subtracts from the OPPONENT's life. This arm subtracts from the
    /// SOURCE's CONTROLLER's life (typically the caster). Both use the
    /// same source-finder pattern (round-24 5-zone walk + stack scan).
    ///
    /// **Distinct from `GainLife` with negative `amount`** — that
    /// pattern raises ("gain -2 life" is schema nonsense). `LoseLife`
    /// is the closed-vocab variant for losing life; negative `amount`
    /// here also raises (defense-in-depth).
    ///
    /// amount=0 is a legal no-op. Negative amount raises.
    LoseLife { amount: i32 },
    /// Target opponent loses `amount` life (CR 118.4). 2 cards in EoE
    /// pool per round-20 measurement (Dubious Delicacy etc.).
    ///
    /// **Distinct from `EachOpponentLosesLife` (round 24)** — that arm
    /// hits ALL opponents (in 2-player land = the one opponent).
    /// `TargetOpponentLosesLife` is targeted: the caster CHOOSES which
    /// opponent loses life. In a 2-player engine these collapse to the
    /// same runtime effect, but the variants stay distinct to mirror
    /// Python's closed Effect vocabulary.
    ///
    /// **First-slice deterministic choice**: the one opponent (= 1 -
    /// controller). Future multiplayer / archenemy work needs an
    /// explicit choice callback. Negative `amount` raises; amount=0
    /// is a legal no-op.
    TargetOpponentLosesLife { amount: i32 },
    /// **PPP Pin 1 (10th Rust slice).** Target player loses `amount`
    /// life. Mirrors Python's `TargetPlayerLosesLife(amount: int = 1)`
    /// (`effects/generic/life.py:80`).
    ///
    /// **Pre-fixed-target variant.** Distinct from
    /// `TargetOpponentLosesLife` (which auto-picks the defending
    /// opponent at resolve time): this variant reads the target
    /// player id from `current_resolve_payload_target_id` (set at
    /// dispatch time by the trigger that knows which player to hit,
    /// e.g. the DISCARDS_CARD `fixed_targets` plumbing for Entropic
    /// Battlecruiser's 1+ tier).
    ///
    /// **First-slice resolve.** Reads the payload-target slot; if
    /// the slot is `Some(pid)` and `pid` resolves to a valid player
    /// id, subtracts `amount` from that player's life. Unset payload
    /// or invalid pid → silent fizzle (mirrors the broader "no target
    /// supplied" handling in Python's `TargetPlayerLosesLife.resolve`
    /// at `effects/generic/life.py:90` which early-returns on empty
    /// targets).
    ///
    /// **amount<0 raises** loudly (defense-in-depth — parse_effect
    /// catches at register time). **amount=0 is a legal no-op**
    /// (mirrors GainLife / LoseLife / TargetOpponentLosesLife
    /// zero-amount conventions).
    TargetPlayerLosesLife { amount: i32 },
    /// Each opponent of the source's controller discards a card (CR
    /// 701.8). Let's Play a Game pattern — 2 cards in the EoE pool
    /// per round-20 measurement. Mirrors Python's
    /// `EachOpponentDiscards(count: int = 1)` dataclass.
    ///
    /// **Distinct from `DiscardCard { target: AnyPlayer }` (round 15)**
    /// — that arm picks the DEFENDING player's hand[0] (caller-
    /// targeted, Mind Rot shape — active player forces the defender to
    /// discard). `EachOpponentDiscards` fires on the SOURCE'S
    /// controller's opponents (source-relative, independent of
    /// active_player at apply-effect time). In two-player land both
    /// collapse to "the one opponent discards hand[0]" but the
    /// variants stay distinct to mirror Python's closed Effect
    /// vocabulary (round-15 pattern).
    ///
    /// **Distinct from `DiscardSelf` (round 24).** Same source-finder
    /// pattern (5-zone walk + stack scan) and same hand[0] +
    /// graveyard-push primitive, but `DiscardSelf` fires on the
    /// CONTROLLER while this arm fires on the OPPONENT of the
    /// controller. Inverted target on an otherwise identical dispatch
    /// shape — composes round-24 EachOpponentLosesLife's opponent-
    /// resolution with round-24 DiscardSelf's discard primitive.
    ///
    /// **Source-finder dispatch (round-23 pattern).** Walks both
    /// players' five tracked zones (hand / library / battlefield /
    /// graveyard / exile) plus the stack to recover the source's
    /// `controller`. Punisher / "let's play a game" effects fire on
    /// whoever controls the source regardless of where the source
    /// currently sits, since the `controller` field doesn't change on
    /// zone moves.
    ///
    /// **Two-player "each opponent" reduces to "the one opponent".**
    /// opponent_id = 1 - controller. The single opponent's hand[0] is
    /// popped to graveyard with CR 400.7 state reset (mirrors round-24
    /// DiscardSelf). Up to `count` discards in a row; empty hand stops
    /// early (Python parity with `for _ in range(self.count): if not
    /// p.hand: break` inside `EachOpponentDiscards.resolve`).
    ///
    /// **Empty opponent hand = silent no-op** (Python parity). **Source
    /// not found anywhere = silent fizzle** (lost-track-of-source
    /// tolerance, mirrors round-24 / round-25 / round-26 shape).
    ///
    /// **`count` shape mirrors Python.** Python's
    /// `EachOpponentDiscards(count: int = 1)` carries a `count: int = 1`
    /// default; this Rust variant mirrors that exactly. Missing
    /// `count` registers as count=1 (NOT a KeyError, mirrors
    /// DiscardSelf / TargetOpponentDiscards default-handling). Negative
    /// `count` raises at register / resolve time.
    EachOpponentDiscards { count: i32 },
    /// Destroy every creature on the battlefield (Wrath of God pattern,
    /// CR 701.7 + 701.4 AOE). Indestructible creatures (CR 702.12)
    /// are NOT destroyed — round-13 keyword check is honored.
    /// Destroyed creatures go to their owner's graveyard (CR 400.6).
    /// 3 cards in the EoE pool per round-20 measurement (Restricted
    /// Office // Lecture Hall and friends).
    ///
    /// **Symmetric AOE: applies to BOTH players' battlefields** — like
    /// DamageEachCreature, this is implicit-target and hits every
    /// creature regardless of controller. No target filter — the
    /// variant doesn't carry a TargetFilter for that reason.
    ///
    /// **Creature-only filter via printing's card_types.** Lands /
    /// artifacts / enchantments / planeswalkers on the battlefield
    /// stay where they are; only objects whose printing has "Creature"
    /// in card_types are destroyed.
    ///
    /// **Indestructible is honored INLINE** (CR 702.12) — unlike
    /// DamageEachCreature (which writes damage and lets the SBA reap),
    /// DestroyAllCreatures moves creatures directly to graveyard. So
    /// the indestructible keyword check happens here, inline, mirroring
    /// the DestroyTarget arm.
    ///
    /// **Per-object state reset on graveyard move** (CR 400.7) —
    /// tapped / summoning_sick / damage / counters all reset on the
    /// way to graveyard, mirroring the DestroyTarget / Sacrifice /
    /// ExileTarget reset convention.
    ///
    /// **Python schema simplification.** Python's
    /// `DestroyAllCreatures` carries optional narrowing fields:
    /// `tapped_only` / `untapped_only` (Split Up modal Wrath) and
    /// `min_power` (Restricted Office "destroy all creatures with
    /// power 3+"). The Rust first slice carries NONE of these — it's
    /// a unit shape ("destroy all creatures, unconditional"). The
    /// narrowed variants wait for richer dispatch plumbing and are
    /// deferred along with the rest of the TargetFilter widening
    /// shape elsewhere in the bridge.
    DestroyAllCreatures,
    /// **Beyond-the-Quiet slice (2026-05-21; 4,698 17lands decks).**
    /// Exile every creature AND every Spacecraft on the battlefield
    /// (CR 701.18 + 701.4 AOE wrath — exile, not destroy). NO targets,
    /// acts on the full battlefield snapshot at resolution. Mirrors
    /// Python's `ExileAllCreaturesAndSpacecraft`
    /// (`effects/generic/exile.py:687`).
    ///
    /// **Symmetric AOE: applies to BOTH players' battlefields** —
    /// implicit-target like `DestroyAllCreatures`. No target filter —
    /// the variant doesn't carry a TargetFilter for that reason.
    ///
    /// **Creature OR Spacecraft filter** (the multi-type predicate;
    /// the Lane T Spacecraft-subtype finding). A permanent is a victim
    /// iff its printing's `card_types` contains "Creature" OR its
    /// printing's `subtypes` contains "Spacecraft" (Spacecraft is a
    /// SUBTYPE, not a card type — Python's `printing.is_spacecraft` at
    /// `cards/base.py:1056-1057` reads `"Spacecraft" in subtypes`).
    /// Spacecraft printings carry the "Artifact" card type today but
    /// the filter mirrors Python's pure-subtype check exactly so a
    /// non-creature non-stationed Spacecraft is still exiled (which is
    /// the whole point of the spell's "whether or not the Spacecraft
    /// has been stationed into creature mode" wording — see the
    /// Python class docstring).
    ///
    /// **Exile bypasses indestructible (CR 702.12).** Unlike
    /// `DestroyAllCreatures` which honors indestructible inline, this
    /// arm does NOT skip indestructible permanents — exile is a zone
    /// change, not a destroy. Mirrors Python's resolve which routes
    /// every victim through `turn.move_off_battlefield(destination=
    /// "exile")` unconditionally.
    ///
    /// **No DIES dispatch on exile.** CR 700.4: "dies" = put into
    /// graveyard FROM the battlefield. Exiled creatures do NOT die;
    /// only LEAVES dispatch fires. Python's `move_off_battlefield`
    /// (`turn.py:253` `if destination == "graveyard" and was_creature`)
    /// gates the dies trigger on the graveyard destination; this arm
    /// mirrors that exact gate by NOT pushing a dies dispatch.
    ///
    /// **Per-object state reset on exile** (CR 400.7) — tapped /
    /// damage / counters / attached_to / exiled_with_source all reset
    /// on the way to exile, mirroring the `ExileTarget` reset
    /// convention. `summoning_sick` is NOT cleared per the CE13
    /// closure (Python's `move_off_battlefield` deliberately omits the
    /// clear; sickness is battlefield-only state).
    ///
    /// **Visit order (bf-ETB-timestamp).** Walks both bfs via
    /// `crate::bf_iter::bf_iter_etb_sorted` (Python's centralized
    /// `state.battlefield` walk is also ETB-monotone). Outcome is
    /// order-invariant: each victim ends up in its OWN owner's exile
    /// regardless of visit order, and per-owner exile insertion order
    /// is preserved.
    ///
    /// **LEAVES / token-cease queueing.** Each victim's id is pushed
    /// onto `pending_leaves_dispatches` (LTB triggers) and queued for
    /// token-cease (CR 111.7 / 704.5d). Both drains run in
    /// `resolve_top_of_stack` AFTER the resolving spell's stack pop —
    /// the Slice-11 `DestroyAllCreatures` defer-past-pop pattern.
    /// Tokens cease to exist on exile per CR 704.5d (the bare wrath-
    /// AOE token-cease site, NOT a stack-cleanup-time SBA — matches
    /// Python's `if obj.printing.is_token: state.log.append(... token
    /// ceases to exist)` short-circuit in `move_off_battlefield`).
    ///
    /// **Python schema simplification.** Python's class is field-free
    /// (no narrowing rider). This Rust variant is also field-free —
    /// the bare wrath shape, no variant payload.
    ExileAllCreaturesAndSpacecraft,
    /// Destroy target non-land permanent on the defending player's
    /// battlefield (CR 701.7 + scoped target filter — Drag to the
    /// Roots / Drill Too Deep pattern). 2 cards in the EoE pool per
    /// round-20 measurement. Scans defender's battlefield top-to-
    /// bottom and picks the first non-Land, non-indestructible
    /// object. Mirrors round-17 DestroyTarget's pick-bf[0] shape
    /// with a Land-skip + indestructible-skip filter. Unit variant —
    /// the "non-land permanent" target shape is intrinsic.
    DestroyTargetNonlandPermanent,
    /// Destroy target artifact on the defending player's battlefield
    /// (CR 701.7, artifact-filter). 2 cards in EoE pool per round-20
    /// measurement (Drill Too Deep, Scour for Scrap). Unit variant.
    ///
    /// **First-slice deterministic choice**: defender's FIRST artifact
    /// (top-to-bottom scan), skipping indestructible (CR 702.12) per
    /// the round-24 DestroyTargetNonlandPermanent convention.
    /// Empty / no-artifacts / only-indestructible-artifacts = silent
    /// fizzle.
    ///
    /// **Distinct from `DestroyTargetNonlandPermanent` (round 24)** —
    /// that arm filters `!card_types.contains("Land")` (picks creatures,
    /// artifacts, enchantments, planeswalkers — anything non-land).
    /// This arm filters `card_types.contains("Artifact")` (artifacts
    /// only). Identical dispatch shape otherwise (defender-side scan,
    /// indestructible-skip, first-match, silent fizzle on empty /
    /// only-indestructible-artifacts). Mirrors Python's
    /// `DestroyTargetArtifact(Destroy)` constructor sugar with
    /// `targets=(TargetSpec(IsArtifact()),)`.
    DestroyTargetArtifact,
    /// Destroy target land on the defending player's battlefield
    /// (CR 701.7, land-filter). 2 cards in EoE pool per round-20
    /// measurement (Dust Bowl etc.). Unit variant.
    ///
    /// **First-slice deterministic choice**: defender's FIRST land
    /// (top-to-bottom scan), skipping indestructible. Mirrors round-25
    /// `DestroyTargetArtifact` exactly with the filter swapped.
    /// Empty / no-lands / only-indestructible-lands = silent fizzle.
    ///
    /// **Distinct from `DestroyTargetArtifact` (round 25)** —
    /// that arm filters `card_types.contains("Artifact")` (artifacts
    /// only). This arm filters `card_types.contains("Land")` (lands
    /// only). Identical dispatch shape otherwise (defender-side scan,
    /// indestructible-skip, first-match, silent fizzle on empty /
    /// only-indestructible-lands). Mirrors Python's
    /// `DestroyTargetLand(Destroy)` constructor sugar with
    /// `targets=(TargetSpec(IsLand()),)`.
    DestroyTargetLand,
    /// **BLOCKED-RUST-EFFECT slice (round 52) — Extinguisher Battleship
    /// ETB head.** Destroy target NONCREATURE permanent (CR 701.7 with
    /// a `!IsCreature() & !IsPlayer()` filter). Unit variant — the
    /// "noncreature permanent" target shape is intrinsic.
    ///
    /// **Distinct from `DestroyTargetNonlandPermanent` (round 24)** —
    /// that arm filters `!card_types.contains("Land")` (picks
    /// creatures, artifacts, enchantments, planeswalkers — anything
    /// non-land; EXCLUDES lands). THIS arm filters
    /// `!card_types.contains("Creature")` (picks artifacts,
    /// enchantments, planeswalkers, LANDS — anything non-creature;
    /// EXCLUDES creatures). The two filters are NOT interchangeable:
    /// `nonland` admits creatures + excludes lands; `noncreature`
    /// admits lands + excludes creatures. The real Extinguisher
    /// Battleship printing parses (faithfully) to Python's
    /// `DestroyTargetNoncreaturePermanent` whose `TargetSpec` filter
    /// is exactly `~IsCreature() & ~IsPlayer()` — so this slice ships
    /// the precise sibling rather than approximating via the
    /// pre-existing nonland arm (no `nonland≈noncreature` shortcut).
    ///
    /// **First-slice deterministic choice**: scan BOTH players'
    /// battlefield in ETB-timestamp order
    /// (`bf_iter::bf_iter_etb_sorted`), pick the FIRST object whose
    /// printing `card_types` does NOT contain "Creature" and which is
    /// not indestructible (CR 702.12). Empty / no-noncreature /
    /// only-indestructible-noncreature = silent fizzle. Players are
    /// structurally excluded (the bf walk only iterates GameObjects).
    /// Identical dispatch/LTB-fanout shape to the round-24
    /// `DestroyTargetNonlandPermanent` arm with the filter predicate
    /// swapped Land→Creature. Mirrors Python's
    /// `effects/generic/destroy.py:DestroyTargetNoncreaturePermanent`.
    DestroyTargetNoncreaturePermanent,
    /// **Effect-uncovered Tier-A — Radiant Strike inner half.**
    /// Destroy the first battlefield permanent matching
    /// (Artifact) OR (Creature AND tapped), in ETB-timestamp order,
    /// skipping indestructible. Field-free unit variant (no payload —
    /// the typed-OR filter is the whole behavior). Mirrors Python's
    /// `DestroyTargetArtifactOrTappedCreature`
    /// (`effects/generic/destroy.py:475`) whose `TargetSpec` filter is
    /// `IsArtifact() | (IsCreature() & IsTapped())`.
    ///
    /// **Distinct from `DestroyTargetArtifact` (round 25)** — that
    /// arm filters `card_types.contains("Artifact")` (artifacts only).
    /// THIS arm admits artifacts OR tapped creatures: an untapped
    /// non-artifact creature is NOT a legal pick (the OR-filter
    /// teeth). Identical dispatch/LTB-fanout shape otherwise (BOTH
    /// players' bf in ETB-timestamp order, indestructible-skip,
    /// first-match, silent fizzle on empty / only-indestructible /
    /// only-untapped-non-artifact-creatures). The `tapped` term is
    /// read off the scanned `GameObject` (live battlefield state, the
    /// round-53 effective-state lesson — read where the real card
    /// would; `obj.tapped` IS the effective tapped state).
    ///
    /// The `Sequence`+`GainLife` outer shell is already live — Radiant
    /// Strike's full effect is
    /// `Sequence([DestroyTargetArtifactOrTappedCreature, GainLife{3}])`,
    /// composed entirely from existing arms + this one new arm.
    DestroyTargetArtifactOrTappedCreature,
    /// **Effect-uncovered Tier-A — Seedship Impact (Lander-rider).**
    /// Destroy the first battlefield (Artifact OR Enchantment) in
    /// ETB-timestamp order, skipping indestructible. If it was
    /// actually destroyed AND its printed mana value ≤ 2, create one
    /// Lander token under the source's controller. Mirrors Python's
    /// `SeedshipImpactEffect` (`effects/bespoke.py:1219`): pick target
    /// (artifact OR `"Enchantment" in card_types`); read
    /// `mv = obj.printing.mana_cost.total` **before** destroy;
    /// `destroy(state, obj)` (returns `False` for indestructible →
    /// **no Lander consolation**); if destroyed AND `mv ≤ 2` →
    /// `CreateToken(Lander)` for the controller.
    ///
    /// **NOT a field-free unit variant (mini-spec deviation,
    /// documented).** The mini-spec sketched a unit variant, but
    /// Rust's registry has NO intrinsic Lander concept —
    /// `RegistryData::is_token` is purely supertype-derived and a
    /// bespoke token-minting arm must carry the Lander's
    /// `token_printing_id: u32` exactly as `Effect::CreateToken`
    /// does (the only bridge-correct option; the bridge ships the pid
    /// via the round-49/56 `TOKEN_PRINTING_SUBSTRATE`). The Python
    /// `SeedshipImpactEffect` is still field-free — it resolves the
    /// Lander intrinsically via `TOKEN_PRINTINGS["Lander"]`; the pid
    /// is a Rust-side registry-index carrier only, dropped at the
    /// bridge (the bridge constructs the bare Python class).
    ///
    /// **Indestructible parity nuance (§2):** the scan SKIPS
    /// indestructible permanents (matching the shipped
    /// `DestroyTargetArtifact` precedent), so an
    /// indestructible-only board → no target → no destroy → no
    /// Lander, byte-equal to Python's `destroy()`-returns-False →
    /// early-return (no Lander consolation). Both engines reach the
    /// same observable. Re-derive the Lander create block privately
    /// (file-iso) — do NOT cross-import the `Effect::CreateToken`
    /// arm.
    SeedshipImpactEffect { token_printing_id: u32 },
    /// **Effect-uncovered Tier-A — Orbital Plunge (Lander-rider).**
    /// Deal `amount` damage to the first targetable creature (the
    /// IsCreature TargetSpec; deterministic first-creature scan over
    /// BOTH battlefields in ETB-timestamp order, matching Python's
    /// centralized `state.battlefield` walk + the
    /// `_slot_target_still_legal` re-check). If excess damage was
    /// dealt — `excess = max(0, amount − max(0, eff_tough −
    /// damage_already_marked))`, evaluated AT RESOLVE *before* the
    /// damage is added — create one Lander token under the source's
    /// controller. Mirrors Python's
    /// `DealDamageToTargetCreatureWithExcessLanderRider`
    /// (`effects/generic/damage.py:327`). Lifelink/deathtouch NOT
    /// modeled (Python parity — same documented simplification).
    /// `amount` (6 for Orbital Plunge) is the only damage payload;
    /// `token_printing_id` is the Lander's Rust registry index (the
    /// same pid-carrier deviation as `SeedshipImpactEffect` — Rust
    /// has no intrinsic Lander; Python's class is field-free + mints
    /// from `TOKEN_PRINTINGS["Lander"]`; the bridge drops the pid).
    ///
    /// **§2 counters-asymmetry (load-bearing).** Python's
    /// `effective_toughness` (`game._compute_effective_pt`) FOLDS
    /// `+1/+1` counters (CR 613 layer 7d). Rust's combat-side
    /// `effective_toughness` getter deliberately does NOT (the
    /// `state_ops_combat.rs` asymmetry note), but the SBA-side lethal
    /// check (`state_ops_sba.rs::collect_sba_victims`) DOES via
    /// `base_toughness + obj.counters + obj.ueot_toughness_bonus()`.
    /// To stay byte-equal with the Python source of truth, THIS arm
    /// computes the remaining-lethal denominator with the
    /// counters-INCLUSIVE formula (re-derived privately per the
    /// file-isolation rule — base override OR printed toughness, plus
    /// `obj.counters`, plus `ueot_toughness_bonus()`), NOT the
    /// combat-side getter. Python is the SoT; matching it is the rule.
    DealDamageToTargetCreatureWithExcessLanderRider {
        amount: i32,
        token_printing_id: u32,
    },
    /// **Effect-uncovered Tier-A — Depressurize.**
    /// Apply a −`power_penalty`/−0 UEOT modifier to the (single,
    /// deterministic-first) target creature on EITHER battlefield;
    /// then if its EFFECTIVE power (post-UEOT, counters-inclusive per
    /// Python's `effective_power`) is ≤ 0, destroy it (respecting
    /// indestructible — CR 702.12). Mirrors Python's
    /// `DepressurizeEffect(power_penalty)`
    /// (`effects/bespoke.py:1531`): `grant_pt_pump_ueot(power=-pp,
    /// toughness=0)`; then `if effective_power(state, obj) <= 0:
    /// destroy(state, obj)`. UEOT-push + the destroy tail are shipped
    /// primitives; this is the fused composite arm only.
    ///
    /// **Verified deviation from `PumpTargetCreatureUntilEOT` (pin
    /// it):** Python's `DepressurizeEffect.legal_targets` is *any
    /// creature* (`is_creature`, NO controller restriction —
    /// `effects/bespoke.py:1545-1552`), so the scan walks BOTH
    /// battlefields in ETB-timestamp order (NOT controller-only,
    /// unlike the bare pump arm).
    ///
    /// **§2 counters-asymmetry (load-bearing, the mini-spec risk):**
    /// the conditional reads POST-UEOT EFFECTIVE power. Python's
    /// `effective_power` folds the just-pushed UEOT bonus AND `+1/+1`
    /// counters (CR 613 layer 7c+7d). Rust's combat-side
    /// `effective_power` getter folds the UEOT bonus but NOT counters.
    /// To stay byte-equal with the Python SoT, this arm computes the
    /// effective power with the counters-INCLUSIVE formula re-derived
    /// privately (file-iso): `base = ueot_base_pt_override().p OR
    /// printed_power`, plus `obj.counters`, plus
    /// `obj.ueot_power_bonus()` (which already includes the −pp just
    /// pushed). Python is the SoT; matching it is the rule.
    DepressurizeEffect { power_penalty: i32 },
    /// **Effect-uncovered Tier-A — Biosynthic Burst.**
    /// On the first creature the source-controller controls (Python's
    /// `legal_targets` IS controller-restricted —
    /// `o.controller_id == controller_id`, `effects/bespoke.py:1484`
    /// — so this scan is controller-only, UNLIKE `DepressurizeEffect`
    /// which is ANY-creature; verified deviation, pinned): add 1 to
    /// `obj.counters` (the +1/+1 counter — the signed-i32 net-counter
    /// field, the EXACT shape `PutPlusOneCounterOnCreatureYouControl`
    /// `:7780` writes via `saturating_add`; the established byte-equal
    /// Python `obj.plus_one_counters` ↔ Rust `obj.counters` mapping);
    /// push a UEOT `GrantKeyword` modifier for each of
    /// {reach, trample, indestructible} (the EXACT push shape the
    /// proven `GrantKeywordsToTargetCreatureUntilEOT` arm `:10559`
    /// uses — `obj.ueot_modifiers.push(UeotModifier::GrantKeyword)`,
    /// lowercase; that arm's continuous-effect ↔ ueot_modifiers
    /// equivalence is already cross-engine-proven byte-equal); set
    /// `obj.tapped = false`. Atomic, one own target. Field-free unit
    /// variant. Mirrors Python's `BiosynthicBurstEffect`
    /// (`effects/bespoke.py:1466`): `add_plus_one_counters(state,
    /// obj, 1)`; `ApplyEffect(GrantKeywords({reach,trample,
    /// indestructible}), EndOfTurn()).resolve(...)`; `obj.tapped =
    /// False`. Counter / multi-kw-UEOT-grant / untap are all shipped
    /// primitives — this is the fused composite arm only.
    ///
    /// **§2 simplification (mirrors the proven sibling, NOT new
    /// debt).** `PutPlusOneCounterOnCreatureYouControl` `:7780` ALSO
    /// does a plain `saturating_add` and does NOT replicate Python's
    /// `add_plus_one_counters` counter-doubling-replacement /
    /// Terrasymbiosis observer-trigger fan-out (no doubler / no
    /// "counters added to your creature" trigger exists in the EoE
    /// pool). This arm follows that identical, already-cross-engine-
    /// proven shape; documented in `docs/SEARCH_LIMITATIONS.md`.
    BiosynthicBurstEffect,
    /// **Effect-uncovered Tier-A — Focus Fire.**
    /// Deal `(base + |{objects the source-controller controls that
    /// are Creature OR Spacecraft}|)` damage to the first attacking-
    /// or-blocking creature, scanned in ETB-timestamp order across
    /// BOTH battlefields (Python's `legal_targets` filter is
    /// `IsCreature ∧ (IsAttacking ∨ IsBlocking)` — NOT controller-
    /// restricted, like `DepressurizeEffect`'s ANY-creature scan, NOT
    /// the controller-only `BiosynthicBurstEffect` scan).
    ///
    /// **The set-union denominator (the §2 subtle-divergence pin).**
    /// `x = base + state.players[controller].battlefield.iter()
    /// .filter(|o| p.card_types ∋ "Creature" || p.card_types ∋
    /// "Spacecraft").count()` — a SINGLE pass over the battlefield
    /// `Vec`; an object that is BOTH a Creature and a Spacecraft
    /// counts ONCE (it is one `Vec` element, the `||` is on the
    /// object, NOT a sum of a creature-count and a spacecraft-count —
    /// a two-scan sum silently over-counts a dual-typed permanent).
    /// Mirrors Python's `FocusFireEffect._amount`
    /// (`effects/bespoke.py:1784`): `base + sum(1 for o in
    /// state.battlefield if o.controller_id == controller_id and
    /// (o.printing.is_creature or o.printing.is_spacecraft))`.
    ///
    /// **The combat-declaration predicate (the first effect-resolve
    /// arm to READ combat state).** A creature is attacking iff its
    /// id is in `state.attackers` (`Vec<u32>`); blocking iff its id
    /// appears as a `blocker_id` in `state.blockers`
    /// (`Vec<(attacker_id, blocker_id)>`). This is the EXACT pair the
    /// `declare_attacker` (`:681`) / `declare_blocker` (`:772`)
    /// primitives push and the multi-blocker cross-engine probes have
    /// proven byte-equal Python↔Rust. Python re-checks
    /// `_slot_target_still_legal` at resolve (CR 608.2b fizzle); if
    /// no creature is attacking-or-blocking the spell still resolves
    /// and the damage fizzles (silent no-op). Damage is applied via
    /// `obj.damage.saturating_add` (the `DealDamageAnyTarget` shape).
    /// X-from-count substrate family (round-32..49); NEW denominator
    /// (creatures∪Spacecraft set-union, +`base` offset) + the
    /// attacking/blocking target filter only — no new substrate kind.
    FocusFireEffect { base: i32 },
    /// **Effect-uncovered Tier-A #7 — Diplomatic Relations.** The
    /// 2-slot heterogeneous-target spell the spell-card-bridge
    /// **Phase 2** lever admits (round-76; spec
    /// `docs/SPELL_CARD_BRIDGE_PHASE1_DESIGN.md` Phase-2 section).
    /// "Target creature you control gets +1/+0 and gains vigilance
    /// until end of turn. It deals damage equal to its power to
    /// target creature an opponent controls."
    ///
    /// **Field-free unit variant** (the `BiosynthicBurstEffect`
    /// idiom): the +1/+0, the vigilance grant, and the
    /// power-equal-to-effective-power damage are all intrinsic — no
    /// dict payload. Mirrors Python's `DiplomaticRelationsEffect`
    /// (`effects/bespoke.py:1884`), which also carries no fields.
    ///
    /// **The 2-slot target grammar (the genuinely-new Phase-2
    /// substrate).** Python's `target_slot_kinds()` is
    /// `("my_creature", "creature")` — slot 0 is a creature the
    /// caster controls (pumped), slot 1 is an opponent's creature
    /// (damaged). Both targets are bound at cast time onto the spell
    /// `StackItem` (`StackItem.target_obj_id` = slot-0 friendly,
    /// `StackItem.target_obj_id_2` = slot-1 opponent) and threaded
    /// to resolve via the existing `current_resolve_payload_target_id`
    /// swap PLUS the new `current_resolve_payload_target_id_2` swap.
    /// The Phase-1 single-target carrier (`target_obj_id`) is
    /// UNCHANGED — Phase 2 only adds the second slot, defaulted
    /// `None`, so every Phase-1 single-target arm is byte-equal.
    ///
    /// **Resolve semantics (mirrors Python `resolve`,
    /// `effects/bespoke.py:1940-2008`, byte-equal to the SoT):**
    /// 1. Locate slot-0 (friendly creature) by the resolve-time
    ///    payload-target swap. Re-validate (CR 608.2b): it must be
    ///    on a battlefield AND still controlled by the caster, else
    ///    the whole effect fizzles silently (Python's `my_obj = None`
    ///    short-circuit returns early — no damage either).
    /// 2. Push a `+1/+0` UEOT `PowerToughnessBonus` modifier (the
    ///    EXACT shape `DepressurizeEffect`/`PumpTargetCreatureUntil
    ///    EOT` push, proven byte-equal) AND a UEOT `GrantKeyword
    ///    {"vigilance"}` modifier (the EXACT shape the proven
    ///    `GrantKeywordsToTargetCreatureUntilEOT` arm uses; its
    ///    continuous-effect↔ueot_modifiers equivalence is already
    ///    cross-engine-proven byte-equal). Both drain at cleanup on
    ///    both engines.
    /// 3. Locate slot-1 (opponent's creature) by the second
    ///    payload-target swap. Re-validate: on a battlefield AND
    ///    controlled by a non-caster, else the damage component
    ///    fizzles (the pump already applied — Python returns after
    ///    the pump in that branch).
    /// 4. Read the slot-0 creature's POST-PUMP effective power
    ///    (counters-inclusive per Python's `effective_power`, the
    ///    `DepressurizeEffect` counters-inclusive formula re-derived
    ///    privately per the file-isolation rule: `base =
    ///    ueot_base_pt_override().p OR printed_power`, plus
    ///    `obj.counters`, plus `obj.ueot_power_bonus()` which now
    ///    includes the just-pushed `+1`) and deal that much damage
    ///    to the slot-1 creature via `obj.damage.saturating_add`
    ///    (the `DealDamageAnyTarget` damage primitive shape). SBA
    ///    handles lethal post-resolve byte-equal.
    ///
    /// All three sub-primitives (UEOT-pump, UEOT-keyword-grant,
    /// damage) are shipped + cross-engine-proven; this is the fused
    /// composite arm + the 2-slot binding only.
    DiplomaticRelationsEffect,
    /// **Effect-uncovered Tier-A — Cut Propulsion (~11,990 17lands
    /// decks).** "Target creature deals damage to itself equal to its
    /// power. If that creature has flying, it deals twice that much
    /// damage to itself instead."
    ///
    /// Mirrors Python's `CutPropulsionEffect`
    /// (`effects/bespoke.py:1821-1879`) — the ONLY EoE card with this
    /// shape. Single-target self-damage with a flying-doubler. The
    /// Phase-1 single-target spell-card-bridge shape
    /// (`target_count_range == (1, 1)`, `target_slot_kinds == ()`,
    /// any-creature filter — Python's `legal_targets` is "every
    /// creature on `state.battlefield`", NOT controller-restricted).
    ///
    /// **Field-free unit variant** (the `BiosynthicBurstEffect` /
    /// `DiplomaticRelationsEffect` idiom): no dict payload — the
    /// damage amount is intrinsic (read from the target's effective
    /// power at resolve time; doubled if the target has flying).
    /// Mirrors Python's `CutPropulsionEffect()` `@dataclass`-style
    /// field-free instance.
    ///
    /// **Resolve semantics (mirrors `bespoke.py:1843-1873` byte-equal
    /// to the SoT):**
    /// 1. Read the cast-time-bound target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    `FocusFireEffect` / `DepressurizeEffect` payload-target swap
    ///    pattern). When `None` (synthetic-test `push_to_stack` path
    ///    that doesn't seat a target), fall back to the
    ///    deterministic-first creature in ETB-timestamp order across
    ///    BOTH battlefields (mirrors `bf_iter_etb_sorted` — Python's
    ///    `state.battlefield` centralized walk = the SoT).
    /// 2. Re-validate the bound target at resolve (CR 608.2b): the
    ///    target must still be on a battlefield AND still satisfy
    ///    `card_types ∋ "Creature"`. Illegal target → silent fizzle
    ///    (Python's `obj is None` short-circuit returns early — no
    ///    damage either).
    /// 3. Read the target's effective power. Saturating into i32 from
    ///    `effective_power(state, obj)` semantics — `base + counters +
    ///    ueot_power_bonus()` (Python's `mtg_bot.game.effective_power`
    ///    folds the same three contributions). If `power <= 0`, silent
    ///    no-op (NO damage event fires — Python's `if power <= 0:
    ///    return` short-circuits before `_apply_damage`, so the
    ///    receiver-side DEALT_DAMAGE dispatch does not fire either).
    /// 4. If the target's effective keyword set contains "flying"
    ///    (`battlefield_has_keyword` — folds printed + UEOT-granted +
    ///    aura-granted + static-anthem keywords; the established
    ///    public reader), damage amount is `power * 2`; else `power`.
    /// 5. Apply damage via `obj.damage.saturating_add(amount_i16)` —
    ///    the established `DealDamageAnyTarget` / `FocusFireEffect`
    ///    damage-application shape. Set `damaged_this_turn = true`
    ///    on the receiver (mirrors Python's `obj.damaged_this_turn =
    ///    True` at `effects/base.py:300`). Dispatch the DEALT_DAMAGE
    ///    receiver-side event for any aura-attached
    ///    `enchanted_creature_dealt_damage` triggers (Cryoshatter,
    ///    Pain for All) — mirrors Python's
    ///    `_fire_enchanted_dealt_damage_triggers` call at the tail of
    ///    `_apply_damage`. The damage marker is set BEFORE the event
    ///    so aura readers see the post-damage state, matching Python
    ///    (which sets `obj.damage += amount` before invoking the
    ///    fire-triggers helper).
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `CutPropulsionEffect()` (field-free construction). Mirrors the
    /// `BiosynthicBurstEffect` / `DiplomaticRelationsEffect` Tier-A
    /// unit-variant convention.
    CutPropulsionEffect,
    /// **Effect-uncovered Tier-A — Invasive Maneuvers (~10,981
    /// 17lands decks).** "Invasive Maneuvers deals 3 damage to target
    /// creature. It deals 5 damage instead if you control a Spacecraft."
    ///
    /// Mirrors Python's `DealDamageToTargetCreatureSpacecraftBoosted`
    /// (`effects/generic/damage.py:850-905`) — the ONLY EoE card with
    /// this exact "deals N damage / M instead if you control a
    /// Spacecraft" shape. Single-target damage with a Spacecraft-on-
    /// caster's-bf conditional amount. The Phase-1 single-target
    /// spell-card-bridge shape (`target_count_range == (1, 1)`,
    /// `target_slot_kinds == ()`, any-creature filter — Python's
    /// `IsCreature()` slot filter is NOT controller-restricted).
    ///
    /// **Two-int field variant** (the `pump_target_creature_until_eot`
    /// idiom; NOT the field-free `CutPropulsionEffect` /
    /// `DiplomaticRelationsEffect` unit-variant idiom — Invasive
    /// Maneuvers carries the printed `3`/`5` integers parameterized
    /// off the printed oracle text by the per-clause
    /// `_DEAL_DAMAGE_N_TARGET_CREATURE_M_IF_SPACECRAFT` recognizer at
    /// `parser/effects.py:4668-4673`). Mirrors Python's
    /// `DealDamageToTargetCreatureSpacecraftBoosted(base=3, boosted=5)`
    /// `@dataclass(frozen=True)` instance. Default `(3, 5)` on both
    /// sides for a key-elided clean round-trip; the dataclass field
    /// defaults match (`base: int = 3`, `boosted: int = 5`).
    ///
    /// **Resolve semantics (mirrors `damage.py:875-897` byte-equal to
    /// the SoT):**
    /// 1. Read the cast-time-bound target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    `FocusFireEffect` / `CutPropulsionEffect` payload-target swap
    ///    pattern). When `None` (synthetic-test `push_to_stack` path
    ///    that doesn't seat a target), fall back to the deterministic-
    ///    first creature in ETB-timestamp order across BOTH bfs
    ///    (mirrors `bf_iter_etb_sorted` — Python's `state.battlefield`
    ///    centralized walk = the SoT).
    /// 2. Re-validate the bound target at resolve (CR 608.2b): the
    ///    target must still be on a battlefield AND its printing must
    ///    still satisfy `card_types ∋ "Creature"`. Illegal target →
    ///    silent fizzle (Python's `_slot_target_still_legal` returns
    ///    early at `damage.py:891-893`).
    /// 3. Source-controller resolution (5-zone walk + stack scan; the
    ///    established sister shape — `FocusFireEffect`,
    ///    `BiosynthicBurstEffect`). Mirrors Python's `controller_id`
    ///    parameter passed into `_amount(state, controller_id)`.
    /// 4. Scan the source-controller's battlefield for any object
    ///    whose printing's `subtypes` contains "Spacecraft" (NOT
    ///    `card_types` — Spacecraft is a SUBTYPE per `cards/base.py:
    ///    1056-1057` `is_spacecraft = "Spacecraft" in subtypes`).
    ///    Mirrors Python's `o.printing.is_spacecraft` predicate at
    ///    `damage.py:876-879`. Amount = `boosted` if any controlled
    ///    Spacecraft present, else `base`.
    /// 5. Apply damage via `obj.damage.saturating_add(amount_i16)` —
    ///    the established `DealDamageAnyTarget` / `CutPropulsionEffect`
    ///    damage-application shape. Set `damaged_this_turn = true`
    ///    on the receiver (mirrors Python's `_apply_damage` at
    ///    `effects/base.py:299-300`). Dispatch the DEALT_DAMAGE
    ///    receiver-side event for any aura-attached "when enchanted
    ///    creature dealt damage" triggers (Cryoshatter, Pain for All)
    ///    — mirrors Python's `_fire_enchanted_dealt_damage_triggers`
    ///    call at the tail of `_apply_damage`,
    ///    `effects/base.py:303-305`. The damage marker is set BEFORE
    ///    the event so aura readers see the post-damage state,
    ///    matching Python (which sets `obj.damage += amount` before
    ///    invoking the fire-triggers helper).
    ///
    /// **Schema parity with Python.** Two-int variant on both sides:
    /// `{ base: i32, boosted: i32 }` ↔ Python's `base: int`,
    /// `boosted: int` dataclass fields. The bridge translates this
    /// Rust variant to `DealDamageToTargetCreatureSpacecraftBoosted(
    /// base=base, boosted=boosted)`.
    InvasiveManeuversEffect { base: i32, boosted: i32 },
    /// **Effect-uncovered single-gated tier — Shattered Wings
    /// (~7,793 17lands decks).** "Destroy target artifact,
    /// enchantment, or creature with flying. Surveil 1."
    ///
    /// Mirrors Python's `DestroyTargetArtifactEnchantmentOrFlyingCreature`
    /// (`effects/generic/destroy.py:501`) — the ONLY EoE card with this
    /// 3-way OR target filter. The full Shattered Wings spell body is
    /// `Sequence(DestroyTargetArtifactEnchantmentOrFlyingCreature(),
    /// Surveil(1))` — the destroy arm is the only new variant; the
    /// `Sequence` wrapper and the `Surveil { count: 1 }` leaf are
    /// already shipped (`Effect::Sequence` round-15 /
    /// `Effect::Surveil` round-15).
    ///
    /// **Field-free unit variant** (the `DestroyTargetArtifactOrTapped
    /// Creature` idiom — the closest sibling). No dict payload — the
    /// 3-way OR filter `IsArtifact() | IsEnchantment() | (IsCreature()
    /// & HasKeyword("flying"))` is intrinsic to the variant. Python's
    /// `__post_init__` sets `targets=(TargetSpec(IsArtifact() |
    /// IsEnchantment() | (IsCreature() & HasKeyword("flying"))),)` and
    /// the dataclass carries no constructor parameters.
    ///
    /// **Resolve semantics (mirrors `Destroy.resolve` at
    /// `effects/generic/destroy.py:54-86` byte-equal to the SoT; the
    /// same shape as the shipped `DestroyTargetArtifactOrTapped
    /// Creature` arm with the predicate widened from `Artifact OR
    /// (Creature AND tapped)` to `Artifact OR Enchantment OR (Creature
    /// AND HasKeyword("flying"))`):**
    /// 1. Read the cast-time-bound target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    `DestroyTargetArtifactOrTappedCreature` /
    ///    `SeedshipImpactEffect` payload-target swap pattern). When
    ///    `None` (synthetic-test `push_to_stack` path that doesn't
    ///    seat a target), fall back to the deterministic-first
    ///    permanent matching the 3-way OR filter in ETB-timestamp
    ///    order across BOTH battlefields (mirrors
    ///    `bf_iter_etb_sorted`).
    /// 2. Re-validate the bound target at resolve (CR 608.2b): the
    ///    target must still be on a battlefield AND still satisfy the
    ///    `IsArtifact OR IsEnchantment OR (IsCreature AND
    ///    HasKeyword("flying"))` filter. The "flying" leg reads
    ///    EFFECTIVE keywords via `battlefield_has_keyword` (folds
    ///    printed + UEOT-granted + aura-granted + anthem-granted
    ///    keywords; matches Python's `HasKeyword.matches` which reads
    ///    `effective_keywords`). Illegal target (gone, or no longer
    ///    matches the filter) → silent fizzle.
    /// 3. Indestructible permanents survive — the scan skips them
    ///    (mirrors the shipped `DestroyTargetArtifact` /
    ///    `DestroyTargetArtifactOrTappedCreature` precedent; Python's
    ///    `turn.destroy()` returns False on indestructible, same
    ///    observable).
    /// 4. Destroy the target via the proven artifact-arm body shape:
    ///    remove from bf, reset zone/tapped/damage/attached_to/
    ///    exiled_with_source, push to graveyard, queue DIES dispatch
    ///    if the victim was a Creature (CR 700.4; the R76-2 / R74-2
    ///    fixed pattern — only a creature death fires DIES — so a
    ///    destroyed artifact or enchantment OR the flying-creature
    ///    leg ALL agree byte-equal with the existing OrTapped
    ///    sibling), queue LEAVES dispatch, queue token-cease.
    ///
    /// **Surveil 1 outer-Sequence rider.** The full Shattered Wings
    /// effect is `Sequence([this destroy, Surveil { count: 1 }])`;
    /// the Surveil child runs via the already-live `Effect::Surveil`
    /// arm immediately after this destroy arm returns Ok(()). No
    /// composition-specific code needed here — the Sequence wrapper
    /// dispatches children in order.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `DestroyTargetArtifactEnchantmentOrFlyingCreature()` (field-
    /// free construction). Mirrors the `DestroyTargetArtifactOr
    /// TappedCreature` Tier-A unit-variant convention.
    DestroyTargetArtifactEnchantmentOrFlyingCreature,
    /// **Effect-uncovered single-gated tier — Exorcise (Duskmourn;
    /// loaded via `data/duskmourn.json` into the EoE-aware test pool
    /// — Effect-uncovered single-gated per the round-58 attribution
    /// scout).** `{1}{W}` Sorcery: "Exile target artifact, enchantment,
    /// or creature with power 4 or greater."
    ///
    /// Mirrors Python's `ExorciseEffect`
    /// (`effects/bespoke.py:2268`) — a parameterless dataclass with a
    /// 3-way OR target filter: (IsArtifact) OR (IsEnchantment) OR
    /// (IsCreature AND effective_power ≥ 4). The Python `_matches`
    /// helper re-evaluates the filter at BOTH legal-target time and
    /// at resolve-time (defense-in-depth + CR 608.2b — a pump between
    /// cast and resolve could drop a 4+ creature's effective power
    /// below 4, fizzling the spell).
    ///
    /// **Field-free unit variant** (the
    /// `DestroyTargetArtifactEnchantmentOrFlyingCreature` /
    /// `DestroyTargetArtifactOrTappedCreature` idiom — the closest
    /// siblings). No dict payload — the 3-way OR filter is intrinsic
    /// to the variant. Python's `ExorciseEffect` carries no
    /// constructor parameters.
    ///
    /// **Resolve semantics (mirrors `ExorciseEffect.resolve` at
    /// `effects/bespoke.py:2317-2347` byte-equal to the SoT; same
    /// shape as the shipped `DestroyTargetArtifactEnchantmentOr
    /// FlyingCreature` arm with two swaps — exile-not-destroy and
    /// power-4+-not-flying):**
    /// 1. Read the cast-time-bound target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    `DestroyTargetArtifactEnchantmentOrFlyingCreature` /
    ///    `DestroyTargetArtifactOrTappedCreature` /
    ///    `SeedshipImpactEffect` payload-target swap pattern; CR
    ///    608.2b). When `None` (synthetic-test `push_to_stack` path
    ///    that doesn't seat a target), fall back to the deterministic-
    ///    first permanent matching the 3-way OR filter in
    ///    ETB-timestamp order across BOTH battlefields (mirrors
    ///    `bf_iter_etb_sorted`).
    /// 2. Re-validate the bound target at resolve (CR 608.2b): the
    ///    target must still be on a battlefield AND still satisfy
    ///    `IsArtifact OR IsEnchantment OR (IsCreature AND
    ///    effective_power ≥ 4)`. The power leg reads EFFECTIVE power
    ///    via the inlined `read_effective_power` helper (same body as
    ///    the round-53 `read_doubling_power` — base after
    ///    `ueot_base_pt_override` / `ueot_becomes_land_creature_pt` /
    ///    printed, plus `ueot_power_bonus`, plus signed `obj.counters`;
    ///    matches Python's `mtg_bot.game.effective_power` =
    ///    `_compute_effective_pt` at `game.py:2138-2251`). Illegal
    ///    target (gone, or no longer matches the filter) → silent
    ///    fizzle, mirroring `ExorciseEffect.resolve`'s
    ///    "no longer matches ... — Exorcise fizzles" branch.
    /// 3. **Indestructible is NOT skipped** here — exile bypasses
    ///    indestructible per CR 702.12 (indestructible only prevents
    ///    destroy + lethal-damage, not exile). Matches the shipped
    ///    `Effect::ExileTarget { AnyCreature }` arm which has NO
    ///    indestructible-skip. Distinguishes Exorcise from the destroy
    ///    siblings (Shattered Wings, etc.) which DO skip indestructible.
    /// 4. Exile the target via the proven `ExileTarget` body shape:
    ///    remove from bf, reset `zone/tapped/damage/counters/attached_to/
    ///    exiled_with_source`, push to the target's owner's exile zone
    ///    (first-slice: owner == controller, so push onto `pidx`'s
    ///    exile), queue DIES dispatch if the victim was a Creature (CR
    ///    700.4 — exile from bf still fires DIES for creatures via
    ///    `mark_left_battlefield_if_nonland`'s downstream + the
    ///    pending_dies path; matching the OrTapped + Shattered Wings
    ///    sibling creature-gate), queue LEAVES dispatch, queue
    ///    token-cease.
    ///
    /// **Power-4 leg reads EFFECTIVE power (NOT printed).** A 3/3
    /// creature carrying +1/+1 counters or pumped by a UEOT modifier
    /// to effective power 4+ is legally targetable on BOTH engines
    /// byte-equal (Python's `effective_power` and Rust's
    /// `read_effective_power` fold the same composition). A 4/4
    /// creature debuffed below 4 effective power between cast and
    /// resolve fizzles on BOTH engines (the resolve-time recheck).
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `ExorciseEffect()` (field-free construction). Mirrors the
    /// `DestroyTargetArtifactEnchantmentOrFlyingCreature` Tier-A
    /// unit-variant convention.
    ExorciseEffect,
    /// **Close Encounter — additional-cost-choice closure (Cycle 3,
    /// 2026-05-21; ~16,098 17lands decks — highest-deck-count BLOCKED
    /// card in the substrate-gated tier).** `{1}{G}` Instant:
    /// "As an additional cost to cast this spell, choose a creature
    /// you control or a warped creature card you own in exile.
    /// Close Encounter deals damage equal to the power of the chosen
    /// creature or card to target creature."
    ///
    /// Mirrors Python's `CloseEncounterEffect`
    /// (`effects/bespoke.py:1065`) — a parameterless dataclass that
    /// reads `state.resolving_additional_cost_choice_id` at resolve
    /// time to look up the cast-time-chosen creature/card, then
    /// deals damage equal to that object's power to the bound
    /// target creature.
    ///
    /// **Field-free unit variant** (the
    /// `ExorciseEffect` / `DiplomaticRelationsEffect` /
    /// `CutPropulsionEffect` idiom — the closest siblings). No dict
    /// payload — the additional-cost choice carrier is the StackItem's
    /// `additional_cost_choice_id` (Cycle 1 substrate), swapped into
    /// `state.current_resolve_additional_cost_choice_id` (Cycle 1
    /// resolve-time transient) by the resolver. The single damage-
    /// target carrier is the StackItem's `target_obj_id`, swapped
    /// into `state.current_resolve_payload_target_id` (the
    /// established Tier-A single-target payload-target convention).
    /// Python's `CloseEncounterEffect` carries no constructor
    /// parameters.
    ///
    /// **Resolve semantics (mirrors `CloseEncounterEffect.resolve` at
    /// `effects/bespoke.py:1094-1128` byte-equal to the SoT):**
    /// 1. Read the cast-time-bound damage-target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    Tier-A payload-target convention; CR 608.2b). When `None`
    ///    (synthetic-test fallback) walk both bfs in
    ///    `bf_iter_etb_sorted` order picking the first Creature.
    /// 2. Re-validate the target at resolve (CR 608.2b): the target
    ///    must still be on a battlefield AND its printing must be a
    ///    Creature. Illegal target → silent fizzle, mirroring
    ///    `CloseEncounterEffect.resolve`'s `if kind != "creature":
    ///    return` short-circuit.
    /// 3. Read the cast-time-bound additional-cost choice id from
    ///    `state.current_resolve_additional_cost_choice_id` (the
    ///    Cycle 1 transient). `None` → silent no-op (Python's
    ///    `if chosen_id is None: return`); the additional-cost flag
    ///    wasn't set on the printing, so the resolve arm is a
    ///    fallback no-op (the substrate is parity-NEUTRAL for non-
    ///    Close-Encounter cards).
    /// 4. Look up the chosen object:
    ///    a. **Battlefield branch**: scan both players' battlefields
    ///       for the chosen id. If found, use EFFECTIVE power
    ///       (`read_doubling_power`: printed + ueot_base_pt_override
    ///       + counters + ueot_power_bonus) — mirrors Python's
    ///       `effective_power(state, chosen)` at `:1112`.
    ///    b. **Exile-warped branch**: if not on a battlefield, scan
    ///       the caster's exile for the id. If found AND the object
    ///       is `warped`, use PRINTED power (`printing.power.unwrap
    ///       _or(0)`) — mirrors Python's `chosen.printing.power or
    ///       0` at `:1121`. The exile pool isn't on the battlefield,
    ///       so continuous effects don't apply (per CR 110.1 — only
    ///       permanents on the battlefield are subject to effects;
    ///       the exiled card is an object in exile, not a
    ///       permanent).
    ///    c. **Gone branch**: chosen object not found in either zone
    ///       (left the bf in response between cast and resolve). Use
    ///       amount = 0 — mirrors Python's `amount = 0` fallback at
    ///       `:1127`.
    /// 5. Apply damage to the bound target via the proven
    ///    `apply_damage_marker` body (`obj.damage.saturating_add
    ///    (amount)` on the target object), then dispatch the
    ///    DEALT_DAMAGE receiver-side event (mirrors Python's
    ///    `_fire_enchanted_dealt_damage_triggers` call at the tail
    ///    of `_apply_damage`).
    ///
    /// **Power read is EFFECTIVE on bf, PRINTED on exile.**
    /// Asymmetric by design — the bf creature could carry +1/+1
    /// counters or a UEOT pump that raises its effective power
    /// above printed; the exile card has no such modifiers (the
    /// card's `counters` are reset on zone change per the round-1
    /// exile reset convention; UEOT modifiers don't apply outside
    /// the battlefield per CR 110.1). Byte-equal with Python on
    /// both legs.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `CloseEncounterEffect()` (field-free construction). Mirrors the
    /// `ExorciseEffect` / `DiplomaticRelationsEffect` Tier-A unit-
    /// variant convention.
    CloseEncounterEffect,
    /// **Effect-uncovered single-gated tier — Emergency Eject
    /// (~6,291 17lands decks).** `{2}{W}` Instant: "Destroy target
    /// nonland permanent. Its controller creates a Lander token."
    /// Mirrors Python's
    /// `DestroyTargetNonlandPermanentWithLanderForController`
    /// (`effects/generic/destroy.py:589`).
    ///
    /// **Same Lander-pid carrier shape as `SeedshipImpactEffect` /
    /// `DealDamageToTargetCreatureWithExcessLanderRider`.** Rust has
    /// no intrinsic Lander concept; a bespoke token-minting arm must
    /// carry the Lander's `token_printing_id: u32` exactly like
    /// `Effect::CreateToken` (the round-49/56
    /// `TOKEN_PRINTING_SUBSTRATE`). Python's class is field-free —
    /// it resolves the Lander intrinsically via
    /// `TOKEN_PRINTINGS["Lander"]`. The bridge **drops** the pid
    /// (the bridge constructs the bare Python class). Both engines
    /// seat the SAME "Lander" printing (Rust via the registered pid,
    /// Python via TOKEN_PRINTINGS) so the created token is
    /// byte-equal in the name-keyed field view.
    ///
    /// **Three deviations from `SeedshipImpactEffect`:**
    /// 1. Filter widens from `IsArtifact OR "Enchantment" in
    ///    card_types` to **non-land permanent** (CR 205.2a — any
    ///    battlefield permanent without the `Land` card type).
    ///    Indestructible permanents are still skipped at the scan
    ///    (matching `DestroyTargetArtifact` / Seedship precedent;
    ///    Python's `turn.destroy()` returns False on indestructible
    ///    → no Lander consolation).
    /// 2. **No MV gate.** Seedship's Lander rider fires iff
    ///    `mv ≤ 2`; Emergency Eject's rider fires whenever the
    ///    destroy succeeds. Drops the printed-MV read entirely.
    /// 3. **Lander recipient = FORMER controller of the destroyed
    ///    permanent** (the target's `controller` field at the moment
    ///    of destroy), NOT the spell's source-controller. Python's
    ///    `DestroyTargetNonlandPermanentWithLanderForController`
    ///    snapshots `former_controller = obj.controller_id` BEFORE
    ///    calling `destroy(state, obj)`, then mints the Lander
    ///    under `former_controller`. This is the load-bearing
    ///    cross-controller distinguishing feature vs Seedship —
    ///    Emergency Eject typically targets an opponent's
    ///    permanent, so the opponent gets the Lander as
    ///    consolation, NOT the caster.
    ///
    /// **Resolve semantics (mirrors `DestroyTargetNonlandPermanent
    /// WithLanderForController.resolve` at `effects/generic/
    /// destroy.py:611-640`):**
    /// 1. Read the cast-time-bound target id from
    ///    `state.current_resolve_payload_target_id` (the established
    ///    `SeedshipImpactEffect` / `DestroyTargetArtifactOrTapped
    ///    Creature` payload-target swap pattern). When `None`
    ///    (synthetic-test `push_to_stack` path that doesn't seat a
    ///    target), fall back to the deterministic-first non-land
    ///    permanent in ETB-timestamp order across BOTH battlefields.
    /// 2. Re-validate at resolve (CR 608.2b): target still on a
    ///    battlefield, still NOT a Land, still NOT indestructible.
    ///    Illegal target → silent fizzle (no Lander either).
    /// 3. **Capture `former_controller` from the scanned object's
    ///    `controller` field BEFORE the destroy** (the load-bearing
    ///    step — once the object moves to graveyard the controller
    ///    field is stamped to owner on the leaving copy, but the
    ///    Lander rider needs the BATTLEFIELD-time controller per CR
    ///    111.4 / 800.4).
    /// 4. Destroy the target via the proven artifact-arm body
    ///    (remove from bf, reset zone/tapped/damage/attached_to/
    ///    exiled_with_source, push to graveyard, queue DIES
    ///    dispatch if victim was Creature, queue LEAVES dispatch,
    ///    queue token-cease).
    /// 5. Mint one Lander under `former_controller` — registry-
    ///    bounds-check the pid, reserve a fresh id from
    ///    `next_object_id`, reserve a fresh `etb_timestamp`, push
    ///    the GameObject onto `former_controller`'s battlefield,
    ///    queue ETB dispatch.
    EmergencyEjectEffect { token_printing_id: u32 },
    /// Deal `amount` damage to any target — creature OR player
    /// (Lightning Bolt pattern, CR 701.4). 7 cards in EoE pool per
    /// round-20 measurement (Debris Field Crusher etc.). Mirrors
    /// Python's `DealDamageAnyTarget(DealDamage)` constructor sugar —
    /// the "any target" filter is intrinsic to the variant, distinct
    /// from `DealDamage { target: AnyPlayer/AnyCreature }` which
    /// requires a target dict.
    ///
    /// First-slice deterministic choice: prefer the defender's first
    /// creature on their battlefield (mirrors the `AnyCreature` arm
    /// in `DealDamage`); fall back to defender-as-player life
    /// subtraction (mirrors the `AnyPlayer` arm) when no creature is
    /// available. Future slice plumbs an explicit choice callback.
    /// Negative `amount` raises at register and at resolve time.
    DealDamageAnyTarget { amount: i32 },
    /// **Round 27 UEOT-framework proof.** Source gets
    /// `+power/+toughness` until end of turn. Mirrors Python's
    /// `PumpSelfUntilEOT(power_bonus, toughness_bonus)` dataclass
    /// (12 cards in the EoE pool — Attack-in-the-Box, etc.).
    ///
    /// **"Self" semantics.** The recipient is the source GameObject
    /// at the top of the stack (i.e. the resolving spell or
    /// activated-ability source). Same `state.resolving_source`
    /// shape Python reads — and the same `source_obj_id`
    /// `apply_effect` parameter the round-23 `PutPlusOneCounterOnSelf`
    /// arm already threads through. If the source has left the
    /// battlefield between cast and resolve, the effect is a silent
    /// no-op (mirrors Python's `if source is None: return`).
    ///
    /// **Mechanism.** Pushes a
    /// `UeotModifier::PowerToughnessBonus { power, toughness }` entry
    /// onto the source object's `ueot_modifiers` Vec on the
    /// battlefield. Effective power/toughness queries
    /// (`battlefield_creature_power`/`_toughness` in
    /// `state_ops_combat.rs`, `effective_toughness` in
    /// `state_ops_sba.rs`) walk the Vec on top of printed P/T at
    /// read time. The bonus auto-expires at the next
    /// `cleanup_step_for_active_player` /
    /// `advance_phase`-through-PHASE_CLEANUP via the round-27
    /// CR 514.3 UEOT-clear pass.
    ///
    /// **Schema simplification vs Python.** Python's
    /// `PumpSelfUntilEOT` also accepts an optional
    /// `granted_keywords: frozenset[str]` to grant keywords UEOT in
    /// the same effect. The Rust first slice does NOT carry that
    /// field — the cards in the EoE pool with keyword riders compose
    /// via separate effect chains (the bridge dispatch can layer a
    /// `GrantKeyword` UEOT modifier in a future PumpSelf variant).
    /// Negative bonuses are legal (Cryptolith Rite-shape debuffs);
    /// raise-on-negative is NOT enforced. count=0/0 is a legal
    /// no-op for parity-test edge cases.
    PumpSelfUntilEOT { power_bonus: i32, toughness_bonus: i32 },
    /// **Round 28 UEOT-coverage extension.** Target creature gets
    /// `+power/+toughness` until end of turn. Mirrors Python's
    /// `PumpTargetCreatureUntilEOT(power_bonus, toughness_bonus)`
    /// dataclass (26 cards in the EoE pool — Giant Growth-shape
    /// instants and sorceries, highest-frequency Effect class not
    /// yet in Rust per r27 measurement).
    ///
    /// **Target semantics — first-slice deterministic.** The
    /// recipient is the FIRST creature on the SOURCE'S CONTROLLER's
    /// battlefield (smallest Vec index). This mirrors the
    /// "you-control target" convention that the vast majority of
    /// pump spells use (Giant Growth: "target creature gets +3/+3
    /// UEOT"; typically cast to buff your own creature). If the
    /// source's controller controls NO creatures on battlefield,
    /// the spell fizzles silently — mirrors Python's
    /// `if not legal_targets: return` short-circuit when `optional`
    /// is set or no creature exists. No
    /// explicit target choice today; rich target enumeration is the
    /// future decision-callback follow-up (G20 substrate already
    /// in place from r27).
    ///
    /// **Source-on-battlefield edge.** If `source_obj_id` itself is
    /// a creature on the source's controller's battlefield (rare for
    /// instants/sorceries, but possible for activated-ability sources
    /// or self-resolving permanents), it IS eligible. The Rust
    /// first-slice does NOT carry Python's `another: bool` field
    /// (which excludes self-as-target on cards like Perigee Beckoner);
    /// a future slice can layer it on.
    ///
    /// **Schema simplification vs Python.** Python's
    /// `PumpTargetCreatureUntilEOT` carries `granted_keywords`,
    /// `you_control_only`, `opponents_only`, `target_subtype`,
    /// `optional`, `another`, `granted_triggered_ability`, and a
    /// `draw_if_target_has_plus_one_counter` rider. The Rust first
    /// slice carries NONE of these — cards with those riders compose
    /// via separate effect chains or wait for a decision-callback-
    /// driven richer target plumbing slice. Negative bonuses are
    /// legal (debuff-shaped pumps); no raise-on-negative guard.
    /// `(0, 0)` is a legal no-op for parity edge cases.
    ///
    /// **Mechanism.** Same UEOT-modifier push as `PumpSelfUntilEOT`,
    /// just on the chosen target rather than the source. The
    /// modifier auto-expires at the next cleanup step via the r27
    /// CR 514.3 UEOT-clear pass.
    PumpTargetCreatureUntilEOT { power_bonus: i32, toughness_bonus: i32 },
    /// **BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner ETB.**
    /// "When this creature enters, until end of turn, ANOTHER target
    /// creature you control gets +power/+toughness and gains 'When
    /// this creature dies, return it to the battlefield tapped under
    /// its owner's control.'" (~21,131 17lands decks.)
    ///
    /// This is the `granted_triggered_ability`-carrying shape of
    /// `PumpTargetCreatureUntilEOT` that the bare round-28 arm
    /// explicitly does NOT carry (see that variant's "Schema
    /// simplification vs Python" doc-comment — the
    /// `granted_triggered_ability` rider was deferred to a follow-up
    /// slice; this IS that slice for the Perigee shape). Mirrors
    /// Python's
    /// `PumpTargetCreatureUntilEOT(power_bonus=2, you_control_only=
    /// True, exclude_self=True, granted_triggered_ability=
    /// TriggeredAbility(effect=PerigeeBeckonerDiesReturnTappedEffect(),
    /// event=Event.DIES, predicate=IsSelf()))`.
    ///
    /// **Target semantics — first-slice deterministic, ANOTHER /
    /// you-control.** Recipient = the FIRST creature on the source-
    /// controller's battlefield (smallest Vec index) that is NOT the
    /// source object itself (Perigee's "ANOTHER target creature you
    /// control"). Empty / only-self → silent fizzle (mirrors Python's
    /// `optional`-free `if not legal_targets: return`). Same
    /// first-creature-on-own-bf scan `PumpTargetCreatureUntilEOT`
    /// uses, plus the source-id exclusion.
    ///
    /// **Mechanism.** Pushes BOTH a `UeotModifier::PowerToughnessBonus
    /// { power: power_bonus, toughness: toughness_bonus }` AND a
    /// `UeotModifier::GrantedDiesReturnTapped` marker onto the chosen
    /// target. The marker is honored by the SBA destroy path
    /// (`state_ops_sba.rs`): if the carrier is destroyed via SBA it
    /// is returned from its owner's graveyard to its OWNER's
    /// battlefield, tapped, dropping ONLY the
    /// `GrantedDiesReturnTapped` marker (no re-fire on a 2nd death)
    /// while KEEPING the `PowerToughnessBonus` — Python's
    /// source-of-truth behavior, since its
    /// `PerigeeBeckonerDiesReturnTappedEffect` re-uses the same object
    /// and `put_into_battlefield` leaves `temp_power_bonus` intact
    /// (round-53 effective-state lesson; see state_ops_sba.rs). Both
    /// modifiers auto-expire at the next CR 514.3 UEOT-clear pass, so
    /// a target that survives the turn loses the rider at cleanup
    /// exactly as Python's `EndOfTurn`-bounded grant does.
    ///
    /// **Honest split (round 60).** This closes the Effect half
    /// byte-equal (pump + the fixed Perigee dies-return-tapped
    /// rider). The GENERAL nested-trigger-grant subsystem (an
    /// arbitrary granted triggered ability via a Rust layer-6
    /// registry analogous to Python's `continuous_effects`
    /// `grant_triggered_ability`) is the deferred multi-round half;
    /// the full Perigee Beckoner card stays loader-gated (also
    /// `warp_cost`, a §2-declined cost-mechanic). See
    /// docs/SEARCH_LIMITATIONS.md round-60.
    PumpTargetCreatureUntilEotGrantDiesReturnTapped {
        power_bonus: i32,
        toughness_bonus: i32,
    },
    /// **BLOCKED-RUST-EFFECT slice — Fight effect arm.** "Target
    /// creature you control fights target creature an opponent
    /// controls" (CR 701.12). Both fighters deal damage equal to
    /// their power to the other simultaneously; per CR 701.12c, if
    /// either fighter has left the battlefield by resolve time, no
    /// damage is dealt at all (the fight fizzles).
    ///
    /// Mirrors Python's `FightTargetCreatureWithTargetCreature`
    /// generic-fight class shipped in the same slice (a sibling of
    /// `EnchantedCreatureFightsTargetEffect`, which is the
    /// Aura-attached-creature-fights-opponent shape used by
    /// Meltstrider's Resolve specifically).
    ///
    /// **Target semantics — first-slice deterministic.** The
    /// SOURCE-side fighter is the FIRST creature on the source-
    /// controller's battlefield (smallest Vec index). The
    /// OPPONENT-side fighter is the FIRST creature on the
    /// opponent's battlefield (1 - source_controller, smallest
    /// Vec index). This mirrors the same first-creature-on-bf
    /// scan PumpTargetCreatureUntilEOT (caster side) and
    /// DealDamageAnyTarget (opponent side) use. Neither side
    /// has explicit target choice today; the future decision-
    /// callback slice plumbs explicit target picks through.
    ///
    /// **Source-controller resolution.** Routes through the
    /// I67 controller-of-effect substrate
    /// (`current_resolve_source_controller` with active_player
    /// fallback) so cross-controller fight (a flicker re-ETB
    /// firing while the OTHER seat is active) routes correctly:
    /// the source-side scan is on the SOURCE'S controller's bf,
    /// not the active player's bf. Same shape as the
    /// PumpTargetCreatureUntilEOT cross-controller routing.
    ///
    /// **Mechanism.** Compute both fighters' effective power BEFORE
    /// applying any damage (CR 701.12 — simultaneous damage event).
    /// If both targets exist at resolve time, deal each fighter's
    /// power as damage to the other via `deal_damage` (the per-
    /// object damage marker the `state_ops_damage` slice exposes).
    /// SBAs are not run inline by this arm; they fire on the next
    /// SBA pass after the resolve, by which point both damage
    /// markers are in place — deathtouch-style "both die" outcomes
    /// surface when the SBA loop runs.
    ///
    /// **Fizzle cases (silent — no damage applied).**
    /// - Source-controller untrackable (source object not in any
    ///   tracked zone) — silent fizzle.
    /// - Source-controller has no creatures on bf — silent fizzle.
    /// - Opponent has no creatures on bf — silent fizzle.
    /// - Either fighter's printing carries `power: None` (non-
    ///   creature, edge case the creature-filter scan should
    ///   prevent but defensive `unwrap_or(0)` keeps us safe).
    ///
    /// **Schema simplification vs Python's full Fight family.**
    /// Python's full Fight class hierarchy carries variants for
    /// "fights up to one" (optional target), "fights with trample
    /// / deathtouch", "fights twice", and the Aura-shape
    /// (enchanted creature fights). The Rust first-slice only
    /// carries the bare bilateral-fight shape ("creature you
    /// control fights creature opponent controls"). The Aura
    /// shape (Meltstrider's Resolve specifically) is documented
    /// as a deferred follow-up in
    /// `docs/EOE_PROBE_READINESS_AUDIT.md` — it needs the
    /// Aura-attachment substrate routed through the source's
    /// `attached_to` field, which the current Aura ETB-trigger
    /// path does not yet plumb end-to-end on Rust. Other
    /// variants (with-trample, with-deathtouch, optional)
    /// compose via future Effect arms.
    ///
    /// **Empty-bf no-op shape.** `(0, 0)` doesn't apply here (no
    /// numeric fields) — the fizzle on missing fighters is the
    /// no-op equivalent.
    FightTargetCreatureWithTargetCreature,
    /// **BLOCKED-RUST-EFFECT slice (round 31) — double target
    /// creature's power until end of turn.** Mirrors Python's
    /// `DoubleTargetCreaturesPowerUEOT` (`effects/generic/pump.py:194`
    /// — Mightform Harmonizer's landfall trigger; 1 EoE card).
    ///
    /// **Mechanism — snapshot-pump.** Read the target's effective
    /// power at RESOLUTION TIME (so any prior pumps / counters / base
    /// PT overrides this turn already factor in), then push a UEOT
    /// `PowerToughnessBonus { power: snapshot, toughness: 0 }`
    /// modifier onto the target. The bonus is fixed at the snapshot
    /// — further counter gains or pumps later this turn do NOT
    /// re-double (matches Python's class docstring: "the bonus is a
    /// one-time snapshot at resolution"). The modifier auto-expires
    /// at the next cleanup step via the r27 CR 514.3 UEOT-clear pass.
    ///
    /// **Target semantics — first-slice deterministic.** The recipient
    /// is the FIRST creature on the SOURCE'S CONTROLLER's battlefield
    /// (smallest Vec index). Mirrors `PumpTargetCreatureUntilEOT`'s
    /// you-control target convention — Mightform Harmonizer's oracle
    /// reads "double the power of target creature **you control**".
    /// Empty bf / no creatures → silent fizzle.
    ///
    /// **Source-controller resolution (I79 substrate).** Routes
    /// through `current_resolve_source_controller` with active_player
    /// fallback. Same shape as `PumpTargetCreatureUntilEOT`'s caster-
    /// side scan.
    ///
    /// **Snapshot ≤ 0 fizzle.** If the chosen target's effective power
    /// at resolve time is ≤ 0, no modifier is pushed (matches Python's
    /// `if snapshot <= 0: return` guard at `pump.py:235`). Doubling
    /// a 0/X creature is a no-op; doubling a -N power would invert via
    /// addition (subtracting again), which Python explicitly avoids.
    ///
    /// **Effective power read.** Uses the private
    /// `read_battlefield_creature_power` helper at
    /// `state_ops_resolve.rs:755` (already in place from the round-17
    /// Fight slice) — that helper inlines `effective_power` to dodge
    /// the per-slice file-isolation borrow split.
    ///
    /// **Schema simplification vs Python.** Unit variant on both
    /// sides; no fields beyond `kind`. Python's class has no
    /// constructor args either (`@dataclass(frozen=True)` with only
    /// inherited Effect fields).
    DoubleTargetCreaturesPowerUEOT,
    /// **BLOCKED-RUST-EFFECT slice (round 32) — deal damage to
    /// target creature an opponent controls equal to the number of
    /// artifacts the source's controller controls.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByArtifactCount`
    /// (`effects/generic/damage.py:406` — Warmaker Gunship's ETB; 1
    /// EoE consumer). First dynamic-X-from-permanent-count primitive
    /// in Rust — pre-positions the X-from-count plumbing for sister
    /// arms (e.g. damage-equal-to-creatures-you-control,
    /// X-equal-to-permanents-controlled).
    ///
    /// **Mechanism — count + apply.** At resolve time, count the
    /// source-controller's battlefield permanents whose printing's
    /// `card_types` contains "Artifact" (matches Python: lands that
    /// are also artifacts count, tokens like Lander count). If
    /// the count is ≤ 0 → silent fizzle. Otherwise apply that
    /// damage amount to the FIRST creature on any opponent's
    /// battlefield (mirrors `DealDamageToOpponentCreature`'s target
    /// shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors `DealDamageToOpponentCreature`'s shape — needs to
    /// resolve even if source has left bf, e.g. counterspelled
    /// trigger that still drives the count from the original ETB
    /// snapshot). If untrackable → silent fizzle.
    ///
    /// **Target semantics — first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent
    /// creature → silent fizzle.
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X is the whole point of the class).
    DealDamageToOpponentCreatureByArtifactCount,
    /// **BLOCKED-RUST-EFFECT slice (round 51) — Monoist Circuit-Feeder
    /// ETB (real EoE card; 14,727 17lands decks).** Until end of turn:
    /// a creature the source-controller controls gets `+X/+0`; a
    /// creature an opponent controls gets `-0/-X`; X = the number of
    /// artifacts the source-controller controls. Mirrors Python's NEW
    /// deterministic-pick class `PumpYoursAndDebuffOpponentByArtifact
    /// CountUntilEOT` (`effects/generic/pump.py` — shipped in this
    /// slice as the BRIDGE SIBLING of the agent-driven, two-required-
    /// target `PairedTargetPumpDebuffUEOT` that the real-card parser
    /// emits). This is the exact `FightTargetCreatureWithTargetCreature`
    /// (deterministic Rust unit-variant + deterministic Python bridge
    /// sibling) vs `EnchantedCreatureFightsTargetEffect` (agent-driven
    /// real-card class) precedent: the deterministic class exists
    /// purely as the cross-engine bridge target; the genuine real-EoE
    /// probe drives the actual parsed `PairedTargetPumpDebuffUEOT`
    /// constrained by a picker shim so the agent's two-target pick is
    /// forced to the FIRST own / FIRST opponent creature, byte-equal
    /// with this arm's deterministic scan.
    ///
    /// **Why this is NOT loader-admitted (§2 bridge-correctness).** The
    /// real Monoist Circuit-Feeder Python card parses to the
    /// AGENT-DRIVEN `PairedTargetPumpDebuffUEOT` (two REQUIRED targets,
    /// CR 601.2c; the controller freely picks ANY legal own creature +
    /// ANY legal opponent creature). Routing that through the
    /// production-deck loader's deterministic-first-creature bridge
    /// would SILENTLY degrade free agent targeting to a fixed
    /// first-creature pick in multi-creature board states — a §2
    /// silent-mistranslation. Per the Fight precedent (the
    /// deterministic `FightTargetCreatureWithTargetCreature` is NOT
    /// loader-admitted for any real card; Meltstrider's Resolve stays
    /// deferred), this slice ships the Effect arm + bridge sibling +
    /// GENUINE real-EoE probe but deliberately does NOT add a
    /// `_SAFE_*` loader admit. Documented in
    /// `docs/SEARCH_LIMITATIONS.md`.
    ///
    /// **Mechanism — count + apply.** At resolve time, resolve the
    /// source-controller (5-zone + stack scan; mirrors round-32
    /// `DealDamageToOpponentCreatureByArtifactCount` shape — must
    /// resolve even if the source has left bf). Count the
    /// source-controller's battlefield permanents whose printing's
    /// `card_types` contains "Artifact" (matches Python: artifact-lands
    /// and artifact tokens count). Clamp `x = count.max(0)`; if
    /// `x <= 0` → silent no-op (matches the round-32..49 X-from-count
    /// zero short-circuit and Python's `if x <= 0: return`). Otherwise:
    /// pick the source-side target = FIRST creature on the
    /// source-controller's bf (smallest Vec index); pick the
    /// opponent-side target = FIRST creature on the opponent's bf.
    /// Each side independently: if missing, that side is a silent
    /// no-op and the OTHER side still applies (CR 608.2c — two
    /// independent "target" clauses, NOT "up to one"; matches Python's
    /// per-slot independent apply). Apply an until-EOT
    /// `PowerToughnessBonus` modifier: source side `{ power: x,
    /// toughness: 0 }`; opponent side `{ power: 0, toughness: -x }`.
    /// End-of-turn cleanup is the existing UEOT-modifier expiry path.
    ///
    /// **Target semantics — first-slice deterministic.** Same
    /// first-creature-on-bf scan `FightTargetCreatureWithTargetCreature`
    /// / `PumpTargetCreatureUntilEOT` use (the established campaign
    /// no-decision-callback convention). The genuine real-EoE probe's
    /// picker shim mirrors this exact order on the agent-driven side.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — X
    /// is computed from state at resolve time, not stored on the
    /// variant. Python's `@dataclass(frozen=True)` deterministic class
    /// has no constructor args. Mirrors the round-32..49 unit-variant
    /// schema convention.
    TwoSubjectPumpAndDebuffByArtifactCount,
    /// **BLOCKED-RUST-EFFECT slice (round 35) — deal damage to target
    /// creature an opponent controls equal to the number of cards in
    /// the source-controller's hand.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByCardsInHand`
    /// (`effects/generic/damage.py:464` — Roaring Furnace's door
    /// trigger; 1 EoE consumer). Sister of round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount` — same
    /// apply-substrate (damage-to-opponent-creature, first-eligible
    /// target, fizzle-on-empty-bf), NEW denominator (hand size in
    /// place of artifact count). Establishes a SECOND orthogonal
    /// X-from-count denominator on the same apply substrate, after
    /// round-32 / round-33 / round-34 walked the substrate axis with
    /// the artifact-count denominator.
    ///
    /// **Mechanism — count + apply.** At resolve time, count the
    /// source-controller's hand size (`state.players[idx].hand.len()`)
    /// — matches Python's `len(state.player_by_id(controller_id).hand)`
    /// at damage.py:511. Note: the source itself is NOT in the
    /// controller's hand at resolve time (it's already on the
    /// battlefield by the time the door trigger / ETB resolves), so a
    /// hand-of-N + the cast Room produces N damage, not N+1. If the
    /// hand size is ≤ 0 → silent fizzle. Otherwise apply that damage
    /// amount to the FIRST creature on any opponent's battlefield
    /// (mirrors `DealDamageToOpponentCreature` /
    /// `DealDamageToOpponentCreatureByArtifactCount` target shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 `DealDamageToOpponentCreatureByArtifactCount`
    /// shape — needs to resolve even if source has left bf, e.g.
    /// counterspelled trigger that still drives the count from the
    /// original ETB snapshot). If untrackable → silent fizzle.
    ///
    /// **Target semantics — first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent
    /// creature → silent fizzle. BIT-FOR-BIT identical to round-32's
    /// target shape (same apply substrate).
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X from hand size is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 unit-variant
    /// schema convention.
    DealDamageToOpponentCreatureByCardsInHand,
    /// **BLOCKED-RUST-EFFECT slice (round 36) — deal damage to target
    /// creature an opponent controls equal to the number of LANDS the
    /// source-controller controls.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByLandsControlled`
    /// (`effects/generic/damage.py` — shipped in this slice; no
    /// pre-existing class). Sister of round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount` and round-35's
    /// `DealDamageToOpponentCreatureByCardsInHand` — same
    /// apply-substrate (damage-to-opponent-creature, first-eligible
    /// target, fizzle-on-empty-bf), NEW denominator (lands-controlled
    /// in place of artifact-count / hand-size). Walks the
    /// X-from-count denominator axis to a THIRD entry after round-35
    /// established the bidirectionally-extensible plumbing across
    /// BOTH axes (substrate × denominator).
    ///
    /// **Mechanism — count + apply.** At resolve time, count the
    /// source-controller's lands on bf
    /// (`printing.card_types.iter().any(|t| t == "Land")`) — matches
    /// Python's `sum(1 for o in state.battlefield if o.controller_id ==
    /// controller_id and o.printing.is_land)` shape (mirrors the
    /// round-32 artifact-count denominator BIT-FOR-BIT, replacing
    /// "Artifact" with "Land"). Scryfall's `card_types` convention
    /// lists "Land" for any land (including non-basic lands; basic
    /// lands also carry "Land" as their sole entry; dual-typed
    /// artifact-lands carry both "Land" and "Artifact" -- they count
    /// for BOTH the artifact-count and the lands-controlled
    /// denominators, mirroring Scryfall's typeline conventions and
    /// Python's `is_land` / `is_artifact` disjunction at the printing
    /// level). If the lands count is ≤ 0 → silent fizzle. Otherwise
    /// apply that damage amount to the FIRST creature on any opponent's
    /// battlefield (mirrors `DealDamageToOpponentCreature` /
    /// `DealDamageToOpponentCreatureByArtifactCount` /
    /// `DealDamageToOpponentCreatureByCardsInHand` target shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 / round-35 sister shape — needs to resolve
    /// even if source has left bf, e.g. counterspelled trigger that
    /// still drives the count from the original ETB snapshot). If
    /// untrackable → silent fizzle.
    ///
    /// **Target semantics — first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent
    /// creature → silent fizzle. BIT-FOR-BIT identical to round-32's
    /// / round-35's target shape (same apply substrate).
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X from lands count is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35
    /// unit-variant schema convention.
    DealDamageToOpponentCreatureByLandsControlled,
    /// **BLOCKED-RUST-EFFECT slice (round 37) -- deal damage to target
    /// creature an opponent controls equal to the number of CREATURES
    /// the source-controller controls.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByCreaturesYouControl`
    /// (`effects/generic/damage.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount`, round-35's
    /// `DealDamageToOpponentCreatureByCardsInHand`, and round-36's
    /// `DealDamageToOpponentCreatureByLandsControlled` -- same
    /// apply-substrate (damage-to-opponent-creature, first-eligible
    /// target, fizzle-on-empty-bf), NEW denominator
    /// (creatures-controlled in place of artifact-count / hand-size /
    /// lands-controlled). Walks the X-from-count denominator axis to a
    /// FOURTH entry, completing the artifact / hand / lands / creatures
    /// quadruplet on this apply substrate. Confirms the round-35
    /// bidirectionally-extensible X-from-count plumbing generalizes
    /// across FOUR denominators on the same apply substrate.
    ///
    /// **Mechanism -- count + apply.** At resolve time, count the
    /// source-controller's creatures on bf
    /// (`printing.card_types.iter().any(|t| t == "Creature")`) --
    /// matches Python's `sum(1 for o in state.battlefield if
    /// o.controller_id == controller_id and o.printing.is_creature)`
    /// shape (mirrors the round-32 artifact-count denominator
    /// BIT-FOR-BIT, replacing "Artifact" with "Creature"). The
    /// source-controller's OWN creatures count toward the
    /// denominator (mirrors Python's vanilla `is_creature` walk over
    /// `state.battlefield` -- including the source if the source is a
    /// creature still on the bf). Dual-typed artifact-creatures count
    /// for BOTH the artifact-count and the creatures-controlled
    /// denominators independently (Scryfall lists both "Artifact" and
    /// "Creature" in card_types; Rust checks each type string
    /// independently in the relevant arm; Python's `is_artifact` and
    /// `is_creature` properties are independent disjunctions over
    /// `card_types`). If the creatures count is <= 0 -> silent
    /// fizzle. Otherwise apply that damage amount to the FIRST
    /// creature on any opponent's battlefield (mirrors
    /// `DealDamageToOpponentCreature` /
    /// `DealDamageToOpponentCreatureByArtifactCount` /
    /// `DealDamageToOpponentCreatureByCardsInHand` /
    /// `DealDamageToOpponentCreatureByLandsControlled` target shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 / round-35 / round-36 sister shape -- needs
    /// to resolve even if source has left bf, e.g. counterspelled
    /// trigger that still drives the count from the original ETB
    /// snapshot). If untrackable -> silent fizzle.
    ///
    /// **Target semantics -- first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent
    /// creature -> silent fizzle. BIT-FOR-BIT identical to round-32's
    /// / round-35's / round-36's target shape (same apply substrate).
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X from creatures count is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 unit-variant schema convention.
    DealDamageToOpponentCreatureByCreaturesYouControl,
    /// **BLOCKED-RUST-EFFECT slice (round 46) -- deal damage to target
    /// creature an opponent controls equal to the number of CARDS in
    /// the source-controller's GRAVEYARD.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByCardsInGraveyard`
    /// (`effects/generic/damage.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount`, round-35's
    /// `DealDamageToOpponentCreatureByCardsInHand`, round-36's
    /// `DealDamageToOpponentCreatureByLandsControlled`, and round-37's
    /// `DealDamageToOpponentCreatureByCreaturesYouControl` -- same
    /// apply-substrate (damage-to-opponent-creature, first-eligible
    /// target, fizzle-on-empty-bf), NEW denominator
    /// (cards-in-graveyard in place of artifact-count / hand-size /
    /// lands-controlled / creatures-controlled). Walks the X-from-count
    /// denominator axis to a FIFTH entry on the round-32 damage
    /// substrate, completing the artifact / hand / lands / creatures /
    /// graveyard quintuplet on this apply substrate. SIXTH overall
    /// X-from-count denominator across the substrate matrix (prior
    /// five: artifact-count / hand-size / lands-controlled /
    /// creatures-controlled / life-total -- round-42 mill uses life
    /// total). Confirms the round-35 bidirectionally-extensible
    /// X-from-count plumbing extends to a graveyard-zone-size
    /// denominator (the prior denominators all read from hand, bf, or
    /// life total -- graveyard is the FIRST zone-size denominator
    /// pulled from a NON-battlefield, NON-hand source).
    ///
    /// **Mechanism -- count + apply.** At resolve time, count the
    /// source-controller's graveyard size
    /// (`state.players[idx].graveyard.len()`) -- matches Python's
    /// `len(state.player_by_id(controller_id).graveyard)` shape. The
    /// source itself is on the battlefield at resolve time for an
    /// ETB-style trigger, so the source does NOT contribute to the
    /// denominator (it's not in the graveyard yet). For an LTB-style
    /// or graveyard-resolve trigger where the source is already in
    /// the graveyard, the source WOULD contribute +1 by virtue of
    /// having moved (no source-exclusion clause, mirroring round-32 /
    /// round-35 / round-36 / round-37 sister "vanilla walk -- no
    /// source-exclusion clause" convention). If the graveyard size is
    /// <= 0 -> silent fizzle. Otherwise apply that damage amount to
    /// the FIRST creature on any opponent's battlefield (mirrors
    /// `DealDamageToOpponentCreature` /
    /// `DealDamageToOpponentCreatureByArtifactCount` /
    /// `DealDamageToOpponentCreatureByCardsInHand` /
    /// `DealDamageToOpponentCreatureByLandsControlled` /
    /// `DealDamageToOpponentCreatureByCreaturesYouControl` target
    /// shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 / round-35 / round-36 / round-37 sister
    /// shape -- needs to resolve even if source has left bf, e.g.
    /// counterspelled trigger that still drives the count from the
    /// original ETB snapshot). If untrackable -> silent fizzle.
    ///
    /// **Target semantics -- first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent creature
    /// -> silent fizzle. BIT-FOR-BIT identical to round-32's /
    /// round-35's / round-36's / round-37's target shape (same apply
    /// substrate).
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X from graveyard size is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 unit-variant schema convention.
    DealDamageToOpponentCreatureByCardsInGraveyard,
    /// **BLOCKED-RUST-EFFECT slice (round 47) -- deal damage to target
    /// creature an opponent controls equal to the source-controller's
    /// LIFE TOTAL.** Mirrors Python's
    /// `DealDamageToOpponentCreatureByLifeTotal`
    /// (`effects/generic/damage.py` -- shipped in this slice; no
    /// pre-existing class). Sister of round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount`, round-35's
    /// `DealDamageToOpponentCreatureByCardsInHand`, round-36's
    /// `DealDamageToOpponentCreatureByLandsControlled`, round-37's
    /// `DealDamageToOpponentCreatureByCreaturesYouControl`, and
    /// round-46's `DealDamageToOpponentCreatureByCardsInGraveyard` --
    /// same apply-substrate (damage-to-opponent-creature,
    /// first-eligible target, fizzle-on-empty-bf), NEW denominator
    /// (source-controller's life total in place of artifact-count /
    /// hand-size / lands-controlled / creatures-controlled /
    /// graveyard-size). Walks the X-from-count denominator axis to a
    /// SIXTH entry on the round-32 damage substrate, COMPLETING the
    /// artifact / hand / lands / creatures / graveyard / life-total
    /// sextuplet on this apply substrate (the damage substrate is now
    /// the first fully-saturated row of the 6-substrate x 6-denominator
    /// X-from-count matrix). Confirms the round-35
    /// bidirectionally-extensible X-from-count plumbing extends to the
    /// life-total scalar denominator on the damage substrate (round-42
    /// already drove life-total on the mill substrate; this slice
    /// confirms the same scalar denominator generalizes onto the
    /// damage apply substrate).
    ///
    /// **Mechanism -- count + apply.** At resolve time, read the
    /// source-controller's life total
    /// (`state.players[idx].life`, an `i32`) and clamp negative life
    /// to zero (`life.max(0)`) -- matches Python's
    /// `max(0, state.player_by_id(controller_id).life)` clamp
    /// (mirrors the round-42 `MillTargetPlayerEqualToYourLifeTotal`
    /// `life.max(0)` precedent at state_ops_resolve.rs:1276; negative
    /// life arises only in arbitrary lose-life-past-zero / infect-
    /// style scenarios, an SBA would normally have ended the game --
    /// the clamp is a defensive silent no-op). If the clamped life is
    /// <= 0 -> silent fizzle. Otherwise apply that damage amount to
    /// the FIRST creature on any opponent's battlefield (mirrors
    /// `DealDamageToOpponentCreature` /
    /// `DealDamageToOpponentCreatureByArtifactCount` /
    /// `DealDamageToOpponentCreatureByCardsInHand` /
    /// `DealDamageToOpponentCreatureByLandsControlled` /
    /// `DealDamageToOpponentCreatureByCreaturesYouControl` /
    /// `DealDamageToOpponentCreatureByCardsInGraveyard` target shape).
    ///
    /// **Source-controller resolution.** 5-zone scan + stack scan
    /// (mirrors round-32 / round-35 / round-36 / round-37 / round-46
    /// sister shape -- needs to resolve even if source has left bf,
    /// e.g. counterspelled trigger that still drives the count from
    /// the original ETB snapshot). If untrackable -> silent fizzle.
    ///
    /// **Target semantics -- first-slice deterministic.** Iterate
    /// opponents in ascending player order; pick the FIRST creature
    /// on each opponent's bf (smallest Vec index). 2-player engine =
    /// exactly one opponent at `1 - controller`. No opponent creature
    /// -> silent fizzle. BIT-FOR-BIT identical to round-32's /
    /// round-35's / round-36's / round-37's / round-46's target shape
    /// (same apply substrate).
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// only the inherited `targets` tuple; no static `amount` field
    /// (the dynamic-X from life total is the whole point of the
    /// class). Mirrors round-32 / round-33 / round-34 / round-35 /
    /// round-36 / round-37 / round-46 unit-variant schema convention.
    DealDamageToOpponentCreatureByLifeTotal,
    /// **BLOCKED-RUST-EFFECT slice (round 33) — put a +1/+1 counter
    /// on this creature for each OTHER creature and/or artifact you
    /// control.** Mirrors Python's
    /// `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact`
    /// (`effects/generic/counters.py:394` — Luxknight Breacher's ETB,
    /// 1 EoE consumer). Second dynamic-X-from-permanent-count
    /// primitive in Rust (sister to round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount`); applies the
    /// X-from-count plumbing to the SELF-counter substrate instead
    /// of the opponent-creature damage substrate.
    ///
    /// **Mechanism — count + apply.** At resolve time, locate the
    /// source on a battlefield zone (mirrors `PutPlusOneCounterOnSelf`
    /// shape — round-23 precedent). Walk the source-controller's
    /// battlefield once; count permanents (other than the source
    /// itself) whose printing's `card_types` contains "Creature" OR
    /// "Artifact" (matches Python's `o.printing.is_creature or
    /// o.printing.is_artifact` disjunction at counters.py:417).
    /// Artifact-creatures count once (the `or` short-circuits — same
    /// as Python). If count ≤ 0 → silent fizzle. Otherwise add that
    /// many `+1/+1` counters to the source via `obj.counters
    /// .saturating_add(count)`.
    ///
    /// **Self exclusion.** The source is excluded from the count
    /// mid-loop (Python: `if o.id == source.id: continue`). Luxknight
    /// Breacher itself is on the bf when the ETB resolves, so without
    /// the exclusion the source would count itself as a +1.
    ///
    /// **Source-controller resolution.** Same shape as
    /// `PutPlusOneCounterOnSelf`'s round-23 battlefield-only scan
    /// (no 5-zone walk needed — the source MUST be on a battlefield
    /// zone for a "+1/+1 counter on this creature" effect to do
    /// anything; if it's gone, silent fizzle is correct). Owner /
    /// controller comes off the located GameObject's `controller`
    /// field, mirroring how `PutPlusOneCounterOnSelf` already locates
    /// its source.
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class has
    /// no constructor args (only the inherited `Effect` fields); the
    /// dynamic-X is the whole point of the class.
    PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    /// **BLOCKED-RUST-EFFECT slice (round 34) — surveil X, where X is
    /// the number of artifacts the source's controller controls.**
    /// Mirrors Python's `Surveil(use_artifact_count=True)`
    /// (`effects/generic/library_lookat.py:569` — Cerebral Download's
    /// "Surveil X, where X is the number of artifacts you control"
    /// clause; 1 EoE consumer). Third dynamic-X-from-permanent-count
    /// primitive in Rust (sister to round-32's
    /// `DealDamageToOpponentCreatureByArtifactCount` and round-33's
    /// `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact`); applies
    /// the X-from-artifact-count denominator (same as round-32) to the
    /// SURVEIL substrate (round-15 `Effect::Surveil`).
    ///
    /// **Mechanism — count + apply.** At resolve time, count the
    /// source-controller's battlefield permanents whose printing's
    /// `card_types` contains "Artifact" (matches Python: lands that
    /// are also artifacts count, tokens like Lander count). Mirrors
    /// the round-32 `DealDamageToOpponentCreatureByArtifactCount`
    /// counting denominator BIT-FOR-BIT — same scan, same predicate,
    /// same `saturating_add` overflow guard. Then pop
    /// `min(count, library_len)` cards off the controller's library
    /// top into a `SurveilChoice` PendingDecision — IDENTICAL post-pop
    /// state shape to the round-15 `Effect::Surveil` resolve arm at
    /// `state_ops_resolve.rs:2692`. Mirrors Surveil's `count = 0`
    /// invariant: every Surveil emits exactly one PendingDecision
    /// (auto-resolve discards an empty SurveilChoice in O(1)) so the
    /// caller's "every Surveil emits exactly one PendingDecision"
    /// contract holds even when the controller has zero artifacts.
    ///
    /// **Source-controller resolution.** Try the
    /// `current_resolve_source_controller` transient slot first
    /// (mirrors round-15 `Effect::Surveil` shape — set by the
    /// resolve-time harness for triggers / cast effects). Fall back
    /// to a 5-zone scan + stack scan if the slot isn't stamped
    /// (mirrors `DealDamageToOpponentCreatureByArtifactCount`
    /// robustness for synthetic harnesses driving `apply_effect_pub`
    /// directly). Silent fizzle (zero-card SurveilChoice with the
    /// active player as controller) only if both fail — preserves the
    /// "always emit exactly one PendingDecision" contract.
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the count is computed from state at resolve time, not stored
    /// on the variant. Python's `@dataclass(frozen=True)` class HAS
    /// fields (`amount: int = 1` and `use_artifact_count: bool =
    /// False`) but the dynamic-X variant is selected at construct
    /// time by `use_artifact_count=True`; the `amount` field is
    /// ignored on the dynamic lane. Rust's unit variant carries no
    /// fields — the dispatch happens on the variant tag, not a runtime
    /// flag, which is the same shape round-32 / round-33 use. The
    /// bridge translator constructs the Python class with
    /// `use_artifact_count=True` set, ignoring any `count` / `amount`
    /// field on the dict (mirrors round-32 / round-33's extra-key
    /// tolerance).
    SurveilXEqualToArtifactCount,
    /// **Round 28 highest-frequency-class coverage.** Run multiple
    /// effects in order as a single composite effect. Mirrors
    /// Python's `Sequence(effects: tuple[Effect, ...])`
    /// (`src/mtg_bot/effects/combinators.py:1314`) — 54 cards in the
    /// EoE pool use it (highest-frequency Python class not yet in
    /// Rust per r27 measurement).
    ///
    /// **Mechanism.** At resolve time, walks the `effects` Vec and
    /// calls `apply_effect` on each child in order with the same
    /// `source_obj_id`. Children share resolve-time state mutations
    /// (e.g. a `Sequence` of `[DrawCards{count:1}, DiscardSelf]`
    /// resolves to "draw, then discard" — a looting effect).
    ///
    /// **Targeting simplification.** Python's `Sequence` carries
    /// cast-time targeting machinery (at most one child may require
    /// a target; the target is passed to that child at resolve).
    /// The Rust first-slice does NOT carry cast-time target plumbing
    /// for nested effects — each child resolves with the same
    /// first-slice deterministic-target convention it uses on its own
    /// (e.g. `TapTarget{AnyCreature}` picks defender's bf[0],
    /// independent of the Sequence wrapper).
    ///
    /// **Recursion.** A Sequence can contain Sequences — the variant
    /// is fully nestable because `apply_effect` is a regular function
    /// and the `Vec<Effect>` type is heap-allocated.
    ///
    /// **Empty Sequence.** A `Sequence { effects: vec![] }` is a
    /// legal no-op (matches Python's empty-tuple-initialized
    /// Sequence).
    Sequence { effects: Vec<Effect> },
    /// **Round 28 UEOT-coverage extension.** Every creature on the
    /// source's controller's battlefield gets `+power/+toughness`
    /// until end of turn. Mirrors Python's
    /// `PumpCreaturesYouControlUntilEOT(power_bonus, toughness_bonus, ...)`
    /// (6 EoE cards — Stage-3 anthem-shape effects).
    ///
    /// **Mechanism.** Locate the source's controller; iterate that
    /// player's battlefield; for every object whose printing is a
    /// creature, push a `UeotModifier::PowerToughnessBonus` onto the
    /// object's `ueot_modifiers` Vec. Effective P/T queries pick up
    /// the bonus; cleanup-step expiration drains the modifiers at the
    /// end of the turn (CR 514.3 via the r27 UEOT framework).
    ///
    /// **Schema simplification.** Python's class also carries
    /// `granted_keywords`, `exclude_source`, `attacking_only`, and
    /// `also_untap_if_not_your_turn`. The Rust first slice carries
    /// ONLY power/toughness. Cards with riders wait for richer
    /// composition. Negative bonuses are legal (controller-side
    /// debuff pumps); `(0, 0)` is a legal no-op for parity edge cases.
    PumpCreaturesYouControlUntilEOT { power_bonus: i32, toughness_bonus: i32 },
    /// **Round 28 coverage extension.** Return the first creature card
    /// from the source-controller's graveyard to that player's hand.
    /// Mirrors Python's `ReturnCreatureFromYourGraveyardToHand`
    /// (`bounce.py:356` — 3 EoE cards).
    ///
    /// **First-slice deterministic target choice.** Same shape as the
    /// existing `ReturnCreatureFromYourGraveyardToBattlefield` arm —
    /// scans the active-player's graveyard front-to-back, picks the
    /// first creature card. Empty / no-creature graveyard fizzles
    /// silently.
    ///
    /// **Schema simplification.** Drops `max_mv` / `or_spacecraft` /
    /// `or_artifact` Python riders.
    ///
    /// **Zone-change reset (CR 400.7).** `tapped`, `summoning_sick`,
    /// `damage`, `counters` reset on move to hand. summoning_sick
    /// matches the round-28 CE1 universal-true default for hand-zone
    /// objects.
    ReturnCreatureFromYourGraveyardToHand,
    /// **Phase 5 rider-widened return-to-hand (Option B — additive
    /// bespoke arm).** Return the first creature OR artifact card
    /// from the source-controller's graveyard to that player's hand.
    /// Mirrors Python's
    /// `ReturnCreatureFromYourGraveyardToHand(or_artifact=True)`
    /// (`bounce.py:356`) — the rider widens the legal-target pool
    /// to admit artifact cards alongside creatures.
    ///
    /// **EoE consumer (audit).** Interceptor Mechan (EoE): "When this
    /// creature enters, return target artifact or creature card from
    /// your graveyard to your hand."
    ///
    /// **Filter contract.** A graveyard card qualifies iff its
    /// printing's `card_types` contains "Creature" OR "Artifact".
    /// (CR 400 / 506 — type predicates read off the printing's
    /// static card_types; the Rust registry stores card_types as
    /// the Scryfall type-line components on the LHS of the em-dash.)
    /// Sister artifact-creatures (printing carrying BOTH "Artifact"
    /// AND "Creature") match the disjunction at the first hit and
    /// are pulled. Cards with neither type are skipped. The scan
    /// preserves the unit-variant's iterator order
    /// (state.players[controller_idx].graveyard.iter().find_map),
    /// so test predictability matches the existing arm exactly.
    ///
    /// **Controller-of-effect.** Mirrors the I82 fix: source/dest
    /// derive from `current_resolve_source_controller`
    /// (falls back to active_player for direct apply_effect_pub
    /// callers). On p0's turn, p1's controlled trigger scans p1's
    /// graveyard and pushes to p1's hand.
    ///
    /// **Zone-change reset (CR 400.7).** Same as the bare unit
    /// variant: `tapped`, `summoning_sick`, `damage`, `counters`,
    /// `attached_to`, `exiled_with_source` all reset on move to hand.
    /// summoning_sick=true matches the round-28 CE1 universal-true
    /// default for hand-zone objects.
    ///
    /// **Schema simplification vs Python.** Drops `max_mv` and the
    /// other rider flags (`or_spacecraft`, `or_enchantment`,
    /// untargeted_all, plus_one_counter, log_suffix). The current
    /// EoE consumer pool needs the bare `or_artifact` widening only;
    /// richer Python axes re-enter as separate parametric Rust
    /// variants when their first real consumer arrives.
    ReturnArtifactOrCreatureFromYourGraveyardToHand,
    /// **Phase 5 rider-widened return-to-hand (Option B — additive
    /// bespoke arm).** Return the first creature OR Spacecraft card
    /// from the source-controller's graveyard to that player's hand.
    /// Mirrors Python's
    /// `ReturnCreatureFromYourGraveyardToHand(or_spacecraft=True)`
    /// (`bounce.py:356`) — the rider widens the legal-target pool
    /// to admit Spacecraft cards alongside creatures.
    ///
    /// **EoE consumer (audit).** Fell Gravship (EoE) suffix: "...
    /// then return a creature or Spacecraft card from your graveyard
    /// to your hand." (Composed inside a Sequence with mill-3 today;
    /// the bare arm here is the leaf-effect substrate.)
    ///
    /// **Filter contract.** A graveyard card qualifies iff EITHER
    /// its printing's `card_types` contains "Creature" OR its
    /// printing's `subtypes` contains "Spacecraft". Spacecraft is a
    /// SUBTYPE (an artifact subtype on the type line, e.g.
    /// "Artifact — Spacecraft"), not a card type — Python's
    /// `printing.is_spacecraft` reads off `"Spacecraft" in subtypes`
    /// at `cards.py` (Subtypes substrate). The Rust mirror reads
    /// `printing.subtypes.iter().any(|s| s == "Spacecraft")` directly
    /// off the static registry, matching the existing subject-has-
    /// subtype helpers at `cards.rs:2141`. No effective_subtypes
    /// layer yet (CR 613 layer 4 — type-modifying continuous
    /// effects). The disjunction is short-circuit: a creature-but-
    /// not-Spacecraft card pulls on the creature side; a Spacecraft-
    /// but-not-creature (e.g. an unactivated artifact-Spacecraft
    /// like Fell Gravship itself before Station) pulls on the
    /// Spacecraft side.
    ///
    /// **Controller-of-effect.** Mirrors the I82 fix: source/dest
    /// derive from `current_resolve_source_controller`. Same
    /// behavior as the bare unit variant and the
    /// or-artifact-rider variant.
    ///
    /// **Zone-change reset (CR 400.7).** Same as the sibling
    /// variants.
    ///
    /// **Schema simplification vs Python.** Drops `max_mv` and other
    /// rider flags (`or_artifact`, `or_enchantment`, untargeted_all,
    /// plus_one_counter, log_suffix). Bare `or_spacecraft` only.
    ReturnCreatureOrSpacecraftFromYourGraveyardToHand,
    /// **Phase 5 rider-widened return-to-hand (Option B — additive
    /// bespoke arm).** Return the first creature OR enchantment card
    /// from the source-controller's graveyard to that player's hand.
    /// Mirrors Python's
    /// `ReturnCreatureFromYourGraveyardToHand(or_enchantment=True)`
    /// (`bounce.py:356`) — the rider widens the legal-target pool
    /// to admit enchantment cards alongside creatures.
    ///
    /// **EoE consumer (audit).** Rescue Skiff (EoE) carries the
    /// "creature or enchantment card from your graveyard" rider on
    /// its ETB trigger but with the destination set to the
    /// battlefield (not hand); the destination-to-hand variant of
    /// the rider is the natural sibling of the existing
    /// `or_artifact` / `or_spacecraft` to-hand arms and lands here
    /// as part of the to-hand rider cluster. A separate
    /// `ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield`
    /// arm (mirror of this one with destination=battlefield) is the
    /// future slice that closes the loop on Rescue Skiff itself.
    ///
    /// **Filter contract.** A graveyard card qualifies iff its
    /// printing's `card_types` contains "Creature" OR "Enchantment".
    /// (CR 400 / 506 — type predicates read off the printing's
    /// static card_types; the Rust registry stores card_types as
    /// the Scryfall type-line components on the LHS of the em-dash.)
    /// Sister enchantment-creatures (printing carrying BOTH
    /// "Enchantment" AND "Creature") match the disjunction at the
    /// first hit and are pulled. Cards with neither type are
    /// skipped. The scan preserves the unit-variant's iterator
    /// order (`state.players[controller_idx].graveyard.iter().find_map`),
    /// so test predictability matches the existing arms exactly.
    ///
    /// **Controller-of-effect.** Mirrors the I82 fix: source/dest
    /// derive from `current_resolve_source_controller`
    /// (falls back to active_player for direct apply_effect_pub
    /// callers). On p0's turn, p1's controlled trigger scans p1's
    /// graveyard and pushes to p1's hand.
    ///
    /// **Zone-change reset (CR 400.7).** Same as the sibling
    /// variants: `tapped`, `summoning_sick`, `damage`, `counters`,
    /// `attached_to`, `exiled_with_source` all reset on move to
    /// hand. summoning_sick=true matches the round-28 CE1
    /// universal-true default for hand-zone objects.
    ///
    /// **Schema simplification vs Python.** Drops `max_mv` and the
    /// other rider flags (`or_artifact`, `or_spacecraft`,
    /// untargeted_all, plus_one_counter, log_suffix). Bare
    /// `or_enchantment` only.
    ReturnEnchantmentOrCreatureFromYourGraveyardToHand,
    /// **scour-for-scrap-cycle-2 (2026-05-21).** Return target artifact
    /// card from the source-controller's graveyard to that player's
    /// hand. Mirrors Python's
    /// `ReturnTargetArtifactCardFromYourGraveyardToHand`
    /// (`effects/generic/bounce.py:338`) — constructor sugar over
    /// `ReturnFromGraveyard(artifact=True, destination="hand",
    /// target_kind="graveyard_artifact")`. Cycle-2 ship-target for
    /// Scour for Scrap's mode-1 leaf — the missing per-leaf Effect
    /// variant whose absence kept Scour DECLINED at the bridge-admit
    /// gate (Lane LL substrate-only landing).
    ///
    /// **Cast-time-bound target shape.** The cast-time per-mode
    /// detector at `state_ops_cast.rs::cast_spell_noncreature` (when
    /// the chosen mode resolves to this variant) binds the
    /// deterministic-first legal target — first artifact card in the
    /// caster's graveyard in iteration order — onto
    /// `current_resolve_payload_target_id`. The resolve arm reads that
    /// payload, re-validates per CR 608.2b (still in graveyard, still
    /// an artifact card), and moves it to the caster's hand. Silent
    /// fizzle on validation fail (target left the graveyard between
    /// cast and resolve, or stopped being an artifact card — defense
    /// in-depth since the printing is immutable).
    ///
    /// **Filter contract.** A graveyard card qualifies iff its
    /// printing's `card_types` contains "Artifact". Mirrors Python's
    /// `ReturnFromGraveyard._filter`'s `printing.is_artifact`
    /// reading and the `target_kind == "graveyard_artifact"` slot
    /// scoping. Artifact-creatures (printings carrying BOTH
    /// "Artifact" AND "Creature") qualify; non-artifact cards
    /// (creatures-only, enchantments, lands) do not.
    ///
    /// **Controller-of-effect.** Inherits the I82 contract:
    /// `controller_idx` derives from
    /// `current_resolve_source_controller`, falling back to
    /// `active_player` for direct apply_effect_pub callers. For the
    /// production Scour cast path the controller IS the caster (per-
    /// mode binding routes through the spell's StackItem.controller
    /// chain).
    ///
    /// **Zone-change reset (CR 400.7).** Same as the sibling
    /// return-to-hand variants: `tapped`, `summoning_sick`, `damage`,
    /// `counters`, `attached_to`, `exiled_with_source` all reset on
    /// move to hand. `summoning_sick=true` matches the round-28 CE1
    /// universal-true default for hand-zone objects.
    ///
    /// **Real EoE consumer.** Scour for Scrap (EoE Instant, {3}{U}):
    /// "Choose one or both — Search your library for an artifact card,
    /// reveal it, put it into your hand, then shuffle. // Return
    /// target artifact card from your graveyard to your hand." The
    /// or_both substrate (Lane LL's Cycle 1) handles the multi-pick
    /// dispatch; this variant closes mode 1's per-leaf parity for the
    /// bridge auto-grow flip.
    ReturnTargetArtifactCardFromYourGraveyardToHand,
    /// **Round 28 coverage extension.** Put `amount` +1/+1 counter(s)
    /// on each creature the source's controller controls. Mirrors
    /// Python's `PutPlusOneCounterOnEachCreatureYouControl(amount: int = 1)`
    /// (counters.py:737 — Atmospheric Greenhouse-shape, 3 EoE cards).
    ///
    /// **Mechanism.** Locate source controller; iterate that player's
    /// battlefield; for every creature, add `amount` to the per-object
    /// `counters` scalar (the same single-scalar counter the existing
    /// PutPlusOneCounterOnSelf / AddCounters arms use — G8 multi-kind
    /// counter map is still future work). Non-creature permanents are
    /// skipped. (0) is a legal no-op; negative `amount` raises.
    ///
    /// **Schema simplification vs Python.** Drops the `exclude_source`
    /// rider ("each OTHER creature you control" — skips the triggering
    /// permanent). Cards with that rider compose via separate effect
    /// chains for now.
    PutPlusOneCounterOnEachCreatureYouControl { amount: i32 },
    /// **Round 28 coverage extension.** Put `amount` +1/+1 counter(s)
    /// on a target creature the source's controller controls. Mirrors
    /// Python's `PutPlusOneCounterOnCreatureYouControl(amount: int = 1)`
    /// (counters.py:844 — Rayblade-Trooper-shape, 3 EoE cards).
    ///
    /// **First-slice deterministic target choice.** Same shape as
    /// `PumpTargetCreatureUntilEOT`: pick the FIRST creature on the
    /// source-controller's battlefield (zone-iteration order). Empty
    /// battlefield or no eligible creatures = silent fizzle.
    ///
    /// **Schema simplification.** Python's class composes via the
    /// `PutPlusOneCounters` base which carries a `targets` filter
    /// (IsCreature & ControlledByYou). The Rust first slice carries
    /// the implicit shape via the resolve arm's selection logic.
    /// negative `amount` raises; `amount=0` is a legal no-op.
    PutPlusOneCounterOnCreatureYouControl { amount: i32 },
    /// **PPP Pin 2 (10th Rust slice).** Put `amount` +1/+1 counter(s)
    /// on target creature (no controller restriction). Mirrors
    /// Python's `PutPlusOneCounterOnTargetCreature(amount: int = 1,
    /// attacking_only: bool = False)` (`counters.py:637`).
    ///
    /// **Bare "target creature" form.** Distinct from
    /// `PutPlusOneCounterOnCreatureYouControl` (own-side only) and
    /// `PutPlusOneCounterOnOpponentsCreature` (opponent-side only):
    /// this variant has NO controller restriction. Hospital Room
    /// half adds an `IsAttacking` rider via `attacking_only=true`.
    ///
    /// **Payload-target seat (Tier-C convention).** Reads the
    /// push-time-seated target from
    /// `current_resolve_payload_target_id`. The seat is computed by
    /// `first_legal_trigger_target` at `push_triggered_ability` time:
    /// scan `bf_iter_etb_sorted` (smallest-bf-index order across both
    /// players) for the first creature (and attacking, when
    /// `attacking_only=true`). `count_min=1` semantics — when no
    /// legal target exists, `lever_trigger_requires_target` causes
    /// the trigger to fizzle (CR 603.3c).
    ///
    /// **Resolve-time re-legality (CR 608.2b).** Target must still be
    /// on SOME bf and still a Creature (and still attacking when the
    /// rider is set). Failure → silent fizzle, byte-equal with
    /// Python's `PutPlusOneCounters.resolve` `_slot_target_still_legal`.
    ///
    /// **amount<0 raises**; **amount=0 is a legal no-op**.
    ///
    /// **6 EoE cards unblocked**: Drix Fatemaker, Mm'menon Uthros
    /// Exile, Optimistic Scavenger (x2 triggers), Surgical Suite //
    /// Hospital Room (`attacking_only=true`), Shrewd Storyteller.
    PutPlusOneCounterOnTargetCreature {
        amount: i32,
        attacking_only: bool,
    },
    /// **Tier-C target-seating sub-substrate (lever-a, Sunset
    /// Saboteur).** Put `amount` +1/+1 counter(s) on a target
    /// creature an OPPONENT of the source's controller controls.
    /// Mirrors Python's
    /// `PutPlusOneCounterOnOpponentsCreature(amount: int = 1)`
    /// (`effects/generic/counters.py` — a `PutPlusOneCounters`
    /// subclass whose `__post_init__` wires the slot filter
    /// `IsCreature() & ControlledByOpponent()`). Sunset Saboteur's
    /// ATTACKS trigger is the sole EoE consumer (6,375 17lands decks).
    ///
    /// **Payload-target convention (NOT a self-scan).** Unlike the
    /// you-control sibling above (which deterministically scans the
    /// source-controller's own bf at apply time), this variant reads
    /// the push-time-seated target from
    /// `state.current_resolve_payload_target_id` exactly like
    /// `Effect::EquipToTargetCreatureYouControl`
    /// (`state_ops_resolve.rs`). The seat is computed by
    /// `first_legal_trigger_target` at `push_triggered_ability` time
    /// (the Tier-C shared-seat micro-substrate) so BOTH engines pick
    /// the SAME object — the deterministic smallest-bf-index
    /// filter-passing creature an opponent controls. A resolve-time
    /// CR 608.2b re-legality check (target still on a bf, still a
    /// creature, still opponent-controlled) silently fizzles a target
    /// that left the bf between push and resolve, byte-equal with
    /// Python's `PutPlusOneCounters.resolve` `_slot_target_still_legal`
    /// re-check. `amount=0` is a legal no-op; negative raises.
    PutPlusOneCounterOnOpponentsCreature { amount: i32 },
    /// **Tier-C target-seating sub-substrate (lever-a, Exosuit
    /// Savior — the `count_min=0` optional-bounce edge).** Return a
    /// target permanent the source's controller controls (and which
    /// is not the source itself, and is not a player) to its owner's
    /// hand. Mirrors Python's
    /// `ReturnTargetPermanentYouControlToOwnersHand(optional=True,
    /// exclude_self=True)` (`effects/generic/bounce.py` — a `Bounce`
    /// subclass whose `__post_init__` wires the slot filter
    /// `ControlledByYou() & ~IsPlayer() & ~IsSource()` with
    /// `count_min=0`, `count_max=1`). Exosuit Savior's ENTERS trigger
    /// is the sole EoE consumer ({2}{W} 2/2 flying, 12,480 17lands
    /// decks).
    ///
    /// **The `count_min=0` optional edge (contrast Sunset
    /// Saboteur).** `requires_target()` reads `False` (CR "you may
    /// return"), so when NO eligible own permanent exists the shared
    /// seat (`first_legal_trigger_target`) returns `None` and
    /// `push_triggered_ability` takes the UNCHANGED `new_trigger`
    /// path → the trigger pushes with NO target → this arm no-ops
    /// (byte-equal with Python's `targets=()` `Bounce.resolve`
    /// early-`return`). It does NOT fizzle-skip-push (that is Sunset
    /// Saboteur's `count_min>=1` semantics — this variant is
    /// deliberately NOT in `lever_trigger_requires_target`). When an
    /// eligible permanent DOES exist both engines deterministically
    /// pick the SAME smallest-bf-index own-non-source-non-player
    /// permanent and bounce it.
    ///
    /// **Payload-target convention (NOT a self-scan).** Reads the
    /// push-time-seated target from
    /// `state.current_resolve_payload_target_id` like
    /// `Effect::EquipToTargetCreatureYouControl` /
    /// `Effect::PutPlusOneCounterOnOpponentsCreature`, with a
    /// CR 608.2b resolve-time re-legality check (target still on the
    /// controller's bf, not the source, not a player) → silent
    /// fizzle if it became illegal between push and resolve,
    /// byte-equal with Python's `Bounce.resolve`/
    /// `_slot_target_still_legal`. The bounce lifecycle mirrors the
    /// existing `Effect::ReturnToHand` arm (zone→HAND, reset
    /// damage/counters/attach, mark left-bf-if-nonland, LTB fan-out
    /// via `pending_leaves_dispatches` + `queue_token_cease`).
    ReturnTargetPermanentYouControlToOwnersHand,
    /// **Round 28 coverage extension.** Add `amount` mana of a single
    /// color chosen by the controller at resolution time. Mirrors
    /// Python's `AddNManaOfChosenColor(amount: int, options: tuple[str, ...])`
    /// (`mana.py:85` — Lotus-Field-shape, 4 EoE cards: "Add three mana
    /// of any one color").
    ///
    /// **First-slice deterministic color choice.** Picks `options[0]`
    /// (first available color, default `"W"`). Mirrors Python's
    /// "first color in options" deterministic fallback when no agent
    /// is seated. Future decision-callback slice plumbs the explicit
    /// player choice through the pending_decisions queue.
    ///
    /// **Schema.** `amount: i32` and `options: Vec<String>` (default
    /// WUBRG). Empty options Vec is illegal; non-WUBRGC entries
    /// raise. amount<0 raises at parse + apply. amount=0 is a legal
    /// no-op (no color minted).
    AddNManaOfChosenColor { amount: i32, options: Vec<String> },
    /// **Round 28 coverage extension.** Search the source-controller's
    /// library for a card matching `filter_name`, reveal it, put it
    /// into that player's hand. Mirrors Python's
    /// `SearchLibraryForCard(filter_name, max_mv=0, may_fail=True)`
    /// (`search.py:30` — 9 EoE cards).
    ///
    /// **Filter vocabulary** (post Round-30 supertypes/subtypes
    /// substrate). Rust dispatches on the same string set Python does
    /// at `search.py:_matches` — arm-for-arm parity:
    ///   - `"basic_land"` → `card_types` contains `"Land"`
    ///     (legacy proxy; the historical reason was the missing
    ///     supertypes field, which the Round-30 substrate now carries.
    ///     Kept on the legacy proxy because every EoE basic land has
    ///     `card_types=["Land"]` with no compound non-basic land
    ///     routing this dispatch path, and changing the proxy now
    ///     would break in-flight fixtures whose CardDef omits the
    ///     supertypes hint).
    ///   - `"basic_plains"` / `"basic_island"` / `"basic_swamp"` /
    ///     `"basic_mountain"` / `"basic_forest"` → `printing.name`
    ///     equals the canonical basic name.
    ///   - `"creature_small"` (with `max_mv`) → `card_types` contains
    ///     `"Creature"` AND `mana_cost.total() <= max_mv`. `max_mv=0`
    ///     means "creatures with cost <= 0" (rarely matches; specify
    ///     a real `max_mv` value).
    ///   - `"artifact"` → `card_types` contains `"Artifact"`.
    ///   - `"legendary_creature"` → `card_types` contains `"Creature"`
    ///     AND `supertypes` contains `"Legendary"` (Round-30
    ///     substrate; powers The Seriema's tutor + Starfield Shepherd's
    ///     dual-clause inner half).
    ///   - `"spacecraft"` → `subtypes` contains `"Spacecraft"`
    ///     (Round-30 substrate).
    ///   - `"sliver"` → `subtypes` contains `"Sliver"` (Round-30
    ///     substrate).
    ///   - `"demon"` → `subtypes` contains `"Demon"` (Round-30
    ///     substrate; powers Demonic Counsel from Duskmourn).
    ///   - unknown filter → silent fizzle (matches Python's
    ///     `_matches` returning False for unknown).
    ///
    /// **Deterministic first-match pick (top-to-bottom).** Library
    /// Vec is bottom-first (lib[len-1] = top); scan via `iter().rev()`.
    /// Real MTG lets the player choose any matching card; deferred
    /// until decision callbacks exist.
    ///
    /// **No shuffle.** Same deferral as
    /// `SearchLibraryForBasicLandTapped` — CR 701.19 mandates shuffle
    /// but the Rust port hasn't externalized RNG yet (G19).
    ///
    /// **`may_fail` flag.** Python's flag controls whether the
    /// controller may choose to skip the fetch. The Rust first-slice
    /// always fetches if a match exists (deterministic choice); the
    /// flag is preserved on the variant for parity round-trip but
    /// not consulted at apply time.
    SearchLibraryForCard {
        filter_name: String,
        max_mv: i32,
        may_fail: bool,
    },
    /// **Round 28 coverage extension.** Every creature on the
    /// source's controller's OPPONENTS' battlefields gets
    /// `+power/+toughness` until end of turn. Mirror of
    /// `PumpCreaturesYouControlUntilEOT` but with the opponent-target
    /// orientation. 2 EoE cards use this shape — debuff anthem.
    ///
    /// **Two-player simplification.** In the current engine,
    /// "opponents" reduces to "the one opponent" = `1 - controller`.
    /// Iterates that player's battlefield; for every creature, push
    /// a `UeotModifier::PowerToughnessBonus`. Negative bonuses are
    /// the typical use case (controller-side curse-shape effects
    /// that shrink the enemy creatures); positive bonuses are
    /// inexpressible in real-life MTG but the variant accepts
    /// them for parity edge cases.
    ///
    /// **Same UEOT cleanup semantics** as PumpSelfUntilEOT — the
    /// modifier auto-expires at the next CR 514.3 cleanup-step pass.
    /// `(0, 0)` short-circuits; empty opponent bf fizzles silently.
    PumpCreaturesOpponentsControlUntilEOT { power_bonus: i32, toughness_bonus: i32 },
    /// **Mouth-of-the-Storm sibling — extended-duration variant.**
    /// Same shape as `PumpCreaturesOpponentsControlUntilEOT` (mass
    /// +/-power / +/-toughness on every creature controlled by the
    /// source-controller's opponents) but the printed duration is
    /// "until your next turn" rather than the bare "until end of
    /// turn". The Python sibling
    /// `PumpCreaturesOpponentsControlUntilEOT(until_caster_next_turn=
    /// True, ...)` writes the modifier through
    /// `grant_pt_pump_until_player_next_turn` (layer-7c registry
    /// entry tagged `expires_at="controller_untap"`,
    /// `expires_at_player_id` = source-controller); the modifier
    /// survives the casting turn's cleanup AND the opponent's full
    /// subsequent turn, expiring only at the start (`run_untap_step`)
    /// of the source-controller's next turn.
    ///
    /// **Real-EoE consumer:** Mouth of the Storm
    /// (EoE Creature, {6}{U} Elemental):
    ///   "When this creature enters, creatures your opponents
    ///    control get -3/-0 until your next turn."
    ///
    /// **Substrate-only first slice — cleanup-gate divergence
    /// deferred.** The current `UeotModifier` enum is implicitly
    /// `expires_at = "cleanup"` (drained wholesale by
    /// `clear_ueot_modifiers_all`/`cleanup_step_for_active_player`);
    /// adding a tagged-lifetime field cascades through the four
    /// effective-P/T / effective-keyword readers in `state.rs`,
    /// the two cleanup primitives, and a new untap-step hook.
    /// That cascade is out of this slice's budget. The first-slice
    /// resolve arm pushes the SAME `UeotModifier::PowerToughnessBonus`
    /// the bare sibling pushes — the post-resolve effective-P/T
    /// observable matches the bare sibling exactly. The divergence
    /// (Python keeps the debuff into the opponent's turn; Rust
    /// drops it at the casting-turn cleanup) is documented in
    /// `docs/SEARCH_LIMITATIONS.md` and is a follow-up Rust slice
    /// (the parametric duration field on `UeotModifier`).
    ///
    /// The discriminator-only sibling matters NOW because:
    ///   1. The bridge can route Mouth of the Storm cleanly
    ///      (extended-duration variant -> Python's
    ///      `until_caster_next_turn=True`).
    ///   2. Once the cleanup-gate slice lands, only THIS variant's
    ///      resolve arm changes — every call site, every parse arm,
    ///      every test pin already targets the right discriminator.
    ///
    /// `(0, 0)` short-circuits; empty opponent bf fizzles silently
    /// (mirrors the bare sibling).
    PumpCreaturesOpponentsControlUntilCasterNextTurn {
        power_bonus: i32,
        toughness_bonus: i32,
    },
    /// **Round 28 coverage extension.** Put a target card from a
    /// graveyard on the bottom of its owner's library. Mirrors
    /// Python's `PutCardFromGraveyardOnBottomOfLibrary`
    /// (`graveyard.py:420` — Chrome-Companion-shape, 2 EoE cards).
    ///
    /// **First-slice deterministic target choice.** Scans graveyards
    /// in player-order (p0 first, then p1). Takes the FIRST card
    /// encountered (`graveyard[0]` if non-empty). Real MTG lets the
    /// activating player choose any card in any graveyard; deferred
    /// until decision callbacks exist.
    ///
    /// **Zone-change reset (CR 400.7).** tapped, summoning_sick,
    /// damage, counters reset on move. summoning_sick=true per r28
    /// CE1 universal default (library zone, gameplay-irrelevant).
    ///
    /// **"Bottom of library" = `lib[len-1]`** post-CE3 (lib[0] = top,
    /// tail = bottom). Append via `lib.push(...)`.
    ///
    /// Empty-graveyards-on-both-sides = silent fizzle.
    PutCardFromGraveyardOnBottomOfLibrary,
    /// **Round 28 coverage extension.** Move all +1/+1 counters from
    /// the source to the first creature on the source-controller's
    /// battlefield. Mirrors Python's
    /// `MoveSelfCountersToTargetCreatureYouControl` (`counters.py:263`
    /// — Dockworker-Drone-shape dies trigger, 2 EoE cards).
    ///
    /// **Mechanism.** Locate source's `counters` scalar; if zero
    /// or source isn't found, silent no-op. Locate first creature on
    /// source-controller's battlefield (zone-iteration order, gates
    /// on `card_types contains "Creature"`). Transfer: subtract from
    /// source's counters (= 0 after transfer), add same amount to
    /// target's counters. Source-not-on-bf path silent fizzle.
    ///
    /// **Semantics — true "move" not "copy".** Source's counters are
    /// cleared on transfer. Python's resolve doesn't subtract from
    /// source because by trigger-time the source is typically in
    /// graveyard (counters are gone with it). The Rust port models
    /// the cleaner "move" semantics: if source still exists with
    /// counters, transfer + clear. Documented divergence for dies-
    /// trigger interactions.
    ///
    /// **First-slice deterministic target choice.** First creature
    /// on source-controller's bf. Real MTG lets the player choose;
    /// deferred until decision callbacks.
    MoveSelfCountersToTargetCreatureYouControl,
    /// **Round 28 coverage extension.** Deal damage equal to the
    /// source's effective power to any target (mirrors Python's
    /// `DealDamageEqualToSourcePowerAnyTarget` — 2 EoE cards,
    /// Mishra's Bauble-style power-scaling damage spells).
    ///
    /// **Mechanism.** Locate the source on the battlefield; read its
    /// effective power (printed + UEOT bonus); deal that many damage
    /// to the first eligible target (defender's first creature, else
    /// defender-as-player — mirrors `DealDamageAnyTarget`'s target
    /// selection shape). If source isn't on battlefield (e.g. spell
    /// resolving and source is on stack), effective power is 0 → no
    /// damage dealt. Source with effective power 0 is a legal no-op.
    /// Source with negative effective power is treated as 0 damage
    /// (defensive — Python clamps at 0).
    DealDamageEqualToSourcePowerAnyTarget,
    /// **Round 28 coverage extension.** Tap target artifact or
    /// creature. Mirrors Python's `TapTargetArtifactOrCreature(Tap)`
    /// (`tap.py:324` — 2 EoE cards).
    ///
    /// **First-slice deterministic target choice.** Same shape as
    /// the existing `Effect::TapTarget { target: AnyCreature }` arm
    /// (defender's first eligible permanent), but the filter
    /// includes Artifact OR Creature instead of just Creature.
    /// Idempotent on already-tapped target (silent no-op). Empty /
    /// no-eligible defender bf → silent fizzle.
    TapTargetArtifactOrCreature,
    /// **Round 28 coverage extension.** Run `effect` only if the
    /// source's controller has at least `min_count` tapped creatures
    /// on their battlefield. Mirrors Python's
    /// `ConditionalOnTappedCreatures(effect, min_count=2)`
    /// (`combinators.py:1007` — Flight-Deck-Coordinator-shape
    /// intervening-if clause, 6 EoE cards).
    ///
    /// **Mechanism.** At resolve time, locate source controller;
    /// count tapped creatures on that player's battlefield; if
    /// `count >= min_count`, recursively call `apply_effect` on the
    /// nested effect with the same source_obj_id. Otherwise silent
    /// no-op (matches Python's "skipped" log line).
    ///
    /// **Recursion.** Box<Effect> for the nested effect — Effect is
    /// heap-allocated so the recursion is safe (same shape as
    /// Sequence's Vec<Effect>).
    ///
    /// **Schema.** `effect` required; `min_count` defaults to 2.
    /// Negative min_count is legal (always-fires) but ill-formed
    /// from a parity perspective; min_count=0 always fires.
    ConditionalOnTappedCreatures { effect: Box<Effect>, min_count: i32 },
    /// **Round 28 coverage extension.** Put a +1/+1 counter on each
    /// of up to `max_n` target creatures. Mirrors Python's
    /// `PutPlusOneCounterOnEachOfUpToNTargetCreatures(max_n=2)`
    /// (`counters.py:179` — Weftblade-Enhancer-shape, 2 EoE cards).
    ///
    /// **First-slice deterministic target choice.** Scans
    /// battlefields in player order (source controller first, then
    /// opponent); picks the first min(max_n, eligible_count)
    /// creatures encountered. Real MTG lets the player pick ANY
    /// creature (either side); deferred until decision callbacks.
    /// Empty / no-eligible-creatures fizzles silently.
    ///
    /// **Schema.** `max_n` defaults to 2. Negative max_n is illegal
    /// and raises at parse. max_n=0 is a legal no-op.
    PutPlusOneCounterOnEachOfUpToNTargetCreatures { max_n: i32 },
    /// **Round 28 coverage extension.** Put `amount` charge counters
    /// on the resolving source. Mirrors Python's
    /// `PutChargeCountersOnSelf(amount=1, amount_from_x=False)`
    /// (`counters.py:549` — Blast-Zone-shape, 2 EoE cards).
    ///
    /// **Single-counter-slot simplification.** Rust's per-object
    /// `counters` scalar doesn't distinguish +1/+1 vs charge vs
    /// loyalty (G8 multi-kind counter map is future work). For
    /// gameplay purposes today, charge counters route to the same
    /// counter slot as +1/+1. Future G8 closure will split these.
    ///
    /// **`amount_from_x` deferred.** Python's flag reads
    /// `state.resolving_x_value` (X-cost spell pattern). Rust
    /// doesn't track X values today; the flag is preserved on the
    /// variant for round-trip but `amount_from_x=true` resolves to
    /// 0 (silent no-op). Future X-cost work flips this.
    ///
    /// **Schema.** `amount` defaults to 1; negative raises at parse;
    /// `amount_from_x` defaults to false.
    PutChargeCountersOnSelf { amount: i32, amount_from_x: bool },
    /// **Round 28 coverage extension.** Create a token that's a copy
    /// of a target creature the source's controller controls. Mirrors
    /// Python's `CreateTokenCopyOfTargetCreatureYouControl`
    /// (`tokens.py:316` — Mirrorpool-shape, 2 EoE cards).
    ///
    /// **First-slice copy semantics.** New GameObject allocated with
    /// the SAME `printing_id` as the target. zone=BATTLEFIELD,
    /// controller = source-controller, summoning_sick=true (token
    /// ETBs sick), other per-object state zeroed. Future slices add
    /// the Token supertype + additional_subtype injection once Rust
    /// has supertype/subtype tracking.
    ///
    /// **First-slice deterministic target.** First creature on
    /// source-controller's bf (same shape as PumpTargetCreatureUntilEOT
    /// et al.). Empty / no eligible target = silent fizzle.
    ///
    /// **Python field drop.** Python's `additional_subtype` field is
    /// deferred (Rust has no subtypes on registry yet).
    CreateTokenCopyOfTargetCreatureYouControl,
    /// **Round 28 coverage extension.** "You may sacrifice an
    /// artifact. If you do, [inner_effect]." Mirrors Python's
    /// `OptionalSacArtifactForEffect` (`combinators.py:299` —
    /// Lithobraking-shape, 2 EoE cards).
    ///
    /// **First-slice deterministic-yes.** Python's default
    /// base-Agent behavior is "yes" (always sac if able). Rust
    /// mirrors that: if source's controller controls at least one
    /// artifact on the battlefield, sacrifice the FIRST one (move
    /// to controller's graveyard with CR 400.7 zone-change reset)
    /// AND apply the inner effect with the same source_obj_id.
    /// If no artifact is available, no-op (no inner effect either —
    /// matches Python's "if not options: return").
    ///
    /// **Recursion.** Box<Effect> for inner; recursive apply_effect.
    OptionalSacArtifactForEffect { inner_effect: Box<Effect> },
    /// **Round 28 coverage extension.** "Target artifact or
    /// creature's owner puts it on top or bottom of their library."
    /// Python class:
    /// `OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary`
    /// (`effects/generic/zone_moves.py:30`). 2 EoE cards.
    ///
    /// **First-slice deterministic-bottom.** Python's resolve calls
    /// `owner.agent.choose_mode("top", "bottom")`. Rust has no
    /// decision-callback substrate yet; we deterministically pick
    /// BOTTOM. A reasonable first-slice mirror because
    /// decision-callback agents typically default-pick the bottom
    /// option when offered top/bottom (least disruptive). Future
    /// slice with `pending_decisions` can promote this to a real
    /// choice.
    ///
    /// **Owner-as-controller simplification.** Python tracks
    /// `owner_id` separately from controller. Rust currently has
    /// no owner field on GameObject; we treat the target's
    /// `controller` as the owner. First-slice synthetic states
    /// never split owner from controller, so this is exact for the
    /// test universe.
    ///
    /// **Target selection.** First defender (= 1 - active_player)
    /// permanent whose printing's card_types contains "Artifact"
    /// OR "Creature". No eligible target = silent fizzle.
    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    /// **Round 28 coverage extension.** "Attach this Equipment to
    /// target creature you control." Mirrors Python's
    /// `AttachToTargetCreatureYouControl`
    /// (`effects/generic/attach.py`), the EoE-Equipment ETB shape
    /// (Atomic Microsizer, Illvoi Light Jammer, Squire's Lightblade,
    /// etc. — 5 EoE cards).
    ///
    /// **First-slice deterministic target.** First creature on the
    /// source-controller's battlefield. Same shape as
    /// PumpTargetCreatureUntilEOT and friends. Empty bf / no
    /// eligible creature → silent fizzle (matches Python's
    /// `if target is None: return`).
    ///
    /// **First-slice scope.** Drops Python's `granted_keywords_ueot`
    /// rider (Squire's Lightblade-style "...and target creature
    /// gains first strike UEOT") — Rust doesn't yet need the rider
    /// for the parity rounds the orchestrator is closing. Drops
    /// Python's "is Equipment" subtype gate too: Rust doesn't track
    /// subtypes on the registry yet, so the effect is gated solely
    /// by what cards print it (only Equipment cards print
    /// `AttachToTargetCreatureYouControl` in EoE).
    ///
    /// **Sets source's `attached_to`.** Looks the source up on the
    /// source-controller's battlefield (an Equipment ETB resolution
    /// happens AFTER the Equipment is on the bf — CR 603.6a), finds
    /// the first creature on that same battlefield, and stamps
    /// `source.attached_to = Some(target_id)`.
    AttachToTargetCreatureYouControl,
    /// **Round 29 coverage extension.** "Equip [cost]:" activated-
    /// ability resolution per CR 702.6. Mirrors Python's
    /// `EquipToTargetCreatureYouControl`
    /// (`effects/generic/attach.py:117`), the activated-ability
    /// sibling of `AttachToTargetCreatureYouControl` (Atomic
    /// Microsizer family — 14 EoE cards, the highest-frequency
    /// in-pool Effect class still missing from Rust).
    ///
    /// **Semantics identical to `AttachToTargetCreatureYouControl`.**
    /// Re-targets the source Equipment onto a creature the
    /// controller controls, replacing any prior attachment by
    /// overwriting `source.attached_to`. The distinct variant exists
    /// for logging/diagnostics parity with Python (`describe`
    /// returns "equip: ..." vs the ETB attach's "attach this
    /// Equipment to ..."): a tooling consumer reading
    /// `effect_to_pydict` can tell ETB-attach from a paid Equip
    /// activation. The resolve mutation itself is identical.
    ///
    /// **First-slice deterministic target.** First creature on the
    /// source-controller's battlefield that isn't the source
    /// itself. Same shape as the Attach variant.
    ///
    /// **First-slice scope.** Drops the granted-keywords rider
    /// (the Python parent class accepts one, but `EquipToTargetCreatureYouControl`
    /// passes the default empty frozenset — equip activations
    /// don't carry UEOT keyword grants on EoE cards).
    EquipToTargetCreatureYouControl,
    /// **Round 28 coverage extension.** Destroy the permanent that
    /// the resolving aura is enchanting. Mirrors Python's
    /// `DestroyEnchantedPermanent` (`destroy.py:213` — Cryoshatter
    /// shape, 4 EoE cards).
    ///
    /// **Reads source's `attached_to`.** The source (typically an
    /// aura/enchantment with a triggered ability) must be on a
    /// battlefield with `attached_to = Some(target_id)`. Source not
    /// on bf, or `attached_to` is None, or the target isn't on a
    /// battlefield → silent fizzle.
    ///
    /// **Mechanism.** Find target by id across both battlefields;
    /// remove from its zone; CR 400.7 zone-change reset (tapped,
    /// summoning_sick, damage, counters, attached_to); push to its
    /// controller's graveyard.
    DestroyEnchantedPermanent,
    /// **Round 28 coverage extension.** Tap the permanent the
    /// resolving aura is attached to. Mirror of
    /// `DestroyEnchantedPermanent` (same target lookup via
    /// `attached_to`) but uses the tap mutation instead of zone
    /// change. 2 EoE cards. Idempotent on already-tapped target.
    /// Source not on bf / no host / target untracked → silent
    /// fizzle.
    TapEnchantedPermanent,
    /// **Round-30 stun-counter substrate.** Put a stun counter on
    /// up to one target tapped creature. Mirrors Python's
    /// `PutStunCounterOnTappedCreature` (`counters.py:592` — Cryogen
    /// Relic activated half, 1 EoE card).
    ///
    /// **First-slice deterministic target.** Walks both players'
    /// battlefield in ETB-timestamp order and picks the FIRST
    /// already-tapped creature it finds (regardless of controller —
    /// the Python class places no controller restriction; "up to one
    /// target tapped creature"). No eligible target (empty bf or no
    /// tapped creatures) → silent fizzle. Real MTG lets the controller
    /// pick any tapped creature; deferred until decision callbacks
    /// extend to this Effect.
    ///
    /// **Mechanism.** On the matched object, increment
    /// `stun_counters` by 1 via `Arc::make_mut` on the target's
    /// battlefield Vec. The CR 701.20 untap-step gate is wired in
    /// `state_ops_steps.rs:untap_all_for_player_inline` (decrement
    /// instead of untap when `stun_counters > 0`).
    PutStunCounterOnTappedCreature,
    /// **Round-30 stun-counter substrate.** Tap target artifact or
    /// creature an opponent controls and put a stun counter on it.
    /// Both effects apply to the SAME chosen target. Mirrors Python's
    /// `TapTargetOpponentPermanentAndStunIt` (`tap.py:217` —
    /// Mechanozoa ETB, 1 EoE card).
    ///
    /// **First-slice deterministic target.** Walks both players'
    /// battlefield in ETB-timestamp order and picks the FIRST
    /// (Artifact OR Creature) permanent NOT controlled by the
    /// source's controller. No eligible target → silent fizzle.
    ///
    /// **Mechanism.** Tap the target (idempotent on already-tapped)
    /// and increment `stun_counters` by 1. Same target for both
    /// mutations — the Python class's docstring is explicit that
    /// "both effects apply to the same chosen target".
    TapTargetOpponentPermanentAndStunIt,
    /// **Round 28 coverage extension.** Exile target creature until
    /// the source leaves the battlefield. Mirrors Python's
    /// `ExileTargetCreatureUntilSourceLeaves` (`effects/generic/exile.py:504`
    /// — All-Fates Stalker / Hardlight Containment ETB, 3 EoE cards).
    ///
    /// **First-slice deterministic target.** First creature on the
    /// defender's (= 1 - active_player) battlefield. Real MTG lets
    /// the controller pick any creature; deferred until decision
    /// callbacks exist. Drops Python's `excluded_subtypes` /
    /// `opponents_only` / `yours_only` filter knobs.
    ///
    /// **Mechanism.** Stamp target's `exiled_with_source =
    /// Some(source_id)` before moving to exile (CR 400.7 zone-change
    /// reset for all other per-object state). The link is consumed
    /// by `return_paired_exiles_for(source_id)` — a manual pymethod
    /// for the harness/Effect to call when the source leaves bf.
    /// Auto-firing the return is a follow-up slice; for now the
    /// link is stamped but the return must be triggered manually.
    ExileTargetCreatureUntilSourceLeaves,
    /// **Round 28 coverage extension.** Exile a non-land permanent
    /// an opponent controls until source leaves the battlefield.
    /// Mirrors Python's `ExileOpponentsNonlandPermanentUntilLeaves`
    /// (`effects/generic/exile.py:588` — 4 EoE cards).
    ///
    /// **First-slice deterministic target.** First permanent on the
    /// defender's bf whose card_types do NOT contain "Land".
    /// Defender = 1 - active_player (first-slice owner-as-controller
    /// shape). No eligible target → silent fizzle.
    ///
    /// **Mechanism** is the same as ExileTargetCreatureUntilSourceLeaves:
    /// stamp target's `exiled_with_source = Some(source_id)` and move
    /// to owner's exile. Manual return via
    /// `return_paired_exiles_for(source_id)`.
    ExileOpponentsNonlandPermanentUntilLeaves,
    /// **Round 14 rider-widened "until leaves" exile (additive
    /// sibling variant — Option B).** Same shape as
    /// `ExileOpponentsNonlandPermanentUntilLeaves` but admits a
    /// `max_mv` mana-value cap on the target permanent. Mirrors
    /// Python's
    /// `ExileOpponentsNonlandPermanentUntilLeaves(max_mv=N)` rider
    /// (`effects/generic/exile.py:587`) — `max_mv > 0` restricts
    /// the legal-target pool to permanents with mana value ≤
    /// `max_mv`; the bare-shape sibling above (no `max_mv`) handles
    /// the no-cap case.
    ///
    /// **EoE consumer (audit).** Seam Rip (EoE) — "When this
    /// enchantment enters, exile target nonland permanent an
    /// opponent controls with mana value 2 or less until this
    /// enchantment leaves the battlefield." Carrier Enchantment
    /// at `{W}`. The capped variant differs from the bare-shape
    /// Banishing Light / Hardlight Containment shape only in the
    /// per-permanent MV cap on the legal-target pool.
    ///
    /// **First-slice deterministic target.** First permanent on the
    /// defender's bf whose `card_types` does NOT contain "Land" AND
    /// whose printing's mana value is ≤ `max_mv`. CR 117.x: mana
    /// value reads off the printing's static mana cost (Rust:
    /// `printing.mana_cost.total()`); X-cost cards count X as 0
    /// while in exile / graveyard / library / hand, matching the
    /// printed-cost reading. No eligible permanent → silent fizzle
    /// (e.g. defender's bf is empty, all bf permanents are lands,
    /// or all non-land permanents have MV > max_mv).
    ///
    /// **Mechanism.** Identical to the bare sibling: stamp
    /// `exiled_with_source = Some(source_obj_id)` on the moved
    /// object, push to owner's exile zone, queue
    /// `pending_leaves_dispatches` for LTB fan-out. Manual return
    /// via `return_paired_exiles_for(source_id)`. CR 400.7
    /// zone-change reset (tapped, summoning_sick=true per r28 CE1,
    /// damage, counters, attached_to) preserved.
    ///
    /// **Controller-of-effect.** Mirrors the I80 fix on the bare
    /// sibling: `controller = current_resolve_source_controller`
    /// (with `active_player` fallback); `defender = 1 - controller`.
    /// On p0's turn, p1's controlled trigger exiles a P0 permanent
    /// with MV ≤ max_mv.
    ///
    /// **Schema simplification vs Python.** Carries `max_mv: i32`
    /// only. Other Python-side riders that would naturally extend
    /// here (cross-controller widening, source-leaves auto-fire) are
    /// out of scope of this variant; the schema mirror is
    /// intentionally minimal, additive widenings get separate
    /// variants per the round-12 / round-13 / this-round precedent.
    ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves { max_mv: i32 },
    /// **Round 28 may-do decision wrapper.** "You may [inner effect]."
    /// Mirrors Python's `MayDo(Effect, prompt)` family (the
    /// `effects/generic/may_do.py` umbrella — "you may sacrifice a
    /// creature; if you do, ..." cards and similar may-pay riders).
    ///
    /// **Resolve semantics.** Pushes a `PendingDecision::MayDoChoice`
    /// onto `state.pending_decisions` and short-circuits the resolve.
    /// The controller either calls `submit_may_do_decision(yes)` to
    /// resolve manually, or `auto_resolve_pending_decisions(strategy)`
    /// to resolve via a strategy. "Yes" recursively applies the inner
    /// effect (with the same `source_obj_id` so "if you do" follow-on
    /// effects retain the resolving spell's identity); "no" drops the
    /// decision and the inner effect is never applied.
    ///
    /// **Prompt.** A free-form string surfaced to the agent via the
    /// `pending_decisions()` snapshot dict. Empty by default
    /// (registrations that omit `prompt`). Not used for any game-
    /// state mutation — purely informational.
    MayDo {
        inner_effect: Box<Effect>,
        prompt: String,
    },
    /// **Round 28 "look at top N, put one in hand, rest to graveyard"
    /// wrapper.** Mirrors Python's
    /// `LookTopNPutOneInHandRestToGraveyard` (Commune-with-Evil
    /// family, 4 EoE cards). Examines the top `n` cards of the
    /// controller's library; controller picks ONE to put into hand,
    /// the rest go to graveyard.
    ///
    /// **Resolve semantics.** Pops `min(n, library_len)` cards off
    /// library top and packages them into a
    /// `PendingDecision::PickCardFromExaminedChoice`. The Effect arm
    /// short-circuits — cards are NOT yet in graveyard or hand. The
    /// controller drains via `submit_pick_card_decision(chosen_id)`
    /// (manual) or `auto_resolve_pending_decisions("default" |
    /// "pick_top")` (deterministic). The auto-resolve / no-agent
    /// default picks the topmost card (cards[0]) — matches Python's
    /// fallback in `LookTopNPutOneInHandRestToGraveyard.resolve`.
    ///
    /// **Empty library short-circuit.** `n=0` or library_len=0 emits
    /// a PendingDecision with an empty `cards` Vec. Drains as an O(1)
    /// no-op via auto-resolve. Mirrors Python's `if not examined:
    /// return` short-circuit.
    LookTopNPutOneInHandRestToGraveyard { n: i32 },
    /// **BLOCKED-RUST-EFFECT slice (round 54) — Famished Worldsire ETB.**
    /// "When this creature enters, look at the top X cards of your
    /// library, where X is this creature's power. Put any number of
    /// land cards from among them onto the battlefield tapped, then
    /// shuffle." Mirrors Python's `FamishedWorldsireETBEffect`
    /// (`effects/bespoke.py:94`) — the ONLY EoE card with this shape,
    /// 2,324 17lands decks.
    ///
    /// **Unit variant; no payload.** X is dynamic (read from the
    /// source's CURRENT effective power at resolve, post-Devour +1/+1
    /// counters), and the "put any number" land selection is a
    /// DETERMINISTIC greedy-all (the Python source-of-truth picks ALL
    /// land cards from the examined set — `bespoke.py:121` — a
    /// documented "any number" → greedy-land simplification that is
    /// strictly ramp-positive; both engines mirror it so NO
    /// PendingDecision is plumbed and there is no §2
    /// silent-mistranslation surface from the choice). Extra dict
    /// fields silently ignored for round-trip dict-shape consistency
    /// (mirrors every other unit-variant parse arm — round-49
    /// `gain_life_equal_to_your_life_total` precedent).
    ///
    /// **Resolve semantics (mirrors `FamishedWorldsireETBEffect`).**
    /// 1. Source-controller resolution via the transient
    ///    `current_resolve_source_controller` (fallback
    ///    `active_player`) — CR 119.5 / 608.2; "your library" is the
    ///    source's controller's library, same shape as
    ///    `SearchLibraryForBasicLandTapped` (round-23/I69).
    /// 2. `X = effective_power(source)` = base (UEOT-SetBasePT
    ///    override / manland) + `ueot_power_bonus()` + signed
    ///    `obj.counters` — the SAME composition Python's
    ///    `_compute_effective_pt` folds in (round-53
    ///    `read_doubling_power` §2 precedent). `X <= 0` → silent
    ///    no-op (matches Python's `if x <= 0: return`; the oracle's
    ///    look-at-top-0 is a degenerate no-op — Famished Worldsire is
    ///    never X<=0 in practice because Devour seeds +1/+1 counters,
    ///    but the clamp mirrors Python for parity on edge inputs).
    /// 3. Examine the top `min(X, library_len)` cards (top = lib[0]
    ///    post-CE3). Remove every LAND card (printing
    ///    `card_types` contains "Land" — Python's `printing.is_land`)
    ///    from the examined window, in top-to-bottom order; put each
    ///    onto the source-controller's battlefield TAPPED, not
    ///    summoning-sick (CR 302.1 is creature-only; a land entering
    ///    via this effect has no summoning sickness regardless),
    ///    controller-stamped to the source's controller. Non-land
    ///    examined cards stay in the library in place (Python's
    ///    `rest` list is never removed — `bespoke.py:122`).
    /// 4. Fan out `Event::Enters` (landfall) for each entered land —
    ///    queued in `pending_etb_dispatches`, drained by the caller
    ///    after the resolving stack item pops, same as
    ///    `SearchLibraryForBasicLandTapped`. Mirrors Python's
    ///    `fire_landfall_triggers(state, land)` per-land call.
    /// 5. Shuffle the WHOLE remaining library (CR 701.19 / 701.22):
    ///    external `state.rng_ref()` Fisher-Yates (byte-identical to
    ///    Python's `random.Random.shuffle` via
    ///    `external_rng::next_below`) when a Python rng is installed,
    ///    xorshift64 fallback otherwise — the exact dual-path block
    ///    `SearchLibraryForBasicLandTapped` uses. Mirrors Python's
    ///    `shuffle_library(state, controller_id)`.
    FamishedWorldsireLookTopXByPowerPutLandsTapped,
    /// **16th Rust slice (TTT Pin 3 closure) — Astelli Reclaimer ETB.**
    /// "When this creature enters, return target noncreature, nonland
    /// permanent card with mana value X or less from your graveyard
    /// to the battlefield, where X is the amount of mana spent to
    /// cast this creature." Mirrors Python's
    /// `AstelliReclaimerETBEffect` (`effects/bespoke.py:25-89`) — the
    /// ONLY EoE card with this shape, 4,565 17lands decks (the largest
    /// single-card closure in the BLOCKED-genuinely_missing tier).
    ///
    /// **Unit variant; no payload.** X is dynamic (read from the
    /// source's printing's `mana_cost.total()` for a normal cast or
    /// `warp_cost.total()` for a warp cast — the `cast_via_warp` flag
    /// rides on the entered permanent's `warp_exile_pending`, set by
    /// `state_ops_cast.rs::cast_creature_warp` and surviving the
    /// move-to-battlefield post-resolve). The target selection is a
    /// DETERMINISTIC first-legal scan over the controller's graveyard
    /// (sorted by graveyard insertion order; ties — none expected —
    /// would break by `id`). Both engines mirror the
    /// first-legal-deterministic pick (Python's `legal_targets`
    /// returns the graveyard scan in insertion order, and the resolve
    /// pops the first via the requires_target target binding).
    ///
    /// **Resolve semantics (mirrors `AstelliReclaimerETBEffect`).**
    /// 1. Source-controller resolution via the transient
    ///    `current_resolve_source_controller` (fallback
    ///    `active_player`) — CR 119.5 / 608.2; "your graveyard" is
    ///    the source's controller's graveyard.
    /// 2. Locate the source's entered-battlefield object by
    ///    `source_obj_id` on the controller's battlefield. If absent
    ///    (the source has already left bf — e.g. counterspelled
    ///    pre-resolve), silent no-op.
    /// 3. Read `cast_via_warp = obj.warp_exile_pending`. Look up the
    ///    source's printing by `obj.printing_id`. If `cast_via_warp`
    ///    AND the printing's `warp_cost` is `Some(cost)`, set
    ///    `x = cost_total(cost)`. Otherwise set
    ///    `x = cost_total(&printing.mana_cost)`.
    /// 4. Walk the controller's graveyard in insertion order. Pick
    ///    the FIRST card whose printing satisfies
    ///    `is_permanent && !is_creature && !is_land && mana_cost.total() <= x`
    ///    (`is_permanent` =
    ///    `card_types intersects {Creature, Artifact, Land,
    ///    Enchantment, Planeswalker}`; the filter rejects Creature
    ///    and Land so the survivors are
    ///    `{Artifact, Enchantment, Planeswalker}` permanent cards).
    ///    If no card matches, silent no-op (Python parity).
    /// 5. Remove the picked card from the graveyard. Stamp the
    ///    standard battlefield-entry fields (
    ///    `zone = ZONE_BATTLEFIELD`, `tapped = false`,
    ///    `summoning_sick = false` per CR 302.1 + 612.6 — the picked
    ///    card is NOT a creature; `controller = controller_u8`,
    ///    `damage = 0`, `counters = 0`, `attached_to = None`,
    ///    `exiled_with_source = None`, fresh
    ///    `etb_timestamp`). Push to controller's battlefield.
    /// 6. Queue `Event::Enters` for the entered card via
    ///    `pending_etb_dispatches.push(entered_id)` so the trigger
    ///    fires AFTER the resolving Astelli Reclaimer ETB trigger
    ///    frame pops (mirrors `SearchLibraryForBasicLandTapped` /
    ///    `FamishedWorldsireLookTopXByPowerPutLandsTapped`).
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// the target is computed at resolve time from the deterministic
    /// graveyard scan, not stored on the variant. Python's
    /// `AstelliReclaimerETBEffect` is a `@dataclass(frozen=True)`
    /// with no fields. The bridge translates this Rust unit variant
    /// to `AstelliReclaimerETBEffect()`. Mirrors the round-49 /
    /// round-54 unit-variant convention.
    AstelliReclaimerETBEffect,
    /// **18th Rust slice (XXX Pin 2 closure) — Roving Actuator Void ETB.**
    /// "Exile up to one target instant or sorcery card with mana value 2
    /// or less from your graveyard. Copy it. You may cast the copy
    /// without paying its mana cost." Mirrors Python's
    /// `RovingActuatorETBEffect` (`effects/bespoke.py:630-718`) — the
    /// ONLY EoE card with this shape, 3,400 17lands decks.
    ///
    /// **Single-field variant.** `max_mv: u8` — the printed mana-value
    /// threshold (2 on Roving Actuator; surfacing the field as a payload
    /// lets future "exile-from-gy MV<=N + cast-copy-for-free" siblings
    /// reuse the variant if they ever appear). Mirrors Python's
    /// `max_mv: int = 2` field on the `@dataclass(frozen=True)`.
    ///
    /// **Resolve semantics (mirrors `RovingActuatorETBEffect.resolve`).**
    /// 1. Source-controller resolution via the transient
    ///    `current_resolve_source_controller` (fallback
    ///    `active_player`) — CR 119.5 / 608.2; "your graveyard" is the
    ///    source's controller's graveyard.
    /// 2. Walk the controller's graveyard in insertion order. Pick the
    ///    FIRST card whose printing satisfies
    ///    `(is_instant || is_sorcery) && mana_cost.total() <= max_mv`
    ///    (mirrors Python's `legal_targets` filter; matches the bespoke
    ///    effect's "graveyard_spell" target-kind set). If no card
    ///    matches, silent no-op (Python parity — "up to one" is fine
    ///    with 0 picks).
    /// 3. Remove the picked card from the graveyard. Stamp
    ///    `zone = ZONE_EXILE` (mirrors Python's `card.zone = Zone.EXILE`)
    ///    and push to controller's exile (the "you exile" rider on the
    ///    Void body). Clone the printing_id BEFORE the move so it
    ///    survives the in-place mutation.
    /// 4. Apply the exiled card's `spell_effect` (a copy of it — the
    ///    Python convention is "the second `spell_effect` resolve fires
    ///    on the engine directly without going through the stack"; the
    ///    copy does NOT go on the stack and can NOT be countered, an
    ///    intentional first-slice simplification). The exiled card's
    ///    `printing.effects` is the single-element list Python's bridge
    ///    builds from `card.spell_effect` (see
    ///    `cross_engine.py:10207-10208`). Iterate the effects and call
    ///    `apply_effect(state, effect, exiled_card_id)` recursively
    ///    with the EXILED card's object id as the source — that's the
    ///    "casting source" of the copy, so any source-controller-aware
    ///    Effect arm (DealDamage / DrawCards / GainLife / ...) reads
    ///    the right controller via the lookup-by-id zone scan + falls
    ///    back to `current_resolve_source_controller` if the zone scan
    ///    misses.
    /// 5. The exiled card lives in exile (we just put it there) so the
    ///    source-controller lookup in recursive `apply_effect` arms
    ///    finds it there. The recursive call MUST happen AFTER the
    ///    push-to-exile so the zone scan succeeds — otherwise arms that
    ///    can't find the source fall through to `active_player` which
    ///    can be wrong.
    ///
    /// **No PendingDecision.** Python's `requires_target() == True` and
    /// `legal_targets()` is non-trivial, but the target binding is a
    /// first-legal pick (mirrors the Astelli Reclaimer slice 16
    /// closure's deterministic-graveyard-pick convention). The copy's
    /// own targets (if its `spell_effect` is targeted, e.g. Plasma Bolt
    /// dealing 2 damage) bind via the recursive arm's standard
    /// first-legal pick (e.g. DealDamage to "the defender = opponent of
    /// the source's controller"). Net effect: parity with Python's
    /// `_PickFirstAgent`-driven test fixture path.
    ///
    /// **Schema parity with Python.** Single-field variant on both
    /// sides — `max_mv: u8` on the Rust enum / `max_mv: int` on the
    /// Python dataclass. The bridge translates this Rust variant to
    /// `RovingActuatorETBEffect(max_mv=max_mv)`.
    RovingActuatorETBEffect { max_mv: u8 },
    /// **19th Rust slice (honest-stop substrate landing) — Meltstrider's
    /// Resolve Aura ETB "fight".** "When this Aura enters, enchanted
    /// creature fights up to one target creature an opponent controls."
    /// Mirrors Python's `EnchantedCreatureFightsTargetEffect`
    /// (`effects/bespoke.py:506-582`) — the ONLY EoE card with this
    /// shape, 15,259 17lands decks (the #1 marginal-value closure
    /// target in `docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md`).
    ///
    /// **HONEST-STOP CLOSURE — substrate landed, full byte-equality
    /// deferred.** The Effect variant + parse + to_pydict + bridge
    /// arms ship here; the resolve arm fizzles silently when the
    /// resolving Aura source has no `attached_to` (which is the
    /// Rust-side reality TODAY — see `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`,
    /// the verify-first re-walk that audited the entire Aura family
    /// as the **largest §2-declined Rust multi-round substrate**:
    /// CardDef has no `is_aura`, the loader admits no Aura, Rust's
    /// spell resolution has no Aura attach path, and `attached_to`
    /// is hard-reset on every Rust zone change). The Python side ALSO
    /// fizzles in the no-`attached_to` case (`bespoke.py:546-548`:
    /// `if aura is None or aura.attached_to is None: return`), so the
    /// no-host branch IS byte-equal across engines. The genuine
    /// full-game byte-equality for Meltstrider's Resolve still requires
    /// the broader Aura attach substrate (Phases 1-2 of the substrate
    /// plan: CardDef aura representation + Rust cast-time enchant-
    /// target decision + Rust Aura-resolution attach + layered grants +
    /// CR-704.5n SBA + loader admit predicate). The shipped resolve
    /// arm is the Phase-3 stub that Phases 1-2 unblock — once Auras
    /// actually set `attached_to` in Rust, this arm's hot-path branch
    /// (the both-fighters-present case) becomes live and exercises
    /// the simultaneous-damage logic already shipped.
    ///
    /// **Why ship the substrate now (not pure-defer).** The variant +
    /// parse + to_pydict + bridge round-trip lets `cards.rs`'s
    /// previously-pre-marked "deferred Aura-shape Fight follow-up"
    /// doc-comment slot (Effect::FightTargetCreatureWithTargetCreature
    /// at cards.rs:1713) move from "deferred follow-up" → "shipped
    /// stub awaiting attach substrate". Future slices that land
    /// Aura attach (the long pole — the agent-driven cast-target
    /// decision substrate shared with Monoist / Fight-Aura per
    /// docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md §3 phase 1) flip the
    /// hot-path live without touching this variant — it is forward-
    /// compatible scaffolding rather than throwaway code. This is the
    /// CE5-style "ship the substrate + an honest-stop sentinel; document
    /// follow-up" pattern (the campaign discipline for §2-declined
    /// substrates that the inventory rated as "small (1-2 cycles)"
    /// but the verify-first re-walk reveals as multi-round).
    ///
    /// **Unit variant on both sides.** Python's
    /// `EnchantedCreatureFightsTargetEffect` is a `@dataclass(frozen=
    /// True)` with no fields; the Rust enum mirrors this. Source-side
    /// fighter is read from `state.resolving_source.attached_to` at
    /// resolve time (the Aura's enchanted creature, NOT a player-picked
    /// friendly creature — the Aura's `attached_to` decides). Opponent-
    /// side fighter is the optional ("up to one") target.
    ///
    /// **Resolve semantics (mirrors `bespoke.py:541-575`).**
    /// 1. Locate the source object (`source_obj_id`) — scan every zone
    ///    (Aura may still be on bf as the ETB trigger fires; mirrors
    ///    `Fight`'s source-controller resolution).
    /// 2. Read `source.attached_to`. If `None` (every Rust Aura today
    ///    because Rust does not set `attached_to` on Aura resolve —
    ///    see `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §2) → silent
    ///    fizzle. **THIS IS THE HONEST-STOP BRANCH** until the
    ///    attach substrate lands; Python takes the same fizzle when
    ///    `attached_to is None` so the cross-engine state stays
    ///    byte-equal in the no-host case.
    /// 3. Resolve the enchanted creature on its bf by id. If not found
    ///    (already left the bf) → silent fizzle (Python parity, mirrors
    ///    CR 701.12c "if either fighter has left the battlefield, the
    ///    fight fizzles").
    /// 4. Pick the opponent-side target — FIRST creature on the
    ///    OPPONENT's bf (deterministic first-legal scan; mirrors
    ///    `FightTargetCreatureWithTargetCreature`'s opponent scan).
    ///    The Aura's controller is the source's controller (CR 119.5);
    ///    "an opponent" is `1 - controller` on the 2-player engine.
    ///    No opponent creature → silent fizzle ("up to one" = legal
    ///    0-pick + the effect resolves with no fight).
    /// 5. Compute BOTH effective powers BEFORE applying any damage
    ///    (CR 701.12 simultaneous damage event). Apply each fighter's
    ///    power as damage to the other via `apply_damage_marker`.
    ///    Dispatch DEALT_DAMAGE events for the receiving creatures —
    ///    mirrors the `FightTargetCreatureWithTargetCreature` arm.
    ///
    /// **Source-substitution helper for Aura `attached_to`.** The new
    /// `resolve_source_for_aura_attached` private helper at
    /// `state_ops_resolve.rs` reads `state.resolving_source.attached_to`
    /// when the source is an Aura; falls back to the source itself for
    /// non-Aura sources. This is the substrate primitive that future
    /// Aura-shape Effect arms (Pain for All's
    /// `PainForAllETBEffect` + `PainForAllDamageEchoEffect` — listed
    /// under `attached_to` source-substitution in the BLOCKED_GENUINELY_
    /// MISSING_INVENTORY substrate-sharing pattern §) reuse.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `EnchantedCreatureFightsTargetEffect()` (field-free construction).
    EnchantedCreatureFightsTarget,
    /// **Pain for All closure (Aura Phase 3 / S4 triggered shape) —
    /// PainForAllETBEffect.** "When this Aura enters, enchanted
    /// creature deals damage equal to its power to any other target."
    /// Mirrors Python's `PainForAllETBEffect`
    /// (`effects/bespoke.py:352-428`) — the ONLY EoE card with this
    /// shape (Pain for All; 4,367 17lands decks, the
    /// `blocked_genuinely_missing` residue Aura per
    /// `docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md` §4 and
    /// `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §1.5 residue list).
    ///
    /// **HONEST-STOP CLOSURE — substrate landed, full byte-equality
    /// deferred symmetrically with slice-19's
    /// `EnchantedCreatureFightsTarget`.** The Effect variant + parse
    /// + to_pydict + bridge arms ship; the resolve arm fizzles
    /// silently when the resolving Aura source has no `attached_to`,
    /// which IS the Rust-side reality today (no Aura-on-resolve
    /// attach path; `attached_to` hard-reset on every Rust zone-change
    /// path — see `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §2). Python
    /// takes the same `if aura is None or aura.attached_to is None:
    /// return` fizzle (`bespoke.py:389-391`), so the no-host branch
    /// is byte-equal across engines.
    ///
    /// **Unit variant on both sides.** Python's `PainForAllETBEffect`
    /// is a `@dataclass(frozen=True)` with no fields. The Rust enum
    /// mirrors that. Damage source is the Aura's enchanted creature
    /// (read from `state.resolving_source.attached_to`); damage
    /// amount is that creature's effective power; the controller picks
    /// "any other target" (creature or player, excluding the
    /// enchanted creature) at resolve time. With Rust's deterministic
    /// first-legal target pick + no attached_to → silent fizzle =
    /// byte-equal.
    ///
    /// **Resolve semantics (mirrors `bespoke.py:383-422`).**
    /// 1. Locate the source object (`source_obj_id`) — scan every zone.
    ///    Source untrackable → silent no-op.
    /// 2. Read `source.attached_to`. If `None` → silent fizzle
    ///    (HONEST-STOP branch; byte-equal with Python).
    /// 3. Resolve the enchanted creature on its controller's bf. Not
    ///    found → silent no-op.
    /// 4. Compute `effective_power(enchanted)`. If ≤ 0 → silent no-op
    ///    (Python parity: `if amount <= 0: return`).
    /// 5. Pick the deterministic first-legal target — FIRST creature
    ///    on the OPPONENT's bf, OR the opponent player when no
    ///    opposing creature exists. Mirror's slice-19's first-legal
    ///    convention.
    /// 6. Apply the damage:
    ///    - player target → subtract `amount` from `victim.life`
    ///      (Python parity: `victim.life -= amount`).
    ///    - creature target → `apply_damage_marker(target, amount)`
    ///      + `damaged_this_turn = true` + dispatch DEALT_DAMAGE.
    ///
    /// **Damage-feedback loop-guard.** The companion echo effect
    /// (`PainForAllDamageEchoEffect`, below) applies its echo damage
    /// as LIFE LOSS (not via `apply_damage_marker`), so DEALT_DAMAGE
    /// is NOT re-dispatched. This mirrors Python's simplification
    /// (`bespoke.py:447-457` — `p.life -= amount`, not the damage
    /// pipeline) and inherently breaks the potential trigger loop
    /// (the echo's life loss does not cause "is dealt damage" to fire
    /// again).
    PainForAllETBEffect,
    /// **Pain for All closure (Aura Phase 3 / S4 triggered shape) —
    /// PainForAllDamageEchoEffect.** "Whenever enchanted creature is
    /// dealt damage, it deals that much damage to each opponent."
    /// Mirrors Python's `PainForAllDamageEchoEffect`
    /// (`effects/bespoke.py:432-463`).
    ///
    /// **Unit variant on both sides.** No fields. Reads
    /// `state.resolving_combat_damage_amount` (already plumbed
    /// engine-wide — every receiver-side damage event writes it,
    /// `state_ops_combat.rs::apply_damage_pass`) and the resolving
    /// effect's controller. For each non-controller, non-lost player,
    /// subtract `amount` from `life` (life-loss, NOT damage pipeline
    /// — this is the loop-guard described above and Python's
    /// `bespoke.py:447-457` shape verbatim).
    ///
    /// **Trigger predicate.** The Python parser wires the trigger as
    /// `Event.DEALT_DAMAGE + IsHost(subject=None)` — the Aura's
    /// trigger fires when the **enchanted creature** (the Aura's
    /// `attached_to` host) receives damage. The matching Rust
    /// `Predicate::IsHost` variant added in this slice carries the
    /// same semantics: matches iff `source.attached_to == subject_id`.
    /// With Rust's current no-`attached_to` reality the predicate
    /// returns false on every dispatch — byte-equal honest-stop with
    /// Python's same-case `IsHost.matches` returning false when
    /// `source.attached_to is None`.
    ///
    /// **Resolve semantics (mirrors `bespoke.py:443-457`).**
    /// 1. Read `state.resolving_combat_damage_amount`. If ≤ 0 →
    ///    silent no-op.
    /// 2. For each player p where `p.id != controller and !has_lost`:
    ///    `p.life -= amount` (life-loss, not damage). Mirrors Python
    ///    `bespoke.py:450-457`.
    ///
    /// **Why life-loss not damage-pipeline.** Python's simplification
    /// is explicitly called out at `bespoke.py:438-441`: "modeling
    /// 'deals damage' as life loss for simplicity in the non-creature-
    /// target case." Both engines take the same shortcut for byte-
    /// equality + the inherent loop-guard.
    PainForAllDamageEchoEffect,
    /// **Aura Phase 4 (Chorale of the Void substrate landing) —
    /// SacrificeSelfUnlessVoidActive.** End-step "sacrifice this Aura
    /// unless Void is active" — the Chorale-of-the-Void sustain. Mirrors
    /// Python's `SacrificeSelfUnlessVoidActive`
    /// (`effects/combinators.py:172-200`) — used today by Chorale of
    /// the Void (2,783 17lands decks; the largest deck-count Aura in
    /// the `blocked_subsystem_gated` tier per
    /// `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §1 sub-shape S4).
    ///
    /// **Unit variant on both sides.** Python's
    /// `SacrificeSelfUnlessVoidActive` is a `@dataclass(frozen=True)`
    /// with no fields. The Rust enum mirrors that. Reads the
    /// `is_void_active` state predicate (already plumbed engine-wide
    /// — `state.rs:2449-2455`: true iff `permanent_left_battlefield_
    /// this_turn || spell_warped_this_turn`) and `current_resolve_
    /// source_controller` to determine which player's source to
    /// sacrifice.
    ///
    /// **Resolve semantics (mirrors `combinators.py:179-197`).**
    /// 1. Locate the source object on its controller's battlefield.
    ///    Missing → silent no-op (Python parity:
    ///    `if source is None: return` at combinators.py:186-188).
    /// 2. If `is_void_active(state)` is true → silent no-op (Python
    ///    parity: `if is_void_active(state): return` at combinators.py
    ///    :189-193 — the Void-active branch just logs and returns).
    /// 3. Otherwise → `sacrifice_source_inline` to send the source
    ///    to its owner's graveyard (mirrors Python's
    ///    `_sacrifice(state, source)` at combinators.py:197).
    ///
    /// **Bridge-correctness today.** The Effect variant ships full
    /// Rust resolve semantics — this is NOT an honest-stop fizzle
    /// (contrast Pain for All / Meltstrider's). Once the loader admit
    /// gate widens to admit Chorale's full trigger surface (deferred
    /// to a follow-up slice — the sibling
    /// `ReanimateTargetCreatureCardFromDefendersGraveyard` ATTACKS
    /// trigger needs targeted-graveyard substrate not yet in Rust),
    /// the end-step sustain trigger flips live byte-equal across
    /// engines for any admitted Aura that includes this Effect alone
    /// in its trigger surface.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `SacrificeSelfUnlessVoidActive()` (field-free construction).
    SacrificeSelfUnlessVoidActive,
    /// **20th Rust slice (VVV Pin 2 closure) — Command Bridge ETB
    /// "sacrifice this unless you tap an untapped permanent you
    /// control".** Mirrors Python's `SacrificeSelfUnlessTapPermanent`
    /// (`effects/combinators.py:96-168`) — the canonical example of the
    /// "land-ETB-trigger path + tap-a-permanent cost decision" subsystem
    /// gap. One direct EoE consumer: **Command Bridge** (24,522 17lands
    /// decks — the highest deck count in the BLOCKED-genuinely_missing
    /// tier per `docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md`).
    ///
    /// **Substrate already in place.** Two pieces of substrate the
    /// closure planning called out as load-bearing checks: land ETB
    /// trigger dispatch and the cost-decision pump.
    ///   1. **Land ETB dispatch — ALREADY WIRED.** `play_land` → the
    ///      shared `move_from_hand_to_battlefield`
    ///      (`rust/src/state_ops_cast.rs:618-681`) which pushes
    ///      `pending_etb_dispatches.push(entered_id)` and drains
    ///      `drain_pending_etb_dispatches`. A land entering the
    ///      battlefield fires ENTERS the same way a creature does, so
    ///      Command Bridge's ETB trigger reaches the dispatcher. No
    ///      additional wiring needed here for the lands lane.
    ///   2. **Cost-decision pump — sibling-shape precedent.** The new
    ///      `PendingDecision::TapPermanentChoice` mirrors `MayDoChoice`'s
    ///      Box<Effect>-on-decline / inner-effect-on-yes shape but with
    ///      "the cost is a tap action, NOT mana" as the gate. On yes
    ///      (controller chose an untapped permanent), tap the chosen
    ///      permanent and DROP the sac. On no (decline / no legal
    ///      candidates), sacrifice the source. Pre-resolve filter
    ///      already excludes the source itself from the candidate set.
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `SacrificeSelfUnlessTapPermanent()` (field-free construction).
    /// The source-and-controller are read from `state.resolving_source`
    /// at resolve time, mirroring the Python convention
    /// (`combinators.py:121` `source = state.resolving_source`).
    ///
    /// **Resolve semantics (mirrors `combinators.py:115-163`).**
    /// 1. Locate the source object (`source_obj_id`) on its controller's
    ///    battlefield. If missing, silent no-op (Python parity:
    ///    `if source is None: return`).
    /// 2. Build the candidate set: untapped permanents controlled by
    ///    the source's controller, EXCLUDING the source itself. The
    ///    source might be a land that entered tapped, but the source-
    ///    exclusion check is structural, not state-dependent (Python
    ///    `o.id != source.id` at `combinators.py:128`).
    /// 3. **Empty candidate set →** sacrifice the source directly (the
    ///    no-choice path; Python `if not candidates: _sacrifice(state,
    ///    source); return` at `combinators.py:130-136`).
    /// 4. **Non-empty candidate set →** push
    ///    `PendingDecision::TapPermanentChoice` onto the queue carrying
    ///    the source id, controller, and the candidate ids. The
    ///    controller drains via `submit_tap_permanent_decision(Some(id))`
    ///    (tap that permanent, keep source) or
    ///    `submit_tap_permanent_decision(None)` (decline → sacrifice).
    ///    Auto-resolve strategy `"default"` declines (mirrors Python's
    ///    no-agent-seated fallback at `combinators.py:139-141`:
    ///    `chosen = None` when `p.agent is None`).
    SacrificeSelfUnlessTapPermanent,
    /// **20th Rust slice (XXX Pin 3 closure -- honest-stop substrate
    /// landing) -- Lightstall Inquisitor each-opponent-exiles-card-
    /// from-hand-with-cast-permission.** "When this creature enters,
    /// each opponent exiles a card from their hand. They may play that
    /// card for as long as it remains exiled, and they may spend mana
    /// as though it were mana of any color to cast that spell. If they
    /// cast a spell this way, they pay {1} more for it. If they play a
    /// land this way, it enters tapped." Mirrors Python's
    /// `EachOpponentExilesCardFromHandWithCastPermission`
    /// (`effects/generic/discard.py:172-214`) -- the ONLY EoE card with
    /// this shape, 4,371 17lands decks.
    ///
    /// **CLOSURE (cast-from-exile-with-cost-hike substrate).** Slice
    /// 20 shipped the Effect variant + parse + to_pydict + bridge +
    /// resolve arm (each-opponent-exile portion). This follow-up slice
    /// closes the persistent permission substrate end-to-end:
    ///   1. `GameObject` carries a new `cast_from_exile_with_lightstall
    ///      _surcharge: bool` field (`state.rs`), stamped `true` on the
    ///      exiled card by the resolve arm below.
    ///   2. `state_ops_legal.rs` surfaces a `CastSpell` (non-land) or
    ///      `PlayLand` (land) action for every exile object with this
    ///      flag, at sorcery timing, scoped to the card's owner (CR
    ///      400.7 reset put `controller = owner` at exile time).
    ///   3. `state_ops_cast.rs::cast_creature` / `cast_spell_noncreature`
    ///      detect the exile origin, pay `printing.mana_cost + {1}`
    ///      generic surcharge (Python `actions.py:1494-1499`), and
    ///      clear the flag at cast time on the stack-resident object
    ///      (Python `actions.py:1505`). `play_land` honors the tapped-
    ///      entry rider: a land played via this permission stamps
    ///      `tapped = true` regardless of the printing's static
    ///      `enters_tapped` (Python `actions.py:1083-1094`) and clears
    ///      the flag at play time.
    /// Lightstall Inquisitor's residue closure moves from
    /// "PARTIAL CLOSED" to "FULL CLOSED" in
    /// `docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md` §3 with this
    /// slice. The 6+-cycle substrate-scope estimate was conservative;
    /// the actual closure was a single-slice landing (~250 LOC across
    /// 4 Rust files + bridge update + tests).
    ///
    /// **Cross-engine parity contract.** A probe that drives ETB
    /// resolution and inspects post-resolve zones (count of cards in
    /// each opponent's hand decreased by 1; count of cards in each
    /// opponent's exile increased by 1; chosen-card identity is the
    /// deterministic hand[0]-or-equivalent first-legal pick) is
    /// byte-equal across engines. A probe that further attempts to
    /// cast the exiled card on the OWNER's next priority NOW also
    /// round-trips: both engines surface the cast at sorcery timing,
    /// apply the +{1} surcharge, and (for lands) stamp tapped-entry.
    ///
    /// **Resolve semantics (mirrors `EachOpponentExilesCardFromHand
    /// WithCastPermission.resolve` step-for-step on the exile portion).**
    /// 1. Source-controller resolution: 5-zone walk (hand / library /
    ///    battlefield / graveyard / exile) on both players plus a stack
    ///    scan to recover the source's `controller`. Mirrors
    ///    `EachOpponentDiscards` (the same source-finder dispatch shape
    ///    -- punisher / "each opponent" effects fire on whoever
    ///    controls the source). Source not found anywhere = silent
    ///    fizzle (lost-track-of-source tolerance).
    /// 2. 2-player "each opponent" reduces to "the one opponent".
    ///    `opponent_idx = (1 - controller) as usize`.
    /// 3. If opponent's hand is empty: silent no-op (Python parity --
    ///    `if not p.hand: continue` on Python side; mirrors
    ///    `EachOpponentDiscards`'s empty-hand short-circuit).
    /// 4. Pop hand[0] of the opponent (deterministic first-legal pick;
    ///    Python's `agent.choose_discard` on the discarding opponent is
    ///    matched in probe by an agent that returns
    ///    `hand_ids[0]`).
    /// 5. CR 400.7 zone-change reset on the moved object: `tapped`,
    ///    `summoning_sick`, `damage`, `counters`, `attached_to`,
    ///    `exiled_with_source` all cleared; `zone = ZONE_EXILE`.
    /// 6. Push the object to the opponent's `exile` vec.
    ///
    /// **No event dispatch.** Unlike `EachOpponentDiscards`, this
    /// effect does NOT dispatch `Event::DiscardsCard` -- the card is
    /// exiled, not discarded (CR 701.8 vs CR 701.13). Python's resolve
    /// does NOT fire `fire_opponent_discards_triggers` either
    /// (`discard.py:172-214` has no event hook; cf. `discard.py:163`
    /// which does, for `EachOpponentDiscards`). Honest-attribution
    /// boundary respected.
    ///
    /// **Schema parity with Python.** Unit variant on both sides -- no
    /// fields. The bridge translates this Rust variant to
    /// `EachOpponentExilesCardFromHandWithCastPermission()` (field-
    /// free construction). Mirrors the round-49 / slice-16
    /// `AstelliReclaimerETBEffect` / slice-19
    /// `EnchantedCreatureFightsTarget` unit-variant convention.
    EachOpponentExilesCardFromHandWithCastPermission,
    /// **21st Rust slice (XXX Pin 1 closure) -- House Cartographer
    /// reveal-until-land library walk.** "Reveal cards from the top of
    /// your library until you reveal a land card. Put that card into
    /// your hand and the rest on the bottom of your library in a random
    /// order." Mirrors Python's `HouseCartographerSurvivalEffect`
    /// (`effects/bespoke.py:2667-2733`) -- the ONLY EoE card with this
    /// shape (the 5th and final EoE Survivor-family probe carrier).
    ///
    /// **Unit variant on both sides.** Python's
    /// `HouseCartographerSurvivalEffect` is a `@dataclass(frozen=True)`
    /// with no fields; the Rust enum mirrors this. The library-walk
    /// state (which cards have been revealed; the land-or-none sentinel)
    /// is entirely transient -- both engines compute it fresh on every
    /// resolve and emit no PendingDecision.
    ///
    /// **Resolve semantics (mirrors `bespoke.py:2690-2726`).**
    /// 1. Source-controller resolution: prefer the transient
    ///    `current_resolve_source_controller` (set by the stack resolver
    ///    around each apply_effect call), fall back to `active_player`.
    ///    "Your library" = the source's controller's library
    ///    (CR 119.5 / 608.2). Mirrors the slice-16 / slice-18 /
    ///    slice-20 source-controller-resolution shape.
    /// 2. Walk the controller's library from the top (index 0):
    ///    pop the top card; if its printing's `card_types` contains
    ///    "Land", break out with the popped card as the revealed land;
    ///    otherwise push the popped card onto the local `revealed`
    ///    Vec and keep walking. Empty library terminates with no
    ///    land found (mirrors Python's `while p.library:` loop).
    /// 3. If a land was revealed: stamp its `zone = ZONE_HAND` and
    ///    push to the controller's `hand` (mirrors `bespoke.py:2706-
    ///    2707`'s `land_card.zone = Zone.HAND; p.hand.append(...)`).
    ///    Apply the CR 400.7 zone-change reset on the moved object
    ///    (`tapped`, `summoning_sick`, `damage`, `counters`,
    ///    `attached_to`, `exiled_with_source` all cleared) so the
    ///    move is byte-equal with Python's library→hand transition.
    /// 4. If no land was found: the `revealed` Vec already holds
    ///    every popped card (every revealed non-land); no hand move.
    /// 5. Shuffle the local `revealed` Vec using the SAME byte-equal
    ///    RNG primitive every other in-Effect shuffle site uses
    ///    (external `random.Random.shuffle` via
    ///    `external_rng::next_below` when an external rng is
    ///    installed; inline xorshift64 fallback otherwise).
    ///    Mirrors Python's `state.rng.shuffle(revealed)` at
    ///    `bespoke.py:2718` -- the "random order" tail of CR
    ///    701.18a.
    /// 6. Stamp each shuffled card's `zone = ZONE_LIBRARY` and push
    ///    in order onto the BOTTOM of the controller's library
    ///    (= `lib.push(card)` — Python uses `p.library.append(card)`
    ///    where index 0 is the top, so append IS the bottom).
    ///    Mirrors `bespoke.py:2719-2722`.
    ///
    /// **Why no PendingDecision.** The Python effect emits no
    /// `choose_*` callback -- the library walk is fully deterministic
    /// (top-to-bottom pop with a `printing.is_land` terminator), and
    /// the random-order tail is owned by `state.rng`, not by the
    /// controller's agent. No cross-engine choice surface to bridge.
    ///
    /// **Why state.rng.shuffle is the right primitive.** Python's
    /// `state.rng.shuffle(revealed)` is `random.Random.shuffle(list)`
    /// applied in-place; the byte-equal Rust mirror is the Fisher-
    /// Yates loop that consumes `external_rng::next_below(rng, i+1)`
    /// for each i in `(1..n).rev()` -- exactly what
    /// `fisher_yates_shuffle_optional` and every other in-Effect
    /// shuffle site (Famished Worldsire / SearchLibraryForBasicLand
    /// Tapped / search_library_for_first_of_type_and_shuffle) does.
    /// When NO external rng is installed, the deterministic xorshift64
    /// fallback (seeded with a per-resolve tuple) keeps the Rust path
    /// deterministic within a fixed-seed simulation; the Python side
    /// in that branch uses the engine's `state.rng` (a seedable
    /// `random.Random`), which IS deterministic too. Full byte-
    /// equality on the random-order tail requires the EXTERNAL rng
    /// path (the test harness installs the same `random.Random` on
    /// both engines via `set_external_rng`); the xorshift64 fallback
    /// is for the Rust-alone case where no external rng is plumbed.
    ///
    /// **Empty-library / no-non-lands corner cases.**
    /// - Empty library at entry: silent no-op (no land, no revealed).
    /// - Library starts with a land: revealed is empty, the land
    ///   goes straight to hand, no shuffle needed.
    /// - No land in the whole library: every card moves to revealed,
    ///   the controller's library ends up empty before the shuffle
    ///   + append; the shuffled revealed list is appended -- net
    ///   effect: the whole library is shuffled. Mirrors Python's
    ///   `bespoke.py:2712-2722` "found no land" branch + the
    ///   shuffle-and-append tail (which acts as a whole-library
    ///   shuffle when revealed == the full library).
    ///
    /// **Schema parity with Python.** Unit variant on both sides --
    /// no fields. The bridge translates this Rust variant to
    /// `HouseCartographerSurvivalEffect()` (field-free construction).
    /// Mirrors the slice-16 `AstelliReclaimerETBEffect` / slice-19
    /// `EnchantedCreatureFightsTarget` / slice-20
    /// `EachOpponentExilesCardFromHandWithCastPermission` unit-
    /// variant convention.
    HouseCartographerSurvivalEffect,
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// Sothera, the Supervoid's "each opponent exiles a creature they
    /// control" with optional source-link bookkeeping.** Mirrors
    /// Python's `EachOpponentExilesOwnCreature(link_to_source: bool)`
    /// (`effects/generic/opponent_punishers.py:148-217`) — the
    /// per-opponent independent-choice primitive used by Sothera, the
    /// Supervoid's DIES trigger ("Whenever a creature you control dies,
    /// each opponent exiles a creature they control. Note the cards
    /// exiled this way.").
    ///
    /// **HONEST-STOP CLOSURE — substrate landed, full byte-equality
    /// deferred (slice-19 Meltstrider precedent).** The Effect variant
    /// + parse + to_pydict + bridge arms ship here; the resolve arm
    /// fizzles silently when the resolving source is missing OR when
    /// `link_to_source = true` (the source-link bookkeeping requires a
    /// new persistent `GameObject.exiled_by_source` field on every Rust
    /// game object PLUS a `_release_exiled_until_leaves` on-leave hook
    /// PLUS the one-shot `state.suppress_exiled_by_source_on_leave_for`
    /// flag — large multi-cycle substrate per the
    /// `BLOCKED_GENUINELY_MISSING_INVENTORY.md` Sothera entry). The
    /// scaffolding closes the Effect-level cross-engine round-trip and
    /// makes the hot-path branch ready-to-flip when the exile-source-
    /// link substrate ships.
    ///
    /// **Schema parity with Python.** Single-field variant on both
    /// sides — `link_to_source: bool` mirrors Python's
    /// `link_to_source: bool = False` dataclass field. Bridge translates
    /// this Rust variant to `EachOpponentExilesOwnCreature(link_to_
    /// source=link_to_source)`.
    EachOpponentExilesOwnCreature { link_to_source: bool },
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// "Put one creature card exiled with this permanent onto the
    /// battlefield under your control with N additional +1/+1 counters
    /// on it".** Mirrors Python's
    /// `ReanimateOneExiledWithSourceWithCounters(counter_amount: int)`
    /// (`effects/generic/exile.py:1121-1216`) — the recall-one half of
    /// Sothera, the Supervoid's end-step ability (only consumer in the
    /// EoE pool today).
    ///
    /// **HONEST-STOP CLOSURE — substrate landed, full byte-equality
    /// deferred (slice-19 Meltstrider precedent).** The resolve arm
    /// fizzles silently because no Rust GameObject carries an
    /// `exiled_by_source` source-link today — the candidate set is
    /// always empty (mirrors Python's same-case `if not candidates:
    /// return` fizzle at `exile.py:1178-1179`). Closing requires the
    /// new persistent source-link field; documented in the SacrificeSelfPreservingExileLink
    /// substrate notes and `BLOCKED_GENUINELY_MISSING_INVENTORY.md`.
    ///
    /// **Schema parity with Python.** Single-field variant on both
    /// sides — `counter_amount: i32` mirrors Python's
    /// `counter_amount: int = 0` dataclass field.
    ReanimateOneExiledWithSourceWithCounters { counter_amount: i32 },
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// "Sacrifice this permanent (preserving the source-link on any
    /// cards still in exile)".** Mirrors Python's
    /// `SacrificeSelfPreservingExileLink`
    /// (`effects/generic/exile.py:1221-1273`) — the sac-second half
    /// of Sothera, the Supervoid's end-step ability.
    ///
    /// **HONEST-STOP CLOSURE — substrate landed, full byte-equality
    /// deferred (slice-19 Meltstrider precedent).** The resolve arm
    /// sacrifices the source via the existing `sacrifice` path. The
    /// one-shot `suppress_exiled_by_source_on_leave_for` arming is a
    /// no-op in Rust today because the on-leave auto-return hook
    /// itself isn't wired — both sides of the suppression machinery
    /// (the auto-return AND the suppression) are deferred together,
    /// so cross-engine state stays byte-equal on the sac event.
    ///
    /// **Schema parity with Python.** Unit variant on both sides —
    /// no fields.
    SacrificeSelfPreservingExileLink,
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// Requiem Monolith's UEOT grant of a quoted dealt-damage trigger.**
    /// Mirrors Python's `GrantTriggeredAbilityUEOT(ability=...)`
    /// (`effects/generic/grant_keywords.py:189-251`) — currently only
    /// EoE consumer is Requiem Monolith, which grants a fixed quoted
    /// trigger ("Whenever this creature is dealt damage, you draw that
    /// many cards and lose that much life.") UEOT to a chosen creature.
    ///
    /// **HONEST-STOP CLOSURE — Effect-level scaffolding only.** Rust
    /// has no general layer-6 grant-triggered-ability registry
    /// (Python's `continuous_effects` + `game.effective_triggered_
    /// abilities`); only the narrow Perigee `GrantedDiesReturnTapped`
    /// marker exists (`state.rs:299`). Rather than building the full
    /// nested-trigger-grant subsystem in this slice, the resolve arm
    /// records nothing — the cross-engine round-trip closes for this
    /// Effect class and the post-resolve hot-path branch becomes
    /// ready-to-flip when the broader grant-triggered-ability registry
    /// lands.
    ///
    /// **Schema parity with Python.** Unit variant on the Rust side —
    /// the Python `ability` field (a `TriggeredAbility` dataclass) is
    /// elided in Rust today because the substrate to *use* it isn't
    /// wired. The bridge consumes the dict's `ability` payload and
    /// reconstructs the Python `GrantTriggeredAbilityUEOT(ability=
    /// TriggeredAbility(...))` from the inner dict; the Rust resolve
    /// is the silent no-op described above. This is the slice-19-style
    /// Effect-level honest-stop pattern.
    GrantTriggeredAbilityUEOTRequiem,
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// "That creature's controller may have <source> deal N damage to
    /// it".** Mirrors Python's
    /// `MayHaveSourceDealDamageToTarget(amount: int)`
    /// (`effects/generic/damage.py:1114-1186`) — Requiem Monolith's
    /// optional-ping body.
    ///
    /// **HONEST-STOP CLOSURE — Effect-level scaffolding only.** The
    /// resolve arm fizzles silently because the target's controller's
    /// agent `choose_yes_no` hook is Python-only — Rust has no
    /// `choose_yes_no` callback wiring at this site. The Python side
    /// ALSO returns early when `target_controller.agent is None`
    /// (`damage.py:1161`), so the no-agent branch IS byte-equal
    /// cross-engine.
    ///
    /// **Schema parity with Python.** Single-field variant on both
    /// sides — `amount: i32` mirrors Python's `amount: int = 1`
    /// dataclass field.
    MayHaveSourceDealDamageToTarget { amount: i32 },
    /// **22nd Rust slice (Sothera + Requiem closures honest-stop) —
    /// "You draw that many cards and lose that much life".** Mirrors
    /// Python's `DrawCardsAndLoseLifeEqualToResolvingDamage`
    /// (`effects/generic/damage.py:1072-1111`) — the granted UEOT
    /// trigger body Requiem Monolith installs on a chosen creature.
    ///
    /// **HONEST-STOP CLOSURE — Effect-level scaffolding only.** This
    /// effect is normally fired as the BODY of a granted UEOT trigger
    /// that the `GrantTriggeredAbilityUEOTRequiem` (above) installed.
    /// Since the granted-trigger subsystem isn't wired in Rust, this
    /// arm only runs in a synthetic test that drives the Effect
    /// directly (e.g. via `apply_effect` with `resolving_combat_damage_
    /// amount` pre-set). When invoked, it reads `state.resolving_
    /// combat_damage_amount`, fizzles silently if 0 or no source, and
    /// otherwise applies `draw_card` × amount + `life -= amount` on
    /// the source's controller (mirroring `damage.py:1089-1108`).
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields.
    DrawCardsAndLoseLifeEqualToResolvingDamage,
    /// **Round 28 modal choose-one wrapper.** "Choose one — [mode_0];
    /// [mode_1]; ..." Mirrors Python's `ModalChooseOne(modes=...)`
    /// (combinators.py:1112, 13 EoE cards) in its first-slice shape:
    /// resolve-time mode picking via the decision-callback substrate.
    ///
    /// **Resolve semantics.** Pushes a
    /// `PendingDecision::ChooseModeChoice` onto the queue and short-
    /// circuits. The controller drains via
    /// `submit_choose_mode_decision(chosen_idx)` (manual) or
    /// `auto_resolve_pending_decisions("default" | "choose_mode_0")`
    /// (deterministic). The chosen mode is recursively applied via
    /// `apply_effect` with the same `source_obj_id`.
    ///
    /// **First-slice scope.** Resolve-time pick only; cast-time mode
    /// routing through `state.resolving_modes` is deferred to Cycle 1
    /// of the ModalChooseOne envelope substrate. The mode list is
    /// bounded by parse rejection — `modes.is_empty()` raises at
    /// register time so the resolve arm always has at least one mode
    /// to fall through to.
    ///
    /// **`or_both` (scour-for-scrap-slice, 2026-05-21).** When `true`,
    /// the printed shape is the "choose one or both" Charm variant —
    /// the controller may pick a 1- or 2-element subset of modes
    /// (mirrors Python's `ModalChooseOne(or_both=True)` at
    /// `combinators.py:1119`). When `false` (the back-compat default
    /// for every pre-`or_both` registered ChooseOneMode), the printed
    /// shape is plain "choose one". Carried as a parse + to_pydict
    /// field for round-trip with the Python source-of-truth; the
    /// cast-time bind (`cast_spell_noncreature`) and the resolve-time
    /// dispatch (`Effect::ChooseOneMode` arm) read it to gate the
    /// multi-pick path. Defaulted to `false` on every existing
    /// `ChooseOneMode { modes }` construction site (back-compat
    /// byte-identical for non-`or_both` modal spells).
    ChooseOneMode { modes: Vec<Effect>, or_both: bool },
    /// **Round 28 if-you-do wrapper.** "[do_action]; if you do,
    /// [then_action]." Mirrors Python's `IfYouDo(do_action,
    /// then_action)` (combinators.py:560, 7+ EoE cards) in its
    /// first-slice shape: do_action assumed implicitly success-on-
    /// yes (i.e. `IfYouDo(MayDo(X), Y)` "looter" form).
    ///
    /// **Resolve semantics.** Pushes a
    /// `PendingDecision::IfYouDoChoice` carrying both nested Effects.
    /// The controller drains via `submit_if_you_do_decision(yes)`
    /// (manual) or `auto_resolve_pending_decisions("default" |
    /// "if_you_do_yes" | "if_you_do_no")` (deterministic). On "yes"
    /// both `do_action` and `then_action` are applied in order via
    /// `apply_effect` with the original `source_obj_id`. On "no"
    /// neither is applied.
    ///
    /// **First-slice limitation.** General IfYouDo with do_actions
    /// that can signal failure (e.g. SacrificeACreature returning
    /// False on empty bf) is deferred — would require apply_effect
    /// to return `Result<bool>` so the IfYouDo arm can branch on
    /// do_action's success. The PyDict shape (`do_action` /
    /// `then_action`) is forward-compatible.
    IfYouDo {
        do_action: Box<Effect>,
        then_action: Box<Effect>,
    },
    /// Each opponent of the source's controller mills `count` cards
    /// (CR 701.13b — the per-opponent variant of `Mill`). Mirrors
    /// Python's `EachOpponentMillsCards(count: int = 1)` (3+ EoE
    /// cards target family). Same source-finder + opponent-as-target
    /// shape as `EachOpponentDiscards` / `EachOpponentLosesLife`.
    ///
    /// **Two-player simplification.** opponent_id = 1 - controller.
    /// The single opponent's library is milled `count` cards (top to
    /// graveyard); empty library stops the inner loop silently (CR
    /// 704.5b applies to *drawing*, not milling — same shape as
    /// `Effect::Mill`).
    ///
    /// **Source not found = silent fizzle** (lost-track-of-source
    /// tolerance, mirrors `EachOpponentDiscards`).
    ///
    /// **Negative `count` raises.** `count=0` is a legal no-op.
    /// Missing `count` defaults to 1 (Python parity).
    EachOpponentMillsCards { count: i32 },
    /// Move every card in each opponent's graveyard to that player's
    /// exile zone (untargeted graveyard wipe). Mirrors Python's
    /// `ExileEachOpponentsGraveyard`
    /// (`effects/generic/exile.py:260`) — the ETB-compound first half
    /// of Dauntless Scrapbot ("When this creature enters, exile each
    /// opponent's graveyard. Create a Lander token."). Structural twin
    /// of `EachOpponentMillsCards` (same source-controller finder +
    /// per-opponent loop), differing only in the inner op:
    /// graveyard → exile (the opponent's own exile zone, CR 406 /
    /// 400.7 zone-change reset) instead of library → graveyard.
    ///
    /// **Two-player simplification.** opponent_id = 1 - controller.
    /// Python iterates `state.players` skipping the controller AND
    /// any `has_lost` player; the Rust 2-player mirror exiles the
    /// single opponent's whole graveyard. (`has_lost` parity: in the
    /// 2-player engine a lost player ends the game, so the gate is a
    /// no-op in practice; the loop short-circuits an empty graveyard
    /// silently either way.)
    ///
    /// **Source not found = silent fizzle** (lost-track-of-source
    /// tolerance, mirrors `EachOpponentMillsCards` /
    /// `EachOpponentDiscards`).
    ///
    /// Unit variant — no parameters (Python class is field-free).
    ExileEachOpponentsGraveyard,
    /// **Round 28 UEOT coverage extension (broader pump filter).**
    /// Target artifact OR creature gets `+power/+toughness` until end
    /// of turn AND optionally gains a list of keywords UEOT. Mirrors
    /// Python's `PumpTargetArtifactOrCreatureUntilEOT` (3+ EoE cards
    /// target family — Skitterclaw-Augur-shape "boost your robot or
    /// your dude" instants — PLUS Reroute Systems mode 0's "gains
    /// indestructible UEOT" granted-keyword shape).
    ///
    /// **Target semantics.** Same first-slice deterministic pattern
    /// as `PumpTargetCreatureUntilEOT` but with a broader filter:
    /// pick the FIRST permanent on the source's controller's
    /// battlefield whose `card_types` contains "Artifact" OR
    /// "Creature". No legal target → silent fizzle.
    ///
    /// **Mechanism.** Same UEOT-modifier push as
    /// `PumpTargetCreatureUntilEOT` (modifier auto-expires on the
    /// next cleanup step via the r27 CR 514.3 UEOT-clear pass).
    /// `(0, 0, [])` short-circuits as a no-op. Negative bonuses are
    /// legal (debuff-shaped). Each granted keyword is pushed as a
    /// `UeotModifier::GrantKeyword` entry (mirrors the
    /// `GrantKeywordsToTargetCreatureUntilEOT` per-keyword fanout).
    ///
    /// **Reroute Systems closure (2026-05-21).** Mode 0 of Reroute
    /// Systems ("Target artifact or creature gains indestructible
    /// until end of turn") is the canonical 0/0/{"indestructible"}
    /// shape — the granted-keywords field is what lets the single
    /// variant cover both the bare-pump form AND the keyword-grant
    /// form Python's unified `Pump` base already handled. Field
    /// defaults to empty for back-compat with every existing parser
    /// site that omits it.
    PumpTargetArtifactOrCreatureUntilEOT {
        power_bonus: i32,
        toughness_bonus: i32,
        granted_keywords: Vec<String>,
    },
    /// **Reroute Systems closure (2026-05-21) — "Deals N damage to
    /// target tapped creature".** Mirrors Python's
    /// `DealDamageToTargetTappedCreature(amount: int)` (constructor
    /// sugar on `DealDamage` with filter `IsCreature() & IsTapped()`;
    /// `effects/generic/damage.py:932-950`). Sibling of
    /// `DealDamage*` family with an at-resolve tapped-recheck per CR
    /// 608.2b — a target that untaps between cast and resolve fails
    /// the recheck and the damage is skipped.
    ///
    /// **Target semantics.** Pick the FIRST tapped creature on
    /// EITHER battlefield in `bf_iter_etb_sorted` order (NOT
    /// controller-restricted — the printed oracle on Reroute Systems
    /// mode 1 says "target tapped creature" with no controller
    /// filter). No legal target → silent fizzle (no damage applied,
    /// no log line — mirrors the unified `DealDamage.resolve` filter
    /// fizzle at `damage.py:78-79`).
    ///
    /// **At-resolve filter recheck.** The cast-time bind locks a
    /// specific target id; the resolve arm re-verifies the target
    /// is STILL a tapped creature on the battlefield before applying
    /// damage. A target that left the bf or untapped between cast
    /// and resolve is silently skipped — same CR 608.2b semantics
    /// the unified Python `DealDamage.resolve` enforces via
    /// `_slot_target_still_legal`.
    ///
    /// **Reroute Systems consumer.** Mode 1 of Reroute Systems
    /// ("Reroute Systems deals 2 damage to target tapped creature")
    /// is the only EoE consumer today; `amount = 2` for that mode.
    /// Forward-compatible with any future "deals N damage to target
    /// tapped creature" printing (the `amount` field is bare i32).
    DealDamageToTargetTappedCreature { amount: i32 },
    /// **Round 28 each-opponent sacrifice.** Each opponent of the
    /// source's controller sacrifices a creature (CR 701.16). Mirrors
    /// Python's `EachOpponentSacrificesCreature` (3+ EoE cards target
    /// family — Cruel-Edict-shape symmetric sacrifice effects).
    ///
    /// **2-player simplification.** opponent_id = 1 - controller. The
    /// single opponent's FIRST creature on battlefield (zone-iteration
    /// order, mirrors PumpTargetCreatureUntilEOT target-pick) goes to
    /// the opponent's graveyard with CR 400.7 zone-change reset.
    ///
    /// **No eligible creature on opponent's bf** → silent fizzle
    /// (mirrors PumpTargetCreatureUntilEOT no-target fizzle).
    ///
    /// **Source not found anywhere** → silent fizzle (mirrors
    /// EachOpponentDiscards / EachOpponentMillsCards).
    EachOpponentSacrificesCreature,
    /// **Round 28 grant-keyword-UEOT family.** Target creature gains a
    /// keyword until end of turn. Mirrors Python's
    /// `GrantUnblockableUEOT` (3+ EoE cards) and forward-compatibly
    /// covers any future `GrantHasteUEOT` / `GrantFlyingUEOT` /
    /// `GrantTrampleUEOT` cards via the same dispatch — just a
    /// different `keyword` string.
    ///
    /// **Mechanism.** Push a `UeotModifier::GrantKeyword { keyword }`
    /// onto the target's `ueot_modifiers` Vec. The modifier auto-
    /// expires at the next cleanup step via the r27 CR 514.3 UEOT-
    /// clear pass. The existing `has_keyword_via_ueot(keyword)`
    /// accessor on `GameObject` surfaces the granted keyword for
    /// combat / blocking checks.
    ///
    /// **Target semantics — first-slice deterministic.** Pick the
    /// FIRST creature on source-controller's battlefield (zone-
    /// iteration order, mirrors PumpTargetCreatureUntilEOT). No
    /// eligible creature → silent fizzle. Python's
    /// `GrantUnblockableUEOT` carries `attacking_only` /
    /// `exclude_self` / `power_at_most` flags; this Rust first slice
    /// does NOT carry any of them (the no-flag form is by far the
    /// most common in real cards — "target creature is unblockable
    /// this turn").
    ///
    /// **Keyword convention.** Lowercase, matches
    /// `has_keyword_via_ueot` lookup. Common values: `"unblockable"`
    /// (Python's "can't be blocked"), `"haste"`, `"flying"`,
    /// `"trample"`, etc. Empty string rejected at parse time.
    GrantKeywordToTargetCreatureUntilEOT { keyword: String },
    /// **Round-15 (post-r14) rider-widened sibling of
    /// `GrantKeywordToTargetCreatureUntilEOT`.** Same shape, but admits
    /// a LIST of keywords plus an `exclude_source` flag for the
    /// "another target creature" variant. Mirrors Python's
    /// `PumpTargetCreatureUntilEOT(power_bonus=0, toughness_bonus=0,
    /// granted_keywords=frozenset({...}), you_control_only=True,
    /// exclude_self=<bool>)` (the keyword-grant-only sub-shape with
    /// no P/T pump) — the bridge constructs the bare-creature-target
    /// Python class with `granted_keywords` populated and the
    /// `exclude_self` flag forwarded.
    ///
    /// **EoE consumer.** Blooming Stinger (1G Plant Scorpion, EoE) is
    /// the unique EoE source for the `exclude_source=true` shape:
    /// "When this creature enters, another target creature you control
    /// gains deathtouch until end of turn." Single-keyword grant
    /// (`["deathtouch"]`) on another target creature you control;
    /// `exclude_source=true` to exclude the resolving Stinger itself
    /// from the legal-target pool. Forward-compatible with future
    /// multi-keyword grants (e.g. "another target creature gains
    /// flying and trample UEOT") via the `Vec<String>` field.
    ///
    /// **Mechanism.** Push one
    /// `UeotModifier::GrantKeyword { keyword }` entry per keyword in
    /// the list onto the target's `ueot_modifiers` Vec. All entries
    /// auto-expire at the next cleanup step via the r27 CR 514.3
    /// UEOT-clear pass. The existing `has_keyword_via_ueot(keyword)`
    /// accessor on `GameObject` surfaces each granted keyword for
    /// combat / blocking checks.
    ///
    /// **Target semantics — first-slice deterministic.** Pick the
    /// FIRST creature on source-controller's battlefield (zone-
    /// iteration order, mirrors `PumpTargetCreatureUntilEOT` /
    /// `GrantKeywordToTargetCreatureUntilEOT`); when
    /// `exclude_source=true`, skip the resolving source itself.
    /// No eligible creature → silent fizzle.
    ///
    /// **Empty `keywords` list.** A pure no-op — no modifiers pushed,
    /// resolution still proceeds (mirrors Python's `if not
    /// self.granted_keywords: <skip grant body>` short-circuit). Parser
    /// arm rejects empty list to keep the dict shape honest.
    ///
    /// **Schema simplification vs Python.** Python's
    /// `PumpTargetCreatureUntilEOT` carries many other riders
    /// (`attacking_only`, `legendary_only`, `power_at_most`, etc.);
    /// this Rust slice carries `granted_keywords` + `exclude_source`
    /// only. Future filter narrowings re-enter as separate Rust
    /// variants per the round-12 / round-13 / round-14 additive-
    /// variant precedent.
    GrantKeywordsToTargetCreatureUntilEOT {
        keywords: Vec<String>,
        exclude_source: bool,
    },
    /// **Round 28 "untap another target permanent".** Mirrors Python's
    /// `UntapTargetOtherPermanent` (Nanoform-Sentinel trigger, 3+ EoE
    /// cards). Distinct from the existing `UntapTarget { target }` arm
    /// — adds an "other than source" filter.
    ///
    /// **Target semantics — first-slice deterministic.** Scan both
    /// players' battlefields in order (p0 first, then p1); pick the
    /// FIRST tapped permanent whose id != source_obj_id. Untap it.
    /// Mirrors Python's `[o for o in state.battlefield if o.id !=
    /// source_id and o.tapped]` filter + pick-first.
    ///
    /// **No eligible permanent** (everything untapped, or only the
    /// source itself is tapped) → silent fizzle.
    UntapTargetOtherPermanent,
    /// **Round 28 "tap another target creature".** Mirrors Python's
    /// `TapAnotherTargetCreature` (Starport-Security activated-ability
    /// family, 2+ EoE cards). Distinct from the existing
    /// `TapTarget { target: AnyCreature }` arm — adds an "other than
    /// source" filter so the resolving source can't tap itself.
    ///
    /// **Target semantics — first-slice deterministic.** Scan both
    /// players' battlefields in zone-iteration order (p0 first); pick
    /// the FIRST creature whose id != source_obj_id. Tap it. Already-
    /// tapped target is a silent no-op (matches the existing
    /// `TapTarget` contract).
    ///
    /// **No eligible creature** → silent fizzle.
    TapAnotherTargetCreature,
    /// **Round 28 mass reanimation.** Mirrors Python's
    /// `ReturnAllCreaturesFromYourGraveyardToBattlefield` (Awakening-
    /// Hall-style "return all creature cards from your graveyard to
    /// the battlefield", 2+ EoE cards). Distinct from the existing
    /// single-target `ReturnCreatureFromYourGraveyardToBattlefield`
    /// arm — returns EVERY creature in the source-controller's
    /// graveyard, not just the first.
    ///
    /// **Resolve semantics.** Iterate source-controller's graveyard in
    /// the stored order (oldest first — Vec.iter from the front);
    /// every object whose printing.card_types contains "Creature" is
    /// moved to the controller's battlefield with the standard ETB
    /// state reset (summoning_sick=true, tapped=false, damage=0,
    /// counters=0, attached_to=None, exiled_with_source=None). Non-
    /// creature graveyard objects stay in graveyard.
    ///
    /// **Empty graveyard / no creatures** → silent no-op (no state
    /// mutation).
    ReturnAllCreaturesFromYourGraveyardToBattlefield,
    /// **Round 28 base-P/T UEOT override (CR 613.4 layer 7b).** Target
    /// creature has base power and toughness `power/toughness` until
    /// end of turn. Mirrors Python's `SetBasePTUntilEOT(power,
    /// toughness)` (Genemorph-Imago-shape "has base P/T N/N this
    /// turn", 2+ EoE cards).
    ///
    /// **Mechanism.** Push a `UeotModifier::SetBasePT` onto the
    /// target's `ueot_modifiers` Vec. The composer
    /// (`battlefield_creature_power` / `_toughness`) reads the LAST
    /// `SetBasePT` and uses its values as the base instead of the
    /// printing's printed P/T. Counters + layer-7c additive bonuses
    /// (`PowerToughnessBonus`) still stack on top (CR 613.7c).
    ///
    /// **Target semantics.** Same first-slice deterministic pattern
    /// as `PumpTargetCreatureUntilEOT`: pick the FIRST creature on
    /// source-controller's battlefield. No eligible creature → silent
    /// fizzle.
    ///
    /// **Negative values are legal.** "Has base 0/0" is meaningful
    /// (the creature dies to SBA toughness <= 0). No guard.
    SetBasePTUntilEOT { power: i32, toughness: i32 },
    /// **Round 28 manifest-dread first-slice.** CR 701.55 — "Look at
    /// the top two cards of your library, manifest one of them, then
    /// put the other into your graveyard." Mirrors Python's
    /// `ManifestDread(count: int = 1)` (Bashful-Beastie family, 10
    /// EoE cards — the highest-frequency unported class).
    ///
    /// **Resolve semantics (count = 1).** Pop `min(2, library_len)`
    /// cards off library top and package them into a
    /// `PendingDecision::ManifestDreadChoice`. The Effect arm short-
    /// circuits — cards are NOT yet on battlefield or in graveyard.
    /// The controller drains via
    /// `submit_manifest_dread_decision(chosen_id)` (manual) or
    /// `auto_resolve_pending_decisions("default" | "manifest_top")`
    /// (deterministic — picks the topmost = cards[0]).
    ///
    /// **Empty / 1-card library partials.** library_len=0 emits a
    /// PendingDecision with an empty `cards` Vec. Drains as an O(1)
    /// no-op. library_len=1 emits a PendingDecision with cards = [top
    /// card]; on submit, the lone card is manifested and there's no
    /// graveyard-bound card. Mirrors Python's
    /// `if not p.library: return` and `if len(examined) == 1:`
    /// partials.
    ///
    /// **Face-down semantics — first-slice simplification.** The
    /// "manifested" card is moved to the controller's battlefield
    /// AS-IS — its original printing remains visible (name, types,
    /// P/T, abilities). Real MTG would set the permanent to a 2/2
    /// vanilla face-down creature with no abilities (CR 701.34 +
    /// 707.2), with the original printing tracked underneath for the
    /// face-up turn (CR 707.4). Modeling face-down state requires
    /// either a face-down flag on GameObject (forces fingerprint
    /// baseline rebuilding) or a face-down-placeholder-printing
    /// design — both deferred. See `docs/SEARCH_LIMITATIONS.md` for
    /// the full scope notice.
    ///
    /// **First-slice scope.** `count = 1` ONLY. Python supports
    /// `count = N` ("manifest dread N times") for Slimy Aquarium /
    /// Weight Room rider effects, and `count = None` (X-times) for
    /// Valgavoth's Onslaught. The Rust resolve arm raises on
    /// `count != 1`; the parse layer accepts any non-negative
    /// `count` so the Effect-enum round-trip is forward-compatible
    /// with a follow-up multi-iter slice.
    ///
    /// **Controller convention.** `state.active_player` — first-
    /// slice shape consistent with Surveil / LookTopN / MayDo /
    /// IfYouDo / ChooseOneMode. Real MTG plumbs the spell controller
    /// through `StackItem.controller`; deferred.
    ManifestDread { count: i32 },
    /// **Round-29 manifest-then-attach composition.** Mirrors Python's
    /// `ManifestDreadThenAttachSelf` (`effects/generic/library_lookat.py:510`
    /// — Conductive-Machete / Killer's-Mask / Cursed-Windbreaker /
    /// Dissection-Tools Duskmourn Equipment ETB family, freq 3 in
    /// the EoE-adjacent pool — included via the duskmourn pool the
    /// coverage script loads). Oracle text: *"When this Equipment
    /// enters, manifest dread, then attach this Equipment to that
    /// creature."*
    ///
    /// **Composition strategy (Option A).** Stitches the round-28
    /// `Effect::ManifestDread` substrate with the round-28 attach
    /// foundation (the `attached_to: Option<u32>` slot on
    /// `GameObject` + the `attach(aura_id, target_id)` pymethod).
    /// Adds NO new `PendingDecision` variant — instead extends
    /// `PendingDecision::ManifestDreadChoice` with a
    /// `then_attach_to_source: bool` field (`false` for the existing
    /// `Effect::ManifestDread` arm, `true` for this one). The submit
    /// dispatcher reads the flag and runs the attach as a follow-up.
    ///
    /// **Resolve.** Pop top `min(2, library_len)` cards into a
    /// `ManifestDreadChoice` with `then_attach_to_source: true`. The
    /// controller is the SOURCE'S controller (not `state.active_player`):
    /// this variant is always an Equipment ETB, so the source lives on
    /// a bf at resolve time and its controller field is authoritative.
    /// Source NOT on a bf → silent fizzle (no pending pushed). Mirrors
    /// Python's `if "Equipment" not in source.printing.subtypes: return`
    /// gate (which fires when there's no Equipment-typed source — same
    /// observable behavior as "no source on bf").
    ///
    /// **Submit + attach.** After the existing manifest-dread routing
    /// runs (chosen → bf with ETB reset, rest → graveyard), if
    /// `then_attach_to_source` is true the dispatcher locates the
    /// source on its controller's bf and stamps
    /// `source.attached_to = manifested_id`. If the source has LTB
    /// during the decision window (destroyed / returned to hand /
    /// bounced), the attach step is a silent partial — the manifest
    /// completes but the attach is skipped. Mirrors Python's
    /// `if manifested is None: return` and the implicit "Equipment
    /// gone" fall-through.
    ///
    /// **Empty / 1-card library partials.** Same as `ManifestDread`:
    /// empty library → pending with empty cards → drain as O(1) no-op
    /// (no manifest, no attach). 1-card library → manifest the lone
    /// card; if `then_attach_to_source` is true, attach to it.
    ///
    /// **First-slice scope.** `count = 1` ONLY (no Python printing of
    /// this class emits other values; the count field is reserved for
    /// schema parity with `ManifestDread`). Count != 1 raises with
    /// the same "deferred — see SEARCH_LIMITATIONS.md" message as
    /// the `Effect::ManifestDread` arm.
    ///
    /// **Schema.** Identical to `ManifestDread`: `count: i32` only,
    /// defaults to 1, parse-accepts any non-negative integer for
    /// forward-compat round-trip.
    ManifestDreadThenAttachSelf { count: i32 },
    /// **Round-29 manifest-then-counter composition (Moldering-Gym /
    /// Weight-Room family).** Mirrors Python's
    /// `ManifestDreadThenPutPlusOneCountersOnIt`
    /// (`effects/generic/library_lookat.py:451` — Duskmourn Room door
    /// triggers Slimy Aquarium (`counter_amount = 1`) and Weight Room
    /// (`counter_amount = 3`)). Oracle text: *"manifest dread, then
    /// put a +1/+1 counter on that creature"* / *"manifest dread,
    /// then put three +1/+1 counters on that creature"*.
    ///
    /// **Composition strategy (Option A — precedent from
    /// `ManifestDreadThenAttachSelf`).** Stitches the round-28
    /// `Effect::ManifestDread` substrate with the existing +1/+1
    /// counter scalar (`GameObject.counters: i32`). Adds NO new
    /// `PendingDecision` variant — instead extends
    /// `PendingDecision::ManifestDreadChoice` with a
    /// `then_plus_one_counter_on_manifested: bool` flag and a
    /// `counter_amount: i32` rider. The submit dispatcher reads the
    /// flag and stamps `obj.counters += counter_amount` on the
    /// manifested creature as a follow-up.
    ///
    /// **Resolve.** Pop top `min(2, library_len)` cards into a
    /// `ManifestDreadChoice` with `then_plus_one_counter_on_manifested
    /// = true` and `counter_amount = self.counter_amount`. Controller
    /// convention matches plain `Effect::ManifestDread`:
    /// `state.active_player` — these are Room-door triggers cast
    /// during the controller's turn, so `active_player` is the
    /// resolving controller. (Differs from `ManifestDreadThenAttachSelf`,
    /// which has to read the source's bf controller because it's an
    /// Equipment ETB; this variant doesn't need the source on a bf at
    /// all — the counter target is the just-manifested face-down, not
    /// the source.)
    ///
    /// **Submit + counter add.** After the existing manifest-dread
    /// routing completes (chosen → bf with ETB reset, rest →
    /// graveyard), if `then_plus_one_counter_on_manifested` is true
    /// AND a card was actually manifested (cards Vec was non-empty
    /// and a chosen card landed on bf), locate the manifested
    /// permanent by id and stamp `obj.counters += counter_amount`.
    /// If the cards Vec was empty (library was exhausted at resolve),
    /// no manifest happened and the counter step is naturally a no-op
    /// — mirrors Python's `if not new_ids: return` early-exit. (The
    /// manifested-id is captured locally inside the submit helper
    /// rather than searched globally, so an LTB-during-decision-window
    /// has nothing to defend against here: the helper just-routed the
    /// chosen card to bf moments earlier, so it's reachable.)
    ///
    /// **Empty / 1-card library partials.** Same as `ManifestDread`:
    /// empty library → pending with empty cards → drain as O(1) no-op
    /// (no manifest, no counter add — matches Python's "no new_ids →
    /// return" path). 1-card library → manifest the lone card and
    /// stamp the counter.
    ///
    /// **First-slice scope.** `count = 1` ONLY at the manifest-dread
    /// iteration level (inherits the `ManifestDread` gate — count > 1
    /// is the multi-iter shape; no Python printing of this class
    /// emits count != 1). `counter_amount` accepts arbitrary
    /// non-negative integers — Slimy Aquarium emits 1, Weight Room
    /// emits 3, future printings may pick other values.
    ///
    /// **Schema.** `count: i32` (manifest-dread iteration count,
    /// defaults to 1) and `counter_amount: i32` (defaults to 1).
    /// Both parse-accept any non-negative integer for forward-compat
    /// round-trip; non-negative gate fires at parse.
    ManifestDreadThenPutPlusOneCountersOnIt { count: i32, counter_amount: i32 },
    /// **Round-29 coverage extension.** Intervening-if effect keyed
    /// on the EoE "Void" condition. At resolve time, if Void is
    /// active (a nonland permanent left the battlefield this turn
    /// OR a spell was warped this turn), apply `then_effect`;
    /// otherwise apply `else_effect` (or no-op if `else_effect` is
    /// None). Mirrors Python's `ConditionalIfVoid(then_effect,
    /// else_effect=None)` (`combinators.py:1409` — Decode-
    /// Transmissions-shape "Void —" rider, 4 EoE cards).
    ///
    /// **Predicate.** Reads `state.permanent_left_battlefield_this_turn`
    /// or `state.spell_warped_this_turn`. Both substrate flags
    /// default to false and have no Rust call sites that set them
    /// yet — until the LTB-tracking / warp-tracking call sites land,
    /// the predicate always evaluates to false and the resolver
    /// always falls through to `else_effect`. This is a deliberate
    /// substrate stub: Python's `is_void_active(state)` reads the
    /// same two fields, and Rust will pick up Void-true behavior
    /// for free once those flags get wired by a future slice.
    ///
    /// **Recursion.** Both branches are `Box<Effect>` so nested
    /// effects (Sequence, other Conditionals) work. `else_effect`
    /// is `Option<Box<Effect>>` to mirror Python's optional second
    /// branch.
    ///
    /// **Schema.** `then_effect` required; `else_effect` optional
    /// (omitted dict key = None on the Rust side).
    ConditionalIfVoid {
        then_effect: Box<Effect>,
        else_effect: Option<Box<Effect>>,
    },
    /// Deal `amount` damage to target creature an opponent controls.
    /// Mirrors Python's `DealDamageToOpponentCreature(DealDamage)`
    /// sugar in `effects/generic/damage.py` — same `DealDamage` arm
    /// as `DealDamageToTargetCreature`, with the target restricted to
    /// `IsCreature() & ControlledByOpponent()` (the source's
    /// caster cannot pick their own creature). 2 EoE cards (Glassworks
    /// // Shattered Yard family).
    ///
    /// **Distinction from `Effect::DealDamage { target: AnyCreature }`** —
    /// that arm picks the FIRST creature on the defender's battlefield
    /// (= `1 - state.active_player`'s bf[0]) without a card-type filter.
    /// `DealDamageToOpponentCreature` filters the same battlefield to
    /// CREATURES ONLY (skipping lands / artifacts / non-creature
    /// permanents) AND scopes the search to OPPONENTS OF THE SOURCE'S
    /// CONTROLLER (rather than the defender of the active player). In
    /// 2-player land, these collapse to the same player when the
    /// source's controller is the active player, but a source whose
    /// controller is `state.players[1]` while `active_player == 1`
    /// (e.g. an instant cast on the opponent's turn) inverts the
    /// target direction — this slice gets that right via the
    /// 5-zone source-controller walk shape (mirrors `LoseLife` /
    /// `TargetOpponentLosesLife` / `EachOpponentDiscards`).
    ///
    /// **First-slice deterministic target choice.** Scan the source-
    /// controller's opponents in player order; for each opponent,
    /// scan their battlefield top-to-bottom and pick the FIRST
    /// `Creature` printing. In 2-player land that's exactly the
    /// opponent's battlefield[0]-that-is-a-creature. Real MTG lets
    /// the controller choose; deferred to the same decision-callback
    /// follow-up as the rest of the `DealDamage` family.
    ///
    /// **Source itself is excluded** — the predicate
    /// `ControlledByOpponent` filters by the printing's `controller`
    /// field, and the source's controller (= caster) is never an
    /// opponent of itself. A source on the opponent's battlefield
    /// is therefore eligible (its `controller` is the opponent),
    /// matching the Python filter's `controller != resolving_controller`
    /// shape. No explicit `id != source_obj_id` guard is needed in
    /// the loop because the predicate already excludes the
    /// resolving-controller side of the battlefield.
    ///
    /// **Negative amount raises** at register time AND at resolve
    /// time (defense-in-depth — mirrors round-3 `DealDamage` and
    /// `DealDamageAnyTarget` shape).
    ///
    /// **No eligible creature** (opponent's bf empty, or holds only
    /// non-creatures) → silent fizzle. Mirrors Python's
    /// `if not legal_targets: return` short-circuit and the
    /// `DealDamage { target: AnyCreature }` fizzle behavior.
    ///
    /// **SBA fires post-resolve.** `resolve_top_of_stack` calls
    /// `auto_resolve_pending` after the effect resolves and SBAs run
    /// via the dispatcher's standard pass; lethal damage (damage >=
    /// effective toughness) flips the creature to graveyard the same
    /// way the existing damage paths do (CR 704.5g).
    DealDamageToOpponentCreature { amount: i32 },
    /// **Round-28 look-at-top-N-reveal-or-counter wrapper.** Mirrors
    /// Python's `LookAtTopNRevealOrCounter(n, filter_name, max_power,
    /// counter_on_miss)` (Pulsar-Squadron-Ace + Living-Phone family,
    /// freq 2 in the EoE pool). Examines the top `n` cards of the
    /// CONTROLLER's library; controller MAY reveal one matching the
    /// filter and put it into hand; the rest go to the bottom of the
    /// library in a "random" order. If no card was revealed (because
    /// none matched, the agent declined, or the library was too small)
    /// AND `counter_on_miss` is true, a +1/+1 counter lands on the
    /// resolving source if it's currently on the battlefield.
    ///
    /// **Resolve semantics.** Pops `min(n, library_len)` cards off
    /// library top and packages them into a
    /// `PendingDecision::LookAtTopNRevealOrCounterChoice`. The Effect
    /// arm short-circuits — examined cards are NOT yet routed. The
    /// controller drains via `submit_look_at_top_n_decision(reveal,
    /// chosen_id)` (manual) or `auto_resolve_pending_decisions("default"
    /// | "reveal_no")` (deterministic, no-reveal — matches Python's
    /// no-agent-seated fallback because the Python class only reveals
    /// when an agent is seated AND the agent picks something).
    ///
    /// **Filter / max_power.** `filter_name` is one of `"spacecraft"`,
    /// `"creature"`, `"artifact"`; `max_power` (default 0 = no cap)
    /// gates the `"creature"` filter — Living Phone uses
    /// `filter_name="creature"`, `max_power=2`. Other `filter_name`
    /// values are rejected at parse. The reveal-side dispatcher
    /// validates the picked card matches the filter (it must be in
    /// the examined set AND match `_matches` semantics); a mismatch
    /// raises `ValueError` transactionally.
    ///
    /// **Bottom-the-rest is "random order".** First-slice
    /// simplification: we route the remainder to the BOTTOM of the
    /// library in popped order (cards[0] — the topmost popped — goes
    /// to the bottom first, then cards[1], etc.). Real MTG asks for a
    /// random order; we use a deterministic popped-order. Documented
    /// in SEARCH_LIMITATIONS as deferred — re-routes via the engine's
    /// rng once the rng-externalization work lands.
    ///
    /// **Empty library short-circuit.** `n=0` or library_len=0 emits a
    /// PendingDecision with an empty `cards` Vec. Drains as an O(1)
    /// no-op via auto-resolve (also triggers the counter rider IF
    /// `counter_on_miss=true` and the source is on bf — matches
    /// Python: no reveal happened, so the miss rider fires).
    ///
    /// **Controller convention.** `state.active_player` — first-slice
    /// shape consistent with Surveil / LookTopN / MayDo / IfYouDo /
    /// ChooseOneMode / ManifestDread.
    LookAtTopNRevealOrCounter {
        n: i32,
        filter_name: String,
        max_power: i32,
        counter_on_miss: bool,
    },
    /// **Round 28 manland animation (CR 613.4 + 613.6 + 613.7b).**
    /// Source land becomes a `power/toughness` creature until end of
    /// turn, optionally gaining a subtype (e.g., "Blinkmoth",
    /// "Elemental"). Mirrors Python's `BecomeLandCreatureUEOT(power,
    /// toughness, granted_subtype, ...)` (13 cards in the EoE pool —
    /// Blinkmoth Nexus / Lavaclaw Reaches / Wandering Fumarole
    /// manland-animate activated abilities).
    ///
    /// **"Self" semantics — manland animation always targets the
    /// activator's own land.** The recipient is the source GameObject
    /// itself (the activated-ability source). Same shape as
    /// `PumpSelfUntilEOT`. If the source has left the battlefield
    /// between activation and resolve, the effect is a silent no-op.
    ///
    /// **The land stays a land.** Per CR 305.3, an animated land is
    /// BOTH a land and a creature for the duration. The `card_types`
    /// of the printing remains unchanged (still contains "Land"); the
    /// `BecomesLandCreature` modifier conceptually adds "Creature" to
    /// the effective type set. Effective-P/T readers
    /// (`battlefield_creature_power` / `_toughness`,
    /// `effective_power` / `effective_toughness`) honor the modifier:
    /// when present, they treat the object as a creature with base
    /// `power`/`toughness` (layer 7b override). Layer 7c additive
    /// bonuses (`PowerToughnessBonus`) still stack on top.
    ///
    /// **Mechanism.** Pushes a `UeotModifier::BecomesLandCreature {
    /// power, toughness, color_or_subtype }` entry onto the source's
    /// `ueot_modifiers` Vec. Auto-expires at the next CR 514.3
    /// UEOT-clear pass (cleanup step), reverting the land to its
    /// non-creature state.
    ///
    /// **Schema simplification vs Python.** Python's
    /// `BecomeLandCreatureUEOT` carries `granted_keywords`,
    /// `unblockable`, `granted_activated_abilities`, and
    /// `granted_triggered_abilities`. The Rust first slice carries
    /// only `power`, `toughness`, and optional `granted_subtype` —
    /// the keyword / unblockable / granted-ability riders compose
    /// via separate effect chains or wait for follow-up slices.
    /// `(0, 0)` with no subtype is a legal no-op for parity edge
    /// cases.
    BecomeLandCreatureUEOT {
        power: i32,
        toughness: i32,
        granted_subtype: Option<String>,
    },
    /// **Round 29 optional-mana-cost wrapper.** "You may pay {cost}. If
    /// you do, [inner_effect]." Mirrors Python's
    /// `OptionalManaCostThenEffect(cost_str, inner_effect)`
    /// (`effects/combinators.py:1469` — Kavaron-Harrier / Oreplate-
    /// Pangolin "may pay {1}; if you do, ..." trigger family, freq 2
    /// in the EoE pool).
    ///
    /// **Resolve semantics.** Pushes a
    /// `PendingDecision::OptionalManaCostChoice` onto the queue and
    /// short-circuits. The controller drains via
    /// `submit_optional_mana_cost_decision(pay)` (manual) or
    /// `auto_resolve_pending_decisions("default" | "optional_pay_yes"
    /// | "optional_pay_no")`. On a `pay=true` submit, the dispatcher
    /// validates the controller can pay the cost from their mana
    /// pool; if so, the cost is deducted atomically and the inner
    /// effect is applied via `apply_effect` (with the original
    /// `source_obj_id` so identity-coupled riders retain identity).
    /// If the pool can't afford the cost, `pay=true` raises
    /// `ValueError` transactionally — no mutation. On `pay=false`,
    /// the decision is dropped.
    ///
    /// **Python parity for the no-agent default.** Python's
    /// `OptionalManaCostThenEffect.resolve` is auto-resolving: it
    /// pays whenever the controller can afford, otherwise the trigger
    /// does nothing. The Rust `"default"` strategy mirrors this — if
    /// the pool can afford the cost, pay it and apply the inner; if
    /// not, drop the decision silently.
    OptionalManaCostThenEffect {
        cost: ManaCost,
        inner_effect: Box<Effect>,
    },
    /// **Round 28 freq-2 closure — Starwinder / Fear-of-Failed-Tests
    /// shape.** "When a creature you control deals combat damage to
    /// a player, you may draw that many cards." Mirrors Python's
    /// `DrawCardsEqualToResolvingCombatDamage`
    /// (`effects/generic/draw_mill.py:89` — 2 EoE cards). The "may"
    /// is engine-flattened to always-yes (same precedent as other
    /// OptionalXyz simplifications elsewhere — see the Python class
    /// docstring).
    ///
    /// **Mechanism.** Reads
    /// `state.resolving_combat_damage_amount` (i32 slot on
    /// GameState), then draws that many cards for the source's
    /// controller (zone-scan + stack-scan fallback, same pattern as
    /// DiscardSelf et al.). `amount <= 0` is a silent no-op.
    /// Negative library length is handled by the standard CR 704.5b
    /// path (player flagged for empty-library loss).
    ///
    /// **Write site (I32, this slice closes the deferral).** The
    /// `resolving_combat_damage_amount` slot is now written
    /// per-event inside
    /// `state_ops_combat.rs::GameState::apply_damage_pass` — set
    /// to `entry.amount` before the per-entry mutations fire, reset
    /// to 0 after. Mirrors Python's per-event set/reset boundary
    /// (`combat.py:618,623` granted dealt-damage trigger resolve,
    /// `actions.py:2684,2699` stack-pushed trigger resolve). The
    /// Effect arm reads the slot; under normal combat the slot is
    /// the dealt amount for whichever event is in flight. The
    /// `set_resolving_combat_damage_amount` test setter remains
    /// available for synthetic / cross-engine harness tests that
    /// emulate the apply-loop write outside a real combat flow.
    /// RUST_PARITY_GAPS.md I32 documents the lifecycle.
    DrawCardsEqualToResolvingCombatDamage,
    /// **G7 Class-C delayed-trigger port (round-78).** The
    /// `WarpExileSelf` delayed triggered ability: at the BEGINNING of
    /// the END step of the turn in which a creature was warp-cast, the
    /// warp creature's controller exiles it (CR 603.7 delayed trigger).
    ///
    /// **Mirrors Python's `WarpExileSelf`
    /// (`src/mtg_bot/effects/generic/exile.py:133`).** Scheduled at
    /// warp-cast time into `state.delayed_triggers`; pushed onto the
    /// stack as a real `StackItem::TriggeredAbility` at PHASE_END_STEP
    /// entry via `run_delayed_end_step_triggers`. The resolver arm
    /// looks up the source obj by `source_obj_id` on the battlefield,
    /// moves it bf → owner's exile with the CR-400.7 reset (damage /
    /// tapped / attached_to / exiled_with_source), stamps
    /// `warped = true` if non-token, and clears the cast-time
    /// `warp_exile_pending` marker. No-op if the source has left the
    /// battlefield between the schedule (cast) and the resolve (end-
    /// step) — countered / died / bounced / already-exiled (CR 603.6c).
    ///
    /// **Unit variant — no payload.** The schedule-time obligation is
    /// carried by the `DelayedTrigger` entry (`source_id`,
    /// `fire_at_turn`, `controller_id`). At push time
    /// (`push_triggered_ability`), the `source_obj_id` flows onto the
    /// `StackItem::TriggeredAbility` and is read by the resolver to
    /// locate the bf object to exile. Same shape as
    /// `PutPlusOneCounterOnSelf` — the resolver uses
    /// `source_obj_id` rather than a target picked at push time.
    WarpExileSelf,
    /// **BBB downstream complement — 7th Rust slice.**
    /// Defending player (each opponent in 2-player) loses half their
    /// life, rounded up. Mirrors Python's
    /// `DefendingPlayerLosesHalfLifeRoundedUp`
    /// (`effects/generic/life.py:570`). Engine approximation: "defending
    /// player" → "each opponent" (correct in 2-player EoE; documented
    /// approximation per the Python class docstring). One EoE card:
    /// Alpharael, Stonechosen — "Void — Whenever Alpharael, Stonechosen
    /// attacks, if Void is active, defending player loses half their
    /// life, rounded up." The Void outer shell is `ConditionalIfVoid`;
    /// this is the inner then_effect.
    ///
    /// **Resolve semantics.** For each player whose id ≠
    /// source-controller AND who has not yet lost: compute
    /// `loss = (p.life + 1) / 2` (integer ceiling of `p.life / 2`),
    /// subtract from `p.life`. Matches Python's
    /// `loss = (p.life + 1) // 2` (floor division on `life + 1`).
    DefendingPlayerLosesHalfLifeRoundedUp,
    /// **BBB downstream complement — 7th Rust slice.**
    /// Run `inner` only if the effect's controller gained at least
    /// `threshold` life this turn at resolution.  Mirrors Python's
    /// `ConditionalIfYouGainedLifeAtLeast(threshold, inner)`
    /// (`effects/combinators.py:966`).  Reads the controller's
    /// `Player.life_gained_this_turn` counter (now tracked in Rust
    /// via the `GainLife` resolve arm + cleanup-step reset).
    ///
    /// EoE card unlocked: Haliya, Guided by Light — "At the beginning
    /// of your end step, draw a card if you've gained 3 or more life
    /// this turn." Parser wraps `DrawCards(1)` in this conditional.
    ///
    /// **Resolve semantics.** Read `controller.life_gained_this_turn`;
    /// if `< threshold`, log and no-op. Otherwise apply `inner`.
    ConditionalIfYouGainedLifeAtLeast {
        threshold: i32,
        inner: Box<Effect>,
    },
    /// **BBB downstream complement — 7th Rust slice.**
    /// Run `effect` only if the triggering source's effective power is
    /// at least `threshold` at resolution. Mirrors Python's
    /// `ConditionalIfSourcePowerAtLeast(effect, threshold=4)`
    /// (`effects/combinators.py:1055`). Reads
    /// `state.resolving_source` (= the source_obj_id at apply_effect
    /// call time — resolved via the existing `source_obj_id` parameter
    /// to `apply_effect`). If the source is absent from the battlefield
    /// (died before resolution) or its effective power is below
    /// `threshold`, no-op.
    ///
    /// EoE cards unlocked: Syr Vondam, Sunstar Exemplar ("while its
    /// power is 4 or greater" on a DIES trigger), Dawnstrike Vanguard,
    /// Frontline War-Rager.  Default threshold = 4 (most EoE cards).
    ///
    /// **Resolve semantics.** Read effective power via
    /// `predicate_subject_effective_power(state, source_obj_id)`.
    /// None (source not on bf) → log + no-op (mirrors Python's
    /// `if source is None: return`).  `power < threshold` → log + no-op.
    /// Otherwise apply `effect`.
    ConditionalIfSourcePowerAtLeast {
        threshold: i32,
        effect: Box<Effect>,
    },
    /// **EEE downstream complement — 8th Rust slice (JJJ Pin 1).**
    /// Dyadrine, Synthesis Amalgam's attack trigger effect.  Mirrors
    /// Python's `DyadrineAttackEffect` (a zero-field frozen dataclass in
    /// `effects/bespoke.py`).
    ///
    /// Oracle text: "Whenever you attack, you may remove a +1/+1 counter
    /// from each of two creatures you control. If you do, draw a card and
    /// create a 2/2 colorless Robot artifact creature token."
    ///
    /// **Resolve semantics (honest-stop — parse-only with sentinel).**
    /// The resolve logic is complex (agent-interactive counter-removal
    /// from two controller-owned creatures, conditional draw, token
    /// creation) and is card-specific.  The Rust resolve arm is a
    /// sentinel no-op: it logs the trigger and skips resolution.  This
    /// is sufficient to unblock Dyadrine at the *parse layer* (Rust no
    /// longer raises PyValueError for `dyadrine_attack_effect` dicts) and
    /// to expose Dyadrine to the cross-engine bridge.  Full behavioral
    /// parity (resolve parity) is deferred to a follow-up slice.
    DyadrineAttackEffect,
    /// **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
    /// Give `amount` poison counters to the target player.  Mirrors
    /// Python's `PutPoisonCountersOnTargetPlayer(amount: int = 1)`
    /// (`effects/generic/counters.py`).
    ///
    /// Oracle text (Virulent Silencer): "Whenever a nontoken artifact
    /// creature you control deals combat damage to a player, that player
    /// gets two poison counters."
    ///
    /// **Parse shape.** Dict: `{"kind":
    /// "put_poison_counters_on_target_player", "amount": <int>}`.
    /// `amount` defaults to 1 when absent (mirrors Python class default).
    ///
    /// **Resolve semantics.** Reads the trigger's resolved target via
    /// `state.current_resolve_payload_target_player_id` (the player id
    /// bound at trigger-fire time from the DEALS_COMBAT_DAMAGE event).
    /// Falls back to subjects[0] if the payload slot is absent.
    /// Increments `target_player.poison_counters` by `amount`.  Per CR
    /// 704.5c, a player with 10+ poison counters loses the game; that
    /// SBA is enforced by `apply_sbas` once the effect resolves.
    PutPoisonCountersOnTargetPlayer {
        amount: i32,
    },
    /// **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for all-
    /// untargeted modal cards, 2026-05-21).** Exile every artifact on
    /// the battlefield with mana value `<= max_mv`. Untargeted —
    /// Ruinous Rampage's anti-artifact mode (the other mode is
    /// `DamageEachOpponent`).
    ///
    /// **Parse shape.** Dict: `{"kind":
    /// "exile_all_artifacts_with_mana_value_at_most", "max_mv": <int>}`.
    /// `max_mv` is REQUIRED (mirrors Python's regex-driven parser at
    /// `cards/parser/effects.py:910` which captures the digit; the
    /// `ExileAllArtifactsWithManaValueAtMost` Python class default is 3
    /// but the parser never emits an instance without an explicit
    /// `max_mv`). Negative `max_mv` is rejected at parse time (a
    /// schema invariant — MV is non-negative per CR 202.3).
    ///
    /// **Resolve semantics (mirrors Python's
    /// `ExileAllArtifactsWithManaValueAtMost.resolve`,
    /// `effects/generic/exile.py:482-494`).** Snapshot every
    /// battlefield object whose printing `is_artifact` AND whose
    /// printing mana_cost total `<= max_mv`; iterate the snapshot and
    /// move each to exile via the same off-battlefield helper the
    /// `DestroyAllCreatures` arm uses (zone-change reset on the way
    /// off — tokens cease to exist per CR 704.5d, non-token cards land
    /// in their owner's exile zone). Lander tokens (MV 0) are hit.
    /// The source itself can be hit if it's an artifact spell (it is
    /// NOT — Ruinous Rampage is a Sorcery, so the source is on the
    /// stack at resolve time, not the battlefield).
    ///
    /// **Negative `max_mv` raises** at resolve (defense-in-depth on
    /// top of the parse-time reject; mirrors the
    /// `DamageEachOpponent` raise-on-negative discipline).
    ExileAllArtifactsWithManaValueAtMost { max_mv: i32 },
    /// **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
    /// 2026-05-21).** Put `amount` charge counters on the cast-time-
    /// bound target permanent IF the target is still a Spacecraft OR
    /// Planet AND is still controlled by the source-controller at
    /// resolve time (CR 608.2b re-check). Drill Too Deep mode 0.
    ///
    /// **Field shape.** Single `amount: i32` field — mirrors Python's
    /// `PutChargeCountersOnTargetSpacecraftOrPlanetYouControl(amount:
    /// int = 5)` dataclass at
    /// `effects/generic/counters.py:489-544`. Default 5 (Drill Too
    /// Deep's printed amount); parser always emits an explicit
    /// `amount` (the regex digit/cardinal capture at
    /// `parser/effects.py:3002-3005`). Negative `amount` rejected at
    /// parse time (schema invariant; mirrors the
    /// `ExileAllArtifactsWithManaValueAtMost` raise-on-negative
    /// precedent).
    ///
    /// **Resolve semantics (CR 608.2b — re-target).** Read the cast-
    /// time-bound target via
    /// `state.current_resolve_payload_target_id` (the proven
    /// `SeedshipImpactEffect` / `DestroyTargetArtifactOrTappedCreature`
    /// payload-target swap pattern). Look up the object on either
    /// battlefield; if the bound id is no longer on the battlefield,
    /// silently fizzle (CR 608.2b — Python's `obj is None` early
    /// return at `counters.py:526-531`). Re-verify the qualification
    /// filter (`subtypes ∋ "Spacecraft"` OR `subtypes ∋ "Planet"`)
    /// AND the controller match (`obj.controller == source_controller`);
    /// fizzle on either mismatch. On success, increment the target's
    /// `charge_counters` field by `amount`.
    ///
    /// **Distinct from `PutChargeCountersOnSelf`** — that arm
    /// increments `obj.counters` on the SOURCE (Blast-Zone shape) and
    /// uses the legacy single-counter-slot G8 simplification; this
    /// arm increments `obj.charge_counters` on the TARGET (Drill Too
    /// Deep mode 0 shape — verify-first checked the SoT class uses
    /// `obj.charge_counters += self.amount`).
    ///
    /// **Modal cast-time binding (Cycle 5 detector table).** The
    /// per-mode target binding override in
    /// `state_ops_cast.rs::cast_spell_noncreature` includes a matching
    /// arm that scans for the first Spacecraft-or-Planet permanent
    /// the source-controller controls — mirroring Python's
    /// `_select_spell_targets` per-mode first-legal convention.
    PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { amount: i32 },
    /// **Multi-target admit-gate Phase 4 widen (2026-05-21).** Return
    /// up to `max_n` target creature cards from the source-controller's
    /// graveyard whose summed mana value is `<= max_total_mv`, each
    /// entering the battlefield with a +1/+1 counter. Models Scout for
    /// Survivors (`{2}{W}` Sorcery, EoE, 4,170 17lands decks): "Return
    /// up to three target creature cards with total mana value 3 or
    /// less from your graveyard to the battlefield. Put a +1/+1 counter
    /// on each of them."
    ///
    /// **Field shape.** Two `i32` fields mirroring Python's
    /// `ReturnUpToNSmallCreaturesFromGraveyardWithCounters(max_n=3,
    /// max_total_mv=3)` constructor sugar at
    /// `effects/generic/bounce.py:296-326`. Both fields non-negative
    /// (schema invariant — count and MV are non-negative; mirrors the
    /// `ExileAllArtifactsWithManaValueAtMost` raise-on-negative
    /// precedent). Per-card MV cap (`max_mv`) is intrinsic — equal to
    /// `max_total_mv` by Python's `__post_init__` — so the resolve
    /// scan filters at the per-card level too.
    ///
    /// **Resolve semantics — substrate-only.** Self-scan the source-
    /// controller's graveyard in front-to-back (deck) order for up to
    /// `max_n` creature cards whose individual mana value AND running
    /// sum-MV deductions both stay within `max_total_mv`, moving each
    /// to that player's battlefield with a +1/+1 counter applied via
    /// `obj.counters += 1` and `summoning_sick=true`, `tapped=false`,
    /// `damage=0`, fresh `etb_timestamp`. Each entrance fires ETB
    /// triggers via `pending_etb_dispatches.push(entered_id)` — the
    /// established multi-entrance pattern.
    ///
    /// **Resolve-time self-scan vs cast-time binding.** This variant
    /// elects the SAME "self-scan the controller's zone at resolve
    /// time" pattern that `ReturnCreatureFromYourGraveyardToBattlefield`
    /// uses (the single-target sibling). CR 601.2c says targets are
    /// chosen at cast time; the cross-engine harness's
    /// `_PickFirstLegalTargetAgent` picks the FIRST-N-legal in
    /// graveyard order, which coincides with the self-scan pick. A
    /// later cycle that wires a multi-target StackItem field (see
    /// `docs/SEARCH_LIMITATIONS.md`) can override the resolve scan
    /// with the StackItem-bound list; today's substrate-only landing
    /// keeps the no-fallback-hybrid discipline by mirroring the
    /// established self-scan precedent the cross-engine harness already
    /// proves byte-equal.
    ///
    /// **Empty / no-creature graveyard or empty filter pool: silent
    /// fizzle.** No targets remain after filtering — return Ok(())
    /// without state mutation. Mirrors the
    /// `ReturnCreatureFromYourGraveyardToBattlefield` early-return
    /// convention.
    ReturnUpToNSmallCreaturesFromGraveyardWithCounters {
        max_n: i32,
        max_total_mv: i32,
    },
    /// **Cracked Skull look-at-hand + may-discard substrate — Cycle 2
    /// (2026-05-22).** "Look at target opponent's hand. You may
    /// choose a nonland card from it. That player discards that
    /// card." Cracked Skull's ETB trigger half. Python class:
    /// `TargetOpponentLookHandMayDiscard` (no fields) at
    /// `src/mtg_bot/effects/generic/discard.py:298-391`.
    ///
    /// **Field-free unit variant on both engines.** Python's
    /// `@dataclass(frozen=True)` has no field slots; the controller-
    /// of-effect (the caster who is making the may-gate decision) and
    /// the target opponent (whose hand is looked at + discarded from)
    /// are both resolved at resolve time — caster via the source-
    /// controller transient (`state.current_resolve_source_controller`)
    /// and the opponent via the standard "1 - controller" in two-
    /// player.
    ///
    /// **Resolve semantics (mirrors `discard.py:330-385`).**
    /// 1. Source-controller resolution: 5-zone walk + stack scan
    ///    (mirrors the round-24 EachOpponentDiscards source-finder
    ///    pattern). Source-not-found → silent fizzle.
    /// 2. 2-player "target opponent" reduces to the one opponent:
    ///    `opponent_idx = (1 - controller) as usize`.
    /// 3. Compute candidates: every nonland card id in
    ///    `target.hand`, in hand-iteration order (matches Python's
    ///    list-comprehension order at `discard.py:352`). Land cards
    ///    are excluded per the printed "nonland card" filter; if a
    ///    card's printing has the Land card type it is filtered.
    /// 4. Empty candidates → silent no-op (Python parity at
    ///    `discard.py:353-358`: "if not options: log, return"). The
    ///    decision is NOT pushed; the Cycle 1 substrate explicitly
    ///    documents this Effect-arm invariant.
    /// 5. Non-empty candidates → stamp `revealed_to[controller] =
    ///    true` on every card in `target.hand` (the printed "look
    ///    at" step's one-way reveal — only the caster sees, the
    ///    public-reveal scope used by `TargetOpponentDiscardsCaster
    ///    Choice` does not apply here). Then push
    ///    `PendingDecision::LookAtOpponentHandMayDiscard {
    ///    source_obj_id, controller, target_player, candidates }`
    ///    onto the queue.
    ///
    /// The controller drains via
    /// `submit_look_at_opponent_hand_may_discard_decision(Some(id))`
    /// (force the opponent to discard that card; mirrors Python's
    /// `target.hand.remove + zone=GRAVEYARD + graveyard.append`) or
    /// `submit_look_at_opponent_hand_may_discard_decision(None)`
    /// (decline the may-gate; no discard fires). Auto-resolve
    /// `"default"` declines (mirrors Python's no-agent-seated
    /// fallback at `discard.py:362-363`).
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `TargetOpponentLookHandMayDiscard()` (field-free
    /// construction). The source-and-controller are read from
    /// `state.current_resolve_source_controller` (or fall through to
    /// `state.active_player`) at resolve time, mirroring the
    /// `SacrificeSelfUnlessTapPermanent` controller-resolution shape.
    ///
    /// **`revealed_to` substrate honest-stop.** Rust's `GameObject`
    /// does NOT today carry a `revealed_to` slot (the per-object
    /// hidden-information reveal flag); Python's `reveal_to(card,
    /// controller)` is a Python-side helper that mutates a per-object
    /// reveal-set. The Cycle 2 resolve arm therefore SKIPS the look-
    /// at reveal step — the `PendingDecision` push is the entirety of
    /// the Rust-side state mutation pre-submit. The cross-engine
    /// observable that matters at parity (the discard / decline
    /// outcome on `target_player.hand` and `target_player.graveyard`)
    /// is byte-equal because neither engine reads `revealed_to` for
    /// any subsequent state mutation. The look-at-step's hidden-info
    /// scope difference is documented honest-stop in
    /// `docs/SEARCH_LIMITATIONS.md`.
    TargetOpponentLookHandMayDiscard,
    /// **Sporogenic Infection — Cycle 1 (2026-05-22) — Aura ETB
    /// opponent-sacrifice trigger half.** "Target opponent sacrifices
    /// a creature of their choice other than enchanted creature."
    /// Sporogenic Infection's ETB trigger; the second trigger
    /// (DEALT_DAMAGE + IsHost + `DestroyEnchantedPermanent`) is
    /// already Rust-bridged (Cracked Skull C2 shipped the admit/emit
    /// for `DestroyEnchantedPermanent`; the Rust resolve arm was
    /// pre-existing). Python class:
    /// `TargetOpponentSacrificesCreatureOtherThanEnchanted` (no
    /// fields) at `src/mtg_bot/effects/generic/opponent_punishers.py`
    /// (lines 68-145).
    ///
    /// **Field-free unit variant on both engines.** Python's
    /// `@dataclass(frozen=True)` carries no slots; the controller of
    /// the trigger (the Aura's controller), the target opponent (the
    /// only opponent in 2-player), and the attached-to host (read off
    /// the Aura source) are all resolved at resolve time.
    ///
    /// **Resolve semantics (mirrors `opponent_punishers.py:100-138`).**
    /// 1. Source-finder: walk `state.players[*].battlefield` for the
    ///    Aura source (the source must be on the battlefield for the
    ///    ETB trigger to have fired; if it has LTB'd between the
    ///    trigger queueing and resolve, silent fizzle).
    /// 2. Excluded host id = `source.attached_to` (the Aura's enchanted
    ///    creature, if any). When the Aura is enchanting a creature
    ///    the target opponent controls (Sporogenic's usual case), the
    ///    host is filtered out of the sac options. When the Aura is
    ///    on a controller-owned creature, the target opponent's
    ///    creatures are unaffected by the exclusion (no overlap).
    /// 3. 2-player "target opponent" reduces to the one opponent:
    ///    `opponent_idx = (1 - controller) as usize`.
    /// 4. Build candidates: every creature on the target opponent's
    ///    battlefield EXCLUDING the host (`o.controller == target_idx
    ///    && o.printing.is_creature && o.id != excluded_id`),
    ///    bf-iteration order (matches Python's list comprehension at
    ///    `opponent_punishers.py:122-127`). Token creatures are NOT
    ///    excluded — the printed text reads "a creature of their
    ///    choice", not "a nontoken creature" (Python parity at
    ///    `opponent_punishers.py:84-86`).
    /// 5. Empty candidates → silent no-op (Python parity at
    ///    `opponent_punishers.py:128`: `if not legal: return`). The
    ///    decision is NOT pushed.
    /// 6. Non-empty candidates → push
    ///    `PendingDecision::OpponentSacrificesCreatureChoice
    ///    { source_obj_id, target_player, candidates }` onto the
    ///    queue. **Note the decision-maker is the OPPONENT
    ///    (`target_player`), not the controller** — the printed text
    ///    is "target opponent sacrifices a creature OF THEIR CHOICE";
    ///    the opponent picks via their own agent's `choose_sacrifice`
    ///    (Python parity at `opponent_punishers.py:128-132`). The
    ///    PendingDecision variant therefore drops the `controller`
    ///    field (no controller-decision in this shape) and carries
    ///    only the seat of the player who makes the choice
    ///    (`target_player`).
    ///
    /// The opponent's agent drains via
    /// `submit_opponent_sacrifices_creature_decision(chosen_id)`
    /// (sacrifice the chosen creature; mirrors Python's
    /// `_sacrifice(state, obj)` at `opponent_punishers.py:138`).
    /// Unlike the Cracked Skull may-gate, there is NO decline path —
    /// the opponent MUST sacrifice one (the trigger forces it; the
    /// printed text is "sacrifices a creature OF THEIR CHOICE", not
    /// "you MAY sacrifice"). Auto-resolve `"default"` picks
    /// `candidates[0]` (the deterministic-first creature in bf-
    /// iteration order; mirrors Python's no-agent-seated fallback at
    /// `opponent_punishers.py:128-129` — `if target.agent is None:
    /// return`, but that path leaves the opponent without making a
    /// choice; the cross-engine probe seats agents on both sides so
    /// the deterministic-first auto-resolve is the parity bridge).
    ///
    /// **Schema parity with Python.** Unit variant on both sides — no
    /// fields. The bridge translates this Rust variant to
    /// `TargetOpponentSacrificesCreatureOtherThanEnchanted()` (field-
    /// free construction). The source identity is read from
    /// `source_obj_id` (the Rust resolve-arm signature carries it
    /// directly — no `state.resolving_source` analog needed, the id
    /// IS the resolving source's id).
    ///
    /// **Why a new PendingDecision variant (not sibling reuse).** The
    /// Cracked Skull C1 `LookAtOpponentHandMayDiscard` substrate has a
    /// controller-decision shape (the caster makes the may-gate
    /// pick). Sporogenic's sacrifice-of-their-choice is an OPPONENT-
    /// decision shape (the target opponent picks which of THEIR
    /// creatures to sacrifice). Distinct seat → distinct variant; the
    /// sibling pattern would be misleading. The new variant
    /// `OpponentSacrificesCreatureChoice` reuses the candidate-id-list
    /// idiom (`Vec<u32>` of legal sacrifice targets) but routes the
    /// submit to `target_player`'s sacrifice path, not
    /// `controller`'s.
    TargetOpponentSacrificesCreatureOtherThanEnchanted,
}

/// **BBB downstream complement — 7th Rust slice.**
/// Two-tier predicate architecture mirroring Python's
/// `StatePredicate` ABC (`effects/state_predicates.py`).  A
/// `StatePredicate` checks a global game-state condition; it is
/// wrapped by `Predicate::StateBound` (a `TriggerPredicate`) to
/// enter the trigger-predicate evaluation path.
///
/// Wired kinds (XX Pin 3 closure):
///   * `VoidActive` — Void is active (a nonland permanent left the
///     battlefield OR a spell was warped this turn).  Reads
///     `state.permanent_left_battlefield_this_turn ||
///     state.spell_warped_this_turn`.  Viewer-independent (global flag).
///   * `AnyPlayerControlsNoCreatures` — at least one player
///     controls no creatures.  Scans all players' battlefields.
///     Viewer-independent.  EoE card: Sothera, the Supervoid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StatePredicate {
    /// Void is active: `permanent_left_battlefield_this_turn ||
    /// spell_warped_this_turn`. Viewer-independent.
    VoidActive,
    /// At least one player controls no creatures (battlefield is
    /// empty or holds only non-Creature permanents). Viewer-
    /// independent. Sothera, the Supervoid's gate predicate.
    AnyPlayerControlsNoCreatures,
    /// **MMM Pin 1 (11th Rust slice) — SpellsCastThisTurn(n).**
    /// The viewer has cast EXACTLY ``n`` spells this turn (counted
    /// via the per-player `Player.spells_cast_this_turn` counter).
    /// Mirrors Python's `SpellsCastThisTurn(n: int)` from
    /// `effects/state_predicates.py:184` — reads
    /// `player.spells_cast_this_turn == n` post-increment of the
    /// cast site (the dispatcher fires AFTER the counter has been
    /// bumped, so the "second spell" trigger latches when the count
    /// flips to exactly 2).
    ///
    /// **VIEWER-RELATIVE.** Reads the source's controller player
    /// id (passed by `Predicate::StateBound::matches` as
    /// `viewer_id`) and indexes into `state.players[viewer_id]`.
    /// This is the first StatePredicate variant that actually
    /// USES the viewer_id parameter — VoidActive and
    /// AnyPlayerControlsNoCreatures both ignore it (global flag
    /// or all-players scan).
    ///
    /// **EoE cards.** Cosmogrand Zenith, Sunstar Lightsmith,
    /// Illvoi Operative, Station Monitor — all carry
    /// `StateBound(SpellsCastThisTurn(n=2))` as the second-spell
    /// gate on an IS_CAST trigger.
    SpellsCastThisTurn { n: i32 },
}

impl StatePredicate {
    /// Evaluate the state predicate.
    ///
    /// `_viewer_id` is the source-controller's player id passed in
    /// from `Predicate::StateBound::matches`.  Both current wired
    /// predicates are viewer-independent; the parameter is present
    /// for forward-compatibility when viewer-relative predicates
    /// (e.g. "you control more lands than an opponent") land.
    pub(crate) fn test(
        &self,
        state: &crate::state::GameState,
        viewer_id: u8,
    ) -> bool {
        match self {
            StatePredicate::VoidActive => {
                state.permanent_left_battlefield_this_turn
                    || state.spell_warped_this_turn
            }
            StatePredicate::AnyPlayerControlsNoCreatures => {
                state.players.iter().any(|p| {
                    !p.battlefield.iter().any(|obj| {
                        state
                            .registry
                            .printings
                            .get(obj.printing_id as usize)
                            .map(|pr| pr.card_types.iter().any(|t| t == "Creature"))
                            .unwrap_or(false)
                    })
                })
            }
            StatePredicate::SpellsCastThisTurn { n } => {
                // **MMM Pin 1 — SpellsCastThisTurn.** Read
                // `player.spells_cast_this_turn == n` for the
                // source's controller (viewer_id resolved by
                // `Predicate::StateBound::matches`). Mirrors
                // Python's `SpellsCastThisTurn.test`:
                //     return player.spells_cast_this_turn == self.n
                // Out-of-range viewer_id (defensive guard for the
                // sentinel path) falls back to `false` — matches
                // Python's `state.player_by_id(viewer_id)` raising
                // KeyError caught by the outer try/except → False.
                let idx = viewer_id as usize;
                if idx >= state.players.len() {
                    return false;
                }
                state.players[idx].spells_cast_this_turn == *n
            }
        }
    }
}

/// **G7 Class-C delayed-trigger port (round-78).** Single entry on
/// `GameState.delayed_triggers`. Mirrors Python's dict entry shape at
/// `src/mtg_bot/actions.py:1554-1559` /
/// `src/mtg_bot/turn.py:773-820`. Scheduled at warp-cast time;
/// scanned at the start of every END step by
/// `state_ops_steps.rs::run_delayed_end_step_triggers`; matching
/// entries (`fire_at_turn == state.turn_number &&
/// controller_id == state.active_player`) are pushed onto the stack
/// as real triggered abilities via `push_triggered_ability` and
/// removed from the Vec. Non-matching entries (e.g. an opponent's-
/// turn warp where `fire_at_turn = turn + 1`) ride to the next end
/// step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayedTrigger {
    /// The turn whose end step the trigger fires on. Mirrors Python's
    /// `"fire_at_turn": state.turn_number + offset` where `offset = 0`
    /// when the caster is the active player (sorcery-speed warp on
    /// own turn) and `1` otherwise (instant-speed warp on opponent's
    /// turn — defer to the caster's next end step). For warp, Phase-1
    /// admitted cards are sorcery-speed Artifact Creatures only, so
    /// `offset = 0` is the common case, but the Vec is generalized to
    /// the 2-offset shape for future instant-speed warps.
    pub fire_at_turn: u32,
    /// The player whose END step pushes the trigger. Same player as
    /// the warp caster (the creature's controller at cast time).
    /// Mirrors Python's `"controller_id": player_id` (`actions.py:
    /// 1556`).
    pub controller_id: u8,
    /// The warp creature's object id. Survives even after the
    /// permanent leaves the battlefield (the resolver re-looks up the
    /// bf and no-ops if the object is gone; CR 603.6c — the trigger
    /// "fires" but has no permanent to act on).
    pub source_id: u32,
    /// The effect to push as a `StackItem::TriggeredAbility`. For G7
    /// Class-C this is exclusively `Effect::WarpExileSelf`; the field
    /// is a full `Effect` to match Python's
    /// `"effect": WarpExileSelf()` dict shape and to leave room for
    /// the other Python delayed-trigger effects
    /// (`SacrificeSourceDelayed`,
    /// `RemoveChargeCountersFromSourceDelayed`,
    /// `ReturnExiledPermanentToOwnerTapped`) when their producers
    /// land in Rust.
    pub effect: Effect,
}

// --- ETB trigger substrate (Slice 1) -----------------------------
//
// Mirrors Python's `src/mtg_bot/effects/predicates.py:Event` +
// `effects/base.py:TriggeredAbility`. First-slice scope (per
// `docs/RUST_ETB_TRIGGER_AUDIT.md` §6.1):
//
// - **Event::Enters only** is wired today. Other variants land in
//   later slices (Slice 3+); listing them here lets the parser /
//   matcher fail loud on out-of-scope events at register time
//   without expanding scope.
// - **Predicate algebra is minimal**: `Always`, `SourceIsSelf`,
//   `SourceIsCreature`, `Not(Box<Predicate>)`, `And(Box<Predicate>,
//   Box<Predicate>)`. Enough to express the smoke-probe shape
//   ("when ANOTHER creature enters under your control" — see
//   audit §5.4 Example A) and the canonical IsSelf shape ("when
//   ~ enters"). Richer predicates (HasSubtype, IsLand, IsArtifact,
//   Controlled, Subject) follow in Slice 3+.
// - **TriggeredAbility carries `event`, `predicate`, `effect`, and
//   `once_per_turn`.** The Python shape also carries `requires_void`
//   / `tier_threshold` / `room_half`; the Rust mirror omits those
//   gates in Slice 1 (they default to "always fire") because the
//   Slice 1 smoke probe doesn't need them. Slice 3+ adds them in
//   lockstep with the Predicate expansion.
//
/// Event taxonomy for the trigger dispatcher (mirror of
/// Python's `Event` enum at `src/mtg_bot/effects/predicates.py`).
/// First-slice scope wires only `Enters`; later slices add the
/// rest. Listed here so the parser can validate the registry-dict
/// `event` key at register time rather than at dispatch time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// "When/whenever X enters the battlefield" — fires from
    /// `dispatch_event(Event::Enters, source_id)` walks. Today's
    /// only wired event (Slice 1).
    Enters,
    /// "When X dies" / "When X is put into a graveyard from
    /// the battlefield". Stub for Slice 4+; no dispatch site yet.
    Dies,
    /// "When you cast a spell" / "When X is cast". Stub for
    /// Slice 5+; no dispatch site yet.
    Cast,
    /// "When X attacks". Slice 7 wires this — dispatched inline at
    /// the `declare_attacker` site after the per-attacker push to
    /// `state.attackers`. SourceIsSelf matches naturally via
    /// `source_id == subject_id` on the bf-snapshot observer set.
    Attacks,
    /// "When X blocks". Slice 9 wires this — dispatched inline at
    /// the `declare_blocker` site after the per-pair push to
    /// `state.blockers`. Single-subject shape: subject = blocker
    /// (multi-subject blocker + attacked-creature predicates are
    /// out of scope until `Predicate::Subject(slot, inner)` lands
    /// alongside Event::DealsCombatDamage). SourceIsSelf
    /// ("whenever this creature blocks") matches naturally via
    /// `source_id == subject_id`.
    Blocks,
    /// "When X deals damage" / combat-damage triggers. Stub for
    /// Slice 12+; no dispatch site yet.
    DealsDamage,
    /// "When X leaves the battlefield". Fires from
    /// `dispatch_event(Event::LeavesBattlefield, leaving_id)` on
    /// every bf-exit primitive (move_object off-bf + SBA destroy).
    /// Slice 8 wires this — see RUST_ETB_TRIGGER_AUDIT.md "Slice 8
    /// completion notice".
    ///
    /// **Relationship to DIES:** every creature death is also a bf
    /// exit, so SBA-destroyed creatures fire BOTH DIES and LEAVES
    /// (push order: DIES first via `pending_dies_dispatches`,
    /// LEAVES second via `pending_leaves_dispatches`). Mirrors
    /// Python's `move_off_battlefield` (turn.py:280-289) which
    /// invokes both `dispatch_event(DIES)` and
    /// `_fire_leaves_battlefield_triggers` in sequence.
    LeavesBattlefield,
    /// "Whenever X becomes tapped". Fires from
    /// `dispatch_event(Event::BecomesTapped, tapped_id)` at every
    /// production tap-the-permanent site that mirrors Python's
    /// `turn.tap_permanent` callers — round-58 BECOMES_TAPPED event
    /// bridge. Mirrors Python's `Event.BECOMES_TAPPED` enum member
    /// (`src/mtg_bot/effects/predicates.py:60` — string value
    /// `"becomes_tapped"`) and its single firing chokepoint
    /// `turn.tap_permanent` (`src/mtg_bot/turn.py:542-571`).
    ///
    /// **Firing-set parity (CR 701.21a + §2 bridge-correctness).**
    /// Python fires this ONLY through `tap_permanent` (the no-op-if-
    /// already-tapped chokepoint), which is called from: tap-for-mana
    /// (`actions.py:auto_tap_for_cost`), the `{T}` activated-ability
    /// `tap_self` cost (`actions.py:_activate_ability`),
    /// `tap_permanent_you_control` activated cost
    /// (`actions.py:2258`), convoke (`actions.py:673`), crew
    /// (`actions.py:2402`), attack-tap of a non-vigilance attacker
    /// (`combat.py:104`, fired BEFORE the per-attacker ATTACKS
    /// dispatch at `combat.py:115`), and the tapper-effect family
    /// (`effects/generic/tap.py`, `effects/combinators.py:163`,
    /// `effects/bespoke.py:496`). Python deliberately does NOT fire
    /// it for permanents that ENTER tapped (`_play_land`'s
    /// `card.tapped = True`, search-land-tapped, reanimate-tapped,
    /// token-created-tapped) nor for regeneration's replacement tap
    /// (`turn.py:1268`) — those are direct `obj.tapped = True`
    /// assignments, not `tap_permanent` calls (CR 701.21a: a
    /// permanent entering tapped never "becomes tapped"). Rust's
    /// emission set matches this exactly; the entering-tapped /
    /// regeneration-tap Rust sites stay raw `obj.tapped = true`.
    BecomesTapped,
    /// "At the beginning of your end step" / "At the beginning of
    /// each end step" — round-14 PHASE_STEP event bridge. Fired by
    /// `advance_phase` when the new phase is `PHASE_END_STEP`.
    ///
    /// **Player subject convention.** Phase-step events use a player
    /// as the subject rather than a game object. The dispatch site
    /// passes `PLAYER_SUBJECT_OFFSET + active_player as u32` as the
    /// `subject_id` so `ControlledByYou` can distinguish player
    /// subjects (id ≥ `PLAYER_SUBJECT_OFFSET`) from game-object
    /// subjects (id < `PLAYER_SUBJECT_OFFSET`). Mirrors Python's
    /// `dispatch_event(state, Event.BEGINNING_OF_END_STEP,
    /// player=state.active_player())` shape — the `player` kwarg IS
    /// the `"player"` subject slot in `EVENT_SUBJECTS`.
    ///
    /// **Predicate shapes.** Most end-step triggers parse to
    /// `Controlled(YOU)` ("at the beginning of YOUR end step") which
    /// maps to `Predicate::ControlledByYou` — fires only for sources
    /// whose controller IS the active player. The "each end step"
    /// variant parses to `Always()` — fires for all sources on every
    /// end step. Both are handled by the existing predicate arms; no
    /// new predicate variants are needed.
    ///
    /// Mirrors Python's `Event.BEGINNING_OF_END_STEP` enum member
    /// (string value `"beginning_of_end_step"`,
    /// `src/mtg_bot/effects/predicates.py:126`).
    BeginningOfEndStep,
    /// "At the beginning of your upkeep" — round-14 PHASE_STEP event
    /// bridge. Fired by `advance_phase` when the new phase is
    /// `PHASE_UPKEEP`. Same player-subject convention as
    /// `BeginningOfEndStep`; same `ControlledByYou` / `Always`
    /// predicate shapes. Mirrors Python's
    /// `Event.BEGINNING_OF_UPKEEP` (string value
    /// `"beginning_of_upkeep"`).
    BeginningOfUpkeep,
    /// "At the beginning of combat on your turn" — round-14
    /// PHASE_STEP event bridge. Fired by `advance_phase` when the
    /// new phase is `PHASE_BEGIN_COMBAT`. Same player-subject
    /// convention and predicate shapes. Mirrors Python's
    /// `Event.BEGINNING_OF_COMBAT` (string value
    /// `"beginning_of_combat"`).
    BeginningOfCombat,
    /// "Whenever you attack" — round-15 YOU_ATTACK event bridge. Fired
    /// once per combat phase when the active player has declared at
    /// least one legal attacker, AFTER all individual `Event::Attacks`
    /// dispatches complete.
    ///
    /// **Distinct from `Event::Attacks`.** `Attacks` fires PER
    /// ATTACKER (subject = the attacking creature); `YouAttack` fires
    /// ONCE PER COMBAT (subject = the attacking player). The MTG oracle
    /// text "whenever [creature X] attacks" → `Event::Attacks`; the
    /// oracle text "whenever you attack" → `Event::YouAttack`.
    ///
    /// **Player subject convention.** Reuses the same
    /// `PLAYER_SUBJECT_OFFSET` sentinel as the phase-step events:
    /// dispatch passes `PLAYER_SUBJECT_OFFSET + active_player as u32`
    /// as the `subject_id`. `ControlledByYou` resolves the player index
    /// directly from the sentinel range; `Always` ignores the subject_id
    /// entirely. No new predicate arms are needed.
    ///
    /// **Dispatch site.** Fired in the `DeclareAttackers` arm of
    /// `apply_action` (state_ops_actions.rs) after the per-attacker
    /// loop, when `state.attackers` is non-empty (i.e. at least one
    /// legal attacker was registered). Mirrors Python's
    /// `_fire_you_attack_triggers(state, declared)` in `combat.py`
    /// which skips the dispatch when `declared` is empty.
    ///
    /// Mirrors Python's `Event.YOU_ATTACK` (string value `"you_attack"`,
    /// `src/mtg_bot/effects/predicates.py`).
    ///
    /// 8 EoE cards unlocked by this event (OO Pin 3 closure).
    YouAttack,
    /// "Whenever you put one or more counters on X" — round-16 (RR Pin 1)
    /// COUNTERS_ADDED event bridge. Single-subject shape: subject = the
    /// permanent that received the counters (``("obj",)`` per Python's
    /// ``EVENT_SUBJECTS``).
    ///
    /// **Dispatch site.** Fired from the ``Effect::AddCounters`` arm
    /// (``state_ops_resolve.rs``) AFTER the counter delta is applied to
    /// the target object, and from ``Effect::PutPlusOneCounterOnSelf``
    /// AFTER the self-counter mutation — both are the canonical Rust
    /// counter-placement chokepoints. Mirrors Python's
    /// ``fire_counters_added_to_your_creature_triggers`` in ``turn.py``
    /// which dispatches ``Event.COUNTERS_ADDED`` after each
    /// ``add_plus_one_counters`` call.
    ///
    /// **Subject.** ``subject_id`` = the object id of the permanent that
    /// received counters. Matches ``source_id == subject_id`` for
    /// ``Predicate::SourceIsSelf`` ("whenever you put counters on this
    /// creature").
    ///
    /// Mirrors Python's ``Event.COUNTERS_ADDED`` (string value
    /// ``"counters_added"``, ``src/mtg_bot/effects/predicates.py``).
    ///
    /// EoE cards unlocked: Terrasymbiosis, Haliya Ascendant Cadet
    /// (RR Pin 1 closure).
    CountersAdded,
    /// "Whenever you sacrifice X" — round-16 (RR Pin 2) IS_SACRIFICED
    /// event bridge. Single-subject shape: subject = the permanent that
    /// was sacrificed (``("obj",)`` per Python's ``EVENT_SUBJECTS``).
    ///
    /// **Distinct from ``Event::Dies``.** Every sacrifice causes a DIES
    /// dispatch AND an IS_SACRIFICED dispatch (sacrifice is a subset of
    /// death). IS_SACRIFICED is the sacrifice-specific event used by
    /// "whenever you sacrifice ..." oracle text, while DIES covers all
    /// battlefield-to-graveyard transitions. Mirrors Python's separation
    /// of ``Event.IS_SACRIFICED`` and ``Event.DIES``.
    ///
    /// **Dispatch site.** Fired from the ``Effect::Sacrifice`` arm
    /// (``state_ops_resolve.rs``) AFTER the sacrificed permanent has been
    /// moved to the graveyard and DIES/LEAVES are queued. Subject =
    /// ``leaving_obj_id`` (the id of the sacrificed permanent). Extra-
    /// observer pattern mirrors DIES: the sacrificed object has already
    /// left the battlefield so it cannot be in the bf observer set; the
    /// dispatch call must use ``dispatch_event_with_extra`` or inline the
    /// extra-observer inclusion for self-triggers.
    ///
    /// Mirrors Python's ``Event.IS_SACRIFICED`` (string value
    /// ``"is_sacrificed"``, ``src/mtg_bot/effects/predicates.py``).
    ///
    /// EoE cards unlocked: Lightless Evangel, Biotech Specialist
    /// (RR Pin 2 closure).
    IsSacrificed,
    /// "Whenever X is dealt damage" — round-16 (RR Pin 3) DEALT_DAMAGE
    /// event bridge. Single-subject shape: subject = the permanent that
    /// RECEIVED damage (``("obj",)`` per Python's ``EVENT_SUBJECTS``).
    ///
    /// **Distinct from ``Event::DealsDamage`` (deals_combat_damage).**
    /// ``DealsDamage`` fires for the DEALER of combat damage; this
    /// ``DealtDamage`` variant fires for the RECEIVER. Two events model
    /// opposite sides of the same damage exchange.
    ///
    /// **Dispatch site.** Fired after damage is marked on a permanent
    /// that is the RECEIVER (not the dealer). In the blocker-vs-attacker
    /// combat loop (``state_ops_resolve.rs``) after each
    /// ``apply_damage_marker`` call that marks a creature with positive
    /// damage. Mirrors Python's ``_fire_enchanted_dealt_damage_triggers``
    /// in ``combat.py`` which calls
    /// ``dispatch_event(state, Event.DEALT_DAMAGE, obj=obj, …)``.
    ///
    /// Mirrors Python's ``Event.DEALT_DAMAGE`` (string value
    /// ``"dealt_damage"``, ``src/mtg_bot/effects/predicates.py``).
    ///
    /// EoE cards unlocked: Requiem Monolith, Pain for All
    /// (RR Pin 3 closure).
    DealtDamage,
    /// "At the beginning of your postcombat main phase" — PPP Pin 3
    /// (10th Rust slice). Fired by `advance_phase` when the new phase
    /// is `PHASE_MAIN2` (the postcombat main step). Single-subject
    /// shape: subject = the active player whose postcombat main is
    /// beginning (`("player",)` per Python's `EVENT_SUBJECTS`).
    ///
    /// **Player subject convention.** Same `PLAYER_SUBJECT_OFFSET`
    /// sentinel as `BeginningOfEndStep`/`BeginningOfUpkeep`/
    /// `BeginningOfCombat`. `ControlledByYou` resolves the player
    /// index from the sentinel range; `Always` ignores the subject_id.
    ///
    /// **The "survivor" predicate pattern.** All 5 blocked EoE cards
    /// share the predicate composition `And(And(Controlled(YOU),
    /// IsSelf), Source(IsTapped))` — "at the beginning of your
    /// postcombat main phase, if this creature is tapped". The
    /// `Source(IsTapped)` predicate is already wired (round-21 RR
    /// closure); the only blocker was this event arm.
    ///
    /// Mirrors Python's `Event.BEGINNING_OF_POSTCOMBAT_MAIN` (string
    /// value `"beginning_of_postcombat_main"`,
    /// `src/mtg_bot/effects/predicates.py:135`).
    ///
    /// 5 EoE cards unblocked: Savior of the Small, Cautious Survivor,
    /// Defiant Survivor, House Cartographer, Shrewd Storyteller.
    BeginningOfPostcombatMain,
    /// **MMM Pin 2 (11th Rust slice) — DiscardsCard.**
    /// "Whenever <player> discards a card" — fired by every primitive
    /// that moves a card from a player's hand to the graveyard via
    /// "discard" semantics (CR 701.8). Distinct from destroy / exile
    /// / sacrifice: only "discard" actions emit this event.
    ///
    /// **Single-subject shape (player subject).** `subject_id =
    /// PLAYER_SUBJECT_OFFSET + discarding_player_id`. Mirrors Python's
    /// `Event.DISCARDS_CARD` whose `EVENT_SUBJECTS` entry is
    /// `("player",)` — the discarding player, not the discarded card.
    /// `Predicate::ControlledByOpponent` ("an opponent discards")
    /// resolves the sentinel directly; `Predicate::Always` ignores it.
    ///
    /// **Dispatch sites (Rust).**
    ///   * `Effect::DiscardCard` AnyPlayer branch (Mind Rot — target
    ///     opponent discards): dispatch on `defender` (the discarding
    ///     player) AFTER the card is moved to graveyard.
    ///   * `Effect::DiscardSelf` (Irreverent Gremlin — source's
    ///     controller discards): dispatch on `controller` after EACH
    ///     card moves to graveyard within the loop (one event per
    ///     discarded card, matching CR 701.8a "the player discards
    ///     each card" semantics).
    ///   * `Effect::EachOpponentDiscards` (Let's Play a Game — each
    ///     opponent discards): dispatch on `opponent_idx` after EACH
    ///     card moves to graveyard.
    /// Cleanup-step over-hand-limit discard and ward-discard riders
    /// are NOT yet wired (no EoE card in the pool exercises them
    /// through `DISCARDS_CARD`; they land in a future slice when a
    /// concrete need surfaces).
    ///
    /// Mirrors Python's `Event.DISCARDS_CARD` (string value
    /// `"discards_card"`, `src/mtg_bot/effects/predicates.py`).
    ///
    /// **EoE cards unblocked.** Entropic Battlecruiser 1+ tier
    /// ("Whenever an opponent discards a card, they lose 3 life").
    /// Full end-to-end firing also requires the `discards_card`
    /// bridge arm (closed here) and the `controlled_by_opponent`
    /// predicate (MMM Pin 3, also closed here).
    DiscardsCard,
}

/// Sentinel offset used by phase-step dispatch to pass a **player
/// subject** through the `subject_id: u32` API without colliding with
/// real game-object IDs.
///
/// Phase-step events (`BeginningOfEndStep`, `BeginningOfUpkeep`,
/// `BeginningOfCombat`) carry a player as their subject (the active
/// player whose phase it is), mirroring Python's `dispatch_event(state,
/// Event.BEGINNING_OF_END_STEP, player=state.active_player())` call.
/// Game-object IDs start from zero and climb monotonically, so we
/// reserve the top of the u32 range as a player-ID namespace:
/// `subject_id = PLAYER_SUBJECT_OFFSET + player_index` (where
/// `player_index` is 0 or 1 for the two-player engine). Object IDs
/// never approach 2^31 in any realistic game, so the namespaces are
/// effectively disjoint.
///
/// `ControlledByYou` in `Predicate::matches` detects the sentinel
/// range and resolves the player index directly rather than walking
/// the game-object zones.
pub(crate) const PLAYER_SUBJECT_OFFSET: u32 = 0x8000_0000;

/// Predicate algebra over the trigger subject (the GameObject the
/// event happened to) and source (the GameObject carrying the
/// ability). First-slice closed-enum minimal surface; mirrors a
/// strict subset of Python's predicate algebra at
/// `src/mtg_bot/effects/predicates.py`.
///
/// **Evaluation contract.** `matches(state, source_id, subject_id)`
/// returns `true` when the predicate is satisfied for this firing.
/// `Always` always matches; `SourceIsSelf` checks that the source
/// IS the subject (canonical "when ~ enters" shape);
/// `SourceIsCreature` checks the source's printing carries the
/// "Creature" card_type; `Not(p)` inverts; `And(a, b)` is logical
/// AND with short-circuit.
#[derive(Clone, Debug)]
pub enum Predicate {
    /// Always matches. Equivalent to no filter.
    Always,
    /// Matches when the source object IS the subject of the event
    /// (canonical "this creature enters" / "when ~ enters" shape).
    SourceIsSelf,
    /// Matches when the source object's printing has "Creature" in
    /// its card_types. Building block for shapes like "when ANOTHER
    /// creature enters — your creature has this ability".
    SourceIsCreature,
    /// **PPP slice — Source(IsTapped()) predicate.** Mirrors Python's
    /// `Source(inner=IsTapped())` (the "if this creature is tapped"
    /// idiom; ``effects/object_predicates.py:253``). The 5 "survivor"
    /// BEGINNING_OF_POSTCOMBAT_MAIN cards all carry this as part of
    /// their predicate composition. Pins the tapped-state check to
    /// the source permanent rather than the (player) event subject.
    SourceIsTapped,
    /// Matches when the **subject's** printing has "Land" in its
    /// card_types. Mirrors Python's `Subject(None, IsLand())` —
    /// the canonical "a land enters" filter used by landfall-shape
    /// triggers (`cards/parser/triggers.py:603-1266`).
    SubjectIsLand,
    /// Matches when the **subject's** printing has "Artifact" in its
    /// card_types. Mirrors Python's `Subject(None, IsArtifact())` —
    /// "an artifact enters" filter for artifact-enters triggers.
    SubjectIsArtifact,
    /// Matches when the **subject's** printing has "Creature" in its
    /// card_types. Mirrors Python's `Subject(None, IsCreature())` —
    /// the canonical "a creature enters" filter for
    /// Panharmonicon-shape triggers. Subject-anchored sibling of
    /// `SourceIsCreature`.
    SubjectIsCreature,
    /// Matches when the **subject's** printing has "Enchantment" in its
    /// card_types. Mirrors Python's `Subject(None, IsEnchantment())` —
    /// the canonical "an enchantment enters" filter for Duskmourn /
    /// EoE "Eerie" ability-word triggers ("Eerie — Whenever an
    /// enchantment you control enters…"). Rooms (which print as
    /// `Enchantment — Room`) carry `"Enchantment"` in their
    /// `card_types` so a Room entering ALSO satisfies this predicate —
    /// matching Python's `IsEnchantment.test` semantics and the
    /// printed Eerie wording.
    SubjectIsEnchantment,
    /// Matches when the **subject** is controlled by the source's
    /// controller. Mirrors Python's `Controlled(YOU)` — the
    /// canonical "under your control" qualifier on ETB triggers
    /// like "whenever a creature enters under your control" and
    /// "whenever a land enters under your control".
    ///
    /// **Slot field (12th Rust slice — Closure A).** Optional
    /// payload mirroring Python's ``Controlled(by, subject:
    /// str | None)`` slot. ``None`` (the back-compat default and
    /// the slot=None Python case) means "the event's only declared
    /// subject" — used by ENTERS / DIES / ATTACKS single-subject
    /// events. ``Some("controller")`` is the slot-aware variant
    /// used by multi-subject events (IS_CAST's ``("spell",
    /// "controller")`` / PLAYS_LAND's ``("land", "controller")``).
    /// The Rust ``matches`` arm is INVARIANT to the slot value for
    /// the currently-dispatched events: subject_id passed to
    /// ``dispatch_event`` always resolves through
    /// ``find_object_controller`` to the same controller id either
    /// way (the spell's controller == the casting player == the
    /// 'controller' slot value). The slot is stored for round-trip
    /// fidelity + Python-bridge symmetry; future events with a
    /// non-actor controller slot would need a slot-driven dispatch
    /// API. 4 EoE cards unblocked (Cosmogrand Zenith, Sunstar
    /// Lightsmith, Illvoi Operative, Station Monitor — all carry
    /// ``Controlled(YOU, subject="controller")`` on IS_CAST).
    ControlledByYou { slot: Option<String> },
    /// **MMM Pin 3 (11th Rust slice) — ControlledByOpponent.** Matches
    /// when the **subject** is controlled by an OPPONENT of the
    /// source's controller. Dual of `ControlledByYou`. Mirrors
    /// Python's `Controlled(by=OPPONENT, subject=None)` — the
    /// canonical "an opponent discards" / "an opponent's creature
    /// dies" qualifier for observer-opponent triggers.
    ///
    /// **Semantics.** Same lookup shape as `ControlledByYou` (resolve
    /// subject's controller, resolve source's controller, compare),
    /// but compares for INEQUALITY rather than equality. Handles the
    /// `PLAYER_SUBJECT_OFFSET` sentinel range identically — phase-step
    /// and DISCARDS_CARD subjects whose `subject_id` encodes a player
    /// index (rather than a game object) resolve directly to the
    /// player index without a zone walk.
    ///
    /// **Missing-id failure mode.** Either id missing → returns
    /// `false` (loud-but-safe; matches `ControlledByYou`'s shape).
    /// Mid-dispatch source eviction or a subject that has already
    /// left the battlefield can both produce missing ids; treating
    /// them as "no match" mirrors Python's
    /// `_resolve_controller_pair` returning `None` on either side.
    ///
    /// **EoE cards unblocked.** Entropic Battlecruiser 1+ tier
    /// (DISCARDS_CARD + ControlledByOpponent + TargetPlayerLosesLife).
    /// Broader "observer-opponent" trigger shapes — opponent-side
    /// DIES / ETBS / ATTACKS — all share this predicate variant.
    ///
    /// **Slot field (12th Rust slice — Closure A).** Symmetric with
    /// ``ControlledByYou``'s slot: optional payload mirroring
    /// Python's ``Controlled(OPPONENT, subject: str | None)``.
    /// Slot=None is back-compat (single-subject DISCARDS_CARD); a
    /// non-None slot covers multi-subject opponent-controlled cases
    /// (PLAYS_LAND's ``Controlled(OPPONENT, subject="controller")``).
    ControlledByOpponent { slot: Option<String> },
    /// Matches when the **subject's** printing carries the given
    /// subtype string. Mirrors Python's `Subject(None, HasSubtype(s))`
    /// — the canonical Kavu/Sliver/Equipment-enters predicate from
    /// `src/mtg_bot/cards/parser/triggers.py:1198`. Case-sensitive
    /// Title Case match against the subject's
    /// `CardPrinting.subtypes` Vec.
    ///
    /// **Effective-subtypes scope.** Python's `HasSubtype` reads
    /// through `effective_subtypes` (CR 613 layer-3/4 — Conspiracy
    /// granting "Sliver", Magus / Dryad layering on lands, etc.).
    /// The Rust mirror reads the printing's static `subtypes`
    /// directly — there's no Rust effective-subtypes layer yet,
    /// and none of the existing card-type predicates
    /// (`SubjectIsLand` / `SubjectIsCreature` / etc.) compute
    /// effective types either. The same upgrade-when-layers-land
    /// note applies symmetrically. EoE pool has no Conspiracy-shape
    /// subtype-granting effects today, so the direct read is
    /// observably equivalent.
    SubjectHasSubtype(String),
    /// **Pain for All closure (Aura Phase 3 / S4 triggered shape) —
    /// IsHost.** Matches when the subject IS the permanent the source
    /// is attached to (i.e. `source.attached_to == subject_id`).
    /// Mirrors Python's `IsHost(subject: str | None = None)`
    /// (`src/mtg_bot/effects/predicates.py:439-467`).
    ///
    /// **Use case.** The canonical Aura/Equipment relational trigger
    /// predicate: "Whenever enchanted creature is dealt damage",
    /// "Whenever equipped creature attacks", etc. The *source* is the
    /// Aura/Equipment; the firing *subject* is its host (the
    /// enchanted/equipped permanent).
    ///
    /// **Match semantics (mirrors Python's `IsHost.matches`).**
    /// Walks zones for `source_id`, reads `attached_to`. If
    /// `attached_to` is `None` (source has lost its host or never
    /// attached) → returns `false` (Python parity:
    /// `if host_id is None: return False`). Otherwise returns
    /// `attached_to == subject_id`.
    ///
    /// **Honest-stop today.** With Rust's no-`attached_to` Aura
    /// resolution reality (see `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`
    /// §2), every Aura source has `attached_to = None` at predicate
    /// time and this returns false on every DEALT_DAMAGE dispatch.
    /// Python's same-case `IsHost.matches` likewise returns false
    /// when `source.attached_to is None` — byte-equal honest-stop.
    /// Once the broader attach substrate lands, the predicate flips
    /// live without further changes.
    ///
    /// **Optional `subject` slot.** Mirrors Python's
    /// `IsHost(subject: str | None)` for multi-subject events where
    /// the subject must be selected by name (Equipment's
    /// `IsHost(subject="attacker")` for the equipped-creature-attacks
    /// surface). For DEALT_DAMAGE (Pain for All) it's `None` and Rust
    /// reads the dispatched `subject_id` directly. The slot is stored
    /// for round-trip fidelity; the Rust dispatcher passes a single
    /// `subject_id` to `matches` regardless of slot today.
    IsHost { subject: Option<String> },
    /// Matches when the **subject's** effective power is ≤ ``n``.
    /// Mirrors Python's ``HasPowerAtMost(n)`` from
    /// ``src/mtg_bot/effects/object_predicates.py:427``.
    /// Used in Vicious Clown / Irreverent Gremlin's ETB trigger:
    /// "Whenever another creature you control with power 2 or less
    /// enters, …" — predicate shape
    /// ``And(And(SubjectIsCreature, HasPowerAtMost{n:2}), ControlledByYou) AND NOT SourceIsSelf``.
    ///
    /// Non-creature subjects (no printed power and no UEOT creature
    /// animation) return ``false`` — mirrors Python's
    /// ``effective_power`` exception-catch → false path.
    HasPowerAtMost { n: i32 },
    /// Logical negation.
    Not(Box<Predicate>),
    /// Logical AND (short-circuits on the left operand's `false`).
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR (short-circuits on the left operand's `true`).
    ///
    /// Mirrors Python's `Or(left, right)` from
    /// `src/mtg_bot/effects/predicates.py:300`. Unlocks predicate
    /// shapes like Weftstalker Ardent / Haliya Guided by Light's
    /// "whenever a creature or artifact you control enters"
    /// (`(SubjectIsCreature OR SubjectIsArtifact) AND ControlledByYou
    /// AND NOT SourceIsSelf`), where Slice 4's `And`-only algebra
    /// couldn't express the disjunction.
    Or(Box<Predicate>, Box<Predicate>),
    /// **BBB downstream complement — 7th Rust slice.**
    /// Wraps a `StatePredicate` and binds it to the firing source's
    /// controller.  Mirrors Python's `StateBound(inner: StatePredicate)`
    /// (`effects/predicates.py:392-406`): evaluates
    /// `inner.test(state, source.controller_id)` at check time.
    ///
    /// This is the Tier-1 TriggerPredicate entry point for the
    /// Tier-2 `StatePredicate` sub-kinds (`VoidActive`,
    /// `AnyPlayerControlsNoCreatures`).  The bridge translates:
    ///   `"state_bound"` outer + `"void_active"` / `"any_player_controls_no_creatures"` inner.
    ///
    /// EoE cards unlocked: all Void-gated triggers (VoidActive inner)
    /// and Sothera, the Supervoid (AnyPlayerControlsNoCreatures inner).
    StateBound(StatePredicate),
}

impl Predicate {
    /// Evaluate the predicate against the firing context.
    ///
    /// `source_id` is the GameObject carrying the ability;
    /// `subject_id` is the GameObject the event happened to (the
    /// creature that just entered the battlefield, for Enters).
    /// Both ids are u32-keyed object ids resolved at dispatch time.
    pub(crate) fn matches(
        &self,
        state: &crate::state::GameState,
        source_id: u32,
        subject_id: u32,
    ) -> bool {
        match self {
            Predicate::Always => true,
            Predicate::SourceIsSelf => {
                // VVV Pin 1 closure 2026-05-20 — Survivor family
                // player-subject tautology mirroring Python's
                // IsSelf.matches at src/mtg_bot/effects/predicates.py
                // :428-436. Phase-step events (e.g.
                // BEGINNING_OF_POSTCOMBAT_MAIN) dispatch with
                // ``PLAYER_SUBJECT_OFFSET + active_player`` as the
                // subject_id sentinel rather than a permanent's id, so
                // the bare ``source_id == subject_id`` always
                // evaluates False for those events. Python's
                // ``IsSelf.matches`` handles this with the
                // ``not hasattr(obj, "printing")`` branch (Players
                // expose ``id`` but no ``printing``), returning True
                // — the dispatcher already iterates one source at a
                // time, so IsSelf is structurally tautological on
                // player-subject events. Mirror the sibling
                // ControlledByYou arm at cards.rs:4758-4807 which
                // already recognises the sentinel range.
                if subject_id >= PLAYER_SUBJECT_OFFSET {
                    true
                } else {
                    source_id == subject_id
                }
            }
            Predicate::SourceIsCreature => {
                source_has_card_type(state, source_id, "Creature")
            }
            Predicate::SourceIsTapped => {
                // **PPP slice — Source(IsTapped()) predicate.** Walk
                // the battlefield zones on both players for an object
                // matching `source_id`; return `obj.tapped`. Objects
                // off the battlefield (the source has changed zones
                // between dispatch and predicate check) → `false`,
                // mirrors Python's `IsTapped.test` which reads
                // `getattr(obj, "tapped", False)` — non-permanent
                // objects have no `tapped` attribute and the helper
                // returns False.
                source_is_tapped(state, source_id)
            }
            Predicate::SubjectIsLand => {
                subject_has_card_type(state, subject_id, "Land")
            }
            Predicate::SubjectIsArtifact => {
                subject_has_card_type(state, subject_id, "Artifact")
            }
            Predicate::SubjectIsCreature => {
                subject_has_card_type(state, subject_id, "Creature")
            }
            Predicate::SubjectIsEnchantment => {
                subject_has_card_type(state, subject_id, "Enchantment")
            }
            Predicate::ControlledByYou { slot: _ } => {
                // Subject's controller == source's controller. Mirrors
                // Python's `Controlled(by=YOU)` on a permanent subject:
                // ``obj.controller_id == source.controller_id``.
                // Each id is resolved by walking the four tracked zones
                // on both players; the helper returns `None` when the
                // id isn't present (the same loud-but-safe shape as
                // `find_source_printing_id`).
                //
                // **Player-subject path (round-14 phase-step events).**
                // Phase-step dispatch passes `PLAYER_SUBJECT_OFFSET +
                // active_player as u32` as the subject_id so we can
                // distinguish player subjects from game-object subjects
                // without a separate predicate variant. When subject_id
                // is in the sentinel range we resolve the player index
                // directly rather than walking game-object zones —
                // mirrors Python's `Controlled(YOU)` on a `Player`
                // subject: ``player.id == source.controller_id``.
                let source_controller =
                    find_object_controller(state, source_id);
                let subject_player_index: Option<u8> =
                    if subject_id >= PLAYER_SUBJECT_OFFSET {
                        let idx = subject_id - PLAYER_SUBJECT_OFFSET;
                        // Bounds-guard: only two players in this engine.
                        if (idx as usize) < state.players.len() {
                            Some(idx as u8)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                if let Some(player_idx) = subject_player_index {
                    // Player-subject path: source.controller == active_player.
                    match source_controller {
                        Some(sc) => sc == player_idx,
                        None => false,
                    }
                } else {
                    // Game-object subject path (normal case).
                    let subject_controller =
                        find_object_controller(state, subject_id);
                    match (source_controller, subject_controller) {
                        (Some(sc), Some(jc)) => sc == jc,
                        // Either id missing — never match (e.g. the
                        // subject has already left the battlefield, or
                        // the source was evicted mid-dispatch).
                        _ => false,
                    }
                }
            }
            Predicate::ControlledByOpponent { slot: _ } => {
                // **MMM Pin 3 — ControlledByOpponent.** Inverse of
                // ControlledByYou. Subject's controller != source's
                // controller. Same PLAYER_SUBJECT_OFFSET sentinel
                // handling as the sibling arm — phase-step /
                // DISCARDS_CARD player subjects resolve their player
                // index directly from the sentinel range. Missing
                // either id → `false` (loud-but-safe, mirrors
                // ControlledByYou's behavior).
                let source_controller =
                    find_object_controller(state, source_id);
                let subject_player_index: Option<u8> =
                    if subject_id >= PLAYER_SUBJECT_OFFSET {
                        let idx = subject_id - PLAYER_SUBJECT_OFFSET;
                        if (idx as usize) < state.players.len() {
                            Some(idx as u8)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                if let Some(player_idx) = subject_player_index {
                    // Player-subject path: source.controller != player_idx.
                    match source_controller {
                        Some(sc) => sc != player_idx,
                        None => false,
                    }
                } else {
                    // Game-object subject path (normal case).
                    let subject_controller =
                        find_object_controller(state, subject_id);
                    match (source_controller, subject_controller) {
                        (Some(sc), Some(jc)) => sc != jc,
                        _ => false,
                    }
                }
            }
            Predicate::SubjectHasSubtype(subtype) => {
                subject_has_subtype(state, subject_id, subtype)
            }
            Predicate::IsHost { subject: _ } => {
                // **Pain for All closure (Aura Phase 3 / S4 triggered
                // shape) — IsHost match.** Walk zones for the source
                // permanent; read `attached_to`. Mirrors Python's
                // `IsHost.matches` (`predicates.py:460-467`):
                //   host_id = getattr(source, "attached_to", None)
                //   if host_id is None: return False
                //   return getattr(obj, "id", None) == host_id
                //
                // Subject-slot rationale: Python carries an optional
                // `subject` name for multi-subject events (Equipment's
                // `IsHost(subject="attacker")` for attacks triggers).
                // The dispatcher passes a single `subject_id` to the
                // match function — for DEALT_DAMAGE (Pain for All's
                // damage-echo trigger), it IS the host candidate
                // directly, so the slot is informational only here.
                //
                // **Honest-stop today.** With Rust's no-`attached_to`
                // Aura resolve reality (every Aura source has
                // `attached_to = None` at predicate-check time —
                // docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md §2),
                // `find_object_attached_to` returns `None` and this
                // returns `false` for every Aura on every dispatch.
                // Python's `IsHost.matches` takes the same false path
                // when `attached_to is None` (`predicates.py:464-466`)
                // — byte-equal honest-stop until the attach substrate
                // lands.
                match find_object_attached_to(state, source_id) {
                    Some(host_id) => host_id == subject_id,
                    None => false,
                }
            }
            Predicate::HasPowerAtMost { n } => {
                // Mirrors Python's HasPowerAtMost.test: read effective
                // power via the crate-visible helper (which replicates
                // the creature-check + UEOT-animation logic of the public
                // `effective_power` getter). Returns false for
                // non-creature subjects and absent ids — identical to
                // Python's exception-catch → false path.
                match crate::state_ops_combat::predicate_subject_effective_power(
                    state,
                    subject_id,
                ) {
                    Some(power) => power <= *n,
                    None => false,
                }
            }
            Predicate::Not(inner) => !inner.matches(state, source_id, subject_id),
            Predicate::And(a, b) => {
                a.matches(state, source_id, subject_id)
                    && b.matches(state, source_id, subject_id)
            }
            Predicate::Or(a, b) => {
                a.matches(state, source_id, subject_id)
                    || b.matches(state, source_id, subject_id)
            }
            Predicate::StateBound(inner) => {
                // Resolve the source-controller's player id and delegate
                // to the inner StatePredicate's `test` method.  Mirrors
                // Python's `StateBound.matches(state, subjects, source)`:
                //   `return self.inner.test(state, source.controller_id)`
                // (`effects/predicates.py:405`).
                // Both current StatePredicate kinds (VoidActive,
                // AnyPlayerControlsNoCreatures) are viewer-independent, so
                // the controller id is passed for forward-compat only.
                let viewer_id = find_object_controller(state, source_id)
                    .unwrap_or(state.active_player);
                inner.test(state, viewer_id)
            }
        }
    }
}

/// Look up an object by id across all four tracked zones of both
/// players and return whether its printing's `card_types` contains
/// the given type. Returns `false` when the object id isn't present
/// or its printing is unregistered (the loud-but-safe shape used by
/// `Predicate::SourceIsCreature`'s original inline lookup).
fn source_has_card_type(
    state: &crate::state::GameState,
    obj_id: u32,
    card_type: &str,
) -> bool {
    object_has_card_type(state, obj_id, card_type)
}

/// **PPP slice — SourceIsTapped helper.** Walks battlefield zones on
/// both players for an object matching `obj_id`; returns
/// `obj.tapped`. Off-battlefield sources (zone-changed between
/// dispatch and predicate check) → `false`, mirrors Python's
/// `IsTapped.test` returning False for non-permanent / missing-
/// attribute objects.
fn source_is_tapped(
    state: &crate::state::GameState,
    obj_id: u32,
) -> bool {
    for player in &state.players {
        for obj in player.battlefield.iter() {
            if obj.id == obj_id {
                return obj.tapped;
            }
        }
    }
    false
}

/// Subject-anchored sibling of `source_has_card_type`. Same shape;
/// the distinct entry point keeps the predicate-arm bodies tidy and
/// the intent self-documenting at the call site.
fn subject_has_card_type(
    state: &crate::state::GameState,
    obj_id: u32,
    card_type: &str,
) -> bool {
    object_has_card_type(state, obj_id, card_type)
}

/// Shared body for `source_has_card_type` / `subject_has_card_type`.
///
/// **G7 S5a:** also walks `state.stack_objects` (the
/// spell-stack-source-zone store, zone=ZONE_STACK). A spell being cast
/// is moved hand → `stack_objects` BEFORE the CAST event dispatches
/// (mirroring Python's `card.zone = Zone.STACK` at `actions.py:1751`
/// preceding the `StackItem` append at `:1759`). Python's CAST
/// predicate dispatch evaluates `subject_is_creature` etc. against the
/// live `card` reference (`dispatch_event(..., spell=card)`,
/// `actions.py:1795`) which is off-hand on the stack — so the Rust
/// subject lookup MUST reach the stack-resident object or the
/// `SubjectIsCreature`/`SubjectIsLand`/`SubjectIsArtifact` CAST
/// triggers (and `SourceIsCreature` respond-to-cast) silently stop
/// firing (`feedback_silent_noop_pipelines`; design §5.2-risk-3 —
/// fixed IN S5a, not deferred).
fn object_has_card_type(
    state: &crate::state::GameState,
    obj_id: u32,
    card_type: &str,
) -> bool {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    if let Some(printing) =
                        state.registry.lookup(obj.printing_id)
                    {
                        return printing
                            .card_types
                            .iter()
                            .any(|t| t == card_type);
                    }
                    return false;
                }
            }
        }
    }
    // G7 S5a 5th zone: the spell-stack-source-zone store.
    for obj in state.stack_objects.iter() {
        if obj.id == obj_id {
            if let Some(printing) = state.registry.lookup(obj.printing_id) {
                return printing.card_types.iter().any(|t| t == card_type);
            }
            return false;
        }
    }
    false
}

/// Look up an object by id across all four tracked zones of both
/// players and return whether its printing's `subtypes` contains the
/// given subtype string. Case-sensitive Title Case match (mirrors the
/// stored shape from the registry-dict parser). Returns `false` when
/// the object id isn't present or its printing is unregistered (the
/// same loud-but-safe shape as `object_has_card_type`).
///
/// **Effective subtypes scope.** Same caveat as `Predicate::SubjectHasSubtype`'s
/// doc comment: reads the printing's static `subtypes` directly,
/// not an `effective_subtypes` layer (which doesn't exist on the
/// Rust side yet — see Python's `mtg_bot.game.effective_subtypes`
/// for the CR 613 layer-3/4 reading the rules-correct path uses).
/// Symmetric with how `object_has_card_type` reads `card_types`
/// directly.
fn subject_has_subtype(
    state: &crate::state::GameState,
    obj_id: u32,
    subtype: &str,
) -> bool {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    if let Some(printing) =
                        state.registry.lookup(obj.printing_id)
                    {
                        return printing
                            .subtypes
                            .iter()
                            .any(|s| s == subtype);
                    }
                    return false;
                }
            }
        }
    }
    // G7 S5a 5th zone: the spell-stack-source-zone store (a cast
    // spell with a `SubjectHasSubtype` CAST trigger lives here at
    // dispatch time — same rationale as `object_has_card_type`).
    for obj in state.stack_objects.iter() {
        if obj.id == obj_id {
            if let Some(printing) = state.registry.lookup(obj.printing_id) {
                return printing.subtypes.iter().any(|s| s == subtype);
            }
            return false;
        }
    }
    false
}

/// Look up an object by id across all four tracked zones of both
/// players and return its controller id. Returns `None` when the
/// object isn't found in any of those zones (already moved off-bf
/// in a way the dispatch caller didn't expect, or an id mismatch).
///
/// **G7 S5a:** also walks `state.stack_objects` so `ControlledByYou`
/// on a cast-spell subject (an on-stack spell, e.g. a
/// `subject_is_creature AND controlled_by_you` CAST observer) resolves
/// the subject's controller. Python's `_cast_spell` sets
/// `card.controller_id = player_id` (`actions.py:1752`) before the
/// CAST dispatch, so the by-reference `card` subject carries the
/// caster's controller; the Rust stack-store object's `controller`
/// was set identically at cast (byte-mirror). Without this the
/// `ControlledByYou` arm returns `None`→`false` and the trigger
/// silently stops firing (design §5.2-risk-3).
fn find_object_controller(
    state: &crate::state::GameState,
    obj_id: u32,
) -> Option<u8> {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    return Some(obj.controller);
                }
            }
        }
    }
    // G7 S5a 5th zone: the spell-stack-source-zone store.
    for obj in state.stack_objects.iter() {
        if obj.id == obj_id {
            return Some(obj.controller);
        }
    }
    None
}

/// **Pain for All closure — `find_object_attached_to` helper.**
/// Walk every tracked zone for ``obj_id`` and return its
/// ``attached_to`` value (the id of the permanent it's attached
/// to as an Aura / Equipment, or ``None`` if unattached / missing).
/// Mirrors the per-player-zone walk shape of
/// :fn:`find_object_controller`, including the G7-S5a
/// `stack_objects` fallthrough so a stack-resident spell whose
/// ETB trigger fires (the Aura cast → ETB dispatch case) can still
/// be located. Used by ``Predicate::IsHost`` to read the source's
/// attach pointer at predicate-check time.
fn find_object_attached_to(
    state: &crate::state::GameState,
    obj_id: u32,
) -> Option<u32> {
    for player in &state.players {
        for zone in [
            &player.hand,
            &player.library,
            &player.battlefield,
            &player.graveyard,
            &player.exile,
        ] {
            for obj in zone.iter() {
                if obj.id == obj_id {
                    return obj.attached_to;
                }
            }
        }
    }
    for obj in state.stack_objects.iter() {
        if obj.id == obj_id {
            return obj.attached_to;
        }
    }
    None
}

/// A triggered ability on a `CardPrinting`. Mirrors Python's
/// `TriggeredAbility` dataclass at
/// `src/mtg_bot/effects/base.py:680`.
///
/// **Slice 1 scope.** Only the predicate-style path is wired
/// (`event` is required and never `None` on the Rust side; Python's
/// optional shape is the legacy string-key path Slice 1 doesn't
/// mirror). `requires_void`, `tier_threshold`, `room_half` from
/// Python are omitted here — defaults of "always fire" suffice for
/// the smoke probe; they land alongside their gates in Slice 3+.
#[derive(Clone, Debug)]
pub struct TriggeredAbility {
    /// The event family this ability listens for (today only
    /// `Event::Enters`).
    pub event: Event,
    /// Filter over the firing context (source object, subject
    /// object, game state). Slice 1 minimum: `Always`,
    /// `SourceIsSelf`, `SourceIsCreature`, with `And` / `Not`
    /// compositions.
    pub predicate: Predicate,
    /// The payload effect to push as a `StackItem::TriggeredAbility`
    /// when this ability fires. Same closed `Effect` vocabulary the
    /// spell-resolve path consumes; the trigger resolver hands the
    /// effect to `apply_effect` exactly the way spell resolution
    /// does.
    pub effect: Effect,
    /// "This ability triggers only once each turn." When true the
    /// dispatcher records `(source_id, ability_index)` in
    /// `GameState.triggered_this_turn` on push and suppresses repeat
    /// fires within the same turn. Reset at cleanup step. Slice 1
    /// wires the ledger but no SLICE-1 test exercises the gate (the
    /// canonical IsSelf-enters trigger fires at most once per
    /// permanent's life anyway).
    pub once_per_turn: bool,
}

// --- Activated-ability substrate (CE20 Slice 1) ------------------
//
// Mirrors Python's `src/mtg_bot/effects/base.py::ActivationCost` +
// `ActivatedAbility`. **First-slice minimal field set** — covers the
// load-bearing closure surface (a mana-cost-paying activated ability
// like Bear Caster's "{1}{G}, T: Draw a card"). Other cost shapes
// from the Python class (sacrifice_self, exile_self, pay_life,
// requires_*, etc.) are deferred to follow-up slices that bring
// their dispatch primitives along — listing them here without a
// dispatch site would let registry-dict callers stamp activations
// that silently no-op the cost portion. The Slice 1 surface:
//
//   * `mana: ManaCost` — payable from the controller's mana pool.
//   * `tap_self: bool` — implicit `{T}` cost; sets `obj.tapped = true`
//     before the effect dispatches.
//   * `is_mana_ability: bool` — when `true` the effect resolves
//     immediately (CR 605.3a — mana abilities bypass the stack); when
//     `false` the ability pushes a `StackItem::TriggeredAbility`-shaped
//     stack slot whose payload effect resolves on the next
//     `resolve_top_of_stack` (mirrors Python's `_activate_ability`
//     stack-push at `actions.py:1928-2002`).
//
// **Deferred to follow-up slices** (each lands with its dispatch
// counterpart, lockstep):
// - Sacrifice-as-cost (`sacrifice_self`, `sacrifice_another_*`,
//   `sacrifice_a_land`, `sacrifice_food`, `sacrifice_a_desert`).
// - Counter-as-cost (`remove_plus_one_counter_from_your_creature`)
//   and planeswalker loyalty-as-cost (`pay_loyalty`).
// - Tap-an-untapped-permanent-you-control (Gene Pollinator shape).
// - Activation-gate predicates (`requires_*` Bonders' Enclave shape).
// - Sorcery-speed gate (`sorcery_speed`) — already implicit in the
//   Slice 1 emission position INSIDE the sorcery-timing block; the
//   per-ability field is a more granular over-ride.
// - Spacecraft tier / Rooms `room_half` / activate_only_once gates.
//
/// Cost composition for an `ActivatedAbility`. First-slice minimal —
/// payable from the controller's mana pool, plus an optional implicit
/// `{T}` tap-self.
#[derive(Clone, Debug, Default)]
pub struct ActivationCost {
    /// Mana portion of the cost. An all-zero `ManaCost` (the
    /// `Default::default()` shape) is a "free" cost — payable with no
    /// pool requirement. The cost-pay primitive consumes mana from the
    /// controller's pool via the same `pay_mana_cost` shape spell casts
    /// use (`state_ops_cast.rs::pay_mana_cost`).
    pub mana: ManaCost,
    /// Implicit `{T}` portion of the cost. When `true`, the
    /// dispatcher sets `obj.tapped = true` on the source after the
    /// affordability check passes; an already-tapped source rejects
    /// the activation. Mirrors Python's
    /// `ActivationCost.tap_self`.
    pub tap_self: bool,
    /// "Sacrifice this permanent" portion of the cost. When `true`,
    /// the dispatcher moves the source from its current battlefield
    /// to its controller's graveyard AFTER mana + tap costs commit
    /// successfully. Mirrors Python's
    /// `ActivationCost.sacrifice_self`. 35 EoE printings carry this
    /// shape — `Honored Knight-Captain`, `Illvoi Galeblade`,
    /// `Cryogen Relic`, every Lander token, etc.
    ///
    /// **Composes with `tap_self`.** `tap_self=true` AND
    /// `sacrifice_self=true` is legal — the source taps before it
    /// dies (observable for triggers that fire on tap state if any).
    /// The dispatch commits in order: mana → tap → sacrifice.
    ///
    /// **Self-only Slice 1 scope.** Targeted sacrifice shapes
    /// (`sacrifice_another_creature`, `sacrifice_a_land`,
    /// `sacrifice_food`, etc.) need target fan-out at
    /// `legal_decisions` emission and a new optional
    /// `target_creature_id` key on the ActivateAbility dict; deferred
    /// to a follow-up slice once Equipment-equip ships the target
    /// substrate.
    ///
    /// **LTB void-flag** (CE13 / I29 closure): the dispatch routes
    /// the sacrificed object through `mark_left_battlefield_if_nonland`
    /// so the existing void-tracking flag stamps correctly. No new
    /// wiring needed — sacrifice rides on the same LTB primitive
    /// every other bf-exit uses.
    pub sacrifice_self: bool,
    /// **CE20 follow-up — Equipment-equip Slice 4.** When `true`, the
    /// activated ability requires the activator to pick a target
    /// creature they control (CR 702.6c — equip targets a creature
    /// you control). Flows into emission as a fan-out (one
    /// `ActivateAbility` dict per legal target creature on the
    /// activator's bf) and into apply as a required
    /// `target_creature_id` key on the action dict.
    ///
    /// Default `false` keeps every pre-Slice-4 caller working: the
    /// dict shape stays {kind, source_id, ability_index} for
    /// non-target activations, and the emission walk emits one
    /// dict-per-source as before.
    ///
    /// **Eligibility** (mirrors Python's `EquipTargetSelector` at
    /// `effects/generic/attach.py` + CR 702.6c):
    ///   - target must be a creature on the activator's battlefield,
    ///   - target must NOT be the source itself (an Equipment can't
    ///     equip itself),
    ///   - target's summoning sickness is NOT a barrier (CR 702.6
    ///     gates the activator's source — equip is a tap-cost-free
    ///     ability so the target's sickness is irrelevant).
    ///
    /// **Retarget shape** (CR 702.6c): activating equip on an
    /// already-attached Equipment with a new target re-stamps
    /// `attached_to`. The dispatch fires the underlying
    /// `Effect::EquipToTargetCreatureYouControl` arm which reads the
    /// payload `target_id` and stamps it (overwriting any prior
    /// `attached_to`). Mana cost paid both times — the activation is
    /// a valid game action even when its observable mutation is a
    /// no-op (idempotent retarget to the same creature is legal).
    ///
    /// **EoE printings affected**: 7 (Auxiliary Boosters, Squire's
    /// Lightblade, Atomic Microsizer, Illvoi Light Jammer, Hylderblade,
    /// Meltstrider's Gear, The Dominion Bracelet).
    pub requires_target_creature_you_control: bool,
    /// **CE20 follow-up Slice 6 — targeted-sacrifice variants.**
    /// "Sacrifice another creature or artifact" — the activator picks
    /// any non-source creature OR artifact on their bf. Mirrors
    /// Python's `ActivationCost.sacrifice_another_creature_or_artifact`
    /// at `effects/base.py:358`. Default `false`.
    ///
    /// Rides on the Slice 4 target-pick substrate: emission fans out
    /// one `ActivateAbility` dict per legal sac target (eligible target
    /// = on activator's bf, NOT the source itself, has `Creature` OR
    /// `Artifact` card_type). Apply arm validates the chosen target,
    /// then commits in Phase 5b — moves the chosen target to its
    /// controller's graveyard (in practice the activator's gy because
    /// you can only choose permanents you control). Same field-reset +
    /// LTB-void-flag + LEAVES queue plumbing as the Slice 2
    /// `sacrifice_self` commit.
    ///
    /// **Composes with `tap_self` + `mana`.** Order:
    /// mana → tap → sac-another → effect.
    pub sacrifice_another_creature_or_artifact: bool,
    /// **CE20 follow-up Slice 6 — targeted-sacrifice variants.**
    /// "Sacrifice a land" — the activator picks any land they
    /// control (Dust Bowl shape). Mirrors Python's
    /// `ActivationCost.sacrifice_a_land` at `effects/base.py:388`.
    /// Default `false`.
    ///
    /// Eligibility: target on activator's bf with `Land` card_type.
    /// The printed text reads "Sacrifice a land" (no "another"
    /// qualifier), so when the source IS a land the source itself is
    /// eligible as the sacrifice target. Python mirrors this lenient
    /// reading.
    pub sacrifice_a_land: bool,
    /// **CE20 follow-up Slice 6 — targeted-sacrifice variants.**
    /// "Sacrifice another artifact" — strict artifact-only filter
    /// (Memorial Vault shape). Mirrors Python's
    /// `ActivationCost.sacrifice_another_artifact` at
    /// `effects/base.py:364`. Default `false`.
    ///
    /// Eligibility: target on activator's bf with `Artifact` card_type,
    /// NOT the source itself ("another" qualifier).
    pub sacrifice_another_artifact: bool,
    /// **CE20 follow-up Slice 7 — subtype-filtered sacrifice variants.**
    /// "Sacrifice a Food" — the activator picks any permanent they
    /// control whose `subtypes` contains "Food". Mirrors Python's
    /// `ActivationCost.sacrifice_food` at `effects/base.py`. Default
    /// `false`. Rides on the subtypes substrate (`CardPrinting.subtypes`)
    /// landed at commit `03aebefb` / `dd15d7ed`.
    ///
    /// Eligibility: target on activator's bf with `subtypes.contains(
    /// "Food")`. Title Case subtype string per CR 205.3 / Scryfall
    /// convention. The source itself IS eligible iff its own subtypes
    /// include "Food" — "a Food" carries no "another" qualifier (matches
    /// the `sacrifice_a_land` lenient reading from Slice 6).
    ///
    /// **EoE printing affected**: 1 (Ragost, Deft Gastronaut —
    /// `{1}, {T}, Sacrifice a Food: Ragost deals 3 damage to each
    /// opponent.`). Ragost's static ability ALSO confers the "Food"
    /// subtype on every artifact the activator controls, but the static
    /// subtype-conferral layer is out of scope; the test fixture uses
    /// pre-stamped Food subtypes on synthetic printings.
    ///
    /// **Composes with `tap_self` + `mana`.** Order:
    /// mana → tap → sac-food → effect.
    pub sacrifice_food: bool,
    /// **CE20 follow-up Slice 7 — subtype-filtered sacrifice variants.**
    /// "Sacrifice a Desert" — the activator picks any permanent they
    /// control whose `subtypes` contains "Desert". Mirrors Python's
    /// `ActivationCost.sacrifice_a_desert` at `effects/base.py`. Default
    /// `false`. Rides on the subtypes substrate.
    ///
    /// Eligibility: target on activator's bf with `subtypes.contains(
    /// "Desert")`. The printed text reads "Sacrifice a Desert" with no
    /// "another" qualifier, so when the source IS a Desert the source
    /// itself is eligible (mirrors `sacrifice_a_land` / `sacrifice_food`
    /// lenient reading).
    ///
    /// **EoE printings affected**: 0 in EoE today; the substrate ships
    /// without a real-card probe — tests use synthetic Desert printings.
    /// Closes the symmetric pair with `sacrifice_food` and unblocks any
    /// future Desert-cost printing without a structural change.
    pub sacrifice_a_desert: bool,
    /// **CE20 follow-up Slice 5 — Planeswalker loyalty (Slice A).**
    /// Signed loyalty cost (CR 606). Positive values are PAY costs
    /// (the walker LOSES N loyalty counters); negative values are
    /// ADD costs (the walker GAINS |N| loyalty counters). Defaults
    /// to `0` (no loyalty cost). Convention: a printed "+2" ability
    /// stamps `loyalty: -2`; a printed "-3" ability stamps
    /// `loyalty: 3`; a printed "0" ability stamps `loyalty: 0`.
    ///
    /// **Emission gate**: positive `loyalty` values gate
    /// `legal_decisions` emission on the source's `loyalty_counters`
    /// — the walker must have at least `cost.loyalty` counters to
    /// afford the activation. Negative values are always affordable
    /// (a +N ability can fire at any loyalty value).
    ///
    /// **Apply commit** (Phase 5d): `obj.loyalty_counters -=
    /// cost.loyalty`. Signed subtraction handles both directions
    /// uniformly. SBA pass at state_ops_sba.rs catches the
    /// `loyalty_counters <= 0` case and moves the walker to its
    /// controller's graveyard (CR 704.5i).
    ///
    /// **EoE printing affected**: 1 (Tezzeret, Cruel Captain — sole
    /// walker in EoE).
    ///
    /// **Out of scope** of Slice A: combat-damage redirect to
    /// planeswalkers (CR 119.5 / Slice B), once-per-turn-per-walker
    /// ledger (CR 606.3 / Slice C).
    pub loyalty: i32,
}

/// An activated ability on a `CardPrinting`. Mirrors a strict subset
/// of Python's `ActivatedAbility` dataclass at
/// `src/mtg_bot/effects/base.py:447`.
///
/// **Slice 1 scope.** The dispatcher reads `cost` (pays it), `effect`
/// (applies it), and `is_mana_ability` (short-circuits the stack push
/// when true). The richer Python fields — `sorcery_speed`,
/// `cost_reductions`, `room_half`, `tier_threshold`,
/// `activate_only_once` — are deferred to follow-up slices that bring
/// their gates along.
#[derive(Clone, Debug)]
pub struct ActivatedAbility {
    /// Cost to activate the ability. The dispatcher checks
    /// affordability (mana-pool side) then commits the cost (tap +
    /// mana drain) before pushing/applying the effect.
    pub cost: ActivationCost,
    /// Effect to fire when the ability resolves. For
    /// `is_mana_ability=true` abilities the effect fires immediately
    /// on activation (CR 605.3a); for `is_mana_ability=false`
    /// abilities the effect is wrapped in a `StackItem` and fires
    /// on the next `resolve_top_of_stack` pop.
    pub effect: Effect,
    /// `true` when this is a mana ability (CR 605.1 — produces mana
    /// AND doesn't target). Mana abilities bypass the stack and
    /// resolve their effect immediately on activation. For Slice 1
    /// the load-bearing case is `false` (Bear Caster's draw goes
    /// through the stack); a future slice will bridge the legacy
    /// `mana_produced` field into a synthesized `is_mana_ability=true`
    /// ActivatedAbility, retiring the separate `mana_produced` path.
    pub is_mana_ability: bool,
    /// **CE20 follow-up — Spacecraft station-charge.** Minimum
    /// `GameObject.charge_counters` the source must have for this
    /// ability to be activatable. Defaults to `0` (no tier gate);
    /// Spacecraft printings stamp this to their printed tier value
    /// (e.g. Lumen-Class Frigate's level-3 station ability stamps
    /// `tier_threshold: 3`).
    ///
    /// **Usability gate, not a cost.** The source's
    /// `charge_counters` is COMPARED against this value at
    /// `legal_decisions` emission AND at apply-time precheck; the
    /// activation does NOT decrement charge counters (sub-tier
    /// activations are simply unavailable). The cost-pay charge-
    /// counter shape (`Pay N charge counters: ...`) has no EoE
    /// printings and is deferred — search the EoE pool's oracle
    /// text for `Pay [0-9]+ charge counter` returns zero hits.
    ///
    /// Mirrors Python's `ActivatedAbility.tier_threshold: int = 0`
    /// at `effects/base.py:447`.
    pub tier_threshold: i32,
}

#[derive(Clone, Debug)]
pub struct CardPrinting {
    pub name: Arc<str>,
    pub mana_cost: ManaCost,
    pub effects: Vec<Effect>,
    /// Printed power. `None` for non-creatures (instants, sorceries,
    /// non-creature lands, etc.). Stored as `i16` to leave headroom
    /// for negatives produced by anthem-like effects (the printed
    /// value itself is always non-negative).
    pub power: Option<i16>,
    /// Printed toughness. `None` for non-creatures. Same rationale
    /// as `power` for the `i16` choice.
    pub toughness: Option<i16>,
    /// Mana the printing produces when tapped, expressed as a list
    /// of single-letter color symbols (W/U/B/R/G/C). `None` for
    /// printings that don't produce mana on tap. A single-color
    /// land like Forest stores `Some(vec!["G"])`. Vec-of-strings
    /// (rather than a single String) leaves room for "tap for {G}{G}"
    /// or future multi-pip producers without re-shaping the field.
    /// "Tap for one of any color" / hybrid producers will need a
    /// richer schema and are deferred.
    pub mana_produced: Option<Vec<String>>,
    /// MTG card types: "Creature", "Land", "Instant", "Sorcery",
    /// "Artifact", "Enchantment", "Planeswalker", "Battle". Most
    /// cards have one type; multi-typed cards (artifact creatures,
    /// enchantment creatures) carry multiple. Empty Vec is allowed
    /// for first-slice synthetic test cards but real cards always
    /// have at least one.
    pub card_types: Vec<String>,
    /// MTG card subtypes (the post-em-dash portion of the type line):
    /// creature subtypes ("Bear", "Goblin", "Kavu"), basic-land subtypes
    /// ("Forest", "Island", "Plains"), artifact subtypes ("Equipment",
    /// "Food"), enchantment subtypes ("Aura", "Saga"), etc.
    ///
    /// **Mixed semantics by design.** MTG conflates basic-land subtypes
    /// (Forest, Mountain, ...) with creature/artifact/enchantment
    /// subtypes into a single comprehensive-rules "subtype" namespace
    /// (CR 205.3). The Rust mirror follows Python's
    /// `CardPrinting.subtypes: tuple[str, ...]` — one flat field
    /// holding every subtype regardless of which card-type slot it
    /// belongs to. Consumers that need land-type-only semantics gate
    /// on `card_types.contains("Land")` first.
    ///
    /// Empty Vec when the type line has no subtypes (vanilla
    /// instants/sorceries, generic artifacts). Stored exactly as
    /// printed — Title Case, ASCII (the EoE pool has no
    /// non-ASCII-named subtypes). Case-sensitive comparison via
    /// `subtypes.iter().any(|s| s == "Forest")` is the canonical
    /// consumer pattern.
    ///
    /// Mirrors Python's `CardPrinting.subtypes: tuple[str, ...]` at
    /// `src/mtg_bot/cards/base.py:501`. Registry-dict input under the
    /// `subtypes` key; missing key parses to empty Vec (the
    /// `#[serde(default)]`-equivalent shape — keeps every pre-substrate
    /// registry-dict caller working without modification).
    pub subtypes: Vec<String>,
    /// MTG card supertypes: "Basic", "Legendary", "Snow", "World",
    /// "Token". Empty Vec when the type line has no supertype prefix
    /// (most cards). Stored as Title Case strings exactly as printed
    /// — case-sensitive comparison via
    /// `supertypes.iter().any(|s| s == "Legendary")` is the canonical
    /// consumer pattern.
    ///
    /// **Use cases.** Distinguishes basic vs non-basic lands
    /// (`is_basic_land` = `card_types contains "Land"` AND
    /// `supertypes contains "Basic"`); legendary-rule processing
    /// (`is_legendary` = `supertypes contains "Legendary"`); tutor
    /// filters that scope by supertype (`legendary_creature` filter
    /// in `Effect::SearchLibraryForCard`).
    ///
    /// Mirrors Python's `CardPrinting.supertypes: tuple[str, ...]`
    /// at `src/mtg_bot/cards/base.py:499`. Registry-dict input under
    /// the `supertypes` key; missing key parses to empty Vec
    /// (back-compat — every pre-supertypes registry dict implicitly
    /// omits the key, matches the `subtypes` substrate's parse shape).
    ///
    /// **Round 30 substrate.** Added to unblock the
    /// `legendary_creature` filter on `Effect::SearchLibraryForCard`
    /// (round-21 bridge slice's deferred filter list — see
    /// state_ops_resolve.rs `SearchLibraryForCard` arm). Pre-Round-30
    /// the field didn't exist; the legacy `is_basic_land` proxy in
    /// `SearchLibraryForBasicLandTapped`'s arm comment is now
    /// historical (it still uses `card_types contains "Land"` because
    /// every EoE basic land has card_types=["Land"] and no compound
    /// non-basic land routes through that arm — keeping the proxy
    /// avoids a behavior change for any in-flight fixture without
    /// `supertypes` set).
    pub supertypes: Vec<String>,
    /// Static keyword abilities printed on the card. Stored as
    /// lowercase strings ("flying", "vigilance", "trample",
    /// "first strike", "double strike", "deathtouch", "lifelink",
    /// "haste", "reach", "menace", "defender"). Keywords with
    /// associated values (protection from X, ward N, hexproof
    /// from X) need a richer schema and are deferred. Empty Vec
    /// when the card has no static keywords.
    pub keywords: Vec<String>,
    /// "When ~ enters the battlefield" effects. Fire on any
    /// permanent landing on the battlefield (creature, artifact,
    /// enchantment, planeswalker). Empty Vec = no ETB. First-slice
    /// scope: the effects fire IMMEDIATELY on entry, no separate
    /// stack item — real MTG triggers them onto the stack and
    /// resolves in priority order; that timing nuance is deferred.
    ///
    /// **Legacy in flight.** Slice 1 of the ETB-trigger closure
    /// (`docs/RUST_ETB_TRIGGER_AUDIT.md`) introduces
    /// `triggered_abilities` (predicate-path) — the audit's "fix
    /// 4 Python-incompatible properties of the inline `etb_effects`
    /// hatch" plan. The cast-and-resolve site still consults
    /// `etb_effects` as a back-compat shim so existing test fixtures
    /// keep working; new fixtures should use `triggered_abilities`.
    /// Slice 2 fans out the predicate path to every ETB-stamping
    /// site, retiring the inline path.
    pub etb_effects: Vec<Effect>,
    /// **ETB-trigger substrate (Slice 1).** Predicate-style
    /// triggered abilities on this printing. Empty Vec = no
    /// predicate-style triggers (back-compat default — every
    /// pre-Slice-1 registry dict implicitly has this empty).
    /// Future slices add more dispatch sites; today only the
    /// cast-and-resolve path consumes this via
    /// `state_ops_resolve::dispatch_event(Event::Enters, ...)`.
    ///
    /// Mirrors Python's `CardPrinting.triggered_abilities` tuple at
    /// `src/mtg_bot/cards/base.py`. Registry dict input under the
    /// `triggered_abilities` key is parsed by
    /// `parse_triggered_ability` (see this file). Missing key in
    /// the dict = empty Vec (the `#[serde(default)]`-equivalent
    /// shape — keeps every existing registry-dict caller working
    /// without modification).
    pub triggered_abilities: Vec<TriggeredAbility>,
    /// **Activated-ability substrate (CE20 Slice 1).** Cost-paying
    /// activated abilities on this printing. Empty Vec = no activated
    /// abilities of this shape (back-compat default — every
    /// pre-CE20-Slice-1 registry dict implicitly has this empty).
    ///
    /// Mirrors Python's `CardPrinting.activated_abilities` tuple at
    /// `src/mtg_bot/cards/base.py`.
    ///
    /// **Post-bridge invariant (CE20 follow-up).** The legacy
    /// `mana_produced` field is now SYNTHESIZED into a prepended
    /// `ActivatedAbility { is_mana_ability: true, cost: tap_self+free,
    /// effect: AddManaToPool(...) }` entry at parse time (see
    /// `apply_mana_produced_bridge`). Both the registry dict-input
    /// path (`parse_printing`) and the synthetic-state direct-construction
    /// path (`state.rs`) route through the shared helper. The legacy
    /// emission block in `state_ops_legal.rs` and the legacy
    /// `is_legacy_mana_produced` branch in `state_ops_actions.rs` are
    /// gone — single unified path through `activated_abilities`.
    ///
    /// Missing key in the registry dict = empty Vec (no special
    /// serde tag needed; the parser's `dict.get_item` returns
    /// `None`-of-PyAny which the same `match` we use for
    /// `triggered_abilities` maps to `Vec::new()`).
    pub activated_abilities: Vec<ActivatedAbility>,
    /// **CE20 Slice 5 (Walker loyalty — `starting_loyalty` ETB seeding).**
    /// Printed starting loyalty for Planeswalker printings (CR 606.5b
    /// — "as a planeswalker enters the battlefield, its controller
    /// puts a number of loyalty counters on it equal to the number
    /// printed in its lower right corner"). Non-walker printings
    /// carry `0` (the default — no observable effect since the
    /// loyalty-counter slot is only read by the walker SBA pass and
    /// the loyalty cost gate, both gated on `card_types.contains(
    /// "Planeswalker")`).
    ///
    /// **Stamped at ETB time** (every bf-entry call site) — see
    /// `crate::state::seed_starting_loyalty_for_etb`. For non-walker
    /// printings the helper is a no-op; for walker printings whose
    /// `loyalty_counters` is 0 (the freshly-cast shape) the helper
    /// stamps the printed value.
    ///
    /// **EoE pool**: 1 walker printing (Tezzeret, Cruel Captain).
    /// Tezzeret's printed loyalty is 4.
    ///
    /// Mirrors Python's `CardPrinting.starting_loyalty: int = 0`.
    pub starting_loyalty: i32,
    /// **warp_cost substrate (Phase 1, round-63).** EoE "warp"
    /// alternative cast cost. `Some(cost)` when the printing has a
    /// pure fixed-`ManaCost` warp (Red Tiger Mechan {1}{R}, Bygone
    /// Colossus {3}); `None` otherwise (back-compat default — every
    /// pre-Phase-1 registry dict omits the `warp_cost` key, which the
    /// parser maps to `None`, identical to the `mana_produced` /
    /// `power` Option-parse shape). Phase 1 carries ONLY the pure
    /// fixed-cost shape; `warp_additional_life_cost` /
    /// `warp_castable_from_graveyard` are Phase 2 (the bridge filter
    /// `_printing_is_bridge_pure_warp_creature` positively rejects
    /// them so they never reach this field in a Phase-1 deckspec).
    ///
    /// Mirrors Python's `CardPrinting.warp_cost: ManaCost | None`
    /// at `src/mtg_bot/cards/base.py:515`. The cast pipeline
    /// (`state_ops_cast.rs::cast_creature_warp`) pays THIS instead of
    /// `mana_cost` when `Action::CastSpell{use_warp:true}`; the
    /// legal emitter (`state_ops_legal.rs`) offers the
    /// `use_warp=true` CastSpell variant iff this is `Some`.
    pub warp_cost: Option<ManaCost>,
    /// **warp_cost substrate (Phase 2).** Timeline Culler's "Warp—{B},
    /// Pay 2 life" life rider: the additional life paid as part of the
    /// warp cast (`0` = no rider, the back-compat default — every
    /// pre-Phase-2 registry dict omits the `warp_additional_life_cost`
    /// key, which the parser maps to `0`, identical to the Phase-1
    /// `warp_cost` Option-parse shape). Mirrors Python's
    /// `CardPrinting.warp_additional_life_cost: int`
    /// (`src/mtg_bot/cards/base.py:521`). Debited from the caster's
    /// life inside `cast_creature_warp` (hand AND the new graveyard
    /// fall-through arm, `state_ops_cast.rs`); the legal emitter gates
    /// the warp variant on `player.life > this` (mirrors Python
    /// `legal.py:269` / `:399` / `:461`).
    pub warp_additional_life_cost: u8,
    /// **warp_cost substrate (Phase 2).** Timeline Culler's "You may
    /// cast this card from your graveyard using its warp ability":
    /// `true` iff the card may be warp-cast from the graveyard zone
    /// (not only the hand). `false` = hand-only (the back-compat
    /// default — every pre-Phase-2 registry dict omits the
    /// `warp_castable_from_graveyard` key). Mirrors Python's
    /// `CardPrinting.warp_castable_from_graveyard: bool`
    /// (`src/mtg_bot/cards/base.py:525`). When `true`, the legal
    /// emitter offers a `use_warp=true` CastSpell for the card while
    /// it is in the graveyard (mirrors the separate graveyard loop at
    /// `legal.py:447-476`); the cast pipeline routes it through the
    /// graveyard fall-through in `cast_creature_warp`.
    pub warp_castable_from_graveyard: bool,
    /// **legal_decisions mana-awareness Slice 3 — cast-time
    /// additional-cost flags.** Mirror Python's
    /// `CardPrinting.additional_cost_sacrifice_creature_or_artifact`
    /// / `..._creature_or_enchantment` / `..._choose_your_creature`
    /// (the 3 flags `legal._additional_costs_payable`,
    /// `src/mtg_bot/legal.py:117-165`, gates every CastSpell emission
    /// on — CR 601.2f/601.2h: a player can't begin to cast a spell
    /// unless they can pay its additional non-mana costs). `false` =
    /// no such additional cost (the back-compat default — every
    /// pre-Slice-3 registry dict omits these keys; same
    /// `match get_item { Some => extract, None => false }` parse
    /// shape as `warp_castable_from_graveyard`).
    ///
    /// **Live-surface incidence is ZERO (verify-first,
    /// evidence-bound).** All 4 EoE printings carrying any of these
    /// flags — Embrace Oblivion, Scrounge for Eternity (Sorcery,
    /// sac creature-or-artifact), Final Vengeance (Sorcery, sac
    /// creature-or-enchantment), Close Encounter (Instant,
    /// choose-your-creature) — are bridge-DECLINED by BOTH
    /// `_printing_is_bridge_simple_spell` and
    /// `_printing_is_bridge_simple_creature`, so NO card on any of
    /// the 6 live Rust CastSpell emission axes carries one. The
    /// design doc (`RUST_LEGAL_DECISIONS_MANA_AWARENESS_DESIGN.md`
    /// §5) predicted exactly this ("the smallest — lowest
    /// live-surface incidence"). The Slice-3 gate is therefore
    /// parity-NEUTRAL by construction (the predicate is `true` for
    /// every live-surface card — never prunes, never narrower than
    /// Python) AND the faithful Python mirror + the single named
    /// extension point if an additional-cost card ever becomes
    /// bridge-admitted. Documented in `docs/SEARCH_LIMITATIONS.md`.
    pub additional_cost_sacrifice_creature_or_artifact: bool,
    /// Final Vengeance shape — "sacrifice a creature or enchantment"
    /// (an Aura is an Enchantment subtype; same controller filter).
    /// See `additional_cost_sacrifice_creature_or_artifact`.
    pub additional_cost_sacrifice_creature_or_enchantment: bool,
    /// Close Encounter shape — "choose a creature you control or a
    /// warped creature card you own in exile" (either source
    /// satisfies the cost). See
    /// `additional_cost_sacrifice_creature_or_artifact`.
    pub additional_cost_choose_your_creature: bool,
    /// **Aura-attachment substrate — Phase 1 STEP 2 (round-78,
    /// 2026-05-20).** `true` iff this printing is an Aura per
    /// `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §3 Phase 1 sub-shape S1
    /// (static P/T-or-keyword Aura, no triggers, no control swap, no
    /// replacement). `false` (the back-compat default — every
    /// pre-Phase-1 registry dict omits the `is_aura` key, which the
    /// parser maps to `false`) for every non-Aura printing AND for
    /// every Aura that the §2 loader predicate
    /// `_printing_is_bridge_simple_aura` declines (so the field is
    /// load-bearing only when paired with a bridge-admitted Aura,
    /// which is currently NONE — STEP 7 wire-up + a later phase's
    /// real-Aura admit is required before any Aura reaches the
    /// registry as `is_aura=true`).
    ///
    /// Mirrors Python's `CardPrinting.is_aura` derivation (`card_types
    /// contains "Enchantment" AND subtypes contains "Aura"` per
    /// `cards/base.py:1060`) — except on the Rust side the value comes
    /// straight from the explicit `is_aura` key the Python bridge
    /// emits (`build_rust_state` at `cross_engine.py:10339`), NOT
    /// derived from subtypes. The bridge ensures the boolean matches
    /// the Python derivation; downstream Rust consumers gate on this
    /// field rather than re-running the derivation.
    ///
    /// **Inert until STEPs 3-6 land.** This slice extends the parser
    /// to STORE the value — no downstream Rust consumer reads it yet
    /// (cast-time enchant-target decision, resolution attach arm,
    /// CR-704.5n SBA, bidir bridge are all later Phase-1 steps —
    /// honest-stopped per the plan). The field round-trips through
    /// the bridge faithfully so STEPs 3-6 can read it without further
    /// substrate changes.
    pub is_aura: bool,
    /// **Aura-attachment substrate — Phase 1 STEP 2.** Which permanent
    /// type this Aura enchants: `"creature"` (Frantic Strength,
    /// Shardmage's Rescue — Phase 1 carriers) or `"artifact"`
    /// (Hardlight Containment, admitted later). Defaults to
    /// `"creature"` (mirrors Python's `CardPrinting.aura_enchant_type`
    /// default at `cards/base.py:507`) when the registry dict omits
    /// the `aura_enchant_type` key OR when the printing is not an
    /// Aura — the field is ignored unless `is_aura` is `true`. Phase 1
    /// restricts to single-type Auras (modal "artifact or creature"
    /// shapes are declined at the loader predicate per the plan §2).
    pub aura_enchant_type: String,
    /// **Aura-attachment substrate — Phase 1 STEP 2.** `true` iff the
    /// printed text is "Enchant <type> YOU CONTROL" (Shardmage's
    /// Rescue). `false` is the default (back-compat — every pre-STEP-2
    /// registry dict omits the key) and the printed shape for Frantic
    /// Strength / Hardlight Containment / etc. The cast-time
    /// enchant-target picker (Phase 1 STEP 3, not in this slice) will
    /// filter legal hosts on this; STEP 2 carries the field through the
    /// bridge ONLY.
    pub aura_yours_only: bool,
    /// **Aura-attachment substrate — Phase 1 STEP 2.** The layer-7
    /// power bonus the attached host receives while this Aura is in
    /// play. `0` (back-compat default — every pre-STEP-2 registry
    /// dict omits the key) for keyword-only Auras and for every
    /// non-Aura. Phase 1 admits ONLY static, layered grants; dynamic
    /// or counter-based bonuses are out of scope. Mirrors Python's
    /// `CardPrinting.equipment_bonuses[0]` (the same `(power, tough,
    /// keywords)` tuple shape Equipment uses — the Python parser
    /// reuses `equipment_bonuses` for Auras and Equipment).
    ///
    /// **Wire shape.** The bridge emits a single dict
    /// `aura_equipment_bonuses = {"power_bonus": int, "toughness_bonus":
    /// int, "granted_keywords": [str, ...]}` (NOT three separate keys
    /// — see `build_rust_state` at `cross_engine.py:10384`). The parse
    /// arm reads the three sub-keys out of that dict.
    pub aura_power_bonus: i16,
    /// **Aura-attachment substrate — Phase 1 STEP 2.** Layer-7
    /// toughness bonus; same parse + default + wire-shape semantics as
    /// `aura_power_bonus`. Mirrors Python's
    /// `CardPrinting.equipment_bonuses[1]`.
    pub aura_toughness_bonus: i16,
    /// **Aura-attachment substrate — Phase 1 STEP 2.** Layer-6
    /// keyword grants the attached host receives while this Aura is in
    /// play (`["trample", "flying", ...]`). Empty Vec (back-compat
    /// default — every pre-STEP-2 registry dict omits the key) for
    /// P/T-only Auras and for every non-Aura.
    ///
    /// **Storage shape.** Lowercase keyword names with spaces
    /// converted to underscores ("first_strike", "double_strike") —
    /// matches the Python bridge's emission at
    /// `cross_engine.py:10387` (`kw.replace(" ", "_")`). The keyword
    /// vocabulary is restricted to `_SAFE_KEYWORDS` at the loader
    /// (`trajectory.py:_SAFE_KEYWORDS`) so the underscore-form is
    /// closed: "first_strike", "double_strike", "flying", "vigilance",
    /// "lifelink", "trample", "haste", "deathtouch", "menace",
    /// "defender", "indestructible", "reach". Mirrors Python's
    /// `CardPrinting.equipment_bonuses[2]` (a frozenset on the Python
    /// side; ordered Vec here for round-trip stability). Future Rust
    /// consumers that compare against the existing `keywords` field
    /// (which stores space-separated lowercase) must underscore-map
    /// before comparison.
    pub aura_granted_keywords: Vec<String>,
    /// Dynamic-P/T formula key (CR 613 layer 7b CDA — Characteristic
    /// Defining Ability). When `Some(key)`, this creature's base
    /// power and toughness are NOT read from `printing.power` /
    /// `printing.toughness` — instead the layer-7 reader computes
    /// the base from the formula keyed by `key` at query time.
    ///
    /// Mirrors Python's `CardPrinting.dynamic_pt_formula: str | None`
    /// at `src/mtg_bot/cards/base.py:837`. Python's
    /// `game._dynamic_base_pt` dispatches on the same key strings
    /// (`"harmonious_grovestrider"`, `"cosmogoyf"`, `"emissary_escort"`).
    /// Rust's `battlefield_creature_power` /
    /// `battlefield_creature_toughness` dispatch on this field
    /// during the layer-7b base-P/T resolution; the dispatch table
    /// must stay in lockstep with Python's `_dynamic_base_pt` per
    /// the "new code is right" SoT-flip discipline.
    ///
    /// **Round-78 dynamic-P/T-formula substrate.** Added to unblock
    /// Harmonious Grovestrider (subsystem inventory #3; 61 17lands
    /// decks). The substrate-extension closes the CardDef-bridge gap
    /// that the round-39 fix (`9c0d440f`) papered over by dropping
    /// dynamic-P/T creatures at the filter. Pre-substrate the field
    /// didn't exist; the registry-dict parser maps the absence of
    /// the `dynamic_pt_formula` key to `None` (the
    /// Option/serde-default pattern every back-compat-added field
    /// uses). When the key is present, the value must be a string
    /// matching one of Python's dispatch keys.
    pub dynamic_pt_formula: Option<String>,
    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** Layer-2 control-change static grant. `true`
    /// iff this Aura's printed text is "You control enchanted
    /// permanent" (Duskmourn's Domination, Tractor Beam's static
    /// half). The Phase-2 S2 resolution-attach path flips the host's
    /// `controller` to the Aura's `controller` and stashes the host's
    /// prior controller in `GameObject.controller_baseline` for revert
    /// on Aura leave.
    ///
    /// `false` is the back-compat default — every pre-S2 registry
    /// dict omits the `aura_grants_control` key. Mirrors Python's
    /// `CardPrinting.attached_grants_control` (`cards/base.py:616`).
    /// Ignored unless `is_aura == true`.
    pub aura_grants_control: bool,
    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** Layer-6 lose-all-abilities static grant. `true`
    /// iff this Aura's printed text contains "Enchanted permanent
    /// loses all abilities" (Duskmourn's Domination, Lignify). On
    /// legal attach, the host's `battlefield_has_keyword` reader
    /// returns `false` for any printed keyword. UEOT-granted
    /// keywords are an honest-stop — the Rust reader currently
    /// suppresses printed keywords only (the layer-6 ordering
    /// interleaving with UEOT grants that were applied BEFORE the
    /// lose-abilities effect is deferred per the plan).
    ///
    /// Mirrors Python's
    /// `CardPrinting.attached_grants_lose_all_abilities`
    /// (`cards/base.py:634`). `false` is the back-compat default;
    /// ignored unless `is_aura == true`.
    pub aura_grants_lose_abilities: bool,
    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** Layer-7b set-base-PT static grant. `Some((p,
    /// t))` overrides the host's printed base P/T while attached
    /// (Lignify "becomes a 0/4 Treefolk"). `None` is the back-compat
    /// default — every pre-S2 registry dict omits the
    /// `aura_set_base_pt` key. Mirrors Python's
    /// `CardPrinting.attached_sets_base_pt` (`cards/base.py:633`).
    ///
    /// **Honest-stop (Phase 2 S2 first deliverable).** The Rust
    /// reader fold for set-base-PT is NOT wired in this slice
    /// (no admitted S2 carrier today uses it — Duskmourn's
    /// Domination carries control-swap + lose-abilities only). The
    /// field round-trips through the bridge faithfully so a future
    /// S2.5 / Lignify-shape slice can light up the readers without
    /// further substrate changes. Ignored unless `is_aura == true`.
    pub aura_set_base_pt: Option<(i16, i16)>,
    /// **Aura-attachment substrate — Phase 2 S2 extension (Tractor
    /// Beam closure, 2026-05-20).** Layer-6-adjacent "doesn't untap
    /// during its controller's untap step" static rider. `true` iff
    /// this Aura's printed text contains "Enchanted permanent
    /// doesn't untap during its controller's untap step" (Tractor
    /// Beam, Stay Hidden Stay Silent). When `true` AND `is_aura`,
    /// the untap-step sweep (`untap_all_for_player_inline` in
    /// `state_ops_steps.rs`) skips any host the Aura is attached to,
    /// so the host stays tapped through its controller's untap step
    /// while the Aura is in play.
    ///
    /// `false` is the back-compat default — every pre-extension
    /// registry dict omits the `aura_prevents_host_untap` key.
    /// Mirrors Python's `CardPrinting.attached_prevents_untap`
    /// (`cards/base.py:604`). Ignored unless `is_aura == true`.
    pub aura_prevents_host_untap: bool,
    /// **Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    /// substrate landing, 2026-05-20).** CR 614 token-creation
    /// replacement rider. `true` iff this Aura's printed text contains
    /// the Moonlit Meditation literal "The first time you would create
    /// one or more tokens each turn, you may instead create that many
    /// tokens that are copies of enchanted permanent." When `true` AND
    /// `is_aura`, the controlling player's first token-creation each
    /// turn is intercepted: the requested tokens are instead created
    /// as copies of the Aura's `attached_to` host (Token supertype
    /// added). Mirrors Python's
    /// `CardPrinting.replaces_first_tokens_with_copies`
    /// (`cards/base.py:651`) and its consumer at
    /// `effects/generic/tokens.py:217-247` (the moonlit-replacement
    /// branch in `CreateToken.resolve`).
    ///
    /// **Substrate-landing scope (Phase 4 S3, 2026-05-20).** This
    /// slice lands the CardPrinting field + parse arm + getter +
    /// bridge round-trip + per-player `created_tokens_this_turn`
    /// state slot. The Rust `CreateToken` arm does NOT yet swap the
    /// token's `printing_id` to the host's printing — the Token-
    /// supertype injection on a host printing is a runtime registry
    /// mutation that doesn't exist on the Rust side today (tokens use
    /// statically-registered `printing_id`s; Python uses an
    /// in-memory `_dc.replace(printing, supertypes=...)` shim). The
    /// loader admit predicate
    /// (`_printing_is_bridge_simple_aura`) stays declined at the
    /// `replaces_first_tokens_with_copies` check pending that
    /// follow-up substrate. **Live-EoE admit delta this slice: +0
    /// cards / +0 decks.**
    ///
    /// `false` is the back-compat default — every pre-S3 registry
    /// dict omits the `aura_replaces_first_tokens_with_copies` key
    /// and the parsed CardPrinting is bit-equal to before this slice
    /// landed. Ignored unless `is_aura == true`.
    pub aura_replaces_first_tokens_with_copies: bool,
    /// **Static / layered abilities substrate — Phase 1 first-deliverable
    /// (round-79, 2026-05-20).** Anthem-shape statics carried by this
    /// printing. Each `StaticAnthem` declares a layer-6 keyword grant
    /// the source contributes to every battlefield creature matching the
    /// anthem's `scope` (under the source's controller). `Vec::new()`
    /// (the back-compat default — every pre-substrate registry dict
    /// omits the `static_anthems` key) for every non-anthem printing.
    ///
    /// Mirrors Python's `CardPrinting.static_abilities` tuple filtered
    /// to the anthem keyword-grant sub-shape (per
    /// `parser/statics.py:_parse_anthem_static_line` — one
    /// `StaticAbility` per granted keyword, each with a layer-6
    /// `ContinuousModifier(GrantKeyword(...))`).
    ///
    /// **Reader site.** `battlefield_has_keyword` walks both players'
    /// battlefields once per call: any source whose printing's
    /// `static_anthems` matches the target's controller AND scope
    /// contributes the granted keywords to the target's effective
    /// keyword set. The layer-6 fold is consistent with the existing
    /// `aura_grants_keyword_to` precedent (Aura Phase 1 STEP 4), the
    /// difference being the source is the static-bearer itself rather
    /// than an attached Aura.
    ///
    /// **Phase 1 scope teeth.** Only the unconditional anthem keyword-
    /// grant shape (Tannuk Steadfast Second's "Other creatures you
    /// control have haste."). Self-scope statics, conditional statics
    /// (Brightspear / Cloudsculpt / Gravblade), and P/T anthems are
    /// honest-stopped to later slices.
    pub static_anthems: Vec<StaticAnthem>,
    /// **Static / layered abilities substrate — Phase 2 (round-80,
    /// 2026-05-20).** Frenzied Baloth's "This spell can't be
    /// countered." rider — a per-printing flag the
    /// `Effect::CounterTargetSpell` resolver consults to skip
    /// targeting this spell. `false` is the back-compat default
    /// — every pre-Phase-2 registry dict omits the
    /// `spell_is_uncounterable` key. Mirrors Python's
    /// `CardPrinting.spell_is_uncounterable` (`cards/base.py:930`).
    pub spell_is_uncounterable: bool,
    /// **Static / layered abilities substrate — Phase 2 (round-80,
    /// 2026-05-20).** Frenzied Baloth's "Creature spells you control
    /// can't be countered." rider — a battlefield-static cast-side
    /// gate, encoded here as a per-printing bool (the simpler shape:
    /// every battlefield permanent under a controller is scanned at
    /// counter-resolve time, and if any one carries this flag the
    /// controller's creature spells on the stack are immune to
    /// counter effects). Mirrors Python's
    /// `StaticAbility.grants_creature_spells_uncounterable` field
    /// (`effects/base.py:577`) collapsed to a per-printing slot for
    /// Phase 2 — only ONE static-ability shape today (Frenzied
    /// Baloth's third line); the per-StaticAbility-tuple
    /// generalization is honest-stopped to a later slice.
    pub grants_creature_spells_uncounterable: bool,
    /// **Static / layered abilities substrate — Phase 3 (round-81,
    /// 2026-05-20).** SELF-scope conditional P/T + keyword grants
    /// carried by this printing. Each `StaticSelfBonus` declares a
    /// layer-7 P/T contribution AND/OR a layer-6 keyword grant
    /// applied to the source itself while the printed condition
    /// matches (CR 613). `Vec::new()` (the back-compat default —
    /// every pre-Phase-3 registry dict omits the
    /// `static_self_bonuses` key) for every non-Phase-3 printing.
    ///
    /// Mirrors Python's `CardPrinting.static_abilities` tuple
    /// filtered to the SELF-scope conditional sub-shape (per
    /// `parser/statics.py:_parse_static_ability_line` — one
    /// `StaticAbility` per layer-payload, all sharing the same
    /// SELF scope and conditional gate).
    ///
    /// **Reader sites.** `battlefield_creature_power` /
    /// `battlefield_creature_toughness` fold in the P/T
    /// contribution; `battlefield_has_keyword` folds in the
    /// keyword grants. Both sites short-circuit when the printing
    /// carries no entries (`is_empty()`), keeping every non-
    /// Phase-3 printing's read path byte-identical to pre-Phase-3.
    ///
    /// **Phase 3 scope teeth.** Only the SELF-scope shape this
    /// slice. Other scopes (`creatures_you_control`, slivers, etc.)
    /// remain in `static_anthems`. Mechan Shieldmate's
    /// `ControlledArtifactEnteredThisTurn` + `RemoveKeyword`
    /// payload, Rampaging Soulrager's
    /// `UnlockedDoorsAmongRoomsAtLeast`, and Wildfire
    /// Wickerfolk's `DeliriumActive` remain honest-stopped (each
    /// needs a Condition variant outside this slice's two-tag
    /// table or a `RemoveKeyword` payload the Rust reader does
    /// not yet fold).
    pub static_self_bonuses: Vec<StaticSelfBonus>,
}

/// **Static / layered abilities substrate — Phase 1 first-deliverable
/// (round-79).** One unconditional anthem-shape static ability: a
/// layer-6 keyword grant whose subject is `scope` (under the source's
/// controller). Mirrors Python's `StaticAbility(scope=...,
/// continuous_effect=ContinuousModifier(layer=6, payload=GrantKeyword(
/// kw)))` shape collapsed to the simpler anthem-keyword tuple — the
/// Rust side groups all keywords for one scope into one struct, the
/// reader site fans out at fold time.
#[derive(Clone, Debug)]
pub struct StaticAnthem {
    /// One of `"creatures_you_control"` /
    /// `"other_creatures_you_control"`. Other Python `StaticScope`
    /// values (`"self"`, `"slivers_you_control"`, etc.) are out of
    /// scope for this slice — the bridge predicate
    /// `_printing_is_bridge_simple_static_anthem` positively rejects
    /// them so they never reach the registry.
    pub scope: String,
    /// Lowercase underscore-form keyword names (e.g. `"haste"`,
    /// `"first_strike"`). Vocabulary restricted to
    /// `trajectory._SAFE_KEYWORDS` by the loader predicate.
    pub granted_keywords: Vec<String>,
}

/// **Static / layered abilities substrate — Phase 3 (round-81,
/// 2026-05-20).** One SELF-scope conditional bonus on a creature:
/// a layer-7 P/T contribution AND/OR a layer-6 keyword grant that
/// the source contributes to ITSELF (`scope=self` in Python's
/// StaticScope) while the printed condition matches. Mirrors
/// Python's `StaticAbility(scope=SELF, condition=<C>,
/// continuous_effect=ContinuousModifier(layer={6 or 7},
/// payload=<GrantKeyword(kw) or PowerToughnessBonus(p, t)>))`
/// shape collapsed into a per-(condition_tag, condition_args)
/// tuple — Gravblade Heavy's parser-emitted two StaticAbilities
/// (sharing condition; one P/T, one keyword) coalesce into one
/// `StaticSelfBonus` carrying both payloads.
///
/// **Reader sites.** `battlefield_creature_power` /
/// `battlefield_creature_toughness` (layer-7c additive fold) and
/// `battlefield_has_keyword` (layer-6 grant fold) consult
/// `static_self_bonuses` on the obj's own printing AFTER checking
/// the condition tag against the current game state. The fold is
/// per-object (not battlefield-walking like anthems): the bonus
/// fires only on the source itself.
#[derive(Clone, Debug)]
pub struct StaticSelfBonus {
    /// Identifier the reader-site evaluator dispatches on. Lockstep
    /// with the Python `_static_self_bonus_condition_from_tag`
    /// helper:
    ///   * `"control_permanent_is_artifact"` — viewer controls any
    ///     artifact permanent (Cloudsculpt Technician, Gravblade
    ///     Heavy).
    ///   * `"at_least_n_spells_cast_this_turn"` — viewer's
    ///     `spells_cast_this_turn >= n` (Brightspear Zealot, n=2).
    ///   * `"controlled_artifact_entered_this_turn"` — viewer's
    ///     `artifacts_entered_under_your_control_this_turn > 0`
    ///     (Mechan Shieldmate, round-82, 2026-05-20).
    pub condition_tag: String,
    /// Integer args for `condition_tag`. Empty for arg-less
    /// conditions; one i32 for the `n` arg of
    /// `at_least_n_spells_cast_this_turn`. Bridge serialization
    /// uses a `[int, ...]` list.
    pub condition_args: Vec<i32>,
    /// Layer-7c additive P/T contribution to the SOURCE while the
    /// condition matches (CR 613.3c).
    pub power_bonus: i16,
    pub toughness_bonus: i16,
    /// Layer-6 keyword grants to the SOURCE while the condition
    /// matches. Lowercase underscore-form names from
    /// `trajectory._SAFE_KEYWORDS` (e.g. `"deathtouch"`).
    pub granted_keywords: Vec<String>,
    /// **Round-82 (2026-05-20, Mechan Shieldmate closure).** Layer-6
    /// keyword REMOVALS from the SOURCE while the condition matches
    /// (Mechan Shieldmate's "this creature loses defender" rider).
    /// Lowercase underscore-form names from
    /// `trajectory._SAFE_KEYWORDS`. `Vec::new()` for every pre-
    /// round-82 entry — every previously-shipped carrier
    /// (Brightspear / Cloudsculpt / Gravblade) leaves this empty.
    pub removed_keywords: Vec<String>,
}

/// Plain Rust container for the printings table. Held behind an
/// `Arc` once a `GameState` snapshots the registry — at that point
/// it's immutable and clonable at refcount-bump cost from any
/// number of game-state copies.
#[derive(Clone, Debug, Default)]
pub struct RegistryData {
    pub printings: Vec<CardPrinting>,
}

impl RegistryData {
    pub fn lookup(&self, id: u32) -> Option<&CardPrinting> {
        self.printings.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.printings.len()
    }

    /// Round-55 token-identity bit. Mirrors Python's
    /// `CardPrinting.is_token` (`cards/base.py:1036`), which is
    /// `"Token" in self.supertypes`. Rust's `CardPrinting` already
    /// carries the Round-30 `supertypes: Vec<String>` substrate, and
    /// the cross-engine bridge round-trips a token printing's
    /// `("Token",)` supertype tuple verbatim through
    /// `build_rust_state`'s `printing_dict["supertypes"]` (so a token
    /// registered cross-engine is `is_token` on BOTH sides). No new
    /// per-printing field is needed — token-ness is a pure derivation
    /// of the existing supertypes Vec, exactly as on the Python side.
    ///
    /// An unknown printing id (registry miss) is treated as
    /// non-token, matching the fail-open convention used by
    /// `mark_left_battlefield_if_nonland` for the LTB void flag.
    /// Token-creation arms (`CreateToken` / `CreateTokenTapped` /
    /// `CreateTokenEachPlayer`) reference a `token_printing_id` that
    /// was registered with the `("Token",)` supertype tuple, so the
    /// freshly-minted GameObjects resolve as tokens here without any
    /// per-object flag.
    pub fn is_token(&self, id: u32) -> bool {
        self.lookup(id)
            .map(|p| p.supertypes.iter().any(|s| s == "Token"))
            .unwrap_or(false)
    }
}

// --- dict -> Rust ------------------------------------------------

fn require<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing required key: {key:?}")))
}

fn parse_mana_cost(dict: &Bound<'_, PyDict>) -> PyResult<ManaCost> {
    let mut cost = ManaCost::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj.extract()?;
        let value: u8 = value_obj.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "mana_cost[{key:?}] must be a non-negative integer (got: {e})"
            ))
        })?;
        match key.as_str() {
            "generic" | "C" => cost.generic = value,
            "W" => cost.white = value,
            "U" => cost.blue = value,
            "B" => cost.black = value,
            "R" => cost.red = value,
            "G" => cost.green = value,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown mana_cost key {other:?}; valid: generic, W, U, B, R, G"
                )))
            }
        }
    }
    Ok(cost)
}

fn parse_target_filter(dict: &Bound<'_, PyDict>) -> PyResult<TargetFilter> {
    let kind: String = require(dict, "kind")?.extract()?;
    match kind.as_str() {
        "any_player" => Ok(TargetFilter::AnyPlayer),
        "any_creature" => Ok(TargetFilter::AnyCreature),
        other => Err(PyValueError::new_err(format!(
            "unknown target filter kind {other:?}; valid: any_player, any_creature"
        ))),
    }
}

fn parse_effect(dict: &Bound<'_, PyDict>) -> PyResult<Effect> {
    let kind: String = require(dict, "kind")?.extract()?;
    match kind.as_str() {
        "deal_damage" => {
            let amount: i32 = require(dict, "amount")?.extract()?;
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("deal_damage.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::DealDamage { amount, target })
        }
        "draw_cards" => {
            let count: i32 = require(dict, "count")?.extract()?;
            Ok(Effect::DrawCards { count })
        }
        "gain_life" => {
            let amount: i32 = require(dict, "amount")?.extract()?;
            Ok(Effect::GainLife { amount })
        }
        "return_to_hand" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("return_to_hand.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::ReturnToHand { target })
        }
        "mill" => {
            let count: i32 = require(dict, "count")?.extract()?;
            Ok(Effect::Mill { count })
        }
        "mill_target_player" => {
            // Python's `MillTargetPlayer(amount: int = 1)` carries a
            // default amount. The Rust bridge uses `count` (mirrors the
            // `mill` arm's field name + the Mill / ScryN / Surveil /
            // DrawCards / DamageEachOpponent count-field convention)
            // and requires the field — missing `count` is a KeyError
            // at register time. Negative `count` surfaces at register
            // time (defense-in-depth; `apply_effect` re-checks).
            let count: i32 = require(dict, "count")?.extract()?;
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "mill_target_player: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::MillTargetPlayer { count })
        }
        "mill_target_player_equal_to_your_life_total" => {
            // BLOCKED-RUST-EFFECT slice (round 42) — mill an opponent
            // a number of cards equal to the source-controller's
            // current life total. Unit variant; no fields beyond
            // `kind` (the count is dynamic — read from state at
            // resolve time). Extra dict fields silently ignored for
            // round-trip dict-shape consistency (mirrors every other
            // unit-variant parse arm — round 30 / 31 / 33 / 34 / 35 /
            // 36 / 37 precedent). Mirrors Python's
            // `MillTargetPlayer(use_controller_life=True)`
            // (`draw_mill.py:151` — Space-Time Anomaly's instant
            // resolution).
            Ok(Effect::MillTargetPlayerEqualToYourLifeTotal)
        }
        "exile_target" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("exile_target.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::ExileTarget { target })
        }
        "tap_target" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("tap_target.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::TapTarget { target })
        }
        "untap_target" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("untap_target.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::UntapTarget { target })
        }
        "sacrifice" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("sacrifice.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::Sacrifice { target })
        }
        "add_counters" => {
            let count: i32 = require(dict, "count")?.extract()?;
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("add_counters.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::AddCounters { target, count })
        }
        "discard_card" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("discard_card.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::DiscardCard { target })
        }
        "scry_n" => {
            let count: i32 = require(dict, "count")?.extract()?;
            Ok(Effect::ScryN { count })
        }
        "destroy_target" => {
            let target_any = require(dict, "target")?;
            let target_dict = target_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("destroy_target.target must be a dict")
            })?;
            let target = parse_target_filter(target_dict)?;
            Ok(Effect::DestroyTarget { target })
        }
        "destroy_target_creature_then_draw_if_undamaged" => {
            // BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB.
            // Payload variant: `draw_amount` (cards the destroyed
            // creature's controller draws iff it carried no marked
            // damage). Missing `draw_amount` defaults to 0 (Python
            // parity: `draw_if_undamaged_amount: int = 0` — a 0 rider
            // is a legal "destroy with no draw" shape). Negative
            // `draw_amount` clamps to 0 at resolve (matching the
            // X-from-count zero short-circuit convention).
            let draw_amount: i32 = match dict.get_item("draw_amount")? {
                Some(v) => v.extract()?,
                None => 0,
            };
            Ok(Effect::DestroyTargetCreatureThenDrawIfUndamaged { draw_amount })
        }
        "damage_each_opponent" => {
            let amount: i32 = require(dict, "amount")?.extract()?;
            Ok(Effect::DamageEachOpponent { amount })
        }
        "damage_each_creature" => {
            let amount: i32 = require(dict, "amount")?.extract()?;
            Ok(Effect::DamageEachCreature { amount })
        }
        "create_token" => {
            let count: i32 = require(dict, "count")?.extract()?;
            let token_printing_id: u32 = require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::CreateToken { count, token_printing_id })
        }
        "create_token_each_player" => {
            // BLOCKED-RUST-EFFECT slice (round 43) -- each living
            // player creates `count` tokens of the given printing
            // under their own control. Sister of the static
            // `create_token` arm above; same field schema (`count`
            // + `token_printing_id`), distinguished by variant
            // (closed-vocab convention -- the each-player vs
            // self-only choice rides on the variant name, not a
            // boolean field, mirroring round-19/30/31/.. sister-
            // variant precedent). Mirrors Python's
            // `CreateToken(token_printing=..., count=N,
            // each_player=True)` (`tokens.py:198`).
            let count: i32 = require(dict, "count")?.extract()?;
            let token_printing_id: u32 = require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::CreateTokenEachPlayer { count, token_printing_id })
        }
        "create_token_tapped" => {
            // Phase 6 token-printing substrate -- Phase 4 (round 50).
            // Sister of the static `create_token` arm above; same
            // two-field schema (`count` + `token_printing_id`),
            // distinguished by variant (closed-vocab convention -- the
            // tapped-vs-untapped choice rides on the variant name, not
            // a boolean field, mirroring the round-43
            // `create_token_each_player` sister + round-19/30/31/..
            // sister-variant precedent). Mirrors Python's
            // `CreateToken(token_printing=..., count=N, tapped=True)`
            // (`tokens.py:286-287`). EoE consumer: Gravpack Monoist
            // ("When this creature dies, create a tapped 2/2 Robot
            // token." -- 25,252 17lands decks).
            let count: i32 = require(dict, "count")?.extract()?;
            let token_printing_id: u32 = require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::CreateTokenTapped { count, token_printing_id })
        }
        "add_mana_to_pool" => {
            let color: String = require(dict, "color")?.extract()?;
            // Validate against the closed mana-color vocab. Shared
            // helper with `parse_mana_produced` — reject "X", "g"
            // (lowercase), "WG" (compound), etc. at register time.
            validate_mana_symbol(&color)?;
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "add_mana_to_pool: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::AddManaToPool { color, amount })
        }
        "search_library_for_basic_land_tapped" => {
            // Python's `SearchLibraryForBasicLandTapped(count: int = 1)`
            // has a count field with default 1. Mirror that here:
            // missing `count` defaults to 1, present `count` extracts
            // as i32. Negative counts surface at register time
            // (defense-in-depth; `apply_effect` re-checks).
            let count: i32 = match dict.get_item("count")? {
                Some(c) => c.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "search_library_for_basic_land_tapped.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "search_library_for_basic_land_tapped: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::SearchLibraryForBasicLandTapped { count })
        }
        "put_plus_one_counter_on_self" => {
            // Python's `PutPlusOneCounterOnSelf(amount: int = 1)` has
            // an `amount` field defaulting to 1. Mirror that here:
            // missing `amount` defaults to 1, present `amount`
            // extracts as i32. Negative `amount` surfaces at register
            // time (defense-in-depth; `apply_effect` re-checks).
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_self.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_self: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::PutPlusOneCounterOnSelf { amount })
        }
        "surveil" => {
            // Python's `Surveil(amount: int = 1)` dataclass has an
            // `amount` field defaulting to 1. The Rust bridge mirrors
            // ScryN's shape (sibling effect: same look-at-top-N pattern,
            // different destination zone) and uses `count` here to stay
            // consistent with ScryN's parser arm. Missing `count` is a
            // KeyError at register time — there's no first-slice
            // default since Surveil's printed value is always present
            // on real cards. Negative `count` surfaces at register time
            // (defense-in-depth; `apply_effect` re-checks).
            let count: i32 = require(dict, "count")?.extract()?;
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "surveil: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::Surveil { count })
        }
        "counter_target_spell" => {
            // Unit variant — no required fields. Python's
            // `CounterTargetSpell.filter_types` is intentionally
            // dropped from the Rust first slice (see the variant's
            // docstring). Any extra keys on the dict are ignored,
            // matching the silent-drop convention for unrecognized
            // keys elsewhere in the parser.
            Ok(Effect::CounterTargetSpell)
        }
        "return_creature_from_your_graveyard_to_battlefield" => {
            // Unit variant — no required fields. Python's unified
            // `ReturnFromGraveyard` carries optional widening / capping
            // fields (`or_enchantment`, `or_spacecraft`, `or_artifact`,
            // `max_mv`, `max_n`, ...) that the Rust first slice
            // intentionally drops (see the variant's docstring). Any
            // extra keys on the dict are ignored, matching the silent-
            // drop convention elsewhere in the parser.
            Ok(Effect::ReturnCreatureFromYourGraveyardToBattlefield)
        }
        "return_enchantment_or_creature_from_your_graveyard_to_battlefield" => {
            // **Phase 5 rider-widened parse arm.** Unit variant; the
            // disjunction (enchantment OR creature) is baked into the
            // variant identity itself, so no fields beyond `kind`.
            // Mirrors Python's
            // `ReturnCreatureFromYourGraveyardToBattlefield(or_enchantment=True)`
            // — extra dict fields are silently ignored for round-trip
            // dict-shape consistency. This is the to-BATTLEFIELD
            // sibling of the existing
            // `return_enchantment_or_creature_from_your_graveyard_to_hand`
            // arm; the destination delta is the load-bearing difference.
            // EoE consumer: Rescue Skiff (real-EoE).
            Ok(Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield)
        }
        "discard_self" => {
            // Python's `DiscardSelf(amount: int = 1)` dataclass has
            // an `amount` field defaulting to 1. Mirror that here:
            // missing `amount` defaults to 1, present `amount`
            // extracts as i32. Negative `amount` surfaces at register
            // time (defense-in-depth; `apply_effect` re-checks).
            // Mirrors PutPlusOneCounterOnSelf's default-handling
            // shape.
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "discard_self.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "discard_self: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::DiscardSelf { amount })
        }
        "each_opponent_loses_life" => {
            // Python's `EachOpponentLosesLife(amount: int)` carries
            // a required `amount` field with no default. Mirror that
            // here: missing `amount` is a KeyError at register time;
            // negative amount surfaces at register time (defense-in-
            // depth; `apply_effect` re-checks).
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "each_opponent_loses_life: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::EachOpponentLosesLife { amount })
        }
        "each_opponent_loses_life_equal_to_artifact_count" => {
            // BLOCKED-RUST-EFFECT slice (round 44) -- each opponent of
            // the source-controller loses life equal to the
            // source-controller's artifact count. Unit variant; no
            // fields beyond `kind` (the count is dynamic -- read from
            // state at resolve time). Extra dict fields silently
            // ignored for round-trip dict-shape consistency (mirrors
            // every other unit-variant parse arm; particularly the
            // round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`, the
            // round-34 sister `surveil_x_equal_to_artifact_count`, and
            // the round-37 sister
            // `deal_damage_to_opponent_creature_by_creatures_you_control`
            // which share the X-from-count plumbing with this arm but
            // use a different apply substrate). Mirrors Python's
            // `EachOpponentLosesLifeEqualToArtifactCount` (`life.py` --
            // shipped in this slice as the FIFTH apply substrate to
            // pick up the X-from-count plumbing; 0 EoE consumers today,
            // slice ships per round-27/31/37 round-trip-surface-coverage
            // rule and to extend the X-from-count substrate axis to a
            // FIFTH entry, confirming the round-35 bidirectionally-
            // extensible plumbing generalizes across both axes).
            Ok(Effect::EachOpponentLosesLifeEqualToArtifactCount)
        }
        "each_opponent_loses_life_equal_to_your_life_total" => {
            // BLOCKED-RUST-EFFECT slice (round 48) -- each opponent of
            // the source-controller loses life equal to the
            // source-controller's CURRENT LIFE TOTAL. Unit variant; no
            // fields beyond `kind` (the count is dynamic -- read from
            // state at resolve time, clamped to >= 0). Extra dict
            // fields silently ignored for round-trip dict-shape
            // consistency (mirrors every other unit-variant parse arm;
            // particularly the round-44 sister
            // `each_opponent_loses_life_equal_to_artifact_count` -- same
            // apply substrate, only the denominator read differs -- and
            // the round-42 `mill_target_player_equal_to_your_life_total`
            // / round-47 `deal_damage_to_opponent_creature_by_life_total`
            // life-total-denominator precedents on OTHER substrates).
            // Mirrors Python's `EachOpponentLosesLifeEqualToYourLifeTotal`
            // (`life.py` -- shipped in this slice as the SECOND
            // denominator on the life-loss-each-opponent apply
            // substrate; 0 EoE consumers today, slice ships per
            // round-27/31/37 round-trip-surface-coverage rule and to
            // extend the X-from-count matrix to a third (substrate,
            // life-total) cell, confirming the round-35 bidirectionally-
            // extensible plumbing generalizes across both axes).
            Ok(Effect::EachOpponentLosesLifeEqualToYourLifeTotal)
        }
        "gain_life_equal_to_artifact_count" => {
            // BLOCKED-RUST-EFFECT slice (round 45) -- the source's
            // controller GAINS life equal to the source-controller's
            // artifact count. Unit variant; no fields beyond `kind`
            // (the count is dynamic -- read from state at resolve
            // time). Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-variant
            // parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`,
            // the round-34 sister `surveil_x_equal_to_artifact_count`,
            // the round-37 sister
            // `deal_damage_to_opponent_creature_by_creatures_you_control`,
            // and the round-44 sister
            // `each_opponent_loses_life_equal_to_artifact_count`
            // which share the X-from-count plumbing with this arm but
            // use a different apply substrate). Mirrors Python's
            // `GainLifeEqualToArtifactCount` (`life.py` -- shipped in
            // this slice as the SIXTH apply substrate to pick up the
            // X-from-count plumbing; 0 EoE consumers today, slice
            // ships per round-27/31/37 round-trip-surface-coverage rule
            // and to extend the X-from-count substrate axis to a SIXTH
            // entry, confirming the round-35 bidirectionally-extensible
            // plumbing generalizes across both axes). This arm is the
            // controller-self MIRROR of round-44's each-opponent-life-
            // loss arm: same denominator (artifact-count), same
            // dispatch shape, inverted recipient (controller-self) AND
            // inverted sign (gain vs loss).
            Ok(Effect::GainLifeEqualToArtifactCount)
        }
        "gain_life_equal_to_your_life_total" => {
            // BLOCKED-RUST-EFFECT slice (round 49) -- the source's
            // controller GAINS life equal to the source-controller's
            // CURRENT LIFE TOTAL. Unit variant; no fields beyond `kind`
            // (the count is dynamic -- read from state at resolve time,
            // clamped to >= 0). Extra dict fields silently ignored for
            // round-trip dict-shape consistency (mirrors every other
            // unit-variant parse arm; particularly the round-45 sister
            // `gain_life_equal_to_artifact_count` -- SAME apply
            // substrate (gain-life-controller-self), only the
            // denominator read differs -- and the round-42
            // `mill_target_player_equal_to_your_life_total` / round-47
            // `deal_damage_to_opponent_creature_by_life_total` /
            // round-48 `each_opponent_loses_life_equal_to_your_life_total`
            // life-total-denominator precedents on OTHER substrates).
            // Mirrors Python's `GainLifeEqualToYourLifeTotal`
            // (`life.py` -- shipped in this slice as the SECOND
            // denominator on the gain-life-controller-self apply
            // substrate; 0 EoE consumers today, slice ships per
            // round-27/31/37 round-trip-surface-coverage rule and to
            // extend the X-from-count matrix to a FOURTH (substrate,
            // life-total) cell, confirming the round-35 bidirectionally-
            // extensible plumbing generalizes across both axes).
            Ok(Effect::GainLifeEqualToYourLifeTotal)
        }
        "lose_life" => {
            // Python's `LoseLife(amount: int)` carries a required
            // `amount` field with no default. Mirror that here: missing
            // `amount` is a KeyError at register time; negative amount
            // surfaces at register time (defense-in-depth;
            // `apply_effect` re-checks). Distinct from
            // `each_opponent_loses_life` — same shape but the
            // dispatch subtracts from the source's CONTROLLER's life
            // (not the opponent's).
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "lose_life: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::LoseLife { amount })
        }
        "target_opponent_loses_life" => {
            // Python's `TargetOpponentLosesLife(amount: int = 1)` has
            // an `amount` field defaulting to 1. Mirror that here:
            // missing `amount` defaults to 1, present `amount` extracts
            // as i32. Negative `amount` surfaces at register time
            // (defense-in-depth; `apply_effect` re-checks). Distinct
            // from `each_opponent_loses_life` — same round-26 source-
            // finder dispatch shape in 2-player land, but the variant
            // stays separate to mirror Python's closed Effect
            // vocabulary (targeted vs each-opponent semantics).
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "target_opponent_loses_life.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "target_opponent_loses_life: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::TargetOpponentLosesLife { amount })
        }
        "target_player_loses_life" => {
            // **PPP Pin 1 (10th Rust slice).** Python's
            // `TargetPlayerLosesLife(amount: int = 1)` (life.py:80)
            // has an `amount` field defaulting to 1. Mirror the
            // sibling `target_opponent_loses_life` arm shape:
            // missing `amount` defaults to 1, present extracts as
            // i32, negative amount surfaces at register time.
            //
            // **Distinct from `target_opponent_loses_life`.**
            // `target_opponent_loses_life` auto-picks the defending
            // opponent at resolve time; `target_player_loses_life`
            // receives a PRE-FIXED target player via the
            // payload-target seat (Entropic Battlecruiser's
            // DISCARDS_CARD trigger pre-wires the discarding player
            // as the target).
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "target_player_loses_life.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "target_player_loses_life: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::TargetPlayerLosesLife { amount })
        }
        "each_opponent_discards" => {
            // Python's `EachOpponentDiscards(count: int = 1)` dataclass
            // has a `count` field defaulting to 1. Mirror that here:
            // missing `count` defaults to 1, present `count` extracts as
            // i32. Negative `count` surfaces at register time (defense-
            // in-depth; `apply_effect` re-checks). Mirrors DiscardSelf /
            // TargetOpponentLosesLife default-handling shape rather than
            // EachOpponentLosesLife's required-amount shape.
            let count: i32 = match dict.get_item("count")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "each_opponent_discards.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "each_opponent_discards: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::EachOpponentDiscards { count })
        }
        "destroy_all_creatures" => {
            // Unit variant — no required fields. Python's
            // `DestroyAllCreatures` carries optional narrowing fields
            // (`tapped_only`, `untapped_only`, `min_power`) that the
            // Rust first slice intentionally drops (see the variant's
            // docstring). Any extra keys on the dict are ignored,
            // matching the silent-drop convention for unrecognized
            // keys elsewhere in the parser.
            Ok(Effect::DestroyAllCreatures)
        }
        "exile_all_creatures_and_spacecraft" => {
            // Beyond-the-Quiet slice (2026-05-21). Unit variant — no
            // required fields. Python's `ExileAllCreaturesAndSpacecraft`
            // is a frozen dataclass with no constructor parameters
            // (mirrors `DestroyAllCreatures`'s parse-arm shape). Extra
            // keys on the dict are silently ignored (the silent-drop
            // convention for unrecognized keys elsewhere in the parser).
            Ok(Effect::ExileAllCreaturesAndSpacecraft)
        }
        "destroy_target_nonland_permanent" => {
            // Unit variant — no required fields. Filter is intrinsic
            // ("non-Land, non-indestructible"). Extra keys ignored.
            Ok(Effect::DestroyTargetNonlandPermanent)
        }
        "destroy_target_artifact" => {
            // Unit variant — no required fields. Filter is intrinsic
            // ("Artifact, non-indestructible"). Extra keys ignored.
            // Mirrors `destroy_target_nonland_permanent` parser arm
            // shape (silent-drop convention for unrecognized keys).
            Ok(Effect::DestroyTargetArtifact)
        }
        "destroy_target_land" => {
            // Unit variant — no required fields. Filter is intrinsic
            // ("Land, non-indestructible"). Extra keys ignored.
            // Mirrors `destroy_target_artifact` parser arm shape
            // (silent-drop convention for unrecognized keys); only the
            // intrinsic filter differs.
            Ok(Effect::DestroyTargetLand)
        }
        "destroy_target_noncreature_permanent" => {
            // Round-52 — Extinguisher Battleship ETB head. Unit
            // variant — no required fields. Filter is intrinsic
            // ("non-Creature, non-indestructible"). Extra keys
            // ignored. Mirrors `destroy_target_nonland_permanent`
            // parser arm shape (silent-drop convention for
            // unrecognized keys); only the intrinsic filter differs
            // (Creature-exclude vs Land-exclude).
            Ok(Effect::DestroyTargetNoncreaturePermanent)
        }
        "destroy_target_artifact_or_tapped_creature" => {
            // Effect-uncovered Tier-A — Radiant Strike inner half.
            // Unit variant — no required fields. Filter is intrinsic
            // ("Artifact OR (Creature AND tapped), non-indestructible").
            // Extra keys ignored. Mirrors `destroy_target_artifact`
            // parser arm shape (silent-drop convention for
            // unrecognized keys); only the intrinsic filter differs
            // (artifact-OR-tapped-creature vs artifact-only).
            Ok(Effect::DestroyTargetArtifactOrTappedCreature)
        }
        "seedship_impact_effect" => {
            // Effect-uncovered Tier-A — Seedship Impact (Lander-rider).
            // Single-field payload: the Lander's `token_printing_id`
            // (registry index). Mirrors the `create_token` parse arm's
            // `require(dict, "token_printing_id")` shape — the bespoke
            // arm needs the pid to mint the Lander (Rust has no
            // intrinsic Lander). Python's `SeedshipImpactEffect` is
            // field-free; the pid is a Rust-side carrier dropped at
            // the bridge.
            let token_printing_id: u32 =
                require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::SeedshipImpactEffect { token_printing_id })
        }
        "deal_damage_to_target_creature_with_excess_lander_rider" => {
            // Effect-uncovered Tier-A — Orbital Plunge (Lander-rider).
            // Two-field payload: `amount` (damage; 6 for Orbital
            // Plunge) + the Lander's `token_printing_id` (registry
            // index — the same pid-carrier shape as
            // `seedship_impact_effect`; Rust has no intrinsic Lander,
            // Python's class is field-free and the bridge drops the
            // pid). Mirrors the `deal_damage_any_target` + the
            // `create_token` parse arms.
            let amount: i32 = require(dict, "amount")?.extract()?;
            let token_printing_id: u32 =
                require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::DealDamageToTargetCreatureWithExcessLanderRider {
                amount,
                token_printing_id,
            })
        }
        "depressurize_effect" => {
            // Effect-uncovered Tier-A — Depressurize. Single-field
            // payload: `power_penalty` (3 for Depressurize). Mirrors
            // the `pump_target_creature_until_eot` parse arm's
            // single-i32-key shape. Python's
            // `DepressurizeEffect(power_penalty=3)` carries the same
            // single field.
            let power_penalty: i32 =
                require(dict, "power_penalty")?.extract()?;
            Ok(Effect::DepressurizeEffect { power_penalty })
        }
        "biosynthic_burst_effect" => {
            // Effect-uncovered Tier-A — Biosynthic Burst. Unit
            // variant; no fields (mirrors the
            // `each_opponent_sacrifices_creature` field-free idiom).
            // Python's `BiosynthicBurstEffect()` carries no fields.
            Ok(Effect::BiosynthicBurstEffect)
        }
        "focus_fire_effect" => {
            // Effect-uncovered Tier-A — Focus Fire. Single `base`
            // field (the printed `2` — "X is 2 plus ..."); the
            // dynamic +creatures-and-Spacecraft count is read at
            // resolve time. Mirrors `FocusFireEffect(base=N)`
            // (`effects/bespoke.py:1753`); copies the
            // `deal_damage_any_target` single-int-field parse shape.
            // Default 2 (Focus Fire's printed base) when absent — the
            // bridge arm uses the same default, so the round-trip is
            // clean even on a key-elided dict.
            let base: i32 = match dict.get_item("base")? {
                Some(v) => v.extract()?,
                None => 2,
            };
            Ok(Effect::FocusFireEffect { base })
        }
        "diplomatic_relations_effect" => {
            // Effect-uncovered Tier-A #7 — Diplomatic Relations
            // (spell-card-bridge Phase 2). Unit variant; no fields
            // (mirrors the `biosynthic_burst_effect` field-free
            // idiom). Python's `DiplomaticRelationsEffect()` carries
            // no fields — the +1/+0 / vigilance / power-equal damage
            // are all intrinsic. The 2-slot targets ride on the
            // StackItem, NOT in this dict.
            Ok(Effect::DiplomaticRelationsEffect)
        }
        "cut_propulsion_effect" => {
            // Effect-uncovered Tier-A — Cut Propulsion. Unit variant;
            // no fields (mirrors the `biosynthic_burst_effect` /
            // `diplomatic_relations_effect` field-free idiom).
            // Python's `CutPropulsionEffect()` carries no fields —
            // the damage amount is intrinsic (read from the target's
            // effective power at resolve, doubled if the target has
            // flying). The single target rides on the StackItem
            // (`current_resolve_payload_target_id`), NOT in this dict.
            Ok(Effect::CutPropulsionEffect)
        }
        "invasive_maneuvers_effect" => {
            // Effect-uncovered Tier-A — Invasive Maneuvers. Two-int
            // variant: `base` (printed `3`) + `boosted` (printed `5`)
            // — the per-clause `_DEAL_DAMAGE_N_TARGET_CREATURE_M_IF_
            // SPACECRAFT` parser at `parser/effects.py:4668-4673`
            // captures the two integers off the printed oracle text.
            // Mirrors Python's
            // `DealDamageToTargetCreatureSpacecraftBoosted(base=3,
            // boosted=5)` `@dataclass(frozen=True)` instance. Default
            // `(3, 5)` on both sides (Python dataclass field defaults
            // `base: int = 3`, `boosted: int = 5`) so a key-elided
            // round-trip is clean. The single target rides on the
            // StackItem (`current_resolve_payload_target_id`), NOT in
            // this dict; the Spacecraft check is intrinsic (resolves
            // off the source-controller's battlefield, NOT in dict).
            let base: i32 = match dict.get_item("base")? {
                Some(v) => v.extract()?,
                None => 3,
            };
            let boosted: i32 = match dict.get_item("boosted")? {
                Some(v) => v.extract()?,
                None => 5,
            };
            Ok(Effect::InvasiveManeuversEffect { base, boosted })
        }
        "destroy_target_artifact_enchantment_or_flying_creature" => {
            // Effect-uncovered single-gated tier — Shattered Wings
            // (~7,793 17lands decks). Unit variant — no required
            // fields. Filter is intrinsic ("Artifact OR Enchantment
            // OR (Creature AND has flying), non-indestructible").
            // Extra keys ignored. Mirrors
            // `destroy_target_artifact_or_tapped_creature` parser
            // arm shape (silent-drop convention for unrecognized
            // keys); only the intrinsic filter differs (3-way OR
            // with the flying-creature leg vs the 2-way artifact-
            // OR-tapped-creature OR).
            Ok(Effect::DestroyTargetArtifactEnchantmentOrFlyingCreature)
        }
        "exorcise_effect" => {
            // Effect-uncovered single-gated tier — Exorcise (Duskmourn).
            // Unit variant — no required fields. Filter is intrinsic
            // ("Artifact OR Enchantment OR (Creature AND effective
            // power >= 4)"). Indestructible NOT skipped (exile bypasses
            // indestructible per CR 702.12). Extra keys ignored.
            // Mirrors `destroy_target_artifact_enchantment_or_flying_
            // creature` parser arm shape — only the two filter swaps
            // differ (exile-not-destroy, power-4-not-flying).
            Ok(Effect::ExorciseEffect)
        }
        "close_encounter_effect" => {
            // Close Encounter — additional-cost-choice closure
            // (Cycle 3, 2026-05-21; ~16,098 17lands decks). Unit
            // variant — no required fields. The damage amount is
            // intrinsic (derived from the chosen object's effective
            // power on bf or printed power on exile, read at
            // resolve via `state.current_resolve_additional_cost_
            // choice_id`). The single damage-target rides on the
            // StackItem (`current_resolve_payload_target_id`); the
            // additional-cost choice rides on the StackItem
            // (`additional_cost_choice_id`, Cycle 1 substrate). Extra
            // keys ignored. Mirrors the `exorcise_effect` /
            // `diplomatic_relations_effect` field-free unit-variant
            // parser arm shape.
            Ok(Effect::CloseEncounterEffect)
        }
        "emergency_eject_effect" => {
            // Effect-uncovered single-gated tier — Emergency Eject
            // (~6,291 17lands decks). Single-field payload: the
            // Lander's `token_printing_id` (registry index). Same
            // pid-carrier shape as `seedship_impact_effect`: Rust
            // has no intrinsic Lander, so the bespoke arm needs the
            // pid to mint the Lander; Python's
            // `DestroyTargetNonlandPermanentWithLanderForController`
            // is field-free and mints from
            // `TOKEN_PRINTINGS["Lander"]`; the bridge drops the pid.
            let token_printing_id: u32 =
                require(dict, "token_printing_id")?.extract()?;
            Ok(Effect::EmergencyEjectEffect { token_printing_id })
        }
        "deal_damage_any_target" => {
            // Python's `DealDamageAnyTarget(DealDamage)` carries the
            // inherited `amount` field plus an `optional: bool` slot
            // flag (added for Pinnacle Kill-Ship's "up to one target
            // creature" shape — flips Python's slot count_min to 0).
            // The "any target" filter is intrinsic to the variant
            // (no `target` dict to parse — distinct from
            // `deal_damage`'s required `target` field). Missing
            // `amount` is a KeyError at register time; negative
            // amount surfaces at register time (defense-in-depth;
            // `apply_effect` re-checks). The `optional` key is
            // tolerated as an ignored extra field — Rust's
            // deterministic at-apply-time target pick (scan
            // defender's bf for first creature, fall back to
            // defender-player) is observably equivalent to Python's
            // optional-target "no-target fizzle" only on probes that
            // drive a single opponent creature on bf, but the dict-
            // shape round-trip needs both sides to accept the same
            // keyset for a clean register-on-both-engines contract
            // (silent-drop convention matches the broader parser).
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "deal_damage_any_target: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::DealDamageAnyTarget { amount })
        }
        "pump_self_until_eot" => {
            // Round-27 UEOT-framework proof. Python's
            // `PumpSelfUntilEOT(power_bonus: int = 0,
            // toughness_bonus: int = 0)` defaults each field to 0;
            // the Rust bridge mirrors the default-handling shape used
            // by `DiscardSelf` / `PutPlusOneCounterOnSelf` /
            // `EachOpponentDiscards` (`dict.get_item` with explicit
            // default rather than `require`). Either field absent =
            // 0; both absent registers a 0/0 no-op (legal for parity
            // edge cases). NEGATIVES are deliberately allowed —
            // Python's PumpSelfUntilEOT accepts negative bonuses for
            // debuff-shaped self-pumps (e.g. cards that shrink
            // themselves for a beneficial side-effect). Mirrors the
            // schema's `int` annotation (no non-negative guard) so a
            // future pool card with a negative self-pump doesn't fail
            // registration here.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_self_until_eot.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_self_until_eot.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpSelfUntilEOT { power_bonus, toughness_bonus })
        }
        "pump_target_creature_until_eot" => {
            // Round-28 UEOT-coverage extension. Mirrors the
            // `pump_self_until_eot` schema: `power_bonus` and
            // `toughness_bonus` default to 0; negatives allowed for
            // debuff-shaped target effects.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_creature_until_eot.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_creature_until_eot.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpTargetCreatureUntilEOT { power_bonus, toughness_bonus })
        }
        "pump_target_creature_until_eot_grant_dies_return_tapped" => {
            // BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner
            // ETB. Same `power_bonus` / `toughness_bonus` schema as
            // `pump_target_creature_until_eot` (both default to 0;
            // negatives allowed for symmetry, though Perigee is +2/+0).
            // The "another / you-control" target restriction and the
            // granted dies-return-tapped rider are baked into the
            // variant identity — no extra dict fields (the rider is
            // the fixed Perigee shape, not a generic ability payload;
            // the general nested-trigger-grant subsystem is the
            // deferred half — see the variant doc-comment + round-60
            // SEARCH_LIMITATIONS). Extra keys silently ignored for
            // round-trip dict-shape consistency.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_creature_until_eot_grant_dies_return_tapped.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_creature_until_eot_grant_dies_return_tapped.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpTargetCreatureUntilEotGrantDiesReturnTapped {
                power_bonus,
                toughness_bonus,
            })
        }
        "fight_target_creature_with_target_creature" => {
            // BLOCKED-RUST-EFFECT slice — Fight effect arm. Unit
            // variant; the bilateral source-creature-fights-opponent-
            // creature shape is baked into the variant identity, so
            // no fields beyond `kind`. Extra dict fields are silently
            // ignored for round-trip dict-shape consistency (mirrors
            // every other unit-variant parse arm).
            //
            // Mirrors Python's `FightTargetCreatureWithTargetCreature`
            // generic-fight class shipped on the bridge in the same
            // slice.
            Ok(Effect::FightTargetCreatureWithTargetCreature)
        }
        "double_target_creatures_power_ueot" => {
            // BLOCKED-RUST-EFFECT slice (round 31) — double target
            // creature's power UEOT. Unit variant; no fields beyond
            // `kind`. Extra dict fields silently ignored for round-
            // trip dict-shape consistency (mirrors every other unit-
            // variant parse arm). Mirrors Python's
            // `DoubleTargetCreaturesPowerUEOT` (`pump.py:194` —
            // Mightform Harmonizer's landfall trigger).
            Ok(Effect::DoubleTargetCreaturesPowerUEOT)
        }
        "deal_damage_to_opponent_creature_by_artifact_count" => {
            // BLOCKED-RUST-EFFECT slice (round 32) — deal damage to
            // an opponent's creature equal to the source-controller's
            // artifact count. Unit variant; no fields beyond `kind`
            // (the count is dynamic — read from state at resolve
            // time). Extra dict fields silently ignored for round-
            // trip dict-shape consistency (mirrors every other unit-
            // variant parse arm). Mirrors Python's
            // `DealDamageToOpponentCreatureByArtifactCount`
            // (`damage.py:406` — Warmaker Gunship's ETB).
            Ok(Effect::DealDamageToOpponentCreatureByArtifactCount)
        }
        "two_subject_pump_and_debuff_by_artifact_count" => {
            // BLOCKED-RUST-EFFECT slice (round 51) — Monoist
            // Circuit-Feeder ETB (real EoE card; 14,727 17lands
            // decks). Until EOT: source-controller's first creature
            // gets +X/+0, an opponent's first creature gets -0/-X,
            // X = source-controller's artifact count. Unit variant;
            // no fields beyond `kind` (X is dynamic — read from
            // state at resolve time, clamped >= 0). Extra dict
            // fields silently ignored for round-trip dict-shape
            // consistency (mirrors every other unit-variant parse
            // arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`
            // — SAME artifact-count denominator read, NEW two-
            // subject until-EOT pump+debuff apply substrate).
            // Mirrors Python's deterministic-pick bridge sibling
            // `PumpYoursAndDebuffOpponentByArtifactCountUntilEOT`
            // (`pump.py`). NOT loader-admitted — the real card's
            // agent-driven `PairedTargetPumpDebuffUEOT` would
            // silently degrade through the deterministic bridge
            // (§2; see the variant docstring + SEARCH_LIMITATIONS).
            Ok(Effect::TwoSubjectPumpAndDebuffByArtifactCount)
        }
        "deal_damage_to_opponent_creature_by_cards_in_hand" => {
            // BLOCKED-RUST-EFFECT slice (round 35) — deal damage to
            // an opponent's creature equal to the source-controller's
            // hand size. Unit variant; no fields beyond `kind` (the
            // count is dynamic — read from state at resolve time).
            // Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-
            // variant parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`
            // which shares the apply substrate with this arm but
            // uses an artifact-count denominator instead). Mirrors
            // Python's `DealDamageToOpponentCreatureByCardsInHand`
            // (`damage.py:464` — Roaring Furnace's door trigger,
            // 1 EoE consumer).
            Ok(Effect::DealDamageToOpponentCreatureByCardsInHand)
        }
        "deal_damage_to_opponent_creature_by_lands_controlled" => {
            // BLOCKED-RUST-EFFECT slice (round 36) — deal damage to
            // an opponent's creature equal to the source-controller's
            // lands count. Unit variant; no fields beyond `kind` (the
            // count is dynamic — read from state at resolve time).
            // Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-
            // variant parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`
            // and the round-35 sister
            // `deal_damage_to_opponent_creature_by_cards_in_hand`
            // which share the apply substrate with this arm but use
            // an artifact-count / hand-size denominator instead).
            // Mirrors Python's
            // `DealDamageToOpponentCreatureByLandsControlled`
            // (`damage.py` -- shipped in this slice as a sister of
            // the round-32/35 classes; 0 EoE consumers today, slice
            // ships per round-27/31 round-trip-surface-coverage rule
            // and to complete the THIRD orthogonal denominator on the
            // round-32 damage-to-opponent-creature substrate).
            Ok(Effect::DealDamageToOpponentCreatureByLandsControlled)
        }
        "deal_damage_to_opponent_creature_by_creatures_you_control" => {
            // BLOCKED-RUST-EFFECT slice (round 37) -- deal damage to
            // an opponent's creature equal to the source-controller's
            // creatures count. Unit variant; no fields beyond `kind`
            // (the count is dynamic -- read from state at resolve
            // time). Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-variant
            // parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`, the
            // round-35 sister
            // `deal_damage_to_opponent_creature_by_cards_in_hand`, and
            // the round-36 sister
            // `deal_damage_to_opponent_creature_by_lands_controlled`
            // which share the apply substrate with this arm but use an
            // artifact-count / hand-size / lands-controlled denominator
            // instead). Mirrors Python's
            // `DealDamageToOpponentCreatureByCreaturesYouControl`
            // (`damage.py` -- shipped in this slice as a sister of the
            // round-32/35/36 classes; 0 EoE consumers today, slice
            // ships per round-27/31 round-trip-surface-coverage rule
            // and to complete the FOURTH orthogonal denominator on the
            // round-32 damage-to-opponent-creature substrate).
            Ok(Effect::DealDamageToOpponentCreatureByCreaturesYouControl)
        }
        "deal_damage_to_opponent_creature_by_cards_in_graveyard" => {
            // BLOCKED-RUST-EFFECT slice (round 46) -- deal damage to
            // an opponent's creature equal to the source-controller's
            // graveyard size. Unit variant; no fields beyond `kind`
            // (the count is dynamic -- read from state at resolve
            // time). Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-variant
            // parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`,
            // the round-35 sister
            // `deal_damage_to_opponent_creature_by_cards_in_hand`, the
            // round-36 sister
            // `deal_damage_to_opponent_creature_by_lands_controlled`,
            // and the round-37 sister
            // `deal_damage_to_opponent_creature_by_creatures_you_control`
            // which share the apply substrate with this arm but use an
            // artifact-count / hand-size / lands-controlled /
            // creatures-controlled denominator instead). Mirrors
            // Python's `DealDamageToOpponentCreatureByCardsInGraveyard`
            // (`damage.py` -- shipped in this slice as a sister of the
            // round-32/35/36/37 classes; 0 EoE consumers today, slice
            // ships per round-27/31 round-trip-surface-coverage rule
            // and to complete the FIFTH orthogonal denominator on the
            // round-32 damage-to-opponent-creature substrate. SIXTH
            // overall X-from-count denominator across the substrate
            // matrix -- prior five: artifact / hand / lands / creatures
            // / life-total; this slice adds GRAVEYARD as the first
            // zone-size denominator pulled from a non-battlefield,
            // non-hand source).
            Ok(Effect::DealDamageToOpponentCreatureByCardsInGraveyard)
        }
        "deal_damage_to_opponent_creature_by_life_total" => {
            // BLOCKED-RUST-EFFECT slice (round 47) -- deal damage to
            // an opponent's creature equal to the source-controller's
            // life total. Unit variant; no fields beyond `kind` (the
            // count is dynamic -- read from state at resolve time).
            // Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors every other unit-variant
            // parse arm; particularly the round-32 sister
            // `deal_damage_to_opponent_creature_by_artifact_count`,
            // the round-35 sister
            // `deal_damage_to_opponent_creature_by_cards_in_hand`, the
            // round-36 sister
            // `deal_damage_to_opponent_creature_by_lands_controlled`,
            // the round-37 sister
            // `deal_damage_to_opponent_creature_by_creatures_you_control`,
            // and the round-46 sister
            // `deal_damage_to_opponent_creature_by_cards_in_graveyard`
            // which share the apply substrate with this arm but use an
            // artifact-count / hand-size / lands-controlled /
            // creatures-controlled / graveyard-size denominator
            // instead). Mirrors Python's
            // `DealDamageToOpponentCreatureByLifeTotal` (`damage.py`
            // -- shipped in this slice as a sister of the
            // round-32/35/36/37/46 classes; 0 EoE consumers today,
            // slice ships per round-27/31 round-trip-surface-coverage
            // rule and to COMPLETE the SIXTH (final) orthogonal
            // denominator on the round-32 damage-to-opponent-creature
            // substrate -- the damage substrate row is now fully
            // saturated across the artifact / hand / lands / creatures
            // / graveyard / life-total denominator axis).
            Ok(Effect::DealDamageToOpponentCreatureByLifeTotal)
        }
        "put_plus_one_counters_on_self_per_other_creature_or_artifact" => {
            // BLOCKED-RUST-EFFECT slice (round 33) — put a +1/+1
            // counter on this creature for each OTHER creature and/or
            // artifact you control. Unit variant; no fields beyond
            // `kind` (the count is dynamic — read from state at
            // resolve time). Extra dict fields silently ignored for
            // round-trip dict-shape consistency (mirrors every other
            // unit-variant parse arm). Mirrors Python's
            // `PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact`
            // (`counters.py:394` — Luxknight Breacher's ETB).
            Ok(Effect::PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact)
        }
        "surveil_x_equal_to_artifact_count" => {
            // BLOCKED-RUST-EFFECT slice (round 34) — surveil X, where
            // X is the source-controller's artifact count. Unit
            // variant; no fields beyond `kind` (the count is dynamic —
            // read from state at resolve time). Extra dict fields
            // silently ignored for round-trip dict-shape consistency
            // (mirrors every other unit-variant parse arm). Mirrors
            // Python's `Surveil(use_artifact_count=True)`
            // (`library_lookat.py:569` — Cerebral Download's
            // surveil-X clause).
            Ok(Effect::SurveilXEqualToArtifactCount)
        }
        "return_creature_from_your_graveyard_to_hand" => {
            // Unit variant; no fields beyond `kind`.
            Ok(Effect::ReturnCreatureFromYourGraveyardToHand)
        }
        "return_artifact_or_creature_from_your_graveyard_to_hand" => {
            // **Phase 5 rider-widened parse arm.** Unit variant; the
            // disjunction (artifact OR creature) is baked into the
            // variant identity itself, so no fields beyond `kind`.
            // Mirrors Python's
            // `ReturnCreatureFromYourGraveyardToHand(or_artifact=True)`
            // — extra dict fields are silently ignored for round-trip
            // dict-shape consistency.
            Ok(Effect::ReturnArtifactOrCreatureFromYourGraveyardToHand)
        }
        "return_creature_or_spacecraft_from_your_graveyard_to_hand" => {
            // **Phase 5 rider-widened parse arm.** Unit variant; the
            // disjunction (creature OR Spacecraft subtype) is baked
            // into the variant identity itself, so no fields beyond
            // `kind`. Mirrors Python's
            // `ReturnCreatureFromYourGraveyardToHand(or_spacecraft=True)`
            // — extra dict fields are silently ignored.
            Ok(Effect::ReturnCreatureOrSpacecraftFromYourGraveyardToHand)
        }
        "return_enchantment_or_creature_from_your_graveyard_to_hand" => {
            // **Phase 5 rider-widened parse arm.** Unit variant; the
            // disjunction (enchantment OR creature) is baked into the
            // variant identity itself, so no fields beyond `kind`.
            // Mirrors Python's
            // `ReturnCreatureFromYourGraveyardToHand(or_enchantment=True)`
            // — extra dict fields are silently ignored for round-trip
            // dict-shape consistency.
            Ok(Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToHand)
        }
        "return_target_artifact_card_from_your_graveyard_to_hand" => {
            // **scour-for-scrap-cycle-2 (2026-05-21).** Unit variant;
            // the artifact-card filter is baked into the variant
            // identity itself, so no fields beyond `kind`. Mirrors
            // Python's `ReturnTargetArtifactCardFromYourGraveyardToHand`
            // (`bounce.py:338`) — a constructor-sugar subclass of
            // `ReturnFromGraveyard` with `artifact=True`,
            // `destination="hand"`, `target_kind="graveyard_artifact"`.
            // Extra dict fields are silently ignored for round-trip
            // dict-shape consistency.
            Ok(Effect::ReturnTargetArtifactCardFromYourGraveyardToHand)
        }
        "put_plus_one_counter_on_each_creature_you_control" => {
            // Python's PutPlusOneCounterOnEachCreatureYouControl
            // dataclass has amount: int = 1. Mirror that.
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_each_creature_you_control.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_each_creature_you_control: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::PutPlusOneCounterOnEachCreatureYouControl { amount })
        }
        "put_card_from_graveyard_on_bottom_of_library" => {
            // Unit variant; no fields.
            Ok(Effect::PutCardFromGraveyardOnBottomOfLibrary)
        }
        "move_self_counters_to_target_creature_you_control" => {
            Ok(Effect::MoveSelfCountersToTargetCreatureYouControl)
        }
        "deal_damage_equal_to_source_power_any_target" => {
            Ok(Effect::DealDamageEqualToSourcePowerAnyTarget)
        }
        "tap_target_artifact_or_creature" => {
            Ok(Effect::TapTargetArtifactOrCreature)
        }
        "create_token_copy_of_target_creature_you_control" => {
            Ok(Effect::CreateTokenCopyOfTargetCreatureYouControl)
        }
        "optional_sac_artifact_for_effect" => {
            let nested = dict.get_item("inner_effect")?.ok_or_else(|| {
                PyValueError::new_err(
                    "optional_sac_artifact_for_effect: missing required 'inner_effect' field",
                )
            })?;
            let nested_dict = nested.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "optional_sac_artifact_for_effect.inner_effect must be a dict",
                )
            })?;
            let inner = parse_effect(nested_dict)?;
            Ok(Effect::OptionalSacArtifactForEffect {
                inner_effect: Box::new(inner),
            })
        }
        "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library" => {
            // Unit variant; no fields.
            Ok(Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary)
        }
        "attach_to_target_creature_you_control" => {
            Ok(Effect::AttachToTargetCreatureYouControl)
        }
        "equip_to_target_creature_you_control" => {
            Ok(Effect::EquipToTargetCreatureYouControl)
        }
        "destroy_enchanted_permanent" => {
            Ok(Effect::DestroyEnchantedPermanent)
        }
        "tap_enchanted_permanent" => {
            Ok(Effect::TapEnchantedPermanent)
        }
        "target_opponent_look_hand_may_discard" => {
            // **Cracked Skull look-at-hand + may-discard substrate —
            // Cycle 2 (2026-05-22).** Unit variant; no payload fields
            // beyond `kind`. Mirrors the slice-19 / slice-20 unit-
            // variant parse arm shape verbatim (extra dict fields
            // silently ignored for round-trip dict-shape consistency).
            // Python's `TargetOpponentLookHandMayDiscard`
            // (`effects/generic/discard.py:298-391`) is a field-free
            // `@dataclass(frozen=True)`; the bridge constructs
            // `TargetOpponentLookHandMayDiscard()` with no args.
            Ok(Effect::TargetOpponentLookHandMayDiscard)
        }
        "target_opponent_sacrifices_creature_other_than_enchanted" => {
            // **Sporogenic Infection — Cycle 1 (2026-05-22).** Unit
            // variant; no payload fields beyond `kind`. Mirrors the
            // sibling `target_opponent_look_hand_may_discard` parse
            // arm shape verbatim. Python's
            // `TargetOpponentSacrificesCreatureOtherThanEnchanted`
            // (`effects/generic/opponent_punishers.py:68-145`) is a
            // field-free `@dataclass(frozen=True)`; the bridge
            // constructs the field-free Python class.
            Ok(Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted)
        }
        "put_stun_counter_on_tapped_creature" => {
            // Unit variant; no fields. Mirrors Python's
            // PutStunCounterOnTappedCreature with no constructor args.
            Ok(Effect::PutStunCounterOnTappedCreature)
        }
        "tap_target_opponent_permanent_and_stun_it" => {
            // Unit variant; no fields. Mirrors Python's
            // TapTargetOpponentPermanentAndStunIt with no constructor
            // args. Both the tap and the stun-counter increment apply
            // to the same chosen target.
            Ok(Effect::TapTargetOpponentPermanentAndStunIt)
        }
        "exile_target_creature_until_source_leaves" => {
            Ok(Effect::ExileTargetCreatureUntilSourceLeaves)
        }
        "exile_opponents_nonland_permanent_until_leaves" => {
            Ok(Effect::ExileOpponentsNonlandPermanentUntilLeaves)
        }
        "exile_opponents_nonland_permanent_with_max_mv_until_leaves" => {
            // **Round 14 rider-widened parse arm.** Carries
            // `max_mv: i32` (required, non-negative). Mirrors
            // Python's `ExileOpponentsNonlandPermanentUntilLeaves(max_mv=N)`
            // (`effects/generic/exile.py:587`). max_mv=0 would mean
            // "no cap" but the bare sibling already covers that
            // case; the rider variant requires max_mv > 0 by
            // convention to keep the two shapes disjoint at the
            // dict-shape layer.
            let max_mv: i32 = match dict.get_item("max_mv")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "exile_opponents_nonland_permanent_with_max_mv_until_leaves.max_mv must be an integer (got: {e})"
                    ))
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "exile_opponents_nonland_permanent_with_max_mv_until_leaves: missing required 'max_mv' field",
                    ));
                }
            };
            if max_mv < 0 {
                return Err(PyValueError::new_err(format!(
                    "exile_opponents_nonland_permanent_with_max_mv_until_leaves: max_mv must be non-negative, got {max_mv}"
                )));
            }
            Ok(Effect::ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves { max_mv })
        }
        "may_do" => {
            let nested = dict.get_item("inner_effect")?.ok_or_else(|| {
                PyValueError::new_err(
                    "may_do: missing required 'inner_effect' field",
                )
            })?;
            let nested_dict = nested.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "may_do.inner_effect must be a dict",
                )
            })?;
            let inner = parse_effect(nested_dict)?;
            let prompt: String = match dict.get_item("prompt")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "may_do.prompt must be a string (got: {e})"
                    ))
                })?,
                None => String::new(),
            };
            Ok(Effect::MayDo {
                inner_effect: Box::new(inner),
                prompt,
            })
        }
        "look_top_n_put_one_in_hand_rest_to_graveyard" => {
            let n: i32 = match dict.get_item("n")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "look_top_n_put_one_in_hand_rest_to_graveyard.n must be an integer (got: {e})"
                    ))
                })?,
                None => 4, // Python's class default (Commune with Evil n=4)
            };
            if n < 0 {
                return Err(PyValueError::new_err(format!(
                    "look_top_n_put_one_in_hand_rest_to_graveyard: n must be non-negative, got {n}"
                )));
            }
            Ok(Effect::LookTopNPutOneInHandRestToGraveyard { n })
        }
        "famished_worldsire_look_top_x_by_power_put_lands_tapped" => {
            // BLOCKED-RUST-EFFECT slice (round 54) — Famished
            // Worldsire ETB. Unit variant; no payload fields beyond
            // `kind` (X is dynamic — read from the source's effective
            // power at resolve, clamped >= 0; land selection is a
            // deterministic greedy-all mirroring Python's
            // `FamishedWorldsireETBEffect`, no decision plumbed).
            // Extra dict fields silently ignored for round-trip
            // dict-shape consistency (mirrors the round-49
            // `gain_life_equal_to_your_life_total` unit-variant parse
            // arm shape verbatim, swapping the literal).
            Ok(Effect::FamishedWorldsireLookTopXByPowerPutLandsTapped)
        }
        "astelli_reclaimer_etb_effect" => {
            // 16th Rust slice (TTT Pin 3 closure) — Astelli Reclaimer
            // ETB. Unit variant; no payload fields beyond `kind` (X is
            // dynamic — read from the source's printing's
            // `mana_cost`/`warp_cost` at resolve based on the
            // `warp_exile_pending` flag the entered permanent carries;
            // target selection is a deterministic first-legal
            // graveyard scan mirroring Python's
            // `AstelliReclaimerETBEffect`). Extra dict fields silently
            // ignored for round-trip dict-shape consistency (mirrors
            // the round-49 `gain_life_equal_to_your_life_total` /
            // round-54 `famished_worldsire_look_top_x_by_power_put_
            // lands_tapped` unit-variant parse arm shape verbatim).
            Ok(Effect::AstelliReclaimerETBEffect)
        }
        "roving_actuator_etb" => {
            // 18th Rust slice (XXX Pin 2 closure) — Roving Actuator
            // Void ETB. Single-field variant — `max_mv: u8` is the
            // printed mana-value threshold (2 on Roving Actuator).
            // Mirrors Python's `RovingActuatorETBEffect(max_mv: int = 2)`
            // dataclass field. Default to 2 if the key is omitted
            // (round-trip dict-shape consistency — same default as the
            // Python dataclass).
            let max_mv_i: i64 = match dict.get_item("max_mv")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "roving_actuator_etb.max_mv must be an integer (got: {e})"
                    ))
                })?,
                None => 2,
            };
            if max_mv_i < 0 || max_mv_i > 255 {
                return Err(PyValueError::new_err(format!(
                    "roving_actuator_etb: max_mv must be in 0..=255, got {max_mv_i}"
                )));
            }
            Ok(Effect::RovingActuatorETBEffect { max_mv: max_mv_i as u8 })
        }
        "enchanted_creature_fights_target" => {
            // 19th Rust slice (honest-stop substrate landing) —
            // Meltstrider's Resolve Aura ETB "fight". Unit variant; no
            // payload fields beyond `kind` (the source-side fighter is
            // the Aura's `attached_to` host, read at resolve time; the
            // opponent-side fighter is the first-legal scan on the
            // opponent's bf — matches `FightTargetCreatureWithTarget
            // Creature`'s deterministic-pick convention). Extra dict
            // fields silently ignored for round-trip dict-shape
            // consistency (mirrors round-49 `gain_life_equal_to_your_
            // life_total` / round-54 `famished_worldsire_look_top_x_
            // by_power_put_lands_tapped` / slice-16
            // `astelli_reclaimer_etb_effect` unit-variant parse arm
            // shape verbatim).
            Ok(Effect::EnchantedCreatureFightsTarget)
        }
        "pain_for_all_etb" => {
            // **Pain for All closure (Aura Phase 3 / S4 triggered
            // shape) — PainForAllETBEffect parse arm.** Unit variant;
            // no payload fields beyond `kind` (the damage source is
            // the Aura's `attached_to` host, read at resolve time;
            // the damage amount is that host's effective power; the
            // target is deterministic first-legal at resolve time).
            // Extra dict fields silently ignored for round-trip dict-
            // shape consistency (mirrors slice-19
            // `enchanted_creature_fights_target` unit-variant parse
            // arm shape verbatim).
            Ok(Effect::PainForAllETBEffect)
        }
        "pain_for_all_damage_echo" => {
            // **Pain for All closure (Aura Phase 3 / S4 triggered
            // shape) — PainForAllDamageEchoEffect parse arm.** Unit
            // variant; no payload fields beyond `kind` (the damage
            // amount is read from `state.resolving_combat_damage_amount`
            // — already plumbed engine-wide; the recipients are each
            // opponent of the resolving controller). Extra dict fields
            // silently ignored. Damage-feedback loop-guard: applies as
            // life-loss not damage-pipeline, so no DEALT_DAMAGE re-fire
            // (Python parity).
            Ok(Effect::PainForAllDamageEchoEffect)
        }
        "sacrifice_self_unless_void_active" => {
            // **Aura Phase 4 (Chorale of the Void substrate landing)
            // — SacrificeSelfUnlessVoidActive parse arm.** Unit
            // variant; no payload fields beyond `kind` (source-and-
            // controller are read from `state.current_resolve_source_
            // controller` + the source's bf membership at resolve
            // time, mirroring the Python `state.resolving_source`
            // convention at combinators.py:186). Extra dict fields
            // silently ignored for round-trip dict-shape consistency
            // (mirrors slice-19 `enchanted_creature_fights_target` /
            // Pain for All `pain_for_all_etb` unit-variant parse arm
            // shape verbatim).
            Ok(Effect::SacrificeSelfUnlessVoidActive)
        }
        "sacrifice_self_unless_tap_permanent" => {
            // 20th Rust slice (VVV Pin 2 closure) — Command Bridge ETB
            // "sacrifice this unless you tap an untapped permanent you
            // control". Unit variant; no payload fields beyond `kind`
            // (the source-and-controller are read from
            // `state.resolving_source` at resolve time, mirroring the
            // Python convention at `combinators.py:121`). Extra dict
            // fields silently ignored for round-trip dict-shape
            // consistency (mirrors slice-19 `enchanted_creature_fights_
            // target` / slice-16 `astelli_reclaimer_etb_effect` /
            // round-49 `gain_life_equal_to_your_life_total` unit-variant
            // parse arm shape verbatim).
            Ok(Effect::SacrificeSelfUnlessTapPermanent)
        }
        "each_opponent_exiles_card_from_hand_lightstall" => {
            // 20th Rust slice (XXX Pin 3 honest-stop closure) --
            // Lightstall Inquisitor each-opponent-exiles-card-from-
            // hand-with-cast-permission. Unit variant; no payload
            // fields beyond `kind`. Mirrors the slice-16 / slice-19
            // unit-variant parse arm shape verbatim. Extra dict fields
            // silently ignored.
            //
            // **Persistent-permission substrate honest-stopped.** The
            // resolve arm closes the each-opponent-exile portion
            // byte-equal to Python; the persistent
            // `cast_from_exile_with_lightstall_surcharge` per-object
            // flag + cast-from-exile legality + +{1} surcharge +
            // tapped-land-entry riders are NOT plumbed in this slice
            // (see the Effect variant docstring for the substrate-
            // scope breakdown and the BLOCKED_GENUINELY_MISSING
            // _INVENTORY §3 entry for the closure-cost estimate).
            Ok(Effect::EachOpponentExilesCardFromHandWithCastPermission)
        }
        "house_cartographer_survival" => {
            // 21st Rust slice (XXX Pin 1 closure) -- House Cartographer
            // reveal-until-land library walk. Unit variant; no payload
            // fields beyond `kind` (the library-walk state is transient,
            // computed fresh on every resolve). Mirrors the slice-16 /
            // slice-19 / slice-20 unit-variant parse arm shape verbatim.
            // Extra dict fields silently ignored for round-trip dict-
            // shape consistency.
            Ok(Effect::HouseCartographerSurvivalEffect)
        }
        "each_opponent_exiles_own_creature" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Single-
            // field variant — `link_to_source: bool` mirrors Python's
            // `link_to_source: bool = False` dataclass field. Default
            // False (matches Python).
            let link_to_source: bool = match dict.get_item("link_to_source")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "each_opponent_exiles_own_creature.link_to_source must be a bool (got: {e})"
                    ))
                })?,
                None => false,
            };
            Ok(Effect::EachOpponentExilesOwnCreature { link_to_source })
        }
        "reanimate_one_exiled_with_source_with_counters" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Single-
            // field variant — `counter_amount: i32` mirrors Python's
            // `counter_amount: int = 0` dataclass field. Default 0.
            let counter_amount: i32 = match dict.get_item("counter_amount")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "reanimate_one_exiled_with_source_with_counters.counter_amount must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::ReanimateOneExiledWithSourceWithCounters { counter_amount })
        }
        "sacrifice_self_preserving_exile_link" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant; no payload fields beyond `kind`. Extra fields
            // silently ignored for round-trip consistency.
            Ok(Effect::SacrificeSelfPreservingExileLink)
        }
        "grant_triggered_ability_ueot_requiem" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant on the Rust side — the Python `ability` field is
            // a `TriggeredAbility` dataclass whose Rust mirror is
            // deferred along with the broader grant-triggered-ability
            // registry. Bridge round-trip stays closed: the to_pydict
            // arm emits only `kind`, the Python bridge arm reconstructs
            // the fixed Requiem-shaped `GrantTriggeredAbilityUEOT(
            // ability=TriggeredAbility(trigger="this_dealt_damage",
            // effect=DrawCardsAndLoseLifeEqualToResolvingDamage()))`.
            // Extra fields silently ignored.
            Ok(Effect::GrantTriggeredAbilityUEOTRequiem)
        }
        "may_have_source_deal_damage_to_target" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Single-
            // field variant — `amount: i32` mirrors Python's
            // `amount: int = 1` dataclass field. Default 1.
            let amount: i32 = match dict.get_item("amount")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "may_have_source_deal_damage_to_target.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            Ok(Effect::MayHaveSourceDealDamageToTarget { amount })
        }
        "draw_cards_and_lose_life_equal_to_resolving_damage" => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant; no payload fields beyond `kind`. Extra fields
            // silently ignored.
            Ok(Effect::DrawCardsAndLoseLifeEqualToResolvingDamage)
        }
        "pump_target_artifact_or_creature_until_eot" => {
            // Round-28 broader-filter pump. Same schema as
            // pump_target_creature_until_eot — power_bonus +
            // toughness_bonus default to 0; negatives allowed.
            // Reroute Systems closure (2026-05-21): `granted_keywords`
            // optional list-of-strings, defaults to empty (back-
            // compat with every existing dict round-trip that omits
            // the field). Each entry pushed as a per-keyword
            // `UeotModifier::GrantKeyword` at resolve time.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_artifact_or_creature_until_eot.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_artifact_or_creature_until_eot.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let granted_keywords: Vec<String> = match dict.get_item("granted_keywords")? {
                Some(kw) => kw.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_target_artifact_or_creature_until_eot.granted_keywords must be a list of strings (got: {e})"
                    ))
                })?,
                None => Vec::new(),
            };
            Ok(Effect::PumpTargetArtifactOrCreatureUntilEOT {
                power_bonus,
                toughness_bonus,
                granted_keywords,
            })
        }
        "deal_damage_to_target_tapped_creature" => {
            // Reroute Systems closure (2026-05-21). Single-field
            // variant: `amount: i32`, REQUIRED (no default — the
            // bare "deals N damage" shape always carries an explicit
            // amount, mirrors round-28 `damage_each_opponent` /
            // `damage_each_creature` REQUIRED-amount precedents).
            // Negative `amount` rejected (CR 119.5 — negative damage
            // amounts are not legal effects).
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "deal_damage_to_target_tapped_creature.amount must be an integer (got: {e})"
                    ))
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "deal_damage_to_target_tapped_creature: 'amount' field is required",
                    ));
                }
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "deal_damage_to_target_tapped_creature.amount must be non-negative (got: {amount})"
                )));
            }
            Ok(Effect::DealDamageToTargetTappedCreature { amount })
        }
        "each_opponent_sacrifices_creature" => {
            // Unit variant; no fields.
            Ok(Effect::EachOpponentSacrificesCreature)
        }
        "untap_target_other_permanent" => {
            Ok(Effect::UntapTargetOtherPermanent)
        }
        "tap_another_target_creature" => {
            Ok(Effect::TapAnotherTargetCreature)
        }
        "return_all_creatures_from_your_graveyard_to_battlefield" => {
            Ok(Effect::ReturnAllCreaturesFromYourGraveyardToBattlefield)
        }
        "manifest_dread" => {
            // Round-28 ManifestDread first-slice. Python's
            // `ManifestDread(count: int = 1)` defaults to 1; we
            // mirror that. Negative count raises at parse time.
            // count != 1 is allowed at parse (so dict round-trip
            // works) but resolve-time-raises with a clear "deferred"
            // error so a typo doesn't silently no-op.
            let count: i32 = match dict.get_item("count")? {
                Some(c) => c.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "manifest_dread.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "manifest_dread: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::ManifestDread { count })
        }
        "manifest_dread_then_attach_self" => {
            // Round-29 ManifestDreadThenAttachSelf composition.
            // Schema mirrors `manifest_dread` exactly: optional `count`
            // field defaulting to 1, non-negative required. The
            // resolve arm gates count != 1 to deferred at the same
            // place as `Effect::ManifestDread`.
            let count: i32 = match dict.get_item("count")? {
                Some(c) => c.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "manifest_dread_then_attach_self.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "manifest_dread_then_attach_self: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::ManifestDreadThenAttachSelf { count })
        }
        "manifest_dread_then_put_plus_one_counters_on_it" => {
            // Round-29 ManifestDreadThenPutPlusOneCountersOnIt
            // (Moldering-Gym / Weight-Room family) composition.
            // Schema mirrors `manifest_dread`'s `count` plus a new
            // `counter_amount` rider — both default to 1, both must
            // be non-negative. Slimy Aquarium emits counter_amount=1,
            // Weight Room emits counter_amount=3.
            let count: i32 = match dict.get_item("count")? {
                Some(c) => c.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "manifest_dread_then_put_plus_one_counters_on_it.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "manifest_dread_then_put_plus_one_counters_on_it: count must be non-negative, got {count}"
                )));
            }
            let counter_amount: i32 = match dict.get_item("counter_amount")? {
                Some(c) => c.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "manifest_dread_then_put_plus_one_counters_on_it.counter_amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if counter_amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "manifest_dread_then_put_plus_one_counters_on_it: counter_amount must be non-negative, got {counter_amount}"
                )));
            }
            Ok(Effect::ManifestDreadThenPutPlusOneCountersOnIt {
                count,
                counter_amount,
            })
        }
        "set_base_pt_until_eot" => {
            // Required `power` and `toughness` integer fields. No
            // defaults — every Python card emission spells both out.
            let power: i32 = dict
                .get_item("power")?
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "set_base_pt_until_eot: missing required 'power' field",
                    )
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "set_base_pt_until_eot.power must be an integer (got: {e})"
                    ))
                })?;
            let toughness: i32 = dict
                .get_item("toughness")?
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "set_base_pt_until_eot: missing required 'toughness' field",
                    )
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "set_base_pt_until_eot.toughness must be an integer (got: {e})"
                    ))
                })?;
            Ok(Effect::SetBasePTUntilEOT { power, toughness })
        }
        "become_land_creature_ueot" => {
            // Round-28 manland animation. Required `power` and
            // `toughness` integer fields — every Python emission of
            // BecomeLandCreatureUEOT spells both out. Optional
            // `granted_subtype` is `None` when absent (matches
            // Python's `granted_subtype: str | None = None` default).
            // Negative power/toughness are accepted (no guard) — the
            // Python dataclass has no raise-on-negative either, and
            // a debuff-shaped manland would still register cleanly.
            let power: i32 = dict
                .get_item("power")?
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "become_land_creature_ueot: missing required 'power' field",
                    )
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "become_land_creature_ueot.power must be an integer (got: {e})"
                    ))
                })?;
            let toughness: i32 = dict
                .get_item("toughness")?
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "become_land_creature_ueot: missing required 'toughness' field",
                    )
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "become_land_creature_ueot.toughness must be an integer (got: {e})"
                    ))
                })?;
            let granted_subtype: Option<String> = match dict.get_item("granted_subtype")? {
                Some(v) if v.is_none() => None,
                Some(v) => {
                    let s: String = v.extract().map_err(|e| {
                        PyValueError::new_err(format!(
                            "become_land_creature_ueot.granted_subtype must be a string or None (got: {e})"
                        ))
                    })?;
                    if s.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                }
                None => None,
            };
            Ok(Effect::BecomeLandCreatureUEOT {
                power,
                toughness,
                granted_subtype,
            })
        }
        "grant_keyword_to_target_creature_until_eot" => {
            // Required `keyword` field: non-empty lowercase string.
            let keyword_raw = dict.get_item("keyword")?.ok_or_else(|| {
                PyValueError::new_err(
                    "grant_keyword_to_target_creature_until_eot: missing required 'keyword' field",
                )
            })?;
            let keyword: String = keyword_raw.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "grant_keyword_to_target_creature_until_eot.keyword must be a string (got: {e})"
                ))
            })?;
            if keyword.is_empty() {
                return Err(PyValueError::new_err(
                    "grant_keyword_to_target_creature_until_eot.keyword must be non-empty",
                ));
            }
            Ok(Effect::GrantKeywordToTargetCreatureUntilEOT { keyword })
        }
        "grant_keywords_to_target_creature_until_eot" => {
            // Round-15 (post-r14) rider-widened sibling of
            // `grant_keyword_to_target_creature_until_eot`. Required
            // `keywords` field: non-empty list of non-empty lowercase
            // strings. Optional `exclude_source` flag (default false)
            // for the "another target creature" sub-shape.
            let keywords_raw = dict.get_item("keywords")?.ok_or_else(|| {
                PyValueError::new_err(
                    "grant_keywords_to_target_creature_until_eot: missing required 'keywords' field",
                )
            })?;
            let keywords: Vec<String> = keywords_raw.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "grant_keywords_to_target_creature_until_eot.keywords must be a list of strings (got: {e})"
                ))
            })?;
            if keywords.is_empty() {
                return Err(PyValueError::new_err(
                    "grant_keywords_to_target_creature_until_eot.keywords must be non-empty",
                ));
            }
            for kw in &keywords {
                if kw.is_empty() {
                    return Err(PyValueError::new_err(
                        "grant_keywords_to_target_creature_until_eot.keywords entries must be non-empty strings",
                    ));
                }
            }
            let exclude_source: bool = match dict.get_item("exclude_source")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "grant_keywords_to_target_creature_until_eot.exclude_source must be a bool (got: {e})"
                    ))
                })?,
                None => false,
            };
            Ok(Effect::GrantKeywordsToTargetCreatureUntilEOT {
                keywords,
                exclude_source,
            })
        }
        "each_opponent_mills_cards" => {
            // Python's `EachOpponentMillsCards(count: int = 1)` dataclass.
            // Missing `count` defaults to 1; negative count raises.
            let count: i32 = match dict.get_item("count")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "each_opponent_mills_cards.count must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "each_opponent_mills_cards: count must be non-negative, got {count}"
                )));
            }
            Ok(Effect::EachOpponentMillsCards { count })
        }
        "exile_each_opponents_graveyard" => {
            // Python's `ExileEachOpponentsGraveyard` dataclass —
            // field-free unit effect (exile.py:260). No keys to read.
            Ok(Effect::ExileEachOpponentsGraveyard)
        }
        "if_you_do" => {
            // Round-28 IfYouDo first-slice: do_action implicitly
            // success-on-yes (i.e. IfYouDo(MayDo(X), Y) form). Both
            // nested effects required; no defaulting.
            let do_raw = dict.get_item("do_action")?.ok_or_else(|| {
                PyValueError::new_err(
                    "if_you_do: missing required 'do_action' field",
                )
            })?;
            let then_raw = dict.get_item("then_action")?.ok_or_else(|| {
                PyValueError::new_err(
                    "if_you_do: missing required 'then_action' field",
                )
            })?;
            let do_dict = do_raw.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "if_you_do.do_action must be a dict (effect-shape)",
                )
            })?;
            let then_dict = then_raw.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "if_you_do.then_action must be a dict (effect-shape)",
                )
            })?;
            let do_action = parse_effect(do_dict)?;
            let then_action = parse_effect(then_dict)?;
            Ok(Effect::IfYouDo {
                do_action: Box::new(do_action),
                then_action: Box::new(then_action),
            })
        }
        "choose_one_mode" => {
            // Round-28 ModalChooseOne first-slice: resolve-time pick
            // only. Required nested 'modes' field is a list of effect
            // dicts; recursively parse each. Empty modes list is
            // rejected at parse time so the resolve arm always has at
            // least one mode to fall through to.
            let nested = dict.get_item("modes")?.ok_or_else(|| {
                PyValueError::new_err(
                    "choose_one_mode: missing required 'modes' field (list of nested effect dicts)",
                )
            })?;
            let items: Vec<Bound<'_, PyAny>> = nested
                .try_iter()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "choose_one_mode.modes must be iterable (got: {e})"
                    ))
                })?
                .collect::<PyResult<Vec<_>>>()?;
            if items.is_empty() {
                return Err(PyValueError::new_err(
                    "choose_one_mode.modes must be non-empty (at least one mode required)",
                ));
            }
            let mut parsed_modes: Vec<Effect> = Vec::with_capacity(items.len());
            for item in items {
                let mode_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "choose_one_mode.modes entries must be dicts (effect-shape)",
                    )
                })?;
                parsed_modes.push(parse_effect(mode_dict)?);
            }
            // **scour-for-scrap-slice (2026-05-21).** Optional
            // `or_both: bool` field. Defaults to `false` so every pre-
            // widen registered ChooseOneMode parses byte-identical.
            // Scour for Scrap (and any future or_both Charm) emits
            // `{"kind": "choose_one_mode", "or_both": true, ...}`.
            let or_both: bool = match dict.get_item("or_both")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "choose_one_mode.or_both must be a bool (got: {e})"
                    ))
                })?,
                None => false,
            };
            Ok(Effect::ChooseOneMode { modes: parsed_modes, or_both })
        }
        "put_charge_counters_on_self" => {
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_charge_counters_on_self.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_charge_counters_on_self: amount must be non-negative, got {amount}"
                )));
            }
            let amount_from_x: bool = match dict.get_item("amount_from_x")? {
                Some(b) => b.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_charge_counters_on_self.amount_from_x must be a bool (got: {e})"
                    ))
                })?,
                None => false,
            };
            Ok(Effect::PutChargeCountersOnSelf { amount, amount_from_x })
        }
        "put_plus_one_counter_on_each_of_up_to_n_target_creatures" => {
            let max_n: i32 = match dict.get_item("max_n")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_each_of_up_to_n_target_creatures.max_n must be an integer (got: {e})"
                    ))
                })?,
                None => 2,
            };
            if max_n < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_each_of_up_to_n_target_creatures: max_n must be non-negative, got {max_n}"
                )));
            }
            Ok(Effect::PutPlusOneCounterOnEachOfUpToNTargetCreatures { max_n })
        }
        "conditional_on_tapped_creatures" => {
            // Required nested effect.
            let nested = dict.get_item("effect")?.ok_or_else(|| {
                PyValueError::new_err(
                    "conditional_on_tapped_creatures: missing required 'effect' field",
                )
            })?;
            let nested_dict = nested.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "conditional_on_tapped_creatures.effect must be a dict (effect-shape)",
                )
            })?;
            let inner = parse_effect(nested_dict)?;
            let min_count: i32 = match dict.get_item("min_count")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "conditional_on_tapped_creatures.min_count must be an integer (got: {e})"
                    ))
                })?,
                None => 2,
            };
            Ok(Effect::ConditionalOnTappedCreatures {
                effect: Box::new(inner),
                min_count,
            })
        }
        "conditional_if_void" => {
            // Required then-branch.
            let then_any = dict.get_item("then_effect")?.ok_or_else(|| {
                PyValueError::new_err(
                    "conditional_if_void: missing required 'then_effect' field",
                )
            })?;
            let then_dict = then_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "conditional_if_void.then_effect must be a dict (effect-shape)",
                )
            })?;
            let then_inner = parse_effect(then_dict)?;
            // Optional else-branch — None when key absent OR explicit Python None.
            let else_inner: Option<Effect> = match dict.get_item("else_effect")? {
                None => None,
                Some(v) if v.is_none() => None,
                Some(v) => {
                    let else_dict = v.downcast::<PyDict>().map_err(|_| {
                        PyValueError::new_err(
                            "conditional_if_void.else_effect must be a dict or None",
                        )
                    })?;
                    Some(parse_effect(else_dict)?)
                }
            };
            Ok(Effect::ConditionalIfVoid {
                then_effect: Box::new(then_inner),
                else_effect: else_inner.map(Box::new),
            })
        }
        "look_at_top_n_reveal_or_counter" => {
            // Round-28 Living-Phone family. Required `n` (Python's class
            // default is 5 for Pulsar Squadron Ace but we make it
            // required at parse to match the every-card-emits-it
            // convention — registrations omit only when the printed
            // body literally says "top 5", which they spell explicitly).
            //
            // Actually — Python's class HAS a default of n=5. Mirror
            // that so dict round-trip with omitted `n` works.
            let n: i32 = match dict.get_item("n")? {
                Some(v) => v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "look_at_top_n_reveal_or_counter.n must be an integer (got: {e})"
                    ))
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "look_at_top_n_reveal_or_counter: missing required 'n' field",
                    ));
                }
            };
            if n < 0 {
                return Err(PyValueError::new_err(format!(
                    "look_at_top_n_reveal_or_counter: n must be non-negative, got {n}"
                )));
            }
            let filter_name: String = match dict.get_item("filter_name")? {
                Some(s) => s.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "look_at_top_n_reveal_or_counter.filter_name must be a string (got: {e})"
                    ))
                })?,
                None => "spacecraft".to_string(), // Python class default
            };
            match filter_name.as_str() {
                "spacecraft" | "creature" | "artifact" => {}
                other => {
                    return Err(PyValueError::new_err(format!(
                        "look_at_top_n_reveal_or_counter: filter_name must be one of \
                         'spacecraft', 'creature', 'artifact' (got: {other:?})"
                    )));
                }
            }
            let max_power: i32 = match dict.get_item("max_power")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "look_at_top_n_reveal_or_counter.max_power must be an integer (got: {e})"
                    ))
                })?,
                None => 0, // Python class default — 0 means "no power cap"
            };
            if max_power < 0 {
                return Err(PyValueError::new_err(format!(
                    "look_at_top_n_reveal_or_counter: max_power must be non-negative, got {max_power}"
                )));
            }
            let counter_on_miss: bool = match dict.get_item("counter_on_miss")? {
                Some(b) => b.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "look_at_top_n_reveal_or_counter.counter_on_miss must be a bool (got: {e})"
                    ))
                })?,
                None => true, // Python class default — Pulsar Squadron Ace shape
            };
            Ok(Effect::LookAtTopNRevealOrCounter {
                n,
                filter_name,
                max_power,
                counter_on_miss,
            })
        }
        "pump_creatures_opponents_control_until_eot" => {
            // Mirror schema of pump_creatures_you_control_until_eot.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_opponents_control_until_eot.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_opponents_control_until_eot.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpCreaturesOpponentsControlUntilEOT { power_bonus, toughness_bonus })
        }
        "pump_creatures_opponents_control_until_caster_next_turn" => {
            // Mouth-of-the-Storm sibling — same field shape as the
            // bare-EOT arm above; the discriminator alone carries
            // the extended-duration distinction (Python translates
            // it to `until_caster_next_turn=True`). The cleanup-
            // gate divergence is documented on the variant + in
            // SEARCH_LIMITATIONS.md.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_opponents_control_until_caster_next_turn.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_opponents_control_until_caster_next_turn.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpCreaturesOpponentsControlUntilCasterNextTurn {
                power_bonus,
                toughness_bonus,
            })
        }
        "search_library_for_card" => {
            // Python's SearchLibraryForCard(filter_name, max_mv=0,
            // may_fail=True).
            let filter_name: String = match dict.get_item("filter_name")? {
                Some(s) => s.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "search_library_for_card.filter_name must be a string (got: {e})"
                    ))
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "search_library_for_card: missing required 'filter_name' field",
                    ));
                }
            };
            let max_mv: i32 = match dict.get_item("max_mv")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "search_library_for_card.max_mv must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let may_fail: bool = match dict.get_item("may_fail")? {
                Some(m) => m.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "search_library_for_card.may_fail must be a bool (got: {e})"
                    ))
                })?,
                None => true,
            };
            Ok(Effect::SearchLibraryForCard {
                filter_name,
                max_mv,
                may_fail,
            })
        }
        "add_n_mana_of_chosen_color" => {
            // Python's AddNManaOfChosenColor(amount: int,
            // options: tuple[str, ...] = ("W","U","B","R","G")).
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "add_n_mana_of_chosen_color.amount must be an integer (got: {e})"
                    ))
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "add_n_mana_of_chosen_color: missing required 'amount' field",
                    ));
                }
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "add_n_mana_of_chosen_color: amount must be non-negative, got {amount}"
                )));
            }
            let options: Vec<String> = match dict.get_item("options")? {
                Some(items) => {
                    let raw: Vec<String> = items.extract().map_err(|e| {
                        PyValueError::new_err(format!(
                            "add_n_mana_of_chosen_color.options must be a list of color strings (got: {e})"
                        ))
                    })?;
                    if raw.is_empty() {
                        return Err(PyValueError::new_err(
                            "add_n_mana_of_chosen_color: options must be non-empty",
                        ));
                    }
                    for c in &raw {
                        if !matches!(c.as_str(), "W" | "U" | "B" | "R" | "G" | "C") {
                            return Err(PyValueError::new_err(format!(
                                "add_n_mana_of_chosen_color: unknown color {c:?}; valid: W, U, B, R, G, C"
                            )));
                        }
                    }
                    raw
                }
                None => vec![
                    "W".into(),
                    "U".into(),
                    "B".into(),
                    "R".into(),
                    "G".into(),
                ],
            };
            Ok(Effect::AddNManaOfChosenColor { amount, options })
        }
        "put_plus_one_counter_on_creature_you_control" => {
            // Python's PutPlusOneCounterOnCreatureYouControl uses
            // PutPlusOneCounters base default of amount=1.
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_creature_you_control.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_creature_you_control: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::PutPlusOneCounterOnCreatureYouControl { amount })
        }
        "put_plus_one_counter_on_target_creature" => {
            // **PPP Pin 2 (10th Rust slice).** Python's
            // `PutPlusOneCounterOnTargetCreature(amount: int = 1,
            // attacking_only: bool = False)` (counters.py:637).
            // Mirrors the sibling
            // `put_plus_one_counter_on_creature_you_control` arm
            // shape for the `amount` default + non-negative contract,
            // adds optional `attacking_only` bool (default false) for
            // the Hospital Room rider.
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_target_creature.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_target_creature: amount must be non-negative, got {amount}"
                )));
            }
            let attacking_only: bool = match dict.get_item("attacking_only")? {
                Some(b) => b.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_target_creature.attacking_only must be a bool (got: {e})"
                    ))
                })?,
                None => false,
            };
            Ok(Effect::PutPlusOneCounterOnTargetCreature {
                amount,
                attacking_only,
            })
        }
        "put_plus_one_counter_on_opponents_creature" => {
            // Tier-C target-seating lever-a (Sunset Saboteur). Python's
            // PutPlusOneCounterOnOpponentsCreature uses the
            // PutPlusOneCounters base default of amount=1. Same
            // tolerate-missing / non-negative contract as the
            // you-control sibling arm above.
            let amount: i32 = match dict.get_item("amount")? {
                Some(a) => a.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_plus_one_counter_on_opponents_creature.amount must be an integer (got: {e})"
                    ))
                })?,
                None => 1,
            };
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_plus_one_counter_on_opponents_creature: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::PutPlusOneCounterOnOpponentsCreature { amount })
        }
        "return_target_permanent_you_control" => {
            // Tier-C target-seating lever-a (Exosuit Savior). A unit
            // variant — the filter (own ∧ not-player ∧ not-source) is
            // structural to the seat's per-variant scan + the
            // resolve-time re-check, not a dict payload field (the
            // recognizer's bare-shape gate enforces the single
            // implicit-target contract). Extra dict fields are
            // tolerated/ignored (same shape as the other unit
            // variants).
            Ok(Effect::ReturnTargetPermanentYouControlToOwnersHand)
        }
        "pump_creatures_you_control_until_eot" => {
            // Round-28 anthem-shape UEOT effect. Mirrors the
            // PumpSelfUntilEOT / PumpTargetCreatureUntilEOT schema:
            // power_bonus + toughness_bonus default to 0, negatives
            // allowed.
            let power_bonus: i32 = match dict.get_item("power_bonus")? {
                Some(p) => p.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_you_control_until_eot.power_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            let toughness_bonus: i32 = match dict.get_item("toughness_bonus")? {
                Some(t) => t.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "pump_creatures_you_control_until_eot.toughness_bonus must be an integer (got: {e})"
                    ))
                })?,
                None => 0,
            };
            Ok(Effect::PumpCreaturesYouControlUntilEOT { power_bonus, toughness_bonus })
        }
        "deal_damage_to_opponent_creature" => {
            // Mirrors Python's `DealDamageToOpponentCreature(DealDamage)`
            // (effects/generic/damage.py) — same shape as
            // `deal_damage_any_target` but the target filter is
            // `IsCreature & ControlledByOpponent`. Negative `amount`
            // surfaces at register time (defense-in-depth; the resolve
            // arm re-checks).
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "deal_damage_to_opponent_creature: amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::DealDamageToOpponentCreature { amount })
        }
        "optional_mana_cost_then_effect" => {
            // Round-29 OptionalManaCostThenEffect — Kavaron-Harrier /
            // Oreplate-Pangolin family. Required `cost` (mana cost
            // dict) and `inner_effect` (nested effect dict).
            let cost_any = dict.get_item("cost")?.ok_or_else(|| {
                PyValueError::new_err(
                    "optional_mana_cost_then_effect: missing required 'cost' field",
                )
            })?;
            let cost_dict = cost_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "optional_mana_cost_then_effect.cost must be a dict",
                )
            })?;
            let cost = parse_mana_cost(cost_dict)?;
            let inner_any = dict.get_item("inner_effect")?.ok_or_else(|| {
                PyValueError::new_err(
                    "optional_mana_cost_then_effect: missing required 'inner_effect' field",
                )
            })?;
            let inner_dict = inner_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "optional_mana_cost_then_effect.inner_effect must be a dict (effect-shape)",
                )
            })?;
            let inner = parse_effect(inner_dict)?;
            Ok(Effect::OptionalManaCostThenEffect {
                cost,
                inner_effect: Box::new(inner),
            })
        }
        "draw_cards_equal_to_resolving_combat_damage" => {
            // Round-28 freq-2 closure — Starwinder / Fear-of-Failed-
            // Tests shape. Unit variant; reads
            // state.resolving_combat_damage_amount at resolve time.
            // No fields to parse.
            Ok(Effect::DrawCardsEqualToResolvingCombatDamage)
        }
        "sequence" => {
            // Round-28 highest-frequency-class coverage. Mirrors
            // Python's Sequence(effects: tuple[Effect, ...]).
            // Field `effects` is a list of nested effect dicts;
            // recursively parse each via this same parse_effect
            // function. Empty list is legal (no-op Sequence).
            let nested = dict.get_item("effects")?.ok_or_else(|| {
                PyValueError::new_err(
                    "sequence: missing required 'effects' field (list of nested effect dicts)",
                )
            })?;
            let items: Vec<Bound<'_, PyAny>> = nested
                .try_iter()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "sequence.effects must be iterable (got: {e})"
                    ))
                })?
                .collect::<PyResult<Vec<_>>>()?;
            let mut children: Vec<Effect> = Vec::with_capacity(items.len());
            for item in items {
                let child_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "sequence.effects entries must be dicts (effect-shape)",
                    )
                })?;
                children.push(parse_effect(child_dict)?);
            }
            Ok(Effect::Sequence { effects: children })
        }
        // **BBB downstream complement — 7th Rust slice (GGG Pin 2).**
        // DefendingPlayerLosesHalfLifeRoundedUp — zero-field effect.
        // Dict shape: `{"kind": "defending_player_loses_half_life_rounded_up"}`.
        // No required sub-fields. Extra fields silently ignored for
        // round-trip dict-shape consistency (mirrors every unit-variant
        // parse arm precedent — ConditionalIfVoid / WarpExileSelf etc.).
        // Mirrors Python's `DefendingPlayerLosesHalfLifeRoundedUp()`
        // (`effects/generic/life.py:570`).
        "defending_player_loses_half_life_rounded_up" => {
            Ok(Effect::DefendingPlayerLosesHalfLifeRoundedUp)
        }
        // **BBB downstream complement — 7th Rust slice (XX Pin 1).**
        // ConditionalIfYouGainedLifeAtLeast — threshold (int) + inner
        // (nested effect dict). Reads `threshold` from dict; recursively
        // parses `inner` via `parse_effect`. Mirrors Python's
        // `ConditionalIfYouGainedLifeAtLeast(threshold, inner)`
        // (`effects/combinators.py:966`) and the Python bridge arm in
        // `cross_engine.py:8694`.
        "conditional_if_you_gained_life_at_least" => {
            let threshold: i32 = require(dict, "threshold")?.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "conditional_if_you_gained_life_at_least.threshold must be an int (got: {e})"
                ))
            })?;
            let inner_any = require(dict, "inner")?;
            let inner_dict = inner_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "conditional_if_you_gained_life_at_least.inner must be a dict"
                )
            })?;
            let inner = parse_effect(inner_dict)?;
            Ok(Effect::ConditionalIfYouGainedLifeAtLeast {
                threshold,
                inner: Box::new(inner),
            })
        }
        // **BBB downstream complement — 7th Rust slice (XX Pin 2).**
        // ConditionalIfSourcePowerAtLeast — threshold (int, default=4)
        // + effect (nested effect dict, accepting both "inner" and
        // "effect" keys for forward-compat with the Python bridge's
        // field-name skew). Mirrors Python's
        // `ConditionalIfSourcePowerAtLeast(effect, threshold=4)`
        // (`effects/combinators.py:1055`) and the Python bridge arm in
        // `cross_engine.py:8716`.
        "conditional_if_source_power_at_least" => {
            let threshold: i32 = dict
                .get_item("threshold")
                .map_err(|e| PyValueError::new_err(format!("dict access error: {e}")))?
                .map(|v| v.extract::<i32>())
                .transpose()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "conditional_if_source_power_at_least.threshold must be an int (got: {e})"
                    ))
                })?
                .unwrap_or(4);
            // Accept both "effect" and "inner" dict keys (mirrors the
            // Python bridge's `nested_key = "effect" if "effect" in eff_dict
            // else "inner"` forward-compat logic).
            let effect_any = if dict
                .get_item("effect")
                .map_err(|e| PyValueError::new_err(format!("dict access error: {e}")))?
                .is_some()
            {
                require(dict, "effect")?
            } else {
                require(dict, "inner")?
            };
            let effect_dict = effect_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "conditional_if_source_power_at_least effect/inner must be a dict"
                )
            })?;
            let effect = parse_effect(effect_dict)?;
            Ok(Effect::ConditionalIfSourcePowerAtLeast {
                threshold,
                effect: Box::new(effect),
            })
        }
        // **EEE downstream complement — 8th Rust slice (JJJ Pin 1).**
        // DyadrineAttackEffect — zero-field bespoke effect for Dyadrine,
        // Synthesis Amalgam's attack trigger.  Dict shape:
        // `{"kind": "dyadrine_attack_effect"}` (no extra fields; mirrors
        // the zero-field Python dataclass).
        //
        // Mirrors Python's `DyadrineAttackEffect()` (bespoke.py).
        // Python bridge arm: `_ta_dict_to_python_effect` kind ==
        // "dyadrine_attack_effect" (wired by EEE bridge slice, round 17).
        "dyadrine_attack_effect" => {
            Ok(Effect::DyadrineAttackEffect)
        }
        // **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
        // PutPoisonCountersOnTargetPlayer — give `amount` poison counters
        // to the target player.  Dict shape: `{"kind":
        // "put_poison_counters_on_target_player", "amount": <int>}`.
        // `amount` defaults to 1 when the key is absent (mirrors
        // Python's `PutPoisonCountersOnTargetPlayer(amount: int = 1)`).
        //
        // Mirrors Python's `PutPoisonCountersOnTargetPlayer`
        // (effects/generic/counters.py).  Python bridge arm wired by
        // EEE bridge slice, round 17.
        "put_poison_counters_on_target_player" => {
            let amount: i32 = dict
                .get_item("amount")
                .map_err(|e| PyValueError::new_err(format!("dict access error: {e}")))?
                .map(|v| v.extract::<i32>())
                .transpose()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "put_poison_counters_on_target_player.amount must be an int (got: {e})"
                    ))
                })?
                .unwrap_or(1);
            Ok(Effect::PutPoisonCountersOnTargetPlayer { amount })
        }
        // **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for
        // all-untargeted modal cards, 2026-05-21).** Parse arm for
        // `ExileAllArtifactsWithManaValueAtMost { max_mv }`. `max_mv`
        // is REQUIRED — mirrors Python's regex-driven parser which
        // always captures the digit, and the bridge arm at
        // `cross_engine.py::exile_all_artifacts_with_mana_value_at_
        // most` which also requires it. Negative `max_mv` is rejected
        // at parse time (schema invariant — MV is non-negative per
        // CR 202.3; mirrors the DamageEachOpponent / DamageEachCreature
        // raise-on-negative shape).
        "exile_all_artifacts_with_mana_value_at_most" => {
            let max_mv: i32 = require(dict, "max_mv")?.extract()?;
            if max_mv < 0 {
                return Err(PyValueError::new_err(format!(
                    "exile_all_artifacts_with_mana_value_at_most.max_mv must be non-negative, got {max_mv}"
                )));
            }
            Ok(Effect::ExileAllArtifactsWithManaValueAtMost { max_mv })
        }
        // **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
        // 2026-05-21).** Parse arm for
        // `PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { amount }`.
        // `amount` is REQUIRED — mirrors Python's regex-driven parser
        // at `parser/effects.py:3002-3005` which captures the digit
        // (or spelled-out cardinal); the matching bridge arm in
        // `cross_engine.py` also requires it. Negative `amount` is
        // rejected at parse time (mirrors the
        // `exile_all_artifacts_with_mana_value_at_most` /
        // `damage_each_opponent` raise-on-negative shape).
        "put_charge_counters_on_target_spacecraft_or_planet_you_control" => {
            let amount: i32 = require(dict, "amount")?.extract()?;
            if amount < 0 {
                return Err(PyValueError::new_err(format!(
                    "put_charge_counters_on_target_spacecraft_or_planet_you_control.amount must be non-negative, got {amount}"
                )));
            }
            Ok(Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { amount })
        }
        // **Multi-target admit-gate Phase 4 widen (2026-05-21).** Parse
        // arm for `ReturnUpToNSmallCreaturesFromGraveyardWithCounters
        // { max_n, max_total_mv }`. Both fields REQUIRED — the matching
        // bridge arm in `cross_engine.py` also requires both. Negative
        // values rejected at parse time (schema invariant; mirrors the
        // `exile_all_artifacts_with_mana_value_at_most` /
        // `damage_each_opponent` raise-on-negative precedent).
        "return_up_to_n_small_creatures_from_graveyard_with_counters" => {
            let max_n: i32 = require(dict, "max_n")?.extract()?;
            let max_total_mv: i32 = require(dict, "max_total_mv")?.extract()?;
            if max_n < 0 {
                return Err(PyValueError::new_err(format!(
                    "return_up_to_n_small_creatures_from_graveyard_with_counters.max_n must be non-negative, got {max_n}"
                )));
            }
            if max_total_mv < 0 {
                return Err(PyValueError::new_err(format!(
                    "return_up_to_n_small_creatures_from_graveyard_with_counters.max_total_mv must be non-negative, got {max_total_mv}"
                )));
            }
            Ok(Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters {
                max_n,
                max_total_mv,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown effect kind {other:?}; valid: deal_damage, draw_cards, gain_life, return_to_hand, mill, mill_target_player, exile_target, tap_target, untap_target, sacrifice, add_counters, discard_card, scry_n, destroy_target, damage_each_opponent, damage_each_creature, create_token, add_mana_to_pool, search_library_for_basic_land_tapped, put_plus_one_counter_on_self, surveil, counter_target_spell, return_creature_from_your_graveyard_to_battlefield, discard_self, each_opponent_loses_life, lose_life, target_opponent_loses_life, each_opponent_discards, destroy_all_creatures, destroy_target_nonland_permanent, destroy_target_artifact, destroy_target_land, focus_fire_effect, deal_damage_any_target, pump_self_until_eot, pump_target_creature_until_eot, sequence, pump_creatures_you_control_until_eot, return_creature_from_your_graveyard_to_hand, put_plus_one_counter_on_each_creature_you_control, put_plus_one_counter_on_creature_you_control, put_plus_one_counter_on_opponents_creature, return_target_permanent_you_control, add_n_mana_of_chosen_color, search_library_for_card, pump_creatures_opponents_control_until_eot, put_card_from_graveyard_on_bottom_of_library, move_self_counters_to_target_creature_you_control, deal_damage_equal_to_source_power_any_target, tap_target_artifact_or_creature, conditional_on_tapped_creatures, put_plus_one_counter_on_each_of_up_to_n_target_creatures, put_charge_counters_on_self, create_token_copy_of_target_creature_you_control, optional_sac_artifact_for_effect, owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library, attach_to_target_creature_you_control, destroy_enchanted_permanent, tap_enchanted_permanent, put_stun_counter_on_tapped_creature, tap_target_opponent_permanent_and_stun_it, exile_target_creature_until_source_leaves, exile_opponents_nonland_permanent_until_leaves, exile_opponents_nonland_permanent_with_max_mv_until_leaves, may_do, look_top_n_put_one_in_hand_rest_to_graveyard, choose_one_mode, if_you_do, each_opponent_mills_cards, pump_target_artifact_or_creature_until_eot, each_opponent_sacrifices_creature, grant_keyword_to_target_creature_until_eot, grant_keywords_to_target_creature_until_eot, untap_target_other_permanent, tap_another_target_creature, return_all_creatures_from_your_graveyard_to_battlefield, set_base_pt_until_eot, manifest_dread, manifest_dread_then_attach_self, manifest_dread_then_put_plus_one_counters_on_it, conditional_if_void, deal_damage_to_opponent_creature, optional_mana_cost_then_effect, draw_cards_equal_to_resolving_combat_damage, double_target_creatures_power_ueot, deal_damage_to_opponent_creature_by_artifact_count, put_plus_one_counters_on_self_per_other_creature_or_artifact, surveil_x_equal_to_artifact_count, deal_damage_to_opponent_creature_by_cards_in_hand, deal_damage_to_opponent_creature_by_lands_controlled, deal_damage_to_opponent_creature_by_creatures_you_control, mill_target_player_equal_to_your_life_total, create_token_each_player, each_opponent_loses_life_equal_to_artifact_count, gain_life_equal_to_artifact_count, deal_damage_to_opponent_creature_by_cards_in_graveyard, deal_damage_to_opponent_creature_by_life_total, exile_each_opponents_graveyard, diplomatic_relations_effect, defending_player_loses_half_life_rounded_up, conditional_if_you_gained_life_at_least, conditional_if_source_power_at_least, dyadrine_attack_effect, put_poison_counters_on_target_player, exile_all_artifacts_with_mana_value_at_most, deal_damage_to_target_tapped_creature, put_charge_counters_on_target_spacecraft_or_planet_you_control, return_up_to_n_small_creatures_from_graveyard_with_counters"
        ))),
    }
}

/// Validate that a string is one of the closed mana-color vocab
/// (W/U/B/R/G/C). Returns the input on success; raises ValueError
/// otherwise. Shared by `parse_mana_produced` and is callable from
/// the `mana_produced` field validator.
fn validate_mana_symbol(sym: &str) -> PyResult<()> {
    match sym {
        "W" | "U" | "B" | "R" | "G" | "C" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "unknown mana symbol {other:?} in mana_produced; valid: W, U, B, R, G, C"
        ))),
    }
}

fn parse_mana_produced(any: Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for item in any.try_iter().map_err(|_| {
        PyValueError::new_err("mana_produced must be an iterable of color strings")
    })? {
        let item = item?;
        let s: String = item.extract().map_err(|_| {
            PyValueError::new_err("mana_produced entries must be strings")
        })?;
        validate_mana_symbol(&s)?;
        out.push(s);
    }
    Ok(out)
}

/// CE20 follow-up — bridge legacy `mana_produced` into synthesized
/// `ActivatedAbility` entries on the printing's `activated_abilities`
/// Vec. When `mana_produced.is_some()`, prepend **one synthesized
/// `is_mana_ability` entry PER produced color**, in the SAME order
/// the symbols appear in `mana_produced`:
///
///   cost:   ManaCost::default() (free) + tap_self=true
///   effect: AddManaToPool { color: <c>, amount: 1 } — one entry per
///           color symbol, NOT aggregated.
///   is_mana_ability: true (CR 605.1 — produces mana, doesn't target).
///
/// `ability_index=0` = first color, `ability_index=1` = second, …
/// (then any explicit registry-supplied entries at indices >= N).
///
/// **Round-62 N1 — per-color shape (the cross-engine fix).** Before
/// round-62 this synthesized ONE ability whose effect aggregated ALL
/// colors (a flat `AddManaToPool` for a single color, a
/// `Sequence(Add c1, Add c2, …)` for multi-color). That disagreed
/// with the Python bridge `cross_engine._card_def_to_python_printing`,
/// which builds **one `ActivatedAbility` per color** in
/// `mana_produced` order (idx0=Add c1, idx1=Add c2). For a
/// single-color land (every basic) the two shapes were bit-equal
/// (one ability adding the one color), so the divergence was
/// invisible — until round-60 admitted the 5 EoE pay-life shocklands
/// (2-color `mana_produced`) into the production bench. Round-61 N3's
/// post-admit sweep caught it: `activate_ability(dual_land, 0)` made
/// Python add 1 color but Rust add BOTH (the whole `Sequence`) → a
/// spurious extra mana in the Rust pool (4 FAIL_DIVERGENCE / 112
/// games, every divergent land a 2-color shockland including U). The
/// CR-correct shape is one mana ability per color (the player chooses
/// which color by picking the ability index); Python was already
/// correct. This rewrite makes Rust emit the same per-color shape so
/// `activate_ability(land, k)` adds exactly the k-th color on BOTH
/// engines. Documented: `SEARCH_LIMITATIONS.md` shockland entry
/// (CLOSED round-62). Single-color lands stay bit-identical (one
/// ability, one color) — verified by the Forest/Llanowar synthetic
/// callers in `state.rs` and the round-62 Rust regression pins.
///
/// Note: `mana_produced=["G","G"]` now yields TWO `{T}: Add G`
/// abilities (one per entry — matches Python's
/// `for color in card.mana_produced`), not one `Add G amount=2`.
/// No real bridge printing carries a duplicate-color `mana_produced`
/// (`_printing_to_card_def` builds it from a `frozenset.produces`,
/// so entries are distinct), but per-entry emission is the
/// Python-exact shape and is the contract both engines now share.
///
/// Shared between `parse_printing` (dict-input registry path) and
/// `make_synthetic_state` (the test-fixture path in `state.rs`) so
/// both surfaces get the same bridge synthesis without duplicating
/// the per-color logic.
pub fn apply_mana_produced_bridge(
    mana_produced: &Option<Vec<String>>,
    existing: Vec<ActivatedAbility>,
) -> PyResult<Vec<ActivatedAbility>> {
    let Some(symbols) = mana_produced else {
        return Ok(existing);
    };
    // One synthesized is_mana_ability ability PER color symbol, in
    // input order — matching Python's
    // `_card_def_to_python_printing` `for color in card.mana_produced`
    // loop exactly (idx0=Add c1, idx1=Add c2, …). For shocklands
    // `trajectory._printing_to_card_def` passes
    // `tuple(sorted(printing.produces))`, so both engines iterate the
    // alphabetically-sorted dual (Watery Grave → idx0=Add B,
    // idx1=Add U); the order parity is the input-order parity.
    let mut bridge_abilities: Vec<ActivatedAbility> =
        Vec::with_capacity(symbols.len());
    for sym in symbols {
        match sym.as_str() {
            "W" | "U" | "B" | "R" | "G" | "C" => {}
            other => {
                // `parse_mana_produced` already rejected unknown
                // symbols at parse time; raise loud as a paranoid
                // double-check that also catches direct callers from
                // `state.rs` that bypass `parse_mana_produced`.
                return Err(PyValueError::new_err(format!(
                    "bridge synth: unknown mana_produced symbol \
                     {other:?}; expected W/U/B/R/G/C"
                )));
            }
        }
        bridge_abilities.push(ActivatedAbility {
            cost: ActivationCost {
                mana: ManaCost::default(),
                tap_self: true,
                sacrifice_self: false,
                requires_target_creature_you_control: false,
                sacrifice_another_creature_or_artifact: false,
                sacrifice_a_land: false,
                sacrifice_another_artifact: false,
                sacrifice_food: false,
                sacrifice_a_desert: false,
                loyalty: 0,
            },
            effect: Effect::AddManaToPool {
                color: sym.clone(),
                amount: 1,
            },
            is_mana_ability: true,
            tier_threshold: 0,
        });
    }
    // Prepend so ability_index=0..N-1 refer to the per-color bridged
    // mana abilities (Python's idx0/idx1 ordering); any explicit
    // registry-supplied entries follow at indices >= N. Every
    // pre-bridge ActivateAbility(forest, 0) caller keeps working
    // bit-equal (single-color → exactly one ability at index 0).
    let mut bridged =
        Vec::with_capacity(existing.len() + bridge_abilities.len());
    bridged.append(&mut bridge_abilities);
    let mut existing = existing;
    bridged.append(&mut existing);
    Ok(bridged)
}

pub fn parse_printing(dict: &Bound<'_, PyDict>) -> PyResult<CardPrinting> {
    let name: String = require(dict, "name")?.extract()?;
    let mana_cost = match dict.get_item("mana_cost")? {
        Some(mc_any) => {
            let mc_dict = mc_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("mana_cost must be a dict")
            })?;
            parse_mana_cost(mc_dict)?
        }
        None => ManaCost::default(),
    };
    // warp_cost substrate (Phase 1, round-63). Same dict shape as
    // `mana_cost` (`{"generic":N,"R":1}`). Missing key -> `None`
    // (back-compat — every pre-Phase-1 registry dict omits it). This
    // `match dict.get_item(...)? { Some => ..., None => default }`
    // shape IS the `#[serde(default)]`-equivalent the existing
    // optional fields all use; no serde derive needed (CardPrinting
    // is built via parse_printing from a PyDict, not deserialized).
    let warp_cost = match dict.get_item("warp_cost")? {
        Some(wc_any) => {
            let wc_dict = wc_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("warp_cost must be a dict")
            })?;
            Some(parse_mana_cost(wc_dict)?)
        }
        None => None,
    };
    // warp_cost substrate (Phase 2): Timeline Culler's two riders.
    // Same back-compat `match get_item { Some => extract, None =>
    // default }` shape as `warp_cost` above (no serde derive). Missing
    // key -> 0 / false (every pre-Phase-2 registry dict omits both;
    // build_rust_state only emits the keys for Timeline Culler).
    let warp_additional_life_cost = match dict
        .get_item("warp_additional_life_cost")?
    {
        Some(v) => v.extract::<u8>()?,
        None => 0,
    };
    let warp_castable_from_graveyard = match dict
        .get_item("warp_castable_from_graveyard")?
    {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    // legal_decisions mana-awareness Slice 3 — cast-time
    // additional-cost flags (Python `_additional_costs_payable`,
    // `legal.py:117-165`). Same back-compat
    // `match get_item { Some => extract, None => false }` shape as
    // `warp_castable_from_graveyard`. Missing key -> false (every
    // pre-Slice-3 registry dict omits these; the 4 EoE
    // additional-cost cards are all bridge-DECLINED so build_rust_state
    // never emits the keys on the live cast surface — the design
    // doc §5 verified-zero-incidence prediction).
    let additional_cost_sacrifice_creature_or_artifact = match dict
        .get_item("additional_cost_sacrifice_creature_or_artifact")?
    {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    let additional_cost_sacrifice_creature_or_enchantment = match dict
        .get_item("additional_cost_sacrifice_creature_or_enchantment")?
    {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    let additional_cost_choose_your_creature = match dict
        .get_item("additional_cost_choose_your_creature")?
    {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    let effects = match dict.get_item("effects")? {
        Some(eff_any) => {
            let mut out = Vec::new();
            for item in eff_any.try_iter()? {
                let item = item?;
                let eff_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each effect must be a dict")
                })?;
                out.push(parse_effect(eff_dict)?);
            }
            out
        }
        None => Vec::new(),
    };
    let power = match dict.get_item("power")? {
        Some(p) => Some(p.extract::<i16>().map_err(|e| {
            PyValueError::new_err(format!(
                "power must be an integer (got: {e})"
            ))
        })?),
        None => None,
    };
    let toughness = match dict.get_item("toughness")? {
        Some(t) => Some(t.extract::<i16>().map_err(|e| {
            PyValueError::new_err(format!(
                "toughness must be an integer (got: {e})"
            ))
        })?),
        None => None,
    };
    let mana_produced = match dict.get_item("mana_produced")? {
        Some(mp_any) => Some(parse_mana_produced(mp_any)?),
        None => None,
    };
    let card_types = match dict.get_item("card_types")? {
        Some(ct_any) => {
            let iter = ct_any.try_iter().map_err(|_| {
                PyValueError::new_err("card_types must be an iterable of type strings")
            })?;
            let mut out = Vec::new();
            for item in iter {
                let s: String = item?.extract().map_err(|_| {
                    PyValueError::new_err("card_types entries must be strings")
                })?;
                out.push(s);
            }
            out
        }
        None => Vec::new(),
    };
    // Subtypes substrate (Slice 1): `subtypes` defaults to empty Vec
    // when absent (back-compat — every pre-substrate registry dict
    // implicitly omits the key). Stored as Title Case strings exactly
    // as printed; the consumer side does case-sensitive `contains`.
    let subtypes = match dict.get_item("subtypes")? {
        Some(st_any) => {
            let iter = st_any.try_iter().map_err(|_| {
                PyValueError::new_err(
                    "subtypes must be an iterable of subtype strings",
                )
            })?;
            let mut out = Vec::new();
            for item in iter {
                let s: String = item?.extract().map_err(|_| {
                    PyValueError::new_err("subtypes entries must be strings")
                })?;
                out.push(s);
            }
            out
        }
        None => Vec::new(),
    };
    // Supertypes substrate (Round 30): `supertypes` defaults to empty
    // Vec when absent (back-compat — every pre-Round-30 registry dict
    // implicitly omits the key). Same parse shape as `subtypes`.
    // Stored as Title Case strings exactly as printed ("Basic",
    // "Legendary", "Snow", "World", "Token"); consumers do
    // case-sensitive `contains`.
    let supertypes = match dict.get_item("supertypes")? {
        Some(st_any) => {
            let iter = st_any.try_iter().map_err(|_| {
                PyValueError::new_err(
                    "supertypes must be an iterable of supertype strings",
                )
            })?;
            let mut out = Vec::new();
            for item in iter {
                let s: String = item?.extract().map_err(|_| {
                    PyValueError::new_err("supertypes entries must be strings")
                })?;
                out.push(s);
            }
            out
        }
        None => Vec::new(),
    };
    let keywords = match dict.get_item("keywords")? {
        Some(kw_any) => {
            let iter = kw_any.try_iter().map_err(|_| {
                PyValueError::new_err("keywords must be an iterable of keyword strings")
            })?;
            let mut out = Vec::new();
            for item in iter {
                let s: String = item?.extract().map_err(|_| {
                    PyValueError::new_err("keywords entries must be strings")
                })?;
                out.push(s.to_lowercase());
            }
            out
        }
        None => Vec::new(),
    };
    let etb_effects = match dict.get_item("etb_effects")? {
        Some(eff_any) => {
            let mut out = Vec::new();
            for item in eff_any.try_iter()? {
                let item = item?;
                let eff_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each etb_effect must be a dict")
                })?;
                out.push(parse_effect(eff_dict)?);
            }
            out
        }
        None => Vec::new(),
    };
    // Slice 1 ETB-trigger substrate: `triggered_abilities` defaults
    // to empty Vec when absent (back-compat for every pre-Slice-1
    // registry dict).
    let mut triggered_abilities = match dict.get_item("triggered_abilities")? {
        Some(ta_any) => {
            let mut out = Vec::new();
            for item in ta_any.try_iter()? {
                let item = item?;
                let ta_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each triggered_ability must be a dict")
                })?;
                out.push(parse_triggered_ability(ta_dict)?);
            }
            out
        }
        None => Vec::new(),
    };
    // ETB-trigger Slice 3: bridge legacy `etb_effects: Vec<Effect>`
    // into the predicate path. Each `etb_effects` entry is the
    // shape "when this enters, apply effect" — i.e. an IsSelf-only
    // ENTERS trigger. Synthesize that as a TriggeredAbility and
    // append to triggered_abilities so the predicate-path dispatch
    // walker (state_ops_resolve::dispatch_event) handles it via the
    // same substrate as every other ENTERS trigger.
    //
    // Synthesis is APPENDED (not prepended) so explicit
    // `triggered_abilities` entries fire in declaration order BEFORE
    // the synthesized self-ETB shim. This preserves the same
    // dispatcher walk order that Slice 1/2's tests pin on (ability
    // index iteration is left-to-right; synthesized entries get
    // higher indices). For printings that DON'T also carry an
    // explicit `triggered_abilities` entry the order doesn't matter.
    //
    // Audit pins (`docs/RUST_ETB_TRIGGER_AUDIT.md` §3.1) the legacy
    // hatch as "the synthesis source, not the runtime path"
    // post-Slice-3 — the `etb_effects` field is still parsed for
    // back-compat with pre-Slice-3 fixtures, but the resolver no
    // longer iterates it directly (Slice 3 retires the inline loop
    // in `resolve_top_of_stack` + `_pause`).
    for effect in &etb_effects {
        triggered_abilities.push(TriggeredAbility {
            event: Event::Enters,
            predicate: Predicate::SourceIsSelf,
            effect: effect.clone(),
            once_per_turn: false,
        });
    }
    // CE20 Slice 1 substrate: `activated_abilities` defaults to empty
    // Vec when absent (back-compat for every pre-CE20-Slice-1 registry
    // dict). Same parse shape as `triggered_abilities`.
    let mut activated_abilities = match dict.get_item("activated_abilities")? {
        Some(aa_any) => {
            let mut out = Vec::new();
            for item in aa_any.try_iter()? {
                let item = item?;
                let aa_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each activated_ability must be a dict")
                })?;
                out.push(parse_activated_ability(aa_dict)?);
            }
            out
        }
        None => Vec::new(),
    };
    // CE20 follow-up — bridge `mana_produced` into
    // `activated_abilities`. See `apply_mana_produced_bridge` for the
    // synthesis details. The bridge prepends one synthesized
    // ActivatedAbility at `ability_index=0` so every pre-bridge
    // caller of `ActivateAbility(forest_id, 0)` keeps working.
    activated_abilities = apply_mana_produced_bridge(
        &mana_produced,
        activated_abilities,
    )?;
    // CE20 Slice 5 (Walker loyalty — starting_loyalty ETB seeding).
    // `starting_loyalty` is optional; defaults to 0 when the key is
    // absent. Stored as i32 (matches `GameObject.loyalty_counters`
    // shape). Negative values are accepted but observable only if a
    // pathological registry dict supplies one — production walkers
    // always print a non-negative starting loyalty.
    let starting_loyalty: i32 = match dict.get_item("starting_loyalty")? {
        Some(sl) => sl.extract::<i32>().map_err(|e| {
            PyValueError::new_err(format!(
                "starting_loyalty must be an integer (got: {e})"
            ))
        })?,
        None => 0,
    };
    // Aura-attachment substrate — Phase 1 STEP 2 (round-78,
    // 2026-05-20). Read the aura keys the Python bridge emits at
    // `cross_engine.py:10339-10391` (`is_aura` + `aura_enchant_type`
    // + `aura_yours_only` + `aura_equipment_bonuses` sub-dict). Every
    // key uses the same back-compat
    // `match get_item { Some => extract, None => default }` shape as
    // every other optional field above — missing key => pre-substrate
    // default, so every pre-Phase-1 registry dict parses bit-identical
    // to before this slice landed.
    //
    // **Inert until STEPs 3-6 land.** STEP 2 stores the values in
    // `CardPrinting` — no Rust consumer reads them yet. The Phase-1
    // remaining steps (cast-time enchant-target decision, resolution
    // attach arm, CR-704.5n SBA, bidir bridge) will read these fields
    // off the printing rather than re-parsing the registry dict.
    let is_aura = match dict.get_item("is_aura")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err("is_aura must be a bool")
        })?,
        None => false,
    };
    // `aura_enchant_type` mirrors Python's default at
    // `cards/base.py:507` ("creature"). The field is ignored unless
    // `is_aura=true`, but we still default it consistently so a
    // direct-construction path (state.rs synthetic printings) doesn't
    // accidentally diverge from the parse-path default.
    let aura_enchant_type = match dict.get_item("aura_enchant_type")? {
        Some(v) => v.extract::<String>().map_err(|_| {
            PyValueError::new_err(
                "aura_enchant_type must be a string \
                 (\"creature\" | \"artifact\")",
            )
        })?,
        None => "creature".to_string(),
    };
    let aura_yours_only = match dict.get_item("aura_yours_only")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err("aura_yours_only must be a bool")
        })?,
        None => false,
    };
    // `aura_equipment_bonuses` is a single nested dict
    // `{"power_bonus": int, "toughness_bonus": int, "granted_keywords":
    // [str, ...]}` (NOT three top-level keys — see
    // `cross_engine.py:10384`). Missing top-level key OR a present-but-
    // None inner key parses to the zero/empty default; an absent
    // top-level key on a non-Aura registers as P/T-or-keyword-less.
    // The bridge omits the top-level key when no bonus is meaningful
    // (the Python `if card.aura_power_bonus or ... or
    // card.aura_granted_keywords:` gate at `cross_engine.py:10374`).
    // Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20).
    // Three new optional keys: `aura_grants_control` (bool),
    // `aura_grants_lose_abilities` (bool), `aura_set_base_pt` (a
    // 2-key sub-dict {"power": int, "toughness": int}). Each missing
    // key parses to its zero default for back-compat — every pre-S2
    // registry dict carries none of them and the parsed CardPrinting
    // is bit-equal to before this slice landed.
    let aura_grants_control = match dict.get_item("aura_grants_control")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err("aura_grants_control must be a bool")
        })?,
        None => false,
    };
    let aura_grants_lose_abilities =
        match dict.get_item("aura_grants_lose_abilities")? {
            Some(v) => v.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "aura_grants_lose_abilities must be a bool",
                )
            })?,
            None => false,
        };
    let aura_set_base_pt: Option<(i16, i16)> =
        match dict.get_item("aura_set_base_pt")? {
            Some(any) => {
                let sub = any.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "aura_set_base_pt must be a dict \
                         {\"power\": int, \"toughness\": int}",
                    )
                })?;
                let p = match sub.get_item("power")? {
                    Some(v) => v.extract::<i16>().map_err(|_| {
                        PyValueError::new_err(
                            "aura_set_base_pt.power must be an integer",
                        )
                    })?,
                    None => 0,
                };
                let t = match sub.get_item("toughness")? {
                    Some(v) => v.extract::<i16>().map_err(|_| {
                        PyValueError::new_err(
                            "aura_set_base_pt.toughness must be an integer",
                        )
                    })?,
                    None => 0,
                };
                Some((p, t))
            }
            None => None,
        };
    // Aura-attachment substrate — Phase 2 S2 extension (Tractor Beam
    // closure, 2026-05-20). One new optional key:
    // `aura_prevents_host_untap` (bool). Missing key parses to
    // `false` per the back-compat pattern every other optional aura
    // flag uses.
    let aura_prevents_host_untap =
        match dict.get_item("aura_prevents_host_untap")? {
            Some(v) => v.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "aura_prevents_host_untap must be a bool",
                )
            })?,
            None => false,
        };
    // Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    // substrate landing, 2026-05-20). One new optional key:
    // `aura_replaces_first_tokens_with_copies` (bool). Missing key
    // parses to `false` per the back-compat pattern every other
    // optional aura flag uses — every pre-S3 registry dict omits the
    // key and the parsed CardPrinting is bit-equal to before.
    let aura_replaces_first_tokens_with_copies =
        match dict.get_item("aura_replaces_first_tokens_with_copies")? {
            Some(v) => v.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "aura_replaces_first_tokens_with_copies must be a bool",
                )
            })?,
            None => false,
        };
    // Static / layered abilities substrate — Phase 1 first-deliverable
    // (round-79). Parse the `static_anthems` list — each entry is a
    // dict `{"scope": str, "granted_keywords": [str, ...]}`. Missing
    // top-level key parses to an empty Vec (back-compat default — every
    // pre-substrate registry dict omits the key, mirroring the
    // `aura_equipment_bonuses` Option-parse shape).
    let static_anthems: Vec<StaticAnthem> = match dict.get_item("static_anthems")? {
        Some(anthems_any) => {
            let iter = anthems_any.try_iter().map_err(|_| {
                PyValueError::new_err(
                    "static_anthems must be an iterable of \
                     {scope, granted_keywords} dicts",
                )
            })?;
            let mut out = Vec::new();
            for item in iter {
                let item = item?;
                let anthem_dict = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "static_anthems entries must be dicts \
                         {scope: str, granted_keywords: [str, ...]}",
                    )
                })?;
                let scope = match anthem_dict.get_item("scope")? {
                    Some(v) => v.extract::<String>().map_err(|_| {
                        PyValueError::new_err(
                            "static_anthems[].scope must be a string",
                        )
                    })?,
                    None => return Err(PyValueError::new_err(
                        "static_anthems[] is missing required key 'scope'",
                    )),
                };
                // Whitelist the scope to the two anthem shapes Phase 1
                // wires. A scope we don't recognize is a bridge-vs-Rust
                // disagreement — fail loudly so a future scope extension
                // can't silently mis-translate.
                if scope != "creatures_you_control"
                    && scope != "other_creatures_you_control"
                {
                    return Err(PyValueError::new_err(format!(
                        "static_anthems[].scope must be \
                         'creatures_you_control' or \
                         'other_creatures_you_control' (got: {})",
                        scope
                    )));
                }
                let kws = match anthem_dict.get_item("granted_keywords")? {
                    Some(kw_any) => {
                        let kw_iter = kw_any.try_iter().map_err(|_| {
                            PyValueError::new_err(
                                "static_anthems[].granted_keywords must \
                                 be an iterable of keyword strings",
                            )
                        })?;
                        let mut kws = Vec::new();
                        for k in kw_iter {
                            let s: String = k?.extract().map_err(|_| {
                                PyValueError::new_err(
                                    "static_anthems[].granted_keywords \
                                     entries must be strings",
                                )
                            })?;
                            kws.push(s);
                        }
                        kws
                    }
                    None => Vec::new(),
                };
                out.push(StaticAnthem {
                    scope,
                    granted_keywords: kws,
                });
            }
            out
        }
        None => Vec::new(),
    };
    // Static / layered abilities substrate — Phase 2 (round-80,
    // 2026-05-20). Frenzied Baloth's two uncounterable flags. Both
    // default to false; the bridge emits the keys only when True.
    let spell_is_uncounterable = match dict.get_item("spell_is_uncounterable")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err("spell_is_uncounterable must be a bool")
        })?,
        None => false,
    };
    let grants_creature_spells_uncounterable =
        match dict.get_item("grants_creature_spells_uncounterable")? {
            Some(v) => v.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "grants_creature_spells_uncounterable must be a bool",
                )
            })?,
            None => false,
        };
    let (aura_power_bonus, aura_toughness_bonus, aura_granted_keywords) =
        match dict.get_item("aura_equipment_bonuses")? {
            Some(bonuses_any) => {
                let bonuses_dict = bonuses_any
                    .downcast::<PyDict>()
                    .map_err(|_| {
                        PyValueError::new_err(
                            "aura_equipment_bonuses must be a dict",
                        )
                    })?;
                let pwr = match bonuses_dict.get_item("power_bonus")? {
                    Some(v) => v.extract::<i16>().map_err(|_| {
                        PyValueError::new_err(
                            "aura_equipment_bonuses.power_bonus must \
                             be an integer",
                        )
                    })?,
                    None => 0,
                };
                let tgh = match bonuses_dict.get_item("toughness_bonus")? {
                    Some(v) => v.extract::<i16>().map_err(|_| {
                        PyValueError::new_err(
                            "aura_equipment_bonuses.toughness_bonus \
                             must be an integer",
                        )
                    })?,
                    None => 0,
                };
                let kws = match bonuses_dict.get_item("granted_keywords")? {
                    Some(kw_any) => {
                        let iter = kw_any.try_iter().map_err(|_| {
                            PyValueError::new_err(
                                "aura_equipment_bonuses.granted_keywords \
                                 must be an iterable of keyword strings",
                            )
                        })?;
                        let mut out = Vec::new();
                        for item in iter {
                            let s: String = item?.extract().map_err(|_| {
                                PyValueError::new_err(
                                    "aura_equipment_bonuses.\
                                     granted_keywords entries must be \
                                     strings",
                                )
                            })?;
                            out.push(s);
                        }
                        out
                    }
                    None => Vec::new(),
                };
                (pwr, tgh, kws)
            }
            None => (0i16, 0i16, Vec::new()),
        };
    // Dynamic-P/T formula substrate (round-78). Optional string key
    // mirroring Python's `CardPrinting.dynamic_pt_formula`. Missing key
    // -> `None` (the Option/serde-default pattern every back-compat-
    // added field uses); a non-string value or a value that's neither
    // a string nor `None` is rejected to surface a registry-dict typo
    // immediately. The recognized keys live in Python's
    // `game._dynamic_base_pt`; this parser does NOT validate the key
    // string here (a future cross-engine probe lane would catch any
    // typo via a parity-divergence rather than at parse time).
    // Static / layered abilities substrate — Phase 3 (round-81,
    // 2026-05-20). Parse the `static_self_bonuses` list — each entry
    // is a dict
    //   {"condition_tag": str,
    //    "condition_args": [int, ...],
    //    "power_bonus": int,
    //    "toughness_bonus": int,
    //    "granted_keywords": [str, ...]}.
    // Missing top-level key parses to an empty Vec (back-compat default
    // — every pre-substrate registry dict omits the key, mirroring the
    // `static_anthems` parse shape).
    let static_self_bonuses: Vec<StaticSelfBonus> =
        match dict.get_item("static_self_bonuses")? {
            Some(bonuses_any) => {
                let iter = bonuses_any.try_iter().map_err(|_| {
                    PyValueError::new_err(
                        "static_self_bonuses must be an iterable of \
                         {condition_tag, condition_args, power_bonus, \
                         toughness_bonus, granted_keywords} dicts",
                    )
                })?;
                let mut out = Vec::new();
                for item in iter {
                    let item = item?;
                    let entry = item.downcast::<PyDict>().map_err(|_| {
                        PyValueError::new_err(
                            "static_self_bonuses entries must be dicts",
                        )
                    })?;
                    let condition_tag = match entry.get_item("condition_tag")? {
                        Some(v) => v.extract::<String>().map_err(|_| {
                            PyValueError::new_err(
                                "static_self_bonuses[].condition_tag \
                                 must be a string",
                            )
                        })?,
                        None => return Err(PyValueError::new_err(
                            "static_self_bonuses[] is missing required \
                             key 'condition_tag'",
                        )),
                    };
                    // Whitelist the tag to the recognized Phase-3
                    // shapes (round-82 adds the third entry). A tag
                    // we don't recognize is a bridge-vs-Rust
                    // disagreement — fail loudly so a future
                    // extension can't silently mis-translate.
                    if condition_tag != "control_permanent_is_artifact"
                        && condition_tag != "at_least_n_spells_cast_this_turn"
                        && condition_tag != "controlled_artifact_entered_this_turn"
                    {
                        return Err(PyValueError::new_err(format!(
                            "static_self_bonuses[].condition_tag must be \
                             'control_permanent_is_artifact', \
                             'at_least_n_spells_cast_this_turn', or \
                             'controlled_artifact_entered_this_turn' \
                             (got: {})",
                            condition_tag
                        )));
                    }
                    let condition_args = match entry.get_item("condition_args")? {
                        Some(args_any) => {
                            let args_iter = args_any.try_iter().map_err(|_| {
                                PyValueError::new_err(
                                    "static_self_bonuses[].condition_args \
                                     must be an iterable of ints",
                                )
                            })?;
                            let mut args = Vec::new();
                            for a in args_iter {
                                let n: i32 = a?.extract().map_err(|_| {
                                    PyValueError::new_err(
                                        "static_self_bonuses[].\
                                         condition_args entries must be \
                                         ints",
                                    )
                                })?;
                                args.push(n);
                            }
                            args
                        }
                        None => Vec::new(),
                    };
                    // Per-tag arity validation — mirrors the Python
                    // `_static_self_bonus_condition_from_tag` helper's
                    // arity checks.
                    match condition_tag.as_str() {
                        "control_permanent_is_artifact" => {
                            if !condition_args.is_empty() {
                                return Err(PyValueError::new_err(format!(
                                    "static_self_bonuses[].condition_args \
                                     must be empty for tag \
                                     'control_permanent_is_artifact' \
                                     (got: {:?})",
                                    condition_args
                                )));
                            }
                        }
                        "at_least_n_spells_cast_this_turn" => {
                            if condition_args.len() != 1 {
                                return Err(PyValueError::new_err(format!(
                                    "static_self_bonuses[].condition_args \
                                     must have exactly 1 int (n) for tag \
                                     'at_least_n_spells_cast_this_turn' \
                                     (got: {:?})",
                                    condition_args
                                )));
                            }
                            if condition_args[0] < 0 {
                                return Err(PyValueError::new_err(format!(
                                    "static_self_bonuses[].condition_args[0] \
                                     (n) must be non-negative for tag \
                                     'at_least_n_spells_cast_this_turn' \
                                     (got: {})",
                                    condition_args[0]
                                )));
                            }
                        }
                        "controlled_artifact_entered_this_turn" => {
                            if !condition_args.is_empty() {
                                return Err(PyValueError::new_err(format!(
                                    "static_self_bonuses[].condition_args \
                                     must be empty for tag \
                                     'controlled_artifact_entered_this_turn' \
                                     (got: {:?})",
                                    condition_args
                                )));
                            }
                        }
                        _ => unreachable!(
                            "condition_tag whitelisted above",
                        ),
                    }
                    let power_bonus = match entry.get_item("power_bonus")? {
                        Some(v) => v.extract::<i16>().map_err(|_| {
                            PyValueError::new_err(
                                "static_self_bonuses[].power_bonus must \
                                 be an integer",
                            )
                        })?,
                        None => 0,
                    };
                    let toughness_bonus = match entry.get_item("toughness_bonus")? {
                        Some(v) => v.extract::<i16>().map_err(|_| {
                            PyValueError::new_err(
                                "static_self_bonuses[].toughness_bonus \
                                 must be an integer",
                            )
                        })?,
                        None => 0,
                    };
                    let granted_keywords = match entry.get_item("granted_keywords")? {
                        Some(kw_any) => {
                            let kw_iter = kw_any.try_iter().map_err(|_| {
                                PyValueError::new_err(
                                    "static_self_bonuses[].granted_keywords \
                                     must be an iterable of strings",
                                )
                            })?;
                            let mut kws = Vec::new();
                            for k in kw_iter {
                                let s: String = k?.extract().map_err(|_| {
                                    PyValueError::new_err(
                                        "static_self_bonuses[].\
                                         granted_keywords entries must \
                                         be strings",
                                    )
                                })?;
                                kws.push(s);
                            }
                            kws
                        }
                        None => Vec::new(),
                    };
                    // **Round-82 (2026-05-20, Mechan Shieldmate
                    // closure).** Parse the optional
                    // `removed_keywords` list — same shape as
                    // `granted_keywords`, just on the layer-6
                    // REMOVE side. Omitted key → empty Vec
                    // (back-compat default; every pre-round-82
                    // entry omits the key and parses to
                    // `Vec::new()`, byte-identical to before this
                    // slice).
                    let removed_keywords = match entry.get_item("removed_keywords")? {
                        Some(kw_any) => {
                            let kw_iter = kw_any.try_iter().map_err(|_| {
                                PyValueError::new_err(
                                    "static_self_bonuses[].removed_keywords \
                                     must be an iterable of strings",
                                )
                            })?;
                            let mut kws = Vec::new();
                            for k in kw_iter {
                                let s: String = k?.extract().map_err(|_| {
                                    PyValueError::new_err(
                                        "static_self_bonuses[].\
                                         removed_keywords entries must \
                                         be strings",
                                    )
                                })?;
                                kws.push(s);
                            }
                            kws
                        }
                        None => Vec::new(),
                    };
                    out.push(StaticSelfBonus {
                        condition_tag,
                        condition_args,
                        power_bonus,
                        toughness_bonus,
                        granted_keywords,
                        removed_keywords,
                    });
                }
                out
            }
            None => Vec::new(),
        };
    let dynamic_pt_formula = match dict.get_item("dynamic_pt_formula")? {
        Some(v) => {
            if v.is_none() {
                None
            } else {
                Some(v.extract::<String>().map_err(|_| {
                    PyValueError::new_err(
                        "dynamic_pt_formula must be a string \
                         (the formula key Python's `_dynamic_base_pt` \
                         dispatches on — `harmonious_grovestrider`, \
                         `cosmogoyf`, or `emissary_escort` today)",
                    )
                })?)
            }
        }
        None => None,
    };
    Ok(CardPrinting {
        name: Arc::from(name),
        mana_cost,
        effects,
        power,
        toughness,
        mana_produced,
        card_types,
        subtypes,
        supertypes,
        keywords,
        etb_effects,
        triggered_abilities,
        activated_abilities,
        starting_loyalty,
        warp_cost,
        warp_additional_life_cost,
        warp_castable_from_graveyard,
        additional_cost_sacrifice_creature_or_artifact,
        additional_cost_sacrifice_creature_or_enchantment,
        additional_cost_choose_your_creature,
        is_aura,
        aura_enchant_type,
        aura_yours_only,
        aura_power_bonus,
        aura_toughness_bonus,
        aura_granted_keywords,
        aura_grants_control,
        aura_grants_lose_abilities,
        aura_set_base_pt,
        aura_prevents_host_untap,
        aura_replaces_first_tokens_with_copies,
        static_anthems,
        spell_is_uncounterable,
        grants_creature_spells_uncounterable,
        static_self_bonuses,
        dynamic_pt_formula,
    })
}

/// Parse one `ActivatedAbility` from a registry dict. Dict shape:
///
/// ```python
/// {
///     "cost": {                         # required; ActivationCost shape
///         "mana": {"generic": 1, "G": 1},  # optional; default = zero cost
///         "tap_self": True,                # optional; default = False
///     },
///     "effect": {...},                  # required; same shape as parse_effect input
///     "is_mana_ability": False,         # optional; default = False
/// }
/// ```
///
/// Slice 1 minimum surface; future slices add richer cost shapes
/// (sacrifice_self, pay_life, requires_*) in lockstep with their
/// dispatch counterparts.
fn parse_activated_ability(
    dict: &Bound<'_, PyDict>,
) -> PyResult<ActivatedAbility> {
    let cost_any = require(dict, "cost")?;
    let cost_dict = cost_any.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("activated_ability.cost must be a dict")
    })?;
    let cost = parse_activation_cost(cost_dict)?;
    let effect_any = require(dict, "effect")?;
    let effect_dict = effect_any.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("activated_ability.effect must be a dict")
    })?;
    let effect = parse_effect(effect_dict)?;
    let is_mana_ability = match dict.get_item("is_mana_ability")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err(
                "activated_ability.is_mana_ability must be a bool",
            )
        })?,
        None => false,
    };
    // CE20 follow-up — Spacecraft station-charge. Optional
    // `tier_threshold: int >= 0`. Missing key defaults to 0 (no
    // gate — back-compat with every pre-Slice-3 caller).
    let tier_threshold = match dict.get_item("tier_threshold")? {
        Some(v) => {
            let n: i32 = v.extract().map_err(|_| {
                PyValueError::new_err(
                    "activated_ability.tier_threshold must be a non-negative int",
                )
            })?;
            if n < 0 {
                return Err(PyValueError::new_err(format!(
                    "activated_ability.tier_threshold must be \
                     non-negative, got {n}"
                )));
            }
            n
        }
        None => 0,
    };
    Ok(ActivatedAbility {
        cost,
        effect,
        is_mana_ability,
        tier_threshold,
    })
}

/// Parse one `ActivationCost` from a registry dict. Slice 1 supports
/// `mana` (optional ManaCost dict) + `tap_self` (optional bool).
/// CE20 follow-up Slice 2 adds `sacrifice_self` (optional bool).
/// CE20 follow-up Slice 4 adds `requires_target_creature_you_control`
/// (optional bool — flags target fan-out at emission + a required
/// `target_creature_id` key on the apply dict). CE20 follow-up Slice 6
/// adds `sacrifice_another_creature_or_artifact`, `sacrifice_a_land`,
/// `sacrifice_another_artifact` (optional bools — targeted-sacrifice
/// variants via `sacrifice_target_id`). CE20 follow-up Slice 7 adds
/// `sacrifice_food` and `sacrifice_a_desert` (optional bools —
/// subtype-filtered sacrifice variants riding on `CardPrinting.subtypes`
/// from the subtypes substrate). Unknown keys raise so future field
/// additions surface as parser errors rather than silent drops.
fn parse_activation_cost(
    dict: &Bound<'_, PyDict>,
) -> PyResult<ActivationCost> {
    let mut cost = ActivationCost::default();
    for (key_obj, _) in dict.iter() {
        let key: String = key_obj.extract()?;
        match key.as_str() {
            "mana"
            | "tap_self"
            | "sacrifice_self"
            | "requires_target_creature_you_control"
            | "sacrifice_another_creature_or_artifact"
            | "sacrifice_a_land"
            | "sacrifice_another_artifact"
            | "sacrifice_food"
            | "sacrifice_a_desert"
            | "loyalty" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown activation_cost key {other:?}; \
                     valid: mana, tap_self, sacrifice_self, \
                     requires_target_creature_you_control, \
                     sacrifice_another_creature_or_artifact, \
                     sacrifice_a_land, sacrifice_another_artifact, \
                     sacrifice_food, sacrifice_a_desert, \
                     loyalty"
                )));
            }
        }
    }
    if let Some(mana_any) = dict.get_item("mana")? {
        let mana_dict = mana_any.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("activation_cost.mana must be a dict")
        })?;
        cost.mana = parse_mana_cost(mana_dict)?;
    }
    if let Some(tap_any) = dict.get_item("tap_self")? {
        cost.tap_self = tap_any.extract::<bool>().map_err(|_| {
            PyValueError::new_err("activation_cost.tap_self must be a bool")
        })?;
    }
    if let Some(sac_any) = dict.get_item("sacrifice_self")? {
        cost.sacrifice_self = sac_any.extract::<bool>().map_err(|_| {
            PyValueError::new_err(
                "activation_cost.sacrifice_self must be a bool",
            )
        })?;
    }
    if let Some(req_any) =
        dict.get_item("requires_target_creature_you_control")?
    {
        cost.requires_target_creature_you_control =
            req_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.requires_target_creature_you_control \
                     must be a bool",
                )
            })?;
    }
    if let Some(sac_another_any) =
        dict.get_item("sacrifice_another_creature_or_artifact")?
    {
        cost.sacrifice_another_creature_or_artifact =
            sac_another_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.sacrifice_another_creature_or_artifact \
                     must be a bool",
                )
            })?;
    }
    if let Some(sac_land_any) = dict.get_item("sacrifice_a_land")? {
        cost.sacrifice_a_land =
            sac_land_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.sacrifice_a_land must be a bool",
                )
            })?;
    }
    if let Some(sac_artifact_any) =
        dict.get_item("sacrifice_another_artifact")?
    {
        cost.sacrifice_another_artifact =
            sac_artifact_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.sacrifice_another_artifact \
                     must be a bool",
                )
            })?;
    }
    // CE20 follow-up Slice 7 — subtype-filtered sacrifice variants.
    // Both fields default `false`; missing keys are back-compat-OK.
    if let Some(sac_food_any) = dict.get_item("sacrifice_food")? {
        cost.sacrifice_food =
            sac_food_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.sacrifice_food must be a bool",
                )
            })?;
    }
    if let Some(sac_desert_any) = dict.get_item("sacrifice_a_desert")? {
        cost.sacrifice_a_desert =
            sac_desert_any.extract::<bool>().map_err(|_| {
                PyValueError::new_err(
                    "activation_cost.sacrifice_a_desert must be a bool",
                )
            })?;
    }
    if let Some(loy_any) = dict.get_item("loyalty")? {
        cost.loyalty = loy_any.extract::<i32>().map_err(|_| {
            PyValueError::new_err(
                "activation_cost.loyalty must be an int (signed; \
                 positive = pay-loyalty, negative = add-loyalty)",
            )
        })?;
    }
    Ok(cost)
}

/// Parse one `TriggeredAbility` from a registry dict. Dict shape:
///
/// ```python
/// {
///     "event": "enters",            # required; lowercase string
///     "effect": {...},              # required; same shape as parse_effect input
///     "predicate": {...},           # optional; defaults to {"kind": "always"}
///     "once_per_turn": False,       # optional; defaults to False
/// }
/// ```
///
/// Slice 1 wired `"enters"`; Slice 5 added `"dies"`; Slice 6 added
/// `"cast"`; Slice 7 added `"attacks"`; Slice 8 added
/// `"leaves_battlefield"`; Slice 9 adds `"blocks"`; round-58 adds
/// `"becomes_tapped"`; round-13 adds `"deals_damage"` alongside the
/// ``dispatch_combat_damage_to_player`` dispatch site in
/// ``state_ops_combat.rs``; round-16 (RR Pins 1+2+3) adds
/// ``"counters_added"``, ``"is_sacrificed"``, and ``"dealt_damage"``.
/// No further events are stub-raised —
/// unrecognised strings fall through to the `other` error arm.
fn parse_triggered_ability(
    dict: &Bound<'_, PyDict>,
) -> PyResult<TriggeredAbility> {
    let event_str: String = require(dict, "event")?.extract().map_err(|_| {
        PyValueError::new_err("triggered_ability.event must be a string")
    })?;
    let event = match event_str.as_str() {
        "enters" => Event::Enters,
        "dies" => Event::Dies,
        "cast" => Event::Cast,
        "attacks" => Event::Attacks,
        "blocks" => Event::Blocks,
        "leaves_battlefield" => Event::LeavesBattlefield,
        // Round-58 BECOMES_TAPPED event bridge: "becomes_tapped"
        // round-trips the SAME way enters/dies/attacks do. Python's
        // `Event.BECOMES_TAPPED` string value is `"becomes_tapped"`
        // (`src/mtg_bot/effects/predicates.py:60`); the dispatcher
        // (`state_ops_resolve::dispatch_event`) and every production
        // tap site emit it in lockstep with Python's `tap_permanent`
        // chokepoint.
        "becomes_tapped" => Event::BecomesTapped,
        // Round-13 DEALS_DAMAGE event bridge: "deals_damage" now
        // round-trips through the parse + dispatch pipeline. Rust's
        // ``Event::DealsDamage`` is fired by
        // ``state_ops_combat::dispatch_combat_damage_to_player`` for
        // each (attacker, defending_player) damage pair where the
        // attacker dealt positive damage directly to the player (not
        // to a blocker). Python's ``Event.DEALS_COMBAT_DAMAGE``
        // (string value ``"deals_combat_damage"``) is the counterpart;
        // the string values differ (``"deals_damage"`` in Rust vs
        // ``"deals_combat_damage"`` in Python) but both carry the same
        // semantic — the bridge translates in
        // ``_ta_dict_to_python_triggered_ability``.
        //
        // Subject shape (two-subject event):
        //   subject = source_id (the damage-dealing creature) — used by
        //   ``Predicate::SourceIsSelf`` which checks
        //   ``source_id == subject_id``. The ``target`` (defending
        //   player) is implicit in the dispatch context (we only call
        //   dispatch for player-targeted damage entries) — Rust's
        //   predicate system does not yet carry a secondary subject_id.
        //   This is equivalent to Python's ``IsSelf(subject="source")``
        //   because the dispatch loop already ensures the target is a
        //   player; an explicit ``SubjectIsPlayer`` predicate is
        //   deferred until a second-subject predicate API lands.
        //
        // Wires: Illvoi Infiltrator (``source_is_self`` -> ``DrawCards(1)``)
        // and the broader controller-based shapes (Starwinder) at the
        // predicate level once ``ControlledByYou`` is available on the
        // source subject — deferred, see SEARCH_LIMITATIONS.md entry
        // for the single-subject-only Rust predicate API.
        "deals_damage" => Event::DealsDamage,
        // Round-14 PHASE_STEP event bridge: the three phase-boundary
        // events for "At the beginning of your/each end step / upkeep /
        // combat" triggered abilities. Fired by ``advance_phase`` at
        // entry to the corresponding phase constant (``PHASE_END_STEP``,
        // ``PHASE_UPKEEP``, ``PHASE_BEGIN_COMBAT``).
        //
        // Subject shape (single player subject):
        //   subject = ``PLAYER_SUBJECT_OFFSET + active_player as u32``
        //   (state_ops_resolve.rs constant). ``ControlledByYou`` in the
        //   predicate arm recognises subject_ids ≥ PLAYER_SUBJECT_OFFSET
        //   and resolves the player index directly, bypassing the normal
        //   game-object lookup. ``Always`` predicates fire unconditionally
        //   (subject_id is irrelevant). No multi-subject complexity —
        //   phase-step events carry exactly one subject slot.
        //
        // Bridge translation (cross_engine.py):
        //   "beginning_of_end_step"  → Event.BEGINNING_OF_END_STEP
        //   "beginning_of_upkeep"    → Event.BEGINNING_OF_UPKEEP
        //   "beginning_of_combat"    → Event.BEGINNING_OF_COMBAT
        // (string values are identical between Rust and Python — no
        // spelling skew like DealsDamage/DEALS_COMBAT_DAMAGE.)
        "beginning_of_end_step" => Event::BeginningOfEndStep,
        "beginning_of_upkeep" => Event::BeginningOfUpkeep,
        "beginning_of_combat" => Event::BeginningOfCombat,
        // Round-15 YOU_ATTACK event bridge: "whenever you attack" triggered
        // abilities. Fires ONCE per combat when the active player declares
        // at least one legal attacker — distinct from ``"attacks"`` which
        // fires per-attacker. Subject = PLAYER_SUBJECT_OFFSET + active_player.
        // Mirrors Python's ``Event.YOU_ATTACK`` (value ``"you_attack"``).
        // 8 EoE cards unlocked (OO Pin 3 closure).
        "you_attack" => Event::YouAttack,
        // Round-16 COUNTERS_ADDED event bridge (RR Pin 1): "whenever you put
        // one or more counters on X" triggered abilities. Single-subject shape:
        // subject = the permanent that received the counters (``("obj",)``).
        // Mirrors Python's ``Event.COUNTERS_ADDED`` (value ``"counters_added"``).
        // No spelling skew — string values are identical.
        // EoE cards unlocked: Terrasymbiosis, Haliya Ascendant Cadet.
        "counters_added" => Event::CountersAdded,
        // Round-16 IS_SACRIFICED event bridge (RR Pin 2): "whenever you
        // sacrifice X" triggered abilities. Single-subject shape: subject =
        // the sacrificed permanent (``("obj",)``). Distinct from DIES (which
        // fires for all battlefield-to-graveyard transitions, not only sacrifice).
        // Mirrors Python's ``Event.IS_SACRIFICED`` (value ``"is_sacrificed"``).
        // No spelling skew.
        // EoE cards unlocked: Lightless Evangel, Biotech Specialist.
        "is_sacrificed" => Event::IsSacrificed,
        // Round-16 DEALT_DAMAGE event bridge (RR Pin 3): "whenever X is
        // dealt damage" triggered abilities. Single-subject shape: subject =
        // the permanent that RECEIVED damage (``("obj",)``). Distinct from
        // DealsDamage / DEALS_COMBAT_DAMAGE (the DEALER side; two subjects).
        // Mirrors Python's ``Event.DEALT_DAMAGE`` (value ``"dealt_damage"``).
        // No spelling skew.
        // EoE cards unlocked: Requiem Monolith, Pain for All.
        "dealt_damage" => Event::DealtDamage,
        // PPP Pin 3 (10th Rust slice): "at the beginning of your
        // postcombat main phase" triggered abilities. Fires on entry
        // to PHASE_MAIN2 (the postcombat main step). Single-subject
        // shape: subject = PLAYER_SUBJECT_OFFSET + active_player.
        // Mirrors Python's `Event.BEGINNING_OF_POSTCOMBAT_MAIN`
        // (value `"beginning_of_postcombat_main"`). No spelling skew.
        // 5 EoE cards unlocked: Savior of the Small, Cautious Survivor,
        // Defiant Survivor, House Cartographer, Shrewd Storyteller.
        "beginning_of_postcombat_main" => Event::BeginningOfPostcombatMain,
        // **MMM Pin 2 (11th Rust slice) — DISCARDS_CARD event.**
        // "Whenever <player> discards a card" — Entropic
        // Battlecruiser's 1+ tier "Whenever an opponent discards a
        // card, they lose 3 life" is the canonical consumer. Single-
        // subject shape: subject = PLAYER_SUBJECT_OFFSET +
        // discarding_player_id. Mirrors Python's
        // `Event.DISCARDS_CARD` (value `"discards_card"`); no
        // spelling skew. Dispatched from the three "discard"
        // primitives (DiscardCard / DiscardSelf / EachOpponentDiscards).
        "discards_card" => Event::DiscardsCard,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown triggered_ability.event {other:?}; \
                 valid: enters, dies, cast, attacks, blocks, \
                 leaves_battlefield, becomes_tapped, deals_damage, \
                 beginning_of_end_step, beginning_of_upkeep, \
                 beginning_of_combat, you_attack, \
                 counters_added, is_sacrificed, dealt_damage, \
                 beginning_of_postcombat_main, discards_card \
                 (Slice 9 + r58 + r13 + r14 + r15 + r16-RR + PPP + MMM)"
            )));
        }
    };
    let effect_any = require(dict, "effect")?;
    let effect_dict = effect_any.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("triggered_ability.effect must be a dict")
    })?;
    let effect = parse_effect(effect_dict)?;
    let predicate = match dict.get_item("predicate")? {
        Some(pred_any) => {
            let pred_dict = pred_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("triggered_ability.predicate must be a dict")
            })?;
            parse_predicate(pred_dict)?
        }
        None => Predicate::Always,
    };
    let once_per_turn = match dict.get_item("once_per_turn")? {
        Some(v) => v.extract::<bool>().map_err(|_| {
            PyValueError::new_err(
                "triggered_ability.once_per_turn must be a bool",
            )
        })?,
        None => false,
    };
    Ok(TriggeredAbility {
        event,
        predicate,
        effect,
        once_per_turn,
    })
}

/// Parse one `Predicate` from a registry dict. Dict shape:
///
/// ```python
/// {"kind": "always"}
/// {"kind": "source_is_self"}
/// {"kind": "source_is_creature"}
/// {"kind": "subject_is_land"}        # Slice 4
/// {"kind": "subject_is_artifact"}    # Slice 4
/// {"kind": "subject_is_creature"}    # Slice 4
/// {"kind": "controlled_by_you"}      # Slice 4
/// {"kind": "subject_has_subtype",    # Subtypes Slice 2
///  "subtype": "Kavu"}
/// {"kind": "not", "inner": {...}}
/// {"kind": "and", "left": {...}, "right": {...}}
/// {"kind": "or", "left": {...}, "right": {...}}     # Or-substrate slice
/// ```
///
/// Slice 1 introduced the base algebra (`always`, `source_is_self`,
/// `source_is_creature`, `not`, `and`). Slice 4 added the subject-
/// anchored type filters + `controlled_by_you` to unlock landfall /
/// artifact-enters / "under your control" trigger shapes from
/// `cards/parser/triggers.py:603-1266`. Subtypes Slice 2 added
/// `subject_has_subtype` for Kavu/Sliver/Equipment-enters shapes
/// (`cards/parser/triggers.py:1198`). The Or-substrate slice added
/// `or` for shapes like Weftstalker Ardent's "whenever a creature
/// or artifact you control enters"
/// (`(subject_is_creature OR subject_is_artifact) AND
/// controlled_by_you`).
fn parse_predicate(dict: &Bound<'_, PyDict>) -> PyResult<Predicate> {
    let kind: String = require(dict, "kind")?.extract()?;
    match kind.as_str() {
        "always" => Ok(Predicate::Always),
        "source_is_self" => Ok(Predicate::SourceIsSelf),
        "source_is_creature" => Ok(Predicate::SourceIsCreature),
        // **PPP slice — Source(IsTapped()) predicate.** Mirrors
        // Python's `Source(inner=IsTapped())`. Used by all 5
        // BEGINNING_OF_POSTCOMBAT_MAIN "survivor" cards.
        "source_is_tapped" => Ok(Predicate::SourceIsTapped),
        "subject_is_land" => Ok(Predicate::SubjectIsLand),
        "subject_is_artifact" => Ok(Predicate::SubjectIsArtifact),
        "subject_is_creature" => Ok(Predicate::SubjectIsCreature),
        // IsEnchantment substrate slice — "an enchantment you control
        // enters" (Duskmourn / EoE Eerie ability-word shape). Mirrors
        // Python's `Subject(None, IsEnchantment())` from
        // `src/mtg_bot/effects/object_predicates.py:142`.
        "subject_is_enchantment" => Ok(Predicate::SubjectIsEnchantment),
        "controlled_by_you" => {
            // **Slot-aware bridge form (12th Rust slice — Closure A).**
            // Optional ``"subject"`` field threads through to the
            // ``slot`` payload of ``ControlledByYou``. Absent →
            // ``None`` (back-compat — single-subject ENTERS / DIES /
            // ATTACKS). Present (typically ``"controller"``) → the
            // slot-aware variant used by IS_CAST / PLAYS_LAND cards
            // like Cosmogrand Zenith ("your second spell each turn"
            // family). The Rust ``matches`` arm is invariant to slot
            // value for the currently-dispatched events (subject_id
            // resolves through ``find_object_controller`` to the same
            // controller either way), so the slot is stored for
            // round-trip fidelity but doesn't change the comparison.
            let slot: Option<String> = match dict.get_item("subject")? {
                Some(v) => Some(v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "predicate.controlled_by_you.subject must be a \
                         string (got: {e})"
                    ))
                })?),
                None => None,
            };
            Ok(Predicate::ControlledByYou { slot })
        }
        // **MMM Pin 3 (11th Rust slice) — controlled_by_opponent.**
        // Dual of `controlled_by_you`. Mirrors Python's
        // `Controlled(by=OPPONENT, subject=None)` — Entropic
        // Battlecruiser's "an opponent discards" trigger gate.
        //
        // **Slot-aware bridge form (12th Rust slice — Closure A).**
        // Symmetric with the ``controlled_by_you`` arm.
        "controlled_by_opponent" => {
            let slot: Option<String> = match dict.get_item("subject")? {
                Some(v) => Some(v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "predicate.controlled_by_opponent.subject must be a \
                         string (got: {e})"
                    ))
                })?),
                None => None,
            };
            Ok(Predicate::ControlledByOpponent { slot })
        }
        // HasPowerAtMost substrate slice — "with power N or less"
        // qualifier for Vicious Clown / Irreverent Gremlin ETB triggers
        // (EoE batch K/N). Dict shape: `{"kind": "has_power_at_most",
        // "n": 2}`. Mirrors Python's `HasPowerAtMost(n)` from
        // `src/mtg_bot/effects/object_predicates.py:427`.
        "has_power_at_most" => {
            let n: i32 = require(dict, "n")?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "predicate.has_power_at_most.n must be an int (got: {e})"
                    ))
                })?;
            Ok(Predicate::HasPowerAtMost { n })
        }
        // **Pain for All closure (Aura Phase 3 / S4 triggered shape).**
        // `is_host` — mirrors Python's `IsHost(subject: str | None)`
        // predicate (`effects/predicates.py:439-467`). Optional
        // `"subject"` field threads through to the slot payload.
        // Used by Pain for All's DEALT_DAMAGE damage-echo trigger
        // (`predicate=IsHost(subject=None)`) and (future) Equipment
        // `IsHost(subject="attacker")` shapes. Same back-compat
        // omission pattern as `controlled_by_you`'s slot:
        // absent → `None` (Pain for All's default); present →
        // explicit slot.
        "is_host" => {
            let subject: Option<String> = match dict.get_item("subject")? {
                Some(v) => Some(v.extract().map_err(|e| {
                    PyValueError::new_err(format!(
                        "predicate.is_host.subject must be a \
                         string (got: {e})"
                    ))
                })?),
                None => None,
            };
            Ok(Predicate::IsHost { subject })
        }
        // Subtypes substrate Slice 2 — subject-anchored HasSubtype.
        // Dict shape: `{"kind": "subject_has_subtype", "subtype": "Kavu"}`.
        // The `subtype` payload is the canonical Title Case string
        // that lives in `CardPrinting.subtypes`.
        "subject_has_subtype" => {
            let subtype: String = require(dict, "subtype")?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "predicate.subject_has_subtype.subtype must be a \
                         string (got: {e})"
                    ))
                })?;
            Ok(Predicate::SubjectHasSubtype(subtype))
        }
        "not" => {
            let inner_any = require(dict, "inner")?;
            let inner_dict = inner_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.not.inner must be a dict")
            })?;
            Ok(Predicate::Not(Box::new(parse_predicate(inner_dict)?)))
        }
        "and" => {
            let left_any = require(dict, "left")?;
            let right_any = require(dict, "right")?;
            let left_dict = left_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.and.left must be a dict")
            })?;
            let right_dict = right_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.and.right must be a dict")
            })?;
            Ok(Predicate::And(
                Box::new(parse_predicate(left_dict)?),
                Box::new(parse_predicate(right_dict)?),
            ))
        }
        // Or-substrate slice — logical disjunction over two sub-
        // predicates. Mirrors the `and` arm's shape exactly. Unlocks
        // "creature OR artifact" / "Kavu OR Sliver" predicate shapes
        // from `cards/parser/triggers.py`.
        "or" => {
            let left_any = require(dict, "left")?;
            let right_any = require(dict, "right")?;
            let left_dict = left_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.or.left must be a dict")
            })?;
            let right_dict = right_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.or.right must be a dict")
            })?;
            Ok(Predicate::Or(
                Box::new(parse_predicate(left_dict)?),
                Box::new(parse_predicate(right_dict)?),
            ))
        }
        // **BBB downstream complement — 7th Rust slice (GGG Pin 1).**
        // `state_bound` wraps a `StatePredicate` sub-kind dict under
        // the `"inner"` key.  Dict shape:
        //   `{"kind": "state_bound", "inner": {"kind": "void_active"}}`.
        // Mirrors Python's `StateBound(inner: StatePredicate)` bridge
        // arm in `cross_engine.py::_ta_dict_to_python_predicate`.
        "state_bound" => {
            let inner_any = require(dict, "inner")?;
            let inner_dict = inner_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(
                    "predicate.state_bound.inner must be a dict (StatePredicate shape)"
                )
            })?;
            Ok(Predicate::StateBound(parse_state_predicate(inner_dict)?))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown predicate kind {other:?}; \
             valid: always, source_is_self, source_is_creature, \
             source_is_tapped, \
             subject_is_land, subject_is_artifact, subject_is_creature, \
             subject_is_enchantment, has_power_at_most, \
             controlled_by_you, controlled_by_opponent, \
             subject_has_subtype, is_host, not, and, or, \
             state_bound"
        ))),
    }
}

/// Parse a Tier-2 `StatePredicate` dict into a `StatePredicate` enum
/// variant.  Called by `parse_predicate` for the `"state_bound"` arm.
///
/// Wired kinds (GGG Pin 1):
///   * `"void_active"` → `StatePredicate::VoidActive`
///   * `"any_player_controls_no_creatures"` →
///     `StatePredicate::AnyPlayerControlsNoCreatures`
///
/// Mirrors Python's `_ta_dict_to_python_state_predicate` in
/// `src/mtg_bot/parity/cross_engine.py`.
fn parse_state_predicate(dict: &Bound<'_, PyDict>) -> PyResult<StatePredicate> {
    let kind: String = require(dict, "kind")?.extract()?;
    match kind.as_str() {
        "void_active" => Ok(StatePredicate::VoidActive),
        "any_player_controls_no_creatures" => {
            Ok(StatePredicate::AnyPlayerControlsNoCreatures)
        }
        // **MMM Pin 1 (11th Rust slice).** Python's
        // `SpellsCastThisTurn(n: int)` from
        // `effects/state_predicates.py`. Dict shape:
        //   `{"kind": "spells_cast_this_turn", "n": 2}`.
        // `n` is required (no default — symmetric with the dataclass
        // having no field default).
        "spells_cast_this_turn" => {
            let n: i32 = require(dict, "n")?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "state_predicate.spells_cast_this_turn.n must \
                         be an int (got: {e})"
                    ))
                })?;
            if n < 0 {
                return Err(PyValueError::new_err(format!(
                    "state_predicate.spells_cast_this_turn.n must be \
                     non-negative, got {n}"
                )));
            }
            Ok(StatePredicate::SpellsCastThisTurn { n })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown StatePredicate kind {other:?}; \
             valid: void_active, any_player_controls_no_creatures, \
             spells_cast_this_turn"
        ))),
    }
}

// --- Rust -> dict (round-trip) -----------------------------------

fn mana_cost_to_pydict<'py>(
    py: Python<'py>,
    cost: &ManaCost,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    if cost.generic > 0 {
        d.set_item("generic", cost.generic)?;
    }
    if cost.white > 0 {
        d.set_item("W", cost.white)?;
    }
    if cost.blue > 0 {
        d.set_item("U", cost.blue)?;
    }
    if cost.black > 0 {
        d.set_item("B", cost.black)?;
    }
    if cost.red > 0 {
        d.set_item("R", cost.red)?;
    }
    if cost.green > 0 {
        d.set_item("G", cost.green)?;
    }
    Ok(d)
}

fn target_filter_to_pydict<'py>(
    py: Python<'py>,
    target: &TargetFilter,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match target {
        TargetFilter::AnyPlayer => d.set_item("kind", "any_player")?,
        TargetFilter::AnyCreature => d.set_item("kind", "any_creature")?,
    }
    Ok(d)
}

pub(crate) fn effect_to_pydict<'py>(
    py: Python<'py>,
    effect: &Effect,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match effect {
        Effect::DealDamage { amount, target } => {
            d.set_item("kind", "deal_damage")?;
            d.set_item("amount", *amount)?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::DrawCards { count } => {
            d.set_item("kind", "draw_cards")?;
            d.set_item("count", *count)?;
        }
        Effect::GainLife { amount } => {
            d.set_item("kind", "gain_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::ReturnToHand { target } => {
            d.set_item("kind", "return_to_hand")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::Mill { count } => {
            d.set_item("kind", "mill")?;
            d.set_item("count", *count)?;
        }
        Effect::MillTargetPlayer { count } => {
            d.set_item("kind", "mill_target_player")?;
            d.set_item("count", *count)?;
        }
        Effect::MillTargetPlayerEqualToYourLifeTotal => {
            // BLOCKED-RUST-EFFECT slice (round 42). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // current life total). Mirrors Python's
            // `MillTargetPlayer(use_controller_life=True)` dict shape
            // — the `amount` field is ignored on the dynamic lane
            // (the count comes from life total, not the static
            // amount). Sister of round-32 / 35 / 36 / 37 (damage
            // substrate), round-33 (counters substrate), round-34
            // (surveil substrate) — FOURTH apply substrate and FIFTH
            // denominator on the X-from-count plumbing.
            d.set_item(
                "kind",
                "mill_target_player_equal_to_your_life_total",
            )?;
        }
        Effect::ExileTarget { target } => {
            d.set_item("kind", "exile_target")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::TapTarget { target } => {
            d.set_item("kind", "tap_target")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::UntapTarget { target } => {
            d.set_item("kind", "untap_target")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::Sacrifice { target } => {
            d.set_item("kind", "sacrifice")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::AddCounters { target, count } => {
            d.set_item("kind", "add_counters")?;
            d.set_item("count", *count)?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::DiscardCard { target } => {
            d.set_item("kind", "discard_card")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::ScryN { count } => {
            d.set_item("kind", "scry_n")?;
            d.set_item("count", *count)?;
        }
        Effect::DestroyTarget { target } => {
            d.set_item("kind", "destroy_target")?;
            d.set_item("target", target_filter_to_pydict(py, target)?)?;
        }
        Effect::DestroyTargetCreatureThenDrawIfUndamaged { draw_amount } => {
            // BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB.
            // Round-trip dict-shape: kind + draw_amount only (no
            // target key — the defender-bf first-slice picker derives
            // its target deterministically, mirroring the
            // ExileEachOpponentsGraveyard / EachOpponentMillsCards
            // unit-shape convention).
            d.set_item("kind", "destroy_target_creature_then_draw_if_undamaged")?;
            d.set_item("draw_amount", *draw_amount)?;
        }
        Effect::DamageEachOpponent { amount } => {
            d.set_item("kind", "damage_each_opponent")?;
            d.set_item("amount", *amount)?;
        }
        Effect::DamageEachCreature { amount } => {
            d.set_item("kind", "damage_each_creature")?;
            d.set_item("amount", *amount)?;
        }
        Effect::CreateToken { count, token_printing_id } => {
            d.set_item("kind", "create_token")?;
            d.set_item("count", *count)?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::CreateTokenEachPlayer { count, token_printing_id } => {
            // BLOCKED-RUST-EFFECT slice (round 43). Two-field schema
            // mirrors the sister `CreateToken` arm above; the
            // each-player vs self-only choice rides on the variant
            // name (and thus on the dict `kind` string), not on a
            // boolean field. Mirrors Python's
            // `CreateToken(each_player=True)` -- the Python class
            // has a single `each_player: bool` field, but the bridge
            // dict-shape uses a distinct `kind` for round-trip
            // clarity.
            d.set_item("kind", "create_token_each_player")?;
            d.set_item("count", *count)?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::CreateTokenTapped { count, token_printing_id } => {
            // Phase 6 token-printing substrate -- Phase 4 (round 50).
            // Two-field schema mirrors the sister `CreateToken` arm
            // above; the tapped-vs-untapped choice rides on the
            // variant name (and thus on the dict `kind` string), not
            // on a boolean field. Mirrors Python's
            // `CreateToken(tapped=True)` -- the Python class has a
            // single `tapped: bool` field, but the bridge dict-shape
            // uses a distinct `kind` for round-trip clarity.
            d.set_item("kind", "create_token_tapped")?;
            d.set_item("count", *count)?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::AddManaToPool { color, amount } => {
            d.set_item("kind", "add_mana_to_pool")?;
            d.set_item("color", color)?;
            d.set_item("amount", *amount)?;
        }
        Effect::SearchLibraryForBasicLandTapped { count } => {
            d.set_item("kind", "search_library_for_basic_land_tapped")?;
            d.set_item("count", *count)?;
        }
        Effect::PutPlusOneCounterOnSelf { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_self")?;
            d.set_item("amount", *amount)?;
        }
        Effect::Surveil { count } => {
            d.set_item("kind", "surveil")?;
            d.set_item("count", *count)?;
        }
        Effect::CounterTargetSpell => {
            d.set_item("kind", "counter_target_spell")?;
        }
        Effect::ReturnCreatureFromYourGraveyardToBattlefield => {
            d.set_item("kind", "return_creature_from_your_graveyard_to_battlefield")?;
        }
        Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield => {
            d.set_item(
                "kind",
                "return_enchantment_or_creature_from_your_graveyard_to_battlefield",
            )?;
        }
        Effect::DiscardSelf { amount } => {
            d.set_item("kind", "discard_self")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentLosesLife { amount } => {
            d.set_item("kind", "each_opponent_loses_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentLosesLifeEqualToArtifactCount => {
            // BLOCKED-RUST-EFFECT slice (round 44). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // battlefield artifact count). Mirrors Python's
            // EachOpponentLosesLifeEqualToArtifactCount dict shape (no
            // static `amount` field on the @dataclass). Sister of
            // round-26 EachOpponentLosesLife with the same recipient
            // set and apply step but a dynamic-X amount via the
            // artifact-count denominator (FIFTH apply substrate to
            // pick up the round-32 X-from-count plumbing).
            d.set_item(
                "kind",
                "each_opponent_loses_life_equal_to_artifact_count",
            )?;
        }
        Effect::EachOpponentLosesLifeEqualToYourLifeTotal => {
            // BLOCKED-RUST-EFFECT slice (round 48). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // current life total, clamped to >= 0). Mirrors Python's
            // EachOpponentLosesLifeEqualToYourLifeTotal dict shape (no
            // static `amount` field on the @dataclass). Sister of
            // round-44 EachOpponentLosesLifeEqualToArtifactCount with
            // the same recipient set and apply step but a life-total
            // denominator (SECOND denominator on the life-loss-each-
            // opponent apply substrate; life-total precedent set by
            // round-42 mill / round-47 damage on other substrates).
            d.set_item(
                "kind",
                "each_opponent_loses_life_equal_to_your_life_total",
            )?;
        }
        Effect::GainLifeEqualToArtifactCount => {
            // BLOCKED-RUST-EFFECT slice (round 45). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // battlefield artifact count). Mirrors Python's
            // GainLifeEqualToArtifactCount dict shape (no static
            // `amount` field on the @dataclass). Sister of round-26
            // GainLife with the SAME recipient (the source's
            // controller) and the SAME apply step (direct life
            // addition) but a dynamic-X amount via the artifact-count
            // denominator (SIXTH apply substrate to pick up the
            // round-32 X-from-count plumbing). Controller-self mirror
            // of round-44 EachOpponentLosesLifeEqualToArtifactCount
            // (same denominator, inverted recipient + sign).
            d.set_item(
                "kind",
                "gain_life_equal_to_artifact_count",
            )?;
        }
        Effect::GainLifeEqualToYourLifeTotal => {
            // BLOCKED-RUST-EFFECT slice (round 49). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // current life total, clamped to >= 0). Mirrors Python's
            // GainLifeEqualToYourLifeTotal dict shape (no static
            // `amount` field on the @dataclass). Sister of round-45
            // GainLifeEqualToArtifactCount with the SAME recipient
            // (the source's controller) and the SAME apply step
            // (direct life addition via gain_life) but a life-total
            // denominator (SECOND denominator on the gain-life-
            // controller-self apply substrate; life-total precedent
            // set by round-42 mill / round-47 damage / round-48
            // each-opponent-loses on other substrates).
            d.set_item(
                "kind",
                "gain_life_equal_to_your_life_total",
            )?;
        }
        Effect::LoseLife { amount } => {
            d.set_item("kind", "lose_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::TargetOpponentLosesLife { amount } => {
            d.set_item("kind", "target_opponent_loses_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::TargetPlayerLosesLife { amount } => {
            // PPP Pin 1 (10th Rust slice). Round-trip pair with the
            // `target_player_loses_life` parse_effect arm.
            d.set_item("kind", "target_player_loses_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentDiscards { count } => {
            d.set_item("kind", "each_opponent_discards")?;
            d.set_item("count", *count)?;
        }
        Effect::DestroyAllCreatures => {
            d.set_item("kind", "destroy_all_creatures")?;
        }
        Effect::ExileAllCreaturesAndSpacecraft => {
            // Beyond-the-Quiet slice (2026-05-21). Bare wrath — kind
            // only (mirrors `DestroyAllCreatures` to_dict shape).
            d.set_item("kind", "exile_all_creatures_and_spacecraft")?;
        }
        Effect::DestroyTargetNonlandPermanent => {
            d.set_item("kind", "destroy_target_nonland_permanent")?;
        }
        Effect::DestroyTargetArtifact => {
            d.set_item("kind", "destroy_target_artifact")?;
        }
        Effect::DestroyTargetLand => {
            d.set_item("kind", "destroy_target_land")?;
        }
        Effect::DestroyTargetNoncreaturePermanent => {
            d.set_item("kind", "destroy_target_noncreature_permanent")?;
        }
        Effect::DestroyTargetArtifactOrTappedCreature => {
            d.set_item("kind", "destroy_target_artifact_or_tapped_creature")?;
        }
        Effect::SeedshipImpactEffect { token_printing_id } => {
            d.set_item("kind", "seedship_impact_effect")?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::DealDamageToTargetCreatureWithExcessLanderRider {
            amount,
            token_printing_id,
        } => {
            d.set_item(
                "kind",
                "deal_damage_to_target_creature_with_excess_lander_rider",
            )?;
            d.set_item("amount", *amount)?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::DepressurizeEffect { power_penalty } => {
            d.set_item("kind", "depressurize_effect")?;
            d.set_item("power_penalty", *power_penalty)?;
        }
        Effect::BiosynthicBurstEffect => {
            // Unit variant — `kind` only (mirrors
            // `each_opponent_sacrifices_creature`).
            d.set_item("kind", "biosynthic_burst_effect")?;
        }
        Effect::FocusFireEffect { base } => {
            // Single `base` int field (the `deal_damage_any_target`
            // single-int-field to_dict shape). Bidirectional with the
            // `focus_fire_effect` parse arm + the cross_engine.py
            // bridge arm + the trajectory.py reverse-serializer leaf.
            d.set_item("kind", "focus_fire_effect")?;
            d.set_item("base", *base)?;
        }
        Effect::DiplomaticRelationsEffect => {
            // Effect-uncovered Tier-A #7 — Diplomatic Relations
            // (spell-card-bridge Phase 2). Unit variant — `kind`
            // only (the `biosynthic_burst_effect` field-free
            // to_dict idiom). Bidirectional with the
            // `diplomatic_relations_effect` parse arm + the
            // cross_engine.py bridge arm + the trajectory.py
            // reverse-serializer leaf. The 2-slot targets are NOT
            // serialized here — they ride on the StackItem.
            d.set_item("kind", "diplomatic_relations_effect")?;
        }
        Effect::CutPropulsionEffect => {
            // Effect-uncovered Tier-A — Cut Propulsion. Unit variant
            // — `kind` only (the `biosynthic_burst_effect` /
            // `diplomatic_relations_effect` field-free to_dict idiom).
            // Bidirectional with the `cut_propulsion_effect` parse
            // arm + the cross_engine.py bridge arm + the trajectory.py
            // reverse-serializer leaf. The single target is NOT
            // serialized here — it rides on the StackItem.
            d.set_item("kind", "cut_propulsion_effect")?;
        }
        Effect::InvasiveManeuversEffect { base, boosted } => {
            // Effect-uncovered Tier-A — Invasive Maneuvers. Two-int
            // variant — `kind` + `base` + `boosted` (the
            // `focus_fire_effect` single-int-field to_dict shape,
            // widened to two ints to mirror Python's
            // `base`/`boosted` dataclass fields). Bidirectional with
            // the `invasive_maneuvers_effect` parse arm + the
            // cross_engine.py bridge arm + the trajectory.py
            // reverse-serializer leaf. The single target is NOT
            // serialized here — it rides on the StackItem.
            d.set_item("kind", "invasive_maneuvers_effect")?;
            d.set_item("base", *base)?;
            d.set_item("boosted", *boosted)?;
        }
        Effect::DestroyTargetArtifactEnchantmentOrFlyingCreature => {
            // Effect-uncovered single-gated tier — Shattered Wings.
            // Unit variant — `kind` only (the
            // `destroy_target_artifact_or_tapped_creature` field-free
            // to_dict idiom). Bidirectional with the
            // `destroy_target_artifact_enchantment_or_flying_creature`
            // parse arm + the cross_engine.py bridge arm + the
            // trajectory.py reverse-serializer leaf. The single
            // target is NOT serialized here — it rides on the
            // StackItem.
            d.set_item(
                "kind",
                "destroy_target_artifact_enchantment_or_flying_creature",
            )?;
        }
        Effect::ExorciseEffect => {
            // Effect-uncovered single-gated tier — Exorcise.
            // Unit variant — `kind` only (the Shattered Wings
            // field-free to_dict idiom). Bidirectional with the
            // `exorcise_effect` parse arm + the cross_engine.py
            // bridge arm + the trajectory.py reverse-serializer
            // leaf. The single target is NOT serialized here — it
            // rides on the StackItem
            // (`current_resolve_payload_target_id`).
            d.set_item("kind", "exorcise_effect")?;
        }
        Effect::CloseEncounterEffect => {
            // Close Encounter — additional-cost-choice closure
            // (Cycle 3, 2026-05-21; ~16,098 17lands decks). Unit
            // variant — `kind` only (the Exorcise / Diplomatic
            // Relations field-free to_dict idiom). Bidirectional
            // with the `close_encounter_effect` parse arm + the
            // `cross_engine.py` bridge arm + the `trajectory.py`
            // reverse-serializer leaf. The single damage-target is
            // NOT serialized here — it rides on the StackItem
            // (`current_resolve_payload_target_id`). The
            // additional-cost choice is also NOT serialized here —
            // it rides on the StackItem
            // (`additional_cost_choice_id`, Cycle 1 substrate).
            d.set_item("kind", "close_encounter_effect")?;
        }
        Effect::EmergencyEjectEffect { token_printing_id } => {
            // Effect-uncovered single-gated tier — Emergency Eject.
            // Single-field payload — `kind` + `token_printing_id`
            // (the same pid-carrier to_dict shape as
            // `seedship_impact_effect`). Bidirectional with the
            // `emergency_eject_effect` parse arm + the cross_engine.py
            // bridge arm + the trajectory.py reverse-serializer leaf.
            // The single target is NOT serialized here — it rides
            // on the StackItem (`current_resolve_payload_target_id`).
            d.set_item("kind", "emergency_eject_effect")?;
            d.set_item("token_printing_id", *token_printing_id)?;
        }
        Effect::DealDamageAnyTarget { amount } => {
            d.set_item("kind", "deal_damage_any_target")?;
            d.set_item("amount", *amount)?;
        }
        Effect::PumpSelfUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_self_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::PumpTargetCreatureUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_target_creature_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::PumpTargetCreatureUntilEotGrantDiesReturnTapped {
            power_bonus,
            toughness_bonus,
        } => {
            // BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner.
            // Round-trips the same two fields the parse arm reads; the
            // another/you-control restriction + dies-return-tapped
            // rider are variant-intrinsic (no extra keys).
            d.set_item(
                "kind",
                "pump_target_creature_until_eot_grant_dies_return_tapped",
            )?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::FightTargetCreatureWithTargetCreature => {
            d.set_item("kind", "fight_target_creature_with_target_creature")?;
        }
        Effect::DoubleTargetCreaturesPowerUEOT => {
            // BLOCKED-RUST-EFFECT slice (round 31). Unit variant; no
            // payload fields beyond `kind`. Mirrors Python's
            // DoubleTargetCreaturesPowerUEOT dict shape.
            d.set_item("kind", "double_target_creatures_power_ueot")?;
        }
        Effect::DealDamageToOpponentCreatureByArtifactCount => {
            // BLOCKED-RUST-EFFECT slice (round 32). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // battlefield artifact count). Mirrors Python's
            // DealDamageToOpponentCreatureByArtifactCount dict shape
            // (no static `amount` field on the @dataclass).
            d.set_item("kind", "deal_damage_to_opponent_creature_by_artifact_count")?;
        }
        Effect::TwoSubjectPumpAndDebuffByArtifactCount => {
            // BLOCKED-RUST-EFFECT slice (round 51). Unit variant; no
            // payload fields beyond `kind` (X is dynamic — computed
            // at resolve time from the source-controller's
            // battlefield artifact count, clamped >= 0). Mirrors
            // Python's PumpYoursAndDebuffOpponentByArtifactCount
            // UntilEOT dict shape (no static field on the
            // @dataclass). Sister of round-32
            // DealDamageToOpponentCreatureByArtifactCount with the
            // SAME artifact-count denominator read but a two-subject
            // until-EOT pump+debuff apply substrate.
            d.set_item(
                "kind",
                "two_subject_pump_and_debuff_by_artifact_count",
            )?;
        }
        Effect::DealDamageToOpponentCreatureByCardsInHand => {
            // BLOCKED-RUST-EFFECT slice (round 35). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // hand size). Mirrors Python's
            // DealDamageToOpponentCreatureByCardsInHand dict shape
            // (no static `amount` field on the @dataclass). Sister of
            // round-32 with the same target shape but a different
            // dynamic-X denominator.
            d.set_item("kind", "deal_damage_to_opponent_creature_by_cards_in_hand")?;
        }
        Effect::DealDamageToOpponentCreatureByLandsControlled => {
            // BLOCKED-RUST-EFFECT slice (round 36). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // battlefield lands count). Mirrors Python's
            // DealDamageToOpponentCreatureByLandsControlled dict shape
            // (no static `amount` field on the @dataclass). Sister of
            // round-32 / round-35 with the same target shape but a
            // different dynamic-X denominator (lands-controlled in
            // place of artifact-count / hand-size).
            d.set_item(
                "kind",
                "deal_damage_to_opponent_creature_by_lands_controlled",
            )?;
        }
        Effect::DealDamageToOpponentCreatureByCreaturesYouControl => {
            // BLOCKED-RUST-EFFECT slice (round 37). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // battlefield creatures count). Mirrors Python's
            // DealDamageToOpponentCreatureByCreaturesYouControl dict
            // shape (no static `amount` field on the @dataclass).
            // Sister of round-32 / round-35 / round-36 with the same
            // target shape but a different dynamic-X denominator
            // (creatures-controlled in place of artifact-count /
            // hand-size / lands-controlled).
            d.set_item(
                "kind",
                "deal_damage_to_opponent_creature_by_creatures_you_control",
            )?;
        }
        Effect::DealDamageToOpponentCreatureByCardsInGraveyard => {
            // BLOCKED-RUST-EFFECT slice (round 46). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // graveyard size). Mirrors Python's
            // DealDamageToOpponentCreatureByCardsInGraveyard dict shape
            // (no static `amount` field on the @dataclass). Sister of
            // round-32 / round-35 / round-36 / round-37 with the same
            // target shape but a different dynamic-X denominator
            // (cards-in-graveyard in place of artifact-count /
            // hand-size / lands-controlled / creatures-controlled).
            // FIFTH denominator on the damage substrate; sixth overall
            // X-from-count denominator across the substrate matrix.
            d.set_item(
                "kind",
                "deal_damage_to_opponent_creature_by_cards_in_graveyard",
            )?;
        }
        Effect::DealDamageToOpponentCreatureByLifeTotal => {
            // BLOCKED-RUST-EFFECT slice (round 47). Unit variant; no
            // payload fields beyond `kind` (count is dynamic --
            // computed at resolve time from the source-controller's
            // clamped life total). Mirrors Python's
            // DealDamageToOpponentCreatureByLifeTotal dict shape (no
            // static `amount` field on the @dataclass). Sister of
            // round-32 / round-35 / round-36 / round-37 / round-46
            // with the same target shape but a different dynamic-X
            // denominator (life-total in place of artifact-count /
            // hand-size / lands-controlled / creatures-controlled /
            // cards-in-graveyard). SIXTH (final) denominator on the
            // damage substrate -- the damage substrate row of the
            // X-from-count matrix is now fully saturated.
            d.set_item(
                "kind",
                "deal_damage_to_opponent_creature_by_life_total",
            )?;
        }
        Effect::PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact => {
            // BLOCKED-RUST-EFFECT slice (round 33). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // battlefield other-creature-or-artifact count). Mirrors
            // Python's PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact
            // dict shape (no static `amount` field on the @dataclass).
            d.set_item(
                "kind",
                "put_plus_one_counters_on_self_per_other_creature_or_artifact",
            )?;
        }
        Effect::SurveilXEqualToArtifactCount => {
            // BLOCKED-RUST-EFFECT slice (round 34). Unit variant; no
            // payload fields beyond `kind` (count is dynamic —
            // computed at resolve time from the source-controller's
            // battlefield artifact count). Mirrors Python's
            // Surveil(use_artifact_count=True) dict shape — the
            // `amount` field is ignored on the dynamic lane (the
            // dispatch happens on the variant tag, not a runtime
            // flag).
            d.set_item("kind", "surveil_x_equal_to_artifact_count")?;
        }
        Effect::PumpCreaturesYouControlUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_creatures_you_control_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::ReturnCreatureFromYourGraveyardToHand => {
            d.set_item("kind", "return_creature_from_your_graveyard_to_hand")?;
        }
        Effect::ReturnArtifactOrCreatureFromYourGraveyardToHand => {
            d.set_item(
                "kind",
                "return_artifact_or_creature_from_your_graveyard_to_hand",
            )?;
        }
        Effect::ReturnCreatureOrSpacecraftFromYourGraveyardToHand => {
            d.set_item(
                "kind",
                "return_creature_or_spacecraft_from_your_graveyard_to_hand",
            )?;
        }
        Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToHand => {
            d.set_item(
                "kind",
                "return_enchantment_or_creature_from_your_graveyard_to_hand",
            )?;
        }
        Effect::ReturnTargetArtifactCardFromYourGraveyardToHand => {
            d.set_item(
                "kind",
                "return_target_artifact_card_from_your_graveyard_to_hand",
            )?;
        }
        Effect::PutPlusOneCounterOnEachCreatureYouControl { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_each_creature_you_control")?;
            d.set_item("amount", *amount)?;
        }
        Effect::PutPlusOneCounterOnCreatureYouControl { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_creature_you_control")?;
            d.set_item("amount", *amount)?;
        }
        Effect::PutPlusOneCounterOnTargetCreature { amount, attacking_only } => {
            // PPP Pin 2 (10th Rust slice). Round-trip pair with the
            // `put_plus_one_counter_on_target_creature` parse_effect
            // arm. `attacking_only` defaults to false on the Python
            // side; always emit the field so the round-trip preserves
            // the rider explicitly.
            d.set_item("kind", "put_plus_one_counter_on_target_creature")?;
            d.set_item("amount", *amount)?;
            d.set_item("attacking_only", *attacking_only)?;
        }
        Effect::PutPlusOneCounterOnOpponentsCreature { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_opponents_creature")?;
            d.set_item("amount", *amount)?;
        }
        Effect::ReturnTargetPermanentYouControlToOwnersHand => {
            d.set_item("kind", "return_target_permanent_you_control")?;
        }
        Effect::AddNManaOfChosenColor { amount, options } => {
            d.set_item("kind", "add_n_mana_of_chosen_color")?;
            d.set_item("amount", *amount)?;
            d.set_item("options", options.clone())?;
        }
        Effect::SearchLibraryForCard {
            filter_name,
            max_mv,
            may_fail,
        } => {
            d.set_item("kind", "search_library_for_card")?;
            d.set_item("filter_name", filter_name.clone())?;
            d.set_item("max_mv", *max_mv)?;
            d.set_item("may_fail", *may_fail)?;
        }
        Effect::PumpCreaturesOpponentsControlUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_creatures_opponents_control_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::PumpCreaturesOpponentsControlUntilCasterNextTurn {
            power_bonus,
            toughness_bonus,
        } => {
            d.set_item("kind", "pump_creatures_opponents_control_until_caster_next_turn")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::PutCardFromGraveyardOnBottomOfLibrary => {
            d.set_item("kind", "put_card_from_graveyard_on_bottom_of_library")?;
        }
        Effect::MoveSelfCountersToTargetCreatureYouControl => {
            d.set_item("kind", "move_self_counters_to_target_creature_you_control")?;
        }
        Effect::DealDamageEqualToSourcePowerAnyTarget => {
            d.set_item("kind", "deal_damage_equal_to_source_power_any_target")?;
        }
        Effect::TapTargetArtifactOrCreature => {
            d.set_item("kind", "tap_target_artifact_or_creature")?;
        }
        Effect::ConditionalOnTappedCreatures { effect, min_count } => {
            d.set_item("kind", "conditional_on_tapped_creatures")?;
            d.set_item("effect", effect_to_pydict(py, effect)?)?;
            d.set_item("min_count", *min_count)?;
        }
        Effect::PutPlusOneCounterOnEachOfUpToNTargetCreatures { max_n } => {
            d.set_item("kind", "put_plus_one_counter_on_each_of_up_to_n_target_creatures")?;
            d.set_item("max_n", *max_n)?;
        }
        Effect::PutChargeCountersOnSelf { amount, amount_from_x } => {
            d.set_item("kind", "put_charge_counters_on_self")?;
            d.set_item("amount", *amount)?;
            d.set_item("amount_from_x", *amount_from_x)?;
        }
        Effect::CreateTokenCopyOfTargetCreatureYouControl => {
            d.set_item("kind", "create_token_copy_of_target_creature_you_control")?;
        }
        Effect::OptionalSacArtifactForEffect { inner_effect } => {
            d.set_item("kind", "optional_sac_artifact_for_effect")?;
            d.set_item("inner_effect", effect_to_pydict(py, inner_effect)?)?;
        }
        Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary => {
            d.set_item("kind", "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library")?;
        }
        Effect::AttachToTargetCreatureYouControl => {
            d.set_item("kind", "attach_to_target_creature_you_control")?;
        }
        Effect::EquipToTargetCreatureYouControl => {
            d.set_item("kind", "equip_to_target_creature_you_control")?;
        }
        Effect::DestroyEnchantedPermanent => {
            d.set_item("kind", "destroy_enchanted_permanent")?;
        }
        Effect::TapEnchantedPermanent => {
            d.set_item("kind", "tap_enchanted_permanent")?;
        }
        Effect::TargetOpponentLookHandMayDiscard => {
            // **Cracked Skull look-at-hand + may-discard substrate —
            // Cycle 2 (2026-05-22) — to_pydict.** Unit variant; no
            // payload fields beyond `kind` (mirrors Python's field-
            // free `@dataclass(frozen=True)` at
            // `effects/generic/discard.py:298-318`). Round-trip dict-
            // shape consistency (slice-19 / slice-20 unit-variant
            // precedent).
            d.set_item("kind", "target_opponent_look_hand_may_discard")?;
        }
        Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted => {
            // **Sporogenic Infection — Cycle 1 (2026-05-22) — to_pydict.**
            // Unit variant; no payload fields beyond `kind`.
            // Round-trip dict-shape consistency with the sibling
            // `target_opponent_look_hand_may_discard` arm.
            d.set_item(
                "kind",
                "target_opponent_sacrifices_creature_other_than_enchanted",
            )?;
        }
        Effect::PutStunCounterOnTappedCreature => {
            // Round-30 stun-counter substrate. Unit variant; no
            // payload fields beyond `kind`. Mirrors Python's
            // PutStunCounterOnTappedCreature dict shape.
            d.set_item("kind", "put_stun_counter_on_tapped_creature")?;
        }
        Effect::TapTargetOpponentPermanentAndStunIt => {
            // Round-30 stun-counter substrate. Unit variant; no
            // payload fields beyond `kind`. Mirrors Python's
            // TapTargetOpponentPermanentAndStunIt dict shape.
            d.set_item("kind", "tap_target_opponent_permanent_and_stun_it")?;
        }
        Effect::ExileTargetCreatureUntilSourceLeaves => {
            d.set_item("kind", "exile_target_creature_until_source_leaves")?;
        }
        Effect::ExileOpponentsNonlandPermanentUntilLeaves => {
            d.set_item("kind", "exile_opponents_nonland_permanent_until_leaves")?;
        }
        Effect::ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves { max_mv } => {
            d.set_item(
                "kind",
                "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
            )?;
            d.set_item("max_mv", *max_mv)?;
        }
        Effect::MayDo { inner_effect, prompt } => {
            d.set_item("kind", "may_do")?;
            d.set_item("inner_effect", effect_to_pydict(py, inner_effect)?)?;
            d.set_item("prompt", prompt.as_str())?;
        }
        Effect::LookTopNPutOneInHandRestToGraveyard { n } => {
            d.set_item("kind", "look_top_n_put_one_in_hand_rest_to_graveyard")?;
            d.set_item("n", *n)?;
        }
        Effect::FamishedWorldsireLookTopXByPowerPutLandsTapped => {
            // BLOCKED-RUST-EFFECT slice (round 54). Unit variant; no
            // payload fields beyond `kind` (X dynamic — source
            // effective power at resolve; greedy-all land selection).
            // Mirrors Python's `FamishedWorldsireETBEffect` dict shape
            // (no static fields on the @dataclass). No other keys
            // emitted (round-trip dict-shape consistency convention —
            // round-49 `gain_life_equal_to_your_life_total` precedent).
            d.set_item(
                "kind",
                "famished_worldsire_look_top_x_by_power_put_lands_tapped",
            )?;
        }
        Effect::AstelliReclaimerETBEffect => {
            // 16th Rust slice (TTT Pin 3 closure). Unit variant; no
            // payload fields beyond `kind` (X dynamic — source
            // printing's mana_cost/warp_cost at resolve; deterministic
            // first-legal graveyard target selection). Mirrors
            // Python's `AstelliReclaimerETBEffect` dict shape (no
            // static fields on the @dataclass). No other keys emitted
            // (round-trip dict-shape consistency convention —
            // round-54 `famished_worldsire_look_top_x_by_power_put_
            // lands_tapped` precedent).
            d.set_item("kind", "astelli_reclaimer_etb_effect")?;
        }
        Effect::RovingActuatorETBEffect { max_mv } => {
            // 18th Rust slice (XXX Pin 2 closure). Single-field
            // variant — `max_mv: u8` mirrors Python's
            // `RovingActuatorETBEffect(max_mv: int = 2)` dataclass
            // field. Always emit the key so the dict round-trips
            // back to the same variant payload.
            d.set_item("kind", "roving_actuator_etb")?;
            d.set_item("max_mv", *max_mv as i64)?;
        }
        Effect::EnchantedCreatureFightsTarget => {
            // 19th Rust slice (honest-stop substrate landing). Unit
            // variant; no payload fields beyond `kind` (mirrors Python's
            // field-free @dataclass(frozen=True) at bespoke.py:506-507).
            // No other keys emitted — round-trip dict-shape consistency
            // (slice-16 `astelli_reclaimer_etb_effect` precedent).
            d.set_item("kind", "enchanted_creature_fights_target")?;
        }
        Effect::PainForAllETBEffect => {
            // **Pain for All closure (Aura Phase 3 / S4) — to_pydict.**
            // Unit variant; mirrors Python's field-free @dataclass
            // (frozen=True) at bespoke.py:352-428. Round-trip
            // consistency (slice-19 `enchanted_creature_fights_target`
            // precedent).
            d.set_item("kind", "pain_for_all_etb")?;
        }
        Effect::PainForAllDamageEchoEffect => {
            // **Pain for All closure (Aura Phase 3 / S4) — to_pydict.**
            // Unit variant; mirrors Python's field-free @dataclass
            // (frozen=True) at bespoke.py:432-463. Round-trip
            // consistency (slice-19 `enchanted_creature_fights_target`
            // precedent).
            d.set_item("kind", "pain_for_all_damage_echo")?;
        }
        Effect::SacrificeSelfUnlessVoidActive => {
            // **Aura Phase 4 (Chorale of the Void substrate landing)
            // — to_pydict.** Unit variant; mirrors Python's field-free
            // @dataclass(frozen=True) at combinators.py:172-200. No
            // other keys emitted — round-trip dict-shape consistency
            // (slice-19 `enchanted_creature_fights_target` / Pain
            // for All `pain_for_all_etb` precedent).
            d.set_item("kind", "sacrifice_self_unless_void_active")?;
        }
        Effect::SacrificeSelfUnlessTapPermanent => {
            // 20th Rust slice (VVV Pin 2 closure). Unit variant; no
            // payload fields beyond `kind` (mirrors Python's field-free
            // class at combinators.py:96-168 — no @dataclass decorator,
            // no fields). No other keys emitted — round-trip dict-shape
            // consistency (slice-19 `enchanted_creature_fights_target`
            // precedent).
            d.set_item("kind", "sacrifice_self_unless_tap_permanent")?;
        }
        Effect::EachOpponentExilesCardFromHandWithCastPermission => {
            // 20th Rust slice (XXX Pin 3 honest-stop closure). Unit
            // variant; no payload fields beyond `kind` (mirrors
            // Python's field-free @dataclass(frozen=True) at
            // `effects/generic/discard.py:172-214`). No other keys
            // emitted -- round-trip dict-shape consistency (slice-16
            // / slice-19 unit-variant precedent).
            d.set_item("kind", "each_opponent_exiles_card_from_hand_lightstall")?;
        }
        Effect::HouseCartographerSurvivalEffect => {
            // 21st Rust slice (XXX Pin 1 closure). Unit variant; no
            // payload fields beyond `kind` (mirrors Python's
            // field-free @dataclass(frozen=True) at
            // `effects/bespoke.py:2667-2733`). No other keys emitted
            // -- round-trip dict-shape consistency (slice-16 /
            // slice-19 / slice-20 unit-variant precedent).
            d.set_item("kind", "house_cartographer_survival")?;
        }
        Effect::EachOpponentExilesOwnCreature { link_to_source } => {
            // 22nd Rust slice (Sothera + Requiem honest-stop).
            // Single-field variant — emit `link_to_source` so the dict
            // round-trips back to the same variant payload.
            d.set_item("kind", "each_opponent_exiles_own_creature")?;
            d.set_item("link_to_source", *link_to_source)?;
        }
        Effect::ReanimateOneExiledWithSourceWithCounters { counter_amount } => {
            // 22nd Rust slice (Sothera + Requiem honest-stop).
            // Single-field variant — emit `counter_amount` for round-
            // trip dict-shape consistency.
            d.set_item("kind", "reanimate_one_exiled_with_source_with_counters")?;
            d.set_item("counter_amount", *counter_amount)?;
        }
        Effect::SacrificeSelfPreservingExileLink => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant; no payload fields beyond `kind`.
            d.set_item("kind", "sacrifice_self_preserving_exile_link")?;
        }
        Effect::GrantTriggeredAbilityUEOTRequiem => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant on the Rust side; the Python `ability` payload
            // is reconstructed by the bridge arm from a fixed Requiem-
            // shaped TriggeredAbility (this Rust variant is reserved
            // for the Requiem-shape grant until the general grant-
            // triggered-ability registry lands). No fields emitted.
            d.set_item("kind", "grant_triggered_ability_ueot_requiem")?;
        }
        Effect::MayHaveSourceDealDamageToTarget { amount } => {
            // 22nd Rust slice (Sothera + Requiem honest-stop).
            // Single-field variant — emit `amount` for round-trip
            // dict-shape consistency.
            d.set_item("kind", "may_have_source_deal_damage_to_target")?;
            d.set_item("amount", *amount)?;
        }
        Effect::DrawCardsAndLoseLifeEqualToResolvingDamage => {
            // 22nd Rust slice (Sothera + Requiem honest-stop). Unit
            // variant; no payload fields beyond `kind`.
            d.set_item("kind", "draw_cards_and_lose_life_equal_to_resolving_damage")?;
        }
        Effect::ChooseOneMode { modes, or_both } => {
            d.set_item("kind", "choose_one_mode")?;
            let mode_dicts: Vec<Bound<'_, PyDict>> = modes
                .iter()
                .map(|m| effect_to_pydict(py, m))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("modes", mode_dicts)?;
            // **scour-for-scrap-slice (2026-05-21).** Emit `or_both`
            // ONLY when `true` so every pre-widen ChooseOneMode dict
            // round-trips byte-identical (the parse arm defaults the
            // missing field to `false`; this asymmetry is intentional
            // and mirrors the `granted_keywords`-when-non-empty idiom
            // a few siblings above).
            if *or_both {
                d.set_item("or_both", true)?;
            }
        }
        Effect::IfYouDo { do_action, then_action } => {
            d.set_item("kind", "if_you_do")?;
            d.set_item("do_action", effect_to_pydict(py, do_action)?)?;
            d.set_item("then_action", effect_to_pydict(py, then_action)?)?;
        }
        Effect::EachOpponentMillsCards { count } => {
            d.set_item("kind", "each_opponent_mills_cards")?;
            d.set_item("count", *count)?;
        }
        Effect::ExileEachOpponentsGraveyard => {
            d.set_item("kind", "exile_each_opponents_graveyard")?;
        }
        Effect::PumpTargetArtifactOrCreatureUntilEOT {
            power_bonus,
            toughness_bonus,
            granted_keywords,
        } => {
            d.set_item("kind", "pump_target_artifact_or_creature_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
            // Reroute Systems closure (2026-05-21): emit
            // `granted_keywords` ONLY when non-empty so existing
            // round-trips (every pre-closure dict that omits the
            // field) stay byte-identical. Mirrors the
            // `GrantKeywordsToTargetCreatureUntilEOT.keywords` emit
            // shape (Vec<String> → Python list-of-strings).
            if !granted_keywords.is_empty() {
                let kw_list: Vec<&str> =
                    granted_keywords.iter().map(|s| s.as_str()).collect();
                d.set_item("granted_keywords", kw_list)?;
            }
        }
        Effect::DealDamageToTargetTappedCreature { amount } => {
            d.set_item("kind", "deal_damage_to_target_tapped_creature")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentSacrificesCreature => {
            d.set_item("kind", "each_opponent_sacrifices_creature")?;
        }
        Effect::GrantKeywordToTargetCreatureUntilEOT { keyword } => {
            d.set_item("kind", "grant_keyword_to_target_creature_until_eot")?;
            d.set_item("keyword", keyword.as_str())?;
        }
        Effect::GrantKeywordsToTargetCreatureUntilEOT { keywords, exclude_source } => {
            d.set_item("kind", "grant_keywords_to_target_creature_until_eot")?;
            // Round-trip the keyword list as a Python list-of-strings.
            let kw_list: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
            d.set_item("keywords", kw_list)?;
            d.set_item("exclude_source", *exclude_source)?;
        }
        Effect::UntapTargetOtherPermanent => {
            d.set_item("kind", "untap_target_other_permanent")?;
        }
        Effect::TapAnotherTargetCreature => {
            d.set_item("kind", "tap_another_target_creature")?;
        }
        Effect::ReturnAllCreaturesFromYourGraveyardToBattlefield => {
            d.set_item("kind", "return_all_creatures_from_your_graveyard_to_battlefield")?;
        }
        Effect::SetBasePTUntilEOT { power, toughness } => {
            d.set_item("kind", "set_base_pt_until_eot")?;
            d.set_item("power", *power)?;
            d.set_item("toughness", *toughness)?;
        }
        Effect::ManifestDread { count } => {
            d.set_item("kind", "manifest_dread")?;
            d.set_item("count", *count)?;
        }
        Effect::ManifestDreadThenAttachSelf { count } => {
            d.set_item("kind", "manifest_dread_then_attach_self")?;
            d.set_item("count", *count)?;
        }
        Effect::ManifestDreadThenPutPlusOneCountersOnIt { count, counter_amount } => {
            d.set_item("kind", "manifest_dread_then_put_plus_one_counters_on_it")?;
            d.set_item("count", *count)?;
            d.set_item("counter_amount", *counter_amount)?;
        }
        Effect::ConditionalIfVoid { then_effect, else_effect } => {
            d.set_item("kind", "conditional_if_void")?;
            d.set_item("then_effect", effect_to_pydict(py, then_effect)?)?;
            match else_effect {
                Some(inner) => {
                    d.set_item("else_effect", effect_to_pydict(py, inner)?)?;
                }
                None => {
                    d.set_item("else_effect", py.None())?;
                }
            }
        }
        Effect::DealDamageToOpponentCreature { amount } => {
            d.set_item("kind", "deal_damage_to_opponent_creature")?;
            d.set_item("amount", *amount)?;
        }
        Effect::LookAtTopNRevealOrCounter {
            n,
            filter_name,
            max_power,
            counter_on_miss,
        } => {
            d.set_item("kind", "look_at_top_n_reveal_or_counter")?;
            d.set_item("n", *n)?;
            d.set_item("filter_name", filter_name.as_str())?;
            d.set_item("max_power", *max_power)?;
            d.set_item("counter_on_miss", *counter_on_miss)?;
        }
        Effect::BecomeLandCreatureUEOT {
            power,
            toughness,
            granted_subtype,
        } => {
            d.set_item("kind", "become_land_creature_ueot")?;
            d.set_item("power", *power)?;
            d.set_item("toughness", *toughness)?;
            // Emit `granted_subtype` as the string when set, None
            // otherwise — round-trip preserves the optionality.
            match granted_subtype {
                Some(s) => d.set_item("granted_subtype", s.as_str())?,
                None => d.set_item("granted_subtype", py.None())?,
            }
        }
        Effect::Sequence { effects } => {
            d.set_item("kind", "sequence")?;
            // Recursively serialize each child via effect_to_pydict
            // (same function recurses safely — Effect is heap-
            // allocated via Vec<Effect>).
            let children: Vec<Bound<'_, PyDict>> = effects
                .iter()
                .map(|child| effect_to_pydict(py, child))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("effects", children)?;
        }
        Effect::OptionalManaCostThenEffect { cost, inner_effect } => {
            d.set_item("kind", "optional_mana_cost_then_effect")?;
            d.set_item("cost", mana_cost_to_pydict(py, cost)?)?;
            d.set_item("inner_effect", effect_to_pydict(py, inner_effect)?)?;
        }
        Effect::DrawCardsEqualToResolvingCombatDamage => {
            d.set_item("kind", "draw_cards_equal_to_resolving_combat_damage")?;
        }
        Effect::WarpExileSelf => {
            // G7 Class-C delayed-trigger port: unit variant; the only
            // bridge content is the `kind` discriminator. The Python
            // bridge arm (`cross_engine.py::_ta_dict_to_python_effect`)
            // maps this back to `WarpExileSelf()`. The effect is
            // scheduled in `state.delayed_triggers` at warp-cast time
            // and pushed onto the stack at the start of the END step
            // by `state_ops_steps.rs::run_delayed_end_step_triggers`.
            d.set_item("kind", "warp_exile_self")?;
        }
        // BBB downstream complement — 7th Rust slice.
        Effect::DefendingPlayerLosesHalfLifeRoundedUp => {
            d.set_item("kind", "defending_player_loses_half_life_rounded_up")?;
        }
        Effect::ConditionalIfYouGainedLifeAtLeast { threshold, inner } => {
            d.set_item("kind", "conditional_if_you_gained_life_at_least")?;
            d.set_item("threshold", *threshold)?;
            d.set_item("inner", effect_to_pydict(py, inner)?)?;
        }
        Effect::ConditionalIfSourcePowerAtLeast { threshold, effect } => {
            d.set_item("kind", "conditional_if_source_power_at_least")?;
            d.set_item("threshold", *threshold)?;
            // Emit as "effect" key to match the Python class field name
            // (ConditionalIfSourcePowerAtLeast.effect, not .inner).
            // The bridge accepts both "effect" and "inner" for forward-compat.
            d.set_item("effect", effect_to_pydict(py, effect)?)?;
        }
        // EEE downstream complement — 8th Rust slice (JJJ Pin 1).
        Effect::DyadrineAttackEffect => {
            d.set_item("kind", "dyadrine_attack_effect")?;
        }
        // EEE downstream complement — 8th Rust slice (JJJ Pin 2).
        Effect::PutPoisonCountersOnTargetPlayer { amount } => {
            d.set_item("kind", "put_poison_counters_on_target_player")?;
            d.set_item("amount", *amount)?;
        }
        // **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for
        // all-untargeted modal cards, 2026-05-21).** Round-trip dict-
        // shape for `ExileAllArtifactsWithManaValueAtMost { max_mv }`.
        // Two-field schema (`kind` + `max_mv`) mirrors the
        // `damage_each_opponent` single-int idiom.
        Effect::ExileAllArtifactsWithManaValueAtMost { max_mv } => {
            d.set_item("kind", "exile_all_artifacts_with_mana_value_at_most")?;
            d.set_item("max_mv", *max_mv)?;
        }
        // **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
        // 2026-05-21).** Round-trip dict-shape for
        // `PutChargeCountersOnTargetSpacecraftOrPlanetYouControl
        // { amount }`. Two-field schema (`kind` + `amount`) mirrors
        // the `damage_each_opponent` / `exile_all_artifacts_with_mana
        // _value_at_most` single-int idiom.
        Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl { amount } => {
            d.set_item(
                "kind",
                "put_charge_counters_on_target_spacecraft_or_planet_you_control",
            )?;
            d.set_item("amount", *amount)?;
        }
        // **Multi-target admit-gate Phase 4 widen (2026-05-21).** Round-
        // trip dict-shape for `ReturnUpToNSmallCreaturesFromGraveyardWith
        // Counters { max_n, max_total_mv }`. Three-field schema (`kind`
        // + `max_n` + `max_total_mv`) — both ints required, mirroring
        // the parse arm's required-fields contract and the Python
        // serializer leaf at `parity/trajectory.py::_python_spell_effect
        // _to_dict::_leaf`.
        Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters {
            max_n,
            max_total_mv,
        } => {
            d.set_item(
                "kind",
                "return_up_to_n_small_creatures_from_graveyard_with_counters",
            )?;
            d.set_item("max_n", *max_n)?;
            d.set_item("max_total_mv", *max_total_mv)?;
        }
    }
    Ok(d)
}

// --- Registry pyclass --------------------------------------------

#[pyclass(name = "Registry")]
#[derive(Debug, Default)]
pub struct Registry {
    /// Builder-side Vec. `register_printing` appends here. A
    /// `snapshot()` call freezes a clone of this Vec into an
    /// `Arc<RegistryData>` that GameStates share. Keeping the
    /// builder separate from the snapshot avoids interior
    /// mutability (Mutex/RwLock) on the snapshot path —
    /// first-slice workflow is "build, snapshot once, never
    /// mutate again", which is what synthetic states do today.
    printings: Vec<CardPrinting>,
}

#[pymethods]
impl Registry {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Register a card printing from a Python dict; returns its
    /// stable u32 id. Unknown `kind` values, missing required keys,
    /// and type mismatches raise rather than silently no-op.
    fn register_printing(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<u32> {
        let printing = parse_printing(dict)?;
        let id = self.printings.len() as u32;
        self.printings.push(printing);
        Ok(id)
    }

    fn __len__(&self) -> usize {
        self.printings.len()
    }

    fn get_printing_name(&self, id: u32) -> PyResult<String> {
        let p = self.lookup(id)?;
        Ok(p.name.to_string())
    }

    fn get_printing_mana_cost<'py>(
        &self,
        py: Python<'py>,
        id: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = self.lookup(id)?;
        mana_cost_to_pydict(py, &p.mana_cost)
    }

    fn get_printing_effects<'py>(
        &self,
        py: Python<'py>,
        id: u32,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let p = self.lookup(id)?;
        p.effects
            .iter()
            .map(|e| effect_to_pydict(py, e))
            .collect()
    }

    /// Printed power for a creature printing; `None` for non-creatures.
    fn get_printing_power(&self, id: u32) -> PyResult<Option<i16>> {
        Ok(self.lookup(id)?.power)
    }

    /// Printed toughness for a creature printing; `None` for non-creatures.
    fn get_printing_toughness(&self, id: u32) -> PyResult<Option<i16>> {
        Ok(self.lookup(id)?.toughness)
    }

    /// Mana symbols this printing produces when tapped, or `None`
    /// when the printing has no mana-production behavior. A
    /// single-color land like Forest returns `Some(["G"])`.
    fn get_printing_mana_produced(&self, id: u32) -> PyResult<Option<Vec<String>>> {
        Ok(self.lookup(id)?.mana_produced.clone())
    }

    /// MTG card types: "Creature", "Land", "Instant", etc.
    fn get_printing_card_types(&self, id: u32) -> PyResult<Vec<String>> {
        Ok(self.lookup(id)?.card_types.clone())
    }

    /// MTG card subtypes: "Forest", "Bear", "Equipment", "Aura", etc.
    /// Empty when the type line has no subtypes (vanilla
    /// instants/sorceries). Mirrors Python's
    /// `CardPrinting.subtypes: tuple[str, ...]`.
    fn get_printing_subtypes(&self, id: u32) -> PyResult<Vec<String>> {
        Ok(self.lookup(id)?.subtypes.clone())
    }

    /// MTG card supertypes: "Basic", "Legendary", "Snow", "World",
    /// "Token". Empty when the type line has no supertype prefix.
    /// Mirrors Python's `CardPrinting.supertypes: tuple[str, ...]`.
    /// Round-30 substrate addition.
    fn get_printing_supertypes(&self, id: u32) -> PyResult<Vec<String>> {
        Ok(self.lookup(id)?.supertypes.clone())
    }

    /// Static printed keywords (lowercase): "flying", "vigilance",
    /// "trample", etc.
    fn get_printing_keywords(&self, id: u32) -> PyResult<Vec<String>> {
        Ok(self.lookup(id)?.keywords.clone())
    }

    /// "When ~ enters the battlefield" effects (round-9 bridge ext).
    fn get_printing_etb_effects<'py>(
        &self,
        py: Python<'py>,
        id: u32,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let p = self.lookup(id)?;
        p.etb_effects.iter().map(|e| effect_to_pydict(py, e)).collect()
    }

    /// **Aura-attachment substrate — Phase 1 STEP 2.** `true` iff this
    /// printing parsed as an Aura (the registry-dict carried
    /// `is_aura=True`). Mirrors Python's `CardPrinting.is_aura`
    /// derivation. Used by the test surface to confirm the STEP-2
    /// parse arm landed.
    fn get_printing_is_aura(&self, id: u32) -> PyResult<bool> {
        Ok(self.lookup(id)?.is_aura)
    }

    /// **Aura-attachment substrate — Phase 1 STEP 2.** Which permanent
    /// type this Aura enchants (`"creature"` | `"artifact"`). Defaults
    /// to `"creature"` for non-Aura printings — gate on `is_aura`
    /// before reading.
    /// Read the dynamic-P/T formula key for a printing, or `None`.
    /// Mirrors Python's ``CardPrinting.dynamic_pt_formula``. Used by
    /// cross-engine probes to inspect the bridge-round-tripped value
    /// and by the dynamic-P/T-formula substrate test surface to pin
    /// the registry shape across builds.
    fn get_printing_dynamic_pt_formula(
        &self,
        id: u32,
    ) -> PyResult<Option<String>> {
        Ok(self.lookup(id)?.dynamic_pt_formula.clone())
    }

    fn get_printing_aura_enchant_type(&self, id: u32) -> PyResult<String> {
        Ok(self.lookup(id)?.aura_enchant_type.clone())
    }

    /// **Aura-attachment substrate — Phase 1 STEP 2.** `true` iff this
    /// Aura's printed text is "Enchant <type> YOU CONTROL" (Shardmage's
    /// Rescue). `false` for non-Auras and for non-yours-only Auras.
    fn get_printing_aura_yours_only(&self, id: u32) -> PyResult<bool> {
        Ok(self.lookup(id)?.aura_yours_only)
    }

    /// **Aura-attachment substrate — Phase 1 STEP 2.** Layer-7 bonus
    /// triple `(power_bonus, toughness_bonus, granted_keywords)`.
    /// pyo3 auto-converts the Rust tuple to a Python tuple. `(0, 0,
    /// [])` for non-Auras and for Auras whose bridge omitted the
    /// `aura_equipment_bonuses` sub-dict.
    fn get_printing_aura_grant(
        &self,
        id: u32,
    ) -> PyResult<(i16, i16, Vec<String>)> {
        let p = self.lookup(id)?;
        Ok((
            p.aura_power_bonus,
            p.aura_toughness_bonus,
            p.aura_granted_keywords.clone(),
        ))
    }

    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** `true` iff this Aura grants control of the
    /// enchanted permanent. Defaults to `false` for non-Auras.
    fn get_printing_aura_grants_control(&self, id: u32) -> PyResult<bool> {
        Ok(self.lookup(id)?.aura_grants_control)
    }

    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** `true` iff this Aura wipes the enchanted
    /// permanent's abilities (layer 6).
    fn get_printing_aura_grants_lose_abilities(
        &self,
        id: u32,
    ) -> PyResult<bool> {
        Ok(self.lookup(id)?.aura_grants_lose_abilities)
    }

    /// **Aura-attachment substrate — Phase 2 S2 (round-79+,
    /// 2026-05-20).** Returns `Some((p, t))` if this Aura overrides
    /// the host's base P/T; `None` otherwise.
    fn get_printing_aura_set_base_pt(
        &self,
        id: u32,
    ) -> PyResult<Option<(i16, i16)>> {
        Ok(self.lookup(id)?.aura_set_base_pt)
    }

    /// **Aura-attachment substrate — Phase 2 S2 extension (Tractor
    /// Beam closure, 2026-05-20).** `true` iff this Aura prevents
    /// its host from untapping during the host's controller's untap
    /// step. Defaults to `false` for non-Auras and for Auras without
    /// the don't-untap rider. Consumed by the cross-engine bridge
    /// read-back (`read_rust_printing_aura_fields`) and by the
    /// untap-step sweep in `state_ops_steps.rs::untap_all_for_player_inline`.
    fn get_printing_aura_prevents_host_untap(
        &self,
        id: u32,
    ) -> PyResult<bool> {
        Ok(self.lookup(id)?.aura_prevents_host_untap)
    }

    /// **Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    /// substrate landing, 2026-05-20).** `true` iff this Aura's
    /// printed text carries the Moonlit Meditation CR 614
    /// "first time you would create one or more tokens each turn"
    /// token-creation replacement rider. Defaults to `false` for
    /// non-Auras and for every pre-S3 registry dict (the bridge omits
    /// the `aura_replaces_first_tokens_with_copies` key when the
    /// CardDef flag is `false`). Consumed by the cross-engine bridge
    /// read-back (`read_rust_printing_aura_fields`); the
    /// `CreateToken` arm does NOT yet wire the replacement semantic
    /// (loader admit stays declined — see the field docstring above).
    fn get_printing_aura_replaces_first_tokens_with_copies(
        &self,
        id: u32,
    ) -> PyResult<bool> {
        Ok(self.lookup(id)?.aura_replaces_first_tokens_with_copies)
    }

    /// **Static / layered abilities substrate — Phase 1 first-
    /// deliverable (round-79).** Surface the printing's
    /// `static_anthems` as a Python-friendly
    /// `list[tuple[str, list[str]]]` shape for round-trip pin tests.
    /// Each element is `(scope, granted_keywords)`. Empty list for
    /// every non-anthem printing (back-compat default).
    fn get_printing_static_anthems(
        &self,
        id: u32,
    ) -> PyResult<Vec<(String, Vec<String>)>> {
        let p = self.lookup(id)?;
        Ok(p.static_anthems
            .iter()
            .map(|a| (a.scope.clone(), a.granted_keywords.clone()))
            .collect())
    }

    /// **Static / layered abilities substrate — Phase 2 (round-80,
    /// 2026-05-20).** `true` iff this printing's spell is
    /// uncounterable (Frenzied Baloth's "This spell can't be
    /// countered."). Defaults to `false` for every pre-Phase-2
    /// printing.
    fn get_printing_spell_is_uncounterable(
        &self,
        id: u32,
    ) -> PyResult<bool> {
        Ok(self.lookup(id)?.spell_is_uncounterable)
    }

    /// **Static / layered abilities substrate — Phase 2 (round-80,
    /// 2026-05-20).** `true` iff this printing grants creature
    /// spells you control uncounterable status (Frenzied Baloth's
    /// "Creature spells you control can't be countered."). Defaults
    /// to `false` for every pre-Phase-2 printing.
    fn get_printing_grants_creature_spells_uncounterable(
        &self,
        id: u32,
    ) -> PyResult<bool> {
        Ok(self.lookup(id)?.grants_creature_spells_uncounterable)
    }

    /// **Static / layered abilities substrate — Phase 3 (round-81,
    /// 2026-05-20).** Returns this printing's `static_self_bonuses`
    /// as a Python-friendly list of tuples
    /// `[(condition_tag, condition_args, power_bonus,
    /// toughness_bonus, granted_keywords, removed_keywords),
    /// ...]`. Empty when the printing has no entries (every pre-
    /// Phase-3 printing).
    ///
    /// **Round-82 (2026-05-20, Mechan Shieldmate closure).** The
    /// tuple grew a 6th element — `removed_keywords` — for the
    /// RemoveKeyword payload fan-out (Mechan Shieldmate's defender
    /// removal). Every pre-round-82 printing serializes its
    /// `removed_keywords` slot as an empty Vec (back-compat default
    /// — the `removed_keywords` registry-dict key is optional).
    fn get_printing_static_self_bonuses(
        &self,
        id: u32,
    ) -> PyResult<Vec<(String, Vec<i32>, i16, i16, Vec<String>, Vec<String>)>> {
        let p = self.lookup(id)?;
        Ok(p.static_self_bonuses
            .iter()
            .map(|b| {
                (
                    b.condition_tag.clone(),
                    b.condition_args.clone(),
                    b.power_bonus,
                    b.toughness_bonus,
                    b.granted_keywords.clone(),
                    b.removed_keywords.clone(),
                )
            })
            .collect())
    }

    /// Microbench helper: round-trip ``n`` copies of the same dict
    /// through the parser. Returns mean per-call microseconds.
    fn register_n_times_micros(
        &mut self,
        dict: &Bound<'_, PyDict>,
        n: usize,
    ) -> PyResult<f64> {
        let start = Instant::now();
        for _ in 0..n {
            let printing = parse_printing(dict)?;
            std::hint::black_box(printing);
        }
        let elapsed = start.elapsed();
        Ok((elapsed.as_secs_f64() * 1_000_000.0) / n as f64)
    }
}

impl Registry {
    fn lookup(&self, id: u32) -> PyResult<&CardPrinting> {
        self.printings.get(id as usize).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "no printing with id {id} (registry has {})",
                self.printings.len()
            ))
        })
    }

    /// Freeze the current builder Vec into an `Arc<RegistryData>`
    /// suitable for sharing into a `GameState`. Used by
    /// `make_synthetic_state` and (later) by Python code that
    /// builds a game state from a finalized registry.
    pub fn snapshot(&self) -> Arc<RegistryData> {
        Arc::new(RegistryData {
            printings: self.printings.clone(),
        })
    }

    /// Push a `CardPrinting` directly (used by `make_synthetic_state`
    /// to seed an in-process registry without the dict round-trip).
    /// Returns the new printing's id.
    pub fn push_printing(&mut self, printing: CardPrinting) -> u32 {
        let id = self.printings.len() as u32;
        self.printings.push(printing);
        id
    }
}
