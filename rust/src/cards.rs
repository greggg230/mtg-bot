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
    /// **First-slice filter vocabulary.** Rust supports a subset of
    /// Python's filters constrained by available registry fields
    /// (Rust has no `supertypes` / `subtypes` registry today):
    ///   - `"basic_land"` → `card_types` contains `"Land"`
    ///     (approximation; Rust can't distinguish basic vs non-basic
    ///     without supertypes).
    ///   - `"basic_plains"` / `"basic_island"` / `"basic_swamp"` /
    ///     `"basic_mountain"` / `"basic_forest"` → `printing.name`
    ///     equals the canonical basic name.
    ///   - `"creature_small"` (with `max_mv`) → `card_types` contains
    ///     `"Creature"` AND `mana_cost.total() <= max_mv`. `max_mv=0`
    ///     means "creatures with cost <= 0" (rarely matches; specify
    ///     a real `max_mv` value).
    ///   - `"artifact"` → `card_types` contains `"Artifact"`.
    ///   - unknown filter → silent fizzle (matches Python's
    ///     `_matches` returning False for unknown).
    ///
    /// **Deferred filters**: `legendary_creature`, `sliver`, `demon`,
    /// `spacecraft` — these require supertypes / subtypes which
    /// aren't on the Rust registry yet. Silent fizzle for now.
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
    /// routing through `state.resolving_modes` and `or_both=True`
    /// (the "choose one or both" Charm variant) are deferred. The
    /// mode list is bounded by parse rejection — `modes.is_empty()`
    /// raises at register time so the resolve arm always has at least
    /// one mode to fall through to.
    ChooseOneMode { modes: Vec<Effect> },
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
    /// **Round 28 UEOT coverage extension (broader pump filter).**
    /// Target artifact OR creature gets `+power/+toughness` until end
    /// of turn. Mirrors Python's
    /// `PumpTargetArtifactOrCreatureUntilEOT` (3+ EoE cards target
    /// family — Skitterclaw-Augur-shape "boost your robot or your
    /// dude" instants).
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
    /// `(0, 0)` short-circuits as a no-op. Negative bonuses are
    /// legal (debuff-shaped).
    PumpTargetArtifactOrCreatureUntilEOT { power_bonus: i32, toughness_bonus: i32 },
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
    /// "Whenever X deals combat damage to Y" — multi-subject event
    /// firing from `state_ops_combat.rs::apply_damage_pass` per
    /// damage entry. Slots: `Source` = the attacking / blocking
    /// creature dealing damage; `Target` = the player or creature
    /// taking damage. Mirrors Python's
    /// `Event.DEALS_COMBAT_DAMAGE` at
    /// `src/mtg_bot/effects/predicates.py:63` with the
    /// `("source", "target")` slot pair from `EVENT_SUBJECTS`.
    ///
    /// **Subject-substrate dependency**: a `Predicate::Subject(slot,
    /// inner)` predicate can gate on either slot. The default
    /// `subject_id` passed to `dispatch_event` is the Source (the
    /// dealer), so `SourceIsSelf` ("whenever ~ deals combat damage")
    /// matches naturally when the dealing creature is the source of
    /// the ability. Slot-anchored predicates target the recipient
    /// via `Subject(Target, ...)`.
    ///
    /// **Dispatch granularity**: fires once per damage entry, so a
    /// trample attacker that hits a blocker AND the defending player
    /// fires twice (once with Target = blocker, once with Target =
    /// player) — mirrors Python's per-event invocation shape at
    /// `src/mtg_bot/combat.py:729-742`.
    DealsCombatDamage,
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
}

/// Named subject slot for multi-subject events. Mirrors the named
/// keys in Python's `EVENT_SUBJECTS` mapping at
/// `src/mtg_bot/effects/predicates.py:205`.
///
/// **Scope (ETB Subject-substrate slice).** The minimum surface
/// needed to unblock multi-subject events. Today's variants:
///
/// - `Source` — the object DEALING / ORIGINATING the event. For
///   `Event::DealsCombatDamage` this is the attacking/blocking
///   creature dealing damage; for `Event::Blocks` (when wired
///   multi-subject) it's the blocker. Mirrors Python's
///   ``"source"`` string slot.
/// - `Target` — the object/player RECEIVING / TARGETED by the event.
///   For `Event::DealsCombatDamage` this is the player or creature
///   being damaged. Mirrors Python's ``"target"`` string slot.
///
/// **Future extension.** Adding a new named slot (e.g. ``Blocker``,
/// ``Attacker`` for multi-subject `Event::Blocks`) requires both an
/// enum variant here AND a matching parser arm in
/// `parse_subject_slot`. Cross-engine bridge translates each variant
/// to its Python string slot name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubjectSlot {
    /// The dealing / originating object. Maps to Python's ``"source"``.
    Source,
    /// The receiving / targeted object. Maps to Python's ``"target"``.
    Target,
}

impl SubjectSlot {
    /// The canonical lowercase string name for this slot. Matches the
    /// dict-key the parser consumes and the Python slot-name strings
    /// in `EVENT_SUBJECTS`.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            SubjectSlot::Source => "source",
            SubjectSlot::Target => "target",
        }
    }
}

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
    /// Matches when the **subject** is controlled by the source's
    /// controller. Mirrors Python's `Controlled(YOU)` — the
    /// canonical "under your control" qualifier on ETB triggers
    /// like "whenever a creature enters under your control" and
    /// "whenever a land enters under your control".
    ControlledByYou,
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
    /// Pulls a named subject slot out of a multi-subject event and
    /// evaluates `inner` against THAT subject (rather than the
    /// default `subject_id` the dispatch site passed in). Mirrors
    /// Python's `Subject(slot, inner)` at
    /// `src/mtg_bot/effects/predicates.py:354` for the named-slot
    /// case (Rust's existing implicit-subject predicates already
    /// cover the `Subject(None, ...)` default-slot case).
    ///
    /// **Single-subject events.** ENTERS / DIES / CAST / ATTACKS /
    /// BLOCKS / LEAVES dispatch today with an empty slot map; a
    /// `Predicate::Subject(slot, inner)` on one of those events
    /// evaluates `false` because the slot isn't bound. The existing
    /// implicit-subject predicates (`SubjectIsCreature`,
    /// `ControlledByYou`, etc.) continue to work unchanged on
    /// single-subject events — `Predicate::Subject` only fires on
    /// the future multi-subject dispatch path
    /// (`Event::DealsCombatDamage` and friends) where the dispatch
    /// site supplies the slot map.
    ///
    /// **Use case.** A DealsCombatDamage "whenever this creature
    /// deals combat damage to a player" gates on the `Target`
    /// slot's recipient via `Subject(Target, ...)`. Similarly a
    /// Skystinger-shape BLOCKS trigger `"whenever ~ blocks a
    /// creature with flying"` would predicate on the attacker
    /// subject via `Subject(Attacker, ...)` once the multi-subject
    /// BLOCKS dispatch lands.
    Subject(SubjectSlot, Box<Predicate>),
}

impl Predicate {
    /// Evaluate the predicate against the firing context.
    ///
    /// `source_id` is the GameObject carrying the ability;
    /// `subject_id` is the GameObject the event happened to (the
    /// creature that just entered the battlefield, for Enters).
    /// Both ids are u32-keyed object ids resolved at dispatch time.
    ///
    /// This is the single-subject entry point — it delegates to
    /// `matches_with_slots` with an empty slot map. A predicate that
    /// uses `Predicate::Subject(slot, inner)` on a single-subject
    /// event evaluates `false` (the named slot isn't bound). All
    /// pre-Subject-substrate predicates ignore the slot map and
    /// continue to evaluate against `subject_id` directly, so this
    /// call is bit-identical to the pre-substrate behavior for
    /// every existing call site.
    pub(crate) fn matches(
        &self,
        state: &crate::state::GameState,
        source_id: u32,
        subject_id: u32,
    ) -> bool {
        self.matches_with_slots(state, source_id, subject_id, &[])
    }

    /// Multi-subject-aware evaluation entry point. `subjects_by_slot`
    /// supplies the named-slot map for multi-subject events
    /// (`Event::DealsCombatDamage` and friends — see `SubjectSlot`).
    /// On single-subject events, callers pass `&[]` (via the thin
    /// `matches` wrapper).
    ///
    /// `subject_id` remains the implicit-default subject (the
    /// pre-substrate single-subject id). Every existing predicate
    /// arm reads it unchanged; only `Predicate::Subject(slot, inner)`
    /// overrides it (replacing `subject_id` for the recursive
    /// `inner` evaluation with the id bound to `slot` in
    /// `subjects_by_slot`). When the slot isn't present, the
    /// predicate evaluates `false` — matching Python's
    /// `_resolve_subject` "missing slot → predicate is false"
    /// short-circuit at
    /// `src/mtg_bot/effects/predicates.py:340`.
    pub(crate) fn matches_with_slots(
        &self,
        state: &crate::state::GameState,
        source_id: u32,
        subject_id: u32,
        subjects_by_slot: &[(SubjectSlot, u32)],
    ) -> bool {
        match self {
            Predicate::Always => true,
            Predicate::SourceIsSelf => source_id == subject_id,
            Predicate::SourceIsCreature => {
                source_has_card_type(state, source_id, "Creature")
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
            Predicate::ControlledByYou => {
                // Subject's controller == source's controller. Mirrors
                // Python's `Controlled(by=YOU)` on a permanent subject:
                // ``obj.controller_id == source.controller_id``.
                // Each id is resolved by walking the four tracked zones
                // on both players; the helper returns `None` when the
                // id isn't present (the same loud-but-safe shape as
                // `find_source_printing_id`).
                let source_controller =
                    find_object_controller(state, source_id);
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
            Predicate::SubjectHasSubtype(subtype) => {
                subject_has_subtype(state, subject_id, subtype)
            }
            Predicate::Not(inner) => {
                !inner.matches_with_slots(state, source_id, subject_id, subjects_by_slot)
            }
            Predicate::And(a, b) => {
                a.matches_with_slots(state, source_id, subject_id, subjects_by_slot)
                    && b.matches_with_slots(state, source_id, subject_id, subjects_by_slot)
            }
            Predicate::Or(a, b) => {
                a.matches_with_slots(state, source_id, subject_id, subjects_by_slot)
                    || b.matches_with_slots(state, source_id, subject_id, subjects_by_slot)
            }
            Predicate::Subject(slot, inner) => {
                // Look up the named slot; if absent, the predicate
                // is false (mirrors Python's `_resolve_subject`
                // short-circuit). When present, evaluate `inner`
                // with `subject_id = slot_id` — this swaps the
                // implicit subject for the duration of the
                // recursive evaluation. The slot map stays intact so
                // a nested `Subject(other_slot, ...)` (theoretical
                // future shape) still resolves correctly.
                let slot_id = subjects_by_slot
                    .iter()
                    .find_map(|(s, id)| if *s == *slot { Some(*id) } else { None });
                match slot_id {
                    Some(id) => inner.matches_with_slots(
                        state,
                        source_id,
                        id,
                        subjects_by_slot,
                    ),
                    None => false,
                }
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
    false
}

/// Look up an object by id across all four tracked zones of both
/// players and return its controller id. Returns `None` when the
/// object isn't found in any of those zones (already moved off-bf
/// in a way the dispatch caller didn't expect, or an id mismatch).
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
        "deal_damage_any_target" => {
            // Python's `DealDamageAnyTarget(DealDamage)` carries only
            // the inherited `amount` field; the "any target" filter is
            // intrinsic to the variant (no `target` dict to parse —
            // distinct from `deal_damage`'s required `target` field).
            // Missing `amount` is a KeyError at register time;
            // negative amount surfaces at register time (defense-in-
            // depth; `apply_effect` re-checks).
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
        "return_creature_from_your_graveyard_to_hand" => {
            // Unit variant; no fields beyond `kind`.
            Ok(Effect::ReturnCreatureFromYourGraveyardToHand)
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
        "exile_target_creature_until_source_leaves" => {
            Ok(Effect::ExileTargetCreatureUntilSourceLeaves)
        }
        "exile_opponents_nonland_permanent_until_leaves" => {
            Ok(Effect::ExileOpponentsNonlandPermanentUntilLeaves)
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
        "pump_target_artifact_or_creature_until_eot" => {
            // Round-28 broader-filter pump. Same schema as
            // pump_target_creature_until_eot — power_bonus +
            // toughness_bonus default to 0; negatives allowed.
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
            Ok(Effect::PumpTargetArtifactOrCreatureUntilEOT { power_bonus, toughness_bonus })
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
            Ok(Effect::ChooseOneMode { modes: parsed_modes })
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
        other => Err(PyValueError::new_err(format!(
            "unknown effect kind {other:?}; valid: deal_damage, draw_cards, gain_life, return_to_hand, mill, mill_target_player, exile_target, tap_target, untap_target, sacrifice, add_counters, discard_card, scry_n, destroy_target, damage_each_opponent, damage_each_creature, create_token, add_mana_to_pool, search_library_for_basic_land_tapped, put_plus_one_counter_on_self, surveil, counter_target_spell, return_creature_from_your_graveyard_to_battlefield, discard_self, each_opponent_loses_life, lose_life, target_opponent_loses_life, each_opponent_discards, destroy_all_creatures, destroy_target_nonland_permanent, destroy_target_artifact, destroy_target_land, deal_damage_any_target, pump_self_until_eot, pump_target_creature_until_eot, sequence, pump_creatures_you_control_until_eot, return_creature_from_your_graveyard_to_hand, put_plus_one_counter_on_each_creature_you_control, put_plus_one_counter_on_creature_you_control, add_n_mana_of_chosen_color, search_library_for_card, pump_creatures_opponents_control_until_eot, put_card_from_graveyard_on_bottom_of_library, move_self_counters_to_target_creature_you_control, deal_damage_equal_to_source_power_any_target, tap_target_artifact_or_creature, conditional_on_tapped_creatures, put_plus_one_counter_on_each_of_up_to_n_target_creatures, put_charge_counters_on_self, create_token_copy_of_target_creature_you_control, optional_sac_artifact_for_effect, owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library, attach_to_target_creature_you_control, destroy_enchanted_permanent, tap_enchanted_permanent, exile_target_creature_until_source_leaves, exile_opponents_nonland_permanent_until_leaves, may_do, look_top_n_put_one_in_hand_rest_to_graveyard, choose_one_mode, if_you_do, each_opponent_mills_cards, pump_target_artifact_or_creature_until_eot, each_opponent_sacrifices_creature, grant_keyword_to_target_creature_until_eot, untap_target_other_permanent, tap_another_target_creature, return_all_creatures_from_your_graveyard_to_battlefield, set_base_pt_until_eot, manifest_dread, manifest_dread_then_attach_self, manifest_dread_then_put_plus_one_counters_on_it, conditional_if_void, deal_damage_to_opponent_creature, optional_mana_cost_then_effect, draw_cards_equal_to_resolving_combat_damage"
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

/// CE20 follow-up — bridge legacy `mana_produced` into a synthesized
/// `ActivatedAbility` entry on the printing's `activated_abilities`
/// Vec. When `mana_produced.is_some()`, prepend one synthesized
/// entry equivalent to the pre-bridge free-mana-ability:
///
///   cost:   ManaCost::default() (free) + tap_self=true
///   effect: AddManaToPool (single-color) or Sequence of AddManaToPool
///           per color (multi-color hybrid). Aggregated by color so
///           `mana_produced=["G","G"]` yields one AddManaToPool with
///           amount=2.
///   is_mana_ability: true (CR 605.1 — produces mana, doesn't target).
///
/// The synthesized entry is PREPENDED — `ability_index=0` continues
/// to refer to "the free mana ability". Explicit registry-supplied
/// `activated_abilities` entries get bumped to indices >= 1. Every
/// pre-bridge caller of `ActivateAbility(forest_id, 0)` keeps
/// working bit-equal.
///
/// Shared between `parse_printing` (dict-input registry path) and
/// `make_synthetic_state` (the test-fixture path in `state.rs`) so
/// both surfaces get the same bridge synthesis without duplicating
/// the field-counting logic.
pub fn apply_mana_produced_bridge(
    mana_produced: &Option<Vec<String>>,
    existing: Vec<ActivatedAbility>,
) -> PyResult<Vec<ActivatedAbility>> {
    let Some(symbols) = mana_produced else {
        return Ok(existing);
    };
    // Build per-color amount totals in W/U/B/R/G/C order.
    let mut counts: [(&str, i32); 6] = [
        ("W", 0), ("U", 0), ("B", 0), ("R", 0), ("G", 0), ("C", 0),
    ];
    for sym in symbols {
        let slot = match sym.as_str() {
            "W" => &mut counts[0].1,
            "U" => &mut counts[1].1,
            "B" => &mut counts[2].1,
            "R" => &mut counts[3].1,
            "G" => &mut counts[4].1,
            "C" => &mut counts[5].1,
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
        };
        *slot += 1;
    }
    let mut bridge_effects: Vec<Effect> = Vec::new();
    for (color, amount) in counts.iter() {
        if *amount > 0 {
            bridge_effects.push(Effect::AddManaToPool {
                color: (*color).to_string(),
                amount: *amount,
            });
        }
    }
    let bridge_effect = if bridge_effects.len() == 1 {
        // Single-color shape (Forest, Mountain, ...) — flat
        // AddManaToPool matches the existing apply path 1:1.
        bridge_effects.into_iter().next().unwrap()
    } else {
        // Multi-color or multi-pip shape (hybrid producers).
        // Wrap in Sequence so apply_effect fires each entry.
        Effect::Sequence { effects: bridge_effects }
    };
    let bridge_ability = ActivatedAbility {
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
        effect: bridge_effect,
        is_mana_ability: true,
        tier_threshold: 0,
    };
    // Prepend so ability_index=0 still refers to the bridged mana
    // ability — every pre-bridge ActivateAbility(forest, 0) caller
    // keeps working.
    let mut bridged = Vec::with_capacity(existing.len() + 1);
    bridged.push(bridge_ability);
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
    Ok(CardPrinting {
        name: Arc::from(name),
        mana_cost,
        effects,
        power,
        toughness,
        mana_produced,
        card_types,
        subtypes,
        keywords,
        etb_effects,
        triggered_abilities,
        activated_abilities,
        starting_loyalty,
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
/// `"leaves_battlefield"`; Slice 9 adds `"blocks"`. Remaining
/// variants (`deals_damage`) still raise because their dispatch
/// sites don't exist yet — allowing them through would let
/// registry-dict callers stamp triggers that never fire, masking
/// the missing-fanout bug.
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
        // ETB-Subject substrate consumer slice — multi-subject
        // dispatch from state_ops_combat.rs::apply_damage_pass.
        "deals_combat_damage" => Event::DealsCombatDamage,
        // Out-of-scope events stay raise-on-register. Each lands
        // alongside its dispatch fan-out so the parser surface stays
        // consistent with the dispatcher's coverage.
        "deals_damage" => {
            return Err(PyValueError::new_err(format!(
                "triggered_ability.event {event_str:?} is not yet wired in Slice 9; \
                 only \"enters\", \"dies\", \"cast\", \"attacks\", \"blocks\", \"leaves_battlefield\", \
                 and \"deals_combat_damage\" dispatch today"
            )));
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown triggered_ability.event {other:?}; \
                 valid: enters, dies, cast, attacks, blocks, leaves_battlefield, \
                 deals_combat_damage"
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
        "subject_is_land" => Ok(Predicate::SubjectIsLand),
        "subject_is_artifact" => Ok(Predicate::SubjectIsArtifact),
        "subject_is_creature" => Ok(Predicate::SubjectIsCreature),
        "controlled_by_you" => Ok(Predicate::ControlledByYou),
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
        // Subject-substrate slice — pull a named subject slot out of
        // a multi-subject event and evaluate `inner` against THAT
        // subject. Dict shape:
        //   {"kind": "subject", "slot": "source"|"target",
        //    "inner": {...}}
        // The slot string maps to a `SubjectSlot` enum variant via
        // `parse_subject_slot`. On single-subject events the slot map
        // is empty so this predicate evaluates `false` — only the
        // multi-subject dispatch path (Event::DealsCombatDamage etc.)
        // supplies the bindings.
        "subject" => {
            let slot_any = require(dict, "slot")?;
            let slot_str: String = slot_any.extract().map_err(|_| {
                PyValueError::new_err(
                    "predicate.subject.slot must be a string",
                )
            })?;
            let slot = parse_subject_slot(&slot_str)?;
            let inner_any = require(dict, "inner")?;
            let inner_dict = inner_any.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("predicate.subject.inner must be a dict")
            })?;
            Ok(Predicate::Subject(
                slot,
                Box::new(parse_predicate(inner_dict)?),
            ))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown predicate kind {other:?}; \
             valid: always, source_is_self, source_is_creature, \
             subject_is_land, subject_is_artifact, subject_is_creature, \
             controlled_by_you, subject_has_subtype, not, and, or, \
             subject"
        ))),
    }
}

/// Parse a `SubjectSlot` string from the predicate-dict `slot` key.
/// Mirrors the inverse of `SubjectSlot::as_str`. Unknown slot names
/// raise loudly so a typo (e.g. `"src"` instead of `"source"`) fails
/// at parse time rather than silently never-matching at dispatch.
fn parse_subject_slot(slot: &str) -> PyResult<SubjectSlot> {
    match slot {
        "source" => Ok(SubjectSlot::Source),
        "target" => Ok(SubjectSlot::Target),
        other => Err(PyValueError::new_err(format!(
            "unknown predicate.subject.slot {other:?}; \
             valid: source, target"
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
        Effect::DiscardSelf { amount } => {
            d.set_item("kind", "discard_self")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentLosesLife { amount } => {
            d.set_item("kind", "each_opponent_loses_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::LoseLife { amount } => {
            d.set_item("kind", "lose_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::TargetOpponentLosesLife { amount } => {
            d.set_item("kind", "target_opponent_loses_life")?;
            d.set_item("amount", *amount)?;
        }
        Effect::EachOpponentDiscards { count } => {
            d.set_item("kind", "each_opponent_discards")?;
            d.set_item("count", *count)?;
        }
        Effect::DestroyAllCreatures => {
            d.set_item("kind", "destroy_all_creatures")?;
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
        Effect::PumpCreaturesYouControlUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_creatures_you_control_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::ReturnCreatureFromYourGraveyardToHand => {
            d.set_item("kind", "return_creature_from_your_graveyard_to_hand")?;
        }
        Effect::PutPlusOneCounterOnEachCreatureYouControl { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_each_creature_you_control")?;
            d.set_item("amount", *amount)?;
        }
        Effect::PutPlusOneCounterOnCreatureYouControl { amount } => {
            d.set_item("kind", "put_plus_one_counter_on_creature_you_control")?;
            d.set_item("amount", *amount)?;
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
        Effect::ExileTargetCreatureUntilSourceLeaves => {
            d.set_item("kind", "exile_target_creature_until_source_leaves")?;
        }
        Effect::ExileOpponentsNonlandPermanentUntilLeaves => {
            d.set_item("kind", "exile_opponents_nonland_permanent_until_leaves")?;
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
        Effect::ChooseOneMode { modes } => {
            d.set_item("kind", "choose_one_mode")?;
            let mode_dicts: Vec<Bound<'_, PyDict>> = modes
                .iter()
                .map(|m| effect_to_pydict(py, m))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("modes", mode_dicts)?;
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
        Effect::PumpTargetArtifactOrCreatureUntilEOT { power_bonus, toughness_bonus } => {
            d.set_item("kind", "pump_target_artifact_or_creature_until_eot")?;
            d.set_item("power_bonus", *power_bonus)?;
            d.set_item("toughness_bonus", *toughness_bonus)?;
        }
        Effect::EachOpponentSacrificesCreature => {
            d.set_item("kind", "each_opponent_sacrifices_creature")?;
        }
        Effect::GrantKeywordToTargetCreatureUntilEOT { keyword } => {
            d.set_item("kind", "grant_keyword_to_target_creature_until_eot")?;
            d.set_item("keyword", keyword.as_str())?;
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
