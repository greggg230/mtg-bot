use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Instant;

use crate::cards::{CardPrinting, ManaCost, Registry, RegistryData};

/// Game object referencing card data by `printing_id`.
///
/// Pre-registry: the object inlined `name: String`, `power: i16`,
/// `toughness: i16` per copy — so each per-zone Vec carried a
/// heap-allocated String per card, and cloning a zone (when CoW
/// fired on first mutation) had to clone every String. With this
/// migration, all card-data (name, p/t once we wire it through,
/// mana cost, effects, ...) lives in the shared
/// `Arc<RegistryData>` on the `GameState`; the GameObject keeps
/// only the per-instance state (id, controller, zone, tapped,
/// summoning sick, damage, counters) plus a `printing_id: u32`
/// pointer into the registry.
///
/// **Round-27 (UEOT framework):** GameObject grows a
/// `ueot_modifiers: Vec<UeotModifier>` — the per-object expiring-
/// modifier stack that powers "until end of turn" effects (CR 514.3).
/// This costs the struct its `Copy` derive: a `Vec<T>` is heap-owned
/// and can't be `memcpy`'d. The empty-Vec common case still keeps
/// the per-object footprint small (a Vec is 24 bytes — pointer +
/// len + cap — so the pre-r27 ~20-byte object becomes ~48 bytes
/// when the Vec is empty, with the Vec's heap allocation deferred
/// until the first push). Code that previously relied on Copy
/// (e.g. `*drained.iter().find()`) now uses `.cloned()`; the
/// `Clone` impl is still cheap for the common empty-modifiers case
/// (a single null heap-ptr + zero len/cap is copied, no allocation).
#[derive(Clone, Debug)]
pub struct GameObject {
    pub id: u32,
    pub printing_id: u32,
    pub controller: u8,
    pub zone: u8,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: i16,
    pub counters: i32,
    /// **CE20 follow-up — Spacecraft station-charge.** Charge
    /// counters tracked per-object, distinct from the `counters`
    /// field above (which is the +1/+1 slot). Defaults to 0;
    /// Spacecraft printings accrue charge counters via the
    /// "station" mechanic (out of scope of this slice — station-as-
    /// cost is a future cost-pay extension) and gate their tiered
    /// activated abilities via `ActivatedAbility.tier_threshold`.
    ///
    /// **Why a distinct field** (vs overloading `counters`): G8
    /// (multi-kind counter open gap, `RUST_PARITY_GAPS.md`) flags
    /// the +1/+1 slot's single-kind ambiguity as a separate issue.
    /// Spacecraft + Walker loyalty + +1/+1 each need their own
    /// counter slot to track independently; overloading the
    /// existing field would have to also resolve G8 in lockstep.
    /// New field keeps the slices disjoint.
    ///
    /// Cleared (reset to `0`) on every zone change (CR 400.7
    /// per-object state reset) via the same field-reset shape
    /// `Effect::Sacrifice` and friends use.
    pub charge_counters: i32,
    /// **CE20 follow-up — Planeswalker loyalty (Slice 5).** Loyalty
    /// counters tracked per-object, distinct from `counters` (+1/+1)
    /// and `charge_counters` (Spacecraft station). Defaults to `0`;
    /// a planeswalker's printing seeds its loyalty via the
    /// `starting_loyalty` field on `CardPrinting`, stamped at ETB by
    /// every walker-entering call site. Activated loyalty abilities
    /// (CR 606) pay positive `cost.loyalty` (decrement) or add
    /// negative-shape (increment) counters; the SBA pass at
    /// `state_ops_sba.rs` moves a walker with `loyalty_counters <= 0`
    /// to its controller's graveyard per CR 704.5i.
    ///
    /// **Why a distinct field**: G8 (multi-kind counter open gap)
    /// applies — +1/+1 / charge / loyalty each need their own slot.
    /// Tezzeret, Cruel Captain (the sole EoE walker) carries this
    /// shape; non-walker objects keep `loyalty_counters = 0` and
    /// never gate on it.
    ///
    /// Cleared (reset to `0`) on every zone change (CR 400.7
    /// per-object state reset).
    pub loyalty_counters: i32,
    /// Round-28 attach foundation — when this object is an Aura /
    /// Equipment that is currently attached to another permanent,
    /// `attached_to` holds the target permanent's GameObject `id`.
    /// `None` for unattached objects (every freshly-allocated object;
    /// every Aura/Equipment that hasn't resolved yet). Mirrors
    /// Python's `GameObject.attached_to: int | None` slot used by
    /// `AttachToTargetCreatureYouControl` (`effects/generic/attach.py`).
    ///
    /// Cleared (reset to `None`) on every zone change (CR 400.7
    /// per-object state reset) — the attach relationship is a
    /// battlefield-only concept and shouldn't survive an Equipment /
    /// Aura leaving the battlefield. New objects default `None`,
    /// so the fingerprint serializer's previously-hardcoded `Null`
    /// emit remains byte-identical for every state where no
    /// `Effect::AttachToTargetCreatureYouControl` has resolved.
    pub attached_to: Option<u32>,
    /// Round-28 "until source leaves" exile linkage. When an Effect
    /// like `Effect::ExileTargetCreatureUntilSourceLeaves` exiles a
    /// permanent, it stamps `exiled_with_source = Some(source_id)`
    /// on the exiled object so that `return_paired_exiles_for(src_id)`
    /// can later move it back to its owner's battlefield. Mirrors
    /// Python's `GameObject.exiled_by_source: int | None` slot used
    /// by `ExileTargetCreatureUntilSourceLeaves` /
    /// `ExileOpponentsNonlandPermanentUntilLeaves` (`effects/generic/exile.py`).
    ///
    /// **First-slice manual hook.** Rust currently does NOT
    /// auto-trigger the return when the source leaves the battlefield;
    /// callers must explicitly invoke `return_paired_exiles_for(src_id)`
    /// at the right tick. This is a documented simplification — wiring
    /// the hook into every bf-leaving primitive (destroy, sacrifice,
    /// exile, return-to-hand, bounce, ...) is a follow-up slice.
    ///
    /// Defaults to `None` for every freshly-allocated object. Cleared
    /// (reset to `None`) on zone-change resets (CR 400.7) — once the
    /// exiled permanent returns to a non-exile zone, the link is
    /// stale. The fingerprint serializer emits this slot as `Null`
    /// when `None`, so default-state pinned baselines stay
    /// byte-identical.
    pub exiled_with_source: Option<u32>,
    /// **Stun-counter substrate (round-30 BLOCKED-RUST-EFFECT slice).**
    /// CR 701.20 — "If a permanent with a stun counter would become
    /// untapped, instead remove a stun counter from it." Tracked
    /// per-object as a non-negative count; the untap-step sweep at
    /// `state_ops_steps.rs:untap_all_for_player_inline` decrements
    /// `stun_counters` by 1 instead of clearing `tapped` whenever
    /// `stun_counters > 0`, mirroring Python's
    /// `src/mtg_bot/turn.py:88-96`.
    ///
    /// **Why a distinct field** (vs overloading `counters`): G8
    /// (multi-kind counter open gap) — stun counters are functionally
    /// orthogonal to +1/+1 / charge / loyalty. New field keeps the
    /// slice disjoint and matches the Python-side per-counter-type
    /// slot precedent at `src/mtg_bot/game.py:537`.
    ///
    /// Defaults to `0` for every freshly-allocated object. Per-counter-
    /// kind reset on zone change is NOT wired in this slice — Python's
    /// model treats stun-counter clearing as a per-effect concern
    /// (graveyard transit drops the field naturally because the bf
    /// object is discarded). Mirrors the charge_counters / loyalty_
    /// counters precedent. The fingerprint serializer does NOT include
    /// this field today (matching the snapshot's existing per-counter-
    /// kind elision); pre-r30 baselines stay bit-equal.
    pub stun_counters: i32,
    /// **CE4 closure — ETB-timestamp ordering.** Monotonic counter
    /// stamped at every battlefield-entry call site (cast →
    /// move_from_hand_to_battlefield, play_land, every ETB pathway in
    /// state_ops_resolve, token creation, reanimation, manifest, …).
    /// Defaults to `0` for every freshly-allocated GameObject; the
    /// non-zero values are written only when a Rust call site explicitly
    /// stamps the object as it enters the battlefield. Used by
    /// `battlefield_snapshot` (state_ops_snapshot.rs) and the
    /// fingerprint's battlefield walk (state_ops_fingerprint.rs) to
    /// emit battlefield objects in ETB order across both players,
    /// mirroring Python's centralized `state.battlefield` list ordered
    /// by `append()` order. Stable-sort by this field is the single
    /// observable surface — the field itself is NOT emitted into the
    /// per-object 8-tuple or the snapshot dict (keeping pre-CE4 pinned
    /// baselines bit-equal for states with no real ETBs, e.g. synthetic
    /// states with battlefields built up front in `make_player`).
    ///
    /// Ties (same `etb_timestamp` — possible only when a single Effect
    /// resolution pushes multiple bf objects without bumping the
    /// counter between pushes) are broken by stable-sort: insertion
    /// order is preserved within the same timestamp. Current call
    /// sites stamp a FRESH timestamp on every push (counter += 1 per
    /// push), so the multi-token / batch-bf-injection arms produce
    /// strictly-monotonic timestamps and the stable-sort tie-break
    /// path is only exercised by states constructed without going
    /// through a stamp site (default-0 case in synthetic-state fixtures).
    pub etb_timestamp: u32,
    /// Round-27 UEOT framework — per-object expiring-modifier stack.
    /// Each entry is a CR-613-layer continuous modifier with implicit
    /// `expires_at = "cleanup"`. Cleared in bulk on entry to
    /// PHASE_CLEANUP (CR 514.3) by `clear_ueot_modifiers_all` /
    /// `cleanup_step_for_active_player`. Readers that compute
    /// effective P/T or effective keywords walk this stack on top of
    /// the printing's printed values.
    ///
    /// Common case (no UEOT effect active on this creature) is an
    /// empty Vec — heap allocation deferred until the first push.
    /// The empty-Vec footprint adds 24 bytes to the per-object
    /// size; that's the cost of losing Copy. Mirrors Python's
    /// `GameObject.temp_power_bonus` / `temp_toughness_bonus` +
    /// the layer-6 keyword-grant / layer-2 control-change /
    /// layer-7b set-base-PT registries on `GameState` (see
    /// `src/mtg_bot/effects/modifiers.py` for the Python shape).
    /// First-slice variants: PowerToughnessBonus (Pump / Anthem
    /// shape, layer 7c) and GrantKeyword (haste / flying /
    /// lifelink shape, layer 6). Future-slice variants per the
    /// brief: BecomesLandCreature (Blinkmoth Nexus shape, layer
    /// 4+6+7b combined).
    pub ueot_modifiers: Vec<UeotModifier>,
    /// **warp_cost substrate (Phase 1, round-63).** Set `true` on a
    /// permanent the moment it enters the battlefield via a warp-cast
    /// (`cast_creature_warp`). The `PHASE_END_STEP`-entry sweep
    /// (`state_ops_steps.rs::warp_exile_sweep_on_end_step`) exiles
    /// every battlefield permanent with this flag at the beginning of
    /// the next end step, then clears it. Mirrors the role of
    /// Python's `GameObject.cast_via_warp` (set at warp-cast,
    /// scheduled for `WarpExileSelf`). Defaults to `false` for every
    /// freshly-allocated object (back-compat — the per-object
    /// fingerprint 8-tuple does NOT carry this slot, so pre-Phase-1
    /// pinned baselines stay byte-equal; only the zone membership the
    /// sweep changes is observable).
    pub warp_exile_pending: bool,
    /// **warp_cost substrate (Phase 1, round-63).** Set `true` when
    /// the `PHASE_END_STEP` sweep moves a warp-cast permanent to
    /// exile. The legal emitter (`state_ops_legal.rs`) offers a BASE
    /// `cast_spell_dict` (use_warp=false) recast at NORMAL
    /// `mana_cost` for every exile object with this flag; the
    /// `cast_creature` exile branch clears it on recast. Mirrors
    /// Python's `GameObject.warped` exile marker
    /// (`effects/generic/exile.py::WarpExileSelf` sets `obj.warped`).
    /// Defaults to `false`; not in the per-object fingerprint tuple
    /// (same back-compat rationale as `warp_exile_pending`).
    pub warped: bool,
    /// **Cast-from-exile-with-cost-hike substrate (Lightstall
    /// Inquisitor residue closure).** Set `true` on a card moved to
    /// exile by `Effect::EachOpponentExilesCardFromHandWithCastPermission`
    /// (Lightstall Inquisitor's ETB). The legal emitter
    /// (`state_ops_legal.rs`) surfaces a `CastSpell` action on every
    /// exile object with this flag where the controller is the
    /// CARD'S OWNER (the player who controls the exile zone the card
    /// lives in -- per CR 400.7, `card.controller` was reset to the
    /// owner when the card moved to exile), at sorcery timing, with
    /// a `+{1}` generic surcharge applied to the printed mana cost.
    /// The cast-pay path in `state_ops_cast.rs`
    /// (`cast_creature` / `cast_spell_noncreature`) honors the
    /// surcharge when this flag is set and clears it on a successful
    /// cast. For lands (the contract: any future card admitting a
    /// non-creature exile pick), `play_land` in
    /// `state_ops_cast.rs` honors the flag by stamping `tapped =
    /// true` on the entering land regardless of the printing's
    /// static `enters_tapped` flag and clears the flag.
    ///
    /// Mirrors Python's
    /// `GameObject.cast_from_exile_with_lightstall_surcharge`
    /// (`src/mtg_bot/game.py:487`). Persists until the card leaves
    /// exile (NOT cleared at cleanup -- the permission survives
    /// across turns); cleared at cast/play time. Defaults to
    /// `false` for every freshly-allocated GameObject. NOT in the
    /// per-object fingerprint tuple (back-compat: pre-substrate
    /// pinned baselines stay byte-equal; only the cast-from-exile
    /// surface, which had no Rust path pre-substrate, becomes
    /// observable).
    pub cast_from_exile_with_lightstall_surcharge: bool,
    /// **Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20).**
    /// Captured baseline of `controller` BEFORE the first
    /// control-changing Aura attached to this object. `None` when no
    /// control-change is active. When the last control-changing Aura
    /// leaves play (SBA bury / move-off-battlefield), the
    /// `controller` field is reverted from this baseline and the
    /// baseline cleared back to `None`.
    ///
    /// Mirrors Python's
    /// `GameObject.controller_id_before_any_aura_swap`
    /// (`game.py:580`). `None` is the back-compat default for every
    /// freshly-allocated object; the per-object fingerprint tuple
    /// does NOT include this slot, so pre-S2 pinned baselines stay
    /// byte-equal.
    pub controller_baseline: Option<u8>,
    /// **Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    /// closure, 2026-05-21).** Per-object "Token" supertype override.
    /// When `true`, every read site that asks "is this a Token?" must
    /// answer YES regardless of whether the underlying
    /// `printing.supertypes` contains `"Token"`. Set when this object
    /// was minted as a Moonlit-Meditation-replacement token — its
    /// `printing_id` points at the host's (non-Token) printing, but
    /// CR 614 says the resulting permanent is still a token. Mirrors
    /// Python's `_dc.replace(printing, supertypes=(*base, "Token"))`
    /// shim at `effects/generic/tokens.py:237-243`: Python builds a
    /// fresh CardPrinting instance with the Token supertype injected,
    /// then `state.new_object(printing, ...)` uses that printing for
    /// the freshly-created object. Rust avoids the runtime registry
    /// mutation (no fresh `printing_id` is minted) and instead carries
    /// the supertype injection per-object — the bridge round-trip is
    /// honest because Python's snapshot of the token reports Token
    /// supertype, and Rust's `is_token` reads through the override.
    ///
    /// `false` is the back-compat default for every freshly-allocated
    /// object — pre-Phase-4-S3-closure objects keep the printing-only
    /// is-token semantics bit-equivalent to before this slice. The
    /// per-object fingerprint tuple does NOT include this slot
    /// (pre-substrate pinned baselines stay byte-equal); the only
    /// observable shift is in the post-Moonlit-resolution snapshot,
    /// which is new behavior in this slice.
    ///
    /// Cleared (reset to `false`) on every zone change (CR 400.7
    /// per-object state reset) — the override is a battlefield-scoped
    /// concept stamped at token-mint time; an object that has ceased
    /// to exist or transited zones drops the marker, exactly as
    /// Python's per-token `state.new_object(printing, ...)` is a
    /// fresh object whose supertype set is fixed at mint time.
    pub token_supertype_override: bool,
    /// **Control-change-UEOT subsystem — Cycle 1 (2026-05-21).**
    /// Per-object UEOT controller override. When `Some(pid)`, every
    /// read site that asks "who controls this object?" must answer
    /// `pid` instead of `self.controller`. Set by
    /// `grant_control_change_ueot` (the substrate for Mutinous
    /// Massacre / Systems Override / Threaten / Act of Treason
    /// shapes); drained back to `None` at the cleanup step
    /// (CR 514.3 / CR 613 layer-2 "until end of turn" expiration).
    ///
    /// Mirrors Python's layer-2 `"control_change"` registry entry
    /// + the per-object `controller_id` write-through cache pattern
    /// (see `src/mtg_bot/game.py:effective_controller` and
    /// `grant_control_change_ueot`). Cycle 1 of this subsystem
    /// (substrate only) provides the field + the
    /// `effective_controller_id` helper + the cleanup-step drain;
    /// Cycle 2 reroutes raw `.controller` read sites through the
    /// helper; Cycle 3 wires the bridge round-trip into the cross-
    /// engine harness; Cycle 4 adds the Mutinous Massacre /
    /// Systems Override Effect variants that mint the override.
    ///
    /// **Distinct from `controller_baseline`.** `controller_baseline`
    /// is the captured-baseline anchor for control-changing Auras
    /// (Phase 2 S2 substrate); the Aura code writes the new
    /// controller into `self.controller` itself and uses
    /// `controller_baseline` to remember what to revert to when the
    /// Aura leaves play. The UEOT path is intentionally lighter-
    /// weight: it never mutates `self.controller`; the override is
    /// the source of truth until cleanup drains it. The two
    /// substrates compose cleanly — when Mutinous Massacre lands on
    /// an Aura-controlled creature, the UEOT override beats the
    /// Aura swap until cleanup, then the Aura swap re-asserts
    /// itself via the unchanged `self.controller`.
    ///
    /// **Back-compat default: `None`.** Every freshly-allocated
    /// GameObject (including `make_object`,
    /// `register_and_inject_object`, and the bridge import path)
    /// starts with `controller_override = None`. The per-object
    /// fingerprint tuple does NOT include this slot, so pre-Cycle-1
    /// pinned baselines stay byte-equal; only states that mint a
    /// UEOT control override observe the new field's effect, and
    /// those are new behavior in this slice.
    pub controller_override: Option<u8>,
}

/// A single CR 613 / CR 514.3 "until end of turn" modifier on a
/// GameObject. Mirrors the Python `LayerPayload` family in
/// `src/mtg_bot/effects/modifiers.py` (`PowerToughnessBonus`,
/// `GrantKeyword`, etc.) — but flattened into a closed enum the
/// way the Rust `Effect` enum flattens the Python `Effect` class
/// hierarchy. Adding a new payload kind is a build-time error in
/// every effective-P/T / effective-keyword reader until it's
/// handled (exhaustiveness gives us "fail loud" for free).
///
/// All variants implicitly carry `expires_at = "cleanup"` — the
/// cleanup-step UEOT clear drains the entire Vec, so no per-entry
/// lifetime tag is needed. Future "until your next turn" entries
/// (Mouth-of-the-Storm shape, see Python's
/// `grant_pt_pump_until_player_next_turn`) need a separate field
/// or a tagged-lifetime variant; not in r27 scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UeotModifier {
    /// CR 613.7c — additive power/toughness bonus. Models pump
    /// effects (PumpSelfUntilEOT, PumpTargetCreatureUntilEOT,
    /// PumpCreaturesYouControlUntilEOT). Multiple entries stack
    /// additively. Negative values model debuffs ("-3/-0").
    PowerToughnessBonus { power: i32, toughness: i32 },
    /// CR 613.6 — grant a keyword ability. Models the "and gains
    /// haste/flying/lifelink/trample/double-strike UEOT" rider
    /// common on pump spells and equipment ETBs. Lowercased
    /// keyword string matches the printing-side keyword vocab
    /// (see `parse_printing`'s `to_lowercase()` pass).
    GrantKeyword { keyword: String },
    /// CR 613.4 + 613.6 + 613.7b — "becomes a P/T color/subtype
    /// land creature UEOT" composite. Models Blinkmoth Nexus /
    /// Wandering Fumarole / Lavaclaw Reaches manland-animate
    /// activated abilities (BecomeLandCreatureUEOT, 13 cards).
    /// Sets base P/T to `power`/`toughness` (layer 7b) and adds
    /// the subtype string to the effective subtype set (layer 4);
    /// the "color" / creature-type combination is encoded in
    /// `color_or_subtype` for first-slice. Future expansion can
    /// split into separate layer-4 / layer-7b variants once the
    /// effective-subtype consumer surface lands.
    BecomesLandCreature {
        power: i32,
        toughness: i32,
        color_or_subtype: String,
    },
    /// CR 613.4 layer 7b — set base P/T (override). Models Python's
    /// `SetBasePTUntilEOT(power, toughness)` (Genemorph-Imago-shape
    /// "target creature has base power and toughness N/N until end of
    /// turn", 2+ EoE cards). The composer reads the LAST `SetBasePT`
    /// entry (most-recently-pushed wins, mirroring CR layer-7 last-
    /// applied semantics) and uses its `power`/`toughness` as the
    /// base instead of the printing's printed values. Layer-7c
    /// additive bonuses (`PowerToughnessBonus`) stack ON TOP of the
    /// set base, matching CR 613.7c.
    SetBasePT { power: i32, toughness: i32 },
    /// **BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner's
    /// granted dies-rider marker.** A UEOT-granted triggered ability
    /// of the fixed shape "When this creature dies, return it to the
    /// battlefield tapped under its owner's control" (CR 603 — a
    /// granted delayed-ish DIES trigger). Pushed onto the pumped
    /// target by `Effect::PumpTargetCreatureUntilEotGrantDiesReturnTapped`
    /// alongside the `PowerToughnessBonus`.
    ///
    /// Rust has no general layer-6 grant-triggered-ability registry
    /// (Python's `continuous_effects` `grant_triggered_ability` /
    /// `game.effective_triggered_abilities`). Rather than building
    /// that whole subsystem, this marker variant models exactly the
    /// Perigee shape — a payload-free flag the SBA destroy path
    /// (`state_ops_sba.rs`) honors: a creature carrying this marker
    /// that is destroyed via SBA is returned from its owner's
    /// graveyard back to its OWNER's battlefield, tapped, with its
    /// UEOT modifiers cleared (new-object semantics — mirrors
    /// Python's `PerigeeBeckonerDiesReturnTappedEffect.resolve` +
    /// the layer-6 grant clear in `effects/bespoke.py:2147`). The
    /// general nested-trigger-grant subsystem stays the loader-gated
    /// multi-round half (see docs/SEARCH_LIMITATIONS.md round-60).
    /// Auto-expires at the next CR 514.3 UEOT-clear pass like every
    /// other `UeotModifier` — so a creature that survives the turn
    /// loses the rider at cleanup, exactly as Python's EndOfTurn-
    /// bounded `ApplyEffect(GrantTriggeredAbility(...))` does.
    GrantedDiesReturnTapped,
}

/// **CE4 closure — reserve the next monotonic ETB timestamp.**
///
/// Reads `state.next_etb_timestamp`, increments the counter (saturating
/// add — u32 is wide enough that a realistic game won't overflow), and
/// returns the just-reserved value for the caller to stamp onto the
/// object before pushing it to a battlefield Vec.
///
/// Centralizes the read-then-bump idiom so every ETB call site stays
/// one-line. Saturating add keeps the corner-case tail-state sortable
/// (if a pathological state ever runs the counter to u32::MAX, all
/// further ETBs land at u32::MAX — still stable-sort-comparable, just
/// indistinguishable in order beyond that point).
pub fn reserve_etb_timestamp(state: &mut GameState) -> u32 {
    let ts = state.next_etb_timestamp;
    state.next_etb_timestamp = state.next_etb_timestamp.saturating_add(1);
    ts
}

/// **R76-1 closure — reserve the next monotonic object id (CR 111.7).**
///
/// Reads `state.next_object_id`, increments the counter (saturating
/// add — u32 is wide enough that a realistic game won't overflow),
/// and returns the just-reserved id for the caller to stamp onto the
/// fresh `GameObject` before pushing it into a zone.
///
/// This is the single monotonic id primitive every mid-game mint
/// site draws from. It mirrors Python's `GameState.new_object`
/// (`src/mtg_bot/game.py:2787-2798`), whose `self.next_object_id +=
/// 1` is the byte-equal source of truth: an object id created during
/// a game is NEVER reused, even after the object ceases to exist
/// (CR 111.7). Centralizing the read-then-bump idiom (same shape as
/// `reserve_etb_timestamp`) keeps every call site one line and makes
/// the non-monotonic `max_id + 1` re-scan that caused R76-1
/// structurally impossible to reintroduce at a mint site.
///
/// Multi-object mints (the per-`count` token loops) call this once
/// per object so the produced ids are strictly increasing and
/// intermix correctly with ids minted by prior / subsequent effects
/// — identical to the per-object `state.next_object_id += 1` Python
/// runs in its own per-token loop.
pub fn reserve_object_id(state: &mut GameState) -> u32 {
    let id = state.next_object_id;
    state.next_object_id = state.next_object_id.saturating_add(1);
    id
}

/// **CE20 Slice 5 (Walker loyalty — `starting_loyalty` ETB seeding).**
/// Per CR 606.5b, a planeswalker enters the battlefield with a number
/// of loyalty counters equal to its printed starting loyalty. This
/// helper looks up `printing.starting_loyalty` and stamps it onto the
/// freshly-entering object's `loyalty_counters` field.
///
/// **No-op conditions**:
///   - the printing's `card_types` doesn't contain `"Planeswalker"`
///     (the field has no observable effect on non-walkers);
///   - the registry lookup fails (defensive — should never happen
///     for a properly-registered printing, but a missing entry
///     leaves the field unchanged rather than panicking);
///   - the printing's `starting_loyalty == 0` (no observable change
///     to the default `obj.loyalty_counters == 0`).
///
/// **Idempotency**: callers may invoke this helper multiple times on
/// the same object; the second invocation OVERWRITES the field (it's
/// not an add). Today every caller stamps exactly once per bf-entry,
/// so the overwrite case is unreachable in practice — but the
/// semantics are documented for safety.
///
/// **Designed for ETB call sites**. The canonical bf-entry primitive
/// (`move_source_post_resolve` in `state_ops_resolve.rs`) and every
/// other ETB-stamping site that hosts a freshly-cast permanent calls
/// this helper alongside `reserve_etb_timestamp`. Synthetic state
/// fixtures (test helpers that inject a walker directly onto the bf
/// without going through cast) keep using the explicit
/// `set_loyalty_counters` test helper.
pub fn seed_starting_loyalty_for_etb(
    state: &GameState,
    obj: &mut GameObject,
) {
    let Some(printing) = state.registry.lookup(obj.printing_id) else {
        return;
    };
    let is_walker = printing
        .card_types
        .iter()
        .any(|t| t == "Planeswalker");
    if !is_walker {
        return;
    }
    obj.loyalty_counters = printing.starting_loyalty;
}

impl GameObject {
    /// Sum of UEOT `PowerToughnessBonus` power deltas on this
    /// object. Used by combat-damage / SBA / parity readers to
    /// compose `printed_power + counters_delta +
    /// ueot_power_bonus` into the effective power for "what does
    /// this creature look like RIGHT NOW" queries.
    pub fn ueot_power_bonus(&self) -> i32 {
        let mut sum = 0i32;
        for m in &self.ueot_modifiers {
            match m {
                UeotModifier::PowerToughnessBonus { power, .. } => sum += *power,
                UeotModifier::BecomesLandCreature { .. } => {}
                UeotModifier::GrantKeyword { .. } => {}
                UeotModifier::SetBasePT { .. } => {}
                // Round-60 Perigee dies-rider marker: no P/T effect
                // (it only changes what happens on death).
                UeotModifier::GrantedDiesReturnTapped => {}
            }
        }
        sum
    }

    /// Sum of UEOT `PowerToughnessBonus` toughness deltas. Same
    /// shape as `ueot_power_bonus`.
    pub fn ueot_toughness_bonus(&self) -> i32 {
        let mut sum = 0i32;
        for m in &self.ueot_modifiers {
            match m {
                UeotModifier::PowerToughnessBonus { toughness, .. } => sum += *toughness,
                UeotModifier::BecomesLandCreature { .. } => {}
                UeotModifier::GrantKeyword { .. } => {}
                UeotModifier::SetBasePT { .. } => {}
                // Round-60 Perigee dies-rider marker: no P/T effect.
                UeotModifier::GrantedDiesReturnTapped => {}
            }
        }
        sum
    }

    /// **Round-28 base-P/T override (CR 613.4 layer 7b).** If any
    /// `UeotModifier::SetBasePT` is on this object, return the
    /// `(power, toughness)` of the LAST one in the Vec (most-recent
    /// push wins, mirroring CR layer-7 last-applied semantics).
    /// Otherwise return `None` and the composer falls back to the
    /// printing's printed P/T.
    pub fn ueot_base_pt_override(&self) -> Option<(i32, i32)> {
        let mut last: Option<(i32, i32)> = None;
        for m in &self.ueot_modifiers {
            if let UeotModifier::SetBasePT { power, toughness } = m {
                last = Some((*power, *toughness));
            }
        }
        last
    }

    /// **Round-28 manland animation (CR 613.4 + 613.7b).** If any
    /// `UeotModifier::BecomesLandCreature` is on this object, return
    /// the `(power, toughness)` of the LAST one in the Vec (most-
    /// recent push wins; mirrors CR layer-7 last-applied semantics).
    /// Otherwise return `None`.
    ///
    /// Used by effective-P/T readers to detect "this land is a
    /// creature RIGHT NOW with base P/T (p, t)". The land also stays
    /// a land for the duration (CR 305.3), but the type-union piece is
    /// handled at the type-set reader level, not on this method.
    pub fn ueot_becomes_land_creature_pt(&self) -> Option<(i32, i32)> {
        let mut last: Option<(i32, i32)> = None;
        for m in &self.ueot_modifiers {
            if let UeotModifier::BecomesLandCreature { power, toughness, .. } = m {
                last = Some((*power, *toughness));
            }
        }
        last
    }

    /// True iff some `UeotModifier::BecomesLandCreature` is on this
    /// object — the "this permanent is a creature RIGHT NOW even if
    /// its printing has no printed power" gate. Used by effective-P/T
    /// readers and the effective-creature-type predicate to fold the
    /// manland-animate window into combat-relevant queries.
    pub fn ueot_is_creature_via_become(&self) -> bool {
        self.ueot_modifiers
            .iter()
            .any(|m| matches!(m, UeotModifier::BecomesLandCreature { .. }))
    }

    /// True iff some `UeotModifier::GrantKeyword` on this object
    /// carries `keyword` (case-sensitive lowercase match — the
    /// printing-side keyword vocab is already lowercased in
    /// `parse_printing`). Used by combat readers as a "does this
    /// creature have haste/flying/... RIGHT NOW" gate that folds
    /// printed keywords with UEOT-granted keywords.
    pub fn ueot_has_keyword(&self, keyword: &str) -> bool {
        for m in &self.ueot_modifiers {
            if let UeotModifier::GrantKeyword { keyword: k } = m {
                if k == keyword {
                    return true;
                }
            }
        }
        false
    }

    /// **Aura-attachment Phase 4 S3 closure (2026-05-21).** Effective
    /// "is this object a Token?" reader that consults BOTH the
    /// per-object `token_supertype_override` flag AND the underlying
    /// printing's supertypes. Returns `true` iff EITHER:
    ///
    /// * The override flag is set (a Moonlit-Meditation-replacement
    ///   token that carries the host's `printing_id`, where the
    ///   host's printing does NOT itself carry `"Token"`), OR
    /// * The underlying printing's `supertypes` Vec contains
    ///   `"Token"` (a normal token whose registry entry was minted
    ///   with the Token supertype).
    ///
    /// Mirrors Python's `obj.printing.is_token` consumer pattern —
    /// in Python, the `_dc.replace(base, supertypes=(*base, "Token"))`
    /// shim produces a printing whose `is_token` already returns
    /// `True`, so Python's consumers don't need a per-object
    /// override branch. Rust's CardPrinting is registry-resident and
    /// statically typed, so we add the override at the GameObject
    /// layer — every call site that asked "is this a Token?" via
    /// `registry.is_token(obj.printing_id)` should migrate to
    /// `obj.effective_is_token(&registry)` to honor the override.
    ///
    /// Pre-Phase-4-S3-closure: `effective_is_token` and
    /// `registry.is_token(printing_id)` agree byte-for-byte (the
    /// override flag is `false` on every GameObject minted by every
    /// pre-substrate call site). Post-closure: the divergence is
    /// observable only on the Moonlit-substituted token path, which
    /// is new behavior in this slice.
    pub fn effective_is_token(&self, registry: &RegistryData) -> bool {
        if self.token_supertype_override {
            return true;
        }
        registry.is_token(self.printing_id)
    }
}

// Zones are `Arc<Vec<GameObject>>` so cloning a `Player` (and therefore
// a `GameState`) is an atomic refcount bump per zone instead of a full
// copy. Mutating a zone goes through `Arc::make_mut`, which clones the
// inner Vec only if any other clone is observing it — same shape as
// the Python side's `_CowList`.
pub type Zone = Arc<Vec<GameObject>>;

/// A single coin in a player's mana pool. Mirrors Python's
/// `ManaCoin` (src/mtg_bot/game.py:2450): one color tag per coin.
///
/// **G7 S6-Class-E ordered-pool slice.** Python represents
/// `Player.mana_pool` as a `list[ManaCoin]`. `auto_tap_for_cost`
/// drains generic cost from that list in **insertion / FIFO order**
/// (`actions.py:964-976`: `for idx, coin in enumerate(p.mana_pool)`),
/// NOT by a fixed color priority. Mirroring that semantics
/// byte-faithfully requires Rust to preserve coin insertion order —
/// which a per-color count struct discards at `add_mana` time.
/// `ManaCoin` is the unit of the ordered representation; the
/// per-color count fields on `ManaPool` remain as a denormalized
/// cache for O(1) reads and back-compat with the existing readers
/// (`get_mana_pool`, `pool_can_cover`, `pool_total`, fingerprint,
/// snapshot — all unchanged). Mutation sites keep the two in sync.
///
/// **Color vocab:** `'W'`, `'U'`, `'B'`, `'R'`, `'G'`, `'C'`. The
/// closed vocabulary matches every other mana-color call site in
/// the engine (`slot_mut` in `state_ops_mana.rs:72-87`, the
/// `mana_produced` symbols, the cost-payment slot lookups). Stored
/// as a single byte for cheap clone/copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManaCoin {
    pub color: u8, // ASCII byte: b'W'/b'U'/b'B'/b'R'/b'G'/b'C'
}

impl ManaCoin {
    /// Construct from a color &str. The closed vocabulary is
    /// W/U/B/R/G/C — anything else returns `None`. Used by every
    /// `add_mana`-shape mutation site to build the coin to push.
    pub fn from_color_str(color: &str) -> Option<Self> {
        let b = match color {
            "W" => b'W',
            "U" => b'U',
            "B" => b'B',
            "R" => b'R',
            "G" => b'G',
            "C" => b'C',
            _ => return None,
        };
        Some(Self { color: b })
    }

    /// The color as a `&'static str` (cheap; no allocation).
    pub fn color_str(&self) -> &'static str {
        match self.color {
            b'W' => "W",
            b'U' => "U",
            b'B' => "B",
            b'R' => "R",
            b'G' => "G",
            b'C' => "C",
            _ => "?", // Defensive; from_color_str gates the vocab.
        }
    }
}

/// Mana pool for one player. Five colors + colorless; capped at u8
/// per color (255 of any color is more than any real game produces).
/// Hybrid / X / snow mana are not yet modeled; future slices add
/// them when a card needs them.
///
/// **G7 S6-Class-E ordered-pool representation.** `coins` is the
/// insertion-ordered SoT (Python parity — `Player.mana_pool` is a
/// `list[ManaCoin]`). The six per-color count fields are a
/// denormalized cache populated and maintained alongside `coins` by
/// every mutation site (`add_mana`, the `pay_mana_cost*` family,
/// `auto_tap_pay_for_cost`, `empty_*_mana_pool*`, the resolver
/// `AddMana` arms, the legal-side `auto_tap_headroom_pool`). The
/// counts let O(1) read paths (`pool_can_cover`, `pool_total`,
/// fingerprint/snapshot, `get_mana_pool`) stay unchanged; the
/// `coins` vec only matters when a DRAIN consumer cares about
/// insertion order (currently: `auto_tap_pay_for_cost` generic
/// drain, which is the Class-E fix lane).
///
/// **Sync invariant**: at every quiescent point,
/// `coins.iter().filter(|c| c.color == b'W').count() == self.white`
/// (and same for U/B/R/G/C). Mutation sites enforce this by either
/// (a) updating both, or (b) recomputing the counts from `coins` at
/// the end of the mutation. `assert_count_sync` (debug-only)
/// validates the invariant on every parity-critical pay/drain path
/// so a future mutation that touches only one side fails loudly
/// instead of producing a silent drift.
///
/// **Not `Copy`** — `Vec<ManaCoin>` makes the struct non-trivially
/// destructible. The two pre-existing by-value sites
/// (`auto_tap_pay_for_cost` snapshot, `auto_tap_headroom_pool`
/// hypothetical pool) clone explicitly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ManaPool {
    pub white: u8,
    pub blue: u8,
    pub black: u8,
    pub red: u8,
    pub green: u8,
    pub colorless: u8,
    /// Insertion-ordered SoT — mirrors Python's `list[ManaCoin]`.
    /// `add_mana` appends; drain consumers that care about FIFO
    /// (Python `auto_tap_for_cost`'s generic drain) iterate this.
    pub coins: Vec<ManaCoin>,
}

impl ManaPool {
    /// Append `count` coins of `color` (ASCII byte). Bumps both the
    /// `coins` vec (insertion-ordered) and the per-color denormalized
    /// count. `color` must be one of `b'W'`, `b'U'`, `b'B'`, `b'R'`,
    /// `b'G'`, `b'C'` — defensive no-op on an unknown byte (every
    /// public caller already validates the color through
    /// `ManaCoin::from_color_str` or the resolver's vocabulary).
    ///
    /// The per-color count uses `saturating_add` (matches the
    /// existing `add_mana` shape — 255 of any one color is more than
    /// any real game produces, and silent wrap would be a worse
    /// failure mode than capping). The `coins` vec grows unbounded
    /// because it carries identity ordering; a pool with 256 coins is
    /// not a real-game scenario.
    pub fn push_coins(&mut self, color: u8, count: u8) {
        let slot = match color {
            b'W' => &mut self.white,
            b'U' => &mut self.blue,
            b'B' => &mut self.black,
            b'R' => &mut self.red,
            b'G' => &mut self.green,
            b'C' => &mut self.colorless,
            _ => return, // Defensive no-op on unknown byte.
        };
        *slot = slot.saturating_add(count);
        for _ in 0..count {
            self.coins.push(ManaCoin { color });
        }
    }

    /// Remove the first `count` coins (FIFO) matching `predicate`,
    /// returning how many were actually removed. Used by drain paths
    /// that need to consume specific coins (e.g. the colored-payment
    /// phase of `auto_tap_pay_for_cost`, which drains coins of the
    /// matching color from the pool side in insertion order). Updates
    /// the per-color count cache in lockstep.
    ///
    /// `predicate(&coin) -> bool` decides whether to consume a coin.
    /// Returns the actual count drained (≤ `count`, ≤ matching coins
    /// present).
    pub fn drain_first_n_where<F: Fn(&ManaCoin) -> bool>(
        &mut self,
        count: u32,
        predicate: F,
    ) -> u32 {
        if count == 0 {
            return 0;
        }
        let mut removed: u32 = 0;
        let mut i = 0;
        while i < self.coins.len() && removed < count {
            if predicate(&self.coins[i]) {
                let c = self.coins.remove(i);
                // Decrement the matching denormalized slot.
                let slot = match c.color {
                    b'W' => &mut self.white,
                    b'U' => &mut self.blue,
                    b'B' => &mut self.black,
                    b'R' => &mut self.red,
                    b'G' => &mut self.green,
                    b'C' => &mut self.colorless,
                    _ => {
                        removed += 1;
                        continue;
                    }
                };
                *slot = slot.saturating_sub(1);
                removed += 1;
            } else {
                i += 1;
            }
        }
        removed
    }

    /// Drain the first `count` coins (FIFO over `coins`, ANY color)
    /// — the byte-mirror of Python's generic-cost drain idiom
    /// `for idx, coin in enumerate(p.mana_pool): ... generic_need -= 1`
    /// (`actions.py:965-973`). Returns the actual count drained.
    /// Updates the per-color count cache in lockstep.
    pub fn drain_first_n_fifo(&mut self, count: u32) -> u32 {
        self.drain_first_n_where(count, |_| true)
    }

    /// Drain the first `count` coins of `color` (FIFO over `coins`).
    /// Returns the actual count drained. Used by colored-payment
    /// paths to keep `coins` in sync with the per-color count cache:
    /// when an algorithm decrements `pool.green` by N, it should
    /// also drain N green coins from the head of `coins`.
    pub fn drain_first_n_color(&mut self, color: u8, count: u32) -> u32 {
        self.drain_first_n_where(count, |coin| coin.color == color)
    }

    /// Repopulate `coins` from the per-color count cache, building a
    /// canonical W/U/B/R/G/C-ordered coin list. Used as a recovery
    /// path for the very small number of legacy paths that mutate
    /// only the per-color counts (a transitional helper — every new
    /// path should keep both sides in sync directly). The order isn't
    /// "real" insertion order, of course; this is a last-resort
    /// rebuilder for sites that intentionally don't track insertion
    /// (e.g. inline cleanup-step resets that are about to zero the
    /// pool anyway).
    pub fn rebuild_coins_from_counts(&mut self) {
        self.coins.clear();
        for _ in 0..self.white {
            self.coins.push(ManaCoin { color: b'W' });
        }
        for _ in 0..self.blue {
            self.coins.push(ManaCoin { color: b'U' });
        }
        for _ in 0..self.black {
            self.coins.push(ManaCoin { color: b'B' });
        }
        for _ in 0..self.red {
            self.coins.push(ManaCoin { color: b'R' });
        }
        for _ in 0..self.green {
            self.coins.push(ManaCoin { color: b'G' });
        }
        for _ in 0..self.colorless {
            self.coins.push(ManaCoin { color: b'C' });
        }
    }

    /// Debug-only invariant check: `coins`'s per-color histogram
    /// matches the denormalized count fields. Runs only under
    /// `debug_assertions`; release builds compile to nothing. Used as
    /// a belt-and-suspenders guard on the pay/drain paths a future
    /// mutation could touch without keeping both sides synced.
    #[inline]
    pub fn assert_count_sync(&self) {
        if cfg!(debug_assertions) {
            let mut w = 0u32;
            let mut u = 0u32;
            let mut b = 0u32;
            let mut r = 0u32;
            let mut g = 0u32;
            let mut c = 0u32;
            for coin in &self.coins {
                match coin.color {
                    b'W' => w += 1,
                    b'U' => u += 1,
                    b'B' => b += 1,
                    b'R' => r += 1,
                    b'G' => g += 1,
                    b'C' => c += 1,
                    _ => {}
                }
            }
            debug_assert_eq!(w, self.white as u32, "ManaPool W drift");
            debug_assert_eq!(u, self.blue as u32, "ManaPool U drift");
            debug_assert_eq!(b, self.black as u32, "ManaPool B drift");
            debug_assert_eq!(r, self.red as u32, "ManaPool R drift");
            debug_assert_eq!(g, self.green as u32, "ManaPool G drift");
            debug_assert_eq!(c, self.colorless as u32, "ManaPool C drift");
        }
    }
}

#[derive(Clone, Debug)]
pub struct Player {
    pub id: u8,
    pub life: i32,
    pub mana_pool: ManaPool,
    pub hand: Zone,
    pub library: Zone,
    pub battlefield: Zone,
    pub graveyard: Zone,
    /// Exile zone (CR 406). Cards exiled by effects land here.
    /// Round-11 foundation addition — round-1 zone primitives
    /// (find_object_zone / move_object) iterate the four pre-r11
    /// zones and don't yet search exile; that's a future-slice
    /// upgrade once anything is read FROM exile (today exile is
    /// write-only via Effect::ExileTarget).
    pub exile: Zone,
    /// Sticky loss flag (CR 104.2). Once `true`, the player has
    /// lost the game and the loss is permanent — subsequent
    /// `gain_life` or other mutations cannot revive them. Mirrors
    /// Python's `Player.has_lost`. Read by `losers()` /
    /// `winner()` / `is_game_over()` / `apply_sbas()`. Written by
    /// `apply_sbas()` (transitions `attempted_draw_on_empty` ->
    /// `has_lost` and the existing `life <= 0` predicate). Round-26
    /// G11 closure. (No explicit `lose_game(p)` primitive yet — the
    /// skipped `test_explicit_lose_game_marks_player` un-skip lands when
    /// the dispatch hook for Phage / Bargain / deck-out effect arms is
    /// wired; this slice only adds the field + SBA transition.)
    pub has_lost: bool,
    /// Cause-signal flag set by the empty-library draw path (CR
    /// 704.5b — a player who attempted to draw from an empty library
    /// loses on the next SBA pass). Written by `draw_card` and the
    /// `Effect::DrawCards` resolution arm when the library pop returns
    /// None; read by `apply_sbas` which then transitions this flag
    /// into the sticky `has_lost`. `losers()` ALSO OR's this flag
    /// in directly so the loss is observable BEFORE the next SBA pass
    /// runs (mirrors Python's eager `has_lost = True` in
    /// `turn.draw_card`, where the loss is visible to the very next
    /// `losers()` read). Round-26 G10 closure.
    pub attempted_draw_on_empty: bool,
    /// Per-player count of `mulligan` / `mulligan_to_n` calls taken on
    /// this state. Incremented at the START of each call; if it
    /// exceeds the practical London-mulligan cap (7 — mirrors Python
    /// `simulate.setup_game(max_mulligans=7)`), the call raises
    /// instead of performing the shuffle. Round-26 G23 closure.
    /// Per CR 103.4 the 8th mulligan would draw a hand of -1 cards,
    /// which is nonsensical; Python guards it by capping the loop at
    /// 7 mulligans before forcing keep. The Rust primitive raises so
    /// callers can't silently exceed the cap.
    ///
    /// NOT serialized into the fingerprint — this is a per-call
    /// safety counter, not part of the gameplay-relevant state digest.
    /// Same `(state, seed)` pre- and post-mulligan must produce the
    /// same fingerprint regardless of how many mulligans were taken
    /// to get there.
    pub mulligan_count: u32,
    /// Per-player cumulative life gained this turn (reset to 0 at the
    /// cleanup step, mirroring Python's `Player.life_gained_this_turn`
    /// which is zeroed at `turn.py:1108`).  Tracks the running sum for
    /// `Effect::ConditionalIfYouGainedLifeAtLeast` — the gate reads
    /// this field at resolve time.  Written by the `Effect::GainLife`
    /// resolve arm (state_ops_resolve.rs) whenever life increases.
    /// Negative `amount` (life loss via GainLife, legal but unusual) does
    /// NOT decrement — only positive gains count, same as Python's
    /// `gain_life()` function which guards on `amount > 0` (game.py:1360).
    pub life_gained_this_turn: i32,
    /// Poison counter count (CR 704.5c).  A player with 10 or more poison
    /// counters loses the game — enforced by `apply_sbas`.  Mirrors
    /// Python's `Player.poison_counters` (game.py).  Written by the
    /// `Effect::PutPoisonCountersOnTargetPlayer` resolve arm
    /// (state_ops_resolve.rs).  Reset is NOT done at cleanup (poison
    /// counters persist between turns per CR 704.5c).
    ///
    /// **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
    pub poison_counters: i32,
    /// **MMM Pin 1 (11th Rust slice) — spells_cast_this_turn.**
    /// Per-player cumulative count of spells cast this turn. Reset
    /// to 0 at the cleanup step (mirroring Python's
    /// `Player.spells_cast_this_turn` zeroed in `turn.py:1105`).
    /// Tracks the running total for
    /// `StatePredicate::SpellsCastThisTurn { n }` — the gate latches
    /// when the counter flips to exactly `n` post-increment.
    ///
    /// **Increment site.** Bumped by `+1` in every Rust `cast_*`
    /// primitive that fires `Event::Cast`, IMMEDIATELY BEFORE the
    /// `dispatch_event(Event::Cast, ...)` call. The pre-dispatch
    /// position mirrors Python's `actions._cast_spell` which
    /// increments `p.spells_cast_this_turn` then dispatches IS_CAST
    /// (`actions.py:1777`/`:1795`), so the StateBound predicate
    /// observes the post-increment value when the trigger fires
    /// ("your second spell" latches the moment the count reaches 2).
    ///
    /// **Scope.** Counts ALL spells cast, regardless of card type
    /// (creature, instant, sorcery, etc.) — same as Python.
    /// Noncreature-spell tracking is a separate counter (Python's
    /// `noncreature_spells_cast_this_turn`); Rust does NOT mirror
    /// that yet (no EoE card uses the noncreature variant in the
    /// triggered-ability path).
    pub spells_cast_this_turn: i32,
    /// **Static / layered abilities substrate — round-82 (Mechan
    /// Shieldmate closure, 2026-05-20).** Per-player cumulative
    /// count of artifact permanents (including artifact creatures
    /// and artifact tokens) that have entered the battlefield
    /// under this player's control this turn. Reset to 0 at the
    /// cleanup step (mirroring Python's
    /// `Player.artifacts_entered_under_your_control_this_turn`
    /// zeroed in `turn.py:1110`).
    ///
    /// **Increment site.** Bumped by `+1` in
    /// `drain_pending_etb_dispatches` for every ENTERS dispatch
    /// whose `entered_id`'s printing has `"Artifact"` in its
    /// `card_types`. The drainer is the centralized chokepoint
    /// every bf-entry path routes through (`play_land` /
    /// `cast_creature` / token-creation arms / register-and-inject
    /// / resolve_top_of_stack), so a one-place bump catches every
    /// site Python's `actions.put_into_battlefield` increment
    /// covers (`actions.py:3142`).
    ///
    /// **Gate consumer.** Read by
    /// `state_ops_combat::static_self_bonus_condition_matches`
    /// when the bonus's `condition_tag` is
    /// `"controlled_artifact_entered_this_turn"` — Mechan
    /// Shieldmate's "as long as an artifact entered the
    /// battlefield under your control this turn, this creature
    /// loses defender" rider.
    pub artifacts_entered_under_your_control_this_turn: i32,
}

/// Discriminant for the two kinds of stack items the engine
/// supports today: spells (or activated abilities sharing the
/// spell-resolve path) and triggered abilities (pushed by
/// `dispatch_event` per `docs/RUST_ETB_TRIGGER_AUDIT.md` Slice 1).
///
/// **Lower-churn variant of the audit's StackItem-enum proposal.**
/// Keeping `StackItem` as a struct (rather than a kind-enum) means
/// every existing call site that reads `item.source_obj_id` /
/// `item.controller` keeps working unchanged; the new `kind`
/// discriminant + `trigger_payload` Option are additive. Spell
/// stack items carry `kind = Spell` and `trigger_payload = None`
/// (default-constructed via `StackItem::new_spell`); trigger stack
/// items carry `kind = TriggeredAbility` and a `Some(payload)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackItemKind {
    /// A spell (or other resolve-via-effects entity that ports
    /// through `move_source_post_resolve`). The default for every
    /// pre-Slice-1 call site.
    Spell,
    /// A triggered ability pushed by `dispatch_event`. Its
    /// `Effect` lives in `trigger_payload`; the resolver fires the
    /// effect via `apply_effect` and DOES NOT call
    /// `move_source_post_resolve` (triggers aren't permanents —
    /// their source already lives wherever the trigger fired from).
    TriggeredAbility,
}

/// Payload carried on a `StackItem::TriggeredAbility`. Mirrors
/// Python's `StackItem.ability` reference (the ability is keyed
/// back to its printing in Python; the Rust side carries the
/// effect inline because a `CardPrinting`'s `triggered_abilities`
/// Vec is reachable via
/// `(source_id -> printing_id -> abilities[ability_index])`, but
/// inlining the effect avoids a round-trip through the registry
/// at resolve time).
///
/// Slice 1 minimum field set. Future slices (target selection,
/// fixed_targets, combat-damage aggregator dedup) extend.
///
/// **CE20 follow-up Slice 4 (Equipment-equip)** grows the payload
/// with `target_id: Option<u32>` to carry the chosen target for
/// activated abilities whose cost flags
/// `requires_target_creature_you_control = true`. The
/// `Effect::EquipToTargetCreatureYouControl` resolve arm reads this
/// field at resolve-time (replacing the first-eligible-creature
/// walk for the payload-target path).
#[derive(Clone, Debug)]
pub struct TriggerPayload {
    /// The effect to fire when this trigger resolves.
    pub effect: crate::cards::Effect,
    /// Stable identifier for the (source_id, ability_index) pair —
    /// matches the once-per-turn ledger key in
    /// `GameState.triggered_this_turn`. `ability_index` is the
    /// index into `source.printing.triggered_abilities`.
    pub ability_index: u32,
    /// **CE20 follow-up Slice 4 (Equipment-equip).** Chosen target
    /// for activations whose cost flags
    /// `requires_target_creature_you_control = true`. The
    /// `Effect::EquipToTargetCreatureYouControl` resolve arm reads
    /// this field. `None` for every non-target trigger /
    /// activation (the pre-Slice-4 shape — ETB triggers, Bear-Caster-
    /// style draw activations, sacrifice-self activations).
    pub target_id: Option<u32>,
}

/// One pending entry on the stack. Two kinds today: spells and
/// triggered abilities (Slice 1 of the ETB-trigger closure —
/// `docs/RUST_ETB_TRIGGER_AUDIT.md`). The kind-discriminant +
/// optional payload pattern keeps every pre-Slice-1 reader of
/// `item.source_obj_id` / `item.controller` working unchanged;
/// new readers (the trigger-resolver arm in
/// `resolve_top_of_stack`) check `kind` first.
#[derive(Clone, Debug)]
pub struct StackItem {
    pub source_obj_id: u32,
    pub controller: u8,
    /// Discriminant. Default `Spell` (every pre-Slice-1 push is
    /// `Spell`); `TriggeredAbility` is set by
    /// `push_triggered_ability` in `state_ops_resolve`.
    pub kind: StackItemKind,
    /// `Some(_)` iff `kind == TriggeredAbility`. The resolver reads
    /// the effect from here and calls `apply_effect` directly,
    /// skipping `move_source_post_resolve` (a trigger has no
    /// post-resolve permanent to place — its source lives wherever
    /// the trigger fired from).
    pub trigger_payload: Option<TriggerPayload>,
    /// **Spell-card-bridge lever, Phase 1 (round-70).** The
    /// deterministic-first bound target for a `StackItemKind::Spell`
    /// whose `spell_effect` is single-required-target. Computed at
    /// cast time in `cast_spell_noncreature` (the cast-time
    /// first-legal scan, mirroring Python's `_select_spell_targets`
    /// binding one target onto the StackItem at cast time) and
    /// swapped into `current_resolve_payload_target_id` by both
    /// resolve entry points so any per-card bespoke Effect arm that
    /// reads the payload-target convention resolves byte-equal
    /// whether driven by a trigger OR a spell cast. `None` for every
    /// pre-lever spell push (back-compat — the discriminant-plus-
    /// optional pattern this struct's doc comment already
    /// establishes; the closed Tier-A spell arms self-scan the
    /// battlefield first-creature and IGNORE this field, so the swap
    /// is a correct no-op for them — it is the forward substrate for
    /// targeted arms that opt into the payload-target convention,
    /// e.g. `EquipToTargetCreatureYouControl`).
    pub target_obj_id: Option<u32>,
    /// **Spell-card-bridge lever, Phase 2 (round-76).** The SECOND
    /// deterministic-first bound target for a `StackItemKind::Spell`
    /// whose `spell_effect` is a 2-slot heterogeneous-target spell
    /// (today: Diplomatic Relations — slot 0 = friendly pumped
    /// creature on `target_obj_id`, slot 1 = opponent's damaged
    /// creature here). Computed at cast time in
    /// `cast_spell_noncreature` (the 2-slot first-legal scan,
    /// mirroring Python's `_select_spell_targets` per-`slot_kinds`
    /// pick) and swapped into `current_resolve_payload_target_id_2`
    /// by both resolve entry points. `None` for every single-target
    /// / pre-Phase-2 spell push (back-compat — the
    /// discriminant-plus-optional pattern; the closed Phase-1
    /// single-target arms self-scan and IGNORE both fields, so the
    /// swap is a correct no-op for them).
    pub target_obj_id_2: Option<u32>,
    /// **ModalChooseOne envelope — Cycle 1 (cast-time mode binding,
    /// 2026-05-21).** The cast-time-picked mode index for a
    /// `StackItemKind::Spell` whose `spell_effect` is an
    /// `Effect::ChooseOneMode { modes }`. Mirrors Python's
    /// `StackItem.chosen_modes: tuple[int, ...]` (`game.py:2342`)
    /// in its single-pick first-slice shape — a 1-tuple's worth of
    /// mode index, the no-`or_both` branch. Per CR 601.2e, modes
    /// are picked at cast time BEFORE the total cost is determined
    /// at 601.2f; storing the pick on the StackItem lets the
    /// resolve arm replay the choice without re-prompting at
    /// resolve time (parity with Python's
    /// `state.resolving_modes = item.chosen_modes` swap at
    /// `actions.py:2712`).
    ///
    /// **First-slice shape (Cycle 1).** `Option<u8>` (single pick),
    /// not a `Vec<u8>` — the cast-time picker
    /// (`cast_spell_noncreature`) populates `Some(0)` for the
    /// deterministic-first no-agent fallback (Python's
    /// `_choose_modes_at_cast` returns `(0,)` when the player's
    /// agent is `None`, the harness convention). `None` for every
    /// pre-cycle push (back-compat — the closed Phase-1 spell arms
    /// + the `PendingDecision::ChooseModeChoice` resolve-time path
    /// both ignore this field today).
    ///
    /// **`or_both` deferral.** Python's
    /// `ModalChooseOne(or_both=True)` returns a 1- or 2-tuple of
    /// mode indices in resolution order. The Cycle 1 substrate is
    /// single-pick only — `or_both` cards stay declined by the
    /// admit predicate. A future cycle can widen this field to
    /// `Option<Vec<u8>>` (or a fixed-arity `[u8; 2]` sibling) when
    /// the `or_both` lane lands.
    pub chosen_mode_idx: Option<u8>,
    /// **Close Encounter additional-cost-choice envelope — Cycle 1
    /// (cast-time choice binding, 2026-05-21).** The cast-time-picked
    /// additional-cost choice id for a `StackItemKind::Spell` whose
    /// printing carries `additional_cost_choose_your_creature = true`
    /// (today: Close Encounter; future siblings as the lane unlocks
    /// more "choose a creature" additional-cost cards). Mirrors
    /// Python's `StackItem.additional_cost_choice_id: Optional[int]`
    /// at `game.py:2314` — the chosen object's id (a creature the
    /// caster controls OR a warped creature card the caster owns in
    /// exile, per CR 601.2f additional-cost-pay).
    ///
    /// **Wire shape**: cast pipeline picks the deterministic-first
    /// legal option at cast time (scan battlefield-creatures-you-
    /// control first, then warped exile creature cards; first-id
    /// wins) and stores it here. The resolve arm reads it via the
    /// `state.current_resolve_additional_cost_choice_id` transient
    /// (the save → set → restore swap pattern, same as
    /// `current_resolve_chosen_mode_idx`). `None` for every spell
    /// whose printing does NOT carry the choose-your-creature flag
    /// — back-compat byte-identical to the pre-Cycle-1 push.
    ///
    /// **Cycle 1 substrate-only scope.** This field stores the
    /// choice; the resolve-arm consumer is Cycle 3 (the
    /// `Effect::CloseEncounterEffect` arm). Cycle 1 ships the
    /// envelope + cast-time bind + transient swap so the substrate
    /// is in place before the Effect variant lands; until Cycle 3
    /// no Effect arm reads it, so the substrate is parity-NEUTRAL
    /// (no Close-Encounter card is bridge-admitted yet — the
    /// `additional_cost_choose_your_creature` flag still triggers
    /// the bridge decline at `trajectory.py::_printing_is_bridge_
    /// simple_spell`'s covered-set check via `CloseEncounterEffect`
    /// not in `covered_python_class_set`).
    pub additional_cost_choice_id: Option<u32>,
    /// **ModalChooseOne envelope — scour-for-scrap-slice (Cycle 1
    /// widen, 2026-05-21).** Multi-pick widening of `chosen_mode_idx`
    /// for `ModalChooseOne(or_both=True)` cards (Scour for Scrap and
    /// any future Charm-shape). When `Some(vec)`, the resolve arm
    /// dispatches EACH chosen mode in supplied order (CR 700.2d —
    /// modes resolve in the controller-picked order). When `None`,
    /// the resolve arm falls through to the legacy single-pick path
    /// via `chosen_mode_idx` (back-compat byte-identical for every
    /// pre-widen modal cast).
    ///
    /// **Why a separate field instead of widening `chosen_mode_idx`
    /// to `Vec<u8>`.** The single-pick field has 26+ read sites
    /// (state_ops_cast, state_ops_resolve, state_ops_stack, test
    /// modules). An additive sibling keeps every existing call site
    /// byte-identical while letting the multi-pick path opt in via
    /// `cast_spell_noncreature`'s `or_both` detector. Refactoring
    /// the single-pick field into `Vec<u8>` is the natural Cycle 2
    /// follow-up once the additive substrate has bedded in; the
    /// honest-stop discipline prefers a clean additive landing here.
    ///
    /// **Default `None` everywhere** (every pre-widen StackItem
    /// constructor + back-compat synthetic test pushes).
    pub chosen_mode_idxs: Option<Vec<u8>>,
    /// **X-value subsystem — Cycle 1 (cast-time X bind envelope,
    /// 2026-05-21).** The cast-time-bound X value for a
    /// `StackItemKind::Spell` whose printing carries an X-cost shape
    /// (today: Zero Point Ballad, Devastating Onslaught — both
    /// gate-declined pre-substrate; siblings as the lane unlocks
    /// more X-cost cards). Mirrors Python's `StackItem.x_value:
    /// Optional[int]` (`game.py`) — the X picked at cast time per
    /// CR 601.2f-h (X is set when the spell is announced and the
    /// total cost is then computed; storing it on the StackItem
    /// lets the resolve arm replay the choice without re-deriving
    /// it from mana paid).
    ///
    /// **Wire shape**: cast pipeline (Cycle 2/3) will populate
    /// `Some(x)` for X-cost spells; the resolve arm reads it via
    /// the `state.current_resolve_x_value` transient (the save →
    /// set → restore swap pattern, mirroring its sibling slots like
    /// `current_resolve_chosen_mode_idx` /
    /// `current_resolve_additional_cost_choice_id`). `None` for
    /// every non-X-cost spell — back-compat byte-identical to the
    /// pre-Cycle-1 push.
    ///
    /// **Cycle 1 substrate-only scope.** This field stores the X
    /// value; NO cast site mints it yet (Cycle 2/3 will add the
    /// admit-predicate widen + cast-pipeline binding for X-cost
    /// spells). Until that lands, every pre-substrate stack item
    /// pushes `None` here, so the field is parity-NEUTRAL — every
    /// cross-engine trajectory stays byte-identical. The resolve-
    /// arm consumers (`Effect::ZeroPointBallad`,
    /// `Effect::DevastatingOnslaught`) are also Cycle 3+ work.
    ///
    /// **Signed `i32`** (not `u32`): CR 107.1b allows X=0 but never
    /// negative on a cast, yet the SoT-parity signedness matches
    /// Python's `int` field shape and leaves room for future
    /// `Effect::*` variants that may treat X arithmetic (e.g. "X
    /// minus 1") natively. The cast-time bind always emits non-
    /// negative values; downstream arms can `clamp(0, ...)` or
    /// saturate as needed.
    ///
    /// **NOT in the per-StackItem fingerprint tuple** so pre-
    /// substrate stacks stay byte-equal — same discipline as
    /// `chosen_mode_idx` / `additional_cost_choice_id` /
    /// `chosen_mode_idxs`.
    pub x_value: Option<i32>,
}

impl StackItem {
    /// Convenience constructor for the spell shape. Every
    /// pre-Slice-1 call site uses this — the discriminant defaults
    /// to `Spell`, the payload to `None`.
    pub(crate) fn new_spell(source_obj_id: u32, controller: u8) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::Spell,
            trigger_payload: None,
            target_obj_id: None,
            target_obj_id_2: None,
            // ModalChooseOne envelope — Cycle 1. The bare untargeted
            // spell push doesn't go through the ChooseOneMode
            // detector (it sees no first-effect `ChooseOneMode`
            // variant); default `None` is back-compat-correct for
            // every pre-cycle non-modal spell.
            chosen_mode_idx: None,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1. Default `None` — the bare untargeted spell
            // push does not see the `additional_cost_choose_your_
            // creature` flag (the dedicated cast-pipeline detector
            // populates `Some(_)` only for the choose-your-creature
            // additional-cost shape, Cycle 2).
            additional_cost_choice_id: None,
            // **scour-for-scrap-slice Cycle 1 widen (2026-05-21).**
            // The multi-pick sibling is `None` here too — same
            // back-compat reasoning.
            chosen_mode_idxs: None,
            // **X-value subsystem Cycle 1 (2026-05-21).** The bare
            // untargeted spell push doesn't go through the X-cost
            // detector (Cycle 2/3 work); default `None` is back-
            // compat-correct for every pre-cycle non-X-cost spell.
            x_value: None,
        }
    }

    /// **Spell-card-bridge lever — Phase 1 (round-70) single-target
    /// + Phase 2 (round-76) 2-slot, unified.** Spell-shape
    /// constructor that carries up to TWO deterministic-first bound
    /// targets. Mirrors the `new_trigger`/`new_trigger_with_target`
    /// pair (CE20 Slice 4): the bare `new_spell` stays the
    /// back-compat untargeted path; this carries the cast-time
    /// chosen target(s) for the spell-cast pipeline.
    ///
    /// - **Single-target spells** (Phase-1 Tier-A — Radiant Strike,
    ///   Depressurize, Focus Fire, Biosynthic Burst, …) pass
    ///   `target_obj_id = Some(first_legal)`, `target_obj_id_2 =
    ///   None`. Byte-identical to the prior single-slot
    ///   `new_spell_with_target` (now subsumed — Phase 2 removed the
    ///   redundant single-slot constructor rather than keep a dead
    ///   zombie alongside the 2-slot superset; the no-fallback-hybrid
    ///   / clean-replacement discipline).
    /// - **2-slot heterogeneous-target spells** (Phase-2 Tier-A —
    ///   Diplomatic Relations: slot 0 = friendly pumped creature,
    ///   slot 1 = opponent's damaged creature) pass both `Some(_)`.
    ///
    /// Both ride into resolve via the
    /// `current_resolve_payload_target_id` /
    /// `current_resolve_payload_target_id_2` swaps.
    pub(crate) fn new_spell_with_targets(
        source_obj_id: u32,
        controller: u8,
        target_obj_id: Option<u32>,
        target_obj_id_2: Option<u32>,
    ) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::Spell,
            trigger_payload: None,
            target_obj_id,
            target_obj_id_2,
            // ModalChooseOne envelope — Cycle 1. Spells that bind
            // a modal pick at cast time use the dedicated
            // `new_spell_with_targets_mode_and_addcost` sibling
            // below; every existing target-binding caller (the
            // closed Tier-A single-target and Phase-2 2-slot arms)
            // is non-modal, so the default `None` is back-compat.
            chosen_mode_idx: None,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1. Default `None` — every pre-Cycle-1 caller
            // (Phase-1 single-target / Phase-2 2-slot) is on a
            // non-additional-cost-choice card, so the back-compat
            // default holds.
            additional_cost_choice_id: None,
            // scour-for-scrap-slice Cycle 1 widen — `None` for the
            // non-or_both target-binding pathway.
            chosen_mode_idxs: None,
            // X-value subsystem Cycle 1: `None` for every non-X-cost
            // pathway (every closed Phase-1 single-target / Phase-2
            // 2-slot caller is on a non-X-cost spell, so the back-
            // compat default holds).
            x_value: None,
        }
    }

    /// **ModalChooseOne envelope — Cycle 1 (cast-time mode binding,
    /// 2026-05-21) + Close Encounter additional-cost-choice envelope
    /// — Cycle 2 (cast-time choice binding, 2026-05-21) + X-value
    /// subsystem — Cycle 1 (cast-time X bind, 2026-05-21).** Spell-
    /// shape constructor that carries the cast-time-picked mode
    /// index PLUS the cast-time-bound additional-cost choice id PLUS
    /// the cast-time-bound X value. Mirrors `new_spell_with_targets`
    /// for the target-carrying superset; extends with the mode-index
    /// slot for modal spells, the additional-cost choice-id slot for
    /// spells whose printing carries
    /// `additional_cost_choose_your_creature = true`, AND the X-value
    /// slot for X-cost spells (Zero Point Ballad, Devastating
    /// Onslaught — Cycle 2/3 wires the cast-pipeline binding).
    ///
    /// Usage: `cast_spell_noncreature` detects each shape
    /// independently:
    /// - Modal detection (first effect is `Effect::ChooseOneMode`)
    ///   binds the deterministic-first mode pick (`Some(0)`,
    ///   matching Python's `_choose_modes_at_cast` no-agent fallback
    ///   at `actions.py:1222-1223`).
    /// - Additional-cost detection
    ///   (`printing.additional_cost_choose_your_creature`) scans the
    ///   caster's battlefield-creatures + warped-creature exile
    ///   cards for the deterministic-first eligible id (matching
    ///   Python's no-agent fallback at `actions.py:1737-1745`).
    /// - X-cost detection (Cycle 2/3 — NOT wired in Cycle 1) will
    ///   read `printing.mana_cost.x_pips > 0` and pick the
    ///   deterministic-first X via the pay-all-floating no-agent
    ///   fallback (mirrors Python's `actions._pick_x_cost_at_cast`).
    ///
    /// The fields are independent — every combinatorial subset is
    /// valid (a non-modal non-additional-cost non-X-cost spell sets
    /// all four to `None`, byte-equal to the pre-cycle
    /// `new_spell_with_targets` push).
    ///
    /// The bound mode is read at resolve time by the
    /// `Effect::ChooseOneMode` arm (already wired); the bound
    /// additional-cost choice is read by the
    /// `Effect::CloseEncounterEffect` arm; the bound X is read by
    /// future `Effect::ZeroPointBallad` /
    /// `Effect::DevastatingOnslaught` arms via the
    /// `state.current_resolve_x_value` transient swap.
    ///
    /// **Subsumption of the prior `new_spell_with_targets_mode_and_addcost`**
    /// (commit `17dc7d02` precedent — clean replacement, not a
    /// fallback hybrid, per the no-fallback-hybrid discipline).
    /// Cycle 1 has only one production call site (`state_ops_cast.rs`
    /// `cast_spell_noncreature`) and zero pre-existing test call
    /// sites, so the parameter addition is byte-clean with a single
    /// `None` literal at the call site.
    pub(crate) fn new_spell_with_targets_mode_addcost_and_x(
        source_obj_id: u32,
        controller: u8,
        target_obj_id: Option<u32>,
        target_obj_id_2: Option<u32>,
        chosen_mode_idx: Option<u8>,
        additional_cost_choice_id: Option<u32>,
        x_value: Option<i32>,
    ) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::Spell,
            trigger_payload: None,
            target_obj_id,
            target_obj_id_2,
            chosen_mode_idx,
            additional_cost_choice_id,
            // scour-for-scrap-slice Cycle 1 widen: the single-pick
            // constructor leaves the multi-pick sibling `None`; the
            // dedicated multi-pick constructor below takes the
            // inverse path (sets `chosen_mode_idxs`, leaves
            // `chosen_mode_idx` `None`).
            chosen_mode_idxs: None,
            x_value,
        }
    }

    /// **ModalChooseOne envelope — scour-for-scrap-slice (Cycle 1
    /// widen, 2026-05-21).** Spell-shape constructor that carries the
    /// cast-time-picked SUBSET of mode indices for a
    /// `ModalChooseOne(or_both=True)` spell (Scour for Scrap and any
    /// future Charm-shape).
    ///
    /// Mirrors `new_spell_with_targets_and_mode` but takes a vec of
    /// chosen mode indices instead of a single Option. The single-
    /// pick sibling stays in place for non-`or_both` modal spells;
    /// this constructor is opt-in via `cast_spell_noncreature`'s
    /// `or_both` detector (which sees `Effect::ChooseOneMode { ...,
    /// or_both: true }` and picks the all-modes default for the
    /// no-agent / deterministic-first fallback).
    ///
    /// **Mutual-exclusion contract.** The single-pick
    /// `chosen_mode_idx` is left `None` so the resolve arm reads the
    /// multi-pick slot first (its precedence rule is "multi-pick wins
    /// when set"); the legacy single-pick path is unaffected.
    /// `additional_cost_choice_id` is also `None` — `or_both` modal
    /// spells aren't on the additional-cost-choose-your-creature
    /// pathway (different cast-time detector branch).
    pub(crate) fn new_spell_with_targets_and_modes(
        source_obj_id: u32,
        controller: u8,
        target_obj_id: Option<u32>,
        target_obj_id_2: Option<u32>,
        chosen_mode_idxs: Option<Vec<u8>>,
    ) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::Spell,
            trigger_payload: None,
            target_obj_id,
            target_obj_id_2,
            // Mutual-exclusion: when the multi-pick path is active,
            // leave the single-pick slot `None`. The resolve arm
            // reads multi first; the legacy path is byte-identical
            // for every non-`or_both` modal.
            chosen_mode_idx: None,
            // `or_both` modal spells aren't on the additional-cost-
            // choose-your-creature pathway — leave `None`.
            additional_cost_choice_id: None,
            chosen_mode_idxs,
            // X-value subsystem Cycle 1: `or_both` modal spells
            // are not on the X-cost pathway (different cast-time
            // detector branch — X-cost reads `mc.x_pips > 0` and
            // modal reads `effects[0] is ChooseOneMode`); leave
            // `None`.
            x_value: None,
        }
    }

    /// Convenience constructor for the trigger shape (Slice 1).
    /// Used by `dispatch_event` / `push_triggered_ability` in
    /// `state_ops_resolve`. Pre-Slice-4 callers (every existing
    /// ETB / dies / cast / attacks dispatch site, plus the CE20
    /// Slice 1 activated-ability dispatch arm for non-target
    /// activations) leave `target_id` as `None` via this
    /// constructor; the Slice 4 target-carrying call site uses
    /// `new_trigger_with_target` below.
    pub(crate) fn new_trigger(
        source_obj_id: u32,
        controller: u8,
        effect: crate::cards::Effect,
        ability_index: u32,
    ) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::TriggeredAbility,
            trigger_payload: Some(TriggerPayload {
                effect,
                ability_index,
                target_id: None,
            }),
            // Spell-card-bridge lever (round-70): the spell-target
            // slot is Spell-kind-only; a trigger carries its target
            // on `trigger_payload.target_id` instead.
            target_obj_id: None,
            // Phase-2 (round-76): the 2-slot spell grammar is
            // Spell-kind-only.
            target_obj_id_2: None,
            // ModalChooseOne envelope — Cycle 1. The cast-time
            // mode-pick slot is Spell-kind-only. A trigger whose
            // body is itself an `Effect::ChooseOneMode` would
            // resolve via the resolve-time
            // `PendingDecision::ChooseModeChoice` fallback today
            // (the path that predates this slice); the cast-time
            // bind only fires on the spell-cast path per CR 601.2e.
            chosen_mode_idx: None,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1. Spell-kind-only — additional costs apply at
            // cast time per CR 601.2f, not when a trigger fires.
            additional_cost_choice_id: None,
            // scour-for-scrap-slice Cycle 1 widen: same Spell-kind-
            // only rationale — the multi-pick slot is `None` for
            // every trigger.
            chosen_mode_idxs: None,
            // X-value subsystem Cycle 1: same Spell-kind-only
            // rationale — X-cost binding happens on the spell-cast
            // pathway per CR 601.2f-h; triggers don't carry an X
            // value. `None` for every trigger.
            x_value: None,
        }
    }

    /// **CE20 follow-up Slice 4 (Equipment-equip).** Trigger-shape
    /// constructor for activated abilities that carry a chosen
    /// target (cost flags
    /// `requires_target_creature_you_control = true`). The
    /// `target_id` is read at resolve-time by the
    /// `Effect::EquipToTargetCreatureYouControl` arm. Non-equip
    /// targeted activations (future slices for targeted sacrifice,
    /// pump-target-creature activations, etc.) reuse this same
    /// constructor.
    pub(crate) fn new_trigger_with_target(
        source_obj_id: u32,
        controller: u8,
        effect: crate::cards::Effect,
        ability_index: u32,
        target_id: u32,
    ) -> Self {
        Self {
            source_obj_id,
            controller,
            kind: StackItemKind::TriggeredAbility,
            trigger_payload: Some(TriggerPayload {
                effect,
                ability_index,
                target_id: Some(target_id),
            }),
            // Spell-card-bridge lever (round-70): Spell-kind-only
            // slot — a trigger's target rides on
            // `trigger_payload.target_id`.
            target_obj_id: None,
            // Phase-2 (round-76): Spell-kind-only 2-slot grammar.
            target_obj_id_2: None,
            // ModalChooseOne envelope — Cycle 1: Spell-kind-only.
            chosen_mode_idx: None,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1: Spell-kind-only.
            additional_cost_choice_id: None,
            // scour-for-scrap-slice Cycle 1 widen: Spell-kind-only.
            chosen_mode_idxs: None,
            // X-value subsystem Cycle 1: Spell-kind-only.
            x_value: None,
        }
    }
}

/// A choice the engine is waiting on. Pushed onto
/// `GameState.pending_decisions` by an effect whose resolution
/// requires a controller-provided choice (e.g. Surveil — which of
/// the observed cards to send to graveyard, which to keep on top).
///
/// Round-27 G20 closure introduces the substrate. First variant:
/// `SurveilChoice` — the cards looked at, the controller, and the
/// source obj id (for log-correlation). The dispatcher
/// `submit_surveil_decision` consumes this variant; round-28 added
/// MayDo / PickCard / ChooseMode / IfYouDo / ManifestDread /
/// LookAtTopNRevealOrCounter variants; round-29 added
/// OptionalManaCost; round-30 G24 closure adds `ScryChoice` (Paris
/// scry-on-keep). All variants follow the same `apply_effect` →
/// push → pause / caller pops → submit shape.
///
/// **State-affecting**: a pending decision IS part of the gameplay-
/// relevant digest. A state with a pending decision differs from
/// one without (the observed cards are off-library-top but not yet
/// in graveyard; the next decision shapes the library + graveyard
/// final shape). `Clone` clones the Vec, so a cloned state carries
/// its pending decisions; CoW is not needed (the Vec is owned per
/// `GameState`, not shared across clones).
#[derive(Clone, Debug)]
pub enum PendingDecision {
    /// Surveil N (CR 701.43). The controller has just looked at
    /// `cards` (popped off library top, top-most first). For each
    /// one they choose either "top of library" (preserving the
    /// caller's submitted order) or "graveyard". Until the caller
    /// submits a decision the cards live in this variant — NOT in
    /// the library, NOT in the graveyard.
    ///
    /// `cards` carries the full `GameObject` payload (POD `Copy`),
    /// not just the id — the submit dispatcher needs the printing_id
    /// + controller + counters when re-pushing into library or
    /// graveyard. Order matches the order popped off library top:
    /// `cards[0]` was the topmost card, `cards[1]` was second-from-
    /// top, etc.
    SurveilChoice {
        source_obj_id: u32,
        controller: u8,
        cards: Vec<GameObject>,
    },
    /// **Round 28 may-do decision substrate.** "You may [inner_effect]"
    /// — controller chooses yes or no at resolve time. Pushed by
    /// `Effect::MayDo` (Lithobraking-style "you may sacrifice an
    /// artifact; if you do, ..." family, EoE pool ≥ 4 cards).
    ///
    /// **Payload.** Carries the inner Effect on the heap. On a "yes"
    /// submit, `submit_may_do_decision(true)` consumes the variant
    /// and recursively calls `apply_effect(state, &inner, source_obj_id)`.
    /// On a "no" submit (or auto-resolve with strategy `"may_do_yes"`),
    /// the variant is dropped without resolving the inner.
    ///
    /// **Python parity.** Matches Python's `MayDo(Effect)` — the
    /// default base-Agent answers "yes," so the auto-resolve strategy
    /// `"default"` / `"may_do_yes"` also says "yes" (preserves the
    /// no-agent-seated semantics).
    MayDoChoice {
        source_obj_id: u32,
        controller: u8,
        inner_effect: Box<crate::cards::Effect>,
    },
    /// **Round 28 pick-one-from-examined decision substrate.** "Look at
    /// the top N cards of your library; put ONE into your hand and the
    /// rest into your graveyard" — controller chooses WHICH examined
    /// card goes to hand. Pushed by
    /// `Effect::LookTopNPutOneInHandRestToGraveyard` (Commune-with-
    /// Evil family, 4 EoE cards).
    ///
    /// **Payload.** Carries the examined GameObjects fully (same shape
    /// as `SurveilChoice`) so the submit dispatcher can re-emit them
    /// with up-to-date zone/controller/counters without re-borrowing
    /// the library. Order matches popped-off-top order: `cards[0]` was
    /// topmost, `cards[1]` second, etc.
    ///
    /// **Submit.** `submit_pick_card_decision(chosen_id)` consumes the
    /// head, routes `chosen_id` to the controller's hand and every
    /// other examined card to the controller's graveyard (in popped
    /// order). `chosen_id` MUST be one of the examined ids; an unknown
    /// id raises `ValueError` transactionally.
    ///
    /// **Auto-resolve default.** `"default"` picks the topmost card
    /// (cards[0]) — matches Python's no-agent-seated fallback in
    /// `LookTopNPutOneInHandRestToGraveyard.resolve`.
    PickCardFromExaminedChoice {
        source_obj_id: u32,
        controller: u8,
        cards: Vec<GameObject>,
    },
    /// **Round 28 choose-one-mode decision substrate.** "Choose one —
    /// [mode_0]; [mode_1]; ..." Pushed by `Effect::ChooseOneMode`
    /// (Python's `ModalChooseOne` first-slice shape, 13 EoE cards).
    ///
    /// **Payload.** Carries every mode's Effect on the heap, indexed
    /// from 0. `submit_choose_mode_decision(chosen_idx)` consumes the
    /// head and applies `modes[chosen_idx]` via the standard
    /// `apply_effect` dispatcher, with the original `source_obj_id`
    /// propagated (so identity-coupled riders retain the resolving
    /// spell's identity).
    ///
    /// **First-slice scope.** Resolve-time mode picking only. Python's
    /// ModalChooseOne supports cast-time mode picks routed through
    /// `state.resolving_modes`, plus `or_both=True` ("choose one or
    /// both" oracle text), plus per-mode target picking. All deferred
    /// — this first slice exposes only the modal-choice substrate.
    ///
    /// **Auto-resolve default.** `"default"` / `"choose_mode_0"` pick
    /// `modes[0]` — matches Python's no-agent-seated fallback. Other
    /// variants' strategies fall through to the same default.
    ChooseModeChoice {
        source_obj_id: u32,
        controller: u8,
        modes: Vec<crate::cards::Effect>,
    },
    /// **Round 28 if-you-do decision substrate.** "[do_action]; if you
    /// do, [then_action]." First-slice form for "looter"-style cards:
    /// `IfYouDo(MayDo(DiscardSelf), DrawCards)` collapses to a single
    /// yes/no choice — yes applies both actions in order, no skips
    /// both.
    ///
    /// **Payload.** Carries both nested Effects on the heap. On a
    /// "yes" submit, the dispatcher applies `do_action` then
    /// `then_action` via `apply_effect_pub` with the same
    /// `source_obj_id` (so identity-coupled riders retain identity).
    /// On "no", neither is applied.
    ///
    /// **Why a separate variant from MayDoChoice.** A MayDo-with-then
    /// could be modeled as an extension of `MayDoChoice` carrying an
    /// optional `then_effect`, but a separate variant keeps the
    /// dispatcher surface honest: a card emits `Effect::IfYouDo`
    /// when oracle text says "...; if you do, ...", and emits
    /// `Effect::MayDo` for "you may ..." (no follow-up). Two
    /// variants, two dispatchers — matches the Python class split
    /// (`MayDo` vs `IfYouDo`).
    ///
    /// **First-slice scope.** Only IfYouDo where `do_action` is
    /// implicitly success-on-yes (i.e. IfYouDo(MayDo(X), Y)). General
    /// IfYouDo with arbitrary success-signaling do_actions (e.g.
    /// SacrificeACreature returning False on empty bf) is deferred —
    /// would need apply_effect to return a Result<bool>.
    IfYouDoChoice {
        source_obj_id: u32,
        controller: u8,
        do_action: Box<crate::cards::Effect>,
        then_action: Box<crate::cards::Effect>,
    },
    /// **Round 28 manifest-dread decision substrate.** CR 701.55 —
    /// "look at top two, manifest one, the other to graveyard."
    /// Pushed by `Effect::ManifestDread` (Bashful-Beastie family, 10
    /// EoE cards). Shape-aligned with `PickCardFromExaminedChoice`
    /// but routes the picked card to BATTLEFIELD (not hand) and the
    /// rest to graveyard.
    ///
    /// **Payload.** Carries the examined GameObjects fully (same
    /// shape as `PickCardFromExaminedChoice`) so the submit
    /// dispatcher can re-emit them into battlefield/graveyard
    /// without re-borrowing the library. Order matches popped-off-
    /// top order: `cards[0]` was topmost, `cards[1]` second, etc.
    /// With a 1-card library the Vec has length 1; with an empty
    /// library it's empty.
    ///
    /// **Submit.** `submit_manifest_dread_decision(chosen_id)`
    /// consumes the head, routes `chosen_id` to the controller's
    /// battlefield (with the standard ETB state reset:
    /// summoning_sick=true, tapped=false, damage=0, counters=0,
    /// attached_to=None, exiled_with_source=None) and every other
    /// examined card to the controller's graveyard (in popped
    /// order). `chosen_id` MUST be one of the examined ids; an
    /// unknown id raises `ValueError` transactionally.
    ///
    /// **Empty-library partial.** An empty `cards` Vec drains as an
    /// O(1) no-op on auto-resolve. Manual submit with any id raises
    /// (no ids in the pending set).
    ///
    /// **Auto-resolve default.** `"default"` / `"manifest_top"` pick
    /// the topmost card (cards[0]) — matches Python's no-agent-
    /// seated fallback in `ManifestDread.resolve`
    /// (`manifest_id = options[0]` before the agent override).
    ///
    /// **Face-down state — first-slice simplification.** The
    /// manifested card lands on bf with its ORIGINAL printing
    /// (name/types/P/T visible). Real MTG face-down semantics (CR
    /// 701.34 + 707) are deferred — see SEARCH_LIMITATIONS.md.
    ///
    /// **`then_attach_to_source` rider.** When `true`, after routing
    /// the chosen card to bf the submit dispatcher locates the source
    /// (by `source_obj_id`) on its controller's bf and stamps
    /// `source.attached_to = manifested_id`. Mirrors Python's
    /// `ManifestDreadThenAttachSelf` (Conductive-Machete-family
    /// Equipment ETB, freq 3). When `false` (default), the dispatcher
    /// runs the plain manifest-dread routing only — preserves the
    /// existing `Effect::ManifestDread` behavior bit-for-bit.
    ///
    /// **LTB during the decision window.** If the source has left the
    /// battlefield between resolve-push and submit (e.g. destroyed by
    /// a separate effect, returned to hand), the attach step is
    /// SILENT PARTIAL: the manifest still completes, but the attach
    /// is skipped (no panic, no error). Mirrors Python's
    /// `next((o for o in state.battlefield if o.id == source_id), None)`
    /// → `if source is None: return` fallback.
    ///
    /// **`then_plus_one_counter_on_manifested` + `counter_amount`
    /// rider.** When the bool flag is `true`, after routing the chosen
    /// card to bf the submit dispatcher locates the just-manifested
    /// permanent (by the captured `manifested_id`) and stamps
    /// `obj.counters += counter_amount`. Mirrors Python's
    /// `ManifestDreadThenPutPlusOneCountersOnIt` (Slimy-Aquarium /
    /// Weight-Room Room-door trigger family — counter_amount=1 and
    /// counter_amount=3 in the EoE pool). When `false` (default), the
    /// counter step is skipped — preserves the existing
    /// `Effect::ManifestDread` and `Effect::ManifestDreadThenAttachSelf`
    /// behaviors bit-for-bit. The two follow-up flags
    /// (`then_attach_to_source` and `then_plus_one_counter_on_manifested`)
    /// are independent — no Python card emits both today, but the
    /// substrate doesn't prevent it: if both are set, the dispatcher
    /// runs the attach then the counter add in that order.
    ///
    /// **Counter step no-op on empty manifest.** If the cards Vec was
    /// empty at resolve (library exhausted), no card was manifested
    /// and the counter step is naturally skipped — the dispatcher's
    /// helper-local `manifested_id` is captured AFTER the manifest
    /// routing succeeds, so an empty Vec means no counter target
    /// exists. Mirrors Python's `if not new_ids: return` early-exit.
    ManifestDreadChoice {
        source_obj_id: u32,
        controller: u8,
        cards: Vec<GameObject>,
        then_attach_to_source: bool,
        then_plus_one_counter_on_manifested: bool,
        counter_amount: i32,
    },
    /// **Round-28 look-at-top-N-reveal-or-counter decision substrate.**
    /// Pushed by `Effect::LookAtTopNRevealOrCounter` (Pulsar-Squadron-
    /// Ace / Living-Phone family, freq 2 in EoE pool). Shape-aligned
    /// with `PickCardFromExaminedChoice` but routes very differently:
    ///
    /// - On a "reveal yes" submit (with a chosen_id): the chosen card
    ///   goes to the controller's HAND; the rest go to the BOTTOM of
    ///   the controller's library (preserving popped order — see the
    ///   "random order" deferred note on `Effect::LookAtTopNRevealOrCounter`).
    /// - On a "reveal no" submit (no chosen_id needed): ALL examined
    ///   cards go to the bottom of the library. THEN, if
    ///   `counter_on_miss` is true and the source is on the
    ///   battlefield, a +1/+1 counter lands on the source.
    ///
    /// **Filter / max_power constraints.** Carried on the variant so
    /// the submit dispatcher can validate that `chosen_id` (when
    /// supplied) matches the filter. The filter check is intrinsic to
    /// the spell's "you may reveal a [filter_name] card" rider —
    /// submitting a non-matching id is a programmer error and raises
    /// `ValueError` transactionally.
    ///
    /// **Payload.** Carries the examined GameObjects (same shape as
    /// `PickCardFromExaminedChoice` / `ManifestDreadChoice`). Order
    /// matches popped-off-top order: `cards[0]` was topmost, `cards[1]`
    /// second, etc.
    ///
    /// **Submit.** `submit_look_at_top_n_decision(reveal, chosen_id)`
    /// consumes the head. `reveal=False` ignores `chosen_id` (callers
    /// pass `None`); `reveal=True` requires `chosen_id` to be one of
    /// the examined ids whose printing matches the filter.
    ///
    /// **Auto-resolve.** `"default"` / `"reveal_no"` route every
    /// examined card to the library bottom (no hand-route) and fire
    /// the `counter_on_miss` rider — matches Python's no-agent-seated
    /// fallback (the Python class only reveals when an agent is
    /// seated AND the agent picks something via
    /// `choose_card_from_library`).
    LookAtTopNRevealOrCounterChoice {
        source_obj_id: u32,
        controller: u8,
        cards: Vec<GameObject>,
        /// One of `"spacecraft"`, `"creature"`, `"artifact"`. Carried
        /// for the submit-time filter check.
        filter_name: String,
        /// Inclusive power cap for `"creature"` filter. 0 means
        /// "no cap" — matches Python's class default.
        max_power: i32,
        /// Whether to put a +1/+1 counter on source if no card was
        /// revealed (Pulsar Squadron Ace = true; Living Phone = false).
        counter_on_miss: bool,
    },
    /// **Round-29 optional-mana-cost decision substrate.** Pushed by
    /// `Effect::OptionalManaCostThenEffect` (Kavaron-Harrier family /
    /// Oreplate-Pangolin: "you may pay {cost}. If you do,
    /// [inner_effect]", freq 2 in EoE pool).
    ///
    /// **Payload.** Carries the mana cost AND the inner Effect on the
    /// heap. On a "yes" submit, `submit_optional_mana_cost_decision(true)`
    /// validates the controller can pay `cost` from their mana pool —
    /// if not, raises `ValueError` transactionally (no mutation). On
    /// success, the cost is deducted atomically before
    /// `apply_effect(state, &inner_effect, source_obj_id)` runs. On a
    /// "no" submit, the decision is dropped without paying or applying
    /// the inner effect.
    ///
    /// **Python parity.** Python's `OptionalManaCostThenEffect.resolve`
    /// is auto-resolving: it pays whenever the controller can afford
    /// the cost, otherwise the trigger does nothing. There is no
    /// agent-mediated choice in Python. The Rust default (auto-resolve
    /// `"default"`) therefore answers "yes if affordable, else no" —
    /// the same observable behavior as Python. The substrate exposes
    /// the choice explicitly so agents CAN override (e.g. "skip paying
    /// even when affordable" for value preservation), but the
    /// no-agent-seated default tracks Python exactly.
    ///
    /// **Auto-resolve strategies.** `"optional_pay_yes"` attempts to
    /// pay (raising ValueError if unaffordable — programmer error to
    /// force-yes when broke); `"optional_pay_no"` drops the decision;
    /// `"default"` does the Python-parity affordability-check.
    OptionalManaCostChoice {
        source_obj_id: u32,
        controller: u8,
        cost: crate::cards::ManaCost,
        inner_effect: Box<crate::cards::Effect>,
    },
    /// **Round-30 G24 closure — Paris scry-on-keep decision substrate.**
    /// CR 103.4 — after keeping a mulliganed hand the controller scrys 1
    /// (looks at the top card of their library and either keeps it on
    /// top or sends it to the bottom). Pushed by `GameState.keep_hand`
    /// when the player's `mulligan_count > 0`.
    ///
    /// **Shape vs `SurveilChoice`.** Scry-1 is functionally a "look at
    /// top 1, route to library-top or library-bottom" — same look-at-top
    /// + per-card destination flow as Surveil, but the route choices
    /// differ (library-top vs library-bottom rather than library-top vs
    /// graveyard). A separate variant keeps the dispatcher surface
    /// honest: `submit_surveil_decision` routes to library/graveyard;
    /// `submit_scry_decision` routes to library-top/library-bottom.
    ///
    /// **Payload.** A single `card` (the top card of the controller's
    /// library, popped off pre-push). Carries the full `GameObject` so
    /// the submit dispatcher can re-emit it into library-top or library-
    /// bottom without re-borrowing the source library. Stored as a
    /// `Vec<GameObject>` of length 1 to mirror the other examined-cards
    /// variants' shape (`SurveilChoice` etc.) — keeps the fingerprint
    /// + dict serialization shape uniform across variants.
    ///
    /// **Submit.** `submit_scry_decision(keep_on_top: bool)` consumes
    /// the head. `keep_on_top=true` pushes the card back to library top
    /// (lib[0]); `keep_on_top=false` pushes the card to the library
    /// bottom (tail of the Vec).
    ///
    /// **Auto-resolve.** `"default"` / `"scry_keep_top"` route the card
    /// back to library top (matches Python's no-agent-seated default —
    /// keeping the top card is the conservative non-decision). The
    /// `"scry_send_bottom"` strategy routes to the bottom.
    ///
    /// **Empty-library partial.** An empty `cards` Vec at push time
    /// means the library was empty when `keep_hand` ran — the variant
    /// is pushed-and-immediately-equivalent-to-empty on auto-resolve
    /// (no card to route). Manual submit with an empty Vec is a no-op
    /// that consumes the head. In practice the caller has just drawn
    /// 7 cards via `mulligan_atomic`, so a 7-deep library at this point
    /// has the top card available; the empty path is defensive.
    ScryChoice {
        source_obj_id: u32,
        controller: u8,
        cards: Vec<GameObject>,
    },
    /// **20th Rust slice (VVV Pin 2 closure) — Command Bridge cost-
    /// decision substrate.** "Sacrifice this unless you tap an
    /// untapped permanent you control" — controller chooses ONE
    /// permanent from the pre-built candidate set to tap (and the
    /// source survives) OR declines (and the source is sacrificed).
    /// Pushed by `Effect::SacrificeSelfUnlessTapPermanent` (Command
    /// Bridge ETB, 24,522 17lands decks — the highest deck-count
    /// card in the BLOCKED-genuinely_missing tier).
    ///
    /// **Payload.** Carries the source id (so the dispatcher can
    /// locate the source to sacrifice if the controller declines)
    /// and the candidate ids (untapped permanents controlled by the
    /// source's controller, source excluded — built at push time by
    /// the resolve arm). Mirrors `SurveilChoice`'s flat `Vec<u32>`
    /// idiom for the choice set rather than the heavier
    /// `Vec<GameObject>` used by `PickCardFromExaminedChoice` —
    /// candidates already live on the battlefield, so re-locating
    /// them by id at submit time is the same pattern as
    /// `submit_pick_card_decision`'s id-based dispatch.
    ///
    /// **Submit.** `submit_tap_permanent_decision(Some(chosen_id))`
    /// taps the chosen permanent (the source stays); `None` declines
    /// → sacrifice the source. `chosen_id` MUST be one of the
    /// candidate ids; an unknown id raises `ValueError`
    /// transactionally (no mutation on failure).
    ///
    /// **Auto-resolve default.** `"default"` declines (matches
    /// Python's no-agent-seated fallback at `combinators.py:139-141`
    /// — `chosen = None` when `p.agent is None`, leading to sacrifice
    /// via the `_sacrifice(state, source)` decline branch at
    /// `combinators.py:147`). Other strategies fall through to the
    /// same decline-default for robustness.
    ///
    /// **Empty candidate set is impossible.** The resolve arm only
    /// pushes this decision when candidates is non-empty (the
    /// empty-set case sacrifices the source directly without ever
    /// queuing a decision — Python parity, `combinators.py:130-136`).
    TapPermanentChoice {
        source_obj_id: u32,
        controller: u8,
        candidates: Vec<u32>,
    },
    /// **Aura-attachment substrate — Phase 1 STEP 3 (round-78,
    /// 2026-05-20).** Cast-time enchant-target choice for an Aura
    /// spell. CR 303.4 / 601.2c — when a player casts an Aura they
    /// MUST choose a legal enchant target as part of casting. Pushed
    /// by `cast_aura` after the mana cost is paid and the spell is
    /// rehomed onto the stack-store; the caster's agent drains via
    /// `submit_enchant_target_decision(Some(target_id))` (binds the
    /// chosen target onto the StackItem's `target_obj_id`) or
    /// `submit_enchant_target_decision(None)` (no target — the Aura
    /// will fizzle on resolve per STEP 4 / Python parity at
    /// `actions.py:2740-2745`).
    ///
    /// **Payload.** Carries the source id (the Aura on the stack —
    /// `source_obj_id` is the same id `cast_aura` pushed onto
    /// `state.stack` as the StackItem's source), the controller (who
    /// makes the choice — the caster), and the candidate ids (every
    /// legal enchant target on EITHER battlefield, computed at push
    /// time by `cast_aura` per the printing's `aura_enchant_type` +
    /// `aura_yours_only` filter; same shape as `TapPermanentChoice`'s
    /// flat `Vec<u32>` idiom).
    ///
    /// **Submit.** `submit_enchant_target_decision(Some(chosen_id))`
    /// writes the chosen target onto the matching StackItem's
    /// `target_obj_id` so STEP 4's resolve arm can read it and attach
    /// the Aura. `chosen_id` MUST be one of the candidate ids (an
    /// unknown id raises `ValueError` transactionally — no mutation
    /// on failure). `submit_enchant_target_decision(None)` is the
    /// explicit "no legal target was chosen" path — the StackItem's
    /// `target_obj_id` stays `None` so STEP 4 fizzles on resolve.
    ///
    /// **Auto-resolve default.** `"default"` / `"enchant_target_first"`
    /// picks `candidates[0]` (the deterministic-first legal host in
    /// `bf_iter_etb_sorted` order — mirrors the
    /// `_PickFirstLegalTargetAgent` bench/probe convention every
    /// other cast-time target picker uses, e.g. Radiant Strike's
    /// destroy half + Focus Fire + Seedship Impact at
    /// `state_ops_cast.rs:1614-1714`).
    ///
    /// **Empty candidate set is impossible.** The cast arm only
    /// pushes this decision when the candidate set is non-empty; the
    /// empty-set case skips the push entirely (no decision queued —
    /// the StackItem rides with `target_obj_id = None` and STEP 4
    /// fizzles on resolve, matching Python's
    /// `actions.py:1354 if legal:` short-circuit). Without this guard
    /// the caster would face an "uncancellable" decision with no
    /// legal id to submit.
    EnchantTargetChoice {
        source_obj_id: u32,
        controller: u8,
        candidates: Vec<u32>,
    },
}

/// `GameState` carries an `Arc<RegistryData>` so every clone shares
/// the printings catalogue at refcount-bump cost. Per-instance
/// state — players, zones, turn counter — clones via the `Player`
/// struct's CoW zones (see `Zone` doc).
///
/// **Clone is implemented manually** (rather than `#[derive(Clone)]`)
/// because the `rng: Option<Py<PyAny>>` field cannot use the pyo3
/// `py-clone` feature without taking on the global "panic when GIL not
/// held" footgun. The manual impl explicitly acquires the GIL via
/// `Python::with_gil` and calls `clone_ref` on the stashed rng, which
/// just bumps the Python refcount. Every other field uses its standard
/// `Clone` implementation.
#[derive(Debug)]
#[pyclass(name = "GameState")]
pub struct GameState {
    pub players: [Player; 2],
    pub registry: Arc<RegistryData>,
    pub active_player: u8,
    pub priority_player: u8,
    pub turn_number: u32,
    pub phase: u8,
    /// Combat: ids of creatures currently declared as attackers.
    pub attackers: Vec<u32>,
    /// Combat: (attacker_id, blocker_id) pairs declared this combat.
    /// One attacker can be blocked by multiple blockers (multi-block);
    /// represented as multiple entries with the same attacker_id.
    pub blockers: Vec<(u32, u32)>,
    /// The stack — top is the LAST element of the Vec for O(1) pop.
    pub stack: Vec<StackItem>,
    /// **G7 Slice S5a — the spell-stack-source-zone store (Option β).**
    /// While a spell (or other resolve-via-`move_source_post_resolve`
    /// entity) is on the stack, its owned `GameObject` lives HERE with
    /// `obj.zone = ZONE_STACK` — NOT in the owner's `hand` Vec.
    ///
    /// **Why a centralized store, not a per-player zone.** Python's
    /// `StackItem` owns the card by reference (`actions.py:1759-1767`
    /// `StackItem(source=card, …)`) — the card is `p.hand.remove`'d at
    /// cast (`actions.py:1403`) and `card.zone = Zone.STACK`
    /// (`actions.py:1751`); it lives in NO zone list, reachable only
    /// via `state.stack[i].source` and read directly at resolve
    /// (`card = item.source`, `actions.py:2708`). Rust keeps `StackItem`
    /// id-only (no `state.rs` StackItem-shape / fingerprint-slot churn —
    /// the design's Option β minimum-blast-radius choice); this Vec is
    /// the place Rust holds the owned object so the id-keyed source
    /// finders still reach it. It is centralized (one per `GameState`,
    /// not per-`Player`) because Python's stack is centralized on
    /// `GameState` and a stack item's source has no per-player zone
    /// while it is on the stack.
    ///
    /// **Excluded from every hand-derived projection by construction.**
    /// `hand_snapshot` / the fingerprint's per-player hand slot / the
    /// determinizer pool / `set_object_printing_id` / the 3
    /// `legal.rs` `players[idx].hand.iter()` cast-emit loops all read
    /// `players[idx].hand` (or the 5 per-player zones) and NEVER this
    /// store — so the on-stack spell vanishes from all of them exactly
    /// as Python's `p.hand.remove(card)` removes it from every
    /// hand-derived Python projection. NOT serialized as its own
    /// fingerprint/snapshot zone (Python has no separate stack-object
    /// zone signature either — the card's stack presence is represented
    /// solely via the `StackItem`/`stack_snapshot` `source_obj_id`
    /// slot, which is byte-unchanged). The 3 source-resolution walks
    /// (`find_source_printing_id` / `take_source` /
    /// `move_source_post_resolve`) and the `dispatch_event`
    /// CAST/DIES/LEAVES observer walk are extended to find a source
    /// here as an additional zone; `take_source` removes from here on
    /// resolve and `move_source_post_resolve` pushes the owned object
    /// to its post-resolve destination (bf / graveyard) byte-equal to
    /// Python's `put_into_battlefield`.
    pub stack_objects: Vec<GameObject>,
    /// **G7 Class-C delayed-trigger port (round-78).** Pending delayed
    /// triggered abilities (CR 603.7), each scheduled at the time
    /// some earlier resolve/cast committed to firing a trigger at a
    /// future end step. Mirrors Python's `GameState.delayed_triggers:
    /// list[dict]` (`src/mtg_bot/game.py:2629`). Each entry carries
    /// `fire_at_turn`, `controller_id`, `source_id`, and the
    /// `Effect` to push.
    ///
    /// **Scanned at START of every END step** by
    /// `state_ops_steps.rs::run_delayed_end_step_triggers` — every
    /// entry whose `fire_at_turn == state.turn_number` and
    /// `controller_id == state.active_player` is converted into a
    /// real triggered ability on the stack via
    /// `push_triggered_ability(source_id, controller_id,
    /// ability_index=0, effect, once_per_turn=false)` and removed
    /// from the Vec. Non-matching entries (e.g. an opponent's-turn
    /// warp scheduled for the next end step) ride.
    ///
    /// **First Rust producer:** `cast_creature_warp` schedules
    /// `WarpExileSelf` here at warp-cast time (replacing the pre-port
    /// stamp-`warp_exile_pending`-and-sweep-at-end-step path). The
    /// Vec is generalized to the full `(fire_at_turn, controller_id,
    /// source_id, effect)` shape Python uses so additional Python
    /// delayed-trigger producers (`SacrificeSourceDelayed`,
    /// `RemoveChargeCountersFromSourceDelayed`,
    /// `ReturnExiledPermanentToOwnerTapped`) can be ported in their
    /// own slices without re-shaping this field.
    ///
    /// **Serialization / fingerprint.** Empty by default (almost
    /// always for non-warp game states); ride the canonical Clone
    /// path. NOT in the per-object fingerprint tuple. The fingerprint
    /// emits the Vec's contents only when non-empty (back-compat:
    /// pre-port baselines pinned with an empty Vec stay byte-equal),
    /// mirroring `pending_decisions`'s "serialize only when
    /// non-empty" shape.
    pub delayed_triggers: Vec<crate::cards::DelayedTrigger>,
    /// Count of consecutive `pass_priority` invocations since the last
    /// non-pass mutation. Round-27 G3 closure: needed by the
    /// `complete_step_via_priority_round` composer to detect "both
    /// players have passed in succession" (CR 117.4) without callers
    /// having to track the count themselves. Reset by any non-pass
    /// state-mutating action (`cast_creature`, `apply_action` for
    /// non-`pass_priority` arms, `advance_phase`, etc.) — see the
    /// reset call sites in `state_ops_cast.rs`, `state_ops_actions.rs`,
    /// and `state_ops_steps.rs`.
    ///
    /// NOT serialized into the fingerprint — like `mulligan_count`,
    /// this is a transient bookkeeping counter, not part of the
    /// gameplay-relevant state digest. Same observable game state
    /// pre- and post-pass must produce the same fingerprint regardless
    /// of how many consecutive passes happened to get there.
    pub consecutive_passes: u8,
    /// EoE "Void" condition tracking — first half. True if any
    /// nonland permanent left the battlefield this turn (cleared at
    /// turn boundary). Mirrors Python's
    /// `GameState.permanent_left_battlefield_this_turn`. Substrate
    /// only in this slice: no Rust call sites set this yet, so it
    /// remains `false` everywhere — the `Effect::ConditionalIfVoid`
    /// resolver reads it but the predicate will always evaluate to
    /// false until the LTB-tracking and warp-tracking call sites
    /// are wired (deferred until those mechanics land). Stays out of
    /// the fingerprint (transient turn-scoped flag; same observable
    /// state pre- and post-turn produces the same digest).
    pub permanent_left_battlefield_this_turn: bool,
    /// EoE "Void" condition tracking — second half. True if any
    /// spell was warped this turn. Mirrors Python's
    /// `GameState.spell_warped_this_turn`. Same substrate-only
    /// status as the LTB flag above.
    pub spell_warped_this_turn: bool,
    /// **Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    /// substrate landing, 2026-05-20).** Per-player "this player has
    /// created one or more tokens this turn" flag, cleared at the
    /// turn boundary alongside the other turn-scoped flags. Mirrors
    /// Python's `GameState.players_who_created_tokens_this_turn`
    /// (`game.py:2721`) — Python uses a `set[int]` of player ids
    /// whose membership is the boolean read at the Moonlit
    /// replacement check; Rust mirrors that as a 2-slot bool array
    /// keyed by `player_idx` (CR 102.1 — exactly two players). The
    /// `Effect::CreateToken` arm flips the controller's slot to
    /// `true` after a non-zero-count token batch creates at least one
    /// token (mirrors Python's `state.players_who_created_tokens_
    /// this_turn.add(recipient)` at `tokens.py:294`); the turn-end
    /// cleanup sweep zeroes both slots (mirrors Python's
    /// `state.players_who_created_tokens_this_turn.clear()` at
    /// `turn.py:1092`).
    ///
    /// **Substrate-only this slice.** The Moonlit-replacement
    /// consumer (the `_find_moonlit_replacement_source` check inside
    /// `CreateToken.resolve` at `tokens.py:217-247`) is NOT yet wired
    /// on the Rust side — Rust's `CreateToken` arm sets the flag but
    /// does NOT swap the token's `printing_id` to the host's printing.
    /// The flag is byte-equal-trackable cross-engine so a future
    /// slice can light up the consumer with the per-player turn
    /// state already in place. Stays out of the fingerprint
    /// (transient turn-scoped flag; same observable state pre- and
    /// post-turn produces the same digest, mirroring the
    /// `permanent_left_battlefield_this_turn` / `spell_warped_this_
    /// turn` pattern).
    pub created_tokens_this_turn: [bool; 2],
    /// Decisions the engine is waiting on. Round-27 G20 closure —
    /// effects whose resolution needs a controller-provided choice
    /// (Surveil, future London bottom-N, future scry-on-keep,
    /// future modal-choose) push a `PendingDecision` here and
    /// short-circuit their `apply_effect` arm. The caller (MCTS,
    /// parity_smoke, test harness) inspects via
    /// `pending_decisions()` / `has_pending_decision()`, resolves
    /// via the variant-specific `submit_*_decision` dispatcher
    /// (e.g. `submit_surveil_decision`).
    ///
    /// **State-affecting** (mirrors Python's pause-and-resume
    /// decision shape — see `mtg_bot.search.resolution_channel` /
    /// `ResolutionSurveilDecision`): a pending decision is part of
    /// the gameplay-relevant digest. The fingerprint serializes
    /// this slot ONLY WHEN non-empty (so existing baselines with
    /// empty queues stay bit-identical). FIFO: the dispatcher
    /// consumes index 0 first.
    pub pending_decisions: Vec<PendingDecision>,
    /// **CE4 closure — monotonic ETB-timestamp source.** Bookkeeping
    /// counter incremented once per battlefield-entry primitive
    /// (move_from_hand_to_battlefield / play_land / every ETB pathway
    /// in state_ops_resolve / token / manifest / reanimation). Each
    /// call site reads the current value via
    /// `stamp_etb_timestamp(&mut state, &mut obj)` (see
    /// `state_ops_zones.rs`), which writes the counter into the
    /// object's `etb_timestamp` and bumps the counter by 1 (saturating
    /// — u32 is wide enough that a realistic game won't overflow, but
    /// a saturating bump on the corner case still produces a sortable
    /// tail-state).
    ///
    /// **Not serialized into the fingerprint.** Like `mulligan_count`
    /// and `consecutive_passes`, this is a transient bookkeeping
    /// counter, not part of the gameplay-relevant digest. Two states
    /// reached via different histories that share the same observable
    /// battlefield ORDER must share a fingerprint regardless of how
    /// many bf-entries happened along the way. The per-object
    /// `etb_timestamp` slot is also kept out of the 8-tuple to
    /// preserve pre-CE4 pinned baselines; the fingerprint reads ETB
    /// order via a stable-sort by `etb_timestamp` at battlefield-walk
    /// time rather than emitting the field itself.
    pub next_etb_timestamp: u32,
    /// **R76-1 closure — monotonic object-id source (CR 111.7).**
    /// Bookkeeping counter incremented once per object minted after
    /// game start (every token-mint site in `state_ops_resolve.rs` —
    /// `Effect::CreateToken` / the each-player / tapped riders /
    /// Seedship Impact / Orbital Plunge Lander riders / the
    /// copy-token arm — plus the `load_deck` /
    /// `register_and_inject_object` setup-injection primitives, which
    /// keep this counter in lockstep with the initial population).
    /// Each mint site reads the current value via
    /// `reserve_object_id(&mut state)` (saturating bump, same shape
    /// as `reserve_etb_timestamp`).
    ///
    /// **Why it exists (the R76-1 bug).** The pre-R76-1 token-mint
    /// arms allocated a new id by scanning every live zone on both
    /// players for the current `max_id` and using `max_id + 1`. That
    /// scan is **non-monotonic**: a token that has ceased to exist
    /// (CR 111.7 — it left the battlefield) no longer contributes to
    /// `max_id`, so a later mint re-derives a value at-or-below the
    /// freed id and **reuses it**, violating the object-identity
    /// invariant (each object created during a game gets a NEW
    /// unique id). Python's `GameState.new_object`
    /// (`src/mtg_bot/game.py:2787-2798`) is the source of truth: a
    /// strict monotonic `self.next_object_id += 1` that never reuses
    /// a freed id. This counter mirrors Python's exactly so token
    /// ids stay byte-equal across engines for the whole game.
    ///
    /// **Seeded at game start** to the number of objects the setup
    /// primitive created (`make_state`'s post-build `next_id`, =
    /// 1 + max initial id = exactly Python's `next_object_id` after
    /// its `new_object` setup loop). The `load_deck` /
    /// `register_and_inject_object` test/setup primitives advance it
    /// past every object they inject so a mid-game mint that follows
    /// an injection still draws a fresh id.
    ///
    /// **Persisted across clones** (CR 111.7 is a game-lifetime
    /// invariant — a determinized clone must keep allocating ids
    /// above every id the parent ever used, never recycle). Like
    /// `next_etb_timestamp`, it is a transient bookkeeping counter
    /// and is **not serialized into the fingerprint** — the per-
    /// object `id` slot already carries the observable identity; two
    /// states that share the same observable object ids must share a
    /// fingerprint regardless of how many mints happened along the
    /// way.
    pub next_object_id: u32,
    /// **Round 28 freq-2 closure — resolving combat-damage amount.**
    /// Bookkeeping slot read by
    /// `Effect::DrawCardsEqualToResolvingCombatDamage` (Starwinder /
    /// Fear-of-Failed-Tests family) at trigger-resolve time to find
    /// out "that many cards" — N is the damage amount the resolving
    /// combat-damage trigger was fired with. Mirrors Python's
    /// `GameState.resolving_combat_damage_amount` (game.py:2666).
    ///
    /// **Production write site (I32).**
    /// `state_ops_combat.rs::GameState::apply_damage_pass` writes the
    /// slot per-PlanEntry — set to `entry.amount` before the per-entry
    /// damage / deathtouch / lifelink mutations fire, reset to 0
    /// after. Mirrors Python's per-event set/reset boundary
    /// (`combat.py:618,623` granted dealt-damage trigger resolve;
    /// `actions.py:2684,2699` stack-pushed trigger resolve). The
    /// `set_resolving_combat_damage_amount` setter remains available
    /// for harness / cross-engine tests that emulate the apply-loop
    /// write outside a real combat flow. See RUST_PARITY_GAPS.md I32.
    ///
    /// Not serialized into the fingerprint — like other transient
    /// resolve-time slots, the same observable game state pre- and
    /// post-resolve must produce the same digest regardless of the
    /// intermediate amount.
    pub resolving_combat_damage_amount: i32,
    /// **EEE downstream complement — 8th Rust slice (JJJ Pin 2).**
    /// The player id (0 or 1) that was the TARGET of the most recently
    /// dispatched combat-damage-to-player event.  Written by
    /// `dispatch_combat_damage_to_player` before calling
    /// `dispatch_event(DealsDamage, source_id)` and reset to `u8::MAX`
    /// (sentinel "no active target") after the dispatch returns.
    ///
    /// Read by `Effect::PutPoisonCountersOnTargetPlayer.resolve` to find
    /// which player receives the poison counters.  Matches Python's
    /// convention where `_fire_combat_damage_to_player_triggers` passes
    /// `target=target_player_obj` to `dispatch_event` and the triggered
    /// ability's effect reads `targets[0]` at resolve time.
    ///
    /// Initialized to `u8::MAX` (= "no active target") at game start.
    /// Not serialized into the fingerprint (transient resolve-time slot).
    pub resolving_combat_damage_target_player_id: u8,
    /// **CE12 closure — sticky winner_id shadow.** Mirrors Python's
    /// `GameState.winner_id`: a write-once `Option<u8>` field set by
    /// `apply_sbas` the FIRST time the SBA pass observes "exactly one
    /// player has `has_lost = True` AND `winner_id_sticky.is_none()`",
    /// and never modified thereafter. Once set, subsequent reads of
    /// the `winner_id` getter return this value even if BOTH players
    /// have `has_lost = True` later (bilateral kill / mutual KO after
    /// a prior sole-survivor moment).
    ///
    /// **Read path.** The `winner_id` getter returns
    /// `winner_id_sticky` if it is `Some(_)`; otherwise it falls back
    /// to the live derivation from the two `has_lost` flags (CE10's
    /// post-closure shape: `(false, true) -> Some(0)`,
    /// `(true, false) -> Some(1)`, otherwise `None`). The fallback
    /// preserves "game not over yet" (both alive) and bilateral
    /// simultaneous-loss-without-prior-write (both lost in the same
    /// SBA pass with no sole-survivor moment in between) — those
    /// shapes still return `None`.
    ///
    /// **Write path.** Inside `apply_sbas`, after the per-player
    /// loss-flag commit pass, we check whether exactly one player now
    /// has `has_lost = True`. If yes AND `winner_id_sticky` is still
    /// `None`, we write `winner_id_sticky = Some(survivor_id)`. The
    /// gate on `is_none()` enforces write-once: once written, never
    /// modified — matches Python's `winner_id is None` gate at
    /// `turn.py:1389`.
    ///
    /// **`is_draw` semantics under sticky.** Independent of
    /// `winner_id_sticky` — `is_draw` is True iff both players have
    /// `has_lost = True`. Mirrors Python: the SBA writer at
    /// `turn.py:1384-1386` gates ONLY on `not state.is_draw`, NOT on
    /// `winner_id is None`. So a sequential-kills shape — first
    /// player dies (SBA writes winner_id), then later both die (SBA
    /// writes is_draw) — ends up with BOTH fields populated:
    /// `winner_id == 0 AND is_draw == True`. This is a valid Python
    /// post-state and Rust mirrors it.
    ///
    /// **Not serialized into the fingerprint.** Like `mulligan_count`,
    /// `consecutive_passes`, and `next_etb_timestamp`, this is
    /// bookkeeping state for the SBA-gated outcome API, not part of
    /// the gameplay-relevant state digest. The derived `winner_id` /
    /// `is_draw` getters ARE part of the field-view 22-tuple that the
    /// cross-engine smoke harness compares, so the OBSERVABLE behavior
    /// is sticky. But two states with the same `has_lost` flags and
    /// the same observable game shape MUST produce the same
    /// fingerprint regardless of whether one path went through a
    /// sole-survivor SBA pass and another didn't — pinned baselines
    /// stay bit-equal.
    pub winner_id_sticky: Option<u8>,
    /// **Effect-rng-threading closure (G19 follow-up) — caller-owned
    /// external rng for in-Effect shuffles.** Optional handle to a
    /// Python `random.Random` instance set by the caller via
    /// `set_external_rng`. When `Some(_)`, the Effect resolve arms
    /// that consume randomness (today only
    /// `Effect::SearchLibraryForBasicLandTapped` post-search Fisher-
    /// Yates per CR 701.19 / 701.22) read draws from this stream via
    /// `crate::external_rng::next_below`, producing byte-identical
    /// permutations to Python's `random.Random.shuffle(list)`. When
    /// `None` (default), the legacy xorshift64 fallback runs unchanged.
    ///
    /// **Caller-owned bookkeeping — not part of gameplay-relevant state.**
    /// This field is NOT serialized into the fingerprint (see I43 in
    /// `docs/RUST_PARITY_GAPS.md`) — two states with the same
    /// gameplay-observable shape must produce the same digest regardless
    /// of whether/which rng is stashed. Production callers (cli, bench)
    /// leave this `None` and stay byte-equivalent to pre-slice behavior;
    /// only the parity harness and cross-engine smoke tests opt in via
    /// `set_external_rng`.
    ///
    /// **Clone semantics.** `clone_ref`'d through the manual `Clone`
    /// impl on `GameState` — two clones share the SAME `random.Random`
    /// instance (Python refcount bump). For the cross-engine smoke use
    /// case this is correct (one rng per game). MCTS-style search-tree
    /// forking that depends on independent rng streams must call
    /// `set_external_rng(None)` on the forked branches.
    pub rng: Option<Py<PyAny>>,
    /// **ETB-trigger substrate (Slice 1) — once-per-turn ledger.**
    /// Set of `(source_obj_id, ability_index)` keys for triggered
    /// abilities flagged `once_per_turn = true` that have already
    /// fired this turn. Cleared at cleanup step (the per-turn
    /// reset is the conceptual analog of Python's
    /// `state.once_per_turn_fired` — see Python's `turn.py:should_fire_trigger`
    /// and `actions.py:3172-3280`).
    ///
    /// **Slice 1 scope.** Substrate only — the ledger is checked
    /// by `dispatch_event` and updated by `push_triggered_ability`,
    /// but Slice 1's smoke probe doesn't exercise the gate (the
    /// canonical IsSelf-enters trigger fires at most once per
    /// permanent's life anyway, so the once-per-turn gate is moot).
    /// Slice 2+ fans out to sites that actually need the gate
    /// (Spacecraft tier-stamp triggers, repeat-fire shapes).
    ///
    /// **Not serialized into the fingerprint.** Like
    /// `mulligan_count`, `consecutive_passes`, `next_etb_timestamp`,
    /// `winner_id_sticky` — this is bookkeeping state, not
    /// gameplay-observable. Two states reached via different
    /// histories that share the same observable battlefield /
    /// stack / hand / library / graveyard / life MUST produce the
    /// same fingerprint regardless of which once-per-turn triggers
    /// have or haven't fired (those would only matter if a
    /// subsequent trigger ATTEMPT fired — and that attempt is
    /// part of the gameplay shape itself).
    pub triggered_this_turn: std::collections::HashSet<(u32, u32)>,
    /// **ETB-trigger substrate (Slice 2) — deferred dispatch queue.**
    /// Effect arms inside `apply_effect` that move permanents onto
    /// the battlefield (RS2-RS12 in `docs/RUST_ETB_TRIGGER_AUDIT.md`)
    /// don't call `dispatch_event` directly — instead they push the
    /// entered permanent's id here, and the surrounding caller
    /// (currently `resolve_top_of_stack`, `play_land`, `cast_creature`,
    /// `move_object`, `move_object_from_graveyard_to_battlefield`,
    /// `register_and_inject_object`, `return_paired_exiles_for`, and
    /// `resolve_manifest_dread_to_battlefield`) drains the queue after
    /// the resolving slot is removed.
    ///
    /// **Why deferred?** Effect arms can mutate the stack (e.g.
    /// `Effect::CounterTargetSpell` removes a stack item; future
    /// trigger-from-trigger paths may push more triggers). Dispatching
    /// inside `apply_effect` would compete with the resolver's
    /// "pop the resolving slot" step — the resolver assumes the
    /// resolving slot is at `stack.last()` at pop time, and an
    /// inline dispatch breaks that invariant by piling a fresh
    /// trigger on top of the resolving slot. Deferring the
    /// dispatch keeps the stack untouched during `apply_effect`
    /// and gives the resolver a clean pop.
    ///
    /// **Drain semantics.** The caller drains by `std::mem::take`
    /// + iterating, calling `dispatch_event(Event::Enters, id)`
    /// per id in push order. Order matches Python's centralized
    /// `state.battlefield` walk (the same ETB-stamp order produced
    /// by the Effect arm). FIFO drain — first entered fires first
    /// (mirrors Python's per-permanent
    /// `put_into_battlefield` -> `_trigger_etb_abilities` loop).
    ///
    /// **Not serialized into the fingerprint.** Bookkeeping like
    /// `triggered_this_turn`. The queue is empty across stack
    /// resolution boundaries (every drain site empties it before
    /// returning to its caller), so the field is observably empty
    /// when fingerprinted.
    pub pending_etb_dispatches: Vec<u32>,
    /// **DIES-trigger substrate (Slice 5) — deferred dispatch queue.**
    /// Sibling of `pending_etb_dispatches`: SBA's destroy path moves
    /// dying creatures into the graveyard mid-loop and queues their
    /// ids here. `apply_sbas` drains the queue via
    /// `drain_pending_dies_dispatches` AFTER the SBA loop completes
    /// so cascade death triggers (a DIES trigger that pushes a stack
    /// item, in turn potentially adding new SBA victims when it
    /// resolves) don't compete with the in-progress SBA iteration.
    ///
    /// **Drain semantics.** FIFO drain — each dying creature's
    /// triggers fire in destruction order (which is bf-ETB-timestamp
    /// order per `collect_sba_victims`'s walk). Mirrors Python's
    /// `turn.move_off_battlefield` -> `dispatch_event(Event.DIES,
    /// {"obj": dead}, extra_observers=[dead])` pattern.
    ///
    /// Like `pending_etb_dispatches`, not serialized into the
    /// fingerprint: the queue is empty across stack resolution
    /// boundaries (every drain site empties it before returning).
    pub pending_dies_dispatches: Vec<u32>,
    /// **LEAVES_BATTLEFIELD substrate (Slice 8) — deferred dispatch
    /// queue.** Sibling of `pending_dies_dispatches`. Every
    /// bf-exit primitive (`move_object` off-bf, SBA destroy)
    /// queues the leaving object's id here; the natural caller
    /// drains via `drain_pending_leaves_dispatches` after the
    /// primitive returns. Mirrors Python's
    /// `_fire_leaves_battlefield_triggers` call in
    /// `move_off_battlefield` (turn.py:289), which fires for ALL
    /// bf-exits (death, exile, bounce, sacrifice).
    ///
    /// **Relationship to DIES queue:** SBA-destroyed creatures push
    /// onto BOTH queues (push order: DIES first, LEAVES second);
    /// `apply_sbas` drains DIES first then LEAVES so DIES triggers
    /// land below LEAVES triggers on the stack — and the stack
    /// being LIFO, LEAVES resolves FIRST. This matches Python's
    /// `move_off_battlefield` ordering at `turn.py:280-289` (DIES
    /// dispatch precedes LEAVES dispatch in source order; Python's
    /// stack is also LIFO so LEAVES resolves first).
    ///
    /// **Drain semantics.** FIFO drain — each leaving permanent's
    /// triggers fire in bf-exit order.
    ///
    /// Like the other dispatch queues, not serialized into the
    /// fingerprint: the queue is empty across stack resolution
    /// boundaries.
    pub pending_leaves_dispatches: Vec<u32>,
    /// **IS_SACRIFICED substrate (Slice 14) — deferred dispatch queue.**
    /// Mirror of `pending_leaves_dispatches` for the `IS_SACRIFICED`
    /// event. The `Effect::Sacrifice` arm at
    /// `state_ops_resolve.rs::apply_effect` pushes the sacrificed
    /// permanent's id here; the natural caller drains via
    /// `drain_pending_is_sacrificed_dispatches` AFTER the resolving
    /// spell's stack pop. Without this defer-after-pop discipline a
    /// direct `dispatch_event` during the sacrifice spell's apply
    /// would push the IS_SACRIFICED trigger ABOVE the still-present
    /// resolving spell, and the spell's epilogue `stack.pop()` would
    /// consume the trigger before it could resolve.
    ///
    /// Mirrors Python's `turn.sacrifice` semantics where
    /// `dispatch_event(Event.IS_SACRIFICED, ...)` fires AFTER the
    /// spell's resolve has returned (the trigger lands at the next
    /// SBA + priority cycle, with the original sacrifice spell no
    /// longer on the stack).
    ///
    /// Like `pending_leaves_dispatches`, not serialized into the
    /// fingerprint: the queue is empty across stack resolution
    /// boundaries.
    pub pending_is_sacrificed_dispatches: Vec<u32>,
    /// **Round-55 token-cease-to-exist queue (CR 111.7 / 704.5d).**
    /// Sibling of `pending_dies_dispatches` / `pending_leaves_
    /// dispatches`. `mark_left_battlefield_if_nonland` (the one
    /// helper every bf-exit site calls) pushes a token's id here
    /// when that printing `is_token`. The id is drained — and the
    /// token removed from whatever non-battlefield zone it landed
    /// in — at the END of `drain_pending_leaves_dispatches` (the
    /// universal post-bf-exit drain, also called by `apply_sbas`
    /// after the DIES drain) and again at the tail of `apply_sbas`
    /// as a belt-and-suspenders sweep for any bf-exit path that
    /// doesn't itself drain LEAVES.
    ///
    /// **Why deferred rather than eager.** Python's
    /// `turn.move_off_battlefield` (turn.py:241-263) removes the
    /// token from the battlefield, then fires DIES (`_fire_dies_
    /// triggers(state, obj)`) and LEAVES with the token passed BY
    /// REFERENCE — the token is in no zone but its own death/leave
    /// triggers still fire. Rust's bf-exit primitives push the
    /// object to its destination zone, queue DIES/LEAVES by id,
    /// then drain. Deferring the cease-removal until AFTER those
    /// drains keeps the token findable as an extra observer during
    /// its own DIES/LEAVES fan-out (faithful to Python's by-ref
    /// semantics) while guaranteeing the final state has the token
    /// in NO zone — byte-equal to Python, whose token never lands
    /// in graveyard/exile/hand at all.
    ///
    /// Not serialized into the fingerprint — empty across stack /
    /// SBA resolution boundaries, like the sibling dispatch queues.
    pub pending_token_cease: Vec<u32>,
    /// **CE20 follow-up Slice 4 (Equipment-equip) — payload-target
    /// resolve-time slot.** Optional `target_id` swapped in by the
    /// stack resolver (and the CE20 activated-ability dispatch
    /// arm for is_mana_ability=true paths) BEFORE `apply_effect`
    /// fires, restored to its prior value after.
    /// `Effect::EquipToTargetCreatureYouControl` (and future
    /// targeted-activation arms — sacrifice-target, pump-target,
    /// etc.) reads this field to find the activator's chosen
    /// target; absent the swap (every pre-Slice-4 caller) the field
    /// stays `None` and the affected arms fall back to their
    /// pre-Slice-4 first-eligible-creature walk.
    ///
    /// **Why a transient field instead of threading through
    /// apply_effect?** Mirrors the `state.active_player` swap
    /// pattern already used by the CE20 Slice 1 bridge follow-up
    /// for source-controller resolution (`state_ops_actions.rs:725`).
    /// Apply-effect has ~90 arms; threading an `Option<u32>`
    /// parameter through every signature would be invasive for
    /// surfaces that don't care. A transient field is additive,
    /// localized to the resolver, and restored after each apply
    /// (so no observable state leaks across resolution boundaries).
    ///
    /// **Not serialized into the fingerprint.** Transient like
    /// `consecutive_passes`, `next_etb_timestamp`, and
    /// `resolving_combat_damage_amount` — same observable game
    /// state pre- and post-resolve produces the same digest
    /// regardless of intermediate target picks.
    pub current_resolve_payload_target_id: Option<u32>,
    /// **Spell-card-bridge lever, Phase 2 (round-76).** The SECOND
    /// resolve-time payload target — the slot-1 / heterogeneous
    /// second target of a 2-slot spell (today: Diplomatic
    /// Relations' opponent-creature damage target; slot 0 is the
    /// friendly pumped creature carried on
    /// `current_resolve_payload_target_id`). Swapped in/out by both
    /// resolve entry points exactly like its single-slot sibling
    /// (save → set → restore bracketing the apply closure). `None`
    /// for every pre-Phase-2 push (back-compat — single-target
    /// spells, triggers, and activations never set it; the closed
    /// Phase-1 single-target arms self-scan and IGNORE both payload
    /// targets, so the swap is a correct no-op for them). It is the
    /// forward substrate for any 2-slot arm that opts into the
    /// payload-target convention (today only
    /// `DiplomaticRelationsEffect`). Transient + not serialized into
    /// the fingerprint, identical discipline to
    /// `current_resolve_payload_target_id`.
    pub current_resolve_payload_target_id_2: Option<u32>,
    /// **CE20 follow-up Slice 5 (Walker loyalty Slice C) —
    /// once-per-turn-per-walker activation ledger.** CR 606.3:
    /// "A player may activate a loyalty ability of a planeswalker
    /// they control only if no player has activated a loyalty
    /// ability of that planeswalker this turn." Keys are
    /// `(controller_idx, walker_obj_id)` pairs; presence means the
    /// walker has already had one of its loyalty abilities
    /// activated this turn.
    ///
    /// **Stamp site**: `ActivateAbility` apply Phase 5e (after the
    /// loyalty-counter mutation in Phase 5d). Only stamped when
    /// the source has card_type Planeswalker — non-walker
    /// activations never touch this ledger.
    ///
    /// **Gate sites**:
    ///   - `legal_decisions` emission walk (state_ops_legal.rs):
    ///     when the source is a walker AND `(controller, source_id)`
    ///     is in the ledger, skip ALL the source's activated
    ///     abilities. This is a per-walker gate, not a per-ability
    ///     gate — the rule says "a loyalty ability", singular, per
    ///     walker per turn.
    ///   - `ActivateAbility` apply precheck (Phase 3c): same shape
    ///     as the emission gate; raises BEFORE any commit on
    ///     direct-caller bypass.
    ///
    /// **Reset**: cleared at the cleanup step
    /// (`cleanup_step_for_active_player`), same site as
    /// `triggered_this_turn`. CR 606.3 is "this turn"-scoped.
    ///
    /// **Why a tuple key and not just walker_obj_id?** CR 606.3 is
    /// per-player-per-walker — multiple players can each activate
    /// their own walker's loyalty ability on the same turn. Today
    /// only the controller activates their own walker's loyalty
    /// abilities (CR 606.1), but the (controller, walker_id) key
    /// shape mirrors `triggered_this_turn` and stays future-proof
    /// for "your opponent can activate your walker" mechanics that
    /// don't exist in EoE.
    ///
    /// **Not serialized into the fingerprint.** Bookkeeping like
    /// `triggered_this_turn`. Two states reached via different
    /// histories that share the same observable battlefield /
    /// stack / hand / library / graveyard / life MUST produce the
    /// same fingerprint regardless of which loyalty abilities have
    /// or haven't been activated — gameplay-observable effect of
    /// the ledger is "the next ATTEMPT to activate fails" which
    /// will surface as a legal_decisions emission shape change at
    /// the priority point that uses the ledger.
    pub loyalty_activated_this_turn: std::collections::HashSet<(u8, u32)>,
    /// **DrawCards controller-of-effect fix — transient resolve-time
    /// source controller.** Set by the stack resolver
    /// (`resolve_top_of_stack` / `_pause`) and by the cost-pay path
    /// for mana abilities (`state_ops_actions.rs`) BEFORE
    /// `apply_effect` fires, restored to its prior value after.
    /// `Effect::DrawCards` (and any future arm that resolves to the
    /// SOURCE's controller, NOT the active player) reads this field
    /// to credit the right player; absent the swap (every pre-fix
    /// caller / direct apply_effect_pub user) the field stays `None`
    /// and DrawCards falls back to `state.active_player` for back-
    /// compat. New apply_effect_pub_with_controller(state, effect,
    /// source_id, controller_idx) helper sets+restores around a
    /// single apply call so direct callers don't have to remember
    /// the swap.
    ///
    /// **Why a transient field instead of threading through
    /// apply_effect?** Same rationale as
    /// `current_resolve_payload_target_id`: apply_effect has ~90
    /// arms, threading an `Option<u8>` through every signature is
    /// invasive for surfaces that don't care. Mirror the existing
    /// payload-target swap pattern (Slice 4) and the
    /// `state.active_player` swap pattern (CE20 Slice 1 bridge).
    ///
    /// **Not serialized into the fingerprint.** Transient like
    /// `current_resolve_payload_target_id` — same observable game
    /// state pre- and post-resolve produces the same digest
    /// regardless of intermediate controller swaps.
    pub current_resolve_source_controller: Option<u8>,
    /// **ModalChooseOne envelope — Cycle 2 (per-mode resolve
    /// dispatch, 2026-05-21).** Transient resolve-time mirror of
    /// the resolving spell's `StackItem.chosen_mode_idx`. Set by
    /// the stack resolver (`resolve_top_of_stack` /
    /// `resolve_top_of_stack_pause`) BEFORE the per-effect
    /// `apply_effect` fan-out for a `StackItemKind::Spell` whose
    /// pushed item carries a cast-time-bound mode pick; restored
    /// to its prior value after. The `Effect::ChooseOneMode`
    /// resolve arm reads this field — if `Some(idx)`, it dispatches
    /// directly to `modes[idx]` via `apply_effect` (mirroring
    /// Python's `state.resolving_modes` replay at
    /// `combinators.py:1213-1215`). If `None` (no cast-time bind —
    /// every nested-as-trigger ChooseOneMode + pre-Cycle-1 push),
    /// the arm falls back to the existing
    /// `PendingDecision::ChooseModeChoice` push (Cycle 1
    /// resolve-time-callback substrate stays intact for non-spell-
    /// cast invocations).
    ///
    /// **Why a transient field instead of threading through
    /// apply_effect.** Same rationale as
    /// `current_resolve_payload_target_id` / `_2` /
    /// `current_resolve_source_controller`: apply_effect has ~90
    /// arms, threading an `Option<u8>` through every signature is
    /// invasive for surfaces that don't care. Mirror the existing
    /// payload-target swap pattern.
    ///
    /// **Not serialized into the fingerprint.** Transient like its
    /// payload-target / source-controller siblings — same observable
    /// game state pre- and post-resolve produces the same digest
    /// regardless of intermediate mode-index swaps.
    pub current_resolve_chosen_mode_idx: Option<u8>,
    /// **Close Encounter additional-cost-choice envelope — Cycle 1
    /// (resolve-time transient mirror, 2026-05-21).** Transient
    /// resolve-time mirror of the resolving spell's
    /// `StackItem.additional_cost_choice_id`. Set by the stack
    /// resolver (`resolve_top_of_stack` / `resolve_top_of_stack_pause`)
    /// BEFORE the per-effect `apply_effect` fan-out for a
    /// `StackItemKind::Spell` whose pushed item carries a cast-time-
    /// bound additional-cost choice; restored to its prior value
    /// after. The `Effect::CloseEncounterEffect` resolve arm (Cycle
    /// 3) reads this field — `Some(id)` is the cast-time chosen
    /// creature (battlefield) or warped creature card (caster's
    /// exile); `None` means no choice was bound (the
    /// additional-cost-choose-your-creature flag wasn't set on the
    /// printing, so the resolve arm — if hit — should be a no-op
    /// fallback).
    ///
    /// **Mirrors Python's `state.resolving_additional_cost_choice_id`
    /// swap** at `actions.py:2681-2683` / `:2711` / `:2698` / `:2795`
    /// (set BEFORE the resolve closure, restored to `None` in the
    /// `finally`). Same save → set → restore bracketing as
    /// `current_resolve_chosen_mode_idx`.
    ///
    /// **Why a transient field instead of threading through
    /// apply_effect.** Same rationale as
    /// `current_resolve_payload_target_id` / `_2` /
    /// `current_resolve_source_controller` /
    /// `current_resolve_chosen_mode_idx`: apply_effect has ~90 arms,
    /// threading an `Option<u32>` through every signature is invasive
    /// for surfaces that don't care. Mirror the existing payload-
    /// target swap pattern.
    ///
    /// **Not serialized into the fingerprint.** Transient like its
    /// payload-target / source-controller / mode-index siblings —
    /// same observable game state pre- and post-resolve produces the
    /// same digest regardless of intermediate choice-id swaps.
    pub current_resolve_additional_cost_choice_id: Option<u32>,
    /// **ModalChooseOne envelope — scour-for-scrap-slice (Cycle 1
    /// widen, 2026-05-21).** Multi-pick transient mirror of
    /// `StackItem.chosen_mode_idxs` (the multi-pick sibling of
    /// `chosen_mode_idx`). Bracketed by the stack resolver — set
    /// from `top.chosen_mode_idxs` before the per-effect
    /// `apply_effect` fan-out and restored after. When `Some(vec)`,
    /// the `Effect::ChooseOneMode` resolve arm dispatches each chosen
    /// mode in order via `apply_effect_pub_with_controller`; when
    /// `None`, the arm falls back to the legacy single-pick
    /// `current_resolve_chosen_mode_idx` path (back-compat with the
    /// pre-widen Cycle-2 resolve dispatch).
    ///
    /// Transient like its single-pick sibling; not serialized into
    /// the fingerprint.
    pub current_resolve_chosen_mode_idxs: Option<Vec<u8>>,
    /// **X-value subsystem — Cycle 1 (resolve-time transient mirror,
    /// 2026-05-21).** Transient resolve-time mirror of the
    /// resolving spell's `StackItem.x_value`. Set by the stack
    /// resolver (`resolve_top_of_stack` / `resolve_top_of_stack_pause`)
    /// BEFORE the per-effect `apply_effect` fan-out for a
    /// `StackItemKind::Spell` whose pushed item carries a cast-time-
    /// bound X value; restored to its prior value after. Future
    /// `Effect::ZeroPointBallad` / `Effect::DevastatingOnslaught`
    /// resolve arms (Cycle 3+) will read this field — `Some(x)` is
    /// the cast-time chosen X; `None` means no X was bound (the
    /// spell wasn't X-cost, so the resolve arm — if hit — should be
    /// a no-op fallback).
    ///
    /// **Mirrors Python's `state.resolving_x_value` swap** at
    /// `actions.py` (set BEFORE the resolve closure, restored to
    /// `None` in the `finally`). Same save → set → restore
    /// bracketing as `current_resolve_chosen_mode_idx` /
    /// `current_resolve_additional_cost_choice_id`.
    ///
    /// **Why a transient field instead of threading through
    /// apply_effect.** Same rationale as every sibling transient:
    /// apply_effect has ~90 arms, threading an `Option<i32>` through
    /// every signature is invasive for surfaces that don't care.
    /// Mirror the existing payload-target / mode-index / addcost
    /// swap pattern.
    ///
    /// **Not serialized into the fingerprint.** Transient like its
    /// payload-target / source-controller / mode-index / addcost
    /// siblings — same observable game state pre- and post-resolve
    /// produces the same digest regardless of intermediate X-value
    /// swaps.
    ///
    /// **Cycle 1 substrate-only.** NO live mint site populates this
    /// field today; every pre-substrate stack push pushes
    /// `x_value: None`, so the swap is a no-op and the back-compat
    /// invariant holds byte-identical.
    pub current_resolve_x_value: Option<i32>,
}

impl Clone for GameState {
    fn clone(&self) -> Self {
        // Cloning a `Py<PyAny>` requires the GIL because it bumps the
        // Python-side refcount. All `GameState::clone` call sites today
        // are inside pymethods (clone_once, clone_n_times_micros, the
        // implicit clones in fingerprint serialization paths) which run
        // under the GIL by construction; `Python::with_gil` is a no-op
        // when the GIL is already held (just hands back a token).
        let rng = self
            .rng
            .as_ref()
            .map(|r| Python::with_gil(|py| r.clone_ref(py)));
        GameState {
            players: self.players.clone(),
            registry: self.registry.clone(),
            active_player: self.active_player,
            priority_player: self.priority_player,
            turn_number: self.turn_number,
            phase: self.phase,
            attackers: self.attackers.clone(),
            blockers: self.blockers.clone(),
            stack: self.stack.clone(),
            // G7 S5a: the stack-object store rides every clone (the
            // determinizer clones the provider then determinizes the
            // clone; a mid-stack clone must keep the stack-resident
            // source so resolve still finds it). CoW-cheap — a Vec of
            // GameObject; almost always 0 or 1 entries.
            stack_objects: self.stack_objects.clone(),
            // G7 Class-C delayed-trigger port: a determinized clone
            // must keep pending delayed triggers so the end-step push
            // still fires on the clone. CoW-cheap — Vec<DelayedTrigger>
            // typically empty or 1-2 entries (one per warp creature
            // pending exile). Same lifecycle as `pending_decisions`.
            delayed_triggers: self.delayed_triggers.clone(),
            consecutive_passes: self.consecutive_passes,
            permanent_left_battlefield_this_turn: self.permanent_left_battlefield_this_turn,
            spell_warped_this_turn: self.spell_warped_this_turn,
            // Phase 4 S3 (Moonlit Meditation substrate landing,
            // 2026-05-20): per-player created-tokens-this-turn flags
            // are gameplay-relevant turn-scoped state — every clone
            // carries them forward (mirrors Python's
            // `players_who_created_tokens_this_turn = set(self...)`
            // copy at `game.py:3370-3372`).
            created_tokens_this_turn: self.created_tokens_this_turn,
            pending_decisions: self.pending_decisions.clone(),
            next_etb_timestamp: self.next_etb_timestamp,
            // R76-1: CR 111.7 is a game-lifetime invariant — a
            // determinized clone must keep allocating object ids above
            // every id the parent ever used (never recycle a freed
            // id). Persist the monotonic counter, mirroring Python's
            // `clone()` copying `state.next_object_id` (game.py:3322).
            next_object_id: self.next_object_id,
            resolving_combat_damage_amount: self.resolving_combat_damage_amount,
            resolving_combat_damage_target_player_id: self.resolving_combat_damage_target_player_id,
            winner_id_sticky: self.winner_id_sticky,
            rng,
            triggered_this_turn: self.triggered_this_turn.clone(),
            pending_etb_dispatches: self.pending_etb_dispatches.clone(),
            pending_dies_dispatches: self.pending_dies_dispatches.clone(),
            pending_leaves_dispatches: self.pending_leaves_dispatches.clone(),
            pending_is_sacrificed_dispatches: self.pending_is_sacrificed_dispatches.clone(),
            pending_token_cease: self.pending_token_cease.clone(),
            current_resolve_payload_target_id: self.current_resolve_payload_target_id,
            current_resolve_payload_target_id_2: self.current_resolve_payload_target_id_2,
            // ModalChooseOne envelope — Cycle 2. Transient mode-
            // index slot; cloned as `Option<u8>` (Copy). Bracketed
            // by the resolver — `None` outside the resolve window
            // by construction, so any clone taken mid-resolve still
            // sees a deterministic snapshot.
            current_resolve_chosen_mode_idx: self.current_resolve_chosen_mode_idx,
            // scour-for-scrap-slice Cycle 1 widen: transient multi-
            // pick slot. Vec<u8> clone is cheap (typically 0-2 idxs).
            current_resolve_chosen_mode_idxs: self.current_resolve_chosen_mode_idxs.clone(),
            // CE20 Slice 5 (Walker loyalty Slice C): clone the once-
            // per-turn-per-walker ledger. CoW-friendly; HashSet's
            // clone is O(n) where n is the number of distinct walker
            // activations this turn (typically 0 or 1 in EoE).
            loyalty_activated_this_turn: self.loyalty_activated_this_turn.clone(),
            // DrawCards controller-of-effect fix: transient resolve-
            // time source controller. Cloned as Option<u8> (Copy).
            current_resolve_source_controller: self.current_resolve_source_controller,
            // Close Encounter additional-cost-choice envelope —
            // Cycle 1. Transient choice-id slot; cloned as
            // `Option<u32>` (Copy). Bracketed by the resolver —
            // `None` outside the resolve window by construction.
            current_resolve_additional_cost_choice_id:
                self.current_resolve_additional_cost_choice_id,
            // X-value subsystem Cycle 1. Transient X-value slot;
            // cloned as `Option<i32>` (Copy). Bracketed by the
            // resolver — `None` outside the resolve window by
            // construction, so any mid-resolve clone still sees a
            // deterministic snapshot.
            current_resolve_x_value: self.current_resolve_x_value,
        }
    }
}

impl GameState {
    /// **Effect-rng-threading closure — borrow the stashed external rng.**
    /// Returns `Some(&Py<PyAny>)` when a Python `random.Random` (or
    /// compatible) has been set via `set_external_rng`, otherwise
    /// `None`. Effect resolve arms that consume randomness call this
    /// to decide between the external-rng path (Python MT byte-for-byte
    /// parity) and the xorshift64 fallback (production cli / bench).
    /// Crate-private — exposed only to sibling rust modules; Python
    /// callers manipulate the rng via the `set_external_rng` pymethod.
    pub(crate) fn rng_ref(&self) -> Option<&Py<PyAny>> {
        self.rng.as_ref()
    }
}

#[pymethods]
impl GameState {
    #[getter]
    fn turn_number(&self) -> u32 {
        self.turn_number
    }

    #[getter]
    fn battlefield_sizes(&self) -> [usize; 2] {
        [self.players[0].battlefield.len(), self.players[1].battlefield.len()]
    }

    #[getter]
    fn hand_sizes(&self) -> [usize; 2] {
        [self.players[0].hand.len(), self.players[1].hand.len()]
    }

    #[getter]
    fn library_sizes(&self) -> [usize; 2] {
        [self.players[0].library.len(), self.players[1].library.len()]
    }

    #[getter]
    fn life_totals(&self) -> [i32; 2] {
        [self.players[0].life, self.players[1].life]
    }

    /// **EEE downstream complement — 8th Rust slice.**
    /// Expose per-player poison counter counts to Python callers (tests
    /// and the parity bridge).  Mirrors `Player.poison_counters` on the
    /// Python side.  Returns `[p0_poison, p1_poison]`.
    #[getter]
    fn poison_counter_totals(&self) -> [i32; 2] {
        [self.players[0].poison_counters, self.players[1].poison_counters]
    }

    /// **EEE downstream complement — 8th Rust slice.**
    /// Python-exposed wrapper around `dispatch_combat_damage_to_player`
    /// for test probes.  Mirrors the internal pub(crate) function that
    /// is called by `apply_damage_pass` during real combat.
    ///
    /// Parameters:
    ///   source_id    — the attacking/blocking creature's object id
    ///   target_player — the player id (0 or 1) who received the damage
    ///   amount        — the amount of damage dealt (used for
    ///                   `resolving_combat_damage_amount` and for
    ///                   `DrawCardsEqualToResolvingCombatDamage` effects)
    ///
    /// Returns `Ok(())` (dispatches the event; any stack items from
    /// triggered abilities are pushed for later resolution).
    fn dispatch_combat_damage_to_player(
        &mut self,
        source_id: u32,
        target_player: u8,
        amount: i32,
    ) -> pyo3::PyResult<()> {
        crate::state_ops_resolve::dispatch_combat_damage_to_player(
            self, source_id, target_player, amount,
        )
    }

    #[getter]
    fn registry_size(&self) -> usize {
        self.registry.len()
    }

    fn clone_once(&self) -> GameState {
        self.clone()
    }

    fn clone_n_times_micros(&self, n: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..n {
            let cloned = self.clone();
            std::hint::black_box(cloned);
        }
        let elapsed = start.elapsed();
        (elapsed.as_secs_f64() * 1_000_000.0) / n as f64
    }

    /// Resolve a GameObject id (searched across both players' four
    /// zones) to its printing's name. Returns `None` (Python `None`)
    /// if no object with that id exists in any zone.
    ///
    /// This is the smallest end-to-end test of the Registry wiring:
    /// it walks every zone, finds the object with matching `id`,
    /// looks up its `printing_id` in `self.registry`, and returns
    /// the printing's name. A failure anywhere in the chain — wrong
    /// id, wrong printing_id, registry not threaded through — shows
    /// up here.
    fn printing_name_of(&self, obj_id: u32) -> Option<String> {
        for player in &self.players {
            for zone in [
                &player.hand,
                &player.library,
                &player.battlefield,
                &player.graveyard,
            ] {
                for obj in zone.iter() {
                    if obj.id == obj_id {
                        return self
                            .registry
                            .lookup(obj.printing_id)
                            .map(|p| p.name.to_string());
                    }
                }
            }
        }
        None
    }

    /// Same as `printing_name_of` but raises KeyError on miss —
    /// useful when Python code wants the loud-fail variant.
    fn require_printing_name_of(&self, obj_id: u32) -> PyResult<String> {
        self.printing_name_of(obj_id).ok_or_else(|| {
            PyKeyError::new_err(format!("no game object with id {obj_id}"))
        })
    }

    /// Test-only setter for the EoE Void condition flags. Sets
    /// `permanent_left_battlefield_this_turn` and
    /// `spell_warped_this_turn` directly. Mirrors Python's
    /// equivalent fields on `GameState`; the production call sites
    /// that flip these (LTB tracking, warp tracking) are not yet
    /// wired in Rust, so tests inject the state directly. Both
    /// arguments default to `False`.
    #[pyo3(signature = (permanent_left_battlefield = false, spell_warped = false))]
    fn set_void_flags(
        &mut self,
        permanent_left_battlefield: bool,
        spell_warped: bool,
    ) {
        self.permanent_left_battlefield_this_turn = permanent_left_battlefield;
        self.spell_warped_this_turn = spell_warped;
    }

    /// Read-only getter for the void-active predicate, mirroring
    /// Python's `is_void_active(state)`. True iff either of the two
    /// void flags is set.
    #[getter]
    fn is_void_active(&self) -> bool {
        self.permanent_left_battlefield_this_turn || self.spell_warped_this_turn
    }

    /// Test-only setter for the resolving-combat-damage amount slot.
    /// Mirrors Python's `state.resolving_combat_damage_amount`. The
    /// production combat-damage assignment code path that would
    /// write this in Rust is deferred (CE9 closure agent owns
    /// `state_ops_combat.rs`); for now tests inject the value
    /// directly to exercise the read-side
    /// (`Effect::DrawCardsEqualToResolvingCombatDamage`).
    fn set_resolving_combat_damage_amount(&mut self, amount: i32) {
        self.resolving_combat_damage_amount = amount;
    }

    /// Read-only getter for the resolving-combat-damage amount slot.
    /// Mirrors Python's `state.resolving_combat_damage_amount`.
    #[getter]
    fn resolving_combat_damage_amount(&self) -> i32 {
        self.resolving_combat_damage_amount
    }

    /// Read-only getter for the LTB half of the Void predicate.
    /// Mirrors Python's field of the same name on `GameState`. True
    /// iff a nonland permanent has left the battlefield this turn
    /// (set inside `move_object` on battlefield-exit; reset on
    /// entry to PHASE_CLEANUP via `advance_phase`). See
    /// `docs/RUST_PARITY_GAPS.md` I29 for the full lifecycle.
    #[getter]
    fn permanent_left_battlefield_this_turn(&self) -> bool {
        self.permanent_left_battlefield_this_turn
    }

    /// Read-only getter for the warp half of the Void predicate.
    /// Mirrors Python's field of the same name on `GameState`. Set
    /// when a spell is cast via its warp alt-cost (no Rust call
    /// site flips this yet — warp casting is a future slice); reset
    /// on entry to PHASE_CLEANUP via `advance_phase` alongside the
    /// LTB half.
    #[getter]
    fn spell_warped_this_turn(&self) -> bool {
        self.spell_warped_this_turn
    }

    /// **Aura-attachment substrate — Phase 4 S3 (Moonlit Meditation
    /// substrate landing, 2026-05-20).** Read-only getter for the
    /// per-player "this player has created one or more tokens this
    /// turn" flag. Returns `True` iff `player_idx`'s slot is set;
    /// `False` for any out-of-range index (defensive — only two
    /// player slots are populated). Mirrors Python's
    /// `(player_id in state.players_who_created_tokens_this_turn)`
    /// read pattern used in the Moonlit Meditation replacement check
    /// at `effects/generic/tokens.py:225`. Flipped by every
    /// non-zero-count `Effect::CreateToken` / `CreateTokenTapped` /
    /// `CreateTokenEachPlayer` / `CreateTokenCopyOfTargetCreature
    /// YouControl` arm; reset to `[false; 2]` on entry to
    /// PHASE_CLEANUP alongside the void flags.
    fn created_tokens_this_turn(&self, player_idx: u8) -> bool {
        let idx = player_idx as usize;
        if idx < self.created_tokens_this_turn.len() {
            self.created_tokens_this_turn[idx]
        } else {
            false
        }
    }

    /// **Effect-rng-threading closure (G19 follow-up) — install a
    /// caller-owned Python `random.Random` for in-Effect shuffles.**
    ///
    /// Pass `Some(random_random_instance)` to opt this state into the
    /// external-rng path: Effect resolve arms that consume randomness
    /// (today only `Effect::SearchLibraryForBasicLandTapped`'s post-
    /// search Fisher-Yates per CR 701.19 / 701.22) will read draws via
    /// `_randbelow`, producing byte-identical permutations to what
    /// `random.Random.shuffle(list)` produces on the Python side. Pass
    /// `None` to clear (return to the xorshift64 fallback).
    ///
    /// Mirrors the substrate established by CE8 Slice 2 for the 8
    /// pymethod-surface shuffle sites (I41) — same fall-through
    /// convention: when `rng` is `None`, the legacy xorshift64 path
    /// runs unchanged. Production cli / bench callers leave this
    /// unset and stay on the xorshift64 path; cross-engine smoke
    /// tests opt in.
    ///
    /// **Lifecycle.** Caller-owned. The harness sets the rng once per
    /// game (before any `apply_effect` / `resolve_top_of_stack` /
    /// `auto_resolve_pending_decisions` call that might fire a shuffle-
    /// consuming Effect) and clears it at end-of-game. Two
    /// `state.clone_once()` siblings share the SAME `random.Random`
    /// instance — for the cross-engine smoke use case this is correct
    /// (one rng per game). MCTS-style search-tree forking that depends
    /// on independent rng streams must call `set_external_rng(None)`
    /// on the forked branches before play continues.
    ///
    /// **Not serialized into the fingerprint.** Bookkeeping state, not
    /// gameplay state. Two `GameState`s with the same observable shape
    /// fingerprint identically regardless of whether/which rng is
    /// stashed (see I43).
    #[pyo3(signature = (rng = None))]
    fn set_external_rng(&mut self, rng: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.rng = rng.map(|r| r.clone().unbind());
        Ok(())
    }
}

fn make_object(id: u32, controller: u8, printing_id: u32) -> GameObject {
    GameObject {
        id,
        printing_id,
        controller,
        zone: 1,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: 0,
        charge_counters: 0,
        loyalty_counters: 0,
        attached_to: None,
        exiled_with_source: None,
        stun_counters: 0,
        // CE4 closure: synthetic-state fixtures pre-populate the
        // battlefield without going through stamp_etb_timestamp, so
        // every object defaults to 0. The stable-sort in the snapshot
        // / fingerprint preserves the make_player insertion order
        // (p0-then-p1, all timestamps tied at 0) — pre-CE4 baselines
        // stay bit-equal.
        etb_timestamp: 0,
        ueot_modifiers: Vec::new(),
        // warp_cost substrate (Phase 1, round-63): no object is
        // warp-pending or warp-exiled at allocation. Back-compat
        // default; not in the per-object fingerprint tuple.
        warp_exile_pending: false,
        warped: false,
        // Cast-from-exile-with-cost-hike substrate (Lightstall
        // Inquisitor residue closure): no freshly-allocated object
        // carries the Lightstall flag; stamped only at Effect-
        // resolution time on the exiled card.
        cast_from_exile_with_lightstall_surcharge: false,
        // Aura-attachment substrate — Phase 2 S2 (round-79+,
        // 2026-05-20): freshly-allocated objects have no captured
        // baseline. Back-compat default; not in the per-object
        // fingerprint tuple.
        controller_baseline: None,
        // Aura-attachment substrate — Phase 4 S3 closure
        // (2026-05-21): freshly-allocated objects have no Token
        // supertype injected; the override is stamped only at
        // Moonlit-Meditation-replacement token-mint time. Back-compat
        // default; not in the per-object fingerprint tuple.
        token_supertype_override: false,
        // Control-change-UEOT substrate — Cycle 1 (2026-05-21):
        // freshly-allocated objects have no UEOT controller override;
        // the override is stamped only by
        // `grant_control_change_ueot` (Mutinous Massacre / Systems
        // Override / Threaten shape). Back-compat default; not in
        // the per-object fingerprint tuple.
        controller_override: None,
    }
}

fn make_player(
    id: u8,
    num_battlefield: usize,
    num_hand: usize,
    num_library: usize,
    base_id: u32,
    forest_pid: u32,
    bears_pid: u32,
    elves_pid: u32,
) -> Player {
    let mut next = base_id;
    let mut battlefield = Vec::with_capacity(num_battlefield);
    let half = num_battlefield / 2;
    for _ in 0..half {
        battlefield.push(make_object(next, id, forest_pid));
        next += 1;
    }
    for _ in half..num_battlefield {
        battlefield.push(make_object(next, id, bears_pid));
        next += 1;
    }
    let mut hand = Vec::with_capacity(num_hand);
    for _ in 0..num_hand {
        hand.push(make_object(next, id, elves_pid));
        next += 1;
    }
    let mut library = Vec::with_capacity(num_library);
    for _ in 0..num_library {
        library.push(make_object(next, id, forest_pid));
        next += 1;
    }
    Player {
        id,
        life: 20,
        mana_pool: ManaPool::default(),
        hand: Arc::new(hand),
        library: Arc::new(library),
        battlefield: Arc::new(battlefield),
        graveyard: Arc::new(Vec::new()),
        exile: Arc::new(Vec::new()),
        has_lost: false,
        attempted_draw_on_empty: false,
        mulligan_count: 0,
        life_gained_this_turn: 0,
        poison_counters: 0,
        spells_cast_this_turn: 0,
        artifacts_entered_under_your_control_this_turn: 0,
    }
}

/// Build a synthetic GameState with a small in-process registry of
/// 3 vanilla printings (Forest, Grizzly Bears, Llanowar Elves).
/// Each GameObject in the resulting state references one of those
/// printings by id — no per-object name/p/t/cost duplication.
///
/// The `make_player` placement loop puts elves in hand first (so
/// the first hand object resolves to "Llanowar Elves"), forests +
/// bears split on the battlefield, and forests in the library.
/// The Python smoke tests rely on that ordering to verify the
/// printing-name lookup chain.
#[pyfunction]
#[pyo3(signature = (num_battlefield = 10, num_hand = 7, num_library = 50))]
pub fn make_synthetic_state(
    num_battlefield: usize,
    num_hand: usize,
    num_library: usize,
) -> GameState {
    // Build the registry up-front, snapshot it, share the Arc.
    //
    // Round-3 bridge extension: each printing now carries optional
    // `power` / `toughness` / `mana_produced`. Forest is non-creature
    // (no P/T) but produces {G}; Grizzly Bears is a 2/2 with no
    // mana production; Llanowar Elves is a 1/1 that for first-slice
    // we model as if `mana_produced` were intrinsic (the real card's
    // "{T}: Add {G}" is an activated ability, but the abilities
    // schema doesn't exist yet — see the docstring on `tap_for_mana`
    // in state_ops_mana.rs and the `SEARCH_LIMITATIONS.md` entry
    // referenced from this slice's commit).
    let mut builder = Registry::default();
    // CE20 follow-up — bridge `mana_produced` into the synthesized
    // `activated_abilities` Vec via the shared
    // `cards::apply_mana_produced_bridge` helper. The
    // synthetic-state Forest / Llanowar Elves printings go through
    // the same bridge synthesis as dict-input registry-parsed
    // printings, so the activated-abilities emission path covers
    // them uniformly post-bridge.
    let forest_mana_produced = Some(vec!["G".to_string()]);
    let forest_activated = crate::cards::apply_mana_produced_bridge(
        &forest_mana_produced,
        Vec::new(),
    )
    .expect("bridge synthesis for Forest");
    let forest_pid = builder.push_printing(CardPrinting {
        name: Arc::from("Forest"),
        mana_cost: ManaCost::default(),
        effects: Vec::new(),
        power: None,
        toughness: None,
        mana_produced: forest_mana_produced,
        card_types: vec!["Land".to_string()],
        // Forest carries the basic-land-type subtype "Forest"
        // (CR 205.3i) — mirrors Python's
        // ``CardPrinting.subtypes=("Forest",)`` at
        // ``cross_engine.py:6337`` for the synthetic Forest printing.
        subtypes: vec!["Forest".to_string()],
        // Forest's printed type line is "Basic Land — Forest"
        // (CR 205.4 supertype). Round-30 supertypes substrate.
        supertypes: vec!["Basic".to_string()],
        keywords: Vec::new(),
        etb_effects: Vec::new(),
        triggered_abilities: Vec::new(),
        activated_abilities: forest_activated,
        starting_loyalty: 0,
        // warp_cost substrate (Phase 1, round-63): synthetic-state
        // printings have no warp. Back-compat default.
        warp_cost: None,
        // warp_cost substrate (Phase 2): no life / graveyard rider on
        // synthetic-state printings. Back-compat defaults.
        warp_additional_life_cost: 0,
        warp_castable_from_graveyard: false,
        // legal_decisions mana-awareness Slice 3: synthetic-state
        // printings carry no cast-time additional cost. Back-compat
        // defaults (the 4 EoE additional-cost cards are all
        // bridge-DECLINED — verified-zero live incidence).
        additional_cost_sacrifice_creature_or_artifact: false,
        additional_cost_sacrifice_creature_or_enchantment: false,
        additional_cost_choose_your_creature: false,
        // Aura-attachment substrate — Phase 1 STEP 2 (round-78):
        // synthetic-state printings are not Auras. Back-compat
        // defaults (every pre-substrate registry dict omits these
        // keys, parser maps to the same defaults).
        is_aura: false,
        aura_enchant_type: "creature".to_string(),
        aura_yours_only: false,
        aura_power_bonus: 0,
        aura_toughness_bonus: 0,
        aura_granted_keywords: Vec::new(),
        // Dynamic-P/T formula substrate (round-78): synthetic-state
        // printings have no dynamic P/T. Back-compat default (every
        // pre-substrate registry dict omits the `dynamic_pt_formula`
        // key, parser maps the absence to None).
        dynamic_pt_formula: None,
        // Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20):
        // synthetic-state printings are not S2 Auras. Back-compat
        // defaults — every pre-S2 registry dict omits these keys.
        aura_grants_control: false,
        aura_grants_lose_abilities: false,
        aura_set_base_pt: None,
        // Phase 2 S2 extension (Tractor Beam closure): synthetic-state
        // printings are not don't-untap Auras.
        aura_prevents_host_untap: false,
        // Phase 4 S3 (Moonlit Meditation substrate landing,
        // 2026-05-20): synthetic-state printings carry no
        // token-creation replacement rider. Back-compat default.
        aura_replaces_first_tokens_with_copies: false,
        // Static / layered abilities substrate — Phase 1 first-
        // deliverable (round-79): synthetic-state printings have no
        // anthem statics. Back-compat default (every pre-substrate
        // registry dict omits the `static_anthems` key).
        static_anthems: Vec::new(),
        // Static / layered abilities substrate — Phase 2 (round-80,
        // 2026-05-20): synthetic-state printings have no
        // uncounterable flags. Back-compat default.
        spell_is_uncounterable: false,
        grants_creature_spells_uncounterable: false,
        // Static / layered abilities substrate — Phase 3 (round-81,
        // 2026-05-20): synthetic-state printings have no SELF-scope
        // conditional bonuses. Back-compat default.
        static_self_bonuses: Vec::new(),
    });
    // Grizzly Bears: no mana_produced → bridge is a no-op (returns
    // Vec::new() unchanged).
    let bears_pid = builder.push_printing(CardPrinting {
        name: Arc::from("Grizzly Bears"),
        mana_cost: ManaCost {
            generic: 1,
            green: 1,
            ..ManaCost::default()
        },
        effects: Vec::new(),
        power: Some(2),
        toughness: Some(2),
        mana_produced: None,
        card_types: vec!["Creature".to_string()],
        // Grizzly Bears' printed type line is "Creature — Bear"
        // (CR 205.3m creature subtype) — mirrors Python's
        // ``subtypes=("Bear",)`` at ``cross_engine.py:6585``.
        subtypes: vec!["Bear".to_string()],
        // Round-30 supertypes substrate: Grizzly Bears has no
        // supertype prefix on its type line.
        supertypes: Vec::new(),
        keywords: Vec::new(),
        etb_effects: Vec::new(),
        triggered_abilities: Vec::new(),
        activated_abilities: Vec::new(),
        starting_loyalty: 0,
        // warp_cost substrate (Phase 1, round-63): synthetic-state
        // printings have no warp. Back-compat default.
        warp_cost: None,
        // warp_cost substrate (Phase 2): no life / graveyard rider on
        // synthetic-state printings. Back-compat defaults.
        warp_additional_life_cost: 0,
        warp_castable_from_graveyard: false,
        // legal_decisions mana-awareness Slice 3: synthetic-state
        // printings carry no cast-time additional cost. Back-compat
        // defaults (the 4 EoE additional-cost cards are all
        // bridge-DECLINED — verified-zero live incidence).
        additional_cost_sacrifice_creature_or_artifact: false,
        additional_cost_sacrifice_creature_or_enchantment: false,
        additional_cost_choose_your_creature: false,
        // Aura-attachment substrate — Phase 1 STEP 2 (round-78):
        // synthetic-state printings are not Auras. Back-compat
        // defaults (every pre-substrate registry dict omits these
        // keys, parser maps to the same defaults).
        is_aura: false,
        aura_enchant_type: "creature".to_string(),
        aura_yours_only: false,
        aura_power_bonus: 0,
        aura_toughness_bonus: 0,
        aura_granted_keywords: Vec::new(),
        // Dynamic-P/T formula substrate (round-78): synthetic-state
        // printings have no dynamic P/T. Back-compat default (every
        // pre-substrate registry dict omits the `dynamic_pt_formula`
        // key, parser maps the absence to None).
        dynamic_pt_formula: None,
        // Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20):
        // synthetic-state printings are not S2 Auras. Back-compat
        // defaults — every pre-S2 registry dict omits these keys.
        aura_grants_control: false,
        aura_grants_lose_abilities: false,
        aura_set_base_pt: None,
        // Phase 2 S2 extension (Tractor Beam closure): synthetic-state
        // printings are not don't-untap Auras.
        aura_prevents_host_untap: false,
        // Phase 4 S3 (Moonlit Meditation substrate landing,
        // 2026-05-20): synthetic-state printings carry no
        // token-creation replacement rider. Back-compat default.
        aura_replaces_first_tokens_with_copies: false,
        // Static / layered abilities substrate — Phase 1 first-
        // deliverable (round-79): synthetic-state printings have no
        // anthem statics. Back-compat default (every pre-substrate
        // registry dict omits the `static_anthems` key).
        static_anthems: Vec::new(),
        // Static / layered abilities substrate — Phase 2 (round-80,
        // 2026-05-20): synthetic-state printings have no
        // uncounterable flags. Back-compat default.
        spell_is_uncounterable: false,
        grants_creature_spells_uncounterable: false,
        // Static / layered abilities substrate — Phase 3 (round-81,
        // 2026-05-20): synthetic-state printings have no SELF-scope
        // conditional bonuses. Back-compat default.
        static_self_bonuses: Vec::new(),
    });
    // Oversimplification: real Llanowar Elves has the activated
    // ability "{T}: Add {G}". The synthetic state's first-slice
    // shim treats `mana_produced` on a creature the same way as on
    // a land — the bridge synthesis now produces the same shape
    // for both (a synthesized ActivatedAbility with tap_self=true,
    // AddManaToPool effect, is_mana_ability=true). The creature
    // sickness gate (CR 302.1) on the new path correctly rejects
    // sick Llanowar Elves emissions.
    let elves_mana_produced = Some(vec!["G".to_string()]);
    let elves_activated = crate::cards::apply_mana_produced_bridge(
        &elves_mana_produced,
        Vec::new(),
    )
    .expect("bridge synthesis for Llanowar Elves");
    let elves_pid = builder.push_printing(CardPrinting {
        name: Arc::from("Llanowar Elves"),
        mana_cost: ManaCost {
            green: 1,
            ..ManaCost::default()
        },
        effects: Vec::new(),
        power: Some(1),
        toughness: Some(1),
        mana_produced: elves_mana_produced,
        card_types: vec!["Creature".to_string()],
        // Llanowar Elves' printed type line is "Creature — Elf Druid"
        // (CR 205.3m). The synthetic fixture omits "Druid" — only
        // "Elf" is load-bearing for current consumers, and the
        // synthetic state's Elves shim is already an oversimplification
        // (see the comment above on its activated-ability shape).
        subtypes: vec!["Elf".to_string()],
        // Round-30 supertypes substrate: Llanowar Elves has no
        // supertype prefix on its type line.
        supertypes: Vec::new(),
        keywords: Vec::new(),
        etb_effects: Vec::new(),
        triggered_abilities: Vec::new(),
        activated_abilities: elves_activated,
        starting_loyalty: 0,
        // warp_cost substrate (Phase 1, round-63): synthetic-state
        // printings have no warp. Back-compat default.
        warp_cost: None,
        // warp_cost substrate (Phase 2): no life / graveyard rider on
        // synthetic-state printings. Back-compat defaults.
        warp_additional_life_cost: 0,
        warp_castable_from_graveyard: false,
        // legal_decisions mana-awareness Slice 3: synthetic-state
        // printings carry no cast-time additional cost. Back-compat
        // defaults (the 4 EoE additional-cost cards are all
        // bridge-DECLINED — verified-zero live incidence).
        additional_cost_sacrifice_creature_or_artifact: false,
        additional_cost_sacrifice_creature_or_enchantment: false,
        additional_cost_choose_your_creature: false,
        // Aura-attachment substrate — Phase 1 STEP 2 (round-78):
        // synthetic-state printings are not Auras. Back-compat
        // defaults (every pre-substrate registry dict omits these
        // keys, parser maps to the same defaults).
        is_aura: false,
        aura_enchant_type: "creature".to_string(),
        aura_yours_only: false,
        aura_power_bonus: 0,
        aura_toughness_bonus: 0,
        aura_granted_keywords: Vec::new(),
        // Dynamic-P/T formula substrate (round-78): synthetic-state
        // printings have no dynamic P/T. Back-compat default (every
        // pre-substrate registry dict omits the `dynamic_pt_formula`
        // key, parser maps the absence to None).
        dynamic_pt_formula: None,
        // Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20):
        // synthetic-state printings are not S2 Auras. Back-compat
        // defaults — every pre-S2 registry dict omits these keys.
        aura_grants_control: false,
        aura_grants_lose_abilities: false,
        aura_set_base_pt: None,
        // Phase 2 S2 extension (Tractor Beam closure): synthetic-state
        // printings are not don't-untap Auras.
        aura_prevents_host_untap: false,
        // Phase 4 S3 (Moonlit Meditation substrate landing,
        // 2026-05-20): synthetic-state printings carry no
        // token-creation replacement rider. Back-compat default.
        aura_replaces_first_tokens_with_copies: false,
        // Static / layered abilities substrate — Phase 1 first-
        // deliverable (round-79): synthetic-state printings have no
        // anthem statics. Back-compat default (every pre-substrate
        // registry dict omits the `static_anthems` key).
        static_anthems: Vec::new(),
        // Static / layered abilities substrate — Phase 2 (round-80,
        // 2026-05-20): synthetic-state printings have no
        // uncounterable flags. Back-compat default.
        spell_is_uncounterable: false,
        grants_creature_spells_uncounterable: false,
        // Static / layered abilities substrate — Phase 3 (round-81,
        // 2026-05-20): synthetic-state printings have no SELF-scope
        // conditional bonuses. Back-compat default.
        static_self_bonuses: Vec::new(),
    });
    let registry = builder.snapshot();

    let p0 = make_player(
        0,
        num_battlefield,
        num_hand,
        num_library,
        0,
        forest_pid,
        bears_pid,
        elves_pid,
    );
    let id_offset = (num_battlefield + num_hand + num_library) as u32;
    let p1 = make_player(
        1,
        num_battlefield,
        num_hand,
        num_library,
        id_offset,
        forest_pid,
        bears_pid,
        elves_pid,
    );
    GameState {
        players: [p0, p1],
        registry,
        active_player: 0,
        priority_player: 0,
        turn_number: 5,
        phase: 0,
        attackers: Vec::new(),
        blockers: Vec::new(),
        stack: Vec::new(),
        // G7 S5a: empty spell-stack-source-zone store at construction;
        // filled by a cast pushing the spell hand → here, drained by
        // `move_source_post_resolve` at resolution.
        stack_objects: Vec::new(),
        // G7 Class-C delayed-trigger port: empty at construction;
        // populated by warp-cast (`cast_creature_warp`) scheduling
        // WarpExileSelf for the next end step, drained at PHASE_END_STEP
        // entry by `run_delayed_end_step_triggers`.
        delayed_triggers: Vec::new(),
        consecutive_passes: 0,
        permanent_left_battlefield_this_turn: false,
        spell_warped_this_turn: false,
        // Phase 4 S3 (Moonlit Meditation substrate landing,
        // 2026-05-20): per-player created-tokens-this-turn flags
        // start cleared at game construction. Both slots default
        // false; CreateToken flips the controller's slot on a
        // non-zero-count batch, turn-cleanup resets both.
        created_tokens_this_turn: [false; 2],
        pending_decisions: Vec::new(),
        next_etb_timestamp: 0,
        // R76-1: seed the monotonic id counter past every object
        // `make_player` created. p0 ids run [0, id_offset); p1 ids
        // run [id_offset, 2*id_offset). The next fresh id is
        // 2*id_offset = exactly 1 + the max initial id, the same
        // value Python's `next_object_id` holds after building the
        // equivalent population via `new_object`.
        next_object_id: id_offset.saturating_mul(2),
        resolving_combat_damage_amount: 0,
        resolving_combat_damage_target_player_id: u8::MAX,
        winner_id_sticky: None,
        // Effect-rng-threading closure: default `None`; cross-engine
        // smoke callers opt in via `set_external_rng(Some(rng))`.
        rng: None,
        // ETB-trigger Slice 1: empty ledger at construction; reset
        // each turn at cleanup.
        triggered_this_turn: std::collections::HashSet::new(),
        // ETB-trigger Slice 2: empty deferred-dispatch queue at
        // construction; drained by the surrounding resolver / move
        // primitive after the resolving slot is removed from the
        // stack.
        pending_etb_dispatches: Vec::new(),
        // ETB-trigger Slice 5: empty deferred-DIES queue at
        // construction; drained by `apply_sbas` after the SBA loop
        // completes.
        pending_dies_dispatches: Vec::new(),
        // ETB-trigger Slice 8: empty deferred-LEAVES queue at
        // construction; drained by the bf-exit primitive's caller
        // (move_object, apply_sbas, etc.) after the primitive
        // returns.
        pending_leaves_dispatches: Vec::new(),
        // Slice 14: empty deferred IS_SACRIFICED queue at
        // construction; drained by the surrounding resolver after
        // the resolving sacrifice spell's stack pop.
        pending_is_sacrificed_dispatches: Vec::new(),
        // Round-55: empty token-cease queue at construction; filled
        // by `mark_left_battlefield_if_nonland` on a token bf-exit,
        // drained at the tail of `drain_pending_leaves_dispatches`
        // and `apply_sbas`.
        pending_token_cease: Vec::new(),
        // CE20 follow-up Slice 4: transient payload-target slot;
        // set by the stack resolver before apply_effect for
        // targeted activations, restored after. None by default.
        current_resolve_payload_target_id: None,
        // Spell-card-bridge lever Phase 2: the second resolve-time
        // payload-target slot (2-slot spells, e.g. Diplomatic
        // Relations). None by default — single-target spells /
        // triggers / activations never set it.
        current_resolve_payload_target_id_2: None,
        // CE20 Slice 5 (Walker loyalty Slice C): empty
        // once-per-turn-per-walker ledger at construction; reset
        // each turn at cleanup.
        loyalty_activated_this_turn: std::collections::HashSet::new(),
        // DrawCards controller-of-effect fix: transient slot, None
        // until the resolver swaps in `top.controller` for the
        // duration of an apply_effect call.
        current_resolve_source_controller: None,
        // ModalChooseOne envelope — Cycle 2: transient mode-index
        // slot. `None` until the resolver swaps in
        // `top.chosen_mode_idx` for the duration of an apply_effect
        // call (the resolve arm reads it; the swap is save → set →
        // restore-bracketed like its payload-target sibling).
        current_resolve_chosen_mode_idx: None,
        // Close Encounter additional-cost-choice envelope —
        // Cycle 1: transient choice-id slot. `None` until the
        // resolver swaps in `top.additional_cost_choice_id` for
        // the duration of an apply_effect call (the
        // `Effect::CloseEncounterEffect` resolve arm — Cycle 3 —
        // reads it; the swap is save → set → restore bracketed
        // like its mode-index sibling).
        current_resolve_additional_cost_choice_id: None,
        // scour-for-scrap-slice Cycle 1 widen: transient multi-pick
        // mode-index slot. `None` until the resolver swaps in
        // `top.chosen_mode_idxs` for the multi-pick (or_both=True)
        // dispatch window.
        current_resolve_chosen_mode_idxs: None,
        // X-value subsystem Cycle 1: transient X-value slot. `None`
        // until the resolver swaps in `top.x_value` for the duration
        // of an apply_effect call (Cycle 3+ will wire the
        // `Effect::ZeroPointBallad` / `Effect::DevastatingOnslaught`
        // resolve arms to read it).
        current_resolve_x_value: None,
    }
}
