# Subsystem-blocked tier — leverage inventory + closure sequence

A focused, current-snapshot **synthesis** of the BLOCKED-subsystem-gated
tier (142 cards post-dynamic-pt-cosmogoyf-closure, the **only**
non-empty BLOCKED tier on `docs/POOL_PARITY.md` after the
2026-05-19/20 massive-closure session). This doc is **not the
authoritative living roadmap** — `docs/SUBSYSTEM_ROADMAP.md` is
that, with its full cycle-by-cycle orchestrator history. This doc
is the short "what should I grind next, why, in what order"
reference.

**Snapshot HEAD:** post-`feature/dynamic-pt-cosmogoyf` merge (round-84,
2026-05-20: Cosmogoyf flipped PROBED on the SAME dispatch table
substrate as round-78 Grovestrider; blocked tier 143→142).
Predecessor was `7a3a0fc9` (post-Mechan Shieldmate closure / Phase
3.5 static abilities — round-82).
**Source data:** `scripts/scout_r58_subsystem_attribution.py` (re-runnable;
the same attribution helper `measure_pool_parity` imports, so this doc
and `POOL_PARITY.md` cannot disagree on the count).
**Tier under analysis:** `BLOCKED — subsystem-gated / Effect-uncovered`
(146 cards). The other two BLOCKED tiers were both driven to **0** this
session — see `docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md` for the
historical itemization (genuinely_missing 2→0 via Pain for All + cast-
from-exile substrate; blocked_non_effect 2→0 via Tractor Beam + Dauntless
Scrapbot).

## 0. This session's progress summary (2026-05-19/21)

The session that started after `ce564786` (the original inventory)
shipped ~22 substrate / closure merges. The original inventory's top-5
recommendations are now overwhelmingly **executed**, not pending. The
2026-05-21 loader_admits-attribution-sweep then closed a residual
measurement gap: the metric helper at
`scripts/measure_pool_parity.loader_admits` was only consulting 4 of
the 9 disjuncts in the real production-deck loader's bridge admit
gate. Folding the remaining 5 disjuncts in + 9 `_parity`-named
delegator probes flipped 9 cards BLOCKED → PROBED (the 5 shocklands +
Brightspear / Cloudsculpt / Gravblade / Mechan Shieldmate) with NO
engine changes — a pure measurement-attribution close, same shape as
the round-83 warp_cost Phase 3 close. The 2026-05-21
tannuk-frenzied-baloth-probes follow-up lane then closed the two
COVERED-pending carriers by writing the missing genuine real-EoE
cross-engine byte-equal probes: Tannuk Steadfast Second (effective-
keyword anthem haste-grant byte-equal on the genuine printing) and
Frenzied Baloth (`spell_is_uncounterable` self-flag + battlefield-
grant `grants_creature_spells_uncounterable` static byte-equal on the
genuine printing). Both flipped COVERED → PROBED, completing the
static-abilities Phase 1+2 measurement attribution.

### Audit lane (2026-05-21, post-probe-batches-HHHH-JJJJ): negative finding

The `feature/blocked-tier-stale-claim-audit` lane, run after the
post-loader_admits-sweep+tannuk-baloth-probes work, performed a
top-to-bottom audit of the 128-card BLOCKED-subsystem tail looking
for the same shape of measurement-attribution gaps that Lane F (Aura
flash widening, +2 PROBED) and Lane H (loader_admits disjunct sweep,
+9 PROBED + 2 COVERED honest-stops) had previously surfaced. The
audit covered:

1. **Loader admit predicate completeness** — all disjuncts of the
   real `trajectory._filter_deck_to_bridge_supported` admit gate are
   mirrored in `measure_pool_parity.loader_admits`. At the time of
   the audit there were 9 disjuncts (basic land, nonbasic mana land,
   pure warp creature, simple creature, simple spell, simple aura,
   simple static-anthem creature, simple uncounterable creature,
   simple static self-conditional creature); round-85's Phase-2
   non-basic-land closure (`feature/non-basic-land-slice`) added a
   10th disjunct (`_printing_is_bridge_simple_tapped_mana_land`,
   admitting the 5 EoE Planet lands) and wired it into
   `loader_admits` in the same commit, preserving the audit's
   "no missing disjuncts" invariant.

2. **`classify_pool` internal consistency** — no card is flagged
   BLOCKED-subsystem while simultaneously being (a) loader-admitted
   and (b) all-Effects-covered: 0 of 128 BLOCKED cards would have
   qualified for COVERED under the existing tier rules. The
   precedence is honest.

3. **Per-card spot-check of single-gated cards** — verified the
   genuine Rust-side absence for each named gate row:
   - Static / layered (2 single-gated: Rampaging Soulrager,
     Wildfire Wickerfolk) — both honest-stopped at
     `rust/src/cards.rs:6831` (Conditions `UnlockedDoorsAmongRoomsAtLeast`
     and `DeliriumActive` not in dispatch table); doc wording
     "not in EoE snapshot" is technically accurate (both come from
     `data/duskmourn.json`, not `data/edge_of_eternities.json`) but
     they ARE in the parity pool — closure requires a per-Condition
     Rust slice regardless.
   - Aura row (1 single-gated: Moonlit Meditation) — genuinely
     blocked at `trajectory.py:4855` on the
     `replaces_first_tokens_with_copies` rider; the per-player
     `created_tokens_this_turn` substrate landed `4dae568e`, but the
     supertype-override mechanism that gates the loader widen is
     still in flight on `feature/aura-supertype-override`.
   - Non-basic land row (18 → 13 single-gated, post round-85) —
     `_printing_is_bridge_nonbasic_mana_land` admits the 5 shocklands
     and the new sibling `_printing_is_bridge_simple_tapped_mana_land`
     (round-85) admits the 5 Planet lands (the parser-flattened
     `enters_tapped=True` mono-color shape; the bridge drops
     `enters_tapped` on both engines — round-39 Ward / round-60
     shockland symmetric-drop precedent). The remaining 13 carry
     painland (parser-structured conditional-tapped — a real
     cross-engine divergence) / clone (Echoing Deeps) / Exalted-static
     (Cathedral of War) / activated-abilities (Secluded Starforge)
     riders that are genuinely unbridged and stay DEFERRED.
   - Cost-reduction row (5 single-gated) — verified each printing
     carries a `cost_reductions` / `external_cost_reductions` tuple
     with no Rust counterpart (substrate documented DEFERRED).
   - Station / Spacecraft 1g (Galvanizing Sawship) — genuine
     spacecraft with `station_threshold=3`; documented DEFERRED.
   - Effect-uncovered 1g spot-check (10/24 sampled) — each emits a
     distinct Python Effect class with NO Rust variant
     (`grep` confirmed against `rust/src/state_ops_resolve.rs` +
     `rust/src/cards.rs`). The diffuse tail is real.

**Conclusion: no stale claims found in the inventory body or the
classifier.** The post-loader_admits-sweep+tannuk-baloth-probes state
is honest and tight — every BLOCKED card is genuinely blocked, every
classifier decision is internally consistent. (Subsequent 2026-05-21
unclassified-residual lane: the audit DID surface a doc-level
attribution gap — the scout's `_gates_for_loader_card` enumeration
only consulted structured `static_abilities` objects, not
printing-level bool flags like
``restricts_noncreature_spells_to_one_per_turn``, so Deafening
Silence was attributed `unclassified` rather than the more honest
`static_rule_text`. The unclassified-residual lane added the new
gate row and moved Deafening Silence; Leyline of Mutation's
`unclassified` attribution is confirmed-honest by the same hand
re-walk.) The only updates the
audit produced are doc-cosmetic refreshes: the §0 before→after
table now carries the post-probe-batches-HHHH-JJJJ column (PROBED
252 → 259), and the §3(2) Aura Phase 4 S3 row + §7 cross-ref are
updated to reflect that the `feature/aura-phase-4-s3` substrate
merged `4dae568e` (consumer wire-up still gated on the supertype-
override mechanism). Headline counts unchanged.

### Top-5-recommendations status (from `ce564786`)

| Original rec | Status | Closure commit(s) |
|---|---|---|
| (1) Static / layered abilities — close 8 single-gated | **PARTIAL** — Phases 1+2+3 SHIPPED; 6 of 8 single-gated cards PROBED (Brightspear Zealot, Cloudsculpt Technician, Gravblade Heavy, Mechan Shieldmate via loader_admits sweep; Tannuk Steadfast Second, Frenzied Baloth via the tannuk-frenzied-baloth-probes follow-up) | `3a7c483c` Phase 1 / `9ccefe08` Phase 2 / `ddd93684` Phase 3 / loader_admits sweep / tannuk-frenzied-baloth-probes |
| (2) Dynamic-P/T formula — Harmonious Grovestrider | **CLOSED** — substrate LANDED round-78; Grovestrider PROBED end-to-end (ward-rider on real printing kept it loader-gated, but the formula-substrate single-gate it cited is closed). Round-84 (`feature/dynamic-pt-cosmogoyf`) extended the SAME dispatch table with the `"cosmogoyf"` arm, flipping Cosmogoyf → PROBED. Single-gated Dynamic-P/T tier now 0. | `6a8b6402` (Grovestrider) + this lane (Cosmogoyf) |
| (3) Aura Phase 2 (S2 control-changing) | **SHIPPED** — Duskmourn admit + Tractor Beam reduce | `643163c6`, plus S2 extension `d4ceecc5` (Tractor Beam) |
| (4) Aura Phase 3 (S4 triggered) | **SHIPPED** — Pain for All admit substrate end-to-end | `35ede8b0` |
| (5) Effect-uncovered per-card drain | continuing (probe batches AAAA-IIII landed in parallel; +18 PROBED) | `5063e8e3`, `93ea4057`, `1a06d714`, … `7c66225e`, … `feature/probe-batch-hhhh`, `feature/probe-batch-iiii` |

### Other major closures this session

* **Pain for All** (`60fea01b`) — closed last BLOCKED-genuinely_missing
  Aura residue via S4 substrate.
* **Tractor Beam** (`d4ceecc5` + `62222ffe`) — closed last
  BLOCKED-blocked_non_effect Aura residue + bench-admit relaxed at cast
  time.
* **Dauntless Scrapbot** (`3af7edea`) — closed last
  BLOCKED-blocked_non_effect carrier (`ExileEachOpponentsGraveyard`).
* **Lightstall Inquisitor** (`9c41ad08` + `00d01937`) — cast-from-exile
  substrate end-to-end + worklist reconciliation.
* **Aura Phase 2 S2.5** (`46d8d393`) — layer-7b set-base-PT reader fold
  (Lignify-shape Auras).
* **Multi-Aura set-base-PT timestamp interleaving** (`959d0766`) — CR
  613.3b ordering for stacked set-PT Auras.
* **Static abilities Phase 2** (`9ccefe08`) —
  `grants_creature_spells_uncounterable` flag (Frenzied Baloth).
* **Static abilities Phase 3** (`ddd93684`) — SELF-scope conditional
  bonus (Brightspear Zealot + Cloudsculpt + Gravblade Heavy).
* **House Cartographer** (`ef8e9d0c`) — bespoke survival effect closed.
* **Sothera + Requiem Monolith** (`5f56ac56`) — 6-Effect-variant slice.
* **Roving Actuator** (`444ae155`) — slice 18 ETB substrate.
* **Meltstrider's Resolve** (`9155ea5c`) — slice 19 Aura-Fight scaffold.
* **All 3 broader-archetype residual axes closed** (`5bca387a` and
  predecessors) — cumulative 23/1500 → 0 divergences; v13 post-session
  resweep (`3e37882f`) 300/300 PASS regression-verifies every substrate.

### POOL_PARITY headline before → after this session

```
                                                  ce564786  →  62222ffe  →  post-warp-P3  →  post-dyn-PT-cosmo  →  post-aura-P1-attr  →  post-loader-admits-sweep  →  post-tannuk-baloth-probes  →  post-probe-batches-HHHH-JJJJ  →  post-non-basic-land-P2  →  post-cut-propulsion  →  post-shattered-wings  →  post-emergency-eject
PROBED                                                221         232          235            236                 238                    250                              252                               259                            264                       266                       267                      269
covered-by-construction                                95          97           97             97                  98                     97                               95                                88                             89                        88                        88                       88
BLOCKED — genuinely_missing                             3           0            0              0                   0                      0                                 0                                 0                              0                         0                         0                        0
BLOCKED — blocked_non_effect                            2           0            0              0                   0                      0                                 0                                 0                              0                         0                         0                        0
BLOCKED — subsystem-gated                             154         146          143            142                 139                    128                               128                               128                            122                       121                       120                      118
```

**post-emergency-eject delta (this lane):** strict **+1 PROBED /
−1 BLOCKED-subsystem-gated** (Emergency Eject, 6,291 17lands
single-gated decks per the round-58 attribution scout). Note: the
post-shattered-wings column on the snapshot table above reads
``267 / 120`` (off-by-one from the canonical
``docs/POOL_PARITY.md`` at the same HEAD which reads ``268 / 119``
— a pre-existing measurement-drift carried over from the
shattered-wings landing); the canonical post-this-lane numbers
that match ``docs/POOL_PARITY.md`` at this HEAD are PROBED ``269``
/ BLOCKED-subsystem ``118``. Genuine
Rust substrate close following the same Lander-pid carrier shape
as Seedship Impact / Orbital Plunge: new
`Effect::EmergencyEjectEffect { token_printing_id }` single-pid
variant (`rust/src/cards.rs`) + parse arm + to_pydict arm;
resolve arm in `rust/src/state_ops_resolve.rs` beside the
Seedship Impact arm (cast-time-bound target via
`current_resolve_payload_target_id`; non-land filter re-checked
at resolve per CR 608.2b; indestructible-skip on the scan; DIES
creature-gated; LEAVES + token-cease drained); the
`current_resolve_payload_target_id` self-scan fallback for
synthetic-test no-target push. Three deviations from Seedship
Impact (load-bearing for parity): (1) filter widens from
artifact-OR-enchantment to non-land permanent; (2) **no MV
gate** — Lander rider is unconditional whenever the destroy
succeeds; (3) **Lander goes to the FORMER controller of the
destroyed permanent** (NOT the source-controller — captured
BEFORE destroy from `obj.controller`). `state_ops_legal.rs::
spell_targets_satisfiable` non-land-permanent + non-
indestructible arm; cast-time `is_emergency_eject` bind in
`state_ops_cast.rs` mirroring `is_seedship_impact`. Bridged via
`cross_engine.py` `emergency_eject_effect` (drops the pid,
constructs the field-free Python
`DestroyTargetNonlandPermanentWithLanderForController`) +
`trajectory.py::_python_spell_effect_to_dict` Lander-pid leaf
emitting symbolic `_token_card_def` (resolved by the seating
pass once `"emergency_eject_effect"` is added to
`_SPELL_TOKEN_PID_KINDS`) — the {2}{W} Emergency Eject Instant
auto-joins the spell-card-bridge admit set (10→11). 8 new
probes: 6 Rust pins (opp-artifact-destroy + Lander-for-opp,
creature-destroy + unconditional Lander, enchantment-destroy +
Lander, indestructible-survives + no-Lander, Land-filtered-out +
silent fizzle, deterministic-first-non-land-pick-skipping-Land),
1 auto-join admit-set pin, 1 GENUINE real-EoE cross-engine
byte-equal probe driving the parsed printing against a 3/3
mv-3 opponent creature → destroyed + Lander under p1 (former
controller) byte-equal on BOTH engines. Round-69 fixture
(`test_r69_max_turns_raise_is_parity_safe_and_recovers_timeout_signal`)
re-derived per its own re-derivation contract: arch=9 seed=1's
bench trajectory legitimately changed (it contains Emergency
Eject); the new fixture is arch=7 seed=0 (also a cap-25
FAIL_TIMEOUT that recovers to PASS at cap-60).

**post-shattered-wings delta (prior lane):** strict **+1 PROBED /
−1 BLOCKED-subsystem-gated** (Shattered Wings, 7,793 17lands
single-gated decks per the round-58 attribution scout). Genuine
Rust substrate close following the Cut Propulsion Tier-A precedent
(`feature/effect-uncovered-slice` / `e2e45b46`): new
`Effect::DestroyTargetArtifactEnchantmentOrFlyingCreature`
field-free unit variant (`rust/src/cards.rs`) + parse arm +
to_pydict arm; resolve arm in `rust/src/state_ops_resolve.rs`
beside the OrTapped sibling (cast-time-bound target via
`current_resolve_payload_target_id`; 3-way OR predicate
re-checked at resolve per CR 608.2b; flying leg reads EFFECTIVE
keywords via `battlefield_has_keyword`; indestructible-skip on
the scan; DIES creature-gated; LEAVES + token-cease drained);
`state_ops_legal.rs::spell_targets_satisfiable` 3-way OR arm;
bridge round-trip in `cross_engine.py` + the
`trajectory.py::_python_spell_effect_to_dict` reverse-serializer
adds two leaves (destroy + Surveil — the outer Sequence's
children) so the {2}{G} Shattered Wings Sorcery auto-joins the
spell-card-bridge admit set (6→9 across Cut Propulsion +
Shattered Wings). 7 new probes: 5 Rust pins (artifact / enchant /
flying-creature destroys + non-flying-creature silent fizzle +
indestructible-survives), 1 auto-join admit-set pin, 1 GENUINE
real-EoE cross-engine byte-equal probe driving the parsed
printing against a 2/2 flyer → destroyed → in graveyard
byte-equal on BOTH engines.

**post-cut-propulsion delta:** strict **+1 PROBED / −1
BLOCKED-subsystem-gated / −1 covered-by-construction** (Cut
Propulsion, 11,990 17lands single-gated decks). Genuine Rust
substrate close (Tier-A precedent — Biosynthic Burst / Focus
Fire / Diplomatic Relations): new `Effect::CutPropulsionEffect`
field-free unit variant + parse + to_pydict + resolve (effective-
power read with counters-inclusive formula; flying-doubler via
`battlefield_has_keyword`; saturating damage apply; receiver-
side DEALT_DAMAGE event dispatched). Bridged via `cross_engine.py`
`cut_propulsion_effect` + `trajectory.py` reverse-serializer
leaf — the {2}{R} Cut Propulsion Instant auto-joins the
spell-card-bridge admit set.

**post-non-basic-land-P2 delta (this lane):** strict **+5 PROBED /
−5 BLOCKED-subsystem-gated** (the 5 EoE Planet lands: Adagia,
Windswept Bastion / Evendo, Waking Haven / Kavaron, Memorial World /
Susur Secundi, Void Altar / Uthros, Titanic Godcore; total 7,617
17lands single-gated decks per the round-58 attribution scout). One
new admit disjunct (`_printing_is_bridge_simple_tapped_mana_land`)
plus 5 `_parity`-named PROBED-tier delegator probes. The 5 Planet
lands are 0-Effect cards admitted by the new disjunct; the
`_parity`-named delegators credit them PROBED-tier directly in the
same commit, so COVERED is unchanged. No engine changes; round-39
Ward / round-60 shockland symmetric-drop precedent (the bridge
drops `enters_tapped` on both engines so the bridged shape is
bit-equal; the parser dropped Station + the tier-gated activated
ability before either engine saw the printing — also symmetric).
Note: the prior `post-probe-batches-HHHH-JJJJ` snapshot column
recorded `COVERED=88 / BLOCKED=128` but the live POOL_PARITY at
the start of this lane was `COVERED=89 / BLOCKED=127` (off-by-one
COVERED↔BLOCKED drift in the historical column, undetected by the
round-tail audit lane). The post-this-lane snapshot is the
authoritative `264 / 89 / 122` row of the regenerated
`docs/POOL_PARITY.md`. Spec: `SEARCH_LIMITATIONS.md` "5 EoE Planet
lands admitted as plain untapped mono-color lands".

The tannuk-frenzied-baloth-probes lane delta is strict **+2 PROBED / −2 COVERED**: Tannuk Steadfast Second and Frenzied Baloth flipped COVERED → PROBED via the two new genuine-printing cross-engine byte-equal `_parity` probes. No engine changes, no subsystem-gated movement, no Effect-class movement.

The **post-probe-batches-HHHH-JJJJ** delta is strict **+7 PROBED / −7 COVERED** (probe batches HHHH `25d89e4c`, IIII `07718a2f`, JJJJ `a31b666c` landed `_parity` delegators that promoted covered cards to PROBED): Lightstall Inquisitor, Seriema, Optimistic Scavenger, Command Bridge (probe batch IIII), three Room cards (probe batch JJJJ via the triggered-ability surface delegators). No engine changes, no subsystem-gated movement. **The 128 subsystem-gated tail has been stable since the loader_admits sweep — this audit lane confirmed no further measurement-attribution gaps in the BLOCKED tier.**

The **post-unclassified-residual-attribution** lane delta (2026-05-21, this lane) is strict **+0 PROBED / +0 COVERED / +0/−0 BLOCKED-subsystem-gated** — the 122 subsystem-gated count is preserved. The lane is a SCOUT-side measurement-attribution close on the Unclassified residual: Deafening Silence (the only EoE printing that carries the parser-recognised ``restricts_noncreature_spells_to_one_per_turn`` bool) moved from the ``unclassified`` gate to a new named row ``static_rule_text`` (printing-level bool fields the Python parser already models but Rust does not mirror). Leyline of Mutation remains in ``unclassified`` (its parser TRULY does not surface either of its two oracle mechanics — opening-hand leyline + 5-color alternative-cost). The hand re-walk this row asked for is complete; both cards now carry a named substrate-cost characterisation in §5 and a divergence pin in ``docs/SEARCH_LIMITATIONS.md``. No engine changes, no Rust changes, no bridge changes — same Lane H / Lane Q precedent (the metric helper and the actual engine never disagreed; only the scout's taxonomy gained a row).

Subsystem-gated dropped by 26 total; **both itemized BLOCKED tiers
reached 0**. The 128 tail is now the *only* non-empty BLOCKED tier.
Round-83 (2026-05-20: warp_cost Phase 3) flipped Red Tiger Mechan,
Bygone Colossus, Timeline Culler → PROBED on the existing Phase 1/1.5/2
substrate via two measurement-attribution fixes (loader_admits widen +
`_parity`-named probe aliases). Round-84 (2026-05-20:
dynamic-pt-cosmogoyf) flipped Cosmogoyf → PROBED via a one-arm
extension to the existing round-78 dispatch table. Aura Phase 1
flash widening attribution close (2026-05-21) flipped Frantic Strength
+ Shardmage's Rescue → PROBED via the same measurement-attribution
shape. The 2026-05-21 loader_admits-attribution-sweep then folded the
5 remaining disjuncts of the real loader into the metric helper +
added 9 `_parity` aliases — 9 more cards flipped BLOCKED → PROBED
(Breeding Pool / Godless Shrine / Sacred Foundry / Stomping Ground /
Watery Grave / Brightspear Zealot / Cloudsculpt Technician /
Gravblade Heavy / Mechan Shieldmate; Tannuk Steadfast Second +
Frenzied Baloth honest-stopped at COVERED, then flipped to PROBED
by the tannuk-frenzied-baloth-probes follow-up the same day).

## 1. Headline (current state)

* **128 cards** are subsystem-gated; **53 single-gated**, **75 multi-gated**
  (post loader_admits-attribution-sweep 2026-05-21: 9 cards flipped
  BLOCKED → PROBED via measurement-attribution close — 5 shocklands +
  Brightspear / Cloudsculpt / Gravblade / Mechan Shieldmate; Tannuk
  Steadfast Second + Frenzied Baloth flipped BLOCKED → COVERED
  pending real-EoE cross-engine byte-equal helpers — see lever (1)
  below). The 75 multi-gated total is unchanged.
* **16 taxonomy gates** in the scout's `_gates_for_loader_card` (post
  2026-05-21 unclassified-residual attribution close — the new
  `Static rule-text (printing-level bool field, no Rust mirror)` row
  catches Deafening Silence's ``restricts_noncreature_spells_to_one_
  per_turn``; the existing 15 gates are unchanged). Of those,
  **14 currently bind ≥1 card** — `BECOMES_TAPPED event` /
  `Planeswalker loyalty` / `Rooms / doors` / `Kicker / Impending` bind
  cards but **0 single-gated**; `Carrier activated abilities` and
  `Non-safe triggered ability` bind 30 / 43 cards but **0 single-gated**.
  These are co-blocker rows, not levers.
* A card may trip multiple gates; `#cards` does NOT sum to 128. The
  honest leverage column remains `decks(1g)`.

## 2. Subsystems by closure leverage (refreshed)

Columns are independent. **Tractability** is the roadmap classification;
**Closure scope** is this doc's per-substrate cost estimate, **revised**
to reflect what substrate already exists post-session.

| Subsystem | #cards | 1gate | decks(1g) | Tractability | Closure scope | Status (post-session) |
|---|---:|---:|---:|---|---|---|
| Static / layered abilities | 9 | 2 | 0 | sprawling | small per remaining card (Phases 1+2+3+3.5 substrate built) | **PARTIAL** — round-82: Mechan Shieldmate CLOSED (Phase 3.5); 2026-05-21 loader_admits-attribution-sweep flipped Brightspear / Cloudsculpt / Gravblade / Mechan Shieldmate → PROBED + Tannuk / Frenzied Baloth → COVERED; 2026-05-21 tannuk-frenzied-baloth-probes follow-up flipped Tannuk + Frenzied Baloth COVERED → PROBED via genuine-printing cross-engine byte-equal helpers; 2 single-gated remain (Soulrager / Wickerfolk — not in EoE snapshot) |
| Non-basic land substrate | 13 | 13 | 8,893 | medium | 13 deferred behind sibling substrates (conditional-tapped / clone / Exalted / activated-abilities) | **PARTIAL** — 2026-05-21 loader_admits-attribution-sweep flipped the 5 shocklands → PROBED (round-60); 2026-05-21 non-basic-land Phase-2 (`feature/non-basic-land-slice`) flipped the 5 Planet lands (Adagia / Evendo / Kavaron / Susur Secundi / Uthros, 7,617 decks) → PROBED via the new `_printing_is_bridge_simple_tapped_mana_land` predicate + round-85 arch-65 bench probe + 5 `_parity`-named PROBED-tier delegators; 13 non-shockland non-Planet lands remain DEFERRED (10 Duskmourn painlands' parser-structured `enters_tapped_unless_any_player_life_at_most` is a real cross-engine divergence; Secluded Starforge / Cathedral of War / Echoing Deeps each behind activated-abilities / Exalted-static / clone substrates) |
| Cost-reduction substrate | 5 | 5 | 53,648 | medium | large (true Rust multi-round; Condition-grammar port) | DEFERRED (unchanged; `docs/COST_REDUCTION_SUBSTRATE_PLAN.md`) |
| warp_cost | 7 | 0 | 0 | medium | Phase 1/1.5/2/3 **SHIPPED** — 3 single-gated Phase-3 cards flipped to PROBED (round-83, 2026-05-20: `feature/warp-cost-phase-3`) | **CLOSED for single-gated** — 7 multi-gated tail remains (static/effect-uncovered co-gates) |
| Dynamic-P/T formula | 0 | 0 | 0 | medium | substrate landed `6a8b6402`; Grovestrider closed round-78; Cosmogoyf closed round-84 (this lane: cosmogoyf-arm extension on the SAME dispatch table) | **CLOSED for single-gated** — Emissary Escort honest-stopped Rust-side (different shape: printed-base + X-from-other-artifact-MV, not CDA-only; absent from EoE snapshot) |
| Station / Spacecraft | 8 | 1 | 5,938 | sprawling | large (§2-declined) | DEFERRED (unchanged) |
| Carrier activated abilities | 30 | 0 | 0 | sprawling | sprawling (per-card; 0 single-gated) | DEFERRED (unchanged) |
| Aura-attachment substrate | 7 | 1 | 633 | sprawling | Phase 1/2/2.5/3 LANDED; **Phase 4 only** (S3 token-replacement Moonlit Meditation 633 decks) | **PARTIAL** — Phase 1 flash widening attribution close (2026-05-21) flipped Frantic Strength + Shardmage's Rescue → PROBED + Duskmourn's Domination → COVERED; only Moonlit Meditation single-gated remains |
| BECOMES_TAPPED event | 2 | 0 | 0 | contained | small (event bridge) — 0 single-gated | DEFERRED (unchanged) |
| Ward (declined) | dropped | — | — | medium | — | row gone from scout (0-binding) |
| Kicker / Impending | 1 | 0 | 0 | medium | medium — 0 single-gated | DEFERRED (unchanged) |
| Rooms / doors | 3 | 0 | 0 | sprawling | large — 0 single-gated, 0 decks | DEFERRED (unchanged) |
| Planeswalker loyalty | 1 | 0 | 0 | sprawling | large — 0 single-gated | DEFERRED (unchanged) |
| Effect-uncovered (diffuse tail) | 94 | 24 | 103,349 | diffuse | per-card | per-card lane continues |
| Non-safe triggered ability | 43 | 0 | 0 | diffuse | per-event-shape — 0 single-gated | DEFERRED (unchanged) |
| Static rule-text (printing-level bool) | 1 | 1 | 12 | medium | one-card slice (Rust bool + counter + filter mirror + bridge + new admit disjunct) | OPEN — Deafening Silence (2026-05-21 attribution split out of Unclassified; see §5 for rationale) |
| Unclassified (honest residual) | 1 | 1 | 0 | diffuse | per-card hand re-walk | OPEN — Leyline of Mutation (parser does not recognise either of its mechanics; see §5) |

**Reconciliation:** 128 subsystem-gated = 53 single-gated + 75
multi-gated; `1gate` column sums to 53 (post 2026-05-21 loader_admits-
attribution-sweep; the Static/layered-abilities row drops from 15/8/34526
to 9/2/0 — Brightspear / Cloudsculpt / Gravblade / Mechan Shieldmate
flipped to PROBED, Tannuk Steadfast Second + Frenzied Baloth flipped
to COVERED then PROBED via the same-day follow-up writing genuine-
printing cross-engine byte-equal helpers. The Non-basic-land row drops
from 23/23/34575 to
18/18/16510 — the 5 shocklands flipped to PROBED via the round-60
bench-probe pin. Post-Aura-Phase-1-flash-widening close 2026-05-21:
the Aura row at 4/1/633 — Frantic Strength + Shardmage's Rescue +
Duskmourn's Domination dropped out of subsystem-gated; only Moonlit
Meditation remains single-gated, on the un-shipped Phase 4 S3
consumer wire-up. Post-dynamic-pt-cosmogoyf round-84; the Dynamic-P/T
row is 0/0/0 — the substrate is fully closed end-to-end for the two
formulas present in the EoE snapshot (Grovestrider + Cosmogoyf); the
third sibling, Emissary Escort, is absent from the EoE snapshot AND
is honest-stopped Rust-side on a different per-formula shape.

## 3. The remaining highest-leverage levers (re-ranked)

Filtered to substrates that are **(a) named-and-bounded** and **(b) not
already fully closed**. Re-ranked by **decks(1g) ÷ remaining-closure-
scope** with the substrate already-paid-for discounts the session
introduced.

### (1) Static / layered abilities — Phase 3.5 SHIPPED (round 82, Mechan Shieldmate closure)

**Status: high-leverage named lever, near-complete.** Phases 1+2+3
SHIPPED (`3a7c483c` / `9ccefe08` / `ddd93684`); Phase 3.5 SHIPPED
round-82 (Mechan Shieldmate). 6 of the 8 originally-single-gated
cards are closed at metric-tier level: 4 PROBED (Brightspear /
Cloudsculpt / Gravblade / Mechan Shieldmate — all flipped to PROBED
by the 2026-05-21 loader_admits-attribution-sweep via thin `_parity`
delegators that re-use the existing `tests/test_static_layered_
abilities_phase3.py::test_real_eoe_byte_equal_*` cross-engine pins)
and 2 originally COVERED → PROBED (Tannuk Steadfast Second + Frenzied
Baloth — the 2026-05-21 loader_admits-attribution-sweep widened the
metric helper to admit them through the loader filter but they lacked
a genuine real-EoE cross-engine byte-equal helper; the static-anthem
Phase-1 / uncounterable Phase-2 modules already exercised SYNTHETIC
`CardDef` carriers, so the same-day tannuk-frenzied-baloth-probes
follow-up lane added the two missing genuine-printing probes
(`test_cross_engine_real_eoe_tannuk_steadfast_second_static_anthem_parity`
exercises the haste anthem grant byte-equal across both engines on the
genuine parsed printing;
`test_cross_engine_real_eoe_frenzied_baloth_static_uncounterable_parity`
exercises both `spell_is_uncounterable` and
`grants_creature_spells_uncounterable` byte-equal). The remaining
single-gated cards under this gate row are:

* **Rampaging Soulrager** — `UnlockedDoorsAmongRoomsAtLeast(2)`
  Condition. Not in the current EoE data snapshot. Would need a
  per-Condition Rust port riding the now-shipped Phase 3 + 3.5
  substrate. **~1 cycle of slice work**.
* **Wildfire Wickerfolk** — `DeliriumActive` Condition + a layer-6
  trample grant + a layer-7 +1/+1. Not in the current EoE data
  snapshot. ~1 cycle of slice work.

**Multi-gated 7 (unchanged):** Exalted Sunborn, Loading Zone, Memorial
Team Leader, Starfield Vocalist (warp+static); Lumen-Class Frigate
(Station+static); Nexus of Fate (effect-uncovered+nonsafe-trig+static);
Green Sun's Zenith (effect-uncovered+static).

* **Why it's still on the list:** the substrate's been built; each
  remaining card is now small scope, not the original medium-large.
* **Closure scope per remaining card:** small (existing layer
  machinery + 1 new Condition tag).
* **Sequencing:** the 2 remaining single-gated are blocked on an EoE
  snapshot refresh.

### (2) Aura-attachment substrate — Phase 4 only

Phases 1+2+2.5+3 SHIPPED. The Phase 1 flash widening
(``_AURA_SAFE_EXTRA_KEYWORDS = {"flash"}``) landed at round-79; the
**measurement-attribution close** landed at 2026-05-21
(`feature/aura-phase-1-flash-widening`, this lane). The lane verified-
first that the trajectory bridge predicate already admits Frantic
Strength + Shardmage's Rescue cleanly (the round-79 widen has been live
since the Aura-substrate Phase 2 first deliverable; both cards parse
clean: pure ``equipment_bonuses`` + a static ``flash`` keyword
symmetrically dropped on both bridge sides), then closed the two
attribution gaps:

  1. ``scripts/measure_pool_parity.loader_admits`` was missing the
     ``_printing_is_bridge_simple_aura`` disjunct — the trajectory
     predicate has 9 disjuncts but the metric helper only consulted 3
     (basic land + simple creature + warp). Adding the Aura disjunct
     flips a 0-Effect keyword-only Aura into the COVERED tier via the
     ``not eff_classes and admitted`` arm. Cross-cutting result:
     Duskmourn's Domination (Phase 2 S2 admit) ALSO flipped from
     BLOCKED → COVERED in the same fix (it was admitted by the real
     loader since round-79's sibling S2 widen but invisible to the
     metric for the identical reason).
  2. Added ``test_cross_engine_real_eoe_frantic_strength_parity`` +
     ``test_cross_engine_real_eoe_shardmage_s_rescue_parity`` to
     ``tests/test_parity_smoke.py`` mirroring the round-84 Cosmogoyf
     dynamic-PT idiom: round-trip the genuine parsed printing through
     ``_printing_to_card_def``, build matched ``DeckSpec`` states
     carrying the Aura + a vanilla bear filler, inject onto the
     battlefield via ``register_and_inject_object`` (Rust) /
     ``state.new_object`` (Python), attach via ``state.attach``
     (Rust) / ``obj.attached_to = host.id`` (Python), then assert
     ``effective_power(host) / effective_toughness(host)`` byte-equal
     cross-engine AND ``compare_fields == []`` over the WHOLE field
     view at the post-attach state. Both probes drive the genuine
     EoE printings end-to-end.

**+2 PROBED-tier cards (Frantic Strength + Shardmage's Rescue);
+1 COVERED-tier card (Duskmourn's Domination); -3 BLOCKED single-
gated.** Headline before → after: PROBED 236→238, byte-equal 333→336,
single-gated 67→64.

The 7-card subsystem-gated row's residue:

* **Single-gated 1 (post-flash-widening attribution close):** Moonlit
  Meditation. **Moonlit Meditation (633 decks) needs Phase 4 S3
  replacement-effect substrate consumer wire-up — substrate landed at
  `7b2f62f0`; the host-as-token swap mechanism is the remaining
  honest-stop and gates the loader admit widen.**
* **Multi-gated 2 (post-Cracked-Skull-C2 closure, 2026-05-22):**
  Chorale of the Void, Sporogenic Infection — both S4-shape, both
  multi-gated with effect-uncovered + nonsafe-trig + Aura. Phase 3
  S4 substrate (`35ede8b0`) is the Aura half; per-card Effect-
  uncovered closures + nonsafe-trigger event bridges are the
  remaining halves.
* **Cracked Skull (PROBED, 2026-05-22, `feature/cracked-skull-
  cycle-2-p1`):** the first Aura-S4 multi-gate-composition card to
  flip end-to-end. Cycle 2 ships
  `Effect::TargetOpponentLookHandMayDiscard` + cross-engine
  bridge arm + Phase-3 admit-set widen for the two-trigger
  composition (ENTERS look-at-hand-may-discard + DEALT_DAMAGE
  destroy-host). 20 parity tests at
  `tests/test_cracked_skull_parity.py`; full closure entry at
  `docs/SEARCH_LIMITATIONS.md` (Cracked Skull section).

Sibling lane 3 (`feature/aura-phase-4`) shipped Phase 4 first
deliverable at `6fa5ac9c` (Effect::SacrificeSelfUnlessVoidActive
substrate for Chorale of the Void's end-step trigger).
**`feature/aura-phase-4-s3` (2026-05-21) ships Phase 4 S3 substrate
landing for Moonlit Meditation** — Rust CardPrinting
``aura_replaces_first_tokens_with_copies`` field + per-player
``created_tokens_this_turn: [bool; 2]`` state slot + turn-cleanup
reset + bridge round-trip + per-token-arm flag flip. **Loader admit
stays declined** (the host-as-token swap mechanism in the Rust
CreateToken arm is the remaining honest-stop — needs a runtime-
registry append primitive OR a per-object Token-supertype override
mechanism, neither exists today). Live-EoE admit delta: +0 cards /
+0 decks; substrate-only landing in the slice-19 / Chorale-Effect
tradition.

* **Why it's still on the list:** the remaining Phase 4 S3 close-out
  is small (the substrate is now paid — the consumer wire-up is the
  CreateToken-arm Aura-consulting + host-printing-as-token swap +
  loader admit widen, ~6-10h estimate gated on the supertype-override
  mechanism), and 633 decks of single-gate leverage is real if small.
* **Closure scope:** small (Phase 4 S3 consumer wire-up on top of
  the now-paid substrate) + per-card S4 closures for the
  multi-gated residue (medium each, gated by Effect-uncovered lane
  progress).

#### Phase 3.5 closure note — Mechan Shieldmate (round-82, 2026-05-20)

Extended the Phase-3 substrate with:

* A third condition tag `controlled_artifact_entered_this_turn`
  in the dispatch table (Python + Rust + parse whitelist),
  reading a new per-Player counter
  `artifacts_entered_under_your_control_this_turn` bumped in
  `drain_pending_etb_dispatches` (the centralized bf-entry
  chokepoint) and reset at cleanup, mirroring Python.
* A layer-6 `RemoveKeyword` payload arm in
  `_static_ability_to_self_conditional_bonus` and the
  `removed_keywords` 6th tuple slot in `static_self_bonuses` /
  `StaticSelfBonus`. The Rust `battlefield_has_keyword` reader
  applies REMOVE folds AFTER the positive (printed + UEOT +
  aura-grant + anthem-grant + self-grant) hit — additive ADDs
  minus per-source REMOVEs — mirroring Python's post-walk
  `kws.discard(payload.keyword)`. Closes Mechan Shieldmate.

Section (1) remaining honest-stops post-Phase-3.5: Rampaging
Soulrager / Wildfire Wickerfolk — both absent from the current
EoE snapshot; would need an EoE refresh + per-Condition slice
(`UnlockedDoorsAmongRoomsAtLeast` / `DeliriumActive`
respectively).

### (3) Effect-uncovered diffuse tail — per-card lane

Not a substrate; per-card closures land 1 card at a time. The cheap
Tier-A single-card lane is **EXHAUSTED**. Cut Propulsion CLOSED
(round-86, 2026-05-21: `feature/effect-uncovered-slice` — Tier-A
shape with intrinsic flying-doubler, 11,990 17lands decks) +
Shattered Wings CLOSED (round-87, 2026-05-21:
`feature/shattered-wings-slice` — single-gated tier, 3-way OR
target filter Artifact OR Enchantment OR (Creature AND flying),
7,793 17lands decks; the outer `Sequence(destroy, Surveil 1)`
runs through the pre-shipped Effect::Sequence + Effect::Surveil
arms) + **Beyond the Quiet CLOSED** (round-88, 2026-05-21:
`feature/beyond-the-quiet-slice` — Effect-uncovered honest-stop
follow-on from the wrath-admit-widening lane; 4,698 17lands decks;
field-free wrath shape `Effect::ExileAllCreaturesAndSpacecraft`
with the Creature-OR-Spacecraft multi-type predicate; exile
bypasses indestructible, NO DIES dispatch per CR 700.4 — Beyond
the Quiet AUTO-JOINED the spell-card-bridge admit set via Lane
Y's existing Phase-3 `(0, 0)` wrath admit arm, no further
guardrail widening needed). **Ruinous Rampage closed 2026-05-21**
via `feature/modal-chooseone-cycle-4` (ModalChooseOne envelope
Cycle 4 — new `Effect::ExileAllArtifactsWithManaValueAtMost` Rust
variant + per-leaf serializer arms for `DealDamageEachOpponent`
and `ExileAllArtifactsWithManaValueAtMost` + admit-predicate widen
on the `(0, 0)`-shape all-untargeted ModalChooseOne path; 305
17lands decks). **Drill Too Deep + Reroute Systems closed
2026-05-21** via `feature/drill-too-deep-slice` (Lane DD —
ModalChooseOne envelope Cycle 6 — first per-mode-target
ModalChooseOne flip; new
`Effect::PutChargeCountersOnTargetSpacecraftOrPlanetYouControl
{ amount: i32 }` Rust variant + matching bridge arm +
trajectory `_leaf` table entry + the per-mode target binding
detector table extension in `cast_spell_noncreature` for the
Spacecraft-or-Planet filter; mode 1 = `DestroyTargetArtifact` was
already shipped on the Rust side, the slice added its missing
`_leaf` serializer entry; Untimely Malfunction co-admitted via the
new `destroy_target_artifact` leaf) AND
`feature/reroute-systems-slice-v2` (Lane EE — flipped Reroute
Systems (6,132 17lands decks) COVERED → PROBED via the per-mode
detector table widen + new
`Effect::DealDamageToTargetTappedCreature` + extended
`Effect::PumpTargetArtifactOrCreatureUntilEOT` with
`granted_keywords`). Combined admit-set growth **17 → 20**.
Then `feature/non-state-rs-tractable` (Effect-uncovered single-gated
tier — Exorcise close; Duskmourn {1}{W} Sorcery; Lane GG / state.rs
conflict-avoidance lane) flipped Exorcise COVERED → PROBED via the
new `Effect::ExorciseEffect` Rust unit variant + bridge arm +
reverse-serializer leaf + the file-local `read_doubling_power`
effective-power reader in the 3-way OR target filter (Artifact OR
Enchantment OR Creature-with-effective-power-4+ — structural sibling
of Shattered Wings's destroy-half filter, with two swaps: exile-not-
destroy per CR 702.12 and power-4-not-flying with the effective-
power read folding `obj.counters` + UEOT bonuses byte-equal with
Python's `effective_power`). Admit set **20 → 21**.
Remaining **16 single-gated** Effect-uncovered members:
Archenemy's Charm, Close Encounter, Coordinated Clobbering,
Darkness, Devastating Onslaught, Mutinous Massacre, Pull Through
the Weft, Scour for Scrap, Scout for Survivors, Singularity
Rupture, Systems Override, Terminal Velocity, Vengeful Possession,
Warping Wail, Zero Point Ballad (15; the prior list's Emergency
Eject + Invasive Maneuvers entries were stale — both shipped
upstream of this dated paragraph).

* **Per-card cost:** 1-3 cycles each — **caveat (2026-05-21)**:
  the next-wrath-substrate triad walk
  (`feature/next-wrath-substrate`) characterized Mutinous Massacre,
  Zero Point Ballad, and Singularity Rupture as **NOT** per-card
  Tier-A slices. Each requires NEW Rust subsystem work, not a
  field-free unit Effect variant in the Beyond-the-Quiet shape:
    * **Mutinous Massacre** — control-change-UEOT subsystem
      (per-object controller override + cleanup-step drain +
      every `o.controller_id` read site routed through an
      effective-controller helper; would also close Systems
      Override). Multi-cycle (3-5).
    * **Zero Point Ballad** — X-value tracking subsystem
      (`StackItem.x_value` + `current_resolve_x_value` bracket
      around resolve) + admit-predicate widen for X-cost spells
      + resolve-time `ChooseFromOwnedDestroyedThisResolve`
      decision push. Multi-cycle (3-4); shares C1 with
      Devastating Onslaught.
    * **Singularity Rupture** — `(0, N)` admit-arm widening for
      "any number of target X" + cross-engine deterministic
      subset-pick policy + `MillHalfTargetPlayersEach` variant.
      Multi-cycle (2-3).

  Full characterization at
  `docs/SEARCH_LIMITATIONS.md` "Next-wrath-substrate triad
  characterization — all 3 candidates require new subsystem work,
  not a single per-card slice (HONEST-STOP, 2026-05-21)".

  **Further caveat (2026-05-21,
  `feature/next-tractable-substrate`)**: a verify-first walk of
  six remaining single-gated candidates — Devastating Onslaught,
  Close Encounter, Scout for Survivors, Systems Override,
  Archenemy's Charm, Scour for Scrap — found every one of the six
  requires multi-cycle subsystem work, not a per-card slice:
    * **Devastating Onslaught** shares Cycle 1 with Zero Point
      Ballad's X-value subsystem, plus needs token-copy substrate
      + delayed-trigger end-step sac. Multi-cycle (3-4).
    * **Close Encounter** needs a brand-new
      `resolving_additional_cost_choice_id` runtime state channel
      + cast-pipeline choice plumbing. Multi-cycle (2-3); 16,098
      decks (highest count among the six).
    * **Scout for Survivors** — **PROBED 2026-05-21**
      (`feature/multi-target-admit-phase4`). Closed via Phase 4
      multi-target admit-gate widen + new Rust
      `Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters`
      variant + resolve-time self-scan (graveyard Vec order,
      per-card MV + running sum-MV budget, +1/+1 counter rider).
      The hexad-lane's "shared substrate unlocks Pull Through the
      Weft + Archenemy's Charm mode 2" claim DID NOT pan out under
      verify-first: Pull Through the Weft needs its own
      `ReturnUpToNNonlandPermanentCardsFromGraveyardToHand` Rust
      variant + parser surgery for its dropped lands clause (~3
      cycles), and Archenemy's Charm needs full 3-mode envelope +
      Mode-3 keyword-grant composition (~3-4 cycles). Scout was
      the only clean substrate-only flip; the other two remain
      DEFERRED with their own per-card slices required. Per
      `docs/SEARCH_LIMITATIONS.md` "Multi-target Phase 4
      resolve-time self-scan" — the resolve-time self-scan is a
      documented deliberate divergence vs CR 601.2c cast-time
      binding (cross-engine harness's `_PickFirstLegalTargetAgent`
      makes the divergence unobservable; real-game multi-target
      pick freedom would require a `StackItem.target_obj_ids:
      Vec<u32>` follow-up).
    * **Systems Override** shares the control-change-UEOT
      subsystem with Mutinous Massacre (C1-C3 of the triad lane).
      Multi-cycle (3-5); +1 cycle for Spacecraft-conditional
      charge-counter rider.
    * **Archenemy's Charm** is a 3-mode ModalChooseOne; mode 2
      rides the multi-target return-to-hand substrate
      (shared with Scout for Survivors), mode 3 is bespoke
      composition. Multi-cycle (3-4).
    * **Scour for Scrap** needs Cycle 1 `chosen_modes:
      Option<u8>` → `Option<Vec<u8>>` widening + two new per-mode
      Effect variants. Multi-cycle (2-3).

  Full characterization at
  `docs/SEARCH_LIMITATIONS.md` "Next-tractable-substrate hexad
  characterization — all 6 candidates require multi-cycle
  subsystem work, not a single per-card slice (HONEST-STOP,
  2026-05-21)". Cross-candidate amortization summary in that
  entry's tail: cheapest-amortized lane is the multi-target
  admit-gate Phase 4 widen (3 candidates), followed by the
  control-change-UEOT lane (2 candidates).

  **Control-change-UEOT Cycle 1 landed (2026-05-21,
  `feature/control-change-ueot-cycle-1`).** Substrate-only:
  `controller_override: Option<u8>` on `GameObject` +
  `effective_controller_id` helper + cleanup-step drain +
  `grant_control_change_ueot(target_id, new_pid)` PyO3 method +
  per-object snapshot key. No live mint site yet — Cycle 2
  reroutes the raw `.controller` read sites; Cycle 3 wires the
  Python ↔ Rust bridge round-trip; Cycle 4 adds the Mutinous
  Massacre / Systems Override Effect variants that mint the
  override. Substrate landing is **back-compat byte-equivalent**
  on production self-play / cross-engine trajectories — every
  override defaults to `None`, the new snapshot key is `None`
  for every pre-substrate object, and `controller_overrides_cleared`
  in the cleanup-summary dict is `0` until Cycle 4 mints
  overrides. Full Cycle 1 entry at `docs/SEARCH_LIMITATIONS.md`
  §1 (Mutinous Massacre).

  **Cracked Skull Cycle 1 landed (2026-05-22,
  `feature/cracked-skull-cycle-1-p1`).** Substrate-only for the
  Aura-S4 multi-gate composition lane's ETB-trigger half (Cracked
  Skull's "look at target opponent's hand; you may choose a
  nonland card from it; that player discards that card" oracle
  text, the second of three Aura-S4 multi-gate composition cards
  alongside Chorale of the Void and Sporogenic Infection). Rust-
  side: `PendingDecision::LookAtOpponentHandMayDiscard
  {source_obj_id, controller, target_player, candidates}` variant
  on `enum PendingDecision`; fingerprint-serialization arm;
  dict-snapshot surface; `submit_look_at_opponent_hand_may_discard
  _decision(Option<u32>)` dispatcher (sibling shape to
  `submit_tap_permanent_decision` — `Some(id)` directs the
  opponent to discard that card; `None` declines the may-gate);
  `push_look_at_opponent_hand_may_discard_choice` synthetic-test
  PyO3 helper (zero production callers in C1; C2's Effect arm
  replaces it as the sole live mint); two auto-resolve
  strategies (`"look_at_hand_decline"` aliases the default-decline
  path, `"look_at_hand_discard_first"` mirrors `tap_permanent_first`).
  Substrate landing is **back-compat byte-equivalent** on
  production self-play / cross-engine trajectories — no production
  caller mints the variant today; pre-Cycle-1 pending-decision
  fingerprints / snapshots stay identical. Cycle 2 wires
  `Effect::TargetOpponentLookHandMayDiscard` + Python ↔ Rust
  bridge + admit-gate at the Aura loader + PROBED flip. 18
  substrate tests pinned at `tests/test_cracked_skull_cycle1.py`.
  Full Cycle 1 entry at `docs/SEARCH_LIMITATIONS.md` (Cracked
  Skull section).

  **Cracked Skull Cycle 2 landed (2026-05-22,
  `feature/cracked-skull-cycle-2-p1`).** Closure of the Aura-S4
  multi-gate composition lane's ETB-trigger half + PROBED flip.
  Cycle 2 ships: `Effect::TargetOpponentLookHandMayDiscard` unit
  variant on `rust/src/cards.rs` (parse + to_pydict + resolve
  arm), cross-engine bridge arm
  `target_opponent_look_hand_may_discard` in
  `src/mtg_bot/parity/cross_engine.py::_ta_dict_to_python_effect`,
  Phase-3 safe-trigger admit-set widen
  (`_aura_triggers_are_phase3_safe`) adding shape 4
  (`ENTERS+IsSelf+TargetOpponentLookHandMayDiscard`) + shape 5
  (`DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent` — Cracked
  Skull's second trigger half), and the matching
  `_python_ta_to_safe_dict_aura_phase3` emitter arms. Net result:
  Cracked Skull progresses from BLOCKED-multi-gate →
  **PROBED** end-to-end (the first Aura with a two-trigger
  composition — ETB look-at-hand-may-discard + DEALT_DAMAGE
  host-event destroy — to ride the bench loader end-to-end). 20
  parity tests pinned at
  `tests/test_cracked_skull_parity.py`. `revealed_to`
  honest-stop documented in `docs/SEARCH_LIMITATIONS.md`
  (Cracked Skull section): Rust's `GameObject` doesn't carry a
  per-object `revealed_to` slot today, but the post-resolve
  cross-engine observable on `target_player.hand` /
  `target_player.graveyard` is byte-equal because neither engine
  reads `revealed_to` for any subsequent state mutation.

* **Pool-flip per cycle:** 1 card + its deck count.
* **Cumulative reach:** largest row but incremental cadence.

### (4) Dynamic-P/T formula — **CLOSED** (round-84, 2026-05-20)

Substrate landed `6a8b6402` (round-78); Harmonious Grovestrider
closed end-to-end (now PROBED at
`tests/test_parity_smoke.py::*_harmonious_grovestrider_*`).
**Cosmogoyf closed round-84** (this lane: `feature/dynamic-pt-cosmogoyf`)
via the SAME dispatch table — added the `"cosmogoyf"` arm to
`state_ops_combat::dynamic_base_pt_pub` (P=N, T=N+1 where
N = controller's-exile-zone count; mirrors Python's
`game._dynamic_base_pt` arm verbatim) + widened the bridge admit
predicate (`trajectory._printing_is_bridge_simple_creature`) from a
single-key match to a module-level admit set
`_DYNAMIC_PT_FORMULA_ADMIT_SET = {"harmonious_grovestrider",
"cosmogoyf"}` so future formula slices land here with a one-key
addition. PROBED at
`tests/test_parity_smoke.py::test_cross_engine_real_eoe_cosmogoyf_dynamic_pt_parity`
(drives the GENUINE parsed EoE Cosmogoyf printing — round-46
real-EoE-signal discipline; Cosmogoyf has no ward / triggered /
activated / static abilities, so unlike Grovestrider the live
printing admits directly through the bridge without needing a
synthetic CardDef).

**Honest-stop:** `"emissary_escort"` formula key remains drop-at-
filter on the bridge AND honest-stopped in the Rust dispatch table.
Its shape is different from the two admitted CDA-only formulas —
Escort reads `printing.power` / `printing.toughness` and ADDS X
(greatest mana value among OTHER artifacts the controller controls);
the CDA-only `dynamic_pt_formula` arm this dispatch table covers
today sets the BASE from scratch. Emissary Escort's slice extends
the dispatch table when it lands; not in the current EoE snapshot,
so no immediate leverage.

### (5) warp_cost remaining-3 — Phase 3 — **SHIPPED** (round-83, 2026-05-20)

Phase 1/1.5/2/3 **all SHIPPED**. The 3 Phase-3 cards — **Bygone Colossus,
Red Tiger Mechan, Timeline Culler** — flipped from
BLOCKED-subsystem-gated → PROBED on `feature/warp-cost-phase-3` (round-83,
2026-05-20). Phases 1+1.5+2 had already shipped the substrate end-to-end
(CardDef warp fields + Rust `CardPrinting.warp_cost` +
`Effect::WarpExileSelf` + `cast_creature_warp` life-debit + graveyard-warp
loop + the bridge filter widen at `trajectory.py:5552`); Phase 3 closed
the two measurement-attribution gaps that kept the 3 cards in the BLOCKED
tier:

1. `scripts/measure_pool_parity.py:loader_admits` did not consult
   `_printing_is_bridge_pure_warp_creature` — widened to include it,
   matching the production-deck loader.
2. The full warp lifecycle probes used `_byte_equal` suffixes invisible
   to the `_smoke_probed_cards` slug-keyed scanner — added 3 thin
   `_parity`-named delegators in `tests/test_parity_smoke.py` so each
   card earns PROBED credit via the canonical lifecycle byte-equality
   evidence.

Net: PROBED tier 232 → 235; warp_cost row 10/3/15,436 → 7/0/0
(multi-gated tail unchanged — Starfield Vocalist + Memorial Team Leader +
Exalted Sunborn + Loading Zone gated also on static_abilities; Codecracker
Hound + Anticausal Vestige + Possibility Technician also on
effect_uncovered+nonsafe_trigger). Bridging warp alone never flipped
those 7 — they need their **other** gate to land first.

**This was lever #2 in the §4 closure sequence; CLOSED.**

## 4. Recommended closure sequence (top 5, post-refresh)

Sorted by **decks(1g) ÷ cycles-remaining**, post-session-substrate:

| # | Lever | Decks(1g) | Cycles | Decks/cycle | Notes |
|---|---|---:|---:|---:|---|
| 1 | **Static / layered abilities — Mechan Shieldmate (Phase 3.5)** | (single card in the 8) | 1 | — | **CLOSED round-82 (2026-05-20)** — `feature/mechan-shieldmate-closure` |
| ~~2~~ | ~~**warp_cost Phase 3**~~ | ~~15,436~~ | — | — | **CLOSED round-83 (2026-05-20)** — `feature/warp-cost-phase-3`; +3 PROBED |
| ~~2~~ | ~~**Dynamic-P/T formula — Cosmogoyf**~~ | ~~small~~ | ~~1-2~~ | — | **CLOSED round-84 (2026-05-20)** — `feature/dynamic-pt-cosmogoyf`; +1 PROBED; one-arm extension to the round-78 dispatch table |
| 3 | **Aura Phase 4 S3 — Moonlit Meditation** | 633 | 2-3 (token-replacement substrate) | ~210 | Substrate LANDED via `feature/aura-phase-4-s3` (`4dae568e`); consumer wire-up gated on the supertype-override mechanism (in flight on `feature/aura-supertype-override`) |
| 4 | **Effect-uncovered per-card drain — next 3-5 cheap single-gated** | varies (~5-15k cumulative) | 3-5 | ~3,000 | Cheaper Tier-A lane exhausted; remainder is Tier-C |

**Sequencing rationale:**

* The original #1 (Static abilities) is **now small-cost-per-card** —
  the substrate cost was paid in the session. Mechan Shieldmate
  closed the last EoE single-gated card on the layer table (round-82).
* **warp_cost** moved up to #2 because its substrate was similarly paid
  for and the deck count was the largest among 1-2-cycle slices;
  closed round-83.
* **Dynamic-P/T formula** (Cosmogoyf) was the cheapest 1-cycle slice
  after Static + warp_cost both closed — the round-78 dispatch table
  was already paid for and Cosmogoyf added a single match arm +
  bridge admit-set widen. Closed round-84.
* **Aura Phase 4 dropped to #4** because attach substrate is paid but
  S3 token-replacement is its own small substrate.
* **Effect-uncovered lane** continues in parallel; pace is bounded by
  per-card Rust scope, not substrate availability.

## 5. Substrates explicitly OFF the top-5

* **Cost-reduction substrate (5 cards, 53,648 single-gate decks).**
  Highest single-gate deck count of any named row, but
  `docs/COST_REDUCTION_SUBSTRATE_PLAN.md` documents the bridge-correct
  realizable yield is **0 in any near-term clean slice** — admitting
  any of the 5 ships a guaranteed inter-engine divergence. Large
  multi-round Rust substrate. Keep deferred.
* **Non-basic land substrate remainder (13 cards, post round-85).**
  Behind sibling substrates (conditional-tapped — Duskmourn painlands;
  Station / activated — Secluded Starforge; clone — Echoing Deeps;
  Exalted-static — Cathedral of War). Not a clean slice. The 5
  Planet lands closed in round-85 (`feature/non-basic-land-slice`,
  `_printing_is_bridge_simple_tapped_mana_land`); the 5 shocklands
  closed in round-60. The remaining 13 are 8,893 cumulative decks
  (Secluded Starforge 7,570 + Cathedral of War 1,205 + Echoing Deeps
  118 + 10 zero-deck Duskmourn painlands).
* **Station / Spacecraft (8 cards, 5,938 decks).** Documented
  §2-decline; large substrate.
* **Carrier activated abilities (30 cards, 0 single-gated).** All
  multi-gated — closing this gate alone flips zero cards.
* **Non-safe triggered ability (43 cards, 0 single-gated).** Per-event-
  shape frontier; co-blocker row, not a lever.
* **Rooms / doors, Planeswalker loyalty, Kicker, BECOMES_TAPPED.** All
  have 0 single-gated members; closing the gate alone flips zero cards.
* **Static rule-text — Deafening Silence (1 card, 12 decks).** **2026-
  05-21 unclassified-residual hand re-walk** characterised this as a
  contained-medium named substrate, NOT a generic unknown: the
  printing carries
  ``restricts_noncreature_spells_to_one_per_turn=True`` (a recognised
  parser bool slot at ``src/mtg_bot/cards/base.py:843``), Python fully
  models the mechanic (per-player ``noncreature_spells_cast_this_turn``
  counter incremented in ``actions.py:1782``, reset at cleanup in
  ``turn.py:1106``, post-filter in ``legal.py:880-893``), but Rust
  carries NO mirror (the comment at ``rust/src/state.rs:972`` is
  explicit), the bridge has no round-trip for the printing bool, and
  no loader admit disjunct admits the printing shape. Closure scope is
  contained-medium: one new Rust ``CardPrinting`` bool slot + one new
  ``Player`` ``noncreature_spells_cast_this_turn: i32`` counter +
  per-cast increment in Rust's noncreature-cast primitives +
  cleanup-step reset + cast-time gate in Rust's legal-decisions
  emitter + bidirectional bridge round-trip of the bool + one new
  ``_printing_is_bridge_simple_cast_restriction_enchantment`` admit
  disjunct + one real-EoE byte-equal probe. Deferred at 12 decks of
  leverage — substantially below the §4 sequencing thresholds. The
  hand re-walk renames the gate ``unclassified → static_rule_text``;
  see `docs/SEARCH_LIMITATIONS.md` "Deafening Silence — Rust does not
  mirror the noncreature-cast restriction" for the divergence pin.
* **Unclassified — Leyline of Mutation (1 card, 0 decks).** **2026-
  05-21 unclassified-residual hand re-walk** confirmed the genuine
  residual: the printing has TWO mechanics oracle-side ("If this card
  is in your opening hand, you may begin the game with it on the
  battlefield." + "You may pay ``{W}{U}{B}{R}{G}`` rather than pay the
  mana cost for spells you cast.") and the parser surfaces NEITHER —
  the printing has no leyline / opening-hand-into-battlefield field,
  no alternative-cost field, no triggered/static/activated/spell
  Effects, no covered Rust counterpart, no bridge carrier. Closure
  requires TWO new substrates (sprawling each): the opening-hand
  leyline mechanic AND the alternative-cost system. Leverage is
  0 decks (the card has 0 17lands appearances). Honest-stopped
  here — the named substrates would each take multi-round campaigns
  with zero in-pool leverage payoff today. See
  `docs/SEARCH_LIMITATIONS.md` "Leyline of Mutation — leyline + alt-
  cost mechanics unparseable" for the divergence pin.

## 6. New sub-gates revealed this session

The substrate closures revealed (or made tractable) a handful of
sub-gates that the original `ce564786` inventory didn't surface:

* **Static abilities Phase 3.5 — keyword-removal layer-6 fold +
  `ControlledArtifactEnteredThisTurn` Condition.** Surfaced by Mechan
  Shieldmate honest-stop in the Phase 3 merge (`ddd93684`); lane 1 in
  flight.
* **Aura Phase 4 S3 — token-replacement-effect substrate.** Surfaced
  by Moonlit Meditation as the only un-shipped Aura sub-shape.
* **Aura S4 multi-gate composition** — Chorale of the Void / Cracked
  Skull / Sporogenic Infection each need their Aura half (shipped
  `35ede8b0`) + Effect-uncovered closure + nonsafe-trigger event bridge.
  These are now exactly that cleanly-decomposed.
* **warp_cost Phase 3 — CLOSED round-83 (2026-05-20).** The 3 Phase-3
  carriers (Bygone Colossus, Red Tiger Mechan, Timeline Culler) flipped
  to PROBED on `feature/warp-cost-phase-3`. The Phase 1/1.5/2 substrate
  was the real lift; Phase 3 turned out to be two measurement-
  attribution fixes (widen `loader_admits` to include
  `_printing_is_bridge_pure_warp_creature` + add 3 `_parity`-named probe
  aliases that delegate to the existing canonical lifecycle probes), not
  the "small grammar extension" the original Phase-3 scoping had
  anticipated. Honest finding: the substrate had already shipped end-
  to-end; only the metric-side wiring was missing.

None of these are wholly new subsystems; they are sub-shapes of
existing substrates that the session's substrate-builds clarified.

## 7. Cross-references

* **`docs/SUBSYSTEM_ROADMAP.md`** — authoritative living roadmap +
  full cycle-by-cycle history. Every claim here traces back to a row
  in the roadmap.
* **`docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`** — 4-phase Aura plan;
  Phases 1+2+2.5+3 LANDED, Phase 4 first-deliverable (`6fa5ac9c`)
  + Phase 4 S3 substrate (`4dae568e`) LANDED; Phase 4 S3 consumer
  wire-up (Moonlit Meditation loader admit widen) gated on
  `feature/aura-supertype-override` in flight.
* **`docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md`** — historical
  itemization of the now-empty genuinely_missing + blocked_non_effect
  tiers. Read together with this doc to understand how the residue
  was retired.
* **`docs/POOL_PARITY.md`** — headline tier table; this doc partitions
  its `subsystem-gated` row.
* **`docs/residue_worklist.json`** — per-card residue source of truth
  (21 closed entries, 0 genuinely_missing, 0 blocked_non_effect).
* **`docs/COST_REDUCTION_SUBSTRATE_PLAN.md`,
  `docs/WARP_COST_SUBSTRATE_PLAN.md`,
  `docs/NONBASIC_LAND_SUBSTRATE_PLAN.md`** — the §2 verify-first
  scoping audits referenced in §3-5.

## 8. Caveats

* **Effect+bridge ≠ full-game-trajectory parity.** Same headline caveat
  as `docs/POOL_PARITY.md`. The session's broader-archetype-sweep
  v13 resweep (`3e37882f`, 300/300 PASS, 0 divergences) is the
  strongest signal yet that the substrate work translates to real-game
  byte-equality on the loader-admitted subset, but it does not extend
  to the gated remainder (the loader-admit predicate is the gate).
* **Scope estimates are calibrated from cycles 1-15+ + this session's
  ~22 merges.** Re-estimate at each closure.
* **Deck counts** are 17lands EoE archetype-build appearances. The
  single-gate `decks(1g)` column is the honest leverage signal.
* **The scout is the source of truth.** Re-run
  `PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe
  scripts/scout_r58_subsystem_attribution.py` to refresh; if this doc
  and the scout disagree, the scout wins, refresh this doc.
