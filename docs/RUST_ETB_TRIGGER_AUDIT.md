# Rust ETB-trigger emission audit

Audit of how ETB-triggered abilities fire in the **Python** engine
today and what infrastructure (if any) the **Rust** engine has to
emit them. Predicts the shape of the next cross-engine divergence
(forecast: **CE19**, the trigger-emitter / event-dispatch path) and
lays out the substrate-prep + per-Effect work needed to wire ETB
triggers in Rust.

This doc is **doc-only** — no code changes here. It is a pre-stage
for the planned ETB-triggers probe + closure slices.

Companion docs:
- [`docs/ORDER_SENSITIVE_SURFACES.md`](ORDER_SENSITIVE_SURFACES.md)
  §2-3, §"CE8 (predicted)" (renumbered CE19 post-CE18). Predicts
  this exact surface as the next CE candidate.
- [`docs/RUST_BF_ITERATION_AUDIT.md`](RUST_BF_ITERATION_AUDIT.md)
  §"CE8 — when Rust grows a `dispatch_event` / trigger emitter"
  (same prediction, with file:line maps for the inline-resolution
  status quo).
- [`docs/RUST_EFFECT_RNG_AUDIT.md`](RUST_EFFECT_RNG_AUDIT.md) —
  audit pattern this doc mirrors (Section breakdown +
  per-site checklist + closure-slice scope estimate).

---

## Section 1: Summary

- **Python ETB-triggered card count (EoE pool):** **145** of 475
  printings carry at least one `TriggeredAbility(event=Event.ENTERS,
  ...)` (predicate-style). Total `Event.ENTERS` trigger entries:
  **152** (a handful of cards carry more than one). Legacy
  string-key `trigger="etb"` count: 0 (Phase 2 migrated all of them
  to the predicate path; see `actions.py:3415`'s comment).
- **Rust trigger-emission infrastructure today:** **NONE.** Rust has
  no `dispatch_event` analog, no `TriggeredAbility` data shape on
  `CardPrinting`, no `Event` enum, no `push_triggered_ability`, no
  trigger-queue state. The closest existing surface is
  `CardPrinting.etb_effects: Vec<Effect>` (`rust/src/cards.rs:1897`)
  which is fired **inline** at stack-resolution time
  (`state_ops_resolve.rs:5419-5431` and the `_pause` counterpart at
  `state_ops_resolve.rs:5483-5495`) — NOT pushed onto the stack as a
  separate StackItem, NOT predicated on an event, and only fires for
  the **just-cast** source (not for other permanents observing its
  arrival).
- **Top-level verdict: substrate-blocked.** Probing the surface
  today will surface CE19, but closing it is NOT a one-slice fix —
  it needs (a) a `TriggeredAbility` data shape, (b) an `Event` enum
  + predicate substrate, (c) a `dispatch_event` walker, (d) a
  trigger StackItem variant + push helper, and (e) fan-out from
  every ETB-stamping site already inventoried by CE4. Realistic
  scope: 1 substrate slice + N per-Effect slices.

---

## Section 2: Python ETB trigger emission

### 2.1 Data shape

`TriggeredAbility` lives at `src/mtg_bot/effects/base.py:679-758`.
The relevant fields for the ENTERS path:

```python
@dataclass(frozen=True)
class TriggeredAbility:
    trigger: str = ""               # legacy string-key (unused on ENTERS post-Phase-2)
    effect: "Effect | None" = None  # the payload Effect that fires
    requires_void: bool = False     # EoE "Void —" gate
    once_per_turn: bool = False     # ledger gate
    tier_threshold: int = 0         # EoE Spacecraft tier gate
    event: "Event | None" = None    # the predicate-path enum value
    predicate: "Predicate | None" = None  # the subject/state filter
    room_half: "str | None" = None  # DSK Rooms half gate
```

A card's triggered abilities live on `CardPrinting.triggered_abilities`
(tuple of `TriggeredAbility`). The Phase-2 migration (per
`src/mtg_bot/actions.py:3414-3417`) routes every ETB-family
trigger — "this enters", "landfall", "an artifact enters under
your control", "a Kavu enters", "an enchantment enters under your
control", etc. — through `event=Event.ENTERS` + a `predicate` that
filters which firings care about which subject. See
`src/mtg_bot/cards/parser/triggers.py:603-1266` for the mapping
table: oracle-text phrase → `(Event.ENTERS, predicate)`.

### 2.2 The dispatch site

`dispatch_event` is at `src/mtg_bot/actions.py:3283-3399`. The
ETB-specific helper that calls it:

```python
# src/mtg_bot/actions.py:3402-3421
def _trigger_etb_abilities(state: GameState, creature: "GameObject") -> None:
    ...
    dispatch_event(state, Event.ENTERS, label="enters", obj=creature)
```

`_trigger_etb_abilities` is called from exactly two sites in the
engine:

| # | Site | What enters |
|---|---|---|
| P-ETB-1 | `src/mtg_bot/actions.py:1139` (inside `_play_land`) | A land just played from hand |
| P-ETB-2 | `src/mtg_bot/actions.py:3130` (inside `put_into_battlefield`) | Every non-land permanent path (resolve, reanimation, search-and-bf, token creation, ManifestDread, copy effects, exile-recall) |

In addition, `turn.py:619` fires `dispatch_event(state, Event.ENTERS,
label="landfall", obj=new_land)` from one site OUTSIDE the standard
ETB path (a non-standard "land enters via grant" — see `turn.py:579`
docstring). All three paths converge on `dispatch_event`.

### 2.3 The walker

`dispatch_event` at `actions.py:3346-3399`:

```python
permanents = list(state.battlefield)       # snapshot, NOT live iter
if extra_observers:
    ...                                     # append off-bf observers (rare)
for source in permanents:
    for ability_index, ability in enumerate(
        source.printing.triggered_abilities
    ):
        if not ability.matches_event(event, subjects, source, state):
            continue
        ...                                 # combat-damage agg dedup, room_half gate
        firings = (
            etb_trigger_multiplier(state, source.controller_id)
            if event is _Event.ENTERS else 1
        )
        for _ in range(firings):
            push_triggered_ability(state, source=source, ability=ability,
                                   ability_index=ability_index, ...)
```

Key shape properties:

1. **Outer iteration: `list(state.battlefield)`** — Python's centralized
   battlefield (ETB-timestamp order). Snapshotted at dispatch entry so
   triggers that ETB new permanents during dispatch don't observe their
   own event in the same pass (the comment at `actions.py:3343-3345`
   makes this explicit).
2. **Inner iteration: `enumerate(source.printing.triggered_abilities)`**
   — fixed per-printing tuple, in declaration order.
3. **Predicate filter:** each `ability.matches_event` re-evaluates the
   predicate against the (event, subjects, source, state) tuple.
4. **`etb_trigger_multiplier`** lets cards like Panharmonicon-shape
   modifiers fire ETB-triggers N times (gated to `Event.ENTERS`
   only; other events fire once).
5. **No APNAP sort** — triggers go on the stack in source's
   `state.battlefield` index order, regardless of controller. CR
   603.3b technically requires APNAP; Python's "single-controller
   queue makes APNAP moot" simplification is documented in
   `RUST_PARITY_PLAN.md:28-30` and `ORDER_SENSITIVE_SURFACES.md` §3.

### 2.4 Stack push

`push_triggered_ability` is at `src/mtg_bot/actions.py:3172-3280`.
It:

1. Gates on `should_fire_trigger(state, source.id, ability_index,
   ability)` (once-per-turn ledger, tier_threshold, void).
2. Picks targets if `ability.effect.target_count_range()` requires
   them (via `ability.effect.legal_targets(...)` + controller's
   `agent.choose_target` loop, with per-slot `target_slot_kinds`).
3. If `fixed_targets` is provided (e.g. for "the player who was
   hit"), skips target selection.
4. Appends a `StackItem(source=source, controller_id=..., ability=ability,
   targets=targets, ...)` to `state.stack`.
5. Logs the trigger line.
6. Returns True on push, False on fizzle (no legal targets / gated
   off).

`StackItem.ability` (`game.py:StackItem` dataclass) carries the
`TriggeredAbility` reference; resolution at `actions.py:2655`+
walks `stack_item.ability.effect` to actually apply the effect.

### 2.5 Stack-push order under APNAP

Per CR 603.3b: each player orders their own simultaneous triggers,
then they go on the stack in APNAP order (active player's first).
Python's dispatcher does NEITHER — it walks
`list(state.battlefield)` in centralized timestamp order, regardless
of controller. Under CE4 (every bf entry gets a fresh strictly-
monotonic `etb_timestamp`), the active player's permanents are NOT
contiguous in the centralized list whenever both players ETB'd
permanents in interleaved order.

**Cross-reference to BF audit:** `RUST_BF_ITERATION_AUDIT.md` §4 ranks
this as the "CE8 substrate-blocked surface" — Python's emitter order
mirrors `state.battlefield`'s order, which CE4 already pinned to
ETB-timestamp. Rust's eventual emitter must walk
`bf_iter_etb_sorted(state)` (the CE5-introduced helper at
`rust/src/bf_iter.rs:61`) for the iteration to match.

### 2.6 EoE card pool count

Counted via `OracleParserCardSource().get_printings()` (475 EoE
printings + parser-emitted overrides):

- **Printings with at least one `Event.ENTERS` TriggeredAbility:**
  **145**
- **Total ENTERS TriggeredAbility entries:** **152**
- **Legacy string-key `trigger="etb"` printings:** 0 (post-Phase-2)

Spot-check examples (first 10 in iteration order):

```
Tezzeret, Cruel Captain
All-Fates Stalker
Astelli Reclaimer
Auxiliary Boosters
Banishing Light
Dockworker Drone
Exosuit Savior
Haliya, Guided by Light
Hardlight Containment
Honored Knight-Captain
```

So ~30% of EoE printings carry an ETB trigger. This is a **large**
surface — once Rust grows the emitter, ETB triggers will fire on
~roughly 1 in 3 bf entries during a typical game, and every fired
trigger is a potential cross-engine compare point.

---

## Section 3: Rust ETB trigger emission today

### 3.1 What Rust has: `etb_effects` (inline application)

`rust/src/cards.rs:1891-1897`:

```rust
/// "When ~ enters the battlefield" effects. Fire on any
/// permanent landing on the battlefield (creature, artifact,
/// enchantment, planeswalker). Empty Vec = no ETB. First-slice
/// scope: the effects fire IMMEDIATELY on entry, no separate
/// stack item — real MTG triggers them onto the stack and
/// resolves in priority order; that timing nuance is deferred.
pub etb_effects: Vec<Effect>,
```

The application sites are inside the stack-resolver, NOT at the
ETB-stamping sites:

| # | Site | When it fires |
|---|---|---|
| R-ETB-1 | `rust/src/state_ops_resolve.rs:5419-5431` (`resolve_top_of_stack`) | When the resolving stack item's source landed on the battlefield post-resolve (CR 608.2m + the round-9 gate) |
| R-ETB-2 | `rust/src/state_ops_resolve.rs:5483-5495` (`resolve_top_of_stack_pause`) | Identical to R-ETB-1 but without auto-resolving pending decisions |

The fire pattern at R-ETB-1:

```rust
// state_ops_resolve.rs:5419-5431
if post_resolve_zone == Some(ZONE_BATTLEFIELD) {
    let etb_effects: Vec<Effect> =
        match find_source_printing_id(self, top.source_obj_id) {
            Some(printing_id) => match self.registry.lookup(printing_id) {
                Some(printing) => printing.etb_effects.clone(),
                None => Vec::new(),
            },
            None => Vec::new(),
        };
    for effect in &etb_effects {
        apply_effect(self, effect, top.source_obj_id)?;
    }
}
```

This is **not** a triggered ability emitter. It's a per-source
"self-ETB effect" hatch with **four** Python-incompatible properties:

1. **Self-only.** Only fires for the source whose stack item is
   resolving — i.e. the just-cast creature's own ETB effects. **No
   other permanent on the battlefield observes the event**, so
   "when ANOTHER creature enters under your control" / "when an
   artifact enters under your control" / Kavu-shape / Panharmonicon-
   shape triggers DO NOT FIRE in Rust today, full stop.
2. **Inline, not on-stack.** No StackItem push, no response window,
   no priority round. Real MTG: ETB triggers stack on top of the
   permanent's spell resolution and resolve LIFO; effects observing
   them (counterspells targeting triggers — CR 701.5 — are illegal,
   but generic stack manipulation isn't). The inline shape means a
   probe that pushes an instant in response to an ETB trigger has
   nothing to push against.
3. **Cast path only.** The R-ETB-1 / R-ETB-2 gate is "did the source's
   stack item just resolve onto the battlefield". ETB effects do NOT
   fire on the non-cast entry paths: reanimation
   (`move_object_from_graveyard_to_battlefield` at
   `state_ops_zones.rs:319-352`), search-and-bf
   (`SearchLibraryForBasicLandTapped` resolve arm at
   `state_ops_resolve.rs:1437-1540`,
   `ReturnCreatureFromYourGraveyardToBattlefield` at
   `state_ops_resolve.rs:2047-2129`), token creation
   (`Effect::CreateToken` resolve arm), `move_object` to
   `ZONE_BATTLEFIELD` (the generic mover at
   `state_ops_zones.rs:227-247`), or ManifestDread face-down flip
   (when those land).
4. **No event predicate.** `etb_effects: Vec<Effect>` is a flat list
   of effects, not a list of `(predicate, effect)` pairs. There's no
   way to encode "trigger only when ANOTHER creature enters" or
   "trigger only when a LAND enters under your control".

### 3.2 What Rust does NOT have

Grep results across `rust/src/`:

| Concept | Rust today | Closest Python equivalent |
|---|---|---|
| `Event` enum (`Event::ENTERS`, `Event::DIES`, etc.) | **absent** | `src/mtg_bot/effects/predicates.py:Event` |
| `TriggeredAbility` struct | **absent** | `src/mtg_bot/effects/base.py:680` |
| `Predicate` trait/enum + `matches_event` | **absent** | `src/mtg_bot/effects/predicates.py` (`Subject`, `IsSelf`, `IsLand`, `Controlled`, etc.) |
| `CardPrinting.triggered_abilities` field | **absent** | `src/mtg_bot/cards/base.py:CardPrinting.triggered_abilities` (Phase 1.5+ tuple) |
| `dispatch_event` walker | **absent** | `src/mtg_bot/actions.py:3283-3399` |
| `push_triggered_ability` helper | **absent** | `src/mtg_bot/actions.py:3172-3280` |
| `StackItem::TriggeredAbility { ... }` variant | **absent** (only `StackItem { source_obj_id, controller }` at `state.rs:388-391`) | `src/mtg_bot/game.py:StackItem.ability` field |
| `should_fire_trigger` gate (once-per-turn / tier / void) | **absent** | `src/mtg_bot/turn.py:should_fire_trigger` |
| `etb_trigger_multiplier` (Panharmonicon-shape) | **absent** | `src/mtg_bot/turn.py:etb_trigger_multiplier` |
| `state.once_per_turn_fired` ledger | **absent** | `state.once_per_turn_fired: set[tuple[int, int]]` on Python `GameState` |

The Rust smoke-harness Python-side ALSO doesn't pass `etb_effects`
in its registry dicts today — see `cross_engine.py:370-384` (the
`printing_dict` keys are `name`, `card_types`, `mana_cost`,
`effects`, `mana_produced`, `power`, `toughness` only). So the
smoke harness has never exercised even the existing inline
`etb_effects` path; that surface is dead code from the smoke's
point of view.

### 3.3 What it would take to wire ETB triggers in Rust

Mapped onto the Python machinery in §2:

1. **`Event` enum.** Mirror `src/mtg_bot/effects/predicates.py:Event`
   — at minimum `ENTERS`, `DIES`, `ATTACKS`, `BLOCKS`,
   `DEALS_COMBAT_DAMAGE`, `CAST`, `RESOLVES`, `LEAVES`. First slice
   could ship `ENTERS` alone.
2. **`Predicate` enum.** Mirror the Python `Predicate` algebra (`Always`,
   `IsSelf`, `IsLand`, `IsArtifact`, `IsCreature`, `Controlled`,
   `Subject(..., Pred)`, `Pred & Pred`, `~Pred`, `HasSubtype(...)`).
   Closed-enum so the bridge surface validates at parse time.
3. **`TriggeredAbility` struct.** Add `event: Option<Event>`,
   `predicate: Option<Predicate>`, `effect: Effect`, plus the
   `requires_void: bool`, `once_per_turn: bool`,
   `tier_threshold: u32`, `room_half: Option<String>` gates from
   `effects/base.py:707-734`.
4. **`CardPrinting.triggered_abilities: Vec<TriggeredAbility>` field.**
   Add to `rust/src/cards.rs:1855-1898`; parse from the registry dict
   alongside `etb_effects` (which becomes a special case of "an
   `IsSelf` ENTERS trigger that fires the effect").
5. **`StackItem` variant for triggered abilities.** Today
   `StackItem` is `{ source_obj_id: u32, controller: u8 }`
   (`state.rs:388-391`). Add `triggered_ability: Option<...>` or
   move to a kind-enum `StackItem::Spell { ... } | StackItem::Trigger { source, ability_index, targets, ... }`.
6. **`push_triggered_ability` helper.** Mirror Python's
   `actions.py:3172-3280`. Includes target selection (the existing
   `state_ops_decisions.rs` pending-decision substrate can carry
   the picking; auto-resolve gives a deterministic default).
7. **`dispatch_event` walker.** Walk
   `bf_iter_etb_sorted(state)` ×
   `enumerate(source.printing.triggered_abilities)`; per-ability,
   re-check predicate; per-match, call `push_triggered_ability`.
8. **`should_fire_trigger` + `state.once_per_turn_fired` ledger.**
   Add the `(source_id, ability_index) -> bool` set to `GameState`
   plus its cleanup-step reset.
9. **Resolver dispatch for trigger stack items.** When a trigger
   item resolves, apply `stack_item.ability.effect` via the existing
   `apply_effect` path (the Effect closure is already in place — the
   trigger payload is just an Effect).
10. **Fan-out from every ETB-stamping site.** See §4. Every site
    that today calls `reserve_etb_timestamp` (or directly writes
    `obj.etb_timestamp = ...`) is a candidate dispatch_event site.

---

## Section 4: Cross-engine compatibility — every Rust ETB-stamping site is a missed trigger

Cross-reference of every Rust `reserve_etb_timestamp` / `etb_timestamp =`
site from the CE4 closure. For each, the column "Triggers fire?" lists
what fires today vs what SHOULD fire under a Python-faithful emitter.

| # | Site | File:line | Why an ETB happens here | Triggers fire today? | Should fire? |
|---|---|---|---|---|---|
| RS1 | `move_source_post_resolve` permanent path | `rust/src/state_ops_resolve.rs:300-302` | Post-resolve permanent placement after cast (CR 608.2m). Followed by the `etb_effects` inline-apply at `state_ops_resolve.rs:5419-5431`. | **Only the just-cast source's own `etb_effects` fire**, inline. No other permanent observes the event. | A `dispatch_event(ENTERS, obj=src)` walking ALL bf permanents × all their triggered_abilities. |
| RS2 | `SearchLibraryForBasicLandTapped` arm | `rust/src/state_ops_resolve.rs:1477, 1494` | Search-and-bf for a basic land tapped (Cultivate / Rampant Growth shape). | **NO** — `apply_effect` arms never call into `etb_effects` resolution. | dispatch_event ENTERS on the searched-up land. (Many landfall triggers in EoE pool key off this.) |
| RS3 | `ReturnCreatureFromYourGraveyardToBattlefield` arm | `rust/src/state_ops_resolve.rs:2103, 2118` | Reanimation. | **NO** | dispatch_event ENTERS on the reanimated creature. |
| RS4 | `CreateToken` allocator block | `rust/src/state_ops_resolve.rs:1287` (block-stamp for N tokens) | Token creation. | **NO** | dispatch_event ENTERS on each created token. |
| RS5 | Surveil-related ETB bump | `rust/src/state_ops_resolve.rs:3778` | Surveil-shape arm that puts something onto the battlefield. | **NO** | dispatch_event ENTERS on the placed object. |
| RS6 | `ReturnAllCreaturesFromYourGraveyardToBattlefield` block-stamp | `rust/src/state_ops_resolve.rs:4805, 4809` | Mass reanimation block. Reserves a contiguous timestamp range so each returned permanent gets a strictly-monotonic stamp. | **NO** | One dispatch_event ENTERS per returned permanent, in stamp order. **Order matters** — death/ETB chains can ripple. |
| RS7 | Exile-recall block on source LTB | `rust/src/state_ops_resolve.rs:5343, 5347` | `return_paired_exiles_for` (Oblivion-Ring-style auto-recall). Also a contiguous block. | **NO** | One dispatch_event ENTERS per recalled permanent. |
| RS8 | "+1 bump" arm | `rust/src/state_ops_resolve.rs:5567` | An Effect arm that adds a single ETB-stamped object. | **NO** | dispatch_event ENTERS. |
| RS9 | Generic `move_object` to battlefield | `rust/src/state_ops_zones.rs:232-233` | Catch-all zone-mover when target is `ZONE_BATTLEFIELD`. Many Effect arms route through this. | **NO** | dispatch_event ENTERS. |
| RS10 | `move_object_from_graveyard_to_battlefield` | `rust/src/state_ops_zones.rs:336, 354` | Reanimation primitive. | **NO** | dispatch_event ENTERS. |
| RS11 | Land play path | `rust/src/state_ops_cast.rs:205, 210` | `_play_land` equivalent (a counter bump rather than `reserve_etb_timestamp`, but the same write effect). | **NO** | dispatch_event ENTERS (label="enters" on cast-land path, or label="landfall" on grant-land path; both share the predicate-style ENTERS family in Python). |
| RS12 | ManifestDread face-down land | `rust/src/state_ops_decisions.rs:525, 540` | ManifestDread choice resolves: the chosen card flips face-down onto bf. | **NO** | dispatch_event ENTERS on the face-down permanent. |

**Gap summary:** **12 distinct bf-entry sites** in Rust today.
**Exactly one of them** (RS1) wires the partial inline `etb_effects`
hatch — and even that one only fires the source's OWN effects, not
other permanents' triggers observing the event. The other 11 sites
silently drop every trigger they should emit.

### 4.1 What Python does at each equivalent site

For sanity-check, Python's converse is:

- Every Python bf-entry path either calls
  `_trigger_etb_abilities(state, obj)` directly
  (`actions.py:1139`, `actions.py:3130`) or calls
  `put_into_battlefield` which then calls it.
- `dispatch_event(state, Event.ENTERS, ...)` is the single
  centralized walker.
- The 145-printing surface (Section 2.6) means ~30% of Python bf
  entries push at least one stack item.

So today, every Rust bf entry that Python would have emitted
triggers on is a silent drop. Under cross-engine smoke, the
divergence is invisible only because the smoke deckspec
deliberately avoids non-vanilla creatures — the moment one of the
145 ETB-trigger cards lands in a smoke deck, the divergence
surfaces.

---

## Section 5: Predicted CE shape (forecast CE19)

### 5.1 Concrete divergence scenario

Inject a creature with an "ETB: draw a card" trigger into both
engines' decks. Pick a freq>=2 EoE example: **All-Fates Stalker**
(spot-checked above) — actually has "When this creature enters,
each opponent gets <X>" or similar; a cleaner illustrative example
is the synthetic "ETB: draw a card" creature (carries one
`TriggeredAbility(event=Event.ENTERS, predicate=IsSelf(),
effect=DrawCards(1))`).

Cast the creature on turn 2 (drop a land + cast).

### 5.2 Python post-cast state (snapshot AFTER stack drains)

- The creature is on `p0.battlefield` (or wherever it was cast for).
- The ETB trigger pushed a `StackItem` (with the `DrawCards(1)`
  effect) right after the creature's spell resolved.
- The smoke harness's auto-resolve walks the stack: the trigger
  resolves, `p0.hand` size **grows by 1**.
- Post-drain: `state.stack` is empty, `state.battlefield` has the
  new creature, `state.players[0].hand` is 1 card larger.

### 5.3 Rust post-cast state (snapshot AFTER stack drains)

- The creature is on `state.players[0].battlefield` (CE4-correct).
- The `etb_effects: Vec<Effect>` inline-fires the `DrawCards(1)` at
  R-ETB-1 (state_ops_resolve.rs:5419), so for this specific
  IsSelf-only ENTERS pattern Rust actually does fire — IF the smoke
  harness passes `etb_effects: [DrawCards(1)]` in the registry
  dict.

So for the simplest "this creature ETBs: draw a card" probe with
predicate=`IsSelf()`, the divergence is **invisible** if the smoke
harness wires `etb_effects` — Rust's inline application produces
the same final state as Python's on-stack version.

### 5.4 The cleaner CE-surfacing probe

The probe that surfaces the divergence has to involve a trigger
whose source is **not** the just-cast permanent:

**Example A: Panharmonicon-shape "another creature enters"**

A pre-existing artifact on the battlefield with:
`TriggeredAbility(event=Event.ENTERS, predicate=Subject(None,
IsCreature()) & ~IsSelf() & Controlled(YOU), effect=DrawCards(1))`.

Then cast a vanilla creature.

- **Python:** `dispatch_event` walks the bf, finds the artifact's
  matching trigger, pushes a `DrawCards(1)` trigger onto the stack
  for the active player. Stack drains. `hand_p0` grows by 1.
- **Rust:** R-ETB-1 only fires the just-cast creature's OWN
  `etb_effects`. The artifact's "another creature enters" trigger
  has no path to fire. `hand_p0` does NOT grow.
- **Divergence at next snapshot:** `hand_p0` field differs by 1.

**Example B: Landfall**

Pre-existing creature on the bf with:
`TriggeredAbility(event=Event.ENTERS, predicate=Subject(None,
IsLand()) & Controlled(YOU), effect=DealDamageAnyTarget(1))`.

Play a land.

- **Python:** lands fire `_trigger_etb_abilities` from `_play_land`
  (`actions.py:1139`); the dispatcher fires the creature's landfall
  trigger; the trigger resolves and deals 1 to opponent.
- **Rust:** `_play_land`'s Rust counterpart (`state_ops_cast.rs:205`
  area) writes the ETB stamp but never calls a trigger walker.
  Opponent's life total does NOT decrease.
- **Divergence:** `life_p1` field differs.

### 5.5 Divergence axis classification

- **Single-axis** for the simplest Panharmonicon-shape probe (just
  `hand_p0`).
- **Multi-axis** for Landfall + on-stack-response probes:
  `life_p1` + `stack` length (Rust empty, Python had the trigger on
  it pre-resolve) + downstream effects.

Both surface CE19 cleanly. The simplest first probe is the
Panharmonicon-shape `hand_p0` size compare — one axis, one trigger,
deterministic.

---

## Section 6: Closure plan

### 6.1 Slice ordering

The closure is **substrate-prep-needed**, not one-slice. Suggested
sequence:

**Slice 1 — substrate (the big one).** Add the data shapes, dispatch
walker, and stack variant. Touches:

- `rust/src/cards.rs`:
  - Add `pub enum Event { Enters, Dies, Attacks, Blocks,
    DealsCombatDamage, Cast, Resolves, Leaves }` (mirror
    `src/mtg_bot/effects/predicates.py:Event`).
  - Add `pub enum Predicate { Always, IsSelf, IsLand, IsArtifact,
    IsCreature, IsEnchantment, Controlled(ControllerRel),
    Subject(Option<SubjectKey>, Box<Predicate>), And(Box<Predicate>,
    Box<Predicate>), Not(Box<Predicate>), HasSubtype(String), ... }`.
  - Add `pub struct TriggeredAbility { event: Option<Event>,
    predicate: Option<Predicate>, effect: Effect, requires_void:
    bool, once_per_turn: bool, tier_threshold: u32, room_half:
    Option<String> }`.
  - Add `pub triggered_abilities: Vec<TriggeredAbility>` field to
    `CardPrinting`. Parse from registry dict.
- `rust/src/state.rs`:
  - Replace `StackItem { source_obj_id, controller }` with a
    kind-enum: `enum StackItem { Spell { source_obj_id, controller
    }, Trigger { source_obj_id, controller, ability_index, targets,
    combat_damage_amount, ... } }`. **Or** add `kind: StackKind`
    discriminant + optional trigger fields (lower-churn).
  - Add `pub once_per_turn_fired: HashSet<(u32, u32)>` ledger.
- New file `rust/src/state_ops_triggers.rs`:
  - `pub fn dispatch_event(state: &mut GameState, event: Event,
    subject_id: u32, label: &str, ...) -> PyResult<()>`. Mirror
    of `actions.py:3283-3399`. Walks `bf_iter_etb_sorted(state)`.
  - `pub fn push_triggered_ability(state, source_id, ability_index,
    ability, targets, ...) -> PyResult<bool>`. Mirror of
    `actions.py:3172-3280`. Target selection routes through the
    existing `state_ops_decisions.rs` pending-decision substrate.
  - `pub fn should_fire_trigger(state, source_id, ability_index,
    ability) -> bool`. Mirror of `turn.py:should_fire_trigger`
    (once-per-turn ledger + tier_threshold + requires_void).
- `rust/src/state_ops_resolve.rs`:
  - Update `resolve_top_of_stack` and `_pause` to dispatch by
    `StackItem` kind: Spell items go through the existing path;
    Trigger items apply `stack_item.ability.effect` and pop.
- `rust/src/state_ops_cleanup.rs`:
  - Clear `state.once_per_turn_fired` at end-of-turn.

Estimated scope: **300-500 LOC** plus tests. Mostly mechanical mirror
of the Python machinery.

**Slice 2 — fire from every ETB-stamping site.** Replace the
existing inline `etb_effects` apply at R-ETB-1 / R-ETB-2 with a
`dispatch_event(ENTERS, obj=src)` call, and add the same call to
all 11 other sites in §4:

```rust
// after every `obj.etb_timestamp = reserve_etb_timestamp(state)`
// + push-to-bf-zone:
crate::state_ops_triggers::dispatch_event(
    state, Event::Enters, obj.id, "enters"
)?;
```

Each call point is ~3 LOC. Total: ~36 LOC across 12 sites + the 2
existing `etb_effects` sites refactored to use the same path.

**Slice 3+ — per-trigger-Effect coverage.** Each `TriggeredAbility`
predicate shape (IsSelf, Subject(None, IsLand) & Controlled(YOU),
Subject(None, IsArtifact()) & Controlled(YOU) & ~IsSelf(),
HasSubtype("Kavu"), etc.) needs predicate-evaluation code +
test fixtures. The EoE pool has ~10-15 distinct predicate templates
(per `cards/parser/triggers.py:603-1266` table). Each is a
small slice (~30-50 LOC + a smoke probe).

**Slice N — Panharmonicon-shape `etb_trigger_multiplier`** (if
there's a freq>=2 EoE card requiring it). Probably out-of-scope for
the first wave.

### 6.2 Cross-cutting concerns

1. **Order:** `dispatch_event` must walk `bf_iter_etb_sorted(state)`
   (CE5-introduced helper at `rust/src/bf_iter.rs:61`), NOT
   `state.players[0].battlefield` followed by
   `state.players[1].battlefield`. The helper is already documented
   as the "future trigger emitter" consumer at `bf_iter.rs:16`. Get
   this right in slice 1 — fixing it later means re-validating every
   smoke deckspec.
2. **APNAP:** Mirror Python's deliberate simplification — no APNAP
   sort. Pin in `docs/SEARCH_LIMITATIONS.md` (Python already has the
   pin per `RUST_PARITY_PLAN.md:28-30`).
3. **Self-observation snapshot:** Mirror Python's `permanents =
   list(state.battlefield)` snapshot at dispatch entry (so ETBs
   triggered by the dispatch don't observe their own event). Today
   Rust's inline R-ETB-1 application has the same effect because
   it fires per-stack-item, not per-emitter pass; the new walker
   must take a `let permanents: Vec<u32> = bf_iter_etb_sorted(state).iter().map(|o| o.id).collect();` snapshot.
4. **Smoke-harness registry-dict wiring:** `cross_engine.py:370-384`
   needs a new `triggered_abilities` key per printing, serialized
   to match the Rust parser. Otherwise probes can't actually
   exercise the new substrate.
5. **GameState clone semantics:** `once_per_turn_fired: HashSet`
   needs clone-on-write or just `Clone` — cheap either way.

### 6.3 Estimated total slice scope

- **Slice 1 (substrate):** ~300-500 LOC + ~150 LOC tests
- **Slice 2 (fan-out from 12 sites):** ~50 LOC + the existing
  `etb_effects` smoke harness wiring + ~50 LOC tests
- **Slice 3 (first per-Effect coverage — IsSelf-only ENTERS):**
  ~50 LOC + the first cross-engine probe (~50 LOC)
- **Slices 4-N (Panharmonicon-shape, landfall, artifact-enters,
  Kavu-enters, etc.):** ~30-50 LOC each × ~10-15 predicate shapes

**Realistic 1000-game-ready outlook:** 5-8 slices over 2-3 sessions
if no surprises. Most likely surprise: the `Predicate` algebra
needs a richer surface than the closed-enum approach (some EoE
parser predicates compose ad-hoc — `Subject(None, X) & Y &
~IsSelf()` is recursive). If `Predicate` ends up needing
`Box<dyn Trait>` or a more complex AST, slice 1 expands by ~100-200
LOC.

---

## Section 7: Probe recommendation

### 7.1 Concrete first-probe selector

The simplest CE19-surfacing probe is **Panharmonicon-shape**:
inject a pre-existing artifact on p0's battlefield carrying a
`TriggeredAbility(event=Event.ENTERS, predicate=Subject(None,
IsCreature()) & Controlled(p0) & ~IsSelf(),
effect=DrawCards(1))`. Then cast a vanilla creature in p0's hand
during step ~168 (mid-game) of the existing
`multi_blocker_deckspec` walk.

**Probe selector skeleton** (modeled on
`counterspell_selector` / `mass_removal_side_channel` in
`src/mtg_bot/parity/cross_engine.py`):

```python
def panharmonicon_etb_selector(side_channel):
    """Inject a pre-existing 'when another creature enters under your
    control, draw a card' artifact on p0's bf, then a vanilla
    creature spell into p0's hand. Cast it.

    Surfaces CE19: Rust's inline etb_effects only fires the just-cast
    source's own effects; the pre-existing artifact's "another creature
    enters" trigger never fires in Rust today.
    """
    inject_at_step = 168
    artifact_def = CardDef(
        name="Witness Drone",
        card_type="Artifact",
        triggered_abilities=[
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=Subject(None, IsCreature()) & Controlled(YOU) & ~IsSelf(),
                effect=DrawCards(1),
            ),
        ],
        ...
    )
    bait_def = CardDef(name="Bait Bears", card_type="Creature", power=2, toughness=2, ...)
    ...
```

### 7.2 Expected behaviors

- **Pre-injection step 168:** clean walk (the existing
  `multi_blocker_deckspec` is byte-identical).
- **Post-inject + cast:** **Python's `hand_p0` grows by 1, Rust's
  does not.** Single-axis divergence on `hand_p0` field count.
- **Stack mid-resolve compare (between cast and full drain):**
  Python's `stack` has the trigger stacked on top of the creature
  spell (length 2); Rust's `stack` has only the creature spell
  (length 1) — multi-axis if probed mid-drain.

### 7.3 Acceptance pin

Once CE19 surfaces, the probe gets xfail-pinned the same way CE18
was pinned in `tests/test_parity_smoke.py`. The xfail is removed by
the closure slices in §6. **Invariant number:** if numbering
continues from CE18's I45, the CE19-closure invariant lands at I46
(or wherever the next gap-closure invariant runs); see
`PHASE5_STATUS.md` for the current ceiling.

### 7.4 What this probe does NOT surface

Out of scope for first-probe:

- **LtB triggers** — different event (Event.LEAVES) + different
  emitter site (the LTB primitives at
  `state_ops_resolve.rs::move_off_battlefield` etc.). Closes the same
  substrate but needs its own probe.
- **Combat-damage triggers** (CR 702.X triggered by `Event.DEALS_COMBAT_DAMAGE`).
  Touches the granted-damage-trigger family that already has a
  partial Python-side observer aggregator (`_fire_combat_damage_to_player_triggers`,
  `actions.py:3365-3370`).
- **Attack / block triggers** (`Event.ATTACKS`, `Event.BLOCKS`).
  Same substrate, different emitter sites.
- **Cast triggers** (`Event.CAST`). Need a different fan-out site
  (`state_ops_cast.rs` after a successful cast).

All four are slice-N follow-ups once the ENTERS substrate is
proven via the Panharmonicon probe.

---

## Appendix: Quick-reference site map

### Python emission sites

| ID | Site | Role |
|---|---|---|
| P-ETB-1 | `src/mtg_bot/actions.py:1139` (`_play_land`) | Land path |
| P-ETB-2 | `src/mtg_bot/actions.py:3130` (`put_into_battlefield`) | Every non-land permanent path |
| P-ETB-3 | `src/mtg_bot/turn.py:619` (out-of-band landfall) | Grant-land hook |

### Python dispatcher / pusher

- `dispatch_event` — `src/mtg_bot/actions.py:3283-3399`
- `_trigger_etb_abilities` — `src/mtg_bot/actions.py:3402-3421`
- `push_triggered_ability` — `src/mtg_bot/actions.py:3172-3280`

### Python data shapes

- `TriggeredAbility` — `src/mtg_bot/effects/base.py:680-758`
- `Event` enum — `src/mtg_bot/effects/predicates.py:Event`
- Phrase → predicate mapping — `src/mtg_bot/cards/parser/triggers.py:603-1266`

### Rust ETB-stamping sites (12 — see §4)

| ID | File:line | Path |
|---|---|---|
| RS1 | `state_ops_resolve.rs:300-302` | Spell resolve → permanent landing |
| RS2 | `state_ops_resolve.rs:1477, 1494` | SearchLibraryForBasicLandTapped |
| RS3 | `state_ops_resolve.rs:2103, 2118` | ReturnCreatureFromYourGraveyardToBattlefield |
| RS4 | `state_ops_resolve.rs:1287` | CreateToken block allocator |
| RS5 | `state_ops_resolve.rs:3778` | Surveil-shape arm bf-push |
| RS6 | `state_ops_resolve.rs:4805, 4809` | Mass-reanimation block |
| RS7 | `state_ops_resolve.rs:5343, 5347` | Exile recall block |
| RS8 | `state_ops_resolve.rs:5567` | "+1 bump" arm |
| RS9 | `state_ops_zones.rs:232-233` | Generic move_object → bf |
| RS10 | `state_ops_zones.rs:336, 354` | move_object_from_graveyard_to_battlefield |
| RS11 | `state_ops_cast.rs:205, 210` | _play_land equivalent |
| RS12 | `state_ops_decisions.rs:525, 540` | ManifestDread face-down flip |

### Rust inline `etb_effects` application (today's only "trigger" path)

- `resolve_top_of_stack` — `rust/src/state_ops_resolve.rs:5419-5431`
- `resolve_top_of_stack_pause` — `rust/src/state_ops_resolve.rs:5483-5495`

Both fire **only the just-cast source's own** `etb_effects`, inline,
no stack push, no event predicate, no observation by other permanents.
Section 3.1 covers the four Python-incompatible properties.

### Helpers ready to consume

- `bf_iter_etb_sorted(state)` — `rust/src/bf_iter.rs:61` (CE5
  closure introduced this; future-trigger-emitter is one of its
  documented consumers per the module docstring at lines 12-16).

---

## Slice 1 completion notice (landed)

`feature/etb-trigger-slice-1` shipped the substrate listed in §6.1:

- `Event` enum (`Enters` wired; `Dies` / `Cast` / `Attacks` /
  `DealsDamage` declared but parser raises loudly on register so
  callers can't stamp triggers that never fire) — `rust/src/cards.rs`.
- `Predicate` algebra: `Always`, `SourceIsSelf`, `SourceIsCreature`,
  `Not(Box<Predicate>)`, `And(Box<Predicate>, Box<Predicate>)`.
  Minimum to express the audit's §5.4 Example A Panharmonicon-shape
  probe (`SourceIsCreature AND NOT SourceIsSelf`) plus the canonical
  IsSelf-enters shape. Richer predicates (HasSubtype, IsLand, etc.)
  follow in Slice 3+.
- `TriggeredAbility { event, predicate, effect, once_per_turn }`
  struct + `CardPrinting.triggered_abilities: Vec<TriggeredAbility>`
  field (defaults to empty when the registry dict omits it —
  back-compat shape for every pre-Slice-1 caller).
- `StackItemKind` discriminant + `TriggerPayload { effect,
  ability_index }` on `StackItem` (lower-churn variant of the
  audit's enum proposal — pre-Slice-1 readers of
  `item.source_obj_id` / `item.controller` keep working unchanged).
- `dispatch_event(state, Event, subject_id)` walker
  (`rust/src/state_ops_resolve.rs`) — snapshots
  `bf_iter_etb_sorted(state)` at entry, walks every printing's
  triggered_abilities, evaluates predicate, pushes matching firings
  via `push_triggered_ability`. Slice 1 dispatches only `Enters`;
  the parser surface keeps that contract in lockstep.
- `push_triggered_ability` helper (same file).
- Once-per-turn ledger on `GameState.triggered_this_turn:
  HashSet<(u32, u32)>` (cleared at `cleanup_step_for_active_player`).
  Substrate-only in Slice 1 — no Slice 1 test fixture exercises the
  gate (the IsSelf shape fires at most once per permanent's life).
- One wiring point: `resolve_top_of_stack` (and `_pause` variant)
  fires `dispatch_event(Event::Enters, source_id)` AFTER the legacy
  inline `etb_effects` loop and AFTER popping the resolving spell's
  stack item. Pop-then-dispatch matters: the trigger goes on top of
  the now-empty stack and resolves LIFO on the next call to
  `resolve_top_of_stack`.

Slice 1 substrate smoke (`tests_rust/test_triggers.py`, 11 tests,
459 LOC): parser surface (defaults, valid kinds, raises on unwired
events / unknown predicates); end-to-end IsSelf draw on cast-resolve;
end-to-end Panharmonicon-shape "another creature enters" probe;
negative-case regression that "another creature" doesn't fire on
the source's own ETB; cleanup-step ledger clear shape.

**Out of scope for Slice 1 (queued for Slice 2+):**

- Fan-out from the other 11 ETB-stamping sites (§4). Today only the
  cast-and-resolve path (RS1) is wired.
- Bridging `etb_effects` into the predicate path (synthesize
  `TriggeredAbility { event=Enters, predicate=SourceIsSelf,
  effect=... }` per `etb_effects` entry at registry-load time, retire
  the inline hatch).
- Smoke-harness `cross_engine.py:370-384` registry-dict wiring —
  needs a `triggered_abilities` key so the harness can actually
  exercise the new substrate against the Python engine for CE19
  closure.
- Richer Predicate variants (HasSubtype, IsLand, IsArtifact,
  Controlled, Subject) — needed for landfall / artifact-enters /
  Kavu-shape predicates from `cards/parser/triggers.py:603-1266`.
- Target selection on triggers that need it (DealDamage with an
  AnyPlayer / AnyCreature target) — today the trigger resolver
  re-uses `apply_effect`'s deferred targeting (defending player /
  first defender bf object); Slice 4+ adds push-time target picking
  + fizzle check.
- Wider `Event` variant coverage (Dies, Cast, Attacks, DealsDamage).
- Panharmonicon-shape `etb_trigger_multiplier`.
- Combat-damage aggregator dedup.

LOC: ~715 production + ~459 tests.

---

## Slice 2 completion notice (landed)

`feature/etb-trigger-slice-2` shipped piece (1) of the Slice 1 prep
notes — fan out `dispatch_event(Event::Enters, ...)` to the 11 other
ETB-stamping sites in §4 — via a deferred-dispatch queue rather than
inline calls.

**Pattern** (`docs/RUST_PARITY_GAPS.md` I47): every bf-entry site
pushes the entered permanent's id onto
`GameState.pending_etb_dispatches: Vec<u32>` instead of dispatching
inline. The surrounding caller (the resolver in spell-resolve cases,
or the bf-entry primitive itself in standalone-method cases) drains
the queue via `state_ops_resolve::drain_pending_etb_dispatches` AFTER
the resolving slot leaves the stack. The deferred path keeps
`apply_effect` stack-quiet, so the resolver's `pop()` always drains
the resolving slot regardless of how many permanents entered.

**Sites wired**:

- RS2 `state_ops_resolve.rs` (`Effect::SearchLibraryForBasicLandTapped`)
- RS3 `state_ops_resolve.rs` (`Effect::ReturnCreatureFromYourGraveyardToBattlefield`)
- RS4 `state_ops_resolve.rs` (`Effect::CreateToken` block allocator; one push per token)
- RS5 `state_ops_resolve.rs` (`Effect::CreateTokenCopyOfTargetCreatureYouControl`)
- RS6 `state_ops_resolve.rs` (`Effect::ReturnAllCreaturesFromYourGraveyardToBattlefield` mass reanim)
- RS7 `state_ops_resolve.rs::return_paired_exiles_for` (signature flipped from `i32` to `PyResult<i32>`)
- RS8 `state_ops_resolve.rs::register_and_inject_object` (test fixture; bf-targets only)
- RS9 `state_ops_zones.rs::move_object` (signature flipped from `bool` to `PyResult<bool>`; bf-targets only)
- RS10 `state_ops_zones.rs::move_object_from_graveyard_to_battlefield`
- RS11 `state_ops_cast.rs::move_from_hand_to_battlefield` (routes `play_land` + `cast_creature` no-stack primitive)
- RS12 `state_ops_decisions.rs::resolve_manifest_dread_to_battlefield`

Plus the queue substrate itself: `GameState.pending_etb_dispatches`
field + `drain_pending_etb_dispatches` helper in `state_ops_resolve.rs`.

**Regression tests** in `tests_rust/test_triggers.py` (13 new RS-site
tests + 1 once-per-turn end-to-end test, on top of Slice 1's 11):
each RS-site has a self-ETB-draw probe verifying the trigger fires +
the controller draws a card; RS9 has an extra negative-case guard
pinning that non-bf moves stay silent; the once-per-turn ledger test
pins suppress + cleanup-reset.

**Rust gate**: 1902/1902 pass (vs Slice 1's 1885 baseline).

**Out of scope for Slice 2 (queued for Slice 3+):**

- Piece (2) from Slice 1's prep notes — bridge legacy `etb_effects`
  into the predicate path by synthesizing
  `TriggeredAbility { event=Enters, predicate=SourceIsSelf,
  effect=... }` per `etb_effects` entry at registry-load time, then
  retire the inline `etb_effects` hatch in `resolve_top_of_stack` /
  `_pause`. Slice 2 leaves the inline hatch in place — double-fire
  surface is dormant today (no EoE printing carries both shapes).
- Smoke-harness `cross_engine.py` registry-dict wiring for
  `triggered_abilities`.
- Richer Predicate variants (HasSubtype, IsLand, IsArtifact,
  Controlled, Subject).
- Wider Event variants (Dies, Cast, Attacks, DealsDamage).
- Panharmonicon-shape `etb_trigger_multiplier`.

---

## Slice 4 completion notice (landed)

`feature/etb-slice-4` extended the `Predicate` algebra with four new
subject-anchored / controller variants so the Rust parser surface can
express the bulk of EoE's ETB-trigger predicate templates from
`src/mtg_bot/cards/parser/triggers.py:603-1266`:

- `Predicate::SubjectIsLand` — Python `Subject(None, IsLand())`.
  The canonical landfall filter ("whenever a land enters").
- `Predicate::SubjectIsArtifact` — Python `Subject(None,
  IsArtifact())`. Artifact-enters triggers.
- `Predicate::SubjectIsCreature` — Python `Subject(None,
  IsCreature())`. Panharmonicon-faithful "a creature enters" filter
  (paired with `ControlledByYou` for "under your control").
- `Predicate::ControlledByYou` — Python `Controlled(by=YOU,
  subject=None)`. The "under your control" qualifier that pairs
  with each subject filter above.

**Implementation shape**:

- New variants in `rust/src/cards.rs::Predicate`. Each evaluates by
  walking the object-id-to-printing lookup that
  `Predicate::SourceIsCreature` already used; the helpers
  (`source_has_card_type` / `subject_has_card_type` /
  `object_has_card_type` / `find_object_controller`) are factored
  out of the old inline lookup so the new arms read tidily.
- `parse_predicate` arms accept the new dict-kind strings
  (`subject_is_land`, `subject_is_artifact`, `subject_is_creature`,
  `controlled_by_you`). The error message lists all valid kinds
  so a stale caller surfaces it at register time.
- `cross_engine.py::_ta_dict_to_python_predicate` translates each
  new kind to its Python counterpart so a CE19-shape probe builds
  the SAME predicate on BOTH engines from one `CardDef`.

**Tests** (15 new in `tests_rust/test_triggers.py`, 6 new in
`tests/test_cross_engine_etb_triggers.py`):

- Parser surface — each new kind parses without error
  (`@pytest.mark.parametrize` over the four kinds).
- `subject_is_land` fires on a searched-up land arrival
  (RS2 path), doesn't fire on a creature arrival.
- `subject_is_artifact` fires on a reanimated artifact (RS10
  path), doesn't fire on a creature arrival.
- Panharmonicon-faithful `subject_is_creature AND
  controlled_by_you`: fires on p0's own creature, skips opponent's
  creature, skips p0's land (creature half fails).
- Bare `controlled_by_you`: accepts p0's own arrivals (creature
  + observer-self), rejects opponent's arrivals.
- Cross-engine bridge: each new kind round-trips
  `{"kind": "..."}` -> the expected Python `TriggerPredicate`.
- Cross-engine bridge: `and` composition of `subject_is_creature`
  + `controlled_by_you` builds the audit's §5.4 Example A
  Panharmonicon-faithful predicate.

**Rust gate**: 1932/1932 pass (Slice 3 baseline 1902 + 15 new
+ ~15 pre-existing bumps from other landed work = 1932).

**Out of scope for Slice 4 (queued for Slice 5+):**

- `HasSubtype(s)` predicate — needed for Kavu-shape / Sliver-shape
  / Equipment-enters triggers. Blocked on Rust growing a `subtypes`
  field on `CardPrinting`; a separate substrate slice (or pulled
  forward alongside the next Slice 5+ event extension).
- Wider Event variants (`Dies`, `Cast`, `Attacks`, `DealsDamage`).
  Each adds new dispatch sites in `state_ops_resolve.rs` /
  `state_ops_combat.rs` / `state_ops_cast.rs`.
- `Subject(slot, inner)` for events with multiple subject slots
  (Blocks: `"blocker"`/`"attacker"`; DealsCombatDamage:
  `"source"`/`"target"`). Slice 4's defaulted single-subject
  shape is enough for single-subject events but a multi-subject
  predicate ergonomic needs a richer schema.
- Panharmonicon-shape `etb_trigger_multiplier` (the "your ETB
  triggers fire one extra time" modifier).

---

## Slice 5 completion notice (landed)

`feature/etb-slice-5` adds `Event::Dies` plus its dispatch wiring at
the SBA destroy site — the load-bearing "creature dies" event for
combat-shape probes.

**Implementation shape:**

- `Event::Dies` parser arm in `rust/src/cards.rs::parse_triggered_ability`
  (the previously raise-on-register `"dies"` event now succeeds).
- `dispatch_event` accepts `Event::Dies` in addition to
  `Event::Enters`. The DIES branch extends the observer set with the
  dying object as an "extra observer" so `Predicate::SourceIsSelf`
  triggers can match the dying creature against itself
  (mirroring Python's `extra_observers=[obj]` shape at
  `src/mtg_bot/turn.py:281`). The dying object is found by walking
  the four-zone lookup — it's now in the graveyard, not on the
  battlefield.
- New `GameState.pending_dies_dispatches: Vec<u32>` sibling of
  `pending_etb_dispatches` — the SBA destroy loop pushes dying ids
  onto this queue mid-iteration, then drains via
  `drain_pending_dies_dispatches` AFTER the loop completes.
- New wiring point: `state_ops_sba.rs::apply_sbas` pushes each dying
  creature's id onto `pending_dies_dispatches` after the move-to-
  graveyard, then drains at the end of the destroy loop. Cascade
  death triggers (a DIES that pushes a new permanent onto bf, then
  THAT permanent dies on the next pass) compose cleanly with
  `apply_sbas_until_quiescent`.
- `cross_engine.py::_ta_dict_to_python_triggered_ability` accepts
  `event="dies"` and produces Python's `Event.DIES`.

**Tests** (10 new in `tests_rust/test_triggers.py`, 1 new
`test_dies_event_parses_via_cross_engine_bridge` for the bridge):

- Parser surface: `dies` parses; `cast` / `attacks` / `deals_damage`
  still raise.
- SourceIsSelf DIES: a 1/1 with "when this creature dies, draw a
  card" fires its trigger after SBA destroys it from lethal damage.
- Subject-creature DIES observer: a Slimefoot-shape observer fires
  when ANOTHER creature dies.
- Subject-creature + controlled_by_you: fires on your own creature
  death, skips opponent's.
- Multiple simultaneous deaths: two creatures both go lethal in the
  same SBA pass; both fire the observer's trigger.
- Cross-engine bridge: `event: dies` round-trips to `Event.DIES`
  in Python.

**Rust gate**: 1942/1942 pass (Slice 4 baseline 1932 + 10 new).

**Out of scope for Slice 5 (queued for Slice 6+):**

- Other LtB sites that should dispatch `Event::Dies` (or its
  `Event::LeavesBattlefield` sibling): sacrifice cost-pay,
  exile-via-bounce, direct destroy-via-Effect, generic move_object
  to graveyard. These need a finer-grained distinction between
  "dies" (CR 700.4: creature moves bf->graveyard) and "leaves bf"
  (any zone change away from bf). For now the SBA destroy path is
  the load-bearing case for combat-shape probes; broader LtB
  coverage lands in its own slice.
- `Event::Cast` + dispatch site in `state_ops_cast.rs` after a
  successful cast — Slice 6.
- `Event::Attacks` + dispatch site in `state_ops_combat.rs` per
  declared attacker — Slice 7.
- `Event::DealsCombatDamage` — multi-subject event needs richer
  `Predicate::Subject(slot, inner)` schema.

---

## Slice 6 completion notice (landed)

`feature/etb-slice-6` adds `Event::Cast` and its dispatch wiring at
the `cast_creature` site. CAST fires after mana payment commits and
before the move-to-bf step's ENTERS dispatch — matching Python's
cast-then-resolve ordering.

**Implementation shape:**

- `Event::Cast` parser arm in `rust/src/cards.rs::parse_triggered_ability`
  (previously raise-on-register `"cast"` event now succeeds; only
  `attacks` and `deals_damage` still raise).
- `dispatch_event` accepts `Event::Cast` in addition to `Enters` /
  `Dies`. The CAST branch reuses the DIES extra-observer logic: the
  cast-spell is still in the hand zone at dispatch time, so the
  four-zone object lookup finds it. `Predicate::SourceIsSelf` triggers
  ("when you cast this") can match the cast spell against itself.
- New wiring point: `state_ops_cast.rs::cast_creature` dispatches
  `Event::Cast` after `pay_mana_cost` commits and BEFORE
  `move_from_hand_to_battlefield` fires its own ENTERS dispatch.
  Push order: CAST first, ENTERS second. Stack-resolution is LIFO so
  ENTERS resolves first (matches Python's cast-pushes-spell-on-stack
  semantics).
- `cross_engine.py::_ta_dict_to_python_triggered_ability` accepts
  `event="cast"` and translates to Python's `Event.IS_CAST` (note
  Python's enum spelling is `IS_CAST` while Rust uses `Cast`).

**Tests** (8 new in `tests_rust/test_triggers.py`):

- Parser surface: `cast` parses; `attacks` / `deals_damage`
  still raise.
- SourceIsSelf CAST: a free-cast creature with "when you cast
  this, draw a card" fires its trigger on cast.
- Subject-creature CAST observer (Talrand-shape): fires on
  ANY creature cast.
- Subject-creature + controlled_by_you CAST observer: fires on
  p0's own cast.
- Both CAST and ENTERS on the same source: 2 triggers stack;
  push order is CAST first, ENTERS second; LIFO resolves ENTERS
  first.
- Cross-engine bridge: `event: cast` round-trips to Python's
  `Event.IS_CAST`.

**Rust gate**: 1961/1961 pass (Slice 5 baseline 1942 + 8 new + 11
unrelated from concurrent CE20 agent work).

**Out of scope for Slice 6 (queued for Slice 7+):**

- `Event::Attacks` + dispatch site in `state_ops_combat.rs::declare_attackers`
  per declared attacker — Slice 7.
- `Event::DealsCombatDamage` — multi-subject event needs richer
  `Predicate::Subject(slot, inner)` schema for the (source,
  target) subject mapping.
- Cascade-shape CAST triggers that fire on the SOURCE's own cast
  via SourceIsSelf — already supported via the extra-observer
  pattern, but the harness-side cast-without-cast-creature-primitive
  paths (search-loop's stack-push without cast_creature) won't
  dispatch CAST today. Slice N is when we add a real stack-push
  cast primitive.
- Cost-modifier triggers (e.g. "spells you cast cost {1} less" —
  these aren't triggers per se, they're cost-replacements; out of
  the dispatch scope).

---

## Slice 7 completion notice (landed)

`feature/etb-slice-7` adds `Event::Attacks` plus its dispatch wiring
at the `declare_attacker` site — the per-attacker event that fires
after each attacker commits during combat.

**Implementation shape:**

- `Event::Attacks` parser arm in `rust/src/cards.rs::parse_triggered_ability`
  (the previously raise-on-register `"attacks"` event now succeeds;
  only `"deals_damage"` still raises).
- `dispatch_event` accepts `Event::Attacks` in addition to ENTERS /
  DIES / CAST. No extra-observer logic needed — the attacker is on
  the battlefield by construction (eligibility gate rejects
  non-bf objects), so the bf observer set already contains it.
  `Predicate::SourceIsSelf` ("whenever this creature attacks")
  matches naturally via `source_id == subject_id`.
- `declare_attacker` signature change: `bool` -> `PyResult<bool>`.
  The dispatch_event call needs error-propagation; the only caller
  (`state_ops_actions.rs::DeclareAttackers` arm) already used
  `call_method1` with `?` so the Python-facing contract stays
  unchanged. Dispatch happens AFTER `self.attackers.push(obj_id)`,
  so the trigger observes the attacker in its post-declare state.
- `cross_engine.py::_ta_dict_to_python_triggered_ability` accepts
  `event="attacks"` and translates to Python's `Event.ATTACKS`
  (no spelling divergence — Python's enum spells it `ATTACKS` too).

**Push site placement:** Rust's `declare_attacker` is per-attacker
(unlike Python's `declare_attackers` plural which is per-attacker-
list), so the natural dispatch site is at the end of
`declare_attacker` after the push to `self.attackers`. The
`Action::DeclareAttackers` arm in `state_ops_actions.rs` iterates
the id list and calls `declare_attacker` per id, so triggers fire
per attacker in declaration order — mirroring Python's
`_fire_attacks_triggers(state, obj)` per-attacker call at
`src/mtg_bot/combat.py:115`.

**Inline-vs-deferred:** Slice 7 uses INLINE dispatch (not the
deferred-queue pattern Slice 2 used for ENTERS and Slice 5 used for
DIES). Reason: `declare_attacker` is a single-permanent primitive,
not a loop-over-victims iteration. There's no risk of re-entering a
loop mid-iteration via a dispatched trigger; the caller-side
`DeclareAttackers` arm naturally batches per-id calls. The CAST
dispatch in Slice 6 uses the same inline shape for the same reason.

**Tests** (9 new in `tests_rust/test_triggers.py`, plus updates to
2 pre-existing tests that pinned the "attacks not yet wired" arm):

- Parser surface: `attacks` parses; only `deals_damage` still
  raises.
- SourceIsSelf ATTACKS: a 2/2 with "when this attacks, draw a card"
  fires its trigger on `declare_attacker`.
- Subject-creature ATTACKS observer (Hellrider-shape): fires on
  ANY attacker.
- subject_is_creature AND controlled_by_you ATTACKS observer:
  fires on own attacker, skips opponent's attacker.
- Multiple attackers: declaring two creatures in sequence fires
  the observer once per attacker (declaration-order push).
- declare_attacker REJECTS (summoning-sick): ATTACKS dispatch
  doesn't fire on the failure path — eligibility gate bails before
  reaching the dispatch line.
- Cross-engine bridge: `event: attacks` round-trips to
  Python's `Event.ATTACKS`.

**Rust gate**: 1979/1979 pass (Slice 6 baseline 1961 + 9 new + 9
existing/modified). Two pre-existing tests
(`test_triggered_abilities_unwired_event_raises`,
`test_slice5_other_events_still_raise[attacks]`) had their
parametrize / event-string updated to use `deals_damage` as the
new representative unwired event.

**Out of scope for Slice 7 (queued for Slice 8+):**

- `Event::LeavesBattlefield` (LTB) + dispatch at the existing
  `mark_left_battlefield_if_nonland` / `move_object`-off-bf hook
  — Slice 8.
- `Event::DealsCombatDamage` — multi-subject event needs richer
  `Predicate::Subject(slot, inner)` schema for the (source,
  target) subject mapping. Likely a substrate slice on its own
  before any DealsCombatDamage dispatch can land.
- `Event::Blocks` — symmetric of `Event::Attacks` for the
  `declare_blocker` primitive; same single-permanent shape, same
  inline dispatch pattern. Tactically defer until a Blocks
  trigger card surfaces in the EoE pool (low-priority).

---

## Slice 8 completion notice (landed)

`feature/etb-slice-8` adds `Event::LeavesBattlefield` (LTB) plus
its dispatch wiring at the two central bf-exit primitives —
`move_object` off-bf (bounce / exile / hand-back / generic
zone-mover) and `apply_sbas` (combat death + zero-toughness SBA).
LTB fires for ALL bf-exits regardless of card type (lands included,
mirroring Python's `_fire_leaves_battlefield_triggers` semantics).

**Implementation shape:**

- `Event::LeavesBattlefield` parser arm in `rust/src/cards.rs::parse_triggered_ability`
  (the previously-unregistered `"leaves_battlefield"` event now
  succeeds; only `"deals_damage"` still raises).
- `dispatch_event` accepts `Event::LeavesBattlefield`. Re-uses the
  Slice 5/6 extra-observer pattern: the leaving object has been
  moved to its destination zone already (graveyard / hand / exile /
  library — `move_object` is generic), so the four-zone walk +
  exile lookup finds it for SourceIsSelf `Predicate` matching.
  Slice 8 generalizes the lookup from 4 zones (Slice 5/6 pattern)
  to 5 zones (adds exile, since bounce-to-exile is a valid LTB
  destination).
- New `GameState.pending_leaves_dispatches: Vec<u32>` sibling of
  `pending_etb_dispatches` / `pending_dies_dispatches`. Every
  bf-exit primitive pushes the leaving id onto this queue; the
  primitive's caller drains via
  `drain_pending_leaves_dispatches` AFTER the primitive's mutation
  completes.
- Two wiring points wired:
  1. `state_ops_zones.rs::move_object` — when source_was_battlefield,
     push obj.id and drain after the destination push completes
     (and after the ENTERS drain for the bounce-and-replay case).
  2. `state_ops_sba.rs::apply_sbas` — each destroyed creature
     pushes onto BOTH `pending_dies_dispatches` and
     `pending_leaves_dispatches`. Drain order: DIES first, then
     LEAVES. The matching stack-push order pins LIFO resolution to
     LEAVES-first, mirroring Python's `move_off_battlefield`
     (turn.py:280-289 fires DIES then LEAVES_BATTLEFIELD; the
     Python stack is also LIFO so LEAVES resolves first).
- `cross_engine.py::_ta_dict_to_python_triggered_ability` accepts
  `event="leaves_battlefield"` and translates to Python's
  `Event.LEAVES_BATTLEFIELD`.

**DIES + LEAVES double-fire pin:** an SBA-destroyed creature fires
BOTH events. This matches Python's source-order semantics, and the
LIFO stack resolves LEAVES first then DIES — locked in by
`test_dies_and_leaves_both_fire_on_sba_destroy_leaves_resolves_first`.

**Wired bf-exit sites (Slice 8 scope):** only the two central
primitives. The 15+ other `mark_left_battlefield_if_nonland` call
sites in `state_ops_resolve.rs` (Effect-arm-level destroys, exiles,
bounces, sacrifice-as-effect) are NOT yet wired — those fire LTB
inline at the Effect arm without routing through `move_object`.
Slice N+ closes those when an LTB-trigger card pool needs them.

**Tests** (9 new in `tests_rust/test_triggers.py`):

- Parser surface: `leaves_battlefield` parses; only `deals_damage`
  still raises.
- SourceIsSelf LEAVES on bounce: 1/1 with "when this leaves, draw"
  fires when bounced to hand via `move_object`.
- SourceIsSelf LEAVES on SBA destroy: same trigger fires on
  lethal-damage path through apply_sbas.
- subject_is_creature LEAVES observer on bounce: fires when ANY
  creature is bounced.
- subject_is_creature LEAVES observer on SBA destroy: fires on
  combat-death path.
- DIES + LEAVES on same source: both fire; LIFO resolves LEAVES
  first (the canonical "either trigger first" cross-engine pin).
- Non-bf-exit move (hand -> graveyard) does NOT fire LEAVES.
  Tight gate pinned.
- Cross-engine bridge: `event: leaves_battlefield` round-trips to
  Python's `Event.LEAVES_BATTLEFIELD`.

**Rust gate**: 1988/1988 pass (Slice 7 baseline 1979 + 9 new).
One pre-existing test
(`test_triggered_abilities_unwired_event_raises`) had its
regex-match updated from "Slice 7" to "Slice 8".

**Out of scope for Slice 8 (queued for Slice 9+):**

- `Event::DealsCombatDamage` — multi-subject event needs richer
  `Predicate::Subject(slot, inner)` schema for the (source,
  target) subject mapping. Substrate slice required first.
- Fan-out from the 15+ `state_ops_resolve.rs` bf-exit sites that
  call `mark_left_battlefield_if_nonland` directly (Effect-arm
  destroys, exiles, bounces, sacrifice-as-effect). Each is a
  one-line `state.pending_leaves_dispatches.push(obj_id) +
  drain_pending_leaves_dispatches(...)` insertion. Today's two
  wiring points (move_object + apply_sbas) cover the load-bearing
  primitives; the bypass paths fire LTB at their Effect arm
  inline, which works for the EoE pool's current LTB
  trigger set but leaves a known divergence surface for
  future Effect-direct destroys.
- `HasSubtype(s)` predicate — needed for Kavu-shape / Sliver-shape
  / Equipment-enters/leaves triggers. Blocked on Rust growing a
  `subtypes` field on `CardPrinting`; substrate slice.
- `Event::Blocks` — symmetric of `Event::Attacks` for
  `declare_blocker`; defer until a Blocks-trigger card lands in
  the EoE pool.

---

## Slice 9 completion notice (landed)

`feature/etb-slice-9` adds `Event::Blocks` plus its dispatch wiring
at the `declare_blocker` site — the per-blocker event that fires
after each blocker is committed during combat. Symmetric to Slice
7's ATTACKS dispatch.

**Implementation shape:**

- `Event::Blocks` parser arm in `rust/src/cards.rs::parse_triggered_ability`
  (the previously-unregistered `"blocks"` event now succeeds; only
  `"deals_damage"` still raises).
- `dispatch_event` accepts `Event::Blocks` in addition to ENTERS /
  DIES / CAST / ATTACKS / LEAVES_BATTLEFIELD. No extra-observer
  logic needed — the blocker is on the battlefield by construction
  (eligibility gate rejects non-bf objects via
  `read_creature_status`), so the bf observer set already contains
  it. `Predicate::SourceIsSelf` ("whenever this creature blocks")
  matches naturally via `source_id == subject_id`.
- `declare_blocker` signature change: `bool` -> `PyResult<bool>`.
  The dispatch_event call needs error-propagation; the only Rust
  caller (`state_ops_actions.rs::DeclareBlockers` arm) already used
  `call_method1` with `?` so the Python-facing contract stays
  unchanged. Dispatch happens AFTER `self.blockers.push((attacker_id,
  blocker_id))`, so the trigger observes the blocker in its
  post-declare state.
- `cross_engine.py::_ta_dict_to_python_triggered_ability` accepts
  `event="blocks"` and translates to Python's `Event.BLOCKS`
  (no spelling divergence — Python's enum spells it `BLOCKS` too).

**Push site placement:** Rust's `declare_blocker` is per-pair
(blocker_id, attacker_id), so the natural dispatch site is at the
end of `declare_blocker` after the push to `self.blockers`. The
`Action::DeclareBlockers` arm in `state_ops_actions.rs` iterates
the assignment list and calls `declare_blocker` per pair, so
triggers fire per blocker in declaration order — mirroring Python's
per-pair `dispatch_event(state, Event.BLOCKS, blocker=...,
attacker=...)` call at `src/mtg_bot/combat.py:274`.

**Inline-vs-deferred:** Slice 9 uses INLINE dispatch (matches Slice
6 CAST and Slice 7 ATTACKS). Reason: `declare_blocker` is a
single-pair primitive, not a loop-over-victims iteration. There's
no risk of re-entering a loop mid-iteration via a dispatched
trigger; the caller-side `DeclareBlockers` arm naturally batches
per-pair calls.

**Single-subject divergence from Python (documented):** Python's
BLOCKS event carries TWO subjects (blocker + attacker) and exposes
both to predicates via the multi-subject schema (see
`EVENT_SUBJECTS` in `src/mtg_bot/effects/predicates.py:211`).
Rust's `dispatch_event` API only threads the blocker today;
multi-subject predicates against the attacker (e.g. Skystinger's
"whenever ~ blocks a creature with flying" — `IsSelf(subject=
"blocker") & HasKeyword("flying", subject="attacker")`) need the
`Predicate::Subject(slot, inner)` substrate that Slice 12 will
introduce alongside `Event::DealsCombatDamage`. Until then, BLOCKS
triggers whose predicates only key on the blocker (SourceIsSelf,
SubjectIsCreature, ControlledByYou, etc.) work as expected; the
multi-subject pin (Skystinger-shape) is deferred.

**Tests** (9 new in `tests_rust/test_triggers.py`, plus updates to
1 pre-existing test that pinned the Slice 8 unwired-event raise):

- Parser surface: `blocks` parses; only `deals_damage` still
  raises.
- SourceIsSelf BLOCKS: a 2/2 with "when this blocks, draw a card"
  fires its trigger on `declare_blocker`.
- subject_is_creature BLOCKS observer (Hellrider-shape symmetric):
  fires on ANY blocker.
- subject_is_creature AND controlled_by_you BLOCKS observer: fires
  on own blocker, skips opponent's blocker.
- Multiple blockers: declaring two blockers in sequence against the
  same attacker fires the observer once per blocker (declaration-
  order push).
- declare_blocker REJECTS (attacker not in `state.attackers`):
  BLOCKS dispatch doesn't fire on the failure path — the
  eligibility gate bails before reaching the dispatch line.
- Cross-engine bridge: `event: blocks` round-trips to Python's
  `Event.BLOCKS`.

**Rust gate**: 2012/2012 pass (Slice 8 baseline 1988 + 9 new from
this slice + 15 unrelated tests that landed adjacent to Slice 8
from the CE20 Spacecraft tier_threshold merge `093c7e17`).
One pre-existing test
(`test_triggered_abilities_unwired_event_raises`) had its
regex-match updated from "Slice 8" to "Slice 9".

**Out of scope for Slice 9 (queued for Slice 10+):**

- `Event::DealsCombatDamage` — multi-subject event still blocked
  on `Predicate::Subject(slot, inner)` substrate.
- Multi-subject BLOCKS predicates (Skystinger "blocks a creature
  with flying"): same substrate dependency as DealsCombatDamage.
- Fan-out from the 15+ `state_ops_resolve.rs` bf-exit sites that
  call `mark_left_battlefield_if_nonland` directly (Slice 8
  carryover; LTB fan-out task).
- `HasSubtype(s)` predicate — still blocked on `CardPrinting.subtypes`
  field.

---

## Slice 10 completion notice (landed)

`feature/etb-slice-10` closes the LTB fan-out carryover from Slice
8: every `state_ops_resolve.rs` Effect-arm bf-exit site that
bypasses `move_object` (13 sites) and the sacrifice-cost path in
`state_ops_actions.rs` (1 site) now push their leaving id onto
`pending_leaves_dispatches`. The drain runs in `resolve_top_of_stack`
post-pop (mirroring the existing Slice 2 ETB-queue defer-past-pop
pattern), so the LTB trigger lands on the now-empty top of the
stack and resolves LIFO on the next priority round.

**Implementation shape:**

- Per Effect-arm site (13 total in `state_ops_resolve.rs`): the
  existing `let printing_id = obj.printing_id;` capture was
  rewritten to a tuple capture
  `let (printing_id, leaving_obj_id) = (obj.printing_id, obj.id);`
  so the leaving object's id survives the
  `Arc::make_mut(...).push(obj)` move. Immediately after the
  pre-existing `mark_left_battlefield_if_nonland(state, printing_id);`
  line, a single
  `state.pending_leaves_dispatches.push(leaving_obj_id);` is
  appended. **No inline drain at the Effect arm.** Reason: the
  resolving spell is still on the stack during `apply_effect`;
  draining LEAVES inline would push the trigger ABOVE the resolving
  spell and the subsequent `stack.pop()` would pop the wrong slot.
- `resolve_top_of_stack` and `resolve_top_of_stack_pause` each gain
  a `drain_pending_leaves_dispatches(self)?;` call sibling to the
  existing `drain_pending_etb_dispatches(self)?;` — same
  defer-past-pop ordering for both the spell-resolution branch and
  the trigger-resolution branch. The drain happens AFTER `popped
  = self.stack.pop()` so the LTB trigger pushes onto the now-empty
  top.
- The sacrifice-cost path in `state_ops_actions.rs::ActivateAbility`
  arm (the `resolved.sacrifice_self` block at line ~664) **does**
  drain inline. Reason: the cost-pay sequence happens BEFORE the
  activation effect is pushed onto the stack, so an inline drain
  pushes the LTB trigger onto an empty top, then the activation
  effect pushes on top of it, then resolve pops activation-then-LTB
  in correct LIFO order. The Effect-arm constraint doesn't apply
  here — there's no resolving spell on the stack at cost-pay time.

**Wired bf-exit sites (Slice 10 scope):**

The 13 Effect-arm sites in `state_ops_resolve.rs` (line refs after
Slice 10 edit):

1. `Effect::ReturnToHand{AnyCreature}` — bounce (line ~823).
2. `Effect::ExileTarget{AnyCreature}` — exile (line ~997).
3. `Effect::Sacrifice{AnyCreature}` — sacrifice-as-effect (line ~1177).
4. `Effect::DestroyTarget{...}` — Doom Blade-shape (line ~1442).
5. `Effect::DestroyTargetNonlandPermanent` — Drag to the Roots
   (line ~2979).
6. `Effect::DestroyTargetArtifact` — Drill Too Deep (line ~3038).
7. `Effect::DestroyTargetLand` — Strip Mine-shape (line ~3098).
8. `Effect::OptionalSacArtifactForEffect` — inner sac (line ~4306).
9. `Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary`
   (line ~4392).
10. `Effect::DestroyEnchantedPermanent` — destroy via aura host
    (line ~4562).
11. `Effect::ExileTargetCreatureUntilSourceLeaves` — Banishing Light-
    shape (line ~4660).
12. `Effect::ExileOpponentsNonlandPermanentUntilLeaves` (line ~4722).
13. `Effect::EachOpponentSacrificesCreature` — Innocent Blood-shape
    (line ~5521).

Plus the sacrifice-cost path in
`state_ops_actions.rs::ActivateAbility` arm (1 site).

**Known LTB-silent residual (out of Slice 10 scope):**

- `Effect::DestroyAllCreatures` (mass destroy at ~`state_ops_resolve.rs:2895`)
  bypasses `mark_left_battlefield_if_nonland` and instead directly
  sets `state.permanent_left_battlefield_this_turn = true;` after
  its loop. This mass-destroy path doesn't fan LTB per-victim
  today — fixing it would need an O(n) push loop AND careful
  observer iteration since each victim's LTB could in principle
  re-enter the loop. Deferred to a future slice when an LTB-on-
  Wrath card surfaces in the EoE pool.

**Tests** (6 new in `tests_rust/test_triggers.py`):

- `test_ltb_fires_on_return_to_hand_effect_arm` — bounce via
  `Effect::ReturnToHand{AnyCreature}` fires SourceIsSelf LTB.
- `test_ltb_fires_on_destroy_target_creature_effect_arm` — Doom
  Blade-shape `Effect::DestroyTarget{AnyCreature}` fires
  SourceIsSelf LTB on the destroyed creature.
- `test_ltb_observer_fires_on_destroy_target_artifact_effect_arm`
  — `subject_is_artifact` LTB observer fires on artifact destroy.
- `test_ltb_fires_on_exile_target_effect_arm` — bf→exile fires
  SourceIsSelf LTB via the 5-zone extra-observer walk (Slice 8
  pattern).
- `test_ltb_fires_on_destroy_target_land_effect_arm` —
  `subject_is_land` LTB observer fires on land destroy. Pins that
  LTB fires for LANDS too (only the void-flag gates on non-land).
- `test_ltb_fires_on_sacrifice_self_cost_path` — sacrifice-cost
  bf-exit (in `state_ops_actions.rs`) fires SourceIsSelf LTB on
  the sacrificed host.

**DrawCards controller-of-effect divergence note:** the
`Effect::DrawCards` resolution path uses `state.active_player`
rather than the stack item's controller — a pre-existing engine
divergence (see `state_ops_resolve.rs:737`). Slice 10's tests
account for this by asserting library-shrink on the active player
(p0), separate from the controller-of-effect bug. Fix is out of
scope of LTB fan-out.

**Rust gate**: 2018/2018 pass (Slice 9 post-merge baseline 2012 +
6 new from this slice). No regressions across any pre-existing
test — confirms the defer-past-pop ordering protects every prior
Effect arm path.

**Out of scope for Slice 10 (queued for Slice 11+):**

- `Event::DealsCombatDamage` — still blocked on
  `Predicate::Subject(slot, inner)` substrate.
- Multi-subject BLOCKS predicates (Skystinger): same substrate
  dependency.
- ~~`Effect::DestroyAllCreatures` LTB fan-out (the 14th known site;
  needs per-victim push loop).~~ **CLOSED by Slice 11** — see
  completion notice below.
- `HasSubtype(s)` predicate — still blocked on
  `CardPrinting.subtypes` field.
- `Effect::DrawCards` controller-of-effect fix — pre-existing
  divergence, separate workstream.

## Slice 11 completion notice (landed)

Closes the 14th LTB-silent Effect arm: `Effect::DestroyAllCreatures`
(Wrath of God shape) at `state_ops_resolve.rs:2887`.

**The fix.** Per-victim push loop inside the existing destroyable-
collection walk. Two adjacent additions land on each iteration AFTER
the victim is pushed into the controller's graveyard:

```rust
let leaving_printing_id = obj.printing_id;
let leaving_obj_id = obj.id;
Arc::make_mut(&mut state.players[pidx].graveyard).push(obj);
mark_left_battlefield_if_nonland(state, leaving_printing_id);
state.pending_leaves_dispatches.push(leaving_obj_id);
```

Pre-Slice-11 the arm stamped `state.permanent_left_battlefield_this_turn
= true;` directly after the loop AND pushed NOTHING onto the LEAVES
queue. Slice 11 routes the void-flag stamping through the shared
`mark_left_battlefield_if_nonland` helper (same as every other bf-exit
in the engine) AND queues each victim's id for LEAVES dispatch. The
drain runs in `resolve_top_of_stack`'s
`drain_pending_leaves_dispatches` call AFTER the resolving Wrath
spell's stack pop — same defer-past-pop pattern as the 13 Slice 10
sites.

**Deletion**: the standalone `let any_destroyed = !destroyable.is_empty();
if any_destroyed { state.permanent_left_battlefield_this_turn = true; }`
post-loop block is removed; the helper handles the flag stamp per-
victim now.

**Wired sites totals after Slice 11:**

- 14 `Effect`-arm bf-exit sites in `state_ops_resolve.rs` (13 single-
  target Slice 10 + 1 mass-destroy Slice 11).
- 1 sacrifice-cost path in `state_ops_actions.rs::ActivateAbility`
  arm (Slice 10).

**Tests** (5 new in `tests_rust/test_triggers.py`):

- `test_ltb_fires_on_destroy_all_creatures_effect_arm` — single
  victim with SourceIsSelf LTB fires on Wrath; post-pop drain pattern
  pinned.
- `test_ltb_fires_for_each_victim_on_mass_destroy` — TWO victims with
  separate self-LTB triggers both land on the stack post-Wrath (one
  per victim).
- `test_ltb_observer_fires_on_destroy_all_creatures_effect_arm` —
  indestructible observer with `subject_is_creature` LTB fires on
  victim's bf-exit (observer-side dispatch path pin).
- `test_void_flag_still_set_on_destroy_all_creatures` — the helper-
  routed flag stamping preserves the pre-Slice-11 void-flag invariant
  (CE13 / I29).
- `test_no_ltb_fired_when_destroy_all_creatures_destroys_nothing` —
  Wrath on an empty bf is a silent no-op (no LTB triggers, no stack
  growth).

**Rust gate**: 2068/2068 pass (Slice 10 post-merge baseline 2063 + 5
new from this slice).

**Out of scope for Slice 11 (queued for Slice 12+):**

- `Event::DealsCombatDamage` — still blocked on
  `Predicate::Subject(slot, inner)` substrate.
- Multi-subject BLOCKS predicates (Skystinger): same substrate
  dependency.
- ~~`HasSubtype(s)` predicate — still blocked on
  `CardPrinting.subtypes` field.~~ **Substrate landed** by
  `feature/subtypes-substrate`. The `Predicate::HasSubtype(String)`
  consumer lands in a follow-up slice off the substrate — see the
  "Subtypes substrate (landed)" section at the bottom of this doc.
- `Effect::DrawCards` controller-of-effect fix — pre-existing
  divergence, separate workstream.

---

## Subtypes substrate (landed — feature/subtypes-substrate)

`CardPrinting.subtypes: Vec<String>` (Slice 1 of the subtypes lane)
mirrors Python's `CardPrinting.subtypes: tuple[str, ...]` at
`src/mtg_bot/cards/base.py:501`. The substrate-only slice:

- Adds the field to the Rust `CardPrinting` struct with
  back-compat-default behavior — a registry dict missing the
  `subtypes` key parses to empty Vec.
- Adds `Registry::get_printing_subtypes(id)` (mirrors the existing
  `get_printing_card_types` / `get_printing_keywords` getter shape).
- Fills the three direct-construction fixtures in `state.rs`:
  Forest → `["Forest"]` (basic-land subtype, CR 205.3i), Grizzly
  Bears → `["Bear"]` (creature subtype, CR 205.3m), Llanowar
  Elves → `["Elf"]` (the synthetic shim drops the printed "Druid"
  — only "Elf" is consumer-relevant today).
- Passthrough in `cross_engine.py::build_rust_state` so cross-engine
  smoke probes see matching subtype tuples on both engines.

**Unblocks**:
- ~~`Predicate::HasSubtype(String)` — the canonical
  Kavu/Sliver/Equipment-enters predicate.~~ **Landed** by
  `feature/subtypes-has-subtype-predicate` (Slice 2). See
  "Subtypes Slice 2" section below.
- `CE20_FOLLOWUP_PLAYBOOK.md` §5 subtype-filtered sacrifice
  variants (`sacrifice_food`, `sacrifice_a_desert`) — different
  lane, separate consumer.

**Rust gate**: 2112/2112 (substrate baseline 2104 + 8 round-trip
+ default + error-path + casing tests).

---

## Subtypes Slice 2 — Predicate::SubjectHasSubtype (landed — feature/subtypes-has-subtype-predicate)

Subject-anchored HasSubtype predicate built on the Slice 1 substrate.
Mirrors Python's `Subject(None, HasSubtype("Kavu"))` from
`src/mtg_bot/cards/parser/triggers.py:1198`.

**Shape**:
- `Predicate::SubjectHasSubtype(String)` variant added next to
  `SubjectIsLand` / `SubjectIsArtifact` / `SubjectIsCreature`.
- Dispatch reads the subject printing's `subtypes: Vec<String>`
  via a new `subject_has_subtype` helper that mirrors
  `object_has_card_type`'s zone-walk shape. Case-sensitive
  Title Case exact match.
- Parser: `parse_predicate` accepts
  `{"kind": "subject_has_subtype", "subtype": "Kavu"}`. Missing
  `subtype` key raises (loud-fail). Non-string raises.
- `cross_engine.py::_ta_dict_to_python_predicate` translates the
  same dict to Python's
  `Subject(slot=None, inner=HasSubtype(name=...))`.

**Effective-subtypes scope**: reads static `subtypes` directly. The
Rust side has no `effective_subtypes` (CR 613 layer-3/4) yet —
symmetric with how the existing card-type predicates read
`card_types` directly. EoE pool has no Conspiracy-shape
subtype-granting effects today, so the two paths are observably
equivalent.

**Side fix landed in same commit**: Slice 1's
`cross_engine.py::build_rust_state` referenced `card.subtypes`
without `CardDef` having that attribute, breaking 105 parity smoke
tests. This slice adds `CardDef.subtypes: tuple[str, ...] = ()` as
the back-compat default and restores parity smoke to 150/150.

**Rust gate**: 2122/2122 (substrate baseline 2112 + 10 new
HasSubtype tests). **Parity smoke**: 150/150 (restored from the
Slice 1 regression).

---

## ETB Subject substrate — Predicate::Subject(slot, inner) (landed — feature/etb-subject-substrate)

The Slice-12 deferred substrate. Closes the multi-subject predicate
dependency that's been blocking `Event::DealsCombatDamage` and the
multi-subject `Event::Blocks` predicate shape (Skystinger
"whenever ~ blocks a creature with flying").

**Shape**:
- New `SubjectSlot { Source, Target }` enum (start minimal — two
  variants is the minimum to unblock `DealsCombatDamage`'s
  `("source", "target")` slot pair from Python's `EVENT_SUBJECTS`).
  Adding a future slot (e.g. `Attacker` for multi-subject `Blocks`)
  requires one variant + a `parse_subject_slot` arm.
- New `Predicate::Subject(SubjectSlot, Box<Predicate>)` variant —
  pulls the named slot's id out of the dispatch-supplied slot map
  and evaluates `inner` against THAT subject, replacing the
  implicit `subject_id` for the recursive evaluation.
- New `Predicate::matches_with_slots(state, source_id, subject_id,
  subjects_by_slot)` evaluation entry point. Old `matches` is a thin
  wrapper that calls it with an empty slot map — every existing
  single-subject dispatch site continues to work unchanged
  (back-compat).
- Parser: `parse_predicate` accepts
  `{"kind": "subject", "slot": "source"|"target", "inner": {...}}`.
  Unknown slot strings raise loudly (`parse_subject_slot`).
  Missing `inner` raises `KeyError` via `require`. Non-dict `inner`
  raises a typed `ValueError`.
- `cross_engine.py::_ta_dict_to_python_predicate` translates to
  Python's `Subject(slot=<str>, inner=<ObjectPredicate>)` shape at
  `src/mtg_bot/effects/predicates.py:354`. Slot strings pass through
  directly (they match Python's `EVENT_SUBJECTS` keys). Supported
  inner kinds: `subject_is_creature`, `subject_is_land`,
  `subject_is_artifact`, `subject_has_subtype` — each maps to its
  Python `ObjectPredicate` counterpart.

**Single-subject events**: today's dispatch sites pass no slot
bindings (`matches_with_slots` is called via the back-compat
wrapper with `&[]`). A `Predicate::Subject(slot, inner)` on an
ENTERS / DIES / CAST / ATTACKS / BLOCKS / LEAVES trigger evaluates
`false` because the named slot isn't bound — mirrors Python's
`_resolve_subject` "missing slot → predicate is false" short-circuit
at `src/mtg_bot/effects/predicates.py:340`. The 12 substrate tests
pin this back-compat contract.

**Unblocks** (deferred to follow-up consumer slices):
- `Event::DealsCombatDamage` — multi-subject dispatch where
  Source = dealer, Target = recipient. Fan-out from
  `state_ops_combat.rs::apply_damage_pass` per damage event.
- Multi-subject `Event::Blocks` predicates (Skystinger-shape) —
  requires an `Attacker` slot variant + multi-subject BLOCKS
  dispatch in `state_ops_combat.rs::declare_blocker`.
- Susurian Voidborn target-bearing arm (TargetOpponentLosesLife)
  is a SEPARATE issue (agent-driven target selection); not solved
  by this substrate.

**Rust gate**: 2216/2216 (substrate baseline 2204 + 12 new Subject
tests covering parser surface, error paths, no-op-on-single-subject
contract, NOT/AND composition, and inner-variant coverage).
