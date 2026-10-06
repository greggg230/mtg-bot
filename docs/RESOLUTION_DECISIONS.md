# Resolution decisions searched in production (action encoding `canonical_v3`)

Status: design note, branch `feature/s2-resolution-decisions` (off
`s2/eoe-training` 5d3d871a + `feature/s2-boardonly-search`),
2026-10-05. Closes the gap described in `docs/SEARCH_LIMITATIONS.md`
§5 ("Resolution decisions are NOT exposed in production training").
Measurements are filled in at §12 as they land.

## 1. Problem

Every mid-resolution choice -- targets, "up to" choices, modes, X, "may"
yes/no, surveil, sacrifice, discard, tutor / opponent-hand picks,
shock-land life, devour -- is made by `DefaultTacticalAgent` (first
legal / always yes / X = max) for both the heuristic and the NN agents,
because production adapters are built with
`expose_resolution_decisions=False`. `docs/SEARCH_HORIZON.md` measured
2-3 of 17 removal-type stack items per 20 games hitting the agent's own
permanents. The policy head has no slots for `Resolution*Decision`s, so
the network can neither prior nor learn them.

Goal: heuristic PIMC and NN-PIMC search every resolution decision in
production play (self-play / data-gen, the worker production runner,
eval / arena, ladder, capture), and the network gets policy supervision
for them -- without throwing away the trained s2 network.

## 2. The existing `expose_resolution_decisions=True` harness was not production-ready

The adapter-side machinery (`ResolutionChannelAgent`, pause / rollback /
replay, `Resolution*Decision` types) existed and had unit tests, but a
census of 200 random-play EoE games with the flag on (2026-10-05)
found:

| defect | effect | fix (this branch) |
|---|---|---|
| rollback restored the engine state but not the adapter's bookkeeping (`_pass_count`, `_step_entered`, combat-phase lists) | a pause while the last `PassPriority` resolved the stack replayed with `_pass_count` already reset: the replay passed priority instead of resolving, forever. **198 / 200 games** hit the 6000-decision cap | the pre-action snapshot covers engine state AND every adapter field the action can touch |
| only priority actions ran inside the pause-catch harness | a callback fired from a factored attacker / blocker decision (attack triggers that target, `BECOMES_TAPPED` triggers) or from the constructor's first advance raised an uncaught `_ResolutionPause`: **2 / 200** games crashed | every non-resolution `apply` (priority action, attacker / blocker decision) and the constructor's advance run inside the harness |
| `ResolutionChannelAgent` did not delegate `choose_card_from_hand` (base `Agent` raises `NotImplementedError`) and silently used base-class defaults for the other non-exposed callbacks | crash on Terminal-Velocity-style effects; latent drift from the tactical defaults | every non-exposed callback delegates to the fallback explicitly |
| PIMC / ISMCTS determinized the sub-adapter's live state but not its pre-action snapshot | the first resolution answer in a sub-search rolled back to the TRUE (un-determinized) state: hidden-information leak | the snapshot is re-taken from the determinized state; cards the pause revealed to the decider are pinned (§9) |
| `canonical_state_key` / `info_set_key` ignored the replay queue | two different answers to a pause leading to the same follow-up pause hashed equal: NN-eval cache hit with the wrong value | the queue is part of both keys |

## 3. Decision kinds

Exposed to search (each is one `_ResolutionPause` kind):

| kind | callback | options the adapter exposes |
|---|---|---|
| target | `choose_target` | the engine's `(kind, id)` tuples (+ `None` when declining is legal), duplicates removed |
| discard | `choose_discard` | decider's hand cards, one per distinct canonical card key |
| mode | `choose_mode` | `0 .. n_modes-1` |
| modes (NEW) | `choose_modes` ("choose one or both") | every non-empty subset of the modes, printed order (3 options for "one or both") |
| sacrifice | `choose_sacrifice` | the engine's legal object ids |
| x | `choose_x` | `0 .. upper_bound` |
| surveil | `choose_surveil` | mill / keep |
| shock_life | `choose_pay_shock_life` | pay / don't |
| card_from_library | `choose_card_from_library` | one option per distinct canonical card key among the matching library cards, + decline |
| card_from_opponent_hand | `choose_card_from_opponent_hand` | opponent's hand, one per distinct canonical card key |
| devour_lands | `choose_devour_lands` | none / all (existing V1 collapse) |
| yes_no | `choose_yes_no` | yes / no |

Deduplication rule: two hand / library cards with the same canonical
zone key (`printing.name` + the modifier flags `canonical_v2` already
sorts by) are interchangeable, so only the first is exposed. This is the
same equivalence the legal-action hand dedup and `canonical_state`
already rely on. Battlefield objects are NOT deduplicated (attachments /
blocks reference ids).

Left on `DefaultTacticalAgent` (cost payment / SBA / not in the EoE
pool), documented in `SEARCH_LIMITATIONS.md` §5:
`choose_permanent_to_tap` (activation cost, ~0.6 / random game),
`choose_convoke_creatures`, `choose_sac_for_mana`, `choose_mana_color`,
`choose_legend_keep`, `choose_replacement_order`, `choose_card_from_hand`
(no EoE card calls it), mulligans.

Random-play census (400 games, every callback, 2026-10-05): target 5543,
surveil 975, sacrifice 672, discard 406, mode 347, yes_no 264,
card_from_library 262, permanent_to_tap 255, x 148, shock_life 69,
convoke 40, modes 28, legend_keep 14, devour 8, opponent-hand 7,
mana_color 3. Max options seen: target 34, library 11, X upper bound 14.

### Per-seat exposure

Exposure is per seat, not per game: each player's agent slot gets the
`ResolutionChannelAgent` when that seat's decisions are searched, else
the tactical agent. The engine asks the deciding player's own agent, so
a game can mix a searched seat with an auto-targeting seat (needed for
the A/B `v30r+b1` vs `heuristic+autotarget`). A searcher's tree then
models the auto-targeting opponent exactly as it plays.

## 4. Slot layout (`canonical_v3`)

`canonical_v3` = `canonical_v2`'s 517 slots **unchanged** (same indices,
same semantics) + a resolution block appended at 517. A v2 mask is
exactly the v3 mask's prefix at every non-pause state, and every
resolution slot is illegal there; at a pause only resolution slots are
legal. All positions are relative to the **decider** (the player the
engine callback asked = `adapter.active_player()` at the pause).

| family | slots | index | canonicalisation |
|---|---|---|---|
| `RES_TARGET_PLAYER` | 2 | 0 = decider, 1 = opponent | -- |
| `RES_TARGET_BF_OWN` / `_OPP` | 24 + 24 | position among that side's permanents | canonical battlefield order (`canonical_state` shape sort; stable for identical shapes) |
| `RES_TARGET_STACK` | 8 | position from the top of the stack | -- |
| `RES_TARGET_GY_OWN` / `_OPP` | 24 + 24 | position in that player's graveyard | `canonical_v2` zone sort |
| `RES_TARGET_EXILE_OWN` / `_OPP` | 24 + 24 | position in that player's exile | `canonical_v2` zone sort |
| `RES_TARGET_DECLINE` | 1 | target `None` | -- |
| `RES_DISCARD` | 16 | rank among the pause's distinct options | `canonical_v2` key sort |
| `RES_MODE` | 8 | mode index | -- |
| `RES_MODES` | 15 | (bitmask of chosen modes) - 1, modes 0-3 | -- |
| `RES_SACRIFICE` | 24 | decider's permanent position | canonical battlefield order |
| `RES_SACRIFICE_CARD` | 8 | rank among the non-battlefield options | `canonical_v2` key sort |
| `RES_X` | 21 | X = 0 .. 20 | -- |
| `RES_SURVEIL` | 2 | 0 = mill, 1 = keep | -- |
| `RES_SHOCK_LIFE` | 2 | 0 = pay, 1 = don't | -- |
| `RES_YES_NO` | 2 | 0 = yes, 1 = no | -- |
| `RES_CARD_FROM_LIBRARY` | 16 + 1 | rank among the distinct option keys (sorted); last = decline | `canonical_v2` key sort |
| `RES_CARD_FROM_OPP_HAND` | 16 | rank among the pause's distinct options | `canonical_v2` key sort |
| `RES_DEVOUR` | 2 | 0 = none, 1 = all | -- |

288 resolution slots; **vocab 517 -> 805**. (`RES_SACRIFICE_CARD`
exists because the engine reuses `choose_sacrifice` as a generic
"pick an id" hook: Close Encounter designates a warped creature card in
exile, Anticausal Vestige puts a card from hand onto the battlefield.) Caps reuse the v2 constants
where a v2 family exists (hand 16, graveyard / exile / own battlefield
24) and are pinned by `test_vocab_size_constant_is_fixed`.

Why the canonical battlefield order (and not v2's raw battlefield order,
which `ACTIVATE_ABILITY` / `TURN_FACE_UP` keep for compatibility): the
MCTS NN-eval cache keys on `canonical_state_key` (battlefield hashed
canonically) plus a raw-id fingerprint of hand / graveyard / exile only.
With raw battlefield order two transposed states (same permanents, other
entry order) would share a cache entry but map target slots to different
objects. With the canonical order they map them identically.

Target resolution of a `(kind, id)` tuple, mirroring the engine's own
`any_damage_target` handling: `kind == "player"` -> player slot;
`kind == "stack"` -> stack slot (engine stack index); otherwise the id is
looked up on the battlefield, then in graveyards, then in exiles, and
finally (`any_damage_target`-style kinds) as a player id. Anything else
(a hand / library card target) is outside the vocabulary (§5). Two
options of one pause that resolve to the same slot (e.g.
`("creature", 7)` and `("permanent", 7)`) are a codec collision: both
read that slot's prior, `decode` returns the first; collisions are
counted (§5).

### Library picks (the "bigger design call")

Options considered:

1. **Raw library position.** Rejected: library order is hidden and
   randomised, and the cap would be the deck size.
2. **Printing id within the card vocabulary** (one slot per name, the
   embedding table's id space). Each slot query would be tied to a card
   identity -- the only family that could learn "fetch the colour you
   need" from content alone. Rejected for now because it couples the
   ACTION vocabulary to the CARD vocabulary: the slot count would follow
   `n_printings` (320 today, a planned break for the cube), ids above
   the cap fold modulo 319 (two options of one tutor can collide), and
   every card-pool change would need these policy rows remapped by name
   exactly like `printing_embed` (`docs/CARD_VOCAB_FINGERPRINT.md`) or
   they silently scramble.
3. **Rank among the distinct option keys** (chosen). Options are
   deduplicated by canonical key, sorted by it, and the pick is its rank
   (cap 16) plus one decline slot. Vocabulary-independent, tiny, and
   canonical (identical cards are one option -- they are
   interchangeable). Its weakness is shared by every positional family:
   the policy decoder's slot queries have no link to a particular token
   (the tokenizer emits no ordinals), so a rank slot can learn
   "first option" / "decline" biases but not card identity. Search and
   the value head make the call; a pointer-style policy (logit from the
   chosen object's own token) would fix this for every family at once
   and is the right next encoding version, not a one-off here.

## 5. Overflow policy (never a silent drop)

`encode` raises `EncodingOverflowError` for a resolution option past a
cap or outside the vocabulary. `legal_action_mask` skips it (the option
stays a legal search child, with prior 0) and counts it in
`action_encoding.resolution_overflow_counts()` (per kind, process-wide);
the first overflow per kind logs a WARNING. In data-gen the dropped
visits are counted per game (`resolution_overflow_visits`) and written
to the shard. A game whose resolution pauses produce **zero** encodable
resolution policy rows raises (`ResolutionSupervisionError`) -- the
"silent 0" the SEARCH_LIMITATIONS §5 analysis warned about cannot ship.
Codec collisions (§4) are counted the same way
(`resolution_collision_counts()`).

## 6. Action-history tokens

Resolution answers ARE strategically relevant (`action_history` keeps
them) and were already tokenised (`ACTION_KIND_VOCAB` ids 12-22 exist in
every checkpoint; never trained). Data-gen and inference both read
`adapter.recent_actions` through `tokenize_state`, so they agree by
construction. Fixes so distinct answers give distinct tokens (these
tokens never occurred in any training row, so changing them cannot move
any trained behaviour):

* target `("player", p)` (and an `any_damage_target` player id) -> no
  printing lookup (it used to look up OBJECT id `p`), target kind
  PLAYER; stack targets carry the stack item's source printing;
* mode -> `mana_cost_paid` scalar = `mode_index / 10`; the new "modes"
  answer reuses the mode kind id with scalar = `bitmask / 10`;
* shock life -> variant NORMAL (pay) / CAST_FOR_FREE (don't), like
  yes / no; devour -> scalar = `len(land_ids) / 10`.

## 7. What the network sees at a pause

The adapter's state at a pause is the **pre-action snapshot** (the
replay harness rolls back), plus the history: the priority action that
paused (e.g. `CastSpell X`) and every earlier answer of this resolution
are its last tokens. The legal mask (resolution family) tells the policy
which kind of choice it is. The pending callback's own context (which
trigger / which effect is asking) is only implicit; a "pause context"
token is listed as a remaining shortcut in SEARCH_LIMITATIONS §5.

Policy-head slot subsetting: `StructuredModel.forward` accepts
`policy_slot_range=(lo, hi)` and runs the cross-attention decoder for
those slot queries only (each query is independent). `NNEvaluator` asks
for `[0, 517)` at non-pause states and the resolution block at pauses,
so (a) a v3 net costs no more per evaluation than v30 did, and (b) the
non-pause computation is the SAME shape as v30's -> bit-identical
logits for an extended checkpoint.

## 8. Versioning

* `action_encoding_version`: `positional_v1` / `canonical_v2` -> 517
  slots, no resolution slots; `canonical_v3` -> 805.
  `action_encoding.action_vocab_size(version)` is the single source;
  `ACTION_VOCAB_SIZE` stays 517 (the v1/v2 size every existing consumer
  and test assumed) and `ACTION_VOCAB_SIZE_V3 = 805`.
* Loaders derive the policy-head width from the checkpoint's
  `action_encoding_version` unless metadata records
  `structured_action_vocab_size` explicitly; a mismatch with the
  state dict is a loud `load_state_dict` error.
* A v1 / v2 structured checkpoint cannot prior a resolution pause.
  `NNPimcConfig.build` refuses to build it as a searched-resolution seat
  (`ValueError` naming the two fixes: extend it, or `+autotarget`) --
  never a silent flat prior.
* Shards: new npzs record `action_encoding_version` and
  `resolution_policy_rows` / `resolution_overflow_visits`. Legacy
  (unstamped, 517-wide) shards are zero-padded into a v3 training
  window only when the operator asserts their encoding with
  `--legacy-policy-encoding canonical_v2` (or
  `$MTG_BOT_LEGACY_POLICY_ENCODING`), mirroring `--shard-card-vocab`:
  width 517 cannot tell `positional_v1` from `canonical_v2`. The pad is
  logged with row / file counts. Any other width or a stamped version
  that differs from the model's is refused.

## 9. Hidden information at a pause

PIMC's sub-adapter is a clone of the real (paused) adapter, determinized
in place. The pre-action snapshot is re-taken from the determinized
state so a resolution answer can never roll back to the true state. Cards
the pause itself revealed to the decider are pinned (never permuted):
the tutor's matching library cards, the opponent's hand for a
Thoughtseize-style pick, the surveilled top card. Without the pin the
decider would choose "card id 41" in a world where id 41 carries a
different printing. `determinize(..., known_card_ids=...)` implements the
pin; ISMCTS's `_fresh_determinization` gets the same treatment.

## 10. Search priors at a pause

* **Heuristic PIMC** runs UCB1 (no priors): every option is expanded and
  evaluated; the legal order is the tie-break. Flat, by design -- the
  heuristic evaluator sees the result of each option once its subtree
  reaches the resolution (`SEARCH_HORIZON.md`: 100 % of cast trees
  contain the resolution).
* **NN PIMC (v3 checkpoint)** reads priors from the extended policy
  head. Freshly extended rows start near-uniform (N(0, 0.02) queries
  attend almost uniformly); search + value head decide until the
  resolution rows are trained.
* The opponent's resolution choices inside the tree are searched too
  (min/max) instead of being modelled as first-legal.

## 11. Keeping the trained network

`mtg_bot extend-resolution-slots` (`mtg_bot.training.resolution_extension`)
copies every parameter of a v2 structured checkpoint bit-exactly and
appends 288 rows to `policy_slot_queries`, drawn N(0, 0.02) (the model's
own init) from a `torch.Generator` seeded deterministically (default
seed recorded in metadata). Metadata keeps `value_calibration`,
`card_vocab`, trunk dims and calibration; it switches
`action_encoding_version` to `canonical_v3`, records
`structured_action_vocab_size = 805` and an `extra.resolution_extension`
provenance block (source version, source weights sha256, seed). The
s2 lineage retrains each cycle from scratch on its recent-shard window,
so the extended checkpoint is what keeps the trained network in the
loop: it drives the next cycles' self-play and is the gate's baseline.

## 12. Measurements

(filled in as they land)
