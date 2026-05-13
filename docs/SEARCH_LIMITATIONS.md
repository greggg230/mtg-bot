# Search and engine limitations / known correctness gaps

This document tracks places where our search or engine deliberately
diverges from optimal play or strict CR conformance, with the
rationale, the failure mode, and the conditions under which it would
matter.

These are NOT bugs to fix on a deadline — they're tracked here so
they don't get rediscovered every few months as "wait, that's broken"
and so we have a clear record of WHY each shortcut was taken.

---

## 1. Combat damage assignment is ordered (outdated CR)

**Location**: `src/mtg_bot/combat.py:_resolve_damage_step` (around line 419).

**What we do today**: when an attacker is blocked by multiple blockers,
the engine assigns damage in `attacker.blocked_by` iteration order,
applying lethal-before-next-blocker for all but the last blocker.
The last blocker absorbs all remaining damage (or it goes to the
player if the attacker has trample).

**Why this is wrong under current CR**: modern MTG rules let the
attacking player assign damage across blockers however they want —
no required order, no "must assign lethal before next." The attacker
can split a 5/5's damage as 1+1+3 across three blockers if they
choose.

**Failure mode**: the bot may "trade" creatures suboptimally. A 5/5
attacker into three 2/2 blockers should be able to deal 2+2+1 to kill
two and leave one wounded. Our engine forces 2+3+0 (kill first, dump
rest into second).

**When it matters**: rarely in early-training — neither side is
playing well enough for damage assignment subtlety to matter.
Increasingly important once the model is strong and combat is the
deciding factor in close games.

**Fix scope when prioritized**:
- Replace the in-order assignment with a damage-distribution
  decision per multi-blocker attacker
- Factored search-side decision: per attacker with N blockers,
  expose N-1 "how much to N-th blocker" decisions (the last one is
  forced — assign whatever's left)
- Engine `_resolve_damage_step` honors the assignment instead of
  computing it
- Tests pinning: assigning 1/1/3 across three blockers vs default
  2/3/0; trample interaction; deathtouch (1-damage-counts-as-lethal)
  interaction

**Impact estimate**: ~5-10% strength gain in mid/late training.
Negligible during early training.

---

## 2. Combat blocker assignment is factored (joint-pattern coverage)

**Location**: `src/mtg_bot/search/combat_decisions.py`,
`src/mtg_bot/search/engine_adapter.py` (`expose_combat_decisions=True`).

**What we do today**: declare-blockers is decomposed per-blocker.
Each blocker chooses independently from `[block attacker A, block
attacker B, ..., don't block]`. Search sees a depth-M tree of width
N+1, NOT a single 1296-way decision.

**Why this is sometimes wrong**: per-blocker decisions break the
coupling between blockers. "Block attacker A with blockers B and C
together to safely trade" requires search to discover the joint
pattern through MCTS visits. With a small iteration budget, the
search can prefer locally-good but jointly-bad blocking patterns.

**Failure mode**: example — opp attacks with a 5/5; defender has two
2/3 blockers. Optimal play: chump-block with one (cheaper trade),
take 5 damage to face. Sub-optimal jointly-correct: double-block
with both (use both blockers, kill the 5/5, lose both blockers).
Factored search may pick the joint-bad pattern that splits the
difference (one blocker blocks, other passes — same as chump but
explored sub-optimally because UCB1 picks each independently).

**Mitigation in place**:
- Tree-reuse across choose() calls (commit `5f1a202`) keeps
  block-assignment subtrees warm across decisions
- PUCT prior from the trained policy biases toward typical good
  joint patterns once the model has learned them
- Forced-move shortcut handles trivial cases (only one blocker)

**When it matters**: at very tight per-choose budgets (e.g., samples=1
nodes=50). At our production budget (samples=4 nodes=200 = 800 iters)
combined with PUCT prior, the factored search reaches reasonable
coverage. As model trains, the prior compensates for factoring loss.

**Fix scope if needed**: introduce a "block group" decision type
that lets search bind multiple blockers at once. Big refactor; not
worth it now.

**Impact estimate**: probably <5% strength gain even in late training.
Below the noise floor against the ordered-damage-assignment fix.

---

## 3. Priority auto-pass via capability hash (opp + active player)

**Location**: `MtgGameStateAdapter.legal_decisions` short-circuits to
`[PassPriority()]` when the priority holder's capability set is empty.
Capability checks live in `src/mtg_bot/search/_player_capabilities.py`
(`has_player_capability_to_act` + `player_capability_hash`); the
adapter cache lives at `src/mtg_bot/search/engine_adapter.py:208-228`
(`_opp_cap_cache` 4-tuple keyed on `(player_id, is_active_player,
hash, can_act)`).

Two flavors:

- **Opp side** (priority on the non-active player): instant-speed-only
  capability check — cards in hand castable at instant speed,
  warp-from-graveyard / impulse-from-exile at instant speed.
- **Active side** (priority on the active player): instant-speed
  union plus, when at sorcery-window timing (stack empty + main
  phase + active priority), sorcery-speed castables, land plays with
  drops remaining, sorcery-speed warp / impulse / Lightstall casts,
  activated abilities (sorcery-speed), and spacecraft station crew.
  At pre-combat / declare-attackers steps, eligible-attacker
  presence is a capability too (defensive — the engine_adapter's
  factored combat phase already handles attackers ahead of priority,
  but a future code path that opens a priority window before
  `_combat_phase` is set won't silently auto-pass through what
  should be an attacker declaration).

The capability check is computed once per priority moment and
re-validated by hashing
`(player.hand, player.mana_available,
player.untapped_permanents_with_abilities, gy_flags, exile_flags,
step/turn/stack/priority/active/continuous_effects_count)` plus, for
active player, `(lands_played_this_turn, is_sorcery_window, life)`.
Cache invalidated on every `apply()`.

**Why this is sometimes wrong**: the hash inputs don't capture every
state change that could affect the priority holder's possible
actions. The same documented limitations apply equally to opp and
active-player capability checks (the same hash machinery feeds
both):

| Edge case | Why missed | Bot misbehavior |
|---|---|---|
| Continuous effect grants the priority holder new instant-speed access (Vedalken Orrery, Teferi Mage of Zhalfir grants them flash) | Hash doesn't depend on continuous-effects layer beyond the count | Bot auto-passes when the priority holder could have cast a (now-flash) creature |
| Cost-reduction effect lands mid-turn (Goblin Electromancer ETBs from active player's spell → opp's / active's instants get cheaper) | Hash uses raw mana_available, not effective costs | Bot auto-passes when the priority holder could afford something whose cost just got reduced |
| Continuous effect grants a permanent a new activated ability (rare) | Hash captures id-set count delta, not effective set | Bot auto-passes when the priority holder could activate the granted ability |
| Effect makes spells castable from a new zone (Yawgmoth's Will → graveyard cast; very rare in current pool) | Zone-castable flags watched at coarse granularity | Bot auto-passes when the priority holder could cast from graveyard |
| Mana-cost change via card-type change (e.g., color-changing effect altering hybrid cost availability) | Hash uses face-value cost, not derived | Bot auto-passes if a now-castable cost wasn't recognized |

**Failure mode**: bot passes priority when the priority holder had a
meaningful action available. Note this is conservative-WRONG — the
bot's miss is "didn't take a chance to act" — never "took an illegal
action" or "made an impossible play."

**Engine-contract interaction**: today's `legal_actions` enumerator
returns early at non-sorcery timing BEFORE walking the activated-
ability block (see `legal.py` — `if not _is_sorcery_timing: return
actions` after the hand-cast loop). This means BOTH opp at non-active
priority AND active player during instant-only windows (combat
substeps, upkeep/draw/end with non-empty stack, etc.) never receive
`ActivateAbility` actions from the engine. The capability check
matches that contract: at non-sorcery timing,
`_has_instant_speed_activatable` returns False unconditionally. If
a future engine change DOES start surfacing `ActivateAbility` at
non-sorcery timing, the capability-check function is the place to
extend (the structural hash already covers permanent shape).

**When it matters**: very rare in the current EoE pool + synthetic
G/R deck. Common enough in formats with lots of cost-reduction or
flash-granting effects (Modern, Legacy, Commander) that this would
need to be tightened up before the bot enters those formats.

**Mitigation**: the hash is conservative — it includes more state
than strictly needed. False negatives (capability claimed empty
when it isn't) are the dangerous case; false positives (hash says
"changed" when capability actually didn't) just trigger an
unnecessary recompute. We err toward the latter.

**Acceptable for**: training (bot sees millions of games; rare
mis-plays don't shift the policy gradient).

**NOT acceptable for**: tournament play, real-money matches, "bot
plays one game live" scenarios where every decision matters.

**Fix scope if/when needed**:
- Include continuous-effect "grants the priority holder instant-speed" predicates in the hash
- Track effective costs (cost-reduction-aware mana check) instead of raw mana
- Specifically scan for `ContinuousEffect` types known to affect action availability
- Add a config flag `--no-capability-autopass` for matches where strict play matters

**Impact estimate**: combined opp + active auto-pass currently fires
on **~84% of priority moments** in random-walk RandomAgent self-play
(seed 0, 500 decisions; opp-only was ~45%). Per-game self-play
profile (v12 weights, samples=4 inner_nodes=200, ISMCTS, base_seed=200,
decision_cap=200): clean wall **52.2s → 48.8s (-6.5%)**;
`legal_actions` calls **433,526 → 156,337 (-64%)** on top of the
opp-only baseline. Strength cost in mis-played positions: negligible
at current model strength + EoE-only pool. See
`docs/perf_active_autopass_20260504.md` for the full delta.

---

---

## 4. Floating mana not modeled — no standalone tap-for-mana action

**Location**: `src/mtg_bot/actions.py` (action class registry), `src/mtg_bot/legal.py` (action enumeration). There is no `TapLandForMana` action class. Mana production happens only inside `_pay_mana_cost` during spell/ability cost payment.

**What we do today**: when the player needs to pay a mana cost, the engine auto-taps untapped lands (and consumes floating mana if any from continuous-effect sources) IN ONE ATOMIC OPERATION. Mana enters and leaves the pool inside the cost-payment step. The player never sees "mana is in pool, decide what to do next."

**Why this is wrong under CR**: real MTG lets a player activate a mana ability at any priority moment, generating mana into their mana pool. The mana persists until end of phase/step. Players can stack up mana from multiple sources, then spend it on a single spell or ability.

**Failure mode**: examples where this matters —
- **X-cost spells**: in real MTG, a player can tap several lands first (floating the mana) and THEN cast Fireball X=5. Our engine bundles tap-for-mana INTO the X-cost cast, which is fine in simple cases but breaks when X is supposed to be chosen based on available mana.
- **Cards counting mana spent**: e.g., "spells you cast cost {1} less if you spent {U} this turn" / Storm-style cards that count spells with mana. Mana spent this phase isn't tracked because mana never persists.
- **Mana ability with sacrifice cost** (e.g., "Sacrifice this artifact: Add {3}"): in real MTG you'd activate it, the artifact is gone, then later cast something with the floating {3}. Our engine doesn't expose the option.
- **Mana doublers not interacting correctly**: a player who wants to "tap, double, accumulate, cast big" can't. (Some doublers DO work since they trigger in the cost-payment burst.)
- **Bouncing-and-replaying lands for mana**: niche, doesn't work without floating.

**When it matters**: rare in current EoE pool (no Storm cards, no mana doublers, X-cost spells are simple). Will matter much more in formats with mana acceleration archetypes (Modern, Legacy, Commander).

**Mitigation**: none currently — this is a wholesale "feature not implemented" situation, not a workaround.

**Fix scope**: substantial.
- Add `TapLandForMana` and `ActivateManaAbility` action classes (no stack — mana abilities don't go on the stack per CR 605)
- Track `mana_pool` already exists on Player; just need pool-add semantics + emptying at end of phase/step (CR 106.4)
- `_pay_mana_cost` would consume from pool first before tapping
- `legal_actions` enumerates tap-for-mana actions when pool isn't full or when conserving lands isn't the obvious choice (heuristic to prune the otherwise-large branching factor)
- Tests: X-cost casting from floating mana, mana doubler interactions, mana persistence to end of phase

**Impact estimate**: <1% strength gain in current EoE pool. Substantially more in formats where it matters.

---

## 5. Resolution decisions are NOT exposed in production training (vocab gap)

**Location**: `src/mtg_bot/training/data_gen.py:503` and `src/mtg_bot/training/self_play.py:438` construct `MtgGameStateAdapter(engine, expose_combat_decisions=True)` with the implicit `expose_resolution_decisions=False` default. Production data-gen and self-play therefore route every mid-resolution callback (target picks, sacrifice choice, X-cost, modes, "may" yes/no, surveil keep/mill, directed discard, devour land subset, etc.) through `DefaultTacticalAgent` (`src/mtg_bot/search/tactical.py`).

**What we do today**: at any mid-resolution decision point, the engine asks the controlling player's installed agent — which the adapter has rewritten to `DefaultTacticalAgent`. Defaults: `choose_target → first legal`, `choose_sacrifice → first legal`, `choose_x → upper_bound` (commit-everything), `choose_mode → 0`, `choose_modes → [0]`, `choose_yes_no → True` (opts into ALL "may" effects), `choose_surveil → False` (always keep), `choose_discard → first legal`, `choose_devour_lands → []`. None of these go through search; none of them get policy supervision.

**Why we can't just flip the flag**: the adapter and search infrastructure for `expose_resolution_decisions=True` are fully implemented (`ResolutionChannelAgent`, the pause / rollback / replay harness, the `Resolution*Decision` types, search-side handling in PIMC and ISMCTS). The dedicated tests in `tests/search/test_resolution_integration.py`, `tests/search/test_pimc.py`, `tests/search/test_info_set.py`, and `tests/search/test_engine_adapter_legal_cache.py` exercise the path with the flag on. **What's missing is the policy-head vocab.** `src/mtg_bot/training/action_encoding.py` has zero slots for any `Resolution*Decision` subclass; `ACTION_VOCAB_SIZE = 517` covers PassPriority, PlayLand, CastSpell, UnlockRoomHalf, ActivateAbility, ActivateStation, TurnFaceUp, AttackerDecision, BlockerDecision and nothing else. `encode(d, state, viewer)` raises `EncodingOverflowError` for every Resolution*Decision; `legal_action_mask` silently drops them.

**Failure mode if the flag were flipped without extending the vocab**:
- `data_gen.py` line ~757 inside `_play_one_game_recording`: `slot = encode(d, engine, active)` raises for every resolution-pause visit; the `except EncodingOverflowError: continue` then silently drops them. At a resolution-pause sample point `counted` ends up at zero and the row falls through to `active_policy = zero_policy / zero_mask`. **Every resolution-pause sample is silent no-op policy supervision** — the row trains nothing. This is exactly the "silent no-op pipelines are pipeline bugs" anti-pattern in `feedback_silent_noop_pipelines.md`.
- `legal_action_mask` at a resolution-pause moment returns all-False (every Resolution*Decision is `EncodingOverflowError`-skipped). NN evaluator's `evaluate_with_policy` defensively returns `np.zeros(vocab_size)` when `mask_np.any() == False` — so the NN delivers ZERO prior at every resolution-pause leaf. PIMC / ISMCTS PUCT then has flat priors over Resolution children; tree search still backs Q-values up, but the network contributes nothing.
- `decode(slot_idx, state, viewer)` cannot return any Resolution*Decision because no slot maps to one. If a search loop ever picked a slot from the policy distribution at a resolution-pause moment (e.g., via fallback or a bug), it would deserialize back to a priority action; the adapter's `_apply_resolution_decision` does `isinstance(decision, expected_type)` and would `TypeError` immediately.

**Strength cost of leaving the flag off (today)**: the NN never gets supervision on target picks / X-values / mode choices / surveil keep-vs-mill / "may" triggers / directed discards / sacrifice picks. These are the highest-leverage tactical choices in MTG and they are all locked at `DefaultTacticalAgent`'s deterministic first-legal / commit-everything / always-yes defaults. The bot picks the wrong target, sacrifices the wrong creature, opts into bad mandatory triggers via `choose_yes_no = True`, and always X=max even when smaller X is correct. Estimated strength cost: substantial — meaningfully larger than any single combat-side limitation in this doc.

**Fix scope when prioritized**:
- Extend `action_encoding.py` to add slot ranges for each `Resolution*Decision` subclass:
  - `RESOLUTION_TARGET[target_kind][target_pos_within_kind]` — needs a stable kind enumeration (battlefield-permanent / hand-card / graveyard-card / player-id / decline) and a per-kind position cap. Battlefield-permanent positions are tricky because they reference both controllers' permanents (can't share `bf_own_pos`).
  - `RESOLUTION_DISCARD[hand_pos]` and `RESOLUTION_SACRIFICE[bf_own_pos]` — both reuse existing positional axes.
  - `RESOLUTION_X[x_value in 0..X_VALUE_CAP]` — needs an X cap (observed: 0-7 in current pool; 16 is generous).
  - `RESOLUTION_MODE[mode_index in 0..MODE_INDEX_CAP]` — modal "choose one" for 1-of-N; cap at 4 (longest current modal is 3 modes).
  - `RESOLUTION_SURVEIL[mill in {True, False}]` — 2 slots.
  - `RESOLUTION_SHOCK_LIFE[pay in {True, False}]` — 2 slots.
  - `RESOLUTION_YES_NO[yes in {True, False}]` — 2 slots.
  - `RESOLUTION_CARD_FROM_LIBRARY[lib_pos]` — needs a library-position cap; tutors typically search the whole library, so the cap is the deck size (60 in standard, can blow out the vocab if naively encoded). Alternative: encode by printing-id within the deck pool's vocabulary. Bigger design call.
  - `RESOLUTION_CARD_FROM_OPPONENT_HAND[opp_hand_pos]` — uses opponent hand position; valid because the caster has revealed the hand at this point.
  - `RESOLUTION_DEVOUR_LANDS[subset_size]` — already V1-collapsed to (empty, full); 2 slots.
- Update `encode` / `decode` / `legal_action_mask` / `_enumerate_legal` to handle the resolution-pause adapter phase (analogous to the existing combat phases — `_combat_phase` already returns kind tags from the adapter; add resolution-kind tags too).
- Update `replay_decisions.py` (which already covers resolution decisions for serialization, no change there).
- The NN architecture: extending `ACTION_VOCAB_SIZE` reshapes the policy head's output `Linear(h_last, ACTION_VOCAB_SIZE)`. **Existing trained weights become unusable** — the policy head will need to be re-initialized for the new slot count and trained from scratch. Value head and feature encoder are unchanged.
- Tests pinning: `test_vocab_size_constant_is_fixed` updates to the new `ACTION_VOCAB_SIZE`. New tests for `encode` / `decode` round-trip at each resolution-pause kind. `legal_action_mask` produces non-empty masks at resolution-pause moments. End-to-end test that `_play_one_game_recording` with `expose_resolution_decisions=True` writes non-zero policy-supervision rows for resolution-decision sample points.
- After all of the above lands, flip the flag in `data_gen.py:503` and `self_play.py:438`.

**Bootstrap concern after the flag flip**: the trained NN policy head will be re-initialized (new vocab size). Iter-1 onward will have to learn target-picking / X-choice / mode-choice from scratch, on top of learning everything else. For the first few iters, search at resolution-pause moments will be effectively flat-prior PUCT — strictly weaker than `DefaultTacticalAgent`'s first-legal heuristic in those few iters. Long-term gain is large; short-term may regress one or two iterations.

**Impact estimate**: large (10s of percentage points strength once trained). The cost of leaving it off scales with how often resolution decisions arise — every burn spell, every modal spell, every X-cost cast, every "may" trigger, every Mind-Rot-style discard. Already significant in the current EoE pool; will only grow as more cards come online.

**Why this is doc'd here instead of a code fix**: the fix is a substantial vocab extension + NN-retrain that warrants its own dedicated PR. This entry exists so the gap is tracked rather than rediscovered. See `feature/expose-resolution-decisions` history (closed without code changes; documentation-only update of this entry) for the audit that produced this writeup.

---

## 6. Phase-based active-player priority prune (training heuristic)

**Location** (planned, in `feature/phase-priority-prune` as of 2026-05-04): `MtgGameStateAdapter.legal_decisions` short-circuits to `[PassPriority()]` when ALL of these hold:
- Active player == priority player == viewer
- Stack is empty
- Current step is one of: **upkeep, draw step, beginning of combat, end step**

The check fires BEFORE the capability hash check (it's cheaper). Capability check still runs for opp priority and for the active player in other phases.

**Why this is sometimes wrong** under CR: real MTG allows casting instants at any priority moment. There ARE legitimate plays in these phases:

| Edge case | Phase | Example | Bot misbehavior |
|---|---|---|---|
| Pump spell before upkeep trigger | Upkeep | "At the start of your upkeep, do X" where X depends on creature P/T | Bot doesn't pump first |
| Mana sink at end of own turn | End step | Floating mana about to disappear; cast instant to use it | Mana wasted (also moot until floating mana modeled — see entry #4) |
| Modal spell missed in main | End step | Forgot to cast a flash creature in main phase; end step is last chance | Bot doesn't cast it |
| Combat trick before declaring attackers | BoC | Pump a creature so it can attack profitably; instant removal on opp blockers; tap opp's blockers | Bot misses these |
| Pre-empting opp's response window | Various | Cast a spell now so opp must respond before their planning starts | Bot doesn't pre-empt |
| Triggers requiring response | Any | "Whenever you cast a spell" triggers timed for these phases | Not fired |

**Failure mode**: bot passes priority and misses a legitimate instant-speed play. Conservative-WRONG (never plays an illegal action; just doesn't take a legal one). The miss happens on the active player's side, where the held-back action would have been their own choice — opp's response windows are unaffected.

**Highest-risk phase**: **beginning of combat**. This is where genuine combat tricks live (pump-then-attack, instant removal of blockers). A trained model would value these. v12 (near-random) doesn't know how to use them anyway.

**When it matters**:
- For training: misses < 0.5% of decisions; all in niche cases the lightly-trained NN couldn't exploit anyway. Bot doesn't learn the niche tactics, but the policy gradient isn't measurably affected.
- For tournament play: real loss. Combat tricks matter. End-step plays matter. Bot gives up clear value here.

**Mitigation**: configurable via the `--no-phase-priority-prune` flag. Default ON for training and self-play data_gen; should be set OFF for tournament/eval matches against real opponents (or against alternate-arch baselines where you want fair search behavior).

**Acceptable for**: training (millions of games; rare miss doesn't shift the gradient).

**NOT acceptable for**: live tournament play, "bot plays one game live" mode, any benchmark whose strength comparison depends on subtle combat or end-step decisions.

**Fix scope if/when needed**: the heuristic is intentionally crude. To recover the missed plays without the runtime cost:
- Train the policy to predict "should this priority moment be skipped?" — let the NN decide rather than a hardcoded phase rule
- Or expose phase + stack-empty as a feature to the policy and let it learn
- Or just disable the heuristic for late-stage training/eval

**Impact estimate**: ~15-30% per-game wall reduction during training, on top of capability auto-pass (entry #3). Composes well — capability catches "no castable spells" case; phase prune catches "have castable spells but wrong phase" case.

---

## 7. Forced second-main land drop (training heuristic)

**Location**: `MtgGameStateAdapter.legal_decisions` (`src/mtg_bot/search/engine_adapter.py`) — see `_should_force_landdrop()`. After the full action enumeration, `PassPriority` is filtered OUT of the returned decisions when ALL of these hold:

- `force_landdrop_enabled` is True (default for SelfPlayJobSpec, OFF for EvalJobSpec)
- Active player == priority player
- Stack is empty
- Step is `POSTCOMBAT_MAIN`
- `lands_played_this_turn == 0`
- ≥1 land in active player's hand
- Active player controls fewer than six lands

The player can still cast spells / activate abilities — but cannot end the priority window without playing a land at some point.

**Why this is sometimes wrong** under CR: real MTG allows the active player to skip their land drop. There ARE legitimate reasons to do so:

| Edge case | Example | Bot misbehavior |
|---|---|---|
| Hold land for opp's land-destruction trigger | Strip Mine in opp hand at sorcery speed next turn; better to deploy after | Bot gives up the land too early |
| Reveal-from-hand trigger that scales with lands in hand | "Whenever an opponent reveals X, do Y" — landfall payoff opposite | Bot loses the trigger value |
| Bottom-of-the-pile color screw | Hand has only off-color land; playing it does nothing useful | Bot wastes a turn on a dead land |
| Mulligan-style overcommit | Hand is 6+ lands and you've capped at 5; another land just floods | Mitigated by the 6-land guard, but 5-land overflow still possible |
| Land-cycling to thin the deck | Cycling a fetchland for tempo when no reasonable play exists | Bot doesn't see the option (cycling not modeled either, see entry #4) |

**Failure mode**: bot is forced to play a land when the optimal play is to hold it. Conservative-WRONG in most cases — playing a land when you have <6 in play is correct ~95% of the time on the EoE card pool, and the heuristic explicitly opts out when you're already flooded. The miss is on niche edge cases that the lightly-trained NN couldn't reason about either.

**Why this isn't pure pruning** (and why it's #7 not #6.5): unlike the phase-prune (which short-circuits BEFORE legal_actions), this filter runs AFTER full enumeration and only removes one option (`PassPriority`). It's a play-quality heuristic disguised as a search filter, not a perf optimization.

**When it matters**:
- For training: bot ALWAYS plays its land at second main when conditions match — removes the "lazy NN holds lands forever" failure mode that early-stage policies fall into. Net positive for training signal quality.
- For tournament play: small loss. The held-land cases above are real but rare; expert-level deck-building works around them. Default OFF for eval to be safe.

**Mitigation**: `force_landdrop_enabled=False` on `MtgGameStateAdapter` (or via SelfPlayJobSpec / EvalJobSpec). EvalJobSpec defaults OFF; SelfPlayJobSpec defaults ON.

**Acceptable for**: self-play training (the policy benefits from never seeing the "hold land for no reason" trajectory).

**NOT acceptable for**: tournament play, eval matches against real opponents, any test where land-timing optimality matters.

**Fix scope if/when needed**: trivial to disable per-spec. To replace with something better, add a "should I play this land?" feature to the policy network and let the NN decide. Same shape of fix as the phase-prune (entry #6).

**Impact estimate**: not a perf win (filter runs AFTER legal_actions, no work skipped). The win is play-quality: trained models from runs WITH the filter ON should hit "land every turn through ~5 lands" reliably, which is a learnable pattern but a slow one without the inductive bias.

---

## 8. Lethal-detect alpha-strike heuristic (training optimization)

**Location**: `src/mtg_bot/search/_lethal_detect.py`,
`MtgGameStateAdapter.legal_decisions` is unaffected — the
short-circuit lives upstream in
`ISMCTS.choose_with_visits` (around line 411 with the
`enable_lethal_detect` constructor flag, default True). When the
adapter is at an `AttackerDecision` priority moment AND
`detect_lethal_attack` returns a non-None lethal subset, the search
loop is skipped and `{decision: 1}` for the matching `AttackerDecision`
is returned.

**What this is**: a HEURISTIC, not a CR-correct optimization. It
is NOT a lossless search shortcut; it's a pattern-matched
short-circuit that fires only when ALL of these hold:

- Single opponent.
- Opp has empty hand (no instants / sorceries to react with).
- Opp has no non-mana activated abilities on any permanent.
- Opp has no triggered abilities keyed on combat events
  (`attacks` / `blocks` / `deals_combat_damage` / `dealt_damage`).
- Opp controls no noncreature permanent with a self-animation
  activated ability (manland-style — `BecomeLandCreatureUEOT` on
  `printing.activated_abilities` of a noncreature permanent).
- None of opp's blockers have first/double strike, indestructible,
  hexproof, or shroud.
- With opp greedily blocking the biggest attackers, the smallest
  unblocked attackers' summed power (minus any lifelink lifegain
  from blockers) clears opp's life total.

When fired, the heuristic emits "attack with all eligible attackers"
since sending fewer can only reduce unblocked damage. The full set
short-circuits ALL per-attacker sub-decisions in the same combat
phase (each sub-decision's `choose_with_visits` call independently
recomputes the heuristic against the same engine state and routes
through it).

**Why this is sometimes wrong**: 

| Edge case | Why missed | Bot misbehavior |
|---|---|---|
| Opp could topdeck a sweeper next turn | We don't see the unrevealed library | Heuristic fires; opp clears the board on their turn — but that doesn't matter because we KILLED them this turn anyway. NOT a misbehavior. |
| Opp has prevent / damage-redirection effects | Continuous-effect detection isn't part of the gates | Bot commits to attacks; combat resolves with prevented damage, opp survives. |
| Opp is a player whose loss condition isn't "life ≤ 0" (poison, milling triggers) | Heuristic only checks `opp.life <= 0` after damage | Bot commits to attacks; opp's actual loss path may not trigger. |
| Damage assignment to multiple blockers wouldn't trade favorably | Heuristic doesn't reason about post-block board state | The bot "wins" but if the kill was actually impossible (e.g., damage prevention, blocker abilities trigger before damage), the bot loses. The CR-correct gates above are designed to rule out these cases — if a gate is missing for a real card, the bot loses the game. |
| First strike on OUR side wasn't checked (CR + factor) | We don't model that strike-step damage helps us; the heuristic counts NORMAL damage step | Conservative — first strike on attackers can ONLY help us (more damage chances), so missing it just means false-negatives (some lethals we don't detect). Safe. |
| Non-life-loss kill conditions (mill / commander / poison) | Not supported | Heuristic always uses life-total; bot relies on search for those win paths. |
| Future self-animation effect class (artifact-animator, copy-of-creature animator, etc.) | Animator gate only checks `isinstance(ability.effect, BecomeLandCreatureUEOT)` — the only self-animation Effect class in the engine today | Bot commits to attacks; opp animates a noncreature artifact / token-source into a fresh blocker, soaks the smallest unblocked attacker, we lose. **When adding a new self-animation Effect class, append it to `_SELF_ANIMATION_EFFECT_CLASSES` in `_lethal_detect.py`.** |

**Failure mode**: bot commits to attack-everything (which is what
the heuristic does today). If the math was wrong and the kill
doesn't actually happen, our creatures are tapped, our turn ends,
opp kills us. **Catastrophic when wrong**, hence the conservative
posture in `_lethal_detect.py` — every gate aborts on any potential
disruption.

**Safety contract**: the gates list here MUST stay in sync with
the implementation. Any new combat-disrupting card mechanic
(damage prevention, redirected damage, "creatures attacking gain
shroud", etc.) needs to either:
1. Be detected by an existing gate (e.g. flagged as a
   non-mana activated ability or combat trigger), OR
2. Add a NEW gate that aborts when the mechanic is present.

**When it matters**:

- **Training**: heuristic fires on guaranteed-lethal turns; bypasses
  the per-turn search budget allowing earlier (uncertain) decisions
  to consume more budget. Per-turn wall savings + cleaner
  end-game training signal (value target = +1.0 with confidence;
  policy target concentrated on the lethal play).
- **Tournament play**: same wins, plus opp's plausible
  response is often already captured by the gates (empty hand
  rules out instants; no combat triggers rules out fog effects).
  Disable via `enable_lethal_detect=False` if you want strict
  search-derived recommendations and full visibility into the
  search policy at end-game positions.

**Mitigation in place**:

- Conservative-ABORT posture: false-positive lethal (declaring win
  when we can't actually win) is catastrophic; false-negative (missing
  real lethal, falling through to search) is just a missed
  optimization. Every gate errs toward false-negative.
- Counter `ISMCTS._lethal_detect_fires` exposes firing rate for
  benchmarking; reset via `reset_lethal_detect_count()`.

**Acceptable for**: training, self-play, casual play.

**Caveats for**: tournament play (still safe correctness-wise; just
hides search-based combat reasoning from the policy).

**Fix scope if/when needed**:
- Extend gates to recognize damage-prevention / redirection effects
  (continuous-effect search at the GameState level).
- Optimize the "send minimal subset" decision so the bot doesn't
  always commit ALL eligible attackers (let some stay back for
  opp's potential counter-attack next turn — but if we detected
  lethal, there IS no next turn for opp; this is a dead optimization).
- Add support for non-life-loss kill conditions
  (poison counters, "lose the game" rider abilities).

**Impact estimate**: minor wall savings per game (depends on
end-game frequency); the bigger win is training-signal quality at
the moment-of-truth turn. Expected per-game wall delta: 1-3% (most
games don't reach an alpha-strike-with-tapped-out-opp position; when
they do, the heuristic wipes 5-15 attacker sub-decisions of search).

---

## 9. Skip-empty-combat priority short-circuit (training heuristic)

**Location**: `MtgGameStateAdapter.legal_decisions` (`src/mtg_bot/search/engine_adapter.py`) — see `_should_skip_empty_combat()`. Returns `[PassPriority()]` immediately (no `legal_actions` enumeration) when ALL of these hold:

- `skip_empty_combat_enabled` is True (default for `SelfPlayJobSpec`, OFF for `EvalJobSpec`)
- Not terminal / not resolution-paused / not in a factored combat sub-decision
- Stack is empty
- Step is one of: `DECLARE_ATTACKERS`, `DECLARE_BLOCKERS`, `COMBAT_DAMAGE`, `END_OF_COMBAT`
- No creature on the battlefield is currently attacking

Fires for **both** active and opponent priority windows: with no attackers there's nothing to pump, redirect, or remove with combat tricks, so opp's instant-speed window is also dead.

**Why this is sometimes wrong** under CR: real MTG allows priority during these windows even with no attacks. Edge cases the heuristic skips:

| Edge case | Step | Bot misbehavior |
|---|---|---|
| Cast non-combat instant during empty combat phase | Any of the 4 | Misses end-of-step effects timed to combat (rare) |
| Counterspell setup for next turn | Any of the 4 | Misses pre-emptive instant timing |
| "At end of combat" triggers fire even with no combat | EoC | The trigger still fires (engine handles), but bot can't respond to it with an instant |
| "Whenever a player passes priority during combat..." trigger | Any of the 4 | Hypothetical card mechanic; would need a new gate |
| Sacrifice for end-of-turn benefit timed at EoC | EoC | Better timed at end step proper anyway, so usually OK |

**Failure mode**: bot passes priority during empty combat steps even when an instant could be cast. Conservative-WRONG (never plays an illegal action; just skips windows where instants are *technically* legal but rarely useful with no attackers).

**Why this composes well with phase-prune** (entry #6): phase-prune already covers `BEGINNING_OF_COMBAT` and `END` steps for active player. This entry covers the OTHER combat sub-steps — and crucially fires for opp priority too (phase-prune is active-only). The capability auto-pass (#3) catches "opp has nothing castable" but doesn't fire when opp has castable instants, which is the case this entry adds.

**When it matters**:

- For training: removes ~10 priority moments per affected combat phase from the search tree. Fires reliably turns 1-2 (no creatures yet) + occasional board-wipe turns. Composes with phase-prune for ~15-20% additional wall reduction estimated.
- For tournament play: marginal correctness loss. Empty-combat-instant plays are rare; expert decks rarely need them. Default OFF for eval.

**Mitigation**: `skip_empty_combat_enabled=False` on `MtgGameStateAdapter` (or via the Spec). `EvalJobSpec` defaults OFF; `SelfPlayJobSpec` defaults ON.

**Acceptable for**: training, self-play data_gen.

**NOT acceptable for**: tournament play, eval matches against real opponents, any test where empty-combat-instant timing matters.

**Fix scope if/when needed**: add gates for triggered abilities that depend on combat steps with no attackers, OR train the policy to skip these windows on its own (same shape of fix as phase-prune).

**Impact estimate**: ~15-20% per-game wall reduction during training (estimated; needs measurement). Composes additively with phase-prune (#6) since it covers the combat sub-steps phase-prune misses.

---

## 10. Skip our priority on opponent's POSTCOMBAT_MAIN (training heuristic)

**Location**: `MtgGameStateAdapter.legal_decisions` (`src/mtg_bot/search/engine_adapter.py`) — see `_should_skip_opp_postcombat_main()`. Returns `[PassPriority()]` immediately when ALL of these hold:

- `skip_opp_postcombat_main_enabled` is True (default for `SelfPlayJobSpec`, OFF for `EvalJobSpec`)
- Not terminal / not resolution-paused / not in a factored combat sub-decision
- Stack is empty
- Step is `POSTCOMBAT_MAIN`
- Player with priority is the **non-active** player (we're being asked to act on opp's turn)

**Why this is sometimes wrong**: real MTG allows priority during opp's main 2 with empty stack. Edge cases the heuristic skips:

| Edge case | Bot misbehavior |
|---|---|
| Counterspell on opp's about-to-be-cast spell next turn | Misses the pre-emptive timing (rare; opp would normally respond at end of turn) |
| Cast a flash creature on opp's end of turn | Better timed AT end step itself (entry #6 phase-prune already auto-passes our active end step, but the natural end-of-opp-turn flash window is at OPP'S end step, not their main 2) |
| Activate a "do X this turn" ability before EoT | Same — usually timed at end step proper |
| Trigger that fires during opp's main 2 | Engine handles trigger firing; we just can't respond with an instant |

**Failure mode**: bot passes priority during opp's main 2 even when an instant could be cast. Conservative-WRONG. The natural reactive timing for our instants on opp's turn is usually:
- BEFORE opp's combat (during their precombat main, where capability auto-pass already fires)
- AT opp's end step (where this heuristic doesn't apply)

So the dead window we're skipping is genuinely a low-value priority moment for our instant-speed plays.

**Why this complements capability auto-pass** (entry #3): capability auto-pass fires for the priority holder when they have nothing castable. This entry catches the remaining case: it IS opp's turn, we DO have a castable instant, but acting at opp's main-2 dead window is essentially never optimal. The capability check would let us through; this rule doesn't.

**When it matters**:

- For training: removes 1-2 priority moments per opp turn (we get priority once during opp's main 2; capability check fires when we can't act, this fires when we can but shouldn't). Estimated 5-6% wall delta.
- For tournament play: marginal correctness loss. Pre-emptive instant timing on opp's main 2 is a niche play. Default OFF for eval.

**Mitigation**: `skip_opp_postcombat_main_enabled=False` on `MtgGameStateAdapter` (or via Spec). `EvalJobSpec` defaults OFF; `SelfPlayJobSpec` defaults ON.

**Acceptable for**: training, self-play data_gen.

**NOT acceptable for**: tournament play, eval matches, any test where pre-emptive instant timing matters.

**Fix scope if/when needed**: trivial to disable per-spec. To recover the lost plays, train the policy to value these windows directly (same shape of fix as the other priority short-circuits).

**Impact estimate**: ~5-6% per-game wall reduction during training. Composes additively with capability auto-pass (#3) since it catches the "have castable but pointless" case capability misses.

---

## 11. Skip-dead-main1 priority short-circuit (training heuristic)

**Location**: `MtgGameStateAdapter.legal_decisions` (`src/mtg_bot/search/engine_adapter.py`) — see `_should_skip_dead_main1()`. Returns `[PassPriority()]` immediately when ALL of these hold:

- `skip_dead_main1_enabled` is True (default for `SelfPlayJobSpec`, OFF for `EvalJobSpec`)
- Not terminal / not resolution-paused / not in a factored combat sub-decision
- Stack is empty
- Step is `PRECOMBAT_MAIN`
- Player with priority IS the active player
- Active player controls 0 creatures on the battlefield
- Active player has no card in hand whose printing has the printed `haste` keyword
- Active player controls no noncreature permanent with an activated ability that animates it (currently checks for `BecomeLandCreatureUEOT`; see fragility note below)

The "no creatures + no haste + no animator" conjunction means nothing the player does at main 1 has a different effect from doing it at main 2 — there's no creature to attack with even if a creature gets cast or animated. Auto-passing main 1 is functionally equivalent to playing through it.

**Why the manland exception**: a creature-land like Mishra's Factory in play means the player CAN attack this turn (animate, then attack). Skipping main 1 in that case loses the chance to play extra lands or cast spells before deciding to animate. The check walks the active player's noncreature permanents looking for an `ActivatedAbility` whose `effect` is `BecomeLandCreatureUEOT`. Mana abilities are skipped (`is_mana_ability == True`).

**Fragility — manland detection is conservative-WRONG**: the `isinstance(ability.effect, BecomeLandCreatureUEOT)` check covers the only self-animation Effect class today, but a future card could introduce a NEW Effect class for artifact-self-animation or other animator patterns. If that lands without updating this check, the heuristic would fire when it shouldn't (false-positive), causing the bot to skip main 1 with an animator available. **Mitigation**: when adding a new self-animation Effect class, add it to the isinstance tuple here (and to the related lethal-detect gate in `_lethal_detect.py`).

**Why this is sometimes wrong** (other than the manland fragility):

| Edge case | Bot misbehavior |
|---|---|
| Cast a land-cycling card or ramp spell at main 1 to enable a turn-2 play | Misses if the card itself has no haste keyword |
| Activate a non-mana ability that needs sorcery timing AND must happen before combat | Edge case; usually a planeswalker activation, none in EoE today |
| Play a turn-1 mana ramp / scry / brainstorm-style effect that wants to happen NOW | Effect still happens at main 2; functional difference is small |
| Cast a sorcery whose effect benefits combat (e.g. tutor for removal) | Tutor at main 1 vs main 2 differs only if the tutored card needs to be cast pre-combat — which our heuristic explicitly rules out (no creatures, no haste) |

**Failure mode**: bot passes priority during dead-main-1 turns even when a sorcery COULD be cast. Conservative-WRONG. The miss is on the rare "I have a sorcery that affects combat OR my opp's combat" case which is genuinely rare on turns where I have 0 creatures and no haste.

**When it matters**:

- For training: removes ~2-4 priority moments per affected turn (active main 1 + opp's response window). Fires reliably turns 1-2 (no creatures yet); rarely after that (once you cast a creature, it sticks). Estimated 2-4% wall delta.
- For tournament play: marginal correctness loss. Default OFF for eval.

**Mitigation**: `skip_dead_main1_enabled=False` on `MtgGameStateAdapter` (or via Spec). `EvalJobSpec` defaults OFF; `SelfPlayJobSpec` defaults ON.

**Acceptable for**: training, self-play data_gen.

**NOT acceptable for**: tournament play, eval matches, scenarios with non-mana sorcery-speed activated abilities that matter pre-combat.

**Fix scope if/when needed**: trivial to disable per-spec. To recover the lost plays, train the policy to value these windows directly. To extend the manland gate to new Effect classes, edit the isinstance tuple in `_should_skip_dead_main1`.

**Impact estimate**: ~2-4% per-game wall reduction during training. Smaller than the others on this branch (only fires on the first 1-2 turns); main value is decision-tree simplification for the early-game policy training signal.

---

## 12. Stop-run sentinel fires at stage boundaries, not mid-game (UI cost-of-control)

**Location**:

- `src/mtg_bot/training/stop_signal.py` (the sentinel-file
  primitives)
- `src/mtg_bot/training/data_gen.py:343` (sentinel poll between
  ``imap_unordered`` results in
  ``generate_training_data_parallel``)
- `src/mtg_bot/training/self_play.py:249` (sentinel poll in
  ``run_match_parallel``'s eval pool)
- `scripts/iterate_alphazero.py` cycle loop (sentinel poll between
  self-play / train / eval stages)
- `scripts/serve_setup_scripts.py:queue_take` (sentinel poll on
  every ``/queue/take`` from a remote)

**What we do today**: when the user clicks the Stop-run button in
the training UI, the back-end writes ``training_progress/.STOP_<id>``.
Producers check for that sentinel at *boundaries* — between
completed-game results in a multiprocessing pool, between
``/queue/take`` requests from a remote, between iter stages
(self-play → train → eval). They do NOT check it inside an
in-flight game's per-decision loop.

**Why we ship this v1 anyway**:
- Mid-game responsiveness would require a per-decision sentinel
  poll inside the search hot-path (PIMC / ISMCTS evaluate
  candidate moves several thousand times per game) and would
  measurably slow self-play. The intended use of Stop is
  abort-and-retry, not pause-mid-decision; a few-minute wait at
  production wall budgets is acceptable.
- In-flight game data is preserved when stops fire at imap-yield
  boundaries (the ``indexed`` list is built incrementally; the
  rows already in it land in the .npz). Mid-game termination
  would lose that game's samples to no purpose.

**Failure mode**: user clicks Stop. The UI ack's immediately and
shows "stop pending…". The producer (multiprocessing pool of
self-play games) finishes the in-flight games on its workers
(could be up to ~2 min at production decision budgets) before its
``imap_unordered`` loop reads the sentinel and breaks. The user
sees "running" for ~minutes after clicking.

**Fix scope when prioritized**:
- *Per-decision sentinel poll* — small code change (add the check
  inside ``MtgGameStateAdapter.legal_decisions`` or the search
  driver's per-iteration loop) but adds an mtime-aware filesystem
  poll to the inner search loop. Mitigatable with caching (only
  re-stat every N decisions / every M ms).
- *HTTP control endpoint* — replace the file sentinel with an
  in-process control channel (queue between UI and producer); is
  more invasive (requires the producer to expose a port, requires
  remotes to call it through Tailscale).

**Impact estimate**: pure cost-of-UX, no strength implications.
For now: stop-click latency = remaining wall time of the longest
in-flight game (= ~30s-2min depending on decision budget).
Acceptable for v1.

---

## 13. Shard script exits via ``os._exit`` (process-cleanup workaround)

**Location**: `scripts/run_self_play_shard.py` — `main()`'s
`finally` block calls `os._exit(exit_code)` after stdio flush.

**What we do today**: the shard script's `main()` is wrapped in
`try / except / finally` such that the LAST thing it does is
`os._exit(exit_code)`. This bypasses Python's normal interpreter
shutdown — no `atexit` handlers, no non-daemon thread joins, no
final GC.

**Why it's the lesser evil**:
`run_shard_pull` / `run_shard_eval_pull` use
`concurrent.futures.ProcessPoolExecutor`. We already replaced the
context-manager exit (`shutdown(wait=True)`) with explicit
`shutdown(wait=False)` (commit `b3ebc39`,
`feature/processpool-shutdown-nowait`) so the executor's cleanup
path doesn't block on hung futures. But Python's at-exit machinery
joins non-daemon threads — and `ProcessPoolExecutor` spawns
internal management threads (`_queue_management_thread` and
friends) that are non-daemon by default and stay alive as long as
any worker process is alive. `shutdown(wait=False)` doesn't kill
those threads; it just doesn't WAIT for them. So `main()` returns,
then Python's normal shutdown blocks forever joining them, while
they're blocked joining still-running C-extension worker code.

By the time we reach the `finally` block, every artifact the
orchestrator depends on is already on disk:
- `.npz` chunks written via `_flush()` inside `run_shard_pull`,
- the per-shard progress JSON via `progress.finalize(...)`,
- the `shard_end` event in `events.jsonl`,
- (eval mode) the per-shard `_result.json` written in
  `run_self_play_shard.py:_run` itself.

The only thing `os._exit` skips is process / thread cleanup — and
the OS reaps the orphan worker processes when the parent dies, so
no resource leaks.

**Failure mode**: any code path that registers an `atexit` handler
expecting it to run on shard exit will silently NOT run. If you
need teardown work, do it explicitly before `_run` returns, not
via `atexit.register`. (Today, nothing in the import graph of
`run_self_play_shard.py` registers a teardown that matters; the
worry is hypothetical / future.)

Concrete repro of the underlying hang (2026-05-05): a Mac shard's
`run_self_play_shard.py` process stayed alive 50+ min after
self-play completed; master's SSH session never saw EOF, so the
orchestrator never proceeded to rsync-back. Required manual
`kill` of the remote Python.

**Fix scope when prioritized** (proper fix that doesn't need
`os._exit`): medium — at least ~50 lines across `run_shard_pull` +
`run_shard_eval_pull`. Either:
- Replace `concurrent.futures.ProcessPoolExecutor` with
  `multiprocessing.Pool` (whose `terminate()` actually kills
  workers — same pattern `bench_per_game_wall.py` already uses),
  or
- Subclass `ProcessPoolExecutor` and override `_queue_management_
  thread` creation to mark it daemon (relies on private API; could
  break across Python versions).

The second is fragile; the first is the recommended direction
when someone has time.

**Impact estimate**: removes the orchestrator hang entirely
(was: per-iteration risk of master SSH session hanging until
manual kill, blocking the entire iteration sweep). No strength
implications, no perf cost on the happy path.

---

## 14. Proliferate primitive proliferates every eligible permanent (no controller choice)

**Location**: `rust/src/state_ops_counters.rs::proliferate` (the
`#[pymethods]` block — search for `fn proliferate`).

**What we do today**: `proliferate()` walks both players'
battlefields and, for every permanent with `counters != 0`, ticks
the counter by 1 in the direction it already has (positive →
+1, negative → −1, zero → unchanged). Saturating arithmetic at the
i32 extremes.

**Why this is wrong under CR 701.25a**: real proliferate is a
*controller choice*. The player resolving the effect picks any
number of permanents and players that already have counters and
chooses, for each, whether to put one more counter of an existing
kind on it. Concretely, three behaviors the primitive does not
expose:

1. *Decline to proliferate at all.* Sometimes the controller has no
   beneficial targets (their own creatures are at zero, opp's
   creatures only have +1/+1 counters they don't want to grow) and
   would prefer the effect resolves as a no-op.
2. *Proliferate only some eligible permanents.* The controller can
   skip their opponent's +1/+1 creatures while proliferating their
   own −1/−1-shrunk opp creatures. Today's primitive does both.
3. *Proliferate poison counters on players.* The Rust engine has no
   poison-counter field at all today, so the player-side half of
   proliferate is a silent no-op by absence — a separate divergence
   tracked here for completeness.

**Failure mode**: in any board state where proliferating an
opponent's +1/+1-counter creature is bad for the player resolving
proliferate (e.g. opp has a Hardened Scales creature with +3/+3
counters; auto-proliferating gives them +4/+4 instead of leaving
them at +3/+3), the primitive picks the worse outcome. Symmetric
case: proliferating own −1/−1-counter creatures (Persist'd creature
that will die at zero toughness next SBA) accelerates the
controller's own loss.

**When it matters**: rarely under the engine's current card pool
(Edge of Eternities focus). Proliferate cards in EoE: none in the
spoiled set as of 2026-05-10. The primitive lands as a building
block for Modern Horizons / War of the Spark-style sets if they
ever enter scope; until a card actually wires into this primitive,
the always-add behavior is unobservable.

**Fix scope when prioritized**: medium — the primitive grows a
`targets: Vec<u32>` parameter (or two — permanents + players) and
the resolution-side wiring in `apply_effect` exposes a per-eligible
target choice through `ChoicePoint`. Search-side: factored
per-target binary choice (proliferate-this-or-not), parallel to how
declare-blockers is factored today (see entry #2). For player
poison counters, the `Player` struct grows a `poison_counters: u8`
field and the primitive's poison-side half lights up.

**Impact estimate**: 0% strength cost today (no card uses
proliferate in EoE). Cards-side cost when proliferate cards enter
scope: ~1-3% strength cost in lopsided board states where the
controller would benefit from skipping the auto-target — bounded
by the rarity of "proliferating my opponent makes me lose".

---

## 15. SearchLibraryForBasicLandTapped: supertypes-proxy filter + xorshift64-only post-search shuffle

**Location**: `rust/src/state_ops_resolve.rs::apply_effect` —
search for the `Effect::SearchLibraryForBasicLandTapped` arm (round-23
effect-coverage slice; G19 closure post-shuffle). Variant declared
in `rust/src/cards.rs`.

**What we do today**: when an
`Effect::SearchLibraryForBasicLandTapped { count }` resolves, the
active player's library is scanned top-to-bottom (Vec iter rev), the
topmost object whose printing's `card_types` contains "Land" is
removed and placed onto the controller's battlefield with
`tapped = true`, `summoning_sick = false`. Repeats up to `count`
times; partial fill is allowed if fewer matches exist. **After all
picks (or the early-`break` on no-more-basic-lands), the library is
Fisher-Yates shuffled per CR 701.19 / 701.22** (G19 closure —
xorshift64-only path; see follow-up gap #16 below for the
cross-engine MT parity deferral).

**Why this diverges from Python / CR**: two remaining shortcuts.

1. *"Basic land" filter is a `card_types` Land check, not a true
   supertypes check.* Python's `is_basic_land` is
   `is_land and "Basic" in supertypes`. The Rust bridge has no
   `supertypes` field today (it ships `card_types` + `keywords` +
   P/T/mana_produced and not much else), so "basic" half of "basic
   land" is approximated by the card_types Land check.
2. *Python's `SearchLibraryForBasicLandTapped.resolve` also does NOT
   call `shuffle_library`*. The docstring claims "then shuffle" but
   the resolve method elides it (Stage-3 simplification documented
   on the class — "picks the FIRST count basic lands in library
   order (no extra shuffle)"). So Rust's G19-closure post-search
   shuffle is technically MORE CR-compliant than Python's path
   today; this introduces a new cross-engine library-order
   divergence (see follow-up gap #16 below).

**Failure mode**:

1. *Supertypes-proxy false positive.* If a card with
   `card_types=["Land"]` but supertypes excluding "Basic" enters the
   library (e.g. a non-basic dual land, fetch land, shock land), the
   Rust dispatch will pick it up; Python's `is_basic_land` will
   reject it. In the current EoE pool no non-basic land coexists
   with `SearchLibraryForBasicLandTapped` on the same engine state
   (the printing carries one card_type only), so the proxy is
   faithful for the present pool — but it WILL diverge once
   non-basic lands enter scope.

**When it matters**:
- Supertypes gap: zero in the EoE pool; non-zero the moment a
  non-basic land joins the bridge.

**Fix scope when prioritized**:
- *Supertypes*: medium — grow `CardPrinting` with a
  `supertypes: Vec<String>` field, extend the parser to require/parse
  it, swap the filter to check `supertypes.iter().any(|s| s ==
  "Basic")`. Touches the bridge schema, parsing, and every test that
  builds a printing dict.

**Impact estimate**:
- Supertypes: zero today (EoE pool only); rises with cards.

---

## ~~16.~~ CLOSED: SearchLibraryForBasicLandTapped consumes the external rng via state.rng (G19 follow-up)

**Closed by** `feature/effect-rng-threading`. The
`Effect::SearchLibraryForBasicLandTapped` resolve arm in
`rust/src/state_ops_resolve.rs::apply_effect` now routes its post-
search Fisher-Yates shuffle through the external rng substrate (CE8
Slice 2 / I41) — when the caller installs a Python `random.Random`
via the new `GameState.set_external_rng(rng)` pymethod, the in-Effect
shuffle reads draws via `external_rng::next_below`, producing byte-
identical permutations to `random.Random.shuffle(list)`. When no rng
is installed (production cli / bench callers), the legacy xorshift64
fallback runs unchanged.

Threading shape: per the
[Effect-rng audit](RUST_EFFECT_RNG_AUDIT.md) §4 "Option A"
recommendation, the rng is stashed on `GameState.rng:
Option<Py<PyAny>>` (set via `set_external_rng`, cleared via
`set_external_rng(None)`) rather than threaded as a parameter
through every intervening call site (`apply_effect` /
`resolve_top_of_stack` / `apply_effect_pub` /
`auto_resolve_pending_decisions`). The Effect arm reads the rng via
`state.rng_ref()` before the in-arm Fisher-Yates loop.

**Pinning tests**:
- `tests_rust/test_resolve.py::test_resolve_search_library_for_basic_land_tapped_external_rng_matches_python_shuffle`
  — post-search library matches `random.Random(seed).shuffle` byte-
  for-byte.
- `tests_rust/test_resolve.py::test_resolve_search_library_for_basic_land_tapped_external_rng_none_falls_back_to_xorshift`
  — backward compat: `set_external_rng(None)` is byte-equivalent to
  never setting it.
- `tests_rust/test_resolve.py::test_resolve_search_library_for_basic_land_tapped_external_rng_differs_from_xorshift`
  — sanity check that the two paths produce distinct permutations.
- `tests_rust/test_resolve.py::test_resolve_search_library_for_basic_land_tapped_external_rng_preserved_across_clone`
  — `GameState.clone()` preserves the stashed rng reference (Python
  refcount bump via the manual `Clone` impl).
- `tests_rust/test_resolve.py::test_resolve_search_library_for_basic_land_tapped_external_rng_clone_isolation`
  — pins the shared-rng-across-clones semantic (single instance,
  consecutive draws).
- `tests/test_parity_smoke_edge_cases.py::test_cross_engine_search_library_for_basic_land_tapped_external_rng_matches_python_mt`
  — cross-engine pin: Rust post-resolve library matches Python's
  MT reference under a shared `random.Random`.
- `tests/test_parity_smoke_edge_cases.py::test_cross_engine_search_library_for_basic_land_tapped_shared_rng_walks_clean_for_multiple_seeds`
  — multi-seed robustness companion (seeds 1, 7, 42, 100).

**Closure invariant**: see I43 in `docs/RUST_PARITY_GAPS.md` for the
`GameState.rng` field lifecycle (caller-set via `set_external_rng`,
consumed by `Effect::SearchLibraryForBasicLandTapped`, not serialized
into fingerprint, clone-preserved via the manual `Clone` impl).

**Python-side follow-up still open.** Python's
`SearchLibraryForBasicLandTapped.resolve` does NOT shuffle today
(Stage-3 simplification documented in #15 above). Closing #15 will
restore full cross-engine library parity end-to-end when the Python
side calls `shuffle_library(state, controller_id)` at the end of
the arm under the same shared rng.

---

## Rust port — `legal_decisions` DeclareAttackers powerset emission

**Location**: `rust/src/state_ops_legal.rs:legal_decisions` — the
DeclareAttackers emission block (search for "DeclareAttackers
extension" in the module docstring + the gated block under
`if self.phase == PHASE_DECLARE_ATTACKERS`).

**What we do today**: during the `DECLARE_ATTACKERS` phase, the Rust
`legal_decisions` enumerator emits one Python dict per subset of
eligible attackers (the powerset). With N eligible attackers and
N ≤ 4, all 2^N subsets emit. With N > 4, only the empty + full
subsets emit (a hard cap, `DECLARE_ATTACKERS_POWERSET_CAP = 4`).

**Why this diverges from Python**: Python's `legal_actions` (in
`src/mtg_bot/legal.py`) does NOT emit DeclareAttackers at all — the
engine.py docstring (lines 10-19) is explicit that combat-declaration
actions "aren't enumerable as legal moves because their decision
space is combinatorial," and Python uses a factored search-side
approach in `src/mtg_bot/search/combat_decisions.py`
(`AttackerDecision` — one per eligible attacker, 2 branches each)
that feeds MCTS at search time. The adapter accumulates per-attacker
decisions into a single engine-level
`DeclareAttackers(frozenset[int])` once all per-creature choices are
collected.

**Failure mode**:
- Wide boards (N > 4 eligible attackers) lose the middle subsets —
  the only emitted choices are "attack with nothing" or "attack with
  everyone." Strategically valuable partial attacks (e.g. "attack
  with the 5/5 and the 2/1 hasty but keep the wall back") aren't in
  the enumeration.
- The parity-smoke harness must skip emission-set comparison during
  the DECLARE_ATTACKERS phase (Python's side emits no
  DeclareAttackers; Rust's side emits up to 2^N).

**When it matters**: any DECLARE_ATTACKERS phase with more than 4
eligible attackers where a partial attack is the right play. Common
late-game scenario; rarer early-game. Pre-trained NN search will
plateau early until the factored shape lands.

**Fix scope when prioritized**:
- Port `search/combat_decisions.py`'s `AttackerDecision` shape to
  Rust as a new `Action::AttackerDecision { attacker_id, attack }`
  variant.
- Replace the powerset emission with one `AttackerDecision` per
  eligible attacker. The adapter (or a search-side accumulator)
  collects all N decisions and dispatches a single
  `DeclareAttackers(frozenset[int])` at apply time.
- The `declare_attacker` primitive stays unchanged.
- The `DECLARE_ATTACKERS_POWERSET_CAP` constant goes away.

**Impact estimate**: ~moderate strength cost on wide-board late-game
scenarios. Negligible during early training where most games end
before either side has 5+ untapped creatures. Will likely be a
visible regression once self-play games push deeper.

---

## Rust port — `legal_decisions` DeclareBlockers product-space emission

**Location**: `rust/src/state_ops_legal.rs:legal_decisions` — the
DeclareBlockers emission block (search for "DeclareBlockers
extension" in the module docstring + the gated block under
`if self.phase == PHASE_DECLARE_BLOCKERS`).

**What we do today**: during the `DECLARE_BLOCKERS` phase (priority
on the defender, stack empty), the Rust `legal_decisions` enumerator
emits one Python dict per legal blocker-to-attacker assignment. With
`n_blockers <= DECLARE_BLOCKERS_PRODUCT_CAP` (3) AND `m_attackers <=
DECLARE_BLOCKERS_MAX_ATTACKERS` (3), all `(M+1)^N` assignments emit
— each blocker independently picks one of M attackers OR "don't
block." With either dimension over its cap, only the canonical "no
blocks" assignment (an empty Vec) emits.

**Why this diverges from Python**: same factored-search story as
DeclareAttackers above — Python's `legal_actions` does NOT emit
DeclareBlockers (engine.py:10-19 is explicit) and instead uses
`BlockerDecision`s (one per eligible blocker, one branch per
attacker plus a "don't block" branch) in
`src/mtg_bot/search/combat_decisions.py`. The adapter accumulates
per-blocker decisions into a single engine-level
`DeclareBlockers(assignments=...)` at apply time.

**Failure mode**:
- Wide boards (N > 3 blockers OR M > 3 attackers) lose all middle
  assignments — the only emitted choice is "no blocks." All
  partial-blocking strategy (chump-blocking the big attacker,
  multi-block to kill a 3/3, leaving small attackers through to save
  the wall) is unreachable from `legal_decisions`.
- The parity-smoke harness must skip emission-set comparison during
  the DECLARE_BLOCKERS phase (Python's side emits no
  DeclareBlockers; Rust's side emits up to `(M+1)^N`).

**When it matters**: any DECLARE_BLOCKERS phase with > 3 blockers
or > 3 attackers. Most early-game scenarios stay under both caps;
mid-to-late game with built-out boards trips the cap regularly.
Pre-trained NN search will plateau on defensive play until the
factored shape lands.

**Fix scope when prioritized**:
- Port `search/combat_decisions.py`'s `BlockerDecision` shape to
  Rust as a new `Action::BlockerDecision { blocker_id, target:
  Option<u32> }` variant.
- Replace the product emission with one `BlockerDecision` per
  eligible blocker (with branches over attackers + None). The
  adapter / search-side accumulator collects all N decisions and
  dispatches a single `DeclareBlockers(assignments=...)` at apply
  time.
- The `declare_blocker` primitive stays unchanged.
- Both `DECLARE_BLOCKERS_PRODUCT_CAP` and
  `DECLARE_BLOCKERS_MAX_ATTACKERS` constants go away.

**Impact estimate**: moderate-to-large strength cost in any state
with non-trivial combat. Combinatorial growth `(M+1)^N` is harder
than DeclareAttackers's `2^N`, so the caps fire sooner and the
information loss is bigger per affected decision.

---

## Rust port — `legal_decisions` block-legality keywords not filtered at enumeration

**Location**: `rust/src/state_ops_legal.rs:legal_decisions`
DeclareBlockers eligibility loop.

**What we do today**: the eligibility filter checks
`!tapped && is_creature`. It does NOT filter by flying/reach
interaction (CR 702.9 — a creature with flying can only be blocked
by a creature with flying or reach), color protection (CR 702.16),
"can't be blocked" (CR 509.1b), skulk (CR 702.118), or
`can_only_block_creatures_with_flying`. All eligible-by-status
blockers appear in every assignment regardless of whether the
attacker-blocker pair is actually legal; the `declare_blocker`
primitive's per-pair `bool` return silently filters illegal pairs at
apply time.

**Why this diverges from Python**: Python's
`src/mtg_bot/combat.py:can_block` runs the full keyword filter at
enumeration time. The Rust enumerator defers the filter to the
primitive instead — bigger emission count + same final state after
apply, mirroring the same trade-off as the DeclareAttackers
defender-deferral above.

**Failure mode**: search wastes visits on assignments containing
illegal pairs (the apply silently drops them, so the post-apply
state matches what an equivalent "illegal-pair-excluded" assignment
would produce). Visit count is inflated but win-rate impact is
proportional to "fraction of emitted assignments containing illegal
pairs" — which can be substantial on a flying-heavy board.

**When it matters**: any DECLARE_BLOCKERS state where the attacker
has flying / "can't be blocked" effects / color protection and the
defender has non-flying / non-reach / wrong-color blockers. EoE has
plenty of flyers; the impact rises with the pool.

**Fix scope when prioritized**: medium. Add `effective_keywords`
bridge to Rust (or a thinner version of `can_block` keyed only on
the flying / cant-be-blocked / skulk keywords first-slice cares
about), then filter at enumeration. The primitive's apply-time gate
stays as defense-in-depth.

**Impact estimate**: at slice landing, low (synthetic seed has no
flyers); rises with EoE pool coverage.

---

## Rust port — `legal_decisions` defender keyword not filtered at enumeration

**Location**: `rust/src/state_ops_legal.rs:legal_decisions`
DeclareAttackers eligibility loop.

**What we do today**: the eligibility filter checks
`!tapped && is_creature && (!summoning_sick || has_haste)`. It does
NOT filter out creatures with the `defender` keyword (CR 702.3).
A defender creature appears in every emitted subset; the
`declare_attacker` primitive's per-id apply-time gate rejects it
silently (returns `false` for that id, the rest of the subset
still gets declared).

**Why this diverges from Python**: Python's
`src/mtg_bot/combat.py:eligible_attackers` (lines 15-23) filters
defender out at enumeration time. The Rust enumerator defers the
filter to the primitive instead — bigger emission count + same
final state after apply.

**Failure mode**: search wastes visits on subsets that include
defender-keyword creatures (the apply silently drops them, so the
post-apply state matches what an equivalent "defender-excluded"
subset would produce). Visit count is inflated but win-rate impact
is minimal.

**When it matters**: rarely. Most untapped creatures aren't
defenders; defender printings are usually walls that stay tapped.
Pre-token-creation training data has zero defender printings in the
EoE pool.

**Fix scope when prioritized**: add a one-line keyword check to the
eligibility loop:
```rust
if printing.keywords.iter().any(|k| k == "defender") {
    continue;
}
```

**Impact estimate**: negligible. Strictly a small redundant-visit
cost; no strength change.

---

## Rust port — `legal_decisions` ActivateAbility approximated to mana producers

**Location**: `rust/src/state_ops_legal.rs:legal_decisions`
ActivateAbility emission loop.

**What we do today**: emit one `ActivateAbility(source_id,
ability_index=0)` per untapped permanent on the priority player's
battlefield whose printing has `mana_produced.is_some()`. No real
ability iteration — `ability_index` is hardcoded to 0.

**Why this diverges from Python**: Python's
`src/mtg_bot/legal.py:612-829` walks
`effective_activated_abilities(state, obj)` per battlefield permanent
and emits one `ActivateAbility(source_id, idx)` per legal-to-pay
ability. The full activated-ability tree (mana abilities, planeswalker
loyalty abilities, equip-style activations, all the cost-gated
variants the Python source filters between lines 631-826) is missing
because Rust's `CardPrinting` (`rust/src/cards.rs:361-396`) has no
`activated_abilities` field — only `mana_produced`, `card_types`,
`power`/`toughness`, `keywords`. Porting the full
`ActivatedAbility` + `ActivationCost` + per-effect dataclass tree
through PyO3 is a much bigger bridge slice.

**Failure mode**: every non-mana activated ability (planeswalker
loyalty, equip / unearth / cycling / channel / activate-X-tap-mana
abilities on non-land sources, etc.) is silently absent from the
emission. Search never visits decision branches that would activate
those abilities. The mana ones DO emit but the dispatch arm at
`state_ops_actions.rs:271-276` is also a stub (raises ValueError) —
so the wire shape ships but the actual ability resolution is the
parity-smoke retry loop's problem.

**When it matters**: the bigger the activated-ability count in a
state, the more search depth Rust skips. EoE has planeswalkers,
loyalty activations, several creature mana producers (Llanowar Elves
in the synthetic seed has `mana_produced=Some(["G"])` and currently
ALWAYS emits — even when summoning-sick — because the creature-tap
sickness filter isn't ported either).

**Fix scope when prioritized**: medium-large. (1) Port `ActivatedAbility`
+ `ActivationCost` + per-effect dataclass tree through PyO3 into
`cards.rs`. (2) Build `effective_activated_abilities` primitive in
Rust (mirrors `src/mtg_bot/game.py`). (3) Replace the
`mana_produced.is_some()` filter in `state_ops_legal.rs` with a real
ability iteration. (4) Land the `activate_ability` primitive in
`state_ops_*.rs` so dispatch flips from "ValueError stub" to a real
state mutation. Each can ship independently.

**Impact estimate**: at the slice's first commit, the enumerator
covers ~1.9% of Round-20 decisions (per the round-23 ActivateAbility
agent's commit note); without the dispatch primitive, the
parity-smoke retry loop catches and skips. Strength impact is
proportional to "activated-ability-bearing decisions in the trace"
× "average gain from playing them optimally" — non-trivial once
planeswalkers + creature mana sources mature.

---

## Rust port — `Effect::ManifestDread` face-down state + count > 1 deferred

**Location**: `rust/src/state_ops_resolve.rs` `Effect::ManifestDread`
resolve arm; `rust/src/state_ops_decisions.rs`
`resolve_manifest_dread_to_battlefield` helper.

**What we do today**: when a ManifestDread effect resolves, the
controller picks one of the top 2 (or 1 if library has 1 card)
examined cards. The picked card is moved to the controller's
battlefield WITH ITS ORIGINAL PRINTING VISIBLE (name, types, P/T,
abilities, color all visible to both players). The other examined
card goes to graveyard. `count > 1` (multi-iter form) and
`count = 0` (no-op) raise at resolve time with a clear "deferred"
message; `count = 1` is the only supported value in the first-slice.

**Why this diverges from real MTG**: CR 701.34 + 707 specifies that
a manifested permanent enters the battlefield FACE DOWN. While
face down, it has:
* No name, no card types other than "Creature", no subtypes
* Base 2/2 power and toughness (CR 707.2c)
* No abilities (except those that function in any zone or affect
  the face-down permanent itself)
* It is colorless
* Its owner can turn it face up at any time by paying the original
  printing's mana cost (CR 707.4) if it is a creature card

The first-slice Rust port skips ALL of these — the manifested card
is functionally a normal permanent of its printing's type. Python's
`src/mtg_bot/effects/generic/library_lookat.py:ManifestDread`
similarly sets `card.face_down = True` but only the face-down flag
is set; the 2/2 vanilla-creature semantics are not threaded through
the engine's effective-keyword / effective-P/T / effective-card-type
readers (which would require a layer-1 characteristic-replacement
implementation across power/toughness/types/abilities). The Rust
port mirrors the Python state today and additionally drops the
`face_down` flag because GameObject doesn't have one.

**Why this is OK as a first-slice**: ManifestDread's primary
strategic value is "pseudo-card-advantage + body on the
battlefield". Modeling the body as a face-down 2/2 vanilla rather
than the original creature's printed P/T loses some accuracy in
combat — a manifested Bashful Beastie (5/5 with stats baked into
its printing) plays as a 5/5 in this engine but should be a 2/2
face-down. But every model trained on this engine sees consistent
ManifestDread behavior — the bias is in the same direction in both
seats and largely washes out for self-play. Real-MTG correctness
matters more when face-up flip costs become tactically relevant
(deferred).

**For `count > 1`**: only a handful of EoE cards reference "manifest
dread N times" (Slimy Aquarium / Weight Room riders) or "manifest
dread X times" (Valgavoth's Onslaught). The first-slice gate makes
those error out loudly rather than silently produce wrong behavior;
the parse layer accepts any non-negative count so the dict round-
trip is forward-compatible with the multi-iter follow-up slice.

**Failure mode (face-down semantics)**:
1. A manifested creature is destroyed by an Auntie's Hovel-style
   "destroy target X creature" that wouldn't legally target a
   face-down 2/2 (face-down has no subtypes). Real MTG: spell
   fizzles. Engine: spell resolves on the visible printing's
   subtype.
2. A face-down creature attacks. Real MTG: combat-damage assigner
   uses the 2/2 P/T. Engine: uses the printing's P/T.
3. A face-down creature is sacrificed for an effect that requires
   "creature with mana value N or less" (face-down has mana value
   0 per CR 707.2d). Real MTG: any face-down satisfies "MV ≤ 0".
   Engine: uses the printed mana value.
4. The "turn face up by paying the mana cost" activated ability is
   missing entirely — the manifest can never flip up in this engine.

**Failure mode (count > 1 gated at resolve)**:
* Slimy Aquarium's "manifest dread, then put a +1/+1 counter on
  that creature" works (count=1, ManifestDreadThenPutPlusOneCounters
  is a separate Python class — and isn't a count > 1 form).
* Anything that genuinely emits ManifestDread(count=2+) will raise
  a ValueError at stack-resolve time. This is loud-fail by design;
  silent partial would be worse.

**When it matters**: face-down state is a moderate-frequency edge
case once the engine sees real games. count > 1 is a low-frequency
edge case (a handful of EoE cards). Both rise in priority once
ManifestDread cards become common in trained decks.

**Fix scope when prioritized (face-down)**: medium. Either (a) add a
`face_down: bool` field to `GameObject` and bake it into
effective-P/T / effective-keyword / effective-types readers and the
fingerprint (requires rebuilding pinned baselines), or (b) introduce
a "face-down placeholder printing" and have ManifestDread swap the
manifested card's printing_id to that placeholder, threading the
original printing through a sidecar `manifested_from_printing_id`
field so face-up flip can restore it. (a) is more invasive but
cleaner; (b) avoids the fingerprint churn but adds a sidecar field
that also needs CR 400.7 zone-change reset semantics.

**Fix scope when prioritized (count > 1)**: small. Iterate the
resolve arm `count` times, each iteration popping + pushing its own
ManifestDreadChoice; the submit dispatcher loops naturally as each
decision drains. Alternatively, carry `remaining_iters: u32` on the
PendingDecision and re-push on submit. Either approach is ~30 LOC.

**Impact estimate**: strength cost is in the low single digits %
for face-down semantics in a mature deck-pool; near zero for
count > 1 in EoE (a few cards). Perf neutral — the face-down
modeling adds zero hot-path work; the count > 1 path only
matters when those specific cards resolve.

---

## Rust port — warp alt-cost cast pipeline absent; `state.spell_warped_this_turn` substrate-only

**Location**: `rust/src/state.rs:758-762` (field declaration),
`rust/src/state_ops_resolve.rs:4986-4998`
(`Effect::ConditionalIfVoid` resolver — sole consumer),
`rust/src/state_ops_actions.rs:245-292` (`Action::CastSpell` arm —
parses `use_warp` flag but ignores it),
`rust/src/state_ops_cast.rs:285-355`
(`GameState::cast_creature` primitive — hand-only, normal mana_cost),
`rust/src/cards.rs:1855-1898` (`CardPrinting` struct — no
`warp_cost` / `warp_castable_from_graveyard` /
`warp_additional_life_cost` fields).

**What we do today**: the `state.spell_warped_this_turn` field
exists on Rust's `GameState`, is reset on entry to PHASE_CLEANUP
(state_ops_steps.rs:294), and is read by the
`Effect::ConditionalIfVoid` resolver via `is_void_active`
(LTB-half OR warp-half). But **NO production code path in Rust sets
the flag**. The only writer is the test-only `set_void_flags`
pymethod (state.rs:971-979). The `Action::CastSpell` dispatch arm
parses the `use_warp` payload field and round-trips it through the
bridge, then explicitly ignores it (state_ops_actions.rs:247
`use_warp: _`) and routes to `cast_creature` — which only consumes
cards from hand and pays the printing's normal `mana_cost`.

**Why this diverges from Python**: Python's
`src/mtg_bot/actions.py:1426-1436,1457` flips
`state.spell_warped_this_turn = True` at two cast-from-zone sites
(warp-from-hand and warp-from-graveyard for Timeline Culler), each
paying the printing's `warp_cost` instead of `mana_cost` and an
optional `warp_additional_life_cost`. Three Python consumers read
the resulting `is_void_active` predicate:
`mtg_bot.effects.state_predicates` (line 105-107),
`mtg_bot.effects.conditions` (315-317), and
`mtg_bot.effects.combinators` (`ConditionalIfVoid` resolver at
1450-1452, plus `is_void_active` lookup at 183-189). Rust currently
ports only the `ConditionalIfVoid` consumer; the other two
combinators / predicates routes have not been ported. So the
bounded impact of the missing warp-half setter is exactly:
`Effect::ConditionalIfVoid`-bearing spells resolved on a turn when
the controller has cast a warp spell take the `else_effect` /
no-op branch instead of the `then_effect` branch.

**Pool exposure**: 50 EoE cards mention "warp" in their oracle text
(measured 2026-05-12 against `data/edge_of_eternities.json`). Per
the round-29 ConditionalIfVoid slice (commit `de4a578c`), there are
4 EoE "Void —" rider cards whose `then_effect` is gated on
`is_void_active` (Decode Transmissions, Hymn of the Faller, and two
others). On a turn where one player warp-casts a spell and a
ConditionalIfVoid rider resolves AFTER that cast, Rust takes the
wrong branch.

**Failure mode**: silent — `Effect::ConditionalIfVoid` resolves
with `void=false` when only a warp-cast event has occurred this
turn (LTB events on the same turn still flip the void predicate
true via I29's LTB half, masking the divergence when both happen).
The closure is "implement warp cast in Rust → the flag setter
follows trivially at the warp-cast site (mirror Python actions.py:
1427,1457)". Until the cast pipeline routes warp distinctly, the
flag has no production write site to wire up.

**Cross-engine smoke surfacing**: the divergence cannot be surfaced
by the existing `tests/test_parity_smoke.py` cross-engine harness
without a warp-cost card in the deckspec, AND a ConditionalIfVoid
rider that resolves AFTER the warp cast in the same turn. Today's
parity-smoke deckspec has neither — surfacing this would require
seeding warp-castable + Void-rider printings into the synthetic
fixture. Deferred until the warp-cost CardPrinting fields land in
Rust (a much larger slice — see "Fix scope" below).

**Why it's OK for now**: warp casting is a Rust-side cast-pipeline
gap, not a flag-substrate gap. The closure is the whole alt-cost
infrastructure (cost-from-zone primitive + `Effect::Warp` + four
new `CardPrinting` fields for `warp_cost` /
`warp_castable_from_graveyard` / `warp_additional_life_cost` /
`cast_via_warp` end-of-turn-exile rider), not a one-line setter.
Adding a setter today would be dead code with no production write
path to hook it into. The flag substrate lives in Rust because
I29's LTB half consumes it; keeping both halves of the
`is_void_active` predicate on `GameState` matches Python's struct
shape so the eventual warp-cast slice slots a one-line write in
exactly where Python's `actions.py:1427` lives — mechanical
mirroring.

**Fix scope when prioritized**: large (full warp alt-cost cast
pipeline).
1. Add `warp_cost: Option<ManaCost>`,
   `warp_castable_from_graveyard: bool`,
   `warp_additional_life_cost: u8`, and `cast_via_warp: bool`
   (per-GameObject) fields to `CardPrinting` / `GameObject` +
   bridge dict parsing in `cards.rs`.
2. Carve the warp-cast cost-payment + zone-removal from
   `state_ops_cast.rs::cast_creature` into a separate
   `cast_via_warp(obj_id, player_id, zone_source)` primitive that
   reads `warp_cost` instead of `mana_cost`, pays the life rider
   (if any), removes from hand OR graveyard depending on
   `warp_castable_from_graveyard`, and **flips
   `state.spell_warped_this_turn = true` at that site** (the
   one-line mirror of `actions.py:1427`).
3. Flip `Action::CastSpell`'s `use_warp` arm in
   `state_ops_actions.rs:245-292` from "ignore the flag" to "route
   to `cast_via_warp` when `use_warp = true`".
4. Add an end-of-turn exile rider for `cast_via_warp = true`
   permanents (mirrors Python's warp-cast leaves-play behavior).
5. Update I29 in `RUST_PARITY_GAPS.md` to add the warp-half setter
   site to the void-flag-lifecycle setter list (currently I29 only
   documents the LTB half's setters; the warp half is noted as
   "still substrate-only" at line 1382-1385 with this doc-only
   audit as the deferral pin).
6. Regression test in `tests_rust/test_void.py`:
   `test_warp_cast_flips_spell_warped_flag` and
   `test_conditional_if_void_resolves_then_branch_after_warp_cast`,
   mirroring the I29 LTB-half tests' shape.

Each step ships independently; (1)+(2)+(3) are the minimum closure
for the flag setter to have a production write site. (4) is
orthogonal to this divergence (it's a separate spell-behavior gap).

**Impact estimate**: bounded to ConditionalIfVoid + warp + same-
turn ordering. Zero impact on any state where (a) no warp spell is
cast that turn, OR (b) no ConditionalIfVoid rider resolves AFTER
the warp cast on that turn, OR (c) a nonland LTB on the same turn
already flipped the void predicate true (the LTB half masks the
warp-half divergence — common in mid-game). The strength cost is
proportional to "warp-cast turns with a ConditionalIfVoid rider
resolving after but no concurrent LTB" — a tiny slice of the
gameplay surface. Perf neutral.

---

## N. Rust DealsCombatDamage trigger dispatch skips player-target damage

**Location**: `rust/src/state_ops_combat.rs::apply_damage_pass` (around the
`creature_target_id` match) + `rust/src/state_ops_resolve.rs::
dispatch_event_multi_subject`.

**What we do today**: when a damage entry is created during combat
damage resolution, the engine dispatches `Event::DealsCombatDamage`
ONLY for entries whose target is a `DamageTarget::Creature(id)`.
Entries with `DamageTarget::Player(player_id)` skip the dispatch
entirely — the player takes life loss + lifelink fires, but no
trigger fan-out happens.

**Why this is wrong under current CR**: CR 119.5 / triggered-ability
fan-out: a creature dealing combat damage to a player is a valid
"deals combat damage" event. Triggers like Illvoi Infiltrator
(`IsSelf(subject="source") & IsPlayer(subject="target")`) should
fire on player-target damage but don't in Rust today.

**Failure mode**: cards with "whenever ~ deals combat damage to a
player" triggered abilities silently never fire when their dealer
hits the defender unblocked. The substrate (`Predicate::Subject(slot,
inner)`) supports the necessary slot lookup; the gap is purely the
dispatch wiring in `apply_damage_pass`.

**When it matters**: any EoE card with a player-target combat
damage trigger. Illvoi Infiltrator, Starwinder-family
(`IsCreature(subject="source") & Controlled(YOU, subject="source") &
IsPlayer(subject="target")`), Haliya Ascendant Cadet aggregated
"creatures with +1/+1 counters deal combat damage to a player" — all
present in Python via `combat.py:729-742` but skipped by Rust.

**Fix scope when prioritized**: introduce a tagged-id design for
the Target slot so the predicate evaluator can distinguish "subject
is a GameObject" (look up printing → card types) from "subject is a
Player" (look up by player.id → never matches IsCreature etc.,
matches a future `IsPlayer` predicate). One approach: switch the
slot map to `(SubjectSlot, SubjectId)` where `SubjectId` is an enum
`Object(u32) | Player(u8)`. Touches the `Predicate::matches_with_slots`
signature and every predicate-arm body. Estimated <200 LOC change
plus tests.

**Impact estimate**: enables a handful of EoE cards' DCD-to-player
triggers in Rust. No strength impact for cards already covered by
the creature-target path.

---

## Tracked but not in this doc

Things tracked elsewhere that you might also want to know about:

- **PIMC's visit-aggregation bug** (workaround in writer-side filter,
  permanent fix is ISMCTS migration which is mostly done)
- **CoW `_fork_if_shared` historical issues** (fixed by the cleanup-
  step fork added in `feature/cast-not-castable-rootcause`)
- **Engine "hangs" turned out to be slow-tail games** (see
  `docs/hang_triage_2026_05_03.md`)
- **EventPattern.subject_filters originally not hashable** (fixed by
  `feature/event-pattern-hashable`)
- **belief-weighted determinizer needs trained policy to be useful**
  (shelved until model is trained enough — see worklog 2026-05-04)
- **Coverage gate at 95.10%** with main typically running 95.15-95.20%

---

## Adding a new entry

When you take a deliberate shortcut (heuristic, approximation,
incomplete CR conformance), add a section here with:
- Where in the code (file:line)
- What the shortcut is
- Why it's OK (or which scenarios make it not-OK)
- Failure mode
- Fix scope estimate
- Impact estimate (perf gain or strength cost)
