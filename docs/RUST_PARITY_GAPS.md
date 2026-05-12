# Rust engine parity gaps and invariants

This document consolidates every parity-load-bearing divergence, parity
invariant, and pinned regression baseline that the Rust port test
suite has discovered through the r25 edge-case wave. It is intended
to be the checklist that the eventual 1000-game cross-engine smoke
harness consumes when it sees a Python-vs-Rust state difference.

## What this is for

When the 1000-game smoke runs Python and Rust engines side-by-side
on the same deck + seed + decision script and observes a divergence,
the operator needs to answer one question quickly: *is this a real
parity bug, or an already-known accepted divergence?*

This file is the first lookup. Each entry below is pinned by a test
file in `tests_rust/` so a regression on either side surfaces as a
test failure on that file, not as a mystery smoke divergence.

The companion file `docs/SEARCH_LIMITATIONS.md` documents
search-side / `legal_decisions` shortcuts; this file is broader and
covers engine-state divergences across all primitives.

How the smoke consumes this doc:

1. Smoke run records the first decision index where Python and Rust
   diverge, plus the two fingerprints.
2. Operator opens this doc and scans the "Confirmed gaps" table for
   a description matching the observable symptom (e.g. "mana pool
   non-empty when it shouldn't be", "creature still on the battlefield
   after lethal damage").
3. If found, the entry's "Likely impact" cell tells them whether to
   filter the divergence out of the smoke summary or to escalate it.
4. If not found, it's a NEW divergence and gets escalated to a
   dedicated parity-debug session.

---

## Confirmed gaps

Each entry: short name, pinning test, current Rust behavior, expected
behavior per CR / Python, parity impact, and closeable / accepted.

### G5. `legal_decisions(non_priority_player)` returns the `[pass_priority]` singleton

- **Pinned by**: `tests_rust/test_priority.py::test_legal_decisions_for_non_priority_player_still_emits_pass_priority`;
  also `tests_rust/test_legal.py::test_legal_decisions_priority_off_emits_only_pass_priority`.
- **Rust today**: Asking `legal_decisions(p)` where `p` is NOT the
  current priority player still returns a `[pass_priority]` list, not
  an empty list.
- **CR / Python expected**: This is intended parity with Python's
  `legal.py:180-181` short-circuit — Python returns the same
  singleton.
- **Impact**: none — this is a parity INVARIANT documented here for
  context because the file-name "gap" would be misleading. Listed
  here so a future Rust refactor that wants to change "off-priority
  → no actions" notices this is intentional Python parity.
- **Closeable**: not a gap — it is the deliberate parity contract.
  See invariant I5 for the matching invariant entry.

### G8. `+1/+1` vs `-1/-1` counters cancel as net-i32 at write time, not as separate stacks at SBA

- **Pinned by**: `tests_rust/test_counters.py::test_plus_one_minus_one_cancellation_quirk_net_i32_collapses_pre_sba`.
- **Rust today**: Counters are a single signed `i32` per object.
  Adding +3 then -2 stores net +1 immediately, BEFORE any SBA pass.
- **CR / Python expected**: CR 704.5r / 704.5p — +1/+1 and -1/-1
  counters are distinct counter kinds and cancel pairwise at SBA
  time. Python stores them as distinct stacks and cancels at
  `state_based_actions`.
- **Impact**: low for destruction outcomes (the post-cancellation
  effective P/T matches), but inexpressible queries like "how many
  +1/+1 counters does this creature have?" diverge. Not directly
  observable through the fingerprint or zone-snapshot surface in
  current scope.
- **Closeable**: requires a multi-kind counter map. Future slice.

### G9. No dedicated counter-snapshot getter; per-object scalar is the snapshot

- **Pinned by**: `tests_rust/test_counters.py::test_counters_surface_through_battlefield_snapshot_round_trip`
  (and its docstring note).
- **Rust today**: Counters surface only through the per-object
  `counters` slot in `battlefield_snapshot` dicts. No `counter_snapshot`
  or similar getter exists.
- **CR / Python expected**: not a CR rule — this is a parity-surface
  gap. Python's snapshot shape has the same single-scalar shape today,
  so no actual divergence yet. Listed here as a forward-looking note:
  when Rust grows a multi-kind counter map (G8), it will need a
  matching snapshot getter or the parity surface will desync.
- **Impact**: zero today, high when G8 is closed.
- **Closeable**: bundled with G8.

### G12. Mutual KO model is two sequential `lose_life` calls, not a single Effect

- **Pinned by**: `tests_rust/test_winloss.py::test_mutual_zero_via_sequential_lose_life_is_draw` (docstring note: "the closest cross-feature analog the Rust engine exposes for 'both die in the same event'").
- **Rust today**: There is no single Effect that drops both players
  to 0 atomically. Mutual-KO outcomes have to be reproduced by two
  back-to-back `lose_life(p, n)` calls followed by `apply_sbas_until_quiescent`.
- **CR / Python expected**: Real MTG sees "both players reduced to 0
  simultaneously" via combat damage or symmetric Effects.
  `is_draw == True` is the parity-load-bearing outcome regardless of
  how the state was constructed.
- **Impact**: low — the observable end-state (`is_draw`, `losers
  == [0, 1]`) matches; only the construction shape diverges.
- **Closeable**: deferred until symmetric-Effect machinery lands.

### G13. `apply_sbas_until_quiescent` is a no-op after game-over

- **Pinned by**: `tests_rust/test_winloss.py::test_apply_sbas_until_quiescent_after_game_over_is_noop`.
- **Rust today**: Once the game is decided (one or both players in
  losers), subsequent `apply_sbas_until_quiescent` calls return 0
  passes and leave outcome state stable.
- **CR / Python expected**: parity invariant — Python also stops
  doing SBA work once the game has ended.
- **Impact**: none — this is the parity-load-bearing post-game-over
  no-op contract. Listed here for clarity. See invariant I7.
- **Closeable**: not a gap. Parity invariant.

### G16. `set_phase` does not fire the untap sweep

- **Pinned by**: `tests_rust/test_steps.py::test_set_phase_does_not_fire_untap_sweep`.
- **Rust today**: `set_phase(PHASE_UNTAP)` is a raw assignment — it
  does NOT trigger `untap_all_for_player` or the summoning-sick
  clear hook. Only `advance_phase` composes those side-effects.
- **CR / Python expected**: not directly a CR gap; this is the
  Rust-internal asymmetry between `set_phase` (test setup primitive)
  and `advance_phase` (game-loop primitive). Python has no `set_phase`
  equivalent — its turn loop only advances.
- **Impact**: zero in real play (the smoke harness never uses
  `set_phase`). Listed here so a future consolidation of the two
  entrypoints updates this test deliberately.
- **Closeable**: not a gap — deliberate asymmetry.

### G17. `register_and_inject_object` rejects negative-amount `AddManaToPool` at parse time

- **Pinned by**: `tests_rust/test_mana.py::test_add_mana_to_pool_effect_rejects_negative_amount_at_register_time`.
- **Rust today**: The bridge `parse_effect` enforces `amount >= 0`
  when reading a printing dict's `add_mana_to_pool` entry; an
  ill-formed printing fails REGISTRATION with `ValueError` rather
  than reaching the stack-resolution path.
- **CR / Python expected**: Parity — Python's printing parser is
  similarly strict; no silent no-op path for negative-amount mana.
- **Impact**: none — parity invariant. Listed because the location
  of the check (parse time, not apply time) is non-obvious for
  cross-engine debugging. See invariant I3.
- **Closeable**: not a gap.

### G18. `ActivateAbility` credits the source's controller, not `active_player`

- **Pinned by**: `tests_rust/test_mana.py::test_activate_ability_credits_source_controller_not_active_player`.
- **Rust today**: The mana-ability arm in `state_ops_actions.rs`
  reads the source's battlefield owner and credits THAT player's
  mana pool, NOT `state.active_player`. Activating a Forest on p1's
  battlefield while p0 is active adds {G} to p1's pool.
- **CR / Python expected**: Parity — Python's `ActivateAbility`
  resolves controller from the source object.
- **Impact**: none — parity invariant. Listed because the choice
  of who-gets-credited is non-obvious. See invariant I9.
- **Closeable**: not a gap.

### G19. SearchLibraryForBasicLandTapped does not shuffle (CR 701.19 / 701.22)

- **Pinned by**: `tests_rust/test_library.py::test_effect_search_library_for_basic_land_tapped_enters_tapped`
  (docstring note); also tracked in `docs/SEARCH_LIMITATIONS.md`.
- **Rust today**: The round-23 first slice does NOT shuffle the
  library after the deterministic first-match-from-top search.
- **CR / Python expected**: CR 701.19 / 701.22 — search-then-shuffle
  effects always shuffle, whether or not a card was found.
- **Impact**: medium. Visible whenever a search effect runs; the
  resulting library order will diverge from Python.
- **Closeable**: yes — pending RNG externalization work to seed the
  shuffle deterministically inside `apply_effect`.

### G24. No scry-on-keep step after mulligan-to-7 (Paris rule)

- **Pinned by**: `tests_rust/test_mulligan.py` module docstring (gap 4).
- **Rust today**: Keeping a mulliganed hand does NOT trigger any
  scry. The Paris rule (one scry after each mulligan you kept on
  7) is not modeled.
- **CR / Python expected**: CR 103.4 Paris mulligan — one scry on
  each mulliganed-then-kept hand. Python's flow models this if
  enabled.
- **Impact**: low — only visible if the parity harness exercises
  the post-mulligan scry decision.
- **Closeable**: yes — pending Paris-rule decision-callback slice.

---

## Parity invariants

Behaviors where Rust matches Python by design. Pinned so a future
refactor on either side that breaks the parity surfaces immediately.

### I1. Fingerprint is order-INSENSITIVE for mana pool

- **Pinned by**: `tests_rust/test_fingerprint.py::test_fingerprint_insensitive_to_mana_pool_add_order`.
- **Invariant**: The fingerprint serializes mana pools by sorted
  color-key (Python: `tuple(sorted({"W":w,...}.items()))`; Rust:
  a fixed-order 6-element vec keyed B/C/G/R/U/W). Adding {W} then
  {B} produces the same digest as adding {B} then {W}.

### I2. Fingerprint hashes printing.NAME but NOT card_types or effects-text

- **Pinned by**: `tests_rust/test_deck.py::test_make_state_fingerprint_differs_when_registry_name_differs`
  (positive case — name change moves the digest); the absence of
  card_types / effects-text in the per-object 8-tuple `(printing_name,
  tapped, summoning_sick, damage, counters, attached_to, controller_id, id)`
  is documented in `test_fingerprint.py` module docstring.
- **Invariant**: Two states identical except for the registry's
  `card_types` list or effect-text on the SAME printing_id will
  fingerprint identically. Two states where printing.NAME differs
  produce different fingerprints.

### I3. Negative-amount `AddManaToPool` rejected at `parse_effect`, not at apply time

- **Pinned by**: `tests_rust/test_mana.py::test_add_mana_to_pool_effect_rejects_negative_amount_at_register_time`,
  `test_add_mana_negative_amount_raises`.
- **Invariant**: There is no silent-no-op path for negative-amount
  mana addition anywhere in the engine. Both `add_mana(p, c, -1)`
  and registering a printing with `{"add_mana_to_pool", amount: -1}`
  raise `ValueError`.

### I4. `legal_decisions(priority_holder)` emits `pass_priority` at index 0

- **Pinned by**: `tests_rust/test_priority.py::test_legal_decisions_for_priority_holder_emits_pass_priority`.
- **Invariant**: Whoever currently holds priority always has at least
  one legal decision, and that decision is always `pass_priority` at
  index 0 (mirrors Python `legal.py:178`).

### I5. `legal_decisions(non_priority)` returns the `[pass_priority]` singleton

- **Pinned by**: `tests_rust/test_priority.py::test_legal_decisions_for_non_priority_player_still_emits_pass_priority`,
  `tests_rust/test_legal.py::test_legal_decisions_priority_off_emits_only_pass_priority`.
- **Invariant**: Asking `legal_decisions(p)` where `p` does NOT hold
  priority returns `[pass_priority]`, NOT an empty list. Mirrors
  Python's `legal.py:180-181` short-circuit.

### I6. Fingerprint preserves CoW isolation after mutation

- **Pinned by**: `tests_rust/test_fingerprint.py::test_fingerprint_clone_isolation_after_mutation`,
  `test_fingerprint_clone_isolation_zone_mutation`,
  `test_fingerprint_clone_replay_equality_after_each_mutation`.
- **Invariant**: `state.clone_once().fingerprint() ==
  state.fingerprint()` at every step of an in-place mutation
  sequence; mutating the source does not move the clone's digest.

### I7. `apply_sbas_until_quiescent` after game-over is a no-op

- **Pinned by**: `tests_rust/test_winloss.py::test_apply_sbas_until_quiescent_after_game_over_is_noop`.
- **Invariant**: Once the game is decided, repeated SBA-loop calls
  return 0 passes and the outcome reads are stable.

### I8. Active and priority player are independent

- **Pinned by**: `tests_rust/test_priority.py::test_pass_priority_does_not_affect_active_player`.
- **Invariant**: `pass_priority` moves only `priority_player`;
  `active_player` only changes on `advance_phase`'s CLEANUP -> UNTAP
  wrap.

### I9. `ActivateAbility` credits source.controller (not active_player)

- **Pinned by**: `tests_rust/test_mana.py::test_activate_ability_credits_source_controller_not_active_player`.
- **Invariant**: Mana from a tapped Forest is always credited to the
  Forest's controller, not the current active player.

### I10. Tap and summoning_sick are independent fields

- **Pinned by**: `tests_rust/test_tap.py::test_untap_all_for_player_leaves_summoning_sick_alone`.
- **Invariant**: `untap_all_for_player` flips only `tapped`;
  `summoning_sick` is owned by a separate sweep in
  `state_ops_steps.rs`.

### I11. Fingerprint sensitivity for each top-level GameState field

- **Pinned by**: `tests_rust/test_fingerprint.py::test_fingerprint_sensitive_to_priority_player`,
  `test_fingerprint_sensitive_to_phase`,
  `test_fingerprint_sensitive_to_active_player_and_turn_number`.
- **Invariant**: Mutating any of `priority_player`, `phase`,
  `active_player`, `turn_number` shifts the digest.

### I12. Fingerprint sensitivity for each GameObject field

- **Pinned by**: `tests_rust/test_fingerprint.py::test_fingerprint_sensitive_to_tapped_flag`,
  `test_fingerprint_sensitive_to_damage_stamp`,
  `test_fingerprint_sensitive_to_counters_value`.
- **Invariant**: Mutating any of `tapped`, `damage`, `counters` on a
  battlefield permanent shifts the digest.

### I13. Fingerprint is order-SENSITIVE for library and stack

- **Pinned by**: `tests_rust/test_fingerprint.py::test_fingerprint_sensitive_to_library_order`,
  `test_fingerprint_sensitive_to_stack_order`.
- **Invariant**: Two states with identical library/stack CONTENTS in
  different orders produce different fingerprints. Python's
  `hash_state` walks the same zones in iteration order, so swaps are
  observable on both sides.

### I14. Empty-library mill / search returns 0 / None without raising

- **Pinned by**: `tests_rust/test_library.py::test_mill_target_player_empty_library_returns_zero`,
  `test_search_and_shuffle_empty_library_returns_none`,
  `test_search_library_by_type_empty_library_returns_none`.
- **Invariant**: Mill / search against an empty library is a quiet
  no-op (returns 0 or None) — the loss path only fires on a *draw*
  attempt against an empty library (and even that is currently G10).

### I15. `set_phase` does not trigger the untap sweep

- **Pinned by**: `tests_rust/test_steps.py::test_set_phase_does_not_fire_untap_sweep`.
- **Invariant**: `set_phase` is a raw assignment; `advance_phase` is
  the composition entrypoint. Only the latter fires untap +
  summoning-sick-clear hooks.

### I16. `find_object_zone` after `make_state` agrees with the deck-load zone-stamp

- **Pinned by**: `tests_rust/test_deck.py::test_find_object_zone_after_make_state_agrees_with_setup`.
- **Invariant**: Every library object reads `ZONE_LIBRARY`; every
  hand object reads `ZONE_HAND` immediately after `make_state`.

### I17. Summoning-sick **creatures** with tap-cost activated abilities are excluded from both dispatch and emission

- **Pinned by**: `tests_rust/test_tap.py::test_activate_ability_on_summoning_sick_creature_with_tap_cost_is_rejected`
  (creature rejected),
  `tests_rust/test_tap.py::test_activate_ability_on_summoning_sick_land_with_tap_cost_is_allowed`
  (land allowed — creature-only gating);
  `tests_rust/test_legal.py::test_legal_decisions_activate_ability_excludes_summoning_sick_creature`
  (creature not emitted),
  `tests_rust/test_legal.py::test_legal_decisions_activate_ability_emits_for_summoning_sick_land`
  (land emitted).
- **Invariant**: CR 302.1 — a **creature** with `summoning_sick == true`
  whose activated ability has a tap cost (e.g. `{T}`) cannot pay
  that cost. **Lands (and other non-creature permanents) are
  exempt** — they can tap for mana the turn they enter the
  battlefield even though every freshly-allocated permanent carries
  `summoning_sick = true` (the CE1 universal-true default). The
  Rust port enforces the gate on BOTH sides:
  `apply_action`'s `ActivateAbility` arm in `state_ops_actions.rs`
  raises `ValueError` (no mutation) and the `legal_decisions`
  emitter in `state_ops_legal.rs` skips emission — but ONLY when
  the source is a creature (`printing.card_types.contains("Creature")`).
  The general gate is
  `is_creature && summoning_sick && requires_tap_cost`; the
  first-slice mana-only filter (the only wired ActivateAbility
  dispatch case today, where every printing's mana ability has an
  implicit `{T}`) gates on `is_creature && summoning_sick`.
  Mirrors Python's `legal.py:661-663`
  (`is_creature_now(obj) && obj.summoning_sick`) and
  `actions.py:_activate_ability` precheck. Closes parity gap G6
  (entry removed from "Confirmed gaps" above); the cast-bears
  probe's sick-land filter in `cross_engine.py::cast_bears_selector`
  is now defensive-only and removable in a future cleanup.

### I18. `advance_phase` auto-fires `draw_card` on entry to PHASE_DRAW, gated by CR 103.7c

- **Pinned by**:
  `tests_rust/test_steps.py::test_advance_phase_draws_a_card_on_entering_draw_step`,
  `tests_rust/test_steps.py::test_advance_phase_first_turn_active_player_skips_draw_step`,
  `tests_rust/test_steps.py::test_advance_phase_p1_first_draw_step_does_draw_card`.
- **Invariant**: CR 504.1 — the active player draws one card on
  entering their draw step. `advance_phase` calls `draw_card` for
  the active player whenever the new phase becomes `PHASE_DRAW`,
  gated by the CR 103.7c first-turn skip (`turn_number != 1`).
  Mirrors Python's `simulate.py` gate, which uses the same
  `state.turn_number != 1` shape (rationale: on turn 1 the active
  player IS the starting player, so `turn_number` alone fully
  determines the skip in a two-player game). Closes parity gap G14
  (entry removed from "Confirmed gaps" above).

### I19. Empty-library draw flags the player; `apply_sbas` locks the loss into the sticky `has_lost`

- **Pinned by**:
  `tests_rust/test_library.py::test_drawcards_from_empty_library_marks_player_for_loss_on_next_sba`,
  `test_drawcards_count_larger_than_library_drains_then_loses_on_sba`,
  `tests_rust/test_zones.py::test_draw_card_from_empty_library_returns_none`
  (no-crash + cause-signal side-effect),
  `tests_rust/test_winloss.py::test_explicit_lose_game_marks_player`,
  `test_each_opponent_loses_life_lethal_then_sba_locks_the_loss`.
- **Invariant** (CR 704.5b / CR 104.2 — round-26 G10 + G11 closure):
  - The `draw_card` primitive, the `Effect::DrawCards` resolution
    arm, AND `advance_phase`'s auto-draw step (the
    `draw_card_inline` mirror used on entry to PHASE_DRAW — see
    `feature/draw-card-inline-flag`) all flip
    `Player.attempted_draw_on_empty = true` when the library pop
    returns `None`. All three entrypoints share the loss-flag
    side-effect so a player who decks out via ANY draw path
    (cast spell that resolves DrawCards, side-channel pymethod
    call, or auto-draw at start of their turn) gets flagged.
  - `losers()` returns a player if **any** of the OR'd-three signals
    are true: `has_lost`, `attempted_draw_on_empty`, or `life <= 0`.
    Read directly (no SBA pass required) so the loss is observable
    eagerly — mirrors Python's `p.has_lost = True` in
    `turn.draw_card`.
  - `apply_sbas` transitions the cause-signals (`attempted_draw_on_empty`
    OR `life <= 0`) into the sticky `Player.has_lost = true`. Once
    `has_lost` is set, no `gain_life` can revive the player. The
    return value of `apply_sbas` remains the count of CREATURES
    destroyed in the pass — the player-loss transition doesn't bump
    it (separate SBA category in CR ordering).
  - The pure-derivation revive semantics on `life <= 0` are preserved
    UNTIL `apply_sbas` runs (so `set_life(0, 0); gain_life(0, 5)`
    without an SBA pass in between still flips the player back out of
    `losers()` — see `test_gain_life_from_zero_revives_player`).
  - The explicit `lose_game(p)` primitive (Phage / Bargain) sets
    `has_lost` directly without going through SBA — useful for
    "you lose the game" cards where life is not the trigger.

### I20. `advance_phase` fully composes phase-boundary side effects (CR 106.4 / 514.1 / 514.2)

- **Pinned by**:
  `tests_rust/test_mana.py::test_mana_pool_empties_on_advance_phase`;
  `tests_rust/test_cleanup.py::test_end_turn_and_start_next_empties_mana_pool`,
  `test_advance_phase_through_cleanup_enforces_hand_size`;
  `tests_rust/test_steps.py::test_advance_phase_empties_both_mana_pools_on_every_tick`,
  `test_advance_phase_empties_mana_pool_across_every_phase_in_lap`,
  `test_advance_phase_into_cleanup_fires_hand_size_discard`,
  `test_advance_phase_into_cleanup_fires_damage_clear`,
  `test_advance_phase_cleanup_hook_targets_old_active_player`,
  `test_advance_phase_cleanup_hook_idempotent_when_hand_at_max`,
  `test_clone_advance_phase_into_cleanup_does_not_affect_original`.
- **Invariant**: Closing the round-26 G1 + G2 gaps. Every
  `advance_phase` tick empties both players' mana pools BEFORE the
  phase counter ticks forward (CR 106.4 — mana pool empties on
  phase exit). Whenever the new phase is `PHASE_CLEANUP`,
  `advance_phase` fires the inline mirror of
  `cleanup_step_for_active_player` on entry: CR 514.1 hand-size
  discard for the active player (max=7, deterministic-end-of-Vec
  pop) followed by CR 514.2 marked-damage clear on both
  battlefields. Combined with the round-2 untap sweep, the
  round-14 summoning-sick clear, and the round-26 G14 auto-draw,
  `advance_phase` now models every phase-entry / phase-exit hook
  the Rust engine carries. The `cleanup_step_for_active_player`
  primitive itself remains layered below CR 106.4 (calling it
  directly does NOT empty mana pools — see
  `test_cleanup.py::test_cleanup_step_does_not_empty_mana_pool_at_primitive_layer`);
  the mana-empty hook is owned exclusively by `advance_phase`.

### I24. `mulligan_atomic` folds shuffle + redraw-7 + bottom-N into one transaction (CR 103.4 London)

- **Pinned by**:
  `tests_rust/test_mulligan.py::test_mulligan_atomic_seventh_call_succeeds_eighth_raises`,
  `test_mulligan_atomic_bottoms_correct_number_for_each_mulligan_count`,
  `test_mulligan_atomic_bottom_count_mismatch_raises`,
  `test_mulligan_atomic_invalid_card_id_raises`,
  `test_mulligan_atomic_hand_size_invariant`,
  `test_mulligan_atomic_library_grows_by_old_hand_minus_bottom`,
  `test_mulligan_atomic_clone_isolation`,
  `test_mulligan_atomic_fingerprint_stable_for_same_seed_and_bottoms`.
- **Invariant** (CR 103.4 / round-27 G21 + G22 closure):
  - `mulligan_atomic(player_id, seed, bottom_card_ids)` is the
    parity-correct entrypoint for the harness. The Nth mulligan
    (per-player `mulligan_count` is N-1 at call time, becomes N
    after the increment) bottoms exactly N cards — the
    `bottom_card_ids.len() == mulligan_count + 1` pre-check fires
    BEFORE the increment / shuffle / bottoming so a wrong-count call
    leaves state byte-identical.
  - Ordering is (1) player_id validation, (2) bottom-count pre-check,
    (3) increment `mulligan_count`, (4) cap check (> 7 raises),
    (5) shuffle library (drain hand + Fisher-Yates with `seed` +
    redraw 7), (6) verify each `bottom_card_id` is in the post-redraw
    hand AND that the list contains no duplicates, (7) remove each
    chosen id from hand and insert at `lib[0]` in REVERSE iteration
    order so the FIRST id supplied lands at the very bottom (matches
    `bottom_cards_from_hand`'s orientation contract).
  - Hand-size invariant: post-call hand size is always
    `7 - mulligan_count` (drew 7, bottomed `mulligan_count`). A caller
    using the atomic primitive can NEVER observe a state with the
    hand still at 7 after the Nth mulligan — closes the G21 / G22
    observable gap.
  - Bottom-count mismatch (`bottom_card_ids.len() !=
    mulligan_count + 1`) raises BEFORE any mutation; the state's
    fingerprint, hand size, library size, and `mulligan_count` are
    all unchanged.
  - Cap shared with the legacy `mulligan` / `mulligan_to_n`: the
    7-call total is per-player and counts atomic calls too. The 8th
    call raises with no zone mutation (the cap check fires before
    the shuffle), though `mulligan_count` itself was bumped (matches
    legacy ordering).
  - Invalid `bottom_card_id` (not in post-redraw hand) OR duplicate
    ids raise AFTER the shuffle has happened — the bottom step is
    skipped but the shuffle has already mutated state. Callers should
    pre-compute valid ids by probing a clone with the same `seed`.
  - The legacy `mulligan`, `mulligan_to_n`, and `bottom_cards_from_hand`
    primitives are KEPT for backwards compatibility; only
    `mulligan_atomic` is the parity-correct entrypoint for new
    callers. Closes parity gaps G21 + G22 (entries removed from
    "Confirmed gaps" above).

### I25. Priority composition layer: `pass_priority` increments a counter, non-pass mutations reset it, `complete_step_via_priority_round` routes between CR 117.4 / 117.5 legs, and `advance_phase` resets priority on the CLEANUP -> UNTAP wrap (CR 117.1a / 117.1c / 117.4)

- **Pinned by** (round-27 G3 + G4 + G15 closure):
  - **Counter increment + reset**: `tests_rust/test_priority.py::test_consecutive_passes_increments_on_each_pass`,
    `test_consecutive_passes_resets_on_non_pass_action`,
    `test_complete_priority_round_resets_consecutive_passes`,
    `test_pass_priority_alone_does_not_advance_step_or_resolve`.
  - **G3 composer (`complete_step_via_priority_round`)**:
    `tests_rust/test_priority.py::test_complete_step_via_priority_round_advances_step_when_stack_empty`,
    `test_complete_step_via_priority_round_resolves_top_when_stack_nonempty`,
    `test_complete_step_via_priority_round_with_multi_item_stack_pops_only_top`,
    `test_complete_step_via_priority_round_resets_priority_to_active_after_resolution`;
    cross-feature: `tests_rust/test_stack.py::test_complete_step_via_priority_round_drains_then_advances`.
  - **G4 (cast / activate reset priority to active)**:
    `tests_rust/test_priority.py::test_cast_spell_resets_priority_to_active`,
    `test_cast_spell_failure_does_not_touch_priority`,
    `test_activate_mana_ability_resets_priority_to_active`.
  - **G15 (advance_phase priority reset on wrap)**:
    `tests_rust/test_priority.py::test_advance_phase_full_lap_resets_priority_to_active_at_wrap`,
    `test_priority_unaffected_by_advance_phase_within_a_turn`;
    `tests_rust/test_steps.py::test_advance_phase_keeps_priority_player_in_sync_with_active`.
- **Invariant** (CR 117.1a / CR 117.1c / CR 117.4 / CR 117.5 / round-27 G3 + G4 + G15 closure):
  - `GameState.consecutive_passes: u8` tracks consecutive `pass_priority`
    calls since the last non-pass mutation. `pass_priority` increments
    it (saturating); every non-pass mutation
    (`cast_creature`, `apply_action` for non-PassPriority arms,
    `advance_phase`, `give_priority_to`, the round-9
    `complete_priority_round` composer, and the new
    `complete_step_via_priority_round` composer's resolution leg)
    resets it to 0.
  - `complete_step_via_priority_round() -> bool` is the harness-facing
    "both players have passed" composer. Internally calls
    `pass_priority` twice, then routes by stack state:
    - **Non-empty stack** → CR 117.4 / 117.5: pop the top, reset
      `priority_player = active_player` (CR 117.1c), reset
      `consecutive_passes`, return `False` (no step advance).
    - **Empty stack** → CR 117.4: call `advance_phase` (which handles
      mana-empty, the per-phase hooks, the CLEANUP -> UNTAP wrap with
      the CR 117.1a priority reset, and the consecutive_passes reset),
      return `True` (step advanced).
    Mirrors Python's `simulate.py:priority_loop` logic (lines
    245-280): when `pass_count >= alive_count()` with `state.stack`
    non-empty, call `resolve_top_of_stack` and reset
    `priority_idx = state.active_player_idx`; with empty stack,
    return (end-of-step). The pop-only stepping-stone shape (no
    effect dispatch) is preserved — real effect dispatch is the
    job of `resolve_top_of_stack` in `state_ops_resolve.rs` per
    the cross-file private-method friction documented on the
    older `complete_priority_round`.
  - `cast_creature` resets `priority_player = active_player` AND
    `consecutive_passes = 0` on a successful cast (CR 117.1c).
    Failure paths (precondition gate, insufficient mana) are a
    no-op on priority — the reset only fires after a successful
    mutation. Mirrors Python's no-stack semantics where the
    primitive fuses push + resolve and "priority returns to AP"
    fires once at the end.
  - `apply_action` resets `priority_player = active_player` AND
    `consecutive_passes = 0` after EVERY non-PassPriority arm
    succeeds (PlayLand / DeclareAttackers / DeclareBlockers /
    CastSpell / ActivateAbility). PassPriority itself is a TRUE
    NO-OP (CE2 closure): the arm leaves `priority_player`,
    `active_player`, AND `consecutive_passes` untouched. Mirrors
    Python's `apply_action(PassPriority)` which is literally
    `return` (src/mtg_bot/actions.py:990-991). Priority management
    flows through the `pass_priority` primitive directly (which
    DOES increment the counter and toggle the priority field) or
    through the composer `complete_step_via_priority_round`
    (which calls the primitive twice). The post-arm reset block
    at the bottom of `apply_action` is skipped via the
    `is_pass_priority` short-circuit so the no-op contract is
    bit-exact.
  - `advance_phase` resets `priority_player = active_player` on the
    CLEANUP -> UNTAP wrap (CR 117.1a — the new active player gets
    priority at the start of their turn). Within-turn step advances
    do NOT touch priority (the within-step CR 117.1a reset is the
    composer's responsibility, not `advance_phase`'s). Mirrors
    Python's `simulate.py:212-213` which sets
    `state.priority_player_idx = 0` at the start of turn 1.
  - `consecutive_passes` is NOT serialized into the fingerprint —
    like `mulligan_count` (I21), it's a transient bookkeeping
    counter, not part of the gameplay-relevant state digest. Pinned
    fingerprint baselines below are unchanged. Closes parity gaps
    G3, G4, and G15 (entries removed from "Confirmed gaps" above).

### I21. `mulligan` / `mulligan_to_n` raise after 7 calls per player (CR 103.4 cap)

- **Pinned by**:
  `tests_rust/test_mulligan.py::test_mulligan_seven_calls_succeed_eighth_raises`,
  `test_mulligan_to_n_seven_calls_succeed_eighth_raises`,
  `test_mulligan_cap_is_shared_between_mulligan_and_mulligan_to_n`,
  `test_mulligan_cap_is_per_player`,
  `test_mulligan_cap_raise_does_not_mutate_state`,
  `test_mulligan_count_does_not_affect_fingerprint`.
- **Invariant** (CR 103.4 / round-26 G23 closure):
  - `Player.mulligan_count: u32` increments at the START of every
    `mulligan` / `mulligan_to_n` call (shared counter between the
    two variants).
  - If the post-increment count exceeds 7, the call raises
    `PyValueError` BEFORE touching any zone. Pinned baselines after
    7 successful mulligans are unchanged when an 8th call raises.
  - Mirrors Python `simulate.setup_game(max_mulligans=7)` — the
    London-mulligan loop allows up to 7 mulligans per player before
    forcing a keep. An 8th call would draw a hand of -1 cards, which
    is nonsensical.
  - The counter is per-player; capping player 0 does not affect
    player 1.
  - `mulligan_count` is NOT serialized into the fingerprint — it's a
    per-call safety counter, not part of the gameplay-relevant state
    digest. Pinned fingerprint baselines below are unchanged. Closes
    parity gap G23 (entry removed from "Confirmed gaps" above).

### I22. Decision-callback substrate via `pending_decisions` (CR 701.43 Surveil; G20 closure)

- **Pinned by**: every test in
  `tests_rust/test_pending_decisions.py`; the round-23 baselines in
  `tests_rust/test_resolve.py::test_resolve_surveil_*` and
  `tests_rust/test_library.py::test_effect_surveil_first_slice_keeps_library_and_top_id_unchanged`
  still hold via the auto-resolve path.
- **Invariant** (round-27 G20 closure):
  - `GameState.pending_decisions: Vec<PendingDecision>` — a FIFO
    queue of choices the engine is waiting on. The first variant is
    `PendingDecision::SurveilChoice { source_obj_id, controller,
    cards }`. Future variants (G22 `LondonBottomN`, G24
    `ScryOnKeep`, future `ModalChooseOne` / `ManifestDread`) follow
    the same shape.
  - `Effect::Surveil` pops `min(count, library_len)` cards off
    library top and packages them into a `SurveilChoice` — the cards
    live in NEITHER library nor graveyard while the decision is
    pending. The arm short-circuits; nothing else mutates.
  - `state.has_pending_decision()` / `state.pending_decisions()` —
    introspection. `state.submit_surveil_decision(top_ids,
    graveyard_ids)` — the dispatcher: every pending id must appear
    in exactly one list; mismatches raise `ValueError`
    transactionally (no mutation on failure). The dispatcher routes
    `top_ids` back to library top in the submitted order (`top_ids[0]`
    is the topmost card after submission) and `graveyard_ids` into
    the controller's graveyard in the submitted order.
  - `state.auto_resolve_pending_decisions(strategy)` — deterministic
    drain for callers that don't model the choice dimension.
    Strategies: `"keep_all_on_top"` (the round-23 baseline, bit-equal
    to the pre-r27 path) and `"send_all_to_graveyard"` (mirror, for
    surveil-for-graveyard-fill effects). An unknown strategy raises
    `ValueError` transactionally.
  - `resolve_top_of_stack` opts INTO `"keep_all_on_top"` auto-resolve
    at the end so existing baseline tests (round-23 Surveil etc.) are
    bit-equal to the pre-r27 path. The new
    `resolve_top_of_stack_pause` pymethod runs the same flow WITHOUT
    auto-resolving — for callers (tests, MCTS, parity_smoke) that
    want to inspect / submit decisions.
  - The queue participates in `Clone` (cloned with the state) and in
    fingerprint serialization: a state with a pending decision
    hashes differently from one without. The fingerprint serializer
    appends the pending-decisions slot ONLY when non-empty, so
    pre-r27 pinned baselines (every state in `test_fingerprint.py`
    has empty `pending_decisions`) remain bit-equal — no baseline
    shifts in this slice.
  - Mirrors Python's pause-and-resume decision shape in
    `src/mtg_bot/search/resolution_channel.py`
    (`ResolutionSurveilDecision`, the `ResolutionChannelAgent` pause
    / rollback / replay harness).
  - **Round-28 extension — `MayDoChoice` variant.** Decision-callback
    substrate now carries a second variant:
    `PendingDecision::MayDoChoice { source_obj_id, controller,
    inner_effect: Box<Effect> }`, pushed by `Effect::MayDo {
    inner_effect, prompt }` — Python's "you may [inner]" wrapper from
    `effects/generic/may_do.py`. Resolve short-circuits exactly as
    Surveil does; `state.submit_may_do_decision(yes: bool)` consumes
    the head, recursively applying the inner effect on "yes" (with the
    same `source_obj_id` so identity-coupled "if you do" riders retain
    the resolving spell's identity) and dropping it on "no". The auto-
    resolve strategy set grew to include `"may_do_yes"` / `"may_do_no"`
    / `"default"`; `"default"` answers "yes" (mirrors Python's base-
    Agent no-seated-agent default), Surveil falls through to
    `"keep_all_on_top"`. `resolve_top_of_stack` (non-pause) auto-
    resolves with `"keep_all_on_top"` which now also routes MayDo to
    "yes" (fallthrough rule). The fingerprint serializer adds a
    MayDoChoice arm (variant-tag + source/controller + inner-effect
    Debug format) so a pending MayDo participates in canonicalization
    just like Surveil. Pinned by the round-28 MayDo block in
    `tests_rust/test_pending_decisions.py` (13 tests covering parse,
    push-pending, submit yes/no, auto-resolve strategies, dispatcher
    cross-variant rejection, fingerprint divergence, clone isolation).
  - **Round-28 extension — `PickCardFromExaminedChoice` variant.**
    Third variant on the multi-variant substrate, pushed by
    `Effect::LookTopNPutOneInHandRestToGraveyard` (Commune-with-Evil
    family, 4 EoE cards). Payload: `{ source_obj_id, controller,
    cards: Vec<GameObject> }` carrying the examined cards in popped
    order (`cards[0]` = topmost). Dispatcher
    `submit_pick_card_decision(chosen_id)` routes `chosen_id` to
    controller's hand and every other examined card to controller's
    graveyard (popped order); unknown id raises `ValueError`
    transactionally. Auto-resolve gains `"pick_top"` (picks
    `cards[0]`) plus `"default"` fallback (picks `cards[0]`) —
    mirrors Python's no-agent-seated fallback in
    `LookTopNPutOneInHandRestToGraveyard.resolve`. Fingerprint
    serializer carries the per-card object signature (cards live in
    NEITHER library nor graveyard while pending — must participate
    in fingerprint to distinguish a pending state from one where
    routing already completed). Pinned by 16 new tests in
    `tests_rust/test_pending_decisions.py` (parse + default-n +
    negative-n; pop+push; submit by id covering topmost / non-
    topmost / unknown id / empty queue; cross-variant dispatcher
    rejection; auto-resolve `pick_top` / `default`; non-pause
    `resolve_top_of_stack` auto-resolve; zero-count / library-clamp;
    fingerprint divergence; clone isolation).
  - **Round-28 extension — `ChooseModeChoice` variant (4th).** Fourth
    variant on the multi-variant substrate, pushed by
    `Effect::ChooseOneMode { modes: Vec<Effect> }` — Python's
    `ModalChooseOne(modes=...)` (combinators.py:1112, 13 EoE cards)
    in its first-slice shape: resolve-time mode picking only (cast-
    time mode routing via `state.resolving_modes`, `or_both`, and
    per-mode target picking deferred). Payload: `{ source_obj_id,
    controller, modes: Vec<Effect> }`. Dispatcher
    `submit_choose_mode_decision(chosen_idx)` validates index range
    transactionally then recursively applies `modes[chosen_idx]` via
    `apply_effect` with the original `source_obj_id`. Auto-resolve
    gains `"choose_mode_0"` (picks `modes[0]`); `"default"` fallback
    also picks `modes[0]` — mirrors Python's no-agent-seated
    fallback in `ModalChooseOne.resolve`. Empty `modes` list is
    rejected at register time so the resolve arm always has at least
    one mode. Fingerprint serializer carries each mode's Debug
    format (same Debug-over-Serialize choice as MayDoChoice). Pinned
    by 14 new tests in `tests_rust/test_pending_decisions.py`:
    parse + missing-modes + empty-modes rejection; resolve-pushes-
    pending; submit mode-0 (draw) + mode-1 (gain-life) + out-of-
    range + empty-queue + cross-variant; auto-resolve
    `choose_mode_0` / `default`; non-pause `resolve_top_of_stack`
    auto-resolve; fingerprint divergence; clone isolation.
  - **Round-28 extension — `IfYouDoChoice` variant (5th).** Fifth
    variant on the multi-variant substrate, pushed by
    `Effect::IfYouDo { do_action, then_action }` — Python's
    `IfYouDo` (combinators.py:560, 7+ EoE cards) in its first-slice
    "looter" shape: do_action assumed implicitly success-on-yes
    (i.e. `IfYouDo(MayDo(X), Y)` form). Payload: `{ source_obj_id,
    controller, do_action: Box<Effect>, then_action: Box<Effect> }`.
    Dispatcher `submit_if_you_do_decision(yes)` applies both nested
    actions in order on "yes", drops both on "no" — with the same
    `source_obj_id` propagated so identity-coupled riders retain
    identity. Auto-resolve gains `"if_you_do_yes"` / `"if_you_do_no"`;
    `"default"` answers "yes" (mirrors Python's no-agent-seated
    `IfYouDo.resolve` — base-Agent says yes to the wrapped MayDo,
    so both fire). The `if_you_do_no` strategy also routes a
    MayDoChoice head to "no" (shared yes/no semantics — keeps a
    multi-variant queue draining predictably). General IfYouDo with
    do_actions that can signal failure (e.g. SacrificeACreature
    returning False on empty bf) is deferred — would require
    `apply_effect` to return `Result<bool>`. Pinned by 14 new tests
    in `tests_rust/test_pending_decisions.py`: parse + missing-
    fields rejection (both directions); resolve-pushes-pending;
    submit yes/no; empty-queue + cross-variant dispatcher rejection;
    auto-resolve `if_you_do_yes` / `if_you_do_no` / `default`; non-
    pause `resolve_top_of_stack` auto-resolve; fingerprint
    divergence; clone isolation.

### I23. UEOT-modifier framework: per-object expiring continuous effects with CR 514.3 cleanup-step expiration

- **Pinned by**:
  `tests_rust/test_ueot.py` (19 tests — full lifecycle including
  effect parsing, effective-P/T accessors, PumpSelfUntilEOT resolve,
  cleanup_step / advance_phase / end_turn_and_start_next clearing,
  CoW preservation across clones);
  `tests_rust/test_combat.py::test_ueot_pumped_attacker_damages_defender_for_effective_power`,
  `test_ueot_pumped_blocker_survives_what_would_have_been_lethal`,
  `test_ueot_pumped_creature_dies_when_ueot_clears`,
  `test_ueot_granted_keyword_via_has_keyword_reader`;
  `tests_rust/test_cleanup.py::test_cleanup_step_no_op_at_initial_state`,
  `test_cleanup_step_returns_summary`,
  `test_clone_then_cleanup_does_not_affect_original`,
  `test_cleanup_is_idempotent_when_already_clean`,
  `test_cleanup_alone_does_not_move_objects_between_zones`
  (each pin the new `ueot_cleared` summary-dict key).
- **Invariant** (CR 613 + CR 514.3 / round-27 G7 closure):
  - `GameObject` carries a `ueot_modifiers: Vec<UeotModifier>` Vec.
    `UeotModifier` is a closed enum with three variants today:
    `PowerToughnessBonus { power, toughness }` (CR 613.7c, pump
    effects), `GrantKeyword { keyword }` (CR 613.6, haste / flying
    / lifelink etc.), and `BecomesLandCreature { power, toughness,
    color_or_subtype }` (CR 613.4 + 7b, manland-animate). Adding a
    new variant is a build-time error in every effective-P/T /
    keyword reader until handled (exhaustiveness via the
    `UeotModifier::… {} => {}` arms in `ueot_power_bonus()` /
    `ueot_toughness_bonus()` / `ueot_has_keyword()`).
  - Adding the Vec costs the struct its `Copy` derive. The empty
    case is a 24-byte null-ptr / zero len/cap (no heap allocation),
    so the per-object footprint grows but stays cheap on the warm
    path. One downstream call site (`state_ops_library.rs`'s
    `find_obj` closure that previously used `*ref` deref) now uses
    `.clone()`.
  - Effective-P/T readers fold UEOT bonuses on top of printed +
    counter contributions: combat's `battlefield_creature_power` /
    `_toughness`, SBA's `collect_sba_victims` /
    `is_lethal`. The public Python-facing getters are
    `effective_power(obj_id)`, `effective_toughness(obj_id)`,
    `has_keyword(obj_id, keyword)` (all on `GameState`,
    Python-callable, mirror Python's `mtg_bot.game.effective_*`).
  - **Intentional asymmetry**: `battlefield_creature_toughness`
    (combat-side) folds UEOT bonus but NOT `obj.counters`; SBA's
    effective_toughness folds BOTH. This mirrors the pre-r27
    asymmetry (combat math sized off printed P/T, SBA's lethal-
    damage / zero-toughness keyed off printed + counters). Future
    slice can promote combat to fold counters once a parity test
    demands it.
  - **CR 514.3 firing point**:
    `cleanup_step_for_active_player` runs CR 514.1 (hand-size
    discard) → CR 514.2 (damage clear) → CR 514.3 (UEOT clear),
    in that order. Its summary dict gains a new `ueot_cleared` key.
    `advance_phase` composes the same UEOT clear via its inline
    mirror `cleanup_step_for_active_player_inline` on entry to
    PHASE_CLEANUP. A new `clear_ueot_modifiers_all` primitive on
    GameState is callable independently for tests that want to
    assert the clear in isolation. Closes parity gap G7 (entry
    removed from "Confirmed gaps" above).
  - **PumpSelfUntilEOT** (12-card EoE pool target) is the proving
    Effect. Parse arm: `pump_self_until_eot` with
    `power_bonus: int = 0`, `toughness_bonus: int = 0` defaults
    (mirrors Python's dataclass shape). Apply arm walks both
    battlefields to find the source by id, pushes a
    `PowerToughnessBonus` modifier onto the source's
    `ueot_modifiers` Vec, no-op if source is not on battlefield.
    Negative bonuses allowed (debuff edge case); 0/0 is a legal
    warm-path no-op (skipped before the Vec push).
  - **UEOT modifiers are NOT serialized into the parity
    fingerprint.** A mid-turn snapshot vs an end-of-turn snapshot
    would naturally differ in raw modifier Vecs but be parity-
    equal at gameplay boundaries (cleanup auto-drains). The 12
    pinned fingerprint baselines below are unchanged — `object_signature`
    in `state_ops_fingerprint.rs` continues to project only
    `(name, tapped, summoning_sick, damage, counters, attached_to,
    controller, id)`. Effective P/T (with bonus) still surfaces
    through the `effective_power` / `effective_toughness` getters
    — that's the gameplay-load-bearing observable, not the raw
    modifier list. A future slice that needs end-of-turn vs mid-
    turn parity at fingerprint level can fold UEOT in and shift
    every baseline; doing so today would bind the smoke harness to
    a Python-side serializer shape that hasn't been pinned.

---

## Pinned fingerprint baselines

Twelve-character lowercase-hex SHA1 digests pinned by the test suite.
Each baseline corresponds to a specific state shape. If any drifts
on the Rust side, either `scripts/parity_smoke.py:hash_state` or the
Rust fingerprint serializer has changed and the matching offline
Python recipe must be updated in lockstep.

### Synthetic states (`engine.make_synthetic_state(...)`)

| Fingerprint    | State shape                                          | Test                                                                 |
| ---            | ---                                                  | ---                                                                  |
| `93ab1bd03f94` | default args: 10 bf / 7 hand / 50 library            | `test_fingerprint.py::test_fingerprint_matches_python_hash_state_default` |
| `e1d70afe6742` | `(2, 3, 5)` — small state                            | `test_fingerprint.py::test_fingerprint_matches_python_hash_state_small`   |
| `3b17024676c6` | `(0, 0, 0)` — empty zones per side                   | `test_fingerprint.py::test_fingerprint_pinned_for_zero_zones`             |
| `56d2910a6ac4` | `(1, 1, 1)` — minimal non-empty                      | `test_fingerprint.py::test_fingerprint_pinned_for_one_each`               |
| `e32afdb188c1` | `(3, 3, 3)` — mid-size symmetric                     | `test_fingerprint.py::test_fingerprint_pinned_for_three_each`             |
| `6ff3d1babb85` | `(10, 10, 10)` — bigger symmetric                    | `test_fingerprint.py::test_fingerprint_pinned_for_ten_each`               |
| `5c67f92a0cb4` | `(0, 5, 0)` — only hands                             | `test_fingerprint.py::test_fingerprint_pinned_for_hand_only`              |
| `f9848ca12e57` | `(5, 0, 0)` — only battlefield                       | `test_fingerprint.py::test_fingerprint_pinned_for_battlefield_only`       |
| `f251f42f8647` | `(0, 0, 5)` — only library                           | `test_fingerprint.py::test_fingerprint_pinned_for_library_only`           |
| `6f067c3988bc` | `(5, 0, 10)` — asymmetric                            | `test_fingerprint.py::test_fingerprint_pinned_for_asymmetric`             |

### Synthetic-state mutation deltas

| Fingerprint    | State + mutation                                       | Test                                                                  |
| ---            | ---                                                    | ---                                                                   |
| `e35e82c9b588` | `(3, 3, 3)` after `tap_object(0)`                      | `test_fingerprint.py::test_fingerprint_pinned_after_tap_mutation`     |
| `d2435eea284f` | `(1, 1, 1)` after `set_life(0, 5)`                     | `test_fingerprint.py::test_fingerprint_pinned_after_life_mutation`    |

### `make_state` canonical baseline

| Fingerprint    | Setup                                                                                       | Test                                                                |
| ---            | ---                                                                                         | ---                                                                 |
| `a1193453edec` | `make_state` with the 2-card Forest+Bears registry, `[0,0,1]` / `[1,1,0]` decks, no hands   | `test_setup.py::test_make_state_canonical_default_fingerprint_pinned` |

---

## Known divergences flagged by cross-engine smoke (xfail-pinned)

Round 27 / Phase 5 — the first cross-engine smoke harness landed in
`scripts/parity_smoke.py --mode cross-engine`. It drives matched
Python and Rust states through N PassPriority decisions on a tiny
synthetic deck (Forest + Grizzly Bears) and reports per-field
divergences with gap-doc lookup. The xfail-pinned tests in
`tests/test_parity_smoke.py` (two as of r27) flag the inaugural
NOVEL divergences — when the underlying gap closes, the xfail flips
to xpass and we either renumber as G<n> below or drop it from this
section.

The inaugural cross-engine NOVEL divergences (NOT yet entered as
proper G-numbered gaps because the closure path may be cosmetic
rather than architectural — TBD by the follow-up slice):

### ~~CE1.~~ CLOSED in round 28: `summoning_sick` default aligned to Python's universal `True`

Closed by the round-28 CE1 slice. Rust's `make_state` (and
`state_ops_deck.rs::set_library_for_player`) now default
`summoning_sick = true` on every freshly-allocated library / hand
GameObject, matching Python's universal default. The field is
gameplay-irrelevant in non-battlefield zones; the cleared-on-untap
semantics still apply on the battlefield. New pymethod
`clear_summoning_sick_for_player(player_id)` exposes the untap-step
sweep so test fixtures that move creatures from hand to battlefield
via `move_object` can mark them as "ready to act" without driving
a full turn loop. Pinned fingerprint baselines shifted: the
`make_state` canonical default digest is now `411d5b757c22` (was
`a1193453edec` in r23). Pin in
`tests_rust/test_setup.py::EXPECTED_MAKE_STATE_CANONICAL_DEFAULT_FP`.
The `make_synthetic_state` test fixture in `state.rs` STAYS at
`summoning_sick=false` for battlefield creatures (test convenience —
synthetic battlefield is "creatures that have been on the
battlefield long enough"). The `register_and_inject_object` test-
injection primitive in `state_ops_resolve.rs` also stays at
`summoning_sick=false` for the same reason. Both are Rust-only test
divergences from Python's default; not load-bearing on the cross-
engine smoke harness which uses `make_state` exclusively.

### ~~CE2.~~ CLOSED: `apply_action(PassPriority)` is now a no-op matching Python

Closed by the round-28 CE2 slice (`feature/ce2-pass-priority`).
Rust's `apply_action({"kind": "pass_priority"})` is now a TRUE
NO-OP — it does NOT flip `priority_player` and it does NOT
increment `consecutive_passes`. This mirrors Python's
`apply_action(PassPriority)` which is literally `return`
(src/mtg_bot/actions.py:990-991). Priority management is owned by
the composer (`complete_step_via_priority_round`) and by callers
driving the `pass_priority` primitive directly — NOT by
`apply_action`. The xfail-pinned cross-engine smoke test
(`tests/test_parity_smoke.py::test_cross_engine_mode_priority_player_unchanged_by_pass_priority`)
flipped to expected-pass with the closure. New regression tests in
`tests_rust/test_priority.py` (5 tests in the "CE2 closure
regression" block) pin the no-op behavior in both directions
(p0 passes, p1 passes), pin the counter-unchanged contract, pin the
active-player-unchanged contract, and pin fingerprint stability.
Pre-existing rust-side tests that asserted the OLD toggle behavior
were flipped to assert the new no-op semantics (in
`tests_rust/test_actions.py` and `tests_rust/test_legal.py`). The
I25 invariant text below was tightened to reflect that
PassPriority via apply_action is now bit-equivalent to Python's
return-fast path.

### ~~CE3.~~ CLOSED: Library orientation aligned to Python's `index 0 = top` convention

Closed by the CE3 closure slice (`feature/ce3-library-pop`). Rust now
treats `lib[0]` as the top of the library (matching Python's
`library.pop(0)` convention) across every library mutator that
touches orientation:

- `state_ops_zones.rs::draw_card` — `lib.pop()` → `lib.remove(0)`.
- `state_ops_steps.rs::draw_card_inline` (the turn-loop draw step
  inline mirror) — same flip.
- `state_ops_mulligan.rs`: `mulligan` / `mulligan_to_n` /
  `mulligan_atomic` all redraw via `lib.remove(0)`;
  `bottom_cards_from_hand` + `mulligan_atomic`'s bottom-step now
  append to the tail (post-CE3 bottom = `lib[len-1]`).
- `state_ops_library.rs`: `scry` reads `lib[..n]` forward;
  `search_library_for_first_of_type` (and the move/shuffle variants)
  iterate forward; `mill_target_player` removes from head; `apply_scry`
  rebuilds with top_ids at the head and bottom_ids at the tail.
- `state_ops_resolve.rs`: `Effect::DrawCards`, `Effect::Mill`,
  `Effect::MillTargetPlayer`, `Effect::Surveil`,
  `Effect::LookTopNPutOneInHandRestToGraveyard`,
  `Effect::ManifestDread`, `Effect::ManifestDreadThenAttachSelf`,
  `Effect::LookAtTopNRevealOrCounter`,
  `Effect::EachOpponentMillsCards`,
  `Effect::SearchLibraryForBasicLandTapped`, and the broader
  `Effect::SearchLibraryForCard` family all walk top-first by
  forward iteration; `Effect::PutCardFromGraveyardOnBottomOfLibrary`
  and `Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary`
  push to the tail (bottom).
- `state_ops_decisions.rs`: `resolve_surveil_with_strategy`
  (`keep_all_on_top`) restores cards via `lib.insert(0, ...)` in
  reverse; the look-at-top-N reveal-yes / reveal-no submit paths
  push to the tail in reverse order so cards[0] (the topmost
  popped) lands at the very bottom (parity with the pre-CE3
  contract).
- `state_ops_choices.rs::choose_top_of_library_for_player` reads
  `library.first()` instead of `library.last()`.
- `state_ops_resolve.rs::register_and_inject_object` — the test-
  injection primitive now inserts injected library cards at `lib[0]`
  (top) so existing test fixtures that expect injected cards to be
  reachable by top-N readers continue to work without per-test
  changes.

The xfail-pinned cross-engine smoke test
`tests/test_parity_smoke.py::test_cross_engine_play_a_land_selector_walks_clean`
flipped to expected-pass; the companion symptom-pin
`test_cross_engine_play_a_land_selector_reports_ce3_symptom_at_step_0`
was removed alongside the xfail per the closure pair contract. The
existing Rust tests that pinned the pre-CE3 "top = lib.last()"
convention (test_zones / test_library / test_deck / test_choices /
test_mulligan / test_pending_decisions / test_resolve) were
updated to assert the post-CE3 "top = lib[0]" convention. Pinned
fingerprint baselines stayed bit-identical — the orientation flip
changed which end is *named* top but did NOT change the underlying
Vec construction order (`make_state` and `make_synthetic_state`
push input order into the Vec; the fingerprint walks that Vec
front-to-back unchanged).

### ~~CE4.~~ CLOSED: `battlefield_snapshot` emits in `etb_timestamp` order (matching Python's centralized append order)

Closed by the CE4 closure slice (`feature/ce4-bf-etb-order`). Option
(2) from the brief — the minimal-surface "stamp + sort" closure — was
taken. The per-player `battlefield: Arc<Vec<GameObject>>` split stays
intact; the snapshot and fingerprint readers stable-sort their
collected refs by a new per-object `etb_timestamp: u32` slot before
emitting.

- **Per-object slot**: `GameObject.etb_timestamp: u32` (defaults to
  `0`). Stamped at every battlefield-entry primitive by reading
  `state.next_etb_timestamp` and bumping the counter
  (`state::reserve_etb_timestamp(&mut state)`). Stamped sites:
  * `state_ops_cast.rs::move_from_hand_to_battlefield` (the
    `play_land` + `cast_creature` shared helper — refactored to take
    `&mut GameState` so it can bump the counter; both call sites
    updated).
  * `state_ops_zones.rs::move_object` (the generic mover —
    conditionally stamps when `target_zone == ZONE_BATTLEFIELD`).
  * `state_ops_zones.rs::move_object_from_graveyard_to_battlefield`
    (the reanimation primitive).
  * `state_ops_resolve.rs::move_source_post_resolve` (the canonical
    CR 608.2m permanent-mover — covers ResolveCreature /
    ResolveArtifact / ResolveEnchantment / ResolvePlaneswalker / any
    permanent-typed source landing post-resolve).
  * `state_ops_resolve.rs::Effect::CreateToken` (multi-token batch —
    reserves a contiguous block of timestamps so the within-effect
    creation order is strictly monotonic; bumps the counter once per
    token).
  * `state_ops_resolve.rs::Effect::CreateTokenCopyOfTargetCreatureYouControl`.
  * `state_ops_resolve.rs::Effect::SearchLibraryForBasicLandTapped`
    (loops over `count`; each searched land gets a fresh timestamp).
  * `state_ops_resolve.rs::Effect::ReturnCreatureFromYourGraveyardToBattlefield`.
  * `state_ops_resolve.rs::ReturnAllCreaturesFromYourGraveyardToBattlefield`
    (multi-bf-push; contiguous timestamp block, one per moved creature).
  * `state_ops_resolve.rs::register_and_inject_object` (test-injection
    primitive; stamps only when `target_zone == ZONE_BATTLEFIELD`).
  * `state_ops_resolve.rs::return_paired_exiles_for` (the round-28
    "until source leaves" manual hook; multi-bf-push, contiguous
    timestamp block).
  * `state_ops_decisions.rs::resolve_manifest_dread_to_battlefield`
    (manifest-dread chosen card → bf).

- **Bookkeeping counter**: `GameState.next_etb_timestamp: u32`
  (defaults to `0`). NOT serialized into the fingerprint — same
  precedent as `mulligan_count` and `consecutive_passes`: transient
  bookkeeping, not gameplay-relevant. Two states reached via
  different histories that share the same observable battlefield
  ORDER share a fingerprint regardless of how many bf-entries
  happened to get there.

- **Per-object slot is also NOT in the fingerprint's 8-tuple.** The
  field is read only as the **stable-sort key** in
  `state_ops_snapshot.rs::battlefield_snapshot` and the matching
  battlefield walk in `state_ops_fingerprint.rs::fingerprint`. The
  per-object 8-tuple shape (`printing_name, tapped, summoning_sick,
  damage, counters, attached_to, controller_id, id`) is unchanged.
  Consequence: every pinned fingerprint baseline involving
  `make_synthetic_state` stays bit-equal, because synthetic-state
  battlefields are built via `make_player` (which doesn't call
  `reserve_etb_timestamp`) — every object has timestamp 0, the
  stable-sort tie-breaks on the original p0-then-p1, insertion-order
  walk, and the emitted sequence matches the pre-CE4 walk byte-for-
  byte. Real ETB-driven flows produce strictly-monotonic timestamps
  (each `reserve_etb_timestamp` call bumps the counter), so the sort
  produces the ETB-interleaved order Python emits.

The xfail-pinned cross-engine smoke test
`tests/test_parity_smoke.py::test_cross_engine_multi_turn_selector_walks_clean`
flipped to expected-pass; the companion symptom-pin
`test_cross_engine_multi_turn_selector_surfaces_ce4_battlefield_ordering`
was removed alongside the xfail per the closure-pair contract
(mirrors the CE3 closure pattern). The pre-CE4 test
`tests_rust/test_snapshot.py::test_battlefield_snapshot_p0_then_p1_ordering`
that pinned the controller-grouped emission order was renamed and
inverted to
`test_battlefield_snapshot_etb_timestamp_ordering`, asserting that
two `register_and_inject_object` calls into ZONE_BATTLEFIELD emit
in the order they were called (p1's Bears first, p0's Forest
second), not in controller-grouped order. All other pinned
fingerprint baselines stayed bit-equal — full rust gate (1693
tests) green post-flip.

The matching cross-engine smoke (50 decisions with `multi_turn_selector`
on `slightly_bigger_deckspec`, seed 42) walks clean across all 50
steps post-closure. Pre-CE4 the same harness aborted at step 28
with the `battlefield` field divergence — Python emitting
Mountain-then-Forest (ETB order), Rust emitting Forest-then-Mountain
(controller order).

### ~~CE6.~~ CLOSED: `collect_sba_victims` walks the battlefield in ETB-timestamp order (via shared `bf_iter_etb_sorted` helper)

> **Note on numbering.** The closure playbook
> (`docs/CE5_CE6_CE7_CLOSURE_PLAYBOOK.md`) reserved CE5 for this
> SBA-collection divergence based on the audit's ranked predictions,
> but the cast_bears_selector probe surfaced a different divergence
> first (the cast-vs-resolve gap below) and claimed the CE5 number.
> Per the playbook's coordination note at the cast-creature CE5
> entry, the planned CE5/CE6/CE7 sequence bumps to CE6/CE7/CE8
> when the corresponding slices land. This entry is the bumped CE6
> (closing the playbook's predicted "CE5 — SBA victim collection")
> per `feature/ce5-sba-order` branch + commit `cdb01fcd`.

Closed by the SBA-collection closure slice (`feature/ce5-sba-order`).
The
SBA victim-collection walk in
`rust/src/state_ops_sba.rs::collect_sba_victims` was a per-player
`for player in &self.players { for obj in player.battlefield.iter() }`
loop — p0-first, p1-second, in each player's per-player battlefield
Vec order. This diverged from Python's `turn.py:1242`
`_state_based_actions_pass` which walks the centralized
`state.battlefield` list (ETB-monotone post-CE4). Whenever ETB
timestamps interleaved across players, the two engines collected
victims in different orders — observable downstream by death-trigger
push order once Rust grows a trigger emitter (gated on CE8 substrate
work).

**Closure shape.** A new shared module
`rust/src/bf_iter.rs` exposes
`pub(crate) fn bf_iter_etb_sorted(state: &GameState) -> Vec<&GameObject>`,
matching the post-CE4 stable-sort precedent already established by
`state_ops_snapshot.rs::battlefield_snapshot` and
`state_ops_fingerprint.rs::fingerprint`. The helper collects refs
across both players' battlefields, then `sort_by_key(|o| o.etb_timestamp)`
— stable, so synthetic-state ties (every bf object at timestamp 0)
preserve the pre-closure p0-then-p1 walk and pinned baselines stay
bit-equal. `collect_sba_victims` now consumes the helper:

```rust
let mut victims = Vec::new();
for obj in crate::bf_iter::bf_iter_etb_sorted(self) {
    // ... unchanged 704.5f / 704.5g predicate logic ...
}
```

Only the iteration shape changed; the predicate logic (indestructible
filter, UEOT folds, effective-toughness calculation, 704.5f-before-
704.5g CR ordering) is bit-equal pre/post.

**Tests.** Five new regression tests in
`tests_rust/test_sba.py` under the "CE5 closure: SBA victim
collection" section heading (the section comment in the test file
retained the playbook's original CE5 label since the test code was
written before the in-flight cast-creature CE5 entry landed):

- `test_collect_sba_victims_walks_etb_order_across_players` — interleaved
  ETB stamps across p0/p1, verifies end-state per-graveyard projection
  (each player's graveyard receives its dead in the relative ETB order
  they died).
- `test_collect_sba_victims_etb_order_same_player_interleave` — pins
  the no-regression case (same-player projection preserves relative
  ETB order through a cross-player stamp interleave).
- `test_collect_sba_victims_etb_order_visible_via_clone_and_count` —
  cross-player simultaneous SBA where p1's creature has a lower ETB
  stamp than p0's; both end up graveyarded.
- `test_collect_sba_victims_empty_when_no_lethal` — empty-case path
  through the helper (no victims emitted).
- `test_collect_sba_victims_synthetic_state_tie_break_preserves_p0_first`
  — pins the synthetic-state tie-break (every timestamp == 0 →
  stable sort preserves p0-then-p1 walk, pre-closure baselines
  bit-equal).

**Fingerprint baselines.** Zero shifts. Synthetic-state bf objects
all carry `etb_timestamp == 0` (because `make_player` doesn't call
`reserve_etb_timestamp`), and the stable-sort tie-break preserves
the pre-closure p0-then-p1 walk on those fixtures. Per-graveyard
contents also stay bit-equal — `apply_sbas` pushes each victim onto
its controller's graveyard, so the same-player projection of any
collection order produces the same per-graveyard sequence. The
closure becomes observable only downstream: the trigger emitter
(future CE-emitter slice) will inherit ETB-ordered death-trigger
pushes, and cross-engine smoke at interleaved-ETB lethal scenarios
will agree on the collection order.

Full rust gate (1698 tests) green post-closure. The new shared
helper is also positioned for the playbook's predicted CE7 (combat
damage assignment, formerly CE6) and CE8 (cross-player find-first
Effect arms, formerly CE7) — those slices will consume
`bf_iter_etb_sorted` rather than re-inlining the collect-and-sort.
The two existing inline sites (`state_ops_snapshot.rs:189`,
`state_ops_fingerprint.rs:316-327`) remain inlined for now;
refactoring them to call the helper is a follow-up post-closure
housekeeping pass.

### I27. SBA victim collection walks battlefield in ETB-timestamp order via shared `bf_iter_etb_sorted` helper

Post-closure (CE6, per `feature/ce5-sba-order`),
`collect_sba_victims` enumerates battlefield objects via
the centralized `crate::bf_iter::bf_iter_etb_sorted` helper, which
stable-sorts refs across both players' battlefields by per-object
`etb_timestamp`. Tie-break preserves the original p0-then-p1
insertion-order walk so synthetic-state fixtures (where every bf
object carries timestamp 0) stay bit-equal with the pre-closure
collection order; real ETB-driven flows (`reserve_etb_timestamp`
bumps) produce strictly-monotonic timestamps and the helper emits
the ETB-interleaved order Python emits via its centralized
`state.battlefield` walk.

The helper is the post-CE4 abstraction step: every battlefield-walking
reader in the engine that needs ETB-ordered cross-player semantics
should call `bf_iter_etb_sorted` rather than re-inline a collect-and-
sort. Today's consumers: `collect_sba_victims` (this invariant).
Planned consumers per `docs/CE5_CE6_CE7_CLOSURE_PLAYBOOK.md` (using
its original numbering): combat damage assignment (playbook's CE6,
bumped to CE7 per the cast-creature CE5 coordination note),
cross-player find-first Effect arms (playbook's CE7, bumped to CE8),
future trigger emitter. The two existing inline stable-sort sites
(`state_ops_snapshot.rs::battlefield_snapshot` line 189 and
`state_ops_fingerprint.rs::fingerprint` lines 316-327) are CE4-closed
and produce identical output to the helper; refactoring them to call
the helper is a follow-up housekeeping pass.

### I26. `battlefield_snapshot` / fingerprint emit in ETB-timestamp monotonic order

Post-CE4 the observable battlefield order is the per-object
`etb_timestamp` sort key, stable-sort tie-breaking on the original
p0-then-p1, insertion-order walk. The invariant: two states with
the same set of bf objects and the same `etb_timestamp` values
produce identical `battlefield_snapshot` outputs and identical
fingerprint battlefield slots.

The fingerprint reads ETB order via this sort and does NOT serialize
`etb_timestamp` into the per-object 8-tuple — so the same gameplay-
relevant content reached via different ETB histories produces the
same digest as long as the relative order of the bf objects ends up
the same. (Two histories that produce different relative bf orders
do produce different digests; that's the intended sensitivity from
the CE4 closure.) Synthetic-state baselines all have timestamp 0
per-object and tie-break to the pre-CE4 walk, so the format is
backward-compatible with every pre-CE4 pinned digest.

The counter `state.next_etb_timestamp` and the per-object slot
`etb_timestamp` are both NOT serialized into the fingerprint
(transient bookkeeping; mirrors the `mulligan_count` /
`consecutive_passes` precedent).

### CE5. `cast_spell` resolves immediately on Rust (no stack) — Python pushes a StackItem and defers resolve

> **Note on playbook divergence.** The pre-staged
> `docs/CE5_CE6_CE7_CLOSURE_PLAYBOOK.md` (merged in `e1989b8b`)
> predicted CE5 would be the SBA victim collection divergence
> based on the ORDER_SENSITIVE_SURFACES.md audit's ranking. The
> cast_bears_selector probe surfaced a different divergence FIRST
> — the cast-vs-resolve gap below — because the probe drove
> `cast_spell` before reaching any SBA-relevant state (no
> creature death on this deck). The playbook's planned "CE5
> closure — SBA victim collection" should be re-labeled as CE6
> when a future slice implements it; its CE6 (damage assignment)
> and CE7 (target picks) bump correspondingly. The playbook's
> shared helper proposal (`bf_iter_etb_sorted`) and its closure
> recipes stay valid — they're contingent on the SBA / combat /
> target-pick surfaces being exercised, which CE5 (this entry)
> doesn't touch.

- **Pinned by**:
  `tests/test_parity_smoke.py::test_cross_engine_cast_bears_selector_walks_clean`
  (strict-xfail, this slice). Companion passing pin
  `test_cross_engine_cast_bears_selector_surfaces_ce5_cast_resolve_at_step_108`
  fixes the exact symptom values (Python stack has 1 item, Rust
  stack is empty; Rust battlefield has the Bears, Python doesn't).
- **Surfaced by**: Phase-5 follow-up #4 — the `cast_bears_selector`
  in `src/mtg_bot/parity/cross_engine.py`. Drives multi-turn play on
  `slightly_bigger_deckspec` until p0 has Grizzly Bears in hand AND
  two untapped non-sick Forests on the battlefield (turn 9 PRECOMBAT_MAIN).
  Activates each Forest in turn for `{G}{G}`, then emits
  `cast_spell(card_id=Bears)`. The first divergence appears on
  **step 108**, the cast step itself:
  - Python `battlefield`: 6 objects (no Bears). Python `stack`:
    `(('None', 0, 0, 0, 0, None, 0, None),)` — one StackItem for the
    Bears spell, signature degenerate because `StackItem` doesn't
    carry a `printing` attribute the zone-signature reader can read.
  - Rust `battlefield`: 7 objects, last is `('Grizzly Bears', tapped=0,
    summoning_sick=1, ...)`. Rust `stack`: empty `()`.
- **Rust today**: `state_ops_cast.rs::cast_creature` (line 298) fuses
  cast + resolve into one primitive. After validating the printing
  type is `"Creature"` and the mana cost is payable from the pool, it
  pays the cost, calls `move_from_hand_to_battlefield`, and stamps
  the new permanent with `summoning_sick = true`. The card never
  transits the stack. The docstring at line 293 explicitly flags
  this: *"No-stack semantics. The Python pipeline pushes a
  `StackItem`, settles priority, then resolves; this primitive
  short-circuits to the resolved state. ETB triggers, 'when X is
  cast' hooks, and any choice points the spell normally presents
  are deferred until a Rust-side stack lands."*
- **CR / Python expected**: CR 601.2 — casting a spell puts it on
  the stack; resolution happens later as a separate step (CR
  608.2). Python's `actions._cast_spell` (line 1361) pushes a
  `StackItem` onto `state.stack` at line 1759 and returns; the
  spell sits there until priority passes (CR 117.4) and
  `resolve_top_of_stack` (line 2655) pops it, runs the ETB
  resolver, and moves the card to the battlefield. The Python
  shape is the parity reference.
- **Impact**: high. Any decision script that casts a non-mana
  spell diverges immediately on the cast step. The cross-engine
  harness on `cast_bears_selector` aborts at step 108 with two
  field views diverged. Any downstream observable that walks
  `state.stack` (counter-spell timing, "when X is cast" triggers,
  the stack's contribution to fingerprint, an opponent's
  response-priority window) sees the divergence cascade. The
  battlefield difference also cascades onto subsequent
  `is_over`, win-determination (Rust marks game-over on empty-
  library draw via I19 at turn 10, Python defers via SBAs — see
  step 119 in the same walk).
- **Closeable**: yes — two-option closure:
  1. **Grow Rust a real stack-and-priority surface**. Refactor
     `cast_creature` to push a `StackItem`-equivalent
     (`crate::stack_item::StackItem`) onto `state.stack` and
     defer the bf transition to a `resolve_top_of_stack`
     primitive matching Python's resolution flow. Mirrors the
     full priority-loop architecture Python already has. Large
     surface — every cast site, every resolution-time hook (ETB
     triggers, "when X is cast" hooks, target re-legality
     checks, ward checks, counterspell timing), every observer
     that wants the in-flight stack contents needs to be wired
     through. This is the long-term "Rust grows the stack"
     direction the multiple `cast_creature` / no-stack-semantics
     docstrings in `state_ops_cast.rs` already flag.
  2. **Drive Python through a side-channel resolve immediately
     after cast**. Harness-side parity bridge: after every
     `cast_spell` action_dict the cross-engine harness applies,
     call `resolve_top_of_stack(py_state)` on the Python side
     so Python's stack drains down to match Rust's empty stack
     and the resolved permanent lands on Python's battlefield.
     This is a HARNESS-only mirror — production Python still
     uses the stack — but parity-aligns the cross-engine view.
     The same pattern as the multi-turn slice's
     `advance_phase_python` Python-side mirror of Rust's
     `advance_phase`. Lowest-cost option; suitable until Rust
     grows its own stack.
- **Side findings from the same walk** (NOT pinned as separate CE
  entries; recorded here so a future closure pass doesn't
  re-discover them):
  - **Rust `ActivateAbility` rejects summoning-sick LANDS.** The
    dispatch arm in `state_ops_actions.rs` (line ~523) gates on
    `obj.summoning_sick` for any source, regardless of whether
    the source is a creature. CR 302.1 restricts summoning
    sickness to creatures tapping for tap-cost abilities; lands
    are exempt. Python's `actions._activate_ability` doesn't
    gate at all (the gate is in `legal.py:661-663` and reads
    `is_creature_now(obj) && obj.summoning_sick`). The
    cross-engine harness sidesteps this by filtering sick
    Forests on the Python-reference side; without the filter
    the harness aborts at step 67 with a Rust raise on the
    freshly-played Forest. Likely root cause: the
    `state_ops_actions.rs` comment claims to mirror Python's
    gate but drops the `is_creature_now` guard. One-line fix
    on the Rust side; document as a G<n> entry alongside the
    CE5 closure when both flow together.
  - **Rust marks game-over eagerly on empty-library draw.** At
    step 119 (turn 10 ap=1 DRAW) Rust's `is_over` flips to
    True with `winner_id=0` because p1 drew from an empty
    library and the I19 `attempted_draw_on_empty` signal
    short-circuits `losers()` without waiting for an SBA pass.
    Python's `is_over()` returns False until SBAs run. This is
    documented as I19 (the eager-loss invariant is intentional
    on the Rust side) but the timing divergence relative to
    Python's SBA-deferred loss is observable on multi-turn
    cross-engine smoke and could be its own CE<n> if a future
    slice wants tighter timing parity.

(Other smoke-flagged divergences may be added by follow-up Phase 5
slices as the deck pool / decision script grows.)

---

## Maintenance

When a slice closes one of the gaps above, drop the entry from the
"Confirmed gaps" table (or move it to invariants if the closing slice
made it parity-equal). When a slice adds a new pinned divergence,
add an entry following the same shape — short name, pinning test,
Rust today, expected, impact, closeable. The 1000-game smoke harness
parses this doc by section header (`### G<n>` and `### I<n>` are the
stable anchors), so keep the numbering monotonic.

The `### CE<n>` anchors above are the cross-engine-smoke-flagged
entries (round 27 onwards). They follow the same shape as G/I entries
but use a separate prefix to make it obvious which divergences were
detected by the cross-engine harness vs the slice-internal Rust tests.
A future slice that closes a CE<n> entry either upgrades it to a
proper G<n> (with a deeper investigation than the smoke could
provide) or drops it entirely if the closure makes it a parity
invariant.
