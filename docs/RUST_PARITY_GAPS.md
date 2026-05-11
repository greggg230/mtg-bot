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

### G1. Mana pool not emptied on phase / step boundary

- **Pinned by**: `tests_rust/test_mana.py::test_mana_pool_persists_across_phase_boundary_documented_quirk`;
  `tests_rust/test_cleanup.py::test_cleanup_does_not_empty_mana_pool_documented_gap`,
  `test_end_turn_and_start_next_does_not_empty_mana_pool_documented_gap`.
- **Rust today**: `advance_phase`, `cleanup_step_for_active_player`,
  and `end_turn_and_start_next` never call `empty_mana_pool` /
  `empty_all_mana_pools`. Mana left in a pool persists indefinitely
  across phase and turn transitions.
- **CR / Python expected**: CR 106.4 — "When a phase or step ends,
  each player's mana pool empties." Python empties mana pools at every
  step/phase boundary.
- **Impact**: medium-to-high. Rust will store mana that Python has
  thrown away, so Rust can pay for spells Python cannot. Observable
  in any game where the active player floats mana out of a step.
- **Closeable**: yes — the `empty_all_mana_pools` primitive exists.
  Pending slice: wire it into `advance_phase` and the cleanup
  composition.

### G2. Hand-size enforcement does not auto-fire at end of turn

- **Pinned by**: `tests_rust/test_cleanup.py::test_advance_phase_through_cleanup_does_not_enforce_hand_size_gap`.
- **Rust today**: `advance_phase` walking through `PHASE_CLEANUP` does
  NOT invoke `cleanup_step_for_active_player` / hand-size discard.
  The caller must explicitly call `cleanup_step_for_active_player`
  or `end_turn_and_start_next` to trigger CR 514.1.
- **CR / Python expected**: CR 514.1 fires automatically at the start
  of the cleanup step.
- **Impact**: medium. Visible whenever a turn ends with > 7 cards in
  hand and the game machinery isn't explicitly calling
  `end_turn_and_start_next`.
- **Closeable**: yes — slice that wires cleanup into `advance_phase`.

### G3. `pass_priority` is a pure toggle at the bare-primitive layer (CLOSED at apply_action layer for the empty-stack branch)

- **Status (pass-priority-step-advance slice)**: CLOSED at the
  `apply_action` layer for the empty-stack/step-advance branch of CR
  117.4. The bare `state.pass_priority()` primitive remains a pure
  toggle (intentional slice-boundary pin), but
  `apply_action({"kind": "pass_priority"})` now wraps the toggle
  with a `consecutive_passes` counter on `GameState` and fires
  `advance_phase()` when both players have passed in succession with
  an empty stack. Non-pass actions reset the counter. This unblocks
  `scripts/parity_smoke.py` in `cross-engine-game` mode: pre-slice
  the harness stuck at decision 0 forever (Rust toggled,
  `legal_decisions` only emitted `pass_priority` from UNTAP without
  ever advancing); post-slice it walks UNTAP→UPKEEP→DRAW→MAIN1 via
  pass-priority pairs and reaches PlayLand / DeclareAttackers /
  DeclareBlockers emissions. See `state_ops_actions.rs` PassPriority
  arm for the implementation; `tests_rust/test_actions.py::test_
  apply_action_pass_priority_*` for the action-layer pins.
- **Still open**: the stack-resolution branch of CR 117.4 (both pass
  with non-empty stack → top resolves) is still routed through
  `state.complete_priority_round()` explicitly rather than composed
  into `apply_action`'s PassPriority arm. A future slice may extend
  the composition to handle stack-non-empty by inlining the pop +
  effect dispatch — for now `apply_action(PassPriority)` on a
  non-empty stack just toggles + increments the counter without
  firing the step-advance (test:
  `test_apply_action_pass_priority_with_nonempty_stack_does_not_advance`).
- **Bare-primitive pins (unchanged, still green)**:
  `tests_rust/test_priority.py::test_both_pass_priority_with_empty_stack_does_not_advance_step`,
  `test_both_pass_priority_with_nonempty_stack_does_not_resolve_top`,
  `test_priority_unaffected_by_phase_advance_in_this_slice`,
  `test_advance_phase_full_lap_does_not_touch_priority`. These pin
  the bare `pass_priority()` primitive's pure-toggle shape — they're
  the slice-boundary lock-in that the action-layer composition
  doesn't leak into the primitive.
- **CR / Python expected**: CR 117.4 — both-pass with empty stack
  ends the step (Rust models this now at the apply_action layer);
  with non-empty stack resolves the top (still deferred). CR 117.1a
  — active player gets priority at the start of each step (still
  open as G15 — `advance_phase` doesn't reset priority).
- **Cross-engine drift fallout**: Python's
  `apply_action(PassPriority())` is still conceptually a no-op (its
  outer search loop drives the step machine separately), so once
  Rust composes step-advance the engines structurally drift on
  step/phase/turn/active_player. The drift is tracked separately as
  G25 (see below).
- **Impact**: pre-slice — HIGH for real-deck parity (the harness
  couldn't reach MAIN1, so cross-engine parity testing on full decks
  was structurally blocked). Post-slice — LOW; the residual gap is
  the still-deferred non-empty-stack branch.

### G4. `cast_creature` / `activate_ability` do not touch priority

- **Pinned by**: `tests_rust/test_priority.py::test_cast_spell_does_not_change_priority_player`,
  `test_activate_mana_ability_does_not_change_priority_player`.
- **Rust today**: Casting a creature or activating a (mana) ability
  leaves `priority_player` unchanged.
- **CR / Python expected**: CR 117.1c — the active player gets
  priority after a spell is cast or an ability resolves.
- **Impact**: low-to-medium. As with G3, the harness composes via
  higher-level primitives; pure cast/activate behavior is observable
  only when the harness inspects priority directly.
- **Closeable**: yes — same slice as G3.

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

### G6. `ActivateAbility` does not check summoning sickness (CR 302.1)

- **Pinned by**: `tests_rust/test_tap.py::test_activate_ability_on_summoning_sick_creature_succeeds`.
- **Rust today**: `apply_action({"kind": "activate_ability", ...})`
  on a summoning-sick mana-producer (e.g. just-cast Llanowar Elves)
  taps the creature and credits the controller's mana pool. Both
  the action dispatcher in `state_ops_actions.rs` and the
  `legal_decisions` enumerator in `state_ops_legal.rs` share this
  gap.
- **CR / Python expected**: CR 302.1 — a summoning-sick creature
  cannot tap to pay tap-cost activations. Python rejects the
  activation.
- **Impact**: HIGH for normal gameplay. Rust will produce mana from
  a just-cast Llanowar Elves; Python won't. Easy to hit in any game
  that casts a mana-producer and then tries to use it the same turn.
- **Closeable**: yes — add a `summoning_sick && requires_tap` gate
  to both `apply_action`'s `activate_ability` arm and `legal.py`'s
  activated-ability emitter. Pending slice.

### G7. Cleanup step does not run "until end of turn" effect expiration (CR 514.3)

- **Pinned by**: docstring on `test_cleanup.py` ("CR 514.3 ('until end
  of turn' effects expire) and CR 514.4 (no-trigger repeat cleanup
  step) are deferred").
- **Rust today**: `cleanup_step_for_active_player` models CR 514.1
  (hand-size discard) and CR 514.2 (damage clear) only.
- **CR / Python expected**: CR 514.3 fires "until end of turn"
  expiration; CR 514.4 specifies the no-trigger repeat behavior.
- **Impact**: zero today (no continuous-effect or triggered-ability
  machinery in Rust yet), high once continuous effects land.
- **Closeable**: deferred until continuous-effect framework exists.

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

### G10. `DrawCards` from empty library is a silent no-op (no deck-out loss)

- **Pinned by**: `tests_rust/test_library.py::test_drawcards_from_empty_library_is_silent_no_op_no_loser`,
  `test_drawcards_count_larger_than_library_drains_no_sba_loss`.
- **Rust today**: `Effect::DrawCards` against an empty library
  short-circuits silently per draw. `apply_sbas` does NOT mark the
  player as a loser. There is no `attempted_draw_on_empty` field on
  the Rust `Player` struct.
- **CR / Python expected**: CR 704.5b — a player who attempted to
  draw from an empty library loses on the next SBA pass. Python
  models this.
- **Impact**: HIGH for games that mill out or draw heavily. Rust
  games that would deck-out in Python continue indefinitely.
- **Closeable**: yes — add `attempted_draw_on_empty: bool` to
  `Player`, set it in the `DrawCards` empty-pop arm, check it in
  `losers()` / `apply_sbas`. Pending slice.

### G11. Game-end loss is pure-derivation from `life <= 0`; no `has_lost` field

- **Pinned by**: `tests_rust/test_winloss.py` module docstring;
  `test_gain_life_from_zero_revives_player`,
  `test_each_opponent_loses_life_lethal_then_revive_flips_outcome`;
  `@pytest.mark.skip` on `test_explicit_lose_game_marks_player`.
- **Rust today**: `losers()` is computed purely from current life
  totals (`life <= 0`). There is no `Player.has_lost` field. A
  player at 0 life who gains life is removed from the loser list and
  the game becomes ongoing again.
- **CR / Python expected**: CR 104.3a / SBA-driven permanent loss —
  a player who drops to 0 life loses, and "you lose the game" effects
  (Phage, Bargain, deck-out) flag the player irrevocably. Python
  preserves the loss across subsequent life changes.
- **Impact**: medium. Observable in any game where a player drops
  to 0 and then gains life (uncommon but legal).
- **Closeable**: yes — add `has_lost: bool` to `Player` and switch
  `losers()` to the OR of `has_lost` and `life <= 0`. The skipped
  `test_explicit_lose_game_marks_player` is the API surface pin for
  the future `state.lose_game(p)` primitive.

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

### G14. `advance_phase` does not auto-fire `draw_card` on the DRAW step

- **Pinned by**: `tests_rust/test_steps.py::test_advance_phase_does_not_draw_a_card`.
- **Rust today**: Walking `UNTAP -> UPKEEP -> DRAW` via `advance_phase`
  does NOT call `draw_card`. The DRAW step is a phase tag only.
- **CR / Python expected**: CR 504.1 — the active player draws one
  card during their draw step. Python's turn machinery models this.
- **Impact**: HIGH for any multi-turn game. Without explicit
  `draw_card` calls from the harness, Rust hands stagnate.
- **Closeable**: yes — slice that wires `draw_card` into
  `advance_phase` on entry to `PHASE_DRAW`.

### G15. `advance_phase`'s CLEANUP -> UNTAP wrap does not reset priority

- **Pinned by**: `tests_rust/test_steps.py::test_advance_phase_keeps_priority_player_in_sync_with_active`;
  `tests_rust/test_priority.py::test_priority_unaffected_by_phase_advance_in_this_slice`,
  `test_advance_phase_full_lap_does_not_touch_priority`.
- **Rust today**: When `advance_phase` wraps CLEANUP -> UNTAP and
  flips `active_player`, the `priority_player` field is not touched.
- **CR / Python expected**: CR 117.1a — the new active player gets
  priority at the start of their turn.
- **Impact**: low when harness drives priority explicitly; high if
  smoke compares the `priority_player_idx` field across the
  turn-boundary.
- **Closeable**: yes — same slice as G3 / G4 (priority + steps
  integration).

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

### G20. Surveil first-slice keeps everything on top (no decision callback)

- **Pinned by**: `tests_rust/test_library.py::test_effect_surveil_first_slice_keeps_library_and_top_id_unchanged`
  (docstring note); also tracked in `docs/SEARCH_LIMITATIONS.md`.
- **Rust today**: `Effect::Surveil` deterministically keeps all
  observed cards on top. No "which to graveyard" decision callback.
- **CR / Python expected**: Real surveil lets the controller route
  each observed card to top-of-library or graveyard.
- **Impact**: low — visible whenever a surveil effect resolves;
  Rust never sends to graveyard.
- **Closeable**: yes — pending decision-callback plumbing for surveil
  and scry.

### G21. `mulligan` always draws 7; no London-style hand-shrink

- **Pinned by**: `tests_rust/test_mulligan.py::test_mulligan_keeps_hand_at_seven_pinning_london_simplification`;
  module-level cross-feature docstring (lines 430-453).
- **Rust today**: `mulligan(player, seed)` always leaves the hand at
  size 7. It does not take a mulligan-count parameter and never
  shrinks the hand. A separate `mulligan_to_n(player, n_cards, seed)`
  primitive provides the smaller-hand draw shape.
- **CR / Python expected**: CR 103.4 — each successive mulligan
  draws a smaller hand (7 → 6 → 5 → ...). London mulligan keeps the
  hand at 7 visually but requires bottoming N cards atomically.
- **Impact**: medium — observable whenever a player mulligans and
  the parity harness checks hand size. The bridge currently expects
  the caller to invoke `bottom_cards_from_hand` as a separate step.
- **Closeable**: yes — accepted simplification. A future "real
  London-mulligan transaction" slice consolidates the two-call flow.

### G22. `mulligan` does not enforce the bottom-N step

- **Pinned by**: `tests_rust/test_mulligan.py` module docstring (gap 5).
- **Rust today**: A caller who calls `mulligan` without then calling
  `bottom_cards_from_hand` ends up with too many cards in hand. The
  primitive does not enforce the bottoming.
- **CR / Python expected**: CR 103.4 — the London-mulligan
  transaction is atomic (shuffle + draw 7 + bottom N). Python's
  flow enforces the bottoming.
- **Impact**: low when the harness drives mulligan via composed
  primitives; medium if direct `mulligan` calls leak into the smoke
  harness.
- **Closeable**: yes — bundled with G21 (atomic London transaction).

### G23. No maximum-mulligan cap

- **Pinned by**: `tests_rust/test_mulligan.py` module docstring (gap 3).
- **Rust today**: `mulligan` can be called indefinitely on the same
  state with no failure. There is no upper bound.
- **CR / Python expected**: CR 103.4 — each mulligan draws a smaller
  hand and eventually you'd mulligan to 0; in practice you stop.
  Python doesn't enforce a hard cap either, but it ties the cap to
  the count parameter.
- **Impact**: zero in normal play; high only for adversarial seeds.
- **Closeable**: yes — bundled with G21.

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

### G25. Cross-engine step/phase/turn/active_player drift downstream of G3's CR 117.4 step-advance composition — ALIGNED at harness layer

- **Pinned by**: `tests/test_parity_smoke.py::test_cross_engine_game_mode_with_alignment_emits_step_aligned_rows`,
  `test_cross_engine_game_mode_no_step_aligned_row_when_no_rust_advance`,
  `test_cross_engine_game_mode_first_unknown_index_past_pre_fix_g25_trap`,
  `test_align_python_step_to_rust_bumps_python_when_rust_advances`,
  `test_align_python_step_to_rust_wraps_turn_on_cleanup_advance`,
  `test_align_python_step_to_rust_no_op_when_neither_engine_advances`,
  `test_g25_mask_removed_from_known_gap_masks`. Shim helper lives at
  `scripts/parity_smoke.py:_align_python_step_to_rust` /
  `_advance_python_step_once` / `_rust_global_step_pos` /
  `_python_global_step_pos`.
- **Rust today**: `apply_action({"kind": "pass_priority"})` composes
  CR 117.4 step-advance via the `consecutive_passes` counter (G3,
  closed at the action layer). When both players pass with an empty
  stack, Rust fires `advance_phase()` immediately.
- **CR / Python expected**: Python's
  `apply_action(PassPriority())` is conceptually a no-op (its outer
  search loop drives the step machine separately — `turn.py` walks
  `STEP_ORDER` independently of the action dispatcher). So Python's
  step/phase/turn/active_player stay frozen across the same
  pass_priority dispatch.
- **Cross-engine consequence (pre-shim)**: once Rust fired any
  step-advance, every SUBSEQUENT action observed Rust ahead of
  Python on the step ladder. The drift persisted across non-
  pass_priority actions and the field-diff harness masked it as
  G25 to walk past — but the mask silently swallowed downstream
  divergences too, dulling the smoke's signal.
- **Resolution (harness-level shim)**: `parity_smoke.py`'s
  cross-engine loop now snapshots Rust's pre-apply
  ``turn_number * 12 + step_index`` ("global step position"),
  applies the action on both engines, and if Rust's position
  advanced past Python's, mirrors the same delta onto Python by
  walking `state.step` through `STEP_ORDER` (and bumping
  `turn_number` / rotating `active_player_idx` on the CLEANUP wrap).
  The mirror is a pointer-set only — it does NOT fire the
  turn-based actions (`run_untap_step`, `run_draw_step`, ...)
  those step entries normally drive. That's deliberate: re-firing
  them would re-introduce divergences on the SAME surfaces G14 /
  G15 / G16 already track, and the shim's contract is "align the
  step pointer only" so the downstream gaps stay observable.
  Each alignment emits a `STEP_ALIGNED (+N step)` row in the trace
  so operators see which step the shim closed. `KNOWN_GAP_MASKS`
  no longer carries a G25 entry — alignment failures (rare; shim
  diagnoses them with a `WARN: step-alignment shim could not reach
  Rust's position` log) surface honestly as UNMASKED DIVERGE rows
  instead.
- **Impact**: HIGH for cross-engine parity testing — closed. The
  first-unknown divergence index in the eoe-draft-style smoke is
  unchanged at decision 6 (the real divergence at the first
  cast_spell is downstream of step alignment, not caused by it).
  The Forest-only seed-42 index moved from decision 0 (G3) /
  decision 6 (post-G3-mask) to decision 18 — a real
  `battlefield` divergence (Python lands ETB with summoning_sick
  set, Rust lands don't), a separate gap unrelated to step
  alignment.
- **Closeable**: closed at the harness layer. A future engine-side
  close-out — extending Python's `apply_action(PassPriority())` to
  compose CR 117.4 step-advance like Rust's — would let the shim
  retire entirely; until then the harness shim is the source of
  truth for cross-engine step alignment.

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

## Maintenance

When a slice closes one of the gaps above, drop the entry from the
"Confirmed gaps" table (or move it to invariants if the closing slice
made it parity-equal). When a slice adds a new pinned divergence,
add an entry following the same shape — short name, pinning test,
Rust today, expected, impact, closeable. The 1000-game smoke harness
parses this doc by section header (`### G<n>` and `### I<n>` are the
stable anchors), so keep the numbering monotonic.
