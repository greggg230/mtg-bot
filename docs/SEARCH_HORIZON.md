# Search horizon, blunder census and the board-only opponent model

2026-10-05, branch `feature/s2-boardonly-search` (off `s2/eoe-training`
5d3d871a). Question: at the s2 production search budget (NN-PIMC,
nodes 200, samples 2, leaf batch 1), how far into the game does the
search actually look, where do its nodes go, and does it make gross
blunders? And: does planning against the *public board only* (not
imagining the opponent's hidden cards) play better than sampling a
concrete opponent hand?

Results: §Findings and §A/B below. Raw outputs:
`analysis_runs/search_horizon_v30/` in the main checkout
(`summary.json`, `report.md`, `decisions.jsonl.gz`, `trees/*.txt`).

## What was built

* **Board-only opponent model** — `opponent_hidden_model="board_only"`
  on `PimcConfig` / `NNPimcConfig` (and the `heurvalue` hybrid);
  ladder label modifier `+boardonly`; `mtg_bot capture
  --opponent-hidden-model board_only`. In each PIMC world every
  opponent hand / library card the searcher can't identify becomes
  the inert `"<hidden card>"` placeholder
  (`mtg_bot.cards.hidden`): never castable / playable / activatable,
  still a card for hand size, discard, mill. The searcher's own
  library is sampled as before. Default `"sampled"` is bit-identical
  to the old search. Limits: `docs/SEARCH_LIMITATIONS.md` ("Board-only
  opponent model").
* **`+s<K>`** ladder modifier: K determinizations sharing the ladder's
  TOTAL node budget (`--nodes` × `--samples`), i.e. `inner_nodes =
  nodes·samples/K`. `v30+b1+boardonly+s1` at `--nodes 200 --samples
  2` is ONE world with 400 nodes. (Note: `--nodes` is the budget PER
  determinization — production "nodes 200, samples 2" is 2 trees of
  200 nodes, 400 node expansions per decision, not 2 × 100.)
* **`MCTS.tree_observer`** — opt-in hook, `None` by default (one
  attribute check per search). When set, every `choose_*` call that
  builds a tree hands the finished root `_Node` to the observer.
* **`mtg_bot.search.horizon_probe`** — `summarize_tree` (post-hoc tree
  statistics) and `render_tree` (readable dumps).
* **`mtg_bot.search.blunder_census`** — four narrow detectors over a
  real game.
* **`mtg_bot.search.horizon_report`** — the driver (plays instrumented
  games, aggregates, writes the report). Command used:

  ```
  PYTHONPATH=src python -m mtg_bot.search.horizon_report \
      --agents v30+b1 v30+b1+boardonly --opponent heuristic \
      --registry-root <main>/trained_weights_s2 \
      --deck-pool <main>/data/17lands/eoe_decks.json.gz \
      --games 20 --nodes 200 --samples 2 --seed 0 \
      --out <main>/analysis_runs/search_horizon_v30
  ```

## Method

`v30+b1` (standard) and `v30+b1+boardonly` each play 20 games against
the fixed `heuristic` PIMC (sampled, nodes 200, samples 2) on the
**same deals**: game `g` uses the ladder's seeds, decks and seat order
for pair `heuristic:v30+b1` (`run_match_parallel(alternate_seats=True)`,
seed 0), so the standard arm's games are the first 20 games of the
`ladder_runs/s2_v30_diag_n200` baseline pairing. Only the v30 seat is
instrumented. Every v30 decision moment is recorded (including forced
ones, which build no tree); every search tree (2 per unforced
decision) is summarised.

### Definitions

* **node** — a tree node below the root. Each MCTS iteration creates
  at most one, so nodes ≈ budget actually spent.
* **ply / depth** — edges from the root. Every engine decision is a
  ply, including forced `PassPriority` moments (the adapter surfaces
  each priority window). **Real choices** = plies whose decision
  point had > 1 legal option.
* **forced** decision — exactly one legal option. **Near-forced** —
  unforced, but the most-visited root child took ≥ 90 % of the visits.
* **decision context** of a real decision: `attack_declaration`
  (factored per-attacker yes/no), `block` (per-blocker), `main_phase`
  (own main phase, empty stack: sorcery-speed window),
  `response_window` (something on the stack), `instant_window` (empty
  stack, not own main phase), `resolution` (never in production play:
  resolution decisions are not exposed; targets / modes / X / discards
  are chosen by `DefaultTacticalAgent`, first legal — see
  `SEARCH_LIMITATIONS.md` "expose_resolution_decisions"). Mulligans are
  also not search decisions (`heuristic_mulligan_decision`).
* **game-time reach** of a node vs the root (engine `turn_number`,
  `step`): `same_step` / `later_step` (same phase) / `later_phase`
  (same turn) / `next_turn` / `two_plus_turns`. Phases: beginning,
  main 1, combat, main 2, ending.
* **node spent at** a decision point = the node's parent. Shares
  reported: at forced points; edges that are a pass; at opponent
  decision points; **imagined-card** edges (an action of an opponent
  card that was hidden at the real root: cast / land / activation /
  attack / block by an object whose identity the determinizer
  sampled) and nodes **under** such an edge.
* **resolution reached** (root child): attack / block → some node in
  that child's subtree is at or past the root turn's combat-damage
  step (or a later turn / game over); a stack-using action (cast,
  non-mana activation) → some node in the subtree has a stack no
  taller than before the action; land / mana ability → immediate.

### Blunder detectors (`mtg_bot.search.blunder_census`)

| detector | definition | false-positive risk |
|---|---|---|
| `self_target_avoidable` | single-target damage / destroy / exile-permanent spell or ability the agent controls, aimed at its own permanent or own face, while an opponent-side target was also legal | modal spells whose chosen mode isn't the harmful one; blink effects (excluded by class name where recognisable) |
| `self_target_forced` | same, but only own targets were legal (casting it at all was the error; for triggers it isn't a choice) | same |
| `chump_attack` | agent attacker blocked, left the battlefield that combat, no blocker left the battlefield (no trade), no trample | alpha strikes where the rest of the team got through; removal mid-combat counts as "died to block" |
| `chump_attack_no_other_damage` | `chump_attack` and the defender lost no life that combat | removal mid-combat |
| `fizzle` | agent spell / ability whose targets are all battlefield objects, all gone when it resolved, not countered (counterspells' stack targets out of scope) | none for "did nothing"; under-counts pointless-but-legal casts and countered-out counterspells |
| `missed_play_creature` / `_other` | agent passed its own main 2 with an empty stack while a sorcery-speed `CastSpell` was legal (creature / other); once per turn | sandbagging; `_other` over-counts removal held because its only legal targets were own creatures (correct to hold) |

## Findings

(filled in from `analysis_runs/search_horizon_v30/summary.json`)

## A/B

(filled in from `ladder_runs/s2_v30_boardonly_n200`)
