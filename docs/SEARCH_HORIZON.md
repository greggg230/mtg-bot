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

20 games per arm, `v30+b1` (standard) vs `v30+b1+boardonly`, both vs
`heuristic`, nodes 200 / samples 2, git `0415d24d`. The 40 games are
bit-reproducible (a rerun after a census-only fix replayed all 40
identically). Numbers are standard / board-only.

**Budget and depth.**

| | standard | board-only |
|---|---|---|
| agent decision moments per game | 256 | 275 |
| ... forced (1 option, no tree built) | 84.5 % | 85.2 % |
| unforced decisions per game | 39.6 | 40.7 |
| unforced with >= 90 % of root visits on one child ("near-forced") | 11.9 % | 15.0 % |
| nodes per tree (budget 200; 2 trees per decision) | 189 | 188 |
| deepest path, plies (mean) | 29.0 | 35.1 |
| deepest path, real choices (mean / median) | 6.1 / 6 | 6.2 / 6 |
| most-visited path: plies / real choices | 27.2 / 5.7 | 33.2 / 5.8 |
| search wall per unforced decision (6 concurrent games + a 20-worker ladder) | 2.70 s | 2.64 s |

- `--nodes 200 --samples 2` means **two trees of ~190 nodes**, not two
  of 100: `Budget(max_nodes=inner_nodes)` is per determinization.
- Only ~40 decisions per game are real choices; 85 % of the agent's
  decision moments are forced priority passes that never reach the
  search (PIMC's forced-move shortcut).

**How far in game time the trees reach.**

| | standard | board-only |
|---|---|---|
| trees whose deepest node is at least a later phase this turn | 94.9 % | 95.3 % |
| ... at least the next turn | 78.5 % | 81.8 % |
| ... two or more turns ahead | 17.0 % | 36.9 % |
| share of all nodes still in the root's step | 12.6 % | 12.4 % |
| share of all nodes in the next turn or later | 47.3 % | 51.7 % |
| chosen attack: tree contains combat damage | 99.4 % (n=306) | 99.3 % (n=286) |
| chosen block: tree contains combat damage | 98.3 % (n=120) | 100 % (n=140) |
| chosen cast / activation: tree contains its resolution | 100 % (n=452) | 100 % (n=466) |

The worry that ~100 nodes never reach the consequences of an action is
**not** borne out: essentially every attack, block and cast tree
contains the action's resolution, and four in five trees reach the next
turn. The depth is bought with forced moves, though — see below — so
"reaches the next turn" mostly means "follows one thin line there".

**Where the nodes go.**

| share of tree nodes | standard | board-only |
|---|---|---|
| spent at forced (single-option) decision points | 63.3 % | 68.1 % |
| that are a `PassPriority` | 72.4 % | 75.4 % |
| at opponent decision points | 51.0 % | 44.5 % |
| that ARE an action of an imagined (determinized) opponent card | 7.2 % | 0 |
| that sit BELOW such an action | 32.0 % | 0 |
| trees containing at least one imagined opponent-card action | 71.7 % | 0 |

- Two thirds of every tree is priority bookkeeping: a forced chain
  between real choices is typically 2-24 plies (the dumps fold them as
  `(+k forced)`). The deepest line holds ~6 real choices.
- Under sampled determinization **a third of the search conditions on
  an opponent hand that was sampled** (and is almost always not the real
  one); board-only removes that, the opponent's turn collapses into
  forced passes, and the same budget reaches two turns ahead twice as
  often.
- Opponent spells inside the tree are auto-targeted first-legal like
  the agent's: the dumps show imagined opponent removal aimed at the
  opponent's own creature (`OPP: Cast Embrace Oblivion -> Virus
  Beetle(opp)` in a sampled world), i.e. the sampled model also
  under-estimates the opponent's removal when it does imagine it.

**Blunder census (per game; 20 games each).**

| detector | standard | board-only |
|---|---|---|
| self-targeted damage / destroy / exile, opponent target was available | 0.05 (1) | 0.10 (2) |
| ... only own targets were legal | 0.05 (1) | 0.05 (1) |
| chump attack (blocked, died, no trade, no trample) | 0.45 (9) | 0.35 (7) |
| ... and the defender lost no life that combat | 0.10 (2) | 0.15 (3) |
| fizzle (all targets gone at resolution) | 0.10 (2) | 0.05 (1) |
| passed main 2 with a castable creature | 0.45 (9) | 0.40 (8) |
| passed main 2 with another castable sorcery-speed spell | 0.50 (10) | 0.70 (14) |
| agent stack items with targets (all auto-targeted) / of them damage-destroy-exile | 80 / 17 | 75 / 17 |

Gross blunders are rare in absolute terms, but **2-3 of the 17
removal-type stack items per arm (12-18 %) hit the agent's own
permanent**, and the causes are the tactical policy, not search depth:
Nebula Dragon's ETB "3 damage to any target" went to the agent's own
Oreplate Pangolin with opponent creatures available, Faller's Faithful's
"destroy UP TO ONE other target creature" destroyed the agent's own
Hullcarver (zero targets was legal), and board-only cast Gravkill with
its auto-target on its own Faller's Faithful. Targets, modes, X,
"may" and "up to" choices are `DefaultTacticalAgent` first-legal /
always-yes (`SEARCH_LIMITATIONS.md`, expose_resolution_decisions entry);
search can only decline to cast. Chump attacks are mostly 1/1s (Virus
Beetle, Robot tokens) swinging into blockers; only ~0.1 per game with
nothing else connecting.

**Tree dumps** (`analysis_runs/search_horizon_v30/trees/`): an attack
declaration and a block (game 0), a removal cast (game 1, Archenemy's
Charm, in an already-lost position, Q ~ -0.8), and an instant-speed response
(game 1, Divert Disaster countering Dockworker Drone), each for both
arms: board, PIMC-aggregated root (`choose_with_subtree`, depth 3,
top 3) and both determinized worlds' trees (5 levels, top 3, forced
chains folded, Q from the actor's view).

## A/B

(filled in from `ladder_runs/s2_v30_boardonly_n200`)
