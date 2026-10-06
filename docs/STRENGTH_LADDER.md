# Strength ladder (`mtg_bot ladder`)

**This file is the project's single absolute-strength table.** Every
other eval in the repo is relative ("new vs current champion" in the
`iterate` acceptance gate, `mtg_bot eval --new X --old Y`). Those tell
you whether a candidate beat its predecessor; they never tell you how
strong anything is on a fixed scale, and before this tool the
production NN lineage had never been played against the two fixed
yardsticks the project ships (`random`, `heuristic`). Paste ladder
results here as they are produced so there is exactly one place to
look.

## What it does

`mtg_bot ladder` is a **local** (`multiprocessing.Pool`, no
coordinator / worker daemons / runs.db) round-robin between any mix of:

- `random` — uniformly random legal decision (`RandomChoice`), the
  Elo anchor (fixed at 0).
- `heuristic` — PIMC with the default `HeuristicConfig()` evaluator at
  the ladder's `--nodes` / `--samples` (the same opponent
  `iterate._eval_match` falls back to when there is no old NN).
- any registered version label under `trained_weights/` (`v5`,
  `v13s`, ...). Architecture, trunk shape and action-encoding version
  come from the version's `metadata.json` via `Registry`, exactly as
  the acceptance gate threads them into `NNPimcConfig`; the
  `structured` architecture's trunk hparams are discovered from the
  sibling metadata file inside each worker.

Each pair plays `--games` seat-alternated games through the same
`run_match_parallel(alternate_seats=True)` path the gate uses, so seat
bias cancels. Every pair uses the same base `--seed`, i.e. the same
sequence of deck matchups (common random numbers across pairs).

Outputs (in `--out`, default `ladder_runs/<UTC timestamp>/`):

- `results.json` — config + every completed pair (`wins_a`, `wins_b`,
  `draws`, `timeouts`, `crashed`, `mean_decisions`, `wall_seconds`,
  `seed`). Written atomically after **each** pair, so a killed run
  loses at most the pair in flight. `--resume <dir>` skips pairs
  already there (in either orientation) and rebuilds the report.
- `ladder.md` — the Elo table with bootstrap 95% CIs, the pairwise
  win-rate matrix with Wilson 95% CIs, per-pair detail, and the config
  (nodes, samples, games, seed, git commit). Also printed to stdout.

## Running it

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli ladder \
    --agents random heuristic v5 v13s \
    --games 500 --nodes 200 --samples 4 --workers 30 --seed 0
```

Useful flags:

| flag | meaning |
|---|---|
| `--agents A B ...` | participants (required unless `--resume`) |
| `--pairs all` / `--pairs a:b,c:d` | round robin (default) or explicit pairs |
| `--games N` | games per pair, seat-alternated (default 500) |
| `--nodes` / `--samples` | search depth, shared defaults from `mtg_bot.cli._common` (200 / 4) |
| `--workers N` | pool size (default `cpu_count - 2`) |
| `--seed S` | base seed (deck sampling + strategy RNGs) |
| `--per-game-timeout-sec` | default `3 * nodes` seconds, as every other subcommand |
| `--registry-root` | default `trained_weights` |
| `--out DIR` / `--resume DIR` | where results go / continue an earlier run |
| `--timeouts-as-draws` | see "Scoring" below |
| `--bootstrap-resamples` | Elo CI resamples (default 200) |

A quick sanity run (a minute or two):

```bash
... ladder --agents random heuristic v13s --games 4 --nodes 8 --samples 1 --workers 2
```

## Scoring

- **Default**: a timed-out or crashed game contributes no signal and
  is dropped from the denominator. This matches the acceptance gate
  (`MatchStats.win_rate`).
- **`--timeouts-as-draws`**: a timed-out game is scored 0.5 / 0.5. A
  wall-clock timeout cannot be attributed to either player, so there
  is deliberately no "timeouts as losses" mode. Crashed games are
  always excluded (engine bugs, not play).
- Draws (decision-cap games) are always 0.5 / 0.5.

Timeouts and crashes are still counted and shown in the matrix
(`t<N>` / `c<N>`) so a pair that lost half its games to timeouts is
visibly untrustworthy.

## Elo

Bradley-Terry maximum likelihood (Hunter's MM iteration) on the
logistic scale, `P(a beats b) = 1 / (1 + 10^((r_b - r_a) / 400))`,
draws as half a win each way. `random` is anchored at 0 (the first
listed agent if `random` is absent). One virtual draw per played pair
keeps an agent that never scores at a finite rating; at ladder game
counts the shift is well under a point. The 95% CI is a percentile
bootstrap: within each pair the scored games are resampled with
replacement, the fit is redone, and the 2.5 / 97.5 percentiles are
reported. If the pairing graph is disconnected the report says so —
ratings across components are not comparable.

Numbers are only comparable within one depth setting: an Elo measured
at `--nodes 8` is not the same scale as one at `--nodes 200`. Record
the config with every table below.

## Results

Paste each run's `ladder.md` Elo table + config here, newest first.

### 2026-09-25 -- v8, the heuristic warm-start net (nodes 100 / samples 2)

`ladder_runs/s2_v8/`, 200 games/pair. v8 = structured net trained on 6,000
heuristic-PIMC self-play games (nodes 400 / samples 2) with post-hoc value
calibration (see docs/SCALING_DIAGNOSTICS.md D4/D5).

| rank | agent | Elo | 95% CI | games | score |
|---:|---|---:|---|---:|---:|
| 1 | heuristic | +470 | [+397, +572] | 199 | 76.4% |
| 2 | **v8** | **+267** | [+213, +343] | 399 | 53.1% |
| 3 | random | +0 | -- | 200 | 17.5% |

Pairwise: heuristic vs v8 152-47 (v8 23.6%); random vs v8 35-165 (v8 82.5%).
For comparison the previous best NN (v5, self-play from random init) LOST
to random 68-132 and took 9/200 from the heuristic. Baseline from 09-23:
heuristic +407, v13s +30, random 0 (different game sample; Elo anchored
to random each run, so compare within a run).

### 2026-09-23 -- first absolute-strength table (nodes 100 / samples 2)

Commit `5c1bc306`, `ladder_runs/prod_n100s2/` (full report snapshot:
`docs/STRENGTH_LADDER_2026-09-23_n100s2.md`). 200 games/pair, seat
alternated, 20 workers, 900 s per-game timeout, seed 0. Depth is HALF
the production self-play budget (nodes 200 / samples 4 was attempted
first as `prod_2026-09-23/` but ~1 in 10 v13s games hit the 600 s
timeout, biasing the sample, so it was killed after the
random-vs-heuristic pair: 29-467).

| rank | agent | Elo | 95% CI | games | score |
|---:|---|---:|---|---:|---:|
| 1 | heuristic | +407 | [+342, +469] | 399 | 90.7% |
| 2 | v13s | +30 | [-12, +74] | 398 | 32.2% |
| 3 | random | +0 | [+0, +0] | 399 | 27.1% |

Pairwise (row's W-L): random vs heuristic 18-182; random vs v13s
90-109 (t1); heuristic vs v13s 19-180 (c1).

**Reading:** the production champion `v13s` (end of the
schema-1 lineage, ~7M training rows, ~4 months of training) is
statistically indistinguishable from random (CI includes 0) and loses
90% of games to the hand-written heuristic evaluator at equal search
budget. The whole pre-2026-09 lineage carried no measurable learning
signal; see the 2026-09-22 worklog entry for the root causes that
`feature/learning-fixes` addressed. Every future snapshot goes on this
ladder against the same three references.


Tool validation only (NOT a result): 2026-09-23, commit 779a87b4,
`--agents random heuristic v13s --games 4 --nodes 8 --samples 1
--workers 2` loaded all three participants (v13s = structured /
canonical_v2 via the registry) and played 12 games in ~20s with 0
timeouts / 0 crashes. At that depth the ordering is noise; a 16-game
`random` vs `heuristic` check at `--nodes 32` gave heuristic 12-4.

### 2026-09-25 -- v9, v10, v11: three always-promote AlphaZero cycles from v8 (nodes 100 / samples 2)

`ladder_runs/s2_v9/`, `s2_v10/`, `s2_v11/`, 200 games/pair each. Each cycle:
1,500 NN-PIMC self-play games driven by the previous version, trained on the
newest 6,000 shards (heuristic corpus displaced progressively), LR 3e-4,
always-promote. Cycles 7-8 (v9, v10) also contained ~40% games from the Mac
worker running pre-calibration code (quarantined afterwards); cycle 9 (v11)
is desktop-only and clean.

| version | Elo | 95% CI | vs heuristic | vs random | held-out value corr (calibrated) |
|---|---:|---|---|---|---:|
| v8 (warm start) | +267 | [+213, +343] | 47-152 (23.6%) | 165-35 (82.5%) | 0.35 |
| v9 | +199 | [+150, +262] | 45-154 (22.6%) | 152-48 (76.0%) | 0.29 |
| v10 | +185 | [+126, +239] | 49-151 (24.5%) | 149-51 (74.5%) | 0.26 |
| v11 | +150 | [+103, +199] | 36-164 (18.0%) | 141-59 (70.5%) | 0.30 |

Reading: monotone Elo decay as NN-play games replace heuristic games in the
window. v11's trainer early-stopped at epoch 6 with best = epoch 1, and its
calibrated corr (0.30) still does not beat the trivial life-differential
baseline (0.34) -- the value head learns nothing beyond "who is ahead on
life" from NN-play outcomes. Response: `current_best` reset to v8, the
3,345 NN-play shards from cycles 7-9 moved to `iterate_data_s2_nnplay_v8_v10/`,
and the schedule switched to `scripts/s2/run_mixed_cycles.sh` (1,500
heuristic games + 1,500 v8-driven NN games per cycle, gated promotion
`S2_ACCEPT=0.55`) so the window stays heuristic-majority.

### 2026-09-26 -- v12 (mixed cycle) and v13 (heuristic-only, 2x data)

| version | training window | Elo | vs heuristic | vs random | held-out value corr (calibrated / life-diff baseline) |
|---|---|---:|---|---|---:|
| v12 | 4,500 heuristic + 1,500 v8-driven NN games; rejected by the 55% arena vs v8 (136-161) | +181 | 45-153 (22.7%) | 148-52 (74.0%) | 0.316 / 0.336 |
| v13 | **12,207 heuristic games only** (no NN-play data), store on E: | +199 | **53-147 (26.5%)** | 152-48 (76.0%) | **0.323 / 0.299** |

v13 is the first net whose value head beats the trivial life-differential
baseline on held-out rows, and the best result vs heuristic so far -- but
its vs-random score is below v8's (165-35), and single 200-game pairs put
v8 and v13 well inside each other's CIs (the heuristic anchor itself
bounces +375..+470 across runs). Decisive comparison: one ladder run with
`v8:v13`, `heuristic:v8`, `heuristic:v13` at 300 games/pair
(`ladder_runs/s2_v8_vs_v13/`, results below when done).

**Head-to-head (2026-09-26, `ladder_runs/s2_v8_vs_v13/`, 300 games/pair, one run, heuristic anchored at 0):**

| pair | result |
|---|---|
| v8 vs v13 | 146-153-1 (dead even) |
| heuristic vs v8 | 237-61-1 (v8 20.5%) |
| heuristic vs v13 | 224-75-1 (v13 25.1%) |

Elo: heuristic 0, v13 -202 [-245, -167], v8 -219 [-260, -187]. Doubling the
imitation corpus (6k -> 12k games) bought ~+17 Elo, inside the CIs. v13 is
`current_best` (tie-break: better value head, better vs heuristic), but the
imitation route is at a plateau roughly 200 Elo below the teacher.

### 2026-09-26 -- D6 value-target A/B: v14 (outcome) vs v15 (search root value), same corpus

Both trained on the same 6,000 dense root-value heuristic games (batches
17-20; `docs/SCALING_DIAGNOSTICS.md` D6). Only `--value-target` differs.

| version | target | Elo | 95% CI | vs heuristic | vs random | held-out corr vs outcome |
|---|---|---:|---|---|---|---:|
| v14 | outcome | +267 | [+213, +335] | 49-151 (24.5%) | 165-35 (82.5%) | 0.333 |
| **v15** | **root** | **+378** | [+313, +472] | **86-112 (43.4%)** | **180-20 (90.0%)** | **0.449** |
| v16 | blend:0.5 | +320 | [+261, +398] | 74-125 (37.2%) | 173-27 (86.5%) | 0.397 |

(heuristic anchored at +462 / +423 / +411 in the three runs.) v15 is the first
net inside ~50 Elo of the teacher; every earlier version sat ~200 below.
Decisive head-to-head vs v13 (300 games/pair) in `ladder_runs/s2_v13_vs_v15/`.

### 2026-09-27 -- root-value AlphaZero cycles 11-13 from v15 (`--value-target root`)

Self-play driven by v15 records the NN-PIMC root value per row; training
regresses it on the newest 6,000 shards (NN-play share grows each cycle:
25 % -> 50 % -> 58 %). Arena = 300 games vs v15 at the cycle's search
depth; ladder at nodes 100 / samples 2, 200 games per pair.

| cycle | self-play | version | held-out corr | arena vs v15 | ladder vs heuristic | vs random | Elo (heuristic same run) |
|---|---|---|---:|---|---|---|---:|
| 11 | 1,500 @ nodes 100 | v17 | 0.446 | 148-150 (49.7 %) | _ladder crashed (CPython 3.13 pool fault)_ | | |
| 12 | 1,500 @ nodes 100 | v18 | 0.441 | 156-144 (52.0 %) | **95-104 (47.7 %)** | 182-18 (91.0 %) | +398 (heuristic +413) |
| 13 | 500 @ nodes 200 | v19 | 0.436 | 151-148 (50.5 %) | 83-116 (41.7 %) | 182-18 (91.0 %) | +398 (heuristic +455) |

Reading: with root-value targets the self-play cycles no longer degrade
the net (the whole v9-v11 story was monotone decay), the value head stays
at corr ~0.44 as NN-play data displaces heuristic data, and v18 posts the
best ladder yet -- but no cycle clears the promotion gate in a 300-game
head-to-head. The gains per cycle (if real) are below what 300 games can
resolve. v18 promoted to `current_best` on the ladder evidence; the next
run switches to always-promote (accept 0.0) at nodes 200 / 1,000 games
per cycle with the ladder as the yardstick.

### 2026-09-28 -- v21-v25: deep-teacher warm start (v22) and blend:0.8 always-promote cycles

"Do both" round (2026-09-27/28): (a) a heuristic corpus at **nodes 1600**
(batches 21-23: 1,415 desktop+Mac games, 101 + 1,484 AWS games on 2 and
14 c7i.4xlarge spot instances) and a warm start on the newest 3,000 of
those shards with `--value-target root` = **v22**; (b) three
always-promote self-play cycles from v22 with `--value-target blend:0.8`
(nodes 200 / 1,000 games, the last two desktop-only because the Mac went
offline). v21 is root-value cycle 15 from v20. Ladder at nodes 100 /
samples 2, 200 games per pair.

| version | how | held-out corr | ladder vs heuristic | vs random | Elo [95 % CI] |
|---|---|---:|---|---|---:|
| v21 | root cycle 15 from v20 | 0.403 | 88-111 (44.2 %) | 183-17 (91.5 %) | +408 [+335, +504] |
| **v22** | deep-teacher warm start, 3,000 games @ nodes 1600, root target | 0.451 | **93-106 (46.7 %)** | 187-13 (93.5 %) | +457 [+378, +565] |
| v23 | blend:0.8 cycle 16 from v22 | 0.446 | 80-119 (40.2 %) | 188-12 (94.0 %) | +471 [+388, +591] |
| v24 | blend:0.8 cycle 17 from v23 | 0.462 | 86-113 (43.2 %) | 180-20 (90.0 %) | +378 [+313, +472] |
| v25 | blend:0.8 cycle 18 from v24 | 0.479 | 71-128 (35.7 %) | 183-17 (91.5 %) | +408 [+343, +522] |

Reading: the 4x deeper teacher did not move the value head (corr 0.451
vs 0.449 for v15's nodes-400 corpus) and v22 lands where v18 did
(46.7 vs 47.7 %). The always-promote blend cycles show the old pattern
in a new form: held-out corr climbs every cycle (0.446 -> 0.479, the
highest in the lineage) while ladder strength drifts down to 35.7 %.
A better fit to the driver's own root values is not a stronger player.
`current_best` was reverted to v22 by hand. Since v15 every version sits
in a 36-48 % band vs heuristic; neither value-target choice nor teacher
depth has cleared 50 %.

### 2026-10-01 -- head isolation, then the search-batching control (nodes 100 / samples 2, 200 games per pair)

Which head loses? Two agent variants (`mtg_bot.search.hybrid_evaluator`,
ladder labels `<v>+valueonly` / `<v>+heurvalue`) split v22:

| agent | search | vs heuristic | vs random |
|---|---|---|---|
| v22 (plain) | PUCT on NN priors, NN value, batch 32 | 93-106 (46.7 %) | 93.5 % |
| v22+valueonly | UCB1, NN value, batch 32 | 93-107 (46.5 %) | 91.0 % |
| v22+heurvalue | PUCT on NN priors, heuristic value, batch 32 | 89-110 (44.7 %) | 92.0 % |

Neither head explains the gap and the effects do not add: three
different head combinations land at the same ~46 %. The one thing
they share is the search: every NN-side agent in this lineage runs
MCTS with ``batch_size=32`` (batched leaf evaluation with virtual
loss; ~3 waves at nodes 100) while the heuristic opponent runs
``batch_size=1``. Controls with the new ``+b<N>`` label modifier:

| agent | vs heuristic (batch 1) |
|---|---|
| heuristic+b32 (same evaluator, batched search) | **83-115 (41.9 %)** |
| v22+b1 | **104-96 (52.0 %)** |
| v22+valueonly+b1 | 103-96 (51.8 %) |

Reading: batching alone costs the *heuristic* 8 points at nodes 100,
and at equal search quality v22 beats the heuristic. The 36-48 % band
every version sat in since v15 was a search handicap, not a learning
ceiling. Production self-play (`worker_daemon_production_runner`),
`iterate`'s arenas (NN seats batch 32, heuristic seat batch 1) and
this ladder (`resolve_agent` hardcoded 32) all carried it; NN-vs-NN
arenas were fair, which is why cycle gating never exposed it. All
earlier "vs heuristic" rows in this file are biased low by roughly
5-8 points; a batch-1 re-ladder of v15/v18/v25 is in progress.

**Batch-1 re-ladder of the lineage and the depth sweep (same day):**

| agent | nodes | vs heuristic (batch 1) |
|---|---:|---|
| v15+b1 | 100 | 102-98 (51.0 %) |
| v18+b1 | 100 | 98-101 (49.2 %) |
| v22+b1 | 100 | 104-96 (52.0 %) |
| v25+b1 | 100 | 95-104 (47.7 %) |
| heuristic+b32 | 200 | 83-117 (41.5 %) |
| v22 (b32) | 200 | 101-98 (50.8 %) |
| v22+b1 | 200 | 106-93 (53.3 %) |
| heuristic+b32 | 400 | 93-106 (46.7 %) |

Reading: at equal search every version since v15 sits at parity with
the heuristic (48-52 %, inside the 200-game CI) -- the plateau is real
but it is at 50 %, not 36-48 %, and the 12-point v15->v25 "decay" the
batch-32 ladders showed shrinks to ~4. The handicap is ~8.5 points at
the self-play depth (nodes 200) and ~3 at 400, and batch 1 costs ~3 %
wall under the 16-worker production shape, so the production default
is now batch 1 (`mtg_bot.search.leaf_batch`;
`docs/BENCHMARKING_FINDINGS.md` §6c). The self-play data of every
cycle so far was generated by the handicapped searcher.

### 2026-10-05 -- board-only opponent model and one-world search (nodes 200 / samples 2, 300 games per pair)

`feature/s2-boardonly-search`; label modifiers `+boardonly` (the
searcher replaces the opponent's hidden cards with inert placeholders
instead of sampling a hand) and `+s<K>` (K worlds sharing the
`nodes x samples` budget). Same deals (seed 0) as the baseline pairing
in `ladder_runs/s2_v30_diag_n200`.

| agent | vs heuristic |
|---|---|
| v30+b1 (baseline) | 158-140 (53.0 %, [47.4, 58.6]) |
| v30+b1+boardonly | 156-142 (52.3 %, [46.7, 58.0]) |
| v30+b1+boardonly+s1 (1 world x 400 nodes) | 164-133 (55.2 %, [49.5, 60.8]) |

Reading: no measurable effect either way (all inside the 300-game CI);
the board-only searcher is not hurt by never seeing an opposing trick.
Horizon / blunder measurements and the method: `docs/SEARCH_HORIZON.md`.
