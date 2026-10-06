# Project rules for Claude

## Development workflow

- **All work happens in a git worktree, not in the main checkout.** The main checkout at `mtg-bot/` stays on `main` and is treated as read-only context for cross-branch comparison. Create a worktree per feature with:
  `git worktree add ../mtg-bot-worktrees/<branch-name> -b feature/<branch-name> main`
  Worktrees live in the sibling directory `mtg-bot-worktrees/`. Remove them after merge with `git worktree remove ../mtg-bot-worktrees/<branch-name>`.
- **Every new feature must be developed on its own branch off `main`.** Do not commit feature work directly to `main`.
- Branch naming: `feature/<short-name>` (e.g. `feature/flying-keyword`, `feature/etb-triggers`).
- **Commit as you go.** Each unit of work is its own commit on the feature branch. Do not batch everything into one monster commit.
- **Tests must pass before every commit.** Run the full test suite; if anything fails, fix it before committing.
- **Do not merge to `main` yourself.** Push the branch for review and wait for the user to approve. Only after explicit approval, merge into `main`.

## Perf monitoring (informational)

- One-time per clone: `git config core.hooksPath hooks`. This points
  git at the version-controlled `hooks/` directory.
- After every commit on `main`, `hooks/post-commit` runs `scripts/bench_clone.py`
  and appends the per-call clone time to `.perf_history.log` (gitignored). It
  also prints the new measurement and the % delta vs the previous logged commit.
- Feature-branch commits are silently skipped (no bench, no log entry).
- Never blocks: if the bench fails or the venv is missing, the hook prints
  to stderr and exits 0 so the commit still succeeds.
- Read the log with `tail .perf_history.log` to see the last several commits'
  perf numbers in context.

## Testing

- Use `pytest`. Tests live under `tests/`.
- **Target coverage: high 90s (≥95%).** Run `pytest --cov=mtg_bot` to check. Before declaring a feature complete, verify coverage has not regressed.
- Every new module needs its own `test_<module>.py`. Cover the happy path, meaningful edge cases, and error paths (invalid inputs, exceptional states).

## Running things

- Virtual env is at `.venv/` in the **main checkout** (`mtg-bot/.venv/`). Worktrees share it. Use `mtg-bot/.venv/Scripts/python.exe` on Windows / `mtg-bot/.venv/bin/python` on macOS-Linux.
- **Recommended interpreter for tests is Python 3.13** (or any 3.12+).
  The shared 3.11 venv hits an intermittent Windows access-violation
  segfault inside coverage.py's C-trace backend during the simulation
  sweep tests. Python 3.12+ has PEP 669 `sys.monitoring`; coverage.py
  auto-uses the `sysmon` backend (already pinned in `pyproject.toml`),
  and the crash does not reproduce. See `docs/segfault-fix.md` for the
  full diagnosis.
  Easiest path: create a parallel 3.13 venv per worktree:
  `py -3.13 -m venv .venv-py313 && .venv-py313/Scripts/python.exe -m pip install -e ".[dev]"`
  then run `PYTHONPATH=src .venv-py313/Scripts/python.exe -m pytest`.
  `.venv-*/` is gitignored.
- **From a worktree, always set `PYTHONPATH=src` when invoking python**, otherwise the editable install in the shared venv resolves `mtg_bot` to the main checkout's source and your worktree changes won't be exercised. Example:
  `PYTHONPATH=src ../../mtg-bot/.venv/Scripts/python.exe -m pytest`
  (Or, on Windows cmd: `set PYTHONPATH=src && ...`.)
- Run tests: `PYTHONPATH=src ../../mtg-bot/.venv/Scripts/python.exe -m pytest` from inside a worktree, or `.venv/Scripts/python.exe -m pytest` from the main checkout.
  (If you're on the 3.11 venv and pytest exits with an access violation,
  switch to the 3.13 venv per the previous bullet — that's the fix.
  See `docs/segfault-fix.md`.)
- **Inner-loop dev iteration: `pytest -n 8 --no-cov`** — ~60s for the
  full suite vs ~270s with the default sequential+coverage (~4.3×
  speedup, 7280 pass / 9 skip / 0 flake observed 2026-05-05 on
  Win11+3.13). Use this while iterating; the default
  with-coverage run remains the gate before commit (coverage must
  stay ≥95%). xdist + sysmon are already dev deps; no setup
  needed. The two contributions are roughly multiplicative:
  `-n 8` parallelizes across 8 worker processes (~3× alone),
  `--no-cov` skips sysmon instrumentation (~3× alone).
- Tests marked `@pytest.mark.isolated` (currently `tests/test_perf_clone.py`)
  are skipped by the default `pytest` invocation — they're sensitive to
  accumulated interpreter state from preceding tests and only run reliably
  alone. Run them with `scripts/run_isolated_tests.py` (own subprocess, no
  coverage), or directly: `pytest -m isolated --override-ini=addopts= --no-cov`.
- Run browser-driven E2E tests (opt-in, skipped by default):
  `MTG_BOT_E2E=1 .venv/Scripts/python.exe -m pytest tests/test_e2e.py`
  First time only: `.venv/Scripts/python.exe -m pip install playwright`
  then `.venv/Scripts/python.exe -m playwright install chromium`.
- Run JS unit tests for the static UI bundle (Vitest + jsdom):
  `npm test`
  First time only: `npm install` (requires Node 18+).
  Tests live under `tests-js/`. They load `src/mtg_bot/static/app.js`
  into a jsdom fixture and call its globals; see `tests-js/setup.js`.
- **GPU available** — `.venv-py313/` has `torch 2.11.0+cu128` (CUDA 12.8 build) installed against an RTX 5090. `torch.cuda.is_available()` returns True; `auto_device()` in `src/mtg_bot/training/trainer.py` picks `cuda` automatically when training. Reinstall the CUDA build with:
  `.venv-py313/Scripts/python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128`
  Smoke check:
  `.venv-py313/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  **Production network (2026-05-17): the small structured trunk** —
  `emb_dim 128, n_layers 2, hidden_dim 256`, v-chain `v1`–`v10`
  (`current_best: v5`, `architecture: structured`) — **not** the
  legacy 1024-1024-512 `_ValuePolicyMLP` (the old `v13`/`v15`/`v16`
  are archived at `trained_weights/archived_pre_structured/`).
  `auto_device()` (in `trainer.py`) still picks `cuda`, falling back
  to `mps` then `cpu`. **The v2 cutover deleted the `run_shard_*`
  SSH path** — production self-play now runs through
  `worker_daemon_production_runner._build_nn_pimc_config`, which
  **hardcodes `device=None` (CPU) by design**; GPU eval is **opt-in
  per worker** via `--nn-device {auto,cpu,cuda}` (default `cpu`,
  byte-identical to prior behavior — `feature/worker-daemon-gpu-device`;
  see [`deploy/README.md`](deploy/README.md)). `NNPimcConfig.device`
  default stays `None` so direct test construction stays CPU —
  required because tests spawn multiprocessing pools and per-worker
  CUDA-context init under Windows-spawn semantics deadlocks.
  **Validated device numbers live in
  [`docs/BENCHMARKING_FINDINGS.md`](docs/BENCHMARKING_FINDINGS.md)
  §4/§2A** (per-process `cuda` ≈ ~2× `cpu` per NN-eval even at this
  small structured trunk — measured, depth-invariant; do not cite a
  multiplier that is not a dated row in that doc). The earlier
  "1024-trunk / GPU dominates on every host" and "workers stay
  CPU-only at 25–65k params" framings are both **stale** —
  superseded by the structured cutover and the §4/§2A measurements.
- Run a game: `.venv/Scripts/python.exe -m mtg_bot`
- Run the UI (localhost): `.venv/Scripts/python.exe -m mtg_bot.ui`
- Run the UI over Tailscale (reachable from phone/laptop on the tailnet):
  `powershell -ExecutionPolicy Bypass -File scripts\serve.ps1`
  First-time firewall rule setup needs an elevated PowerShell.
- Warm the card-image cache (optional, for nicer first-load):
  `.venv/Scripts/python.exe scripts/fetch_card_images.py`
- Ad-hoc screenshot the live UI:
  `.venv/Scripts/python.exe scripts/screenshot_ui.py --url http://127.0.0.1:5002 --out ui.png`

## CLI: `mtg_bot <subcommand>`

The project has a single CLI dispatcher at `mtg_bot.cli`:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli <subcommand> [args]
```

After `pip install -e .` rebuilds, plain `mtg_bot <subcommand>` works
on PATH. Subcommands:

- `iterate`  — AlphaZero training loop (formerly `scripts/iterate_alphazero.py`)
- `train`    — train the value+policy network from saved data
- `serve`    — bundle/queue HTTP server + v2 coordinator (port 5099)
- `eval`     — head-to-head eval between two registered NN versions, e.g.
  `mtg_bot eval --new v16 --old v13 --games 3000`
- `capture`  — capture annotated NN-driven self-play replays for the UI
- `bench <kind>` — perf benchmarks: `clone`, `clone-zones`, `nn-eval-server`,
  `per-game-wall`
- `ladder`   — **local** fixed-reference strength ladder with Elo: plays
  `random` / `heuristic` / any registered version round-robin on a
  `multiprocessing.Pool` (no coordinator, no runs.db), fits Bradley-Terry
  Elo anchored at `random` = 0 with bootstrap CIs, writes
  `ladder_runs/<ts>/{results.json,ladder.md}` and supports `--resume`.
  This is the project's only absolute-strength measure; results are
  collected in `docs/STRENGTH_LADDER.md`. E.g.
  `mtg_bot ladder --agents random heuristic v13s --games 500`

The shared search-depth + queue-URL + deck-pool defaults live in
`mtg_bot.cli._common` so eval/capture/bench all pick up the current
self-play depth automatically — there is exactly **one** place to
update the default `--nodes` (currently 200, the iter_005 self-play
depth). See the "Eval depth" scope reminder below.

The three remaining production scripts
(`scripts/iterate_alphazero.py`, `scripts/train_nn.py`,
`scripts/serve_setup_scripts.py`) and the four bench scripts
(`scripts/bench_clone.py`, `scripts/bench_clone_zones.py`,
`scripts/bench_nn_eval_server.py`, `scripts/bench_per_game_wall.py`)
remain as **5-line shims** that dispatch into the cli module — every
the bundle/queue server and `hooks/post-commit` keep working without
modification. New code should prefer the `mtg_bot <subcommand>`
form. The v2-cutover deleted `scripts/run_self_play_shard.py` and
`mtg_bot shard` along with the rest of the SSH path.

Throwaway investigation scripts (`scripts/scout_*.py`,
`scripts/profile_*.py`, ad-hoc `scripts/trace_*.py`) intentionally
**stay** under `scripts/` — they're agent-generated diagnostic tools,
not production entrypoints, and porting them would add CLI noise
without value. Don't migrate them.

## Distributed flow (v2 — coordinator + workers, REST)

Post v2-cutover, the cluster is one **coordinator** (the master's
`mtg_bot serve` on port 5099 with the v2 endpoints + heartbeat
poller + queue.db) plus N **workers** (`mtg_bot.worker_daemon` on
port 5198, running as a long-lived native service on each machine).
All traffic is REST over Tailscale; no SSH, no per-iter shard
processes.

The wire contract:
[`docs/DISTRIBUTED_API_v2.md`](docs/DISTRIBUTED_API_v2.md).

Runtime walkthrough of an `mtg_bot iterate` run:
[`docs/DISTRIBUTED_RUN_FLOW.md`](docs/DISTRIBUTED_RUN_FLOW.md).

Day-to-day operations — running iterations, monitoring, aborting:
[`docs/RUNNING_DISTRIBUTED.md`](docs/RUNNING_DISTRIBUTED.md).

Per-machine bootstrap (Python, repo clone, Tailscale, native
service install): [`docs/SETUP_DISTRIBUTED.md`](docs/SETUP_DISTRIBUTED.md)
and the operator guide at [`deploy/README.md`](deploy/README.md).

Worker daemons own their own per-machine operational config
(`--n-workers`, `--deck-pool-path`, `--weights-cache-root`,
`--host`) via CLI flags baked into the native service file. The
orchestrator's `/v2/start_*` request body carries ONLY `stage_id`
and `coordinator_url`; the orchestrator never overrides per-machine
knobs. `shards_config.json` on the master is a URL list:
`{"worker_urls": ["http://<tailnet-host>:5198", ...]}`.

Workers self-update over REST: `POST /v2/self_update` (IDLE-only)
pulls the coordinator's `/repo.bundle` (Tailscale-only, no GitHub
creds — AWS-ready), conditionally rebuilds the Rust engine + deps
only when `rust/`/`Cargo.*`/`pyproject.toml` changed, then re-execs
in place. Pipeline in `mtg_bot.worker_daemon_self_update`; endpoint +
`UPDATING` state + `git_commit`/`last_update` heartbeat fields in
`mtg_bot.worker_daemon`. Operational caveats (pristine-checkout
requirement, Windows Task-Scheduler re-exec, rebuild heuristic) are
in [`deploy/README.md`](deploy/README.md).

**Canonical entrypoint (Phase C unified launcher):** Start an
iteration cycle with `mtg_bot iterate-distributed --config
configs/launch/iter_NNN.toml`. The launcher reads one TOML, pre-
registers the cycle row in `runs.db` (UI sees it immediately),
auto-starts the training UI via the hub if not up, verifies every
configured worker's `/v2/heartbeat == IDLE`, renders a per-cycle
ephemeral `shards_config.json`, prints a clickable Tailscale link
to stderr, then spawns `mtg_bot iterate` as a child subprocess with
the right argv. The cycle row is **guaranteed** to end with a
terminal status — no more silently-stuck `running` rows on a
crash / signal / hard kill. Full operator guide + TOML schema in
[`docs/RUNNING_DISTRIBUTED.md`](docs/RUNNING_DISTRIBUTED.md);
template config at
[`configs/launch/iter_TEMPLATE.toml`](configs/launch/iter_TEMPLATE.toml).
Direct `mtg_bot iterate` invocations still work but are the legacy
manual flow — new operators should use the launcher unconditionally.

## Run database

Every `mtg_bot` subcommand records its run into a SQLite DB at
`training_progress/runs.db` alongside its existing progress JSON.

**Phase 2 invariant (current):** the training UI under
`src/mtg_bot/training_ui/` reads `runs.db` as the single source of
truth. To keep JSON-only producers (today: `iterate`, `train`,
`capture`, `bench` — Phase 1 only wired `mtg_bot eval`)
visible in the dashboard without a manual ops step, the UI runs an
**idempotent backfill from `progress_dir/*.json` into the DB at the
top of each `/api/runs` request**. JSON-side richer fields the DB
schema doesn't carry (machines arrays, training-loss arrays, replay
hints, ...) flow through `run_progress.extra_json` and fan out at
read time.

**Producer dual-write stays.** Do not remove the JSON writes from
producers in this PR; if a UI bug surfaces, a one-line revert of
`mtg_bot.training_ui.app` restores the previous behavior, and the
JSONs on disk remain authoritative.

**Schema** (defined in `src/mtg_bot/cli/runs/db.py`):

- `runs` — one row per invocation: `run_id`, `kind`
  (iterate / self_play / eval / bench / capture / training), `status`
  (running / completed / failed / killed), `started_at`, `ended_at`,
  `git_commit`, `config_json` (full argparse config), `parent_run_id`,
  `label`, `notes`.
- `run_progress` — upsert keyed on `(run_id, shard_label)`. Aggregate
  row uses `shard_label IS NULL`; per-shard rows carry the shard's
  display label. Counters: `n_games_done`, `n_games_total`,
  `wins_new`, `wins_old`, `draws`, `crashed`, `timed_out`,
  `elapsed_seconds`, `extra_json` (free-form).
- `run_events` — append-only event log (started, stage_change,
  promoted, ...).
- `run_artifacts` — registers on-disk byproducts (the existing
  progress JSON, weights.pt, npzs, replay files).

**Env vars** (read fresh on each `client.*` call, so test fixtures
and late `configure(...)` writes propagate):

- `MTG_BOT_RUN_DB_PATH=training_progress/runs.db` — direct-SQLite
  transport. Used by master subcommands (the master has direct
  filesystem access).
- `MTG_BOT_RUN_DB_URL=http://desktop-j05412i:5099` — REST transport.
  Used by any non-master process that needs to write runs.db rows.
  (The v2 worker daemon writes per-game progress to the coordinator's
  v2 endpoints, not directly to runs.db — the coordinator owns the
  durability boundary.)
- Neither set → every `client.*` call is a no-op. Callers' existing
  JSON writes still run unchanged.

**Backup** (online — no need to stop the master):

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli db backup
# default destination: training_progress/backups/runs.db.<UTC-stamp>.bak
```

`mtg_bot db backup --out PATH` overrides the default destination.
Other `mtg_bot db` actions:

- `mtg_bot db dump --run-id X` — pretty-print one run's full record.
- `mtg_bot db query "SELECT ..."` — read-only SELECT escape hatch
  (rejects writes via regex + sqlite read-only URI).
- `mtg_bot db backfill --source training_progress` — idempotently
  upsert every `*.json` run record under the source dir into the
  DB. Invoked automatically by the UI at the top of each
  `/api/runs` request, but exposed here for one-shot batch use
  (e.g. after a fresh checkout, or to clear a stale-running flag
  with `--stale-after-hours 0`).

REST endpoints on the existing port-5099 `mtg_bot serve`:

- `POST /run/start`, `POST /run/<id>/progress`, `POST /run/<id>/event`,
  `POST /run/<id>/finish`, `POST /run/<id>/artifact`,
  `GET /run/<id>`, `GET /runs?status=&kind=&since=&limit=`.

All write endpoints serialize through `runs.db.WRITE_LOCK` so
concurrent POSTs from many remotes don't race the SQLite writer.
Pass `--runs-db-path ''` to `mtg_bot serve` to disable the run-DB
endpoints (default: `training_progress/runs.db`).

## Scope reminders

- Card pool target: **Edge of Eternities** (Scryfall set code `eoe`). Raw data is cached at `data/edge_of_eternities.json`; regenerate via `scripts/fetch_eoe.py`.
- Self-play / eval data-gen and the iterate-AlphaZero loop run on the real 17lands deck pool at `data/17lands/eoe_decks.json.gz` (5000 EoE archetypes). `iterate_alphazero.py` requires `--deck-pool` and `data_gen` / `self_play` raise on missing pool — there is no synthetic mono-color fallback any more.
- Architectural decisions that are locked in: full stack-and-priority simulation (not resolve-immediately); Scryfall JSON as the card data source.
- **Eval-gate threshold has a single source of truth.** `mtg_bot.training.gate.EVAL_GATE_THRESHOLD` is the one constant; iterate's `--acceptance-threshold` default, `scripts/eval_early_stop_watcher.py`'s `--threshold` default, and the training UI's gate-band overlay all read it. Edit that file to change the gate everywhere — never hardcode a literal in a new consumer. Currently 0.525 (raised from 0.51 post-iter_003 to require a slightly larger effect size before promoting; the Wilson 95% CI lower bound from the watcher still does the noise-rejection work — AlphaGo Zero-style "any statistically significant improvement above the floor promotes"). The JS-side `tests-js/fixtures/gate.js` mirrors the constant for unit-test fixtures (Python literals can't cross the language boundary directly); keep them aligned when changing the Python value.
- **Eval-gate uses a 2000-game default + early-stop watcher.** `mtg_bot iterate`'s `--eval-games` defaults to 2000, sized to give close-call evals real statistical headroom (~+/- 2.2 pts at p=50%). Lopsided evals (random-init bootstrap, post-promotion improvements) finish in ~50-100 games once the early-stop watcher (`scripts/eval_early_stop_watcher.py`) writes a stop sentinel — the watcher polls eval progress JSONs, computes Wilson 95% CI on win-rate after `--min-decisive` games, and stops when the CI lower bound clears the acceptance threshold (PASS) or the upper bound falls below it (FAIL). Run the watcher alongside iterate: `python scripts/eval_early_stop_watcher.py --alpha 0.025 --min-decisive 30` (the `--threshold` default tracks `EVAL_GATE_THRESHOLD`; pass it explicitly only to override). Without the watcher, eval phases run to completion — pass `--eval-games 500` explicitly if you don't need the extended headroom.
- **Eval depth must match self-play depth.** The metric we care about is "did v_{N+1} beat v_N at the deployment configuration", and the deployment configuration is whatever depth self-play used. Eval at a different depth measures something else — a network trained at depth=200 may not generalize linearly to depth=400, so a deeper eval doesn't extrapolate cleanly. `mtg_bot.cli.iterate` (formerly `iterate_alphazero.py`) already uses `args.nodes` for both self-play and eval (one knob). Standalone eval invocations should use `mtg_bot eval` (which inherits `--nodes` from `mtg_bot.cli._common`'s defaults) so the depth tracks the current iter's self-play config automatically. There is now exactly one place to update the default — `mtg_bot/cli/_common.py:DEFAULT_NODES`. The pre-iter_005 cleanup found a one-off eval script that had been templated when iter_004 ran at 400 nodes and never re-templated when iter_005 dropped to 200; the consolidated `mtg_bot eval` makes that class of bug structurally impossible.

## Action encoding versions

The action-encoding codec at `src/mtg_bot/training/action_encoding.py`
maps engine `Decision` objects to fixed slot indices for the policy
head. It carries a `version` parameter so the slot semantics can
evolve without breaking trained weights.

**Versions:**

- `"positional_v1"` (default, back-compat with v13/v15/v16):
  positions in hand / graveyard / exile come from the engine's raw
  zone list. Two states differing only in the ordering of `card.id`s
  produce DIFFERENT slot mappings even though their forward play is
  identical. The MCTS NN-eval cache (`mcts._cache_key`) defends
  against the resulting drift via a slot_fingerprint cache-key
  extension (commit 2f50fca on main).

- `"canonical_v2"`: hand / graveyard / exile are sorted by
  `(printing.name, modifier_tuple..., card.id)` (the same forward-
  relevant tuple `canonical_state._canonical_zone_card` builds, with
  `card.id` as the final tiebreaker for determinism) before slot
  assignment. Two canonically-equivalent states share their slot
  mappings, making the slot_fingerprint workaround a no-op.

- `"canonical_v3"` (2026-10-06, `feature/s2-resolution-decisions`):
  `canonical_v2`'s 517 slots unchanged + a 288-slot resolution block
  (targets by zone/side, discard, mode(s), sacrifice, X, surveil,
  shock life, yes/no, tutor / opponent-hand picks, devour) = 805.
  `action_encoding.action_vocab_size(version)` is the width. Each
  option's slot is fixed at pause time by the resolution channel agent
  (`mtg_bot.search.resolution_slots`). Full design:
  [`docs/RESOLUTION_DECISIONS.md`](docs/RESOLUTION_DECISIONS.md).
  A v2 checkpoint becomes v3 WITHOUT retraining via `mtg_bot
  extend-resolution-slots` (bit-identical outside resolution pauses).

**Resolution decisions are searched in production** (since
2026-10-06): `HeuristicsConfig.search_resolution_decisions` (default
ON; `MTG_BOT_SEARCH_RESOLUTION_DECISIONS=0` restores the old
all-`DefaultTacticalAgent` behaviour; the test session runs with it
OFF via `tests/conftest.py`, new-default tests opt in). Per-agent
opt-out: `PimcConfig` / `NNPimcConfig` `search_resolution_decisions`
(ladder `+autotarget`). A searched PUCT seat needs a v3 policy head --
v1/v2 are refused, never run on all-zero priors. Training a v3 model on
a window that still holds legacy 517-wide shards needs
`--legacy-policy-encoding canonical_v2` (or
`MTG_BOT_LEGACY_POLICY_ENCODING=canonical_v2`): they are zero-padded
explicitly, with counts in `training.policy_report`.

**Metadata flag:**

Each registered model's `metadata.json` carries
`action_encoding_version` (defaults to `"positional_v1"` when absent
— v13/v15/v16's metadata predates the field and back-compat-defaults
correctly). `Registry.load_metadata` reads it; loaders thread it into
`NNEvaluator(action_encoding_version=...)`, which in turn threads it
through every `legal_action_mask` call. `MCTS` and `ISMCTS` pull it
off `evaluator.action_encoding_version` at construction time so PUCT
prior lookups use the matching slot mapping.

**Migration:**

1. v13/v15/v16 stay on v1 — their metadata's missing field defaults
   to v1, behavior is bit-identical to before this feature landed.
2. v17+ training opts into v2 by passing
   `--action-encoding-version canonical_v2` to `mtg_bot train`. The
   flag writes through to `ModelMetadata.action_encoding_version`,
   which loaders read back at inference.
3. Switching versions is a v-chain break — old weights cannot be
   reused (the trained policy head memorized the v1 slot mapping;
   loading it under v2 would scramble priors).
4. v2 weights make the slot_fingerprint workaround a no-op (canonical
   states have canonical slot mappings — there's no cross-state-to-
   different-slots case left to defend against). The workaround
   stays in place for v1 weights.

**End-to-end wiring (closed by `feature/v17-distributed-encoding`):**

- `iterate` reads `bootstrap_meta.action_encoding_version` and
  threads it through both the self-play spec (single
  `action_encoding_version` field on `SelfPlayJobSpec` — both seats
  share a version in self-play) and the eval spec (separate
  `new_action_encoding_version` + `old_action_encoding_version`
  on `EvalJobSpec` so cross-version evals like v17[v2] vs v13[v1]
  work). `train_nn.py`'s `--action-encoding-version` flag flows
  through to `ModelMetadata.action_encoding_version` as before.
- All three distributed runners (`run_shard_local`,
  `run_shard_pull`, `run_shard_eval_pull`) thread the version into
  `NNPimcConfig`. `run_shard_pull`'s worker-args tuple is the
  13-element form `data_gen._play_one_parallel_game` already
  supports.

**Out of scope of v2 today:**

- v2 has only been verified at the codec level + a local-only
  smoke (50 self-play / 1 epoch / 50 eval against a tiny
  random-init bootstrap); no production-scale v17 weights trained
  against it exist yet. The first real v2 training run will
  surface any inference-side wiring gap that the smoke missed.

## Card vocabulary fingerprint

The structured net indexes `printing_embed` (and the Step-4
`hand_belief_queries`) by **printing id** = position in the sorted
name list of the active card source (scheme `sorted_names_v1`,
`mtg_bot.training.card_vocab`; rows above `n_printings - 1` fold by
modulo). Any card-pool change, or the parser claiming one more card,
renumbers those ids. Full design + runbook:
[`docs/CARD_VOCAB_FINGERPRINT.md`](docs/CARD_VOCAB_FINGERPRINT.md).

- **Every structured checkpoint records its vocabulary** in
  `metadata.json` `card_vocab` (`{format, id_scheme, size, sha256,
  names, provenance}`); `Registry.register` refuses one without it.
  `mtg_bot train` stamps the vocabulary of its **training shards**;
  `NNEvaluator.save` stamps a `<weights>.card_vocab.json` sidecar.
- **Loaders remap, never guess**: every structured load
  (`NNEvaluator`, `from_metadata`, `NNPimcConfig.build`, the SHM eval
  server) goes through `card_vocab.load_structured_checkpoint`. Same
  vocabulary → bit-identical load. Changed → embedding rows remapped by
  card name (shared cards keep their trained rows, new cards get a
  fresh model's init, dropped cards removed, table grown to
  `len(vocab)+1`), logged at WARNING as `CARD VOCABULARY REMAP`.
  No vocabulary → `MissingCardVocabError`.
- **Pre-fingerprint checkpoints** (every `trained_weights_s2/v1`–`v26`
  and `trained_weights/` version: the same 475-name vocabulary,
  `25ac27bcad11`) are stamped once with `mtg_bot card-vocab stamp
  --registry-root <R> --all-versions --use-metadata-commit --sidecar`
  (writes only `vN/weights.pt.card_vocab.json`; `Registry.load_metadata`
  and the coordinator's `/v2/weights/<v>/metadata` merge it in, so
  workers receive it with no extra fetch).
- **Shards** record their id space (`structured_card_vocab` npz key:
  vocabulary + `n_printings_cap`). The trainer refuses mixed id spaces
  and unstamped (pre-fingerprint) shards unless `--shard-card-vocab
  FILE` / `$MTG_BOT_SHARD_CARD_VOCAB` asserts their vocabulary
  (`mtg_bot card-vocab export --git-commit <sha> --out FILE`).
- Not a v-chain break: the codec and the id scheme are unchanged. A new
  id scheme must bump `card_vocab.id_scheme` (the fingerprint covers it).

## NN representation roadmap

**Status (2026-05-17 — verified against `trained_weights/`): Steps 1
& 3 have LANDED.** The production v-chain is `architecture:
structured` for every registered version `v1`–`v10` (`current_best:
v5`); the NN input is built by `mtg_bot.training.state_tokenizer` →
`StructuredModel` (per-object tokens, learned printing embeddings,
last-32 action history, viewer-relative redaction). The 51-dim
hand-engineered featurizer at `src/mtg_bot/features.py` + flat
`_ValuePolicyMLP` is the **legacy pre-structured path** — those
weights (`v11`–`v16`, 1024-1024-512 trunk) are **archived** at
`trained_weights/archived_pre_structured/`, not production. The
committed four-step plan and its dated implementation status are in
**[`docs/NN_REPRESENTATION_ROADMAP.md`](docs/NN_REPRESENTATION_ROADMAP.md)**.

Summary:

1. **Step 1 — LANDED** — per-object transformer input with learned
   printing embeddings, INCLUDING last-N action tokens. Single
   architectural rewrite, single v-chain break.
2. **Step 2 — NOT landed** — offline deck-belief model trained on
   17lands data; independent of the main NN. Hidden info is still
   handled by PIMC determinization (opponent hand = anonymous slots).
3. **Step 3 — LANDED** — folded into Step 1 (last-32 action tokens
   shipped with the structured input rewrite).
4. **Step 4 — NOT landed** — hand-belief auxiliary head on Step 1's
   trunk, conditioned on Step 2's deck-belief.

Net-scale caveat: the shipped structured trunk is **small** (`emb_dim
128, n_layers 2, hidden_dim 256`); structured-input-landed ≠
large-trunk. **Which scaling lever is binding is being decided
empirically in [`docs/SCALING_DIAGNOSTICS.md`](docs/SCALING_DIAGNOSTICS.md)**
(pre-registered, eval-gate-adjudicated). As of 2026-05-17: trunk
**width is NOT indicated** (no overfit; value head still learning on
the current trunk), **search depth is a VALIDATED gate-promotable
lever** (v5 is depth-limited: `v5@400/@800` beat `v5@200` 66 %/72 %,
Wilson-LB ≥ 0.525, monotone), and the more-epochs value-head lever is
under test. Read that doc before any width/depth/data scaling
decision; perf/economics cross-ref `docs/BENCHMARKING_FINDINGS.md`
§1/§2/§2A.

Both belief outputs feed importance-sampled PIMC determinizations
(ReBeL / DeepStack pattern).

Read the doc before making any architectural change to the NN input or
the trunk. It captures the data-cost analysis, the open design
questions to nail down before Step 1 starts, validation strategy per
step, and why we explicitly rejected Phase 0 (aux heads on the current
featurizer) and Phase 3 (MuZero) from the earlier research-style
version of this roadmap.

## Benchmarking findings (single source of truth)

**All performance/economics measurements live in
[`docs/BENCHMARKING_FINDINGS.md`](docs/BENCHMARKING_FINDINGS.md)** —
dated, with method, context, and the honest corrections. Read it
before quoting any perf/cost multiplier, and **add a dated row when
you measure something new** rather than asserting from priors or help
text. Load-bearing facts as of 2026-05-17 (see the doc for full
context + caveats): AWS CPU spot ≈ **$1.70–1.75 / 1000 games**, ~20–40×
local-owned cost (AWS = elastic burst only); validated **`device=cuda`
≈ 2× per-game vs `cpu`** on the production config path (currently
*unrealized* — the worker daemon is CPU-by-design; opt-in wiring is
`feature/worker-daemon-gpu-device`); the **batched SHM eval server is
~2× SLOWER** than per-process cuda at the current (tiny) net scale —
do not wire it in (re-measure at a larger trunk). The earlier
"3–10× GPU/SHM" claims (incl. `iterate` help text + dead `--use-shm-*`
flags/`use_shm_server` spec fields) are **discredited** — they came
from a benchmark that could not load the structured net; treat that
lore as wrong until a dated row says otherwise.

## Documenting deliberate divergences

Any shortcut, heuristic, search optimization, or incorrect / outdated implementation of real MTG comprehensive rules **must be documented in `docs/SEARCH_LIMITATIONS.md`** at the time it lands.

This applies to (non-exhaustive):

- Engine code that intentionally diverges from current CR (e.g., outdated combat damage assignment, simplified mana ability rules, unmodeled state-based actions, missing layers).
- Search-side optimizations that approximate or skip legal decisions (e.g., capability-based priority short-circuits, factored decompositions of joint decisions, heuristic move pruning).
- Workarounds that "land" via filter rather than fix (e.g., the data_gen writer-side root-legal visit filter).
- Known-incorrect interactions between specific card mechanics that we accept rather than fix.

For each entry, the doc requires: location (file:line), what the shortcut is, why it's OK or which scenarios make it not-OK, failure mode, fix-scope estimate, and impact estimate (perf gain or strength cost).

If you ship a feature that takes any deliberate shortcut without updating `SEARCH_LIMITATIONS.md`, you're shipping technical debt that nobody else will know about. Pause and add the entry first.
