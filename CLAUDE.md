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
  Network scale (1024-1024-512 trunk, ~10M params, dual-head AlphaZero — v13 / v15 / v16) is now large enough that GPU inference dominates CPU on every host that has a GPU. As of `feature/eval-gpu-auto-device`: (1) `auto_device()` falls back to `mps` when CUDA isn't available (Apple Silicon remote), and (2) the four NNPimcConfig construction sites in `src/mtg_bot/training/distributed.py` (run_shard_local self-play, run_shard_pull self-play, run_shard_eval_pull new+old) explicitly pass `device=auto_device()`. Each call resolves locally on the host where the shard runs — master picks CUDA, Mac picks MPS, Win10 picks CUDA. `NNPimcConfig.device` default stays `None` (which maps to NNEvaluator's CPU default) so direct test construction `NNPimcConfig(...)` continues to use CPU — needed because tests spawn multiprocessing pools and per-worker CUDA-context init on Windows-spawn semantics deadlocks. The earlier "workers stay CPU-only / Python↔CUDA round-trip overhead beats tiny matmuls" guidance was correct for the v2/v3 25k-65k-param scale and is now stale for production paths.
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
- `shard`    — run one self-play / eval shard (SSH-invoked from master)
- `serve`    — bundle/queue HTTP server (port 5099)
- `eval`     — head-to-head eval between two registered NN versions, e.g.
  `mtg_bot eval --new v16 --old v13 --games 3000`
- `capture`  — capture annotated NN-driven self-play replays for the UI
- `bench <kind>` — perf benchmarks: `clone`, `clone-zones`, `nn-eval-server`,
  `per-game-wall`

The shared search-depth + queue-URL + deck-pool defaults live in
`mtg_bot.cli._common` so eval/capture/bench all pick up the current
self-play depth automatically — there is exactly **one** place to
update the default `--nodes` (currently 200, the iter_005 self-play
depth). See the "Eval depth" scope reminder below.

The four production scripts (`scripts/iterate_alphazero.py`,
`scripts/train_nn.py`, `scripts/run_self_play_shard.py`,
`scripts/serve_setup_scripts.py`) and the four bench scripts
(`scripts/bench_clone.py`, `scripts/bench_clone_zones.py`,
`scripts/bench_nn_eval_server.py`, `scripts/bench_per_game_wall.py`)
remain as **5-line shims** that dispatch into the cli module — every
existing SSH invocation, the bundle/queue server, and
`hooks/post-commit` keep working without modification. New code
should prefer the `mtg_bot <subcommand>` form.

Throwaway investigation scripts (`scripts/scout_*.py`,
`scripts/profile_*.py`, ad-hoc `scripts/trace_*.py`) intentionally
**stay** under `scripts/` — they're agent-generated diagnostic tools,
not production entrypoints, and porting them would add CLI noise
without value. Don't migrate them.

## Distributed flow (master + remotes)

The cluster is one master (Win11 desktop) + N remotes (Mac, Win10).
For day-to-day operations — running iterations, running remote
benches that surface in the training UI, monitoring, aborting — see
**[`docs/RUNNING_DISTRIBUTED.md`](docs/RUNNING_DISTRIBUTED.md)**.

For one-time per-machine bootstrap (Python, repo clone, SSH keys,
Tailscale enrollment), see
[`docs/SETUP_DISTRIBUTED.md`](docs/SETUP_DISTRIBUTED.md).

## Run database

Every `mtg_bot` subcommand records its run into a SQLite DB at
`training_progress/runs.db` alongside its existing progress JSON.
**Phase 1 invariant: dual-write only.** The training UI under
`src/mtg_bot/training_ui/` keeps reading the JSON files — do not
remove or rename those writes. Phase 2 swaps the UI; until then
the DB is a shadow store.

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
  Used by SSH-launched remote shards (no shared filesystem).
- Neither set → every `client.*` call is a no-op. Phase 1 invariant:
  callers' existing JSON writes still run unchanged.

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
- **Eval-gate uses a 2000-game default + early-stop watcher.** `mtg_bot iterate`'s `--eval-games` defaults to 2000, sized to give close-call evals real statistical headroom (~+/- 2.2 pts at p=50%). Lopsided evals (random-init bootstrap, post-promotion improvements) finish in ~50-100 games once the early-stop watcher (`scripts/eval_early_stop_watcher.py`) writes a stop sentinel — the watcher polls eval progress JSONs, computes Wilson 95% CI on win-rate after `--min-decisive` games, and stops when the CI lower bound clears the acceptance threshold (PASS) or the upper bound falls below it (FAIL). Run the watcher alongside iterate: `python scripts/eval_early_stop_watcher.py --threshold 0.52 --alpha 0.025 --min-decisive 30`. Without the watcher, eval phases run to completion — pass `--eval-games 500` explicitly if you don't need the extended headroom.
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

## Documenting deliberate divergences

Any shortcut, heuristic, search optimization, or incorrect / outdated implementation of real MTG comprehensive rules **must be documented in `docs/SEARCH_LIMITATIONS.md`** at the time it lands.

This applies to (non-exhaustive):

- Engine code that intentionally diverges from current CR (e.g., outdated combat damage assignment, simplified mana ability rules, unmodeled state-based actions, missing layers).
- Search-side optimizations that approximate or skip legal decisions (e.g., capability-based priority short-circuits, factored decompositions of joint decisions, heuristic move pruning).
- Workarounds that "land" via filter rather than fix (e.g., the data_gen writer-side root-legal visit filter).
- Known-incorrect interactions between specific card mechanics that we accept rather than fix.

For each entry, the doc requires: location (file:line), what the shortcut is, why it's OK or which scenarios make it not-OK, failure mode, fix-scope estimate, and impact estimate (perf gain or strength cost).

If you ship a feature that takes any deliberate shortcut without updating `SEARCH_LIMITATIONS.md`, you're shipping technical debt that nobody else will know about. Pause and add the entry first.
