# Running the distributed flow + bench → UI

How to operate the cluster after one-time setup is done. For
per-machine bootstrap (Python, repo clone, SSH keys, Tailscale),
see [`SETUP_DISTRIBUTED.md`](SETUP_DISTRIBUTED.md).

## Architecture at a glance

The cluster is **one master + N remotes**, all on the same Tailnet.

| Role | Hosts (today) | What runs there |
|---|---|---|
| Master | Win11 desktop (`desktop-j05412i`) | Hub dashboard (port 9000), training UI (port 5003), iteration driver, distributed pull-queue server, training (CUDA), eval acceptance gate |
| Remote | Mac (`greggs-macbook-pro`), Win10 desktop (`desktop-g72aptg`) | Self-play workers pulling games from the queue. Optional standalone bench. |

The master holds the source of truth for **everything stateful**:
- `iterate_data/` per-iteration artifacts (datasets, replays, logs)
- `trained_weights/` versioned weights (registry)
- `training_progress/` JSON files the UI reads
- The pull-queue server (one HTTP endpoint per active iteration)

Remotes are **stateless workers** — they pull batches of game indices
from the master's queue, run them, push results back. A remote going
offline mid-iteration is recovered by the queue handing the same
batch to another worker on next pull.

## Daily-ops checklist (master)

Run these once when you sit down at the desktop:

1. **Hub dashboard** (manages all per-project services) — start if not
   already running:
   ```bash
   cd "C:\Users\Gregg Keithley\hub" && python app.py
   ```
   Reachable at [http://localhost:9000](http://localhost:9000) /
   [http://desktop-j05412i:9000](http://desktop-j05412i:9000) over
   Tailscale. The hub has start/stop buttons for the project services
   listed in `PROJECTS` in `hub/app.py`.

2. **Training UI** (renders progress JSON files) — start via the
   hub's `mtg_bot_training_ui` row, or directly:
   ```bash
   .venv-py313/Scripts/python.exe -u -m mtg_bot.training_ui \
       --port 5003 \
       --progress-dir "C:\Users\Gregg Keithley\Documents\Claude\Projects\mtg-bot\training_progress"
   ```
   Reachable at [http://localhost:5003](http://localhost:5003) /
   [http://desktop-j05412i:5003](http://desktop-j05412i:5003).
   Renders one card per JSON file in `--progress-dir`; auto-detects
   `kind` field (`data_gen` / `eval` / `bench`).

3. **Verify Mac is reachable** (smoke test):
   ```bash
   ssh -o BatchMode=yes greggs-macbook-pro "uname -a"
   ```
   Tailscale SSH should auth without prompting.

4. **Snapshot the run database** (online, no service interruption):
   ```bash
   PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli db backup
   ```
   Lands at `training_progress/backups/runs.db.<UTC-stamp>.bak`.
   `runs.db` is the single source of truth for cluster-wide run
   history (kind / status / config / progress / events / artifacts),
   so a daily snapshot is cheap insurance against a corruption or
   accidental schema-mutating edit. The DB lives only on master
   (Phase 1 — never on remotes); the snapshot does too. See the
   "Run database" section in `CLAUDE.md` for schema details and
   the `mtg_bot db dump` / `query` introspection commands.

## Running a self-play iteration

The `mtg_bot iterate` subcommand runs the
**self-play → train → eval-gate → bump-version** loop. (The
``scripts/iterate_alphazero.py`` thin shim still works for legacy
SSH / cmd-line habits and dispatches to the same code.)

### Local-only (single-machine)

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli iterate \
    --bootstrap-version v6 \
    --n-iterations 5 \
    --games-per-iteration 500 \
    --samples 4 \
    --nodes 200 \
    --workers 16 \
    --deck-pool data/17lands/eoe_decks.json.gz
```

`--deck-pool` is **required** — every self-play / eval game samples
from the real EoE pool. There is no synthetic-deck fallback.

Self-play runs on this machine's `multiprocessing.Pool`. Training
runs on this machine's GPU (auto-detected). Eval runs on this
machine's CPU pool.

### Distributed (master + remotes)

Add `--shards-config` pointing at a JSON file describing the cluster:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli iterate \
    --bootstrap-version v6 \
    --n-iterations 5 \
    --games-per-iteration 1000 \
    --shards-config shards_config.json \
    --deck-pool data/17lands/eoe_decks.json.gz
```

`shards_config.json` shape (see
[`SETUP_DISTRIBUTED.md` § 3](SETUP_DISTRIBUTED.md) for the full
schema):

```json
{
  "shards": [
    {"name": "master",     "host": "localhost",          "n_workers": 16},
    {"name": "mac",        "host": "greggs-macbook-pro", "n_workers": 8},
    {"name": "win10",      "host": "desktop-g72aptg",    "n_workers": 12}
  ]
}
```

What happens during a distributed iteration:

1. **Master spawns the pull-queue HTTP server** for this iteration
   (default port chosen at startup; logged to stdout).
2. **Master SSHes each remote** in parallel and starts
   `scripts/run_self_play_shard.py` (a 5-line shim that dispatches into
   ``mtg_bot.cli.shard``) with the spec base64-encoded as
   `--spec-json`. Each remote shells the script as a background process
   and returns; the master polls the queue for completion.
3. **Each shard worker** pulls game-index batches from the master's
   queue, runs them locally, writes per-game progress to its local
   `training_progress/<run_id>_shard_<host>.json` file (and to the
   shared dataset on disk via final-stage rsync).
4. **Master's pull-queue** times out batches that don't return
   (default 600s) and re-hands them to another worker.
5. **Iteration completes** when all games are done or the master
   receives Ctrl-C. Shard JSONs land back at the master via the
   final rsync stage in `run_self_play_shard.py`.

Per-shard progress JSONs are written to the master's
`training_progress/` and surface in the UI **automatically** — one
card per shard.

## Running a remote bench → UI

The Mac (or any remote) can run a standalone per-game-wall benchmark
that streams to the master's training UI. Two pieces:

### On the remote: emit progress

```bash
ssh greggs-macbook-pro "cd ~/projects/mtg-bot && \
    PYTHONPATH=src nohup .venv-py313/bin/python -u -m mtg_bot.cli bench per-game-wall \
        --n-games 50 \
        --workers 8 \
        --bootstrap-version v6 \
        --progress-dir training_progress \
        --label mac-bench-post-combat-prunes \
    > /tmp/mac_bench.log 2>&1 &"
```

(``scripts/bench_per_game_wall.py`` still works as a legacy shim.)

Key flags:
- `--progress-dir` — opt in to `BenchProgress` JSON emission. When
  unset, the script behaves as before (just stdout percentiles).
- `--run-id` — defaults to `bench_<UTC-timestamp>`; pin if you want a
  predictable filename for the sync helper.
- `--label` — free-form label used in the UI card metadata (e.g.
  `mac-standalone-bench`).

The script writes progress to
`~/projects/mtg-bot/training_progress/bench_<timestamp>.json` on the
remote. Find the file with:
```bash
ssh greggs-macbook-pro "ls -t ~/projects/mtg-bot/training_progress/bench_*.json | head -1"
```

### On the master: sync to the UI's progress_dir

```bash
MSYS_NO_PATHCONV=1 python -u scripts/sync_remote_bench_progress.py \
    --remote-host greggs-macbook-pro \
    --remote-path /Users/greggkeithley/projects/mtg-bot/training_progress/bench_20260505T050504Z.json \
    --local-path "training_progress/bench_20260505T050504Z.json" \
    --interval-sec 5 &
```

The helper polls the remote file via `scp` every 5s, copies it to
the master's `training_progress/`, and stops when the JSON's
`_status` field is `completed` or `failed`. The training UI picks up
the local file on its next poll and renders a bench card.

**Critical Git Bash gotcha**: on Windows Git Bash, the absolute Unix
path `/Users/...` gets mangled to `C:/Program Files/Git/Users/...`
unless you set `MSYS_NO_PATHCONV=1`. Symptom: `scp` complains
`No such file or directory`. Always set the env var when invoking
the sync helper from Git Bash.

### What you'll see in the UI

A new card appears in the training UI labeled with your `--label`
value, showing:

- Run ID + git commit
- Single pairing (self-play has no challenger/defender, so
  winner=None for every game; **the Wilson CI column is vacuous —
  ignore it**)
- Live `n_games_done / n_games_total` progress bar
- Elapsed time

Once `--n-games` complete, the bench script prints percentiles
(`p50, p75, p90, p95, p99`) and a per-timeout-candidate table to
stdout (visible via `ssh greggs-macbook-pro "tail /tmp/mac_bench.log"`).

## Tailing a running iteration

The master's iteration driver streams to stdout. To check on it from
another machine:

```bash
ssh desktop-j05412i "tail -f ~/Documents/Claude/Projects/mtg-bot/iterate_data/iter_NNN/master.log"
```

Per-iteration artifacts live in
`mtg-bot/iterate_data/iter_NNN/`:
- `master.log` — driver stdout
- `self_play_<config>_shard_<host>.json` — per-shard progress (also
  visible in the UI)
- `eval_<challenger>_vs_<defender>_shard_<host>.json` — eval-match
  progress (also visible in the UI)
- `dataset_<config>.parquet` — final concatenated training data

## Aborting a run

- **Iteration (graceful, preferred)**: click "Stop run" in the
  training UI on the cycle card. This writes a `.STOP_<cycle_id>`
  sentinel under `training_progress/`. Iterate's cycle loop, the
  local self-play / eval pools, and the **distributed orchestrator**
  (`run_distributed_self_play` / `run_distributed_eval`) all poll
  the sentinel and exit cleanly at the next stage / dispatch
  boundary. The queue server's `/queue/take` also reads the
  sentinel and starts returning a drain signal so any in-flight
  remote shards finish their current game and stop pulling new
  ones; partial counts (eval) and partial npzs (self-play) are
  preserved through the orchestrator's return value.

  Stop-check propagation is honored end-to-end as of
  `feature/distributed-stop-check`. Before that, the distributed
  orchestrator ignored the sentinel — eval would grind to its
  full `--eval-games` cap regardless of the watcher's PASS / FAIL
  decision. Standalone CLI evals (`mtg_bot eval`) still need
  `Ctrl-C` since they don't have a cycle-level sentinel.
- **Iteration (force)**: `Ctrl-C` the master's `mtg_bot iterate`
  process. The queue server shuts down; in-flight remote shards
  finish their current game then exit cleanly. Partial dataset is
  preserved.
- **Eval early-stop watcher**: run alongside iterate to
  short-circuit close-call evals once the Wilson 95% CI is
  conclusive vs the threshold. Writes the same `.STOP_<cycle_id>`
  sentinel as the UI button, so the same end-to-end propagation
  applies — distributed evals now actually stop dispatching
  rather than running to the full cap. See
  [`scripts/eval_early_stop_watcher.py`](../scripts/eval_early_stop_watcher.py).
- **Single shard**: SSH to the host and `pkill -f run_self_play_shard`
  (or `Stop-Process` on Windows). Master's queue will re-hand the
  unfinished batches to another worker.
- **Bench**: `ssh greggs-macbook-pro "pkill -f bench_per_game_wall"`.
  Sync helper exits on its next poll.

## Common pitfalls

- **Mac auth fails on `git fetch`**: re-auth via deploy key or `gh
  auth login` (per the 2026-05-04 fix). Symptoms: `git pull` on the
  Mac shell prompts for username/password (no TTY in the SSH path).
- **MSYS path mangling on Git Bash**: see the bench-sync section
  above. Always `MSYS_NO_PATHCONV=1` for `scp` invocations using
  absolute Unix paths.
- **Remote is N commits behind**: the iteration driver auto-pulls
  before launching shards; standalone bench requires a manual `git
  pull` first.
- **Shard JSON not appearing in UI**: check that the `--progress-dir`
  on the remote matches the path the shard worker is writing to, and
  that the `sync_remote_bench_progress.py` helper is running on the
  master with the right `--remote-path`.
- **`scripts/run_self_play_shard.py` failing on Windows path
  encoding**: paths in `SelfPlayJobSpec.to_dict()` are normalized to
  forward slashes; if you see `\\`-mangled paths in the spec, check
  that the master is on a recent enough commit to have the
  `_posix(p)` normalizer.
- **Hub on port 9000 is dead**: `cd hub && python app.py`. The hub is
  the central control panel — the training UI and other services'
  start/stop buttons depend on it.

## See also

- [`SETUP_DISTRIBUTED.md`](SETUP_DISTRIBUTED.md) — one-time per-machine
  bootstrap (Python, repo, SSH, Tailscale).
- [`SEARCH_LIMITATIONS.md`](SEARCH_LIMITATIONS.md) — every documented
  shortcut / heuristic / CR divergence in the engine + search code.
  Knowing what's in the SelfPlayJobSpec defaults vs EvalJobSpec
  defaults matters for interpreting bench / eval numbers.
- The `hub/app.py` `PROJECTS` list — registered project services
  (start/stop buttons in the dashboard).
