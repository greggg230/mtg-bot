# Deploy: mtg-bot worker daemon

Operator guide for getting the mtg-bot worker daemon running on each
machine in the cluster. Companion to `docs/DISTRIBUTED_API_v2.md` (the
wire contract) and `docs/DISTRIBUTED_RUN_FLOW.md` (the runtime walk-
through).

## What's here

| File | OS | Purpose |
|---|---|---|
| `systemd/mtg-bot-worker.service` | Linux | systemd unit |
| `launchd/com.mtg-bot.worker.plist` | macOS | launchd plist |
| `windows/install-mtg-bot-worker.ps1` | Windows | Worker daemon Task Scheduler installer |
| `windows/install-mtg-bot-coordinator.ps1` | Windows (master only) | Coordinator Task Scheduler installer |

Each file is a template with placeholders. Fill them in for the
machine you're deploying on, install via the OS's native service
mechanism, then verify the daemon is up.

## Per-machine prerequisites

1. **Clone the repo** and create a Python 3.13 venv:

   ```bash
   git clone https://github.com/<you>/mtg-bot.git
   cd mtg-bot
   py -3.13 -m venv .venv-py313       # Windows
   python3.13 -m venv .venv-py313     # Linux/macOS
   .venv-py313/Scripts/python -m pip install -e .[dev]   # Windows
   .venv-py313/bin/python -m pip install -e .[dev]       # POSIX
   ```

2. **Install Tailscale and join the tailnet.** The worker daemon
   binds to the machine's tailnet IP so the coordinator can poll it.

   ```bash
   tailscale up
   tailscale ip -4   # note this — it's the daemon's --host
   ```

3. **Pre-stage the deck pool** at `data/17lands/eoe_decks.json.gz`
   (the daemon's default; override via `--deck-pool-path`).

## Linux (systemd)

```bash
# 1. Copy the unit, edit the placeholders.
sudo cp deploy/systemd/mtg-bot-worker.service \
    /etc/systemd/system/mtg-bot-worker.service
sudo $EDITOR /etc/systemd/system/mtg-bot-worker.service
#   replace ${USER}, ${HOME}, ${REPO_ROOT}, ${VENV_PYTHON},
#   ${TAILNET_IP}, ${N_WORKERS}

# 2. Enable + start.
sudo systemctl daemon-reload
sudo systemctl enable mtg-bot-worker
sudo systemctl start mtg-bot-worker

# 3. Verify.
sudo systemctl status mtg-bot-worker
journalctl -u mtg-bot-worker -f
curl http://<tailnet-ip>:5198/v2/heartbeat
```

## macOS (launchd)

```bash
# 1. Copy the plist, edit the placeholders.
cp deploy/launchd/com.mtg-bot.worker.plist ~/Library/LaunchAgents/
$EDITOR ~/Library/LaunchAgents/com.mtg-bot.worker.plist
#   replace {{HOME}}, {{REPO_ROOT}}, {{VENV_PYTHON}}, {{TAILNET_IP}},
#   {{N_WORKERS}}

# 2. Load.
launchctl load -w ~/Library/LaunchAgents/com.mtg-bot.worker.plist

# 3. Verify.
launchctl list | grep mtg-bot
tail -f ~/Library/Logs/mtg-bot/worker.out.log
curl http://<tailnet-ip>:5198/v2/heartbeat
```

The plist sets `PATH` (includes `{{HOME}}/.cargo/bin` + `/opt/homebrew/bin`)
and `VIRTUAL_ENV` (`{{REPO_ROOT}}/.venv-py313`) in `EnvironmentVariables`
so `POST /v2/self_update` can rebuild the Rust engine. launchd's default
agent `PATH` excludes both the rustup toolchain and Homebrew, and
maturin's auto-detect doesn't recognise the `.venv-py313` directory name
— without these the rebuild step fails on any commit touching
`rust/`/`Cargo.*`. If you install rust to a non-default prefix, edit the
`PATH` value accordingly.

## Windows (Task Scheduler)

Run from **elevated** PowerShell (Task Scheduler with SYSTEM principal
needs admin).

### Worker daemon (every Windows machine)

```powershell
cd <repo-root>
.\deploy\windows\install-mtg-bot-worker.ps1 `
  -TailnetIP "100.96.1.7" `
  -NWorkers 4 `
  -RepoRoot "C:\Users\gregg\projects\mtg-bot" `
  -VenvPython "C:\Users\gregg\projects\mtg-bot\.venv-py313\Scripts\python.exe"

# Verify
Get-ScheduledTask -TaskName MTGBotWorker
Get-ScheduledTaskInfo -TaskName MTGBotWorker

# Start now (or reboot)
Start-ScheduledTask -TaskName MTGBotWorker

# Heartbeat smoke check
Invoke-RestMethod -Uri "http://<tailnet-ip>:5198/v2/heartbeat"
```

### Coordinator (master only — one per cluster)

The coordinator runs ONLY on the master. It's the existing
`mtg_bot serve` process extended with the v2 endpoints + heartbeat
poller + queue.db; the install script just wraps it in a
Task-Scheduler-as-SYSTEM entry.

```powershell
cd <repo-root>
.\deploy\windows\install-mtg-bot-coordinator.ps1 `
  -RepoRoot "C:\Users\gregg\projects\mtg-bot" `
  -VenvPython "C:\Users\gregg\projects\mtg-bot\.venv-py313\Scripts\python.exe"

# Verify
Get-ScheduledTask -TaskName MTGBotCoordinator
Get-ScheduledTaskInfo -TaskName MTGBotCoordinator

# Start now
Start-ScheduledTask -TaskName MTGBotCoordinator

# Health check
Invoke-RestMethod -Uri "http://127.0.0.1:5099/v2/healthz"
```

### Cleanup of the cutover-day one-shot smoke task

The cutover-day smoke registered an `MTGBotWorkerSmoke` one-shot
task on Win10 (and possibly the master) to get a daemon running
without elevation. After installing the proper persistent
`MTGBotWorker` task, remove the one-shot:

```powershell
Unregister-ScheduledTask -TaskName MTGBotWorkerSmoke -Confirm:$false
```

## Verify the daemon is up (any OS)

```bash
curl http://<tailnet-ip>:5198/v2/heartbeat | jq .
```

A healthy daemon returns 200 with a body like:

```json
{
  "worker_id": "greggs-macbook-pro",
  "state": "IDLE",
  "current_stage_id": null,
  "games_done": 0,
  "games_claimed_in_flight": 0,
  "last_progress_at": null,
  "error": null,
  "uptime_s": 12.4,
  "process_started_at": "2026-05-15T18:00:00Z",
  "host": {
    "pool_size": 0,
    "active_workers": 0,
    "cpu_percent": 4.2,
    "memory_percent": 38.1,
    "memory_used_gb": 12.2,
    "memory_total_gb": 32.0,
    "load_avg_1m": 0.4
  }
}
```

## Cutover-day playbook

Once each machine in the cluster has a daemon running and ALIVE per
the coordinator's heartbeat poller:

1. **Bring up the coordinator.** On the master:

   ```bash
   cd <mtg-bot>
   .venv-py313/Scripts/python -m mtg_bot.cli serve   # port 5099
   ```

   The v2 blueprint registers automatically; the heartbeat poller
   reads `shards_config.json` on startup.

2. **Update `shards_config.json`** to the URL-only post-cutover
   shape (this is checked into the repo as a reference; copy to the
   master's working directory):

   ```json
   {
     "worker_urls": [
       "http://desktop-j05412i:5198",
       "http://greggs-macbook-pro:5198",
       "http://desktop-g72aptg:5198"
     ]
   }
   ```

3. **Confirm coordinator sees every worker as ALIVE.** The training
   UI's per-machine sub-cards should turn green within ~10s of
   coordinator boot (the heartbeat poller's first tick).

   Or query the coordinator's heartbeat-poller state directly:

   ```bash
   curl http://desktop-j05412i:5099/v2/healthz | jq .
   ```

4. **Run the cutover-day smoke script** before promoting the cluster
   to a real iter:

   ```bash
   PYTHONPATH=src .venv-py313/Scripts/python.exe deploy/smoke-test.py \
       --shards-config shards_config.json \
       --coordinator-url http://desktop-j05412i:5099 \
       --weights-version v6 \
       --games-per-worker 2
   ```

   The script POSTs a tiny synthetic stage (2 games, 32 nodes,
   1 sample) to each worker via the v2 REST contract, polls each
   stage to `complete` (or times out after `--timeout-s`, default
   300s), and reports pass/fail per worker plus a final
   `/v2/heartbeat` snapshot. Exit code is 0 iff every worker passes.

   Re-run after every daemon restart, coordinator restart, and on
   cutover day before promoting the cluster to a real iter.

5. **Run `mtg_bot iterate`** as usual. The orchestrator reads the
   URL list, builds the stage spec, POSTs `/v2/stage/create` on the
   coordinator, POSTs `/v2/start_self_play` on each worker, polls
   the stage to `complete`, then moves on to train + eval.

6. **Stop the cluster** with `systemctl stop`, `launchctl unload`,
   or `Stop-ScheduledTask`. Each daemon's `/v2/stop` endpoint is
   also available for graceful drain on individual workers.

## Worker-owned config

Each daemon's CLI flags are baked into the native service file's
invocation. The orchestrator NEVER overrides these — `n_workers`,
`deck_pool_path`, `weights_cache_root` are decided per-machine. See
`docs/DISTRIBUTED_API_v2.md` "Worker-owned config" section for the
rationale.

| Flag | Default | When to override |
|---|---|---|
| `--host` | `127.0.0.1` | Production: the machine's tailnet IP. |
| `--port` | `5198` | Only if 5198 is taken. |
| `--n-workers` | `cpu_count() - 2` | Tune per host: Mac 8-core → 6; Win10 6-core → 4; master 24-core → 16. |
| `--deck-pool-path` | `data/17lands/eoe_decks.json.gz` | Smoke tests against a small pool. |
| `--weights-cache-root` | `trained_weights` | Shared filesystem deployments where weights live elsewhere. |
| `--worker-id` | hostname | Two daemons on the same box; otherwise leave alone. |
| `--nn-device` | `cpu` | **GPU hosts only.** `cuda` (or `auto`) enables GPU NN-eval. See below. |
| `--spot-watch` | OFF | **AWS EC2 spot workers only.** See below. |

## AWS EC2 spot workers — `--spot-watch`

EC2 **spot** instances can be reclaimed by AWS with a ~2-minute
warning. By default a reclaimed worker just dies mid-game: its
in-flight claims then sit unreclaimed until the coordinator marks it
`DEAD` (~90s of missed heartbeats) or the slower claim-TTL sweep
fires (1200s default). That ~90s is wasted cluster time on every
spot reclaim.

`--spot-watch` (a worker-owned flag, OFF by default — leave it OFF on
on-prem workers and in tests) starts a daemon-thread poller that
watches the EC2 instance-metadata interruption endpoint (IMDSv2). On
the ~2-minute reclaim warning it triggers **the exact same graceful
drain as `POST /v2/stop`**: the worker stops claiming new games and
finishes/reports (or releases) its in-flight game locally, in
seconds. **Net effect: the ~90s heartbeat-DEAD reclaim lag is
removed from the common spot-reclaim case** — the coordinator gets
the games back almost immediately instead of waiting out the
liveness timeout.

**Enable it on the EC2 service file.** Add the flag to the daemon's
`ExecStart` (systemd is the EC2-Linux deployment vehicle — see the
"Linux (systemd)" section; the shipped
`deploy/systemd/mtg-bot-worker.service` has a commented `--spot-watch`
line ready to uncomment):

```ini
ExecStart=${VENV_PYTHON} -m mtg_bot.worker_daemon \
  --host ${TAILNET_IP} \
  --port 5198 \
  --n-workers ${N_WORKERS} \
  --deck-pool-path ${REPO_ROOT}/data/17lands/eoe_decks.json.gz \
  --weights-cache-root ${REPO_ROOT}/trained_weights \
  --spot-watch
```

If you bootstrap EC2 spot instances via **user-data** that templates
the unit (or runs the daemon directly), append `--spot-watch` to the
same invocation. No other config is needed — IMDSv2 is on by default
on EC2 and the endpoint is link-local; nothing leaves the instance.

### Operational notes / caveats

- **Win here is "stop claiming + faster LOCAL drain", not instant
  coordinator reclaim.** As of this feature the v2 coordinator has
  **no** proactive per-worker "reclaim my claims now" endpoint. A
  game still running when the instance finally vanishes falls back
  to the existing heartbeat / claim-TTL reclaim — same as today,
  just with no *new* claims piled on and the common
  finishes-in-time case fully drained. A coordinator-side
  instant-reclaim endpoint is possible future work (noted in
  `docs/DISTRIBUTED_API_v2.md`), deliberately out of scope here.
- **On-prem / non-AWS workers must NOT set `--spot-watch`.** Off
  AWS there is no IMDS at `169.254.169.254`; the poller would just
  log tolerated transport errors forever (harmless, but pointless
  noise). The flag is OFF by default precisely so this is opt-in.
- **A flaky IMDS call never kills the watcher or the daemon.**
  Transport/parse failures are logged and polling continues; the
  daemon's `/v2/heartbeat` stays responsive throughout. The watcher
  fires its drain **exactly once** and only on a real interruption
  notice — IMDS errors never spuriously drain a healthy worker.
- **Spot reclaim is still a reclaim.** `--spot-watch` makes the
  drain graceful and fast; it does not prevent the instance going
  away. A game that takes longer than AWS's ~2-minute warning to
  finish is still lost on that instance and re-queued by the
  coordinator's existing fallback — `--spot-watch` only guarantees
  no *new* work is claimed once the warning lands.

## GPU hosts — `--nn-device`

By default the worker daemon runs NN-eval on **CPU**
(`--nn-device cpu`, which maps internally to the historical
`device=None` the production runner has always used). This default
is deliberate and OFF — every non-GPU worker and the entire existing
test suite are byte-unaffected, exactly like `--spot-watch`.

On a host with an NVIDIA GPU, `--nn-device cuda` (or `--nn-device
auto`, which picks `cuda` → `mps` → `cpu` per host) moves NN-eval to
the GPU. **Validated A/B measured ≈2× per-game at production depth on
this path:** a clean back-to-back A/B on the production
structured-correct config path
(`worker_daemon_production_runner._build_nn_pimc_config`, production
depth `inner_nodes=200` / `samples=4` / structured v6, 14 workers,
same box, GPU-util-sampled) measured `device=cuda` ≈ **2.13× faster
per game** and ≈ **1.95× higher throughput** than `device=cpu`, with
the GPU genuinely used (42% mean / 93% peak util; the cpu arm's GPU
idle). Default stays OFF because per-worker CUDA-context init under
Windows-spawn can deadlock — enable it only on hosts you've confirmed
have a working CUDA torch build.

**This is PER-PROCESS eval.** Each subprocess-pool worker builds its
own `NNEvaluator` on the GPU. The batched **SHM eval server** (one
shared GPU evaluator fanned out to all workers) is a separate,
further optimization and is **not included here** — this change only
makes the existing per-process evaluator able to use the GPU.

**Correctness — why this is safe under spawn/fork.** The daemon
parent / claim-loop thread **never** calls `auto_device()` /
`torch.cuda.is_available()`. The selector is carried as a plain
string (`"auto"` / `"cuda"`) through `ClaimLoopConfig` into the
production runner and only resolved **inside the subprocess** in
`NNPimcConfig.build()` — the same proven deferred-sentinel pattern
the distributed shard runners and the `per_game_wall` bench use.
Priming the CUDA driver in the parent before the subprocess pool
spawns is exactly what breaks Windows-spawn (deadlock) /
Linux-fork ("Cannot re-initialize CUDA in forked subprocess"); the
sentinel pattern structurally avoids it.

**Enable it on the GPU host's service file.** Worker-owned config —
bake `--nn-device cuda` into the daemon's `ExecStart`, same place as
`--n-workers` / `--deck-pool-path`:

```ini
ExecStart=${VENV_PYTHON} -m mtg_bot.worker_daemon \
  --host ${TAILNET_IP} \
  --port 5198 \
  --n-workers ${N_WORKERS} \
  --deck-pool-path ${REPO_ROOT}/data/17lands/eoe_decks.json.gz \
  --weights-cache-root ${REPO_ROOT}/trained_weights \
  --nn-device cuda
```

(On Windows that's the Task-Scheduler action's argument list; on
Linux/systemd it's the unit's `ExecStart` as shown.) Leave the `cpu`
default on CPU-only workers and in tests. Smoke-check the host's
torch build first:
`python -c "import torch; print(torch.cuda.is_available())"`.

On Windows, do **not** hand-edit the task — re-run the sanctioned
installer with `-NnDevice`, which re-registers via the
Task-Scheduler API (no `schtasks` 261-char limit) and prompts once
for the account password via the standard `Get-Credential` dialog
(`Set-ScheduledTask` cannot modify a stored-password task without
the password; `schtasks /Change /TR` overflows 261 chars on the
repo's absolute paths):

```powershell
.\deploy\windows\install-mtg-bot-worker.ps1 `
  -TailnetIP <ip> -NWorkers <n> -RepoRoot <repo> -VenvPython <py> `
  -NnDevice cuda
# then: Start-ScheduledTask -TaskName MTGBotWorker  (when IDLE)
```

`-NnDevice` defaults to empty → the registered action is
byte-identical to the pre-GPU installer (no-op on CPU-only hosts).

## Self-update (`POST /v2/self_update`)

Push new code to a worker without SSHing in or touching the box.
The worker pulls the coordinator's existing `/repo.bundle` (no
GitHub creds — Tailscale only, so this works identically on the
master, Mac, Win10, and ephemeral AWS instances), `git reset
--hard`s to the ref, conditionally rebuilds the Rust engine + deps,
and re-execs in place.

Pre-req: the worker must be **IDLE** (mid-stage → 409; `/v2/stop`
first or wait for natural drain). Roll the whole fleet by iterating
the `worker_urls` in `shards_config.json`:

```bash
# one worker
curl -X POST http://<tailnet-ip>:5198/v2/self_update \
  -H 'Content-Type: application/json' \
  -d '{"coordinator_url":"http://desktop-j05412i:5099"}'

# verify it landed (git_commit advances; state returns to IDLE;
# a build failure shows state ERROR + last_update.error)
curl -s http://<tailnet-ip>:5198/v2/heartbeat | jq '{state,git_commit,last_update}'
```

`ref` is optional (default `main`) — pass e.g.
`{"...":"...","ref":"feature/x"}` to put a worker on a branch for a
one-off test (the coordinator's bundle is `--all`, so every branch
is fetchable).

**Operational caveats:**

- **Worker checkouts must stay pristine.** Self-update runs `git
  reset --hard <ref>` (leaving a detached HEAD at the target SHA)
  and deliberately does NOT `git clean`, so gitignored artifacts
  (`.venv-py313/`, `trained_weights/`, the deck pool) survive. Any
  *tracked-file* local drift on a worker IS discarded. Never hand-
  edit tracked files on a worker box; treat it as deploy-managed.
- **Conditional-rebuild heuristic.** A Rust rebuild fires only when
  `rust/`, `Cargo.toml`, `Cargo.lock`, or `pyproject.toml` changed
  between the old and target commit; deps reinstall only on
  `pyproject.toml`. This is a *path* heuristic — a dependency
  upgrade that lands without a `pyproject.toml` edit (e.g. a `pip
  install` done out-of-band, or a transitive bump) will NOT be
  picked up. Force a full rebuild by self-updating to a commit that
  touches `pyproject.toml`, or rebuild manually. `pyproject.toml`
  conservatively triggers BOTH a Rust and a deps rebuild.
- **Windows re-exec under Task Scheduler.** `os.execv` keeps the PID
  on POSIX (systemd/launchd never see the unit "exit" — clean,
  supervisor-independent). On Windows the CRT spawns a *replacement*
  process and exits the original; the `MTGBotWorker` Task Scheduler
  action tracks the original PID, so after a self-update the task
  instance reads as ended even though a healthy detached daemon is
  serving. The daemon still works (heartbeat green, claims served);
  the cosmetic mismatch clears on the next reboot / `Start-
  ScheduledTask`. POSIX workers (Mac, AWS) have no such caveat —
  prefer them as the self-update target where it matters. A
  follow-up could add a Windows-specific spawn+exit shim that
  re-registers the task; not done today.
- **Brief unavailability during re-exec.** <1s socket-rebind gap
  (werkzeug sets `SO_REUSEADDR`); invisible to the coordinator's
  STALE-tolerant (30s) heartbeat poller. A worker mid-self-update
  looks exactly like a normal process restart.

## Troubleshooting

| Symptom | Probable cause | Fix |
|---|---|---|
| `Connection refused` on `/v2/heartbeat` | daemon not running, or `--host` is `127.0.0.1` and you're hitting the tailnet IP | Check service status; restart with the tailnet IP. |
| Coordinator shows worker as STALE/DEAD | daemon process crashed; supervisor will restart on its own backoff | `journalctl -u mtg-bot-worker -f` / `~/Library/Logs/mtg-bot/worker.err.log` to find the exception. |
| `/v2/start_self_play` 409 `worker_busy` | previous stage didn't drain | `POST /v2/stop` to that worker, or wait for natural drain. |
| `/v2/self_update` 409 `worker_busy` | worker mid-stage or mid-update | `POST /v2/stop` then retry, or wait for natural drain. |
| Worker stuck `ERROR` after self-update, `last_update.error` set | bundle fetch / `git reset` / rebuild failed (detail in `error`) | Fix the cause (network, dirty tree, build break) then re-`POST /v2/self_update`; or `clear_error` path / restart the service. Code may already be on the new SHA but the process is on the old modules. |
| Self-update succeeded but Rust behaves like old code | conditional-rebuild heuristic skipped the rebuild (no `rust/`/`Cargo*`/`pyproject.toml` delta) | Self-update to a commit touching `pyproject.toml`, or rebuild manually (`maturin develop --release`). |
| Stage stuck at `running` | one worker has a stale claim past its TTL | Coordinator's heartbeat-driven re-queue (~90s) or claim TTL sweep (1200s default) will recover; check coordinator logs. |
