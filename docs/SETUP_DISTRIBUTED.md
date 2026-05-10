# Distributed self-play — per-machine setup

How to bring a remote machine online for `iterate_alphazero
--shards-config <path>`. One-time work per machine; afterwards
`git pull` (run automatically by the orchestrator before each iter)
keeps the code current.

The two target machines today are:
- **Win10 + 1080Ti box** — Windows 10, OpenSSH server, CPU-only
  torch install (the 1080Ti's CUDA isn't useful for self-play
  workers; they don't instantiate CUDA tensors).
- **MacBook Pro M-series + 48 GB RAM** — macOS, Apple Silicon
  native PyTorch wheels, Remote Login (built-in SSH server).

## Quick path: serve setup scripts from the master

The master can host the per-OS install scripts on a tiny Tailscale-
accessible HTTP server, with the master's repo URL + ssh public key
already templated in. Each remote then runs a single one-liner — no
manual scp / env vars / pubkey paste.

**On the master**, once:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli serve
```

(Default port 5099. Auto-detects the master's tailnet name, the
repo's git remote, and the user's `~/.ssh/id_ed25519.pub`.
``scripts/serve_setup_scripts.py`` still works as a legacy shim.)

**On each remote** (one-time):

```bash
# macOS: in Terminal
curl -sSL http://<master-tailnet-name>:5099/setup_remote.sh | bash

# Windows: in elevated PowerShell
iex (iwr http://<master-tailnet-name>:5099/setup_remote.ps1).Content
```

What that does:
- Installs Python 3.13 (Homebrew on Mac, winget on Windows) if missing.
- `git clone` the repo to `~/projects/mtg-bot` (Mac) or
  `%USERPROFILE%\Projects\mtg-bot` (Windows).
- Creates `.venv-py313`, `pip install -e .[dev]`, plus PyTorch
  (native ARM on Mac, CPU-only on Windows).
- Appends the master's public key to `~/.ssh/authorized_keys` with
  the right permissions.
- Windows only: enables OpenSSH Server, opens the firewall, sets
  the right ACL on `authorized_keys` (silent reject without it).
- Runs the distributed-self-play tests as a smoke check.

Two manual remaining steps per remote:
1. **Install + log in to Tailscale** (browser auth flow can't be
   scripted). Mac: `brew install --cask tailscale && open
   /Applications/Tailscale.app`. Windows: download from
   https://tailscale.com/download/windows.
2. **Smoke test SSH** from the master:
   `ssh <user>@<remote-tailnet-name> "echo ok"` — should work without
   prompting.

Then back on the master, push the bootstrap data files (one-time
per remote, plus once per `current_best` advancement):

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe scripts/push_master_data.py \
    --shards-config shards_config.json --shard <remote-name>
# OR explicit:
PYTHONPATH=src .venv-py313/Scripts/python.exe scripts/push_master_data.py \
    --host <remote-tailnet-name> --user <u> --repo-path <p>
```

That copies the bootstrap version's weights + the EOE deck pool to
the remote's repo (matching relative paths, so `iterate` finds them).

The long-form below is the manual path — useful for debugging when
the quick path doesn't apply (no Tailscale yet, scripts misbehaving,
etc.). Both go through the same logical steps with OS-specific
commands called out where they differ.

---

## 0. Pre-reqs on the master

Before touching a remote, make sure the master has:

- **An SSH key pair** — `ssh-keygen -t ed25519 -C "mtg-bot orchestrator"`.
  Default path is `~/.ssh/id_ed25519` on Windows + Mac. The public
  half (`~/.ssh/id_ed25519.pub`) is what you'll add to each remote.
- **Tailscale**, already running. Each remote will join the same
  tailnet so SSH targets resolve via MagicDNS (e.g.
  `ssh user@win-1080ti`).

---

## 1. Per-machine Python + repo + venv

### Win10 + 1080Ti box

1. **Install Python 3.13** (official installer from python.org).
   Tick "Add Python to PATH" during install.
2. **Clone the repo** somewhere stable, e.g.
   `C:\Users\<you>\Projects\mtg-bot`:
   ```cmd
   cd C:\Users\<you>\Projects
   git clone https://github.com/<your-fork>/mtg-bot.git
   cd mtg-bot
   ```
3. **Create the venv + install deps**:
   ```cmd
   py -3.13 -m venv .venv-py313
   .venv-py313\Scripts\python.exe -m pip install -e .[dev]
   ```
4. **Install torch CPU build** (skip CUDA — workers don't use it):
   ```cmd
   .venv-py313\Scripts\python.exe -m pip install torch ^
       --index-url https://download.pytorch.org/whl/cpu
   ```
5. **Copy bootstrap weights + deck pool** from the master.
   From master cmd / PowerShell:
   ```powershell
   scp trained_weights\v6\weights.pt `
       <user>@win-1080ti:C:/Users/<you>/Projects/mtg-bot/trained_weights/v6/weights.pt
   scp trained_weights\v6\metadata.json `
       <user>@win-1080ti:C:/Users/<you>/Projects/mtg-bot/trained_weights/v6/metadata.json
   scp data\17lands\eoe_decks.json.gz `
       <user>@win-1080ti:C:/Users/<you>/Projects/mtg-bot/data/17lands/eoe_decks.json.gz
   ```
   (You may need to `mkdir trained_weights/v6` and
   `mkdir data/17lands` on the remote first.)

### MacBook Pro M-series

1. **Install Python 3.13**:
   ```bash
   brew install python@3.13
   ```
   (Or grab the official installer if you'd rather not use brew.)
2. **Clone the repo**, e.g. to `~/projects/mtg-bot`:
   ```bash
   cd ~/projects
   git clone https://github.com/<your-fork>/mtg-bot.git
   cd mtg-bot
   ```
3. **Create the venv + install deps + torch (native ARM)**:
   ```bash
   python3.13 -m venv .venv-py313
   .venv-py313/bin/pip install -e .[dev]
   .venv-py313/bin/pip install torch
   ```
4. **Copy bootstrap weights + deck pool** from the master:
   ```powershell
   scp trained_weights\v6\weights.pt `
       <user>@macbook:/Users/<you>/projects/mtg-bot/trained_weights/v6/weights.pt
   scp trained_weights\v6\metadata.json `
       <user>@macbook:/Users/<you>/projects/mtg-bot/trained_weights/v6/metadata.json
   scp data\17lands\eoe_decks.json.gz `
       <user>@macbook:/Users/<you>/projects/mtg-bot/data/17lands/eoe_decks.json.gz
   ```

### Smoke test the install (each machine)

SSH into the machine, then:

```bash
cd <repo path>
PYTHONPATH=src .venv-py313/bin/python -c "import torch, mtg_bot; print(torch.__version__)"
PYTHONPATH=src .venv-py313/bin/python -m pytest tests/training/test_distributed.py --no-cov -q
```

(Substitute `.venv-py313\Scripts\python.exe` and `set PYTHONPATH=src`
on Windows.) Expect 24 passing distributed tests.

---

## 2. SSH server on each remote

### Win10 + 1080Ti box

1. **Settings → Apps → Optional Features → Add a feature →
   OpenSSH Server**.
2. Open PowerShell **as administrator** and start the service:
   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' `
     -Enabled True -Direction Inbound -Protocol TCP -Action Allow `
     -LocalPort 22
   ```
3. **Add the master's public key** to
   `C:\Users\<you>\.ssh\authorized_keys` (create the file + dir if
   missing). One line per key.
4. **Permissions matter on Windows OpenSSH** — the
   `authorized_keys` file must be owned by your user only. From
   PowerShell:
   ```powershell
   icacls $env:USERPROFILE\.ssh\authorized_keys /inheritance:r
   icacls $env:USERPROFILE\.ssh\authorized_keys /grant:r "$env:USERNAME:(F)"
   ```
   Without this OpenSSH silently rejects the key.

### MacBook Pro M-series

1. **System Settings → General → Sharing → Remote Login** → enable.
2. **Add the master's public key** to
   `~/.ssh/authorized_keys` (create the file + dir if missing).
   Permissions:
   ```bash
   mkdir -p ~/.ssh && chmod 700 ~/.ssh
   touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
   ```

### Tailscale on each remote

1. Install Tailscale (`brew install tailscale` on Mac; the official
   installer on Windows).
2. Run `tailscale up` and authenticate.
3. Note the tailnet hostname (`tailscale status` shows it). You'll
   use this as the SSH target.

### Smoke test SSH from the master

```bash
ssh <user>@<remote-tailnet-name> "echo ok"
```

Should print `ok` without prompting for a password. If prompted,
the key didn't take — re-check the `authorized_keys` location +
permissions.

---

## 3. Shards-config JSON

Drop a file like `shards_config.json` in the master's repo root
(or wherever convenient — pass the path to `--shards-config`):

```json
{
  "shards": [
    {
      "host": "local"
    },
    {
      "host": "win-1080ti",
      "user": "<you>",
      "repo_path": "C:\\Users\\<you>\\Projects\\mtg-bot",
      "venv_python": ".venv-py313\\Scripts\\python.exe"
    },
    {
      "host": "macbook",
      "user": "<you>",
      "repo_path": "/Users/<you>/projects/mtg-bot",
      "venv_python": ".venv-py313/bin/python"
    }
  ]
}
```

Notes:
- **`host`** is the SSH target — Tailscale magic-DNS hostname,
  bare IP, or `local` for the master. All three are fine.
- **Total games per iter** comes from `iterate_alphazero.py`'s
  `--games-per-iteration` (every shard pulls from one shared
  queue; slow machines pull fewer, fast ones pull more — there
  is no per-shard quota in this file).
- **`repo_path`** is the absolute path to the remote's repo
  working tree. The orchestrator does `cd` into this before
  running anything.
- **`venv_python`** is relative to `repo_path`, OS-specific
  (`.venv-py313\\Scripts\\python.exe` on Windows,
  `.venv-py313/bin/python` on macOS / Linux). The orchestrator
  detects Windows by the `.exe` suffix and switches to cmd.exe
  command syntax.
- **`seed_offset`** is auto-assigned as `position * 1_000_000`
  when omitted. Override if you want explicit per-shard seeds.
- **`ssh_port`** defaults to 22 — set if your remote runs sshd on
  a non-standard port.
- A legacy **`n_games`** field on any shard entry is rejected with
  a clear error pointing at `--games-per-iteration` — drop it from
  any older config files.

---

## 4. End-to-end smoke

### Lightweight: 6 games × 3 shards (~5 min)

Pass a small `--games-per-iteration` (e.g. 6) and let the three
shards drain it from the shared queue:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe \
  -m mtg_bot.cli iterate \
  --bootstrap-version v6 --n-iterations 1 \
  --games-per-iteration 6 \
  --epochs 5 --batch-size 256 --learning-rate 1e-3 \
  --acceptance-threshold 0.55 --eval-games 4 \
  --enable-mulligans --per-game-timeout-sec 450 \
  --workers 24 --seed 0 \
  --deck-pool data/17lands/eoe_decks.json.gz \
  --data-dir iterate_data \
  --shards-config shards_smoke.json
```

Expect:
- Three per-shard cards appear in the training UI at port 5003,
  each labeled `iter_001_self_play_v6_shard_<label>`.
- Each shard finishes (status="completed").
- An npz lands in `iterate_data/` per shard.
- The training step runs after all shards complete.
- iter_001 finalizes — registers a new model version.

### Full run

Once the smoke is clean, kick off the real iteration:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe \
  -m mtg_bot.cli iterate \
  --bootstrap-version v6 --n-iterations 5 \
  --games-per-iteration 500 \
  --epochs 100 --batch-size 256 --learning-rate 1e-3 \
  --acceptance-threshold 0.55 --eval-games 200 \
  --enable-mulligans --per-game-timeout-sec 450 \
  --workers 24 --seed 0 \
  --deck-pool data/17lands/eoe_decks.json.gz \
  --data-dir iterate_data \
  --shards-config shards_config.json
```

`--games-per-iteration` is the single source of truth for the
iter's total game count: every shard (local + remote) pulls from
the master's shared queue until it drains.

---

## 5. Common pitfalls

- **`Permission denied (publickey)`** — almost always a permissions
  issue on the remote's `authorized_keys`. On Windows, run the
  `icacls` lines above. On Mac, `chmod 700 ~/.ssh && chmod 600
  ~/.ssh/authorized_keys`.
- **Hostname doesn't resolve** — `tailscale status` should list
  the remote. If MagicDNS is off (or the remote isn't logged in),
  use the bare Tailscale IP instead.
- **`scp` writes the wrong path** — the WIN-side path needs forward
  slashes after the colon (`scp file user@host:C:/Users/...`),
  even though Windows uses backslashes natively. cmd.exe accepts
  forward slashes for filesystem operations.
- **Remote's repo is dirty** — the orchestrator's `git pull` will
  refuse on uncommitted changes. SSH in and `git stash` (or
  commit / discard) on the remote.
- **Weights / decks missing on the remote** — initial sync is
  manual (see step 1.5). The orchestrator does NOT (yet) ship
  weights when `current_best` advances; once iterate produces a
  v7+ you'll need to scp those before the next iter or use
  `--bootstrap-version v6` so the remote can rely on the static
  set already there.
