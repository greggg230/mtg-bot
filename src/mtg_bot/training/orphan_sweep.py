"""Orphan-process sweep for distributed self-play / iterate runs.

When the AlphaZero iterate-loop master is killed (taskkill /F /T,
ctrl-c that doesn't propagate, watchdog termination, an SSH yank
mid-game on a remote shard, etc.), the multiprocessing-spawn worker
children that ``ProcessPoolExecutor`` opened can survive their parent
and become orphans. On Windows they get re-parented to the System
process; on POSIX to init (PID 1). Either way they keep holding RAM
(50-700 MB each), keep accumulating CPU, and compete with the next
iteration's fresh workers, killing self-play throughput.

Today's session manually cleaned up 24 orphans on Win11, 141 on Mac,
and 4 on Win10. This module is the safety net so that mess doesn't
happen again: every ``iterate_alphazero.py`` startup sweeps for
prior-run orphans on the master AND on every remote in the shards
config before that remote's first dispatch.

## Matching rules — what counts as an orphan we should kill

We match conservatively. False positives (killing the user's
unrelated python — a Jupyter kernel, an editor LSP, a sibling agent's
test run) are a *serious* bug. So a process must satisfy ALL of:

1. Image basename is ``python``, ``python3``, ``python.exe``,
   ``pythonw.exe``, or ``python3.X[.exe]``.
2. Its full argv (joined with spaces, lowercased) contains at least
   one of:

   - ``run_self_play_shard.py``    (the shard wrapper)
   - ``iterate_alphazero.py``      (the master itself)
   - ``multiprocessing.spawn``     (Windows spawn launcher)
   - ``multiprocessing-fork``      (POSIX fork-launcher's argv tag)
   - ``from multiprocessing.resource_tracker``  (the auxiliary
     resource-tracker process MP spawns alongside its workers)

3. It is NOT the calling process, AND not in the calling process's
   descendant tree. We compute the descendant set up-front so a fresh
   master never sweeps its own children mid-launch.

4. (Optional, controlled by ``require_repo_root_match``.) When True
   (the local-master case), the argv must additionally reference the
   master's repo path. ``multiprocessing.spawn`` workers carry a
   ``--multiprocessing-fork tracker_fd=N`` argv that doesn't include
   the repo path — but their full command line on Windows starts with
   the venv python's absolute path, which IS under the repo. So we
   inspect both argv and the executable path for the substring.

   When False (the remote-shard case), we skip the repo-root check
   because we don't know the remote's repo path here. Instead we
   rely on the argv pattern + the fact that no other long-running
   pyproject on the remote would match these strings. If the user
   has another mtg-bot checkout running on the remote, that's a known
   limitation — see ``docs/SEARCH_LIMITATIONS.md``.

The matcher is in ``_argv_matches`` for unit-testing.

## Process inspection — no psutil dep

psutil is not in ``pyproject.toml`` and we don't want to add a binary
dep just for a sweep that runs ~once per iterate launch. So we
shell out:

- **Windows local:** PowerShell ``Get-CimInstance Win32_Process``
  returns ``ProcessId``, ``ParentProcessId``, ``ExecutablePath``,
  ``CommandLine``. Reliable, no admin needed for processes we own.
  (``wmic`` is deprecated and absent on Win11 builds without the
  feature pack.)
- **POSIX local:** ``ps -e -o pid=,ppid=,comm=,args=`` is portable
  across macOS and Linux.
- **Remote sweep:** SSH a one-shot Python invocation that imports
  this module and runs ``sweep_local_orphans``. Lets us share the
  matcher logic instead of re-implementing it in shell.

## What this module does NOT do

- It does not kill processes a level beyond pattern match — no
  "kill anything that looks like Python and uses memory > X". The
  matcher is the only authority on what dies.
- It does not retry forever. One ``terminate()``, then a short
  ``kill()`` if needed, then move on. A wedged orphan that survives
  both is logged and left alone — better to surface that than to
  wedge the master's startup.
- It does not require admin / root. If a process is unkillable
  because it's owned by another user, we log and skip.

Public API:

- ``sweep_local_orphans(*, repo_root, label="local") -> list[int]``
- ``sweep_remote_orphans(shard, *, label=None) -> list[int]``
- ``find_orphan_pids_local(*, repo_root, self_pid, descendant_pids,
  process_listing) -> list[int]`` — pure function, drives the unit
  tests.
- ``_argv_matches(argv, exe_path, repo_root, require_repo_root_match)``
  — pure function, drives the matcher tests.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Argv patterns that mark a process as a self-play / iterate worker.
# Match is case-insensitive (Windows is case-insensitive on filesystem;
# we want consistency).
_ARGV_PATTERNS: tuple[str, ...] = (
    "run_self_play_shard.py",
    "iterate_alphazero.py",
    "multiprocessing.spawn",
    "multiprocessing-fork",
    "from multiprocessing.resource_tracker",
)

# Image basenames that count as Python.
_PYTHON_BASENAMES: tuple[str, ...] = (
    "python", "python3", "python.exe", "pythonw.exe",
)


@dataclass(frozen=True)
class ProcessInfo:
    """A snapshot row from the platform's process list. ``args`` is
    the full command line as a single string (we don't try to
    re-tokenize — Windows's ``Get-CimInstance`` returns it that way
    and re-tokenization is lossy). ``exe`` is the executable path
    (may be empty / unknown on POSIX ``ps``)."""

    pid: int
    ppid: int
    exe: str
    args: str

    def to_dict(self) -> dict:
        return {
            "pid": self.pid, "ppid": self.ppid,
            "exe": self.exe, "args": self.args,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessInfo":
        return cls(
            pid=int(d["pid"]),
            ppid=int(d.get("ppid", 0)),
            exe=str(d.get("exe", "")),
            args=str(d.get("args", "")),
        )


def _python_basename(exe: str) -> str:
    """Return the lowercased basename of ``exe`` if it looks like a
    Python interpreter, else ``""``. Handles versioned names like
    ``python3.13`` and ``python3.13.exe``. Empty exe → empty string."""
    if not exe:
        return ""
    base = Path(exe).name.lower()
    if base in _PYTHON_BASENAMES:
        return base
    # python3.X[.exe]
    if base.startswith("python") and (
        base.endswith(".exe") or "." in base or base == "python"
    ):
        # Extra-permissive: any "python..." prefix counts. We're only
        # using this as a coarse first filter; the argv match is what
        # actually narrows the kill set.
        return base
    return ""


def _argv_matches(
    argv: str,
    exe: str,
    *,
    repo_root: Optional[str],
    require_repo_root_match: bool,
) -> bool:
    """Pure pattern-match decision: does this argv+exe pair look
    like a self-play / iterate orphan we should consider killing?

    See module docstring for the full rule list. Case-insensitive
    substring search throughout.

    ``repo_root`` is the absolute path to the master's repo working
    tree. ``require_repo_root_match=True`` adds the constraint that
    argv or exe must reference that path — used on the local sweep
    to defend against killing an unrelated Python that happens to
    say ``multiprocessing.spawn``. The remote sweep passes False
    because we don't know the remote's repo path here.
    """
    argv_l = argv.lower()
    exe_l = (exe or "").lower()
    if not any(p in argv_l for p in _ARGV_PATTERNS):
        return False
    if require_repo_root_match:
        if not repo_root:
            # Caller asked for repo-root match but didn't supply one
            # — refuse to match anything (fail-safe).
            return False
        rr = repo_root.lower().replace("\\", "/")
        # Compare with both separator forms — Windows argvs may
        # contain either. Normalize to forward-slash for the haystack.
        argv_norm = argv_l.replace("\\", "/")
        exe_norm = exe_l.replace("\\", "/")
        if rr not in argv_norm and rr not in exe_norm:
            return False
    return True


def _list_processes_local() -> list[ProcessInfo]:
    """Return one row per visible process with its pid, ppid, exe,
    and full argv. Empty list on transient failure (we'd rather skip
    a sweep than crash the master's startup).
    """
    if sys.platform == "win32":
        return _list_processes_windows()
    return _list_processes_posix()


def _list_processes_windows() -> list[ProcessInfo]:
    """PowerShell ``Get-CimInstance Win32_Process``. Returns JSON the
    parent reads. The CIM provider doesn't need admin for processes
    the user owns. Field set: ProcessId, ParentProcessId, Name,
    ExecutablePath, CommandLine.
    """
    ps_script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,"
        "ExecutablePath,CommandLine | "
        "ConvertTo-Json -Compress -Depth 2"
    )
    try:
        cp = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", ps_script,
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    try:
        # ``strict=False`` permits literal control characters (0x00..0x1F)
        # inside JSON strings, per RFC 8259 leniency. Real-world
        # ``Get-CimInstance Win32_Process`` output on Win11 has been
        # observed with at least one process whose ``CommandLine`` field
        # carries a stray control char (e.g. ``\x01``) — strict-mode
        # parsing rejects the entire ~160 KB document, which would
        # otherwise leave us silently returning ``[]`` and skipping the
        # whole sweep. Best-effort listing → permit the leniency.
        raw = json.loads(cp.stdout, strict=False)
    except json.JSONDecodeError:
        return []
    # Single result returns a dict, multiple returns a list.
    if isinstance(raw, dict):
        raw = [raw]
    out: list[ProcessInfo] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        pid = row.get("ProcessId")
        ppid = row.get("ParentProcessId") or 0
        if pid is None:
            continue
        try:
            pid_i = int(pid)
            ppid_i = int(ppid)
        except (TypeError, ValueError):
            continue
        exe = row.get("ExecutablePath") or row.get("Name") or ""
        cmdline = row.get("CommandLine") or ""
        out.append(ProcessInfo(
            pid=pid_i, ppid=ppid_i, exe=str(exe), args=str(cmdline),
        ))
    return out


def _list_processes_posix() -> list[ProcessInfo]:
    """``ps -e -o pid=,ppid=,comm=,args=``. Portable to macOS + Linux.
    ``args`` is the full original argv joined with spaces; ``comm`` is
    the image basename (we use it as the ``exe`` proxy since ``ps``'s
    ``args`` always starts with the same).
    """
    try:
        cp = subprocess.run(
            ["ps", "-e", "-o", "pid=,ppid=,comm=,args="],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if cp.returncode != 0:
        return []
    out: list[ProcessInfo] = []
    for line in cp.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # First three whitespace-separated tokens are pid, ppid, comm;
        # rest is args.
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid_i = int(parts[0])
            ppid_i = int(parts[1])
        except ValueError:
            continue
        comm = parts[2]
        args = parts[3] if len(parts) >= 4 else comm
        out.append(ProcessInfo(
            pid=pid_i, ppid=ppid_i, exe=comm, args=args,
        ))
    return out


def _build_descendant_set(
    rows: Iterable[ProcessInfo], root_pid: int,
) -> set[int]:
    """BFS the parent→child graph implied by ``rows`` starting from
    ``root_pid``. Returns the set of every descendant PID
    (NOT including ``root_pid`` itself). Used to make sure the master's
    own multiprocessing children aren't on the kill list during its
    startup sweep.
    """
    by_ppid: dict[int, list[int]] = {}
    for r in rows:
        by_ppid.setdefault(r.ppid, []).append(r.pid)
    out: set[int] = set()
    frontier: list[int] = list(by_ppid.get(root_pid, []))
    while frontier:
        pid = frontier.pop()
        if pid in out or pid == root_pid:
            continue
        out.add(pid)
        frontier.extend(by_ppid.get(pid, []))
    return out


def find_orphan_pids_local(
    *,
    repo_root: Optional[str],
    self_pid: int,
    descendant_pids: set[int],
    process_listing: Iterable[ProcessInfo],
    require_repo_root_match: bool = True,
) -> list[int]:
    """Pure function: given a process snapshot, return the PIDs that
    match the orphan-sweep rules. Drives the unit tests.

    The kill decision is split out from the platform-specific
    listing/killing so the matcher can be exercised with synthetic
    rows.
    """
    out: list[int] = []
    for row in process_listing:
        if row.pid == self_pid or row.pid in descendant_pids:
            continue
        if not _python_basename(row.exe) and not _python_basename_in_args(
            row.args,
        ):
            # If exe is empty AND argv doesn't start with python, skip.
            # This protects against ps rows where comm is truncated.
            continue
        if not _argv_matches(
            row.args, row.exe,
            repo_root=repo_root,
            require_repo_root_match=require_repo_root_match,
        ):
            continue
        out.append(row.pid)
    return sorted(out)


def _python_basename_in_args(args: str) -> bool:
    """Cheap fallback when ``exe`` is empty: the first token of args
    is typically the executable. Returns True if that token's basename
    looks like a Python interpreter.
    """
    if not args:
        return False
    try:
        first = args.split(None, 1)[0]
    except IndexError:
        return False
    return bool(_python_basename(first))


def _terminate_pid(pid: int, *, kill_grace_s: float = 2.0) -> bool:
    """Best-effort terminate then kill of a single PID. Returns True
    if the process is gone (or wasn't there), False if it survived
    both.

    On Windows we shell out to ``taskkill`` because ``os.kill`` doesn't
    exist for arbitrary signals there. ``taskkill /F /PID`` is
    equivalent to ``TerminateProcess`` and bypasses the per-process
    cleanup we don't trust to run anyway. POSIX gets a SIGTERM, then
    SIGKILL if it's still around after ``kill_grace_s``.
    """
    if sys.platform == "win32":
        try:
            cp = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        # taskkill returns 0 on success, 128 on "not found", 1 on
        # access denied. Treat 128 as "already gone" = success.
        return cp.returncode in (0, 128)
    # POSIX
    try:
        os.kill(pid, 15)  # SIGTERM
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    deadline = time.monotonic() + kill_grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # probe
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.05)
    try:
        os.kill(pid, 9)  # SIGKILL
    except ProcessLookupError:
        return True
    except OSError:
        return False
    # Final probe.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def kill_pids(pids: Iterable[int]) -> list[int]:
    """Terminate every PID in ``pids`` and return the subset that's
    confirmed gone. Best-effort — wedged PIDs are silently skipped.
    """
    pids_list = list(dict.fromkeys(int(p) for p in pids))
    killed: list[int] = []
    for pid in pids_list:
        if _terminate_pid(pid):
            killed.append(pid)
    return killed


def sweep_local_orphans(
    *,
    repo_root: str,
    label: str = "local",
    print_fn=None,
) -> list[int]:
    """Run the full sweep on the local machine. Returns the PIDs we
    confirmed killed.

    ``repo_root`` is the master's repo working tree; required so the
    matcher can reject Python processes that don't reference the
    project. ``label`` shows up in the printed message.
    ``print_fn`` lets tests capture output; defaults to printing to
    stdout. Loud is good — silent is bad.
    """
    pf = print_fn or (lambda s: print(s, flush=True))
    rows = _list_processes_local()
    if not rows:
        pf(f"[sweep] {label}: process listing failed; skipping sweep")
        return []
    self_pid = os.getpid()
    descendants = _build_descendant_set(rows, self_pid)
    candidates = find_orphan_pids_local(
        repo_root=repo_root,
        self_pid=self_pid,
        descendant_pids=descendants,
        process_listing=rows,
        require_repo_root_match=True,
    )
    if not candidates:
        pf(f"[sweep] {label}: no orphan processes found")
        return []
    pf(
        f"[sweep] {label}: found {len(candidates)} orphan candidate "
        f"PIDs: {candidates}"
    )
    killed = kill_pids(candidates)
    survivors = [p for p in candidates if p not in killed]
    pf(
        f"[sweep] killed {len(killed)} orphan processes on {label}: "
        f"{killed}"
    )
    if survivors:
        pf(
            f"[sweep] {label}: WARNING {len(survivors)} candidate "
            f"PIDs survived terminate+kill: {survivors}"
        )
    return killed


def _build_remote_sweep_command(shard) -> str:
    """Emit the one-line shell command the remote runs to sweep its
    own orphans. Imports this module via ``-c`` and calls
    ``sweep_local_orphans``. Uses ``require_repo_root_match=False``
    on the remote because the remote's argv may use a different
    absolute path layout than the master's.
    """
    repo_path = shard.repo_path or ""
    venv_python = shard.venv_python or ""
    # The remote sweep deliberately uses require_repo_root_match=False:
    # we don't know whether the remote has multiple checkouts and the
    # argv pattern is already a strong signal there. The remote also
    # doesn't have a competing Jupyter/editor running self-play
    # patterns — see module docstring's limitations note.
    py = (
        "import sys; "
        "from mtg_bot.training.orphan_sweep import sweep_local_orphans; "
        f"sweep_local_orphans(repo_root={repo_path!r}, label='remote')"
    )
    is_windows = (venv_python or "").endswith(".exe")
    if is_windows:
        py_q = py.replace('"', '\\"')
        return (
            f'cd /d "{repo_path}" && '
            f'set PYTHONPATH=src && '
            f'"{venv_python}" -c "{py_q}"'
        )
    return (
        f'cd {shlex.quote(repo_path)} && '
        f'PYTHONPATH=src {shlex.quote(venv_python)} '
        f'-c {shlex.quote(py)}'
    )


def sweep_remote_orphans(
    shard,
    *,
    ssh_cmd_factory=None,
    run_subprocess=None,
    print_fn=None,
) -> list[int]:
    """SSH a sweep to the remote. Returns the PIDs the remote killed
    (parsed from its stdout). Best-effort — a failed sweep prints a
    warning and returns []. We do NOT raise: a flaky sweep should
    never block the master's iter from launching.

    ``ssh_cmd_factory(shard, remote_command) -> list[str]`` and
    ``run_subprocess(cmd, *, timeout)`` are injection points for
    testing — production callers leave them as None and the
    distributed-module versions are imported lazily.
    """
    pf = print_fn or (lambda s: print(s, flush=True))
    if ssh_cmd_factory is None or run_subprocess is None:
        # Lazy import to avoid a circular dependency at module load
        # time (distributed.py imports things from training/__init__).
        from mtg_bot.training.distributed import (
            _ssh_cmd as _default_ssh_cmd,
            _run_subprocess as _default_run,
        )
        if ssh_cmd_factory is None:
            ssh_cmd_factory = _default_ssh_cmd
        if run_subprocess is None:
            run_subprocess = _default_run
    label = getattr(shard, "display_label", None) or getattr(
        shard, "host", "remote",
    )
    sweep_cmd = _build_remote_sweep_command(shard)
    try:
        cp = run_subprocess(
            ssh_cmd_factory(shard, sweep_cmd), timeout=120,
        )
    except subprocess.CalledProcessError as e:
        pf(
            f"[sweep] {label}: remote sweep failed (exit "
            f"{e.returncode}): {(e.stderr or '').strip()[:200]}"
        )
        return []
    except (subprocess.TimeoutExpired, OSError) as e:
        pf(
            f"[sweep] {label}: remote sweep error: "
            f"{type(e).__name__}: {e}"
        )
        return []
    stdout = (cp.stdout or "").strip()
    if stdout:
        for line in stdout.splitlines():
            pf(f"[sweep] {label} (remote): {line}")
    return _parse_killed_pids(stdout)


def _parse_killed_pids(stdout: str) -> list[int]:
    """Parse the ``[sweep] killed N orphan processes on <label>: [pids]``
    line emitted by ``sweep_local_orphans``. Returns the PID list, or
    empty if the line is absent / malformed. Tolerant to extra noise
    lines around it.

    Looks for the LAST ``[...]`` pair on the line (the PID list),
    since the line itself starts with ``[sweep]`` which would otherwise
    confuse a naive find/find pair.
    """
    out: list[int] = []
    for line in (stdout or "").splitlines():
        # Match the killed line specifically. The "found N candidates"
        # line uses the same brackets but a different verb.
        if "killed" not in line or "orphan processes" not in line:
            continue
        rb = line.rfind("]")
        if rb < 0:
            continue
        lb = line.rfind("[", 0, rb)
        if lb < 0:
            continue
        inner = line[lb + 1 : rb].strip()
        if not inner:
            return []
        for tok in inner.split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
    return out
