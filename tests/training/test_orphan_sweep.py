"""Tests for ``mtg_bot.training.orphan_sweep``.

Covers:

- ``_argv_matches`` matcher rules: each positive pattern, repo-root
  filtering, case-insensitivity, exclusions.
- ``find_orphan_pids_local`` with synthetic process listings:
  PPID=1 orphans get killed, the master's own descendants don't,
  unrelated Python (Jupyter, editor LSP) doesn't.
- ``_build_descendant_set`` for the descendant-tree walk.
- ``kill_pids`` happy path + injected failures (mocking
  ``_terminate_pid``).
- ``sweep_remote_orphans`` SSH happy path (parses killed line) and
  failure path (CalledProcessError) — uses dependency injection.
- ``_parse_killed_pids`` line parsing: present, malformed, multiple,
  zero killed.
- The killed-line emission contract from ``sweep_local_orphans``
  (loud — silent is bad).

Tests don't actually spawn or kill OS processes (would race with the
test runner's own xdist workers and is platform-specific). The
matcher and orchestration are pure-Python and exercise the kill
boundary via the ``_terminate_pid`` injection point.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from mtg_bot.training.orphan_sweep import (
    ProcessInfo,
    _argv_matches,
    _build_descendant_set,
    _parse_killed_pids,
    _python_basename,
    find_orphan_pids_local,
    kill_pids,
    sweep_local_orphans,
    sweep_remote_orphans,
)


# ---- _python_basename ---------------------------------------------

def test_python_basename_recognizes_common_names() -> None:
    assert _python_basename("/usr/bin/python") == "python"
    assert _python_basename("/usr/bin/python3") == "python3"
    assert _python_basename("/usr/bin/python3.13") == "python3.13"
    assert _python_basename("C:\\venv\\Scripts\\python.exe") == "python.exe"
    assert _python_basename("C:\\venv\\Scripts\\pythonw.exe") == "pythonw.exe"
    # case-insensitive match
    assert _python_basename("C:\\venv\\Scripts\\Python.exe") == "python.exe"


def test_python_basename_rejects_unrelated() -> None:
    assert _python_basename("/usr/bin/bash") == ""
    assert _python_basename("C:\\Windows\\notepad.exe") == ""
    assert _python_basename("") == ""


# ---- _argv_matches ------------------------------------------------

REPO = "C:/Users/Gregg/projects/mtg-bot"


def test_argv_matches_run_self_play_shard() -> None:
    argv = (
        f"{REPO}/.venv-py313/Scripts/python.exe "
        f"scripts/run_self_play_shard.py --mode=pull --spec-b64=AAA"
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_iterate_alphazero() -> None:
    argv = (
        f"{REPO}/.venv-py313/Scripts/python.exe "
        f"scripts/iterate_alphazero.py --bootstrap v4"
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_multiprocessing_spawn() -> None:
    # Synthetic but representative of what mp.spawn workers look like
    # on Windows: argv references the venv python (under repo) and
    # carries ``multiprocessing.spawn``.
    argv = (
        f"{REPO}/.venv-py313/Scripts/python.exe -c "
        f"\"from multiprocessing.spawn import spawn_main; ...\""
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_multiprocessing_fork() -> None:
    # POSIX fork-launcher path: argv contains "multiprocessing-fork"
    # which is what concurrent.futures uses on Linux.
    argv = (
        f"{REPO}/.venv-py313/bin/python -c "
        f"from multiprocessing-fork import...; tracker_fd=12"
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_resource_tracker() -> None:
    argv = (
        f"{REPO}/.venv/bin/python -c "
        f"\"from multiprocessing.resource_tracker import main\""
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_case_insensitive() -> None:
    # Windows returns paths with mixed casing sometimes.
    argv = (
        f"{REPO.upper()}/.venv-py313/Scripts/python.exe "
        f"scripts/RUN_SELF_PLAY_SHARD.py"
    )
    assert _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_rejects_no_pattern() -> None:
    argv = f"{REPO}/.venv/Scripts/python.exe -m pytest tests/"
    assert not _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_rejects_pattern_outside_repo() -> None:
    """Repo-root match required, argv references mp.spawn but in a
    different repo. Must NOT match (defends against killing a
    sibling Jupyter or unrelated Python that imports
    multiprocessing).
    """
    argv = (
        "C:/Users/Gregg/projects/some-other-tool/.venv/Scripts/python.exe "
        "-c \"from multiprocessing.spawn import spawn_main\""
    )
    assert not _argv_matches(
        argv, "", repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_repo_match_via_exe() -> None:
    """argv may not contain the repo path (mp.spawn workers' argv is
    short), but exe will. Match should succeed via exe."""
    argv = (
        '"C:\\Users\\Gregg\\projects\\mtg-bot\\.venv-py313\\Scripts\\python.exe" '
        '-c "from multiprocessing.spawn import spawn_main; spawn_main()"'
    )
    exe = "C:\\Users\\Gregg\\projects\\mtg-bot\\.venv-py313\\Scripts\\python.exe"
    assert _argv_matches(
        argv, exe, repo_root=REPO, require_repo_root_match=True,
    )


def test_argv_matches_remote_skips_repo_match() -> None:
    """On the remote, we don't know the repo path, so
    require_repo_root_match=False relies only on the argv pattern."""
    argv = (
        "/Users/greggkeithley/projects/mtg-bot/.venv-py313/bin/python "
        "scripts/run_self_play_shard.py --mode=pull"
    )
    assert _argv_matches(
        argv, "", repo_root=None, require_repo_root_match=False,
    )


def test_argv_matches_repo_required_but_missing_returns_false() -> None:
    """Caller asked for repo-root match but didn't supply one
    (None or empty) — fail-safe to False rather than match
    everything."""
    argv = "python scripts/run_self_play_shard.py --mode=pull"
    assert not _argv_matches(
        argv, "", repo_root=None, require_repo_root_match=True,
    )
    assert not _argv_matches(
        argv, "", repo_root="", require_repo_root_match=True,
    )


# ---- _build_descendant_set ----------------------------------------

def test_build_descendant_set_walks_tree() -> None:
    rows = [
        ProcessInfo(pid=100, ppid=1, exe="", args=""),
        ProcessInfo(pid=101, ppid=100, exe="", args=""),
        ProcessInfo(pid=102, ppid=100, exe="", args=""),
        ProcessInfo(pid=103, ppid=101, exe="", args=""),
        ProcessInfo(pid=200, ppid=1, exe="", args=""),
    ]
    descendants = _build_descendant_set(rows, root_pid=100)
    assert descendants == {101, 102, 103}


def test_build_descendant_set_no_children() -> None:
    rows = [
        ProcessInfo(pid=100, ppid=1, exe="", args=""),
    ]
    assert _build_descendant_set(rows, root_pid=100) == set()


def test_build_descendant_set_root_not_in_listing() -> None:
    rows = [
        ProcessInfo(pid=200, ppid=1, exe="", args=""),
    ]
    # Master PID isn't in the listing (race: process disappeared
    # between the snapshot and the call). Returns empty set, doesn't
    # blow up.
    assert _build_descendant_set(rows, root_pid=99999) == set()


def test_build_descendant_set_handles_cycles() -> None:
    """Defensive: a malformed listing claiming pid X has ppid Y AND
    pid Y has ppid X must terminate, not loop forever. Real OS
    listings won't produce cycles but the BFS shouldn't trust that.
    """
    rows = [
        ProcessInfo(pid=100, ppid=200, exe="", args=""),
        ProcessInfo(pid=200, ppid=100, exe="", args=""),
    ]
    # Walk from 100: it should visit 200 and stop. (200's only child
    # is 100, which equals root_pid, so we skip it.)
    descendants = _build_descendant_set(rows, root_pid=100)
    assert descendants == {200}


# ---- find_orphan_pids_local ---------------------------------------

def _make_listing(repo: str = REPO) -> list[ProcessInfo]:
    """A representative process listing the master might see at
    startup: itself, its mp children, prior-run orphans, and
    unrelated Python.
    """
    return [
        # The master itself.
        ProcessInfo(
            pid=1000,
            ppid=500,  # cmd.exe / shell
            exe=f"{repo}/.venv/Scripts/python.exe",
            args=(
                f"{repo}/.venv/Scripts/python.exe "
                f"scripts/iterate_alphazero.py --bootstrap v4"
            ),
        ),
        # The master's own children — must be excluded.
        ProcessInfo(
            pid=1001, ppid=1000,
            exe=f"{repo}/.venv/Scripts/python.exe",
            args=(
                f'"{repo}/.venv/Scripts/python.exe" -c '
                f'"from multiprocessing.spawn import spawn_main; ..."'
            ),
        ),
        ProcessInfo(
            pid=1002, ppid=1000,
            exe=f"{repo}/.venv/Scripts/python.exe",
            args=(
                f'"{repo}/.venv/Scripts/python.exe" -c '
                f'"from multiprocessing.spawn import spawn_main; ..."'
            ),
        ),
        # Orphan with PPID=1 from a prior run.
        ProcessInfo(
            pid=2000, ppid=1,
            exe=f"{repo}/.venv/bin/python",
            args=(
                f"{repo}/.venv/bin/python "
                f"scripts/run_self_play_shard.py --mode=pull"
            ),
        ),
        # mp child whose parent is dead — ppid points to a PID that
        # no longer exists in the listing. Re-parented to System (4)
        # / init (1) on the real OS; our matcher keys on argv, not
        # on a "ppid is dead" probe.
        ProcessInfo(
            pid=2001, ppid=4,
            exe=f"{repo}/.venv/Scripts/python.exe",
            args=(
                f'"{repo}/.venv/Scripts/python.exe" -c '
                f'"from multiprocessing.spawn import spawn_main; ..."'
            ),
        ),
        # Unrelated Python — Jupyter kernel. MUST NOT be killed.
        ProcessInfo(
            pid=3000, ppid=1,
            exe="C:/Users/Gregg/jupyter-venv/Scripts/python.exe",
            args=(
                'C:/Users/Gregg/jupyter-venv/Scripts/python.exe -m '
                'ipykernel_launcher -f /tmp/kernel.json'
            ),
        ),
        # Unrelated Python — VSCode pylance / language server. MUST
        # NOT be killed.
        ProcessInfo(
            pid=3001, ppid=1,
            exe="C:/Users/Gregg/AppData/.../ms-python/pylance.exe",
            args=(
                "C:/Users/Gregg/AppData/.../ms-python/pylance.exe "
                "--node-ipc"
            ),
        ),
        # Non-Python entirely.
        ProcessInfo(
            pid=4000, ppid=1, exe="/usr/bin/bash",
            args="/usr/bin/bash -c 'something'",
        ),
    ]


def test_find_orphan_pids_excludes_self_and_descendants() -> None:
    rows = _make_listing()
    descendants = _build_descendant_set(rows, root_pid=1000)
    assert descendants == {1001, 1002}
    pids = find_orphan_pids_local(
        repo_root=REPO,
        self_pid=1000,
        descendant_pids=descendants,
        process_listing=rows,
    )
    # 2000 (orphan shard) and 2001 (orphan mp.spawn under repo)
    # should match. 1001/1002 are descendants. 3000/3001/4000
    # don't reference the repo or pattern.
    assert pids == [2000, 2001]


def test_find_orphan_pids_unrelated_python_not_killed() -> None:
    """Jupyter + pylance MUST NOT match even when the matcher runs
    without the master being in the listing."""
    rows = [r for r in _make_listing() if r.pid not in (1000, 1001, 1002)]
    pids = find_orphan_pids_local(
        repo_root=REPO,
        self_pid=99999,  # master not present
        descendant_pids=set(),
        process_listing=rows,
    )
    assert 3000 not in pids
    assert 3001 not in pids
    assert 4000 not in pids
    # Orphans still found.
    assert 2000 in pids
    assert 2001 in pids


def test_find_orphan_pids_dead_parent_chain() -> None:
    """Reproduces today's Mac case: mp children whose parent ppid
    points at PID 1 (init reparenting). All should be killed.
    """
    rows = []
    for pid in range(5000, 5008):
        rows.append(ProcessInfo(
            pid=pid, ppid=1,
            exe=f"{REPO}/.venv-py313/bin/python",
            args=(
                f"{REPO}/.venv-py313/bin/python -c "
                f"\"from multiprocessing.spawn import spawn_main\""
            ),
        ))
    pids = find_orphan_pids_local(
        repo_root=REPO,
        self_pid=99,
        descendant_pids=set(),
        process_listing=rows,
    )
    assert pids == sorted(range(5000, 5008))


def test_find_orphan_pids_remote_mode_relaxed_repo_match() -> None:
    """remote=False/repo-root=None: matcher uses pattern only. Jupyter
    must still be safe (no run_self_play_shard.py or iterate
    pattern), but mp.spawn workers from a different absolute path
    still match — the remote may have a different layout.
    """
    rows = [
        ProcessInfo(
            pid=2000, ppid=1,
            exe="/Users/g/mtg-bot/.venv-py313/bin/python",
            args=(
                "/Users/g/mtg-bot/.venv-py313/bin/python "
                "scripts/run_self_play_shard.py --mode=pull"
            ),
        ),
        ProcessInfo(
            pid=3000, ppid=1,
            exe="/Users/g/jupyter/.venv/bin/python",
            args="/Users/g/jupyter/.venv/bin/python -m ipykernel",
        ),
    ]
    pids = find_orphan_pids_local(
        repo_root=None,
        self_pid=99,
        descendant_pids=set(),
        process_listing=rows,
        require_repo_root_match=False,
    )
    assert pids == [2000]


def test_find_orphan_pids_skips_non_python_rows() -> None:
    """A row whose exe AND first argv token aren't Python must be
    skipped, even if argv accidentally contains the pattern (e.g. a
    log-tailing bash that's grepping for ``run_self_play_shard.py``).
    """
    rows = [
        ProcessInfo(
            pid=6000, ppid=1, exe="/bin/bash",
            args="bash -c 'tail -f run_self_play_shard.py.log'",
        ),
    ]
    pids = find_orphan_pids_local(
        repo_root=REPO,
        self_pid=99,
        descendant_pids=set(),
        process_listing=rows,
        require_repo_root_match=False,
    )
    assert pids == []


# ---- kill_pids ----------------------------------------------------

def test_kill_pids_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_terminate(pid: int, *, kill_grace_s: float = 2.0) -> bool:
        calls.append(pid)
        return True

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._terminate_pid", fake_terminate,
    )
    out = kill_pids([100, 200, 300])
    assert out == [100, 200, 300]
    assert calls == [100, 200, 300]


def test_kill_pids_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """kill_pids should dedupe — duplicate PIDs in the input mean
    the listing produced something weird, terminate-once is correct."""
    calls: list[int] = []

    def fake_terminate(pid: int, *, kill_grace_s: float = 2.0) -> bool:
        calls.append(pid)
        return True

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._terminate_pid", fake_terminate,
    )
    out = kill_pids([100, 100, 200, 100])
    assert out == [100, 200]
    assert calls == [100, 200]


def test_kill_pids_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some PIDs fail to terminate; survivors are reported but
    successful kills still come back."""
    def fake_terminate(pid: int, *, kill_grace_s: float = 2.0) -> bool:
        return pid != 200  # pretend 200 is unkillable

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._terminate_pid", fake_terminate,
    )
    out = kill_pids([100, 200, 300])
    assert out == [100, 300]


# ---- _parse_killed_pids -------------------------------------------

def test_parse_killed_pids_present() -> None:
    line = "[sweep] killed 3 orphan processes on local: [100, 200, 300]"
    assert _parse_killed_pids(line) == [100, 200, 300]


def test_parse_killed_pids_zero() -> None:
    line = "[sweep] killed 0 orphan processes on local: []"
    assert _parse_killed_pids(line) == []


def test_parse_killed_pids_with_other_lines() -> None:
    text = (
        "[sweep] local: found 2 orphan candidate PIDs: [100, 200]\n"
        "[sweep] killed 2 orphan processes on local: [100, 200]\n"
    )
    assert _parse_killed_pids(text) == [100, 200]


def test_parse_killed_pids_absent() -> None:
    """When sweep_local_orphans found nothing, no killed line is
    emitted; parser returns []."""
    text = "[sweep] local: no orphan processes found\n"
    assert _parse_killed_pids(text) == []


def test_parse_killed_pids_malformed() -> None:
    """Best-effort: missing brackets / non-digit content shouldn't
    crash."""
    assert _parse_killed_pids("") == []
    assert _parse_killed_pids("[sweep] killed 1 orphan processes on x") == []


def test_parse_killed_pids_no_open_bracket() -> None:
    """A line with ``killed`` but no ``[`` at all (unlikely but
    defensive): match-line predicates pass, then bracket lookup
    returns no result. Skip it.
    """
    # A line containing ']' but no '['.
    line = "killed 1 orphan processes on x: data]"
    assert _parse_killed_pids(line) == []


def test_parse_killed_pids_no_close_bracket() -> None:
    """``killed`` line with no ``]`` at all — rb is -1, skip."""
    line = "killed 1 orphan processes on x: data"
    assert _parse_killed_pids(line) == []


def test_parse_killed_pids_skips_non_digit_tokens() -> None:
    """A token that isn't all-digit (e.g. an empty entry from a
    trailing comma) gets skipped, the rest still parse."""
    line = "[sweep] killed 2 orphan processes on x: [100, , 200, abc]"
    assert _parse_killed_pids(line) == [100, 200]


# ---- sweep_local_orphans ------------------------------------------

def test_sweep_local_orphans_emits_killed_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract the user reads (and that the remote-sweep parser
    relies on): the killed line includes the kill count and the
    PID list. Loud is good — silent is bad.
    """
    rows = _make_listing()
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._list_processes_local",
        lambda: rows,
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.os.getpid", lambda: 1000,
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._terminate_pid",
        lambda pid, kill_grace_s=2.0: True,
    )
    captured: list[str] = []

    killed = sweep_local_orphans(
        repo_root=REPO, label="local",
        print_fn=captured.append,
    )
    assert sorted(killed) == [2000, 2001]
    found_killed = any(
        "killed 2 orphan processes on local: [2000, 2001]" in s
        for s in captured
    )
    assert found_killed, captured


def test_sweep_local_orphans_no_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No orphans → loud "no orphan processes found" line. Don't
    silently return."""
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._list_processes_local",
        lambda: [
            ProcessInfo(
                pid=1000, ppid=1, exe="/usr/bin/python",
                args="python -m pytest tests/",
            ),
        ],
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.os.getpid", lambda: 99,
    )
    captured: list[str] = []
    killed = sweep_local_orphans(
        repo_root=REPO, label="local",
        print_fn=captured.append,
    )
    assert killed == []
    assert any("no orphan processes found" in s for s in captured)


def test_sweep_local_orphans_listing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-listing failure (subprocess crashed, JSON parse
    failed) is non-fatal — return [] and log."""
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._list_processes_local",
        lambda: [],
    )
    captured: list[str] = []
    killed = sweep_local_orphans(
        repo_root=REPO, label="local",
        print_fn=captured.append,
    )
    assert killed == []
    assert any("process listing failed" in s for s in captured)


def test_sweep_local_orphans_survivor_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a candidate PID can't be killed, log it loudly so the
    user can intervene manually."""
    rows = _make_listing()
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._list_processes_local",
        lambda: rows,
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.os.getpid", lambda: 1000,
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep._terminate_pid",
        # Pretend 2001 is unkillable.
        lambda pid, kill_grace_s=2.0: pid != 2001,
    )
    captured: list[str] = []
    killed = sweep_local_orphans(
        repo_root=REPO, label="local",
        print_fn=captured.append,
    )
    assert killed == [2000]
    assert any(
        "WARNING" in s and "2001" in s for s in captured
    ), captured


# ---- sweep_remote_orphans -----------------------------------------

class _FakeShard:
    def __init__(
        self,
        host: str = "fake-remote",
        repo_path: str = "/home/g/mtg-bot",
        venv_python: str = ".venv-py313/bin/python",
        label: str = "",
    ) -> None:
        self.host = host
        self.repo_path = repo_path
        self.venv_python = venv_python
        self.label = label
        self.user = "g"
        self.ssh_port = 22

    @property
    def display_label(self) -> str:
        return self.label or self.host


def test_sweep_remote_orphans_happy_path() -> None:
    shard = _FakeShard()

    def fake_ssh_cmd(s, cmd):
        # Cmd should embed sweep_local_orphans and the repo path.
        assert "sweep_local_orphans" in cmd
        assert s.repo_path in cmd
        return ["ssh", "g@fake-remote", cmd]

    fake_stdout = (
        "[sweep] remote: found 3 orphan candidate PIDs: [9001, 9002, 9003]\n"
        "[sweep] killed 3 orphan processes on remote: [9001, 9002, 9003]\n"
    )

    def fake_run(cmd, *, timeout=None):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=fake_stdout, stderr="",
        )

    captured: list[str] = []
    pids = sweep_remote_orphans(
        shard,
        ssh_cmd_factory=fake_ssh_cmd,
        run_subprocess=fake_run,
        print_fn=captured.append,
    )
    assert pids == [9001, 9002, 9003]
    # Each remote stdout line is mirrored locally.
    assert any("9001" in s for s in captured)


def test_sweep_remote_orphans_called_process_error() -> None:
    shard = _FakeShard()

    def fake_run(cmd, *, timeout=None):
        raise subprocess.CalledProcessError(
            returncode=255, cmd=cmd, stderr="ssh: connect failed",
        )

    captured: list[str] = []
    pids = sweep_remote_orphans(
        shard,
        ssh_cmd_factory=lambda s, c: ["ssh", c],
        run_subprocess=fake_run,
        print_fn=captured.append,
    )
    assert pids == []
    assert any(
        "remote sweep failed" in s and "fake-remote" in s
        for s in captured
    ), captured


def test_sweep_remote_orphans_timeout() -> None:
    shard = _FakeShard()

    def fake_run(cmd, *, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)

    captured: list[str] = []
    pids = sweep_remote_orphans(
        shard,
        ssh_cmd_factory=lambda s, c: ["ssh", c],
        run_subprocess=fake_run,
        print_fn=captured.append,
    )
    assert pids == []
    assert any("remote sweep error" in s for s in captured)


def test_sweep_remote_orphans_oserror() -> None:
    shard = _FakeShard()

    def fake_run(cmd, *, timeout=None):
        raise OSError("ENOENT: ssh not found")

    captured: list[str] = []
    pids = sweep_remote_orphans(
        shard,
        ssh_cmd_factory=lambda s, c: ["ssh", c],
        run_subprocess=fake_run,
        print_fn=captured.append,
    )
    assert pids == []
    assert any("OSError" in s for s in captured)


def test_sweep_remote_orphans_windows_command_format() -> None:
    """Windows remote: cmd must use ``cd /d`` and quoted
    ``"<venv>" -c "..."`` so the remote cmd.exe parses it."""
    shard = _FakeShard(
        host="windesk", repo_path="C:\\Users\\g\\mtg-bot",
        venv_python=".venv-py313\\Scripts\\python.exe",
    )

    captured_cmd: list[str] = []

    def fake_ssh_cmd(s, cmd):
        captured_cmd.append(cmd)
        return ["ssh", cmd]

    def fake_run(cmd, *, timeout=None):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="[sweep] killed 0 orphan processes on remote: []\n",
            stderr="",
        )

    sweep_remote_orphans(
        shard,
        ssh_cmd_factory=fake_ssh_cmd,
        run_subprocess=fake_run,
        print_fn=lambda s: None,
    )
    assert captured_cmd
    assert "cd /d" in captured_cmd[0]
    assert ".exe" in captured_cmd[0]


def test_sweep_remote_orphans_posix_command_format() -> None:
    """POSIX remote: cmd uses ``cd <quoted>`` and ``-c <quoted>``."""
    shard = _FakeShard(
        host="mac", repo_path="/Users/g/mtg-bot",
        venv_python=".venv-py313/bin/python",
    )
    captured_cmd: list[str] = []

    def fake_ssh_cmd(s, cmd):
        captured_cmd.append(cmd)
        return ["ssh", cmd]

    def fake_run(cmd, *, timeout=None):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="[sweep] killed 0 orphan processes on remote: []\n",
            stderr="",
        )

    sweep_remote_orphans(
        shard,
        ssh_cmd_factory=fake_ssh_cmd,
        run_subprocess=fake_run,
        print_fn=lambda s: None,
    )
    assert captured_cmd
    cmd = captured_cmd[0]
    assert "cd /d" not in cmd
    assert "PYTHONPATH=src" in cmd


# ---- ProcessInfo round-trip ---------------------------------------

def test_process_info_round_trip() -> None:
    pi = ProcessInfo(pid=100, ppid=1, exe="/usr/bin/python", args="x y z")
    pi2 = ProcessInfo.from_dict(pi.to_dict())
    assert pi2 == pi


def test_process_info_from_dict_handles_missing_fields() -> None:
    pi = ProcessInfo.from_dict({"pid": 100})
    assert pi.pid == 100
    assert pi.ppid == 0
    assert pi.exe == ""
    assert pi.args == ""


# ---- _list_processes_local smoke test -----------------------------

def test_list_processes_local_returns_rows() -> None:
    """Real OS process listing — at least the test process itself
    should show up. Regression guard: PowerShell / ps invocation
    failure would silently return [].
    """
    from mtg_bot.training.orphan_sweep import _list_processes_local
    rows = _list_processes_local()
    # Don't assert exact match — process counts are noisy on shared
    # CI. Just confirm the listing is non-empty and well-formed.
    assert len(rows) > 0
    own_pid = subprocess.run(
        # cheap pid sanity probe
        ["echo", "ok"], capture_output=True, text=True, shell=False,
    )
    del own_pid  # unused; just confirms subprocess works
    pids = {r.pid for r in rows}
    import os as _os
    assert _os.getpid() in pids


# ---- _list_processes_windows mocked --------------------------------

def _windows_proc_json(rows: list[dict]) -> str:
    import json as _json
    return _json.dumps(rows if len(rows) > 1 else (rows[0] if rows else {}))


def test_list_processes_windows_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PowerShell returned a list of process records → we get one
    ProcessInfo per dict. Single-record output (PowerShell's
    "auto-unwrap to dict") also works."""
    from mtg_bot.training import orphan_sweep as _os_mod

    rows_in = [
        {
            "ProcessId": 1234, "ParentProcessId": 5678,
            "Name": "python.exe",
            "ExecutablePath": "C:\\venv\\Scripts\\python.exe",
            "CommandLine": "python.exe scripts/run_self_play_shard.py",
        },
        {
            "ProcessId": 999, "ParentProcessId": 1,
            "Name": "python.exe",
            "ExecutablePath": "",
            "CommandLine": "",
        },
    ]
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=0,
        stdout=_windows_proc_json(rows_in), stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_windows()
    pids = {r.pid for r in out}
    assert 1234 in pids
    assert 999 in pids


def test_list_processes_windows_single_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One process = PowerShell unwraps to a single dict, not a list.
    The parser must promote it back to a list."""
    from mtg_bot.training import orphan_sweep as _os_mod
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=0,
        stdout='{"ProcessId": 7,"ParentProcessId": 1,"Name": "python"}',
        stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_windows()
    assert len(out) == 1
    assert out[0].pid == 7


def test_list_processes_windows_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod

    def boom(*a, **k):
        raise OSError("synthetic")

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run", boom,
    )
    assert _os_mod._list_processes_windows() == []


def test_list_processes_windows_returns_empty_on_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=1, stdout="", stderr="boom",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    assert _os_mod._list_processes_windows() == []


def test_list_processes_windows_returns_empty_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=0,
        stdout="not actually json", stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    assert _os_mod._list_processes_windows() == []


def test_list_processes_windows_tolerates_control_chars_in_commandline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: real-world ``Get-CimInstance Win32_Process`` output on
    Win11 has been observed with a literal control char (e.g. ``\\x01``)
    embedded in some process's ``CommandLine`` field. Strict-mode
    ``json.loads`` rejects the whole ~160 KB document at column ~135728
    with ``Invalid control character at: line 1 column N``, the previous
    behavior silently swallowed the JSONDecodeError and returned [],
    masking a working-but-broken process listing on this machine.
    ``strict=False`` (RFC 8259 leniency) accepts control chars inside
    strings — listing succeeds, sweep proceeds.
    """
    from mtg_bot.training import orphan_sweep as _os_mod
    # Synthetic JSON: build the array by hand so we can splice a
    # control char straight into a CommandLine string. ``json.dumps``
    # would escape it (), defeating the test.
    raw_json = (
        '['
        '{"ProcessId":1234,"ParentProcessId":1,"Name":"python.exe",'
        '"ExecutablePath":"C:\\\\venv\\\\python.exe",'
        '"CommandLine":"python.exe --evil-arg=\x01-with-ctrlchar"},'
        '{"ProcessId":42,"ParentProcessId":1,"Name":"python.exe",'
        '"ExecutablePath":"","CommandLine":""}'
        ']'
    )
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=0,
        stdout=raw_json, stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_windows()
    pids = {r.pid for r in out}
    # Both rows must be parsed — a strict-mode parser would reject the
    # entire document on the first ``\x01`` and return [].
    assert pids == {1234, 42}
    # The control char survives into ``args`` (it's in the source data;
    # we just don't want it to break parsing).
    by_pid = {r.pid: r for r in out}
    assert "\x01" in by_pid[1234].args


def test_list_processes_windows_skips_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows missing ProcessId, with non-int values, or that are
    strings instead of dicts must be skipped silently rather than
    crash the whole listing."""
    from mtg_bot.training import orphan_sweep as _os_mod
    rows_in = [
        # Missing ProcessId entirely.
        {"ParentProcessId": 1, "Name": "x"},
        # PID is a string that can't parse.
        {"ProcessId": "notanint", "ParentProcessId": 1},
        # Bare string instead of dict.
        "garbage row",
        # Valid one.
        {"ProcessId": 42, "ParentProcessId": 1,
         "ExecutablePath": "p", "CommandLine": "p"},
    ]
    fake_cp = subprocess.CompletedProcess(
        args=["powershell"], returncode=0,
        stdout=_windows_proc_json(rows_in), stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_windows()
    assert len(out) == 1
    assert out[0].pid == 42


# ---- _list_processes_posix mocked ----------------------------------

def test_list_processes_posix_parses_ps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    ps_out = (
        "  100     1 python  python scripts/run_self_play_shard.py\n"
        "  101   100 python  python -c \"from multiprocessing.spawn import \"\n"
        "  bogus_line\n"  # malformed: no ints
        "  200     1 bash    bash -c 'sleep 1'\n"
    )
    fake_cp = subprocess.CompletedProcess(
        args=["ps"], returncode=0, stdout=ps_out, stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_posix()
    pids = {r.pid for r in out}
    assert 100 in pids
    assert 101 in pids
    assert 200 in pids


def test_list_processes_posix_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod

    def boom(*a, **k):
        raise OSError("synthetic")

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run", boom,
    )
    assert _os_mod._list_processes_posix() == []


def test_list_processes_posix_nonzero_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    fake_cp = subprocess.CompletedProcess(
        args=["ps"], returncode=1, stdout="", stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    assert _os_mod._list_processes_posix() == []


def test_list_processes_posix_skips_blank_and_garbage_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``ps`` parser must skip blank lines (line 298) and lines
    where pid/ppid aren't ints (lines 307-308) silently."""
    from mtg_bot.training import orphan_sweep as _os_mod
    ps_out = (
        "\n"
        "   \n"
        "  notapid notappid python python args\n"  # ValueError
        "  100     1 python  python -c \"work\"\n"
    )
    fake_cp = subprocess.CompletedProcess(
        args=["ps"], returncode=0, stdout=ps_out, stderr="",
    )
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: fake_cp,
    )
    out = _os_mod._list_processes_posix()
    assert {r.pid for r in out} == {100}


# ---- _python_basename_in_args edge cases ---------------------------

def test_python_basename_in_args_empty() -> None:
    from mtg_bot.training.orphan_sweep import _python_basename_in_args
    assert _python_basename_in_args("") is False


def test_python_basename_in_args_python_first_token() -> None:
    from mtg_bot.training.orphan_sweep import _python_basename_in_args
    assert _python_basename_in_args("python -c 'x'") is True
    assert _python_basename_in_args("/usr/bin/python3.13 script.py") is True


def test_python_basename_in_args_non_python() -> None:
    from mtg_bot.training.orphan_sweep import _python_basename_in_args
    assert _python_basename_in_args("bash -c 'x'") is False


# ---- _list_processes_local platform dispatch -----------------------

def test_list_processes_local_dispatches_by_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-cover the dispatch in ``_list_processes_local`` so both
    branches show up in coverage."""
    from mtg_bot.training import orphan_sweep as _os_mod

    sentinel = [ProcessInfo(pid=1, ppid=1, exe="x", args="x")]
    monkeypatch.setattr(
        _os_mod, "_list_processes_windows", lambda: sentinel,
    )
    monkeypatch.setattr(
        _os_mod, "_list_processes_posix", lambda: sentinel,
    )
    # Force win32.
    monkeypatch.setattr(_os_mod.sys, "platform", "win32")
    assert _os_mod._list_processes_local() is sentinel
    monkeypatch.setattr(_os_mod.sys, "platform", "darwin")
    assert _os_mod._list_processes_local() is sentinel


# ---- _terminate_pid mocked -----------------------------------------

def test_terminate_pid_windows_taskkill_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a[0], returncode=0, stdout="", stderr="",
        ),
    )
    assert _os_mod._terminate_pid(1234) is True


def test_terminate_pid_windows_taskkill_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """taskkill /F returns 128 for "process not found" — counts as
    success (the orphan is gone, which is what we wanted)."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a[0], returncode=128, stdout="", stderr="not found",
        ),
    )
    assert _os_mod._terminate_pid(9999) is True


def test_terminate_pid_windows_taskkill_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returncode 1 (access denied) is a real failure; we report
    it back so the caller can warn the user."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a[0], returncode=1, stdout="", stderr="denied",
        ),
    )
    assert _os_mod._terminate_pid(1234) is False


def test_terminate_pid_windows_taskkill_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "win32")

    def boom(*a, **k):
        raise OSError("ENOENT")

    monkeypatch.setattr(
        "mtg_bot.training.orphan_sweep.subprocess.run", boom,
    )
    assert _os_mod._terminate_pid(1234) is False


def test_terminate_pid_posix_sigterm_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX path: SIGTERM, probe with kill(0). Probe raises
    ProcessLookupError → success."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        sent.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError("gone")
        # SIGTERM accepted — process will "die" before next probe.

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    assert _os_mod._terminate_pid(1234, kill_grace_s=0.01) is True
    # First call was SIGTERM(15).
    assert sent[0] == (1234, 15)


def test_terminate_pid_posix_already_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process disappeared before we got to it; SIGTERM raises
    ProcessLookupError. Counts as success."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    assert _os_mod._terminate_pid(1234) is True


def test_terminate_pid_posix_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM denied (EPERM) → return False; we don't have
    permission, surface to caller."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        raise PermissionError("EPERM")

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    assert _os_mod._terminate_pid(1234) is False


def test_terminate_pid_posix_oserror_on_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic OSError on SIGTERM is a hard failure."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        raise OSError("nope")

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    assert _os_mod._terminate_pid(1234) is False


def test_terminate_pid_posix_escalates_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process doesn't die after SIGTERM grace → SIGKILL fires.
    SIGKILL succeeds → final probe says gone → True.
    """
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")
    log: list[tuple[int, int]] = []
    state = {"alive": True}

    def fake_kill(pid, sig):
        log.append((pid, sig))
        if sig == 0:  # probe
            if state["alive"]:
                return None  # still here
            raise ProcessLookupError("gone")
        if sig == 9:
            state["alive"] = False
            return None
        # sig 15 (SIGTERM) accepted but doesn't kill it
        return None

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    # Tiny grace → loop exits quickly to escalation.
    assert _os_mod._terminate_pid(
        1234, kill_grace_s=0.001,
    ) is True
    # Should have included a SIGKILL.
    sigs = [s for _, s in log]
    assert 9 in sigs


def test_terminate_pid_posix_sigkill_lookup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILL itself races: process disappears between escalation
    and the kill call. ProcessLookupError → True (it's gone)."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        if sig == 15:
            return None  # SIGTERM accepted but ignored
        if sig == 0:
            return None  # still alive after grace
        if sig == 9:
            raise ProcessLookupError("just died")
        return None

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    assert _os_mod._terminate_pid(1234, kill_grace_s=0.001) is True


def test_terminate_pid_posix_sigkill_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILL raises OSError (very rare — protected process). Returns
    False so caller can log the survivor."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        if sig == 15:
            return None
        if sig == 0:
            return None
        if sig == 9:
            raise OSError("EPERM")
        return None

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    assert _os_mod._terminate_pid(1234, kill_grace_s=0.001) is False


def test_terminate_pid_posix_oserror_on_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe raises OSError that isn't ProcessLookupError (e.g.
    EPERM mid-loop). Returns False so caller can log it."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    state = {"first": True}

    def fake_kill(pid, sig):
        if sig == 15:
            return None
        if sig == 0:
            raise OSError("EPERM")
        return None

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    assert _os_mod._terminate_pid(1234, kill_grace_s=0.01) is False
    del state


def test_terminate_pid_posix_final_probe_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM ignored, kill_grace_s elapses, SIGKILL called, but
    final probe says still alive — return False (the unkillable case
    we surface to the user via the warning line)."""
    from mtg_bot.training import orphan_sweep as _os_mod
    monkeypatch.setattr(_os_mod.sys, "platform", "linux")

    def fake_kill(pid, sig):
        # Always returns None — process refuses to die.
        return None

    monkeypatch.setattr(_os_mod.os, "kill", fake_kill)
    monkeypatch.setattr(_os_mod.time, "sleep", lambda _s: None)
    assert _os_mod._terminate_pid(1234, kill_grace_s=0.001) is False


# ---- sweep_remote_orphans default factories -----------------------

def test_sweep_remote_orphans_imports_default_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ssh_cmd_factory / run_subprocess are None, the function
    must lazily pull in distributed.py's defaults. Coverage gate for
    the import branch."""
    from mtg_bot.training import orphan_sweep as _os_mod

    captured: list[list[str]] = []

    def fake_run(cmd, *, timeout=None):
        captured.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="[sweep] killed 0 orphan processes on remote: []\n",
            stderr="",
        )

    # Patch _run_subprocess at import path; ssh_cmd default still used.
    monkeypatch.setattr(
        "mtg_bot.training.distributed._run_subprocess",
        fake_run,
    )
    shard = _FakeShard(
        host="some-host", repo_path="/h/mtg-bot",
        venv_python=".venv/bin/python",
    )
    pids = _os_mod.sweep_remote_orphans(shard, print_fn=lambda s: None)
    assert pids == []
    # Confirm _run_subprocess actually got called via lazy import.
    assert captured
