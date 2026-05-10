"""Tests for ``mtg_bot.cli.shard`` (the runner formerly known as
``scripts/run_self_play_shard.py``; the script is now a thin shim
that just calls into this module).

The deeper integration (run a fake shard end-to-end via
``run_shard_pull``) is covered by ``tests/training/test_distributed.py``.
This module exists to pin the runner's ``os._exit`` finalizer
behavior — the fix for the post-self-play orchestrator hang where
remote shard Python processes don't exit even after all data is
flushed and progress JSONs are written. See
``docs/SEARCH_LIMITATIONS.md`` for full rationale.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """Fresh-import ``mtg_bot.cli.shard`` — the live module the shim
    routes into. Force-reload so each test gets independent module
    state (matters when monkeypatching ``_run``, ``os``, etc.)."""
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "mtg_bot.cli.shard" in sys.modules:
        del sys.modules["mtg_bot.cli.shard"]
    return importlib.import_module("mtg_bot.cli.shard")


def test_main_calls_os_exit_zero_on_help() -> None:
    """``--help`` makes argparse raise ``SystemExit(0)``. The
    finalizer must catch that and exit via ``os._exit(0)``, NOT
    propagate the SystemExit (which would let normal interpreter
    shutdown run, defeating the whole point)."""
    module = _load_module()
    with mock.patch.object(module.os, "_exit") as mock_exit:
        # argparse prints --help and SystemExit(0)s. Our wrapper
        # catches that and routes through os._exit.
        module.main(["--help"])
    mock_exit.assert_called_once_with(0)


def test_main_calls_os_exit_nonzero_on_missing_required_args() -> None:
    """argparse raises ``SystemExit(2)`` when required args are
    missing. The finalizer preserves that exit code."""
    module = _load_module()
    with mock.patch.object(module.os, "_exit") as mock_exit:
        # No --spec-b64 / --shard-b64 / --queue-url / etc. argparse
        # will SystemExit(2). Suppress the noisy stderr it prints.
        module.main([])
    mock_exit.assert_called_once()
    (code,), _ = mock_exit.call_args
    # argparse uses 2 for "argument error" — preserve it.
    assert code == 2


def test_main_calls_os_exit_one_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any non-SystemExit exception inside ``_run`` must be
    swallowed by the finalizer and routed through ``os._exit(1)``,
    NOT propagate. If it propagated, the at-exit thread-join hang
    we're trying to dodge would re-appear, defeating the point."""
    module = _load_module()

    def _boom(_argv: object) -> None:
        raise RuntimeError("synthetic failure inside _run")

    monkeypatch.setattr(module, "_run", _boom)
    with mock.patch.object(module.os, "_exit") as mock_exit:
        module.main(["--ignored-because-_run-is-stubbed"])
    mock_exit.assert_called_once_with(1)


def test_main_calls_os_exit_zero_on_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_run`` returns normally (the happy path — shard
    finished, npzs flushed, progress JSON written), the finalizer
    exits 0."""
    module = _load_module()

    def _ok(_argv: object) -> None:
        return None

    monkeypatch.setattr(module, "_run", _ok)
    with mock.patch.object(module.os, "_exit") as mock_exit:
        module.main([])
    mock_exit.assert_called_once_with(0)


def test_main_flushes_stdio_before_os_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finalizer flushes stdout + stderr before ``os._exit``
    so the parent SSH process sees the final ``{"npz_paths": [...]}``
    line (and any buffered traceback on the error path) before
    the kernel tears the pipe down."""
    module = _load_module()

    flushed: list[str] = []

    class _StreamProbe:
        def __init__(self, name: str) -> None:
            self._name = name

        def flush(self) -> None:
            flushed.append(self._name)

        # Forward write so argparse / traceback don't blow up if
        # they touch the stream while it's swapped in. We don't
        # care what they print — only that flush() is called.
        def write(self, _s: str) -> int:
            return len(_s)

    monkeypatch.setattr(module, "_run", lambda _argv: None)
    monkeypatch.setattr(sys, "stdout", _StreamProbe("stdout"))
    monkeypatch.setattr(sys, "stderr", _StreamProbe("stderr"))

    with mock.patch.object(module.os, "_exit") as mock_exit:
        module.main([])

    assert "stdout" in flushed
    assert "stderr" in flushed
    mock_exit.assert_called_once_with(0)


def test_main_tolerates_flush_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stdio flush failures (closed pipe etc.) must not prevent
    ``os._exit`` from running. The whole point of this fix is that
    ``os._exit`` ALWAYS fires from the finally block."""
    module = _load_module()

    class _BrokenStream:
        def flush(self) -> None:
            raise BrokenPipeError("synthetic")

        def write(self, _s: str) -> int:
            return 0

    monkeypatch.setattr(module, "_run", lambda _argv: None)
    monkeypatch.setattr(sys, "stdout", _BrokenStream())
    monkeypatch.setattr(sys, "stderr", _BrokenStream())

    with mock.patch.object(module.os, "_exit") as mock_exit:
        module.main([])

    mock_exit.assert_called_once_with(0)


def test_run_shard_pull_does_not_call_os_exit() -> None:
    """``os._exit`` belongs in the SCRIPT wrapper, NOT in the
    library function. ``run_shard_pull`` is also called from
    in-process unit tests (``tests/training/test_distributed.py``);
    if it called ``os._exit`` it would terminate the test runner.

    This is a static-source check rather than a runtime one — we
    can't realistically call ``run_shard_pull`` end-to-end from a
    unit test cheaply, and a runtime check would be unable to
    distinguish "didn't reach os._exit on this code path" from
    "doesn't have one anywhere".
    """
    distributed_path = (
        _REPO_ROOT / "src" / "mtg_bot" / "training" / "distributed.py"
    )
    src = distributed_path.read_text(encoding="utf-8")
    # If this assertion ever fires, someone added os._exit to the
    # library — push it to the script wrapper instead.
    assert "os._exit" not in src, (
        "os._exit should live only in mtg_bot.cli.shard's main(), "
        "not in the library function. Library functions are called "
        "from in-process unit tests where os._exit would kill the "
        "test runner."
    )


def test_install_kill_handlers_registers_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_install_kill_handlers`` must register SIGTERM (always) and
    SIGHUP/SIGINT (when available). Defends against the regression
    where a future refactor accidentally drops one — the watchdog
    SSH yank only fires SIGTERM, but SIGHUP is what an SSH session
    drop sends to a child still attached to the controlling tty.
    """
    import signal as _signal
    module = _load_module()
    seen: list[int] = []

    def fake_signal(sig, handler):
        seen.append(int(sig))
        return None

    monkeypatch.setattr(_signal, "signal", fake_signal)
    module._install_kill_handlers()
    # SIGTERM always present.
    assert _signal.SIGTERM in seen
    # SIGINT always present (POSIX + Windows).
    assert _signal.SIGINT in seen
    # SIGHUP only on POSIX.
    if hasattr(_signal, "SIGHUP"):
        assert _signal.SIGHUP in seen


def test_install_kill_handlers_swallows_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If signal.signal raises ValueError ("only main thread can
    install signal handlers"), the handler-installer must not
    propagate. Defense-in-depth for the unlikely path where this
    runs from a non-main thread."""
    import signal as _signal
    module = _load_module()

    def boom(sig, handler):
        raise ValueError("not main thread")

    monkeypatch.setattr(_signal, "signal", boom)
    # Must not raise.
    module._install_kill_handlers()


def test_terminate_active_children_calls_terminate_join_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sequence is: terminate(); join(timeout); kill() if alive.
    Verifies the survivor branch fires kill()."""
    module = _load_module()

    class _FakeChild:
        def __init__(self, alive_after_join: bool) -> None:
            self._alive_after_join = alive_after_join
            self.calls: list[str] = []

        def terminate(self) -> None:
            self.calls.append("terminate")

        def join(self, timeout: float | None = None) -> None:
            self.calls.append(f"join:{timeout!r}")

        def is_alive(self) -> bool:
            return self._alive_after_join

        def kill(self) -> None:
            self.calls.append("kill")

    cooperative = _FakeChild(alive_after_join=False)
    survivor = _FakeChild(alive_after_join=True)

    monkeypatch.setattr(
        "multiprocessing.active_children",
        lambda: [cooperative, survivor],
    )
    module._terminate_active_children()
    # Both got terminate() + join().
    assert "terminate" in cooperative.calls
    assert any(c.startswith("join") for c in cooperative.calls)
    assert "terminate" in survivor.calls
    assert any(c.startswith("join") for c in survivor.calls)
    # Only the survivor got kill().
    assert "kill" not in cooperative.calls
    assert "kill" in survivor.calls


def test_terminate_active_children_swallows_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child whose terminate/join/kill raises (e.g. already dead,
    permission error) must not stop the loop — survivors after it
    still get cleaned up."""
    module = _load_module()

    class _BoomChild:
        def terminate(self) -> None:
            raise RuntimeError("boom-terminate")

        def join(self, timeout: float | None = None) -> None:
            raise RuntimeError("boom-join")

        def is_alive(self) -> bool:
            return True

        def kill(self) -> None:
            raise RuntimeError("boom-kill")

    cleaned = {"terminate": False, "kill": False}

    class _OkChild:
        def terminate(self) -> None:
            cleaned["terminate"] = True

        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return True

        def kill(self) -> None:
            cleaned["kill"] = True

    monkeypatch.setattr(
        "multiprocessing.active_children",
        lambda: [_BoomChild(), _OkChild()],
    )
    module._terminate_active_children()
    assert cleaned["terminate"]
    assert cleaned["kill"]


def test_signal_handler_terminates_children_then_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SIGTERM/SIGHUP handler must call _terminate_active_children
    and then os._exit(128 + signum). It is critical that os._exit
    is called even if children-cleanup raises — otherwise a wedged
    child would block shutdown forever."""
    module = _load_module()
    called = {"term": 0}

    def fake_term() -> None:
        called["term"] += 1

    monkeypatch.setattr(module, "_terminate_active_children", fake_term)

    with mock.patch.object(module.os, "_exit") as mock_exit:
        module._signal_killchildren_and_exit(15, None)

    assert called["term"] == 1
    mock_exit.assert_called_once_with(128 + 15)


def test_signal_handler_exits_even_if_terminate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a hung _terminate_active_children must not stop
    the os._exit. Otherwise a SIGTERM could be partially handled
    and we'd hang in interpreter shutdown."""
    module = _load_module()

    def boom_term() -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(module, "_terminate_active_children", boom_term)
    with mock.patch.object(module.os, "_exit") as mock_exit:
        module._signal_killchildren_and_exit(1, None)
    mock_exit.assert_called_once_with(128 + 1)


def test_main_installs_kill_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shard wrapper's main() should call _install_kill_handlers
    before _run executes — otherwise SIGTERM during work doesn't
    clean up children. Verifies the wiring."""
    module = _load_module()
    called = {"installed": False}

    def fake_install() -> None:
        called["installed"] = True

    monkeypatch.setattr(module, "_install_kill_handlers", fake_install)
    monkeypatch.setattr(module, "_run", lambda _argv: None)
    monkeypatch.setattr(module, "_terminate_active_children", lambda: None)

    with mock.patch.object(module.os, "_exit"):
        module.main([])

    assert called["installed"]


def test_main_terminates_children_before_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clean-exit path also runs _terminate_active_children. If a
    pool worker is somehow still alive when we hit the finally block,
    we want it dead before os._exit (which detaches it from us)."""
    module = _load_module()
    called = {"term": 0}

    monkeypatch.setattr(module, "_install_kill_handlers", lambda: None)
    monkeypatch.setattr(module, "_run", lambda _argv: None)
    monkeypatch.setattr(
        module, "_terminate_active_children",
        lambda: called.__setitem__("term", called["term"] + 1),
    )

    with mock.patch.object(module.os, "_exit"):
        module.main([])

    assert called["term"] == 1
