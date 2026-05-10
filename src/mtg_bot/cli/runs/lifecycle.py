"""High-level lifecycle helpers for subcommand wiring.

The Phase 1 brief asks every ``mtg_bot`` subcommand (``iterate``,
``train``, ``eval``, ``capture``, ``bench``, ``shard``) to push a
start_run / progress / finish_run trio into the run database
alongside its existing JSON writes. This module is the convenience
layer that does the standard wrapping:

- Compute / pass-through the git commit so the run row has a
  reproducible code-version anchor.
- Wrap the body in a try/finally so killed runs get marked
  ``status=killed`` (not left dangling at ``running``) and crashed
  runs get marked ``status=failed``.
- Register the existing JSON file as an artifact so the DB knows
  where to find it.

Each subcommand calls one of the high-level functions below, e.g.
``run_lifecycle("eval", run_id, config_dict, progress_json_path,
work=lambda: run_distributed_eval(...))``. The work callable runs
in a try/finally; its return value becomes the lifecycle return
value.

Usage stays opt-in via the env vars MTG_BOT_RUN_DB_PATH /
MTG_BOT_RUN_DB_URL — when unconfigured every helper is a no-op
(see :mod:`mtg_bot.cli.runs.client`).
"""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

from mtg_bot.cli.runs import client


T = TypeVar("T")


def git_commit_short() -> str:
    """Best-effort current git short SHA. Returns ``"unknown"`` if
    git isn't on PATH or this isn't a git checkout — never raises,
    so subcommand startup never blocks on a missing git toolchain
    (the run row's ``git_commit`` column is informative, not
    structural).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return "unknown"


def serialize_config(config: dict[str, Any]) -> str:
    """Serialize a config dict to JSON suitable for the
    ``runs.config_json`` column. Non-JSON-serializable values
    (``Path``, ``datetime``, ...) are str()'d via the ``default``
    callback so subcommands don't have to pre-stringify."""
    return json.dumps(config, sort_keys=True, default=str)


@contextmanager
def run_lifecycle(
    *,
    run_id: str,
    kind: str,
    config: dict[str, Any],
    progress_json_path: Optional[Path] = None,
    parent_run_id: Optional[str] = None,
    label: Optional[str] = None,
) -> Iterator[None]:
    """Context manager that wraps a subcommand's body with the
    start_run / finish_run pair (and registers the existing
    progress JSON as an artifact if the path is provided).

    Status mapping:
    - normal exit -> ``completed``
    - ``KeyboardInterrupt`` or ``SystemExit`` -> ``killed`` (re-raises)
    - any other exception -> ``failed`` (re-raises)

    Re-raising is important: the lifecycle helper is purely
    supplementary; the actual work's exception path must continue
    to propagate so the caller's existing CI / orchestrator behavior
    is unchanged.
    """
    client.start_run(
        run_id,
        kind=kind,
        config_json=serialize_config(config),
        git_commit=git_commit_short(),
        parent_run_id=parent_run_id,
        label=label,
    )
    if progress_json_path is not None:
        register_progress_artifact(run_id, progress_json_path)
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        client.finish_run(run_id, "killed")
        raise
    except BaseException as exc:  # noqa: BLE001
        client.finish_run(run_id, "failed", notes=f"{type(exc).__name__}: {exc}")
        raise
    else:
        client.finish_run(run_id, "completed")


def register_progress_artifact(
    run_id: str, progress_json_path: Path,
) -> None:
    """Register the existing-on-disk progress JSON as an artifact.
    The path may not yet exist when called (subcommands typically
    create the JSON during run_distributed_*); we register
    optimistically with ``bytes=None`` and let any later
    re-registration (post-finalize) overwrite via the upsert."""
    client.push_artifact(
        run_id,
        artifact_type="progress_json",
        path=str(progress_json_path),
        bytes_=(
            progress_json_path.stat().st_size
            if progress_json_path.exists() else None
        ),
    )


def push_eval_summary(
    run_id: str,
    *,
    wins_new: int,
    wins_old: int,
    draws: int,
    crashed: int = 0,
    elapsed_seconds: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Push a single aggregate progress row summarizing an eval run.
    Used by ``mtg_bot eval`` after ``run_distributed_eval`` returns.
    """
    client.push_progress(
        run_id,
        n_games_done=wins_new + wins_old + draws + crashed,
        n_games_total=wins_new + wins_old + draws + crashed,
        elapsed_seconds=elapsed_seconds,
        wins_new=wins_new,
        wins_old=wins_old,
        draws=draws,
        crashed=crashed,
        extra=extra,
    )


__all__ = [
    "git_commit_short",
    "push_eval_summary",
    "register_progress_artifact",
    "run_lifecycle",
    "serialize_config",
]
