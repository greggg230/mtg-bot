"""Run-record persistence for ``mtg_bot`` subcommands.

Phase 1 of the JSON -> SQLite migration: every ``mtg_bot`` subcommand
that currently writes a progress JSON also dual-writes a row into the
SQLite ``runs.db`` under ``training_progress/``. The training UI keeps
reading JSON for now (Phase 2 swaps the UI), so this package is purely
additive — failures to write the DB are logged but never abort the
underlying run.

Public API lives in :mod:`mtg_bot.cli.runs.client`:

- :func:`client.start_run` — register a new run
- :func:`client.push_progress` — upsert a progress row
- :func:`client.push_event` — append an event
- :func:`client.push_artifact` — register an artifact
- :func:`client.finish_run` — mark a run terminal

The same client API works in-process (writes directly via
:mod:`mtg_bot.cli.runs.db`) AND across hosts (POSTs to the master's
``serve`` endpoints). The ``MTG_BOT_RUN_DB_URL`` env var picks REST;
``MTG_BOT_RUN_DB_PATH`` picks direct SQLite. If neither is set, the
client is a no-op (caller's existing JSON writes still run unchanged).
"""

from mtg_bot.cli.runs import client, db, lifecycle, models

__all__ = ["client", "db", "lifecycle", "models"]
