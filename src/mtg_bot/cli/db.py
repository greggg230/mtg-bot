"""``mtg_bot db <action>`` — run-database admin dispatcher.

Sub-actions:

- ``backup``  — online ``VACUUM INTO`` snapshot of ``runs.db``
- ``dump``    — pretty-print one run's full record (run +
  progress + last 100 events + artifacts), the same shape that
  ``GET /run/<id>`` returns.
- ``query``   — read-only ad-hoc SQL escape hatch. Refuses
  statements that aren't a single ``SELECT``.

The ``backup`` and ``dump`` actions delegate to
:mod:`mtg_bot.cli.runs.backup` (where the heavy lifting lives, plus
the existing tests that pin its behavior). ``query`` is implemented
here because it doesn't fit alongside the backup helpers — it's an
operator escape hatch, not a structured tool.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mtg_bot.cli.runs import backup as _backup_mod, db as _runs_db


_DEFAULT_DB_PATH = Path("training_progress") / "runs.db"
_DEFAULT_BACKUP_DIR = Path("training_progress") / "backups"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
    r"VACUUM|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> bool:
    """Cheap guard for ``mtg_bot db query``. ``True`` iff ``sql`` is a
    single statement that begins with SELECT and contains no
    schema-mutating or write keywords. Not a real SQL parser — meant
    only as a tripwire for accidental ``mtg_bot db query "DELETE
    FROM runs"`` operator slips. The DB connection is also opened
    read-only as a belt-and-suspenders backstop."""
    if ";" in sql.rstrip().rstrip(";"):
        # Reject anything that looks like multiple statements (a
        # trailing semicolon is allowed and stripped by sqlite anyway).
        return False
    if not _SELECT_RE.match(sql):
        return False
    if _FORBIDDEN_RE.search(sql):
        return False
    return True


def run_query(db_path: Path, sql: str) -> list[dict]:
    """Execute ``sql`` (must pass :func:`is_safe_select`) against
    ``db_path`` opened read-only, return rows as dicts.

    Read-only mode means even a sneaky write that slipped past the
    regex check (e.g. ``SELECT ... FROM hidden_view_that_writes``)
    fails at the DB layer — defense in depth.
    """
    if not is_safe_select(sql):
        raise ValueError(
            "mtg_bot db query refuses non-SELECT statements; got: "
            f"{sql.strip()[:80]!r}"
        )
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"runs DB not found at {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mtg_bot db",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_backup = sub.add_parser(
        "backup",
        help="Online VACUUM INTO snapshot of runs.db.",
    )
    p_backup.add_argument(
        "--db", type=Path, default=_DEFAULT_DB_PATH,
        help=f"Source DB (default {_DEFAULT_DB_PATH}).",
    )
    p_backup.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Destination file. Default: "
            f"{_DEFAULT_BACKUP_DIR}/runs.db.<UTC-timestamp>.bak"
        ),
    )

    p_dump = sub.add_parser(
        "dump", help="Pretty-print a run's full record.",
    )
    p_dump.add_argument(
        "--db", type=Path, default=_DEFAULT_DB_PATH,
        help=f"Source DB (default {_DEFAULT_DB_PATH}).",
    )
    p_dump.add_argument(
        "--run-id", required=True,
        help="run_id to dump.",
    )

    p_query = sub.add_parser(
        "query",
        help="Run a read-only SELECT against the runs DB.",
    )
    p_query.add_argument(
        "--db", type=Path, default=_DEFAULT_DB_PATH,
        help=f"Source DB (default {_DEFAULT_DB_PATH}).",
    )
    p_query.add_argument(
        "sql",
        help=(
            "A single SELECT statement. Anything else is rejected "
            "(use sqlite3 directly if you really need to write)."
        ),
    )
    p_query.add_argument(
        "--format", choices=["json", "tsv"], default="json",
        help="Output format (default json).",
    )

    args = parser.parse_args(argv)

    if args.action == "backup":
        dest = args.out
        if dest is None:
            dest = _DEFAULT_BACKUP_DIR / f"runs.db.{_timestamp()}.bak"
        out = _backup_mod.backup_db(args.db, dest)
        size = out.stat().st_size
        print(f"backup ok: {out} ({size:,} bytes)")
        return

    if args.action == "dump":
        record = _backup_mod.dump_run(args.db, args.run_id)
        print(json.dumps(record, indent=2, sort_keys=True, default=str))
        return

    if args.action == "query":
        rows = run_query(args.db, args.sql)
        if args.format == "json":
            print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        else:  # tsv
            if not rows:
                return
            cols = list(rows[0].keys())
            print("\t".join(cols))
            for row in rows:
                print("\t".join("" if row[c] is None else str(row[c])
                                for c in cols))
        return


if __name__ == "__main__":  # pragma: no cover
    main()
