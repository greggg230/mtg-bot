"""``mtg_bot db backup|dump`` — admin tooling for the run database.

``backup`` uses ``VACUUM INTO`` for an online consistent snapshot —
no need to stop the master to grab a copy. The output file is a
plain SQLite DB with the same schema and row counts.

``dump`` pretty-prints a single run's record (run + progress rows +
last 100 events + artifacts) for debugging.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mtg_bot.cli.runs import db as _db


_DEFAULT_DB_PATH = Path("training_progress") / "runs.db"
_DEFAULT_BACKUP_DIR = Path("training_progress") / "backups"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_db(src: Path, dest: Path) -> Path:
    """Online backup ``src`` -> ``dest`` via ``VACUUM INTO``. Returns
    the destination path. Creates ``dest.parent`` if missing.

    ``VACUUM INTO`` is the SQLite-blessed safe-while-live backup —
    it acquires a read lock, copies + compacts pages into the new
    file, and releases. Concurrent writers see no interruption.
    """
    src = Path(src)
    dest = Path(dest)
    if not src.is_file():
        raise FileNotFoundError(f"runs DB not found at {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # VACUUM INTO refuses to overwrite. Surface a clear error
        # rather than the cryptic "output file already exists" one.
        raise FileExistsError(f"backup destination already exists: {dest}")
    conn = _db.connect(src)
    try:
        # ``VACUUM INTO`` does not allow placeholder binding for the
        # path argument in older SQLite. Inline-quote with the SQL-
        # safe single-quote escape — the path is operator-controlled
        # (not user-network input) so injection isn't a concern, but
        # we still escape defensively.
        escaped = str(dest).replace("'", "''")
        conn.execute(f"VACUUM INTO '{escaped}'")
    finally:
        conn.close()
    return dest


def dump_run(db_path: Path, run_id: str) -> dict:
    """Read the full record for ``run_id``. Returns the same shape
    that ``GET /run/<run_id>`` returns; raises if the run is not
    found."""
    conn = _db.connect(db_path)
    try:
        record = _db.get_run(conn, run_id)
    finally:
        conn.close()
    if record is None:
        raise KeyError(f"no run with run_id {run_id!r} in {db_path}")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    """``mtg_bot db <subcommand>`` dispatcher."""
    parser = argparse.ArgumentParser(
        prog="mtg_bot db",
        description="Run database admin tooling.",
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

    args = parser.parse_args(argv)

    if args.action == "backup":
        dest = args.out
        if dest is None:
            dest = _DEFAULT_BACKUP_DIR / f"runs.db.{_timestamp()}.bak"
        out = backup_db(args.db, dest)
        size = out.stat().st_size
        print(f"backup ok: {out} ({size:,} bytes)")
        return
    if args.action == "dump":
        record = dump_run(args.db, args.run_id)
        print(json.dumps(record, indent=2, sort_keys=True, default=str))
        return


if __name__ == "__main__":  # pragma: no cover
    main()
