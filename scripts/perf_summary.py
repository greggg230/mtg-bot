"""Aggregate the latest ``bench_results/*.csv`` per primitive into a
unified perf summary table — plus a 1000-game smoke wall-time estimate.

Each Rust primitive bench (``scripts/bench_rust_clone.py``,
``scripts/bench_rust_apply_action.py``, ...) writes a per-run CSV
under ``bench_results/<primitive>_<UTC-timestamp>.csv``. Over time
the directory accumulates one CSV per bench invocation; eyeballing
the wall of files for "what's the current per-call cost for each
primitive?" is tedious and noisy. This script does the aggregation
in one place:

- Glob ``bench_results/*.csv``.
- Group files by primitive (filename prefix before the timestamp).
- For each primitive, pick the most recent CSV (by the trailing
  ``YYYYMMDDTHHMMSSZ`` stamp in the filename, falling back to the
  file's mtime if the stamp isn't parseable).
- Print a unified table:
  primitive | state_size | latest_us_per_call | csv_filename
- Compute a "1000-game smoke wall-time estimate" using
  representative per-game call ratios for each primitive. The
  ratios are spelled out in ``CALLS_PER_GAME`` below so a future
  reader can adjust them as the real parity-smoke workload
  evolves.

Each primitive's CSV has its own column schema (clone uses
``us_per_call``, apply_action uses ``apply_us`` for the isolated
apply cost, etc.); the per-primitive column-name mapping lives in
``PRIMITIVE_SPECS``.

Run from a worktree:

    PYTHONPATH=src ../../mtg-bot/.venv-py313/Scripts/python.exe \\
        scripts/perf_summary.py

Optional flags:

- ``--csv PATH``: write the summary to ``PATH`` instead of stdout.
- ``--state-size LABEL``: filter the table to a single state-size
  label (e.g. ``"(5, 7, 30)"``). The 1000-game estimate is still
  computed; it uses the filtered rows if any survive, else falls
  back to a representative midpoint per primitive.
- ``--bench-dir PATH``: override the default ``bench_results/``
  directory (mostly useful for tests).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from dataclasses import dataclass


# --- Primitive specs ---------------------------------------------------
#
# Maps the CSV filename prefix to (display-name, us-per-call column).
# The display-name is what we show in the table; the column is the
# field in the CSV that represents per-call cost in microseconds.
#
# Note: ``apply_action`` reports the *isolated* per-call cost in the
# ``apply_us`` column (combined-minus-clone subtraction inside
# ``bench_rust_apply_action.py``). The unified table treats that as
# the canonical apply_action cost.
#
# ``fingerprint`` and ``legal_decisions`` bench scripts don't yet
# write CSVs at the time of this script's introduction; their entries
# here are forward-looking so the aggregator picks them up the moment
# those scripts grow CSV output (matching the r25 ``<primitive>_<
# timestamp>.csv`` pattern). Primitives with no matching CSV in
# ``bench_results/`` are silently skipped in the per-primitive loop
# but get a "missing" note in the table footer.

@dataclass(frozen=True)
class PrimitiveSpec:
    prefix: str           # filename prefix, e.g. "clone"
    display_name: str     # column value, e.g. "clone"
    us_column: str        # CSV column that holds the per-call us number


PRIMITIVE_SPECS: list[PrimitiveSpec] = [
    PrimitiveSpec("clone", "clone", "us_per_call"),
    PrimitiveSpec("apply_action", "apply_action", "apply_us"),
    PrimitiveSpec("fingerprint", "fingerprint", "fp_us"),
    PrimitiveSpec("legal_decisions", "legal_decisions", "us_per_call"),
]

# Map prefix -> spec, but sorted longest-first so e.g. ``apply_action_*``
# doesn't accidentally match the ``apply_*`` prefix if one ever exists.
_PREFIX_ORDER = sorted(
    PRIMITIVE_SPECS, key=lambda s: len(s.prefix), reverse=True
)


# --- 1000-game wall-time estimate --------------------------------------
#
# Representative per-game call counts for the parity-smoke trajectory
# walker. These are ROUGH order-of-magnitude estimates; the exact
# numbers vary per game depending on stack depth, board complexity,
# and lookahead. Adjust here if profiling later disagrees.
#
# - ``fingerprint``: called once per step for the cross-engine
#   trajectory equality check. ~200 steps/game is a typical small-
#   format game length.
# - ``apply_action``: also once per step (one decision applied per
#   tick). Mirrors the fingerprint count.
# - ``legal_decisions``: once per step to enumerate the legal moves
#   for the active player.
# - ``clone``: dominated by MCTS-like lookahead — if/when the
#   trajectory walker grows a search-side branch, clone cost scales
#   with branching. ~1500/game is a placeholder for that future
#   integration; today's parity-smoke trajectory walker is closer to
#   ~200 clones/game (one per step). The 1500 number deliberately
#   over-estimates so the wall-time number stays a conservative
#   upper bound; tighten this once parity_smoke gains a real search.

CALLS_PER_GAME: dict[str, int] = {
    "fingerprint": 200,
    "apply_action": 200,
    "legal_decisions": 200,
    "clone": 1500,
}

GAMES_FOR_ESTIMATE = 1000

# The state size we benchmark for the wall-time estimate. Bench CSVs
# carry multiple rows (one per state-size grid entry); for the smoke
# estimate we pick a "representative midpoint" — close to what the
# trajectory walker actually drives.
REPRESENTATIVE_STATE_SIZE = "(5, 7, 30)"


# --- Filename / CSV plumbing -------------------------------------------

_TIMESTAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)\.csv$")


def parse_primitive_from_filename(name: str) -> str | None:
    """Return the primitive prefix for ``name`` (e.g. ``clone``) or
    ``None`` if no known primitive matches.

    Matches against ``PRIMITIVE_SPECS`` longest-prefix-first so that
    e.g. ``apply_action_20260511T...csv`` resolves to ``apply_action``
    and not a shorter accidental prefix.
    """
    for spec in _PREFIX_ORDER:
        if name.startswith(spec.prefix + "_"):
            return spec.prefix
    return None


def spec_for_prefix(prefix: str) -> PrimitiveSpec | None:
    for spec in PRIMITIVE_SPECS:
        if spec.prefix == prefix:
            return spec
    return None


def parse_timestamp_from_filename(name: str) -> str | None:
    """Return the trailing ``YYYYMMDDTHHMMSSZ`` stamp from ``name``,
    or ``None`` if the name doesn't carry one.
    """
    m = _TIMESTAMP_RE.search(name)
    return m.group(1) if m else None


def find_latest_per_primitive(
    bench_dir: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Glob CSVs under ``bench_dir`` and return ``{primitive: latest_csv}``.

    "Latest" prefers the trailing timestamp stamp in the filename
    (deterministic, machine-clock-independent for replay-style debug
    cases). If a CSV doesn't carry a parseable stamp the file's
    mtime is used instead — that keeps hand-renamed test fixtures
    workable.
    """
    by_primitive: dict[str, list[tuple[str, pathlib.Path]]] = {}
    for path in sorted(bench_dir.glob("*.csv")):
        primitive = parse_primitive_from_filename(path.name)
        if primitive is None:
            continue
        stamp = parse_timestamp_from_filename(path.name)
        if stamp is None:
            # mtime fallback keeps the comparator monotonic.
            stamp = f"mtime:{path.stat().st_mtime:020.6f}"
        by_primitive.setdefault(primitive, []).append((stamp, path))

    latest: dict[str, pathlib.Path] = {}
    for primitive, entries in by_primitive.items():
        entries.sort(key=lambda t: t[0])
        latest[primitive] = entries[-1][1]
    return latest


@dataclass
class SummaryRow:
    primitive: str
    state_size: str
    latest_us_per_call: float
    csv_filename: str


def rows_from_csv(
    primitive: str, csv_path: pathlib.Path
) -> list[SummaryRow]:
    """Parse ``csv_path`` and return one ``SummaryRow`` per CSV row.

    The us-per-call column name varies per primitive; we look it up
    in ``PRIMITIVE_SPECS``. Rows missing the expected column are
    skipped with a note printed to stderr (rather than crashing) —
    a partial CSV is more useful than a hard fail.
    """
    spec = spec_for_prefix(primitive)
    if spec is None:
        return []

    out: list[SummaryRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if spec.us_column not in row:
                print(
                    f"perf_summary: skipping row in {csv_path.name} — "
                    f"missing column {spec.us_column!r}",
                    file=sys.stderr,
                )
                continue
            try:
                us = float(row[spec.us_column])
            except (TypeError, ValueError):
                continue
            label = row.get("label") or row.get("state_size") or "?"
            out.append(
                SummaryRow(
                    primitive=spec.display_name,
                    state_size=label,
                    latest_us_per_call=us,
                    csv_filename=csv_path.name,
                )
            )
    return out


# --- Summary computation -----------------------------------------------

def build_summary(
    bench_dir: pathlib.Path,
    state_size_filter: str | None = None,
) -> tuple[list[SummaryRow], list[str]]:
    """Return ``(rows, missing_primitives)`` for ``bench_dir``.

    ``rows`` is the unified table — one entry per (primitive, state-
    size) pair found in the latest CSV per primitive. ``missing_primit-
    ives`` lists the primitives in ``PRIMITIVE_SPECS`` that have no
    CSV in ``bench_dir``; the caller surfaces those in the output so
    a future reader can tell what's missing vs what's slow.
    """
    if not bench_dir.exists():
        return [], [s.prefix for s in PRIMITIVE_SPECS]

    latest = find_latest_per_primitive(bench_dir)
    rows: list[SummaryRow] = []
    for spec in PRIMITIVE_SPECS:
        path = latest.get(spec.prefix)
        if path is None:
            continue
        primitive_rows = rows_from_csv(spec.prefix, path)
        if state_size_filter is not None:
            primitive_rows = [
                r for r in primitive_rows
                if r.state_size == state_size_filter
            ]
        rows.extend(primitive_rows)

    missing = [
        s.prefix for s in PRIMITIVE_SPECS if s.prefix not in latest
    ]
    return rows, missing


def estimate_1000_game_wall_seconds(
    rows: list[SummaryRow],
    representative_state_size: str = REPRESENTATIVE_STATE_SIZE,
) -> tuple[float, dict[str, float]]:
    """Estimate wall-time in seconds for ``GAMES_FOR_ESTIMATE`` parity-
    smoke games using the per-primitive call ratios in ``CALLS_PER_GAME``
    and the per-call us numbers from ``rows``.

    Returns ``(total_seconds, per_primitive_seconds)``. For each
    primitive, we pick the row at ``representative_state_size`` if
    present, else fall back to the median per-primitive us-per-call.
    Primitives missing from ``rows`` contribute zero (and a "missing"
    line in the printed summary surfaces them).
    """
    # Group rows by primitive for lookup.
    by_primitive: dict[str, list[SummaryRow]] = {}
    for r in rows:
        by_primitive.setdefault(r.primitive, []).append(r)

    per_primitive_seconds: dict[str, float] = {}
    total = 0.0
    for primitive, calls_per_game in CALLS_PER_GAME.items():
        prim_rows = by_primitive.get(primitive, [])
        if not prim_rows:
            per_primitive_seconds[primitive] = 0.0
            continue
        # Prefer the representative state-size row; else median.
        match = next(
            (r for r in prim_rows if r.state_size == representative_state_size),
            None,
        )
        if match is not None:
            us = match.latest_us_per_call
        else:
            sorted_us = sorted(r.latest_us_per_call for r in prim_rows)
            us = sorted_us[len(sorted_us) // 2]
        seconds = (
            us
            * calls_per_game
            * GAMES_FOR_ESTIMATE
            / 1_000_000.0
        )
        per_primitive_seconds[primitive] = seconds
        total += seconds
    return total, per_primitive_seconds


# --- Output ------------------------------------------------------------

def format_table(
    rows: list[SummaryRow], missing: list[str]
) -> str:
    """Return the unified table as a string.

    Layout intentionally mirrors the per-bench scripts so the columns
    line up visually for someone who's used to reading them.
    """
    lines: list[str] = []
    lines.append(
        "  primitive        | state_size         | "
        "latest us/call | csv_filename"
    )
    lines.append(
        "  -----------------|--------------------|"
        "----------------|------------------------------------------"
    )
    for r in rows:
        lines.append(
            f"  {r.primitive:<16} | {r.state_size:<18} |"
            f"   {r.latest_us_per_call:8.3f} us  | {r.csv_filename}"
        )
    if not rows:
        lines.append("  (no primitives with matching CSVs found)")
    if missing:
        lines.append("")
        lines.append(
            "  missing primitives (no CSV in bench_results/): "
            + ", ".join(missing)
        )
    return "\n".join(lines)


def format_estimate(
    total_seconds: float, per_primitive: dict[str, float]
) -> str:
    """Format the 1000-game wall-time estimate, with the per-primitive
    math spelled out so a future reader can tweak the ratios.
    """
    lines: list[str] = []
    lines.append(
        f"{GAMES_FOR_ESTIMATE}-game parity-smoke wall-time estimate"
    )
    lines.append(
        f"  representative state size: {REPRESENTATIVE_STATE_SIZE}"
    )
    lines.append("  per-primitive math:")
    for primitive, calls_per_game in CALLS_PER_GAME.items():
        seconds = per_primitive.get(primitive, 0.0)
        lines.append(
            f"    {primitive:<16}  {calls_per_game:>5} calls/game * "
            f"{GAMES_FOR_ESTIMATE} games  ->  {seconds:7.2f} s"
        )
    lines.append("")
    lines.append(f"  total: {total_seconds:7.2f} s  "
                 f"(~{total_seconds / 60.0:.2f} min)")
    return "\n".join(lines)


def write_summary_csv(
    rows: list[SummaryRow], out_path: pathlib.Path
) -> None:
    """Write the unified summary to ``out_path`` as a CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["primitive", "state_size", "latest_us_per_call", "csv_filename"]
        )
        for r in rows:
            writer.writerow(
                [
                    r.primitive,
                    r.state_size,
                    f"{r.latest_us_per_call:.6f}",
                    r.csv_filename,
                ]
            )


# --- CLI ---------------------------------------------------------------

def _default_bench_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "bench_results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate bench_results/*.csv per primitive into a "
            "unified perf summary table."
        )
    )
    parser.add_argument(
        "--bench-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory containing bench_results/*.csv "
            "(default: repo-root bench_results/)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help=(
            "Write the summary table to this CSV path instead of "
            "stdout. The wall-time estimate still prints to stdout."
        ),
    )
    parser.add_argument(
        "--state-size",
        type=str,
        default=None,
        help=(
            "Filter the table to a single state-size label (e.g. "
            "'(5, 7, 30)'). The wall-time estimate is unaffected "
            "structurally — it always uses the representative state "
            "size if present."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bench_dir = args.bench_dir or _default_bench_dir()
    rows, missing = build_summary(bench_dir, args.state_size)

    if args.csv is not None:
        write_summary_csv(rows, args.csv)
        print(f"wrote summary: {args.csv}")
    else:
        print(format_table(rows, missing))
    print()
    total, per_primitive = estimate_1000_game_wall_seconds(rows)
    print(format_estimate(total, per_primitive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
