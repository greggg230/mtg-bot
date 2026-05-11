"""Generate a catalog of the Rust-engine test surface under ``tests_rust/``.

Walks every ``tests_rust/test_*.py`` file, parses the source via
``ast.parse``, and extracts:

  * the module-level docstring (often a short description of which
    primitive surface the file covers);
  * the list of test function names (``test_*``) and each test's first
    docstring line (if any);
  * a per-file test count and a cross-file total.

Writes a markdown catalog to ``docs/RUST_TEST_INVENTORY.md`` (auto-
overwritten on each run) and, with ``--stdout``, prints the same
markdown to stdout instead of writing the file.

Useful for spotting:

  * redundancy (two files testing the same surface) — sibling-file
    docstrings make this jump out;
  * gaps (a Rust-side primitive with no dedicated test file);
  * onboarding (a single page that maps "primitive surface ->
    where its tests live").

Run with::

    PYTHONPATH=src \\
    ../../mtg-bot/.venv-py313/Scripts/python.exe \\
    scripts/generate_test_inventory.py

This script does NOT depend on the Rust extension being built — it
only parses Python source. The companion test under
``tests/test_generate_test_inventory.py`` runs it end-to-end against
the live ``tests_rust/`` tree.
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_RUST_DIR = REPO_ROOT / "tests_rust"
OUTPUT_DOC = REPO_ROOT / "docs" / "RUST_TEST_INVENTORY.md"


# Files under ``tests_rust/`` that are NOT test modules even though
# they match ``test_*.py``-adjacent file names; conftest / __init__ /
# the directory README are excluded by the glob below already, but
# this set is the explicit safety net in case the glob is ever
# loosened.
NON_TEST_FILES: set[str] = {"conftest.py", "__init__.py"}


# ---------------------------------------------------------------- model


class TestInfo:
    """One ``test_*`` function: name + first-line docstring (or empty)."""

    __slots__ = ("name", "docstring_first_line")

    def __init__(self, name: str, docstring_first_line: str):
        self.name = name
        self.docstring_first_line = docstring_first_line


class FileInfo:
    """One ``test_*.py`` file's collected metadata."""

    __slots__ = ("path", "module_docstring_first_line", "module_docstring", "tests")

    def __init__(
        self,
        path: Path,
        module_docstring_first_line: str,
        module_docstring: str,
        tests: list[TestInfo],
    ):
        self.path = path
        self.module_docstring_first_line = module_docstring_first_line
        self.module_docstring = module_docstring
        self.tests = tests


# ---------------------------------------------------------------- parsing


def _first_docstring_line(docstring: str | None) -> str:
    """Return the first non-empty line of a docstring, or empty string."""
    if not docstring:
        return ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_test_file(path: Path) -> FileInfo:
    """Parse a single test file and extract its module + test metadata."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    module_docstring = ast.get_docstring(tree) or ""
    module_first_line = _first_docstring_line(module_docstring)

    tests: list[TestInfo] = []
    # Top-level functions only — pytest collects top-level ``test_*``
    # functions and any methods on ``Test*`` classes, but the
    # ``tests_rust/`` tree currently uses only top-level functions.
    # If a class-based test style ever appears, extend here.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                tests.append(TestInfo(node.name, _first_docstring_line(doc)))

    return FileInfo(
        path=path,
        module_docstring_first_line=module_first_line,
        module_docstring=module_docstring,
        tests=tests,
    )


def collect_test_files(tests_dir: Path = TESTS_RUST_DIR) -> list[FileInfo]:
    """Walk ``tests_rust/`` and return one ``FileInfo`` per ``test_*.py``."""
    files: list[FileInfo] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name in NON_TEST_FILES:
            continue
        files.append(parse_test_file(path))
    return files


# ---------------------------------------------------------------- guess-description


def _humanize_stem(stem: str) -> str:
    """Turn ``test_play_land`` into ``Play land``."""
    # Drop the ``test_`` prefix.
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    words = stem.split("_")
    if not words:
        return ""
    return " ".join(words).capitalize()


def describe_file(info: FileInfo) -> str:
    """Best-effort one-line description for the summary table.

    Prefer the module docstring's first line; otherwise fall back to
    a humanized version of the filename stem.
    """
    if info.module_docstring_first_line:
        return info.module_docstring_first_line
    return _humanize_stem(info.path.stem) + " (no module docstring)"


# ---------------------------------------------------------------- markdown


def render_markdown(files: list[FileInfo]) -> str:
    total_tests = sum(len(f.tests) for f in files)
    total_files = len(files)

    lines: list[str] = []
    lines.append(
        "Generated by `scripts/generate_test_inventory.py` — "
        "DO NOT EDIT MANUALLY. Run the script to refresh."
    )
    lines.append("")
    lines.append("# Rust-engine test inventory")
    lines.append("")
    lines.append(
        "Auto-generated catalog of every test under `tests_rust/`. For each "
        "file: a one-line description (from the module docstring), the "
        "total test count, and the per-test function names + first-line "
        "docstrings. Use this page to find which file covers a given "
        "primitive surface, spot redundancy, and detect gaps in coverage."
    )
    lines.append("")
    lines.append(
        "The catalog is intentionally shallow — it does NOT execute the "
        "tests, so it works without the Rust extension built. The "
        "companion script reads each file via `ast.parse` and walks the "
        "top-level `test_*` definitions."
    )
    lines.append("")

    # Headline numbers.
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- **{total_files} test files** under `tests_rust/`.")
    lines.append(
        f"- **{total_tests} top-level `test_*` functions** across all files."
    )
    lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| File | Tests | Description |")
    lines.append("|---|---:|---|")
    for info in files:
        desc = describe_file(info)
        # Cap the description at a reasonable length so the table
        # stays readable.
        if len(desc) > 140:
            desc = desc[:137] + "..."
        lines.append(
            f"| [`{info.path.name}`](#{_anchor(info.path.name)}) "
            f"| {len(info.tests)} | {desc} |"
        )
    lines.append("")

    # Per-file detail.
    lines.append("## Per-file detail")
    lines.append("")
    for info in files:
        lines.append(f"### `{info.path.name}`")
        lines.append("")
        if info.module_docstring_first_line:
            lines.append(f"> {info.module_docstring_first_line}")
            lines.append("")
        lines.append(f"**Test count:** {len(info.tests)}")
        lines.append("")
        if not info.tests:
            lines.append("_No top-level `test_*` functions found._")
            lines.append("")
            continue
        lines.append("| # | Test function | First-line docstring |")
        lines.append("|---:|---|---|")
        for i, t in enumerate(info.tests, start=1):
            doc = t.docstring_first_line or "_(no docstring)_"
            if len(doc) > 140:
                doc = doc[:137] + "..."
            lines.append(f"| {i} | `{t.name}` | {doc} |")
        lines.append("")

    # Footer.
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Total files: **{total_files}**")
    lines.append(f"- Total tests: **{total_tests}**")
    lines.append("")

    return "\n".join(lines) + "\n"


def _anchor(filename: str) -> str:
    """GitHub-style markdown anchor for a filename heading.

    GitHub strips backticks and special chars, lowercases, and
    replaces spaces with hyphens. ``test_setup.py`` -> ``test_setuppy``.
    """
    out = filename.lower().replace(".", "")
    # Hyphens for word separators (we don't use any today, but keep
    # the rule honest in case a file with hyphens ever appears).
    out = out.replace(" ", "-")
    return out


# ---------------------------------------------------------------- entrypoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate docs/RUST_TEST_INVENTORY.md from the tests_rust/ tree."
        )
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the markdown to stdout instead of writing the doc file.",
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=TESTS_RUST_DIR,
        help=(
            "Directory to scan (default: tests_rust/ next to this script). "
            "Used by tests that point at a temp tree."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DOC,
        help=(
            "Where to write the markdown when --stdout is not set "
            "(default: docs/RUST_TEST_INVENTORY.md)."
        ),
    )
    args = parser.parse_args(argv)

    files = collect_test_files(args.tests_dir)
    md = render_markdown(files)

    if args.stdout:
        print(md)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output} ({len(files)} files, "
              f"{sum(len(f.tests) for f in files)} tests).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
