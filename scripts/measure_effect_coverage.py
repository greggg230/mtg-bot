"""Measure Rust Effect bridge coverage vs Python's Effect vocabulary.

Produces a side-by-side comparison report of:
  * the set of distinct Python ``Effect`` class names that appear when
    parsing the EoE (+ stellar_sights + special_guests_eoe + duskmourn)
    pool via ``OracleParserCardSource``;
  * the set of Rust ``Effect`` enum variant names parsed out of
    ``rust/src/cards.rs``.

Writes a markdown summary to ``docs/RUST_EFFECT_COVERAGE.md`` (auto-
overwritten on each run) AND prints the same markdown to stdout.

Run with::

    PYTHONPATH=src \\
    ../../mtg-bot/.venv-py313/Scripts/python.exe \\
    scripts/measure_effect_coverage.py

Rust → Python name mapping
--------------------------
Most Rust variants are 1:1 with Python class names by CamelCase
(e.g. ``Effect::DealDamage`` ↔ Python ``DealDamage``,
``Effect::Surveil`` ↔ Python ``Surveil``,
``Effect::DestroyAllCreatures`` ↔ Python ``DestroyAllCreatures``).

The handful of intentional name differences are listed in
``RUST_TO_PYTHON_OVERRIDES`` below — for example, the Rust bridge's
``ReturnToHand`` variant exists as a generic primitive but maps to
Python's ``ReturnTargetCreatureToOwnersHand`` /
``ReturnTargetPermanentYouControlToOwnersHand`` family, so the
override pins it to the most common Python counterpart.

This script does not depend on the Rust extension being built — it
parses ``rust/src/cards.rs`` as plain text. The companion test
``tests/test_effect_coverage.py`` does ``pytest.importorskip
("mtg_bot_engine")`` and is skipped on fresh checkouts.
"""

from __future__ import annotations

import io
import re
import sys
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CARDS_FILE = REPO_ROOT / "rust" / "src" / "cards.rs"
OUTPUT_DOC = REPO_ROOT / "docs" / "RUST_EFFECT_COVERAGE.md"

# Rust variant name -> Python class name override. Empty mapping means
# "use the variant name verbatim" (the 1:1 case). Populate this if a
# Rust variant intentionally maps to a Python class with a different
# spelling.
RUST_TO_PYTHON_OVERRIDES: dict[str, str] = {
    # The Rust bridge's generic ``DealDamage`` is the family-head
    # primitive; Python doesn't instantiate the bare ``DealDamage``
    # base class on any card — only its subclasses appear. The
    # highest-frequency in-pool subclass is
    # ``DealDamageToTargetCreature``.
    "DealDamage": "DealDamageToTargetCreature",
    # The Rust bridge's generic primitive corresponds to the
    # creature/permanent-bounce family on the Python side. Python
    # does not have a top-level ``ReturnToHand`` class; the closest
    # high-frequency counterpart is the bounce-to-owners-hand path.
    "ReturnToHand": "ReturnTargetCreatureToOwnersHand",
    # Rust's ``Mill`` variant is the controller-mills-self primitive;
    # Python's class for that is ``SelfMill``. (Mill-opponent has its
    # own ``MillTargetPlayer`` class on the Python side, which maps
    # 1:1 below.)
    "Mill": "SelfMill",
    # Rust's ``ExileTarget`` is the generic exile-target-creature
    # primitive; the closest high-frequency Python class is
    # ``ExileTargetCreature``.
    "ExileTarget": "ExileTargetCreature",
    # Rust's ``TapTarget`` / ``UntapTarget`` / ``Sacrifice`` /
    # ``DiscardCard`` / ``AddCounters`` / ``DestroyTarget`` have no
    # direct 1:1 Python counterpart with the same name — the Python
    # side uses more specific class names. These map to the closest
    # observed Python class by intent.
    "TapTarget": "TapTargetArtifactOrCreature",
    "UntapTarget": "UntapTargetOtherPermanent",
    "Sacrifice": "SacrificeSelf",
    "DiscardCard": "TargetOpponentDiscardsCasterChoice",
    "AddCounters": "PutPlusOneCounterOnTargetCreature",
    "DestroyTarget": "DestroyTargetCreature",
    # The Rust bridge's ``DamageEachOpponent`` is the generic
    # damage-to-each-opponent primitive; the in-pool Python class is
    # ``DealDamageEachOpponent`` (4 cards). The unrelated
    # ``EachOpponentLosesLife`` is a separate Rust variant with its
    # own 1:1 mapping below.
    "DamageEachOpponent": "DealDamageEachOpponent",
    "DamageEachCreature": "DealNDamageToEachCreature",
    "DealDamageAnyTarget": "DealDamageAnyTarget",
    # Rust's ``ChooseOneMode { modes: Vec<Effect> }`` is the modal
    # "choose one" combinator (CR 700.2 / 700.2d) — Python's
    # ``ModalChooseOne`` (``effects/combinators.py``). Names differ
    # because Rust collapses the "ChooseOne" + "Mode" pair into a
    # single variant name while Python keeps the historical
    # ``Modal*`` prefix.
    "ChooseOneMode": "ModalChooseOne",
    # Rust's ``GrantKeywordToTargetCreatureUntilEOT { keyword }`` is
    # a keyword-agnostic parametric generalization (1:N). Python's
    # current vocabulary spells each keyword as its own class —
    # today only ``GrantUnblockableUEOT`` (3 EoE cards) is in pool,
    # so the override pins to that. Once a future pool ships a
    # ``GrantHasteUEOT`` / ``GrantFlyingUEOT`` class, the single
    # Rust variant will cover those Python classes too; the
    # override only nominates the highest-frequency current member
    # for the coverage-by-class metric.
    "GrantKeywordToTargetCreatureUntilEOT": "GrantUnblockableUEOT",
}

# Rust variants that have NO direct Python counterpart in the current
# pool because the underlying mechanic doesn't appear on any pool card.
# These are NOT considered orphans for the regression check — they're
# variants the bridge built ahead of demand (e.g. scry, which the EoE
# pool happens not to use). Adding a Rust variant here is a deliberate
# decision; the orphan test stays strict about everything else.
RUST_AHEAD_OF_POOL: set[str] = {
    # No ``Scry`` class in the EoE-set vocab today (Surveil is the
    # in-pool sibling). When a pool with scry-bearing cards is added,
    # remove this entry and add an override or 1:1 ``ScryN`` class.
    "ScryN",
    # The Rust ``SetBasePTUntilEOT { power, toughness }`` variant
    # mirrors Python's ``SetBasePTUntilEOT`` (effects/generic/pump.py,
    # CR 613.4 layer 7b base-P/T override). The Python class is
    # defined but not currently emitted by any pool card — the only
    # in-pool base-P/T setter today is Genemorph Imago's
    # ``SetBasePTConditionalUntilEOT`` (the conditional cousin, which
    # calls ``grant_set_base_pt_ueot`` directly without composing the
    # unconditional class). When a future pool ships a printing that
    # parses to the bare ``SetBasePTUntilEOT``, drop this entry — the
    # 1:1 identity mapping will pick it up.
    "SetBasePTUntilEOT",
}


# ---------------------------------------------------------------- Python side


def _walk_effect(eff, *, classes, examples, parent_card_name, seen_ids):
    if eff is None:
        return
    obj_id = id(eff)
    if obj_id in seen_ids:
        return
    seen_ids.add(obj_id)

    cls_name = type(eff).__name__
    classes[cls_name] += 1
    examples[cls_name].add(parent_card_name)

    if is_dataclass(eff):
        for f in fields(eff):
            try:
                v = getattr(eff, f.name)
            except Exception:
                continue
            _maybe_walk(v, classes, examples, parent_card_name, seen_ids)
    else:
        for v in getattr(eff, "__dict__", {}).values():
            _maybe_walk(v, classes, examples, parent_card_name, seen_ids)


def _maybe_walk(v, classes, examples, parent_card_name, seen_ids):
    if v is None or isinstance(v, (str, bytes, int, float, bool, frozenset)):
        return
    if hasattr(v, "resolve") and hasattr(v, "describe"):
        _walk_effect(
            v,
            classes=classes,
            examples=examples,
            parent_card_name=parent_card_name,
            seen_ids=seen_ids,
        )
        return
    if isinstance(v, (list, tuple, set)):
        for item in v:
            _maybe_walk(item, classes, examples, parent_card_name, seen_ids)
        return
    if is_dataclass(v):
        for f in fields(v):
            try:
                inner = getattr(v, f.name)
            except Exception:
                continue
            _maybe_walk(inner, classes, examples, parent_card_name, seen_ids)


def collect_python_effect_vocab():
    """Return ``(classes: Counter, examples: dict, n_printings: int)``."""
    from mtg_bot.cards.parser.source import OracleParserCardSource

    src = OracleParserCardSource()
    printings = src.get_printings()

    classes: Counter[str] = Counter()
    examples: dict[str, set[str]] = defaultdict(set)
    n_printings = 0

    for name, printing in printings.items():
        n_printings += 1
        seen_ids_card: set[int] = set()
        try:
            _maybe_walk(
                printing.spell_effect, classes, examples, name, seen_ids_card
            )
            for ta in printing.triggered_abilities:
                _maybe_walk(ta, classes, examples, name, seen_ids_card)
            for aa in getattr(printing, "activated_abilities", ()) or ():
                _maybe_walk(aa, classes, examples, name, seen_ids_card)
            for sa in getattr(printing, "static_abilities", ()) or ():
                _maybe_walk(sa, classes, examples, name, seen_ids_card)
        except Exception:
            continue

    return classes, examples, n_printings


# ---------------------------------------------------------------- Rust side


_RUST_VARIANT_RE = re.compile(
    # Match a line that starts a variant: optional whitespace,
    # CamelCase identifier, optional `{ … }` or `(…)` or unit, then
    # comma. We anchor on the property that variants appear as the
    # first non-whitespace identifier on a line inside the enum block,
    # NOT preceded by `///` (doc-comment) or `//` (line comment) or
    # `impl ` / `pub ` etc.
    r"^\s+([A-Z][A-Za-z0-9]+)\s*(\{|,|\()",
    re.MULTILINE,
)


def collect_rust_effect_variants(cards_rs: Path = RUST_CARDS_FILE) -> list[str]:
    """Parse ``rust/src/cards.rs`` and return the list of Effect
    enum variant names in source order. Pure text parse — no Rust
    toolchain needed."""
    text = cards_rs.read_text(encoding="utf-8")

    # Find the start of the `pub enum Effect {` block.
    m = re.search(r"pub enum Effect\s*\{", text)
    if not m:
        raise RuntimeError(f"could not find `pub enum Effect {{` in {cards_rs}")
    start = m.end()

    # Walk forward through balanced braces to find the closing `}` of
    # the enum block.
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        raise RuntimeError(f"unbalanced braces in Effect enum in {cards_rs}")
    block = text[start:i]

    # Strip line and block comments so the variant regex doesn't pick
    # up CamelCase tokens inside docstrings.
    # Remove `///` doc comments and `//` line comments.
    block_no_comments = re.sub(r"//[^\n]*", "", block)
    # Remove `/* ... */` block comments (defensive — none in cards.rs
    # today, but cheap to strip).
    block_no_comments = re.sub(r"/\*.*?\*/", "", block_no_comments, flags=re.S)

    variants: list[str] = []
    seen: set[str] = set()
    for mv in _RUST_VARIANT_RE.finditer(block_no_comments):
        name = mv.group(1)
        # Filter out anything that doesn't look like a variant — skip
        # known false positives like Rust keywords / type names that
        # might appear at line start (none observed today but defensive).
        if name in {"Self", "Some", "None", "Ok", "Err"}:
            continue
        if name not in seen:
            seen.add(name)
            variants.append(name)
    return variants


# ---------------------------------------------------------------- mapping


def map_rust_to_python(variant: str) -> str:
    """Return the Python class name that corresponds to a Rust variant.

    Default: identity (1:1 CamelCase). Override via
    ``RUST_TO_PYTHON_OVERRIDES`` for the documented mismatches.
    """
    return RUST_TO_PYTHON_OVERRIDES.get(variant, variant)


# ---------------------------------------------------------------- markdown


def render_markdown(
    py_classes: Counter,
    py_examples: dict,
    rust_variants: list[str],
    n_printings: int,
) -> str:
    py_class_set = set(py_classes.keys())
    rust_py_names: list[tuple[str, str]] = [
        (v, map_rust_to_python(v)) for v in rust_variants
    ]
    rust_py_name_set = {py for _, py in rust_py_names}

    total_emissions = sum(py_classes.values())

    # Coverage by class-count: classes whose Python name has a Rust
    # counterpart.
    covered_classes = sorted(py_class_set & rust_py_name_set)
    uncovered_classes = sorted(py_class_set - rust_py_name_set)

    # Orphan check: a Rust variant is an "orphan" iff its Python
    # counterpart name doesn't appear anywhere in the parsed pool AND
    # the variant isn't on the explicit RUST_AHEAD_OF_POOL allowlist
    # (variants the bridge built before any pool card uses them).
    # The orphan listing carries both names so the warning shows the
    # Rust side ("which variant is wrong") and the mapped Python side
    # ("what we expected to find on a card").
    rust_orphans: list[tuple[str, str]] = sorted(
        (rust_name, py)
        for rust_name, py in rust_py_names
        if py not in py_class_set and rust_name not in RUST_AHEAD_OF_POOL
    )
    ahead_of_pool = sorted(
        rust_name for rust_name in rust_variants
        if rust_name in RUST_AHEAD_OF_POOL
    )

    n_total = len(py_class_set)
    n_covered = len(covered_classes)
    pct_classes = (100.0 * n_covered / n_total) if n_total else 0.0

    # Coverage by card-emissions weight.
    covered_emissions = sum(py_classes[c] for c in covered_classes)
    pct_emissions = (
        (100.0 * covered_emissions / total_emissions) if total_emissions else 0.0
    )

    # Frequency tiers.
    high_freq = [c for c, n in py_classes.items() if n >= 10]
    mid_freq = [c for c, n in py_classes.items() if 2 <= n < 10]
    long_tail = [c for c, n in py_classes.items() if n == 1]

    high_covered = sorted(c for c in high_freq if c in rust_py_name_set)
    high_uncovered = sorted(c for c in high_freq if c not in rust_py_name_set)
    mid_covered = sum(1 for c in mid_freq if c in rust_py_name_set)
    long_covered = sum(1 for c in long_tail if c in rust_py_name_set)

    lines: list[str] = []
    lines.append(
        "Generated by `scripts/measure_effect_coverage.py` — "
        "DO NOT EDIT MANUALLY. Run the script to refresh."
    )
    lines.append("")
    lines.append("# Rust Effect bridge coverage vs Python vocab")
    lines.append("")
    lines.append(
        "This page is the rolling-progress snapshot for the Rust engine "
        "port's `Effect` enum. It compares the closed-vocabulary Rust "
        "side (variants in `rust/src/cards.rs`) against the open-ended "
        "Python side (every `Effect` subclass observed when the parser "
        "walks the EoE + stellar_sights + special_guests_eoe + "
        "duskmourn pools)."
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        f"- **{n_covered} of {n_total} Python Effect classes covered "
        f"({pct_classes:.1f}%)** — by distinct-class count."
    )
    lines.append(
        f"- **{covered_emissions} of {total_emissions} per-card "
        f"emissions covered ({pct_emissions:.1f}%)** — weighted by "
        f"how many cards each class appears on. This is the more "
        f"load-bearing metric: high-frequency classes pull the bulk "
        f"of cards into Rust even at low class-count coverage."
    )
    lines.append(
        f"- **{len(rust_variants)} Rust `Effect` variants** in "
        f"`rust/src/cards.rs` today; {n_printings} parser-claimed "
        f"printings across the loaded pools."
    )
    if rust_orphans:
        orphan_strs = ", ".join(
            f"`Effect::{r}` -> `{p}`" for r, p in rust_orphans
        )
        lines.append(
            f"- **WARNING: {len(rust_orphans)} Rust variant(s) with no "
            f"Python counterpart**: {orphan_strs}. "
            f"This indicates bridge drift — either Python emits the class "
            f"under a different name, or the Rust variant is unused. Audit "
            f"`RUST_TO_PYTHON_OVERRIDES` / `RUST_AHEAD_OF_POOL` in "
            f"`scripts/measure_effect_coverage.py`."
        )
    else:
        lines.append(
            "- **0 Rust variants without a Python counterpart** — the "
            "bridge is in sync with the Python vocabulary."
        )
    if ahead_of_pool:
        lines.append(
            f"- **{len(ahead_of_pool)} Rust variant(s) built ahead of "
            f"the pool** (allowlisted in `RUST_AHEAD_OF_POOL`): "
            f"{', '.join('`' + n + '`' for n in ahead_of_pool)}. These "
            f"have no pool card exercising them today and are NOT "
            f"counted as orphans."
        )
    lines.append("")

    lines.append("## Coverage by frequency tier")
    lines.append("")
    lines.append(
        f"Python Effect classes are split by how many cards they "
        f"appear on. The Rust port has prioritized the high-frequency "
        f"head where cards-covered-per-variant is highest."
    )
    lines.append("")

    # Tier 1: high-frequency.
    lines.append(f"### Frequency >= 10 ({len(high_freq)} classes)")
    lines.append("")
    lines.append(
        f"- Covered by Rust: **{len(high_covered)} / {len(high_freq)}** "
        f"({100.0 * len(high_covered) / max(len(high_freq), 1):.1f}%)"
    )
    lines.append("")
    lines.append("| Python class | Frequency | Rust support |")
    lines.append("|---|---:|---|")
    high_freq_sorted = sorted(high_freq, key=lambda c: (-py_classes[c], c))
    for c in high_freq_sorted:
        mark = "yes" if c in rust_py_name_set else "no"
        lines.append(f"| `{c}` | {py_classes[c]} | {mark} |")
    lines.append("")

    # Tier 2: mid-frequency.
    lines.append(f"### Frequency 2-9 ({len(mid_freq)} classes)")
    lines.append("")
    lines.append(
        f"- Covered by Rust: **{mid_covered} / {len(mid_freq)}** "
        f"({100.0 * mid_covered / max(len(mid_freq), 1):.1f}%)"
    )
    lines.append("")

    # Tier 3: long-tail.
    lines.append(f"### Frequency 1 ({len(long_tail)} singletons)")
    lines.append("")
    lines.append(
        f"- Covered by Rust: **{long_covered} / {len(long_tail)}** "
        f"({100.0 * long_covered / max(len(long_tail), 1):.1f}%)"
    )
    lines.append(
        "- The long tail is dominated by bespoke per-card classes; "
        "porting these to Rust before the generic vocabulary fills out "
        "would be poor return on investment."
    )
    lines.append("")

    # Rust variant inventory.
    lines.append("## Rust variants and their Python counterparts")
    lines.append("")
    lines.append("| # | Rust variant | Python class | Python frequency |")
    lines.append("|---:|---|---|---:|")
    for i, (rust_name, py_name) in enumerate(rust_py_names, start=1):
        freq = py_classes.get(py_name, 0)
        lines.append(f"| {i} | `Effect::{rust_name}` | `{py_name}` | {freq} |")
    lines.append("")

    # Uncovered list — abridged because it's the bulk of the work
    # ahead, and we already show the high-frequency uncovered head in
    # the tier table.
    if high_uncovered:
        lines.append("## High-frequency Python classes NOT yet in Rust")
        lines.append("")
        lines.append(
            "These are the next-best ports by cards-covered-per-variant "
            "and are the recommended top of the to-do queue:"
        )
        lines.append("")
        lines.append("| Python class | Frequency | Example card |")
        lines.append("|---|---:|---|")
        for c in sorted(high_uncovered, key=lambda c: (-py_classes[c], c)):
            ex = sorted(py_examples.get(c, set()))[0] if py_examples.get(c) else "?"
            lines.append(f"| `{c}` | {py_classes[c]} | {ex} |")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entrypoint


def main():
    py_classes, py_examples, n_printings = collect_python_effect_vocab()
    rust_variants = collect_rust_effect_variants()
    md = render_markdown(py_classes, py_examples, rust_variants, n_printings)

    OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOC.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
