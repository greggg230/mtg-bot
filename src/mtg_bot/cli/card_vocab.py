"""``mtg_bot card-vocab`` — inspect, export and stamp card vocabularies.

Every structured checkpoint records the card vocabulary its printing
embeddings were trained on (``metadata.json`` ``card_vocab``); loaders
refuse a checkpoint without one and remap a changed one by card name.
Checkpoints registered before that fingerprint existed have to be
stamped once. See ``docs/CARD_VOCAB_FINGERPRINT.md``.

Actions::

    mtg_bot card-vocab show   --registry-root R (--versions v1 v2 | --all-versions)
    mtg_bot card-vocab show   --version-dir R/v22 [--compare-current]
    mtg_bot card-vocab export (--git-commit SHA [--pool P] | --current) --out FILE
    mtg_bot card-vocab stamp  --registry-root R --all-versions \\
                              --use-metadata-commit --sidecar
    mtg_bot card-vocab stamp  --version-dir R/v22 --git-commit SHA \\
                              --out-dir COPY_DIR
    mtg_bot card-vocab diff   OLD_VOCAB_FILE NEW_VOCAB_FILE [--n-printings 320]

``stamp`` never modifies an existing file: ``--sidecar`` adds
``weights.pt.card_vocab.json`` next to ``weights.pt`` (the original ``weights.pt`` /
``metadata.json`` stay byte-identical; ``Registry.load_metadata``, the
loaders and the coordinator's ``/v2/weights/<v>/metadata`` all honour
it), and ``--out-dir`` / ``--out-root`` write a stamped COPY.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from mtg_bot.training.card_vocab import (
    METADATA_KEY,
    SIDECAR_FILENAME,
    SIDECAR_SUFFIX,
    CardVocab,
    CardVocabError,
    MissingCardVocabError,
    sidecar_path_for,
    write_sidecar,
    load_card_vocab_file,
    resolve_checkpoint_card_vocab,
    row_for_raw_id,
    write_card_vocab_file,
    _now_iso,
)


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--version-dir", type=Path, action="append", default=[],
        help="A version directory (weights.pt + metadata.json). Repeatable.",
    )
    p.add_argument("--registry-root", type=Path, default=None)
    p.add_argument("--versions", nargs="+", default=[])
    p.add_argument(
        "--all-versions", action="store_true",
        help="Every complete version under --registry-root.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtg_bot card-vocab",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True)

    show = sub.add_parser("show", help="Print checkpoints' stored vocabularies.")
    _add_selection(show)
    show.add_argument(
        "--compare-current", action="store_true",
        help="Also compare with the current (active-source) vocabulary.",
    )

    export = sub.add_parser("export", help="Write a vocabulary file.")
    src = export.add_mutually_exclusive_group(required=True)
    src.add_argument("--git-commit", default=None)
    src.add_argument(
        "--current", action="store_true",
        help="The vocabulary of this checkout's active card source.",
    )
    export.add_argument(
        "--pool", default=None,
        help=(
            "Card pool (MTG_BOT_CARD_POOL) to reconstruct under; only "
            "for --git-commit, and only commits with a pool registry."
        ),
    )
    export.add_argument("--repo", type=Path, default=None)
    export.add_argument("--out", type=Path, required=True)

    stamp = sub.add_parser(
        "stamp",
        help="Stamp pre-fingerprint checkpoints with their vocabulary.",
    )
    _add_selection(stamp)
    vsrc = stamp.add_mutually_exclusive_group(required=True)
    vsrc.add_argument(
        "--use-metadata-commit", action="store_true",
        help="Reconstruct at each checkpoint's recorded git_commit.",
    )
    vsrc.add_argument("--git-commit", default=None)
    vsrc.add_argument("--vocab-file", type=Path, default=None)
    stamp.add_argument("--pool", default=None)
    stamp.add_argument("--repo", type=Path, default=None)
    mode = stamp.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sidecar", action="store_true",
        help=(
            f"Write <weights>{SIDECAR_SUFFIX} (registry: {SIDECAR_FILENAME}) "
            f"next to the original weights."
        ),
    )
    mode.add_argument(
        "--out-dir", type=Path, default=None,
        help="Write a stamped copy of ONE --version-dir here.",
    )
    mode.add_argument(
        "--out-root", type=Path, default=None,
        help="Write stamped copies as <out-root>/<version>/.",
    )
    stamp.add_argument(
        "--allow-commit-mismatch", action="store_true",
        help=(
            "Stamp even when the chosen vocabulary differs from (or "
            "cannot be checked against) the vocabulary at the "
            "checkpoint's own recorded git_commit."
        ),
    )
    stamp.add_argument("--dry-run", action="store_true")

    diff = sub.add_parser("diff", help="Compare two vocabulary files.")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    diff.add_argument(
        "--n-printings", type=int, default=None,
        help="Also count shared cards whose embedding ROW moves at this cap.",
    )
    return parser


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _selected_dirs(args, parser) -> list[Path]:
    dirs = [Path(d) for d in args.version_dir]
    if args.registry_root is not None:
        # Enumerated directly rather than via ``Registry(...)``, whose
        # constructor creates the root and re-applies ACL inheritance:
        # this tool must not create or touch anything it only reads.
        root = Path(args.registry_root)
        if not root.is_dir():
            parser.error(f"--registry-root {root} is not a directory")
        if args.all_versions:
            names = _list_versions(root)
        else:
            names = list(args.versions)
        if not names:
            parser.error("--registry-root needs --versions or --all-versions")
        for name in names:
            dirs.append(Path(args.registry_root) / name)
    elif args.versions or args.all_versions:
        parser.error("--versions / --all-versions need --registry-root")
    if not dirs:
        parser.error("select checkpoints with --version-dir or --registry-root")
    for d in dirs:
        if not (d / "weights.pt").is_file() or not (d / "metadata.json").is_file():
            parser.error(f"{d} is not a version dir (weights.pt + metadata.json)")
    return dirs


def _list_versions(root: Path) -> list[str]:
    """Complete ``v<N>`` version dirs under ``root``, oldest first."""
    from mtg_bot.training.registry import _VERSION_RE

    found = []
    for child in root.iterdir():
        match = _VERSION_RE.match(child.name)
        if (
            match is not None and child.is_dir()
            and (child / "weights.pt").is_file()
            and (child / "metadata.json").is_file()
        ):
            found.append((int(match.group(1)), child.name))
    return [name for _, name in sorted(found)]


def _read_metadata(version_dir: Path) -> dict:
    with (version_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


class _Reconstructor:
    """Memoised ``reconstruct_card_vocab`` (one extraction per commit)."""

    def __init__(self, pool: Optional[str], repo: Optional[Path]) -> None:
        self.pool = pool
        self.repo = repo
        self._cache: dict[str, object] = {}

    def at(self, commit: str):
        from mtg_bot.training.card_vocab_git import (
            reconstruct_card_vocab,
            resolve_commit,
        )

        sha = resolve_commit(commit, repo=self.repo)
        if sha not in self._cache:
            print(
                f"# reconstructing card vocabulary at {sha[:12]}"
                f"{f' (pool {self.pool})' if self.pool else ''} ...",
                flush=True,
            )
            self._cache[sha] = reconstruct_card_vocab(
                sha, pool=self.pool, repo=self.repo,
            )
        return self._cache[sha]


# ---------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------


def _cmd_show(args, parser) -> int:
    current = None
    if args.compare_current:
        from mtg_bot.training.card_vocab import current_card_vocab

        current = current_card_vocab()
        print(f"current vocabulary {current.short_sha} ({current.size} names)")
    rc = 0
    for d in _selected_dirs(args, parser):
        meta = _read_metadata(d)
        if meta.get("architecture") != "structured":
            print(f"{d}: architecture {meta.get('architecture')!r} (no card vocabulary)")
            continue
        try:
            resolved = resolve_checkpoint_card_vocab(d / "weights.pt")
        except MissingCardVocabError:
            print(f"{d}: UNSTAMPED (git_commit {meta.get('git_commit')})")
            rc = 1
            continue
        except CardVocabError as exc:
            print(f"{d}: INVALID: {exc}")
            rc = 1
            continue
        line = (
            f"{d}: vocab {resolved.vocab.short_sha} ({resolved.vocab.size} "
            f"names) from {resolved.origin}"
        )
        if current is not None:
            line += (
                "; matches current" if current.sha256 == resolved.vocab.sha256
                else "; DIFFERS from current (will be remapped at load)"
            )
        print(line)
    return rc


def _cmd_export(args, parser) -> int:
    if args.current:
        if args.pool:
            parser.error(
                "--pool applies to --git-commit only; for --current set "
                "MTG_BOT_CARD_POOL in the environment"
            )
        from mtg_bot.training.card_vocab import current_card_vocab

        vocab = current_card_vocab()
        provenance = {"method": "current_checkout", "exported_at": _now_iso()}
    else:
        rec = _Reconstructor(args.pool, args.repo).at(args.git_commit)
        vocab = rec.vocab
        provenance = {**rec.provenance(), "exported_at": _now_iso()}
    write_card_vocab_file(args.out, vocab, provenance=provenance)
    print(
        f"wrote {args.out}: vocab {vocab.short_sha} ({vocab.size} names)"
    )
    return 0


def _stamp_one(
    version_dir: Path,
    vocab: CardVocab,
    provenance: dict,
    args,
) -> str:
    weights = version_dir / "weights.pt"
    meta = _read_metadata(version_dir)
    if args.sidecar:
        target = sidecar_path_for(weights)
        if target.exists():
            existing = resolve_checkpoint_card_vocab(weights)
            if existing.vocab.sha256 != vocab.sha256:
                raise CardVocabError(
                    f"{target} already stamps a different vocabulary "
                    f"({existing.vocab.short_sha})"
                )
            return f"already stamped ({target})"
        if args.dry_run:
            return f"would write {target}"
        write_sidecar(vocab, weights, provenance=provenance)
        check = resolve_checkpoint_card_vocab(weights)
        if check.vocab.sha256 != vocab.sha256:  # pragma: no cover - guard
            raise CardVocabError(f"{target}: stamp did not round-trip")
        return f"wrote {target}"
    out = args.out_dir if args.out_dir is not None else (
        Path(args.out_root) / version_dir.name
    )
    if Path(out).resolve() == version_dir.resolve():
        raise CardVocabError("refusing to stamp in place; pick another --out")
    if out.exists() and any(out.iterdir()):
        raise CardVocabError(f"{out} exists and is not empty")
    if args.dry_run:
        return f"would write a stamped copy to {out}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, out / "weights.pt")
    stamped = dict(meta)
    stamped[METADATA_KEY] = vocab.to_dict(provenance=provenance)
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(stamped, f, indent=2, sort_keys=True)
    check = resolve_checkpoint_card_vocab(out / "weights.pt")
    if check.vocab.sha256 != vocab.sha256:  # pragma: no cover - guard
        raise CardVocabError(f"{out}: stamp did not round-trip")
    return f"wrote stamped copy {out}"


def _cmd_stamp(args, parser) -> int:
    dirs = _selected_dirs(args, parser)
    if args.out_dir is not None and len(dirs) != 1:
        parser.error("--out-dir stamps exactly one --version-dir; use --out-root")
    if args.vocab_file is not None and args.pool:
        parser.error("--pool applies to git reconstruction, not --vocab-file")
    recon = _Reconstructor(args.pool, args.repo)
    fixed: Optional[tuple[CardVocab, dict]] = None
    if args.vocab_file is not None:
        vocab = load_card_vocab_file(args.vocab_file)
        fixed = (vocab, {"method": "vocab_file", "vocab_file": str(args.vocab_file)})
    elif args.git_commit is not None:
        rec = recon.at(args.git_commit)
        fixed = (rec.vocab, rec.provenance())

    failures = 0
    for d in dirs:
        meta = _read_metadata(d)
        label = d.name
        try:
            if meta.get("architecture") != "structured":
                print(f"{label}: skipped (architecture {meta.get('architecture')!r})")
                continue
            if meta.get(METADATA_KEY):
                embedded = CardVocab.from_dict(meta[METADATA_KEY])
                print(f"{label}: already carries vocab {embedded.short_sha} in metadata.json")
                continue
            meta_commit = meta.get("git_commit")
            if fixed is None:
                if not meta_commit:
                    raise CardVocabError(
                        "metadata records no git_commit; pass --git-commit "
                        "or --vocab-file"
                    )
                rec = recon.at(meta_commit)
                vocab, provenance = rec.vocab, rec.provenance()
                check = "is the checkpoint's own git_commit"
            else:
                vocab, provenance = fixed
                if not meta_commit:
                    check = "skipped (metadata records no git_commit)"
                    if not args.allow_commit_mismatch:
                        raise CardVocabError(
                            "cannot cross-check against the checkpoint's "
                            "commit (none recorded); pass "
                            "--allow-commit-mismatch to stamp anyway"
                        )
                else:
                    try:
                        own = recon.at(meta_commit).vocab
                    except CardVocabError as exc:
                        if not args.allow_commit_mismatch:
                            raise CardVocabError(
                                f"cannot reconstruct the vocabulary at the "
                                f"checkpoint's git_commit {meta_commit}: "
                                f"{exc}. Pass --allow-commit-mismatch to "
                                f"stamp anyway."
                            ) from exc
                        own = None
                        check = f"skipped ({meta_commit} not reconstructable)"
                    if own is not None:
                        if own.sha256 != vocab.sha256 and not args.allow_commit_mismatch:
                            raise CardVocabError(
                                f"the chosen vocabulary {vocab.short_sha} "
                                f"({vocab.size} names) differs from the one at "
                                f"the checkpoint's git_commit {meta_commit} "
                                f"({own.short_sha}, {own.size} names). If its "
                                f"training shards really came from elsewhere, "
                                f"pass --allow-commit-mismatch."
                            )
                        check = (
                            "matches the checkpoint's git_commit"
                            if own.sha256 == vocab.sha256
                            else "DIFFERS from the checkpoint's git_commit (allowed)"
                        )
            provenance = {
                **provenance,
                "stamped_at": _now_iso(),
                "stamped_by": "mtg_bot card-vocab stamp",
                "checkpoint_git_commit": meta_commit,
                "commit_check": check,
            }
            result = _stamp_one(d, vocab, provenance, args)
            print(
                f"{label}: vocab {vocab.short_sha} ({vocab.size} names; "
                f"{check}): {result}"
            )
        except CardVocabError as exc:
            failures += 1
            print(f"{label}: FAILED: {exc}", file=sys.stderr)
    if failures:
        print(f"{failures} checkpoint(s) not stamped", file=sys.stderr)
        return 1
    return 0


def _cmd_diff(args, parser) -> int:
    old = load_card_vocab_file(args.old)
    new = load_card_vocab_file(args.new)
    old_ids = old.id_map
    new_ids = new.id_map
    shared = [n for n in new.names if n in old_ids]
    renumbered = [n for n in shared if old_ids[n] != new_ids[n]]
    print(f"old  {old.short_sha} {old.size} names")
    print(f"new  {new.short_sha} {new.size} names")
    print(f"kept {len(shared)}  new {new.size - len(shared)}  dropped {old.size - len(shared)}")
    print(f"shared cards whose raw printing id changes: {len(renumbered)}")
    if args.n_printings is not None:
        n = int(args.n_printings)
        moved = [
            name for name in shared
            if row_for_raw_id(old_ids[name], n) != row_for_raw_id(new_ids[name], n)
        ]
        print(
            f"shared cards whose embedding ROW changes at n_printings={n}: "
            f"{len(moved)}"
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "show":
            return _cmd_show(args, parser)
        if args.action == "export":
            return _cmd_export(args, parser)
        if args.action == "stamp":
            return _cmd_stamp(args, parser)
        return _cmd_diff(args, parser)
    except CardVocabError as exc:
        print(f"card-vocab: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
