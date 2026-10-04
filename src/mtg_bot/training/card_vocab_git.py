"""Reconstruct the card vocabulary a past commit's tokenizer used.

Pre-fingerprint checkpoints and shards record no vocabulary, but they
record (or their operator knows) the commit they came from. The
vocabulary at a commit is reproducible exactly: extract that commit's
``src/mtg_bot`` + ``data/`` with ``git archive`` into a scratch
directory and ask THAT code's own tokenizer for its name -> id map
(``state_tokenizer._printing_to_id_map``) in a subprocess whose
``PYTHONPATH`` points at the extracted tree. Nothing about the current
checkout leaks in, and nothing in the repository or the working tree is
modified.

Used by ``mtg_bot card-vocab export`` / ``stamp``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from mtg_bot.training.card_vocab import CardVocab, CardVocabError

POOL_ENV_VAR = "MTG_BOT_CARD_POOL"
"""The card-pool selector env var of the commits that have a pool
registry (``mtg_bot.cards.pools`` on the cube/roto branches). Spelled
out here because the reconstructed commit may or may not have that
module; the snippet refuses a pool on a commit without one rather than
silently ignoring it."""

_REPO_DEFAULT = Path(__file__).resolve().parents[3]

# Runs inside the extracted commit. Prints exactly one JSON line.
_SNIPPET = r"""
import json, os, sys
pool = os.environ.get(%(env)r)
info = {"pool": pool or "default"}
try:
    import mtg_bot.cards.pools as _pools
except ImportError:
    _pools = None
if pool:
    if _pools is None:
        print(json.dumps({"error": "this commit has no card-pool registry "
                          "(mtg_bot.cards.pools), so a pool cannot be selected"}))
        sys.exit(3)
    _pools.resolve_pool_name(pool)
if _pools is not None:
    info["active_pool"] = _pools.active_pool_name()
from mtg_bot.training import state_tokenizer as st
from mtg_bot.training.structured_model import DEFAULT_N_PRINTINGS
items = sorted(st._printing_to_id_map().items(), key=lambda kv: kv[1])
if [i for _, i in items] != list(range(1, len(items) + 1)):
    print(json.dumps({"error": "printing ids are not 1..N"}))
    sys.exit(3)
info["names"] = [n for n, _ in items]
info["default_n_printings"] = int(DEFAULT_N_PRINTINGS)
print(json.dumps(info))
""" % {"env": POOL_ENV_VAR}


class CardVocabReconstructionError(CardVocabError):
    """The vocabulary at a commit could not be reconstructed."""


@dataclass(frozen=True)
class ReconstructedVocab:
    vocab: CardVocab
    git_commit: str  # full sha
    pool: str  # "default" or the pool name
    default_n_printings: int

    def provenance(self) -> dict:
        return {
            "method": "reconstructed_from_git",
            "git_commit": self.git_commit,
            "pool": self.pool,
            "default_n_printings": self.default_n_printings,
        }


def _git(repo: Path, args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )


def resolve_commit(commit: str, *, repo: Optional[Path] = None) -> str:
    """Full sha of ``commit`` in ``repo`` (raises if unknown)."""
    repo = Path(repo) if repo is not None else _REPO_DEFAULT
    proc = _git(repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if proc.returncode != 0:
        raise CardVocabReconstructionError(
            f"unknown git commit {commit!r} in {repo}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout.decode("ascii").strip()


def _extract(repo: Path, sha: str, dest: Path) -> None:
    proc = _git(repo, ["archive", "--format=tar", sha, "src/mtg_bot", "data"])
    if proc.returncode != 0:
        raise CardVocabReconstructionError(
            f"git archive {sha[:12]} src/mtg_bot data failed: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:  # pragma: no cover - Python < 3.12
            tf.extractall(dest)


def reconstruct_card_vocab(
    commit: str,
    *,
    pool: Optional[str] = None,
    repo: Optional[Path] = None,
    python: Optional[str] = None,
) -> ReconstructedVocab:
    """The vocabulary ``commit``'s tokenizer produced under ``pool``
    (``None`` = that commit's default pool)."""
    repo = Path(repo) if repo is not None else _REPO_DEFAULT
    sha = resolve_commit(commit, repo=repo)
    with tempfile.TemporaryDirectory(prefix=f"card_vocab_{sha[:12]}_") as td:
        root = Path(td)
        _extract(repo, sha, root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop(POOL_ENV_VAR, None)
        if pool:
            env[POOL_ENV_VAR] = pool
        proc = subprocess.run(
            [python or sys.executable, "-c", _SNIPPET],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    payload = None
    if lines:
        try:
            payload = json.loads(lines[-1])
        except ValueError:
            payload = None
    if proc.returncode != 0 or not isinstance(payload, dict) or "error" in payload:
        detail = (
            payload.get("error") if isinstance(payload, dict) and "error" in payload
            else (proc.stderr or proc.stdout)[-2000:]
        )
        raise CardVocabReconstructionError(
            f"could not reconstruct the card vocabulary at {sha[:12]}"
            f"{f' (pool {pool})' if pool else ''}: {detail}"
        )
    try:
        vocab = CardVocab(tuple(payload["names"]))
    except CardVocabError as exc:
        raise CardVocabReconstructionError(
            f"commit {sha[:12]}'s printing-id map is not a "
            f"sorted_names_v1 vocabulary: {exc}"
        ) from exc
    return ReconstructedVocab(
        vocab=vocab,
        git_commit=sha,
        pool=str(payload.get("pool") or "default"),
        default_n_printings=int(payload["default_n_printings"]),
    )


__all__ = [
    "CardVocabReconstructionError",
    "ReconstructedVocab",
    "reconstruct_card_vocab",
    "resolve_commit",
]
