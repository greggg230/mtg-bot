"""Extend a trained structured checkpoint with the ``canonical_v3``
resolution slots, keeping everything it learned.

The structured policy head scores one learned query per action slot
(``StructuredModel.policy_slot_queries``, shape ``(vocab, emb_dim)``)
through a shared cross-attention decoder; every other parameter is
independent of the vocabulary size. ``canonical_v3`` appends the
resolution block after ``canonical_v2``'s 517 slots without moving
any of them (``docs/RESOLUTION_DECISIONS.md``), so extending a v2
checkpoint is exactly:

* copy every tensor bit-for-bit,
* append ``RESOLUTION_BLOCK_SIZE`` rows to ``policy_slot_queries``,
  drawn ``N(0, 0.02)`` -- the model's own initialisation -- from a
  ``torch.Generator`` with a recorded, deterministic seed,
* rewrite the metadata: ``action_encoding_version = "canonical_v3"``
  and ``training.structured_action_vocab_size`` (flat and nested), plus
  an ``extra.resolution_extension`` provenance block. Value
  calibration, card vocabulary, trunk dims and training history are
  carried over unchanged.

The extended network produces the bit-identical value and the
identical logits on the 517 old slots (the inference path scores
``policy_slot_range=(0, 517)`` at priority moments -- the same
computation the source checkpoint ran); the new rows start
near-uniform and are trained by the first resolution-aware cycle.

CLI: ``mtg_bot extend-resolution-slots`` (``main`` below).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from mtg_bot.search import resolution_slots
from mtg_bot.training import action_encoding as ae

POLICY_SLOT_QUERIES_KEY = "policy_slot_queries"
INIT_STD = 0.02
"""``StructuredModel`` initialises ``policy_slot_queries`` with
``nn.init.normal_(std=0.02)``; new rows use the same distribution."""

DEFAULT_SEED = 20261005
"""Default ``torch.Generator`` seed for the appended rows. Recorded in
the metadata; pass ``--seed`` to draw different rows."""

EXTENSION_KEY = "resolution_extension"

_EXTENDABLE_VERSIONS = (
    ae.ACTION_ENCODING_VERSION_V1, ae.ACTION_ENCODING_VERSION_V2,
)


class ResolutionExtensionError(ValueError):
    """The source checkpoint can't be extended (wrong architecture,
    already extended, unexpected policy-head width)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_slot_rows(
    n_rows: int, emb_dim: int, *, seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``n_rows`` x ``emb_dim`` rows drawn ``N(0, INIT_STD)`` from a
    generator seeded with ``seed`` (deterministic across processes)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = torch.empty(n_rows, emb_dim, dtype=torch.float32)
    rows.normal_(0.0, INIT_STD, generator=gen)
    return rows.to(dtype)


def extend_state_dict(
    state_dict: dict[str, torch.Tensor], *, seed: int = DEFAULT_SEED,
) -> dict[str, torch.Tensor]:
    """Return a new state dict: every tensor of ``state_dict`` copied
    unchanged, ``policy_slot_queries`` grown from
    ``ACTION_VOCAB_SIZE`` (517) to ``ACTION_VOCAB_SIZE_V3`` rows. Key
    order is preserved."""
    if POLICY_SLOT_QUERIES_KEY not in state_dict:
        raise ResolutionExtensionError(
            f"state dict has no {POLICY_SLOT_QUERIES_KEY!r}: not a "
            f"structured checkpoint"
        )
    queries = state_dict[POLICY_SLOT_QUERIES_KEY]
    if queries.dim() != 2 or queries.shape[0] != ae.ACTION_VOCAB_SIZE:
        raise ResolutionExtensionError(
            f"{POLICY_SLOT_QUERIES_KEY} has shape {tuple(queries.shape)};"
            f" expected ({ae.ACTION_VOCAB_SIZE}, emb_dim) -- only a "
            f"positional_v1 / canonical_v2 head can be extended"
        )
    n_new = ae.ACTION_VOCAB_SIZE_V3 - ae.ACTION_VOCAB_SIZE
    fresh = fresh_slot_rows(
        n_new, int(queries.shape[1]), seed=seed, dtype=queries.dtype,
    ).to(queries.device)
    out: dict[str, torch.Tensor] = type(state_dict)()  # keep OrderedDict
    for key, value in state_dict.items():
        if key == POLICY_SLOT_QUERIES_KEY:
            out[key] = torch.cat([value, fresh], dim=0)
        else:
            out[key] = value
    return out


def extend_metadata(
    metadata: dict[str, Any], *,
    source_version: str,
    source_weights_sha256: str,
    seed: int,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """The extended checkpoint's ``metadata.json`` dict."""
    if metadata.get("architecture") != "structured":
        raise ResolutionExtensionError(
            f"{source_version}: architecture "
            f"{metadata.get('architecture')!r} -- only structured "
            f"checkpoints have a slot-query policy head to extend"
        )
    old_version = metadata.get(
        "action_encoding_version", ae.ACTION_ENCODING_VERSION_V1,
    )
    if old_version not in _EXTENDABLE_VERSIONS:
        raise ResolutionExtensionError(
            f"{source_version}: action_encoding_version {old_version!r} "
            f"is not extendable (expected one of {_EXTENDABLE_VERSIONS})"
        )
    if not metadata.get("card_vocab"):
        raise ResolutionExtensionError(
            f"{source_version}: no card_vocab in metadata (stamp the "
            f"checkpoint first: mtg_bot card-vocab stamp)"
        )
    if old_version == ae.ACTION_ENCODING_VERSION_V1:
        # v3 extends v2's canonical slot order; a positional_v1 head
        # memorised raw-zone positions, so its 517 rows would be read
        # under a different mapping.
        raise ResolutionExtensionError(
            f"{source_version}: positional_v1 heads can't be extended "
            f"(canonical_v3 extends canonical_v2's slot mapping)"
        )
    out = copy.deepcopy(metadata)
    new_vocab = ae.ACTION_VOCAB_SIZE_V3
    out["action_encoding_version"] = ae.ACTION_ENCODING_VERSION_V3
    training = dict(out.get("training") or {})
    training["structured_action_vocab_size"] = new_vocab
    nested = dict(training.get("structured_config") or {})
    nested["action_vocab_size"] = new_vocab
    training["structured_config"] = nested
    out["training"] = training
    extra = dict(out.get("extra") or {})
    extra[EXTENSION_KEY] = {
        "source_version": source_version,
        "source_weights_sha256": source_weights_sha256,
        "source_action_encoding_version": old_version,
        "old_action_vocab_size": ae.ACTION_VOCAB_SIZE,
        "new_action_vocab_size": new_vocab,
        "n_new_slots": new_vocab - ae.ACTION_VOCAB_SIZE,
        "resolution_block_size": resolution_slots.RESOLUTION_BLOCK_SIZE,
        "init": f"normal(0, {INIT_STD})",
        "seed": int(seed),
        "extended_at": _now_iso(),
        "extension_git_commit": git_commit,
    }
    out["extra"] = extra
    # Evaluations / promotions recorded for the source do not transfer.
    out["evaluation"] = {}
    note = (
        f"{source_version} + canonical_v3 resolution slots "
        f"({new_vocab - ae.ACTION_VOCAB_SIZE} fresh policy rows, "
        f"seed {seed}); no training"
    )
    prior = out.get("notes") or ""
    out["notes"] = f"{note}. Source notes: {prior}" if prior else note
    return out


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parent,
        )
        return r.stdout.strip() or None
    except Exception:  # pragma: no cover - git missing
        return None


def extend_checkpoint_dir(
    source_dir: Path, out_dir: Path, *,
    seed: int = DEFAULT_SEED,
    version: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write ``out_dir/weights.pt`` + ``out_dir/metadata.json`` for the
    extended checkpoint of ``source_dir`` (a registry version dir).
    ``version`` defaults to ``out_dir``'s name. Returns the written
    metadata dict. The source directory is only read."""
    from mtg_bot.training.card_vocab import effective_metadata_dict

    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if out_dir.resolve() == source_dir.resolve():
        raise ResolutionExtensionError("out_dir must differ from source_dir")
    weights = source_dir / "weights.pt"
    meta = effective_metadata_dict(source_dir)
    source_version = str(meta.get("version") or source_dir.name)
    new_meta = extend_metadata(
        meta, source_version=source_version,
        source_weights_sha256=_file_sha256(weights),
        seed=seed, git_commit=_git_commit(),
    )
    new_meta["version"] = version or out_dir.name
    new_meta["created_at"] = _now_iso()
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    extended = extend_state_dict(state_dict, seed=seed)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ResolutionExtensionError(
            f"{out_dir} exists and is not empty (pass overwrite=True)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(extended, out_dir / "weights.pt")
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2, sort_keys=True)
    return new_meta


def register_extended(
    registry_root: Path, source_version: str, *,
    seed: int = DEFAULT_SEED,
) -> str:
    """Extend ``registry_root/source_version`` and register the result
    as the registry's next sequential version. Returns its id."""
    from mtg_bot.training.registry import ModelMetadata, Registry

    registry = Registry(Path(registry_root))
    source_dir = Path(registry_root) / source_version
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "staged"
        new_meta = extend_checkpoint_dir(source_dir, staged, seed=seed)
        new_meta["version"] = registry.next_version_id()
        return registry.register(
            staged / "weights.pt", ModelMetadata.from_dict(new_meta),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mtg_bot extend-resolution-slots",
        description=(
            "Append the canonical_v3 resolution-decision policy slots to "
            "a trained canonical_v2 structured checkpoint (every trained "
            "parameter copied bit-exactly). See "
            "docs/RESOLUTION_DECISIONS.md."
        ),
    )
    p.add_argument("--registry-root", type=Path, required=True,
                   help="registry holding the source version")
    p.add_argument("--version", required=True,
                   help="source version label, e.g. v30")
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument(
        "--out-dir", type=Path,
        help="write the extended checkpoint here (weights.pt + "
             "metadata.json; the dir name becomes its label)",
    )
    dest.add_argument(
        "--register", action="store_true",
        help="register the extended checkpoint in --registry-root as "
             "the next sequential version",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"generator seed for the new rows "
                        f"(default {DEFAULT_SEED})")
    p.add_argument("--overwrite", action="store_true",
                   help="allow a non-empty --out-dir")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.register:
        new_version = register_extended(
            args.registry_root, args.version, seed=args.seed,
        )
        print(
            f"registered {new_version} = {args.version} + canonical_v3 "
            f"resolution slots in {args.registry_root}"
        )
        return 0
    meta = extend_checkpoint_dir(
        args.registry_root / args.version, args.out_dir,
        seed=args.seed, overwrite=args.overwrite,
    )
    print(
        f"wrote {args.out_dir} ({meta['version']}) = {args.version} + "
        f"canonical_v3 resolution slots"
    )
    return 0


__all__ = [
    "DEFAULT_SEED",
    "EXTENSION_KEY",
    "INIT_STD",
    "ResolutionExtensionError",
    "build_parser",
    "extend_checkpoint_dir",
    "extend_metadata",
    "extend_state_dict",
    "fresh_slot_rows",
    "main",
    "register_extended",
]
