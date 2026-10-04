"""Register a fresh random-init structured net (scalar schema v2) as the
start of a new v-chain in its own registry root.

The old ``trained_weights/`` lineage (v1..v14, current_best v13s) was
trained on scalar schema 1 with the pre-2026-09 loop defects (no life /
board scalars, row-level val split, no exploration, empty action
history at inference, root-perspective opponent priors). It is kept
as-is for the ladder; the corrected loop starts from scratch here.

Usage::

    PYTHONPATH=src .venv-py313/Scripts/python.exe scripts/bootstrap_structured_s2.py \
        --registry-root trained_weights_s2 [--emb-dim 128 --n-heads 4 \
        --n-layers 2 --hidden-dim 256]

Then::

    mtg_bot iterate --bootstrap-version v1 --registry-root trained_weights_s2 \
        --data-dir iterate_data_s2 --progress-dir training_progress_s2 \
        --run-label-prefix s2_ \
        --architecture structured --scalar-schema-version 2 ...

``--run-label-prefix s2_`` is REQUIRED when this lineage writes to the
shared coordinator runs.db. Its ``iterate_state.json`` restarts at
``iterations_completed = 0``, so without the prefix its first cycle
would claim ``iter_001_cycle`` -- a row the old lineage already owns,
and ``/run/start_or_restart`` answers HTTP 409. With the prefix the
ids become ``s2_iter_001_cycle`` / ``s2_iter_001_self_play_v1`` / ...
Through the unified launcher the equivalent is ``[iter].label_prefix
= "s2_"`` (or ``--iter.label-prefix s2_``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE  # noqa: E402
from mtg_bot.training.card_vocab import current_card_vocab  # noqa: E402
from mtg_bot.training.registry import ModelMetadata, Registry  # noqa: E402
from mtg_bot.search.action_history import DEFAULT_ACTION_HISTORY_N  # noqa: E402
from mtg_bot.training.structured_model import (  # noqa: E402
    DEFAULT_N_PRINTINGS, StructuredModel, StructuredModelConfig,
)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=_HERE.parent,
        ).stdout.strip()
    except Exception:  # pragma: no cover - best effort
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry-root", type=Path, default=Path("trained_weights_s2"))
    p.add_argument("--version", default=None, help="default: next id (v1 in an empty root)")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--action-history-n", type=int, default=DEFAULT_ACTION_HISTORY_N)
    p.add_argument("--n-printings", type=int, default=DEFAULT_N_PRINTINGS)
    p.add_argument("--scalar-schema-version", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = StructuredModelConfig(
        emb_dim=args.emb_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        hidden_dim=args.hidden_dim, action_history_n=args.action_history_n,
        n_printings=args.n_printings,
        scalar_schema_version=args.scalar_schema_version,
    )
    model = StructuredModel(cfg)
    n_params = sum(t.numel() for t in model.parameters())

    structured_config = {
        "emb_dim": cfg.emb_dim, "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers, "hidden_dim": cfg.hidden_dim,
        "action_history_n": cfg.action_history_n,
        "n_printings": cfg.n_printings,
        "action_vocab_size": ACTION_VOCAB_SIZE,
        "dropout": getattr(cfg, "dropout", 0.0),
        "scalar_schema_version": cfg.scalar_schema_version,
    }
    meta = ModelMetadata(
        version="",
        created_at="",
        feature_dim=0,
        hidden_sizes=[],
        with_policy=True,
        architecture="structured",
        action_encoding_version="canonical_v2",
        scalar_schema_version=cfg.scalar_schema_version,
        # A random-init net is born in the CURRENT card vocabulary
        # (its printing rows are indexed by today's tokenizer ids).
        card_vocab=current_card_vocab().to_dict(
            provenance={"source": "random_init", "git_commit": _git_commit()},
        ),
        training={
            "notes": "cold-start random init; no training data",
            "structured_config": dict(structured_config),
            "structured_emb_dim": cfg.emb_dim,
            "structured_n_heads": cfg.n_heads,
            "structured_n_layers": cfg.n_layers,
            "structured_hidden_dim": cfg.hidden_dim,
            "structured_action_history_n": cfg.action_history_n,
            "structured_n_printings": cfg.n_printings,
            "structured_scalar_schema_version": cfg.scalar_schema_version,
        },
        git_commit=_git_commit(),
        notes=(
            "Fresh-init structured bootstrap for the scalar-schema-v2 "
            "lineage (corrected loop, 2026-09). PyTorch default init; "
            f"{n_params} params."
        ),
        extra={
            "structured_config": dict(structured_config),
            "lineage": "structured_s2_chain",
            "lineage_start": True,
        },
    )
    registry = Registry(args.registry_root)
    with tempfile.TemporaryDirectory() as td:
        wp = Path(td) / "weights.pt"
        torch.save(model.state_dict(), wp)
        version = registry.register(wp, meta, version=args.version)
    print(
        f"registered {version} in {args.registry_root} "
        f"({n_params} params, scalar schema {cfg.scalar_schema_version}, "
        f"emb {cfg.emb_dim} / heads {cfg.n_heads} / layers {cfg.n_layers} "
        f"/ hidden {cfg.hidden_dim})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
