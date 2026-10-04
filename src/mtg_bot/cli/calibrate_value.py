"""``mtg_bot calibrate-value`` — re-fit a registered version's value
temperature on a held-out split.

Training already fits a calibration and writes it into
``metadata.value_calibration`` (see
:mod:`mtg_bot.training.value_calibration`). This subcommand exists for
the cases training can't cover:

- a version registered BEFORE the calibration landed (every ``v1``..
  ``v7`` in the s2 lineage carries no ``value_calibration`` key and so
  runs at the identity temperature);
- re-fitting against a different / newer held-out pool, e.g. after the
  self-play distribution shifts under the version;
- auditing: ``--dry-run`` prints the raw-vs-calibrated numbers without
  touching ``metadata.json``.

It never touches ``weights.pt``. The temperature is metadata: the
loader (``NNEvaluator.from_metadata``) pushes it into
``StructuredModelConfig.value_temperature`` and
``StructuredModel.forward`` applies it, so re-calibrating is a
JSON write and every consumer picks it up on next load.

The held-out split is reproduced the way the trainer built it —
:func:`mtg_bot.training.trainer.grouped_train_val_split` over the
dataset's ``group_ids`` with the version's recorded ``val_split`` and
``rng_seed`` — so "the held-out split" means the same rows it meant
during training as long as the data pool is the same. Pass
``--val-split`` / ``--seed`` / ``--data-dir`` to score against a
different split on purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_REGISTRY_ROOT = Path("trained_weights")
DEFAULT_BATCH_SIZE = 256


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtg_bot calibrate-value",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", required=True,
        help="Registered version id to calibrate, e.g. v8.",
    )
    parser.add_argument(
        "--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT,
        help=(
            "Registry directory holding <version>/weights.pt + "
            "metadata.json. Default: trained_weights."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help=(
            "Directory of structured .npz self-play files to score "
            "against. Default: the data_dir recorded in the "
            "version's metadata."
        ),
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help=(
            "Cap to the N most recent .npz files (by mtime), matching "
            "`mtg_bot train --window-recent-files`. Default: the "
            "window_recent_files recorded in the version's metadata, "
            "else every file."
        ),
    )
    parser.add_argument(
        "--val-split", type=float, default=None,
        help=(
            "Held-out fraction (by GAME group). Default: the "
            "val_split recorded in the version's metadata, else 0.1."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Split RNG seed. Default: the rng_seed recorded in the "
            "version's metadata, else 0. Must match the training "
            "seed for the split to be the SAME held-out rows."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Scoring batch size (inference only).",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device; default auto (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes for scoring. Default 0.",
    )
    parser.add_argument(
        "--fit-bias", action="store_true",
        help=(
            "Also fit an additive pre-tanh bias b, i.e. "
            "tanh(z / T + b). OFF by default and deliberately so: a "
            "free bias makes the T -> infinity limit 'predict the "
            "held-out label mean', a CONSTANT predictor that can "
            "score under the gate's constant-0 ceiling on any "
            "imbalanced split. Temperature alone degenerates to "
            "exactly the constant-0 predictor, so it can never "
            "manufacture a pass. Use this for diagnosis, not for a "
            "number you intend to gate on."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the fitted calibration and the raw-vs-calibrated "
            "held-out numbers without writing metadata.json."
        ),
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Also write the fitted calibration + metrics to this path.",
    )
    parser.add_argument(
        "--shard-card-vocab", type=str, default=None,
        help=(
            "Card-vocabulary file asserting the vocabulary of "
            "pre-fingerprint (unstamped) shards, as for `mtg_bot train "
            "--shard-card-vocab`. Default: $MTG_BOT_SHARD_CARD_VOCAB."
        ),
    )
    return parser


def _resolve_from_metadata(metadata, key: str, default):
    training = getattr(metadata, "training", {}) or {}
    val = training.get(key)
    return default if val is None else val


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Subset

    from mtg_bot.search.nn_evaluator import (
        ARCHITECTURE_STRUCTURED,
        NNEvaluator,
    )
    from mtg_bot.training.registry import Registry
    from mtg_bot.training.structured_dataset import (
        load_structured_dataset,
        structured_collate,
    )
    from mtg_bot.training.trainer import (
        auto_device,
        grouped_train_val_split,
    )
    from mtg_bot.training.value_gate import (
        compute_holdout_value_metrics,
    )

    registry = Registry(args.registry_root)
    if not registry.has(args.version):
        print(
            f"version {args.version!r} not found under "
            f"{args.registry_root}",
            file=sys.stderr,
        )
        return 2
    metadata = registry.load_metadata(args.version)
    if getattr(metadata, "architecture", None) != ARCHITECTURE_STRUCTURED:
        print(
            f"{args.version} is architecture "
            f"{getattr(metadata, 'architecture', None)!r}; "
            f"calibrate-value only supports "
            f"{ARCHITECTURE_STRUCTURED!r} (the flat MLP heads expose "
            f"no pre-tanh logit).",
            file=sys.stderr,
        )
        return 2

    data_dir = args.data_dir or Path(
        _resolve_from_metadata(metadata, "data_dir", "iterate_data")
    )
    if not Path(data_dir).is_dir():
        print(f"--data-dir {data_dir} is not a directory", file=sys.stderr)
        return 2
    max_files = args.max_files
    if max_files is None:
        max_files = _resolve_from_metadata(
            metadata, "window_recent_files", None,
        )
    val_split = args.val_split
    if val_split is None:
        val_split = float(
            _resolve_from_metadata(metadata, "val_split", 0.1)
        )
    seed = args.seed
    if seed is None:
        seed = int(_resolve_from_metadata(metadata, "rng_seed", 0))
    if not (0.0 < val_split < 1.0):
        print(
            f"--val-split must be in (0, 1) to have held-out rows; "
            f"got {val_split}",
            file=sys.stderr,
        )
        return 2

    device = args.device or auto_device()
    print(
        f"# calibrate-value {args.version}: data={data_dir} "
        f"max_files={max_files} val_split={val_split} seed={seed} "
        f"device={device}"
    )

    # Load the model through the registry loader so every architecture
    # knob (trunk dims, scalar schema, value_head_input) comes off
    # metadata. A calibration already stored is applied to ``value``
    # but NOT to ``value_logit`` — the fit below is therefore always
    # absolute (it re-derives T from the raw logit), so re-running
    # this command is idempotent rather than compounding.
    evaluator = NNEvaluator.from_metadata(
        registry.weights_path(args.version), metadata, device=device,
    )
    model = evaluator.model
    model.eval()

    # The held-out shards' printing ids must live in the loaded model's
    # input space: the current card vocabulary (the loader remapped the
    # checkpoint to it) bucketed at the model's n_printings. Scoring a
    # model on ids from another id space would fit a temperature to
    # scrambled card identities.
    from mtg_bot.training.card_vocab import (
        CardVocabError,
        current_card_vocab,
        legacy_shard_vocab_from_arg,
        resolve_shards_card_vocab,
    )
    from mtg_bot.training.structured_dataset import _select_paths

    try:
        shard_vocab = resolve_shards_card_vocab(
            _select_paths(Path(data_dir), max_files),
            legacy_vocab=legacy_shard_vocab_from_arg(args.shard_card_vocab),
        )
    except CardVocabError as exc:
        print(f"card vocabulary: {exc}", file=sys.stderr)
        return 2
    model_vocab = current_card_vocab()
    model_cap = int(model.config.n_printings)
    # ``None`` = no structured rows at all; the loader / "no samples"
    # checks below report that case.
    if shard_vocab is not None and (
        shard_vocab.vocab.sha256 != model_vocab.sha256
        or shard_vocab.n_printings_cap != model_cap
    ):
        print(
            f"card vocabulary mismatch: the held-out shards use vocabulary "
            f"{shard_vocab.vocab.short_sha} ({shard_vocab.vocab.size} "
            f"names) at n_printings cap {shard_vocab.n_printings_cap}, but "
            f"{args.version} is loaded in vocabulary "
            f"{model_vocab.short_sha} ({model_vocab.size} names) at "
            f"n_printings {model_cap}. Calibrate on shards generated "
            f"under the current card vocabulary.",
            file=sys.stderr,
        )
        return 2

    dataset = load_structured_dataset(Path(data_dir), max_files=max_files)
    try:
        n = len(dataset)
        if n == 0:
            print(f"no samples in {data_dir}", file=sys.stderr)
            return 2
        group_ids = getattr(dataset, "group_ids", None)
        if group_ids is None:
            group_ids = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        _, val_idx = grouped_train_val_split(
            np.asarray(group_ids), val_split, rng,
        )
        if val_idx.shape[0] == 0:
            print(
                "held-out split is empty; nothing to calibrate",
                file=sys.stderr,
            )
            return 2
        loader = DataLoader(
            Subset(dataset, sorted(val_idx.tolist())),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=structured_collate,
            num_workers=args.num_workers,
        )
        preds, logits, targets = _score(model, loader, device, torch)
    finally:
        dataset.close()

    metrics = compute_holdout_value_metrics(
        preds, targets, None, logits, fit_bias=args.fit_bias,
    )
    calibration = metrics.calibration or {}
    print(f"# held-out rows ............ {metrics.n_rows}")
    # "as loaded" = whatever calibration metadata already carried.
    # The fit below always works from the RAW pre-tanh logit, so
    # ``raw_mse`` inside the calibration dict is the uncalibrated
    # number regardless of what was stored.
    print(f"# as-loaded MSE ............ {metrics.value_mse:.4f}")
    print(f"# uncalibrated MSE ......... {calibration.get('raw_mse')}")
    print(
        f"# calibrated MSE ........... "
        f"{metrics.calibrated_value_mse:.4f}"
    )
    print(f"# cross-fitted MSE ......... {calibration.get('cv_mse')}")
    print(f"# constant-0 MSE ........... {metrics.constant_zero_mse:.4f}")
    print(f"# corr(pred, outcome) ...... {metrics.value_corr}")
    print(
        f"# mean |pred| raw / calib .. {metrics.pred_abs_mean:.4f} / "
        f"{metrics.calibrated_pred_abs_mean:.4f}"
    )
    print(
        f"# temperature / bias ....... "
        f"{calibration.get('temperature')} / "
        f"{calibration.get('bias')}"
    )

    payload = {
        "version": args.version,
        "value_calibration": calibration,
        "holdout_value_metrics": metrics.as_dict(),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"# wrote {args.json_out}")
    if args.dry_run:
        print("# --dry-run: metadata.json NOT modified")
        return 0
    registry.set_value_calibration(
        args.version, calibration, holdout_metrics=metrics.as_dict(),
    )
    print(
        f"# wrote value_calibration into "
        f"{args.registry_root / args.version / 'metadata.json'}"
    )
    return 0


def _score(model, loader, device, torch):
    """Run the model over the held-out loader, returning
    ``(preds, logits, targets)`` as float64 numpy arrays.

    ``value_logit`` is required here (unlike in the trainer, which
    degrades gracefully): this command exists to fit a temperature on
    the pre-tanh logit, and silently falling back to the weaker
    post-tanh shrink would misrepresent what was written to metadata.
    """
    device_obj = torch.device(device)
    preds: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            on_device = {
                k: v.to(device_obj) for k, v in batch.items()
            }
            out = model(on_device)
            if "value_logit" not in out:
                raise RuntimeError(
                    "model forward returned no 'value_logit'; "
                    "calibrate-value needs the pre-tanh logit"
                )
            preds.append(out["value"].float().cpu().numpy())
            logits.append(out["value_logit"].float().cpu().numpy())
            targets.append(
                on_device["outcome"].float().cpu().numpy()
            )
    return (
        np.concatenate(preds),
        np.concatenate(logits),
        np.concatenate(targets),
    )


if __name__ == "__main__":
    sys.exit(main())
