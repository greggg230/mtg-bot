"""Capture annotated v3 replays from a small NN-driven self-play
batch. Diagnostic tool — every priority-moment snapshot in the
captured replays carries the search-tree annotation (chosen action,
candidates with visits / Q / prior / leaf NN value, search budget,
elapsed_ms). The training UI replay viewer renders these in the
decision-detail panel.

Usage::

    python -m mtg_bot.cli capture --bootstrap-version v13 \\
        --replays-dir replays/v13_diag

Then point the training UI at ``replays/v13_diag/`` and step through
a game to see, decision by decision, what the model considered.

Single-machine, in-process — no shards config, no queue server, no
training. Replays land at
``<replays-dir>/<replay_prefix>_<seed>_<idx>.json.gz``.
"""

from __future__ import annotations

from mtg_bot.search.leaf_batch import SEARCH_LEAF_BATCH_SIZE

import argparse
import time
from pathlib import Path

from mtg_bot.cli import _common
from mtg_bot.cli._common import depth_parser, runtime_parser
from mtg_bot.decks import decks_from_json
from mtg_bot.search.determinize import OPPONENT_HIDDEN_MODELS
from mtg_bot.training.config import NNPimcConfig
from mtg_bot.training.data_gen import (
    ReplayCaptureConfig,
    generate_training_data_parallel,
)
from mtg_bot.training.registry import Registry


def main(argv=None) -> None:  # pragma: no cover
    # Argparse + single dispatch into ``generate_training_data_parallel``.
    # The dispatch target is the module under test elsewhere; this
    # ``main`` is straight-line plumbing.
    p = argparse.ArgumentParser(
        description=__doc__,
        parents=[depth_parser(), runtime_parser()],
    )
    p.add_argument(
        "--bootstrap-version", default="v13",
        help="Registered version to play both seats (default v13).",
    )
    p.add_argument(
        "--registry-root", type=Path,
        default=Path("trained_weights"),
    )
    p.add_argument("--n-games", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--replays-dir", type=Path, required=True,
        help="Directory replays land in (created if missing).",
    )
    p.add_argument("--replay-prefix", default="game")
    p.add_argument(
        "--opponent-hidden-model", choices=OPPONENT_HIDDEN_MODELS,
        default="sampled",
        help=(
            "How each seat's search models the opponent's hidden "
            "cards: 'sampled' (default; imagine a hand from their deck) "
            "or 'board_only' (inert placeholders -- see "
            "mtg_bot.search.determinize)."
        ),
    )
    args = p.parse_args(argv)
    _common.resolve_per_game_timeout(args)

    registry = Registry(args.registry_root)
    weights_path = registry.weights_path(args.bootstrap_version)
    meta = registry.load_metadata(args.bootstrap_version)
    hidden_sizes = (
        tuple(meta.hidden_sizes)
        if getattr(meta, "hidden_sizes", None) else None
    )
    arch = getattr(meta, "architecture", "shared")
    policy_hidden = (
        tuple(meta.training.get("policy_hidden_sizes") or ())
        or None
    )

    args.replays_dir.mkdir(parents=True, exist_ok=True)
    deck_pool = decks_from_json(args.deck_pool) if args.deck_pool else None

    cfg = NNPimcConfig(
        weights_path=str(weights_path),
        samples=args.samples,
        inner_nodes=args.nodes,
        batch_size=SEARCH_LEAF_BATCH_SIZE,
        with_policy=True,
        puct=True,
        hidden_sizes=hidden_sizes,
        architecture=arch,
        policy_hidden_sizes=policy_hidden,
        opponent_hidden_model=args.opponent_hidden_model,
    )

    replay_cfg = ReplayCaptureConfig(
        save_dir=str(args.replays_dir),
        capture_every=1,
        nn_weights_path=str(weights_path),
        nn_version_label=args.bootstrap_version,
        nn_hidden_sizes=hidden_sizes,
        nn_with_policy=True,
        nn_architecture=arch,
        nn_policy_hidden_sizes=policy_hidden,
        # Sidecar metadata.json sits next to weights.pt in the registry;
        # threading it lets the per-game annotator construct via
        # NNEvaluator.from_metadata so non-default structured-trunk
        # dims load (iter_019 BUG #3).
        nn_metadata_path=str(weights_path.parent / "metadata.json"),
        filename_prefix=args.replay_prefix,
    )

    print(
        f"[capture] {args.bootstrap_version} self-play, "
        f"n_games={args.n_games}, samples={args.samples}, "
        f"nodes={args.nodes}, workers={args.workers}, "
        f"opponent_hidden_model={args.opponent_hidden_model}, "
        f"replays -> {args.replays_dir}"
    )

    t0 = time.perf_counter()
    features, _outcomes, _policy_targets, _action_masks = (
        generate_training_data_parallel(
            p0_factory=cfg.build,
            p1_factory=cfg.build,
            n_games=args.n_games,
            base_seed=args.seed,
            n_workers=args.workers,
            with_policy=True,
            per_game_timeout_sec=args.per_game_timeout_sec,
            deck_pool=deck_pool,
            replay_config=replay_cfg,
        )
    )
    elapsed = time.perf_counter() - t0

    n_replays = len(list(args.replays_dir.glob("*.json.gz")))
    print(
        f"[capture] done in {elapsed/60:.1f}min: "
        f"{features.shape[0]} samples across {args.n_games} games, "
        f"{n_replays} replay file(s) under {args.replays_dir}"
    )


if __name__ == "__main__":
    main()
