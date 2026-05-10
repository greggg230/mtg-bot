"""Microbenchmark: local NN inference per worker vs centralized server.

Spins up N worker processes that each fire a fixed number of value
requests on randomly-generated FEATURE_DIM-sized vectors; reports
total wall, requests/s, and per-request latency percentiles.

Two modes:

- ``--mode local``: each worker holds its own ``NNEvaluator`` and
  runs inference in-process. This is today's behavior in
  ``data_gen.py`` / ``run_match_parallel``.
- ``--mode server``: one shared :class:`NNEvalServer` runs in a
  child process; workers ship feature vectors to it via
  :class:`NNEvalClient`.

The server-mode bench captures the part of the win that lives in
batching. It does NOT exercise the engine — the workers ship random
vectors, not real game features — so it's a clean view of the IPC
+ batching cost vs the in-process forward cost. End-to-end (real
game) wins are smaller because feature extraction + MCTS overhead
is part of the budget; see ``scripts/evaluate_nn.py --use-nn-server``.

Usage::

    python scripts/bench_nn_eval_server.py --mode local  --workers 24 --requests-per 200
    python scripts/bench_nn_eval_server.py --mode server --workers 24 --requests-per 200 \
        --server-batch-size 32 --server-batch-wait-ms 5

Pass ``--with-policy`` to bench the dual-head path (single +
policy mask). Numbers logged on the launching process; per-worker
latencies are aggregated via a multiprocessing.Queue.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

def _local_worker(args: tuple) -> tuple[float, list[float]]:
    """Worker for ``--mode local``. Loads its own NNEvaluator and
    fires N requests."""
    weights_path, requests_per, with_policy, seed, hidden_sizes = args
    import torch
    torch.set_num_threads(1)
    from mtg_bot.features import FEATURE_DIM
    from mtg_bot.search.nn_evaluator import NNEvaluator
    rng = np.random.default_rng(seed)

    ev_kwargs: dict = {
        "weights_path": Path(weights_path) if weights_path else None,
        "with_policy": with_policy,
    }
    if hidden_sizes is not None:
        ev_kwargs["hidden_sizes"] = tuple(hidden_sizes)
    ev = NNEvaluator(**ev_kwargs)
    latencies: list[float] = []
    started = time.perf_counter()
    feats = rng.standard_normal(
        (requests_per, FEATURE_DIM),
    ).astype(np.float32)
    if with_policy:
        from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
        masks = rng.integers(
            0, 2, size=(requests_per, ACTION_VOCAB_SIZE),
        ).astype(bool)
        # Ensure each row has at least one legal action.
        masks[:, 0] = True
        for i in range(requests_per):
            t0 = time.perf_counter()
            x_t = __import__("torch").from_numpy(feats[i]).unsqueeze(0)
            mask_t = __import__("torch").from_numpy(masks[i])
            with __import__("torch").no_grad():
                v_t, l_t = ev.model(x_t)
                _ = v_t.item()
                # Apply mask + softmax to mirror the server path.
                ll = l_t.squeeze(0).masked_fill(
                    ~mask_t, float("-inf"),
                )
                _ = __import__(
                    "torch.nn.functional",
                    fromlist=["softmax"],
                ).softmax(ll, dim=-1).numpy()
            latencies.append(time.perf_counter() - t0)
    else:
        for i in range(requests_per):
            t0 = time.perf_counter()
            x_t = __import__("torch").from_numpy(feats[i]).unsqueeze(0)
            with __import__("torch").no_grad():
                _ = ev.model(x_t).item()
            latencies.append(time.perf_counter() - t0)
    return time.perf_counter() - started, latencies


def _server_worker(args: tuple) -> tuple[float, list[float]]:
    """Worker for ``--mode server``. Connects to the server and ships
    requests."""
    handle_dict, requests_per, with_policy, seed = args
    import torch
    torch.set_num_threads(1)
    from mtg_bot.features import FEATURE_DIM
    from mtg_bot.search.nn_eval_server import (
        MSG_KIND_VALUE,
        MSG_KIND_VALUE_POLICY,
        NNEvalClient,
        NNServerHandle,
    )
    rng = np.random.default_rng(seed)
    handle = NNServerHandle(
        address=tuple(handle_dict["address"])
        if isinstance(handle_dict["address"], (list, tuple))
        else handle_dict["address"],
        authkey=handle_dict["authkey"],
        with_policy=handle_dict["with_policy"],
    )
    client = NNEvalClient(handle)
    feats = rng.standard_normal(
        (requests_per, FEATURE_DIM),
    ).astype(np.float32)
    latencies: list[float] = []
    started = time.perf_counter()
    if with_policy:
        from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
        masks = rng.integers(
            0, 2, size=(requests_per, ACTION_VOCAB_SIZE),
        ).astype(bool)
        masks[:, 0] = True
        for i in range(requests_per):
            t0 = time.perf_counter()
            _ = client._call(
                MSG_KIND_VALUE_POLICY, (feats[i], masks[i]),
            )
            latencies.append(time.perf_counter() - t0)
    else:
        for i in range(requests_per):
            t0 = time.perf_counter()
            _ = client._call(MSG_KIND_VALUE, feats[i])
            latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - started
    client.close()
    return elapsed, latencies


def main(argv=None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["local", "server"], required=True,
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--requests-per", type=int, default=200)
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path to .pt weights. Random init if omitted.",
    )
    parser.add_argument("--with-policy", action="store_true")
    parser.add_argument("--server-batch-size", type=int, default=32)
    parser.add_argument(
        "--server-batch-wait-ms", type=float, default=5.0,
    )
    parser.add_argument(
        "--server-device", default=None,
        help="cpu or cuda. Default: auto.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="If given, append a JSON line summary to this file.",
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=None,
        help=(
            "Trunk shape for the MLP. Defaults to NNEvaluator's "
            "(128, 128, 64). Pass e.g. '1024 1024 512' to bench "
            "the v13/v15/v16 production trunk."
        ),
    )
    args = parser.parse_args(argv)

    requests_total = args.workers * args.requests_per
    print(
        f"# bench_nn_eval_server: mode={args.mode} "
        f"workers={args.workers} requests_per={args.requests_per} "
        f"requests_total={requests_total} "
        f"with_policy={args.with_policy}"
    )

    proxy = None
    if args.mode == "server":
        import torch
        from mtg_bot.search.nn_eval_server import (
            launch_server_subprocess,
        )
        device = args.server_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        proxy = launch_server_subprocess(
            weights_path=args.weights,
            device=device,
            with_policy=args.with_policy,
            batch_size=args.server_batch_size,
            batch_wait_ms=args.server_batch_wait_ms,
            hidden_sizes=(
                tuple(args.hidden_sizes)
                if args.hidden_sizes is not None else None
            ),
        )
        h = proxy.handle()
        print(
            f"# server: device={device} "
            f"batch_size={args.server_batch_size} "
            f"batch_wait_ms={args.server_batch_wait_ms} "
            f"address={h.address}"
        )
        handle_dict = {
            "address": h.address,
            "authkey": h.authkey,
            "with_policy": h.with_policy,
        }
        per_args = [
            (handle_dict, args.requests_per, args.with_policy, 1000 + i)
            for i in range(args.workers)
        ]
        worker_fn = _server_worker
    else:
        hidden_sizes_t = (
            tuple(args.hidden_sizes)
            if args.hidden_sizes is not None else None
        )
        per_args = [
            (
                str(args.weights) if args.weights else None,
                args.requests_per,
                args.with_policy,
                1000 + i,
                hidden_sizes_t,
            )
            for i in range(args.workers)
        ]
        worker_fn = _local_worker

    started = time.perf_counter()
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(worker_fn, per_args)
    finally:
        if proxy is not None:
            proxy.stop(timeout=10.0)
    wall = time.perf_counter() - started

    per_worker_walls = [r[0] for r in results]
    all_latencies: list[float] = []
    for _, lats in results:
        all_latencies.extend(lats)
    all_latencies.sort()

    def pct(p: float) -> float:
        if not all_latencies:
            return 0.0
        idx = max(0, min(len(all_latencies) - 1,
                         int(p / 100.0 * len(all_latencies))))
        return all_latencies[idx]

    rps = requests_total / wall if wall > 0 else 0.0
    print(
        f"# total_wall={wall:.3f}s rps={rps:.1f}"
    )
    print(
        f"# per-worker wall: "
        f"min={min(per_worker_walls):.3f} "
        f"median={statistics.median(per_worker_walls):.3f} "
        f"max={max(per_worker_walls):.3f}"
    )
    print(
        f"# request latency (ms): "
        f"p50={pct(50)*1e3:.3f} "
        f"p95={pct(95)*1e3:.3f} "
        f"p99={pct(99)*1e3:.3f} "
        f"mean={(sum(all_latencies)/len(all_latencies))*1e3:.3f}"
    )

    if args.out is not None:
        summary = {
            "mode": args.mode,
            "workers": args.workers,
            "requests_per": args.requests_per,
            "with_policy": args.with_policy,
            "hidden_sizes": (
                list(args.hidden_sizes)
                if args.hidden_sizes is not None else None
            ),
            "server_batch_size": args.server_batch_size,
            "server_batch_wait_ms": args.server_batch_wait_ms,
            "total_wall_s": wall,
            "rps": rps,
            "lat_p50_ms": pct(50) * 1e3,
            "lat_p95_ms": pct(95) * 1e3,
            "lat_p99_ms": pct(99) * 1e3,
            "lat_mean_ms": sum(all_latencies) / len(all_latencies)
            * 1e3,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"# wrote summary to {args.out}")


if __name__ == "__main__":
    main()
