"""End-to-end v2 smoke for a HEURISTIC (warm-start corpus) stage.

Drives the real production self-play runner through the real claim
loop against a real coordinator app (Flask test_client -- no sockets,
so this never collides with the live :5099 / :5198 daemons), and
asserts the shard the coordinator rehydrates into its
``iterate_data`` dir carries the NN path's exact npz schema.

This is the in-process equivalent of the distributed smoke: stage
create -> claim -> play -> artifact upload -> report_result -> npz on
the coordinator's disk, with nothing stubbed but the transport.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from mtg_bot import worker_daemon
from mtg_bot import worker_daemon_claim_loop as cl
from mtg_bot import worker_daemon_production_runner as prr
from mtg_bot.cli import serve as serve_mod
from mtg_bot.cli.runs.coordinator_artifact_fetcher import (
    LocalArtifactFetcher,
    MultiSchemeFetcher,
)
from mtg_bot.distributed_rest_orchestrator import make_self_play_stage_spec
from mtg_bot.worker_daemon_storage import LocalFileUploader

STAGE_ID = "smoke_iter_001_self_play_heuristic"


def _test_client_adapter(coordinator_app):
    client = coordinator_app.test_client()

    def _strip_base(url: str) -> str:
        for prefix in ("http://", "https://"):
            if url.startswith(prefix):
                return url[url.index("/", len(prefix)):]
        return url

    def http_get(url, timeout=30.0):
        resp = client.get(_strip_base(url))
        return resp.status_code, resp.data, dict(resp.headers)

    def http_post_json(url, payload, timeout=30.0):
        resp = client.post(_strip_base(url), json=payload)
        return resp.status_code, resp.data, dict(resp.headers)

    return http_get, http_post_json


def _build_coordinator_app(tmp_path: Path):
    return serve_mod.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 KEY",
        master_tailnet_name="myhost",
        port=5099,
        queue_db_path=tmp_path / "queue.db",
        weights_root=tmp_path / "trained_weights",
        iterate_data_dir=tmp_path / "iterate_data",
        replays_root=tmp_path / "replays",
        shards_config_path=tmp_path / "shards.json",
        enable_heartbeat_poller=False,
        artifact_fetcher=MultiSchemeFetcher(local=LocalArtifactFetcher()),
        report_executor=None,
    )


def _save_tiny_deck_pool(tmp_path: Path) -> Path:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from _test_decks import simple_test_deck_pool
    finally:
        sys.path.pop(0)

    decks = simple_test_deck_pool()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "pool.json.gz"
    with gzip.open(path, "wb") as f:
        f.write(json.dumps([
            {"draft_id": d.draft_id, "build_index": d.build_index,
             "cards": d.cards}
            for d in decks
        ]).encode("utf-8"))
    return path


@pytest.mark.slow
def test_heuristic_stage_drains_and_lands_nn_compatible_npzs(tmp_path):
    coord_app = _build_coordinator_app(tmp_path)
    coord_client = coord_app.test_client()

    spec = make_self_play_stage_spec(
        stage_id=STAGE_ID,
        n_games=2,
        base_seed=1_000_000,
        deck_pool_size=2,
        weights_version="unused",
        nodes=4,
        samples=1,
        per_game_timeout_sec=300.0,
        claim_ttl_seconds=600,
        evaluator="heuristic",
        action_encoding_version="canonical_v2",
        structured_tokens=True,
        structured_scalar_schema_version=2,
        move_temperature=1.0,
        temperature_cutoff=30,
    )
    res = coord_client.post("/v2/stage/create", json=spec.to_dict())
    assert res.status_code == 200, res.get_json()

    http_get, http_post_json = _test_client_adapter(coord_app)

    def _exploding_http_get(url, timeout=30.0):
        # The production runner's weights-fetch hook. A heuristic stage
        # must never reach it.
        raise AssertionError(f"unexpected weights fetch: {url}")

    runner = prr.build_self_play_runner(
        coordinator_url="http://coord-test:5099",
        stage_id=STAGE_ID,
        n_workers=1,
        weights_cache_root=tmp_path / "weights_cache",
        worker_id="smoke-worker",
        log_event=None,
        deck_pool_path=str(_save_tiny_deck_pool(tmp_path / "pool")),
        http_get=_exploding_http_get,
    )

    state = worker_daemon.WorkerState()
    cfg = cl.ClaimLoopConfig(
        coordinator_url="http://coord-test:5099",
        stage_id=STAGE_ID,
        kind="self_play",
        n_workers=1,
        worker_id="smoke-worker",
        weights_root=tmp_path / "weights_cache",
        shard_label="smoke-worker",
        retry_backoff_s=(0.0, 0.0, 0.0),
        http_get=http_get,
        http_post_json=http_post_json,
        artifact_uploader=LocalFileUploader(tmp_path / "artifacts"),
    )
    cbs = cl.StateCallbacks(
        is_stop_requested=state.is_stop_requested,
        on_claim=state.on_claim,
        on_report=state.on_report,
        on_drop=state.on_drop,
        mark_running=state.mark_running,
        mark_stopping=state.mark_stopping,
        mark_idle=state.mark_idle,
        mark_error=state.mark_error,
    )
    loop = cl.ClaimLoopRunner(cfg, cbs, runner)
    state.try_start("self_play", STAGE_ID, n_workers=1)
    t = threading.Thread(target=loop.run, daemon=True)
    t.start()

    deadline = time.time() + 300.0
    while time.time() < deadline:
        counters = coord_client.get(f"/v2/stage/{STAGE_ID}").get_json()
        if counters.get("status") == "complete":
            break
        time.sleep(0.25)
    t.join(timeout=60.0)
    assert not t.is_alive(), "claim loop did not exit"

    counters = coord_client.get(f"/v2/stage/{STAGE_ID}").get_json()
    assert counters["n_games_done"] == 2, counters
    assert counters.get("crashed", 0) == 0, counters
    assert counters.get("upload_failed", 0) == 0, counters

    npzs = sorted((tmp_path / "iterate_data").glob(f"{STAGE_ID}_*.npz"))
    assert len(npzs) == 2, [p.name for p in npzs]
    expected_cols = {
        "features", "outcomes", "policy_targets", "action_masks",
        "structured_token_type", "structured_token_emb_ids",
        "structured_token_scalars", "structured_attention_mask",
        "root_values", "game_ids",
        # The card vocabulary + cap the printing ids were tokenised in.
        "structured_card_vocab",
    }
    from mtg_bot.training.card_vocab import (
        current_card_vocab,
        decode_shard_card_vocab,
    )
    from mtg_bot.training.data_gen import shard_n_printings_cap

    total_rows = 0
    for p in npzs:
        with np.load(p) as d:
            assert set(d.files) == expected_cols, (p.name, set(d.files))
            assert d["structured_token_scalars"].shape[-1] == 29
            assert decode_shard_card_vocab(d["structured_card_vocab"]) == (
                current_card_vocab(), shard_n_printings_cap(),
            )
            total_rows += int(d["features"].shape[0])
    assert total_rows > 0
