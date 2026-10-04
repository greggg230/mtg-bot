"""Tests for ``GET /v2/weights/<version>`` and
``GET /v2/weights/<version>/metadata``.
"""

from __future__ import annotations

import json
from pathlib import Path

from mtg_bot.cli import serve as serve_mod


def _make_app(tmp_path: Path):
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
    )


def test_weights_200_streams_bytes(tmp_path):
    app = _make_app(tmp_path)
    client = app.test_client()
    weights_dir = tmp_path / "trained_weights" / "v6"
    weights_dir.mkdir(parents=True)
    fake_bytes = b"\x00\x01\x02\x03fake-torch-weights"
    (weights_dir / "weights.pt").write_bytes(fake_bytes)
    res = client.get("/v2/weights/v6")
    assert res.status_code == 200
    assert res.mimetype == "application/octet-stream"
    assert res.data == fake_bytes


def test_weights_404_when_missing(tmp_path):
    app = _make_app(tmp_path)
    client = app.test_client()
    res = client.get("/v2/weights/v999")
    assert res.status_code == 404


def test_weights_metadata_200(tmp_path):
    app = _make_app(tmp_path)
    client = app.test_client()
    weights_dir = tmp_path / "trained_weights" / "v6"
    weights_dir.mkdir(parents=True)
    payload = {"action_encoding_version": "canonical_v2", "trunk": "1024-512"}
    (weights_dir / "metadata.json").write_text(json.dumps(payload))
    res = client.get("/v2/weights/v6/metadata")
    assert res.status_code == 200
    assert res.mimetype == "application/json"
    assert res.get_json() == payload


def test_weights_metadata_404(tmp_path):
    app = _make_app(tmp_path)
    client = app.test_client()
    res = client.get("/v2/weights/v999/metadata")
    assert res.status_code == 404


def test_weights_rejects_unsafe_version(tmp_path):
    app = _make_app(tmp_path)
    client = app.test_client()
    # Flask's URL routing strips trailing slashes; %2F triggers the
    # converter to refuse the request. Either way, no 200 + no
    # filesystem traversal.
    res = client.get("/v2/weights/..%2Fetc")
    assert res.status_code in (400, 404)


# ---------------------------------------------------------------------
# Card-vocab stamping sidecars (pre-fingerprint checkpoints)
# ---------------------------------------------------------------------


def _version_with_metadata(tmp_path: Path, payload: dict) -> Path:
    weights_dir = tmp_path / "trained_weights" / "v6"
    weights_dir.mkdir(parents=True)
    (weights_dir / "weights.pt").write_bytes(b"structured-weights")
    (weights_dir / "metadata.json").write_text(json.dumps(payload))
    return weights_dir


def test_weights_metadata_without_sidecar_is_served_byte_identical(tmp_path):
    app = _make_app(tmp_path)
    raw = '{"architecture":"structured",  "version": "v6"}'
    weights_dir = _version_with_metadata(tmp_path, {})
    (weights_dir / "metadata.json").write_text(raw)
    res = app.test_client().get("/v2/weights/v6/metadata")
    assert res.status_code == 200
    assert res.data == raw.encode()


def test_weights_metadata_merges_a_valid_sidecar(tmp_path):
    from mtg_bot.training.card_vocab import CardVocab, write_sidecar

    app = _make_app(tmp_path)
    payload = {"architecture": "structured", "version": "v6"}
    weights_dir = _version_with_metadata(tmp_path, payload)
    vocab = CardVocab.from_names(["Alpha", "Beta"])
    write_sidecar(vocab, weights_dir / "weights.pt", provenance={"git_commit": "abc"})
    before = (weights_dir / "metadata.json").read_bytes()
    res = app.test_client().get("/v2/weights/v6/metadata")
    assert res.status_code == 200
    body = res.get_json()
    assert body["architecture"] == "structured"
    assert CardVocab.from_dict(body["card_vocab"]) == vocab
    assert body["card_vocab"]["provenance"] == {"git_commit": "abc"}
    # The registry's own file is untouched.
    assert (weights_dir / "metadata.json").read_bytes() == before


def test_weights_metadata_with_invalid_sidecar_fails_loudly(tmp_path):
    from mtg_bot.training.card_vocab import CardVocab, write_sidecar

    app = _make_app(tmp_path)
    weights_dir = _version_with_metadata(tmp_path, {"architecture": "structured"})
    write_sidecar(CardVocab.from_names(["A"]), weights_dir / "weights.pt")
    # Weights replaced after stamping: the stamp no longer applies.
    (weights_dir / "weights.pt").write_bytes(b"different-weights-now")
    res = app.test_client().get("/v2/weights/v6/metadata")
    assert res.status_code == 500
    assert "card-vocab sidecar" in res.get_json()["error"]
