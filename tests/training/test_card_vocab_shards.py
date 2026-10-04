"""Card-vocabulary fingerprint on the training-data side.

Structured shards store PRE-tokenised printing ids, so they are in a
card-vocabulary id space too. These tests pin that:

* every structured shard written by ``save_dataset`` or by a worker's
  per-game encoder records its vocabulary + bucketing cap;
* the trainer refuses mixed id spaces and unstamped shards (unless the
  operator asserts their vocabulary), and stamps the trained model with
  the SHARDS' vocabulary;
* ``calibrate-value`` refuses held-out shards whose id space differs
  from the loaded model's.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mtg_bot.cli import calibrate_value as cli_calibrate
from mtg_bot.cli import train as cli_train
from mtg_bot.features import FEATURE_DIM
from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
from mtg_bot.training.card_vocab import (
    LEGACY_SHARD_N_PRINTINGS_CAP,
    SHARD_CARD_VOCAB_ENV,
    SHARD_KEY,
    CardVocab,
    ShardCardVocabError,
    current_card_vocab,
    decode_shard_card_vocab,
    legacy_shard_vocab_from_arg,
    resolve_shards_card_vocab,
    shard_card_vocab_arrays,
    write_card_vocab_file,
)
from mtg_bot.training.data_gen import (
    save_dataset,
    shard_card_vocab_entries,
    shard_n_printings_cap,
)
from mtg_bot.training.structured_model import (
    DEFAULT_N_PRINTINGS,
    N_EMB_FIELDS,
    N_SCALAR_FIELDS_V2,
)


def _arrays(n_rows: int, seed: int = 0, seq_len: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    policy = np.zeros((n_rows, ACTION_VOCAB_SIZE), dtype=np.float32)
    masks = np.zeros((n_rows, ACTION_VOCAB_SIZE), dtype=np.bool_)
    policy[:, 0], policy[:, 1] = 0.6, 0.4
    masks[:, :2] = True
    return {
        "features": rng.standard_normal((n_rows, FEATURE_DIM)).astype(np.float32),
        "outcomes": rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n_rows),
        "policy_targets": policy,
        "action_masks": masks,
        "structured_token_type": rng.integers(0, 5, (n_rows, seq_len)).astype(np.int64),
        "structured_token_emb_ids": rng.integers(
            0, 3, (n_rows, seq_len, N_EMB_FIELDS),
        ).astype(np.int64),
        "structured_token_scalars": rng.standard_normal(
            (n_rows, seq_len, N_SCALAR_FIELDS_V2),
        ).astype(np.float32),
        "structured_attention_mask": np.ones((n_rows, seq_len), dtype=np.bool_),
    }


def _write(path: Path, n_rows: int, stamp, seed: int = 0) -> Path:
    """``stamp``: "current" (save_dataset's own stamp), None (a legacy
    pre-fingerprint shard), or a dict of npz entries to write."""
    a = _arrays(n_rows, seed)
    if stamp == "current":
        save_dataset(
            a["features"], a["outcomes"], path,
            policy_targets=a["policy_targets"], action_masks=a["action_masks"],
            structured_token_type=a["structured_token_type"],
            structured_token_emb_ids=a["structured_token_emb_ids"],
            structured_token_scalars=a["structured_token_scalars"],
            structured_attention_mask=a["structured_attention_mask"],
        )
        return path
    np.savez_compressed(path, **a, **(stamp or {}))
    return path


@pytest.fixture(autouse=True)
def _no_run_db(monkeypatch):
    monkeypatch.delenv("MTG_BOT_RUN_DB_PATH", raising=False)
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.delenv("MTG_BOT_STRUCTURED_CACHE_DIR", raising=False)
    monkeypatch.delenv(SHARD_CARD_VOCAB_ENV, raising=False)


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------


def test_shard_entry_round_trip_and_errors():
    v = CardVocab.from_names(["a", "b"])
    entry = shard_card_vocab_arrays(v, 320)
    assert set(entry) == {SHARD_KEY}
    assert entry[SHARD_KEY].dtype == np.uint8
    assert decode_shard_card_vocab(entry[SHARD_KEY]) == (v, 320)
    raw = np.frombuffer(b"{bad", dtype=np.uint8)
    with pytest.raises(ShardCardVocabError, match="unreadable"):
        decode_shard_card_vocab(raw, label="x.npz")
    with pytest.raises(ShardCardVocabError, match="JSON object"):
        decode_shard_card_vocab(np.frombuffer(b"[1]", dtype=np.uint8))
    payload = json.loads(entry[SHARD_KEY].tobytes())
    payload["n_printings_cap"] = 1
    with pytest.raises(ShardCardVocabError, match="n_printings_cap"):
        decode_shard_card_vocab(np.frombuffer(json.dumps(payload).encode(), np.uint8))
    payload["n_printings_cap"] = 320
    payload["sha256"] = "0" * 64
    with pytest.raises(ShardCardVocabError, match="does not match"):
        decode_shard_card_vocab(np.frombuffer(json.dumps(payload).encode(), np.uint8))


def test_data_gen_tokenises_at_the_recorded_cap():
    assert shard_n_printings_cap() == DEFAULT_N_PRINTINGS
    # The legacy cap is history (every pre-fingerprint shard), not a
    # mirror of the current default.
    assert LEGACY_SHARD_N_PRINTINGS_CAP == 320
    entry = shard_card_vocab_entries()
    assert decode_shard_card_vocab(entry[SHARD_KEY]) == (
        current_card_vocab(), DEFAULT_N_PRINTINGS,
    )


def test_save_dataset_stamps_structured_shards_only(tmp_path):
    stamped = _write(tmp_path / "s.npz", 3, "current")
    with np.load(stamped) as data:
        assert decode_shard_card_vocab(data[SHARD_KEY]) == (
            current_card_vocab(), DEFAULT_N_PRINTINGS,
        )
    a = _arrays(3)
    flat = tmp_path / "flat.npz"
    save_dataset(a["features"], a["outcomes"], flat)
    with np.load(flat) as data:
        assert SHARD_KEY not in data.files


def test_worker_npz_encoder_stamps_structured_chunks():
    from mtg_bot.worker_daemon_production_runner import _encode_npz_bytes

    a = _arrays(0)
    chunk = (
        a["features"], a["outcomes"], a["policy_targets"], a["action_masks"],
        None, "completed", [], np.zeros((0,), dtype=np.int32),
    )
    with np.load(io.BytesIO(_encode_npz_bytes(chunk, True))) as data:
        assert decode_shard_card_vocab(data[SHARD_KEY])[0] == current_card_vocab()
    with np.load(io.BytesIO(_encode_npz_bytes(chunk, False))) as data:
        assert SHARD_KEY not in data.files


# ---------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------

_OTHER = CardVocab.from_names(["Some Other", "Vocabulary"])


def test_resolve_uniform_stamped_shards(tmp_path):
    paths = [_write(tmp_path / f"{i}.npz", 2, "current", i) for i in range(3)]
    res = resolve_shards_card_vocab(paths)
    assert res.vocab == current_card_vocab()
    assert (res.n_printings_cap, res.n_stamped, res.n_legacy) == (320, 3, 0)
    assert not res.legacy_assumed
    prov = res.provenance("ignored")
    assert prov["n_stamped_files"] == 3 and "legacy_shards_asserted_by" not in prov


def test_resolve_refuses_mixed_vocabularies_and_caps(tmp_path):
    a = _write(tmp_path / "a.npz", 2, "current")
    b = _write(tmp_path / "b.npz", 2, shard_card_vocab_arrays(_OTHER, 320))
    with pytest.raises(ShardCardVocabError, match="2 different") as exc:
        resolve_shards_card_vocab([a, b])
    assert "a.npz" in str(exc.value) and "b.npz" in str(exc.value)
    c = _write(
        tmp_path / "c.npz", 2,
        shard_card_vocab_arrays(current_card_vocab(), 512),
    )
    with pytest.raises(ShardCardVocabError, match="cap 512"):
        resolve_shards_card_vocab([a, c])


def test_legacy_shards_need_an_explicit_assertion(tmp_path):
    legacy = [_write(tmp_path / f"l{i}.npz", 2, None, i) for i in range(2)]
    with pytest.raises(ShardCardVocabError, match="never guessed"):
        resolve_shards_card_vocab(legacy)
    res = resolve_shards_card_vocab(legacy, legacy_vocab=_OTHER)
    assert res.vocab == _OTHER and res.legacy_assumed
    assert (res.n_legacy, res.n_printings_cap) == (2, LEGACY_SHARD_N_PRINTINGS_CAP)
    assert res.provenance("file.json")["legacy_shards_asserted_by"] == "file.json"


def test_legacy_plus_stamped_must_agree(tmp_path):
    legacy = _write(tmp_path / "l.npz", 2, None)
    stamped = _write(tmp_path / "s.npz", 2, "current")
    res = resolve_shards_card_vocab(
        [legacy, stamped], legacy_vocab=current_card_vocab(),
    )
    assert (res.n_stamped, res.n_legacy) == (1, 1)
    with pytest.raises(ShardCardVocabError, match="differs"):
        resolve_shards_card_vocab([legacy, stamped], legacy_vocab=_OTHER)


def test_unjudged_files(tmp_path):
    empty_legacy = _write(tmp_path / "e.npz", 0, None)
    a = _arrays(2)
    flat = tmp_path / "flat.npz"
    np.savez_compressed(flat, features=a["features"], outcomes=a["outcomes"])
    assert resolve_shards_card_vocab([empty_legacy, flat]) is None
    stamped = _write(tmp_path / "s.npz", 2, "current")
    res = resolve_shards_card_vocab([empty_legacy, flat, stamped])
    assert (res.n_stamped, res.n_empty_legacy) == (1, 1)
    # A structured-only shard (no ``outcomes``) is sized off its ids.
    tokens_only = tmp_path / "t.npz"
    a = _arrays(3)
    np.savez_compressed(
        tokens_only,
        structured_token_emb_ids=a["structured_token_emb_ids"],
    )
    with pytest.raises(ShardCardVocabError, match="1 training shards"):
        resolve_shards_card_vocab([tokens_only])


def test_legacy_vocab_arg_and_env(tmp_path, monkeypatch):
    f = tmp_path / "v.json"
    write_card_vocab_file(f, _OTHER)
    assert legacy_shard_vocab_from_arg(None) is None
    assert legacy_shard_vocab_from_arg(str(f)) == _OTHER
    monkeypatch.setenv(SHARD_CARD_VOCAB_ENV, str(f))
    assert legacy_shard_vocab_from_arg(None) == _OTHER


# ---------------------------------------------------------------------
# Trainer + calibrate-value CLIs
# ---------------------------------------------------------------------


def _train(data_dir: Path, registry_root: Path, *extra: str) -> None:
    cli_train.main([
        "--data-dir", str(data_dir),
        "--registry-root", str(registry_root),
        "--epochs", "1", "--batch-size", "4", "--val-split", "0.25",
        "--device", "cpu",
        "--progress-path", str(registry_root.parent / "progress" / "train.json"),
        "--architecture", "structured",
        "--min-holdout-value-quality", "0",
        *extra,
    ])


def _meta(registry_root: Path, version: str = "v1") -> dict:
    return json.loads((registry_root / version / "metadata.json").read_text())


def test_trainer_stamps_model_with_the_shards_vocabulary(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(2):
        _write(data / f"s{i}.npz", 12, "current", i)
    _train(data, tmp_path / "reg")
    meta = _meta(tmp_path / "reg")
    stored = CardVocab.from_dict(meta["card_vocab"])
    assert stored == current_card_vocab()
    prov = meta["card_vocab"]["provenance"]
    assert prov["source"] == "training_shards" and prov["n_stamped_files"] == 2


def test_trainer_refuses_legacy_shards_without_assertion(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "old.npz", 12, None)
    with pytest.raises(SystemExit) as exc:
        _train(data, tmp_path / "reg")
    assert "card vocabulary" in str(exc.value) and "never guessed" in str(exc.value)
    assert not (tmp_path / "reg" / "v1").exists()


def test_trainer_accepts_asserted_legacy_vocab_and_stamps_it(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "old.npz", 12, None)
    vocab_file = tmp_path / "legacy.json"
    write_card_vocab_file(vocab_file, _OTHER)
    _train(data, tmp_path / "reg", "--shard-card-vocab", str(vocab_file))
    meta = _meta(tmp_path / "reg")
    assert CardVocab.from_dict(meta["card_vocab"]) == _OTHER
    assert meta["card_vocab"]["provenance"]["legacy_shards_asserted_by"] == str(vocab_file)
    # The env var is the same assertion (inherited by iterate's train
    # subprocess).
    monkeypatch.setenv(SHARD_CARD_VOCAB_ENV, str(vocab_file))
    _train(data, tmp_path / "reg")
    assert CardVocab.from_dict(_meta(tmp_path / "reg", "v2")["card_vocab"]) == _OTHER


def test_trainer_refuses_mixed_shards(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "a.npz", 12, "current")
    _write(data / "b.npz", 12, shard_card_vocab_arrays(_OTHER, 320))
    with pytest.raises(SystemExit) as exc:
        _train(data, tmp_path / "reg")
    assert "different card vocabularies" in str(exc.value)


def test_trainer_refuses_a_cap_the_model_does_not_have(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "a.npz", 12, shard_card_vocab_arrays(current_card_vocab(), 64))
    with pytest.raises(SystemExit) as exc:
        _train(data, tmp_path / "reg")
    assert exc.value.code == 2  # argparse error
    assert not (tmp_path / "reg" / "v1").exists()


def test_calibrate_refuses_held_out_shards_from_another_id_space(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        _write(data / f"s{i}.npz", 24, "current", i)
    _train(data, tmp_path / "reg")
    other = tmp_path / "other"
    other.mkdir()
    for i in range(2):
        _write(other / f"o{i}.npz", 24, shard_card_vocab_arrays(_OTHER, 320), i)
    rc = cli_calibrate.main([
        "--version", "v1", "--registry-root", str(tmp_path / "reg"),
        "--data-dir", str(other), "--device", "cpu", "--dry-run",
    ])
    assert rc == 2
    assert "card vocabulary mismatch" in capsys.readouterr().err
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write(legacy / "l.npz", 24, None)
    rc = cli_calibrate.main([
        "--version", "v1", "--registry-root", str(tmp_path / "reg"),
        "--data-dir", str(legacy), "--device", "cpu", "--dry-run",
    ])
    assert rc == 2
    assert "never guessed" in capsys.readouterr().err
    # Same-id-space shards calibrate fine.
    rc = cli_calibrate.main([
        "--version", "v1", "--registry-root", str(tmp_path / "reg"),
        "--data-dir", str(data), "--device", "cpu", "--dry-run",
    ])
    assert rc == 0
