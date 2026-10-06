"""``mtg_bot.training.resolution_extension``: appending the
canonical_v3 resolution slots to a trained canonical_v2 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.training import action_encoding as ae
from mtg_bot.training import resolution_extension as rx
from mtg_bot.training.registry import Registry
from mtg_bot.training.structured_model import (
    StructuredModel, StructuredModelConfig,
)
from tests._card_vocab_helpers import current_vocab_dict
from tests.search._resolution_helpers import (
    chaos_adapter, paused_bombard_adapter,
)

_TINY = {"emb_dim": 16, "n_heads": 2, "n_layers": 1, "hidden_dim": 32}


def _write_v2_checkpoint(version_dir: Path, *, encoding="canonical_v2"):
    torch.manual_seed(0)
    model = StructuredModel(StructuredModelConfig(**_TINY, scalar_schema_version=2))
    # Train-like perturbation so rows are not all at init.
    with torch.no_grad():
        model.policy_slot_queries.add_(torch.randn_like(model.policy_slot_queries))
    version_dir.mkdir(parents=True)
    torch.save(model.state_dict(), version_dir / "weights.pt")
    meta = {
        "version": version_dir.name,
        "created_at": "2026-10-05T00:00:00Z",
        "feature_dim": 51,
        "hidden_sizes": [],
        "with_policy": True,
        "architecture": "structured",
        "action_encoding_version": encoding,
        "scalar_schema_version": 2,
        "value_calibration": {"method": "logit_temperature", "temperature": 0.9, "bias": 0.0},
        "card_vocab": current_vocab_dict(),
        "training": {
            "structured_config": dict(_TINY, scalar_schema_version=2),
            **{f"structured_{k}": v for k, v in _TINY.items()},
            "structured_scalar_schema_version": 2,
        },
        "training_history": {"best_val_epoch": 3},
        "evaluation": {"vs_v1": {"win_rate": 0.6}},
        "git_commit": "abc12345",
        "notes": "trained",
        "extra": {},
    }
    (version_dir / "metadata.json").write_text(json.dumps(meta))
    return model, meta


def test_extend_state_dict_copies_everything_and_appends_seeded_rows(tmp_path):
    model, _ = _write_v2_checkpoint(tmp_path / "v7")
    sd = model.state_dict()
    out = rx.extend_state_dict(sd, seed=5)
    assert list(out.keys()) == list(sd.keys())
    for k, v in sd.items():
        if k == rx.POLICY_SLOT_QUERIES_KEY:
            assert torch.equal(out[k][:517], v)
            assert out[k].shape == (805, 16)
        else:
            assert out[k] is v
    again = rx.extend_state_dict(sd, seed=5)
    assert torch.equal(again[rx.POLICY_SLOT_QUERIES_KEY], out[rx.POLICY_SLOT_QUERIES_KEY])
    other = rx.extend_state_dict(sd, seed=6)
    assert not torch.equal(other[rx.POLICY_SLOT_QUERIES_KEY][517:], out[rx.POLICY_SLOT_QUERIES_KEY][517:])
    big = rx.fresh_slot_rows(4000, 16, seed=1)
    assert abs(float(big.std()) - rx.INIT_STD) < 0.002
    assert abs(float(big.mean())) < 0.002


def test_extend_state_dict_refuses_non_v2_heads():
    with pytest.raises(rx.ResolutionExtensionError, match="not a structured"):
        rx.extend_state_dict({"trunk.w": torch.zeros(2)})
    with pytest.raises(rx.ResolutionExtensionError, match="only a positional_v1"):
        rx.extend_state_dict({rx.POLICY_SLOT_QUERIES_KEY: torch.zeros(805, 4)})


def test_extend_metadata_rewrites_encoding_and_keeps_calibration(tmp_path):
    _, meta = _write_v2_checkpoint(tmp_path / "v7")
    out = rx.extend_metadata(
        meta, source_version="v7", source_weights_sha256="ff", seed=9,
        git_commit="deadbeef",
    )
    assert out["action_encoding_version"] == "canonical_v3"
    assert out["training"]["structured_action_vocab_size"] == 805
    assert out["training"]["structured_config"]["action_vocab_size"] == 805
    assert out["value_calibration"] == meta["value_calibration"]
    assert out["card_vocab"] == meta["card_vocab"]
    assert out["training_history"] == meta["training_history"]
    assert out["evaluation"] == {}
    ext = out["extra"][rx.EXTENSION_KEY]
    assert ext["source_version"] == "v7" and ext["seed"] == 9
    assert ext["n_new_slots"] == 288 and ext["extension_git_commit"] == "deadbeef"
    assert out["notes"].startswith("v7 + canonical_v3")
    assert meta["action_encoding_version"] == "canonical_v2"  # not mutated


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(architecture="shared"), "only structured"),
    (lambda m: m.update(action_encoding_version="canonical_v3"), "not extendable"),
    (lambda m: m.update(action_encoding_version="positional_v1"), "positional_v1 heads"),
    (lambda m: m.update(card_vocab={}), "no card_vocab"),
])
def test_extend_metadata_refusals(tmp_path, mutate, match):
    _, meta = _write_v2_checkpoint(tmp_path / "v7")
    mutate(meta)
    with pytest.raises(rx.ResolutionExtensionError, match=match):
        rx.extend_metadata(meta, source_version="v7", source_weights_sha256="", seed=1)


def test_extended_checkpoint_is_bit_identical_outside_pauses(tmp_path):
    _write_v2_checkpoint(tmp_path / "v7")
    meta = rx.extend_checkpoint_dir(tmp_path / "v7", tmp_path / "v7r")
    assert meta["version"] == "v7r"
    with pytest.raises(rx.ResolutionExtensionError, match="not empty"):
        rx.extend_checkpoint_dir(tmp_path / "v7", tmp_path / "v7r")
    rx.extend_checkpoint_dir(tmp_path / "v7", tmp_path / "v7r", overwrite=True)
    with pytest.raises(rx.ResolutionExtensionError, match="must differ"):
        rx.extend_checkpoint_dir(tmp_path / "v7", tmp_path / "v7")

    def load(d):
        m = json.loads((d / "metadata.json").read_text())
        return NNEvaluator.from_metadata(d / "weights.pt", m)

    old, new = load(tmp_path / "v7"), load(tmp_path / "v7r")
    assert (old.action_vocab_size, new.action_vocab_size) == (517, 805)
    import random
    rng = random.Random(0)
    adapter = chaos_adapter(2, expose_resolution_decisions=False)
    n = 0
    while not adapter.is_terminal() and n < 120:
        who = adapter.active_player()
        vo, po = old.evaluate_with_policy(adapter, who)
        vn, pn = new.evaluate_with_policy(adapter, who)
        assert vo == vn
        assert np.array_equal(po, pn[:517]) and not pn[517:].any()
        adapter.apply(rng.choice(list(adapter.legal_decisions())))
        n += 1
    # At a pause the extended net priors the resolution options.
    paused = paused_bombard_adapter()
    v, policy = new.evaluate_with_policy(paused, 0)
    assert policy[:517].sum() == 0
    assert policy[517:].sum() == pytest.approx(1.0, abs=1e-5)
    _, zero = old.evaluate_with_policy(paused, 0)
    assert not zero.any()


def test_register_extended_and_cli(tmp_path, capsys):
    root = tmp_path / "reg"
    _write_v2_checkpoint(root / "v3")
    new_id = rx.register_extended(root, "v3", seed=2)
    assert new_id == "v4"
    meta = Registry(root).load_metadata("v4")
    assert meta.action_encoding_version == "canonical_v3"
    assert meta.extra[rx.EXTENSION_KEY]["source_version"] == "v3"
    assert rx.main([
        "--registry-root", str(root), "--version", "v3",
        "--out-dir", str(tmp_path / "staged" / "v3r"),
    ]) == 0
    assert (tmp_path / "staged" / "v3r" / "weights.pt").is_file()
    assert rx.main(["--registry-root", str(root), "--version", "v3", "--register"]) == 0
    assert Registry(root).has("v5")
    out = capsys.readouterr().out
    assert "registered v5" in out and "wrote" in out


def test_cli_dispatcher_knows_the_subcommand():
    from mtg_bot.cli import _SUBCOMMANDS
    assert _SUBCOMMANDS["extend-resolution-slots"] == "mtg_bot.training.resolution_extension"


def _find_real_v30() -> Path:
    """``trained_weights_s2/v30`` of this checkout, or of the main
    checkout next to a worktree (``mtg-bot-worktrees/<wt>``)."""
    here = Path(__file__).resolve()
    for root in (here.parents[2], here.parents[3].parent / "mtg-bot"):
        cand = root / "trained_weights_s2" / "v30"
        if (cand / "weights.pt").is_file():
            return cand
    return here.parents[2] / "trained_weights_s2" / "v30"


_REAL_V30 = _find_real_v30()


@pytest.mark.skipif(
    not (_REAL_V30 / "weights.pt").is_file(),
    reason="real s2 v30 checkpoint not available on this machine",
)
def test_real_v30_extension_is_bit_identical(tmp_path):
    """The production case: a copy of real ``trained_weights_s2/v30``
    (read only) extended; value and the 517 old policy slots are
    bit-identical on real states, value calibration and card vocab
    carried over."""
    meta = rx.extend_checkpoint_dir(_REAL_V30, tmp_path / "v30r")
    src_meta = json.loads((_REAL_V30 / "metadata.json").read_text())
    assert meta["value_calibration"] == src_meta["value_calibration"]
    assert meta["card_vocab"]["sha256"] == src_meta["card_vocab"]["sha256"]
    old = NNEvaluator.from_metadata(_REAL_V30 / "weights.pt", src_meta)
    new = NNEvaluator.from_metadata(tmp_path / "v30r" / "weights.pt", meta)
    import random
    rng = random.Random(3)
    adapter = chaos_adapter(7, expose_resolution_decisions=False)
    for _ in range(40):
        if adapter.is_terminal():
            break
        who = adapter.active_player()
        vo, po = old.evaluate_with_policy(adapter, who)
        vn, pn = new.evaluate_with_policy(adapter, who)
        assert vo == vn
        assert np.array_equal(po, pn[:517])
        adapter.apply(rng.choice(list(adapter.legal_decisions())))


def test_set_current_best_and_cli_flag(tmp_path, capsys):
    root = tmp_path / "reg"
    _write_v2_checkpoint(root / "v3")
    assert rx.set_current_best(root, "v9") is None  # no state file yet
    (root / "iterate_state.json").write_text(json.dumps({
        "current_best": "v3", "history": [{"iteration": 1}],
        "iterations_completed": 1,
    }))
    assert rx.main([
        "--registry-root", str(root), "--version", "v3", "--register",
        "--set-current-best",
    ]) == 0
    state = json.loads((root / "iterate_state.json").read_text())
    assert state["current_best"] == "v4"
    assert state["history"] == [{"iteration": 1}]
    assert "current_best: v3 -> v4" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rx.main([
            "--registry-root", str(root), "--version", "v3",
            "--out-dir", str(tmp_path / "x"), "--set-current-best",
        ])


def test_set_current_best_without_state_file_is_reported(tmp_path, capsys):
    root = tmp_path / "reg"
    _write_v2_checkpoint(root / "v3")
    assert rx.main([
        "--registry-root", str(root), "--version", "v3", "--register",
        "--set-current-best",
    ]) == 0
    assert "current_best not set" in capsys.readouterr().out


def test_iterate_lineage_encoding_carries_the_extension_forward():
    from mtg_bot.cli.iterate import resolve_lineage_action_encoding as r

    assert r("canonical_v2", None) == "canonical_v2"
    assert r("canonical_v2", "canonical_v2") == "canonical_v2"
    assert r("canonical_v2", "canonical_v3") == "canonical_v3"
    with pytest.raises(ValueError, match="only the"):
        r("positional_v1", "canonical_v2")
    with pytest.raises(ValueError, match="only the"):
        r("canonical_v3", "canonical_v2")
