"""Resolution supervision through data-gen, shards and the trainer's
policy-width plan (``docs/RESOLUTION_DECISIONS.md`` §5 / §8)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mtg_bot.decks import Deck
from mtg_bot.search.heuristics_config import (
    SEARCH_RESOLUTION_DECISIONS_ENV, HeuristicsConfig,
)
from mtg_bot.training import action_encoding as ae
from mtg_bot.training import data_gen as dg
from mtg_bot.training.config import HeuristicConfig, PimcConfig
from mtg_bot.training.structured_dataset import (
    PolicyWidthError, _FileScan, load_structured_dataset, plan_policy_width,
)
from tests.search._resolution_helpers import chaos_decklists


def _chaos_pool(seed=0):
    a, b = chaos_decklists(seed)
    return [
        Deck(draft_id="chaos_a", build_index=0, cards=a),
        Deck(draft_id="chaos_b", build_index=0, cards=b),
    ]


_TINY_PIMC = PimcConfig(samples=1, inner_nodes=6, heuristic=HeuristicConfig())


def _game(version, *, searched=True, seed=0, game_idx=0):
    return dg._generate_one_game_data(
        game_idx=game_idx, p0_factory=_TINY_PIMC.build,
        p1_factory=_TINY_PIMC.build, base_seed=seed, decision_cap=1500,
        sample_every=1, deck_pool=_chaos_pool(seed),
        heuristics=HeuristicsConfig(search_resolution_decisions=searched),
        action_encoding_version=version, structured_tokens=True,
    )


# ---------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------


def test_resolution_stats_helpers():
    s = dg.new_resolution_stats("canonical_v3")
    assert s == {
        "action_encoding_version": "canonical_v3",
        "resolution_pause_samples": 0, "resolution_policy_rows": 0,
        "resolution_overflow_visits": 0,
    }
    arrays = dg.resolution_stats_arrays(dict(s, resolution_policy_rows=3))
    assert str(arrays["action_encoding_version"]) == "canonical_v3"
    assert int(arrays["resolution_policy_rows"]) == 3
    assert dg.resolution_stats_arrays(None) == {}
    merged = dg.merge_resolution_stats([
        dict(s, resolution_policy_rows=2), None,
        dict(s, resolution_policy_rows=5, resolution_pause_samples=6),
    ])
    assert merged["resolution_policy_rows"] == 7
    assert merged["resolution_pause_samples"] == 6
    assert dg.merge_resolution_stats([None, {}]) is None
    with pytest.raises(ValueError, match="different action-encoding"):
        dg.merge_resolution_stats([s, dg.new_resolution_stats("canonical_v2")])
    masks = np.zeros((4, 805), dtype=bool)
    masks[1, 600] = masks[3, 517] = masks[2, 3] = True
    derived = dg.resolution_stats_from_masks("canonical_v3", masks)
    assert derived["resolution_policy_rows"] == 2
    assert derived["resolution_pause_samples"] == -1
    assert dg.resolution_stats_from_masks(
        "canonical_v2", np.zeros((2, 517), dtype=bool),
    )["resolution_policy_rows"] == 0


def test_empty_arrays_width():
    assert dg._empty_arrays()[2].shape == (0, 517)
    assert dg._empty_arrays(805)[3].shape == (0, 805)
    chunk = dg._empty_arrays(805) + (None, "crashed", [], dg._empty_game_ids())
    f, o, p, m, g = dg._concat_chunks([chunk])
    assert p.shape == (0, 805)


# ---------------------------------------------------------------------
# data-gen
# ---------------------------------------------------------------------


def test_v3_game_writes_resolution_rows_with_counts():
    found = False
    for seed in range(4):
        chunk = _game("canonical_v3", seed=seed)
        stats = chunk[8]
        policy, masks = chunk[2], chunk[3]
        assert policy.shape[1] == masks.shape[1] == 805
        res_rows = int(masks[:, 517:].any(axis=1).sum())
        assert res_rows == stats["resolution_policy_rows"]
        assert stats["resolution_policy_rows"] <= stats["resolution_pause_samples"]
        if stats["resolution_pause_samples"]:
            found = True
            # Resolution rows carry policy mass only in the block.
            rows = masks[:, 517:].any(axis=1)
            assert not masks[rows, :517].any()
            assert np.allclose(policy[rows, 517:].sum(axis=1), 1.0, atol=1e-5)
    assert found, "no seed reached a resolution pause"


def test_v2_encoding_with_searched_resolution_fails_loudly():
    raised = False
    for seed in range(4):
        try:
            chunk = _game("canonical_v2", seed=seed)
        except dg.ResolutionSupervisionError as exc:
            assert "0 encodable resolution policy rows" in str(exc)
            raised = True
            break
        assert chunk[8]["resolution_pause_samples"] == 0
    assert raised


def test_auto_targeting_game_has_no_pauses():
    chunk = _game("canonical_v2", searched=False)
    assert chunk[8]["resolution_pause_samples"] == 0
    assert chunk[2].shape[1] == 517


# ---------------------------------------------------------------------
# Shard writers
# ---------------------------------------------------------------------


def _rows(n, width):
    return (
        np.zeros((n, 51), np.float32), np.zeros((n,), np.float32),
        np.zeros((n, width), np.float32), np.zeros((n, width), bool),
    )


def test_save_dataset_stamps_and_checks_width(tmp_path):
    f, o, p, m = _rows(3, 805)
    m[0, 600] = True
    p[0, 600] = 1.0
    path = tmp_path / "a.npz"
    dg.save_dataset(
        f, o, path, policy_targets=p, action_masks=m,
        action_encoding_version="canonical_v3",
    )
    with np.load(path) as z:
        assert str(z["action_encoding_version"]) == "canonical_v3"
        assert int(z["resolution_policy_rows"]) == 1
        assert int(z["resolution_pause_samples"]) == -1
    with pytest.raises(ValueError, match="805 slots"):
        dg.save_dataset(
            *_rows(2, 517)[:2], tmp_path / "b.npz",
            policy_targets=_rows(2, 517)[2], action_masks=_rows(2, 517)[3],
            resolution_stats=dg.new_resolution_stats("canonical_v3"),
        )


def test_worker_npz_writer_stamps_the_chunk_stats():
    from mtg_bot import worker_daemon_production_runner as prr

    f, o, p, m = _rows(2, 805)
    stats = dict(dg.new_resolution_stats("canonical_v3"), resolution_policy_rows=1)
    chunk = (f, o, p, m, 0, "completed", [], np.zeros(2, np.int32), stats)
    with np.load(io.BytesIO(prr._encode_npz_bytes(chunk))) as z:
        assert str(z["action_encoding_version"]) == "canonical_v3"
        assert int(z["resolution_policy_rows"]) == 1
    assert prr._policy_width("canonical_v3") == 805
    assert prr._policy_width(None) == 517


# ---------------------------------------------------------------------
# Trainer-side policy-width plan
# ---------------------------------------------------------------------


def _scan(width, version=None, n_rows=4, name="f"):
    return _FileScan(
        path=Path(name), n_rows=n_rows, row_lengths=np.zeros(n_rows, np.int64),
        n_tokens=0, emb_width=7, scalar_width=29, policy_width=width,
        encoding_version=version,
    )


def test_plan_policy_width_rules():
    plan = plan_policy_width(
        [
            _scan(805, "canonical_v3", name="new"),
            _scan(517, "canonical_v2", name="stamped_v2"),
            _scan(517, None, name="legacy"),
            _scan(None, name="value_only"),
            _scan(517, "positional_v1", n_rows=0, name="empty"),
        ],
        action_encoding_version="canonical_v3",
        legacy_policy_encoding="canonical_v2",
    )
    assert plan.width == 805
    assert plan.pad_paths == {Path("stamped_v2"), Path("legacy")}
    r = plan.report
    assert (r["copied_files"], r["padded_files"], r["padded_rows"]) == (1, 2, 8)
    assert (r["zero_row_files"], r["files_without_policy"]) == (1, 1)
    # v2 training: legacy 517 files are copied as before.
    p2 = plan_policy_width([_scan(517)], action_encoding_version="canonical_v2")
    assert p2.width == 517 and not p2.pad_paths


@pytest.mark.parametrize("scans,version,legacy,match", [
    ([_scan(517)], "canonical_v3", None, "legacy-policy-encoding"),
    ([_scan(805, "canonical_v3")], "canonical_v2", None, "can't train"),
    ([_scan(517, "positional_v1")], "canonical_v3", None, "can't train"),
    ([_scan(600, "canonical_v3")], "canonical_v3", None, "stamped"),
    ([_scan(600)], "canonical_v3", "canonical_v2", "neither"),
])
def test_plan_policy_width_refusals(scans, version, legacy, match):
    with pytest.raises(PolicyWidthError, match=match):
        plan_policy_width(
            scans, action_encoding_version=version,
            legacy_policy_encoding=legacy,
        )


def _write_struct_npz(path, width, version=None, *, res_row=False):
    n = 2
    f, o, p, m = _rows(n, width)
    p[:, 0] = 1.0
    m[:, 0] = True
    if res_row:
        p[1, 0] = 0.0
        m[1, 0] = False
        p[1, 520] = 1.0
        m[1, 520] = True
    arrays = {
        "features": f, "outcomes": o, "policy_targets": p,
        "action_masks": m,
        "structured_token_type": np.ones((n, 3), np.int64),
        "structured_token_emb_ids": np.zeros((n, 3, 7), np.int64),
        "structured_token_scalars": np.zeros((n, 3, 29), np.float32),
        "structured_attention_mask": np.ones((n, 3), bool),
        "game_ids": np.arange(n, dtype=np.int32),
    }
    if version is not None:
        arrays.update(dg.resolution_stats_arrays(dg.new_resolution_stats(version)))
    np.savez_compressed(path, **arrays)


def test_load_structured_dataset_pads_explicitly_and_reports(tmp_path):
    _write_struct_npz(tmp_path / "a_legacy.npz", 517)
    _write_struct_npz(tmp_path / "b_new.npz", 805, "canonical_v3", res_row=True)
    with pytest.raises(PolicyWidthError, match="legacy-policy-encoding"):
        load_structured_dataset(tmp_path, action_encoding_version="canonical_v3")
    ds = load_structured_dataset(
        tmp_path, action_encoding_version="canonical_v3",
        legacy_policy_encoding="canonical_v2",
    )
    try:
        assert ds.policy_width == 805 and len(ds) == 4
        rep = ds.policy_report
        assert rep["padded_rows"] == 2 and rep["resolution_target_rows"] == 1
        legacy_row = ds[0]
        assert legacy_row["policy_target"].shape == (805,)
        assert float(legacy_row["policy_target"][0]) == 1.0
        assert not bool(legacy_row["action_mask"][517:].any())
    finally:
        ds.close()
    # The pre-2026-10-06 inferred path still refuses mixed widths.
    with pytest.raises(ValueError, match="width mismatch"):
        load_structured_dataset(tmp_path)


def test_trainer_refuses_a_policy_width_mismatch(tmp_path):
    from mtg_bot.training.structured_model import StructuredModelConfig
    from mtg_bot.training.trainer import train_structured_model

    _write_struct_npz(tmp_path / "a.npz", 805, "canonical_v3")
    ds = load_structured_dataset(tmp_path, action_encoding_version="canonical_v3")
    try:
        assert ds.policy_report["copied_files"] == 1
        with pytest.raises(ValueError, match="policy width mismatch"):
            train_structured_model(
                ds, epochs=1, config=StructuredModelConfig(
                    emb_dim=8, n_heads=2, n_layers=1, hidden_dim=8,
                    scalar_schema_version=2,
                ),
                device="cpu", weights_out_path=tmp_path / "w.pt",
            )
    finally:
        ds.close()


def test_in_memory_dataset_has_no_policy_report():
    from mtg_bot.training.structured_dataset import StructuredDataset

    ds = StructuredDataset.__new__(StructuredDataset)
    ds._memmap_dir = str(Path("does-not-exist"))
    assert ds.policy_report == {}
