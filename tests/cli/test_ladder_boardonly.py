"""Ladder label modifiers ``+boardonly`` and ``+s<K>`` and the config
threading of ``opponent_hidden_model`` (2026-10-05)."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from mtg_bot.cli import ladder as L
from mtg_bot.search.determinize import determinize
from mtg_bot.training.config import (
    HeuristicConfig, NNPimcConfig, PimcConfig,
)


def _policy_version(root: Path, label: str = "v30") -> None:
    d = root / label
    d.mkdir(parents=True)
    (d / "weights.pt").write_bytes(b"not-real-weights")
    (d / "metadata.json").write_text(json.dumps({
        "version": label, "created_at": "2026-10-05T21:52:56Z",
        "feature_dim": 51, "hidden_sizes": [],
        "with_policy": True, "architecture": "structured",
        "action_encoding_version": "canonical_v2",
        "training": {"structured_config": {"emb_dim": 128}},
        "git_commit": "5d3d871a",
    }), encoding="utf-8")


# ---------------------------------------------------------------------
# split_search_modifiers
# ---------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    ("v30", ("v30", "sampled", 2, 200)),
    ("v30+b1", ("v30+b1", "sampled", 2, 200)),
    ("v30+boardonly", ("v30", "board_only", 2, 200)),
    ("v30+b1+boardonly", ("v30+b1", "board_only", 2, 200)),
    ("v30+boardonly+b1", ("v30+b1", "board_only", 2, 200)),
    ("v30+b1+boardonly+s1", ("v30+b1", "board_only", 1, 400)),
    ("v30+s4", ("v30", "sampled", 4, 100)),
    ("heuristic+s1", ("heuristic", "sampled", 1, 400)),
    ("v30+valueonly+boardonly+b1", ("v30+valueonly+b1", "board_only", 2, 200)),
])
def test_split_search_modifiers(label, expected):
    assert L.split_search_modifiers(label, nodes=200, samples=2) == expected


def test_s_modifier_floor_divides_total_budget():
    assert L.split_search_modifiers("v30+s3", nodes=200, samples=2) == (
        "v30", "sampled", 3, 133,
    )


@pytest.mark.parametrize("label,match", [
    ("v30+boardonly+boardonly", "given twice"),
    ("v30+s1+s2", "more than one"),
    ("v30+s0", "K >= 1"),
    ("v30+s999", "no node"),
])
def test_split_search_modifiers_rejects(label, match):
    with pytest.raises(ValueError, match=match):
        L.split_search_modifiers(label, nodes=2, samples=1)


def test_unrelated_s_prefixed_modifier_not_swallowed(tmp_path):
    _policy_version(tmp_path)
    with pytest.raises(ValueError, match="unknown agent variant"):
        L.resolve_agent("v30+sx", registry_root=tmp_path, nodes=4, samples=1)


# ---------------------------------------------------------------------
# resolve_agent
# ---------------------------------------------------------------------


def test_heuristic_boardonly_builds_board_only_pimc(tmp_path):
    s = L.resolve_agent(
        "heuristic+boardonly", registry_root=tmp_path, nodes=200, samples=2,
    )
    d = s.description
    assert d["kind"] == "heuristic_pimc"
    assert d["opponent_hidden_model"] == "board_only"
    assert (d["samples"], d["inner_nodes"]) == (2, 200)
    strat = s.factory(0)
    assert isinstance(strat._determinize, functools.partial)
    assert strat._determinize.keywords == {"opponent_hidden_model": "board_only"}


def test_plain_heuristic_keeps_sampled_default(tmp_path):
    s = L.resolve_agent("heuristic", registry_root=tmp_path, nodes=200, samples=2)
    assert s.description["opponent_hidden_model"] == "sampled"
    assert s.factory(0)._determinize is determinize


def test_nn_boardonly_s1_reshapes_only_that_seat(tmp_path):
    _policy_version(tmp_path)
    a = L.resolve_agent(
        "v30+b1+boardonly+s1", registry_root=tmp_path, nodes=200, samples=2,
    )
    d = a.description
    assert a.label == "v30+b1+boardonly+s1"
    assert d["kind"] == "nn_pimc" and d["version"] == "v30"
    assert d["opponent_hidden_model"] == "board_only"
    assert (d["samples"], d["inner_nodes"], d["batch_size"]) == (1, 400, 1)
    b = L.resolve_agent("v30+b1", registry_root=tmp_path, nodes=200, samples=2)
    assert b.description["opponent_hidden_model"] == "sampled"
    assert (b.description["samples"], b.description["inner_nodes"]) == (2, 200)


def test_boardonly_composes_with_nocal_and_heurvalue(tmp_path):
    _policy_version(tmp_path)
    nocal = L.resolve_agent(
        "v30+nocal+boardonly", registry_root=tmp_path, nodes=8, samples=1,
    ).description
    assert nocal["apply_value_calibration"] is False
    assert nocal["opponent_hidden_model"] == "board_only"
    hv = L.resolve_agent(
        "v30+heurvalue+boardonly+s1", registry_root=tmp_path, nodes=8, samples=2,
    ).description
    assert hv["kind"] == "hybrid_heur_value_nn_policy"
    assert hv["nn"]["opponent_hidden_model"] == "board_only"
    assert (hv["samples"], hv["inner_nodes"]) == (1, 16)


def test_random_rejects_search_modifiers(tmp_path):
    with pytest.raises(ValueError, match="does not search"):
        L.resolve_agent("random+boardonly", registry_root=tmp_path, nodes=4, samples=1)


def test_boardonly_labels_survive_pair_parsing():
    agents = ["heuristic", "v30+b1+boardonly", "v30+b1+boardonly+s1"]
    assert L.parse_pairs(
        "heuristic:v30+b1+boardonly,heuristic:v30+b1+boardonly+s1", agents,
    ) == [("heuristic", agents[1]), ("heuristic", agents[2])]


# ---------------------------------------------------------------------
# Config validation / serialisation
# ---------------------------------------------------------------------


def test_configs_validate_and_roundtrip_opponent_model():
    with pytest.raises(ValueError, match="opponent_hidden_model"):
        PimcConfig(opponent_hidden_model="bogus")
    with pytest.raises(ValueError, match="opponent_hidden_model"):
        NNPimcConfig(weights_path="w.pt", opponent_hidden_model="bogus")
    p = PimcConfig(samples=1, inner_nodes=5, heuristic=HeuristicConfig(),
                   opponent_hidden_model="board_only")
    assert PimcConfig.from_dict(p.to_dict()) == p
    n = NNPimcConfig(weights_path="w.pt", opponent_hidden_model="board_only")
    assert NNPimcConfig.from_dict(n.to_dict()) == n
    # Old serialised dicts (no key) load as the default.
    d = n.to_dict()
    del d["opponent_hidden_model"]
    assert NNPimcConfig.from_dict(d).opponent_hidden_model == "sampled"
    d = p.to_dict()
    del d["opponent_hidden_model"]
    assert PimcConfig.from_dict(d).opponent_hidden_model == "sampled"


def test_belief_weighted_rejects_board_only():
    with pytest.raises(ValueError, match="belief"):
        NNPimcConfig(
            weights_path="w.pt", use_ismcts=True, use_belief_weighted=True,
            opponent_hidden_model="board_only",
        )
