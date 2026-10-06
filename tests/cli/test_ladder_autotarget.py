"""``mtg_bot ladder`` ``+autotarget``: a seat whose resolution decisions
are made by ``DefaultTacticalAgent`` (the pre-2026-10-06 behaviour),
and the up-front refusal of an NN seat that can't prior them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtg_bot.cli import ladder as L
from mtg_bot.search.heuristics_config import SEARCH_RESOLUTION_DECISIONS_ENV


def _version(root: Path, label: str, encoding: str, with_policy=True) -> None:
    d = root / label
    d.mkdir(parents=True)
    (d / "weights.pt").write_bytes(b"x")
    (d / "metadata.json").write_text(json.dumps({
        "version": label, "created_at": "x", "feature_dim": 51,
        "hidden_sizes": [], "with_policy": with_policy,
        "architecture": "structured", "action_encoding_version": encoding,
    }), encoding="utf-8")


@pytest.fixture
def searched(monkeypatch):
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "1")


def test_split_autotarget_modifier():
    assert L.split_autotarget_modifier("heuristic") == ("heuristic", True)
    assert L.split_autotarget_modifier("v30r+b1+autotarget") == ("v30r+b1", False)
    assert L.split_autotarget_modifier("v30+autotarget+nocal") == ("v30+nocal", False)
    with pytest.raises(ValueError, match="given twice"):
        L.split_autotarget_modifier("x+autotarget+autotarget")


def test_heuristic_and_random_seats(searched, tmp_path):
    h = L.resolve_agent("heuristic", registry_root=tmp_path, nodes=4, samples=1)
    assert h.description["search_resolution_decisions"] is True
    ha = L.resolve_agent(
        "heuristic+autotarget", registry_root=tmp_path, nodes=4, samples=1,
    )
    assert ha.description["search_resolution_decisions"] is False
    assert ha.factory(0).search_resolution_decisions is False
    r = L.resolve_agent("random+autotarget", registry_root=tmp_path, nodes=4, samples=1)
    assert r.description == {"kind": "random", "search_resolution_decisions": False}
    assert r.factory(0).search_resolution_decisions is False
    assert L.resolve_agent(
        "random", registry_root=tmp_path, nodes=4, samples=1,
    ).factory(0).search_resolution_decisions is True


def test_v2_nn_seat_needs_autotarget_or_an_extension(searched, tmp_path):
    _version(tmp_path, "v30", "canonical_v2")
    with pytest.raises(ValueError, match=r"v30\+autotarget"):
        L.resolve_agent("v30", registry_root=tmp_path, nodes=4, samples=1)
    with pytest.raises(ValueError, match="no resolution-decision slots"):
        L.resolve_agent("v30+heurvalue", registry_root=tmp_path, nodes=4, samples=1)
    ok = L.resolve_agent(
        "v30+b1+autotarget", registry_root=tmp_path, nodes=4, samples=1,
    )
    assert ok.description["search_resolution_decisions"] is False
    # value-only seats run UCB1: no priors needed.
    vo = L.resolve_agent("v30+valueonly", registry_root=tmp_path, nodes=4, samples=1)
    assert vo.description["search_resolution_decisions"] is True


def test_v3_nn_seat_searches(searched, tmp_path):
    _version(tmp_path, "v30r", "canonical_v3")
    s = L.resolve_agent("v30r+b1", registry_root=tmp_path, nodes=4, samples=1)
    assert s.description["search_resolution_decisions"] is True
    assert s.description["action_encoding_version"] == "canonical_v3"


def test_switch_off_skips_the_refusal(monkeypatch, tmp_path):
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "0")
    _version(tmp_path, "v30", "canonical_v2")
    s = L.resolve_agent("v30", registry_root=tmp_path, nodes=4, samples=1)
    assert s.description["kind"] == "nn_pimc"


def test_heurvalue_policy_head_check_comes_first(searched, tmp_path):
    _version(tmp_path, "v9", "canonical_v2", with_policy=False)
    with pytest.raises(ValueError, match="policy head"):
        L.resolve_agent("v9+heurvalue", registry_root=tmp_path, nodes=4, samples=1)
