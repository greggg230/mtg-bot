"""The production switch for searched resolution decisions: the
game-level ``HeuristicsConfig`` field (env default), the per-agent
opt-out on the search configs and the per-seat exposure every game
constructor derives from them."""

from __future__ import annotations

import pytest

from mtg_bot.search.engine_adapter_factory import (
    ResolutionPriorUnavailableError, resolution_seats_for,
    strategy_searches_resolution,
)
from mtg_bot.search.heuristics_config import (
    SEARCH_RESOLUTION_DECISIONS_ENV, HeuristicsConfig,
    default_search_resolution_decisions,
)
from mtg_bot.training.config import (
    HeuristicConfig, NNPimcConfig, NNServerPimcConfig, PimcConfig,
    _can_prior_resolution, _tag_resolution,
)


class _S:
    """Stand-in built strategy."""

    def __init__(self, search=True, can_prior=True):
        self.search_resolution_decisions = search
        self.can_prior_resolution_decisions = can_prior
        self.resolution_prior_note = "stub seat"


@pytest.mark.parametrize("raw,expected", [
    (None, True), ("", True), ("1", True), ("TRUE", True), ("on", True),
    ("0", False), ("false", False), ("No", False), ("off", False),
])
def test_env_default(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(SEARCH_RESOLUTION_DECISIONS_ENV, raising=False)
    else:
        monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, raw)
    assert default_search_resolution_decisions() is expected
    assert HeuristicsConfig().search_resolution_decisions is expected
    # Not a training shortcut: disabled() keeps the process default.
    assert HeuristicsConfig.disabled().search_resolution_decisions is expected
    assert HeuristicsConfig.from_dict({}).search_resolution_decisions is expected


def test_env_default_rejects_garbage(monkeypatch):
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "maybe")
    with pytest.raises(ValueError, match="expected one of"):
        default_search_resolution_decisions()


def test_heuristics_round_trip_carries_the_switch(monkeypatch):
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "1")
    cfg = HeuristicsConfig(search_resolution_decisions=False)
    assert cfg.to_dict()["search_resolution_decisions"] is False
    assert HeuristicsConfig.from_dict(cfg.to_dict()) == cfg


def test_resolution_seats_for(monkeypatch):
    on = HeuristicsConfig(search_resolution_decisions=True)
    off = HeuristicsConfig(search_resolution_decisions=False)
    both = {0: _S(), 1: _S()}
    assert resolution_seats_for(both, on) == frozenset({0, 1})
    assert resolution_seats_for(both, off) == frozenset()
    assert resolution_seats_for({0: _S(), 1: _S(search=False)}, on) == {0}
    # A plain object without the attributes searches (default).
    assert resolution_seats_for({0: object()}, on) == {0}
    assert strategy_searches_resolution(object())
    # heuristics=None follows the env default.
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "0")
    assert resolution_seats_for(both, None) == frozenset()
    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "1")
    assert resolution_seats_for(both, None) == {0, 1}
    with pytest.raises(ResolutionPriorUnavailableError, match="stub seat"):
        resolution_seats_for({0: _S(can_prior=False)}, on)
    # ... but an opted-out seat with no priors is fine.
    assert resolution_seats_for({0: _S(search=False, can_prior=False)}, on) == set()


def test_pimc_config_tags_its_strategy():
    on = PimcConfig(samples=1, inner_nodes=4, heuristic=HeuristicConfig()).build(0)
    assert on.search_resolution_decisions and on.can_prior_resolution_decisions
    off = PimcConfig(
        samples=1, inner_nodes=4, search_resolution_decisions=False,
    )
    built = off.build(0)
    assert built.search_resolution_decisions is False
    d = off.to_dict()
    assert d["search_resolution_decisions"] is False
    assert PimcConfig.from_dict(d) == off
    legacy = dict(d)
    legacy.pop("search_resolution_decisions")
    assert PimcConfig.from_dict(legacy).search_resolution_decisions is True


def test_nn_pimc_config_round_trip_and_prior_capability():
    cfg = NNPimcConfig(weights_path="w.pt", search_resolution_decisions=False)
    d = cfg.to_dict()
    assert d["search_resolution_decisions"] is False
    assert NNPimcConfig.from_dict(d) == cfg
    d.pop("search_resolution_decisions")
    assert NNPimcConfig.from_dict(d).search_resolution_decisions is True

    class _Ev:
        def __init__(self, v):
            self.action_encoding_version = v

    assert _can_prior_resolution(_Ev("canonical_v2"), puct=False)
    assert not _can_prior_resolution(_Ev("canonical_v2"), puct=True)
    assert _can_prior_resolution(_Ev("canonical_v3"), puct=True)
    assert not _can_prior_resolution(object(), puct=True)


def test_tag_resolution_and_server_config():
    s = _tag_resolution(_S(), search=False, can_prior=True, note="n")
    assert (s.search_resolution_decisions, s.resolution_prior_note) == (False, "n")
    from mtg_bot.search.nn_eval_server import NNServerHandle

    handle = NNServerHandle(address=("localhost", 1), authkey=b"x", with_policy=True)
    built = NNServerPimcConfig(server_handle=handle, puct=True).build(0)
    assert built.can_prior_resolution_decisions is False
    built2 = NNServerPimcConfig(server_handle=handle, puct=False).build(0)
    assert built2.can_prior_resolution_decisions is True


def test_match_game_exposes_only_opted_in_seats(monkeypatch):
    """``self_play._play_one_match_game`` builds the adapter with
    per-seat exposure: a searched heuristic seat vs an auto-targeting
    one."""
    from mtg_bot.search import engine_adapter_factory as factory
    from mtg_bot.training import self_play
    from tests._test_decks import simple_test_deck_pool

    monkeypatch.setenv(SEARCH_RESOLUTION_DECISIONS_ENV, "1")
    seen = {}
    real = factory.make_game_state_adapter

    def spy(engine, **kwargs):
        seen["seats"] = kwargs.get("expose_resolution_decisions")
        return real(engine, **kwargs)

    monkeypatch.setattr(self_play, "make_game_state_adapter", spy)
    a = PimcConfig(samples=1, inner_nodes=2)
    b = PimcConfig(samples=1, inner_nodes=2, search_resolution_decisions=False)
    self_play._play_one_match_game(
        0, a.build, b.build, 0, 30, deck_pool=simple_test_deck_pool(),
    )
    assert seen["seats"] == frozenset({0})
