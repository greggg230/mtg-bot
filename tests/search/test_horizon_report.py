"""``mtg_bot.search.horizon_report``: instrumented game loop,
aggregation, report writing and the CLI (tiny heuristic budgets)."""

from __future__ import annotations

import gzip
import json

import pytest

from mtg_bot.decks import Deck, decks_to_json
from mtg_bot.search import horizon_report as hr
from mtg_bot.search.blunder_census import DETECTORS
from mtg_bot.training.config import HeuristicConfig, PimcConfig
from tests._test_decks import simple_test_decklist


def _pool():
    return [
        Deck(draft_id=f"d{i}", build_index=0, cards=sorted(simple_test_decklist(c)))
        for i, c in enumerate("GRWUB")
    ]


def _task(g: int, model: str = "sampled", dump: bool = True) -> dict:
    agent = PimcConfig(samples=1, inner_nodes=12, heuristic=HeuristicConfig(),
                       opponent_hidden_model=model)
    opp = PimcConfig(samples=1, inner_nodes=6, heuristic=HeuristicConfig())
    return {
        "agent_label": f"h+{model}", "agent_factory": agent.build,
        "opp_factory": opp.build, "game_idx": g, "base_seed": 3,
        "decision_cap": 4000, "dump": dump, "deck_pool": _pool(),
    }


@pytest.fixture(scope="module")
def games():
    return {
        model: [hr.play_instrumented_game(_task(g, model)) for g in range(2)]
        for model in ("sampled", "board_only")
    }


def test_game_records_are_complete(games):
    for model, gs in games.items():
        for g in gs:
            assert "error" not in g, g.get("error")
            assert g["agent_seat"] == (1 if g["game_idx"] % 2 == 0 else 0)
            assert g["decisions"], "agent made no decisions"
            assert set(g["census"]) == set(DETECTORS)
            for d in g["decisions"]:
                assert d["n_options"] >= 1
                assert (len(d["trees"]) == 0) == (d["n_options"] == 1)
                for t in d["trees"]:
                    assert t["n_nodes"] <= 12


def test_board_only_never_imagines_opponent_cards(games):
    for g in games["board_only"]:
        for d in g["decisions"]:
            for t in d["trees"]:
                assert t["nodes_imagined_edge"] == 0
                assert t["nodes_under_imagined"] == 0


def test_same_deals_across_modes(games):
    a = [(g["game_idx"], g["agent_seat"]) for g in games["sampled"]]
    b = [(g["game_idx"], g["agent_seat"]) for g in games["board_only"]]
    assert a == b


def test_aggregate_and_write_outputs(tmp_path, games):
    per_agent = {"h+sampled": games["sampled"], "h+board_only": games["board_only"]}
    summary = hr.write_outputs(tmp_path, {"games": 2, "opponent": "h"}, per_agent)
    a = summary["agents"]["h+sampled"]
    assert a["games"] == 2
    assert a["agent_wins"] + a["agent_losses"] + a["capped_or_draw"] == 2
    assert 0.0 <= a["forced_share"] <= 1.0
    assert a["trees"]["n_trees"] > 0
    shares = a["trees"]["node_reach_share"]
    assert abs(sum(shares.values()) - 1.0) < 1e-3
    assert set(a["blunders"]) == set(DETECTORS)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Headline" in report and "`h+board_only`" in report
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(loaded["agents"]) == set(per_agent)
    with gzip.open(tmp_path / "decisions.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    assert len(rows) == sum(len(g["decisions"]) for gs in per_agent.values() for g in gs)
    for label, files in summary["tree_dumps"].items():
        for fn in files:
            text = (tmp_path / "trees" / fn).read_text(encoding="utf-8")
            assert "root options" in text and "determinized world 1" in text


def test_aggregate_handles_errors_and_empty():
    out = hr.aggregate([{"agent_label": "x", "game_idx": 0, "error": "boom"}])
    assert out["games"] == 0 and out["errors"] == ["boom"]
    assert out["trees"] == {"n_trees": 0}


def test_play_instrumented_game_reports_errors():
    bad = _task(0)
    bad["deck_pool"] = []
    res = hr.play_instrumented_game(bad)
    assert "error" in res and "deck_pool is empty" in res["error"]


def test_main_rejects_missing_deck_pool(tmp_path):
    with pytest.raises(SystemExit):
        hr.main([
            "--agents", "heuristic", "--registry-root", str(tmp_path),
            "--deck-pool", str(tmp_path / "nope.json.gz"),
            "--out", str(tmp_path / "out"),
        ])


def test_main_end_to_end(tmp_path):
    pool_path = tmp_path / "pool.json.gz"
    decks_to_json(_pool(), pool_path)
    rc = hr.main([
        "--agents", "heuristic", "heuristic+boardonly",
        "--registry-root", str(tmp_path), "--deck-pool", str(pool_path),
        "--games", "1", "--nodes", "6", "--samples", "1", "--workers", "1",
        "--dump-games", "1", "--out", str(tmp_path / "out"),
    ])
    assert rc == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["agents"]) == {"heuristic", "heuristic+boardonly"}
    desc = summary["config"]["agent_descriptions"]["heuristic+boardonly"]
    assert desc["opponent_hidden_model"] == "board_only"


def test_dump_path_renders_trees(tmp_path, monkeypatch):
    """Force every multi-option decision into a dump category so the
    dump renderers (aggregated subtree, board, per-world trees) run."""
    monkeypatch.setattr(hr, "_dump_category", lambda adapter, ctx, legal: "attack_declaration")
    res = hr.play_instrumented_game(_task(1))
    assert "error" not in res, res.get("error")
    dump = res["dumps"]["attack_declaration"]
    assert dump["root"] and dump["worlds"] and dump["board"]
    assert any("N=" in ln for ln in dump["aggregated_subtree"])
    summary = hr.write_outputs(tmp_path, {"games": 1}, {"h": [res]})
    (fn,) = summary["tree_dumps"]["h"]
    text = (tmp_path / "trees" / fn).read_text(encoding="utf-8")
    assert "board at the decision" in text and "AGENT life=" in text


def test_decision_context_and_dump_category_classification():
    from mtg_bot.actions import CastSpell, PassPriority
    from mtg_bot.search.combat_decisions import AttackerDecision, BlockerDecision
    from mtg_bot.search.resolution_decisions import ResolutionTargetDecision
    from types import SimpleNamespace
    from mtg_bot.turn import Step

    eng = SimpleNamespace(stack=[], players=[SimpleNamespace(id=0), SimpleNamespace(id=1)],
                          active_player_idx=0, step=Step.PRECOMBAT_MAIN, battlefield=[])
    ad = SimpleNamespace(engine_state=eng)
    atk = [AttackerDecision(attacker_id=1, attack=True), AttackerDecision(attacker_id=1, attack=False)]
    assert hr.decision_context(ad, 0, atk) == "attack_declaration"
    assert hr._dump_category(ad, "attack_declaration", atk) == "attack_declaration"
    blk = [BlockerDecision(blocker_id=2, attacker_id=None)]
    assert hr.decision_context(ad, 0, blk) == "block"
    assert hr.decision_context(ad, 0, [ResolutionTargetDecision(target=("player", 0))]) == "resolution"
    assert hr.decision_context(ad, 0, [PassPriority()]) == "main_phase"
    assert hr.decision_context(ad, 1, [PassPriority()]) == "instant_window"
    eng.stack = [object()]
    assert hr.decision_context(ad, 0, [PassPriority()]) == "response_window"
    # Response window with a non-pass option (card id unknown -> not removal).
    legal = [PassPriority(), CastSpell(card_id=99)]
    assert hr._dump_category(ad, "response_window", legal) == "response_window"
    assert hr._dump_category(ad, "response_window", [PassPriority()]) is None
    assert hr._dump_category(ad, "main_phase", legal) is None
