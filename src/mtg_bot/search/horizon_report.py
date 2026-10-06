"""Search-horizon + blunder report: play instrumented games and
summarise what the agent's search trees actually contained.

For each agent label (a ladder label, e.g. ``v30+b1`` and
``v30+b1+boardonly``) play ``--games`` seat-alternated games against
a fixed opponent label (default ``heuristic``) on the SAME deals
(seeds, decks and seat order mirror ``mtg_bot ladder``'s
``run_match_parallel(alternate_seats=True)``: game ``g`` puts the
opponent in seat 0 when ``g`` is even). Every decision of the agent
is recorded; every search tree it builds is summarised through the
``MCTS.tree_observer`` hook (:mod:`mtg_bot.search.horizon_probe`);
the real game is scanned by :class:`~mtg_bot.search.blunder_census.
BlunderCensus`. A few consequential decisions are searched with
``choose_with_subtree`` and dumped as readable trees.

Outputs (``--out``): ``summary.json`` (aggregates per agent + per
game), ``report.md``, ``decisions.jsonl.gz`` (one row per agent
decision) and ``trees/*.txt``. Method and definitions:
``docs/SEARCH_HORIZON.md``.

Usage::

    PYTHONPATH=src python -m mtg_bot.search.horizon_report \\
        --agents v30+b1 v30+b1+boardonly --opponent heuristic \\
        --registry-root ../../mtg-bot/trained_weights_s2 \\
        --deck-pool ../../mtg-bot/data/17lands/eoe_decks.json.gz \\
        --games 20 --nodes 200 --samples 2 --workers 10 \\
        --out ../../mtg-bot/analysis_runs/search_horizon_v30
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from mtg_bot.search import horizon_probe as hp
from mtg_bot.search.blunder_census import DETECTORS, BlunderCensus, effect_harm_classes
from mtg_bot.turn import MAIN_STEPS

DUMP_CATEGORIES: tuple[str, ...] = (
    "attack_declaration", "block", "removal_cast", "response_window",
)
_WORKER_DECKS: Optional[list] = None


# ---------------------------------------------------------------------
# Decision context
# ---------------------------------------------------------------------


def decision_context(adapter: Any, actor: int, legal: list) -> str:
    """Coarse kind of a real decision point (see docs)."""
    kinds = {hp.decision_kind(d) for d in legal}
    if kinds & {"attack", "no_attack"}:
        return "attack_declaration"
    if kinds & {"block", "no_block"}:
        return "block"
    if "resolution" in kinds:
        return "resolution"
    eng = adapter.engine_state
    if eng.stack:
        return "response_window"
    active_id = eng.players[eng.active_player_idx].id
    if active_id == actor and eng.step in MAIN_STEPS:
        return "main_phase"
    return "instant_window"


def _is_removal_cast_point(adapter: Any, legal: list) -> bool:
    eng = adapter.engine_state
    for d in legal:
        if hp.decision_kind(d) != "cast":
            continue
        card = hp.find_object(eng, d.card_id)
        if card is not None and effect_harm_classes(card.printing.spell_effect):
            return True
    return False


def _dump_category(adapter: Any, ctx: str, legal: list) -> Optional[str]:
    if ctx in ("attack_declaration", "block"):
        return ctx
    if ctx in ("main_phase", "instant_window", "response_window") and (
        _is_removal_cast_point(adapter, legal)
    ):
        return "removal_cast"
    if ctx == "response_window" and any(
        hp.decision_kind(d) != "pass" for d in legal
    ):
        return "response_window"
    return None


def hidden_opponent_ids(adapter: Any, viewer: int) -> frozenset[int]:
    """Object ids of the opponent's hand / library cards ``viewer``
    can't identify at the real root (the determinizer's targets)."""
    from mtg_bot.search.determinize import viewer_knows
    eng = adapter.engine_state
    out: set[int] = set()
    for p in eng.players:
        if p.id == viewer:
            continue
        for zone in (p.hand, p.library):
            for c in zone:
                if not viewer_knows(c, viewer):
                    out.add(c.id)
    return frozenset(out)


# ---------------------------------------------------------------------
# One instrumented game
# ---------------------------------------------------------------------


def _init_worker(deck_pool_path: str) -> None:
    global _WORKER_DECKS
    from mtg_bot.decks import decks_from_json
    _WORKER_DECKS = decks_from_json(Path(deck_pool_path))


def play_instrumented_game(task: dict[str, Any]) -> dict[str, Any]:
    """Play one game; return per-decision records, tree stats, census
    and (optionally) tree dumps. ``task`` keys: ``agent_label``,
    ``agent_factory``, ``opp_factory``, ``game_idx``, ``base_seed``,
    ``decision_cap``, ``dump`` (bool), ``deck_pool`` (optional; else
    the worker's pool)."""
    try:
        return _play(task)
    except Exception:  # pragma: no cover - surfaced in the summary
        return {
            "agent_label": task["agent_label"], "game_idx": task["game_idx"],
            "error": traceback.format_exc(),
        }


def _play(task: dict[str, Any]) -> dict[str, Any]:
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search.engine_adapter_factory import (
        make_game_state_adapter, resolution_seats_for,
    )
    from mtg_bot.simulate import setup_game
    from mtg_bot.training._worker_threads import pin_worker_torch_threads
    from mtg_bot.training.data_gen import _sample_decks_for_game

    g = int(task["game_idx"])
    base_seed = int(task["base_seed"])
    deck_pool = task.get("deck_pool") or _WORKER_DECKS
    # Seat order mirrors run_match_parallel(alternate_seats=True) for
    # the ladder pair "opponent:agent": even g -> opponent is seat 0.
    agent_seat = 1 if g % 2 == 0 else 0
    p0_factory = task["opp_factory"] if agent_seat == 1 else task["agent_factory"]
    p1_factory = task["agent_factory"] if agent_seat == 1 else task["opp_factory"]
    engine_seed = base_seed + g
    deck_p0, deck_p1 = _sample_decks_for_game(deck_pool, g, base_seed)
    engine = setup_game(
        [RandomAgent(rng=random.Random(engine_seed + 1)),
         RandomAgent(rng=random.Random(engine_seed + 2))],
        decks=[deck_p0.cards, deck_p1.cards], seed=engine_seed,
    )
    strategies = {
        0: p0_factory(base_seed + 100 * g),
        1: p1_factory(base_seed + 100 * g + 50),
    }
    # Same per-seat resolution-decision exposure as every production
    # game (``self_play._play_one_match_game``).
    adapter = make_game_state_adapter(
        engine, expose_combat_decisions=True,
        expose_resolution_decisions=resolution_seats_for(strategies, None),
    )
    pin_worker_torch_threads()
    agent = strategies[agent_seat]

    collected: list[tuple[Any, int]] = []
    current = {"hidden": frozenset(), "keep": False, "trees": []}

    def observer(root: Any, root_player: int) -> None:
        stats = hp.summarize_tree(root, root_player, current["hidden"])
        stats.pop("pv", None)
        collected.append((stats, root_player))
        if current["keep"]:
            current["trees"].append(hp.render_tree(
                root, root_player, max_depth=5, top_k=3,
                actor_name=lambda pid: "AGENT" if pid == agent_seat else "OPP",
            ))

    inner = agent.inner_factory

    def wrapped_inner():
        m = inner()
        m.tree_observer = observer
        return m

    agent.inner_factory = wrapped_inner

    census = BlunderCensus(seat=agent_seat)
    decisions: list[dict[str, Any]] = []
    dumps: dict[str, dict[str, Any]] = {}
    n_total = 0
    cap = int(task.get("decision_cap", 5000))
    t_game = time.perf_counter()
    while not adapter.is_terminal() and n_total < cap:
        actor = adapter.active_player()
        census.observe_moment(adapter)
        if actor != agent_seat:
            d = strategies[actor].choose(adapter, actor)
        else:
            legal = list(adapter.legal_decisions())
            ctx = decision_context(adapter, actor, legal)
            eng = adapter.engine_state
            rec: dict[str, Any] = {
                "turn": eng.turn_number, "step": eng.step.value,
                "context": ctx, "n_options": len(legal),
                "legal_kinds": dict(Counter(hp.decision_kind(x) for x in legal)),
            }
            cat = (
                _dump_category(adapter, ctx, legal)
                if task.get("dump") and len(legal) > 1 else None
            )
            want_dump = cat is not None and cat not in dumps
            current["hidden"] = hidden_opponent_ids(adapter, actor)
            current["keep"] = want_dump
            current["trees"] = []
            collected.clear()
            t0 = time.perf_counter()
            if want_dump:
                chosen, cands = agent.choose_with_subtree(
                    adapter, actor, tree_depth=3, tree_top_k=3,
                )
            else:
                chosen, cands = agent.choose_with_full_stats(adapter, actor)
            rec["ms"] = int((time.perf_counter() - t0) * 1000)
            d = chosen if chosen is not None else legal[0]
            rec["chosen"] = hp.decision_kind(d)
            total_v = sum(int(c["visits"]) for c in cands) or 1
            rec["top_visit_share"] = (
                max(int(c["visits"]) for c in cands) / total_v if cands else 1.0
            )
            rec["trees"] = [s for s, _ in collected]
            decisions.append(rec)
            if want_dump:
                dumps[cat] = {
                    "turn": eng.turn_number, "step": eng.step.value,
                    "context": ctx,
                    "chosen": hp.label_decision(d, adapter, None, actor),
                    "root": [
                        {
                            "label": hp.label_decision(c["decision"], adapter, None, actor),
                            "visits": int(c["visits"]),
                            "q": c.get("q"), "prior": c.get("prior"),
                        }
                        for c in sorted(cands, key=lambda c: -c["visits"])
                    ],
                    "aggregated_subtree": _render_aggregated(cands, adapter, actor),
                    "worlds": current["trees"],
                    "board": _board_text(adapter, actor),
                }
            current["keep"] = False
        census.before_decision(adapter, actor, d)
        adapter.apply(d)
        census.after_decision(adapter)
        n_total += 1
    census.finish(adapter)
    winner = adapter.winner() if adapter.is_terminal() else None
    return {
        "agent_label": task["agent_label"],
        "game_idx": g,
        "agent_seat": agent_seat,
        "winner": winner,
        "agent_won": None if winner is None else (winner == agent_seat),
        "n_decisions_total": n_total,
        "capped": not adapter.is_terminal(),
        "turns": adapter.engine_state.turn_number,
        "wall_s": round(time.perf_counter() - t_game, 1),
        "decisions": decisions,
        "census": dict(census.counts),
        "census_targeted_items": census.targeted_items,
        "census_harmful_targeted_items": census.harmful_targeted_items,
        "census_events": census.events,
        "dumps": dumps,
    }


def _render_aggregated(cands: list, adapter: Any, viewer: int, indent: str = "  ") -> list[str]:
    """Render ``choose_with_subtree``'s PIMC-aggregated candidate tree
    (labels resolved against the REAL root state; ids of hidden cards
    that only exist in a determinized world show as ``#id``)."""
    lines: list[str] = []
    for c in sorted(cands, key=lambda c: -c["visits"]):
        q = c.get("q")
        p = c.get("prior")
        lines.append(
            f"{indent}{hp.label_decision(c['decision'], adapter, None, viewer)}"
            f"   N={c['visits']} Q={'n/a' if q is None else f'{q:+.2f}'}"
            f" P={'n/a' if p is None else f'{p:.2f}'}"
        )
        lines.extend(_render_aggregated(c.get("subtree") or [], adapter, viewer, indent + "    "))
    return lines


def _board_text(adapter: Any, viewer: int) -> list[str]:
    eng = adapter.engine_state
    out = []
    for p in eng.players:
        who = "AGENT" if p.id == viewer else "OPP"
        perms = [
            f"{o.printing.name}{'(T)' if o.tapped else ''}"
            + (f" {o.printing.power}/{o.printing.toughness}" if o.printing.is_creature else "")
            for o in eng.battlefield if o.controller_id == p.id
        ]
        hand = (
            [c.printing.name for c in p.hand] if p.id == viewer
            else [f"<{len(p.hand)} hidden>"]
        )
        out.append(f"{who} life={p.life} library={len(p.library)} hand={hand}")
        out.append(f"    battlefield: {', '.join(perms) or '-'}")
    if eng.stack:
        out.append("stack (top last): " + ", ".join(
            f"{it.source.printing.name}->{list(it.targets)}" for it in eng.stack
        ))
    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def _mean(xs: list) -> Optional[float]:
    return round(statistics.fmean(xs), 3) if xs else None


def _share(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def _opt_bucket(n: int) -> str:
    return "1" if n <= 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"


def aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one agent label's per-game results."""
    ok = [g for g in games if "error" not in g]
    decs = [d for g in ok for d in g["decisions"]]
    trees = [t for d in decs for t in d["trees"]]
    n_dec = len(decs)
    forced = [d for d in decs if d["n_options"] <= 1]
    unforced = [d for d in decs if d["n_options"] > 1]
    ctx_counts = Counter(d["context"] for d in decs)
    out: dict[str, Any] = {
        "games": len(ok),
        "errors": [g.get("error", "")[-400:] for g in games if "error" in g],
        "agent_wins": sum(1 for g in ok if g["agent_won"] is True),
        "agent_losses": sum(1 for g in ok if g["agent_won"] is False),
        "capped_or_draw": sum(1 for g in ok if g["agent_won"] is None),
        "mean_turns": _mean([g["turns"] for g in ok]),
        "agent_decisions": n_dec,
        "agent_decisions_per_game": _share(n_dec, len(ok)),
        "forced_share": _share(len(forced), n_dec),
        "near_forced_share_of_unforced": _share(
            sum(1 for d in unforced if d["top_visit_share"] >= 0.9), len(unforced),
        ),
        "options_histogram": dict(Counter(_opt_bucket(d["n_options"]) for d in decs)),
        "contexts": {
            c: {
                "n": n,
                "forced": sum(1 for d in decs if d["context"] == c and d["n_options"] <= 1),
                "mean_options": _mean([d["n_options"] for d in decs if d["context"] == c]),
                "chosen": dict(Counter(d["chosen"] for d in decs if d["context"] == c)),
            }
            for c, n in ctx_counts.most_common()
        },
        "mean_search_ms_unforced": _mean([d["ms"] for d in unforced]),
    }
    out["trees"] = _tree_aggregate(trees)
    out["trees_by_context"] = {
        c: _tree_aggregate([t for d in decs if d["context"] == c for t in d["trees"]])
        for c in ctx_counts
    }
    # Resolution reached for the CHOSEN root child, by its kind.
    res: dict[str, Counter] = defaultdict(Counter)
    for t in trees:
        if t["chosen_index"] is None:
            continue
        ch = t["root_children"][t["chosen_index"]]
        res[ch["kind"]][ch["resolution"]] += 1
    out["chosen_resolution"] = {
        k: {
            **dict(v),
            "reached_share": _share(
                v.get("reached", 0), v.get("reached", 0) + v.get("not_reached", 0),
            ),
        }
        for k, v in sorted(res.items())
    }
    any_res: dict[str, Counter] = defaultdict(Counter)
    for t in trees:
        for ch in t["root_children"]:
            any_res[ch["kind"]][ch["resolution"]] += 1
    out["all_root_children_resolution"] = {
        k: {
            **dict(v),
            "reached_share": _share(
                v.get("reached", 0), v.get("reached", 0) + v.get("not_reached", 0),
            ),
        }
        for k, v in sorted(any_res.items())
    }
    totals = Counter()
    for g in ok:
        totals.update(g["census"])
    out["blunders"] = {
        k: {"total": int(totals.get(k, 0)), "per_game": _share(int(totals.get(k, 0)), len(ok))}
        for k in DETECTORS
    }
    out["targeted_items"] = sum(g.get("census_targeted_items", 0) for g in ok)
    out["harmful_targeted_items"] = sum(
        g.get("census_harmful_targeted_items", 0) for g in ok
    )
    return out


def _tree_aggregate(trees: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trees)
    if not n:
        return {"n_trees": 0}
    nodes = sum(t["n_nodes"] for t in trees)
    reach_nodes = Counter()
    leaf_reach = Counter()
    for t in trees:
        reach_nodes.update(t["reach_nodes"])
        leaf_reach.update(t["leaf_reach"])
    max_reach = Counter(t["max_reach"] for t in trees)
    cum = {}
    for i, cat in enumerate(hp.REACH_CATEGORIES):
        cum[cat] = _share(
            sum(max_reach.get(c, 0) for c in hp.REACH_CATEGORIES[i:]), n,
        )
    return {
        "n_trees": n,
        "nodes_per_tree_mean": _mean([t["n_nodes"] for t in trees]),
        "nodes_per_tree_min": min(t["n_nodes"] for t in trees),
        "root_options_mean": _mean([t["root_options"] for t in trees]),
        "max_depth_mean": _mean([t["max_depth"] for t in trees]),
        "max_depth_median": statistics.median(t["max_depth"] for t in trees),
        "max_choice_depth_mean": _mean([t["max_choice_depth"] for t in trees]),
        "mean_node_depth": _mean([t["mean_depth"] for t in trees]),
        "pv_len_mean": _mean([t["pv_len"] for t in trees]),
        "pv_choice_len_mean": _mean([t["pv_choice_len"] for t in trees]),
        "max_reach_share": {c: _share(max_reach.get(c, 0), n) for c in hp.REACH_CATEGORIES},
        "max_reach_at_least": cum,
        "pv_reach_share": {
            c: _share(sum(1 for t in trees if t["pv_reach"] == c), n)
            for c in hp.REACH_CATEGORIES
        },
        "node_reach_share": {c: _share(reach_nodes.get(c, 0), nodes) for c in hp.REACH_CATEGORIES},
        "leaf_reach_share": {
            c: _share(leaf_reach.get(c, 0), sum(leaf_reach.values()))
            for c in hp.REACH_CATEGORIES
        },
        "node_share_forced": _share(sum(t["nodes_forced"] for t in trees), nodes),
        "node_share_pass_edge": _share(sum(t["nodes_pass_edge"] for t in trees), nodes),
        "node_share_opp_point": _share(sum(t["nodes_opp_point"] for t in trees), nodes),
        "node_share_imagined_edge": _share(sum(t["nodes_imagined_edge"] for t in trees), nodes),
        "node_share_under_imagined": _share(sum(t["nodes_under_imagined"] for t in trees), nodes),
        "node_share_terminal": _share(sum(t["nodes_terminal"] for t in trees), nodes),
    }


# ---------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_outputs(
    out_dir: Path, config: dict[str, Any], per_agent: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trees_dir = out_dir / "trees"
    trees_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"config": config, "agents": {}}
    with gzip.open(out_dir / "decisions.jsonl.gz", "wt", encoding="utf-8") as f:
        for label, games in per_agent.items():
            for g in games:
                for i, d in enumerate(g.get("decisions", [])):
                    f.write(json.dumps({"agent": label, "game": g["game_idx"], "i": i, **d}) + "\n")
    dump_files: dict[str, list[str]] = {}
    for label, games in per_agent.items():
        agg = aggregate(games)
        agg["per_game"] = [
            {k: g.get(k) for k in (
                "game_idx", "agent_seat", "agent_won", "turns",
                "n_decisions_total", "capped", "wall_s", "census",
            )} | {"error": "error" in g}
            for g in sorted(games, key=lambda g: g["game_idx"])
        ]
        agg["census_events"] = [
            {"game": g["game_idx"], **e}
            for g in games for e in g.get("census_events", [])
        ]
        summary["agents"][label] = agg
        files = []
        for cat in DUMP_CATEGORIES:
            for g in sorted(games, key=lambda g: g["game_idx"]):
                dump = g.get("dumps", {}).get(cat)
                if dump is None:
                    continue
                safe = label.replace("+", "_")
                path = trees_dir / f"{safe}__{cat}__g{g['game_idx']}.txt"
                path.write_text(_dump_text(label, g, cat, dump), encoding="utf-8")
                files.append(path.name)
                break
        dump_files[label] = files
    summary["tree_dumps"] = dump_files
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def _dump_text(label: str, game: dict, cat: str, dump: dict) -> str:
    lines = [
        f"agent {label}  game {game['game_idx']} (agent seat {game['agent_seat']})",
        f"category: {cat}   context: {dump['context']}   turn {dump['turn']} {dump['step']}",
        f"chosen: {dump['chosen']}",
        "",
        "board at the decision (agent's view):",
        *("  " + ln for ln in dump["board"]),
        "",
        "root options, PIMC-aggregated over the determinized worlds "
        "(choose_with_subtree; Q from the agent's view, P = policy prior):",
        *("  " + f"{r['label']}   N={r['visits']} "
          f"Q={'n/a' if r['q'] is None else format(r['q'], '+.2f')} "
          f"P={'n/a' if r['prior'] is None else format(r['prior'], '.2f')}"
          for r in dump["root"]),
        "",
        "aggregated subtree (tree_depth=3, tree_top_k=3; labels from the real state):",
        *dump["aggregated_subtree"],
        "",
    ]
    for i, world in enumerate(dump["worlds"]):
        lines.append(f"--- determinized world {i + 1}: full tree (top 3 per node, 5 levels; "
                     "Q = mean value from the ACTOR's view; forced chains folded) ---")
        lines.extend(world)
        lines.append("")
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    labels = list(summary["agents"])
    L: list[str] = [
        "# Search horizon + blunder report",
        "",
        f"Agents {', '.join(labels)} vs `{cfg.get('opponent')}`; "
        f"{cfg.get('games')} games each (same deals), nodes {cfg.get('nodes')}, "
        f"samples {cfg.get('samples')}, seed {cfg.get('seed')}, git {cfg.get('git_commit')}. "
        "Definitions: `docs/SEARCH_HORIZON.md`.",
        "",
        "## Headline",
        "",
        "| metric | " + " | ".join(f"`{l}`" for l in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]

    def row(name: str, fn) -> None:
        vals = []
        for l in labels:
            try:
                vals.append(fn(summary["agents"][l]))
            except (KeyError, TypeError, ZeroDivisionError):
                vals.append("n/a")
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    row("agent W-L (draw/capped)", lambda a: f"{a['agent_wins']}-{a['agent_losses']} ({a['capped_or_draw']})")
    row("agent decisions / game", lambda a: f"{a['agent_decisions_per_game']:.0f}")
    row("forced decisions (1 option)", lambda a: _pct(a["forced_share"]))
    row("near-forced (top child >= 90% visits) of unforced", lambda a: _pct(a["near_forced_share_of_unforced"]))
    row("trees", lambda a: str(a["trees"]["n_trees"]))
    row("nodes per tree (mean)", lambda a: f"{a['trees']['nodes_per_tree_mean']:.0f}")
    row("tree depth in plies (mean of max)", lambda a: f"{a['trees']['max_depth_mean']:.1f}")
    row("real choices on deepest path (mean of max)", lambda a: f"{a['trees']['max_choice_depth_mean']:.1f}")
    row("PV length plies / real choices", lambda a: f"{a['trees']['pv_len_mean']:.1f} / {a['trees']['pv_choice_len_mean']:.1f}")
    for cat in hp.REACH_CATEGORIES[1:]:
        row(f"trees reaching >= {cat}", lambda a, c=cat: _pct(a["trees"]["max_reach_at_least"][c]))
    row("nodes still in the root's step", lambda a: _pct(a["trees"]["node_reach_share"]["same_step"]))
    row("nodes at next turn or later", lambda a: _pct(
        a["trees"]["node_reach_share"]["next_turn"] + a["trees"]["node_reach_share"]["two_plus_turns"]))
    row("nodes spent at forced points", lambda a: _pct(a["trees"]["node_share_forced"]))
    row("nodes that are a pass", lambda a: _pct(a["trees"]["node_share_pass_edge"]))
    row("nodes at opponent decision points", lambda a: _pct(a["trees"]["node_share_opp_point"]))
    row("nodes = imagined opp-card action", lambda a: _pct(a["trees"]["node_share_imagined_edge"]))
    row("nodes under an imagined opp-card action", lambda a: _pct(a["trees"]["node_share_under_imagined"]))
    for kind in ("attack", "no_attack", "block", "no_block", "cast", "activate"):
        row(f"chosen `{kind}`: tree reaches its resolution", lambda a, k=kind: (
            f"{_pct(a['chosen_resolution'][k]['reached_share'])} "
            f"(n={a['chosen_resolution'][k].get('reached', 0) + a['chosen_resolution'][k].get('not_reached', 0)})"
        ))
    row("search ms per unforced decision", lambda a: f"{a['mean_search_ms_unforced']:.0f}")
    L += ["", "## Blunder census (per game)", "",
          "| detector | " + " | ".join(f"`{l}`" for l in labels) + " |",
          "|---|" + "---|" * len(labels)]
    for det in DETECTORS:
        row(det, lambda a, d=det: f"{a['blunders'][d]['per_game']:.2f} ({a['blunders'][d]['total']})")
    row("agent targeted stack items (auto-targeted)", lambda a: str(a["targeted_items"]))
    row("... of which damage/destroy/exile", lambda a: str(a["harmful_targeted_items"]))
    L += ["", "## Decision contexts", ""]
    for l in labels:
        a = summary["agents"][l]
        L.append(f"### `{l}`")
        L.append("")
        L.append("| context | n | forced | mean options | chosen kinds | nodes/tree | reach >= next turn |")
        L.append("|---|---|---|---|---|---|---|")
        for c, info in a["contexts"].items():
            ta = a["trees_by_context"].get(c, {})
            L.append(
                f"| {c} | {info['n']} | {info['forced']} | {info['mean_options']} | "
                f"{info['chosen']} | {ta.get('nodes_per_tree_mean', 'n/a')} | "
                f"{_pct((ta.get('max_reach_at_least') or {}).get('next_turn'))} |"
            )
        L.append("")
    L += ["## Tree dumps", ""]
    for l, files in summary.get("tree_dumps", {}).items():
        for fn in files:
            L.append(f"- `trees/{fn}`")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True,
            text=True, check=True, cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    from mtg_bot.cli.ladder import resolve_agent

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--opponent", default="heuristic")
    p.add_argument("--registry-root", type=Path, required=True)
    p.add_argument("--deck-pool", type=Path, required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--nodes", type=int, default=200)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--decision-cap", type=int, default=5000)
    p.add_argument("--dump-games", type=int, default=6,
                   help="Only the first N games per agent capture tree dumps.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.deck_pool.is_file():
        # Fail before spawning: a worker initializer that raises makes
        # the pool respawn workers forever.
        p.error(f"--deck-pool {args.deck_pool} does not exist")

    opp = resolve_agent(args.opponent, registry_root=args.registry_root,
                        nodes=args.nodes, samples=args.samples)
    specs = {
        lab: resolve_agent(lab, registry_root=args.registry_root,
                           nodes=args.nodes, samples=args.samples)
        for lab in args.agents
    }
    tasks = [
        {
            "agent_label": lab, "agent_factory": spec.factory,
            "opp_factory": opp.factory, "game_idx": g, "base_seed": args.seed,
            "decision_cap": args.decision_cap, "dump": g < args.dump_games,
        }
        for g in range(args.games) for lab, spec in specs.items()
    ]
    config = {
        "agents": args.agents, "opponent": args.opponent, "games": args.games,
        "nodes": args.nodes, "samples": args.samples, "seed": args.seed,
        "deck_pool": str(args.deck_pool), "registry_root": str(args.registry_root),
        "git_commit": _git_commit(),
        "agent_descriptions": {k: v.description for k, v in specs.items()},
        "opponent_description": opp.description,
    }
    print(f"[horizon] {len(tasks)} games on {args.workers} workers -> {args.out}", flush=True)
    per_agent: dict[str, list] = {lab: [] for lab in args.agents}
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(str(args.deck_pool),), maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(play_instrumented_game, tasks):
            per_agent[res["agent_label"]].append(res)
            done = sum(len(v) for v in per_agent.values())
            status = "ERROR" if "error" in res else (
                f"won={res['agent_won']} turns={res['turns']} wall={res['wall_s']}s"
            )
            print(f"[horizon] {done}/{len(tasks)} {res['agent_label']} g{res['game_idx']}: {status}",
                  flush=True)
    config["wall_s"] = round(time.perf_counter() - t0, 1)
    summary = write_outputs(args.out, config, per_agent)
    print(f"[horizon] wrote {args.out / 'summary.json'} and report.md "
          f"({sum(len(v) for v in summary['tree_dumps'].values())} tree dumps)", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
