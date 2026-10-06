"""``mtg_bot ladder`` — local fixed-reference strength ladder with Elo.

Every other eval in the project is *relative*: "new vs current
champion" (``iterate``'s acceptance gate, ``mtg_bot eval``). None of
them ever tells you how strong a network is on an absolute scale, and
the production lineage has never been played against the two fixed
yardsticks the project already ships — the uniformly-random strategy
and the heuristic-evaluator PIMC. This subcommand fills that gap:

* a set of participants (``random``, ``heuristic``, any registered
  version label under ``trained_weights/``),
* a set of pairings (round robin or explicit),
* ``--games`` seat-alternated games per pair via the same
  :func:`mtg_bot.training.self_play.run_match_parallel` path the
  acceptance gate uses (local ``multiprocessing.Pool`` — no
  coordinator, no worker daemons, no runs.db),
* a Bradley-Terry / logistic maximum-likelihood Elo fit over all
  pairs with ``random`` anchored at 0 and a bootstrap 95% CI,
* the raw pairwise win-rate matrix with Wilson 95% CIs,
* incremental, resumable persistence (``results.json`` in the out
  dir, atomic ``os.replace`` writes) and a ``ladder.md`` report.

Scoring conventions (see ``--timeouts-as-draws``): by default a
timed-out or crashed game contributes NO signal — it is dropped from
the denominator, matching the project's gate convention
(``MatchStats.win_rate``). With ``--timeouts-as-draws`` a timed-out
game is scored 0.5 / 0.5. Neither player can be blamed for a
per-game wall-clock timeout, so there is deliberately no
"timeouts-as-losses" mode. Crashed games are always excluded (they
are engine bugs, not play).

Elo scale: the standard logistic, ``P(a beats b) = 1 / (1 + 10^((r_b
- r_a)/400))``. The fit is the Hunter (2004) minorization-maximization
iteration for Bradley-Terry with draws counted as half a win each
way. One virtual draw per played pair keeps an agent that never
scores (e.g. ``random`` at 0/500) at a finite rating; at ladder game
counts the shift is well under a point.

See ``docs/STRENGTH_LADDER.md`` for usage and the running results
table.
"""

from __future__ import annotations

from mtg_bot.search.leaf_batch import SEARCH_LEAF_BATCH_SIZE

import argparse
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from mtg_bot.cli import _common
from mtg_bot.cli._common import depth_parser, runtime_parser
from mtg_bot.search.strategy import RandomChoice, SearchStrategy

# Reserved participant labels. Anything else is looked up in the
# registry (``trained_weights/<label>/``).
RANDOM_LABEL = "random"
HEURISTIC_LABEL = "heuristic"
VARIANT_SUFFIXES = frozenset({"valueonly", "heurvalue"})


def _split_label(label: str) -> tuple[str, Optional[str], Optional[int]]:
    """``"v22+heurvalue+b1"`` -> ``("v22", "heurvalue", 1)``.

    Modifiers after ``+``: a head variant from ``VARIANT_SUFFIXES``
    (at most one) and/or ``b<N>`` = MCTS leaf batch size (default
    ``SEARCH_LEAF_BATCH_SIZE`` = 1 for every agent since 2026-10-01; ``heuristic+b32`` / ``v22+b1``
    are the search-batching controls, 2026-10-01).
    """
    parts = label.split("+")
    base, variant, batch = parts[0], None, None
    for mod in parts[1:]:
        if mod in VARIANT_SUFFIXES:
            if variant is not None:
                raise ValueError(f"{label!r}: more than one head variant")
            variant = mod
        elif mod.startswith("b") and mod[1:].isdigit() and int(mod[1:]) >= 1:
            batch = int(mod[1:])
        else:
            raise ValueError(
                f"unknown agent variant {mod!r} in {label!r}; expected one "
                f"of {sorted(VARIANT_SUFFIXES)} or b<N> (leaf batch size)"
            )
    return base, variant, batch
# ``<version>+nocal``: load the version WITHOUT its post-hoc value
# calibration (``metadata.value_calibration``). Every NN seat applies
# the calibration since 2026-10-04; before that ``NNPimcConfig.build``
# silently dropped it, so ``+nocal`` reproduces every earlier ladder /
# arena / self-play evaluator and measures what the calibration is
# worth in play.
NOCAL_MODIFIER = "nocal"


def split_calibration_modifier(label: str) -> tuple[str, bool]:
    """``"v22+nocal"`` -> ``("v22", False)``; ``"v22"`` -> ``("v22",
    True)``. Returns the label with any ``+nocal`` removed (other
    ``+`` modifiers keep their order) and whether the value
    calibration is applied. ``+nocal`` twice is an error."""
    parts = label.split("+")
    kept = [parts[0]] + [m for m in parts[1:] if m != NOCAL_MODIFIER]
    n_nocal = len(parts) - len(kept)
    if n_nocal > 1:
        raise ValueError(f"{label!r}: '+{NOCAL_MODIFIER}' given twice")
    return "+".join(kept), n_nocal == 0

# ``<label>+boardonly``: the seat's search does not imagine the
# opponent's hidden cards -- every opponent hand / library card it
# can't see becomes an inert placeholder in each PIMC world
# (``opponent_hidden_model="board_only"``, see
# ``mtg_bot.search.determinize``). ``<label>+s<K>``: K determinizations
# sharing the ladder's TOTAL node budget (``--nodes`` x ``--samples``),
# i.e. ``inner_nodes = nodes * samples // K`` -- ``+s1`` gives one world
# every node. Both compose with every other modifier and work for
# ``heuristic`` too (2026-10-05).
BOARD_ONLY_MODIFIER = "boardonly"


def split_search_modifiers(
    label: str, *, nodes: int, samples: int,
) -> tuple[str, str, int, int]:
    """Strip ``+boardonly`` / ``+s<K>`` from ``label``.

    Returns ``(label_without_them, opponent_hidden_model, samples,
    inner_nodes)`` where the last two are the seat's PIMC shape:
    unchanged ``(samples, nodes)`` without ``+s<K>``, else ``(K,
    nodes * samples // K)`` (total node budget preserved). Each
    modifier may appear once; ``K`` must be >= 1 and leave >= 1 node
    per world.
    """
    parts = label.split("+")
    kept = [parts[0]]
    model = "sampled"
    k: Optional[int] = None
    for mod in parts[1:]:
        if mod == BOARD_ONLY_MODIFIER:
            if model != "sampled":
                raise ValueError(f"{label!r}: '+{BOARD_ONLY_MODIFIER}' given twice")
            model = "board_only"
        elif len(mod) > 1 and mod[0] == "s" and mod[1:].isdigit():
            if k is not None:
                raise ValueError(f"{label!r}: more than one '+s<K>' modifier")
            k = int(mod[1:])
            if k < 1:
                raise ValueError(f"{label!r}: '+s<K>' needs K >= 1")
        else:
            kept.append(mod)
    if k is None:
        return "+".join(kept), model, samples, nodes
    inner = (nodes * samples) // k
    if inner < 1:
        raise ValueError(
            f"{label!r}: {k} worlds leave no node each "
            f"(total budget {nodes} x {samples})"
        )
    return "+".join(kept), model, k, inner

RESULTS_FILENAME = "results.json"
REPORT_FILENAME = "ladder.md"
DEFAULT_GAMES = 500
DEFAULT_BOOTSTRAP_RESAMPLES = 200
# One virtual draw per played pair (0.5 win each way). Keeps the MLE
# finite when one side never scores. See module docstring.
PRIOR_DRAWS_PER_PAIR = 1.0


# --------------------------------------------------------------------------
# Participants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomConfig:
    """Seat factory for the uniformly-random baseline. Mirrors the
    ``PimcConfig`` / ``NNPimcConfig`` ``build(seed)`` contract so its
    bound ``build`` pickles across the ``mp.Pool`` boundary exactly
    like the other seat factories."""

    def build(self, seed: int = 0) -> SearchStrategy:
        return RandomChoice(rng=random.Random(seed))


@dataclass(frozen=True)
class AgentSpec:
    """A resolved ladder participant: label + picklable seat factory
    + a JSON-able description for the results file."""

    label: str
    factory: Callable[[int], SearchStrategy]
    description: dict[str, Any] = field(default_factory=dict)


def resolve_agent(
    label: str,
    *,
    registry_root: Path,
    nodes: int,
    samples: int,
) -> AgentSpec:
    """Turn a participant label into a seat factory.

    * ``random`` -> :class:`RandomConfig`.
    * ``heuristic`` -> PIMC with ``HeuristicConfig()`` at the ladder's
      ``nodes`` / ``samples`` (the same fallback opponent
      ``iterate._eval_match`` uses when there's no old NN).
    * anything else -> a registered version under ``registry_root``.
      Architecture / trunk shape / action-encoding are read from the
      version's ``metadata.json`` via :class:`Registry` exactly the
      way ``_eval_match`` / ``capture`` thread them into
      ``NNPimcConfig``; the ``structured`` architecture's trunk
      hparams, value-head input and value calibration are discovered
      by ``NNPimcConfig.build`` from the sibling metadata file in the
      worker. ``<version>+nocal`` loads it without the calibration.
    """
    from mtg_bot.training import (
        HeuristicConfig, NNPimcConfig, PimcConfig, Registry,
    )

    version, apply_value_calibration = split_calibration_modifier(label)
    # ``+boardonly`` / ``+s<K>`` reshape this seat's search only.
    stripped, opponent_hidden_model, samples, nodes = split_search_modifiers(
        version, nodes=nodes, samples=samples,
    )
    if stripped != version and stripped.split("+")[0] == RANDOM_LABEL:
        raise ValueError(
            f"{label!r}: '+{BOARD_ONLY_MODIFIER}' / '+s<K>' reshape a "
            f"search; '{RANDOM_LABEL}' does not search"
        )
    version = stripped
    if not apply_value_calibration and version.split("+")[0] in (
        RANDOM_LABEL, HEURISTIC_LABEL,
    ):
        raise ValueError(
            f"{label!r}: '+{NOCAL_MODIFIER}' needs a registered version "
            f"(it switches off a network's value calibration)"
        )

    if label == RANDOM_LABEL:
        return AgentSpec(
            label=label, factory=RandomConfig().build,
            description={"kind": "random"},
        )
    # ``+nocal`` is stripped first (``version``); the remaining
    # modifiers are the head variant / leaf batch size.
    base, variant, batch = _split_label(version)
    if base == HEURISTIC_LABEL:
        if variant is not None:
            raise ValueError(f"{label!r}: head variants need a registered version")
        cfg = PimcConfig(
            samples=samples, inner_nodes=nodes,
            heuristic=HeuristicConfig(), batch_size=batch or 1,
            opponent_hidden_model=opponent_hidden_model,
        )
        return AgentSpec(
            label=label, factory=cfg.build,
            description={"kind": "heuristic_pimc", **cfg.to_dict()},
        )

    # Head-isolation variants (2026-10-01): ``<version>+valueonly`` runs
    # the net's value head under plain UCB1 (no policy priors, i.e. the
    # heuristic's search); ``<version>+heurvalue`` runs PUCT on the
    # net's policy priors with the heuristic's static value. See
    # ``mtg_bot.search.hybrid_evaluator`` for the rationale.
    version = base

    registry = Registry(registry_root)
    if not registry.has(version):
        available = ", ".join(registry.list_versions()) or "<none>"
        raise ValueError(
            f"unknown agent {label!r}: not 'random', not 'heuristic', "
            f"and not a registered version under {registry_root} "
            f"(available: {available})"
        )
    weights_path = registry.weights_path(version)
    meta = registry.load_metadata(version)
    hidden_sizes = tuple(meta.hidden_sizes) if meta.hidden_sizes else None
    policy_hidden = (
        tuple(meta.training.get("policy_hidden_sizes") or ()) or None
    )
    cfg = NNPimcConfig(
        weights_path=str(weights_path),
        samples=samples,
        inner_nodes=nodes,
        batch_size=batch or SEARCH_LEAF_BATCH_SIZE,
        with_policy=bool(meta.with_policy),
        puct=bool(meta.with_policy) and variant != "valueonly",
        hidden_sizes=hidden_sizes,
        architecture=meta.architecture,
        policy_hidden_sizes=policy_hidden,
        action_encoding_version=meta.action_encoding_version,
        apply_value_calibration=apply_value_calibration,
        opponent_hidden_model=opponent_hidden_model,
    )
    if variant == "heurvalue":
        from mtg_bot.search.hybrid_evaluator import HybridPimcConfig
        if not meta.with_policy:
            raise ValueError(
                f"{label!r}: the heurvalue variant needs a policy head, "
                f"but {version} was trained without one"
            )
        hcfg = HybridPimcConfig(
            nn=cfg, heuristic=HeuristicConfig(),
            samples=samples, inner_nodes=nodes,
        )
        return AgentSpec(
            label=label, factory=hcfg.build,
            description={
                "kind": "hybrid_heur_value_nn_policy",
                "version": version,
                "git_commit": meta.git_commit,
                "created_at": meta.created_at,
                **hcfg.to_dict(),
            },
        )
    return AgentSpec(
        label=label, factory=cfg.build,
        description={
            "kind": "nn_pimc_value_only" if variant == "valueonly" else "nn_pimc",
            "version": version,
            "git_commit": meta.git_commit,
            "created_at": meta.created_at,
            # The recorded fit; ``apply_value_calibration`` (from
            # cfg.to_dict) says whether this seat applies it.
            "value_calibration": dict(meta.value_calibration or {}),
            **cfg.to_dict(),
        },
    )


# --------------------------------------------------------------------------
# Pairings + per-pair results
# --------------------------------------------------------------------------


def parse_pairs(spec: str, agents: Sequence[str]) -> list[tuple[str, str]]:
    """``"all"`` -> round robin over ``agents`` (order preserved);
    otherwise ``"a:b,c:d"``. Every name must be in ``agents``; a pair
    can't be self-vs-self."""
    agents = list(agents)
    if spec.strip().lower() == "all":
        return list(itertools.combinations(agents, 2))
    pairs: list[tuple[str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise ValueError(
                f"bad pair {chunk!r}: expected 'a:b' (or --pairs all)"
            )
        a, b = parts[0].strip(), parts[1].strip()
        for name in (a, b):
            if name not in agents:
                raise ValueError(
                    f"pair {chunk!r} names {name!r} which is not in "
                    f"--agents {agents}"
                )
        if a == b:
            raise ValueError(f"pair {chunk!r} is self-vs-self")
        pairs.append((a, b))
    if not pairs:
        raise ValueError("--pairs produced no pairings")
    return pairs


def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


@dataclass
class PairResult:
    """Outcome of one ``a`` vs ``b`` match. ``wins_a`` / ``wins_b`` /
    ``draws`` are completed games; ``timeouts`` / ``crashed`` are
    excluded from those counts (see ``MatchStats``)."""

    a: str
    b: str
    games_requested: int
    wins_a: int
    wins_b: int
    draws: int
    timeouts: int
    crashed: int
    mean_decisions: float
    wall_seconds: float
    seed: int
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PairResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def play_pair(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    *,
    n_games: int,
    seed: int,
    n_workers: int,
    per_game_timeout_sec: Optional[float],
    deck_pool: list,
) -> PairResult:
    """Play ``n_games`` seat-alternated games of ``a`` vs ``b`` on a
    local pool. Seat 0 in the aggregated stats always means ``a``
    (``run_match_parallel(alternate_seats=True)`` remaps odd games)."""
    from mtg_bot.training.self_play import run_match_parallel

    t0 = time.perf_counter()
    stats = run_match_parallel(
        p0_factory=spec_a.factory,
        p1_factory=spec_b.factory,
        n_games=n_games,
        base_seed=seed,
        n_workers=n_workers,
        per_game_timeout_sec=per_game_timeout_sec,
        deck_pool=deck_pool,
        alternate_seats=True,
    )
    return PairResult(
        a=spec_a.label, b=spec_b.label,
        games_requested=n_games,
        wins_a=int(stats.wins.get(0, 0)),
        wins_b=int(stats.wins.get(1, 0)),
        draws=int(stats.draws),
        timeouts=int(stats.timed_out),
        crashed=int(stats.crashed),
        mean_decisions=float(stats.avg_decisions),
        wall_seconds=time.perf_counter() - t0,
        seed=seed,
        completed_at=_now_iso(),
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a sibling temp file +
    ``os.replace`` so a crash mid-write never leaves a truncated
    results file for ``--resume`` to choke on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_results(out_dir: Path) -> dict[str, Any]:
    path = out_dir / RESULTS_FILENAME
    if not path.is_file():
        return {"config": {}, "agents": {}, "pairs": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("config", {})
    data.setdefault("agents", {})
    data.setdefault("pairs", {})
    return data


def find_completed(
    results: dict[str, Any], a: str, b: str,
) -> Optional[PairResult]:
    """A pair is "done" in either orientation — the ladder is
    symmetric, so ``b:a`` from a previous run satisfies ``a:b``."""
    pairs = results.get("pairs", {})
    if pair_key(a, b) in pairs:
        return PairResult.from_dict(pairs[pair_key(a, b)])
    if pair_key(b, a) in pairs:
        return PairResult.from_dict(pairs[pair_key(b, a)])
    return None


# --------------------------------------------------------------------------
# Scoring, Elo, confidence intervals
# --------------------------------------------------------------------------


def pair_points(
    r: PairResult, *, timeouts_as_draws: bool,
) -> tuple[float, float, int]:
    """``(points_a, points_b, n_scored)`` under the chosen scoring
    mode. Draws are 0.5 each. Timeouts are dropped by default or
    0.5 each with ``timeouts_as_draws``. Crashes are always dropped."""
    pts_a = r.wins_a + 0.5 * r.draws
    pts_b = r.wins_b + 0.5 * r.draws
    n = r.wins_a + r.wins_b + r.draws
    if timeouts_as_draws:
        pts_a += 0.5 * r.timeouts
        pts_b += 0.5 * r.timeouts
        n += r.timeouts
    return pts_a, pts_b, n


def wilson_interval(
    successes: float, n: int, z: float = 1.959964,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns
    ``(0.0, 1.0)`` for ``n == 0`` (no information)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fit_elo(
    agents: Sequence[str],
    pair_scores: dict[tuple[str, str], tuple[float, float, int]],
    *,
    anchor: str,
    prior_draws: float = PRIOR_DRAWS_PER_PAIR,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> dict[str, Optional[float]]:
    """Bradley-Terry maximum likelihood via Hunter's MM iteration.

    ``pair_scores[(a, b)] = (points_a, points_b, n)`` — points are
    wins + half-draws (see :func:`pair_points`). Returns an Elo per
    agent (``400 * log10(gamma)``) shifted so ``anchor`` sits at 0.
    Agents with no scored games get ``None``. If the pairing graph
    is disconnected, agents outside the anchor's component are
    still fitted but their scale is arbitrary relative to the
    anchor — :func:`connected_components` lets the report say so.
    """
    idx = {name: i for i, name in enumerate(agents)}
    k = len(agents)
    n_mat = np.zeros((k, k))
    w = np.zeros(k)
    for (a, b), (pa, pb, n) in pair_scores.items():
        if n <= 0:
            continue
        i, j = idx[a], idx[b]
        n_tot = n + prior_draws
        n_mat[i, j] += n_tot
        n_mat[j, i] += n_tot
        w[i] += pa + 0.5 * prior_draws
        w[j] += pb + 0.5 * prior_draws
    played = n_mat.sum(axis=1) > 0
    gamma = np.ones(k)
    for _ in range(max_iter):
        denom = np.zeros(k)
        for i in range(k):
            if not played[i]:
                continue
            js = np.nonzero(n_mat[i])[0]
            denom[i] = np.sum(n_mat[i, js] / (gamma[i] + gamma[js]))
        new = gamma.copy()
        mask = played & (denom > 0)
        new[mask] = w[mask] / denom[mask]
        # Fix the scale each step (BT is invariant to a global
        # multiplier); normalizing to the geometric mean keeps the
        # iteration from drifting.
        gm = np.exp(np.mean(np.log(new[played]))) if played.any() else 1.0
        new[played] /= gm
        delta = np.max(np.abs(new[played] - gamma[played])) if played.any() else 0.0
        gamma = new
        if delta < tol:
            break
    elo = 400.0 * np.log10(gamma)
    anchor_i = idx.get(anchor)
    if anchor_i is not None and played[anchor_i]:
        elo = elo - elo[anchor_i]
    elif played.any():
        # Anchor absent or unplayed: anchor the first played agent.
        first = int(np.argmax(played))
        elo = elo - elo[first]
    return {
        name: (float(elo[i]) if played[i] else None)
        for name, i in idx.items()
    }


def connected_components(
    agents: Sequence[str],
    pair_scores: dict[tuple[str, str], tuple[float, float, int]],
) -> list[set[str]]:
    """Components of the "played at least one scored game" graph."""
    adj: dict[str, set[str]] = {a: set() for a in agents}
    for (a, b), (_, _, n) in pair_scores.items():
        if n > 0:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    comps: list[set[str]] = []
    for a in agents:
        if a in seen:
            continue
        stack, comp = [a], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def bootstrap_elo_ci(
    agents: Sequence[str],
    results: Sequence[PairResult],
    *,
    anchor: str,
    timeouts_as_draws: bool,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Optional[tuple[float, float]]]:
    """Percentile bootstrap of the Elo fit: within each pair, resample
    the scored games (win-a / win-b / draw multinomial) with
    replacement, refit, take the ``alpha/2`` and ``1-alpha/2``
    quantiles per agent. Agents with no games get ``None``."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {a: [] for a in agents}
    for _ in range(n_resamples):
        scores: dict[tuple[str, str], tuple[float, float, int]] = {}
        for r in results:
            counts = [r.wins_a, r.wins_b, r.draws]
            if timeouts_as_draws:
                counts[2] += r.timeouts
            n = sum(counts)
            if n <= 0:
                continue
            probs = np.asarray(counts, dtype=float) / n
            wa, wb, d = rng.multinomial(n, probs)
            scores[(r.a, r.b)] = (wa + 0.5 * d, wb + 0.5 * d, n)
        fit = fit_elo(agents, scores, anchor=anchor)
        for a in agents:
            if fit[a] is not None:
                samples[a].append(fit[a])
    out: dict[str, Optional[tuple[float, float]]] = {}
    for a in agents:
        if not samples[a]:
            out[a] = None
            continue
        lo, hi = np.percentile(
            samples[a], [100 * alpha / 2, 100 * (1 - alpha / 2)],
        )
        out[a] = (float(lo), float(hi))
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt_ci(ci: Optional[tuple[float, float]]) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:+.0f}, {ci[1]:+.0f}]"


def build_report(
    agents: Sequence[str],
    results: Sequence[PairResult],
    config: dict[str, Any],
    *,
    timeouts_as_draws: bool,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = 0,
) -> str:
    """Render ``ladder.md``: Elo table, pairwise matrix, config."""
    agents = list(agents)
    anchor = RANDOM_LABEL if RANDOM_LABEL in agents else (agents[0] if agents else "")
    scores = {
        (r.a, r.b): pair_points(r, timeouts_as_draws=timeouts_as_draws)
        for r in results
    }
    elo = fit_elo(agents, scores, anchor=anchor)
    ci = bootstrap_elo_ci(
        agents, results, anchor=anchor,
        timeouts_as_draws=timeouts_as_draws,
        n_resamples=bootstrap_resamples, seed=bootstrap_seed,
    )
    comps = connected_components(agents, scores)

    # Per-agent totals under the scoring mode.
    points: dict[str, float] = {a: 0.0 for a in agents}
    games: dict[str, int] = {a: 0 for a in agents}
    for (a, b), (pa, pb, n) in scores.items():
        points[a] += pa
        points[b] += pb
        games[a] += n
        games[b] += n

    lines: list[str] = []
    lines.append("# Strength ladder")
    lines.append("")
    mode = (
        "timeouts scored as draws (0.5/0.5); crashes excluded"
        if timeouts_as_draws
        else "timeouts and crashes excluded from the denominator (gate convention)"
    )
    lines.append(f"Scoring: {mode}.")
    lines.append(
        f"Elo: Bradley-Terry MLE, anchor `{anchor}` = 0, "
        f"bootstrap 95% CI over {bootstrap_resamples} resamples."
    )
    if len([c for c in comps if any(games[a] > 0 for a in c)]) > 1:
        lines.append("")
        lines.append(
            "**WARNING: the pairing graph is disconnected — Elo values "
            "in different components are NOT comparable.** Components: "
            + "; ".join(
                "{" + ", ".join(sorted(c)) + "}" for c in comps
            )
        )
    lines.append("")
    lines.append("## Elo")
    lines.append("")
    lines.append("| rank | agent | Elo | 95% CI | games | score |")
    lines.append("|---:|---|---:|---|---:|---:|")
    ranked = sorted(
        agents,
        key=lambda a: (elo[a] is None, -(elo[a] or 0.0)),
    )
    for rank, a in enumerate(ranked, start=1):
        e = elo[a]
        e_str = f"{e:+.0f}" if e is not None else "n/a"
        score = f"{points[a] / games[a]:.1%}" if games[a] else "n/a"
        lines.append(
            f"| {rank} | {a} | {e_str} | {_fmt_ci(ci[a])} | "
            f"{games[a]} | {score} |"
        )

    lines.append("")
    lines.append("## Pairwise results")
    lines.append("")
    lines.append(
        "Each cell: row's win rate vs column with Wilson 95% CI, then "
        "`W-L-D` from the row's perspective (`t`=timeouts, `c`=crashed, "
        "both excluded from W-L-D)."
    )
    lines.append("")
    header = "| | " + " | ".join(agents) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(agents))
    by_key: dict[tuple[str, str], PairResult] = {}
    for r in results:
        by_key[(r.a, r.b)] = r
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append("—")
                continue
            r = by_key.get((a, b))
            flipped = False
            if r is None:
                r = by_key.get((b, a))
                flipped = True
            if r is None:
                cells.append("")
                continue
            wa, wb = (r.wins_b, r.wins_a) if flipped else (r.wins_a, r.wins_b)
            pa, _, n = pair_points(
                PairResult(**{**r.to_dict(), "wins_a": wa, "wins_b": wb}),
                timeouts_as_draws=timeouts_as_draws,
            )
            lo, hi = wilson_interval(pa, n)
            rate = f"{pa / n:.1%} [{lo:.0%}, {hi:.0%}]" if n else "n/a"
            extra = ""
            if r.timeouts:
                extra += f" t{r.timeouts}"
            if r.crashed:
                extra += f" c{r.crashed}"
            cells.append(f"{rate}<br>{wa}-{wb}-{r.draws}{extra}")
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-pair detail")
    lines.append("")
    lines.append(
        "| pair | requested | wins_a | wins_b | draws | timeouts | "
        "crashed | mean decisions | wall (s) | seed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.a} vs {r.b} | {r.games_requested} | {r.wins_a} | "
            f"{r.wins_b} | {r.draws} | {r.timeouts} | {r.crashed} | "
            f"{r.mean_decisions:.0f} | {r.wall_seconds:.0f} | {r.seed} |"
        )

    lines.append("")
    lines.append("## Config")
    lines.append("")
    for k in sorted(config):
        lines.append(f"- `{k}`: `{config[k]}`")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mtg_bot ladder",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[depth_parser(), runtime_parser()],
    )
    p.add_argument(
        "--agents", nargs="+", default=None,
        help=(
            "Participants: 'random', 'heuristic', and/or registered "
            "version labels (e.g. v5 v13s). Required unless --resume "
            "(then defaults to the resumed run's agents). Search "
            "modifiers after '+': b<N> (leaf batch), nocal, valueonly / "
            "heurvalue, boardonly (don't imagine the opponent's hidden "
            "cards), s<K> (K worlds sharing nodes x samples), e.g. "
            "v30+b1+boardonly+s1."
        ),
    )
    p.add_argument(
        "--pairs", default="all",
        help="'all' (round robin, default) or explicit 'a:b,c:d'.",
    )
    p.add_argument(
        "--games", type=int, default=DEFAULT_GAMES,
        help=f"Games per pair, seat-alternated (default {DEFAULT_GAMES}).",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Pool size (default cpu_count - 2).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output dir (default ladder_runs/<UTC timestamp>/).",
    )
    p.add_argument(
        "--resume", type=Path, default=None,
        help=(
            "Existing ladder run dir; pairs already in its results.json "
            "are skipped and the report is rebuilt over everything."
        ),
    )
    p.add_argument(
        "--registry-root", type=Path, default=Path("trained_weights"),
        help="Model registry root (default trained_weights).",
    )
    p.add_argument(
        "--timeouts-as-draws", action="store_true",
        help=(
            "Score each timed-out game as 0.5/0.5 instead of dropping "
            "it. A wall-clock timeout can't be attributed to either "
            "player, so there is deliberately no 'as losses' mode. "
            "Crashed games are always excluded."
        ),
    )
    p.add_argument(
        "--bootstrap-resamples", type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"Bootstrap resamples for the Elo CI (default {DEFAULT_BOOTSTRAP_RESAMPLES}).",
    )
    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit_short() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _common.resolve_per_game_timeout(args)

    if args.resume is not None:
        out_dir = args.resume
        if not (out_dir / RESULTS_FILENAME).is_file():
            parser.error(f"--resume {out_dir}: no {RESULTS_FILENAME} there")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.out or Path("ladder_runs") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(out_dir)
    agents: list[str] = list(args.agents or [])
    if not agents:
        agents = list(results["config"].get("agents") or [])
    if not agents:
        parser.error("--agents is required (or --resume a dir that has them)")
    if len(set(agents)) != len(agents):
        parser.error(f"--agents has duplicates: {agents}")
    try:
        pairs = parse_pairs(args.pairs, agents)
    except ValueError as e:
        parser.error(str(e))

    n_workers = (
        args.workers if args.workers is not None
        else max(1, (os.cpu_count() or 1) - 2)
    )
    if n_workers < 1:
        parser.error("--workers must be >= 1")

    config = {
        "agents": agents,
        "pairs": args.pairs,
        "games": args.games,
        "nodes": args.nodes,
        "samples": args.samples,
        "seed": args.seed,
        "workers": n_workers,
        "per_game_timeout_sec": args.per_game_timeout_sec,
        "deck_pool": str(args.deck_pool),
        "registry_root": str(args.registry_root),
        "timeouts_as_draws": bool(args.timeouts_as_draws),
        "git_commit": _git_commit_short(),
        "started_at": results["config"].get("started_at") or _now_iso(),
    }
    results["config"] = config

    # Resolve every participant up front so a typo fails before any
    # game is played (and before the deck pool load).
    specs: dict[str, AgentSpec] = {}
    for label in agents:
        try:
            specs[label] = resolve_agent(
                label, registry_root=args.registry_root,
                nodes=args.nodes, samples=args.samples,
            )
        except Exception as e:  # noqa: BLE001 — surface, don't mask
            sys.exit(f"[ladder] cannot load agent {label!r}: {e}")
        results["agents"][label] = specs[label].description
        print(f"[ladder] loaded {label}: {specs[label].description.get('kind')}")

    todo = [(a, b) for a, b in pairs if find_completed(results, a, b) is None]
    print(
        f"[ladder] {len(pairs)} pairs, {len(pairs) - len(todo)} already "
        f"done, {len(todo)} to play; games/pair={args.games}, "
        f"nodes={args.nodes}, samples={args.samples}, workers={n_workers}, "
        f"per_game_timeout={args.per_game_timeout_sec}s, out={out_dir}"
    )
    atomic_write_json(out_dir / RESULTS_FILENAME, results)

    if todo:
        if not args.deck_pool.is_file():
            sys.exit(f"[ladder] --deck-pool {args.deck_pool} not found")
        from mtg_bot.decks import decks_from_json
        deck_pool = decks_from_json(args.deck_pool)
        if not deck_pool:
            sys.exit(f"[ladder] --deck-pool {args.deck_pool} loaded 0 decks")

        for a, b in todo:
            print(f"[ladder] playing {a} vs {b} ({args.games} games) ...", flush=True)
            r = play_pair(
                specs[a], specs[b],
                n_games=args.games, seed=args.seed, n_workers=n_workers,
                per_game_timeout_sec=args.per_game_timeout_sec,
                deck_pool=deck_pool,
            )
            results["pairs"][pair_key(a, b)] = r.to_dict()
            atomic_write_json(out_dir / RESULTS_FILENAME, results)
            print(
                f"[ladder] {a} vs {b}: {r.wins_a}-{r.wins_b}-{r.draws} "
                f"(timeouts={r.timeouts}, crashed={r.crashed}, "
                f"mean_decisions={r.mean_decisions:.0f}, "
                f"wall={r.wall_seconds:.0f}s)", flush=True,
            )

    all_results = [
        PairResult.from_dict(d) for d in results["pairs"].values()
    ]
    # Only report agents that are part of this ladder (a resumed dir
    # may hold pairs for agents dropped from --agents).
    reported = [r for r in all_results if r.a in agents and r.b in agents]
    report = build_report(
        agents, reported, config,
        timeouts_as_draws=bool(args.timeouts_as_draws),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.seed,
    )
    (out_dir / REPORT_FILENAME).write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"[ladder] wrote {out_dir / REPORT_FILENAME}")


if __name__ == "__main__":
    main()
