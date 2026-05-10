"""Tests for ``mtg_bot.training.replay_capture`` and the
``ReplayCaptureConfig`` integration with data_gen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mtg_bot.replay import load_replay
from mtg_bot.training import (
    PimcConfig, ReplayCaptureConfig, ReplayRecorder,
    generate_training_data, generate_training_data_parallel,
    heuristic_eval_annotator, random_choice_factory,
)
from tests._test_decks import simple_test_decklists, simple_test_deck_pool


# ---------- ReplayRecorder ---------------------------------------------------


def test_recorder_with_no_annotators_records_raw_snapshots():
    """A bare recorder accumulates serialize_state output unchanged."""
    from mtg_bot.simulate import setup_game
    state = setup_game(
        agents=[__import__("mtg_bot.agents", fromlist=["RandomAgent"])
                .RandomAgent(),
                __import__("mtg_bot.agents", fromlist=["RandomAgent"])
                .RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    rec = ReplayRecorder()
    rec.record(state)
    assert len(rec.snapshots) == 1
    snap = rec.snapshots[0]
    # Should have the standard serialize_state keys.
    assert "turn_number" in snap
    assert "players" in snap
    # No annotator → no eval keys.
    assert "heuristic_eval" not in snap
    assert "nn_eval" not in snap
    # Recorded with decision=None (default) → key present, value None.
    assert snap["decision"] is None


def test_recorder_stores_decision_when_provided():
    """Passing ``decision=`` threads through to the snapshot's
    ``decision`` JSON field."""
    from mtg_bot.actions import PassPriority
    from mtg_bot.agents import RandomAgent
    from mtg_bot.simulate import setup_game
    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    rec = ReplayRecorder()
    rec.record(state, decision=PassPriority())
    snap = rec.snapshots[0]
    assert snap["decision"] == {"kind": "PassPriority"}


def test_heuristic_annotator_attaches_eval_per_snapshot():
    from mtg_bot.simulate import setup_game
    from mtg_bot.agents import RandomAgent
    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    rec = ReplayRecorder(annotators=[heuristic_eval_annotator()])
    rec.record(state)
    snap = rec.snapshots[0]
    assert "heuristic_eval" in snap
    h = snap["heuristic_eval"]
    assert "p0" in h and "p1" in h
    assert -1.0 <= h["p0"] <= 1.0
    assert -1.0 <= h["p1"] <= 1.0


def test_to_replay_packages_metadata_and_snapshots():
    rec = ReplayRecorder()
    # Fake a snapshot directly for unit-test simplicity.
    rec.snapshots.append({"turn_number": 1, "step": "upkeep"})
    replay = rec.to_replay(metadata={"game_idx": 7, "winner": 0})
    assert replay.metadata == {"game_idx": 7, "winner": 0}
    assert replay.snapshots == [{"turn_number": 1, "step": "upkeep"}]


# ---------- record_search_annotation -----------------------------------------


def test_record_search_annotation_attaches_to_snapshot():
    """``record_search_annotation`` writes the annotation to an
    already-recorded snapshot's ``search`` key. ``record`` initializes
    the field as None so the annotator hook is purely additive."""
    from mtg_bot.actions import PassPriority
    from mtg_bot.agents import RandomAgent
    from mtg_bot.simulate import setup_game
    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    rec = ReplayRecorder()
    rec.record(state)              # snapshot 0
    rec.record(state, decision=PassPriority())  # snapshot 1
    # Both initialized with search=None.
    assert rec.snapshots[0]["search"] is None
    assert rec.snapshots[1]["search"] is None
    annotation = {
        "chosen": {"kind": "PassPriority"},
        "candidates": [{
            "action": {"kind": "PassPriority"}, "visits": 5,
            "q": 0.1, "prior": None, "leaf_value": None,
        }],
        "samples": 1, "inner_nodes": 50, "elapsed_ms": 12,
    }
    rec.record_search_annotation(1, annotation)
    assert rec.snapshots[0]["search"] is None
    assert rec.snapshots[1]["search"] == annotation


def test_record_search_annotation_rejects_out_of_range():
    """Out-of-range snapshot indices raise IndexError, not silent
    no-ops (off-by-one bugs in callers should fail loud)."""
    rec = ReplayRecorder()
    # Empty: any index is out of range.
    with pytest.raises(IndexError):
        rec.record_search_annotation(0, {"chosen": None, "candidates": []})
    # Add one snapshot, then test ±1 boundaries.
    rec.snapshots.append({"turn_number": 1})
    with pytest.raises(IndexError):
        rec.record_search_annotation(-1, {"chosen": None, "candidates": []})
    with pytest.raises(IndexError):
        rec.record_search_annotation(1, {"chosen": None, "candidates": []})


def test_build_search_annotation_serializes_actions_and_sorts():
    """``build_search_annotation`` runs each candidate through
    ``decision_to_json``, sorts by visits desc, and surfaces the
    strategy's PIMC ``samples`` + inner MCTS ``budget_factory``
    node count."""
    import random
    from mtg_bot.actions import PassPriority, PlayLand
    from mtg_bot.search import (
        Budget, MCTS, RandomRolloutEvaluator,
    )
    from mtg_bot.search.pimc import PIMC
    from mtg_bot.training import build_search_annotation

    inner_factory = lambda: MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=137),
        rng=random.Random(0),
    )
    pimc = PIMC(inner_factory=inner_factory, samples=4)
    candidates = [
        {"decision": PlayLand(card_id=5), "visits": 3,
         "q": -0.1, "prior": None, "leaf_value": None},
        {"decision": PassPriority(), "visits": 10,
         "q": 0.42, "prior": 0.5, "leaf_value": 0.3},
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(),
        candidates=candidates,
        strategy=pimc,
        elapsed_ms=99,
    )
    assert annotation["chosen"] == {"kind": "PassPriority"}
    # Sorted by visits desc — PassPriority(10) comes before PlayLand(3).
    assert annotation["candidates"][0]["action"] == {"kind": "PassPriority"}
    assert annotation["candidates"][1]["action"] == {
        "kind": "PlayLand", "card_id": 5,
    }
    assert annotation["candidates"][0]["visits"] == 10
    assert annotation["candidates"][0]["q"] == 0.42
    assert annotation["samples"] == 4
    assert annotation["inner_nodes"] == 137
    assert annotation["elapsed_ms"] == 99


def test_build_search_annotation_handles_raw_mcts():
    """When the strategy is a raw MCTS (no PIMC wrapper), ``samples``
    comes back as 1 (default attr) and ``inner_nodes`` reads from the
    strategy's own budget_factory."""
    import random
    from mtg_bot.actions import PassPriority
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.training import build_search_annotation

    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=64),
        rng=random.Random(0),
    )
    annotation = build_search_annotation(
        chosen=PassPriority(),
        candidates=[
            {"decision": PassPriority(), "visits": 2,
             "q": 0.0, "prior": None, "leaf_value": None},
        ],
        strategy=mcts,
        elapsed_ms=None,
    )
    # MCTS doesn't carry a ``samples`` attribute; helper falls back to 1.
    assert annotation["samples"] == 1
    assert annotation["inner_nodes"] == 64
    assert annotation["elapsed_ms"] is None


def test_build_search_annotation_serialized_size_under_budget():
    """The capture-storage budget assumes a per-decision search
    annotation serializes to ~1 KB raw / ~300 B gzipped. Verify a
    representative annotation (5 candidates, all numeric fields
    present) lands under that ceiling — guards against an
    accidental shape blow-up that would torpedo the per-iter
    replay-dir size budget."""
    import gzip
    import json as _json
    import random
    from mtg_bot.actions import (
        ActivateAbility, CastSpell, PassPriority, PlayLand,
    )
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.pimc import PIMC
    from mtg_bot.training import build_search_annotation

    pimc = PIMC(
        inner_factory=lambda: MCTS(
            evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
            budget_factory=lambda: Budget(max_nodes=200),
            rng=random.Random(0),
        ),
        samples=8,
    )
    decs = [
        PassPriority(),
        PlayLand(card_id=42),
        CastSpell(card_id=7, kicked=True),
        ActivateAbility(source_id=11, ability_index=0),
        CastSpell(card_id=8),
    ]
    candidates = [
        {"decision": d, "visits": 50 - i * 9,
         "q": 0.123 + i * 0.01,
         "prior": 0.2 - i * 0.04,
         "leaf_value": 0.05 + i * 0.02}
        for i, d in enumerate(decs)
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(), candidates=candidates,
        strategy=pimc, elapsed_ms=42,
    )
    raw = _json.dumps(annotation).encode("utf-8")
    gz = gzip.compress(raw)
    # Spec target: ~1 KB raw, ~300 B gzipped. Generous headroom.
    assert len(raw) < 2048, (
        f"raw annotation too big: {len(raw)} bytes; "
        f"budget assumed ~1 KB"
    )
    assert len(gz) < 600, (
        f"gzipped annotation too big: {len(gz)} bytes; "
        f"budget assumed ~300 B"
    )


# ---------- ReplayCaptureConfig ---------------------------------------------


def test_capture_config_should_capture_respects_modulus():
    cfg = ReplayCaptureConfig(save_dir="x", capture_every=3)
    assert cfg.should_capture(0)
    assert not cfg.should_capture(1)
    assert not cfg.should_capture(2)
    assert cfg.should_capture(3)
    assert cfg.should_capture(6)


def test_capture_config_build_recorder_passes_arch_to_nn_annotator(
    tmp_path: Path,
):
    """ReplayCaptureConfig.build_recorder() must propagate hidden_sizes
    and with_policy to nn_eval_annotator → NNEvaluator. Without that,
    a config tagged with a modern (dual-head, big-trunk) checkpoint
    blows up at load time on a state_dict mismatch — which broke the
    first iterate run with deck_pool + replay capture in production."""
    import torch

    from mtg_bot.search.nn_evaluator import _ValuePolicyMLP

    hidden_sizes = (32, 24, 16)
    weights_path = tmp_path / "tiny_dual_head.pt"
    model = _ValuePolicyMLP(hidden_sizes=hidden_sizes)
    torch.save(model.state_dict(), weights_path)

    cfg = ReplayCaptureConfig(
        save_dir=str(tmp_path),
        nn_weights_path=str(weights_path),
        nn_version_label="tiny",
        nn_hidden_sizes=hidden_sizes,
        nn_with_policy=True,
    )
    # build_recorder() loads the NN — would raise RuntimeError if
    # the architecture kwargs weren't threaded through.
    recorder = cfg.build_recorder()
    # Two annotators: the always-on heuristic plus the NN one.
    assert len(recorder.annotators) == 2


def test_nn_eval_annotator_annotates_running_and_terminal_states(
    tmp_path: Path,
):
    """nn_eval_annotator's inner ``annotate`` returns ``nn_eval`` with
    p0/p1 floats + version on a running state, and the terminal value
    fast-path on a finished state. Pins both branches."""
    import torch

    from mtg_bot.search.nn_evaluator import _ValuePolicyMLP
    from mtg_bot.simulate import setup_game
    from mtg_bot.training.replay_capture import nn_eval_annotator

    hidden_sizes = (16, 16)
    weights_path = tmp_path / "tiny.pt"
    torch.save(
        _ValuePolicyMLP(hidden_sizes=hidden_sizes).state_dict(),
        weights_path,
    )
    annotate = nn_eval_annotator(
        weights_path,
        version_label="tiny",
        hidden_sizes=hidden_sizes,
        with_policy=True,
    )

    # Build a real running engine state via setup_game so the
    # evaluator gets a well-formed input. RandomAgent isn't needed —
    # the annotator only inspects state, never asks for decisions.
    from mtg_bot.agents import RandomAgent
    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(),
        seed=0,
    )
    out = annotate(state)
    assert "nn_eval" in out
    assert out["nn_eval"]["version"] == "tiny"
    assert isinstance(out["nn_eval"]["p0"], float)
    assert isinstance(out["nn_eval"]["p1"], float)

    # Terminal-state branch: hand-set the engine to "p0 wins" and
    # confirm the annotator reads off the engine state directly
    # (skipping the NN forward).
    state.winner_id = 0
    state.is_draw = False
    out_terminal = annotate(state)
    assert out_terminal["nn_eval"]["p0"] == 1.0
    assert out_terminal["nn_eval"]["p1"] == -1.0
    assert out_terminal["nn_eval"]["version"] == "tiny"


def test_capture_config_replay_path_is_deterministic_in_inputs(tmp_path: Path):
    cfg = ReplayCaptureConfig(
        save_dir=str(tmp_path), capture_every=1, filename_prefix="exp1",
    )
    p = cfg.replay_path(base_seed=42, game_idx=3)
    assert p.parent == tmp_path
    assert "exp1" in p.name
    assert "seed42" in p.name
    assert "idx0003" in p.name
    assert p.suffixes == [".json", ".gz"]


# ---------- end-to-end with sequential data_gen -----------------------------


def test_generate_training_data_writes_replays_when_configured(tmp_path: Path):
    """Run the sequential data gen with replay capture enabled;
    verify .json.gz files appear and can be loaded."""
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    features, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    # Training data still produced as before.
    assert features.shape[0] == outcomes.shape[0]
    # One replay file per game.
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 2
    # Each file is a valid Replay.
    for f in files:
        replay = load_replay(f)
        assert len(replay.snapshots) > 0
        # Each snapshot has heuristic_eval (default annotator).
        assert all("heuristic_eval" in s for s in replay.snapshots)
        # Metadata captured.
        assert "winner" in replay.metadata
        assert "n_decisions" in replay.metadata
        assert replay.metadata["base_seed"] == 42


def test_generate_training_data_capture_every_thins_files(tmp_path: Path):
    """With capture_every=2, only games at idx 0, 2 produce files."""
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=2)
    generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4, base_seed=0, replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    # Games 0 and 2 captured.
    assert len(files) == 2
    names = [f.name for f in files]
    assert any("idx0000" in n for n in names)
    assert any("idx0002" in n for n in names)


def test_generate_training_data_no_replay_config_writes_no_files(tmp_path: Path):
    """When replay_config is None (default), no files appear even if
    we were to inspect a directory after the run."""
    save_dir = tmp_path / "replays"
    save_dir.mkdir()
    generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=0, deck_pool=simple_test_deck_pool(),
    )
    assert list(save_dir.glob("*.json.gz")) == []


# ---------- parallel data_gen with replay capture ---------------------------


def test_generate_training_data_parallel_writes_replays(tmp_path: Path):
    """Workers each save their own game's replay file. Same per-game
    output as sequential because per-game seeds are deterministic."""
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    features, _ = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3, base_seed=7, n_workers=2, replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    assert features.shape[0] > 0
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 3


def test_parallel_replay_files_match_sequential(tmp_path: Path):
    """Stronger determinism check: replay files written by the
    parallel path should match those written by the sequential path
    for the same base_seed (per-game payload identical, file names
    identical)."""
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"
    seq_cfg = ReplayCaptureConfig(save_dir=str(seq_dir), capture_every=1)
    par_cfg = ReplayCaptureConfig(save_dir=str(par_dir), capture_every=1)
    generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=99, replay_config=seq_cfg, deck_pool=simple_test_deck_pool(),
    )
    generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=99, n_workers=2, replay_config=par_cfg, deck_pool=simple_test_deck_pool(),
    )
    seq_files = sorted(p.name for p in seq_dir.glob("*.json.gz"))
    par_files = sorted(p.name for p in par_dir.glob("*.json.gz"))
    assert seq_files == par_files
    # And the file CONTENTS should be identical (per-game replays
    # are deterministic in seed).
    for name in seq_files:
        seq_replay = load_replay(seq_dir / name)
        par_replay = load_replay(par_dir / name)
        assert seq_replay.snapshots == par_replay.snapshots
        # Metadata may differ in `captured_at` (timestamp), so compare
        # the deterministic fields specifically.
        for key in ("game_idx", "base_seed", "engine_seed", "winner",
                    "n_decisions", "annotated_with_nn_version"):
            assert seq_replay.metadata[key] == par_replay.metadata[key]


# ---------- decision capture (Phase 6 prereq) -------------------------------


def test_v2_replay_round_trips_decisions(tmp_path: Path):
    """Build a recorder, hand it several representative decisions,
    save the replay, load it back. Each snapshot's ``decision``
    deserializes to the same Decision instance."""
    from mtg_bot.actions import (
        ActivateAbility, CastSpell, DeclareAttackers, PassPriority, PlayLand,
    )
    from mtg_bot.agents import RandomAgent
    from mtg_bot.replay import REPLAY_FORMAT_VERSION, save_replay
    from mtg_bot.search.combat_decisions import AttackerDecision, BlockerDecision
    from mtg_bot.search.resolution_decisions import ResolutionTargetDecision
    from mtg_bot.simulate import setup_game
    from mtg_bot.training.replay_decisions import decision_from_json

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    rec = ReplayRecorder()
    expected = [
        None,  # initial pre-game state
        PassPriority(),
        PlayLand(card_id=42),
        CastSpell(card_id=7, kicked=True),
        ActivateAbility(source_id=11, ability_index=0),
        DeclareAttackers(attacker_ids=frozenset({3, 5})),
        AttackerDecision(attacker_id=3, attack=True),
        BlockerDecision(blocker_id=8, attacker_id=None),
        ResolutionTargetDecision(target=("creature", 7)),
    ]
    for d in expected:
        rec.record(state, decision=d)
    replay = rec.to_replay(metadata={"game_idx": 1})
    assert replay.format_version == REPLAY_FORMAT_VERSION

    path = tmp_path / "v2.json.gz"
    save_replay(replay, path)
    loaded = load_replay(path)
    assert len(loaded.snapshots) == len(expected)
    for snap, original in zip(loaded.snapshots, expected):
        if original is None:
            assert snap["decision"] is None
        else:
            assert decision_from_json(snap["decision"]) == original


def test_end_to_end_pimc_replay_capture_attaches_search_annotations(
    tmp_path: Path,
):
    """Run a sequential data_gen with PIMC on both sides and replay
    capture enabled — at least one captured snapshot should carry a
    populated ``search`` annotation with non-empty candidates."""
    from mtg_bot.training import PimcConfig
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 1
    replay = load_replay(files[0])
    assert replay.format_version == 3
    annotated = [
        s for s in replay.snapshots if s.get("search") is not None
    ]
    assert annotated, "expected at least one search-annotated snapshot"
    # First annotated snapshot should have a chosen action and at
    # least one candidate.
    sample = annotated[0]["search"]
    assert sample["chosen"] is not None
    assert isinstance(sample["candidates"], list)
    assert len(sample["candidates"]) >= 1
    # Per-candidate fields are present (None or numeric — type is
    # the contract).
    c0 = sample["candidates"][0]
    assert "action" in c0
    assert isinstance(c0["visits"], int)
    assert "q" in c0 and "prior" in c0 and "leaf_value" in c0
    # PIMC-of-MCTS budget threaded through.
    assert sample["samples"] == 1
    assert sample["inner_nodes"] == 4


def test_search_annotation_lands_on_pre_apply_snapshot(tmp_path: Path):
    """Unit-level pin of the data_gen call-site fix: with PIMC enabled
    and capture_every=1, the FIRST search annotation produced by the
    very first priority-action iteration must land on
    ``snapshots[0]`` (the recorded initial state), NOT
    ``snapshots[1]`` (the post-apply state of that first decision).

    Pre-fix bug: the call site passed ``len(snapshots) - 1`` so the
    annotation always landed on the snap that was just recorded
    (post-apply). Post-fix: it passes ``len(snapshots) - 2`` so the
    annotation lands on the snap whose state was the search root
    (the one recorded BEFORE the apply that just happened).
    """
    from mtg_bot.training import PimcConfig
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    replay = load_replay(files[0])
    # Initial snapshot must carry an annotation now (the first
    # priority-action search ran on its state).
    assert replay.snapshots[0].get("search") is not None, (
        "expected snap[0] (initial pre-game state) to carry the "
        "annotation from the first priority-action search; with the "
        "fix, every priority-action search annotation lands on its "
        "pre-apply snapshot"
    )


def test_search_annotation_aligns_with_pre_apply_snapshot_state(
    tmp_path: Path,
):
    """Regression: the search annotation at ``snapshots[i]`` describes
    the search that ran with ``snapshots[i].state`` as the root.
    Equivalently, ``snapshots[i].search.chosen`` must equal
    ``snapshots[i+1].decision`` for every annotated snap.

    This was bug #LDF: data_gen / self_play attached the annotation to
    the POST-apply snapshot (``len(snapshots) - 1``), one snapshot too
    late. The viewer renders the annotation alongside the snapshot's
    state, so candidates appeared to be illegal in the displayed state
    (e.g. a sorcery-window search with PlayLand candidates would land
    on the next snap whose state had already advanced to opp's instant-
    timing window with only PassPriority castable). The fix attaches
    to ``len(snapshots) - 2`` instead so every snap.search aligns with
    snap.state.

    Equivalent invariants (any one breaking → bug):
    * snap[i].search.chosen == snap[i+1].decision (the chosen action
      WAS what got applied on the way to the next snapshot).
    * snap[i].search exists only when snap[i+1] exists (a search must
      have produced a real follow-up snapshot).
    """
    from mtg_bot.training import PimcConfig
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 1
    replay = load_replay(files[0])

    # Find every annotated snap and verify the alignment invariant.
    n_aligned_checks = 0
    for i, snap in enumerate(replay.snapshots):
        s = snap.get("search")
        if s is None:
            continue
        # An annotated snap must have a successor (the apply happened
        # to produce it).
        assert i + 1 < len(replay.snapshots), (
            f"snap[{i}] has search annotation but no follow-up snapshot "
            f"(would mean an apply was recorded with no resulting "
            f"post-apply state)"
        )
        chosen = s.get("chosen")
        applied = replay.snapshots[i + 1].get("decision")
        assert chosen == applied, (
            f"snap[{i}].search.chosen={chosen!r} but "
            f"snap[{i+1}].decision={applied!r} — annotation/state "
            f"misalignment (the search ran on a different root than "
            f"what this snap shows)"
        )
        n_aligned_checks += 1
    # Sanity: at least a handful of priority-action snaps should have
    # been annotated; otherwise the test would silently pass with
    # zero checks.
    assert n_aligned_checks >= 5, (
        f"only {n_aligned_checks} annotated snaps found; expected "
        f"at least 5 for a single game"
    )

    # Also: the LAST snapshot (terminal or budget-cap) should NOT carry
    # a search annotation — there's no follow-up apply to align it
    # with. (Initial snap[0] CAN now carry one — the fix moves the
    # first decision's annotation onto it.)
    assert replay.snapshots[-1].get("search") is None, (
        "last snapshot carries a search annotation — there's no "
        "follow-up snapshot to align it with"
    )


def test_end_to_end_replay_capture_writes_v3_with_search_field(
    tmp_path: Path,
):
    """Run a sequential data_gen with replay capture enabled and
    verify the resulting replay files are v3 and every snapshot
    carries a ``search`` field (None for the snapshots that
    couldn't be annotated — random play has no search tree, so
    the field stays None for the random-strategy game). Pins the
    ``record`` → ``search=None`` initialization so v3 schema is
    honored end-to-end."""
    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=1, base_seed=42, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 1
    replay = load_replay(files[0])
    # v3 snapshots all have a ``search`` key.
    assert replay.format_version == 3
    assert all("search" in s for s in replay.snapshots)
    # Random play has no search tree → every search field stays None.
    assert all(s["search"] is None for s in replay.snapshots)


# ---------- Diagnostic deep-tree capture (children field) -------------------


def test_build_search_annotation_root_only_emits_children_none():
    """Back-compat invariant: when ``deep_capture=False`` (the
    default) and candidates have no ``subtree`` key, every emitted
    candidate carries ``children=None`` — readers ignoring the field
    see exactly the pre-feature shape."""
    import random
    from mtg_bot.actions import PassPriority, PlayLand
    from mtg_bot.search import (
        Budget, MCTS, RandomRolloutEvaluator,
    )
    from mtg_bot.training import build_search_annotation

    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=8),
        rng=random.Random(0),
    )
    candidates = [
        {"decision": PassPriority(), "visits": 7,
         "q": 0.1, "prior": None, "leaf_value": None},
        {"decision": PlayLand(card_id=2), "visits": 3,
         "q": 0.0, "prior": None, "leaf_value": None},
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(), candidates=candidates,
        strategy=mcts, elapsed_ms=5,
    )
    for c in annotation["candidates"]:
        assert c["children"] is None


def test_build_search_annotation_deep_capture_translates_subtree():
    """When candidates carry an in-memory ``subtree`` key, encode it
    recursively as the on-disk ``children`` field. Empty subtrees
    become ``[]`` (not None) so the chevron-hidden contract is
    honored (``length === 0`` reads the same as ``no children``)."""
    import random
    from mtg_bot.actions import PassPriority, PlayLand
    from mtg_bot.search import (
        Budget, MCTS, RandomRolloutEvaluator,
    )
    from mtg_bot.training import build_search_annotation

    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=8),
        rng=random.Random(0),
    )
    candidates = [
        {
            "decision": PassPriority(), "visits": 10, "q": 0.1,
            "prior": None, "leaf_value": None,
            "subtree": [
                {
                    "decision": PlayLand(card_id=4), "visits": 5,
                    "q": -0.05, "prior": None, "leaf_value": 0.1,
                    "subtree": [],
                },
            ],
        },
        {
            "decision": PlayLand(card_id=2), "visits": 3, "q": 0.0,
            "prior": None, "leaf_value": None, "subtree": [],
        },
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(), candidates=candidates,
        strategy=mcts, elapsed_ms=5, deep_capture=True,
    )
    # First candidate (most-visited) has 1 child; child is a
    # serialized PlayLand with the right stats.
    first = annotation["candidates"][0]
    assert isinstance(first["children"], list)
    assert len(first["children"]) == 1
    grand = first["children"][0]
    assert grand["action"] == {"kind": "PlayLand", "card_id": 4}
    assert grand["visits"] == 5
    assert grand["q"] == -0.05
    assert grand["leaf_value"] == 0.1
    # Grandchild had subtree=[] → on-disk children=[] (sentinel for
    # "deep-capture, no further children" rather than None).
    assert grand["children"] == []
    # Second candidate had subtree=[] (no expanded grandchildren) but
    # deep_capture=True still emits []  — the viewer's chevron logic
    # treats len==0 the same as None (hide), so this stays UI-equivalent.
    assert annotation["candidates"][1]["children"] == []


def test_build_search_annotation_subtree_round_trips_through_gzip(
    tmp_path: Path,
):
    """A deep-captured search annotation embedded in a Replay must
    survive gzip-JSON serialization unchanged: the recursive
    ``children`` lists, the action shapes, and every numeric field."""
    import gzip
    import json as _json
    from mtg_bot.actions import CastSpell, PassPriority, PlayLand
    from mtg_bot.replay import REPLAY_FORMAT_VERSION, Replay, save_replay
    from mtg_bot.training.replay_capture import _encode_children

    # Hand-craft the on-disk shape directly (no MCTS involved — pure
    # serialization round-trip test).
    candidates_in_memory = [
        {
            "decision": PassPriority(), "visits": 12, "q": 0.2,
            "prior": 0.5, "leaf_value": 0.1,
            "subtree": [
                {
                    "decision": PlayLand(card_id=4), "visits": 6,
                    "q": -0.05, "prior": 0.2, "leaf_value": -0.02,
                    "subtree": [
                        {
                            "decision": CastSpell(card_id=9), "visits": 2,
                            "q": 0.3, "prior": 0.1, "leaf_value": 0.4,
                            "subtree": [],
                        },
                    ],
                },
                {
                    "decision": PassPriority(), "visits": 3,
                    "q": 0.0, "prior": 0.1, "leaf_value": None,
                    "subtree": [],
                },
            ],
        },
    ]
    encoded_root_children = _encode_children(
        {"subtree": candidates_in_memory}, deep_capture=True,
    )
    snap = {
        "turn_number": 1, "step": "precombat_main", "decision": None,
        "search": {
            "chosen": {"kind": "PassPriority"},
            "candidates": encoded_root_children,
            "samples": 1, "inner_nodes": 200, "elapsed_ms": 7,
        },
    }
    replay = Replay(
        format_version=REPLAY_FORMAT_VERSION,
        metadata={"game_idx": 0},
        snapshots=[snap],
    )
    path = tmp_path / "deep.json.gz"
    save_replay(replay, path)
    # Inspect the on-disk gzip: load + json-parse and check the
    # nested shape survives byte-identical.
    with gzip.open(path, "rt") as f:
        loaded = _json.load(f)
    out_root = loaded["snapshots"][0]["search"]["candidates"][0]
    assert out_root["action"] == {"kind": "PassPriority"}
    assert out_root["visits"] == 12
    # Two depth-1 children.
    assert len(out_root["children"]) == 2
    # First depth-1 child is PlayLand(4) with one depth-2 child.
    d1 = out_root["children"][0]
    assert d1["action"] == {"kind": "PlayLand", "card_id": 4}
    assert d1["visits"] == 6
    assert len(d1["children"]) == 1
    d2 = d1["children"][0]
    assert d2["action"]["kind"] == "CastSpell"
    assert d2["action"]["card_id"] == 9
    assert d2["visits"] == 2
    # Depth-2 child had no further children.
    assert d2["children"] == []
    # Second depth-1 sibling.
    sib = out_root["children"][1]
    assert sib["action"] == {"kind": "PassPriority"}
    assert sib["children"] == []
    # Schema-version unchanged — deep capture is purely additive.
    assert loaded["format_version"] == REPLAY_FORMAT_VERSION
    assert REPLAY_FORMAT_VERSION == 3


def test_build_search_annotation_deep_capture_size_bounded():
    """A realistic deep-mode annotation (depth=3, top-K=5, 5 root
    candidates) lands well under the 50 KB soft warning threshold —
    confirming that diagnostic captures are bounded and won't blow
    out the disk on a typical game (~270 priority moments)."""
    import random
    from mtg_bot.actions import (
        ActivateAbility, CastSpell, PassPriority, PlayLand,
    )
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.training import build_search_annotation

    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=200),
        rng=random.Random(0),
    )

    def _leaf(d, v):
        return {
            "decision": d, "visits": v, "q": 0.05, "prior": 0.2,
            "leaf_value": 0.0, "subtree": [],
        }

    def _branch(d, v, children):
        return {
            "decision": d, "visits": v, "q": 0.1, "prior": 0.3,
            "leaf_value": 0.05, "subtree": children,
        }

    # 5 root candidates, each with 5 depth-1 children, each with 5
    # depth-2 leaves. Matches what a real `--capture-tree-depth 3
    # --capture-top-k 5` run would produce on a node-rich snapshot.
    decs_l2 = [PassPriority(), PlayLand(card_id=1), CastSpell(card_id=2),
               ActivateAbility(source_id=3, ability_index=0),
               CastSpell(card_id=4)]
    decs_l1 = [PlayLand(card_id=10), CastSpell(card_id=11),
               PassPriority(), CastSpell(card_id=12),
               ActivateAbility(source_id=13, ability_index=0)]
    decs_l0 = [PassPriority(), PlayLand(card_id=20),
               CastSpell(card_id=21), CastSpell(card_id=22),
               ActivateAbility(source_id=23, ability_index=0)]
    candidates = [
        _branch(d0, 50 - 7 * i, [
            _branch(d1, 10 - j, [
                _leaf(d2, 3 - k) for k, d2 in enumerate(decs_l2)
            ])
            for j, d1 in enumerate(decs_l1)
        ])
        for i, d0 in enumerate(decs_l0)
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(), candidates=candidates,
        strategy=mcts, elapsed_ms=42, deep_capture=True,
    )
    import json as _json
    raw = len(_json.dumps(annotation).encode("utf-8"))
    # Sanity: deep mode SHOULD be much bigger than shallow's 2 KB
    # cap, but well under the 50 KB soft-warning threshold so a
    # full game's worth of ~270 snapshots stays single-digit MB.
    assert raw > 2048, f"deep annotation suspiciously small: {raw} B"
    assert raw < 50_000, (
        f"deep annotation exceeded 50 KB threshold: {raw} B"
    )


def test_build_search_annotation_depth_one_matches_choose_with_full_stats():
    """``deep_capture=False`` + plain candidates (no subtree key)
    must produce a per-snapshot annotation that's structurally
    identical to today's ``build_search_annotation`` output, modulo
    the new ``children=None`` field on each candidate. The on-disk
    shape stays back-compat with v3 readers because they ignore
    unknown fields."""
    import random
    from mtg_bot.actions import PassPriority, PlayLand
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.training import build_search_annotation

    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=8),
        rng=random.Random(0),
    )
    candidates = [
        {"decision": PassPriority(), "visits": 7, "q": 0.1,
         "prior": None, "leaf_value": None},
        {"decision": PlayLand(card_id=2), "visits": 3, "q": 0.0,
         "prior": None, "leaf_value": None},
    ]
    annotation = build_search_annotation(
        chosen=PassPriority(), candidates=candidates,
        strategy=mcts, elapsed_ms=5,
    )
    # Same five core fields, plus children=None.
    expected_core_keys = {
        "action", "visits", "q", "prior", "leaf_value", "children",
    }
    for c in annotation["candidates"]:
        assert set(c.keys()) == expected_core_keys
        assert c["children"] is None


def test_mcts_choose_with_subtree_walks_tree():
    """MCTS.choose_with_subtree must surface real tree structure
    under each root candidate (depth=2 walks one level past root)."""
    import random
    from mtg_bot.actions import PassPriority
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    adapter = MtgGameStateAdapter(state)
    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=80),
        rng=random.Random(0),
    )
    chosen, candidates = mcts.choose_with_subtree(
        adapter, 0, tree_depth=2, tree_top_k=3,
    )
    assert chosen is not None or candidates  # forced moves still ok
    # Every root candidate has the subtree key (real list, possibly empty
    # — depends on whether the budget reached that branch).
    for c in candidates:
        assert "subtree" in c
        assert isinstance(c["subtree"], list)
    # Children should be ≤ top_k=3 at depth 1.
    for c in candidates:
        assert len(c["subtree"]) <= 3


def test_mcts_choose_with_subtree_depth_one_matches_full_stats():
    """At ``tree_depth=1`` the new method's candidate stats match
    ``choose_with_full_stats`` exactly (modulo the new empty
    ``subtree`` key). Pins the back-compat invariant."""
    import random
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    # Two independent MCTS instances with identical RNG seeding +
    # the same root state — the searches are deterministic, so the
    # per-root-child stats must match across the two methods.
    a = MtgGameStateAdapter(state)
    b = MtgGameStateAdapter(state)
    mcts_a = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=64),
        rng=random.Random(0),
    )
    mcts_b = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=64),
        rng=random.Random(0),
    )
    chosen_a, cands_a = mcts_a.choose_with_full_stats(a, 0)
    chosen_b, cands_b = mcts_b.choose_with_subtree(
        b, 0, tree_depth=1, tree_top_k=None,
    )
    assert chosen_a == chosen_b
    assert len(cands_a) == len(cands_b)
    # Strip subtree from b for shape-equivalence check.
    cands_b_stripped = [
        {k: v for k, v in c.items() if k != "subtree"}
        for c in cands_b
    ]
    # Order doesn't matter at the root in choose_with_full_stats
    # (no sort), but choose_with_subtree sorts root by visits desc.
    # Compare as sets of frozensets-of-items.
    def _key(c):
        return (c["decision"], c["visits"])
    assert sorted(cands_a, key=_key) == sorted(cands_b_stripped, key=_key)
    # subtree on every root cand at depth=1 is empty list.
    for c in cands_b:
        assert c["subtree"] == []


def test_mcts_choose_with_subtree_invalid_depth_raises():
    """tree_depth < 1 is a programmer error; raise rather than
    silently returning empty results."""
    import random
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    adapter = MtgGameStateAdapter(state)
    mcts = MCTS(
        evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
        budget_factory=lambda: Budget(max_nodes=4),
        rng=random.Random(0),
    )
    with pytest.raises(ValueError):
        mcts.choose_with_subtree(adapter, 0, tree_depth=0)


def test_pimc_choose_with_subtree_aggregates_across_subsearches():
    """PIMC.choose_with_subtree returns the same root-level shape as
    choose_with_full_stats (visits summed across K worlds, q/prior/
    leaf_value visit-weighted means) PLUS recursive ``subtree`` lists
    aggregated the same way at each level."""
    import random
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.search.pimc import PIMC
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    adapter = MtgGameStateAdapter(state)
    pimc = PIMC(
        inner_factory=lambda: MCTS(
            evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
            budget_factory=lambda: Budget(max_nodes=40),
            rng=random.Random(0),
        ),
        samples=2,
        rng=random.Random(0),
    )
    chosen, candidates = pimc.choose_with_subtree(
        adapter, 0, tree_depth=2, tree_top_k=2,
    )
    # Root keeps all candidates regardless of top_k.
    assert chosen is not None or candidates
    for c in candidates:
        assert "subtree" in c
        # Depth-1 (children of root candidates) capped at top_k=2.
        assert len(c["subtree"]) <= 2
        # Every grandchild also has a subtree key (empty since
        # depth=2 stops there).
        for grand in c["subtree"]:
            assert grand["subtree"] == []


def test_pimc_choose_with_subtree_depth_one_back_compat():
    """At depth=1, PIMC.choose_with_subtree returns root candidates
    matching choose_with_full_stats, with empty subtree on each."""
    import random
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.search.pimc import PIMC
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    adapter = MtgGameStateAdapter(state)
    pimc = PIMC(
        inner_factory=lambda: MCTS(
            evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
            budget_factory=lambda: Budget(max_nodes=32),
            rng=random.Random(0),
        ),
        samples=2,
        rng=random.Random(0),
    )
    _, candidates = pimc.choose_with_subtree(
        adapter, 0, tree_depth=1, tree_top_k=None,
    )
    for c in candidates:
        assert c["subtree"] == []


def test_pimc_choose_with_subtree_invalid_depth_raises():
    import random
    from mtg_bot.agents import RandomAgent
    from mtg_bot.search import Budget, MCTS, RandomRolloutEvaluator
    from mtg_bot.search.engine_adapter import MtgGameStateAdapter
    from mtg_bot.search.pimc import PIMC
    from mtg_bot.simulate import setup_game

    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=0,
    )
    adapter = MtgGameStateAdapter(state)
    pimc = PIMC(
        inner_factory=lambda: MCTS(
            evaluator=RandomRolloutEvaluator(rng=random.Random(0)),
            budget_factory=lambda: Budget(max_nodes=4),
            rng=random.Random(0),
        ),
        samples=2,
    )
    with pytest.raises(ValueError):
        pimc.choose_with_subtree(adapter, 0, tree_depth=0)


def test_replay_capture_config_has_tree_capture_fields():
    """ReplayCaptureConfig exposes tree_depth and tree_top_k with
    production-safe defaults (1, None)."""
    cfg = ReplayCaptureConfig(save_dir="x")
    assert cfg.tree_depth == 1
    assert cfg.tree_top_k is None
    cfg2 = ReplayCaptureConfig(
        save_dir="x", tree_depth=3, tree_top_k=5,
    )
    assert cfg2.tree_depth == 3
    assert cfg2.tree_top_k == 5


def test_end_to_end_data_gen_deep_capture_writes_children(tmp_path: Path):
    """Run a sequential data_gen with PIMC + deep capture (depth=2,
    top_k=2). Resulting replay has at least one snapshot whose
    search.candidates carry populated ``children`` lists; non-deep
    captures (default) still emit ``children=None`` exclusively."""
    from mtg_bot.training import PimcConfig
    config = PimcConfig(samples=1, inner_nodes=8)
    factory = lambda seed, _c=config: _c.build(seed)

    save_dir = tmp_path / "deep_replays"
    cfg = ReplayCaptureConfig(
        save_dir=str(save_dir), capture_every=1,
        tree_depth=2, tree_top_k=2,
    )
    generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 1
    replay = load_replay(files[0])
    assert replay.format_version == 3  # NOT bumped to 4
    annotated = [
        s for s in replay.snapshots if s.get("search") is not None
    ]
    assert annotated
    saw_populated = False
    for s in annotated:
        for c in s["search"]["candidates"]:
            assert "children" in c
            # children is either None (forced moves with no real
            # search), [] (no expanded grandchildren), or a real list.
            if c["children"]:
                saw_populated = True
                assert len(c["children"]) <= 2  # top_k=2
                # Recursive shape check on first depth-1 child.
                for grand in c["children"]:
                    assert "action" in grand
                    assert isinstance(grand["visits"], int)
                    # children at depth=2 boundary is [] (terminate)
                    # because tree_depth=2 means root + 1 deeper.
                    assert grand["children"] == []
    assert saw_populated, (
        "no candidate with populated children — deep capture "
        "should produce at least one"
    )


def test_end_to_end_data_gen_default_depth_emits_no_children(
    tmp_path: Path,
):
    """Production default (tree_depth=1, tree_top_k=None): every
    annotated candidate has children=None. No size regression vs
    pre-feature replays."""
    from mtg_bot.training import PimcConfig
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)

    save_dir = tmp_path / "shallow"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, replay_config=cfg,
        deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    replay = load_replay(files[0])
    annotated = [
        s for s in replay.snapshots if s.get("search") is not None
    ]
    assert annotated
    for s in annotated:
        for c in s["search"]["candidates"]:
            # Default path: children=None on every candidate.
            assert c["children"] is None


def test_recorder_warns_on_oversized_deep_annotation():
    """The soft size sanity check fires a warning when a deep
    annotation exceeds the 50 KB threshold but does NOT raise."""
    import warnings as _warnings
    from mtg_bot.training.replay_capture import (
        _DEEP_ANNOTATION_SIZE_WARN_BYTES,
    )

    rec = ReplayRecorder()
    # Build an artificial annotation just over the threshold by
    # padding chosen action with a long string. (children=[long str]
    # makes it deep.)
    big_payload = "x" * (_DEEP_ANNOTATION_SIZE_WARN_BYTES + 100)
    annotation = {
        "chosen": {"kind": "Filler", "pad": big_payload},
        "candidates": [{
            "action": {"kind": "PassPriority"}, "visits": 1,
            "q": None, "prior": None, "leaf_value": None,
            "children": [
                {
                    "action": {"kind": "X", "pad": big_payload},
                    "visits": 1, "q": None, "prior": None,
                    "leaf_value": None, "children": [],
                },
            ],
        }],
    }
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        size = rec.maybe_warn_deep_annotation_size(annotation)
    assert size is not None
    assert size > _DEEP_ANNOTATION_SIZE_WARN_BYTES
    assert len(caught) == 1
    assert "deep search annotation" in str(caught[0].message)


def test_recorder_no_warning_on_shallow_annotation():
    """Production root-only annotations (children=None on every
    candidate) skip the size check entirely — no overhead, no
    spurious warnings."""
    import warnings as _warnings

    rec = ReplayRecorder()
    annotation = {
        "chosen": {"kind": "PassPriority"},
        "candidates": [
            {"action": {"kind": "PassPriority"}, "visits": 5,
             "q": 0.1, "prior": None, "leaf_value": None,
             "children": None},
        ],
        "samples": 1, "inner_nodes": 50, "elapsed_ms": 5,
    }
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        size = rec.maybe_warn_deep_annotation_size(annotation)
    assert size is None
    assert len(caught) == 0


def test_recorder_no_warning_on_small_deep_annotation():
    """Deep annotation under threshold: size measured but no warning."""
    import warnings as _warnings

    rec = ReplayRecorder()
    annotation = {
        "chosen": {"kind": "PassPriority"},
        "candidates": [{
            "action": {"kind": "PassPriority"}, "visits": 1,
            "q": None, "prior": None, "leaf_value": None,
            "children": [
                {"action": {"kind": "PassPriority"}, "visits": 1,
                 "q": None, "prior": None, "leaf_value": None,
                 "children": []},
            ],
        }],
    }
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        size = rec.maybe_warn_deep_annotation_size(annotation)
    assert size is not None
    assert len(caught) == 0


def test_end_to_end_replay_capture_records_decisions(tmp_path: Path):
    """Run two games with replay capture enabled. Every snapshot
    after the first carries a non-None decision that round-trips
    through ``decision_from_json``."""
    from mtg_bot.training.replay_decisions import decision_from_json

    save_dir = tmp_path / "replays"
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=1)
    generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 2
    saw_a_real_game = False
    for f in files:
        replay = load_replay(f)
        assert len(replay.snapshots) >= 1
        # First snapshot is the pre-game initial state — no decision.
        assert replay.snapshots[0]["decision"] is None
        # Every subsequent snapshot must carry an action that decodes.
        for snap in replay.snapshots[1:]:
            saw_a_real_game = True
            assert snap["decision"] is not None
            decoded = decision_from_json(snap["decision"])
            # Sanity: encoding is bijective end-to-end.
            from mtg_bot.training.replay_decisions import decision_to_json
            assert decision_to_json(decoded) == snap["decision"]
    # At least one game advanced past turn 0; otherwise this test
    # would silently pass with zero decisions checked.
    assert saw_a_real_game
