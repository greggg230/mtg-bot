"""Every path that turns registered weights into an evaluator applies
the checkpoint's value-head options the same way.

Found 2026-10-04: ``NNPimcConfig.build`` -- the builder behind every
worker self-play / eval game, iterate's local arenas and ``mtg_bot
ladder`` -- read the structured trunk dims from the sibling
``metadata.json`` itself and never passed ``value_calibration`` or
``value_head_input`` to ``NNEvaluator``, unlike
``NNEvaluator.from_metadata``. Every played game ran the value head
uncalibrated while the held-out gate graded the calibrated one.

Pins:

* the shared resolver (``resolve_value_calibration`` /
  ``structured_kwargs_from_metadata`` /
  ``check_flat_metadata_value_options``): identity default, the
  ``+nocal`` opt-out, and a loud failure for every calibration or
  head input the builder cannot apply;
* ``NNPimcConfig.build``, the worker subprocess self-play and eval
  bodies, the ladder (``vN`` / ``vN+nocal``) and the SHM server all
  produce the same values as ``from_metadata`` on real game states;
* the same agreement on a COPY of a real registered checkpoint
  (``trained_weights_s2/v22`` from the main checkout; skipped when
  the gitignored weights are absent).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mtg_bot.agents import RandomAgent
from mtg_bot.features import FEATURE_DIM
from mtg_bot.search import MtgGameStateAdapter
from mtg_bot.search.nn_evaluator import (
    ARCHITECTURE_STRUCTURED,
    NNEvaluator,
    check_flat_metadata_value_options,
    resolve_value_calibration,
    structured_kwargs_from_metadata,
)
from mtg_bot.simulate import setup_game
from mtg_bot.training import NNPimcConfig, Registry
from mtg_bot.training.registry import ModelMetadata
from tests._card_vocab_helpers import current_vocab_dict
from tests._test_decks import simple_test_decklists


CALIBRATION = {
    "method": "logit_temperature",
    "temperature": 0.5,
    "bias": 0.1,
    "scale": 1.0,
    "n_rows": 1000,
}
TRUNK = {
    "emb_dim": 16, "n_heads": 2, "n_layers": 1, "hidden_dim": 32,
    "action_history_n": 4, "n_printings": 32,
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _register_structured(
    root: Path,
    *,
    version: str = "v1",
    calibration: dict[str, Any] | None = None,
    head_input: str = "trunk_no_printing",
) -> Path:
    """Register a small random-init structured checkpoint the way
    ``mtg_bot train`` does (nested ``structured_config`` + flat
    ``structured_value_head_input``) and return its weights path."""
    torch.manual_seed(1234)
    ev = NNEvaluator(
        architecture=ARCHITECTURE_STRUCTURED,
        structured_emb_dim=TRUNK["emb_dim"],
        structured_n_heads=TRUNK["n_heads"],
        structured_n_layers=TRUNK["n_layers"],
        structured_hidden_dim=TRUNK["hidden_dim"],
        structured_action_history_n=TRUNK["action_history_n"],
        structured_n_printings=TRUNK["n_printings"],
        structured_value_head_input=head_input,
    )
    src = root.parent / f"src_{root.name}_{version}.pt"
    ev.save(src)
    meta = ModelMetadata(
        version=version,
        created_at="2026-10-04T00:00:00Z",
        feature_dim=FEATURE_DIM,
        hidden_sizes=[],
        with_policy=True,
        architecture=ARCHITECTURE_STRUCTURED,
        action_encoding_version="canonical_v2",
        scalar_schema_version=int(ev.model.config.scalar_schema_version),
        value_calibration=dict(calibration or {}),
        card_vocab=current_vocab_dict(),
        training={
            "structured_config": {**TRUNK, "value_head_input": head_input},
            "structured_value_head_input": head_input,
        },
    )
    registry = Registry(root)
    registry.register(src, meta, version=version)
    return registry.weights_path(version)


def _states(n: int = 12, seed: int = 3) -> list[MtgGameStateAdapter]:
    """``n`` non-terminal mid-game states from random play."""
    rng = random.Random(seed)
    out: list[MtgGameStateAdapter] = []
    game = 0
    while len(out) < n:
        s = MtgGameStateAdapter(setup_game(
            [RandomAgent(), RandomAgent()],
            decks=simple_test_decklists(), seed=seed + game,
        ))
        game += 1
        for step in range(300):
            if s.is_terminal():
                break
            if step % 13 == 4:
                out.append(s.clone())
            s.apply(rng.choice(s.legal_decisions()))
    return out[:n]


def _values(evaluator: Any, states: list) -> np.ndarray:
    return np.array([
        evaluator.evaluate(s, s.active_player()) for s in states
    ])


def _seat_evaluator(factory) -> NNEvaluator:
    """The NNEvaluator inside a PIMC seat factory's inner MCTS."""
    return factory(0).inner_factory().evaluator


@pytest.fixture(scope="module")
def states() -> list[MtgGameStateAdapter]:
    return _states()


@pytest.fixture
def calibrated_version(tmp_path: Path) -> Path:
    return _register_structured(
        tmp_path / "registry", calibration=CALIBRATION,
    )


def _reference(weights_path: Path, *, apply: bool = True) -> NNEvaluator:
    registry = Registry(weights_path.parent.parent)
    return NNEvaluator.from_metadata(
        weights_path,
        registry.load_metadata(weights_path.parent.name),
        apply_value_calibration=apply,
    )


# --------------------------------------------------------------------------
# The shared resolver
# --------------------------------------------------------------------------


def test_resolver_identity_when_absent_or_empty():
    assert resolve_value_calibration({}) == (1.0, 0.0)
    assert resolve_value_calibration({"value_calibration": {}}) == (1.0, 0.0)
    assert resolve_value_calibration(
        {"value_calibration": {"method": "identity"}},
    ) == (1.0, 0.0)


def test_resolver_reads_top_level_then_training_fallback():
    assert resolve_value_calibration(
        {"value_calibration": CALIBRATION},
    ) == (0.5, 0.1)
    assert resolve_value_calibration(
        {"training": {"value_calibration": CALIBRATION}},
    ) == (0.5, 0.1)


def test_resolver_opt_out_returns_identity():
    assert resolve_value_calibration(
        {"value_calibration": CALIBRATION}, apply_value_calibration=False,
    ) == (1.0, 0.0)


@pytest.mark.parametrize("record, match", [
    ({"method": "output_shrink", "scale": 0.3}, "output_shrink"),
    ({"method": "platt", "temperature": 2.0}, "unknown method"),
    ({"method": "identity", "temperature": 2.0}, "refusing to guess"),
    ({"method": "logit_temperature", "temperature": 2.0, "scale": 0.5},
     "scale=0.5"),
])
def test_resolver_refuses_what_it_cannot_apply(record, match):
    with pytest.raises(ValueError, match=match):
        resolve_value_calibration({"value_calibration": record})
    # ...even when the caller opted out of applying it: a corrupt
    # record is not hidden behind ``+nocal``.
    with pytest.raises(ValueError, match=match):
        resolve_value_calibration(
            {"value_calibration": record}, apply_value_calibration=False,
        )


def test_flat_mlp_refuses_calibration_and_head_input():
    with pytest.raises(ValueError, match="flat MLP"):
        check_flat_metadata_value_options({"value_calibration": CALIBRATION})
    # Opting out is fine for an MLP: nothing to apply.
    check_flat_metadata_value_options(
        {"value_calibration": CALIBRATION}, apply_value_calibration=False,
    )
    with pytest.raises(ValueError, match="value_head_input"):
        check_flat_metadata_value_options(
            {"training": {"structured_value_head_input": "trunk_no_printing"}},
        )
    check_flat_metadata_value_options(
        {"training": {"structured_config": {"value_head_input": "trunk"}}},
    )


def test_structured_kwargs_agree_for_dict_and_model_metadata(
    calibrated_version: Path,
):
    raw = json.loads(
        (calibrated_version.parent / "metadata.json").read_text("utf-8"),
    )
    meta = ModelMetadata.from_dict(raw)
    a = structured_kwargs_from_metadata(raw)
    b = structured_kwargs_from_metadata(meta)
    assert a == b
    assert a["structured_value_temperature"] == 0.5
    assert a["structured_value_bias"] == 0.1
    assert a["structured_value_head_input"] == "trunk_no_printing"
    assert a["structured_emb_dim"] == TRUNK["emb_dim"]
    nocal = structured_kwargs_from_metadata(raw, apply_value_calibration=False)
    assert (nocal["structured_value_temperature"],
            nocal["structured_value_bias"]) == (1.0, 0.0)


def test_raw_dict_without_scalar_schema_is_schema_1():
    """A pre-v2 ``metadata.json`` carries no scalar schema anywhere;
    the raw-dict path keeps the old ``NNPimcConfig.build`` default."""
    assert structured_kwargs_from_metadata(
        {"training": {}},
    )["structured_scalar_schema_version"] == 1


# --------------------------------------------------------------------------
# NNPimcConfig.build == from_metadata
# --------------------------------------------------------------------------


def _pimc(weights_path: Path, **overrides) -> NNPimcConfig:
    kwargs: dict[str, Any] = dict(
        weights_path=str(weights_path), samples=1, inner_nodes=2,
        batch_size=1, with_policy=True, puct=True,
        architecture=ARCHITECTURE_STRUCTURED,
        action_encoding_version="canonical_v2",
    )
    kwargs.update(overrides)
    return NNPimcConfig(**kwargs)


def test_nnpimc_build_matches_from_metadata(calibrated_version, states):
    seat = _seat_evaluator(_pimc(calibrated_version).build)
    ref = _reference(calibrated_version)
    cfg = seat.model.config
    assert cfg.value_temperature == 0.5 and cfg.value_bias == 0.1
    assert cfg.value_head_input == "trunk_no_printing"
    np.testing.assert_array_equal(
        _values(seat, states), _values(ref, states),
    )


def test_nnpimc_nocal_matches_uncalibrated_from_metadata(
    calibrated_version, states,
):
    seat = _seat_evaluator(
        _pimc(calibrated_version, apply_value_calibration=False).build,
    )
    assert seat.model.config.value_temperature == 1.0
    # value_head_input is still honoured: +nocal only drops the fit.
    assert seat.model.config.value_head_input == "trunk_no_printing"
    raw = _values(seat, states)
    np.testing.assert_array_equal(
        raw, _values(_reference(calibrated_version, apply=False), states),
    )
    cal = _values(_reference(calibrated_version), states)
    assert not np.allclose(raw, cal)
    # Exactly the recorded fit: tanh(logit / T + b) on the raw logit.
    np.testing.assert_allclose(
        cal, np.tanh(np.arctanh(raw) / 0.5 + 0.1), rtol=1e-4, atol=1e-5,
    )


def test_nnpimc_build_defers_architecture_to_the_sidecar(
    calibrated_version, states,
):
    """``architecture=None`` (bench / older callers) with a structured
    sidecar loads the structured net from metadata instead of trying an
    MLP; ``action_encoding_version=None`` takes the sidecar's."""
    seat = _seat_evaluator(_pimc(
        calibrated_version, architecture=None, action_encoding_version=None,
    ).build)
    assert seat.is_structured
    assert seat.action_encoding_version == "canonical_v2"
    np.testing.assert_array_equal(
        _values(seat, states), _values(_reference(calibrated_version), states),
    )


@pytest.mark.parametrize("field, value", [
    ("action_encoding_version", "positional_v1"),
    ("architecture", "shared"),
])
def test_nnpimc_build_refuses_a_config_that_contradicts_the_sidecar(
    calibrated_version, field, value,
):
    meta_path = calibrated_version.parent / "metadata.json"
    cfg = _pimc(calibrated_version, **{field: value})
    if field == "architecture":
        # A flat config against a structured sidecar: the sidecar's
        # calibration cannot be applied to an MLP -> loud.
        with pytest.raises(ValueError, match="flat MLP"):
            cfg.build(0)
    else:
        with pytest.raises(ValueError, match="disagrees"):
            cfg.build(0)
    assert meta_path.is_file()


def test_nnpimc_build_fills_a_minimal_sidecar_from_the_config(
    tmp_path, states,
):
    """Hand-written sidecars may omit ``architecture``; the config's
    value fills it and the calibration in the sidecar still applies."""
    weights = _register_structured(tmp_path / "reg", calibration=CALIBRATION)
    meta_path = weights.parent / "metadata.json"
    raw = json.loads(meta_path.read_text("utf-8"))
    del raw["architecture"]
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    seat = _seat_evaluator(_pimc(weights).build)
    assert seat.model.config.value_temperature == 0.5


def test_nnpimc_build_refuses_unreadable_sidecar(calibrated_version):
    (calibrated_version.parent / "metadata.json").write_text(
        "{not json", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot read"):
        _pimc(calibrated_version).build(0)
    (calibrated_version.parent / "metadata.json").write_text(
        "[1, 2]", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a JSON object"):
        _pimc(calibrated_version).build(0)


def test_nnpimc_build_refuses_output_shrink_on_a_structured_sidecar(
    tmp_path,
):
    weights = _register_structured(
        tmp_path / "reg",
        calibration={"method": "output_shrink", "scale": 0.4},
    )
    with pytest.raises(ValueError, match="output_shrink"):
        _pimc(weights).build(0)


def test_nnpimc_build_refuses_calibrated_flat_checkpoint(tmp_path):
    """An MLP checkpoint whose sidecar declares a calibration cannot be
    honoured; ``apply_value_calibration=False`` loads it explicitly."""
    d = tmp_path / "v1"
    d.mkdir()
    weights = d / "weights.pt"
    NNEvaluator(with_policy=True).save(weights)
    (d / "metadata.json").write_text(json.dumps({
        "architecture": "shared", "value_calibration": CALIBRATION,
    }), encoding="utf-8")
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=1, inner_nodes=2,
        batch_size=1, with_policy=True, puct=True, architecture="shared",
    )
    with pytest.raises(ValueError, match="flat MLP"):
        cfg.build(0)
    NNPimcConfig.from_dict(
        {**cfg.to_dict(), "apply_value_calibration": False},
    ).build(0)


def test_nnpimc_apply_value_calibration_round_trips():
    cfg = NNPimcConfig(weights_path="w.pt", apply_value_calibration=False)
    d = cfg.to_dict()
    assert d["apply_value_calibration"] is False
    assert NNPimcConfig.from_dict(d) == cfg
    # Specs serialised before the field existed apply the calibration.
    del d["apply_value_calibration"]
    assert NNPimcConfig.from_dict(d).apply_value_calibration is True


# --------------------------------------------------------------------------
# Worker subprocess bodies (self-play + eval) == from_metadata
# --------------------------------------------------------------------------


def _worker_runner(monkeypatch, weights_dir: Path):
    import mtg_bot.worker_daemon_production_runner as prr

    monkeypatch.setattr(prr, "_SUBPROCESS_STATE", {"deck_pool": ["deck"]})
    monkeypatch.setattr(
        prr, "fetch_weights_if_missing",
        lambda *, coordinator_url, version, weights_root: weights_dir,
    )
    return prr


def test_worker_self_play_seat_matches_from_metadata(
    monkeypatch, calibrated_version, states,
):
    prr = _worker_runner(monkeypatch, calibrated_version.parent)
    captured: dict[str, Any] = {}

    def fake_play(**kwargs):
        captured.update(kwargs)
        return {}, "completed", 0, 0

    monkeypatch.setattr(prr, "_play_self_play_game", fake_play)
    monkeypatch.setattr(prr, "_encode_npz_bytes", lambda *a, **k: b"")
    prr._subprocess_play_self_play(
        claim={"weights": {"p0": "v1"}, "nodes": 3, "samples": 1},
        coordinator_url="http://unused", weights_cache_root="unused",
        deck_pool_path=None,
    )
    seat = _seat_evaluator(captured["factory"])
    assert seat.model.config.value_temperature == 0.5
    np.testing.assert_array_equal(
        _values(seat, states), _values(_reference(calibrated_version), states),
    )


def test_worker_eval_seats_match_from_metadata(
    monkeypatch, calibrated_version, states,
):
    prr = _worker_runner(monkeypatch, calibrated_version.parent)
    captured: dict[str, Any] = {}

    def fake_eval(**kwargs):
        captured.update(kwargs)
        return 0, 10, "completed"

    monkeypatch.setattr(prr, "_play_eval_game", fake_eval)
    prr._subprocess_play_eval(
        claim={"weights": {"new": "v1", "old": "v1"}, "nodes": 3,
               "samples": 1},
        coordinator_url="http://unused", weights_cache_root="unused",
        deck_pool_path=None,
    )
    ref = _values(_reference(calibrated_version), states)
    for key in ("p0_factory", "p1_factory"):
        np.testing.assert_array_equal(
            _values(_seat_evaluator(captured[key]), states), ref,
        )


# --------------------------------------------------------------------------
# Ladder + SHM server
# --------------------------------------------------------------------------


def test_ladder_seats_with_and_without_nocal(calibrated_version, states):
    from mtg_bot.cli import ladder as L

    root = calibrated_version.parent.parent
    cal = L.resolve_agent("v1", registry_root=root, nodes=2, samples=1)
    nocal = L.resolve_agent("v1+nocal", registry_root=root, nodes=2, samples=1)
    assert cal.description["apply_value_calibration"] is True
    assert nocal.description["apply_value_calibration"] is False
    assert nocal.label == "v1+nocal" and nocal.description["version"] == "v1"
    assert nocal.description["value_calibration"]["temperature"] == 0.5
    np.testing.assert_array_equal(
        _values(_seat_evaluator(cal.factory), states),
        _values(_reference(calibrated_version), states),
    )
    np.testing.assert_array_equal(
        _values(_seat_evaluator(nocal.factory), states),
        _values(_reference(calibrated_version, apply=False), states),
    )


@pytest.mark.parametrize("label, match", [
    ("heuristic+nocal", "needs a registered version"),
    ("random+nocal", "needs a registered version"),
    ("v1+nocal+nocal", "given twice"),
])
def test_ladder_nocal_rejects_misuse(tmp_path, label, match):
    from mtg_bot.cli import ladder as L

    with pytest.raises(ValueError, match=match):
        L.resolve_agent(label, registry_root=tmp_path, nodes=2, samples=1)


def test_split_calibration_modifier_keeps_other_modifiers():
    from mtg_bot.cli import ladder as L

    assert L.split_calibration_modifier("v22") == ("v22", True)
    assert L.split_calibration_modifier("v22+nocal") == ("v22", False)
    assert L.split_calibration_modifier("v22+nocal+b1") == ("v22+b1", False)


def test_shm_server_from_metadata_matches(calibrated_version):
    from mtg_bot.search.shm_nn_eval_server import SharedTensorNNEvalServer

    registry = Registry(calibrated_version.parent.parent)
    meta = registry.load_metadata("v1")
    server = SharedTensorNNEvalServer.from_metadata(
        calibrated_version, meta, device="cpu", n_slots=2,
        batch_size=1, batch_wait_ms=0.0,
    )
    ref = _reference(calibrated_version)
    assert server._model.config == ref.model.config
    for name, p in ref.model.state_dict().items():
        assert torch.equal(server._model.state_dict()[name], p)
    nocal = SharedTensorNNEvalServer.from_metadata(
        calibrated_version, meta, apply_value_calibration=False,
        device="cpu", n_slots=2, batch_size=1, batch_wait_ms=0.0,
    )
    assert nocal._model.config.value_temperature == 1.0


def test_shm_server_from_metadata_flat_checkpoint(tmp_path):
    from mtg_bot.search.shm_nn_eval_server import SharedTensorNNEvalServer

    weights = tmp_path / "w.pt"
    NNEvaluator(with_policy=True, hidden_sizes=(8, 8)).save(weights)
    meta = {"architecture": "shared", "with_policy": True,
            "hidden_sizes": [8, 8]}
    server = SharedTensorNNEvalServer.from_metadata(
        weights, meta, device="cpu", n_slots=2, batch_size=1,
        batch_wait_ms=0.0,
    )
    assert server.handle().with_policy is True
    with pytest.raises(ValueError, match="flat MLP"):
        SharedTensorNNEvalServer.from_metadata(
            weights, {**meta, "value_calibration": CALIBRATION},
            device="cpu", n_slots=2, batch_size=1, batch_wait_ms=0.0,
        )


def test_from_metadata_flat_refuses_value_head_input(tmp_path):
    weights = tmp_path / "w.pt"
    NNEvaluator(with_policy=True).save(weights)
    with pytest.raises(ValueError, match="value_head_input"):
        NNEvaluator.from_metadata(weights, {
            "architecture": "shared", "with_policy": True,
            "training": {"structured_value_head_input": "trunk_no_printing"},
        })


# --------------------------------------------------------------------------
# A copy of a REAL registered checkpoint (main checkout's v22)
# --------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _real_version_dir(version: str = "v22") -> Path | None:
    for root in (
        _REPO_ROOT / "trained_weights_s2",
        _REPO_ROOT.parent.parent / "mtg-bot" / "trained_weights_s2",
    ):
        d = root / version
        if (d / "weights.pt").is_file() and (d / "metadata.json").is_file():
            return d
    return None


@pytest.fixture
def real_checkpoint_copy(tmp_path: Path) -> Path:
    src = _real_version_dir()
    if src is None:
        pytest.skip(
            "trained_weights_s2/v22 not found (gitignored; lives in the "
            "main checkout)"
        )
    dst = tmp_path / "trained_weights_s2" / "v22"
    shutil.copytree(src, dst)
    return dst / "weights.pt"


def test_real_checkpoint_ladder_and_worker_paths_match_from_metadata(
    monkeypatch, real_checkpoint_copy, states,
):
    from mtg_bot.cli import ladder as L

    weights = real_checkpoint_copy
    meta = json.loads((weights.parent / "metadata.json").read_text("utf-8"))
    temperature = meta["value_calibration"]["temperature"]
    assert meta["value_calibration"]["method"] == "logit_temperature"
    assert temperature != 1.0

    ref = _values(_reference(weights), states)
    ref_raw = _values(_reference(weights, apply=False), states)
    assert not np.allclose(ref, ref_raw)

    # Ladder seats.
    root = weights.parent.parent
    cal = L.resolve_agent("v22", registry_root=root, nodes=2, samples=1)
    nocal = L.resolve_agent("v22+nocal", registry_root=root, nodes=2, samples=1)
    cal_seat = _seat_evaluator(cal.factory)
    assert cal_seat.model.config.value_temperature == pytest.approx(temperature)
    np.testing.assert_array_equal(_values(cal_seat, states), ref)
    np.testing.assert_array_equal(
        _values(_seat_evaluator(nocal.factory), states), ref_raw,
    )

    # Worker self-play body, reading the cached (copied) version dir.
    import mtg_bot.worker_daemon_production_runner as prr

    _worker_runner(monkeypatch, weights.parent)
    captured: dict[str, Any] = {}

    def fake_play(**kwargs):
        captured.update(kwargs)
        return {}, "completed", 0, 0

    monkeypatch.setattr(prr, "_play_self_play_game", fake_play)
    monkeypatch.setattr(prr, "_encode_npz_bytes", lambda *a, **k: b"")
    prr._subprocess_play_self_play(
        claim={"weights": {"p0": "v22"}, "nodes": 3, "samples": 1},
        coordinator_url="http://unused", weights_cache_root="unused",
        deck_pool_path=None,
    )
    np.testing.assert_array_equal(
        _values(_seat_evaluator(captured["factory"]), states), ref,
    )
