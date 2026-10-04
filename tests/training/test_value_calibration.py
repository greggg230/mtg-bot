"""Tests for :mod:`mtg_bot.training.value_calibration`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mtg_bot.training.value_calibration import (
    CALIBRATION_IDENTITY,
    CALIBRATION_LOGIT_TEMPERATURE,
    CALIBRATION_OUTPUT_SHRINK,
    ValueCalibration,
    apply_calibration,
    apply_logit_temperature,
    calibrate_holdout,
    fit_logit_temperature,
    fit_output_shrink,
)
from mtg_bot.training.value_gate import (
    compute_holdout_value_metrics,
    evaluate_value_gate,
    pearson_corr,
)


def _saturated_corpus(
    n: int = 40_000, corr: float = 0.3, seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise the exact failure mode the gate rejected.

    Returns ``(logits, targets)`` where targets are ±1 with equal
    probability and the pre-tanh logits are LARGE (so ``tanh`` is
    saturated near ±1) but only weakly aligned with the target — the
    2026-09-25 run's "as informative as one scalar, wildly
    over-confident" signature.
    """
    rng = np.random.default_rng(seed)
    targets = rng.choice([-1.0, 1.0], size=n)
    # latent = corr * target + noise, then blown up by 4x so tanh
    # saturates. The blow-up is a pure monotone rescale, so it does
    # not change the correlation of the latent with the target.
    latent = corr * targets + math.sqrt(1.0 - corr ** 2) * rng.normal(
        size=n,
    )
    return 4.0 * latent, targets


def test_saturated_predictor_is_worse_than_constant_zero_raw():
    logits, targets = _saturated_corpus()
    preds = np.tanh(logits)
    raw_mse = float(np.mean((preds - targets) ** 2))
    r = pearson_corr(preds, targets)
    assert 0.25 < r < 0.35, r
    assert float(np.mean(np.abs(preds))) > 0.8
    # The whole point: correlation 0.3 and yet WORSE than predicting 0.
    assert raw_mse > 1.0


def test_temperature_fit_beats_constant_zero_on_the_same_rows():
    logits, targets = _saturated_corpus()
    t, b = fit_logit_temperature(logits, targets)
    assert b == 0.0  # temperature-only by default
    assert t > 1.0  # a saturated head needs shrinking, not sharpening
    cal = apply_logit_temperature(logits, t, b)
    cal_mse = float(np.mean((cal - targets) ** 2))
    raw_mse = float(np.mean((np.tanh(logits) - targets) ** 2))
    assert raw_mse > 1.0
    assert cal_mse < 1.0
    # A calibrated predictor with correlation r scores ~1 - r**2.
    r = pearson_corr(cal, targets)
    assert cal_mse == pytest.approx(1.0 - r ** 2, abs=0.02)
    # Un-saturating RECOVERS correlation (a saturated tanh is a
    # lossy nonlinearity; Pearson r is not rank-based, so squashing
    # everything onto ±1 costs real correlation).
    assert r > pearson_corr(np.tanh(logits), targets)
    # ...and the output is no longer shouting.
    assert float(np.mean(np.abs(cal))) < 0.5


def test_calibrate_holdout_reports_raw_and_calibrated():
    logits, targets = _saturated_corpus()
    calibration, cal_preds = calibrate_holdout(targets, logits=logits)
    assert calibration.method == CALIBRATION_LOGIT_TEMPERATURE
    assert calibration.n_rows == targets.shape[0]
    assert calibration.raw_mse > 1.0
    assert calibration.calibrated_mse < 1.0
    # The cross-fitted score (fit on half, score on the other half)
    # must also clear 1.0 — otherwise the win was the two free
    # parameters seeing the rows they were graded on.
    assert calibration.cv_mse is not None
    assert calibration.cv_mse < 1.0
    assert calibration.cv_mse == pytest.approx(
        calibration.calibrated_mse, abs=0.01,
    )
    assert calibration.raw_pred_abs_mean > 0.8
    assert calibration.calibrated_pred_abs_mean < 0.5
    assert cal_preds.shape == targets.shape


def test_temperature_fit_leaves_a_calibrated_head_alone():
    """A head that is already well-scaled should get T ~ 1.

    Built so that ``tanh(logit)`` IS the conditional mean of the
    label: draw a logit, let ``q = tanh(logit)``, then sample the ±1
    label with ``P(+1) = (1 + q) / 2``. The MSE-optimal temperature
    for that corpus is exactly 1, so the fit has nowhere to go.
    """
    rng = np.random.default_rng(3)
    logits = rng.normal(size=60_000)
    q = np.tanh(logits)
    targets = np.where(rng.random(logits.shape[0]) < (1 + q) / 2, 1.0, -1.0)
    t, b = fit_logit_temperature(logits, targets)
    assert 0.85 < t < 1.2
    assert abs(b) < 0.05


def test_every_layer_defaults_to_temperature_only():
    """The bias default must agree at EVERY layer.

    ``compute_holdout_value_metrics`` is the layer the trainer calls,
    so a stale ``fit_bias=True`` there would silently hand the gate a
    two-parameter fit even though both fitters default to one.
    Caught in production on 2026-09-25: a 9-epoch fit came back with
    ``bias=-0.0791`` after the fitters had already been pinned.
    """
    import inspect

    from mtg_bot.training import value_calibration as vc

    for fn in (
        vc.fit_logit_temperature,
        vc.calibrate_holdout,
        compute_holdout_value_metrics,
    ):
        assert (
            inspect.signature(fn).parameters["fit_bias"].default is False
        ), fn.__name__

    logits, targets = _saturated_corpus(n=4_000)
    metrics = compute_holdout_value_metrics(
        np.tanh(logits), targets, None, logits,
    )
    assert metrics.calibration["bias"] == 0.0


def test_fit_bias_is_off_by_default_and_opt_in_when_asked():
    logits, targets = _saturated_corpus()
    t, b = fit_logit_temperature(logits, targets)
    assert b == 0.0
    assert t > 1.0
    # Opt-in still works — it is a diagnosis tool, not the gate path.
    _, b_on = fit_logit_temperature(logits, targets, fit_bias=True)
    assert isinstance(b_on, float)


def test_a_constant_head_cannot_calibrate_its_way_past_the_gate():
    """A zero-variance value head must come out of calibration
    UNCHANGED.

    It carries no ranking, so there is no scale to fit — and every
    monotone one-parameter family over a constant input degenerates
    into a fitted intercept (``tanh(c/T)`` sweeps all of ``(0, 1)``),
    which on an imbalanced split lands at the held-out label mean and
    scores ``1 - mean(y)**2 < 1``. That would be a gate pass bought
    with zero information.
    """
    rng = np.random.default_rng(17)
    # Deliberately imbalanced labels, as a small held-out split is.
    targets = np.where(rng.random(400) < 0.6, 1.0, -1.0)
    logits = np.full(targets.shape, 12.0)  # saturated AND constant

    assert fit_logit_temperature(logits, targets) == (1.0, 0.0)
    assert fit_logit_temperature(
        logits, targets, fit_bias=True,
    ) == (1.0, 0.0)
    assert fit_output_shrink(np.tanh(logits), targets) == 1.0

    metrics = compute_holdout_value_metrics(
        np.tanh(logits), targets, None, logits,
    )
    assert metrics.calibrated_value_mse == pytest.approx(
        metrics.value_mse,
    )
    assert metrics.calibrated_value_mse > metrics.constant_zero_mse
    assert not evaluate_value_gate(metrics).passed


def test_fit_logit_temperature_degenerate_inputs_return_identity():
    assert fit_logit_temperature(np.array([1.0]), np.array([1.0])) == (
        1.0, 0.0,
    )
    assert fit_logit_temperature(
        np.zeros(10), np.ones(10),
    ) == (1.0, 0.0)
    assert fit_logit_temperature(
        np.array([np.nan, 1.0]), np.array([1.0, -1.0]),
    ) == (1.0, 0.0)


def test_fit_logit_temperature_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_logit_temperature(np.zeros(3), np.zeros(4))


def test_max_fit_rows_strides_without_changing_the_answer():
    logits, targets = _saturated_corpus(n=60_000)
    full = fit_logit_temperature(logits, targets, max_fit_rows=0)
    strided = fit_logit_temperature(logits, targets, max_fit_rows=5_000)
    assert strided[0] == pytest.approx(full[0], rel=0.1)
    assert strided[1] == pytest.approx(full[1], abs=0.05)


def test_output_shrink_is_the_closed_form_least_squares():
    rng = np.random.default_rng(11)
    targets = rng.choice([-1.0, 1.0], size=5_000)
    preds = np.tanh(4.0 * (0.3 * targets + rng.normal(size=5_000)))
    a = fit_output_shrink(preds, targets)
    expected = float(np.dot(preds, targets) / np.dot(preds, preds))
    assert a == pytest.approx(min(1.0, max(0.0, expected)))
    assert 0.0 <= a <= 1.0
    shrunk_mse = float(np.mean((a * preds - targets) ** 2))
    raw_mse = float(np.mean((preds - targets) ** 2))
    # The fallback is weaker than the temperature fit but never worse
    # than the raw head.
    assert shrunk_mse <= raw_mse


def test_output_shrink_clips_a_sign_flipped_head_to_zero():
    targets = np.array([1.0, -1.0, 1.0, -1.0])
    preds = -targets
    assert fit_output_shrink(preds, targets) == 0.0


def test_output_shrink_degenerate_inputs():
    assert fit_output_shrink(np.zeros(4), np.ones(4)) == 1.0
    assert fit_output_shrink(np.array([]), np.array([])) == 1.0
    assert fit_output_shrink(
        np.array([np.inf, 1.0]), np.array([1.0, 1.0]),
    ) == 1.0
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_output_shrink(np.zeros(3), np.zeros(4))


def test_calibrate_holdout_falls_back_to_shrink_without_logits():
    logits, targets = _saturated_corpus()
    preds = np.tanh(logits)
    calibration, cal_preds = calibrate_holdout(targets, preds=preds)
    assert calibration.method == CALIBRATION_OUTPUT_SHRINK
    assert 0.0 < calibration.scale < 1.0
    assert calibration.calibrated_mse < calibration.raw_mse
    assert np.allclose(cal_preds, calibration.scale * preds)


def test_calibrate_holdout_empty_is_identity():
    calibration, preds = calibrate_holdout(np.zeros((0,)))
    assert calibration.method == CALIBRATION_IDENTITY
    assert calibration.is_identity
    assert preds.shape == (0,)


def test_calibrate_holdout_needs_logits_or_preds():
    with pytest.raises(ValueError, match="needs logits or preds"):
        calibrate_holdout(np.array([1.0, -1.0]))


def test_calibrate_holdout_row_mismatch_raises():
    with pytest.raises(ValueError, match="!= targets rows"):
        calibrate_holdout(np.zeros(4), logits=np.zeros(5))
    with pytest.raises(ValueError, match="!= targets rows"):
        calibrate_holdout(np.zeros(4), preds=np.zeros(5))


def test_apply_logit_temperature_rejects_non_positive_temperature():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and > 0"):
            apply_logit_temperature(np.zeros(3), bad)


def test_apply_calibration_dispatches_and_refuses_the_wrong_input():
    temp = ValueCalibration(
        method=CALIBRATION_LOGIT_TEMPERATURE, temperature=2.0, bias=0.5,
    )
    z = np.array([1.0, -2.0])
    assert np.allclose(
        apply_calibration(temp, logits=z), np.tanh(z / 2.0 + 0.5),
    )
    with pytest.raises(ValueError, match="needs pre-tanh logits"):
        apply_calibration(temp, preds=z)

    shrink = ValueCalibration(
        method=CALIBRATION_OUTPUT_SHRINK, scale=0.25,
    )
    assert np.allclose(apply_calibration(shrink, preds=z), 0.25 * z)
    with pytest.raises(ValueError, match="needs post-tanh preds"):
        apply_calibration(shrink, logits=z)

    ident = ValueCalibration()
    assert np.allclose(apply_calibration(ident, preds=z), z)
    assert np.allclose(apply_calibration(ident, logits=z), np.tanh(z))
    with pytest.raises(ValueError, match="needs logits or preds"):
        apply_calibration(ident)


def test_value_calibration_round_trips_through_json_dict():
    calibration = ValueCalibration(
        method=CALIBRATION_LOGIT_TEMPERATURE,
        temperature=3.25,
        bias=-0.125,
        n_rows=17,
        raw_mse=1.3,
        calibrated_mse=0.9,
        cv_mse=0.91,
    )
    back = ValueCalibration.from_dict(calibration.as_dict())
    assert back == calibration
    assert not back.is_identity


def test_value_calibration_from_missing_dict_is_identity():
    for missing in (None, {}):
        cal = ValueCalibration.from_dict(missing)
        assert cal.is_identity
        assert cal.temperature == 1.0
        assert cal.bias == 0.0
        assert cal.scale == 1.0


def test_as_dict_scrubs_non_finite_floats():
    cal = ValueCalibration(raw_mse=float("nan"), cv_mse=float("inf"))
    d = cal.as_dict()
    assert d["raw_mse"] is None
    assert d["cv_mse"] is None


def test_from_dict_tolerates_garbage_numbers():
    cal = ValueCalibration.from_dict(
        {"method": "logit_temperature", "raw_mse": "not-a-number"},
    )
    assert cal.raw_mse is None


# ---------------------------------------------------------------------
# End-to-end through the gate: the rejected-run signature must flip
# from FAIL to PASS on calibration alone, and a genuinely worthless
# head must still FAIL.
# ---------------------------------------------------------------------


def test_gate_judges_the_calibrated_mse():
    logits, targets = _saturated_corpus()
    preds = np.tanh(logits)

    raw_only = compute_holdout_value_metrics(
        preds, targets, calibrate=False,
    )
    assert raw_only.calibration is None
    assert raw_only.gate_value_mse == raw_only.value_mse
    assert not evaluate_value_gate(raw_only).passed

    calibrated = compute_holdout_value_metrics(
        preds, targets, None, logits,
    )
    assert calibrated.value_mse > 1.0
    assert calibrated.calibrated_value_mse < 1.0
    assert calibrated.gate_value_mse == pytest.approx(
        calibrated.calibration["cv_mse"],
    )
    decision = evaluate_value_gate(calibrated)
    assert decision.passed
    assert "raw" in decision.reason


def test_gate_still_rejects_a_head_with_no_signal_after_calibration():
    rng = np.random.default_rng(5)
    targets = rng.choice([-1.0, 1.0], size=20_000)
    logits = 4.0 * rng.normal(size=targets.shape[0])  # zero correlation
    metrics = compute_holdout_value_metrics(
        np.tanh(logits), targets, None, logits,
    )
    # Calibration can only shrink it toward the constant-0 predictor,
    # which scores exactly the 1.0 ceiling — never under it.
    assert metrics.calibrated_value_mse >= 0.99
    assert not evaluate_value_gate(metrics).passed


def test_metrics_as_dict_is_json_safe_and_carries_the_calibration():
    import json

    logits, targets = _saturated_corpus(n=2_000)
    metrics = compute_holdout_value_metrics(
        np.tanh(logits), targets, None, logits,
    )
    d = metrics.as_dict()
    json.dumps(d)  # must not raise on NaN/Inf
    assert d["calibration"]["method"] == CALIBRATION_LOGIT_TEMPERATURE
    assert d["gate_value_mse"] is not None
    assert d["calibrated_value_mse"] < d["value_mse"]


def test_compute_metrics_rejects_mismatched_logit_rows():
    with pytest.raises(ValueError, match="!= preds rows"):
        compute_holdout_value_metrics(
            np.zeros(4), np.zeros(4), None, np.zeros(5),
        )


def test_registry_set_value_calibration_rewrites_only_metadata(
    tmp_path,
):
    """``mtg_bot calibrate-value``'s storage primitive: metadata in,
    weights untouched."""
    from mtg_bot.training.registry import ModelMetadata, Registry
    from tests._card_vocab_helpers import current_vocab_dict

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-really-a-checkpoint")
    registry = Registry(tmp_path / "reg")
    version = registry.register(
        weights,
        ModelMetadata(
            version="", created_at="", feature_dim=51, hidden_sizes=[],
            architecture="structured",
            card_vocab=current_vocab_dict(),
            training_history={"train_losses": [1.0]},
        ),
    )
    assert registry.load_metadata(version).value_calibration == {}

    registry.set_value_calibration(
        version,
        {"method": CALIBRATION_LOGIT_TEMPERATURE, "temperature": 3.0},
        holdout_metrics={"n_rows": 5},
    )
    reloaded = registry.load_metadata(version)
    assert reloaded.value_calibration["temperature"] == 3.0
    # Pre-existing history survives; the audit trail is additive.
    assert reloaded.training_history["train_losses"] == [1.0]
    assert reloaded.training_history[
        "holdout_value_metrics_recalibrated"
    ] == {"n_rows": 5}
    assert (
        registry.weights_path(version).read_bytes()
        == b"not-really-a-checkpoint"
    )


def test_registry_metadata_round_trips_the_calibration(tmp_path):
    import json

    from mtg_bot.training.registry import ModelMetadata

    meta = ModelMetadata(
        version="v1", created_at="now", feature_dim=51, hidden_sizes=[],
        value_calibration={"method": "logit_temperature", "temperature": 2.0},
    )
    back = ModelMetadata.from_dict(json.loads(json.dumps(meta.to_dict())))
    assert back.value_calibration == meta.value_calibration
    # An old metadata.json with no such key loads as the identity.
    legacy = ModelMetadata.from_dict({
        "version": "v0", "created_at": "then", "feature_dim": 51,
        "hidden_sizes": [],
    })
    assert legacy.value_calibration == {}
    assert "value_calibration" not in legacy.extra
