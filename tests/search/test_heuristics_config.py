"""Tests for ``mtg_bot.search.heuristics_config.HeuristicsConfig``.

Covers:
- Default construction (production self-play default = all True).
- ``disabled()`` classmethod returns the all-False counterpart.
- ``to_dict`` / ``from_dict`` round-trip.
- ``from_dict`` with missing keys defaults missing flags to True
  (matches ``SelfPlayJobSpec.from_dict``'s prior shape — eval callers
  build on top of ``disabled().to_dict()`` instead).
- Frozen instances reject mutation.
"""

from __future__ import annotations

import pytest

from mtg_bot.search.heuristics_config import HeuristicsConfig


def test_default_construction_all_true():
    cfg = HeuristicsConfig()
    assert cfg.phase_prune_enabled is True
    assert cfg.force_landdrop_enabled is True
    assert cfg.skip_empty_combat_enabled is True
    assert cfg.skip_opp_postcombat_main_enabled is True
    assert cfg.skip_dead_main1_enabled is True


def test_disabled_returns_all_false():
    cfg = HeuristicsConfig.disabled()
    assert cfg.phase_prune_enabled is False
    assert cfg.force_landdrop_enabled is False
    assert cfg.skip_empty_combat_enabled is False
    assert cfg.skip_opp_postcombat_main_enabled is False
    assert cfg.skip_dead_main1_enabled is False


def test_to_dict_keys_match_field_names():
    cfg = HeuristicsConfig.disabled()
    expected_keys = {
        "phase_prune_enabled",
        "force_landdrop_enabled",
        "skip_empty_combat_enabled",
        "skip_opp_postcombat_main_enabled",
        "skip_dead_main1_enabled",
        # B3-S1: the single atomic-revert decision-source flip knob
        # travels with the heuristics through the same serialization.
        "decision_source",
        # 2026-10-06: the searched-resolution-decisions switch.
        "search_resolution_decisions",
    }
    assert set(cfg.to_dict().keys()) == expected_keys


def test_to_dict_from_dict_round_trip_default():
    cfg = HeuristicsConfig()
    rebuilt = HeuristicsConfig.from_dict(cfg.to_dict())
    assert rebuilt == cfg


def test_to_dict_from_dict_round_trip_disabled():
    cfg = HeuristicsConfig.disabled()
    rebuilt = HeuristicsConfig.from_dict(cfg.to_dict())
    assert rebuilt == cfg


def test_to_dict_from_dict_round_trip_mixed():
    cfg = HeuristicsConfig(
        phase_prune_enabled=False,
        force_landdrop_enabled=True,
        skip_empty_combat_enabled=False,
        skip_opp_postcombat_main_enabled=True,
        skip_dead_main1_enabled=False,
    )
    rebuilt = HeuristicsConfig.from_dict(cfg.to_dict())
    assert rebuilt == cfg


def test_from_dict_empty_returns_all_true():
    """Missing keys default to True — production self-play default.
    Eval-default callers build on top of ``disabled().to_dict()``."""
    cfg = HeuristicsConfig.from_dict({})
    assert cfg == HeuristicsConfig()


def test_from_dict_missing_key_defaults_to_true():
    cfg = HeuristicsConfig.from_dict({
        "phase_prune_enabled": False,
        # other keys omitted
    })
    assert cfg.phase_prune_enabled is False
    assert cfg.force_landdrop_enabled is True
    assert cfg.skip_empty_combat_enabled is True
    assert cfg.skip_opp_postcombat_main_enabled is True
    assert cfg.skip_dead_main1_enabled is True


def test_from_dict_disabled_overlay_pattern():
    """Eval callers can rebuild eval defaults by overlaying onto
    ``disabled().to_dict()``."""
    base = HeuristicsConfig.disabled().to_dict()
    base.update({"phase_prune_enabled": True})
    cfg = HeuristicsConfig.from_dict(base)
    assert cfg.phase_prune_enabled is True
    assert cfg.force_landdrop_enabled is False


def test_from_dict_coerces_non_bool_truthy():
    """``bool(...)`` coercion mirrors prior per-flag guards in the
    job specs — JSON might surface 0/1 ints."""
    cfg = HeuristicsConfig.from_dict({
        "phase_prune_enabled": 1,
        "force_landdrop_enabled": 0,
        "skip_empty_combat_enabled": "yes",
        "skip_opp_postcombat_main_enabled": "",
        "skip_dead_main1_enabled": None,
    })
    assert cfg.phase_prune_enabled is True
    assert cfg.force_landdrop_enabled is False
    assert cfg.skip_empty_combat_enabled is True
    assert cfg.skip_opp_postcombat_main_enabled is False
    assert cfg.skip_dead_main1_enabled is False


def test_frozen_rejects_mutation():
    cfg = HeuristicsConfig()
    with pytest.raises(Exception):
        # FrozenInstanceError is the dataclass-specific subtype but
        # AttributeError also satisfies "mutation is rejected".
        cfg.phase_prune_enabled = False  # type: ignore[misc]


def test_equal_instances_compare_equal():
    a = HeuristicsConfig(phase_prune_enabled=False)
    b = HeuristicsConfig(phase_prune_enabled=False)
    assert a == b


def test_disabled_differs_from_default():
    assert HeuristicsConfig() != HeuristicsConfig.disabled()
