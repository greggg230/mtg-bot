"""``canonical_v3``: canonical_v2 + the resolution block
(``docs/RESOLUTION_DECISIONS.md`` §4)."""

from __future__ import annotations

import numpy as np
import pytest

from mtg_bot.actions import PassPriority
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision, ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision, ResolutionDiscardDecision,
    ResolutionModeDecision, ResolutionModesDecision,
    ResolutionSacrificeDecision, ResolutionShockLifeDecision,
    ResolutionSurveilDecision, ResolutionTargetDecision, ResolutionXDecision,
    ResolutionYesNoDecision,
)
from mtg_bot.training import action_encoding as ae
from tests.search._resolution_helpers import paused_bombard_adapter

V2, V3 = ae.ACTION_ENCODING_VERSION_V2, ae.ACTION_ENCODING_VERSION_V3


def test_vocab_sizes_per_version():
    assert ae.action_vocab_size(ae.ACTION_ENCODING_VERSION_V1) == 517
    assert ae.action_vocab_size(V2) == 517
    assert ae.action_vocab_size(V3) == ae.ACTION_VOCAB_SIZE_V3 == 805
    assert ae.SLOT_RESOLUTION_BASE == ae.ACTION_VOCAB_SIZE == 517
    assert not ae.has_resolution_slots(V2)
    assert ae.has_resolution_slots(V3)
    with pytest.raises(ValueError):
        ae.action_vocab_size("canonical_v9")


def test_is_resolution_decision_covers_every_answer_type():
    for d in (
        ResolutionTargetDecision(None), ResolutionDiscardDecision(1),
        ResolutionModeDecision(0), ResolutionModesDecision((0,)),
        ResolutionSacrificeDecision(1), ResolutionXDecision(0),
        ResolutionSurveilDecision(True), ResolutionShockLifeDecision(True),
        ResolutionYesNoDecision(True), ResolutionCardFromLibraryDecision(None),
        ResolutionCardFromOpponentHandDecision(1),
        ResolutionDevourLandsDecision(()),
    ):
        assert ae.is_resolution_decision(d)
        assert ae._resolution_option(d) is not None or isinstance(
            d, (ResolutionTargetDecision, ResolutionCardFromLibraryDecision),
        )
    assert not ae.is_resolution_decision(PassPriority())


def test_mask_at_a_pause_is_exactly_the_option_slots():
    adapter = paused_bombard_adapter()
    mask = ae.legal_action_mask(adapter, 0, version=V3)
    assert mask.shape == (805,)
    assert not mask[:517].any()
    expected = {
        517 + off for off in adapter.resolution_slot_offsets if off is not None
    }
    assert set(np.flatnonzero(mask).tolist()) == expected
    assert len(expected) >= 2
    # v2 has no resolution slots; the non-decider has no decision.
    assert not ae.legal_action_mask(adapter, 0, version=V2).any()
    assert ae.legal_action_mask(adapter, 0, version=V2).shape == (517,)
    assert not ae.legal_action_mask(adapter, 1, version=V3).any()
    assert ae._enumerate_legal(adapter, 1) == []
    assert ae._enumerate_legal(adapter, 0) == list(adapter.legal_decisions())


def test_encode_decode_round_trip_at_a_pause():
    adapter = paused_bombard_adapter()
    for d in adapter.legal_decisions():
        slot = ae.encode(d, adapter, 0, version=V3)
        assert slot >= 517
        assert ae.decode(slot, adapter, 0, version=V3) == d


def test_encode_resolution_errors():
    adapter = paused_bombard_adapter()
    d = adapter.legal_decisions()[0]
    with pytest.raises(ae.EncodingOverflowError, match="no slot under"):
        ae.encode(d, adapter, 0, version=V2)
    with pytest.raises(ae.EncodingOverflowError, match="outside a resolution"):
        ae.encode(d, adapter.engine_state, 0, version=V3)
    with pytest.raises(ae.EncodingOverflowError, match="not the decider"):
        ae.encode(d, adapter, 1, version=V3)
    with pytest.raises(ae.EncodingOverflowError, match="not an option"):
        ae.encode(
            ResolutionTargetDecision(("creature", 987654)), adapter, 0,
            version=V3,
        )
    with pytest.raises(ae.EncodingOverflowError, match="not an option"):
        ae.encode(ResolutionYesNoDecision(True), adapter, 0, version=V3)
    # An option whose slot overflowed at pause time.
    adapter._resolution_slot_offsets = (None,) + tuple(
        adapter._resolution_slot_offsets[1:]
    )
    with pytest.raises(ae.EncodingOverflowError, match="no slot in the"):
        ae.encode(adapter.legal_decisions()[0], adapter, 0, version=V3)


def test_decode_resolution_errors():
    adapter = paused_bombard_adapter()
    slot = ae.encode(adapter.legal_decisions()[0], adapter, 0, version=V3)
    with pytest.raises(ae.IllegalSlotError, match="not the decider"):
        ae.decode(slot, adapter, 1, version=V3)
    with pytest.raises(ae.IllegalSlotError, match="not an option"):
        ae.decode(804, adapter, 0, version=V3)
    with pytest.raises(ae.IllegalSlotError, match="not in"):
        ae.decode(slot, adapter, 0, version=V2)
    with pytest.raises(ae.IllegalSlotError, match="not paused"):
        ae.decode(slot, adapter.engine_state, 0, version=V3)


def test_v3_base_slots_equal_v2_outside_pauses():
    from tests.search._resolution_helpers import chaos_adapter
    import random

    rng = random.Random(1)
    adapter = chaos_adapter(4, expose_resolution_decisions=False)
    checked = 0
    for _ in range(300):
        if adapter.is_terminal():
            break
        who = adapter.active_player()
        m3 = ae.legal_action_mask(adapter, who, version=V3)
        m2 = ae.legal_action_mask(adapter, who, version=V2)
        assert np.array_equal(m3[:517], m2)
        assert not m3[517:].any()
        for d in list(adapter.legal_decisions())[:3]:
            try:
                s2 = ae.encode(d, adapter, who, version=V2)
            except ae.EncodingOverflowError:
                continue
            assert ae.encode(d, adapter, who, version=V3) == s2
            assert ae.decode(s2, adapter, who, version=V3) == ae.decode(
                s2, adapter, who, version=V2,
            )
        checked += 1
        adapter.apply(rng.choice(list(adapter.legal_decisions())))
    assert checked > 50
