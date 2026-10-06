"""Unit tests for ``ResolutionChannelAgent`` — the engine-agent
shim that either consumes queued answers during replay or raises
``_ResolutionPause`` to request a new search decision.

Integration with the adapter is covered in
``test_resolution_integration.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtg_bot.search.resolution_channel import (
    KIND_DISCARD, KIND_MODE, KIND_SACRIFICE, KIND_TARGET,
    ResolutionChannelAgent, _ResolutionPause,
)
from mtg_bot.search.resolution_decisions import (
    ResolutionDiscardDecision, ResolutionModeDecision,
    ResolutionSacrificeDecision, ResolutionTargetDecision,
)
from mtg_bot.search.tactical import DefaultTacticalAgent

# The engine always hands the callback its live state; the channel
# agent reads it at pause time (option dedup, slot offsets). These
# unit tests drive the agent directly, so give it an empty stand-in.
_STATE = SimpleNamespace(players=[], battlefield=[], stack=[])


# ---------------------------------------------------------------------
# Exposed callbacks: queue-hit and pause-on-empty
# ---------------------------------------------------------------------

def test_empty_queue_pauses_on_choose_target():
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_target(
            state=_STATE, player_id=3,
            legal_targets=[("creature", 10), ("creature", 11)],
        )
    pause = excinfo.value
    assert pause.kind == KIND_TARGET
    assert pause.player_id == 3
    assert pause.options == (("creature", 10), ("creature", 11))


def test_queue_hit_returns_answer_on_choose_target():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 7)))
    result = agent.choose_target(
        state=_STATE, player_id=0,
        legal_targets=[("creature", 7), ("creature", 8)],
    )
    assert result == ("creature", 7)


def test_empty_queue_pauses_on_choose_discard():
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_discard(
            state=_STATE, player_id=1, hand_card_ids=[99, 100],
        )
    assert excinfo.value.kind == KIND_DISCARD
    assert excinfo.value.options == (99, 100)


def test_queue_hit_returns_answer_on_choose_discard():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionDiscardDecision(card_id=42))
    result = agent.choose_discard(
        state=_STATE, player_id=0, hand_card_ids=[42, 43],
    )
    assert result == 42


def test_empty_queue_pauses_on_choose_mode():
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_mode(
            state=_STATE, player_id=0,
            mode_descriptions=["a", "b", "c"],
        )
    assert excinfo.value.kind == KIND_MODE
    assert excinfo.value.options == (0, 1, 2)


def test_queue_hit_returns_answer_on_choose_mode():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionModeDecision(mode_index=2))
    result = agent.choose_mode(
        state=_STATE, player_id=0, mode_descriptions=["a", "b", "c"],
    )
    assert result == 2


def test_empty_queue_pauses_on_choose_sacrifice():
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_sacrifice(
            state=_STATE, player_id=0, legal_object_ids=[10, 20],
        )
    assert excinfo.value.kind == KIND_SACRIFICE


def test_queue_hit_returns_answer_on_choose_sacrifice():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionSacrificeDecision(object_id=20))
    result = agent.choose_sacrifice(
        state=_STATE, player_id=0, legal_object_ids=[10, 20],
    )
    assert result == 20


# ---------------------------------------------------------------------
# Queue state management
# ---------------------------------------------------------------------

def test_queue_consumes_entries_in_order():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 1)))
    agent.push_answer(ResolutionTargetDecision(target=("creature", 2)))
    first = agent.choose_target(_STATE, 0, [("creature", 1), ("creature", 2)])
    second = agent.choose_target(_STATE, 0, [("creature", 1), ("creature", 2)])
    assert first == ("creature", 1)
    assert second == ("creature", 2)


def test_reset_queue_clears_queue_and_cursor():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 1)))
    agent.choose_target(_STATE, 0, [("creature", 1)])
    agent.reset_queue()
    with pytest.raises(_ResolutionPause):
        agent.choose_target(_STATE, 0, [("creature", 1)])


def test_rewind_preserves_queue_and_restarts_cursor():
    """After rewind, the first callback returns the first queue
    entry again — this is how the adapter replays from a rollback."""
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 1)))
    agent.push_answer(ResolutionTargetDecision(target=("creature", 2)))
    # First play-through: consume both.
    assert agent.choose_target(_STATE, 0, []) == ("creature", 1)
    assert agent.choose_target(_STATE, 0, []) == ("creature", 2)
    # Rewind and replay: should see entries from position 0 again.
    agent.rewind()
    assert agent.choose_target(_STATE, 0, []) == ("creature", 1)
    assert agent.choose_target(_STATE, 0, []) == ("creature", 2)


def test_snapshot_and_restore_preserve_queue_state():
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 5)))
    agent.push_answer(ResolutionDiscardDecision(card_id=77))
    agent.choose_target(_STATE, 0, [])  # advance cursor to 1

    snap = agent.snapshot()

    other = ResolutionChannelAgent()
    other.restore(snap)
    # ``other`` should continue from cursor=1: next callback is
    # the discard.
    assert other.choose_discard(_STATE, 0, [77]) == 77


def test_snapshot_returns_independent_copy():
    """Mutating the snapshot dict must not affect the agent."""
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionTargetDecision(target=("creature", 1)))
    snap = agent.snapshot()
    snap["queue"].append("junk")
    # Agent's own queue is unaffected.
    assert agent._queue == [
        ResolutionTargetDecision(target=("creature", 1))
    ]


# ---------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------

def test_queue_type_mismatch_raises():
    """Queueing a ``DiscardDecision`` but the engine asks for a
    target → TypeError from the channel agent. This indicates a
    bug in the adapter's queue management."""
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionDiscardDecision(card_id=1))
    with pytest.raises(TypeError):
        agent.choose_target(_STATE, 0, [("creature", 1)])


# ---------------------------------------------------------------------
# Remaining 6 exposed callbacks (X, surveil, shock-life, library pick,
# opponent-hand pick, devour lands) — same queue-or-pause behavior
# as the first four.
# ---------------------------------------------------------------------

def test_empty_queue_pauses_on_choose_x():
    from mtg_bot.search.resolution_channel import KIND_X
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_x(
            state=_STATE, player_id=0, card_name="test", upper_bound=3,
        )
    assert excinfo.value.kind == KIND_X
    # Options are every int from 0 to upper_bound inclusive.
    assert excinfo.value.options == (0, 1, 2, 3)


def test_queue_hit_returns_answer_on_choose_x():
    from mtg_bot.search.resolution_decisions import ResolutionXDecision
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionXDecision(x_value=5))
    assert agent.choose_x(_STATE, 0, "test", upper_bound=10) == 5


def test_empty_queue_pauses_on_choose_surveil():
    from mtg_bot.search.resolution_channel import KIND_SURVEIL
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_surveil(_STATE, 0, top_card_id=7)
    assert excinfo.value.kind == KIND_SURVEIL
    assert excinfo.value.options == (True, False)


def test_queue_hit_returns_answer_on_choose_surveil():
    from mtg_bot.search.resolution_decisions import (
        ResolutionSurveilDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionSurveilDecision(mill=True))
    assert agent.choose_surveil(_STATE, 0, top_card_id=42) is True


def test_empty_queue_pauses_on_choose_pay_shock_life():
    from mtg_bot.search.resolution_channel import KIND_SHOCK_LIFE
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_pay_shock_life(_STATE, 0, life_cost=2)
    assert excinfo.value.kind == KIND_SHOCK_LIFE
    assert excinfo.value.options == (True, False)


def test_queue_hit_returns_answer_on_choose_pay_shock_life():
    from mtg_bot.search.resolution_decisions import (
        ResolutionShockLifeDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionShockLifeDecision(pay=True))
    assert agent.choose_pay_shock_life(_STATE, 0, life_cost=2) is True


def test_empty_queue_pauses_on_choose_card_from_library():
    """Library picks include ``None`` (decline) as a legal option."""
    from mtg_bot.search.resolution_channel import KIND_CARD_FROM_LIBRARY
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_card_from_library(_STATE, 0, option_ids=[10, 20])
    assert excinfo.value.kind == KIND_CARD_FROM_LIBRARY
    assert excinfo.value.options == (10, 20, None)


def test_queue_hit_returns_answer_on_choose_card_from_library():
    from mtg_bot.search.resolution_decisions import (
        ResolutionCardFromLibraryDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionCardFromLibraryDecision(card_id=20))
    assert agent.choose_card_from_library(_STATE, 0, [10, 20]) == 20


def test_queue_hit_returns_none_on_declined_library_pick():
    from mtg_bot.search.resolution_decisions import (
        ResolutionCardFromLibraryDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionCardFromLibraryDecision(card_id=None))
    assert agent.choose_card_from_library(_STATE, 0, [10, 20]) is None


def test_empty_queue_pauses_on_choose_card_from_opponent_hand():
    from mtg_bot.search.resolution_channel import (
        KIND_CARD_FROM_OPPONENT_HAND,
    )
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_card_from_opponent_hand(
            _STATE, 0, opponent_hand_ids=[101, 102],
        )
    assert excinfo.value.kind == KIND_CARD_FROM_OPPONENT_HAND
    assert excinfo.value.options == (101, 102)


def test_queue_hit_returns_answer_on_choose_card_from_opponent_hand():
    from mtg_bot.search.resolution_decisions import (
        ResolutionCardFromOpponentHandDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(
        ResolutionCardFromOpponentHandDecision(card_id=102)
    )
    assert agent.choose_card_from_opponent_hand(_STATE, 0, [101, 102]) == 102


def test_empty_queue_pauses_on_choose_devour_lands():
    """Devour: V1 exposes only ``()`` and ``tuple(land_ids)``."""
    from mtg_bot.search.resolution_channel import KIND_DEVOUR_LANDS
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_devour_lands(_STATE, 0, land_ids=[50, 51, 52])
    assert excinfo.value.kind == KIND_DEVOUR_LANDS
    assert excinfo.value.options == ((), (50, 51, 52))


def test_queue_hit_returns_all_lands_on_devour_decision():
    from mtg_bot.search.resolution_decisions import (
        ResolutionDevourLandsDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(
        ResolutionDevourLandsDecision(land_ids=(50, 51, 52))
    )
    result = agent.choose_devour_lands(_STATE, 0, [50, 51, 52])
    assert result == [50, 51, 52]


def test_queue_hit_returns_no_lands_on_devour_decline():
    from mtg_bot.search.resolution_decisions import (
        ResolutionDevourLandsDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionDevourLandsDecision(land_ids=()))
    result = agent.choose_devour_lands(_STATE, 0, [50, 51, 52])
    assert result == []


def test_empty_queue_pauses_on_choose_yes_no():
    from mtg_bot.search.resolution_channel import KIND_YES_NO
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as excinfo:
        agent.choose_yes_no(_STATE, 0, prompt="Take the may?")
    assert excinfo.value.kind == KIND_YES_NO
    assert excinfo.value.options == (True, False)


def test_queue_hit_returns_yes_on_choose_yes_no():
    from mtg_bot.search.resolution_decisions import (
        ResolutionYesNoDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionYesNoDecision(yes=True))
    assert agent.choose_yes_no(_STATE, 0, "Take the may?") is True


def test_queue_hit_returns_no_on_choose_yes_no():
    from mtg_bot.search.resolution_decisions import (
        ResolutionYesNoDecision,
    )
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionYesNoDecision(yes=False))
    assert agent.choose_yes_no(_STATE, 0, "Take the may?") is False


# ---------------------------------------------------------------------
# Still-delegated callbacks: combat declarations + choose_action
# (all 10 decision callbacks are now exposed as resolution decisions,
# but declare_attackers/blockers/choose_action go through the adapter's
# state machines, not through this channel agent's pause mechanism).
# ---------------------------------------------------------------------

def test_declare_attackers_delegates_to_fallback():
    fallback = DefaultTacticalAgent()
    agent = ResolutionChannelAgent(fallback=fallback)

    class _Obj:
        def __init__(self, oid):
            self.id = oid

    eligible = [_Obj(10), _Obj(11)]
    # Default tactical attacks with everything eligible.
    assert agent.declare_attackers(None, 0, eligible) == [10, 11]


def test_choose_action_delegates_to_fallback():
    """The channel agent should NOT handle choose_action; that's
    the adapter's job. Delegate to fallback for safety."""
    from mtg_bot.actions import PassPriority
    fallback = DefaultTacticalAgent()
    agent = ResolutionChannelAgent(fallback=fallback)
    legal = [PassPriority()]
    assert agent.choose_action(_STATE, 0, legal) == PassPriority()
