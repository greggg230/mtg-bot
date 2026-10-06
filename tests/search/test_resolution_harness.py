"""The pause / rollback / replay harness under real play
(``docs/RESOLUTION_DECISIONS.md`` §2): bookkeeping rollback, combat and
constructor capture, per-seat exposure, determinization at a pause and
the canonical / info-set keys."""

from __future__ import annotations

import collections
import random

import numpy as np
import pytest

from mtg_bot.actions import PassPriority
from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.cards.source import get_active_source
from mtg_bot.game import Zone
from mtg_bot.search import engine_adapter as ea_mod
from mtg_bot.search.canonical_state import canonical_state_key
from mtg_bot.search.combat_decisions import AttackerDecision
from mtg_bot.search.determinize import determinize_adapter_in_place
from mtg_bot.search.engine_adapter import (
    MtgGameStateAdapter, resolve_resolution_seats,
)
from mtg_bot.search.info_set import info_set_key
from mtg_bot.search.resolution_channel import (
    KIND_MODES, ResolutionChannelAgent, _ResolutionPause, mode_subsets,
)
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision, ResolutionModesDecision,
    ResolutionTargetDecision, ResolutionYesNoDecision,
)
from mtg_bot.search.tactical import DefaultTacticalAgent
from mtg_bot.simulate import setup_game
from mtg_bot.training import action_encoding as ae
from mtg_bot.turn import Step
from tests._test_decks import simple_test_decklists
from tests.search._resolution_helpers import (
    chaos_adapter, paused_bombard_adapter,
)


# ---------------------------------------------------------------------
# Real-play sweep
# ---------------------------------------------------------------------


def test_random_play_sweep_terminates_with_exact_codec_at_every_pause():
    """Random play with every seat exposed: every game ends (no
    replay loop), and at every pause the canonical_v3 mask is exactly
    the encodable options, encode/decode round-trip, the v2 mask and
    the non-decider's mask are empty."""
    kinds: collections.Counter = collections.Counter()
    for seed in range(6):
        rng = random.Random(seed)
        adapter = chaos_adapter(seed)
        n = 0
        while not adapter.is_terminal() and n < 4000:
            legal = list(adapter.legal_decisions())
            if adapter.is_resolution_paused:
                who = adapter.active_player()
                kinds[adapter.resolution_kind] += 1
                mask = ae.legal_action_mask(adapter, who, version="canonical_v3")
                assert not mask[: ae.SLOT_RESOLUTION_BASE].any()
                assert not ae.legal_action_mask(
                    adapter, who, version="canonical_v2",
                ).any()
                assert not ae.legal_action_mask(
                    adapter, 1 - who, version="canonical_v3",
                ).any()
                encoded = set()
                for d in legal:
                    try:
                        slot = ae.encode(d, adapter, who, version="canonical_v3")
                    except ae.EncodingOverflowError:
                        continue
                    assert mask[slot]
                    if slot in encoded:
                        continue
                    encoded.add(slot)
                    assert ae.decode(slot, adapter, who, version="canonical_v3") == d
                assert encoded == set(np.flatnonzero(mask).tolist())
            adapter.apply(rng.choice(legal))
            n += 1
        assert adapter.is_terminal(), f"seed {seed} did not finish in {n}"
    assert sum(kinds.values()) > 0
    assert len(kinds) >= 3, kinds


# ---------------------------------------------------------------------
# Regression: a pause while the last pass resolves the stack
# ---------------------------------------------------------------------


def test_pause_during_stack_resolving_pass_resolves_after_the_answer():
    """Vaultguard Trooper's end-step "may" pauses while the second
    PassPriority resolves it. Before the bookkeeping rollback the replay
    ran with ``_pass_count`` already reset, so it passed priority again
    instead of resolving -- the trigger stayed on the stack and random
    games looped forever."""
    printings = get_active_source().get_printings()
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=7,
    )
    for name in ("Wild Bear", "Sapling"):
        c = state.new_object(SYNTHETIC_CREATURES[name], 0, Zone.BATTLEFIELD)
        c.summoning_sick = False
        c.tapped = True
        state.battlefield.append(c)
    vg = state.new_object(printings["Vaultguard Trooper"], 0, Zone.BATTLEFIELD)
    vg.summoning_sick = False
    state.battlefield.append(vg)
    for _ in range(3):
        state.players[0].hand.append(
            state.new_object(BASIC_LANDS["Forest"], 0, Zone.HAND),
        )
    adapter = MtgGameStateAdapter(
        state, expose_combat_decisions=False,
        expose_resolution_decisions=True,
    )
    for _ in range(40):
        if adapter.is_resolution_paused:
            break
        adapter.apply(PassPriority())
    else:
        pytest.fail("never reached the yes_no pause")
    assert adapter.resolution_kind == "yes_no"
    assert len(adapter.engine_state.stack) == 1
    adapter.apply(ResolutionYesNoDecision(yes=False))
    assert not adapter.is_resolution_paused
    # The trigger RESOLVED (not merely "priority passed").
    assert len(adapter.engine_state.stack) == 0
    assert adapter._pre_action_clone is None
    assert adapter._pre_action_bookkeeping is None


def test_bookkeeping_snapshot_round_trip():
    adapter = paused_bombard_adapter()
    adapter._pass_count = 1
    adapter._pending_attackers = [5, 6]
    adapter._chosen_blocks = {1: 2}
    snap = adapter._bookkeeping_snapshot()
    adapter._pass_count = 0
    adapter._pending_attackers = []
    adapter._chosen_blocks = {}
    adapter._restore_bookkeeping(snap)
    assert adapter._pass_count == 1
    assert adapter._pending_attackers == [5, 6]
    assert adapter._chosen_blocks == {1: 2}


# ---------------------------------------------------------------------
# Capture beyond priority actions
# ---------------------------------------------------------------------


def _attack_ready_adapter():
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=11,
    )
    for name in ("Wild Bear", "Sapling"):
        c = state.new_object(SYNTHETIC_CREATURES[name], 0, Zone.BATTLEFIELD)
        c.summoning_sick = False
        state.battlefield.append(c)
    return MtgGameStateAdapter(state, expose_resolution_decisions=True)


def test_callback_fired_by_a_factored_attacker_decision_pauses_and_replays(
    monkeypatch,
):
    """A callback raised while the last AttackerDecision finalises the
    declaration (attack triggers, BECOMES_TAPPED triggers) used to
    escape the harness uncaught. It now pauses; the answer replays the
    attacker decision from the snapshot, pending list included."""
    adapter = _attack_ready_adapter()
    for _ in range(60):
        if adapter.combat_phase == "attackers":
            break
        adapter.apply(list(adapter.legal_decisions())[0])
    else:
        pytest.skip("no declare-attackers phase reached")
    calls = {"n": 0}
    real_finalize = MtgGameStateAdapter._finalize_attacker_phase

    def finalize_with_may(self):
        calls["n"] += 1
        ap = self._state.active_player()
        ap.agent.choose_yes_no(self._state, ap.id, "attack trigger?")
        return real_finalize(self)

    monkeypatch.setattr(
        MtgGameStateAdapter, "_finalize_attacker_phase", finalize_with_may,
    )
    while adapter.combat_phase == "attackers":
        head = adapter._pending_attackers[0]
        pending_before = list(adapter._pending_attackers)
        adapter.apply(AttackerDecision(attacker_id=head, attack=True))
        if adapter.is_resolution_paused:
            break
    assert adapter.is_resolution_paused
    assert adapter.resolution_kind == "yes_no"
    # Rolled back: the last attacker is pending again.
    assert adapter.combat_phase == "attackers"
    assert adapter._pending_attackers == pending_before
    adapter.apply(ResolutionYesNoDecision(yes=True))
    assert not adapter.is_resolution_paused
    assert adapter.combat_phase != "attackers"
    assert any(o.attacking for o in adapter.engine_state.battlefield)
    assert calls["n"] == 2  # first attempt + one replay


def test_callback_fired_by_the_constructor_advance_pauses(monkeypatch):
    """The constructor's first forward drive runs inside the harness:
    a pre-priority callback becomes a pause, answered via ``apply``."""
    real_untap = ea_mod.run_untap_step
    fired = {"n": 0}

    def untap_with_may(state):
        fired["n"] += 1
        p = state.players[state.active_player_idx]
        p.agent.choose_yes_no(state, p.id, "untap trigger?")
        return real_untap(state)

    monkeypatch.setattr(ea_mod, "run_untap_step", untap_with_may)
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=5,
    )
    adapter = MtgGameStateAdapter(state, expose_resolution_decisions=True)
    assert adapter.is_resolution_paused
    assert repr(adapter._pending_replay_decision) == "<advance>"
    monkeypatch.setattr(ea_mod, "run_untap_step", real_untap)
    adapter.apply(ResolutionYesNoDecision(yes=False))
    assert not adapter.is_resolution_paused
    assert adapter.engine_state.step not in (None, Step.UNTAP)


# ---------------------------------------------------------------------
# Per-seat exposure
# ---------------------------------------------------------------------


def test_resolve_resolution_seats():
    assert resolve_resolution_seats(True, [0, 1]) == frozenset({0, 1})
    assert resolve_resolution_seats(False, [0, 1]) == frozenset()
    assert resolve_resolution_seats(None, [0, 1]) == frozenset()
    assert resolve_resolution_seats([1], [0, 1]) == frozenset({1})


def test_per_seat_exposure_installs_agents_per_seat_and_survives_clone():
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=2,
    )
    adapter = MtgGameStateAdapter(state, expose_resolution_decisions={1})
    assert adapter.resolution_seats == frozenset({1})
    p0, p1 = adapter.engine_state.players
    assert isinstance(p0.agent, DefaultTacticalAgent)
    assert isinstance(p1.agent, ResolutionChannelAgent)
    clone = adapter.clone()
    c0, c1 = clone.engine_state.players
    assert isinstance(c0.agent, DefaultTacticalAgent)
    assert isinstance(c1.agent, ResolutionChannelAgent)
    assert c1.agent is not p1.agent


def test_unexposed_seat_keeps_auto_targeting():
    """Bombard's caster (P0) is not exposed: casting picks the first
    legal target without a pause."""
    from mtg_bot.actions import CastSpell
    from tests.search._resolution_helpers import (
        bombard_state, drive_to_precombat_main,
    )

    adapter = MtgGameStateAdapter(
        bombard_state(), expose_resolution_decisions={1},
    )
    drive_to_precombat_main(adapter)
    cast = next(
        a for a in adapter.legal_decisions() if isinstance(a, CastSpell)
    )
    adapter.apply(cast)
    assert not adapter.is_resolution_paused
    assert len(adapter.engine_state.stack) == 1


# ---------------------------------------------------------------------
# Hidden information at a pause
# ---------------------------------------------------------------------


def test_determinizing_a_paused_adapter_resyncs_snapshot_and_pins_revealed():
    adapter = paused_bombard_adapter()
    lib = adapter.engine_state.players[0].library
    pinned = [lib[0].id, lib[1].id]
    pinned_printings = {c.id: c.printing.name for c in lib[:2]}
    adapter._resolution_revealed_ids = tuple(pinned)
    for seed in range(5):
        sub = adapter.clone()
        determinize_adapter_in_place(sub, 0, random.Random(seed))
        sub_lib = {c.id: c.printing.name for c in sub.engine_state.players[0].library}
        snap_lib = {
            c.id: c.printing.name
            for c in sub._pre_action_clone.players[0].library
        }
        assert sub_lib == snap_lib  # snapshot follows the determinized world
        for cid in pinned:
            assert sub_lib[cid] == pinned_printings[cid]
    # The source adapter's own snapshot is untouched.
    assert {
        c.id: c.printing.name for c in adapter._pre_action_clone.players[0].library
    } == {c.id: c.printing.name for c in adapter.engine_state.players[0].library}


def test_answer_in_a_determinized_world_rolls_back_inside_that_world():
    adapter = paused_bombard_adapter()
    sub = adapter.clone()
    determinize_adapter_in_place(sub, 0, random.Random(3))
    world_hand = sorted(c.printing.name for c in sub.engine_state.players[1].hand)
    sub.apply(sub.legal_decisions()[0])
    assert not sub.is_resolution_paused
    assert sorted(
        c.printing.name for c in sub.engine_state.players[1].hand
    ) == world_hand


def test_resync_is_a_noop_when_not_paused():
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=2,
    )
    adapter = MtgGameStateAdapter(state, expose_resolution_decisions=True)
    adapter.resync_pause_snapshot()
    assert adapter._pre_action_clone is None


# ---------------------------------------------------------------------
# Keys + introspection
# ---------------------------------------------------------------------


def test_canonical_and_info_set_keys_include_the_replay_queue():
    adapter = paused_bombard_adapter()
    a = adapter.clone()
    b = adapter.clone()
    a._channel_agent.push_answer(ResolutionTargetDecision(target=None))
    assert canonical_state_key(a) != canonical_state_key(b)
    assert info_set_key(a, 0) != info_set_key(b, 0)
    assert a.resolution_queue == (ResolutionTargetDecision(target=None),)


def test_pause_introspection_properties():
    adapter = paused_bombard_adapter()
    assert adapter.resolution_kind == "target"
    assert len(adapter.resolution_options) == len(adapter.resolution_slot_offsets)
    # The decider's live hand is "known at the pause" (pinned by
    # determinizations).
    # (Bombard itself is on the stack at the callback, not in hand.)
    assert set(adapter.resolution_revealed_ids) >= {
        c.id for c in adapter.engine_state.players[0].hand
        if c.printing.name != "Bombard"
    }
    assert adapter.resolution_queue == ()
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=2,
    )
    plain = MtgGameStateAdapter(state)
    assert plain.resolution_queue == ()
    assert not plain.is_resolution_paused


# ---------------------------------------------------------------------
# Channel agent additions
# ---------------------------------------------------------------------


def test_mode_subsets():
    assert mode_subsets(2) == ((0,), (1,), (0, 1))
    assert len(mode_subsets(4)) == 15
    assert mode_subsets(0) == ((0,),)
    big = mode_subsets(8)
    assert big[:8] == tuple((i,) for i in range(8))
    assert big[-1] == tuple(range(8))


def test_choose_modes_pauses_and_replays():
    from types import SimpleNamespace
    state = SimpleNamespace(players=[], battlefield=[], stack=[])
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as exc:
        agent.choose_modes(state, 0, ["a", "b"])
    assert exc.value.kind == KIND_MODES
    assert exc.value.options == ((0,), (1,), (0, 1))
    agent.rewind()
    agent.push_answer(ResolutionModesDecision(mode_indices=(0, 1)))
    assert agent.choose_modes(state, 0, ["a", "b"]) == [0, 1]


def test_non_exposed_callbacks_delegate_to_the_fallback():
    calls = []

    class Recorder(DefaultTacticalAgent):
        def __getattribute__(self, name):
            if name.startswith("choose_") or name.startswith("declare_"):
                calls.append(name)
            return super().__getattribute__(name)

    agent = ResolutionChannelAgent(fallback=Recorder())
    agent.choose_card_from_hand(None, 0, [7, 8])
    agent.choose_permanent_to_tap(None, 0, [3])
    agent.choose_mana_color(None, 0, ["G"])
    agent.choose_convoke_creatures(None, 0, [1], 1, [])
    agent.choose_sac_for_mana(None, 0, [(1, (("C", 1),))], 1, [])
    agent.choose_legend_keep(None, 0, [4, 5])
    agent.choose_bottom(None, 0, [], 0)
    agent.choose_replacement_order([object()], None, 0)
    assert agent.choose_card_from_hand(None, 0, [7, 8]) == 7
    for name in (
        "choose_card_from_hand", "choose_permanent_to_tap",
        "choose_mana_color", "choose_convoke_creatures",
        "choose_sac_for_mana", "choose_legend_keep", "choose_bottom",
        "choose_replacement_order",
    ):
        assert name in calls


def test_devour_with_no_lands_offers_only_none_and_library_reveals():
    from types import SimpleNamespace
    state = SimpleNamespace(players=[], battlefield=[], stack=[])
    agent = ResolutionChannelAgent()
    with pytest.raises(_ResolutionPause) as exc:
        agent.choose_devour_lands(state, 0, [])
    assert exc.value.options == ((),)
    with pytest.raises(_ResolutionPause) as exc:
        agent.choose_card_from_library(state, 0, [11, 12])
    assert exc.value.options == (11, 12, None)
    assert exc.value.revealed_ids == (11, 12)
    with pytest.raises(_ResolutionPause) as exc:
        agent.choose_surveil(state, 0, 33)
    assert exc.value.revealed_ids == (33,)
    pause = _ResolutionPause("x", 0, (0, 1))
    assert pause.slot_offsets == (None, None)
    agent.reset_queue()
    agent.push_answer(ResolutionCardFromLibraryDecision(card_id=None))
    assert agent.choose_card_from_library(state, 0, [11]) is None


# ---------------------------------------------------------------------
# Determinized worlds must replay to the observed pause
# ---------------------------------------------------------------------


def test_replay_reaches_same_pause_in_the_observed_world():
    adapter = paused_bombard_adapter()
    assert adapter.replay_reaches_same_pause()
    # The probe ran on a clone: the adapter is still paused, unchanged.
    assert adapter.is_resolution_paused
    adapter._resolution_options = (("player", 0),)
    assert not adapter.replay_reaches_same_pause()
    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=2,
    )
    assert MtgGameStateAdapter(state).replay_reaches_same_pause()


def test_determinized_clone_resamples_then_falls_back(monkeypatch):
    import importlib
    det = importlib.import_module("mtg_bot.search.determinize")

    adapter = paused_bombard_adapter()
    before = det.world_fallback_counts()
    sub = det.determinized_clone(adapter, 0, random.Random(0))
    assert sub.is_resolution_paused and sub is not adapter
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(MtgGameStateAdapter, "replay_reaches_same_pause", flaky)
    det.determinized_clone(adapter, 0, random.Random(1))
    mid = det.world_fallback_counts()
    assert mid["resampled"] == before["resampled"] + 1
    monkeypatch.setattr(
        MtgGameStateAdapter, "replay_reaches_same_pause", lambda self: False,
    )
    true_world = det.determinized_clone(adapter, 0, random.Random(2), attempts=2)
    after = det.world_fallback_counts()
    assert after["fallback_true_world"] == mid["fallback_true_world"] + 1
    assert after["resampled"] == mid["resampled"] + 2
    # The fallback is an undetermined clone: hidden zones untouched.
    assert [c.printing.name for c in true_world.engine_state.players[1].library] == [
        c.printing.name for c in adapter.engine_state.players[1].library
    ]


def test_determinized_clone_away_from_a_pause():
    import importlib
    det = importlib.import_module("mtg_bot.search.determinize")

    state = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=2,
    )
    adapter = MtgGameStateAdapter(state)
    sub = det.determinized_clone(adapter, 0, random.Random(0))
    assert sub is not adapter and not sub.is_resolution_paused


def test_known_at_pause_covers_hand_options_and_earlier_answers():
    from mtg_bot.search.resolution_decisions import ResolutionDiscardDecision

    adapter = paused_bombard_adapter()
    state = adapter.engine_state
    agent = ResolutionChannelAgent()
    agent.push_answer(ResolutionDiscardDecision(card_id=4242))
    agent._queue_idx = 1  # the earlier answer was consumed
    with pytest.raises(_ResolutionPause) as exc:
        agent.choose_discard(state, 0, [c.id for c in state.players[0].hand][:2])
    known = set(exc.value.revealed_ids)
    assert {c.id for c in state.players[0].hand} <= known
    assert 4242 in known
