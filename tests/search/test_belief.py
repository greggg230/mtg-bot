"""Tests for ``mtg_bot.search.belief.BeliefWeightedDeterminizer``.

Phase 6 of the ISMCTS migration. The sampler is a particle-filter
posterior over opponent hidden state given the action history. These
tests pin the behaviors that flow from that:

- Empty history → reduces to uniform ``determinize``.
- Likelihood scoring favors candidates consistent with observed
  actions (high prob under the NN beats low prob).
- Long histories don't underflow log-sum-exp.
- The (printings, history-index) cache short-circuits repeat NN
  queries.
- Zero-probability actions (e.g. action illegal under a candidate
  determinization) don't annihilate the particle — the floor keeps
  it alive at a heavy penalty.
- Callback contract: returns an ``EngineGameState``, viewer's hand
  preserved, opponent's hand printings come from the deck-list-
  consistent unknown pool.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Optional

import numpy as np
import pytest

from mtg_bot.agents import RandomAgent
from mtg_bot.actions import PassPriority
from mtg_bot.search.belief import (
    BeliefWeightedDeterminizer, PriorityMoment,
    _categorical_draw, _effective_sample_size, _normalize_log_weights,
    _resample_particles, _slot_log_prob,
)
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.simulate import setup_game
from mtg_bot.training import action_encoding as ae
from tests._test_decks import simple_test_decklists  # noqa: E402


# ---------- helpers ---------------------------------------------------------


def _engine_state(seed: int = 42):
    """Fresh engine state from the synthetic G/R deck pair."""
    return setup_game([RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=seed,
    )


def _adapter(seed: int = 42) -> MtgGameStateAdapter:
    # Disable training-time priority short-circuits so belief search
    # tests exercise the actual decision branching they're designed
    # for instead of being short-circuited at the root.
    return MtgGameStateAdapter(
        _engine_state(seed), heuristics=HeuristicsConfig.disabled(),
    )


def _pick_rare_hand_printing(state, player_id: int) -> Optional[str]:
    """Pick a printing in ``player_id``'s hand that occurs only once
    in the hand+library pool. Used by the likelihood-favoring tests
    to move the posterior measurably away from the uniform prior.
    Returns ``None`` if no singleton is available in the hand.
    """
    pool = Counter()
    for c in state.players[player_id].hand:
        pool[c.printing.name] += 1
    for c in state.players[player_id].library:
        pool[c.printing.name] += 1
    for c in state.players[player_id].hand:
        if pool[c.printing.name] == 1:
            return c.printing.name
    return None


def _drive_to_opponent_moments(
    adapter: MtgGameStateAdapter,
    opponent: int,
    max_steps: int = 80,
    multi_action_only: bool = False,
) -> list[PriorityMoment]:
    """Walk the adapter forward driving each priority moment with the
    first legal action and capturing pre-action snapshots whenever
    ``opponent`` had priority. Returns the captured priority-moment
    log; the adapter is left at whatever state it advanced into.

    With ``multi_action_only=True`` only moments where the opponent
    had a non-trivial choice (>1 legal decision) are recorded — the
    likelihood is degenerate at forced moves and they don't help the
    posterior, so the likelihood-favoring tests filter to those.
    """
    log: list[PriorityMoment] = []
    for _ in range(max_steps):
        if adapter.is_terminal():
            break
        legal = adapter.legal_decisions()
        if not legal:
            break
        actor = adapter.active_player()
        decision = legal[0]
        if actor == opponent and (
            not multi_action_only or len(legal) > 1
        ):
            log.append(
                PriorityMoment(
                    state=adapter.engine_state.clone(),
                    decision=decision,
                    actor=actor,
                )
            )
        adapter.apply(decision)
    return log


class _UniformPolicyEvaluator:
    """Returns zero-logits + zero-value. Combined with the legal
    mask, that's a uniform-over-legal policy. Lets us test the
    plumbing without involving torch / a trained model.
    """

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_logits(self, state, player_id):
        self.calls += 1
        return 0.0, np.zeros(ae.ACTION_VOCAB_SIZE, dtype=np.float32)


class _DeterministicSlotEvaluator:
    """Returns logits that put high mass on a chosen slot for a
    chosen-printing-tuple key, low mass elsewhere.

    Used for the likelihood-favors-consistent-hand test: configure
    the evaluator to score "PassPriority" highly when the opponent's
    hand contains a particular printing tuple, and uniformly
    otherwise. The posterior should then concentrate on candidate
    determinizations matching that tuple.
    """

    def __init__(
        self,
        favored_printing_names: Optional[set[str]] = None,
        favored_slot: int = ae.encode(
            PassPriority(), _engine_state(0), 0,
        ),
        boost: float = 10.0,
    ) -> None:
        self.favored_printing_names = favored_printing_names or set()
        self.favored_slot = favored_slot
        self.boost = boost
        self.calls = 0

    def evaluate_logits(self, state, player_id):
        self.calls += 1
        logits = np.zeros(ae.ACTION_VOCAB_SIZE, dtype=np.float32)
        # The real ``NNEvaluator.evaluate_logits`` takes a GameState
        # protocol object (the adapter); ``BeliefWeightedDeterminizer``
        # passes the same. Reach through to the engine state for hand
        # access.
        engine = getattr(state, "engine_state", state)
        hand = engine.players[player_id].hand
        names = {c.printing.name for c in hand}
        if self.favored_printing_names <= names:
            logits[self.favored_slot] = self.boost
        return 0.0, logits


# ---------- input validation ------------------------------------------------


def test_belief_rejects_zero_m_candidates():
    with pytest.raises(ValueError, match="m_candidates"):
        BeliefWeightedDeterminizer(
            opponent_policy_evaluator=_UniformPolicyEvaluator(),
            m_candidates=0,
        )


def test_belief_rejects_zero_cache_size():
    with pytest.raises(ValueError, match="cache_max_size"):
        BeliefWeightedDeterminizer(
            opponent_policy_evaluator=_UniformPolicyEvaluator(),
            cache_max_size=0,
        )


def test_belief_rejects_evaluator_without_evaluate_logits():
    class NoLogits:
        def evaluate(self, state, player_id):
            return 0.0

    with pytest.raises(ValueError, match="evaluate_logits"):
        BeliefWeightedDeterminizer(opponent_policy_evaluator=NoLogits())


# ---------- empty-history reduction to uniform ------------------------------


def test_empty_history_matches_uniform_distribution():
    """With no history the posterior IS the prior: weights are
    uninformative, the chosen particle is determined entirely by
    ``rng``. Verify that across many samples the distribution of
    sampled opponent hands matches what the uniform sampler produces
    from the same seeds.
    """
    state = _engine_state(seed=42)
    viewer = 0
    opponent = 1

    # Capture the multiset of opponent hand printings under the
    # belief sampler with no history vs the uniform sampler, for
    # 50 independent rngs. They should match exactly because both
    # collapse to ``determinize`` with the same caller-supplied rng.
    from mtg_bot.search.determinize import determinize as uniform

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=[],
        m_candidates=8,
    )
    belief_results = []
    uniform_results = []
    for seed in range(50):
        # Identical input states for both samplers.
        s_b = state.clone()
        s_u = state.clone()
        sampler(s_b, viewer, random.Random(seed), in_place=True)
        uniform(s_u, viewer, random.Random(seed), in_place=True)
        belief_results.append(
            tuple(c.printing.name for c in s_b.players[opponent].hand)
        )
        uniform_results.append(
            tuple(c.printing.name for c in s_u.players[opponent].hand)
        )

    # The empty-history fast path bypasses the particle filter and
    # reduces to uniform ``determinize`` with the caller's rng — so
    # the two streams should agree exactly.
    assert belief_results == uniform_results
    # And the result should be non-trivial: at least 5 distinct
    # opponent hand multisets across 50 samples.
    assert len(set(belief_results)) >= 5


def test_empty_history_ess_equals_m_candidates():
    state = _engine_state(seed=42)
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=[],
        m_candidates=16,
    )
    sampler(state.clone(), viewer_id=0, rng=random.Random(0))
    # Empty history → no particle filter is run; ess defaults to N.
    assert sampler.last_ess == 16
    assert sampler.last_evaluator_calls == 0


# ---------- likelihood scoring favors consistent hands ----------------------


def test_likelihood_scoring_concentrates_on_consistent_hand():
    """Build a state with a one-step opponent priority moment and
    ``PassPriority`` as the action taken. Configure the evaluator so
    that ``PassPriority`` gets boosted logits when the opponent's
    hand contains a target printing — and uniform otherwise.

    Sample many times under the belief filter and verify that the
    target-printing frequency in the sampled opponent hand is higher
    than under uniform sampling.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    viewer = 0
    log = _drive_to_opponent_moments(
        adapter, opponent, max_steps=60, multi_action_only=True,
    )
    if not log:
        pytest.skip("no opponent multi-action priority moments under seed=11")

    # Pick a printing the opponent currently holds that's RARE in
    # the pool (1 copy in hand+library) — that way the prior puts
    # low mass on candidate worlds that include it, and the boost
    # has room to move the posterior measurably above the prior.
    target_name = _pick_rare_hand_printing(
        log[0].state, opponent,
    )
    if target_name is None:
        pytest.skip("no rare-singleton printing in opp hand at this moment")

    favored_evaluator = _DeterministicSlotEvaluator(
        favored_printing_names={target_name},
        boost=10.0,
    )
    # Use the HISTORICAL state itself as the source — that aligns
    # the per-particle seed's "hand at historical moment" with its
    # "hand at current moment" so the posterior over opponent hand
    # composition is observable. Without this, the seed's effect on
    # historical scoring is decorrelated from its effect on the
    # returned current-state hand (different hidden pools = different
    # permutation results).
    source_state = log[0].state
    # Fresh sampler per trial: the production posterior cache
    # (constant played_history → fixed particles across iterations)
    # is correct for ISMCTS but collapses all 80 trials onto one
    # particle set when the sampler is reused. This test wants
    # independent samples; create a new sampler each trial.
    posterior_with_target = 0
    n_trials = 80
    for trial in range(n_trials):
        sampler = BeliefWeightedDeterminizer(
            opponent_policy_evaluator=favored_evaluator,
            played_history=log[:1],
            m_candidates=64,
            # Disable the conditional-activation gate; this test
            # exercises the likelihood-separation property even with
            # a single opponent moment, which is below the default
            # threshold of 3.
            min_opponent_moments=0,
        )
        s = source_state.clone()
        sampler(s, viewer, random.Random(1000 + trial), in_place=True)
        names = {c.printing.name for c in s.players[opponent].hand}
        if target_name in names:
            posterior_with_target += 1

    # Uniform-prior baseline: same setup but no history. The target
    # printing's frequency should be the prior probability.
    prior_with_target = 0
    for trial in range(n_trials):
        sampler_prior = BeliefWeightedDeterminizer(
            opponent_policy_evaluator=favored_evaluator,
            played_history=[],
            m_candidates=64,
            min_opponent_moments=0,
        )
        s = source_state.clone()
        sampler_prior(
            s, viewer, random.Random(1000 + trial), in_place=True,
        )
        names = {c.printing.name for c in s.players[opponent].hand}
        if target_name in names:
            prior_with_target += 1

    # Posterior should land the target in the opponent's hand
    # measurably more often than the prior — by at least a 5pp
    # margin. The boost is heavy (logit=10 vs 0); even with a
    # uniform-when-not-favored fallback, the ratio of posterior
    # weight on hands-with-target vs hands-without should be
    # ~exp(slot_log_p_with_target - slot_log_p_without) per moment.
    posterior_rate = posterior_with_target / n_trials
    prior_rate = prior_with_target / n_trials
    assert posterior_rate > prior_rate + 0.05, (
        f"posterior ({posterior_rate:.2%}) should exceed prior "
        f"({prior_rate:.2%}) by at least 5pp when likelihood "
        f"favors hands containing {target_name!r}"
    )


def test_score_particle_log_likelihood_separates_consistent_vs_inconsistent():
    """Direct unit test on the per-particle log-likelihood: a hand
    where the favored printing appears yields a higher log-weight
    than one where it doesn't.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    viewer = 0
    log = _drive_to_opponent_moments(
        adapter, opponent, max_steps=60, multi_action_only=True,
    )
    if not log:
        pytest.skip("no opponent multi-action moments under seed=11")

    target = _pick_rare_hand_printing(log[0].state, opponent)
    if target is None:
        pytest.skip("no rare-singleton printing in opp hand at this moment")
    favored = _DeterministicSlotEvaluator(
        favored_printing_names={target}, boost=10.0,
    )
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=favored,
        played_history=log[:1],
        m_candidates=4,
    )

    # Try many seeds and partition log-likelihoods by whether the
    # candidate determinization places ``target`` in the opponent's
    # hand at the historical moment. Bigger log-weight on the
    # "target-in-hand" stratum.
    from mtg_bot.search.determinize import determinize

    with_target_logws = []
    without_target_logws = []
    for seed in range(200):
        # Reproduce exactly the per-moment determinization
        # ``_score_particle`` would do for ``seed`` at h_idx=0.
        moment = log[0]
        reconstructed = moment.state.clone()
        # Mirror BeliefWeightedDeterminizer's per-moment seed: at
        # h_idx=0 the seed is unmixed so the moment's permutation
        # matches the current-state permutation we'd resample to.
        determinize(
            reconstructed, viewer,
            random.Random(seed), in_place=True,
        )
        names = {
            c.printing.name
            for c in reconstructed.players[opponent].hand
        }
        logw = sampler._score_particle(
            adapter.engine_state.clone(), viewer, seed,
        )
        if target in names:
            with_target_logws.append(logw)
        else:
            without_target_logws.append(logw)

    if not with_target_logws or not without_target_logws:
        pytest.skip("seed sweep didn't sample both strata")
    avg_with = float(np.mean(with_target_logws))
    avg_without = float(np.mean(without_target_logws))
    # The boost is logit=10 vs uniform-zero on the favored slot, so
    # log P(slot|favored) - log P(slot|uniform) ≈ 10 - log K (where K
    # is the legal-action count). Even with realistic K (~5-15) the
    # ratio should be strongly positive.
    # margin is K-sensitive (smaller legal sets => smaller absolute logit gap); 0.5 keeps a meaningful directional check without false-failing on dedup-shrunk K.
    assert avg_with > avg_without + 0.5, (
        f"avg log-weight with target ({avg_with:.3f}) should exceed "
        f"avg log-weight without ({avg_without:.3f}) by >0.5; "
        f"diff={avg_with - avg_without:.3f}"
    )


# ---------- numerical hygiene -----------------------------------------------


def test_long_history_does_not_underflow():
    """30 priority moments × ~log(uniform-over-K) per moment yields a
    log-likelihood of order -tens. The resampler must not produce
    NaN / Inf weights and the normalized vector must sum to 1.
    """
    adapter = _adapter(seed=37)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=200)
    if len(log) < 5:
        pytest.skip("not enough opponent priority moments at seed=37")
    long_log = log[:30] if len(log) >= 30 else log

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=long_log,
        m_candidates=8,
    )

    # ``__call__`` should run and produce a valid state.
    s = adapter.engine_state.clone()
    out = sampler(s, viewer_id=0, rng=random.Random(7), in_place=True)
    assert out is s
    # Last ess should be in [1, m_candidates] — finite.
    assert math.isfinite(sampler.last_ess)
    assert 0.0 < sampler.last_ess <= 8.0


def test_normalize_log_weights_handles_minus_inf():
    log_weights = np.array(
        [0.0, -math.inf, -math.inf, -1.0], dtype=np.float64,
    )
    p = _normalize_log_weights(log_weights)
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
    # The two -inf entries should get zero weight.
    assert p[1] == 0.0
    assert p[2] == 0.0
    # Index 0 (highest log-weight) should get more mass than index 3.
    assert p[0] > p[3]


def test_normalize_log_weights_all_minus_inf_falls_back_uniform():
    log_weights = np.array(
        [-math.inf, -math.inf, -math.inf], dtype=np.float64,
    )
    p = _normalize_log_weights(log_weights)
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
    # Uniform fallback: every entry equals 1/N.
    assert all(abs(x - 1.0 / 3.0) < 1e-9 for x in p)


def test_normalize_log_weights_large_negative_values():
    """Log-weights of order -60 each shouldn't underflow because of
    the log-sum-exp shift."""
    log_weights = np.array(
        [-60.0, -61.0, -62.0, -63.0], dtype=np.float64,
    )
    p = _normalize_log_weights(log_weights)
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
    assert all(0.0 <= x <= 1.0 for x in p)
    assert p[0] > p[3]


def test_effective_sample_size_uniform_equals_n():
    p = np.full(8, 1.0 / 8, dtype=np.float64)
    assert math.isclose(_effective_sample_size(p), 8.0, abs_tol=1e-9)


def test_effective_sample_size_concentrated_equals_one():
    p = np.zeros(8, dtype=np.float64)
    p[3] = 1.0
    assert math.isclose(_effective_sample_size(p), 1.0, abs_tol=1e-9)


# ---------- caching ---------------------------------------------------------


def test_cache_hits_reduce_evaluator_calls():
    """Two ``__call__`` invocations with the same input should reuse
    the cache and dramatically cut the second call's evaluator-call
    count.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)
    if not log:
        pytest.skip("no opponent priority moments under seed=11")

    evaluator = _UniformPolicyEvaluator()
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=evaluator,
        played_history=log[:3],
        m_candidates=16,
    )

    # First call: cold cache.
    s1 = adapter.engine_state.clone()
    sampler(s1, viewer_id=0, rng=random.Random(123), in_place=True)
    first_call_count = sampler.last_evaluator_calls
    assert first_call_count > 0

    # Second call: SAME seed → same particle seeds → same cache
    # keys. Should be served almost entirely from cache.
    s2 = adapter.engine_state.clone()
    sampler(s2, viewer_id=0, rng=random.Random(123), in_place=True)
    second_call_count = sampler.last_evaluator_calls
    assert second_call_count == 0, (
        f"cache should fully cover identical seed; got "
        f"{second_call_count} fresh evaluator calls"
    )


def test_cache_eviction_bounded():
    """With cache_max_size set tight, the cache shouldn't grow
    unboundedly across many calls.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)
    if not log:
        pytest.skip("no opponent priority moments under seed=11")

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log[:2],
        m_candidates=4,
        cache_max_size=8,
    )
    for trial in range(20):
        s = adapter.engine_state.clone()
        sampler(
            s, viewer_id=0,
            rng=random.Random(trial * 31), in_place=True,
        )
    # Cache must respect the bound.
    assert len(sampler._cache) <= 8


# ---------- zero-prob action floor ------------------------------------------


def test_slot_log_prob_floor_for_illegal_slot():
    logits = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Mask says slot 1 is illegal.
    mask = np.array([True, False, True], dtype=bool)
    floor = math.log(1e-12)
    p = _slot_log_prob(logits, mask, slot=1, floor=floor)
    assert p == floor


def test_slot_log_prob_floor_for_out_of_range_slot():
    logits = np.array([0.5, 0.5], dtype=np.float32)
    mask = np.array([True, True], dtype=bool)
    floor = -25.0
    assert _slot_log_prob(logits, mask, slot=99, floor=floor) == floor
    assert _slot_log_prob(logits, mask, slot=-1, floor=floor) == floor


def test_zero_prob_action_does_not_eliminate_particle():
    """An action whose slot is masked illegal under the candidate
    determinization should yield ``log_prob_floor``, not -inf — the
    particle survives with a heavy but finite penalty.
    """
    # Build a synthetic test directly through ``_score_particle``:
    # construct a PriorityMoment whose decision can't be encoded in
    # the candidate's reconstructed state. The simplest trigger is
    # an unencodable card_id.
    from mtg_bot.actions import CastSpell

    adapter = _adapter(seed=11)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)
    if not log:
        pytest.skip("no opponent priority moments under seed=11")

    # Replace the first moment's decision with a CastSpell against a
    # non-existent card id. encode() will raise; the floor catches it.
    bogus_moment = PriorityMoment(
        state=log[0].state.clone(),
        decision=CastSpell(card_id=10**9),
        actor=opponent,
    )

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=[bogus_moment],
        m_candidates=4,
        log_prob_floor=math.log(1e-9),
    )
    floor = math.log(1e-9)
    log_w = sampler._score_particle(
        adapter.engine_state.clone(), viewer_id=0, seed=42,
    )
    # All particles get the floor; total log-weight == 1 * floor.
    assert math.isclose(log_w, floor, abs_tol=1e-9)
    # The sampler still runs (no NaN, no exception).
    s = adapter.engine_state.clone()
    sampler(s, viewer_id=0, rng=random.Random(0), in_place=True)


# ---------- determinizer callback contract ----------------------------------


def test_callback_returns_engine_game_state():
    """The callback must return an ``EngineGameState`` (matching the
    uniform determinizer's return type) so ISMCTS can drop it in.
    """
    adapter = _adapter(seed=11)
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=[],
    )
    out = sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(0), in_place=True,
    )
    from mtg_bot.game import GameState
    assert isinstance(out, GameState)


def test_callback_preserves_viewer_hand():
    """The viewer's own hand is known to them — printings must NOT
    be permuted by determinization. Same invariant the uniform
    sampler holds; verify the belief sampler upholds it too.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)

    viewer = 0
    pre_hand = [
        (c.id, c.printing.name)
        for c in adapter.engine_state.players[viewer].hand
    ]
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log,
        m_candidates=4,
    )
    s = adapter.engine_state.clone()
    sampler(s, viewer_id=viewer, rng=random.Random(0), in_place=True)
    post_hand = [
        (c.id, c.printing.name)
        for c in s.players[viewer].hand
    ]
    assert pre_hand == post_hand


def test_callback_opponent_hand_drawn_from_deck_pool():
    """The opponent's resampled hand printings must be a subset of
    the deck-list-consistent unknown pool — i.e. printings that
    actually exist in the opponent's hand+library before
    determinization.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    viewer = 0
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)

    pool = Counter()
    for c in adapter.engine_state.players[opponent].hand:
        pool[c.printing.name] += 1
    for c in adapter.engine_state.players[opponent].library:
        pool[c.printing.name] += 1

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log[:2],
        m_candidates=8,
    )
    for trial in range(10):
        s = adapter.engine_state.clone()
        sampler(
            s, viewer_id=viewer,
            rng=random.Random(trial), in_place=True,
        )
        post_hand = Counter(
            c.printing.name for c in s.players[opponent].hand
        )
        post_lib = Counter(
            c.printing.name for c in s.players[opponent].library
        )
        # hand+library multiset equals the pre-state pool by
        # construction (determinize permutes; doesn't add or drop).
        combined = post_hand + post_lib
        assert combined == pool


def test_callback_engine_state_consistent_no_share_groups():
    """Successive ``__call__`` invocations must not corrupt the
    source engine state via CoW share-group bugs (same hazard the
    uniform sampler defends against in determinize._determinize_player).
    """
    adapter = _adapter(seed=11)
    opponent = 1
    viewer = 0
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log[:1],
        m_candidates=4,
    )
    # Snapshot the source state's library multiset.
    source_lib_pre = Counter(
        c.printing.name
        for c in adapter.engine_state.players[opponent].library
    )
    for trial in range(15):
        s = adapter.engine_state.clone()
        sampler(
            s, viewer_id=viewer,
            rng=random.Random(trial * 7), in_place=True,
        )
    source_lib_post = Counter(
        c.printing.name
        for c in adapter.engine_state.players[opponent].library
    )
    # The source state's library multiset must be unchanged.
    assert source_lib_pre == source_lib_post


def test_callback_in_place_false_leaves_input_untouched():
    """``in_place=False`` should clone before mutating, so the input
    state is unaffected.
    """
    adapter = _adapter(seed=11)
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=[],
    )
    src = adapter.engine_state.clone()
    pre_opp_hand_ids = [c.id for c in src.players[1].hand]
    out = sampler(src, viewer_id=0, rng=random.Random(0), in_place=False)
    assert out is not src
    post_opp_hand_ids = [c.id for c in src.players[1].hand]
    # Source IDs unchanged. (Printings would be unchanged anyway in
    # the in_place=False case — this just verifies we didn't trash
    # the source.)
    assert pre_opp_hand_ids == post_opp_hand_ids


# ---------- categorical-draw helper -----------------------------------------


def test_categorical_draw_distribution():
    """Sample many times from a known distribution and verify
    empirical frequencies match within a reasonable tolerance.
    """
    p = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    rng = random.Random(0)
    counts = Counter()
    for _ in range(2000):
        idx = _categorical_draw(p, rng)
        counts[idx] += 1
    freq = {k: v / 2000 for k, v in counts.items()}
    assert abs(freq.get(0, 0) - 0.7) < 0.05
    assert abs(freq.get(1, 0) - 0.2) < 0.05
    assert abs(freq.get(2, 0) - 0.1) < 0.05


def test_categorical_draw_concentrated_distribution():
    p = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rng = random.Random(42)
    for _ in range(100):
        assert _categorical_draw(p, rng) == 1


# ---------- runtime-integration regressions --------------------------------
#
# Both regressions caught by the smoke run on 2026-05-04 right after the
# Phase 6.5 wireup landed. The agent's existing tests passed because they
# used a mock evaluator that didn't call ``state.is_terminal()`` and the
# cache-key path never hit a real ``StaticAbility``-bearing printing. The
# 3-way bench at production budget exercises both at game 1's first
# decision under the synthetic G/R deck.


class _ProtocolEvaluator:
    """Mimics ``NNEvaluator.evaluate_logits``: calls ``state.is_terminal()``
    and ``state.winner()`` on the input — i.e. requires the GameState
    protocol (the adapter), not a raw engine state.
    """

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_logits(self, state, player_id):
        self.calls += 1
        # The real evaluator does this exact check first thing.
        if state.is_terminal():
            return 0.0, np.zeros(ae.ACTION_VOCAB_SIZE, dtype=np.float32)
        # Touch winner() too; some real evaluators do.
        _ = state.winner()
        return 0.0, np.zeros(ae.ACTION_VOCAB_SIZE, dtype=np.float32)


def test_score_particle_passes_adapter_to_evaluator():
    """Regression: the belief sampler used to pass the raw engine
    state to ``evaluate_logits``, which then crashed with
    ``AttributeError: 'GameState' object has no attribute
    'is_terminal'``. The fix wraps the reconstructed state in
    ``MtgGameStateAdapter`` before the evaluator call.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=20)
    if not log:
        pytest.skip("no opponent priority moments under seed=11")

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_ProtocolEvaluator(),
        played_history=log[:2],
        m_candidates=4,
    )
    s = adapter.engine_state.clone()
    # Should not raise.
    sampler(s, viewer_id=0, rng=random.Random(99), in_place=True)


def test_opponent_hand_key_is_hashable_for_real_printings():
    """Regression: the cache key used to be a tuple of ``CardPrinting``
    objects, which is unhashable because ``EventPattern.subject_filters``
    is a ``Mapping`` defaulting to ``dict``. The fix keys by printing
    name. Verify the cache key is hashable for a real synthetic-deck
    state (the G/R pool exercises ``StaticAbility``-bearing printings
    like ``Loading Zone``).
    """
    from mtg_bot.search.belief import _opponent_hand_key

    state = _engine_state(seed=0)
    key = _opponent_hand_key(state, opponent=1)
    # Must hash without raising; the bug was a TypeError here.
    hash(key)


def test_min_opponent_moments_gate_skips_belief_below_threshold():
    """Conditional-activation gate: with fewer than the configured
    threshold of opponent moments, the sampler short-circuits to
    uniform without calling the evaluator.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(adapter, opponent=1, max_steps=20)
    if len(log) < 2:
        pytest.skip("need at least 2 opponent moments")
    evaluator = _UniformPolicyEvaluator()
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=evaluator,
        played_history=log[:2],  # only 2 opponent moments
        m_candidates=4,
        min_opponent_moments=3,  # threshold above current count
    )
    s = adapter.engine_state.clone()
    sampler(s, viewer_id=0, rng=random.Random(7), in_place=True)
    assert evaluator.calls == 0, (
        "gate should skip the evaluator when opponent moments are "
        "below threshold"
    )
    assert sampler.last_evaluator_calls == 0


def test_min_opponent_moments_gate_runs_belief_at_threshold():
    """Above threshold, belief runs normally."""
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(adapter, opponent=1, max_steps=20)
    if len(log) < 3:
        pytest.skip("need at least 3 opponent moments")
    evaluator = _UniformPolicyEvaluator()
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=evaluator,
        played_history=log[:3],
        m_candidates=4,
        min_opponent_moments=3,
    )
    s = adapter.engine_state.clone()
    sampler(s, viewer_id=0, rng=random.Random(7), in_place=True)
    assert evaluator.calls > 0, (
        "evaluator should be called when opponent moments meet "
        "threshold"
    )


def test_belief_end_to_end_with_protocol_evaluator_on_synthetic_deck():
    """End-to-end: drive several opponent moments, then run the
    sampler under a protocol-conforming evaluator and confirm it
    returns without raising and produces a valid post-state.
    """
    adapter = _adapter(seed=11)
    opponent = 1
    viewer = 0
    log = _drive_to_opponent_moments(adapter, opponent, max_steps=30)
    if len(log) < 2:
        pytest.skip("need at least 2 opponent moments under seed=11")

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_ProtocolEvaluator(),
        played_history=log,
        m_candidates=8,
    )
    s = adapter.engine_state.clone()
    out = sampler(s, viewer_id=viewer, rng=random.Random(7), in_place=True)
    assert out is s  # in_place
    # Viewer's hand size preserved by determinization.
    assert (
        len(out.players[viewer].hand)
        == len(adapter.engine_state.players[viewer].hand)
    )


# ---------- persistent particles (SIS) -------------------------------------


def test_persistent_particles_reuse_seeds_across_calls():
    """When played_history grows by a single moment, the sampler must
    reuse the SAME particle seeds — only their log-weights are
    updated. Cold-call cost goes from O(m * H_total) to O(m * Δh).

    Captures the underlying invariant: the seeds tuple in
    ``_particles_state`` is identity-stable across same-viewer calls
    when the resample trigger doesn't fire (we set the threshold to
    zero to suppress resampling for this test).
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(adapter, opponent=1, max_steps=30)
    if len(log) < 4:
        pytest.skip("need at least 4 opponent moments under seed=11")

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log[:1],
        m_candidates=8,
        min_opponent_moments=0,
        ess_resample_threshold=0.0,  # disable resampling
    )
    sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(0), in_place=False,
    )
    seeds_after_call_1 = sampler._particles_state[1]
    assert sampler._particles_state[3] == 1  # scored 1 moment

    # Append a moment and call again — same seeds, new history_len.
    sampler.played_history = tuple(log[:2])
    sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(99), in_place=False,
    )
    seeds_after_call_2 = sampler._particles_state[1]
    assert sampler._particles_state[3] == 2  # now scored 2 moments
    assert seeds_after_call_2 == seeds_after_call_1

    # And another — still same seeds.
    sampler.played_history = tuple(log[:3])
    sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(123), in_place=False,
    )
    assert sampler._particles_state[1] == seeds_after_call_1
    assert sampler._particles_state[3] == 3


def test_persistent_particles_incremental_scoring_matches_batch():
    """Building the history one moment at a time — calling the
    sampler after each append — must produce the same per-particle
    log-weights as a single from-scratch call against the full
    history. SIS likelihood is additive across moments by
    construction; this pins that invariant in code.

    We use the deterministic-seed path (Random(0) drives both the
    initial seed draw and downstream use) and disable resampling so
    the only divergence source would be a math bug, not a
    multinomial draw difference.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(adapter, opponent=1, max_steps=30)
    if len(log) < 4:
        pytest.skip("need at least 4 opponent moments under seed=11")
    history = log[:4]

    # Sampler A: incremental — append one moment at a time, call
    # between each append.
    incremental = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=history[:1],
        m_candidates=4,
        min_opponent_moments=0,
        ess_resample_threshold=0.0,
    )
    rng_inc = random.Random(2027)
    incremental(
        adapter.engine_state.clone(), viewer_id=0, rng=rng_inc,
        in_place=False,
    )
    for i in range(2, len(history) + 1):
        incremental.played_history = tuple(history[:i])
        incremental(
            adapter.engine_state.clone(), viewer_id=0, rng=rng_inc,
            in_place=False,
        )

    # Sampler B: from-scratch — start with the full history.
    one_shot = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=history,
        m_candidates=4,
        min_opponent_moments=0,
        ess_resample_threshold=0.0,
    )
    rng_one = random.Random(2027)
    one_shot(
        adapter.engine_state.clone(), viewer_id=0, rng=rng_one,
        in_place=False,
    )

    # Same seeds (drawn from same rng) and same per-moment math →
    # bit-identical log-weights. (Resampling disabled.)
    assert incremental._particles_state[1] == one_shot._particles_state[1]
    inc_weights = incremental._particles_state[2]
    one_weights = one_shot._particles_state[2]
    assert np.allclose(inc_weights, one_weights, atol=1e-12), (
        f"incremental weights {inc_weights} should equal "
        f"one-shot weights {one_weights}"
    )


def test_resample_triggered_by_low_ess():
    """Configure an evaluator that puts heavy weight on hands
    containing a particular printing. Append several moments and
    confirm ``last_resample_count`` increments — i.e. the SIS pool
    actually triggers a resample once weight concentrates on a
    minority of seeds.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(
        adapter, opponent=1, max_steps=30, multi_action_only=True,
    )
    if len(log) < 3:
        pytest.skip("need at least 3 opponent multi-action moments")
    target = _pick_rare_hand_printing(log[0].state, opponent=1)
    if target is None:
        pytest.skip("no rare-singleton printing in opp hand at this moment")

    favored = _DeterministicSlotEvaluator(
        favored_printing_names={target}, boost=10.0,
    )
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=favored,
        played_history=log[:1],
        m_candidates=16,
        min_opponent_moments=0,
        ess_resample_threshold=0.5,
    )
    rng = random.Random(7)
    initial_count = sampler.last_resample_count
    # Drive the history forward; biasing evaluator concentrates
    # weight onto the few seeds whose reconstructed hand contains
    # ``target``, dropping ESS until the threshold trips.
    for i in range(1, min(len(log), 6) + 1):
        sampler.played_history = tuple(log[:i])
        sampler(
            adapter.engine_state.clone(), viewer_id=0,
            rng=rng, in_place=False,
        )
    # At least one resample fired across that walk.
    assert sampler.last_resample_count > initial_count, (
        f"expected ess-based resample to fire under heavy bias; "
        f"last_resample_count={sampler.last_resample_count}"
    )


def test_resample_disabled_at_threshold_zero():
    """ess_resample_threshold=0 must disable the resampler entirely
    even under a degenerate evaluator that drives ESS to 1.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(
        adapter, opponent=1, max_steps=30, multi_action_only=True,
    )
    if len(log) < 2:
        pytest.skip("need at least 2 opponent multi-action moments")
    target = _pick_rare_hand_printing(log[0].state, opponent=1)
    if target is None:
        pytest.skip("no rare-singleton printing in opp hand at this moment")
    favored = _DeterministicSlotEvaluator(
        favored_printing_names={target}, boost=10.0,
    )
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=favored,
        played_history=log[:2],
        m_candidates=16,
        min_opponent_moments=0,
        ess_resample_threshold=0.0,
    )
    sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(7), in_place=False,
    )
    assert sampler.last_resample_count == 0


def test_viewer_change_reinitializes_pool():
    """Reusing one sampler across two viewers must scrap and
    re-init the particle pool: per-viewer log-weights mean different
    things (each viewer's opponent is a different player). The
    ``viewer_id`` slot of ``_particles_state`` reflects the latest
    call, and ``scored_history_len`` resets to the (now)
    full-history scoring after the fresh init.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(adapter, opponent=1, max_steps=30)
    if not log:
        pytest.skip("no opponent moments under seed=11")

    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log,
        m_candidates=4,
        min_opponent_moments=0,
        ess_resample_threshold=0.0,
    )
    sampler(
        adapter.engine_state.clone(), viewer_id=0,
        rng=random.Random(0), in_place=False,
    )
    state_v0 = sampler._particles_state
    assert state_v0[0] == 0
    # Now switch viewer. The pool must re-init under the new
    # viewer_id; the stored viewer flips to 1.
    sampler(
        adapter.engine_state.clone(), viewer_id=1,
        rng=random.Random(7),  # different rng so seeds differ too
        in_place=False,
    )
    state_v1 = sampler._particles_state
    assert state_v1[0] == 1
    assert state_v1[1] != state_v0[1], (
        "fresh seed draw under different viewer should yield "
        "distinct seeds"
    )


def test_score_particle_compatibility_wrapper_still_works():
    """The single-particle ``_score_particle`` API is used by the
    existing ``test_score_particle_log_likelihood_separates_consistent_vs_inconsistent``
    test. The SIS refactor preserves it as a thin wrapper around
    ``_score_particles_batched``.
    """
    adapter = _adapter(seed=11)
    log = _drive_to_opponent_moments(
        adapter, opponent=1, max_steps=30, multi_action_only=True,
    )
    if not log:
        pytest.skip("no opponent multi-action moments under seed=11")
    sampler = BeliefWeightedDeterminizer(
        opponent_policy_evaluator=_UniformPolicyEvaluator(),
        played_history=log[:2],
        m_candidates=4,
        min_opponent_moments=0,
    )
    log_w = sampler._score_particle(
        adapter.engine_state.clone(), viewer_id=0, seed=2027,
    )
    assert math.isfinite(log_w)


def test_resample_particles_helper_returns_zero_weights():
    """Direct unit test on the resample helper: returns
    ``len(seeds)`` indices drawn from the categorical and zero
    log-weights (uniform priors over the resampled pool).
    """
    seeds = [1001, 1002, 1003, 1004]
    # Mass concentrated on index 0.
    p = np.array([0.97, 0.01, 0.01, 0.01], dtype=np.float64)
    rng = random.Random(0)
    new_seeds, new_log_weights = _resample_particles(seeds, p, rng)
    assert len(new_seeds) == 4
    # Resampled seeds are drawn FROM the input pool.
    for s in new_seeds:
        assert s in seeds
    # Index 0 dominates: most copies should be 1001.
    assert new_seeds.count(1001) >= 3
    assert np.allclose(new_log_weights, 0.0)


def test_resample_threshold_validation():
    with pytest.raises(ValueError, match="ess_resample_threshold"):
        BeliefWeightedDeterminizer(
            opponent_policy_evaluator=_UniformPolicyEvaluator(),
            ess_resample_threshold=-0.1,
        )
    with pytest.raises(ValueError, match="ess_resample_threshold"):
        BeliefWeightedDeterminizer(
            opponent_policy_evaluator=_UniformPolicyEvaluator(),
            ess_resample_threshold=1.5,
        )
