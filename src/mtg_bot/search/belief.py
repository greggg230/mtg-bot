"""Belief-weighted determinization for ISMCTS (Phase 6).

A particle-filter sampler that draws determinizations from the
posterior over opponent hidden state, conditioned on the opponent's
observed action history. Replaces the uniform sampler in
``mtg_bot.search.determinize`` for the ``ISMCTS.determinize_fn``
callback. See ``docs/ISMCTS_MIGRATION.md`` (Phase 6) for the design
spec and ``docs/PHASE6_SNAPSHOT_AUDIT.md`` for why the priority-moment
log is supplied in-memory by the caller rather than read from disk.

Bayesian setup. For a candidate hidden-info hypothesis ``H`` (one
sampled determinization) and the opponent's observed action sequence
``A = (a_1, ..., a_T)`` taken at priority moments ``s_1, ..., s_T``::

    P(H | A) ∝ P(A | H) P(H)
    log P(A | H) = sum_t log P(a_t | s_t, H)

The prior ``P(H)`` is the uniform sampler we already have. The
likelihood ``P(a_t | s_t, H)`` comes from the trained policy head:
reconstruct the historical state with hand identities consistent
with ``H``, masked-softmax over the legal-action vocab, read off the
slot encoding ``a_t``.

The class generates ``m_candidates`` uniform draws, scores each by
log-likelihood (with a floor on zero-probability slots so a single
inconsistent action doesn't annihilate a particle), normalizes via
log-sum-exp, and resamples one. Returns the chosen
``MtgGameStateAdapter`` — same callback shape ISMCTS already invokes.

Historical-state reconstruction strategy. Each candidate is
materially identified by an integer particle seed; the same seed is
re-used to determinize a clone of each historical priority-moment
state. This is the "sample-and-project" approach in the audit's open
question 2: rather than time-rewind a fixed candidate hand to align
with each historical opponent zone shape, we re-determinize each
historical clone with the candidate's seed. The seed encodes the
particle's RNG identity; downstream effects of "this particle drew
permutation X at the root" propagate into "this particle's reading of
historical moment t" coherently because both are determinizations
sharing one ``random.Random(seed)`` per particle. This sidesteps the
zone-shape mismatch problem that arises when the candidate's
current-state hand is a subset/superset of the historical hand.

Caching. The likelihood evaluation reads from the cloned-and-
determinized historical state through ``opponent_policy_evaluator
.evaluate_logits`` plus ``action_encoding.legal_action_mask``. Calls
that share the same (printings tuple, history index) tuple return
identical results, so an LRU-bounded dict keyed on that pair holds
``(value, logits)`` and the legal mask. The eviction policy is FIFO
once the cache exceeds ``cache_max_size``; this is "good enough"
since the access pattern is heavily weighted toward recent particles
within a single ``__call__`` and we don't need true LRU semantics.

Numerical hygiene. All log-likelihood accumulation is in log-space.
Zero-probability action slots get floored at ``log(1e-12)`` to keep
the particle alive with a soft penalty rather than collapsing to
``-inf`` (Cowling et al. 2015 §4 discuss this; the floor is a soft
penalty, not absolute exclusion). Effective sample size is computed
post-normalization; values below ``ess_warn_threshold`` are surfaced
through the ``last_ess`` attribute for caller-side diagnostics
without changing search behavior — for Phase 6 this is informational.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

import numpy as np

from mtg_bot.game import GameState as EngineGameState
from mtg_bot.search.determinize import determinize as _uniform_determinize
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.protocol import Decision, PlayerID

# Floor for log-probability of an action that has zero probability
# under a candidate hand. log(1e-12) ≈ -27.6. Keeps the particle
# alive at a heavy but finite penalty instead of total elimination.
_LOG_PROB_FLOOR = math.log(1e-12)


class PriorityMoment:
    """One historical opponent priority moment.

    Stored in the in-memory log the caller maintains. Holds the
    pre-action engine state clone (``state``), the action that was
    actually taken (``decision``), and the player who held priority
    (``actor``). The state clone is a real ``EngineGameState`` — not
    a serialized snapshot — so the per-particle re-determinization
    can call into the engine's clone/legal_action machinery without
    deserialization. The audit (option (b)) recommends this in-memory
    representation; it is not written to disk.
    """

    __slots__ = ("state", "decision", "actor")

    def __init__(
        self,
        state: EngineGameState,
        decision: Decision,
        actor: PlayerID,
    ) -> None:
        self.state = state
        self.decision = decision
        self.actor = actor


class BeliefWeightedDeterminizer:
    """Particle-filter sampler over hidden-info hypotheses about the
    opponent.

    Use as a drop-in for ``ISMCTS(determinize_fn=...)``. With an
    empty ``played_history`` the sampler degenerates to the uniform
    sampler (posterior = prior). With a non-empty history each
    candidate's log-weight is the sum of historical-action log-
    probabilities under the candidate's hidden state, evaluated by
    the supplied ``opponent_policy_evaluator``.

    The callback contract matches the uniform determinizer:
    ``__call__(engine_state, viewer_id, rng, in_place=True)``
    returns one ``EngineGameState`` (in-place mutation of the input
    when ``in_place=True``). ISMCTS calls it once per iteration.
    """

    def __init__(
        self,
        opponent_policy_evaluator: Any,
        played_history: Optional[Sequence[PriorityMoment]] = None,
        m_candidates: int = 64,
        cache_max_size: int = 4096,
        log_prob_floor: float = _LOG_PROB_FLOOR,
        ess_warn_threshold: float = 1.0,
        nn_batch_size: int = 256,
        min_opponent_moments: int = 3,
        ess_resample_threshold: float = 0.5,
    ) -> None:
        """Construct the sampler.

        ``opponent_policy_evaluator`` must expose ``evaluate_logits(
        state, player_id) → (value, logits)`` — the same shape
        ``NNEvaluator(with_policy=True)`` and the test stubs in
        ``test_ismcts.py`` implement.

        ``played_history`` is the in-memory list of opponent priority
        moments collected by the caller as the game progresses. An
        empty / ``None`` history short-circuits to uniform sampling.
        Each entry is a ``PriorityMoment`` — see that class for the
        required shape. The caller is responsible for populating this
        log; this module never reads on-disk replay files.

        ``m_candidates`` controls the proposal pool size. Higher is a
        better posterior approximation at higher cost; the cost is
        ``O(m_candidates * len(played_history))`` policy queries per
        ``__call__``, mitigated by the cache.

        ``cache_max_size`` bounds the ((hand_names, library_names),
        history_index) cache. FIFO eviction;
        ``OrderedDict.popitem(last=False)``.

        ``log_prob_floor`` is the minimum log-probability assigned to
        an action with zero probability under a candidate hand —
        typically because the candidate determinization makes the
        opponent's actual action illegal in the reconstructed state.
        Defaults to log(1e-12); larger floors weaken the prior, smaller
        ones approximate hard exclusion.

        ``ess_warn_threshold`` is the effective-sample-size floor
        below which the particle filter is flagged as degenerate;
        ``last_ess`` is set on each ``__call__`` and a value
        ``< ess_warn_threshold`` is informational for now.

        ``nn_batch_size`` is the chunk size for the batched
        ``batch_evaluate_logits`` call. The sampler collects all
        cache-miss states across particles+moments, dedupes by cache
        key, then evaluates them in one or more chunks of this size.
        Default 256 — large enough to amortize Python<->torch
        overhead on CPU and to fill GPU SMs without overcommitting
        memory at the typical (m_candidates=64) * (history=25) =
        1600-state working set.

        ``min_opponent_moments`` (default 3) is a conditional-
        activation gate: the sampler falls back to the uniform
        determinizer until ``played_history`` contains at least this
        many opponent (non-viewer) priority moments. Below the
        threshold the posterior is approximately uniform anyway —
        too few observations for the likelihood to differentiate
        from the prior — and skipping the m * H scoring pass cuts
        cold-call cost on early-game decisions where wallclock
        dominates. Set to 0 to disable the gate (always run belief).

        ``ess_resample_threshold`` (default 0.5) controls when
        Sequential Importance Sampling triggers a multinomial
        resample of the particle pool: after each weight update,
        if ``ESS < ess_resample_threshold * m_candidates`` the
        sampler draws ``m_candidates`` new seeds with probabilities
        proportional to current normalized weights, replaces the
        particle pool with those new seeds, and resets all log-
        weights to 0 (uniform prior over the replaced pool). The
        diagnostic counter ``last_resample_count`` records the
        cumulative resample fires across the determinizer's life.
        Set to 0 to disable resampling (pure SIS, will degenerate
        on long histories).
        """
        if m_candidates < 1:
            raise ValueError(
                f"m_candidates must be >= 1, got {m_candidates!r}"
            )
        if cache_max_size < 1:
            raise ValueError(
                f"cache_max_size must be >= 1, got {cache_max_size!r}"
            )
        if nn_batch_size < 1:
            raise ValueError(
                f"nn_batch_size must be >= 1, got {nn_batch_size!r}"
            )
        if min_opponent_moments < 0:
            raise ValueError(
                f"min_opponent_moments must be >= 0, got "
                f"{min_opponent_moments!r}"
            )
        if not (0.0 <= ess_resample_threshold <= 1.0):
            raise ValueError(
                f"ess_resample_threshold must be in [0, 1], got "
                f"{ess_resample_threshold!r}"
            )
        if not callable(getattr(
            opponent_policy_evaluator, "evaluate_logits", None,
        )):
            raise ValueError(
                "opponent_policy_evaluator must expose "
                "evaluate_logits(state, player_id); got "
                f"{type(opponent_policy_evaluator).__name__}."
            )
        self.opponent_policy_evaluator = opponent_policy_evaluator
        # Mirror MCTS / ISMCTS: pull the action-encoding version off the
        # evaluator if it has one (NNEvaluator does), else default to v1.
        # Threaded into the ``ae.legal_action_mask`` and ``ae.encode``
        # calls in ``_score_work_items_batched`` below so opponent-
        # policy slot lookups use the same convention the network was
        # trained against.
        self._action_encoding_version = getattr(
            opponent_policy_evaluator,
            "action_encoding_version",
            "positional_v1",
        )
        self.played_history: tuple[PriorityMoment, ...] = tuple(
            played_history or (),
        )
        self.m_candidates = m_candidates
        self.cache_max_size = cache_max_size
        self.log_prob_floor = log_prob_floor
        self.ess_warn_threshold = ess_warn_threshold
        self.nn_batch_size = nn_batch_size
        self.min_opponent_moments = min_opponent_moments
        self.ess_resample_threshold = ess_resample_threshold
        # FIFO cache keyed on (opponent_hand_printings_tuple,
        # history_index) → (legal_mask, logits). Uses an OrderedDict
        # so popitem(last=False) gives FIFO eviction, no extra
        # bookkeeping.
        self._cache: "OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )
        # Diagnostic counters reset on every ``__call__``.
        self.last_ess: float = 0.0
        self.last_evaluator_calls: int = 0
        # Cumulative diagnostic — increments every time SIS triggers
        # a resample. Tests + perf-monitor read this to verify the
        # ESS-based resampler is firing under degenerate weights.
        self.last_resample_count: int = 0
        # Persistent particle pool (Sequential Importance Sampling).
        # ``_particles_state`` holds (viewer_id, seeds, log_weights,
        # scored_history_len). Across ``__call__`` invocations the
        # pool is REUSED and its log-weights are updated
        # INCREMENTALLY by scoring only newly-appended moments —
        # cold-call cost goes from O(m * H_total) to O(m * Δh). The
        # pool re-initializes when the viewer changes (rare; one
        # determinizer per ISMCTS instance) or on the first call
        # past the conditional-activation gate (initial scoring of
        # all current moments). When ESS falls below the configured
        # fraction of m_candidates the seeds are resampled
        # multinomially with replacement and weights reset to zero.
        self._particles_state: Optional[
            tuple[PlayerID, tuple[int, ...], np.ndarray, int]
        ] = None

    # ------------------------------------------------------------------
    # Determinizer-callback surface
    # ------------------------------------------------------------------

    def __call__(
        self,
        engine_state: EngineGameState,
        viewer_id: PlayerID,
        rng: random.Random,
        *,
        in_place: bool = False,
    ) -> EngineGameState:
        """Sample one belief-weighted determinization.

        Mirrors the ``determinize.determinize`` signature so it's a
        drop-in for ``ISMCTS.determinize_fn``. With an empty
        ``played_history`` returns ``determinize(engine_state,
        viewer_id, rng, in_place=in_place)`` — the prior IS the
        uniform sampler.

        Otherwise: draws ``m_candidates`` particle seeds, scores each
        by log-likelihood under ``played_history``, normalizes, and
        resamples one particle. The chosen particle's seed is then
        applied (via ``determinize``) to the input state.

        ``in_place=True`` mutates ``engine_state``; ``in_place=False``
        clones first. Same convention as the uniform determinizer.
        """
        # Reset per-call diagnostics.
        self.last_evaluator_calls = 0

        if not self.played_history:
            # No history → posterior == prior. Uniform sampler is the
            # MLE / sole correct answer; skip the particle-filter
            # plumbing.
            self.last_ess = float(self.m_candidates)
            return _uniform_determinize(
                engine_state, viewer_id, rng, in_place=in_place,
            )

        # Conditional-activation gate: until we've observed at least
        # ``min_opponent_moments`` opponent (non-viewer) actions, the
        # likelihood signal is too weak to differentiate the
        # posterior from the uniform prior in any consistent way —
        # and the cold-call cost is wasted on early-game decisions
        # where wallclock dominates iteration throughput. O(H)
        # check; cheaper than the gate's avoided work by orders of
        # magnitude.
        if self.min_opponent_moments > 0:
            opp_moment_count = sum(
                1 for m in self.played_history if m.actor != viewer_id
            )
            if opp_moment_count < self.min_opponent_moments:
                self.last_ess = float(self.m_candidates)
                return _uniform_determinize(
                    engine_state, viewer_id, rng, in_place=in_place,
                )

        # Persistent particle filter (Sequential Importance Sampling).
        # Across ``__call__`` invocations within a game we reuse the
        # SAME m_candidates seeds and update their log-weights
        # incrementally as the played_history grows. Cold-call cost
        # per game-move goes from O(m * H_total) to O(m * Δh) where
        # Δh is the number of moments appended since last call.
        history_len = len(self.played_history)
        state = self._particles_state
        if state is not None and state[0] != viewer_id:
            # Different viewer than the pool was initialized for —
            # scrap and start over. (Architecturally: each ISMCTS
            # strategy owns its own determinizer, so this branch is
            # exercised primarily by tests reusing a sampler across
            # opponents.)
            state = None
            self._particles_state = None

        if state is None:
            # First scoring pass for this viewer: draw a fresh seed
            # pool and score against ALL currently observed moments.
            seeds = [
                rng.randrange(1 << 62) for _ in range(self.m_candidates)
            ]
            log_weights = self._score_particles_batched(
                viewer_id, seeds, h_start=0, h_end=history_len,
            )
        else:
            scored_viewer, seeds_tuple, log_weights, scored_len = state
            seeds = list(seeds_tuple)
            log_weights = log_weights.copy()
            if history_len > scored_len:
                # Incremental update: score only the newly-appended
                # moments and add to existing log-weights. Pre-existing
                # weights stay valid because log P(A_{1:T} | H) =
                # log P(A_{1:t} | H) + log P(A_{t+1:T} | H) — the per-
                # moment log-likelihoods are additive.
                delta = self._score_particles_batched(
                    viewer_id, seeds,
                    h_start=scored_len, h_end=history_len,
                )
                log_weights = log_weights + delta
            elif history_len < scored_len:
                # History shrank — abnormal (the caller is supposed
                # to only append). Treat as a fresh init for safety.
                seeds = [
                    rng.randrange(1 << 62)
                    for _ in range(self.m_candidates)
                ]
                log_weights = self._score_particles_batched(
                    viewer_id, seeds, h_start=0, h_end=history_len,
                )
            # else history unchanged → reuse cached log_weights.

        normalized = _normalize_log_weights(log_weights)
        self.last_ess = _effective_sample_size(normalized)

        # ESS-based resampling: when weight is concentrated on a
        # small subset of particles the effective sample size drops
        # and SIS becomes a degenerate point estimate. Drawing
        # m_candidates new seeds with probability proportional to
        # current weights (and resetting log-weights to zero)
        # restores diversity at the cost of a fresh scoring pass
        # over the full history (the resampled seeds carry no past).
        # We pay that cost lazily — the next __call__ that adds new
        # moments will incrementally score them; the ALREADY-scored
        # history is rolled into the seeds' identity by the
        # categorical resample step.
        if (
            self.ess_resample_threshold > 0.0
            and self.last_ess
            < self.ess_resample_threshold * self.m_candidates
            and self.m_candidates > 1
        ):
            seeds, log_weights = _resample_particles(
                seeds, normalized, rng,
            )
            normalized = _normalize_log_weights(log_weights)
            # Post-resample ESS == m_candidates (uniform weights).
            self.last_ess = float(self.m_candidates)
            self.last_resample_count += 1

        self._particles_state = (
            viewer_id, tuple(seeds), log_weights, history_len,
        )

        # Sample one particle index from the categorical(normalized).
        # ``rng.random()`` is the user-supplied stream; using it
        # (instead of np.random) keeps the whole sampler reproducible
        # under the caller's seed.
        chosen_idx = _categorical_draw(normalized, rng)
        chosen_seed = seeds[chosen_idx]

        # Re-apply the chosen particle's seed to the input state.
        # ``random.Random(seed)`` reproduces the determinization
        # (printing permutation) we scored.
        return _uniform_determinize(
            engine_state, viewer_id,
            random.Random(chosen_seed),
            in_place=in_place,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_particles_batched(
        self,
        viewer_id: PlayerID,
        seeds: Sequence[int],
        h_start: int = 0,
        h_end: Optional[int] = None,
    ) -> np.ndarray:
        """Score every particle in ``seeds`` against the played
        history slice ``[h_start, h_end)`` in one batched pass.

        Returns a length-``len(seeds)`` log-likelihood array — the
        same shape and semantics as calling ``_score_particle`` once
        per seed, but with the NN evaluations batched. ``h_end``
        defaults to ``len(self.played_history)``.

        The slice form lets the SIS persistent-particle path score
        only newly-appended moments per game-move. Per-moment seed
        mixing still uses the absolute ``h_idx`` so the cache keys
        and per-moment determinizations remain bit-equivalent to the
        full-pass version.

        Algorithm (three phases, all opponent moments only):

        1. **Collect**: for each (particle, h_idx) pair where
           ``moment.actor != viewer_id``, clone the historical state,
           re-determinize under the particle's seed (mixed with
           ``h_idx``, matching the per-particle code path), and
           record a work item. Compute the cache key from the
           reconstructed hand + library names, and decide which work
           items map to cache misses.

        2. **Evaluate**: batch the cache-miss states in chunks of
           ``self.nn_batch_size`` and call
           ``opponent_policy_evaluator.batch_evaluate_logits`` once
           per chunk. Compute legal masks per state. Stash
           ``(mask, logits)`` into the cache.

        3. **Score**: walk the work items, look up the (mask,
           logits) from the cache, encode ``moment.decision`` to a
           slot, and accumulate ``_slot_log_prob`` into the
           per-particle weight.

        The fast equivalence with the per-particle path is exact
        when ``nn_batch_size`` is large enough to fit all misses in
        one chunk. With chunking, the only behavioral difference is
        that cache fill happens in batches, not per-call — visible
        as ``last_evaluator_calls`` reporting batch count rather
        than per-state count (each batched call counts as 1 here).
        Tests pin the math, not the call-count granularity.
        """
        from mtg_bot.training import action_encoding as ae

        if h_end is None:
            h_end = len(self.played_history)

        # Phase 1: collect work items. Each entry is a tuple of
        # (particle_idx, cache_key, reconstructed_engine_state,
        # opponent_id, decision). Cache_key is None when we can
        # short-circuit the work item (e.g. encoding will fail; we
        # use a sentinel "skip and floor" mark instead).
        work_items: list[
            tuple[int, Any, EngineGameState, PlayerID, Decision]
        ] = []
        miss_states: list[MtgGameStateAdapter] = []
        miss_keys: list[Any] = []
        miss_engine_states: list[EngineGameState] = []
        miss_opponents: list[PlayerID] = []
        seen_in_misses: dict[Any, int] = {}

        for p_idx, seed in enumerate(seeds):
            for h_idx in range(h_start, h_end):
                moment = self.played_history[h_idx]
                if moment.actor == viewer_id:
                    continue
                opponent = moment.actor
                # Mix scheme matches the per-particle reference path
                # (h_idx=0 uses raw seed; later h_idx mixes via the
                # golden-ratio constant) so this batched version is
                # bit-equivalent to the per-particle one for the
                # same inputs.
                if h_idx == 0:
                    moment_rng = random.Random(seed)
                else:
                    mix = (
                        seed * 0x9E3779B97F4A7C15 + h_idx
                    ) & ((1 << 63) - 1)
                    moment_rng = random.Random(mix)
                reconstructed = moment.state.clone()
                _uniform_determinize(
                    reconstructed, viewer_id, moment_rng,
                    in_place=True,
                )
                hand_key = _opponent_hand_key(reconstructed, opponent)
                cache_key = (hand_key, h_idx)
                work_items.append(
                    (p_idx, cache_key, reconstructed, opponent,
                     moment.decision),
                )
                # Queue a fresh evaluation for any cache key we
                # haven't already cached and haven't already queued
                # within this call.
                if (
                    cache_key not in self._cache
                    and cache_key not in seen_in_misses
                ):
                    seen_in_misses[cache_key] = len(miss_states)
                    miss_states.append(
                        MtgGameStateAdapter(reconstructed),
                    )
                    miss_engine_states.append(reconstructed)
                    miss_keys.append(cache_key)
                    miss_opponents.append(opponent)

        # Phase 2: batched NN evaluation. ``batch_evaluate_logits``
        # takes a single ``player_id`` per call. In a 2-player game
        # all opponent moments share one opponent (the non-viewer);
        # we partition by ``opponent_id`` defensively for
        # robustness against multi-opponent extensions.
        if miss_states:
            by_opp: dict[PlayerID, list[int]] = {}
            for k, opp in enumerate(miss_opponents):
                by_opp.setdefault(opp, []).append(k)
            for opp, indices in by_opp.items():
                # Chunk to avoid memory blow-up on huge histories.
                for chunk_start in range(
                    0, len(indices), self.nn_batch_size,
                ):
                    chunk = indices[
                        chunk_start: chunk_start + self.nn_batch_size
                    ]
                    states_chunk = [miss_states[i] for i in chunk]
                    results = self._batch_or_loop(states_chunk, opp)
                    self.last_evaluator_calls += 1
                    for k, (_, logits) in zip(chunk, results):
                        engine = miss_engine_states[k]
                        mask = ae.legal_action_mask(
                            engine, opp,
                            version=self._action_encoding_version,
                        )
                        self._cache[miss_keys[k]] = (mask, logits)
                    self._evict_if_oversize()

        # Phase 3: per-work-item scoring. Walk the items in original
        # order so per-particle accumulation is deterministic.
        log_weights = np.zeros(len(seeds), dtype=np.float64)
        for p_idx, cache_key, reconstructed, opponent, decision in (
            work_items
        ):
            cached = self._cache.get(cache_key)
            if cached is None:
                # Should not happen — phase 2 just populated every
                # missing key. Defensive: floor and continue.
                log_weights[p_idx] += self.log_prob_floor
                continue
            mask, logits = cached
            # LRU bump (cheap).
            self._cache.move_to_end(cache_key)
            try:
                slot = ae.encode(
                    decision, reconstructed, opponent,
                    version=self._action_encoding_version,
                )
            except ae.EncodingOverflowError:
                # Action slot unreachable in this candidate.
                log_weights[p_idx] += self.log_prob_floor
                continue
            except Exception:
                # Encoding error (e.g. zone mismatch under candidate
                # determinization). Same handling.
                log_weights[p_idx] += self.log_prob_floor
                continue
            log_p = _slot_log_prob(
                logits, mask, slot, self.log_prob_floor,
            )
            log_weights[p_idx] += log_p
        return log_weights

    def _score_particle(
        self,
        engine_state: EngineGameState,
        viewer_id: PlayerID,
        seed: int,
    ) -> float:
        """Return ``log P(played_history | particle)``.

        Thin compatibility wrapper around
        ``_score_particles_batched`` for the single-particle case;
        kept for tests that probe the per-particle math directly.

        For each historical priority moment whose actor is NOT
        ``viewer_id``, clones the historical state, re-determinizes
        it under the particle's seed, evaluates the opponent NN's
        policy, and adds ``log P(action_taken)``. Sum across moments.

        Viewer-action moments are SKIPPED — only opponent moments
        contribute likelihood signal: viewer-action probability is
        independent of the candidate hand.
        """
        # ``engine_state`` is intentionally unused — the historical
        # moment states (``moment.state``) carry the per-moment
        # context; the live current state is only used by ``__call__``
        # for the final resample step.
        _ = engine_state
        log_weights = self._score_particles_batched(viewer_id, [seed])
        return float(log_weights[0])

    def _batch_or_loop(
        self,
        states: Sequence[MtgGameStateAdapter],
        player_id: PlayerID,
    ) -> list[tuple[float, np.ndarray]]:
        """Call the evaluator's batched API if available; otherwise
        fall back to a loop over per-state ``evaluate_logits``. Lets
        the sampler interoperate with the test mocks (which only
        implement ``evaluate_logits``) and the production
        ``NNEvaluator`` (which has both) on the same code path.
        """
        batch_fn = getattr(
            self.opponent_policy_evaluator,
            "batch_evaluate_logits", None,
        )
        if callable(batch_fn):
            return batch_fn(states, player_id)
        return [
            self.opponent_policy_evaluator.evaluate_logits(s, player_id)
            for s in states
        ]

    def _evict_if_oversize(self) -> None:
        while len(self._cache) > self.cache_max_size:
            self._cache.popitem(last=False)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _opponent_hand_key(
    state: EngineGameState, opponent: PlayerID,
) -> tuple[Any, ...]:
    """Cache key built from the opponent's hand + library printing
    NAMES (in zone order). Library order matters because the NN may
    consult library top via features (we don't know which features it
    uses, so be conservative). Including library order trades a bit
    of cache-miss rate for correctness.

    We key by ``printing.name`` rather than the printing object
    itself: at least one ``CardPrinting`` field (``StaticAbility`` ->
    ``ReplacementEffect`` -> ``EventPattern.subject_filters``, a
    ``Mapping`` defaulting to ``dict``) is mutably-typed and breaks
    ``hash``. Names are unique per card and ``extract_features`` only
    reads identity-equivalent fields, so name keys are sufficient.
    Fixing ``EventPattern.subject_filters`` to a frozen container is
    the right structural fix; tracked as a follow-up.
    """
    player = state.players[opponent]
    hand = tuple(c.printing.name for c in player.hand)
    library = tuple(c.printing.name for c in player.library)
    return (hand, library)


def _slot_log_prob(
    logits: np.ndarray,
    mask: np.ndarray,
    slot: int,
    floor: float,
) -> float:
    """Compute ``log P(slot)`` under the masked-softmax policy.

    Returns ``floor`` when the slot is masked illegal or when the
    softmax has no support (no legal slots). Otherwise computes
    log-softmax in the numerically-safe form (subtract max, log-sum-
    exp) and returns the slot's component.
    """
    if slot < 0 or slot >= logits.shape[0]:
        return floor
    if not bool(mask[slot]):
        return floor
    if not mask.any():
        return floor
    # Apply mask as -inf in a working copy.
    row = logits.astype(np.float64, copy=True)
    row[~mask] = -np.inf
    row_max = float(row.max())
    if not math.isfinite(row_max):
        return floor
    shifted = row - row_max
    # Slot's pre-softmax shifted logit; normalize via log-sum-exp.
    log_z = math.log(float(np.exp(shifted).sum()))
    log_p = float(shifted[slot]) - log_z
    if not math.isfinite(log_p):
        return floor
    if log_p < floor:
        return floor
    return log_p


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Convert log-weights to a normalized probability vector via
    log-sum-exp. Returns a length-N float64 vector summing to 1.

    All-``-inf`` input degrades to a uniform distribution: that's the
    "no particle has any support" pathological case; uniform is the
    least-bad fallback (and surfaces via ``last_ess == N``).
    """
    finite_mask = np.isfinite(log_weights)
    if not finite_mask.any():
        n = log_weights.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    max_lw = float(log_weights[finite_mask].max())
    shifted = log_weights - max_lw
    # ``-inf`` after shift → exp = 0, drops out of the sum.
    weights = np.exp(shifted)
    z = float(weights.sum())
    if z <= 0.0 or not math.isfinite(z):
        n = log_weights.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    return (weights / z).astype(np.float64, copy=False)


def _effective_sample_size(normalized: np.ndarray) -> float:
    """Kish's effective sample size: ``1 / sum(p_i^2)``. Range
    [1, N]; N = uniform, 1 = all weight on one particle. Used as a
    degeneracy monitor — Phase 6 surfaces this for diagnostics; later
    phases would trigger stratified resampling on low ESS.
    """
    s = float(np.square(normalized).sum())
    if s <= 0.0:
        return 0.0
    return 1.0 / s


def _resample_particles(
    seeds: Sequence[int],
    normalized: np.ndarray,
    rng: random.Random,
) -> tuple[list[int], np.ndarray]:
    """Multinomial resample of the particle pool.

    Draws ``len(seeds)`` indices from a categorical distribution
    parameterized by ``normalized`` (with replacement) using
    ``_categorical_draw`` and the caller's ``rng`` — keeping the
    whole sampler reproducible under one seed. Returns ``(new_seeds,
    fresh_log_weights)`` where ``fresh_log_weights`` is an all-zeros
    array (uniform priors over the resampled pool, ready to receive
    incremental updates from future moments).

    Concentration of the categorical doesn't degrade resample
    correctness — when one particle holds nearly all weight the
    resample collapses the pool to copies of that particle's seed.
    Subsequent incremental updates against new moments will then
    discover differential likelihood across those copies (different
    h_idx mixing produces different reconstructed hands per moment),
    re-spreading the weights.
    """
    n = len(seeds)
    new_seeds: list[int] = []
    for _ in range(n):
        idx = _categorical_draw(normalized, rng)
        new_seeds.append(seeds[idx])
    return new_seeds, np.zeros(n, dtype=np.float64)


def _categorical_draw(
    probabilities: np.ndarray, rng: random.Random,
) -> int:
    """Sample one index from a categorical distribution using
    ``rng.random()`` for reproducibility under the caller's seed.

    Inverse-CDF sampling: cumulate probabilities, draw u in [0, 1),
    return the first index with cumsum > u. O(N) worst case;
    ``m_candidates`` is small (default 64) so this is negligible.
    """
    u = rng.random()
    cumulative = 0.0
    last = probabilities.shape[0] - 1
    for i, p in enumerate(probabilities):
        cumulative += float(p)
        if u < cumulative:
            return i
    # Floating-point safety net: if u landed past the last cumulative
    # due to rounding, return the last index.
    return last


__all__ = [
    "BeliefWeightedDeterminizer",
    "PriorityMoment",
    "_resample_particles",
]
