"""Diagnostic tests for NN-evaluator canonicality.

Surfaced from the per-search NN-eval cache (``feature/nn-eval-cache``)
investigation: that branch's eval cache is bit-exact under deterministic
test evaluators, but in real-NN benches the visit distributions differed
between cache-on and cache-off in 4 of 8 states. Working hypothesis was
that the feature encoder reads raw ``card.id`` ordering and so two
canonically-equivalent states (e.g. three Forests in hand with ids
permuted) produce slightly different value/policy outputs.

These tests pin the **canonical contract**: two states that differ only
in card identity (object ``.id``) but otherwise have identical printings,
modifiers, and zone *contents at each position* MUST produce bit-equal
NN outputs from both heads. With the v13 weights present, all checks
PASS — the encoder is canonical, refuting the "encoder reads raw ids"
hypothesis.

Why the encoder is canonical (forensic notes for a future maintainer):

- ``mtg_bot.features.extract_features`` only emits aggregates (counts /
  sums / max P+T) per zone and per side. There's no per-card slot, no
  embedding indexed by ``card.id``, no positional indexing by id. Two
  states with permuted card.ids in hand (same printings, same list
  order) produce identical feature vectors — and the value/policy
  heads are pure functions of the feature vector and the legal-action
  mask. The legal-action mask in ``training.action_encoding`` is
  positional-by-zone-index, so different ids at the same hand position
  set the same mask bits.

Implication for the cache investigation: the visit-distribution drift
between cache-on and cache-off must come from a non-encoder source.
Plausible suspects, in priority order, for the next investigator:

- **Rollout / determinization RNG.** With the cache on, the FIRST
  reaching of a transposition-equivalent state pays the NN-eval cost
  and seeds whatever stochastic downstream effects depend on the
  visit timing; the second reaching gets a free hit and so its RNG
  draws happen at a different "global step" within the search.
- **Tie-breaking.** PUCT picks among children with equal UCT scores
  by argmax-of-ties, and the cache changes the wall-clock order in
  which Q-values get updated, so two children that were equal-Q at
  the moment of selection without the cache may not be equal-Q at
  the same moment with the cache (or vice versa).
- **Cache-key coverage.** ``canonical_state_key`` deliberately drops
  some adapter-side fields (e.g., ``rng / next_object_id /
  next_timestamp``) on the rationale that they don't affect forward
  play; if any of those silently feeds into a downstream RNG path,
  the cache reuses an eval keyed off "equivalent forward state" but
  the search downstream of the eval consumes the dropped state.

None of those is an encoder bug; they're search-loop / cache-key
considerations and need the user's input to prioritize.

The v13 weights are loaded from the parent main checkout's
``trained_weights/v13/`` directory. The dir is gitignored and lives
only in the main checkout, so the worktree reaches across to it via
absolute path. Skipped when the weights aren't present (e.g., on a
fresh clone before any training has run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mtg_bot.cards import BASIC_LANDS
from mtg_bot.features import extract_features
from mtg_bot.game import GameState, Player, Zone
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.training.action_encoding import legal_action_mask
from mtg_bot.training.registry import Registry
from mtg_bot.turn import Step


# ---------------------------------------------------------------------------
# Path to v13 weights. Worktrees don't carry trained_weights/ (gitignored,
# present only in the main checkout). Skip cleanly if missing.
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
# tests/training/test_nn_canonicality.py
#   parents[0] = .../tests/training
#   parents[1] = .../tests
#   parents[2] = .../nn-encoder-canonicality (worktree root)
#   parents[3] = .../mtg-bot-worktrees
#   parents[4] = .../Projects (sibling of mtg-bot)
_MAIN_CHECKOUT_TRAINED_WEIGHTS = (
    _THIS.parents[4] / "mtg-bot" / "trained_weights"
)


def _v13_evaluator() -> NNEvaluator:
    """Load v13 from the main checkout's trained_weights, mirroring the
    recipe in ``scripts/capture_annotated_replays.py``. Returns a
    dual-head ``NNEvaluator`` (with_policy=True).
    """
    if not _MAIN_CHECKOUT_TRAINED_WEIGHTS.is_dir():
        pytest.skip(
            f"trained_weights/ not found at {_MAIN_CHECKOUT_TRAINED_WEIGHTS}"
        )
    registry = Registry(_MAIN_CHECKOUT_TRAINED_WEIGHTS)
    if "v13" not in registry.list_versions():
        pytest.skip("v13 not registered in trained_weights/")
    weights_path = registry.weights_path("v13")
    meta = registry.load_metadata("v13")
    hidden_sizes = (
        tuple(meta.hidden_sizes) if getattr(meta, "hidden_sizes", None)
        else None
    )
    arch = getattr(meta, "architecture", "shared")
    policy_hidden = (
        tuple(meta.training.get("policy_hidden_sizes") or ()) or None
    )
    kwargs: dict = {
        "weights_path": weights_path,
        "with_policy": True,
        "architecture": arch,
    }
    if hidden_sizes is not None:
        kwargs["hidden_sizes"] = hidden_sizes
    if policy_hidden is not None:
        kwargs["policy_hidden_sizes"] = policy_hidden
    return NNEvaluator(**kwargs)


# ---------------------------------------------------------------------------
# Test-state fixtures. Two two-player states differing ONLY in the
# ``.id`` ordering of three same-printing Forests in P0's hand.
# ---------------------------------------------------------------------------


def _make_state_with_three_forests_in_p0_hand(
    hand_id_order: tuple[int, int, int],
) -> GameState:
    """Build a 2-player priority-window state with three Forests in
    P0's hand. The hand list ORDER mirrors ``hand_id_order`` exactly:
    ``hand[i]`` has ``.id == hand_id_order[i]``. P0 also has a Forest
    on the battlefield (untapped, summoning-sick-cleared), to make the
    legal-action set non-trivial. P1 is empty.

    All other state is fixed: precombat main, P0 active and on
    priority, turn 1, life 20/20, no graveyards / exiles / stack /
    continuous effects.
    """
    state = GameState()
    state.players = [Player(id=0), Player(id=1)]
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = Step.PRECOMBAT_MAIN
    state.turn_number = 1

    forest = BASIC_LANDS["Forest"]

    # Allocate the hand cards in ANY order, then fix their ids to the
    # requested ordering. ``new_object`` allocates from
    # ``state.next_object_id``; we override ``.id`` afterwards. Same
    # printing (Forest), same modifier states (none), same zone (hand).
    p0 = state.players[0]
    for target_id in hand_id_order:
        obj = state.new_object(forest, owner_id=0, zone=Zone.HAND)
        obj.id = target_id
        p0.hand.append(obj)

    # Bump next_object_id past whatever ids we forced so subsequent
    # allocations don't collide. Battlefield Forest gets a fresh id
    # well past the hand ids.
    state.next_object_id = max(hand_id_order) + 100

    bf_forest = state.new_object(forest, owner_id=0, zone=Zone.BATTLEFIELD)
    bf_forest.summoning_sick = False  # lands aren't summoning-sick anyway
    state.battlefield.append(bf_forest)

    return state


def test_features_are_canonical_under_card_id_permutation():
    """Sanity / pre-NN-load check. Two states differing only by hand
    card.id ordering produce identical feature vectors.

    This is the strong precondition for canonical NN output. If this
    fails, the encoder is reading raw ids — fix it before checking
    NN outputs.
    """
    state_a = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    state_b = _make_state_with_three_forests_in_p0_hand((81, 82, 83))

    feats_a = extract_features(state_a, viewer_id=0)
    feats_b = extract_features(state_b, viewer_id=0)

    np.testing.assert_array_equal(feats_a, feats_b)


def test_legal_action_mask_is_canonical_under_card_id_permutation():
    """Companion to the feature-canonicality check: the policy head's
    output IS post-mask-and-softmax, so we need the mask to be canonical
    too. The mask is positional by hand-list index, not by id, so two
    states with the same printings at the same hand positions produce
    the same mask regardless of underlying ids.
    """
    state_a = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    state_b = _make_state_with_three_forests_in_p0_hand((81, 82, 83))

    mask_a = legal_action_mask(state_a, viewer=0)
    mask_b = legal_action_mask(state_b, viewer=0)

    np.testing.assert_array_equal(mask_a, mask_b)


# ---------------------------------------------------------------------------
# The headline diagnostic: NN outputs are bit-equal across permuted
# card-id states. Loads real v13 weights.
# ---------------------------------------------------------------------------


def test_v13_value_head_is_canonical_under_card_id_permutation():
    """Two states differing ONLY in hand card.id ordering produce
    bit-equal value-head output from v13.

    If this fails, the encoder leaks raw ids into the feature tensor —
    that's the latent bug the cache investigation surfaced. Tolerance
    is ``1e-7`` (single-precision float epsilon ballpark) to absorb
    any innocuous floating-point reordering inside the model; a real
    leak would produce visibly different outputs (the NN is only
    weakly-shaped by training, so even tiny input differences move
    the output more than epsilon).
    """
    evaluator = _v13_evaluator()
    state_a = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    state_b = _make_state_with_three_forests_in_p0_hand((81, 82, 83))

    value_a = evaluator._evaluate_nonterminal(state_a, player_id=0)
    value_b = evaluator._evaluate_nonterminal(state_b, player_id=0)

    assert abs(value_a - value_b) < 1e-7, (
        f"value head differs under card.id permutation: "
        f"value_a={value_a!r} value_b={value_b!r} "
        f"|delta|={abs(value_a - value_b):.3e}"
    )


def test_v13_policy_head_is_canonical_under_card_id_permutation():
    """Two states differing ONLY in hand card.id ordering produce
    bit-equal policy-head distributions from v13.

    Compares the masked + softmaxed policy tensor (length
    ACTION_VOCAB_SIZE = 517). Both states have the same legal slots
    (positional mask), so the legal subset and its probability
    assignments must match.
    """
    evaluator = _v13_evaluator()
    state_a = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    state_b = _make_state_with_three_forests_in_p0_hand((81, 82, 83))
    adapter_a = MtgGameStateAdapter(state_a)
    adapter_b = MtgGameStateAdapter(state_b)

    value_a, policy_a = evaluator.evaluate_with_policy(adapter_a, player_id=0)
    value_b, policy_b = evaluator.evaluate_with_policy(adapter_b, player_id=0)

    assert abs(value_a - value_b) < 1e-7
    np.testing.assert_allclose(policy_a, policy_b, atol=1e-7, rtol=0)


def test_v13_logits_are_canonical_under_card_id_permutation():
    """Raw (un-masked, un-softmaxed) policy logits are bit-equal too.
    This is the strictest check: the value head AND the policy logits
    fall out of the same forward pass on the same feature tensor, so
    if the feature tensors match, all output channels must.

    A leak that affects only the masked/softmaxed policy would imply
    something funny in masking; this test rules that out by checking
    raw logits directly.
    """
    evaluator = _v13_evaluator()
    state_a = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    state_b = _make_state_with_three_forests_in_p0_hand((81, 82, 83))
    adapter_a = MtgGameStateAdapter(state_a)
    adapter_b = MtgGameStateAdapter(state_b)

    value_a, logits_a = evaluator.evaluate_logits(adapter_a, player_id=0)
    value_b, logits_b = evaluator.evaluate_logits(adapter_b, player_id=0)

    assert abs(value_a - value_b) < 1e-7
    np.testing.assert_allclose(logits_a, logits_b, atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# Battery: a few additional permutations to make sure no permutation
# slips through. Cheap (3 forwards × 6 perms = 18 forwards on a 51-dim
# input — well under a second).
# ---------------------------------------------------------------------------


_PERMUTATIONS = [
    (71, 72, 73),
    (73, 72, 71),
    (72, 73, 71),
    (200, 201, 202),
    (1_000_001, 1_000_002, 1_000_003),
]


@pytest.mark.parametrize("perm", _PERMUTATIONS)
def test_v13_outputs_match_baseline_across_permutations(perm):
    """Battery check: every permutation produces outputs bit-equal
    (within 1e-7) to the canonical baseline (71, 72, 73). Uses the
    raw-logits path so we get value + policy in one forward pass.
    """
    evaluator = _v13_evaluator()
    base = _make_state_with_three_forests_in_p0_hand((71, 72, 73))
    other = _make_state_with_three_forests_in_p0_hand(perm)
    base_adapter = MtgGameStateAdapter(base)
    other_adapter = MtgGameStateAdapter(other)

    v_base, logits_base = evaluator.evaluate_logits(base_adapter, player_id=0)
    v_other, logits_other = evaluator.evaluate_logits(
        other_adapter, player_id=0,
    )

    assert abs(v_base - v_other) < 1e-7, (
        f"perm={perm}: value differs base={v_base!r} other={v_other!r}"
    )
    np.testing.assert_allclose(logits_base, logits_other, atol=1e-7, rtol=0)
