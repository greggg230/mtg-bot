"""Skeleton model for Step 1 of the NN representation roadmap.

This module pins the architecture decisions resolved in
``docs/NN_REPRESENTATION_ROADMAP.md`` (Resolved design decisions §1-§6)
as a working PyTorch skeleton. The class instantiates every trainable
submodule the production model will need and runs a clean
``forward(batch)`` over a dict of variable-length token tensors so
implementation slices (data-pipeline conversion, training-loop
integration, SHM-server compat path) can target a stable shape.

NOT YET IMPLEMENTED in this skeleton:

- The data pipeline that builds the per-token tensors from a live
  ``GameState`` / ``MtgGameStateAdapter``. The pipeline is a separate
  slice (see roadmap "Implementation slices") and is the single
  remaining piece between this skeleton and an end-to-end trainable
  model.
- The hand-belief auxiliary head's loss head. The trunk-side query
  table and cross-attention pathway are present and forward-passable;
  the data targets and loss function land with Step 4.

The skeleton is designed to be importable + forward-runnable with no
engine dependency — tests construct random integer + float batches
and confirm shapes, gradient flow, and mask handling.

Architecture summary
====================

Input shape (per batch element):

    {
        "token_type":      LongTensor(seq_len)           # 0..N_TOKEN_TYPES-1
        "token_emb_ids":   LongTensor(seq_len, N_EMB_FIELDS)
        "token_scalars":   FloatTensor(seq_len, N_SCALAR_FIELDS)
        "attention_mask":  BoolTensor(seq_len)           # True = real token
    }

Trunk:

    1. Per-field embedding lookup → sum into a single emb_dim vector per
       token. Token-type embedding adds a learned bias per token-type
       (CLS / state-permanent / state-hand / ... / action-event).
    2. Prepend learned [CLS] token (handled in forward).
    3. K stacked ``nn.TransformerEncoderLayer`` (pre-norm).
    4. Final per-token representation. The CLS position pools for value;
       the per-token positions feed the cross-attention policy decoder.

Heads:

    - Value head: 2-layer MLP from CLS → scalar (tanh).
    - Policy head: ``action_vocab_size`` learned slot embeddings query
      the per-token trunk output via ``nn.MultiheadAttention``; a small
      MLP turns the attended representation into one logit per slot.
    - Hand-belief head (aux, stubbed): ``n_printings`` learned printing
      queries do the same cross-attention; emit per-printing logits.

All hyperparameters are constructor arguments with sensible defaults
matching the roadmap recommendations (emb_dim=128, n_heads=4,
n_layers=2, hidden_dim=256). The action vocab pulls from
``mtg_bot.training.action_encoding.ACTION_VOCAB_SIZE`` so policy
output stays in lockstep with the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE


# ---------------------------------------------------------------------
# Token-type vocab. Token types tag what KIND of game-state entry a row
# is (battlefield permanent vs hand card vs action event vs ...). The
# trunk uses this to add a per-type learned bias to the input embedding.
# ---------------------------------------------------------------------

TOKEN_TYPE_CLS = 0
TOKEN_TYPE_GLOBAL = 1
TOKEN_TYPE_PERMANENT_VIEWER = 2
TOKEN_TYPE_PERMANENT_OPPONENT = 3
TOKEN_TYPE_HAND_VIEWER = 4
TOKEN_TYPE_HAND_OPPONENT = 5  # known-card hand-belief queries; usually empty
TOKEN_TYPE_GRAVEYARD_VIEWER = 6
TOKEN_TYPE_GRAVEYARD_OPPONENT = 7
TOKEN_TYPE_EXILE_VIEWER = 8
TOKEN_TYPE_EXILE_OPPONENT = 9
TOKEN_TYPE_STACK_ITEM = 10
TOKEN_TYPE_ACTION_EVENT = 11

N_TOKEN_TYPES = 12
"""Number of distinct token-type ids. Bumped when a new type is added."""


# ---------------------------------------------------------------------
# Per-token embedding-field vocab. Each row carries an integer tuple of
# this length; the index space below tells the model what each slot
# means. Adding a new field bumps N_EMB_FIELDS and one vocab constant.
# ---------------------------------------------------------------------

# Field slots in ``token_emb_ids[seq_len, N_EMB_FIELDS]``.
EMB_FIELD_PRINTING = 0
EMB_FIELD_ACTION_KIND = 1
EMB_FIELD_CONTROLLER = 2
EMB_FIELD_TARGET_PRINTING = 3
EMB_FIELD_TARGET_KIND = 4
EMB_FIELD_VARIANT = 5
EMB_FIELD_STEP = 6

N_EMB_FIELDS = 7

# Field slots in ``token_scalars[seq_len, scalar_width]``.
#
# The scalar row is VERSIONED ("scalar schema"). Schema v1 is the
# original 3-slot row every pre-v2 checkpoint (v1..v14, v13r/v13s,
# ...) was trained against; schema v2 appends the numeric board /
# life state the v1 row could not see (tapped, P/T, damage,
# counters, life totals, zone sizes, land counts, ...). The v1 slots
# come FIRST and keep their indices, so a v2 row's ``[:3]`` prefix is
# exactly the v1 row. Old weights stay loadable because the model's
# ``scalar_proj`` width is picked from
# ``StructuredModelConfig.scalar_schema_version`` (which loaders read
# from ``ModelMetadata.scalar_schema_version``; missing = 1).
#
# --- schema v1 slots (all token kinds) ---------------------------
SCALAR_FIELD_TURN_OFFSET = 0
SCALAR_FIELD_MANA_COST_PAID = 1
SCALAR_FIELD_PUBLIC_MASK = 2

N_SCALAR_FIELDS = 3
"""Width of the schema-v1 scalar row (kept for back-compat)."""

# --- schema v2 slots: battlefield permanents (both sides) --------
# Zero for every non-permanent token kind. Normalizers are fixed
# constants (not data statistics) so the on-disk contract is stable:
# P/T and damage are /8, counters are /4, flags are 0/1. Values are
# clamped so a freak board (a 40/40) can't blow up the projection.
SCALAR_FIELD_TAPPED = 3
SCALAR_FIELD_IS_CREATURE = 4
SCALAR_FIELD_POWER = 5
"""``effective_power / 8`` (layer-7 aware), clamp [-2, 2]; creatures only."""
SCALAR_FIELD_TOUGHNESS = 6
"""``effective_toughness / 8``, clamp [-2, 2]; creatures only."""
SCALAR_FIELD_DAMAGE = 7
"""``damage marked / 8``, clamp [0, 2]."""
SCALAR_FIELD_PT_COUNTERS = 8
"""Net ``(+1/+1 counters - -1/-1 counters) / 4``, clamp [-2, 2]."""
SCALAR_FIELD_OTHER_COUNTERS = 9
"""Sum of non-P/T counters (charge / time / stun / loyalty / lore /
flying) ``/ 4``, clamp [0, 2]."""
SCALAR_FIELD_SUMMONING_SICK = 10
"""1 if a creature that is summoning sick, else 0."""
SCALAR_FIELD_ATTACKING = 11
SCALAR_FIELD_BLOCKING = 12

# --- schema v2 slots: the [GLOBAL] token -------------------------
# Zero for every other token kind. Viewer-relative (viewer / opp).
SCALAR_FIELD_VIEWER_LIFE = 13
"""``life / 20``, clamp [-1, 3]."""
SCALAR_FIELD_OPP_LIFE = 14
SCALAR_FIELD_VIEWER_POISON = 15
"""``poison / 10``, clamp [0, 1]."""
SCALAR_FIELD_OPP_POISON = 16
SCALAR_FIELD_VIEWER_LIBRARY = 17
"""``library size / 40``, clamp [0, 2]."""
SCALAR_FIELD_OPP_LIBRARY = 18
SCALAR_FIELD_VIEWER_HAND = 19
"""``hand size / 7``, clamp [0, 2]."""
SCALAR_FIELD_OPP_HAND = 20
SCALAR_FIELD_TURN_NUMBER = 21
"""``turn_number / 20``, clamp [0, 1]."""
SCALAR_FIELD_VIEWER_UNTAPPED_LANDS = 22
"""``untapped lands / 8``, clamp [0, 2]."""
SCALAR_FIELD_OPP_UNTAPPED_LANDS = 23
SCALAR_FIELD_VIEWER_LANDS = 24
"""``lands / 8``, clamp [0, 2]."""
SCALAR_FIELD_OPP_LANDS = 25
SCALAR_FIELD_STACK_SIZE = 26
"""``len(stack) / 4``, clamp [0, 2]."""
SCALAR_FIELD_IS_VIEWER_TURN = 27
SCALAR_FIELD_VIEWER_HAS_PRIORITY = 28

N_SCALAR_FIELDS_V2 = 29
"""Width of the schema-v2 scalar row (v1 slots first, then the 26
appended board / global slots)."""

SCALAR_SCHEMA_V1 = 1
SCALAR_SCHEMA_V2 = 2
SUPPORTED_SCALAR_SCHEMA_VERSIONS: tuple[int, ...] = (
    SCALAR_SCHEMA_V1, SCALAR_SCHEMA_V2,
)
DEFAULT_SCALAR_SCHEMA_VERSION = SCALAR_SCHEMA_V2
"""Schema newly-constructed models / tokenizer calls use when the
caller doesn't say otherwise. Loaders NEVER rely on this default --
they read the version off the checkpoint's metadata."""

_SCALAR_WIDTH_BY_SCHEMA: dict[int, int] = {
    SCALAR_SCHEMA_V1: N_SCALAR_FIELDS,
    SCALAR_SCHEMA_V2: N_SCALAR_FIELDS_V2,
}


def scalar_width_for_schema(version: int) -> int:
    """Return the ``token_scalars`` row width for a scalar schema
    version. Raises ``ValueError`` on an unknown version so a typo
    in metadata / CLI flags fails at construction time rather than
    as a cryptic shape mismatch deep in ``load_state_dict``."""
    try:
        return _SCALAR_WIDTH_BY_SCHEMA[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"unknown scalar_schema_version {version!r}; expected one "
            f"of {SUPPORTED_SCALAR_SCHEMA_VERSIONS}"
        ) from None


# ---------------------------------------------------------------------
# Sub-vocabs. NA-token id 0 in every embedding table means
# "field not applicable for this token" — e.g. action-kind on a
# battlefield permanent, or target on a pass-priority action.
# ---------------------------------------------------------------------

# Action-kind vocab (22 + NA = 23). Order doesn't matter to the model;
# the data pipeline will pick stable integer codes when it lands.
ACTION_KIND_VOCAB_SIZE = 23
"""NA + 9 priority + 2 combat + 11 resolution = 23."""

# Controller vocab: NA / viewer / opponent.
CONTROLLER_VOCAB_SIZE = 3

# Target-kind vocab: NA / player / permanent / stack-item / hand-card /
# exile-card.
TARGET_KIND_VOCAB_SIZE = 6

# Variant vocab: NA + 7 (normal / kicked / warp / cast_for_free /
# impending / room_half_0 / room_half_1 / private_choice).
VARIANT_VOCAB_SIZE = 9

# Step vocab matches features.py's 0..11.
STEP_VOCAB_SIZE = 13
"""NA + 12 game steps = 13."""

# Default printing vocab; the registry's printing table for EoE today is
# ~287, but the model is built to scale. id 0 is reserved as NA.
DEFAULT_N_PRINTINGS = 320


# ---------------------------------------------------------------------
# Value-head input routing. See ``StructuredModelConfig
# .value_head_input`` and the ``docs/SCALING_DIAGNOSTICS.md`` D4 entry.
# ---------------------------------------------------------------------

VALUE_HEAD_INPUT_TRUNK = "trunk"
"""Default / historical: the value head reads the [CLS] position of
the one trunk pass the policy head also reads."""

VALUE_HEAD_INPUT_TRUNK_NO_PRINTING = "trunk_no_printing"
"""Identity-blind value path: a SECOND trunk pass (same weights) over
token embeddings built with the printing / target-printing embedding
contributions dropped. The value head reads that pass's [CLS]; the
policy head keeps the full-information pass. Costs one extra trunk
forward per evaluation and adds ZERO parameters, so every existing
checkpoint still loads under either setting."""

SUPPORTED_VALUE_HEAD_INPUTS: tuple[str, ...] = (
    VALUE_HEAD_INPUT_TRUNK,
    VALUE_HEAD_INPUT_TRUNK_NO_PRINTING,
)


class _GELUDropout(nn.Sequential):
    """GELU followed by dropout, as ONE ``nn.Sequential`` slot.

    The heads are ``nn.Sequential(Linear, GELU, Linear)``; their
    parameter keys are ``<head>.0.*`` and ``<head>.2.*``. Inserting a
    bare ``nn.Dropout`` between the GELU and the second Linear would
    renumber that Linear to index 3 and every checkpoint trained
    before this change would fail to load. Wrapping GELU + Dropout in
    one submodule keeps index 1 occupied by a parameterless container,
    so the on-disk key set is byte-identical. ``p=0`` is an identity
    dropout, so the numerics are unchanged too.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__(nn.GELU(), nn.Dropout(p))


@dataclass
class StructuredModelConfig:
    """Hyperparameters for :class:`StructuredModel`.

    Defaults match the values called out in the roadmap's
    "Resolved design decisions §6":

    - ``emb_dim`` = 128 (token + trunk hidden dim)
    - ``n_heads`` = 4
    - ``n_layers`` = 2
    - ``hidden_dim`` = 256 (MLP inside transformer encoder layers)
    - ``action_history_n`` = 32 (the §3 default action-history window)
    - ``n_printings`` = 320 (room for EoE + small slack)
    - ``action_vocab_size`` = ``ACTION_VOCAB_SIZE`` (517 today)

    Frozen-ish: dataclass mutability is intentional so test setups
    can shrink the config when constructing a tiny model for unit
    tests (e.g. ``emb_dim=8``).
    """

    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    hidden_dim: int = 256
    dropout: float = 0.0
    """Dropout probability applied inside the transformer encoder
    layers, the policy / hand-belief cross-attention, AND the value /
    policy / hand-belief head MLPs.

    The dataclass default stays 0.0 so constructing a model for
    INFERENCE (or reconstructing an old checkpoint's config) is
    unchanged; the training CLI defaults ``--dropout`` to 0.1. 0.0
    was the setting under which the s2 lineage's value heads
    memorised game identity — see
    :mod:`mtg_bot.training.value_gate`."""
    action_history_n: int = 32
    n_printings: int = DEFAULT_N_PRINTINGS
    action_vocab_size: int = ACTION_VOCAB_SIZE
    enable_hand_belief_head: bool = False
    """Step 4 lights this up. Skeleton has the trainable submodules
    in place but ``forward`` only returns hand-belief logits when
    this is True."""
    scalar_schema_version: int = DEFAULT_SCALAR_SCHEMA_VERSION
    """Which ``token_scalars`` row layout the model consumes (see the
    ``SCALAR_FIELD_*`` block). Sets ``scalar_proj``'s input width via
    :attr:`scalar_width`. Pre-v2 checkpoints are schema 1 (3 slots);
    loaders read the version off ``ModelMetadata`` so old weights
    keep loading. The tokenizer must be called with the SAME version
    (``tokenize_state(..., scalar_schema_version=...)``) or the batch
    width won't match ``scalar_proj``."""
    value_head_input: str = VALUE_HEAD_INPUT_TRUNK
    """Which representation the VALUE head consumes — see
    ``SUPPORTED_VALUE_HEAD_INPUTS``.

    ``"trunk"`` (default, and what every pre-2026-09-25 checkpoint was
    trained under) shares one trunk pass with the policy head.

    ``"trunk_no_printing"`` runs the value head off a second pass of
    the SAME trunk over identity-blind embeddings: the per-printing
    (``EMB_FIELD_PRINTING`` / ``EMB_FIELD_TARGET_PRINTING``) lookups
    are dropped, leaving token type, controller, action kind, target
    kind, variant, step and the schema-v2 scalars (mana value proxies,
    P/T, damage, counters, life, zone sizes, ...). The value head then
    physically cannot key on "which exact deck pair is this", which is
    what the s2 lineage memorised. Adds no parameters; costs one extra
    trunk forward per evaluation."""
    value_temperature: float = 1.0
    """Post-hoc value calibration: the value output is
    ``tanh(logit / value_temperature + value_bias)``.

    1.0 / 0.0 is the identity and is what every checkpoint trained
    before :mod:`mtg_bot.training.value_calibration` existed uses —
    the forward pass short-circuits to the exact previous
    computation, so old weights are bit-identical. Fitted post-
    training on the held-out split and carried in
    ``ModelMetadata.value_calibration``; loaders set it here so every
    consumer of ``out["value"]`` (search, both eval servers, the
    replay annotator) sees calibrated values with no per-consumer
    wiring. NOT part of the state_dict — it is metadata, so a
    re-calibration does not rewrite weights."""
    value_bias: float = 0.0
    """See :attr:`value_temperature`."""

    def __post_init__(self) -> None:
        # Validate eagerly so a bad version surfaces at config time.
        scalar_width_for_schema(self.scalar_schema_version)
        if self.value_head_input not in SUPPORTED_VALUE_HEAD_INPUTS:
            raise ValueError(
                f"value_head_input must be one of "
                f"{SUPPORTED_VALUE_HEAD_INPUTS}; got "
                f"{self.value_head_input!r}"
            )
        temp = float(self.value_temperature)
        if not (temp > 0.0) or temp != temp or temp in (
            float("inf"), float("-inf"),
        ):
            raise ValueError(
                f"value_temperature must be finite and > 0; got "
                f"{self.value_temperature!r}"
            )

    @property
    def scalar_width(self) -> int:
        """``token_scalars`` row width implied by
        ``scalar_schema_version``."""
        return scalar_width_for_schema(self.scalar_schema_version)


class StructuredModel(nn.Module):
    """Step 1 skeleton: attention over per-object tokens with
    last-N action tokens, learned printing embeddings, [CLS] pooling,
    and a cross-attention policy decoder.

    The model is the resolved-design counterpart of the current
    ``_ValuePolicyMLP`` in ``nn_evaluator.py``. It does NOT yet replace
    that model in the production training loop — the data-pipeline
    slice is the missing piece. This class exists so subsequent slices
    can target a stable shape.

    Forward contract:

        out = model(batch)
        out["value"]                  # (B,) float in [-1, +1]
        out["policy_logits"]          # (B, action_vocab_size)
        out["hand_belief_logits"]     # (B, n_printings)  -- only when
                                      #   enable_hand_belief_head=True

    Padding handling: tokens with ``attention_mask=False`` are
    ignored by every attention layer (key_padding_mask) and contribute
    no gradient.
    """

    def __init__(self, config: Optional[StructuredModelConfig] = None) -> None:
        super().__init__()
        self.config = config or StructuredModelConfig()
        cfg = self.config
        if cfg.emb_dim % cfg.n_heads != 0:
            raise ValueError(
                f"emb_dim ({cfg.emb_dim}) must be divisible by "
                f"n_heads ({cfg.n_heads}); MultiheadAttention requires it."
            )

        # --- Embedding tables ----------------------------------------
        # Printing embedding — shared between state-zone tokens
        # (source_printing per battlefield/hand/graveyard/exile/stack
        # entry) and action-event tokens (source_printing /
        # target_printing of the action). This is the "card2vec"
        # cluster substrate from the goal section.
        self.printing_embed = nn.Embedding(
            cfg.n_printings, cfg.emb_dim, padding_idx=0,
        )
        # Token-type embedding adds a learned bias per token kind.
        self.token_type_embed = nn.Embedding(
            N_TOKEN_TYPES, cfg.emb_dim,
        )
        # Action-event sub-vocab embeddings.
        self.action_kind_embed = nn.Embedding(
            ACTION_KIND_VOCAB_SIZE, cfg.emb_dim, padding_idx=0,
        )
        self.controller_embed = nn.Embedding(
            CONTROLLER_VOCAB_SIZE, cfg.emb_dim, padding_idx=0,
        )
        self.target_kind_embed = nn.Embedding(
            TARGET_KIND_VOCAB_SIZE, cfg.emb_dim, padding_idx=0,
        )
        self.variant_embed = nn.Embedding(
            VARIANT_VOCAB_SIZE, cfg.emb_dim, padding_idx=0,
        )
        self.step_embed = nn.Embedding(
            STEP_VOCAB_SIZE, cfg.emb_dim, padding_idx=0,
        )
        # Scalars get a single linear projection into ``emb_dim`` to add
        # alongside the categorical embeddings. Keeping the per-field
        # contribution additive (rather than concatenating) means
        # missing-scalar slots can be zeroed without a shape change.
        # Input width follows the config's scalar schema (3 for v1
        # checkpoints, N_SCALAR_FIELDS_V2 for v2).
        self.scalar_proj = nn.Linear(cfg.scalar_width, cfg.emb_dim)
        # Learned [CLS] token prepended to every input sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Transformer trunk ---------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.trunk = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers,
        )

        # --- Value head (consumes the CLS position) ------------------
        # ``_GELUDropout`` occupies the single Sequential slot the
        # bare ``nn.GELU()`` used to, so the state_dict keys stay
        # ``value_head.0.*`` / ``value_head.2.*`` and every existing
        # checkpoint keeps loading (see the class docstring).
        self.value_head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            _GELUDropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # --- Policy head ---------------------------------------------
        # Cross-attention decoder. One learned query vector per
        # action-vocab slot; each slot attends to the trunk's per-token
        # output and produces a single logit through a small MLP.
        self.policy_slot_queries = nn.Parameter(
            torch.zeros(cfg.action_vocab_size, cfg.emb_dim)
        )
        nn.init.normal_(self.policy_slot_queries, std=0.02)
        self.policy_decoder_attn = nn.MultiheadAttention(
            embed_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            _GELUDropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # --- Hand-belief aux head (Step 4 stub) ----------------------
        # Same shape as policy head, but the queries are per-printing
        # rather than per-action-slot. Trained submodules exist so
        # weight initialization, optimizer state, and checkpoint format
        # are stable — but the loss term lands with Step 4.
        self.hand_belief_queries = nn.Parameter(
            torch.zeros(cfg.n_printings, cfg.emb_dim)
        )
        nn.init.normal_(self.hand_belief_queries, std=0.02)
        self.hand_belief_attn = nn.MultiheadAttention(
            embed_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.hand_belief_head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            _GELUDropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    # ----------------------------------------------------------------
    # Forward helpers
    # ----------------------------------------------------------------

    def _embed_tokens(
        self,
        token_type: torch.Tensor,
        token_emb_ids: torch.Tensor,
        token_scalars: torch.Tensor,
        *,
        include_printing: bool = True,
    ) -> torch.Tensor:
        """Project the raw integer + scalar token tensors into a single
        ``(B, seq_len, emb_dim)`` embedding tensor.

        Per-field embeddings are SUMMED (not concatenated) so the
        per-token feature width stays at ``emb_dim`` regardless of
        how many fields are added later. Padding_idx=0 on every
        categorical table makes NA fields contribute the zero vector,
        which keeps "this token has no target_printing" cheap to
        represent without a separate field-presence mask.

        ``include_printing=False`` drops the two printing lookups
        (source + target), which is exactly the identity-blind
        representation ``value_head_input='trunk_no_printing'``
        feeds the value head. Because the contributions are summed,
        "drop" is literally "don't add" — no masking, no shape
        change, and the remaining fields keep their learned meaning.
        """
        # Per-categorical-field embedding lookups.
        if include_printing:
            printing_e = self.printing_embed(
                token_emb_ids[..., EMB_FIELD_PRINTING]
            )
        action_kind_e = self.action_kind_embed(
            token_emb_ids[..., EMB_FIELD_ACTION_KIND]
        )
        controller_e = self.controller_embed(
            token_emb_ids[..., EMB_FIELD_CONTROLLER]
        )
        if include_printing:
            target_printing_e = self.printing_embed(
                token_emb_ids[..., EMB_FIELD_TARGET_PRINTING]
            )
        target_kind_e = self.target_kind_embed(
            token_emb_ids[..., EMB_FIELD_TARGET_KIND]
        )
        variant_e = self.variant_embed(
            token_emb_ids[..., EMB_FIELD_VARIANT]
        )
        step_e = self.step_embed(
            token_emb_ids[..., EMB_FIELD_STEP]
        )
        # Per-token-type bias.
        type_e = self.token_type_embed(token_type)
        # Scalar projection.
        scalar_e = self.scalar_proj(token_scalars)
        # Sum all contributions into the per-token embedding. The
        # full-information branch keeps the exact original summation
        # ORDER: float addition is not associative, and reordering it
        # would perturb every existing checkpoint's output in the last
        # ulp for no reason.
        if include_printing:
            return (
                printing_e + action_kind_e + controller_e
                + target_printing_e + target_kind_e + variant_e
                + step_e + type_e + scalar_e
            )
        return (
            action_kind_e + controller_e
            + target_kind_e + variant_e
            + step_e + type_e + scalar_e
        )

    def _prepend_cls(
        self,
        token_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend the learned ``[CLS]`` token to every sequence and
        mark its mask position as valid.
        """
        batch_size = token_emb.shape[0]
        cls = self.cls_token.expand(batch_size, 1, -1)
        token_emb = torch.cat([cls, token_emb], dim=1)
        cls_mask = torch.ones(
            batch_size, 1, dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        return token_emb, attention_mask

    # ----------------------------------------------------------------
    # Public forward
    # ----------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_slot_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full trunk + heads on a packed-batch dict.

        ``policy_slot_range=(lo, hi)`` runs the policy decoder for slot
        queries ``lo .. hi-1`` only and returns ``policy_logits`` of
        shape ``(B, hi - lo)``. Each slot query attends independently,
        so the logits equal the corresponding columns of the full
        forward; inference uses it to score only the slot family that
        can be legal (the 517 priority / combat slots at a priority
        moment, the ``canonical_v3`` resolution block at a
        resolution pause -- ``docs/RESOLUTION_DECISIONS.md`` §7). The
        range ``(0, 517)`` on an extended v2 checkpoint is the exact
        computation the unextended checkpoint ran. ``None`` (training,
        default) scores every slot.

        Required keys in ``batch``:

        - ``token_type``     ``LongTensor(B, seq_len)``
        - ``token_emb_ids``  ``LongTensor(B, seq_len, N_EMB_FIELDS)``
        - ``token_scalars``  ``FloatTensor(B, seq_len, N_SCALAR_FIELDS)``
        - ``attention_mask`` ``BoolTensor(B, seq_len)`` —
          True = real token, False = padding.

        Returns a dict of:

        - ``value``               ``(B,)``        in ``[-1, +1]``
        - ``value_logit``         ``(B,)``        pre-tanh, pre-
          calibration (the input a temperature fit needs)
        - ``policy_logits``       ``(B, action_vocab_size)``
        - ``hand_belief_logits``  ``(B, n_printings)`` — only when
          ``config.enable_hand_belief_head=True``.
        """
        token_type = batch["token_type"]
        token_emb_ids = batch["token_emb_ids"]
        token_scalars = batch["token_scalars"]
        attention_mask = batch["attention_mask"]
        expected_width = self.config.scalar_width
        if token_scalars.shape[-1] != expected_width:
            raise ValueError(
                f"token_scalars width {token_scalars.shape[-1]} does not "
                f"match this model's scalar schema v"
                f"{self.config.scalar_schema_version} (expects "
                f"{expected_width} slots). The tokenizer / dataset was "
                f"produced under a different scalar_schema_version; "
                f"re-tokenize with the model's version or load a model "
                f"trained on this width."
            )

        token_emb = self._embed_tokens(
            token_type, token_emb_ids, token_scalars,
        )
        # Prepend [CLS]; the CLS position is always valid, so the
        # mask grows by one slot of ones up front.
        raw_attention_mask = attention_mask
        token_emb, attention_mask = self._prepend_cls(
            token_emb, attention_mask,
        )

        # TransformerEncoder uses ``src_key_padding_mask`` semantics
        # where True = "ignore this position." We flip our True=valid
        # mask accordingly.
        key_padding_mask = ~attention_mask.bool()
        trunk_out = self.trunk(
            token_emb, src_key_padding_mask=key_padding_mask,
        )

        # --- Value head (CLS pool) ----------------------------------
        # ``trunk_no_printing`` re-runs the SAME trunk weights over
        # embeddings with the per-printing lookups dropped, so the
        # value head sees board shape / life / counts but not card
        # identity. The policy head always reads the full pass.
        if self.config.value_head_input == (
            VALUE_HEAD_INPUT_TRUNK_NO_PRINTING
        ):
            blind_emb = self._embed_tokens(
                token_type, token_emb_ids, token_scalars,
                include_printing=False,
            )
            blind_emb, _ = self._prepend_cls(
                blind_emb, raw_attention_mask,
            )
            value_source = self.trunk(
                blind_emb, src_key_padding_mask=key_padding_mask,
            )
        else:
            value_source = trunk_out
        cls_vec = value_source[:, 0, :]
        value_logit = self.value_head(cls_vec).squeeze(-1)
        temperature = float(self.config.value_temperature)
        bias = float(self.config.value_bias)
        if temperature == 1.0 and bias == 0.0:
            # Identity calibration: keep the original expression so
            # pre-calibration checkpoints are bit-identical.
            value = torch.tanh(value_logit)
        else:
            value = torch.tanh(value_logit / temperature + bias)

        # --- Policy head (cross-attention decoder) ------------------
        # The slot queries are shared across the batch — we expand to
        # B and run cross-attention against the trunk output.
        batch_size = trunk_out.shape[0]
        queries = self.policy_slot_queries
        if policy_slot_range is not None:
            lo, hi = policy_slot_range
            if not (0 <= lo < hi <= queries.shape[0]):
                raise ValueError(
                    f"policy_slot_range {policy_slot_range!r} outside "
                    f"[0, {queries.shape[0]}]"
                )
            queries = queries[lo:hi]
        slot_q = queries.unsqueeze(0).expand(
            batch_size, -1, -1,
        )
        # Trunk output rows are keys/values. Mask out padding so the
        # decoder doesn't attend to it.
        attn_out, _ = self.policy_decoder_attn(
            query=slot_q,
            key=trunk_out,
            value=trunk_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        policy_logits = self.policy_head(attn_out).squeeze(-1)

        out: dict[str, torch.Tensor] = {
            "value": value,
            # Pre-tanh, pre-calibration. Exposed so the post-hoc
            # temperature fit (``value_calibration``) has something to
            # fit against — a squashed ±0.999 output has already lost
            # the information a temperature would restore.
            "value_logit": value_logit,
            "policy_logits": policy_logits,
        }

        if self.config.enable_hand_belief_head:
            hand_q = self.hand_belief_queries.unsqueeze(0).expand(
                batch_size, -1, -1,
            )
            hand_attn_out, _ = self.hand_belief_attn(
                query=hand_q,
                key=trunk_out,
                value=trunk_out,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hand_belief_logits = (
                self.hand_belief_head(hand_attn_out).squeeze(-1)
            )
            out["hand_belief_logits"] = hand_belief_logits

        return out


def make_dummy_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 8,
    config: Optional[StructuredModelConfig] = None,
    pad_lengths: Optional[list[int]] = None,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a random-but-valid batch dict for testing / smoke runs.

    Used by ``tests/training/test_structured_model.py`` and any
    downstream slice that wants a "dummy input that compiles" without
    standing up the data pipeline.

    ``pad_lengths`` — optional ``list[int]`` of per-row valid lengths
    (``<= seq_len``). When None, every row is fully valid (no
    padding). When provided, positions past the row's length are
    masked False; their integer / scalar contents stay nonzero
    so we can verify the mask actually suppresses them (gradients
    on masked-only inputs should be zero w.r.t. the embedding rows
    that only those positions touched — see the variable-length
    test).

    Returns a dict with the four required forward-input tensors.
    """
    cfg = config or StructuredModelConfig()
    gen = torch.Generator()
    gen.manual_seed(seed)

    token_type = torch.randint(
        low=0, high=N_TOKEN_TYPES, size=(batch_size, seq_len),
        generator=gen,
    )
    # Per-field categorical IDs respect each field's vocab size.
    field_vocabs = [
        cfg.n_printings,           # printing
        ACTION_KIND_VOCAB_SIZE,    # action_kind
        CONTROLLER_VOCAB_SIZE,     # controller
        cfg.n_printings,           # target_printing
        TARGET_KIND_VOCAB_SIZE,    # target_kind
        VARIANT_VOCAB_SIZE,        # variant
        STEP_VOCAB_SIZE,           # step
    ]
    token_emb_ids = torch.stack(
        [
            torch.randint(
                low=0, high=v, size=(batch_size, seq_len), generator=gen,
            )
            for v in field_vocabs
        ],
        dim=-1,
    )
    token_scalars = torch.randn(
        (batch_size, seq_len, cfg.scalar_width), generator=gen,
    )

    if pad_lengths is None:
        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool,
        )
    else:
        if len(pad_lengths) != batch_size:
            raise ValueError(
                f"pad_lengths must be length {batch_size}; "
                f"got {len(pad_lengths)}"
            )
        attention_mask = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool,
        )
        for row_idx, length in enumerate(pad_lengths):
            if length < 0 or length > seq_len:
                raise ValueError(
                    f"pad_lengths[{row_idx}]={length} outside "
                    f"[0, {seq_len}]"
                )
            attention_mask[row_idx, :length] = True
        # Guard: TransformerEncoder + MultiheadAttention raise NaN
        # when an entire row is fully padded. Force at least one
        # valid position per row to keep the smoke clean.
        for row_idx in range(batch_size):
            if not attention_mask[row_idx].any():
                attention_mask[row_idx, 0] = True

    batch: dict[str, torch.Tensor] = {
        "token_type": token_type,
        "token_emb_ids": token_emb_ids,
        "token_scalars": token_scalars,
        "attention_mask": attention_mask,
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


__all__ = [
    "ACTION_KIND_VOCAB_SIZE",
    "CONTROLLER_VOCAB_SIZE",
    "DEFAULT_N_PRINTINGS",
    "EMB_FIELD_ACTION_KIND",
    "EMB_FIELD_CONTROLLER",
    "EMB_FIELD_PRINTING",
    "EMB_FIELD_STEP",
    "EMB_FIELD_TARGET_KIND",
    "EMB_FIELD_TARGET_PRINTING",
    "EMB_FIELD_VARIANT",
    "N_EMB_FIELDS",
    "DEFAULT_SCALAR_SCHEMA_VERSION",
    "N_SCALAR_FIELDS",
    "N_SCALAR_FIELDS_V2",
    "N_TOKEN_TYPES",
    "SCALAR_FIELD_ATTACKING",
    "SCALAR_FIELD_BLOCKING",
    "SCALAR_FIELD_DAMAGE",
    "SCALAR_FIELD_IS_CREATURE",
    "SCALAR_FIELD_IS_VIEWER_TURN",
    "SCALAR_FIELD_MANA_COST_PAID",
    "SCALAR_FIELD_OPP_HAND",
    "SCALAR_FIELD_OPP_LANDS",
    "SCALAR_FIELD_OPP_LIBRARY",
    "SCALAR_FIELD_OPP_LIFE",
    "SCALAR_FIELD_OPP_POISON",
    "SCALAR_FIELD_OPP_UNTAPPED_LANDS",
    "SCALAR_FIELD_OTHER_COUNTERS",
    "SCALAR_FIELD_POWER",
    "SCALAR_FIELD_PT_COUNTERS",
    "SCALAR_FIELD_PUBLIC_MASK",
    "SCALAR_FIELD_STACK_SIZE",
    "SCALAR_FIELD_SUMMONING_SICK",
    "SCALAR_FIELD_TAPPED",
    "SCALAR_FIELD_TOUGHNESS",
    "SCALAR_FIELD_TURN_NUMBER",
    "SCALAR_FIELD_TURN_OFFSET",
    "SCALAR_FIELD_VIEWER_HAND",
    "SCALAR_FIELD_VIEWER_HAS_PRIORITY",
    "SCALAR_FIELD_VIEWER_LANDS",
    "SCALAR_FIELD_VIEWER_LIBRARY",
    "SCALAR_FIELD_VIEWER_LIFE",
    "SCALAR_FIELD_VIEWER_POISON",
    "SCALAR_FIELD_VIEWER_UNTAPPED_LANDS",
    "SCALAR_SCHEMA_V1",
    "SCALAR_SCHEMA_V2",
    "SUPPORTED_SCALAR_SCHEMA_VERSIONS",
    "scalar_width_for_schema",
    "STEP_VOCAB_SIZE",
    "StructuredModel",
    "StructuredModelConfig",
    "TARGET_KIND_VOCAB_SIZE",
    "TOKEN_TYPE_ACTION_EVENT",
    "TOKEN_TYPE_CLS",
    "TOKEN_TYPE_EXILE_OPPONENT",
    "TOKEN_TYPE_EXILE_VIEWER",
    "TOKEN_TYPE_GLOBAL",
    "TOKEN_TYPE_GRAVEYARD_OPPONENT",
    "TOKEN_TYPE_GRAVEYARD_VIEWER",
    "TOKEN_TYPE_HAND_OPPONENT",
    "TOKEN_TYPE_HAND_VIEWER",
    "TOKEN_TYPE_PERMANENT_OPPONENT",
    "TOKEN_TYPE_PERMANENT_VIEWER",
    "TOKEN_TYPE_STACK_ITEM",
    "VARIANT_VOCAB_SIZE",
    "make_dummy_batch",
]
