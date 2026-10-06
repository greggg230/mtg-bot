"""Content-addressed canonical hash of an MCTS-visible game state.

Used by the per-search NN-eval cache in :class:`MCTS` to deduplicate
leaf-evaluation work when multiple paths through the search tree
reach equivalent states (transpositions). Within a single
``choose_with_visits`` call, the cache maps ``canonical_state_key
(adapter)`` -> ``(value, policy_logits)``; the second time the search
reaches an equivalent state, the cached pair is reused and the NN
forward pass is skipped.

Equivalence rules — what's hashed and what's dropped
====================================================

The hash is **forward-relevant**: two adapters that produce identical
distributions over future sequences of legal decisions (under the
same agent / RNG) must hash equal; two adapters whose futures could
differ must hash differently.

Hashed (forward-relevant)
-------------------------

* **Adapter-side decision sub-state** (engine_adapter.py): ``_pass_count``,
  ``_combat_phase``, ``_pending_attackers``, ``_chosen_attackers``,
  ``_pending_blockers``, ``_chosen_blocks``, ``_combat_defender_id``,
  the resolution-pause block (``_resolution_paused``,
  ``_resolution_kind``, ``_resolution_player``, ``_resolution_options``,
  the channel agent's replay queue -- answers already given within the
  pending action -- and ``_pending_replay_decision``). The scout flagged that
  engine-state-only hashing produces false-positive transpositions in
  mid-combat-declaration sub-states; we include the whole adapter-side
  decision sub-state to avoid that class of bug.

* **Per-zone canonicalization** (per :class:`mtg_bot.game.Player`):
    - **Hand**: sorted tuple of ``(printing.name, warped, cast_via_warp,
      playable_from_exile_this_turn, play_from_exile_requires_kavu,
      cast_from_exile_with_lightstall_surcharge)``. Card.id dropped
      (mirrors the hand-action dedup we recently shipped).
    - **Graveyard**: sorted tuple of ``(printing.name,)``. ``warped``
      flag also included (a warped card in the graveyard might be
      castable). NOTE: a future card reading "bottom of graveyard" or
      "most-recently-graveyarded" would invalidate the order-invariance
      — none in EoE today.
    - **Exile**: sorted tuple of ``(printing.name, faceup-flag,
      exile-reason-tag-as-flag-tuple)`` so ``warped`` cards in exile
      (castable) are distinct from non-warp exiled copies.
    - **Library**: hash JUST ``len(library)``. Correct for
      PUCT-MCTS because chance nodes re-diverge — different orderings
      of the same library produce different futures via re-draw, but
      because the search has already determinized once at PIMC root
      and won't redraw until the next chance node, what we see here
      is fixed-order. If a future code path were to fix determinized
      library order BEFORE search starts, this must be tightened to
      include the top-N revealed cards.
    - **Battlefield**: full per-object structural hash —
      ``(printing.name, controller_id, tapped, summoning_sick,
      counters_tuple, attached_to_canonical_idx, *all_per-object_state)``.
      Sorted by the same canonical key so ordering doesn't matter.

* **Continuous-effects registry — content, not count.** The scout
  used ``len(state.continuous_effects)`` as a proxy. Production walks
  each entry by ``(layer, sublayer, timestamp, source_id_canonical,
  kind, target_filter, target_id_canonical, params_repr,
  expires_at, expires_at_player_id)``. ``source_id`` and ``target_id``
  remap through the canonical-battlefield index where applicable.
  Two registries with the same content in different list order are
  treated as equivalent (we sort by the structural key) — the engine
  itself sorts by ``(layer, sublayer, timestamp)`` at apply-time, so
  any deterministic ordering produces the same effective game state.

* **Stack target ids**: each ``StackItem.targets`` is a tuple of
  ``(kind, raw_id)``. ``raw_id`` gets remapped through the canonical-
  battlefield index. Today within one MCTS search the id namespace
  is stable across clones, but we don't bake that assumption into
  the hash — it's brittle and ``state.clone()`` semantics may evolve.

* **Per-player aggregates**: ``life``, ``poison_counters``,
  ``lands_played_this_turn``, ``lands_entered_this_turn``,
  ``spells_cast_this_turn``, ``noncreature_spells_cast_this_turn``,
  ``life_gained_this_turn``,
  ``artifacts_entered_under_your_control_this_turn``,
  ``has_lost``, mana pool counts, hand size,
  ``attacked_this_turn`` as a sorted tuple of canonical permanent
  indices (NOT raw GameObject refs).

* **Game-state scalars**: ``turn_number``, ``active_player_idx``,
  ``priority_player_idx``, ``step``,
  ``permanent_left_battlefield_this_turn``, ``spell_warped_this_turn``,
  ``extra_turn_queue``, ``creatures_died_this_turn``,
  ``prevent_combat_damage_this_turn``, ``delayed_triggers``,
  ``once_per_turn_fired``, ``ability_resolutions_this_turn``,
  ``players_who_created_tokens_this_turn``,
  ``pending_end_of_combat_sacrifices`` (remapped through canonical
  battlefield index).

Dropped (deliberately, with rationale)
--------------------------------------

* ``state.log`` — pure UI / history.
* Raw ``GameObject.id`` — used only as the input to canonicalization.
* ``GameObject.publicly_known_to`` — per-determinization scope means
  the cache lives within one PIMC sample where this doesn't vary
  across the search. An ISMCTS-direct path that reused this hash
  WITHOUT determinizing first would need to re-include this.
* ``state.rng`` — same forward play distribution across different
  RNG snapshots is "same state" for cache purposes (chance nodes
  re-sample regardless).
* ``state.next_object_id`` and ``state.next_timestamp`` — per-
  instance counters; the value influences nothing forward-relevant.
* ``_layered_cache_version`` and ``_layer6_wiped_cache`` — debug
  bookkeeping; recomputable.
* Mid-resolution mirror fields (``resolving_*``) — only meaningful
  during in-flight resolution, not at the priority moments where
  search holds the cache.

Output format
=============

The implementation builds a nested tuple of pure-Python primitives
and hashes its ``marshal.dumps`` serialization via blake2b to
produce a 32-byte digest. ``marshal`` is a defensible serializer
here: it's C-implemented (~5x faster than ``repr``), is
deterministic for a fixed Python version, supports every primitive
the canonical state tuple emits (None, bool, int, float, str,
bytes, tuple, frozenset, set, list, dict), and refuses anything
else with a TypeError — which protects against accidental
inclusion of non-primitives that would silently hash by
``id()``. The cache lifetime is one search call, so the
"format may change between Python versions" warning attached to
``marshal`` doesn't affect us. The hash is content-addressed
in the cryptographic sense — a 32-byte random-distributed key
— so collisions are negligible (birthday bound at 2^128 entries).

If you're unsure whether a field is forward-relevant, INCLUDE IT.
False-negatives (states that are equivalent but hash differently)
just lose cache hits — a perf cost. False-positives (states that
aren't equivalent but hash equal) corrupt search.
"""

from __future__ import annotations

import hashlib
import marshal
from collections import Counter
from typing import Any

from mtg_bot.game import (
    ContinuousEffect, GameObject, GameState, ManaCoin, Player, StackItem,
)


def _canonical_perm_key(
    obj: GameObject, attached_idx: int,
) -> tuple:
    """Forward-relevant per-object key for a battlefield permanent.

    ``attached_idx`` is the canonical battlefield index of
    ``obj.attached_to``, or -1 when the object is not attached.

    Performance: the field list overlaps heavily with ``_shape_key``;
    callers that have already computed a shape key for this object
    should use ``_canonical_perm_key_from_shape`` instead to avoid the
    re-walk. This function exists for tests / direct callers that
    don't have a precomputed shape key handy.
    """
    return _canonical_perm_key_from_shape(obj, _shape_key(obj), attached_idx)


def _canonical_perm_key_from_shape(
    obj: GameObject, shape: tuple, attached_idx: int,
) -> tuple:
    """Like :func:`_canonical_perm_key`, but reuses an already-computed
    ``shape`` key (the output of :func:`_shape_key` for ``obj``).

    The full permanent key is the shape key plus a small trailing set
    of fields that the shape key deliberately omits (they're either
    raw-id-bearing — ``blocked_by``, ``attached_to``,
    ``exiled_by_source`` — or per-castable-from-exile flags that the
    index-build doesn't need to disambiguate on, since same-printing
    same-shape permanents that differ only in those flags are still
    structurally equivalent for the index's "first occurrence wins"
    rule). Bundling the cached shape key into the prefix lets us avoid
    re-touching ~30 attributes per battlefield permanent during hash
    construction (the index build already paid for the shape walk).
    """
    return shape + (
        tuple(obj.blocked_by),
        attached_idx,
        obj.playable_from_exile_this_turn,
        obj.play_from_exile_requires_kavu,
        obj.cast_from_exile_with_lightstall_surcharge,
        # exiled_by_source is a raw id pointing at a battlefield (or
        # formerly-battlefield) source; for the canonicalization we
        # collapse it to a presence flag because the source id itself
        # is per-instance. If two arms of the search both have an
        # equipment "tagged by Oblivion Ring", the tag's identity
        # doesn't affect future play distinctly from "is tagged".
        obj.exiled_by_source is not None,
    )


def _canonical_zone_card(obj: GameObject) -> tuple:
    """Forward-relevant key for a card in a non-battlefield zone.

    Used for hand / graveyard / exile entries — these don't have the
    battlefield-permanent state machinery, and ``card.id`` is dropped
    so two equivalent same-printing copies hash identically.
    """
    return (
        obj.printing.name,
        obj.warped,
        obj.cast_via_warp,
        obj.face_down,
        obj.playable_from_exile_this_turn,
        obj.play_from_exile_requires_kavu,
        obj.cast_from_exile_with_lightstall_surcharge,
    )


def _canonical_mana_pool(pool: list[ManaCoin]) -> tuple:
    """Mana pool: count by ``(color, spend_restriction)``. Order in
    the pool list doesn't affect downstream play.

    ``c.spend_restriction`` is ``Optional[str]`` — None for unrestricted
    coins, a non-empty tag string ("artifact_spell",
    "colorless_eldrazi", etc., see :class:`ManaCoin` docstring) for
    restricted coins. We coerce None to the empty string ``""`` BEFORE
    counting so the outer ``tuple(sorted(...))`` over
    ``Counter.items()`` doesn't crash when one coin has
    ``spend_restriction=None`` and another has ``"artifact_spell"``:
    Python tuple comparison would otherwise hit ``TypeError: '<' not
    supported between instances of 'NoneType' and 'str'`` when
    comparing two ``((color, restriction), count)`` pairs whose
    ``(color, restriction)`` keys disagree only on the second slot.
    Empty string is a safe sentinel: real restriction tags are always
    non-empty strings, so ``""`` cannot collide with a real value.
    Surfaced live in iter_005 self-play (1 crash / 1000 games)."""
    return tuple(sorted(
        Counter(
            (c.color, c.spend_restriction if c.spend_restriction is not None else "")
            for c in pool
        ).items()
    ))


def _canonical_stack_item(
    item: StackItem,
    raw_id_to_canon: dict[int, int],
) -> tuple:
    """Forward-relevant key for a stack item.

    ``targets`` is remapped through ``raw_id_to_canon`` so that
    "Doom Blade targeting permanent A" and "Doom Blade targeting an
    equivalent copy A'" hash identically. Permanent ids are remapped
    through the canonical-battlefield index; any non-permanent target
    id (player ids, stack ids) falls through unchanged with -1 if it
    wasn't a permanent.
    """
    canon_targets = tuple(
        (
            kind,
            raw_id_to_canon.get(raw_id, raw_id)
            if kind == "permanent" else raw_id,
        )
        for kind, raw_id in item.targets
    )
    cost_paid_canon = tuple(
        raw_id_to_canon.get(o.id, -1)
        for o in item.cost_paid_objects
    )
    return (
        item.source.printing.name,
        item.controller_id,
        item.ability is not None,
        type(item.ability).__name__ if item.ability is not None else None,
        canon_targets,
        item.countered,
        item.x_value,
        item.kicked,
        item.chosen_modes,
        item.cast_room_half,
        item.combat_damage_amount,
        item.combat_damage_source_controller_id,
        item.counter_amount,
        item.additional_cost_choice_id is not None,
        cost_paid_canon,
    )


def _canonical_continuous_effect(
    ce: ContinuousEffect,
    raw_id_to_canon: dict[int, int],
) -> tuple:
    """Forward-relevant key for a :class:`ContinuousEffect` registry
    entry.

    ``source_id`` and ``target_id`` get remapped through the canonical
    battlefield index when present. Two registries with the same
    content in different list order produce equivalent game state
    (the engine sorts by ``(layer, sublayer, timestamp)`` when
    applying), so the caller sorts the per-effect tuples to get an
    order-invariant hash for the registry as a whole.

    NOTE: ``timestamp`` is included verbatim. Different timestamps
    produce different layer-application order within the same layer
    + sublayer, and that affects effective P/T / abilities. We don't
    canonicalize timestamps to "rank within layer" because identical-
    rank but different absolute timestamps don't otherwise differ in
    forward play.
    """
    # Optional[int] fields use ``-1`` as the absent-sentinel rather
    # than ``None`` so the per-effect tuples stay sortable: the caller
    # does ``tuple(sorted(...))`` over the registry, Python tuple
    # comparison walks elements positionally, and one effect with
    # ``target_id=None`` against another with ``target_id=42`` raises
    # ``TypeError: '<' not supported between instances of NoneType
    # and int``. -1 is safe — object ids, turn numbers, and player
    # ids are all >= 0 in this engine, so a real value can never
    # collide with the sentinel.
    return (
        ce.layer,
        ce.sublayer,
        ce.timestamp,
        raw_id_to_canon.get(ce.source_id, ce.source_id),
        ce.kind,
        ce.target_filter,
        (
            raw_id_to_canon.get(ce.target_id, ce.target_id)
            if ce.target_id is not None else -1
        ),
        repr(ce.params),
        ce.expires_at if ce.expires_at is not None else -1,
        ce.expires_at_player_id if ce.expires_at_player_id is not None else -1,
    )


def _shape_key(obj: GameObject) -> tuple:
    """Subset of the structural key used to build the canonical
    battlefield index. We can't use the full ``_canonical_perm_key``
    because the full key references ``attached_to`` via canonical
    index — exactly what we're trying to build here.

    Includes printing.name + every per-instance-comparable scalar
    that distinguishes "the same printing on the battlefield" from
    "a structurally distinct copy". Excludes ``attached_to`` and
    ``blocked_by`` (lists of raw ids; remap target).

    Optional fields use sentinel coercion (None -> -1 for Optional[int],
    None -> "" for Optional[str]) so the shape tuple stays sortable.
    The shape is the sort key used by
    :func:`_build_canonical_battlefield_index` — Python tuple
    comparison walks elements positionally and would raise
    ``TypeError: '<' not supported between 'NoneType' and 'str'/'int'``
    when two battlefield permanents differ at a slot where one carries
    None and the other carries a real value (e.g. one Painter's Servant
    that picked a color vs one that hasn't, or a permanent whose
    controller has been aura-swapped vs one that hasn't, or a "becomes
    a copy" target vs an unmodified peer). The sentinels (-1 for ids /
    counters / turn numbers — real values are always >= 0; "" for
    string tags — real values like printing names and color codes are
    always non-empty strings) cannot collide with real values, so the
    canonical-index identity rule ("structurally equivalent permanents
    receive the same canonical index") is preserved.
    """
    return (
        obj.printing.name,
        obj.controller_id,
        obj.tapped,
        obj.summoning_sick,
        obj.attacking,
        obj.blocking is not None,
        obj.damage,
        obj.damaged_this_turn,
        obj.marked_lethal,
        obj.plus_one_counters,
        obj.minus_one_counters,
        obj.charge_counters,
        obj.loyalty_counters,
        obj.stun_counters,
        obj.time_counters,
        obj.lore_counters,
        obj.warped,
        obj.cast_via_warp,
        obj.cast_via_impending,
        obj.impending_active,
        obj.cast_via_hand,
        obj.face_down,
        obj.phased_out,
        obj.regeneration_shield,
        obj.permanent_is_creature,
        obj.enduring_returned_as_enchantment,
        obj.has_activated_once_only_ability,
        # Optional[str] — None for non-Painter sources, "W"/"U"/... for
        # a Painter that resolved its color choice. "" sentinel.
        obj.painter_chosen_color if obj.painter_chosen_color is not None else "",
        tuple(sorted(obj.unlocked_halves)),
        obj.temp_power_bonus,
        obj.temp_toughness_bonus,
        # Optional[int] — None until a layer-2 control-change captures
        # the baseline; -1 sentinel (player ids are always >= 0).
        (
            obj.controller_id_before_any_aura_swap
            if obj.controller_id_before_any_aura_swap is not None else -1
        ),
        # Optional[CardPrinting] -> str-or-"" — None until a "becomes a
        # copy" effect mutates ``printing``. "" sentinel (real
        # ``CardPrinting.name`` values are always non-empty strings).
        obj.original_printing.name
        if obj.original_printing is not None else "",
    )


def _build_canonical_battlefield_index(
    state: GameState,
) -> tuple[list[int], dict[int, int], list[tuple]]:
    """Build the canonical-battlefield index used for ``attached_to``
    and stack-target remapping.

    Returns ``(sorted_indices_into_battlefield, raw_id_to_canonical_idx,
    shapes_in_sorted_order)``. The third element lets the caller reuse
    the per-object shape keys (the bulk of each permanent's
    canonical-key fields) without re-walking ~30 attributes per
    permanent during hash construction.

    Critically: structurally-equivalent permanents (same ``_shape_key``)
    receive the SAME canonical index — specifically, the canonical
    index of the first occurrence in the sorted order. This is what
    makes "aura attached to copy A" and "aura attached to equivalent
    copy A'" hash identically. If we instead gave each permanent a
    unique sort position (via raw-id tiebreak), the canonical index
    would differ between the two copies and the attached_to-remapped
    keys would diverge, breaking the equivalence rule.

    The forward-play justification: two structurally-equivalent
    permanents are interchangeable from the rules engine's
    perspective; whichever one a continuous effect / aura targets,
    the resulting game state is the same. So they MUST share a
    canonical index for hash equivalence to hold.
    """
    bf = state.battlefield
    if not bf:
        return [], {}, []
    # Compute every shape key exactly once. The previous implementation
    # called ``_shape_key`` twice per permanent (once as the sort key
    # and once during the dedup walk); a 14-permanent battlefield made
    # this ~28 walks per hash, which compounded into 20%+ of total
    # canonical-state-key time on big-board states like declare_atk.
    shapes = [_shape_key(o) for o in bf]
    indices = sorted(range(len(bf)), key=shapes.__getitem__)
    sorted_shapes = [shapes[i] for i in indices]
    raw_id_to_canon: dict[int, int] = {}
    last_shape: tuple | None = None
    last_canon_idx = -1
    for pos, i in enumerate(indices):
        shape = sorted_shapes[pos]
        if shape != last_shape:
            last_canon_idx = pos
            last_shape = shape
        raw_id_to_canon[bf[i].id] = last_canon_idx
    return indices, raw_id_to_canon, sorted_shapes


def _canonical_player(
    p: Player, raw_id_to_canon: dict[int, int],
) -> tuple:
    """Forward-relevant per-player aggregate."""
    hand_keys = tuple(sorted(_canonical_zone_card(c) for c in p.hand))
    grave_keys = tuple(sorted(_canonical_zone_card(c) for c in p.graveyard))
    exile_keys = tuple(sorted(_canonical_zone_card(c) for c in p.exile))
    attacked_canon = tuple(sorted(
        raw_id_to_canon.get(o.id, -1) for o in p.attacked_this_turn
    ))
    return (
        p.id,
        p.life,
        p.starting_life,
        p.poison_counters,
        len(p.library),
        len(p.hand),
        hand_keys,
        grave_keys,
        exile_keys,
        _canonical_mana_pool(p.mana_pool),
        p.lands_played_this_turn,
        p.lands_entered_this_turn,
        p.spells_cast_this_turn,
        p.noncreature_spells_cast_this_turn,
        p.life_gained_this_turn,
        p.artifacts_entered_under_your_control_this_turn,
        attacked_canon,
        p.has_lost,
    )


def _canonical_state_tuple(adapter: Any) -> tuple:
    """Build the nested tuple that the digest is derived from.

    Exposed (under-prefix) for tests that want to inspect the exact
    structural input rather than just the digest.
    """
    state: GameState = getattr(adapter, "engine_state", adapter)

    indices, raw_id_to_canon, sorted_shapes = (
        _build_canonical_battlefield_index(state)
    )

    bf_keys = []
    for pos, i in enumerate(indices):
        obj = state.battlefield[i]
        attached_idx = (
            raw_id_to_canon.get(obj.attached_to, -1)
            if obj.attached_to is not None else -1
        )
        bf_keys.append(_canonical_perm_key_from_shape(
            obj, sorted_shapes[pos], attached_idx,
        ))

    stack_keys = tuple(
        _canonical_stack_item(it, raw_id_to_canon) for it in state.stack
    )

    # Continuous-effects registry: structural walk; sort to drop
    # list-order dependence.
    ce_keys = tuple(sorted(
        _canonical_continuous_effect(ce, raw_id_to_canon)
        for ce in state.continuous_effects
    ))

    player_keys = tuple(
        _canonical_player(p, raw_id_to_canon) for p in state.players
    )

    # Adapter-side decision sub-state. Fields living around
    # engine_adapter.py:215, 229-248 — all materially affect the next
    # legal action. ``getattr`` with defaults keeps this resilient if
    # ``adapter`` isn't actually an MtgGameStateAdapter (e.g. a unit
    # test passes the engine state directly).
    adapter_block: tuple
    if hasattr(adapter, "_pass_count"):
        # Resolution-pause options often contain raw ids; the natural
        # remap uses the canonical index. We ``repr`` the options
        # tuple for simplicity — its structure varies by
        # ``_resolution_kind`` and isn't always remappable. Fast-path
        # the (very common) empty-tuple case to skip the ``repr`` call
        # entirely when nothing's pending.
        res_opts = adapter._resolution_options
        res_opts_key = "" if not res_opts else repr(res_opts)
        # Answers already given within the pending action: two
        # different answers to a pause can lead to the SAME follow-up
        # pause on the same pre-action snapshot, so the queue is part
        # of the forward-relevant state.
        channel = getattr(adapter, "_channel_agent", None)
        res_queue_key = (
            repr(channel.queue)
            if channel is not None and channel.queue else ""
        )
        pending_replay = adapter._pending_replay_decision
        adapter_block = (
            adapter._pass_count,
            adapter._combat_phase,
            tuple(adapter._pending_attackers),
            tuple(adapter._chosen_attackers),
            tuple(adapter._pending_blockers),
            tuple(sorted(adapter._chosen_blocks.items())),
            adapter._combat_defender_id,
            adapter._resolution_paused,
            adapter._resolution_kind,
            adapter._resolution_player,
            res_opts_key,
            res_queue_key,
            repr(pending_replay) if pending_replay is not None else None,
        )
    else:
        adapter_block = ()

    # Delayed triggers / once-per-turn / token-creation /
    # pending-end-of-combat-sacrifices: each is a small structure of
    # ids. Where the id lives on the battlefield, remap through
    # canonical index; where the id is a player id (delayed-trigger
    # controller, token-creator), leave as-is. Delayed-trigger values
    # can be arbitrary engine objects (e.g. effect-class instances
    # like ``WarpExileSelf``) that ``marshal.dumps`` rejects with
    # ``ValueError``; we ``repr`` non-id values so the canonical key
    # remains a marshallable primitive tree. ``repr`` is deterministic
    # for the engine's frozen-dataclass / tuple effect types.
    delayed_canon = tuple(
        tuple(sorted(
            (
                k,
                (
                    raw_id_to_canon.get(v, v)
                    if k in ("target_id", "source_id") else (
                        v if isinstance(v, (int, float, str, bool, type(None)))
                        else repr(v)
                    )
                ),
            )
            for k, v in dt.items()
        ))
        for dt in state.delayed_triggers
    )
    once_per_turn_canon = tuple(sorted(
        (raw_id_to_canon.get(sid, sid), idx)
        for sid, idx in state.once_per_turn_fired
    ))
    pending_eoc_sacs_canon = tuple(sorted(
        raw_id_to_canon.get(oid, -1)
        for oid in state.pending_end_of_combat_sacrifices
    ))
    ability_resolutions_canon = tuple(sorted(
        (repr(k), v) for k, v in state.ability_resolutions_this_turn.items()
    ))
    return (
        state.turn_number,
        state.active_player_idx,
        state.priority_player_idx,
        state.step.value if state.step is not None else None,
        state.permanent_left_battlefield_this_turn,
        state.spell_warped_this_turn,
        tuple(state.extra_turn_queue),
        state.creatures_died_this_turn,
        state.prevent_combat_damage_this_turn,
        state.suppress_exiled_by_source_on_leave_for is not None,
        state.last_exile_recall_count,
        state.is_draw,
        state.winner_id,
        delayed_canon,
        once_per_turn_canon,
        ability_resolutions_canon,
        tuple(sorted(state.players_who_created_tokens_this_turn)),
        pending_eoc_sacs_canon,
        ce_keys,
        player_keys,
        tuple(bf_keys),
        stack_keys,
        adapter_block,
    )


def canonical_state_key(adapter: Any) -> bytes:
    """Return a 32-byte content-addressed hash of an MCTS-visible
    state.

    See module docstring for the equivalence rules. The result is
    suitable as a ``dict`` key. Hashing the same state twice produces
    the same bytes; equivalent states (per the rules) hash equal;
    inequivalent states hash differently with cryptographic confidence.

    ``adapter`` may be either a :class:`MtgGameStateAdapter` (the
    canonical case used by MCTS) or a raw :class:`mtg_bot.game.GameState`
    (used by tests / scripts that don't need the adapter-side decision
    sub-state). When given a raw engine state, the adapter-block tail
    is the empty tuple — the hash is well-defined but DIFFERENT from
    the same state wrapped in an adapter, which is the correct
    behavior (an adapter at, say, ``_pass_count=2`` is not equivalent
    to a raw engine state where ``_pass_count`` is implicitly zero).
    """
    tup = _canonical_state_tuple(adapter)
    # ``marshal.dumps`` is ~5x faster than ``repr(tup).encode()`` for
    # the deeply-nested primitive tuple we build here, and produces
    # deterministic bytes for a fixed Python version. Per-search cache
    # lifetime makes cross-version stability irrelevant.
    #
    # If a non-primitive slips into the tuple (engine evolution adds a
    # new field whose values aren't already coerced via ``repr``),
    # ``marshal.dumps`` raises ``ValueError: unmarshallable object``.
    # We fall back to ``repr(tup).encode("utf-8")`` rather than crashing
    # the search — the digest still satisfies the equivalence rules
    # (repr is deterministic for the engine's frozen-dataclass / tuple
    # types), it's just slower. In practice the production code paths
    # exercised in tests stay on the marshal fast path; the fallback is
    # a defensive guard, not a routine path.
    try:
        return hashlib.blake2b(marshal.dumps(tup), digest_size=32).digest()
    except ValueError:
        return hashlib.blake2b(
            repr(tup).encode("utf-8"), digest_size=32,
        ).digest()
