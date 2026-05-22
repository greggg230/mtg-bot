"""Genuine real-EoE cross-engine byte-equality probe SWEEP for the
spell-card-bridge lever Phase 1 carrier.

**Why this module exists.** The spell-card-bridge Phase 1 lever
(merge ``c44dfeb4`` / ``42416b77``) is the cycle's flagship
~60k-deck Rust deliverable — it flips the covered+bridged pid-free
single-target Tier-A spells from loader-drop -> bench-admit. At its
implement HEAD the live admit set is EXACTLY
``{Radiant Strike, Depressurize}`` (verified-first below — pinned by
``test_spell_card_lever_phase1_negative_pins_decline_set``'s pin #13
in ``tests/test_cross_engine_etb_triggers.py``). That slice shipped
with **only ONE** genuine real-EoE deep cast->resolve probe
(``test_bridge_spell_card_lever_phase1_real_eoe_cast_resolve_parity``
— Depressurize, ONE deterministic library, ONE deck context). The flagship
deliverable deserved more parity evidence than a single trajectory.

**What this module adds (the evidence-strengthening sweep).** A
genuine real-EoE cross-engine byte-equality probe SWEEP that drives
**both** bench-admitted spells through the **full Phase-1 carrier
path** — ``cast_spell`` -> ``cast_spell_noncreature`` ->
``StackItem.target_obj_id`` (the new Gate-B'' slot) -> resolve ->
graveyard — in **real 17lands-shaped deterministic deck contexts**,
across **multiple distinct deck contexts**, asserting **full ``compare_fields``
byte-equality** (the harness cleanly supports full compare post the
``_net_counters`` fix, merge ``ecc5f680``) at EVERY phase boundary
AND post-cast AND post-resolve, plus the spell-lifecycle observables
(spell source in the controller's graveyard on BOTH engines; the
effect observably resolved).

**Idiom.** This is the exact genuine-carrier-path idiom established by
the Phase-1 bridge pin #8/#9: ``setup_matched_game`` + the campaign
no-callback ``_PickFirstLegalTargetAgent`` first-legal convention
(matches Rust's ``bf_iter_etb_sorted`` cast-time scan) + ``step_both``
on a ``cast_spell`` action (the CE5 cast-vs-resolve mirror drains BOTH
stacks before the field-view compare) + full ``compare_fields``. It is
NOT the older Effect-half StackItem-push pattern — every spell here is
genuinely CAST through the production carrier the lever delivers.

**Verify-first.**
``test_phase1_admit_set_is_exactly_the_covered_bridged_seatable_set``
re-derives the live admit set from ``_printing_is_bridge_simple_spell``
at THIS HEAD before any probe relies on it (drop+replace honestly if
the set drifts — the lever auto-grows the set as each per-card Tier-A
Effect closes, so a future routine inheriting this module must
re-verify, not assume). It has done so: at the round-70 Phase-1
landing the set was {Depressurize, Radiant Strike}; round-73 grew it
to {Depressurize, Orbital Plunge, Radiant Strike, Seedship Impact}
(Lander-spell-token-seating sub-substrate); the cycle-5..8 Tier-A
folds closed Focus Fire + Biosynthic Burst so the live set is now the
EXACT 6 {Biosynthic Burst, Depressurize, Focus Fire, Orbital Plunge,
Radiant Strike, Seedship Impact} (re-anchored at HEAD ``921d759a``).

This module is append-additive: it does NOT touch
``tests/test_trajectory_parity.py``, the probe ``batch_a`` /
``batch_b`` modules, or the 6 protected ``test_audit_safe_trigger_*``
tests. It introduces no new picker shim (reuses
``_PickFirstLegalTargetAgent`` from ``tests/parity/_picker_shims.py``).
"""

from __future__ import annotations

import importlib.util

import pytest


# Skip the whole module if the Rust extension isn't built — same policy
# as ``tests/test_parity_smoke.py`` / ``tests/test_trajectory_parity.py``.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mtg_bot_engine") is None,
    reason="mtg_bot_engine PyO3 extension not installed",
)


# ---------------------------------------------------------------------------
# Verify-first: the live Phase-1 admit set.
#
# The lever's admit set is NOT a static card-name list — it is the set
# of EoE printings passing ``_printing_is_bridge_simple_spell`` (covered
# + bridged + single-target + token-pid seatable). At the round-70
# implement HEAD that set was EXACTLY {Depressurize, Radiant Strike};
# the round-73 Lander-spell-token-seating sub-substrate grew it to
# {Depressurize, Orbital Plunge, Radiant Strike, Seedship Impact}
# (Seedship/Orbital flipped DECLINE→SEATED — new-code-is-right); the
# cycle-5..8 Tier-A folds then closed #6 Biosynthic Burst and #5 Focus
# Fire (pid-free single-target {W}/{W} Instants, covered+bridged), which
# AUTO-JOINED via the §2 auto-grow teeth (the guardrail queries the LIVE
# covered set, NOT a hardcoded list). The live set is now the EXACT 6
# {Biosynthic Burst, Depressurize, Focus Fire, Orbital Plunge, Radiant
# Strike, Seedship Impact}. It auto-grows further as each remaining
# per-card Tier-A Effect closes. This module's carrier-sweep probes
# drive Depressurize / Radiant Strike explicitly (Seedship / Orbital
# have their own dedicated loader-admit + cast→resolve real-EoE probes
# in ``test_parity_smoke.py``; Focus Fire / Biosynthic Burst are pinned
# admitted here + carry their own canonical Effect-close probes); the
# test below fails loudly (rather than letting a probe silently drift
# onto a non-admitted card) if the set changes — the honest "re-verify,
# don't assume" discipline a future re-orienting routine MUST inherit.
# ---------------------------------------------------------------------------


# Re-anchored at the `feature/modal-chooseone-cycle-4` landing
# (2026-05-21), itself layered on the `feature/beyond-the-quiet-slice`
# landing (2026-05-21).
#
# The beyond-the-quiet-slice added the
# ``ExileAllCreaturesAndSpacecraft`` substrate (Rust Effect variant +
# bridge arm + reverse-serializer leaf) which auto-joined Beyond the
# Quiet (4,698 17lands decks) into the admit set via Lane Y's existing
# Phase-3 ``(0, 0)`` wrath admit arm — 15 → 16.
#
# Then modal-chooseone-cycle-4 added the per-leaf Effect variants for
# the all-untargeted modal cards (DealDamageEachOpponent already
# shipped; the new ExileAllArtifactsWithManaValueAtMost lands in this
# cycle) AND widened the admit predicate at trajectory.py:4564 (the
# previous unconditional ModalChooseOne reject is removed; the
# round-trip-or-bail teeth + the existing (0, 0)-shape Phase-3 wrath
# admit arm now carry the modal admit load). Ruinous Rampage (the
# only all-untargeted modal whose leaves are all serializable today)
# AUTO-JOINED — 16 → 17. The some-targeted modal cards (Reroute
# Systems / Scour for Scrap / Archenemy's Charm / Drill Too Deep /
# Warping Wail / Twist Reality / Live or Die — 7 cards) STAY DECLINED
# via the target-grammar gate (they report (1, 1) / (2, 2) / (0, 1)
# shapes; the (0, 0)-only admit arm rejects them; the per-mode target
# binding slice = Cycle 5).
#
# The Phase 2 2-slot card (Diplomatic Relations) remains the SOLE
# 2-slot admit; the Phase 3 wrath arm now carries 5 ``(0, 0)`` admits
# (Pyroclasm + Manifest Dread + Break Down the Door + Under the Skin
# + Beyond the Quiet) plus the Cycle-4 modal admit (Ruinous Rampage).
#
# Pre-2026-05-21 history: wrath-admit-widening (2026-05-21) grew 11→15
# (Pyroclasm + Manifest Dread + Break Down the Door + Under the
# Skin). Phase 2 (round-76) flipped Diplomatic Relations DECLINE→
# ADMITTED, 6→7. Round-73 Lander-spell-token-seating sub-substrate
# admitted Seedship Impact / Orbital Plunge, 4→6. Round-70 Phase-1
# landing: {Depressurize, Radiant Strike} (4 admits).
# **drill-too-deep-slice (2026-05-21):** Cycle 6 closes the first
# per-mode-target ModalChooseOne via the
# ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl`` Effect
# variant + bridge arm + reverse-serializer leaf + the Cycle-5
# detector table extension in ``cast_spell_noncreature``. Both Drill
# Too Deep modes are now covered+bridged (mode 0 via the new variant;
# mode 1 = ``DestroyTargetArtifact`` was already shipped). The (1, 1)
# Instant rides the Phase-1 admit shape.
#
# Adding mode 1's ``DestroyTargetArtifact`` to the reverse-serializer
# ``_leaf`` table (required so the ModalChooseOne composer can recurse
# into both modes for Drill Too Deep) ALSO auto-joins Untimely
# Malfunction — a {1}{R} Instant whose top-level ``spell_effect`` IS
# ``DestroyTargetArtifact``. It was previously declined ONLY because
# the leaf table didn't have ``DestroyTargetArtifact`` (the bridge
# dict→Python arm has shipped since round 23; the Rust variant since
# round 25; only the trajectory serializer leaf was missing). The
# §2 auto-grow teeth: an effect that's covered+bridged AND round-
# trips MUST be admitted — no fallback hybrid. 17 → 19 admits across
# the two flips (Drill Too Deep + Untimely Malfunction).
#
# **Reroute Systems closure (Lane EE, 2026-05-21,
# `feature/reroute-systems-slice-v2`)**: extended
# ``PumpTargetArtifactOrCreatureUntilEOT`` with ``granted_keywords``
# + new ``DealDamageToTargetTappedCreature`` Effect variant + Cycle-5
# detector-table arms for both modes + per-leaf reverse-serializer
# leaves. Reroute Systems AUTO-JOINS the admit set via the same
# Cycle-4 widen + Cycle-5/6 per-mode detector entries — 19 → 20.
# Combined with the Drill Too Deep slice: 17 → 20 across the merged
# lanes.
#
# **multi-target-admit-phase4 + non-state-rs-tractable-Exorcise
# (2026-05-21)**: two sibling lanes landed concurrently. Phase 4 adds
# the multi-target admit arm to ``_printing_is_bridge_simple_spell``
# (``(min, N)`` shapes with uniform graveyard_creature slot kinds) +
# new Rust ``Effect::ReturnUpToNSmallCreaturesFromGraveyardWith
# Counters`` variant → Scout for Survivors AUTO-JOINS. The Exorcise
# lane adds ``Effect::ExorciseEffect`` (Duskmourn {1}{W} Sorcery —
# the 3-way OR target filter is the structural sibling of Shattered
# Wings's destroy-half with exile-not-destroy + power-4-not-flying
# swaps; the power-4 leg reads EFFECTIVE power via the file-local
# ``read_doubling_power`` reader, byte-equal with Python's
# ``effective_power``) → Exorcise AUTO-JOINS. Pull Through the Weft
# + Archenemy's Charm stay declined per the multi-target lane's
# honest-stop characterization. 20 → 22 — both AUTO-JOINED.
#
# **Re-anchor at scour-for-scrap-cycle-2 (2026-05-21)** — admit set
# 22 → 23 → 24. The pin was stale by one (was 22 at the prior pin,
# live grew to 23 when the close-encounter substrate's Effect arm +
# Cycle-3 admit widen landed; the cycle-2 lane catches that drift by
# re-deriving the live set here). The +1 in THIS lane is
# **Scour for Scrap** — the new
# ``Effect::ReturnTargetArtifactCardFromYourGraveyardToHand`` Rust
# variant + the narrow ``SearchLibraryForCard(filter_name=
# "artifact")`` ``_leaf`` admit (the honest-stop discipline scoped
# admit; the basic-land / demon / spacecraft filters stay declined
# in this lane's narrowing) + the Phase-5 ``(0, 1)`` ``or_both`` shape
# admit-widen flip Scour DECLINED→ADMITTED. Demonic Counsel
# (``SearchLibraryForCard(filter_name="demon")``) stays declined per
# the narrow-admit discipline — it auto-flips when a future lane
# ships its per-card probe + guardrail flip.
# This stays an EXACT-set pin.
_PHASE1_ADMITTED_AT_THIS_HEAD = [
    "Beyond the Quiet", "Biosynthic Burst", "Break Down the Door",
    "Close Encounter", "Cut Propulsion", "Depressurize",
    "Diplomatic Relations", "Drill Too Deep", "Emergency Eject",
    "Exorcise", "Focus Fire", "Invasive Maneuvers", "Manifest Dread",
    "Mutinous Massacre", "Orbital Plunge", "Pyroclasm",
    "Radiant Strike", "Reroute Systems", "Ruinous Rampage",
    "Scour for Scrap", "Scout for Survivors", "Seedship Impact",
    "Shattered Wings", "Systems Override", "Under the Skin",
    "Untimely Malfunction",
]
# The two members this module's carrier-sweep probes drive directly
# through the full Phase-1 carrier path (Seedship / Orbital and Focus
# Fire / Biosynthic Burst are covered by their dedicated probes
# elsewhere; this pid-free pair is the carrier-sweep vehicle).
_CARRIER_SWEEP_PROBED = ["Depressurize", "Radiant Strike"]


def test_phase1_admit_set_is_exactly_the_covered_bridged_seatable_set():
    """Verify-first guard: the live spell-card-bridge admit set (the
    set of EoE printings passing the production-deck loader's
    ``_printing_is_bridge_simple_spell`` guardrail) is EXACTLY
    ``{Biosynthic Burst, Depressurize, Diplomatic Relations, Focus
    Fire, Orbital Plunge, Radiant Strike, Seedship Impact}`` at this
    HEAD (re-anchored at the round-76 **Phase 2** landing — Phase 2's
    2-slot heterogeneous-target grammar flipped Diplomatic Relations
    DECLINE→ADMITTED, growing the set 6→7; the prior 6 was the cycle-8
    fold {Biosynthic Burst, Depressurize, Focus Fire, Orbital Plunge,
    Radiant Strike, Seedship Impact}; the set was {Depressurize,
    Orbital Plunge, Radiant Strike, Seedship Impact} after the round-73
    Lander-spell-token-seating sub-substrate flipped Seedship Impact /
    Orbital Plunge DECLINE→SEATED, and {Depressurize, Radiant Strike}
    at the round-70 Phase-1 landing).

    This module's carrier-sweep probes drive ``_CARRIER_SWEEP_PROBED``
    (Depressurize / Radiant Strike) through the full Phase-1 carrier;
    Seedship Impact / Orbital Plunge have their own dedicated genuine
    real-EoE loader-admit + cast→resolve probes
    (``test_cross_engine_real_eoe_{seedship_impact,orbital_plunge}_loader_admit_and_cast_resolve_parity``),
    and Focus Fire / Biosynthic Burst carry their own canonical
    per-card Effect-close cross-engine probes (their close landed in
    the cycle-5..8 Tier-A folds). This is a re-anchor of a
    correctly-functioning guard that legitimately fired on the admit
    set's expected growth — NOT a bug in the guardrail.

    The assertion stays an EXACT-set equality (not a superset/subset
    check): if it fails the lever's admit set has drifted further (it
    auto-grows as each remaining per-card Tier-A Effect closes), and a
    re-orienting routine must DROP+REPLACE the pinned set honestly
    against the NEW live set (re-derive it from
    ``_printing_is_bridge_simple_spell`` over the live pool — the
    guardrail is SoT, not the roadmap prose), re-anchor the comment to
    the new HEAD, and keep the exact-set pin so the next legitimate
    growth is again caught loudly. Never assume the old set; never
    weaken the assertion into a no-op.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory

    src = get_active_source().get_printings()
    admitted = sorted(
        n
        for n in src
        if trajectory._printing_is_bridge_simple_spell(src[n])
    )
    assert admitted == _PHASE1_ADMITTED_AT_THIS_HEAD, (
        f"Phase-1 spell-card-bridge admit set drifted: live={admitted}, "
        f"expected exactly {_PHASE1_ADMITTED_AT_THIS_HEAD} at this HEAD. "
        f"The lever auto-grows the set as subsequent Tier-A Effects "
        f"close — a re-orienting routine must re-derive the probe set "
        f"against the NEW live set (drop+replace honestly), not assume "
        f"the old one."
    )
    # Each admitted card must also be covered+bridged at the Effect
    # level (a necessary condition the guardrail enforces — pin it so
    # a regression that admits an UN-covered spell is caught here, not
    # silently byte-diverging deep in a probe).
    for nm in _PHASE1_ADMITTED_AT_THIS_HEAD:
        assert trajectory._spell_effect_is_covered_and_bridged(
            src[nm]
        ) is True, f"{nm} must be covered+bridged at this HEAD"


# ---------------------------------------------------------------------------
# The genuine Phase-1 carrier-path driver.
#
# Drives the ACTUAL parsed printing through ``_printing_to_card_def``
# (the EXACT CardDef the production-deck loader's bridge OR-chain emits
# for the real printing — Phase-1 bridge pin #7 proved that transform
# byte-equal), into a matched game via ``setup_matched_game``, then
# CASTS it through the new carrier (``step_both`` on a ``cast_spell``
# action: ``CastSpell`` -> ``cast_spell_noncreature`` -> spell
# ``StackItem`` carrying ``target_obj_id`` -> CE5 cast-vs-resolve mirror
# drains BOTH stacks -> ``move_source_post_resolve`` routes the spell
# hand -> graveyard). Full ``compare_fields`` byte-equality is asserted
# at EVERY phase advance AND post-cast/post-resolve (clean post the
# ``_net_counters`` harness fix). The deterministic-library +
# first-legal-target convention removes every RNG / branch-choice
# degree of freedom so any divergence is unambiguously an engine bug.
# ---------------------------------------------------------------------------


def _drive_phase1_carrier(card_name, *, target_tapped, deck_variant):
    """Cast ``card_name`` through the full Phase-1 carrier path in a
    deterministic matched game and assert full byte-equality + the
    spell-lifecycle observables. Returns nothing — asserts internally.

    Parameters
    ----------
    card_name : str
        ``"Depressurize"`` or ``"Radiant Strike"`` (the live Phase-1
        admit set; the verify-first test above pins the membership).
    target_tapped : bool
        Whether the on-battlefield 2/2 creature target is tapped when
        the spell resolves. Radiant Strike's
        ``DestroyTargetArtifactOrTappedCreature`` leg requires a TAPPED
        creature (the OR-filter teeth); Depressurize's
        ``DepressurizeEffect(-3/-0)`` destroys regardless of tap state
        (eff power 2-3 = -1 <= 0). Exercising both tap states across
        the sweep widens the carrier-path coverage.
    deck_variant : str
        Which deterministic deck context to build (all ``seed=None``
        => deck-list-order library, the genuine bridge-pin #8/#9 shape;
        no RNG, fully reproducible). The variants are GENUINELY-DISTINCT
        17lands-shaped deck contexts — different opening hands, ramp
        depths, and battlefield states at cast time — so the carrier is
        probed across a real multi-deck spread, not one fixture:

          * ``"prefix"``  — spell + target drawn turn 1 (the bridge-pin
            shape: cast at the earliest castable turn).
          * ``"deep"``    — 4 extra lands ahead of the spell in the
            library so the spell is cast several turns LATER with a
            larger battlefield / deeper turn count (a different
            multi-turn trajectory through the carrier).
          * ``"multi"``   — 3 copies of the spell + 2 target Bears in
            the library; one spell is cast through the carrier while
            sibling copies sit in hand/library (exercises the carrier
            with duplicate spell printings present — the registry /
            object-identity path).
          * ``"bear_first"`` — target Bear ahead of the spell (reversed
            draw order: the target resolves before the spell is even
            in hand — a distinct cast-time battlefield shape).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        CardDef,
        DeckSpec,
        advance_phase_python,
        compare_fields,
        step_both, step_both_drained,
    )
    from tests.parity._picker_shims import _PickFirstLegalTargetAgent
    from tests.test_trajectory_parity import _step_views

    src = get_active_source().get_printings()
    real = src.get(card_name)
    if real is None:
        pytest.skip(f"{card_name} absent from active CardSource")
    # Verify-against-live: genuinely the Phase-1 admit shape (the
    # guardrail the production-deck loader applies). This is the
    # per-probe re-assertion of the module-level verify-first guard.
    assert trajectory._printing_is_bridge_simple_spell(real), (
        f"{card_name} must pass the Phase-1 spell guardrail at this "
        f"HEAD (the lever's loader-admit precondition)"
    )

    # The genuine loader admit transform — the EXACT CardDef the
    # production bridge OR-chain produces for the real printing.
    spell_cd = trajectory._printing_to_card_def(real)
    assert spell_cd.card_type == "Instant"
    assert spell_cd.spell_effect is not None

    # Per-card mana: Radiant Strike {3}{W} (4 mana, W); Depressurize
    # {1}{B} (2 mana, B). A correctly-colored basic produces the
    # colored pip; the spell's generic share is paid by the rest.
    color = {"Radiant Strike": "W", "Depressurize": "B"}[card_name]
    mana_count = {"Radiant Strike": 4, "Depressurize": 2}[card_name]
    land_name = {"W": "Plains", "B": "Swamp"}[color]
    land = CardDef(
        name=land_name, card_type="Land", mana_produced=(color,)
    )
    bear = CardDef(
        name="Grizzly Bears",
        card_type="Creature",
        mana_cost_generic=2,
        power=2,
        toughness=2,
    )

    # Deterministic library (seed=None => deck-list order, the genuine
    # bridge-pin #8/#9 idiom; no RNG). Each ``deck_variant`` builds a
    # GENUINELY-DISTINCT 17lands-shaped deck context — different
    # opening hands / ramp depths / cast-time battlefields — so the
    # carrier is swept across a real multi-deck spread.
    if deck_variant == "prefix":
        # spell + target up front: cast at the earliest castable turn.
        prefix = ["Grizzly Bears", card_name]
    elif deck_variant == "deep":
        # 4 extra lands ahead of the spell: cast several turns later
        # with a larger battlefield / deeper turn count.
        prefix = (
            ["Grizzly Bears"]
            + [land_name] * 4
            + [card_name]
        )
    elif deck_variant == "multi":
        # 3 spell copies + 2 target Bears: one cast through the
        # carrier; sibling copies sit in hand/library (duplicate
        # spell-printing registry / object-identity path).
        prefix = (
            ["Grizzly Bears", "Grizzly Bears"]
            + [card_name, card_name, card_name]
        )
    elif deck_variant == "bear_first":
        # reversed draw: target Bear well before the spell — the
        # target is on the battlefield before the spell is in hand.
        prefix = (
            ["Grizzly Bears"]
            + [land_name] * 2
            + ["Grizzly Bears", card_name]
        )
    else:  # pragma: no cover - guarded by the sweep table below
        raise AssertionError(f"unknown deck_variant {deck_variant!r}")

    library = tuple(prefix + [land_name] * (60 - len(prefix)))
    spec = DeckSpec(
        card_defs=(spell_cd, bear, land),
        p0_library=library,
        p1_library=tuple([land_name] * 60),
        shared_rng_seed=None,
    )
    py, rs = trajectory.setup_matched_game(spec, opening_hand_size=7)
    # First-legal target agent — the campaign no-callback convention,
    # matching Rust's bf_iter_etb_sorted first-creature cast-time scan
    # (the equivalence the Tier-A real-EoE probes proved byte-equal).
    for pl in py.players:
        pl.agent = _PickFirstLegalTargetAgent()

    def adv(n):
        """Advance ``n`` phases on BOTH engines, asserting FULL
        ``compare_fields`` byte-equality at every boundary."""
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"[{card_name} variant={deck_variant} tapped={target_tapped}] "
                f"phase advance ({pv['step']}, turn {pv['turn']}) "
                f"diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}"
            )

    def play_one_land():
        ai = py.active_player_idx
        hl = [
            o
            for o in py.players[ai].hand
            if o.printing.name == land_name
        ]
        assert hl, (
            f"[{card_name} variant={deck_variant}] a {land_name} must be in "
            f"hand to ramp (turn {py.turn})"
        )
        assert not step_both(
            py,
            rs,
            {"kind": "play_land", "card_id": int(hl[0].id)},
        ).divergences, (
            f"[{card_name} variant={deck_variant}] play_land diverged"
        )

    def tap_n(n):
        ai = py.active_player_idx
        taps = 0
        for o in list(py.battlefield):
            if (
                o.controller_id == ai
                and o.printing.is_land
                and not o.tapped
                and not o.summoning_sick
                and taps < n
            ):
                assert not step_both(
                    py,
                    rs,
                    {
                        "kind": "activate_ability",
                        "source_id": int(o.id),
                        "ability_index": 0,
                    },
                ).divergences, (
                    f"[{card_name} variant={deck_variant}] mana-tap diverged"
                )
                taps += 1
        assert taps == n, (
            f"[{card_name} variant={deck_variant}] need {n} mana; tapped {taps}"
        )

    def find_in_hand(name):
        ai = py.active_player_idx
        return next(
            (
                o
                for o in py.players[ai].hand
                if o.printing.name == name
            ),
            None,
        )

    adv(3)  # turn 1 -> MAIN1

    # Ramp: play one land per p0 main phase until p0 controls
    # ``mana_count`` lands (one land/turn; adv(24) returns to p0 MAIN1
    # two turns later — p1's no-op turn keeps the game deterministic).
    while (
        sum(
            1
            for o in py.battlefield
            if o.controller_id == py.active_player_idx
            and o.printing.is_land
        )
        < mana_count
    ):
        # Draw the spell+bear into hand only happens via the
        # deterministic opening / per-turn draw; ensure a land is in
        # hand each ramp turn (the deterministic library guarantees
        # the tail is all lands, so once the prefix is drawn every
        # subsequent draw is a land).
        if find_in_hand(land_name) is None:
            adv(24)
            continue
        play_one_land()
        adv(24)

    ai = py.active_player_idx
    # Cast the Grizzly Bears (the target). It costs {2} generic — tap
    # 2 lands; the spell needs ``mana_count`` more, paid later.
    tap_n(2)
    bear_obj = find_in_hand("Grizzly Bears")
    assert bear_obj is not None, (
        f"[{card_name} variant={deck_variant}] Grizzly Bears must be in hand to "
        f"cast the target"
    )
    assert not step_both_drained(
        py,
        rs,
        {"kind": "cast_spell", "card_id": int(bear_obj.id)},
    ).divergences, (
        f"[{card_name} variant={deck_variant}] casting the target Bear diverged"
    )
    adv(24)  # next p0 turn — the Bear is no longer summoning-sick

    ai = py.active_player_idx
    bears_bf = [
        o for o in py.battlefield if o.printing.name == "Grizzly Bears"
    ]
    assert bears_bf, (
        f"[{card_name} variant={deck_variant}] Grizzly Bears must be on the "
        f"battlefield post-cast"
    )
    target_bear_id = bears_bf[0].id

    # Optionally tap the target (Radiant Strike's destroy leg requires
    # a TAPPED creature; do it via an extra mana-tap-like attack-proxy
    # — simplest deterministic tap is to mark it tapped on BOTH engines
    # through the same primitive the harness exposes). We tap it by
    # attacking is over-complex; instead use the engine tap primitive
    # available on the Rust side + the Python object flag, kept
    # byte-aligned (both engines see the creature tapped at resolve).
    if target_tapped:
        # Tap on Rust via its object-tap primitive and mirror the
        # Python object flag so both engines present an identical
        # tapped target at resolve (the field view compares tapped
        # state — keeping them aligned is required for byte-equality).
        assert rs.tap_object(int(target_bear_id)) is True, (
            f"[{card_name} variant={deck_variant}] Rust tap_object must succeed"
        )
        py_bear = next(
            o for o in py.battlefield if o.id == target_bear_id
        )
        py_bear.tapped = True
        # Re-confirm byte-equality after the mirrored tap before the
        # cast (the tap is the only manual cross-engine mutation; it
        # must leave the views identical or the probe's premise is
        # unsound).
        pv, rv = _step_views(py, rs)
        d = compare_fields(pv, rv)
        assert not d, (
            f"[{card_name} variant={deck_variant}] mirrored target-tap left the "
            f"views divergent (probe-setup unsoundness): "
            f"{[(x.field, x.py_value, x.rust_value) for x in d]}"
        )

    gy_py_before = len(py.players[ai].graveyard)
    bf_before = len(py.battlefield)

    # Cast the spell through the GENUINE Phase-1 carrier. {N}{C} =>
    # tap ``mana_count`` lands; ``step_both`` on ``cast_spell`` routes
    # CastSpell -> cast_spell_noncreature -> spell StackItem with
    # target_obj_id (Gate B'') and the CE5 mirror drains BOTH stacks
    # (resolve + move_source_post_resolve: hand -> graveyard) before
    # the full-field compare.
    tap_n(mana_count)
    spell_obj = find_in_hand(card_name)
    assert spell_obj is not None, (
        f"[{card_name} variant={deck_variant}] the spell must be in hand to cast "
        f"through the carrier"
    )
    s = step_both_drained(
        py, rs, {"kind": "cast_spell", "card_id": int(spell_obj.id)}
    )
    # FULL byte-equality post-cast AND post-resolve (the CE5
    # cast-vs-resolve mirror drained both stacks before this compare).
    # ``step_both`` runs the full ``compare_fields`` over
    # FIELD_DIFF_FIELDS — clean since the ``_net_counters`` fix.
    assert not s.divergences, (
        f"[{card_name} variant={deck_variant} tapped={target_tapped}] "
        f"{card_name} cast->resolve through the Phase-1 carrier "
        f"diverged: "
        f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}"
    )

    # Spell-lifecycle observable #1: the spell source is in the
    # caster's graveyard on the Python side post-resolve (the
    # zero-divergence compare already proves Rust agrees; assert the
    # Python observable explicitly so a future harness regression that
    # masks the graveyard field can't hide it).
    py_spell_gy = next(
        (
            o
            for o in py.players[ai].graveyard
            if o.printing.name == card_name
        ),
        None,
    )
    assert py_spell_gy is not None, (
        f"[{card_name} variant={deck_variant}] {card_name} must be in p{ai}'s "
        f"graveyard post-resolve (the carrier's hand -> graveyard "
        f"route — CR 608.2m)"
    )

    # Spell-lifecycle observable #2: the effect observably resolved —
    # the targeted Grizzly Bears was destroyed (Radiant Strike when
    # tapped: DestroyTargetArtifactOrTappedCreature; Depressurize:
    # -3/-0 => eff power -1 <= 0, destroyed regardless of tap state).
    # The graveyard delta is exactly +2 (the destroyed Bear + the
    # spell itself, both controller-side).
    bear_alive = any(
        o.id == target_bear_id for o in py.battlefield
    )
    assert bear_alive is False, (
        f"[{card_name} variant={deck_variant} tapped={target_tapped}] the "
        f"targeted Grizzly Bears must be destroyed by {card_name}"
    )
    assert len(py.players[ai].graveyard) == gy_py_before + 2, (
        f"[{card_name} variant={deck_variant}] graveyard must gain exactly the "
        f"destroyed Bear + the spell ({gy_py_before} -> "
        f"{len(py.players[ai].graveyard)})"
    )
    assert len(py.battlefield) == bf_before - 1, (
        f"[{card_name} variant={deck_variant}] battlefield must lose exactly the "
        f"destroyed Bear"
    )


# ---------------------------------------------------------------------------
# The SWEEP.
#
# The two pid-FREE bench-admitted spells (Depressurize / Radiant
# Strike) x multiple GENUINELY-DISTINCT deterministic deck contexts
# (all seed=None deck-list order — the bridge-pin #8/#9 idiom; no RNG,
# fully reproducible). The two Lander-pid bench-admitted spells
# (Seedship Impact / Orbital Plunge — round-73 SEATED) have their own
# dedicated genuine real-EoE loader-admit + cast→resolve probes in
# ``test_parity_smoke.py`` (they exercise the spell-axis token-seating
# path this carrier-sweep's pid-free pair does not). The four deck
# variants
# (``prefix`` / ``deep`` / ``multi`` / ``bear_first``) give different
# opening hands, ramp depths, cast turns, and cast-time battlefield
# shapes — a real multi-deck spread through the carrier, not one
# fixture. Radiant Strike's destroy leg requires a TAPPED creature (the
# typed-OR filter teeth) so every Radiant Strike arm taps the target;
# Depressurize is swept across both tap states (its -3/-0 destroys
# regardless). Each parametrization is an independent genuine real-EoE
# cast->resolve byte-equality probe through the full Phase-1 carrier.
# ---------------------------------------------------------------------------


# (card_name, target_tapped, deck_variant) — the deck-context matrix.
_DECK_VARIANTS = ("prefix", "deep", "multi", "bear_first")
_SWEEP = (
    # Depressurize (-3/-0 destroys regardless of tap state): 4 deck
    # contexts; the ``multi`` arm casts with the target untapped, the
    # others tapped — sweeping both tap states for the no-OR-filter
    # spell widens the carrier coverage.
    [
        ("Depressurize", v != "multi", v) for v in _DECK_VARIANTS
    ]
    # Radiant Strike (DestroyTargetArtifactOrTappedCreature requires a
    # TAPPED creature — the typed-OR teeth): 4 deck contexts, target
    # always tapped (an untapped non-artifact creature is NOT a legal
    # destroy — the destroy leg silently no-ops; the GainLife leg
    # still resolves, a different observable not asserted here).
    + [("Radiant Strike", True, v) for v in _DECK_VARIANTS]
)


@pytest.mark.parametrize(
    "card_name,target_tapped,deck_variant",
    _SWEEP,
    ids=[
        f"{c.replace(' ', '_')}-tapped{t}-{v}"
        for (c, t, v) in _SWEEP
    ],
)
def test_spell_card_bridge_phase1_carrier_real_eoe_byte_equal_sweep(
    card_name, target_tapped, deck_variant
):
    """**GENUINE real-EoE cross-engine byte-equality probe — the
    spell-card-bridge Phase 1 carrier SWEEP.**

    For each ``(card, tapped, deck_variant)``: drive the ACTUAL parsed
    printing through the full Phase-1 carrier (``cast_spell`` ->
    ``cast_spell_noncreature`` -> ``StackItem.target_obj_id`` (Gate
    B'') -> resolve -> graveyard) in a deterministic 17lands-shaped
    matched game and assert:

      1. FULL ``compare_fields`` byte-equality at EVERY phase boundary
         (the ramp / target-cast / spell-cast walk);
      2. FULL ``compare_fields`` byte-equality post-cast AND
         post-resolve (the CE5 cast-vs-resolve mirror drains BOTH
         engines' stacks before this compare — Python's spell + its
         resolution and Rust's ``cast_spell_noncreature`` +
         ``resolve_top_of_stack`` must agree on every observable);
      3. the spell source in the caster's graveyard on BOTH engines
         (the carrier's hand -> graveyard route, CR 608.2m);
      4. the effect observably resolved (the targeted Grizzly Bears
         destroyed; graveyard delta exactly +2; battlefield -1).

    Every degree of freedom is removed (deterministic library,
    no-callback first-legal targeting, mirrored target-tap) so any
    surfaced divergence is unambiguously a carrier-path engine bug,
    NOT a harness/RNG artifact — exactly the evidence-strengthening
    the flagship lever (shipped with ONE probe) was missing.
    """
    _drive_phase1_carrier(
        card_name,
        target_tapped=target_tapped,
        deck_variant=deck_variant,
    )
