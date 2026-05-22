"""Canonical, production-importable Rust-Effect-coverage primitives.

This module is the **single definition** of:

  * the Python ``Effect``-vocab walker (``_walk_effect`` / ``_maybe_walk``
    / ``collect_python_effect_vocab``) — what distinct Python ``Effect``
    class names the EoE pool emits;
  * the Rust ``Effect`` enum text-parser
    (``collect_rust_effect_variants``) + the Rust→Python name map
    (``map_rust_to_python`` / ``RUST_TO_PYTHON_OVERRIDES`` /
    ``RUST_AHEAD_OF_POOL``);
  * the derived **covered set** (``covered_python_class_set``) and the
    per-card emitted-class query (``card_effect_classes``).

WHY THIS LIVES UNDER ``src/mtg_bot/`` (the round-66 covered-set
extraction precursor)
---------------------------------------------------------------------
``covered_python_class_set`` / ``card_effect_classes`` used to live in
``scripts/measure_pool_parity.py``, and the parser primitives in
``scripts/measure_effect_coverage.py``. The spell-card-bridge Phase 1
guardrail (`docs/SPELL_CARD_BRIDGE_PHASE1_DESIGN.md` §3.6) needs the
live covered-set query from a *production-package* predicate
(`src/mtg_bot/parity/trajectory.py::_spell_effect_is_covered_and_
bridged`). A production module importing a ``scripts/`` module is a
layering inversion. So the **canonical definitions move HERE** and the
two scripts now re-import from this module — keeping exactly ONE
definition of "covered" (the §2-teeth requirement: the guardrail must
query the live covered set, never a hardcoded list, and the pool-parity
metric + the coverage doc must never disagree on what "covered" means).

This is a **pure relocation / refactor** — the parse rule, the walker,
the override tables and the covered-set computation are byte-for-byte
the previous ``scripts/`` implementations; behavior is unchanged. The
companion test ``tests/test_effect_coverage_module.py`` pins
identical output vs the prior scripts implementation on the live pool.

This module does NOT depend on the Rust extension being built — the
Rust side is a plain-text parse of ``rust/src/cards.rs`` (no
``mtg_bot_engine`` import), exactly as before.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path

# src/mtg_bot/parity/effect_coverage.py -> parity -> mtg_bot -> src ->
# repo root. (The previous home, scripts/measure_effect_coverage.py,
# resolved REPO_ROOT as parents[1]; from here it is parents[3]. The
# resolved path is identical.)
REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_CARDS_FILE = REPO_ROOT / "rust" / "src" / "cards.rs"

# Rust variant name -> Python class name override. Empty mapping means
# "use the variant name verbatim" (the 1:1 case). Populate this if a
# Rust variant intentionally maps to a Python class with a different
# spelling.
RUST_TO_PYTHON_OVERRIDES: dict[str, str] = {
    # The Rust bridge's generic ``DealDamage`` is the family-head
    # primitive; Python doesn't instantiate the bare ``DealDamage``
    # base class on any card — only its subclasses appear. The
    # highest-frequency in-pool subclass is
    # ``DealDamageToTargetCreature``.
    "DealDamage": "DealDamageToTargetCreature",
    # The Rust bridge's generic primitive corresponds to the
    # creature/permanent-bounce family on the Python side. Python
    # does not have a top-level ``ReturnToHand`` class; the closest
    # high-frequency counterpart is the bounce-to-owners-hand path.
    "ReturnToHand": "ReturnTargetCreatureToOwnersHand",
    # Rust's ``Mill`` variant is the controller-mills-self primitive;
    # Python's class for that is ``SelfMill``. (Mill-opponent has its
    # own ``MillTargetPlayer`` class on the Python side, which maps
    # 1:1 below.)
    "Mill": "SelfMill",
    # Rust's ``ExileTarget`` is the generic exile-target-creature
    # primitive; the closest high-frequency Python class is
    # ``ExileTargetCreature``.
    "ExileTarget": "ExileTargetCreature",
    # Rust's ``TapTarget`` / ``UntapTarget`` / ``Sacrifice`` /
    # ``DiscardCard`` / ``AddCounters`` / ``DestroyTarget`` have no
    # direct 1:1 Python counterpart with the same name — the Python
    # side uses more specific class names. These map to the closest
    # observed Python class by intent.
    "TapTarget": "TapTargetArtifactOrCreature",
    "UntapTarget": "UntapTargetOtherPermanent",
    "Sacrifice": "SacrificeSelf",
    "DiscardCard": "TargetOpponentDiscardsCasterChoice",
    "AddCounters": "PutPlusOneCounterOnTargetCreature",
    "DestroyTarget": "DestroyTargetCreature",
    # The Rust bridge's ``DamageEachOpponent`` is the generic
    # damage-to-each-opponent primitive; the in-pool Python class is
    # ``DealDamageEachOpponent`` (4 cards). The unrelated
    # ``EachOpponentLosesLife`` is a separate Rust variant with its
    # own 1:1 mapping below.
    "DamageEachOpponent": "DealDamageEachOpponent",
    "DamageEachCreature": "DealNDamageToEachCreature",
    "DealDamageAnyTarget": "DealDamageAnyTarget",
    # Rust ``ChooseOneMode`` is the modal-choice combinator family head;
    # the Python equivalent is ``ModalChooseOne`` (in
    # ``effects/combinators.py``). The naming diverged when the
    # Rust-side parity bridge ported the discriminator first.
    "ChooseOneMode": "ModalChooseOne",
    # Rust ``GrantKeywordToTargetCreatureUntilEOT`` is the
    # parametric grant-arbitrary-keyword-UEOT primitive. Python doesn't
    # have a 1:1 generic class; the closest in-pool consumer is
    # ``GrantUnblockableUEOT`` (subclass of the keyword-specific
    # ``GrantUnblockable``). 1:N — multiple Python classes could match
    # depending on the keyword being granted; this picks the
    # highest-frequency one in the EoE pool.
    "GrantKeywordToTargetCreatureUntilEOT": "GrantUnblockableUEOT",
    # Rust ``ReturnArtifactOrCreatureFromYourGraveyardToHand`` and
    # ``ReturnCreatureOrSpacecraftFromYourGraveyardToHand`` are
    # additive bespoke variants extending the bare
    # ``ReturnCreatureFromYourGraveyardToHand`` family with rider
    # widenings (or_artifact, or_spacecraft). Python keeps a single
    # unified class ``ReturnCreatureFromYourGraveyardToHand`` and
    # toggles the rider via constructor flags — both Rust variants
    # share the Python class.
    "ReturnArtifactOrCreatureFromYourGraveyardToHand":
        "ReturnCreatureFromYourGraveyardToHand",
    "ReturnCreatureOrSpacecraftFromYourGraveyardToHand":
        "ReturnCreatureFromYourGraveyardToHand",
    # --- Round-52 refresh: rider-variant overrides for the
    # token / surveil / mill matrix-saturation siblings.
    #
    # These four Rust variants are NOT new Python classes — they are
    # rider-flag selections on an existing, high-frequency, in-pool
    # Python class (same precedent as the ``ReturnArtifactOrCreature
    # ...`` family directly above: one unified Python class, the
    # binary choice rides a constructor flag, the closed-vocab Rust
    # side spells each flag as its own variant). The bridge arms in
    # ``cross_engine._ta_dict_to_python_effect`` are the authority for
    # each mapping — each constructs the Python class shown here with
    # the noted flag set:
    #
    #   * ``Effect::CreateTokenEachPlayer`` →
    #     ``CreateToken(each_player=True)`` (cross_engine.py:1945;
    #     ``tokens.py`` ``CreateToken``, in-pool freq 41).
    #   * ``Effect::CreateTokenTapped`` →
    #     ``CreateToken(tapped=True)`` (cross_engine.py:7636).
    #   * ``Effect::SurveilXEqualToArtifactCount`` →
    #     ``Surveil(use_artifact_count=True)`` (cross_engine.py:6882;
    #     ``library_lookat.py`` ``Surveil``, in-pool freq 16).
    #   * ``Effect::MillTargetPlayerEqualToYourLifeTotal`` →
    #     ``MillTargetPlayer(use_controller_life=True)``
    #     (cross_engine.py:2077; ``draw_mill.py`` ``MillTargetPlayer``,
    #     in-pool freq 2).
    #
    # Each target Python class was verified to exist + be pool-walked
    # at round-52 refresh time. Mapping to the real in-pool class (NOT
    # suppressing the orphan check) is what clears the false WARNING
    # for these four: the bridge genuinely round-trips them through
    # the named class, so the override is the correct relationship,
    # not a silencer. (The 8 ``DealDamage*``/``EachOpponent*``/
    # ``GainLife*`` matrix siblings DO have same-named Python classes
    # but 0 in-pool consumers — those are handled by the
    # ``RUST_AHEAD_OF_POOL`` allowlist below, where an override would
    # be *wrong* because the same-named class already exists.)
    "CreateTokenEachPlayer": "CreateToken",
    "CreateTokenTapped": "CreateToken",
    "SurveilXEqualToArtifactCount": "Surveil",
    "MillTargetPlayerEqualToYourLifeTotal": "MillTargetPlayer",
    # 19th Rust slice (honest-stop substrate landing) -- the Rust
    # variant intentionally omits the ``Effect`` suffix to match the
    # cards.rs deferred-slot doc-comment naming convention
    # (``EnchantedCreatureFightsTarget`` is the slot the round-17 bare
    # bilateral Fight doc-comment named as the Aura-shape follow-up).
    # The Python class keeps the ``Effect`` suffix per the existing
    # ``effects/bespoke.py:506`` ``EnchantedCreatureFightsTargetEffect``
    # spelling. Bridge round-trip works end-to-end (verified via
    # test_probe_yyy_downstream.py + test_divergence_pins_yyy.py); this
    # override clears the false orphan WARNING by reflecting the real
    # round-trip relationship rather than silencing it (same precedent
    # as the slice-16 / round-52 family).
    "EnchantedCreatureFightsTarget": "EnchantedCreatureFightsTargetEffect",
    # --- Round-56 N2 fold: Famished Worldsire override.
    #
    # Round-54 N2 shipped Effect::FamishedWorldsireLookTopXByPowerPut
    # LandsTapped (cards.rs:5320/7437) + a wired bridge arm
    # (cross_engine.py:7870 — the bridge pin passes the cron's
    # bridged-arm gate, so this is NOT real bridge drift). But the
    # 1:1 identity rule mapped the variant to a same-named Python
    # class that does NOT exist: the REAL Famished Worldsire printing
    # parses to the bespoke ``FamishedWorldsireETBEffect``
    # (``effects/bespoke.py`` — verified to exist + be pool-walked at
    # round-56 refresh time, in-pool freq 1, the Famished Worldsire
    # card itself). So the coverage doc emitted a false
    # "1 Rust variant with no Python counterpart" WARNING. Mapping to
    # the real in-pool class is the CORRECT relationship (the bridge
    # genuinely round-trips this Effect through ``FamishedWorldsireETB
    # Effect``), NOT a suppression — exactly the round-52 N3
    # matrix-variant override-fold class (same-named class absent →
    # override to the real class; do NOT allowlist via
    # RUST_AHEAD_OF_POOL, which is for variants whose same-named class
    # exists but has 0 in-pool consumers — the opposite case).
    "FamishedWorldsireLookTopXByPowerPutLandsTapped":
        "FamishedWorldsireETBEffect",
    # --- Faller's Faithful + Perigee Beckoner override-fold (the same
    # round-56 Famished-Worldsire-precedent class: a 1:1-identity-rule
    # variant whose SAME-named Python class does NOT exist, but whose
    # bridge arm genuinely round-trips through a real, differently-
    # spelled, pool-walked Python class — so the override is the
    # CORRECT relationship, NOT a suppression, and NOT a
    # ``RUST_AHEAD_OF_POOL`` allowlist, which is reserved for variants
    # whose same-named class EXISTS but has 0 in-pool consumers — the
    # opposite case).
    #
    #   * ``Effect::DestroyTargetCreatureThenDrawIfUndamaged
    #     { draw_amount }`` (round-era commit ``1ee30e1e``, Faller's
    #     Faithful ETB half, ~14,108 17lands decks). The bridge arm
    #     ``cross_engine.py`` ``destroy_target_creature_then_draw_if_
    #     undamaged`` constructs Python ``DestroyTargetCreature(
    #     draw_if_undamaged_amount=draw_amount, optional=True,
    #     exclude_self=True)`` (``effects/generic/destroy.py:94``). The
    #     GENUINE parsed Faller's Faithful printing parses to that EXACT
    #     class (verified: ``DestroyTargetCreature`` in-pool freq 9, and
    #     the ``Faller's Faithful`` printing itself is in its
    #     pool-walked example set). The "BLOCKED-RUST-EFFECT /
    #     card-stays-gated" note in the bridge arm refers to the
    #     FULL-CARD loader gate (Faller's Faithful is not on the
    #     production-deck ``_SAFE_*`` allowlist — the residue-worklist
    #     ``closed`` entry), NOT the Effect-level orphan question: the
    #     Effect half closes byte-equal and round-trips through the real
    #     pool-walked class, which is exactly the override case.
    #   * ``Effect::PumpTargetCreatureUntilEotGrantDiesReturnTapped
    #     { power_bonus, toughness_bonus }`` (round-60 commit
    #     ``18d6b6cb``, Perigee Beckoner ETB, ~21,131 17lands decks).
    #     The bridge arm ``cross_engine.py`` ``pump_target_creature_
    #     until_eot_grant_dies_return_tapped`` constructs Python
    #     ``PumpTargetCreatureUntilEOT(power_bonus=..., toughness_bonus
    #     =..., you_control_only=True, exclude_self=True,
    #     granted_triggered_ability=TriggeredAbility(...,
    #     effect=PerigeeBeckonerDiesReturnTappedEffect(), ...))``
    #     (``effects/generic/pump.py:258``). The GENUINE parsed Perigee
    #     Beckoner printing parses to that EXACT class+params (verified:
    #     ``PumpTargetCreatureUntilEOT`` in-pool freq 26, and the
    #     ``Perigee Beckoner`` printing itself is in its pool-walked
    #     example set). Same FULL-CARD loader gate (Perigee Beckoner is
    #     not on ``_SAFE_*`` — the residue-worklist ``closed`` entry,
    #     also independently ``warp_cost``-gated) vs Effect-half-closes
    #     split as Faller's; the Effect round-trips through the real
    #     pool-walked class → override, not allowlist.
    "DestroyTargetCreatureThenDrawIfUndamaged": "DestroyTargetCreature",
    "PumpTargetCreatureUntilEotGrantDiesReturnTapped":
        "PumpTargetCreatureUntilEOT",
    # 22nd Rust slice (Sothera + Requiem honest-stop) -- the Rust
    # variant intentionally carries the ``Requiem`` suffix to mark it
    # as the Requiem-Monolith-shape grant arm. Rust has no general
    # layer-6 grant-triggered-ability registry yet (only the narrow
    # Perigee ``GrantedDiesReturnTapped`` marker exists; see
    # ``state.rs:299``). The Python class is the general
    # ``GrantTriggeredAbilityUEOT`` (``effects/generic/grant_keywords
    # .py:189-251``); the bridge arm reconstructs it from the fixed
    # Requiem shape (trigger=this_dealt_damage,
    # effect=DrawCardsAndLoseLifeEqualToResolvingDamage()). The Real
    # Requiem Monolith printing parses to a Sequence whose first child
    # IS ``GrantTriggeredAbilityUEOT(ability=...)`` -- verified
    # round-trip via test_probe_sothera_requiem_downstream.py +
    # test_requiem_monolith.py. Same precedent as the Faller's Faithful
    # / Perigee Beckoner overrides above (same-named class does not
    # exist; bridge round-trips through a real pool-walked class).
    "GrantTriggeredAbilityUEOTRequiem": "GrantTriggeredAbilityUEOT",
    # --- Invasive Maneuvers slice (Effect-uncovered Tier-A; ~10,981
    # 17lands decks). Rust ``Effect::InvasiveManeuversEffect { base,
    # boosted }`` (cards.rs two-int variant) → Python class
    # ``DealDamageToTargetCreatureSpacecraftBoosted``
    # (``effects/generic/damage.py:850``). The variant name is the
    # card-name-tagged variant (mirrors the Cut Propulsion /
    # Diplomatic Relations / Biosynthic Burst card-name-tagged
    # convention); the Python side uses the descriptive
    # ``DealDamageToTargetCreatureSpacecraftBoosted`` class spelling.
    # The bridge arm in ``cross_engine.py``
    # ``invasive_maneuvers_effect`` constructs the Python class with
    # ``base=base, boosted=boosted``. Same Famished-Worldsire-precedent
    # override class (same-named class absent → map to the real
    # pool-walked class; NOT a RUST_AHEAD_OF_POOL allowlist).
    "InvasiveManeuversEffect": "DealDamageToTargetCreatureSpacecraftBoosted",
    # --- Emergency Eject slice (Effect-uncovered single-gated tier;
    # ~6,291 17lands decks). Rust ``Effect::EmergencyEjectEffect {
    # token_printing_id }`` (cards.rs single-pid variant; same
    # Lander-pid carrier shape as ``SeedshipImpactEffect`` and
    # ``DealDamageToTargetCreatureWithExcessLanderRider``) → Python
    # class ``DestroyTargetNonlandPermanentWithLanderForController``
    # (``effects/generic/destroy.py:589``). The Rust variant name is
    # the card-name-tagged convention (mirrors Cut Propulsion /
    # Diplomatic Relations / Invasive Maneuvers / Biosynthic Burst);
    # the Python class is the descriptive
    # ``DestroyTargetNonlandPermanentWithLanderForController`` spelling.
    # The bridge arm in ``cross_engine.py`` ``emergency_eject_effect``
    # constructs the field-free Python class (the pid is dropped at
    # the bridge — Python mints the Lander intrinsically via
    # ``TOKEN_PRINTINGS["Lander"]``, same as Seedship Impact / Orbital
    # Plunge). Same Famished-Worldsire-precedent override class.
    "EmergencyEjectEffect":
        "DestroyTargetNonlandPermanentWithLanderForController",
}

# Rust variants that have NO direct Python counterpart in the current
# pool because the underlying mechanic doesn't appear on any pool card.
# These are NOT considered orphans for the regression check — they're
# variants the bridge built ahead of demand (e.g. scry, which the EoE
# pool happens not to use). Adding a Rust variant here is a deliberate
# decision; the orphan test stays strict about everything else.
RUST_AHEAD_OF_POOL: set[str] = {
    # No ``Scry`` class in the EoE-set vocab today (Surveil is the
    # in-pool sibling). When a pool with scry-bearing cards is added,
    # remove this entry and add an override or 1:1 ``ScryN`` class.
    "ScryN",
    # Rust ``SetBasePTUntilEOT`` is a base-PT-overwrite primitive (sets
    # the target's base P/T to a fixed value until end of turn) that
    # has no in-pool Python consumer yet. The Rust bridge needs it for
    # parity tests against a future card; add the matching Python class
    # when an EoE card surfaces a consumer.
    "SetBasePTUntilEOT",
    # Rust ``FightTargetCreatureWithTargetCreature`` — bare bilateral
    # fight primitive (Rust feature/rust-fight-effect, commit ab1bd52d).
    # Python class exists at ``effects/generic/damage.py:631`` but no
    # EoE card consumes the bare shape yet; Diplomatic Relations
    # consumes the two-target bespoke variant. Add an override or
    # remove this entry when the bare-shape consumer lands.
    "FightTargetCreatureWithTargetCreature",
    # Rust ``ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield``
    # — additive enchantment-or-creature widening of the bare
    # ``ReturnCreatureFromYourGraveyardToBattlefield`` family (parallels
    # the existing OrArtifact / OrSpacecraft overrides for the ToHand
    # sibling). No in-pool EoE consumer for the widened-to-enchantment
    # variant yet; promote to an override mapping to
    # ``ReturnCreatureFromYourGraveyardToBattlefield`` when one lands.
    "ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield",
    # Rust ``ReturnEnchantmentOrCreatureFromYourGraveyardToHand`` — the
    # ToHand sibling of the variant above; same OrEnchantment widening
    # pattern. Promote to an override mapping to
    # ``ReturnCreatureFromYourGraveyardToHand`` when an EoE card
    # surfaces the widened-to-enchantment shape.
    "ReturnEnchantmentOrCreatureFromYourGraveyardToHand",
    # Rust ``PumpCreaturesOpponentsControlUntilCasterNextTurn`` —
    # extended-duration sibling of ``PumpCreaturesOpponentsControlUntilEOT``
    # (Rust commit 8e5d7852). Python has the UntilEOT variant but no
    # UntilCasterNextTurn-duration analog; allowlist until a card pool
    # surfaces an extended-duration consumer.
    "PumpCreaturesOpponentsControlUntilCasterNextTurn",
    # Rust ``ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves`` —
    # parametric MaxMv-filter rider on the bare
    # ``ExileOpponentsNonlandPermanentUntilLeaves`` family. Python has
    # the bare class at ``effects/generic/exile.py:588`` but no card
    # consumes the MaxMv-filtered variant in the current pool.
    "ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves",
    # Rust ``GrantKeywordsToTargetCreatureUntilEOT`` — plural-keywords
    # sibling of the existing singular ``GrantKeywordToTargetCreatureUntilEOT``
    # override (which maps to ``GrantUnblockableUEOT``). The plural
    # form takes a list of keywords and has no 1:N mapping that's
    # obviously correct; allowlist until a clean Python target emerges.
    "GrantKeywordsToTargetCreatureUntilEOT",
    # --- Round-52 refresh: X-from-count matrix-saturation siblings
    # with a same-named Python class but ZERO in-pool consumers.
    #
    # Rounds 44–49 shipped the X-from-count primitive matrix
    # (6 apply-substrates × 6 count-denominators) per the
    # round-27/31 round-trip-surface-coverage rule. Per the round-49
    # audit refresh (``docs/EOE_PROBE_READINESS_AUDIT.md``), every
    # one of these denominator-sweep variants landed with **0 EoE
    # consumers** — they exist for bidirectional round-trip surface,
    # not because a pool card uses them. Each has a real Python
    # class with the EXACT same name (verified to exist at round-52
    # refresh time — see ``effects/generic/damage.py`` /
    # ``effects/generic/life.py``), but the parser never instantiates
    # it because no pool card carries the shape, so the class is
    # absent from the parsed-pool vocab.
    #
    # This is precisely the ``RUST_AHEAD_OF_POOL`` contract ("variants
    # the bridge built before any pool card uses them") — NOT an
    # override case: the same-named class already exists, so mapping
    # it elsewhere via ``RUST_TO_PYTHON_OVERRIDES`` would be *wrong*.
    # Allowlisting here is the correct, non-suppression fix that
    # clears the false orphan WARNING; remove an entry + add a 1:1
    # class when an EoE printing surfaces a consumer for that cell.
    # (Sibling token/surveil/mill rider-variants — which DO map onto
    # an in-pool class via a constructor flag — are handled by
    # ``RUST_TO_PYTHON_OVERRIDES`` above instead.)
    "DealDamageToOpponentCreatureByCardsInGraveyard",   # r46, damage row
    "DealDamageToOpponentCreatureByCreaturesYouControl",  # r37, damage row
    "DealDamageToOpponentCreatureByLandsControlled",    # r36, damage row
    "DealDamageToOpponentCreatureByLifeTotal",          # r47, damage row 6/6
    "EachOpponentLosesLifeEqualToArtifactCount",        # r44
    "EachOpponentLosesLifeEqualToYourLifeTotal",        # r48
    "GainLifeEqualToArtifactCount",                     # r45
    "GainLifeEqualToYourLifeTotal",                     # r49
    # Round-51 N2 Monoist Circuit-Feeder deterministic bridge sibling.
    # Rust ``Effect::TwoSubjectPumpAndDebuffByArtifactCount`` →
    # Python ``PumpYoursAndDebuffOpponentByArtifactCountUntilEOT``
    # (cross_engine.py:7719). The real Monoist Circuit-Feeder printing
    # parses to the AGENT-DRIVEN ``PairedTargetPumpDebuffUEOT`` class,
    # not this deterministic-pick class — so the deterministic class
    # exists (verified ``effects/generic/pump.py``) but is never
    # pool-walked. Same FightTargetCreatureWithTargetCreature
    # (deterministic, ahead-of-pool) vs agent-driven-real-card
    # precedent already allowlisted above; ahead-of-pool, not an
    # override, for the same reason.
    "TwoSubjectPumpAndDebuffByArtifactCount",
    # **G7 Class-C delayed-trigger port (round-78).** Rust
    # ``Effect::WarpExileSelf`` is the warp end-step delayed trigger
    # the cast site (``cast_creature_warp``) schedules onto
    # ``state.delayed_triggers``. Python has the exact same-named
    # class at ``effects/generic/exile.py:133`` (``WarpExileSelf``),
    # but the parsed-pool vocab never observes it directly because
    # it is NOT a printing's `triggered_abilities` member — it is
    # spawned by the cast site as a delayed trigger entry, NOT by a
    # printing's static trigger definition. The
    # ``OracleParserCardSource`` walks per-printing triggered
    # abilities; the delayed-trigger spawn site is dynamic and
    # therefore invisible to the pool walker. This is precisely the
    # ``RUST_AHEAD_OF_POOL`` contract: a same-named Python class
    # exists (verified — used by `actions.py:1554`'s
    # `delayed_triggers.append`) but the static-printing walker
    # doesn't see it. Override-mapping it elsewhere would be wrong.
    "WarpExileSelf",
    # **Control-change-UEOT subsystem — Cycle 5 (2026-05-22).** Rust
    # ``Effect::RemoveChargeCountersFromSourceDelayed`` is the
    # Systems Override end-step delayed trigger the resolve site
    # (``Effect::SystemsOverrideEffect`` arm in state_ops_resolve.rs)
    # schedules onto ``state.delayed_triggers`` when the gained-
    # control target is a Spacecraft. Python has the exact same-named
    # class at ``effects/generic/delayed_actions.py:120``
    # (``RemoveChargeCountersFromSourceDelayed``), but the parsed-pool
    # vocab never observes it directly because it is NOT a printing's
    # ``triggered_abilities`` member — it is spawned by the Systems
    # Override resolve as a delayed-trigger entry, NOT by a printing's
    # static trigger definition. SystemsOverrideEffect's resolve body
    # creates the instance dynamically inside an ``if obj.printing.is
    # _spacecraft:`` branch; the dataclass walk in
    # :func:`_walk_effect` only walks declared fields, not method-
    # local constructions. Same RUST_AHEAD_OF_POOL discipline as
    # ``WarpExileSelf`` above — override-mapping it elsewhere would
    # be wrong.
    "RemoveChargeCountersFromSourceDelayed",
}


# ---------------------------------------------------------------- Python side


def _walk_effect(eff, *, classes, examples, parent_card_name, seen_ids):
    if eff is None:
        return
    obj_id = id(eff)
    if obj_id in seen_ids:
        return
    seen_ids.add(obj_id)

    cls_name = type(eff).__name__
    classes[cls_name] += 1
    examples[cls_name].add(parent_card_name)

    if is_dataclass(eff):
        for f in fields(eff):
            try:
                v = getattr(eff, f.name)
            except Exception:
                continue
            _maybe_walk(v, classes, examples, parent_card_name, seen_ids)
    else:
        for v in getattr(eff, "__dict__", {}).values():
            _maybe_walk(v, classes, examples, parent_card_name, seen_ids)


def _maybe_walk(v, classes, examples, parent_card_name, seen_ids):
    if v is None or isinstance(v, (str, bytes, int, float, bool, frozenset)):
        return
    if hasattr(v, "resolve") and hasattr(v, "describe"):
        _walk_effect(
            v,
            classes=classes,
            examples=examples,
            parent_card_name=parent_card_name,
            seen_ids=seen_ids,
        )
        return
    if isinstance(v, (list, tuple, set)):
        for item in v:
            _maybe_walk(item, classes, examples, parent_card_name, seen_ids)
        return
    if is_dataclass(v):
        for f in fields(v):
            try:
                inner = getattr(v, f.name)
            except Exception:
                continue
            _maybe_walk(inner, classes, examples, parent_card_name, seen_ids)


def collect_python_effect_vocab():
    """Return ``(classes: Counter, examples: dict, n_printings: int)``."""
    from mtg_bot.cards.parser.source import OracleParserCardSource

    src = OracleParserCardSource()
    printings = src.get_printings()

    classes: Counter[str] = Counter()
    examples: dict[str, set[str]] = defaultdict(set)
    n_printings = 0

    for name, printing in printings.items():
        n_printings += 1
        seen_ids_card: set[int] = set()
        try:
            _maybe_walk(
                printing.spell_effect, classes, examples, name, seen_ids_card
            )
            for ta in printing.triggered_abilities:
                _maybe_walk(ta, classes, examples, name, seen_ids_card)
            for aa in getattr(printing, "activated_abilities", ()) or ():
                _maybe_walk(aa, classes, examples, name, seen_ids_card)
            for sa in getattr(printing, "static_abilities", ()) or ():
                _maybe_walk(sa, classes, examples, name, seen_ids_card)
        except Exception:
            continue

    return classes, examples, n_printings


# ---------------------------------------------------------------- Rust side


_RUST_VARIANT_RE = re.compile(
    # Match a line that starts a variant: optional whitespace,
    # CamelCase identifier, optional `{ … }` or `(…)` or unit, then
    # comma. We anchor on the property that variants appear as the
    # first non-whitespace identifier on a line inside the enum block,
    # NOT preceded by `///` (doc-comment) or `//` (line comment) or
    # `impl ` / `pub ` etc.
    r"^\s+([A-Z][A-Za-z0-9]+)\s*(\{|,|\()",
    re.MULTILINE,
)


def collect_rust_effect_variants(cards_rs: Path = RUST_CARDS_FILE) -> list[str]:
    """Parse ``rust/src/cards.rs`` and return the list of Effect
    enum variant names in source order. Pure text parse — no Rust
    toolchain needed."""
    text = cards_rs.read_text(encoding="utf-8")

    # Find the start of the `pub enum Effect {` block.
    m = re.search(r"pub enum Effect\s*\{", text)
    if not m:
        raise RuntimeError(f"could not find `pub enum Effect {{` in {cards_rs}")
    start = m.end()

    # Walk forward through balanced braces to find the closing `}` of
    # the enum block.
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        raise RuntimeError(f"unbalanced braces in Effect enum in {cards_rs}")
    block = text[start:i]

    # Strip line and block comments so the variant regex doesn't pick
    # up CamelCase tokens inside docstrings.
    # Remove `///` doc comments and `//` line comments.
    block_no_comments = re.sub(r"//[^\n]*", "", block)
    # Remove `/* ... */` block comments (defensive — none in cards.rs
    # today, but cheap to strip).
    block_no_comments = re.sub(r"/\*.*?\*/", "", block_no_comments, flags=re.S)

    variants: list[str] = []
    seen: set[str] = set()
    for mv in _RUST_VARIANT_RE.finditer(block_no_comments):
        name = mv.group(1)
        # Filter out anything that doesn't look like a variant — skip
        # known false positives like Rust keywords / type names that
        # might appear at line start (none observed today but defensive).
        if name in {"Self", "Some", "None", "Ok", "Err"}:
            continue
        if name not in seen:
            seen.add(name)
            variants.append(name)
    return variants


# ---------------------------------------------------------------- mapping


def map_rust_to_python(variant: str) -> str:
    """Return the Python class name that corresponds to a Rust variant.

    Default: identity (1:1 CamelCase). Override via
    ``RUST_TO_PYTHON_OVERRIDES`` for the documented mismatches.
    """
    return RUST_TO_PYTHON_OVERRIDES.get(variant, variant)


# -------------------------------------------------- the covered set


def covered_python_class_set() -> set[str]:
    """The set of Python ``Effect`` class names that have a live Rust
    ``Effect::`` variant + (by the coverage script's mapping) a
    counterpart — i.e. ``py_class_set ∩ rust_py_name_set``. This is the
    EXACT "covered" relation ``measure_effect_coverage`` reports as its
    class-count headline; reused so the two metrics can never disagree
    on what "covered" means."""
    py_classes, _examples, _n = collect_python_effect_vocab()
    rust_variants = collect_rust_effect_variants()
    rust_py_name_set = {map_rust_to_python(v) for v in rust_variants}
    return set(py_classes.keys()) & rust_py_name_set


# ------------------------------------------------- per-card Effect classes


def card_effect_classes(printing) -> set[str]:
    """The distinct Python ``Effect`` class names a single printing
    emits, using ``measure_effect_coverage``'s exact recursive walker
    (``_maybe_walk`` over ``spell_effect`` + ``triggered_abilities`` +
    ``activated_abilities`` + ``static_abilities``). Returns an empty
    set for a pure-vanilla / keyword-only card."""
    classes: Counter[str] = Counter()
    examples: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[int] = set()
    name = getattr(printing, "name", "?")
    try:
        _maybe_walk(
            printing.spell_effect, classes, examples, name, seen_ids
        )
        for ta in printing.triggered_abilities:
            _maybe_walk(ta, classes, examples, name, seen_ids)
        for aa in getattr(printing, "activated_abilities", ()) or ():
            _maybe_walk(aa, classes, examples, name, seen_ids)
        for sa in getattr(printing, "static_abilities", ()) or ():
            _maybe_walk(sa, classes, examples, name, seen_ids)
    except Exception:
        # Mirror the coverage walker's defensive contract: a card the
        # walker can't introspect contributes no classes (it then
        # falls to a BLOCKED tier — never spuriously "covered").
        return set()
    return set(classes.keys())
