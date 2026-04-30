"""Triggered-ability grammar.

A triggered-ability line of the form ``When/Whenever <condition>,
<effect>.`` decomposes into a condition phrase and an effect phrase.
Each is handled by its own parser — the condition by
``_parse_trigger_phrase`` (a literal-phrase lookup against
``_TRIGGER_PHRASES`` and ``_PREDICATE_TRIGGER_PHRASES``), the effect
by ``_parse_effect_phrase`` from ``effects``. Adding a new condition
or effect grows its respective grammar without touching the line-
level pattern.

Two trigger shapes coexist during the predicate-trigger migration:

- **String-key style** (legacy): the condition phrase maps to a flat
  string key (``"dies"``, ``"attacks"``, …) and the engine's per-event
  ``_fire_*`` functions hardcode-match it. Phases later than this one
  will migrate the remaining string-key keys to the predicate path.
- **Predicate style** (new): the condition phrase maps to an
  ``(Event, Predicate)`` pair and the engine's centralized
  ``dispatch_event`` walks every triggered ability looking for a
  matching ``event`` whose ``predicate`` evaluates true. Phase 2
  migrated the ENTERS-event family (``etb`` / ``landfall`` /
  ``artifact_enters_you_control`` and the qualified variants); phase 4
  migrated the DIES / LEAVES_BATTLEFIELD families (this-creature-dies,
  this-creature-leaves-the-battlefield, counter-creature-dies, and
  the broader your-creature-or-artifact-dies / another-creature-dies-
  or-exiled observer patterns); phase 5 migrated the ATTACKS family
  (this-creature-attacks); phase 6 migrated the turn-structure family
  (``beginning_of_upkeep`` / ``beginning_of_combat`` /
  ``beginning_of_end_step`` and the ``each``-end-step variant) — these
  trigger keys aren't produced by oracle-text parsing (every
  turn-structure trigger is wired directly in the static-card
  registries) so this module's parser tables stay unchanged for that
  phase, but the engine-side predicate dispatch now subsumes the
  string-key path. Phase 7 migrated the BECOMES_TAPPED family
  (this-creature-becomes-tapped — parsed from oracle text — plus the
  aura ``enchanted_permanent_becomes_tapped`` variant which is only
  built statically) and the aura/equipment relational ATTACKS variants
  (``enchanted_creature_attacks``, ``equipped_creature_attacks``) plus
  the aura DEALT_DAMAGE family (``enchanted_creature_dealt_damage``)
  — all built directly in static-card registries via the new
  ``IsHost`` predicate, so the parser tables grow only by the
  ``this creature becomes tapped`` entry moving from
  ``_TRIGGER_PHRASES`` to ``_PREDICATE_TRIGGER_PHRASES``. Batch 11
  added the type-qualified self-ETB / self-attack family
  (``this Spacecraft enters``, ``this artifact enters``,
  ``this enchantment enters``, ``this Aura enters``,
  ``this Spacecraft attacks``) — every entry maps to the same
  ``IsSelf()`` predicate as the pre-existing ``"this creature
  enters"`` / ``"this creature attacks"`` siblings because the
  predicate's object-identity check is type-blind. The phrases are
  forward-looking grammar coverage today: every EoE printing that
  carries them is a non-creature card type that doesn't flow through
  the creature-builder ``_parse_clause`` pipeline.

Some triggered-ability lines are prefixed by an *ability-word* label
followed by an em-dash (e.g. ``Landfall — Whenever a land you
control enters, …``). Ability words are reminder labels with no
rules consequence — the substance is the ``Whenever …`` clause that
follows. ``_strip_ability_word_prefix`` strips any such label
generically so the body parses through the normal grammar; we never
need to enumerate ability words individually.
"""

from __future__ import annotations

import re

from mtg_bot.cards.base import (
    ConditionalOnTappedCreatures,
    Effect,
    PutPlusOneCounterOnSelf,
    PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    TriggeredAbility,
)
from mtg_bot.cards.parser.effects import (
    _SMALL_CARDINAL_PATTERN,
    _parse_amount,
    _parse_effect_phrase,
)
from mtg_bot.effects.object_predicates import (
    HasCounter,
    HasKeyword,
    HasPowerAtMost,
    IsArtifact,
    IsCreature,
    IsEnchantment,
    IsLand,
    IsNontoken,
    IsPlayer,
    IsTapped,
)
from mtg_bot.effects.state_predicates import (
    Delirium,
    OpponentControlsMoreLandsThanYou,
    SpellsCastThisTurn,
    VoidActive,
)
from mtg_bot.effects.predicates import (
    OPPONENT,
    YOU,
    Always,
    Controlled,
    Event,
    IsSelf,
    Predicate,
    Source,
    StateBound,
    Subject,
)


_TRIGGERED_ABILITY_LINE = re.compile(
    r"^(?:When|Whenever) (.+?), (.+)\.\"?$"
)
# The optional trailing ``\"`` allows oracle lines whose effect phrase
# ends with a quoted ability description — e.g. Station Monitor's
# "…create a Drone token with flying and "This token can block only
# creatures with flying."" where the outer closing double-quote follows
# the period inside the quoted sub-text.  Standard trigger lines ending
# with bare ``.`` are unchanged (the ``\"?`` matches zero quotes).


# ETB-replacement-style "enters with counters" line.
#
# Scryfall prints these as a standalone sentence — ``This creature
# enters with a +1/+1 counter on it.`` (Dockworker Drone) or with a
# numeric variant like ``This creature enters with two +1/+1 counters
# on it.`` Mechanically these are static replacement effects (CR 614),
# but the engine doesn't model replacements today; the static-card
# convention is to express them as a ``TriggeredAbility(event=ENTERS,
# predicate=IsSelf(), effect=PutPlusOneCounterOnSelf(amount=N))`` pair
# — see Dockworker Drone in ``cards/eoe_creatures.py`` for the
# canonical static layout. The parser mirrors that convention here:
# the line shape is recognized as a triggered-ability source even
# though its surface text doesn't carry a ``When``/``Whenever``
# keyword, and the produced ``TriggeredAbility`` matches the static
# entry shape verbatim.
#
# The body capture group covers everything between ``enters with`` and
# the trailing period. ``_parse_enters_with_body`` translates the body
# to an ``Effect`` — currently the bare ``a +1/+1 counter on it`` form,
# the spelled-out / digit ``N +1/+1 counters on it`` form (forward-
# looking — no EoE printing uses N>1 today, but the grammar shape is
# uniform and trivially extends), and the per-other-creature-or-
# artifact scaling form (Luxknight Breacher).
_ENTERS_WITH_REPLACEMENT_LINE = re.compile(
    r"^This creature enters with (.+)\.$"
)


# "Enters with N +1/+1 counters on it" — the bare counter-creation
# body. The amount tail accepts ``a`` (= 1), spelled-out small
# cardinals (``two`` etc.), and bare digits; ``a`` requires the
# singular ``counter`` tail, the others require the plural
# ``counters`` tail. Maps to ``PutPlusOneCounterOnSelf(amount=N)``.
_ENTERS_WITH_PLUS_ONE_COUNTERS_BODY = re.compile(
    rf"^(?:a \+1/\+1 counter|"
    rf"(\d+|{_SMALL_CARDINAL_PATTERN}) \+1/\+1 counters) on it$"
)


# "Enters with a +1/+1 counter on it for each other creature and/or
# artifact you control" — Luxknight Breacher's printed body. Maps to
# ``PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()``, the same
# engine class the static registry uses. The phrase is matched as a
# literal (no captures): the per-other-creature-or-artifact scaling
# is the engine class's whole semantics, so there are no parameters
# to extract.
_ENTERS_WITH_PLUS_ONE_PER_OTHER_CREATURE_OR_ARTIFACT_BODY = re.compile(
    r"^a \+1/\+1 counter on it for each other "
    r"creature and/or artifact you control$"
)


def _parse_enters_with_body(body: str) -> Effect | None:
    """Translate the body of an ``enters with`` line to the engine
    ``Effect`` the static registry would carry on the same printing.

    Returns ``None`` if the body shape isn't claimed — the caller
    propagates the rejection up to ``_parse_triggered_ability_line``,
    which surfaces it as a parse failure for the whole oracle line.

    Tried in most-specific-first order: the per-other-creature-or-
    artifact scaling shape carries a ``for each ...`` tail that the
    bare counter-creation body's regex doesn't tolerate, but ordering
    explicitly here keeps the precedence intent visible.
    """
    if _ENTERS_WITH_PLUS_ONE_PER_OTHER_CREATURE_OR_ARTIFACT_BODY.match(
        body
    ) is not None:
        return PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()
    m = _ENTERS_WITH_PLUS_ONE_COUNTERS_BODY.match(body)
    if m is not None:
        # Group 1 is the captured count word/digit for the plural
        # ``N +1/+1 counters`` shape; the bare ``a +1/+1 counter``
        # alternative has no capture, so a missing group resolves
        # to amount=1.
        amount = _parse_amount(m.group(1)) if m.group(1) else 1
        return PutPlusOneCounterOnSelf(amount=amount)
    return None


# Intervening-if shape: ``When/Whenever <condition>, if <gate>, <effect>.``
# (CR 603.4). The gate sits BETWEEN the trigger condition and the
# effect, gated on game state at both trigger time and resolve time.
# We rely on the regex's non-greedy ``<condition>`` and ``<gate>`` to
# stop at the first ``, `` boundary so the leading clause stays
# minimal; the effect captures the rest of the sentence up to the
# trailing period. A line that doesn't carry an ``if`` middle clause
# falls through to ``_TRIGGERED_ABILITY_LINE`` unchanged.
_TRIGGERED_ABILITY_INTERVENING_IF_LINE = re.compile(
    r"^(?:When|Whenever) (.+?), if (.+?), (.+)\.$"
)


# Intervening-while shape: ``When/Whenever <condition> while <gate>,
# <effect>.`` Mechanically equivalent to the middle-clause
# intervening-if shape (CR 603.4) — the gate sits BETWEEN the trigger
# condition and the effect, gated on game state at trigger time. The
# surface uses ``while`` instead of ``, if`` and the gate isn't
# delimited by a leading ``, `` boundary; the gate captures everything
# between `` while `` and the trailing ``, ``. Hand That Feeds prints
# ``Delirium — Whenever this creature attacks while there are four or
# more card types among cards in your graveyard, it gets +2/+0 and
# gains menace until end of turn.`` — the ``while`` clause is the
# canonical Delirium gate phrase, same predicate as the
# ``_INTERVENING_IF_PHRASES`` middle-clause table reuses.
#
# Tried AFTER the standard ``, if <gate>, `` intervening-if regex so a
# line carrying both shapes (none today) claims via the more-explicit
# ``, if `` shape first; the ``while`` shape's ``<condition>`` capture
# is non-greedy and stops at the first `` while `` boundary, keeping
# the trigger condition's body intact.
_TRIGGERED_ABILITY_INTERVENING_WHILE_LINE = re.compile(
    r"^(?:When|Whenever) (.+?) while (.+?), (.+)\.$"
)


# Trailing-if shape: ``When/Whenever <condition>, <effect> if <gate>.``
# Mechanically equivalent to the middle-clause intervening-if shape
# above (CR 603.4) — the gate sits AFTER the effect rather than
# between the trigger condition and the effect, but the trigger-time
# game-state check is the same. The Delirium "tail-rider" surface
# (``... if you have four or more card types among cards in your
# graveyard.``) is the canonical example; future trailing gates can
# add entries to the gate phrasebook without touching this regex.
#
# The condition capture is non-greedy to stop at the first ``, ``
# boundary; the effect capture is non-greedy too because the gate
# clause is identified by its leading `` if `` and trailing period.
# We anchor the gate at `` if `` (with surrounding word boundaries
# in the spelling — leading space, trailing period) so an ``if``
# inside the effect itself (as part of e.g. a sub-clause future
# extension) doesn't accidentally trigger this branch.
#
# Tried AFTER the middle-clause intervening-if regex so any line
# carrying both ``, if <gate>,`` AND ``... if <gate>.`` claims via
# the more-specific middle shape first. No DSK printing today
# combines the two; this ordering is purely defensive.
_TRIGGERED_ABILITY_TRAILING_IF_LINE = re.compile(
    r"^(?:When|Whenever) (.+?), (.+?) if (.+)\.$"
)


# Turn-structure trigger line shape: ``At the beginning of <when>,
# <effect>.`` Distinct from the ``When``/``Whenever`` shape above
# because the trigger word is ``At`` and the condition body is a
# fixed-phrase time descriptor (``your end step``, ``each end step``,
# etc.) rather than a free-form event clause. The ``<when>`` capture
# is non-greedy to stop at the first ``, `` boundary so the time
# descriptor stays minimal; the effect captures the rest of the
# sentence up to the trailing period.
#
# Two intervening-clause variants live alongside this base shape:
#
# - **Tapped-creatures effect-wrapper** (Frontline War-Rager / Flight-
#   Deck Coordinator / Sami / Sunstar Chaplain / Vaultguard Trooper /
#   Dawnstrike Vanguard): ``At the beginning of <when>, if you control
#   two or more tapped creatures, <effect>.`` The middle ``if`` clause
#   doesn't AND a predicate onto the trigger event the way the
#   standard ``_TRIGGERED_ABILITY_INTERVENING_IF_LINE`` does — the
#   static registry wraps the inner effect in
#   ``ConditionalOnTappedCreatures(effect=inner, min_count=2)`` and
#   leaves the trigger predicate at ``Controlled(YOU)``. Mirroring that
#   here keeps parser/static parity exact.
# - **Trailing-if-gained-life clause** (Haliya, Guided by Light): ``At
#   the beginning of <when>, <effect> if you've gained N or more life
#   this turn.`` The ``if`` rides AFTER the effect (not before, and
#   not in the middle) and the static wraps the inner effect in
#   ``ConditionalIfYouGainedLifeAtLeast(threshold=N, inner=...)``.
#
# Both variants are tried before the bare shape so the more-specific
# clause structure wins. Each variant has its own regex below.
_AT_THE_BEGINNING_OF_LINE = re.compile(
    r"^At the beginning of (.+?), (.+)\.$"
)


# Void ability-word prefix specifically — when the trigger line starts
# with ``Void — At the beginning of <when>, …`` the prefix is a marker
# that the produced ``TriggeredAbility`` must carry
# ``requires_void=True`` (mirroring the static-registry layout — see
# Insatiable Skittermaw, Voidforged Titan, Kavaron Skywarden,
# Hylderblade in ``cards/eoe_creatures.py``). The body of every Void
# ``At the beginning of <when>`` line in the EoE pool also carries the
# matching ``if a nonland permanent left the battlefield this turn or
# a spell was warped this turn, `` middle clause; that clause is
# elided here because the engine's ``requires_void`` flag is the
# canonical gate (the gate phrase and the flag are 1:1 by
# construction — every Void rider in print today uses the same gate
# wording).
#
# The standard ``_strip_ability_word_prefix`` helper only strips an
# ability-word prefix in front of ``When``/``Whenever`` (its lookahead
# guards against accidental matches in type-line strings), so the
# ``Void — At …`` shape doesn't flow through it. Detecting the prefix
# explicitly here keeps the Void-flag handling local to the new line
# shape — extending the generic helper would also mean coupling its
# strip semantics to a new "this prefix sets a flag" side effect,
# which is harder to keep readable.
_VOID_PREFIX = "Void — "


# Void-rider middle clause — the "intervening-if" surface text on a
# Void ``At the beginning of <when>`` line. The body parser strips
# this prefix when it follows a ``Void — `` line marker; the
# ``requires_void=True`` flag on the produced ``TriggeredAbility``
# replaces the gate's predicate-side semantics.
_VOID_GATE_PREFIX = (
    "if a nonland permanent left the battlefield this turn or "
    "a spell was warped this turn, "
)


# Tapped-creatures effect-wrapper prefix on the body of an
# ``At the beginning of <when>, <body>.`` line. When the body starts
# with this exact phrase, the inner effect (the tail after the comma)
# is wrapped in ``ConditionalOnTappedCreatures(effect=inner,
# min_count=2)`` and the trigger predicate is left at whatever the
# time-condition phrasebook yielded (typically ``Controlled(YOU)``).
#
# This mirrors the static-registry shape exactly: see Frontline War-
# Rager / Flight-Deck Coordinator / Sami / Sunstar Chaplain /
# Vaultguard Trooper / Dawnstrike Vanguard in
# ``cards/eoe_creatures.py``. The static does NOT layer the gate as a
# predicate AND on the trigger event — it wraps the effect — so the
# parser can't use the standard intervening-if predicate path here.
#
# ``min_count`` is fixed at 2 because every EoE printing today uses
# the "two or more" threshold; the engine class accepts an arbitrary
# min_count, and a future printing with "three or more tapped
# creatures" would land its own prefix entry rather than parameterizing
# this one (parser/static parity is easier when the surface text and
# the engine constant match 1:1).
_TAPPED_CREATURES_GATE_PREFIX = "if you control two or more tapped creatures, "


# DSK Survival ability-word prefix — when the trigger line starts with
# ``Survival — At the beginning of <when>, …`` the prefix is a marker
# that the produced ``TriggeredAbility`` is the Survival post-combat-
# main fire. The ability word itself carries no rules consequence
# (CR 207.2c — ability words are flavor-only labels), so we strip the
# prefix and let the body parse through the same path as any other
# ``At the beginning of …`` line. Detecting it explicitly here keeps
# the strip surface narrow: the generic ``_strip_ability_word_prefix``
# helper above only removes ability-word prefixes in front of
# ``When``/``Whenever`` bodies (its lookahead guards against accidental
# matches in type-line strings), so the ``Survival — At …`` shape
# doesn't flow through it. Mirrors the ``Void — `` precedent
# (``_VOID_PREFIX``) — same shape, no flag because Survival's gate is
# the standard ``IsTapped`` intervening-if predicate, not an engine-
# wide flag like ``requires_void``.
_SURVIVAL_PREFIX = "Survival — "


# Survival-rider middle clause — the "intervening-if" surface text on a
# Survival ``At the beginning of your second main phase`` line. The
# body parser strips this prefix and AND-s the
# ``IsSelf() & IsTapped()`` predicate onto the trigger's base
# predicate, so the resulting ``TriggeredAbility`` fires only when the
# source is tapped at the start of its controller's second main phase.
# Two surface variants exist in print: the canonical
# ``"if this creature is tapped, "`` (every non-named printing) and a
# named-self variant where the printing's own short name appears in
# place of ``"this creature"`` (Kona, Rescue Beastie / Rip, Spawn
# Hunter). The named variant is normalized into the canonical
# ``"this creature"`` form by ``_substitute_self_name`` before this
# lookup runs, so we only need to recognize the canonical phrase here.
_SURVIVAL_GATE_PREFIX = "if this creature is tapped, "


# Time-condition phrases for the ``At the beginning of`` line shape →
# tuple of (Event, Predicate). The condition is the verbatim oracle
# phrase between ``At the beginning of`` and the first ``, ``
# boundary — ``your end step``, ``each end step``, ``your upkeep``,
# etc. Each entry maps to the matching turn-structure event and the
# controller-gate predicate appropriate for the surface text:
# ``your X`` reads as "this is MY phase/step" via ``Controlled(YOU)``
# (the predicate matches a Player subject by ``player.id ==
# source.controller_id``), and ``each X`` reads as "any player's
# phase/step" via ``Always()`` (Ragost's untap-self trigger fires on
# every end step regardless of whose it is).
#
# Today only ``BEGINNING_OF_END_STEP`` mappings are populated — every
# unclaimed creature in the EoE pool with a turn-structure trigger
# uses the end-step variant. The ``BEGINNING_OF_UPKEEP`` and
# ``BEGINNING_OF_COMBAT`` events are reachable when a future printing
# carries them; the phrasebook structure scales to those without
# further regex work.
_AT_BEGINNING_OF_PHRASES: dict[str, tuple[Event, Predicate]] = {
    "your end step": (Event.BEGINNING_OF_END_STEP, Controlled(YOU)),
    "each end step": (Event.BEGINNING_OF_END_STEP, Always()),
    # ``At the beginning of combat on your turn`` — Molecular Modifier's
    # pump-and-grant trigger and Ouroboroid's mass-counter trigger.
    # Surface form pairs the ``BEGINNING_OF_COMBAT`` event with the
    # ``Controlled(YOU)`` controller gate (the trigger fires only on
    # the controller's own combat). Mirrors the ``your end step``
    # / ``each X`` precedent: ``your X`` reads as
    # ``Controlled(YOU)``; an ``each X`` variant would map to
    # ``Always()``, but no EoE printing today carries
    # ``each combat`` so only the controller-gated form is wired.
    "combat on your turn": (
        Event.BEGINNING_OF_COMBAT, Controlled(YOU),
    ),
    # ``At the beginning of your second main phase`` — DSK Survival
    # ability-word trigger. Every printed Survival creature in
    # Duskmourn carries the same surface shape:
    # ``Survival — At the beginning of your second main phase, if this
    # creature is tapped, <effect>.`` The ``Controlled(YOU)`` arm gates
    # to the active player's own post-combat main; the ``if this
    # creature is tapped`` intervening-if clause is handled separately
    # by ``_parse_at_the_beginning_of_line`` (it AND-s an
    # ``IsSelf() & IsTapped()`` predicate onto the trigger). Mirrors
    # the ``your end step`` / ``combat on your turn`` precedent.
    "your second main phase": (
        Event.BEGINNING_OF_POSTCOMBAT_MAIN, Controlled(YOU),
    ),
}


def _parse_at_beginning_of_phrase(
    phrase: str,
) -> tuple[Event, Predicate] | None:
    return _AT_BEGINNING_OF_PHRASES.get(phrase.strip())


# Trailing once-per-turn rider on a triggered-ability line. Scryfall
# prints this as a second sentence glued to the trigger body with a
# single space separator: ``<trigger sentence>. This ability triggers
# only once each turn.`` Mechan Assembler (and Nanoform Sentinel,
# whose effect is out of scope today) carry this rider. Stripping the
# suffix and setting ``once_per_turn=True`` on the resulting
# ``TriggeredAbility`` mirrors the static registry's
# ``TriggeredAbility(..., once_per_turn=True)`` shape exactly. The
# engine field already exists; this is purely a parser-grammar
# addition.
_ONCE_PER_TURN_RIDER_SUFFIX = " This ability triggers only once each turn."


# Trigger-condition phrases → engine-level trigger key (legacy
# string-key shape). Pre-phase-2 this dict held every recognized
# condition; phase 2 lifted the ENTERS-event family into
# ``_PREDICATE_TRIGGER_PHRASES`` below. Phases 4-9 migrated every
# remaining parsed trigger phrase off the string-key path — this
# table is now empty. The lookup helper still exists so future
# trigger phrases that haven't yet been promoted to the predicate
# shape have a place to land.
_TRIGGER_PHRASES: dict[str, str] = {}


# Trigger-condition phrases for the predicate-trigger path → tuple of
# (Event, Predicate). Phase 2 migrated the ENTERS-event family —
# every condition phrase here corresponds to an ETB-family trigger
# whose static cards now express the firing condition via a
# ``(event=Event.ENTERS, predicate=...)`` ``TriggeredAbility``.
#
# The ``"another"`` variants for artifact/etc. enters use the same
# event but layer ``~IsSelf()`` onto the predicate so the source
# permanent is excluded from the firing pool — the engine no longer
# needs special-case "exclude the entering object" logic at the
# fire site for these.
_PREDICATE_TRIGGER_PHRASES: dict[
    str,
    tuple[Event, Predicate] | tuple[Event, Predicate, bool],
] = {
    # ``Whenever this creature enters`` — the source's own ETB.
    "this creature enters": (Event.ENTERS, IsSelf()),
    # Land-enters ("landfall"). The ``Landfall —`` ability-word label,
    # when present in oracle text, is stripped by
    # ``_strip_ability_word_prefix`` before this lookup runs.
    "a land you control enters": (
        Event.ENTERS, Subject(None, IsLand()) & Controlled(YOU),
    ),
    # Artifact-enters-you-control. Both ``an artifact you control
    # enters`` (Mm'menon, Uthros Exile) and ``another artifact you
    # control enters`` (Mechan Assembler, Oreplate Pangolin) collapse
    # to the same predicate. The legacy
    # ``_fire_artifact_enters_triggers`` engine helper excluded the
    # entering object from the firing pool unconditionally, so the
    # ``a`` vs ``another`` wording was a surface distinction without
    # behavioral impact — preserving that here keeps the migration
    # behavior-identical to the legacy fire path.
    "an artifact you control enters": (
        Event.ENTERS, Subject(None, IsArtifact()) & Controlled(YOU) & ~IsSelf(),
    ),
    "another artifact you control enters": (
        Event.ENTERS, Subject(None, IsArtifact()) & Controlled(YOU) & ~IsSelf(),
    ),
    # ``Whenever an opponent plays a land`` — Burgeoning. The
    # ``Event.PLAYS_LAND`` dispatch carries ``("land", "controller")``
    # subjects (see `effects/predicates.py` ``_EVENT_SUBJECTS``); the
    # ``Controlled(OPPONENT, subject="controller")`` predicate gates the
    # firing pool to land-plays whose acting player is an opponent of
    # the source's controller.
    "an opponent plays a land": (
        Event.PLAYS_LAND, Controlled(by=OPPONENT, subject="controller"),
    ),
    # ``When this creature dies`` — the source's own death (Gravpack
    # Monoist's create-Robot trigger and the synthetic ``draw 2 cards``
    # death triggers used in parser tests). Maps to the centralized
    # DIES dispatch path; the dying object is included as a candidate
    # observer via ``extra_observers`` so ``IsSelf()`` matches even
    # though the object has been moved off the battlefield.
    "this creature dies": (Event.DIES, IsSelf()),
    # ``When this creature leaves the battlefield`` (Anticausal Vestige).
    # The compound ``enters or leaves`` form (Cryogen Relic) is a
    # different shape and intentionally stays out of this entry's
    # scope.
    "this creature leaves the battlefield": (
        Event.LEAVES_BATTLEFIELD, IsSelf(),
    ),
    # ``When this artifact is put into a graveyard from the
    # battlefield`` (Nutrient Block). Per CR 700-series the
    # graveyard-from-battlefield zone change is a strict subset of
    # LEAVES_BATTLEFIELD (the latter also fires on exile / hand /
    # library returns). The static-registry ``Nutrient Block`` entry
    # already simplifies the printed graveyard-zone trigger to
    # LEAVES_BATTLEFIELD with the documented "slightly looser — also
    # fires if exiled" caveat; the parser mirrors that simplification
    # so per-card parity holds. The type-blind ``IsSelf()`` predicate
    # handles the artifact identity check — same shape as the
    # ``this creature leaves the battlefield`` entry above.
    "this artifact is put into a graveyard from the battlefield": (
        Event.LEAVES_BATTLEFIELD, IsSelf(),
    ),
    # ``Whenever this creature attacks`` — the source's own attack
    # (Spacecraft attack triggers, Stellar Sights creatures, etc.).
    # ``Event.ATTACKS`` carries a single subject ``"attacker"``;
    # ``IsSelf()`` resolves it via ``_resolve_subject``'s single-
    # subject default. Phase 5 migrated every ``trigger="attacks"``
    # static card to this predicate shape.
    "this creature attacks": (Event.ATTACKS, IsSelf()),
    # ``Whenever this creature becomes tapped`` — the source's own
    # tap (Starfighter Pilot, Nanoform Sentinel, Astrobiologist,
    # Sterling Hippogriff, etc.). ``Event.BECOMES_TAPPED`` carries a
    # single subject ``"obj"``; ``IsSelf()`` resolves it via the
    # single-subject default. Phase 7 migrated every
    # ``trigger="becomes_tapped"`` static card to this predicate
    # shape.
    "this creature becomes tapped": (Event.BECOMES_TAPPED, IsSelf()),
    # ``Whenever a creature you control with a +1/+1 counter on it
    # dies`` (Meltstrider Eulogist). Predicate: another creature you
    # control bearing a +1/+1 counter — the firing source's own death
    # is naturally excluded because predicate matching considers the
    # SUBJECT (dying object), and the most common static reading of
    # this phrase is "any creature you control" including self. The
    # engine fires once per death observer; if a printing also wants
    # to gate self-firing, add an explicit ``& ~IsSelf()`` in its
    # static definition (Meltstrider Eulogist's printed text doesn't
    # have that qualifier so we keep it broad).
    "a creature you control with a +1/+1 counter on it dies": (
        Event.DIES,
        Subject(None, IsCreature() & HasCounter("+1/+1")) & Controlled(YOU),
    ),
    # ``Whenever you cast an artifact spell`` — Pinnacle Emissary's
    # "whenever you cast an artifact spell, create a Drone token."
    # ``Event.IS_CAST`` carries subjects ``("spell", "controller")``;
    # ``Subject("spell", IsArtifact())`` matches the spell being cast
    # and ``Controlled(YOU, subject="controller")`` gates on it being
    # cast by the source's controller (vs an opponent).
    "you cast an artifact spell": (
        Event.IS_CAST,
        Subject("spell", IsArtifact()) & Controlled(YOU, subject="controller"),
    ),
    # ``Whenever you cast your second spell each turn`` — phase 8
    # migrated from ``trigger="second_spell_each_turn"`` to the
    # predicate path. ``Event.IS_CAST`` carries subjects ``("spell",
    # "controller")``; the ``Controlled(YOU, subject="controller")``
    # half gates "you cast" (vs an opponent), and
    # ``SpellsCastThisTurn(n=2)`` latches the moment the controller's
    # running total flips to 2 — mirroring the legacy
    # ``p.spells_cast_this_turn == 2`` gate at the
    # ``actions._fire_second_spell_triggers`` call site. Used by
    # Illvoi Operative, Sunstar Lightsmith, Cosmogrand Zenith,
    # Station Monitor.
    "you cast your second spell each turn": (
        Event.IS_CAST,
        Controlled(YOU, subject="controller") & StateBound(SpellsCastThisTurn(n=2)),
    ),
    # ``Whenever you sacrifice another creature or artifact`` — phase
    # 8 migrated from ``trigger="sacrifice_creature_or_artifact"``.
    # ``Event.IS_SACRIFICED`` carries the sacrificed permanent as
    # ``"obj"``; the predicate filters to creature-or-artifact subjects
    # under your control, and ``~IsSelf()`` excludes the source's own
    # sacrifice (matching the legacy ``other.id == obj.id: continue``
    # skip in ``turn.sacrifice``). Used by Lightless Evangel.
    "you sacrifice another creature or artifact": (
        Event.IS_SACRIFICED,
        Subject(None, IsCreature() | IsArtifact()) & Controlled(YOU) & ~IsSelf(),
    ),
    # ``Whenever this creature deals combat damage to a player``
    # (Illvoi Infiltrator). Phase 9 migrated from
    # ``trigger="combat_damage_to_player"``. ``Event.DEALS_COMBAT_DAMAGE``
    # carries subjects ``("source", "target")`` — the source is the
    # damage-dealing permanent, the target is the damaged player or
    # creature. ``IsSelf(subject="source")`` ties the trigger to the
    # source's own damage-dealing; ``IsPlayer(subject="target")`` gates
    # "to a player" vs "to a creature". The broader ``a creature you
    # control deals combat damage`` variant (Starwinder) stays out of
    # the parser today — it's built directly in the static-card
    # registry — but a future addition would compose
    # ``IsCreature(subject="source") & Controlled(YOU,
    # subject="source") & IsPlayer(subject="target")`` here.
    "this creature deals combat damage to a player": (
        Event.DEALS_COMBAT_DAMAGE,
        IsSelf(subject="source") & Subject("target", IsPlayer()),
    ),
    # ``Whenever this creature blocks a creature with flying``
    # (Skystinger). Phase 9 migrated from
    # ``trigger="blocks_creature_with_flying"``. ``Event.BLOCKS``
    # carries subjects ``("blocker", "attacker")``;
    # ``IsSelf(subject="blocker")`` ties the trigger to the source
    # being the blocker; ``HasKeyword("flying", subject="attacker")``
    # gates on the attacker having flying right now (printed,
    # granted, or temp). Composes naturally to other block-filter
    # triggers — ``HasKeyword("trample", subject="attacker")`` etc.
    # — when those land in the card pool.
    "this creature blocks a creature with flying": (
        Event.BLOCKS,
        IsSelf(subject="blocker") & Subject("attacker", HasKeyword("flying")),
    ),
    # ``Whenever a creature you control dies`` (Sothera, the
    # Supervoid). Batch 9. Broader than the existing
    # ``a creature you control with a +1/+1 counter on it dies``
    # phrase — no counter qualifier, fires on every death of a
    # creature you control. The static-card registry does NOT layer
    # ``~IsSelf()`` on Sothera's printed text, so the source itself
    # dying still triggers (a creature you control includes itself
    # while it's still on the battlefield at the moment its death
    # event resolves through the dispatcher's ``extra_observers``
    # path).
    "a creature you control dies": (
        Event.DIES, Subject(None, IsCreature()) & Controlled(YOU),
    ),
    # ``Whenever you sacrifice an artifact`` (Biotech Specialist).
    # Batch 9. Distinct from the existing
    # ``you sacrifice another creature or artifact`` (Lightless
    # Evangel) phrase — narrower type filter (artifact only) and the
    # ``an``-vs-``another`` surface distinction. The static registry
    # for Biotech layers ``~IsSelf()`` for symmetry with the
    # ``another`` family even though Biotech itself is a non-artifact
    # creature (so the source can never be the sacrificed artifact);
    # mirroring that here keeps parser/static parity exact and
    # leaves room for a future artifact-source card to print the
    # same phrase without re-introducing self-fire.
    "you sacrifice an artifact": (
        Event.IS_SACRIFICED,
        Subject(None, IsArtifact()) & Controlled(YOU) & ~IsSelf(),
    ),
    # ``Whenever a creature you control deals combat damage to a
    # player`` (Starwinder). Batch 9. Broader than the existing
    # ``this creature deals combat damage to a player`` phrase
    # (Illvoi Infiltrator) — fires for any creature you control,
    # not just the source. ``Event.DEALS_COMBAT_DAMAGE`` carries
    # subjects ``("source", "target")``;
    # ``IsCreature(subject="source") & Controlled(YOU,
    # subject="source")`` gates "a creature you control" on the
    # damage dealer; ``IsPlayer(subject="target")`` gates "to a
    # player" vs "to a creature". Mirrors the static registry's
    # Starwinder predicate exactly.
    "a creature you control deals combat damage to a player": (
        Event.DEALS_COMBAT_DAMAGE,
        Subject("source", IsCreature())
        & Controlled(YOU, subject="source")
        & Subject("target", IsPlayer()),
    ),
    # ``Whenever one or more creatures you control deal combat damage
    # to a player`` (Locker Room — DSK Room half). Same per-event
    # predicate as the ``a creature you control deals combat damage
    # to a player`` Starwinder entry above, but the printed
    # "one or more" wording carries an implicit aggregated semantic:
    # one fire per combat regardless of how many of the controller's
    # creatures connect. Modeled by the existing
    # ``TriggeredAbility.once_per_turn=True`` flag — the per-source
    # ``DEALS_COMBAT_DAMAGE`` event still dispatches once per
    # connecting attacker, but the ledger caps Locker Room's draw
    # at a single fire per turn (which lines up 1:1 with one fire
    # per combat in our single-combat-per-turn model). The 3-tuple
    # value form threads ``once_per_turn=True`` through to the
    # produced ``TriggeredAbility`` — the dispatcher OR-combines this
    # phrase-level flag with the trailing-rider
    # ``This ability triggers only once each turn.`` flag so future
    # printings carrying both wordings stay correct.
    "one or more creatures you control deal combat damage to a player": (
        Event.DEALS_COMBAT_DAMAGE,
        Subject("source", IsCreature())
        & Controlled(YOU, subject="source")
        & Subject("target", IsPlayer()),
        True,
    ),
    # ``Whenever a nontoken creature you control with a +1/+1
    # counter on it dies`` (Rayblade Trooper). Batch 9. Stricter
    # variant of the existing ``a creature you control with a
    # +1/+1 counter on it dies`` (Meltstrider Eulogist) — adds
    # ``IsNontoken()`` so token creatures dying don't satisfy the
    # trigger. Mirrors the static registry's Rayblade predicate
    # exactly.
    "a nontoken creature you control with a +1/+1 counter on it dies": (
        Event.DIES,
        Subject(None, IsCreature() & IsNontoken() & HasCounter("+1/+1"))
        & Controlled(YOU),
    ),
    # ``Whenever another creature or artifact you control enters``
    # (Weftstalker Ardent). Batch 9. Compound type qualifier — the
    # entering object can be either a creature OR an artifact under
    # your control, with the source itself excluded via ``~IsSelf()``
    # (the ``another`` qualifier). Mirrors the static registry's
    # Weftstalker predicate exactly. The ``a creature or artifact you
    # control enters`` (no ``another``) variant is not in the EoE
    # pool so it isn't added here — would be the same predicate
    # following the ``an artifact you control enters`` collapse
    # precedent if it ever ships.
    "another creature or artifact you control enters": (
        Event.ENTERS,
        Subject(None, IsCreature() | IsArtifact()) & Controlled(YOU) & ~IsSelf(),
    ),
    # Batch 11 — type-qualified self-ETB / self-attack family. Every
    # entry in this group maps to the same ``IsSelf()`` predicate as
    # the pre-existing ``"this creature enters"`` / ``"this creature
    # attacks"`` entries: the source's identity is what matters, not
    # what type qualifier the printed text uses. The predicate's
    # object-identity check is type-blind, so the parser collapses
    # ``"this Spacecraft enters"`` / ``"this artifact enters"`` /
    # ``"this enchantment enters"`` / ``"this Aura enters"`` /
    # ``"this Spacecraft attacks"`` onto the same shape as the
    # creature-typed sibling.
    #
    # The EoE printings that carry these phrases today are all non-
    # creature card types (Spacecraft are artifacts; the Aura/
    # Enchantment/Artifact phrasings appear on their respective non-
    # creature card types) and don't currently flow through the
    # creature-builder ``_parse_clause`` pipeline that calls this
    # phrasebook — so the additions are forward-looking grammar
    # coverage in the same shape as batch 10's ``"it's your turn"``
    # static-condition entry. Adding them keeps the parser ready for
    # reprints and for future creature-typed printings (e.g. an
    # artifact-creature whose oracle text says ``"this artifact
    # enters"``) without lying about engine semantics. Every entry's
    # static-card sibling already builds via ``IsSelf()`` directly
    # (see ``cards/eoe_spells.py`` Pinnacle Starcage / Tractor Beam /
    # Larval Scoutlander / Specimen Freighter etc.), so parser/
    # static-registry parity holds out of the box.
    "this Spacecraft enters": (Event.ENTERS, IsSelf()),
    "this artifact enters": (Event.ENTERS, IsSelf()),
    "this enchantment enters": (Event.ENTERS, IsSelf()),
    "this Aura enters": (Event.ENTERS, IsSelf()),
    "this Spacecraft attacks": (Event.ATTACKS, IsSelf()),
    # ``Whenever you attack`` — DSK's once-per-declare-attackers trigger
    # family (Razorkin Hordecaller, Most Valuable Slayer, Lionheart
    # Glimmer). Distinct from ``"this creature attacks"`` (which fires
    # per-attacker via ``Event.ATTACKS``): ``Event.YOU_ATTACK`` carries
    # a single Player subject (``"player"``) and fires exactly ONCE per
    # declare-attackers pass when the active player has attacked with
    # at least one creature. ``Controlled(YOU)`` against a Player
    # subject reads as "the attacking player is the source's controller"
    # — same shape used by the turn-structure events
    # (``BEGINNING_OF_UPKEEP`` etc.). The trigger doesn't multiply with
    # attacker count, so the per-pass dispatch is the right semantic
    # for the printed phrasing.
    "you attack": (Event.YOU_ATTACK, Controlled(YOU)),
    # ``Whenever another creature you control with power 2 or less
    # enters`` — Vicious Clown's pump-self trigger and Irreverent
    # Gremlin's discard-and-draw trigger both print this exact
    # condition phrase. The predicate filters the entering subject to
    # creatures the source's controller controls whose effective power
    # is at most 2, with ``~IsSelf()`` excluding the source itself
    # (the ``another`` qualifier). The threshold is parameterized via
    # ``HasPowerAtMost(n=2)`` so a future printing with a different
    # threshold (``power 3 or less`` etc.) lands as a separate phrase
    # entry rather than a regex-driven dispatch — same pattern as
    # the existing fixed-text trigger phrasebook.
    "another creature you control with power 2 or less enters": (
        Event.ENTERS,
        Subject(None, IsCreature() & HasPowerAtMost(n=2)) & Controlled(YOU) & ~IsSelf(),
    ),
    # ``When you unlock this door`` — Duskmourn Room door triggers. The
    # canonical static-side wiring uses ``Event.ENTERS`` + ``IsSelf()``
    # routed through the Room's per-half ``unlocked_halves`` gate (see
    # ``_unlock_room_half`` / ``_fire_unlock_door_triggers`` in
    # ``actions.py``) — same predicate shape as ``"this creature
    # enters"``. The room builder threads each half's flattened
    # ``TriggeredAbility`` onto the parent printing with ``room_half``
    # tagged so the engine fires the door trigger only when its half
    # is unlocked (cast-time for the cast half, ``UnlockRoomHalf`` for
    # the other). This phrasebook entry is the unified-grammar entry
    # point — every Room half whose body reads ``"When you unlock this
    # door, <effect>"`` flows through the same parser path the rest of
    # the trigger grammar uses.
    "you unlock this door": (Event.ENTERS, IsSelf()),
}


# Compound trigger-condition phrases for the predicate-trigger path → a
# tuple of (Event, Predicate) pairs, one per condition arm. A single
# oracle line whose condition is ``X or Y`` (e.g. Thawbringer's
# ``When this creature enters or dies``) decomposes into multiple
# ``TriggeredAbility`` entries — one per arm — sharing the same effect.
# This mirrors the static-registry layout, where Thawbringer's two
# events live as two separate ``TriggeredAbility`` entries on the
# printing.
#
# The compound table is consulted BEFORE the single ``_PREDICATE_
# TRIGGER_PHRASES`` lookup so the more-specific compound shape wins.
# An entry here can co-exist with single-shape entries for either arm
# (``"this creature enters"`` and ``"this creature dies"`` both already
# live in ``_PREDICATE_TRIGGER_PHRASES`` — the compound entry below
# composes them without re-implementing either arm's predicate).
_COMPOUND_PREDICATE_TRIGGER_PHRASES: dict[
    str, tuple[tuple[Event, Predicate], ...]
] = {
    # ``When this creature enters or dies`` (Thawbringer) — fires once
    # on entry and once on death. The two arms share the same effect at
    # the line level; the parser produces one ``TriggeredAbility`` per
    # arm so the engine's per-event dispatch sees them as independent
    # observers (which matches the static-registry's two-entry layout).
    "this creature enters or dies": (
        (Event.ENTERS, IsSelf()),
        (Event.DIES, IsSelf()),
    ),
    # ``Whenever this creature enters or attacks`` — fires once on
    # entry and once on each attack. Surfaces as the named-self form
    # ``"Whenever <name> enters or attacks"`` on Haliya, Ascendant
    # Cadet and Syr Vondam, the Lucent (the parser does not yet
    # substitute the printing's own name for ``"this creature"`` in
    # trigger-condition phrases, so neither claims end-to-end today —
    # this entry is the trigger-phrase half of those pending claims).
    # The two arms share the same effect at the line level; the parser
    # produces one ``TriggeredAbility`` per arm to mirror the static-
    # registry's two-entry layout (Haliya / Syr Vondam each carry two
    # separate ``TriggeredAbility`` entries with identical effects in
    # their static definitions).
    "this creature enters or attacks": (
        (Event.ENTERS, IsSelf()),
        (Event.ATTACKS, IsSelf()),
    ),
    # ``When this artifact enters or leaves the battlefield`` (Cryogen
    # Relic) — fires once on entry and once on leave. Mirrors the
    # static-registry layout for Cryogen Relic, which carries two
    # separate ``TriggeredAbility`` entries (ENTERS+IsSelf, then
    # LEAVES_BATTLEFIELD+IsSelf) with the same effect. The type-blind
    # ``IsSelf()`` predicate handles the artifact identity check —
    # mirrors the ``"this artifact enters"`` entry above.
    "this artifact enters or leaves the battlefield": (
        (Event.ENTERS, IsSelf()),
        (Event.LEAVES_BATTLEFIELD, IsSelf()),
    ),
    # Duskmourn "Eerie" ability-word trigger — fires both when an
    # enchantment under your control enters AND when you fully unlock
    # a Room. The ``Eerie — `` ability-word prefix strips through the
    # generic ``_strip_ability_word_prefix`` helper before this lookup
    # runs (no Eerie-specific prefix code needed; ability words have
    # no rules consequence per CR 207.2c). The printed surface joins
    # the two arms with ``and whenever`` rather than ``or`` —
    # mechanically a single oracle line produces two independent
    # ``TriggeredAbility`` entries (one per event) that share the
    # same effect, mirroring the existing Thawbringer ``enters or
    # dies`` precedent.
    #
    # Arm 1 (ENTERS): ``IsEnchantment() & Controlled(YOU)`` — any
    # enchantment under the source's controller entering, including
    # Rooms (which carry ``"Enchantment"`` in ``card_types``) and the
    # source itself if it's an Enchantment Creature (Fear of Infinity).
    # No ``~IsSelf()`` because the printed text is "an enchantment you
    # control enters" with no "another" qualifier.
    #
    # Arm 2 (ROOM_FULLY_UNLOCKED): ``Controlled(YOU, subject="room")``
    # — the Room that just transitioned to all-halves-unlocked is
    # under the source's controller. The dispatch fires from
    # ``_unlock_room_half`` at the unlock-action transition; the
    # event subject ``"room"`` is the Room ``GameObject`` itself.
    "an enchantment you control enters and whenever you fully unlock a Room": (
        (Event.ENTERS, Subject(None, IsEnchantment()) & Controlled(YOU)),
        (Event.ROOM_FULLY_UNLOCKED, Controlled(YOU, subject="room")),
    ),
}


# Trigger-condition phrases that read as ``X or Y dies/enters`` but
# whose two arms collapse onto a SINGLE predicate in the static
# registry — the wording is inclusive (``this creature or another
# <subset> you control``) and the static carries one
# ``TriggeredAbility`` whose predicate covers both halves at once.
# Distinct from ``_COMPOUND_PREDICATE_TRIGGER_PHRASES`` (which
# produces one ``TriggeredAbility`` per arm) — these collapsed
# phrases produce exactly one ability.
#
# The dispatcher's ``extra_observers=[dead]`` plumbing at the DIES
# fire site is what lets the source's own death satisfy a
# ``Controlled(YOU)`` predicate even though the dying object has
# already been moved off the battlefield by the time the trigger
# fires; the ``this creature or another …`` wording is explicit
# about including self in the firing pool.
_COLLAPSED_PREDICATE_TRIGGER_PHRASES: dict[
    str, tuple[Event, Predicate]
] = {
    # ``Whenever this creature or another creature or artifact you
    # control dies`` (Susurian Voidborn). The two arms (``this
    # creature`` + ``another creature or artifact you control``)
    # collapse to a single ``Controlled(YOU) & (creature OR artifact)``
    # predicate — no ``~IsSelf()`` because the leading ``this creature``
    # arm explicitly includes self. Mirrors the static registry's
    # Susurian Voidborn predicate exactly.
    "this creature or another creature or artifact you control dies": (
        Event.DIES,
        Subject(None, IsCreature() | IsArtifact()) & Controlled(YOU),
    ),
}


def _parse_collapsed_predicate_trigger_phrase(
    phrase: str,
) -> tuple[Event, Predicate] | None:
    return _COLLAPSED_PREDICATE_TRIGGER_PHRASES.get(phrase.strip())


def _parse_compound_predicate_trigger_phrase(
    phrase: str,
) -> tuple[tuple[Event, Predicate], ...] | None:
    return _COMPOUND_PREDICATE_TRIGGER_PHRASES.get(phrase.strip())


# Intervening-if gate phrases → predicate. The phrase is the verbatim
# oracle text between ``if`` and the trailing comma of the middle
# clause (after stripping surrounding whitespace). Each predicate is a
# game-state check evaluated against the source's controller and the
# current battlefield — none of these gates depend on a per-event
# subject, so they compose with the trigger's existing event predicate
# via plain ``And``. The dispatcher's predicate evaluation handles the
# trigger-time gate (CR 603.4 trigger check); the matching resolve-
# time recheck on the same predicate is not yet wired through the
# resolve pipeline, so a state change between trigger and resolve is
# a known gap (the existing static cards using these gates rely on
# the same trigger-time-only check today via ``requires_void`` /
# ``ConditionalIf*`` effect wrappers, which are themselves single-
# point gates).
#
# The "Void —" ability-word prefix on Alpharael Stonechosen / Roving
# Actuator strips through ``_strip_ability_word_prefix`` before this
# lookup runs, so the ``"a nonland permanent left the battlefield
# this turn or a spell was warped this turn"`` gate phrase here covers
# both prefixed and bare wordings of the Void rider.
_INTERVENING_IF_PHRASES: dict[str, Predicate] = {
    "an opponent controls more lands than you": (
        StateBound(OpponentControlsMoreLandsThanYou())
    ),
    (
        "a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn"
    ): StateBound(VoidActive()),
    # Duskmourn "Delirium" gate — the canonical printed phrasing of
    # the four-card-types-in-graveyard threshold. Pairs with the
    # generic ``Delirium — `` ability-word prefix that strips through
    # ``_strip_ability_word_prefix`` (no Delirium-specific helper
    # needed; the prefix has no rules consequence per CR 207.2c).
    # Mirrors the ``VoidActive()`` gate's role for the EoE Void
    # mechanic — same shape, different trigger-time game-state check.
    "there are four or more card types among cards in your graveyard": (
        StateBound(Delirium())
    ),
    # DSK Enduring cycle: "When this creature dies, if it was a
    # creature, return it to the battlefield under its owner's
    # control. It's an enchantment." Every Enduring creature in DSK
    # carries the same intervening-if gate, paired with the cycle's
    # return-as-enchantment effect body.
    #
    # The gate predicate is ``IsCreature()`` — it evaluates against
    # the dispatched subject (the dying object) via its printing's
    # ``card_types``. Every Enduring printing is an Enchantment
    # Creature, so the gate is naturally true on the first death
    # (the printing has "Creature"). After the effect returns the
    # card "as an enchantment", the engine's ``is_creature_now``
    # short-circuits to False (via the new
    # ``enduring_returned_as_enchantment`` flag) so the
    # ``_fire_dies_triggers`` ``was_creature`` gate at the call site
    # in ``move_off_battlefield`` already prevents a second death
    # from re-firing the trigger. The gate predicate composed here
    # is therefore behaviorally redundant in the engine today, but
    # we register it so parser/static parity stays exact and the
    # gate is in place if a future engine change ever loosened the
    # call-site ``was_creature`` check.
    "it was a creature": Subject(None, IsCreature()),
}


# Trailing-if gate phrases — the gate clause that follows the effect
# in the ``When/Whenever <condition>, <effect> if <gate>.`` shape.
# Distinct from ``_INTERVENING_IF_PHRASES`` (middle-clause gates)
# because the printed phrasing differs: middle-clause gates lead
# with an existential subject (``there are four or more card
# types``), trailing gates lead with the controller's pronoun
# (``you have four or more card types``). The predicate semantics
# are identical — both trigger-time checks delegate to the same
# ``Delirium()`` evaluation — but the surface text doesn't reuse,
# so we keep separate phrasebooks rather than registering both
# phrasings under one entry and risking a maintainer assuming
# every middle-clause phrase has a trailing twin (it doesn't —
# Void / opponent-more-lands gates have no printed trailing form).
_TRAILING_IF_PHRASES: dict[str, Predicate] = {
    "you have four or more card types among cards in your graveyard": (
        StateBound(Delirium())
    ),
}


def _parse_intervening_if_phrase(phrase: str) -> Predicate | None:
    return _INTERVENING_IF_PHRASES.get(phrase.strip())


# Intervening-``while`` gate phrases. The DSK Delirium "while X"
# shape (Hand That Feeds) reuses the same surface text as the
# middle-clause ``, if X,`` shape, so the phrasebook is identical
# today — a single registered entry for the four-or-more-card-types
# threshold. Kept as a separate dict (vs. routing the ``while`` lookup
# into ``_INTERVENING_IF_PHRASES``) so future ``while``-only or
# ``if``-only gate phrasings land cleanly on their own table without
# silently widening the other surface form's grammar.
_INTERVENING_WHILE_PHRASES: dict[str, Predicate] = {
    "there are four or more card types among cards in your graveyard": (
        StateBound(Delirium())
    ),
}


def _parse_intervening_while_phrase(phrase: str) -> Predicate | None:
    return _INTERVENING_WHILE_PHRASES.get(phrase.strip())


def _parse_trailing_if_phrase(phrase: str) -> Predicate | None:
    """Look up ``phrase`` (a trailing-``if`` gate clause from the
    ``... if <gate>.`` tail) in ``_TRAILING_IF_PHRASES``. Returns the
    matching ``Predicate`` or ``None`` when the phrase isn't claimed.
    """
    return _TRAILING_IF_PHRASES.get(phrase.strip())


# Ability-word prefix on a triggered-ability line. An ability word is
# one or more capitalized words followed by `` — `` and a triggered-
# ability body. Per the comprehensive rules ability words have no
# rules consequence, so we strip whatever label appears without
# enumerating them — Landfall, Magecraft, Constellation, Domain, etc.
# all flow through the same path. The lookahead requires
# ``When``/``Whenever`` after the dash so type-line strings (e.g.
# ``Creature — Dragon``) and other em-dash uses don't accidentally
# match.
_ABILITY_WORD_PREFIX = re.compile(
    r"^[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)* — (?=When |Whenever )"
)


def _strip_ability_word_prefix(line: str) -> str:
    """Strip a generic ``<Ability Word> — `` prefix from ``line`` if
    one is present in front of a ``When``/``Whenever`` body.

    Ability words carry no rules meaning, so the parser doesn't need
    to know which ability word it's looking at — the substance lives
    entirely in the ``Whenever …`` clause that follows. Returns
    ``line`` unchanged when no such prefix is present.
    """
    return _ABILITY_WORD_PREFIX.sub("", line, count=1)


def _parse_trigger_phrase(phrase: str) -> str | None:
    return _TRIGGER_PHRASES.get(phrase.strip())


def _parse_predicate_trigger_phrase(
    phrase: str,
) -> tuple[Event, Predicate] | None:
    """Look up ``phrase`` in the predicate-trigger phrasebook.

    Returns ``(Event, Predicate)`` for a recognized phrase or ``None``
    otherwise. Some entries' value form is a 3-tuple
    ``(Event, Predicate, once_per_turn)`` carrying an implicit
    aggregated semantic (Locker Room's "one or more creatures you
    control deal combat damage to a player" is the canonical example,
    one fire per combat regardless of how many sources connect). The
    once-per-turn flag is consulted via the sibling
    ``_predicate_trigger_phrase_is_once_per_turn`` helper so this
    primary lookup keeps its long-standing ``(Event, Predicate)`` shape
    for direct test consumers — the public surface stays stable while
    the dispatcher gains the new flag through the parallel helper.
    """
    raw = _PREDICATE_TRIGGER_PHRASES.get(phrase.strip())
    if raw is None:
        return None
    return raw[0], raw[1]


def _predicate_trigger_phrase_is_once_per_turn(phrase: str) -> bool:
    """Return whether the predicate-trigger phrasebook entry for
    ``phrase`` carries an implicit ``once_per_turn=True`` rider.

    Returns ``False`` if the phrase isn't in the phrasebook OR if the
    value form is the legacy 2-tuple ``(Event, Predicate)``. Returns
    ``True`` only for entries whose value is a 3-tuple
    ``(Event, Predicate, True)`` — the printed-wording-aggregated
    surfaces (Locker Room's "one or more creatures you control deal
    combat damage to a player" is the only such entry today). The
    dispatcher OR-combines this flag with the trailing-rider
    ``This ability triggers only once each turn.`` flag stripped
    earlier — both are honest sources of ``once_per_turn=True`` and
    either alone is sufficient for the final ``TriggeredAbility``.
    """
    raw = _PREDICATE_TRIGGER_PHRASES.get(phrase.strip())
    if raw is None or len(raw) < 3:
        return False
    return bool(raw[2])


def _substitute_self_name(line: str, name: str) -> str:
    """Replace the printing's own name (and the short-form name for
    legendary commas) with ``"this creature"`` inside the line.

    Scryfall oracle text refers to a legendary creature's source by
    its short name (the substring before the first comma — ``Haliya``
    on Haliya, Ascendant Cadet, ``Syr Vondam`` on Syr Vondam, the
    Lucent). Non-legendary printings refer to themselves either as
    ``"this creature"`` (the canonical form already accepted by every
    trigger-phrase table) or by their full printed name. This helper
    rewrites both the full and short forms uniformly, so the
    downstream condition / effect grammar only needs the canonical
    surface.

    The substitution is whole-word (regex word-boundary anchored) so
    a substring match inside another token can't fire — e.g. a
    hypothetical name ``"Sami"`` doesn't accidentally match inside
    ``"sample"``. The full name is tried before the short form so a
    printing whose full name appears verbatim in the oracle text
    (rare; usually the short form is used) still picks up the more-
    specific replacement first.
    """
    full_pattern = r"\b" + re.escape(name) + r"\b"
    line = re.sub(full_pattern, "this creature", line)
    short = name.split(",", 1)[0]
    if short and short != name:
        short_pattern = r"\b" + re.escape(short) + r"\b"
        line = re.sub(short_pattern, "this creature", line)
    return line


def _parse_at_the_beginning_of_line(
    match: "re.Match[str]",
    requires_void: bool = False,
    is_survival: bool = False,
) -> TriggeredAbility | None:
    """Body parser for ``At the beginning of <when>, <effect>.`` lines.

    ``match`` is the result of ``_AT_THE_BEGINNING_OF_LINE.match(line)``
    — group 1 is the time-condition phrase (``your end step`` etc.) and
    group 2 is the effect tail (the rest of the sentence between the
    first ``, `` and the trailing period). The time phrase resolves via
    ``_parse_at_beginning_of_phrase`` to the matching ``(Event,
    Predicate)`` pair; the effect tail is parsed via
    ``_parse_effect_phrase`` (the same primitive every other trigger
    body uses, so the unified effect grammar stays the single source
    of truth).

    ``requires_void`` is True when the calling site already detected a
    leading ``Void — `` prefix on the original line. In that case the
    body MUST start with the verbatim ``_VOID_GATE_PREFIX`` middle
    clause — the gate phrase is elided (the engine's
    ``requires_void`` flag is the canonical gate) and the remaining
    inner effect parses through the unified grammar. A void-flagged
    line whose body does NOT carry the matching gate is rejected
    outright: the printed Void rider is always paired with the gate
    phrase, and a missing gate would mean we're misreading the line.

    Returns ``None`` if any piece doesn't parse — the line has no
    other valid interpretation (unlike the When/Whenever shape, which
    has multiple line-level fallbacks); rejecting here propagates as a
    parse failure for the whole oracle line so the surrounding card
    stays unclaimed rather than silently dropping the trigger.
    """
    when_match = _parse_at_beginning_of_phrase(match.group(1))
    if when_match is None:
        return None
    event, predicate = when_match
    body = match.group(2)
    # Void rider — the line was preceded by a ``Void — `` prefix. The
    # body must carry the verbatim void-gate middle clause; strip it
    # and parse the remainder as a bare effect. The
    # ``requires_void=True`` flag on the produced ``TriggeredAbility``
    # is the canonical gate — mirrors the static-registry layout for
    # Insatiable Skittermaw / Voidforged Titan / Kavaron Skywarden /
    # Hylderblade exactly. Tried BEFORE the tapped-creatures wrapper
    # below because the two prefixes are disjoint (no printing today
    # combines a Void rider with a tapped-creatures gate), but
    # ordering ``requires_void`` first keeps the Void path's
    # "elide gate, set flag" handling local.
    if requires_void:
        if not body.startswith(_VOID_GATE_PREFIX):
            return None
        inner_phrase = body[len(_VOID_GATE_PREFIX):]
        inner = _parse_effect_phrase(inner_phrase)
        if inner is None:
            return None
        return TriggeredAbility(
            event=event, predicate=predicate, effect=inner,
            requires_void=True,
        )
    # Survival rider — the line was preceded by a ``Survival — ``
    # prefix. The body must carry the verbatim ``if this creature is
    # tapped, `` middle clause; strip it and parse the remainder as a
    # bare effect. The intervening-if gate becomes the predicate
    # ``IsSelf() & IsTapped()``, AND-ed onto the trigger's base
    # predicate (``Controlled(YOU)``) so the firing condition reads
    # "the source's controller is the active player AND the source is
    # tapped" — matching CR 603.4 trigger-time intervening-if check on
    # the dispatched event. Tried BEFORE the tapped-creatures effect-
    # wrapper below because the two prefixes are distinguished by the
    # caller's ``is_survival`` flag (no printing today combines a
    # Survival rider with a tapped-creatures gate). Mirrors the Void
    # rider precedent above structurally.
    if is_survival:
        if not body.startswith(_SURVIVAL_GATE_PREFIX):
            return None
        inner_phrase = body[len(_SURVIVAL_GATE_PREFIX):]
        inner = _parse_effect_phrase(inner_phrase)
        if inner is None:
            return None
        return TriggeredAbility(
            event=event,
            predicate=predicate & IsSelf() & Source(IsTapped()),
            effect=inner,
        )
    # Tapped-creatures effect-wrapper variant: ``if you control two or
    # more tapped creatures, <inner-effect>``. Detect the prefix, parse
    # the inner-effect tail through the unified effect grammar, and
    # wrap in ``ConditionalOnTappedCreatures``. Tried BEFORE the bare
    # body parse so the more-specific shape wins. Mirrors the static-
    # registry layout exactly — the gate is an effect wrapper, NOT a
    # predicate AND on the trigger event.
    if body.startswith(_TAPPED_CREATURES_GATE_PREFIX):
        inner_phrase = body[len(_TAPPED_CREATURES_GATE_PREFIX):]
        inner = _parse_effect_phrase(inner_phrase)
        if inner is None:
            return None
        return TriggeredAbility(
            event=event, predicate=predicate,
            effect=ConditionalOnTappedCreatures(
                effect=inner, min_count=2,
            ),
        )
    effect = _parse_effect_phrase(body)
    if effect is None:
        return None
    return TriggeredAbility(
        event=event, predicate=predicate, effect=effect,
    )


def _parse_triggered_ability_line(
    line: str, *, name: str | None = None,
) -> TriggeredAbility | tuple[TriggeredAbility, ...] | None:
    """Parse a single triggered-ability oracle line.

    Returns a single ``TriggeredAbility`` for a standard
    single-condition line, a tuple of ``TriggeredAbility`` (one per
    arm) for a compound-condition line whose phrase is claimed by
    ``_COMPOUND_PREDICATE_TRIGGER_PHRASES`` (e.g. Thawbringer's
    ``When this creature enters or dies``), or ``None`` if no shape
    matches.

    ``name``, when supplied, is the card's own printed name. Name
    references in the oracle text — both the full printed name and
    the short form (the substring before the first comma, used for
    legendary creatures) — are rewritten to the canonical
    ``"this creature"`` form before condition / effect lookup, so a
    printing whose oracle text reads ``"Whenever Haliya enters or
    attacks, …"`` flows through the same trigger-phrase table as
    the un-named ``"Whenever this creature enters or attacks, …"``
    surface. Cards that print themselves only as ``"this creature"``
    pass through unchanged (the substitution is a no-op when the
    name doesn't appear in the line).

    The clause layer (``_parse_clause``) normalizes both shapes into
    the ``triggers`` tuple on ``_ParsedClause`` so callers downstream
    don't need to special-case the compound path.
    """
    if name is not None:
        line = _substitute_self_name(line, name)
    line = _strip_ability_word_prefix(line)
    # ETB-replacement-style ``This creature enters with …`` line. This
    # surface shape doesn't carry a ``When``/``Whenever`` keyword and so
    # would never match the standard trigger-line regex below; checking
    # it up front turns the line into a ``TriggeredAbility(event=ENTERS,
    # predicate=IsSelf(), effect=...)`` mirroring the static-registry
    # convention for "enters with counters" cards (the engine doesn't
    # model replacement effects, so static and parser both express the
    # printed replacement as an ETB trigger). A failed body lookup
    # propagates as ``None`` from this branch — we don't fall through
    # to the When/Whenever paths, because an unparseable ``enters
    # with`` body has no other valid interpretation as a trigger line.
    enters_with_match = _ENTERS_WITH_REPLACEMENT_LINE.match(line)
    if enters_with_match is not None:
        effect = _parse_enters_with_body(enters_with_match.group(1))
        if effect is None:
            return None
        return TriggeredAbility(
            event=Event.ENTERS, predicate=IsSelf(), effect=effect,
        )
    # Trailing once-per-turn rider: ``<trigger sentence>. This ability
    # triggers only once each turn.`` Strip the suffix BEFORE running
    # any sentence-shape regex so the trigger body parses normally; the
    # ``once_per_turn=True`` flag is glued onto whichever
    # ``TriggeredAbility`` the body produces (intervening-if path or
    # plain-trigger path). Doing the strip up front keeps both shapes
    # supporting the rider with a single source of truth — the
    # alternative (rider-aware variants of each line regex) would
    # double the grammar without behavioral benefit.
    once_per_turn = False
    if line.endswith(_ONCE_PER_TURN_RIDER_SUFFIX):
        line = line[: -len(_ONCE_PER_TURN_RIDER_SUFFIX)]
        once_per_turn = True
    # Turn-structure ``At the beginning of <when>, <effect>.`` shape.
    # Tried before the standard ``When``/``Whenever`` paths because the
    # leading word is ``At`` and the line regex below wouldn't match it
    # anyway, but ordering by surface form keeps the dispatch narrative
    # readable. The body parser handles bare and intervening-clause
    # variants in one place — if any variant claims the line, return its
    # produced ``TriggeredAbility``; otherwise fall through to the
    # When/Whenever paths below (a future ``At ...`` shape that doesn't
    # parse here would still get a chance via the bare regex if it
    # happened to fit, though no such surface exists today).
    #
    # A leading ``Void — `` prefix is detected here and stripped before
    # the regex match so the body parser can see the bare ``At the
    # beginning of …`` shape. The flag is threaded through to the body
    # parser, which sets ``requires_void=True`` on the produced
    # ``TriggeredAbility`` and elides the matching void-gate middle
    # clause from the effect body. The standard
    # ``_strip_ability_word_prefix`` helper above only strips ability-
    # word prefixes in front of ``When``/``Whenever`` (its lookahead
    # guards against accidental matches in type-line strings), so the
    # ``Void —`` strip for ``At …`` shapes lives here instead.
    at_line = line
    requires_void = False
    # Only consume the ``Void — `` prefix if it's followed by an
    # ``At the beginning of`` body — the parser still needs the
    # prefix in place for the When/Whenever Void paths handled below
    # (those go through the generic ``_strip_ability_word_prefix``
    # which has already run; in the When/Whenever case the prefix is
    # already gone by the time we get here, but if a future helper
    # rearranges that order this guard keeps the two paths from
    # stealing each other's lines).
    if (
        at_line.startswith(_VOID_PREFIX)
        and at_line[len(_VOID_PREFIX):].startswith("At the beginning of ")
    ):
        at_line = at_line[len(_VOID_PREFIX):]
        requires_void = True
    # ``Survival — `` ability-word prefix on an ``At the beginning of``
    # body. Strip the prefix and thread the flag through to the body
    # parser, which will then expect the canonical ``if this creature
    # is tapped, `` middle clause and AND
    # ``IsSelf() & IsTapped()`` onto the trigger predicate. Mirrors
    # the ``Void —`` strip above structurally — both prefixes attach
    # to a turn-structure trigger and modify how the body parses, but
    # ``Survival`` lands its gate as a predicate (no engine flag) while
    # ``Void`` sets the ``requires_void`` flag.
    is_survival = False
    if (
        at_line.startswith(_SURVIVAL_PREFIX)
        and at_line[len(_SURVIVAL_PREFIX):].startswith("At the beginning of ")
    ):
        at_line = at_line[len(_SURVIVAL_PREFIX):]
        is_survival = True
    at_match = _AT_THE_BEGINNING_OF_LINE.match(at_line)
    if at_match is not None:
        parsed = _parse_at_the_beginning_of_line(
            at_match,
            requires_void=requires_void,
            is_survival=is_survival,
        )
        if parsed is not None:
            return parsed
        return None
    # Intervening-if shape (CR 603.4) tried first: a recognized gate
    # phrase between the trigger condition and the effect collapses to
    # a single ``TriggeredAbility`` with the gate predicate AND-ed onto
    # the event's base predicate. If the gate phrase isn't recognized
    # the line is rejected outright (we don't fall back to the
    # no-intervening-if shape because the surface text would then
    # silently lose its gate, producing an over-firing trigger). This
    # matters: an unrecognized gate is a parser failure to claim the
    # card, not a license to drop the gate.
    iif_match = _TRIGGERED_ABILITY_INTERVENING_IF_LINE.match(line)
    if iif_match is not None:
        condition = iif_match.group(1)
        gate_phrase = iif_match.group(2)
        gate_pred = _parse_intervening_if_phrase(gate_phrase)
        if gate_pred is None:
            return None
        effect = _parse_effect_phrase(iif_match.group(3))
        if effect is None:
            return None
        # Only the predicate-style trigger path supports an AND-able
        # gate predicate; the legacy string-key path can't carry a
        # predicate at all, so a gated trigger must resolve to an event
        # that's already on the predicate side. The two table lookups
        # below preserve the same precedence as the no-gate path; if
        # only the legacy string-key table claims the condition phrase
        # we decline rather than silently drop the gate.
        pred_match = _parse_predicate_trigger_phrase(condition)
        if pred_match is None:
            return None
        event, base_predicate = pred_match
        phrase_once = _predicate_trigger_phrase_is_once_per_turn(condition)
        return TriggeredAbility(
            event=event,
            predicate=base_predicate & gate_pred,
            effect=effect,
            once_per_turn=once_per_turn or phrase_once,
        )
    # Intervening-``while`` shape: ``When/Whenever <condition> while
    # <gate>, <effect>.`` Same trigger-time gate semantics as the
    # middle-clause intervening-if path (CR 603.4) — same predicate
    # AND-ed onto the same base trigger predicate — but the printed
    # surface text uses ``while`` instead of ``, if``. Hand That
    # Feeds is the canonical example. Tried AFTER the ``, if `` regex
    # so a line carrying both shapes (none today) claims via the
    # more-explicit form first; an unrecognized while-gate rejects
    # outright (same don't-silently-drop-the-gate discipline as
    # the ``, if `` path).
    iwhile_match = _TRIGGERED_ABILITY_INTERVENING_WHILE_LINE.match(line)
    if iwhile_match is not None:
        condition = iwhile_match.group(1)
        gate_phrase = iwhile_match.group(2)
        gate_pred = _parse_intervening_while_phrase(gate_phrase)
        if gate_pred is None:
            return None
        effect = _parse_effect_phrase(iwhile_match.group(3))
        if effect is None:
            return None
        pred_match = _parse_predicate_trigger_phrase(condition)
        if pred_match is None:
            return None
        event, base_predicate = pred_match
        phrase_once = _predicate_trigger_phrase_is_once_per_turn(condition)
        return TriggeredAbility(
            event=event,
            predicate=base_predicate & gate_pred,
            effect=effect,
            once_per_turn=once_per_turn or phrase_once,
        )
    # Trailing-if shape: ``When/Whenever <condition>, <effect> if
    # <gate>.`` Mechanically equivalent to the middle-clause
    # intervening-if path above — same predicate AND-ed onto the same
    # base trigger predicate. The Delirium "tail-rider" surface
    # (``... if you have four or more card types among cards in your
    # graveyard.``) is the canonical example. Tried AFTER the middle-
    # clause path so a line carrying both forms claims the more-
    # specific middle shape first; an unrecognized trailing gate
    # rejects the whole line (same don't-silently-drop-the-gate
    # discipline as the middle path).
    tif_match = _TRIGGERED_ABILITY_TRAILING_IF_LINE.match(line)
    if tif_match is not None:
        condition = tif_match.group(1)
        effect_phrase = tif_match.group(2)
        gate_phrase = tif_match.group(3)
        gate_pred = _parse_trailing_if_phrase(gate_phrase)
        if gate_pred is not None:
            effect = _parse_effect_phrase(effect_phrase)
            if effect is None:
                return None
            pred_match = _parse_predicate_trigger_phrase(condition)
            if pred_match is None:
                return None
            event, base_predicate = pred_match
            phrase_once = _predicate_trigger_phrase_is_once_per_turn(
                condition,
            )
            return TriggeredAbility(
                event=event,
                predicate=base_predicate & gate_pred,
                effect=effect,
                once_per_turn=once_per_turn or phrase_once,
            )
        # The trailing-if regex over-matches: any trigger line whose
        # effect contains a sub-clause beginning `` if `` (e.g. an ``if
        # you do, …`` rider on a future printing) will land here even
        # though the trailing-``if`` interpretation isn't intended. If
        # the gate phrase isn't claimed by the trailing-if phrasebook
        # we fall through to the bare ``_TRIGGERED_ABILITY_LINE``
        # parse below rather than rejecting outright — distinct from
        # the middle-clause path's reject-on-unknown-gate discipline,
        # because the middle-clause regex anchors on `` if `` flanked
        # by `` , `` boundaries that are far less likely to appear
        # incidentally inside an effect body.
    m = _TRIGGERED_ABILITY_LINE.match(line)
    if m is None:
        return None
    condition = m.group(1)
    effect = _parse_effect_phrase(m.group(2))
    if effect is None:
        return None
    # Compound predicate-trigger lookup wins when the condition phrase
    # is claimed by the multi-arm table (e.g. ``"this creature enters
    # or dies"`` → two ``(Event, Predicate)`` arms). The single-arm
    # tables below are consulted only if the compound lookup misses,
    # so a future addition that overlaps a compound and a single
    # phrase resolves predictably to the compound shape — matching the
    # surface-text intent ("X or Y" should fire on both events, not
    # silently collapse to one).
    compound_match = _parse_compound_predicate_trigger_phrase(condition)
    if compound_match is not None:
        return tuple(
            TriggeredAbility(event=event, predicate=predicate, effect=effect)
            for event, predicate in compound_match
        )
    # Collapsed-arm predicate-trigger lookup: the condition reads as
    # ``X or Y`` but the two arms share an event AND collapse onto a
    # single predicate (no ``~IsSelf()`` — the wording explicitly
    # includes self). Produces ONE ``TriggeredAbility`` whose
    # predicate covers both arms, mirroring the static registry's
    # single-entry layout for cards like Susurian Voidborn. Tried
    # AFTER the multi-arm compound table (so a phrase that wants
    # two separate abilities still gets them) and BEFORE the single
    # phrase table (so the inclusive wording wins over an
    # accidentally-overlapping single-arm match).
    collapsed_match = _parse_collapsed_predicate_trigger_phrase(condition)
    if collapsed_match is not None:
        event, predicate = collapsed_match
        return TriggeredAbility(
            event=event,
            predicate=predicate,
            effect=effect,
            once_per_turn=once_per_turn,
        )
    # Predicate-style trigger lookup wins when the condition phrase is
    # claimed by both tables — phase 2 only registers ENTERS-family
    # phrases on the predicate side, so today the two tables are
    # disjoint, but precedence here makes the migration order safe to
    # extend in subsequent phases.
    pred_match = _parse_predicate_trigger_phrase(condition)
    if pred_match is not None:
        event, predicate = pred_match
        phrase_once = _predicate_trigger_phrase_is_once_per_turn(condition)
        return TriggeredAbility(
            event=event,
            predicate=predicate,
            effect=effect,
            once_per_turn=once_per_turn or phrase_once,
        )
    trigger = _parse_trigger_phrase(condition)
    if trigger is None:
        return None
    return TriggeredAbility(
        trigger=trigger, effect=effect, once_per_turn=once_per_turn,
    )
