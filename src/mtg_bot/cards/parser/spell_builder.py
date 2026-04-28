"""Instant / sorcery ``CardPrinting`` builder.

Single-line spells are parsed straight via the unified spell-effect
phrase parser. Multi-line spells split on ``\\n`` into clauses and
combine them via one of three combinators — ``ConditionalIfVoid`` for
the Plasma-Bolt-shaped Void rider, ``ModalChooseOne`` for Scryfall's
``Choose one —`` modal shape, otherwise ``Sequence`` if every
clause parses as a plain spell-effect sentence. Any unparseable
clause skips the whole spell.

A leading clause that is just an unimplemented spell-only keyword
plus its parenthesized reminder text (currently only ``Convoke``)
is dropped before the remaining clauses are parsed — the static
side ships these spells without modeling the keyword, so the
parser must do the same to stay in parity. See
``_SPELL_KEYWORD_PREAMBLE_RE`` for the exact recognized shapes.

Cost-reduction rider clauses (``This spell costs {N} less to cast
if <condition>.``) are likewise stripped from the effect-clause
stream and surfaced separately on the ``CardPrinting`` via the
``cost_reductions`` kwarg. The ``_detect_cost_reductions`` helper
scans the original text for those rules; ``_parse_spell_oracle_
text`` drops the matching clauses before combinator dispatch.

The modal branch handles two Scryfall printed header shapes:
``Choose one —`` (single-mode pick) and ``Choose one or both —``
(controller may resolve either mode alone or both). Each is followed
by two or more mode lines that begin with ``• `` (bullet + space)
and contain a single sentence parseable by the unified spell-effect
grammar. The ``Choose one or both`` shape constructs
``ModalChooseOne(or_both=True)`` so the engine routes mode-pick
through the agent's ``choose_modes`` (plural) hook. ``Choose two``,
multi-line modes, and ``up to two target`` mode-counts remain out of
scope; the parser declines those spells until the engine grows the
matching machinery.
"""

from __future__ import annotations

import re

from mtg_bot.cards.base import (
    BiosynthicBurstEffect,
    CardPrinting,
    ConditionalIfVoid,
    CostReduction,
    CutPropulsionEffect,
    DealDamageAnyTarget,
    DealDamageToTargetCreatureSpacecraftBoosted,
    DepressurizeEffect,
    DiplomaticRelationsEffect,
    Effect,
    FocusFireEffect,
    ModalChooseOne,
    MutinousMassacreEffect,
    PumpCreaturesYouControlUntilEOT,
    PumpTargetCreatureUntilEOT,
    Sequence,
    SingularityRuptureEffect,
    make_instant,
    make_sorcery,
)
from mtg_bot.cards.parser.cost_reduction import _parse_cost_reduction_line
from mtg_bot.cards.parser.effects import _parse_spell_effect_phrase
from mtg_bot.cards.parser.text import _strip_reminder_text
from mtg_bot.cards.parser.type_line import _parse_type_line


# Modal spell headers: ``Choose one —`` (single-mode pick) or
# ``Choose one or both —`` (controller may resolve either mode alone
# or both — currently only Scour for Scrap in EoE). Both em-dashes are
# U+2014. ``Choose two`` and other mode-counts remain out of scope.
_MODAL_CHOOSE_ONE_HEADER_RE = re.compile(r"^Choose one —$")
_MODAL_CHOOSE_ONE_OR_BOTH_HEADER_RE = re.compile(r"^Choose one or both —$")
# A mode line begins with U+2022 BULLET + a single space, then the
# mode's effect sentence. Trailing period is preserved here and stripped
# downstream by the spell-effect-phrase normalizer.
_MODAL_MODE_LINE_RE = re.compile(r"^• (.+)$")


# Plasma-Bolt-shaped Void rider: a second clause of the form
# "Void — <Name> deals M damage instead if a nonland permanent left the
# battlefield this turn or a spell was warped this turn."
# Pairs with a preceding "<Name> deals N damage to any target." clause
# to form a ConditionalIfVoid switching the damage amount.
_SPELL_VOID_DAMAGE_REPLACEMENT = (
    r"^Void — {name} deals (\d+) damage instead "
    r"if a nonland permanent left the battlefield this turn "
    r"or a spell was warped this turn\.$"
)


# Tragic-Trajectory-shaped Void rider: a second clause of the form
# "Void — That creature gets ±M/±M' until end of turn instead if a
# nonland permanent left the battlefield this turn or a spell was
# warped this turn."
# Pairs with a preceding "Target creature gets ±N/±N' until end of
# turn." clause to form a ConditionalIfVoid switching the pump
# bonuses. The carry-over subject ``That creature`` refers back to
# the target chosen on the first clause; the engine reuses the
# same target when resolving either branch (single-target spell).
_SPELL_VOID_PUMP_REPLACEMENT = re.compile(
    r"^Void — That creature gets ([+-]\d+)/([+-]\d+) "
    r"until end of turn instead "
    r"if a nonland permanent left the battlefield this turn "
    r"or a spell was warped this turn\.$"
)


# Sentence-boundary splitter for multi-sentence oracle-text lines.
# Some EoE printings pack two effect sentences onto a single oracle
# line separated by ``. `` rather than ``\n`` (e.g. Radiant Strike's
# ``"Destroy target artifact or tapped creature. You gain 3 life."``).
# We carve the line at the period-then-uppercase boundary so each
# sentence flows through the per-clause parser independently.
#
# The negative lookahead for ``If you do,`` keeps the splitter from
# breaking the ``You may pay/sacrifice/discard {COST}. If you do,
# <inner-effect>.`` family of compound primitives — those are matched
# as a single phrase by ``_YOU_MAY_PAY_THEN_EFFECT`` /
# ``_OPTIONAL_DISCARD_THEN_DRAW`` / ``_OPTIONAL_SAC_ANOTHER_FOR_*``
# down in ``effects``, and the embedded sentence boundary inside
# those phrases is required (the regexes match a literal ``\. ``
# between the cost and the consequence).
#
# Decimal numbers and abbreviations don't appear in MTG oracle text,
# so a simple ``period + whitespace + uppercase`` boundary suffices —
# no need for a fuller sentence tokenizer.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=\.)\s+(?=[A-Z])(?!If you do,)")


def _split_sentences(line: str) -> list[str]:
    """Split a single oracle-text line at sentence boundaries.

    Returns ``[line]`` unchanged when the line contains no internal
    boundary. Otherwise returns the list of per-sentence pieces, each
    with surrounding whitespace stripped. Empty pieces are filtered
    out so a stray double-period or trailing whitespace doesn't
    inject blank clauses.

    The boundary is ``period + whitespace + uppercase letter``, with
    a negative lookahead for ``If you do,`` so the
    ``You may pay/sacrifice/discard {COST}. If you do, …`` compound
    primitives don't get split — those are matched as a single phrase
    by the effect grammar and the embedded sentence boundary inside
    them is load-bearing.
    """
    pieces = [p.strip() for p in _SENTENCE_BOUNDARY_RE.split(line)]
    return [p for p in pieces if p]


# Spell-only keyword preamble: a leading clause that's just a static
# keyword followed by its parenthesized reminder text. The keyword
# line itself contributes no ``Effect`` — its behavior is wired via
# a flag on ``CardPrinting`` (e.g. Convoke → ``has_convoke=True``).
# Stripping this preamble lets the remaining clauses parse as a
# single-effect or Sequence spell.
#
# Some keywords here may be intentionally unimplemented (the engine
# treats the spell as if the keyword weren't there). Add a name only
# after confirming the matching ``cards/eoe_spells.py`` entry sets
# the corresponding printing flag — otherwise dropping the preamble
# would silently diverge from the static implementation.
_SPELL_PREAMBLE_KEYWORDS = ("Convoke",)
_SPELL_KEYWORD_PREAMBLE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(k) for k in _SPELL_PREAMBLE_KEYWORDS)
    + r") \(.+\)\.?$"
)


# Additional-cost preamble: a leading clause of the form
# ``As an additional cost to cast this spell, <cost-clause>.``. The
# preamble itself contributes no ``Effect`` — its behavior is wired via
# a flag on ``CardPrinting`` (e.g. ``additional_cost_sacrifice_creature_
# or_artifact=True`` for Embrace Oblivion / Scrounge for Eternity).
# Stripping this preamble lets the remaining clauses parse as a
# single-effect or Sequence spell.
#
# The cost-clause body is matched against a small whitelist of
# recognized cost shapes; each maps to a specific printing-flag dict.
# Unrecognized cost clauses leave the spell unclaimed (the preamble
# isn't dropped) so the parser doesn't silently diverge from the
# static implementation.
_ADDITIONAL_COST_PREAMBLE_RE = re.compile(
    r"^As an additional cost to cast this spell, (.+)\.$"
)
# Recognized additional-cost bodies → printing flags. Keys are the
# captured ``(.+)`` group (without trailing period); values are the
# dict of flags applied to the resulting ``CardPrinting`` via
# ``make_instant`` / ``make_sorcery``.
_ADDITIONAL_COST_BODY_FLAGS: dict[str, dict] = {
    "sacrifice an artifact or creature": {
        "additional_cost_sacrifice_creature_or_artifact": True,
    },
}


def _is_additional_cost_preamble(clause: str) -> bool:
    """Return ``True`` if ``clause`` is a recognized additional-cost
    preamble (``As an additional cost to cast this spell, <cost>.``
    where ``<cost>`` is one of the whitelisted cost bodies). The
    caller drops these preamble clauses before parsing the rest of
    the spell. Unrecognized cost bodies return ``False`` so the
    spell stays unclaimed rather than silently dropping the cost.
    """
    m = _ADDITIONAL_COST_PREAMBLE_RE.match(clause.strip())
    if m is None:
        return False
    return m.group(1) in _ADDITIONAL_COST_BODY_FLAGS


def _additional_cost_flags(clause: str) -> dict:
    """Return the printing-flag dict for a recognized additional-cost
    preamble ``clause``. Returns an empty dict if the clause isn't a
    recognized preamble (caller should ``_is_additional_cost_preamble``
    first to gate the call). Keeps the per-shape flag mapping
    centralized in ``_ADDITIONAL_COST_BODY_FLAGS`` so adding a new
    cost shape is a one-line entry there plus a one-line dispatch
    here is the only edit needed.
    """
    m = _ADDITIONAL_COST_PREAMBLE_RE.match(clause.strip())
    if m is None:
        return {}
    return _ADDITIONAL_COST_BODY_FLAGS.get(m.group(1), {})


def _is_spell_keyword_preamble(clause: str) -> bool:
    """Return ``True`` if ``clause`` is a recognized unimplemented
    spell-keyword preamble (keyword name + parenthesized reminder
    text). The caller drops these preamble clauses before parsing the
    rest of the spell. See ``_SPELL_PREAMBLE_KEYWORDS`` for the
    whitelist."""
    return _SPELL_KEYWORD_PREAMBLE_RE.match(clause.strip()) is not None


def _parse_void_damage_replacement(
    phrase: str, name: str, base_effect: Effect
) -> Effect | None:
    """If ``phrase`` is a Void rider replacing the damage amount of
    ``base_effect``, return the void-branch ``Effect`` (same shape as
    base, new amount). Returns ``None`` otherwise.

    Currently only handles ``DealDamageAnyTarget`` as the base — the
    only damage shape that pairs with a Void rider in the EoE pool we
    support today (Plasma Bolt). Extend per-card as new pairings show up.
    """
    if not isinstance(base_effect, DealDamageAnyTarget):
        return None
    name_re = re.escape(name)
    m = re.match(
        _SPELL_VOID_DAMAGE_REPLACEMENT.format(name=name_re), phrase.strip()
    )
    if m is None:
        return None
    return DealDamageAnyTarget(amount=int(m.group(1)))


def _parse_void_pump_replacement(
    phrase: str, base_effect: Effect
) -> Effect | None:
    """If ``phrase`` is a Void rider replacing the pump bonuses of
    ``base_effect``, return the void-branch ``Effect`` (same shape as
    base, new bonuses). Returns ``None`` otherwise.

    Currently only handles ``PumpTargetCreatureUntilEOT`` as the base —
    the only pump shape that pairs with a Void rider in the EoE pool
    we support today (Tragic Trajectory). The carry-over subject
    ``That creature`` refers back to the target chosen on the first
    clause; the engine reuses the same target when resolving either
    branch (single-target spell).

    Mirrors the structure of ``_parse_void_damage_replacement`` —
    keeping the two as separate helpers (vs one polymorphic function)
    keeps the per-shape regex literal explicit and the per-shape
    engine class wiring obvious. Extend per-card as new pairings
    surface; e.g. a future Void rider replacing
    ``DealDamageToTargetCreature`` would land as a third helper.

    The ``base_effect``'s bonuses are not used to compute the void
    branch — only the type is matched. The void branch's bonuses
    come entirely from the rider clause itself.
    """
    if not isinstance(base_effect, PumpTargetCreatureUntilEOT):
        return None
    m = _SPELL_VOID_PUMP_REPLACEMENT.match(phrase.strip())
    if m is None:
        return None
    return PumpTargetCreatureUntilEOT(
        power_bonus=int(m.group(1)),
        toughness_bonus=int(m.group(2)),
    )


def _parse_modal_choose_one(
    clauses: list[str], name: str
) -> Effect | None:
    """If ``clauses`` is a ``Choose one —`` or ``Choose one or both —``
    modal-spell shape, return the corresponding ``ModalChooseOne``
    effect; otherwise ``None``.

    Recognized shapes (from Scryfall printed text):

      ``Choose one —``               (or ``Choose one or both —``)
      ``• <mode 1 sentence>``
      ``• <mode 2 sentence>``
      ``• <mode 3 sentence>``  (optional)
      ...

    Each mode line is parsed by ``_parse_spell_effect_phrase`` after
    stripping the leading ``• `` (bullet + space). If any mode fails
    to parse, the whole shape is rejected (returns ``None``) — partial
    modal effects would silently change a spell's playable behavior.

    A header by itself (zero mode lines) or a header with exactly one
    mode line is rejected — printed modal spells always have at least
    two modes, and a single-mode "modal" spell would be indistinguishable
    from a non-modal effect. ``Choose two`` and other variants don't
    match either header regex and fall through. The ``or both`` shape
    constructs ``ModalChooseOne(or_both=True)`` so the engine routes
    mode-pick through the agent's ``choose_modes`` (plural) hook.
    """
    if not clauses:
        return None
    if _MODAL_CHOOSE_ONE_HEADER_RE.match(clauses[0]) is not None:
        or_both = False
    elif _MODAL_CHOOSE_ONE_OR_BOTH_HEADER_RE.match(clauses[0]) is not None:
        or_both = True
    else:
        return None
    mode_lines = clauses[1:]
    if len(mode_lines) < 2:
        return None
    parsed_modes: list[Effect] = []
    for line in mode_lines:
        m = _MODAL_MODE_LINE_RE.match(line)
        if m is None:
            return None
        eff = _parse_spell_effect_phrase(m.group(1), name)
        if eff is None:
            return None
        parsed_modes.append(eff)
    return ModalChooseOne(modes=tuple(parsed_modes), or_both=or_both)


# Whole-spell exact-text dispatch table. Each key is the full oracle
# text after reminder-text stripping (parenthesized reminder tail
# removed) but BEFORE sentence splitting — i.e. exactly what a single
# oracle-text line looks like once parentheticals are gone. The value
# is the engine-class instance that the static registry uses for that
# card.
#
# This tier exists for spells whose oracle text is a single sentence-
# block that the static registry models with a one-off engine class
# (no reusable primitive is meaningful). The per-clause grammar
# couldn't claim these because at least one of their internal
# sentences (e.g. ``If that creature has flying, it deals twice that
# much damage to itself instead.``) doesn't parse as a standalone
# effect — and matching the per-sentence grammar to those forms
# would yield primitives no other card uses.
#
# Entries here must be exact verbatim oracle text (post reminder
# stripping). A future printing whose oracle text differs by even
# one character won't match — by design, since this tier is for
# named one-offs, not a reusable shape.
_WHOLE_SPELL_EXACT_TEXT: dict[str, Effect] = {
    # Cut Propulsion: target creature self-damages by its power; if
    # it has flying, it self-damages twice that amount instead.
    (
        "Target creature deals damage to itself equal to its power. "
        "If that creature has flying, "
        "it deals twice that much damage to itself instead."
    ): CutPropulsionEffect(),
    # Mutinous Massacre: parity-of-mana-value wrath + steal-and-haste
    # all-creatures rider. Reminder tail ``(Zero is even.)`` is
    # stripped before lookup.
    (
        "Choose odd or even. "
        "Destroy each creature with mana value of the chosen quality. "
        "Then gain control of all creatures until end of turn. "
        "Untap them. They gain haste until end of turn."
    ): MutinousMassacreEffect(),
    # Singularity Rupture: wrath + opt-in mass mill of half-library.
    (
        "Destroy all creatures, then any number of target players "
        "each mill half their library, rounded down."
    ): SingularityRuptureEffect(),
    # Depressurize: -3/-0 UEOT + state-based death check ("if power is 0
    # or less, destroy it"). The engine class folds the pump and the
    # conditional destroy into one resolve step, so the parser maps the
    # full two-sentence text to the single bespoke class.
    (
        "Target creature gets -3/-0 until end of turn. "
        "Then if that creature's power is 0 or less, destroy it."
    ): DepressurizeEffect(power_penalty=3),
    # Focus Fire: variable-X burn whose X reads "2 plus the number of
    # creatures and/or Spacecraft you control". The bespoke engine
    # class computes X at resolve time; the parser dispatches on the
    # printed surface form.
    (
        "Focus Fire deals X damage to target attacking or blocking "
        "creature, where X is 2 plus the number of creatures and/or "
        "Spacecraft you control."
    ): FocusFireEffect(base=2),
    # Diplomatic Relations: pump-and-grant-vigilance on your creature
    # plus a fight-shaped damage rider where the chosen first target
    # deals damage equal to its (post-pump) power to a chosen second
    # target an opponent controls. Two-target spell — the bespoke
    # class threads both targets through one resolve step.
    (
        "Target creature you control gets +1/+0 and gains vigilance "
        "until end of turn. It deals damage equal to its power to "
        "target creature an opponent controls."
    ): DiplomaticRelationsEffect(),
    # Biosynthic Burst: counter + grant-three-keywords + untap on a
    # creature you control. The static class wires the +1/+1 counter,
    # the reach/trample/indestructible UEOT grant, and the untap into
    # one resolve.
    (
        "Put a +1/+1 counter on target creature you control. "
        "It gains reach, trample, and indestructible until end of "
        "turn. Untap it."
    ): BiosynthicBurstEffect(),
    # Zealous Display: anthem + opp-turn-untap rider. The
    # ``also_untap_if_not_your_turn`` flag on the engine class fires
    # only when the spell resolves on an opponent's turn, untapping
    # every creature that received the pump.
    (
        "Creatures you control get +2/+0 until end of turn. "
        "If it's not your turn, untap those creatures."
    ): PumpCreaturesYouControlUntilEOT(
        power_bonus=2,
        toughness_bonus=0,
        also_untap_if_not_your_turn=True,
    ),
    # Dual-Sun Technique: grant double strike + conditional draw rider
    # ("If it has a +1/+1 counter on it, draw a card"). The engine
    # class's ``draw_if_target_has_plus_one_counter`` field captures
    # the rider; the keyword grant rides the standard pump path.
    (
        "Target creature you control gains double strike until end of "
        "turn. If it has a +1/+1 counter on it, draw a card."
    ): PumpTargetCreatureUntilEOT(
        power_bonus=0,
        toughness_bonus=0,
        granted_keywords=frozenset({"double strike"}),
        you_control_only=True,
        draw_if_target_has_plus_one_counter=1,
    ),
    # Invasive Maneuvers: bare burn with a Spacecraft-state boost
    # ("It deals 5 damage instead if you control a Spacecraft").
    # The bespoke class picks ``base`` vs ``boosted`` by inspecting
    # the controller's battlefield at resolve time.
    (
        "Invasive Maneuvers deals 3 damage to target creature. "
        "It deals 5 damage instead if you control a Spacecraft."
    ): DealDamageToTargetCreatureSpacecraftBoosted(base=3, boosted=5),
}


def _try_whole_spell_exact_text(text: str) -> Effect | None:
    """If ``text`` (post reminder-text stripping) exactly matches a
    whole-spell entry in ``_WHOLE_SPELL_EXACT_TEXT``, return the
    associated engine-class instance; otherwise ``None``.

    Tried at the top of ``_parse_spell_oracle_text`` BEFORE the
    per-clause grammar runs. Each key is exact-match, so the
    dispatch is O(1) per spell — no risk of accidentally catching
    spells whose text overlaps a key.
    """
    return _WHOLE_SPELL_EXACT_TEXT.get(text.strip())


def _parse_spell_oracle_text(text: str, name: str) -> Effect | None:
    """Parse a spell's full oracle text into a single ``Effect``.

    Single-line: parsed by ``_parse_spell_effect_phrase`` directly.
    Multi-line: each ``\\n``-separated clause is parsed; clauses are
    combined either via ``ConditionalIfVoid`` (when the second clause
    is a Void replacement of the first damage effect), via
    ``ModalChooseOne`` (Scryfall ``Choose one —`` or
    ``Choose one or both —`` shape), or via ``Sequence`` (every clause
    is a plain spell-effect sentence).
    Leading clauses that are unimplemented spell-keyword preambles
    (e.g. Vote Out's Convoke) are dropped before combinator dispatch
    so they don't fail the per-clause parse. ``This spell costs {N}
    less to cast if <condition>.`` cost-reduction riders are likewise
    dropped here — the rule is captured separately by
    ``_detect_cost_reductions`` and threaded onto the resulting
    ``CardPrinting`` via the ``cost_reductions`` kwarg.

    A single oracle-text line that packs two effect sentences
    separated by ``. `` (Radiant Strike's
    ``"Destroy target artifact or tapped creature. You gain 3 life."``,
    Sami's Curiosity's ``"You gain 2 life. Create a Lander token."``)
    is further carved at the sentence boundary by ``_split_sentences``
    after reminder-text stripping and cost-reduction dropping. Each
    resulting sentence is then a clause in its own right — the
    Sequence combinator path picks them up the same way it would for
    a printing that put each sentence on its own ``\\n``-separated
    line. Returns ``None`` if any surviving clause fails to parse.
    """
    # Whole-spell exact-text dispatch: try a one-off engine-class
    # match against the full reminder-stripped oracle text BEFORE
    # any clause-level processing. Reserved for cards whose oracle
    # text is a single sentence-block modeled by a dedicated engine
    # class with no parameters — see ``_WHOLE_SPELL_EXACT_TEXT`` for
    # the full table.
    stripped_full = _strip_reminder_text(text)
    if stripped_full:
        whole = _try_whole_spell_exact_text(stripped_full)
        if whole is not None:
            return whole
    clauses = [c.strip() for c in text.split("\n") if c.strip()]
    # Drop leading spell-keyword preambles BEFORE per-clause reminder
    # stripping. The preamble shape (``"Convoke (...)"``) is recognized
    # via the trailing parenthesized reminder, so we have to detect
    # those preamble clauses while their parentheses are still intact.
    while clauses and _is_spell_keyword_preamble(clauses[0]):
        clauses = clauses[1:]
    # Drop leading recognized additional-cost preambles. The preamble
    # itself contributes no ``Effect`` — the corresponding printing
    # flag is threaded onto the resulting ``CardPrinting`` separately
    # via ``_detect_additional_cost_flags`` (called from
    # ``_build_spell_printing``). Unrecognized cost bodies leave the
    # preamble in place; the spell will fail to parse downstream and
    # stay unclaimed rather than silently dropping the cost.
    while clauses and _is_additional_cost_preamble(clauses[0]):
        clauses = clauses[1:]
    # Now drop per-clause parenthesized reminder text. Empty results
    # (a clause that was nothing but a parenthetical) are filtered out.
    clauses = [
        stripped
        for stripped in (_strip_reminder_text(c) for c in clauses)
        if stripped
    ]
    # Drop recognized cost-reduction rider clauses — they don't
    # contribute an ``Effect`` (the engine consumes them at cast time
    # via ``CostReduction`` rules on the printing). ``_build_spell_
    # printing`` calls ``_detect_cost_reductions`` separately to thread
    # those rules onto the resulting ``CardPrinting``.
    clauses = [
        c for c in clauses if _parse_cost_reduction_line(c) is None
    ]
    # Carve any line that packs two effect sentences on a single
    # ``. ``-separated line into independent clauses. Done AFTER
    # reminder-text stripping (so quoted/parenthesized inner periods
    # don't fire) and AFTER cost-reduction dropping (so a multi-
    # sentence rider — none ship today, but defensive — never gets
    # half-dropped). Single-sentence clauses round-trip unchanged.
    clauses = [s for c in clauses for s in _split_sentences(c)]
    if not clauses:
        return None
    if len(clauses) == 1:
        return _parse_spell_effect_phrase(clauses[0], name)
    # Two-clause Void-replacement shape (Plasma Bolt's damage rider,
    # Tragic Trajectory's pump rider). The dispatch tries each
    # supported base/rider pairing in turn and picks the first that
    # both parses the base-clause cleanly AND matches the rider's
    # surface form for that base shape.
    if len(clauses) == 2:
        first = _parse_spell_effect_phrase(clauses[0], name)
        if first is not None:
            void_then = _parse_void_damage_replacement(
                clauses[1], name, first
            )
            if void_then is None:
                void_then = _parse_void_pump_replacement(clauses[1], first)
            if void_then is not None:
                return ConditionalIfVoid(
                    then_effect=void_then, else_effect=first
                )
    # Modal shape: ``Choose one —`` / ``Choose one or both —`` header
    # line + ≥2 bullet mode lines. Tried before the Sequence fallback
    # because Sequence would uselessly attempt to parse the modal
    # header as a spell-effect sentence, fail, and return None —
    # losing any modal spell whose modes are individually parseable.
    modal = _parse_modal_choose_one(clauses, name)
    if modal is not None:
        return modal
    # Sequence shape: every clause must be a plain spell-effect sentence.
    parsed: list[Effect] = []
    for clause in clauses:
        eff = _parse_spell_effect_phrase(clause, name)
        if eff is None:
            return None
        parsed.append(eff)
    return Sequence(effects=tuple(parsed))


def _detect_preamble_flags(text: str) -> dict:
    """Scan ``text``'s leading keyword-preamble clauses and return a
    dict of ``CardPrinting`` flags they imply (e.g.
    ``{"has_convoke": True}`` when "Convoke (...)" leads the oracle
    text). Mirrors the preamble-stripping in
    ``_parse_spell_oracle_text``."""
    flags: dict = {}
    for clause in [c.strip() for c in text.split("\n") if c.strip()]:
        if not _is_spell_keyword_preamble(clause):
            break
        if clause.startswith("Convoke"):
            flags["has_convoke"] = True
    return flags


def _detect_additional_cost_flags(text: str) -> dict:
    """Scan ``text``'s leading additional-cost preamble clauses and
    return a dict of ``CardPrinting`` flags they imply (e.g.
    ``{"additional_cost_sacrifice_creature_or_artifact": True}`` when
    ``"As an additional cost to cast this spell, sacrifice an artifact
    or creature."`` leads the oracle text).

    Mirrors the preamble-stripping in ``_parse_spell_oracle_text`` —
    iterates through leading clauses while each is a recognized
    additional-cost preamble, accumulates the flags, and stops at the
    first non-preamble clause. Multiple stacked additional-cost
    preambles aren't a printed shape today, but the loop handles them
    naturally if a future card ships them.

    Walks past keyword-preamble clauses (the existing Convoke shape)
    in case a future card stacks both shapes (Convoke + additional
    cost). Today no card does, but skipping them keeps the dispatcher
    order-insensitive: the flag set returned doesn't depend on
    whether the spell author put the additional-cost line first or
    after the keyword preamble.
    """
    flags: dict = {}
    for clause in [c.strip() for c in text.split("\n") if c.strip()]:
        if _is_spell_keyword_preamble(clause):
            continue
        if not _is_additional_cost_preamble(clause):
            break
        flags.update(_additional_cost_flags(clause))
    return flags


def _detect_cost_reductions(text: str) -> tuple[CostReduction, ...]:
    """Return every cost-reduction rule implied by ``text``'s recognized
    ``This spell costs {N} less to cast if <condition>.`` clauses, in
    the order they appear. Reminder text is stripped per clause
    before recognition, so a rider buried after a parenthesized
    flavor still matches.

    Mirrors the rider-dropping pass in ``_parse_spell_oracle_text``;
    callers pass the result via the ``cost_reductions`` kwarg on
    ``make_instant`` / ``make_sorcery``.
    """
    rules: list[CostReduction] = []
    for raw in text.split("\n"):
        clause = _strip_reminder_text(raw)
        if not clause:
            continue
        rule = _parse_cost_reduction_line(clause)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def _build_spell_printing(card: dict) -> CardPrinting | None:
    """Try to produce a ``CardPrinting`` for an instant or sorcery
    ``card`` (a Scryfall record). Returns ``None`` if the card is
    neither an instant nor a sorcery, or if its oracle text doesn't
    parse into a supported single-effect or composite shape.
    """
    supertypes, card_types, subtypes = _parse_type_line(
        card.get("type_line", "")
    )
    if "Instant" not in card_types and "Sorcery" not in card_types:
        return None
    text = card.get("oracle_text", "").strip()
    if not text:
        return None
    effect = _parse_spell_oracle_text(text, card["name"])
    if effect is None:
        return None
    factory = make_instant if "Instant" in card_types else make_sorcery
    return factory(
        name=card["name"],
        cost=card["mana_cost"],
        effect=effect,
        cost_reductions=_detect_cost_reductions(text),
        **_detect_preamble_flags(text),
        **_detect_additional_cost_flags(text),
    )
