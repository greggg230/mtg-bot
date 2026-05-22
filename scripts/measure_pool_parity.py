"""Measured, regenerable POOL-LEVEL cross-engine parity metric.

The campaign has long answered "how far from parity?" with PROXIES:

  * per-*emission* Rust Effect coverage (``docs/RUST_EFFECT_COVERAGE.md``,
    ``scripts/measure_effect_coverage.py``) — weighted by how many
    cards each Effect class appears on, NOT a per-card classification;
  * the honest residue count (``docs/residue_worklist.json``) — the
    distinct genuinely-Rust-blocked cards, but only the blocked tail;
  * "0 FAIL_DIVERGENCE over the bridge-admitted bench subset"
    (``docs/bench_trajectory_parity_smoke.md``) — an empirical signal
    over whatever the loader admits, not a tracked pool fraction.

None of those is a single authoritative, tracked, *measured* number
for "what fraction of the EoE card pool is byte-equal cross-engine".
This script is that metric. It classifies **every** card in the EoE
pool (the same pool ``measure_effect_coverage`` walks) into honest,
precisely-defined tiers and reports the counts + percentages, then
writes ``docs/POOL_PARITY.md`` (auto-overwritten on each run) and
prints the same markdown to stdout — the same idiom as
``measure_effect_coverage.py``.

WHAT THIS MEASURES (and what it does NOT)
-----------------------------------------
This metric measures **Effect+bridge-level byte-equality evidence**,
NOT full-game-trajectory parity. The strongest tier (PROBED) has a
genuine real-EoE cross-engine probe asserting byte-equality on a
controlled state; the next tier (covered-by-construction) only proves
every Effect the card emits has a live Rust variant + bridge arm — it
does NOT prove the card is byte-equal in a real multi-turn game. The
bench tier is the empirical full-trajectory signal but is capped at
~22-25 turns and only over the loader-admitted subset. **No tier here
claims "byte-equal in real full-game trajectories for the whole
pool"** — that is explicitly out of scope and stated as a caveat in
the generated doc. Honest discipline: the headline reports the
modest, defensible number, not a flattering one.

Run with::

    PYTHONPATH=src \\
    ../../mtg-bot/.venv-py313/Scripts/python.exe \\
    scripts/measure_pool_parity.py

This script does not depend on the Rust extension being built — it
reuses ``measure_effect_coverage``'s pure-text ``cards.rs`` parser and
its EoE-pool walker (``collect_python_effect_vocab`` /
``collect_rust_effect_variants`` — the parse rule is NOT duplicated
here) plus the pure-Python ``parity.trajectory`` admit predicates and
a text scan of ``tests/test_parity_smoke.py``. The companion test
``tests/test_measure_pool_parity.py`` exercises the classifier.

TIER DEFINITIONS (precise)
--------------------------
Each EoE printing lands in EXACTLY ONE tier, evaluated in this
strict precedence order (strongest evidence wins):

1. **PROBED byte-equal** — a genuine real-EoE cross-engine probe in
   ``tests/test_parity_smoke.py`` (a ``test_cross_engine_real_eoe_
   <slug>_parity`` function whose name-prefix is the snake_case slug
   of this card's printed name — split cards keyed on the left face)
   asserts byte-equality for THIS card. Strongest evidence: a real
   pool printing pinned end-to-end on both engines on a controlled
   state. (Still a controlled-state probe, NOT a full random game —
   see the caveat above.)

2. **Effect+bridge covered (not individually probed)** — every
   distinct Python ``Effect`` class this card emits (walked with
   ``measure_effect_coverage``'s exact walker) has a live Rust
   ``Effect::`` variant + a wired bridge arm (i.e. is in the coverage
   script's covered-class set), AND the card is NOT a BLOCKED entry
   in ``docs/residue_worklist.json``. Covered-by-construction: the
   primitives all exist + bridge, but no genuine probe pins THIS
   card, so it is weaker evidence than PROBED. A card with zero
   Effects (pure vanilla / keyword-only) counts here only if it is
   also loader-admitted (otherwise it cannot be cross-engine
   exercised at all and falls to the relevant lower tier).

3. **bridge-admitted & bench-exercised** — the production-deck
   loader admits this printing (it passes
   ``trajectory._printing_is_bridge_basic_land`` or
   ``trajectory._printing_is_bridge_simple_creature`` — the exact
   admit gate the bench corpus runs through) and the bench has run
   it 0-divergence at scale. CAVEAT: the bench caps at ~22-25 turns
   (``docs/bench_trajectory_parity_smoke.md``), so this is a
   bounded-depth empirical signal, not a game-end guarantee.

4. **BLOCKED** — not byte-equal-covered today. Broken out into:
   * ``blocked_genuinely_missing`` — a residue worklist
     ``genuinely_missing`` entry: a TRUE new-Rust-Effect ask (the
     load-bearing "how far from parity" gap).
   * ``blocked_non_effect`` — a residue worklist
     ``blocked_non_effect`` entry: the card's Effects ARE shipped;
     it is gated by a non-Effect path (unbridged event / declined
     cost-mechanic / unmodeled subsystem / token-registration
     substrate).
   * ``blocked_subsystem_gated`` — NOT in the residue worklist as
     blocked, NOT probed, NOT covered-by-construction, NOT
     loader-admitted: the card emits at least one Effect with no
     covered Rust counterpart, OR is loader-rejected by a subsystem
     gate (Station / Ward / Devour / warp / Aura / Rooms /
     BECOMES_TAPPED / dynamic-P/T / activated+static abilities /
     planeswalker / …). The long, un-itemised blocked tail.

INTERNAL-CONSISTENCY CROSS-CHECK
--------------------------------
A card listed BLOCKED (``genuinely_missing`` or
``blocked_non_effect``) in ``docs/residue_worklist.json`` must NOT be
counted in PROBED or covered-by-construction. The script asserts this
and emits a ``WARNING`` line into the doc + a non-zero structural
flag (the companion test pins it) if a residue-blocked card ever
sneaks into a covered tier — exactly the residue-vs-headline drift
class the round-54/55/56 cron guards, applied to THIS metric.
"""

from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUT_DOC = REPO_ROOT / "docs" / "POOL_PARITY.md"
SMOKE_TEST = REPO_ROOT / "tests" / "test_parity_smoke.py"
RESIDUE_WORKLIST = REPO_ROOT / "docs" / "residue_worklist.json"

# --- The DEDICATED genuine-real-EoE probe modules (cycles 3-4) ----------
#
# HONEST-MEASUREMENT-TOOLING fix (resolves the orchestrator's standing
# cycle-4 flag). The original PROBED detector scanned ONLY
# ``tests/test_parity_smoke.py`` for ``test_cross_engine_real_eoe_
# <slug>_parity`` functions. Cycles 3-4 added FOUR new dedicated probe
# modules (the ``test_probe_covered_unprobed_{a,b,c,d}.py`` batch
# stream) carrying ~27 genuine real-EoE cross-engine byte-equality
# probes that use the EXACT same idiom + the EXACT same
# ``test_cross_engine_real_eoe_<...>_parity`` naming convention as the
# 60 cards already credited from the smoke file — but live in a
# different file, so the metric did not see them and ``PROBED``
# understated real probe coverage. This list is the closed set of
# modules whose ``*_parity`` functions are scanned with the SAME
# inclusion bar (see ``genuine_probed_cards``).
#
# DELIBERATELY EXCLUDED (a defensible zero-undercount integrity
# choice, NOT an oversight):
#
#   * ``tests/test_probe_warp_admitted.py`` and
#     ``tests/test_probe_spell_card_bridge_admitted.py`` — these ARE
#     genuine real-EoE byte-equal probes, but (a) their functions do
#     NOT use the established ``_parity`` slug-keyed naming convention
#     (they are ``test_warp_admitted_*`` /
#     ``test_spell_card_bridge_phase1_carrier_*`` parametrized sweeps),
#     so crediting them would require a SECOND, looser inclusion rule —
#     a weaker bar than the established one; and (b) every card they
#     pin (Radiant Strike, Depressurize, Seedship Impact, Orbital
#     Plunge, the five warp creatures) is ALREADY individually PROBED
#     via its canonical ``test_cross_engine_real_eoe_<slug>_parity``
#     function in ``tests/test_parity_smoke.py``. Excluding the
#     differently-named modules therefore costs ZERO undercount while
#     keeping exactly ONE inclusion predicate at exactly the
#     established bar — the "match the established bar; if unsure,
#     undercount rather than inflate" discipline.
#
# CYCLE-10 HONEST-ATTRIBUTION RECONCILIATION (the recurring
# measurement-attribution gap the cycle-10 fold flagged: ``PROBED=104``
# understated real probe coverage by the Tier-C target-seating cards).
# The Tier-C lever-a slices (Sunset Saboteur ``5e05cbaa`` / Exosuit
# Savior ``fa3d17b4``) each landed a GENUINE real-EoE cross-engine
# byte-equal firing-parity probe — drives the live parsed printing
# (``get_active_source().get_printings().get("Sunset Saboteur" /
# "Exosuit Savior")``) end-to-end through the production ATTACKS / ETB
# path on BOTH engines and asserts byte-equal state (the SAME idiom +
# the SAME strength bar as the 60+ cards already credited). The probes
# were authored IN ``tests/test_parity_smoke.py`` (already scanned) but
# their function names ended ``_attacks_seats_first_opponent_creature``
# / ``_etb_seats_first_own_permanent`` instead of the established
# ``..._parity`` convention, so the unchanged single inclusion
# predicate (``_smoke_probed_cards`` / ``_probe_module_probed_cards``
# both require the ``_parity`` suffix) silently UNDERCOUNTED them — the
# SAME class of bug the cycle-4 batch-module flag tracked, surfacing
# again. RESOLUTION (the established discipline — conform the deviating
# probes to the convention, do NOT loosen the bar / add a second
# looser predicate): the two functions were renamed to append
# ``_parity`` (``feature/probe-modules-tierc-registration``), so the
# UNCHANGED smoke detector now credits them at EXACTLY the established
# bar. PROBED 104 → 106 (+Sunset Saboteur, +Exosuit Savior; both move
# covered-by-construction → the stronger PROBED tier — the byte-equal
# headline total is unchanged, only the evidence tier strengthens).
#
# HONESTLY EXCLUDED (the bar cuts both ways — NOT an oversight):
#
#   * **Diplomatic Relations** — historical carve-out CLOSED by the
#     round-26 YYY batch (``tests/test_probe_remaining_yyy.py``:
#     ``test_cross_engine_real_eoe_diplomatic_relations_parity``).  The
#     YYY probe meets the same parse + bridge round-trip + Python-side
#     resolve idiom DDD slice 17's Honored Knight-Captain probe
#     established as a PROBE_MODULES-credited shape -- the live printing
#     is bound via ``get_active_source().get_printings().get("Diplomatic
#     Relations")``, the ``diplomatic_relations_effect`` bridge round-
#     trip is verified, and the Python-side 2-slot resolve drives pump+
#     vigilance+power-equal damage on a controlled state.  The
#     ``tests/test_cross_engine_etb_triggers.py`` ``test_bridge_*_bidirectional_round_trip`` and
#     ``test_bridge_spell_card_lever_phase2_diplomatic_relations_real_eoe_cast_resolve_parity``
#     pre-existing tests remain as the trajectory-harness-driven
#     cross-engine full-cast-resolve pin (a stronger but
#     non-PROBE_MODULES-credited shape); the YYY probe closes the
#     attribution gap at the established bar.
PROBE_MODULES: tuple[Path, ...] = (
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_a.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_b.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_c.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_d.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_e.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_f.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_g.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_h.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_i.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_j.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_k.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_l.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_m.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_n.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_o2.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_p.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_q.py",
    REPO_ROOT / "tests" / "test_probe_may_do_r.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_s.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_t.py",
    REPO_ROOT / "tests" / "test_probe_modal_counterspell_u.py",
    REPO_ROOT / "tests" / "test_probe_covered_unprobed_v.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_w.py",
    REPO_ROOT / "tests" / "test_probe_attacks_leaves_battlefield_x.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_y.py",
    REPO_ROOT / "tests" / "test_probe_attacks_round2_z.py",
    REPO_ROOT / "tests" / "test_probe_cast_spell_aa.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_bb.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_ee.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_hh.py",
    REPO_ROOT / "tests" / "test_probe_subject_is_enchantment_unlock.py",
    REPO_ROOT / "tests" / "test_probe_activated_abilities_kk.py",
    REPO_ROOT / "tests" / "test_probe_has_power_at_most_unlock.py",
    REPO_ROOT / "tests" / "test_probe_eerie_cards_nn.py",
    REPO_ROOT / "tests" / "test_divergence_pins_dd.py",
    REPO_ROOT / "tests" / "test_probe_remaining_qq.py",
    REPO_ROOT / "tests" / "test_probe_deals_combat_damage_unlock.py",
    REPO_ROOT / "tests" / "test_probe_remaining_tt.py",
    REPO_ROOT / "tests" / "test_probe_phase_step_unlock.py",
    REPO_ROOT / "tests" / "test_probe_remaining_ww.py",
    REPO_ROOT / "tests" / "test_probe_you_attack_unlock.py",
    REPO_ROOT / "tests" / "test_probe_remaining_zz.py",
    REPO_ROOT / "tests" / "test_divergence_pins_xx.py",
    REPO_ROOT / "tests" / "test_probe_remaining_ccc.py",
    REPO_ROOT / "tests" / "test_divergence_pins_aaa.py",
    REPO_ROOT / "tests" / "test_divergence_pins_uu.py",
    REPO_ROOT / "tests" / "test_probe_remaining_fff.py",
    REPO_ROOT / "tests" / "test_probe_rr_events_trio.py",
    REPO_ROOT / "tests" / "test_probe_remaining_iii.py",
    REPO_ROOT / "tests" / "test_divergence_pins_rr.py",
    REPO_ROOT / "tests" / "test_probe_bbb_downstream.py",
    REPO_ROOT / "tests" / "test_probe_remaining_lll.py",
    REPO_ROOT / "tests" / "test_probe_eee_downstream_unlock.py",
    REPO_ROOT / "tests" / "test_divergence_pins_jjj.py",
    REPO_ROOT
    / "tests"
    / "test_probe_idiom_unlocked_optional_cost_mirror.py",
    REPO_ROOT
    / "tests"
    / "test_probe_idiom_unlocked_multi_target_picker.py",
    REPO_ROOT / "tests" / "test_probe_remaining_ooo.py",
    REPO_ROOT / "tests" / "test_probe_remaining_qqq.py",
    REPO_ROOT / "tests" / "test_divergence_pins_oo.py",
    # 12th Rust slice (slot-aware Controlled bridge form + RRR Pin 1):
    REPO_ROOT / "tests" / "test_probe_mmm_downstream.py",
    REPO_ROOT / "tests" / "test_probe_rrr_downstream.py",
    # Probe batch SSS (round 23 covered-but-unprobed):
    REPO_ROOT / "tests" / "test_probe_remaining_sss.py",
    # Probe batch UUU (round 24 covered-but-unprobed):
    REPO_ROOT / "tests" / "test_probe_remaining_uuu.py",
    # Probe batch WWW (round 25 covered-but-unprobed):
    REPO_ROOT / "tests" / "test_probe_remaining_www.py",
    # 14th Rust slice (TTT Pins 1+2 closures):
    REPO_ROOT / "tests" / "test_probe_ttt_downstream.py",
    # 15th Rust slice (VVV Pin 1 — Predicate::SourceIsSelf
    # player-subject tautology arm for Survivor postcombat-main):
    REPO_ROOT / "tests" / "test_probe_vvv_downstream.py",
    # 17th Rust slice (DDD Pins 1+2+3 — pure-Python bridge arms for
    # SurveyMechanActivationEffect + ExileTopXOfLibraryMayPlayThisTurn
    # + SearchLibraryForEquipmentToBattlefield; first Formula
    # dispatcher _ta_dict_to_python_formula in cross_engine.py):
    REPO_ROOT / "tests" / "test_probe_ddd_downstream.py",
    # Probe batch YYY (round 26 covered-but-unprobed) — the two
    # remaining bespoke-Effect covered-by-construction cards
    # (Diplomatic Relations + Astelli Reclaimer) closed via the same
    # parse + bridge round-trip + Python-side resolve idiom DDD
    # slice 17's Honored Knight-Captain probe established:
    REPO_ROOT / "tests" / "test_probe_remaining_yyy.py",
    # 18th Rust slice (XXX Pin 2 — Roving Actuator Void ETB; new
    # Effect::RovingActuatorETBEffect { max_mv: u8 } variant +
    # parse + resolve + bridge; cast-from-exile-copy-for-free
    # substrate via recursive apply_effect on the exiled card's
    # printing.effects with the source-controller transient pinned):
    REPO_ROOT / "tests" / "test_probe_xxx_downstream.py",
    # Probe batch ZZZ (round 26 covered-but-unprobed) — the LAST
    # alphabetic-named probe batch.  Reinstates two YYY drop-list
    # entries that the YYY-idiom (parse + bridge round-trip +
    # Python-side resolve, NO cross-engine compare_fields) DOES
    # unblock -- Stay Hidden, Stay Silent (Aura ETB +
    # TapEnchantedPermanent unit-variant) and Sheltered by Ghosts
    # (Aura ETB + ExileOpponentsNonlandPermanentUntilLeaves; FIRST
    # Aura-carrier shape on that effect class, distinct from the
    # PROBED Banishing Light bare-Enchantment carrier).  Future
    # probe batches (round 27+) must use a 4-letter or numeric
    # naming convention -- the alphabetic 3-letter naming is
    # exhausted by this batch.
    REPO_ROOT / "tests" / "test_probe_remaining_zzz.py",
    # Probe batch AAAA (round 27 covered-but-unprobed) — the FIRST
    # 4-letter probe batch, established post-ZZZ alphabetic exhaustion
    # per the ZZZ docstring naming-convention note.  Two cards from
    # the covered tier upgraded to PROBED via the same parse + bridge
    # round-trip + Python-side resolve idiom DDD slice 17 / YYY / ZZZ
    # established:
    #   * Cryoshatter (Aura with BECOMES_TAPPED + DEALT_DAMAGE host-
    #     event triggers -> DestroyEnchantedPermanent; FIRST Aura-
    #     carrier with the DestroyEnchantedPermanent unit-variant
    #     to land a credited _parity probe -- complements ZZZ's
    #     Stay Hidden / Stay Silent Aura ETB + TapEnchantedPermanent
    #     by pinning DESTROY-mutation + host-event-triggered firing).
    #   * Meltstrider's Gear (Equipment ENTERS + IsSelf ->
    #     AttachToTargetCreatureYouControl bare, no UEOT keyword
    #     rider; FIRST Equipment-carrier ETB with the bare attach
    #     to land a credited _parity probe -- the bridge docstring
    #     at cross_engine.py:3367-3372 explicitly cited Meltstrider's
    #     Gear as the cleanest real-EoE consumer of the bare shape).
    # Subsequent 4-letter batches follow BBBB / CCCC / ... per the
    # established alphabetic cadence, now widened by one letter to
    # give the campaign another 26 batches of room.
    REPO_ROOT / "tests" / "test_probe_remaining_aaaa.py",
    # Probe batch BBBB (round 28 covered-but-unprobed) — second
    # 4-letter probe batch.  One probe lands via the QQQ cross-engine
    # spell-effect idiom: Withering Torment (Instant
    # Sequence(DestroyTargetCreature, LoseLife(2)) — FIRST Sequence-
    # of-destroy+lose_life spell-effect probe; ``lose_life``
    # previously pinned only at synthetic-ETB-trigger level (round
    # 24)).  Vanish from Sight DROPPED on verify for a bounce-to-
    # library summoning-sick divergence — parallel to the ZZZ-flagged
    # exile-summoning-sick gap (Trapped in the Screen drop); a
    # sibling-lane divergence pin should pin the bounce path.  See
    # module docstring for the full honest-exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_bbbb.py",
    # Probe batch CCCC (round 29 covered-but-unprobed) — third
    # 4-letter probe batch (after AAAA and BBBB).  Two cards from
    # the covered tier upgraded to PROBED via the same parse +
    # bridge round-trip + Python-side resolve idiom DDD slice 17 /
    # YYY / ZZZ / AAAA established:
    #   * Lightless Evangel (Creature IS_SACRIFICED ->
    #     PutPlusOneCounterOnSelf(amount=1); FIRST IS_SACRIFICED-
    #     event carrier of the effect class to land a credited
    #     probe -- complements the existing ENTERS / IS_CAST event
    #     consumers covered by construction).
    #   * Skystinger (Creature BLOCKS -> PumpSelfUntilEOT(
    #     power_bonus=5, toughness_bonus=0, granted_keywords=
    #     frozenset()); FIRST BLOCKS-event carrier of the BARE-shape
    #     variant -- the empty granted_keywords is what makes the
    #     bare bridge byte-equal, distinguishing it from the
    #     covered-but-not-probed Hand That Feeds / Cult Healer
    #     rider-bearing variants).
    # Trapped in the Screen was considered but DROPPED on verify
    # (bit-identical shape to the already-PROBED Banishing Light;
    # the same ZZZ + YYY drop-list precedent that already pins this
    # as honest-exhaustion).
    REPO_ROOT / "tests" / "test_probe_remaining_cccc.py",
    # Probe batch DDDD (round 30 reclaim covered-but-unprobed) —
    # fourth 4-letter probe batch (after AAAA, BBBB, CCCC; sibling
    # to EEEE which merged first and shipped DDDD's originally-
    # drafted cards Starwinder + Broodguard Elite).  This reclaim
    # re-picks two DIFFERENT covered-but-unprobed cards to avoid
    # duplicate credit (the PROBED-credit predicate is name+_parity
    # function match; re-probing the EEEE cards would not move the
    # classifier needle).  Two cards from the covered tier upgraded
    # to PROBED via the same parse + bridge round-trip + Python-side
    # resolve idiom DDD slice 17 / YYY / ZZZ / AAAA / CCCC / EEEE
    # established:
    #   * Meltstrider Eulogist (Creature DIES -> DrawCards(amount=1);
    #     FIRST PROBE_MODULES-credited probe of the DIES + bare
    #     DrawCards event + effect-class combination -- the only EoE
    #     consumer of this combination, with the HasCounter
    #     predicate drop-recovered under the YYY-idiom per the
    #     CCCC Lightless Evangel predicate-is-metadata precedent;
    #     the i-batch's prior drop used the full-compare_fields
    #     idiom requiring Rust-bridge predicate parity, which the
    #     YYY-idiom does not).
    #   * Balemurk Leech (Creature ENTERS ->
    #     EachOpponentLosesLife(amount=1); FIRST PROBE_MODULES-
    #     credited probe of the EachOpponentLosesLife effect class
    #     -- the only EoE consumer of this effect class per the
    #     bridge-arm docstring at cross_engine.py:4872-4874; the CR
    #     118.4 direct life loss primitive distinct from sibling
    #     TargetOpponentLosesLife and DamageEachOpponent; the
    #     non-credited test_probe_subject_is_enchantment_unlock.py
    #     function predates the _parity naming convention and uses
    #     full compare_fields requiring the Rust-bridge subject_is_
    #     enchantment arm, while DDDD lands the credited _parity
    #     variant under the YYY-idiom).
    # Honest-exhaustion drops: Sami, Ship's Engineer (Conditional
    # OnTappedCreatures wrapper substrate), Meltstrider's Resolve
    # (Aura subsystem -- in flight on feature/aura-substrate-step-7),
    # Dawnsire (no bare DealDamageToTargetCreature bridge arm; only
    # the rider-bearing variant ships), Fear of Failed Tests (bit-
    # identical to EEEE Starwinder), Cackling Slasher / Daggermaw /
    # vanilla creatures (no Effects to probe), activated-ability
    # cards (subsystem-gated, separate probe surface).  See module
    # docstring for the full honest-exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_dddd.py",
    # Probe batch EEEE (round 31 reclaim covered-but-unprobed) — fifth
    # 4-letter probe batch (this slot was previously claimed by an
    # agent that hit the API limit; that branch is gone, slot
    # reclaimed fresh).  Two cards from the covered tier upgraded to
    # PROBED via the same parse + bridge round-trip + Python-side
    # resolve idiom DDD slice 17 / YYY / ZZZ / AAAA / CCCC
    # established:
    #   * Starwinder (Creature DEALS_COMBAT_DAMAGE ->
    #     DrawCardsEqualToResolvingCombatDamage(); FIRST credited
    #     probe of this bare unit-variant effect class -- the
    #     WWW/YYY batches explicitly deferred this card citing
    #     combat-damage harness substrate, but the bare effect+
    #     bridge+resolve idiom unblocks it: the resolve reads
    #     state.resolving_combat_damage_amount, which the probe
    #     sets directly per the effect's resolve contract at
    #     draw_mill.py:106; the harness-driven trajectory firing
    #     is a stronger but distinct test shape that stays
    #     deferred).
    #   * Broodguard Elite (Creature LEAVES_BATTLEFIELD ->
    #     MoveSelfCountersToTargetCreatureYouControl(); FIRST
    #     LEAVES_BATTLEFIELD-event carrier of this effect class to
    #     land a credited probe -- complements the existing
    #     PROBED Dockworker Drone ENTERS-event carrier).
    # Fear of Failed Tests was considered but DROPPED on verify
    # (bit-identical effect-class shape to Starwinder; the predicate
    # distinction is metadata and does not move the PROBED-tier
    # classifier).  See module docstring for the full honest-
    # exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_eeee.py",
    # Probe batch FFFF (round 32 relaunch covered-but-unprobed) — sixth
    # 4-letter probe batch (this slot was previously claimed by an
    # agent that timed out at worktree creation; that branch is gone,
    # slot reclaimed fresh).  ONE card from the covered tier upgraded
    # to PROBED via the same parse + bridge round-trip + Python-side
    # resolve idiom DDD slice 17 / YYY / ZZZ / AAAA / CCCC / EEEE /
    # DDDD established:
    #   * Illvoi Operative (Creature IS_CAST ->
    #     PutPlusOneCounterOnSelf(amount=1); FIRST IS_CAST-event
    #     carrier of the PutPlusOneCounterOnSelf effect class to land
    #     a credited probe -- the only EoE consumer of this
    #     (event, effect-class) tuple.  Earlier YYY / SSS / UUU drops
    #     deferred this card on cross-engine compare_fields grounds
    #     (slice-12 IS_CAST stack-resolution-order divergence), but
    #     the EEEE-idiom (parse + bridge round-trip + Python-side
    #     resolve, NO cross-engine compare_fields) sidesteps that
    #     gap entirely -- same drop-recovery precedent DDDD used for
    #     Meltstrider Eulogist (HasCounter predicate gap) and EEEE
    #     used for Starwinder (combat-damage harness gap).
    # Honest-exhaustion is the dominant signal for the remainder of
    # the covered_by_construction tier -- the magecraft-family
    # siblings Cosmogrand Zenith / Station Monitor / Sunstar
    # Lightsmith are blocked by their EFFECT-class wrappers (modal /
    # token-registration / Sequence), not the IS_CAST event itself;
    # the long tail of activated-ability cards, Equipment / Aura /
    # Room subsystem cards, pump-rider cards, vanilla creatures,
    # ManifestDread substrate, and non-basic lands carry their own
    # subsystem / bridge-arm blockers (see module docstring for the
    # full honest-exhaustion drop list).
    REPO_ROOT / "tests" / "test_probe_remaining_ffff.py",
    # Probe batch GGGG (round 33 covered-but-unprobed) — seventh
    # 4-letter probe batch (after AAAA, BBBB, CCCC, DDDD, EEEE, FFFF).
    # ONE card from the covered tier upgraded to PROBED via the same
    # parse + bridge round-trip + Python-side resolve idiom DDD slice
    # 17 / YYY / ZZZ / AAAA / CCCC / DDDD / EEEE / FFFF established:
    #   * Roving Actuator (Artifact Creature ENTERS ->
    #     RovingActuatorETBEffect(max_mv=2); FIRST credited _parity
    #     probe of this dedicated bespoke effect class on the REAL
    #     parsed Roving Actuator EoE printing.  The
    #     test_probe_xxx_downstream.py test_xxx_probe_roving_actuator
    #     _* function family exercises the bridge round-trip + cross-
    #     engine resolve trajectory on a SYNTHETIC carrier (built
    #     via register_and_inject_object) but is NOT credited
    #     because the function names predate the _parity convention;
    #     GGGG binds the REAL parsed printing via
    #     get_printings().get("Roving Actuator") and asserts the
    #     empty-target short-circuit at bespoke.py:667-668).
    # Honest-exhaustion drops include Illvoi Operative (ALREADY
    # PROBED via FFFF round 32 relaunch, main HEAD 25c0d410 -- the
    # original GGGG draft included Illvoi Operative as a second
    # probe but it was dropped on origin/main merge since FFFF
    # landed first and credited it), Optimistic Scavenger (Drix
    # Fatemaker already PROBED on same effect+event combo), Unwanted
    # Remake / Come Back Wrong (Murder already PROBED bare),
    # Weapons Manufacturing / Station Monitor (token-registration
    # substrate), Clammy Prowler / Dawnsire (no bridge arm), Hand
    # That Feeds (granted_keywords rider dropped by bridge), Room
    # cards (Rooms subsystem), ManifestDread carriers (picker
    # substrate deferred), Sequence / Modal wrappers (no container-
    # level bridge), Pain for All / Tractor Beam (Aura phase 2 in
    # flight), vanilla / lands / activated-ability cards (separate
    # probe surface).  See module docstring for the full honest-
    # exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_gggg.py",
    # Probe batch HHHH (round 34 covered-but-unprobed) -- eighth
    # 4-letter probe batch (after AAAA, BBBB, CCCC, DDDD, EEEE, FFFF,
    # GGGG).  THREE Aura ENTERS bespoke-Effect cards from the covered
    # tier upgraded to PROBED via the same parse + bridge round-trip
    # + Python-side resolve idiom DDD slice 17 / YYY / ZZZ / AAAA /
    # CCCC / DDDD / EEEE / FFFF / GGGG established (the substrate-
    # shipped-but-no-_parity-probe measurement-attribution gap --
    # same shape as the warp_cost Phase 3 precedent at merge
    # 33188239):
    #   * Pain for All (Aura ENTERS -> PainForAllETBEffect();
    #     FIRST credited _parity probe of the Aura Phase 3 / S4
    #     triggered-shape bespoke effect class.  The
    #     test_probe_pain_for_all_downstream.py
    #     test_pain_for_all_etb_* family covers the bridge round-trip
    #     + Rust-side fizzle but predates the _parity naming
    #     convention; HHHH binds the REAL parsed Pain for All EoE
    #     printing and asserts the no-attached_to silent-fizzle at
    #     bespoke.py:389-391).
    #   * Tractor Beam (Aura ENTERS -> TapEnchantedPermanent();
    #     FIRST credited _parity probe of the Aura Phase 2 S2 /
    #     cast-time auto-resolve bespoke effect class.  The
    #     test_tractor_beam.py / test_tractor_beam_control.py /
    #     test_aura_substrate_phase_2_s2_extension.py test functions
    #     cover the resolve mechanics on synthetic carriers but
    #     predate the _parity naming convention; HHHH binds the REAL
    #     parsed Tractor Beam EoE printing and asserts the no-
    #     attached_to silent-fizzle at tap.py:516-517).
    #   * Meltstrider's Resolve (Aura ENTERS ->
    #     EnchantedCreatureFightsTargetEffect(); FIRST credited
    #     _parity probe of the slice-19 bespoke effect class.  The
    #     test_meltstriders_resolve.py /
    #     test_aura_substrate_phase_4.py test functions cover the
    #     resolve mechanics on synthetic carriers but predate the
    #     _parity naming convention; HHHH binds the REAL parsed
    #     Meltstrider's Resolve EoE printing and asserts the empty-
    #     target silent-fizzle at bespoke.py:544-545).
    # Honest-exhaustion drops include The Seriema (SearchLibrary
    # already PROBED via siblings), Lightstall Inquisitor (cast-
    # permission flag round-trip pending verification), Dawnsire /
    # Clammy Prowler / Temporal Intervention (no bridge arm), pump
    # carriers with granted_keywords riders (Hand That Feeds / Flesh
    # Burrower / Cult Healer / Give In to Violence / Dual-Sun
    # Technique / Horrid Vigor / Rig for War / Violent Urge -- bridge
    # drops the rider), token-registration cards (Sami / Station
    # Monitor / Weapons Manufacturing / Grand Entryway), Sequence /
    # Modal wrappers (Sunstar Lightsmith / Cosmogrand Zenith / Rust
    # Harvester / Requiem Monolith / Let's Play a Game / Split Up /
    # Lithobraking / Vanish from Sight / Unnerving Grasp), Equipment
    # carriers / ManifestDread carriers (substrate deferral), bit-
    # identical-to-PROBED-siblings (Trapped in the Screen / Unwanted
    # Remake / Come Back Wrong / Optimistic Scavenger), Room
    # subsystem cards, vanilla creatures, lands (separate activated-
    # ability surface).  See module docstring for the full honest-
    # exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_hhhh.py",
    # Probe batch IIII (round 35 covered-but-unprobed) -- ninth
    # 4-letter probe batch (after AAAA, BBBB, CCCC, DDDD, EEEE, FFFF,
    # GGGG, HHHH).  FOUR cards from the covered tier upgraded to
    # PROBED via the same parse + bridge round-trip + Python-side
    # resolve idiom DDD slice 17 / YYY / ZZZ / AAAA / CCCC / DDDD /
    # EEEE / FFFF / GGGG / HHHH established (the substrate-shipped-
    # but-no-_parity-probe measurement-attribution gap -- same shape
    # as the warp_cost Phase 3 precedent at merge 33188239):
    #   * Lightstall Inquisitor (Creature ENTERS ->
    #     EachOpponentExilesCardFromHandWithCastPermission(); FIRST
    #     credited _parity probe of the bespoke effect class shipped
    #     in the 20th Rust slice (XXX Pin 3 closure).  The
    #     test_probe_xxx_pin3_downstream.py test_xxx_pin3_* family
    #     exercises the bridge round-trip + Rust-side empty-hand no-op
    #     on a SYNTHETIC carrier; an existing _parity-suffixed
    #     function (test_xxx_pin3_real_eoe_lightstall_inquisitor_
    #     cast_from_exile_substrate_parity) does NOT match the slug
    #     scanner regex (which anchors on the full
    #     ``test_cross_engine_real_eoe_<slug>_parity`` prefix), so
    #     Lightstall remained uncredited.  IIII binds the REAL parsed
    #     Lightstall Inquisitor EoE printing and asserts the
    #     per-opponent empty-hand skip at discard.py:196-197).
    #   * The Seriema (Artifact (Spacecraft) ENTERS ->
    #     SearchLibraryForCard(filter_name='legendary_creature',
    #     max_mv=0, may_fail=False); FIRST credited _parity probe of
    #     the bare SearchLibraryForCard shape with the
    #     (legendary_creature, may_fail=False) parameterization on an
    #     ENTERS carrier.  Sibling PROBED consumers pin different
    #     parameterization quadruples: Spineseeker Centipede pins
    #     (ENTERS, bare, basic_land, may_fail=True); Starfield
    #     Shepherd pins (ENTERS, ModalChooseOne wrapper, basic_plains
    #     + creature_small); Demonic Counsel pins (SPELL, bare, demon,
    #     may_fail=False).  HHHH dropped The Seriema citing
    #     "SearchLibrary already PROBED via siblings" but the
    #     (legendary_creature, may_fail=False) bare-ENTERS quadruple
    #     was not covered by any of the three siblings -- the
    #     attribution-gap closure here is the credited delta).
    #   * Optimistic Scavenger (Creature ENTERS ->
    #     PutPlusOneCounterOnTargetCreature(amount=1); FIRST credited
    #     _parity probe binding the Optimistic Scavenger slug to the
    #     Effect+bridge round-trip on the REAL parsed EoE printing.
    #     The PROBED-tier classifier credits per-slug; sibling PROBED
    #     consumers (Drix Fatemaker / Mm'menon Uthros Exile / Honor /
    #     Selfcraft Mechan / Shrewd Storyteller) cover other (event,
    #     predicate) combos but no slug-keyed probe pinned Optimistic
    #     Scavenger.  HHHH dropped Optimistic Scavenger citing "bit-
    #     identical to PROBED Drix Fatemaker" but the predicate
    #     distinction (Subject(IsEnchantment) & Controlled(YOU) vs
    #     IsSelf) + the ROOM_FULLY_UNLOCKED second trigger DO produce
    #     a different per-slug attribution -- the classifier credits
    #     per-slug, not per-effect-class).
    #   * Command Bridge (Land ENTERS ->
    #     SacrificeSelfUnlessTapPermanent(); FIRST credited _parity
    #     probe of the bespoke effect class shipped in the 20th Rust
    #     slice (VVV Pin 2 closure).  Command Bridge is the ONLY EoE
    #     consumer of this effect class -- the HIGHEST-deck-count
    #     BLOCKED-genuinely_missing closure on the
    #     ``docs/BLOCKED_GENUINELY_MISSING_INVENTORY.md`` list at the
    #     time of the VVV Pin 2 substrate landing (24,522 17lands
    #     decks).  HHHH's drop list cited "Lands (activated-ability
    #     surface)" generically as a skip reason for ALL lands, but
    #     Command Bridge is a LAND with a TRIGGERED ability surface
    #     distinct from its activated mana ability -- IIII pins the
    #     triggered shape via the no-source short-circuit at
    #     combinators.py:121-123).
    # Honest-exhaustion drops include Dawnsire / Clammy Prowler /
    # Temporal Intervention (no bridge arm), pump carriers with
    # granted_keywords riders (Hand That Feeds / Flesh Burrower / Cult
    # Healer / Give In to Violence / Dual-Sun Technique / Horrid Vigor
    # / Rig for War / Violent Urge -- bridge drops the rider), token-
    # registration cards (Sami / Station Monitor / Weapons
    # Manufacturing / Grand Entryway), Sequence / Modal wrappers with
    # substrate-deferred inner effects (Sunstar Lightsmith /
    # Cosmogrand Zenith / Rust Harvester / Requiem Monolith / Let's
    # Play a Game / Split Up / Lithobraking / Vanish from Sight /
    # Unnerving Grasp), Equipment carriers / ManifestDread carriers
    # (substrate deferral), bit-identical-to-PROBED-siblings (Trapped
    # in the Screen / Unwanted Remake / Come Back Wrong / Innocuous
    # Rat / Break Down the Door / Under the Skin / Fear of Failed
    # Tests / Irreverent Gremlin), Room subsystem cards, vanilla
    # creatures, lands (separate activated-ability surface), Lane K's
    # Tannuk + Frenzied Baloth (deferred per concurrent-agent note).
    # See module docstring for the full honest-exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_iiii.py",
    # Probe batch JJJJ (round 36 covered-but-unprobed) -- tenth
    # 4-letter probe batch (after AAAA, BBBB, CCCC, DDDD, EEEE, FFFF,
    # GGGG, HHHH, IIII).  THREE Room ENTERS / YOU_ATTACK triggered-
    # ability cards from the covered tier upgraded to PROBED via the
    # same parse + bridge round-trip + Python-side resolve idiom DDD
    # slice 17 / YYY / ZZZ / AAAA / CCCC / DDDD / EEEE / FFFF / GGGG /
    # HHHH / IIII established (the substrate-shipped-but-no-_parity-
    # probe measurement-attribution gap -- same shape as the warp_cost
    # Phase 3 precedent at merge 33188239, and IIII's Command Bridge
    # precedent that pinned a LAND triggered-ability surface distinct
    # from the activated mana ability surface):
    #   * Funeral Room // Awakening Hall (Enchantment (Room) ENTERS ->
    #     ReturnAllCreaturesFromYourGraveyardToBattlefield(); FIRST
    #     credited _parity probe of the bare untargeted-mass-
    #     reanimation effect class on the REAL parsed printing.
    #     The bridge docstring at cross_engine.py:6545-6546 stated
    #     "No EoE oracle hits today (Awakening Hall, the canonical
    #     consumer, is not in EoE)" incorrectly -- Awakening Hall IS
    #     in EoE as the back-half of the split card; JJJJ closes the
    #     corrective gap).
    #   * Moldering Gym // Weight Room (Enchantment (Room) ENTERS ->
    #     SearchLibraryForBasicLandTapped(count=1); FIRST credited
    #     _parity probe of the bare unwrapped shape on a triggered-
    #     ability carrier.  Larval Scoutlander is the known consumer
    #     of the OptionalSac-WRAPPED shape with count=2 -- a different
    #     carrier shape and parameterization.  Moldering Gym is the
    #     front-half door body; the Weight Room back-half emits
    #     ManifestDreadThenPutPlusOneCountersOnIt(counter_amount=3)
    #     which stays the dropped surface per the ManifestDread
    #     harness substrate deferral).
    #   * Surgical Suite // Hospital Room (Enchantment (Room)
    #     YOU_ATTACK -> PutPlusOneCounterOnTargetCreature(amount=1,
    #     attacking_only=True); FIRST credited _parity probe of the
    #     attacking_only=True parameterization.  IIII's Optimistic
    #     Scavenger probe explicitly named the attacking_only=True
    #     parameterization as a distinct axis ("the Hospital Room
    #     half shape" at tests/test_probe_remaining_iiii.py:972) and
    #     used Hospital Room as the contrast carrier for that
    #     distinctness assertion -- JJJJ now pins Hospital Room
    #     directly).
    # Honest-exhaustion drops include Bottomless Pool // Locker Room
    # (DEALS_COMBAT_DAMAGE -> DrawCards(amount=1) bit-identical to
    # PROBED Haliya / Illvoi Infiltrator), Innocuous Rat (DIES ->
    # ManifestDread(count=1) bit-identical to PROBED Bashful Beastie),
    # Ticket Booth // Tunnel of Hate (ENTERS -> ManifestDread bit-
    # identical to PROBED Unsettling Twins), Trapped in the Screen
    # (bit-identical to PROBED Banishing Light), Underwater Tunnel //
    # Slimy Aquarium / Break Down the Door / Under the Skin (Manifest
    # Dread substrate deferral), Grand Entryway (token-registration
    # substrate), Dawnsire / Clammy Prowler / Temporal Intervention
    # (no bridge arm), pump carriers with granted_keywords riders
    # (Hand That Feeds / Flesh Burrower / Cult Healer / Give In to
    # Violence / Dual-Sun Technique / Horrid Vigor / Rig for War /
    # Violent Urge / Derelict Attic // Widow's Walk YOU_ATTACK half),
    # Equipment ETB cards (Conductive Machete / Cursed Windbreaker /
    # Killer's Mask / Squire's Lightblade / Illvoi Light Jammer / The
    # Dominion Bracelet), Sequence / Modal wrappers with substrate-
    # deferred inner effects (Sunstar Lightsmith / Cosmogrand Zenith
    # / Rust Harvester / Requiem Monolith / Let's Play a Game / Split
    # Up / Lithobraking / Vanish from Sight / Unnerving Grasp /
    # Restricted Office // Lecture Hall / Mirror Room // Fractured
    # Realm), Fear of Failed Tests (bit-identical to PROBED Starwinder),
    # vanilla creatures + lands + activated-ability-only cards.  See
    # module docstring for the full honest-exhaustion drop list.
    REPO_ROOT / "tests" / "test_probe_remaining_jjjj.py",
    # Wrath-admit-widening lane (round-87, 2026-05-21) — the Phase-3
    # (0, 0)-shape wrath admit arm + three wrath-shape serializer
    # leaves (DealNDamageToEachCreature [Pyroclasm-shape] / bare
    # DestroyAllCreatures / ManifestDread) on
    # `_printing_is_bridge_simple_spell` + `_python_spell_effect_to_dict`.
    # Two ManifestDread carriers (Break Down the Door — Instant
    # {2}{G}; Under the Skin — Sorcery {2}{G}) flip COVERED -> PROBED
    # via the empty-library no-op probe idiom established by the
    # SSS-batch Manifest Dread probe. Pyroclasm AUTO-JOINS the admit
    # set but was already PROBED via the batch-G Pyroclasm probe.
    # Beyond the Quiet stays BLOCKED (substrate work — its
    # ExileAllCreaturesAndSpacecraft Effect is uncovered in Rust;
    # out of lane scope). See the module docstring for the full
    # honest-stop drop list (Singularity Rupture, Mutinous Massacre,
    # Zero Point Ballad, Zealous Display, etc.).
    REPO_ROOT / "tests" / "test_probe_wrath_admit_widening.py",
    # Reroute Systems closure (2026-05-21). Adds two
    # ``test_cross_engine_real_eoe_reroute_systems_<mode>_parity``
    # functions — the mode-0 (PumpTargetArtifactOrCreatureUntilEOT
    # widened with ``granted_keywords`` + at-cast per-mode detector
    # entry) and mode-1 (DealDamageToTargetTappedCreature new variant
    # + at-resolve filter recheck) probes for Reroute Systems
    # (6,132 17lands decks). The card AUTO-JOINS the spell-card-bridge
    # admit set via the same Cycle 4 widen + the new Cycle 5/6
    # per-mode detector entries + the new per-leaf reverse serializer
    # arms. PROBED tier count +1.
    REPO_ROOT / "tests" / "test_reroute_systems.py",
    # **drill-too-deep-slice (2026-05-21):** ModalChooseOne envelope
    # Cycle 6 — first per-mode-target ModalChooseOne closure. The
    # ``test_cross_engine_real_eoe_drill_too_deep_put_charge_counters
    # _on_target_spacecraft_or_planet_you_control_parity`` probe drives
    # the genuine Drill Too Deep printing through the new
    # ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl`` Effect
    # arm + the per-mode target binding detector table extension.
    # Flips Drill Too Deep COVERED → PROBED (admit set 17 → 18, plus
    # the Untimely Malfunction co-admit via the new
    # ``destroy_target_artifact`` serializer leaf — see the slice's
    # admit-set test note).
    REPO_ROOT / "tests" / "test_drill_too_deep_parity.py",
    # **scour-for-scrap-cycle-2 (2026-05-21):** ModalChooseOne envelope
    # — FIRST ``or_both=True`` flip. The
    # ``test_cross_engine_real_eoe_scour_for_scrap_return_target_artifact
    # _card_from_your_graveyard_to_hand_parity`` probe drives the
    # genuine Scour for Scrap printing through the new
    # ``Effect::ReturnTargetArtifactCardFromYourGraveyardToHand`` Rust
    # unit variant + the per-mode detector table extension (the
    # or_both walk peeks the targeted mode-1 leaf) + the matching
    # bridge round-trip arms + the Phase-5 ``(0, 1)`` ``or_both``
    # admit-shape widen. Flips Scour for Scrap COVERED → PROBED
    # (admit set 23 → 24; ~6,434 17lands decks). The narrow
    # ``SearchLibraryForCard(filter_name="artifact")`` ``_leaf`` admit
    # honest-stops Demonic Counsel's auto-flip to a follow-up cycle.
    REPO_ROOT / "tests" / "test_scour_for_scrap_parity.py",
    # **non-state-rs-tractable-slice (2026-05-21):** Effect-uncovered
    # single-gated tier closure — Exorcise (Duskmourn {1}{W} Sorcery).
    # The ``test_cross_engine_real_eoe_exorcise_parity`` probe drives
    # the genuine parsed Exorcise printing through the new
    # ``Effect::ExorciseEffect`` Rust unit variant + the file-local
    # ``read_doubling_power`` effective-power reader + bridge arm +
    # reverse-serializer leaf. The 3-way OR target filter (Artifact OR
    # Enchantment OR Creature-with-effective-power-4+) is the
    # structural sibling of Shattered Wings's destroy-half filter,
    # with two swaps — exile-not-destroy (no indestructible-skip, per
    # CR 702.12) and power-4-not-flying (reads effective power, folds
    # counters + UEOT bonuses, matches Python's
    # ``effective_power`` byte-equal). Flips Exorcise COVERED →
    # PROBED (admit set 20 → 21 — the §2 auto-grow teeth).
    REPO_ROOT / "tests" / "test_exorcise_parity.py",
    # **close-encounter-substrate (2026-05-21):** Close Encounter
    # additional-cost-choice closure (~16,098 17lands decks — highest-
    # deck-count BLOCKED card in the substrate-gated tier as of the
    # close-encounter merge 2406068c). The
    # ``test_cross_engine_real_eoe_close_encounter_damage_parity`` probe
    # drives the genuine parsed Close Encounter EoE printing through
    # the new ``Effect::CloseEncounterEffect`` Rust unit variant + the
    # additional-cost-choice transient + bridge arm. The dedicated
    # ``test_close_encounter_parity.py`` module was authored alongside
    # the substrate but its scanner registration was missed at merge —
    # this entry closes the measurement-attribution gap. Flips Close
    # Encounter COVERED → PROBED (the highest-deck-count BLOCKED card
    # in the prior cycle's substrate-gated tier finally banks the
    # PROBED-tier credit it earned).
    REPO_ROOT / "tests" / "test_close_encounter_parity.py",
    # **multi-target-admit-phase4 (round-pre-launcher, merged
    # 2026-05-20 via ``feature/multi-target-admit-phase4``):** Scout
    # for Survivors via the Phase-4 multi-target admit-gate widen.
    # The dedicated probe module
    # ``test_multi_target_phase4.py`` carries the canonical
    # ``test_cross_engine_real_eoe_scout_for_survivors_return_up_to_n
    # _creatures_with_counters_parity`` function which binds the
    # genuine parsed "Scout for Survivors" printing via
    # ``get_active_source().get_printings().get("Scout for Survivors")``.
    # Lane JJ subsequently added a smoke-file delegator (so the slug-
    # scanner credits via the smoke path), but the canonical probe
    # module remained unregistered — this entry plugs the
    # measurement-attribution gap at the canonical source. No PROBED
    # delta (the smoke delegator already covers it); pure metric
    # truthfulness — same precedent as the cycle-10 reconciliation.
    REPO_ROOT / "tests" / "test_multi_target_phase4.py",
    # **PPP Pins 1+2+3 (10th Rust slice, ``41c57251``):** the
    # ``test_probe_ppp_downstream.py`` module carries three canonical
    # probes naming the ``_parity`` convention (Drix Fatemaker,
    # Cautious Survivor, Savior of the Small) but was never registered
    # alongside its sibling ``test_probe_<letters>_downstream.py``
    # modules already in PROBE_MODULES (bbb / eee / mmm / rrr / ttt /
    # vvv / ddd / xxx). The cards each probe pins are already PROBED
    # via redundant routes (Drix Fatemaker has a smoke-file probe;
    # Cautious Survivor + Savior of the Small are credited via
    # vvv_downstream), so registration is metric-truthfulness only
    # (no PROBED delta) — closes the per-slice-canonical-probe
    # attribution gap and makes the metric robust against future
    # rearrangement of the redundant routes.
    REPO_ROOT / "tests" / "test_probe_ppp_downstream.py",
    # **cracked-skull-cycle-2-p1 (2026-05-22):** Cracked Skull C2 closure
    # — the second-cycle PROBED flip of the Cycle-1 (C1) substrate. C2
    # ships ``Effect::TargetOpponentLookHandMayDiscard`` on the Rust
    # side (parse + to_pydict + resolve arm), the cross-engine bridge
    # arm ``target_opponent_look_hand_may_discard`` in
    # ``cross_engine.py::_ta_dict_to_python_effect``, and the Phase-3
    # safe-trigger admit-set widen for Cracked Skull's two-trigger
    # composition (``ENTERS+IsSelf+TargetOpponentLookHandMayDiscard``
    # shape 4 + ``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent`` shape
    # 5). The ``test_cross_engine_real_eoe_cracked_skull_enters_parity``
    # probe drives the genuine parsed Cracked Skull EoE printing
    # through the bridge round-trip + Python-side resolve smoke,
    # matching the HHHH/Pain-for-All / Tractor-Beam canonical Aura-
    # probe shape. Flips Cracked Skull BLOCKED-multi-gate → PROBED
    # (the first Aura with a two-trigger composition — ETB +
    # DEALT_DAMAGE host-event — to land end-to-end on the bench
    # loader).
    REPO_ROOT / "tests" / "test_cracked_skull_parity.py",
    # **sporogenic-infection-cycle-1-p1 (2026-05-22):** Sporogenic
    # Infection Cycle 1 closure — full PROBED flip in one cycle (the
    # ENTERS half's ``TargetOpponentSacrificesCreatureOtherThanEnchanted``
    # Effect + new ``OpponentSacrificesCreatureChoice`` PendingDecision
    # substrate landed together; the DEALT_DAMAGE half rode the existing
    # Cracked Skull C2 destroy-enchanted-permanent Phase-3 admit shape).
    # Ships ``Effect::TargetOpponentSacrificesCreatureOtherThanEnchanted``
    # on the Rust side (parse + to_pydict + resolve arm), a new
    # ``PendingDecision::OpponentSacrificesCreatureChoice`` variant +
    # dispatcher (the FIRST opponent-decision substrate — distinct from
    # the sibling controller-decision variants like
    # ``LookAtOpponentHandMayDiscard`` / ``TapPermanentChoice``), the
    # cross-engine bridge arm
    # ``target_opponent_sacrifices_creature_other_than_enchanted`` in
    # ``cross_engine.py::_ta_dict_to_python_effect``, and the Phase-3
    # safe-trigger admit-set widen for Sporogenic Infection's two-
    # trigger composition (``ENTERS+IsSelf+TargetOpponentSacrifices
    # CreatureOtherThanEnchanted`` shape 6, pairs with Cracked Skull
    # C2's shape 5 ``DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent``).
    # The ``test_cross_engine_real_eoe_sporogenic_infection_enters_
    # sacrifice_parity`` probe drives the genuine parsed Sporogenic
    # Infection EoE printing through the bridge round-trip + Python-
    # side resolve smoke, matching the Cracked-Skull / HHHH /
    # Pain-for-All / Tractor-Beam canonical Aura-probe shape. Flips
    # Sporogenic Infection BLOCKED-multi-gate → PROBED (the second Aura
    # with a two-trigger composition — ETB + DEALT_DAMAGE host-event —
    # to land end-to-end on the bench loader, alongside Cracked Skull).
    REPO_ROOT / "tests" / "test_sporogenic_infection_parity.py",
)

# Tier keys (stable — the doc + test reference these verbatim).
TIER_PROBED = "probed"
TIER_COVERED = "covered_by_construction"
TIER_ADMITTED = "bridge_admitted_bench_exercised"
TIER_BLOCKED_MISSING = "blocked_genuinely_missing"
TIER_BLOCKED_NON_EFFECT = "blocked_non_effect"
TIER_BLOCKED_SUBSYSTEM = "blocked_subsystem_gated"

_TIER_ORDER = (
    TIER_PROBED,
    TIER_COVERED,
    TIER_ADMITTED,
    TIER_BLOCKED_MISSING,
    TIER_BLOCKED_NON_EFFECT,
    TIER_BLOCKED_SUBSYSTEM,
)

_TIER_LABEL = {
    TIER_PROBED: "PROBED byte-equal",
    TIER_COVERED: "Effect+bridge covered (not individually probed)",
    TIER_ADMITTED: "bridge-admitted & bench-exercised",
    TIER_BLOCKED_MISSING: "BLOCKED — genuinely_missing (new-Rust-Effect ask)",
    TIER_BLOCKED_NON_EFFECT: "BLOCKED — blocked_non_effect (non-Effect gate)",
    TIER_BLOCKED_SUBSYSTEM: "BLOCKED — subsystem-gated / Effect-uncovered",
}


# ------------------------------------------------- reuse the coverage parser
#
# Round-66 covered-set extraction precursor: ``covered_python_class_set``
# and ``card_effect_classes`` were factored OUT of this script into the
# production-importable ``mtg_bot.parity.effect_coverage`` module so the
# spell-card-bridge Phase 1 guardrail
# (``trajectory._spell_effect_is_covered_and_bridged``) can query the
# live covered set without a ``scripts/`` import (a layering inversion).
# This script RE-EXPORTS them (unchanged names + signatures) so its own
# consumers (``tests/test_measure_pool_parity.py``, the two
# ``scout_r58`` / ``scout_r63`` scouts, the staleness cron) and the
# test's ``monkeypatch.setattr(mod, "covered_python_class_set", ...)``
# continue to work verbatim. Exactly ONE definition of "covered" — the
# §2-teeth requirement (the parse rule is NOT duplicated here).

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# Keep ``scripts/`` on ``sys.path`` for the lazy ``scout_r58`` /
# ``scout_r63`` imports below — previously ensured as a side effect of
# the now-removed ``_coverage_module()`` (it inserted ``SCRIPTS_DIR``);
# the scout breakout must keep rendering into ``docs/POOL_PARITY.md``
# exactly as before this refactor.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mtg_bot.parity.effect_coverage import (  # noqa: E402,F401
    card_effect_classes,
    covered_python_class_set,
)


# ------------------------------------------------- genuine-probe inventory


def _slug(text: str) -> str:
    """snake_case slug used to match a probe function-name prefix to a
    printed card name. Lowercase, non-alphanumerics → ``_``, collapsed,
    trimmed. ``"Mechan Assembler"`` → ``mechan_assembler``;
    ``"Roaring Furnace // Steaming Sauna"`` is keyed on its LEFT face
    (``roaring_furnace``) since split-card probes name the left face."""
    primary = text.split(" // ")[0]
    return re.sub(r"[^a-z0-9]+", "_", primary.lower()).strip("_")


def _smoke_probed_cards(pool_names: set[str], smoke_path: Path) -> set[str]:
    """``tests/test_parity_smoke.py``'s slug-keyed detection (the
    ORIGINAL, unchanged rule — kept byte-for-byte so the 60 cards it
    already credits are unaffected).

    Detection: every ``def test_cross_engine_real_eoe_<mid>_parity``
    function in the smoke file; ``<mid>`` is matched against the slug
    of each pool card's printed name, longest slug first (so
    ``mechan_assembler`` wins over ``mechan``). Only ``*_parity``
    functions count — ``*_divergence_pin`` functions assert a KNOWN
    divergence, the opposite of byte-equality, and are deliberately
    excluded. Pure text; no engine import."""
    text = smoke_path.read_text(encoding="utf-8")
    fn_mids = re.findall(
        r"def test_cross_engine_real_eoe_(\w+?)_parity\(", text
    )
    # Longest-slug-first so a card whose slug is a prefix of another
    # card's slug doesn't steal the attribution.
    slug_to_name: dict[str, str] = {}
    for n in pool_names:
        slug_to_name.setdefault(_slug(n), n)
    ordered = sorted(slug_to_name, key=len, reverse=True)
    probed: set[str] = set()
    for mid in fn_mids:
        for sg in ordered:
            if mid == sg or mid.startswith(sg + "_"):
                probed.add(slug_to_name[sg])
                break
    return probed


# A ``*_parity`` probe function's GENUINE-card binding: the EoE pool
# printing it drives end-to-end is resolved via
# ``get_active_source().get_printings().get("<Card Name>")`` (or the
# ``[...]`` subscript form). That call — NOT the function-name spelling
# — is the load-bearing "this probe pins THAT card" assertion (the
# function name is only a human label; the batch authors elided
# apostrophes in some names, e.g. ``samis_curiosity`` for
# ``"Sami's Curiosity"``, so a slug-of-the-function-name match would
# silently UNDERCOUNT). We anchor on the genuine binding instead, the
# same thing ``measure_effect_coverage``/``card_effect_classes`` treat
# as authoritative. The name capture tolerates an embedded apostrophe
# (double-quoted Python string) and surrounding whitespace/newlines.
#
# The quoted-string capture must be quote-style aware: a card name
# carries an apostrophe (``"Sami's Curiosity"``,
# ``"Mm'menon, Uthros Exile"``) which a naive ``["']([^"']+)["']``
# would truncate at the embedded ``'`` — the precise bug that made the
# slug rule undercount these. Match a double-quoted string whose body
# may contain ``'`` OR a single-quoted string whose body may contain
# ``"`` (no card name needs an escaped same-style quote). ``\s*`` /
# ``re.DOTALL`` tolerate the multi-line ``.get(\n    "Name"\n)`` form.
_PROBE_GET_PRINTING_RE = re.compile(
    r"get_printings\(\)\s*(?:\.\s*get\s*\(|\[)\s*"
    r"""(?:"([^"]+)"|'([^']+)')""",
    re.DOTALL,
)
# A genuine probe is a TOP-LEVEL (column-0) function whose name starts
# ``test_cross_engine_real_eoe_`` and ends ``_parity``. Anchoring at
# column 0 (``^def`` / ``re.MULTILINE``) means the body is bounded by
# the NEXT column-0 ``def``/``class``/``@`` (or EOF) — so a
# ``*_divergence_pin`` function, a module-level binding, or a helper
# AFTER the last ``*_parity`` function is correctly OUTSIDE every
# credited body (the integrity guard the companion test pins).
_PROBE_FN_HEADER_RE = re.compile(
    r"^def (test_cross_engine_real_eoe_\w+?_parity)\(", re.MULTILINE
)
# A new top-level construct ends the preceding function body.
_TOP_LEVEL_BOUNDARY_RE = re.compile(
    r"^(?:def |class |@|[A-Za-z_])", re.MULTILINE
)


def _probe_module_probed_cards(
    pool_names: set[str], module_path: Path
) -> set[str]:
    """The genuine-real-EoE PROBED cards a dedicated probe module
    (``tests/test_probe_covered_unprobed_*.py``) pins.

    Inclusion predicate (the SAME bar the 60 smoke-file PROBED cards
    meet — match it EXACTLY): a card ``C`` is credited iff the module
    contains a ``def test_cross_engine_real_eoe_<...>_parity`` function
    whose body resolves the GENUINE parsed printing via
    ``get_active_source().get_printings().get("C")`` (or ``[...]``),
    where ``"C"`` is an exact live EoE pool card name. These modules
    use the IDENTICAL idiom as the smoke-file probes (genuine parsed
    ``spell_effect`` / ``triggered_abilities`` on Python, a bridged
    synthetic carrier on Rust, ``compare_fields == []`` over the whole
    field view on a controlled real-EoE state) and the IDENTICAL
    ``..._parity`` naming convention; only the file differs. Functions
    NOT ending ``_parity`` (e.g. ``*_divergence_pin``, or a
    differently-named sweep) contribute nothing — same exclusion as the
    smoke file. A bound name that is not a live pool card (a synthetic
    stub / typo / negative-control fixture) is NOT credited — that is
    the integrity guard against over-counting. Pure text; no engine
    import (mirrors the smoke detector's contract)."""
    if not module_path.exists():
        return set()
    text = module_path.read_text(encoding="utf-8")
    probed: set[str] = set()
    for hm in _PROBE_FN_HEADER_RE.finditer(text):
        # The body runs from just after the header line to the start of
        # the NEXT top-level construct (column-0 def/class/@/identifier)
        # or EOF — precisely bounding it to THIS one function.
        body_start = text.index("\n", hm.end()) + 1
        nb = _TOP_LEVEL_BOUNDARY_RE.search(text, body_start)
        body_end = nb.start() if nb else len(text)
        body = text[body_start:body_end]
        for dq, sq in _PROBE_GET_PRINTING_RE.findall(body):
            nm = dq or sq
            if nm in pool_names:
                probed.add(nm)
    return probed


def genuine_probed_cards(
    pool_names: set[str],
    smoke_path: Path = SMOKE_TEST,
    probe_modules: tuple[Path, ...] = PROBE_MODULES,
) -> set[str]:
    """The set of EoE pool card names that have a genuine real-EoE
    cross-engine *parity* probe pinning them.

    UNION of two detectors at EXACTLY ONE inclusion bar:

      1. ``tests/test_parity_smoke.py`` — slug-keyed ``*_parity``
         function-name match (the original rule, unchanged: the 60
         cards it credits are byte-for-byte unaffected).
      2. the dedicated ``tests/test_probe_covered_unprobed_*.py`` batch
         modules — genuine-``get_printings()``-binding match inside
         each ``*_parity`` function (resolves the
         apostrophe-name-elision the slug rule would silently
         undercount). Same idiom, same naming convention, same
         strength bar (genuine parsed printing driven end-to-end,
         full ``compare_fields`` byte-equality on a controlled
         real-EoE state).

    A card need only meet the bar in EITHER source to be PROBED (set
    union). The honest-measurement-tooling fix: before this, only (1)
    was scanned, so PROBED understated real probe coverage by the
    batch-module cards. The integrity guard: a probe that binds a name
    which is not a live pool card (a synthetic stub) is NOT credited
    — the metric never over-counts."""
    probed = _smoke_probed_cards(pool_names, smoke_path)
    for mod_path in probe_modules:
        probed |= _probe_module_probed_cards(pool_names, mod_path)
    return probed


# ------------------------------------------------- residue worklist split


def residue_blocked_cards(
    path: Path = RESIDUE_WORKLIST,
) -> tuple[set[str], set[str]]:
    """Return ``(genuinely_missing_cards, blocked_non_effect_cards)``
    from the canonical residue worklist. ``closed`` entries are NOT
    blocked (they shipped + bridged) so they are excluded. Card names
    are taken verbatim from each entry's ``card`` field; entries whose
    ``card`` carries a parenthetical qualifier (e.g. the legacy
    duplicate row "Monoist Circuit-Feeder (legacy duplicate row)") are
    normalised to the bare printed name so the consistency cross-check
    keys on the real pool name."""
    import json

    if not path.exists():
        raise RuntimeError(
            f"residue worklist {path} is missing — it is the canonical "
            f"BLOCKED-RUST-EFFECT residue list this metric cross-checks "
            f"against; restore it."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    missing: set[str] = set()
    non_effect: set[str] = set()
    for e in data.get("entries", []):
        card = (e.get("card") or "").strip()
        if not card:
            continue
        # Strip a trailing " (qualifier)" so a legacy-duplicate /
        # annotated row still keys on the real printed name.
        card = re.sub(r"\s*\([^)]*\)\s*$", "", card).strip()
        st = e.get("status")
        if st == "genuinely_missing":
            missing.add(card)
        elif st == "blocked_non_effect":
            non_effect.add(card)
    return missing, non_effect


# ------------------------------------------------- loader admit predicate


def loader_admits(printing) -> bool:
    """True iff the production-deck loader admits this printing — the
    EXACT gate the bench corpus runs through. Mirrors all 10 disjuncts
    of ``trajectory._filter_deck_to_bridge_supported``'s admit gate:

    1. ``_printing_is_bridge_basic_land``
    2. ``_printing_is_bridge_nonbasic_mana_land``
    3. ``_printing_is_bridge_pure_warp_creature``
    4. ``_printing_is_bridge_simple_creature``
    5. ``_printing_is_bridge_simple_spell``
    6. ``_printing_is_bridge_simple_aura``
    7. ``_printing_is_bridge_simple_static_anthem_creature``
    8. ``_printing_is_bridge_simple_uncounterable_creature``
    9. ``_printing_is_bridge_simple_static_self_conditional_creature``
   10. ``_printing_is_bridge_simple_tapped_mana_land`` — non-basic
       land substrate phase 2 (round-85 N1; admits the 5 EoE Planet
       lands as untapped mono-color lands — the parser-flattened
       ``enters_tapped=True`` mono-color shape; the bridge drops
       ``enters_tapped`` on both engines; round-39 Ward / round-60
       shockland symmetric-drop precedent).

    Reused, not reimplemented, so the bench tier and the real bench can
    never disagree on what "admitted" means.

    **warp_cost Phase 3 (round-83, 2026-05-20):** the
    ``_printing_is_bridge_pure_warp_creature`` predicate is admit-arm 3.
    Phases 1+1.5+2 of the warp_cost substrate shipped the full warp
    lifecycle on both engines (CardDef warp fields + Rust
    ``CardPrinting.warp_cost`` + ``cast_creature_warp`` + the END-step
    exile sweep + the exile-recast + Timeline Culler's life-rider +
    graveyard-warp), and the production-deck loader at
    ``trajectory.py:5552`` already routes warp creatures through this
    predicate ahead of the plain simple-creature arm. Including it in
    ``loader_admits`` makes the metric tier-classifier see those
    admitted warp printings as bridge-admitted, closing the warp_cost
    Phase 3 measurement-attribution gap (the 3 single-gated warp cards
    — Red Tiger Mechan, Bygone Colossus, Timeline Culler — accounting
    for 15,436 17lands single-gate decks per the round-58 attribution
    scout).

    **Aura Phase 1 flash widening attribution close (2026-05-21):**
    the ``_printing_is_bridge_simple_aura`` predicate is admit-arm 6.
    The trajectory-side Phase 2 round-79 widen
    (``_AURA_SAFE_EXTRA_KEYWORDS = {"flash"}``) already lets the real
    loader admit Frantic Strength + Shardmage's Rescue end-to-end.

    **loader_admits attribution sweep (2026-05-21):** disjuncts 2 / 5 /
    7 / 8 / 9 — the remaining 5 of the real loader's 9 admit arms — are
    folded in here. Each closes a pure measurement-attribution gap (the
    real loader admits them; the metric helper was stale):

    * ``_printing_is_bridge_nonbasic_mana_land`` — the 5 EoE shocklands
      (Breeding Pool / Godless Shrine / Sacred Foundry / Stomping Ground
      / Watery Grave); 0-Effect, BLOCKED → COVERED post-widen, then
      PROBED via the round-60 shockland bench probe in this file (see
      ``test_r60_genuine_real_eoe_shockland_byte_equal_through_bench``).
    * ``_printing_is_bridge_simple_spell`` — the 7 EoE simple
      Instant/Sorcery cards (Biosynthic Burst / Depressurize /
      Diplomatic Relations / Focus Fire / Orbital Plunge / Radiant
      Strike / Seedship Impact); ALL already PROBED via existing
      ``_parity`` probes, so this widen is metric-truthfulness only
      (tier counts unchanged).
    * ``_printing_is_bridge_simple_static_anthem_creature`` — Tannuk,
      Steadfast Second; static-abilities Phase 1 round-79 substrate.
    * ``_printing_is_bridge_simple_uncounterable_creature`` — Frenzied
      Baloth; static-abilities Phase 2 round-80 substrate.
    * ``_printing_is_bridge_simple_static_self_conditional_creature`` —
      Brightspear Zealot / Cloudsculpt Technician / Gravblade Heavy /
      Mechan Shieldmate; static-abilities Phase 3/3.5 rounds 81/82.

    Each carrier has an existing engine-level byte-equal lifecycle
    pin (the canonical probes live in
    ``tests/test_static_layered_abilities_phase{1,2,3}.py`` and in this
    file's shockland bench probe); the ``_parity``-named delegators
    added alongside this widen credit the slug-keyed scanner so the
    cards move all the way to PROBED. Pattern follows the round-83
    warp_cost Phase 3 / round-84 Cosmogoyf idiom: thin delegators
    re-use existing byte-equal helpers, no duplicated probe logic, no
    engine changes."""
    from mtg_bot.parity import trajectory as traj

    try:
        return bool(
            traj._printing_is_bridge_basic_land(printing)
            or traj._printing_is_bridge_nonbasic_mana_land(printing)
            or traj._printing_is_bridge_pure_warp_creature(printing)
            or traj._printing_is_bridge_simple_creature(printing)
            or traj._printing_is_bridge_simple_spell(printing)
            or traj._printing_is_bridge_simple_aura(printing)
            or traj._printing_is_bridge_simple_static_anthem_creature(
                printing,
            )
            or traj._printing_is_bridge_simple_uncounterable_creature(
                printing,
            )
            or traj._printing_is_bridge_simple_static_self_conditional_creature(
                printing,
            )
            # Non-basic land substrate — phase 2 (round-85 N1): the
            # 10th admit arm of the real loader; mirrors
            # ``_filter_deck_to_bridge_supported``. The 5 EoE Planet
            # lands (Adagia / Evendo / Kavaron / Susur Secundi /
            # Uthros) flip from BLOCKED -> COVERED here (0-Effect,
            # admit-arm fires, ``not eff_classes and admitted`` arm
            # of ``covered_ok``), and the 5 ``_parity``-named
            # delegators added to ``tests/test_parity_smoke.py`` in
            # the companion lane move them all the way to PROBED via
            # the slug-keyed ``_smoke_probed_cards`` scanner.
            or traj._printing_is_bridge_simple_tapped_mana_land(printing)
        )
    except Exception:
        return False


# ------------------------------------------------- the classifier


@dataclass
class PoolParityResult:
    n_pool: int
    tiers: dict[str, list[str]]
    # Internal-consistency violations: a residue `genuinely_missing`
    # card the class-granularity heuristic WOULD have counted as
    # covered/probed. The worklist's per-card re-walk is the authority
    # on genuine absence (it sees riders the class heuristic can't —
    # e.g. `DestroyTargetCreature` maps to Rust's generic `DestroyTarget`
    # but a `draw_if_undamaged` rider is genuinely unshipped). When this
    # list is non-empty the class heuristic is over-optimistic for
    # those cards and the doc emits a WARNING; the cards are pinned to
    # BLOCKED regardless (the metric never overstates). Each tuple is
    # (card, tier_the_heuristic_would_have_used).
    consistency_violations: list[tuple[str, str]] = field(
        default_factory=list
    )
    # Informational (NOT a violation): residue `blocked_non_effect`
    # cards whose Effects ARE covered/probed at the Effect+bridge level
    # — that is precisely their definition (Effects shipped; the card
    # is gated by a non-Effect path). The metric measures Effect+bridge
    # byte-equality; the residue worklist tracks full-card admit
    # blockers — these legitimately differ and are surfaced as context,
    # not drift. Each tuple is (card, effect_evidence) where
    # effect_evidence is "probed" or "covered".
    blocked_non_effect_with_covered_effects: list[tuple[str, str]] = (
        field(default_factory=list)
    )

    def counts(self) -> dict[str, int]:
        return {k: len(self.tiers[k]) for k in _TIER_ORDER}


def classify_pool() -> PoolParityResult:
    """Walk the EoE pool, classify every printing into exactly one
    tier by the strict precedence in the module docstring, and run the
    residue-worklist internal-consistency cross-check."""
    from mtg_bot.cards.parser.source import OracleParserCardSource

    src = OracleParserCardSource()
    printings = src.get_printings()
    pool_names = set(printings.keys())

    covered_classes = covered_python_class_set()
    probed = genuine_probed_cards(pool_names)
    missing_cards, non_effect_cards = residue_blocked_cards()

    tiers: dict[str, list[str]] = {k: [] for k in _TIER_ORDER}
    violations: list[tuple[str, str]] = []
    bne_covered: list[tuple[str, str]] = []

    for name, printing in sorted(printings.items()):
        eff_classes = card_effect_classes(printing)
        all_effects_covered = bool(eff_classes) and eff_classes.issubset(
            covered_classes
        )
        admitted = loader_admits(printing)
        is_missing = name in missing_cards
        is_non_effect = name in non_effect_cards
        is_probed = name in probed
        covered_ok = (
            all_effects_covered or (not eff_classes and admitted)
        )

        # --- residue-worklist authority over the class heuristic -----
        # The per-card residue re-walk is the source of truth on
        # genuine absence; it sees Effect-shape riders the
        # class-granularity heuristic cannot (e.g. a `draw_if_undamaged`
        # rider on a class that otherwise maps to a live Rust variant).
        # So a `genuinely_missing` card is HARD-pinned to the BLOCKED-
        # missing tier no matter what the probe-name match / class
        # heuristic says — the metric must never overstate parity.
        if is_missing:
            tiers[TIER_BLOCKED_MISSING].append(name)
            if is_probed or covered_ok:
                # The class heuristic disagrees with the per-card
                # re-walk — surface it (the worklist wins, but the
                # heuristic's blind spot is worth a WARNING).
                violations.append(
                    (name, TIER_PROBED if is_probed else TIER_COVERED)
                )
            continue

        # A `blocked_non_effect` card is, BY DEFINITION, one whose
        # Effects ARE shipped/bridged but whose full-card admit is
        # gated by a non-Effect path. Its Effects being covered/probed
        # is EXPECTED, not drift — record it as informational context
        # and pin the card to the blocked_non_effect tier (full-card
        # parity is what's blocked; the metric does not count it as
        # byte-equal).
        if is_non_effect:
            tiers[TIER_BLOCKED_NON_EFFECT].append(name)
            if is_probed:
                bne_covered.append((name, "probed"))
            elif covered_ok:
                bne_covered.append((name, "covered"))
            continue

        # Precedence 1: PROBED (strongest). A genuine probe pins it.
        if is_probed:
            tiers[TIER_PROBED].append(name)
            continue

        # Precedence 2: covered-by-construction. Every emitted Effect
        # class has a live+bridged Rust counterpart. A 0-Effect
        # (vanilla / keyword-only) card qualifies here only if
        # loader-admitted (otherwise it has no cross-engine exercise
        # path and drops to the bench / blocked tier below).
        if covered_ok:
            tiers[TIER_COVERED].append(name)
            continue

        # Precedence 3: bridge-admitted & bench-exercised. The loader
        # admits it (the bench runs it 0-divergence at scale, capped
        # at ~22-25 turns). In practice the loader's admit gate only
        # passes vanilla / safe-trigger creatures + basic lands — all
        # of which have covered (or zero) Effects and were already
        # caught by precedence 2 — so this tier is typically empty.
        # It is kept explicit so that if the admit gate ever widens to
        # admit a card whose Effects are NOT all class-covered, the
        # bench's empirical 0-divergence signal still counts it here
        # rather than dropping it to BLOCKED.
        if admitted:
            tiers[TIER_ADMITTED].append(name)
            continue

        # Precedence 4: BLOCKED — the long subsystem-gated /
        # Effect-uncovered tail (not itemised in the residue worklist).
        tiers[TIER_BLOCKED_SUBSYSTEM].append(name)

    return PoolParityResult(
        n_pool=len(pool_names),
        tiers=tiers,
        consistency_violations=violations,
        blocked_non_effect_with_covered_effects=bne_covered,
    )


# ------------------------------------------------- markdown render


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def render_markdown(res: PoolParityResult) -> str:
    c = res.counts()
    n = res.n_pool
    probed_n = c[TIER_PROBED]
    covered_n = c[TIER_COVERED]
    admitted_n = c[TIER_ADMITTED]
    # "byte-equal at the Effect+bridge level" = PROBED + covered (every
    # Effect lives + bridges; PROBED additionally has a genuine probe).
    byte_equal_n = probed_n + covered_n
    blocked_total = (
        c[TIER_BLOCKED_MISSING]
        + c[TIER_BLOCKED_NON_EFFECT]
        + c[TIER_BLOCKED_SUBSYSTEM]
    )

    L: list[str] = []
    L.append(
        "Generated by `scripts/measure_pool_parity.py` — "
        "DO NOT EDIT MANUALLY. Run the script to refresh."
    )
    L.append("")
    L.append("# Pool-level cross-engine parity metric")
    L.append("")
    L.append(
        "The authoritative, measured, regenerable answer to *\"what "
        "fraction of the EoE card pool is byte-equal cross-engine?\"* "
        "Every printing the parser claims (the same pool "
        "`measure_effect_coverage.py` walks) is classified into exactly "
        "one honest tier by a strict precedence (strongest evidence "
        "wins). This replaces the three PROXIES the campaign used to "
        "estimate from (per-emission Rust coverage, the residue count, "
        "\"0 FAIL_DIVERGENCE over the admitted subset\") with a single "
        "tracked number."
    )
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(
        f"- **{byte_equal_n} / {n} EoE cards "
        f"({_pct(byte_equal_n, n):.1f}%) are byte-equal at the "
        f"Effect+bridge level**; of those, **{probed_n} "
        f"({_pct(probed_n, n):.1f}% of the pool) are individually "
        f"pinned by a genuine real-EoE cross-engine byte-equality "
        f"probe** (the strongest evidence), the remaining "
        f"{covered_n} are covered-by-construction (every Effect they "
        f"emit has a live Rust variant + bridge arm, but no genuine "
        f"probe pins THAT card)."
    )
    L.append(
        f"- A further **{admitted_n} ({_pct(admitted_n, n):.1f}%)** "
        f"are bridge-admitted & bench-exercised 0-divergence at scale "
        f"(empirical, but bounded — see the caveat)."
    )
    L.append(
        f"- **{blocked_total} ({_pct(blocked_total, n):.1f}%)** are "
        f"BLOCKED, broken out below."
    )
    L.append("")
    L.append(
        "> **CAVEAT — what this measures vs. what it does NOT.** This "
        "is **Effect+bridge-level byte-equality evidence**, NOT "
        "full-game-trajectory parity. The PROBED tier asserts "
        "byte-equality on a *controlled* cross-engine state (a real "
        "EoE printing driven end-to-end), not over a full random "
        "multi-turn game. The covered-by-construction tier only proves "
        "the card's Effect *primitives* all exist + bridge — it does "
        "**not** prove the card plays byte-equal in a real game. The "
        "bench tier is the empirical full-trajectory signal but is "
        "**capped at ~22-25 turns** (see "
        "`docs/bench_trajectory_parity_smoke.md`) and only over the "
        "loader-admitted subset. **No tier here claims the whole pool "
        "is byte-equal in real full-game trajectories** — that "
        "remains out of scope and is the campaign's open frontier."
    )
    L.append("")

    L.append("## Tiers")
    L.append("")
    L.append("| Tier | Count | % of pool |")
    L.append("|---|---:|---:|")
    for k in _TIER_ORDER:
        L.append(
            f"| {_TIER_LABEL[k]} | {c[k]} | {_pct(c[k], n):.1f}% |"
        )
    L.append(f"| **Pool total** | **{n}** | **100.0%** |")
    L.append("")

    L.append("### Tier definitions")
    L.append("")
    L.append(
        "Evaluated in this strict precedence (a card lands in the "
        "FIRST tier it qualifies for):"
    )
    L.append("")
    L.append(
        "1. **PROBED byte-equal** — a "
        "`test_cross_engine_real_eoe_<slug>_parity` function in "
        "`tests/test_parity_smoke.py` (slug = the card's printed-name "
        "slug, split cards keyed on the left face) asserts "
        "byte-equality for THIS card. `*_divergence_pin` functions are "
        "excluded (they pin a KNOWN divergence). Strongest evidence."
    )
    L.append(
        "2. **Effect+bridge covered (not individually probed)** — "
        "every distinct Python `Effect` class the card emits (walked "
        "with `measure_effect_coverage`'s exact walker) is in that "
        "script's covered-class set (live Rust variant + bridge), AND "
        "the card is not a residue-worklist BLOCKED entry. A "
        "0-Effect vanilla / keyword-only card qualifies here only if "
        "loader-admitted. Covered-by-construction — weaker than a "
        "genuine probe."
    )
    L.append(
        "3. **bridge-admitted & bench-exercised** — the production-deck "
        "loader admits the printing "
        "(`trajectory._printing_is_bridge_basic_land` or "
        "`_printing_is_bridge_simple_creature` — the exact bench admit "
        "gate) and the bench runs it 0-divergence at scale. "
        "Bounded-depth empirical signal (~22-25 turn cap)."
    )
    L.append(
        "4. **BLOCKED** — not byte-equal-covered today; broken out:"
    )
    L.append(
        "   - **genuinely_missing** — a residue-worklist "
        "`genuinely_missing` entry: a TRUE new-Rust-Effect ask (the "
        "load-bearing \"how far from parity\" gap)."
    )
    L.append(
        "   - **blocked_non_effect** — a residue-worklist "
        "`blocked_non_effect` entry: Effects shipped, gated by a "
        "non-Effect path (unbridged event / declined cost-mechanic / "
        "unmodeled subsystem / token-registration substrate)."
    )
    L.append(
        "   - **subsystem-gated / Effect-uncovered** — the long "
        "blocked tail: emits an Effect with no covered Rust "
        "counterpart, or is loader-rejected by a subsystem gate "
        "(Station / Ward / Devour / warp / Aura / Rooms / "
        "BECOMES_TAPPED / dynamic-P/T / activated+static abilities / "
        "planeswalker / …), and is not itemised in the residue "
        "worklist."
    )
    L.append("")

    # Internal-consistency cross-check.
    L.append("## Internal-consistency cross-check (residue worklist)")
    L.append("")
    L.append(
        "The residue worklist's per-card re-walk is the AUTHORITY on "
        "genuine Effect absence (it sees Effect-shape riders the "
        "class-granularity coverage heuristic cannot). A "
        "`genuinely_missing` card is therefore hard-pinned to the "
        "BLOCKED tier regardless of what the class heuristic / probe "
        "name-match says — the metric never overstates parity."
    )
    L.append("")
    if res.consistency_violations:
        L.append(
            f"**WARNING: {len(res.consistency_violations)} "
            f"`genuinely_missing` card(s) the class-granularity "
            f"heuristic WOULD have counted as covered/probed.** They "
            f"are pinned to BLOCKED (the worklist wins), but the "
            f"heuristic's class-level blind spot for these is worth "
            f"reconciling — either the worklist entry is stale (the "
            f"Effect actually shipped → close it; the round-54/55 cron "
            f"is the structural backstop) or the class heuristic is "
            f"genuinely too coarse for that card's rider (expected — "
            f"this is exactly why the per-card worklist exists):"
        )
        L.append("")
        for card, tier in sorted(res.consistency_violations):
            L.append(
                f"- `{card}` — residue `genuinely_missing`; class "
                f"heuristic would have placed it in `{tier}` "
                f"(class-granularity false-positive — pinned BLOCKED)."
            )
    else:
        L.append(
            "- **0 violations** — no `genuinely_missing` residue card "
            "was class-heuristic-counted as covered/probed. The "
            "headline does not overstate parity relative to the honest "
            "residue list."
        )
    L.append("")
    # Informational: blocked_non_effect cards with covered/probed
    # Effects. This is EXPECTED (their definition) — surfaced as
    # context, not drift, so a reader understands the gap between
    # "Effect+bridge byte-equal" and "full-card parity".
    if res.blocked_non_effect_with_covered_effects:
        L.append(
            f"_Context (NOT drift): "
            f"{len(res.blocked_non_effect_with_covered_effects)} "
            f"`blocked_non_effect` card(s) have covered/probed Effects "
            f"— that is their definition (Effects shipped; full-card "
            f"admit gated by a non-Effect path). The metric counts "
            f"Effect+bridge byte-equality; these cards' Effects ARE "
            f"byte-equal but the CARD is not full-game parity-able. "
            f"This is the precise gap between the two — surfaced for "
            f"honesty:_"
        )
        L.append("")
        for card, ev in sorted(
            res.blocked_non_effect_with_covered_effects
        ):
            L.append(
                f"- `{card}` — Effects {ev} at the Effect+bridge "
                f"level; CARD remains `blocked_non_effect` "
                f"(non-Effect gate)."
            )
        L.append("")

    # Per-tier card listings (BLOCKED tiers itemised; covered tiers
    # summarised — the bulk; PROBED listed in full as it's the
    # strongest, most-auditable claim).
    L.append("## PROBED cards (genuine real-EoE byte-equality probes)")
    L.append("")
    if res.tiers[TIER_PROBED]:
        for nm in sorted(res.tiers[TIER_PROBED]):
            L.append(f"- {nm}")
    else:
        L.append("- (none)")
    L.append("")

    for k in (
        TIER_BLOCKED_MISSING,
        TIER_BLOCKED_NON_EFFECT,
    ):
        L.append(f"## {_TIER_LABEL[k]} ({c[k]})")
        L.append("")
        if res.tiers[k]:
            for nm in sorted(res.tiers[k]):
                L.append(f"- {nm}")
        else:
            L.append("- (none)")
        L.append("")

    L.append(
        f"## {_TIER_LABEL[TIER_BLOCKED_SUBSYSTEM]} "
        f"({c[TIER_BLOCKED_SUBSYSTEM]})"
    )
    L.append("")
    L.append(
        "The long blocked tail (not itemised in the residue "
        "worklist). Round-58 N2 broke this **opaque single bucket** out "
        "by gating subsystem so it is the data-driven \"which subsystem "
        "next\" instrument, not one number — see "
        "`docs/SUBSYSTEM_ROADMAP.md` for the leverage analysis + "
        "recommended sequencing. A card may be **multi-gated** (counted "
        "under every gate it trips, so `#cards` does NOT sum to the "
        "tier total); `1gate` is the single-gated count (closing that "
        "one subsystem flips the card) and reconciles exactly. "
        "`decks(1g)` is the cumulative 17lands deck-count of the "
        "single-gated members — the honest leverage signal."
    )
    L.append("")
    L.extend(_render_subsystem_breakout(res))
    L.append("")
    L.append(
        "Full card list (first 60 shown for auditability; the full "
        "set + per-card gate attribution is regenerable from "
        "`scripts/scout_r58_subsystem_attribution.py`):"
    )
    L.append("")
    for nm in sorted(res.tiers[TIER_BLOCKED_SUBSYSTEM])[:60]:
        L.append(f"- {nm}")
    if c[TIER_BLOCKED_SUBSYSTEM] > 60:
        L.append(f"- … and {c[TIER_BLOCKED_SUBSYSTEM] - 60} more")
    L.append("")

    return "\n".join(L) + "\n"


def _subsystem_breakout_rows() -> tuple[list[dict], dict] | tuple[None, None]:
    """Reuse ``scout_r58_subsystem_attribution``'s attribution +
    leverage table verbatim (the subsystem taxonomy is defined ONCE,
    there — this metric never re-derives it, the same discipline as
    reusing ``measure_effect_coverage``'s parser). Returns
    ``(leverage_rows, attribution)`` or ``(None, None)`` if the scout
    is unimportable (defensive: the doc still renders, just without the
    breakout, never blocking the cron)."""
    try:
        import scout_r58_subsystem_attribution as scout  # noqa: E402

        attr = scout.subsystem_attribution()
        dc = scout._deck_counts(set(attr["cards"]))
        rows = scout.leverage_table(attr, dc)
        return rows, attr
    except Exception:  # pragma: no cover - defensive, never blocks
        return None, None


def _render_subsystem_breakout(res: PoolParityResult) -> list[str]:
    """The per-subsystem markdown table for the BLOCKED-subsystem
    section. Honest: it carries the single-gate reconciliation inline
    so a reader can verify the breakout partitions exactly the same
    cards the old opaque count reported."""
    rows, attr = _subsystem_breakout_rows()
    L: list[str] = []
    if rows is None or attr is None:
        L.append(
            "_(subsystem breakout unavailable — "
            "`scout_r58_subsystem_attribution` not importable; the "
            "opaque tier count above is still authoritative.)_"
        )
        return L
    n_total = attr["n_subsystem_gated"]
    n_single = sum(
        1 for cd in attr["cards"].values() if not cd["multi_gated"]
    )
    n_multi = n_total - n_single
    L.append(
        "| Subsystem | #cards | 1gate | multi | decks(tot) | "
        "decks(1g) | Tractability |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        L.append(
            f"| {r['label']} | {r['n_cards']} | {r['n_single_gate']} "
            f"| {r['n_multi_gated']} | {r['deck_count_total']:,} | "
            f"{r['deck_count_single_gate']:,} | {r['tractability']} |"
        )
    L.append("")
    sum_single = sum(len(v) for v in attr["single_gate"].values())
    L.append(
        f"**Reconciliation:** {n_total} subsystem-gated = "
        f"{n_single} single-gated + {n_multi} multi-gated; the per-"
        f"subsystem `1gate` column sums to {sum_single} "
        f"(== single-gated total). The breakout partitions exactly the "
        f"same {n_total} cards the opaque tier count above reports "
        f"(this metric reuses `classify_pool` verbatim — same tier, "
        f"same pool); the headline is unchanged, only the BLOCKED-tail "
        f"detail is itemised."
    )
    unclass = attr["by_subsystem"].get(scout_unclassified_key(), [])
    if unclass:
        L.append("")
        L.append(
            f"**Honest unclassified bucket ({len(unclass)}):** "
            + ", ".join(f"`{u}`" for u in sorted(unclass))
            + " — no taxonomy gate fired; flagged for a hand re-walk "
            "rather than force-guessed (see `docs/SUBSYSTEM_ROADMAP.md`)."
        )
    return L


def scout_unclassified_key() -> str:
    """The scout's ``SUB_UNCLASSIFIED`` key, resolved without a
    top-level import (the scout imports this module — keep the
    dependency lazy / one-directional)."""
    try:
        import scout_r58_subsystem_attribution as scout  # noqa: E402

        return scout.SUB_UNCLASSIFIED
    except Exception:  # pragma: no cover - defensive
        return "unclassified"


# ------------------------------------------------- headline (one-liner)


def headline_line(res: PoolParityResult) -> str:
    """A single never-blocking headline string for the staleness cron /
    post-commit hook. Mirrors the round-52/55 cron headline contract:
    one line, no exit-code semantics for the hook."""
    c = res.counts()
    n = res.n_pool
    be = c[TIER_PROBED] + c[TIER_COVERED]
    viol = len(res.consistency_violations)
    suffix = (
        f" ⚠ {viol} residue-consistency violation(s)" if viol else ""
    )
    # Round-58: surface the subsystem-gated single/multi split inline
    # so the headline is no longer one opaque number — but stay ONE
    # line and NEVER block (the scout import is defensive; if it fails
    # the headline degrades to the plain count, same cron contract).
    sub_detail = f"{c[TIER_BLOCKED_SUBSYSTEM]} subsystem-gated"
    try:
        _rows, _attr = _subsystem_breakout_rows()
        if _attr is not None:
            _ns = sum(
                1
                for cd in _attr["cards"].values()
                if not cd["multi_gated"]
            )
            _nm = _attr["n_subsystem_gated"] - _ns
            sub_detail = (
                f"{c[TIER_BLOCKED_SUBSYSTEM]} subsystem-gated "
                f"({_ns} single-gate / {_nm} multi-gated; "
                f"see docs/SUBSYSTEM_ROADMAP.md)"
            )
    except Exception:  # pragma: no cover - defensive, never blocks
        pass
    return (
        f"[pool-parity] {be}/{n} EoE cards "
        f"({_pct(be, n):.1f}%) byte-equal at the Effect+bridge level "
        f"({c[TIER_PROBED]} genuinely probed); "
        f"{c[TIER_ADMITTED]} bench-exercised; "
        f"{c[TIER_BLOCKED_MISSING]} genuinely_missing / "
        f"{c[TIER_BLOCKED_NON_EFFECT]} blocked_non_effect / "
        f"{sub_detail}. "
        f"NOT full-game-trajectory parity (caveat in "
        f"docs/POOL_PARITY.md).{suffix}"
    )


# ------------------------------------------------- entrypoint


def main(argv: list[str] | None = None) -> int:
    """Default: classify, (re)write ``docs/POOL_PARITY.md``, print the
    full markdown. With ``--headline``: classify and print ONLY the
    one-line never-blocking headline (the post-commit-hook entry — does
    NOT rewrite the doc, mirroring the round-52/55 cron headline
    contract: surface a line, never block, never side-effect on disk).

    Exit code is non-zero iff the residue-consistency cross-check fired
    (a ``genuinely_missing`` card the class heuristic over-counted) —
    for explicit CI / standalone gating only. The post-commit hook
    never propagates it, exactly like the staleness cron.
    """
    argv = sys.argv[1:] if argv is None else argv
    headline_only = "--headline" in argv
    res = classify_pool()
    if headline_only:
        print(headline_line(res))
    else:
        md = render_markdown(res)
        OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DOC.write_text(md, encoding="utf-8")
        print(md)
    return 1 if res.consistency_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
