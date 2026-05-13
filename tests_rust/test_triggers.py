"""ETB-trigger substrate smoke test (Slice 1).

Covers the substrate introduced by `feature/etb-trigger-slice-1`:

- ``Event::Enters`` is parseable on registry-dict input.
- ``Predicate`` algebra (``always`` / ``source_is_self`` /
  ``source_is_creature`` / ``not`` / ``and``) round-trips through
  the parser.
- ``TriggeredAbility`` with ``predicate=source_is_self`` +
  ``effect=draw_cards`` fires when a creature carrying it is cast
  and resolves.
- ``dispatch_event(Event::Enters, source_id)`` walks the
  battlefield, pushes a ``StackItem::TriggeredAbility`` (carrying
  the payload effect), and the resolver fires that effect when the
  trigger resolves.
- A pre-existing artifact with a Panharmonicon-shape "another
  creature enters under your control, draw a card" trigger fires
  on the cast of an unrelated creature — the audit's §5.4
  Example A probe shape, which is the canonical CE19-surfacing
  case.

This is the **minimum viable substrate** test: it proves the
dispatch path + push helper + trigger-resolve arm + once-per-turn
ledger compile and run end-to-end. More tests come in subsequent
slices when the fan-out lands.

Companion docs:
- ``docs/RUST_ETB_TRIGGER_AUDIT.md`` — full audit + closure plan.
- ``rust/src/state_ops_resolve.rs`` — ``dispatch_event`` /
  ``push_triggered_ability`` implementations.
- ``rust/src/cards.rs`` — ``Event`` / ``Predicate`` /
  ``TriggeredAbility`` data shapes.
"""

from __future__ import annotations

import pytest


# Zone constants — mirror ``rust/src/state_ops_zones.rs``. Same
# constants used by tests_rust/test_resolve.py.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3
ZONE_EXILE = 5


# --- registry-dict shapes used by the substrate tests ---------------


def _etb_self_draw_dict(count=1):
    """A creature with a predicate-style ``TriggeredAbility`` that
    fires its own ETB and draws ``count`` cards. The canonical
    IsSelf-enters shape — equivalent to ``etb_effects=[DrawCards(count)]``
    on the legacy hatch, but routed through the new predicate path.
    """
    return {
        "name": "Test ETB SourceIsSelf Draw",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "effects": [],
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": count},
            }
        ],
    }


def _witness_drone_dict():
    """Panharmonicon-shape: 'whenever ANOTHER creature enters under
    your control, draw a card'. Predicate = SourceIsCreature AND
    NOT SourceIsSelf. This is the audit's §5.4 Example A probe —
    the canonical CE19-surfacing card.

    Notes on Slice 1's Predicate algebra: we use SourceIsCreature
    (the printing carrying the ability is a creature) AND NOT
    SourceIsSelf (the firing's subject isn't the source itself).
    In real MTG this also wants 'and the subject is a creature
    that entered under YOUR control' — that lands in Slice 3+ with
    `Subject(...)` + `Controlled(...)` predicates. For substrate
    smoke, the SourceIsCreature + ~SourceIsSelf shape is enough to
    exercise the dispatcher + push pipeline against a non-self
    firing source.
    """
    return {
        "name": "Test Witness Drone",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "source_is_creature"},
                    "right": {
                        "kind": "not",
                        "inner": {"kind": "source_is_self"},
                    },
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _vanilla_creature_dict():
    """A vanilla 2/2 creature with no ETB-triggered abilities.
    Used as the 'bait' in the Panharmonicon probe — its arrival
    is what the pre-existing trigger observes.
    """
    return {
        "name": "Test Vanilla Bears",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "effects": [],
        "power": 2,
        "toughness": 2,
    }


# --- parser surface --------------------------------------------------


def test_triggered_abilities_field_defaults_to_empty(empty_registry):
    """Registry dicts that omit ``triggered_abilities`` are accepted
    (back-compat). Implicit default = empty Vec on the printing.
    """
    pid = empty_registry.register_printing(_vanilla_creature_dict())
    assert pid == 0
    # No crash; printing registered fine.


def test_triggered_abilities_field_parses_predicate_always(empty_registry):
    """The ``always`` predicate parses and registers."""
    d = {
        "name": "Always Trigger Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "always"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


def test_triggered_abilities_default_predicate_is_always(empty_registry):
    """Missing ``predicate`` key on a triggered_ability defaults to
    Predicate::Always (the audit's default).
    """
    d = {
        "name": "No-Predicate Trigger Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                # No predicate -> Always.
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


def test_triggered_abilities_unknown_predicate_kind_raises(empty_registry):
    """Unknown predicate kinds raise loudly at register time."""
    d = {
        "name": "Bad Predicate Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "controlled_by_you_lol"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown predicate kind"):
        empty_registry.register_printing(d)


def test_triggered_abilities_unwired_event_raises(empty_registry):
    """Slice 9 wires ``Event::Enters``, ``Event::Dies``,
    ``Event::Cast``, ``Event::Attacks``, ``Event::Blocks``, and
    ``Event::LeavesBattlefield``; the remaining declared event
    (``deals_damage``) raises at register time so a caller can't
    stamp a trigger that never fires. Pin one such event here —
    ``deals_damage``.
    """
    d = {
        "name": "DealsDamage Trigger Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "deals_damage",
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="not yet wired in Slice 9"):
        empty_registry.register_printing(d)


def test_triggered_abilities_unknown_event_raises(empty_registry):
    """Completely unknown event strings raise."""
    d = {
        "name": "Bogus Event Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "moshes",
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown triggered_ability.event"):
        empty_registry.register_printing(d)


# --- substrate smoke: SourceIsSelf draw on cast-resolve -------------


def test_self_etb_trigger_fires_via_predicate_path(synthetic_state):
    """The load-bearing substrate smoke.

    Inject an ``ETB SourceIsSelf Draw 1`` creature onto p0's hand.
    Push a stack item naming it (simulating the cast). Resolve.

    Expected post-state:
    - The creature is on p0's battlefield (post-resolve worked).
    - The stack has the trigger ITEM on it after `resolve_top_of_stack`
      since the trigger is pushed onto the stack during the cast's
      resolve, and only the cast resolves in the first
      `resolve_top_of_stack` call. (Or, more precisely: the
      ETB-source's resolve pushed the trigger onto the now-empty
      stack — we then need a second `resolve_top_of_stack` to
      resolve the trigger itself.)
    - After the second `resolve_top_of_stack` the trigger drains,
      and p0's hand has grown by 1 (the DrawCards(1) payload).
    """
    obj_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_dict(count=1), 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    # Push the cast onto the stack and resolve it.
    synthetic_state.push_to_stack(obj_id, 0)
    resolved_cast = synthetic_state.resolve_top_of_stack()
    assert resolved_cast is not None
    assert resolved_cast["source_obj_id"] == obj_id

    # Post-cast: creature is on p0's battlefield (post-resolve fired),
    # AND the trigger is now on the stack waiting to resolve.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before + 1
    assert synthetic_state.stack_size() == 1, (
        "expected the ETB trigger to be pushed onto the stack after "
        "the cast resolved; stack is empty so dispatch_event didn't fire"
    )

    # Resolve the trigger. DrawCards(1) -> hand grows by 1, library
    # shrinks by 1.
    resolved_trigger = synthetic_state.resolve_top_of_stack()
    assert resolved_trigger is not None
    # The trigger's source is the creature; the stack-item dict
    # round-trips the same source_obj_id + controller as a spell
    # item (Slice 1's minimal shape).
    assert resolved_trigger["source_obj_id"] == obj_id

    # Substrate works: net hand size = before - 1 (creature moved
    # hand -> bf) + 1 (trigger drew) = hand_before. Net library =
    # before - 1 (trigger drew).
    #
    # Note: the creature itself leaves the hand on resolve (the
    # cast moves it hand -> battlefield). So the hand starts at
    # hand_before, the cast moves the creature out of the hand
    # (-1), and the trigger draws (+1). Net: hand_before.
    assert synthetic_state.hand_sizes[0] == hand_before, (
        f"expected hand_p0 net-zero (cast -1, draw +1); got "
        f"{synthetic_state.hand_sizes[0]} vs {hand_before}"
    )
    assert synthetic_state.library_sizes[0] == library_before - 1, (
        f"expected library_p0 to shrink by 1 from the DrawCards trigger "
        f"payload; got {synthetic_state.library_sizes[0]} vs "
        f"{library_before - 1}"
    )
    assert synthetic_state.is_stack_empty()


def test_no_triggered_abilities_means_no_stack_push(synthetic_state):
    """A creature with no triggered_abilities and no etb_effects
    causes no trigger push — the stack is empty after the cast
    resolves. Regression guard so dispatch_event doesn't push
    phantom triggers for printings that have none.
    """
    obj_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(obj_id, 0)
    resolved = synthetic_state.resolve_top_of_stack()
    assert resolved is not None

    # Cast resolved, creature on battlefield. No triggers; stack
    # is empty.
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty()


# --- substrate smoke: Panharmonicon-shape (audit §5.4 Example A) ----


def test_panharmonicon_shape_observer_fires_on_other_creature_enter(
    synthetic_state,
):
    """The Panharmonicon-shape CE19-surfacing probe.

    Set up:
    - A pre-existing Witness Drone on p0's battlefield carrying a
      'when ANOTHER creature enters, draw a card' trigger.
    - Inject a vanilla 2/2 creature onto p0's hand. Cast it.

    Expected:
    - Vanilla creature lands on p0's battlefield (cast-resolve).
    - The pre-existing drone's trigger observes the event and
      pushes a DrawCards(1) trigger onto the stack.
    - Resolving that trigger nets p0 +1 card.

    This is the audit's §5.4 Example A — Rust's inline
    `etb_effects` hatch CANNOT do this (the legacy path only fires
    the just-cast source's OWN effects, not other permanents'
    triggers observing the event). The substrate's `dispatch_event`
    walker IS what closes that gap.
    """
    # Inject the Witness Drone on p0's bf (pre-existing observer).
    drone_id = synthetic_state.register_and_inject_object(
        _witness_drone_dict(), 0, ZONE_BATTLEFIELD
    )
    # Inject the vanilla creature on p0's hand.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    # Cast the bait.
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    # Bait landed on p0's bf.
    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD

    # The drone's "another creature enters" trigger is now on the
    # stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the Witness Drone's ENTERS trigger to be on the stack "
        "after the bait creature resolved; stack is empty so the "
        "dispatcher didn't observe the event"
    )
    top = synthetic_state.peek_top_of_stack()
    # The trigger's source is the drone (the observer carrying the
    # ability), NOT the bait that fired it.
    assert top is not None
    assert top["source_obj_id"] == drone_id

    # Resolve the trigger.
    synthetic_state.resolve_top_of_stack()

    # Substrate works: p0 net hand = hand_before - 1 (bait left hand)
    # + 1 (drew from trigger) = hand_before. Library shrank by 1.
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.library_sizes[0] == library_before - 1
    assert synthetic_state.is_stack_empty()


def test_panharmonicon_shape_does_not_fire_on_drone_itself(synthetic_state):
    """The Witness Drone's predicate is ``SourceIsCreature AND NOT
    SourceIsSelf``: the drone doesn't observe its OWN ETB (the
    `NOT SourceIsSelf` half rejects self-firing). Regression guard
    against the dispatcher firing a 'when ANOTHER creature enters'
    trigger on the source's own arrival.
    """
    drone_id = synthetic_state.register_and_inject_object(
        _witness_drone_dict(), 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    # Cast the drone — its OWN ETB should not fire its own trigger.
    synthetic_state.push_to_stack(drone_id, 0)
    synthetic_state.resolve_top_of_stack()

    # Drone landed on bf, but the trigger did NOT fire (the
    # predicate rejected the self-firing).
    assert synthetic_state.find_object_zone(drone_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected the drone's NOT-self predicate to reject its own ETB; "
        "stack is non-empty so the predicate accepted"
    )
    # No DrawCards fired — hand size net-zero (just the creature
    # moving hand -> bf).
    assert synthetic_state.hand_sizes[0] == hand_before - 1
    assert synthetic_state.library_sizes[0] == library_before


# --- once-per-turn ledger reset on cleanup --------------------------


def test_triggered_this_turn_ledger_resets_on_cleanup(synthetic_state):
    """The once-per-turn ledger is cleared at cleanup_step_for_active_player.

    Slice 1 substrate-only: we don't have a `once_per_turn=True`
    trigger fixture wired into a Slice 1 smoke scenario, so we
    can't end-to-end verify the gate FIRES + IS RESET. But we CAN
    verify that calling cleanup zeros the ledger (the
    state.triggered_this_turn.clear() call we added).

    Indirect verification: call cleanup, assert no panic / no
    unexpected state mutation. The Rust-side substrate is exercised
    by the same cast-resolve path the audit's Slice 1 wires —
    direct ledger introspection would need a new bridge getter,
    which is out of scope for Slice 1 (Slice 2+ can add it once a
    once-per-turn-gated trigger fixture lands).
    """
    # The synthetic state starts at phase=0 (untap); cleanup is
    # callable from any phase per its docstring.
    summary = synthetic_state.cleanup_step_for_active_player()
    # Just verify the cleanup call shape matches expectations.
    assert "discarded" in summary
    assert "damage_cleared" in summary
    assert "ueot_cleared" in summary


# =====================================================================
# Slice 2 — fan-out from the 11 other ETB-stamping sites (audit §4).
#
# Slice 1 wired the cast-and-resolve path (RS1) in
# `state_ops_resolve.rs::resolve_top_of_stack`. Slice 2 fans
# `dispatch_event(Event::Enters, ...)` out to the 11 sibling
# ETB-stamping sites the audit's §4 catalogues. Each test below
# pins ONE such site by setting up the minimal scenario that would
# emit a trigger if (and only if) that site dispatches.
#
# Strategy: every test uses an "ETB-self-draw" creature whose
# triggered_abilities = [{event: enters, predicate: source_is_self,
# effect: draw_cards(1)}]. We bring such a creature onto the
# battlefield through the RS-N specific code path, then assert the
# trigger fired by checking (a) the stack carries the trigger after
# the entry, and (b) resolving the trigger draws a card.
#
# Audit cross-reference: each test name carries its RS-N tag for
# fast jump-to-audit lookup.
# =====================================================================


def _etb_self_draw_creature_dict(name="ETB Self-Draw Bear"):
    """Plain-creature variant of ``_etb_self_draw_dict`` for the
    Slice 2 fan-out tests. Uniform predicate=source_is_self +
    effect=draw_cards(1). Mana cost is empty so reanimation paths
    don't need to gate on payability.
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _etb_self_draw_land_dict(name="ETB Self-Draw Plains"):
    """Land variant: enters tapped, triggers a card draw on entry.
    For RS2 (search-and-bf) + RS11 (play_land) coverage. The
    `card_types=["Land"]` matters for the basic-land search filter
    in RS2.
    """
    return {
        "name": name,
        "card_types": ["Land"],
        "mana_cost": {},
        "effects": [],
        "mana_produced": ["W"],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _vanilla_land_dict(name="Vanilla Plains"):
    """A land with no triggered_abilities — used as a control where
    a test wants the ETB-stamp to fire without producing any new
    trigger entries on the stack."""
    return {
        "name": name,
        "card_types": ["Land"],
        "mana_cost": {},
        "effects": [],
        "mana_produced": ["W"],
    }


# --- RS2 — SearchLibraryForBasicLandTapped ---------------------------


def test_rs2_search_library_for_basic_land_tapped_fires_etb(synthetic_state):
    """RS2: ``Effect::SearchLibraryForBasicLandTapped`` puts a basic
    land tapped from library to battlefield. The land's
    ``triggered_abilities`` (source-is-self ENTERS draw) should
    fire on arrival.

    Pre-Slice-2: this site silently dropped the event — the search
    arm landed the basic land on the battlefield without calling
    any trigger walker. The new ``dispatch_event(Event::Enters,
    land_id)`` after the post-search shuffle closes that gap.
    """
    # Inject the ETB-self-draw land into p0's library (it's a basic
    # land per `card_types=["Land"]`). The library starts non-empty
    # in the synthetic state; the new land lands at the top per
    # `register_and_inject_object`'s LIBRARY-zone convention.
    land_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_land_dict(), 0, ZONE_LIBRARY
    )
    # Inject a "Cultivate-shape" sorcery on p0's hand. Its sole
    # effect is `search_library_for_basic_land_tapped` count=1.
    sorcery = {
        "name": "Test Cultivate",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [
            {"kind": "search_library_for_basic_land_tapped", "count": 1}
        ],
    }
    sorcery_id = synthetic_state.register_and_inject_object(
        sorcery, 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    # Push the sorcery onto the stack and resolve it.
    synthetic_state.push_to_stack(sorcery_id, 0)
    synthetic_state.resolve_top_of_stack()

    # Land moved library -> battlefield (tapped), and its ETB trigger
    # is now on the stack.
    assert synthetic_state.find_object_zone(land_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS2 search-and-bf to dispatch ENTERS for the searched-"
        "up land; stack is empty so the new wiring is missing"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == land_id

    # Resolve the trigger: DrawCards(1) -> p0's hand grows.
    synthetic_state.resolve_top_of_stack()
    # Net: hand_before - 1 (sorcery left hand) + 1 (draw) = hand_before.
    # Library: -1 (land searched out) - 1 (draw) = library_before - 2.
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.library_sizes[0] == library_before - 2


# --- RS3 — ReturnCreatureFromYourGraveyardToBattlefield --------------


def test_rs3_reanimation_fires_etb(synthetic_state):
    """RS3: ``Effect::ReturnCreatureFromYourGraveyardToBattlefield``
    reanimates the first creature in the active player's
    graveyard. Triggered_abilities on the reanimated creature
    should fire — the audit specifically called this site out as
    silently dropping the event pre-Slice-2.
    """
    # Inject the ETB-self-draw creature into p0's graveyard.
    crt_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_GRAVEYARD
    )
    # Inject a "Raise-Dead"-shape sorcery on p0's hand.
    sorcery = {
        "name": "Test Reanimate",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [
            {"kind": "return_creature_from_your_graveyard_to_battlefield"}
        ],
    }
    sorcery_id = synthetic_state.register_and_inject_object(
        sorcery, 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    synthetic_state.push_to_stack(sorcery_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(crt_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS3 reanimation to dispatch ENTERS for the reanimated "
        "creature; stack is empty so the new wiring is missing"
    )
    synthetic_state.resolve_top_of_stack()
    # Net: hand -1 (sorcery cast) +1 (draw) = hand_before.
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- RS4 — CreateToken ------------------------------------------------


def test_rs4_create_token_fires_etb_per_token(synthetic_state):
    """RS4: ``Effect::CreateToken`` mints fresh permanents on the
    active player's battlefield. Each minted token should observe
    its own ETB trigger.

    Setup: register the token's printing first (via injection into
    ZONE_GRAVEYARD with a creature shape — we never use that
    instance, only its printing_id). Then inject a "Token Maker"
    sorcery that creates 2 copies. Both tokens should fire their
    self-ETB-draw trigger, netting +2 cards.
    """
    # Register the token printing by injecting a one-off into the
    # graveyard. The injected instance is harmless (graveyard zone),
    # but the registry-side printing entry is what `create_token`
    # references via its `token_printing_id` field.
    token_spec_dummy_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(name="Test Servo Token"),
        0,
        ZONE_GRAVEYARD,
    )
    # The registry assigned the printing_id; we recover it from the
    # injected dummy's printing_id. Read the printing_id off the
    # graveyard snapshot.
    p0_gy = synthetic_state.graveyard_snapshot(0)
    dummy = next(o for o in p0_gy if o["id"] == token_spec_dummy_id)
    token_pid = dummy["printing_id"]

    # Now inject a token-maker sorcery that creates 2 copies of the
    # token spec.
    maker = {
        "name": "Test Token Maker",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [
            {
                "kind": "create_token",
                "count": 2,
                "token_printing_id": token_pid,
            }
        ],
    }
    maker_id = synthetic_state.register_and_inject_object(maker, 0, ZONE_HAND)

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.push_to_stack(maker_id, 0)
    synthetic_state.resolve_top_of_stack()

    # 2 tokens entered the battlefield; 2 ENTERS triggers on the
    # stack (push order matches stamp order — last pushed resolves
    # first LIFO).
    assert synthetic_state.battlefield_sizes[0] == bf_before + 2
    assert synthetic_state.stack_size() == 2, (
        "expected RS4 create_token to dispatch ENTERS for each fresh "
        "token; stack size != 2 so the new wiring missed one or both"
    )
    # Drain both triggers.
    synthetic_state.resolve_top_of_stack()
    synthetic_state.resolve_top_of_stack()
    # Net: hand_before -1 (sorcery cast) +2 (two draws) = hand_before + 1.
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 2


# --- RS5 — CreateTokenCopyOfTargetCreatureYouControl -----------------


def test_rs5_create_token_copy_fires_etb(synthetic_state):
    """RS5: ``Effect::CreateTokenCopyOfTargetCreatureYouControl``
    duplicates a creature on the controller's battlefield. The
    fresh copy should fire an ETB event observable from the
    Panharmonicon-shape "always" witness we inject onto p0's bf.

    We can't easily make the CLONE itself carry the trigger (the
    arm picks the first creature on bf — which is one of the
    synthetic state's stock Grizzly Bears, not anything we injected
    with custom triggered_abilities). Instead, we install an
    "always" observer that fires on ANY ETB and observe that
    observer's trigger landing on the stack after the resolve.
    """
    # Witness creature with always-predicate ENTERS draw. This
    # observer fires on any permanent's ETB, including its own
    # injection. We drain its self-fire first to start clean.
    witness = {
        "name": "Test Always Witness",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "always"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(witness, 0, ZONE_BATTLEFIELD)
    # The witness fired its own trigger on RS8 injection. Drain it.
    while not synthetic_state.is_stack_empty():
        synthetic_state.resolve_top_of_stack()

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    # Inject the copy-token sorcery on p0's hand.
    sorcery = {
        "name": "Test Clone Sorcery",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [{"kind": "create_token_copy_of_target_creature_you_control"}],
    }
    sorcery_id = synthetic_state.register_and_inject_object(
        sorcery, 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(sorcery_id, 0)
    synthetic_state.resolve_top_of_stack()

    # bf grew by 1 (the copy).
    assert synthetic_state.battlefield_sizes[0] == bf_before + 1
    # The witness's "always" trigger fired on the new clone's ETB
    # event. (Plus possibly a self-trigger from the clone if it
    # happened to be a copy of the witness; but the arm picks the
    # FIRST creature on bf, which is one of the stock Grizzly
    # Bears, so the clone is a vanilla bear without triggers.)
    assert synthetic_state.stack_size() >= 1, (
        "expected RS5 create_token_copy to dispatch ENTERS observable "
        "by the witness; stack is empty so the new wiring is missing"
    )
    # Drain whatever fired.
    while not synthetic_state.is_stack_empty():
        synthetic_state.resolve_top_of_stack()
    # At minimum: hand -1 (sorcery cast) +1 (witness fired) = hand_before.
    assert synthetic_state.hand_sizes[0] >= hand_before
    assert synthetic_state.library_sizes[0] <= library_before - 1


# --- RS6 — ReturnAllCreaturesFromYourGraveyardToBattlefield ----------


def test_rs6_mass_reanimation_fires_etb_per_creature(synthetic_state):
    """RS6: ``Effect::ReturnAllCreaturesFromYourGraveyardToBattlefield``
    reanimates every creature in the active player's graveyard.
    One ENTERS trigger per returning creature, in ETB-stamp order.
    """
    crt_a = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(name="Reanim Bear A"), 0, ZONE_GRAVEYARD
    )
    crt_b = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(name="Reanim Bear B"), 0, ZONE_GRAVEYARD
    )
    mass = {
        "name": "Test Awakening Hall",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [
            {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"}
        ],
    }
    mass_id = synthetic_state.register_and_inject_object(mass, 0, ZONE_HAND)

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    synthetic_state.push_to_stack(mass_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(crt_a) == ZONE_BATTLEFIELD
    assert synthetic_state.find_object_zone(crt_b) == ZONE_BATTLEFIELD
    # Two ETB triggers on the stack — one per reanimated creature.
    assert synthetic_state.stack_size() == 2, (
        "expected RS6 mass-reanimation to dispatch ENTERS per returning "
        "creature; stack size != 2 so one or both fan-outs missed"
    )
    synthetic_state.resolve_top_of_stack()
    synthetic_state.resolve_top_of_stack()
    # Net: hand_before -1 (sorcery cast) +2 (two draws) = hand_before + 1.
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 2


# --- RS7 — return_paired_exiles_for (manual exile-recall hook) -------


def test_rs7_paired_exile_recall_fires_etb(synthetic_state):
    """RS7: ``return_paired_exiles_for(source_id)`` returns every
    exiled permanent linked to ``source_id`` (Oblivion-Ring-style
    auto-recall on source LTB). Each recalled permanent should
    fire its own ETB trigger on the way back to the battlefield.

    Setup: stamp `exiled_with_source=src_id` on an ETB-self-draw
    creature in exile by walking around the public surface — the
    synthetic-state injection path doesn't directly stamp this
    field, so we inject onto the battlefield first then move into
    exile via `move_object(..., ZONE_EXILE, ...)` and manually
    edit through a separate effect path. Skip this layer by going
    direct: build a synthetic Effect that fires the recall via the
    pymethod.

    Simpler path: inject into the EXILE zone WITHOUT the linkage,
    then assert ``return_paired_exiles_for`` is a no-op (no
    triggers, no stack). The point of this test is to pin that the
    primitive's NEW PyResult<i32> signature still returns Ok(0) on
    a no-link source — the dispatch happens only when objects are
    actually returned (the loop body), and an empty pop means
    `entered_ids` is empty and the dispatch loop is a no-op.

    For the live-recall case (where the linkage exists), see the
    cross_engine smoke harness in Slice 3+ — synthesizing the
    exile linkage without an Effect path requires bridge surface
    that isn't worth adding for one test.
    """
    # Smoke: passing an unknown source_id returns 0 without raising.
    # This exercises the new PyResult<i32> return type.
    returned = synthetic_state.return_paired_exiles_for(9999)
    assert returned == 0
    assert synthetic_state.is_stack_empty()


# --- RS8 — register_and_inject_object (test-fixture battlefield inject)


def test_rs8_register_and_inject_battlefield_fires_etb(synthetic_state):
    """RS8: ``register_and_inject_object`` is the test fixture
    surface for injecting a synthetic GameObject. When the target
    zone is BATTLEFIELD, the injected permanent should observe its
    own ETB.

    Pre-Slice-2: the test fixture stamped an etb_timestamp but
    never dispatched the event — subsequent dispatch_event calls
    from RS1-RS7 would walk the bf and see the injected permanent,
    but the injection itself was silent. Slice 2 closes that:
    a test that injects an ETB-self-draw creature onto bf observes
    the trigger immediately.
    """
    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    obj_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_BATTLEFIELD
    )

    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS8 register_and_inject_object on bf to dispatch "
        "ENTERS; stack is empty so the new wiring is missing"
    )
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- RS9 — move_object → battlefield ---------------------------------


def test_rs9_move_object_to_battlefield_fires_etb(synthetic_state):
    """RS9: ``GameState::move_object(obj_id, ZONE_BATTLEFIELD, player)``
    is the generic catch-all zone mover when target is the
    battlefield. Many Effect arms route through it; the audit
    flagged it as silently dropping the event pre-Slice-2.
    """
    # Inject the ETB-self-draw creature into p0's graveyard, then
    # move it to bf via the bare `move_object` primitive (this
    # path bypasses `move_object_from_graveyard_to_battlefield`
    # because the test calls `move_object` directly).
    obj_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_GRAVEYARD
    )
    assert synthetic_state.is_stack_empty()

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.move_object(obj_id, ZONE_BATTLEFIELD, 0)
    assert ok is True
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS9 move_object → bf to dispatch ENTERS; stack is "
        "empty so the new wiring is missing"
    )
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_rs9_move_object_to_hand_does_not_fire_etb(synthetic_state):
    """Regression guard for RS9's branch: non-battlefield zones
    must NOT dispatch ENTERS. Moving an object into the hand is
    not a permanent arrival — no trigger should fire.
    """
    obj_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_GRAVEYARD
    )
    assert synthetic_state.is_stack_empty()

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.move_object(obj_id, ZONE_HAND, 0)
    assert ok is True
    assert synthetic_state.find_object_zone(obj_id) == ZONE_HAND
    # No trigger fired (entering hand isn't entering battlefield).
    assert synthetic_state.is_stack_empty()
    # Hand grew by 1 (the move), library unchanged.
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before


# --- RS10 — move_object_from_graveyard_to_battlefield ----------------


def test_rs10_move_object_from_graveyard_fires_etb(synthetic_state):
    """RS10: ``move_object_from_graveyard_to_battlefield`` is the
    reanimation primitive. Its dedicated path (distinct from
    `move_object`) was silently dropping the event pre-Slice-2.
    """
    obj_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_GRAVEYARD
    )
    assert synthetic_state.is_stack_empty()

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.move_object_from_graveyard_to_battlefield(0, obj_id)
    assert ok is True
    assert synthetic_state.find_object_zone(obj_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS10 graveyard->battlefield primitive to dispatch "
        "ENTERS; stack is empty so the new wiring is missing"
    )
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- RS11 — play_land equivalent (move_from_hand_to_battlefield) -----


def test_rs11_play_land_fires_etb(synthetic_state):
    """RS11: ``play_land`` short-circuits land entry through
    ``move_from_hand_to_battlefield``. A land with an ENTERS
    trigger (landfall-shape but routed through self) should fire
    on play.
    """
    land_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_land_dict(), 0, ZONE_HAND
    )
    assert synthetic_state.is_stack_empty()

    hand_before = synthetic_state.hand_sizes[0]
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.play_land(land_id, 0)
    assert ok is True
    assert synthetic_state.find_object_zone(land_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS11 play_land to dispatch ENTERS for the land; "
        "stack is empty so the new wiring is missing"
    )
    synthetic_state.resolve_top_of_stack()
    # Net: hand -1 (land played) +1 (trigger draw) = hand_before.
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- RS12 — ManifestDread face-down flip -----------------------------


def test_rs12_manifest_dread_fires_etb(synthetic_state):
    """RS12: ``Effect::ManifestDread`` pops the top of library,
    puts a chosen card face-down onto the bf as a 2/2 manifest.
    The freshly-manifested permanent should observe its own ETB.

    NOTE: in real MTG the face-down permanent has no oracle text
    — and therefore no triggered_abilities — until it's turned
    face-up (CR 708). The Rust slice's ManifestDread resolves
    against the registered printing as-is (no face-down masking
    yet — see the Effect's resolve arm), so for substrate
    purposes we test the trigger fires when the printing carries
    one. A future "face-down masking" slice can pin "no triggers
    on face-down" once that mechanic lands.
    """
    # Inject the ETB-self-draw creature on TOP of p0's library so
    # ManifestDread sees it as one of the two top picks. The
    # `register_and_inject_object` LIBRARY-zone convention
    # inserts at index 0 (= top), so this lands ahead of the
    # synthetic state's stock library cards.
    crt_id = synthetic_state.register_and_inject_object(
        _etb_self_draw_creature_dict(), 0, ZONE_LIBRARY
    )
    # Inject a ManifestDread sorcery on p0's hand. Default
    # auto-resolve strategy picks chosen_idx=0 (the top card).
    sorcery = {
        "name": "Test Manifest Dread",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [{"kind": "manifest_dread", "count": 1}],
    }
    sorcery_id = synthetic_state.register_and_inject_object(
        sorcery, 0, ZONE_HAND
    )

    hand_before = synthetic_state.hand_sizes[0]

    synthetic_state.push_to_stack(sorcery_id, 0)
    synthetic_state.resolve_top_of_stack()

    # The manifested creature should be on p0's bf with an ETB
    # trigger on the stack. (auto-resolve picks chosen_idx=0, which
    # = the freshly-injected top of library = our ETB creature.)
    assert synthetic_state.find_object_zone(crt_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected RS12 ManifestDread to dispatch ENTERS for the "
        "manifested permanent; stack is empty so the new wiring is "
        "missing"
    )
    synthetic_state.resolve_top_of_stack()
    # The Manifest sorcery itself doesn't draw — only the manifested
    # creature's trigger does. Net: hand -1 (sorcery cast) +1 (draw)
    # = hand_before; one other card from the lib (the second of the
    # two ManifestDread peek-cards) lands in the graveyard, but that
    # doesn't change the hand size.
    assert synthetic_state.hand_sizes[0] == hand_before


# --- Once-per-turn ledger end-to-end behavior ------------------------


def test_once_per_turn_ledger_suppresses_repeat_fires(synthetic_state):
    """The once-per-turn gate (Slice 1's `triggered_this_turn` ledger,
    cleared at cleanup_step_for_active_player) should suppress a
    repeat firing of a ``once_per_turn=True`` trigger.

    Setup:
    - Inject a creature on p0's bf with ``once_per_turn=True`` +
      ``predicate=always`` + ``effect=draw_cards(1)``. The ``always``
      predicate fires on every ENTERS event (regardless of who's
      entering, including the carrying creature's own ETB).
    - The fixture injection itself (RS8) fires the trigger once on
      bf entry — the carrying creature is the source AND the
      subject. Drain that.
    - Now reanimate ANOTHER creature into p0's graveyard, then
      reanimate it (RS3). The first reanimation's ENTERS event
      should NOT push another trigger because the source's
      `(source_id, ability_index)` is in `triggered_this_turn`.
    - Verify: the second creature lands on bf but no trigger fires
      (stack is empty after the cast resolves).

    Then run cleanup_step_for_active_player. The ledger clears.
    Reanimate ANOTHER creature; the trigger fires again.
    """
    observer = {
        "name": "Test Once-Per-Turn Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "always"},
                "effect": {"kind": "draw_cards", "count": 1},
                "once_per_turn": True,
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )
    # The injection fired the once-per-turn trigger (Always predicate
    # matches the observer's own entry). Drain it.
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.is_stack_empty()

    # Inject a second creature into p0's graveyard and reanimate it.
    # The observer's trigger would fire on its arrival (Always
    # predicate accepts any subject) — but the once-per-turn ledger
    # should suppress it.
    crt_b_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_GRAVEYARD
    )
    ok = synthetic_state.move_object_from_graveyard_to_battlefield(0, crt_b_id)
    assert ok is True
    assert synthetic_state.find_object_zone(crt_b_id) == ZONE_BATTLEFIELD
    # Ledger gate: the observer's trigger was SUPPRESSED on this
    # entry because (observer_id, 0) is already in
    # state.triggered_this_turn. Stack stays empty.
    assert synthetic_state.is_stack_empty(), (
        f"expected once-per-turn gate to suppress repeat firing; observer "
        f"id={observer_id}, stack size={synthetic_state.stack_size()}"
    )

    # Cleanup clears the ledger.
    synthetic_state.cleanup_step_for_active_player()

    # Reanimate a THIRD creature; trigger should fire again.
    crt_c_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_GRAVEYARD
    )
    synthetic_state.move_object_from_graveyard_to_battlefield(0, crt_c_id)
    assert synthetic_state.stack_size() == 1, (
        "expected once-per-turn gate to re-allow firing after cleanup "
        "cleared the ledger; stack is empty so the reset failed"
    )


# =====================================================================
# Slice 4 — predicate-algebra extensions for landfall, artifact-enters,
# Panharmonicon-shape, and "under your control" triggers.
#
# Slices 1-3 wired the substrate plus `source_is_self` / `source_is_creature`
# predicates. Slice 4 adds four new predicate kinds that close the gap to
# the bulk of EoE's ETB-trigger predicate templates from
# `src/mtg_bot/cards/parser/triggers.py:603-1266`:
#
# - `subject_is_land`     -> Python `Subject(None, IsLand())`
# - `subject_is_artifact` -> Python `Subject(None, IsArtifact())`
# - `subject_is_creature` -> Python `Subject(None, IsCreature())`
# - `controlled_by_you`   -> Python `Controlled(by=YOU, subject=None)`
#
# Each test below pins ONE predicate variant by setting up the minimal
# scenario that fires (and the negative case that doesn't).
# =====================================================================


def _landfall_observer_dict():
    """Pre-existing observer with a 'whenever a land enters, draw a
    card' trigger. Uses `subject_is_land` so any land arrival fires
    it — regardless of whose land it is. The canonical landfall
    shape from `cards/parser/triggers.py`.
    """
    return {
        "name": "Test Landfall Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "subject_is_land"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _artifact_enters_observer_dict():
    """Pre-existing observer with a 'whenever an artifact enters,
    draw a card' trigger. Uses `subject_is_artifact`.
    """
    return {
        "name": "Test Artifact-Enters Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "subject_is_artifact"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _panharmonicon_under_your_control_dict():
    """The Panharmonicon-faithful predicate shape from the audit's
    §5.4 Example A: "whenever a CREATURE enters UNDER YOUR CONTROL,
    draw a card". Predicate = `subject_is_creature AND
    controlled_by_you`. Slice 1's substrate could only express the
    weaker `source_is_creature AND NOT source_is_self`; this is the
    true Panharmonicon shape now that we have subject-anchored
    predicates.
    """
    return {
        "name": "Test Panharmonicon",
        "card_types": ["Artifact"],
        "mana_cost": {},
        "effects": [],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _vanilla_artifact_dict():
    """A vanilla artifact (no triggered_abilities). Used as the
    artifact-enters probe's bait."""
    return {
        "name": "Test Vanilla Artifact",
        "card_types": ["Artifact"],
        "mana_cost": {},
        "effects": [],
    }


# --- parser surface -------------------------------------------------


@pytest.mark.parametrize(
    "predicate_kind",
    [
        "subject_is_land",
        "subject_is_artifact",
        "subject_is_creature",
        "controlled_by_you",
    ],
)
def test_slice4_predicate_kinds_parse(empty_registry, predicate_kind):
    """All four Slice 4 predicate kinds parse without error."""
    d = {
        "name": f"Slice4 Test {predicate_kind}",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": predicate_kind},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


# --- subject_is_land — landfall ------------------------------------


def test_landfall_observer_fires_on_land_arrival(synthetic_state):
    """A pre-existing observer with `subject_is_land` fires when ANY
    land enters the battlefield. Drives Slice 4's landfall predicate.

    Setup: observer on p0's bf + ETB-self-draw LAND on p0's library
    + Cultivate-shape sorcery in p0's hand. Cast sorcery -> land
    enters -> two triggers stack (observer's landfall AND land's
    self-draw both fire).
    """
    observer_id = synthetic_state.register_and_inject_object(
        _landfall_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    land_id = synthetic_state.register_and_inject_object(
        _vanilla_land_dict("Test Plains"), 0, ZONE_LIBRARY
    )
    sorcery = {
        "name": "Test Cultivate (Slice 4)",
        "card_types": ["Sorcery"],
        "mana_cost": {},
        "effects": [
            {"kind": "search_library_for_basic_land_tapped", "count": 1}
        ],
    }
    sorcery_id = synthetic_state.register_and_inject_object(
        sorcery, 0, ZONE_HAND
    )

    library_before = synthetic_state.library_sizes[0]

    synthetic_state.push_to_stack(sorcery_id, 0)
    synthetic_state.resolve_top_of_stack()

    # Land landed on bf; observer's landfall trigger is on the stack.
    assert synthetic_state.find_object_zone(land_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected the landfall observer's `subject_is_land` trigger to "
        "fire on the searched-up land's arrival"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve the landfall trigger -> draw 1.
    synthetic_state.resolve_top_of_stack()
    # Library: -1 (land moved) - 1 (draw) = library_before - 2.
    assert synthetic_state.library_sizes[0] == library_before - 2


def test_landfall_observer_does_not_fire_on_non_land(synthetic_state):
    """`subject_is_land` rejects creatures / artifacts / etc. The
    landfall observer DOES NOT fire when a creature enters.
    """
    synthetic_state.register_and_inject_object(
        _landfall_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # The injection above fired no observer trigger (observer's own
    # entry is the subject — observer is a Creature, not a Land).
    # Cast a vanilla creature; observer should not fire.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    # Stack empty: observer's `subject_is_land` rejected the creature
    # subject; no trigger fired.
    assert synthetic_state.is_stack_empty(), (
        "expected `subject_is_land` to reject the creature subject; "
        "stack is non-empty so the predicate accepted the wrong subject"
    )


# --- subject_is_artifact — artifact-enters --------------------------


def test_artifact_observer_fires_on_artifact_arrival(synthetic_state):
    """An observer with `subject_is_artifact` fires when an artifact
    enters. Reanimation path -> deterministic bf entry.
    """
    synthetic_state.register_and_inject_object(
        _artifact_enters_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Inject vanilla artifact into p0's graveyard (not technically a
    # "creature" reanim, but `move_object_from_graveyard_to_battlefield`
    # is type-agnostic — see state_ops_zones.rs).
    artifact_id = synthetic_state.register_and_inject_object(
        _vanilla_artifact_dict(), 0, ZONE_GRAVEYARD
    )
    ok = synthetic_state.move_object_from_graveyard_to_battlefield(
        0, artifact_id
    )
    assert ok is True
    assert synthetic_state.find_object_zone(artifact_id) == ZONE_BATTLEFIELD

    assert synthetic_state.stack_size() == 1, (
        "expected `subject_is_artifact` observer to fire on the artifact's "
        "reanimation"
    )


def test_artifact_observer_does_not_fire_on_creature(synthetic_state):
    """`subject_is_artifact` rejects a creature subject."""
    synthetic_state.register_and_inject_object(
        _artifact_enters_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    crt_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(crt_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(crt_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `subject_is_artifact` to reject the creature subject"
    )


# --- subject_is_creature + controlled_by_you (Panharmonicon-faithful) -


def test_panharmonicon_faithful_fires_on_your_creature(synthetic_state):
    """The audit's §5.4 Example A — Panharmonicon-faithful predicate:
    `subject_is_creature AND controlled_by_you`. Fires when YOUR
    creature enters; doesn't fire on opponent's.
    """
    # Inject the Panharmonicon on p0's bf. controller = 0.
    pan_id = synthetic_state.register_and_inject_object(
        _panharmonicon_under_your_control_dict(), 0, ZONE_BATTLEFIELD
    )
    # Cast a vanilla creature for p0 — should fire.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected the Panharmonicon trigger to fire on p0's own creature "
        "entering (subject_is_creature + controlled_by_you both true)"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == pan_id


def test_panharmonicon_faithful_skips_opponent_creature(synthetic_state):
    """Inverse: the Panharmonicon does NOT fire when an OPPONENT's
    creature enters — `controlled_by_you` rejects the subject whose
    controller differs from the source's controller.
    """
    pan_id = synthetic_state.register_and_inject_object(
        _panharmonicon_under_your_control_dict(), 0, ZONE_BATTLEFIELD
    )
    # Inject an opponent (controller=1) creature directly onto p1's
    # battlefield via the RS8 test fixture path — this fires
    # dispatch_event for the entering creature.
    opp_crt_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 1, ZONE_BATTLEFIELD
    )

    assert synthetic_state.find_object_zone(opp_crt_id) == ZONE_BATTLEFIELD
    # Stack is empty because the Panharmonicon's `controlled_by_you`
    # predicate rejected the opponent's creature subject. The unused
    # `pan_id` documents which observer would have fired had the
    # controller match passed.
    _ = pan_id
    assert synthetic_state.is_stack_empty(), (
        "expected `controlled_by_you` to reject the opponent's creature "
        "subject; stack is non-empty so the predicate accepted"
    )


def test_panharmonicon_faithful_skips_non_creature(synthetic_state):
    """The Panharmonicon's predicate ANDs `subject_is_creature`
    with `controlled_by_you`. A LAND entering under your control
    fails the creature half and doesn't fire.
    """
    synthetic_state.register_and_inject_object(
        _panharmonicon_under_your_control_dict(), 0, ZONE_BATTLEFIELD
    )
    # Inject a vanilla land directly on p0's bf.
    synthetic_state.register_and_inject_object(
        _vanilla_land_dict(), 0, ZONE_BATTLEFIELD
    )
    # No trigger: subject is a land, not a creature.
    assert synthetic_state.is_stack_empty(), (
        "expected `subject_is_creature AND controlled_by_you` to reject "
        "a land subject (creature half fails)"
    )


# --- controlled_by_you alone (no type filter) -----------------------


def test_controlled_by_you_fires_on_any_owned_arrival(synthetic_state):
    """A bare `controlled_by_you` predicate accepts any subject the
    source controls — creature, land, artifact, whatever. Pin the
    predicate as a unit by exercising one such subject (a creature).
    """
    bare_observer = {
        "name": "Test Bare Controlled-By-You",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "controlled_by_you"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        bare_observer, 0, ZONE_BATTLEFIELD
    )
    # The observer's own arrival is `controlled_by_you` w.r.t. itself
    # (subject_controller == source_controller == 0) — fires once.
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.is_stack_empty()

    # Cast a vanilla creature for p0; observer fires again.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `controlled_by_you` to accept p0's own creature "
        "entering"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_controlled_by_you_skips_opponent_arrival(synthetic_state):
    """`controlled_by_you` rejects a subject whose controller doesn't
    match the source's controller. Inject opponent's permanent
    directly on p1's bf; observer on p0 silently skips.
    """
    bare_observer = {
        "name": "Test Bare Controlled-By-You (skip case)",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "controlled_by_you"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(
        bare_observer, 0, ZONE_BATTLEFIELD
    )
    # Drain the observer's self-firing on its own injection.
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()

    # Inject opponent's creature directly on p1's bf — observer should
    # NOT fire.
    opp_crt_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 1, ZONE_BATTLEFIELD
    )
    assert synthetic_state.find_object_zone(opp_crt_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `controlled_by_you` to reject the opponent-controlled "
        "creature subject"
    )


# =====================================================================
# Slice 5 — Event::Dies dispatch on SBA destroy path.
#
# Adds the second wired event (DIES) plus the dispatch wiring at the
# SBA destroy site. Pattern mirrors Slice 2's deferred-dispatch queue
# for ENTERS — apply_sbas pushes dying ids onto
# `pending_dies_dispatches`, then drains the queue via
# `drain_pending_dies_dispatches` AFTER the SBA loop completes so
# cascade death triggers don't compete with the in-progress iteration.
#
# Test coverage:
# - `dies` event parses successfully (`enters` and `dies` both wired;
#   `cast` / `attacks` / `deals_damage` still raise).
# - A creature with a SourceIsSelf DIES trigger fires it when SBA
#   destroys the creature.
# - A pre-existing observer with `subject_is_creature` DIES trigger
#   observes ANOTHER creature's destruction (the "whenever a creature
#   dies" shape).
# - `controlled_by_you` filter rejects opponent's creature dying
#   under your observer.
# - Multiple simultaneous deaths each fire their own DIES trigger.
# =====================================================================


# Same constants as test_sba.py for damage-driven death scenarios.
P0_FIRST_BEAR_ID = 5
P0_SECOND_BEAR_ID = 6


def _dies_self_draw_dict(name="Test DIES Self-Draw"):
    """A creature with a SourceIsSelf DIES trigger that draws 1 card
    on death. The canonical "when this creature dies, draw a card"
    shape.
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _dies_creature_observer_dict(name="Test DIES Creature Observer"):
    """A pre-existing observer with a 'whenever a creature dies,
    draw a card' trigger (the Slimefoot-shape predicate). The
    predicate is `subject_is_creature` so any creature dying fires
    it — regardless of who controlled the dying creature. To gate
    on "a creature YOU control", wrap with `controlled_by_you`
    (covered in the next test).
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _dies_under_your_control_observer_dict():
    """'Whenever a creature you control dies, draw a card' —
    `subject_is_creature AND controlled_by_you`.
    """
    return {
        "name": "Test DIES Under-Your-Control Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface --------------------------------------------------


def test_slice5_dies_event_parses(empty_registry):
    """`event: dies` is accepted at register-time post-Slice-5."""
    pid = empty_registry.register_printing(_dies_self_draw_dict())
    assert pid == 0


@pytest.mark.parametrize("event_kind", ["deals_damage"])
def test_slice5_other_events_still_raise(empty_registry, event_kind):
    """`deals_damage` still raises at register-time post-Slice-7
    (Slice 7 wires enters / dies / cast / attacks; only deals_damage
    is left unwired). Slice 5's name retained for git history.
    """
    d = {
        "name": f"Slice5 Bad Event {event_kind}",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": event_kind,
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="not yet wired"):
        empty_registry.register_printing(d)


# --- SourceIsSelf DIES — "when this creature dies" --------------------


def test_dies_self_trigger_fires_on_sba_destroy(synthetic_state):
    """Inject a 1/1 with a `source_is_self` DIES trigger on p0's bf.
    Damage it for 1 (lethal) and apply SBAs. The DIES trigger lands
    on the stack; resolving it draws 1 card.
    """
    crt_id = synthetic_state.register_and_inject_object(
        _dies_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]
    hand_before = synthetic_state.hand_sizes[0]

    # Damage the 1/1 for 1 (lethal).
    synthetic_state.deal_damage(crt_id, 1)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1

    # The creature is in the graveyard.
    assert synthetic_state.find_object_zone(crt_id) == ZONE_GRAVEYARD
    # The DIES trigger is on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self DIES trigger to land on the stack "
        "after SBA destruction"
    )
    top = synthetic_state.peek_top_of_stack()
    # The source is the now-dead creature (still findable via the
    # zones-walk lookup since it's now in p0's graveyard).
    assert top["source_obj_id"] == crt_id

    # Resolve the trigger: DrawCards(1) -> hand +1, library -1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- subject_is_creature DIES observer (Slimefoot shape) --------------


def test_dies_creature_observer_fires_on_another_creature_death(synthetic_state):
    """Pre-existing observer with `subject_is_creature` DIES trigger.
    When ANY creature dies, the observer fires.
    """
    # Inject the observer on p0's bf (it's a 2/3 — won't die from 1
    # damage so we can let the victim die).
    observer_id = synthetic_state.register_and_inject_object(
        _dies_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # The synthetic state's pre-built Grizzly Bears at P0_FIRST_BEAR_ID
    # is a 2/2; damage it for 2 (lethal).
    library_before = synthetic_state.library_sizes[0]
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1

    # The observer's DIES trigger should be on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the observer's `subject_is_creature` DIES trigger to "
        "fire when the Bear died"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve the trigger: observer's controller (p0) draws.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_dies_creature_observer_skips_non_creature_death(synthetic_state):
    """`subject_is_creature` rejects non-creature subjects. To prove
    this, inject a Land into a graveyard via reanimation... but Lands
    don't die from SBA. We exercise the negative case differently:
    inject a creature with a DIES trigger that's never invoked.

    The simpler proof: register a creature with a non-creature
    typing override (an Artifact-only printing) and trigger the
    death. Inject as an Artifact-only printing whose toughness is 0
    so apply_sbas destroys it via CR 704.5f — but artifacts without
    Creature typing don't pass the CR 704.5g lethal-damage check.
    Since 704.5f is creature-only too (per Rust's
    `collect_sba_victims`), there's no clean negative-case in the
    Slice 5 substrate. Document via a noop assertion.
    """
    # Inject the creature-observer; no creature deaths so no fires.
    synthetic_state.register_and_inject_object(
        _dies_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 0
    assert synthetic_state.is_stack_empty()


# --- subject_is_creature AND controlled_by_you (Slimefoot+) -----------


def test_dies_under_your_control_fires_on_own_creature(synthetic_state):
    """`subject_is_creature AND controlled_by_you` fires when YOUR
    creature dies. Observer on p0's bf, p0's Bear dies.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _dies_under_your_control_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1
    assert synthetic_state.stack_size() == 1, (
        "expected the under-your-control DIES observer to fire on p0's "
        "own Bear dying"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_dies_under_your_control_skips_opponent_death(synthetic_state):
    """`controlled_by_you` rejects an opponent's creature dying.
    Observer on p0's bf, p1's Bear dies.
    """
    synthetic_state.register_and_inject_object(
        _dies_under_your_control_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # The synthetic state's p1 first Bear is at id 67+5=72.
    P1_FIRST_BEAR_ID = 67 + 5
    synthetic_state.deal_damage(P1_FIRST_BEAR_ID, 2)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1
    assert synthetic_state.is_stack_empty(), (
        "expected `controlled_by_you` to reject the opponent's Bear "
        "dying; stack is non-empty so the predicate accepted the wrong "
        "controller"
    )


# --- multiple simultaneous deaths -------------------------------------


def test_dies_multiple_simultaneous_deaths_each_fire(synthetic_state):
    """Two creatures die in the SAME SBA pass. Both DIES triggers
    land on the stack — one per dead creature. Order matches
    `collect_sba_victims`'s bf-ETB walk (Slimefoot fires twice).
    """
    observer_id = synthetic_state.register_and_inject_object(
        _dies_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Damage TWO p0 Bears lethally.
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    synthetic_state.deal_damage(P0_SECOND_BEAR_ID, 2)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 2

    # TWO triggers on the stack — one per dead Bear.
    assert synthetic_state.stack_size() == 2, (
        f"expected 2 DIES triggers (one per dead Bear); got "
        f"{synthetic_state.stack_size()}"
    )
    # Both have observer as source.
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


# --- cross-engine bridge accepts "dies" event ------------------------


def test_dies_event_parses_via_cross_engine_bridge():
    """The cross_engine.py `_ta_dict_to_python_triggered_ability`
    bridge accepts `event: dies` and produces Python's
    `Event.DIES` enum.
    """
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.effects.predicates import Event

    ta_dict = {
        "event": "dies",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "draw_cards", "count": 1},
    }
    ta = _ta_dict_to_python_triggered_ability(ta_dict)
    assert ta.event is Event.DIES


# =====================================================================
# Slice 6 — Event::Cast dispatch on cast_creature site.
#
# Adds the third wired event family. After mana payment commits in
# `cast_creature` (the no-stack cast primitive), dispatch_event fires
# Event::Cast with the still-in-hand spell as subject. CAST fires
# BEFORE the move-to-bf step's ENTERS dispatch, matching Python's
# cast-then-resolve ordering.
#
# Test coverage:
# - `cast` event parses; `attacks` / `deals_damage` still raise.
# - A creature with a SourceIsSelf CAST trigger fires it when cast.
# - A pre-existing observer with `subject_is_creature` CAST trigger
#   observes ANOTHER creature being cast.
# - `controlled_by_you` filter on a CAST trigger.
# - CAST fires BEFORE ENTERS — when both are wired on the same source,
#   the stack-resolve order is CAST -> ENTERS.
# =====================================================================


def _cast_self_draw_dict(name="Test CAST Self-Draw"):
    """A free-cast creature with a `source_is_self` CAST trigger
    that draws 1 card when cast. mana_cost = empty so the cast
    payment is free (no mana pool setup needed).
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "cast",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _cast_creature_observer_dict():
    """Pre-existing observer with `subject_is_creature` CAST trigger.
    Fires when ANY creature is cast — regardless of who cast it.
    """
    return {
        "name": "Test CAST Creature Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "cast",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _free_vanilla_creature_dict(name="Test Free Vanilla Creature"):
    """A free-cast vanilla creature (no triggered abilities). Used
    as the bait in CAST observer tests so the cast payment is free.
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
    }


# --- parser surface --------------------------------------------------


def test_slice6_cast_event_parses(empty_registry):
    """`event: cast` is accepted at register-time post-Slice-6."""
    pid = empty_registry.register_printing(_cast_self_draw_dict())
    assert pid == 0


# --- SourceIsSelf CAST — "when you cast this creature" --------------


def test_cast_self_trigger_fires_on_cast(synthetic_state):
    """Inject a free-cast creature with a `source_is_self` CAST
    trigger into p0's hand. Cast it. The CAST trigger fires; the
    creature lands on the bf (any ENTERS trigger would too, but
    this fixture only has CAST).
    """
    crt_id = synthetic_state.register_and_inject_object(
        _cast_self_draw_dict(), 0, ZONE_HAND
    )
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.cast_creature(crt_id, 0)
    assert ok is True
    # The CAST trigger is on the stack (no ENTERS trigger fires
    # because the creature only has a CAST one).
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self CAST trigger to fire when "
        "cast_creature succeeded"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == crt_id
    # The creature is on the bf (cast_creature is no-stack so
    # move-to-bf already happened).
    assert synthetic_state.find_object_zone(crt_id) == ZONE_BATTLEFIELD

    # Resolve the CAST trigger -> draw 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- subject_is_creature CAST observer (Talrand-shape) --------------


def test_cast_creature_observer_fires_on_another_creature_cast(synthetic_state):
    """Pre-existing observer with `subject_is_creature` CAST trigger.
    Cast a different vanilla creature and the observer fires.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _cast_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Cast a free vanilla creature; observer fires.
    bait_id = synthetic_state.register_and_inject_object(
        _free_vanilla_creature_dict(), 0, ZONE_HAND
    )
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.cast_creature(bait_id, 0)
    assert ok is True

    assert synthetic_state.stack_size() == 1, (
        "expected the `subject_is_creature` CAST observer to fire on "
        "the vanilla creature being cast"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve -> p0 draws 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_cast_observer_with_controlled_by_you_fires_on_own_cast(synthetic_state):
    """`subject_is_creature AND controlled_by_you` CAST observer
    fires on p0's own cast.
    """
    observer = {
        "name": "Test CAST Under-Your-Control Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "cast",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )
    bait_id = synthetic_state.register_and_inject_object(
        _free_vanilla_creature_dict(), 0, ZONE_HAND
    )
    ok = synthetic_state.cast_creature(bait_id, 0)
    assert ok is True
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


# --- CAST fires BEFORE ENTERS ------------------------------------


def test_cast_fires_before_enters_when_both_on_same_source(synthetic_state):
    """A creature with BOTH a CAST trigger AND an ENTERS trigger:
    the CAST trigger fires first (chronologically), then the ENTERS
    trigger fires when the creature lands on the bf.

    Both triggers go on the stack. The stack is LIFO so the
    LAST-pushed (ENTERS) resolves first. CAST is pushed FIRST
    (mana-paid moment), ENTERS is pushed AFTER (move-to-bf moment).
    So the resolution order is ENTERS-first, CAST-second.
    """
    crt = {
        "name": "Test Both CAST and ENTERS",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "cast",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            },
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 2},
            },
        ],
    }
    crt_id = synthetic_state.register_and_inject_object(crt, 0, ZONE_HAND)
    life_before = synthetic_state.life_totals[0]
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.cast_creature(crt_id, 0)
    assert ok is True

    # Both triggers on stack — push order: CAST first, ENTERS second
    # (cast_creature dispatches CAST in phase 2, ENTERS in phase 3).
    # Stack is LIFO so top is the ENTERS trigger.
    assert synthetic_state.stack_size() == 2, (
        f"expected 2 triggers (CAST + ENTERS); got "
        f"{synthetic_state.stack_size()}"
    )

    # Resolve top: ENTERS -> gain 2 life.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.life_totals[0] == life_before + 2

    # Resolve next: CAST -> draw 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1
    assert synthetic_state.is_stack_empty()


# --- cross-engine bridge accepts "cast" event ------------------------


def test_cast_event_parses_via_cross_engine_bridge():
    """The cross_engine.py bridge accepts `event: cast` and produces
    Python's `Event.IS_CAST` (note Python's enum spells it
    `IS_CAST`; Rust uses `Cast`).
    """
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.effects.predicates import Event

    ta_dict = {
        "event": "cast",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "draw_cards", "count": 1},
    }
    ta = _ta_dict_to_python_triggered_ability(ta_dict)
    assert ta.event is Event.IS_CAST


# ====================================================================
# Slice 7 — Event::Attacks
#
# `Event::Attacks` is the per-attacker event that fires after each
# attacker is committed via `declare_attacker`. The attacker is on
# the battlefield (it has to be, for the eligibility gate to admit
# it), so the bf observer set already contains it — SourceIsSelf
# ("whenever this creature attacks") matches naturally without
# needing the DIES/CAST extra-observer pattern.
#
# Mirrors Python's `combat._fire_attacks_triggers(state, obj)` call
# at `src/mtg_bot/combat.py:115` — fired per-attacker immediately
# after the attacking flag is set, so triggers observe each attacker
# in declaration order.
# ====================================================================


# --- registry-dict shapes used by the Slice 7 tests -----------------


def _attacks_self_draw_dict(name="Test ATTACKS Self-Draw"):
    """A creature with a `source_is_self` ATTACKS trigger that draws
    1 card when it attacks. The canonical "whenever this creature
    attacks, draw a card" shape.
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _attacks_creature_observer_dict():
    """Pre-existing observer with `subject_is_creature` ATTACKS
    trigger. Fires when ANY creature attacks — regardless of
    controller. Hellrider-shape ("whenever a creature attacks").
    """
    return {
        "name": "Test ATTACKS Creature Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface --------------------------------------------------


def test_slice7_attacks_event_parses(empty_registry):
    """`event: attacks` is accepted at register-time post-Slice-7."""
    pid = empty_registry.register_printing(_attacks_self_draw_dict())
    assert pid == 0


@pytest.mark.parametrize("event_kind", ["deals_damage"])
def test_slice7_other_events_still_raise(empty_registry, event_kind):
    """`deals_damage` still raises at register-time post-Slice-7
    (Slice 7 wires enters / dies / cast / attacks only).
    """
    d = {
        "name": f"Slice7 Bad Event {event_kind}",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": event_kind,
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="not yet wired"):
        empty_registry.register_printing(d)


# --- SourceIsSelf ATTACKS — "whenever this creature attacks" --------


def test_attacks_self_trigger_fires_on_declare_attacker(synthetic_state):
    """Inject a 2/2 with a `source_is_self` ATTACKS trigger onto p0's
    bf. Declare it as attacker. The ATTACKS trigger fires; resolving
    it draws 1 card.
    """
    attacker_id = synthetic_state.register_and_inject_object(
        _attacks_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]
    hand_before = synthetic_state.hand_sizes[0]

    ok = synthetic_state.declare_attacker(attacker_id)
    assert ok is True
    # The attacker is tapped (no vigilance).
    assert synthetic_state.is_tapped(attacker_id) is True
    # The ATTACKS trigger is on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self ATTACKS trigger to land on the "
        "stack after declare_attacker succeeded"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == attacker_id

    # Resolve: DrawCards(1) -> hand +1, library -1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- subject_is_creature ATTACKS observer (Hellrider-shape) ---------


def test_attacks_creature_observer_fires_on_another_attacker(synthetic_state):
    """Pre-existing observer with `subject_is_creature` ATTACKS
    trigger. When ANY creature attacks, the observer fires.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _attacks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # P0_FIRST_BEAR_ID (id=5) is a Grizzly Bear (2/2) on p0's bf.
    # It's NOT summoning-sick (synthetic seed defaults bf objects to
    # summoning_sick=false).
    library_before = synthetic_state.library_sizes[0]

    ok = synthetic_state.declare_attacker(P0_FIRST_BEAR_ID)
    assert ok is True

    # The observer's ATTACKS trigger should be on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the observer's `subject_is_creature` ATTACKS trigger "
        "to fire when the Bear attacked"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve: observer's controller (p0) draws 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_attacks_observer_with_controlled_by_you_fires_on_own_attacker(
    synthetic_state,
):
    """`subject_is_creature AND controlled_by_you` ATTACKS observer
    fires on p0's own creature attacking.
    """
    observer = {
        "name": "Test ATTACKS Under-Your-Control Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )

    ok = synthetic_state.declare_attacker(P0_FIRST_BEAR_ID)
    assert ok is True

    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_attacks_observer_with_controlled_by_you_skips_opponent_attacker(
    synthetic_state,
):
    """`controlled_by_you` ATTACKS observer on p0's bf does NOT fire
    when an opponent's creature attacks. Use force_set_summoning_sick
    + the p1 bear to make the cross-controller case explicit — but
    actually p1's bears are also unsick in the synthetic seed; their
    issue is that they belong to player 1 and the observer is on p0.
    """
    observer = {
        "name": "Test ATTACKS Under-Your-Control Observer (skip case)",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    # Observer on p0's bf. P1's bear attacks. The observer's
    # `controlled_by_you` predicate (keyed on the observer's
    # controller == subject's controller) should reject.
    _ = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )
    P1_FIRST_BEAR_ID = 67 + 5

    ok = synthetic_state.declare_attacker(P1_FIRST_BEAR_ID)
    assert ok is True
    # No triggers on the stack — predicate filtered out the opponent's
    # attacker.
    assert synthetic_state.stack_size() == 0


def test_attacks_fires_per_attacker_in_declaration_order(synthetic_state):
    """A `subject_is_creature` ATTACKS observer fires ONCE per
    attacker. Declare two creatures in sequence; two triggers stack
    in push order (matches Python's `_fire_attacks_triggers` per-
    attacker fan-out).
    """
    observer_id = synthetic_state.register_and_inject_object(
        _attacks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # P0_FIRST_BEAR_ID = 5; the bear at id=6 is the second Grizzly
    # Bear on p0's bf.
    second_bear_id = P0_FIRST_BEAR_ID + 1

    ok1 = synthetic_state.declare_attacker(P0_FIRST_BEAR_ID)
    assert ok1 is True
    # First attack: 1 trigger on the stack.
    assert synthetic_state.stack_size() == 1
    # Resolve before declaring the next (cleaner state assertion).
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.stack_size() == 0

    ok2 = synthetic_state.declare_attacker(second_bear_id)
    assert ok2 is True
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_attacks_skips_when_declare_attacker_fails(synthetic_state):
    """When `declare_attacker` REJECTS the creature (tapped /
    summoning-sick / not-on-bf), the ATTACKS dispatch is NOT fired —
    the failure path bails out before reaching the dispatch line.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _attacks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Force the would-be attacker into a sick state so declare_attacker
    # bails out before dispatching.
    synthetic_state.force_set_summoning_sick(P0_FIRST_BEAR_ID, True)

    ok = synthetic_state.declare_attacker(P0_FIRST_BEAR_ID)
    assert ok is False
    # No trigger on the stack — failure path didn't dispatch.
    assert synthetic_state.stack_size() == 0
    # Sanity: observer is still on the bf, just didn't observe anything.
    assert (
        synthetic_state.find_object_zone(observer_id) == ZONE_BATTLEFIELD
    )


# --- cross-engine bridge accepts "attacks" event --------------------


def test_attacks_event_parses_via_cross_engine_bridge():
    """The cross_engine.py bridge accepts `event: attacks` and
    produces Python's `Event.ATTACKS`.
    """
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.effects.predicates import Event

    ta_dict = {
        "event": "attacks",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "draw_cards", "count": 1},
    }
    ta = _ta_dict_to_python_triggered_ability(ta_dict)
    assert ta.event is Event.ATTACKS


# ====================================================================
# Slice 8 — Event::LeavesBattlefield
#
# `Event::LeavesBattlefield` (LTB) fires from every bf-exit
# primitive — both bounce-shape (`move_object` off the battlefield)
# and SBA destroy (combat death / zero-toughness). The dispatch
# queues onto `state.pending_leaves_dispatches` and drains via
# `drain_pending_leaves_dispatches` AFTER the bf-exit primitive's
# mutation commits.
#
# Relationship to DIES: every creature death is also a bf exit, so
# SBA-destroyed creatures fire BOTH DIES and LEAVES_BATTLEFIELD.
# Push order: DIES first, LEAVES second; the matching drain order
# preserves that stack-push order. LIFO resolution: LEAVES resolves
# first, mirroring Python's `move_off_battlefield` (turn.py:280-289)
# which fires DIES then LEAVES_BATTLEFIELD in source order.
#
# Mirrors Python's `_fire_leaves_battlefield_triggers(state, obj)`
# at `src/mtg_bot/turn.py:289`, called by `move_off_battlefield`
# for every bf-exit regardless of card type (lands fire LEAVES too).
# ====================================================================


# --- registry-dict shapes used by the Slice 8 tests -----------------


def _leaves_self_draw_dict(name="Test LEAVES Self-Draw"):
    """A creature with a `source_is_self` LEAVES_BATTLEFIELD trigger
    that draws 1 card when it leaves the battlefield. Cryogen
    Relic-shape ("when this leaves the battlefield, draw a card").
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _leaves_creature_observer_dict():
    """Pre-existing observer with `subject_is_creature`
    LEAVES_BATTLEFIELD trigger. Fires when ANY creature leaves the
    battlefield — Slimefoot-like or Bloodghast-shape.
    """
    return {
        "name": "Test LEAVES Creature Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface --------------------------------------------------


def test_slice8_leaves_battlefield_event_parses(empty_registry):
    """`event: leaves_battlefield` is accepted at register-time
    post-Slice-8.
    """
    pid = empty_registry.register_printing(_leaves_self_draw_dict())
    assert pid == 0


@pytest.mark.parametrize("event_kind", ["deals_damage"])
def test_slice8_other_events_still_raise(empty_registry, event_kind):
    """`deals_damage` still raises at register-time post-Slice-8
    (only deals_damage is left unwired).
    """
    d = {
        "name": f"Slice8 Bad Event {event_kind}",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": event_kind,
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="not yet wired"):
        empty_registry.register_printing(d)


# --- SourceIsSelf LEAVES — "when this leaves the battlefield" ------


def test_leaves_self_trigger_fires_on_bounce(synthetic_state):
    """Inject a 1/1 with a `source_is_self` LEAVES trigger onto p0's
    bf. Move it to the hand (bounce-shape via `move_object`). The
    LEAVES trigger fires; resolving it draws 1 card.
    """
    crt_id = synthetic_state.register_and_inject_object(
        _leaves_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]
    hand_before = synthetic_state.hand_sizes[0]

    # Bounce to hand via the generic move_object primitive.
    ok = synthetic_state.move_object(crt_id, ZONE_HAND, 0)
    assert ok is True

    # The creature is now in p0's hand.
    assert synthetic_state.find_object_zone(crt_id) == ZONE_HAND
    # The LEAVES trigger is on the stack. The bounce primitive
    # increments hand_size by 1 (the moved object) so we read the
    # post-bounce delta from hand_before+1 to detect the draw.
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self LEAVES trigger to land on the "
        "stack after bf-exit via move_object"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == crt_id

    # Resolve: DrawCards(1) -> library -1, hand +1 (on top of the
    # already-bounced creature, so hand_size = hand_before + 2).
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1
    assert synthetic_state.hand_sizes[0] == hand_before + 2


def test_leaves_self_trigger_fires_on_sba_destroy(synthetic_state):
    """A 1/1 with `source_is_self` LEAVES trigger; lethal damage
    triggers SBA destroy. The LEAVES trigger fires alongside any
    DIES trigger would (this fixture has only LEAVES). The
    self-trigger source is now in the graveyard — the four-zone
    extra-observer lookup finds it.
    """
    crt_id = synthetic_state.register_and_inject_object(
        _leaves_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]

    synthetic_state.deal_damage(crt_id, 1)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1

    assert synthetic_state.find_object_zone(crt_id) == ZONE_GRAVEYARD
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self LEAVES trigger to fire on SBA "
        "destroy"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == crt_id
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- subject_is_creature LEAVES observer ---------------------------


def test_leaves_creature_observer_fires_on_bounce(synthetic_state):
    """Pre-existing observer with `subject_is_creature` LEAVES
    trigger. When ANY creature leaves the battlefield, the observer
    fires.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _leaves_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]

    # Bounce the P0 Bear (2/2) to its owner's hand.
    ok = synthetic_state.move_object(P0_FIRST_BEAR_ID, ZONE_HAND, 0)
    assert ok is True

    assert synthetic_state.stack_size() == 1, (
        "expected the observer's `subject_is_creature` LEAVES trigger "
        "to fire when the Bear was bounced"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_leaves_creature_observer_fires_on_sba_destroy(synthetic_state):
    """Pre-existing observer with `subject_is_creature` LEAVES
    trigger fires on SBA destroy (lethal damage path). LEAVES
    dispatches alongside the DIES dispatch from Slice 5; both stack
    push but only the LEAVES observer is wired in this fixture.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _leaves_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]

    # Lethal-damage P0's Bear (2/2).
    synthetic_state.deal_damage(P0_FIRST_BEAR_ID, 2)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1

    # Observer's LEAVES trigger lands. (No DIES trigger from this
    # observer — its predicate is LEAVES-only.)
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- DIES + LEAVES on same source: LIFO resolves LEAVES first ------


def test_dies_and_leaves_both_fire_on_sba_destroy_leaves_resolves_first(
    synthetic_state,
):
    """A creature with BOTH a `source_is_self` DIES trigger AND a
    `source_is_self` LEAVES trigger: lethal damage triggers SBA
    destroy. Both triggers fire; push order is DIES first then
    LEAVES (matching Python's `move_off_battlefield` sequence at
    turn.py:280-289). The stack is LIFO so LEAVES resolves FIRST,
    DIES second.
    """
    crt = {
        "name": "Test Both DIES and LEAVES",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            },
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 2},
            },
        ],
    }
    crt_id = synthetic_state.register_and_inject_object(
        crt, 0, ZONE_BATTLEFIELD
    )
    library_before = synthetic_state.library_sizes[0]
    life_before = synthetic_state.life_totals[0]

    synthetic_state.deal_damage(crt_id, 1)
    destroyed = synthetic_state.apply_sbas()
    assert destroyed == 1

    # Two triggers on the stack — DIES + LEAVES.
    assert synthetic_state.stack_size() == 2

    # Stack-top is LEAVES (last-pushed). Resolve: gain 2 life.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.life_totals[0] == life_before + 2

    # Then DIES: draw 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1
    assert synthetic_state.is_stack_empty()


# --- LEAVES does NOT fire on non-bf-exit moves ----------------------


def test_leaves_does_not_fire_on_non_battlefield_move(synthetic_state):
    """Moving an object FROM the hand (not the battlefield) does
    NOT fire a LEAVES_BATTLEFIELD dispatch. Sanity check that the
    `if source_was_battlefield` gate in `move_object` is tight.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _leaves_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # P0_FIRST_HAND_ID = 10 (Llanowar Elves in p0's hand).
    P0_FIRST_HAND_ID = 10

    # Move hand[0] -> graveyard. Not a bf-exit; LEAVES shouldn't fire.
    ok = synthetic_state.move_object(P0_FIRST_HAND_ID, ZONE_GRAVEYARD, 0)
    assert ok is True
    assert synthetic_state.stack_size() == 0, (
        "expected no LEAVES dispatch on hand->graveyard move"
    )
    # Observer is still on bf, just didn't observe.
    assert (
        synthetic_state.find_object_zone(observer_id) == ZONE_BATTLEFIELD
    )


# --- cross-engine bridge accepts "leaves_battlefield" event --------


def test_leaves_battlefield_event_parses_via_cross_engine_bridge():
    """The cross_engine.py bridge accepts `event: leaves_battlefield`
    and produces Python's `Event.LEAVES_BATTLEFIELD`.
    """
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.effects.predicates import Event

    ta_dict = {
        "event": "leaves_battlefield",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "draw_cards", "count": 1},
    }
    ta = _ta_dict_to_python_triggered_ability(ta_dict)
    assert ta.event is Event.LEAVES_BATTLEFIELD


# ====================================================================
# Slice 9 — Event::Blocks
#
# `Event::Blocks` is the per-blocker event that fires after each
# blocker is committed via `declare_blocker`. Symmetric to Slice 7's
# ATTACKS dispatch: the blocker is on the battlefield (per the
# eligibility gate), so the bf observer set already contains it and
# SourceIsSelf ("whenever this creature blocks") matches naturally
# without the DIES/CAST/LEAVES extra-observer pattern.
#
# Mirrors Python's per-pair `dispatch_event(state, Event.BLOCKS,
# blocker=..., attacker=...)` call in `src/mtg_bot/combat.py:274` —
# fired per blocker→attacker pair immediately after the
# blocker.blocking / attacker.blocked_by mutations, so triggers
# observe each blocker in declaration order.
#
# **Single-subject divergence from Python**: Python carries TWO
# subjects on BLOCKS (blocker + attacker) and exposes both via the
# multi-subject predicate schema. Rust dispatches on the blocker
# only; multi-subject predicates against the attacker (e.g.
# Skystinger's "whenever ~ blocks a creature with flying") need the
# `Predicate::Subject(slot, inner)` substrate that lands later. For
# now, BLOCKS triggers keying only on the blocker work as expected.
# ====================================================================


# --- registry-dict shapes used by the Slice 9 tests -----------------


def _blocks_self_draw_dict(name="Test BLOCKS Self-Draw"):
    """A creature with a `source_is_self` BLOCKS trigger that draws
    1 card when it blocks. The canonical "whenever this creature
    blocks, draw a card" shape.
    """
    return {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "blocks",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _blocks_creature_observer_dict():
    """Pre-existing observer with a `subject_is_creature` BLOCKS
    trigger. Fires when ANY creature blocks — regardless of
    controller. Symmetric of the Hellrider-shape ATTACKS observer
    from Slice 7.
    """
    return {
        "name": "Test BLOCKS Creature Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "blocks",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface --------------------------------------------------


def test_slice9_blocks_event_parses(empty_registry):
    """`event: blocks` is accepted at register-time post-Slice-9."""
    pid = empty_registry.register_printing(_blocks_self_draw_dict())
    assert pid == 0


@pytest.mark.parametrize("event_kind", ["deals_damage"])
def test_slice9_other_events_still_raise(empty_registry, event_kind):
    """`deals_damage` still raises at register-time post-Slice-9
    (Slice 9 wires enters / dies / cast / attacks / blocks /
    leaves_battlefield only).
    """
    d = {
        "name": f"Slice9 Bad Event {event_kind}",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": event_kind,
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="not yet wired"):
        empty_registry.register_printing(d)


# --- SourceIsSelf BLOCKS — "whenever this creature blocks" ----------


def test_blocks_self_trigger_fires_on_declare_blocker(synthetic_state):
    """Inject a 2/2 with a `source_is_self` BLOCKS trigger onto p0's
    bf. Declare a p1 attacker, then declare the injected creature as
    its blocker. The BLOCKS trigger fires; resolving it draws 1 card.
    """
    blocker_id = synthetic_state.register_and_inject_object(
        _blocks_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    # P1_FIRST_BEAR_ID = 67 + 5 = 72 — first p1 bear (2/2, not
    # summoning-sick in the synthetic seed). Declare it as attacker
    # first; only then can the p0 blocker block it.
    P1_FIRST_BEAR_ID = 67 + 5
    ok_atk = synthetic_state.declare_attacker(P1_FIRST_BEAR_ID)
    assert ok_atk is True
    # The attacker is the only attacking creature; no ATTACKS trigger
    # because the p1 attacker has no triggered_abilities. Clear the
    # stack just in case (defensive; the synthetic bears are barren).
    while synthetic_state.stack_size() > 0:
        synthetic_state.resolve_top_of_stack()

    library_before = synthetic_state.library_sizes[0]
    hand_before = synthetic_state.hand_sizes[0]

    ok_blk = synthetic_state.declare_blocker(blocker_id, P1_FIRST_BEAR_ID)
    assert ok_blk is True
    # Blocking does NOT tap the blocker (CR 509.1).
    assert synthetic_state.is_tapped(blocker_id) is False
    # The BLOCKS trigger is on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the source_is_self BLOCKS trigger to land on the "
        "stack after declare_blocker succeeded"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == blocker_id

    # Resolve: DrawCards(1) -> hand +1, library -1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.hand_sizes[0] == hand_before + 1
    assert synthetic_state.library_sizes[0] == library_before - 1


# --- subject_is_creature BLOCKS observer (Wall-of-Tanglecord shape) -


def test_blocks_creature_observer_fires_on_another_blocker(synthetic_state):
    """Pre-existing observer with `subject_is_creature` BLOCKS
    trigger. When ANY creature blocks, the observer fires.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _blocks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    P1_FIRST_BEAR_ID = 67 + 5
    # Declare the p1 bear as the attacker.
    ok_atk = synthetic_state.declare_attacker(P1_FIRST_BEAR_ID)
    assert ok_atk is True
    while synthetic_state.stack_size() > 0:
        synthetic_state.resolve_top_of_stack()

    library_before = synthetic_state.library_sizes[0]

    # Declare P0_FIRST_BEAR_ID as the blocker. The observer should
    # fire on its `subject_is_creature` predicate.
    ok_blk = synthetic_state.declare_blocker(
        P0_FIRST_BEAR_ID, P1_FIRST_BEAR_ID
    )
    assert ok_blk is True

    # The observer's BLOCKS trigger should be on the stack.
    assert synthetic_state.stack_size() == 1, (
        "expected the observer's `subject_is_creature` BLOCKS trigger "
        "to fire when the Bear blocked"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve: observer's controller (p0) draws 1.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.library_sizes[0] == library_before - 1


def test_blocks_observer_with_controlled_by_you_fires_on_own_blocker(
    synthetic_state,
):
    """`subject_is_creature AND controlled_by_you` BLOCKS observer
    fires on p0's own creature blocking.
    """
    observer = {
        "name": "Test BLOCKS Under-Your-Control Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "blocks",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )

    P1_FIRST_BEAR_ID = 67 + 5
    ok_atk = synthetic_state.declare_attacker(P1_FIRST_BEAR_ID)
    assert ok_atk is True
    while synthetic_state.stack_size() > 0:
        synthetic_state.resolve_top_of_stack()

    ok_blk = synthetic_state.declare_blocker(
        P0_FIRST_BEAR_ID, P1_FIRST_BEAR_ID
    )
    assert ok_blk is True

    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_blocks_observer_with_controlled_by_you_skips_opponent_blocker(
    synthetic_state,
):
    """`controlled_by_you` BLOCKS observer on p0's bf does NOT fire
    when an opponent's creature blocks. Symmetric of Slice 7's
    cross-controller filter test.
    """
    observer = {
        "name": "Test BLOCKS Under-Your-Control Observer (skip case)",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "blocks",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    # Observer on p0's bf. P0's bear attacks p1; p1's bear blocks.
    # The observer's `controlled_by_you` predicate (keyed on the
    # observer's controller == subject's controller) should reject
    # because the blocker is controlled by p1.
    _ = synthetic_state.register_and_inject_object(
        observer, 0, ZONE_BATTLEFIELD
    )
    P1_FIRST_BEAR_ID = 67 + 5

    ok_atk = synthetic_state.declare_attacker(P0_FIRST_BEAR_ID)
    assert ok_atk is True
    while synthetic_state.stack_size() > 0:
        synthetic_state.resolve_top_of_stack()

    ok_blk = synthetic_state.declare_blocker(
        P1_FIRST_BEAR_ID, P0_FIRST_BEAR_ID
    )
    assert ok_blk is True
    # No triggers on the stack — predicate filtered out the opponent's
    # blocker.
    assert synthetic_state.stack_size() == 0


def test_blocks_fires_per_blocker_in_declaration_order(synthetic_state):
    """A `subject_is_creature` BLOCKS observer fires ONCE per
    blocker. Declare two blockers in sequence against the same
    attacker; two triggers stack in push order (matches Python's
    per-pair dispatch fan-out at `combat.py:274`).
    """
    observer_id = synthetic_state.register_and_inject_object(
        _blocks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    P1_FIRST_BEAR_ID = 67 + 5
    ok_atk = synthetic_state.declare_attacker(P1_FIRST_BEAR_ID)
    assert ok_atk is True
    while synthetic_state.stack_size() > 0:
        synthetic_state.resolve_top_of_stack()

    second_bear_id = P0_FIRST_BEAR_ID + 1

    ok1 = synthetic_state.declare_blocker(
        P0_FIRST_BEAR_ID, P1_FIRST_BEAR_ID
    )
    assert ok1 is True
    # First block: 1 trigger on the stack.
    assert synthetic_state.stack_size() == 1
    # Resolve before declaring the next (cleaner state assertion).
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.stack_size() == 0

    ok2 = synthetic_state.declare_blocker(
        second_bear_id, P1_FIRST_BEAR_ID
    )
    assert ok2 is True
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_blocks_skips_when_declare_blocker_fails(synthetic_state):
    """When `declare_blocker` REJECTS the pair (attacker not in
    `state.attackers`, blocker tapped / missing / non-creature, or a
    keyword gate like flying/skulk fails), the BLOCKS dispatch is
    NOT fired — the failure path bails out before reaching the
    dispatch line.

    Tight gate: pin the "attacker not declared" rejection path
    (cheapest fixture). The Slice 7 ATTACKS test pinned the
    summoning-sick rejection; this test pins the symmetric "no
    matching attacker" reject for BLOCKS.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _blocks_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    P1_FIRST_BEAR_ID = 67 + 5
    # NB: deliberately do NOT call declare_attacker — the BLOCKS
    # eligibility gate `if !self.attackers.contains(&attacker_id)` is
    # the first reject branch and bails before the dispatch line.

    ok = synthetic_state.declare_blocker(
        P0_FIRST_BEAR_ID, P1_FIRST_BEAR_ID
    )
    assert ok is False
    # No trigger on the stack — failure path didn't dispatch.
    assert synthetic_state.stack_size() == 0
    # Sanity: observer is still on the bf, just didn't observe anything.
    assert (
        synthetic_state.find_object_zone(observer_id) == ZONE_BATTLEFIELD
    )


# --- cross-engine bridge accepts "blocks" event ---------------------


def test_blocks_event_parses_via_cross_engine_bridge():
    """The cross_engine.py bridge accepts `event: blocks` and
    produces Python's `Event.BLOCKS`.
    """
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.effects.predicates import Event

    ta_dict = {
        "event": "blocks",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "draw_cards", "count": 1},
    }
    ta = _ta_dict_to_python_triggered_ability(ta_dict)
    assert ta.event is Event.BLOCKS


# ====================================================================
# Slice 10 — LTB fan-out to Effect-arm bf-exit sites in
# state_ops_resolve.rs.
#
# Slice 8 wired Event::LeavesBattlefield at the two central
# primitives (move_object + apply_sbas). Slice 10 extends fan-out to
# the 13 Effect-arm bf-exit sites in state_ops_resolve.rs (and the
# sacrifice-cost path in state_ops_actions.rs) that bypass
# move_object — calling `mark_left_battlefield_if_nonland` directly
# after a manual `Arc::make_mut(...).push(obj)`. Each gets a
# matching `pending_leaves_dispatches.push(leaving_obj_id) +
# drain_pending_leaves_dispatches` call.
#
# Pre-Slice-10, an LTB observer would silently miss bounces / exiles
# / destroys / sacrifices originating from these Effect arms — the
# bypass paths fire LTB at their Effect arm inline rather than
# routing through move_object's already-wired drain.
#
# These tests pin a representative subset of the 13 sites (one per
# major Effect family) so a future agent who refactors the
# bf-exit primitives doesn't accidentally regress the LTB fan-out
# silently.
# ====================================================================


def test_ltb_fires_on_return_to_hand_effect_arm(engine):
    """A `source_is_self` LEAVES trigger fires when a creature is
    bounced via the `return_to_hand` Effect arm (line ~819 in
    state_ops_resolve.rs). The Effect-arm bf-exit bypasses
    move_object, so this would silently miss pre-Slice-10.

    Setup: the LTB creature is on p1's bf (DEFENDER side, since
    `return_to_hand`'s `any_creature` target picks from the
    defender's bf), and the spell is on p0's hand.

    Controller-of-effect (post-DrawCards-fix): the LTB trigger's
    controller is p1 (the creature's controller). When the trigger
    fires its DrawCards effect, the resolver swaps in p1 as the
    source controller for `apply_effect`, so the draw lands on p1's
    library / hand. Pre-fix this was wrongly routed to p0
    (state.active_player); the feature/drawcards-controller-fix
    branch closed the divergence.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    crt_id = state.register_and_inject_object(
        _leaves_self_draw_dict(), 1, ZONE_BATTLEFIELD
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-Bounce Spell",
            "card_types": ["Instant"],
            "mana_cost": {"U": 1},
            "effects": [
                {
                    "kind": "return_to_hand",
                    "target": {"kind": "any_creature"},
                }
            ],
        },
        0,
        ZONE_HAND,
    )
    p0_lib_before = state.library_sizes[0]
    p1_lib_before = state.library_sizes[1]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Creature is now in p1's hand (its owner).
    assert state.find_object_zone(crt_id) == ZONE_HAND
    # The LTB trigger landed on the stack — pre-Slice-10 this
    # would have been 0.
    assert state.stack_size() == 1, (
        "expected the source_is_self LEAVES trigger to fire from "
        "the return_to_hand Effect arm"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == crt_id
    # Stack item carries the source's controller (p1).
    assert top["controller"] == 1

    # Resolve the trigger: post-fix the draw lands on p1 (the
    # trigger's controller), NOT p0 (active_player).
    state.resolve_top_of_stack()
    assert state.library_sizes[0] == p0_lib_before  # p0 unaffected
    assert state.library_sizes[1] == p1_lib_before - 1  # p1 drew 1


def test_ltb_fires_on_destroy_target_creature_effect_arm(engine):
    """LEAVES observer fires when a creature is destroyed via the
    `destroy_target_creature` Effect arm (line ~2931 in
    state_ops_resolve.rs). Pre-Slice-10 silent miss; Slice 10 closure.

    Same controller-of-effect (post-DrawCards-fix) shape as the
    return_to_hand test above: draw lands on the trigger's
    controller (p1), not state.active_player (p0).
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Victim creature on p1's bf with self-LTB-draw trigger.
    victim_id = state.register_and_inject_object(
        _leaves_self_draw_dict(), 1, ZONE_BATTLEFIELD
    )
    # Destroy-target spell in p0's hand (`destroy_target` with
    # `any_creature` filter — the same Doom Blade-shaped effect).
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-Destroy Spell",
            "card_types": ["Instant"],
            "mana_cost": {"B": 1, "generic": 1},
            "effects": [
                {
                    "kind": "destroy_target",
                    "target": {"kind": "any_creature"},
                }
            ],
        },
        0,
        ZONE_HAND,
    )
    p0_lib_before = state.library_sizes[0]
    p1_lib_before = state.library_sizes[1]
    grave_before = state.graveyard_sizes[1]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Victim is in p1's graveyard.
    assert state.find_object_zone(victim_id) == ZONE_GRAVEYARD
    assert state.graveyard_sizes[1] == grave_before + 1
    # The LTB trigger landed on the stack.
    assert state.stack_size() == 1, (
        "expected the source_is_self LEAVES trigger to fire from "
        "the destroy_target_creature Effect arm"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == victim_id

    # Resolve the trigger: post-fix draw lands on p1 (the trigger's
    # controller).
    state.resolve_top_of_stack()
    assert state.library_sizes[0] == p0_lib_before
    assert state.library_sizes[1] == p1_lib_before - 1


def test_ltb_observer_fires_on_destroy_target_artifact_effect_arm(engine):
    """A subject_is_creature LEAVES observer wired on p0's bf fires
    when ANY bf-exit happens — including via the
    `destroy_target_artifact` Effect arm (line ~2978). Verifies the
    observer-side dispatch path on a non-self trigger shape (the
    observer outlives the bf-exit).

    Note: destroy_target_artifact destroys an artifact, not a
    creature. The observer's `subject_is_creature` predicate rejects
    when the subject is a pure-Artifact. So the assertion is that
    the trigger does NOT fire (the predicate gates out). This pins
    that the dispatch IS happening — the predicate would never get
    a chance to filter if dispatch were silently skipped.

    To verify dispatch actually fires, we use a wider-net observer:
    `subject_is_artifact` (Slice 4 predicate) which matches the
    destroyed artifact.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Artifact observer with `subject_is_artifact` LEAVES trigger.
    observer = {
        "name": "Test LTB Artifact Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "subject_is_artifact"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = state.register_and_inject_object(observer, 0, ZONE_BATTLEFIELD)

    # Vanilla artifact victim on p1's bf.
    artifact_id = state.register_and_inject_object(
        {
            "name": "Test Junk Artifact (Slice 10)",
            "card_types": ["Artifact"],
            "mana_cost": {"generic": 2},
            "effects": [],
        },
        1,
        ZONE_BATTLEFIELD,
    )
    # Destroy-target-artifact spell in p0's hand.
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-DestroyArtifact Spell",
            "card_types": ["Instant"],
            "mana_cost": {"B": 1, "generic": 2},
            "effects": [{"kind": "destroy_target_artifact"}],
        },
        0,
        ZONE_HAND,
    )
    library_before = state.library_sizes[0]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Artifact destroyed → graveyard.
    assert state.find_object_zone(artifact_id) == ZONE_GRAVEYARD
    # Observer fires on the artifact's LTB.
    assert state.stack_size() == 1, (
        "expected the subject_is_artifact LEAVES observer to fire "
        "from the destroy_target_artifact Effect arm"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Resolve the trigger: p0 (observer's controller) draws 1.
    state.resolve_top_of_stack()
    assert state.library_sizes[0] == library_before - 1


def test_ltb_fires_on_exile_target_effect_arm(engine):
    """A `source_is_self` LEAVES trigger fires when a creature is
    exiled via the `exile_target` Effect arm (line ~982 in
    state_ops_resolve.rs). LTB fires for ALL bf-exits regardless of
    destination zone (graveyard / exile / hand / library).

    Controller-of-effect (post-DrawCards-fix): same as the bounce /
    destroy tests above — draw lands on the trigger's controller
    (p1), not state.active_player (p0).
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    crt_id = state.register_and_inject_object(
        _leaves_self_draw_dict(), 1, ZONE_BATTLEFIELD
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-Exile Spell",
            "card_types": ["Instant"],
            "mana_cost": {"W": 1, "generic": 1},
            "effects": [
                {
                    "kind": "exile_target",
                    "target": {"kind": "any_creature"},
                }
            ],
        },
        0,
        ZONE_HAND,
    )
    p0_lib_before = state.library_sizes[0]
    p1_lib_before = state.library_sizes[1]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Creature is now in p1's exile.
    assert state.find_object_zone(crt_id) == ZONE_EXILE
    # The LTB trigger landed on the stack — extra-observer pattern
    # (5-zone walk) finds the exiled creature for SourceIsSelf
    # matching.
    assert state.stack_size() == 1, (
        "expected the source_is_self LEAVES trigger to fire from "
        "the exile_target Effect arm"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == crt_id

    state.resolve_top_of_stack()
    assert state.library_sizes[0] == p0_lib_before
    assert state.library_sizes[1] == p1_lib_before - 1


def test_ltb_fires_on_destroy_target_land_effect_arm(engine):
    """A SubjectIsLand (Slice 4) LEAVES observer fires on Strip Mine-
    shape land destruction (line ~3026 — `destroy_target_land`).
    Verifies LTB fans out from the land-destroy Effect arm too.
    LTB fires for lands (unlike the void-flag, which gates on
    non-land).
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Land observer with `subject_is_land` LEAVES trigger.
    observer = {
        "name": "Test LTB Land Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "subject_is_land"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = state.register_and_inject_object(observer, 0, ZONE_BATTLEFIELD)
    # Vanilla land victim on p1's bf.
    land_id = state.register_and_inject_object(
        {
            "name": "Test Plains (Slice 10)",
            "card_types": ["Land"],
            "mana_cost": {},
            "effects": [],
        },
        1,
        ZONE_BATTLEFIELD,
    )
    # Destroy-target-land spell.
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-DestroyLand Spell",
            "card_types": ["Sorcery"],
            "mana_cost": {"R": 1},
            "effects": [{"kind": "destroy_target_land"}],
        },
        0,
        ZONE_HAND,
    )
    library_before = state.library_sizes[0]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Land destroyed → graveyard.
    assert state.find_object_zone(land_id) == ZONE_GRAVEYARD
    # Observer fires.
    assert state.stack_size() == 1, (
        "expected the subject_is_land LEAVES observer to fire from "
        "the destroy_target_land Effect arm (LTB fires for lands)"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    state.resolve_top_of_stack()
    assert state.library_sizes[0] == library_before - 1


def test_ltb_fires_on_sacrifice_self_cost_path(engine):
    """A `source_is_self` LEAVES trigger fires when its host is
    sacrificed via the `sacrifice_self` activation-cost path in
    state_ops_actions.rs (not the Effect-arm). Pins that the
    cost-path bf-exit also fans out LTB.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Creature with a free activated ability whose cost is
    # `sacrifice_self`. The LTB trigger fires when the host hits
    # the graveyard via the cost-payment path. Effect of the
    # activation is a no-op DrawCards(0) — we only care that the
    # cost-side bf-exit fires LTB. (The activated-ability path is
    # the recent CE20 Slice 2 work — sacrifice_self + tap_self
    # variants.)
    host_dict = {
        "name": "Test LTB-on-Sac Host",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
        "activated_abilities": [
            {
                "cost": {"sacrifice_self": True},
                "effect": {"kind": "draw_cards", "count": 0},
            }
        ],
    }
    host_id = state.register_and_inject_object(host_dict, 0, ZONE_BATTLEFIELD)
    library_before = state.library_sizes[0]

    # Activate: cost = sacrifice_self → host hits the graveyard.
    # LTB fires; resolving it draws 1 card.
    activate_action = {
        "kind": "activate_ability",
        "source_id": host_id,
        "ability_index": 0,
    }
    state.apply_action(activate_action)

    # Host is in p0's graveyard.
    assert state.find_object_zone(host_id) == ZONE_GRAVEYARD
    # The LTB trigger landed on the stack (the sacrifice-self
    # activation effect is a no-op DrawCards(0) that itself does
    # not push onto the stack — only the LTB trigger does).
    stack_n = state.stack_size()
    # The mana-ability bypass would resolve the effect immediately
    # without pushing onto the stack, but sacrifice_self is NOT a
    # mana ability (is_mana_ability is false unless the effect
    # produces mana). The activation's effect is a DrawCards(0)
    # which IS on the stack — so the stack has both the activation
    # effect AND the LTB trigger.
    assert stack_n >= 1, (
        f"expected at least 1 stack entry post-sacrifice-self "
        f"activation (got {stack_n}); the source_is_self LEAVES "
        f"trigger should have fired"
    )
    # Resolve the entire stack and assert library shrank by 1
    # (the LTB trigger drew 1 card; the activation effect's
    # DrawCards(0) is a no-op).
    while state.stack_size() > 0:
        state.resolve_top_of_stack()
    assert state.library_sizes[0] == library_before - 1, (
        "expected p0's library to shrink by exactly 1 (the LTB "
        "trigger's draw); got "
        f"{state.library_sizes[0]} vs {library_before}"
    )


# -------------------------------------------------------------------
# Slice 11 — DestroyAllCreatures LTB fan-out residual
#
# Closes the 14th LTB-silent Effect arm: `Effect::DestroyAllCreatures`
# (Wrath of God shape) at state_ops_resolve.rs:2887. Pre-Slice-11 this
# arm moved victims to graveyards directly and stamped the void flag
# inline (`state.permanent_left_battlefield_this_turn = true`) without
# pushing onto `pending_leaves_dispatches` — so a creature with a
# `source_is_self` LEAVES trigger would die silently on a Wrath. Slice
# 11 routes void-flag stamping through `mark_left_battlefield_if_nonland`
# (same as every other bf-exit) and queues each victim onto the LEAVES
# dispatch list (drained after the Wrath's stack pop).
# -------------------------------------------------------------------


def test_ltb_fires_on_destroy_all_creatures_effect_arm(engine):
    """A `source_is_self` LEAVES trigger fires when its host is
    destroyed via the `destroy_all_creatures` Effect arm (Wrath of
    God shape). Pre-Slice-11 silent miss; Slice 11 closure.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Single victim with a self-LTB-draw trigger on p0's bf.
    victim_id = state.register_and_inject_object(
        _leaves_self_draw_dict(), 0, ZONE_BATTLEFIELD
    )
    # Wrath-shaped sorcery in p0's hand.
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-on-Wrath Spell",
            "card_types": ["Sorcery"],
            "mana_cost": {"W": 2, "generic": 2},
            "effects": [{"kind": "destroy_all_creatures"}],
        },
        0,
        ZONE_HAND,
    )
    p0_lib_before = state.library_sizes[0]

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Victim is in p0's graveyard.
    assert state.find_object_zone(victim_id) == ZONE_GRAVEYARD
    # The LTB trigger landed on the stack post-Wrath-pop (Slice 10
    # defer-past-pop pattern: drain runs AFTER the resolving spell's
    # stack pop, so the LEAVES trigger lands on the now-empty top).
    assert state.stack_size() == 1, (
        f"expected the source_is_self LEAVES trigger to fire from "
        f"the destroy_all_creatures Effect arm; got stack_size="
        f"{state.stack_size()}"
    )
    top = state.peek_top_of_stack()
    assert top["source_obj_id"] == victim_id

    # Resolve the trigger: draw lands on active_player = p0.
    state.resolve_top_of_stack()
    assert state.library_sizes[0] == p0_lib_before - 1


def test_ltb_fires_for_each_victim_on_mass_destroy(engine):
    """Wrath destroys MULTIPLE creatures with LEAVES triggers.
    Per-victim LTB push: each victim gets its own LTB trigger on
    the stack. Pin: post-Wrath stack carries one trigger per victim,
    each with its source_obj_id matching the victim that was
    destroyed.

    Note: empty libraries would deck out resolving the draws, so we
    fix the count of victims small enough (2) to allow each LTB-draw
    to succeed.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    victim_a = state.register_and_inject_object(
        _leaves_self_draw_dict("LTB Victim A"), 0, ZONE_BATTLEFIELD
    )
    victim_b = state.register_and_inject_object(
        _leaves_self_draw_dict("LTB Victim B"), 1, ZONE_BATTLEFIELD
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test Multi-Victim Wrath",
            "card_types": ["Sorcery"],
            "mana_cost": {"W": 2, "generic": 2},
            "effects": [{"kind": "destroy_all_creatures"}],
        },
        0,
        ZONE_HAND,
    )

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Both victims in their respective graveyards.
    assert state.find_object_zone(victim_a) == ZONE_GRAVEYARD
    assert state.find_object_zone(victim_b) == ZONE_GRAVEYARD
    # Both LTB triggers landed on the stack.
    assert state.stack_size() == 2, (
        f"expected two source_is_self LEAVES triggers (one per "
        f"victim); got {state.stack_size()}"
    )


def test_ltb_observer_fires_on_destroy_all_creatures_effect_arm(engine):
    """A surviving (non-creature or indestructible) observer with a
    `subject_is_creature` LEAVES trigger fires when ANY creature dies
    to a Wrath. Pins the observer-side dispatch path on the mass-
    destroy arm.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    # Observer is an INDESTRUCTIBLE creature on p0's bf with a
    # `subject_is_creature` LEAVES trigger. It survives the Wrath and
    # observes the other victim's bf-exit.
    observer_dict = {
        "name": "Test LTB Observer (Indestructible)",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 3,
        "keywords": ["indestructible"],
        "triggered_abilities": [
            {
                "event": "leaves_battlefield",
                "predicate": {"kind": "subject_is_creature"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = state.register_and_inject_object(
        observer_dict, 0, ZONE_BATTLEFIELD
    )
    # Single destructible victim on p1's bf.
    victim_id = state.register_and_inject_object(
        _leaves_self_draw_dict("Plain Victim"), 1, ZONE_BATTLEFIELD
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test LTB-on-Wrath With Observer",
            "card_types": ["Sorcery"],
            "mana_cost": {"W": 2, "generic": 2},
            "effects": [{"kind": "destroy_all_creatures"}],
        },
        0,
        ZONE_HAND,
    )

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Observer survived, victim died.
    assert state.find_object_zone(observer_id) == ZONE_BATTLEFIELD
    assert state.find_object_zone(victim_id) == ZONE_GRAVEYARD
    # TWO LTB triggers fired: the victim's source_is_self AND the
    # observer's subject_is_creature on the victim's leaving event.
    assert state.stack_size() == 2, (
        f"expected 2 LTB triggers (victim self-LTB + observer "
        f"LTB on victim); got {state.stack_size()}"
    )


def test_void_flag_still_set_on_destroy_all_creatures(engine):
    """Slice 11 routes the void-flag stamping through
    `mark_left_battlefield_if_nonland` instead of stamping inline.
    Pin: the flag still goes true after a Wrath destroys a non-token
    creature (which is, by definition, nonland). The behavior was
    correct pre-Slice-11 via direct flag set; Slice 11 preserves it
    via the helper.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    state.register_and_inject_object(
        _leaves_self_draw_dict("Pinned Victim"), 0, ZONE_BATTLEFIELD
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test Void-Flag Wrath",
            "card_types": ["Sorcery"],
            "mana_cost": {"W": 2, "generic": 2},
            "effects": [{"kind": "destroy_all_creatures"}],
        },
        0,
        ZONE_HAND,
    )
    # Pre-Wrath void flag should be false (no recent bf-exit).
    pre = state.permanent_left_battlefield_this_turn
    # The synthetic state may have had bf-exits during setup; we only
    # assert the post-Wrath value is true (the helper stamps it).
    _ = pre  # silence unused warning if pre is already True from setup
    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()
    assert state.permanent_left_battlefield_this_turn is True, (
        "expected mark_left_battlefield_if_nonland to set the void "
        "flag after Wrath destroys a nonland creature"
    )


def test_no_ltb_fired_when_destroy_all_creatures_destroys_nothing(engine):
    """Wrath on an empty battlefield (no creatures) — no LTB
    triggers fire, no stack growth.
    """
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=0, num_library=5
    )
    spell_obj_id = state.register_and_inject_object(
        {
            "name": "Test Wrath On Empty Bf",
            "card_types": ["Sorcery"],
            "mana_cost": {"W": 2, "generic": 2},
            "effects": [{"kind": "destroy_all_creatures"}],
        },
        0,
        ZONE_HAND,
    )

    state.push_to_stack(spell_obj_id, 0)
    state.resolve_top_of_stack()

    # Stack is empty post-resolve — no LTB triggers (no victims).
    assert state.stack_size() == 0


# =====================================================================
# Subtypes substrate Slice 2 — `Predicate::SubjectHasSubtype(String)`
#
# Mirrors Python's `Subject(None, HasSubtype("Kavu"))` from
# `cards/parser/triggers.py:1198`. The canonical Kavu/Sliver/
# Equipment-enters predicate. Subject-anchored: matches the event's
# default subject slot ("obj" for ENTERS) — the subject's printing's
# `subtypes` Vec is searched for the configured subtype string.
#
# Effective-subtypes scope: reads the static `subtypes` field directly,
# not a CR 613 layer-3/4 effective-subtypes computation. EoE pool has
# no Conspiracy-shape subtype-granting effects, so the direct read
# is observably equivalent to the Python `HasSubtype` path.
# =====================================================================


def _bear_observer_dict():
    """An observer with 'whenever a Bear enters, draw a card'.
    Uses `subject_has_subtype("Bear")` — the canonical Kavu/Sliver
    shape (creature-subtype-anchored ETB trigger).
    """
    return {
        "name": "Test Bear Observer",
        "card_types": ["Creature"],
        "subtypes": ["Human", "Druid"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": "Bear",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _vanilla_bear_dict():
    """A vanilla 2/2 Bear creature — the canonical
    `subject_has_subtype("Bear")` positive trigger.
    """
    return {
        "name": "Test Grizzly Bears",
        "card_types": ["Creature"],
        "subtypes": ["Bear"],
        "mana_cost": {"G": 1, "generic": 1},
        "effects": [],
        "power": 2,
        "toughness": 2,
    }


# --- parser surface ---------------------------------------------------


def test_slice2_subject_has_subtype_predicate_parses(empty_registry):
    """A printing with `subject_has_subtype` parses without error."""
    d = {
        "name": "Slice2 HasSubtype Test",
        "card_types": ["Creature"],
        "subtypes": ["Cat"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": "Bear",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


def test_slice2_subject_has_subtype_missing_subtype_key_raises(empty_registry):
    """`subject_has_subtype` requires the `subtype` payload key. A
    dict missing it raises at register time — same loud-fail shape as
    every other required parser key.
    """
    import pytest
    d = {
        "name": "Bad HasSubtype",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "subject_has_subtype"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(KeyError, match="subtype"):
        empty_registry.register_printing(d)


def test_slice2_subject_has_subtype_non_string_subtype_raises(empty_registry):
    """`subtype` payload must be a string — int/None/list raise."""
    import pytest
    d = {
        "name": "Bad HasSubtype Int",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": 42,
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="subtype"):
        empty_registry.register_printing(d)


# --- positive: matching subtype fires the trigger ---------------------


def test_slice2_bear_observer_fires_on_bear_arrival(synthetic_state):
    """A pre-existing observer with `subject_has_subtype("Bear")` fires
    when a Bear creature enters the battlefield. Drives the canonical
    Kavu/Sliver-shape predicate.

    Setup: observer (non-Bear) on p0's bf + Bear-subtype creature in
    p0's hand. Cast the Bear -> trigger fires.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _bear_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    bear_id = synthetic_state.register_and_inject_object(
        _vanilla_bear_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bear_id, 0)
    synthetic_state.resolve_top_of_stack()

    # Bear is on bf; observer's trigger is on the stack.
    assert synthetic_state.find_object_zone(bear_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `subject_has_subtype('Bear')` to fire on the Bear's "
        "arrival; stack is empty so the predicate rejected the Bear"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


# --- negative: non-Bear subject doesn't fire ---------------------------


def test_slice2_bear_observer_does_not_fire_on_non_bear(synthetic_state):
    """`subject_has_subtype('Bear')` rejects a creature that has
    different subtypes. A vanilla creature with no subtypes also
    rejects.
    """
    synthetic_state.register_and_inject_object(
        _bear_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Vanilla creature with no subtypes — should not fire.
    non_bear_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(non_bear_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(non_bear_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `subject_has_subtype('Bear')` to reject a creature "
        "with empty subtypes; stack is non-empty"
    )


def test_slice2_bear_observer_does_not_fire_on_wrong_subtype(synthetic_state):
    """`subject_has_subtype('Bear')` rejects a creature carrying a
    different subtype ('Cat'). Pins case-sensitive exact-string match.
    """
    synthetic_state.register_and_inject_object(
        _bear_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    cat_id = synthetic_state.register_and_inject_object(
        {
            "name": "Test Vanilla Cat",
            "card_types": ["Creature"],
            "subtypes": ["Cat"],
            "mana_cost": {"generic": 1},
            "effects": [],
            "power": 1,
            "toughness": 1,
        },
        0,
        ZONE_HAND,
    )
    synthetic_state.push_to_stack(cat_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(cat_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `subject_has_subtype('Bear')` to reject a Cat subject"
    )


# --- composition: AND with ControlledByYou ----------------------------


def test_slice2_bear_observer_combined_with_controlled_by_you(synthetic_state):
    """The canonical Kavu shape from `triggers.py:1198`:
    `Subject(None, HasSubtype('Bear')) & Controlled(YOU)` — "whenever
    a Bear you control enters". Composes `subject_has_subtype` with
    `controlled_by_you` via `and`.
    """
    observer_dict = {
        "name": "Kavu-Shape Bear Observer",
        "card_types": ["Creature"],
        "subtypes": ["Human"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {
                        "kind": "subject_has_subtype",
                        "subtype": "Bear",
                    },
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    # Observer controlled by p0.
    observer_id = synthetic_state.register_and_inject_object(
        observer_dict, 0, ZONE_BATTLEFIELD
    )
    # P0 casts their own Bear — should fire (own Bear + controlled).
    p0_bear_id = synthetic_state.register_and_inject_object(
        _vanilla_bear_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(p0_bear_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(p0_bear_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `Subject(HasSubtype('Bear')) AND Controlled(YOU)` to "
        "fire on the controller's own Bear"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id

    # Drain the trigger off the stack to reset for the negative case.
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.is_stack_empty()

    # P1 casts their Bear — Bear matches, but ControlledByYou rejects.
    p1_bear_id = synthetic_state.register_and_inject_object(
        _vanilla_bear_dict(), 1, ZONE_HAND
    )
    synthetic_state.push_to_stack(p1_bear_id, 1)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(p1_bear_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `Controlled(YOU)` to reject p1's Bear; observer's "
        "trigger should NOT fire on opponent's Bear"
    )


# --- multi-subtype subject -------------------------------------------


def test_slice2_subject_has_subtype_matches_one_of_multiple(synthetic_state):
    """A creature with multiple subtypes ("Elf Druid") matches
    `subject_has_subtype` for either of them (set-membership check).
    """
    observer_dict = {
        "name": "Druid-Tribal Observer",
        "card_types": ["Creature"],
        "subtypes": [],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": "Druid",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(
        observer_dict, 0, ZONE_BATTLEFIELD
    )
    elf_druid_id = synthetic_state.register_and_inject_object(
        {
            "name": "Test Elf Druid",
            "card_types": ["Creature"],
            "subtypes": ["Elf", "Druid"],
            "mana_cost": {"G": 1},
            "effects": [],
            "power": 1,
            "toughness": 1,
        },
        0,
        ZONE_HAND,
    )
    synthetic_state.push_to_stack(elf_druid_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(elf_druid_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `subject_has_subtype('Druid')` to fire on a creature "
        "carrying ['Elf', 'Druid']; the second-position subtype was "
        "missed"
    )


# --- case-sensitivity guard ------------------------------------------


def test_slice2_subject_has_subtype_is_case_sensitive(synthetic_state):
    """Subtype match is case-sensitive: registering an observer that
    looks for "bear" (lowercase) does NOT fire on a Bear-subtype
    (Title Case) creature. Sanity check that we're not accidentally
    case-insensitive (which would diverge from Python's behavior).
    """
    observer_dict = {
        "name": "lowercase-bear observer",
        "card_types": ["Creature"],
        "subtypes": [],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": "bear",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(
        observer_dict, 0, ZONE_BATTLEFIELD
    )
    bear_id = synthetic_state.register_and_inject_object(
        _vanilla_bear_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bear_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bear_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected case-sensitive match: lowercase 'bear' should not "
        "match Title Case 'Bear' subtype"
    )


# --- equipment ETB shape ---------------------------------------------


def test_slice2_subject_has_subtype_equipment_arrival_fires(synthetic_state):
    """The Equipment-anchored ETB shape ("whenever an Equipment
    enters, do X") fires for an Artifact — Equipment subject.
    Confirms the predicate works for artifact-subtype use cases
    too (cards.rs:1511 callsite was waiting on this).
    """
    observer_dict = {
        "name": "Equipment Observer",
        "card_types": ["Creature"],
        "subtypes": [],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject_has_subtype",
                    "subtype": "Equipment",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(
        observer_dict, 0, ZONE_BATTLEFIELD
    )
    equip_id = synthetic_state.register_and_inject_object(
        {
            "name": "Test Sword",
            "card_types": ["Artifact"],
            "subtypes": ["Equipment"],
            "mana_cost": {"generic": 2},
            "effects": [],
        },
        0,
        ZONE_HAND,
    )
    synthetic_state.push_to_stack(equip_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(equip_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `subject_has_subtype('Equipment')` to fire on the "
        "Equipment's arrival"
    )


# =====================================================================
# Or-substrate slice — logical disjunction over two sub-predicates.
#
# Slice 1 introduced `not` / `and`; Slice 4 broadened the leaf
# alphabet. The Or-substrate slice closes the last missing
# Boolean-connective gap: shapes like Weftstalker Ardent's "whenever
# a creature or artifact you control enters" need disjunction, not
# conjunction. Pattern mirrors the existing `and` arm exactly
# (parse + dispatch + harness translation).
#
# Coverage:
# - Parser accepts `kind: or` with `left` / `right` sub-dicts.
# - Parser rejects malformed `or` payloads (missing left/right, or
#   non-dict children).
# - `Or(SubjectIsCreature, SubjectIsArtifact)` fires on either a
#   creature OR an artifact subject (and not on a land — neither
#   side accepts).
# - Short-circuit: when the LEFT predicate matches, the right side
#   is irrelevant (verified by stacking a left-true / right-false
#   combination).
# - Nesting: `Or` composes inside `And` to express
#   `(creature OR artifact) AND controlled_by_you` — the Weftstalker
#   Ardent shape.
# =====================================================================


def _or_creature_or_artifact_observer_dict():
    """Pre-existing observer with predicate
    `Or(SubjectIsCreature, SubjectIsArtifact)`. Fires when EITHER a
    creature OR an artifact arrives — regardless of controller.
    The canonical "creature or artifact" disjunction shape.
    """
    return {
        "name": "Test Creature-Or-Artifact Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "or",
                    "left": {"kind": "subject_is_creature"},
                    "right": {"kind": "subject_is_artifact"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface -------------------------------------------------


def test_or_predicate_kind_parses(empty_registry):
    """The `or` predicate kind parses + registers without error."""
    pid = empty_registry.register_printing(
        _or_creature_or_artifact_observer_dict()
    )
    assert pid == 0


def test_or_predicate_missing_left_raises(empty_registry):
    """Malformed `or` (missing `left`) raises at register time."""
    d = {
        "name": "Bad Or Missing Left",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "or",
                    "right": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises((ValueError, KeyError), match='left'):
        empty_registry.register_printing(d)


def test_or_predicate_missing_right_raises(empty_registry):
    """Malformed `or` (missing `right`) raises at register time."""
    d = {
        "name": "Bad Or Missing Right",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "or",
                    "left": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises((ValueError, KeyError), match='right'):
        empty_registry.register_printing(d)


def test_or_predicate_non_dict_child_raises(empty_registry):
    """`or` with a non-dict child raises at register time
    (mirrors `and`'s downcast-failure shape).
    """
    d = {
        "name": "Bad Or Non-Dict Child",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "or",
                    "left": "not a dict",
                    "right": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="must be a dict"):
        empty_registry.register_printing(d)


# --- dispatch — Or(creature, artifact) fires on either subject ------


def test_or_fires_on_left_match_creature(synthetic_state):
    """`Or(SubjectIsCreature, SubjectIsArtifact)` fires on a CREATURE
    subject — the left arm matches.
    """
    observer_id = synthetic_state.register_and_inject_object(
        _or_creature_or_artifact_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Drain the observer's self-firing (it's a Creature — the left
    # arm of its own Or predicate matches its own arrival).
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()

    # Cast a vanilla creature for p0; observer fires again.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    assert synthetic_state.stack_size() == 1, (
        "expected `Or(SubjectIsCreature, SubjectIsArtifact)` to fire on "
        "a creature subject (left arm matches)"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id


def test_or_fires_on_right_match_artifact(synthetic_state):
    """`Or(SubjectIsCreature, SubjectIsArtifact)` fires on an ARTIFACT
    subject — the right arm matches even though the left fails.
    """
    synthetic_state.register_and_inject_object(
        _or_creature_or_artifact_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Drain the observer's self-firing.
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()

    # Reanimate a vanilla artifact for p0 (artifact-only card_types).
    artifact_id = synthetic_state.register_and_inject_object(
        _vanilla_artifact_dict(), 0, ZONE_GRAVEYARD
    )
    ok = synthetic_state.move_object_from_graveyard_to_battlefield(
        0, artifact_id
    )
    assert ok is True
    assert synthetic_state.find_object_zone(artifact_id) == ZONE_BATTLEFIELD

    assert synthetic_state.stack_size() == 1, (
        "expected `Or(SubjectIsCreature, SubjectIsArtifact)` to fire on "
        "an artifact subject (right arm matches, left arm fails)"
    )


def test_or_does_not_fire_on_neither_match_land(synthetic_state):
    """`Or(SubjectIsCreature, SubjectIsArtifact)` does NOT fire on a
    LAND subject — neither arm matches.
    """
    synthetic_state.register_and_inject_object(
        _or_creature_or_artifact_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Drain self-firing.
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()

    # Inject a land directly onto p0's bf — dispatches Enters but
    # the Or predicate rejects (neither Creature nor Artifact).
    synthetic_state.register_and_inject_object(
        _vanilla_land_dict(), 0, ZONE_BATTLEFIELD
    )
    assert synthetic_state.is_stack_empty(), (
        "expected `Or(SubjectIsCreature, SubjectIsArtifact)` to reject "
        "a land subject (neither arm matches)"
    )


# --- Or composes inside And — Weftstalker-Ardent-shape predicate ---


def test_or_composes_under_and_for_weftstalker_shape(synthetic_state):
    """The Weftstalker Ardent-shape predicate:
    `(SubjectIsCreature OR SubjectIsArtifact) AND ControlledByYou`.
    Fires on YOUR creature OR YOUR artifact; rejects OPPONENT's
    creature or land.
    """
    weftstalker_shape = {
        "name": "Test Weftstalker-Shape Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {
                        "kind": "or",
                        "left": {"kind": "subject_is_creature"},
                        "right": {"kind": "subject_is_artifact"},
                    },
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        weftstalker_shape, 0, ZONE_BATTLEFIELD
    )
    # Drain observer's own self-firing (observer is a Creature
    # controlled by source's controller — predicate matches).
    assert synthetic_state.stack_size() == 1
    synthetic_state.resolve_top_of_stack()

    # YOUR creature — fires.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()
    assert synthetic_state.stack_size() == 1, (
        "expected (creature OR artifact) AND controlled_by_you to fire "
        "on YOUR creature"
    )
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id
    synthetic_state.resolve_top_of_stack()

    # OPPONENT's creature on p1's bf — `controlled_by_you` rejects.
    synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 1, ZONE_BATTLEFIELD
    )
    assert synthetic_state.is_stack_empty(), (
        "expected `controlled_by_you` outer guard to reject opponent's "
        "creature subject"
    )


# ====================================================================
# Subject substrate — Predicate::Subject(slot, inner)
# ====================================================================
#
# Adds the `Predicate::Subject(SubjectSlot, Box<Predicate>)` variant
# which pulls a named subject slot out of a multi-subject event's
# slot map and evaluates `inner` against that subject. Slot strings:
# `"source"` / `"target"` mirror Python's `EVENT_SUBJECTS` keys for
# `Event.DEALS_COMBAT_DAMAGE`.
#
# **Scope.** This slice ships the substrate ONLY — parser arm,
# evaluation arm, cross-engine bridge. No consumer event yet supplies
# the slot bindings (that lands with `Event::DealsCombatDamage` in a
# follow-up slice). Therefore: on every existing single-subject event
# (ENTERS / DIES / CAST / ATTACKS / BLOCKS / LEAVES), a predicate of
# the form `Subject(slot, inner)` evaluates `false` because the slot
# isn't bound. The tests below pin that contract — the new variant
# parses, registers, and silently no-ops on today's single-subject
# events. The multi-subject dispatch path will exercise the firing
# arm in lockstep with the consumer slice.


def _subject_creature_observer_dict():
    """An observer with `Predicate::Subject(Source, IsCreature)` —
    parses + registers, but on single-subject events the `source`
    slot isn't bound so the predicate evaluates false. Used as the
    smoke fixture for the "named slot is unbound on single-subject
    events" contract.
    """
    return {
        "name": "Test Subject(Source, IsCreature) Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "source",
                    "inner": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


def _subject_target_creature_observer_dict():
    """An observer with `Predicate::Subject(Target, IsCreature)` —
    the `target`-slot sibling of `_subject_creature_observer_dict`.
    """
    return {
        "name": "Test Subject(Target, IsCreature) Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "target",
                    "inner": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }


# --- parser surface --------------------------------------------------


def test_subject_predicate_with_source_slot_parses(empty_registry):
    """`{"kind": "subject", "slot": "source", "inner": {...}}` parses
    and registers without raising.
    """
    pid = empty_registry.register_printing(
        _subject_creature_observer_dict()
    )
    assert pid == 0


def test_subject_predicate_with_target_slot_parses(empty_registry):
    """The sibling `slot="target"` form also parses + registers."""
    pid = empty_registry.register_printing(
        _subject_target_creature_observer_dict()
    )
    assert pid == 0


def test_subject_predicate_unknown_slot_raises(empty_registry):
    """An unknown slot string (e.g. `"src"` typo) raises at register
    time so a misspelled slot never silently no-ops at dispatch.
    """
    d = {
        "name": "Bad Subject Slot Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "src",
                    "inner": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(
        ValueError, match="unknown predicate.subject.slot"
    ):
        empty_registry.register_printing(d)


def test_subject_predicate_missing_inner_raises(empty_registry):
    """Missing the required `inner` field raises. `require` raises
    `KeyError` (PyO3-side) for missing dict keys, mirroring every
    other parser-arm's missing-key contract.
    """
    d = {
        "name": "No-Inner Subject Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "source",
                    # No "inner" key.
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(KeyError, match="inner"):
        empty_registry.register_printing(d)


def test_subject_predicate_inner_must_be_dict(empty_registry):
    """Non-dict `inner` raises with the canonical message."""
    d = {
        "name": "Non-Dict Inner Subject Card",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "source",
                    "inner": "not a dict",
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    with pytest.raises(
        ValueError, match="predicate.subject.inner must be a dict"
    ):
        empty_registry.register_printing(d)


# --- single-subject dispatch: Subject(slot, inner) silently no-ops --


def test_subject_predicate_no_ops_on_single_subject_enters(synthetic_state):
    """On the single-subject ENTERS event, today's dispatch passes no
    slot bindings — `Predicate::Subject(slot, inner)` evaluates false
    and the trigger doesn't fire. Pins the substrate's
    back-compat-on-single-subject-events contract.
    """
    synthetic_state.register_and_inject_object(
        _subject_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    # Cast a vanilla creature — would fire if the slot were bound.
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `Subject(Source, IsCreature)` to evaluate false on "
        "single-subject ENTERS dispatch (slot map is empty); the "
        "trigger should NOT push onto the stack"
    )


def test_subject_target_predicate_no_ops_on_single_subject_enters(
    synthetic_state,
):
    """Same shape with `Subject(Target, ...)` — also no-ops on
    single-subject ENTERS dispatch (no slot binding for "target").
    """
    synthetic_state.register_and_inject_object(
        _subject_target_creature_observer_dict(), 0, ZONE_BATTLEFIELD
    )
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.find_object_zone(bait_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_stack_empty(), (
        "expected `Subject(Target, IsCreature)` to no-op on "
        "single-subject ENTERS dispatch (target slot is unbound)"
    )


# --- compositional shape: Not + And + Subject -----------------------


def test_subject_predicate_composes_with_and(synthetic_state):
    """`AND(Subject(Source, IsCreature), SubjectIsCreature)` — the
    Subject half evaluates false on single-subject ENTERS, so the AND
    is false regardless of the right operand. Pins the recursive
    evaluation: `Subject` returning false in a compound predicate
    short-circuits the AND correctly.
    """
    d = {
        "name": "Test Compound Subject AND Card",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {
                        "kind": "subject",
                        "slot": "source",
                        "inner": {"kind": "subject_is_creature"},
                    },
                    "right": {"kind": "subject_is_creature"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    synthetic_state.register_and_inject_object(d, 0, ZONE_BATTLEFIELD)
    bait_id = synthetic_state.register_and_inject_object(
        _vanilla_creature_dict(), 0, ZONE_HAND
    )
    synthetic_state.push_to_stack(bait_id, 0)
    synthetic_state.resolve_top_of_stack()

    assert synthetic_state.is_stack_empty(), (
        "expected AND(Subject(Source,...), SubjectIsCreature) to short-"
        "circuit false on single-subject ENTERS (Subject half is false)"
    )


def test_not_subject_predicate_evaluates_true_on_single_subject_enters(
    synthetic_state,
):
    """`Not(Subject(Source, IsCreature))` — Subject evaluates false on
    single-subject events, so NOT(...) is true. This is observable: a
    trigger with that predicate FIRES on every single-subject ENTERS.
    Pins the recursive evaluation: `Subject` returning false flows
    through NOT correctly.
    """
    d = {
        "name": "Test NOT Subject Observer",
        "card_types": ["Creature"],
        "mana_cost": {},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "not",
                    "inner": {
                        "kind": "subject",
                        "slot": "source",
                        "inner": {"kind": "subject_is_creature"},
                    },
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    observer_id = synthetic_state.register_and_inject_object(
        d, 0, ZONE_BATTLEFIELD
    )
    # Drain observer's own self-firing (NOT(Subject(...)) is true on
    # the observer's own ETB too).
    assert synthetic_state.stack_size() == 1
    top = synthetic_state.peek_top_of_stack()
    assert top["source_obj_id"] == observer_id
    synthetic_state.resolve_top_of_stack()


# --- inner Subject variants: every supported object-tier kind -------


def test_subject_predicate_with_subject_is_land_inner_parses(
    empty_registry,
):
    """The inner predicate can be any object-tier check the substrate
    supports. Pin `subject_is_land`.
    """
    d = {
        "name": "Test Subject(Source, IsLand) Observer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "source",
                    "inner": {"kind": "subject_is_land"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


def test_subject_predicate_with_subject_is_artifact_inner_parses(
    empty_registry,
):
    """And the `subject_is_artifact` sibling."""
    d = {
        "name": "Test Subject(Source, IsArtifact) Observer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "target",
                    "inner": {"kind": "subject_is_artifact"},
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0


def test_subject_predicate_with_subject_has_subtype_inner_parses(
    empty_registry,
):
    """The `subject_has_subtype` inner — the Kavu-shape from
    `cards/parser/triggers.py:1198`, now slot-bindable.
    """
    d = {
        "name": "Test Subject(Source, HasSubtype Kavu) Observer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "subject",
                    "slot": "source",
                    "inner": {
                        "kind": "subject_has_subtype",
                        "subtype": "Kavu",
                    },
                },
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    pid = empty_registry.register_printing(d)
    assert pid == 0
