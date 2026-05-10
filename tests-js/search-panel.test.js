// Tests for the unified Decision panel renderer (renderSearchPanel) in
// app.js. Verifies the panel hides for snapshots without ``search`` and
// without engine-state recommendations, shows + tabulates candidates for
// annotated snapshots, marks the chosen row, formats numbers to 3 dp,
// supports the collapse toggle, and falls back to /api/recommend for
// snapshots that have engine state but no annotation.

import './setup.js';
import { describe, it, expect, vi } from 'vitest';

function _baseSnap() {
  return {
    turn_number: 1,
    step: 'precombat_main',
    is_over: false,
    is_draw: false,
    winner_id: null,
    active_player_idx: 0,
    log: [],
    players: [
      {
        id: 0, life: 20, hand: [], hand_size: 0, library: [],
        library_size: 30, graveyard: [], graveyard_size: 0,
        exile: [], exile_size: 0, battlefield: [], mana_pool: [],
        poison_counters: 0, has_lost: false,
      },
      {
        id: 1, life: 20, hand: [], hand_size: 0, library: [],
        library_size: 30, graveyard: [], graveyard_size: 0,
        exile: [], exile_size: 0, battlefield: [], mana_pool: [],
        poison_counters: 0, has_lost: false,
      },
    ],
    stack: [],
    decision: null,
  };
}

describe('search-panel module surface', () => {
  it('exposes renderSearchPanel and toggleSearchPanel as globals', () => {
    expect(typeof globalThis.renderSearchPanel).toBe('function');
    expect(typeof globalThis.toggleSearchPanel).toBe('function');
  });
});

describe('renderSearchPanel hidden states', () => {
  it('hides the panel when snap.search is absent', () => {
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    const panel = document.getElementById('search-panel');
    expect(panel.classList.contains('hidden')).toBe(true);
  });

  it('hides the panel when snap.search is null', () => {
    const snap = _baseSnap();
    snap.search = null;
    globalThis.renderSearchPanel(snap);
    expect(document.getElementById('search-panel').classList.contains('hidden')).toBe(true);
  });

  it('hides the panel when candidates list is empty', () => {
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [],
      samples: 1, inner_nodes: 50, elapsed_ms: 5,
    };
    globalThis.renderSearchPanel(snap);
    expect(document.getElementById('search-panel').classList.contains('hidden')).toBe(true);
  });
});

describe('renderSearchPanel populated states', () => {
  function _annotated() {
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        {
          action: { kind: 'PassPriority' }, visits: 32,
          q: 0.4213, prior: 0.7, leaf_value: 0.3982,
        },
        {
          action: { kind: 'PlayLand', card_id: 5 }, visits: 7,
          q: -0.1, prior: 0.2, leaf_value: -0.05,
        },
        {
          action: { kind: 'CastSpell', card_id: 9, kicked: true },
          visits: 0, q: null, prior: 0.05, leaf_value: null,
        },
      ],
      samples: 8, inner_nodes: 200, elapsed_ms: 42,
    };
    return snap;
  }

  it('shows the panel and tabulates each candidate', () => {
    globalThis.renderSearchPanel(_annotated());
    const panel = document.getElementById('search-panel');
    expect(panel.classList.contains('hidden')).toBe(false);
    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows.length).toBe(3);
  });

  it('sorts candidates by visits descending', () => {
    globalThis.renderSearchPanel(_annotated());
    const rows = Array.from(document.querySelectorAll('#search-rows tr'));
    const visits = rows.map(r => parseInt(r.children[1].textContent, 10));
    expect(visits).toEqual([32, 7, 0]);
  });

  it('marks the chosen candidate row with the .chosen class', () => {
    globalThis.renderSearchPanel(_annotated());
    const chosenRows = document.querySelectorAll('#search-rows tr.chosen');
    expect(chosenRows.length).toBe(1);
    // Action cell is column 0; chosen action is PassPriority.
    expect(chosenRows[0].children[0].textContent).toBe('PassPriority');
  });

  it('formats Q / prior / leaf_value to 3 decimal places', () => {
    globalThis.renderSearchPanel(_annotated());
    const firstRow = document.querySelector('#search-rows tr');
    // Columns: action, visits, q, prior, leaf_value.
    expect(firstRow.children[2].textContent).toBe('0.421');
    expect(firstRow.children[3].textContent).toBe('0.700');
    expect(firstRow.children[4].textContent).toBe('0.398');
  });

  it('renders an em dash for null Q / null leaf_value', () => {
    globalThis.renderSearchPanel(_annotated());
    const rows = document.querySelectorAll('#search-rows tr');
    // Last row has q=null and leaf_value=null.
    const last = rows[rows.length - 1];
    expect(last.children[2].textContent).toBe('—');
    expect(last.children[4].textContent).toBe('—');
    // But prior=0.05 still renders as a number.
    expect(last.children[3].textContent).toBe('0.050');
  });

  it('formats budget as samples × inner_nodes plus total visits and ms', () => {
    globalThis.renderSearchPanel(_annotated());
    const budget = document.getElementById('search-budget').textContent;
    expect(budget).toContain('8 × 200 nodes');
    expect(budget).toContain('39 total visits');  // 32 + 7 + 0
    expect(budget).toContain('42 ms');
  });

  it('falls back to "N samples" when inner_nodes is null', () => {
    const snap = _annotated();
    snap.search.inner_nodes = null;
    globalThis.renderSearchPanel(snap);
    const budget = document.getElementById('search-budget').textContent;
    expect(budget).toContain('8 samples');
    expect(budget).not.toContain('×');
  });

  it('omits the elapsed-ms readout when the field is null', () => {
    const snap = _annotated();
    snap.search.elapsed_ms = null;
    globalThis.renderSearchPanel(snap);
    const budget = document.getElementById('search-budget').textContent;
    expect(budget).not.toContain('ms');
  });

  it('shows the chosen action label in the header', () => {
    globalThis.renderSearchPanel(_annotated());
    const chosen = document.getElementById('search-chosen').textContent;
    expect(chosen).toContain('PassPriority');
    expect(chosen.toLowerCase()).toContain('chose');
  });
});

describe('toggleSearchPanel', () => {
  it('toggles the body visibility and updates the toggle glyph', () => {
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [{
        action: { kind: 'PassPriority' }, visits: 1,
        q: null, prior: null, leaf_value: null,
      }],
      samples: 1, inner_nodes: 1, elapsed_ms: 1,
    };
    globalThis.renderSearchPanel(snap);
    const body = document.getElementById('search-body');
    const toggle = document.getElementById('search-toggle');
    // Starts expanded.
    expect(body.classList.contains('hidden')).toBe(false);
    globalThis.toggleSearchPanel();
    expect(body.classList.contains('hidden')).toBe(true);
    expect(toggle.textContent.trim()).toBe('+');
    globalThis.toggleSearchPanel();
    expect(body.classList.contains('hidden')).toBe(false);
    // The toggle uses an HTML entity for the minus sign; either '-'
    // or '−' (U+2212) is acceptable. Just confirm it's not '+'.
    expect(toggle.textContent.trim()).not.toBe('+');
  });
});

describe('formatAction helper', () => {
  it('renders bare-kind decisions as just the kind', () => {
    expect(globalThis.formatAction({ kind: 'PassPriority' })).toBe('PassPriority');
  });

  it('renders kind(field=value, ...) for decisions with extra fields', () => {
    expect(globalThis.formatAction({ kind: 'PlayLand', card_id: 7 }))
      .toBe('PlayLand(card_id=7)');
  });

  it('summarizes list-valued fields by length', () => {
    const out = globalThis.formatAction({
      kind: 'DeclareAttackers', attacker_ids: [1, 2, 3],
    });
    expect(out).toBe('DeclareAttackers(attacker_ids=[3])');
  });

  // ----- name resolution -------------------------------------------------

  it('resolves PlayLand card_id to the card name', () => {
    const map = new Map([[71, 'Swamp']]);
    expect(globalThis.formatAction({ kind: 'PlayLand', card_id: 71 }, map))
      .toBe('PlayLand — Swamp');
  });

  it('resolves CastSpell card_id to the card name', () => {
    const map = new Map([[51, 'Hylderblade']]);
    expect(globalThis.formatAction(
      { kind: 'CastSpell', card_id: 51, use_warp: false, kicked: false,
        cast_for_free: false, use_impending: false, cast_room_half: null },
      map,
    )).toBe('CastSpell — Hylderblade');
  });

  it('appends modifier tags for CastSpell when set', () => {
    const map = new Map([[22, 'Rayblade Trooper']]);
    expect(globalThis.formatAction(
      { kind: 'CastSpell', card_id: 22, use_warp: true, kicked: false,
        cast_for_free: false, use_impending: false, cast_room_half: null },
      map,
    )).toBe('CastSpell — Rayblade Trooper (warp)');
  });

  it('resolves ActivateAbility source_id and keeps the ability index', () => {
    const map = new Map([[12, 'Sol Ring']]);
    expect(globalThis.formatAction(
      { kind: 'ActivateAbility', source_id: 12, ability_index: 0 },
      map,
    )).toBe('ActivateAbility — Sol Ring [0]');
  });

  it('renders AttackerDecision(attack=true) as "Attack — <name>"', () => {
    const map = new Map([[44, 'Comet Crawler']]);
    expect(globalThis.formatAction(
      { kind: 'AttackerDecision', attacker_id: 44, attack: true },
      map,
    )).toBe('Attack — Comet Crawler');
  });

  it('renders AttackerDecision(attack=false) with a "no attack" suffix', () => {
    const map = new Map([[44, 'Comet Crawler']]);
    expect(globalThis.formatAction(
      { kind: 'AttackerDecision', attacker_id: 44, attack: false },
      map,
    )).toBe('AttackerDecision — Comet Crawler (no attack)');
  });

  it('renders BlockerDecision as "Block — <blocker> blocks <attacker>"', () => {
    const map = new Map([
      [1, 'Dauntless Scrapbot'],
      [62, 'Nova Hellkite'],
    ]);
    expect(globalThis.formatAction(
      { kind: 'BlockerDecision', blocker_id: 1, attacker_id: 62 },
      map,
    )).toBe('Block — Dauntless Scrapbot blocks Nova Hellkite');
  });

  it('falls back to legacy form when an id is unresolvable', () => {
    // Empty map: no name found, legacy id-only string returned.
    expect(globalThis.formatAction(
      { kind: 'PlayLand', card_id: 71 }, new Map(),
    )).toBe('PlayLand(card_id=71)');
  });

  it('falls back to legacy form when only one of two ids resolves', () => {
    const map = new Map([[1, 'Dauntless Scrapbot']]);
    // attacker_id missing → legacy.
    const out = globalThis.formatAction(
      { kind: 'BlockerDecision', blocker_id: 1, attacker_id: 62 },
      map,
    );
    expect(out).toBe('BlockerDecision(blocker_id=1, attacker_id=62)');
  });

  it('treats undefined idToName as an empty map', () => {
    // No second arg — should not crash and should fall back.
    expect(globalThis.formatAction({ kind: 'PlayLand', card_id: 71 }))
      .toBe('PlayLand(card_id=71)');
  });

  it('leaves unknown kinds in the legacy stringified form', () => {
    expect(globalThis.formatAction(
      { kind: 'MysteryNewDecision', some_id: 99, flag: true },
      new Map([[99, 'Unrelated Card']]),
    )).toBe('MysteryNewDecision(some_id=99, flag=true)');
  });

  it('PassPriority is unaffected by an idToName map', () => {
    expect(globalThis.formatAction({ kind: 'PassPriority' }, new Map([[1, 'X']])))
      .toBe('PassPriority');
  });
});

describe('search panel renders names from snapshot zones', () => {
  it('uses card names from the current snapshot for candidate labels', () => {
    // Install a game whose current snapshot has a hand card with id 71.
    globalThis.game = {
      gameId: 'g1', currentIdx: 0, snapshotCount: 1,
      snapCache: new Map(), index: [], fullLog: [],
      playing: false, playTimer: null,
    };
    const snap = _baseSnap();
    snap.players[1].hand = [
      { id: 71, name: 'Swamp' },
      { id: 51, name: 'Hylderblade' },
    ];
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        { action: { kind: 'PassPriority' }, visits: 100,
          q: 0.1, prior: 0.5, leaf_value: 0.0 },
        { action: { kind: 'PlayLand', card_id: 71 }, visits: 30,
          q: 0.2, prior: 0.3, leaf_value: 0.0 },
        { action: { kind: 'CastSpell', card_id: 51, use_warp: false,
                    kicked: false, cast_for_free: false,
                    use_impending: false, cast_room_half: null },
          visits: 10, q: 0.0, prior: 0.2, leaf_value: 0.0 },
      ],
      samples: 1, inner_nodes: 100, elapsed_ms: 1,
    };
    globalThis.game.snapCache.set(0, snap);
    globalThis.renderSearchPanel(snap);
    const rows = Array.from(document.querySelectorAll('#search-rows tr'));
    const labels = rows.map(r => r.children[0].textContent);
    expect(labels).toEqual([
      'PassPriority',
      'PlayLand — Swamp',
      'CastSpell — Hylderblade',
    ]);
  });

  it('renders the chosen-row header label by name', () => {
    globalThis.game = {
      gameId: 'g1', currentIdx: 0, snapshotCount: 1,
      snapCache: new Map(), index: [], fullLog: [],
      playing: false, playTimer: null,
    };
    const snap = _baseSnap();
    snap.players[0].battlefield = [{ id: 44, name: 'Comet Crawler' }];
    snap.search = {
      chosen: { kind: 'AttackerDecision', attacker_id: 44, attack: true },
      candidates: [
        { action: { kind: 'AttackerDecision', attacker_id: 44, attack: true },
          visits: 5, q: 0, prior: 0.5, leaf_value: 0 },
      ],
      samples: 1, inner_nodes: 50, elapsed_ms: 2,
    };
    globalThis.game.snapCache.set(0, snap);
    globalThis.renderSearchPanel(snap);
    const chosen = document.getElementById('search-chosen').textContent;
    expect(chosen).toBe('Chose: Attack — Comet Crawler');
  });

  it('falls back to id-only labels when no game/snapshot is loaded', () => {
    // No globalThis.game — render the annotation directly. The
    // formatter should still produce a string (id-only fallback).
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PlayLand', card_id: 71 },
      candidates: [
        { action: { kind: 'PlayLand', card_id: 71 }, visits: 1,
          q: 0, prior: 0.5, leaf_value: 0 },
      ],
      samples: 1, inner_nodes: 1, elapsed_ms: 1,
    };
    globalThis.game = null;
    globalThis.renderSearchPanel(snap);
    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows[0].children[0].textContent).toBe('PlayLand(card_id=71)');
  });
});

// -------------------------------------------------------------------
// Recommend-fallback path: snapshot has no ``search`` annotation but
// the server has an engine state, so renderSearchPanel hits
// /api/recommend and renders the response into the same table shape.
// -------------------------------------------------------------------

// Install a fake game so the recommend fetch URL builds; install a
// fetch stub that returns ``payload`` and waits one microtask cycle
// so the test can flush promises with `await Promise.resolve()` then
// `await Promise.resolve()` (one for the fetch promise, one for the
// .then callback).
function _setupRecommend(payload) {
  globalThis.game = {
    gameId: 'g1', currentIdx: 5,
    snapshotCount: 10,
    snapCache: new Map(), index: [], fullLog: [],
    playing: false, playTimer: null,
  };
  const calls = [];
  globalThis.fetch = vi.fn().mockImplementation((url) => {
    calls.push(url);
    return Promise.resolve({
      ok: true, status: 200,
      json: async () => payload,
      text: async () => '',
    });
  });
  return { calls };
}

async function _flush() {
  // Several microtask ticks to chase: fetch promise → response → r.json()
  // → fetchJSON's await → renderSearchPanel's .then. The exact number
  // depends on how the test's mock chains the response, so we burn a
  // generous batch to make sure all queued microtasks have run.
  for (let i = 0; i < 10; i++) {
    await Promise.resolve();
  }
}

describe('renderSearchPanel recommend fallback', () => {
  it('renders pimc recommendations into the table when no search annotation', async () => {
    const { calls } = _setupRecommend({
      priority_player_id: 0,
      recommendations: [
        { label: 'Cast Goblin Runner', score: 0.55, action_type: 'CastSpell',
          source_id: 1, source_name: 'GR', scryfall_id: null, detail: '{R}' },
        { label: 'Pass priority', score: 0.30, action_type: 'PassPriority',
          source_id: null, source_name: null, scryfall_id: null, detail: null },
        { label: 'Play Mountain', score: 0.15, action_type: 'PlayLand',
          source_id: 2, source_name: 'Mtn', scryfall_id: null, detail: null },
      ],
      engine: 'pimc', engine_version: '0.1.0',
    });
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    await _flush();

    expect(calls.length).toBe(1);
    expect(calls[0]).toContain('/api/recommend/g1/5');

    const panel = document.getElementById('search-panel');
    expect(panel.classList.contains('hidden')).toBe(false);

    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows.length).toBe(3);

    // Column 0 = action label (verbatim from r.label).
    expect(rows[0].children[0].textContent).toBe('Cast Goblin Runner');
    // Column 1 = visits, em dash for recommend-fallback.
    expect(rows[0].children[1].textContent).toBe('—');
    // Column 2 = Q, the rec score formatted to 3 dp.
    expect(rows[0].children[2].textContent).toBe('0.550');
    // Columns 3 + 4 = prior + leaf, both em dash.
    expect(rows[0].children[3].textContent).toBe('—');
    expect(rows[0].children[4].textContent).toBe('—');

    // No row gets the chosen tint — there is no ground-truth chosen.
    expect(document.querySelectorAll('#search-rows tr.chosen').length).toBe(0);

    // Header reads the priority + engine line, not "Chose: ...".
    expect(document.getElementById('search-chosen').textContent)
      .toBe('Priority: Player 0');
    expect(document.getElementById('search-budget').textContent)
      .toContain('pimc v0.1.0');
  });

  it('hides the panel when the recommend response is engine="stub"', async () => {
    _setupRecommend({
      priority_player_id: 0,
      recommendations: [
        { label: 'Pass priority', score: 0.0, action_type: 'PassPriority',
          source_id: null, source_name: null, scryfall_id: null, detail: null },
      ],
      engine: 'stub', engine_version: '0.0.1',
    });
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    await _flush();

    const panel = document.getElementById('search-panel');
    expect(panel.classList.contains('hidden')).toBe(true);
  });

  it('hides the panel when the recommend response has zero recommendations', async () => {
    _setupRecommend({
      priority_player_id: 0,
      recommendations: [],
      engine: 'pimc', engine_version: '0.1.0',
    });
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    await _flush();

    expect(document.getElementById('search-panel')
      .classList.contains('hidden')).toBe(true);
  });

  it('search annotation wins over recommend when both are available', async () => {
    const { calls } = _setupRecommend({
      priority_player_id: 0,
      recommendations: [
        { label: 'X', score: 0.99, action_type: 'X', source_id: null,
          source_name: null, scryfall_id: null, detail: null },
      ],
      engine: 'pimc', engine_version: '0.1.0',
    });
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        { action: { kind: 'PassPriority' }, visits: 5,
          q: 0.1, prior: 0.5, leaf_value: 0.2 },
      ],
      samples: 1, inner_nodes: 50, elapsed_ms: 3,
    };
    globalThis.renderSearchPanel(snap);
    await _flush();

    // Recommend was not fetched — annotation took the synchronous path.
    expect(calls.length).toBe(0);

    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows.length).toBe(1);
    // The annotation's visits column has a real number, not em dash.
    expect(rows[0].children[1].textContent).toBe('5');
    expect(rows[0].children[2].textContent).toBe('0.100');
    // Chosen row tint applied.
    expect(document.querySelectorAll('#search-rows tr.chosen').length).toBe(1);
  });

  it('hides the panel when neither annotation nor game is set', () => {
    // No setup — globalThis.game stays null and there's no search.
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    // Synchronous: pre-emptive hide, then early-return on null game.
    expect(document.getElementById('search-panel')
      .classList.contains('hidden')).toBe(true);
  });

  // -----------------------------------------------------------------
  // Diagnostic deep-tree capture: candidate rows with non-empty
  // ``children`` arrays grow a chevron; click toggles inline child
  // rows below the parent. Recursive expand. Production root-only
  // captures (``children`` null/undefined/empty) keep today's
  // rendering exactly — no chevron, no extra rows.
  // -----------------------------------------------------------------
});

describe('renderSearchPanel deep-capture expand/collapse', () => {
  function _deepAnnotated() {
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        {
          action: { kind: 'PassPriority' }, visits: 50,
          q: 0.4, prior: 0.6, leaf_value: 0.3,
          children: [
            {
              action: { kind: 'PlayLand', card_id: 5 }, visits: 25,
              q: 0.2, prior: 0.3, leaf_value: 0.1,
              children: [
                {
                  action: { kind: 'CastSpell', card_id: 9 }, visits: 12,
                  q: 0.05, prior: 0.2, leaf_value: 0.0,
                  children: [],
                },
                {
                  action: { kind: 'PassPriority' }, visits: 8,
                  q: -0.1, prior: 0.4, leaf_value: -0.05,
                  children: [],
                },
              ],
            },
            {
              action: { kind: 'PassPriority' }, visits: 5,
              q: -0.05, prior: 0.1, leaf_value: 0.0,
              children: [],
            },
          ],
        },
        {
          action: { kind: 'PlayLand', card_id: 7 }, visits: 10,
          q: -0.1, prior: 0.2, leaf_value: -0.05,
          children: null,
        },
      ],
      samples: 4, inner_nodes: 200, elapsed_ms: 30,
    };
    return snap;
  }

  it('shows a chevron only on rows whose children list is non-empty', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows.length).toBe(2);
    // First root row (PassPriority, 50 visits): children populated → expandable chevron.
    const chev0 = rows[0].querySelector('.search-chevron.expandable');
    expect(chev0).not.toBeNull();
    expect(chev0.textContent).toBe('▸');
    // Second row (PlayLand, children=null): chevron span is non-expandable / empty.
    const chev1 = rows[1].querySelector('.search-chevron.expandable');
    expect(chev1).toBeNull();
  });

  it('hides the chevron on rows whose children list is empty (no expand)', () => {
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        {
          action: { kind: 'PassPriority' }, visits: 5,
          q: 0.0, prior: 0.5, leaf_value: 0.0,
          children: [],   // deep-mode capture but no expanded grandchildren.
        },
      ],
      samples: 1, inner_nodes: 10, elapsed_ms: 1,
    };
    globalThis.renderSearchPanel(snap);
    const chev = document.querySelector('#search-rows .search-chevron.expandable');
    expect(chev).toBeNull();
  });

  it('rendering a root-only capture (children=null) is identical to today', () => {
    // Same as the populated-state test, no children field at all.
    const snap = _baseSnap();
    snap.search = {
      chosen: { kind: 'PassPriority' },
      candidates: [
        {
          action: { kind: 'PassPriority' }, visits: 32,
          q: 0.4, prior: 0.7, leaf_value: 0.3,
        },
        {
          action: { kind: 'PlayLand', card_id: 5 }, visits: 7,
          q: -0.1, prior: 0.2, leaf_value: -0.05,
        },
      ],
      samples: 1, inner_nodes: 50, elapsed_ms: 5,
    };
    globalThis.renderSearchPanel(snap);
    // No expandable chevrons anywhere.
    const expandable = document.querySelectorAll(
      '#search-rows .search-chevron.expandable',
    );
    expect(expandable.length).toBe(0);
    // No child rows (search-row-child class) appended.
    const children = document.querySelectorAll(
      '#search-rows .search-row-child',
    );
    expect(children.length).toBe(0);
  });

  it('clicking the chevron expands child rows directly below the parent', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const rows = document.querySelectorAll('#search-rows tr');
    const chev = rows[0].querySelector('.search-chevron.expandable');
    expect(chev.textContent).toBe('▸');
    // Click to expand.
    chev.click();
    // After expand: chevron flips, two child rows appear right below.
    expect(chev.textContent).toBe('▾');
    const childRows = document.querySelectorAll('#search-rows .search-row-child');
    expect(childRows.length).toBe(2);
    // Order is by visits desc — PlayLand(25 visits) first, PassPriority(5) second.
    expect(childRows[0].children[0].textContent).toContain('PlayLand');
    expect(childRows[0].children[1].textContent).toBe('25');
    expect(childRows[1].children[0].textContent).toContain('PassPriority');
    expect(childRows[1].children[1].textContent).toBe('5');
    // Child rows' chosen-tint check: deeper rows should NOT carry the
    // chosen tint even if their action equals the root chosen action.
    expect(childRows[0].classList.contains('chosen')).toBe(false);
    expect(childRows[1].classList.contains('chosen')).toBe(false);
  });

  it('clicking the chevron a second time collapses the child rows', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const chev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    chev.click();  // expand
    expect(
      document.querySelectorAll('#search-rows .search-row-child').length,
    ).toBe(2);
    chev.click();  // collapse
    expect(chev.textContent).toBe('▸');
    expect(
      document.querySelectorAll('#search-rows .search-row-child').length,
    ).toBe(0);
  });

  it('recursively expands depth-2 children when their chevrons are clicked', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    // Expand depth 1 first.
    const rootChev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    rootChev.click();
    // Depth-1 row 0 (PlayLand 25) has its own children — expandable.
    const depth1Rows = document.querySelectorAll('#search-rows .search-row-child');
    expect(depth1Rows.length).toBe(2);
    const d1Chev = depth1Rows[0].querySelector('.search-chevron.expandable');
    expect(d1Chev).not.toBeNull();
    d1Chev.click();
    // After depth-2 expand: 4 child rows total (the original 2 + 2 grandchildren).
    const allChildren = document.querySelectorAll('#search-rows .search-row-child');
    expect(allChildren.length).toBe(4);
    // Grandchild rows appear right after their parent (depth-1 row 0).
    // Order is depth-1[0], grand[0], grand[1], depth-1[1].
    expect(allChildren[1].children[0].textContent).toContain('CastSpell');
    expect(allChildren[1].children[1].textContent).toBe('12');
    expect(allChildren[2].children[0].textContent).toContain('PassPriority');
    expect(allChildren[2].children[1].textContent).toBe('8');
  });

  it('collapsing a parent also removes its expanded descendant rows', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const rootChev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    rootChev.click();  // expand depth 1
    const depth1Rows = document.querySelectorAll('#search-rows .search-row-child');
    const d1Chev = depth1Rows[0].querySelector('.search-chevron.expandable');
    d1Chev.click();    // expand depth 2 under first depth-1
    expect(
      document.querySelectorAll('#search-rows .search-row-child').length,
    ).toBe(4);
    // Now collapse the root — every descendant should be cleared.
    rootChev.click();
    expect(
      document.querySelectorAll('#search-rows .search-row-child').length,
    ).toBe(0);
  });

  it('child action labels go through formatAction (same name resolution)', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const chev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    chev.click();
    const childRows = document.querySelectorAll('#search-rows .search-row-child');
    // formatAction emits "PlayLand(card_id=5)" for the first child.
    // textContent of the action cell includes the chevron span text +
    // the action label; we just assert the action label substring.
    expect(childRows[0].children[0].textContent).toContain(
      'PlayLand(card_id=5)',
    );
  });

  it('child rows use indented padding scaled by depth', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const rootChev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    rootChev.click();
    const depth1Row = document.querySelector('#search-rows .search-row-child');
    const d1Pad = parseInt(depth1Row.children[0].style.paddingLeft, 10);
    expect(d1Pad).toBeGreaterThan(0);
    const d1Chev = depth1Row.querySelector('.search-chevron.expandable');
    d1Chev.click();
    const depth2Rows = document.querySelectorAll('#search-rows .search-row-child');
    // Depth-2 row sits right after depth-1[0]; index 1 in the
    // overall child-row collection.
    const d2Pad = parseInt(depth2Rows[1].children[0].style.paddingLeft, 10);
    expect(d2Pad).toBeGreaterThan(d1Pad);
  });

  it('chosen-row tint stays on root only — no descendant gets tinted', () => {
    globalThis.renderSearchPanel(_deepAnnotated());
    const rootChev = document.querySelector(
      '#search-rows .search-chevron.expandable',
    );
    rootChev.click();
    // Root chosen row is the first row.
    const rows = document.querySelectorAll('#search-rows tr');
    const chosenRows = document.querySelectorAll('#search-rows tr.chosen');
    expect(chosenRows.length).toBe(1);
    expect(chosenRows[0]).toBe(rows[0]);
  });
});

describe('renderSearchPanel stale-fetch path', () => {
  it('discards stale recommend responses when the user scrubs forward', async () => {
    // Two snapshots back-to-back: the first fetch should be ignored
    // because the second renderSearchPanel call bumps the token.
    let resolveFirst;
    const firstPromise = new Promise(r => { resolveFirst = r; });
    let resolveSecond;
    const secondPromise = new Promise(r => { resolveSecond = r; });
    let callCount = 0;
    globalThis.game = {
      gameId: 'g1', currentIdx: 5, snapshotCount: 10,
      snapCache: new Map(), index: [], fullLog: [],
      playing: false, playTimer: null,
    };
    globalThis.fetch = vi.fn().mockImplementation(() => {
      callCount += 1;
      const r = (callCount === 1) ? firstPromise : secondPromise;
      return r.then(payload => ({
        ok: true, status: 200,
        json: async () => payload,
        text: async () => '',
      }));
    });
    const snap = _baseSnap();
    globalThis.renderSearchPanel(snap);
    globalThis.renderSearchPanel(snap);

    // Resolve the FIRST fetch with a payload that would normally render.
    resolveFirst({
      priority_player_id: 0,
      recommendations: [{
        label: 'STALE', score: 0.99, action_type: 'X',
        source_id: null, source_name: null, scryfall_id: null, detail: null,
      }],
      engine: 'pimc', engine_version: '0.1.0',
    });
    await _flush();
    // First response is stale — panel should still be hidden.
    expect(document.getElementById('search-panel')
      .classList.contains('hidden')).toBe(true);
    expect(document.querySelectorAll('#search-rows tr').length).toBe(0);

    // Now resolve the second; it should win.
    resolveSecond({
      priority_player_id: 0,
      recommendations: [{
        label: 'FRESH', score: 0.5, action_type: 'X',
        source_id: null, source_name: null, scryfall_id: null, detail: null,
      }],
      engine: 'pimc', engine_version: '0.1.0',
    });
    await _flush();
    expect(document.getElementById('search-panel')
      .classList.contains('hidden')).toBe(false);
    const rows = document.querySelectorAll('#search-rows tr');
    expect(rows.length).toBe(1);
    expect(rows[0].children[0].textContent).toBe('FRESH');
  });
});
