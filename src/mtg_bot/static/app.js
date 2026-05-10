// MTG Bot viewer — vanilla JS front-end for the Flask JSON API.
//
// Contract: /api/play returns { game_id, snapshot_count, ... }, then we fetch
// /api/snapshot/<game_id>/<idx> for each state we want to render. For now we
// only render the final snapshot; the scrubber is a follow-up commit.

const els = {
  form: document.getElementById('play-form'),
  status: document.getElementById('status'),
  board: document.getElementById('board'),
  logPanel: document.getElementById('log-panel'),
  logBody: document.getElementById('log-body'),
  rowP0: document.getElementById('row-p0'),
  rowP1: document.getElementById('row-p1'),
  rowStack: document.getElementById('row-stack'),
  preview: document.getElementById('card-preview'),
  scrubber: document.getElementById('scrubber'),
  slider: document.getElementById('scrub-slider'),
  scrubLabel: document.getElementById('scrub-label'),
  scrubCounter: document.getElementById('scrub-counter'),
  btnFirst: document.getElementById('btn-first'),
  btnPrevTurn: document.getElementById('btn-prev-turn'),
  btnPrevStep: document.getElementById('btn-prev-step'),
  btnPrev: document.getElementById('btn-prev'),
  btnPlay: document.getElementById('btn-play'),
  btnNext: document.getElementById('btn-next'),
  btnNextStep: document.getElementById('btn-next-step'),
  btnNextTurn: document.getElementById('btn-next-turn'),
  btnLast: document.getElementById('btn-last'),
  combatOverlay: document.getElementById('combat-overlay'),
  zoneModal: document.getElementById('zone-modal'),
  zoneModalTitle: document.getElementById('zone-modal-title'),
  zoneModalCards: document.getElementById('zone-modal-cards'),
  zoneModalClose: document.getElementById('zone-modal-close'),
  phaseTracker: document.getElementById('phase-tracker'),
  phaseList: document.getElementById('phase-list'),
  searchPanel: document.getElementById('search-panel'),
  searchToggle: document.getElementById('search-toggle'),
  searchBody: document.getElementById('search-body'),
  searchChosen: document.getElementById('search-chosen'),
  searchBudget: document.getElementById('search-budget'),
  searchRows: document.getElementById('search-rows'),
};

// `var` (not `let`) so JS unit tests can override. Tracks whether the
// decision-detail panel's body is collapsed (header + toggle still
// visible) vs expanded. Independent of whether the panel itself is
// hidden — that's driven entirely by snapshot.search presence.
var searchCollapsed = false;

// Engine step order + display labels. Mirrors mtg_bot.turn.STEP_ORDER.
const PHASE_STEPS = [
  { key: 'untap',               label: 'Untap',     group: '' },
  { key: 'upkeep',              label: 'Upkeep',    group: '' },
  { key: 'draw',                label: 'Draw',      group: '' },
  { key: 'precombat_main',      label: 'Main 1',    group: 'main' },
  { key: 'beginning_of_combat', label: 'B. Combat', group: 'combat' },
  { key: 'declare_attackers',   label: 'D. Atk',    group: 'combat' },
  { key: 'declare_blockers',    label: 'D. Blk',    group: 'combat' },
  { key: 'combat_damage',       label: 'Damage',    group: 'combat' },
  { key: 'end_of_combat',       label: 'E. Combat', group: 'combat' },
  { key: 'postcombat_main',     label: 'Main 2',    group: 'main' },
  { key: 'end',                 label: 'End',       group: '' },
  { key: 'cleanup',             label: 'Cleanup',   group: '' },
];

const SVG_NS = 'http://www.w3.org/2000/svg';

// Active game state. One of these is live at a time; running a new game
// replaces it. snapCache memoizes fetched snapshots so scrubbing back and
// forth doesn't re-hit the API. `var` (not `let`) so JS unit tests can
// seed window.game = {...} before invoking nav helpers.
var game = null;  // { gameId, snapshotCount, summary, snapCache, currentIdx, playing, playTimer }

const PLAY_INTERVAL_MS = 700;

els.form.addEventListener('submit', (ev) => { ev.preventDefault(); runGame(); });
els.btnFirst   .addEventListener('click', () => goTo(0));
els.btnPrevTurn.addEventListener('click', () => goTo(findPrevTurnIdx()));
els.btnPrevStep.addEventListener('click', () => goTo(findPrevStepIdx()));
els.btnPrev    .addEventListener('click', () => goTo(game ? game.currentIdx - 1 : 0));
els.btnNext    .addEventListener('click', () => goTo(game ? game.currentIdx + 1 : 0));
els.btnNextStep.addEventListener('click', () => goTo(findNextStepIdx()));
els.btnNextTurn.addEventListener('click', () => goTo(findNextTurnIdx()));
els.btnLast    .addEventListener('click', () => goTo(game ? game.snapshotCount - 1 : 0));
els.btnPlay    .addEventListener('click', togglePlay);
els.slider     .addEventListener('input', (ev) => goTo(parseInt(ev.target.value, 10)));
if (els.searchToggle) {
  els.searchToggle.addEventListener('click', toggleSearchPanel);
}

// Keyboard shortcuts. Guard against firing while typing in the form.
window.addEventListener('keydown', (ev) => {
  // Esc always closes the zone modal first; do this before the
  // typing-in-form guard so closing works from anywhere.
  const zoneModal = document.getElementById('zone-modal');
  if (ev.key === 'Escape' && zoneModal && !zoneModal.classList.contains('hidden')) {
    ev.preventDefault();
    closeZoneModal();
    return;
  }
  if (!game) return;
  const tag = (ev.target && ev.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (ev.key === 'ArrowLeft') {
    ev.preventDefault();
    if (ev.shiftKey) goTo(findPrevTurnIdx());
    else if (ev.altKey) goTo(findPrevStepIdx());
    else goTo(game.currentIdx - 1);
  } else if (ev.key === 'ArrowRight') {
    ev.preventDefault();
    if (ev.shiftKey) goTo(findNextTurnIdx());
    else if (ev.altKey) goTo(findNextStepIdx());
    else goTo(game.currentIdx + 1);
  } else if (ev.key === 'Home') {
    ev.preventDefault();
    goTo(0);
  } else if (ev.key === 'End') {
    ev.preventDefault();
    goTo(game.snapshotCount - 1);
  } else if (ev.key === ' ') {
    ev.preventDefault();
    togglePlay();
  }
});

async function runGame() {
  stopPlay();  // if an earlier game was playing
  const params = new URLSearchParams();
  for (const [k, v] of new FormData(els.form)) params.set(k, v);
  setStatus('Running…', 'muted');
  setButton(false);
  try {
    const summary = await fetchJSON('/api/play?' + params.toString());
    await activateGame(summary);
  } catch (e) {
    setStatus('Error: ' + e.message, 'loss');
  } finally {
    setButton(true);
  }
}

// Load a saved replay from the server's replays_root and switch the
// scrubber to it. ``filename`` is just the basename; the server side
// validates path traversal. Used by the ``?replay=`` URL handler so
// links like ``http://<host>:5002/?replay=foo.json.gz`` open the
// viewer pre-loaded on a captured game.
async function loadReplay(filename) {
  stopPlay();
  setStatus('Loading replay ' + filename + '…', 'muted');
  setButton(false);
  try {
    const summary = await fetchJSON(
      '/api/load_replay/' + encodeURIComponent(filename),
    );
    // Replays open at snapshot 0 so the reviewer can step forward
    // through the game; live games still land on the final state.
    await activateGame(summary, 0);
    setStatus('Loaded replay ' + filename, 'muted');
  } catch (e) {
    setStatus('Replay load failed: ' + e.message, 'loss');
  } finally {
    setButton(true);
  }
}

// Shared "set up the scrubber from an /api/play or /api/load_replay
// response" path. Both endpoints return the same {game_id,
// snapshot_count, ...} shape, so the activation work is identical.
// ``startIdx`` selects the initial snapshot — defaults to the last
// (final state) for live games; replays pass 0 to start at turn 1.
async function activateGame(summary, startIdx) {
  // Fetch the compact (turn, step) index so step/turn navigation is O(1)
  // per click instead of N fetches. Fetch the full log too so the log
  // panel can show every event (past and future) and the user can click
  // ahead to skip forward.
  const [index, fullLog] = await Promise.all([
    fetchJSON(`/api/snapshots_index/${summary.game_id}`),
    fetchJSON(`/api/log/${summary.game_id}`),
  ]);
  game = {
    gameId: summary.game_id,
    snapshotCount: summary.snapshot_count,
    summary,
    index,                   // [{turn_number, step, log_size}, ...]
    fullLog,                 // [str, str, ...] — every log line in the game
    snapCache: new Map(),
    currentIdx: 0,
    playing: false,
    playTimer: null,
  };
  els.slider.min = 0;
  els.slider.max = Math.max(0, summary.snapshot_count - 1);
  els.scrubber.classList.remove('hidden');
  els.phaseTracker.classList.remove('hidden');
  const target = (startIdx == null) ? summary.snapshot_count - 1 : startIdx;
  await goTo(target);
}

// On page load: if the URL carries ``?replay=<filename>``, jump
// straight into that replay instead of waiting for the user to
// submit the play form. Tolerates fetch errors silently — the
// status bar shows the failure but the form stays usable.
(async () => {
  const params = new URLSearchParams(window.location.search);
  const replayName = params.get('replay');
  if (replayName) {
    await loadReplay(replayName);
  }
})();

async function goTo(idx) {
  if (!game) return;
  idx = Math.max(0, Math.min(game.snapshotCount - 1, idx));
  game.currentIdx = idx;
  if (!game.snapCache.has(idx)) {
    try {
      const snap = await fetchJSON(`/api/snapshot/${game.gameId}/${idx}`);
      game.snapCache.set(idx, snap);
    } catch (e) {
      setStatus('Snapshot fetch failed: ' + e.message, 'loss');
      stopPlay();
      return;
    }
  }
  const snap = game.snapCache.get(idx);
  renderSnapshot(game.summary, snap);
  els.slider.value = idx;
  els.scrubCounter.textContent = `${idx + 1} / ${game.snapshotCount}`;
  els.scrubLabel.textContent = scrubLabelFor(snap);
  updateScrubberButtons();
  if (game.playing && idx >= game.snapshotCount - 1) stopPlay();
}

function scrubLabelFor(snap) {
  const step = snap.step || 'setup';
  return `Turn ${snap.turn_number} · ${step}`;
}

function togglePlay() {
  if (!game) return;
  if (game.playing) stopPlay();
  else startPlay();
}

function startPlay() {
  if (!game || game.playing) return;
  // If already at the end, rewind so play means something.
  if (game.currentIdx >= game.snapshotCount - 1) {
    goTo(0);
  }
  game.playing = true;
  els.btnPlay.classList.add('playing');
  els.btnPlay.innerHTML = '&#9616;&#9616;';  // pause glyph
  game.playTimer = setInterval(() => goTo(game.currentIdx + 1), PLAY_INTERVAL_MS);
}

function stopPlay() {
  if (!game) return;
  game.playing = false;
  els.btnPlay.classList.remove('playing');
  els.btnPlay.innerHTML = '&#9654;';  // play glyph
  if (game.playTimer) {
    clearInterval(game.playTimer);
    game.playTimer = null;
  }
}

function updateScrubberButtons() {
  const atStart = game.currentIdx <= 0;
  const atEnd = game.currentIdx >= game.snapshotCount - 1;
  els.btnFirst.disabled = atStart;
  els.btnPrev.disabled = atStart;
  els.btnPrevStep.disabled = atStart;
  els.btnPrevTurn.disabled = atStart;
  els.btnNext.disabled = atEnd;
  els.btnNextStep.disabled = atEnd;
  els.btnNextTurn.disabled = atEnd;
  els.btnLast.disabled = atEnd;
}

// ------------------------------------------------------------------------
// Step / turn navigation. Uses the compact game.index so each lookup is
// a synchronous linear scan over tiny (turn_number, step) objects rather
// than N snapshot fetches.
// ------------------------------------------------------------------------

function findNextStepIdx() {
  if (!game) return 0;
  const cur = game.index[game.currentIdx];
  for (let i = game.currentIdx + 1; i < game.snapshotCount; i++) {
    const s = game.index[i];
    if (s.turn_number !== cur.turn_number || s.step !== cur.step) return i;
  }
  return game.snapshotCount - 1;
}

function findPrevStepIdx() {
  if (!game) return 0;
  const cur = game.index[game.currentIdx];
  // First walk back past all snapshots that share the current step.
  let i = game.currentIdx - 1;
  while (i >= 0) {
    const s = game.index[i];
    if (s.turn_number !== cur.turn_number || s.step !== cur.step) break;
    i--;
  }
  if (i < 0) return 0;
  // Now i is the last snapshot of the previous step. Walk back to its first.
  const target = game.index[i];
  while (i > 0) {
    const prev = game.index[i - 1];
    if (prev.turn_number !== target.turn_number || prev.step !== target.step) break;
    i--;
  }
  return i;
}

function findNextTurnIdx() {
  if (!game) return 0;
  const cur = game.index[game.currentIdx];
  for (let i = game.currentIdx + 1; i < game.snapshotCount; i++) {
    if (game.index[i].turn_number > cur.turn_number) return i;
  }
  return game.snapshotCount - 1;
}

function findPrevTurnIdx() {
  if (!game) return 0;
  const cur = game.index[game.currentIdx];
  // Walk back past the current turn.
  let i = game.currentIdx - 1;
  while (i >= 0 && game.index[i].turn_number >= cur.turn_number) i--;
  if (i < 0) return 0;
  // Then back to the first snapshot of that previous turn.
  const target = game.index[i].turn_number;
  while (i > 0 && game.index[i - 1].turn_number === target) i--;
  return i;
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = 'status ' + (cls || '');
}

function setButton(enabled) {
  els.form.querySelector('button').disabled = !enabled;
}

function renderSnapshot(summary, snap) {
  if (snap.is_draw) setStatus(`Draw — ${snap.turn_number} turns`, 'draw');
  else if (snap.is_over && snap.winner_id !== null) setStatus(`Player ${snap.winner_id} wins — turn ${snap.turn_number}`, 'win');
  else setStatus(`Turn ${snap.turn_number} · ${snap.step || '-'}`, 'muted');

  els.board.classList.remove('hidden');
  els.logPanel.classList.remove('hidden');

  const p0 = snap.players.find(p => p.id === 0);
  const p1 = snap.players.find(p => p.id === 1);
  renderPlayer(els.rowP1, p1, snap, /* isTop */ true);
  renderPlayer(els.rowP0, p0, snap, /* isTop */ false);
  renderStack(els.rowStack, snap);
  renderLog(els.logBody, snap);
  renderPhaseTracker(snap);
  drawCombatArrows(snap);
  renderSearchPanel(snap);
}

// Toggle the decision-detail panel's body open/closed. Header + toggle
// stay visible either way, but rows + meta block hide when collapsed
// so the user can keep the panel pinned without burning vertical space
// during long scrubs.
function toggleSearchPanel() {
  searchCollapsed = !searchCollapsed;
  if (els.searchBody) {
    els.searchBody.classList.toggle('hidden', searchCollapsed);
  }
  if (els.searchToggle) {
    els.searchToggle.innerHTML = searchCollapsed ? '+' : '&minus;';
  }
}

// Monotonically-increasing token used to discard stale recommend
// responses when the user scrubs through snapshots faster than the
// recommend endpoint can answer. Each renderSearchPanel call bumps
// this; the recommend fetch's then() compares against the latest
// value and bails if a newer call has started.
//
// `var` (not `let`) so JS unit tests can reset it before each test.
var searchPanelToken = 0;

// Render the unified Decision panel into the side panel. There are
// three priority-ordered modes:
//
//   1. snapshot has a real ``search`` annotation (v3 replays produced
//      with per-decision MCTS/PIMC stats) — render the full table
//      with Visits / Q / Prior / Leaf V populated. This is the most
//      informative view; it always wins when available.
//   2. no annotation, but the server has an engine state for this
//      snapshot (live games, /api/play flows) — fetch /api/recommend
//      and render its recommendations into the same table. Visits /
//      Prior / Leaf V columns degrade to em dash; Q gets the score
//      formatted to 3 dp. The chosen-row highlight does not apply
//      (no ground-truth chosen action).
//   3. neither — hide the panel entirely. The stub recommender
//      (engine="stub") returns score=0.0 for everything and is not
//      worth rendering; that's the all-zeros bug we're fixing.
function renderSearchPanel(snap) {
  if (!els.searchPanel) return;
  // Bump the request token so any in-flight recommend fetches from a
  // previous snapshot will discard their results when they resolve.
  const token = ++searchPanelToken;

  const search = snap && snap.search;
  if (search && Array.isArray(search.candidates) && search.candidates.length > 0) {
    _renderSearchAnnotation(search);
    return;
  }

  // Mode 2 / 3: ask the engine. Hide the panel pre-emptively while
  // we wait so a stale render doesn't briefly show the previous
  // snapshot's table; if the response comes back with usable data
  // we'll un-hide it.
  els.searchPanel.classList.add('hidden');
  if (!game) return;
  const url = `/api/recommend/${game.gameId}/${game.currentIdx}?n=10`;
  fetchJSON(url).then(res => {
    // Ignore stale responses (a newer renderSearchPanel call ran).
    if (token !== searchPanelToken) return;
    // Stub recommender means we have no real engine evaluation —
    // showing a panel with all-zero scores is precisely the bug
    // we're fixing, so hide rather than render.
    if (res.engine === 'stub') return;
    if (!res.recommendations || res.recommendations.length === 0) return;
    _renderRecommendFallback(res);
  }).catch(() => {
    // Recommend can fail on terminal snapshots, missing engine
    // state, etc. Stay hidden — same UX as "no data."
  });
}

// Helper: render the v3 "search" annotation into the panel.
function _renderSearchAnnotation(search) {
  els.searchPanel.classList.remove('hidden');
  // Build a fresh id->name map from the current snapshot so action
  // labels can render card / permanent names instead of bare numeric
  // ids. Falls back gracefully (id-only formatting) when the snapshot
  // isn't available (e.g. tests that don't install ``game``).
  const idToName = _buildCardIdMap(_currentSnapshot());
  // Chosen action label — falls back to the most-visited candidate's
  // action when ``chosen`` is missing (defensive; the writer always
  // includes it but readers shouldn't crash on a future writer that
  // omits it).
  const chosenJson = search.chosen
    || (search.candidates.length > 0 ? search.candidates[0].action : null);
  els.searchChosen.textContent = chosenJson
    ? `Chose: ${formatAction(chosenJson, idToName)}` : '';
  // Budget / timing readout. ``samples`` × ``inner_nodes`` is the
  // PIMC-of-MCTS shape; raw MCTS reports ``samples=1`` so the line
  // reads "1 × N nodes" which is still informative. ``elapsed_ms``
  // is optional — printed when present.
  const totalVisits = search.candidates.reduce(
    (acc, c) => acc + (c.visits | 0), 0,
  );
  const budgetParts = [];
  if (typeof search.samples === 'number') {
    if (typeof search.inner_nodes === 'number') {
      budgetParts.push(`${search.samples} × ${search.inner_nodes} nodes`);
    } else {
      budgetParts.push(`${search.samples} samples`);
    }
  }
  budgetParts.push(`${totalVisits} total visits`);
  if (typeof search.elapsed_ms === 'number') {
    budgetParts.push(`${search.elapsed_ms} ms`);
  }
  els.searchBudget.textContent = budgetParts.join(' · ');
  // Rows: candidates come pre-sorted by visits desc from the writer,
  // but defensive re-sort in case a future producer changes order
  // (e.g. rewrites for stable diffing across iterations).
  const rows = [...search.candidates].sort(
    (a, b) => (b.visits | 0) - (a.visits | 0),
  );
  els.searchRows.replaceChildren();
  for (const c of rows) {
    const tr = _buildSearchRow(c, chosenJson, /*depth=*/0, idToName);
    els.searchRows.appendChild(tr);
  }
}

// Builds a single search-table row plus (lazily) any child rows when
// the user clicks the chevron. Returns the parent <tr>; child rows
// are appended to the same tbody just below the parent on toggle.
//
// Diagnostic deep capture: the v3 search annotation now optionally
// carries a recursive ``children`` array per candidate. Production
// captures leave it null/empty (no chevron, identical to today's
// rendering); replays from ``capture_annotated_replays.py
// --capture-tree-depth N`` carry up to N levels of nested children.
// Click the chevron to reveal one level; click on a deeper row's
// chevron to reveal another. Recursive expand. The chosen-row tint
// stays on the root chosen action only — there's no notion of
// "chosen" deeper in the tree. ``depth`` controls left padding on
// the action cell so each level visually nests.
function _buildSearchRow(c, chosenJson, depth, idToName) {
  const tr = document.createElement('tr');
  tr.className = depth === 0 ? 'search-row' : 'search-row search-row-child';
  if (depth === 0) {
    const isChosen = chosenJson && actionsEqual(c.action, chosenJson);
    if (isChosen) tr.classList.add('chosen');
  }

  const tdAction = document.createElement('td');
  tdAction.className = 'action';
  // Indent deeper rows. Each depth step adds ~12px of padding so the
  // tree shape is visually clear without bloating the column.
  if (depth > 0) {
    tdAction.style.paddingLeft = `${12 * depth + 4}px`;
  }
  // Chevron span (visible only when this row has expandable children).
  const chev = document.createElement('span');
  chev.className = 'search-chevron';
  const hasChildren = Array.isArray(c.children) && c.children.length > 0;
  if (hasChildren) {
    chev.textContent = '▸';  // ▸ collapsed by default
    chev.classList.add('expandable');
  } else {
    // Hidden but reserves no horizontal width when not expandable so
    // root-only captures look identical to pre-feature rendering.
    chev.textContent = '';
  }
  tdAction.appendChild(chev);
  tdAction.appendChild(document.createTextNode(formatAction(c.action, idToName)));
  tr.appendChild(tdAction);

  const tdVisits = document.createElement('td');
  tdVisits.className = 'num';
  tdVisits.textContent = String(c.visits | 0);
  tr.appendChild(tdVisits);

  const tdQ = document.createElement('td');
  tdQ.className = 'num';
  tdQ.textContent = c.q == null ? '—' : Number(c.q).toFixed(3);
  tr.appendChild(tdQ);

  const tdPrior = document.createElement('td');
  tdPrior.className = 'num';
  tdPrior.textContent = c.prior == null ? '—' : Number(c.prior).toFixed(3);
  tr.appendChild(tdPrior);

  const tdLeaf = document.createElement('td');
  tdLeaf.className = 'num';
  tdLeaf.textContent = c.leaf_value == null
    ? '—' : Number(c.leaf_value).toFixed(3);
  tr.appendChild(tdLeaf);

  if (hasChildren) {
    let expanded = false;
    let childRows = [];
    const toggle = () => {
      if (!expanded) {
        // Sort children by visits desc (defensive — writer already
        // does this) and append immediately after the parent row.
        const sorted = [...c.children].sort(
          (a, b) => (b.visits | 0) - (a.visits | 0),
        );
        let anchor = tr;
        for (const child of sorted) {
          const childTr = _buildSearchRow(child, chosenJson, depth + 1, idToName);
          anchor.after(childTr);
          anchor = childTr;
          childRows.push(childTr);
        }
        chev.textContent = '▾';  // ▾ expanded
        expanded = true;
      } else {
        // Collapse: remove our direct child rows. Each removed child
        // recursively cleans up its own descendants because expanding
        // a child appends to the tbody (not nested inside a parent
        // <tr>), so to fully collapse we'd track descendant rows too.
        // Simpler approach: collapse calls remove() on each direct
        // child row, and any descendants below were appended right
        // after — they're orphans now, also need removing. We track
        // this by walking siblings until we hit a row at the same
        // or shallower depth.
        for (const r of childRows) {
          // Remove descendant rows that were inserted immediately
          // after this child while it was expanded.
          let next = r.nextSibling;
          while (
            next &&
            next.classList &&
            next.classList.contains('search-row-child')
          ) {
            // Only remove rows whose depth is greater than this
            // child row's depth — siblings at the same depth belong
            // to a different parent's expansion.
            const nextDepth = parseInt(
              next.dataset.searchRowDepth || '0', 10,
            );
            const myDepth = parseInt(
              r.dataset.searchRowDepth || '0', 10,
            );
            if (nextDepth > myDepth) {
              const toRemove = next;
              next = next.nextSibling;
              toRemove.remove();
            } else {
              break;
            }
          }
          r.remove();
        }
        childRows = [];
        chev.textContent = '▸';  // ▸ collapsed
        expanded = false;
      }
    };
    chev.addEventListener('click', (ev) => {
      ev.stopPropagation();
      toggle();
    });
  }

  // Track depth on the row's dataset so collapse can clean up
  // descendant rows that were appended as flat siblings.
  tr.dataset.searchRowDepth = String(depth);
  return tr;
}

// Helper: render a /api/recommend response (engine=pimc) into the
// same panel + table shape as the search annotation, degrading
// visits/prior/leaf to em dash and putting the rec score in the Q
// column. No chosen-row tint — this isn't a captured decision.
function _renderRecommendFallback(res) {
  els.searchPanel.classList.remove('hidden');
  els.searchChosen.textContent =
    `Priority: Player ${res.priority_player_id}`;
  els.searchBudget.textContent =
    `${res.engine} v${res.engine_version} · ${res.recommendations.length} options`;
  // Recs come pre-sorted by score desc from recommend_from_state;
  // re-sort defensively for the same reason as the annotation path.
  const rows = [...res.recommendations].sort(
    (a, b) => (b.score || 0) - (a.score || 0),
  );
  els.searchRows.replaceChildren();
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.className = 'search-row';

    const tdAction = document.createElement('td');
    tdAction.className = 'action';
    tdAction.textContent = r.label;
    tr.appendChild(tdAction);

    const tdVisits = document.createElement('td');
    tdVisits.className = 'num';
    tdVisits.textContent = '—';
    tr.appendChild(tdVisits);

    const tdQ = document.createElement('td');
    tdQ.className = 'num';
    tdQ.textContent = r.score == null ? '—' : Number(r.score).toFixed(3);
    tr.appendChild(tdQ);

    const tdPrior = document.createElement('td');
    tdPrior.className = 'num';
    tdPrior.textContent = '—';
    tr.appendChild(tdPrior);

    const tdLeaf = document.createElement('td');
    tdLeaf.className = 'num';
    tdLeaf.textContent = '—';
    tr.appendChild(tdLeaf);

    els.searchRows.appendChild(tr);
  }
}

// Return the snapshot the user is currently viewing, or null if no
// game is loaded yet. Pulled out so ``_renderSearchAnnotation`` can
// build a card-id→name map without poking at module-globals inline.
function _currentSnapshot() {
  if (!game || !game.snapCache) return null;
  return game.snapCache.get(game.currentIdx) || null;
}

// Walk every zone of a snapshot and collect ``id -> name`` entries
// for cards / permanents / stack objects. Used by ``formatAction`` to
// render human-readable labels in the Decision panel.
//
// Zones covered: ``players[].hand``, ``players[].battlefield``,
// ``players[].graveyard``, ``players[].exile``, ``stack``. Each entry
// has ``id`` (int permanent / card id) and ``name`` (string).
//
// Returns a Map keyed by numeric id. Missing snapshot or malformed
// shape returns an empty map — callers treat that as "fall back to
// id-only formatting."
function _buildCardIdMap(snap) {
  const out = new Map();
  if (!snap || typeof snap !== 'object') return out;
  const collect = (arr) => {
    if (!Array.isArray(arr)) return;
    for (const c of arr) {
      if (c && typeof c === 'object'
          && typeof c.id === 'number'
          && typeof c.name === 'string') {
        out.set(c.id, c.name);
      }
    }
  };
  if (Array.isArray(snap.players)) {
    for (const p of snap.players) {
      if (!p || typeof p !== 'object') continue;
      collect(p.hand);
      collect(p.battlefield);
      collect(p.graveyard);
      collect(p.exile);
    }
  }
  collect(snap.stack);
  return out;
}

// Render the human-readable label for a captured Decision JSON.
//
// Known kinds get a "Kind — Name [extra]" form so replay scrubbing
// reads at a glance:
//   PlayLand(card_id=71)              -> "PlayLand — Swamp"
//   CastSpell(card_id=78, ...)        -> "CastSpell — Hylderblade"
//   ActivateAbility(source_id=12,
//                   ability_index=0)  -> "ActivateAbility — Sol Ring [0]"
//   AttackerDecision(attacker_id=44,
//                    attack=true)     -> "Attack — Comet Crawler"
//   AttackerDecision(... attack=false)-> "AttackerDecision — Comet Crawler (no attack)"
//   BlockerDecision(blocker_id=1,
//                   attacker_id=62)   -> "Block — Dauntless Scrapbot blocks Nova Hellkite"
//   PassPriority                      -> "PassPriority"
//
// Unknown kinds OR unresolvable ids fall back to the legacy
// ``Kind(field=value, ...)`` shape so the panel never goes blank.
//
// ``idToName`` is optional — when undefined / empty the formatter
// degrades gracefully to id-only labels. Pure: returns a string,
// never touches the DOM.
function formatAction(actionJson, idToName) {
  if (!actionJson || typeof actionJson !== 'object') return String(actionJson);
  const map = (idToName instanceof Map) ? idToName : new Map();
  const kind = actionJson.kind || '?';
  const lookup = (id) => {
    if (typeof id !== 'number') return null;
    return map.get(id) || null;
  };

  if (kind === 'PlayLand') {
    const name = lookup(actionJson.card_id);
    if (name) return `PlayLand — ${name}`;
  } else if (kind === 'CastSpell') {
    const name = lookup(actionJson.card_id);
    if (name) {
      const tags = [];
      if (actionJson.use_warp)      tags.push('warp');
      if (actionJson.kicked)        tags.push('kicked');
      if (actionJson.cast_for_free) tags.push('free');
      if (actionJson.use_impending) tags.push('impending');
      if (actionJson.cast_room_half) tags.push(`half=${actionJson.cast_room_half}`);
      const suffix = tags.length > 0 ? ` (${tags.join(', ')})` : '';
      return `CastSpell — ${name}${suffix}`;
    }
  } else if (kind === 'UnlockRoomHalf') {
    const name = lookup(actionJson.card_id);
    if (name) {
      const half = actionJson.half_name ? ` [${actionJson.half_name}]` : '';
      return `UnlockRoomHalf — ${name}${half}`;
    }
  } else if (kind === 'ActivateAbility') {
    const name = lookup(actionJson.source_id);
    if (name) {
      const idx = actionJson.ability_index;
      const suffix = (typeof idx === 'number') ? ` [${idx}]` : '';
      return `ActivateAbility — ${name}${suffix}`;
    }
  } else if (kind === 'ActivateStation') {
    const ship = lookup(actionJson.spacecraft_id);
    const crew = lookup(actionJson.crew_id);
    if (ship && crew) return `ActivateStation — ${crew} → ${ship}`;
  } else if (kind === 'TurnFaceUp') {
    const name = lookup(actionJson.card_id);
    if (name) return `TurnFaceUp — ${name}`;
  } else if (kind === 'AttackerDecision') {
    const name = lookup(actionJson.attacker_id);
    if (name) {
      return actionJson.attack
        ? `Attack — ${name}`
        : `AttackerDecision — ${name} (no attack)`;
    }
  } else if (kind === 'BlockerDecision') {
    const blocker = lookup(actionJson.blocker_id);
    const attacker = lookup(actionJson.attacker_id);
    if (blocker && attacker) {
      return `Block — ${blocker} blocks ${attacker}`;
    }
  } else if (kind === 'ResolutionDiscardDecision') {
    const name = lookup(actionJson.card_id);
    if (name) return `Discard — ${name}`;
  } else if (kind === 'ResolutionSacrificeDecision') {
    const name = lookup(actionJson.object_id);
    if (name) return `Sacrifice — ${name}`;
  } else if (kind === 'ResolutionCardFromLibraryDecision') {
    const name = lookup(actionJson.card_id);
    if (name) return `ChooseFromLibrary — ${name}`;
  } else if (kind === 'ResolutionCardFromOpponentHandDecision') {
    const name = lookup(actionJson.card_id);
    if (name) return `ChooseFromOpponentHand — ${name}`;
  } else if (kind === 'ResolutionTargetDecision') {
    const target = actionJson.target;
    if (Array.isArray(target) && target.length === 2) {
      const tName = lookup(target[1]);
      if (tName) return `Target — ${tName}`;
    }
  }

  // Fallback: legacy ``kind(field=value, ...)`` shape. Unknown kinds
  // and shapes whose id couldn't be resolved both land here.
  const fields = Object.entries(actionJson).filter(([k]) => k !== 'kind');
  if (fields.length === 0) return kind;
  const parts = fields.map(([k, v]) => {
    if (Array.isArray(v)) return `${k}=[${v.length}]`;
    return `${k}=${v}`;
  });
  return `${kind}(${parts.join(', ')})`;
}

// Stable structural equality for two encoded actions. Decision JSON is
// shallow (string kind + scalar / list fields), so a JSON.stringify
// round-trip is safe — the encoder always serializes deterministic
// orderings (sorted attacker_ids, etc).
function actionsEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch (e) {
    return false;
  }
}

function renderPhaseTracker(snap) {
  const cur = snap.step;  // string or null
  const curIdx = PHASE_STEPS.findIndex(s => s.key === cur);
  els.phaseList.replaceChildren();
  PHASE_STEPS.forEach((step, i) => {
    const li = document.createElement('li');
    li.className = 'phase-step';
    if (step.group) li.classList.add(step.group);
    if (curIdx >= 0) {
      if (i === curIdx) li.classList.add('current');
      else if (i < curIdx) li.classList.add('past');
    }
    li.textContent = step.label;
    li.title = step.key;
    els.phaseList.appendChild(li);
  });
}

function drawCombatArrows(snap) {
  const svg = els.combatOverlay;
  // Clear previous arrows but keep the <defs> block (marker definitions).
  Array.from(svg.querySelectorAll('line,path')).forEach(n => {
    if (n.parentNode === svg) svg.removeChild(n);
  });

  const bfCenter = id => {
    const el = document.querySelector(`.battlefield .card[data-obj-id="${id}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  };
  const playerCenter = pid => {
    const row = document.getElementById(`row-p${pid}`);
    const life = row && row.querySelector('.life');
    if (!life) return null;
    const r = life.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  };

  for (const p of snap.players) {
    for (const o of p.battlefield) {
      if (!o.attacking) continue;
      const from = bfCenter(o.id);
      if (!from) continue;
      if (o.blocked_by && o.blocked_by.length > 0) {
        for (const bid of o.blocked_by) {
          const to = bfCenter(bid);
          if (to) svg.appendChild(arrowLine(from, to, true));
        }
      } else if (o.attacking_defender) {
        const to = playerCenter(o.attacking_defender.id);
        if (to) svg.appendChild(arrowLine(from, to, false));
      }
    }
  }
}

function arrowLine(from, to, isBlock) {
  const line = document.createElementNS(SVG_NS, 'line');
  line.setAttribute('x1', from.x);
  line.setAttribute('y1', from.y);
  line.setAttribute('x2', to.x);
  line.setAttribute('y2', to.y);
  line.setAttribute('class', 'combat-arrow' + (isBlock ? ' block' : ''));
  line.setAttribute(
    'marker-end',
    isBlock ? 'url(#arrow-head-block)' : 'url(#arrow-head-attack)',
  );
  return line;
}

// Arrows are positioned in viewport coords, so reposition on scroll/resize.
var _redrawScheduled = false;  // var so tests can reset (see top of file)
function _scheduleArrowRedraw() {
  if (_redrawScheduled || !game) return;
  _redrawScheduled = true;
  requestAnimationFrame(() => {
    _redrawScheduled = false;
    const snap = game.snapCache.get(game.currentIdx);
    if (snap) drawCombatArrows(snap);
  });
}
window.addEventListener('scroll', _scheduleArrowRedraw, true);
window.addEventListener('resize', _scheduleArrowRedraw);

function renderPlayer(root, p, snap, isTop) {
  root.replaceChildren();
  root.classList.toggle('active', snap.active_player_idx === p.id && !snap.is_over);

  root.appendChild(sidePanel(p, snap));

  const region = document.createElement('div');
  region.className = 'zones-region';
  const bf = battlefieldEl(p, isTop);
  const hand = handEl(p);
  // Opponent (top): hand above their battlefield; You (bottom): battlefield above hand.
  region.append(isTop ? hand : bf, isTop ? bf : hand);
  root.appendChild(region);
}

function sidePanel(p, snap) {
  const side = document.createElement('div');
  side.className = 'side-panel';

  const nameRow = document.createElement('div');
  nameRow.className = 'name-row';
  const pname = document.createElement('span');
  pname.className = 'pname';
  if (snap.is_over && !snap.is_draw && snap.winner_id === p.id) pname.classList.add('winner');
  if (p.has_lost) pname.classList.add('loser');
  const activeDot = (snap.active_player_idx === p.id && !snap.is_over) ? ' ⬤' : '';
  pname.textContent = `Player ${p.id}${activeDot}`;
  nameRow.appendChild(pname);
  side.appendChild(nameRow);

  const lifeLabel = document.createElement('div');
  lifeLabel.className = 'life-label';
  lifeLabel.textContent = 'life';
  side.appendChild(lifeLabel);

  const life = document.createElement('div');
  let lifeCls = 'life';
  if (p.life <= 5) lifeCls += ' low';
  else if (p.life >= 25) lifeCls += ' high';
  life.className = lifeCls;
  life.textContent = p.life;
  life.dataset.playerId = p.id;
  side.appendChild(life);

  const zones = document.createElement('div');
  zones.className = 'zones';
  // Library is hidden info — don't make it clickable. Graveyard is public,
  // and Exile is public; both open a modal listing the actual cards.
  zones.append(
    zoneBox('Hand', p.hand_size),
    zoneBox('Library', p.library_size),
    zoneBox('Graveyard', p.graveyard_size,
            p.graveyard_size > 0
              ? () => openZoneModal(`Player ${p.id} — Graveyard`, p.graveyard)
              : null),
  );
  side.appendChild(zones);

  if (p.exile_size > 0) {
    const row = document.createElement('div');
    row.className = 'zones';
    row.appendChild(zoneBox(
      'Exile', p.exile_size,
      () => openZoneModal(`Player ${p.id} — Exile`, p.exile),
    ));
    side.appendChild(row);
  }

  if (p.mana_pool && p.mana_pool.length > 0) {
    side.appendChild(manaPoolEl(p.mana_pool));
  }

  if (p.poison_counters) {
    side.appendChild(poisonEl(p.poison_counters));
  }

  return side;
}

function poisonEl(count) {
  const row = document.createElement('div');
  row.className = 'poison-row';
  const label = document.createElement('span');
  label.className = 'poison-label';
  label.textContent = '☠ Poison';
  const c = document.createElement('span');
  c.className = 'poison-count' + (count >= 10 ? ' lethal' : '');
  c.textContent = count;
  row.append(label, c);
  return row;
}

function manaPoolEl(coins) {
  const row = document.createElement('div');
  row.className = 'mana-pool';
  const label = document.createElement('span');
  label.className = 'mana-pool-label';
  label.textContent = 'Mana';
  row.appendChild(label);
  coins.forEach(coin => {
    const c = document.createElement('span');
    c.className = 'mana-coin ' + (coin.color || 'C');
    if (coin.spend_restriction) {
      c.classList.add('restricted');
      c.title = `Can only be spent on: ${coin.spend_restriction}`;
    } else {
      c.title = coin.color === 'any' ? 'Any-color mana' : `${coin.color} mana`;
    }
    c.textContent = coin.color === 'any' ? '*'
                   : coin.color === 'C' ? '◇' : coin.color;
    row.appendChild(c);
  });
  return row;
}

function zoneBox(label, count, onClick) {
  const div = document.createElement('div');
  div.className = 'zone-box';
  const l = document.createElement('div');
  l.className = 'zlabel';
  l.textContent = label;
  const c = document.createElement('div');
  c.className = 'zcount';
  c.textContent = count;
  div.append(l, c);
  if (onClick) {
    div.classList.add('clickable');
    div.addEventListener('click', onClick);
    div.tabIndex = 0;
    div.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        onClick();
      }
    });
  }
  return div;
}

// ------------------------------------------------------------------------
// Zone-content modal (graveyard / exile browser)
// ------------------------------------------------------------------------

function openZoneModal(title, cards) {
  els.zoneModalTitle.textContent = title;
  els.zoneModalCards.replaceChildren();
  if (!cards || cards.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'zone-modal-empty';
    empty.textContent = '(empty)';
    els.zoneModalCards.appendChild(empty);
  } else {
    cards.forEach((c) => {
      els.zoneModalCards.appendChild(cardEl(c, { permanent: false }));
    });
  }
  els.zoneModal.hidden = false;
  els.zoneModal.classList.remove('hidden');
}

function closeZoneModal() {
  els.zoneModal.hidden = true;
  els.zoneModal.classList.add('hidden');
  hidePreview();
}

els.zoneModalClose.addEventListener('click', closeZoneModal);
els.zoneModal.addEventListener('click', (ev) => {
  // Click on the dim backdrop (not the inner box) closes.
  if (ev.target === els.zoneModal) closeZoneModal();
});

function battlefieldEl(p, isTop) {
  const bf = document.createElement('div');
  bf.className = 'battlefield';
  const lands    = p.battlefield.filter(o =>  o.is_land);
  const others   = p.battlefield.filter(o => !o.is_land && !o.is_creature_now);
  const creatures = p.battlefield.filter(o => !o.is_land &&  o.is_creature_now);

  const creaturesRow = rowOfCards(groupBattlefield(creatures), 'bf-row creatures', true);

  // Back row: lands on the left, non-land non-creatures on the right.
  const backRow = document.createElement('div');
  backRow.className = 'bf-row back';
  groupBattlefield(lands).forEach(c => backRow.appendChild(cardEl(c, { permanent: true })));
  if (lands.length > 0 && others.length > 0) {
    const sep = document.createElement('div');
    sep.className = 'back-row-sep';
    backRow.appendChild(sep);
  }
  groupBattlefield(others).forEach(c => backRow.appendChild(cardEl(c, { permanent: true })));

  // Back row is furthest from the stack; creatures row is closest.
  bf.append(isTop ? backRow : creaturesRow, isTop ? creaturesRow : backRow);
  return bf;
}

// Group identical permanents into stacks rendered as a single card with an
// "Nx" badge. Two permanents collapse only if every visually-distinguishing
// piece of state matches: name, tap status, summoning sickness, attack/block
// status, every kind of counter, damage, attached_to host, station status.
// (If anything that affects rendering differs, they get their own stack.)
// Returns a list of "render objects" in the original order; each is just a
// reference to the first card in its group with a synthetic ``stack_count``
// field telling cardEl how many to draw.
function groupBattlefield(cards) {
  const groups = new Map();
  const order = [];
  for (const c of cards) {
    const key = stackKey(c);
    if (groups.has(key)) {
      groups.get(key).stack_count += 1;
    } else {
      // Shallow-copy with stack_count so we don't mutate the snapshot.
      const proxy = Object.assign({}, c, { stack_count: 1 });
      groups.set(key, proxy);
      order.push(proxy);
    }
  }
  return order;
}

function stackKey(c) {
  // Anything that produces a different visual on the card must be in this
  // key. Use the field names cardEl reads. Ordering is fixed so the JSON
  // serialization is canonical for any equal pair of permanents.
  return JSON.stringify([
    c.name,
    !!c.tapped,
    !!c.summoning_sick && !!c.is_creature_now,
    !!c.attacking,
    !!c.blocking,
    !!c.is_token,
    c.plus_one_counters || 0,
    c.damage || 0,
    c.stun_counters || 0,
    c.loyalty_counters || 0,
    c.charge_counters || 0,
    !!c.is_stationed,
    c.attached_to ? c.attached_to.id : null,
    c.effective_power ?? null,
    c.effective_toughness ?? null,
  ]);
}

function handEl(p) {
  const hand = document.createElement('div');
  hand.className = 'hand';
  p.hand.forEach(c => hand.appendChild(cardEl(c, { permanent: false })));
  // If hand_size > revealed cards, the rest are hidden from us — render
  // face-down placeholders so the count is still visible.
  const hidden = Math.max(0, (p.hand_size || 0) - p.hand.length);
  for (let i = 0; i < hidden; i++) {
    hand.appendChild(faceDownCardEl());
  }
  return hand;
}

function faceDownCardEl() {
  const el = document.createElement('div');
  el.className = 'card face-down';
  el.title = 'Face-down (opponent hand hidden)';
  return el;
}

function rowOfCards(cards, cls, permanent) {
  const row = document.createElement('div');
  row.className = cls;
  cards.forEach(c => row.appendChild(cardEl(c, { permanent })));
  return row;
}

function cardEl(c, { permanent }) {
  const el = document.createElement('div');
  el.className = 'card';
  if (c.id !== undefined) el.dataset.objId = c.id;
  if (c.scryfall_id) {
    el.style.backgroundImage = `url('/card_image/${c.scryfall_id}/small')`;
  } else {
    el.classList.add('no-image');
    const name = document.createElement('div');
    name.className = 'fallback-name';
    name.textContent = c.name;
    el.appendChild(name);
    if (c.type_line) {
      const t = document.createElement('div');
      t.className = 'fallback-type';
      t.textContent = c.type_line;
      el.appendChild(t);
    }
    if (c.is_creature && c.power !== null && c.power !== undefined) {
      const pt = document.createElement('div');
      pt.className = 'fallback-pt';
      pt.textContent = `${c.power}/${c.toughness}`;
      el.appendChild(pt);
    }
  }
  if (permanent) {
    if (c.tapped) el.classList.add('tapped');
    if (c.summoning_sick && c.is_creature_now) el.classList.add('summoning-sick');
    if (c.attacking) el.classList.add('attacking');
    if (c.blocking) el.classList.add('blocking');
    if (c.is_token) el.classList.add('token');
    if (c.stack_count && c.stack_count > 1) {
      el.classList.add('stacked');
      const sb = document.createElement('div');
      sb.className = 'stack-badge';
      sb.textContent = `${c.stack_count}x`;
      el.appendChild(sb);
    }

    if (c.is_creature_now) {
      const pt = document.createElement('div');
      pt.className = 'pt-badge';
      const pow = c.effective_power ?? '?';
      const tgh = c.effective_toughness ?? '?';
      pt.textContent = `${pow}/${tgh}`;
      el.appendChild(pt);
    }
    if (c.plus_one_counters) {
      const b = document.createElement('div');
      b.className = 'counter-badge';
      b.textContent = `+${c.plus_one_counters}`;
      el.appendChild(b);
    }
    if (c.damage) {
      const b = document.createElement('div');
      b.className = 'damage-badge';
      b.textContent = c.damage;
      el.appendChild(b);
    }
    if (c.stun_counters) {
      const b = document.createElement('div');
      b.className = 'stun-badge';
      b.textContent = c.stun_counters;
      b.title = `${c.stun_counters} stun counter(s)`;
      el.appendChild(b);
    }
    if (c.is_planeswalker && c.loyalty_counters) {
      const b = document.createElement('div');
      b.className = 'loyalty-badge';
      const inner = document.createElement('span');
      inner.textContent = c.loyalty_counters;
      b.appendChild(inner);
      b.title = `${c.loyalty_counters} loyalty`;
      el.appendChild(b);
    }
    if (c.attached_to) {
      el.classList.add('attached');
      const tag = document.createElement('div');
      tag.className = 'attached-to';
      tag.textContent = '⚔ ' + c.attached_to.name;
      tag.title = 'Attached to ' + c.attached_to.name;
      el.appendChild(tag);
    }
    if (c.is_spacecraft && c.station_threshold) {
      const bar = document.createElement('div');
      bar.className = 'station-bar' + (c.is_stationed ? ' stationed' : '');
      bar.textContent = `${c.charge_counters}/${c.station_threshold}`;
      bar.title = c.is_stationed
        ? `Stationed (${c.charge_counters}/${c.station_threshold} charge counters)`
        : `${c.charge_counters}/${c.station_threshold} charge counters` +
          ` — ${c.station_threshold - c.charge_counters} more to station`;
      el.appendChild(bar);
    }
  }

  // Warp indicator. Only meaningful for cards in exile (a card on the
  // battlefield/hand never has warped=true), but the badge is rendered
  // unconditionally on the field — graveyard cards never carry it
  // either since the flag clears when they leave exile via cast or
  // bounce. Marks cards that were exiled by their own warp ability,
  // which can be re-cast from exile at normal cost. Cards exiled by
  // other means (Banishing Light) don't get this — only the engine
  // gates re-cast on the same flag.
  if (c.warped) {
    el.classList.add('warped');
    const wb = document.createElement('div');
    wb.className = 'warp-badge';
    wb.textContent = 'WARP';
    wb.title = 'Warp-exiled — castable from exile at normal cost';
    el.appendChild(wb);
  }

  el.addEventListener('mouseenter', (ev) => showPreview(ev, c));
  el.addEventListener('mousemove', positionPreview);
  el.addEventListener('mouseleave', hidePreview);
  return el;
}

function showPreview(ev, c) {
  const prev = els.preview;
  if (c.scryfall_id) {
    prev.style.backgroundImage = `url('/card_image/${c.scryfall_id}/normal')`;
    prev.classList.remove('no-image');
    prev.textContent = '';
  } else {
    prev.style.backgroundImage = '';
    prev.classList.add('no-image');
    prev.textContent = c.name + (c.type_line ? `  —  ${c.type_line}` : '');
  }
  prev.hidden = false;
  positionPreview(ev);
}

function positionPreview(ev) {
  const prev = els.preview;
  const pad = 16;
  const w = prev.offsetWidth || 280;
  const h = prev.offsetHeight || 392;
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + w + pad > window.innerWidth) x = ev.clientX - w - pad;
  if (y + h + pad > window.innerHeight) y = ev.clientY - h - pad;
  prev.style.left = `${Math.max(0, x)}px`;
  prev.style.top = `${Math.max(0, y)}px`;
}

function hidePreview() { els.preview.hidden = true; }

function renderStack(root, snap) {
  root.replaceChildren();
  if (!snap.stack || snap.stack.length === 0) {
    root.classList.add('empty');
    root.textContent = 'Stack is empty';
    return;
  }
  root.classList.remove('empty');
  const label = document.createElement('span');
  label.className = 'muted';
  label.textContent = `Stack (${snap.stack.length}):`;
  root.appendChild(label);
  snap.stack.forEach((item, idx) => {
    const el = stackItemEl(item);
    el.dataset.stackIdx = idx;
    root.appendChild(el);
  });
}

function stackItemEl(item) {
  const wrap = document.createElement('div');
  wrap.className = 'stack-item';
  if (item.countered) wrap.classList.add('countered-spell');

  // Mini card image (or text fallback) — hover gives the same large preview
  // the battlefield cards use.
  const img = document.createElement('div');
  img.className = 'stack-item-img';
  if (item.scryfall_id) {
    img.style.backgroundImage = `url('/card_image/${item.scryfall_id}/small')`;
  } else {
    img.classList.add('no-image');
    img.textContent = item.name;
  }
  const previewCard = {
    scryfall_id: item.scryfall_id,
    name: item.name,
    type_line: item.is_ability ? 'Ability' : 'Spell',
  };
  img.addEventListener('mouseenter', (ev) => showPreview(ev, previewCard));
  img.addEventListener('mousemove', positionPreview);
  img.addEventListener('mouseleave', hidePreview);
  wrap.appendChild(img);

  const text = document.createElement('div');
  text.className = 'stack-item-text';

  const line1 = document.createElement('div');
  const nameSpan = document.createElement('span');
  nameSpan.className = 'stack-item-name';
  nameSpan.textContent = item.name;
  line1.appendChild(nameSpan);
  const cc = document.createElement('span');
  cc.className = 'controller';
  cc.textContent = `P${item.controller_id}`;
  line1.appendChild(cc);
  if (item.mana_cost) {
    const m = document.createElement('span');
    m.className = 'mana-cost';
    m.textContent = item.mana_cost;
    line1.appendChild(m);
  }
  text.appendChild(line1);

  if (item.is_ability && item.ability_description) {
    const desc = document.createElement('div');
    desc.className = 'ability-desc';
    desc.textContent = item.ability_description;
    text.appendChild(desc);
  }

  const meta = document.createElement('div');
  meta.className = 'stack-item-meta';
  if (item.x_value) {
    const x = document.createElement('span');
    x.className = 'x-value';
    x.textContent = `X=${item.x_value}`;
    meta.appendChild(x);
  }
  if (item.kicked) {
    const k = document.createElement('span');
    k.className = 'kicked';
    k.textContent = 'kicker';
    meta.appendChild(k);
  }
  if (item.countered) {
    const c = document.createElement('span');
    c.className = 'countered';
    c.textContent = 'countered';
    meta.appendChild(c);
  }
  if (item.targets && item.targets.length > 0) {
    const t = document.createElement('span');
    t.className = 'targets';
    t.textContent = '→ ' + item.targets
      .map(tg => tg.name || `${tg.kind} #${tg.id}`)
      .join(', ');
    meta.appendChild(t);
  }
  if (meta.children.length > 0) text.appendChild(meta);

  wrap.appendChild(text);
  return wrap;
}

function renderLog(body, snap) {
  body.replaceChildren();
  // Prefer the full game log (set by activateGame) so the user sees
  // every event, past and future, and can click ahead to skip
  // forward. Fall back to snap.log if the full log isn't available
  // (older replays / out-of-game state / tests that don't set game).
  const fullLog = game && game.fullLog;
  const entries = fullLog || snap.log || [];
  const currentLogSize = (fullLog && game.index && game.index[game.currentIdx])
    ? game.index[game.currentIdx].log_size
    : entries.length;
  const recentCount = (snap.log_since_last || []).length;
  // Range of "this-step" entries: the ones added since the previous
  // snapshot. With the full log present we derive it from the index;
  // otherwise we trust snap.log_since_last.
  const firstRecent = (fullLog && currentLogSize !== undefined)
    ? currentLogSize - recentCount
    : entries.length - recentCount;
  entries.forEach((e, i) => {
    const d = document.createElement('div');
    d.className = 'log-entry';
    if (recentCount > 0 && i >= firstRecent && i < firstRecent + recentCount) {
      d.classList.add('this-step');
    } else if (currentLogSize !== undefined && i >= currentLogSize) {
      d.classList.add('future');
    }
    renderLogEntryText(d, e);
    // Click jumps (forward or backward) to the first snapshot that
    // introduced this log entry. Index entries carry log_size
    // (cumulative len(state.log) at that snapshot); the first idx
    // with log_size > i is the snapshot where entry i first appeared.
    d.addEventListener('click', () => {
      const target = firstSnapshotWithLogEntry(i);
      if (target !== null && target !== game.currentIdx) goTo(target);
    });
    body.appendChild(d);
  });
  // Scroll the current step's entries into view (the highlighted
  // band). Important: scope scrolling to the log body's own
  // overflow rather than calling scrollIntoView, which would scroll
  // the document and push the fixed bottom controls off-screen on
  // desktop. We compute the target's position relative to the body
  // via getBoundingClientRect (independent of offsetParent quirks)
  // and center it inside the visible region.
  if (fullLog && currentLogSize !== undefined) {
    const target = body.children[Math.max(0, currentLogSize - 1)];
    if (target && typeof body.getBoundingClientRect === 'function') {
      const tr = target.getBoundingClientRect();
      const br = body.getBoundingClientRect();
      const targetTopInBody = tr.top - br.top + body.scrollTop;
      body.scrollTop = targetTopInBody
        - body.clientHeight / 2
        + target.offsetHeight / 2;
    }
  } else {
    body.scrollTop = body.scrollHeight;
  }
}

// Render a log entry's text into ``el``. If the entry contains the
// "(warp)" suffix produced by a warp cast (see actions.cast_spell), tag
// the entry with the .warp class and replace the inline "(warp)" with
// a magenta WARP badge so the alt cast mode is unmistakable in the log.
function renderLogEntryText(el, text) {
  const warpIdx = text.indexOf(' (warp)');
  if (warpIdx === -1) {
    el.textContent = text;
    return;
  }
  el.classList.add('warp');
  el.appendChild(document.createTextNode(text.slice(0, warpIdx)));
  const badge = document.createElement('span');
  badge.className = 'warp-badge-inline';
  badge.textContent = 'WARP';
  el.appendChild(document.createTextNode(' '));
  el.appendChild(badge);
  const tail = text.slice(warpIdx + ' (warp)'.length);
  if (tail) el.appendChild(document.createTextNode(tail));
}

// Binary search over game.index for the smallest snapshot idx whose log
// already contains entry ``logIdx``. Returns null if the index lacks
// log_size data (older replays / out-of-game state).
function firstSnapshotWithLogEntry(logIdx) {
  if (!game || !game.index || game.index.length === 0) return null;
  if (game.index[0].log_size === undefined) return null;
  let lo = 0;
  let hi = game.index.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (game.index[mid].log_size > logIdx) hi = mid;
    else lo = mid + 1;
  }
  return game.index[lo].log_size > logIdx ? lo : null;
}

// ============================================================================
// Mode tabs — Watch / Play
// ============================================================================

const tabWatch   = document.getElementById('tab-watch');
const tabPlay    = document.getElementById('tab-play');
const panelWatch = document.getElementById('play-form');
const panelPlay  = document.getElementById('iplay-form');

function switchMode(toPlay) {
  const playMode = !!toPlay;
  tabWatch.classList.toggle('active', !playMode);
  tabPlay .classList.toggle('active',  playMode);
  tabWatch.setAttribute('aria-selected', String(!playMode));
  tabPlay .setAttribute('aria-selected', String( playMode));
  panelWatch.classList.toggle('hidden',  playMode);
  panelPlay .classList.toggle('hidden', !playMode);
  // Hide the watch scrubber/phase-tracker when entering play mode
  if (playMode) {
    els.scrubber    .classList.add('hidden');
    els.phaseTracker.classList.add('hidden');
  }
}

tabWatch.addEventListener('click', () => switchMode(false));
tabPlay .addEventListener('click', () => switchMode(true));

// ============================================================================
// Interactive play — client-side state + poll loop
// ============================================================================

var iplay = null;  // { sessionId, humanPlayerId, pollTimer, lastPending }
const POLL_INTERVAL_MS = 250;

const elsI = {
  form:          document.getElementById('iplay-form'),
  seed:          document.getElementById('iplay-seed'),
  myColor:       document.getElementById('iplay-my-color'),
  botColor:      document.getElementById('iplay-bot-color'),
  playerId:      document.getElementById('iplay-player-id'),
  maxTurns:      document.getElementById('iplay-max-turns'),
  decisionPanel: document.getElementById('decision-panel'),
  waiting:       document.getElementById('decision-waiting'),
  body:          document.getElementById('decision-body'),
  prompt:        document.getElementById('decision-prompt'),
  options:       document.getElementById('decision-options'),
  modal:         document.getElementById('decision-modal'),
  modalTitle:    document.getElementById('decision-modal-title'),
  modalBody:     document.getElementById('decision-modal-body'),
};

elsI.form.addEventListener('submit', (ev) => { ev.preventDefault(); startInteractiveGame(); });

// Space = pass priority, Enter = confirm (attackers / blockers).
window.addEventListener('keydown', (ev) => {
  if (!iplay) return;
  const tag = (ev.target && ev.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (ev.key === ' ' || ev.key === 'Enter') {
    const btn = elsI.options.querySelector(
      ev.key === ' '
        ? '.action-btn-pass'
        : '.action-btn-confirm',
    );
    if (btn) { ev.preventDefault(); btn.click(); }
  }
});

async function startInteractiveGame() {
  stopInteractiveGame();
  const humanPlayer = parseInt(elsI.playerId.value, 10);
  const myColor     = elsI.myColor.value;
  const botColor    = elsI.botColor.value;
  const c1 = humanPlayer === 0 ? myColor : botColor;
  const c2 = humanPlayer === 1 ? myColor : botColor;
  setStatus('Starting game…', 'muted');
  elsI.form.querySelector('button').disabled = true;
  try {
    const body = {
      seed:         parseInt(elsI.seed.value, 10) || 0,
      c1, c2,
      human_player: humanPlayer,
      max_turns:    parseInt(elsI.maxTurns.value, 10) || 300,
    };
    const data = await postJSON('/api/interactive/new', body);
    iplay = {
      sessionId:     data.session_id,
      humanPlayerId: data.human_player_id,
      pollTimer:     null,
      lastPending:   null,
    };
    elsI.decisionPanel.classList.remove('hidden');
    els.board    .classList.remove('hidden');
    els.logPanel .classList.remove('hidden');
    schedulePoll();
  } catch (e) {
    setStatus('Error: ' + e.message, 'loss');
  } finally {
    elsI.form.querySelector('button').disabled = false;
  }
}

function stopInteractiveGame() {
  if (!iplay) return;
  clearTimeout(iplay.pollTimer);
  iplay = null;
  elsI.decisionPanel.classList.add('hidden');
  closeDecisionModal();
}

function schedulePoll() {
  if (!iplay) return;
  iplay.pollTimer = setTimeout(pollState, POLL_INTERVAL_MS);
}

async function pollState() {
  if (!iplay) return;
  try {
    const state = await fetchJSON(`/api/interactive/state/${iplay.sessionId}`);
    handleState(state);
  } catch (e) {
    setStatus('Poll error: ' + e.message, 'loss');
    schedulePoll();
  }
}

function handleState(state) {
  if (!iplay) return;

  if (state.snapshot) {
    // Render the live board. We pass a minimal summary-like object since
    // renderSnapshot only needs winner_id/is_draw from it for the status bar;
    // everything else comes from the snapshot itself.
    const summary = {
      winner_id: state.result ? state.result.winner_id : null,
      is_draw:   state.result ? state.result.is_draw   : false,
    };
    renderSnapshot(summary, state.snapshot);
    renderInteractiveLog(state.snapshot);
  }

  if (state.is_over) {
    const r = state.result || {};
    if (r.timed_out) setStatus('Game timed out', 'loss');
    else if (r.error)  setStatus('Game error: ' + r.error, 'loss');
    else if (r.is_draw) setStatus(`Draw — ${r.turn_number} turns`, 'draw');
    else if (r.winner_id != null) {
      const youWon = r.winner_id === iplay.humanPlayerId;
      setStatus(
        `${youWon ? 'You win' : 'Bot wins'} — turn ${r.turn_number}`,
        youWon ? 'win' : 'loss',
      );
    }
    elsI.waiting.classList.add('hidden');
    elsI.body   .classList.add('hidden');
    return;
  }

  if (state.pending) {
    elsI.waiting.classList.add('hidden');
    elsI.body   .classList.remove('hidden');
    renderDecision(state.pending);
  } else {
    elsI.body   .classList.add('hidden');
    elsI.waiting.classList.remove('hidden');
    schedulePoll();
  }
}

// ---------------------------------------------------------------------------
// Submit a response to the current pending decision
// ---------------------------------------------------------------------------

async function submitAction(response) {
  if (!iplay) return;
  closeDecisionModal();
  elsI.waiting.classList.remove('hidden');
  elsI.body   .classList.add('hidden');
  try {
    await postJSON(`/api/interactive/action/${iplay.sessionId}`, { response });
  } catch (e) {
    setStatus('Submit error: ' + e.message, 'loss');
  }
  schedulePoll();
}

// ---------------------------------------------------------------------------
// Decision rendering — top-level dispatcher
// ---------------------------------------------------------------------------

function renderDecision(pending) {
  const type = pending.type;

  // Auto-submit when there is only one possible choice — no point making the
  // player click through it (most common case: solo PassPriority).
  if (type === 'choose_action' && pending.options.length === 1) {
    submitAction({ idx: 0 }); return;
  }
  if (type === 'choose_target' && pending.options.length === 1 && !pending.can_decline) {
    submitAction({ idx: 0 }); return;
  }
  if (type === 'choose_discard' && pending.cards && pending.cards.length === 1) {
    submitAction({ id: pending.cards[0].id }); return;
  }
  if (type === 'choose_sacrifice' && pending.objects && pending.objects.length === 1) {
    submitAction({ id: pending.objects[0].id }); return;
  }
  if (type === 'choose_mode' && pending.modes && pending.modes.length === 1) {
    submitAction({ idx: 0 }); return;
  }
  if (type === 'choose_modes' && pending.modes && pending.modes.length === 1) {
    submitAction({ indices: [0] }); return;
  }
  if (type === 'choose_card' && pending.cards && pending.cards.length === 1 && !pending.can_decline) {
    submitAction({ id: pending.cards[0].id }); return;
  }
  if (type === 'choose_one_id' && pending.objects && pending.objects.length === 1) {
    submitAction({ id: pending.objects[0].id }); return;
  }
  if (type === 'choose_mana_color' && pending.options && pending.options.length === 1) {
    submitAction({ color: pending.options[0] }); return;
  }
  if (type === 'declare_attackers' && (!pending.eligible || pending.eligible.length === 0)) {
    submitAction({ ids: [] }); return;
  }
  if (type === 'declare_blockers' && (!pending.eligible || pending.eligible.length === 0)) {
    submitAction({ assignments: {} }); return;
  }

  switch (type) {
    case 'choose_action':     renderChooseAction(pending);    break;
    case 'declare_attackers': renderDeclareAttackers(pending); break;
    case 'declare_blockers':  renderDeclareBlockers(pending);  break;
    case 'choose_target':     renderChooseTarget(pending);    break;
    case 'choose_discard':    renderChooseDiscardBoard(pending);                                          break;
    case 'choose_sacrifice':  renderChooseObj(pending, 'Choose a permanent to sacrifice', (id) => submitAction({ id })); break;
    case 'choose_surveil':    renderChooseSurveil(pending);   break;
    case 'choose_mode':       renderChooseMode(pending);      break;
    case 'choose_modes':      renderChooseModes(pending);     break;
    case 'choose_yes_no':     renderChooseYesNo(pending);     break;
    case 'choose_card':       renderChooseCard(pending, pending.prompt,
                                (id, decline) => decline ? submitAction({ decline: true }) : submitAction({ id })); break;
    case 'choose_x':          renderChooseX(pending);         break;
    case 'choose_multi_ids':  renderChooseMultiIds(pending);  break;
    case 'choose_mana_color': renderChooseManaColor(pending); break;
    case 'choose_one_id':     renderChooseObj(pending, pending.prompt, (id) => submitAction({ id })); break;
    default:
      elsI.prompt.textContent = `Unknown decision: ${type}`;
  }
}

// ---------------------------------------------------------------------------
// choose_action: button row at the bottom (inline, not modal)
// ---------------------------------------------------------------------------

// MTGO-like board-click renderers: instead of a button row, the player
// interacts directly with cards on the rendered board. The click handlers
// are attached after each renderSnapshot call (since renderSnapshot rebuilds
// the DOM and would otherwise wipe them).

function renderChooseAction(pending) {
  // Group options by the object they affect so a single click on one card
  // maps to one or more candidate actions.
  const byCardId = new Map();    // hand-card id  -> [opt, ...]
  const byPermId = new Map();    // battlefield id -> [opt, ...]
  let passOpt = null;

  pending.options.forEach(opt => {
    if (opt.type === 'pass') { passOpt = opt; return; }
    if (opt.type === 'play_land' || opt.type === 'cast_spell') {
      const list = byCardId.get(opt.card_id) || [];
      list.push(opt);
      byCardId.set(opt.card_id, list);
    } else if (opt.type === 'activate') {
      const list = byPermId.get(opt.source_id) || [];
      list.push(opt);
      byPermId.set(opt.source_id, list);
    } else if (opt.type === 'station') {
      const list = byPermId.get(opt.spacecraft_id) || [];
      list.push(opt);
      byPermId.set(opt.spacecraft_id, list);
    }
  });

  // Bottom panel: status + Pass button. Cards / permanents are clickable
  // in-place on the board.
  const playableCount = byCardId.size + byPermId.size;
  elsI.prompt.textContent =
    playableCount === 0
      ? 'You have priority — pass to continue.'
      : 'Click a card to play, a permanent to activate, or pass priority.';
  elsI.options.replaceChildren();
  if (passOpt) {
    const passBtn = document.createElement('button');
    passBtn.type = 'button';
    passBtn.className = 'action-btn action-btn-pass';
    passBtn.textContent = 'Pass priority (Space)';
    passBtn.addEventListener('click', () => submitAction({ idx: passOpt.idx }));
    elsI.options.appendChild(passBtn);
  }

  // Highlight clickable cards in hand.
  byCardId.forEach((opts, cid) => {
    document.querySelectorAll(`.hand .card[data-obj-id="${cid}"]`).forEach(el => {
      el.classList.add('actionable');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        chooseFromOptions(el, opts);
      });
    });
  });

  // Highlight clickable permanents on battlefield.
  byPermId.forEach((opts, oid) => {
    document.querySelectorAll(`.battlefield .card[data-obj-id="${oid}"]`).forEach(el => {
      el.classList.add('actionable');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        chooseFromOptions(el, opts);
      });
    });
  });
}

// Click on a card with multiple available actions: pop a small menu
// next to the card. Single-action: submit directly.
function chooseFromOptions(anchorEl, opts) {
  if (opts.length === 1) {
    submitAction({ idx: opts[0].idx });
    return;
  }
  const menu = document.createElement('div');
  menu.className = 'options-popover';
  opts.forEach(opt => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'options-popover-item';
    btn.textContent = opt.label;
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      menu.remove();
      submitAction({ idx: opt.idx });
    });
    menu.appendChild(btn);
  });

  // Position the popover near the clicked card. Prefer below; flip above
  // if it would extend past the viewport (P0's hand sits at the bottom of
  // the screen, so a below-the-card popover almost always overflows).
  document.body.appendChild(menu);
  const r = anchorEl.getBoundingClientRect();
  const menuRect = menu.getBoundingClientRect();
  let top = r.bottom + 4;
  if (top + menuRect.height > window.innerHeight) {
    top = Math.max(4, r.top - menuRect.height - 4);
  }
  let left = r.left;
  if (left + menuRect.width > window.innerWidth) {
    left = Math.max(4, window.innerWidth - menuRect.width - 4);
  }
  menu.style.position = 'fixed';
  menu.style.left = `${left}px`;
  menu.style.top  = `${top}px`;

  // Dismiss on outside click.
  setTimeout(() => {
    const dismiss = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.remove();
        document.removeEventListener('click', dismiss);
      }
    };
    document.addEventListener('click', dismiss);
  }, 0);
}

// ---------------------------------------------------------------------------
// declare_attackers: click your creatures on the board to toggle them
// ---------------------------------------------------------------------------

function renderDeclareAttackers(pending) {
  iplay.attackerSet = new Set();

  elsI.prompt.textContent =
    'Declare attackers — click your creatures to toggle, then confirm.';
  elsI.options.replaceChildren();

  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'action-btn action-btn-confirm';
  confirmBtn.textContent = 'Confirm attackers (0)';
  confirmBtn.addEventListener('click', () => {
    submitAction({ ids: Array.from(iplay.attackerSet) });
  });
  elsI.options.appendChild(confirmBtn);

  pending.eligible.forEach(creature => {
    document.querySelectorAll(`.battlefield .card[data-obj-id="${creature.id}"]`).forEach(el => {
      el.classList.add('actionable');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (iplay.attackerSet.has(creature.id)) {
          iplay.attackerSet.delete(creature.id);
          document.querySelectorAll(`[data-obj-id="${creature.id}"]`).forEach(e =>
            e.classList.remove('attacker-pending'));
        } else {
          iplay.attackerSet.add(creature.id);
          document.querySelectorAll(`[data-obj-id="${creature.id}"]`).forEach(e =>
            e.classList.add('attacker-pending'));
        }
        confirmBtn.textContent = `Confirm attackers (${iplay.attackerSet.size})`;
      });
    });
  });
}

// ---------------------------------------------------------------------------
// declare_blockers: click blocker → click attacker → assigned. Click blocker
// again to unassign / select a different attacker.
// ---------------------------------------------------------------------------

function renderDeclareBlockers(pending) {
  iplay.blockerAssignments = {};
  iplay.selectedBlocker = null;

  elsI.prompt.textContent =
    'Declare blockers — click your blocker, then click the attacker it blocks. Click the blocker again to unassign.';
  elsI.options.replaceChildren();

  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'action-btn action-btn-confirm';
  confirmBtn.textContent = 'Confirm blockers (0)';
  confirmBtn.addEventListener('click', () => {
    submitAction({ assignments: iplay.blockerAssignments });
  });
  elsI.options.appendChild(confirmBtn);

  const updateConfirmCount = () => {
    const n = Object.keys(iplay.blockerAssignments).length;
    confirmBtn.textContent = `Confirm blockers (${n})`;
  };

  // Eligible blockers (the human's creatures).
  pending.eligible.forEach(blocker => {
    document.querySelectorAll(`.battlefield .card[data-obj-id="${blocker.id}"]`).forEach(el => {
      el.classList.add('actionable');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        // Toggle: if already assigned, unassign. If selected, deselect.
        // Otherwise select for assignment.
        if (iplay.blockerAssignments[blocker.id] !== undefined) {
          delete iplay.blockerAssignments[blocker.id];
          document.querySelectorAll(`[data-obj-id="${blocker.id}"]`).forEach(e =>
            e.classList.remove('blocker-committed'));
          updateConfirmCount();
        } else if (iplay.selectedBlocker === blocker.id) {
          iplay.selectedBlocker = null;
          document.querySelectorAll(`[data-obj-id="${blocker.id}"]`).forEach(e =>
            e.classList.remove('blocker-selected'));
        } else {
          if (iplay.selectedBlocker !== null) {
            document.querySelectorAll(`[data-obj-id="${iplay.selectedBlocker}"]`).forEach(e =>
              e.classList.remove('blocker-selected'));
          }
          iplay.selectedBlocker = blocker.id;
          document.querySelectorAll(`[data-obj-id="${blocker.id}"]`).forEach(e =>
            e.classList.add('blocker-selected'));
        }
      });
    });
  });

  // Attackers (the opponent's creatures that are attacking this player).
  // Clicking commits the currently-selected blocker to this attacker.
  pending.attackers.forEach(attacker => {
    document.querySelectorAll(`.battlefield .card[data-obj-id="${attacker.id}"]`).forEach(el => {
      el.classList.add('actionable-target');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (iplay.selectedBlocker === null) return;
        const bid = iplay.selectedBlocker;
        iplay.blockerAssignments[bid] = attacker.id;
        document.querySelectorAll(`[data-obj-id="${bid}"]`).forEach(e => {
          e.classList.remove('blocker-selected');
          e.classList.add('blocker-committed');
        });
        iplay.selectedBlocker = null;
        updateConfirmCount();
      });
    });
  });
}

// ---------------------------------------------------------------------------
// choose_target
// ---------------------------------------------------------------------------

function renderChooseTarget(pending) {
  // Inline panel: status + decline button (when applicable). Targets are
  // clicked directly on the board.
  elsI.prompt.textContent =
    'Choose a target — click a highlighted card or life total.';
  elsI.options.replaceChildren();
  if (pending.can_decline) {
    const declineBtn = document.createElement('button');
    declineBtn.type = 'button';
    declineBtn.className = 'action-btn action-btn-pass';
    declineBtn.textContent = 'Decline (no target)';
    declineBtn.addEventListener('click', () => submitAction({ decline: true }));
    elsI.options.appendChild(declineBtn);
  }

  // Highlight each legal target in-place. opt.kind identifies what to
  // search for; opt.id is interpreted by kind. Player kind targets the
  // life total; stack kind targets the stack item at that index;
  // everything else is a battlefield permanent (or anything with the
  // matching data-obj-id).
  const tagged = [];  // elements that need cleanup if user re-renders
  pending.options.forEach((opt, idx) => {
    const submit = (ev) => {
      ev.stopPropagation();
      submitAction({ idx });
    };
    if (opt.kind === 'player') {
      const life = document.querySelector(`.life[data-player-id="${opt.id}"]`);
      if (life) {
        life.classList.add('actionable-target');
        life.style.cursor = 'crosshair';
        life.title = `Target ${opt.label}`;
        life.addEventListener('click', submit);
        tagged.push(life);
      }
    } else if (opt.kind === 'stack') {
      const item = document.querySelector(`.stack-item[data-stack-idx="${opt.id}"]`);
      if (item) {
        item.classList.add('actionable-target');
        item.style.cursor = 'crosshair';
        item.title = `Target ${opt.label}`;
        item.addEventListener('click', submit);
        tagged.push(item);
      }
    } else {
      document.querySelectorAll(`.card[data-obj-id="${opt.id}"]`).forEach(el => {
        el.classList.add('actionable-target');
        el.title = `Target ${opt.label}`;
        el.addEventListener('click', submit);
        tagged.push(el);
      });
    }
  });
}

// ---------------------------------------------------------------------------
// choose_surveil
// ---------------------------------------------------------------------------

function renderChooseSurveil(pending) {
  openDecisionModal('Surveil', () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const card = pending.card || {};
    const info = document.createElement('p');
    info.className = 'decision-hint';
    info.textContent = `Top of library: ${card.name || '(unknown)'}`;
    body.appendChild(info);
    const row = document.createElement('div');
    row.className = 'decision-yes-no';
    const btnKeep = document.createElement('button');
    btnKeep.type = 'button'; btnKeep.className = 'decision-yes';
    btnKeep.textContent = 'Keep on top';
    btnKeep.addEventListener('click', () => submitAction({ mill: false }));
    const btnMill = document.createElement('button');
    btnMill.type = 'button'; btnMill.className = 'decision-no';
    btnMill.textContent = 'Mill to graveyard';
    btnMill.addEventListener('click', () => submitAction({ mill: true }));
    row.append(btnKeep, btnMill);
    body.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// choose_mode (single)
// ---------------------------------------------------------------------------

function renderChooseMode(pending) {
  openDecisionModal('Choose a Mode', () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const list = document.createElement('div');
    list.className = 'decision-list';
    (pending.modes || []).forEach(m => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'decision-list-item';
      btn.textContent = m.description;
      btn.addEventListener('click', () => submitAction({ idx: m.idx }));
      list.appendChild(btn);
    });
    body.appendChild(list);
  });
}

// ---------------------------------------------------------------------------
// choose_modes (multi-select)
// ---------------------------------------------------------------------------

function renderChooseModes(pending) {
  openDecisionModal('Choose Modes', () => {
    const selected = new Set();
    const body = elsI.modalBody;
    body.replaceChildren();
    const hint = document.createElement('p');
    hint.className = 'decision-hint';
    hint.textContent = 'Select one or more modes, then confirm.';
    body.appendChild(hint);
    (pending.modes || []).forEach(m => {
      const label = document.createElement('label');
      label.className = 'decision-checkbox';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.addEventListener('change', () => {
        if (cb.checked) selected.add(m.idx);
        else selected.delete(m.idx);
      });
      label.append(cb, ` ${m.description}`);
      body.appendChild(label);
    });
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'modal-confirm';
    btn.textContent = 'Confirm';
    btn.addEventListener('click', () => {
      const indices = Array.from(selected);
      submitAction({ indices: indices.length ? indices : [0] });
    });
    body.appendChild(btn);
  });
}

// ---------------------------------------------------------------------------
// choose_yes_no
// ---------------------------------------------------------------------------

function renderChooseYesNo(pending) {
  openDecisionModal('Choose', () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const p = document.createElement('p');
    p.className = 'decision-hint';
    p.textContent = pending.prompt || 'Yes or No?';
    body.appendChild(p);
    const row = document.createElement('div');
    row.className = 'decision-yes-no';
    const y = document.createElement('button');
    y.type = 'button'; y.className = 'decision-yes';
    y.textContent = 'Yes';
    y.addEventListener('click', () => submitAction({ yes: true }));
    const n = document.createElement('button');
    n.type = 'button'; n.className = 'decision-no';
    n.textContent = 'No';
    n.addEventListener('click', () => submitAction({ yes: false }));
    row.append(y, n);
    body.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// choose_x
// ---------------------------------------------------------------------------

function renderChooseX(pending) {
  openDecisionModal(`Choose X — ${pending.card_name || ''}`, () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const p = document.createElement('p');
    p.className = 'decision-hint';
    p.textContent = `Choose a value for X (0 – ${pending.upper_bound}).`;
    body.appendChild(p);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = 0;
    input.max = pending.upper_bound;
    input.value = pending.upper_bound;
    input.className = 'choose-x-range';
    const label = document.createElement('span');
    label.className = 'choose-x-label';
    label.textContent = `X = ${pending.upper_bound}`;
    input.addEventListener('input', () => { label.textContent = `X = ${input.value}`; });
    body.append(input, label);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'modal-confirm';
    btn.textContent = 'Confirm';
    btn.addEventListener('click', () => submitAction({ value: parseInt(input.value, 10) }));
    body.appendChild(btn);
  });
}

// ---------------------------------------------------------------------------
// choose_mana_color
// ---------------------------------------------------------------------------

function renderChooseManaColor(pending) {
  openDecisionModal('Choose Mana Color', () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const row = document.createElement('div');
    row.className = 'mana-color-row';
    (pending.options || []).forEach(color => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `mana-color-btn mana-coin ${color}`;
      btn.textContent = color === 'C' ? '◇' : color;
      btn.title = color;
      btn.addEventListener('click', () => submitAction({ color }));
      row.appendChild(btn);
    });
    body.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// choose_multi_ids (devour lands, convoke, sac-for-mana)
// ---------------------------------------------------------------------------

function renderChooseMultiIds(pending) {
  openDecisionModal(pending.prompt || 'Choose permanents', () => {
    const selected = new Set();
    const body = elsI.modalBody;
    body.replaceChildren();
    const hint = document.createElement('p');
    hint.className = 'decision-hint';
    hint.textContent = 'Select any number, then confirm.';
    body.appendChild(hint);
    const grid = document.createElement('div');
    grid.className = 'decision-card-grid';
    (pending.objects || []).forEach(opt => {
      const el = makeSelectableCard(opt, selected, opt.id);
      grid.appendChild(el);
    });
    body.appendChild(grid);
    const row = document.createElement('div');
    row.className = 'decision-yes-no';
    const confirm = document.createElement('button');
    confirm.type = 'button'; confirm.className = 'modal-confirm';
    confirm.textContent = 'Confirm';
    confirm.addEventListener('click', () => submitAction({ ids: Array.from(selected) }));
    const skip = document.createElement('button');
    skip.type = 'button'; skip.className = 'decision-no';
    skip.textContent = 'None / Skip';
    skip.addEventListener('click', () => submitAction({ ids: [] }));
    row.append(confirm, skip);
    body.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// choose_discard: highlight the player's hand cards and let them click one
// ---------------------------------------------------------------------------

function renderChooseDiscardBoard(pending) {
  elsI.prompt.textContent = 'Discard a card — click a card in your hand.';
  elsI.options.replaceChildren();

  (pending.cards || []).forEach(card => {
    document.querySelectorAll(`.hand .card[data-obj-id="${card.id}"]`).forEach(el => {
      el.classList.add('actionable');
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        submitAction({ id: card.id });
      });
    });
  });
}

// ---------------------------------------------------------------------------
// choose_card: list of named cards to click
// ---------------------------------------------------------------------------

function renderChooseCard(pending, title, onPick) {
  openDecisionModal(title, () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const grid = document.createElement('div');
    grid.className = 'decision-card-grid';
    (pending.cards || []).forEach(c => {
      const el = document.createElement('div');
      el.className = 'decision-card-option';
      if (c.scryfall_id) {
        el.style.backgroundImage = `url('/card_image/${c.scryfall_id}/small')`;
        el.classList.add('has-art');
      } else {
        const nm = document.createElement('span');
        nm.className = 'option-name';
        nm.textContent = c.name;
        el.appendChild(nm);
      }
      el.title = c.name;
      el.addEventListener('click', () => onPick(c.id, false));
      grid.appendChild(el);
    });
    body.appendChild(grid);
    if (pending.can_decline) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'decision-no';
      btn.textContent = 'Decline';
      btn.addEventListener('click', () => onPick(null, true));
      body.appendChild(btn);
    }
  });
}

// ---------------------------------------------------------------------------
// choose_sacrifice / choose_one_id: list of objects, click one
// ---------------------------------------------------------------------------

function renderChooseObj(pending, title, onPick) {
  openDecisionModal(title, () => {
    const body = elsI.modalBody;
    body.replaceChildren();
    const list = document.createElement('div');
    list.className = 'decision-list';
    (pending.objects || []).forEach(o => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'decision-list-item';
      btn.textContent = o.name;
      btn.addEventListener('click', () => onPick(o.id));
      list.appendChild(btn);
    });
    body.appendChild(list);
  });
}

// ---------------------------------------------------------------------------
// Decision modal helpers
// ---------------------------------------------------------------------------

function openDecisionModal(title, buildFn) {
  elsI.modalTitle.textContent = title;
  elsI.modalBody.replaceChildren();
  buildFn();
  elsI.modal.hidden = false;
  elsI.modal.classList.remove('hidden');
}

function closeDecisionModal() {
  elsI.modal.hidden = true;
  elsI.modal.classList.add('hidden');
}

elsI.modal.addEventListener('click', (ev) => {
  if (ev.target === elsI.modal) closeDecisionModal();
});

// ---------------------------------------------------------------------------
// makeSelectableCard: a toggleable card element for multi-select decisions
// ---------------------------------------------------------------------------

function makeSelectableCard(opt, selectedSet, key) {
  const el = document.createElement('div');
  el.className = 'decision-card-option selectable';
  if (opt.scryfall_id) {
    el.style.backgroundImage = `url('/card_image/${opt.scryfall_id}/small')`;
    el.classList.add('has-art');
  } else {
    const nm = document.createElement('span');
    nm.className = 'option-name';
    nm.textContent = opt.name;
    el.appendChild(nm);
  }
  el.title = opt.name;
  el.addEventListener('click', () => {
    if (selectedSet.has(key)) {
      selectedSet.delete(key);
      el.classList.remove('selected');
    } else {
      selectedSet.add(key);
      el.classList.add('selected');
    }
  });
  return el;
}

// ---------------------------------------------------------------------------
// Interactive log renderer (uses snap.log directly, no full-log index)
// ---------------------------------------------------------------------------

function renderInteractiveLog(snap) {
  const body = els.logBody;
  const entries = snap.log || [];
  const recentCount = (snap.log_since_last || []).length;
  const firstRecent = entries.length - recentCount;
  body.replaceChildren();
  entries.forEach((e, i) => {
    const d = document.createElement('div');
    d.className = 'log-entry';
    if (recentCount > 0 && i >= firstRecent) d.classList.add('this-step');
    renderLogEntryText(d, e);
    body.appendChild(d);
  });
  body.scrollTop = body.scrollHeight;
}

// ---------------------------------------------------------------------------
// postJSON helper
// ---------------------------------------------------------------------------

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${text}`);
  }
  return r.json();
}
