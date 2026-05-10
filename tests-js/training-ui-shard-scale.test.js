// Tests for the per-shard "scale workers" controls inside the
// training UI's per-machine sub-cards.
//
// Each row in the .machines slot of a data_gen / eval / training card
// now ends in a small editable numeric input + Apply button. Clicking
// Apply POSTs to the master's mtg_bot serve endpoint
// (POST /api/shards/<host>/workers, body { n_workers: N }) and
// surfaces a "pending" hint until the next iter's progress JSON shows
// the new n_workers value live. This file exercises:
//
// - The control renders with the current n_workers prefilled.
// - Apply triggers a POST to the right URL with the right body.
// - A successful POST stores a pendingScales entry; the next render
//   shows the pending hint with the new value.
// - The pending hint clears once the polled run shows the new
//   n_workers (i.e. the next iter has started).
// - A failed POST surfaces an error hint.
// - Apply is disabled while a POST is in flight (no double-fire).
// - Out-of-range / non-integer inputs are rejected client-side
//   without firing a POST.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeEach, vi } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TPL_PATH = path.join(
  __dirname, '..', 'src', 'mtg_bot', 'training_ui',
  'templates', 'index.html',
);

function extractInlineScript(html) {
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
  for (const m of scripts) {
    const attrs = m[1] || '';
    if (!/\bsrc\s*=/.test(attrs)) return m[2];
  }
  throw new Error('inline <script> not found in training UI template');
}

const TEMPLATE_HTML = fs.readFileSync(TPL_PATH, 'utf8');
const INLINE_JS = extractInlineScript(TEMPLATE_HTML);

beforeEach(() => {
  document.documentElement.innerHTML =
    '<head><meta charset="utf-8"></head><body>'
    + '<div class="page">'
    + '<header><span id="refresh-indicator">loading…</span></header>'
    + '<div id="active"></div>'
    + '<div id="older"></div>'
    + '</div>'
    + '<div class="modal-overlay" id="stop-modal">'
    + '<div class="modal">'
    + '<h2 id="stop-modal-title">Stop run?</h2>'
    + '<p id="stop-modal-body"></p>'
    + '<div class="modal-actions">'
    + '<button id="stop-modal-cancel">Cancel</button>'
    + '<button id="stop-modal-confirm">Stop run</button>'
    + '</div></div></div>'
    + '</body>';

  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ runs: [] }),
    text: async () => '',
  });
  globalThis.setInterval = vi.fn(() => 0);
  globalThis.Chart = vi.fn(function () {
    return { update: () => {}, destroy: () => {}, data: {} };
  });

  // Set a deterministic location so serveBase()/mainUiBase() resolve.
  // jsdom's default is http://localhost/.
  Object.defineProperty(window, 'location', {
    writable: true,
    value: new URL('http://desktop-j05412i:5003/'),
  });

  (0, eval)(INLINE_JS);
});

function basePayload(overrides = {}) {
  return {
    run_id: 'iter_007_self_play_v9',
    kind: 'data_gen',
    status: 'running',
    started_at: '2026-05-04T12:00:00+00:00',
    n_games_total: 500,
    n_games_done: 100,
    n_workers: 30,
    n_samples: 12000,
    p0_wins: 60,
    p1_wins: 40,
    draws: 0,
    crashed: 2,
    elapsed_seconds: 605.0,
    estimated_remaining_seconds: 2000.0,
    games_per_hour: 600,
    samples_per_hour: 72000,
    machines: [
      {
        label: 'local',
        status: 'running',
        n_games_done: 60,
        n_workers: 8,
        crashed: 0,
        timed_out: 0,
        elapsed_seconds: 600.0,
        per_game_seconds: 10.0,
      },
      {
        label: 'win10',
        status: 'running',
        n_games_done: 40,
        n_workers: 4,
        crashed: 0,
        timed_out: 0,
        elapsed_seconds: 540.0,
        per_game_seconds: 13.5,
      },
    ],
    ...overrides,
  };
}

function rowFor(card, host) {
  const rows = card.querySelectorAll('.machine-row');
  return Array.from(rows).find(
    (r) => r.querySelector('.machine-label').textContent === host,
  );
}

describe('training UI per-shard scale workers control', () => {
  it('renders an input prefilled with the current n_workers', () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    const localRow = rowFor(card, 'local');
    const input = localRow.querySelector(
      '.machine-scale [data-field="scale-input"]',
    );
    expect(input).not.toBeNull();
    expect(input.type).toBe('number');
    expect(input.min).toBe('1');
    expect(input.max).toBe('64');
    expect(input.value).toBe('8');

    const win10Row = rowFor(card, 'win10');
    const input2 = win10Row.querySelector(
      '.machine-scale [data-field="scale-input"]',
    );
    expect(input2.value).toBe('4');
  });

  it('Apply POSTs to /api/shards/<host>/workers on port 5099', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    // Stub fetch with a controllable resolve so we can assert on
    // the request shape AND the post-resolution state.
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: true, host: 'local', n_workers: 12 }),
    });
    globalThis.fetch = fetchMock;

    const localRow = rowFor(card, 'local');
    const input = localRow.querySelector('[data-field="scale-input"]');
    const btn = localRow.querySelector('[data-field="scale-apply"]');
    input.value = '12';
    btn.click();

    // Let the async chain inside applyShardScale resolve.
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    // The first fetch call should be the POST; subsequent calls (from
    // refresh()) hit /api/runs and are unrelated.
    const postCall = fetchMock.mock.calls.find(
      (c) => String(c[0]).includes('/api/shards/'),
    );
    expect(postCall).toBeDefined();
    expect(postCall[0]).toBe(
      'http://desktop-j05412i:5099/api/shards/local/workers',
    );
    expect(postCall[1].method).toBe('POST');
    const body = JSON.parse(postCall[1].body);
    expect(body).toEqual({ n_workers: 12 });
  });

  it('shows a "pending" hint after a successful POST', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: true, host: 'local', n_workers: 12 }),
    });

    const localRow = rowFor(card, 'local');
    const input = localRow.querySelector('[data-field="scale-input"]');
    const btn = localRow.querySelector('[data-field="scale-apply"]');
    input.value = '12';
    btn.click();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    // Re-render the card with the SAME payload (the in-flight iter
    // still reports n_workers=8) — pending hint should show.
    fillRun(card, run);
    const localRow2 = rowFor(card, 'local');
    const hint = localRow2.querySelector('.machine-scale-hint');
    expect(hint.classList.contains('pending')).toBe(true);
    expect(hint.textContent).toContain('12');
    expect(hint.textContent).toContain('next iter');
  });

  it('clears the pending hint once the live n_workers matches', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ ok: true, host: 'local', n_workers: 12 }),
    });

    const localRow = rowFor(card, 'local');
    localRow.querySelector('[data-field="scale-input"]').value = '12';
    localRow.querySelector('[data-field="scale-apply"]').click();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    // Now simulate the next iter starting: a fresh run payload that
    // reports n_workers=12 for local. The pending hint should clear.
    const next = basePayload();
    next.machines[0] = { ...next.machines[0], n_workers: 12 };
    fillRun(card, next);
    const localRowNext = rowFor(card, 'local');
    const hint = localRowNext.querySelector('.machine-scale-hint');
    expect(hint.classList.contains('pending')).toBe(false);
    expect(hint.classList.contains('error')).toBe(false);
    expect(hint.textContent).toBe('');
  });

  it('shows an error hint when the POST fails', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false, status: 400,
      json: async () => ({ error: 'unknown host' }),
    });

    const localRow = rowFor(card, 'local');
    localRow.querySelector('[data-field="scale-input"]').value = '12';
    localRow.querySelector('[data-field="scale-apply"]').click();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    fillRun(card, run);
    const hint = rowFor(card, 'local').querySelector('.machine-scale-hint');
    expect(hint.classList.contains('error')).toBe(true);
    expect(hint.textContent).toContain('unknown host');
  });

  it('shows an error hint on a network exception', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    globalThis.fetch = vi.fn().mockRejectedValue(new Error('offline'));

    const localRow = rowFor(card, 'local');
    localRow.querySelector('[data-field="scale-input"]').value = '12';
    localRow.querySelector('[data-field="scale-apply"]').click();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    fillRun(card, run);
    const hint = rowFor(card, 'local').querySelector('.machine-scale-hint');
    expect(hint.classList.contains('error')).toBe(true);
    expect(hint.textContent).toContain('network');
  });

  it('rejects out-of-range values client-side without POSTing', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ runs: [] }),
    });
    globalThis.fetch = fetchMock;

    const localRow = rowFor(card, 'local');
    localRow.querySelector('[data-field="scale-input"]').value = '0';
    localRow.querySelector('[data-field="scale-apply"]').click();
    await new Promise((r) => setTimeout(r, 0));

    // No POST fired.
    const postCall = fetchMock.mock.calls.find(
      (c) => String(c[0]).includes('/api/shards/'),
    );
    expect(postCall).toBeUndefined();

    fillRun(card, run);
    const hint = rowFor(card, 'local').querySelector('.machine-scale-hint');
    expect(hint.classList.contains('error')).toBe(true);
    expect(hint.textContent).toContain('1-64');
  });

  it('rejects non-integer values client-side', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ runs: [] }),
    });
    globalThis.fetch = fetchMock;

    const localRow = rowFor(card, 'local');
    localRow.querySelector('[data-field="scale-input"]').value = '8.5';
    localRow.querySelector('[data-field="scale-apply"]').click();
    await new Promise((r) => setTimeout(r, 0));

    const postCall = fetchMock.mock.calls.find(
      (c) => String(c[0]).includes('/api/shards/'),
    );
    expect(postCall).toBeUndefined();
  });

  it('disables Apply while a POST is in flight (no double-fire)', async () => {
    const run = basePayload();
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);

    // Hand-rolled deferred so we can hold the request open until we
    // assert the disabled state.
    let resolveFetch;
    const pending = new Promise((r) => { resolveFetch = r; });
    const fetchMock = vi.fn().mockReturnValue(pending);
    globalThis.fetch = fetchMock;

    const localRow = rowFor(card, 'local');
    const input = localRow.querySelector('[data-field="scale-input"]');
    const btn = localRow.querySelector('[data-field="scale-apply"]');
    input.value = '12';
    btn.click();

    // After the click, the in-flight Set has the host so subsequent
    // double-clicks early-return inside applyShardScale. Synchronously
    // verify the button is disabled.
    expect(btn.disabled).toBe(true);

    // Click again — should NOT fire a second POST.
    btn.click();
    await new Promise((r) => setTimeout(r, 0));
    const postCalls = fetchMock.mock.calls.filter(
      (c) => String(c[0]).includes('/api/shards/'),
    );
    expect(postCalls.length).toBe(1);

    // Resolve the held POST so test cleanup completes.
    resolveFetch({
      ok: true, status: 200,
      json: async () => ({ ok: true, host: 'local', n_workers: 12 }),
    });
    await new Promise((r) => setTimeout(r, 0));
  });

  it('renders the scale control on eval-kind cards too', () => {
    const run = basePayload({
      kind: 'eval',
      wins_new: 60, wins_old: 40, draws: 0,
      win_rate: 0.6, opponent_label: 'v8',
    });
    const card = ensureCard(run);
    document.getElementById('active').appendChild(card);
    fillRun(card, run);
    const localRow = rowFor(card, 'local');
    expect(
      localRow.querySelector('.machine-scale [data-field="scale-input"]'),
    ).not.toBeNull();
    expect(
      localRow.querySelector('.machine-scale [data-field="scale-apply"]'),
    ).not.toBeNull();
  });
});
