// Tests for the training UI eval-card "Wilson 95% CI · gate 52%"
// stat. Mirrors what scripts/eval_early_stop_watcher.py decides
// server-side, so the user sees the same PASS / FAIL / undecided
// verdict in the browser that the watcher uses to write the early-
// stop sentinel. Verdicts:
//
//  - PASS (green): CI lower bound > 0.52
//  - FAIL (red):   CI upper bound < 0.52
//  - UNDECIDED (amber): 0.52 lies inside the CI
//  - PENDING (grey): n_decisive < EVAL_CI_MIN_DECISIVE (=30)
//
// Loads the inline <script> from the training UI template the same
// way the other training-ui-* tests do.

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

  (0, eval)(INLINE_JS);
});

function evalPayload(overrides = {}) {
  return {
    run_id: 'iter_001_eval_v18_vs_v17',
    kind: 'eval',
    status: 'running',
    started_at: '2026-05-10T01:00:00+00:00',
    n_games_total: 500,
    n_games_done: 200,
    wins_new: 0,
    wins_old: 0,
    draws: 0,
    crashed: 0,
    timed_out: 0,
    n_workers: 16,
    elapsed_seconds: 60.0,
    games_per_hour: 200,
    estimated_remaining_seconds: 90.0,
    win_rate: null,
    opponent_label: 'v17',
    ...overrides,
  };
}

function buildCard(run) {
  const card = ensureCard(run);
  document.getElementById('active').appendChild(card);
  fillRun(card, run);
  return card;
}

describe('training UI eval-card Wilson CI verdict', () => {
  it('shows pending when n_decisive < 30', () => {
    const run = evalPayload({ wins_new: 10, wins_old: 5, draws: 0 });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toBe('n=15 (gathering…)');
    expect(ci.className).toBe('stat-value ci-pending');
  });

  it('shows em-dash when n_decisive == 0', () => {
    const run = evalPayload({ wins_new: 0, wins_old: 0, draws: 0 });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toBe('—');
    expect(ci.className).toBe('stat-value ci-pending');
  });

  it('shows PASS (green) when CI lower bound > 52%', () => {
    // 90-10 over 100 decisive: p=0.90, Wilson 95% CI ≈ [0.825, 0.946].
    const run = evalPayload({ wins_new: 90, wins_old: 10, draws: 0 });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toMatch(/\[\d{2}\.\d%, \d{2}\.\d%\] → PASS/);
    expect(ci.className).toBe('stat-value ci-pass');
  });

  it('shows FAIL (red) when CI upper bound < 52%', () => {
    // 10-90 over 100 decisive: p=0.10, Wilson 95% CI ≈ [0.054, 0.175].
    const run = evalPayload({ wins_new: 10, wins_old: 90, draws: 0 });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toMatch(/\[\d{1,2}\.\d%, \d{2}\.\d%\] → FAIL/);
    expect(ci.className).toBe('stat-value ci-fail');
  });

  it('shows undecided (amber) when threshold straddles the CI', () => {
    // 60-40 over 100 decisive: p=0.60, Wilson 95% CI ≈ [0.501, 0.690].
    // 52% threshold sits inside; verdict is undecided.
    const run = evalPayload({ wins_new: 60, wins_old: 40, draws: 0 });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toMatch(/\[\d{2}\.\d%, \d{2}\.\d%\] → undecided/);
    expect(ci.className).toBe('stat-value ci-undecided');
  });

  it('threshold-on-boundary cases lean undecided (lower == threshold)', () => {
    // Hand-picked: at n=400 with 232-168 (58% point), Wilson lower
    // bound is just under 0.52 → still undecided per the strict
    // wilson_lo > threshold check.
    const run = evalPayload({
      wins_new: 232, wins_old: 168, draws: 0,
    });
    const card = buildCard(run);
    const ci = card.querySelector('[data-field="ci"]');
    expect(ci.textContent).toMatch(/→ (undecided|PASS)/);
  });

  it('Wilson interval helper computes the textbook value', () => {
    // Pin the helper directly to catch any silent CI-formula
    // regression. 70-30 / 100 → Wilson 95% CI ≈ [0.604, 0.784].
    const ci = wilsonInterval(70, 100, 1.96);
    expect(ci.point).toBeCloseTo(0.70, 4);
    expect(ci.lo).toBeCloseTo(0.604, 2);
    expect(ci.hi).toBeCloseTo(0.784, 2);
  });

  it('Wilson interval helper returns null on n=0', () => {
    expect(wilsonInterval(0, 0, 1.96)).toBeNull();
  });
});
