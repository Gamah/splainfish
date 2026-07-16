/**
 * app.js — Browser controller for splainfish.
 *
 * Flow: paste PGN -> load Stockfish (from CDN) + NNUE net (same-origin) behind a
 * progress bar -> analyse every move -> interactive board + explanations.
 *
 * chessground and chess.js are pulled from a CDN as ES modules; Stockfish and
 * the NNUE net are loaded lazily on first analyse.
 */

import { Chessground } from 'https://esm.sh/chessground@9.2.1';
import { Chess } from 'https://esm.sh/chess.js@1.4.0';

import { StockfishEngine } from './engine.js';
import { loadNnue } from './nnue/loader.js';
import { analyseGame } from './pipeline.js';

// ---------------------------------------------------------------------------
// Element refs
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const els = {
  paste: $('paste-view'),
  pgn: $('pgn-input'),
  depth: $('depth-input'),
  depthVal: $('depth-val'),
  onlyMistakes: $('only-mistakes'),
  analyseBtn: $('analyse-btn'),
  demoBtn: $('demo-btn'),
  pasteError: $('paste-error'),

  loading: $('loading-view'),
  loadingLabel: $('loading-label'),
  loadingBar: $('loading-bar'),
  loadingPct: $('loading-pct'),

  analysis: $('analysis-view'),
  board: $('board'),
  moveList: $('move-list'),
  moveLabel: $('move-label'),
  evalFill: $('eval-fill'),
  evalNum: $('eval-num'),
  explain: $('explain-panel'),
  btnPrev: $('btn-prev'),
  btnNext: $('btn-next'),
  newGameBtn: $('new-game-btn'),

  about: $('about-modal'),
  aboutOpen: $('about-open'),
  aboutClose: $('about-close'),
};

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

let engine = null;   // StockfishEngine, loaded once
let weights = null;  // parsed NNUE, loaded once
let moves = [];      // analysis records
let currentIdx = -1;
let ground = null;   // chessground instance
let viewMode = localStorage.getItem('sf_viewMode') || 'simple';

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------

function show(view) {
  for (const v of [els.paste, els.loading, els.analysis]) v.hidden = true;
  view.hidden = false;
}

function setProgress(frac, label) {
  els.loadingLabel.textContent = label;
  if (frac === null || frac === undefined) {
    els.loadingBar.style.width = '100%';
    els.loadingBar.classList.add('indeterminate');
    els.loadingPct.textContent = '';
  } else {
    els.loadingBar.classList.remove('indeterminate');
    const pct = Math.round(frac * 100);
    els.loadingBar.style.width = pct + '%';
    els.loadingPct.textContent = pct + '%';
  }
}

// ---------------------------------------------------------------------------
// Analyse flow
// ---------------------------------------------------------------------------

async function ensureEngineAndNet() {
  if (!engine) {
    engine = new StockfishEngine();
    await engine.load((frac, label) => {
      setProgress(frac, label === 'network'
        ? 'pulling StockFish source…'
        : 'pulling StockFish source…');
    });
  }
  if (!weights) {
    weights = await loadNnue((frac) => setProgress(frac, 'loading NNUE network…'));
  }
}

async function runAnalysis(pgn) {
  els.pasteError.textContent = '';
  show(els.loading);
  setProgress(0, 'starting…');

  try {
    await ensureEngineAndNet();

    const depth = Number(els.depth.value);
    const onlyMistakes = els.onlyMistakes.checked;

    setProgress(null, 'analysing moves…');
    moves = await analyseGame({
      Chess, engine, weights, pgn, depth, onlyMistakes,
      onMove: (i, total, san) => {
        setProgress(i / total, `analysing move ${i + 1}/${total}: ${san}`);
      },
    });

    if (!moves.length) {
      throw new Error('No moves to show (try disabling "only mistakes").');
    }
    startAnalysisView();
  } catch (err) {
    show(els.paste);
    els.pasteError.textContent = err.message;
  }
}

// ---------------------------------------------------------------------------
// Analysis view
// ---------------------------------------------------------------------------

function startAnalysisView() {
  show(els.analysis);
  if (!ground) {
    ground = Chessground(els.board, {
      viewOnly: true,
      coordinates: true,
      animation: { enabled: true, duration: 200 },
    });
  }
  buildMoveList();
  const firstBad = moves.findIndex((m) =>
    ['inaccuracy', 'mistake', 'blunder'].includes(m.quality));
  selectMove(firstBad >= 0 ? firstBad : 0);
}

function buildMoveList() {
  let html = '';
  let pairOpen = false;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    if (m.color === 'white') {
      if (pairOpen) html += '</div>';
      html += `<div class="move-pair"><span class="move-num">${m.move_number}.</span>`;
      pairOpen = true;
    }
    const glyph = m.quality_glyph || '';
    html += `<button class="move-btn" id="mb${i}" data-idx="${i}">` +
            `<span class="dot" style="background:${m.quality_color}"></span>` +
            `<span>${m.move_san}${glyph}</span></button>`;
    if (m.color === 'black' || i === moves.length - 1) {
      if (m.color === 'white') html += '<span></span>';
      html += '</div>';
      pairOpen = false;
    }
  }
  els.moveList.innerHTML = html;
  els.moveList.querySelectorAll('.move-btn').forEach((b) =>
    b.addEventListener('click', () => selectMove(Number(b.dataset.idx))));
}

function uciSquares(uci) {
  if (!uci || uci.length < 4) return [null, null];
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

function updateEvalBar(cp) {
  const clamped = Math.max(-600, Math.min(600, cp));
  const pct = (clamped / 1200) * 50;
  if (cp >= 0) {
    els.evalFill.style.left = '50%';
    els.evalFill.style.width = pct + '%';
    els.evalFill.style.background = 'var(--sf-white)';
  } else {
    els.evalFill.style.left = (50 + pct) + '%';
    els.evalFill.style.width = (-pct) + '%';
    els.evalFill.style.background = 'var(--sf-black)';
  }
  const pawns = Math.abs(cp) / 100;
  els.evalNum.textContent = (cp >= 0 ? '+' : '−') + pawns.toFixed(2);
}

function selectMove(idx) {
  if (idx < 0 || idx >= moves.length) return;
  currentIdx = idx;
  const m = moves[idx];

  els.moveList.querySelectorAll('.move-btn').forEach((b) => b.classList.remove('active'));
  const btn = $('mb' + idx);
  if (btn) { btn.classList.add('active'); btn.scrollIntoView({ block: 'nearest' }); }

  const [from, to] = uciSquares(m.move_uci);
  ground.set({
    fen: m.fen_after.split(' ')[0],
    lastMove: from && to ? [from, to] : undefined,
  });

  const dots = m.color === 'black' ? '…' : '.';
  els.moveLabel.textContent = `${m.move_number}${dots} ${m.move_san}${m.quality_glyph || ''}`;
  updateEvalBar(m.sf_eval_after ?? m.eval_after_cp ?? 0);
  renderExplanation(m);

  els.btnPrev.disabled = idx === 0;
  els.btnNext.disabled = idx === moves.length - 1;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function renderExplanation(m) {
  let attrHtml = '';
  if (m.complex_groups && m.complex_groups.length) {
    const maxPct = Math.max(...m.complex_groups.map((g) => g.pct_of_total));
    for (const g of m.complex_groups) {
      const barW = maxPct > 0 ? (g.pct_of_total / maxPct) * 100 : 0;
      const barColor = g.direction === 'positive' ? 'var(--sf-green)' : 'var(--sf-red)';
      attrHtml += `
        <div class="attr-group">
          <div class="attr-header">
            <span class="attr-name">${esc(g.group)}</span>
            <span class="attr-cp">${g.pct_of_total}%</span>
          </div>
          <div class="attr-bar-bg"><div class="attr-bar-fill" style="width:${barW}%;background:${barColor}"></div></div>
          <div class="attr-sentence">${esc(g.sentence)}</div>
        </div>`;
    }
    attrHtml += `<div class="complex-note">${esc(m.complex_note || '')}</div>`;
  } else {
    attrHtml = '<div class="explain-para">No attribution data for this move.</div>';
  }

  const simpleParas = (m.simple_paragraphs || [])
    .map((p) => `<div class="explain-para">${esc(p)}</div>`).join('');

  const isSimple = viewMode === 'simple';
  els.explain.innerHTML = `
    <div class="quality-row">
      <span class="quality-badge" style="background:${m.quality_color}22;color:${m.quality_color};border:1px solid ${m.quality_color}55">
        ${esc(m.quality_label)}
      </span>
    </div>
    <div class="explain-headline">${esc(m.simple_headline || m.move_san)}</div>
    <div class="view-toggle">
      <button class="toggle-btn ${isSimple ? 'active' : ''}" data-mode="simple">Simple</button>
      <button class="toggle-btn ${!isSimple ? 'active' : ''}" data-mode="complex">Detailed</button>
    </div>
    <div class="view-body view-simple" ${isSimple ? '' : 'hidden'}>
      ${simpleParas || '<div class="explain-para">No explanation available.</div>'}
    </div>
    <div class="view-body view-complex" ${isSimple ? 'hidden' : ''}>${attrHtml}</div>`;

  els.explain.querySelectorAll('.toggle-btn').forEach((b) =>
    b.addEventListener('click', () => setView(b.dataset.mode)));
}

function setView(mode) {
  viewMode = mode;
  localStorage.setItem('sf_viewMode', mode);
  els.explain.querySelectorAll('.toggle-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.mode === mode));
  const simple = els.explain.querySelector('.view-simple');
  const complex = els.explain.querySelector('.view-complex');
  if (simple) simple.hidden = mode !== 'simple';
  if (complex) complex.hidden = mode !== 'complex';
}

function navigate(dir) { selectMove(currentIdx + dir); }

// ---------------------------------------------------------------------------
// Demo game (the Immortal Game)
// ---------------------------------------------------------------------------

const DEMO_PGN = `[Event "Casual game"]
[Site "London"]
[Date "1851.06.21"]
[White "Anderssen, Adolf"]
[Black "Kieseritzky, Lionel"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5
8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3
Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2
Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0`;

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

els.depth.addEventListener('input', () => { els.depthVal.textContent = els.depth.value; });
els.analyseBtn.addEventListener('click', () => {
  const pgn = els.pgn.value.trim();
  if (!pgn) { els.pasteError.textContent = 'Paste a PGN first.'; return; }
  runAnalysis(pgn);
});
els.demoBtn.addEventListener('click', () => { els.pgn.value = DEMO_PGN; runAnalysis(DEMO_PGN); });
els.newGameBtn.addEventListener('click', () => show(els.paste));
els.btnPrev.addEventListener('click', () => navigate(-1));
els.btnNext.addEventListener('click', () => navigate(1));
document.addEventListener('keydown', (e) => {
  if (els.analysis.hidden) return;
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});

els.aboutOpen.addEventListener('click', (e) => { e.preventDefault(); els.about.hidden = false; });
els.aboutClose.addEventListener('click', () => { els.about.hidden = true; });
els.about.addEventListener('click', (e) => { if (e.target === els.about) els.about.hidden = true; });

els.depthVal.textContent = els.depth.value;
show(els.paste);
