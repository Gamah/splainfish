/**
 * check_features_parity.mjs — Diff web/js/nnue/features.js against the real
 * splainfish/features.py. Reads ref_features.py's JSON on stdin.
 *
 * Run via `make test-parity`.
 */
import { Chess } from '../vendor/chess.js/chess.js';
import {
  WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
  boardFromChessJs, computeFeatures, diffFeatures, halfkaLabel, halfkaIndex,
  threatIndices,
} from '../web/js/nnue/features.js';

const raw = await new Promise((resolve) => {
  let buf = '';
  process.stdin.on('data', (d) => (buf += d));
  process.stdin.on('end', () => resolve(buf));
});
const ref = JSON.parse(raw);

let failures = 0;
const MAX_REPORT = 8;
function fail(msg) {
  failures++;
  if (failures <= MAX_REPORT) console.log('  FAIL: ' + msg);
  else if (failures === MAX_REPORT + 1) console.log('  ... (further failures suppressed)');
}

const eqArr = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const colorOf = (s) => (s === 'white' ? WHITE : BLACK);

function featuresFor(fen) {
  const game = new Chess(fen);
  return computeFeatures(boardFromChessJs(game));
}

// --- 1. halfka_index arithmetic ---------------------------------------------
{
  let n = 0;
  for (const c of ref.index_arithmetic) {
    const got = halfkaIndex(
      colorOf(c.perspective), colorOf(c.piece_color),
      c.piece_type, c.piece_sq, c.king_sq,
    );
    if (got !== c.index) {
      fail(`halfkaIndex(${c.perspective},${c.piece_color},pt=${c.piece_type},` +
           `psq=${c.piece_sq},ksq=${c.king_sq}) py=${c.index} js=${got}`);
    }
    n++;
  }
  console.log(`halfka_index:      ${n} argument combinations compared`);
}

// --- 2. threat tables --------------------------------------------------------
{
  // Rebuilt here from the module's exported behaviour via threatIndices on a
  // synthetic board would be indirect; instead check the tables the Python
  // exposes directly against a fresh computation of the same quantities.
  let n = 0;
  const NUM_TARGETS = { 1: 2, 2: 5, 3: 4, 4: 4, 5: 5 };
  const ATTACKERS = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN];
  for (const pt of ATTACKERS) {
    let offset = 0;
    for (const p of ATTACKERS) {
      if (p === pt) break;
      offset += NUM_TARGETS[p] * 64;
    }
    if (offset !== ref.threat_tables.base_offsets[String(pt)]) {
      fail(`threat base offset pt=${pt} py=${ref.threat_tables.base_offsets[String(pt)]} js=${offset}`);
    }
    n++;
  }
  for (const [name, table] of Object.entries(ref.threat_tables.orient)) {
    for (let sq = 0; sq < 64; sq++) {
      const got = name === 'black' ? sq ^ 56 : sq;
      if (got !== table[sq]) fail(`threat orient ${name} sq=${sq} py=${table[sq]} js=${got}`);
      n++;
    }
  }
  console.log(`threat tables:     ${n} entries compared`);
}

// --- 3. labels ---------------------------------------------------------------
{
  let n = 0;
  for (const c of ref.labels) {
    const got = halfkaLabel(c.idx, colorOf(c.perspective));
    const want = c.label;
    const mismatch =
      got.kingBucket !== want.king_bucket ||
      got.pieceColor !== want.piece_color ||
      got.pieceType !== want.piece_type ||
      got.pieceSq !== want.piece_sq ||
      got.perspective !== want.perspective;
    if (mismatch) {
      fail(`halfkaLabel(${c.idx}, ${c.perspective})\n` +
           `    py=${JSON.stringify(want)}\n    js=${JSON.stringify(got)}`);
    }
    n++;
  }
  console.log(`halfka_label:      ${n} decodings compared`);
}

// --- 4. full feature sets + diffs over random playouts -----------------------
{
  let nPos = 0, nIdx = 0;
  for (const p of ref.positions) {
    for (const which of ['before', 'after']) {
      const fen = which === 'before' ? p.fen_before : p.fen_after;
      const want = which === 'before' ? p.features_before : p.features_after;
      const got = featuresFor(fen);

      const checks = [
        ['halfka_white', got.halfkaWhite, want.halfka_white],
        ['halfka_black', got.halfkaBlack, want.halfka_black],
        ['threat_white', got.threatWhite, want.threat_white],
        ['threat_black', got.threatBlack, want.threat_black],
      ];
      for (const [name, g, w] of checks) {
        // Python builds threat lists in piece-iteration order; compare as
        // multisets since the accumulator sums them and order is irrelevant.
        const gs = [...g].sort((a, b) => a - b);
        const ws = [...w].sort((a, b) => a - b);
        if (!eqArr(gs, ws)) {
          fail(`${which} ${name} @ ${fen}\n    py(${ws.length})=${ws.slice(0, 12)}...\n` +
               `    js(${gs.length})=${gs.slice(0, 12)}...`);
        }
        nIdx += ws.length;
      }
      if (got.wkingSq !== want.wking_sq) fail(`${which} wking @ ${fen}`);
      if (got.bkingSq !== want.bking_sq) fail(`${which} bking @ ${fen}`);
    }

    // diff
    const fb = featuresFor(p.fen_before);
    const fa = featuresFor(p.fen_after);
    const d = diffFeatures(fb, fa);
    const dchecks = [
      ['halfka_white_gained', d.halfkaWhiteGained], ['halfka_white_lost', d.halfkaWhiteLost],
      ['halfka_black_gained', d.halfkaBlackGained], ['halfka_black_lost', d.halfkaBlackLost],
      ['threat_white_gained', d.threatWhiteGained], ['threat_white_lost', d.threatWhiteLost],
      ['threat_black_gained', d.threatBlackGained], ['threat_black_lost', d.threatBlackLost],
    ];
    for (const [name, g] of dchecks) {
      const w = p.diff[name];
      if (!eqArr(g, w)) {
        fail(`diff ${name} @ ${p.fen_before} (${p.move_uci})\n` +
             `    py=${JSON.stringify(w)}\n    js=${JSON.stringify(g)}`);
      }
    }
    nPos++;
  }
  console.log(`positions:         ${nPos} playout positions, ${nIdx} feature indices compared`);
}

console.log(failures === 0 ? '\nAll features parity checks PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
