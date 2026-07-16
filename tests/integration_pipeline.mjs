/**
 * integration_pipeline.mjs — Run the whole browser analysis pipeline in Node.
 *
 * Drives the real web/js/pipeline.js with a real Stockfish (the lite WASM under
 * tests/vendor-sf, driven via the npm loader) and the real committed NNUE net.
 * The only thing swapped out is the browser StockfishEngine (fetch + Worker):
 * a Node adapter here exposes the same analyse() interface, so pipeline.js,
 * probe.js, features.js, and explain.js all run exactly as they do in the app.
 *
 * This is the end-to-end smoke test for the browser tool minus the DOM.
 * Requires the lite engine staged under tests/vendor-sf (see Makefile
 * test-integration, which stages it and is skipped if absent).
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { Chess } from '../vendor/chess.js/chess.js';
import { parseNnue } from '../web/js/nnue/parser.js';
import { analyseGame } from '../web/js/pipeline.js';

const HERE = new URL('.', import.meta.url).pathname;
const require = createRequire(HERE);

// ---- Node adapter with the browser StockfishEngine.analyse() interface ------
function nodeEngine() {
  const initEngine = require('./vendor-sf/loader.cjs');
  const enginePath = HERE + 'vendor-sf/stockfish-18-lite-single.js';
  let capture = [];
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  return new Promise((resolve, reject) => {
    initEngine(enginePath, (err, engine) => {
      if (err) return reject(err);
      const origWrite = process.stdout.write.bind(process.stdout);
      process.stdout.write = (c, ...a) => { capture.push(c.toString()); return true; };
      const cmd = (c) => engine.sendCommand(c);
      cmd('uci');
      setTimeout(() => {
        resolve({
          async analyse(fen, { depth = 12, multipv = 3 } = {}) {
            capture = [];
            cmd('ucinewgame');
            cmd(`setoption name MultiPV value ${multipv}`);
            cmd(`position fen ${fen}`);
            cmd(`go depth ${depth}`);
            for (let i = 0; i < 150; i++) { await wait(60); if (capture.join('').includes('bestmove')) break; }
            const byRank = new Map();
            for (const line of capture.join('').split('\n')) {
              if (!line.startsWith('info') || !line.includes(' pv ')) continue;
              const t = line.split(/\s+/);
              let rank = 1, cp = null, mate = null;
              for (let k = 0; k < t.length; k++) {
                if (t[k] === 'multipv') rank = parseInt(t[++k], 10);
                else if (t[k] === 'cp') cp = parseInt(t[++k], 10);
                else if (t[k] === 'mate') mate = parseInt(t[++k], 10);
                else if (t[k] === 'pv') break;
              }
              if (cp === null && mate === null) continue;
              const movesIdx = t.indexOf('pv');
              const mvs = movesIdx >= 0 ? t.slice(movesIdx + 1) : [];
              byRank.set(rank, {
                rank, scoreCp: cp ?? 0, mateIn: mate, moves: mvs,
                scoreCpWhite(whiteToMove) {
                  let v = this.mateIn !== null ? (this.mateIn > 0 ? 30000 : -30000) : this.scoreCp;
                  return whiteToMove ? v : -v;
                },
              });
            }
            return { fen, depth, pvLines: [...byRank.values()].sort((a, b) => a.rank - b.rank) };
          },
          _restoreStdout() { process.stdout.write = origWrite; },
        });
      }, 800);
    });
  });
}

// ---------------------------------------------------------------------------

const PGN = `[Event "test"]
[White "A"]
[Black "B"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O Nge7 1-0`;

let failures = 0;
const fail = (m) => { failures++; console.log('  FAIL: ' + m); };

const engine = await nodeEngine();
const netPath = HERE + '../web/nnue/nn-1c0000000000.nnue';
const buf = readFileSync(netPath);
const weights = parseNnue(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
process.stderr.write(`net parsed: ${weights.arch.name}\n`);

const records = await analyseGame({
  Chess, engine, weights, pgn: PGN, depth: 10, multipv: 2,
  onMove: (i, total, san) => process.stderr.write(`  move ${i + 1}/${total}: ${san}\n`),
});
engine._restoreStdout();

console.log(`\nanalysed ${records.length} moves`);
if (records.length !== 14) fail(`expected 14 moves, got ${records.length}`);

const QUALS = ['best', 'excellent', 'good', 'inaccuracy', 'mistake', 'blunder', 'forced'];
for (const [i, r] of records.entries()) {
  if (!r.move_san) fail(`move ${i}: missing move_san`);
  if (!QUALS.includes(r.quality)) fail(`move ${i} (${r.move_san}): bad quality ${r.quality}`);
  if (!r.fen_after || r.fen_after.split(' ').length < 4) fail(`move ${i}: bad fen_after`);
  if (typeof r.sf_eval_after !== 'number' || !Number.isFinite(r.sf_eval_after)) fail(`move ${i}: bad sf_eval_after`);
  if (!Array.isArray(r.simple_paragraphs)) fail(`move ${i}: missing simple_paragraphs`);
  if (!Array.isArray(r.complex_groups)) fail(`move ${i}: missing complex_groups`);
  // The internal-eval leak must be gone: no paragraph should show absurd pawn counts.
  for (const p of r.simple_paragraphs) {
    const m = p.match(/([+-]?\d[\d.]*) pawns/);
    if (m && Math.abs(parseFloat(m[1])) > 100) fail(`move ${i} (${r.move_san}): absurd pawn value in "${p}"`);
    if (/\d{5,}/.test(p)) fail(`move ${i} (${r.move_san}): 5+ digit number leaked in "${p}"`);
  }
}

// Show a sample so a human can eyeball it.
const sample = records[8] || records[records.length - 1];
process.stderr.write('\nsample record (move ' + sample.move_san + '):\n');
process.stderr.write('  quality: ' + sample.quality_label + '\n');
process.stderr.write('  headline: ' + sample.simple_headline + '\n');
for (const p of sample.simple_paragraphs) process.stderr.write('  • ' + p + '\n');
process.stderr.write('  eval bar (white cp): ' + sample.sf_eval_after + '\n');

console.log(failures === 0 ? '\nIntegration pipeline PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
