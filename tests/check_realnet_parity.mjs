/**
 * check_realnet_parity.mjs — Verify web/js/nnue/parser.js parses a real .nnue
 * byte-identically to the Python. Reads ref_realnet.py's fingerprint on stdin.
 *
 * Usage: node check_realnet_parity.mjs <path.nnue>
 *
 * The whole file is read into memory (~71 MB net -> ~160 MB of typed arrays),
 * so run node with a raised heap.
 */
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Chess } from '../vendor/chess.js/chess.js';
import { parseNnue } from '../web/js/nnue/parser.js';
import { boardFromChessJs, computeFeatures } from '../web/js/nnue/features.js';
import { forward } from '../web/js/nnue/probe.js';

const path = process.argv[2];
if (!path) {
  console.error('usage: node check_realnet_parity.mjs <path.nnue>');
  process.exit(2);
}

const ref = JSON.parse(await new Promise((resolve) => {
  let buf = '';
  process.stdin.on('data', (d) => (buf += d));
  process.stdin.on('end', () => resolve(buf));
}));

let failures = 0;
const check = (name, got, want) => {
  if (got !== want) { failures++; console.log(`  FAIL ${name}: js=${got} py=${want}`); }
};

// Match ref_realnet.checksum: sha256 of the little-endian bytes, first 16 hex.
function checksum(typedArr) {
  const bytes = new Uint8Array(typedArr.buffer, typedArr.byteOffset, typedArr.byteLength);
  return createHash('sha256').update(bytes).digest('hex').slice(0, 16);
}
function sum(typedArr) {
  let s = 0n;
  for (let i = 0; i < typedArr.length; i++) s += BigInt(typedArr[i]);
  return Number(s);
}
function checkSamples(name, arr, samples) {
  for (const [i, v] of samples) {
    if (arr[i] !== v) { failures++; console.log(`  FAIL ${name} sample[${i}]: js=${arr[i]} py=${v}`); return; }
  }
}

const buf = readFileSync(path);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const w = parseNnue(ab);
const ft = w.featureTransformer;

check('arch', w.arch.name, ref.arch);
check('l1', w.l1, ref.l1);
check('halfka_dims', w.halfkaDims, ref.halfka_dims);
check('has_threats', w.hasThreats, ref.has_threats);

check('ft.biases_sum', sum(ft.biases), ref.ft.biases_sum);
check('ft.biases_checksum', checksum(ft.biases), ref.ft.biases_checksum);
checkSamples('ft.biases', ft.biases, ref.ft.biases_sample);

check('ft.weights_sum', sum(ft.weights), ref.ft.weights_sum);
check('ft.weights_checksum', checksum(ft.weights), ref.ft.weights_checksum);
checkSamples('ft.weights', ft.weights, ref.ft.weights_sample);

check('ft.psqt_sum', sum(ft.psqtWeights), ref.ft.psqt_sum);
check('ft.psqt_checksum', checksum(ft.psqtWeights), ref.ft.psqt_checksum);

ref.stacks.forEach((rs, i) => {
  const s = w.layerStacks[i];
  check(`stack[${i}].fc0_biases_sum`, sum(s.fc0Biases), rs.fc0_biases_sum);
  check(`stack[${i}].fc0_weights_checksum`, checksum(s.fc0Weights), rs.fc0_weights_checksum);
  checkSamples(`stack[${i}].fc0_weights`, s.fc0Weights, rs.fc0_weights_sample);
  check(`stack[${i}].fc1_biases_sum`, sum(s.fc1Biases), rs.fc1_biases_sum);
  check(`stack[${i}].fc1_weights_checksum`, checksum(s.fc1Weights), rs.fc1_weights_checksum);
  check(`stack[${i}].fc2_biases_sum`, sum(s.fc2Biases), rs.fc2_biases_sum);
  check(`stack[${i}].fc2_weights_checksum`, checksum(s.fc2Weights), rs.fc2_weights_checksum);
});

// End-to-end: parse -> features -> forward on the real net.
const closeEnough = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));
for (const rf of ref.forward ?? []) {
  const board = boardFromChessJs(new Chess(rf.fen));
  const act = forward(board, computeFeatures(board), w);
  check(`forward centipawns @ ${rf.fen.slice(0, 24)}`, act.centipawns, rf.centipawns);
  for (let i = 0; i < rf.fc0_pre_sample.length; i++) {
    if (!closeEnough(act.fc0Pre[i], rf.fc0_pre_sample[i])) {
      failures++;
      console.log(`  FAIL forward fc0_pre[${i}] @ ${rf.fen.slice(0, 20)}: js=${act.fc0Pre[i]} py=${rf.fc0_pre_sample[i]}`);
    }
  }
}

const nChecks = 8 + ref.stacks.length * 7 + (ref.forward?.length ?? 0);
console.log(
  failures === 0
    ? `\nReal-net parse MATCHES Python byte-for-byte (${w.arch.name}, ${nChecks}+ checks)`
    : `\n${failures} FAILURES`,
);
process.exit(failures === 0 ? 0 : 1);
