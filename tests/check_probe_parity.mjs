/**
 * check_probe_parity.mjs — Diff web/js/nnue/probe.js against the real
 * splainfish/probe.py.
 *
 * Reads the synthetic weight dumps and expected activations produced by
 * tests/ref_probe.py. Deliberately bypasses parser.js (covered separately by
 * check_parser_parity.mjs) and feeds probe.js the weights directly.
 *
 * Usage: node check_probe_parity.mjs <ref-dir>
 *
 * Floats are compared with a relative tolerance rather than exactly: numpy's
 * matmul dispatches to BLAS, which is free to reassociate the sums and use FMA,
 * so the last bits legitimately differ from a naive JS loop.
 */
import { readFileSync } from 'node:fs';
import { Chess } from '../vendor/chess.js/chess.js';
import { boardFromChessJs, computeFeatures, diffFeatures, WHITE } from '../web/js/nnue/features.js';
import { probe } from '../web/js/nnue/probe.js';
import { ARCHITECTURES } from '../web/js/nnue/parser.js';

const FOLD_ARCH = ARCHITECTURES['halfka-1536'];
const CONCAT_ARCH = ARCHITECTURES['halfka-1024-threats'];
const SF18_THREAT_DIMS = CONCAT_ARCH.threatDims;

const refDir = process.argv[2];
if (!refDir) {
  console.error('usage: node check_probe_parity.mjs <ref-dir>');
  process.exit(2);
}

const REL_TOL = 1e-9;
const ABS_TOL = 1e-6;

let failures = 0;
const MAX_REPORT = 10;
function fail(msg) {
  failures++;
  if (failures <= MAX_REPORT) console.log('  FAIL: ' + msg);
  else if (failures === MAX_REPORT + 1) console.log('  ... (further failures suppressed)');
}

function closeEnough(a, b) {
  if (a === b) return true;
  const diff = Math.abs(a - b);
  if (diff <= ABS_TOL) return true;
  return diff <= REL_TOL * Math.max(Math.abs(a), Math.abs(b));
}

function cmpVec(name, got, want, ctx) {
  if (got.length !== want.length) {
    fail(`${ctx} ${name}: length ${got.length} vs ${want.length}`);
    return;
  }
  for (let i = 0; i < want.length; i++) {
    if (!closeEnough(got[i], want[i])) {
      fail(`${ctx} ${name}[${i}] py=${want[i]} js=${got[i]} (absdiff=${Math.abs(got[i] - want[i]).toExponential(3)})`);
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Binary dump reader — layout documented in tests/ref_probe.py:dump_weights
// ---------------------------------------------------------------------------

class Cursor {
  constructor(buf) {
    // Node Buffer.slice() aliases subarray (shares memory), so .buffer would be
    // the whole allocation pool. Re-wrap as a plain Uint8Array to get copying
    // slice() semantics.
    this.buf = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    this.pos = 0;
  }
  u32() { const v = new DataView(this.buf.buffer, this.buf.byteOffset + this.pos, 4).getUint32(0, true); this.pos += 4; return v; }
  u8() { return this.buf[this.pos++]; }
  arr(Ctor, count) {
    const bytes = count * Ctor.BYTES_PER_ELEMENT;
    // Copy out: the offset is not guaranteed to be element-aligned.
    const slice = this.buf.slice(this.pos, this.pos + bytes);
    this.pos += bytes;
    return new Ctor(slice.buffer);
  }
}

// ft_style_tag (0 = fold, 1 = concat) -> architecture + derived fc dimensions.
function dimsFor(arch) {
  const fold = arch.ftStyle === 'fold';
  return {
    arch,
    l1: arch.l1,
    halfka: arch.halfkaDims,
    fc0Out: arch.fc0Out,
    fc0In: fold ? arch.l1 : arch.l1 * 2,
    fc1Out: arch.fc1Out,
    fc1In: fold ? (arch.fc0Out - 1) * 2 : arch.fc0Out * 2,
    fc2In: fold ? arch.fc1Out : arch.fc0Out * 2 + arch.fc1Out * 2,
  };
}
const DIMS = [dimsFor(FOLD_ARCH), dimsFor(CONCAT_ARCH)];

function loadWeights(path) {
  const c = new Cursor(readFileSync(path));
  const styleTag = c.u32();
  const d = DIMS[styleTag];
  if (!d) throw new Error(`unknown ft_style_tag ${styleTag}`);

  // Dump order (see dump_weights): biases, weights, has_threats flag (u8),
  // threat_weights (only when flagged), then stacks.
  const biases = c.arr(Int16Array, d.l1);
  const weights = c.arr(Int16Array, d.halfka * d.l1);
  const hasThreatsFlag = c.u8() === 1;
  const threatWeightsReal = hasThreatsFlag
    ? c.arr(Int8Array, SF18_THREAT_DIMS * d.l1) : null;

  const layerStacks = [];
  for (let i = 0; i < 8; i++) {
    layerStacks.push({
      fc0Biases: c.arr(Int32Array, d.fc0Out),
      fc0Weights: c.arr(Int8Array, d.fc0Out * d.fc0In),
      fc1Biases: c.arr(Int32Array, d.fc1Out),
      fc1Weights: c.arr(Int8Array, d.fc1Out * d.fc1In),
      fc2Biases: c.arr(Int32Array, 1),
      fc2Weights: c.arr(Int8Array, d.fc2In),
    });
  }
  if (c.pos !== c.buf.length) {
    throw new Error(`trailing bytes: read ${c.pos} of ${c.buf.length}`);
  }

  return {
    arch: d.arch,
    description: 'synthetic',
    featureTransformer: {
      biases, weights, weightStride: d.l1,
      psqtWeights: null, psqtStride: 8,
      threatWeights: threatWeightsReal, threatStride: d.l1, threatPsqtWeights: null,
    },
    layerStacks,
    l1: d.l1, halfkaDims: d.halfka,
    fc0Out: d.fc0Out, fc1Out: d.fc1Out,
    hasThreats: hasThreatsFlag,
  };
}

// ---------------------------------------------------------------------------

const ref = JSON.parse(readFileSync(`${refDir}/probe-ref.json`, 'utf8'));

for (const name of ['sf16', 'sf18']) {
  const cases = ref[name];
  if (!cases) continue;
  process.stdout.write(`loading ${name} weights... `);
  const weights = loadWeights(`${refDir}/weights-${name}.bin`);
  console.log('ok');

  let nAct = 0, nAttr = 0;
  for (const [i, c] of cases.entries()) {
    const ctx = `${name}[${i}] ${c.move_uci}`;

    const gb = new Chess(c.fen_before);
    const ga = new Chess(c.fen_after);
    const bb = boardFromChessJs(gb);
    const ba = boardFromChessJs(ga);
    const fb = computeFeatures(bb);
    const fa = computeFeatures(ba);
    const fd = diffFeatures(fb, fa);

    const r = probe(bb, ba, fb, fa, fd, weights, bb.turn);

    // Scalars
    if (r.evalBeforeCp !== c.eval_before_cp) {
      fail(`${ctx} eval_before_cp py=${c.eval_before_cp} js=${r.evalBeforeCp}`);
    }
    if (r.evalAfterCp !== c.eval_after_cp) {
      fail(`${ctx} eval_after_cp py=${c.eval_after_cp} js=${r.evalAfterCp}`);
    }
    if (r.deltaCp !== c.delta_cp) {
      fail(`${ctx} delta_cp py=${c.delta_cp} js=${r.deltaCp}`);
    }

    // Activations, both positions
    cmpVec('act_before.fc0_pre', r.actBefore.fc0Pre, c.act_before.fc0_pre, ctx);
    cmpVec('act_before.fc1_pre', r.actBefore.fc1Pre, c.act_before.fc1_pre, ctx);
    cmpVec('act_before.fc2_pre', r.actBefore.fc2Pre, c.act_before.fc2_pre, ctx);
    cmpVec('act_after.fc0_pre', r.actAfter.fc0Pre, c.act_after.fc0_pre, ctx);
    cmpVec('act_after.fc1_pre', r.actAfter.fc1Pre, c.act_after.fc1_pre, ctx);
    cmpVec('act_after.fc2_pre', r.actAfter.fc2Pre, c.act_after.fc2_pre, ctx);
    cmpVec('ft_acc_white_head', r.actBefore.ftAccWhite.subarray(0, 24),
           c.act_before.ft_acc_white_head, ctx);
    cmpVec('ft_acc_black_head', r.actBefore.ftAccBlack.subarray(0, 24),
           c.act_before.ft_acc_black_head, ctx);
    if (!closeEnough(r.actBefore.skip, c.act_before.skip)) {
      fail(`${ctx} act_before.skip py=${c.act_before.skip} js=${r.actBefore.skip}`);
    }
    nAct += r.actBefore.fc0Pre.length + r.actBefore.fc1Pre.length + 48;

    // Attribution
    if (r.featureAttributions.length !== c.n_attributions) {
      fail(`${ctx} n_attributions py=${c.n_attributions} js=${r.featureAttributions.length}`);
    }
    for (const [j, want] of c.top_attributions.entries()) {
      const got = r.featureAttributions[j];
      if (!got) { fail(`${ctx} attribution[${j}] missing`); continue; }
      if (got.featureIdx !== want.feature_idx || got.direction !== want.direction) {
        fail(`${ctx} attr[${j}] idx/dir py=${want.feature_idx}/${want.direction} js=${got.featureIdx}/${got.direction}`);
      }
      if (got.pieceType !== want.piece_type || got.pieceColor !== want.piece_color) {
        fail(`${ctx} attr[${j}] piece py=${want.piece_color} ${want.piece_type} js=${got.pieceColor} ${got.pieceType}`);
      }
      if (!closeEnough(got.contribution, want.contribution)) {
        fail(`${ctx} attr[${j}] contribution py=${want.contribution} js=${got.contribution}`);
      }
      nAttr++;
    }

    // Groups
    if (r.groupedAttributions.length !== c.grouped.length) {
      fail(`${ctx} group count py=${c.grouped.length} js=${r.groupedAttributions.length}`);
    }
    for (const [j, want] of c.grouped.entries()) {
      const got = r.groupedAttributions[j];
      if (!got) { fail(`${ctx} group[${j}] missing`); continue; }
      if (got.group !== want.group) {
        fail(`${ctx} group[${j}] name py=${want.group} js=${got.group}`);
      }
      if (got.featureCount !== want.feature_count) {
        fail(`${ctx} group[${j}] count py=${want.feature_count} js=${got.featureCount}`);
      }
      if (got.direction !== want.direction) {
        fail(`${ctx} group[${j}] direction py=${want.direction} js=${got.direction}`);
      }
      if (!closeEnough(got.contribution, want.contribution)) {
        fail(`${ctx} group[${j}] contribution py=${want.contribution} js=${got.contribution}`);
      }
    }
  }
  console.log(`${name}: ${cases.length} positions, ${nAct} activations, ${nAttr} attributions compared`);
}

console.log(failures === 0 ? '\nAll probe parity checks PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
