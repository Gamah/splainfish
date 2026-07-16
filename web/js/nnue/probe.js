/**
 * probe.js — NNUE forward pass and activation probing.
 *
 * Port of splainfish/probe.py. Weight matrices arrive from parser.js as flat
 * typed arrays with an explicit row stride, so numpy's `W @ x` and `W.T @ v`
 * become the matVec/matTVec helpers below.
 *
 * This mirrors the Python faithfully, including its approximations: the
 * back-projection is a first-order Taylor expansion at the midpoint
 * activations, and the SF16 feature-transformer fold Jacobian is approximated.
 * See splainfish/probe.py and the README's "Limitations and honesty" section.
 *
 * SF16 forward pass:
 *   FT accumulator: i16 accumulated -> clip [0,127] -> fold pairs -> 1536
 *   fc_0: 1536 -> 16   (SqrClippedReLU[0:15] + ClippedReLU[0:15] -> 30)
 *   fc_1: 30 -> 32     (ClippedReLU -> 32)
 *   fc_2: 32 -> 1
 *   Skip: fc_0_raw[15] * (600*OutputScale) / (127*(1<<WeightScaleBits))
 *
 * SF18 forward pass:
 *   FT accumulator: i16 accumulated -> clip [0,255] -> 1024 per perspective
 *   fc_0: 2048 -> 32   (SqrClippedReLU + ClippedReLU -> 64)
 *   fc_1: 64 -> 32     (SqrClippedReLU + ClippedReLU -> 64)
 *   fc_2: 128 -> 1
 *   Skip: fc_0_raw[-2] - fc_0_raw[-1]
 */

import {
  WEIGHT_SCALE_BITS, WEIGHT_SCALE, HIDDEN_ONE_VAL,
  OUTPUT_SCALE, FT_MAX_VAL, LAYER_STACKS,
} from './parser.js';
import { WHITE, BLACK, halfkaLabel } from './features.js';

// Fixed layout geometry per forward-pass style (see splainfish/probe.py). These
// are not read from the file; dispatch is on weights.arch.ftStyle.
const FOLD_L1      = 1536;
const FOLD_FC0_OUT = 16; // 15 outputs + 1 skip
const FOLD_FC1_OUT = 32;
const CONCAT_L2 = 32;
const CONCAT_L3 = 32;

const SF16_L1 = FOLD_L1, SF16_FC0_OUT = FOLD_FC0_OUT, SF16_FC1_OUT = FOLD_FC1_OUT;
const SF18_L1 = 1024, SF18_L2 = CONCAT_L2, SF18_L3 = CONCAT_L3;

// ---------------------------------------------------------------------------
// Linear algebra over flat row-major typed arrays
// ---------------------------------------------------------------------------

/** y[r] = sum_c W[r][c] * x[c] + b[r]   (numpy: W @ x + b) */
function matVec(W, rows, cols, x, b) {
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += W[base + c] * x[c];
    y[r] = acc + (b ? b[r] : 0);
  }
  return y;
}

/** y[c] = sum_r W[r][c] * v[r]   (numpy: W.T @ v) */
function matTVec(W, rows, cols, v) {
  const y = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    const vr = v[r];
    if (vr === 0) continue;
    const base = r * cols;
    for (let c = 0; c < cols; c++) y[c] += W[base + c] * vr;
  }
  return y;
}

function concat(a, b) {
  const out = new Float64Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function dot(a, b, n = a.length) {
  let acc = 0;
  for (let i = 0; i < n; i++) acc += a[i] * b[i];
  return acc;
}

function sub(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// ---------------------------------------------------------------------------
// King bucket -> LayerStack selection
// ---------------------------------------------------------------------------

const KING_LAYER_BUCKET = [
   0,  1,  2,  3,  3,  2,  1,  0,
   4,  5,  6,  7,  7,  6,  5,  4,
   8,  9, 10, 11, 11, 10,  9,  8,
   8,  9, 10, 11, 11, 10,  9,  8,
  12, 12, 13, 13, 13, 13, 12, 12,
  12, 12, 13, 13, 13, 13, 12, 12,
  14, 14, 15, 15, 15, 15, 14, 14,
  14, 14, 15, 15, 15, 15, 14, 14,
];

function layerStackIndex(board) {
  let ksq = board.king(board.turn);
  if ((ksq & 7) < 4) ksq ^= 7;
  return KING_LAYER_BUCKET[ksq] % LAYER_STACKS;
}

// ---------------------------------------------------------------------------
// Forward pass primitives
// ---------------------------------------------------------------------------

/** FT accumulator for one perspective, clipped to [0, clampMax]. */
function accumulateFt(features, ft, perspective, clampMax = FT_MAX_VAL) {
  const l1 = ft.weightStride;
  const acc = new Float64Array(l1);
  for (let i = 0; i < l1; i++) acc[i] = ft.biases[i];

  const halfka = perspective === WHITE ? features.halfkaWhite : features.halfkaBlack;
  for (const idx of halfka) {
    const base = idx * l1;
    for (let i = 0; i < l1; i++) acc[i] += ft.weights[base + i];
  }

  if (ft.threatWeights) {
    const threats = perspective === WHITE ? features.threatWhite : features.threatBlack;
    const tStride = ft.threatStride;
    for (const idx of threats) {
      const base = idx * tStride;
      for (let i = 0; i < l1; i++) acc[i] += ft.threatWeights[base + i];
    }
  }

  for (let i = 0; i < l1; i++) acc[i] = clamp(acc[i], 0, clampMax);
  return acc;
}

function sqrClippedRelu(x, scaleBits) {
  const scale = 1 << (scaleBits + 1);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const c = clamp(x[i], 0, scale - 1);
    out[i] = (c * c) / (scale * scale);
  }
  return out;
}

function clippedRelu(x, scaleBits) {
  const scale = 1 << (scaleBits + 1);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = clamp(x[i], 0, scale - 1) / scale;
  return out;
}

// ---------------------------------------------------------------------------
// SF16 forward
// ---------------------------------------------------------------------------

function forwardSf16(board, features, weights) {
  const stack = weights.layerStacks[layerStackIndex(board)];
  const ft = weights.featureTransformer;
  const isWhite = board.turn === WHITE;

  // SF16 clips the accumulator to 127, not FT_MAX_VAL.
  const accW = accumulateFt(features, ft, WHITE, 127);
  const accB = accumulateFt(features, ft, BLACK, 127);

  const H = SF16_L1 / 2; // 768

  /** out[j] = clip(acc[j],0,127) * clip(acc[j+H],0,127) / 128 */
  const ftFold = (acc) => {
    const out = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      const lo = clamp(acc[j], 0, 127);
      const hi = clamp(acc[j + H], 0, 127);
      out[j] = (lo * hi) / 128;
    }
    return out;
  };

  const usFold   = ftFold(isWhite ? accW : accB);
  const themFold = ftFold(isWhite ? accB : accW);
  const ftOut = concat(usFold, themFold); // (1536,)

  const fc0Pre = matVec(stack.fc0Weights, SF16_FC0_OUT, SF16_L1, ftOut, stack.fc0Biases);

  // SqrClippedReLU / ClippedReLU over [0:15]; index 15 is the skip neuron.
  const fc0Head = fc0Pre.subarray(0, SF16_FC0_OUT - 1);
  const fc0Sqr = sqrClippedRelu(fc0Head, WEIGHT_SCALE_BITS + 1);
  const fc0Lin = clippedRelu(fc0Head, WEIGHT_SCALE_BITS + 1);
  const fc0Concat = concat(fc0Sqr, fc0Lin); // (30,)

  const fc1Pre = matVec(stack.fc1Weights, SF16_FC1_OUT, 30, fc0Concat, stack.fc1Biases);
  const fc1Out = clippedRelu(fc1Pre, WEIGHT_SCALE_BITS); // (32,)

  const fc2Pre = matVec(stack.fc2Weights, 1, SF16_FC1_OUT, fc1Out, stack.fc2Biases);

  const skipRaw = fc0Pre[SF16_FC0_OUT - 1];
  const skip = (skipRaw * (600 * OUTPUT_SCALE)) / (127 * WEIGHT_SCALE);

  const rawOut = fc2Pre[0] + skip;
  const denominator = 127 * WEIGHT_SCALE; // 8128
  let cp = Math.trunc((rawOut * 600 * OUTPUT_SCALE) / denominator);
  if (!isWhite) cp = -cp;

  return {
    ftAccWhite: accW, ftAccBlack: accB, ftOut,
    fc0Pre, fc0Sqr, fc0Lin, fc0Concat,
    fc1Pre, fc1Out, fc1Sqr: null,
    fc2Pre, skip, centipawns: cp,
  };
}

// ---------------------------------------------------------------------------
// SF18 forward
// ---------------------------------------------------------------------------

function forwardSf18(board, features, weights) {
  const stack = weights.layerStacks[layerStackIndex(board)];
  const ft = weights.featureTransformer;
  const isWhite = board.turn === WHITE;

  const accW = accumulateFt(features, ft, WHITE);
  const accB = accumulateFt(features, ft, BLACK);

  const ftOut = isWhite ? concat(accW, accB) : concat(accB, accW); // (2048,)

  const fc0Pre = matVec(stack.fc0Weights, SF18_L2, SF18_L1 * 2, ftOut, stack.fc0Biases);
  const fc0Sqr = sqrClippedRelu(fc0Pre, WEIGHT_SCALE_BITS + 1);
  const fc0Lin = clippedRelu(fc0Pre, WEIGHT_SCALE_BITS + 1);
  const fc0Concat = concat(fc0Sqr, fc0Lin); // (64,)

  const fc1Pre = matVec(stack.fc1Weights, SF18_L3, SF18_L2 * 2, fc0Concat, stack.fc1Biases);
  const fc1Sqr = sqrClippedRelu(fc1Pre, WEIGHT_SCALE_BITS);
  const fc1Lin = clippedRelu(fc1Pre, WEIGHT_SCALE_BITS);
  const fc1Out = concat(fc1Sqr, fc1Lin); // (64,)

  const concatAll = concat(fc0Concat, fc1Out); // (128,)
  const fc2Pre = matVec(
    stack.fc2Weights, 1, SF18_L2 * 2 + SF18_L3 * 2, concatAll, stack.fc2Biases,
  );

  const skip = fc0Pre[SF18_L2 - 2] - fc0Pre[SF18_L2 - 1];
  const rawOut = fc2Pre[0] + skip;

  const numerator = 600 * OUTPUT_SCALE;
  const denominator = HIDDEN_ONE_VAL * WEIGHT_SCALE * 2; // 16384
  let cp = Math.trunc((rawOut * numerator) / denominator);
  if (!isWhite) cp = -cp;

  return {
    ftAccWhite: accW, ftAccBlack: accB, ftOut,
    fc0Pre, fc0Sqr, fc0Lin, fc0Concat,
    fc1Pre, fc1Out, fc1Sqr,
    fc2Pre, skip, centipawns: cp,
  };
}

export function forward(board, features, weights) {
  return weights.arch.ftStyle === "fold"
    ? forwardSf16(board, features, weights)
    : forwardSf18(board, features, weights);
}

// ---------------------------------------------------------------------------
// Back-projection
// ---------------------------------------------------------------------------

/**
 * Back-project the output delta through fc2 -> fc1 -> fc0 -> FT.
 * First-order Taylor approximation at the midpoint activations.
 * Returns the attribution vector over the FT output.
 */
function backProject(actB, actA, stack, weights) {
  const isFold = weights.arch.ftStyle === "fold";

  if (isFold) {
    const deltaFc1Out = sub(actA.fc1Out, actB.fc1Out);       // (32,)
    const deltaFc0Sqr = sub(actA.fc0Sqr, actB.fc0Sqr);       // (15,)
    const deltaFc0Lin = sub(actA.fc0Lin, actB.fc0Lin);       // (15,)
    const deltaFc0Concat = concat(deltaFc0Sqr, deltaFc0Lin); // (30,)

    // fc2 weights row 0, first 32 entries.
    const w2Full = stack.fc2Weights.subarray(0, SF16_FC1_OUT);

    // ReLU Jacobian at the midpoint of fc1_pre.
    const scale = 1 << (WEIGHT_SCALE_BITS + 1);
    const gate = new Float64Array(SF16_FC1_OUT);
    for (let i = 0; i < SF16_FC1_OUT; i++) {
      const mid = (actB.fc1Pre[i] + actA.fc1Pre[i]) / 2;
      const jac = mid >= 0 && mid < scale - 1 ? 1 : 0;
      gate[i] = w2Full[i] * deltaFc1Out[i] * jac;
    }

    // w1 is (32, 30): w1.T @ gate -> (30,)
    const gFc0ViaFc1 = matTVec(stack.fc1Weights, SF16_FC1_OUT, 30, gate);

    const totalFc0 = new Float64Array(30);
    for (let i = 0; i < 30; i++) totalFc0[i] = gFc0ViaFc1[i] + deltaFc0Concat[i];

    // Linear branch only: fc0_weights[:15] against total_fc0[15:].
    const w0Lin = stack.fc0Weights.subarray(0, (SF16_FC0_OUT - 1) * SF16_L1);
    const linHalf = totalFc0.subarray(SF16_FC0_OUT - 1); // (15,)
    return matTVec(w0Lin, SF16_FC0_OUT - 1, SF16_L1, linHalf); // (1536,)
  }

  const L2 = SF18_L2, L3 = SF18_L3;
  const w2 = stack.fc2Weights; // (1, 128) -> flat 128
  const w2Fc0 = w2.subarray(0, L2 * 2);
  const w2Fc1 = w2.subarray(L2 * 2);

  const deltaFc0Sqr = sub(actA.fc0Sqr, actB.fc0Sqr);
  const deltaFc0Lin = sub(actA.fc0Lin, actB.fc0Lin);
  const deltaFc0Concat = concat(deltaFc0Sqr, deltaFc0Lin); // (64,)

  const deltaFc1Lin = sub(actA.fc1Out.subarray(L3), actB.fc1Out.subarray(L3)); // (32,)

  // w2_fc1[L3:] * delta_fc1_lin
  const fc1LinWeighted = new Float64Array(L3);
  for (let i = 0; i < L3; i++) fc1LinWeighted[i] = w2Fc1[L3 + i] * deltaFc1Lin[i];

  // w1 is (L3, L2*2): w1.T @ weighted -> (L2*2,), already fc0_concat-width.
  const deltaFc0ViaFc1 = matTVec(stack.fc1Weights, L3, L2 * 2, fc1LinWeighted);

  const totalFc0 = new Float64Array(L2 * 2);
  for (let i = 0; i < L2 * 2; i++) {
    totalFc0[i] = w2Fc0[i] * deltaFc0Concat[i] + deltaFc0ViaFc1[i];
  }

  // Linear branch: w0.T @ total_fc0[L2:] -> (L1*2,). The slice is (L2,), which
  // matches w0's L2 rows.
  const linHalf = totalFc0.subarray(L2);
  return matTVec(stack.fc0Weights, L2, SF18_L1 * 2, linHalf);
}

// ---------------------------------------------------------------------------
// Grouping
// ---------------------------------------------------------------------------

function groupLabel(att, movingColor) {
  const pc = att.pieceColor;
  const pt = att.pieceType;
  const mover = movingColor === WHITE ? 'White' : 'Black';
  const isOwn = pc === mover;
  const near = att.kingBucket <= 7;

  if (pt === 'king') return 'king position';
  if (pt === 'queen') {
    if (!isOwn && near) return 'enemy queen threatening king area';
    if (!isOwn) return 'enemy queen activity';
    if (near) return 'own queen near king (defensive)';
    return 'own queen activity';
  }
  if (pt === 'rook') {
    if (!isOwn && near) return 'enemy rook pressure near king';
    if (!isOwn) return 'enemy rook activity';
    return 'own rook activity';
  }
  if (pt === 'knight' || pt === 'bishop') {
    if (!isOwn && near) return `enemy ${pt} attacking king zone`;
    if (!isOwn) return `enemy ${pt} activity`;
    return `own ${pt} activity / coordination`;
  }
  if (pt === 'pawn') {
    if (isOwn && near) return 'own pawn shield';
    if (!isOwn && near) return 'enemy pawn advance near king';
    if (isOwn) return 'own pawn structure';
    return 'enemy pawn structure';
  }
  return 'other';
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

/**
 * Run the forward pass on both positions, diff the activations, and attribute
 * the output delta back onto the input features that changed.
 */
export function probe(
  boardBefore, boardAfter, featBefore, featAfter, featDiff, weights, movingColor,
  sfEvalBefore = null, sfEvalAfter = null,
) {
  const actB = forward(boardBefore, featBefore, weights);
  const actA = forward(boardAfter, featAfter, weights);
  const deltaCp = actA.centipawns - actB.centipawns;

  const ftDeltaWhite = sub(actA.ftAccWhite, actB.ftAccWhite);
  const ftDeltaBlack = sub(actA.ftAccBlack, actB.ftAccBlack);
  const fc0Delta = sub(actA.fc0Pre, actB.fc0Pre);
  const fc1Delta = sub(actA.fc1Pre, actB.fc1Pre);

  const stack = weights.layerStacks[layerStackIndex(boardBefore)];
  const ftAttr = backProject(actB, actA, stack, weights);

  const l1 = weights.l1;
  const isWhiteTurn = boardBefore.turn === WHITE;
  const isFold = weights.arch.ftStyle === "fold";

  // Split the attribution vector by perspective.
  const half = isFold ? l1 / 2 : l1;
  const firstHalf = ftAttr.subarray(0, half);
  const secondHalf = ftAttr.subarray(half);
  const ftAttrW = isWhiteTurn ? firstHalf : secondHalf;
  const ftAttrB = isWhiteTurn ? secondHalf : firstHalf;

  const featureAttributions = [];
  const ftW = weights.featureTransformer;
  const H = l1 / 2;

  const add = (changed, direction, perspective, attrVec) => {
    for (const idx of changed) {
      const info = halfkaLabel(idx, perspective);
      const base = idx * ftW.weightStride;
      const wCol = ftW.weights.subarray(base, base + l1);

      let norm = 0;
      for (let i = 0; i < l1; i++) norm += wCol[i] * wCol[i];
      norm = Math.sqrt(norm) + 1e-8;

      let contrib;
      if (isFold) {
        // The Python approximates the FT fold Jacobian with a uniform 1/128 and
        // dots the normalised (lower + upper) halves of the weight column
        // against the folded attribution.
        let acc = 0;
        for (let j = 0; j < H; j++) {
          acc += ((wCol[j] + wCol[j + H]) / (128 * norm)) * attrVec[j];
        }
        contrib = acc;
      } else {
        contrib = dot(wCol, attrVec, l1) / norm;
      }

      if (direction === 'lost') contrib = -contrib;

      featureAttributions.push({
        featureIdx: idx,
        featureType: 'halfka',
        direction,
        perspective,
        pieceColor: info.pieceColor,
        pieceType: info.pieceType,
        pieceSq: info.pieceSq,
        kingBucket: info.kingBucket,
        contribution: contrib,
      });
    }
  };

  add(featDiff.halfkaWhiteGained, 'gained', WHITE, ftAttrW);
  add(featDiff.halfkaWhiteLost,   'lost',   WHITE, ftAttrW);
  add(featDiff.halfkaBlackGained, 'gained', BLACK, ftAttrB);
  add(featDiff.halfkaBlackLost,   'lost',   BLACK, ftAttrB);

  featureAttributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const groups = new Map();
  for (const att of featureAttributions) {
    const label = groupLabel(att, movingColor);
    let g = groups.get(label);
    if (!g) {
      g = { group: label, contribution: 0, featureCount: 0, direction: 'positive', features: [] };
      groups.set(label, g);
    }
    g.contribution += att.contribution;
    g.featureCount += 1;
    g.features.push(att);
  }

  const grouped = [...groups.values()]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  for (const g of grouped) g.direction = g.contribution >= 0 ? 'positive' : 'negative';

  return {
    actBefore: actB,
    actAfter: actA,
    evalBeforeCp: actB.centipawns,
    evalAfterCp: actA.centipawns,
    deltaCp,
    ftDeltaWhite, ftDeltaBlack, fc0Delta, fc1Delta,
    featureAttributions,
    groupedAttributions: grouped,
    sfEvalBefore, sfEvalAfter,
  };
}
