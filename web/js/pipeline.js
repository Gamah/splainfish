/**
 * pipeline.js — Analyse a whole game in the browser.
 *
 * Browser port of splainfish/pipeline.py's analyse_game. Ties together chess.js
 * (move generation + PGN), the Stockfish worker (evals), and the NNUE probe
 * (attribution) to produce the same per-move records the HTML viewer consumes.
 *
 * chess.js is imported by the caller and passed in, so this module has no CDN
 * dependency of its own.
 */

import { boardFromChessJs, computeFeatures, diffFeatures } from './nnue/features.js';
import { probe } from './nnue/probe.js';
import { build } from './explain.js';

const MATE_CP = 30000;

/** Stockfish reports from side-to-move; convert an analysis to White's frame. */
function scoreCpWhite(pvLine, whiteToMove) {
  if (!pvLine) return 0;
  return pvLine.scoreCpWhite(whiteToMove);
}

function pvToSan(ChessCtor, uciMoves, fen, maxMoves = 6) {
  const san = [];
  const b = new ChessCtor(fen);
  for (const uci of uciMoves.slice(0, maxMoves)) {
    try {
      const mv = b.move({
        from: uci.slice(0, 2), to: uci.slice(2, 4),
        promotion: uci.length > 4 ? uci[4] : undefined,
      });
      if (!mv) break;
      san.push(mv.san);
    } catch { break; }
  }
  return san;
}

/**
 * Analyse every move of a game.
 *
 * @param {object}   opts
 * @param {Function} opts.Chess         chess.js Chess constructor
 * @param {object}   opts.engine        loaded StockfishEngine
 * @param {object}   opts.weights       parsed NNUEWeights
 * @param {string}   opts.pgn           PGN text
 * @param {number}   opts.depth         search depth
 * @param {number}   opts.multipv       MultiPV lines
 * @param {boolean}  opts.onlyMistakes  keep only inaccuracy/mistake/blunder
 * @param {Function} opts.onMove        (index, total, san) progress callback
 * @returns {Promise<object[]>} per-move explanation records
 */
export async function analyseGame({
  Chess, engine, weights, pgn,
  depth = 14, multipv = 3, onlyMistakes = false, onMove,
}) {
  const game = new Chess();
  try {
    game.loadPgn(pgn);
  } catch (err) {
    throw new Error(`Could not parse PGN: ${err.message}`);
  }

  // chess.js history with SAN + FEN before each move.
  const history = game.history({ verbose: true });
  if (!history.length) throw new Error('No moves found in PGN.');

  const results = [];
  const walk = new Chess();

  // Analyse the initial position once; reuse across the loop like the Python.
  let prevAnalysis = await engine.analyse(walk.fen(), { depth, multipv });

  for (let i = 0; i < history.length; i++) {
    const mv = history[i];
    const fenBefore = walk.fen();
    const movingColorWhite = walk.turn() === 'w';
    const moveNumber = Number(fenBefore.split(' ')[5]);
    const moveSan = mv.san;

    onMove?.(i, history.length, moveSan);

    const sfEvalBefore = scoreCpWhite(prevAnalysis.pvLines[0], movingColorWhite);
    const bestPv = prevAnalysis.pvLines[0];
    let bestMoveSan = null;
    let bestLineSan = [];
    if (bestPv && bestPv.moves.length) {
      const sans = pvToSan(Chess, bestPv.moves, fenBefore);
      bestMoveSan = sans[0] ?? bestPv.moves[0];
      bestLineSan = sans;
    }

    const boardBefore = boardFromChessJs(new Chess(fenBefore));
    const featBefore = computeFeatures(boardBefore);

    // Apply the move.
    walk.move(mv.san);
    const fenAfter = walk.fen();

    const nextAnalysis = await engine.analyse(fenAfter, { depth, multipv });
    // After the move it is the opponent to move; scoreCpWhite needs the side to
    // move at fenAfter, which is !movingColorWhite.
    const sfEvalAfter = scoreCpWhite(nextAnalysis.pvLines[0], !movingColorWhite);

    const boardAfter = boardFromChessJs(new Chess(fenAfter));
    const featAfter = computeFeatures(boardAfter);
    const fdiff = diffFeatures(featBefore, featAfter);

    const movingColor = movingColorWhite ? 1 : 0; // WHITE=1, BLACK=0
    const probeResult = probe(
      boardBefore, boardAfter, featBefore, featAfter, fdiff, weights, movingColor,
      sfEvalBefore, sfEvalAfter,
    );

    const record = build({
      result: probeResult,
      moveSan,
      moveUci: mv.from + mv.to + (mv.promotion ?? ''),
      fenBefore,
      fenAfter,
      moveNumber,
      movingColor,
      bestMoveSan,
      bestLineSan,
      sfEvalBefore,
      sfEvalAfter,
    });

    // Carry the SF evals through under the keys the viewer reads.
    record.sf_eval_before = sfEvalBefore;
    record.sf_eval_after = sfEvalAfter;

    const keep = !onlyMistakes
      || ['inaccuracy', 'mistake', 'blunder'].includes(record.quality);
    if (keep) results.push(record);

    prevAnalysis = nextAnalysis;
  }

  return results;
}

export { MATE_CP };
