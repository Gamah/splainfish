/**
 * features.js — HalfKAv2_hm and FullThreats input feature indices.
 *
 * Port of splainfish/features.py, which in turn mirrors:
 *   src/nnue/features/half_ka_v2_hm.h  (make_index)
 *   src/nnue/features/full_threats.h   (make_index)
 *
 * Square indexing follows python-chess (a1=0, b1=1, ... h8=63) so the index
 * arithmetic transcribes directly from the Python. chess.js uses algebraic
 * names externally and 0x88 internally, so callers convert at the boundary via
 * boardFromChessJs().
 *
 * Attack generation is implemented here rather than taken from chess.js:
 * FullThreats needs every square a piece attacks including defended friendly
 * pieces, whereas chess.js only reports legal moves.
 */

// ---------------------------------------------------------------------------
// Piece / colour constants (numerically identical to python-chess)
// ---------------------------------------------------------------------------

export const WHITE = 1;
export const BLACK = 0;

export const PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6;

export const SQUARE_NB = 64;

const squareFile = (sq) => sq & 7;
const squareRank = (sq) => sq >> 3;

const FILE_NAMES = 'abcdefgh';
export const squareName = (sq) => FILE_NAMES[squareFile(sq)] + (squareRank(sq) + 1);

// ---------------------------------------------------------------------------
// HalfKAv2_hm constants
// ---------------------------------------------------------------------------

const PS_W_PAWN   = 0 * SQUARE_NB;
const PS_B_PAWN   = 1 * SQUARE_NB;
const PS_W_KNIGHT = 2 * SQUARE_NB;
const PS_B_KNIGHT = 3 * SQUARE_NB;
const PS_W_BISHOP = 4 * SQUARE_NB;
const PS_B_BISHOP = 5 * SQUARE_NB;
const PS_W_ROOK   = 6 * SQUARE_NB;
const PS_B_ROOK   = 7 * SQUARE_NB;
const PS_W_QUEEN  = 8 * SQUARE_NB;
const PS_B_QUEEN  = 9 * SQUARE_NB;
const PS_KING     = 10 * SQUARE_NB;
export const PS_NB = 11 * SQUARE_NB; // 704

/**
 * PieceSquareIndex[perspective][piece], keyed [color][pieceType].
 * From the perspective's point of view, its own pieces map to the PS_W_* slots
 * and the opponent's to PS_B_*.
 */
const PSI = {
  [WHITE]: {
    [WHITE]: {
      [PAWN]: PS_W_PAWN, [KNIGHT]: PS_W_KNIGHT, [BISHOP]: PS_W_BISHOP,
      [ROOK]: PS_W_ROOK, [QUEEN]: PS_W_QUEEN,   [KING]: PS_KING,
    },
    [BLACK]: {
      [PAWN]: PS_B_PAWN, [KNIGHT]: PS_B_KNIGHT, [BISHOP]: PS_B_BISHOP,
      [ROOK]: PS_B_ROOK, [QUEEN]: PS_B_QUEEN,   [KING]: PS_KING,
    },
  },
  [BLACK]: {
    [BLACK]: {
      [PAWN]: PS_W_PAWN, [KNIGHT]: PS_W_KNIGHT, [BISHOP]: PS_W_BISHOP,
      [ROOK]: PS_W_ROOK, [QUEEN]: PS_W_QUEEN,   [KING]: PS_KING,
    },
    [WHITE]: {
      [PAWN]: PS_B_PAWN, [KNIGHT]: PS_B_KNIGHT, [BISHOP]: PS_B_BISHOP,
      [ROOK]: PS_B_ROOK, [QUEEN]: PS_B_QUEEN,   [KING]: PS_KING,
    },
  },
};

/** KingBuckets[sq] — which of 32 king-position buckets a square maps to. */
export const KING_BUCKETS = [
  28, 29, 30, 31, 31, 30, 29, 28,
  24, 25, 26, 27, 27, 26, 25, 24,
  20, 21, 22, 23, 23, 22, 21, 20,
  16, 17, 18, 19, 19, 18, 17, 16,
  12, 13, 14, 15, 15, 14, 13, 12,
   8,  9, 10, 11, 11, 10,  9,  8,
   4,  5,  6,  7,  7,  6,  5,  4,
   0,  1,  2,  3,  3,  2,  1,  0,
];

/** Mirror sq horizontally when the king sits on the a..d files. */
const orient = (sq, ksq) => (squareFile(ksq) < 4 ? sq ^ 7 : sq);

/**
 * HalfKAv2_hm feature index for one (perspective, piece, square).
 * index = KingBucket[oriented_ksq] * PS_NB + PieceSquareOffset + oriented_piece_sq
 */
export function halfkaIndex(perspective, pieceColor, pieceType, pieceSq, kingSq) {
  const psOffset = PSI[perspective][pieceColor][pieceType];
  const oKsq = orient(kingSq, kingSq);
  const oPsq = orient(pieceSq, kingSq);
  return KING_BUCKETS[oKsq] * PS_NB + psOffset + oPsq;
}

// ---------------------------------------------------------------------------
// Attack generation
// ---------------------------------------------------------------------------

const KNIGHT_DELTAS = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING_DELTAS   = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];
const BISHOP_RAYS   = [[1, 1], [1, -1], [-1, -1], [-1, 1]];
const ROOK_RAYS     = [[0, 1], [1, 0], [0, -1], [-1, 0]];

const onBoard = (f, r) => f >= 0 && f < 8 && r >= 0 && r < 8;
const toSq = (f, r) => r * 8 + f;

function steppingAttacks(sq, deltas) {
  const f = squareFile(sq), r = squareRank(sq);
  const out = [];
  for (const [df, dr] of deltas) {
    const nf = f + df, nr = r + dr;
    if (onBoard(nf, nr)) out.push(toSq(nf, nr));
  }
  return out;
}

function slidingAttacks(sq, rays, occupied) {
  const f = squareFile(sq), r = squareRank(sq);
  const out = [];
  for (const [df, dr] of rays) {
    let nf = f + df, nr = r + dr;
    while (onBoard(nf, nr)) {
      const s = toSq(nf, nr);
      out.push(s);
      if (occupied[s]) break; // blocker itself is attacked, nothing beyond it
      nf += df;
      nr += dr;
    }
  }
  return out;
}

/**
 * Squares attacked by the piece standing on `sq`.
 * Matches python-chess Board.attacks(): ignores pins and legality, includes
 * defended friendly pieces, and for pawns returns the capture diagonals
 * regardless of occupancy.
 */
export function attacksFrom(board, sq) {
  const piece = board.pieces[sq];
  if (!piece) return [];
  const { type, color } = piece;

  switch (type) {
    case KNIGHT: return steppingAttacks(sq, KNIGHT_DELTAS);
    case KING:   return steppingAttacks(sq, KING_DELTAS);
    case PAWN: {
      const dr = color === WHITE ? 1 : -1;
      const f = squareFile(sq), r = squareRank(sq);
      const out = [];
      for (const df of [-1, 1]) {
        if (onBoard(f + df, r + dr)) out.push(toSq(f + df, r + dr));
      }
      return out;
    }
    case BISHOP: return slidingAttacks(sq, BISHOP_RAYS, board.occupied);
    case ROOK:   return slidingAttacks(sq, ROOK_RAYS, board.occupied);
    case QUEEN:  return slidingAttacks(sq, BISHOP_RAYS.concat(ROOK_RAYS), board.occupied);
    default:     return [];
  }
}

// ---------------------------------------------------------------------------
// FullThreats
// ---------------------------------------------------------------------------

/**
 * map[attackerType-1][targetType-1] = sub-index within the attack pair.
 * -1 means the combination is not encoded. King attacks are not encoded at all.
 */
const THREAT_MAP = [
  // target:  P   N   B   R   Q   K
  [ 0,  1, -1,  2, -1, -1], // attacker P
  [ 0,  1,  2,  3,  4, -1], // attacker N
  [ 0,  1,  2,  3, -1, -1], // attacker B
  [ 0,  1,  2,  3, -1, -1], // attacker R
  [ 0,  1,  2,  3,  4, -1], // attacker Q
];
const NUM_TARGETS = [2, 5, 4, 4, 5]; // P,N,B,R,Q

const THREAT_ATTACKERS = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN];

/** Black's perspective flips the rank. */
const threatOrient = (sq, perspective) => (perspective === BLACK ? sq ^ 56 : sq);

/** Starting offset in the threat vector for a given attacker type. */
function threatBaseOffset(attackerType) {
  let offset = 0;
  for (const p of THREAT_ATTACKERS) {
    if (p === attackerType) return offset;
    offset += NUM_TARGETS[p - 1] * SQUARE_NB;
  }
  return -1; // king does not attack in the threat features
}

/** All active FullThreats indices for `perspective`. */
export function threatIndices(board, perspective) {
  const indices = [];
  for (const attType of THREAT_ATTACKERS) {
    const attIdx = attType - 1;
    const base = threatBaseOffset(attType);
    for (const attSq of board.pieceSquares(attType, perspective)) {
      for (const tgtSq of attacksFrom(board, attSq)) {
        const target = board.pieces[tgtSq];
        if (!target) continue;
        const sub = THREAT_MAP[attIdx][target.type - 1];
        if (sub < 0) continue;
        indices.push(base + sub * SQUARE_NB + threatOrient(tgtSq, perspective));
      }
    }
  }
  return indices;
}

// ---------------------------------------------------------------------------
// Board adapter
// ---------------------------------------------------------------------------

const CHESSJS_TYPE = { p: PAWN, n: KNIGHT, b: BISHOP, r: ROOK, q: QUEEN, k: KING };

/**
 * Minimal board view over a chess.js instance, in python-chess square indexing.
 * Snapshots the position; it does not track subsequent chess.js mutations.
 */
export function boardFromChessJs(game) {
  const pieces = new Array(64).fill(null);
  const occupied = new Uint8Array(64);

  // chess.js board() is rank 8 first; python-chess indexes rank 1 first.
  const rows = game.board();
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const cell = rows[r][f];
      if (!cell) continue;
      const sq = toSq(f, 7 - r);
      pieces[sq] = {
        type: CHESSJS_TYPE[cell.type],
        color: cell.color === 'w' ? WHITE : BLACK,
      };
      occupied[sq] = 1;
    }
  }

  return {
    pieces,
    occupied,
    turn: game.turn() === 'w' ? WHITE : BLACK,
    pieceSquares(type, color) {
      const out = [];
      for (let sq = 0; sq < 64; sq++) {
        const p = pieces[sq];
        if (p && p.type === type && p.color === color) out.push(sq);
      }
      return out;
    },
    king(color) {
      for (let sq = 0; sq < 64; sq++) {
        const p = pieces[sq];
        if (p && p.type === KING && p.color === color) return sq;
      }
      return null;
    },
  };
}

// ---------------------------------------------------------------------------
// Active feature sets
// ---------------------------------------------------------------------------

/** Active HalfKA and Threat feature indices for both perspectives. */
export function computeFeatures(board) {
  const wk = board.king(WHITE);
  const bk = board.king(BLACK);

  const halfkaWhite = [];
  const halfkaBlack = [];

  for (let sq = 0; sq < 64; sq++) {
    const piece = board.pieces[sq];
    if (!piece) continue;
    halfkaWhite.push(halfkaIndex(WHITE, piece.color, piece.type, sq, wk));
    halfkaBlack.push(halfkaIndex(BLACK, piece.color, piece.type, sq, bk));
  }

  return {
    halfkaWhite,
    halfkaBlack,
    threatWhite: threatIndices(board, WHITE),
    threatBlack: threatIndices(board, BLACK),
    wkingSq: wk,
    bkingSq: bk,
  };
}

// ---------------------------------------------------------------------------
// Feature diff
// ---------------------------------------------------------------------------

/** gained = newly active in b; lost = active in a but not b. Both sorted. */
function diffPair(a, b) {
  const sa = new Set(a);
  const sb = new Set(b);
  const gained = [...sb].filter((x) => !sa.has(x)).sort((x, y) => x - y);
  const lost   = [...sa].filter((x) => !sb.has(x)).sort((x, y) => x - y);
  return { gained, lost };
}

/** Exact diff of active features between two positions, per perspective. */
export function diffFeatures(before, after) {
  const w  = diffPair(before.halfkaWhite, after.halfkaWhite);
  const b  = diffPair(before.halfkaBlack, after.halfkaBlack);
  const tw = diffPair(before.threatWhite, after.threatWhite);
  const tb = diffPair(before.threatBlack, after.threatBlack);

  return {
    halfkaWhiteGained: w.gained, halfkaWhiteLost: w.lost,
    halfkaBlackGained: b.gained, halfkaBlackLost: b.lost,
    threatWhiteGained: tw.gained, threatWhiteLost: tw.lost,
    threatBlackGained: tb.gained, threatBlackLost: tb.lost,
  };
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

const PIECE_NAME = {
  [PAWN]: 'pawn', [KNIGHT]: 'knight', [BISHOP]: 'bishop',
  [ROOK]: 'rook', [QUEEN]: 'queen',   [KING]: 'king',
};
const COLOR_NAME = { [WHITE]: 'White', [BLACK]: 'Black' };

/** Decode a HalfKAv2_hm index back into a human-readable descriptor. */
export function halfkaLabel(idx, perspective) {
  const bucket = Math.floor(idx / PS_NB);
  const remainder = idx % PS_NB;

  let pieceColor = null;
  let pieceType = null;
  let pieceSqOriented = null;

  const table = PSI[perspective];

  // Both kings share the PS_KING offset, so this lookup is ambiguous and the
  // first match wins. The Python resolves it by dict insertion order, and
  // _PSI_WHITE lists White's pieces first while _PSI_BLACK lists Black's first
  // — i.e. the perspective's own colour is always tried first. Match that, or
  // king features get labelled with the wrong colour from Black's perspective.
  const colorOrder = perspective === WHITE ? [WHITE, BLACK] : [BLACK, WHITE];

  outer:
  for (const color of colorOrder) {
    for (const type of [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]) {
      const offset = table[color][type];
      if (offset <= remainder && remainder < offset + SQUARE_NB) {
        pieceColor = color;
        pieceType = type;
        pieceSqOriented = remainder - offset;
        break outer;
      }
    }
  }

  return {
    kingBucket: bucket,
    pieceColor: COLOR_NAME[pieceColor] ?? '?',
    pieceType: PIECE_NAME[pieceType] ?? '?',
    pieceSq: pieceSqOriented !== null ? squareName(pieceSqOriented) : '?',
    perspective: COLOR_NAME[perspective] ?? '?',
  };
}
