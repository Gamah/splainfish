"""
features.py — Compute HalfKAv2_hm and FullThreats input feature indices.

Mirrors the logic in:
  src/nnue/features/half_ka_v2_hm.h  (make_index)
  src/nnue/features/full_threats.h   (make_index)

Given a chess.Board, returns the set of active feature indices for each
king perspective (WHITE, BLACK).  The diff of two such sets is the exact
list of features that changed between two positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import chess
import numpy as np


# ---------------------------------------------------------------------------
# HalfKAv2_hm constants (mirrored from SF18 source)
# ---------------------------------------------------------------------------

SQUARE_NB = 64

# Piece-square offsets (PS_*) — indexed as (piece_type, color) from perspective
PS_W_PAWN   = 0 * SQUARE_NB
PS_B_PAWN   = 1 * SQUARE_NB
PS_W_KNIGHT = 2 * SQUARE_NB
PS_B_KNIGHT = 3 * SQUARE_NB
PS_W_BISHOP = 4 * SQUARE_NB
PS_B_BISHOP = 5 * SQUARE_NB
PS_W_ROOK   = 6 * SQUARE_NB
PS_B_ROOK   = 7 * SQUARE_NB
PS_W_QUEEN  = 8 * SQUARE_NB
PS_B_QUEEN  = 9 * SQUARE_NB
PS_KING     = 10 * SQUARE_NB
PS_NB       = 11 * SQUARE_NB   # 704

# PieceSquareIndex[perspective][piece] from half_ka_v2_hm.h
# perspective WHITE: W pieces are "us", B pieces are "them"
# perspective BLACK: B pieces are "us", W pieces are "them"
_PSI_WHITE = {
    (chess.WHITE, chess.PAWN):   PS_W_PAWN,
    (chess.WHITE, chess.KNIGHT): PS_W_KNIGHT,
    (chess.WHITE, chess.BISHOP): PS_W_BISHOP,
    (chess.WHITE, chess.ROOK):   PS_W_ROOK,
    (chess.WHITE, chess.QUEEN):  PS_W_QUEEN,
    (chess.WHITE, chess.KING):   PS_KING,
    (chess.BLACK, chess.PAWN):   PS_B_PAWN,
    (chess.BLACK, chess.KNIGHT): PS_B_KNIGHT,
    (chess.BLACK, chess.BISHOP): PS_B_BISHOP,
    (chess.BLACK, chess.ROOK):   PS_B_ROOK,
    (chess.BLACK, chess.QUEEN):  PS_B_QUEEN,
    (chess.BLACK, chess.KING):   PS_KING,
}
# For black's perspective, own/enemy are swapped
_PSI_BLACK = {
    (chess.BLACK, chess.PAWN):   PS_W_PAWN,
    (chess.BLACK, chess.KNIGHT): PS_W_KNIGHT,
    (chess.BLACK, chess.BISHOP): PS_W_BISHOP,
    (chess.BLACK, chess.ROOK):   PS_W_ROOK,
    (chess.BLACK, chess.QUEEN):  PS_W_QUEEN,
    (chess.BLACK, chess.KING):   PS_KING,
    (chess.WHITE, chess.PAWN):   PS_B_PAWN,
    (chess.WHITE, chess.KNIGHT): PS_B_KNIGHT,
    (chess.WHITE, chess.BISHOP): PS_B_BISHOP,
    (chess.WHITE, chess.ROOK):   PS_B_ROOK,
    (chess.WHITE, chess.QUEEN):  PS_B_QUEEN,
    (chess.WHITE, chess.KING):   PS_KING,
}

# KingBuckets[sq] — which of 32 king-position buckets this square maps to
# (from half_ka_v2_hm.h, divided by PS_NB for the bucket index)
_KB_RAW = [
    28,29,30,31, 31,30,29,28,
    24,25,26,27, 27,26,25,24,
    20,21,22,23, 23,22,21,20,
    16,17,18,19, 19,18,17,16,
    12,13,14,15, 15,14,13,12,
     8, 9,10,11, 11,10, 9, 8,
     4, 5, 6, 7,  7, 6, 5, 4,
     0, 1, 2, 3,  3, 2, 1, 0,
]
KING_BUCKETS = _KB_RAW  # index = square (a1=0 … h8=63)

# OrientTBL[sq]: if king is on a..d files (files 0-3), mirror square horizontally.
# SQ_A1=0, SQ_H1=7 used as sentinel for "flip" / "no flip"
# The table tells us what XOR mask to apply to the square:
#   file 0-3 → XOR with 7 (flip file)
#   file 4-7 → XOR with 0 (no flip)
def _orient(sq: int, ksq: int) -> int:
    """Mirror sq horizontally if king is on a..d file."""
    if chess.square_file(ksq) < 4:   # king on queenside → mirror
        return sq ^ 7
    return sq


def halfka_index(perspective: chess.Color, piece_color: chess.Color,
                 piece_type: chess.PieceType, piece_sq: int, king_sq: int) -> int:
    """
    Compute the HalfKAv2_hm feature index for one (perspective, piece, square).

    Index = KingBucket[oriented_ksq] * PS_NB + PieceSquareOffset + oriented_piece_sq
    Total feature space: 32 * PS_NB * 64 / 2 = 45056  (mirroring halves the king axis)
    """
    psi = _PSI_WHITE if perspective == chess.WHITE else _PSI_BLACK
    ps_offset = psi[(piece_color, piece_type)]

    o_ksq = _orient(king_sq, king_sq)
    o_psq = _orient(piece_sq, king_sq)

    bucket = KING_BUCKETS[o_ksq]
    return bucket * PS_NB + ps_offset + o_psq


# ---------------------------------------------------------------------------
# FullThreats feature index computation
# ---------------------------------------------------------------------------
# FullThreats encodes attack relationships: for each (attacker, target_type,
# target_sq) triple that is actually attacked, a feature fires.
# Dimensions = 60720 (from full_threats.h)

# map[attacker_type-1][target_type-1] = sub-index within the attack pair
# -1 means this combination is not encoded
_THREAT_MAP = [
    # target: P  N   B   R   Q   K
    [ 0,  1, -1,  2, -1, -1],   # attacker P
    [ 0,  1,  2,  3,  4, -1],   # attacker N
    [ 0,  1,  2,  3, -1, -1],   # attacker B
    [ 0,  1,  2,  3, -1, -1],   # attacker R
    [ 0,  1,  2,  3,  4, -1],   # attacker Q
    # King attacks not encoded (K row absent)
]
# numValidTargets per attacker (from full_threats.h)
_NUM_TARGETS = [2, 5, 4, 4, 5]   # P,N,B,R,Q

# OrientTBL for threats: file 0-3 → XOR 7, else XOR 0 (same as HalfKA but different table)
def _threat_orient(sq: int, perspective: chess.Color) -> int:
    """Orient square for FullThreats: black perspective flips rank."""
    if perspective == chess.BLACK:
        return sq ^ 56   # flip rank (a1↔a8)
    return sq


def _threat_base_offset(attacker_type: chess.PieceType) -> int:
    """Starting offset in the 60720-dim threat vector for a given attacker type."""
    # Each attacker type occupies (num_targets × 64) entries
    offset = 0
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for p in pieces:
        if p == attacker_type:
            return offset
        offset += _NUM_TARGETS[p - 1] * SQUARE_NB
    return -1   # king doesn't attack (not in threat features)


def threat_indices(board: chess.Board, perspective: chess.Color) -> list[int]:
    """
    Return all active FullThreats feature indices for `perspective`.

    A threat feature fires when a piece of color `perspective` attacks a
    square occupied by any piece (friend or foe), encoding the
    (attacker_type, target_type, target_sq) triple.
    """
    us = perspective
    indices = []

    attacker_pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    target_pieces   = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for att_type in attacker_pieces:
        att_idx = att_type - 1   # 0-indexed for _THREAT_MAP
        base = _threat_base_offset(att_type)

        for att_sq in board.pieces(att_type, us):
            attacks = board.attacks(att_sq)
            for tgt_sq in attacks:
                piece = board.piece_at(tgt_sq)
                if piece is None:
                    continue
                tgt_type = piece.piece_type
                tgt_idx  = tgt_type - 1
                sub = _THREAT_MAP[att_idx][tgt_idx]
                if sub < 0:
                    continue
                o_tgt = _threat_orient(tgt_sq, perspective)
                idx = base + sub * SQUARE_NB + o_tgt
                indices.append(idx)

    return indices


# ---------------------------------------------------------------------------
# Active feature sets for a full position
# ---------------------------------------------------------------------------

@dataclass
class PositionFeatures:
    """Active HalfKA and Threat feature indices for both perspectives."""
    halfka_white: list[int]   # HalfKAv2_hm active indices, white king perspective
    halfka_black: list[int]   # HalfKAv2_hm active indices, black king perspective
    threat_white: list[int]   # FullThreats active indices, white perspective
    threat_black: list[int]   # FullThreats active indices, black perspective
    wking_sq: int
    bking_sq: int


def compute_features(board: chess.Board) -> PositionFeatures:
    """Compute all active input features for a position."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    halfka_w = []
    halfka_b = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pc, pt = piece.color, piece.piece_type

        halfka_w.append(halfka_index(chess.WHITE, pc, pt, sq, wk))
        halfka_b.append(halfka_index(chess.BLACK, pc, pt, sq, bk))

    return PositionFeatures(
        halfka_white=halfka_w,
        halfka_black=halfka_b,
        threat_white=threat_indices(board, chess.WHITE),
        threat_black=threat_indices(board, chess.BLACK),
        wking_sq=wk,
        bking_sq=bk,
    )


# ---------------------------------------------------------------------------
# Feature diff
# ---------------------------------------------------------------------------

@dataclass
class FeatureDiff:
    """
    Exact diff of active features between two positions, per perspective.
    gained = features newly active in position B
    lost   = features that were active in A but not in B
    """
    halfka_white_gained: list[int]
    halfka_white_lost:   list[int]
    halfka_black_gained: list[int]
    halfka_black_lost:   list[int]
    threat_white_gained: list[int]
    threat_white_lost:   list[int]
    threat_black_gained: list[int]
    threat_black_lost:   list[int]


def diff_features(before: PositionFeatures, after: PositionFeatures) -> FeatureDiff:
    def _diff(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
        sa, sb = set(a), set(b)
        return sorted(sb - sa), sorted(sa - sb)   # gained, lost

    wg, wl = _diff(before.halfka_white, after.halfka_white)
    bg, bl = _diff(before.halfka_black, after.halfka_black)
    twg, twl = _diff(before.threat_white, after.threat_white)
    tbg, tbl = _diff(before.threat_black, after.threat_black)

    return FeatureDiff(
        halfka_white_gained=wg, halfka_white_lost=wl,
        halfka_black_gained=bg, halfka_black_lost=bl,
        threat_white_gained=twg, threat_white_lost=twl,
        threat_black_gained=tbg, threat_black_lost=tbl,
    )


# ---------------------------------------------------------------------------
# Human-readable feature label helpers
# ---------------------------------------------------------------------------

_PIECE_NAME = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen",  chess.KING: "king",
}
_COLOR_NAME = {chess.WHITE: "White", chess.BLACK: "Black"}


def halfka_label(idx: int, perspective: chess.Color) -> dict:
    """
    Decode a HalfKAv2_hm feature index into a human-readable dict.
    Returns: {king_bucket, piece_color, piece_type, piece_sq, perspective}
    """
    bucket = idx // PS_NB
    remainder = idx % PS_NB

    # Find piece type and color from remainder
    piece_color = None
    piece_type  = None
    piece_sq_oriented = None

    psi = _PSI_WHITE if perspective == chess.WHITE else _PSI_BLACK
    for (pc, pt), offset in psi.items():
        if offset <= remainder < offset + SQUARE_NB:
            piece_color = pc
            piece_type  = pt
            piece_sq_oriented = remainder - offset
            break

    return {
        "king_bucket":      bucket,
        "piece_color":      _COLOR_NAME.get(piece_color, "?"),
        "piece_type":       _PIECE_NAME.get(piece_type, "?"),
        "piece_sq":         chess.square_name(piece_sq_oriented) if piece_sq_oriented is not None else "?",
        "perspective":      _COLOR_NAME.get(perspective, "?"),
    }
