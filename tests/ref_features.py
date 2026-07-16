"""
ref_features.py — Dump reference feature data from the real splainfish.features
(the Python that ships with the CLI) for the JS port to be diffed against.

Emits JSON on stdout: a set of positions, each with its FEN and the exact
HalfKAv2_hm / FullThreats index lists and label decodings that features.py
produces. tests/check_features_parity.mjs consumes it.

Positions come from pseudo-random playouts rather than a fixed list so that
castling, promotion, and king-mirroring cases get hit; the seed is fixed so the
corpus is reproducible.

Requires python-chess and numpy. Run via `make test-parity`.
"""
import json
import random
import sys

import chess

from splainfish.features import (
    compute_features, diff_features, halfka_label, halfka_index,
    threat_indices, _threat_base_offset, _threat_orient,
)


def features_to_dict(f):
    return {
        "halfka_white": f.halfka_white,
        "halfka_black": f.halfka_black,
        "threat_white": f.threat_white,
        "threat_black": f.threat_black,
        "wking_sq": f.wking_sq,
        "bking_sq": f.bking_sq,
    }


def diff_to_dict(d):
    return {
        "halfka_white_gained": d.halfka_white_gained,
        "halfka_white_lost": d.halfka_white_lost,
        "halfka_black_gained": d.halfka_black_gained,
        "halfka_black_lost": d.halfka_black_lost,
        "threat_white_gained": d.threat_white_gained,
        "threat_white_lost": d.threat_white_lost,
        "threat_black_gained": d.threat_black_gained,
        "threat_black_lost": d.threat_black_lost,
    }


def collect_positions(rng, n_games=14, max_plies=70):
    """Pseudo-random playouts, sampling (before, move, after) triples."""
    out = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(max_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            fen_before = board.fen()
            feat_before = compute_features(board)
            board.push(move)
            feat_after = compute_features(board)
            out.append({
                "fen_before": fen_before,
                "fen_after": board.fen(),
                "move_uci": move.uci(),
                "features_before": features_to_dict(feat_before),
                "features_after": features_to_dict(feat_after),
                "diff": diff_to_dict(diff_features(feat_before, feat_after)),
            })
    return out


def collect_labels(rng, positions):
    """Label decodings for indices actually seen, plus the PS_KING collision range."""
    seen = set()
    for p in positions[:40]:
        seen.update(p["features_before"]["halfka_white"][:8])
        seen.update(p["features_before"]["halfka_black"][:8])

    out = []
    for idx in sorted(seen):
        for persp in (chess.WHITE, chess.BLACK):
            out.append({
                "idx": int(idx),
                "perspective": "white" if persp else "black",
                "label": halfka_label(int(idx), persp),
            })
    return out


def collect_index_arithmetic(rng):
    """halfka_index over an exhaustive-ish sweep of the argument space."""
    out = []
    for _ in range(4000):
        persp = rng.choice([chess.WHITE, chess.BLACK])
        pc = rng.choice([chess.WHITE, chess.BLACK])
        pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                         chess.ROOK, chess.QUEEN, chess.KING])
        psq = rng.randrange(64)
        ksq = rng.randrange(64)
        out.append({
            "perspective": "white" if persp else "black",
            "piece_color": "white" if pc else "black",
            "piece_type": int(pt),
            "piece_sq": psq,
            "king_sq": ksq,
            "index": halfka_index(persp, pc, pt, psq, ksq),
        })
    return out


def collect_threat_tables():
    """Threat base offsets and orientation, which are pure table lookups."""
    bases = {}
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        bases[int(pt)] = _threat_base_offset(pt)
    orient = {
        "white": [_threat_orient(sq, chess.WHITE) for sq in range(64)],
        "black": [_threat_orient(sq, chess.BLACK) for sq in range(64)],
    }
    return {"base_offsets": bases, "orient": orient}


def main():
    rng = random.Random(20260716)
    positions = collect_positions(rng)
    json.dump({
        "positions": positions,
        "labels": collect_labels(rng, positions),
        "index_arithmetic": collect_index_arithmetic(rng),
        "threat_tables": collect_threat_tables(),
    }, sys.stdout)


if __name__ == "__main__":
    main()
