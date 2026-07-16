"""
ref_realnet.py — Reference fingerprint of a real .nnue parsed by the Python.

Loads an actual network with splainfish.nnue_parser and emits a compact JSON
fingerprint (shapes, checksums, and sampled values) that the JS parser is
checked against. This is the end-to-end parser test on real weights, as opposed
to ref_parser.py which covers the LEB128/permutation primitives in isolation.

Usage:  python3 tests/ref_realnet.py <path-to.nnue>

Requires numpy. Run via `make test-realnet`.
"""
import hashlib
import json
import sys

import chess
import numpy as np

from splainfish.nnue_parser import load
from splainfish.features import compute_features
from splainfish.probe import forward

# A few positions to evaluate end-to-end with the real net.
FORWARD_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


def checksum(arr):
    """Order-sensitive checksum of a typed array, stable across platforms."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def sample(arr, n=32, seed=12345):
    """Deterministic (index, value) samples for a spot-check independent of the sum."""
    flat = np.ascontiguousarray(arr).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, flat.size, size=min(n, flat.size))
    return [[int(i), int(flat[i])] for i in sorted(idx)]


def main():
    if len(sys.argv) != 2:
        print("usage: ref_realnet.py <path.nnue>", file=sys.stderr)
        sys.exit(2)

    w = load(sys.argv[1])
    ft = w.feature_transformer

    out = {
        "arch": w.arch.name,
        "l1": w.l1,
        "halfka_dims": w.halfka_dims,
        "has_threats": w.has_threats,
        "description": w.description[:80],
        "ft": {
            "biases_shape": list(ft.biases.shape),
            "biases_sum": int(ft.biases.astype(np.int64).sum()),
            "biases_checksum": checksum(ft.biases),
            "biases_sample": sample(ft.biases),
            "weights_shape": list(ft.weights.shape),
            "weights_sum": int(ft.weights.astype(np.int64).sum()),
            "weights_checksum": checksum(ft.weights),
            "weights_sample": sample(ft.weights, n=64),
            "psqt_sum": int(ft.psqt_weights.astype(np.int64).sum()),
            "psqt_checksum": checksum(ft.psqt_weights),
        },
        "stacks": [],
    }

    for s in w.layer_stacks:
        out["stacks"].append({
            "fc0_biases_sum": int(s.fc0_biases.astype(np.int64).sum()),
            "fc0_weights_checksum": checksum(s.fc0_weights),
            "fc0_weights_sample": sample(s.fc0_weights),
            "fc1_biases_sum": int(s.fc1_biases.astype(np.int64).sum()),
            "fc1_weights_checksum": checksum(s.fc1_weights),
            "fc2_biases_sum": int(s.fc2_biases.astype(np.int64).sum()),
            "fc2_weights_checksum": checksum(s.fc2_weights),
        })

    # End-to-end forward pass on the real net for a handful of positions.
    out["forward"] = []
    for fen in FORWARD_FENS:
        board = chess.Board(fen)
        act = forward(board, compute_features(board), w)
        out["forward"].append({
            "fen": fen,
            "centipawns": int(act.centipawns),
            "fc0_pre_sample": [float(x) for x in act.fc0_pre[:8]],
            "skip": float(act.skip),
        })

    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
