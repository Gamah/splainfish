"""
ref_probe.py — Reference data for the probe.js port.

Builds a synthetic NNUEWeights (random but seeded), runs the real
splainfish.probe forward pass and attribution over real positions, and writes:

  <out>/weights-sf16.bin , <out>/weights-sf18.bin
      raw little-endian weight dumps, in the layout documented below. This is
      NOT the .nnue on-disk format -- nnue_parser is covered separately by
      tests/ref_parser.py, so this deliberately bypasses it and feeds probe.py
      directly.

  <out>/probe-ref.json
      per-position expected activations and attributions.

Weight magnitudes are kept small on purpose: full-range int16 weights saturate
every clip in the network, which would make the comparison pass trivially
without exercising the activation boundaries.

Requires python-chess and numpy. Run via `make test-parity`.
"""
import argparse
import json
import pathlib
import random
import sys

import chess
import numpy as np

from splainfish.nnue_parser import (
    NNUEWeights, LayerStackWeights, FeatureTransformerWeights,
    PSQT_BUCKETS, LAYER_STACKS,
    ARCH_HALFKA_1536, ARCH_HALFKA_1024_THREATS,
)

# The two architectures under test, named by their FT style for the dump.
FOLD_ARCH = ARCH_HALFKA_1536            # halfka-1536, no threats
CONCAT_ARCH = ARCH_HALFKA_1024_THREATS  # halfka-1024-threats

SF16_L1, SF16_HALFKA_DIMS = FOLD_ARCH.l1, FOLD_ARCH.halfka_dims
SF16_FC0_OUT, SF16_FC1_OUT = FOLD_ARCH.fc0_out, FOLD_ARCH.fc1_out
SF18_L1, SF18_HALFKA_DIMS = CONCAT_ARCH.l1, CONCAT_ARCH.halfka_dims
SF18_THREAT_DIMS = CONCAT_ARCH.threat_dims
SF18_L2, SF18_L3 = CONCAT_ARCH.fc0_out, CONCAT_ARCH.fc1_out
from splainfish.features import compute_features, diff_features
from splainfish.probe import probe, forward, _layer_stack_index


def rnd_i16(rng, shape, lo, hi):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(np.int16)


def rnd_i8(rng, shape, lo, hi):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(np.int8)


def rnd_i32(rng, shape, lo, hi):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(np.int32)


def build_stack(rng, fc0_in, fc0_out, fc1_in, fc1_out, fc2_in):
    return LayerStackWeights(
        fc0_biases=rnd_i32(rng, fc0_out, -2048, 2048),
        fc0_weights=rnd_i8(rng, (fc0_out, fc0_in), -24, 24),
        fc1_biases=rnd_i32(rng, fc1_out, -2048, 2048),
        fc1_weights=rnd_i8(rng, (fc1_out, fc1_in), -24, 24),
        fc2_biases=rnd_i32(rng, 1, -2048, 2048),
        fc2_weights=rnd_i8(rng, (1, fc2_in), -24, 24),
    )


def build_sf16(rng):
    # Weight magnitudes are chosen so the FT accumulator straddles both clip
    # bounds. Too small and it never reaches the ceiling (127 here, 255 for
    # SF18), leaving the upper clamp untested -- mutation testing caught exactly
    # that with a +-8 range, where the accumulator sits ~10 sigma below the cap.
    ft = FeatureTransformerWeights(
        biases=rnd_i16(rng, SF16_L1, -48, 48),
        weights=rnd_i16(rng, (SF16_HALFKA_DIMS, SF16_L1), -28, 28),
        psqt_weights=np.zeros((SF16_HALFKA_DIMS, PSQT_BUCKETS), np.int32),
        threat_weights=None,
        threat_psqt_weights=None,
    )
    stacks = [
        build_stack(rng, SF16_L1, SF16_FC0_OUT, 30, SF16_FC1_OUT, SF16_FC1_OUT)
        for _ in range(LAYER_STACKS)
    ]
    return NNUEWeights(
        arch=FOLD_ARCH, description="synthetic-sf16",
        feature_transformer=ft, layer_stacks=stacks,
    )


def build_sf18(rng):
    # See build_sf16 on magnitude choice; SF18 clips the accumulator at 255.
    ft = FeatureTransformerWeights(
        biases=rnd_i16(rng, SF18_L1, -48, 48),
        weights=rnd_i16(rng, (SF18_HALFKA_DIMS, SF18_L1), -56, 56),
        psqt_weights=np.zeros((SF18_HALFKA_DIMS, PSQT_BUCKETS), np.int32),
        threat_weights=rnd_i8(rng, (SF18_THREAT_DIMS, SF18_L1), -4, 4),
        threat_psqt_weights=np.zeros((SF18_THREAT_DIMS, PSQT_BUCKETS), np.int32),
    )
    stacks = [
        build_stack(rng, SF18_L1 * 2, SF18_L2, SF18_L2 * 2, SF18_L3,
                    SF18_L2 * 2 + SF18_L3 * 2)
        for _ in range(LAYER_STACKS)
    ]
    return NNUEWeights(
        arch=CONCAT_ARCH, description="synthetic-sf18",
        feature_transformer=ft, layer_stacks=stacks,
    )


def dump_weights(w, path):
    """
    Raw little-endian dump. Layout (all arrays C-order, no padding):

      u32  ft_style_tag         (0 = fold, 1 = concat)
      i16  ft.biases            [l1]
      i16  ft.weights           [halfka_dims * l1]
      u8   has_threats
      i8   ft.threat_weights    [threat_dims * l1]   (only if has_threats)
      for each of 8 layer stacks:
        i32 fc0_biases  [fc0_out]        i8 fc0_weights [fc0_out * fc0_in]
        i32 fc1_biases  [fc1_out]        i8 fc1_weights [fc1_out * fc1_in]
        i32 fc2_biases  [1]              i8 fc2_weights [1 * fc2_in]
    """
    ft = w.feature_transformer
    style_tag = 0 if w.arch.ft_style == "fold" else 1
    with open(path, "wb") as f:
        f.write(np.array([style_tag], dtype="<u4").tobytes())
        f.write(ft.biases.astype("<i2").tobytes())
        f.write(ft.weights.astype("<i2").tobytes())
        f.write(np.array([1 if w.has_threats else 0], dtype=np.uint8).tobytes())
        if w.has_threats:
            f.write(ft.threat_weights.astype(np.int8).tobytes())
        for s in w.layer_stacks:
            f.write(s.fc0_biases.astype("<i4").tobytes())
            f.write(s.fc0_weights.astype(np.int8).tobytes())
            f.write(s.fc1_biases.astype("<i4").tobytes())
            f.write(s.fc1_weights.astype(np.int8).tobytes())
            f.write(s.fc2_biases.astype("<i4").tobytes())
            f.write(s.fc2_weights.astype(np.int8).tobytes())


def collect_positions(rng, n_games, max_plies):
    out = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(max_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            before = board.copy()
            board.push(move)
            out.append((before, board.copy(), move))
    return out


def run_case(weights, before, after, move):
    fb = compute_features(before)
    fa = compute_features(after)
    fd = diff_features(fb, fa)
    r = probe(before, after, fb, fa, fd, weights, before.turn)

    def top_attrs(atts, n=12):
        return [{
            "feature_idx": int(a.feature_idx),
            "direction": a.direction,
            "perspective": "white" if a.perspective else "black",
            "piece_type": a.piece_type,
            "piece_color": a.piece_color,
            "king_bucket": int(a.king_bucket),
            "contribution": float(a.contribution),
        } for a in atts[:n]]

    return {
        "fen_before": before.fen(),
        "fen_after": after.fen(),
        "move_uci": move.uci(),
        "stack_index": int(_layer_stack_index(before)),
        "eval_before_cp": int(r.eval_before_cp),
        "eval_after_cp": int(r.eval_after_cp),
        "delta_cp": int(r.delta_cp),
        "act_before": {
            "fc0_pre": [float(x) for x in r.act_before.fc0_pre],
            "fc1_pre": [float(x) for x in r.act_before.fc1_pre],
            "fc2_pre": [float(x) for x in r.act_before.fc2_pre],
            "skip": float(r.act_before.skip),
            "ft_acc_white_head": [float(x) for x in r.act_before.ft_acc_white[:24]],
            "ft_acc_black_head": [float(x) for x in r.act_before.ft_acc_black[:24]],
        },
        "act_after": {
            "fc0_pre": [float(x) for x in r.act_after.fc0_pre],
            "fc1_pre": [float(x) for x in r.act_after.fc1_pre],
            "fc2_pre": [float(x) for x in r.act_after.fc2_pre],
            "skip": float(r.act_after.skip),
        },
        "n_attributions": len(r.feature_attributions),
        "top_attributions": top_attrs(r.feature_attributions),
        "grouped": [{
            "group": g.group,
            "contribution": float(g.contribution),
            "feature_count": int(g.feature_count),
            "direction": g.direction,
        } for g in r.grouped_attributions],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--plies", type=int, default=6)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(20260716)
    positions = collect_positions(py_rng, args.games, args.plies)
    print(f"positions: {len(positions)}", file=sys.stderr)

    result = {}
    for name, builder in (("sf16", build_sf16), ("sf18", build_sf18)):
        print(f"building {name} weights...", file=sys.stderr)
        w = builder(np.random.default_rng(20260716))
        dump_weights(w, out / f"weights-{name}.bin")
        print(f"  dumped {(out / f'weights-{name}.bin').stat().st_size/1e6:.0f} MB",
              file=sys.stderr)
        cases = []
        for i, (b, a, m) in enumerate(positions):
            cases.append(run_case(w, b, a, m))
            print(f"  {name} case {i+1}/{len(positions)}", file=sys.stderr)
        result[name] = cases

    with open(out / "probe-ref.json", "w") as f:
        json.dump(result, f)
    print(f"wrote {out/'probe-ref.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
