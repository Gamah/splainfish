"""
probe.py — NNUE forward pass re-implementation and activation probing.

Supports both SF16 and SF18 network formats (detected automatically from
the loaded NNUEWeights version field).

SF16 forward pass:
  FT accumulator: i16 accumulated → clip [0,255] → uint8
  fc_0: (L1=1536 uint8) × (i8 weights) → i32 → SqrClippedReLU[0:15] + ClippedReLU[0:15] → 30
  fc_1: (30 uint8, pad→32) × (i8 weights) → i32 → ClippedReLU → 32
  fc_2: (32 uint8) × (i8 weights) → i32
  Skip: fc_0_raw[15] * (600*OutputScale) / (127*(1<<WeightScaleBits))
  Output: fc_2[0] + skip → scale to centipawns

SF18 forward pass:
  FT accumulator: i16 accumulated → clip [0,255] → uint8 (L1=1024 per perspective, concat → 2048)
  fc_0: (2048 uint8) × (i8 weights) → i32 → SqrClippedReLU + ClippedReLU → 64
  fc_1: (64 uint8) × (i8 weights) → i32 → SqrClippedReLU + ClippedReLU → 64
  fc_2: (128 uint8) × (i8 weights) → i32
  Skip: fc_0_raw[-2] - fc_0_raw[-1]
  Output: (fc_2[0] + skip) * 600*OutputScale / (HiddenOneVal*(1<<WeightScaleBits)*2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import chess
import numpy as np

from .nnue_parser import (
    NNUEWeights, LayerStackWeights, FeatureTransformerWeights,
    VERSION_SF16, VERSION_SF18,
    WEIGHT_SCALE_BITS, WEIGHT_SCALE, HIDDEN_ONE_VAL,
    OUTPUT_SCALE, FT_MAX_VAL, PSQT_BUCKETS, LAYER_STACKS,
    SF16_L1, SF16_FC0_OUT, SF16_FC1_OUT,
    SF18_L1, SF18_L2, SF18_L3,
)
from .features import PositionFeatures, FeatureDiff, halfka_label


# ---------------------------------------------------------------------------
# King bucket → LayerStack selection
# Same table for both SF16 and SF18 (8 stacks from 64 squares)
# ---------------------------------------------------------------------------
_KING_LAYER_BUCKET = [
     0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,
     8,  9, 10, 11, 11, 10,  9,  8,
     8,  9, 10, 11, 11, 10,  9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
]

def _layer_stack_index(board: chess.Board) -> int:
    ksq = board.king(board.turn)
    if chess.square_file(ksq) < 4:
        ksq = ksq ^ 7
    return _KING_LAYER_BUCKET[ksq] % LAYER_STACKS


# ---------------------------------------------------------------------------
# Forward pass primitives
# ---------------------------------------------------------------------------

def _accumulate_ft(features: PositionFeatures, ft: FeatureTransformerWeights,
                   perspective: chess.Color, clamp_max: int = FT_MAX_VAL) -> np.ndarray:
    """Compute FT accumulator for one perspective, return clipped float64."""
    acc = ft.biases.astype(np.int32)
    halfka = features.halfka_white if perspective == chess.WHITE else features.halfka_black
    for idx in halfka:
        acc += ft.weights[idx].astype(np.int32)
    if ft.threat_weights is not None:
        threats = features.threat_white if perspective == chess.WHITE else features.threat_black
        for idx in threats:
            acc += ft.threat_weights[idx].astype(np.int32)
    return np.clip(acc, 0, clamp_max).astype(np.float64)


def _sqr_clipped_relu(x: np.ndarray, scale_bits: int) -> np.ndarray:
    scale = float(1 << (scale_bits + 1))
    clipped = np.clip(x, 0.0, scale - 1)
    return (clipped * clipped) / (scale * scale)


def _clipped_relu(x: np.ndarray, scale_bits: int) -> np.ndarray:
    scale = float(1 << (scale_bits + 1))
    return np.clip(x, 0.0, scale - 1) / scale


def _fc(x: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return weights.astype(np.float64) @ x + biases.astype(np.float64)


# ---------------------------------------------------------------------------
# Activation containers
# ---------------------------------------------------------------------------

@dataclass
class Activations:
    ft_acc_white: np.ndarray   # (L1,)
    ft_acc_black: np.ndarray   # (L1,)
    ft_out:       np.ndarray   # (L1 or L1*2,) — the input to fc_0

    fc0_pre:    np.ndarray
    fc0_sqr:    np.ndarray
    fc0_lin:    np.ndarray
    fc0_concat: np.ndarray

    fc1_pre:    np.ndarray
    fc1_out:    np.ndarray     # after activation (lin only for SF16, lin for SF18)
    fc1_sqr:    Optional[np.ndarray]   # None for SF16

    fc2_pre:    np.ndarray
    skip:       float
    centipawns: int


def _forward_sf16(board: chess.Board, features: PositionFeatures,
                  weights: NNUEWeights) -> Activations:
    """SF16 forward pass."""
    stack = weights.layer_stacks[_layer_stack_index(board)]
    ft = weights.feature_transformer
    is_white = board.turn == chess.WHITE

    # FT: each perspective outputs L1/2=768 values after clipping+packing
    # The transform packs both perspectives into a 1536-element buffer:
    #   [us[0:768] | them[0:768]] (each half is [lower|upper] of one perspective)
    acc_w = _accumulate_ft(features, ft, chess.WHITE, clamp_max=127)  # (1536,) SF16 clips to 127
    acc_b = _accumulate_ft(features, ft, chess.BLACK, clamp_max=127)

    H = SF16_L1 // 2  # 768

    def _ft_fold(acc: np.ndarray) -> np.ndarray:
        """
        SF16 FT squaring: for each j in 0..767:
          out[j] = clip(acc[j], 0, 127) * clip(acc[j+H], 0, 127) / 128
        Returns 768-element float64 array.
        """
        lo = np.clip(acc[:H], 0.0, 127.0)
        hi = np.clip(acc[H:], 0.0, 127.0)
        return lo * hi / 128.0

    us_fold  = _ft_fold(acc_w if is_white else acc_b)   # (768,)
    them_fold = _ft_fold(acc_b if is_white else acc_w)  # (768,)
    ft_out = np.concatenate([us_fold, them_fold])        # (1536,)

    # fc_0: in=1536 → out=16 (FC_0_OUTPUTS+1)
    fc0_pre = _fc(ft_out, stack.fc0_weights, stack.fc0_biases)
    # SqrClippedReLU on [0:15], ClippedReLU on [0:15], skip uses [15]
    fc0_sqr = _sqr_clipped_relu(fc0_pre[:SF16_FC0_OUT-1], WEIGHT_SCALE_BITS + 1)
    fc0_lin = _clipped_relu(fc0_pre[:SF16_FC0_OUT-1], WEIGHT_SCALE_BITS + 1)
    fc0_concat = np.concatenate([fc0_sqr, fc0_lin])   # (30,)

    # fc_1: in=30(pad32) → out=32
    fc1_pre = _fc(fc0_concat, stack.fc1_weights, stack.fc1_biases)
    fc1_out = _clipped_relu(fc1_pre, WEIGHT_SCALE_BITS)   # (32,)

    # fc_2: in=32 → out=1
    fc2_pre = _fc(fc1_out, stack.fc2_weights, stack.fc2_biases)

    # Skip: fc_0_raw[FC_0_OUTPUTS] (index 15) * (600*OutputScale)/(127*(1<<WeightScaleBits))
    skip_raw = float(fc0_pre[SF16_FC0_OUT - 1])
    skip = skip_raw * (600 * OUTPUT_SCALE) / (127 * WEIGHT_SCALE)

    raw_out = float(fc2_pre[0]) + skip
    # Scale to centipawns: multiply by 600*OutputScale / (127*(1<<WeightScaleBits))
    # Actually fc2 output already includes the skip in integer domain;
    # the final scaling is the same as sf18 but with a different denominator.
    # From SF16 architecture propagate():
    #   fwdOut = fc_0_out[FC_0_OUTPUTS] * (600*OutputScale) / (127*(1<<WeightScaleBits))
    #   outputValue = fc_2_out[0] + fwdOut
    # fc_2_out is already in the quantized output space:
    #   real_value = fc_2_out * 600*OutputScale / (127*(1<<WeightScaleBits)*something)
    # This is complex; approximate with the SF18 scaling as proportional.
    denominator = 127 * WEIGHT_SCALE  # 127 * 64 = 8128
    cp = int(raw_out * 600 * OUTPUT_SCALE / denominator)
    if not is_white:
        cp = -cp

    return Activations(
        ft_acc_white=acc_w, ft_acc_black=acc_b, ft_out=ft_out,
        fc0_pre=fc0_pre, fc0_sqr=fc0_sqr, fc0_lin=fc0_lin, fc0_concat=fc0_concat,
        fc1_pre=fc1_pre, fc1_out=fc1_out, fc1_sqr=None,
        fc2_pre=fc2_pre, skip=skip, centipawns=cp,
    )


def _forward_sf18(board: chess.Board, features: PositionFeatures,
                  weights: NNUEWeights) -> Activations:
    """SF18 forward pass."""
    stack = weights.layer_stacks[_layer_stack_index(board)]
    ft = weights.feature_transformer
    is_white = board.turn == chess.WHITE

    acc_w = _accumulate_ft(features, ft, chess.WHITE)   # (1024,)
    acc_b = _accumulate_ft(features, ft, chess.BLACK)   # (1024,)

    if is_white:
        ft_out = np.concatenate([acc_w, acc_b])
    else:
        ft_out = np.concatenate([acc_b, acc_w])

    # fc_0: (2048) → 32
    fc0_pre = _fc(ft_out, stack.fc0_weights, stack.fc0_biases)
    fc0_sqr = _sqr_clipped_relu(fc0_pre, WEIGHT_SCALE_BITS + 1)
    fc0_lin = _clipped_relu(fc0_pre, WEIGHT_SCALE_BITS + 1)
    fc0_concat = np.concatenate([fc0_sqr, fc0_lin])   # (64,)

    # fc_1: (64) → 32
    fc1_pre = _fc(fc0_concat, stack.fc1_weights, stack.fc1_biases)
    fc1_sqr = _sqr_clipped_relu(fc1_pre, WEIGHT_SCALE_BITS)
    fc1_lin = _clipped_relu(fc1_pre, WEIGHT_SCALE_BITS)
    fc1_out = np.concatenate([fc1_sqr, fc1_lin])   # (64,)

    # fc_2: (128) → 1
    concat_all = np.concatenate([fc0_concat, fc1_out])   # (128,)
    fc2_pre = _fc(concat_all, stack.fc2_weights, stack.fc2_biases)

    # Skip: fc_0_raw[-2] - fc_0_raw[-1]
    skip = float(fc0_pre[SF18_L2 - 2] - fc0_pre[SF18_L2 - 1])
    raw_out = float(fc2_pre[0]) + skip

    numerator   = 600 * OUTPUT_SCALE
    denominator = HIDDEN_ONE_VAL * WEIGHT_SCALE * 2   # 16384
    cp = int(raw_out * numerator / denominator)
    if not is_white:
        cp = -cp

    return Activations(
        ft_acc_white=acc_w, ft_acc_black=acc_b, ft_out=ft_out,
        fc0_pre=fc0_pre, fc0_sqr=fc0_sqr, fc0_lin=fc0_lin, fc0_concat=fc0_concat,
        fc1_pre=fc1_pre, fc1_out=fc1_out, fc1_sqr=fc1_sqr,
        fc2_pre=fc2_pre, skip=skip, centipawns=cp,
    )


def forward(board: chess.Board, features: PositionFeatures,
            weights: NNUEWeights) -> Activations:
    if weights.version == VERSION_SF16:
        return _forward_sf16(board, features, weights)
    else:
        return _forward_sf18(board, features, weights)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

@dataclass
class FeatureAttribution:
    feature_idx:  int
    feature_type: str        # "halfka" or "threat"
    direction:    str        # "gained" or "lost"
    perspective:  chess.Color
    piece_color:  str
    piece_type:   str
    piece_sq:     str
    king_bucket:  int
    contribution: float


@dataclass
class GroupedAttribution:
    group:         str
    contribution:  float
    feature_count: int
    direction:     str
    features:      list[FeatureAttribution] = field(default_factory=list)


@dataclass
class ProbeResult:
    act_before:    Activations
    act_after:     Activations
    eval_before_cp: int
    eval_after_cp:  int
    delta_cp:       int
    ft_delta_white: np.ndarray
    ft_delta_black: np.ndarray
    fc0_delta:      np.ndarray
    fc1_delta:      np.ndarray
    feature_attributions: list[FeatureAttribution]
    grouped_attributions: list[GroupedAttribution]
    sf_eval_before: Optional[int] = None
    sf_eval_after:  Optional[int] = None


def _back_project(act_b: Activations, act_a: Activations,
                  stack: LayerStackWeights, weights: NNUEWeights) -> np.ndarray:
    """
    Back-project the output delta through fc2→fc1→fc0→FT.
    Returns attribution vector of shape (L1,) for each perspective separately.
    Uses linear approximation (first-order Taylor) at the midpoint activations.
    """
    is_sf16 = weights.version == VERSION_SF16
    l1 = weights.l1

    # --- fc2 gradient w.r.t. concat_all input ---
    w2 = stack.fc2_weights[0].astype(np.float64)   # (fc2_in,)

    if is_sf16:
        # SF16: fc_1 output (32,) is direct input to fc_2.
        # Back-project: output delta → fc1_out → fc0_concat → ft_out
        delta_fc1_out = act_a.fc1_out - act_b.fc1_out           # (32,)
        delta_fc0_sqr = act_a.fc0_sqr - act_b.fc0_sqr           # (15,)
        delta_fc0_lin = act_a.fc0_lin - act_b.fc0_lin           # (15,)
        delta_fc0_concat = np.concatenate([delta_fc0_sqr, delta_fc0_lin])  # (30,)

        # fc2 weights: (1, 32) → (32,)
        w2_full = w2[:SF16_FC1_OUT]   # (32,)

        # Back-project through fc1 (30→32): approx gradient at fc0_concat level
        w1 = stack.fc1_weights.astype(np.float64)   # (32, 30)
        # d(fc1_out)/d(fc0_concat) ≈ w1.T, weighted by w2 and relu jacobian
        mid_fc1 = (act_b.fc1_pre + act_a.fc1_pre) / 2.0
        scale = float(1 << (WEIGHT_SCALE_BITS + 1))
        jac1 = ((mid_fc1 >= 0) & (mid_fc1 < scale - 1)).astype(np.float64)
        g_fc0_via_fc1 = w1.T @ (w2_full * delta_fc1_out * jac1)   # (30,)

        total_fc0 = g_fc0_via_fc1 + delta_fc0_concat              # (30,)

        # Back-project through fc0 linear branch (15 weights from 1536 inputs)
        # fc0_weights[:15] is the linear branch (indices 0..14)
        # fc0_weights[15] is the skip neuron — excluded from concat
        w0_lin = stack.fc0_weights[:SF16_FC0_OUT-1].astype(np.float64)   # (15, 1536)
        # Use linear half of total_fc0 (second 15 elements)
        g_ft = w0_lin.T @ total_fc0[SF16_FC0_OUT-1:]   # (1536,)
        return g_ft
    else:
        # SF18
        L2, L3 = SF18_L2, SF18_L3
        w2_fc0 = w2[:L2 * 2]
        w2_fc1 = w2[L2 * 2:]
        delta_fc0_sqr = act_a.fc0_sqr - act_b.fc0_sqr
        delta_fc0_lin = act_a.fc0_lin - act_b.fc0_lin
        delta_fc0_concat = np.concatenate([delta_fc0_sqr, delta_fc0_lin])

        delta_fc1_sqr = (act_a.fc1_sqr - act_b.fc1_sqr) if act_a.fc1_sqr is not None else np.zeros(L3)
        delta_fc1_lin = act_a.fc1_out[L3:] - act_b.fc1_out[L3:]
        delta_fc1_concat = np.concatenate([delta_fc1_sqr, delta_fc1_lin])

        w1 = stack.fc1_weights.astype(np.float64)   # (L3, L2*2)
        fc1_lin_weighted = w2_fc1[L3:] * delta_fc1_lin
        delta_fc0_via_fc1 = w1.T @ fc1_lin_weighted

        total_fc0 = (w2_fc0 * delta_fc0_concat
                     + np.concatenate([np.zeros(L2), delta_fc0_via_fc1]))

        w0 = stack.fc0_weights.astype(np.float64)   # (L2, L1*2)
        g_ft = w0.T @ total_fc0[L2:]   # linear branch, shape (L1*2,)
        return g_ft


def _group_label(att: FeatureAttribution, moving_color: chess.Color) -> str:
    pc, pt = att.piece_color, att.piece_type
    mover  = "White" if moving_color == chess.WHITE else "Black"
    enemy  = "Black" if moving_color == chess.WHITE else "White"
    is_own = (pc == mover)
    near   = att.king_bucket <= 7

    if pt == "king":   return "king position"
    if pt == "queen":
        if not is_own and near:  return "enemy queen threatening king area"
        if not is_own:           return "enemy queen activity"
        if near:                 return "own queen near king (defensive)"
        return "own queen activity"
    if pt == "rook":
        if not is_own and near:  return "enemy rook pressure near king"
        if not is_own:           return "enemy rook activity"
        return "own rook activity"
    if pt in ("knight", "bishop"):
        if not is_own and near:  return f"enemy {pt} attacking king zone"
        if not is_own:           return f"enemy {pt} activity"
        return f"own {pt} activity / coordination"
    if pt == "pawn":
        if is_own and near:       return "own pawn shield"
        if not is_own and near:   return "enemy pawn advance near king"
        if is_own:                return "own pawn structure"
        return "enemy pawn structure"
    return "other"


def probe(
    board_before: chess.Board, board_after: chess.Board,
    feat_before: PositionFeatures, feat_after: PositionFeatures,
    feat_diff: FeatureDiff, weights: NNUEWeights,
    moving_color: chess.Color,
    sf_eval_before: Optional[int] = None,
    sf_eval_after: Optional[int] = None,
) -> ProbeResult:
    act_b = forward(board_before, feat_before, weights)
    act_a = forward(board_after,  feat_after,  weights)
    delta_cp = act_a.centipawns - act_b.centipawns

    ft_delta_w  = act_a.ft_acc_white - act_b.ft_acc_white
    ft_delta_bk = act_a.ft_acc_black - act_b.ft_acc_black
    fc0_delta   = act_a.fc0_pre - act_b.fc0_pre
    fc1_delta   = act_a.fc1_pre - act_b.fc1_pre

    stack = weights.layer_stacks[_layer_stack_index(board_before)]
    ft_attr = _back_project(act_b, act_a, stack, weights)   # (L1 or L1*2,)

    l1 = weights.l1
    is_white_turn = board_before.turn == chess.WHITE

    # Split attribution by perspective
    if weights.version == VERSION_SF16:
        # ft_attr is (1536,) for the packed [us_lower|them_lower|us_upper|them_upper]
        # Approximate: first half → us perspective, second → them
        if is_white_turn:
            ft_attr_w, ft_attr_b = ft_attr[:l1//2], ft_attr[l1//2:]
        else:
            ft_attr_b, ft_attr_w = ft_attr[:l1//2], ft_attr[l1//2:]
    else:
        # SF18: ft_attr is (L1*2,), clearly split at L1
        if is_white_turn:
            ft_attr_w, ft_attr_b = ft_attr[:l1], ft_attr[l1:]
        else:
            ft_attr_b, ft_attr_w = ft_attr[:l1], ft_attr[l1:]

    feature_attributions: list[FeatureAttribution] = []

    def _add(changed: list[int], direction: str, perspective: chess.Color,
             attr_vec: np.ndarray):
        ft_w = weights.feature_transformer
        l1 = weights.l1
        H = l1 // 2  # half-dimension (768 for SF16, 512 for SF18)
        is_sf16 = weights.version == VERSION_SF16

        for idx in changed:
            info = halfka_label(idx, perspective)
            w_col = ft_w.weights[idx].astype(np.float64)  # (l1,) e.g. (1536,)

            if is_sf16:
                # attr_vec is (768,) — the folded perspective output.
                # The gradient through the FT fold for neuron j:
                #   d(fold_out[j]) / d(acc[j])     = clip(acc[j+H],0,127)/128
                #   d(fold_out[j]) / d(acc[j+H])   = clip(acc[j],  0,127)/128
                # We approximate with uniform 1/128 and use the dot of w_col
                # projected through the fold Jacobian onto attr_vec.
                # Simple approximation: split w_col into lower and upper halves
                # and dot each with attr_vec, weighted by 1/128.
                w_lo = w_col[:H]    # contributes to fold via upper half gate
                w_hi = w_col[H:]    # contributes to fold via lower half gate
                # attr_vec represents sensitivity of output to fold_out
                contrib = float(np.dot((w_lo + w_hi) / (128.0 * (np.linalg.norm(w_col) + 1e-8)),
                                       attr_vec[:H]))
            else:
                w_norm = np.linalg.norm(w_col) + 1e-8
                contrib = float(np.dot(w_col / w_norm, attr_vec[:l1]))
            if direction == "lost":
                contrib = -contrib
            feature_attributions.append(FeatureAttribution(
                feature_idx=idx, feature_type="halfka",
                direction=direction, perspective=perspective,
                piece_color=info["piece_color"], piece_type=info["piece_type"],
                piece_sq=info["piece_sq"], king_bucket=info["king_bucket"],
                contribution=contrib,
            ))

    _add(feat_diff.halfka_white_gained, "gained", chess.WHITE, ft_attr_w)
    _add(feat_diff.halfka_white_lost,   "lost",   chess.WHITE, ft_attr_w)
    _add(feat_diff.halfka_black_gained, "gained", chess.BLACK, ft_attr_b)
    _add(feat_diff.halfka_black_lost,   "lost",   chess.BLACK, ft_attr_b)

    feature_attributions.sort(key=lambda a: abs(a.contribution), reverse=True)

    groups: dict[str, GroupedAttribution] = {}
    for att in feature_attributions:
        label = _group_label(att, moving_color)
        if label not in groups:
            groups[label] = GroupedAttribution(label, 0.0, 0, "positive")
        groups[label].contribution += att.contribution
        groups[label].feature_count += 1
        groups[label].features.append(att)

    grouped = sorted(groups.values(), key=lambda g: abs(g.contribution), reverse=True)
    for g in grouped:
        g.direction = "positive" if g.contribution >= 0 else "negative"

    return ProbeResult(
        act_before=act_b, act_after=act_a,
        eval_before_cp=act_b.centipawns, eval_after_cp=act_a.centipawns,
        delta_cp=delta_cp,
        ft_delta_white=ft_delta_w, ft_delta_black=ft_delta_bk,
        fc0_delta=fc0_delta, fc1_delta=fc1_delta,
        feature_attributions=feature_attributions,
        grouped_attributions=grouped,
        sf_eval_before=sf_eval_before, sf_eval_after=sf_eval_after,
    )
