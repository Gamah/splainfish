"""
nnue_parser.py — Parse a Stockfish .nnue file and expose weights for probing.

Supports two network versions discovered from source:

  SF16  (version 0x7AF32F20)
  ---------------------------------------------------------------
  Feature transformer: HalfKAv2_hm only (no FullThreats)
    L1 = 1536, HALFKA_DIMS = 22528, PSQT_BUCKETS = 8
    FT storage: COMPRESSED_LEB128 for biases(i16), weights(i16), psqt(i32)
  Architecture: FC_0_OUTPUTS=15 (+1 skip=16), FC_1_OUTPUTS=32
    fc_0: in=1536 → out=16   (SqrClippedReLU + ClippedReLU concat → 30)
    fc_1: in=30 (pad→32) → out=32  (ClippedReLU → 32)
    fc_2: in=32 → out=1
    Skip: fc_0_out[15] * scalar into output
  8 LayerStacks, each 25832 bytes; last stack may be truncated (apt export bug)
    → truncated stack is replaced with a copy of stack 6

  SF18  (version 0x6A448AFA)
  ---------------------------------------------------------------
  Feature transformer: HalfKAv2_hm + FullThreats
    L1 = 1024, HALFKA_DIMS = 45056, THREAT_DIMS = 60720, PSQT_BUCKETS = 8
    FT storage: LEB128 ("LEB128 " magic, 7 bytes)
  Architecture: L2=32, L3=32
    fc_0: in=1024*2 → out=32   (SqrClippedReLU + ClippedReLU → 64)
    fc_1: in=64 → out=32       (SqrClippedReLU + ClippedReLU → 64)
    fc_2: in=64+64 → out=1
    Skip: fc_0_pre[-2] - fc_0_pre[-1] added before scaling
  8 LayerStacks; weights stored with LEB128 in FT, raw i8 in fc layers

Quantization constants (same for both):
  WeightScaleBits = 6, WEIGHT_SCALE = 64
  HiddenOneVal = 128, OutputScale = 16, FT_MAX_VAL = 255
"""

from __future__ import annotations

import struct
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional
import copy

import numpy as np


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
WEIGHT_SCALE_BITS = 6
WEIGHT_SCALE      = 1 << WEIGHT_SCALE_BITS   # 64
HIDDEN_ONE_VAL    = 128
OUTPUT_SCALE      = 16
FT_MAX_VAL        = 255
PSQT_BUCKETS      = 8
LAYER_STACKS      = 8

# Version magic bytes
VERSION_SF16 = 0x7AF32F20
VERSION_SF18 = 0x6A448AFA

# SF16 architecture
SF16_L1          = 1536
SF16_HALFKA_DIMS = 22528
SF16_FC0_OUT     = 16    # FC_0_OUTPUTS+1 (15+1)
SF16_FC1_OUT     = 32

# SF18 architecture
SF18_L1          = 1024
SF18_HALFKA_DIMS = 45056
SF18_THREAT_DIMS = 60720
SF18_L2          = 32
SF18_L3          = 32

# LEB128 magic strings
LEB128_MAGIC_SF16 = b"COMPRESSED_LEB128"
LEB128_MAGIC_SF18 = b"LEB128 "


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]

def _read_array_le(f: BinaryIO, dtype: np.dtype, count: int) -> np.ndarray:
    raw = f.read(count * dtype.itemsize)
    if len(raw) < count * dtype.itemsize:
        raise EOFError(f"Expected {count * dtype.itemsize} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=dtype).copy()

def _read_leb128_sf16(f: BinaryIO, dtype: np.dtype, count: int) -> np.ndarray:
    """Read COMPRESSED_LEB128-encoded array (SF16 format)."""
    magic = f.read(len(LEB128_MAGIC_SF16))
    if magic != LEB128_MAGIC_SF16:
        raise ValueError(f"Expected COMPRESSED_LEB128 magic, got {magic!r}")
    byte_count = _read_u32(f)
    data = f.read(byte_count)
    result = np.zeros(count, dtype=dtype)
    pos = 0
    for idx in range(count):
        value = 0
        shift = 0
        while True:
            byte = data[pos]; pos += 1
            value |= (byte & 0x7F) << (shift % 32)
            shift += 7
            if (byte & 0x80) == 0:
                if not (shift >= 32 or (byte & 0x40) == 0):
                    value |= ~((1 << shift) - 1)
                result[idx] = value
                break
    return result

def _read_leb128_sf18(f: BinaryIO, dtype: np.dtype, count: int) -> np.ndarray:
    """Read LEB128-encoded array (SF18 format, magic = 'LEB128 ')."""
    magic = f.read(len(LEB128_MAGIC_SF18))
    if magic != LEB128_MAGIC_SF18:
        raise ValueError(f"Expected 'LEB128 ' magic, got {magic!r}")
    byte_count = _read_u32(f)
    data = f.read(byte_count)
    result = np.zeros(count, dtype=dtype)
    pos = 0
    for idx in range(count):
        value = 0
        shift = 0
        while True:
            byte = data[pos]; pos += 1
            value |= (byte & 0x7F) << (shift % 32)
            shift += 7
            if (byte & 0x80) == 0:
                if not (shift >= 32 or (byte & 0x40) == 0):
                    value |= ~((1 << shift) - 1)
                result[idx] = value
                break
    return result


# ---------------------------------------------------------------------------
# Weight container dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureTransformerWeights:
    biases:              np.ndarray   # (L1,)           int16
    weights:             np.ndarray   # (HALFKA_DIMS, L1) int16
    psqt_weights:        np.ndarray   # (HALFKA_DIMS, PSQT_BUCKETS) int32
    # SF18 only:
    threat_weights:      Optional[np.ndarray] = None   # (THREAT_DIMS, L1) int8
    threat_psqt_weights: Optional[np.ndarray] = None   # (THREAT_DIMS, PSQT_BUCKETS) int32


@dataclass
class LayerStackWeights:
    """
    SF16: fc0 in=1536→16, fc1 in=30(pad32)→32, fc2 in=32→1
    SF18: fc0 in=2048→32, fc1 in=64→32, fc2 in=128→1
    """
    fc0_biases:  np.ndarray   # (out0,)       int32
    fc0_weights: np.ndarray   # (out0, in0)   int8
    fc1_biases:  np.ndarray   # (out1,)       int32
    fc1_weights: np.ndarray   # (out1, in1)   int8
    fc2_biases:  np.ndarray   # (1,)          int32
    fc2_weights: np.ndarray   # (1, in2)      int8


@dataclass
class NNUEWeights:
    version:             int
    description:         str
    feature_transformer: FeatureTransformerWeights
    layer_stacks:        list[LayerStackWeights]
    # Architecture info
    l1:                  int   # FT half-dimension
    halfka_dims:         int
    fc0_out:             int
    fc1_out:             int
    has_threats:         bool


# ---------------------------------------------------------------------------
# Layer parser (shared logic for fc_* layers)
# ---------------------------------------------------------------------------

def _ceil_to_multiple(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult

MAX_SIMD = 32

def _read_fc_layer(f: BinaryIO, in_dims: int, out_dims: int,
                   allow_eof: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one AffineTransform layer (raw little-endian, not LEB128).
    Biases: out_dims × int32
    Weights: out_dims × padded_in_dims × int8, with SF weight-index permutation undone.
    Returns (biases[out_dims], weights[out_dims, in_dims])
    """
    padded = _ceil_to_multiple(in_dims, MAX_SIMD)
    try:
        biases = _read_array_le(f, np.dtype("<i4"), out_dims)
        raw_w  = _read_array_le(f, np.dtype("int8"), out_dims * padded)
    except EOFError:
        if allow_eof:
            # Return zeros for truncated last stack
            return (np.zeros(out_dims, dtype=np.int32),
                    np.zeros((out_dims, in_dims), dtype=np.int8))
        raise

    # Undo SF weight-index permutation
    perm = np.empty(out_dims * padded, dtype=np.int64)
    for i in range(out_dims * padded):
        perm[i] = ((i // 4) % (padded // 4)) * (out_dims * 4) + (i // padded) * 4 + (i % 4)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    unpermuted = raw_w[inv_perm].reshape(out_dims, padded)[:, :in_dims]
    return biases, unpermuted


# ---------------------------------------------------------------------------
# SF16 parser
# ---------------------------------------------------------------------------

def _parse_sf16(f: BinaryIO, desc: str) -> NNUEWeights:
    # Feature Transformer hash
    _ft_hash = _read_u32(f)

    # FT biases: COMPRESSED_LEB128 i16[L1]
    biases = _read_leb128_sf16(f, np.dtype("int16"), SF16_L1)

    # FT weights: COMPRESSED_LEB128 i16[HALFKA_DIMS * L1]
    w_flat = _read_leb128_sf16(f, np.dtype("int16"), SF16_HALFKA_DIMS * SF16_L1)
    weights = w_flat.reshape(SF16_HALFKA_DIMS, SF16_L1)

    # PSQT weights: COMPRESSED_LEB128 i32[HALFKA_DIMS * PSQT_BUCKETS]
    psqt_flat = _read_leb128_sf16(f, np.dtype("int32"), SF16_HALFKA_DIMS * PSQT_BUCKETS)
    psqt = psqt_flat.reshape(SF16_HALFKA_DIMS, PSQT_BUCKETS)

    ft = FeatureTransformerWeights(
        biases=biases, weights=weights, psqt_weights=psqt,
        threat_weights=None, threat_psqt_weights=None,
    )

    # Layer stacks
    # fc_0: in=1536 → out=16
    # fc_1: in=FC_0_OUTPUTS*2=30 (pad→32) → out=32
    # fc_2: in=FC_1_OUTPUTS=32 → out=1
    stacks: list[LayerStackWeights] = []
    last_good: Optional[LayerStackWeights] = None

    for i in range(LAYER_STACKS):
        try:
            _stack_hash = _read_u32(f)
            b0, w0 = _read_fc_layer(f, SF16_L1,        SF16_FC0_OUT)
            b1, w1 = _read_fc_layer(f, (SF16_FC0_OUT - 1) * 2, SF16_FC1_OUT)  # 30→pad32
            b2, w2 = _read_fc_layer(f, SF16_FC1_OUT,   1)
            stack = LayerStackWeights(b0, w0, b1, w1, b2, w2)
            last_good = stack
            stacks.append(stack)
        except (EOFError, struct.error):
            # Truncated last stack (SF16 apt export bug) — clone previous
            if last_good is not None:
                import copy
                stacks.append(copy.deepcopy(last_good))
            else:
                # No valid stacks at all — fill zeros
                stacks.append(LayerStackWeights(
                    np.zeros(SF16_FC0_OUT, np.int32), np.zeros((SF16_FC0_OUT, SF16_L1), np.int8),
                    np.zeros(SF16_FC1_OUT, np.int32), np.zeros((SF16_FC1_OUT, 32), np.int8),
                    np.zeros(1, np.int32), np.zeros((1, SF16_FC1_OUT), np.int8),
                ))

    return NNUEWeights(
        version=VERSION_SF16, description=desc,
        feature_transformer=ft, layer_stacks=stacks,
        l1=SF16_L1, halfka_dims=SF16_HALFKA_DIMS,
        fc0_out=SF16_FC0_OUT, fc1_out=SF16_FC1_OUT,
        has_threats=False,
    )


# ---------------------------------------------------------------------------
# SF18 parser
# ---------------------------------------------------------------------------

def _parse_sf18(f: BinaryIO, desc: str) -> NNUEWeights:
    _ft_hash = _read_u32(f)

    # FT biases: LEB128 i16[L1]
    biases = _read_leb128_sf18(f, np.dtype("int16"), SF18_L1)

    # Threat weights: raw little-endian i8[THREAT_DIMS * L1]
    tw_flat = _read_array_le(f, np.dtype("int8"), SF18_THREAT_DIMS * SF18_L1)
    threat_weights = tw_flat.reshape(SF18_THREAT_DIMS, SF18_L1)

    # Threat PSQT: LEB128 i32[THREAT_DIMS * PSQT_BUCKETS]
    tpsqt_flat = _read_leb128_sf18(f, np.dtype("int32"), SF18_THREAT_DIMS * PSQT_BUCKETS)
    threat_psqt = tpsqt_flat.reshape(SF18_THREAT_DIMS, PSQT_BUCKETS)

    # Main weights: LEB128 i16[HALFKA_DIMS * L1]
    w_flat = _read_leb128_sf18(f, np.dtype("int16"), SF18_HALFKA_DIMS * SF18_L1)
    weights = w_flat.reshape(SF18_HALFKA_DIMS, SF18_L1)

    # PSQT: LEB128 i32[HALFKA_DIMS * PSQT_BUCKETS]
    psqt_flat = _read_leb128_sf18(f, np.dtype("int32"), SF18_HALFKA_DIMS * PSQT_BUCKETS)
    psqt = psqt_flat.reshape(SF18_HALFKA_DIMS, PSQT_BUCKETS)

    ft = FeatureTransformerWeights(
        biases=biases, weights=weights, psqt_weights=psqt,
        threat_weights=threat_weights, threat_psqt_weights=threat_psqt,
    )

    # Layer stacks: 8 of them
    # fc_0: in=L1*2=2048 → out=32
    # fc_1: in=L2*2=64   → out=32
    # fc_2: in=L2*2+L3*2=128 → out=1
    stacks: list[LayerStackWeights] = []
    for _ in range(LAYER_STACKS):
        _stack_hash = _read_u32(f)
        b0, w0 = _read_fc_layer(f, SF18_L1 * 2, SF18_L2)
        b1, w1 = _read_fc_layer(f, SF18_L2 * 2, SF18_L3)
        b2, w2 = _read_fc_layer(f, SF18_L2 * 2 + SF18_L3 * 2, 1)
        stacks.append(LayerStackWeights(b0, w0, b1, w1, b2, w2))

    return NNUEWeights(
        version=VERSION_SF18, description=desc,
        feature_transformer=ft, layer_stacks=stacks,
        l1=SF18_L1, halfka_dims=SF18_HALFKA_DIMS,
        fc0_out=SF18_L2, fc1_out=SF18_L3,
        has_threats=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: str | Path) -> NNUEWeights:
    """
    Parse a Stockfish .nnue file. Detects version from header and dispatches
    to the appropriate parser. Raises ValueError on unrecognised version.
    """
    with open(path, "rb") as f:
        version   = _read_u32(f)
        _filehash = _read_u32(f)
        desc_len  = _read_u32(f)
        desc      = f.read(desc_len).decode("utf-8", errors="replace")

        if version == VERSION_SF16:
            return _parse_sf16(f, desc)
        elif version == VERSION_SF18:
            return _parse_sf18(f, desc)
        else:
            raise ValueError(
                f"Unrecognised NNUE version: {version:#010x}. "
                f"Supported: SF16 ({VERSION_SF16:#010x}), SF18 ({VERSION_SF18:#010x})"
            )
