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

# ---------------------------------------------------------------------------
# File header
# ---------------------------------------------------------------------------
# A .nnue file starts with:
#   u32  version    — a *constant* format tag, unchanged across Stockfish
#                     releases. It says nothing about the architecture.
#   u32  arch_hash  — a hash of the architecture string. THIS is what identifies
#                     the layout, and it is what we dispatch on.
#   u32  desc_len
#   utf8 description[desc_len]
#
# Every real net observed reports version 0x7AF32F20, including nets whose
# layouts differ completely. Dispatching on it (as this parser used to) is like
# reading a PNG's magic bytes to work out which camera took the photo.
FORMAT_VERSION = 0x7AF32F20

# LEB128 magic strings, per architecture.
LEB128_MAGIC_COMPRESSED = b"COMPRESSED_LEB128"
LEB128_MAGIC_PLAIN      = b"LEB128 "


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Architecture:
    """
    One NNUE layout. Named for what it *is* — never for a Stockfish release,
    since the file carries no engine version.
    """
    name:        str
    l1:          int            # feature transformer half-dimension
    halfka_dims: int
    fc0_out:     int
    fc1_out:     int
    has_threats: bool
    threat_dims: Optional[int]
    ft_magic:    bytes          # LEB128 magic used by the feature transformer
    ft_style:    str            # "fold" or "concat" — how the FT feeds fc_0
    verified:    bool           # has this been exercised against a real net?


# HalfKAv2_hm, L1=1536, no threat features.
#   fc_0: 1536 -> 16 (15 outputs + 1 skip), fc_1: 30 -> 32, fc_2: 32 -> 1
# The only architecture confirmed against a real network (nn-1c0000000000).
ARCH_HALFKA_1536 = Architecture(
    name="halfka-1536",
    l1=1536, halfka_dims=22528, fc0_out=16, fc1_out=32,
    has_threats=False, threat_dims=None,
    ft_magic=LEB128_MAGIC_COMPRESSED, ft_style="fold",
    verified=True,
)

# HalfKAv2_hm + FullThreats, L1=1024.
#   fc_0: 2048 -> 32, fc_1: 64 -> 32, fc_2: 128 -> 1
#
# UNVERIFIED and deliberately unregistered: no observed net has an arch hash
# that maps here, so nothing reaches this code. It is retained as a starting
# point for whoever implements a threat-bearing architecture, not because it is
# known to describe one. Note that features.threat_indices only spans 1,280 of
# the 60,720 threat rows, so the indexing is incomplete regardless.
ARCH_HALFKA_1024_THREATS = Architecture(
    name="halfka-1024-threats",
    l1=1024, halfka_dims=45056, fc0_out=32, fc1_out=32,
    has_threats=True, threat_dims=60720,
    ft_magic=LEB128_MAGIC_PLAIN, ft_style="concat",
    verified=False,
)

# arch_hash -> architecture. The registry is the whole point: a new Stockfish
# net drops, its hash is either known or it is not, and an unknown hash fails
# immediately with a clear message instead of an IndexError 70 MB into a LEB128
# stream.
ARCH_BY_HASH: dict[int, Architecture] = {
    0x1C1020F2: ARCH_HALFKA_1536,   # nn-1c0000000000.nnue
}

# Observed but not implemented — listed so the error message can say so.
KNOWN_UNSUPPORTED: dict[int, str] = {
    0xEC102EF2: "nn-c288c895ea92.nnue (current big net, 109 MB)",
    0x1C103C92: "nn-37f18f62d772.nnue (current small net, 3.3 MB)",
}


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

def _read_leb128(f: BinaryIO, magic: bytes, dtype: np.dtype, count: int) -> np.ndarray:
    """
    Read a magic-prefixed, length-prefixed LEB128 array.

    The varint encoding is identical across architectures; only the magic
    string differs (COMPRESSED_LEB128 vs 'LEB128 '), so it is passed in.
    """
    got = f.read(len(magic))
    if got != magic:
        raise ValueError(f"Expected {magic!r} magic, got {got!r}")
    byte_count = _read_u32(f)
    data = f.read(byte_count)
    result = np.zeros(count, dtype=dtype)
    pos = 0
    for idx in range(count):
        value = 0
        shift = 0
        while True:
            if pos >= len(data):
                raise EOFError(
                    f"LEB128 stream exhausted after {idx} of {count} values — "
                    f"the architecture's dimensions do not match this file"
                )
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
    arch:                Architecture
    description:         str
    feature_transformer: FeatureTransformerWeights
    layer_stacks:        list[LayerStackWeights]

    # Convenience passthroughs to the architecture, kept so existing callers
    # (probe.py) that read weights.l1 etc. keep working unchanged.
    @property
    def l1(self) -> int:            return self.arch.l1
    @property
    def halfka_dims(self) -> int:   return self.arch.halfka_dims
    @property
    def fc0_out(self) -> int:       return self.arch.fc0_out
    @property
    def fc1_out(self) -> int:       return self.arch.fc1_out
    @property
    def has_threats(self) -> bool:  return self.arch.has_threats


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
# Architecture-driven parser
# ---------------------------------------------------------------------------

def _parse_arch(f: BinaryIO, arch: Architecture, desc: str) -> NNUEWeights:
    """Parse the feature transformer and layer stacks for a known architecture."""
    _ft_hash = _read_u32(f)
    l1 = arch.l1
    halfka = arch.halfka_dims

    # FT biases: LEB128 i16[L1]
    biases = _read_leb128(f, arch.ft_magic, np.dtype("int16"), l1)

    threat_weights = None
    threat_psqt = None
    if arch.has_threats:
        # Threat weights are stored raw, not LEB128.
        tw_flat = _read_array_le(f, np.dtype("int8"), arch.threat_dims * l1)
        threat_weights = tw_flat.reshape(arch.threat_dims, l1)
        tpsqt_flat = _read_leb128(f, arch.ft_magic, np.dtype("int32"),
                                  arch.threat_dims * PSQT_BUCKETS)
        threat_psqt = tpsqt_flat.reshape(arch.threat_dims, PSQT_BUCKETS)

    # FT weights: LEB128 i16[HALFKA_DIMS * L1]
    w_flat = _read_leb128(f, arch.ft_magic, np.dtype("int16"), halfka * l1)
    weights = w_flat.reshape(halfka, l1)

    # PSQT: LEB128 i32[HALFKA_DIMS * PSQT_BUCKETS]
    psqt_flat = _read_leb128(f, arch.ft_magic, np.dtype("int32"), halfka * PSQT_BUCKETS)
    psqt = psqt_flat.reshape(halfka, PSQT_BUCKETS)

    ft = FeatureTransformerWeights(
        biases=biases, weights=weights, psqt_weights=psqt,
        threat_weights=threat_weights, threat_psqt_weights=threat_psqt,
    )

    # Layer stack dimensions differ by FT style:
    #   fold:   fc_0 L1 -> fc0_out, fc_1 (fc0_out-1)*2 -> fc1_out, fc_2 fc1_out -> 1
    #   concat: fc_0 L1*2 -> fc0_out, fc_1 fc0_out*2 -> fc1_out,
    #           fc_2 fc0_out*2 + fc1_out*2 -> 1
    if arch.ft_style == "fold":
        fc0_in = l1
        fc1_in = (arch.fc0_out - 1) * 2
        fc2_in = arch.fc1_out
    else:
        fc0_in = l1 * 2
        fc1_in = arch.fc0_out * 2
        fc2_in = arch.fc0_out * 2 + arch.fc1_out * 2

    stacks: list[LayerStackWeights] = []
    last_good: Optional[LayerStackWeights] = None

    for _ in range(LAYER_STACKS):
        try:
            _stack_hash = _read_u32(f)
            b0, w0 = _read_fc_layer(f, fc0_in, arch.fc0_out)
            b1, w1 = _read_fc_layer(f, fc1_in, arch.fc1_out)
            b2, w2 = _read_fc_layer(f, fc2_in, 1)
            stack = LayerStackWeights(b0, w0, b1, w1, b2, w2)
            last_good = stack
            stacks.append(stack)
        except (EOFError, struct.error):
            # Truncated last stack (an apt-packaged export bug seen on the
            # halfka-1536 net) — clone the previous stack rather than fail.
            if last_good is not None:
                stacks.append(copy.deepcopy(last_good))
            else:
                stacks.append(LayerStackWeights(
                    np.zeros(arch.fc0_out, np.int32), np.zeros((arch.fc0_out, fc0_in), np.int8),
                    np.zeros(arch.fc1_out, np.int32), np.zeros((arch.fc1_out, fc1_in), np.int8),
                    np.zeros(1, np.int32), np.zeros((1, fc2_in), np.int8),
                ))

    return NNUEWeights(arch=arch, description=desc,
                       feature_transformer=ft, layer_stacks=stacks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: str | Path) -> NNUEWeights:
    """
    Parse a Stockfish .nnue file.

    Dispatches on the architecture hash (the second u32), NOT the format version
    (the first u32, which is a constant and identifies nothing). Raises
    ValueError for an architecture that is recognised-but-unimplemented or
    entirely unknown.
    """
    with open(path, "rb") as f:
        version   = _read_u32(f)
        arch_hash = _read_u32(f)
        desc_len  = _read_u32(f)
        desc      = f.read(desc_len).decode("utf-8", errors="replace")

        arch = ARCH_BY_HASH.get(arch_hash)
        if arch is not None:
            return _parse_arch(f, arch, desc)

        supported = ", ".join(
            f"{a.name} ({h:#010x})" for h, a in ARCH_BY_HASH.items()
        )
        if arch_hash in KNOWN_UNSUPPORTED:
            raise ValueError(
                f"NNUE architecture {arch_hash:#010x} is recognised but not yet "
                f"implemented: {KNOWN_UNSUPPORTED[arch_hash]}. "
                f"Implemented architectures: {supported}."
            )
        raise ValueError(
            f"Unknown NNUE architecture {arch_hash:#010x} "
            f"(format version {version:#010x}). "
            f"Implemented architectures: {supported}."
        )
