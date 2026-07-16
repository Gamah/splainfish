"""
Pure-Python reference for the two trickiest parts of the nnue parser port,
transcribed from splainfish/nnue_parser.py with numpy replaced by plain lists.

Emits JSON so the JS port can be diffed against it.

Feeding random bytes to the decoder (rather than round-tripping a hand-written
encoder) tests exactly what matters -- JS decode == Python decode -- without
depending on an encoder that could be wrong in both places.
"""
import json
import random
import sys


def wrap_i16(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


def wrap_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def read_leb128(data, count, wrap):
    """Transcribed from _read_leb128_sf16/_read_leb128_sf18 (they are identical)."""
    result = []
    pos = 0
    for _ in range(count):
        value = 0
        shift = 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 0x7F) << (shift % 32)
            shift += 7
            if (byte & 0x80) == 0:
                if not (shift >= 32 or (byte & 0x40) == 0):
                    value |= ~((1 << shift) - 1)
                result.append(wrap(value))
                break
    return result


def ceil_to_multiple(n, mult):
    return ((n + mult - 1) // mult) * mult


def fc_permute(raw_w, in_dims, out_dims, max_simd=32):
    """Transcribed from _read_fc_layer's permutation half."""
    padded = ceil_to_multiple(in_dims, max_simd)
    n = out_dims * padded
    perm = [0] * n
    for i in range(n):
        perm[i] = ((i // 4) % (padded // 4)) * (out_dims * 4) + (i // padded) * 4 + (i % 4)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    unpermuted = [raw_w[inv_perm[j]] for j in range(n)]
    # reshape(out_dims, padded)[:, :in_dims]
    rows = []
    for r in range(out_dims):
        rows.append(unpermuted[r * padded:r * padded + in_dims])
    return rows


def gen_leb_stream(rng, count):
    """Random but structurally valid varint stream: groups ending in a clear MSB."""
    out = bytearray()
    produced = 0
    while produced < count:
        ngroups = rng.randint(1, 5)
        for g in range(ngroups):
            b = rng.randint(0, 0xFF)
            if g == ngroups - 1:
                b &= 0x7F      # terminator
            else:
                b |= 0x80      # continuation
            out.append(b)
        produced += 1
    return bytes(out)


def main():
    rng = random.Random(20260716)
    cases = []

    for trial in range(60):
        count = rng.randint(1, 40)
        data = gen_leb_stream(rng, count)
        cases.append({
            "data": list(data),
            "count": count,
            "i16": read_leb128(data, count, wrap_i16),
            "i32": read_leb128(data, count, wrap_i32),
        })

    perm_cases = []
    for in_dims, out_dims in [(1536, 16), (30, 32), (32, 1), (2048, 32), (64, 32), (128, 1)]:
        padded = ceil_to_multiple(in_dims, 32)
        raw = [rng.randint(-128, 127) for _ in range(out_dims * padded)]
        perm_cases.append({
            "in_dims": in_dims,
            "out_dims": out_dims,
            "raw": raw,
            "rows": fc_permute(raw, in_dims, out_dims),
        })

    json.dump({"leb": cases, "perm": perm_cases}, sys.stdout)


if __name__ == "__main__":
    main()
