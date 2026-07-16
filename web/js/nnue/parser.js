/**
 * parser.js — Parse a Stockfish .nnue file in the browser.
 *
 * Port of splainfish/nnue_parser.py. Reads from an ArrayBuffer rather than a
 * file handle; otherwise the format handling is intended to match the Python
 * byte for byte.
 *
 * Weight matrices are kept as flat typed arrays with explicit row striding
 * rather than arrays-of-arrays: the SF18 feature transformer alone is
 * 45056 x 1024 int16 (~92 MB), and a nested representation would add an object
 * header per row.
 *
 * Supported versions:
 *   SF16  0x7AF32F20  HalfKAv2_hm only,          L1=1536
 *   SF18  0x6A448AFA  HalfKAv2_hm + FullThreats, L1=1024
 */

// ---------------------------------------------------------------------------
// Shared constants (mirrors nnue_parser.py)
// ---------------------------------------------------------------------------

export const WEIGHT_SCALE_BITS = 6;
export const WEIGHT_SCALE      = 1 << WEIGHT_SCALE_BITS; // 64
export const HIDDEN_ONE_VAL    = 128;
export const OUTPUT_SCALE      = 16;
export const FT_MAX_VAL        = 255;
export const PSQT_BUCKETS      = 8;
export const LAYER_STACKS      = 8;

// A .nnue file starts with u32 version, u32 arch_hash, u32 desc_len, utf8 desc.
// The version is a constant format tag (identical across Stockfish releases and
// architectures); the arch_hash identifies the layout. Dispatch is on the hash.
export const FORMAT_VERSION = 0x7af32f20;

const LEB128_MAGIC_COMPRESSED = 'COMPRESSED_LEB128';
const LEB128_MAGIC_PLAIN = 'LEB128 ';

const MAX_SIMD = 32;

/**
 * Registry of known architectures, keyed by arch hash. Mirrors
 * splainfish/nnue_parser.py. Only halfka-1536 is verified against a real net.
 */
export const ARCHITECTURES = {
  'halfka-1536': {
    name: 'halfka-1536',
    l1: 1536, halfkaDims: 22528, fc0Out: 16, fc1Out: 32,
    hasThreats: false, threatDims: null,
    ftMagic: LEB128_MAGIC_COMPRESSED, ftStyle: 'fold',
    verified: true,
  },
  // Unverified and unregistered below (no arch hash maps to it) — retained as a
  // starting point for a threat-bearing architecture, not known to describe one.
  'halfka-1024-threats': {
    name: 'halfka-1024-threats',
    l1: 1024, halfkaDims: 45056, fc0Out: 32, fc1Out: 32,
    hasThreats: true, threatDims: 60720,
    ftMagic: LEB128_MAGIC_PLAIN, ftStyle: 'concat',
    verified: false,
  },
};

export const ARCH_BY_HASH = {
  0x1c1020f2: ARCHITECTURES['halfka-1536'], // nn-1c0000000000.nnue
};

// Observed but not implemented — so the error can name them.
export const KNOWN_UNSUPPORTED = {
  0xec102ef2: 'nn-c288c895ea92.nnue (current big net, 109 MB)',
  0x1c103c92: 'nn-37f18f62d772.nnue (current small net, 3.3 MB)',
};

// ---------------------------------------------------------------------------
// Byte reader
// ---------------------------------------------------------------------------

class ByteReader {
  constructor(buffer) {
    this.view = new DataView(buffer);
    this.bytes = new Uint8Array(buffer);
    this.pos = 0;
  }

  get remaining() {
    return this.bytes.length - this.pos;
  }

  u32() {
    if (this.remaining < 4) throw new RangeError('EOF reading u32');
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }

  ascii(n) {
    const s = String.fromCharCode(...this.bytes.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }

  utf8(n) {
    const s = new TextDecoder('utf-8').decode(this.bytes.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }

  /** Raw little-endian typed array. Copies, so the result outlives the buffer. */
  array(Ctor, count) {
    const bytes = count * Ctor.BYTES_PER_ELEMENT;
    if (this.remaining < bytes) {
      throw new RangeError(`EOF: expected ${bytes} bytes, got ${this.remaining}`);
    }
    // Can't construct a typed array directly on an unaligned offset, so copy the
    // byte range out first and reinterpret that.
    const slice = this.bytes.slice(this.pos, this.pos + bytes);
    this.pos += bytes;
    return new Ctor(slice.buffer);
  }
}

// ---------------------------------------------------------------------------
// LEB128
// ---------------------------------------------------------------------------

/**
 * Decode `count` LEB128 varints from `data` into a new typed array.
 *
 * Both SF16 and SF18 use the same varint encoding and differ only in the magic
 * string, so this is shared. Accumulation happens in JS 32-bit bitwise space,
 * which matches the Python's arbitrary-precision accumulate followed by a
 * truncating store into an int16/int32 numpy array.
 *
 * Exported for the parity test against the Python reference.
 */
export function decodeLeb128(data, Ctor, count) {
  const out = new Ctor(count);
  let pos = 0;
  for (let idx = 0; idx < count; idx++) {
    let value = 0;
    let shift = 0;
    for (;;) {
      const byte = data[pos++];
      value |= (byte & 0x7f) << (shift % 32);
      shift += 7;
      if ((byte & 0x80) === 0) {
        // Sign-extend unless the value already filled the width or the sign bit
        // of the final group is clear.
        if (!(shift >= 32 || (byte & 0x40) === 0)) {
          value |= ~((1 << shift) - 1);
        }
        out[idx] = value;
        break;
      }
    }
  }
  return out;
}

/** Read a magic-prefixed, length-prefixed LEB128 block from the stream. */
function readLeb128(reader, magic, Ctor, count) {
  const got = reader.ascii(magic.length);
  if (got !== magic) {
    throw new Error(`Expected ${JSON.stringify(magic)} magic, got ${JSON.stringify(got)}`);
  }
  const byteCount = reader.u32();
  const data = reader.bytes.subarray(reader.pos, reader.pos + byteCount);
  reader.pos += byteCount;
  if (byteCount < count) {
    // A varint is at least one byte, so fewer bytes than values means the
    // architecture's dimensions do not match this file.
    throw new Error(
      `LEB128 block too short: ${byteCount} bytes for ${count} values — ` +
      'architecture does not match this file',
    );
  }
  return decodeLeb128(data, Ctor, count);
}

// ---------------------------------------------------------------------------
// Affine layer reader
// ---------------------------------------------------------------------------

function ceilToMultiple(n, mult) {
  return Math.floor((n + mult - 1) / mult) * mult;
}

/**
 * Undo Stockfish's SIMD weight-index permutation and strip the padding.
 *
 * Python builds perm, inverts it, then gathers: unperm[j] = raw[invPerm[j]].
 * Since invPerm[perm[i]] == i, that is equivalent to scattering directly:
 *   unperm[perm[i]] = raw[i]
 * which avoids materialising the inverse permutation.
 *
 * Returns a row-major Int8Array with stride inDims.
 * Exported for the parity test against the Python reference.
 */
export function unpermuteFcWeights(rawW, inDims, outDims) {
  const padded = ceilToMultiple(inDims, MAX_SIMD);
  const total = outDims * padded;
  const unperm = new Int8Array(total);
  const quarterPadded = padded / 4;
  for (let i = 0; i < total; i++) {
    const p =
      (Math.floor(i / 4) % quarterPadded) * (outDims * 4) +
      Math.floor(i / padded) * 4 +
      (i % 4);
    unperm[p] = rawW[i];
  }

  // Drop the SIMD padding: keep the first inDims of each padded row.
  const weights = new Int8Array(outDims * inDims);
  for (let r = 0; r < outDims; r++) {
    weights.set(unperm.subarray(r * padded, r * padded + inDims), r * inDims);
  }
  return weights;
}

/**
 * Read one AffineTransform layer (raw little-endian, never LEB128).
 *
 * Returns { biases: Int32Array(outDims), weights: Int8Array(outDims * inDims) }
 * where weights is row-major with stride inDims.
 */
function readFcLayer(reader, inDims, outDims) {
  const padded = ceilToMultiple(inDims, MAX_SIMD);
  const biases = reader.array(Int32Array, outDims);
  const rawW = reader.array(Int8Array, outDims * padded);
  return { biases, weights: unpermuteFcWeights(rawW, inDims, outDims) };
}

/** Zero-filled stand-in for a stack that ran off the end of the file. */
function zeroStack(fc0Out, fc0In, fc1Out, fc1In, fc2In) {
  return {
    fc0Biases: new Int32Array(fc0Out),
    fc0Weights: new Int8Array(fc0Out * fc0In),
    fc1Biases: new Int32Array(fc1Out),
    fc1Weights: new Int8Array(fc1Out * fc1In),
    fc2Biases: new Int32Array(1),
    fc2Weights: new Int8Array(fc2In),
  };
}

function cloneStack(s) {
  return {
    fc0Biases: s.fc0Biases.slice(),
    fc0Weights: s.fc0Weights.slice(),
    fc1Biases: s.fc1Biases.slice(),
    fc1Weights: s.fc1Weights.slice(),
    fc2Biases: s.fc2Biases.slice(),
    fc2Weights: s.fc2Weights.slice(),
  };
}

// ---------------------------------------------------------------------------
// Architecture-driven parse
// ---------------------------------------------------------------------------

function parseArch(reader, arch, description) {
  reader.u32(); // feature transformer hash

  const { l1, halfkaDims, ftMagic } = arch;

  const biases = readLeb128(reader, ftMagic, Int16Array, l1);

  let threatWeights = null;
  let threatPsqtWeights = null;
  if (arch.hasThreats) {
    // Threat weights are stored raw, not LEB128.
    threatWeights = reader.array(Int8Array, arch.threatDims * l1);
    threatPsqtWeights = readLeb128(
      reader, ftMagic, Int32Array, arch.threatDims * PSQT_BUCKETS,
    );
  }

  const weights = readLeb128(reader, ftMagic, Int16Array, halfkaDims * l1);
  const psqtWeights = readLeb128(reader, ftMagic, Int32Array, halfkaDims * PSQT_BUCKETS);

  const featureTransformer = {
    biases,
    weights,
    weightStride: l1,
    psqtWeights,
    psqtStride: PSQT_BUCKETS,
    threatWeights,
    threatStride: l1,
    threatPsqtWeights,
  };

  // fc layer dimensions per FT style (see splainfish/nnue_parser.py:_parse_arch)
  let fc0In, fc1In, fc2In;
  if (arch.ftStyle === 'fold') {
    fc0In = l1;
    fc1In = (arch.fc0Out - 1) * 2;
    fc2In = arch.fc1Out;
  } else {
    fc0In = l1 * 2;
    fc1In = arch.fc0Out * 2;
    fc2In = arch.fc0Out * 2 + arch.fc1Out * 2;
  }

  const stacks = [];
  let lastGood = null;

  for (let i = 0; i < LAYER_STACKS; i++) {
    try {
      reader.u32(); // stack hash
      const l0 = readFcLayer(reader, fc0In, arch.fc0Out);
      const l1w = readFcLayer(reader, fc1In, arch.fc1Out);
      const l2 = readFcLayer(reader, fc2In, 1);
      const stack = {
        fc0Biases: l0.biases, fc0Weights: l0.weights,
        fc1Biases: l1w.biases, fc1Weights: l1w.weights,
        fc2Biases: l2.biases, fc2Weights: l2.weights,
      };
      lastGood = stack;
      stacks.push(stack);
    } catch (err) {
      if (!(err instanceof RangeError)) throw err;
      // Truncated final stack (an apt-export bug seen on halfka-1536) — clone
      // the last stack that parsed rather than fail.
      stacks.push(
        lastGood
          ? cloneStack(lastGood)
          : zeroStack(arch.fc0Out, fc0In, arch.fc1Out, fc1In, fc2In),
      );
    }
  }

  return {
    arch,
    description,
    featureTransformer,
    layerStacks: stacks,
    // Passthroughs so callers (probe.js) can read weights.l1 etc.
    l1,
    halfkaDims,
    fc0Out: arch.fc0Out,
    fc1Out: arch.fc1Out,
    hasThreats: arch.hasThreats,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Parse a .nnue file from an ArrayBuffer.
 *
 * Dispatches on the architecture hash (the second u32), NOT the format version.
 * Throws for a recognised-but-unimplemented or unknown architecture.
 */
export function parseNnue(buffer) {
  const reader = new ByteReader(buffer);

  const version = reader.u32();
  const archHash = reader.u32() >>> 0;
  const descLen = reader.u32();
  const description = reader.utf8(descLen);

  const arch = ARCH_BY_HASH[archHash];
  if (arch) return parseArch(reader, arch, description);

  const hex = (h) => '0x' + (h >>> 0).toString(16).padStart(8, '0');
  const supported = Object.entries(ARCH_BY_HASH)
    .map(([h, a]) => `${a.name} (${hex(Number(h))})`).join(', ');

  if (KNOWN_UNSUPPORTED[archHash]) {
    throw new Error(
      `NNUE architecture ${hex(archHash)} is recognised but not yet implemented: ` +
      `${KNOWN_UNSUPPORTED[archHash]}. Implemented: ${supported}.`,
    );
  }
  throw new Error(
    `Unknown NNUE architecture ${hex(archHash)} (format version ${hex(version)}). ` +
    `Implemented: ${supported}.`,
  );
}
