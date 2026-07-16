/**
 * Diff the JS nnue parser internals against the pure-Python reference.
 * Reads the reference JSON on stdin.
 */
import { decodeLeb128, unpermuteFcWeights } from '../web/js/nnue/parser.js';

const raw = await new Promise((resolve) => {
  let buf = '';
  process.stdin.on('data', (d) => (buf += d));
  process.stdin.on('end', () => resolve(buf));
});
const ref = JSON.parse(raw);

let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL: ' + msg); };

// --- LEB128 ---
let lebChecked = 0;
for (const [n, c] of ref.leb.entries()) {
  const data = Uint8Array.from(c.data);

  const got16 = Array.from(decodeLeb128(data, Int16Array, c.count));
  if (JSON.stringify(got16) !== JSON.stringify(c.i16)) {
    fail(`leb case ${n} i16\n    py=${JSON.stringify(c.i16)}\n    js=${JSON.stringify(got16)}`);
  }

  const got32 = Array.from(decodeLeb128(data, Int32Array, c.count));
  if (JSON.stringify(got32) !== JSON.stringify(c.i32)) {
    fail(`leb case ${n} i32\n    py=${JSON.stringify(c.i32)}\n    js=${JSON.stringify(got32)}`);
  }
  lebChecked += c.count * 2;
}
console.log(`LEB128: ${ref.leb.length} streams, ${lebChecked} values compared`);

// --- FC permutation ---
let permChecked = 0;
for (const c of ref.perm) {
  const rawW = Int8Array.from(c.raw);
  const got = unpermuteFcWeights(rawW, c.in_dims, c.out_dims);
  const want = c.rows.flat();
  if (got.length !== want.length) {
    fail(`perm ${c.in_dims}x${c.out_dims}: length ${got.length} vs ${want.length}`);
    continue;
  }
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) {
      fail(`perm ${c.in_dims}x${c.out_dims}: idx ${i} py=${want[i]} js=${got[i]}`);
      break;
    }
  }
  permChecked += want.length;
}
console.log(`FC permutation: ${ref.perm.length} shapes, ${permChecked} weights compared`);

console.log(failures === 0 ? '\nAll parity checks PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
