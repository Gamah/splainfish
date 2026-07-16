/**
 * loader.js — Fetch and cache the NNUE network, then parse it.
 *
 * The .nnue is committed under web/nnue/ and served same-origin by GitHub Pages
 * (no public host serves the networks with CORS, so runtime fetch from upstream
 * is impossible). It is ~71 MB, so it is cached in IndexedDB after first load
 * and the download is reported for a progress bar.
 */

import { parseNnue } from './parser.js';

const NET_URL = new URL('../../nnue/nn-1c0000000000.nnue', import.meta.url).href;
const NET_KEY = 'nn-1c0000000000';

const DB_NAME = 'splainfish';
const STORE = 'nnue';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      tx.onsuccess = () => resolve(tx.result ?? null);
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    return null; // IndexedDB unavailable (private mode etc.) — just refetch.
  }
}

async function idbPut(key, value) {
  try {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite').objectStore(STORE).put(value, key);
      tx.onsuccess = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    /* caching is best-effort */
  }
}

async function fetchNet(onProgress) {
  const resp = await fetch(NET_URL);
  if (!resp.ok) throw new Error(`NNUE fetch failed: HTTP ${resp.status}`);

  const total = Number(resp.headers.get('content-length')) || 0;
  if (!resp.body || !total) {
    onProgress?.(null);
    return await resp.arrayBuffer();
  }

  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received / total);
  }
  const out = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { out.set(c, pos); pos += c.length; }
  return out.buffer;
}

/**
 * Load the network weights, using the IndexedDB cache when present.
 * onProgress(frac) reports download progress on a cache miss (frac may be null
 * when the length is unknown); it is not called on a cache hit.
 * Returns the parsed NNUEWeights.
 */
export async function loadNnue(onProgress) {
  let buf = await idbGet(NET_KEY);
  if (!buf) {
    buf = await fetchNet(onProgress);
    await idbPut(NET_KEY, buf);
  }
  return parseNnue(buf);
}
