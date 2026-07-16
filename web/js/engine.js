/**
 * engine.js — Stockfish UCI engine, loaded from CDN at runtime.
 *
 * Browser counterpart of splainfish/engine.py. Stockfish is not bundled; it is
 * fetched from a public CDN and instantiated as a Web Worker. Download progress
 * is reported so the UI can show a loading bar (the WASM is several MB).
 *
 * Only the single-threaded lite build is used: it needs no COOP/COEP headers
 * and therefore runs on GitHub Pages without a cross-origin-isolation shim.
 */

const STOCKFISH_VERSION = '18.0.8';
// jsDelivr rejects the stockfish package (over its 150 MB package cap), so use
// unpkg, which serves the individual files with permissive CORS.
const CDN_BASE = `https://unpkg.com/stockfish@${STOCKFISH_VERSION}/bin`;
const ENGINE_JS = `${CDN_BASE}/stockfish-18-lite-single.js`;
const ENGINE_WASM = `${CDN_BASE}/stockfish-18-lite-single.wasm`;

const MATE_CP = 30000;

/** Fetch a URL as an ArrayBuffer, reporting progress via onProgress(frac,label). */
async function fetchWithProgress(url, label, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${label}: HTTP ${resp.status} from ${url}`);

  const total = Number(resp.headers.get('content-length')) || 0;
  if (!resp.body || !total) {
    // No streaming or unknown length — fall back to a single opaque wait.
    onProgress?.(null, label);
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
    onProgress?.(received / total, label);
  }
  const out = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { out.set(c, pos); pos += c.length; }
  return out.buffer;
}

/**
 * A parsed principal-variation line.
 * scoreCp is from the side-to-move's perspective; scoreCpWhite() flips it.
 */
class PVLine {
  constructor({ rank, scoreCp, mateIn, moves, depth }) {
    this.rank = rank;
    this.scoreCp = scoreCp;
    this.mateIn = mateIn;
    this.moves = moves;
    this.depth = depth;
  }

  get isMate() { return this.mateIn !== null; }

  scoreCpWhite(whiteToMove) {
    let cp;
    if (this.mateIn !== null) cp = this.mateIn > 0 ? MATE_CP : -MATE_CP;
    else cp = this.scoreCp;
    return whiteToMove ? cp : -cp;
  }
}

export class StockfishEngine {
  constructor() {
    this._worker = null;
    this._listeners = new Set();
  }

  /**
   * Load Stockfish from the CDN and start it as a worker.
   * onProgress(frac, label) is called during the WASM download; frac may be
   * null when the length is unknown.
   */
  async load(onProgress) {
    // Fetch the loader script and wasm so we can report progress, then hand the
    // worker a blob URL that points the engine's locateFile at our wasm blob.
    const [jsBuf, wasmBuf] = [
      await fetchWithProgress(ENGINE_JS, 'engine', onProgress),
      await fetchWithProgress(ENGINE_WASM, 'network', onProgress),
    ];

    const wasmBlobUrl = URL.createObjectURL(
      new Blob([wasmBuf], { type: 'application/wasm' }),
    );

    // Shim: define locateFile before the engine script runs so it uses our blob
    // instead of trying to re-fetch the wasm by relative path.
    const shim = `var Module={locateFile:function(p){return p.endsWith(".wasm")?${JSON.stringify(wasmBlobUrl)}:p;}};\n`;
    const jsText = shim + new TextDecoder().decode(jsBuf);
    const jsBlobUrl = URL.createObjectURL(
      new Blob([jsText], { type: 'text/javascript' }),
    );

    this._worker = new Worker(jsBlobUrl);
    this._worker.onmessage = (e) => {
      const line = typeof e.data === 'string' ? e.data : e.data?.data ?? '';
      for (const fn of this._listeners) fn(line);
    };

    await this._send('uci', (line) => line.includes('uciok'));
    await this._send('isready', (line) => line.includes('readyok'));
  }

  _onLine(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }

  /** Send a command; resolve when `until(line)` is true (or immediately if no predicate). */
  _send(cmd, until) {
    if (!until) { this._worker.postMessage(cmd); return Promise.resolve(); }
    return new Promise((resolve) => {
      const off = this._onLine((line) => { if (until(line)) { off(); resolve(); } });
      this._worker.postMessage(cmd);
    });
  }

  /**
   * Analyse a position to a fixed depth.
   * Returns { fen, depth, pvLines: PVLine[] } sorted by multipv rank.
   */
  analyse(fen, { depth = 14, multipv = 3 } = {}) {
    const byRank = new Map();
    return new Promise((resolve) => {
      const off = this._onLine((line) => {
        if (line.startsWith('info') && line.includes(' pv ')) {
          const pv = parseInfoLine(line);
          if (pv) byRank.set(pv.rank, pv);
        } else if (line.startsWith('bestmove')) {
          off();
          const pvLines = [...byRank.values()].sort((a, b) => a.rank - b.rank);
          resolve({ fen, depth, pvLines });
        }
      });
      this._worker.postMessage('ucinewgame');
      this._worker.postMessage(`setoption name MultiPV value ${multipv}`);
      this._worker.postMessage(`position fen ${fen}`);
      this._worker.postMessage(`go depth ${depth}`);
    });
  }

  quit() {
    try { this._worker?.postMessage('quit'); } catch { /* ignore */ }
    this._worker?.terminate();
    this._worker = null;
  }
}

/** Parse one `info ... pv ...` line into a PVLine, or null if incomplete. */
function parseInfoLine(line) {
  const tokens = line.split(/\s+/);
  let rank = 1, depth = 0, scoreCp = null, mateIn = null;
  const moves = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === 'multipv') rank = parseInt(tokens[++i], 10);
    else if (t === 'depth') depth = parseInt(tokens[++i], 10);
    else if (t === 'cp') scoreCp = parseInt(tokens[++i], 10);
    else if (t === 'mate') mateIn = parseInt(tokens[++i], 10);
    else if (t === 'pv') { moves.push(...tokens.slice(i + 1)); break; }
  }
  if (scoreCp === null && mateIn === null) return null;
  return new PVLine({ rank, scoreCp: scoreCp ?? 0, mateIn, moves, depth });
}

export { PVLine, MATE_CP };
