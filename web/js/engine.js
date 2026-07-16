/**
 * engine.js — Stockfish UCI engine, self-hosted and run as a Web Worker.
 *
 * Browser counterpart of splainfish/engine.py. Stockfish is vendored alongside
 * the app in web/vendor/ and served same-origin, so the worker loads it from a
 * real URL and Emscripten fetches the wasm sitting next to the script. (Loading
 * it from a CDN via a blob-URL worker failed to fetch the wasm inside the worker.)
 *
 * Only the single-threaded lite build is used: it needs no COOP/COEP headers
 * and therefore runs under any static host without a cross-origin-isolation shim.
 */

// Stockfish 18.0.8 (lite, single-threaded) is self-hosted next to the app
// (web/vendor/), NOT loaded from a CDN. Serving it same-origin lets the worker
// load directly from a real URL so Emscripten resolves the .wasm sitting beside
// the .js automatically — no blob URL, no shim, no cross-origin fetch (all of
// which failed inside the worker). Resolved relative to this module's URL, so it
// works under any deploy subpath.
const ENGINE_JS = new URL('../vendor/stockfish-18-lite-single.js', import.meta.url);

const MATE_CP = 30000;

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
   * Load Stockfish from the vendored same-origin script and start it as a worker.
   * onProgress(frac, label) is called with a null frac (indeterminate) since the
   * worker downloads its own wasm where we can't observe progress.
   */
  async load(onProgress) {
    // Load the worker straight from the same-origin vendored script. Emscripten's
    // default locateFile resolves the .wasm next to it (same directory, same
    // origin), so no blob URL, shim, or cross-origin fetch is involved. We can't
    // observe the wasm download from here, so just show an indeterminate bar.
    onProgress?.(null, 'network');

    this._worker = new Worker(ENGINE_JS);
    this._worker.onmessage = (e) => {
      const line = typeof e.data === 'string' ? e.data : e.data?.data ?? '';
      for (const fn of this._listeners) fn(line);
    };
    // Without this, a worker/wasm init failure is swallowed and load() hangs
    // forever on the uci handshake below, freezing the UI on the progress bar.
    const workerError = new Promise((_, reject) => {
      this._worker.onerror = (e) =>
        reject(new Error(`Stockfish worker failed: ${e.message || 'wasm init error'}`));
    });

    const handshake = (async () => {
      await this._send('uci', (line) => line.includes('uciok'));
      await this._send('isready', (line) => line.includes('readyok'));
    })();

    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Stockfish load timed out after 30s')), 30_000),
    );

    await Promise.race([handshake, workerError, timeout]);
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
