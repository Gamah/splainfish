# splainfish

Explains what the Stockfish NNUE network *actually reacted to* between two chess
positions — in plain English a novice can understand.

## What it does

Most chess tools tell you **how much** a move changed the evaluation. splainfish
explains **why** — by running the NNUE forward pass on both positions, diffing
the network activations layer by layer (FT → L1 → L2), and back-projecting the
output change onto the input features that changed.

The result is a ranked list of chess concepts (e.g. "enemy queen threatening
king area", "own pawn shield", "own knight activity") that the network most
responded to, in two views:

- **Simple** — 2-3 plain sentences
- **Detailed** — a bar chart of ranked feature groups

No LLM. All explanation is deterministic from the NNUE activations.

## Two ways to run it

### 1. In the browser (`web/`)

Paste a PGN, and everything runs client-side: Stockfish (WASM, loaded from a
CDN) provides the evals, and the NNUE probe runs in JavaScript. Nothing is
uploaded. This is what deploys to GitHub Pages.

```bash
# Serve web/ over http (a module app needs http, not file://)
cd web && python3 -m http.server 8000
# open http://localhost:8000
```

On first analyse it downloads Stockfish (~7 MB) and the NNUE network (~71 MB,
served same-origin from `web/nnue/` and cached in IndexedDB afterwards), behind
a progress bar.

To deploy: publish the `web/` directory as the GitHub Pages root. The network
file **must** be a normal committed file — Git LFS pointers are not resolved by
Pages.

### 2. Command line (Python)

```bash
make setup                                   # build Stockfish, install deps
make report PGN=mygame.pgn OUTPUT=mygame.html
make demo                                    # the Immortal Game
```

The CLI writes the same self-contained HTML report the browser produces, with
zero external dependencies (pieces inlined) — email it or host it anywhere.

## How it works

```
Position A ─► Stockfish ─► eval, best move, PV
     │
     │  HalfKAv2_hm feature computation
     ▼
Feature diff (gained / lost indices)
     │
     ▼
NNUE forward pass (re-implemented; numpy in Python, typed arrays in JS)
  FT accumulator → fc_0 → fc_1 → fc_2 + skip
     │
     ▼
Activation delta at each layer
     │
     ▼
Back-projection fc_2 → fc_1 → fc_0 → FT   (first-order Taylor at the midpoint)
     │
     ▼
Per-feature attribution → grouped by concept → English sentences
     │
     ▼
Interactive HTML (board + move list + explanations)
```

## Supported networks

A `.nnue` file does **not** record a Stockfish version. Its first 4 bytes are a
constant format tag (identical across releases); the architecture is identified
by the **second** 4 bytes — the architecture hash. splainfish dispatches on that
hash:

| Architecture | Arch hash | Layout | Status |
|---|---|---|---|
| `halfka-1536` | `0x1C1020F2` | HalfKAv2_hm, L1=1536 | supported (net `nn-1c0000000000`, committed) |
| current big net | `0xEC102EF2` | different | recognised, not implemented |
| current small net | `0x1C103C92` | different | recognised, not implemented |

An unknown architecture fails with a clear message rather than a corrupt parse.
Supporting the current nets means implementing their (different) layout — see
the honesty section.

## Verification

The browser re-implements the whole NNUE pipeline in JavaScript. It is checked
against the Python (the reference) by parity tests:

```bash
make test              # parity + real-net + integration
make test-parity       # LEB128, feature indexing, forward pass, back-projection
make test-realnet      # JS parses the real net byte-for-byte like Python
make test-integration  # the whole browser pipeline, end-to-end, in Node
```

The parity tests are mutation-tested: deliberately breaking either
implementation makes them fail.

## Limitations and honesty

- **The engine evaluations shown are Stockfish's**, exact from its UCI output.
  splainfish's own re-implemented forward pass does **not** produce reliable
  centipawn magnitudes on the supported net — its output is dominated by
  layer-stack biases and is used only for attribution *direction*, never
  displayed. Fixing the internal eval means matching Stockfish's exact quantised
  inference and validating it against a real Stockfish binary; that work is not
  done.
- **Attribution is a first-order approximation** (a linear Taylor expansion at
  the midpoint activation). It shows relative direction and emphasis, not
  calibrated centipawns. For sharp tactical positions it spreads more diffusely.
- **Only the `halfka-1536` architecture is implemented.** The current Stockfish
  nets use a different, larger layout (and would exceed GitHub's 100 MB file
  limit uncompressed anyway).

## Project layout

```
splainfish/          Python package (CLI + reference implementation)
  nnue_parser.py     parse .nnue; dispatch on architecture hash
  features.py        HalfKAv2_hm + FullThreats feature indices; position diff
  probe.py           NNUE forward pass; activation diff; back-projection
  explain.py         group attributions → English
  pipeline.py        orchestration (Stockfish + probe + explain)
  render.py          self-contained HTML report
  cli.py             entry point
web/                 browser app (GitHub Pages)
  index.html         paste-a-PGN UI, rotaliate-styled
  js/                engine, NNUE parser/probe, pipeline, explain (JS port)
  nnue/              the committed network
  css/               generated piece stylesheet
tests/               JS↔Python parity + integration tests
tools/               build helpers (piece stylesheet generator)
vendor/              chess.js (test dep), rhosgfx pieces (CC0)
```

## Licensing

splainfish's own code is under the GamahCode license v1.2 (see `LICENSE`). The
browser app loads chessground, chess.js, and Stockfish from public CDNs at
runtime — they are used, not redistributed. The one committed third-party file
is the NNUE network (a Stockfish network, GPL-3.0), which GitHub Pages must
serve same-origin. Pieces are CC0. See `NOTICE` for the full breakdown.

## Requirements

- **Browser app**: any modern browser. No build step.
- **CLI**: Python 3.10+, numpy, python-chess, a Stockfish binary, and a matching
  `.nnue` (the `halfka-1536` net).
