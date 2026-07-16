# splainfish

Explains what the Stockfish NNUE network *actually reacted to* between two chess positions — in plain English a novice can understand.

## What it does

Most chess analysis tools tell you **how much** a move changed the evaluation. This tool explains **why** — by running the NNUE forward pass on both positions, diffing the network activations layer-by-layer (FT → L1 → L2), and back-projecting the output delta onto the input features that changed.

The result is a ranked list of chess concepts (e.g. "enemy queen threatening king area", "own pawn shield", "own knight activity") that the network most responded to, with two views:

- **Simple**: 2-3 human sentences, novice-friendly
- **Detailed**: bar chart of ranked feature groups with contribution magnitudes

No LLM. All explanation is deterministic from NNUE activations.

## How it works

```
Position A ──► Stockfish UCI ──► eval_before, best_move, PV
     │
     │  HalfKAv2_hm + FullThreats feature computation
     │
     ▼
Feature diff (gained / lost indices)
     │
     ▼
NNUE forward pass (re-implemented in numpy)
  FT accumulator (L1 per perspective)
  → fc_0 (SqrClippedReLU + ClippedReLU)
  → fc_1 (SqrClippedReLU + ClippedReLU)
  → fc_2 + skip connection
     │
     ▼
Activation delta at each layer
     │
     ▼
Back-projection: fc_2 → fc_1 → fc_0 → FT
(first-order Taylor approximation at midpoint activations)
     │
     ▼
Per-feature attribution scores
     │
     ▼
Grouped by semantic concept → English sentences
     │
     ▼
Self-contained HTML report (board + move list + explanations)
```

## Supported network versions

| Version | Hash | Architecture |
|---------|------|-------------|
| SF16 (apt) | `0x7AF32F20` | HalfKAv2_hm, L1=1536, FC0=16, FC1=32 |
| SF18 (latest) | `0x6A448AFA` | HalfKAv2_hm + FullThreats, L1=1024, L2=32, L3=32 |

The parser reads the version from the file header and adapts automatically.

## Quick start

```bash
# Install deps and build Stockfish (clones latest from GitHub)
make setup

# Analyse a PGN
make report PGN=mygame.pgn OUTPUT=mygame.html

# Run the built-in demo (Immortal Game, 1851)
make demo
```

## Manual usage

```bash
# Activate virtualenv
source .venv/bin/activate

# Single FEN + move
python -m splainfish.cli \
  --fen "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" \
  --move f6g4 \
  --stockfish ./stockfish-src/src/stockfish \
  --nnue ./stockfish-src/src/nn-*.nnue \
  --html out.html

# PGN file, only show inaccuracies/mistakes/blunders
python -m splainfish.cli \
  --pgn game.pgn --html out.html \
  --only-mistakes --depth 20

# JSON output (raw attribution data)
python -m splainfish.cli --pgn game.pgn --json
```

## HTML viewer features

- Interactive board (click forward/back, keyboard arrows)
- Eval bar showing centipawn shift
- Move list with quality glyphs (??/!/etc.)
- **Simple / Detailed toggle** (remembers your preference)
  - Simple: plain English explanation
  - Detailed: horizontal bar chart of NNUE feature group contributions
- Jumps to first mistake on load
- Zero external dependencies — host anywhere

## Architecture

```
splainfish/
  nnue_parser.py   Parse .nnue binary; SF16 (COMPRESSED_LEB128) and SF18 (LEB128) formats
  features.py      HalfKAv2_hm + FullThreats feature index computation; position diff
  probe.py         NNUE forward pass (numpy); activation diff; back-projection attribution
  explain.py       Group attributions → English sentences; simple + complex views
  pipeline.py      Orchestration: SF UCI + NNUE probe + explain → JSON
  render.py        Self-contained HTML generator
  cli.py           Entry point (--pgn / --fen / --html / --json)
Makefile           Clone SF, build, export NNUE, run pipeline
```

## Limitations and honesty

- **Eval accuracy**: our numpy forward pass approximates the quantized integer network. The centipawn numbers shown in the HTML use Stockfish's actual UCI eval (exact); our internal eval is only used to determine attribution direction.
- **Attribution is a first-order approximation**: the back-projection uses a linear Taylor approximation at the midpoint activation. For highly nonlinear positions (forced mates, tactical combinations) the attribution groups may spread more diffusely than the "real" reason.
- **SF16 apt export bug**: the apt-packaged SF16 truncates the last layer stack when `export_net` is called. We detect and recover from this by cloning the penultimate stack.
- **FullThreats (SF18 only)**: the threat feature group is computed but its attribution is not yet fully back-projected (the threat features feed the FT accumulator separately; this path is a TODO).

## Requirements

- Python 3.10+
- numpy ≥ 1.24
- python-chess ≥ 1.10
- Stockfish binary (any version ≥ 16 for UCI; 16 or 18 for NNUE probing)
- A `.nnue` file matching your Stockfish version
