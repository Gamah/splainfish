"""
render.py — Generate a self-contained HTML report from a list of Explanation dicts.

The output is a single .html file with no external dependencies.
Features:
  - Interactive board (SVG pieces, click forward/back through moves)
  - Per-move eval bar showing centipawn change
  - Simple / Complex toggle (persists selection across moves)
  - Complex view: horizontal bar chart of attribution groups
  - Move list with quality glyphs; click to jump to any move
  - Responsive layout
"""

from __future__ import annotations

import json
from typing import Optional


# ---------------------------------------------------------------------------
# Chess piece SVGs (subset of Cburnett set, public domain)
# Embedded inline so the HTML is truly self-contained.
# ---------------------------------------------------------------------------

# We embed piece SVGs as data URIs. Each piece is a compact SVG path.
# Pieces: K Q R B N P in White and Black.
# Using Unicode chess symbols rendered via SVG text for compactness.
_PIECE_UNICODE = {
    "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
    "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
}

_PIECE_FEN_MAP = {
    "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
    "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
}


def _fen_to_board_array(fen: str) -> list[Optional[str]]:
    """Convert FEN position string to 64-element array (a8=0 … h1=63) of piece keys."""
    position = fen.split()[0]
    board = []
    for char in position:
        if char == "/":
            continue
        elif char.isdigit():
            board.extend([None] * int(char))
        else:
            board.append(_PIECE_FEN_MAP.get(char))
    return board   # index 0 = a8, index 63 = h1


def render_html(results: list[dict], title: str = "SplainFish") -> str:
    """Render a list of Explanation dicts into a self-contained HTML string."""

    # Embed the data as a JS variable
    json_data = json.dumps(results, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
/* =========================================================
   Reset & base
   ========================================================= */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:        #1a1a2e;
  --surface:   #16213e;
  --card:      #0f3460;
  --accent:    #e94560;
  --accent2:   #533483;
  --text:      #eaeaea;
  --muted:     #8a8aaa;
  --border:    #2a2a4a;
  --sq-light:  #f0d9b5;
  --sq-dark:   #b58863;
  --sq-hl:     rgba(235, 97, 0, 0.55);
  --radius:    8px;
  font-family: 'Segoe UI', system-ui, sans-serif;
  color: var(--text);
  background: var(--bg);
}}
body {{ display: flex; flex-direction: column; min-height: 100vh; }}

/* =========================================================
   Layout
   ========================================================= */
header {{
  padding: 16px 24px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
}}
header h1 {{ font-size: 1.2rem; font-weight: 600; color: var(--text); }}
header .sub {{ font-size: 0.8rem; color: var(--muted); }}

.layout {{
  display: grid;
  grid-template-columns: 220px 1fr 340px;
  grid-template-rows: 1fr;
  flex: 1;
  overflow: hidden;
  height: calc(100vh - 57px);
}}

/* =========================================================
   Move list (left sidebar)
   ========================================================= */
.sidebar {{
  background: var(--surface);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px 0;
}}
.move-pair {{
  display: grid;
  grid-template-columns: 28px 1fr 1fr;
  align-items: stretch;
  font-size: 0.82rem;
}}
.move-num {{
  color: var(--muted);
  padding: 5px 6px;
  text-align: right;
  line-height: 1.6;
  font-size: 0.75rem;
}}
.move-btn {{
  padding: 5px 6px;
  cursor: pointer;
  border-radius: 4px;
  transition: background 0.15s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: flex; align-items: center; gap: 3px;
}}
.move-btn:hover {{ background: var(--border); }}
.move-btn.active {{ background: var(--accent2); color: #fff; }}
.move-btn .glyph {{ font-size: 0.9em; }}
.move-btn .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  flex-shrink: 0;
}}

/* =========================================================
   Board panel (center)
   ========================================================= */
.board-panel {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 20px;
  overflow: auto;
}}
.board-wrap {{
  position: relative;
  user-select: none;
}}
.board-svg {{
  width: min(420px, 90vw);
  height: min(420px, 90vw);
  display: block;
}}
.board-controls {{
  display: flex; gap: 8px; align-items: center;
}}
.ctrl-btn {{
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 18px;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 0.9rem;
  transition: background 0.15s;
}}
.ctrl-btn:hover {{ background: var(--accent2); }}
.ctrl-btn:disabled {{ opacity: 0.4; cursor: default; }}

.move-label {{
  font-size: 1.1rem;
  font-weight: 600;
  text-align: center;
  min-height: 1.6em;
}}
.eval-bar-wrap {{
  width: min(420px, 90vw);
  display: flex; align-items: center; gap: 10px;
}}
.eval-bar-bg {{
  flex: 1; height: 10px; border-radius: 5px;
  background: linear-gradient(to right, #222 50%, #eee 50%);
  position: relative; overflow: hidden;
}}
.eval-bar-fill {{
  position: absolute; top: 0; bottom: 0; left: 50%;
  transition: width 0.3s, background 0.3s;
  border-radius: 5px;
}}
.eval-num {{ font-size: 0.85rem; color: var(--muted); min-width: 48px; text-align: right; }}

/* =========================================================
   Explanation panel (right)
   ========================================================= */
.explain-panel {{
  background: var(--surface);
  border-left: 1px solid var(--border);
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.quality-badge {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}}
.explain-headline {{
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.4;
}}
.explain-para {{
  font-size: 0.88rem;
  color: #c8c8e0;
  line-height: 1.6;
}}

/* Toggle */
.view-toggle {{
  display: flex;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  width: fit-content;
}}
.toggle-btn {{
  padding: 6px 16px;
  font-size: 0.82rem;
  cursor: pointer;
  background: transparent;
  color: var(--muted);
  border: none;
  transition: background 0.15s, color 0.15s;
}}
.toggle-btn.active {{
  background: var(--accent2);
  color: #fff;
}}

.view-simple, .view-complex {{ display: none; flex-direction: column; gap: 12px; }}
.view-simple.visible, .view-complex.visible {{ display: flex; }}

/* Complex: attribution bars */
.attr-group {{
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px;
  background: var(--card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}}
.attr-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}}
.attr-name {{
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text);
  flex: 1;
}}
.attr-cp {{
  font-size: 0.78rem;
  color: var(--muted);
  white-space: nowrap;
}}
.attr-bar-bg {{
  height: 6px;
  border-radius: 3px;
  background: var(--border);
  overflow: hidden;
}}
.attr-bar-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s;
}}
.attr-sentence {{
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.5;
  margin-top: 2px;
}}
.complex-note {{
  font-size: 0.72rem;
  color: var(--muted);
  line-height: 1.5;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}}

/* empty state */
.empty-state {{
  color: var(--muted);
  font-size: 0.88rem;
  text-align: center;
  margin-top: 40px;
}}

/* scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>SplainFish</h1>
    <div class="sub">NNUE-backed move analysis · Stockfish 18</div>
  </div>
</header>

<div class="layout">
  <!-- LEFT: move list -->
  <div class="sidebar" id="moveList"></div>

  <!-- CENTER: board -->
  <div class="board-panel">
    <div class="move-label" id="moveLabel">Select a move</div>
    <div class="eval-bar-wrap">
      <span style="font-size:0.75rem;color:var(--muted)">Black</span>
      <div class="eval-bar-bg">
        <div class="eval-bar-fill" id="evalBarFill"></div>
      </div>
      <span style="font-size:0.75rem;color:var(--muted)">White</span>
      <span class="eval-num" id="evalNum"></span>
    </div>
    <div class="board-wrap">
      <svg class="board-svg" id="boardSvg" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="board-controls">
      <button class="ctrl-btn" id="btnPrev" onclick="navigate(-1)">◀ Prev</button>
      <button class="ctrl-btn" id="btnNext" onclick="navigate(1)">Next ▶</button>
    </div>
  </div>

  <!-- RIGHT: explanation -->
  <div class="explain-panel" id="explainPanel">
    <div class="empty-state">Select a move from the list to see the explanation.</div>
  </div>
</div>

<script>
// =========================================================
// Data
// =========================================================
const MOVES = {json_data};

// =========================================================
// State
// =========================================================
let currentIdx = -1;
// "simple" | "complex"
let viewMode = localStorage.getItem("ce_viewMode") || "simple";

// =========================================================
// Board rendering
// =========================================================
const PIECE_UNICODE = {{
  wK:"♔",wQ:"♕",wR:"♖",wB:"♗",wN:"♘",wP:"♙",
  bK:"♚",bQ:"♛",bR:"♜",bB:"♝",bN:"♞",bP:"♟"
}};
const PIECE_FEN = {{
  K:"wK",Q:"wQ",R:"wR",B:"wB",N:"wN",P:"wP",
  k:"bK",q:"bQ",r:"bR",b:"bB",n:"bN",p:"bP"
}};

function fenToArray(fen) {{
  const pos = fen.split(" ")[0];
  const arr = [];
  for (const ch of pos) {{
    if (ch === "/") continue;
    else if (/\\d/.test(ch)) for (let i=0;i<+ch;i++) arr.push(null);
    else arr.push(PIECE_FEN[ch] || null);
  }}
  return arr; // index 0 = a8, 63 = h1
}}

function renderBoard(fen, fromSq, toSq) {{
  const svg = document.getElementById("boardSvg");
  const arr = fenToArray(fen);
  const SQ = 50;
  const files = "abcdefgh";
  let out = "";

  // Squares
  for (let r = 0; r < 8; r++) {{
    for (let f = 0; f < 8; f++) {{
      const idx = r * 8 + f;
      const x = f * SQ, y = r * SQ;
      const light = (r + f) % 2 === 0;
      let fill = light ? "#f0d9b5" : "#b58863";

      // Rank/file labels
      if (f === 0) out += `<text x="${{x+2}}" y="${{y+12}}" font-size="10" fill="${{light?"#b58863":"#f0d9b5"}}" font-family="sans-serif">${{8-r}}</text>`;
      if (r === 7) out += `<text x="${{x+38}}" y="${{y+48}}" font-size="10" fill="${{light?"#b58863":"#f0d9b5"}}" font-family="sans-serif">${{files[f]}}</text>`;

      // Move highlight
      const sqName = files[f] + (8-r);
      const hl = (sqName === fromSq || sqName === toSq);
      if (hl) fill = light ? "#cdd16f" : "#aaa23a";

      out += `<rect x="${{x}}" y="${{y}}" width="${{SQ}}" height="${{SQ}}" fill="${{fill}}"/>`;

      // Piece
      const piece = arr[idx];
      if (piece) {{
        const isWhite = piece[0] === "w";
        const sym = PIECE_UNICODE[piece];
        out += `<text x="${{x+SQ/2}}" y="${{y+SQ/2+12}}"
          text-anchor="middle" font-size="34"
          fill="${{isWhite ? "#fff" : "#000"}}"
          stroke="${{isWhite ? "#000" : "#fff"}}"
          stroke-width="0.8"
          paint-order="stroke"
          font-family="serif">${{sym}}</text>`;
      }}
    }}
  }}

  svg.innerHTML = out;
}}

function uciToSquares(uci) {{
  if (!uci || uci.length < 4) return [null, null];
  return [uci.slice(0,2), uci.slice(2,4)];
}}

// =========================================================
// Eval bar
// =========================================================
function updateEvalBar(cp) {{
  const fill = document.getElementById("evalBarFill");
  const num  = document.getElementById("evalNum");
  // cp is White's perspective; clamp to ±600cp for display
  const clamped = Math.max(-600, Math.min(600, cp));
  const pct = (clamped / 1200) * 50;   // ±50% from center
  if (cp >= 0) {{
    fill.style.left = "50%";
    fill.style.width = pct + "%";
    fill.style.background = "#eee";
  }} else {{
    fill.style.left = (50 + pct) + "%";
    fill.style.width = (-pct) + "%";
    fill.style.background = "#333";
  }}
  const pawns = Math.abs(cp) / 100;
  num.textContent = (cp >= 0 ? "+" : "-") + pawns.toFixed(2);
}}

// =========================================================
// Move list
// =========================================================
function buildMoveList() {{
  const el = document.getElementById("moveList");
  let html = "";
  let pairOpen = false;
  let moveNum = null;

  for (let i = 0; i < MOVES.length; i++) {{
    const m = MOVES[i];
    if (m.color === "white") {{
      if (pairOpen) html += "</div>";
      html += `<div class="move-pair">`;
      html += `<div class="move-num">${{m.move_number}}.</div>`;
      pairOpen = true;
      moveNum = m.move_number;
    }}
    const glyph = m.quality_glyph || "";
    const dot = `<span class="dot" style="background:${{m.quality_color}}"></span>`;
    html += `<div class="move-btn" id="mb${{i}}" onclick="selectMove(${{i}})">
      ${{dot}}<span>${{m.move_san}}${{glyph}}</span></div>`;

    // If black move or last move, close pair
    if (m.color === "black" || i === MOVES.length - 1) {{
      if (m.color === "white") html += `<div></div>`; // empty black slot
      html += "</div>";
      pairOpen = false;
    }}
  }}
  el.innerHTML = html;
}}

// =========================================================
// Explanation panel
// =========================================================
function renderExplanation(m) {{
  const panel = document.getElementById("explainPanel");

  // Attribution bar chart for complex view
  let attrHtml = "";
  if (m.complex_groups && m.complex_groups.length > 0) {{
    const maxPct = Math.max(...m.complex_groups.map(g => g.pct_of_total));
    for (const g of m.complex_groups) {{
      const barW = maxPct > 0 ? (g.pct_of_total / maxPct * 100) : 0;
      const barColor = g.direction === "positive" ? "#5aa65a" : "#c03030";
      const cpSign = g.contribution_cp >= 0 ? "+" : "";
      attrHtml += `
      <div class="attr-group">
        <div class="attr-header">
          <div class="attr-name">${{g.group}}</div>
          <div class="attr-cp">${{cpSign}}${{g.contribution_cp.toFixed(2)}} pawns (${{g.pct_of_total}}%)</div>
        </div>
        <div class="attr-bar-bg">
          <div class="attr-bar-fill" style="width:${{barW}}%;background:${{barColor}}"></div>
        </div>
        <div class="attr-sentence">${{g.sentence}}</div>
      </div>`;
    }}
    attrHtml += `<div class="complex-note">${{m.complex_note || ""}}</div>`;
  }} else {{
    attrHtml = `<div class="explain-para">No attribution data available for this move.</div>`;
  }}

  // Simple paragraphs
  const simpleParas = (m.simple_paragraphs || [])
    .map(p => `<div class="explain-para">${{p}}</div>`).join("");

  const isSimple = viewMode === "simple";
  const isCx     = viewMode === "complex";

  panel.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:4px">
      <span class="quality-badge" style="background:${{m.quality_color}}22;color:${{m.quality_color}};border:1px solid ${{m.quality_color}}44">
        ${{m.quality_label}}
      </span>
      <div class="explain-headline">${{m.simple_headline || m.move_san}}</div>
    </div>

    <div class="view-toggle">
      <button class="toggle-btn ${{isSimple?"active":""}}" onclick="setView('simple')">Simple</button>
      <button class="toggle-btn ${{isCx?"active":""}}" onclick="setView('complex')">Detailed</button>
    </div>

    <div class="view-simple ${{isSimple?"visible":""}}" id="viewSimple">
      ${{simpleParas || '<div class="explain-para">No explanation available.</div>'}}
    </div>
    <div class="view-complex ${{isCx?"visible":""}}" id="viewComplex">
      ${{attrHtml}}
    </div>
  `;
}}

// =========================================================
// Navigation
// =========================================================
function selectMove(idx) {{
  if (idx < 0 || idx >= MOVES.length) return;
  currentIdx = idx;
  const m = MOVES[idx];

  // Update active in list
  document.querySelectorAll(".move-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById("mb" + idx);
  if (btn) {{ btn.classList.add("active"); btn.scrollIntoView({{block:"nearest"}}); }}

  // Board
  const [fromSq, toSq] = uciToSquares(m.move_uci);
  renderBoard(m.fen_after, fromSq, toSq);

  // Label
  const num = m.move_number;
  const dots = m.color === "black" ? "..." : ".";
  document.getElementById("moveLabel").textContent =
    `${{num}}${{dots}} ${{m.move_san}}${{m.quality_glyph || ""}}`;

  // Eval bar
  updateEvalBar(m.sf_eval_after ?? m.eval_after_cp ?? 0);

  // Explanation
  renderExplanation(m);

  // Nav buttons
  document.getElementById("btnPrev").disabled = idx === 0;
  document.getElementById("btnNext").disabled = idx === MOVES.length - 1;
}}

function navigate(dir) {{
  selectMove(currentIdx + dir);
}}

function setView(mode) {{
  viewMode = mode;
  localStorage.setItem("ce_viewMode", mode);
  // Update toggle buttons
  document.querySelectorAll(".toggle-btn").forEach(b => {{
    b.classList.toggle("active", b.textContent.toLowerCase().startsWith(mode));
  }});
  // Swap visible divs
  const simple  = document.getElementById("viewSimple");
  const complex = document.getElementById("viewComplex");
  if (simple)  simple.classList.toggle("visible",  mode === "simple");
  if (complex) complex.classList.toggle("visible", mode === "complex");
}}

// Keyboard nav
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft")  navigate(-1);
  if (e.key === "ArrowRight") navigate(1);
}});

// =========================================================
// Init
// =========================================================
buildMoveList();
if (MOVES.length > 0) {{
  // Start on first significant move, or just move 0
  const firstBad = MOVES.findIndex(m => ["inaccuracy","mistake","blunder"].includes(m.quality));
  selectMove(firstBad >= 0 ? firstBad : 0);
}}
</script>
</body>
</html>"""

    return html
