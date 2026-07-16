"""
render.py — Generate a self-contained HTML report from Explanation dicts.

The output is a single .html file with no external dependencies: the board uses
the rhosgfx piece set (CC0) inlined as data URIs, styled to match the browser
app (web/index.html), which is adapted from ../rotaliate. Features:

  - Board (click / arrow-key through moves), rhosgfx pieces
  - Per-move eval bar (Stockfish centipawns)
  - Simple / Detailed toggle (persists selection)
  - Detailed view: horizontal bar chart of attribution groups
  - Move list with quality dots; click to jump

Unlike the browser app this file references nothing external, so it can be
emailed or hosted anywhere.
"""

from __future__ import annotations

import json
from typing import Optional

from tools.gen_pieces_css import piece_data_uris

_PIECE_FEN_MAP = {
    "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
    "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
}


def render_html(results: list[dict], title: str = "splainfish") -> str:
    """Render a list of Explanation dicts into a self-contained HTML string."""
    json_data = json.dumps(results, separators=(",", ":"))
    piece_uris = json.dumps(piece_data_uris(), separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:#101014; --surface:#16161c; --panel:#1b1b22; --border:#2a2a33;
  --text:#d6d6dc; --muted:#7a7a85; --accent:#5b8def; --green:#3dd68c;
  --red:#e5484d; --yellow:#f0c000;
  --mono: ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  --sq-light:#b9c4cc; --sq-dark:#5f7688;
  --sf-white:#e8e8ea; --sf-black:#34343d;
}}
body {{ background:var(--bg); color:var(--text);
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif; font-size:13px;
  min-height:100dvh; display:flex; flex-direction:column; overflow-x:hidden; }}
header {{ background:var(--surface); border-bottom:1px solid var(--border);
  padding:10px 16px; display:flex; align-items:center; gap:12px; }}
h1 {{ font-family:var(--mono); font-size:15px; font-weight:600; letter-spacing:.5px; }}
h1 .fish {{ color:var(--accent); }}
header .tagline {{ color:var(--muted); font-size:12px; }}
header .spacer {{ margin-left:auto; }}
#bmc {{ font-size:12px; color:var(--muted); text-decoration:none; padding:5px 12px;
  border-radius:4px; transition:all .15s; }}
#bmc:hover {{ color:var(--yellow); background:rgba(240,192,0,.06); }}

.layout {{ flex:1; display:grid; grid-template-columns:210px 1fr 340px; min-height:0; }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} }}

.sidebar {{ background:var(--surface); border-right:1px solid var(--border);
  overflow-y:auto; padding:12px 8px; }}
.move-pair {{ display:grid; grid-template-columns:26px 1fr 1fr; align-items:center; gap:2px; }}
.move-num {{ color:var(--muted); font-size:11px; font-family:var(--mono);
  text-align:right; padding-right:4px; }}
.move-btn {{ display:flex; align-items:center; gap:5px; padding:4px 6px;
  background:none; border:none; color:var(--text); cursor:pointer; border-radius:4px;
  font-size:12.5px; font-family:inherit; text-align:left; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.move-btn:hover {{ background:var(--panel); }}
.move-btn.active {{ background:rgba(91,141,239,.15); color:var(--accent); }}
.dot {{ width:6px; height:6px; border-radius:50%; flex-shrink:0; }}

.board-panel {{ display:flex; flex-direction:column; align-items:center;
  gap:14px; padding:20px; overflow:auto; }}
#move-label {{ font-size:15px; font-weight:600; font-family:var(--mono); min-height:1.4em; }}
.eval-wrap {{ width:min(440px,88vw); display:flex; align-items:center; gap:10px;
  font-size:11px; color:var(--muted); }}
.eval-bg {{ flex:1; height:8px; border-radius:5px; position:relative; overflow:hidden;
  background:linear-gradient(to right,var(--sf-black) 50%,var(--sf-white) 50%); }}
#eval-fill {{ position:absolute; top:0; bottom:0; left:50%; border-radius:5px;
  transition:width .3s,background .3s; }}
#eval-num {{ font-family:var(--mono); color:var(--text); min-width:52px; text-align:right; }}

.board {{ width:min(440px,88vw); aspect-ratio:1; display:grid;
  grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr);
  border:1px solid var(--border); border-radius:4px; overflow:hidden;
  box-shadow:0 8px 32px rgba(0,0,0,.5); user-select:none; }}
.sq {{ position:relative; }}
.sq.light {{ background:var(--sq-light); }}
.sq.dark {{ background:var(--sq-dark); }}
.sq.hl::before {{ content:""; position:absolute; inset:0; background:rgba(91,141,239,.32); }}
.sq .pc {{ position:absolute; inset:6%; background-size:contain;
  background-repeat:no-repeat; background-position:center; z-index:1; }}
.sq .coord {{ position:absolute; font-size:9px; font-family:var(--mono); opacity:.7; }}
.sq .coord.file {{ right:2px; bottom:1px; }}
.sq .coord.rank {{ left:2px; top:1px; }}

.controls {{ display:flex; gap:8px; }}
.ctrl {{ padding:7px 18px; background:var(--panel); border:1px solid var(--border);
  color:var(--text); border-radius:4px; cursor:pointer; font-size:13px;
  font-family:inherit; transition:all .12s; }}
.ctrl:hover:not(:disabled) {{ border-color:var(--muted); }}
.ctrl:disabled {{ opacity:.35; cursor:default; }}

.explain {{ background:var(--surface); border-left:1px solid var(--border);
  overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px;
  font-weight:700; letter-spacing:.5px; text-transform:uppercase; }}
.headline {{ font-size:15px; font-weight:600; line-height:1.4; }}
.para {{ font-size:13px; color:#c3c3d0; line-height:1.65; }}
.toggle {{ display:flex; border:1px solid var(--border); border-radius:4px;
  overflow:hidden; width:fit-content; }}
.tbtn {{ padding:6px 16px; font-size:12px; cursor:pointer; background:transparent;
  color:var(--muted); border:none; font-family:inherit; transition:all .12s; }}
.tbtn.active {{ background:rgba(91,141,239,.15); color:var(--accent); }}
.vbody {{ display:flex; flex-direction:column; gap:10px; }}
.attr {{ display:flex; flex-direction:column; gap:5px; padding:10px;
  background:var(--panel); border:1px solid var(--border); border-radius:4px; }}
.attr-h {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
.attr-name {{ font-size:12.5px; font-weight:600; }}
.attr-cp {{ font-size:11px; color:var(--muted); font-family:var(--mono); }}
.attr-bg {{ height:6px; border-radius:3px; background:var(--border); overflow:hidden; }}
.attr-fill {{ height:100%; border-radius:3px; transition:width .3s; }}
.attr-s {{ font-size:12px; color:var(--muted); line-height:1.5; }}
.note {{ font-size:11px; color:var(--muted); line-height:1.5;
  border-top:1px solid var(--border); padding-top:10px; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
</style>
</head>
<body>
<header>
  <h1><span class="fish">splain</span>fish</h1>
  <span class="tagline">what the engine actually reacted to</span>
  <span class="spacer"></span>
  <a id="bmc" href="https://buymeacoffee.com/gamah" target="_blank" rel="noopener">☕ buy me a coffee</a>
</header>
<div class="layout">
  <div class="sidebar" id="move-list"></div>
  <div class="board-panel">
    <div id="move-label">—</div>
    <div class="eval-wrap">
      <span>Black</span>
      <div class="eval-bg"><div id="eval-fill"></div></div>
      <span>White</span><span id="eval-num"></span>
    </div>
    <div class="board" id="board"></div>
    <div class="controls">
      <button class="ctrl" id="prev">◀ Prev</button>
      <button class="ctrl" id="next">Next ▶</button>
    </div>
  </div>
  <div class="explain" id="explain"></div>
</div>
<script>
const MOVES = {json_data};
const PIECES = {piece_uris};
const FEN_MAP = {{K:"wK",Q:"wQ",R:"wR",B:"wB",N:"wN",P:"wP",k:"bK",q:"bQ",r:"bR",b:"bB",n:"bN",p:"bP"}};
let idx = -1;
let viewMode = localStorage.getItem("sf_viewMode") || "simple";

function fenToArray(fen) {{
  const pos = fen.split(" ")[0]; const arr = [];
  for (const ch of pos) {{
    if (ch === "/") continue;
    else if (/\\d/.test(ch)) for (let i=0;i<+ch;i++) arr.push(null);
    else arr.push(FEN_MAP[ch] || null);
  }}
  return arr; // 0=a8 .. 63=h1
}}
function esc(s) {{ return String(s).replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}}[c])); }}

function renderBoard(fen, from, to) {{
  const arr = fenToArray(fen); const files="abcdefgh"; let html="";
  for (let r=0;r<8;r++) for (let f=0;f<8;f++) {{
    const i=r*8+f; const light=(r+f)%2===0;
    const sqName=files[f]+(8-r);
    const hl=(sqName===from||sqName===to)?" hl":"";
    let inner="";
    if (f===0) inner+=`<span class="coord rank" style="color:${{light?'var(--sq-dark)':'var(--sq-light)'}}">${{8-r}}</span>`;
    if (r===7) inner+=`<span class="coord file" style="color:${{light?'var(--sq-dark)':'var(--sq-light)'}}">${{files[f]}}</span>`;
    const p=arr[i];
    if (p) inner+=`<span class="pc" style="background-image:url('${{PIECES[p]}}')"></span>`;
    html+=`<div class="sq ${{light?'light':'dark'}}${{hl}}">${{inner}}</div>`;
  }}
  document.getElementById("board").innerHTML=html;
}}
function uci(s) {{ return (!s||s.length<4)?[null,null]:[s.slice(0,2),s.slice(2,4)]; }}

function updateEval(cp) {{
  const fill=document.getElementById("eval-fill"), num=document.getElementById("eval-num");
  const c=Math.max(-600,Math.min(600,cp)), pct=(c/1200)*50;
  if (cp>=0) {{ fill.style.left="50%"; fill.style.width=pct+"%"; fill.style.background="var(--sf-white)"; }}
  else {{ fill.style.left=(50+pct)+"%"; fill.style.width=(-pct)+"%"; fill.style.background="var(--sf-black)"; }}
  num.textContent=(cp>=0?"+":"−")+(Math.abs(cp)/100).toFixed(2);
}}

function buildList() {{
  let html="", open=false;
  for (let i=0;i<MOVES.length;i++) {{
    const m=MOVES[i];
    if (m.color==="white") {{ if(open) html+="</div>"; html+=`<div class="move-pair"><span class="move-num">${{m.move_number}}.</span>`; open=true; }}
    html+=`<button class="move-btn" id="mb${{i}}" onclick="sel(${{i}})"><span class="dot" style="background:${{m.quality_color}}"></span><span>${{esc(m.move_san)}}${{esc(m.quality_glyph||"")}}</span></button>`;
    if (m.color==="black"||i===MOVES.length-1) {{ if(m.color==="white") html+="<span></span>"; html+="</div>"; open=false; }}
  }}
  document.getElementById("move-list").innerHTML=html;
}}

function renderExplain(m) {{
  let attr="";
  if (m.complex_groups && m.complex_groups.length) {{
    const mx=Math.max(...m.complex_groups.map(g=>g.pct_of_total));
    for (const g of m.complex_groups) {{
      const w=mx>0?(g.pct_of_total/mx*100):0;
      const col=g.direction==="positive"?"var(--green)":"var(--red)";
      attr+=`<div class="attr"><div class="attr-h"><span class="attr-name">${{esc(g.group)}}</span><span class="attr-cp">${{g.pct_of_total}}%</span></div><div class="attr-bg"><div class="attr-fill" style="width:${{w}}%;background:${{col}}"></div></div><div class="attr-s">${{esc(g.sentence)}}</div></div>`;
    }}
    attr+=`<div class="note">${{esc(m.complex_note||"")}}</div>`;
  }} else attr=`<div class="para">No attribution data for this move.</div>`;
  const paras=(m.simple_paragraphs||[]).map(p=>`<div class="para">${{esc(p)}}</div>`).join("");
  const s=viewMode==="simple";
  document.getElementById("explain").innerHTML=`
    <div><span class="badge" style="background:${{m.quality_color}}22;color:${{m.quality_color}};border:1px solid ${{m.quality_color}}55">${{esc(m.quality_label)}}</span></div>
    <div class="headline">${{esc(m.simple_headline||m.move_san)}}</div>
    <div class="toggle"><button class="tbtn ${{s?'active':''}}" onclick="setView('simple')">Simple</button><button class="tbtn ${{s?'':'active'}}" onclick="setView('complex')">Detailed</button></div>
    <div class="vbody vsimple" ${{s?'':'hidden'}}>${{paras||'<div class="para">No explanation available.</div>'}}</div>
    <div class="vbody vcomplex" ${{s?'hidden':''}}>${{attr}}</div>`;
}}
function setView(mode) {{
  viewMode=mode; localStorage.setItem("sf_viewMode",mode);
  document.querySelectorAll(".tbtn").forEach(b=>b.classList.toggle("active",b.textContent.toLowerCase().startsWith(mode==="simple"?"simple":"detail")));
  const si=document.querySelector(".vsimple"), cx=document.querySelector(".vcomplex");
  if (si) si.hidden=mode!=="simple"; if (cx) cx.hidden=mode!=="complex";
}}

function sel(i) {{
  if (i<0||i>=MOVES.length) return; idx=i; const m=MOVES[i];
  document.querySelectorAll(".move-btn").forEach(b=>b.classList.remove("active"));
  const b=document.getElementById("mb"+i); if(b){{b.classList.add("active");b.scrollIntoView({{block:"nearest"}});}}
  const [f,t]=uci(m.move_uci); renderBoard(m.fen_after,f,t);
  const dots=m.color==="black"?"…":"."; document.getElementById("move-label").textContent=`${{m.move_number}}${{dots}} ${{m.move_san}}${{m.quality_glyph||""}}`;
  updateEval(m.sf_eval_after ?? m.eval_after_cp ?? 0);
  renderExplain(m);
  document.getElementById("prev").disabled=i===0;
  document.getElementById("next").disabled=i===MOVES.length-1;
}}
function nav(d) {{ sel(idx+d); }}
document.getElementById("prev").onclick=()=>nav(-1);
document.getElementById("next").onclick=()=>nav(1);
document.addEventListener("keydown",e=>{{ if(e.key==="ArrowLeft")nav(-1); if(e.key==="ArrowRight")nav(1); }});
buildList();
if (MOVES.length) {{ const bad=MOVES.findIndex(m=>["inaccuracy","mistake","blunder"].includes(m.quality)); sel(bad>=0?bad:0); }}
</script>
</body>
</html>"""
