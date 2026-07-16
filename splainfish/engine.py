"""
engine.py — Stockfish UCI wrapper.

Provides centipawn eval, best move, principal variation, and MultiPV
analysis for any position given as a FEN string.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional

import chess


# Centipawn value at which we clamp "forced mate" lines so they don't
# blow up downstream arithmetic.
MATE_CP = 30_000


@dataclass
class PVLine:
    rank: int           # 1 = best, 2 = second-best, …
    score_cp: int       # centipawns, from side-to-move perspective
    mate_in: Optional[int]  # None if not a forced mate
    moves: list[str]    # UCI move strings
    depth: int

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None

    def score_cp_white(self, board: chess.Board) -> int:
        """Return score in White's perspective (positive = White better)."""
        cp = MATE_CP if self.mate_in and self.mate_in > 0 else (
            -MATE_CP if self.mate_in else self.score_cp
        )
        return cp if board.turn == chess.WHITE else -cp


@dataclass
class PositionAnalysis:
    fen: str
    depth: int
    pv_lines: list[PVLine] = field(default_factory=list)

    @property
    def best(self) -> Optional[PVLine]:
        return self.pv_lines[0] if self.pv_lines else None

    def score_cp_white(self, board: chess.Board) -> int:
        if self.best is None:
            return 0
        return self.best.score_cp_white(board)


class StockfishEngine:
    """
    Thin wrapper around the Stockfish process via UCI.
    Spawns one process and reuses it for all queries.
    """

    def __init__(self, path: Optional[str] = None, depth: int = 18, multipv: int = 3):
        self.path = path or os.environ.get("STOCKFISH_PATH") or self._find_stockfish()
        self.depth = depth
        self.multipv = multipv
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _find_stockfish(self) -> str:
        candidates = [
            "stockfish",
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
            # also try PATH lookup
            try:
                subprocess.run([c, "--help"], capture_output=True, timeout=1)
                return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        raise FileNotFoundError(
            "Stockfish not found. Install it or set STOCKFISH_PATH."
        )

    def _start(self):
        self._proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name MultiPV value {self.multipv}")
        self._send("isready")
        self._read_until("readyok")

    def close(self):
        if self._proc:
            self._send("quit")
            self._proc.wait(timeout=5)
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _send(self, cmd: str):
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(self, marker: str) -> list[str]:
        lines = []
        while True:
            line = self._proc.stdout.readline().rstrip("\n")
            lines.append(line)
            if marker in line:
                return lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, fen: str) -> PositionAnalysis:
        """Return a PositionAnalysis for the given FEN."""
        with self._lock:
            self._send("ucinewgame")
            self._send(f"position fen {fen}")
            self._send(f"go depth {self.depth}")
            raw_lines = self._read_until("bestmove")

        return self._parse(fen, raw_lines)

    def _parse(self, fen: str, lines: list[str]) -> PositionAnalysis:
        # Collect the *last* info line for each MultiPV rank at the
        # highest depth actually reached.
        best_for_rank: dict[int, dict] = {}

        for line in lines:
            if not line.startswith("info"):
                continue
            tokens = line.split()
            if "multipv" not in tokens:
                continue

            def tok(key: str):
                try:
                    return tokens[tokens.index(key) + 1]
                except (ValueError, IndexError):
                    return None

            rank = int(tok("multipv") or 0)
            depth = int(tok("depth") or 0)
            seldepth = tok("seldepth")

            score_cp = None
            mate_in = None
            if "score" in tokens:
                si = tokens.index("score")
                kind = tokens[si + 1]
                val = int(tokens[si + 2])
                if kind == "cp":
                    score_cp = val
                elif kind == "mate":
                    mate_in = val
                    score_cp = MATE_CP if val > 0 else -MATE_CP

            moves = []
            if "pv" in tokens:
                pi = tokens.index("pv")
                moves = tokens[pi + 1:]

            if rank and (rank not in best_for_rank or depth >= best_for_rank[rank]["depth"]):
                best_for_rank[rank] = {
                    "rank": rank,
                    "depth": depth,
                    "score_cp": score_cp or 0,
                    "mate_in": mate_in,
                    "moves": moves,
                }

        pv_lines = [
            PVLine(
                rank=d["rank"],
                score_cp=d["score_cp"],
                mate_in=d["mate_in"],
                moves=d["moves"],
                depth=d["depth"],
            )
            for d in sorted(best_for_rank.values(), key=lambda x: x["rank"])
        ]

        return PositionAnalysis(
            fen=fen,
            depth=max((d["depth"] for d in best_for_rank.values()), default=0),
            pv_lines=pv_lines,
        )
