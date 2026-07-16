"""
pipeline.py — Top-level orchestration.

1. Accepts a PGN string, two FENs, or a FEN+move
2. For each move: runs Stockfish UCI (for eval + best move), computes features,
   runs the NNUE probe, builds the Explanation
3. Returns a list of Explanation dicts ready for JSON serialisation

The .nnue file is loaded once and reused for all positions.
"""

from __future__ import annotations

import dataclasses
import json
import io
import os
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

from .engine import StockfishEngine
from .nnue_parser import load as load_nnue, NNUEWeights
from .features import compute_features, diff_features
from .probe import probe
from .explain import build, Explanation


def _find_nnue(sf_binary: str) -> Path:
    """
    Ask Stockfish which .nnue file it's using via UCI option 'EvalFile',
    then locate that file relative to the binary or in known paths.
    """
    import subprocess, re
    proc = subprocess.run(
        [sf_binary], input="uci\nquit\n", capture_output=True, text=True, timeout=10
    )
    # Look for: option name EvalFile type string default <name>
    m = re.search(r"option name EvalFile type string default (\S+)", proc.stdout)
    if not m:
        raise RuntimeError("Could not determine NNUE filename from Stockfish UCI output")
    nnue_name = m.group(1)

    # Search for the file next to the binary, in CWD, and in common paths
    candidates = [
        Path(sf_binary).parent / nnue_name,
        Path(".") / nnue_name,
        Path("stockfish") / nnue_name,
        Path("/usr/share/games/stockfish") / nnue_name,
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find NNUE file '{nnue_name}'. "
        f"Make sure it is next to the Stockfish binary or in the current directory."
    )


def _pv_to_san(pv_uci: list[str], board: chess.Board, max_moves: int = 6) -> list[str]:
    san = []
    b = board.copy()
    for uci in pv_uci[:max_moves]:
        try:
            m = chess.Move.from_uci(uci)
            san.append(b.san(m))
            b.push(m)
        except Exception:
            break
    return san


def analyse_game(
    pgn_text: str,
    sf_binary: Optional[str] = None,
    nnue_path: Optional[str] = None,
    depth: int = 18,
    multipv: int = 3,
    verbose: bool = False,
    only_mistakes: bool = False,
) -> list[dict]:
    """
    Analyse every move in a PGN and return a list of serialisable Explanation dicts.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN")

    with StockfishEngine(path=sf_binary, depth=depth, multipv=multipv) as engine:
        # Locate and load NNUE weights
        binary = engine.path
        nnue_file = Path(nnue_path) if nnue_path else _find_nnue(binary)
        if verbose:
            print(f"Loading NNUE weights from {nnue_file} ...")
        weights = load_nnue(nnue_file)
        if verbose:
            print(f"  Network description: {weights.description}")

        results: list[dict] = []
        board = game.board()
        prev_analysis = engine.analyse(board.fen())

        for node in game.mainline():
            move = node.move
            move_san = board.san(move)
            fen_before = board.fen()
            moving_color = board.turn
            move_number = board.fullmove_number

            if verbose:
                color_str = "White" if moving_color == chess.WHITE else "Black"
                print(f"  [{move_number}{'.' if moving_color==chess.WHITE else '...'}] "
                      f"{move_san} ({color_str})")

            # Stockfish best move + eval before
            sf_eval_before = prev_analysis.score_cp_white(board)
            best_pv = prev_analysis.best
            best_move_san = None
            best_line_san = []
            if best_pv and best_pv.moves:
                try:
                    best_move_san = board.san(chess.Move.from_uci(best_pv.moves[0]))
                except Exception:
                    best_move_san = best_pv.moves[0]
                best_line_san = _pv_to_san(best_pv.moves, board)

            # Compute features before move
            feat_before = compute_features(board)

            # Apply move
            board.push(move)
            fen_after = board.fen()

            # Stockfish eval after
            next_analysis = engine.analyse(fen_after)
            sf_eval_after = next_analysis.score_cp_white(board)

            # Compute features after move
            feat_after = compute_features(board)
            fdiff = diff_features(feat_before, feat_after)

            # NNUE probe
            board_before_obj = chess.Board(fen_before)
            board_after_obj  = chess.Board(fen_after)
            probe_result = probe(
                board_before=board_before_obj,
                board_after=board_after_obj,
                feat_before=feat_before,
                feat_after=feat_after,
                feat_diff=fdiff,
                weights=weights,
                moving_color=moving_color,
                sf_eval_before=sf_eval_before,
                sf_eval_after=sf_eval_after,
            )

            expl = build(
                result=probe_result,
                move_san=move_san,
                move_uci=move.uci(),
                fen_before=fen_before,
                fen_after=fen_after,
                move_number=move_number,
                moving_color=moving_color,
                best_move_san=best_move_san,
                best_line_san=best_line_san,
                sf_eval_before=sf_eval_before,
                sf_eval_after=sf_eval_after,
            )

            d = dataclasses.asdict(expl)
            # Add SF evals for reference
            d["sf_eval_before"] = sf_eval_before
            d["sf_eval_after"]  = sf_eval_after

            results.append(d)

            # Only mistakes?
            if only_mistakes and expl.quality not in ("inaccuracy", "mistake", "blunder"):
                results.pop()

            prev_analysis = next_analysis

    return results


def analyse_fen_move(
    fen: str,
    move_uci: str,
    sf_binary: Optional[str] = None,
    nnue_path: Optional[str] = None,
    depth: int = 18,
    multipv: int = 3,
    verbose: bool = False,
) -> dict:
    board = chess.Board(fen)
    move  = chess.Move.from_uci(move_uci)
    pgn   = chess.pgn.Game()
    pgn.setup(board)
    pgn.add_main_variation(move)
    results = analyse_game(
        pgn_text=str(pgn),
        sf_binary=sf_binary,
        nnue_path=nnue_path,
        depth=depth,
        multipv=multipv,
        verbose=verbose,
    )
    return results[0] if results else {}


def to_json(results: list[dict], indent: int = 2) -> str:
    return json.dumps(results, indent=indent)
