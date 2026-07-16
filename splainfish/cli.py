"""
cli.py — Command-line interface for splainfish.

Usage
-----
  # Analyse a PGN file → HTML report
  python -m splainfish.cli --pgn game.pgn --html report.html

  # Analyse a PGN string → HTML
  python -m splainfish.cli --pgn "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6?? 4. Ng5" --html out.html

  # FEN + single move → HTML
  python -m splainfish.cli --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \\
         --move e2e4 --html out.html

  # JSON output instead of HTML
  python -m splainfish.cli --pgn game.pgn --json

  # Only show inaccuracies, mistakes, blunders
  python -m splainfish.cli --pgn game.pgn --html out.html --only-mistakes

  # Custom Stockfish binary and NNUE file
  python -m splainfish.cli --pgn game.pgn --html out.html \\
         --stockfish ./stockfish --nnue ./nn-current.nnue

  # Adjust depth (default 18)
  python -m splainfish.cli --pgn game.pgn --html out.html --depth 14
"""

from __future__ import annotations

import argparse
import os
import sys
import json

from .pipeline import analyse_game, analyse_fen_move, to_json
from .render import render_html


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Explain chess moves using NNUE activation probing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--pgn", metavar="PGN_OR_FILE",
                     help="PGN string or path to a .pgn file")
    inp.add_argument("--fen", metavar="FEN",
                     help="FEN string (requires --move)")
    parser.add_argument("--move", metavar="UCI",
                        help="Move in UCI notation, required with --fen")

    # Output
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--html", metavar="FILE",
                     help="Write self-contained HTML report to FILE")
    out.add_argument("--json", action="store_true",
                     help="Print JSON to stdout")

    # Engine / NNUE
    parser.add_argument("--stockfish", metavar="PATH",
                        help="Path to Stockfish binary (default: auto-detect)")
    parser.add_argument("--nnue", metavar="PATH",
                        help="Path to .nnue file (default: auto-detect from binary)")
    parser.add_argument("--depth", type=int, default=18,
                        help="Stockfish search depth (default: 18)")
    parser.add_argument("--multipv", type=int, default=3,
                        help="MultiPV lines from Stockfish (default: 3)")

    # Filters
    parser.add_argument("--only-mistakes", action="store_true",
                        help="Only include inaccuracies, mistakes, and blunders")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress to stderr")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------------
    if args.fen:
        if not args.move:
            parser.error("--fen requires --move")
        results = [analyse_fen_move(
            fen=args.fen,
            move_uci=args.move,
            sf_binary=args.stockfish,
            nnue_path=args.nnue,
            depth=args.depth,
            multipv=args.multipv,
            verbose=args.verbose,
        )]
    else:
        pgn_text = args.pgn
        if os.path.isfile(pgn_text):
            with open(pgn_text, encoding="utf-8") as f:
                pgn_text = f.read()
        results = analyse_game(
            pgn_text=pgn_text,
            sf_binary=args.stockfish,
            nnue_path=args.nnue,
            depth=args.depth,
            multipv=args.multipv,
            verbose=args.verbose,
            only_mistakes=args.only_mistakes,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.json:
        print(to_json(results))
    elif args.html:
        html = render_html(results)
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report written to {args.html}  ({len(results)} moves analysed)")
    else:
        # Default: print simple text summary to stdout
        for m in results:
            clr = "White" if m["color"] == "white" else "Black"
            num = m["move_number"]
            dots = "." if m["color"] == "white" else "..."
            print(f"\n{'='*56}")
            print(f"{num}{dots} {m['move_san']}{m['quality_glyph']}  [{m['quality_label']}]")
            print(f"  Eval: {m['eval_before_cp']/100:+.2f} → {m['eval_after_cp']/100:+.2f} cp")
            print(f"  {m['simple_headline']}")
            for p in m.get("simple_paragraphs", []):
                print(f"  {p}")
            if m.get("complex_groups"):
                print(f"  Top factors:")
                for g in m["complex_groups"][:3]:
                    sign = "+" if g["contribution_cp"] >= 0 else ""
                    print(f"    • {g['group']}: {sign}{g['contribution_cp']:.2f}p ({g['pct_of_total']}%)")


if __name__ == "__main__":
    main()
