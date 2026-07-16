"""
explain.py — Convert ProbeResult attribution groups into plain English.

Two explanation modes:
  simple  — 2-3 sentences focused on the top 1-2 groups, novice-friendly
  complex — full ranked breakdown with contribution magnitudes per group

The output is a dict serialisable to JSON so the HTML viewer can render both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import chess

from .probe import ProbeResult, GroupedAttribution
from .classifier import MoveQuality


# ---------------------------------------------------------------------------
# Classification thresholds (centipawn loss from moving side)
# ---------------------------------------------------------------------------

def _classify_delta(delta_cp: int, moving_color: chess.Color) -> MoveQuality:
    """Classify the move quality from the eval delta (White's perspective)."""
    # Loss from mover's perspective
    loss = -delta_cp if moving_color == chess.WHITE else delta_cp
    if loss <= 10:
        return MoveQuality.BEST
    elif loss <= 30:
        return MoveQuality.EXCELLENT
    elif loss <= 60:
        return MoveQuality.GOOD
    elif loss <= 120:
        return MoveQuality.INACCURACY
    elif loss <= 250:
        return MoveQuality.MISTAKE
    else:
        return MoveQuality.BLUNDER


QUALITY_GLYPH = {
    MoveQuality.BEST:       "",
    MoveQuality.EXCELLENT:  "!",
    MoveQuality.GOOD:       "",
    MoveQuality.INACCURACY: "?!",
    MoveQuality.MISTAKE:    "?",
    MoveQuality.BLUNDER:    "??",
    MoveQuality.FORCED:     "",
}

QUALITY_LABEL = {
    MoveQuality.BEST:       "Best move",
    MoveQuality.EXCELLENT:  "Excellent",
    MoveQuality.GOOD:       "Good",
    MoveQuality.INACCURACY: "Inaccuracy",
    MoveQuality.MISTAKE:    "Mistake",
    MoveQuality.BLUNDER:    "Blunder",
    MoveQuality.FORCED:     "Forced",
}

QUALITY_COLOR = {
    MoveQuality.BEST:       "#5aa65a",
    MoveQuality.EXCELLENT:  "#96bc4b",
    MoveQuality.GOOD:       "#b0b0b0",
    MoveQuality.INACCURACY: "#f0c040",
    MoveQuality.MISTAKE:    "#e07030",
    MoveQuality.BLUNDER:    "#c03030",
    MoveQuality.FORCED:     "#808080",
}


# ---------------------------------------------------------------------------
# English templates per group
# ---------------------------------------------------------------------------

def _cp_str(cp: float) -> str:
    return f"{abs(cp) / 100:.1f}"


# Human-readable templates for each semantic group, keyed by direction.
# {mover} = side that moved, {enemy} = opponent
_GROUP_TEMPLATES = {
    "own pawn shield": {
        "negative": "This move weakened the {mover} king's pawn cover, leaving it more exposed to attacks.",
        "positive": "This move improved the {mover} king's pawn shield, making it harder to attack.",
    },
    "enemy pawn advance near king": {
        "negative": "Enemy pawns moved closer to the {mover} king, increasing the threat of a breakthrough.",
        "positive": "The {enemy} pawn pressure near the king was reduced.",
    },
    "enemy queen threatening king area": {
        "negative": "The {enemy} queen now has more influence near the {mover} king, creating serious attacking possibilities.",
        "positive": "The {enemy} queen's access to the {mover} king's area was reduced.",
    },
    "enemy rook pressure near king": {
        "negative": "A {enemy} rook gained access to lines near the {mover} king, increasing the danger.",
        "positive": "The {enemy} rook pressure near the {mover} king was relieved.",
    },
    "enemy knight attacking king zone": {
        "negative": "A {enemy} knight moved into the {mover} king's zone, directly threatening key squares around it.",
        "positive": "A {enemy} knight was removed from the {mover} king's zone.",
    },
    "enemy bishop attacking king zone": {
        "negative": "A {enemy} bishop now targets the {mover} king's zone from a distance.",
        "positive": "A {enemy} bishop's diagonal toward the {mover} king was closed.",
    },
    "enemy queen activity": {
        "negative": "The {enemy} queen gained a more active position, increasing overall pressure.",
        "positive": "The {enemy} queen's activity was restricted.",
    },
    "enemy rook activity": {
        "negative": "A {enemy} rook became more active, potentially targeting open lines.",
        "positive": "The {enemy} rook's activity was reduced.",
    },
    "enemy knight activity": {
        "negative": "A {enemy} knight reached a more active square, controlling more of the board.",
        "positive": "A {enemy} knight's influence was reduced.",
    },
    "enemy bishop activity": {
        "negative": "A {enemy} bishop opened up a more powerful diagonal.",
        "positive": "A {enemy} bishop's diagonal was blocked or restricted.",
    },
    "own queen activity / coordination": {
        "negative": "The {mover} queen became less active or lost coordination with other pieces.",
        "positive": "The {mover} queen moved to a more active square, increasing its influence.",
    },
    "own rook activity": {
        "negative": "A {mover} rook lost activity or access to open lines.",
        "positive": "A {mover} rook gained access to an open or semi-open file.",
    },
    "own knight activity / coordination": {
        "negative": "A {mover} knight moved to a less effective square.",
        "positive": "A {mover} knight reached a strong, centralized square.",
    },
    "own bishop activity / coordination": {
        "negative": "A {mover} bishop's diagonal was blocked or became less effective.",
        "positive": "A {mover} bishop was activated on a more powerful diagonal.",
    },
    "own pawn structure": {
        "negative": "This move created a weakness in the {mover} pawn structure.",
        "positive": "This move improved the {mover} pawn structure.",
    },
    "enemy pawn structure": {
        "negative": "The {enemy} pawn structure became more solid or threatening.",
        "positive": "The {enemy} pawn structure was weakened.",
    },
    "king position": {
        "negative": "The king moved to a less safe square.",
        "positive": "The king moved to a safer or more active square.",
    },
    "own queen near king (defensive)": {
        "negative": "The {mover} queen moved away from a defensive role near the king.",
        "positive": "The {mover} queen took up a defensive position near the king.",
    },
}

_FALLBACK_TEMPLATE = {
    "negative": "The {mover} position became less favourable in terms of {group}.",
    "positive": "The {mover} position improved in terms of {group}.",
}


def _sentence(group: str, direction: str, mover: str, enemy: str) -> str:
    templates = _GROUP_TEMPLATES.get(group, _FALLBACK_TEMPLATE)
    tmpl = templates.get(direction, _FALLBACK_TEMPLATE[direction])
    return tmpl.format(mover=mover, enemy=enemy, group=group)


# ---------------------------------------------------------------------------
# Complexity detector
# ---------------------------------------------------------------------------

def _is_complex(groups: list[GroupedAttribution], delta_cp: int) -> bool:
    """
    A position is 'complex' (no single dominant explanation) if:
    - The top group accounts for < 40% of the total absolute attribution
    - OR there are 5+ groups each contributing > 10% of the total
    """
    if not groups:
        return False
    total = sum(abs(g.contribution) for g in groups)
    if total < 1e-6:
        return False
    top_share = abs(groups[0].contribution) / total
    significant = sum(1 for g in groups if abs(g.contribution) / total > 0.10)
    return top_share < 0.40 or significant >= 5


# ---------------------------------------------------------------------------
# Main explanation builder
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    move_san:       str
    move_uci:       str
    fen_before:     str
    fen_after:      str
    move_number:    int
    color:          str          # "white" | "black"
    eval_before_cp: int          # White's perspective
    eval_after_cp:  int
    delta_cp:       int

    quality:        str          # MoveQuality.value
    quality_label:  str
    quality_glyph:  str
    quality_color:  str

    is_complex:     bool

    # Simple explanation
    simple_headline:   str
    simple_paragraphs: list[str]

    # Complex explanation
    complex_headline:  str
    complex_groups:    list[dict]   # [{group, contribution_cp, direction, sentence, feature_count}]
    complex_note:      str

    # Internal for debugging
    top_groups:     list[str]

    # Best move from SF
    best_move_san:  Optional[str] = None
    best_line_san:  list[str] = None

    # Neuron-level data for optional display
    fc0_delta_top:  list[dict] = None   # top changed neurons at fc0
    fc1_delta_top:  list[dict] = None


def build(
    result: ProbeResult,
    move_san: str,
    move_uci: str,
    fen_before: str,
    fen_after: str,
    move_number: int,
    moving_color: chess.Color,
    best_move_san: Optional[str] = None,
    best_line_san: Optional[list[str]] = None,
    sf_eval_before: Optional[int] = None,
    sf_eval_after: Optional[int] = None,
) -> Explanation:
    mover = "White" if moving_color == chess.WHITE else "Black"
    enemy = "Black" if moving_color == chess.WHITE else "White"

    # Always use Stockfish actual evals for classification and display.
    # Our internal forward pass cp is only used for attribution direction.
    eval_before = sf_eval_before if sf_eval_before is not None else result.eval_before_cp
    eval_after  = sf_eval_after  if sf_eval_after  is not None else result.eval_after_cp
    sf_delta = eval_after - eval_before   # White perspective

    quality = _classify_delta(sf_delta, moving_color)
    loss_cp = max(0, (-sf_delta if moving_color == chess.WHITE else sf_delta))

    groups = result.grouped_attributions
    is_complex = _is_complex(groups, sf_delta)

    # ------------------------------------------------------------------
    # Simple explanation
    # ------------------------------------------------------------------

    # Headline
    glyph = QUALITY_GLYPH[quality]
    display_move = f"{move_san}{glyph}" if glyph else move_san

    if quality == MoveQuality.BEST:
        simple_headline = f"Best move: {display_move}"
    elif quality in (MoveQuality.EXCELLENT, MoveQuality.GOOD):
        simple_headline = f"{QUALITY_LABEL[quality]}: {display_move}"
    elif quality == MoveQuality.INACCURACY:
        simple_headline = f"Inaccuracy: {display_move} — about {_cp_str(loss_cp)} pawns below best"
    elif quality == MoveQuality.MISTAKE:
        simple_headline = f"Mistake: {display_move} — loses about {_cp_str(loss_cp)} pawns"
    else:
        simple_headline = f"Blunder: {display_move} — loses about {_cp_str(loss_cp)} pawns"

    simple_paragraphs: list[str] = []

    if quality in (MoveQuality.BEST, MoveQuality.EXCELLENT, MoveQuality.GOOD):
        simple_paragraphs.append(
            f"The evaluation barely changed ({result.eval_before_cp/100:+.2f} → "
            f"{result.eval_after_cp/100:+.2f}), meaning this move maintained or "
            f"improved {mover}'s position."
        )
        if groups:
            top = groups[0]
            simple_paragraphs.append(_sentence(top.group, top.direction, mover, enemy))
    elif is_complex:
        simple_paragraphs.append(
            f"The evaluation shifted about {_cp_str(loss_cp)} pawns against {mover}. "
            f"This position is quite complex — several factors contributed to the evaluation change "
            f"rather than one clear reason. The main themes were: "
            f"{', '.join(g.group for g in groups[:3])}."
        )
        simple_paragraphs.append(
            "Switch to the detailed view to see a full breakdown of each contributing factor."
        )
    else:
        # Top 1-2 groups dominate — explain those
        top_groups = [g for g in groups[:2] if abs(g.contribution) > 0.01]
        if top_groups:
            primary = top_groups[0]
            s1 = _sentence(primary.group, primary.direction, mover, enemy)
            simple_paragraphs.append(
                f"The evaluation moved {_cp_str(abs(sf_delta))} pawns "
                f"{'against' if quality in (MoveQuality.MISTAKE, MoveQuality.BLUNDER) else 'for'} "
                f"{mover}. {s1}"
            )
            if len(top_groups) > 1:
                secondary = top_groups[1]
                s2 = _sentence(secondary.group, secondary.direction, mover, enemy)
                simple_paragraphs.append(f"Additionally: {s2.lower()}")

        if best_move_san and quality not in (MoveQuality.BEST, MoveQuality.FORCED):
            line = " ".join(best_line_san[:5]) if best_line_san else ""
            simple_paragraphs.append(
                f"Stockfish preferred {best_move_san} instead."
                + (f" Best line: {line}" if line else "")
            )

    # ------------------------------------------------------------------
    # Complex explanation
    # ------------------------------------------------------------------

    complex_headline = (
        f"{QUALITY_LABEL[quality]}: {display_move}  "
        f"[{eval_before/100:+.2f} → {eval_after/100:+.2f}]"
    )

    total_attr = sum(abs(g.contribution) for g in groups) or 1.0
    complex_groups_out = []
    for g in groups:
        pct = abs(g.contribution) / total_attr * 100
        complex_groups_out.append({
            "group":          g.group,
            "contribution_cp": round(g.contribution / 100, 2),
            "direction":      g.direction,
            "pct_of_total":   round(pct, 1),
            "feature_count":  g.feature_count,
            "sentence":       _sentence(g.group, g.direction, mover, enemy),
        })

    complex_note = (
        "Contributions are derived by back-projecting the eval delta through the NNUE "
        "network (L0→L2) to the input feature groups that changed between the two positions. "
        "The percentages show each group's share of the total attributed change."
    )

    if result.sf_eval_before is not None and result.sf_eval_after is not None:
        ours_before = result.eval_before_cp
        ours_after  = result.eval_after_cp
        err_before  = abs(ours_before - result.sf_eval_before)
        err_after   = abs(ours_after  - result.sf_eval_after)
        if err_before > 50 or err_after > 50:
            complex_note += (
                f" Note: our internal eval ({ours_before/100:+.2f} / {ours_after/100:+.2f}) "
                f"differs from Stockfish's ({result.sf_eval_before/100:+.2f} / "
                f"{result.sf_eval_after/100:+.2f}) — attribution proportions are "
                f"still meaningful but absolute magnitudes may vary."
            )

    # Top changed neurons for optional debug display
    import numpy as np
    fc0_top = sorted(
        [{"neuron": int(i), "delta": float(result.fc0_delta[i])}
         for i in range(len(result.fc0_delta))],
        key=lambda x: abs(x["delta"]), reverse=True
    )[:10]
    fc1_top = sorted(
        [{"neuron": int(i), "delta": float(result.fc1_delta[i])}
         for i in range(len(result.fc1_delta))],
        key=lambda x: abs(x["delta"]), reverse=True
    )[:10]

    return Explanation(
        move_san=move_san,
        move_uci=move_uci,
        fen_before=fen_before,
        fen_after=fen_after,
        move_number=move_number,
        color="white" if moving_color == chess.WHITE else "black",
        eval_before_cp=eval_before,
        eval_after_cp=eval_after,
        delta_cp=sf_delta,
        quality=quality.value,
        quality_label=QUALITY_LABEL[quality],
        quality_glyph=glyph,
        quality_color=QUALITY_COLOR[quality],
        is_complex=is_complex,
        simple_headline=simple_headline,
        simple_paragraphs=simple_paragraphs,
        complex_headline=complex_headline,
        complex_groups=complex_groups_out,
        complex_note=complex_note,
        top_groups=[g.group for g in groups[:5]],
        best_move_san=best_move_san,
        best_line_san=best_line_san or [],
        fc0_delta_top=fc0_top,
        fc1_delta_top=fc1_top,
    )
