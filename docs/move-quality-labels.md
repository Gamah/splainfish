# Move quality labels

splainfish tags every move with a quality label. This page explains what each
label means, the exact thresholds behind it, and — importantly — where the
number comes from and what the *explanation* under it does and doesn't claim.

## Where the label comes from

The label is driven entirely by **Stockfish's own evaluation**, not by the NNUE
attribution that produces the written explanation.

For each move, splainfish asks Stockfish for the evaluation of the position
**before** and **after** the move (a fixed-depth search, converted to White's
point of view). The difference is the move's effect:

```
delta      = eval_after − eval_before          (White's frame, centipawns)
loss (cp)  = the part of delta that is bad for the side that moved
```

A "centipawn" is 1/100th of a pawn, so a loss of 120 cp ≈ 1.2 pawns. The label
is a pure function of that loss:

| Label | Glyph | Centipawn loss | Pawns lost |
|-------|:-----:|----------------|------------|
| **Best** | – | 0 – 10 | ≤ 0.10 |
| **Excellent** | ! | 11 – 30 | ≤ 0.30 |
| **Good** | – | 31 – 60 | ≤ 0.60 |
| **Inaccuracy** | ?! | 61 – 120 | ≤ 1.20 |
| **Mistake** | ? | 121 – 250 | ≤ 2.50 |
| **Blunder** | ?? | > 250 | > 2.50 |

`Forced` is a reserved label for positions with a single reasonable reply; the
current browser build classifies purely by centipawn loss and does not emit it.

The thresholds are the same numbers used by the Python reference implementation
(`splainfish/explain.py`) so the browser app and the CLI agree move-for-move.

### Notes on the numbers

- **Loss is one-sided.** Only changes *against* the moving side count. Improving
  your position past what the engine expected does not make a move "more than
  best" — it is still simply Best.
- **Depth matters.** All evals come from the same fixed search depth (the *Depth*
  control in the app). Deeper searches give steadier labels, especially in sharp
  tactical positions; shallow searches can misjudge a move that only pays off a
  few moves later.
- **Mate scores** are treated as very large evaluations, so walking into a forced
  mate — or missing one — reliably lands in Blunder territory.

## What the *explanation* is (and isn't)

The label answers *how much* a move changed the evaluation. The paragraph under
it answers *why*, and it comes from a different source: splainfish runs
Stockfish's NNUE forward pass on both positions, diffs the network activations
layer by layer, and back-projects the change onto human chess concepts (a pawn
shield, a knight reaching the king's zone, a rook finding an open file).

That attribution shows **relative direction and emphasis, not calibrated
centipawns**. When the app says a factor "contributed," it means that concept
moved the network's output in that direction — not that it was worth a specific
number of pawns. The centipawn figures shown in headlines always come from
Stockfish's evaluation, never from the NNUE forward pass.

No language model is involved anywhere; the explanations are derived
deterministically from the activations.

## Where the concepts are defined in the code

The concept names you see in explanations — "enemy pawn structure", "own pawn
shield", "enemy knight attacking king zone", and so on — are not free text. Each
one is produced by two small, reviewable pieces of code:

1. **Bucketing a feature into a concept** —
   [`groupLabel()` in `web/js/nnue/probe.js`](https://github.com/Gamah/splainfish/blob/master/web/js/nnue/probe.js#L346).
   Every NNUE input feature that changed is a (piece colour, piece type,
   king-relative square) triple. `groupLabel()` maps that triple to a concept:
   e.g. an opponent pawn that is *not* near your king becomes
   `enemy pawn structure`, an opponent knight *near* your king becomes
   `enemy knight attacking king zone`. The "near the king" test is a single
   king-bucket threshold (`att.kingBucket <= 7`), also visible in that function.

2. **Turning a concept into a sentence** —
   [`GROUP_TEMPLATES` in `web/js/explain.js`](https://github.com/Gamah/splainfish/blob/master/web/js/explain.js#L53).
   Each concept has a `positive` and `negative` phrasing; the sign of the
   attribution picks which one is shown, with `{mover}`/`{enemy}` filled in.

So to audit any explanation, read those two functions: `groupLabel()` decides
*which* concept a change belongs to, and `GROUP_TEMPLATES` decides *how it is
worded*. The Python reference implementation mirrors both
(`splainfish/probe.py` and `splainfish/explain.py`) so the CLI and browser agree.

> Line anchors above can drift as the code changes; if one points at the wrong
> line, search the linked file for `groupLabel` or `GROUP_TEMPLATES`.
