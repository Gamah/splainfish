"""classifier.py — Move quality enum (shared by explain.py and pipeline.py)."""
from enum import Enum

class MoveQuality(Enum):
    BEST       = "best"
    EXCELLENT  = "excellent"
    GOOD       = "good"
    INACCURACY = "inaccuracy"
    MISTAKE    = "mistake"
    BLUNDER    = "blunder"
    FORCED     = "forced"
