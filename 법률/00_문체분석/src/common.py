from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
EXAMPLES = RESULTS / "examples"
LOGS = ROOT / "logs"


def ensure_directories() -> None:
    for path in (
        DATA / "raw",
        DATA / "interim",
        DATA / "processed",
        TABLES,
        FIGURES,
        EXAMPLES,
        LOGS,
        ROOT / ".mplconfig",
    ):
        path.mkdir(parents=True, exist_ok=True)


def bh_adjust(pvalues: pd.Series | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def wilson_interval(successes: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib as mpl
    from matplotlib import font_manager

    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            prop = font_manager.FontProperties(fname=str(candidate))
            mpl.rcParams["font.family"] = prop.get_name()
            break
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
        }
    )


METRIC_LABELS = {
    "noun_share": "명사 비율",
    "verb_share": "동사 비율",
    "predicate_share": "서술어 비율",
    "particle_share": "조사 비율",
    "nominality": "명사성",
    "sent_eojeol_mean": "문장당 어절 수",
    "sent_char_mean": "문장당 글자 수",
    "hanja_share": "한자 비율",
}

