"""
API 응답 조합.
"""

from __future__ import annotations

from .scoring import score_itinerary


def _r(v, n=4):
    """응답 직전에만 반올림한다 (내부 계산은 항상 full precision)."""
    if isinstance(v, dict):
        return {k: _r(x, n) for k, x in v.items()}
    return round(v, n) if isinstance(v, float) else v


def build_response(stop_scores: list[dict], variant: str, valid: dict) -> dict:
    return {
        "variant": variant,
        "stops": stop_scores,
        "validity": valid,
        "summary": {
            "trip_score": score_itinerary(stop_scores, variant),
            "stops": len(stop_scores),
        },
    }
