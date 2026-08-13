"""
KTCI 계산 코어 + 여행 일정 집계.

워크북에서 확정한 규칙만 구현한다 (하위지수 정의는 subindex.py, 일 대표값은 aggregate.py).
  - 계절별 가중치 3종: data / survey_2014_adapted / hybrid
  - hybrid = alpha_survey * w_2014_adapted + (1 - alpha_survey) * w_data
    alpha: 봄 0.45 / 여름 0.00 / 가을 0.25 / 겨울 0.30
  - 결측 처리: weighted_available (관측 가능한 영역만 남기고 가중치 재정규화)
  - 가용 영역이 2개 미만이면 KTCI 결측 (함안군 케이스와 동일)

KTCI는 일 단위 지수다. 따라서 점수의 단위도 (장소, 날짜)이며,
같은 날 같은 도시의 여러 스탑은 동일한 KTCI를 공유한다.
스탑 간 차이는 장소(격자)와 노출도/체류시간에서 나온다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from .subindex import DEFAULT_CFG, DailyObs, SubIndexConfig, compute_subindices, explain

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ktci_weights.json"
with CONFIG_PATH.open(encoding="utf-8") as f:
    KTCI_CONFIG = json.load(f)

COMPONENTS: list[str] = KTCI_CONFIG["components"]
MIN_COMPONENTS: int = KTCI_CONFIG["missing_rule"]["min_components"]

# JSON에는 소수 6자리로 반올림돼 있으므로 로드 시 합이 정확히 1이 되도록 재정규화
for _variant, _seasons in KTCI_CONFIG["weights"].items():
    for _season, _w in _seasons.items():
        _tot = sum(_w.values())
        if _tot > 0:
            _seasons[_season] = {c: v / _tot for c, v in _w.items()}

SEASON_OF_MONTH = {3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn",
                   12: "winter", 1: "winter", 2: "winter"}

# 노출도: 실외 활동일수록 그날 날씨를 그대로 체감한다
EXPOSURE_FACTOR = {"outdoor": 1.0, "mixed": 0.5, "transit": 0.35, "indoor": 0.15}


def season_of(month: int) -> str:
    return SEASON_OF_MONTH[month]


def get_weights(season: str, variant: str = "hybrid") -> dict[str, float]:
    try:
        return KTCI_CONFIG["weights"][variant][season]
    except KeyError as exc:
        raise ValueError(f"unknown season/variant: {season}/{variant}") from exc


# ---------------------------------------------------------------- KTCI 코어
@dataclass
class KTCIResult:
    ktci: float | None                      # 0~1 스트레스. None = 결측
    subindices: dict[str, float | None]
    weights_used: dict[str, float]          # 재정규화 후 실제 적용 가중치
    available: list[str]
    missing: list[str]
    coverage: float                         # 원 가중치 기준 관측 커버리지
    season: str = ""
    note: str = ""

    @property
    def score_100(self) -> float | None:
        """사람이 읽는 '날씨 기반 여행 점수' (0~100, 높을수록 좋음)."""
        return None if self.ktci is None else round(100.0 * (1.0 - self.ktci), 1)

    @property
    def top_component(self) -> tuple[str, float] | None:
        """감점 기여 1위 영역과 그 비중."""
        if self.ktci is None or self.ktci == 0:
            return None
        contrib = {c: self.weights_used[c] * self.subindices[c] for c in self.available}
        c = max(contrib, key=contrib.get)
        return c, contrib[c] / self.ktci


def ktci_from_subindices(sub: dict[str, float | None], season: str,
                         variant: str = "hybrid") -> KTCIResult:
    base = get_weights(season, variant)
    available = [c for c in COMPONENTS if sub.get(c) is not None]
    missing = [c for c in COMPONENTS if c not in available]
    coverage = sum(base[c] for c in available)

    if len(available) < MIN_COMPONENTS or coverage <= 0:
        return KTCIResult(None, sub, {}, available, missing, coverage, season,
                          f"가용 영역 {len(available)}개 < 최소 {MIN_COMPONENTS}개 -> 결측 처리")

    w = {c: base[c] / coverage for c in available}      # weighted_available 재정규화
    ktci = sum(w[c] * sub[c] for c in available)
    note = "" if not missing else f"결측 {','.join(missing)} -> 가중치 재정규화(커버리지 {coverage:.1%})"
    return KTCIResult(ktci, sub, w, available, missing, coverage, season, note)


def score_day(daily: DailyObs, day: date, variant: str = "hybrid",
              cfg: SubIndexConfig = DEFAULT_CFG) -> KTCIResult:
    """(장소, 날짜) 하나에 대한 KTCI."""
    season = season_of(day.month)
    return ktci_from_subindices(compute_subindices(daily, cfg), season, variant)


# ---------------------------------------------------------------- 일정 집계
@dataclass
class Stop:
    name: str
    lat: float
    lon: float
    start: datetime
    end: datetime
    exposure: str = "outdoor"

    @property
    def day(self) -> date:
        return self.start.date()

    @property
    def hours(self) -> float:
        return max((self.end - self.start).total_seconds() / 3600.0, 0.0)

    @property
    def grid_key(self) -> tuple:
        """예보 캐시 키. 소수 2자리(~1km)면 같은 격자로 본다."""
        return (round(self.lat, 2), round(self.lon, 2), self.day)


@dataclass
class StopScore:
    stop: Stop
    result: KTCIResult
    daily: DailyObs
    detail: dict = field(default_factory=dict)   # subindex.explain() 결과

    @property
    def exposure_factor(self) -> float:
        return EXPOSURE_FACTOR.get(self.stop.exposure, 1.0)

    @property
    def felt_ktci(self) -> float | None:
        """노출도까지 반영한 체감 스트레스."""
        return None if self.result.ktci is None else self.exposure_factor * self.result.ktci


def score_stop(stop: Stop, daily: DailyObs, variant: str = "hybrid",
               cfg: SubIndexConfig = DEFAULT_CFG) -> StopScore:
    res = score_day(daily, stop.day, variant, cfg)
    return StopScore(stop, res, daily, explain(daily, cfg))


def score_itinerary(stop_scores: list[StopScore]) -> dict:
    """
    체류시간 가중평균 + 노출도 감쇠 -> 여행 전체 점수.

    노출도는 '스탑 간 비중'이 아니라 '체감하는 날씨 스트레스의 크기'를 줄인다.
    비중으로만 쓰면 같은 날 같은 도시(= 같은 KTCI)에서는 아무 효과가 없기 때문이다.
        trip = sum(h * exposure * ktci) / sum(h)
    """
    num = den = 0.0
    scored, skipped = [], []
    for ss in stop_scores:
        if ss.result.ktci is None:
            skipped.append(ss)
            continue
        num += ss.stop.hours * ss.exposure_factor * ss.result.ktci
        den += ss.stop.hours
        scored.append(ss)

    trip = num / den if den > 0 else None
    by_day: dict[str, list[float]] = {}
    for ss in scored:
        by_day.setdefault(ss.stop.day.isoformat(), []).append(ss.felt_ktci)

    return {
        "trip_ktci": round(trip, 6) if trip is not None else None,
        "trip_score": round(100 * (1 - trip), 1) if trip is not None else None,
        "grade": grade_of(trip),
        "scored_stops": len(scored),
        "skipped_stops": [s.stop.name for s in skipped],
        "best_stop": min(scored, key=lambda s: s.felt_ktci).stop.name if scored else None,
        "worst_stop": max(scored, key=lambda s: s.felt_ktci).stop.name if scored else None,
        "daily_score": {d: round(100 * (1 - sum(v) / len(v)), 1) for d, v in by_day.items()},
    }


def grade_of(ktci: float | None) -> str | None:
    if ktci is None:
        return None
    score = 100 * (1 - ktci)
    for cut, label in ((85, "매우 좋음"), (70, "좋음"), (55, "보통"), (40, "나쁨")):
        if score >= cut:
            return label
    return "매우 나쁨"
