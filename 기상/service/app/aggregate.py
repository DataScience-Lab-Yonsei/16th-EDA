"""
시간별 관측/예보 -> 일 대표값(DailyObs)

유효조건: 각 변수의 일 유효시간 >= 18/24 (75%). 미달이면 그 항목은 None으로 두고
scoring 단계의 weighted_available 규칙이 처리한다.

주의사항 두 가지:
  - 강수는 결측을 0으로 채우지 않는다. 유효시간 미달이면 rn_day/rn_hr1_max 모두 None.
  - O3는 일평균이 아니라 '일 최대 8시간 이동평균'이다.
"""

from __future__ import annotations

from datetime import date, datetime

from .subindex import DailyObs

MIN_VALID_HOURS = 18  # 24시간 중 75%

# 가스 분자량 (ug/m3 -> ppm 변환용, 25C 1atm 기준 몰부피 24.45 L/mol)
MOLAR_MASS = {"o3": 48.00, "no2": 46.0055, "so2": 64.066, "co": 28.010}


def ugm3_to_ppm(value: float | None, species: str) -> float | None:
    """Open-Meteo 등은 가스를 ug/m3로 준다. 앵커표는 ppm 기준이므로 변환한다."""
    if value is None:
        return None
    return float(value) * 24.45 / (MOLAR_MASS[species] * 1000.0)


def _valid(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def _agg(values: list[float | None], fn, min_hours: int = MIN_VALID_HOURS):
    got = _valid(values)
    return fn(got) if len(got) >= min_hours else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def o3_max_8h(o3_hourly: list[float | None], min_hours: int = MIN_VALID_HOURS) -> float | None:
    """일 최대 8시간 이동평균. 각 창은 8시간 중 6시간(75%) 이상 유효해야 인정."""
    if len(_valid(o3_hourly)) < min_hours:
        return None
    best = None
    for i in range(0, max(len(o3_hourly) - 7, 0)):
        window = _valid(o3_hourly[i:i + 8])
        if len(window) >= 6:
            m = _mean(window)
            best = m if best is None else max(best, m)
    return best


def build_daily(hourly: dict[datetime, dict], day: date,
                min_hours: int = MIN_VALID_HOURS) -> DailyObs:
    """
    hourly: {datetime(정시): {ta, hm, ws, rn, pm25, pm10, o3, no2, so2, co}}
            (단위 - ta C / hm % / ws m/s / rn mm(1시간) / PM ug/m3 / 가스 ppm)
    """
    rows = [v for t, v in sorted(hourly.items()) if t.date() == day]
    if not rows:
        return DailyObs()

    col = lambda k: [r.get(k) for r in rows]

    ta = col("ta")
    hm = col("hm")
    ws = col("ws")
    rn = col("rn")

    ta_ok = len(_valid(ta)) >= min_hours
    rn_ok = len(_valid(rn)) >= min_hours  # 강수: 결측을 0으로 채우지 않는다

    return DailyObs(
        ta_mean=_agg(ta, _mean, min_hours),
        ta_max=max(_valid(ta)) if ta_ok else None,
        ta_min=min(_valid(ta)) if ta_ok else None,
        hm_mean=_agg(hm, _mean, min_hours),
        hm_min=min(_valid(hm)) if len(_valid(hm)) >= min_hours else None,
        ws_mean=_agg(ws, _mean, min_hours),
        ws_max=max(_valid(ws)) if len(_valid(ws)) >= min_hours else None,
        rn_day=sum(_valid(rn)) if rn_ok else None,
        rn_hr1_max=max(_valid(rn)) if rn_ok else None,
        pm25=_agg(col("pm25"), _mean, min_hours),
        pm10=_agg(col("pm10"), _mean, min_hours),
        o3_8h_max=o3_max_8h(col("o3"), min_hours),
        no2=_agg(col("no2"), _mean, min_hours),
        so2=_agg(col("so2"), _mean, min_hours),
        co=_agg(col("co"), _mean, min_hours),
    )


def daily_coverage(hourly: dict[datetime, dict], day: date) -> dict[str, int]:
    """변수별 유효시간 수 (유효조건 미달 원인을 확인할 때 쓴다)."""
    rows = [v for t, v in sorted(hourly.items()) if t.date() == day]
    keys = ["ta", "hm", "ws", "rn", "pm25", "pm10", "o3", "no2", "so2", "co"]
    return {k: len([r for r in rows if r.get(k) is not None]) for k in keys}
