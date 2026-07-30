"""
일 대표값 -> KTCI 5개 영역 스트레스 하위지수 S_* (0=좋음, 1=나쁨)

스케일링 사양 그대로 구현.
  - 모든 지표는 선행연구 임계값을 앵커로 하는 구간선형(piecewise-linear) 함수
  - 앵커 (xi, si) 사이 선형보간, 결과는 [0,1] 클리핑
  - 여러 변수 결합은 **최댓값 결합** (한 요인이라도 위험하면 그날 스트레스가 높다)

    S_기온   = max( heat(AT_주간), cold(AT_야간) )
    S_습도   = max( rh_dev(RH), thi(THI) )
    S_대기질 = max( s_PM2.5, s_PMc, s_O3, s_NO2, s_SO2, s_CO )
    S_바람   = f(WS_최대)                      (옵션: 0.7·f(최대) + 0.3·f(평균))
    S_강수   = max( amount(RN_DAY), intensity(RN_HR1_최대) )

앵커표에 명시되지 않아 이 파일에서 정한 값은 ASSUMPTION 주석으로 전부 표시했다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Number = float | int | None


# ---------------------------------------------------------------- 구간선형
def piecewise(x: Number, anchors: list[tuple[float, float]]) -> float | None:
    """앵커점 사이 선형보간 후 [0,1] 클리핑. anchors는 x 오름차순."""
    if x is None:
        return None
    x = float(x)
    if x <= anchors[0][0]:
        return _clip01(anchors[0][1])
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return _clip01(y0 + (y1 - y0) * t)
    return _clip01(anchors[-1][1])


def _clip01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _max_or_none(*vals: float | None) -> float | None:
    got = [v for v in vals if v is not None]
    return max(got) if got else None


# ---------------------------------------------------------------- 체감온도
def wet_bulb_stull(ta: float, rh: float) -> float:
    """Stull(2011) 근사 습구온도. 기상청 여름 체감온도식의 입력."""
    return (ta * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + math.atan(ta + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)


def apparent_temp_heat(ta_max: Number, rh_mean: Number) -> float | None:
    """기상청 여름 체감온도(열지수). 일최고기온 + 일평균습도."""
    if ta_max is None:
        return None
    if rh_mean is None:
        return float(ta_max)  # 습도 결측 시 기온 그대로 (임의 가산하지 않음)
    ta, rh = float(ta_max), float(rh_mean)
    tw = wet_bulb_stull(ta, rh)
    return (-0.2442 + 0.55399 * tw + 0.45535 * ta
            - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0)


def apparent_temp_cold(ta_min: Number, ws_mean: Number) -> float | None:
    """기상청 겨울 체감온도(풍속냉각). 일최저기온 + 풍속. 적용범위 T<=10C, V>=4.8km/h."""
    if ta_min is None:
        return None
    ta = float(ta_min)
    if ws_mean is None:
        return ta
    v_kmh = float(ws_mean) * 3.6
    if ta > 10.0 or v_kmh < 4.8:
        return ta  # 공식 적용범위 밖 -> 기온 그대로
    p = v_kmh ** 0.16
    return 13.12 + 0.6215 * ta - 11.37 * p + 0.3965 * p * ta


# ---------------------------------------------------------------- 앵커표
@dataclass(frozen=True)
class Anchors:
    # --- 기온: UTCI 열/냉 스트레스 구간 경계
    # 무스트레스 ~26 / 중등도 26-32 / 강함 32-38 / 매우강함 38-46 / 극한 >46
    # ASSUMPTION: UTCI 극한 등급은 상한이 열려 있어 1.00 지점을 54도로 뒀다
    #             (직전 구간 폭 8도를 그대로 연장). 이 경우 폭염주의보 33도는 0.29,
    #             경보 35도는 0.38이 된다. 관광 맥락에서 더 가파르게 보려면 여기만 조정.
    heat: tuple = ((26.0, 0.00), (32.0, 0.25), (38.0, 0.50), (46.0, 0.75), (54.0, 1.00))
    # 냉스트레스: 약함 9~0 / 강함 0~-13 / 매우강함 -13~-27 / 극한 <-40
    cold: tuple = ((9.0, 0.00), (0.0, 0.25), (-13.0, 0.50), (-27.0, 0.75), (-40.0, 1.00))

    # --- 습도: ASHRAE 실내쾌적 40-60% 이탈
    # ASSUMPTION: 쾌적대 밖은 선형, 0%/100%에서 1.00
    rh_dry: tuple = ((0.0, 1.00), (40.0, 0.00))
    rh_wet: tuple = ((60.0, 0.00), (100.0, 1.00))
    # THI(불쾌지수): 낮음 <68 / 보통 68-75 / 높음 75-80(50% 불쾌) / 매우높음 >=80
    # ASSUMPTION: 4등급 등간격(0/0.33/0.67/1.00), 1.00 지점은 THI 86
    thi: tuple = ((68.0, 0.00), (75.0, 0.33), (80.0, 0.67), (86.0, 1.00))

    # --- 대기질: 한국 CAI 등급경계 -> 0/0.25/0.50/0.75, 매우나쁨 상한 1.00
    pm25: tuple = ((0, 0.00), (15, 0.25), (35, 0.50), (75, 0.75), (150, 1.00))
    pmc: tuple = ((0, 0.00), (15, 0.25), (45, 0.50), (75, 0.75), (150, 1.00))
    o3: tuple = ((0, 0.00), (0.030, 0.25), (0.090, 0.50), (0.150, 0.75), (0.300, 1.00))
    no2: tuple = ((0, 0.00), (0.030, 0.25), (0.060, 0.50), (0.200, 0.75), (0.400, 1.00))
    so2: tuple = ((0, 0.00), (0.020, 0.25), (0.050, 0.50), (0.150, 0.75), (0.300, 1.00))
    co: tuple = ((0, 0.00), (2, 0.25), (9, 0.50), (15, 0.75), (30, 1.00))

    # --- 바람: 보퍼트 + 기상청 강풍특보(주의보 14 / 경보 21 m/s)
    wind: tuple = ((3.3, 0.00), (8.0, 0.25), (13.9, 0.50), (17.2, 0.70),
                   (21.0, 0.85), (24.5, 1.00))

    # --- 강수
    precip_amount: tuple = ((0, 0.00), (10, 0.20), (30, 0.40), (50, 0.60),
                            (80, 0.80), (150, 1.00))
    precip_intensity: tuple = ((0, 0.00), (3, 0.25), (15, 0.50), (30, 0.75), (50, 1.00))


@dataclass
class SubIndexConfig:
    anchors: Anchors = field(default_factory=Anchors)
    wind_blend_mean: bool = False   # True면 0.7*f(최대) + 0.3*f(평균) 옵션
    wind_blend_ratio: float = 0.7


DEFAULT_CFG = SubIndexConfig()


def _cold_stress(at: Number, a: Anchors) -> float | None:
    """cold 앵커는 기온이 낮아질수록 스트레스가 커진다 (표가 x 내림차순)."""
    if at is None:
        return None
    pts = sorted((-x, y) for x, y in a.cold)  # 부호 반전해 오름차순화
    return piecewise(-float(at), pts)


# ---------------------------------------------------------------- 일 대표값
@dataclass
class DailyObs:
    """
    하루 x 한 지점의 대표값. 유효시간 조건(>=18/24)을 통과하지 못한 항목은 None.
    기온 C / 습도 % / 풍속 m/s / 강수 mm / PM ug/m3 / 가스 ppm
    """
    ta_mean: Number = None
    ta_max: Number = None
    ta_min: Number = None
    hm_mean: Number = None
    hm_min: Number = None
    ws_mean: Number = None
    ws_max: Number = None
    rn_day: Number = None       # 일강수 합계
    rn_hr1_max: Number = None   # 일 최대 1시간 강수
    pm25: Number = None
    pm10: Number = None
    o3_8h_max: Number = None    # O3는 일 최대 8시간 이동평균
    no2: Number = None
    so2: Number = None
    co: Number = None


# ---------------------------------------------------------------- 하위지수
def s_thermal(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> float | None:
    """S_기온 = max( heat(AT_주간), cold(AT_야간) )"""
    a = cfg.anchors
    return _max_or_none(
        piecewise(apparent_temp_heat(d.ta_max, d.hm_mean), list(a.heat)),
        _cold_stress(apparent_temp_cold(d.ta_min, d.ws_mean), a),
    )


def _rh_dev(d: DailyObs, a: Anchors) -> float | None:
    devs = []
    for rh in (d.hm_mean, d.hm_min):
        if rh is None:
            continue
        rh = float(rh)
        if rh < 40.0:
            devs.append(piecewise(rh, list(a.rh_dry)))
        elif rh > 60.0:
            devs.append(piecewise(rh, list(a.rh_wet)))
        else:
            devs.append(0.0)
    return max(devs) if devs else None


def _thi(d: DailyObs) -> float | None:
    """THI = 1.8T - 0.55(1 - RH/100)(1.8T - 26) + 32"""
    if d.ta_mean is None or d.hm_mean is None:
        return None
    t, rh = float(d.ta_mean), float(d.hm_mean)
    return 1.8 * t - 0.55 * (1 - rh / 100.0) * (1.8 * t - 26) + 32


def s_humidity(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> float | None:
    """S_습도 = max( rh_dev(RH), thi(THI) )"""
    a = cfg.anchors
    return _max_or_none(_rh_dev(d, a), piecewise(_thi(d), list(a.thi)))


def _pmc(d: DailyObs) -> float | None:
    """조대분획 PMc = max(PM10 - PM2.5, 0). PM2.5 결측 시 PM10을 보수적으로 사용."""
    if d.pm10 is None:
        return None
    if d.pm25 is None:
        return float(d.pm10)
    return max(float(d.pm10) - float(d.pm25), 0.0)


def s_airquality(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> float | None:
    """S_대기질 = max( 6개 물질 ). PM10-PM2.5 이중계산을 PMc 분리로 회피."""
    a = cfg.anchors
    return _max_or_none(
        piecewise(d.pm25, list(a.pm25)),
        piecewise(_pmc(d), list(a.pmc)),
        piecewise(d.o3_8h_max, list(a.o3)),
        piecewise(d.no2, list(a.no2)),
        piecewise(d.so2, list(a.so2)),
        piecewise(d.co, list(a.co)),
    )


def s_wind(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> float | None:
    """S_바람 = f(WS_최대). 옵션으로 평균 혼합."""
    a = cfg.anchors
    f_max = piecewise(d.ws_max, list(a.wind))
    f_mean = piecewise(d.ws_mean, list(a.wind))
    if not cfg.wind_blend_mean:
        return f_max if f_max is not None else f_mean
    if f_max is None or f_mean is None:
        return _max_or_none(f_max, f_mean)
    r = cfg.wind_blend_ratio
    return _clip01(r * f_max + (1 - r) * f_mean)


def s_precip(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> float | None:
    """
    S_강수 = max( amount(RN_DAY), intensity(RN_HR1_최대) )
    ! 강수는 결측을 0으로 보지 않는다. 값이 없으면 None(결측)이다.
    """
    a = cfg.anchors
    return _max_or_none(
        piecewise(d.rn_day, list(a.precip_amount)),
        piecewise(d.rn_hr1_max, list(a.precip_intensity)),
    )


def compute_subindices(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> dict[str, float | None]:
    """KTCI 5개 영역명 -> 0~1 스트레스 (산출 불가 시 None)."""
    return {
        "thermal": s_thermal(d, cfg),
        "humidity": s_humidity(d, cfg),
        "wind": s_wind(d, cfg),
        "precipitation": s_precip(d, cfg),
        "air_quality": s_airquality(d, cfg),
    }


def explain(d: DailyObs, cfg: SubIndexConfig = DEFAULT_CFG) -> dict[str, dict]:
    """각 영역에서 어떤 구성요소가 최댓값을 만들었는지까지 돌려준다 (설명용)."""
    a = cfg.anchors
    at_day = apparent_temp_heat(d.ta_max, d.hm_mean)
    at_night = apparent_temp_cold(d.ta_min, d.ws_mean)
    thi = _thi(d)
    parts = {
        "thermal": {"heat(AT_주간)": piecewise(at_day, list(a.heat)),
                    "cold(AT_야간)": _cold_stress(at_night, a)},
        "humidity": {"rh_dev": _rh_dev(d, a), "thi": piecewise(thi, list(a.thi))},
        "wind": {"f(WS_max)": piecewise(d.ws_max, list(a.wind)),
                 "f(WS_mean)": piecewise(d.ws_mean, list(a.wind))},
        "precipitation": {"amount(RN_DAY)": piecewise(d.rn_day, list(a.precip_amount)),
                          "intensity(RN_HR1)": piecewise(d.rn_hr1_max, list(a.precip_intensity))},
        "air_quality": {"PM2.5": piecewise(d.pm25, list(a.pm25)),
                        "PMc": piecewise(_pmc(d), list(a.pmc)),
                        "O3": piecewise(d.o3_8h_max, list(a.o3)),
                        "NO2": piecewise(d.no2, list(a.no2)),
                        "SO2": piecewise(d.so2, list(a.so2)),
                        "CO": piecewise(d.co, list(a.co))},
    }
    final = compute_subindices(d, cfg)
    out = {}
    for comp, sub in parts.items():
        got = {k: v for k, v in sub.items() if v is not None}
        out[comp] = {"S": final[comp], "driver": max(got, key=got.get) if got else None,
                     "parts": sub}
    out["thermal"]["AT_주간"] = at_day
    out["thermal"]["AT_야간"] = at_night
    out["humidity"]["THI"] = thi
    return out
