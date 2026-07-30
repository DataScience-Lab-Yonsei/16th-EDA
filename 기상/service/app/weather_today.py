"""
전국 264개 시군구의 실시간 날씨 점수를 요청마다 새로 계산한다 (정적 파일 없음).

refresh_weather_data.py(로컬 스크립트)와 같은 계산이지만, 대용량 원본 CSV
대신 미리 뽑아둔 계수/보간테이블(config/at_coefficients.json,
config/aq_stress_tables.json)을 쓴다 - 서버리스 함수에 89MB짜리 CSV를
넣을 수는 없어서다.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from .providers import now_kst

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
KMA_AUTH_KEY = "x6QFqFUGS7KkBahVBpuyOQ"
SEOUL_AQ_AUTH_KEY = "5179547953646c6438357172564e4b"

with (CONFIG_DIR / "at_coefficients.json").open() as f:
    _AT = json.load(f)
COEF_DAY = np.array(_AT["coef_day"])
COEF_NIGHT = np.array(_AT["coef_night"])

with (CONFIG_DIR / "aq_stress_tables.json").open() as f:
    _AQ = json.load(f)
PM25_X, PM25_Y = np.array(_AQ["pm25_x"]), np.array(_AQ["pm25_y"])
PMC_X, PMC_Y = np.array(_AQ["pmc_x"]), np.array(_AQ["pmc_y"])

with (CONFIG_DIR / "signgu_centroids.json").open(encoding="utf-8") as f:
    REGIONS = json.load(f)

with (CONFIG_DIR / "seasonal_weights.json").open() as f:
    SEASON_WEIGHTS = json.load(f)

SEASONS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11], "winter": [12, 1, 2]}
MONTH_TO_SEASON = {m: s for s, ms in SEASONS.items() for m in ms}

HEAT_ANCHORS = [(26, 0.0), (32, 0.25), (38, 0.50), (46, 0.75), (54, 1.0)]
COLD_ANCHORS_NEG = [(-9, 0.0), (0, 0.25), (13, 0.50), (27, 0.75), (40, 1.0)]
WIND_ANCHORS = [(3.3, 0.0), (8.0, 0.25), (13.9, 0.50), (17.2, 0.70), (21, 0.85), (24.5, 1.0)]
AMT_ANCHORS = [(0, 0.0), (10, 0.20), (30, 0.40), (50, 0.60), (80, 0.80), (150, 1.0)]
INT_ANCHORS = [(0, 0.0), (3, 0.25), (15, 0.50), (30, 0.75), (50, 1.0)]


def piecewise(x, anchors):
    xs = np.array([a[0] for a in anchors]); ys = np.array([a[1] for a in anchors])
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    return np.clip(np.interp(x, xs, ys, left=ys[0], right=ys[-1]), 0, 1)


def at_day_fn(ta, rh):
    ta = np.asarray(ta, dtype=float); rh = np.asarray(rh, dtype=float)
    out = ta.copy()
    mask = ta >= 27
    tm, rm = ta[mask], rh[mask]
    Xm = np.column_stack([np.ones_like(tm), tm, rm, tm*rm, tm**2, rm**2, tm**2*rm, tm*rm**2, tm**3, rm**3])
    out[mask] = Xm @ COEF_DAY
    return out


def at_night_fn(ta, ws):
    ta = np.asarray(ta, dtype=float); ws = np.asarray(ws, dtype=float)
    out = ta.copy()
    mask = ta < 11
    tm, wm = ta[mask], ws[mask]
    Xm = np.column_stack([np.ones_like(tm), tm, wm, tm*wm, tm**2, wm**2, tm**2*wm, tm*wm**2, tm**3])
    out[mask] = Xm @ COEF_NIGHT
    return out


def s_thermal(t1h, reh, wsd):
    heat = piecewise(at_day_fn(t1h, reh), HEAT_ANCHORS)
    cold = piecewise(-at_night_fn(t1h, wsd), COLD_ANCHORS_NEG)
    return np.maximum(heat, cold)


def thi_calc(ta, rh):
    return 1.8 * ta - 0.55 * (1 - rh / 100) * (1.8 * ta - 26) + 32


def s_humidity(t1h, reh):
    thi_s = piecewise(thi_calc(t1h, reh), [(68, 0.0), (75, 0.5), (80, 1.0)])
    rh_dev = np.clip(np.where(reh <= 40, (40 - reh) / 40, np.where(reh >= 60, (reh - 60) / 40, 0)), 0, 1)
    return np.maximum(thi_s, rh_dev)


def s_wind(wsd):
    return piecewise(wsd, WIND_ANCHORS)


def s_precip(rn1):
    return np.maximum(piecewise(rn1, AMT_ANCHORS), piecewise(rn1, INT_ANCHORS))


def s_pm25_fn(pm25):
    return np.clip(np.interp(pm25, PM25_X, PM25_Y, left=PM25_Y[0], right=PM25_Y[-1]), 0, 1)


def s_pmc_fn(pmc):
    return np.clip(np.interp(pmc, PMC_X, PMC_Y, left=PMC_Y[0], right=PMC_Y[-1]), 0, 1)


def s_airquality(pm25, pm10):
    pmc = max(pm10 - pm25, 0.0)
    return max(float(s_pm25_fn(pm25)), float(s_pmc_fn(pmc)))


GRADE_HEAD = {
    "아주좋음": "실외 활동하기 아주 좋은 날씨예요", "좋음": "실외 활동에 좋은 날씨예요",
    "무난": "실외 활동하기 무난해요", "주의": "가벼운 실외 활동은 괜찮아요", "나쁨": "실외 활동은 조금 주의하세요",
}


def grade_of_score(k):
    if k is None or np.isnan(k):
        return ""
    return "아주좋음" if k >= 85 else "좋음" if k >= 70 else "무난" if k >= 60 else "주의" if k >= 50 else "나쁨"


def danger_factors(at_day, at_night, pm25, pm10, rn_day, rn_hr1, ws_max):
    d = []
    if at_day is not None and at_day >= 35:
        d.append((3, "폭염, 온열질환 주의"))
    if at_night is not None and at_night <= -12:
        d.append((3, "한파, 한랭질환 주의"))
    if (pm25 is not None and pm25 >= 76) or (pm10 is not None and pm10 >= 151):
        d.append((3, "미세먼지 매우 나쁨, 마스크 착용"))
    if (rn_day is not None and rn_day >= 80) or (rn_hr1 is not None and rn_hr1 >= 30):
        d.append((4, "호우, 침수·안전 주의"))
    if ws_max is not None and ws_max >= 21:
        d.append((3, "강풍, 안전 주의"))
    return d


def gentle_notes(at_day, at_night, pm25, pm10, rn_day, ws_max):
    n = []
    if at_day is not None and 33 <= at_day < 35:
        n.append("더위 주의")
    elif at_night is not None and -12 < at_night <= -6:
        n.append("쌀쌀해요, 옷 챙기세요")
    if (pm25 is not None and 36 <= pm25 < 76) or (pm10 is not None and 81 <= pm10 < 151):
        n.append("미세먼지 주의")
    if rn_day is not None and 30 <= rn_day < 80:
        n.append("비 많음, 우산 필수")
    elif rn_day is not None and 5 <= rn_day < 30:
        n.append("비 소식, 우산 챙기세요")
    if ws_max is not None and 14 <= ws_max < 21:
        n.append("바람 강함")
    return n


def build_display(ktci, at_day, at_night, pm25, pm10, rn_day, rn_hr1, ws_max):
    d = danger_factors(at_day, at_night, pm25, pm10, rn_day, rn_hr1, ws_max)
    if d:
        d.sort(key=lambda x: -x[0])
        phrases = [p for _, p in d]
        cap = 45 if len(d) >= 2 else 55
        grade = "나쁨" if len(d) >= 2 else "주의"
        return round(min(ktci, cap), 1), grade, f"⚠️ {', '.join(phrases)} — 실외 활동에 주의하세요"

    g = grade_of_score(ktci)
    notes = gentle_notes(at_day, at_night, pm25, pm10, rn_day, ws_max)
    if g in ("아주좋음", "좋음"):
        comment = f"{GRADE_HEAD[g]} · {notes[0]}" if notes else GRADE_HEAD[g]
    else:
        comment = f"{GRADE_HEAD[g]} — {', '.join(notes)}" if notes else GRADE_HEAD[g]
    return round(ktci, 1), g, comment


def season_weights_no_air(season):
    w = {k: v for k, v in SEASON_WEIGHTS[season].items() if k != "air_quality"}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def parse_rn1(val):
    if val in ("강수없음", "", None):
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        m = re.search(r"([\d.]+)", str(val))
        return float(m.group(1)) if m else 0.0


def pick_base_time():
    now = now_kst()
    if now.minute < 40:
        now -= timedelta(hours=1)
    return now.strftime("%Y%m%d"), now.strftime("%H00")


async def fetch_ncst(client: httpx.AsyncClient, sem: asyncio.Semaphore, nx: int, ny: int, base_date: str, base_time: str):
    url = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst"
    params = {"pageNo": 1, "numOfRows": 20, "dataType": "JSON", "base_date": base_date,
              "base_time": base_time, "nx": nx, "ny": ny, "authKey": KMA_AUTH_KEY}
    async with sem:
        try:
            r = await client.get(url, params=params, timeout=10)
            data = r.json()
            body = data.get("response", {}).get("body")
            if not body:
                return (nx, ny), None
            items = {i["category"]: i["obsrValue"] for i in body["items"]["item"]}
            return (nx, ny), items
        except Exception:
            return (nx, ny), None


async def fetch_seoul_air_quality(client: httpx.AsyncClient):
    if not SEOUL_AQ_AUTH_KEY:
        return {}
    url = f"http://openAPI.seoul.go.kr:8088/{SEOUL_AQ_AUTH_KEY}/json/RealtimeCityAir/1/25/"
    try:
        r = await client.get(url, timeout=10)
        rows = r.json()["RealtimeCityAir"]["row"]
        out = {}
        for row in rows:
            name = row.get("MSRSTN_NM")
            pm10, pm25 = row.get("PM"), row.get("FPM")
            if name and pm10 not in (None, "", "-") and pm25 not in (None, "", "-"):
                out[name] = {"PM10": float(pm10), "PM25": float(pm25)}
        return out
    except Exception:
        return {}


_CACHE: dict = {"key": None, "data": None}
_LOCK = asyncio.Lock()
MIN_HEALTHY_DISTRICTS = 100  # 이보다 적게 받아오면 KMA 쪽 문제로 보고 이전 캐시를 유지한다

# 공유 캐시: 서버리스 인스턴스가 여러 개 떠도(동시접속 몰릴 때) 전부 같은 캐시를 보게 하기
# 위한 저장소. 인메모리 캐시(_CACHE)는 같은 인스턴스가 재사용될 때만 도움이 되고, 인스턴스가
# 여러 개 뜨면 각자 따로 KMA를 불러버린다 - 그걸 막아준다.
# 실제 데이터(60KB+)는 Vercel Blob에 "그 시간대 전용 경로"로 저장한다(덮어쓰기 하면 CDN이
# 예전 캐시를 한동안 계속 돌려주는 걸 확인해서, 시간대마다 새 경로를 씀). Edge Config에는
# 그 Blob의 URL만 담은 작은 "포인터"만 저장한다(Edge Config는 Hobby 플랜에서 전체 8KB
# 제한이라 264개 지역 데이터를 통째로 넣을 수 없다).
EDGE_CONFIG_ID = os.environ.get("EDGE_CONFIG_ID")
EDGE_CONFIG_READ_TOKEN = os.environ.get("EDGE_CONFIG_READ_TOKEN")
VERCEL_API_TOKEN = os.environ.get("VERCEL_API_TOKEN")
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")
EDGE_CONFIG_ENABLED = bool(EDGE_CONFIG_ID and EDGE_CONFIG_READ_TOKEN and VERCEL_API_TOKEN)
BLOB_ENABLED = bool(BLOB_READ_WRITE_TOKEN)
SHARED_CACHE_ENABLED = EDGE_CONFIG_ENABLED and BLOB_ENABLED


async def _pointer_read(client: httpx.AsyncClient) -> dict | None:
    try:
        r = await client.get(
            f"https://edge-config.vercel.com/{EDGE_CONFIG_ID}/item/weather_pointer",
            params={"token": EDGE_CONFIG_READ_TOKEN}, timeout=5,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


async def _pointer_write(client: httpx.AsyncClient, base_date: str, base_time: str, blob_url: str) -> None:
    try:
        await client.patch(
            f"https://api.vercel.com/v1/edge-config/{EDGE_CONFIG_ID}/items",
            json={"items": [{"operation": "upsert", "key": "weather_pointer",
                             "value": {"baseDate": base_date, "baseTime": base_time, "blobUrl": blob_url}}]},
            headers={"Authorization": f"Bearer {VERCEL_API_TOKEN}"}, timeout=10,
        )
    except Exception:
        pass  # 포인터 쓰기 실패는 무시한다 - 다음 요청이 다시 계산하면 그만이다


async def _blob_write(client: httpx.AsyncClient, base_date: str, base_time: str, data: dict) -> str | None:
    pathname = f"weather_{base_date}_{base_time}.json"
    try:
        r = await client.put(
            f"https://blob.vercel-storage.com/{pathname}",
            content=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}", "x-api-version": "7",
                     "x-add-random-suffix": "0", "content-type": "application/json"},
            timeout=15,
        )
        return r.json().get("url") if r.status_code == 200 else None
    except Exception:
        return None


async def _blob_read(client: httpx.AsyncClient, url: str) -> dict | None:
    try:
        r = await client.get(url, timeout=8)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


async def compute_all() -> dict:
    """KMA는 매시 40분에만 새 값을 발표하므로, 그 발표 주기(base_date/base_time)가 바뀔 때만
    다시 불러온다. 인메모리 캐시로 같은 인스턴스 재요청을 빠르게 막고, Blob+Edge Config
    포인터로 인스턴스가 여러 개 떠도(동시접속) 전부 같은 캐시를 공유하게 한다."""
    base_date, base_time = pick_base_time()
    key = (base_date, base_time)

    if _CACHE["key"] == key and _CACHE["data"] is not None:
        return _CACHE["data"]

    async with _LOCK:
        if _CACHE["key"] == key and _CACHE["data"] is not None:
            return _CACHE["data"]

        async with httpx.AsyncClient() as client:
            pointer = await _pointer_read(client) if SHARED_CACHE_ENABLED else None
            if pointer and (pointer.get("baseDate"), pointer.get("baseTime")) == key:
                shared = await _blob_read(client, pointer["blobUrl"])
                if shared:
                    _CACHE["key"] = key
                    _CACHE["data"] = shared
                    return shared

            fresh = await _compute_fresh(base_date, base_time)
            healthy = len(fresh["districts"]) >= MIN_HEALTHY_DISTRICTS
            # 성공 여부와 무관하게 이번 시간대는 "시도했음"으로 표시해 재요청마다 KMA를 다시
            # 부르지 않게 한다(한도 보호). 다만 결과가 부실하면 예전 공유 캐시가 있으면 그걸,
            # 없으면 인메모리 캐시라도 계속 내보낸다 - 텅 빈 화면보다 낫다.
            _CACHE["key"] = key
            if healthy:
                _CACHE["data"] = fresh
                if SHARED_CACHE_ENABLED:
                    blob_url = await _blob_write(client, base_date, base_time, fresh)
                    if blob_url:
                        await _pointer_write(client, base_date, base_time, blob_url)
            elif pointer:
                shared = await _blob_read(client, pointer["blobUrl"])
                if shared:
                    _CACHE["data"] = shared
                elif _CACHE["data"] is None:
                    _CACHE["data"] = fresh
            elif _CACHE["data"] is None:
                _CACHE["data"] = fresh
            return _CACHE["data"]


async def _compute_fresh(base_date: str, base_time: str) -> dict:
    season = MONTH_TO_SEASON[now_kst().month]
    weights = season_weights_no_air(season)
    weights_full = {k: v for k, v in SEASON_WEIGHTS[season].items()}

    unique_cells = sorted({(r["nx"], r["ny"]) for r in REGIONS})

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(80)
        cell_results, seoul_aq = await asyncio.gather(
            asyncio.gather(*(fetch_ncst(client, sem, nx, ny, base_date, base_time) for nx, ny in unique_cells)),
            fetch_seoul_air_quality(client),
        )
    cell_obs = dict(cell_results)

    results = []
    for r in REGIONS:
        obs = cell_obs.get((r["nx"], r["ny"]))
        if obs is None:
            continue
        try:
            t1h = float(obs.get("T1H", "nan"))
            reh = float(obs.get("REH", "nan"))
            wsd = float(obs.get("WSD", "nan"))
        except ValueError:
            continue
        rn1 = parse_rn1(obs.get("RN1"))

        scores = {
            "thermal": (1 - s_thermal(np.array([t1h]), np.array([reh]), np.array([wsd]))[0]) * 100,
            "humidity": (1 - s_humidity(np.array([t1h]), np.array([reh]))[0]) * 100,
            "wind": (1 - s_wind(np.array([wsd]))[0]) * 100,
            "precipitation": (1 - s_precip(np.array([rn1]))[0]) * 100,
        }

        aq = seoul_aq.get(r["signguNm"]) if r["province"] == "서울특별시" else None
        if aq is not None:
            aq_stress = s_airquality(aq["PM25"], aq["PM10"])
            scores["air_quality"] = (1 - aq_stress) * 100
            ktci = sum(scores[k] * weights_full[k] for k in weights_full)
        else:
            ktci = sum(scores[k] * weights[k] for k in weights)

        at_day = float(at_day_fn(np.array([t1h]), np.array([reh]))[0])
        at_night = float(at_night_fn(np.array([t1h]), np.array([wsd]))[0])
        pm25 = aq["PM25"] if aq is not None else None
        pm10 = aq["PM10"] if aq is not None else None
        display_score, grade, comment = build_display(
            ktci=float(ktci), at_day=at_day, at_night=at_night,
            pm25=pm25, pm10=pm10, rn_day=rn1, rn_hr1=rn1, ws_max=wsd,
        )

        result = {
            "signguCode": r["signguCode"], "signguNm": r["signguNm"], "province": r["province"],
            "lat": r["lat"], "lon": r["lon"],
            "T1H": t1h, "REH": reh, "WSD": wsd, "RN1": rn1,
            "score": round(float(ktci), 1),
            "displayScore": display_score, "grade": grade, "comment": comment,
        }
        if aq is not None:
            result["PM10"] = aq["PM10"]
            result["PM25"] = aq["PM25"]
        results.append(result)

    return {
        "updatedAt": now_kst().strftime("%Y-%m-%d %H:%M"),
        "baseDate": base_date, "baseTime": base_time, "season": season,
        "districts": results,
    }
