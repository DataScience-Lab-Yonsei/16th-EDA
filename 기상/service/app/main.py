"""
여행 일정 -> 날씨 기반 여행 점수 API.

실행:
    uvicorn app.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

from datetime import date as date_type, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import geocode, regions as regions_mod
from .aggregate import build_daily, daily_coverage
from .providers import KMAProvider, OpenMeteoProvider, now_kst
from .response import build_response
from .trip import build_trip
from .scoring import (COMPONENTS, EXPOSURE_FACTOR, KTCI_CONFIG, Stop,
                      get_weights, score_stop)
from .weather_today import compute_all as compute_weather_today

# 기상청 API허브 인증키 (프로젝트 전체에서 공통으로 쓰는 키)
KMA_AUTH_KEY = 'x6QFqFUGS7KkBahVBpuyOQ'

app = FastAPI(title="Weather-based Trip Score (KTCI)", version="0.3.0")
provider = KMAProvider(auth_key=KMA_AUTH_KEY, fallback_provider=OpenMeteoProvider())

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    # Vercel 배포본에는 static/이 없다 - 그쪽은 Vercel이 정적 파일을 직접 서빙한다.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")


@app.get("/weather/today")
async def weather_today():
    """전국 264개 시군구 실시간 날씨 점수. 요청마다 KMA·서울 대기질을 새로 불러와 계산한다."""
    return await compute_weather_today()


@app.get("/regions/search")
def regions_search(q: str, limit: int = 12):
    """시군구 자동완성. 이름이 겹치는 곳(중구 6개 등)은 시도가 다른 항목이 모두 나온다."""
    return {"query": q, "results": regions_mod.search(q, limit)}


class DayIn(BaseModel):
    date: date_type = Field(..., examples=["2026-07-28"])
    regions: list[str] = Field(..., description="signguCode 목록, 방문 순서대로",
                               examples=[["11110", "11140"]])


class TripIn(BaseModel):
    days: list[DayIn]
    variant: str = Field("hybrid", description="hybrid | data | survey_2014_adapted")
    alternatives: bool = Field(True, description="예보 범위 안의 더 좋은 날짜를 함께 계산")


@app.post("/score/trip")
async def score_trip(payload: TripIn):
    """일별로 방문할 시군구를 순서대로 받아 지역별·일자별·여행 전체 점수를 낸다."""
    if not payload.days:
        raise HTTPException(400, "일정이 비어 있습니다")

    codes, meta = [], {}
    for d in payload.days:
        if not d.regions:
            raise HTTPException(400, f"{d.date}에 시군구가 없습니다")
        for c in d.regions:
            try:
                meta[c] = regions_mod.get(c)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if c not in codes:
                codes.append(c)

    horizon = max((d.date - now_kst().date()).days for d in payload.days) + 2
    hourly_by_region = {}
    for c in codes:
        try:
            lat, lon = await regions_mod.coords(c)
        except ValueError as e:
            raise HTTPException(503, str(e))
        hourly_by_region[c] = await provider.fetch_hourly(lat, lon, days=max(1, horizon))

    plan = [{"date": d.date, "regions": d.regions} for d in payload.days]
    return build_trip(plan, hourly_by_region, meta, payload.variant, payload.alternatives)


@app.get("/geocode")
async def geocode_search(q: str, limit: int = 5):
    """장소명으로 좌표 찾기. UI의 장소 검색이 이 엔드포인트를 쓴다."""
    return {"query": q, "results": await geocode.search(q, limit)}


def _r(v, n=4):
    """응답 직전에만 반올림한다 (내부 계산은 항상 full precision)."""
    if isinstance(v, dict):
        return {k: _r(x, n) for k, x in v.items()}
    return round(v, n) if isinstance(v, float) else v


class StopIn(BaseModel):
    name: str = Field(..., examples=["경복궁"])
    lat: float = Field(..., examples=[37.5796])
    lon: float = Field(..., examples=[126.9770])
    start: datetime = Field(..., examples=["2026-07-27T10:00:00"])
    end: datetime = Field(..., examples=["2026-07-27T12:00:00"])
    exposure: str = Field("outdoor", description="outdoor | mixed | transit | indoor")


class ItineraryIn(BaseModel):
    stops: list[StopIn]
    variant: str = Field("hybrid", description="hybrid | data | survey_2014_adapted")


@app.get("/health")
def health():
    return {"ok": True, "source": KTCI_CONFIG["source"], "components": COMPONENTS,
            "granularity": "daily (KTCI는 일 단위 지수)",
            "regions": len(regions_mod.REGIONS),
            "regions_with_coords": regions_mod.cached_count()}


@app.get("/weights/{season}")
def weights(season: str, variant: str = "hybrid"):
    try:
        return {"season": season, "variant": variant,
                "weights": _r(get_weights(season, variant)),
                "alpha_survey": KTCI_CONFIG["alpha_survey"].get(season)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/score")
async def score(payload: ItineraryIn):
    if not payload.stops:
        raise HTTPException(400, "stops가 비어 있습니다")
    for s in payload.stops:
        if s.end <= s.start:
            raise HTTPException(400, f"'{s.name}': end가 start보다 빠르거나 같습니다")
        if s.start.date() != s.end.date():
            raise HTTPException(400, f"'{s.name}': 자정을 넘는 스탑은 날짜별로 나눠 주세요")
        if s.exposure not in EXPOSURE_FACTOR:
            raise HTTPException(400, f"'{s.name}': exposure는 {list(EXPOSURE_FACTOR)} 중 하나여야 합니다")

    stops = [Stop(s.name, s.lat, s.lon, s.start, s.end, s.exposure) for s in payload.stops]

    # 같은 격자·같은 날짜는 한 번만 호출한다 (스탑 수만큼 부르면 금방 레이트리밋)
    horizon = max((st.day - now_kst().date()).days for st in stops) + 2
    cache: dict[tuple, dict] = {}
    hourly_cache: dict[tuple, dict] = {}
    for st in stops:
        loc = (round(st.lat, 2), round(st.lon, 2))
        if loc not in hourly_cache:
            hourly_cache[loc] = await provider.fetch_hourly(st.lat, st.lon, days=max(1, horizon))
        if st.grid_key not in cache:
            cache[st.grid_key] = {"hourly": hourly_cache[loc],
                                  "daily": build_daily(hourly_cache[loc], st.day)}

    stop_scores = [score_stop(st, cache[st.grid_key]["daily"], payload.variant) for st in stops]
    valid = {st.name: daily_coverage(cache[st.grid_key]["hourly"], st.day) for st in stops}
    return build_response(stop_scores, payload.variant, valid)
