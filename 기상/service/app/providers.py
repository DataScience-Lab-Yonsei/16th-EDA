"""
예보 데이터 제공자.

인터페이스는 하나뿐이다:
    fetch_hourly(lat, lon, days) -> {datetime(정시): {ta, hm, ws, rn,
                                                      pm25, pm10, o3, no2, so2, co}}
    단위: ta C / hm % / ws m/s / rn mm(1시간) / PM ug/m3 / 가스 ppm

KTCI는 일 단위 지수이므로 시간별 값을 받아 aggregate.build_daily() 로 일 대표값을
만든 뒤 채점한다. 유효조건(>=18/24)을 판정하려면 하루치 전체가 필요하다.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from .aggregate import ugm3_to_ppm

KST = "Asia/Seoul"
HourlyMap = dict[datetime, dict]


def now_kst() -> datetime:
    """서버가 어느 시간대에서 돌든(Vercel은 UTC) KMA 발표시각은 항상 KST 기준이다."""
    return datetime.now(ZoneInfo(KST)).replace(tzinfo=None)


# ---------------------------------------------------------------- Open-Meteo
class OpenMeteoProvider:
    """키 불필요. 기상 최대 16일 + 대기질(PM2.5/PM10/O3/NO2/SO2/CO) 예보."""

    FORECAST = "https://api.open-meteo.com/v1/forecast"
    AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 15.0):
        self._client = client
        self._timeout = timeout

    async def fetch_hourly(self, lat: float, lon: float, days: int = 7) -> HourlyMap:
        w_params = {
            "latitude": lat, "longitude": lon, "timezone": KST,
            "forecast_days": max(1, min(days, 16)), "wind_speed_unit": "ms",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        }
        a_params = {
            "latitude": lat, "longitude": lon, "timezone": KST,
            "forecast_days": max(1, min(days, 5)),
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        }

        client, close = self._client, False
        if client is None:
            client, close = httpx.AsyncClient(timeout=self._timeout), True
        try:
            wr, ar = await asyncio.gather(
                client.get(self.FORECAST, params=w_params),
                client.get(self.AIR, params=a_params),
                return_exceptions=True,
            )
        finally:
            if close:
                await client.aclose()

        obs: HourlyMap = {}
        if not isinstance(wr, Exception):
            h = wr.json()["hourly"]
            for i, ts in enumerate(h["time"]):
                obs[datetime.fromisoformat(ts)] = {
                    "ta": h["temperature_2m"][i],
                    "hm": h["relative_humidity_2m"][i],
                    "ws": h["wind_speed_10m"][i],
                    "rn": h["precipitation"][i],
                }
        if not isinstance(ar, Exception):
            h = ar.json()["hourly"]
            for i, ts in enumerate(h["time"]):
                t = datetime.fromisoformat(ts)
                row = obs.setdefault(t, {})
                row["pm25"] = h["pm2_5"][i]
                row["pm10"] = h["pm10"][i]
                row["o3"] = ugm3_to_ppm(h["ozone"][i], "o3")
                row["no2"] = ugm3_to_ppm(h["nitrogen_dioxide"][i], "no2")
                row["so2"] = ugm3_to_ppm(h["sulphur_dioxide"][i], "so2")
                row["co"] = ugm3_to_ppm(h["carbon_monoxide"][i], "co")
        return obs


# ------------------------------------------------------- Open-Meteo (대기질 전용)
class OpenMeteoAirProvider:
    """KMA 예보엔 대기질이 없어서 보충용으로만 쓴다. 키 불필요, PM2.5/PM10/O3/NO2/SO2/CO 5일 예보."""

    AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 15.0):
        self._client = client
        self._timeout = timeout

    async def fetch_hourly(self, lat: float, lon: float, days: int = 5) -> HourlyMap:
        params = {
            "latitude": lat, "longitude": lon, "timezone": KST,
            "forecast_days": max(1, min(days, 5)),
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        }
        client, close = self._client, False
        if client is None:
            client, close = httpx.AsyncClient(timeout=self._timeout), True
        try:
            r = await client.get(self.AIR, params=params)
        finally:
            if close:
                await client.aclose()

        obs: HourlyMap = {}
        if r.status_code == 200:
            h = r.json()["hourly"]
            for i, ts in enumerate(h["time"]):
                obs[datetime.fromisoformat(ts)] = {
                    "pm25": h["pm2_5"][i],
                    "pm10": h["pm10"][i],
                    "o3": ugm3_to_ppm(h["ozone"][i], "o3"),
                    "no2": ugm3_to_ppm(h["nitrogen_dioxide"][i], "no2"),
                    "so2": ugm3_to_ppm(h["sulphur_dioxide"][i], "so2"),
                    "co": ugm3_to_ppm(h["carbon_monoxide"][i], "co"),
                }
        return obs


# ---------------------------------------------------------------- 기상청(KMA)
def dfs_xy_conv(lat: float, lon: float) -> tuple[int, int]:
    """위경도 -> 기상청 격자 nx, ny (Lambert Conformal Conic, 기상청 공식 파라미터)."""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136
    DEGRAD = math.pi / 180.0
    re, slat1, slat2 = RE / GRID, SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn

    ra = re * sf / math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5) ** sn
    theta = lon * DEGRAD - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


class KMAProvider:
    """
    기상청 API허브(apihub.kma.go.kr) 단기예보(getVilageFcst). 사용 카테고리:
      TMP 기온 / REH 습도 / WSD 풍속 / PCP 1시간 강수량
    미세먼지·가스는 포함되지 않으므로 air_provider(에어코리아 등)를 주입해 합친다.
    예보 범위가 +3일이라 그 이후 일정은 그 이상 값이 없다(결측 처리는 scoring의
    weighted_available가 담당).
    """

    URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
    BASE_TIMES = ["2300", "2000", "1700", "1400", "1100", "0800", "0500", "0200"]

    def __init__(self, auth_key: str, fallback_provider=None, timeout: float = 15.0):
        self.auth_key = auth_key
        self.fallback_provider = fallback_provider
        self._timeout = timeout

    def _latest_base(self, now: datetime) -> tuple[str, str]:
        for bt in self.BASE_TIMES:
            issued = now.replace(hour=int(bt[:2]), minute=int(bt[2:]), second=0, microsecond=0)
            if now >= issued + timedelta(minutes=10):
                return issued.strftime("%Y%m%d"), bt
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "2300"

    async def fetch_hourly(self, lat: float, lon: float, days: int = 3) -> HourlyMap:
        nx, ny = dfs_xy_conv(lat, lon)
        base_date, base_time = self._latest_base(now_kst())
        params = {"authKey": self.auth_key, "dataType": "JSON", "numOfRows": 1000,
                  "pageNo": 1, "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(self.URL, params=params)
            r.raise_for_status()
            items = r.json()["response"]["body"]["items"]["item"]

        buckets: dict[datetime, dict] = {}
        for it in items:
            t = datetime.strptime(it["fcstDate"] + it["fcstTime"], "%Y%m%d%H%M")
            buckets.setdefault(t, {})[it["category"]] = it["fcstValue"]

        obs: HourlyMap = {t: {"ta": _num(c.get("TMP")), "hm": _num(c.get("REH")),
                              "ws": _num(c.get("WSD")), "rn": _pcp(c.get("PCP"))}
                          for t, c in buckets.items()}

        if self.fallback_provider is not None:
            # 대기질은 KMA에 아예 없으니 항상 채운다. 기온·습도·바람·강수는 KMA 단기예보가
            # 발표 시점에서 멀어질수록(대략 +2.5일 이후) 1시간 -> 3시간 간격으로 성겨지는데,
            # 그 빈 시각만 Open-Meteo로 메운다 - 실제 KMA 관측이 있는 시각은 건드리지 않는다.
            fb = await self.fallback_provider.fetch_hourly(lat, lon, days)
            for t, v in fb.items():
                row = obs.setdefault(t, {})
                for k in ("pm25", "pm10", "o3", "no2", "so2", "co"):
                    row[k] = v.get(k)
                for k in ("ta", "hm", "ws", "rn"):
                    if row.get(k) is None and v.get(k) is not None:
                        row[k] = v.get(k)
        return obs


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pcp(v):
    """기상청 PCP는 '강수없음' / '1mm 미만' / '30.0~50.0mm' 같은 문자열로 온다."""
    if v is None:
        return None
    v = str(v).strip()
    if v in ("강수없음", "-", "", "0"):
        return 0.0
    if "미만" in v:
        return 0.5
    if "~" in v:
        v = v.split("~")[0]
    v = v.replace("이상", "").replace("mm", "").strip()
    try:
        return float(v)
    except ValueError:
        return None
