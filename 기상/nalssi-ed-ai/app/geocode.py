"""
장소명 -> 위경도.

카카오 로컬 API(키워드 검색)를 쓴다. '경복궁', '감천문화마을' 같은 국내 관광지(POI)
정확도가 Nominatim/Open-Meteo보다 훨씬 높고, 프로젝트 전체에서 이미 같은 키를 쓰고 있다.
"""

from __future__ import annotations

import httpx

KAKAO_KEY = "f7c0f8b37fd587afb82fb71ccfcf780b"
KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


async def search(query: str, limit: int = 5) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(KEYWORD_URL,
                                 params={"query": query, "size": min(limit, 15)},
                                 headers={"Authorization": f"KakaoAK {KAKAO_KEY}"})
            r.raise_for_status()
            return [{
                "name": _short(h["place_name"]),
                "address": h.get("road_address_name") or h.get("address_name", ""),
                "lat": float(h["y"]), "lon": float(h["x"]), "source": "kakao",
            } for h in r.json().get("documents", [])]
        except Exception:
            return []


def _short(name: str) -> str:
    return name if len(name) <= 40 else name[:39] + "…"
