"""
시군구 레지스트리.

워크북 행정계층 시트의 264개 지역을 그대로 쓴다. 이름이 겹치는 곳이 많아서
(중구 6곳, 동구 6곳, 서구 5곳, 남구·북구 4곳, 강서구·고성군 2곳) 화면과 조회 모두
signguCode를 키로 쓰고 사람에게는 '시도 + 시군구'로 보여준다.

좌표는 config/regions.json에 비어 있으면 요청 시점에 지오코딩해 메모리에 캐시한다.
scripts/build_region_coords.py를 한 번 돌려 두면 그 파일에 박히므로 이후엔 호출이 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "regions.json"

with CONFIG_PATH.open(encoding="utf-8") as f:
    _DATA = json.load(f)

REGIONS: list[dict] = _DATA["regions"]
BY_CODE: dict[str, dict] = {r["code"]: r for r in REGIONS}

# 실행 중 채워지는 좌표 캐시 (regions.json이 비어 있을 때만 사용)
_coord_cache: dict[str, tuple[float, float]] = {
    r["code"]: (r["lat"], r["lon"]) for r in REGIONS if r.get("lat") is not None
}


def get(code: str) -> dict:
    try:
        return BY_CODE[str(code)]
    except KeyError:
        raise ValueError(f"모르는 시군구 코드입니다: {code}") from None


def search(q: str, limit: int = 12) -> list[dict]:
    """이름·시도·코드로 찾는다. '중구'처럼 겹치는 이름은 시도가 다른 항목이 모두 나온다."""
    q = q.strip()
    if not q:
        return []
    exact, prefix, partial = [], [], []
    for r in REGIONS:
        if r["code"].startswith(q):
            exact.append(r)
        elif r["name"] == q:
            exact.append(r)
        elif r["name"].startswith(q) or r["label"].startswith(q):
            prefix.append(r)
        elif q in r["label"] or q in r["sido"]:
            partial.append(r)
    out, seen = [], set()
    for r in exact + prefix + partial:
        if r["code"] not in seen:
            seen.add(r["code"])
            out.append(_public(r))
        if len(out) >= limit:
            break
    return out


def _public(r: dict) -> dict:
    out = {k: r[k] for k in ("code", "name", "sido", "sido_short", "region_group", "label")}
    out["lat"], out["lon"] = r.get("lat"), r.get("lon")
    return out


async def coords(code: str) -> tuple[float, float]:
    """시군구 대표 좌표. 없으면 '시도 + 시군구'로 지오코딩해 캐시한다."""
    code = str(code)
    if code in _coord_cache:
        return _coord_cache[code]
    from . import geocode          # httpx 의존을 여기서만 진다 (조회는 오프라인에서도 동작)

    r = get(code)
    hits = await geocode.search(r["query"], limit=1)
    if not hits:
        raise ValueError(f"{r['label']}의 좌표를 찾지 못했습니다. "
                         f"scripts/build_region_coords.py를 먼저 실행해 주세요.")
    pos = (hits[0]["lat"], hits[0]["lon"])
    _coord_cache[code] = pos
    return pos


def cached_count() -> int:
    return len(_coord_cache)
