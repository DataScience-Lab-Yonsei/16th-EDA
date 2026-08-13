"""
264개 시군구 대표 좌표를 1회 채워 config/regions.json에 저장한다.

    python scripts/build_region_coords.py

Nominatim 이용정책이 초당 1회라 전체 264곳에 약 5분이 걸린다. 한 번 돌려 두면
서버는 이 파일만 읽으므로 이후 실행에서는 지오코딩 호출이 아예 없다.
중간에 끊겨도 이미 채운 좌표는 저장되므로 다시 돌리면 남은 곳부터 이어서 채운다.

카카오 로컬 API 키가 있다면 KAKAO_KEY 환경변수를 주면 그쪽을 쓴다. 훨씬 빠르고
국내 행정구역 정확도가 높다.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "regions.json"
KAKAO_KEY = os.environ.get("KAKAO_KEY")
UA = "ktci-trip-score/0.4 (research prototype)"


async def kakao(client: httpx.AsyncClient, query: str):
    r = await client.get("https://dapi.kakao.com/v2/local/search/address.json",
                         params={"query": query},
                         headers={"Authorization": f"KakaoAK {KAKAO_KEY}"})
    r.raise_for_status()
    docs = r.json().get("documents", [])
    return (float(docs[0]["y"]), float(docs[0]["x"])) if docs else None


async def nominatim(client: httpx.AsyncClient, query: str):
    r = await client.get("https://nominatim.openstreetmap.org/search",
                         params={"q": query, "format": "jsonv2", "limit": 1,
                                 "countrycodes": "kr", "accept-language": "ko"},
                         headers={"User-Agent": UA})
    r.raise_for_status()
    hits = r.json()
    return (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None


async def main():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    todo = [r for r in data["regions"] if r.get("lat") is None]
    total = len(data["regions"])

    if not todo:
        print(f"이미 {total}곳 모두 채워져 있습니다.")
        return

    source = "카카오" if KAKAO_KEY else "Nominatim"
    delay = 0.05 if KAKAO_KEY else 1.1
    print(f"{source}로 {len(todo)}곳을 채웁니다 (전체 {total}곳). "
          f"예상 {len(todo) * delay / 60:.1f}분\n")

    fetch = kakao if KAKAO_KEY else nominatim
    done = failed = 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i, r in enumerate(todo, 1):
            try:
                pos = await fetch(client, r["query"])
                if pos:
                    r["lat"], r["lon"] = round(pos[0], 5), round(pos[1], 5)
                    done += 1
                    mark = "OK  "
                else:
                    failed += 1
                    mark = "없음"
            except Exception as e:
                failed += 1
                mark = f"실패({type(e).__name__})"
            print(f"  [{i:3}/{len(todo)}] {mark} {r['label']}")

            if i % 20 == 0:                       # 중간 저장
                CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            await asyncio.sleep(delay)

    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    filled = sum(1 for r in data["regions"] if r.get("lat") is not None)
    print(f"\n채움 {done} · 실패 {failed} · 누적 {filled}/{total}")
    if failed:
        print("실패한 곳은 다시 실행하면 이어서 시도합니다.")


if __name__ == "__main__":
    asyncio.run(main())
