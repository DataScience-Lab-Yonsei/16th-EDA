"""
일별 · 시군구별 여행 날씨 점수.

입력은 '며칟날 어느 시군구를 순서대로 도는가'다.

    지역-일 점수 = 그 시군구 좌표의 그날 KTCI
    일자 점수    = 그날 방문한 시군구 점수의 평균
    여행 점수    = 일자 점수의 평균 (날짜마다 같은 무게)

여행 점수를 지역-일 전체 평균이 아니라 일자 평균으로 잡은 이유: 하루에 여러 곳을 도는
날이 있으면 그날 날씨가 과대 반영되기 때문이다. 두 값 모두 응답에 담아 비교할 수 있게 했다.

계획 수정을 돕기 위해 각 지역마다 예보 범위 안의 '더 좋은 날짜'를 함께 계산한다.
어차피 예보를 통째로 받아 두므로 추가 호출 없이 나온다.
"""

from __future__ import annotations

from datetime import date

from .aggregate import build_daily, daily_coverage
from .providers import now_kst
from .response import _r
from .scoring import grade_of, score_day, season_of


def score_region_day(hourly: dict, day: date, variant: str = "hybrid") -> dict | None:
    """(시군구, 날짜) 하나. 예보가 없거나 영역이 부족하면 None."""
    daily = build_daily(hourly, day)
    res = score_day(daily, day, variant)
    if res.ktci is None:
        return None
    return {"result": res, "daily": daily}


def forecast_dates(hourly: dict) -> list[date]:
    return sorted({t.date() for t in hourly})


def alternatives(hourly: dict, current: date, variant: str, current_score: float,
                 limit: int = 3) -> list[dict]:
    """예보 범위 안에서 현재 날짜보다 점수가 높은 날들. 좋은 순."""
    out = []
    for d in forecast_dates(hourly):
        if d == current or d < now_kst().date():
            continue
        scored = score_region_day(hourly, d, variant)
        if scored is None:
            continue
        s = scored["result"].score_100
        if s > current_score + 1:          # 1점 이내 차이는 권할 이유가 없다
            out.append({"date": d.isoformat(), "score": s,
                        "gain": round(s - current_score, 1),
                        "grade": grade_of(scored["result"].ktci)})
    out.sort(key=lambda x: -x["score"])
    return out[:limit]


def build_trip(plan: list[dict], hourly_by_region: dict[str, dict],
               region_meta: dict[str, dict], variant: str = "hybrid",
               with_alternatives: bool = True) -> dict:
    """
    plan: [{"date": date, "regions": ["11110", ...]}, ...]   (순서 유지)
    hourly_by_region: {code: 시간별 예보}
    region_meta: {code: regions.get(code)}
    """
    days_out, day_scores, all_region_scores = [], [], []

    for entry in plan:
        day: date = entry["date"]
        regions_out = []

        for order, code in enumerate(entry["regions"], start=1):
            meta = region_meta[code]
            hourly = hourly_by_region.get(code, {})
            scored = score_region_day(hourly, day, variant)

            if scored is None:
                regions_out.append({
                    "order": order, "code": code, "name": meta["name"],
                    "label": meta["label"], "region_group": meta["region_group"],
                    "score": None, "grade": None,
                    "note": "예보 범위를 벗어났거나 관측 영역이 2개 미만이라 점수를 낼 수 없습니다.",
                    "alternatives": [],
                })
                continue

            res, daily = scored["result"], scored["daily"]
            score = res.score_100
            all_region_scores.append(score)
            top = res.top_component

            regions_out.append({
                "order": order, "code": code, "name": meta["name"],
                "label": meta["label"], "region_group": meta["region_group"],
                "score": score, "grade": grade_of(res.ktci), "ktci": _r(res.ktci),
                "subindices": _r(res.subindices), "weights_used": _r(res.weights_used),
                "missing": res.missing,
                "top_component": {"component": top[0], "share": _r(top[1])} if top else None,
                "daily_values": _r(dict(daily.__dict__)),
                "valid_hours": daily_coverage(hourly, day),
                "alternatives": alternatives(hourly, day, variant, score) if with_alternatives else [],
                "note": res.note,
            })

        scored_today = [r["score"] for r in regions_out if r["score"] is not None]
        day_score = round(sum(scored_today) / len(scored_today), 1) if scored_today else None
        if day_score is not None:
            day_scores.append(day_score)

        days_out.append({
            "date": day.isoformat(),
            "season": season_of(day.month),
            "score": day_score,
            "grade": grade_of(1 - day_score / 100) if day_score is not None else None,
            "regions": regions_out,
        })

    trip = round(sum(day_scores) / len(day_scores), 1) if day_scores else None
    flat = round(sum(all_region_scores) / len(all_region_scores), 1) if all_region_scores else None

    ranked = [(d["date"], r) for d in days_out for r in d["regions"] if r["score"] is not None]
    ranked.sort(key=lambda p: p[1]["score"])

    return {
        "summary": {
            "trip_score": trip,
            "grade": grade_of(1 - trip / 100) if trip is not None else None,
            "region_day_mean": flat,
            "day_count": len(days_out),
            "region_day_count": len(all_region_scores),
            "worst": _ref(ranked[0]) if ranked else None,
            "best": _ref(ranked[-1]) if ranked else None,
            "fixable": [
                {**_ref(p), "best_alternative": p[1]["alternatives"][0]}
                for p in ranked if p[1]["alternatives"]
            ][:5],
        },
        "variant": variant,
        "days": days_out,
        "method": ("지역-일 점수 = 그 시군구의 그날 KTCI · "
                   "일자 점수 = 그날 시군구 점수의 평균 · "
                   "여행 점수 = 일자 점수의 평균(날짜마다 같은 무게)"),
    }


def _ref(pair) -> dict:
    d, r = pair
    return {"date": d, "code": r["code"], "label": r["label"], "score": r["score"]}
