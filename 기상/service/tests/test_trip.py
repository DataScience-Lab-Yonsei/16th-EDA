"""일별·시군구별 여행 점수 검증. 예보 API 없이 가짜 시간별 데이터로 돌린다."""
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import regions as R
from app.trip import alternatives, build_trip, score_region_day

TODAY = date.today()
D1, D2 = TODAY + timedelta(days=1), TODAY + timedelta(days=2)


def hourly(days, kind):
    """kind별 하루치 시간 데이터를 여러 날 만든다."""
    out = {}
    for d in days:
        for h in range(24):
            t = datetime(d.year, d.month, d.day, h)
            if kind == "clear":
                out[t] = dict(ta=18 + 5 * (1 - abs(h - 14) / 14), hm=52, ws=1.6, rn=0.0,
                              pm25=9, pm10=18, o3=0.025, no2=0.01, so2=0.002, co=0.3)
            elif kind == "storm":
                out[t] = dict(ta=15, hm=94, ws=9.0, rn=3.5, pm25=12, pm10=24,
                              o3=0.02, no2=0.01, so2=0.002, co=0.3)
            else:  # dusty
                out[t] = dict(ta=12, hm=35, ws=5.0, rn=0.0, pm25=68, pm10=150,
                              o3=0.05, no2=0.04, so2=0.008, co=1.1)
    return out


# ---------------------------------------------------------------- 지역 레지스트리
def test_264_regions_loaded():
    assert len(R.REGIONS) == 264
    assert R.get("11110")["name"] == "종로구"
    assert R.get("26350")["region_group"] == "영남권"


def test_duplicate_names_are_disambiguated():
    """중구는 6곳이다. 검색은 전부 돌려주고, 라벨로 구분돼야 한다."""
    hits = R.search("중구", limit=20)
    assert len(hits) == 6
    labels = {h["label"] for h in hits}
    assert len(labels) == 6
    assert "서울 중구" in labels and "부산 중구" in labels


def test_search_by_code_and_partial_name():
    assert R.search("11110")[0]["code"] == "11110"
    assert any(h["name"] == "해운대구" for h in R.search("해운대"))


def test_every_region_has_group_and_query():
    groups = {"수도권", "강원권", "충청권", "영남권", "호남권", "제주권"}
    for r in R.REGIONS:
        assert r["region_group"] in groups
        assert r["query"].startswith(r["sido"]) and r["name"] in r["query"]


# ---------------------------------------------------------------- 집계
def _plan_and_meta(spec):
    plan = [{"date": d, "regions": codes} for d, codes in spec]
    meta = {c: R.get(c) for _, codes in spec for c in codes}
    return plan, meta


def test_day_score_is_mean_of_regions():
    plan, meta = _plan_and_meta([(D1, ["11110", "26350"])])
    hb = {"11110": hourly([D1], "clear"), "26350": hourly([D1], "storm")}
    out = build_trip(plan, hb, meta, with_alternatives=False)
    regs = out["days"][0]["regions"]
    assert len(regs) == 2
    expect = round((regs[0]["score"] + regs[1]["score"]) / 2, 1)
    assert abs(out["days"][0]["score"] - expect) < 0.05


def test_trip_score_weights_days_equally():
    """1일차 2곳, 2일차 1곳. 여행 점수는 지역 평균이 아니라 일자 평균이어야 한다."""
    plan, meta = _plan_and_meta([(D1, ["11110", "11140"]), (D2, ["51150"])])
    hb = {"11110": hourly([D1], "clear"), "11140": hourly([D1], "clear"),
          "51150": hourly([D2], "storm")}
    out = build_trip(plan, hb, meta, with_alternatives=False)
    d1, d2 = (d["score"] for d in out["days"])
    assert abs(out["summary"]["trip_score"] - round((d1 + d2) / 2, 1)) < 0.05
    # 지역-일 단순 평균은 1일차 쪽으로 쏠려 다른 값이 나온다
    assert out["summary"]["region_day_mean"] != out["summary"]["trip_score"]


def test_visit_order_is_preserved():
    plan, meta = _plan_and_meta([(D1, ["26350", "11110", "51150"])])
    hb = {c: hourly([D1], "clear") for c in ["26350", "11110", "51150"]}
    out = build_trip(plan, hb, meta, with_alternatives=False)
    assert [r["order"] for r in out["days"][0]["regions"]] == [1, 2, 3]
    assert [r["code"] for r in out["days"][0]["regions"]] == ["26350", "11110", "51150"]


def test_storm_scores_below_clear():
    plan, meta = _plan_and_meta([(D1, ["11110", "26350"])])
    hb = {"11110": hourly([D1], "clear"), "26350": hourly([D1], "storm")}
    out = build_trip(plan, hb, meta, with_alternatives=False)
    clear, storm = out["days"][0]["regions"]
    assert clear["score"] > storm["score"]
    assert storm["top_component"]["component"] == "precipitation"


def test_dust_day_is_flagged_by_air_quality():
    plan, meta = _plan_and_meta([(D1, ["11110"])])
    out = build_trip(plan, {"11110": hourly([D1], "dusty")}, meta, with_alternatives=False)
    r = out["days"][0]["regions"][0]
    assert r["top_component"]["component"] in ("air_quality", "humidity")
    assert r["subindices"]["air_quality"] >= 0.5


# ---------------------------------------------------------------- 계획 수정
def test_alternatives_find_a_better_date():
    """1일차는 비, 2일차는 맑음 -> 1일차 방문에 2일차를 추천해야 한다."""
    h = {**hourly([D1], "storm"), **hourly([D2], "clear")}
    cur = score_region_day(h, D1)["result"].score_100
    alts = alternatives(h, D1, "hybrid", cur)
    assert alts and alts[0]["date"] == D2.isoformat()
    assert alts[0]["gain"] > 0 and alts[0]["score"] > cur


def test_no_alternatives_when_already_best():
    h = {**hourly([D1], "clear"), **hourly([D2], "storm")}
    cur = score_region_day(h, D1)["result"].score_100
    assert alternatives(h, D1, "hybrid", cur) == []


def test_fixable_list_points_at_the_worst_first():
    plan, meta = _plan_and_meta([(D1, ["11110", "26350"])])
    both = {**hourly([D1], "storm"), **hourly([D2], "clear")}
    hb = {"11110": {**hourly([D1], "clear"), **hourly([D2], "clear")}, "26350": both}
    out = build_trip(plan, hb, meta)
    fixable = out["summary"]["fixable"]
    assert fixable and fixable[0]["code"] == "26350"
    assert fixable[0]["best_alternative"]["score"] > fixable[0]["score"]


def test_out_of_forecast_range_is_reported_not_crashed():
    plan, meta = _plan_and_meta([(D1, ["11110"])])
    out = build_trip(plan, {"11110": {}}, meta, with_alternatives=False)
    r = out["days"][0]["regions"][0]
    assert r["score"] is None and r["note"]
    assert out["summary"]["trip_score"] is None


# ---------------------------------------------------------------- UI 계약
def test_ui_reads_existing_fields():
    plan, meta = _plan_and_meta([(D1, ["11110"])])
    out = build_trip(plan, {"11110": {**hourly([D1], "clear"), **hourly([D2], "clear")}}, meta)
    assert {"summary", "variant", "days", "method"} <= set(out)
    assert {"trip_score", "grade", "region_day_mean", "region_day_count",
            "worst", "best", "fixable"} <= set(out["summary"])
    r = out["days"][0]["regions"][0]
    assert {"order", "code", "name", "label", "region_group", "score", "grade",
            "subindices", "weights_used", "missing", "top_component",
            "alternatives", "valid_hours"} <= set(r)


def test_ui_constants_match_backend():
    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()
    ui = re.search(r'const COMPONENTS=\[(.*?)\]', html).group(1)
    from app.scoring import COMPONENTS
    assert [c.strip().strip('"') for c in ui.split(",")] == COMPONENTS
