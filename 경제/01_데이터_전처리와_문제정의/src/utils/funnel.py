"""Phase 1 분석: 퍼널 베이스라인 · B2B 가설 검증 · 격차 분해.

각 함수는 polars DataFrame 을 돌려주고, 렌더링은 :mod:`src.run_phase1` 이 맡는다.
"""
from __future__ import annotations

import polars as pl

from src.utils.sessions import DEFAULT_WINDOW, WINDOWS, scan_cart_lines, scan_events

Y = f"y_{DEFAULT_WINDOW}"  # 기본 타겟(7일)


# ══════════════════════════════════════════════════════════════════════
# 1-A. 퍼널 베이스라인
# ══════════════════════════════════════════════════════════════════════
def funnel_overview(store: str) -> dict:
    """스토어 단위 퍼널 요약."""
    ev = scan_events(store)
    cl = scan_cart_lines(store)

    base = ev.select(
        pl.col("user_id").n_unique().alias("users"),
        pl.col("session30").n_unique().alias("sessions30"),
        pl.col("user_session").n_unique().alias("sessions_raw"),
        pl.len().alias("events"),
    ).collect().row(0, named=True)

    by_type = dict(
        ev.group_by("event_type").agg(pl.len().alias("n")).collect().iter_rows()
    )

    c = cl.select(
        pl.len().alias("cart_lines"),
        pl.col("user_id").n_unique().alias("cart_users"),
        pl.col(Y).sum().alias("converted"),
        pl.col(Y).mean().alias("cvr"),
        pl.col("removed_after_cart").mean().alias("removed_rate"),
        pl.col("cart_repeat_cnt").mean().alias("repeat_mean"),
    ).collect().row(0, named=True)

    return {"store": store, **base, **{f"n_{k}": v for k, v in by_type.items()}, **c}


def funnel_stages(store: str) -> pl.DataFrame:
    """(user, session30, product) 단위 3단 퍼널: view → cart → purchase."""
    ev = scan_events(store)
    return (
        ev.group_by("user_id", "session30", "product_id")
        .agg(
            (pl.col("event_type") == "view").any().alias("viewed"),
            (pl.col("event_type") == "cart").any().alias("carted"),
            (pl.col("event_type") == "purchase").any().alias("purchased"),
        )
        .select(
            pl.len().alias("pairs"),
            pl.col("viewed").sum().alias("viewed"),
            pl.col("carted").sum().alias("carted"),
            pl.col("purchased").sum().alias("purchased"),
            (pl.col("viewed") & pl.col("carted")).sum().alias("view_then_cart"),
            (pl.col("carted") & pl.col("purchased")).sum().alias("cart_then_purchase"),
        )
        .collect()
    )


def monthly(store: str) -> pl.DataFrame:
    return (
        scan_cart_lines(store)
        .with_columns(pl.col("t").dt.convert_time_zone("Europe/Moscow")
                      .dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("cart_lines"),
            pl.col(Y).sum().alias("converted"),
            pl.col(Y).mean().alias("cvr"),
        )
        .sort("month")
        .collect()
    )


def by_price_band(store: str, q: int = 5) -> pl.DataFrame:
    """가격 분위별 전환. 물량과 전환율을 함께 봐야 headroom 이 보인다."""
    cl = scan_cart_lines(store).filter(pl.col("price").is_not_null())
    return (
        cl.with_columns(
            ((pl.col("price").rank("ordinal") - 1) * q // pl.len()).alias("band")
        )
        .group_by("band")
        .agg(
            pl.col("price").min().alias("lo"),
            pl.col("price").max().alias("hi"),
            pl.len().alias("cart_lines"),
            pl.col(Y).mean().alias("cvr"),
            (pl.len() * (1 - pl.col(Y).mean())).alias("headroom"),
        )
        .sort("band")
        .collect()
    )


def by_brand(store: str, top: int = 15) -> pl.DataFrame:
    return (
        scan_cart_lines(store)
        .group_by(pl.col("brand").fill_null("(unknown)"))
        .agg(
            pl.len().alias("cart_lines"),
            pl.col(Y).mean().alias("cvr"),
            (pl.len() * (1 - pl.col(Y).mean())).round(0).alias("headroom"),
            pl.col("price").median().alias("price_med"),
        )
        .sort("cart_lines", descending=True)
        .head(top)
        .collect()
    )


def price_missing_effect(store: str) -> pl.DataFrame:
    """Phase 0-D 에서 승격된 가설: 가격 미설정 상품은 전환이 낮은가?"""
    return (
        scan_cart_lines(store)
        .group_by("price_missing")
        .agg(pl.len().alias("cart_lines"), pl.col(Y).mean().alias("cvr"))
        .sort("price_missing")
        .collect()
    )


# ══════════════════════════════════════════════════════════════════════
# 1-B. B2B 가설(H0) 검증
# ══════════════════════════════════════════════════════════════════════
def purchases_per_user(store: str) -> pl.DataFrame:
    """유저별 총 구매 라인 수 분포 + 상위 집중도(파레토)."""
    per = (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .group_by("user_id")
        .agg(pl.len().alias("n"))
        .collect()
        .sort("n", descending=True)
    )
    tot = per["n"].sum()
    out = {"buyers": per.height, "purchase_lines": int(tot)}
    for p in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(per.height * p))
        out[f"top{int(p*100)}pct_share"] = float(per["n"].head(k).sum() / tot)
    for q in (0.5, 0.9, 0.99):
        out[f"p{int(q*100)}"] = float(per["n"].quantile(q))
    out["max"] = int(per["n"].max())
    out["gt5_share"] = float((per["n"] > 5).mean())
    return pl.DataFrame([out])


def orders(store: str) -> pl.DataFrame:
    """주문 = 구매가 포함된 (user, session30). 라인 수·금액 분포."""
    o = (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .group_by("user_id", "session30")
        .agg(
            pl.col("product_id").n_unique().alias("lines"),
            pl.col("price").sum().alias("value"),
        )
        .collect()
    )
    return pl.DataFrame([{
        "orders": o.height,
        "lines_p50": float(o["lines"].median()),
        "lines_p90": float(o["lines"].quantile(0.9)),
        "lines_max": int(o["lines"].max()),
        "lines_mean": float(o["lines"].mean()),
        "value_p50": float(o["value"].median()),
        "value_p90": float(o["value"].quantile(0.9)),
        "value_mean": float(o["value"].mean()),
    }])


def repeat_purchase_interval(store: str) -> pl.DataFrame:
    """동일 (user, product) 재구매 간격 — 규칙적이면 재발주 패턴."""
    p = (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .select("user_id", "product_id", "event_time")
        .unique()
        .sort("user_id", "product_id", "event_time")
        .with_columns(
            pl.col("event_time").diff().over("user_id", "product_id")
            .dt.total_seconds().alias("gap_s")
        )
        .filter(pl.col("gap_s").is_not_null() & (pl.col("gap_s") > 0))
        .collect()
    )
    if not p.height:
        return pl.DataFrame([{"repeats": 0}])
    d = p["gap_s"] / 86400
    return pl.DataFrame([{
        "repeats": p.height,
        "days_p25": float(d.quantile(0.25)),
        "days_p50": float(d.median()),
        "days_p75": float(d.quantile(0.75)),
        "within_30d_share": float((d <= 30).mean()),
    }])


def hour_dow_profile(store: str) -> pl.DataFrame:
    """구매의 요일×시간대 분포(모스크바). 평일 업무시간 집중 → B2B."""
    return (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .group_by("dow_msk", "hour_msk")
        .agg(pl.len().alias("n"))
        .sort("dow_msk", "hour_msk")
        .collect()
    )


def business_hours_share(store: str) -> pl.DataFrame:
    """평일(월~금) 09~19시 구매 비중 vs 무작위 기대치."""
    ev = scan_events(store).filter(pl.col("event_type") == "purchase")
    r = ev.select(
        pl.len().alias("n"),
        ((pl.col("dow_msk") <= 5) & pl.col("hour_msk").is_between(9, 18)).sum().alias("biz"),
        (pl.col("dow_msk") <= 5).sum().alias("weekday"),
    ).collect().row(0, named=True)
    return pl.DataFrame([{
        "purchases": r["n"],
        "business_hours_share": r["biz"] / r["n"],
        "weekday_share": r["weekday"] / r["n"],
        # 균등분포 기대치: 평일 5/7, 업무시간 10/24
        "expected_business": (5 / 7) * (10 / 24),
        "expected_weekday": 5 / 7,
    }])


def cohort_retention(store: str, months: int = 5) -> pl.DataFrame:
    """첫 구매 월 기준 코호트의 이후 월 재구매 잔존율."""
    p = (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .with_columns(pl.col("event_time").dt.convert_time_zone("Europe/Moscow")
                      .dt.truncate("1mo").alias("m"))
        .group_by("user_id", "m").agg(pl.len().alias("n"))
        .collect()
    )
    first = p.group_by("user_id").agg(pl.col("m").min().alias("m0"))
    p = p.join(first, on="user_id")
    p = p.with_columns(
        ((pl.col("m").dt.year() - pl.col("m0").dt.year()) * 12
         + (pl.col("m").dt.month() - pl.col("m0").dt.month())).alias("k")
    )
    size = p.group_by("m0").agg(pl.col("user_id").n_unique().alias("cohort"))
    ret = (
        p.group_by("m0", "k").agg(pl.col("user_id").n_unique().alias("active"))
        .join(size, on="m0")
        .with_columns((pl.col("active") / pl.col("cohort")).alias("rate"))
        .sort("m0", "k")
    )
    return ret.filter(pl.col("k") < months)


# ══════════════════════════════════════════════════════════════════════
# 1-C. 격차 분해
# ══════════════════════════════════════════════════════════════════════
def alignment_ladder(store: str) -> pl.DataFrame:
    """측정 인공물을 한 단계씩 제거하며 전환율이 어떻게 변하는지 추적.

    "6.6배 격차"의 얼마가 진짜 행동 차이이고 얼마가 정의 차이인지 가른다.
    """
    ev = scan_events(store)
    cl = scan_cart_lines(store)
    rows = []

    # (0) 원본 user_session, 세션 단위: cart 포함 세션 중 purchase 도 있는 비율
    s = (
        ev.filter(pl.col("user_session").is_not_null())
        .group_by("user_id", "user_session")
        .agg(
            (pl.col("event_type") == "cart").any().alias("c"),
            (pl.col("event_type") == "purchase").any().alias("p"),
        )
        .filter(pl.col("c"))
        .select(pl.len().alias("n"), pl.col("p").mean().alias("r"))
        .collect().row(0, named=True)
    )
    rows.append(("① 원본 세션 · 세션 단위", s["n"], s["r"]))

    # (1) 재세션화(30분), 세션 단위
    s = (
        ev.group_by("user_id", "session30")
        .agg(
            (pl.col("event_type") == "cart").any().alias("c"),
            (pl.col("event_type") == "purchase").any().alias("p"),
        )
        .filter(pl.col("c"))
        .select(pl.len().alias("n"), pl.col("p").mean().alias("r"))
        .collect().row(0, named=True)
    )
    rows.append(("② 30분 재세션화 · 세션 단위", s["n"], s["r"]))

    # (2) cart_line 단위 + 동일 세션 내 구매만 인정
    same = (
        ev.group_by("user_id", "session30", "product_id")
        .agg(
            (pl.col("event_type") == "cart").any().alias("c"),
            (pl.col("event_type") == "purchase").any().alias("p"),
        )
        .filter(pl.col("c"))
        .select(pl.len().alias("n"), pl.col("p").mean().alias("r"))
        .collect().row(0, named=True)
    )
    rows.append(("③ cart_line 단위 · 동일 세션 귀속", same["n"], same["r"]))

    # (3~) cart_line 단위 + 시간 윈도우 귀속
    n = cl.select(pl.len()).collect().item()
    for name in WINDOWS:
        r = cl.select(pl.col(f"y_{name}").mean()).collect().item()
        label = f"④ cart_line 단위 · {name} 윈도우 귀속"
        if name == DEFAULT_WINDOW:
            label += "  ★기본"
        rows.append((label, n, r))

    return pl.DataFrame(rows, schema=["stage", "denominator", "cvr"], orient="row")


def price_common_support(bins: int = 40) -> pl.DataFrame:
    """두 스토어의 cart 가격 분포 겹침 정도.

    겹치는 구간이 없으면 '가격 구성 차이로 격차를 설명한다'는 분해는
    외삽(extrapolation)이 되어 성립하지 않는다. 분해 전에 반드시 확인.
    """
    out = []
    for store in ("cosmetics", "electronics"):
        s = (
            scan_cart_lines(store)
            .filter(pl.col("price").is_not_null())
            .select("price")
            .collect()["price"]
        )
        out.append({
            "store": store,
            "p01": float(s.quantile(0.01)), "p25": float(s.quantile(0.25)),
            "p50": float(s.median()), "p75": float(s.quantile(0.75)),
            "p99": float(s.quantile(0.99)),
        })
    return pl.DataFrame(out)


def overlap_region_comparison(lo: float, hi: float) -> pl.DataFrame:
    """공통 지지 구간(lo~hi)으로 제한한 뒤 두 스토어 전환율 비교."""
    out = []
    for store in ("cosmetics", "electronics"):
        r = (
            scan_cart_lines(store)
            .filter(pl.col("price").is_between(lo, hi))
            .select(pl.len().alias("n"), pl.col(Y).mean().alias("cvr"))
            .collect().row(0, named=True)
        )
        out.append({"store": store, **r})
    return pl.DataFrame(out)


def time_to_purchase_curve(store: str) -> pl.DataFrame:
    """cart 이후 경과 시간별 누적 전환율 — 리마인더 최적 시점의 근거."""
    cl = scan_cart_lines(store).select("tt_purchase_s").collect()
    n = cl.height
    marks = [1, 3, 6, 12, 24, 48, 72, 24 * 7, 24 * 14, 24 * 30]
    rows = []
    for h in marks:
        c = (cl["tt_purchase_s"] <= h * 3600).sum()
        rows.append({"hours": h, "cum_conv": int(c), "cum_rate": c / n})
    return pl.DataFrame(rows)


def with_prior_purchases(store: str) -> pl.DataFrame:
    """cart_line 에 **cart 시점 t 이전** 누적 구매 라인 수를 붙인다.

    §1.3 누수 금지 규칙의 적용. 패널 전체 구매 수를 쓰면 cart 이후의 구매까지
    세어버려, "미구매 유저의 전환율은 0" 같은 **동어반복**이 나온다.
    backward as-of 조인으로 t 직전까지의 누적 구매만 집계한다.
    """
    pur = (
        scan_events(store)
        .filter(pl.col("event_type") == "purchase")
        .select("user_id", "event_time")
        .sort("user_id", "event_time")
        .with_columns((pl.col("event_time").cum_count().over("user_id")).alias("prior_purchases"))
        .select("user_id", pl.col("event_time").alias("tp"), "prior_purchases")
        .sort("tp")
        .collect()
    )
    cl = scan_cart_lines(store).sort("t").collect()
    out = cl.join_asof(pur, left_on="t", right_on="tp", by="user_id", strategy="backward")
    return out.with_columns(pl.col("prior_purchases").fill_null(0))


def conversion_pockets(store: str, min_lines: int = 5000) -> pl.DataFrame:
    """Cosmetics 내부에서 Electronics 수준(50%+) 전환을 내는 세그먼트 탐색.

    외부 벤치마크보다 강한 증거다 — 동일 업태·동일 플랫폼이므로 이식 가능성이
    비교할 수 없이 높다. 모든 분할 변수는 **cart 시점에 관측 가능한 것만** 쓴다.
    """
    cl = with_prior_purchases(store)

    tiers = (
        pl.when(pl.col("prior_purchases") == 0).then(pl.lit("0. 구매이력 없음"))
        .when(pl.col("prior_purchases") <= 5).then(pl.lit("1. 1-5"))
        .when(pl.col("prior_purchases") <= 20).then(pl.lit("2. 6-20"))
        .when(pl.col("prior_purchases") <= 100).then(pl.lit("3. 21-100"))
        .otherwise(pl.lit("4. 100+"))
    )

    return (
        cl.with_columns(
            tiers.alias("prior_tier"),
            (pl.col("cart_repeat_cnt") > 1).alias("multi_qty"),
        )
        .group_by("prior_tier", "multi_qty")
        .agg(
            pl.len().alias("cart_lines"),
            pl.col(Y).mean().alias("cvr"),
            (pl.len() * (1 - pl.col(Y).mean())).round(0).alias("headroom"),
        )
        .filter(pl.col("cart_lines") >= min_lines)
        .sort("cvr", descending=True)
    )


def direct_to_cart(store: str) -> pl.DataFrame:
    """같은 세션에 선행 view 가 있었는가(검토형) vs 없었는가(리스트 직행형).

    cart 시점에 관측 가능하므로 누수가 없다.
    """
    ev = scan_events(store)
    viewed = (
        ev.filter(pl.col("event_type") == "view")
        .group_by("user_id", "session30", "product_id")
        .agg(pl.col("event_time").min().alias("t_view"))
    )
    cl = scan_cart_lines(store).join(
        viewed, on=["user_id", "session30", "product_id"], how="left"
    )
    return (
        cl.with_columns((pl.col("t_view") <= pl.col("t")).fill_null(False).alias("viewed_first"))
        .group_by("viewed_first")
        .agg(
            pl.len().alias("cart_lines"),
            pl.col(Y).mean().alias("cvr"),
            (pl.len() * (1 - pl.col(Y).mean())).round(0).alias("headroom"),
        )
        .sort("viewed_first")
        .collect()
    )
