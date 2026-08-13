"""세션 재구성과 `cart_line` 테이블 생성.

Phase 0-B 에서 원본 ``user_session`` 이 브라우징 세션이 아님이 확인되었다
(길이 p99.9 = 71일). 따라서 **30분 무활동 기준 재세션화를 기본 정의로 삼고**,
원본 세션은 Phase 7 강건성 대조군으로만 남긴다.

``cart_line`` 은 이 프로젝트의 분석 원자(atom)다::
 
    cart_line = (user_id, product_id, 세션 내 최초 cart 시각 t)

Phase 0-A 에서 완전중복 cart 행이 **수량 단위 로깅**임이 확인되었으므로,
중복은 버리지 않고 ``cart_repeat_cnt`` 로 보존한다(주문 수량 대리변수).

한 유저이력 보존 Hash
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.utils.config import DATA_PROC
from src.utils.loader import clean, scan

#: 재세션화 무활동 임계값(업계 표준 30분).
SESSION_GAP = timedelta(minutes=30)

#: 귀속 윈도우 후보. 7일이 기본, 나머지는 민감도 분석용(§Phase 2-B).
WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
DEFAULT_WINDOW = "7d"


def events_path(store: str):
    return DATA_PROC / f"{store}_events.parquet"


def cart_lines_path(store: str):
    return DATA_PROC / f"{store}_cart_lines.parquet"


# ── 1. 재세션화 ────────────────────────────────────────────────────────
def build_events(store: str, *, overwrite: bool = False):
    """30분 무활동 기준 ``session30`` 을 붙인 이벤트 테이블 생성."""
    out = events_path(store)
    if out.exists() and not overwrite:
        return out

    df = clean(scan(store)).sort("user_id", "event_time").collect()

    gap = pl.col("event_time").diff().over("user_id")
    # 유저의 첫 이벤트는 gap 이 null → 항상 새 세션.
    is_new = (gap.is_null() | (gap > SESSION_GAP)).cast(pl.Int32)
    # user_id 로 정렬돼 있으므로 전역 cum_sum 이 곧 전역 고유 세션 ID 가 된다.
    df = df.with_columns(is_new.cum_sum().alias("session30"))

    df.write_parquet(out, compression="zstd")
    return out


def scan_events(store: str) -> pl.LazyFrame:
    return pl.scan_parquet(events_path(store))


# ── 2. cart_line ───────────────────────────────────────────────────────
def build_cart_lines(store: str, *, overwrite: bool = False):
    """cart_line 테이블 생성 + 구매 귀속.

    귀속 방식
    ---------
    Phase 0/사전조사에서 cosmetics 구매의 48.6% 가 **동일 세션에 cart 기록이
    없음**을 확인했다. 따라서 세션 단위가 아니라 ``(user_id, product_id)``
    단위로, cart 시각 t 이후 **가장 이른 구매**를 forward as-of 조인으로 찾고
    경과 시간을 잰다. 윈도우별 타겟은 이 경과 시간에서 파생한다.
    """
    out = cart_lines_path(store)
    if out.exists() and not overwrite:
        return out

    ev = scan_events(store)

    # (a) cart 이벤트 → (user, session30, product) 단위로 collapse
    carts = (
        ev.filter(pl.col("event_type") == "cart")
        .group_by("user_id", "session30", "product_id")
        .agg(
            pl.col("event_time").min().alias("t"),
            pl.len().alias("cart_repeat_cnt"),
            pl.col("price").first().alias("price"),
            pl.col("price_missing").first().alias("price_missing"),
            pl.col("brand").first().alias("brand"),
            pl.col("category_id").first().alias("category_id"),
        )
        .sort("t")
        .collect()
    )

    # (b) 구매 이벤트
    purch = (
        ev.filter(pl.col("event_type") == "purchase")
        .select("user_id", "product_id", pl.col("event_time").alias("t_purchase"))
        .unique()
        .sort("t_purchase")
        .collect()
    )

    # (c) forward as-of: cart 시각 이후 첫 구매
    cl = carts.join_asof(
        purch,
        left_on="t",
        right_on="t_purchase",
        by=["user_id", "product_id"],
        strategy="forward",
    )

    # (d) 명시적 장바구니 제거: 같은 (user, session30, product) 에서 cart 이후 remove
    removes = (
        ev.filter(pl.col("event_type") == "remove_from_cart")
        .group_by("user_id", "session30", "product_id")
        .agg(pl.col("event_time").max().alias("t_remove"))
        .collect()
    )
    if removes.height:
        cl = cl.join(removes, on=["user_id", "session30", "product_id"], how="left")
    else:
        cl = cl.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("t_remove"))

    # (e) 같은 세션에서 담기보다 **먼저** 조회했는가 (검토형 vs 리스트 직행형)
    #     세션 내 조회 유무(선후 무관)와는 다른 지표다 — 인과적으로 의미 있는 쪽은
    #     '담기 이전' 이므로 이것을 정식 피처로 삼는다.
    viewed = (
        ev.filter(pl.col("event_type") == "view")
        .group_by("user_id", "session30", "product_id")
        .agg(pl.col("event_time").min().alias("t_view"))
        .collect()
    )
    cl = cl.join(viewed, on=["user_id", "session30", "product_id"], how="left")

    # (f) 타겟 파생
    cl = cl.with_columns(
        (pl.col("t_purchase") - pl.col("t")).dt.total_seconds().alias("tt_purchase_s"),
        (pl.col("t_remove") > pl.col("t")).fill_null(False).alias("removed_after_cart"),
        (pl.col("t_view") <= pl.col("t")).fill_null(False).alias("viewed_first"),
    )
    cl = cl.with_columns(
        [
            (pl.col("tt_purchase_s") <= w.total_seconds())
            .fill_null(False)
            .alias(f"y_{name}")
            for name, w in WINDOWS.items()
        ]
    )

    # (f) 우측 절단: 패널 종료까지 남은 시간이 윈도우보다 짧으면 관측 부족
    t_end = ev.select(pl.col("event_time").max()).collect().item()
    cl = cl.with_columns(
        (pl.lit(t_end) - pl.col("t")).dt.total_seconds().alias("obs_horizon_s")
    )
    cl = cl.with_columns(
        [
            (pl.col("obs_horizon_s") < w.total_seconds()).alias(f"censored_{name}")
            for name, w in WINDOWS.items()
        ]
    )

    # (h) 3-way 결과 라벨 (Phase 2)
    #     "명시적 제거" 와 "조용한 이탈" 은 원인이 다르므로 반드시 구분한다 —
    #     전자는 재고려·가격불만, 후자는 이탈·망각이며 필요한 액션도 다르다.
    #     제거 후 재구매하는 경우가 있으므로 구매를 최우선으로 판정한다.
    ycol = f"y_{DEFAULT_WINDOW}"
    cl = cl.with_columns(
        pl.when(pl.col(ycol)).then(pl.lit("purchased"))
        .when(pl.col("removed_after_cart")).then(pl.lit("explicitly_removed"))
        .otherwise(pl.lit("silently_abandoned"))
        .alias("outcome")
    )

    cl.write_parquet(out, compression="zstd")
    return out


def scan_cart_lines(store: str) -> pl.LazyFrame:
    return pl.scan_parquet(cart_lines_path(store))
