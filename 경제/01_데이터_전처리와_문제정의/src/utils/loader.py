"""원본 CSV → Parquet 변환 및 로딩.

설계 원칙
---------
Phase 0 이 만드는 ``data/processed/{store}.parquet`` 는 **원본의 충실한 미러**다.
타입 지정 · 타임존 처리 · 파생 시간 컬럼만 추가하고, **행은 하나도 버리지 않는다.**

행 필터링(음수 가격, 완전중복 등)을 Phase 0 에서 하지 않는 이유:

* **완전중복 행**: event_time 의 해상도가 1초라, 같은 초에 발생한 장바구니
  수량 증가는 원본에서 완전히 동일한 행으로 기록된다. 이를 Phase 0 에서
  제거하면 Phase 2 의 ``cart_repeat_cnt`` 신호가 파괴된다. 중복의 의미
  판정은 cart_line 을 정의하는 Phase 2 의 몫이다.
* **음수/0 가격**: 소수(0.0005%)지만 제거 여부는 분석 목적에 따라 다르다.

대신 아래 :func:`clean` 을 downstream 공용 필터로 제공한다.
"""
from __future__ import annotations

import polars as pl

from src.utils.config import (
    COSMETICS_CSVS,
    ELECTRONICS_CSV,
    EVENT_TIME_FORMAT,
    EVENT_TYPE_ENUM,
    RAW_SCHEMA,
    TZ_LOCAL,
    parquet_path,
)


def _source_paths(store: str) -> list:
    if store == "cosmetics":
        return list(COSMETICS_CSVS)
    if store == "electronics":
        return [ELECTRONICS_CSV]
    raise ValueError(f"unknown store: {store}")


def scan_raw(store: str) -> pl.LazyFrame:
    """원본 CSV 를 타입 지정하여 lazy 로 읽고, 시간 컬럼을 정규화한다."""
    lf = pl.scan_csv(
        _source_paths(store),
        schema_overrides=RAW_SCHEMA,
        # 빈 문자열은 결측으로 취급(brand/category_code 가 다수 해당).
        null_values=[""],
        low_memory=True,
    )

    event_time_utc = (
        pl.col("event_time")
        .str.strip_suffix(" UTC")
        .str.to_datetime(EVENT_TIME_FORMAT, time_unit="us")
        .dt.replace_time_zone("UTC")
    )
    local = event_time_utc.dt.convert_time_zone(TZ_LOCAL)

    return lf.with_columns(
        event_time_utc.alias("event_time"),
        pl.col("event_type").cast(EVENT_TYPE_ENUM),
        # 모스크바 로컬 기준 파생 시간 필드. 시간대/요일 해석은 반드시 로컬로
        # 해야 의미가 있다(§Phase 0-4).
        local.dt.date().alias("date_msk"),
        local.dt.hour().cast(pl.Int8).alias("hour_msk"),
        local.dt.weekday().cast(pl.Int8).alias("dow_msk"),  # 1=월 … 7=일
    )


def convert_to_parquet(store: str, *, overwrite: bool = False) -> "object":
    """CSV → Parquet 스트리밍 변환. 이미 있으면 건너뛴다."""
    out = parquet_path(store)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    scan_raw(store).sink_parquet(out, compression="zstd")
    return out


def scan(store: str) -> pl.LazyFrame:
    """변환된 Parquet 를 lazy 로 읽는다(분석 진입점)."""
    return pl.scan_parquet(parquet_path(store))


def clean(
    lf: pl.LazyFrame,
    *,
    null_invalid_price: bool = True,
    drop_null_session: bool = False,
) -> pl.LazyFrame:
    """downstream 공용 정제. 필터 기준을 한 곳에 모아 Phase 간 불일치를 막는다.

    가격 처리 — **행 삭제가 아니라 결측 처리**
    ------------------------------------------
    price<=0 은 cosmetics 에 104,288 건(0.50%) 있고 전체 상품의 40.3% 가
    한 번쯤 겪는다. 다만 '항상 0원'인 상품은 667 개뿐이라, 이는 상품 자체의
    문제가 아니라 **가격 미설정 기간**으로 보인다.

    행을 지우면 해당 cart_line 자체가 사라져 타겟 y 관측을 잃는다. 반면 가격만
    결측 처리하면 y 는 보존되고, LightGBM 은 결측을 그대로 다룰 수 있으며,
    무엇보다 ``price_missing`` 자체가 **"가격이 안 붙은 상품은 전환이 낮다"**
    는 가설을 검정할 수 있는 피처가 된다. 정보량 기준으로 결측 처리가 우월하다.

    세션 키
    -------
    ``user_session`` 단독은 키로 쓰면 안 된다 — 272 개 세션이 복수 user_id 에
    걸쳐 있다(최대 8명). 세션 단위 연산은 반드시 ``["user_id", "user_session"]``
    복합 키를 쓴다. :func:`session_key` 참조.
    """
    if null_invalid_price:
        lf = lf.with_columns(
            (pl.col("price") <= 0).alias("price_missing"),
            pl.when(pl.col("price") > 0).then(pl.col("price")).alias("price"),
        )
    if drop_null_session:
        lf = lf.filter(pl.col("user_session").is_not_null())
    return lf


#: 세션 단위 group_by 에 쓸 복합 키(§clean 독스트링 참조).
SESSION_KEY = ["user_id", "user_session"]
