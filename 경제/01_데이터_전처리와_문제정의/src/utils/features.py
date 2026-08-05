"""Phase 3 — 누수 없는 피처 엔지니어링.

누수 방지 설계 (§1.3)
---------------------
모든 이력 피처는 **cart 시각 t 보다 엄격히 이전**의 이벤트만으로 계산한다.
이를 관례가 아니라 **구조**로 강제하기 위해 두 장치를 쓴다.

1. **동일 시각 붕괴(collapse)**: 원본 `event_time` 해상도가 1초라 같은 초에
   여러 이벤트가 찍힌다. 정렬 후 누적합을 쓰면 *같은 초* 이벤트가 정렬 순서에
   따라 우연히 포함되어 버린다. 그래서 먼저 `(key, event_time)` 으로 집계해
   **시각당 한 행**으로 만든 뒤 누적합을 취한다.
2. **t - 1µs 로 as-of 조인**: `strategy="backward"` 는 동률을 포함하므로,
   t 에서 1마이크로초를 빼 조인한다. 이러면 t 시각 이벤트(= 지금 이 cart 자신
   포함)가 확실히 배제된다.

같은 초에 일어난 구매를 "이전"으로 볼지 모호한데, **보수적으로 배제**한다.
1초 해상도에서 동시각 이벤트의 선후는 확정할 수 없기 때문이다.

이 성질은 :mod:`src.test_leakage` 가 무작위 표본을 원시 이벤트로 재계산해
검증한다.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.utils.config import DATA_PROC
from src.utils.sessions import scan_cart_lines, scan_events

#: as-of 조인 시 t 시각 이벤트를 배제하기 위한 간격.
EPS = timedelta(microseconds=1)


def features_path(store: str):
    return DATA_PROC / f"{store}_features.parquet"


def _is(et: str) -> pl.Expr:
    return (pl.col("event_type") == et).cast(pl.Int32)


def _collapsed_timeline(ev: pl.LazyFrame, keys: list[str], aggs: dict) -> pl.DataFrame:
    """`keys` 별 시각당 1행 타임라인 + 누적합.

    같은 시각 이벤트를 먼저 합쳐야 누적합이 '시각 경계' 에서 깔끔하게 끊긴다.
    """
    tl = (
        ev.group_by(*keys, "event_time")
        .agg(**aggs)
        .sort(*keys, "event_time")
        .with_columns([pl.col(c).cum_sum().over(keys).alias(c) for c in aggs])
        .collect()
    )
    return tl.sort("event_time")


def _asof(cl: pl.DataFrame, tl: pl.DataFrame, by: list[str], cols: list[str],
          fill: float | None = 0) -> pl.DataFrame:
    """t 직전(엄격히 이전) 시점의 누적 상태를 붙인다."""
    out = cl.with_columns((pl.col("t") - EPS).alias("_t")).sort("_t").join_asof(
        tl.select(*by, "event_time", *cols),
        left_on="_t", right_on="event_time", by=by, strategy="backward",
    )
    # join_asof 는 우측 키(event_time)를 남긴다. 그대로 두면 다음 _asof 호출에서
    # 'event_time_right' 가 중복 생성돼 실패하므로 매번 정리한다.
    out = out.drop([c for c in ("_t", "event_time", "event_time_right")
                    if c in out.columns])
    if fill is not None:
        out = out.with_columns([pl.col(c).fill_null(fill) for c in cols])
    return out


def build_features(store: str, *, overwrite: bool = False):
    out = features_path(store)
    if out.exists() and not overwrite:
        return out

    ev = scan_events(store)
    cl = scan_cart_lines(store).collect()

    # ── A. 유저 이력 ────────────────────────────────────────────────
    u_tl = _collapsed_timeline(ev, ["user_id"], {
        "u_views": _is("view").sum(),
        "u_carts": _is("cart").sum(),
        "u_purchases": _is("purchase").sum(),
        "u_removes": _is("remove_from_cart").sum(),
        "u_spend": (pl.col("price") * _is("purchase")).sum(),
    })
    ucols = ["u_views", "u_carts", "u_purchases", "u_removes", "u_spend"]
    cl = _asof(cl, u_tl, ["user_id"], ucols)

    # 마지막 활동 시각 / 첫 활동 시각 (recency, tenure)
    u_times = _collapsed_timeline(ev, ["user_id"], {"_n": pl.len()})
    u_times = u_times.select("user_id", "event_time",
                             pl.col("event_time").alias("u_last_seen"))
    cl = _asof(cl, u_times, ["user_id"], ["u_last_seen"], fill=None)

    u_first = ev.group_by("user_id").agg(
        pl.col("event_time").min().alias("u_first_seen")).collect()
    cl = cl.join(u_first, on="user_id", how="left")

    # 마지막 구매 시각
    p_tl = _collapsed_timeline(
        ev.filter(pl.col("event_type") == "purchase"), ["user_id"], {"_n": pl.len()})
    p_tl = p_tl.select("user_id", "event_time",
                       pl.col("event_time").alias("u_last_purchase"))
    cl = _asof(cl, p_tl, ["user_id"], ["u_last_purchase"], fill=None)

    # 구매가 포함된 세션 수 → 주문당 라인 수(대량구매 성향, §0.8-F 의 행동 축)
    ord_tl = _collapsed_timeline(
        ev.filter(pl.col("event_type") == "purchase")
          .group_by("user_id", "session30")
          .agg(pl.col("event_time").min().alias("event_time")),
        ["user_id"], {"u_orders": pl.len()})
    cl = _asof(cl, ord_tl, ["user_id"], ["u_orders"])

    # ── B. 유저 × 상품 이력 ─────────────────────────────────────────
    up_tl = _collapsed_timeline(ev, ["user_id", "product_id"], {
        "up_carts": _is("cart").sum(),
        "up_purchases": _is("purchase").sum(),
        "up_removes": _is("remove_from_cart").sum(),
    })
    cl = _asof(cl, up_tl, ["user_id", "product_id"],
               ["up_carts", "up_purchases", "up_removes"])

    # ── C. 상품 이력 ────────────────────────────────────────────────
    p2_tl = _collapsed_timeline(ev, ["product_id"], {
        "p_views": _is("view").sum(),
        "p_carts": _is("cart").sum(),
        "p_purchases": _is("purchase").sum(),
        "p_price_sum": pl.col("price").fill_null(0).sum(),
        "p_price_cnt": pl.col("price").is_not_null().cast(pl.Int32).sum(),
    })
    cl = _asof(cl, p2_tl, ["product_id"],
               ["p_views", "p_carts", "p_purchases", "p_price_sum", "p_price_cnt"])

    # ── D. 세션 맥락 ────────────────────────────────────────────────
    s_tl = _collapsed_timeline(ev, ["user_id", "session30"], {"s_depth": pl.len()})
    cl = _asof(cl, s_tl, ["user_id", "session30"], ["s_depth"])

    s_start = ev.group_by("user_id", "session30").agg(
        pl.col("event_time").min().alias("s_start")).collect()
    cl = cl.join(s_start, on=["user_id", "session30"], how="left")

    # ── E. 파생 ────────────────────────────────────────────────────
    day = 86400.0
    cl = cl.with_columns(
        # 유저
        ((pl.col("t") - pl.col("u_first_seen")).dt.total_seconds() / day)
        .alias("u_tenure_d"),
        ((pl.col("t") - pl.col("u_last_seen")).dt.total_seconds() / day)
        .alias("u_recency_d"),
        ((pl.col("t") - pl.col("u_last_purchase")).dt.total_seconds() / day)
        .alias("u_recency_purchase_d"),
        (pl.col("u_purchases") == 0).alias("u_never_purchased"),
        (pl.col("u_views") + pl.col("u_carts") + pl.col("u_purchases")
         + pl.col("u_removes") == 0).alias("u_is_first_event"),
        # 라플라스 평활 — 분모가 0~2인 유저가 많아 생평균은 0/1 로 튄다
        ((pl.col("u_purchases") + 1) / (pl.col("u_carts") + 5)).alias("u_prior_cvr"),
        ((pl.col("u_removes") + 1) / (pl.col("u_carts") + 5)).alias("u_prior_remove_rate"),
        (pl.col("u_purchases") / pl.col("u_orders").clip(1)).alias("u_lines_per_order"),
        (pl.col("u_spend") / pl.col("u_purchases").clip(1)).alias("u_avg_line_value"),
        # 유저 × 상품
        (pl.col("up_purchases") > 0).alias("up_bought_before"),
        (pl.col("up_carts") > 0).alias("up_carted_before"),
        # 상품
        ((pl.col("p_purchases") + 1) / (pl.col("p_carts") + 5)).alias("p_prior_cvr"),
        (pl.col("p_price_sum") / pl.col("p_price_cnt").clip(1)).alias("p_prior_price_mean"),
        # 세션 맥락
        ((pl.col("t") - pl.col("s_start")).dt.total_seconds()).alias("s_elapsed_s"),
        # 시간
        pl.col("t").dt.convert_time_zone("Europe/Moscow").dt.hour().alias("hour_msk"),
        pl.col("t").dt.convert_time_zone("Europe/Moscow").dt.weekday().alias("dow_msk"),
        pl.col("t").dt.convert_time_zone("Europe/Moscow").dt.month().alias("month"),
    )
    cl = cl.with_columns(
        (pl.col("dow_msk") >= 6).alias("is_weekend"),
        # 현재 가격이 이 상품의 과거 평균 대비 얼마나 싼가 = 관측 가능한 '할인'
        (pl.col("price") / pl.col("p_prior_price_mean")).alias("price_vs_prior"),
    )

    cl.write_parquet(out, compression="zstd")
    return out


def scan_features(store: str) -> pl.LazyFrame:
    return pl.scan_parquet(features_path(store))


def univariate_auc(store: str, cols: list[str], target: str = "y_7d") -> pl.DataFrame:
    """피처 하나만으로 타겟을 얼마나 가르는가 (단변량 AUC).

    순위 기반이라 단조변환에 불변이고 스케일에 영향받지 않는다. 0.5 는 무신호,
    0.5 에서 멀수록 신호가 강하다(0.5 미만이면 음의 방향).

    결측은 해당 피처 계산에서만 제외하고 커버리지를 함께 보고한다 — 결측을
    0 으로 메우면 없는 신호가 만들어진다.
    """
    df = scan_features(store).select(cols + [target]).collect()
    y = df[target].cast(pl.Int8)
    rows = []
    for c in cols:
        s = df[c]
        if s.dtype == pl.Boolean:
            s = s.cast(pl.Int8)
        mask = s.is_not_null()
        sv, yv = s.filter(mask), y.filter(mask)
        n_pos = int(yv.sum())
        n_neg = int(len(yv) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        # 동점은 평균 순위로 처리해야 AUC 가 왜곡되지 않는다
        r = sv.rank("average")
        auc = (float(r.filter(yv == 1).sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        rows.append({
            "feature": c,
            "auc": auc,
            "abs_signal": abs(auc - 0.5),
            "coverage": float(mask.mean()),
        })
    return pl.DataFrame(rows).sort("abs_signal", descending=True)


def redundancy(store: str, cols: list[str], thresh: float = 0.9) -> pl.DataFrame:
    """상관이 매우 높은 피처 쌍 — 해석 시 서로의 중요도를 갉아먹는다."""
    df = scan_features(store).select(cols).collect()
    num = [c for c in cols if df[c].dtype in
           (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32, pl.Boolean)]
    d = df.select([pl.col(c).cast(pl.Float64) for c in num])
    corr = d.corr()
    rows = []
    for i, a in enumerate(num):
        for j, b in enumerate(num):
            if j <= i:
                continue
            v = corr[a][j]
            if v is not None and abs(v) >= thresh:
                rows.append({"a": a, "b": b, "corr": float(v)})
    return pl.DataFrame(rows, schema={"a": pl.String, "b": pl.String, "corr": pl.Float64}
                        ).sort("corr", descending=True)


#: 모델 입력 피처 (cart 시점 관측 가능한 것만).
FEATURE_COLS = [
    # 담기 행동 — Phase 1 에서 가장 강한 신호로 확인됨(§0.8-G/H)
    "cart_repeat_cnt", "viewed_first",
    # 오퍼
    "price", "price_missing", "price_vs_prior",
    # 유저 이력
    "u_views", "u_carts", "u_purchases", "u_removes", "u_spend",
    "u_tenure_d", "u_recency_d", "u_recency_purchase_d",
    "u_never_purchased", "u_prior_cvr", "u_prior_remove_rate",
    "u_lines_per_order", "u_avg_line_value", "u_orders",
    # 유저 × 상품
    "up_carts", "up_purchases", "up_removes", "up_bought_before", "up_carted_before",
    # 상품
    "p_views", "p_carts", "p_purchases", "p_prior_cvr", "p_prior_price_mean",
    # 맥락
    "s_depth", "s_elapsed_s", "hour_msk", "dow_msk", "month", "is_weekend",
]
