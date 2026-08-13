"""Phase 0 품질 게이트: 변환된 Parquet 을 검사하고 마크다운 리포트를 만든다."""
from __future__ import annotations

import polars as pl

from src.utils.config import EVENT_TYPES, TZ_LOCAL
from src.utils.loader import scan


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.3%}" if whole else "—"


def profile(store: str) -> dict:
    """스토어 하나에 대한 품질 지표 일괄 산출."""
    lf = scan(store)
    cols = lf.collect_schema().names()

    n_rows = lf.select(pl.len()).collect().item()

    nulls = (
        lf.select([pl.col(c).null_count().alias(c) for c in cols]).collect().row(0, named=True)
    )

    dtypes = {c: str(dt) for c, dt in zip(cols, lf.collect_schema().dtypes())}

    events = (
        lf.group_by("event_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )

    # 가격 이상치
    price = lf.select(
        pl.col("price").lt(0).sum().alias("neg"),
        pl.col("price").eq(0).sum().alias("zero"),
        pl.col("price").min().alias("min"),
        pl.col("price").quantile(0.25).alias("p25"),
        pl.col("price").median().alias("p50"),
        pl.col("price").quantile(0.75).alias("p75"),
        pl.col("price").quantile(0.99).alias("p99"),
        pl.col("price").max().alias("max"),
        pl.col("price").null_count().alias("null"),
    ).collect().row(0, named=True)

    # 기간 · 월별 분포
    span = lf.select(
        pl.col("event_time").min().alias("start"),
        pl.col("event_time").max().alias("end"),
    ).collect().row(0, named=True)

    monthly = (
        lf.group_by(pl.col("date_msk").dt.truncate("1mo").alias("month"))
        .agg(pl.len().alias("n"))
        .sort("month")
        .collect()
    )

    # 카디널리티
    card = lf.select(
        pl.col("user_id").n_unique().alias("users"),
        pl.col("product_id").n_unique().alias("products"),
        pl.col("user_session").n_unique().alias("sessions"),
        pl.col("brand").n_unique().alias("brands"),
        pl.col("category_id").n_unique().alias("category_ids"),
        pl.col("category_code").n_unique().alias("category_codes"),
    ).collect().row(0, named=True)

    # 완전중복 행(제거하지 않고 계측만 한다 — loader 독스트링 참조)
    n_distinct = lf.unique().select(pl.len()).collect().item()
    dup_rows = n_rows - n_distinct

    # 세션 무결성: 한 세션이 여러 user_id 에 걸치는가?
    # (통계는 null 세션을 제외해야 의미가 있다 — null 은 group_by 에서 하나의
    #  거대한 가짜 세션으로 뭉쳐 최대 길이·이벤트 수를 오염시킨다.)
    sess = (
        lf.filter(pl.col("user_session").is_not_null())
        .group_by("user_session")
        .agg(
            pl.col("user_id").n_unique().alias("n_users"),
            pl.col("event_time").min().alias("t0"),
            pl.col("event_time").max().alias("t1"),
            pl.len().alias("n_events"),
        )
        .collect()
    )
    multi_user_sessions = int(sess.filter(pl.col("n_users") > 1).height)

    # 월 경계를 넘는 세션(원본이 월별 파일로 쪼개져 있어 확인이 필요)
    cross_month = int(
        sess.filter(
            pl.col("t0").dt.convert_time_zone(TZ_LOCAL).dt.month()
            != pl.col("t1").dt.convert_time_zone(TZ_LOCAL).dt.month()
        ).height
    )

    dur = (sess["t1"] - sess["t0"]).dt.total_seconds()
    sess_stats = {
        "n": int(sess.height),
        "events_p50": int(sess["n_events"].median()),
        "events_p90": int(sess["n_events"].quantile(0.9)),
        "events_max": int(sess["n_events"].max()),
        "single_event_share": float((sess["n_events"] == 1).mean()),
        "dur_p50_s": float(dur.median()),
        "dur_p90_s": float(dur.quantile(0.9)),
        "dur_p99_h": float(dur.quantile(0.99)) / 3600,
        "dur_p999_h": float(dur.quantile(0.999)) / 3600,
        "dur_max_h": float(dur.max()) / 3600,
        "over_30min_share": float((dur > 1800).mean()),
        "over_24h_share": float((dur > 86400).mean()),
    }

    # ── 이슈 진단 ────────────────────────────────────────────────────
    # 완전중복 행의 event_type 구성: 수량 단위 로깅인지 판별하는 근거
    dup_groups = lf.group_by(cols).agg(pl.len().alias("c")).filter(pl.col("c") > 1)
    dup_by_type = (
        dup_groups.group_by("event_type")
        .agg((pl.col("c") - 1).sum().alias("excess"))
        .sort("excess", descending=True)
        .collect()
    )
    dup_max_repeat = int(dup_groups.select(pl.col("c").max()).collect().item() or 0)

    # null user_session 의 event_type 구성
    null_sess = (
        lf.filter(pl.col("user_session").is_null())
        .group_by("event_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )
    n_null_sess = lf.filter(pl.col("user_session").is_null()).select(pl.len()).collect().item()

    # price<=0 의 event_type 구성 + 상품 확산도
    bad_price = (
        lf.filter(pl.col("price") <= 0)
        .group_by("event_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )
    prod_price = (
        lf.group_by("product_id")
        .agg(pl.len().alias("n"), (pl.col("price") <= 0).sum().alias("bad"))
        .collect()
    )
    price_diag = {
        "products_total": int(prod_price.height),
        "products_any_bad": int((prod_price["bad"] > 0).sum()),
        "products_all_bad": int((prod_price["bad"] == prod_price["n"]).sum()),
    }

    return {
        "dup_by_type": dup_by_type,
        "dup_max_repeat": dup_max_repeat,
        "null_sess": null_sess,
        "n_null_sess": n_null_sess,
        "bad_price": bad_price,
        "price_diag": price_diag,
        "store": store,
        "n_rows": n_rows,
        "dtypes": dtypes,
        "nulls": nulls,
        "events": events,
        "price": price,
        "span": span,
        "monthly": monthly,
        "card": card,
        "dup_rows": dup_rows,
        "multi_user_sessions": multi_user_sessions,
        "cross_month_sessions": cross_month,
        "sess": sess_stats,
    }


def render(profiles: list[dict]) -> str:
    """프로파일 목록을 마크다운 리포트로 렌더링."""
    L: list[str] = []
    A = L.append

    A("# Phase 0 — 데이터 품질 리포트")
    A("")
    A("> 자동 생성: `python -m src.run_phase0`")
    A("> 대상: `data/processed/{store}.parquet` (원본 충실 미러 — 행 필터링 없음)")
    A("")

    # ── 요약 ──
    A("## 1. 요약")
    A("")
    A("| 스토어 | 행 수 | 기간 (UTC) | 유저 | 상품 | 세션 | 브랜드 |")
    A("|---|---:|---|---:|---:|---:|---:|")
    for p in profiles:
        s, c = p["span"], p["card"]
        A(
            f"| **{p['store']}** | {_fmt(p['n_rows'])} | "
            f"{s['start']:%Y-%m-%d} ~ {s['end']:%Y-%m-%d} | "
            f"{_fmt(c['users'])} | {_fmt(c['products'])} | "
            f"{_fmt(c['sessions'])} | {_fmt(c['brands'])} |"
        )
    A("")

    # ── 이벤트 구성 ──
    A("## 2. 이벤트 구성")
    A("")
    A("| event_type | " + " | ".join(p["store"] for p in profiles) + " |")
    A("|---|" + "---:|" * len(profiles))
    for et in EVENT_TYPES:
        cells = []
        for p in profiles:
            row = p["events"].filter(pl.col("event_type") == et)
            if row.height == 0:
                cells.append("**없음**")
            else:
                n = int(row["n"][0])
                cells.append(f"{_fmt(n)} ({n / p['n_rows']:.1%})")
        A(f"| {et} | " + " | ".join(cells) + " |")
    A("")

    # ── 결측 ──
    A("## 3. 결측률")
    A("")
    cols = list(profiles[0]["nulls"].keys())
    A("| 컬럼 | dtype | " + " | ".join(p["store"] for p in profiles) + " |")
    A("|---|---|" + "---:|" * len(profiles))
    for c in cols:
        dt = profiles[0]["dtypes"][c]
        cells = [_pct(p["nulls"][c], p["n_rows"]) for p in profiles]
        A(f"| `{c}` | {dt} | " + " | ".join(cells) + " |")
    A("")

    # ── 가격 ──
    A("## 4. 가격 분포 및 이상치")
    A("")
    A("| 지표 | " + " | ".join(p["store"] for p in profiles) + " |")
    A("|---|" + "---:|" * len(profiles))
    for key, label in [
        ("min", "최소"), ("p25", "p25"), ("p50", "중앙값"),
        ("p75", "p75"), ("p99", "p99"), ("max", "최대"),
    ]:
        A(f"| {label} | " + " | ".join(f"${p['price'][key]:,.2f}" for p in profiles) + " |")
    for key, label in [("neg", "음수 가격"), ("zero", "0원 가격"), ("null", "결측 가격")]:
        cells = [f"{_fmt(int(p['price'][key]))} ({_pct(int(p['price'][key]), p['n_rows'])})"
                 for p in profiles]
        A(f"| **{label}** | " + " | ".join(cells) + " |")
    A("")

    # ── 세션 ──
    A("## 5. 세션 무결성")
    A("")
    A("| 지표 | " + " | ".join(p["store"] for p in profiles) + " |")
    A("|---|" + "---:|" * len(profiles))
    rows = [
        ("세션 수", lambda p: _fmt(p["sess"]["n"])),
        ("세션당 이벤트 중앙값", lambda p: _fmt(p["sess"]["events_p50"])),
        ("세션당 이벤트 p90", lambda p: _fmt(p["sess"]["events_p90"])),
        ("세션당 이벤트 최대", lambda p: _fmt(p["sess"]["events_max"])),
        ("단일 이벤트 세션 비중", lambda p: f"{p['sess']['single_event_share']:.1%}"),
        ("세션 길이 중앙값", lambda p: f"{p['sess']['dur_p50_s']:.0f}초"),
        ("세션 길이 p90", lambda p: f"{p['sess']['dur_p90_s'] / 60:.1f}분"),
        ("세션 길이 p99", lambda p: f"{p['sess']['dur_p99_h']:.1f}시간"),
        ("세션 길이 p99.9", lambda p: f"{p['sess']['dur_p999_h']:.0f}시간"),
        ("세션 길이 최대", lambda p: f"{p['sess']['dur_max_h']:.0f}시간"),
        ("⚠️ 30분 초과 세션 비중", lambda p: f"{p['sess']['over_30min_share']:.2%}"),
        ("⚠️ 24시간 초과 세션 비중", lambda p: f"{p['sess']['over_24h_share']:.2%}"),
        ("⚠️ 복수 user_id 세션", lambda p: _fmt(p["multi_user_sessions"])),
        ("⚠️ 월 경계 횡단 세션", lambda p: _fmt(p["cross_month_sessions"])),
        ("⚠️ 완전중복 행", lambda p: f"{_fmt(p['dup_rows'])} ({_pct(p['dup_rows'], p['n_rows'])})"),
    ]
    for label, fn in rows:
        A(f"| {label} | " + " | ".join(fn(p) for p in profiles) + " |")
    A("")
    A("> ⚠️ **`user_session` 은 브라우징 세션이 아니다.** cosmetics 는 p99.9 가 "
      "1,703시간(71일), electronics 는 p99 가 이미 144시간(6일)이다. 상당수가 "
      "세션이 아니라 **영속 쿠키/디바이스 ID** 처럼 동작한다. → §7-2 결정 참조.")
    A("")

    # ── 월별 ──
    A("## 6. 월별 이벤트 수 (모스크바 기준)")
    A("")
    for p in profiles:
        A(f"**{p['store']}**")
        A("")
        A("| 월 | 행 수 |")
        A("|---|---:|")
        for row in p["monthly"].iter_rows(named=True):
            A(f"| {row['month']:%Y-%m} | {_fmt(int(row['n']))} |")
        A("")
    A("> 마지막 월의 소량 행은 UTC→MSK(+3h) 변환으로 월말 자정 근처 이벤트가 "
      "다음 달로 넘어간 것이다. 정상.")
    A("")

    # ── 이슈 진단 및 결정 ──
    A("## 7. 이슈 진단 및 처리 결정")
    A("")

    A("### 7-1. 완전중복 행 = 수량 단위 로깅 (제거하지 않음)")
    A("")
    A("event_time 해상도가 1초라, 같은 초에 일어난 수량 변경이 완전히 동일한 "
      "행으로 남는다. 중복의 event_type 구성이 이를 뒷받침한다:")
    A("")
    A("| 스토어 | 초과 행의 event_type 구성 | 최대 반복 |")
    A("|---|---|---:|")
    for p in profiles:
        parts = [f"`{r['event_type']}` {_fmt(int(r['excess']))}"
                 for r in p["dup_by_type"].iter_rows(named=True)]
        A(f"| {p['store']} | " + ", ".join(parts) + f" | {_fmt(p['dup_max_repeat'])}회 |")
    A("")
    A("**결정: Phase 0 에서 제거하지 않는다.** cosmetics 초과 행의 89.4% 가 "
      "`remove_from_cart` 이며, 이를 지우면 해당 이벤트 물량의 약 25% 가 "
      "사라진다. 중복의 의미(수량 vs 오류) 판정은 `cart_line` 을 정의하는 "
      "**Phase 2 의 몫**이며, 반복 횟수는 `cart_repeat_cnt` 로 피처화한다.")
    A("")

    A("### 7-2. `user_session` 은 세션이 아니다 → 재세션화 필요")
    A("")
    A("§5 에서 확인했듯 세션 길이 극단값이 패널 전체 길이에 육박한다. "
      "**Phase 2 에서 30분 무활동 기준 재세션화를 (강건성 확인이 아니라) "
      "기본 정의로 채택**하고, 원본 `user_session` 은 대조군으로만 쓴다.")
    A("")
    A("또한 `user_session` 단독은 키로 쓸 수 없다 — 복수 `user_id` 에 걸친 "
      "세션이 존재한다(§5). 세션 단위 연산은 반드시 복합 키 "
      "`['user_id', 'user_session']` (`loader.SESSION_KEY`) 를 사용한다.")
    A("")

    A("### 7-3. `user_session` 결측")
    A("")
    A("| 스토어 | 결측 행 | event_type 구성 |")
    A("|---|---:|---|")
    for p in profiles:
        parts = [f"`{r['event_type']}` {_fmt(int(r['n']))}"
                 for r in p["null_sess"].iter_rows(named=True)]
        A(f"| {p['store']} | {_fmt(p['n_null_sess'])} | " + (", ".join(parts) or "—") + " |")
    A("")
    A("cosmetics 의 결측 세션에는 **`purchase` 가 단 한 건도 없다** — 무작위 "
      "결측이 아니라 특정 로깅 경로의 결함으로 보인다. 물량이 0.02% 로 "
      "미미해 세션 기반 분석에서는 제외하되, **비전환 쪽으로 치우친 결측**임을 "
      "기록해 둔다.")
    A("")

    A("### 7-4. `price <= 0` → 행 삭제가 아니라 결측 처리")
    A("")
    A("| 스토어 | price<=0 행 | event_type 구성 | 전체 상품 | 1회 이상 발생 상품 | 항상 0원 상품 |")
    A("|---|---:|---|---:|---:|---:|")
    for p in profiles:
        d = p["price_diag"]
        parts = [f"`{r['event_type']}` {_fmt(int(r['n']))}"
                 for r in p["bad_price"].iter_rows(named=True)]
        n_bad = int(p["price"]["neg"]) + int(p["price"]["zero"])
        A(f"| {p['store']} | {_fmt(n_bad)} | " + (", ".join(parts) or "—") +
          f" | {_fmt(d['products_total'])} | {_fmt(d['products_any_bad'])} "
          f"({d['products_any_bad'] / d['products_total']:.1%}) | {_fmt(d['products_all_bad'])} |")
    A("")
    A("cosmetics 는 전체 상품의 40.3% 가 한 번쯤 `price<=0` 을 겪지만 **'항상 "
      "0원'인 상품은 667개(1.2%)뿐**이다. 즉 상품 결함이 아니라 **가격 미설정 "
      "기간**이다.")
    A("")
    A("**결정: 행을 지우지 않고 `price` 만 null 로 만든 뒤 `price_missing` "
      "플래그를 세운다** (`loader.clean()`). 행을 지우면 해당 `cart_line` 의 "
      "타겟 `y` 관측 자체를 잃지만, 결측 처리하면 `y` 는 보존되고 LightGBM 은 "
      "결측을 그대로 처리하며, 무엇보다 **\"가격이 안 붙은 상품은 전환이 "
      "낮은가\"** 를 검정할 수 있는 피처가 하나 생긴다.")
    A("")

    return "\n".join(L)
