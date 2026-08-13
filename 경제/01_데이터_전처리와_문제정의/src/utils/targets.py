"""Phase 2 — 타겟 변수 정식화: 3-way 결과 · 윈도우 민감도 · 절단 · headroom."""
from __future__ import annotations

import polars as pl

from src.utils.sessions import DEFAULT_WINDOW, WINDOWS, scan_cart_lines

Y = f"y_{DEFAULT_WINDOW}"


def outcome_mix(store: str) -> pl.DataFrame:
    """3-way 결과 분포. 명시적 제거와 조용한 이탈은 원인도 액션도 다르다."""
    cl = scan_cart_lines(store)
    return (
        cl.group_by("outcome")
        .agg(pl.len().alias("cart_lines"))
        .with_columns((pl.col("cart_lines") / pl.col("cart_lines").sum()).alias("share"))
        .sort("cart_lines", descending=True)
        .collect()
    )


def removal_vs_conversion(store: str) -> pl.DataFrame:
    """장바구니에서 뺐던 건은 결국 사는가? — 명시적 제거의 신호 가치."""
    return (
        scan_cart_lines(store)
        .group_by("removed_after_cart")
        .agg(pl.len().alias("cart_lines"), pl.col(Y).mean().alias("cvr"))
        .sort("removed_after_cart")
        .collect()
    )


def window_sensitivity(store: str) -> pl.DataFrame:
    """윈도우 W 별 전환율 + 절단 비중. 결론이 W 에 흔들리는지 확인용."""
    cl = scan_cart_lines(store)
    rows = []
    for name in WINDOWS:
        r = cl.select(
            pl.col(f"y_{name}").mean().alias("cvr_all"),
            pl.col(f"censored_{name}").mean().alias("censored_share"),
        ).collect().row(0, named=True)
        # 절단된 cart_line 을 제외했을 때
        r2 = cl.filter(~pl.col(f"censored_{name}")).select(
            pl.len().alias("n_uncensored"),
            pl.col(f"y_{name}").mean().alias("cvr_uncensored"),
        ).collect().row(0, named=True)
        rows.append({"window": name, **r, **r2})
    return pl.DataFrame(rows)


def lift_and_reach(store: str, flags: list[str]) -> pl.DataFrame:
    """이진 조건별 **효과 크기(lift) × 도달 범위(reach)**.

    왜 필요한가: lift 만 보면 판단을 그르친다. `price_missing` 은 전환율을
    6배 가르지만 물량의 0.7% 에만 해당해 총량 기여가 거의 없다. MD 의사결정은
    "이 조건을 전부 고쳤을 때 추가로 생기는 구매 **개수**" 로 해야 한다.

    ``potential`` = 해당 조건의 cart_line 을 나머지 수준의 전환율로 끌어올렸을
    때 생기는 증분 구매 수. 상한선(ceiling)이지 예측치가 아니다.
    """
    cl = scan_cart_lines(store).with_columns(
        (pl.col("cart_repeat_cnt") > 1).alias("multi_qty")
    ).collect()
    n_all = cl.height
    rows = []
    for f in flags:
        g = cl.group_by(f).agg(pl.len().alias("n"), pl.col(Y).mean().alias("cvr"))
        g = g.filter(pl.col(f).is_not_null())
        if g.height != 2:
            continue
        on = g.filter(pl.col(f))
        off = g.filter(~pl.col(f))
        n_on, cvr_on = int(on["n"][0]), float(on["cvr"][0])
        n_off, cvr_off = int(off["n"][0]), float(off["cvr"][0])
        # 낮은 쪽을 높은 쪽까지 끌어올린다고 가정한 상한
        lo_n, lo_cvr, hi_cvr = ((n_on, cvr_on, cvr_off) if cvr_on < cvr_off
                                else (n_off, cvr_off, cvr_on))
        rows.append({
            "flag": f,
            "reach": n_on / n_all,
            "cvr_true": cvr_on,
            "cvr_false": cvr_off,
            "lift": cvr_on / cvr_off if cvr_off else float("nan"),
            "potential": round(lo_n * (hi_cvr - lo_cvr)),
        })
    return pl.DataFrame(rows).sort("potential", descending=True)


def headroom_by(store: str, dims: list[str], min_lines: int = 20000) -> pl.DataFrame:
    """행동 축(§0.8-F)으로 자른 회수 가능 물량 맵."""
    return (
        scan_cart_lines(store)
        .with_columns((pl.col("cart_repeat_cnt") > 1).alias("multi_qty"))
        .group_by(dims)
        .agg(
            pl.len().alias("cart_lines"),
            pl.col(Y).mean().alias("cvr"),
            (pl.len() * (1 - pl.col(Y).mean())).round(0).alias("headroom"),
        )
        .filter(pl.col("cart_lines") >= min_lines)
        .sort("headroom", descending=True)
        .collect()
    )
