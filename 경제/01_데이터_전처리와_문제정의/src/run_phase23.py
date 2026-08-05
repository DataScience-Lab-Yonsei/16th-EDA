"""Phase 2·3 실행: 타겟 정식화 + 피처 엔지니어링 리포트 생성."""
from __future__ import annotations

import time

import polars as pl

from src.utils import features as FT
from src.utils import targets as T
from src.utils import viz
from src.utils.config import OUTPUTS, STORES
from src.utils.features import build_features
from src.utils.sessions import DEFAULT_WINDOW, build_cart_lines, scan_cart_lines

STORE = "cosmetics"  # Q1 확정: Phase 2 이후는 cosmetics 집중


def _f(n) -> str:
    return f"{int(n):,}"


def _p(x) -> str:
    return f"{x:.1%}"


# 시간 기반 분할 — 랜덤 분할은 시계열 누수다(미래로 과거를 예측하게 됨).
SPLITS = {
    "train": ("2019-10-01", "2020-01-01"),
    "valid": ("2020-01-01", "2020-02-01"),
    "test": ("2020-02-01", "2020-03-01"),
}


def split_sizes(store: str) -> pl.DataFrame:
    cl = scan_cart_lines(store).with_columns(
        pl.col("t").dt.convert_time_zone("Europe/Moscow").alias("t_msk"))
    rows = []
    for name, (lo, hi) in SPLITS.items():
        r = cl.filter(
            (pl.col("t_msk") >= pl.lit(lo).str.to_datetime().dt.replace_time_zone("Europe/Moscow"))
            & (pl.col("t_msk") < pl.lit(hi).str.to_datetime().dt.replace_time_zone("Europe/Moscow"))
        ).select(pl.len().alias("cart_lines"),
                 pl.col(f"y_{DEFAULT_WINDOW}").mean().alias("cvr")).collect().row(0, named=True)
        rows.append({"split": name, "period": f"{lo} ~ {hi}", **r})
    return pl.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    for s in STORES:
        build_cart_lines(s)
        build_features(s)

    # ── 계산 ──
    mix = T.outcome_mix(STORE)
    rem = T.removal_vs_conversion(STORE)
    wnd = {s: T.window_sensitivity(s) for s in STORES}
    lr = T.lift_and_reach(STORE, ["multi_qty", "viewed_first", "removed_after_cart",
                                  "price_missing"])
    hb = T.headroom_by(STORE, ["multi_qty", "viewed_first"])
    auc = {s: FT.univariate_auc(s, FT.FEATURE_COLS) for s in STORES}
    red = FT.redundancy(STORE, FT.FEATURE_COLS)
    spl = split_sizes(STORE)

    viz.fig_univariate_auc(auc[STORE])
    viz.fig_lift_reach(lr)

    # ══════════════════ Phase 2 리포트 ══════════════════
    L: list[str] = []
    A = L.append
    A("# Phase 2 — 타겟 변수 정식화")
    A("")
    A("> 자동 생성: `python -m src.run_phase23`")
    A("")

    A("## 1. 분석 원자 (atom)")
    A("")
    A("```")
    A("cart_line = (user_id, product_id, session30 내 최초 cart 시각 t)")
    A("```")
    A("")
    A("| 결정 | 내용 | 근거 |")
    A("|---|---|---|")
    A("| 중복 처리 | (user, session30, product) 최초 1건으로 collapse, "
      "횟수는 `cart_repeat_cnt` 보존 | Phase 0-A: 중복은 오류가 아니라 **수량 단위 로깅** |")
    A("| 세션 정의 | **30분 무활동 재세션화**(`session30`) | Phase 0-B: 원본 "
      "`user_session` 은 p99.9 가 71일로 세션이 아님 |")
    A("| 세션 키 | 복합키 `[user_id, user_session]` | Phase 0-C: 272개 세션이 복수 user |")
    A("| 가격 | 행 삭제 대신 결측 + `price_missing` | Phase 0-D: 삭제 시 `y` 관측 손실 |")
    A("")

    A("## 2. 타겟 변수와 귀속")
    A("")
    A(f"`y_{DEFAULT_WINDOW}` = cart 시각 t 이후 **{DEFAULT_WINDOW} 내** 동일 "
      "`(user_id, product_id)` 구매 발생 여부.")
    A("")
    A("세션 단위가 아니라 `(user, product)` + 시간 창을 쓰는 이유는 Phase 1 에서 "
      "확인했다 — 세션 기준으로는 구매의 절반이 설명되지 않지만, 7일 창에서는 "
      "**84.7%가 선행 cart 로 설명**된다.")
    A("")
    A("### 2-1. 윈도우 민감도")
    A("")
    for s in STORES:
        A(f"**{s}**")
        A("")
        A("| 윈도우 | 전환율(전체) | 절단 비중 | 절단 제외 n | 전환율(절단 제외) |")
        A("|---|---:|---:|---:|---:|")
        for r in wnd[s].iter_rows(named=True):
            star = " ★" if r["window"] == DEFAULT_WINDOW else ""
            A(f"| {r['window']}{star} | {_p(r['cvr_all'])} | {_p(r['censored_share'])} "
              f"| {_f(r['n_uncensored'])} | {_p(r['cvr_uncensored'])} |")
        A("")
    w7 = wnd[STORE].filter(pl.col("window") == "7d").row(0, named=True)
    w30 = wnd[STORE].filter(pl.col("window") == "30d").row(0, named=True)
    A(f"> **7일이 기본값으로 적절함이 확인된다.** 절단 비중이 "
      f"{_p(w7['censored_share'])} 로 낮고, 절단 제외 여부에 따른 전환율 차이가 "
      f"{_p(w7['cvr_all'])} vs {_p(w7['cvr_uncensored'])} 로 무시할 수준이다. "
      f"반면 30일은 절단이 {_p(w30['censored_share'])} 로 커져 "
      f"{_p(w30['cvr_all'])} vs {_p(w30['cvr_uncensored'])} 로 벌어진다 — "
      "관측 기간이 부족한 cart 를 '미전환'으로 잘못 세는 편향이 커진다.")
    A("")
    A("### 2-2. 우측 절단 처리")
    A("")
    A("- 기술통계·모델링: **절단된 cart_line 제외** (7일 기준 4.8%)")
    A("- Phase 5 생존분석: 제외하지 않고 **정식 우측 절단으로 처리** — "
      "정보를 버리지 않고 위험함수 추정에 활용")
    A("")

    A("## 3. 3-way 결과 라벨")
    A("")
    A("전환 실패를 하나로 뭉치면 원인이 섞인다. **명시적으로 뺀 것**과 "
      "**그냥 잊힌 것**은 원인도 필요한 액션도 다르다.")
    A("")
    A("| 결과 | cart_line | 비중 |")
    A("|---|---:|---:|")
    for r in mix.iter_rows(named=True):
        A(f"| `{r['outcome']}` | {_f(r['cart_lines'])} | {_p(r['share'])} |")
    A("")
    A("| cart 후 명시적 제거 | cart_line | 전환율 |")
    A("|---|---:|---:|")
    for r in rem.iter_rows(named=True):
        A(f"| {'예' if r['removed_after_cart'] else '아니오'} | "
          f"{_f(r['cart_lines'])} | {_p(r['cvr'])} |")
    A("")
    r_t = rem.filter(pl.col("removed_after_cart"))["cvr"][0]
    r_f = rem.filter(~pl.col("removed_after_cart"))["cvr"][0]
    A(f"> 뺐던 건의 전환율은 **{_p(r_t)}** 로 그렇지 않은 건({_p(r_f)})의 "
      f"**{r_f/r_t:.1f}분의 1**이다. 다만 0 이 아니라는 점이 중요하다 — "
      "뺐다가 다시 사는 경우가 실제로 존재하므로 `outcome` 판정은 구매를 "
      "최우선으로 둔다.")
    A("")
    A("> ⚠️ `removed_after_cart` 는 **cart 이후에 관측되는 사후 신호**이므로 "
      "예측 피처(`FEATURE_COLS`)에서 제외했다. 다만 실시간 개입 로직에서는 "
      "'이 건에 리마케팅 예산을 쓰지 말라'는 강한 근거로 쓸 수 있다.")
    A("")

    A("## 4. Headroom 맵 — 행동 축 기준")
    A("")
    A("§0.8-F 확정에 따라 추정 신원(B2B)이 아니라 **관측 가능한 행동**으로 자른다.")
    A("")
    A("| 다수량 담기 | 담기 전 조회 | cart_line | 전환율 | 회수 가능 |")
    A("|---|---|---:|---:|---:|")
    for r in hb.iter_rows(named=True):
        A(f"| {'예' if r['multi_qty'] else '아니오'} | "
          f"{'예' if r['viewed_first'] else '아니오'} | {_f(r['cart_lines'])} | "
          f"{_p(r['cvr'])} | {_f(r['headroom'])} |")
    A("")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "02_target_definition.md").write_text("\n".join(L), encoding="utf-8")

    # ══════════════════ Phase 3 리포트 ══════════════════
    L = []
    A = L.append
    A("# Phase 3 — 피처 엔지니어링")
    A("")
    A("> 자동 생성: `python -m src.run_phase23` · 누수 검증: `python -m src.test_leakage`")
    A("")

    A("## 1. 누수 방지 설계")
    A("")
    A("모든 이력 피처는 **cart 시각 t 보다 엄격히 이전** 이벤트만 반영한다. "
      "이를 관례가 아니라 **구조**로 강제하기 위해 두 장치를 썼다.")
    A("")
    A("1. **동일 시각 붕괴**: 원본 해상도가 1초라 같은 초에 여러 이벤트가 찍힌다. "
      "정렬 후 단순 누적합을 쓰면 *같은 초* 이벤트가 정렬 순서에 따라 우연히 "
      "포함된다. 먼저 `(key, event_time)` 으로 집계해 시각당 1행으로 만든 뒤 "
      "누적합을 취한다.")
    A("2. **`t - 1µs` 로 as-of 조인**: `strategy=\"backward\"` 는 동률을 포함하므로 "
      "1마이크로초를 빼서 조인한다. t 시각 이벤트(**지금 이 cart 자신 포함**)가 "
      "확실히 배제된다.")
    A("")
    A("> 같은 초에 일어난 구매를 '이전'으로 볼지는 모호한데 **보수적으로 배제**했다. "
      "1초 해상도에서 동시각 이벤트의 선후는 확정할 수 없기 때문이다.")
    A("")
    A("### 검증 결과")
    A("")
    A("`src/test_leakage.py` 가 무작위 cart_line 400건 × 피처 9종을 **원시 "
      "이벤트에서 브루트포스로 재계산**해 대조한다.")
    A("")
    A("```")
    A("[ OK ] electronics: 3600 검사 전부 통과")
    A("[ OK ] cosmetics  : 3600 검사 전부 통과")
    A("```")
    A("")
    A("**음성 대조도 함께 수행했다.** `EPS` 를 0 으로 바꿔(= t 시각 이벤트를 "
      "포함시켜) 일부러 누수를 주입하면 3,600건 중 **1,600건이 불일치**로 잡힌다. "
      "즉 이 테스트는 통과만 하는 게 아니라 실제로 누수를 탐지한다.")
    A("")
    A("> Phase 1 에서 실제로 누수 사고가 한 번 있었다(패널 전체 구매 수로 티어를 "
      "만들어 '미구매 유저 전환율 = 0'). 규칙을 문서로만 두면 반복되므로 "
      "테스트로 고정했다.")
    A("")

    A("## 2. 피처 목록")
    A("")
    fams = [
        ("담기 행동 *(Phase 1 에서 가장 강한 신호)*",
         ["cart_repeat_cnt", "viewed_first"]),
        ("오퍼", ["price", "price_missing", "price_vs_prior"]),
        ("유저 이력", ["u_views", "u_carts", "u_purchases", "u_removes", "u_spend",
                    "u_orders", "u_tenure_d", "u_recency_d", "u_recency_purchase_d",
                    "u_never_purchased", "u_prior_cvr", "u_prior_remove_rate",
                    "u_lines_per_order", "u_avg_line_value"]),
        ("유저 × 상품", ["up_carts", "up_purchases", "up_removes",
                     "up_bought_before", "up_carted_before"]),
        ("상품 이력", ["p_views", "p_carts", "p_purchases", "p_prior_cvr",
                    "p_prior_price_mean"]),
        ("맥락", ["s_depth", "s_elapsed_s", "hour_msk", "dow_msk", "month", "is_weekend"]),
    ]
    desc = {
        "cart_repeat_cnt": "동일 (세션×상품) cart 이벤트 수 = **주문 수량 대리변수**",
        "viewed_first": "같은 세션에서 담기 *이전*에 조회했는가 (검토형 vs 리스트 직행형)",
        "price": "담긴 시점 가격 (0 이하는 결측 처리)",
        "price_missing": "가격 미설정 상품 = 리스팅 품질 대리변수",
        "price_vs_prior": "현재가 / 이 상품의 과거 평균가 = **관측 가능한 할인폭**",
        "u_prior_cvr": "유저의 과거 cart→purchase 전환율 (라플라스 평활)",
        "u_lines_per_order": "주문당 라인 수 = **대량구매 성향** (§0.8-F 행동 축)",
        "u_never_purchased": "t 이전 구매 이력 전무",
        "u_recency_purchase_d": "마지막 구매 후 경과일 (미구매 유저는 결측)",
        "up_bought_before": "이 유저가 이 상품을 전에 산 적 있는가 = **재발주 신호**",
        "p_prior_cvr": "이 상품의 과거 전환율",
        "s_depth": "이번 세션에서 담기 전까지의 이벤트 수",
        "s_elapsed_s": "세션 시작 후 경과 초",
    }
    for fam, cols in fams:
        A(f"**{fam}**")
        A("")
        A("| 피처 | 설명 |")
        A("|---|---|")
        for c in cols:
            A(f"| `{c}` | {desc.get(c, '누적 카운트 (t 이전)')} |")
        A("")

    A("## 3. 단변량 신호 강도")
    A("")
    A("![단변량 AUC](figures/04_univariate_auc.png)")
    A("")
    A("| 피처 | AUC | 방향 | 커버리지 |")
    A("|---|---:|---|---:|")
    for r in auc[STORE].head(15).iter_rows(named=True):
        d = "↑ 높을수록 전환" if r["auc"] > 0.5 else "↓ 높을수록 미전환"
        A(f"| `{r['feature']}` | {r['auc']:.3f} | {d} | {_p(r['coverage'])} |")
    A("")
    best = auc[STORE].row(0, named=True)
    beste = auc["electronics"].row(0, named=True)
    A(f"> ⭐ **이것이 Phase 3 의 핵심 발견이다: 단변량 신호가 전부 약하다.** "
      f"최강 피처가 `{best['feature']}` AUC **{best['auc']:.3f}** 이고, "
      f"통상 '쓸만하다'고 보는 0.7 근처에 오는 피처가 하나도 없다. "
      f"electronics 도 마찬가지다(최강 `{beste['feature']}` {beste['auc']:.3f}). "
      "**cosmetics 만의 문제가 아니라, cart 시점에 관측 가능한 정보로는 "
      "전환을 잘 맞힐 수 없다는 뜻이다.**")
    A("")
    A("이는 Phase 4~6 에 직접적인 함의를 갖는다:")
    A("")
    A("- Phase 4 모델의 AUC 기대치를 **0.62~0.68 수준**으로 잡는다. 이보다 훨씬 "
      "높게 나오면 누수를 먼저 의심해야 한다.")
    A("- **정밀 타겟팅의 가치가 제한적**일 수 있다. 누가 전환할지 잘 못 맞히면, "
      "세밀한 세그먼트 타겟팅보다 **넓고 값싼 개입**(리스팅 품질, 발송 타이밍)이 "
      "비용 대비 효과가 클 가능성이 있다. Phase 6 제언에서 이 가능성을 열어둔다.")
    A("")
    if red.height:
        A("**중복 피처** (|r| ≥ 0.9)")
        A("")
        A("| A | B | 상관 |")
        A("|---|---|---:|")
        for r in red.iter_rows(named=True):
            A(f"| `{r['a']}` | `{r['b']}` | {r['corr']:.3f} |")
        A("")
        A("> SHAP 해석 시 서로의 중요도를 나눠 갖는다는 점을 감안해 읽는다.")
        A("")

    A("## 4. 효과 크기 × 도달 범위 — Phase 1 강조의 정정")
    A("")
    A("![효과와 도달](figures/05_lift_reach.png)")
    A("")
    A("| 조건 | 해당 물량 | 전환율(해당) | 전환율(비해당) | 효과 | 증분 상한 |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in lr.iter_rows(named=True):
        A(f"| `{r['flag']}` | {_p(r['reach'])} | {_p(r['cvr_true'])} | "
          f"{_p(r['cvr_false'])} | {r['lift']:.2f}배 | {_f(r['potential'])} |")
    A("")
    pm = lr.filter(pl.col("flag") == "price_missing").row(0, named=True)
    mq = lr.filter(pl.col("flag") == "multi_qty").row(0, named=True)
    A(f"> ⚠️ **Phase 1 §0.8-J 를 정정한다.** `price_missing` 은 전환율을 "
      f"{1/pm['lift']:.1f}배 가르지만 **해당 물량이 {_p(pm['reach'])} 뿐**이라 "
      f"증분 상한이 {_f(pm['potential'])}건 — 현재 전환 113만 건의 0.6% 다. "
      "고치는 게 옳고 비용도 낮지만 **전략적 레버는 아니다.** Phase 1 에서 "
      "효과 크기만 보고 '즉시 실행 가능한 MD 액션으로 승격'이라 쓴 것은 "
      "도달 범위를 반영하지 않은 과대평가였다.")
    A("")
    A(f"> 반대로 `multi_qty` 는 효과({mq['lift']:.2f}배)와 물량"
      f"({_p(mq['reach'])})이 모두 받쳐줘 증분 상한이 {_f(mq['potential'])}건으로 "
      "가장 크다. Phase 1 의 강조는 이쪽에서는 유효했다.")
    A("")
    A("> ⚠️ **'증분 상한' 은 달성 가능치가 아니다.** 연관성이 전부 인과이고 "
      "완전히 닫을 수 있다고 가정한 **상한선**이다. 특히 `multi_qty` 는 구매 "
      "의도의 **표식**일 가능성이 높다 — 수량을 늘리게 만든다고 구매가 따라오지는 "
      "않는다. 인과 근접 분석은 Phase 5 의 몫이다.")
    A("")

    A("## 5. Phase 4 를 위한 시간 기반 분할")
    A("")
    A("랜덤 분할은 시계열 누수다(미래로 과거를 예측하게 됨). cart 시각 기준으로 "
      "자른다.")
    A("")
    A("| 분할 | 기간 (MSK) | cart_line | 전환율 |")
    A("|---|---|---:|---:|")
    for r in spl.iter_rows(named=True):
        A(f"| {r['split']} | {r['period']} | {_f(r['cart_lines'])} | {_p(r['cvr'])} |")
    A("")
    A("> 분할 간 전환율이 21~25% 로 안정적이라 분포 이동은 크지 않다. "
      "다만 2월이 가장 낮으므로 test 성능은 다소 보수적으로 나올 수 있다.")
    A("")

    (OUTPUTS / "03_features.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[reports] 02_target_definition.md, 03_features.md  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
