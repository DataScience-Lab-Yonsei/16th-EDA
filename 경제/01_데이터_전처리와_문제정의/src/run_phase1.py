"""Phase 1 실행: 퍼널 베이스라인 · B2B 가설 검증 · 격차 분해 리포트 생성."""
from __future__ import annotations

import time

import polars as pl

from src.utils import funnel as F
from src.utils import viz
from src.utils.config import OUTPUTS, STORES
from src.utils.sessions import DEFAULT_WINDOW, build_cart_lines, build_events

STORE_KO = {"cosmetics": "cosmetics", "electronics": "electronics"}


def _f(n) -> str:
    return f"{int(n):,}"


def _p(x) -> str:
    return f"{x:.1%}"


def main() -> None:
    t0 = time.time()
    for s in STORES:
        build_events(s)
        build_cart_lines(s)

    ov = {s: F.funnel_overview(s) for s in STORES}
    stages = {s: F.funnel_stages(s) for s in STORES}
    mon = {s: F.monthly(s) for s in STORES}
    band = {s: F.by_price_band(s) for s in STORES}
    pmiss = {s: F.price_missing_effect(s) for s in STORES}
    ppu = {s: F.purchases_per_user(s).row(0, named=True) for s in STORES}
    ords = {s: F.orders(s).row(0, named=True) for s in STORES}
    rep = {s: F.repeat_purchase_interval(s).row(0, named=True) for s in STORES}
    biz = {s: F.business_hours_share(s).row(0, named=True) for s in STORES}
    lad = {s: F.alignment_ladder(s) for s in STORES}
    ttp = {s: F.time_to_purchase_curve(s) for s in STORES}
    d2c = {s: F.direct_to_cart(s) for s in STORES}
    pockets = F.conversion_pockets("cosmetics")
    brands = F.by_brand("cosmetics", 10)
    support = F.price_common_support()
    overlap = F.overlap_region_comparison(3.0, 40.0)

    # ── 그림 ──
    viz.fig_alignment_ladder({s: lad[s].to_dict(as_series=False) for s in STORES})
    viz.fig_time_to_purchase({s: ttp[s].to_dict(as_series=False) for s in STORES})
    seg = (
        pockets.with_columns(
            (pl.col("prior_tier").str.replace(r"^\d+\.\s*", "")
             + pl.when(pl.col("multi_qty")).then(pl.lit(" · 다수량"))
               .otherwise(pl.lit(" · 단일수량"))).alias("label")
        )
        .sort("headroom", descending=True).head(6)
    )
    viz.fig_headroom(seg.to_dict(as_series=False), benchmark=ov["electronics"]["cvr"])

    L: list[str] = []
    A = L.append

    A("# Phase 1 — 퍼널 베이스라인 · B2B 가설 · 격차 분해")
    A("")
    A("> 자동 생성: `python -m src.run_phase1`")
    A(f"> 타겟: `y_{DEFAULT_WINDOW}` — cart 시점 t 이후 {DEFAULT_WINDOW} 내 "
      "동일 (user, product) 구매")
    A("")

    # ══ 요약 ══
    cos, ele = ov["cosmetics"], ov["electronics"]
    gap0 = lad["electronics"]["cvr"][0] / lad["cosmetics"]["cvr"][0]
    gapf = ele["cvr"] / cos["cvr"]
    A("## 요약 — 핵심 결론 5가지")
    A("")
    A(f"1. **\"6.6배 격차\"는 과장이었다.** 동일 정의(원본 세션)로 전체 패널을 "
      f"보면 {gap0:.2f}배이고, cart 중복·귀속 윈도우를 정렬하면 **{gapf:.2f}배**로 "
      "줄어든다. 사전조사의 6.6배는 단일 월 표본 + 세션 정의 아티팩트였다.")
    A(f"2. **남은 {gapf:.2f}배는 가격 구성으로 설명되지 않는다.** 공통 가격대"
      "($3~$40)로 제한하면 격차는 오히려 **커진다**(§3-2). 구조적·행동적 차이다.")
    nv = {s: d2c[s].filter(~pl.col("viewed_first"))["cart_lines"][0] for s in STORES}
    A(f"3. **cosmetics 장바구니의 {_p(nv['cosmetics'] / cos['cart_lines'])}는 "
      f"담기 전에 그 상품을 본 적이 없다** (electronics 는 "
      f"{_p(nv['electronics'] / ele['cart_lines'])}). 탐색형 쇼핑이 아니라 "
      "**리스트에서 바로 담는 발주형**이다. 이 한 지표가 두 스토어의 성격 차이를 "
      "가장 잘 요약한다.")
    A("4. **cosmetics 안에 이미 electronics 수준의 전환율 구간이 존재한다.** "
      "수량을 늘려 담은 cart 는 전환율이 모든 구매이력 구간에서 일관되게 "
      "**1.5~1.8배** 높고, 상위 구간은 51%로 electronics 벤치마크에 도달한다(§4).")
    A(f"5. **가장 큰 회수 물량은 '구매이력 없음 × 단일수량'** — cart_line "
      f"{_f(pockets.filter((pl.col('prior_tier') == '0. 구매이력 없음') & ~pl.col('multi_qty'))['cart_lines'][0])}건"
      f"({_p(pockets.filter((pl.col('prior_tier') == '0. 구매이력 없음') & ~pl.col('multi_qty'))['cart_lines'][0] / cos['cart_lines'])}), "
      "전환율 20.4%. 전환율이 낮아서가 아니라 **물량이 압도적**이라 여기가 기회다.")
    A("")

    # ══ 1-A ══
    A("## 1. 퍼널 베이스라인")
    A("")
    A("### 1-1. 스토어 요약")
    A("")
    A("| 지표 | cosmetics | electronics |")
    A("|---|---:|---:|")
    rows = [
        ("유저", lambda o: _f(o["users"])),
        ("이벤트", lambda o: _f(o["events"])),
        ("세션(30분 재구성)", lambda o: _f(o["sessions30"])),
        ("**cart_line**", lambda o: f"**{_f(o['cart_lines'])}**"),
        ("cart 유저", lambda o: _f(o["cart_users"])),
        ("전환(7일)", lambda o: _f(o["converted"])),
        ("**전환율(7일)**", lambda o: f"**{_p(o['cvr'])}**"),
        ("cart 후 명시적 제거율", lambda o: _p(o["removed_rate"]) if o["removed_rate"] else "—(미로깅)"),
        ("cart 수량반복 평균", lambda o: f"{o['repeat_mean']:.2f}"),
    ]
    for label, fn in rows:
        A(f"| {label} | {fn(cos)} | {fn(ele)} |")
    A("")

    A("### 1-2. 3단 퍼널 — (user, 세션, 상품) 단위")
    A("")
    A("| 단계 | cosmetics | electronics |")
    A("|---|---:|---:|")
    sc, se = stages["cosmetics"].row(0, named=True), stages["electronics"].row(0, named=True)
    A(f"| 조회된 (유저×세션×상품) | {_f(sc['viewed'])} | {_f(se['viewed'])} |")
    A(f"| 장바구니 담김 | {_f(sc['carted'])} | {_f(se['carted'])} |")
    A(f"| 같은 세션에 조회 기록 있음 *(선후 무관)* | {_f(sc['view_then_cart'])} "
      f"({sc['view_then_cart']/sc['carted']:.1%}) | {_f(se['view_then_cart'])} "
      f"({se['view_then_cart']/se['carted']:.1%}) |")
    vf = {s: d2c[s].filter(pl.col("viewed_first")) for s in STORES}
    A(f"| **담기 *이전*에 조회** | **{_f(vf['cosmetics']['cart_lines'][0])} "
      f"({vf['cosmetics']['cart_lines'][0]/cos['cart_lines']:.1%})** | "
      f"**{_f(vf['electronics']['cart_lines'][0])} "
      f"({vf['electronics']['cart_lines'][0]/ele['cart_lines']:.1%})** |")
    A("")
    A("> 두 행은 다른 것을 잰다. 위 행은 세션 안 어디든 조회가 있었는지, 아래 행은 "
      "**담기보다 먼저** 조회했는지다. 인과적으로 의미 있는 쪽은 아래 행이므로 "
      "이후 분석은 아래 정의(`viewed_first`)를 쓴다.")
    A("")
    A("> ⭐ **가장 큰 구조적 차이.** electronics 는 사실상 모든 장바구니 담기가 "
      "상품 조회 뒤에 일어나는 정상 탐색 퍼널이다. cosmetics 는 그 반대다 — "
      "5건 중 4건은 담기 전에 그 상품을 보지 않는다. 재구매·발주 리스트에서 바로 "
      "담는 행동으로, **'장바구니에 담았다'는 사실이 두 스토어에서 서로 다른 의미**"
      "임을 뜻한다.")
    A("")

    A("### 1-3. 월별 추이")
    A("")
    A("| 월 | cosmetics cart_line | cosmetics 전환율 | electronics cart_line | electronics 전환율 |")
    A("|---|---:|---:|---:|---:|")
    mc = {r["month"].strftime("%Y-%m"): r for r in mon["cosmetics"].iter_rows(named=True)}
    me = {r["month"].strftime("%Y-%m"): r for r in mon["electronics"].iter_rows(named=True)}
    for k in sorted(set(mc) | set(me)):
        c, e = mc.get(k), me.get(k)
        A(f"| {k} | {_f(c['cart_lines']) if c else '—'} | {_p(c['cvr']) if c else '—'} "
          f"| {_f(e['cart_lines']) if e else '—'} | {_p(e['cvr']) if e else '—'} |")
    A("")
    A("> 마지막 행(2020-03 / 2021-03)은 UTC→MSK 변환에 따른 월말 잔여분으로 표본이 "
      "작아 해석 대상이 아니다. cosmetics 는 11월(블랙프라이데이)이 최고, 2월이 최저다.")
    A("")

    A("### 1-4. 가격대별 — 두 스토어가 정반대")
    A("")
    for s in STORES:
        A(f"**{s}**")
        A("")
        A("| 분위 | 가격 구간 | cart_line | 전환율 | 회수 가능 |")
        A("|---|---|---:|---:|---:|")
        for r in band[s].iter_rows(named=True):
            A(f"| Q{r['band']+1} | ${r['lo']:,.2f} ~ ${r['hi']:,.2f} | "
              f"{_f(r['cart_lines'])} | {_p(r['cvr'])} | {_f(r['headroom'])} |")
        A("")
    A("> **electronics 는 가격이 오를수록 전환율이 단조 감소**(58.4%→43.2%)하는 "
      "전형적인 소비자 가격 민감성을 보인다. **cosmetics 는 거의 평평하다**"
      "(28.3%→21.7%, 비단조). 가격이 전환의 주 동인이 아니라는 뜻이며, "
      "가격 할인 중심 전략의 기대 효과가 낮음을 시사한다.")
    A("")

    A("### 1-5. 가격 미설정 상품 — Phase 0-D 가설 검증됨")
    A("")
    A("| price_missing | cart_line | 전환율 |")
    A("|---|---:|---:|")
    for r in pmiss["cosmetics"].iter_rows(named=True):
        A(f"| {r['price_missing']} | {_f(r['cart_lines'])} | {_p(r['cvr'])} |")
    A("")
    r0 = pmiss["cosmetics"].filter(~pl.col("price_missing"))["cvr"][0]
    r1 = pmiss["cosmetics"].filter(pl.col("price_missing"))["cvr"][0]
    A(f"> ⭐ 가격이 붙지 않은 상품의 전환율은 **{_p(r1)}** 로 정상 상품"
      f"({_p(r0)})의 **{r0/r1:.1f}분의 1**이다. Phase 0 에서 원안대로 이 행들을 "
      "삭제했다면 이 발견 자체가 사라졌을 것이다. **리스팅 품질 관리가 즉시 실행 "
      "가능한 MD 액션**이며, Phase 3 피처로 확정한다.")
    A("")

    A("### 1-6. 브랜드 (cosmetics 상위 10)")
    A("")
    A("| 브랜드 | cart_line | 전환율 | 회수 가능 | 중간가 |")
    A("|---|---:|---:|---:|---:|")
    for r in brands.iter_rows(named=True):
        A(f"| {r['brand']} | {_f(r['cart_lines'])} | {_p(r['cvr'])} | "
          f"{_f(r['headroom'])} | ${r['price_med']:.2f} |")
    A("")
    A("> 브랜드 간 전환율 편차가 19~27% 로 좁아 **브랜드는 강한 타겟팅 레버가 "
      "아니다**. 브랜드 미상이 물량의 43%를 차지하지만 전환율은 평균 수준이라, "
      "브랜드 결측 자체는 문제 신호가 아니다(가격 결측과 대조적).")
    A("")

    # ══ 1-B ══
    A("## 2. B2B 가설(H0) 검증")
    A("")
    A("사전조사에서 세운 가설: *cosmetics 의 핵심 구매층은 일반 소비자가 아니라 "
      "네일샵/살롱의 전문 구매자이며, cart 를 발주 리스트로 쓴다.*")
    A("")
    A("| 검증 항목 | cosmetics | electronics | 판정 |")
    A("|---|---:|---:|---|")
    A(f"| 주문당 라인 수 (중앙값) | **{ords['cosmetics']['lines_p50']:.0f}** | "
      f"{ords['electronics']['lines_p50']:.0f} | ✅ 지지 |")
    A(f"| 주문당 라인 수 (평균) | {ords['cosmetics']['lines_mean']:.1f} | "
      f"{ords['electronics']['lines_mean']:.1f} | ✅ 지지 |")
    A(f"| 주문 금액 (중앙값) | ${ords['cosmetics']['value_p50']:,.2f} | "
      f"${ords['electronics']['value_p50']:,.2f} | — |")
    A(f"| 유저당 구매 라인 (중앙값) | **{ppu['cosmetics']['p50']:.0f}** | "
      f"{ppu['electronics']['p50']:.0f} | ✅ 지지 |")
    A(f"| 유저당 구매 라인 (최대) | {_f(ppu['cosmetics']['max'])} | "
      f"{_f(ppu['electronics']['max'])} | ✅ 지지 |")
    A(f"| 구매자 중 5건 초과 비중 | **{_p(ppu['cosmetics']['gt5_share'])}** | "
      f"{_p(ppu['electronics']['gt5_share'])} | ✅ 지지 |")
    A(f"| 상위 10% 구매자 집중도 | {_p(ppu['cosmetics']['top10pct_share'])} | "
      f"{_p(ppu['electronics']['top10pct_share'])} | ◐ 약한 지지 |")
    A(f"| 동일 상품 재구매 간격 (중앙값) | {rep['cosmetics']['days_p50']:.1f}일 | "
      f"{rep['electronics']['days_p50']*24*60:.0f}분 | ✅ 지지 |")
    A(f"| 평일 업무시간(09~19시) 구매 비중 | {_p(biz['cosmetics']['business_hours_share'])} | "
      f"{_p(biz['electronics']['business_hours_share'])} | ❌ **반증** |")
    A(f"| 평일 구매 비중 | {_p(biz['cosmetics']['weekday_share'])} | "
      f"{_p(biz['electronics']['weekday_share'])} | ❌ **반증** |")
    A("")
    A("> electronics 의 '재구매 간격 중앙값 수 분' 은 실제 재구매가 아니라 **동일 "
      "주문 내 중복 구매 이벤트**다. 진짜 재구매로 보기 어려우므로 비교에서 "
      "가중치를 낮춰 읽어야 한다.")
    A("")
    A("### 판정: **부분 지지 — 그러나 'B2B'라는 이름은 붙이지 않는다**")
    A("")
    A("**구매 바스켓 구조는 가설을 강하게 지지한다.** 주문당 중앙값 6라인, "
      "유저당 중앙값 6건 구매, 구매자의 54%가 5건 초과, 동일 상품을 약 10일 "
      "주기로 재구매 — 저가 소모품을 정기적으로 대량 보충하는 패턴이다.")
    A("")
    A(f"**그러나 시간 프로파일은 가설을 반증한다.** 평일 업무시간 구매 비중이 "
      f"{_p(biz['cosmetics']['business_hours_share'])} 인데, **electronics 도 "
      f"{_p(biz['electronics']['business_hours_share'])} 로 사실상 동일**하다. "
      "업무용 구매라면 나타나야 할 시간대 편중이 관측되지 않는다.")
    A("")
    A("→ 따라서 **모집단을 'B2B vs 소비자'로 이분하지 않는다.** 사업자 신원을 "
      "확인할 외부 데이터가 없고, 관측된 것은 신원이 아니라 **행동**이다. "
      "세그멘테이션 축은 추정한 신원이 아니라 **관측 가능한 행동"
      "(바스켓 크기 · 재구매 주기 · 담기 방식)** 으로 잡는다. "
      "이 편이 데이터로 뒷받침되고 타겟팅에도 바로 쓸 수 있다.")
    A("")

    # ══ 1-C ══
    A("## 3. 격차 분해")
    A("")
    A("### 3-1. Step 1 — 측정 인공물 정렬")
    A("")
    A("![정렬 사다리](figures/01_alignment_ladder.png)")
    A("")
    A("| 단계 | cosmetics 분모 | cosmetics | electronics | 격차 |")
    A("|---|---:|---:|---:|---:|")
    for i in range(lad["cosmetics"].height):
        rc = lad["cosmetics"].row(i, named=True)
        re_ = lad["electronics"].row(i, named=True)
        A(f"| {rc['stage']} | {_f(rc['denominator'])} | {_p(rc['cvr'])} | "
          f"{_p(re_['cvr'])} | {re_['cvr']/rc['cvr']:.2f}배 |")
    A("")
    A("**정렬로 사라지는 격차의 정체:**")
    A("")
    A("- **cart 중복 계상**: cosmetics 는 (세션×상품) 중복 cart 가 훨씬 많아 "
      "원시 cart 행 기준 분모가 부풀려진다 → cart_line 으로 collapse")
    A("- **귀속 윈도우**: cosmetics 구매는 담은 뒤 한참 있다 일어난다. 동일 세션 "
      f"기준이면 {_p(lad['cosmetics']['cvr'][2])} 지만 7일 창에서는 "
      f"{_p(lad['cosmetics']['cvr'][5])} 로 뛴다. electronics 는 같은 조정에도 "
      "거의 움직이지 않는다(즉시 구매형)")
    A("")
    A("> 사전조사에서 보고한 **\"6.6배\"는 2019-10 단일 월 표본**의 값이었다. "
      f"전체 패널·동일 정의로 다시 재면 {gap0:.2f}배이며, 정렬 후에는 "
      f"**{gapf:.2f}배**다. 초기 수치를 그대로 결론에 썼다면 개선 여지를 "
      "3배 가까이 과대평가했을 것이다.")
    A("")

    A("### 3-2. Step 2 — 구성 효과인가? (공통 지지 검정)")
    A("")
    A("| 스토어 | p01 | p25 | 중앙값 | p75 | p99 |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in support.iter_rows(named=True):
        A(f"| {r['store']} | ${r['p01']:,.2f} | ${r['p25']:,.2f} | ${r['p50']:,.2f} "
          f"| ${r['p75']:,.2f} | ${r['p99']:,.2f} |")
    A("")
    A("두 스토어의 가격 분포는 중앙값이 25배 차이나 **거의 겹치지 않는다**. "
      "따라서 '가격 구성을 맞추면 격차가 사라지는가' 를 재가중으로 답하려는 시도는 "
      "대부분 외삽이 되어 성립하지 않는다. 대신 **실제로 겹치는 구간에서만** "
      "비교한다:")
    A("")
    A("| $3 ~ $40 구간 | cart_line | 전환율 |")
    A("|---|---:|---:|")
    for r in overlap.iter_rows(named=True):
        A(f"| {r['store']} | {_f(r['n'])} | {_p(r['cvr'])} |")
    ratio = overlap.filter(pl.col("store") == "electronics")["cvr"][0] / \
        overlap.filter(pl.col("store") == "cosmetics")["cvr"][0]
    A("")
    A(f"> ⭐ **가격을 통제하면 격차는 줄지 않고 오히려 {ratio:.2f}배로 커진다** "
      f"(전체 {gapf:.2f}배). 즉 남은 격차는 **가격 구성 때문이 아니다.** "
      "동시에 이것은 '가격을 낮춰 전환을 올린다'는 전략의 기대 효과가 "
      "낮다는 §1-4 의 관찰과도 일치한다.")
    A("")
    A("**결론: 격차는 구성(composition)이 아니라 행동·구조(behavior)에서 온다.** "
      "가장 유력한 후보는 §1-2 의 담기 방식 차이다 — electronics 는 조회 후 담는 "
      "구매 의도의 표현이지만, cosmetics 의 cart 는 상당 부분 **위시리스트·발주 "
      "메모**로 쓰인다. 이건 가격 정책으로 좁힐 수 있는 격차가 아니다.")
    A("")

    A("### 3-3. Step 3 — cosmetics 내부의 고전환 구간")
    A("")
    A("외부 벤치마크보다 강한 증거다. **동일 스토어 안에서 이미 electronics "
      "수준을 내는 구간**이 있다면, 그 조건이 곧 타겟팅 전략이 된다.")
    A("")
    A("| 구매이력(cart 시점 이전) | 수량 | cart_line | 전환율 | 회수 가능 |")
    A("|---|---|---:|---:|---:|")
    for r in pockets.iter_rows(named=True):
        A(f"| {r['prior_tier']} | {'다수량' if r['multi_qty'] else '단일'} | "
          f"{_f(r['cart_lines'])} | {_p(r['cvr'])} | {_f(r['headroom'])} |")
    A("")
    A("> ⚠️ **누수 주의**: `구매이력` 은 반드시 **cart 시점 t 이전** 누적 구매만 "
      "센다. 패널 전체 구매 수를 쓰면 '미구매 유저의 전환율 = 0' 이라는 "
      "동어반복이 나온다(실제로 초안에서 발생했고 수정함).")
    A("")
    A("**두 가지 발견:**")
    A("")
    A("1. ⭐ **다수량 담기의 효과는 모든 구매이력 구간에서 일관된다** "
      "(1.5~1.8배). 구간마다 반복되므로 **구성 효과가 아니라 담기 행동 자체의 "
      "신호**로 볼 근거가 된다. 상위 구간은 51.2% 로 **electronics 벤치마크"
      f"({_p(ele['cvr'])})에 도달**한다.")
    A("2. **구매이력의 효과는 그보다 약하다** (20.4% → 27~31%). 즉 '기존 구매자를 "
      "타겟하라' 보다 **'담는 방식이 진지한 cart 를 식별하라'** 가 더 강한 신호다.")
    A("")
    A("![회수 가능 물량 맵](figures/03_headroom_map.png)")
    A("")

    A("### 3-4. 담기 방식 — 조회 선행 여부")
    A("")
    A("| 스토어 | 같은 세션 조회 선행 | cart_line | 전환율 |")
    A("|---|---|---:|---:|")
    for s in STORES:
        for r in d2c[s].iter_rows(named=True):
            A(f"| {s} | {'예' if r['viewed_first'] else '아니오'} | "
              f"{_f(r['cart_lines'])} | {_p(r['cvr'])} |")
    A("")
    A("cosmetics 에서는 조회를 선행한 cart 의 전환율이 **29.1% vs 22.2%** 로 "
      "약 7%p 높다. 물량이 90만 건이라 무시할 수 없는 구간이다. "
      "electronics 는 방향이 반대지만 비조회 표본이 1,000건뿐이라 해석하지 않는다.")
    A("")

    A("### 3-5. 전환 타이밍 — 리마인더는 언제 보내야 하는가")
    A("")
    A("![전환 타이밍](figures/02_time_to_purchase.png)")
    A("")
    A("| 경과 | cosmetics 누적 전환율 | electronics 누적 전환율 |")
    A("|---|---:|---:|")
    tc = {r["hours"]: r for r in ttp["cosmetics"].iter_rows(named=True)}
    te = {r["hours"]: r for r in ttp["electronics"].iter_rows(named=True)}
    for h in [1, 6, 24, 72, 168, 720]:
        lbl = {1: "1시간", 6: "6시간", 24: "24시간", 72: "3일", 168: "7일", 720: "30일"}[h]
        A(f"| {lbl} | {_p(tc[h]['cum_rate'])} | {_p(te[h]['cum_rate'])} |")
    A("")
    share24 = tc[24]["cum_rate"] / tc[168]["cum_rate"]
    A(f"> **7일 전환의 {share24:.0%}가 24시간 안에 끝난다.** 1시간 시점에 이미 "
      f"{tc[1]['cum_rate']/tc[168]['cum_rate']:.0%} 가 완료된다. 개입의 여지는 "
      "**담긴 직후 몇 시간~하루** 구간에 몰려 있고, 그 뒤 위험률은 빠르게 꺼진다. "
      "Phase 5 의 생존분석에서 세그먼트별로 정밀화한다.")
    A("")

    # ══ 결론 ══
    A("## 4. 결론 및 Phase 2 이관")
    A("")
    A("### 확정된 사실")
    A("")
    A(f"- cosmetics cart→purchase 전환율(7일) = **{_p(cos['cvr'])}**, "
      f"electronics = **{_p(ele['cvr'])}**, 격차 **{gapf:.2f}배**")
    A("- 격차는 **가격 구성으로 설명되지 않음** — 공통 가격대에서 오히려 확대")
    A(f"- 두 스토어의 근본 차이는 **담기 방식**: 담기 이전 조회 비율 "
      f"{vf['cosmetics']['cart_lines'][0]/cos['cart_lines']:.1%} vs "
      f"{vf['electronics']['cart_lines'][0]/ele['cart_lines']:.1%}")
    A("- cosmetics 내부에 **electronics 수준(51%)의 구간이 이미 존재** (다수량 담기)")
    A("- 전환 타이밍이 매우 앞쪽에 몰림 → 리마케팅 창은 **수 시간 단위**")
    A("")
    A("### 계획 수정")
    A("")
    A("| # | 내용 |")
    A("|---|---|")
    A("| **F** | **B2B 이분법 폐기.** H0 는 바스켓 구조에서는 지지되나 시간 "
      "프로파일에서 반증됨. 세그멘테이션 축을 추정 신원이 아닌 **관측 행동**"
      "(수량·조회선행·재구매주기·바스켓크기)으로 확정 |")
    A("| **G** | **`cart_repeat_cnt`(다수량 담기)를 Phase 3 최우선 피처로 승격.** "
      "구매이력보다 강한 신호이며 cart 시점에 관측 가능해 누수가 없음 |")
    A("| **H** | **`viewed_first`(조회 선행) 피처 신설** — §3-4 |")
    A("| **I** | 가격 관련 전략의 기대 효과를 낮게 잡는다. cosmetics 전환율은 "
      "가격대에 거의 무반응(§1-4)이고 가격 통제 후에도 격차가 남음(§3-2) |")
    A("| **J** | **`price_missing` 은 실행 가능한 MD 액션으로 승격** — 전환율 "
      "6분의 1(§1-5). 리스팅 품질 관리 |")
    A("")
    A("### 다음 단계 — Phase 2")
    A("")
    A("cart_line 테이블은 이미 구축되어 있다(`src/sessions.py`). Phase 2 에서는 "
      "3-way 결과 라벨(구매 / 명시적 제거 / 조용한 이탈)을 확정하고, "
      "귀속 윈도우 민감도와 우측 절단 처리를 정식화한다.")
    A("")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "01_funnel_baseline.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
