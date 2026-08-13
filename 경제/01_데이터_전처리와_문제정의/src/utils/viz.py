"""Phase 1 그림. 정적 PNG 이므로 호버 레이어는 해당 없음.

색은 검증된 기본 팔레트에서 카테고리 슬롯 1(blue)·2(aqua)를 고정 순서로 쓴다
(`validate_palette.js --mode light` 통과, 인접 CVD ΔE 73.6). aqua 는 밝은
표면에서 대비 2.74:1 로 3:1 미만이라 **relief 규칙**이 적용된다 → 모든 계열에
직접 라벨을 단다.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import FIGURES

# ── 팔레트 (light) ─────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"
SERIES = {"cosmetics": "#2a78d6", "electronics": "#1baf7a"}
ACCENT = "#e34948"  # 참조선용(계열 색과 구분되는 슬롯 6)

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
    "font.size": 11,
})


def _clean(ax, *, grid_axis="y"):
    """축을 후퇴시킨다 — 데이터가 주인공."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return p


# ── Fig 1. 정렬 사다리 ─────────────────────────────────────────────────
def fig_alignment_ladder(ladders: dict, name="01_alignment_ladder"):
    """정의를 한 단계씩 정렬할 때 두 스토어 격차가 어떻게 좁혀지는가.

    형태: 단계별 그룹 수평 막대 + 우측에 배율 주석. 배율이 헤드라인이므로
    막대만으로 끝내지 않고 숫자를 직접 붙인다.
    """
    stages = ladders["cosmetics"]["stage"]
    cos = np.array(ladders["cosmetics"]["cvr"], dtype=float)
    ele = np.array(ladders["electronics"]["cvr"], dtype=float)

    y = np.arange(len(stages))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(11, 0.86 * len(stages) + 1.6))

    ax.barh(y + h / 2 + 0.02, ele, h, color=SERIES["electronics"], zorder=3)
    ax.barh(y - h / 2 - 0.02, cos, h, color=SERIES["cosmetics"], zorder=3)

    for yy, v in zip(y + h / 2 + 0.02, ele):
        ax.text(v + 0.008, yy, f"{v:.1%}", va="center", fontsize=10, color=INK2)
    for yy, v in zip(y - h / 2 - 0.02, cos):
        ax.text(v + 0.008, yy, f"{v:.1%}", va="center", fontsize=10, color=INK2)

    xmax = max(ele.max(), cos.max())
    for yy, c, e in zip(y, cos, ele):
        ax.text(xmax * 1.30, yy, f"{e / c:.2f}배", va="center", ha="right",
                fontsize=11, color=INK, fontweight="bold")
    ax.text(xmax * 1.30, y[0] + 0.75, "격차", va="center", ha="right",
            fontsize=10, color=MUTED)

    ax.set_yticks(y, stages, fontsize=10.5)
    ax.set_xlim(0, xmax * 1.34)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(ax, grid_axis="x")
    ax.set_xlabel("cart → purchase 전환율")
    ax.set_title("정의를 정렬할수록 격차는 줄어든다 — 3.93배에서 2.20배로",
                 fontsize=13.5, color=INK, pad=14, loc="left")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=SERIES["cosmetics"], label="cosmetics"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES["electronics"], label="electronics"),
    ], frameon=False, fontsize=10, ncol=2,
        loc="lower left", bbox_to_anchor=(0, -0.20 / len(stages) - 0.12))
    return _save(fig, name)


# ── Fig 2. cart 이후 경과시간별 누적 전환 ──────────────────────────────
def fig_time_to_purchase(curves: dict, name="02_time_to_purchase"):
    """리마인더를 언제 보내야 하는가에 직접 답하는 그림."""
    fig, ax = plt.subplots(figsize=(10, 5.4))

    for store, df in curves.items():
        x = np.array(df["hours"], dtype=float)
        yv = np.array(df["cum_rate"], dtype=float)
        ax.plot(x, yv, color=SERIES[store], linewidth=2, marker="o",
                markersize=5.5, zorder=3)
        ax.text(x[-1] * 1.06, yv[-1], store, color=INK2, fontsize=11,
                va="center", fontweight="bold")

    ax.axvline(24, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    # 축 좌표로 배치 — 데이터 좌표를 쓰면 로그축·자동 y범위와 어긋나 축 밖으로 나간다.
    ax.text(24, 0.02, " 24시간", color=MUTED, fontsize=10,
            transform=ax.get_xaxis_transform(), va="bottom")

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 6, 12, 24, 72, 168, 720],
                  ["1h", "3h", "6h", "12h", "24h", "3d", "7d", "30d"])
    ax.set_xlim(0.85, 2600)  # 우측 직접 라벨이 들어갈 여백
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(ax)
    ax.set_xlabel("cart 이후 경과 시간 (로그 축)")
    ax.set_ylabel("누적 전환율", labelpad=10)
    ax.set_title("전환은 초반에 몰린다 — cosmetics 는 7일 전환의 83%가 24시간 안에 끝난다",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 3. Headroom 맵 ────────────────────────────────────────────────
def fig_headroom(seg, benchmark: float, name="03_headroom_map"):
    """회수 가능 물량 순위.

    형태 선택: 산점도(물량×전환율)를 먼저 시도했으나 세그먼트 6개 중 3개가
    한 지점에 뭉쳐 라벨이 겹쳤다. 이 그림의 실제 용도는 **우선순위 매기기**이므로
    순위형 수평 막대가 맞는 형태다. 크기 비교가 정확해지고 라벨 충돌도 사라진다.
    전환율·물량은 각 막대에 직접 라벨로 붙여 정보를 잃지 않는다.
    """
    hr = np.array(seg["headroom"], dtype=float)
    cvr = np.array(seg["cvr"], dtype=float)
    n = np.array(seg["cart_lines"], dtype=float)
    labels = list(seg["label"])

    order = np.argsort(hr)  # 수평 막대는 아래에서 위로 그려진다
    hr, cvr, n = hr[order], cvr[order], n[order]
    labels = [labels[i] for i in order]
    y = np.arange(len(hr))

    fig, ax = plt.subplots(figsize=(11, 0.72 * len(hr) + 2.0))
    ax.barh(y, hr / 1000, height=0.62, color=SERIES["cosmetics"], zorder=3)

    for yy, h, c, nn in zip(y, hr, cvr, n):
        ax.text(h / 1000 + hr.max() / 1000 * 0.02, yy,
                f"{h/1000:,.0f}K   ·   전환율 {c:.1%}   ·   cart {nn/1000:,.0f}K",
                va="center", fontsize=10, color=INK2)

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlim(0, hr.max() / 1000 * 1.62)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}K")
    _clean(ax, grid_axis="x")
    # U+2212(−)는 AppleGothic 에 글리프가 없어 두부로 깨진다 → ASCII 하이픈.
    ax.set_xlabel("회수 가능 물량 = cart_line x (1 - 전환율)")
    ax.set_title(f"기회는 전환율이 아니라 물량에 있다 — electronics 벤치마크는 {benchmark:.0%}",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 4. 단변량 신호 강도 ────────────────────────────────────────────
def fig_univariate_auc(df, name="04_univariate_auc", top=18):
    """피처 하나하나가 얼마나 약한지 보여주는 그림.

    형태: |AUC - 0.5| 순위 막대. 0.5 가 무신호이므로 편차 자체가 신호 강도다.
    관례적 기준선(0.6 = 약, 0.7 = 쓸만)을 같이 그려 절대적 위치를 보게 한다.
    """
    d = df.head(top)
    labels = list(d["feature"])[::-1]
    auc = np.array(d["auc"], dtype=float)[::-1]
    y = np.arange(len(labels))
    dev = np.abs(auc - 0.5)

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(labels) + 2.0))
    ax.barh(y, dev, height=0.62, color=SERIES["cosmetics"], zorder=3)
    for yy, dv, a in zip(y, dev, auc):
        arrow = "↑" if a > 0.5 else "↓"
        ax.text(dv + dev.max() * 0.02, yy, f"AUC {a:.3f} {arrow}",
                va="center", fontsize=9.5, color=INK2)

    for x, lbl in [(0.10, "AUC 0.60 — 약함"), (0.20, "AUC 0.70 — 쓸만함")]:
        ax.axvline(x, color=ACCENT, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
        ax.text(x, len(labels) - 0.3, f" {lbl}", color=ACCENT, fontsize=9.5, va="top")

    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlim(0, 0.235)
    _clean(ax, grid_axis="x")
    ax.set_xlabel("신호 강도 = |AUC - 0.5|")
    ax.set_title("단변량 신호는 전부 약하다 — 최강 피처도 AUC 0.565",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 5. 효과 크기 × 도달 범위 ───────────────────────────────────────
def fig_lift_reach(df, name="05_lift_reach"):
    """lift 만 보면 판단을 그르친다는 것을 한 장으로.

    형태: 증분 상한(개수) 순위 막대 + lift·도달범위 직접 라벨. 크기 비교가
    목적이므로 막대가 맞고, 두 보조 지표는 라벨로 붙여 정보를 잃지 않는다.
    """
    d = df.sort("potential")
    labels = list(d["flag"])
    pot = np.array(d["potential"], dtype=float)
    lift = np.array(d["lift"], dtype=float)
    reach = np.array(d["reach"], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) + 2.2))
    ax.barh(y, pot / 1000, height=0.6, color=SERIES["cosmetics"], zorder=3)
    for yy, p, lf, rc in zip(y, pot, lift, reach):
        ax.text(p / 1000 + pot.max() / 1000 * 0.02, yy,
                f"{p/1000:,.0f}K   ·   효과 {lf:.2f}배   ·   해당 물량 {rc:.1%}",
                va="center", fontsize=10, color=INK2)

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlim(0, pot.max() / 1000 * 1.75)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}K")
    _clean(ax, grid_axis="x")
    ax.set_xlabel("증분 구매 개수 상한 (연관성 기준 — 달성 가능치가 아님)")
    ax.set_title("효과가 커도 해당 물량이 적으면 총량은 못 움직인다",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 6. SHAP 중요도 ─────────────────────────────────────────────────
def fig_shap(df, name="06_shap_importance", top=15):
    """모델이 실제로 무엇을 보고 있는가. 방향은 라벨로 붙인다."""
    d = df.head(top)
    labels = list(d["feature"])[::-1]
    val = np.array(d["mean_abs_shap"], dtype=float)[::-1]
    dirn = np.array(d["direction"], dtype=float)[::-1]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 2.0))
    ax.barh(y, val, height=0.62, color=SERIES["cosmetics"], zorder=3)
    for yy, v, dv in zip(y, val, dirn):
        if abs(dv) < 0.15:
            lab = "비단조/상호작용"
        else:
            lab = "높을수록 전환 ↑" if dv > 0 else "높을수록 전환 ↓"
        ax.text(v + val.max() * 0.02, yy, lab, va="center", fontsize=9.5, color=INK2)

    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlim(0, val.max() * 1.55)
    _clean(ax, grid_axis="x")
    ax.set_xlabel("평균 |SHAP| (로그오즈 기여)")
    ax.set_title("모델이 보는 것 — 유저의 과거 전환 습관과 담기 방식",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 7. 누적 이득 곡선 ──────────────────────────────────────────────
def fig_gains(df, name="07_gains_curve"):
    """상위 몇 %를 타겟하면 전환의 몇 %를 잡는가 — 타겟팅의 한계를 보여준다."""
    x = np.concatenate([[0], np.array(df["top_pct"], dtype=float)])
    yv = np.concatenate([[0], np.array(df["captured"], dtype=float) * 100])

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.5,
            linestyle=(0, (5, 4)), zorder=2)
    ax.text(72, 78, "무작위 타겟팅", color=MUTED, fontsize=10, rotation=32)
    ax.plot(x, yv, color=SERIES["cosmetics"], linewidth=2, marker="o",
            markersize=5.5, zorder=3)
    ax.text(x[-1] - 2, yv[-1] - 7, "모델", color=SERIES["cosmetics"],
            fontsize=11, fontweight="bold", ha="right")

    i10 = list(df["top_pct"]).index(10)
    c10 = float(df["captured"][i10]) * 100
    ax.plot([10, 10], [0, c10], color=ACCENT, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.plot([0, 10], [c10, c10], color=ACCENT, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"상위 10% 를 타겟해도\n전환의 {c10:.0f}% 만 잡힌다",
                (10, c10), textcoords="offset points", xytext=(16, -6),
                fontsize=10.5, color=ACCENT, linespacing=1.4)

    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _clean(ax)
    ax.set_xlabel("예측 점수 상위 비율 (타겟 대상)")
    ax.set_ylabel("잡아낸 전환 비율")
    ax.set_title("정밀 타겟팅은 벽에 부딪힌다 — 곡선이 대각선에서 멀리 못 간다",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 8. 규칙 세그먼트 ───────────────────────────────────────────────
def fig_rule_segments(df, benchmark: float, name="08_rule_segments"):
    """전환율과 물량의 상충 — 전환율 높은 칸은 작고, 큰 칸은 전환율이 낮다."""
    d = df.sort("cvr")
    labels = list(d["label"])
    cvr = np.array(d["cvr"], dtype=float)
    n = np.array(d["cart_lines"], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(labels) + 2.2))
    ax.barh(y, cvr, height=0.6, color=SERIES["cosmetics"], zorder=3)
    for yy, c, nn in zip(y, cvr, n):
        # 벤치마크 선 근처의 라벨은 선 너머로 밀어 겹침을 피한다
        x = c + 0.008
        if abs(c - benchmark) < 0.035:
            x = benchmark + 0.014
        ax.text(x, yy, f"{c:.1%}   ·   cart {nn/1000:,.0f}K",
                va="center", fontsize=10, color=INK2)

    ax.axvline(benchmark, color=ACCENT, linewidth=1.5, linestyle=(0, (5, 3)), zorder=2)
    ax.text(benchmark, len(labels) - 0.35, f" electronics {benchmark:.0%}",
            color=ACCENT, fontsize=9.5, va="top")

    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlim(0, max(cvr.max(), benchmark) * 1.45)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(ax, grid_axis="x")
    ax.set_xlabel("cart → purchase 전환율 (7일)")
    ax.set_title("전환율이 높은 세그먼트일수록 물량이 작다",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 9. 위험함수 ────────────────────────────────────────────────────
def fig_hazard(df, name="09_hazard"):
    """"아직 안 산 사람이 지금 살 확률" — 개입 시점을 고르는 데 직접 쓰인다.

    누적 전환율(Fig 2)과 다르다. 누적은 계속 올라가지만 위험률은 개입 여지가
    언제 사라지는지를 보여준다.

    축 선택: 로그 시간축 위에 막대를 그리면 **막대 폭이 구간 길이를 왜곡한다**
    (0-1h 막대가 5-7d 막대보다 넓게 보임). 구간 길이가 불균등하므로 등폭
    범주축을 쓰고, 불균등하다는 사실은 라벨과 캡션으로 알린다.
    """
    lo = np.array(df["from_h"], dtype=float)
    hi = np.array(df["to_h"], dtype=float)
    hz = np.array(df["hazard"], dtype=float)
    x = np.arange(len(hz))

    def _lab(a, b):
        f = lambda v: f"{v:.0f}h" if v < 24 else f"{v/24:.0f}d"
        return f"{f(a)}-{f(b)}"

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x, hz, width=0.72, color=SERIES["cosmetics"], zorder=3)
    ax.text(x[0], hz[0], f"{hz[0]:.1%}", va="bottom", ha="center",
            fontsize=12, color=INK, fontweight="bold")
    for xx, h in zip(x[1:], hz[1:]):
        ax.text(xx, h, f"{h:.1%}", va="bottom", ha="center", fontsize=9.5, color=INK2)

    ax.annotate("", xy=(1, hz[1] + 0.012), xytext=(0.08, hz[0] - 0.008),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    ax.text(1.15, hz[0] * 0.62, f"{hz[0]/hz[1]:.0f}분의 1로\n한 시간 만에 꺼진다",
            color=ACCENT, fontsize=11, linespacing=1.4)

    ax.set_xticks(x, [_lab(a, b) for a, b in zip(lo, hi)], fontsize=9.5)
    ax.set_ylim(0, hz.max() * 1.16)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(ax)
    ax.set_xlabel("cart 이후 경과 구간 (구간 길이는 서로 다름)")
    ax.set_ylabel("구간 위험률", labelpad=10)
    ax.set_title("개입의 창은 첫 1시간이다",
                 fontsize=13.5, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 10. pooled vs 유저 고정효과 ────────────────────────────────────
def fig_fe(df, name="10_within_user_fe"):
    """유저 고정효과를 걸면 계수가 얼마나 줄어드는가 = 유저 간 구성 효과의 크기."""
    d = df.sort("pooled")
    labels = list(d["term"])
    pooled = np.array(d["pooled"], dtype=float) * 100
    fe = np.array(d["fe"], dtype=float) * 100
    fese = np.array(d["fe_se"], dtype=float) * 100
    y = np.arange(len(labels))
    h = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 0.78 * len(labels) + 2.0))
    ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
    ax.barh(y + h / 2 + 0.02, pooled, h, color=SERIES["electronics"], zorder=3)
    ax.barh(y - h / 2 - 0.02, fe, h, color=SERIES["cosmetics"], zorder=3,
            xerr=1.96 * fese, error_kw=dict(ecolor=INK2, lw=1.2, capsize=3))

    for yy, v in zip(y + h / 2 + 0.02, pooled):
        ax.text(v + (0.25 if v >= 0 else -0.25), yy, f"{v:+.1f}pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=INK2)
    # FE 막대는 오차막대가 붙으므로 라벨을 그만큼 더 밀어야 겹치지 않는다
    for yy, v, e in zip(y - h / 2 - 0.02, fe, 1.96 * fese):
        off = e + 0.3
        ax.text(v + (off if v >= 0 else -off), yy, f"{v:+.1f}pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=INK2)

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlim(min(fe.min(), pooled.min()) - 4, max(fe.max(), pooled.max()) + 4)
    _clean(ax, grid_axis="x")
    ax.set_xlabel("전환율에 대한 효과 (퍼센트포인트) · 막대는 95% 신뢰구간")
    ax.set_title("유저 고정효과를 걸어도 수량 효과는 거의 안 줄어든다",
                 fontsize=13, color=INK, pad=14, loc="left")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=SERIES["electronics"], label="pooled (유저 간 차이 포함)"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES["cosmetics"], label="유저 고정효과 (같은 사람 안에서)"),
    ], frameon=False, fontsize=10, ncol=2, loc="lower left",
        bbox_to_anchor=(0, -0.20 / len(labels) - 0.14))
    return _save(fig, name)


# ── Fig 11. 반증 검정 ──────────────────────────────────────────────────
def fig_placebo(df, name="11_falsification"):
    """위약 결과(과거 구매)에도 같은 연관이 나오면 그만큼은 인과가 아니다."""
    labels = list(df["flag"])
    real = np.array(df["real_lift"], dtype=float)
    plac = np.array(df["placebo_lift"], dtype=float)
    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.axhline(1.0, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.text(0.995, 1.0, "효과 없음  ", color=MUTED, fontsize=10, ha="right",
            va="bottom", transform=ax.get_yaxis_transform())
    ax.bar(x - w / 2 - 0.01, real, w, color=SERIES["cosmetics"], zorder=3)
    ax.bar(x + w / 2 + 0.01, plac, w, color=ACCENT, zorder=3)

    for xx, v in zip(x - w / 2 - 0.01, real):
        ax.text(xx, v + 0.03, f"{v:.2f}배", ha="center", fontsize=10.5, color=INK2)
    for xx, v in zip(x + w / 2 + 0.01, plac):
        ax.text(xx, v + 0.03, f"{v:.2f}배", ha="center", fontsize=10.5, color=INK2)

    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylim(0, max(real.max(), plac.max()) * 1.25)
    _clean(ax)
    ax.set_ylabel("전환 리프트", labelpad=10)
    ax.set_title("위약 검정 실패 — 담기가 '과거' 구매까지 예측한다",
                 fontsize=13, color=INK, pad=14, loc="left")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=SERIES["cosmetics"], label="실제 (담기 이후 7일 구매)"),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT, label="위약 (담기 이전 7일 구매)"),
    ], frameon=False, fontsize=10, loc="upper left")
    return _save(fig, name)


# ── Fig 12. 임팩트 사이징 ──────────────────────────────────────────────
def fig_impact(rows, baseline: int, name="12_impact_sizing"):
    """방어 가능한 액션이 총량을 얼마나 움직이는가 — 기준선 대비로 보여준다.

    절대값만 보면 "5천 건"이 커 보인다. 현재 전환 113만 건 대비 몇 %인지를
    함께 붙여야 의사결정이 왜곡되지 않는다.
    """
    labels = [r["action"] for r in rows][::-1]
    lo = np.array([r["lo"] for r in rows], dtype=float)[::-1]
    mid = np.array([r["mid"] for r in rows], dtype=float)[::-1]
    hi = np.array([r["hi"] for r in rows], dtype=float)[::-1]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 0.85 * len(labels) + 2.2))
    ax.barh(y, mid, height=0.30, color=SERIES["cosmetics"], zorder=3)
    ax.set_ylim(-0.65, len(labels) - 0.35)
    ax.errorbar(mid, y, xerr=[mid - lo, hi - mid], fmt="none",
                ecolor=INK2, elinewidth=1.4, capsize=5, zorder=4)
    for yy, m, h in zip(y, mid, hi):
        ax.text(h + baseline * 0.004, yy, f"{m:,.0f}건  ·  현재 전환의 {m/baseline:.2%}",
                va="center", fontsize=10.5, color=INK2)

    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlim(0, max(hi) * 2.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:,.0f}K")
    _clean(ax, grid_axis="x")
    ax.set_xlabel("증분 구매 개수 (막대=기본 시나리오, 범위=보수~낙관)")
    ax.set_title(f"방어 가능한 액션은 총량을 거의 못 움직인다 — 현재 전환 {baseline/1e6:.2f}M 건",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 13. 검정력 곡선 ────────────────────────────────────────────────
def fig_power(raw, win, name="13_power_curve"):
    """실험을 며칠 돌려야 하는가. 윈저화의 효과를 함께 보여준다."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for df, color, lbl in [(raw, SERIES["electronics"], "원본 지표"),
                           (win, SERIES["cosmetics"], "p99 윈저화")]:
        x = np.array(df["rel_mde"], dtype=float) * 100
        d = np.array(df["months"], dtype=float) * 30.44
        ax.plot(x, d, color=color, linewidth=2, marker="o", markersize=6, zorder=3)
        ax.text(x[-1] + 0.25, d[-1], lbl, color=color, fontsize=10.5,
                va="center", fontweight="bold")

    ax.axhline(30, color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)), zorder=2)
    ax.text(0.99, 30, "1개월  ", color=MUTED, fontsize=10, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())

    ax.set_yscale("log")
    ax.set_yticks([5, 10, 30, 60, 120], ["5일", "10일", "30일", "60일", "120일"])
    ax.set_xlim(2.2, 18)
    _clean(ax)
    ax.set_xlabel("탐지하려는 상대 효과 크기 (MDE, %)")
    ax.set_ylabel("필요 기간 (2군 · 검정력 80%)", labelpad=10)
    ax.set_title("5% 효과를 잡으려면 약 3주 — 윈저화가 기간을 3분의 1 줄여준다",
                 fontsize=13, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 12(개정). 결론별 신뢰 지도 ─────────────────────────────────────
#: 순서형 신뢰 등급 → 단일 색상(blue) 램프. 밝은 표면에서 2:1 을 넘도록
#: step 250 이상만 쓴다(팔레트 규칙).
CONF_STEPS = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab"}
CONF_NAME = {1: "매우 약함", 2: "약함", 3: "중간", 4: "강함"}


def fig_confidence_map(rows, name="12_confidence_map"):
    """무엇을 얼마나 믿을 수 있는가 — 이 분석의 결론을 한 장에 정리.

    형태: 신뢰 등급이 순서형이므로 단일 색상 램프 + 막대 길이 이중 부호화.
    색만으로 등급을 나르지 않도록 등급명을 직접 라벨로 붙인다.
    """
    d = sorted(rows, key=lambda r: r["conf"])
    labels = [r["claim"] for r in d]
    conf = np.array([r["conf"] for r in d], dtype=float)
    notes = [r["evidence"] for r in d]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12.5, 0.62 * len(labels) + 2.2))
    ax.barh(y, conf, height=0.58, zorder=3,
            color=[CONF_STEPS[int(c)] for c in conf])
    for yy, c, nt in zip(y, conf, notes):
        ax.text(c + 0.07, yy, f"{CONF_NAME[int(c)]}  ·  {nt}",
                va="center", fontsize=10, color=INK2)

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xticks([1, 2, 3, 4], [CONF_NAME[i] for i in (1, 2, 3, 4)], fontsize=10)
    ax.set_xlim(0, 9.2)
    _clean(ax, grid_axis="x")
    # 라벨 영역 때문에 축이 넓다. xlabel 을 눈금 구간에 맞춰 왼쪽 정렬한다.
    ax.set_xlabel("결론의 신뢰 수준", loc="left")
    ax.set_title("이 분석이 무엇을 얼마나 주장할 수 있는가",
                 fontsize=13.5, color=INK, pad=14, loc="left")
    return _save(fig, name)


# ── Fig 13(개정). 주장이 검정을 통과한 경로 ────────────────────────────
def fig_gauntlet(stages, name="13_evidence_gauntlet"):
    """`multi_qty` 주장이 설계를 하나씩 통과하다 마지막에 무너지는 과정.

    단위 통일: 네 단계를 모두 **전환 리프트(배)** 로 환산해 같은 축에 놓는다.
    고정효과는 퍼센트포인트로 나오므로 단일수량 기저 전환율을 써서 리프트로
    바꾼다 — 그러지 않으면 서로 다른 단위를 한 축에 그리게 된다.
    """
    labels = [s["stage"] for s in stages][::-1]
    vals = np.array([s["value"] for s in stages], dtype=float)[::-1]
    ok = [s["pass"] for s in stages][::-1]
    notes = [s["note"] for s in stages][::-1]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12.5, 0.58 * len(labels) + 2.2))
    ax.axvline(1.0, color=MUTED, linewidth=1.3, linestyle=(0, (5, 4)), zorder=2)
    ax.text(1.0, 0.985, " 효과 없음", color=MUTED, fontsize=9.5, va="top",
            transform=ax.get_xaxis_transform())
    ax.barh(y, vals, height=0.44, zorder=3,
            color=[SERIES["cosmetics"] if o else ACCENT for o in ok])
    for yy, v, o, nt in zip(y, vals, ok, notes):
        # 이모지(✅/❌)는 AppleGothic 에 글리프가 없어 두부로 깨진다 → 텍스트로.
        tag = "통과" if o else "실패"
        col = SERIES["cosmetics"] if o else ACCENT
        ax.text(v + 0.04, yy, tag, va="center", fontsize=10.5,
                color=col, fontweight="bold")
        ax.text(v + 0.20, yy, nt, va="center", fontsize=10, color=INK2)

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlim(0, vals.max() * 1.95)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}배")
    _clean(ax, grid_axis="x")
    ax.set_xlabel("전환 리프트 (다수량 / 단일수량)")
    ax.set_title("`multi_qty` 주장은 세 검정을 통과하고 마지막에 무너졌다",
                 fontsize=13.5, color=INK, pad=14, loc="left")
    return _save(fig, name)
