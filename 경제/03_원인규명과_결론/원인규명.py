"""
===============================================================================
REES46 Cosmetics — PPT 15~18페이지 분석 및 시각화 코드
===============================================================================

입력
data/cosmetics_train.csv

===============================================================================
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# =============================================================================
# 0. 설정
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_15_18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATA_SOURCE: str | None = None
DPI = 300


DARK = "#102A6A"
INDIGO = "#5848B2"
PURPLE = "#8177C4"
LAVENDER = "#AEB5D1"
WHITE = "#FFFFFF"


def set_korean_font() -> None:
    candidates = [
        "Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic",
        "Noto Sans CJK KR", "Noto Sans KR", "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


def clean_axes(ax: plt.Axes, *, remove_x: bool = True, remove_y: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=DARK)
    if remove_x:
        ax.set_xticks([])
    if remove_y:
        ax.set_yticks([])


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(
        path,
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"[저장] {path}")


# =============================================================================
# 1. 데이터 로딩
# =============================================================================
def find_data_source() -> Path:
    if DATA_SOURCE:
        path = Path(DATA_SOURCE)
        if not path.exists():
            raise FileNotFoundError(f"DATA_SOURCE가 존재하지 않습니다: {path}")
        return path

    candidates = [
        BASE_DIR / "data" / "cosmetics_train.csv",
        BASE_DIR / "cosmetics_train.csv",
        BASE_DIR / "data_filtering.zip",
        Path("/mnt/data/data_filtering.zip"),
        Path("/mnt/data/data/cosmetics_train.csv"),
        Path("/mnt/data/data_filtering_extracted/data/cosmetics_train.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "cosmetics_train.csv 또는 data_filtering.zip을 찾지 못했습니다. "
        "DATA_SOURCE에 파일 경로를 지정하세요."
    )


def resolve_csv(source: Path) -> Path:
    if source.suffix.lower() == ".csv":
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError(f"지원하지 않는 입력 형식입니다: {source}")

    csv_path = BASE_DIR / "data" / "cosmetics_train.csv"
    if csv_path.exists():
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    member = "data/cosmetics_train.csv"
    print(f"[압축 해제] {source} -> {csv_path}")
    with zipfile.ZipFile(source) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"압축 파일에 {member}가 없습니다.")
        with zf.open(member) as src, open(csv_path, "wb") as dst:
            while block := src.read(16 * 1024 * 1024):
                dst.write(block)
    return csv_path


def load_data(csv_path: Path) -> pd.DataFrame:
    cols = [
        "user_hash", "outcome", "y_7d", "u_never_purchased",
        "viewed_first", "up_bought_before", "up_carted_before",
        "price", "p_prior_cvr",
    ]
    dtype = {
        "user_hash": "category",
        "outcome": "category",
        "y_7d": "boolean",
        "u_never_purchased": "boolean",
        "viewed_first": "boolean",
        "up_bought_before": "boolean",
        "up_carted_before": "boolean",
        "price": "float32",
        "p_prior_cvr": "float32",
    }
    print(f"[데이터 로딩] {csv_path}")
    df = pd.read_csv(csv_path, usecols=cols, dtype=dtype)

    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")

    
    ever_buy_users = set(
        df.loc[df["outcome"].eq("purchased"), "user_hash"].dropna().unique()
    )
    df["ever_buyer"] = df["user_hash"].isin(ever_buy_users)
    return df


# =============================================================================
# 2. 15페이지 — 가격대·상품 과거 CVR별 전환율
# =============================================================================
def analyze_slide15(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    price_band = pd.cut(
        df["price"],
        bins=[0, 2, 4, 7, 15, np.inf],
        labels=["~$2", "$2–4", "$4–7", "$7–15", "$15+"],
    )
    price_cvr = df.groupby(price_band, observed=True)["y_7d"].mean() * 100

    product_band = pd.qcut(
        df["p_prior_cvr"].clip(0, 1),
        q=[0, 0.50, 0.80, 0.95, 1.00],
        labels=["하위 50%", "중", "상", "최상위 5%"],
        duplicates="drop",
    )
    product_cvr = df.groupby(product_band, observed=True)["y_7d"].mean() * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(price_cvr.index.astype(str), price_cvr.values,
                  color=LAVENDER, width=0.56)
    ax.bar_label(
        bars, labels=[f"{v:.0f}%" for v in price_cvr.values], padding=4,
        color=DARK, fontsize=16, fontweight="bold",
    )
    ax.set_ylim(0, 35)
    clean_axes(ax, remove_x=False)
    ax.tick_params(axis="x", labelsize=13, colors=DARK, pad=10)
    save_figure(fig, "slide15_price_cvr.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(product_cvr.index.astype(str), product_cvr.values,
                  color=INDIGO, width=0.56)
    ax.bar_label(
        bars, labels=[f"{v:.0f}%" for v in product_cvr.values], padding=4,
        color=DARK, fontsize=16, fontweight="bold",
    )
    ax.set_ylim(0, 38)
    clean_axes(ax, remove_x=False)
    ax.tick_params(axis="x", labelsize=13, colors=DARK, pad=10)
    save_figure(fig, "slide15_product_cvr.png")

    print("\n[15페이지] 가격대별 전환율")
    print(price_cvr.round(2).to_string())
    print("\n[15페이지] 상품 과거 CVR 구간별 전환율")
    print(product_cvr.round(2).to_string())

    metrics = {f"slide15_price_{k}": float(v) for k, v in price_cvr.items()}
    metrics.update({f"slide15_product_{k}": float(v) for k, v in product_cvr.items()})
    return price_cvr, product_cvr, metrics


# =============================================================================
# 3. 16페이지 — 전체 기간 사후 관점·담은 시점 사전 관점
# =============================================================================
def analyze_slide16(df: pd.DataFrame) -> Dict[str, float]:
    baseline = float(df["y_7d"].mean() * 100)

    # 사후적 관점: 전체 기간의 최종 구매 여부로 cart row를 분해
    final_never_mask = ~df["ever_buyer"]
    share_never = float(final_never_mask.mean() * 100)
    share_buyer = 100 - share_never
    cvr_final_never = float(df.loc[final_never_mask, "y_7d"].mean() * 100)
    cvr_final_buyer = float(df.loc[~final_never_mask, "y_7d"].mean() * 100)

    fig, ax = plt.subplots(figsize=(9.5, 2.45))
    ax.barh([0], [share_never], color=LAVENDER, height=0.45)
    ax.barh([0], [share_buyer], left=[share_never], color=INDIGO, height=0.45)
    ax.text(
        share_never / 2, 0,
        f"전체기간 무구매 유저\n{share_never:.1f}%",
        ha="center", va="center", color=DARK, fontsize=17, fontweight="bold",
    )
    ax.text(
        share_never + share_buyer / 2, 0,
        f"구매 경험 유저\n{share_buyer:.1f}%",
        ha="center", va="center", color=WHITE, fontsize=17, fontweight="bold",
    )
    ax.text(share_never / 2, -0.47, f"전환율 {cvr_final_never:.1f}%",
            ha="center", color=DARK, fontsize=14, fontweight="bold")
    ax.text(share_never + share_buyer / 2, -0.47,
            f"전환율 {cvr_final_buyer:.1f}%",
            ha="center", color=DARK, fontsize=14, fontweight="bold")
    ax.text(50, 0.62, f"전체 평균 {baseline:.1f}%",
            ha="center", color=DARK, fontsize=19, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 0.85)
    clean_axes(ax)
    save_figure(fig, "slide16_retrospective_mix.png")

    # 사전적 관점: 담은 시점까지의 구매 경험 여부
    cvr_prior_never = float(
        df.loc[df["u_never_purchased"], "y_7d"].mean() * 100
    )
    cvr_prior_buyer = float(
        df.loc[~df["u_never_purchased"], "y_7d"].mean() * 100
    )
    gap = cvr_prior_buyer - cvr_prior_never

    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    ax.hlines(0, cvr_prior_never, cvr_prior_buyer, color=LAVENDER, lw=4)
    ax.scatter([cvr_prior_never], [0], s=1800, color=LAVENDER, zorder=3)
    ax.scatter([cvr_prior_buyer], [0], s=1800, color=INDIGO, zorder=3)
    ax.text(cvr_prior_never, 0.38, f"{cvr_prior_never:.1f}%",
            ha="center", color=DARK, fontsize=20, fontweight="bold")
    ax.text(cvr_prior_buyer, 0.38, f"{cvr_prior_buyer:.1f}%",
            ha="center", color=DARK, fontsize=20, fontweight="bold")
    ax.text((cvr_prior_never + cvr_prior_buyer) / 2, 0.08,
            f"+{gap:.1f}%p", ha="center", color=INDIGO,
            fontsize=18, fontweight="bold")
    ax.text(cvr_prior_never, -0.45, "아직 구매 경험 없음",
            ha="center", color=DARK, fontsize=14, fontweight="bold")
    ax.text(cvr_prior_buyer, -0.45, "이미 구매 경험 있음",
            ha="center", color=DARK, fontsize=14, fontweight="bold")
    ax.set_xlim(18, 32)
    ax.set_ylim(-0.7, 0.7)
    clean_axes(ax)
    save_figure(fig, "slide16_prospective_gap.png")

    print("\n[16페이지]")
    print(f"전체 평균: {baseline:.2f}%")
    print(f"사후 관점 구성: {share_never:.2f}% vs {share_buyer:.2f}%")
    print(f"사후 관점 전환율: {cvr_final_never:.2f}% vs {cvr_final_buyer:.2f}%")
    print(f"사전 관점 전환율: {cvr_prior_never:.2f}% vs {cvr_prior_buyer:.2f}% (+{gap:.2f}%p)")

    return {
        "slide16_baseline": baseline,
        "slide16_final_never_share": share_never,
        "slide16_final_buyer_share": share_buyer,
        "slide16_final_never_cvr": cvr_final_never,
        "slide16_final_buyer_cvr": cvr_final_buyer,
        "slide16_prior_never_cvr": cvr_prior_never,
        "slide16_prior_buyer_cvr": cvr_prior_buyer,
        "slide16_prior_gap_pp": gap,
    }


# =============================================================================
# 4. 17페이지 — 과거 행동 이력별 전환율
# =============================================================================
def analyze_slide17(df: pd.DataFrame) -> Dict[str, float]:
    baseline = float(df["y_7d"].mean() * 100)
    rates = pd.Series({
        "이 상품을 전에 구매": float(
            df.loc[df["up_bought_before"], "y_7d"].mean() * 100
        ),
        "이 상품을 전에 담음": float(
            df.loc[df["up_carted_before"], "y_7d"].mean() * 100
        ),
        "담기 전 조회": float(
            df.loc[df["viewed_first"], "y_7d"].mean() * 100
        ),
    })

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(rates))
    bars = ax.barh(y, rates.values,
                   color=[INDIGO, PURPLE, LAVENDER], height=0.46)
    ax.invert_yaxis()
    ax.axvline(baseline, color=DARK, lw=2, ls=(0, (4, 4)))
    ax.text(baseline, -0.72, f"전체 기준 {baseline:.1f}%",
            ha="center", color=DARK, fontsize=13, fontweight="bold")

    for i, (label, value) in enumerate(rates.items()):
        ax.text(-0.8, i, label, ha="right", va="center",
                color=DARK, fontsize=15, fontweight="bold")
        ax.text(value + 1.2, i, f"{value:.1f}%", ha="left", va="center",
                color=DARK, fontsize=15, fontweight="bold")
        ax.text(61, i, f"{value / baseline:.1f}×", ha="right", va="center",
                color=INDIGO, fontsize=15, fontweight="bold")

    ax.text(61, -0.72, "기준 대비", ha="right", color=LAVENDER,
            fontsize=13, fontweight="bold")
    ax.set_xlim(-1, 62)
    ax.set_ylim(len(rates) - 0.35, -0.9)
    clean_axes(ax)
    save_figure(fig, "slide17_conversion_drivers.png")

    print("\n[17페이지]")
    print(f"전체 기준: {baseline:.2f}%")
    print(rates.round(2).to_string())

    return {
        "slide17_baseline": baseline,
        "slide17_prior_purchase": float(rates.iloc[0]),
        "slide17_prior_cart": float(rates.iloc[1]),
        "slide17_viewed_first": float(rates.iloc[2]),
    }


# =============================================================================
# 5. 18페이지 — 유저 구성·반복 담기 경험
# =============================================================================
def analyze_slide18(df: pd.DataFrame) -> Dict[str, float]:
    total_users = int(df["user_hash"].nunique())
    buyer_users = int(
        df.loc[df["outcome"].eq("purchased"), "user_hash"].nunique()
    )
    never_users = total_users - buyer_users
    never_share = never_users / total_users * 100
    buyer_share = buyer_users / total_users * 100

    fig, ax = plt.subplots(figsize=(10.2, 2.25))
    ax.barh([0], [never_share], color=INDIGO, height=0.52)
    ax.barh([0], [buyer_share], left=[never_share],
            color=LAVENDER, height=0.52)
    ax.text(never_share / 2, 0, f"{never_share:.1f}%\n{never_users:,}명",
            ha="center", va="center", color=WHITE,
            fontsize=18, fontweight="bold")
    ax.text(never_share + buyer_share / 2, 0,
            f"{buyer_share:.1f}%\n{buyer_users:,}명",
            ha="center", va="center", color=DARK,
            fontsize=18, fontweight="bold")
    ax.text(never_share / 2, -0.52, "담았지만 5개월 내 무구매",
            ha="center", color=DARK, fontsize=13, fontweight="bold")
    ax.text(never_share + buyer_share / 2, -0.52, "구매 경험 유저",
            ha="center", color=DARK, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 0.65)
    clean_axes(ax)
    save_figure(fig, "slide18_user_mix.png")

    # cart 시점까지 구매 경험 여부별 동일 상품 과거 담기 경험률
    repeat_never = float(
        df.loc[df["u_never_purchased"], "up_carted_before"].mean() * 100
    )
    repeat_buyer = float(
        df.loc[~df["u_never_purchased"], "up_carted_before"].mean() * 100
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = ["무구매 유저", "구매 경험 유저"]
    vals = [repeat_never, repeat_buyer]
    bars = ax.bar(labels, vals, color=[LAVENDER, INDIGO], width=0.48)
    ax.bar_label(
        bars, labels=[f"{v:.1f}%" for v in vals], padding=4,
        color=DARK, fontsize=17, fontweight="bold",
    )
    ax.set_ylim(0, 18)
    clean_axes(ax, remove_x=False)
    ax.tick_params(axis="x", labelsize=13, colors=DARK, pad=10)
    save_figure(fig, "slide18_repeat_cart_gap.png")

    print("\n[18페이지]")
    print(f"전체 유저: {total_users:,}명")
    print(f"담았지만 무구매: {never_users:,}명 ({never_share:.2f}%)")
    print(f"구매 경험: {buyer_users:,}명 ({buyer_share:.2f}%)")
    print(f"반복 담기 경험: {repeat_never:.2f}% vs {repeat_buyer:.2f}%")

    return {
        "slide18_total_users": total_users,
        "slide18_never_users": never_users,
        "slide18_buyer_users": buyer_users,
        "slide18_never_share": never_share,
        "slide18_buyer_share": buyer_share,
        "slide18_repeat_never": repeat_never,
        "slide18_repeat_buyer": repeat_buyer,
    }


# =============================================================================
# 6. 전체 실행
# =============================================================================
def main() -> None:
    source = find_data_source()
    csv_path = resolve_csv(source)
    df = load_data(csv_path)

    print(f"[행 수] {len(df):,}")
    print(f"[유저 수] {df['user_hash'].nunique():,}")
    print(f"[전체 7일 전환율] {df['y_7d'].mean() * 100:.2f}%")

    all_metrics: Dict[str, float] = {}
    _, _, m15 = analyze_slide15(df)
    all_metrics.update(m15)
    all_metrics.update(analyze_slide16(df))
    all_metrics.update(analyze_slide17(df))
    all_metrics.update(analyze_slide18(df))

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in all_metrics.items()]
    ).to_csv(
        OUTPUT_DIR / "metrics_15_18.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n[완료] 결과 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[오류] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
