from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import DATA, FIGURES, METRIC_LABELS, TABLES, configure_matplotlib, ensure_directories


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_composition(metrics: pd.DataFrame) -> None:
    all_years = metrics.loc[metrics["metric_included"] & metrics["year"].between(1950, 2026)].copy()
    annual = (
        all_years.groupby("year", observed=True)
        .agg(
            n=("precedent_id", "size"),
            third_share=("rule_court_instance", lambda s: (s == "third").mean()),
            median_tokens=("mp_tokens", "median"),
        )
        .reset_index()
    )
    group_counts = (
        all_years.loc[all_years["year"].between(1980, 2025)]
        .groupby(["year", "rule_crime_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    group_share = group_counts.div(group_counts.sum(axis=1), axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(annual["year"], annual["n"], color="#4C78A8", width=0.9)
    ax.axvspan(2025.5, 2026.5, color="#999999", alpha=0.18)
    ax.set(title="연도별 판결문 수", ylabel="건")
    ax = axes[0, 1]
    ax.plot(annual["year"], annual["third_share"] * 100, color="#F58518", lw=2)
    ax.set(title="3심 판결 비중", ylabel="%", ylim=(0, 105))
    ax = axes[1, 0]
    ax.plot(annual["year"], annual["median_tokens"], color="#54A24B", lw=2)
    ax.set(title="판결문당 형태소 수 중앙값", ylabel="형태소")
    ax = axes[1, 1]
    group_share.plot.area(ax=ax, stacked=True, alpha=0.85, linewidth=0)
    ax.set(title="범죄군 구성", ylabel="비중", xlabel="연도", ylim=(0, 1))
    ax.legend(title="범죄군", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle("F1. 분석 표본 구성과 길이 변화", fontsize=16)
    save(fig, "F1_composition")


def figure_main_trends(raw: pd.DataFrame, std: pd.DataFrame) -> None:
    panels = ["noun_share", "verb_share", "particle_share", "sent_eojeol_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharex=True)
    for ax, metric in zip(axes.flat, panels):
        r = raw.loc[raw["metric"] == metric].sort_values("year")
        s = std.loc[std["metric"] == metric].sort_values("year")
        scale = 100 if metric.endswith("share") else 1
        ax.plot(r["year"], r["mean"] * scale, color="#8C8C8C", lw=1.5, ls="--", label="원자료 평균")
        reliable = s["coverage"] >= 0.8
        ax.plot(s.loc[reliable, "year"], s.loc[reliable, "estimate"] * scale, color="#4C78A8", lw=2.4, label="구성 보정")
        ax.fill_between(
            s.loc[reliable, "year"].to_numpy(),
            (s.loc[reliable, "ci_low"] * scale).to_numpy(),
            (s.loc[reliable, "ci_high"] * scale).to_numpy(),
            color="#4C78A8",
            alpha=0.16,
        )
        ax.scatter(s.loc[~reliable, "year"], s.loc[~reliable, "estimate"] * scale, facecolors="none", edgecolors="#4C78A8", s=32, label="커버리지 <80%")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("%" if scale == 100 else "어절")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("F2. 판결문 문체의 원자료 추세와 사건 구성 보정 추세", fontsize=16)
    save(fig, "F2_main_trends")


def figure_small_multiples(std: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True, sharex=True)
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        g = std.loc[(std["metric"] == metric) & (std["coverage"] >= 0.8)].sort_values("year").copy()
        g["z"] = (g["estimate"] - g["estimate"].mean()) / g["estimate"].std(ddof=1)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.plot(g["year"], g["z"], color="#4C78A8", lw=2)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("연도 평균 대비 SD")
    fig.suptitle("F3. 보정된 문체 지표의 공통 척도 비교", fontsize=16)
    save(fig, "F3_metric_small_multiples")


def figure_heatmap(std: pd.DataFrame) -> None:
    usable = std.loc[std["coverage"] >= 0.8].copy()
    pivot = usable.pivot(index="metric", columns="year", values="estimate").reindex(list(METRIC_LABELS))
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1), axis=0)
    z.index = [METRIC_LABELS[x] for x in z.index]
    fig, ax = plt.subplots(figsize=(15, 5.6), constrained_layout=True)
    sns.heatmap(z, cmap="vlag", center=0, vmin=-2.5, vmax=2.5, ax=ax, cbar_kws={"label": "연도 평균 대비 SD"})
    ax.set(title="F4. 지표별 변화 시점", xlabel="연도", ylabel="")
    ticks = np.arange(0, len(z.columns), 5)
    ax.set_xticks(ticks + 0.5, [str(z.columns[i]) for i in ticks], rotation=0)
    save(fig, "F4_metric_heatmap")


def figure_variants(annual: pd.DataFrame, summary: pd.DataFrame) -> None:
    keys = summary["key"].tolist()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    for ax, key in zip(axes.flat, keys):
        g = annual.loc[(annual["key"] == key) & (annual["total"] >= 10)].sort_values("year").copy()
        if g.empty:
            continue
        g["smooth"] = g["old_share"].rolling(3, center=True, min_periods=1).mean()
        ax.scatter(g["year"], g["old_share"] * 100, s=np.clip(np.sqrt(g["total"]) * 1.5, 8, 50), alpha=0.35, color="#4C78A8")
        ax.plot(g["year"], g["smooth"] * 100, color="#4C78A8", lw=2.2)
        info = summary.loc[summary["key"] == key].iloc[0]
        if pd.notna(info["t50"]) and 1980 <= info["t50"] <= 2025:
            ax.axvline(info["t50"], color="#E45756", lw=1.3, ls="--")
            ax.text(info["t50"] + 0.5, 6, f"50%: {info['t50']:.0f}", color="#555555")
        ax.set_title(f"{info['label_old']} → {info['label_new']}")
        ax.set_ylabel("기존형 비율(%)")
        ax.set_ylim(0, 102)
    fig.suptitle("F5. 표현 변이쌍의 교체 추세", fontsize=16)
    save(fig, "F5_variant_transitions")


def figure_forest(trends: pd.DataFrame) -> None:
    g = trends.loc[trends["length_control"]].copy()
    g["outcome_sd"] = g["effect_per_decade"] / g["effect_sd_per_decade"]
    g["lo_sd"] = g["ci_low"] / g["outcome_sd"]
    g["hi_sd"] = g["ci_high"] / g["outcome_sd"]
    g = g.sort_values("effect_sd_per_decade")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.axvline(0, color="#777777", lw=1)
    ax.errorbar(
        g["effect_sd_per_decade"], y,
        xerr=[g["effect_sd_per_decade"] - g["lo_sd"], g["hi_sd"] - g["effect_sd_per_decade"]],
        fmt="o", color="#4C78A8", ecolor="#4C78A8", capsize=3,
    )
    ax.set_yticks(y, g["metric_label"])
    ax.set(title="F6. 보정 모형의 10년당 변화량", xlabel="문서 간 표준편차(SD) 단위")
    for yi, (_, row) in enumerate(g.iterrows()):
        ax.text(row["hi_sd"] + 0.01, yi, f"q={row['q_value']:.2g}", va="center", fontsize=9)
    save(fig, "F6_effect_forest")


def main() -> None:
    ensure_directories()
    configure_matplotlib()
    metrics = pd.read_parquet(DATA / "processed" / "document_metrics.parquet")
    raw = pd.read_csv(TABLES / "annual_raw_metrics.csv")
    std = pd.read_csv(TABLES / "annual_standardized_metrics.csv")
    trends = pd.read_csv(TABLES / "trend_models.csv")
    variants = pd.read_csv(TABLES / "variant_annual.csv")
    variant_summary = pd.read_csv(TABLES / "variant_summary.csv")
    figure_composition(metrics)
    figure_main_trends(raw, std)
    figure_small_multiples(std)
    figure_heatmap(std)
    figure_variants(variants, variant_summary)
    figure_forest(trends)
    print(f"Created 6 figures in {FIGURES}")


if __name__ == "__main__":
    main()
