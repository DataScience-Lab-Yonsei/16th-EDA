from __future__ import annotations

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".mplconfig"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, METRIC_LABELS, TABLES, configure_matplotlib, ensure_directories


KEY_METRICS = ["noun_share", "verb_share", "particle_share", "sent_eojeol_mean"]
GROUP_ORDER = [
    ("심급", "third"),
    ("심급", "second"),
    ("심급", "first"),
    ("범죄군", "재산범죄"),
    ("범죄군", "강력/폭력범죄"),
    ("범죄군", "성범죄"),
    ("범죄군", "교통범죄"),
    ("범죄군", "마약범죄"),
    ("범죄군", "기타"),
]
GROUP_LABELS = {
    ("심급", "third"): "3심",
    ("심급", "second"): "2심",
    ("심급", "first"): "1심",
    ("범죄군", "재산범죄"): "재산범죄",
    ("범죄군", "강력/폭력범죄"): "강력·폭력",
    ("범죄군", "성범죄"): "성범죄",
    ("범죄군", "교통범죄"): "교통범죄",
    ("범죄군", "마약범죄"): "마약범죄",
    ("범죄군", "기타"): "기타",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def subgroup_forest(subgroup: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, KEY_METRICS):
        g = subgroup.loc[
            subgroup["metric"].eq(metric) & subgroup["group_type"].ne("전체")
        ].copy()
        g["key"] = list(zip(g["group_type"], g["group_name"]))
        g = g.set_index("key").reindex(GROUP_ORDER).dropna(subset=["effect_per_decade"])
        scale = 100 if metric.endswith("share") else 1
        y = np.arange(len(g))
        colors = ["#F58518" if key[0] == "심급" else "#4C78A8" for key in g.index]
        ax.axvline(0, color="#777777", lw=1)
        for yi, (key, row), color in zip(y, g.iterrows(), colors):
            ax.errorbar(
                row["effect_per_decade"] * scale,
                yi,
                xerr=np.array(
                    [
                        [
                            (row["effect_per_decade"] - row["ci_low"]) * scale,
                            (row["ci_high"] - row["effect_per_decade"]) * scale,
                        ]
                    ]
                ).T,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
            )
        ax.set_yticks(y, [GROUP_LABELS[key] for key in g.index])
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("%p/10년" if scale == 100 else "어절/10년")
        ax.grid(axis="y", visible=False)
    fig.suptitle("F7. 범죄군·심급별 보정 추세", fontsize=16)
    save(fig, "F7_subgroup_robustness")


def adjusted_method_comparison(
    direct: pd.DataFrame, year_fe: pd.DataFrame, balanced: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharex=True)
    for ax, metric in zip(axes.flat, KEY_METRICS):
        scale = 100 if metric.endswith("share") else 1
        d = direct.loc[
            direct["metric"].eq(metric) & direct["coverage"].ge(0.8)
        ].sort_values("year")
        y = year_fe.loc[year_fe["metric"].eq(metric)].sort_values("year")
        b = balanced.loc[balanced["metric"].eq(metric)].sort_values("year")
        ax.plot(
            d["year"],
            d["estimate"] * scale,
            color="#4C78A8",
            lw=2.3,
            label="직접표준화",
        )
        ax.plot(
            y["year"],
            y["estimate"] * scale,
            color="#F58518",
            lw=1.8,
            ls="--",
            label="연도 더미 회귀",
        )
        ax.plot(
            b["year"],
            b["estimate"] * scale,
            color="#54A24B",
            lw=1.8,
            alpha=0.9,
            label="균형 재표집",
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("%" if scale == 100 else "어절")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("F8. 보정 방법을 바꿔도 유지되는 방향", fontsize=16)
    save(fig, "F8_adjustment_robustness")


def breakpoint_intervals(breakpoints: pd.DataFrame) -> None:
    g = breakpoints.copy().sort_values("candidate_break_year")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.errorbar(
        g["candidate_break_year"],
        y,
        xerr=[
            g["candidate_break_year"] - g["ci_low"],
            g["ci_high"] - g["candidate_break_year"],
        ],
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=3,
    )
    ax.set_yticks(y, g["metric_label"])
    ax.set_xlim(1987, 2020)
    ax.set_xlabel("탐색적 변화점 후보 연도와 블록 부트스트랩 95% 구간")
    ax.set_title("F9. 변화점은 점이 아니라 구간으로 해석해야 한다")
    for year in [1990, 2000, 2010]:
        ax.axvline(year, color="#BBBBBB", lw=0.8, ls=":")
    save(fig, "F9_breakpoint_uncertainty")


def main() -> None:
    ensure_directories()
    configure_matplotlib()
    subgroup = pd.read_csv(TABLES / "robustness_subgroups.csv")
    direct = pd.read_csv(TABLES / "annual_standardized_metrics.csv")
    year_fe = pd.read_csv(TABLES / "year_fe_adjusted_metrics.csv")
    balanced = pd.read_csv(TABLES / "balanced_annual_metrics.csv")
    breakpoints = pd.read_csv(TABLES / "breakpoint_bootstrap.csv")
    subgroup_forest(subgroup)
    adjusted_method_comparison(direct, year_fe, balanced)
    breakpoint_intervals(breakpoints)
    print(f"Created F7-F9 in {FIGURES}")


if __name__ == "__main__":
    main()
