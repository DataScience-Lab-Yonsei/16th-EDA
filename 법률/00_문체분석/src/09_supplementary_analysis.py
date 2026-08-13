from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from common import (
    DATA,
    FIGURES,
    METRIC_LABELS,
    TABLES,
    bh_adjust,
    configure_matplotlib,
    ensure_directories,
)


MAIN_METRICS = [
    "noun_share",
    "verb_share",
    "particle_share",
    "sent_eojeol_mean",
]
ALL_METRICS = list(METRIC_LABELS)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def scale_for(metric: str) -> float:
    return 100.0 if metric.endswith("share") else 1.0


def unit_for(metric: str) -> str:
    return "%p" if metric.endswith("share") else "어절"


def hac_regressions(standardized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in ALL_METRICS:
        group = standardized.loc[
            standardized["metric"].eq(metric) & standardized["coverage"].ge(0.8)
        ].sort_values("year")
        year_c = group["year"].astype(float) - 2000.0
        design = sm.add_constant(year_c.to_numpy())
        fit = sm.OLS(group["estimate"].to_numpy(), design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3, "use_correction": True},
        )
        effect = float(fit.params[1] * 10)
        ci = fit.conf_int()[1] * 10
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n_years": int(len(group)),
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "effect_per_decade": effect,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "p_value": float(fit.pvalues[1]),
                "hac_lag": 3,
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_adjust(result["p_value"])
    result.to_csv(TABLES / "hac_annual_regression.csv", index=False, encoding="utf-8-sig")
    return result


def regression_line_figure(
    standardized: pd.DataFrame, hac: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.2), constrained_layout=True, sharex=True
    )
    for ax, metric in zip(axes.flat, MAIN_METRICS):
        group = standardized.loc[
            standardized["metric"].eq(metric) & standardized["coverage"].ge(0.8)
        ].sort_values("year")
        scale = scale_for(metric)
        x = group["year"].astype(float).to_numpy()
        year_c = x - 2000.0
        design = sm.add_constant(year_c)
        fit = sm.OLS(group["estimate"].to_numpy(), design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3, "use_correction": True},
        )
        grid = np.linspace(x.min(), x.max(), 240)
        grid_design = np.column_stack([np.ones_like(grid), grid - 2000.0])
        fitted = grid_design @ fit.params
        covariance = np.asarray(fit.cov_params())
        pred_se = np.sqrt(
            np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design)
        )
        ax.scatter(
            x,
            group["estimate"].to_numpy() * scale,
            s=26,
            color="#4C78A8",
            alpha=0.72,
            label="구성 보정 연도값",
            zorder=3,
        )
        ax.plot(
            grid,
            fitted * scale,
            color="#E45756",
            lw=2.5,
            label="선형 회귀선",
            zorder=4,
        )
        ax.fill_between(
            grid,
            (fitted - 1.96 * pred_se) * scale,
            (fitted + 1.96 * pred_se) * scale,
            color="#E45756",
            alpha=0.14,
            label="95% 신뢰구간",
        )
        info = hac.loc[hac["metric"].eq(metric)].iloc[0]
        effect = info["effect_per_decade"] * scale
        low = info["ci_low"] * scale
        high = info["ci_high"] * scale
        ax.text(
            0.02,
            0.04,
            f"10년당 {effect:+.3f} {unit_for(metric)}\n"
            f"95% CI {low:+.3f} ~ {high:+.3f}, q={info['q_value']:.2g}",
            transform=ax.transAxes,
            fontsize=9.2,
            color="#333333",
            bbox={
                "facecolor": "white",
                "edgecolor": "#D9E2F3",
                "alpha": 0.92,
                "boxstyle": "round,pad=0.35",
            },
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("%" if scale == 100 else "어절")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8.8)
    fig.suptitle(
        "F10. 구성 보정 연도값과 선형 회귀선 (Newey-West HAC, lag 3)",
        fontsize=16,
    )
    save_figure(fig, "F10_adjusted_regression_lines")


def raw_median_iqr_figure(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.2), constrained_layout=True, sharex=True
    )
    for ax, metric in zip(axes.flat, MAIN_METRICS):
        group = raw.loc[raw["metric"].eq(metric)].sort_values("year")
        scale = scale_for(metric)
        ax.fill_between(
            group["year"].to_numpy(),
            group["q25"].to_numpy() * scale,
            group["q75"].to_numpy() * scale,
            color="#4C78A8",
            alpha=0.16,
            label="IQR",
        )
        ax.plot(
            group["year"],
            group["median"] * scale,
            color="#4C78A8",
            lw=2.3,
            label="중앙값",
        )
        ax.plot(
            group["year"],
            group["mean"] * scale,
            color="#888888",
            lw=1.4,
            ls="--",
            label="평균",
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("%" if scale == 100 else "어절")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("F11. 원자료 중앙값, IQR과 문서평균", fontsize=16)
    save_figure(fig, "F11_raw_median_iqr")


def annual_aggregation(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.loc[
        metrics["main_analysis"]
        & metrics["year"].between(1980, 2025)
    ].copy()
    rows: list[dict] = []
    for year, group in work.groupby("year", observed=True):
        denominators = {
            "noun_share": group["mp_tokens"].sum(),
            "verb_share": group["mp_tokens"].sum(),
            "particle_share": group["mp_tokens"].sum(),
            "sent_eojeol_mean": group["sentence_count"].sum(),
        }
        pooled = {
            "noun_share": group["noun_count"].sum() / denominators["noun_share"],
            "verb_share": group["verb_count"].sum() / denominators["verb_share"],
            "particle_share": group["particle_count"].sum()
            / denominators["particle_share"],
            "sent_eojeol_mean": group["eojeol_count"].sum()
            / denominators["sent_eojeol_mean"],
        }
        for metric in MAIN_METRICS:
            rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "document_mean": float(group[metric].mean()),
                    "pooled": float(pooled[metric]),
                    "n_documents": int(len(group)),
                }
            )
    annual = pd.DataFrame(rows)
    annual.to_csv(
        TABLES / "aggregation_annual.csv", index=False, encoding="utf-8-sig"
    )
    return annual


def aggregation_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in MAIN_METRICS:
        group = annual.loc[
            annual["metric"].eq(metric) & annual["year"].between(1985, 2025)
        ].sort_values("year")
        row: dict[str, float | str] = {
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
        }
        for method in ["document_mean", "pooled"]:
            x = group["year"].astype(float).to_numpy() - 2000.0
            fit = sm.OLS(group[method].to_numpy(), sm.add_constant(x)).fit()
            row[f"{method}_effect_per_decade"] = float(fit.params[1] * 10)
            for year in [1985, 2025]:
                values = group.loc[group["year"].eq(year), method]
                row[f"{method}_{year}"] = (
                    float(values.iloc[0]) if not values.empty else np.nan
                )
        row["same_slope_direction"] = bool(
            np.sign(row["document_mean_effect_per_decade"])
            == np.sign(row["pooled_effect_per_decade"])
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(
        TABLES / "aggregation_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def aggregation_figure(annual: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.2), constrained_layout=True, sharex=True
    )
    for ax, metric in zip(axes.flat, MAIN_METRICS):
        group = annual.loc[annual["metric"].eq(metric)].sort_values("year")
        scale = scale_for(metric)
        ax.plot(
            group["year"],
            group["document_mean"] * scale,
            color="#4C78A8",
            lw=2.2,
            label="문서평균",
        )
        ax.plot(
            group["year"],
            group["pooled"] * scale,
            color="#F28E2B",
            lw=1.9,
            ls="--",
            label="토큰·문장 풀링",
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("%" if scale == 100 else "어절")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("F12. 문서평균과 풀링 집계의 원자료 추세 비교", fontsize=16)
    save_figure(fig, "F12_aggregation_sensitivity")


def length_control_comparison() -> pd.DataFrame:
    trends = pd.read_csv(TABLES / "trend_models.csv")
    trends["length_control"] = (
        trends["length_control"].astype(str).str.lower().eq("true")
    )
    records: list[dict] = []
    for metric in ALL_METRICS:
        group = trends.loc[trends["metric"].eq(metric)].copy()
        record: dict[str, float | str | bool] = {
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
        }
        for value, suffix in [(False, "without_length"), (True, "with_length")]:
            row = group.loc[group["length_control"].eq(value)].iloc[0]
            record[f"{suffix}_effect_sd_per_decade"] = float(
                row["effect_sd_per_decade"]
            )
            record[f"{suffix}_q_value"] = float(row["q_value"])
        record["same_direction"] = bool(
            np.sign(record["without_length_effect_sd_per_decade"])
            == np.sign(record["with_length_effect_sd_per_decade"])
        )
        records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(
        TABLES / "length_control_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main() -> None:
    ensure_directories()
    configure_matplotlib()
    standardized = pd.read_csv(TABLES / "annual_standardized_metrics.csv")
    raw = pd.read_csv(TABLES / "annual_raw_metrics.csv")
    metrics = pd.read_parquet(DATA / "processed" / "document_metrics.parquet")

    hac = hac_regressions(standardized)
    regression_line_figure(standardized, hac)
    raw_median_iqr_figure(raw)

    annual = annual_aggregation(metrics)
    aggregation_summary(annual)
    aggregation_figure(annual)
    length_control_comparison()

    print(
        hac.loc[
            hac["metric"].isin(MAIN_METRICS),
            ["metric_label", "effect_per_decade", "ci_low", "ci_high", "q_value"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
