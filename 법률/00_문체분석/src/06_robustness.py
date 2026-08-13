from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from common import DATA, METRIC_LABELS, TABLES, bh_adjust, ensure_directories


INPUT = DATA / "processed" / "document_metrics.parquet"
METRICS = [
    "noun_share",
    "verb_share",
    "predicate_share",
    "particle_share",
    "nominality",
    "sent_eojeol_mean",
    "sent_char_mean",
    "hanja_share",
]
KEY_METRICS = ["noun_share", "verb_share", "particle_share", "sent_eojeol_mean"]
CELL_COLS = ["rule_court_instance", "rule_crime_group"]
RNG_SEED = 20260723


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    work = df.loc[df["main_analysis"]].copy()
    work["year"] = work["year"].astype(int)
    work["year_c"] = work["year"].astype(float) - 2000
    work["log_tokens"] = np.log1p(work["mp_tokens"].astype(float))
    court_n = work["court_name"].value_counts()
    work["court_fe_group"] = work["court_name"].where(
        work["court_name"].map(court_n).ge(150), "other"
    )
    for col in [
        "rule_court_instance",
        "rule_crime_group",
        "rule_document_focus",
        "court_fe_group",
        "court_name",
    ]:
        work[col] = work[col].fillna("unknown").astype(str)
    return work


def available_controls(df: pd.DataFrame) -> list[str]:
    candidates = [
        "rule_court_instance",
        "rule_crime_group",
        "rule_document_focus",
        "court_fe_group",
    ]
    return [col for col in candidates if df[col].nunique(dropna=False) > 1]


def fit_one_trend(df: pd.DataFrame, metric: str) -> dict:
    model_df = df.dropna(subset=[metric, "court_name"]).copy()
    lo, hi = model_df[metric].quantile([0.005, 0.995])
    model_df["outcome"] = model_df[metric].clip(lo, hi)
    controls = "".join(f" + C({col})" for col in available_controls(model_df))
    formula = "outcome ~ year_c + log_tokens" + controls
    court_count = int(model_df["court_name"].nunique())
    if court_count >= 2:
        model = smf.ols(formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["court_name"]}
        )
        inference = "court_cluster"
    else:
        model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
        inference = "HC3_single_court"
    coef = float(model.params["year_c"] * 10)
    se = float(model.bse["year_c"] * 10)
    return {
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "n": int(model.nobs),
        "court_count": court_count,
        "inference": inference,
        "effect_per_decade": coef,
        "ci_low": coef - 1.96 * se,
        "ci_high": coef + 1.96 * se,
        "p_value": float(model.pvalues["year_c"]),
    }


def subgroup_trends(work: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, str, pd.DataFrame]] = [("전체", "전체", work)]
    for value, group in work.groupby("rule_crime_group", observed=True):
        if len(group) >= 300:
            groups.append(("범죄군", str(value), group))
    for value, group in work.groupby("rule_court_instance", observed=True):
        if value != "unclear" and len(group) >= 300:
            groups.append(("심급", str(value), group))

    rows = []
    for group_type, group_name, group in groups:
        for metric in METRICS:
            try:
                row = fit_one_trend(group, metric)
                row.update({"group_type": group_type, "group_name": group_name})
                rows.append(row)
            except Exception as exc:
                warnings.warn(f"Subgroup model failed: {group_type}/{group_name}/{metric}: {exc}")

    result = pd.DataFrame(rows)
    result["q_value"] = np.nan
    for _, idx in result.groupby(["group_type", "group_name"]).groups.items():
        result.loc[idx, "q_value"] = bh_adjust(result.loc[idx, "p_value"])

    overall_sign = (
        result.loc[result["group_type"].eq("전체")]
        .set_index("metric")["effect_per_decade"]
        .map(np.sign)
    )
    result["same_direction_as_overall"] = [
        bool(np.sign(effect) == overall_sign.get(metric, np.nan))
        for metric, effect in zip(result["metric"], result["effect_per_decade"])
    ]
    return result


def year_fe_adjusted(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    formula_controls = (
        " + log_tokens + C(rule_court_instance) + C(rule_crime_group)"
        " + C(rule_document_focus) + C(court_fe_group)"
    )
    for metric in KEY_METRICS:
        model_df = work.dropna(subset=[metric]).copy()
        lo, hi = model_df[metric].quantile([0.005, 0.995])
        model_df["outcome"] = model_df[metric].clip(lo, hi)
        model = smf.ols("outcome ~ C(year)" + formula_controls, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["court_name"]}
        )
        design_info = model.model.data.design_info
        original_x = np.asarray(model.model.exog)
        mean_design = original_x.mean(axis=0)
        columns = list(design_info.column_names)
        year_cols = [i for i, name in enumerate(columns) if name.startswith("C(year)[T.")]
        cov = np.asarray(model.cov_params())
        params = np.asarray(model.params)

        metric_rows = []
        for year in sorted(model_df["year"].unique()):
            g = mean_design.copy()
            g[year_cols] = 0.0
            year_col = f"C(year)[T.{int(year)}]"
            if year_col in columns:
                g[columns.index(year_col)] = 1.0
            estimate = float(g @ params)
            se = float(np.sqrt(max(g @ cov @ g, 0.0)))
            metric_rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": estimate,
                    "se": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                }
            )
        rows.extend(metric_rows)

        metric_df = pd.DataFrame(metric_rows)
        direct = pd.read_csv(TABLES / "annual_standardized_metrics.csv")
        direct = direct.loc[
            direct["metric"].eq(metric) & direct["coverage"].ge(0.8),
            ["year", "estimate"],
        ].rename(columns={"estimate": "direct_estimate"})
        compare = metric_df.merge(direct, on="year", how="inner")
        summaries.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "correlation_with_direct_standardization": compare[
                    ["estimate", "direct_estimate"]
                ].corr().iloc[0, 1],
                "mean_absolute_difference": (
                    compare["estimate"] - compare["direct_estimate"]
                ).abs().mean(),
                "n_years": len(compare),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def balanced_resampling(
    work: pd.DataFrame, n_boot: int = 500, per_cell: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = work.loc[
        work["year"].between(1985, 2025)
        & work["rule_court_instance"].ne("unclear")
    ].copy()
    years = np.array(sorted(frame["year"].unique()), dtype=int)
    cell_coverage = (
        frame.groupby(CELL_COLS, observed=True)["year"].nunique().rename("n_years")
    )
    common_cells = [tuple(x) for x in cell_coverage.loc[cell_coverage.eq(len(years))].index]
    if not common_cells:
        raise RuntimeError("No composition cells are observed in every year.")

    grouped: dict[tuple[int, str, str], np.ndarray] = {}
    for (year, instance, crime), group in frame.groupby(
        ["year", *CELL_COLS], observed=True
    ):
        key = (int(year), str(instance), str(crime))
        if (str(instance), str(crime)) in common_cells:
            grouped[key] = group[METRICS].to_numpy(dtype=float)

    point = np.full((len(years), len(METRICS)), np.nan)
    for yi, year in enumerate(years):
        cell_means = [
            np.nanmean(grouped[(int(year), instance, crime)], axis=0)
            for instance, crime in common_cells
        ]
        point[yi] = np.nanmean(np.vstack(cell_means), axis=0)

    rng = np.random.default_rng(RNG_SEED)
    boot_annual = np.full((n_boot, len(years), len(METRICS)), np.nan)
    for b in range(n_boot):
        for yi, year in enumerate(years):
            sampled_cell_means = []
            for instance, crime in common_cells:
                values = grouped[(int(year), instance, crime)]
                indices = rng.integers(0, len(values), size=per_cell)
                sampled_cell_means.append(np.nanmean(values[indices], axis=0))
            boot_annual[b, yi] = np.nanmean(np.vstack(sampled_cell_means), axis=0)

    x = years.astype(float)
    x_centered = x - x.mean()
    denom = float(x_centered @ x_centered)
    point_slopes = (x_centered[:, None] * point).sum(axis=0) / denom * 10
    boot_slopes = (
        (x_centered[None, :, None] * boot_annual).sum(axis=1) / denom * 10
    )

    summary_rows = []
    for mi, metric in enumerate(METRICS):
        slopes = boot_slopes[:, mi]
        summary_rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n_years": len(years),
                "common_cells": len(common_cells),
                "sample_per_cell_year": per_cell,
                "bootstrap_repetitions": n_boot,
                "effect_per_decade": point_slopes[mi],
                "bootstrap_median": float(np.nanmedian(slopes)),
                "ci_low": float(np.nanquantile(slopes, 0.025)),
                "ci_high": float(np.nanquantile(slopes, 0.975)),
                "positive_probability": float(np.nanmean(slopes > 0)),
            }
        )

    annual_rows = []
    for yi, year in enumerate(years):
        for mi, metric in enumerate(METRICS):
            values = boot_annual[:, yi, mi]
            annual_rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": point[yi, mi],
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                }
            )

    cells_df = pd.DataFrame(common_cells, columns=CELL_COLS)
    return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows), cells_df


def hash_value(value: str, salt: int) -> float:
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def placebo_tests(work: pd.DataFrame, n_placebos: int = 1) -> pd.DataFrame:
    rows = []
    formula = (
        "placebo ~ year_c + log_tokens + C(rule_court_instance)"
        " + C(rule_crime_group) + C(rule_document_focus) + C(court_fe_group)"
    )
    for salt in range(n_placebos):
        model_df = work.copy()
        model_df["placebo"] = [
            hash_value(str(value), salt) for value in model_df["precedent_id"]
        ]
        model = smf.ols(formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["court_name"]}
        )
        coef = float(model.params["year_c"] * 10)
        se = float(model.bse["year_c"] * 10)
        rows.append(
            {
                "placebo_id": f"sha256_{salt:02d}",
                "effect_per_decade": coef,
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "p_value": float(model.pvalues["year_c"]),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_adjust(result["p_value"])
    result["significant_after_bh"] = result["q_value"] < 0.05
    return result


def best_segmented_fit(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for bp in range(int(x.min()) + 8, int(x.max()) - 7):
        design = np.column_stack([np.ones(len(x)), x - 2000, np.maximum(0, x - bp)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        resid = y - fitted
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, bp, beta, fitted)
    if best is None:
        raise RuntimeError("No valid breakpoint candidate.")
    return best[1], best[2], best[3]


def moving_block_sample(resid: np.ndarray, rng: np.random.Generator, block: int = 3) -> np.ndarray:
    n = len(resid)
    starts = rng.integers(0, n, size=math.ceil(n / block))
    sampled = []
    for start in starts:
        sampled.extend(resid[(start + offset) % n] for offset in range(block))
    return np.asarray(sampled[:n], dtype=float)


def breakpoint_bootstrap(
    standardized: pd.DataFrame, n_boot: int = 1000
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for metric, group in standardized.groupby("metric", observed=True):
        group = group.loc[group["coverage"].ge(0.8)].sort_values("year")
        x = group["year"].to_numpy(dtype=float)
        y = group["estimate"].to_numpy(dtype=float)
        if len(group) < 20:
            continue
        bp, beta, fitted = best_segmented_fit(x, y)
        resid = y - fitted
        boot_bp = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            y_star = fitted + moving_block_sample(resid, rng, block=3)
            boot_bp[i] = best_segmented_fit(x, y_star)[0]
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "candidate_break_year": bp,
                "bootstrap_median": float(np.median(boot_bp)),
                "ci_low": float(np.quantile(boot_bp, 0.025)),
                "ci_high": float(np.quantile(boot_bp, 0.975)),
                "share_within_2_years": float(np.mean(np.abs(boot_bp - bp) <= 2)),
                "slope_before_per_decade": float(beta[1] * 10),
                "slope_after_per_decade": float((beta[1] + beta[2]) * 10),
                "bootstrap_repetitions": n_boot,
                "note": "moving_block_residual_bootstrap_exploratory_not_causal",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    metrics = pd.read_parquet(INPUT)
    work = prepare(metrics)
    standardized = pd.read_csv(TABLES / "annual_standardized_metrics.csv")

    subgroup = subgroup_trends(work)
    year_fe, year_fe_summary = year_fe_adjusted(work)
    balanced, balanced_annual, common_cells = balanced_resampling(work)
    placebo = placebo_tests(work)
    breakpoints = breakpoint_bootstrap(standardized)

    subgroup.to_csv(TABLES / "robustness_subgroups.csv", index=False, encoding="utf-8-sig")
    year_fe.to_csv(TABLES / "year_fe_adjusted_metrics.csv", index=False, encoding="utf-8-sig")
    year_fe_summary.to_csv(
        TABLES / "year_fe_comparison.csv", index=False, encoding="utf-8-sig"
    )
    balanced.to_csv(
        TABLES / "robustness_balanced_resampling.csv", index=False, encoding="utf-8-sig"
    )
    balanced_annual.to_csv(
        TABLES / "balanced_annual_metrics.csv", index=False, encoding="utf-8-sig"
    )
    common_cells.to_csv(
        TABLES / "balanced_common_cells.csv", index=False, encoding="utf-8-sig"
    )
    placebo.to_csv(TABLES / "placebo_tests.csv", index=False, encoding="utf-8-sig")
    breakpoints.to_csv(
        TABLES / "breakpoint_bootstrap.csv", index=False, encoding="utf-8-sig"
    )

    print("Subgroup direction agreement:")
    print(
        subgroup.loc[subgroup["group_type"].ne("전체")]
        .groupby(["group_type", "metric_label"])["same_direction_as_overall"]
        .agg(["sum", "count"])
        .to_string()
    )
    print("\nBalanced resampling:")
    print(
        balanced[
            ["metric_label", "effect_per_decade", "ci_low", "ci_high"]
        ].to_string(index=False)
    )
    print("\nPlacebo significant after BH:", int(placebo["significant_after_bh"].sum()))
    print("\nBreakpoint bootstrap:")
    print(
        breakpoints[
            ["metric_label", "candidate_break_year", "ci_low", "ci_high"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
