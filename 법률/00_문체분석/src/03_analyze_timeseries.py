from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import DATA, METRIC_LABELS, TABLES, bh_adjust, ensure_directories, wilson_interval


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
STANDARD_CELL_COLS = ["rule_court_instance", "rule_crime_group"]
CONTROL_COLS = ["rule_court_instance", "rule_crime_group", "rule_document_focus"]


def raw_annual(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric in METRICS:
        out = (
            df.groupby("year", observed=True)[metric]
            .agg(n="count", mean="mean", median="median", sd="std", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
        )
        out["metric"] = metric
        out["se"] = out["sd"] / np.sqrt(out["n"])
        out["ci_low"] = out["mean"] - 1.96 * out["se"]
        out["ci_high"] = out["mean"] + 1.96 * out["se"]
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def standardized_annual(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in STANDARD_CELL_COLS:
        work[col] = work[col].fillna("unknown").astype(str)
    weights = work.groupby(STANDARD_CELL_COLS, observed=True).size().rename("weight") / len(work)
    weights = weights.reset_index()
    pieces = []
    for metric in METRICS:
        cells = (
            work.groupby(["year", *STANDARD_CELL_COLS], observed=True)[metric]
            .agg(cell_mean="mean", cell_var="var", cell_n="count")
            .reset_index()
            .merge(weights, on=STANDARD_CELL_COLS, how="left")
        )
        for year, group in cells.groupby("year", observed=True):
            group = group.loc[group["cell_n"] > 0].copy()
            coverage = float(group["weight"].sum())
            if coverage <= 0:
                continue
            group["w_norm"] = group["weight"] / coverage
            estimate = float((group["w_norm"] * group["cell_mean"]).sum())
            cell_se2 = group["cell_var"].fillna(0) / group["cell_n"]
            se = float(np.sqrt(((group["w_norm"] ** 2) * cell_se2).sum()))
            pieces.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "estimate": estimate,
                    "se": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "coverage": coverage,
                    "observed_cells": len(group),
                }
            )
    return pd.DataFrame(pieces)


def trend_models(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["year_c"] = work["year"].astype(float) - 2000
    work["log_tokens"] = np.log1p(work["mp_tokens"].astype(float))
    court_n = work["court_name"].value_counts()
    work["court_fe_group"] = work["court_name"].where(work["court_name"].map(court_n).ge(150), "other")
    for col in [*CONTROL_COLS, "court_fe_group", "court_name"]:
        work[col] = work[col].fillna("unknown").astype(str)

    rows = []
    controls = " + C(rule_court_instance) + C(rule_crime_group) + C(rule_document_focus) + C(court_fe_group)"
    for metric in METRICS:
        model_df = work.dropna(subset=[metric]).copy()
        lo, hi = model_df[metric].quantile([0.005, 0.995])
        model_df["outcome"] = model_df[metric].clip(lo, hi)
        outcome_sd = float(model_df["outcome"].std(ddof=1))
        for length_control in (False, True):
            formula = "outcome ~ year_c" + (" + log_tokens" if length_control else "") + controls
            try:
                model = smf.ols(formula, data=model_df).fit(
                    cov_type="cluster", cov_kwds={"groups": model_df["court_name"]}
                )
                coef = float(model.params["year_c"] * 10)
                se = float(model.bse["year_c"] * 10)
                pvalue = float(model.pvalues["year_c"])
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "length_control": length_control,
                        "n": int(model.nobs),
                        "effect_per_decade": coef,
                        "ci_low": coef - 1.96 * se,
                        "ci_high": coef + 1.96 * se,
                        "effect_sd_per_decade": coef / outcome_sd if outcome_sd else np.nan,
                        "p_value": pvalue,
                        "r_squared": float(model.rsquared),
                    }
                )
            except Exception as exc:
                warnings.warn(f"Trend model failed for {metric}: {exc}")
    result = pd.DataFrame(rows)
    result["q_value"] = np.nan
    for _, idx in result.groupby("length_control").groups.items():
        result.loc[idx, "q_value"] = bh_adjust(result.loc[idx, "p_value"])
    return result


def breakpoint_candidates(standardized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in standardized.groupby("metric", observed=True):
        group = group.loc[group["coverage"] >= 0.8].sort_values("year")
        x = group["year"].to_numpy(dtype=float)
        y = group["estimate"].to_numpy(dtype=float)
        if len(group) < 20:
            continue
        best = None
        for bp in range(int(x.min()) + 8, int(x.max()) - 7):
            design = np.column_stack([np.ones(len(x)), x - 2000, np.maximum(0, x - bp)])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design @ beta
            sse = float(resid @ resid)
            if best is None or sse < best[0]:
                best = (sse, bp, beta)
        if best:
            _, bp, beta = best
            rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "candidate_break_year": bp,
                    "slope_before_per_decade": beta[1] * 10,
                    "slope_after_per_decade": (beta[1] + beta[2]) * 10,
                    "note": "exploratory_candidate_not_causal",
                }
            )
    return pd.DataFrame(rows)


def variant_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(DATA.parent / "config" / "variant_pairs.csv", encoding="utf-8")
    annual_rows = []
    summary_rows = []
    for pair in pairs.to_dict("records"):
        old_col = f"variant_{pair['key']}_old"
        new_col = f"variant_{pair['key']}_new"
        annual = df.groupby("year", observed=True)[[old_col, new_col]].sum().reset_index()
        annual["total"] = annual[old_col] + annual[new_col]
        annual["old_share"] = annual[old_col] / annual["total"].replace(0, np.nan)
        intervals = [wilson_interval(a, b) for a, b in zip(annual[old_col], annual["total"])]
        annual["ci_low"] = [x[0] for x in intervals]
        annual["ci_high"] = [x[1] for x in intervals]
        annual["key"] = pair["key"]
        annual["label_old"] = pair["label_old"]
        annual["label_new"] = pair["label_new"]
        annual_rows.append(annual.rename(columns={old_col: "old_n", new_col: "new_n"}))

        fit = annual.loc[annual["total"] >= 10].dropna(subset=["old_share"])
        t50 = slope = pvalue = np.nan
        if len(fit) >= 10 and fit["old_share"].nunique() > 1:
            exog = sm.add_constant(fit["year"].astype(float))
            model = sm.GLM(
                fit["old_share"], exog, family=sm.families.Binomial(), freq_weights=fit["total"]
            ).fit()
            intercept, slope = map(float, model.params)
            pvalue = float(model.pvalues.iloc[1])
            if slope != 0:
                candidate = -intercept / slope
                if 1900 <= candidate <= 2100:
                    t50 = candidate
        summary_rows.append(
            {
                "key": pair["key"],
                "label_old": pair["label_old"],
                "label_new": pair["label_new"],
                "total_old": int(annual[old_col].sum()),
                "total_new": int(annual[new_col].sum()),
                "logit_year_slope": slope,
                "p_value": pvalue,
                "t50": t50,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["q_value"] = bh_adjust(summary["p_value"])
    return pd.concat(annual_rows, ignore_index=True), summary


def main() -> None:
    ensure_directories()
    metrics = pd.read_parquet(INPUT)
    main_df = metrics.loc[metrics["main_analysis"]].copy()

    raw = raw_annual(main_df)
    standardized = standardized_annual(main_df)
    trends = trend_models(main_df)
    breakpoints = breakpoint_candidates(standardized)
    variants, variant_summary = variant_trends(main_df)

    raw.to_csv(TABLES / "annual_raw_metrics.csv", index=False, encoding="utf-8-sig")
    standardized.to_csv(TABLES / "annual_standardized_metrics.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(TABLES / "trend_models.csv", index=False, encoding="utf-8-sig")
    breakpoints.to_csv(TABLES / "breakpoint_candidates.csv", index=False, encoding="utf-8-sig")
    variants.to_csv(TABLES / "variant_annual.csv", index=False, encoding="utf-8-sig")
    variant_summary.to_csv(TABLES / "variant_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Main sample: {len(main_df):,}")
    print(trends.loc[trends["length_control"], ["metric_label", "effect_per_decade", "ci_low", "ci_high", "p_value", "q_value"]].to_string(index=False))
    print("\nStandardization coverage:")
    print(standardized.groupby("metric")["coverage"].agg(["min", "median", "max"]).to_string())


if __name__ == "__main__":
    main()
