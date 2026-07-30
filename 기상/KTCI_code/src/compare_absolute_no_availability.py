"""Spearman 절댓값을 유지하고 관측률만 제외한 KTCI 민감도 분석.

증거점수:
    E = sqrt(abs(Spearman) * decile_amplitude)

개별 날짜의 결측값은 0으로 대체하지 않으며, 사용 가능한 구성요소의
가중치 합으로 다시 나누는 방식은 기본 파이프라인과 동일하게 유지한다.
"""
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def weighted_available(scores, weights, min_components=2):
    positive = {k: v for k, v in weights.items() if v > 0 and k in scores.columns}
    cols = list(positive)
    w = pd.Series(positive)
    valid = scores[cols].notna()
    numerator = scores[cols].mul(w, axis=1).sum(axis=1, min_count=1)
    denominator = valid.mul(w, axis=1).sum(axis=1)
    return numerator.div(denominator).where(valid.sum(axis=1) >= min_components)


def top_bottom(y, index, q=.20):
    d = pd.DataFrame({"y": y, "index": index}).dropna()
    lo, hi = d["index"].quantile([q, 1-q])
    bottom = d.loc[d["index"] <= lo, "y"].mean()
    top = d.loc[d["index"] >= hi, "y"].mean()
    return top, bottom, top-bottom


def performance(d, col):
    x = d[["year", "tourism_residual_log", "tourism_change_pct", col]].dropna()
    rho = x[[col, "tourism_residual_log"]].corr(method="spearman").iloc[0, 1]
    top, bottom, diff = top_bottom(x["tourism_change_pct"], x[col])
    train, test = x[x.year.isin([2023, 2024])], x[x.year.eq(2025)]
    model = LinearRegression().fit(train[[col]], train["tourism_residual_log"])
    pred = model.predict(test[[col]])
    return {
        "index": col, "n": len(x), "spearman": rho,
        "top20_mean_pct": top, "bottom20_mean_pct": bottom,
        "top_bottom_diff_pp": diff,
        "test_r2": r2_score(test["tourism_residual_log"], pred),
        "test_mae_log": mean_absolute_error(test["tourism_residual_log"], pred),
    }


cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
df = pd.read_csv(ROOT / cfg["project"]["input_file"], encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["weekday"] = df.date.dt.weekday
month_to_season = {m: s for s, months in cfg["analysis"]["seasons"].items() for m in months}
df["season"] = df.month.map(month_to_season)

df["log_visitors"] = np.log1p(df["visitor_total"].clip(lower=0))
group = df.groupby(["signguCode", "year", "month", "weekday"])["log_visitors"]
gsum, gcount = group.transform("sum"), group.transform("count")
df["baseline_log_loo"] = (gsum - df.log_visitors) / (gcount - 1)
df.loc[gcount < 2, "baseline_log_loo"] = np.nan
df["tourism_residual_log"] = df.log_visitors - df.baseline_log_loo
df["tourism_change_pct"] = np.expm1(df.tourism_residual_log) * 100

stress_map = cfg["analysis"]["stress_columns"]
scores = pd.DataFrame(index=df.index)
for component, col in stress_map.items():
    scores[component] = (1 - pd.to_numeric(df[col], errors="coerce")).clip(0, 1) * 100

old = pd.read_csv(TABLES / "step4_seasonal_weight_decomposition.csv")
rows = []
for _, r in old.iterrows():
    directional = abs(float(r.spearman))
    evidence = math.sqrt(directional * max(float(r.decile_amplitude_log), 0.0))
    rows.append({
        "season": r.season,
        "component": r.component,
        "spearman": r.spearman,
        "absolute_spearman": directional,
        "decile_amplitude_log": r.decile_amplitude_log,
        "availability_reference_only": r.availability,
        "current_weight": r.data_weight,
        "new_evidence_no_availability": evidence,
    })
new_weights = pd.DataFrame(rows)
new_weights["new_weight_absolute_no_availability"] = new_weights.groupby("season")[
    "new_evidence_no_availability"
].transform(lambda x: x / x.sum())
new_weights["weight_change_pp"] = (
    new_weights.new_weight_absolute_no_availability - new_weights.current_weight
) * 100

for season in cfg["analysis"]["seasons"]:
    idx = df.season.eq(season)
    current_w = old[old.season.eq(season)].set_index("component").data_weight.to_dict()
    new_w = new_weights[new_weights.season.eq(season)].set_index("component")[
        "new_weight_absolute_no_availability"
    ].to_dict()
    df.loc[idx, "KTCI_current"] = weighted_available(scores.loc[idx], current_w, 2)
    df.loc[idx, "KTCI_new_absolute"] = weighted_available(scores.loc[idx], new_w, 2)

    legacy = cfg["benchmark_2014"][season].copy()
    legacy.pop("cloud_unavailable", None)
    legacy = {
        "thermal": legacy["thermal"],
        "precipitation": legacy["precipitation"],
        "wind": legacy["wind"],
    }
    total = sum(legacy.values())
    legacy = {k: v/total for k, v in legacy.items()}
    df.loc[idx, "KTCI_2014_adapted"] = weighted_available(scores.loc[idx], legacy, 2)

common = df.dropna(subset=["KTCI_current", "KTCI_new_absolute", "KTCI_2014_adapted"]).copy()
performance_rows = []
for season in ["all", "spring", "summer", "autumn", "winter"]:
    d = common if season == "all" else common[common.season.eq(season)]
    for col in ["KTCI_current", "KTCI_new_absolute", "KTCI_2014_adapted"]:
        result = performance(d, col)
        result["season"] = season
        performance_rows.append(result)
performance_df = pd.DataFrame(performance_rows)

TABLES.mkdir(parents=True, exist_ok=True)
new_weights.to_csv(TABLES / "alternative_absolute_no_availability_weights.csv",
                   index=False, encoding="utf-8-sig")
performance_df.to_csv(TABLES / "alternative_absolute_no_availability_performance.csv",
                      index=False, encoding="utf-8-sig")

summary = {
    "formula": "sqrt(abs(spearman) * decile_amplitude); availability excluded",
    "common_rows": len(common),
    "outputs": [
        "alternative_absolute_no_availability_weights.csv",
        "alternative_absolute_no_availability_performance.csv",
    ],
}
(OUT / "logs" / "alternative_absolute_comparison_metadata.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))

