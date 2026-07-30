"""현재 KTCI와 두 민감도 대안의 동일 표본 성능 비교."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def weighted_available(scores, weights, min_components=2):
    positive = {k: v for k, v in weights.items() if v > 0 and k in scores.columns}
    cols = list(positive)
    w = pd.Series(positive)
    valid = scores[cols].notna()
    numerator = scores[cols].mul(w, axis=1).sum(axis=1, min_count=1)
    denominator = valid.mul(w, axis=1).sum(axis=1)
    return numerator.div(denominator).where(valid.sum(axis=1) >= min_components)


def performance(data, column):
    x = data[["year", "tourism_residual_log", "tourism_change_pct", column]].dropna()
    rho = x[[column, "tourism_residual_log"]].corr(method="spearman").iloc[0, 1]
    low, high = x[column].quantile([0.2, 0.8])
    bottom = x.loc[x[column] <= low, "tourism_change_pct"].mean()
    top = x.loc[x[column] >= high, "tourism_change_pct"].mean()
    train = x[x.year.isin([2023, 2024])]
    test = x[x.year.eq(2025)]
    model = LinearRegression().fit(train[[column]], train["tourism_residual_log"])
    pred = model.predict(test[[column]])
    return {
        "model": column,
        "n": len(x),
        "spearman": rho,
        "top20_mean_pct": top,
        "bottom20_mean_pct": bottom,
        "top_bottom_diff_pp": top - bottom,
        "test_r2": r2_score(test["tourism_residual_log"], pred),
        "test_mae_log": mean_absolute_error(test["tourism_residual_log"], pred),
    }


cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
df = pd.read_csv(ROOT / cfg["project"]["input_file"], encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["weekday"] = df.date.dt.weekday
month_to_season = {
    month: season
    for season, months in cfg["analysis"]["seasons"].items()
    for month in months
}
df["season"] = df.month.map(month_to_season)

df["log_visitors"] = np.log1p(df["visitor_total"].clip(lower=0))
group = df.groupby(["signguCode", "year", "month", "weekday"])["log_visitors"]
group_sum = group.transform("sum")
group_count = group.transform("count")
df["baseline_log_loo"] = (group_sum - df.log_visitors) / (group_count - 1)
df.loc[group_count < 2, "baseline_log_loo"] = np.nan
df["tourism_residual_log"] = df.log_visitors - df.baseline_log_loo
df["tourism_change_pct"] = np.expm1(df.tourism_residual_log) * 100

scores = pd.DataFrame(index=df.index)
for component, column in cfg["analysis"]["stress_columns"].items():
    scores[component] = (1 - pd.to_numeric(df[column], errors="coerce")).clip(0, 1) * 100

current = pd.read_csv(TABLES / "step4_seasonal_weight_decomposition.csv")
directional = pd.read_csv(TABLES / "alternative_directional_no_availability_weights.csv")
absolute = pd.read_csv(TABLES / "alternative_absolute_no_availability_weights.csv")

for season in cfg["analysis"]["seasons"]:
    idx = df.season.eq(season)
    current_w = current[current.season.eq(season)].set_index("component").data_weight.to_dict()
    directional_w = directional[directional.season.eq(season)].set_index("component")[
        "new_weight_directional_no_availability"
    ].to_dict()
    absolute_w = absolute[absolute.season.eq(season)].set_index("component")[
        "new_weight_absolute_no_availability"
    ].to_dict()
    df.loc[idx, "KTCI_current"] = weighted_available(scores.loc[idx], current_w)
    df.loc[idx, "KTCI_positive_zero_no_availability"] = weighted_available(
        scores.loc[idx], directional_w
    )
    df.loc[idx, "KTCI_absolute_no_availability"] = weighted_available(
        scores.loc[idx], absolute_w
    )

common_columns = [
    "KTCI_current",
    "KTCI_positive_zero_no_availability",
    "KTCI_absolute_no_availability",
]
common = df.dropna(subset=common_columns).copy()
rows = []
for season in ["all", "spring", "summer", "autumn", "winter"]:
    part = common if season == "all" else common[common.season.eq(season)]
    for column in common_columns:
        row = performance(part, column)
        row["season"] = season
        rows.append(row)

pd.DataFrame(rows).to_csv(
    TABLES / "three_model_common_sample_performance.csv",
    index=False,
    encoding="utf-8-sig",
)
print({"common_rows": len(common), "output": "three_model_common_sample_performance.csv"})
