from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

SEASON_KO = {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
COMP_KO = {
    "thermal": "온열",
    "humidity": "습도",
    "wind": "바람",
    "precipitation": "강수",
    "air_quality": "대기질",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def weighted_available(scores: pd.DataFrame, weights: dict[str, float], min_n: int = 2) -> pd.Series:
    cols = [c for c in weights if c in scores.columns]
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    valid = scores[cols].notna()
    numerator = scores[cols].mul(w, axis=1).sum(axis=1, min_count=1)
    denominator = valid.mul(w, axis=1).sum(axis=1)
    out = numerator.div(denominator).where(valid.sum(axis=1) >= min_n)
    return out


def top_bottom(y: pd.Series, index: pd.Series, q: float) -> tuple[float, float, float, int]:
    d = pd.DataFrame({"y": y, "index": index}).dropna()
    if len(d) < 30 or d["index"].nunique() < 5:
        return np.nan, np.nan, np.nan, len(d)
    lo, hi = d["index"].quantile([q, 1 - q])
    bottom = d.loc[d["index"] <= lo, "y"].mean()
    top = d.loc[d["index"] >= hi, "y"].mean()
    return top, bottom, top - bottom, len(d)


def performance(df: pd.DataFrame, index_col: str, q: float, train_years: list[int], test_years: list[int]) -> dict:
    d = df[["year", "tourism_residual_log", "tourism_change_pct", index_col]].dropna()
    rho = d[[index_col, "tourism_residual_log"]].corr(method="spearman").iloc[0, 1] if len(d) else np.nan
    top, bottom, diff, n = top_bottom(d["tourism_change_pct"], d[index_col], q)
    tr = d[d.year.isin(train_years)]
    te = d[d.year.isin(test_years)]
    if len(tr) >= 30 and len(te) >= 30:
        model = LinearRegression().fit(tr[[index_col]], tr["tourism_residual_log"])
        pred = model.predict(te[[index_col]])
        r2 = r2_score(te["tourism_residual_log"], pred)
        mae = mean_absolute_error(te["tourism_residual_log"], pred)
    else:
        r2 = mae = np.nan
    return {
        "index": index_col, "n": n, "spearman": rho,
        "top20_mean_pct": top, "bottom20_mean_pct": bottom,
        "top_bottom_diff_pp": diff, "test_r2": r2, "test_mae_log": mae,
    }


def region_group(code: int) -> str:
    p = int(code) // 1000
    first2 = int(code) // 1000
    if first2 in (11, 28, 41): return "수도권"
    if first2 in (42, 51): return "강원권"
    if first2 in (30, 36, 43, 44): return "충청권"
    if first2 in (29, 45, 46, 52): return "호남권"
    if first2 in (26, 27, 31, 47, 48): return "영남권"
    if first2 in (50,): return "제주권"
    return "기타"


def plot_save(fig, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix("." + fmt), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent
    cfg = load_config(config_path)
    out = root / cfg["project"]["output_dir"]
    tables, figs, logs, diagnostics = out / "tables", out / "figures", out / "logs", out / "diagnostics"
    for p in (tables, figs, logs, diagnostics):
        p.mkdir(parents=True, exist_ok=True)

    style = cfg["style"]
    plt.rcParams["font.family"] = style["font_family"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=style["font_family"])

    input_path = root / cfg["project"]["input_file"]
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")
    raw_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = ["signguCode", "signguNm", "date", "visitor_total", "S_temp", "S_humidity", "S_wind", "S_precip", "S_airquality"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 열 누락: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    month_to_season = {m: s for s, months in cfg["analysis"]["seasons"].items() for m in months}
    df["season"] = df["month"].map(month_to_season)
    df["region_group"] = df["signguCode"].map(region_group)
    stress_map = cfg["analysis"]["stress_columns"]
    stress_cols = list(stress_map.values())
    for c in stress_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    quality = []
    for c in df.columns:
        x = df[c]
        quality.append({
            "variable": c, "dtype": str(x.dtype), "n": len(x), "missing_n": int(x.isna().sum()),
            "missing_pct": float(x.isna().mean() * 100), "unique_n": int(x.nunique(dropna=True)),
            "min": float(x.min()) if pd.api.types.is_numeric_dtype(x) and x.notna().any() else None,
            "max": float(x.max()) if pd.api.types.is_numeric_dtype(x) and x.notna().any() else None,
        })
    quality_df = pd.DataFrame(quality)
    save_csv(quality_df, tables / "step1_data_quality.csv")

    stress_range = pd.DataFrame([{
        "variable": c, "below_0_n": int((df[c] < 0).sum()), "above_1_n": int((df[c] > 1).sum()),
        "min": df[c].min(), "max": df[c].max()
    } for c in stress_cols])
    save_csv(stress_range, diagnostics / "stress_range_check.csv")

    haman = df[df["signguCode"] == 48730]
    haman_diag = pd.DataFrame([{
        "issue": "함안군 기상 결측", "signguCode": 48730, "rows": len(haman),
        **{f"{c}_missing_pct": float(haman[c].isna().mean() * 100) if len(haman) else np.nan for c in stress_cols},
        "action": "원자료를 임의 보간하지 않고 weighted_available로 관측 가능한 대기질만 사용. 최소 2개 영역 조건 때문에 KTCI는 결측 처리."
    }])
    save_csv(haman_diag, diagnostics / "haman_missing_diagnostic.csv")

    hierarchy = df[["signguCode", "signguNm"]].drop_duplicates().copy()
    hierarchy["is_city_level_name"] = hierarchy["signguNm"].astype(str).str.endswith("시")
    hierarchy["possible_overlap_group"] = hierarchy["signguCode"].astype(str).str[:3]
    counts = hierarchy.groupby("possible_overlap_group")["signguCode"].transform("count")
    hierarchy["possible_parent_child_overlap"] = hierarchy["is_city_level_name"] & (counts > 1)
    hierarchy["aggregation_policy"] = "관광객 합산 분석에서 제외; 행 단위 상관·잔차 분석만 사용"
    save_csv(hierarchy, diagnostics / "administrative_hierarchy_diagnostic.csv")

    df["log_visitors"] = np.log1p(df["visitor_total"].clip(lower=0))
    gcols = cfg["analysis"]["baseline_group"]
    grp = df.groupby(gcols, dropna=False)["log_visitors"]
    gsum, gcount = grp.transform("sum"), grp.transform("count")
    df["baseline_log_loo"] = (gsum - df["log_visitors"]) / (gcount - 1)
    df.loc[gcount < 2, "baseline_log_loo"] = np.nan
    df["tourism_residual_log"] = df["log_visitors"] - df["baseline_log_loo"]
    df["tourism_change_pct"] = np.expm1(df["tourism_residual_log"]) * 100

    corr = df[stress_cols].corr(method="spearman")
    corr.to_csv(tables / "step2_stress_correlation.csv", encoding="utf-8-sig")

    target_rows = []
    for comp, c in stress_map.items():
        sub = df[[c, "tourism_residual_log"]].dropna()
        rho = sub.corr(method="spearman").iloc[0, 1]
        low = sub[c].quantile(.2); high = sub[c].quantile(.8)
        lo_y = sub.loc[sub[c] <= low, "tourism_residual_log"].mean()
        hi_y = sub.loc[sub[c] >= high, "tourism_residual_log"].mean()
        target_rows.append({
            "component": comp, "variable": c, "n": len(sub), "spearman_stress_vs_tourism": rho,
            "low_stress_mean_log": lo_y, "high_stress_mean_log": hi_y,
            "high_minus_low_log": hi_y - lo_y, "availability": len(sub) / len(df)
        })
    target_corr = pd.DataFrame(target_rows)
    save_csv(target_corr, tables / "step3_stress_tourism_relationship.csv")

    weight_rows = []
    curve_rows = []
    for season, sdf in df.groupby("season"):
        ev = []
        for comp, c in stress_map.items():
            sub = sdf[[c, "tourism_residual_log"]].dropna()
            rho = sub.corr(method="spearman").iloc[0, 1] if len(sub) else np.nan
            q = sub.assign(bin=pd.qcut(sub[c], q=10, duplicates="drop")).groupby("bin", observed=True).agg(
                stress_mean=(c, "mean"), tourism_mean=("tourism_residual_log", "mean"), n=(c, "size")
            ).reset_index(drop=True) if len(sub) else pd.DataFrame()
            amplitude = float(q["tourism_mean"].max() - q["tourism_mean"].min()) if len(q) else np.nan
            availability = len(sub) / len(sdf) if len(sdf) else np.nan
            evidence = math.sqrt(max(abs(rho), 1e-8) * max(amplitude, 1e-8) * max(availability, 1e-8))
            direction_ok = bool(rho <= 0) if pd.notna(rho) else False
            ev.append((comp, evidence))
            weight_rows.append({
                "season": season, "component": comp, "variable": c, "n": len(sub), "spearman": rho,
                "decile_amplitude_log": amplitude, "availability": availability,
                "direction_expected_negative": direction_ok, "evidence": evidence,
            })
            if len(q):
                q["season"], q["component"] = season, comp
                curve_rows.append(q)
        total = sum(v for _, v in ev)
        for comp, evidence in ev:
            for row in reversed(weight_rows):
                if row["season"] == season and row["component"] == comp:
                    row["data_weight"] = evidence / total if total else np.nan
                    break
    weights = pd.DataFrame(weight_rows)
    save_csv(weights, tables / "step4_seasonal_weight_decomposition.csv")
    if curve_rows:
        save_csv(pd.concat(curve_rows, ignore_index=True), tables / "step6_decile_response_curves.csv")

    scores = pd.DataFrame(index=df.index)
    for comp, c in stress_map.items():
        scores[comp] = (1 - df[c]).clip(0, 1) * 100
        df[f"score_{comp}"] = scores[comp]

    for season in cfg["analysis"]["seasons"]:
        idx = df["season"] == season
        wdata = weights[weights.season.eq(season)].set_index("component")["data_weight"].to_dict()
        df.loc[idx, "KTCI_data"] = weighted_available(scores.loc[idx], wdata, cfg["missing"]["minimum_available_components"])
        raw2014 = cfg["benchmark_2014"][season].copy()
        raw2014.pop("cloud_unavailable", None)
        common = {"thermal": raw2014["thermal"], "precipitation": raw2014["precipitation"], "wind": raw2014["wind"]}
        s = sum(common.values()); w2014 = {k: v / s for k, v in common.items()}
        df.loc[idx, "KTCI_2014_adapted"] = weighted_available(scores.loc[idx], w2014, 2)

    alpha_rows = []
    chosen = {}
    for season in cfg["analysis"]["seasons"]:
        d = df[(df.season == season) & df.year.isin(cfg["analysis"]["train_years"])].copy()
        best_alpha, best_rho = 0.0, -np.inf
        for alpha in cfg["analysis"]["hybrid_alpha_grid"]:
            hybrid = alpha * d["KTCI_2014_adapted"] + (1 - alpha) * d["KTCI_data"]
            rho = pd.DataFrame({"h": hybrid, "y": d["tourism_residual_log"]}).corr(method="spearman").iloc[0, 1]
            alpha_rows.append({"season": season, "alpha_survey": alpha, "train_spearman": rho})
            if pd.notna(rho) and rho > best_rho:
                best_alpha, best_rho = alpha, rho
        chosen[season] = best_alpha
        idx = df.season.eq(season)
        df.loc[idx, "KTCI_hybrid"] = best_alpha * df.loc[idx, "KTCI_2014_adapted"] + (1 - best_alpha) * df.loc[idx, "KTCI_data"]
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df["selected"] = alpha_df.apply(lambda r: r.alpha_survey == chosen[r.season], axis=1)
    save_csv(alpha_df, tables / "step7_hybrid_alpha_search.csv")

    perf_rows = []
    for season in ["all"] + list(cfg["analysis"]["seasons"]):
        sdf = df if season == "all" else df[df.season.eq(season)]
        for region in ["전국"] + sorted(x for x in df.region_group.dropna().unique() if x != "기타"):
            rdf = sdf if region == "전국" else sdf[sdf.region_group.eq(region)]
            for model in ["KTCI_data", "KTCI_2014_adapted", "KTCI_hybrid"]:
                row = performance(rdf, model, cfg["analysis"]["top_bottom_quantile"], cfg["analysis"]["train_years"], cfg["analysis"]["test_years"])
                row.update({"season": season, "region_group": region})
                perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    save_csv(perf, tables / "step7_index_performance.csv")

    year_perf = []
    for year, ydf in df.groupby("year"):
        for model in ["KTCI_data", "KTCI_2014_adapted", "KTCI_hybrid"]:
            row = performance(ydf, model, cfg["analysis"]["top_bottom_quantile"], [year], [year])
            row["year"] = year
            year_perf.append(row)
    save_csv(pd.DataFrame(year_perf), tables / "step7_year_stability.csv")

    scored_cols = ["signguCode", "signguNm", "date", "year", "season", "region_group", "visitor_total",
                   "tourism_residual_log", "tourism_change_pct"] + stress_cols + \
                  [f"score_{c}" for c in stress_map] + ["KTCI_data", "KTCI_2014_adapted", "KTCI_hybrid"]
    save_csv(df[scored_cols].head(50000), tables / "scored_observations_sample.csv")

    colors = style["colors"]; formats = style["formats"]; dpi = style["dpi"]
    qplot = quality_df[quality_df.variable.isin(stress_cols)].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=qplot, x="missing_pct", y="variable", color=colors["data_driven"], ax=ax)
    ax.set(title="스트레스 스코어 결측률", xlabel="결측률 (%)", ylabel="")
    plot_save(fig, figs / "step1_missing_rates", formats, dpi)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("스트레스 스코어 간 Spearman 상관 관계")
    plot_save(fig, figs / "step2_stress_correlation", formats, dpi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = target_corr.sort_values("spearman_stress_vs_tourism")
    sns.barplot(data=t, x="spearman_stress_vs_tourism", y="component", color=colors["data_driven"], ax=ax)
    ax.axvline(0, color="#333333", lw=1)
    ax.set(
        title="스트레스 스코어와 혼잡도 잔차의 상관 관계",
        xlabel="Spearman 상관 관계 (음수일수록 스트레스 스코어 증가 시 관광 감소)",
        ylabel="",
    )
    plot_save(fig, figs / "step3_stress_tourism", formats, dpi)

    wp = weights.pivot(index="component", columns="season", values="data_weight")
    wp = wp[[s for s in cfg["analysis"]["seasons"] if s in wp.columns]]
    fig, ax = plt.subplots(figsize=(10, 5))
    wp.plot(kind="bar", ax=ax, color=[colors[s] for s in wp.columns])
    ax.set(title="계절별 데이터 기반 가중치", xlabel="", ylabel="가중치")
    ax.legend([SEASON_KO.get(s, s) for s in wp.columns], title="계절")
    plt.xticks(rotation=0)
    plot_save(fig, figs / "step4_seasonal_weights", formats, dpi)

    regional = perf[(perf.season == "all")].pivot(index="region_group", columns="index", values="spearman")
    fig, ax = plt.subplots(figsize=(10, 5))
    regional.plot(kind="bar", ax=ax, color=[colors["benchmark_2014"], colors["data_driven"], colors["hybrid"]])
    ax.set(title="권역별 지수 Spearman 상관 관계 비교", xlabel="", ylabel="Spearman 상관 관계")
    plt.xticks(rotation=0)
    plot_save(fig, figs / "step5_regional_performance", formats, dpi)

    nat = perf[(perf.region_group == "전국") & (perf.season != "all")].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=nat, x="season", y="top_bottom_diff_pp", hue="index",
                palette=[colors["data_driven"], colors["benchmark_2014"], colors["hybrid"]], ax=ax)
    ax.set(title="계절별 상·하위 20% 관광 변화 차이", xlabel="", ylabel="차이 (%p)")
    plot_save(fig, figs / "step7_seasonal_top_bottom", formats, dpi)

    fig, ax = plt.subplots(figsize=(9, 5))
    for season, adf in alpha_df.groupby("season"):
        ax.plot(adf.alpha_survey, adf.train_spearman, marker="o", ms=3, label=SEASON_KO[season], color=colors[season])
    ax.set(
        title="Hybrid α 탐색 (2023–2024 학습)",
        xlabel="설문 benchmark 비중 α",
        ylabel="학습 Spearman 상관 관계",
    )
    ax.legend()
    plot_save(fig, figs / "step7_hybrid_alpha_search", formats, dpi)

    selected_alpha = ", ".join(f"{SEASON_KO[s]} {a:.2f}" for s, a in chosen.items())
    national = perf[(perf.region_group == "전국") & (perf.season == "all")][
        ["index", "spearman", "top_bottom_diff_pp", "test_r2", "test_mae_log"]
    ]
    report = f"""# 3개년 계절별 KTCI EDA 최종 보고서


- 분석기간: {df.date.min().date()} ~ {df.date.max().date()}
- 관측치: {len(df):,}행, 시군구 코드 {df.signguCode.nunique():,}개
- 스트레스 정의: 0=좋음, 1=나쁨. 최종 적합도 점수는 `100 × (1-스트레스)`입니다.
- Hybrid의 설문 benchmark 비중 α(2023–2024 학습에서 선택): {selected_alpha}


1. **습도 33.94% 결측**: 임의 보간하지 않았습니다. 지수 계산 시 관측 가능한 구성요소의 가중치 합으로 다시 나누는 `weighted_available` 방식을 사용했습니다.
2. **함안군**: 온열·습도·바람·강수가 전부 결측이고 대기질만 존재했습니다. 최소 2개 영역이 필요하므로 함안군 KTCI는 결측으로 남겼습니다. 근거 없는 최근접 관측소 대체는 하지 않았습니다.
3. **시/구 계층 중복**: 관광객을 권역별로 합산하지 않았습니다. 각 행의 '평소 대비 관광 변화'를 계산한 뒤 상관과 구분력을 평가하여 이중합산을 피했습니다.
4. **2014 benchmark**: 원 논문의 계절별 가중치는 정확히 기록했지만, 현재 파일에 최고·평균기온과 운량 원변수가 없습니다. 따라서 온도 두 항목은 `thermal`로 합치고, 운량은 제외한 뒤 사용 가능한 공통영역에서 재정규화했습니다. 이는 **2014 계절별 가중치 adapted benchmark**이며 원 KTCI 완전 재현이 아닙니다.


결측률, 범위, 중복, 날짜, 행정계층을 점검했습니다. 스트레스 값은 모두 0~1 범위였습니다. 습도 결측은 구조적(AWS 비관측) 결측이므로 0으로 채우지 않았습니다.

Spearman 상관으로 동일한 나쁜 날씨가 여러 영역에 중복 반영되는 정도를 확인했습니다. 이 표는 인과관계가 아니라 정보 중복 진단입니다.

시군구·연도·월·요일이 같은 날짜끼리 '자기 자신을 제외한 평소 관광량'을 만들었습니다. 현재 날짜를 기준 평균에 넣지 않아 차이가 인위적으로 작아지는 것을 막았습니다.

각 계절에서 스트레스와 관광잔차의 순위상관, 스트레스 10분위 관광반응 폭, 데이터 가용률을 결합해 계절별 증거점수와 가중치를 만들었습니다.

관광객 수를 지역 간 합산하지 않고 권역별 행만 분리해 동일한 성능지표를 계산했습니다. 따라서 시/구 계층 중복이 합계에 이중 반영되지 않습니다.

원 기상값이 아니라 이미 piecewise-linear로 변환된 스트레스가 입력이므로, 새로운 Spline 점수함수를 덧씌우지 않았습니다. 대신 스트레스 10분위별 관광반응을 사용해 비선형성과 임계구간을 검증했습니다.

- Data-driven: 계절별 데이터 증거 가중치
- 2014 adapted benchmark: 동일한 0~100 점수에 2014 설문 계절가중치의 공통영역 재정규화
- Hybrid: 두 지수를 α로 결합하며 α는 2023–2024 학습기간에서만 선택, 2025년에 검증


{national.to_markdown(index=False)}

R²=1이면 관광잔차를 완벽히 예측하고, 0이면 평균 예측과 비슷하며, 음수이면 평균보다 못합니다. KTCI는 관광객 수 전체 예측모델이 아니므로 R²가 낮을 수 있습니다. 이 연구에서는 좋은 날과 나쁜 날의 순서를 보는 Spearman과 상·하위 20% 차이를 함께 봅니다.


- 현재 입력은 영역 스트레스만 포함하므로 원 기상값별 임계점 재추정은 불가능합니다.
- 2014 KTCI의 운량 점수를 직접 계산할 수 없어 adapted benchmark를 사용했습니다.
- 공휴일·축제·가격·교통 등 비기상 요인은 포함되지 않았습니다.
- Hybrid α는 더 긴 외부기간에서 재검증할 필요가 있습니다.
"""
    (root / "report").mkdir(exist_ok=True)
    (root / "report" / "KTCI_3Year_Final_Report.md").write_text(report, encoding="utf-8")

    metadata = {
        "input_file": str(input_path), "input_sha256": raw_hash, "rows": len(df),
        "columns": list(df.columns), "date_min": str(df.date.min()), "date_max": str(df.date.max()),
        "selected_hybrid_alpha": chosen, "python": sys.version,
    }
    (logs / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (logs / "run_complete.txt").write_text("SUCCESS\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "rows": len(df), "selected_alpha": chosen}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
