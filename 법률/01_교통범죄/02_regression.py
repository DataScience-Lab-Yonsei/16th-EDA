#!/usr/bin/env python
# coding: utf-8
"""교통범죄 파트를 팀원 분석 형식에 맞게 확장한다.

입력은 01_preprocess_tag.py가 만든 사건별 분석 데이터다.
작은 표본에서 회귀가 실패하거나 완전분리가 발생하는 경우를 숨기지 않고,
모형 진단·기술통계·Fisher 정확검정을 함께 출력한다.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact


DEFAULT_INPUT = Path("New_anal_traffic.csv")
DEFAULT_OUTPUT = Path(".")

LAWYER_LABELS = {
    "public": "국선",
    "lawyer": "개인변호사",
    "lawfirm": "법무법인",
}
SUBGROUP_LABELS = {
    "drunk_driving": "음주운전",
    "hit_run_injury": "도주치상",
}
FACTOR_LABELS = {
    "settlement": "합의·처벌불원",
    "damage_recovery": "피해회복",
    "remorse": "반성",
    "no_prior": "초범·전과 없음",
    "same_prior": "동종전과·누범",
    "high_road_risk": "고위험 도로행위·사고",
}
FAVORABLE_FACTORS = ["settlement", "damage_recovery", "remorse", "no_prior"]


def wilson_interval(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = events / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half, center + half


def result_table(model, model_name: str, outcome: str, n: int) -> pd.DataFrame:
    params = model.params
    ci = model.conf_int()
    result = pd.DataFrame(
        {
            "model": model_name,
            "outcome": outcome,
            "n": n,
            "term": params.index,
            "coefficient": params.values,
            "std_error": model.bse.values,
            "OR": np.exp(np.clip(params.values, -700, 700)),
            "CI_lower": np.exp(np.clip(ci.iloc[:, 0].values, -700, 700)),
            "CI_upper": np.exp(np.clip(ci.iloc[:, 1].values, -700, 700)),
            "p_value": model.pvalues.values,
            "pseudo_r2": getattr(model, "prsquared", np.nan),
            "aic": model.aic,
        }
    )
    result["significant_05"] = result["p_value"] < 0.05
    return result


def safe_logit(
    data: pd.DataFrame,
    *,
    model_name: str,
    outcome: str,
    formula: str,
    required_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    model_data = data[required_columns].dropna().copy()
    events = int(model_data[outcome].sum()) if len(model_data) else 0
    non_events = int(len(model_data) - events)
    parameter_count = (
        1
        + (2 if "lawyer" in required_columns else 0)
        + int("crime_subgroup" in required_columns)
        + sum(
            column in required_columns
            for column in [
                "decision_year",
                "settlement",
                "damage_recovery",
                "remorse",
                "no_prior",
                "same_prior",
                "high_road_risk",
            ]
        )
    )
    minimum_cell = min(events, non_events)
    diagnostic: dict[str, object] = {
        "model": model_name,
        "outcome": outcome,
        "formula": formula,
        "n": len(model_data),
        "events": events,
        "non_events": non_events,
        "parameter_count_approx": parameter_count,
        "events_per_parameter": (
            minimum_cell / parameter_count if parameter_count else np.nan
        ),
        "status": "",
        "converged": "",
        "warning": "",
    }
    if len(model_data) < 12 or model_data[outcome].nunique() < 2:
        diagnostic["status"] = "not_fitted"
        diagnostic["warning"] = "표본 또는 결과의 0/1 변이가 부족함"
        return pd.DataFrame(), diagnostic
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.logit(formula=formula, data=model_data).fit(
                disp=False, maxiter=500
            )
        converged = bool(model.mle_retvals.get("converged"))
        warning_messages = [str(item.message) for item in caught]
        if diagnostic["events_per_parameter"] < 5:
            warning_messages.append("최소 결과셀 기준 사건/모수비 < 5: 심한 과적합 위험")
        diagnostic["status"] = "fitted" if converged else "nonconverged"
        diagnostic["converged"] = converged
        diagnostic["warning"] = " | ".join(dict.fromkeys(warning_messages))
        if not converged:
            return pd.DataFrame(), diagnostic
        return result_table(model, model_name, outcome, len(model_data)), diagnostic
    except Exception as exc:
        diagnostic["status"] = "failed"
        diagnostic["warning"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), diagnostic


def run_extended_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    broad = data[data["analysis_eligible_broad"].eq(1)].copy()
    strict = data[data["analysis_eligible"].eq(1)].copy()
    factor = broad[broad["factor_evidence_reliable"].eq(1)].copy()

    for frame in [broad, strict, factor]:
        frame["lawyer"] = pd.Categorical(
            frame["lawyer"], categories=["public", "lawyer", "lawfirm"]
        )
        frame["crime_subgroup"] = pd.Categorical(
            frame["crime_subgroup"],
            categories=["drunk_driving", "hit_run_injury"],
        )

    specs = [
        {
            "data": broad,
            "model_name": "A1_basic_suspension",
            "outcome": "outcome_suspended",
            "formula": (
                "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
                " + C(crime_subgroup) + decision_year"
            ),
            "required_columns": [
                "outcome_suspended",
                "lawyer",
                "crime_subgroup",
                "decision_year",
            ],
        },
        {
            "data": factor,
            "model_name": "A1_factor_control",
            "outcome": "outcome_suspended",
            "formula": (
                "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
                " + C(crime_subgroup) + settlement + same_prior"
                " + high_road_risk"
            ),
            "required_columns": [
                "outcome_suspended",
                "lawyer",
                "crime_subgroup",
                "settlement",
                "same_prior",
                "high_road_risk",
            ],
        },
        {
            "data": factor,
            "model_name": "A1_enhanced_favorable_factors",
            "outcome": "outcome_suspended",
            "formula": (
                "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
                " + C(crime_subgroup) + settlement + same_prior"
                " + high_road_risk + damage_recovery + remorse + no_prior"
            ),
            "required_columns": [
                "outcome_suspended",
                "lawyer",
                "crime_subgroup",
                "settlement",
                "same_prior",
                "high_road_risk",
                "damage_recovery",
                "remorse",
                "no_prior",
            ],
        },
        {
            "data": factor,
            "model_name": "A2_settlement",
            "outcome": "settlement",
            "formula": (
                "settlement ~ C(lawyer, Treatment(reference='public'))"
                " + C(crime_subgroup) + same_prior + high_road_risk"
            ),
            "required_columns": [
                "settlement",
                "lawyer",
                "crime_subgroup",
                "same_prior",
                "high_road_risk",
            ],
        },
        {
            "data": strict,
            "model_name": "A3_strict_suspension_robustness",
            "outcome": "outcome_suspended",
            "formula": (
                "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
            ),
            "required_columns": ["outcome_suspended", "lawyer"],
        },
        {
            "data": strict,
            "model_name": "A3_reduced_or_suspended",
            "outcome": "outcome_han",
            "formula": (
                "outcome_han ~ C(lawyer, Treatment(reference='public'))"
                " + C(crime_subgroup)"
            ),
            "required_columns": ["outcome_han", "lawyer", "crime_subgroup"],
        },
        {
            "data": factor,
            "model_name": "X1_high_road_risk",
            "outcome": "outcome_suspended",
            "formula": "outcome_suspended ~ high_road_risk",
            "required_columns": ["outcome_suspended", "high_road_risk"],
        },
        {
            "data": broad,
            "model_name": "X2_crime_subgroup",
            "outcome": "outcome_suspended",
            "formula": "outcome_suspended ~ C(crime_subgroup)",
            "required_columns": ["outcome_suspended", "crime_subgroup"],
        },
    ]

    result_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for spec in specs:
        result, diagnostic = safe_logit(**spec)
        diagnostics.append(diagnostic)
        if not result.empty:
            result_frames.append(result)
    results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )
    return results, pd.DataFrame(diagnostics)


def binary_test(
    data: pd.DataFrame,
    *,
    predictor: str,
    outcome: str = "outcome_suspended",
    comparison: str,
    scope: str,
) -> dict[str, object]:
    subset = data[[predictor, outcome]].dropna().copy()
    table = pd.crosstab(subset[predictor], subset[outcome])
    row: dict[str, object] = {
        "scope": scope,
        "comparison": comparison,
        "predictor": predictor,
        "outcome": outcome,
        "n": len(subset),
        "exposed_n": int((subset[predictor] == 1).sum()),
        "exposed_events": int(
            ((subset[predictor] == 1) & (subset[outcome] == 1)).sum()
        ),
        "exposed_rate": subset.loc[
            subset[predictor] == 1, outcome
        ].mean(),
        "reference_n": int((subset[predictor] == 0).sum()),
        "reference_events": int(
            ((subset[predictor] == 0) & (subset[outcome] == 1)).sum()
        ),
        "reference_rate": subset.loc[
            subset[predictor] == 0, outcome
        ].mean(),
        "OR": np.nan,
        "p_value": np.nan,
        "method": "Fisher exact (two-sided)",
    }
    if set(table.index) == {0, 1} and set(table.columns) == {0, 1}:
        matrix = table.loc[[1, 0], [1, 0]].to_numpy()
        row["OR"], row["p_value"] = fisher_exact(matrix)
    return row


def pairwise_lawyer_fisher(
    data: pd.DataFrame, outcome: str, scope: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in ["lawyer", "lawfirm"]:
        subset = data[data["lawyer"].isin(["public", group])].dropna(
            subset=[outcome]
        )
        table = pd.crosstab(subset["lawyer"], subset[outcome])
        if (
            "public" not in table.index
            or group not in table.index
            or 0 not in table.columns
            or 1 not in table.columns
        ):
            continue
        matrix = [
            [table.loc[group, 1], table.loc[group, 0]],
            [table.loc["public", 1], table.loc["public", 0]],
        ]
        odds_ratio, p_value = fisher_exact(matrix)
        rows.append(
            {
                "scope": scope,
                "comparison": (
                    f"{LAWYER_LABELS[group]} vs {LAWYER_LABELS['public']}"
                ),
                "predictor": "lawyer",
                "outcome": outcome,
                "n": len(subset),
                "exposed_n": int(sum(matrix[0])),
                "exposed_events": int(matrix[0][0]),
                "exposed_rate": matrix[0][0] / sum(matrix[0]),
                "reference_n": int(sum(matrix[1])),
                "reference_events": int(matrix[1][0]),
                "reference_rate": matrix[1][0] / sum(matrix[1]),
                "OR": odds_ratio,
                "p_value": p_value,
                "method": "Fisher exact (two-sided)",
            }
        )
    return pd.DataFrame(rows)


def make_favorable_rates(factor: pd.DataFrame) -> pd.DataFrame:
    frame = factor.copy()
    frame["favorable_count"] = frame[FAVORABLE_FACTORS].sum(axis=1).astype(int)
    frame["favorable_group"] = frame["favorable_count"].map(
        lambda value: str(value) if value < 2 else "2개 이상"
    )
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby("favorable_group", observed=True):
        events = int(subset["outcome_suspended"].sum())
        n = len(subset)
        low, high = wilson_interval(events, n)
        rows.append(
            {
                "favorable_group": group,
                "n": n,
                "suspended_n": events,
                "suspended_rate": events / n,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    ordering = {"0": 0, "1": 1, "2개 이상": 2}
    result = pd.DataFrame(rows)
    result["_order"] = result["favorable_group"].map(ordering)
    return result.sort_values("_order").drop(columns="_order")


def make_crime_profile(broad: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subgroup, subset in broad.groupby("crime_subgroup", observed=True):
        reliable = subset[subset["factor_evidence_reliable"].eq(1)]
        row: dict[str, object] = {
            "crime_subgroup": subgroup,
            "crime_label": SUBGROUP_LABELS.get(subgroup, subgroup),
            "n": len(subset),
            "suspended_n": int(subset["outcome_suspended"].sum()),
            "suspended_rate": subset["outcome_suspended"].mean(),
            "factor_reliable_n": len(reliable),
            "bac_known_n": int(subset["bac"].notna().sum()),
            "injury_weeks_known_n": int(
                subset["injury_weeks_max"].notna().sum()
            ),
        }
        for factor, label in FACTOR_LABELS.items():
            known = reliable[factor].dropna()
            row[f"{factor}_n"] = int(known.sum()) if len(known) else 0
            row[f"{factor}_rate"] = known.mean() if len(known) else np.nan
            row[f"{factor}_label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def make_lawyer_subgroup_rates(broad: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subgroup, lawyer), subset in broad.groupby(
        ["crime_subgroup", "lawyer"], observed=True
    ):
        events = int(subset["outcome_suspended"].sum())
        n = len(subset)
        low, high = wilson_interval(events, n)
        rows.append(
            {
                "crime_subgroup": subgroup,
                "crime_label": SUBGROUP_LABELS.get(subgroup, subgroup),
                "lawyer": lawyer,
                "lawyer_label": LAWYER_LABELS.get(lawyer, lawyer),
                "n": n,
                "suspended_n": events,
                "suspended_rate": events / n,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return pd.DataFrame(rows)


def make_special_profiles(broad: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    drunk = broad[broad["crime_subgroup"].eq("drunk_driving")].copy()
    bac_order = ["BAC_0.03_to_0.08", "BAC_0.08_to_0.20", "BAC_0.20_plus"]
    bac_labels = {
        "BAC_0.03_to_0.08": "0.03% 이상~0.08% 미만",
        "BAC_0.08_to_0.20": "0.08% 이상~0.20% 미만",
        "BAC_0.20_plus": "0.20% 이상",
    }
    known_bac = drunk[drunk["bac_type"].isin(bac_order)]
    for category in bac_order:
        subset = known_bac[known_bac["bac_type"].eq(category)]
        rows.append(
            {
                "analysis": "음주운전 BAC 구간",
                "category": category,
                "category_label": bac_labels[category],
                "n": len(subset),
                "suspended_n": int(subset["outcome_suspended"].sum()),
                "suspended_rate": (
                    subset["outcome_suspended"].mean() if len(subset) else np.nan
                ),
                "note": "BAC가 본문에서 추출된 광의 표본만 사용",
            }
        )

    hitrun = broad[broad["crime_subgroup"].eq("hit_run_injury")].copy()
    hitrun["injury_group"] = pd.cut(
        hitrun["injury_weeks_max"],
        bins=[-np.inf, 2, 5, np.inf],
        labels=["2주 이하", "3~5주", "6주 이상"],
    )
    known_injury = hitrun[hitrun["injury_group"].notna()]
    for category in ["2주 이하", "3~5주", "6주 이상"]:
        subset = known_injury[
            known_injury["injury_group"].astype(str).eq(category)
        ]
        rows.append(
            {
                "analysis": "도주치상 상해기간",
                "category": category,
                "category_label": category,
                "n": len(subset),
                "suspended_n": int(subset["outcome_suspended"].sum()),
                "suspended_rate": (
                    subset["outcome_suspended"].mean() if len(subset) else np.nan
                ),
                "note": "전치 주수가 본문에서 추출된 광의 표본만 사용",
            }
        )
    return pd.DataFrame(rows)


def run_exact_tests(
    broad: pd.DataFrame, factor: pd.DataFrame
) -> pd.DataFrame:
    test_rows: list[dict[str, object]] = []
    factor = factor.copy()
    factor["favorable_any"] = (
        factor[FAVORABLE_FACTORS].sum(axis=1) >= 1
    ).astype(int)
    for predictor, label in [
        ("favorable_any", "유리한 양형사유 1개 이상 vs 없음"),
        ("settlement", "합의·처벌불원 기재 vs 없음"),
        ("damage_recovery", "피해회복 기재 vs 없음"),
        ("remorse", "반성 기재 vs 없음"),
        ("no_prior", "초범·전과 없음 기재 vs 없음"),
        ("same_prior", "동종전과·누범 기재 vs 없음"),
        ("high_road_risk", "고위험 도로행위·사고 기재 vs 없음"),
    ]:
        test_rows.append(
            binary_test(
                factor,
                predictor=predictor,
                comparison=label,
                scope="양형이유 신뢰 표본",
            )
        )

    subgroup = broad.copy()
    subgroup["hit_run_indicator"] = subgroup["crime_subgroup"].eq(
        "hit_run_injury"
    ).astype(int)
    test_rows.append(
        binary_test(
            subgroup,
            predictor="hit_run_indicator",
            comparison="도주치상 vs 음주운전",
            scope="광의 표본",
        )
    )

    drunk = broad[
        broad["crime_subgroup"].eq("drunk_driving") & broad["bac"].notna()
    ].copy()
    drunk["bac_020_plus"] = drunk["bac"].ge(0.2).astype(int)
    test_rows.append(
        binary_test(
            drunk,
            predictor="bac_020_plus",
            comparison="BAC 0.20% 이상 vs 0.20% 미만",
            scope="BAC 추출 음주운전",
        )
    )

    frames = [
        pd.DataFrame(test_rows),
        pairwise_lawyer_fisher(
            broad, "outcome_suspended", "광의 표본·집행유예"
        ),
        pairwise_lawyer_fisher(
            factor, "settlement", "양형이유 신뢰 표본·합의"
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def create_figures(
    favorable: pd.DataFrame,
    crime_profile: pd.DataFrame,
    lawyer_rates: pd.DataFrame,
    exact_tests: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = ["#90CAF9", "#4DB6AC", "#FFCC80"]
    bars = ax.bar(
        favorable["favorable_group"],
        favorable["suspended_rate"] * 100,
        color=colors[: len(favorable)],
    )
    for bar, row in zip(bars, favorable.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{int(row.suspended_n)}/{int(row.n)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 100)
    ax.set_xlabel("유리한 양형사유 개수")
    ax.set_ylabel("집행유예 비율(%)")
    ax.set_title("유리한 양형사유 개수와 집행유예 비율")
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_favorable_factor_count.png", dpi=180)
    plt.close(fig)

    factor_columns = [
        "settlement_rate",
        "damage_recovery_rate",
        "remorse_rate",
        "no_prior_rate",
        "same_prior_rate",
        "high_road_risk_rate",
        "suspended_rate",
    ]
    labels = [
        "합의",
        "피해회복",
        "반성",
        "초범",
        "동종전과",
        "고위험행위",
        "집행유예",
    ]
    heat = crime_profile.set_index("crime_label")[factor_columns] * 100
    heat.columns = labels
    fig, ax = plt.subplots(figsize=(10, 3.4))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "출현·결과 비율(%)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("죄유형별 양형사유·결과 프로필")
    fig.tight_layout()
    fig.savefig(output_dir / "fig05_crime_type_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    pivot = lawyer_rates.pivot(
        index="crime_label", columns="lawyer_label", values="suspended_rate"
    )
    desired = [
        label
        for label in ["국선", "개인변호사", "법무법인"]
        if label in pivot.columns
    ]
    pivot = pivot[desired] * 100
    pivot.plot(kind="bar", ax=ax, color=["#90CAF9", "#4DB6AC", "#FFB74D"])
    ax.set_xlabel("")
    ax.set_ylabel("집행유예 비율(%)")
    ax.set_ylim(0, 100)
    ax.set_title("죄유형 × 변호사 유형별 집행유예 비율")
    ax.legend(title="변호사 유형")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "fig06_subgroup_lawyer_rates.png", dpi=180)
    plt.close(fig)

    selected = exact_tests[
        exact_tests["predictor"].isin(
            [
                "favorable_any",
                "settlement",
                "remorse",
                "same_prior",
                "high_road_risk",
            ]
        )
    ].copy()
    selected["exposed_percent"] = selected["exposed_rate"] * 100
    selected["reference_percent"] = selected["reference_rate"] * 100
    plot_frame = selected.melt(
        id_vars=["comparison"],
        value_vars=["reference_percent", "exposed_percent"],
        var_name="group",
        value_name="rate",
    )
    plot_frame["group"] = plot_frame["group"].map(
        {"reference_percent": "미기재", "exposed_percent": "기재"}
    )
    fig, ax = plt.subplots(figsize=(10, 6.2))
    sns.barplot(
        data=plot_frame,
        y="comparison",
        x="rate",
        hue="group",
        palette=["#B0BEC5", "#26A69A"],
        ax=ax,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("집행유예 비율(%)")
    ax.set_ylabel("")
    ax.set_title("교통범죄 양형사유 기재와 집행유예 비율")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "fig07_factor_outcome_rates.png", dpi=180)
    plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_결과 없음_"

    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    data: pd.DataFrame,
    models: pd.DataFrame,
    diagnostics: pd.DataFrame,
    favorable: pd.DataFrame,
    crime_profile: pd.DataFrame,
    lawyer_rates: pd.DataFrame,
    exact_tests: pd.DataFrame,
    special_profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    broad = data[data["analysis_eligible_broad"].eq(1)]
    strict = data[data["analysis_eligible"].eq(1)]
    factor = broad[broad["factor_evidence_reliable"].eq(1)]
    significant = (
        models[
            models["significant_05"] & models["term"].ne("Intercept")
        ].copy()
        if not models.empty
        else pd.DataFrame()
    )
    high_risk = exact_tests[
        exact_tests["predictor"].eq("high_road_risk")
    ].iloc[0]
    subgroup = exact_tests[
        exact_tests["predictor"].eq("hit_run_indicator")
    ].iloc[0]
    lines = [
        "# 교통범죄 확장분석 — 팀원 분석 형식 대응본",
        "",
        "## 분석 표본",
        "",
        f"- 배정 죄명 고유사건: **{len(data)}건**",
        f"- 변호사·형종이 확인된 광의 표본: **{len(broad)}건**",
        f"- 중대 병합죄·복수 피고인 등을 제외한 주 분석 표본: **{len(strict)}건**",
        f"- 양형이유 구간을 신뢰할 수 있는 표본: **{len(factor)}건**",
        "",
        "## 분석 1. 변호사·죄유형과 집행유예",
        "",
        "- 광의 표본의 기본 보정모형은 변호사 유형, 죄유형, 판결연도를 함께 투입했다.",
        "- 양형사유 통제모형과 반성·피해회복 보강모형도 실행했으나, 작은 표본과 희소 셀 때문에 추정 안정성이 낮거나 실패했다.",
        "- 기본 보정모형에서 국선 대비 개인변호사·법무법인의 차이는 5% 수준에서 유의하지 않았다.",
        "",
        "## 분석 2. 변호사 종류와 합의·처벌불원",
        "",
        f"- 신뢰 가능한 양형이유 표본에서 합의 기재는 **{int(factor['settlement'].sum())}/{len(factor)}건**이었다.",
        "- 합의 모형은 사건 수가 적어 안정적인 다변량 추정이 되지 않았다.",
        "- 국선 대비 개인변호사·법무법인의 합의 차이는 Fisher 정확검정에서도 유의하지 않았다.",
        "",
        "## 분석 3. 집행유예만을 결과로 둔 강건성 검증",
        "",
        f"- 주 분석 표본 {len(strict)}건 중 집행유예는 **{int(strict['outcome_suspended'].sum())}건**이었다.",
        "- 국선 대비 개인변호사 OR 1.33(p=0.858), 법무법인 OR 1.60(p=0.736)으로 모두 유의하지 않았다.",
        "- 감경+집행유예 결합결과는 비교 가능한 비집행유예·비감경 사건이 없어 회귀를 적합할 수 없었다.",
        "",
        "## 분석 4. 죄유형별 변호사·집행유예 비율",
        "",
        md_table(
            lawyer_rates,
            [
                "crime_label",
                "lawyer_label",
                "n",
                "suspended_n",
                "suspended_rate",
            ],
        ),
        "",
        "- 도주치상은 특히 죄유형×변호사 셀이 0~3건으로 작아 방향만 기술할 수 있다.",
        "",
        "## 추가분석 1. 유리한 양형사유 개수와 집행유예",
        "",
        md_table(
            favorable,
            ["favorable_group", "n", "suspended_n", "suspended_rate"],
        ),
        "",
        "- 유리한 사유가 0개인 경우보다 1개인 경우 집행유예 비율이 높았지만, 2개 이상에서는 다시 낮아져 단조 증가가 아니었다.",
        "- 유리한 사유 1개 이상과 없음의 Fisher 정확검정 p값은 1.000으로 유의하지 않았다.",
        "",
        "## 추가분석 2. 죄유형별 양형사유·결과 프로필",
        "",
        md_table(
            crime_profile,
            [
                "crime_label",
                "n",
                "suspended_rate",
                "settlement_rate",
                "remorse_rate",
                "same_prior_rate",
                "high_road_risk_rate",
            ],
        ),
        "",
        f"- 도주치상 집행유예율은 {subgroup['exposed_rate']:.1%}, 음주운전은 {subgroup['reference_rate']:.1%}였으나 "
        f"Fisher p={subgroup['p_value']:.3f}으로 유의하지 않았다.",
        "",
        "## 추가분석 3. 교통범죄 특화 탐색",
        "",
        f"- 고위험 도로행위·사고 기재 사건의 집행유예율은 {high_risk['exposed_rate']:.1%}, 미기재 사건은 "
        f"{high_risk['reference_rate']:.1%}였다(OR {high_risk['OR']:.2f}, Fisher p={high_risk['p_value']:.3f}). "
        "방향은 흥미롭지만 역인과·선택편향과 작은 표본 때문에 확증 결과로 쓰면 안 된다.",
        "- 음주운전 광의 표본 26건 중 BAC가 추출된 사건은 6건뿐이었다. 0.08~0.20%는 1/4, 0.20% 이상은 0/2가 집행유예였다.",
        "- 도주치상 7건 중 전치 주수가 추출된 사건은 2건뿐이어서 상해 정도에 따른 차이는 판단할 수 없었다.",
        "",
        md_table(
            special_profiles,
            [
                "analysis",
                "category_label",
                "n",
                "suspended_n",
                "suspended_rate",
                "note",
            ],
        ),
        "",
        "## 모형 실행 상태",
        "",
        md_table(
            diagnostics,
            [
                "model",
                "n",
                "events",
                "non_events",
                "parameter_count_approx",
                "events_per_parameter",
                "status",
                "warning",
            ],
        ),
        "",
        "## 유의성 요약",
        "",
    ]
    if significant.empty:
        lines.append("- 절편을 제외하고 5% 수준에서 유의한 회귀계수는 없었다.")
    else:
        lines.append(
            md_table(
                significant,
                [
                    "model",
                    "term",
                    "OR",
                    "CI_lower",
                    "CI_upper",
                    "p_value",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## 발표용 안전한 결론",
            "",
            "1. 교통범죄에서는 국선 대비 개인변호사·법무법인이 집행유예와 유의하게 관련된다는 증거를 확인하지 못했다.",
            "2. 변호사 유형과 합의·처벌불원 기재 사이에도 유의한 차이를 확인하지 못했다.",
            "3. 유리한 양형사유 개수와 집행유예율은 단조 증가하지 않았으며, 개별 사유도 유의하지 않았다.",
            "4. 도주치상·고위험 도로행위에서 집행유예율이 높게 보이는 방향이 있었지만 작은 셀과 사건 선택 때문에 탐색적 결과에 그친다.",
            "5. 따라서 ‘변호사 효과 없음’으로 단정하기보다 ‘현재 교통범죄 표본으로는 효과를 식별하지 못했다’고 표현해야 한다.",
        ]
    )
    (output_dir / "result.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    broad = data[data["analysis_eligible_broad"].eq(1)].copy()
    strict = data[data["analysis_eligible"].eq(1)].copy()
    factor = broad[broad["factor_evidence_reliable"].eq(1)].copy()

    models, diagnostics = run_extended_models(data)
    favorable = make_favorable_rates(factor)
    crime_profile = make_crime_profile(broad)
    lawyer_rates = make_lawyer_subgroup_rates(broad)
    special_profiles = make_special_profiles(broad)
    exact_tests = run_exact_tests(broad, factor)

    models.to_csv(
        output_dir / "regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        output_dir / "model_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    favorable.to_csv(
        output_dir / "favorable_factor_rates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    crime_profile.to_csv(
        output_dir / "crime_type_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lawyer_rates.to_csv(
        output_dir / "lawyer_subgroup_rates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    special_profiles.to_csv(
        output_dir / "special_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exact_tests.to_csv(
        output_dir / "exact_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_report(
        data,
        models,
        diagnostics,
        favorable,
        crime_profile,
        lawyer_rates,
        exact_tests,
        special_profiles,
        output_dir,
    )
    summary = {
        "broad_n": len(broad),
        "strict_n": len(strict),
        "factor_reliable_n": len(factor),
        "broad_suspended_n": int(broad["outcome_suspended"].sum()),
        "strict_suspended_n": int(strict["outcome_suspended"].sum()),
        "fitted_models_n": int(diagnostics["status"].eq("fitted").sum()),
        "significant_non_intercept_n": int(
            (
                models["significant_05"] & models["term"].ne("Intercept")
            ).sum()
            if not models.empty
            else 0
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console_lines = [
        "[교통범죄 확장분석 실행 결과]",
        "",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        "[모형 진단]",
        diagnostics[
            [
                "model",
                "n",
                "events",
                "non_events",
                "events_per_parameter",
                "status",
                "warning",
            ]
        ].to_string(index=False),
        "",
        "[정상 수렴 모형의 비절편 결과]",
        models.loc[
            models["term"].ne("Intercept"),
            ["model", "term", "OR", "CI_lower", "CI_upper", "p_value"],
        ].to_string(index=False),
        "",
        "[Fisher 정확검정]",
        exact_tests[
            [
                "scope",
                "comparison",
                "n",
                "exposed_rate",
                "reference_rate",
                "OR",
                "p_value",
            ]
        ].to_string(index=False),
    ]
    console_text = "\n".join(console_lines)
    (output_dir / "SCREENSHOT_CONSOLE.txt").write_text(
        console_text, encoding="utf-8"
    )
    print(console_text)


if __name__ == "__main__":
    main()
